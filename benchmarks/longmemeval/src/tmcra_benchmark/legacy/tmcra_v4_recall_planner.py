"""Strict TMCRA V4 recall role planning.

The planner describes ranking and composition roles. It cannot disable a local
retrieval layer; execution remains a controller responsibility.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from tmcra_v3_recall_planner import (
    DEEPSEEK_FLASH_MODEL,
    RecallPlannerError,
    RecallPlannerResponseError,
    _reject_gold,
    _sha256,
)

ROLE_PLAN_SCHEMA = "tmcra.recall-role-plan.v1"
PLANNER_VERSION = ROLE_PLAN_SCHEMA
PLANNER_PROMPT_VERSION = "tmcra-recall-role-planner-2026-07-12.3"
PLAN_FIELDS = frozenset({"schema_version", "resolved_query", "query_kind", "temporal_focus", "conflict_policy", "layers"})
LAYER_NAMES = ("source", "fast", "slow")
LAYER_FIELDS = frozenset({"role", "weight"})
ROLE_VALUES = frozenset({"primary", "support", "context", "conflict", "evidence", "atomic", "bridge"})
ROLE_PRIORS = {"primary": 1.0, "evidence": 1.0, "atomic": 1.0, "conflict": 0.9, "support": 0.75, "bridge": 0.65, "context": 0.5}
QUERY_KINDS = frozenset({"fact", "event", "state", "preference", "goal", "constraint", "relationship", "task", "decision", "comparison", "historical", "unknown"})
QUERY_KIND_EXTENSION_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
TEMPORAL_FOCUSES = frozenset({"current", "recent", "future", "historical", "timeless", "mixed", "unknown"})
CONFLICT_POLICIES = frozenset({"prefer_recent", "prefer_durable", "preserve_parallel", "compare", "surface_uncertainty"})

SYSTEM_PROMPT = f"""You are the TMCRA V4 recall control plane.
Return exactly one JSON object using schema {ROLE_PLAN_SCHEMA} and no other keys:
{{"schema_version":"{ROLE_PLAN_SCHEMA}","resolved_query":"standalone query",
"query_kind":"fact|event|state|preference|goal|constraint|relationship|task|decision|comparison|historical|unknown",
"temporal_focus":"current|recent|future|historical|timeless|mixed|unknown",
"conflict_policy":"prefer_recent|prefer_durable|preserve_parallel|compare|surface_uncertainty",
"layers":{{"source":{{"role":"primary|support|context|conflict|evidence|atomic|bridge","weight":0.0}},
"fast":{{"role":"primary|support|context|conflict|evidence|atomic|bridge","weight":0.0}},
"slow":{{"role":"primary|support|context|conflict|evidence|atomic|bridge","weight":0.0}}}}}}

Weights are bounded in [0, 1]. A zero weight is valid but is never a retrieval
gate: source, fast, and slow local candidate paths still run whenever their
inventories exist. Never emit disabled or excluded layers, scores, candidate
IDs, benchmark fields, answer text, or evidence text. Invalid output is a hard
failure; do not retry or return a fallback plan."""


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _weight(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecallPlannerError(f"{path} must be a finite number in [0, 1]")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise RecallPlannerError(f"{path} must be a finite number in [0, 1]")
    return result


def validate_recall_role_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != PLAN_FIELDS:
        raise RecallPlannerError("RecallRolePlan root must contain exactly the required schema fields")
    if value.get("schema_version") != ROLE_PLAN_SCHEMA:
        raise RecallPlannerError(f"schema_version must be {ROLE_PLAN_SCHEMA!r}")
    resolved_query = _text(value.get("resolved_query"))
    if not resolved_query:
        raise RecallPlannerError("resolved_query is required")
    if len(resolved_query) > 2000:
        raise RecallPlannerError("resolved_query exceeds 2000 characters")
    query_kind = _text(value.get("query_kind"))
    if query_kind not in QUERY_KINDS and QUERY_KIND_EXTENSION_RE.fullmatch(query_kind) is None:
        raise RecallPlannerError(
            "query_kind must be a documented value or a bounded snake_case extension"
        )
    temporal_focus = _text(value.get("temporal_focus"))
    if temporal_focus not in TEMPORAL_FOCUSES:
        raise RecallPlannerError(f"unsupported temporal_focus: {temporal_focus!r}")
    conflict_policy = _text(value.get("conflict_policy"))
    if conflict_policy not in CONFLICT_POLICIES:
        raise RecallPlannerError(f"unsupported conflict_policy: {conflict_policy!r}")
    layers = value.get("layers")
    if not isinstance(layers, Mapping) or set(layers) != set(LAYER_NAMES):
        raise RecallPlannerError("layers must contain exactly source, fast, and slow")
    normalized_layers: dict[str, dict[str, Any]] = {}
    for layer in LAYER_NAMES:
        entry = layers[layer]
        if not isinstance(entry, Mapping) or set(entry) != LAYER_FIELDS:
            raise RecallPlannerError(f"layers.{layer} must contain exactly role and weight")
        role = _text(entry.get("role"))
        if role not in ROLE_VALUES:
            raise RecallPlannerError(f"layers.{layer}.role is invalid or disables a layer")
        normalized_layers[layer] = {"role": role, "weight": _weight(entry.get("weight"), f"layers.{layer}.weight")}
    if not any(entry["weight"] > 0.0 for entry in normalized_layers.values()):
        raise RecallPlannerError("recall role plan cannot assign zero weight to every layer")
    return {"schema_version": ROLE_PLAN_SCHEMA, "resolved_query": resolved_query, "query_kind": query_kind, "temporal_focus": temporal_focus, "conflict_policy": conflict_policy, "layers": normalized_layers}


def _validate_recent_dialogue(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) > 8:
        raise RecallPlannerError("recent_dialogue must contain at most 8 turn objects")
    output: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"turn_index", "speaker", "text"}:
            raise RecallPlannerError(f"recent_dialogue[{index}] has an invalid schema")
        try:
            turn_index = int(item["turn_index"])
        except (TypeError, ValueError) as exc:
            raise RecallPlannerError(f"recent_dialogue[{index}].turn_index must be an integer") from exc
        speaker, text = _text(item.get("speaker")).lower(), _text(item.get("text"))
        if speaker not in {"user", "assistant"} or not text or len(text) > 4000:
            raise RecallPlannerError(f"recent_dialogue[{index}] has an invalid speaker or text")
        output.append({"turn_index": turn_index, "speaker": speaker, "text": text})
    if any(left["turn_index"] >= right["turn_index"] for left, right in zip(output, output[1:])):
        raise RecallPlannerError("recent_dialogue must be in strictly increasing turn order")
    _reject_gold(output, "recent_dialogue")
    return output


def _normalize_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise RecallPlannerError("recall role planner success response lacks usage")

    def integer(name: str, *aliases: str, required: bool = False) -> tuple[int, bool]:
        for key in (name, *aliases):
            if value.get(key) is None:
                continue
            raw = value.get(key)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or int(raw) < 0:
                raise RecallPlannerError(f"usage.{key} is invalid")
            return int(raw), True
        if required:
            raise RecallPlannerError(f"usage.{name} is missing")
        return 0, False

    prompt, _ = integer("prompt_tokens", "input_tokens", required=True)
    completion, _ = integer("completion_tokens", "output_tokens", required=True)
    hit, has_hit = integer(
        "prompt_cache_hit_tokens", "cache_read_input_tokens", "cached_tokens"
    )
    miss, has_miss = integer("prompt_cache_miss_tokens", "cache_miss_input_tokens")
    if hit > prompt or miss > prompt or (has_hit and has_miss and hit + miss != prompt):
        raise RecallPlannerError("planner cache usage does not balance prompt tokens")
    if not has_hit:
        hit = prompt - miss
    if not has_miss:
        miss = prompt - hit
    total, has_total = integer("total_tokens")
    if not has_total:
        total = prompt + completion
    elif total < prompt + completion:
        raise RecallPlannerError("usage.total_tokens is smaller than prompt plus completion")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "prompt_cache_hit_tokens": hit,
        "prompt_cache_miss_tokens": miss,
        "total_tokens": total,
    }


def _metadata_v4(
    *,
    physical_call_id: str,
    key_index: int,
    started: float,
    finish_reason: str,
    content: str = "",
    usage: Mapping[str, Any] | None = None,
    http_status: int | None = None,
    error_type: str | None = None,
    request_sha256: str = "",
    response_id: str = "",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "physical_call_id": physical_call_id,
        "physical_api_call": True,
        "physical_api_calls": 1,
        "stage": "recall_planner",
        "provider": "deepseek",
        "model": DEEPSEEK_FLASH_MODEL,
        "api_key_index": key_index,
        "latency_seconds": round(time.time() - started, 3),
        "response_sha256": _sha256(content),
        "finish_reason": finish_reason,
        "status": "completed" if finish_reason == "stop" else finish_reason,
        "planner_version": PLANNER_VERSION,
        "prompt_version": PLANNER_PROMPT_VERSION,
        "request_sha256": request_sha256,
        "response_id": response_id,
    }
    if usage is not None:
        normalized_usage = dict(usage)
        metadata.update(normalized_usage)
        metadata["usage"] = normalized_usage
    if http_status is not None:
        metadata["http_status"] = int(http_status)
    if error_type:
        metadata["error_type"] = error_type
    return metadata


class DeepSeekFlashRecallRolePlanner:
    """One physical Flash request; invalid output is never retried or repaired."""

    def __init__(self, *, base_url: str, model: str, api_keys: Sequence[str], timeout: float = 30, max_tokens: int = 512) -> None:
        self.base_url = _text(base_url).rstrip("/")
        self.model = model
        self.api_keys = list(dict.fromkeys(_text(key) for key in api_keys if _text(key)))
        self.timeout, self.max_tokens, self.request_index = max(1.0, float(timeout)), max(128, int(max_tokens)), 0
        if not self.base_url or not self.api_keys:
            raise RecallPlannerError("planner base_url and API key pool are required")
        if self.model != DEEPSEEK_FLASH_MODEL:
            raise RecallPlannerError(f"recall role planner requires {DEEPSEEK_FLASH_MODEL}, got {self.model!r}")

    def plan(self, *, query: str, question_date: str, available_layers: Mapping[str, Any], recent_dialogue: Sequence[Mapping[str, Any]] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        query, question_date = _text(query), _text(question_date)
        if not query or not question_date:
            raise RecallPlannerError("query and question_date are required")
        if not isinstance(available_layers, Mapping) or set(available_layers) != set(LAYER_NAMES):
            raise RecallPlannerError("available_layers must contain exactly source, fast, and slow summaries")
        _reject_gold(available_layers, "available_layers")
        dialogue = _validate_recent_dialogue(recent_dialogue)
        payload = {"query": query, "question_date": question_date, "recent_dialogue": dialogue, "available_layers": dict(available_layers)}
        key_index = self.request_index % len(self.api_keys)
        self.request_index += 1
        body = {"model": DEEPSEEK_FLASH_MODEL, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}], "temperature": 0, "max_tokens": self.max_tokens, "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"}, "enable_thinking": False}
        request_sha256 = _sha256(json.dumps(body, ensure_ascii=False, sort_keys=True))
        physical_call_id = "dsc_" + uuid.uuid4().hex
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_keys[key_index]}"}, method="POST")
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                http_status = int(response.getcode())
                raw_http = response.read().decode("utf-8")
                response_payload = json.loads(raw_http)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RecallPlannerResponseError(f"recall role planner HTTP {exc.code}: {detail}", response_content=detail, request_metadata=_metadata_v4(physical_call_id=physical_call_id, key_index=key_index, started=started, finish_reason="http_error", content=detail, http_status=exc.code, request_sha256=request_sha256)) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raw = locals().get("raw_http", "")
            raise RecallPlannerResponseError("recall role planner returned invalid HTTP JSON", response_content=raw, request_metadata=_metadata_v4(physical_call_id=physical_call_id, key_index=key_index, started=started, finish_reason="invalid_http_json", content=raw, error_type=exc.__class__.__name__, request_sha256=request_sha256, http_status=locals().get("http_status"))) from exc
        except Exception as exc:
            raise RecallPlannerResponseError(f"recall role planner request failed: {exc.__class__.__name__}: {exc}", request_metadata=_metadata_v4(physical_call_id=physical_call_id, key_index=key_index, started=started, finish_reason="request_error", error_type=exc.__class__.__name__, request_sha256=request_sha256)) from exc
        try:
            usage = _normalize_usage(
                response_payload.get("usage") if isinstance(response_payload, Mapping) else None
            )
        except RecallPlannerError as exc:
            raise RecallPlannerResponseError(
                f"recall role planner response usage is invalid: {exc}",
                response_content=raw_http,
                request_metadata=_metadata_v4(
                    physical_call_id=physical_call_id,
                    key_index=key_index,
                    started=started,
                    finish_reason="invalid_usage",
                    content=raw_http,
                    http_status=http_status,
                    request_sha256=request_sha256,
                    response_id=_text(response_payload.get("id")) if isinstance(response_payload, Mapping) else "",
                ),
            ) from exc
        choices = response_payload.get("choices") if isinstance(response_payload, Mapping) else None
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
            raise RecallPlannerResponseError("recall role planner response must contain exactly one choice", response_content=raw_http, request_metadata=_metadata_v4(physical_call_id=physical_call_id, key_index=key_index, started=started, finish_reason="invalid_response", content=raw_http, usage=usage, request_sha256=request_sha256, http_status=http_status, response_id=_text(response_payload.get("id"))))
        choice, message = choices[0], choices[0].get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        finish_reason = _text(choice.get("finish_reason"))
        metadata = _metadata_v4(physical_call_id=physical_call_id, key_index=key_index, started=started, finish_reason=finish_reason, content=content if isinstance(content, str) else raw_http, usage=usage, request_sha256=request_sha256, http_status=http_status, response_id=_text(response_payload.get("id")))
        if finish_reason != "stop" or not isinstance(content, str):
            raise RecallPlannerResponseError("recall role planner response did not finish with a JSON string", response_content=raw_http, request_metadata=metadata)
        try:
            plan = validate_recall_role_plan(json.loads(content))
        except (json.JSONDecodeError, RecallPlannerError) as exc:
            raise RecallPlannerResponseError(f"recall role planner returned invalid RecallRolePlan: {exc}", response_content=content, request_metadata=metadata) from exc
        if plan["query_kind"] not in QUERY_KINDS:
            metadata["validation_warnings"] = [
                {
                    "code": "noncanonical_query_kind",
                    "query_kind": plan["query_kind"],
                    "disposition": "preserved_as_standard_query",
                }
            ]
        return plan, metadata


RecallRolePlanner = DeepSeekFlashRecallRolePlanner
DeepSeekFlashRecallPlanner = DeepSeekFlashRecallRolePlanner
validate_plan = validate_recall_role_plan


def _source_parents(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = item.get("source_parents")
    if raw is None and isinstance(item.get("source_parent"), Mapping):
        raw = [item["source_parent"]]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise RecallPlannerError("slow candidate requires source_parents")
    output: list[dict[str, Any]] = []
    for parent in raw:
        if not isinstance(parent, Mapping):
            raise RecallPlannerError("slow candidate source_parent must be an object")
        try:
            start, end = int(parent["evidence_char_start"]), int(parent["evidence_char_end"])
            session, chunk = int(parent["session_index"]), int(parent["parent_chunk_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RecallPlannerError("slow candidate source_parent lacks an integer evidence span") from exc
        if start < 0 or end <= start:
            raise RecallPlannerError("slow candidate source_parent has an invalid evidence span")
        output.append({**dict(parent), "session_index": session, "parent_chunk_index": chunk, "evidence_char_start": start, "evidence_char_end": end})
    return output


def apply_recall_role_plan(plan: Mapping[str, Any], source_candidates: Sequence[Mapping[str, Any]], fast_candidates: Sequence[Mapping[str, Any]], slow_candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create units without gating; only source candidates can become evidence."""
    normalized = validate_recall_role_plan(plan)
    raw_priorities = {layer: normalized["layers"][layer]["weight"] * ROLE_PRIORS[normalized["layers"][layer]["role"]] for layer in LAYER_NAMES}
    total_priority = sum(raw_priorities.values())
    if total_priority <= 0.0:
        raise RecallPlannerError("recall role plan cannot assign zero normalized priority to every layer")
    normalized_priorities = {layer: raw_priorities[layer] / total_priority for layer in LAYER_NAMES}
    units: list[dict[str, Any]] = []
    for layer, candidates, key, kind in (("source", source_candidates, "source_candidate", "source_window"), ("fast", fast_candidates, "fast_candidate", "fast_atomic"), ("slow", slow_candidates, "slow_candidate", "slow_capsule")):
        for layer_rank, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, Mapping):
                raise RecallPlannerError(f"{layer} candidate must be a mapping")
            item = dict(candidate)
            if layer == "slow":
                item["source_parents"] = _source_parents(candidate)
            within_layer_score = 1.0 / float(layer_rank)
            units.append({"unit_type": kind, "layer": layer, "canonical_slot": _text(candidate.get("canonical_slot")) or layer, key: item, "layer_role": normalized["layers"][layer]["role"], "layer_weight": normalized["layers"][layer]["weight"], "role_prior": ROLE_PRIORS[normalized["layers"][layer]["role"]], "normalized_priority": normalized_priorities[layer], "layer_rank": layer_rank, "within_layer_score": within_layer_score, "priority_score": normalized_priorities[layer] * within_layer_score})
    if not units:
        raise RecallPlannerError("recall role plan produced no candidate units")
    return units


def layer_weight(plan: Mapping[str, Any], layer: str) -> float:
    normalized = validate_recall_role_plan(plan)
    if layer not in LAYER_NAMES:
        raise RecallPlannerError(f"unknown recall layer: {layer!r}")
    return float(normalized["layers"][layer]["weight"])


def normalized_layer_priorities(plan: Mapping[str, Any]) -> dict[str, float]:
    normalized = validate_recall_role_plan(plan)
    raw = {layer: normalized["layers"][layer]["weight"] * ROLE_PRIORS[normalized["layers"][layer]["role"]] for layer in LAYER_NAMES}
    total = sum(raw.values())
    if total <= 0.0:
        raise RecallPlannerError("recall role plan cannot assign zero normalized priority to every layer")
    return {layer: raw[layer] / total for layer in LAYER_NAMES}
