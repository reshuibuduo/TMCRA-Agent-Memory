"""Strict DeepSeek Flash control plane for recall composition."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any


PLANNER_VERSION = "tmcra.recall-plan.v4"
PLANNER_PROMPT_VERSION = "tmcra-recall-planner-2026-07-11.1"
DEEPSEEK_FLASH_MODEL = "deepseek-v4-flash"
MODES = frozenset(
    {
        "FAST_ONLY",
        "SLOW_ONLY",
        "SLOW_WITH_FAST_OVERRIDE",
        "FAST_WITH_SLOW_CONTEXT",
        "CONFLICT_COMPARE",
    }
)
PLAN_FIELDS = frozenset(
    {
        "mode",
        "primary_layer",
        "fast_role",
        "slow_role",
        "requires_conflict_pairs",
        "decision_code",
        "resolved_query",
        "decision_reason",
        "planner_version",
    }
)
_EXPECTED_POLICIES = {
    "FAST_ONLY": ("fast", "primary", "excluded", False),
    "SLOW_ONLY": ("slow", "excluded", "primary", False),
    "SLOW_WITH_FAST_OVERRIDE": ("slow", "override", "primary", False),
    "FAST_WITH_SLOW_CONTEXT": ("fast", "primary", "context_only", False),
    "CONFLICT_COMPARE": ("fast", "conflict_candidate", "conflict_candidate", True),
}
_EXPECTED_DECISION_CODES = {
    "FAST_ONLY": "recent_event",
    "SLOW_ONLY": "stable_or_historical_durable",
    "SLOW_WITH_FAST_OVERRIDE": "current_durable",
    "FAST_WITH_SLOW_CONTEXT": "current_task_with_durable_context",
    "CONFLICT_COMPARE": "explicit_conflict_or_comparison",
}

SYSTEM_PROMPT = f"""You are the retrieval control plane of a production hierarchical memory graph.
Return exactly one JSON object with these fields and no others:
{{"mode":"FAST_ONLY|SLOW_ONLY|SLOW_WITH_FAST_OVERRIDE|FAST_WITH_SLOW_CONTEXT|CONFLICT_COMPARE",
  "primary_layer":"fast|slow",
  "fast_role":"primary|excluded|override|conflict_candidate",
  "slow_role":"primary|excluded|context_only|conflict_candidate",
  "requires_conflict_pairs":true|false,
  "decision_code":"recent_event|stable_or_historical_durable|current_durable|current_task_with_durable_context|explicit_conflict_or_comparison",
  "resolved_query":"standalone retrieval query in the user's language",
  "decision_reason":"concise reason grounded only in the query and recent dialogue",
  "planner_version":"{PLANNER_VERSION}"}}

Mode contracts are exact:
- FAST_ONLY: explicitly recent episodic event, current task, or interaction lookup that needs no durable user background.
- SLOW_ONLY: explicitly historical baseline, long-term background, or stable summary where the current effective value is not requested.
- SLOW_WITH_FAST_OVERRIDE: the default for a present/current durable identity, preference, relationship, constraint, or state. Slow is primary and fast must check for recent corrections even when the query does not say "latest".
- FAST_WITH_SLOW_CONTEXT: a current event/task is primary and durable user context is background-only.
- CONFLICT_COMPARE: the query explicitly asks about change, contradiction, before/after, or competing states.

Hard routing order:
1. Explicit comparison, contradiction, or before/after change -> CONFLICT_COMPARE.
2. A present/current/latest/still standing fact -> SLOW_WITH_FAST_OVERRIDE, never SLOW_ONLY.
3. A current task or event that needs durable personalization -> FAST_WITH_SLOW_CONTEXT.
4. An isolated recent event, task, or exact recent-turn lookup -> FAST_ONLY.
5. SLOW_ONLY is allowed only for an explicitly historical baseline or stable long-term summary.
6. An unresolved or uncertain follow-up must never use SLOW_ONLY; resolve it from recent_dialogue first.

Resolve references using recent_dialogue only when needed. resolved_query must preserve the user's language and intent,
be independently searchable without recent_dialogue, and must not answer the query or invent a memory fact. If the query
is already standalone or the dialogue is unrelated, copy the query unchanged. The current query always overrides stale
or unrelated dialogue.

Never emit scores, weights, candidate IDs, benchmark labels, or an answer to the query. Never use a layer marked
unavailable. Choose exactly one mode. This planner chooses roles only; it does not rank or fuse evidence."""


class RecallPlannerError(RuntimeError):
    pass


class RecallPlannerResponseError(RecallPlannerError):
    def __init__(self, message: str, *, response_content: str = "", request_metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.response_content = response_content
        self.request_metadata = dict(request_metadata or {})


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metadata(*, key_index: int, started: float, finish_reason: str, content: str = "", usage: Mapping[str, Any] | None = None, http_status: int | None = None, error_type: str | None = None, request_sha256: str = "") -> dict[str, Any]:
    usage = usage or {}
    result = {
        "provider": "deepseek",
        "model": DEEPSEEK_FLASH_MODEL,
        "api_key_index": key_index,
        "physical_call_count": 1,
        "latency_seconds": round(time.time() - started, 3),
        "response_sha256": _sha256(content),
        "finish_reason": finish_reason,
        "planner_version": PLANNER_VERSION,
        "prompt_version": PLANNER_PROMPT_VERSION,
        "request_sha256": request_sha256,
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }
    if http_status is not None:
        result["http_status"] = int(http_status)
    if error_type:
        result["error_type"] = error_type
    return result


def _reject_gold(value: Any, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if "gold" in key_text or "benchmark" in key_text or "expected_answer" in key_text:
                raise RecallPlannerError(f"{path}.{key} is not permitted in planner input")
            _reject_gold(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_gold(item, f"{path}[{index}]")


def validate_recall_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != PLAN_FIELDS:
        raise RecallPlannerError("RecallPlan root must contain exactly the required schema fields")
    mode = _text(value.get("mode"))
    if mode not in MODES:
        raise RecallPlannerError(f"unsupported RecallPlan mode: {mode!r}")
    primary_layer, fast_role, slow_role, requires_pairs = _EXPECTED_POLICIES[mode]
    if (
        value.get("primary_layer") != primary_layer
        or value.get("fast_role") != fast_role
        or value.get("slow_role") != slow_role
        or value.get("requires_conflict_pairs") is not requires_pairs
    ):
        raise RecallPlannerError(f"RecallPlan role policy does not match mode {mode}")
    if value.get("decision_code") != _EXPECTED_DECISION_CODES[mode]:
        raise RecallPlannerError(f"RecallPlan decision_code does not match mode {mode}")
    reason = _text(value.get("decision_reason"))
    if not reason:
        raise RecallPlannerError("RecallPlan decision_reason is required")
    resolved_query = _text(value.get("resolved_query"))
    if not resolved_query:
        raise RecallPlannerError("RecallPlan resolved_query is required")
    if len(resolved_query) > 2000:
        raise RecallPlannerError("RecallPlan resolved_query exceeds 2000 characters")
    if value.get("planner_version") != PLANNER_VERSION:
        raise RecallPlannerError(f"RecallPlan planner_version must be {PLANNER_VERSION!r}")
    return {field: value[field] for field in PLAN_FIELDS}


def _validate_recent_dialogue(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RecallPlannerError("recent_dialogue must be a sequence of turn objects")
    if len(value) > 8:
        raise RecallPlannerError("recent_dialogue may contain at most 8 turns")
    output: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"turn_index", "speaker", "text"}:
            raise RecallPlannerError(f"recent_dialogue[{index}] must contain exactly turn_index, speaker, and text")
        try:
            turn_index = int(item["turn_index"])
        except (TypeError, ValueError) as exc:
            raise RecallPlannerError(f"recent_dialogue[{index}].turn_index must be an integer") from exc
        speaker = _text(item.get("speaker")).lower()
        text = _text(item.get("text"))
        if speaker not in {"user", "assistant"} or not text:
            raise RecallPlannerError(f"recent_dialogue[{index}] has an invalid speaker or empty text")
        if len(text) > 4000:
            raise RecallPlannerError(f"recent_dialogue[{index}].text exceeds 4000 characters")
        output.append({"turn_index": turn_index, "speaker": speaker, "text": text})
    if any(left["turn_index"] >= right["turn_index"] for left, right in zip(output, output[1:])):
        raise RecallPlannerError("recent_dialogue must be in strictly increasing turn order")
    _reject_gold(output, "recent_dialogue")
    return output


def _layer_available(value: Any) -> bool:
    if isinstance(value, Mapping) and "available" in value:
        return value.get("available") is True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(value)
    return bool(value)


def validate_plan_availability(plan: Mapping[str, Any], available_layers: Mapping[str, Any]) -> None:
    fast_available = _layer_available(available_layers.get("fast"))
    slow_available = _layer_available(available_layers.get("slow"))
    mode = plan["mode"]
    if mode in {"FAST_ONLY", "SLOW_WITH_FAST_OVERRIDE", "FAST_WITH_SLOW_CONTEXT", "CONFLICT_COMPARE"} and not fast_available:
        raise RecallPlannerError(f"RecallPlan mode {mode} requires an unavailable fast layer")
    if mode in {"SLOW_ONLY", "SLOW_WITH_FAST_OVERRIDE", "FAST_WITH_SLOW_CONTEXT", "CONFLICT_COMPARE"} and not slow_available:
        raise RecallPlannerError(f"RecallPlan mode {mode} requires an unavailable slow layer")


class DeepSeekFlashRecallPlanner:
    """One physical Flash call per plan request; errors are never retried or routed."""

    def __init__(self, *, base_url: str, model: str, api_keys: Sequence[str], timeout: float = 30, max_tokens: int = 512) -> None:
        self.base_url = _text(base_url).rstrip("/")
        self.model = model
        self.api_keys = list(dict.fromkeys(_text(key) for key in api_keys if _text(key)))
        self.timeout = max(1.0, float(timeout))
        self.max_tokens = max(128, int(max_tokens))
        self.request_index = 0
        if not self.base_url or not self.api_keys:
            raise RecallPlannerError("planner base_url and API key pool are required")
        if self.model != DEEPSEEK_FLASH_MODEL:
            raise RecallPlannerError(f"recall planner requires {DEEPSEEK_FLASH_MODEL}, got {self.model!r}")

    def plan(self, *, query: str, question_date: str, available_layers: Mapping[str, Any], recent_dialogue: Sequence[Mapping[str, Any]] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if not _text(query) or not _text(question_date):
            raise RecallPlannerError("query and question_date are required")
        if not isinstance(available_layers, Mapping) or set(available_layers) - {"fast", "slow"}:
            raise RecallPlannerError("available_layers may contain only fast and slow summaries")
        _reject_gold(available_layers, "available_layers")
        dialogue = _validate_recent_dialogue(recent_dialogue)
        payload = {"query": query, "question_date": question_date, "recent_dialogue": dialogue, "available_layers": dict(available_layers)}
        key_index = self.request_index % len(self.api_keys)
        self.request_index += 1
        request_body = {
            "model": DEEPSEEK_FLASH_MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        }
        request_sha256 = _sha256(json.dumps(request_body, ensure_ascii=False, sort_keys=True))
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_keys[key_index]}"},
            method="POST",
        )
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_http = response.read().decode("utf-8")
                response_payload = json.loads(raw_http)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RecallPlannerResponseError(f"recall planner HTTP {exc.code}: {detail}", response_content=detail, request_metadata=_metadata(key_index=key_index, started=started, finish_reason="http_error", content=detail, http_status=exc.code, request_sha256=request_sha256)) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raw = locals().get("raw_http", "")
            raise RecallPlannerResponseError("recall planner returned invalid HTTP JSON", response_content=raw, request_metadata=_metadata(key_index=key_index, started=started, finish_reason="invalid_http_json", content=raw, error_type=exc.__class__.__name__, request_sha256=request_sha256)) from exc
        except Exception as exc:
            raise RecallPlannerResponseError(f"recall planner request failed: {exc.__class__.__name__}: {exc}", request_metadata=_metadata(key_index=key_index, started=started, finish_reason="request_error", error_type=exc.__class__.__name__, request_sha256=request_sha256)) from exc
        usage = response_payload.get("usage") if isinstance(response_payload, Mapping) else {}
        choices = response_payload.get("choices") if isinstance(response_payload, Mapping) else None
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
            raise RecallPlannerResponseError("recall planner response must contain exactly one choice", response_content=raw_http, request_metadata=_metadata(key_index=key_index, started=started, finish_reason="invalid_response", content=raw_http, usage=usage if isinstance(usage, Mapping) else {}, request_sha256=request_sha256))
        choice = choices[0]
        content = (choice.get("message") or {}).get("content") if isinstance(choice.get("message"), Mapping) else None
        finish_reason = _text(choice.get("finish_reason"))
        metadata = _metadata(key_index=key_index, started=started, finish_reason=finish_reason, content=content if isinstance(content, str) else raw_http, usage=usage if isinstance(usage, Mapping) else {}, request_sha256=request_sha256)
        if finish_reason != "stop" or not isinstance(content, str):
            raise RecallPlannerResponseError("recall planner response did not finish with a JSON string", response_content=raw_http, request_metadata=metadata)
        try:
            plan = validate_recall_plan(json.loads(content))
        except (json.JSONDecodeError, RecallPlannerError) as exc:
            raise RecallPlannerResponseError(f"recall planner returned invalid RecallPlan: {exc}", response_content=content, request_metadata=metadata) from exc
        try:
            validate_plan_availability(plan, available_layers)
        except RecallPlannerError as exc:
            raise RecallPlannerResponseError(
                f"recall planner selected an unavailable layer: {exc}",
                response_content=content,
                request_metadata=metadata,
            ) from exc
        return plan, metadata


def _source_parents(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_values: list[Any] = []
    if isinstance(item.get("source_parent"), Mapping):
        raw_values.append(item["source_parent"])
    if isinstance(item.get("source_parents"), Sequence) and not isinstance(item.get("source_parents"), (str, bytes)):
        raw_values.extend(item["source_parents"])
    parents: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for raw in raw_values:
        if not isinstance(raw, Mapping):
            raise RecallPlannerError("slow capsule source parent must be an object")
        try:
            location = (int(raw["session_index"]), int(raw["parent_chunk_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RecallPlannerError("slow capsule source parent lacks integer session/parent coordinates") from exc
        if location in seen:
            continue
        seen.add(location)
        parents.append(dict(raw))
    return parents


def _group_by_slot(items: Sequence[Mapping[str, Any]], *, layer: str, require_parent: bool) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise RecallPlannerError(f"{layer} candidate must be a mapping")
        slot = _text(item.get("canonical_slot"))
        if not slot:
            raise RecallPlannerError(f"{layer} candidate lacks canonical_slot")
        if require_parent and not _source_parents(item):
            raise RecallPlannerError(f"slow capsule {slot!r} lacks source_parent mapping")
        grouped.setdefault(slot, []).append(item)
    return grouped


def apply_recall_plan(plan: Mapping[str, Any], fast_candidates: Sequence[Mapping[str, Any]], slow_capsules: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compose evidence units by role, without combining or recalculating layer scores."""
    normalized = validate_recall_plan(plan)
    fast = _group_by_slot(fast_candidates, layer="fast", require_parent=False)
    slow = _group_by_slot(slow_capsules, layer="slow", require_parent=True)
    mode = normalized["mode"]
    units: list[dict[str, Any]] = []
    if mode == "FAST_ONLY":
        return [
            {"unit_type": "fast_primary", "canonical_slot": slot, "primary_layer": "fast", "fast_candidate": candidate}
            for slot, candidates in fast.items()
            for candidate in candidates
        ]
    if mode == "SLOW_ONLY":
        return [
            {"unit_type": "slow_primary", "canonical_slot": slot, "primary_layer": "slow", "slow_capsule": capsule}
            for slot, capsules in slow.items()
            for capsule in capsules
        ]
    if mode == "SLOW_WITH_FAST_OVERRIDE":
        for slot, capsules in slow.items():
            for capsule in capsules:
                units.append(
                    {
                        "unit_type": "slow_primary_with_fast_override",
                        "canonical_slot": slot,
                        "primary_layer": "slow",
                        "slow_capsule": capsule,
                        "fast_overrides": list(fast.get(slot, [])),
                    }
                )
        return units
    if mode == "FAST_WITH_SLOW_CONTEXT":
        all_slow_context = [capsule for capsules in slow.values() for capsule in capsules]
        for slot, candidates in fast.items():
            for candidate in candidates:
                units.append(
                    {
                        "unit_type": "fast_primary_with_slow_context",
                        "canonical_slot": slot,
                        "primary_layer": "fast",
                        "fast_candidate": candidate,
                        "slow_context": list(all_slow_context),
                    }
                )
        return units
    for slot in fast.keys() & slow.keys():
        units.append(
            {
                "unit_type": "conflict_group",
                "canonical_slot": slot,
                "primary_layer": "fast",
                "fast_candidates": list(fast[slot]),
                "slow_capsules": list(slow[slot]),
            }
        )
    if not units and fast and slow:
        ranked_fast = [candidate for candidates in fast.values() for candidate in candidates]
        ranked_slow = [capsule for capsules in slow.values() for capsule in capsules]
        pair_count = max(len(ranked_fast), len(ranked_slow))
        for index in range(pair_count):
            units.append(
                {
                    "unit_type": "conflict_group",
                    "canonical_slot": f"semantic_conflict.{index}",
                    "primary_layer": "fast",
                    "fast_candidates": [ranked_fast[min(index, len(ranked_fast) - 1)]],
                    "slow_capsules": [ranked_slow[min(index, len(ranked_slow) - 1)]],
                }
            )
    if not units:
        raise RecallPlannerError("CONFLICT_COMPARE produced no semantic conflict pairs")
    return units
