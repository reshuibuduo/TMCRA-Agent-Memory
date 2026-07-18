"""TMCRA V4 online retrieval controller.

V3 supplies the index format, model implementations, graph adapter, and helper
functions. V4 owns recall execution and composition: source, fast, and slow
paths run independently before bounded role weights are applied.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from tmcra_v4_recall_planner import (
    DEEPSEEK_FLASH_MODEL,
    DeepSeekFlashRecallRolePlanner,
    RecallPlannerError,
    apply_recall_role_plan,
    layer_weight,
    normalized_layer_priorities,
    validate_recall_role_plan,
)
from tmcra_v4_route_policy import (
    PRODUCTION_FINAL_TOP_K,
    RETRIEVAL_CONTRACT_SCHEMA,
    SOURCE_COVERAGE_TRACE_K,
    RoutePolicyError,
    validate_production_packing_budget,
    validate_production_retrieval_mode,
)
from tmcra_v4_slow_graph import (
    PatchValidationError,
    validate_semantic_summary,
)

_V3: Any | None = None
ONLINE_INDEX_SCHEMA_VERSION = "tmcra.v4.online-index.3"
ONLINE_INDEX_REPORT_SCHEMA_VERSION = "tmcra.v4.online-index-report.1"
SLOW_INVENTORY_SCHEMA_VERSION = "tmcra.v4.slow-inventory.1"
SLOW_SUMMARY_CONTRACT_VERSION = "tmcra.v4.slow-lossless-summary.2"
CURRENT_SLOW_PROMPT_VERSION = "tmcra-v4-slow-graph-2026-07-14.16"
CURRENT_SLOW_PARTITION_CONTRACT_VERSION = "tmcra.v4.slow-semantic-partition.2"
RUNTIME_SCHEMA_VERSION = "tmcra.v4.online-retrieval.6"
SESSION_ORDERING_POLICY = "session_rrf_then_chronological_v1"
RECENT_DIALOGUE_MAX_TURNS = 8
RECENT_DIALOGUE_MAX_CHARS = 4000
ROW_CHECKPOINT_SCHEMA = "tmcra.v4.retrieval-row-checkpoint.1"
PLANNER_DECISION_SCHEMA = "tmcra.v4.recall-planner-decision.1"
RUN_STAGING_SCHEMA = "tmcra.v4.retrieval-staging.2"
PACKING_BUDGET_MODES = frozenset({"fixed", "adaptive"})
COMPOSITION_MODES = frozenset({"layered", "source-only-diagnostic"})
EXECUTION_LANES = frozenset({"production", "diagnostic"})
COMPLEX_QUERY_KINDS = frozenset({"comparison", "historical"})
COMPLEX_TEMPORAL_FOCUSES = frozenset({"historical", "mixed"})
COMPLEX_CONFLICT_POLICIES = frozenset(
    {"compare", "preserve_parallel", "surface_uncertainty"}
)
SIMPLE_QUERY_KINDS = frozenset({"fact"})
SIMPLE_TEMPORAL_FOCUSES = frozenset({"timeless"})
CURRENT_FAST_RECORD_STATES = frozenset(
    {"active", "parallel_active", "promoted", "challenged"}
)


def _v3() -> Any:
    global _V3
    if _V3 is None:
        try:
            import tmcra_v3_online_runtime as runtime
        except Exception as exc:
            raise RuntimeError("TMCRA V3 runtime dependencies are unavailable") from exc
        _V3 = runtime
    return _V3


def __getattr__(name: str) -> Any:
    if name in {"OnlineModels", "scope_counts", "scope_fingerprint", "load_recent_dialogue_context", "load_native_harness", "append_layered_retrieval_audit", "graph_runtime_env", "read_jsonl"}:
        return getattr(_v3(), name)
    raise AttributeError(name)


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _validated_runtime_route(
    args: argparse.Namespace, *, label: str
) -> tuple[str, str, str, int]:
    composition_mode = str(_arg(args, "composition_mode", "layered"))
    execution_lane = str(_arg(args, "execution_lane", "production"))
    packing_budget_mode = str(_arg(args, "packing_budget_mode", "fixed"))
    top_k = _arg(args, "top_k", PRODUCTION_FINAL_TOP_K)
    try:
        validate_production_retrieval_mode(
            composition_mode, execution_lane=execution_lane
        )
        validate_production_packing_budget(
            packing_budget_mode,
            top_k,
            execution_lane=execution_lane,
        )
    except RoutePolicyError as exc:
        raise RuntimeError(f"{label}: retrieval route policy rejected the run: {exc}") from exc
    return composition_mode, execution_lane, packing_budget_mode, int(top_k)


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalized_grounding_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _lossless_summary_projection(claims: Any) -> str:
    """Project final claims to the V4.7 stored-summary representation."""
    if not isinstance(claims, list) or not claims:
        raise RuntimeError("lossless Slow summary projection requires claims")
    texts: list[str] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            raise RuntimeError(f"lossless Slow summary claim {index} is not an object")
        text = " ".join(_clean(claim.get("text")).split())
        if not text:
            raise RuntimeError(f"lossless Slow summary claim {index} lacks text")
        texts.append(text)
    return " ".join(texts)


def _current_summary_contract(*values: Any) -> bool:
    for value in values:
        if isinstance(value, Mapping):
            if (
                value.get("summary_contract_version") == SLOW_SUMMARY_CONTRACT_VERSION
                or value.get("prompt_version") == CURRENT_SLOW_PROMPT_VERSION
                or value.get("partition_contract_version")
                == "tmcra.v4.slow-semantic-partition.1"
            ):
                return True
            if _current_summary_contract(value.get("provenance")):
                return True
        elif value == SLOW_SUMMARY_CONTRACT_VERSION:
            return True
    return False


def _validate_active_capsule_partition(
    slow: Sequence[Mapping[str, Any]],
) -> None:
    """Allow controlled compound-support fanout while rejecting duplicate claims."""
    by_region: dict[str, dict[str, Mapping[str, Any]]] = {}
    for item in slow:
        if item.get("candidate_kind") != "capsule_summary":
            continue
        if _clean(item.get("status")).casefold() not in {"active", "challenged"}:
            continue
        if not _current_summary_contract(item):
            continue
        region_key = _clean(item.get("region_key"))
        if not region_key:
            raise RuntimeError(
                "current Slow inventory candidate lacks region_key; legacy fallback is not allowed"
            )
        capsule_id = _clean(item.get("capsule_id"))
        if not capsule_id:
            raise RuntimeError("current Slow inventory candidate lacks capsule_id")
        if (
            item.get("partition_contract_version")
            != CURRENT_SLOW_PARTITION_CONTRACT_VERSION
        ):
            raise RuntimeError(
                f"{capsule_id}: Slow semantic partition contract is stale or missing"
            )
        by_region.setdefault(region_key, {})[capsule_id] = item

    for region_key, capsules in by_region.items():
        evidence_locations: dict[
            str, list[tuple[str, int, str, str, str]]
        ] = {}
        claim_identity_capsules: dict[tuple[str, str], set[str]] = {}
        for capsule_id, summary in capsules.items():
            for claim_index, claim in enumerate(list(summary.get("claims") or [])):
                if not isinstance(claim, Mapping):
                    continue
                slot = _clean(claim.get("canonical_slot"))
                claim_text = " ".join(_clean(claim.get("text")).casefold().split())
                if slot and claim_text:
                    claim_identity_capsules.setdefault(
                        (slot, claim_text), set()
                    ).add(capsule_id)
                for role in ("support", "counterevidence"):
                    for evidence_id in list(claim.get(role) or []):
                        evidence_id = _clean(evidence_id)
                        if evidence_id:
                            evidence_locations.setdefault(evidence_id, []).append(
                                (
                                    capsule_id,
                                    claim_index,
                                    role,
                                    slot,
                                    claim_text,
                                )
                            )
        invalid_repeated_evidence: dict[str, dict[str, Any]] = {}
        for evidence_id, locations in evidence_locations.items():
            if len(locations) <= 1:
                continue
            roles = {role for _, _, role, _, _ in locations}
            claim_identities = [
                (slot, text) for _, _, _, slot, text in locations
            ]
            if roles == {"support"} and len(set(claim_identities)) == len(
                claim_identities
            ):
                continue
            invalid_repeated_evidence[evidence_id] = {
                "locations": [
                    f"{capsule}:{claim_index}:{role}"
                    for capsule, claim_index, role, _, _ in locations
                ],
                "roles": sorted(roles),
                "duplicate_semantic_binding": (
                    len(set(claim_identities)) != len(claim_identities)
                ),
            }
        if invalid_repeated_evidence:
            raise RuntimeError(
                f"region {region_key}: duplicate evidence citation across active Slow capsules: "
                f"{json.dumps(invalid_repeated_evidence, ensure_ascii=False, sort_keys=True)}"
            )
        duplicated_claims = {
            f"{slot}\u241f{text}": sorted(capsule_ids)
            for (slot, text), capsule_ids in claim_identity_capsules.items()
            if len(capsule_ids) > 1
        }
        if duplicated_claims:
            raise RuntimeError(
                f"region {region_key}: semantic claim appears in more than one active Slow capsule: "
                f"{json.dumps(duplicated_claims, ensure_ascii=False, sort_keys=True)}"
            )


def resolve_packing_budget(
    plan: Mapping[str, Any],
    *,
    mode: str,
    fixed_k: int,
    simple_k: int,
    standard_k: int,
    complex_k: int,
) -> tuple[int, dict[str, Any]]:
    """Resolve a per-query evidence budget from the validated recall plan."""
    normalized = validate_recall_role_plan(plan)
    if mode not in PACKING_BUDGET_MODES:
        raise RuntimeError(f"unsupported packing budget mode: {mode!r}")
    values = (fixed_k, simple_k, standard_k, complex_k)
    if any(isinstance(value, bool) or int(value) <= 0 for value in values):
        raise RuntimeError("packing budgets must be positive integers")
    if not simple_k <= standard_k <= complex_k:
        raise RuntimeError(
            "adaptive packing budgets must satisfy simple <= standard <= complex"
        )
    if mode == "fixed":
        return int(fixed_k), {
            "mode": "fixed",
            "tier": "fixed",
            "budget": int(fixed_k),
            "reasons": ["explicit_fixed_budget"],
        }

    query_kind = normalized["query_kind"]
    temporal_focus = normalized["temporal_focus"]
    conflict_policy = normalized["conflict_policy"]
    complex_reasons: list[str] = []
    if query_kind in COMPLEX_QUERY_KINDS:
        complex_reasons.append(f"query_kind:{query_kind}")
    if temporal_focus in COMPLEX_TEMPORAL_FOCUSES:
        complex_reasons.append(f"temporal_focus:{temporal_focus}")
    if conflict_policy in COMPLEX_CONFLICT_POLICIES:
        complex_reasons.append(f"conflict_policy:{conflict_policy}")
    if complex_reasons:
        tier, budget, reasons = "complex", int(complex_k), complex_reasons
    elif (
        query_kind in SIMPLE_QUERY_KINDS
        and temporal_focus in SIMPLE_TEMPORAL_FOCUSES
        and conflict_policy not in COMPLEX_CONFLICT_POLICIES
    ):
        tier, budget, reasons = "simple", int(simple_k), [
            f"query_kind:{query_kind}",
            f"temporal_focus:{temporal_focus}",
        ]
    else:
        tier, budget, reasons = "standard", int(standard_k), [
            f"query_kind:{query_kind}",
            f"temporal_focus:{temporal_focus}",
            f"conflict_policy:{conflict_policy}",
        ]
    return budget, {
        "mode": "adaptive",
        "tier": tier,
        "budget": budget,
        "reasons": reasons,
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def project_recent_dialogue(
    value: Sequence[Mapping[str, Any]] | None,
    *,
    max_turns: int = RECENT_DIALOGUE_MAX_TURNS,
    max_chars: int = RECENT_DIALOGUE_MAX_CHARS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project a dialogue tail without truncating text or orphaning replies."""
    if max_turns <= 0 or max_chars <= 0:
        raise RuntimeError("recent dialogue projection limits must be positive")
    if value is None:
        value = []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError("recent dialogue must be a sequence")

    normalized: list[dict[str, Any]] = []
    for position, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {
            "turn_index",
            "speaker",
            "text",
        }:
            raise RuntimeError(
                f"recent dialogue turn {position} has an invalid schema"
            )
        try:
            turn_index = int(item["turn_index"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"recent dialogue turn {position} has an invalid turn_index"
            ) from exc
        speaker = _clean(item.get("speaker")).lower()
        text = _clean(item.get("text"))
        if speaker not in {"user", "assistant"} or not text:
            raise RuntimeError(
                f"recent dialogue turn {position} has an invalid speaker or text"
            )
        normalized.append(
            {"turn_index": turn_index, "speaker": speaker, "text": text}
        )
    if any(
        left["turn_index"] >= right["turn_index"]
        for left, right in zip(normalized, normalized[1:])
    ):
        raise RuntimeError("recent dialogue is not in strictly increasing turn order")

    excluded: list[dict[str, Any]] = []

    def reject(item: Mapping[str, Any], reason: str) -> None:
        excluded.append(
            {
                "turn_index": int(item["turn_index"]),
                "speaker": str(item["speaker"]),
                "text_chars": len(str(item["text"])),
                "reason": reason,
            }
        )

    tail = normalized[-max_turns:]
    for item in normalized[: max(0, len(normalized) - len(tail))]:
        reject(item, "outside_turn_budget")

    projected: list[dict[str, Any]] = []
    index = 0
    while index < len(tail):
        item = tail[index]
        if item["speaker"] == "assistant":
            reject(
                item,
                "text_over_limit"
                if len(item["text"]) > max_chars
                else "orphan_assistant",
            )
            index += 1
            continue

        following = tail[index + 1] if index + 1 < len(tail) else None
        if following is not None and following["speaker"] == "assistant":
            pair = (item, following)
            oversized = [member for member in pair if len(member["text"]) > max_chars]
            if oversized:
                oversized_ids = {int(member["turn_index"]) for member in oversized}
                for member in pair:
                    reject(
                        member,
                        "text_over_limit"
                        if int(member["turn_index"]) in oversized_ids
                        else "paired_with_excluded_turn",
                    )
            else:
                projected.extend(dict(member) for member in pair)
            index += 2
            continue

        if len(item["text"]) > max_chars:
            reject(item, "text_over_limit")
        else:
            projected.append(dict(item))
        index += 1

    reason_counts: dict[str, int] = {}
    for item in excluded:
        reason = str(item["reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    metadata = {
        "policy": "complete_user_assistant_pairs_no_truncation_v1",
        "input_count": len(normalized),
        "considered_count": len(tail),
        "included_count": len(projected),
        "excluded_count": len(excluded),
        "included_turn_indexes": [int(item["turn_index"]) for item in projected],
        "excluded_turns": excluded,
        "excluded_by_reason": reason_counts,
        "max_turns": max_turns,
        "max_chars_per_turn": max_chars,
        "text_truncation_count": 0,
    }
    return projected, metadata


def source_coverage_trace(
    candidates: Sequence[Mapping[str, Any]], *, limit: int = SOURCE_COVERAGE_TRACE_K
) -> list[dict[str, Any]]:
    """Return text-free source coordinates for offline coverage evaluation."""
    if limit <= 0:
        raise RuntimeError("source coverage trace limit must be positive")
    trace: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates[:limit], start=1):
        session_id = _clean(candidate.get("session_id"))
        candidate_id = _clean(candidate.get("candidate_id"))
        if not session_id or not candidate_id:
            raise RuntimeError("source candidate lacks an auditable identity")
        try:
            location = {
                "session_index": int(candidate["session_index"]),
                "parent_chunk_index": int(candidate["parent_chunk_index"]),
                "subchunk_index": int(candidate["subchunk_index"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("source candidate lacks an auditable location") from exc
        trace.append(
            {
                "rank": rank,
                "candidate_id": candidate_id,
                "session_id": session_id,
                **location,
            }
        )
    return trace


def planner_from_env() -> DeepSeekFlashRecallRolePlanner:
    pool = [item.strip() for item in os.environ.get("TMCRA_RECALL_PLANNER_API_KEY_POOL", "").split(",") if item.strip()]
    return DeepSeekFlashRecallRolePlanner(base_url=os.environ.get("TMCRA_RECALL_PLANNER_BASE_URL", ""), model=os.environ.get("TMCRA_RECALL_PLANNER_MODEL", DEEPSEEK_FLASH_MODEL), api_keys=pool)


def execute_local_candidate_paths(*, inventories: Mapping[str, Sequence[Any]], source_runner: Any, fast_runner: Any, slow_runner: Any) -> dict[str, Any]:
    """Run every supplied local generator for every nonempty inventory.

    Roles and weights are intentionally absent from this function. Exceptions
    propagate and there is no retry or fallback.
    """
    runners = {"source": source_runner, "fast": fast_runner, "slow": slow_runner}
    result: dict[str, Any] = {}
    for layer in ("source", "fast", "slow"):
        inventory = inventories.get(layer) or []
        result[layer] = runners[layer]() if inventory else []
    return result


def _hydrate_fast_semantic_records(
    db_path: Path,
    scope_id: str,
    semantic_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not semantic_records:
        return []
    ids = [_clean(item.get("memory_id")) for item in semantic_records]
    if not all(ids) or len(ids) != len(set(ids)):
        raise RuntimeError(f"{scope_id}: fast semantic index identities are invalid")
    rows: list[tuple[Any, ...]] = []
    connection = sqlite3.connect(db_path)
    try:
        for offset in range(0, len(ids), 400):
            batch = ids[offset : offset + 400]
            placeholders = ",".join("?" for _ in batch)
            rows.extend(
                connection.execute(
                    "SELECT memory_id,value,state,metadata_json FROM records "
                    f"WHERE scope_id=? AND memory_id IN ({placeholders})",
                    (scope_id, *batch),
                ).fetchall()
            )
    finally:
        connection.close()
    if len(rows) != len(ids):
        found = {str(memory_id) for memory_id, *_ in rows}
        missing = sorted(set(ids) - found)
        raise RuntimeError(f"{scope_id}: indexed fast semantic records are missing: {missing[:5]}")
    by_id = {
        str(memory_id): (value, str(state), raw_metadata)
        for memory_id, value, state, raw_metadata in rows
    }
    if set(by_id) != set(ids):
        missing = sorted(set(ids) - set(by_id))
        raise RuntimeError(f"{scope_id}: indexed fast semantic records are missing: {missing[:5]}")

    parsed: dict[str, tuple[str, str, dict[str, Any]]] = {}
    source_ids: set[str] = set()
    for memory_id in ids:
        value, state, raw_metadata = by_id[memory_id]
        try:
            metadata = json.loads(raw_metadata)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{memory_id}: fast semantic metadata is invalid JSON") from exc
        if not isinstance(metadata, Mapping):
            raise RuntimeError(f"{memory_id}: fast semantic metadata is not an object")
        if (
            _clean(state) not in CURRENT_FAST_RECORD_STATES
            or _clean(metadata.get("memory_layer")) != "fast"
            or _clean(metadata.get("content_variant")) != "product_semantic_memory"
            or _clean(metadata.get("node_kind")) != "atomic_user_assertion"
            or metadata.get("atomic_evidence_leaf") is not True
            or _clean(metadata.get("authority")) != "user_assertion"
            or not isinstance(value, str)
            or not value.strip()
        ):
            raise RuntimeError(f"{memory_id}: indexed fast semantic record is malformed or inactive")
        source_record_id = _clean(metadata.get("source_record_id"))
        if not source_record_id:
            raise RuntimeError(f"{memory_id}: fast semantic record lacks source_record_id")
        source_ids.add(source_record_id)
        parsed[memory_id] = (value, state, dict(metadata))

    source_rows: list[tuple[Any, ...]] = []
    connection = sqlite3.connect(db_path)
    try:
        source_id_list = sorted(source_ids)
        for offset in range(0, len(source_id_list), 400):
            batch = source_id_list[offset : offset + 400]
            placeholders = ",".join("?" for _ in batch)
            source_rows.extend(
                connection.execute(
                    "SELECT memory_id,value,state,metadata_json FROM records "
                    f"WHERE scope_id=? AND memory_id IN ({placeholders})",
                    (scope_id, *batch),
                ).fetchall()
            )
    finally:
        connection.close()
    if len(source_rows) != len(source_ids):
        found = {str(memory_id) for memory_id, *_ in source_rows}
        missing = sorted(source_ids - found)
        raise RuntimeError(f"{scope_id}: immutable source records are missing: {missing[:5]}")
    source_by_id = {
        str(memory_id): (value, str(state), raw_metadata)
        for memory_id, value, state, raw_metadata in source_rows
    }
    if set(source_by_id) != source_ids:
        missing = sorted(source_ids - set(source_by_id))
        raise RuntimeError(f"{scope_id}: immutable source records are missing: {missing[:5]}")

    def indexed_identity_mismatch(
        indexed: Mapping[str, Any], identity: Mapping[str, Any], memory_id: str
    ) -> None:
        for field, expected in identity.items():
            if field in indexed and indexed[field] != expected:
                raise RuntimeError(f"{memory_id}: indexed fast semantic identity differs from SQLite")

    output: list[dict[str, Any]] = []
    for indexed in semantic_records:
        memory_id = _clean(indexed.get("memory_id"))
        value, state, metadata = parsed[memory_id]
        source_record_id = _clean(metadata.get("source_record_id"))
        source_record_value, source_record_state, raw_source_metadata = source_by_id[
            source_record_id
        ]
        try:
            source_metadata = json.loads(raw_source_metadata)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{memory_id}: immutable source metadata is invalid JSON") from exc
        if not isinstance(source_metadata, Mapping):
            raise RuntimeError(f"{memory_id}: immutable source metadata is not an object")
        if (
            _clean(source_metadata.get("content_variant")) != "source_message"
            or _clean(source_metadata.get("node_kind")) != "immutable_source_message"
            or source_metadata.get("immutable_evidence_leaf") is not True
            or _clean(source_metadata.get("source_record_id")) != source_record_id
        ):
            raise RuntimeError(f"{memory_id}: immutable source record is malformed")
        source_value = source_metadata.get("raw_content")
        if (
            not isinstance(source_value, str)
            or not source_value
            or source_record_state != "evidence"
            or _normalized_grounding_text(source_record_value)
            != _normalized_grounding_text(source_value)
            or source_metadata.get("source_span") != source_value
            or source_metadata.get("source_turn_text") != source_value
        ):
            raise RuntimeError(f"{memory_id}: immutable source content is malformed")
        source_scope = source_metadata.get("scope_id")
        if source_scope is not None and _clean(source_scope) != scope_id:
            raise RuntimeError(f"{memory_id}: immutable source scope differs from requested scope")
        try:
            source_session = int(source_metadata["session_index"])
            source_parent = int(source_metadata["message_index"])
            semantic_session = int(metadata["session_index"])
            semantic_parent = int(metadata.get("parent_chunk_index", metadata.get("message_index")))
            evidence_start = int(metadata["evidence_char_start"])
            evidence_end = int(metadata["evidence_char_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{memory_id}: fast semantic identity has unusable coordinates") from exc
        if (
            source_session < 0
            or source_parent < 0
            or semantic_session != source_session
            or semantic_parent != source_parent
            or evidence_start < 0
            or evidence_end <= evidence_start
            or evidence_end > len(source_value)
        ):
            raise RuntimeError(f"{memory_id}: fast semantic identity has unusable coordinates")
        for field in (
            "session_id",
            "message_id",
            "event_id",
            "historical_date",
            "timestamp",
        ):
            source_identity_value = source_metadata.get(field)
            semantic_identity_value = metadata.get(field)
            if (
                source_identity_value is not None
                or semantic_identity_value is not None
            ) and source_identity_value != semantic_identity_value:
                raise RuntimeError(
                    f"{memory_id}: fast semantic {field} differs from immutable source"
                )
        semantic_raw_content = metadata.get("raw_content")
        semantic_source_span = metadata.get("source_span")
        if (
            not isinstance(semantic_raw_content, str)
            or not semantic_raw_content
            or semantic_source_span != semantic_raw_content
            or metadata.get("source_turn_text") != source_value
        ):
            raise RuntimeError(f"{memory_id}: fast semantic quote metadata is malformed")
        explicit_quote = metadata.get("evidence_quote")
        if explicit_quote is not None and explicit_quote != semantic_raw_content:
            raise RuntimeError(f"{memory_id}: fast semantic quote aliases disagree")
        evidence_quote = semantic_raw_content
        if source_value[evidence_start:evidence_end] != evidence_quote:
            raise RuntimeError(f"{memory_id}: fast evidence span does not match immutable source quote")
        slot = _clean(metadata.get("canonical_slot") or metadata.get("canonical_slot_key"))
        if not slot:
            raise RuntimeError(f"{memory_id}: fast semantic record lacks canonical_slot")
        if metadata.get("canonical_slot") is not None and metadata.get("canonical_slot_key") is not None and _clean(metadata.get("canonical_slot")) != _clean(metadata.get("canonical_slot_key")):
            raise RuntimeError(f"{memory_id}: fast semantic canonical slot aliases disagree")
        memory_type = _clean(metadata.get("memory_type"))
        durability = _clean(metadata.get("durability") or metadata.get("durability_class"))
        temporal_status = _clean(metadata.get("temporal_status") or metadata.get("target_status"))
        source_parent_identity = {
            "session_index": source_session,
            "parent_chunk_index": source_parent,
            "source_record_id": source_record_id,
            "evidence_char_start": evidence_start,
            "evidence_char_end": evidence_end,
        }
        provenance = {
            "memory_layer": "fast",
            "content_variant": "product_semantic_memory",
            "source_record_id": source_record_id,
            "semantic_memory_id": memory_id,
        }
        indexed_identity_mismatch(
            indexed,
            {
                "canonical_slot": slot,
                "canonical_slot_key": slot,
                "source_record_id": source_record_id,
                "source_parent": source_parent_identity,
                "provenance": provenance,
                "record_state": state,
                "memory_type": memory_type,
                "durability": durability,
                "temporal_status": temporal_status,
                "evidence_quote": evidence_quote,
                "evidence_char_start": evidence_start,
                "evidence_char_end": evidence_end,
            },
            memory_id,
        )
        output.append(
            {
                "memory_id": memory_id,
                "text": value,
                "record_state": state,
                "canonical_slot": slot,
                "source_parent": source_parent_identity,
                "provenance": provenance,
                "memory_type": memory_type,
                "durability": durability,
                "temporal_status": temporal_status,
            }
        )
    return output


def _map_fast_candidates_with_slots(candidates: Sequence[Mapping[str, Any]], semantic_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Apply the V3 semantic-leaf mapper to Fast graph candidates."""
    mapped = _v3()._fast_candidates_with_slots(candidates, semantic_records)
    by_id = {
        _clean(item.get("memory_id")): item
        for item in semantic_records
        if _clean(item.get("memory_id"))
    }
    for item in mapped:
        memories = []
        for memory_id in list(item.get("semantic_record_ids") or []):
            record = by_id.get(_clean(memory_id))
            if not isinstance(record, Mapping) or not _clean(record.get("text")):
                continue
            memories.append(
                {
                    "memory_id": _clean(record.get("memory_id")),
                    "canonical_slot": _clean(record.get("canonical_slot")),
                    "text": _clean(record.get("text")),
                    "record_state": _clean(record.get("record_state")),
                    "memory_type": _clean(record.get("memory_type")),
                    "durability": _clean(record.get("durability")),
                    "temporal_status": _clean(record.get("temporal_status")),
                    "source_parent": dict(record.get("source_parent") or {}),
                    "provenance": dict(record.get("provenance") or {}),
                }
            )
        item["semantic_memories"] = memories
    return mapped


def _get_graph_adapter(
    *,
    harness: Any,
    scope_id: str,
    db_path: Path,
    cache: dict[tuple[str, str], Any] | None,
) -> tuple[Any, bool]:
    """Keep at most one graph adapter alive during a batched retrieval run."""
    key = (scope_id, str(db_path))
    if cache is None:
        return harness.build_adapter(scope_id, db_path), False
    adapter = cache.get(key)
    if adapter is not None:
        return adapter, True
    if cache:
        cache.clear()
        gc.collect()
        torch = _v3().torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    adapter = harness.build_adapter(scope_id, db_path)
    cache[key] = adapter
    return adapter, False


def _graph_fast_path(*, qid: str, runtime_question: str, fast: Sequence[Mapping[str, Any]], dense_scores: Any, dense_rank: Mapping[int, int], semantic_records: Sequence[Mapping[str, Any]], args: argparse.Namespace, harness: Any, scope_id: str, db_path: Path, graph_adapter_cache: dict[tuple[str, str], Any] | None, models: Any) -> tuple[list[dict[str, Any]], dict[str, Any], float, float]:
    v3 = _v3()
    parents: dict[tuple[int, int], list[int]] = {}
    for index, candidate in enumerate(fast):
        parents.setdefault((int(candidate["session_index"]), int(candidate["parent_chunk_index"])), []).append(index)
    adapter, reused = _get_graph_adapter(
        harness=harness,
        scope_id=scope_id,
        db_path=db_path,
        cache=graph_adapter_cache,
    )
    started = time.time()
    retrieval = adapter.retrieve(runtime_question, top_k=_arg(args, "graph_top_k", 12))
    elapsed = time.time() - started
    metadata = dict(getattr(retrieval, "metadata", {}) or {})
    if _clean(metadata.get("retrieval_mode")) != "hybrid_node_scored" or not bool(metadata.get("hybrid_enabled")):
        raise RuntimeError(f"{qid}: graph fast path is not active")
    selected = list(metadata.get("selected_event_ids") or [])
    recall = list(metadata.get("recall_event_ids") or [])
    final = list(metadata.get("final_hit_event_ids") or [])
    if not selected:
        raise RuntimeError(f"{qid}: graph fast path selected no events")
    slow_ids = [str(item) for item in [*selected, *recall, *final] if str(item).startswith("slow.")]
    if slow_ids:
        raise RuntimeError(f"{qid}: fast graph crossed the slow-layer boundary: {','.join(dict.fromkeys(slow_ids))}")
    valid_locations = set(parents)
    selected_parents, selected_unmapped = v3.ordered_graph_parents(selected, valid_locations=valid_locations, strict_prefix=True)
    recall_parents, recall_unmapped = v3.ordered_graph_parents(recall, valid_locations=valid_locations)
    final_parents, final_unmapped = v3.ordered_graph_parents(final, valid_locations=valid_locations)
    if selected_unmapped or final_unmapped:
        raise RuntimeError(f"{qid}: fast graph events cannot map to persisted chunks")
    graph_parents = list(dict.fromkeys([*selected_parents, *recall_parents]))[: int(_arg(args, "graph_k", 24))]
    order = v3.expand_parent_locations(graph_parents, parents)
    selected_indexes = set(v3.expand_parent_locations(selected_parents, parents))
    final_indexes = set(v3.expand_parent_locations(final_parents, parents))
    fast_runtime = []
    graph_rank: dict[int, int] = {}
    for rank, index in enumerate(order):
        item = dict(fast[index])
        item["channels"] = {"dense_score": float(dense_scores[index]), "dense_rank_rr": v3.rrank(dense_rank[index]), "graph_rank_rr": v3.rrank(rank), "graph_selected": float(index in selected_indexes), "graph_final": float(index in final_indexes), "recency_norm": float(item["session_index"]) / max(1, max(int(candidate["session_index"]) for candidate in fast))}
        fast_runtime.append(item)
        graph_rank[index] = rank
    if not fast_runtime:
        raise RuntimeError(f"{qid}: graph fast path produced no candidates")
    cross_started = time.time()
    representations, logits = models.encode_cross(runtime_question, [item["text"] for item in fast_runtime])
    channels = v3.CHANNEL_NAMES
    channel_tensor = v3.torch.tensor([[item["channels"][name] for name in channels] for item in fast_runtime], dtype=v3.torch.float32, device=models.device)
    with v3.torch.inference_mode():
        scores = models.fusion(representations.unsqueeze(0), logits.unsqueeze(0), channel_tensor.unsqueeze(0), v3.torch.ones((1, len(fast_runtime)), dtype=v3.torch.bool, device=models.device), ablation="full")[0].detach().cpu()
    cross_elapsed = time.time() - cross_started
    for item, score, semantic in zip(fast_runtime, scores.tolist(), logits.detach().cpu().tolist()):
        item["score"], item["semantic_logit"] = float(score), float(semantic)
    ranked = sorted(fast_runtime, key=lambda item: (-float(item["score"]), str(item.get("candidate_id", ""))))
    ranked = _map_fast_candidates_with_slots(ranked, semantic_records)
    graph = {"skipped": False, "adapter_reused": reused, "retrieval_mode": metadata.get("retrieval_mode"), "hybrid_enabled": metadata.get("hybrid_enabled"), "selected_event_ids": selected, "recall_event_ids": recall, "final_hit_event_ids": final, "unmapped_recall_event_ids": recall_unmapped, "layer": "fast"}
    return ranked, graph, elapsed, cross_elapsed


def _source_local_path(*, runtime_question: str, fast: Sequence[Mapping[str, Any]], fast_vectors: Any, models: Any, args: argparse.Namespace) -> tuple[list[dict[str, Any]], Any, dict[int, int], float, float]:
    """BGE dense/cross retrieval over immutable source windows."""
    v3 = _v3()
    started = time.time()
    dense_scores = fast_vectors @ models.dense.encode_one(runtime_question)
    dense_order = sorted(range(len(fast)), key=lambda index: (-float(dense_scores[index]), index))
    dense_rank = {index: rank for rank, index in enumerate(dense_order)}
    indexes = []
    selected_parents: set[tuple[int, int]] = set()
    for index in dense_order:
        parent = (
            int(fast[index]["session_index"]),
            int(fast[index]["parent_chunk_index"]),
        )
        if parent in selected_parents:
            continue
        selected_parents.add(parent)
        indexes.append(index)
        if len(indexes) >= int(_arg(args, "dense_k", 32)):
            break
    if not indexes:
        raise RuntimeError("source local path produced no candidates")
    runtime = [dict(fast[index]) for index in indexes]
    cross_started = time.time()
    dense_elapsed = cross_started - started
    representations, logits = models.encode_cross(runtime_question, [item["text"] for item in runtime])
    cross_elapsed = time.time() - cross_started
    for item, semantic in zip(runtime, logits.detach().cpu().tolist()):
        item["semantic_logit"] = float(semantic)
    ranked = sorted(runtime, key=lambda item: (-float(item["semantic_logit"]), str(item.get("candidate_id", ""))))
    ranked = _collapse_source_parents(ranked, fast)
    for rank, item in enumerate(ranked, start=1):
        item["source_rank"] = rank - 1
        item["source_path"] = "bge_dense_cross"
        item["score"] = item["semantic_logit"]
    return ranked, dense_scores, dense_rank, dense_elapsed, cross_elapsed


def _collapse_source_parents(
    ranked_representatives: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Turn ranked subchunks into lossless parent evidence units."""
    by_parent: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for candidate in inventory:
        key = (
            int(candidate["session_index"]),
            int(candidate["parent_chunk_index"]),
        )
        by_parent.setdefault(key, []).append(candidate)
    output: list[dict[str, Any]] = []
    source_parent_payloads: dict[str, tuple[str, str]] = {}
    for representative in ranked_representatives:
        key = (
            int(representative["session_index"]),
            int(representative["parent_chunk_index"]),
        )
        members = sorted(
            by_parent.get(key) or [],
            key=lambda item: (
                int(item["source_char_start"]),
                int(item["source_char_end"]),
                int(item["subchunk_index"]),
            ),
        )
        if not members:
            raise RuntimeError(f"source parent has no inventory members: {key}")
        temporal_identity = (
            _clean(members[0].get("historical_date")),
            _clean(members[0].get("timestamp")),
            _clean(members[0].get("message_role") or members[0].get("role")),
        )
        if any(
            (
                _clean(member.get("historical_date")),
                _clean(member.get("timestamp")),
                _clean(member.get("message_role") or member.get("role")),
            )
            != temporal_identity
            for member in members[1:]
        ):
            raise RuntimeError(f"source parent temporal metadata is inconsistent: {key}")
        prefix = str(members[0]["text"]).split("\n", 1)[0]
        assembled = ""
        assembled_end = 0
        for member in members:
            start = int(member["source_char_start"])
            end = int(member["source_char_end"])
            member_prefix, separator, payload = str(member["text"]).partition("\n")
            if not separator or member_prefix != prefix or end - start != len(payload):
                raise RuntimeError(f"source parent subchunk metadata is inconsistent: {key}")
            if start > assembled_end:
                raise RuntimeError(f"source parent subchunks contain a gap: {key}")
            overlap = max(0, assembled_end - start)
            assembled += payload[overlap:]
            assembled_end = max(assembled_end, end)
        candidate_id = f"parent::{representative['session_id']}:{key[1]}"
        payload_identity = (assembled, temporal_identity[2])
        previous_payload = source_parent_payloads.get(candidate_id)
        if previous_payload is not None:
            if previous_payload != payload_identity:
                raise RuntimeError(
                    f"source parent candidate ID collision has different content: {candidate_id}"
                )
            continue
        source_parent_payloads[candidate_id] = payload_identity
        parent = dict(representative)
        parent.update(
            {
                "candidate_id": candidate_id,
                "text": assembled,
                "subchunk_index": 0,
                "source_char_start": 0,
                "source_char_end": assembled_end,
                "evidence_unit_kind": "source_parent",
                "member_candidate_ids": [str(item["candidate_id"]) for item in members],
                "member_subchunk_indexes": [int(item["subchunk_index"]) for item in members],
                "historical_date": temporal_identity[0],
                "timestamp": temporal_identity[1],
                "message_role": temporal_identity[2],
            }
        )
        output.append(parent)
    return output


def load_v4_layered_inventory(
    db_path: Path,
    scope_id: str,
    parents: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a typed Slow summary/claim inventory above V3's source mapping."""
    v3 = _v3()
    raw_claim_candidates, semantic_records = v3.load_layered_inventory(
        db_path, scope_id, parents
    )
    if not raw_claim_candidates:
        return [], semantic_records

    values_by_memory_id: dict[str, str] = {}
    record_metadata_by_memory_id: dict[str, dict[str, Any]] = {}
    patch_metadata_by_id: dict[str, dict[str, Any]] = {}
    con = sqlite3.connect(db_path)
    try:
        record_columns = {
            str(row[1])
            for row in con.execute("PRAGMA table_info(records)").fetchall()
        }
        columns = "memory_id,value"
        if "metadata_json" in record_columns:
            columns += ",metadata_json"
        for row in con.execute(
            f"SELECT {columns} FROM records WHERE scope_id=?", (scope_id,)
        ).fetchall():
            memory_id, value = row[:2]
            values_by_memory_id[str(memory_id)] = str(value)
            if len(row) > 2 and row[2]:
                try:
                    metadata = json.loads(row[2])
                except (TypeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"{memory_id}: Slow record metadata is not valid JSON"
                    ) from exc
                if not isinstance(metadata, dict):
                    raise RuntimeError(f"{memory_id}: Slow record metadata is not an object")
                record_metadata_by_memory_id[str(memory_id)] = metadata
        patch_table = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='slow_graph_patches'"
        ).fetchone()
        if patch_table is not None:
            for patch_id, raw_metadata in con.execute(
                "SELECT patch_id,call_metadata_json FROM slow_graph_patches"
            ).fetchall():
                try:
                    metadata = json.loads(raw_metadata)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"{patch_id}: Slow patch metadata is not valid JSON"
                    ) from exc
                if not isinstance(metadata, dict):
                    raise RuntimeError(f"{patch_id}: Slow patch metadata is not an object")
                patch_metadata_by_id[str(patch_id)] = metadata
    finally:
        con.close()

    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in raw_claim_candidates:
        candidate = dict(raw)
        memory_id = _clean(candidate.get("memory_id"))
        if not memory_id:
            raise RuntimeError("Slow claim candidate lacks memory_id")
        grouped.setdefault(memory_id, []).append(candidate)

    summary_candidates: list[dict[str, Any]] = []
    claim_candidates: list[dict[str, Any]] = []
    for memory_id in sorted(grouped):
        members = list(grouped[memory_id])
        capsule_ids = {_clean(item.get("capsule_id")) for item in members}
        revisions = {item.get("revision") for item in members}
        if len(capsule_ids) != 1 or "" in capsule_ids or len(revisions) != 1:
            raise RuntimeError(f"{memory_id}: Slow claim identity is inconsistent")
        capsule_id = next(iter(capsule_ids))
        revision = next(iter(revisions))
        if not isinstance(revision, int) or revision < 1:
            raise RuntimeError(f"{memory_id}: Slow revision is invalid")
        summary = values_by_memory_id.get(memory_id, "")
        record_metadata = record_metadata_by_memory_id.get(memory_id, {})
        patch_metadata = patch_metadata_by_id.get(
            _clean(record_metadata.get("patch_id")), {}
        )
        current_summary_contract = _current_summary_contract(
            record_metadata,
            patch_metadata,
            *(member.get("provenance") for member in members),
        )
        claims: list[dict[str, Any]] = []
        all_parents: list[dict[str, Any]] = []
        seen_parents: set[tuple[Any, ...]] = set()
        for member in members:
            member_claims = list(member.get("claims") or [])
            if len(member_claims) != 1 or not isinstance(member_claims[0], Mapping):
                raise RuntimeError(
                    f"{member.get('candidate_id')}: Slow claim candidate is not atomic"
                )
            claims.append(dict(member_claims[0]))
            for parent in list(member.get("source_parents") or []):
                if not isinstance(parent, Mapping):
                    raise RuntimeError(
                        f"{member.get('candidate_id')}: Slow source parent is invalid"
                    )
                identity = (
                    parent.get("session_index"),
                    parent.get("parent_chunk_index"),
                    parent.get("source_record_id"),
                    parent.get("evidence_char_start"),
                    parent.get("evidence_char_end"),
                )
                if identity not in seen_parents:
                    seen_parents.add(identity)
                    all_parents.append(dict(parent))
        try:
            summary = validate_semantic_summary(
                summary,
                claims,
                label=f"{memory_id} summary",
            )
        except PatchValidationError as exc:
            raise RuntimeError(
                f"{memory_id}: Slow summary violates the V4 inventory contract: {exc}"
            ) from exc
        if current_summary_contract:
            expected_summary = _lossless_summary_projection(claims)
            if summary != expected_summary:
                raise RuntimeError(
                    f"{memory_id}: current V4.7 Slow summary is not the exact lossless "
                    f"claim projection; expected {expected_summary!r}"
                )

        summary_candidate_id = f"capsule-summary::{capsule_id}:r{revision}"
        child_ids = [_clean(item.get("candidate_id")) for item in members]
        if any(not item for item in child_ids) or len(set(child_ids)) != len(child_ids):
            raise RuntimeError(f"{memory_id}: Slow claim candidate IDs are invalid")
        slots = sorted({_clean(item.get("canonical_slot")) for item in members})
        summary_candidate = {
                "inventory_schema_version": SLOW_INVENTORY_SCHEMA_VERSION,
                "candidate_kind": "capsule_summary",
                "candidate_id": summary_candidate_id,
                "memory_id": memory_id,
                "capsule_id": capsule_id,
                "revision": revision,
                "status": _clean(members[0].get("status")),
                "canonical_slot": slots[0] if len(slots) == 1 else "slow.summary",
                "claims": claims,
                "source_parents": all_parents,
                "text": summary,
                "child_claim_candidate_ids": child_ids,
                "provenance": {
                    "memory_layer": "slow",
                    "content_variant": "slow_memory_capsule",
                    "capsule_id": capsule_id,
                    "revision": revision,
                    "candidate_kind": "capsule_summary",
                    "source_parents": all_parents,
                },
            }
        if current_summary_contract:
            summary_candidate["summary_contract_version"] = SLOW_SUMMARY_CONTRACT_VERSION
            summary_candidate["provenance"]["summary_contract_version"] = (
                SLOW_SUMMARY_CONTRACT_VERSION
            )
        partition_contract_version = _clean(
            record_metadata.get("partition_contract_version")
        )
        if partition_contract_version:
            summary_candidate["partition_contract_version"] = (
                partition_contract_version
            )
            summary_candidate["provenance"]["partition_contract_version"] = (
                partition_contract_version
            )
        region_key = _clean(record_metadata.get("region_key"))
        if not region_key:
            region_key = _clean(members[0].get("region_key"))
        if not region_key:
            region_key = _clean(
                (members[0].get("provenance") or {}).get("region_key")
            )
        if region_key:
            summary_candidate["region_key"] = region_key
            summary_candidate["provenance"]["region_key"] = region_key
        summary_candidates.append(summary_candidate)
        for member in members:
            claim = {
                **member,
                "inventory_schema_version": SLOW_INVENTORY_SCHEMA_VERSION,
                "candidate_kind": "capsule_claim",
                "capsule_summary_candidate_id": summary_candidate_id,
                "capsule_summary_text": summary,
            }
            claim["provenance"] = {
                **dict(member.get("provenance") or {}),
                "candidate_kind": "capsule_claim",
                "capsule_summary_candidate_id": summary_candidate_id,
            }
            if current_summary_contract:
                claim["summary_contract_version"] = SLOW_SUMMARY_CONTRACT_VERSION
                claim["provenance"]["summary_contract_version"] = (
                    SLOW_SUMMARY_CONTRACT_VERSION
                )
            if region_key:
                claim["region_key"] = region_key
            claim_candidates.append(claim)

    inventory = sorted(
        [*summary_candidates, *claim_candidates],
        key=lambda item: (
            0 if item["candidate_kind"] == "capsule_summary" else 1,
            str(item["candidate_id"]),
        ),
    )
    candidate_ids = [str(item["candidate_id"]) for item in inventory]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("V4 Slow inventory candidate IDs are not unique")
    _validate_active_capsule_partition(inventory)
    return inventory, semantic_records


def _slow_local_path(*, runtime_question: str, slow: Sequence[Mapping[str, Any]], slow_vectors: Any, models: Any, args: argparse.Namespace) -> tuple[list[dict[str, Any]], float, float]:
    started = time.time()
    if len(slow) != len(slow_vectors):
        raise RuntimeError("V4 Slow inventory/vector count mismatch")
    if any(
        item.get("inventory_schema_version") != SLOW_INVENTORY_SCHEMA_VERSION
        for item in slow
    ):
        raise RuntimeError("V4 Slow inventory schema mismatch")
    scores = slow_vectors @ models.dense.encode_one(runtime_question)
    summary_indexes = [
        index
        for index, item in enumerate(slow)
        if item.get("candidate_kind") == "capsule_summary"
    ]
    claim_indexes = [
        index
        for index, item in enumerate(slow)
        if item.get("candidate_kind") == "capsule_claim"
    ]
    if not summary_indexes or not claim_indexes:
        raise RuntimeError(
            "V4 Slow inventory requires both summary and claim candidates"
        )
    k = int(_arg(args, "slow_dense_k", 24))
    summary_order = sorted(
        summary_indexes, key=lambda index: (-float(scores[index]), index)
    )
    claim_order = sorted(
        claim_indexes, key=lambda index: (-float(scores[index]), index)
    )
    selected_summaries = summary_order[: min(len(summary_order), k)]
    selected_direct_claims = claim_order[: min(len(claim_order), k)]
    claim_index_by_id = {
        _clean(slow[index].get("candidate_id")): index for index in claim_indexes
    }
    summary_hit_by_claim: dict[str, dict[str, Any]] = {}
    selected_claim_indexes = set(selected_direct_claims)
    for summary_rank, summary_index in enumerate(selected_summaries):
        summary = slow[summary_index]
        summary_id = _clean(summary.get("candidate_id"))
        children = list(summary.get("child_claim_candidate_ids") or [])
        if not summary_id or not children:
            raise RuntimeError("Slow summary candidate has no claim expansion mapping")
        for raw_child_id in children:
            child_id = _clean(raw_child_id)
            child_index = claim_index_by_id.get(child_id)
            if child_index is None:
                raise RuntimeError(
                    f"Slow summary expansion references an unknown claim: {child_id}"
                )
            child = slow[child_index]
            if (
                _clean(child.get("capsule_summary_candidate_id")) != summary_id
                or _clean(child.get("capsule_id"))
                != _clean(summary.get("capsule_id"))
                or child.get("revision") != summary.get("revision")
            ):
                raise RuntimeError(
                    f"Slow summary/claim expansion identity mismatch: {child_id}"
                )
            selected_claim_indexes.add(child_index)
            previous = summary_hit_by_claim.get(child_id)
            hit = {
                "summary_candidate_id": summary_id,
                "summary_text": _clean(summary.get("text")),
                "summary_dense_score": float(scores[summary_index]),
                "summary_dense_rank": summary_rank,
            }
            if previous is not None and previous != hit:
                raise RuntimeError(
                    f"Slow claim is attached to multiple summary candidates: {child_id}"
                )
            summary_hit_by_claim[child_id] = hit

    direct_rank_by_index = {
        index: rank for rank, index in enumerate(claim_order)
    }
    ranked: list[dict[str, Any]] = []
    for index in sorted(
        selected_claim_indexes,
        key=lambda item: (direct_rank_by_index[item], item),
    ):
        candidate = dict(slow[index])
        candidate_id = _clean(candidate.get("candidate_id"))
        summary_hit = summary_hit_by_claim.get(candidate_id)
        is_direct = index in selected_direct_claims
        if not is_direct and summary_hit is None:
            raise RuntimeError("Slow claim entered the shortlist without a retrieval route")
        candidate.update(
            {
                "slow_dense_score": float(scores[index]),
                "slow_dense_rank": direct_rank_by_index[index],
                "direct_claim_hit": is_direct,
                "summary_expansion_hit": summary_hit is not None,
                "summary_hit": dict(summary_hit) if summary_hit is not None else None,
                "selection_routes": [
                    route
                    for route, enabled in (
                        ("direct_claim", is_direct),
                        ("capsule_summary", summary_hit is not None),
                    )
                    if enabled
                ],
            }
        )
        ranked.append(candidate)
    if not ranked:
        raise RuntimeError("slow local path produced an empty claim shortlist")
    cross_started = time.time()
    dense_elapsed = cross_started - started
    cross_texts = [
        "User memory summary: "
        + _clean(item.get("capsule_summary_text"))
        + "\nSpecific supported memory: "
        + _clean(item.get("text"))
        for item in ranked
    ]
    if any(
        not _clean(item.get("capsule_summary_text")) or not _clean(item.get("text"))
        for item in ranked
    ):
        raise RuntimeError("Slow claim shortlist lacks summary or claim text")
    _, logits = models.encode_cross(runtime_question, cross_texts)
    cross_elapsed = time.time() - cross_started
    for item, score in zip(ranked, logits.detach().cpu().tolist()):
        item["semantic_logit"] = float(score)
        item["slow_retrieval_trace"] = {
            "inventory_schema_version": SLOW_INVENTORY_SCHEMA_VERSION,
            "direct_claim_hit": bool(item["direct_claim_hit"]),
            "claim_candidate_id": _clean(item.get("candidate_id")),
            "claim_dense_rank": int(item["slow_dense_rank"]),
            "claim_dense_score": float(item["slow_dense_score"]),
            "summary_expansion_hit": bool(item["summary_expansion_hit"]),
            "summary_hit": item.get("summary_hit"),
            "final_claim_cross_score": float(score),
            "source_parents": [
                dict(parent) for parent in list(item.get("source_parents") or [])
            ],
        }
    ranked.sort(key=lambda item: (-float(item["semantic_logit"]), str(item.get("candidate_id", ""))))
    return ranked, dense_elapsed, cross_elapsed


def _unit_windows(unit: Mapping[str, Any], source_candidates: Sequence[Mapping[str, Any]], *, qid: str) -> list[dict[str, Any]]:
    by_parent: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for candidate in source_candidates:
        by_parent.setdefault((int(candidate["session_index"]), int(candidate["parent_chunk_index"])), []).append(candidate)
    output: dict[tuple[int, int, int], dict[str, Any]] = {}

    def add(candidate: Mapping[str, Any], role: str, capsule: Mapping[str, Any] | None = None) -> None:
        key = (int(candidate["session_index"]), int(candidate["parent_chunk_index"]), int(candidate["subchunk_index"]))
        item = output.setdefault(key, {**dict(candidate), "roles": [], "capsules": []})
        if role not in item["roles"]:
            item["roles"].append(role)
        if capsule is not None and capsule not in item["capsules"]:
            item["capsules"].append(dict(capsule))

    if unit["unit_type"] == "source_window":
        add(unit["source_candidate"], "source")
    elif unit["unit_type"] == "fast_atomic":
        fast_candidate = unit["fast_candidate"]
        location = (int(fast_candidate["session_index"]), int(fast_candidate["parent_chunk_index"]))
        matches = list(by_parent.get(location) or [])
        if not matches:
            raise RuntimeError(f"{qid}: fast candidate is not mapped to an immutable source window: {location}")
        for candidate in matches:
            add(candidate, "fast")
    elif unit["unit_type"] == "slow_capsule":
        capsule = unit["slow_candidate"]
        for parent in capsule["source_parents"]:
            location = (int(parent["session_index"]), int(parent["parent_chunk_index"]))
            matches = [candidate for candidate in by_parent.get(location, []) if int(candidate["source_char_end"]) > int(parent["evidence_char_start"]) and int(candidate["source_char_start"]) < int(parent["evidence_char_end"])]
            if not matches:
                raise RuntimeError(f"{qid}: slow source_parent is unmapped during descent: {location}")
            for candidate in matches:
                add(candidate, "slow", capsule)
    else:
        raise RuntimeError(f"{qid}: unsupported V4 recall unit {unit['unit_type']}")
    return list(output.values())


def pack_recall_role_units(
    units: Sequence[Mapping[str, Any]],
    source_candidates: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    qid: str,
    return_stats: bool = False,
    required_layers: Sequence[str] = (),
    required_source_session_count: int = 1,
) -> Any:
    if top_k <= 0:
        raise RuntimeError("top_k must be positive")
    if required_source_session_count <= 0:
        raise RuntimeError("required_source_session_count must be positive")
    packed: list[tuple[Mapping[str, Any], list[dict[str, Any]]]] = []
    used: set[tuple[int, int]] = set()
    source_sessions: set[str] = set()
    budget_excluded = 0
    duplicate_units = 0
    ordered_units = sorted(units, key=lambda unit: (-float(unit.get("priority_score", 0.0)), str(unit.get("layer", "")), str(unit.get("canonical_slot", ""))))
    required = list(dict.fromkeys(str(layer) for layer in required_layers))
    if any(layer not in {"source", "fast", "slow"} for layer in required):
        raise RuntimeError(f"{qid}: required recall layer is invalid")
    selected_unit_ids: set[int] = set()

    def track_source_sessions(windows: Sequence[Mapping[str, Any]]) -> None:
        for item in windows:
            session_id = _clean(item.get("session_id"))
            if session_id:
                source_sessions.add(session_id)

    def add_unit(
        unit: Mapping[str, Any], *, required_layer: str | None = None
    ) -> bool:
        nonlocal budget_excluded, duplicate_units
        windows = _unit_windows(unit, source_candidates, qid=qid)
        unique = [item for item in windows if (int(item["session_index"]), int(item["parent_chunk_index"])) not in used]
        if not unique:
            duplicate_units += 1
            # The physical window is already budgeted, but this layer still
            # contributes ranking/provenance information to that window.
            packed.append((unit, windows))
            selected_unit_ids.add(id(unit))
            if unit.get("layer") == "source":
                track_source_sessions(windows)
            return True
        if len(unique) > top_k and not packed:
            raise RuntimeError(f"{qid}: first atomic recall unit exceeds strict packing budget {top_k}")
        if len(used) + len(unique) > top_k:
            if required_layer is not None:
                raise RuntimeError(
                    f"{qid}: required {required_layer} unit exceeds strict packing budget {top_k}"
                )
            budget_excluded += 1
            return False
        packed.append((unit, windows))
        selected_unit_ids.add(id(unit))
        used.update((int(item["session_index"]), int(item["parent_chunk_index"])) for item in unique)
        if unit.get("layer") == "source":
            track_source_sessions(windows)
        return True

    for layer in required:
        candidates = [unit for unit in ordered_units if unit.get("layer") == layer]
        if layer == "fast":
            semantic = [
                unit
                for unit in candidates
                if list((unit.get("fast_candidate") or {}).get("semantic_record_ids") or [])
            ]
            candidates = semantic or candidates
        if not candidates:
            raise RuntimeError(f"{qid}: required {layer} layer has no candidate unit")
        add_unit(candidates[0], required_layer=layer)

    if "source" in required and required_source_session_count > 1:
        source_units = [unit for unit in ordered_units if unit.get("layer") == "source"]
        for unit in source_units:
            candidate = unit.get("source_candidate")
            session_id = (
                _clean(candidate.get("session_id"))
                if isinstance(candidate, Mapping)
                else ""
            )
            if not session_id or session_id in source_sessions:
                continue
            if len(used) >= top_k:
                break
            add_unit(unit)
            if len(source_sessions) >= required_source_session_count:
                break

    for unit in ordered_units:
        if id(unit) in selected_unit_ids:
            continue
        add_unit(unit)
    if not packed:
        raise RuntimeError(f"{qid}: recall role plan produced no packable evidence units")
    if return_stats:
        return packed, {
            "budget_excluded_unit_count": budget_excluded,
            "duplicate_unit_count": duplicate_units,
            "required_layers": required,
            "source_session_diversity_target": required_source_session_count,
            "source_session_diversity_selected": len(source_sessions),
        }
    return packed


def required_source_session_count(plan: Mapping[str, Any]) -> int:
    """Reserve comparison coverage without inspecting benchmark labels."""
    query_kind = _clean(plan.get("query_kind"))
    temporal_focus = _clean(plan.get("temporal_focus"))
    if temporal_focus in {"historical", "recent", "mixed"} and query_kind in {
        "comparison",
        "event",
        "historical",
    }:
        return 3
    return 1


def _unit_base_score(unit: Mapping[str, Any]) -> float:
    """Return the within-layer reciprocal rank, never a raw model score."""
    return float(unit.get("within_layer_score", 1.0 / max(1, int(unit.get("layer_rank", 1)))))


def _unit_contribution(unit: Mapping[str, Any]) -> dict[str, Any]:
    fast_candidate = unit.get("fast_candidate")
    active_semantic = bool(
        unit.get("layer") == "fast"
        and isinstance(fast_candidate, Mapping)
        and list(fast_candidate.get("semantic_record_ids") or [])
    )
    return {
        "layer": unit["layer"],
        "role": unit["layer_role"],
        "weight": unit["layer_weight"],
        "normalized_priority": unit["normalized_priority"],
        "within_layer_score": unit["within_layer_score"],
        "priority_score": unit["priority_score"],
        "active_semantic": active_semantic,
    }


def _fast_memory_attachments(unit: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate = unit.get("fast_candidate")
    if unit.get("layer") != "fast" or not isinstance(candidate, Mapping):
        return []
    attachments: list[dict[str, Any]] = []
    for memory in list(candidate.get("semantic_memories") or []):
        if not isinstance(memory, Mapping):
            raise RuntimeError("fast semantic memory attachment is not an object")
        memory_id = _clean(memory.get("memory_id"))
        slot = _clean(memory.get("canonical_slot"))
        text = _clean(memory.get("text"))
        provenance = memory.get("provenance")
        source_parent = memory.get("source_parent")
        if (
            not memory_id
            or not slot
            or not text
            or not isinstance(provenance, Mapping)
            or not isinstance(source_parent, Mapping)
        ):
            raise RuntimeError("fast semantic memory attachment is incomplete")
        attachments.append(
            {
                "role": "fast_context",
                "memory_id": memory_id,
                "canonical_slot": slot,
                "text": text,
                "record_state": _clean(memory.get("record_state")),
                "memory_type": _clean(memory.get("memory_type")),
                "durability": _clean(memory.get("durability")),
                "temporal_status": _clean(memory.get("temporal_status")),
                "source_parent": dict(source_parent),
                "provenance": dict(provenance),
            }
        )
    return attachments


def _slow_memory_contexts(capsules: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for capsule in capsules:
        claims = list(capsule.get("claims") or [])
        if len(claims) != 1 or not isinstance(claims[0], Mapping):
            raise RuntimeError("slow retrieval candidate must contain exactly one claim")
        claim = claims[0]
        provenance = capsule.get("provenance")
        context = {
            "role": "slow_context",
            "capsule_id": _clean(capsule.get("capsule_id")),
            "revision": int(capsule.get("revision", 0)),
            "status": _clean(capsule.get("status")),
            "claim_id": _clean(claim.get("claim_id")),
            "canonical_slot": _clean(capsule.get("canonical_slot")),
            "capsule_summary": _clean(capsule.get("capsule_summary_text")),
            "capsule_summary_candidate_id": _clean(
                capsule.get("capsule_summary_candidate_id")
            ),
            "claim_text": _clean(capsule.get("text")),
            "support": list(claim.get("support") or []),
            "counterevidence": list(claim.get("counterevidence") or []),
            "source_parents": [dict(item) for item in list(capsule.get("source_parents") or [])],
            "provenance": dict(provenance) if isinstance(provenance, Mapping) else {},
            "retrieval_trace": dict(capsule.get("slow_retrieval_trace") or {}),
        }
        if (
            not context["capsule_id"]
            or context["revision"] < 1
            or not context["claim_id"]
            or not context["canonical_slot"]
            or not context["capsule_summary"]
            or not context["capsule_summary_candidate_id"]
            or not context["claim_text"]
            or not context["source_parents"]
            or not context["provenance"]
            or not context["retrieval_trace"]
        ):
            raise RuntimeError("slow retrieval claim context is incomplete")
        contexts.append(context)
    return contexts


def _mark_newer_fast_overrides(windows: Sequence[dict[str, Any]]) -> None:
    slow_by_slot: dict[str, dict[str, Any]] = {}
    for window in windows:
        for context in list(window.get("memory_contexts") or []):
            if not isinstance(context, Mapping):
                continue
            slot = _clean(context.get("canonical_slot"))
            if not slot:
                continue
            parents = [
                parent
                for parent in list(context.get("source_parents") or [])
                if isinstance(parent, Mapping)
            ]
            latest = max(
                (
                    int(parent.get("session_index", -1)),
                    int(parent.get("parent_chunk_index", parent.get("message_index", -1))),
                )
                for parent in parents
            )
            state = slow_by_slot.setdefault(slot, {"latest": latest, "support": set()})
            state["latest"] = max(state["latest"], latest)
            state["support"].update(_clean(value) for value in context.get("support") or [])
    for window in windows:
        for attachment in list(window.get("attachments") or []):
            if not isinstance(attachment, dict):
                continue
            slot = _clean(attachment.get("canonical_slot"))
            slow_state = slow_by_slot.get(slot)
            parent = attachment.get("source_parent")
            if not slow_state or not isinstance(parent, Mapping):
                continue
            location = (
                int(parent.get("session_index", -1)),
                int(parent.get("parent_chunk_index", parent.get("message_index", -1))),
            )
            if (
                location > slow_state["latest"]
                and _clean(attachment.get("memory_id")) not in slow_state["support"]
            ):
                attachment["role"] = "override"
                attachment["precedence"] = "newer_fast_evidence"


def _selected_layer_window_counts(
    windows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = {"source": 0, "fast": 0, "slow": 0}
    for window in windows:
        metadata = window.get("retrieval_metadata")
        contributions = (
            metadata.get("layer_contributions") if isinstance(metadata, Mapping) else []
        )
        layers = {
            _clean(item.get("layer"))
            for item in contributions
            if isinstance(item, Mapping)
        }
        for layer in layers:
            if layer in counts:
                counts[layer] += 1
    return counts


def _order_evidence_windows(
    evidence_by_location: Mapping[tuple[int, int, int], dict[str, Any]],
    *,
    conflict_policy: str,
) -> list[dict[str, Any]]:
    windows = list(evidence_by_location.values())
    for original_order, item in enumerate(windows):
        contributions = list(
            item["retrieval_metadata"].get("layer_contributions") or []
        )
        active_fast = [
            value
            for value in contributions
            if value.get("layer") == "fast" and value.get("active_semantic") is True
        ]
        item["retrieval_metadata"]["active_fast_support"] = bool(active_fast)
        item["retrieval_metadata"]["fast_within_layer_score"] = max(
            (float(value["within_layer_score"]) for value in active_fast),
            default=0.0,
        )
        item["retrieval_metadata"]["packing_priority_score"] = max(
            (float(value["priority_score"]) for value in contributions),
            default=0.0,
        )
        item["_original_order"] = original_order
    if conflict_policy == "prefer_recent":
        windows.sort(
            key=lambda item: (
                0 if item["retrieval_metadata"]["active_fast_support"] else 1,
                -int(item["session_index"])
                if item["retrieval_metadata"]["active_fast_support"]
                else 0,
                -int(item["parent_chunk_index"])
                if item["retrieval_metadata"]["active_fast_support"]
                else 0,
                -float(item["retrieval_metadata"]["fast_within_layer_score"]),
                int(item["_original_order"]),
            )
        )

    # Retrieval ranks chunks, while the answer layer consumes dialogue. Group
    # the already-selected chunks at the immutable session occurrence boundary,
    # then restore source order within each session. Membership and Top-K do not
    # change, so this cannot manufacture or hide evidence.
    sessions: dict[int, dict[str, Any]] = {}
    for pre_session_rank, item in enumerate(windows, start=1):
        session_index = int(item["session_index"])
        state = sessions.setdefault(
            session_index,
            {
                "rrf": 0.0,
                "count": 0,
                "best_rank": pre_session_rank,
                "active_fast": False,
            },
        )
        state["rrf"] += 1.0 / float(pre_session_rank)
        state["count"] += 1
        state["best_rank"] = min(int(state["best_rank"]), pre_session_rank)
        state["active_fast"] = bool(
            state["active_fast"]
            or item["retrieval_metadata"]["active_fast_support"]
        )
        item["retrieval_metadata"]["pre_session_order_rank"] = pre_session_rank

    def session_key(session_index: int) -> tuple[Any, ...]:
        state = sessions[session_index]
        recent_prefix: tuple[Any, ...] = ()
        if conflict_policy == "prefer_recent":
            recent_prefix = (
                0 if state["active_fast"] else 1,
                -session_index if state["active_fast"] else 0,
            )
        return (
            *recent_prefix,
            -float(state["rrf"]),
            int(state["best_rank"]),
            session_index,
        )

    session_order = sorted(sessions, key=session_key)
    session_rank = {
        session_index: rank for rank, session_index in enumerate(session_order, start=1)
    }
    windows.sort(
        key=lambda item: (
            session_rank[int(item["session_index"])],
            int(item["parent_chunk_index"]),
            int(item.get("subchunk_index", 0)),
            int(item["retrieval_metadata"]["pre_session_order_rank"]),
        )
    )
    for rank, item in enumerate(windows, start=1):
        state = sessions[int(item["session_index"])]
        item.pop("_original_order", None)
        item["rank"] = rank
        item["retrieval_metadata"].update(
            {
                "session_ordering_policy": SESSION_ORDERING_POLICY,
                "session_order_rank": session_rank[int(item["session_index"])],
                "session_support_rrf": round(float(state["rrf"]), 8),
                "session_selected_window_count": int(state["count"]),
            }
        )
    return windows


def _attach_source_group_context(
    evidence_windows: Sequence[Mapping[str, Any]],
    source_inventory: Sequence[Mapping[str, Any]],
    *,
    max_parent_distance: int = 2,
    max_context_members: int = 2,
    max_context_chars: int = 3600,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Attach nearby immutable source parents without changing Top-K membership.

    Retrieval ranks source parents independently, but a fact can depend on a
    nearby turn in the same session. Context parents are assigned once to the
    nearest selected parent and remain separately identifiable by provenance.
    """
    if max_parent_distance < 0 or max_context_members < 0 or max_context_chars < 0:
        raise RuntimeError("source group context limits must be non-negative")
    output: list[dict[str, Any]] = []
    for item in evidence_windows:
        current = dict(item)
        current["retrieval_metadata"] = dict(item.get("retrieval_metadata") or {})
        current["source_group_id"] = (
            f"source-group::{item['session_id']}:{int(item['parent_chunk_index'])}"
        )
        current["source_group_context"] = []
        output.append(current)

    selected_locations = {
        (int(item["session_index"]), int(item["parent_chunk_index"]))
        for item in output
    }
    selected_by_session: dict[int, list[tuple[int, int]]] = {}
    for output_index, item in enumerate(output):
        selected_by_session.setdefault(int(item["session_index"]), []).append(
            (int(item["parent_chunk_index"]), output_index)
        )

    candidates: list[tuple[int, int, int, Mapping[str, Any]]] = []
    seen_inventory: set[tuple[int, int]] = set()
    for item in source_inventory:
        location = (int(item["session_index"]), int(item["parent_chunk_index"]))
        if location in seen_inventory:
            continue
        seen_inventory.add(location)
        if location in selected_locations or location[0] not in selected_by_session:
            continue
        nearest = min(
            (
                abs(location[1] - selected_parent),
                output_index,
            )
            for selected_parent, output_index in selected_by_session[location[0]]
        )
        distance, output_index = nearest
        if distance <= max_parent_distance:
            candidates.append((distance, output_index, location[1], item))

    attached_chars = [0 for _ in output]
    attached_count = 0
    for distance, output_index, _, item in sorted(
        candidates,
        key=lambda value: (
            value[0],
            value[1],
            value[2],
            str(value[3].get("source_record_id") or ""),
        ),
    ):
        target = output[output_index]
        members = target["source_group_context"]
        text = str(item.get("text") or "")
        if not text or len(members) >= max_context_members:
            continue
        if attached_chars[output_index] + len(text) > max_context_chars:
            continue
        members.append(
            {
                "relationship": "session_neighbor",
                "parent_distance": distance,
                "session_id": str(item["session_id"]),
                "session_index": int(item["session_index"]),
                "parent_chunk_index": int(item["parent_chunk_index"]),
                "source_record_id": str(item.get("source_record_id") or ""),
                "source_char_start": int(item.get("source_char_start", 0)),
                "source_char_end": int(item.get("source_char_end", len(text))),
                "historical_date": _clean(item.get("historical_date")),
                "timestamp": _clean(item.get("timestamp")),
                "message_role": _clean(
                    item.get("message_role") or item.get("role")
                ),
                "text": text,
            }
        )
        attached_chars[output_index] += len(text)
        attached_count += 1

    groups_with_context = 0
    for item in output:
        count = len(item["source_group_context"])
        item["retrieval_metadata"]["source_group_context_count"] = count
        item["retrieval_metadata"]["source_group_context_chars"] = sum(
            len(member["text"]) for member in item["source_group_context"]
        )
        groups_with_context += int(count > 0)
    return output, {
        "source_group_count": len(output),
        "groups_with_context_count": groups_with_context,
        "attached_context_parent_count": attached_count,
    }


def retrieve_one(
    row: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    harness: Any,
    models: Any,
    planner: DeepSeekFlashRecallRolePlanner | None = None,
    route_override: Mapping[str, Any] | None = None,
    route_override_metadata: Mapping[str, Any] | None = None,
    planner_decision_callback: Callable[
        [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], None
    ]
    | None = None,
    graph_adapter_cache: dict[tuple[str, str], Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.time()
    qid, question, question_date = _clean(row.get("question_id")), _clean(row.get("question")), _clean(row.get("question_date"))
    if not qid or not question:
        raise RuntimeError("online retrieval manifest row lacks qid or question")
    if any(key in row for key in ("answer", "gold_answer", "answer_session_ids", "labels", "supervision", "benchmark", "expected_answer")):
        raise RuntimeError(f"{qid}: runtime retrieval manifest contains forbidden evaluation labels")
    (
        composition_mode,
        execution_lane,
        packing_budget_mode,
        configured_top_k,
    ) = _validated_runtime_route(args, label=qid)
    v3 = _v3()
    if int(_arg(args, "slow_dense_k", 24)) <= 0:
        raise RuntimeError("slow_dense_k must be positive")
    db_path, scope_id, index_path = Path(row["db_path"]).resolve(), _clean(row.get("scope_id")), Path(row["index_path"]).resolve()
    fast, fast_vectors, slow, slow_vectors, semantic_records, payload = load_online_index(
        index_path, db_path, scope_id
    )
    semantic_records = _hydrate_fast_semantic_records(
        db_path, scope_id, semantic_records
    )
    counts_before = v3.scope_counts(db_path, scope_id)
    if counts_before["records"] != int(payload["graph_counts_at_index"]["records"]):
        raise RuntimeError(f"{qid}: graph records changed after online index creation")
    graph_fingerprint = v3.scope_fingerprint(db_path, scope_id)
    if graph_fingerprint != _clean(payload.get("graph_fingerprint")):
        raise RuntimeError(f"{qid}: graph fingerprint changed after online index creation")
    raw_recent_dialogue = v3.load_recent_dialogue_context(
        db_path,
        scope_id,
        current_query=question,
        limit=RECENT_DIALOGUE_MAX_TURNS,
    )
    recent_dialogue, recent_dialogue_projection = project_recent_dialogue(
        raw_recent_dialogue
    )
    available_layers = {
        "source": {"available": bool(fast), "candidate_count": len(fast)},
        "fast": {"available": bool(fast), "candidate_count": len(fast)},
        "slow": {
            "available": bool(slow),
            "capsule_count": int(payload.get("slow_capsule_head_count", 0)),
            "summary_candidate_count": int(
                payload.get("slow_summary_candidate_count", 0)
            ),
            "claim_candidate_count": int(
                payload.get("slow_claim_candidate_count", 0)
            ),
        },
    }
    if route_override is None:
        if planner is None:
            raise RuntimeError("V4 recall planner is required")
        raw_plan, planner_metadata = planner.plan(
            query=question,
            question_date=question_date or "unknown",
            recent_dialogue=recent_dialogue,
            available_layers=available_layers,
        )
    else:
        raw_plan = route_override
        planner_metadata = dict(
            route_override_metadata
            or {
                "physical_api_call": False,
                "physical_api_calls": 0,
                "stage": "recall_planner",
                "status": "route_override",
                "planner_version": "route_override",
                "prompt_version": "route_override",
            }
        )
    plan = validate_recall_role_plan(raw_plan)
    if planner_decision_callback is not None:
        planner_decision_callback(
            plan,
            planner_metadata,
            {
                "graph_fingerprint": graph_fingerprint,
                "recent_dialogue_projection": recent_dialogue_projection,
                "available_layers": available_layers,
            },
        )
    runtime_question = f"{plan['resolved_query']}\nQuestion date: {question_date}" if question_date else plan["resolved_query"]

    # These three calls are independent of all role and weight values.
    if fast:
        source_candidates, dense_scores, dense_rank, source_elapsed, source_cross_elapsed = _source_local_path(runtime_question=runtime_question, fast=fast, fast_vectors=fast_vectors, models=models, args=args)
        parent_representatives = []
        seen_inventory_parents: set[tuple[int, int]] = set()
        for candidate in fast:
            parent_key = (
                int(candidate["session_index"]),
                int(candidate["parent_chunk_index"]),
            )
            if parent_key not in seen_inventory_parents:
                seen_inventory_parents.add(parent_key)
                parent_representatives.append(candidate)
        source_evidence_inventory = _collapse_source_parents(
            parent_representatives, fast
        )
        fast_ranked, graph_metadata, graph_elapsed, fast_cross_elapsed = _graph_fast_path(qid=qid, runtime_question=runtime_question, fast=fast, dense_scores=dense_scores, dense_rank=dense_rank, semantic_records=semantic_records, args=args, harness=harness, scope_id=scope_id, db_path=db_path, graph_adapter_cache=graph_adapter_cache, models=models)
        dense_elapsed = source_elapsed
    else:
        source_candidates, source_evidence_inventory, graph_metadata, fast_ranked, source_elapsed, dense_elapsed, source_cross_elapsed, graph_elapsed, fast_cross_elapsed = [], [], {"skipped": True}, [], 0.0, 0.0, 0.0, 0.0, 0.0
    if slow:
        slow_ranked, slow_elapsed, slow_cross_elapsed = _slow_local_path(runtime_question=runtime_question, slow=slow, slow_vectors=slow_vectors, models=models, args=args)
    else:
        slow_ranked, slow_elapsed, slow_cross_elapsed = [], 0.0, 0.0
    composition_plan = plan
    composition_fast = fast_ranked
    composition_slow = slow_ranked
    if composition_mode == "source-only-diagnostic":
        composition_plan = {
            **plan,
            "layers": {
                "source": {"role": "primary", "weight": 1.0},
                "fast": {"role": "context", "weight": 0.0},
                "slow": {"role": "context", "weight": 0.0},
            },
        }
        composition_fast = []
        composition_slow = []
    try:
        units = apply_recall_role_plan(
            composition_plan,
            source_candidates,
            composition_fast,
            composition_slow,
        )
    except RecallPlannerError as exc:
        raise RuntimeError(f"{qid}: invalid V4 recall role composition: {exc}") from exc
    units.sort(key=lambda unit: (-float(unit["priority_score"]), str(unit.get("layer", "")), str(unit.get("canonical_slot", ""))))
    packing_budget, packing_budget_decision = resolve_packing_budget(
        plan,
        mode=packing_budget_mode,
        fixed_k=configured_top_k,
        simple_k=int(_arg(args, "adaptive_simple_k", 8)),
        standard_k=int(_arg(args, "adaptive_standard_k", 12)),
        complex_k=int(_arg(args, "adaptive_complex_k", 16)),
    )
    fast_semantic_shortlist_count = sum(
        int(bool(list(candidate.get("semantic_memories") or [])))
        for candidate in fast_ranked
    )
    required_layers: list[str] = []
    if source_candidates:
        required_layers.append("source")
    if composition_mode == "layered":
        if fast_ranked:
            required_layers.append("fast")
        if slow_ranked:
            required_layers.append("slow")
    packed_units, packing_stats = pack_recall_role_units(
        units,
        source_evidence_inventory,
        top_k=packing_budget,
        qid=qid,
        return_stats=True,
        required_layers=required_layers,
        required_source_session_count=required_source_session_count(plan),
    )

    evidence_by_location: dict[tuple[int, int, int], dict[str, Any]] = {}
    for unit, entries in packed_units:
        contribution = _unit_contribution(unit)
        unit_attachments = _fast_memory_attachments(unit)
        for entry in entries:
            capsules = list(entry.get("capsules") or [])
            contexts = _slow_memory_contexts(capsules)
            location = (int(entry["session_index"]), int(entry["parent_chunk_index"]), int(entry["subchunk_index"]))
            fast_candidate = unit.get("fast_candidate")
            semantic_record_ids = (
                list(fast_candidate.get("semantic_record_ids") or [])
                if isinstance(fast_candidate, Mapping)
                else []
            )
            unit_provenance = (
                fast_candidate.get("provenance")
                if isinstance(fast_candidate, Mapping)
                else entry.get("provenance")
            )
            raw_provenance = (
                [item.get("provenance") for item in capsules]
                if capsules
                else [unit_provenance]
            )
            provenance = [dict(item) for item in raw_provenance if isinstance(item, Mapping)]
            candidate = {"memory_id": entry.get("candidate_id"), "source_record_id": entry.get("source_record_id"), "source_char_start": entry.get("source_char_start"), "source_char_end": entry.get("source_char_end"), "session_id": entry["session_id"], "session_index": entry["session_index"], "parent_chunk_index": entry["parent_chunk_index"], "subchunk_index": entry["subchunk_index"], "historical_date": _clean(entry.get("historical_date")), "timestamp": _clean(entry.get("timestamp")), "message_role": _clean(entry.get("message_role") or entry.get("role")), "rank": 0, "score": entry.get("score", entry.get("semantic_logit")), "semantic_logit": entry.get("semantic_logit"), "channels": entry.get("channels"), "text": entry["text"], "unit_type": unit["unit_type"], "unit_types": [unit["unit_type"]], "canonical_slot": unit["canonical_slot"], "canonical_slots": [unit["canonical_slot"]], "role": list(entry["roles"]), "provenance": provenance, "semantic_record_ids": semantic_record_ids, "memory_contexts": contexts, "attachments": [dict(item) for item in unit_attachments], "retrieval_metadata": {"plan_layer": unit["layer"], "plan_role": unit["layer_role"], "plan_weight": unit["layer_weight"], "role_prior": unit["role_prior"], "normalized_priority": unit["normalized_priority"], "within_layer_score": unit["within_layer_score"], "priority_score": unit["priority_score"], "layer_contributions": [contribution]}}
            candidate["scope_id"] = scope_id
            candidate["db_path"] = str(db_path)
            existing = evidence_by_location.get(location)
            if existing is None:
                evidence_by_location[location] = candidate
            else:
                for field in ("unit_types", "canonical_slots", "role", "provenance", "semantic_record_ids", "memory_contexts", "attachments"):
                    for item in candidate[field]:
                        if item not in existing[field]:
                            existing[field].append(item)
                if contribution not in existing["retrieval_metadata"]["layer_contributions"]:
                    existing["retrieval_metadata"]["layer_contributions"].append(contribution)
                existing["retrieval_metadata"][f"{unit['layer']}_weight"] = unit["layer_weight"]
    _mark_newer_fast_overrides(list(evidence_by_location.values()))
    evidence_windows = _order_evidence_windows(
        evidence_by_location,
        conflict_policy=plan["conflict_policy"],
    )
    evidence_windows, source_group_stats = _attach_source_group_context(
        evidence_windows,
        source_evidence_inventory,
    )
    selected_layer_window_counts = _selected_layer_window_counts(evidence_windows)
    candidate_paths_executed = {
        "source": bool(fast),
        "fast": bool(fast),
        "slow": bool(slow),
    }
    evidence = {
        "schema_version": v3.SCHEMA_VERSION,
        "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        "question_id": qid,
        "question": question,
        "question_date": question_date,
        "question_type": _clean(row.get("question_type")),
        "selected_session_ids": list(
            dict.fromkeys(str(item["session_id"]) for item in evidence_windows)
        ),
        "recall_plan": plan,
        "retrieval_contract": {
            "schema_version": RETRIEVAL_CONTRACT_SCHEMA,
            "execution_lane": execution_lane,
            "composition_mode": composition_mode,
            "inventory_counts": {
                "source": len(source_candidates),
                "fast": len(fast_ranked),
                "fast_semantic": fast_semantic_shortlist_count,
                "slow_capsule_heads": int(
                    payload.get("slow_capsule_head_count", 0)
                ),
                "slow_summaries": int(
                    payload.get("slow_summary_candidate_count", 0)
                ),
                "slow_claims": int(payload.get("slow_claim_candidate_count", 0)),
                "slow_ranked_claims": len(slow_ranked),
                "slow": len(slow_ranked),
            },
            "candidate_paths_executed": candidate_paths_executed,
            "required_selected_layers": required_layers,
            "selected_layer_window_counts": selected_layer_window_counts,
            "packing_budget_mode": packing_budget_decision["mode"],
            "packing_budget": packing_budget,
            "source_coverage_trace_k": SOURCE_COVERAGE_TRACE_K,
            "final_window_count": len(evidence_windows),
            "source_session_diversity_target": int(
                packing_stats["source_session_diversity_target"]
            ),
            "source_session_diversity_selected": int(
                packing_stats["source_session_diversity_selected"]
            ),
        },
        "evidence_windows": evidence_windows,
    }
    debug = {
        "question_id": qid,
        "scope_id": scope_id,
        "db_path": str(db_path),
        "index_path": str(index_path),
        "recall_plan": plan,
        "planner_resolved_query": plan["resolved_query"],
        "planner": planner_metadata,
        "recent_dialogue_count": len(recent_dialogue),
        "recent_dialogue_projection": recent_dialogue_projection,
        "cross_layer_weighted_fusion": False,
        "execution_lane": execution_lane,
        "composition_mode": composition_mode,
        "graph": graph_metadata,
        "candidate_paths_executed": candidate_paths_executed,
        "required_selected_layers": required_layers,
        "selected_layer_window_counts": selected_layer_window_counts,
        "normalized_layer_priority": normalized_layer_priorities(plan),
        "inventory_count": len(fast) + len(slow),
        "union_count": len(fast_ranked) + len(slow_ranked),
        "source_inventory_count": len(fast),
        "fast_inventory_count": len(fast),
        "slow_capsule_count": int(payload.get("slow_capsule_head_count", 0)),
        "slow_capsule_head_count": int(
            payload.get("slow_capsule_head_count", 0)
        ),
        "slow_summary_candidate_count": int(
            payload.get("slow_summary_candidate_count", 0)
        ),
        "slow_claim_candidate_count": int(
            payload.get("slow_claim_candidate_count", 0)
        ),
        "slow_summary_hit_count": sum(
            int(bool(item.get("summary_expansion_hit"))) for item in slow_ranked
        ),
        "slow_direct_claim_hit_count": sum(
            int(bool(item.get("direct_claim_hit"))) for item in slow_ranked
        ),
        "slow_retrieval_trace": [
            dict(item.get("slow_retrieval_trace") or {}) for item in slow_ranked
        ],
        "slow_dense_k": int(_arg(args, "slow_dense_k", 24)),
        "fast_semantic_record_count": len(semantic_records),
        "fast_semantic_shortlist_count": fast_semantic_shortlist_count,
        "fast_semantic_state_policy": payload.get("fast_semantic_state_policy"),
        "source_candidate_count": len(source_candidates),
        "source_coverage_trace_k": SOURCE_COVERAGE_TRACE_K,
        "source_top24_candidates": source_coverage_trace(source_candidates),
        "packing_budget_decision": packing_budget_decision,
        "source_candidate_pool_trace": source_coverage_trace(
            source_candidates, limit=len(source_candidates)
        ) if source_candidates else [],
        "fast_shortlist_count": len(fast_ranked),
        "slow_shortlist_count": len(slow_ranked),
        "planned_unit_count": len(units),
        "packed_unit_count": len(packed_units),
        "packing_budget_top_k": packing_budget,
        "budget_excluded_unit_count": int(packing_stats["budget_excluded_unit_count"]),
        "duplicate_unit_count": int(packing_stats["duplicate_unit_count"]),
        "selected_count": len(evidence_windows),
        "source_group_stats": source_group_stats,
        "active_fast_supported_selected_count": sum(
            int(bool(item["retrieval_metadata"]["active_fast_support"]))
            for item in evidence_windows
        ),
        "conflict_policy_rerank_applied": plan["conflict_policy"]
        == "prefer_recent",
        "session_coherent_ordering": True,
        "session_ordering_policy": SESSION_ORDERING_POLICY,
        "atomic_unit_packing": True,
        "restart_boundary_verified": True,
        "graph_counts_before_query": counts_before,
        "graph_fingerprint": graph_fingerprint,
        "checkpoint": str(getattr(models, "checkpoint_path", "")),
        "checkpoint_sha256": getattr(models, "checkpoint_sha256", ""),
        "cross_model_revision": (
            getattr(models, "cross_manifest", {}).get("revision")
            if isinstance(getattr(models, "cross_manifest", {}), Mapping)
            else None
        ),
        "latency_sec": {
            "graph": round(graph_elapsed, 4),
            "source_dense_cross": round(source_elapsed + source_cross_elapsed, 4),
            "fast_graph_fusion": round(graph_elapsed + fast_cross_elapsed, 4),
            "slow_dense_cross": round(slow_elapsed + slow_cross_elapsed, 4),
            "dense": round(dense_elapsed, 4),
            "cross": round(
                source_cross_elapsed + fast_cross_elapsed + slow_cross_elapsed, 4
            ),
            "total": round(time.time() - started, 4),
        },
    }
    return evidence, debug


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for key in ("db_path", "index_path"):
        if key in normalized:
            normalized[key] = str(Path(str(normalized[key])).absolute())
    return normalized


def _row_artifact_path(directory: Path, index: int, qid: str) -> Path:
    return directory / f"row_{index:06d}_{hashlib.sha256(qid.encode('utf-8')).hexdigest()[:12]}.json"


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return dict(value)


def _load_persisted_retrieval_audit(
    *,
    row: Mapping[str, Any],
    out_dir: Path,
    graph_fingerprint: str,
) -> dict[str, Any] | None:
    db_path = Path(str(row["db_path"])).resolve()
    scope_id = _clean(row.get("scope_id"))
    qid = _clean(row.get("question_id"))
    operation_id = _v3().layered_retrieval_operation_id(out_dir, scope_id, qid)
    with closing(sqlite3.connect(db_path)) as connection:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='audit_retrieval_log'"
        ).fetchone()
        if table is None:
            return None
        raw_rows = connection.execute(
            'SELECT event_index,payload_json FROM "audit_retrieval_log" '
            "WHERE scope_id=? ORDER BY event_index",
            (scope_id,),
        ).fetchall()
    matches: list[tuple[int, dict[str, Any]]] = []
    for event_index, raw_payload in raw_rows:
        try:
            payload = json.loads(raw_payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"{qid}: invalid persisted retrieval audit at event {event_index}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                f"{qid}: persisted retrieval audit at event {event_index} is not an object"
            )
        if operation_id in {
            _clean(payload.get("operation_id")),
            _clean(payload.get("idempotency_key")),
        }:
            matches.append((int(event_index), dict(payload)))
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"{qid}: duplicate persisted retrieval audit operation")
    event_index, payload = matches[0]
    expected = {
        "event_kind": "tmcra.v3.layered_retrieval",
        "operation_id": operation_id,
        "question_id": qid,
        "query": _clean(row.get("question")),
        "question_date": _clean(row.get("question_date")),
        "graph_fingerprint": graph_fingerprint,
    }
    for key, value in expected.items():
        if _clean(payload.get(key)) != value:
            raise RuntimeError(
                f"{qid}: persisted retrieval audit {key} does not match the frozen query"
            )
    if payload.get("runtime_input_has_gold") is not False:
        raise RuntimeError(f"{qid}: persisted retrieval audit is not gold-free")
    evidence_sha256 = _clean(payload.get("evidence_sha256"))
    if len(evidence_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in evidence_sha256.lower()
    ):
        raise RuntimeError(f"{qid}: persisted retrieval audit lacks evidence_sha256")
    payload["recall_plan"] = validate_recall_role_plan(payload.get("recall_plan"))
    payload["_event_index"] = event_index
    payload["_payload_sha256"] = _digest(
        {key: value for key, value in payload.items() if not key.startswith("_")}
    )
    return payload


def _assert_audit_matches_result(
    *,
    payload: Mapping[str, Any],
    evidence: Mapping[str, Any],
    debug: Mapping[str, Any],
    operation_id: str,
) -> None:
    qid = _clean(evidence.get("question_id"))
    expected_evidence_sha256 = _digest(dict(evidence))
    checks = {
        "operation_id": (_clean(payload.get("operation_id")), operation_id),
        "question_id": (_clean(payload.get("question_id")), qid),
        "graph_fingerprint": (
            _clean(payload.get("graph_fingerprint")),
            _clean(debug.get("graph_fingerprint")),
        ),
        "evidence_sha256": (
            _clean(payload.get("evidence_sha256")),
            expected_evidence_sha256,
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise RuntimeError(
                f"{qid}: persisted retrieval audit {label} disagrees with replayed result"
            )
    if validate_recall_role_plan(payload.get("recall_plan")) != validate_recall_role_plan(
        evidence.get("recall_plan")
    ):
        raise RuntimeError(f"{qid}: persisted retrieval plan disagrees with result")


def _write_planner_decision(
    *,
    path: Path,
    row_index: int,
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    planner_metadata: Mapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    qid = _clean(row.get("question_id"))
    if planner_metadata.get("physical_api_call") is not True or int(
        planner_metadata.get("physical_api_calls", 0) or 0
    ) != 1:
        raise RuntimeError(f"{qid}: new planner decision lacks one physical API call")
    _atomic_write_json(
        path,
        {
            "schema_version": PLANNER_DECISION_SCHEMA,
            "row_index": row_index,
            "question_id": qid,
            "row_identity_sha256": _digest(_row_identity(row)),
            "graph_fingerprint": _clean(context.get("graph_fingerprint")),
            "recall_plan": validate_recall_role_plan(plan),
            "planner_metadata": dict(planner_metadata),
            "recent_dialogue_projection": dict(
                context.get("recent_dialogue_projection") or {}
            ),
            "available_layers": dict(context.get("available_layers") or {}),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def _load_planner_decision(
    *,
    path: Path,
    row_index: int,
    row: Mapping[str, Any],
    graph_fingerprint: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = _read_json_object(path)
    qid = _clean(row.get("question_id"))
    if (
        value.get("schema_version") != PLANNER_DECISION_SCHEMA
        or int(value.get("row_index", -1)) != row_index
        or _clean(value.get("question_id")) != qid
        or _clean(value.get("row_identity_sha256"))
        != _digest(_row_identity(row))
        or _clean(value.get("graph_fingerprint")) != graph_fingerprint
    ):
        raise RuntimeError(f"{qid}: durable planner decision identity mismatch")
    metadata = value.get("planner_metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError(f"{qid}: durable planner decision lacks metadata")
    if metadata.get("physical_api_call") is not True or int(
        metadata.get("physical_api_calls", 0) or 0
    ) != 1:
        raise RuntimeError(f"{qid}: durable planner decision call count is invalid")
    value["recall_plan"] = validate_recall_role_plan(value.get("recall_plan"))
    value["planner_metadata"] = dict(metadata)
    return value


def _load_explicit_planner_replays(
    replay_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    graph_fingerprints: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    evidence_path = replay_dir / "evidence_windows.jsonl"
    debug_path = replay_dir / "retrieval_debug.jsonl"
    if not evidence_path.is_file() or not debug_path.is_file():
        raise RuntimeError("explicit planner replay directory lacks committed retrieval files")
    evidence_rows = _v3().read_jsonl(evidence_path)
    debug_rows = _v3().read_jsonl(debug_path)
    evidence_by_qid = {_clean(item.get("question_id")): item for item in evidence_rows}
    debug_by_qid = {_clean(item.get("question_id")): item for item in debug_rows}
    expected_qids = [_clean(row.get("question_id")) for row in rows]
    if (
        len(evidence_by_qid) != len(evidence_rows)
        or len(debug_by_qid) != len(debug_rows)
        or set(evidence_by_qid) != set(expected_qids)
        or set(debug_by_qid) != set(expected_qids)
    ):
        raise RuntimeError("explicit planner replay inventory differs from query manifest")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = _clean(row.get("question_id"))
        evidence = evidence_by_qid[qid]
        debug = debug_by_qid[qid]
        if (
            _clean(evidence.get("question")) != _clean(row.get("question"))
            or _clean(evidence.get("question_date")) != _clean(row.get("question_date"))
            or _clean(debug.get("graph_fingerprint")) != graph_fingerprints[qid]
        ):
            raise RuntimeError(f"{qid}: explicit planner replay identity is stale")
        evidence_plan = validate_recall_role_plan(evidence.get("recall_plan"))
        debug_plan = validate_recall_role_plan(debug.get("recall_plan"))
        if evidence_plan != debug_plan:
            raise RuntimeError(f"{qid}: explicit planner replay plans disagree")
        output[qid] = {
            "recall_plan": evidence_plan,
            "source_dir": str(replay_dir.resolve()),
        }
    return output


def _write_row_checkpoint(
    *,
    path: Path,
    row_index: int,
    row: Mapping[str, Any],
    evidence: Mapping[str, Any],
    debug: Mapping[str, Any],
) -> None:
    _atomic_write_json(
        path,
        {
            "schema_version": ROW_CHECKPOINT_SCHEMA,
            "row_index": row_index,
            "question_id": _clean(row.get("question_id")),
            "row_identity_sha256": _digest(_row_identity(row)),
            "graph_fingerprint": _clean(debug.get("graph_fingerprint")),
            "evidence_sha256": _digest(dict(evidence)),
            "debug_sha256": _digest(dict(debug)),
            "evidence": dict(evidence),
            "debug": dict(debug),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def _load_row_checkpoint(
    *,
    path: Path,
    row_index: int,
    row: Mapping[str, Any],
    out_dir: Path,
    graph_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not path.is_file():
        return None
    value = _read_json_object(path)
    qid = _clean(row.get("question_id"))
    if (
        value.get("schema_version") != ROW_CHECKPOINT_SCHEMA
        or int(value.get("row_index", -1)) != row_index
        or _clean(value.get("question_id")) != qid
        or _clean(value.get("row_identity_sha256"))
        != _digest(_row_identity(row))
        or _clean(value.get("graph_fingerprint")) != graph_fingerprint
    ):
        raise RuntimeError(f"{qid}: retrieval row checkpoint identity mismatch")
    evidence, debug = value.get("evidence"), value.get("debug")
    if not isinstance(evidence, Mapping) or not isinstance(debug, Mapping):
        raise RuntimeError(f"{qid}: retrieval row checkpoint payload is invalid")
    evidence, debug = dict(evidence), dict(debug)
    if (
        _digest(evidence) != _clean(value.get("evidence_sha256"))
        or _digest(debug) != _clean(value.get("debug_sha256"))
        or _clean(evidence.get("question_id")) != qid
        or _clean(debug.get("question_id")) != qid
        or Path(str(debug.get("db_path"))).resolve()
        != Path(str(row["db_path"])).resolve()
        or Path(str(debug.get("index_path"))).resolve()
        != Path(str(row["index_path"])).resolve()
    ):
        raise RuntimeError(f"{qid}: retrieval row checkpoint content mismatch")
    audit = _load_persisted_retrieval_audit(
        row=row,
        out_dir=out_dir,
        graph_fingerprint=graph_fingerprint,
    )
    if audit is None:
        raise RuntimeError(f"{qid}: row checkpoint has no persisted retrieval audit")
    operation_id = _v3().layered_retrieval_operation_id(
        out_dir, _clean(row.get("scope_id")), qid
    )
    _assert_audit_matches_result(
        payload=audit,
        evidence=evidence,
        debug=debug,
        operation_id=operation_id,
    )
    return evidence, debug


def _validated_slow_inventory_counts(
    slow: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    candidate_ids: set[str] = set()
    summaries: dict[str, Mapping[str, Any]] = {}
    claims: dict[str, Mapping[str, Any]] = {}
    for item in slow:
        if not isinstance(item, Mapping):
            raise RuntimeError("V4 Slow inventory candidate is not an object")
        if item.get("inventory_schema_version") != SLOW_INVENTORY_SCHEMA_VERSION:
            raise RuntimeError("V4 Slow inventory schema mismatch")
        candidate_id = _clean(item.get("candidate_id"))
        if not candidate_id or candidate_id in candidate_ids:
            raise RuntimeError("V4 Slow inventory candidate ID is missing or duplicate")
        candidate_ids.add(candidate_id)
        kind = item.get("candidate_kind")
        if kind == "capsule_summary":
            summaries[candidate_id] = item
        elif kind == "capsule_claim":
            claims[candidate_id] = item
        else:
            raise RuntimeError(f"V4 Slow inventory candidate kind is invalid: {kind!r}")
    if bool(summaries) != bool(claims):
        raise RuntimeError(
            "V4 Slow inventory must contain summary and claim candidates together"
        )
    referenced_claims: set[str] = set()
    for summary_id, summary in summaries.items():
        children = list(summary.get("child_claim_candidate_ids") or [])
        summary_claims = list(summary.get("claims") or [])
        if not children or len(children) != len(summary_claims):
            raise RuntimeError(
                f"{summary_id}: Slow summary child/claim cardinality mismatch"
            )
        try:
            validate_semantic_summary(
                summary.get("text"),
                summary_claims,
                label=f"{summary_id} summary",
            )
        except PatchValidationError as exc:
            raise RuntimeError(
                f"{summary_id}: Slow summary violates the inventory contract: {exc}"
            ) from exc
        if _current_summary_contract(summary):
            expected_summary = _lossless_summary_projection(summary_claims)
            if _clean(summary.get("text")) != expected_summary:
                raise RuntimeError(
                    f"{summary_id}: current V4.7 Slow summary is not the exact lossless "
                    f"claim projection; expected {expected_summary!r}"
                )
        for child_id in children:
            child_id = _clean(child_id)
            child = claims.get(child_id)
            if child is None or child_id in referenced_claims:
                raise RuntimeError(
                    f"{summary_id}: Slow summary child mapping is missing or duplicate"
                )
            if (
                _clean(child.get("capsule_summary_candidate_id")) != summary_id
                or _clean(child.get("capsule_summary_text"))
                != _clean(summary.get("text"))
                or _clean(child.get("capsule_id"))
                != _clean(summary.get("capsule_id"))
                or child.get("revision") != summary.get("revision")
            ):
                raise RuntimeError(
                    f"{child_id}: Slow summary/claim identity is inconsistent"
                )
            child_claims = list(child.get("claims") or [])
            if len(child_claims) != 1 or not isinstance(child_claims[0], Mapping):
                raise RuntimeError(f"{child_id}: Slow claim candidate is not atomic")
            if not list(child.get("source_parents") or []):
                raise RuntimeError(f"{child_id}: Slow claim has no Source descent")
            referenced_claims.add(child_id)
    if referenced_claims != set(claims):
        raise RuntimeError("V4 Slow inventory contains unreferenced claim candidates")
    _validate_active_capsule_partition(slow)
    return {
        "slow_candidate_count": len(slow),
        "slow_capsule_head_count": len(summaries),
        "slow_summary_candidate_count": len(summaries),
        "slow_claim_candidate_count": len(claims),
    }


def load_online_index(
    path: Path,
    expected_db: Path,
    expected_scope: str,
) -> tuple[
    list[dict[str, Any]],
    Any,
    list[dict[str, Any]],
    Any,
    list[dict[str, Any]],
    dict[str, Any],
]:
    v3 = _v3()
    payload = v3.torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"V4 online index is not an object: {path}")
    if payload.get("schema_version") != ONLINE_INDEX_SCHEMA_VERSION:
        raise RuntimeError(
            f"V4 online index schema mismatch; claim-only V3 indexes cannot be reused: {path}"
        )
    if payload.get("slow_inventory_schema_version") != SLOW_INVENTORY_SCHEMA_VERSION:
        raise RuntimeError(f"V4 Slow inventory contract mismatch: {path}")
    if payload.get("fast_semantic_state_policy") != v3.FAST_SEMANTIC_STATE_POLICY:
        raise RuntimeError(f"V4 online index fast semantic state policy mismatch: {path}")
    if _clean(payload.get("scope_id")) != expected_scope:
        raise RuntimeError(f"V4 online index scope mismatch: {path}")
    if Path(str(payload.get("db_path", ""))).resolve() != expected_db.resolve():
        raise RuntimeError(f"V4 online index database mismatch: {path}")
    text_dim = payload.get("text_dim")
    if not isinstance(text_dim, int) or text_dim <= 0:
        raise RuntimeError(f"V4 online index text dimension is invalid: {path}")
    candidates = list(payload.get("fast_candidates") or [])
    fast_vectors = payload.get("fast_vectors")
    slow = list(payload.get("slow_inventory") or [])
    slow_vectors = payload.get("slow_vectors")
    semantic_records = list(payload.get("fast_semantic_records") or [])
    if (
        not candidates
        or fast_vectors is None
        or tuple(fast_vectors.shape) != (len(candidates), text_dim)
    ):
        raise RuntimeError(f"V4 online source index payload is incomplete: {path}")
    _validate_source_candidate_temporal_metadata(candidates)
    if slow_vectors is None or tuple(slow_vectors.shape) != (len(slow), text_dim):
        raise RuntimeError(f"V4 online Slow index payload is incomplete: {path}")
    if any("labels" in candidate for candidate in [*candidates, *slow]):
        raise RuntimeError("V4 runtime index must not contain benchmark labels")
    slow_counts = _validated_slow_inventory_counts(slow)
    for key, expected in slow_counts.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"V4 online Slow index count mismatch for {key}: {path}"
            )
    if _clean(payload.get("slow_inventory_sha256")) != _digest(slow):
        raise RuntimeError(f"V4 online Slow inventory digest mismatch: {path}")
    return (
        candidates,
        fast_vectors.float().contiguous(),
        slow,
        slow_vectors.float().contiguous(),
        semantic_records,
        dict(payload),
    )


def _validate_source_candidate_temporal_metadata(
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise RuntimeError(f"V4 Source candidate {index} is not an object")
        missing = [
            field
            for field in (
                "session_id",
                "source_record_id",
                "historical_date",
                "timestamp",
                "message_role",
            )
            if not _clean(candidate.get(field))
        ]
        if missing:
            raise RuntimeError(
                f"V4 Source candidate {index} lacks production temporal metadata: "
                + ",".join(missing)
            )
        if _clean(candidate.get("message_role")) not in {
            "user",
            "assistant",
            "system",
            "tool",
        }:
            raise RuntimeError(
                f"V4 Source candidate {index} has an invalid message_role"
            )


def command_build_index(args: argparse.Namespace) -> None:
    v3 = _v3()
    rows = v3.read_jsonl(Path(args.scope_manifest))
    device = v3.torch.device(args.device)
    if device.type == "cuda" and not v3.torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    vectorizer: Any | None = None
    started = time.time()
    report_rows: list[dict[str, Any]] = []
    reused_index_count = 0
    for row_index, row in enumerate(rows, start=1):
        db_path = Path(row["db_path"]).resolve()
        scope_id = _clean(row.get("scope_id"))
        index_path = Path(row["index_path"]).resolve()
        expected = {
            "schema_version": ONLINE_INDEX_SCHEMA_VERSION,
            "slow_inventory_schema_version": SLOW_INVENTORY_SCHEMA_VERSION,
            "fast_semantic_state_policy": v3.FAST_SEMANTIC_STATE_POLICY,
            "scope_id": scope_id,
            "db_path": str(db_path),
            "subchunk_chars": int(args.subchunk_chars),
            "subchunk_overlap": int(args.subchunk_overlap),
            "embedding_model": str(Path(args.embedding_model).resolve()),
            "embedding_max_length": int(args.embedding_max_length),
            "text_dim": int(args.text_dim),
            "strict_no_truncation": True,
        }
        if index_path.exists():
            payload = v3.torch.load(index_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping):
                raise RuntimeError(f"existing V4 online index is not an object: {index_path}")
            mismatches = {
                key: {"expected": value, "actual": payload.get(key)}
                for key, value in expected.items()
                if payload.get(key) != value
            }
            current_fingerprint = v3.scope_fingerprint(db_path, scope_id)
            if _clean(payload.get("graph_fingerprint")) != current_fingerprint:
                mismatches["graph_fingerprint"] = {
                    "expected": current_fingerprint,
                    "actual": payload.get("graph_fingerprint"),
                }
            if mismatches:
                raise RuntimeError(
                    "existing online index is incompatible with the V4 summary/claim contract: "
                    f"{index_path} mismatches={json.dumps(mismatches, sort_keys=True)}"
                )
            loaded = load_online_index(index_path, db_path, scope_id)
            loaded_payload = loaded[-1]
            report_row = {
                "question_id": _clean(row.get("question_id")),
                "scope_id": scope_id,
                "db_path": str(db_path),
                "index_path": str(index_path),
                "parent_count": int(loaded_payload["parent_count"]),
                "candidate_count": int(loaded_payload["candidate_count"]),
                "slow_capsule_head_count": int(
                    loaded_payload["slow_capsule_head_count"]
                ),
                "slow_summary_candidate_count": int(
                    loaded_payload["slow_summary_candidate_count"]
                ),
                "slow_claim_candidate_count": int(
                    loaded_payload["slow_claim_candidate_count"]
                ),
                "fast_semantic_record_count": int(
                    loaded_payload["fast_semantic_record_count"]
                ),
                "fast_semantic_state_policy": loaded_payload[
                    "fast_semantic_state_policy"
                ],
                "graph_counts": dict(loaded_payload["graph_counts_at_index"]),
                "graph_fingerprint": current_fingerprint,
                "reused_existing_index": True,
            }
            reused_index_count += 1
            report_rows.append(report_row)
            print(
                json.dumps(
                    {
                        "status": "index_reused",
                        "row": row_index,
                        "total": len(rows),
                        **report_row,
                    }
                ),
                flush=True,
            )
            continue
        if vectorizer is None:
            vectorizer = v3.BgeM3DenseVectorizer(
                dim=args.text_dim,
                model_path=args.embedding_model,
                device=str(device),
                max_length=args.embedding_max_length,
                strict_max_length=True,
            )
        graph_fingerprint = v3.scope_fingerprint(db_path, scope_id)
        parents = v3.load_persisted_parent_chunks(db_path, scope_id)
        candidates = v3.parent_subchunks(
            parents,
            scope_id=scope_id,
            subchunk_chars=args.subchunk_chars,
            subchunk_overlap=args.subchunk_overlap,
        )
        slow, semantic_records = load_v4_layered_inventory(
            db_path, scope_id, parents
        )
        slow_counts = _validated_slow_inventory_counts(slow)
        fast_vectors = vectorizer.encode_batch(
            [candidate["text"] for candidate in candidates],
            batch_size=args.batch_size,
        )
        slow_vectors = (
            vectorizer.encode_batch(
                [candidate["text"] for candidate in slow],
                batch_size=args.batch_size,
            )
            if slow
            else v3.torch.empty((0, args.text_dim), dtype=v3.torch.float32)
        )
        fast_vectors = fast_vectors.to(v3.torch.float16).contiguous()
        slow_vectors = slow_vectors.to(v3.torch.float16).contiguous()
        counts = v3.scope_counts(db_path, scope_id)
        if v3.scope_fingerprint(db_path, scope_id) != graph_fingerprint:
            raise RuntimeError(
                f"{scope_id}: graph changed while the V4 online index was being built"
            )
        payload = {
            **expected,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "graph_counts_at_index": counts,
            "graph_fingerprint": graph_fingerprint,
            "parent_count": len(parents),
            "candidate_count": len(candidates),
            **slow_counts,
            "fast_semantic_record_count": len(semantic_records),
            "fast_candidates": candidates,
            "fast_vectors": fast_vectors,
            "slow_inventory": slow,
            "slow_inventory_sha256": _digest(slow),
            "slow_vectors": slow_vectors,
            "fast_semantic_records": semantic_records,
        }
        v3.atomic_torch_save(payload, index_path)
        report_row = {
            "question_id": _clean(row.get("question_id")),
            "scope_id": scope_id,
            "db_path": str(db_path),
            "index_path": str(index_path),
            "parent_count": len(parents),
            "candidate_count": len(candidates),
            "slow_capsule_head_count": slow_counts["slow_capsule_head_count"],
            "slow_summary_candidate_count": slow_counts[
                "slow_summary_candidate_count"
            ],
            "slow_claim_candidate_count": slow_counts[
                "slow_claim_candidate_count"
            ],
            "fast_semantic_record_count": len(semantic_records),
            "fast_semantic_state_policy": v3.FAST_SEMANTIC_STATE_POLICY,
            "graph_counts": counts,
            "graph_fingerprint": graph_fingerprint,
            "reused_existing_index": False,
        }
        report_rows.append(report_row)
        print(
            json.dumps(
                {
                    "status": "indexed",
                    "row": row_index,
                    "total": len(rows),
                    **report_row,
                }
            ),
            flush=True,
        )
    report = {
        "status": "complete",
        "schema_version": ONLINE_INDEX_REPORT_SCHEMA_VERSION,
        "row_count": len(report_rows),
        "parent_count": sum(row["parent_count"] for row in report_rows),
        "candidate_count": sum(row["candidate_count"] for row in report_rows),
        "slow_capsule_head_count": sum(
            row["slow_capsule_head_count"] for row in report_rows
        ),
        "slow_summary_candidate_count": sum(
            row["slow_summary_candidate_count"] for row in report_rows
        ),
        "slow_claim_candidate_count": sum(
            row["slow_claim_candidate_count"] for row in report_rows
        ),
        "reused_index_count": reused_index_count,
        "elapsed_sec": round(time.time() - started, 3),
        "rows": report_rows,
    }
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(out_report, json.dumps(report, indent=2, sort_keys=True) + "\n")


def _staging_runtime_configuration(args: argparse.Namespace) -> dict[str, Any]:
    (
        composition_mode,
        execution_lane,
        packing_budget_mode,
        top_k,
    ) = _validated_runtime_route(args, label="retrieval runtime")

    def path_value(name: str) -> str:
        value = _arg(args, name, None)
        return str(Path(value).resolve()) if value not in {None, ""} else ""

    return {
        "retrieval_contract_schema": RETRIEVAL_CONTRACT_SCHEMA,
        "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        "online_index_schema_version": ONLINE_INDEX_SCHEMA_VERSION,
        "slow_inventory_schema_version": SLOW_INVENTORY_SCHEMA_VERSION,
        "execution_lane": execution_lane,
        "composition_mode": composition_mode,
        "packing_budget_mode": packing_budget_mode,
        "top_k": top_k,
        "adaptive_simple_k": int(_arg(args, "adaptive_simple_k", 8)),
        "adaptive_standard_k": int(_arg(args, "adaptive_standard_k", 12)),
        "adaptive_complex_k": int(_arg(args, "adaptive_complex_k", 16)),
        "source_coverage_trace_k": SOURCE_COVERAGE_TRACE_K,
        "checkpoint": path_value("checkpoint"),
        "cross_model": path_value("cross_model"),
        "repo": path_value("repo"),
        "harness": path_value("harness"),
        "node_model": path_value("node_model"),
        "path_model": path_value("path_model"),
        "embedding_model": path_value("embedding_model"),
        "device": str(_arg(args, "device", "cuda")),
        "graph_device": str(_arg(args, "graph_device", "cuda")),
        "candidate_event_k": int(_arg(args, "candidate_event_k", 24)),
        "support_path_k": int(_arg(args, "support_path_k", 3)),
        "path_tunnel_rescue_k": int(_arg(args, "path_tunnel_rescue_k", 2)),
        "graph_top_k": int(_arg(args, "graph_top_k", 12)),
        "dense_k": int(_arg(args, "dense_k", 32)),
        "slow_dense_k": int(_arg(args, "slow_dense_k", 24)),
        "graph_k": int(_arg(args, "graph_k", 24)),
        "cross_max_length": int(_arg(args, "cross_max_length", 1280)),
        "cross_batch_size": int(_arg(args, "cross_batch_size", 24)),
        "embedding_max_length": int(_arg(args, "embedding_max_length", 512)),
    }


def command_retrieve(args: argparse.Namespace) -> None:
    runtime_configuration = _staging_runtime_configuration(args)
    v3 = _v3()
    rows = v3.read_jsonl(Path(args.query_manifest))
    if not rows:
        raise RuntimeError("query manifest is empty")
    # Keep checkpoint identity tied to the stable logical path, not whichever
    # physical volume currently backs that path.
    out_dir = Path(args.out_dir).absolute()
    if out_dir.exists():
        raise RuntimeError(f"retrieval output already exists: {out_dir}")
    resume = bool(getattr(args, "resume", False))
    staging = out_dir.with_name(f".{out_dir.name}.staging")
    staging_preexisted = staging.exists()
    if staging_preexisted and not resume:
        raise RuntimeError(
            f"retrieval staging exists; explicit --resume is required: {staging}"
        )
    if staging_preexisted and not staging.is_dir():
        raise RuntimeError(f"retrieval staging is not a directory: {staging}")
    if not staging_preexisted:
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir(parents=False, exist_ok=False)
    checkpoint_dir = staging / "row_checkpoints"
    planner_dir = staging / "planner_decisions"
    checkpoint_dir.mkdir(exist_ok=True)
    planner_dir.mkdir(exist_ok=True)

    staging_identity = {
        "schema_version": RUN_STAGING_SCHEMA,
        "out_dir": str(out_dir),
        "query_count": len(rows),
        "question_ids": [_clean(row.get("question_id")) for row in rows],
        "row_identity_sha256": [_digest(_row_identity(row)) for row in rows],
        "runtime_configuration": runtime_configuration,
    }
    identity_path = staging / "staging_identity.json"
    if identity_path.is_file():
        if _read_json_object(identity_path) != staging_identity:
            raise RuntimeError("retrieval staging identity does not match query manifest")
    elif staging_preexisted:
        raise RuntimeError("retrieval staging lacks a durable identity")
    else:
        _atomic_write_json(identity_path, staging_identity)

    graph_fingerprints = {
        _clean(row.get("question_id")): v3.scope_fingerprint(
            Path(str(row["db_path"])).resolve(), _clean(row.get("scope_id"))
        )
        for row in rows
    }
    planner_replay_dir = _clean(getattr(args, "planner_replay_dir", ""))
    explicit_planner_replays = (
        _load_explicit_planner_replays(
            Path(planner_replay_dir).resolve(), rows, graph_fingerprints
        )
        if planner_replay_dir
        else {}
    )
    persisted_audits = {
        _clean(row.get("question_id")): _load_persisted_retrieval_audit(
            row=row,
            out_dir=out_dir,
            graph_fingerprint=graph_fingerprints[_clean(row.get("question_id"))],
        )
        for row in rows
    }
    if not resume:
        already_audited = [qid for qid, value in persisted_audits.items() if value]
        if already_audited:
            raise RuntimeError(
                "persisted retrieval work exists; explicit --resume is required: "
                + ",".join(already_audited)
            )

    planner = None if len(explicit_planner_replays) == len(rows) else planner_from_env()
    harness, models = v3.load_native_harness(Path(args.harness), Path(args.repo)), v3.OnlineModels(args)
    harness.disable_topic_bucket_runtime()
    evidence_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], Any] = {}
    started = time.time()
    checkpoint_reused_count = 0
    planner_decision_replay_count = 0
    audit_plan_replay_count = 0
    explicit_planner_replay_count = 0
    new_planner_call_count = 0
    newly_appended_audit_count = 0
    for index, row in enumerate(rows, start=1):
        qid = _clean(row.get("question_id"))
        graph_fingerprint = graph_fingerprints[qid]
        checkpoint_path = _row_artifact_path(checkpoint_dir, index, qid)
        checkpoint = _load_row_checkpoint(
            path=checkpoint_path,
            row_index=index,
            row=row,
            out_dir=out_dir,
            graph_fingerprint=graph_fingerprint,
        )
        if checkpoint is not None:
            evidence, debug = checkpoint
            checkpoint_reused_count += 1
            evidence_rows.append(evidence)
            debug_rows.append(debug)
            print(
                json.dumps(
                    {
                        "status": "checkpoint_reused",
                        "row": index,
                        "total": len(rows),
                        "question_id": qid,
                    }
                ),
                flush=True,
            )
            continue

        audit = persisted_audits[qid]
        decision_path = _row_artifact_path(planner_dir, index, qid)
        decision = _load_planner_decision(
            path=decision_path,
            row_index=index,
            row=row,
            graph_fingerprint=graph_fingerprint,
        )
        route_override: Mapping[str, Any] | None = None
        route_metadata: Mapping[str, Any] | None = None
        explicit_replay = explicit_planner_replays.get(qid)
        if explicit_replay is not None:
            route_override = explicit_replay["recall_plan"]
            if decision is not None and validate_recall_role_plan(
                decision.get("recall_plan")
            ) != validate_recall_role_plan(route_override):
                raise RuntimeError(f"{qid}: durable and explicit planner replays disagree")
            if audit is not None and validate_recall_role_plan(
                audit.get("recall_plan")
            ) != validate_recall_role_plan(route_override):
                raise RuntimeError(f"{qid}: audit and explicit planner replays disagree")
            route_metadata = {
                "physical_api_call": False,
                "physical_api_calls": 0,
                "stage": "recall_planner",
                "status": "explicit_replay",
                "planner_version": "frozen_committed_retrieval",
                "prompt_version": "frozen_committed_retrieval",
                "replayed_without_api": True,
                "replay_physical_api_call": False,
                "replay_source": "explicit_committed_retrieval",
                "replay_source_dir": explicit_replay["source_dir"],
            }
            explicit_planner_replay_count += 1
        elif decision is not None:
            route_override = decision["recall_plan"]
            if audit is not None and validate_recall_role_plan(
                audit.get("recall_plan")
            ) != validate_recall_role_plan(route_override):
                raise RuntimeError(f"{qid}: planner decision and audit plan disagree")
            route_metadata = {
                **dict(decision["planner_metadata"]),
                "replayed_without_api": True,
                "replay_physical_api_call": False,
                "replay_source": "durable_planner_decision",
            }
            planner_decision_replay_count += 1
        elif audit is not None:
            route_override = audit["recall_plan"]
            route_metadata = {
                "physical_api_call": True,
                "physical_api_calls": 1,
                "stage": "recall_planner",
                "provider": "deepseek",
                "model": DEEPSEEK_FLASH_MODEL,
                "status": "completed_response_metadata_lost_after_process_failure",
                "planner_version": "recovered_from_layered_retrieval_audit",
                "prompt_version": "unknown_prior_process",
                "historical_call_usage_known": False,
                "replayed_without_api": True,
                "replay_physical_api_call": False,
                "replay_source": "persisted_layered_retrieval_audit",
                "recovery_operation_id": _clean(audit.get("operation_id")),
                "recovery_payload_sha256": _clean(audit.get("_payload_sha256")),
            }
            audit_plan_replay_count += 1

        def persist_new_planner_decision(
            plan: Mapping[str, Any],
            metadata: Mapping[str, Any],
            context: Mapping[str, Any],
        ) -> None:
            _write_planner_decision(
                path=decision_path,
                row_index=index,
                row=row,
                plan=plan,
                planner_metadata=metadata,
                context=context,
            )

        evidence, debug = retrieve_one(
            row,
            args=args,
            harness=harness,
            models=models,
            planner=planner,
            route_override=route_override,
            route_override_metadata=route_metadata,
            planner_decision_callback=(
                persist_new_planner_decision if route_override is None else None
            ),
            graph_adapter_cache=cache,
        )
        if route_override is None:
            new_planner_call_count += 1
        operation_id = v3.layered_retrieval_operation_id(out_dir, str(debug["scope_id"]), str(evidence["question_id"]))
        if audit is not None:
            _assert_audit_matches_result(
                payload=audit,
                evidence=evidence,
                debug=debug,
                operation_id=operation_id,
            )
        persisted = v3.append_layered_retrieval_audit(repo=Path(args.repo), db_path=Path(debug["db_path"]), scope_id=str(debug["scope_id"]), operation_id=operation_id, evidence=evidence, debug=debug)
        audit_payload = dict(persisted.get("payload") or {})
        _assert_audit_matches_result(
            payload=audit_payload,
            evidence=evidence,
            debug=debug,
            operation_id=operation_id,
        )
        newly_appended_audit_count += int(bool(persisted.get("appended")))
        debug["layered_retrieval_audit"] = {"operation_id": operation_id, "query_id": _clean(audit_payload.get("query_id")), "event_total": int(persisted.get("event_total") or 0), "trimmed_total": int(persisted.get("trimmed_total") or 0), "newly_appended": bool(persisted.get("appended"))}
        if not _clean(debug["layered_retrieval_audit"]["query_id"]):
            raise RuntimeError(f"{qid}: persisted retrieval audit lacks query_id")
        _write_row_checkpoint(
            path=checkpoint_path,
            row_index=index,
            row=row,
            evidence=evidence,
            debug=debug,
        )
        evidence_rows.append(evidence)
        debug_rows.append(debug)
        print(json.dumps({"status": "retrieved", "row": index, "total": len(rows), "question_id": evidence["question_id"], "inventory": debug["inventory_count"], "selected": debug["selected_count"], "latency_sec": debug["latency_sec"]["total"]}), flush=True)
    latencies = [float(row["latency_sec"]["total"]) for row in debug_rows]
    budget_tiers = {
        tier: sum(
            int(dict(row.get("packing_budget_decision") or {}).get("tier") == tier)
            for row in debug_rows
        )
        for tier in ("fixed", "simple", "standard", "complex")
    }
    layer_window_totals = {
        layer: sum(
            int(dict(row.get("selected_layer_window_counts") or {}).get(layer, 0))
            for row in debug_rows
        )
        for layer in ("source", "fast", "slow")
    }
    report = {
        "status": "complete",
        "schema_version": "tmcra.v4.online-retrieval-report.6",
        "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        "online_index_schema_version": ONLINE_INDEX_SCHEMA_VERSION,
        "slow_inventory_schema_version": SLOW_INVENTORY_SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "query_count": len(rows),
        "execution_lane": str(args.execution_lane),
        "composition_mode": str(args.composition_mode),
        "production_layered_contract_enforced": (
            str(args.execution_lane) == "production"
            and str(args.composition_mode) == "layered"
        ),
        "candidate_paths_executed_count": sum(
            int(all(row["candidate_paths_executed"].values())) for row in debug_rows
        ),
        "required_layer_row_counts": {
            layer: sum(
                int(layer in list(row.get("required_selected_layers") or []))
                for row in debug_rows
            )
            for layer in ("source", "fast", "slow")
        },
        "selected_layer_window_totals": layer_window_totals,
        "fast_semantic_shortlist_total": sum(
            int(row.get("fast_semantic_shortlist_count", 0)) for row in debug_rows
        ),
        "avg_budget_excluded_unit_count": round(
            sum(row["budget_excluded_unit_count"] for row in debug_rows) / len(debug_rows),
            4,
        ),
        "avg_duplicate_unit_count": round(
            sum(row["duplicate_unit_count"] for row in debug_rows) / len(debug_rows),
            4,
        ),
        "avg_latency_sec": round(sum(latencies) / len(latencies), 4),
        "max_latency_sec": round(max(latencies), 4),
        "elapsed_sec": round(time.time() - started, 3),
        "evidence": str(out_dir / "evidence_windows.jsonl"),
        "debug": str(out_dir / "retrieval_debug.jsonl"),
        "source_coverage_trace_k": SOURCE_COVERAGE_TRACE_K,
        "packing_budget_mode": str(args.packing_budget_mode),
        "configured_top_k": int(args.top_k),
        "max_final_window_count": max(
            int(row.get("selected_count", 0)) for row in debug_rows
        ),
        "atomic_unit_packing": True,
        "packing_budget_tier_counts": budget_tiers,
        "cross_layer_weighted_fusion": False,
        "answer_attachment_contract": "object-list-v1",
        "ranking_metadata_field": "retrieval_metadata",
        "session_coherent_ordering": True,
        "session_ordering_policy": SESSION_ORDERING_POLICY,
        "row_checkpoint_policy": ROW_CHECKPOINT_SCHEMA,
        "checkpoint_reused_count": checkpoint_reused_count,
        "explicit_planner_replay_count": explicit_planner_replay_count,
        "planner_decision_replay_count": planner_decision_replay_count,
        "audit_plan_replay_count": audit_plan_replay_count,
        "new_planner_call_count": new_planner_call_count,
        "newly_appended_retrieval_audit_count": newly_appended_audit_count,
        "historical_planner_calls_without_usage_count": sum(
            int(dict(row.get("planner") or {}).get("historical_call_usage_known") is False)
            for row in debug_rows
        ),
    }
    _atomic_write(staging / "evidence_windows.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in evidence_rows))
    _atomic_write(staging / "retrieval_debug.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in debug_rows))
    _atomic_write(staging / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(staging, out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-model", default=os.getenv("TMCRA_BGE_M3_PATH", "BAAI/bge-m3"))
    parser.add_argument("--text-dim", type=int, default=1024)
    parser.add_argument("--embedding-max-length", type=int, default=8192)
    parser.add_argument("--device", default="cuda")


def main() -> int:
    parser = argparse.ArgumentParser(description="TMCRA V4 production online index and retrieval runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("build-index")
    index.add_argument("--scope-manifest", required=True); index.add_argument("--out-report", required=True); index.add_argument("--subchunk-chars", type=int, default=1800); index.add_argument("--subchunk-overlap", type=int, default=200); index.add_argument("--batch-size", type=int, default=16); add_common_model_args(index)
    retrieve = sub.add_parser("retrieve")
    retrieve.add_argument("--query-manifest", required=True); retrieve.add_argument("--out-dir", required=True); retrieve.add_argument("--resume", action="store_true"); retrieve.add_argument("--planner-replay-dir"); retrieve.add_argument("--checkpoint", required=True); retrieve.add_argument("--cross-model", default=os.getenv("TMCRA_BGE_RERANKER_PATH", "BAAI/bge-reranker-v2-m3")); retrieve.add_argument("--cross-max-length", type=int, default=1280); retrieve.add_argument("--cross-batch-size", type=int, default=24); retrieve.add_argument("--repo", required=True); retrieve.add_argument("--harness", required=True); retrieve.add_argument("--node-model", required=True); retrieve.add_argument("--path-model", required=True); retrieve.add_argument("--graph-device", default="cuda"); retrieve.add_argument("--candidate-event-k", type=int, default=24); retrieve.add_argument("--support-path-k", type=int, default=3); retrieve.add_argument("--path-tunnel-rescue-k", type=int, default=2); retrieve.add_argument("--graph-top-k", type=int, default=12); retrieve.add_argument("--dense-k", type=int, default=32); retrieve.add_argument("--slow-dense-k", type=int, default=24); retrieve.add_argument("--graph-k", type=int, default=24); retrieve.add_argument("--execution-lane", choices=sorted(EXECUTION_LANES), default="production"); retrieve.add_argument("--composition-mode", choices=sorted(COMPOSITION_MODES), default="layered"); retrieve.add_argument("--packing-budget-mode", choices=sorted(PACKING_BUDGET_MODES), default="fixed"); retrieve.add_argument("--top-k", type=int, default=8); retrieve.add_argument("--adaptive-simple-k", type=int, default=8); retrieve.add_argument("--adaptive-standard-k", type=int, default=12); retrieve.add_argument("--adaptive-complex-k", type=int, default=16); add_common_model_args(retrieve)
    args = parser.parse_args()
    (command_build_index if args.command == "build-index" else command_retrieve)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
