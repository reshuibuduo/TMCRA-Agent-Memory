"""Fail-closed routing policy for the promoted TMCRA answer lane."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tmcra_v4_evidence_operations import (
    PACKET_COMPILER_VERSION,
    PACKET_SCHEMA,
    PLAN_SCHEMA,
)


POLICY_SCHEMA = "tmcra.v4.production-route-policy.2"
RETRIEVAL_CONTRACT_SCHEMA = "tmcra.v4.layered-retrieval-contract.3"
RETRIEVAL_INVENTORY_COUNT_FIELDS = (
    "source",
    "fast",
    "fast_semantic",
    "slow",
    "slow_capsule_heads",
    "slow_summaries",
    "slow_claims",
    "slow_ranked_claims",
)
PRODUCTION_LANE = "layered_memory_operation_bound_gpt54"
PRODUCTION_COMPOSITION_MODE = "layered"
PRODUCTION_PACKING_BUDGET_MODE = "fixed"
PRODUCTION_FINAL_TOP_K = 8
SOURCE_COVERAGE_TRACE_K = 24
PRODUCTION_PACKET_FIELD = "compiled_evidence_packet"
PRODUCTION_ANSWER_PROTOCOL = "evidence_operation_bound_v6"
PRODUCTION_ANSWER_MODEL = "gpt-5.4"
PRODUCTION_ANSWER_RUNNER = "run_tmcra_v4_gpt54_answers.py"
SHADOW_PACKET_FIELD = "semantic_evidence_packet"
SHADOW_ANSWER_PROTOCOL = "semantic_advisory_bound_v2"
RETRIEVAL_MODES = {"layered", "source-only-diagnostic"}
FORBIDDEN_ANSWER_FACING_FIELDS = {
    "answer",
    "gold_answer",
    "expected_answer",
    "answer_session_ids",
    "labels",
    "supervision",
}


class RoutePolicyError(RuntimeError):
    pass


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _source_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _source_identity(item: Mapping[str, Any]) -> tuple[Any, ...]:
    session_id = _text(item.get("session_id"))
    parent_chunk_index = int(item.get("parent_chunk_index", 0))
    source_group_id = _text(item.get("source_group_id")) or (
        f"source-group::{session_id}:{parent_chunk_index}"
    )
    context = item.get("source_group_context") or []
    if not isinstance(context, Sequence) or isinstance(context, (str, bytes)):
        raise RoutePolicyError("Source group context is not an array")
    context_identity = tuple(
        (
            _text(member.get("relationship")),
            int(member.get("parent_distance", 0)),
            _text(member.get("session_id")),
            int(member.get("session_index", 0)),
            int(member.get("parent_chunk_index", 0)),
            _text(member.get("source_record_id")),
            int(member.get("source_char_start", 0)),
            int(
                member.get(
                    "source_char_end",
                    len(member.get("text")) if isinstance(member.get("text"), str) else 0,
                )
            ),
            _source_text(member.get("text")),
        )
        for member in context
        if isinstance(member, Mapping)
    )
    if len(context_identity) != len(context):
        raise RoutePolicyError("Source group context contains a non-object member")
    return (
        _text(item.get("db_path")),
        _text(item.get("scope_id")),
        _text(item.get("source_record_id")),
        session_id,
        int(item.get("session_index", 0)),
        int(item.get("parent_chunk_index", 0)),
        int(item.get("subchunk_index", 0)),
        int(item.get("source_char_start", 0)),
        int(
            item.get(
                "source_char_end",
                len(item.get("text")) if isinstance(item.get("text"), str) else 0,
            )
        ),
        _source_text(item.get("text")),
        source_group_id,
        context_identity,
    )


def assert_production_answer_runner(path: Path) -> None:
    if path.name != PRODUCTION_ANSWER_RUNNER:
        raise RoutePolicyError(
            f"production answer runner must be {PRODUCTION_ANSWER_RUNNER}, got {path.name or path}"
        )


def validate_production_retrieval_mode(
    composition_mode: Any, *, execution_lane: Any
) -> None:
    lane = _text(execution_lane)
    mode = _text(composition_mode)
    if lane not in {"production", "diagnostic"}:
        raise RoutePolicyError("retrieval execution lane must be production or diagnostic")
    if mode not in RETRIEVAL_MODES:
        raise RoutePolicyError(f"retrieval composition mode is invalid: {mode!r}")
    if lane == "production" and mode != PRODUCTION_COMPOSITION_MODE:
        raise RoutePolicyError(
            "production retrieval requires layered composition; "
            f"got {mode or '<missing>'!r}"
        )


def validate_production_packing_budget(
    packing_budget_mode: Any,
    top_k: Any,
    *,
    execution_lane: Any,
) -> None:
    lane = _text(execution_lane)
    mode = _text(packing_budget_mode)
    if lane not in {"production", "diagnostic"}:
        raise RoutePolicyError("retrieval execution lane must be production or diagnostic")
    if mode not in {"fixed", "adaptive"}:
        raise RoutePolicyError(f"retrieval packing budget mode is invalid: {mode!r}")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise RoutePolicyError("retrieval packing budget must be a positive integer")
    if lane == "production" and (
        mode != PRODUCTION_PACKING_BUDGET_MODE or top_k != PRODUCTION_FINAL_TOP_K
    ):
        raise RoutePolicyError(
            "production retrieval requires a fixed Top8 packing budget; "
            f"got mode={mode or '<missing>'!r}, top_k={top_k!r}"
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RoutePolicyError(f"{field} must be a nonnegative integer")
    return value


def _layer_contribution_counts(windows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"source": 0, "fast": 0, "slow": 0}
    for window in windows:
        metadata = window.get("retrieval_metadata")
        contributions = metadata.get("layer_contributions") if isinstance(metadata, Mapping) else None
        if not isinstance(contributions, Sequence) or isinstance(contributions, (str, bytes)):
            raise RoutePolicyError("production evidence window lacks layer contributions")
        seen: set[str] = set()
        for contribution in contributions:
            if not isinstance(contribution, Mapping):
                raise RoutePolicyError("production layer contribution is not an object")
            layer = _text(contribution.get("layer"))
            if layer not in counts:
                raise RoutePolicyError(f"production evidence contains unknown layer {layer!r}")
            seen.add(layer)
        for layer in seen:
            counts[layer] += 1
    return counts


def _validate_retrieval_contract(
    row: Mapping[str, Any],
    windows: Sequence[Mapping[str, Any]],
    *,
    qid: str,
    expected_lane: str = "production",
) -> dict[str, Any]:
    contract = row.get("retrieval_contract")
    if not isinstance(contract, Mapping):
        raise RoutePolicyError(f"{qid}: production evidence lacks retrieval contract")
    if contract.get("schema_version") != RETRIEVAL_CONTRACT_SCHEMA:
        raise RoutePolicyError(f"{qid}: retrieval contract schema is invalid")
    validate_production_retrieval_mode(
        contract.get("composition_mode"), execution_lane=contract.get("execution_lane")
    )
    actual_lane = _text(contract.get("execution_lane"))
    if actual_lane != expected_lane:
        if expected_lane == "production" and actual_lane == "diagnostic":
            raise RoutePolicyError(
                f"{qid}: diagnostic retrieval cannot enter production evaluation"
            )
        raise RoutePolicyError(
            f"{qid}: retrieval lane is {actual_lane!r}, expected {expected_lane!r}"
        )

    inventory = contract.get("inventory_counts")
    paths = contract.get("candidate_paths_executed")
    required = contract.get("required_selected_layers")
    selected = contract.get("selected_layer_window_counts")
    packing_budget_mode = _text(contract.get("packing_budget_mode"))
    packing_budget = contract.get("packing_budget")
    source_trace_k = contract.get("source_coverage_trace_k")
    final_window_count = contract.get("final_window_count")
    try:
        validate_production_packing_budget(
            packing_budget_mode,
            packing_budget,
            execution_lane=actual_lane,
        )
    except RoutePolicyError as exc:
        raise RoutePolicyError(f"{qid}: {exc}") from exc
    if source_trace_k != SOURCE_COVERAGE_TRACE_K:
        raise RoutePolicyError(
            f"{qid}: Source coverage trace contract must be Top{SOURCE_COVERAGE_TRACE_K}"
        )
    if final_window_count != len(windows):
        raise RoutePolicyError(f"{qid}: final window count does not match evidence")
    if expected_lane == "production" and len(windows) > PRODUCTION_FINAL_TOP_K:
        raise RoutePolicyError(
            f"{qid}: production final evidence exceeds Top{PRODUCTION_FINAL_TOP_K}"
        )
    if not isinstance(inventory, Mapping) or set(inventory) != set(
        RETRIEVAL_INVENTORY_COUNT_FIELDS
    ):
        raise RoutePolicyError(f"{qid}: retrieval inventory contract is invalid")
    if not isinstance(paths, Mapping) or set(paths) != {"source", "fast", "slow"}:
        raise RoutePolicyError(f"{qid}: candidate path execution contract is invalid")
    if any(not isinstance(paths[layer], bool) for layer in ("source", "fast", "slow")):
        raise RoutePolicyError(f"{qid}: candidate path execution flags must be booleans")
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        raise RoutePolicyError(f"{qid}: required selected layers are invalid")
    required_values = [_text(value) for value in required]
    required_layers = set(required_values)
    if (
        not required_layers
        or len(required_values) != len(required_layers)
        or not required_layers.issubset({"source", "fast", "slow"})
    ):
        raise RoutePolicyError(f"{qid}: required selected layers are invalid")
    inventory_counts = {
        layer: _nonnegative_int(
            inventory[layer], field=f"{qid}: inventory_counts.{layer}"
        )
        for layer in RETRIEVAL_INVENTORY_COUNT_FIELDS
    }
    if inventory_counts["fast_semantic"] > inventory_counts["fast"]:
        raise RoutePolicyError(
            f"{qid}: semantic Fast shortlist exceeds the Fast shortlist"
        )
    if (
        inventory_counts["slow_capsule_heads"]
        != inventory_counts["slow_summaries"]
        or inventory_counts["slow_ranked_claims"] != inventory_counts["slow"]
        or inventory_counts["slow_claims"]
        < inventory_counts["slow_capsule_heads"]
        or inventory_counts["slow_ranked_claims"]
        > inventory_counts["slow_claims"]
        or (
            inventory_counts["slow_claims"] > 0
            and inventory_counts["slow_capsule_heads"] == 0
        )
    ):
        raise RoutePolicyError(
            f"{qid}: Slow summary/claim inventory counts are inconsistent"
        )
    expected_required = {
        layer
        for layer in ("source", "fast", "slow")
        if inventory_counts[layer] > 0
    }
    if expected_lane == "production" and required_layers != expected_required:
        raise RoutePolicyError(
            f"{qid}: required selected layers do not match nonempty inventories"
        )
    actual = _layer_contribution_counts(windows)
    if not isinstance(selected, Mapping) or set(selected) != set(actual):
        raise RoutePolicyError(f"{qid}: selected layer counts are invalid")
    selected_counts = {
        layer: _nonnegative_int(
            selected[layer], field=f"{qid}: selected_layer_window_counts.{layer}"
        )
        for layer in actual
    }
    if selected_counts != actual:
        raise RoutePolicyError(f"{qid}: selected layer counts do not match final evidence")
    missing = sorted(layer for layer in required_layers if actual[layer] <= 0)
    if missing:
        raise RoutePolicyError(
            f"{qid}: production evidence omitted required layers: {','.join(missing)}"
        )
    for layer in ("source", "fast", "slow"):
        if inventory_counts[layer] > 0 and paths.get(layer) is not True:
            raise RoutePolicyError(f"{qid}: available {layer} candidate path did not execute")

    if actual["slow"] > 0:
        slow_contexts = [
            context
            for window in windows
            for context in list(window.get("memory_contexts") or [])
            if isinstance(context, Mapping) and _text(context.get("role")) == "slow_context"
        ]
        if not slow_contexts or any(
            not _text(context.get("capsule_id"))
            or not _text(context.get("claim_id"))
            or not _text(context.get("claim_text"))
            or not isinstance(context.get("provenance"), Mapping)
            for context in slow_contexts
        ):
            raise RoutePolicyError(f"{qid}: slow contribution lacks auditable claim context")
    if actual["fast"] > 0 and inventory_counts["fast_semantic"] > 0:
        fast_contexts = [
            attachment
            for window in windows
            for attachment in list(window.get("attachments") or [])
            if isinstance(attachment, Mapping)
            and _text(attachment.get("role")) in {"fast_context", "override"}
        ]
        if not fast_contexts or any(
            not _text(context.get("memory_id"))
            or not _text(context.get("canonical_slot"))
            or not _text(context.get("text"))
            or not isinstance(context.get("provenance"), Mapping)
            for context in fast_contexts
        ):
            raise RoutePolicyError(f"{qid}: fast contribution lacks auditable semantic context")
        if any(
            _text(context.get("role")) == "override"
            and _text(context.get("precedence")) != "newer_fast_evidence"
            for context in fast_contexts
        ):
            raise RoutePolicyError(
                f"{qid}: Fast override lacks controller-verified precedence"
            )
    return actual


def _validate_retrieval_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_lane: str
) -> dict[str, Any]:
    if not rows:
        raise RoutePolicyError("retrieval evidence is empty")
    qids: set[str] = set()
    layer_window_counts = {"source": 0, "fast": 0, "slow": 0}
    for index, row in enumerate(rows):
        qid = _text(row.get("question_id"))
        if not qid or qid in qids:
            raise RoutePolicyError(
                f"retrieval evidence row {index} has a missing or duplicate question_id"
            )
        qids.add(qid)
        leaked = sorted(FORBIDDEN_ANSWER_FACING_FIELDS.intersection(row))
        if leaked:
            raise RoutePolicyError(
                f"{qid}: answer-facing retrieval contains benchmark fields: {', '.join(leaked)}"
            )
        windows = row.get("evidence_windows")
        if (
            not isinstance(windows, Sequence)
            or isinstance(windows, (str, bytes))
            or not windows
        ):
            raise RoutePolicyError(f"{qid}: retrieval evidence windows are missing")
        typed_windows = [item for item in windows if isinstance(item, Mapping)]
        if len(typed_windows) != len(windows):
            raise RoutePolicyError(f"{qid}: retrieval evidence contains a non-object window")
        counts = _validate_retrieval_contract(
            row,
            typed_windows,
            qid=qid,
            expected_lane=expected_lane,
        )
        for layer, count in counts.items():
            layer_window_counts[layer] += count
    return {
        "schema_version": RETRIEVAL_CONTRACT_SCHEMA,
        "execution_lane": expected_lane,
        "question_count": len(qids),
        "layer_window_counts": layer_window_counts,
        "ready_for_evidence_compilation": True,
    }


def validate_production_retrieval_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _validate_retrieval_rows(rows, expected_lane="production")


def validate_diagnostic_retrieval_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _validate_retrieval_rows(rows, expected_lane="diagnostic")


def validate_production_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RoutePolicyError("production evidence is empty")
    qids: list[str] = []
    source_window_count = 0
    layer_window_counts = {"source": 0, "fast": 0, "slow": 0}
    for index, row in enumerate(rows):
        qid = _text(row.get("question_id"))
        if not qid or qid in qids:
            raise RoutePolicyError(f"production evidence row {index} has a missing or duplicate question_id")
        qids.append(qid)
        if SHADOW_PACKET_FIELD in row:
            raise RoutePolicyError(f"{qid}: semantic shadow packet is forbidden in the production lane")
        packet = row.get(PRODUCTION_PACKET_FIELD)
        if not isinstance(packet, Mapping) or packet.get("schema_version") != PACKET_SCHEMA:
            raise RoutePolicyError(f"{qid}: production lane requires a compiled evidence packet")
        if _text(packet.get("question_id")) != qid:
            raise RoutePolicyError(f"{qid}: compiled packet identity mismatch")
        if packet.get("packet_compiler_version") != PACKET_COMPILER_VERSION:
            raise RoutePolicyError(f"{qid}: compiled packet compiler contract is stale")
        question_contract = packet.get("question_contract")
        operation_plan = packet.get("operation_plan")
        requirement_coverage = packet.get("requirement_coverage")
        operation_results = packet.get("operation_results")
        evidence_bundles = packet.get("evidence_bundles")
        if not isinstance(question_contract, Mapping):
            raise RoutePolicyError(f"{qid}: compiled packet question contract is missing")
        if not _text(question_contract.get("question")):
            raise RoutePolicyError(f"{qid}: compiled packet question contract is incomplete")
        if _text(row.get("question")) and _text(question_contract.get("question")) != _text(
            row.get("question")
        ):
            raise RoutePolicyError(f"{qid}: compiled packet question contract drifted")
        if (
            not isinstance(operation_plan, Mapping)
            or operation_plan.get("schema_version") != PLAN_SCHEMA
        ):
            raise RoutePolicyError(f"{qid}: compiled packet operation plan is missing")
        for field, value in (
            ("requirements", operation_plan.get("requirements")),
            ("operations", operation_plan.get("operations")),
            ("bundles", operation_plan.get("bundles")),
            ("requirement_coverage", requirement_coverage),
            ("operation_results", operation_results),
            ("evidence_bundles", evidence_bundles),
        ):
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise RoutePolicyError(
                    f"{qid}: compiled packet {field} must be an array"
                )
            if any(not isinstance(item, Mapping) for item in value):
                raise RoutePolicyError(
                    f"{qid}: compiled packet {field} contains a non-object"
                )
        requirement_count = question_contract.get("requirement_count")
        operation_count = question_contract.get("operation_count")
        if requirement_count != len(requirement_coverage):
            raise RoutePolicyError(
                f"{qid}: compiled packet requirement count is inconsistent"
            )
        if operation_count != len(operation_results):
            raise RoutePolicyError(
                f"{qid}: compiled packet operation count is inconsistent"
            )
        windows = row.get("evidence_windows")
        reservoir = packet.get("raw_evidence_reservoir")
        if (
            not isinstance(windows, Sequence)
            or isinstance(windows, (str, bytes))
            or not windows
            or not isinstance(reservoir, Sequence)
            or isinstance(reservoir, (str, bytes))
            or not reservoir
        ):
            raise RoutePolicyError(f"{qid}: Source evidence is missing")
        typed_windows = [item for item in windows if isinstance(item, Mapping)]
        if len(typed_windows) != len(windows):
            raise RoutePolicyError(f"{qid}: production evidence contains a non-object window")
        layer_counts = _validate_retrieval_contract(row, typed_windows, qid=qid)
        for layer, count in layer_counts.items():
            layer_window_counts[layer] += count
        expected = [
            _source_identity(item)
            for item in windows
            if isinstance(item, Mapping)
        ]
        actual = [
            _source_identity(item)
            for item in reservoir
            if isinstance(item, Mapping)
        ]
        if expected != actual:
            raise RoutePolicyError(f"{qid}: compiled packet does not preserve Source evidence exactly")
        graph_fields = ("memory_contexts", "attachments", "provenance", "retrieval_metadata")
        expected_graph = [
            {field: item.get(field) for field in graph_fields}
            for item in typed_windows
        ]
        actual_graph = [
            {field: item.get(field) for field in graph_fields}
            for item in reservoir
            if isinstance(item, Mapping)
        ]
        if _canonical_json(expected_graph) != _canonical_json(actual_graph):
            raise RoutePolicyError(
                f"{qid}: compiled packet does not preserve layered memory context exactly"
            )
        source_window_count += len(expected)
    return {
        "schema_version": POLICY_SCHEMA,
        "lane": PRODUCTION_LANE,
        "question_count": len(qids),
        "source_window_count": source_window_count,
        "layer_window_counts": layer_window_counts,
        "answer_protocol": PRODUCTION_ANSWER_PROTOCOL,
        "answer_model": PRODUCTION_ANSWER_MODEL,
        "promotion_eligible": True,
    }


def validate_production_answers(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RoutePolicyError("production answers are empty")
    for index, row in enumerate(rows):
        qid = _text(row.get("question_id")) or f"row {index}"
        if row.get("answer_protocol") != PRODUCTION_ANSWER_PROTOCOL:
            raise RoutePolicyError(f"{qid}: answer protocol is not production-approved")
        if row.get("answer_model") != PRODUCTION_ANSWER_MODEL:
            raise RoutePolicyError(f"{qid}: answer model is not production-approved")
