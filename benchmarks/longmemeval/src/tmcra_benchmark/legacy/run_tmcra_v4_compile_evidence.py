#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from run_tmcra_v4_build import DEFAULT_WRITER_ENV, _key_pool, _load_shell_environment
from tmcra_v4_evidence_operations import (
    PACKET_COMPILER_VERSION,
    PACKET_SCHEMA,
    PLAN_SCHEMA,
    build_evidence_catalog,
    compile_evidence_packet,
    operation_plan_structural_risks,
    unbound_memory_requirement_ids,
    validate_operation_plan,
)
from tmcra_v4_evidence_planner import (
    DeepSeekEvidenceOperationPlanner,
    EvidencePlannerError,
    PROMPT_VERSION as EVIDENCE_PLANNER_PROMPT_VERSION,
    normalize_planner_output,
)
from tmcra_v4_route_policy import (
    RoutePolicyError,
    validate_diagnostic_retrieval_rows,
    validate_production_retrieval_rows,
)


class EvidenceCompileError(RuntimeError):
    pass


ARTIFACT_BINDING_SCHEMA = "tmcra.v4.evidence-compiler-artifact-binding.1"
REVIEW_POLICY_VERSION = "tmcra.v4.evidence-review-policy.4"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _retrieval_debug_report(
    path: Path,
    expected_qids: Sequence[str],
) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceCompileError(f"retrieval debug is missing: {path}")
    raw = path.read_bytes()
    try:
        rows = [
            json.loads(line)
            for line in raw.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceCompileError(f"retrieval debug is invalid JSONL: {path}") from exc
    actual_qids = [
        _text(row.get("question_id") or row.get("qid"))
        if isinstance(row, Mapping)
        else ""
        for row in rows
    ]
    if actual_qids != list(expected_qids):
        raise EvidenceCompileError(
            "retrieval debug rows do not match frozen evidence qid order"
        )
    return {
        "schema_version": "tmcra.v4.compiled-retrieval-debug-binding.1",
        "source_path": str(path),
        "row_count": len(rows),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _stage_retrieval_debug(
    source: Path,
    out_dir: Path,
    expected_qids: Sequence[str],
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    report = _retrieval_debug_report(source, expected_qids)
    if report["source_sha256"] != expected_sha256:
        raise EvidenceCompileError("retrieval debug changed during evidence compilation")
    raw = source.read_bytes()
    destination = out_dir / "retrieval_debug.jsonl"
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    temporary.write_bytes(raw)
    os.replace(temporary, destination)
    artifact_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    if artifact_sha256 != expected_sha256:
        raise EvidenceCompileError("staged retrieval debug failed its integrity check")
    return {
        **report,
        "artifact_path": destination.name,
        "artifact_sha256": artifact_sha256,
        "status": "staged",
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temporary.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl_atomic(path: Path, row: Mapping[str, Any]) -> None:
    rows = _read_jsonl(path) if path.is_file() else []
    rows.append(dict(row))
    _atomic_jsonl(path, rows)


def _identity(row: Mapping[str, Any]) -> str:
    payload = {
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "question_date": row.get("question_date"),
        "evidence_windows": row.get("evidence_windows"),
        # Retrieval is part of the compiler input. In particular, a diagnostic
        # row must never collide with a production row for the same question.
        "retrieval_contract": row.get("retrieval_contract"),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _planner_contract(
    *,
    provider: str,
    model: str,
    review_policy_version: str = REVIEW_POLICY_VERSION,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "prompt_version": EVIDENCE_PLANNER_PROMPT_VERSION,
        "review_policy_version": review_policy_version,
        "plan_schema": PLAN_SCHEMA,
    }


def _packet_compiler_contract(*, compiler_version: str = PACKET_COMPILER_VERSION) -> dict[str, Any]:
    return {
        "packet_schema": PACKET_SCHEMA,
        "compiler_version": compiler_version,
    }


def _artifact_binding(
    row: Mapping[str, Any],
    *,
    provider: str,
    model: str,
    compiler_version: str = PACKET_COMPILER_VERSION,
    review_policy_version: str = REVIEW_POLICY_VERSION,
) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_BINDING_SCHEMA,
        "input_sha256": _identity(row),
        "input": {
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "question_date": row.get("question_date"),
            "evidence_windows": row.get("evidence_windows"),
            "retrieval_route_contract": row.get("retrieval_contract"),
        },
        "planner": _planner_contract(
            provider=provider,
            model=model,
            review_policy_version=review_policy_version,
        ),
        "packet_compiler": _packet_compiler_contract(compiler_version=compiler_version),
    }


def _binding_sha256(binding: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(binding), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _saved_binding(
    saved: Mapping[str, Any],
    saved_row: Mapping[str, Any],
) -> dict[str, Any]:
    binding = saved.get("artifact_binding")
    if isinstance(binding, Mapping):
        return dict(binding)
    planner = saved.get("planner")
    if not isinstance(planner, Mapping):
        raise EvidenceCompileError("completed artifact lacks its planner contract")
    provider = _text(planner.get("provider"))
    model = _text(planner.get("model"))
    if not provider or not model:
        raise EvidenceCompileError("completed artifact lacks its planner provider/model binding")
    packet = saved_row.get("compiled_evidence_packet")
    persisted_plan_schema = (
        packet.get("operation_plan", {}).get("schema_version")
        if isinstance(packet, Mapping)
        and isinstance(packet.get("operation_plan"), Mapping)
        else None
    )
    if persisted_plan_schema != PLAN_SCHEMA:
        raise EvidenceCompileError("completed artifact planner contract schema is stale")
    # Legacy artifacts did not persist this envelope. Reconstructing it is
    # safe only from the persisted row and planner metadata; both are checked
    # against the current invocation before reuse.
    packet_compiler_version = (
        _text(packet.get("packet_compiler_version"))
        if isinstance(packet, Mapping)
        else ""
    )
    return _artifact_binding(
        saved_row,
        provider=provider,
        model=model,
        compiler_version=packet_compiler_version or "legacy-unknown",
        review_policy_version=(
            _text(planner.get("review_policy_version")) or "legacy-unknown"
        ),
    )


def _assert_artifact_binding(
    saved: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    input_sha256: str,
    expected_binding: Mapping[str, Any],
) -> None:
    saved_row = saved.get("row")
    if not isinstance(saved_row, Mapping):
        raise EvidenceCompileError("completed artifact lacks its persisted input row")
    if _identity(saved_row) != input_sha256:
        raise EvidenceCompileError("persisted compiler identity mismatch")
    current_binding = dict(expected_binding)
    persisted_binding = _saved_binding(saved, saved_row)
    if persisted_binding.get("schema_version") not in {None, ARTIFACT_BINDING_SCHEMA}:
        raise EvidenceCompileError("completed artifact binding schema is stale")
    if persisted_binding.get("input_sha256") != input_sha256:
        raise EvidenceCompileError("persisted compiler identity mismatch")
    if persisted_binding.get("input") != current_binding.get("input"):
        raise EvidenceCompileError("completed artifact input contract mismatch")
    if persisted_binding.get("planner") != current_binding.get("planner"):
        raise EvidenceCompileError("completed artifact planner contract/provider/model mismatch")
    persisted_packet_contract = persisted_binding.get("packet_compiler")
    current_packet_contract = current_binding.get("packet_compiler")
    if not isinstance(persisted_packet_contract, Mapping) or not isinstance(current_packet_contract, Mapping):
        raise EvidenceCompileError("completed artifact packet compiler contract is missing")
    if persisted_packet_contract.get("packet_schema") != current_packet_contract.get("packet_schema"):
        raise EvidenceCompileError("completed artifact packet schema contract mismatch")
    if row.get("retrieval_contract") != saved_row.get("retrieval_contract"):
        raise EvidenceCompileError("completed artifact retrieval route contract mismatch")

    persisted_packet = saved_row.get("compiled_evidence_packet")
    if isinstance(persisted_packet, Mapping) and persisted_packet.get("schema_version") != PACKET_SCHEMA:
        raise EvidenceCompileError("completed artifact packet schema is stale")


def _cost_cny(calls: Sequence[Mapping[str, Any]]) -> float | None:
    if any(call.get("provider") != "deepseek" for call in calls):
        return None
    total = 0.0
    for call in calls:
        usage = dict(call.get("usage") or {})
        total += (
            int(usage.get("prompt_cache_miss_tokens", 0)) * 3.0
            + int(usage.get("completion_tokens", 0)) * 6.0
            + int(usage.get("prompt_cache_hit_tokens", 0)) * 0.025
        ) / 1_000_000.0
    return round(total, 8)


def _physical_calls(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        for key in ("calls", "prior_calls"):
            nested = value.get(key)
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                for item in nested:
                    visit(item)
        call_id = _text(value.get("physical_call_id"))
        if value.get("physical_api_call") is True and call_id:
            calls.setdefault(call_id, dict(value))

    for value in values:
        visit(value)
    return list(calls.values())


def _aggregate_planner_calls(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not calls:
        raise EvidenceCompileError("planner call list is empty")
    usage_keys = ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "total_tokens")
    usage = {key: sum(int((call.get("usage") or {}).get(key, 0)) for call in calls) for key in usage_keys}
    return {
        **dict(calls[-1]),
        "physical_api_calls": len(calls),
        "usage": usage,
        "review_applied": len(calls) > 1,
        "calls": [dict(call) for call in calls],
    }


RELATIVE_EVENT_REFERENCE_PATTERN = re.compile(
    r"\b(?:when|before|after)\b", re.IGNORECASE
)
RELATIVE_DISTANCE_PATTERN = re.compile(
    r"\b(?:ago|before|after)\b", re.IGNORECASE
)


def _relative_event_anchor_review_needed(
    plan: Mapping[str, Any], catalog: Mapping[str, Any] | None
) -> bool:
    if not isinstance(catalog, Mapping):
        return False
    question = _text(catalog.get("question"))
    if not (
        RELATIVE_EVENT_REFERENCE_PATTERN.search(question)
        and RELATIVE_DISTANCE_PATTERN.search(question)
    ):
        return False
    evidence_by_id = {
        _text(item.get("evidence_id")): item
        for item in catalog.get("evidence") or []
        if isinstance(item, Mapping)
    }
    anchor_ids = [
        _text(value)
        for value in catalog.get("lexical_anchor_ids") or []
        if _text(value) in evidence_by_id
    ]
    anchor_sessions = {
        _text(evidence_by_id[evidence_id].get("session_id"))
        for evidence_id in anchor_ids
    }
    if len(anchor_sessions - {""}) < 2:
        return False
    question_atom_ids = {
        _text(item.get("atom_id"))
        for item in catalog.get("atoms") or []
        if isinstance(item, Mapping) and item.get("evidence_id") == "QUESTION"
    }
    for operation in plan.get("operations") or []:
        if (
            not isinstance(operation, Mapping)
            or operation.get("operation_type") != "date_difference"
            or not question_atom_ids.intersection(operation.get("input_atom_ids") or [])
        ):
            continue
        bound_ids = {
            _text(value) for value in operation.get("input_evidence_ids") or []
        }
        bound_sessions = {
            _text(evidence_by_id[value].get("session_id"))
            for value in bound_ids
            if value in evidence_by_id
        }
        if any(
            evidence_id not in bound_ids
            and _text(evidence_by_id[evidence_id].get("session_id"))
            not in bound_sessions
            for evidence_id in anchor_ids
        ):
            return True
    return False


def _required_memory_premise_binding(
    plan: Mapping[str, Any], catalog: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if not isinstance(catalog, Mapping):
        return None
    anchor_ids = {
        _text(value)
        for value in catalog.get("lexical_anchor_ids") or []
        if _text(value)
    }
    if not anchor_ids:
        return None
    required_memory_ids = {
        _text(premise.get("premise_id"))
        for premise in (plan.get("task_contract") or {}).get("premises") or []
        if isinstance(premise, Mapping)
        and premise.get("source") == "memory"
        and premise.get("necessity") == "required"
        and _text(premise.get("premise_id"))
    }
    if not required_memory_ids:
        return None
    bound_ids = {
        _text(value)
        for requirement in plan.get("requirements") or []
        if isinstance(requirement, Mapping)
        and _text(requirement.get("requirement_id")) in required_memory_ids
        for value in requirement.get("evidence_ids") or []
        if _text(value)
    }
    if not bound_ids or bound_ids.intersection(anchor_ids):
        return None
    return {
        "required_memory_requirement_ids": sorted(required_memory_ids),
        "bound_evidence_ids": sorted(bound_ids),
        "lexical_anchor_ids": sorted(anchor_ids),
    }


def _review_context_for_plan(
    plan: Mapping[str, Any], catalog: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    missing_requirement_ids = unbound_memory_requirement_ids(plan)
    structural_risks = operation_plan_structural_risks(plan)
    relative_event_anchor_risk = _relative_event_anchor_review_needed(plan, catalog)
    memory_relevance_signal = _required_memory_premise_binding(plan, catalog)
    review_reasons = [
        *structural_risks,
        *(["unbound_required_memory_premises"] if missing_requirement_ids else []),
        *(["relative_event_anchor_not_bound"] if relative_event_anchor_risk else []),
        *(
            ["required_memory_premise_outside_query_anchors"]
            if memory_relevance_signal
            else []
        ),
    ]
    if not review_reasons:
        return None
    relative_anchor_instruction = (
        " This review has already established that another remembered event "
        "is the relative-time endpoint. QUESTION is forbidden as a "
        "date_difference input; bind at least two distinct Source event dates."
        if relative_event_anchor_risk
        else ""
    )
    memory_relevance_instruction = (
        " The required memory premise is currently bound only to evidence that "
        "falls outside the query's lexical anchors. Do not satisfy the memory "
        "contract with an unrelated remembered detail. Reinspect the candidate "
        "evidence, including user-authored Source text and source-group context, "
        "and bind memory that materially constrains the answer target. Lexical "
        "anchors are review candidates rather than a hard allowlist, so an "
        "implicitly related memory may remain only when the rebuilt plan changes "
        "the evidence binding or adds relevant corroborating evidence."
        if memory_relevance_signal
        else ""
    )
    hard_constraints: dict[str, Any] = {}
    if relative_event_anchor_risk:
        hard_constraints.update(
            {
                "forbidden_date_difference_evidence_ids": ["QUESTION"],
                "minimum_distinct_source_event_anchors": 2,
            }
        )
    if memory_relevance_signal:
        hard_constraints["forbidden_required_memory_evidence_ids"] = list(
            memory_relevance_signal["bound_evidence_ids"]
        )
    return {
        "initial_plan": dict(plan),
        "review_reasons": review_reasons,
        "missing_requirement_ids": missing_requirement_ids,
        "memory_premise_relevance_signal": memory_relevance_signal or {},
        "hard_constraints": hard_constraints,
        "instruction": (
            "Rebuild the complete plan. Treat a missing state as a hypothesis, "
            "not proof of absence; for memory-conditioned generation, bind the "
            "remembered constraints instead of searching for a ready-made answer. "
            "When a relative-time question names another remembered event with "
            "when, before, or after, compare the distinct event anchors and do not "
            "default to the question date unless the wording explicitly makes it "
            "the comparison endpoint."
            + relative_anchor_instruction
            + memory_relevance_instruction
        ),
    }


def _review_resolution_failure(
    plan: Mapping[str, Any],
    catalog: Mapping[str, Any] | None,
    review_context: Mapping[str, Any] | None,
) -> str:
    reasons = set((review_context or {}).get("review_reasons") or [])
    if (
        "relative_event_anchor_not_bound" in reasons
        and _relative_event_anchor_review_needed(plan, catalog)
    ):
        return (
            "relative event-anchor review still binds QUESTION instead of "
            "two distinct Source event anchors"
        )
    if "required_memory_premise_outside_query_anchors" in reasons:
        current_signal = _required_memory_premise_binding(plan, catalog)
        initial_signal = (review_context or {}).get(
            "memory_premise_relevance_signal"
        ) or {}
        if current_signal and set(current_signal.get("bound_evidence_ids") or []) == set(
            initial_signal.get("bound_evidence_ids") or []
        ):
            return (
                "required memory-premise relevance review kept the same "
                "out-of-anchor evidence binding"
            )
    return ""


def _recover_plan_from_failure(
    failure_path: Path,
    *,
    question_id: str,
    input_sha256: str,
    catalog: Mapping[str, Any],
    expected_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not failure_path.is_file():
        return None
    persisted = json.loads(failure_path.read_text(encoding="utf-8"))
    if (
        persisted.get("question_id") != question_id
        or persisted.get("input_sha256") != input_sha256
    ):
        raise EvidenceCompileError(
            f"{question_id}: persisted compiler failure identity mismatch"
        )
    metadata = dict(persisted.get("planner") or {})
    if _text(metadata.get("prompt_version")) != EVIDENCE_PLANNER_PROMPT_VERSION:
        return None
    if _text(metadata.get("review_policy_version")) != REVIEW_POLICY_VERSION:
        return None
    if expected_binding is not None:
        expected_planner = expected_binding.get("planner")
        if not isinstance(expected_planner, Mapping) or any(
            metadata.get(field) != expected_planner.get(field)
            for field in ("provider", "model", "prompt_version")
        ):
            return None
    raw_plan = metadata.get("raw_plan")
    if not isinstance(raw_plan, Mapping) and metadata.get(
        "review_resolution_failed"
    ):
        review_context = metadata.get("review_context")
        if isinstance(review_context, Mapping):
            raw_plan = review_context.get("initial_plan")
    if not isinstance(raw_plan, Mapping):
        return None
    try:
        plan, warnings = normalize_planner_output(raw_plan, catalog)
    except Exception:
        # The journal is an optimization, not a permanent failure latch. A raw
        # response that cannot satisfy the current contract must be replanned.
        return None
    return plan, {
        **metadata,
        "raw_plan": dict(raw_plan),
        "recovered_from_persisted_raw_plan": True,
        "normalization_warnings": warnings,
    }


def _replan_context_from_failure(
    failure_path: Path,
    *,
    question_id: str,
    input_sha256: str,
    expected_binding: Mapping[str, Any] | None = None,
    catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not failure_path.is_file():
        return None
    persisted = json.loads(failure_path.read_text(encoding="utf-8"))
    if (
        persisted.get("question_id") != question_id
        or persisted.get("input_sha256") != input_sha256
    ):
        raise EvidenceCompileError(
            f"{question_id}: persisted compiler failure identity mismatch"
        )
    metadata = dict(persisted.get("planner") or {})
    if _text(metadata.get("prompt_version")) != EVIDENCE_PLANNER_PROMPT_VERSION:
        return None
    if expected_binding is not None:
        expected_planner = expected_binding.get("planner")
        if not isinstance(expected_planner, Mapping) or any(
            metadata.get(field) != expected_planner.get(field)
            for field in ("provider", "model", "prompt_version")
        ):
            return None
    raw_plan = metadata.get("raw_plan")
    error = _text(persisted.get("error"))
    if not isinstance(raw_plan, Mapping) or not error:
        return None
    candidate_memory_evidence_ids = [
        _text(value)
        for value in (catalog or {}).get("lexical_anchor_ids") or []
        if _text(value)
    ]
    return {
        "initial_plan": dict(raw_plan),
        "review_reasons": ["persisted_plan_validation_failure"],
        "prior_validation_error": error,
        "candidate_memory_evidence_ids": candidate_memory_evidence_ids,
        "instruction": (
            "Rebuild the complete plan and correct the prior runtime contract "
            f"validation failure: {error}. Inspect the supplied evidence again. "
            "For memory-conditioned generation, bind at least one relevant "
            "required memory preference, constraint, owned tools, or resources. "
            "For recommendation or advice, a previously mentioned owned tool or "
            "accessory that can directly address the target problem is a valid "
            "memory premise. Do not choose an unrelated preference merely because "
            "it is more explicit; the premise must share the target domain or "
            "intended action with the current question. Reinspect the listed "
            "candidate_memory_evidence_ids plus their user-authored Source and "
            "source-group context, and bind the relevant parent Evidence ID. "
            "The candidate list is a review hint, not a hard allowlist. If no "
            "relevant memory is supportable, "
            "preserve an unbound memory requirement instead of relabeling query "
            "context as remembered evidence."
        ),
    }


def _refresh_completed_artifact(
    saved: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    question_id: str,
    input_sha256: str,
    expected_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    if (
        saved.get("question_id") != question_id
    ):
        raise EvidenceCompileError(
            f"{question_id}: persisted compiler identity mismatch"
        )
    saved_row = saved.get("row")
    if saved.get("input_sha256") != input_sha256 and (
        not isinstance(saved_row, Mapping) or _identity(saved_row) != input_sha256
    ):
        raise EvidenceCompileError(
            f"{question_id}: persisted compiler identity mismatch"
        )
    output = dict(saved)
    if expected_binding is None:
        persisted_binding = saved.get("artifact_binding")
        planner = (
            persisted_binding.get("planner")
            if isinstance(persisted_binding, Mapping)
            else saved.get("planner")
        )
        if not isinstance(planner, Mapping):
            raise EvidenceCompileError(
                f"{question_id}: completed artifact lacks its planner contract"
            )
        expected_binding = _artifact_binding(
            row,
            provider=_text(planner.get("provider")),
            model=_text(planner.get("model")),
        )
    _assert_artifact_binding(
        saved,
        row,
        input_sha256=input_sha256,
        expected_binding=expected_binding,
    )
    planner_metadata = output.get("planner")
    if (
        not isinstance(planner_metadata, Mapping)
        or _text(planner_metadata.get("prompt_version"))
        != EVIDENCE_PLANNER_PROMPT_VERSION
    ):
        raise EvidenceCompileError(
            f"{question_id}: completed artifact uses a stale evidence planner contract"
        )
    saved_row = output.get("row")
    packet = saved_row.get("compiled_evidence_packet") if isinstance(saved_row, Mapping) else None
    if not isinstance(packet, Mapping):
        raise EvidenceCompileError(
            f"{question_id}: completed artifact lacks a compiled evidence packet"
        )
    binding_changed = output.get("artifact_binding") != dict(expected_binding)
    if packet.get("packet_compiler_version") == PACKET_COMPILER_VERSION:
        if binding_changed:
            output["artifact_binding"] = dict(expected_binding)
            output["compiler_binding_sha256"] = _binding_sha256(expected_binding)
        return output, binding_changed
    plan = packet.get("operation_plan")
    if not isinstance(plan, Mapping):
        raise EvidenceCompileError(
            f"{question_id}: completed artifact lacks its persisted operation plan"
        )
    output_row = dict(row)
    output_row["compiled_evidence_packet"] = compile_evidence_packet(row, plan)
    output["row"] = output_row
    output["packet_compiler_version"] = PACKET_COMPILER_VERSION
    output["artifact_binding"] = dict(expected_binding)
    output["compiler_binding_sha256"] = _binding_sha256(expected_binding)
    output["local_packet_recompile_count"] = int(
        output.get("local_packet_recompile_count", 0)
    ) + 1
    return output, True


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile immutable TMCRA evidence into operation-bound answer packets")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qid-list", type=Path)
    parser.add_argument(
        "--retrieval-debug",
        type=Path,
        help="retrieval_debug.jsonl to bind into the compiled evaluation artifact",
    )
    parser.add_argument(
        "--require-retrieval-debug",
        action="store_true",
        help="fail before planner calls unless retrieval debug is present and qid-aligned",
    )
    parser.add_argument("--writer-env", type=Path, default=DEFAULT_WRITER_ENV)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--planner-provider", choices=("deepseek", "xiaomi_mimo"), default="deepseek")
    parser.add_argument("--planner-model")
    parser.add_argument("--planner-base-url")
    parser.add_argument("--planner-key-file", type=Path)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="explicitly compile a diagnostic retrieval lane that cannot enter production evaluation",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise EvidenceCompileError("workers must be positive")
    rows = _read_jsonl(args.evidence.resolve())
    if args.qid_list:
        qids = [line.strip() for line in args.qid_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_qid = {_text(row.get("question_id")): row for row in rows}
        if not qids or len(qids) != len(set(qids)) or any(qid not in by_qid for qid in qids):
            raise EvidenceCompileError("qid list is empty, duplicated, or absent from evidence")
        rows = [by_qid[qid] for qid in qids]
    try:
        route_report = (
            validate_diagnostic_retrieval_rows(rows)
            if args.diagnostic
            else validate_production_retrieval_rows(rows)
        )
    except RoutePolicyError as exc:
        raise EvidenceCompileError(
            f"retrieval route failed before evidence planner calls: {exc}"
        ) from exc
    expected_qids = [_text(row.get("question_id")) for row in rows]
    retrieval_debug_source = (
        args.retrieval_debug.resolve()
        if args.retrieval_debug
        else args.evidence.resolve().parent / "retrieval_debug.jsonl"
    )
    retrieval_debug_preflight: dict[str, Any] | None = None
    if retrieval_debug_source.is_file():
        retrieval_debug_preflight = _retrieval_debug_report(
            retrieval_debug_source,
            expected_qids,
        )
    elif args.require_retrieval_debug:
        raise EvidenceCompileError(
            f"retrieval debug is required before planner calls: {retrieval_debug_source}"
        )
    if args.planner_provider == "deepseek":
        environment = _load_shell_environment(args.writer_env.resolve())
        keys = _key_pool(environment)
        base_url = args.planner_base_url or environment.get("TMCRA_DEEPSEEK_WRITER_BASE_URL") or environment.get("TMCRA_WRITER_BASE_URL") or "https://api.deepseek.com/v1"
        planner_model = args.planner_model or "deepseek-v4-pro"
    else:
        if not args.planner_key_file or not args.planner_key_file.is_file():
            raise EvidenceCompileError("xiaomi_mimo requires --planner-key-file")
        key = args.planner_key_file.read_text(encoding="utf-8").strip()
        if not key:
            raise EvidenceCompileError("planner key file is empty")
        keys = [key]
        base_url = args.planner_base_url or "https://api.xiaomimimo.com/v1"
        planner_model = args.planner_model or "mimo-v2.5"
    out_dir = args.out_dir.resolve()
    journal_dir = out_dir / "rows"
    journal_dir.mkdir(parents=True, exist_ok=True)
    failure_history_path = out_dir / "planner_failure_history.jsonl"
    lock = threading.Lock()

    def compile_one(index: int, row: Mapping[str, Any]) -> dict[str, Any]:
        qid = _text(row.get("question_id"))
        identity = _identity(row)
        binding = _artifact_binding(
            row,
            provider=args.planner_provider,
            model=planner_model,
        )
        artifact = journal_dir / f"{index:06d}_{qid}.json"
        failure_path = journal_dir / f"{index:06d}_{qid}.failure.json"
        if artifact.is_file():
            saved = json.loads(artifact.read_text(encoding="utf-8"))
            refreshed, changed = _refresh_completed_artifact(
                saved,
                row,
                question_id=qid,
                input_sha256=identity,
                expected_binding=binding,
            )
            if changed:
                with lock:
                    _atomic_json(artifact, refreshed)
            return refreshed
        catalog = build_evidence_catalog(row)
        planner = DeepSeekEvidenceOperationPlanner(
            base_url=base_url,
            api_keys=[keys[index % len(keys)]],
            timeout=args.timeout,
            model=planner_model,
            provider=args.planner_provider,
        )
        planner_calls: list[dict[str, Any]] = []
        recovered_from_persisted_raw_plan = False
        try:
            persisted_replan_context = _replan_context_from_failure(
                failure_path,
                question_id=qid,
                input_sha256=identity,
                expected_binding=binding,
                catalog=catalog,
            )
            recovered = _recover_plan_from_failure(
                failure_path,
                question_id=qid,
                input_sha256=identity,
                catalog=catalog,
                expected_binding=binding,
            )
            if recovered is not None:
                plan, recovered_metadata = recovered
                recovered_from_persisted_raw_plan = True
                planner_calls.append(recovered_metadata)
            if not recovered_from_persisted_raw_plan:
                plan, metadata = planner.plan(
                    catalog,
                    review_context=persisted_replan_context,
                )
                planner_calls.append(metadata)
            review_context = _review_context_for_plan(plan, catalog)
            if review_context is not None:
                plan, review_metadata = planner.plan(
                    catalog,
                    review_context=review_context,
                )
                planner_calls.append(review_metadata)
                review_failure = _review_resolution_failure(
                    plan,
                    catalog,
                    review_context,
                )
                if review_failure:
                    raise EvidencePlannerError(
                        review_failure,
                        metadata={
                            **review_metadata,
                            "review_resolution_failed": True,
                            "review_context": review_context,
                        },
                    )
            metadata = _aggregate_planner_calls(planner_calls)
            metadata["review_policy_version"] = REVIEW_POLICY_VERSION
            metadata["recovered_from_persisted_raw_plan"] = (
                recovered_from_persisted_raw_plan
            )
            metadata["new_physical_api_calls_this_resume"] = len(planner_calls) - int(
                recovered_from_persisted_raw_plan
            )
        except EvidencePlannerError as exc:
            failure = {
                "failure_id": f"ecf_{time.time_ns()}",
                "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "question_id": qid,
                "input_sha256": identity,
                "artifact_binding": binding,
                "compiler_binding_sha256": _binding_sha256(binding),
                "status": "failed",
                "error": str(exc),
                "planner": {
                    **exc.metadata,
                    "review_policy_version": REVIEW_POLICY_VERSION,
                    **({"prior_calls": planner_calls} if planner_calls else {}),
                },
            }
            with lock:
                _atomic_json(failure_path, failure)
                _append_jsonl_atomic(failure_history_path, failure)
            raise
        plan = validate_operation_plan(plan, catalog)
        packet = compile_evidence_packet(row, plan)
        output_row = dict(row)
        output_row["compiled_evidence_packet"] = packet
        result = {
            "question_id": qid,
            "input_sha256": identity,
            "artifact_binding": binding,
            "compiler_binding_sha256": _binding_sha256(binding),
            "status": "completed",
            "planner": metadata,
            "row": output_row,
        }
        with lock:
            _atomic_json(artifact, result)
            failure_path.unlink(missing_ok=True)
        return result

    results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(compile_one, index, row): _text(row.get("question_id")) for index, row in enumerate(rows)}
        for future in as_completed(futures):
            qid = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append({"question_id": qid, "error": f"{exc.__class__.__name__}: {exc}"})
            else:
                results[qid] = result
    if failures:
        _atomic_json(out_dir / "failures.json", {"failures": failures})
        raise EvidenceCompileError(f"evidence compilation failed for {len(failures)} rows")
    ordered_results = [results[_text(row.get("question_id"))] for row in rows]
    (out_dir / "failures.json").unlink(missing_ok=True)
    ordered_rows = [item["row"] for item in ordered_results]
    calls = [item["planner"] for item in ordered_results]
    failure_history = _read_jsonl(failure_history_path) if failure_history_path.is_file() else []
    completed_physical_calls = _physical_calls(calls)
    all_physical_calls = _physical_calls(
        [*calls, *(dict(item.get("planner") or {}) for item in failure_history)]
    )
    _atomic_jsonl(out_dir / "evidence_windows.jsonl", ordered_rows)
    retrieval_debug_report = (
        _stage_retrieval_debug(
            retrieval_debug_source,
            out_dir,
            expected_qids,
            expected_sha256=retrieval_debug_preflight["source_sha256"],
        )
        if retrieval_debug_preflight is not None
        else {
            "schema_version": "tmcra.v4.compiled-retrieval-debug-binding.1",
            "status": "not_provided",
        }
    )
    _atomic_json(
        out_dir / "report.json",
        {
            "schema_version": "tmcra.v4.evidence-compiler-run.1",
            "status": "complete",
            "row_count": len(ordered_rows),
            "physical_call_count": len(all_physical_calls),
            "completed_artifact_physical_call_count": len(completed_physical_calls),
            "failure_history_record_count": len(failure_history),
            "locally_recompiled_row_count": sum(
                int(item.get("local_packet_recompile_count", 0) > 0)
                for item in ordered_results
            ),
            "packet_compiler_version": PACKET_COMPILER_VERSION,
            "planner_provider": args.planner_provider,
            "planner_model": planner_model,
            "retrieval_route_report": route_report,
            "retrieval_debug": retrieval_debug_report,
            "operation_count": sum(len(row["compiled_evidence_packet"]["operation_results"]) for row in ordered_rows),
            "requirement_count": sum(len(row["compiled_evidence_packet"]["requirement_coverage"]) for row in ordered_rows),
            "structural_risk_count": sum(
                len(row["compiled_evidence_packet"].get("structural_risk_signals") or [])
                for row in ordered_rows
            ),
            "typed_semantics_accepted_count": sum(
                len((row["compiled_evidence_packet"].get("typed_semantics_report") or {}).get("accepted") or [])
                for row in ordered_rows
            ),
            "typed_semantics_rejected_count": sum(
                len((row["compiled_evidence_packet"].get("typed_semantics_report") or {}).get("rejected") or [])
                for row in ordered_rows
            ),
            "exact_cost_cny": _cost_cny(all_physical_calls),
        },
    )
    (out_dir / "COMPILE_COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
