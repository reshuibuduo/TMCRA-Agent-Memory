#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from tmcra_v4_evidence_operations import (
    ANSWER_SCHEMA,
    PACKET_SCHEMA,
    EvidenceOperationError,
    validate_evidence_bound_answer,
)
from tmcra_v4_semantic_evidence import (
    SEMANTIC_ANSWER_SCHEMA,
    SEMANTIC_PACKET_SCHEMA,
    SemanticEvidenceError,
    normalize_semantic_answer_output,
    semantic_packet_to_advisory_packet,
    validate_semantic_answer,
)
from tmcra_v4_route_policy import (
    PRODUCTION_ANSWER_PROTOCOL,
    RoutePolicyError,
    SHADOW_ANSWER_PROTOCOL,
    validate_production_answers,
    validate_production_evidence,
)


FORBIDDEN_EVALUATION_FIELDS = {
    "answer",
    "gold_answer",
    "answer_session_ids",
    "labels",
    "supervision",
}
USER_MEMORY_CUE_LIMIT = 32
USER_MEMORY_CUE_TEXT_LIMIT = 640


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def fixed_answer_environment(answer_window_limit: int) -> dict[str, str]:
    return {
        "TMCRA_ANSWER_WINDOW_PLANNER_MODE": "off",
        "TMCRA_ANSWER_WINDOW_PLANNER_SEMANTIC_MODE": "off",
        "TMCRA_LLM_EVIDENCE_SELECTOR_MODE": "off",
        "TMCRA_EVIDENCE_UNIT_PLANNER_MODE": "off",
        "TMCRA_UNIFIED_OPERATION_PLANNER_MODE": "off",
        "TMCRA_LLM_CHANNEL_PLANNER_MODE": "off",
        "TMCRA_FINAL_ANSWER_SURFACE_CHANNEL_MODE": "off",
        "TMCRA_ANSWER_REASONING_ORGANIZER": "off",
        "TMCRA_LLM_CACHE_MODE": "off",
        "TMCRA_LME_FORCE_ANSWER": "1",
        "TMCRA_HTTP_JSON_MAX_ATTEMPTS": "1",
        "TMCRA_ANSWER_EVIDENCE_WINDOW_LIMIT": str(max(1, answer_window_limit)),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"row is not an object at {path}:{line_no}")
            rows.append(value)
    if not rows:
        raise RuntimeError(f"no rows: {path}")
    return rows


def evidence_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def atomic_write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def bind_existing_answers_to_evidence(
    existing: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, str], int]:
    evidence_by_qid = {clean_text(row.get("question_id")): row for row in rows}
    if not all(evidence_by_qid) or len(evidence_by_qid) != len(rows):
        raise RuntimeError("answer evidence contains missing or duplicate question_id")
    hashes = {qid: evidence_sha256(row) for qid, row in evidence_by_qid.items()}
    backfills = 0
    for qid, row in existing.items():
        if qid not in evidence_by_qid:
            continue
        expected = evidence_by_qid[qid]
        expected_hash = hashes[qid]
        observed_hash = clean_text(row.get("evidence_sha256"))
        if observed_hash and observed_hash != expected_hash:
            raise RuntimeError(f"{qid}: existing answer is bound to different retrieval evidence")
        if not observed_hash:
            if (
                clean_text(row.get("question")) != clean_text(expected.get("question"))
                or list(row.get("selected_session_ids") or [])
                != list(expected.get("selected_session_ids") or [])
                or clean_text(row.get("answer_model")) != "gpt-5.4"
            ):
                raise RuntimeError(f"{qid}: legacy answer cannot be bound to current evidence")
            row["evidence_sha256"] = expected_hash
            backfills += 1
    return hashes, backfills


def ordered_completed_answers(
    existing: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    qids = [clean_text(row.get("question_id")) for row in rows]
    if not all(qids) or len(qids) != len(set(qids)):
        raise RuntimeError("answer evidence contains missing or duplicate question_id")
    unexpected = sorted(set(existing) - set(qids))
    if unexpected:
        raise RuntimeError(
            f"existing answer output contains qids outside the frozen request: {unexpected[:10]}"
        )
    return [dict(existing[qid]) for qid in qids if qid in existing]


def load_harness(path: Path):
    module_name = "tmcra_v3_answer_harness"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import answer harness: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def reject_evaluation_fields(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        present = sorted(FORBIDDEN_EVALUATION_FIELDS.intersection(row))
        if present:
            qid = clean_text(row.get("question_id")) or "<missing-qid>"
            raise RuntimeError(
                f"{qid}: answer input contains evaluation-only fields: {present}"
            )


def render_hierarchical_evidence(window: Mapping[str, Any]) -> str:
    source = clean_text(window.get("text"))
    if not source:
        raise RuntimeError("evidence window has empty source text")
    blocks: list[str] = []
    for context in list(window.get("memory_contexts") or []):
        if not isinstance(context, Mapping):
            raise RuntimeError("memory_context must be an object")
        claim = clean_text(context.get("claim_text"))
        slot = clean_text(context.get("canonical_slot"))
        role = clean_text(context.get("role"))
        if not claim or not slot or not role:
            raise RuntimeError("memory_context lacks role, canonical_slot, or claim_text")
        blocks.append(f"[TMCRA slow memory role={role} slot={slot}]\n{claim}")
    for attachment in list(window.get("attachments") or []):
        if not isinstance(attachment, Mapping):
            raise RuntimeError("memory attachment must be an object")
        role = clean_text(attachment.get("role"))
        slot = clean_text(attachment.get("canonical_slot"))
        if role == "context_only":
            summary = clean_text(attachment.get("summary"))
            if not summary or not slot:
                raise RuntimeError("slow context attachment lacks slot or summary")
            blocks.append(
                f"[TMCRA slow memory role=context_only slot={slot}]\n{summary}"
            )
        elif role in {"fast_context", "override"}:
            override_text = clean_text(attachment.get("text"))
            if not override_text or not slot:
                raise RuntimeError("fast memory attachment lacks slot or text")
            if role == "override":
                blocks.append(
                    "[TMCRA fast memory role=override "
                    f"precedence=newer_fast_evidence slot={slot}]\n"
                    "This newer fast-layer evidence overrides conflicting slow-memory "
                    f"claims.\n{override_text}"
                )
            else:
                blocks.append(
                    f"[TMCRA fast memory role=context slot={slot}]\n{override_text}"
                )
        else:
            raise RuntimeError(f"unsupported memory attachment role: {role!r}")
    blocks.append("[TMCRA immutable source evidence]\n" + source)
    return "\n\n".join(blocks)


def build_user_memory_cues(
    evidence: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a compact index of durable claims and user-authored evidence.

    The index only points into the immutable Source reservoir. Round-robin
    emission prevents an early, verbose source from hiding cues in later
    retrieval windows.
    """
    cues_by_source: list[list[dict[str, Any]]] = []
    for item in evidence:
        evidence_id = clean_text(item.get("evidence_id"))
        source_record_id = clean_text(item.get("source_record_id"))
        source_cues: list[dict[str, Any]] = []

        def add_cue(
            kind: str,
            text: Any,
            *,
            record_id: str = source_record_id,
            historical_date: Any = None,
            timestamp: Any = None,
        ) -> None:
            normalized = clean_text(text)
            if not normalized:
                return
            source_cues.append(
                {
                    "evidence_id": evidence_id,
                    "kind": kind,
                    "source_record_id": record_id,
                    "historical_date": clean_text(
                        historical_date
                        if historical_date is not None
                        else item.get("historical_date")
                    ),
                    "timestamp": clean_text(
                        timestamp if timestamp is not None else item.get("timestamp")
                    ),
                    "text": normalized[:USER_MEMORY_CUE_TEXT_LIMIT],
                    "text_truncated": len(normalized) > USER_MEMORY_CUE_TEXT_LIMIT,
                }
            )

        for context in list(item.get("memory_contexts") or []):
            if isinstance(context, Mapping):
                add_cue(
                    "slow_claim",
                    context.get("claim_text") or context.get("capsule_summary"),
                )
        for attachment in list(item.get("attachments") or []):
            if isinstance(attachment, Mapping):
                add_cue(
                    "fast_memory",
                    attachment.get("text") or attachment.get("summary"),
                )
        if clean_text(item.get("message_role")).lower() == "user":
            add_cue("user_source", item.get("text"))
        for context in list(item.get("source_group_context") or []):
            if not isinstance(context, Mapping):
                continue
            role = clean_text(
                context.get("message_role") or context.get("role")
            ).lower()
            if role == "user":
                add_cue(
                    "user_context",
                    context.get("text"),
                    record_id=clean_text(context.get("source_record_id")),
                    historical_date=context.get("historical_date"),
                    timestamp=context.get("timestamp"),
                )
        cues_by_source.append(source_cues)

    output: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    max_source_cues = max((len(items) for items in cues_by_source), default=0)
    for cue_index in range(max_source_cues):
        for source_cues in cues_by_source:
            if cue_index >= len(source_cues):
                continue
            cue = source_cues[cue_index]
            fingerprint = clean_text(cue["text"]).casefold()
            if fingerprint in seen_text:
                continue
            seen_text.add(fingerprint)
            output.append({"cue_order": len(output) + 1, **cue})
            if len(output) >= USER_MEMORY_CUE_LIMIT:
                return output
    return output


EVIDENCE_OPERATION_ANSWER_PROMPT = f"""You answer one memory question from a compiled, immutable evidence packet.
Return exactly one JSON object and no prose using this schema:
{{"schema_version":"{ANSWER_SCHEMA}","answerability":"sufficient|insufficient","claims":[{{"claim_id":"C01","text":"one factual claim","origin":"memory_fact|memory_inference|memory_derived|query_context|model_knowledge","support_ids":["E01"],"computation_ids":["O01"]}}],"missing_requirements":["R01"],"answer":"short complete answer"}}

Read task_intent.output_origin before answering. task_intent and planner_source_groups are routing hints, not truth or answerability decisions. requirement_catalog lists the response requirement IDs but deliberately carries no planner missing/satisfied judgment. Determine support or absence yourself from the complete Source reservoir. user_memory_cues is a deterministic navigation index of slow-memory claims and user-authored Source or source-group context; it is not separate proof, so inspect the referenced Source before binding a remembered claim. For memory_conditioned_generation, first recover the user's grounded preferences or constraints and then generate the requested recommendation or advice; a ready-made historical answer is not required. Carry forward relevant prior choices, owned tools, resources, genres, constraints, and already-adopted solutions instead of defaulting to generic advice. When several relevant cues differ in specificity, preserve the narrower grounded detail rather than replacing it with a broad category. The final answer itself must explicitly show how at least one specific remembered constraint shaped the response; do not leave personalization only inside the claims array. Do not declare the task insufficient merely because the exact requested recommendation was not historical when grounded memory can personalize a new answer and model knowledge can fulfill the current target. Query context never requires Source evidence IDs. Model knowledge does not require Source evidence IDs when it is allowed for memory-conditioned generation. External-tool premises remain unresolved until a tool result exists.
Source evidence is ordered by frozen retrieval priority and then by source order within each selected session; source_order is not an event date. Each Source item carries its immutable historical_date and timestamp when available. Resolve words such as today and yesterday against that Source item's timestamp, never against the question date. Inspect source_group_context as neighboring source context and use each neighbor's own timestamp. Before answering, internally build an exact evidence ledger for the requested entity: list every matching event or item, its event date or stated relative time, state (actual, planned, hypothetical, or mentioned), and Source ID. Deduplicate repeated mentions of the same event or item before counting. For latest/earliest/order/duration questions, compare event times grounded in Source text plus its timestamp, not source_order or database session_index. If a when, before, or after clause names another remembered event, treat that event as a candidate temporal endpoint and compare both Source dates before using the question date. Use the question date only when the wording actually asks for distance from the query time.
Each Source item may include memory_contexts and attachments. A slow_context is the slow graph's durable interpretation and must retain its capsule, claim, and provenance identity. A fast_context is an atomic current-memory interpretation. An attachment marked override with precedence=newer_fast_evidence has controller-verified precedence over a conflicting slow claim in the same canonical slot. These graph views organize meaning but do not replace Source proof: support final remembered claims with the Source evidence ID carrying the context, and inspect its immutable text before relying on it.
Use only supplied Source evidence IDs and verified computation IDs. memory_fact claims require direct support_ids. memory_inference claims require support_ids for every Source premise and represent source-grounded comparison or synthesis that has no verified operation result. memory_derived claims require computation_ids. query_context claims may have empty binding lists in every task when they state only information supplied by the current question, such as the question date or arithmetic derived solely from it. Only memory_conditioned_generation may use model_knowledge claims with empty binding lists; label their origin explicitly, and include at least one memory_fact, memory_inference, or memory_derived claim showing how remembered constraints shaped the answer. Never label remembered information as query_context or model_knowledge. planner_operation_candidates expose no result because their operand selection has not been semantically certified; independently inspect their Source supports. lexical_anchor_ids are deterministic search hints, not proof. Never transfer values between different entities, people, places, events, or activities. Preserve requested temporal order, distinct values, totals, and update precedence.
For preferences, use the specific historical experiences in Source evidence. Treat named entities and activities exactly: a shorter, broader, or related term is not the requested multiword entity unless the source explicitly says they are equivalent. For false premises or entity mismatch, explicitly state that the requested target is not established, then optionally state related counterevidence. For count/list questions, enumerate distinct matching items before producing the total. For changing facts, apply latest update precedence using event time and source-group context. A sufficient answer must cover every requirement and have no missing_requirements. An insufficient answer must name supplied requirement IDs that remain unresolved after inspecting all Source evidence. Never use hidden supervision or outside facts for remembered claims."""


SEMANTIC_EVIDENCE_ANSWER_PROMPT = f"""Execute one verified memory task contract.
Return exactly one JSON object and no prose using this schema:
{{"schema_version":"{SEMANTIC_ANSWER_SCHEMA}","response_status":"answered|partial|answered_with_context|requires_tool|clarification|not_answerable","claims":[{{"claim_id":"C01","text":"one final claim","origin":"memory_fact|memory_derived|model_knowledge|tool_result","premise_ids":["P01"],"source_claim_ids":["M01"],"computation_ids":["O01"]}}],"unresolved_premise_ids":[],"answer":"concise complete response"}}

The task contract defines the requested target and output shape. The answerability certificate is authoritative; do not redo retrieval, invent missing memory, or change its answer policy.
For memory_fact claims, use only Source claim IDs bound to the cited premises. For memory_derived claims, use only completed computation IDs bound to the cited premises. Preserve entity, time, polarity, modality, cardinality, ordering, and update state.
When the policy is answer_with_memory_context, treat remembered claims as constraints or background. You may produce model_knowledge claims needed to fulfill the current request, but never describe them as remembered facts and never attach Source claim IDs to them.
When the policy is answer_partial, answer only the supported portion, explicitly state the unresolved limitation, use response_status partial, and preserve the certificate's unresolved premise IDs. For a memory-conditioned task, the partial answer must explicitly use at least one bound memory claim rather than falling back to a generic response. Never turn a partial count or other incomplete derived aggregate into a final total.
When the policy is call_tool, clarify, or abstain, use the matching response_status and explain the unresolved need without fabricating an answer. unresolved_premise_ids must exactly match the certificate.
Do not cite raw evidence IDs directly. Do not use benchmark labels or hidden supervision."""


def render_compiled_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    if packet.get("schema_version") != PACKET_SCHEMA:
        raise RuntimeError("compiled evidence packet schema is invalid")
    evidence = list(packet.get("raw_evidence_reservoir") or [])
    evidence_ids = {clean_text(item.get("evidence_id")) for item in evidence}
    if not evidence or not all(evidence_ids):
        raise RuntimeError("compiled evidence packet has no source reservoir")
    bundles = []
    for bundle in list(packet.get("evidence_bundles") or []):
        ids = list(bundle.get("evidence_ids") or [])
        if any(item not in evidence_ids for item in ids):
            raise RuntimeError("compiled evidence bundle cites unknown source")
        bundles.append(
            {
                "bundle_id": bundle.get("bundle_id"),
                "role": bundle.get("role"),
                "evidence_ids": ids,
            }
        )
    question_contract = dict(packet.get("question_contract") or {})
    task_contract = packet.get("task_contract")
    if not isinstance(task_contract, Mapping):
        task_contract = question_contract.get("task_contract")
    question_contract.pop("task_contract", None)
    requirements = list(packet.get("requirement_coverage") or [])
    premise_by_id = {
        clean_text(item.get("premise_id")): item
        for item in (
            task_contract.get("premises") or []
            if isinstance(task_contract, Mapping)
            else []
        )
        if isinstance(item, Mapping)
    }
    output_origin = (
        clean_text(task_contract.get("output_origin"))
        if isinstance(task_contract, Mapping)
        else clean_text(question_contract.get("output_origin"))
    )
    requirement_catalog = []
    bound_constraints = []
    for requirement in requirements:
        requirement_id = clean_text(requirement.get("requirement_id"))
        evidence_ids_for_requirement = [
            item
            for item in list(requirement.get("evidence_ids") or [])
            if item in evidence_ids
        ]
        premise = premise_by_id.get(requirement_id) or {}
        description = clean_text(requirement.get("description"))
        source_kind = clean_text(premise.get("source")) or "memory"
        catalog_item = {
            "requirement_id": requirement_id,
            "description": (
                "remembered personalization constraints; inspect all Source evidence"
                if output_origin == "memory_conditioned_generation"
                and source_kind == "memory"
                and not evidence_ids_for_requirement
                else description
            ),
            "source_kind": source_kind,
        }
        if evidence_ids_for_requirement:
            catalog_item["candidate_source_ids"] = evidence_ids_for_requirement
        requirement_catalog.append(catalog_item)
        if evidence_ids_for_requirement and isinstance(premise, Mapping):
            bound_constraints.append(
                {
                    "premise_id": requirement_id,
                    "description": clean_text(premise.get("description")) or description,
                    "source_kind": source_kind,
                    "grounded_constraints": list(premise.get("grounded_constraints") or []),
                    "candidate_source_ids": evidence_ids_for_requirement,
                }
            )
    operation_results = list(packet.get("operation_results") or [])
    verified_operations = [
        dict(item)
        for item in operation_results
        if item.get("status") == "completed"
        and item.get("answer_authoritative") is True
    ]
    planner_operation_candidates = [
        {
            "operation_id": item.get("operation_id"),
            "operation_type": item.get("operation_type"),
            "support_ids": list(item.get("support_ids") or []),
            "semantic_status": "uncertified_operand_selection",
        }
        for item in operation_results
        if item.get("status") == "completed"
        and item.get("answer_authoritative") is not True
    ]
    task_intent: dict[str, Any] = {"output_origin": output_origin}
    if isinstance(task_contract, Mapping):
        task_intent["target"] = dict(task_contract.get("target") or {})
        task_intent["output"] = dict(task_contract.get("output") or {})
        task_intent["bound_constraints"] = bound_constraints
    rendered = {
        "answer_view_schema": "tmcra.v4.answer-evidence-view.5",
        "question_contract": question_contract,
        "task_intent": task_intent,
        "user_memory_cues": build_user_memory_cues(evidence),
        "requirement_catalog": requirement_catalog,
        "verified_operation_results": verified_operations,
        "planner_operation_candidates": planner_operation_candidates,
        "planner_source_groups": bundles,
        "lexical_anchor_ids": list(packet.get("lexical_anchor_ids") or []),
        "source_evidence": [
            {
                "source_order": index,
                "evidence_id": item.get("evidence_id"),
                "session_id": item.get("session_id"),
                "session_index": item.get("session_index"),
                "parent_chunk_index": item.get("parent_chunk_index"),
                "historical_date": item.get("historical_date"),
                "timestamp": item.get("timestamp"),
                "message_role": item.get("message_role"),
                "question_overlap_terms": item.get("question_overlap_terms") or [],
                "question_overlap_score": int(item.get("question_overlap_score", 0)),
                "text": item.get("text"),
                "source_group_id": item.get("source_group_id"),
                "source_group_context": list(item.get("source_group_context") or []),
                "memory_contexts": list(item.get("memory_contexts") or []),
                "attachments": list(item.get("attachments") or []),
                "provenance": list(item.get("provenance") or []),
            }
            for index, item in enumerate(evidence, start=1)
        ],
    }
    if isinstance(packet.get("semantic_advisory"), Mapping):
        rendered["semantic_advisory"] = dict(packet["semantic_advisory"])
    return rendered


def parse_operation_bound_answer(raw: str, packet: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"operation-bound answer is not strict JSON: {exc}") from exc
    try:
        return validate_evidence_bound_answer(value, packet)
    except EvidenceOperationError as exc:
        raise RuntimeError(f"operation-bound answer failed validation: {exc}") from exc


def render_semantic_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    if packet.get("schema_version") != SEMANTIC_PACKET_SCHEMA:
        raise RuntimeError("semantic evidence packet schema is invalid")
    contract = packet.get("task_contract")
    certificate = packet.get("answerability_certificate")
    if not isinstance(contract, Mapping) or not isinstance(certificate, Mapping):
        raise RuntimeError("semantic evidence packet lacks contract or certificate")
    premise_by_id = {
        clean_text(item.get("premise_id")): item
        for item in contract.get("premises") or []
        if isinstance(item, Mapping)
    }
    claims_by_id = {
        clean_text(item.get("claim_id")): item
        for item in packet.get("claim_ledger") or []
        if isinstance(item, Mapping)
    }
    evidence_by_id = {
        clean_text(item.get("evidence_id")): item
        for item in packet.get("raw_evidence_reservoir") or []
        if isinstance(item, Mapping)
    }
    operation_by_id = {
        clean_text(item.get("operation_id")): item
        for item in (packet.get("resolution_program") or {}).get("operation_results") or []
        if isinstance(item, Mapping)
    }
    resolved_premises = []
    for binding in packet.get("premise_bindings") or []:
        if not isinstance(binding, Mapping):
            raise RuntimeError("semantic premise binding must be an object")
        premise_id = clean_text(binding.get("premise_id"))
        if premise_id not in premise_by_id:
            raise RuntimeError("semantic premise binding cites an unknown premise")
        claim_ids = list(binding.get("claim_ids") or [])
        evidence_ids = list(binding.get("evidence_ids") or [])
        operation_ids = list(binding.get("operation_ids") or [])
        if any(item not in claims_by_id for item in claim_ids):
            raise RuntimeError("semantic premise binding cites an unknown Source claim")
        if any(item not in evidence_by_id for item in evidence_ids):
            raise RuntimeError("semantic premise binding cites unknown evidence")
        if any(item not in operation_by_id for item in operation_ids):
            raise RuntimeError("semantic premise binding cites an unknown operation")
        resolved_premises.append(
            {
                "premise": premise_by_id[premise_id],
                "coverage": binding.get("coverage"),
                "relation": binding.get("relation"),
                "source_claims": [claims_by_id[item] for item in claim_ids],
                "operation_results": [operation_by_id[item] for item in operation_ids],
                "source_evidence": [
                    {
                        "evidence_id": item,
                        "session_id": evidence_by_id[item].get("session_id"),
                        "session_index": evidence_by_id[item].get("session_index"),
                        "text": evidence_by_id[item].get("text"),
                    }
                    for item in evidence_ids
                ],
            }
        )
    return {
        "query_context": dict(packet.get("query_context") or {}),
        "task_contract": dict(contract),
        "answerability_certificate": dict(certificate),
        "resolved_premises": resolved_premises,
        "non_memory_premises": [
            premise
            for premise in contract.get("premises") or []
            if isinstance(premise, Mapping) and premise.get("source") != "memory"
        ],
        "target_operation_results": [
            operation_by_id[item]
            for item in certificate.get("target_operation_ids") or []
            if item in operation_by_id
        ],
    }


def parse_semantic_bound_answer(raw: str, packet: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"semantic-bound answer is not strict JSON: {exc}") from exc
    try:
        return validate_semantic_answer(normalize_semantic_answer_output(value, packet), packet)
    except SemanticEvidenceError as exc:
        raise RuntimeError(f"semantic-bound answer failed validation: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--answer-window-limit", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--qid-list", default="")
    parser.add_argument("--lane", choices=("production", "shadow"), default="production")
    args = parser.parse_args()
    answer_model = clean_text(os.getenv("TMCRA_ANSWER_MODEL"))
    answer_base_url = clean_text(os.getenv("TMCRA_ANSWER_BASE_URL"))
    answer_api_key = clean_text(os.getenv("TMCRA_ANSWER_API_KEY"))
    if answer_model != "gpt-5.4":
        raise RuntimeError(f"fixed answer model must be exactly gpt-5.4, got {answer_model!r}")
    if not answer_base_url or not answer_api_key:
        raise RuntimeError("TMCRA_ANSWER_BASE_URL and TMCRA_ANSWER_API_KEY are required")
    fixed_answer_layer = fixed_answer_environment(args.answer_window_limit)
    os.environ.update(fixed_answer_layer)
    rows = read_jsonl(Path(args.evidence))
    if args.qid_list:
        requested = [
            clean_text(value)
            for value in Path(args.qid_list).read_text(encoding="utf-8").splitlines()
            if clean_text(value)
        ]
        by_qid = {clean_text(row.get("question_id")): row for row in rows}
        missing = [qid for qid in requested if qid not in by_qid]
        if missing:
            raise RuntimeError(f"requested qids are missing from evidence: {missing[:10]}")
        rows = [by_qid[qid] for qid in requested]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("no answer rows selected")
    reject_evaluation_fields(rows)
    if args.lane == "production":
        try:
            validate_production_evidence(rows)
        except RoutePolicyError as exc:
            raise RuntimeError(
                f"production route failed before GPT-5.4 calls: {exc}"
            ) from exc
    harness = load_harness(Path(args.harness))
    configured_base_url, configured_model, configured_api_key = harness.answer_llm_config()
    if clean_text(configured_model) != "gpt-5.4":
        raise RuntimeError(f"harness resolved unexpected model: {configured_model!r}")
    if not clean_text(configured_base_url) or not clean_text(configured_api_key):
        raise RuntimeError("harness did not resolve the required GPT5.4 endpoint")

    # The legacy harness contains optional answer-side repair hooks. Keep its
    # primary prompt, but make one logical GPT call the enforced contract.
    harness.complete_profile_answer_from_evidence = lambda _question, answer, _windows: answer
    call_state = threading.local()
    original_chat_completion = harness.chat_completion

    def counted_chat_completion(*call_args: Any, **call_kwargs: Any) -> str:
        call_state.count = int(getattr(call_state, "count", 0)) + 1
        return original_chat_completion(*call_args, **call_kwargs)

    harness.chat_completion = counted_chat_completion
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_attempt_dir = out_dir / "raw_attempts"
    raw_attempt_dir.mkdir(exist_ok=True)
    answers_path = out_dir / "answers.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if answers_path.exists():
        for row in read_jsonl(answers_path):
            qid = clean_text(row.get("question_id"))
            if qid:
                if qid in existing:
                    raise RuntimeError(f"duplicate existing answer row: {qid}")
                existing[qid] = row
    if existing and args.lane == "production":
        try:
            validate_production_answers(list(existing.values()))
        except RoutePolicyError as exc:
            raise RuntimeError(
                "existing answers use an incompatible production protocol; "
                "start a new answer output directory"
            ) from exc
    evidence_hashes, legacy_evidence_hash_backfills = bind_existing_answers_to_evidence(
        existing, rows
    )
    lock = threading.Lock()

    def answer_one(row: Mapping[str, Any]) -> dict[str, Any]:
        qid = clean_text(row.get("question_id"))
        question = clean_text(row.get("question"))
        question_date = clean_text(row.get("question_date"))
        runtime_question = f"{question}\nQuestion date: {question_date}" if question_date else question
        semantic_packet = row.get("semantic_evidence_packet")
        operation_packet = row.get("compiled_evidence_packet")
        if args.lane == "production":
            if isinstance(semantic_packet, Mapping):
                raise RuntimeError(f"{qid}: semantic packets are forbidden in the production lane")
            if not isinstance(operation_packet, Mapping):
                raise RuntimeError(f"{qid}: production lane requires a compiled evidence packet")
            packet = operation_packet
            rendered_packet = render_compiled_packet(packet)
            answer_prompt = EVIDENCE_OPERATION_ANSWER_PROMPT
            answer_parser = parse_operation_bound_answer
            answer_protocol = PRODUCTION_ANSWER_PROTOCOL
        else:
            if not isinstance(semantic_packet, Mapping) or isinstance(operation_packet, Mapping):
                raise RuntimeError(f"{qid}: shadow lane requires only a semantic evidence packet")
            packet = semantic_packet_to_advisory_packet(semantic_packet)
            rendered_packet = render_compiled_packet(packet)
            answer_prompt = EVIDENCE_OPERATION_ANSWER_PROMPT
            answer_parser = parse_operation_bound_answer
            answer_protocol = SHADOW_ANSWER_PROTOCOL
        messages = [
            {"role": "system", "content": answer_prompt},
            {
                "role": "user",
                "content": json.dumps(rendered_packet, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        started = time.time()
        last_error: Exception | None = None
        answer = ""
        evidence_binding: dict[str, Any] = {}
        attempts_used = 0
        logical_answer_calls = 0
        validation_error = ""
        for attempt in range(1, max(1, args.attempts) + 1):
            attempts_used = attempt
            call_state.count = 0
            try:
                attempt_path = raw_attempt_dir / f"{qid}.attempt_{attempt}.json"
                if attempt_path.is_file():
                    persisted = json.loads(attempt_path.read_text(encoding="utf-8"))
                    if persisted.get("question_id") != qid or persisted.get("evidence_sha256") != evidence_hashes[qid]:
                        raise RuntimeError(f"{qid}: persisted raw answer identity mismatch")
                    raw_answer = str(persisted["raw_answer"])
                    attempt_call_count = 0
                else:
                    attempt_messages = list(messages)
                    if validation_error:
                        attempt_messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "The previous JSON failed the deterministic validator: "
                                    f"{validation_error}. Correct only the JSON structure or evidence bindings. "
                                    "Use the same compiled packet and return the required schema."
                                ),
                            }
                        )
                    raw_answer = harness.chat_completion(
                        configured_base_url,
                        configured_model,
                        attempt_messages,
                        max_tokens=int(os.getenv("TMCRA_ANSWER_MAX_TOKENS", "1536")),
                        temperature=0.0,
                        api_key=configured_api_key,
                    )
                    attempt_call_count = int(getattr(call_state, "count", 0))
                    if attempt_call_count != 1:
                        raise RuntimeError(
                            f"{qid}: answer attempt made {attempt_call_count} logical GPT calls; expected exactly 1"
                        )
                    temporary = attempt_path.with_name(f".{attempt_path.name}.tmp.{os.getpid()}.{time.time_ns()}")
                    temporary.write_text(
                        json.dumps(
                            {
                                "question_id": qid,
                                "evidence_sha256": evidence_hashes[qid],
                                "attempt": attempt,
                                "raw_answer": raw_answer,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    os.replace(temporary, attempt_path)
                logical_answer_calls += attempt_call_count
                if attempt_call_count not in {0, 1}:
                    raise RuntimeError(
                        f"{qid}: invalid physical call count for answer attempt"
                    )
                evidence_binding = answer_parser(raw_answer, packet)
                answer = evidence_binding["answer"]
                break
            except Exception as exc:
                last_error = exc
                validation_error = f"{exc.__class__.__name__}: {exc}"
                if attempt >= max(1, args.attempts):
                    break
                time.sleep(min(8.0, 1.5 * (2 ** (attempt - 1))))
        if not answer:
            assert last_error is not None
            raise RuntimeError(
                f"{qid}: GPT5.4 failed after {max(1, args.attempts)} same-model attempts: "
                f"{last_error.__class__.__name__}:{last_error}"
            )
        return {
            "question_id": qid,
            "question_type": clean_text(row.get("question_type")),
            "question": question,
            "evidence_sha256": evidence_hashes[qid],
            "hypothesis": answer,
            "evidence_binding": evidence_binding,
            "answer_protocol": answer_protocol,
            "answer_model": configured_model,
            "answer_attempt_limit": max(1, args.attempts),
            "answer_attempts_used": attempts_used,
            "logical_answer_calls": logical_answer_calls,
            "selected_session_ids": list(row.get("selected_session_ids") or []),
            "latency_sec": round(time.time() - started, 3),
            "error": "",
        }

    pending = [row for row in rows if clean_text(row.get("question_id")) not in existing]
    failures: list[dict[str, str]] = []
    completed = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(answer_one, row): clean_text(row.get("question_id")) for row in pending}
        for future in as_completed(futures):
            qid = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append({"question_id": qid, "error": f"{exc.__class__.__name__}:{str(exc)[:500]}"})
                continue
            with lock:
                with answers_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                existing[qid] = result
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    print(
                        json.dumps(
                            {
                                "completed_this_run": completed,
                                "pending_this_run": len(pending),
                                "total_saved": len(existing),
                                "failures": len(failures),
                                "elapsed_sec": round(time.time() - started, 3),
                            }
                        ),
                        flush=True,
                    )
    final_rows = ordered_completed_answers(existing, rows)
    atomic_write_jsonl(answers_path, final_rows)
    protocols = sorted({str(row.get("answer_protocol") or "") for row in final_rows})
    report = {
        "status": "complete" if len(final_rows) == len(rows) and not failures else "incomplete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "answer_model": configured_model,
        "lane": args.lane,
        "promotion_eligible": args.lane == "production",
        "fixed_answer_layer": fixed_answer_layer,
        "disabled_harness_hooks": ["complete_profile_answer_from_evidence"],
        "max_logical_answer_calls_per_success": max(1, args.attempts),
        "hierarchical_memory_context_rendering": True,
        "compiled_evidence_packet_required": True,
        "answer_protocol": protocols[0] if len(protocols) == 1 else "mixed",
        "evidence_binding_persisted": True,
        "answers_bound_to_evidence_sha256": True,
        "legacy_evidence_hash_backfills": legacy_evidence_hash_backfills,
        "requested": len(rows),
        "completed": len(final_rows),
        "failure_count": len(failures),
        "avg_latency_sec": round(sum(float(row["latency_sec"]) for row in final_rows) / max(1, len(final_rows)), 3),
        "answers": str(answers_path),
    }
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
