"""Evidence compilation and deterministic operations for TMCRA V4.

This module never retrieves, drops, summarizes, or rewrites source evidence.
It assigns stable packet IDs, extracts verifiable atoms, validates operation
plans against those IDs, executes deterministic calculations, and validates
answer claims against the resulting packet.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from tmcra_v4_task_contract import (
    RISK_PLANNER_MISSING_WITH_PLAUSIBLE_SOURCE,
    TaskContractError,
    structural_risk_signals,
    validate_task_contract,
)
from tmcra_v4_typed_semantics import evaluate_proposals


CATALOG_SCHEMA = "tmcra.evidence-atom-catalog.v1"
GRAPH_SCHEMA = "tmcra.query-evidence-graph.v1"
PLAN_SCHEMA = "tmcra.evidence-operation-plan.v1"
PACKET_SCHEMA = "tmcra.compiled-evidence-packet.v1"
PACKET_COMPILER_VERSION = "tmcra-v4-packet-compiler-2026-07-14.7"
ANSWER_SCHEMA = "tmcra.evidence-bound-answer.v2"
ANSWER_CLAIM_ORIGINS = frozenset(
    {
        "memory_fact",
        "memory_inference",
        "memory_derived",
        "query_context",
        "model_knowledge",
    }
)

OPERATION_TYPES = frozenset(
    {
        "date_difference",
        "date_order",
        "numeric_sum",
        "numeric_difference",
        "numeric_average",
        "count_distinct",
        "ordered_unique_list",
        "entity_exact_match",
        "entity_mismatch",
        "set_difference",
    }
)
ATOM_TYPES = frozenset({"date", "number", "currency", "quantity", "entity"})
REQUIREMENT_STATES = frozenset(
    {"satisfied", "conflicting", "missing", "invalid_operation"}
)

MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
MONTHS.update(
    {name.lower(): index for index, name in enumerate(calendar.month_abbr) if name}
)
DATE_PATTERNS = (
    re.compile(r"\b(?P<year>20\d{2})[-/](?P<month>0?[1-9]|1[0-2])[-/](?P<day>0?[1-9]|[12]\d|3[01])(?=T|\b)"),
    re.compile(
        r"\b(?P<month_name>January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+"
        r"(?P<day>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?(?:,)?\s+(?P<year>20\d{2})\b",
        re.IGNORECASE,
    ),
)
NUMBER_PATTERN = re.compile(
    r"(?<![\w.])(?P<currency>[$€£¥])?\s*(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?:\s*(?P<unit>days?|weeks?|months?|years?|hours?|minutes?|times?|miles?|kilometers?|km|kg|lbs?|percent|%))?"
    r"(?!\w)",
    re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
QUESTION_STOPWORDS = frozenset(
    {"a", "an", "and", "are", "at", "be", "did", "do", "for", "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "the", "to", "was", "what", "when", "with"}
)
SOURCE_TIMESTAMP_PATTERN = re.compile(
    r"\btimestamp=(?P<year>20\d{2})-(?P<month>0?[1-9]|1[0-2])-(?P<day>0?[1-9]|[12]\d|3[01])(?!\d)"
)
RELATIVE_DAY_PATTERNS = (
    (re.compile(r"\b(?:yesterday|the day before)\b", re.IGNORECASE), -1, "source_timestamp_minus_1_day"),
    (re.compile(r"\btoday\b", re.IGNORECASE), 0, "source_timestamp_same_day"),
)
EXPLICIT_RELATIVE_DURATION_PATTERNS = (
    (
        re.compile(
            r"\b(?:(?:about|around|roughly)\s+)?(?P<count>\d+|a|an|one)\s+days?\s+ago\b",
            re.IGNORECASE,
        ),
        1,
        0,
    ),
    (
        re.compile(
            r"\b(?:(?:about|around|roughly)\s+)?(?P<count>\d+|a|an|one)\s+weeks?\s+ago\b",
            re.IGNORECASE,
        ),
        7,
        1,
    ),
    (
        re.compile(
            r"\b(?:(?:about|around|roughly)\s+)?(?P<count>\d+|a|an|one)\s+months?\s+ago\b",
            re.IGNORECASE,
        ),
        30,
        3,
    ),
)


class EvidenceOperationError(RuntimeError):
    pass


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _ids(value: Any, *, path: str, allowed: set[str], allow_empty: bool = False) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvidenceOperationError(f"{path} must be an ID array")
    output = [_text(item) for item in value]
    if (not output and not allow_empty) or any(not item or item not in allowed for item in output):
        raise EvidenceOperationError(f"{path} contains an unknown or empty ID")
    if len(output) != len(set(output)):
        raise EvidenceOperationError(f"{path} contains duplicate IDs")
    return output


def _normalize_date(match: re.Match[str]) -> str | None:
    groups = match.groupdict()
    try:
        month = (
            MONTHS[groups["month_name"].lower()]
            if groups.get("month_name")
            else int(groups["month"])
        )
        value = date(int(groups["year"]), month, int(groups["day"]))
    except (KeyError, TypeError, ValueError):
        return None
    return value.isoformat()


def _overlaps(span: tuple[int, int], spans: Sequence[tuple[int, int]]) -> bool:
    return any(span[1] > other[0] and span[0] < other[1] for other in spans)


def _lexical_terms(value: str) -> set[str]:
    return {
        match.group(0).casefold()
        for match in WORD_PATTERN.finditer(value)
        if len(match.group(0)) >= 3 and match.group(0).casefold() not in QUESTION_STOPWORDS
    }


def _extract_atoms_from_text(
    text: str,
    *,
    evidence_id: str,
    counters: dict[str, int],
    source_timestamp: str = "",
    source_historical_date: str = "",
) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    date_spans: list[tuple[int, int]] = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if _overlaps(span, date_spans):
                continue
            normalized = _normalize_date(match)
            if not normalized:
                continue
            counters["date"] += 1
            date_spans.append(span)
            atoms.append(
                {
                    "atom_id": f"D{counters['date']:03d}",
                    "atom_type": "date",
                    "raw_text": match.group(0),
                    "normalized_value": normalized,
                    "unit": "date",
                    "evidence_id": evidence_id,
                    "char_start": span[0],
                    "char_end": span[1],
                }
            )
    if evidence_id != "QUESTION":
        timestamp_match = SOURCE_TIMESTAMP_PATTERN.search(text)
        timestamp_value = _normalize_date(timestamp_match) if timestamp_match else None
        if not timestamp_value and source_timestamp:
            structured_match = SOURCE_TIMESTAMP_PATTERN.search(
                f"timestamp={source_timestamp}"
            )
            timestamp_value = (
                _normalize_date(structured_match) if structured_match else None
            )
        if not timestamp_value and source_historical_date:
            for pattern in DATE_PATTERNS:
                structured_match = pattern.search(source_historical_date)
                if structured_match:
                    timestamp_value = _normalize_date(structured_match)
                    if timestamp_value:
                        break
        if timestamp_value:
            base_date = date.fromisoformat(timestamp_value)
            for pattern, offset_days, derivation in RELATIVE_DAY_PATTERNS:
                for match in pattern.finditer(text):
                    counters["date"] += 1
                    atoms.append(
                        {
                            "atom_id": f"D{counters['date']:03d}",
                            "atom_type": "date",
                            "raw_text": match.group(0),
                            "normalized_value": (base_date + timedelta(days=offset_days)).isoformat(),
                            "unit": "date",
                            "evidence_id": evidence_id,
                            "char_start": match.start(),
                            "char_end": match.end(),
                            "derivation": derivation,
                        }
                    )
    for match in NUMBER_PATTERN.finditer(text):
        span = (match.start(), match.end())
        if _overlaps(span, date_spans):
            continue
        raw_number = match.group("number").replace(",", "")
        try:
            numeric = Decimal(raw_number)
        except InvalidOperation:
            continue
        currency = _text(match.group("currency"))
        unit = _text(match.group("unit")).lower()
        atom_type = "currency" if currency else ("quantity" if unit else "number")
        counters[atom_type] += 1
        prefix = {"currency": "C", "quantity": "Q", "number": "N"}[atom_type]
        atoms.append(
            {
                "atom_id": f"{prefix}{counters[atom_type]:03d}",
                "atom_type": atom_type,
                "raw_text": match.group(0),
                "normalized_value": str(numeric.normalize()),
                "unit": currency or unit,
                "evidence_id": evidence_id,
                "char_start": span[0],
                "char_end": span[1],
            }
        )
    atoms.sort(key=lambda item: (int(item["char_start"]), str(item["atom_id"])))
    return atoms


def build_evidence_catalog(row: Mapping[str, Any]) -> dict[str, Any]:
    question = _text(row.get("question"))
    windows = row.get("evidence_windows")
    if not question or not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)) or not windows:
        raise EvidenceOperationError("evidence row requires a question and source windows")
    counters = {key: 0 for key in ATOM_TYPES}
    evidence: list[dict[str, Any]] = []
    atoms: list[dict[str, Any]] = []

    def object_list(window: Mapping[str, Any], field: str, *, window_index: int) -> list[dict[str, Any]]:
        raw = window.get(field) or []
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise EvidenceOperationError(
                f"evidence_windows[{window_index}].{field} must be an array"
            )
        output: list[dict[str, Any]] = []
        for item_index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise EvidenceOperationError(
                    f"evidence_windows[{window_index}].{field}[{item_index}] must be an object"
                )
            output.append(dict(item))
        return output

    for index, window in enumerate(windows, start=1):
        if not isinstance(window, Mapping):
            raise EvidenceOperationError(f"evidence_windows[{index - 1}] must be an object")
        raw_text = window.get("text")
        text = raw_text if isinstance(raw_text, str) else ""
        session_id = _text(window.get("session_id"))
        if not text.strip() or not session_id:
            raise EvidenceOperationError(f"evidence_windows[{index - 1}] lacks source text or session ID")
        evidence_id = f"E{index:02d}"
        raw_context = window.get("source_group_context") or []
        if not isinstance(raw_context, Sequence) or isinstance(raw_context, (str, bytes)):
            raise EvidenceOperationError(
                f"evidence_windows[{index - 1}].source_group_context must be an array"
            )
        source_group_context: list[dict[str, Any]] = []
        for context_index, member in enumerate(raw_context):
            if not isinstance(member, Mapping):
                raise EvidenceOperationError(
                    f"evidence_windows[{index - 1}].source_group_context[{context_index}] must be an object"
                )
            member_text = member.get("text")
            member_session_id = _text(member.get("session_id"))
            if (
                not isinstance(member_text, str)
                or not member_text.strip()
                or member_session_id != session_id
            ):
                raise EvidenceOperationError(
                    f"evidence_windows[{index - 1}].source_group_context[{context_index}] is invalid"
                )
            source_group_context.append(
                {
                    "relationship": _text(member.get("relationship")) or "session_neighbor",
                    "parent_distance": int(member.get("parent_distance", 0)),
                    "session_id": member_session_id,
                    "session_index": int(member.get("session_index", window.get("session_index", 0))),
                    "parent_chunk_index": int(member.get("parent_chunk_index", 0)),
                    "source_record_id": _text(member.get("source_record_id")),
                    "source_char_start": int(member.get("source_char_start", 0)),
                    "source_char_end": int(
                        member.get("source_char_end", len(member_text))
                    ),
                    "historical_date": _text(member.get("historical_date")),
                    "timestamp": _text(member.get("timestamp")),
                    "message_role": _text(member.get("message_role")),
                    "text": member_text,
                }
            )
        evidence.append(
            {
                "evidence_id": evidence_id,
                "scope_id": _text(window.get("scope_id")),
                "db_path": _text(window.get("db_path")),
                "session_id": session_id,
                "session_index": int(window.get("session_index", 0)),
                "parent_chunk_index": int(window.get("parent_chunk_index", 0)),
                "subchunk_index": int(window.get("subchunk_index", 0)),
                "historical_date": _text(window.get("historical_date")),
                "timestamp": _text(window.get("timestamp")),
                "message_role": _text(window.get("message_role")),
                "text": text,
                "source_record_id": _text(window.get("source_record_id")),
                "source_char_start": int(window.get("source_char_start", 0)),
                "source_char_end": int(window.get("source_char_end", len(text))),
                "retrieval_metadata": dict(window.get("retrieval_metadata") or {}),
                "source_group_id": _text(window.get("source_group_id"))
                or f"source-group::{session_id}:{int(window.get('parent_chunk_index', 0))}",
                "source_group_context": source_group_context,
                "memory_contexts": object_list(
                    window, "memory_contexts", window_index=index - 1
                ),
                "attachments": object_list(
                    window, "attachments", window_index=index - 1
                ),
                "provenance": object_list(
                    window, "provenance", window_index=index - 1
                ),
            }
        )
        atoms.extend(
            _extract_atoms_from_text(
                text,
                evidence_id=evidence_id,
                counters=counters,
                source_timestamp=_text(window.get("timestamp")),
                source_historical_date=_text(window.get("historical_date")),
            )
        )
        for member in source_group_context:
            atoms.extend(
                _extract_atoms_from_text(
                    member["text"],
                    evidence_id=evidence_id,
                    counters=counters,
                    source_timestamp=_text(member.get("timestamp")),
                    source_historical_date=_text(member.get("historical_date")),
                )
            )
    question_terms = _lexical_terms(question)
    for item in evidence:
        overlap_terms = sorted(question_terms & _lexical_terms(item["text"]))
        item["question_overlap_terms"] = overlap_terms
        item["question_overlap_score"] = len(overlap_terms)
    question_atoms = _extract_atoms_from_text(
        question, evidence_id="QUESTION", counters=counters
    )
    question_date = _text(row.get("question_date"))
    if question_date:
        question_atoms.extend(
            _extract_atoms_from_text(
                question_date, evidence_id="QUESTION", counters=counters
            )
        )
    atoms.extend(question_atoms)
    if len({item["atom_id"] for item in atoms}) != len(atoms):
        raise EvidenceOperationError("atom IDs are not unique")
    return {
        "schema_version": CATALOG_SCHEMA,
        "question_id": _text(row.get("question_id")),
        "question": question,
        "question_date": question_date,
        "evidence": evidence,
        "lexical_anchor_ids": [
            item["evidence_id"]
            for item in sorted(
                evidence,
                key=lambda candidate: (
                    -int(candidate["question_overlap_score"]),
                    int(str(candidate["evidence_id"])[1:]),
                ),
            )
            if item["question_overlap_score"] > 0
        ][:8],
        "atoms": atoms,
    }


def build_query_evidence_graph(catalog: Mapping[str, Any]) -> dict[str, Any]:
    if catalog.get("schema_version") != CATALOG_SCHEMA:
        raise EvidenceOperationError("catalog schema is invalid")
    evidence = list(catalog.get("evidence") or [])
    atoms = list(catalog.get("atoms") or [])
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    sessions: set[str] = set()
    for item in evidence:
        evidence_id, session_id = _text(item.get("evidence_id")), _text(item.get("session_id"))
        if not evidence_id or not session_id:
            raise EvidenceOperationError("catalog evidence identity is invalid")
        nodes.append({"node_id": evidence_id, "node_type": "evidence"})
        session_node = f"S:{session_id}"
        if session_node not in sessions:
            sessions.add(session_node)
            nodes.append({"node_id": session_node, "node_type": "session", "session_id": session_id})
        edges.append({"edge_type": "belongs_to_session", "source": evidence_id, "target": session_node, "support_ids": [evidence_id]})
    evidence_ids = {item["evidence_id"] for item in evidence}
    for atom in atoms:
        atom_id, evidence_id = _text(atom.get("atom_id")), _text(atom.get("evidence_id"))
        if not atom_id:
            raise EvidenceOperationError("catalog atom identity is invalid")
        nodes.append({"node_id": atom_id, "node_type": "atom", "atom_type": atom.get("atom_type"), "normalized_value": atom.get("normalized_value")})
        if evidence_id in evidence_ids:
            edges.append({"edge_type": "derived_from", "source": atom_id, "target": evidence_id, "support_ids": [evidence_id]})
    return {
        "schema_version": GRAPH_SCHEMA,
        "question_id": catalog.get("question_id"),
        "nodes": nodes,
        "edges": edges,
    }


def _validate_typed_semantics(
    value: Any,
    *,
    evidence_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, Mapping) or set(value) != {"observations", "proposals"}:
        raise EvidenceOperationError("typed_semantics fields are invalid")
    observations_value = value.get("observations")
    proposals_value = value.get("proposals")
    if not isinstance(observations_value, list) or not isinstance(proposals_value, list):
        raise EvidenceOperationError("typed_semantics observations and proposals must be arrays")
    observations: list[dict[str, Any]] = []
    observation_ids: set[str] = set()
    for index, raw in enumerate(observations_value):
        if not isinstance(raw, Mapping):
            raise EvidenceOperationError(f"typed_semantics.observations[{index}] must be an object")
        observation_id = _text(raw.get("observation_id"))
        if not observation_id or observation_id in observation_ids:
            raise EvidenceOperationError(
                f"typed_semantics.observations[{index}].observation_id is invalid"
            )
        current = dict(raw)
        current["evidence_ids"] = _ids(
            raw.get("evidence_ids"),
            path=f"typed_semantics.observations[{index}].evidence_ids",
            allowed=evidence_ids,
        )
        observation_ids.add(observation_id)
        observations.append(current)
    proposals: list[dict[str, Any]] = []
    proposal_ids: set[str] = set()
    for index, raw in enumerate(proposals_value):
        if not isinstance(raw, Mapping):
            raise EvidenceOperationError(f"typed_semantics.proposals[{index}] must be an object")
        proposal_id = _text(raw.get("candidate_id"))
        if not proposal_id or proposal_id in proposal_ids:
            raise EvidenceOperationError(
                f"typed_semantics.proposals[{index}].candidate_id is invalid"
            )
        current = dict(raw)
        for field in ("source_evidence_ids", "evidence_ids"):
            if field in current:
                current[field] = _ids(
                    current[field],
                    path=f"typed_semantics.proposals[{index}].{field}",
                    allowed=evidence_ids,
                    allow_empty=True,
                )
        proposal_ids.add(proposal_id)
        proposals.append(current)
    return {"observations": observations, "proposals": proposals}


def validate_operation_plan(value: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict[str, Any]:
    required_root = {"schema_version", "requirements", "operations", "bundles"}
    optional_root = {"task_contract", "typed_semantics"}
    if (
        not isinstance(value, Mapping)
        or not required_root.issubset(value)
        or not set(value).issubset(required_root | optional_root)
    ):
        raise EvidenceOperationError("operation plan root fields are invalid")
    if value.get("schema_version") != PLAN_SCHEMA:
        raise EvidenceOperationError("operation plan schema_version is invalid")
    evidence_ids = {_text(item.get("evidence_id")) for item in catalog.get("evidence") or []}
    atom_ids = {_text(item.get("atom_id")) for item in catalog.get("atoms") or []}
    requirements_value, operations_value, bundles_value = value.get("requirements"), value.get("operations"), value.get("bundles")
    if not all(isinstance(item, list) for item in (requirements_value, operations_value, bundles_value)):
        raise EvidenceOperationError("requirements, operations, and bundles must be arrays")
    requirements: list[dict[str, Any]] = []
    requirement_ids: set[str] = set()
    for index, item in enumerate(requirements_value):
        if not isinstance(item, Mapping) or set(item) != {"requirement_id", "description", "evidence_ids"}:
            raise EvidenceOperationError(f"requirements[{index}] fields are invalid")
        requirement_id, description = _text(item.get("requirement_id")), _text(item.get("description"))
        if not requirement_id or requirement_id in requirement_ids or not description:
            raise EvidenceOperationError(f"requirements[{index}] identity is invalid")
        requirement_ids.add(requirement_id)
        requirements.append({"requirement_id": requirement_id, "description": description, "evidence_ids": _ids(item.get("evidence_ids"), path=f"requirements[{index}].evidence_ids", allowed=evidence_ids, allow_empty=True)})
    operations: list[dict[str, Any]] = []
    operation_ids: set[str] = set()
    for index, item in enumerate(operations_value):
        required = {"operation_id", "operation_type", "input_atom_ids", "input_evidence_ids", "parameters"}
        if not isinstance(item, Mapping) or set(item) != required:
            raise EvidenceOperationError(f"operations[{index}] fields are invalid")
        operation_id, operation_type = _text(item.get("operation_id")), _text(item.get("operation_type"))
        if not operation_id or operation_id in operation_ids or operation_type not in OPERATION_TYPES:
            raise EvidenceOperationError(f"operations[{index}] identity or type is invalid")
        if not isinstance(item.get("parameters"), Mapping):
            raise EvidenceOperationError(f"operations[{index}].parameters must be an object")
        operation_ids.add(operation_id)
        operations.append(
            {
                "operation_id": operation_id,
                "operation_type": operation_type,
                "input_atom_ids": _ids(item.get("input_atom_ids"), path=f"operations[{index}].input_atom_ids", allowed=atom_ids),
                "input_evidence_ids": _ids(item.get("input_evidence_ids"), path=f"operations[{index}].input_evidence_ids", allowed=evidence_ids),
                "parameters": dict(item["parameters"]),
            }
        )
    bundles: list[dict[str, Any]] = []
    bundle_ids: set[str] = set()
    for index, item in enumerate(bundles_value):
        if not isinstance(item, Mapping) or set(item) != {"bundle_id", "role", "evidence_ids"}:
            raise EvidenceOperationError(f"bundles[{index}] fields are invalid")
        bundle_id, role = _text(item.get("bundle_id")), _text(item.get("role"))
        if not bundle_id or bundle_id in bundle_ids or not role:
            raise EvidenceOperationError(f"bundles[{index}] identity is invalid")
        bundle_ids.add(bundle_id)
        bundles.append({"bundle_id": bundle_id, "role": role, "evidence_ids": _ids(item.get("evidence_ids"), path=f"bundles[{index}].evidence_ids", allowed=evidence_ids)})
    normalized: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "requirements": requirements,
        "operations": operations,
        "bundles": bundles,
    }
    if "task_contract" in value:
        try:
            task_contract = validate_task_contract(value["task_contract"])
        except TaskContractError as exc:
            raise EvidenceOperationError(f"task_contract is invalid: {exc}") from exc
        premise_ids = {
            item["premise_id"] for item in task_contract["premises"]
        }
        required_memory_ids = {
            item["premise_id"]
            for item in task_contract["premises"]
            if item["source"] == "memory" and item["necessity"] == "required"
        }
        if not requirement_ids.issubset(premise_ids):
            raise EvidenceOperationError(
                "every legacy requirement must share an ID with a task premise"
            )
        if not required_memory_ids.issubset(requirement_ids):
            raise EvidenceOperationError(
                "every required memory premise must share an ID with a legacy requirement"
            )
        normalized["task_contract"] = task_contract
    if "typed_semantics" in value:
        normalized["typed_semantics"] = _validate_typed_semantics(
            value["typed_semantics"], evidence_ids=evidence_ids
        )
    return normalized


def _decimal(atom: Mapping[str, Any]) -> Decimal:
    if atom.get("atom_type") not in {"number", "currency", "quantity"}:
        raise EvidenceOperationError(f"atom {atom.get('atom_id')} is not numeric")
    try:
        return Decimal(str(atom["normalized_value"]))
    except (KeyError, InvalidOperation) as exc:
        raise EvidenceOperationError(f"atom {atom.get('atom_id')} has an invalid number") from exc


def _date(atom: Mapping[str, Any]) -> date:
    if atom.get("atom_type") != "date":
        raise EvidenceOperationError(f"atom {atom.get('atom_id')} is not a date")
    try:
        return datetime.strptime(str(atom["normalized_value"]), "%Y-%m-%d").date()
    except (KeyError, ValueError) as exc:
        raise EvidenceOperationError(f"atom {atom.get('atom_id')} has an invalid date") from exc


def _json_number(value: Decimal) -> int | float:
    return int(value) if value == value.to_integral_value() else float(value)


def execute_operation(operation: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict[str, Any]:
    atoms = {_text(item.get("atom_id")): item for item in catalog.get("atoms") or []}
    input_ids = list(operation.get("input_atom_ids") or [])
    selected = [atoms[item] for item in input_ids if item in atoms]
    if len(selected) != len(input_ids):
        raise EvidenceOperationError("operation references an unavailable atom")
    operation_type = _text(operation.get("operation_type"))
    parameters = dict(operation.get("parameters") or {})
    result: dict[str, Any]
    if operation_type == "date_difference":
        if len(selected) != 2:
            raise EvidenceOperationError("date_difference requires exactly two date atoms")
        days = (_date(selected[1]) - _date(selected[0])).days
        if not bool(parameters.get("signed", False)):
            days = abs(days)
        result = {"value": days, "unit": "days"}
    elif operation_type == "date_order":
        ordered = sorted(selected, key=lambda item: (_date(item), input_ids.index(item["atom_id"])))
        result = {"ordered_atom_ids": [item["atom_id"] for item in ordered], "ordered_values": [item["normalized_value"] for item in ordered]}
    elif operation_type in {"numeric_sum", "numeric_difference", "numeric_average"}:
        values = [_decimal(item) for item in selected]
        if not values:
            raise EvidenceOperationError(f"{operation_type} requires numeric atoms")
        units = {str(item.get("unit") or "") for item in selected}
        if len(units) > 1 and not bool(parameters.get("allow_mixed_units", False)):
            raise EvidenceOperationError(f"{operation_type} received mixed units")
        if operation_type == "numeric_sum":
            value = sum(values, Decimal(0))
        elif operation_type == "numeric_difference":
            if len(values) != 2:
                raise EvidenceOperationError("numeric_difference requires exactly two atoms")
            value = values[0] - values[1]
        else:
            value = sum(values, Decimal(0)) / Decimal(len(values))
        result = {"value": _json_number(value), "unit": next(iter(units), "")}
    elif operation_type == "count_distinct":
        if not selected or any(item.get("atom_type") != "entity" for item in selected):
            raise EvidenceOperationError("count_distinct requires entity atoms")
        result = {"value": len(dict.fromkeys(str(item.get("normalized_value")) for item in selected)), "unit": "items"}
    elif operation_type == "ordered_unique_list":
        values = list(dict.fromkeys(str(item.get("normalized_value")) for item in selected))
        result = {"values": values, "count": len(values)}
    elif operation_type in {"entity_exact_match", "entity_mismatch"}:
        if len(selected) != 2:
            raise EvidenceOperationError(f"{operation_type} requires exactly two atoms")
        equal = str(selected[0].get("normalized_value")).casefold() == str(selected[1].get("normalized_value")).casefold()
        result = {"value": equal if operation_type == "entity_exact_match" else not equal}
    elif operation_type == "set_difference":
        left_ids = parameters.get("left_atom_ids")
        right_ids = parameters.get("right_atom_ids")
        left = _ids(left_ids, path="parameters.left_atom_ids", allowed=set(input_ids), allow_empty=True)
        right = _ids(right_ids, path="parameters.right_atom_ids", allowed=set(input_ids), allow_empty=True)
        right_values = {str(atoms[item].get("normalized_value")) for item in right}
        result = {"values": [str(atoms[item].get("normalized_value")) for item in left if str(atoms[item].get("normalized_value")) not in right_values]}
    else:
        raise EvidenceOperationError(f"unsupported deterministic operation: {operation_type}")
    return {
        "operation_id": operation["operation_id"],
        "operation_type": operation_type,
        "status": "completed",
        "input_atom_ids": input_ids,
        "support_ids": list(operation.get("input_evidence_ids") or []),
        "result": result,
    }


def execute_operation_plan(plan: Mapping[str, Any], catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    validated = validate_operation_plan(plan, catalog)
    output: list[dict[str, Any]] = []
    for operation in validated["operations"]:
        try:
            output.append(execute_operation(operation, catalog))
        except EvidenceOperationError as exc:
            output.append(
                {
                    "operation_id": operation["operation_id"],
                    "operation_type": operation["operation_type"],
                    "status": "error",
                    "input_atom_ids": operation["input_atom_ids"],
                    "support_ids": operation["input_evidence_ids"],
                    "error": str(exc),
                }
            )
    return output


def _explicit_relative_day_target(question: str) -> dict[str, Any] | None:
    for pattern, multiplier, tolerance in EXPLICIT_RELATIVE_DURATION_PATTERNS:
        match = pattern.search(question)
        if match is None:
            continue
        raw_count = match.group("count").casefold()
        count = 1 if raw_count in {"a", "an", "one"} else int(raw_count)
        return {
            "surface": match.group(0),
            "expected_days": count * multiplier,
            "tolerance_days": max(tolerance, tolerance * count),
        }
    return None


def certify_operation_results(
    catalog: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Certify only a unique operation matching an explicit query duration."""
    output = [dict(item) for item in results]
    for item in output:
        if item.get("status") == "completed":
            item["answer_authoritative"] = False
    target = _explicit_relative_day_target(_text(catalog.get("question")))
    if target is None:
        return output
    candidates = [
        item
        for item in output
        if item.get("status") == "completed"
        and item.get("operation_type") == "date_difference"
        and isinstance(item.get("result"), Mapping)
        and item["result"].get("unit") == "days"
        and isinstance(item["result"].get("value"), int)
        and abs(item["result"]["value"] - target["expected_days"])
        <= target["tolerance_days"]
    ]
    identities = {
        (
            item["result"]["value"],
            tuple(sorted(_text(value) for value in item.get("support_ids") or [])),
        )
        for item in candidates
    }
    if len(candidates) != 1 or len(identities) != 1:
        return output
    candidates[0]["answer_authoritative"] = True
    candidates[0]["certification"] = {
        "schema_version": "tmcra.explicit-relative-duration-certification.v1",
        **target,
    }
    return output


def operation_plan_structural_risks(plan: Mapping[str, Any]) -> list[str]:
    """Compute local review signals without trusting model-supplied risk fields."""
    contract = plan.get("task_contract")
    if not isinstance(contract, Mapping):
        return []
    risks = structural_risk_signals(contract, planner_present=True)
    if (
        unbound_memory_requirement_ids(plan)
        and RISK_PLANNER_MISSING_WITH_PLAUSIBLE_SOURCE not in risks
    ):
        risks.append(RISK_PLANNER_MISSING_WITH_PLAUSIBLE_SOURCE)
    return risks


def unbound_memory_requirement_ids(plan: Mapping[str, Any]) -> list[str]:
    """Return only required memory premises that lack Source bindings."""
    contract = plan.get("task_contract")
    premise_by_id = {
        _text(item.get("premise_id")): item
        for item in (contract.get("premises") if isinstance(contract, Mapping) else [])
        if isinstance(item, Mapping)
    }
    output: list[str] = []
    for requirement in plan.get("requirements") or []:
        if not isinstance(requirement, Mapping) or requirement.get("evidence_ids"):
            continue
        requirement_id = _text(requirement.get("requirement_id"))
        premise = premise_by_id.get(requirement_id)
        if premise is None:
            output.append(requirement_id)
            continue
        if (
            premise.get("source") == "memory"
            and premise.get("necessity") == "required"
        ):
            output.append(requirement_id)
    return output


def _execute_typed_semantics(
    plan: Mapping[str, Any],
    *,
    evidence_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    typed = plan.get("typed_semantics")
    if not isinstance(typed, Mapping):
        return [], {
            "status": "not_proposed",
            "advisory": True,
            "authoritative": False,
            "accepted": [],
            "rejected": [],
        }
    evaluated = evaluate_proposals(
        typed.get("observations") or [], typed.get("proposals") or []
    )
    accepted_results: list[dict[str, Any]] = []
    accepted_report: list[dict[str, Any]] = []
    rejected_report = [dict(item) for item in evaluated.get("rejected") or []]
    used_operation_ids = {
        _text(item.get("operation_id"))
        for item in plan.get("operations") or []
        if isinstance(item, Mapping)
    }
    for index, item in enumerate(evaluated.get("accepted") or [], start=1):
        current = dict(item)
        support_ids = list(current.get("source_evidence_ids") or [])
        unknown = sorted(set(support_ids) - evidence_ids)
        if unknown:
            current["status"] = "rejected"
            current.pop("value", None)
            current.pop("value_kind", None)
            current.pop("unit", None)
            current.pop("result", None)
            current.pop("operation_results", None)
            current.setdefault("diagnostics", []).append(
                {
                    "code": "UNKNOWN_SOURCE_EVIDENCE",
                    "message": "typed proposal cites evidence outside the immutable reservoir",
                    "path": "source_evidence_ids",
                }
            )
            rejected_report.append(current)
            continue
        operation_id = f"TS{index:03d}"
        while operation_id in used_operation_ids:
            index += 1
            operation_id = f"TS{index:03d}"
        used_operation_ids.add(operation_id)
        result = {
            "value": current.get("value"),
            "value_kind": current.get("value_kind"),
            "unit": current.get("unit"),
        }
        accepted_results.append(
            {
                "operation_id": operation_id,
                "operation_type": _text((current.get("result") or {}).get("operation"))
                or "typed_semantic_program",
                "status": "completed",
                "input_atom_ids": [],
                "support_ids": support_ids,
                "result": result,
                "typed_candidate_id": current.get("candidate_id"),
                "advisory": True,
                "authoritative": False,
            }
        )
        accepted_report.append(
            {
                "candidate_id": current.get("candidate_id"),
                "operation_id": operation_id,
                "support_ids": support_ids,
                "result": result,
            }
        )
    report = {
        "status": "evaluated",
        "advisory": True,
        "authoritative": False,
        "accepted": accepted_report,
        "rejected": rejected_report,
    }
    return accepted_results, report


def compile_evidence_packet(row: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    catalog = build_evidence_catalog(row)
    graph = build_query_evidence_graph(catalog)
    validated = validate_operation_plan(plan, catalog)
    operation_results = certify_operation_results(
        catalog, execute_operation_plan(validated, catalog)
    )
    typed_results, typed_report = _execute_typed_semantics(
        validated,
        evidence_ids={item["evidence_id"] for item in catalog["evidence"]},
    )
    operation_results.extend(typed_results)
    operation_by_id = {item["operation_id"]: item for item in operation_results}
    evidence_by_id = {item["evidence_id"]: item for item in catalog["evidence"]}
    task_contract = validated.get("task_contract")
    premise_by_id = {
        _text(item.get("premise_id")): item
        for item in (
            task_contract.get("premises")
            if isinstance(task_contract, Mapping)
            else []
        )
        if isinstance(item, Mapping)
    }
    requirement_coverage: list[dict[str, Any]] = []
    for requirement in validated["requirements"]:
        premise = premise_by_id.get(requirement["requirement_id"])
        premise_source = _text(premise.get("source")) if premise else "memory"
        if requirement["evidence_ids"]:
            state = "satisfied"
            coverage_origin = "source_evidence"
        elif premise_source in {"query_context", "model_knowledge"}:
            state = "satisfied"
            coverage_origin = premise_source
        else:
            state = "missing"
            coverage_origin = premise_source or "memory"
        requirement_coverage.append(
            {
                **requirement,
                "state": state,
                "premise_source": premise_source,
                "coverage_origin": coverage_origin,
            }
        )
    if any(item["status"] == "error" for item in operation_results):
        failed_support = {support for item in operation_results if item["status"] == "error" for support in item["support_ids"]}
        for requirement in requirement_coverage:
            if failed_support.intersection(requirement["evidence_ids"]):
                requirement["state"] = "invalid_operation"
    bundles = [
        {
            **bundle,
            "evidence": [evidence_by_id[item] for item in bundle["evidence_ids"]],
        }
        for bundle in validated["bundles"]
    ]
    structural_risks = operation_plan_structural_risks(validated)
    question_contract: dict[str, Any] = {
        "question": catalog["question"],
        "question_date": catalog["question_date"],
        "requirement_count": len(requirement_coverage),
        "operation_count": len(operation_results),
    }
    if isinstance(task_contract, Mapping):
        question_contract["output_origin"] = task_contract["output_origin"]
        question_contract["task_contract"] = task_contract
    return {
        "schema_version": PACKET_SCHEMA,
        "packet_compiler_version": PACKET_COMPILER_VERSION,
        "question_id": catalog["question_id"],
        "question_contract": question_contract,
        "task_contract": task_contract,
        "structural_risk_signals": structural_risks,
        "typed_semantics_report": typed_report,
        "requirement_coverage": requirement_coverage,
        "operation_results": operation_results,
        "evidence_bundles": bundles,
        "raw_evidence_reservoir": catalog["evidence"],
        "lexical_anchor_ids": list(catalog.get("lexical_anchor_ids") or []),
        "atom_catalog": catalog,
        "query_evidence_graph": graph,
        "operation_plan": validated,
        "operation_result_ids": sorted(operation_by_id),
    }


def validate_evidence_bound_answer(value: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, Any]:
    root_fields = {"schema_version", "answerability", "claims", "missing_requirements", "answer"}
    if not isinstance(value, Mapping) or not root_fields.issubset(value):
        raise EvidenceOperationError("answer root fields are invalid")
    if value.get("schema_version") != ANSWER_SCHEMA:
        raise EvidenceOperationError("answer schema_version is invalid")
    answerability = _text(value.get("answerability"))
    if answerability not in {"sufficient", "insufficient"}:
        raise EvidenceOperationError("answerability is invalid")
    evidence_ids = {_text(item.get("evidence_id")) for item in packet.get("raw_evidence_reservoir") or []}
    operation_ids = {
        _text(item.get("operation_id"))
        for item in packet.get("operation_results") or []
        if item.get("status") == "completed"
        and item.get("answer_authoritative") is True
    }
    requirement_ids = {_text(item.get("requirement_id")) for item in packet.get("requirement_coverage") or []}
    task_contract = packet.get("task_contract")
    if not isinstance(task_contract, Mapping):
        question_contract = packet.get("question_contract")
        task_contract = (
            question_contract.get("task_contract")
            if isinstance(question_contract, Mapping)
            else None
        )
    output_origin = (
        _text(task_contract.get("output_origin"))
        if isinstance(task_contract, Mapping)
        else ""
    )
    claims_value = value.get("claims")
    if not isinstance(claims_value, list):
        raise EvidenceOperationError("claims must be an array")
    claims: list[dict[str, Any]] = []
    claim_ids: set[str] = set()
    for index, claim in enumerate(claims_value):
        identity_fields = {"claim_id", "text"}
        if not isinstance(claim, Mapping) or not identity_fields.issubset(claim):
            raise EvidenceOperationError(f"claims[{index}] fields are invalid")
        claim_id, text = _text(claim.get("claim_id")), _text(claim.get("text"))
        if not claim_id or claim_id in claim_ids or not text:
            raise EvidenceOperationError(f"claims[{index}] identity is invalid")
        claim_ids.add(claim_id)
        support_ids = _ids(claim.get("support_ids", []), path=f"claims[{index}].support_ids", allowed=evidence_ids, allow_empty=True)
        origin = _text(claim.get("origin"))
        raw_computation_ids = claim.get("computation_ids", [])
        try:
            computation_ids = _ids(
                raw_computation_ids,
                path=f"claims[{index}].computation_ids",
                allowed=operation_ids,
                allow_empty=True,
            )
        except EvidenceOperationError:
            # Older answer prompts used memory_derived for both deterministic
            # operation results and source-grounded synthesis. Preserve the
            # cited Source binding while canonicalizing only that legacy case.
            if origin != "memory_derived" or not support_ids:
                raise
            computation_ids = []
            origin = "memory_inference"
        if not origin:
            origin = "memory_derived" if computation_ids else "memory_fact"
        if origin == "memory_derived" and not computation_ids and support_ids:
            origin = "memory_inference"
        if origin not in ANSWER_CLAIM_ORIGINS:
            raise EvidenceOperationError(f"claims[{index}].origin is invalid")
        if origin == "memory_fact" and not support_ids:
            raise EvidenceOperationError(f"claims[{index}] lacks source support")
        if origin == "memory_inference" and not support_ids:
            raise EvidenceOperationError(f"claims[{index}] lacks source support")
        if origin == "memory_derived" and not computation_ids:
            raise EvidenceOperationError(f"claims[{index}] lacks computation support")
        if origin in {"query_context", "model_knowledge"}:
            if output_origin != "memory_conditioned_generation":
                raise EvidenceOperationError(
                    f"claims[{index}].origin is not allowed for {output_origin or 'unknown'}"
                )
            if support_ids or computation_ids:
                raise EvidenceOperationError(
                    f"claims[{index}] non-memory origin cannot cite memory evidence"
                )
        claims.append({"claim_id": claim_id, "text": text, "origin": origin, "support_ids": support_ids, "computation_ids": computation_ids})
    missing = _ids(value.get("missing_requirements"), path="missing_requirements", allowed=requirement_ids, allow_empty=True)
    answer = _text(value.get("answer"))
    if not answer:
        raise EvidenceOperationError("answer text is empty")
    # Planner coverage is advisory. The answer model still sees the immutable
    # raw reservoir and may correct either a false missing or false satisfied
    # judgment. Final claims remain strictly bound to supplied evidence or
    # deterministic operation results.
    if answerability == "sufficient" and (missing or not claims):
        raise EvidenceOperationError("sufficient answer has unresolved requirements")
    if answerability == "insufficient" and not missing:
        raise EvidenceOperationError("insufficient answer must identify missing requirements")
    if (
        answerability == "sufficient"
        and output_origin == "memory_conditioned_generation"
        and not any(
            claim["origin"]
            in {"memory_fact", "memory_inference", "memory_derived"}
            for claim in claims
        )
    ):
        raise EvidenceOperationError(
            "memory-conditioned answer lacks a memory-bound claim"
        )
    return {"schema_version": ANSWER_SCHEMA, "answerability": answerability, "claims": claims, "missing_requirements": missing, "answer": answer}
