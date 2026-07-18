"""Semantic evidence contracts layered above immutable TMCRA Source evidence."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from tmcra_v4_evidence_operations import (
    OPERATION_TYPES,
    PACKET_SCHEMA,
    EvidenceOperationError,
    build_evidence_catalog,
    execute_operation_plan,
    validate_operation_plan,
)
from tmcra_v4_typed_semantics import evaluate_proposals


TASK_CONTRACT_SCHEMA = "tmcra.task-contract.v2"
RESOLUTION_PLAN_SCHEMA = "tmcra.evidence-resolution-plan.v1"
SEMANTIC_PACKET_SCHEMA = "tmcra.semantic-evidence-packet.v1"
SEMANTIC_ANSWER_SCHEMA = "tmcra.semantic-bound-answer.v1"

OUTPUT_SHAPES = {
    "scalar",
    "list",
    "set",
    "count",
    "boolean",
    "date",
    "duration",
    "structured",
    "free_text",
}
OUTPUT_ORIGINS = {
    "memory_direct",
    "memory_derived",
    "memory_conditioned",
    "external_required",
}
PREMISE_ROLES = {"fact", "operand", "constraint", "scope", "counterevidence"}
PREMISE_NECESSITY = {"required", "optional"}
PREMISE_SOURCES = {"memory", "query_context", "model_knowledge", "external_tool"}
ALLOWED_DERIVATIONS = set(OPERATION_TYPES) | {
    "semantic_composition",
    "constraint_application",
}
CLAIM_POLARITIES = {"positive", "negative", "unknown"}
CLAIM_MODALITIES = {
    "asserted",
    "experienced",
    "preferred",
    "planned",
    "recommended",
    "uncertain",
}
BINDING_COVERAGE = {"complete", "partial", "conflicting", "absent"}
BINDING_RELATIONS = {"direct", "derived", "constraint", "context", "counterevidence"}
EXECUTABILITY = {
    "directly_answerable",
    "partially_answerable",
    "derivable",
    "memory_conditioned_generation",
    "requires_external_tool",
    "requires_clarification",
    "not_answerable",
}
CLAIM_OPERATION_TYPES = {
    "date_difference",
    "date_order",
    "numeric_sum",
    "numeric_difference",
    "numeric_average",
    "count_distinct",
    "ordered_unique_list",
    "latest_state",
    "entity_exact_match",
    "entity_mismatch",
}
ANSWER_POLICIES = {
    "answer",
    "answer_partial",
    "answer_with_memory_context",
    "call_tool",
    "clarify",
    "abstain",
}
ANSWER_STATUSES = {
    "answered",
    "partial",
    "answered_with_context",
    "requires_tool",
    "clarification",
    "not_answerable",
}
CLAIM_ORIGINS = {"memory_fact", "memory_derived", "model_knowledge", "tool_result"}


class SemanticEvidenceError(ValueError):
    pass


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: Any, *, path: str, allowed: set[str] | None = None) -> list[str]:
    if not isinstance(value, list):
        raise SemanticEvidenceError(f"{path} must be an array")
    output: list[str] = []
    for item in value:
        item_text = _text(item)
        if not item_text:
            raise SemanticEvidenceError(f"{path} contains an empty value")
        if allowed is not None and item_text not in allowed:
            raise SemanticEvidenceError(f"{path} contains unknown value: {item_text}")
        if item_text in output:
            raise SemanticEvidenceError(f"{path} contains a duplicate value: {item_text}")
        output.append(item_text)
    return output


def _align_quote(quote: str, text: str) -> str:
    if quote in text:
        return quote
    location = text.casefold().find(quote.casefold())
    if location >= 0:
        return text[location : location + len(quote)]
    tokens = quote.split()
    if not tokens:
        return ""
    match = re.search(
        r"\s+".join(re.escape(token) for token in tokens),
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(0)

    punctuation = str.maketrans(
        {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u00a0": " ",
        }
    )

    def searchable(value: str) -> tuple[str, list[int]]:
        output: list[str] = []
        positions: list[int] = []
        prior_space = False
        for index, character in enumerate(value):
            normalized = unicodedata.normalize("NFKC", character).translate(punctuation).casefold()
            for current in normalized:
                if current.isspace():
                    if prior_space:
                        continue
                    current = " "
                    prior_space = True
                else:
                    prior_space = False
                output.append(current)
                positions.append(index)
        return "".join(output), positions

    normalized_quote, _ = searchable(quote)
    normalized_text, positions = searchable(text)
    location = normalized_text.find(normalized_quote)
    if location < 0 or not normalized_quote:
        return ""
    return text[positions[location] : positions[location + len(normalized_quote) - 1] + 1]


def validate_task_contract(
    value: Mapping[str, Any], query_context: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    root = {"schema_version", "target", "output", "scope", "premises", "allowed_derivations"}
    if not isinstance(value, Mapping) or set(value) != root:
        raise SemanticEvidenceError("task contract root fields are invalid")
    if value.get("schema_version") != TASK_CONTRACT_SCHEMA:
        raise SemanticEvidenceError("task contract schema_version is invalid")
    target = value.get("target")
    if not isinstance(target, Mapping) or set(target) != {"description", "subject", "relation"}:
        raise SemanticEvidenceError("task contract target fields are invalid")
    normalized_target = {key: _text(target.get(key)) for key in ("description", "subject", "relation")}
    if not all(normalized_target.values()):
        raise SemanticEvidenceError("task contract target is incomplete")
    output = value.get("output")
    output_fields = {"shape", "cardinality", "ordering", "unit", "origin"}
    if not isinstance(output, Mapping) or set(output) != output_fields:
        raise SemanticEvidenceError("task contract output fields are invalid")
    normalized_output = {key: _text(output.get(key)) for key in output_fields}
    if normalized_output["shape"] not in OUTPUT_SHAPES or normalized_output["origin"] not in OUTPUT_ORIGINS:
        raise SemanticEvidenceError("task contract output shape or origin is invalid")
    if not normalized_output["cardinality"] or not normalized_output["ordering"]:
        raise SemanticEvidenceError("task contract output cardinality or ordering is empty")
    scope = value.get("scope")
    if not isinstance(scope, Mapping) or set(scope) != {"temporal", "entity", "context"}:
        raise SemanticEvidenceError("task contract scope fields are invalid")
    normalized_scope = {key: _text(scope.get(key)) for key in ("temporal", "entity", "context")}
    premises_value = value.get("premises")
    if not isinstance(premises_value, list):
        raise SemanticEvidenceError("task contract premises must be an array")
    premises: list[dict[str, str]] = []
    premise_ids: set[str] = set()
    for index, item in enumerate(premises_value):
        fields = {"premise_id", "description", "role", "necessity", "source", "context_quote"}
        if not isinstance(item, Mapping) or set(item) != fields:
            raise SemanticEvidenceError(f"premises[{index}] fields are invalid")
        premise = {key: _text(item.get(key)) for key in fields}
        if not premise["premise_id"] or premise["premise_id"] in premise_ids or not premise["description"]:
            raise SemanticEvidenceError(f"premises[{index}] identity is invalid")
        if premise["role"] not in PREMISE_ROLES or premise["necessity"] not in PREMISE_NECESSITY:
            raise SemanticEvidenceError(f"premises[{index}] role or necessity is invalid")
        if premise["source"] not in PREMISE_SOURCES:
            raise SemanticEvidenceError(f"premises[{index}] source is invalid")
        if premise["source"] == "query_context":
            if not premise["context_quote"]:
                raise SemanticEvidenceError(f"premises[{index}] query context quote is empty")
            if query_context is not None:
                context_text = "\n".join(
                    _text(query_context.get(key))
                    for key in ("question", "question_date")
                    if _text(query_context.get(key))
                )
                aligned = _align_quote(premise["context_quote"], context_text)
                if not aligned:
                    raise SemanticEvidenceError(f"premises[{index}] quote is not in query context")
                premise["context_quote"] = aligned
        elif premise["context_quote"]:
            raise SemanticEvidenceError(f"premises[{index}] non-context premise has a context quote")
        premise_ids.add(premise["premise_id"])
        premises.append(premise)
    if not any(item["necessity"] == "required" for item in premises):
        raise SemanticEvidenceError("task contract needs at least one required premise")
    if normalized_output["origin"] in {"memory_direct", "memory_derived", "memory_conditioned"} and not any(
        item["source"] == "memory" and item["necessity"] == "required"
        for item in premises
    ):
        raise SemanticEvidenceError("memory task needs at least one required memory premise")
    derivations = _string_list(value.get("allowed_derivations"), path="allowed_derivations", allowed=ALLOWED_DERIVATIONS)
    return {
        "schema_version": TASK_CONTRACT_SCHEMA,
        "target": normalized_target,
        "output": normalized_output,
        "scope": normalized_scope,
        "premises": premises,
        "allowed_derivations": derivations,
    }


def task_contract_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    question = _text(row.get("question"))
    if not question:
        raise SemanticEvidenceError("question is empty")
    payload = {
        "question": question,
        "question_date": _text(row.get("question_date")) or "unknown",
    }
    dialogue_state = row.get("dialogue_state")
    if dialogue_state is not None:
        if not isinstance(dialogue_state, Mapping):
            raise SemanticEvidenceError("dialogue_state must be an object")
        payload["dialogue_state"] = dict(dialogue_state)
    return payload


_SOURCE_REVIEW_STOPWORDS = {
    "a", "an", "and", "are", "for", "from", "has", "have", "in", "is",
    "of", "on", "the", "to", "user", "with", "request", "current",
}


def _source_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 1 and token not in _SOURCE_REVIEW_STOPWORDS
    }


def normalize_task_contract_sources(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    contract = validate_task_contract(contract)
    premises = [dict(item) for item in contract["premises"]]
    context = [item for item in premises if item["source"] == "query_context"]
    required_memory_count = sum(
        item["source"] == "memory" and item["necessity"] == "required"
        for item in premises
    )
    dropped: set[str] = set()
    warnings: list[str] = []
    for memory_premise in premises:
        if memory_premise["source"] != "memory":
            continue
        memory_tokens = _source_tokens(memory_premise["description"])
        if len(memory_tokens) < 2:
            continue
        for context_premise in context:
            context_tokens = _source_tokens(
                context_premise["description"] + " " + context_premise["context_quote"]
            )
            can_drop_required = (
                memory_premise["necessity"] != "required" or required_memory_count > 1
            )
            if memory_tokens.issubset(context_tokens) and can_drop_required:
                dropped.add(memory_premise["premise_id"])
                if memory_premise["necessity"] == "required":
                    required_memory_count -= 1
                warnings.append(
                    f"{memory_premise['premise_id']}:dropped_query_context_duplicate_of_{context_premise['premise_id']}"
                )
                break
    normalized = dict(contract)
    normalized["premises"] = [
        item for item in premises if item["premise_id"] not in dropped
    ]
    return validate_task_contract(normalized), warnings


def task_contract_source_review_reasons(contract: Mapping[str, Any]) -> list[str]:
    contract = validate_task_contract(contract)
    memory = [item for item in contract["premises"] if item["source"] == "memory"]
    context = [item for item in contract["premises"] if item["source"] == "query_context"]

    reasons: list[str] = []
    for memory_premise in memory:
        memory_tokens = _source_tokens(memory_premise["description"])
        for context_premise in context:
            context_tokens = _source_tokens(
                context_premise["description"] + " " + context_premise["context_quote"]
            )
            if not memory_tokens or not context_tokens:
                continue
            overlap = len(memory_tokens & context_tokens)
            containment = overlap / min(len(memory_tokens), len(context_tokens))
            if overlap >= 2 and containment >= 0.6:
                reasons.append(
                    f"{memory_premise['premise_id']} overlaps query-context premise {context_premise['premise_id']}"
                )
    return reasons


def requires_exhaustive_resolution_review(contract: Mapping[str, Any]) -> bool:
    """Return whether Source membership must be exhaustively audited."""
    contract = validate_task_contract(contract)
    output = contract["output"]
    return output["origin"] in {"memory_direct", "memory_derived"} and output[
        "shape"
    ] in {"count", "list", "set"}


def normalize_resolution_output(
    value: Mapping[str, Any],
    contract: Mapping[str, Any],
    catalog: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Remove only semantically unused resolver output before strict validation."""
    contract = validate_task_contract(contract)
    normalized = dict(value)
    warnings: list[str] = []
    memory_premises = {
        item["premise_id"] for item in contract["premises"] if item["source"] == "memory"
    }
    valid_evidence_ids = {
        _text(item.get("evidence_id")) for item in (catalog or {}).get("evidence") or []
    }
    valid_atom_ids = {
        _text(item.get("atom_id")) for item in (catalog or {}).get("atoms") or []
    }
    evidence_text = {
        _text(item.get("evidence_id")): str(item.get("text") or "")
        for item in (catalog or {}).get("evidence") or []
    }
    claim_inputs = []
    valid_input_claim_ids: set[str] = set()
    for index, raw in enumerate(value.get("claims") or []):
        if not isinstance(raw, Mapping) or catalog is None:
            claim_inputs.append(dict(raw) if isinstance(raw, Mapping) else raw)
            if isinstance(raw, Mapping) and _text(raw.get("claim_id")):
                valid_input_claim_ids.add(_text(raw.get("claim_id")))
            continue
        claim = dict(raw)
        claim_id = _text(claim.get("claim_id"))
        evidence_id = _text(claim.get("evidence_id"))
        quote = _text(claim.get("source_quote"))
        aligned = _align_quote(quote, evidence_text.get(evidence_id, ""))
        if (
            not claim_id
            or claim_id in valid_input_claim_ids
            or evidence_id not in valid_evidence_ids
            or not _text(claim.get("subject"))
            or not _text(claim.get("predicate"))
            or not aligned
        ):
            warnings.append(f"claims[{index}]:dropped_invalid_or_ungrounded_claim")
            continue
        claim["source_quote"] = aligned
        claim_inputs.append(claim)
        valid_input_claim_ids.add(claim_id)
    bindings = []
    for index, raw in enumerate(value.get("bindings") or []):
        if isinstance(raw, Mapping) and _text(raw.get("premise_id")) not in memory_premises:
            warnings.append(f"bindings[{index}]:dropped_non_memory_premise")
            continue
        binding = dict(raw) if isinstance(raw, Mapping) else raw
        if isinstance(binding, dict) and catalog is not None:
            filtered_claim_ids = [
                item
                for item in (binding.get("claim_ids") or [])
                if isinstance(item, str) and item in valid_input_claim_ids
            ]
            if filtered_claim_ids != binding.get("claim_ids"):
                warnings.append(f"bindings[{index}]:dropped_invalid_claim_reference")
                binding["claim_ids"] = filtered_claim_ids
        bindings.append(binding)
    operations = []
    for index, raw in enumerate(value.get("operations") or []):
        if not isinstance(raw, Mapping):
            operations.append(raw)
            continue
        operation = dict(raw)
        operation_id = _text(operation.get("operation_id"))
        output_refs = [
            item
            for item in (operation.get("output_refs") or [])
            if item == "TARGET" or item in memory_premises
        ]
        if output_refs != operation.get("output_refs"):
            warnings.append(f"operations[{index}]:dropped_non_memory_output_ref")
            operation["output_refs"] = output_refs
        if catalog is not None:
            input_claim_ids = [
                item
                for item in (operation.get("input_claim_ids") or [])
                if isinstance(item, str) and item in valid_input_claim_ids
            ]
            if input_claim_ids != operation.get("input_claim_ids"):
                warnings.append(f"operations[{index}]:dropped_invalid_input_claim_id")
                operation["input_claim_ids"] = input_claim_ids
            input_evidence_ids = [
                item
                for item in (operation.get("input_evidence_ids") or [])
                if isinstance(item, str) and item in valid_evidence_ids
            ]
            if input_evidence_ids != operation.get("input_evidence_ids"):
                warnings.append(f"operations[{index}]:dropped_unknown_input_evidence_id")
                operation["input_evidence_ids"] = input_evidence_ids
            input_atom_ids = [
                item
                for item in (operation.get("input_atom_ids") or [])
                if isinstance(item, str) and item in valid_atom_ids
            ]
            if input_atom_ids != operation.get("input_atom_ids"):
                warnings.append(f"operations[{index}]:dropped_unknown_input_atom_id")
                operation["input_atom_ids"] = input_atom_ids
        unauthorized = _text(operation.get("operation_type")) not in contract["allowed_derivations"]
        operation_type = _text(operation.get("operation_type"))
        claim_operand_count = len(operation.get("input_claim_ids") or [])
        atom_operand_count = len(operation.get("input_atom_ids") or [])
        originally_had_claims = bool(raw.get("input_claim_ids"))
        exact_two = operation_type in {
            "date_difference",
            "numeric_difference",
            "entity_exact_match",
            "entity_mismatch",
        }
        if (
            (originally_had_claims and not claim_operand_count)
            or (exact_two and claim_operand_count and claim_operand_count != 2)
            or (exact_two and atom_operand_count and atom_operand_count != 2)
            or (not claim_operand_count and not atom_operand_count)
        ):
            warnings.append(f"operations[{index}]:dropped_operation_with_invalid_arity")
            continue
        if (
            unauthorized
            and contract["output"]["origin"] != "memory_derived"
        ):
            warnings.append(f"operations[{index}]:dropped_unauthorized_non_derived_operation")
            continue
        operations.append(operation)
    operation_ids = {
        _text(item.get("operation_id")) for item in operations if isinstance(item, Mapping)
    }
    operation_outputs = {
        _text(item.get("operation_id")): set(item.get("output_refs") or [])
        for item in operations
        if isinstance(item, Mapping)
    }
    for index, binding in enumerate(bindings):
        if not isinstance(binding, dict):
            continue
        premise_id = _text(binding.get("premise_id"))
        valid_binding_operations = [
            operation_id
            for operation_id in (binding.get("operation_ids") or [])
            if operation_id in operation_ids
            and premise_id in operation_outputs.get(operation_id, set())
        ]
        if valid_binding_operations != binding.get("operation_ids"):
            warnings.append(f"bindings[{index}]:dropped_operation_not_outputting_premise")
            binding["operation_ids"] = valid_binding_operations
        if _text(binding.get("coverage")) == "absent" and _text(
            binding.get("relation")
        ) != "counterevidence":
            if binding.get("claim_ids") or binding.get("evidence_ids") or binding.get("operation_ids"):
                warnings.append(f"bindings[{index}]:cleared_support_from_absent_binding")
            binding["claim_ids"] = []
            binding["evidence_ids"] = []
            binding["operation_ids"] = []
        if (
            _text(binding.get("coverage")) == "absent"
            and _text(binding.get("relation")) not in BINDING_RELATIONS
        ):
            warnings.append(f"bindings[{index}]:normalized_absent_relation")
            binding["relation"] = "direct"
        if _text(binding.get("coverage")) == "complete" and not (
            binding.get("claim_ids") or binding.get("operation_ids")
        ):
            warnings.append(f"bindings[{index}]:downgraded_unsupported_complete_to_absent")
            binding["coverage"] = "absent"
            binding["evidence_ids"] = []
        if _text(binding.get("coverage")) == "partial" and not (
            binding.get("claim_ids") or binding.get("operation_ids")
        ):
            warnings.append(f"bindings[{index}]:downgraded_unsupported_partial_to_absent")
            binding["coverage"] = "absent"
            binding["evidence_ids"] = []
        if (
            _text(binding.get("coverage")) == "complete"
            and _text(binding.get("relation")) == "derived"
            and not binding.get("operation_ids")
        ):
            warnings.append(f"bindings[{index}]:downgraded_uncomputed_derived_to_partial")
            binding["coverage"] = "partial"
            binding["relation"] = "direct"
    used_claims = {
        claim_id
        for binding in bindings
        if isinstance(binding, Mapping)
        for claim_id in (binding.get("claim_ids") or [])
        if isinstance(claim_id, str)
    } | {
        claim_id
        for operation in operations
        if isinstance(operation, Mapping)
        for claim_id in (operation.get("input_claim_ids") or [])
        if isinstance(claim_id, str)
    }
    claims = []
    for index, raw in enumerate(claim_inputs):
        if isinstance(raw, Mapping) and _text(raw.get("claim_id")) not in used_claims:
            warnings.append(f"claims[{index}]:dropped_unused_claim")
            continue
        claims.append(dict(raw) if isinstance(raw, Mapping) else raw)
    normalized["claims"] = claims
    normalized["bindings"] = bindings
    normalized["operations"] = operations
    return normalized, warnings


def resolution_payload(
    row: Mapping[str, Any],
    contract: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "question": _text(row.get("question")),
        "question_date": _text(row.get("question_date")) or "unknown",
        "task_contract": validate_task_contract(contract, task_contract_payload(row)),
        "lexical_anchor_ids": list(catalog.get("lexical_anchor_ids") or []),
        "evidence": [
            {
                "evidence_id": item["evidence_id"],
                "session_id": item["session_id"],
                "session_index": item["session_index"],
                "parent_chunk_index": item["parent_chunk_index"],
                "question_overlap_terms": item.get("question_overlap_terms") or [],
                "question_overlap_score": int(item.get("question_overlap_score", 0)),
                "text": item["text"],
            }
            for item in catalog.get("evidence") or []
        ],
        "atoms": [
            {
                "atom_id": item["atom_id"],
                "atom_type": item["atom_type"],
                "raw_text": item["raw_text"],
                "normalized_value": item["normalized_value"],
                "unit": item["unit"],
                "evidence_id": item["evidence_id"],
                **({"derivation": item["derivation"]} if item.get("derivation") else {}),
            }
            for item in catalog.get("atoms") or []
        ],
    }


def validate_resolution_plan(
    value: Mapping[str, Any],
    contract: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    contract = validate_task_contract(contract)
    root = {"schema_version", "claims", "bindings", "operations"}
    if not isinstance(value, Mapping) or set(value) != root:
        raise SemanticEvidenceError("resolution plan root fields are invalid")
    if value.get("schema_version") != RESOLUTION_PLAN_SCHEMA:
        raise SemanticEvidenceError("resolution plan schema_version is invalid")
    evidence_by_id = {
        _text(item.get("evidence_id")): item for item in catalog.get("evidence") or []
    }
    evidence_ids = set(evidence_by_id)
    claims_value = value.get("claims")
    if not isinstance(claims_value, list):
        raise SemanticEvidenceError("resolution claims must be an array")
    claims: list[dict[str, str]] = []
    claim_ids: set[str] = set()
    claim_evidence: dict[str, str] = {}
    claim_fields = {
        "claim_id",
        "subject",
        "predicate",
        "object",
        "valid_time",
        "polarity",
        "modality",
        "evidence_id",
        "source_quote",
    }
    for index, item in enumerate(claims_value):
        if not isinstance(item, Mapping) or set(item) != claim_fields:
            raise SemanticEvidenceError(f"claims[{index}] fields are invalid")
        claim = {key: _text(item.get(key)) for key in claim_fields}
        if (
            not claim["claim_id"]
            or claim["claim_id"] in claim_ids
            or not claim["subject"]
            or not claim["predicate"]
            or claim["evidence_id"] not in evidence_ids
            or not claim["source_quote"]
        ):
            raise SemanticEvidenceError(f"claims[{index}] identity or grounding is invalid")
        if claim["polarity"] not in CLAIM_POLARITIES or claim["modality"] not in CLAIM_MODALITIES:
            raise SemanticEvidenceError(f"claims[{index}] polarity or modality is invalid")
        aligned_quote = _align_quote(
            claim["source_quote"],
            str(evidence_by_id[claim["evidence_id"]].get("text") or ""),
        )
        if not aligned_quote:
            raise SemanticEvidenceError(f"claims[{index}].source_quote is not an exact Source span")
        claim["source_quote"] = aligned_quote
        claim_ids.add(claim["claim_id"])
        claim_evidence[claim["claim_id"]] = claim["evidence_id"]
        claims.append(claim)

    premise_ids = {item["premise_id"] for item in contract["premises"]}
    memory_premise_ids = {
        item["premise_id"]
        for item in contract["premises"]
        if item["source"] == "memory"
    }
    operations_value = value.get("operations")
    if not isinstance(operations_value, list):
        raise SemanticEvidenceError("resolution operations must be an array")
    operation_ids: set[str] = set()
    operation_outputs: dict[str, set[str]] = {}
    operations: list[dict[str, Any]] = []
    executable_operations: list[dict[str, Any]] = []
    operation_fields = {
        "operation_id",
        "operation_type",
        "input_atom_ids",
        "input_evidence_ids",
        "input_claim_ids",
        "parameters",
        "output_refs",
    }
    for index, item in enumerate(operations_value):
        if not isinstance(item, Mapping) or set(item) != operation_fields:
            raise SemanticEvidenceError(f"operations[{index}] fields are invalid")
        operation = dict(item)
        output_refs = _string_list(
            operation.pop("output_refs"),
            path=f"operations[{index}].output_refs",
            allowed=memory_premise_ids | {"TARGET"},
        )
        operation_id = _text(operation.get("operation_id"))
        operation_allowed = _text(operation.get("operation_type")) in contract[
            "allowed_derivations"
        ] or (
            "semantic_composition" in contract["allowed_derivations"]
            and _text(operation.get("operation_type")) in OPERATION_TYPES
        )
        if (
            not operation_id
            or operation_id in operation_ids
            or not output_refs
            or not operation_allowed
        ):
            raise SemanticEvidenceError(f"operations[{index}] identity or output premises are invalid")
        operation_ids.add(operation_id)
        operation_outputs[operation_id] = set(output_refs)
        input_claim_ids = _string_list(
            operation.pop("input_claim_ids"),
            path=f"operations[{index}].input_claim_ids",
            allowed=claim_ids,
        )
        if input_claim_ids:
            if _text(operation.get("operation_type")) not in CLAIM_OPERATION_TYPES:
                raise SemanticEvidenceError(f"operations[{index}] does not support claim operands")
            if operation.get("input_atom_ids"):
                raise SemanticEvidenceError(f"operations[{index}] mixes claim and atom operands")
            claim_support = {claim_evidence[claim_id] for claim_id in input_claim_ids}
            if not claim_support.issubset(set(operation.get("input_evidence_ids") or [])):
                raise SemanticEvidenceError(f"operations[{index}] omits evidence for claim operands")
        else:
            executable_operations.append(operation)
        operations.append(
            {
                **operation,
                "input_claim_ids": input_claim_ids,
                "output_refs": output_refs,
            }
        )
    operation_plan = {
        "schema_version": "tmcra.evidence-operation-plan.v1",
        "requirements": [
            {
                "requirement_id": premise["premise_id"],
                "description": premise["description"],
                "evidence_ids": [],
            }
            for premise in contract["premises"]
            if premise["source"] == "memory"
        ],
        "operations": executable_operations,
        "bundles": [],
    }

    bindings_value = value.get("bindings")
    if not isinstance(bindings_value, list):
        raise SemanticEvidenceError("resolution bindings must be an array")
    bindings: list[dict[str, Any]] = []
    bound_premises: set[str] = set()
    evidence_by_premise: dict[str, list[str]] = {}
    for index, item in enumerate(bindings_value):
        fields = {
            "premise_id",
            "claim_ids",
            "evidence_ids",
            "operation_ids",
            "coverage",
            "relation",
        }
        if not isinstance(item, Mapping) or set(item) != fields:
            raise SemanticEvidenceError(f"bindings[{index}] fields are invalid")
        premise_id = _text(item.get("premise_id"))
        if premise_id not in memory_premise_ids or premise_id in bound_premises:
            raise SemanticEvidenceError(f"bindings[{index}].premise_id is invalid")
        current_claims = _string_list(item.get("claim_ids"), path=f"bindings[{index}].claim_ids", allowed=claim_ids)
        current_evidence = _string_list(item.get("evidence_ids"), path=f"bindings[{index}].evidence_ids", allowed=evidence_ids)
        current_operations = _string_list(item.get("operation_ids"), path=f"bindings[{index}].operation_ids", allowed=operation_ids)
        coverage, relation = _text(item.get("coverage")), _text(item.get("relation"))
        if coverage not in BINDING_COVERAGE or relation not in BINDING_RELATIONS:
            raise SemanticEvidenceError(f"bindings[{index}] coverage or relation is invalid")
        claimed_evidence = {claim_evidence[claim_id] for claim_id in current_claims}
        if not claimed_evidence.issubset(set(current_evidence)):
            raise SemanticEvidenceError(f"bindings[{index}] omits evidence used by its claims")
        if coverage == "absent" and (current_claims or current_operations) and relation != "counterevidence":
            raise SemanticEvidenceError(f"bindings[{index}] absent coverage has semantic support")
        if coverage == "complete" and not (current_claims or current_operations):
            raise SemanticEvidenceError(f"bindings[{index}] complete coverage lacks claims or operations")
        if relation == "derived" and coverage == "complete" and not current_operations:
            raise SemanticEvidenceError(f"bindings[{index}] complete derived coverage lacks an operation")
        if any(premise_id not in operation_outputs[operation_id] for operation_id in current_operations):
            raise SemanticEvidenceError(f"bindings[{index}] cites an operation that does not output this premise")
        bound_premises.add(premise_id)
        evidence_by_premise[premise_id] = current_evidence
        bindings.append(
            {
                "premise_id": premise_id,
                "claim_ids": current_claims,
                "evidence_ids": current_evidence,
                "operation_ids": current_operations,
                "coverage": coverage,
                "relation": relation,
            }
        )
    if bound_premises != memory_premise_ids:
        raise SemanticEvidenceError("resolution plan must bind every memory premise exactly once")
    operations_by_premise = {
        premise_id: {
            operation_id
            for operation_id, outputs in operation_outputs.items()
            if premise_id in outputs
        }
        for premise_id in memory_premise_ids
    }
    binding_by_premise = {item["premise_id"]: item for item in bindings}
    if any(
        not operations_by_premise[premise_id].issubset(
            set(binding_by_premise[premise_id]["operation_ids"])
        )
        for premise_id in memory_premise_ids
    ):
        raise SemanticEvidenceError("resolution operation outputs are missing from premise bindings")
    required_memory_ids = {
        item["premise_id"]
        for item in contract["premises"]
        if item["source"] == "memory" and item["necessity"] == "required"
    }
    memory_inputs_complete = all(
        binding_by_premise[premise_id]["coverage"] == "complete"
        for premise_id in required_memory_ids
    )
    if (
        contract["output"]["origin"] == "memory_derived"
        and memory_inputs_complete
        and not any("TARGET" in outputs for outputs in operation_outputs.values())
        and "semantic_composition" not in contract["allowed_derivations"]
    ):
        raise SemanticEvidenceError("memory-derived task lacks an operation that outputs TARGET")
    for requirement in operation_plan["requirements"]:
        requirement["evidence_ids"] = evidence_by_premise[requirement["requirement_id"]]
    operation_plan["bundles"] = [
        {
            "bundle_id": f"B{index:03d}",
            "role": binding["relation"],
            "evidence_ids": binding["evidence_ids"],
        }
        for index, binding in enumerate(bindings, start=1)
        if binding["evidence_ids"]
    ]
    try:
        validate_operation_plan(operation_plan, catalog)
    except EvidenceOperationError as exc:
        raise SemanticEvidenceError(f"resolution deterministic operation is invalid: {exc}") from exc
    return {
        "schema_version": RESOLUTION_PLAN_SCHEMA,
        "claims": claims,
        "bindings": bindings,
        "operations": operations,
    }


def _operation_plan_from_resolution(
    contract: Mapping[str, Any], resolution: Mapping[str, Any]
) -> dict[str, Any]:
    binding_by_premise = {item["premise_id"]: item for item in resolution["bindings"]}
    return {
        "schema_version": "tmcra.evidence-operation-plan.v1",
        "requirements": [
            {
                "requirement_id": premise["premise_id"],
                "description": premise["description"],
                "evidence_ids": binding_by_premise[premise["premise_id"]]["evidence_ids"],
            }
            for premise in contract["premises"]
            if premise["source"] == "memory"
        ],
        "operations": [
            {
                key: value
                for key, value in operation.items()
                if key not in {"output_refs", "input_claim_ids"}
            }
            for operation in resolution["operations"]
            if not operation["input_claim_ids"]
        ],
        "bundles": [
            {
                "bundle_id": f"B{index:03d}",
                "role": binding["relation"],
                "evidence_ids": binding["evidence_ids"],
            }
            for index, binding in enumerate(resolution["bindings"], start=1)
            if binding["evidence_ids"]
        ],
    }


def _execute_claim_operation(
    operation: Mapping[str, Any], claims_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    operation_type = operation["operation_type"]
    claims = [claims_by_id[item] for item in operation["input_claim_ids"]]
    parameters = dict(operation.get("parameters") or {})
    field = _text(parameters.get("field"))
    if not field and operation_type in {"date_difference", "date_order"}:
        field = (
            "valid_time"
            if all(_text(item.get("valid_time")) for item in claims)
            else "object"
        )
    field = field or "subject"
    if field not in {"subject", "predicate", "object", "valid_time"}:
        raise SemanticEvidenceError(f"claim operation field is invalid: {field}")
    values = [_text(item.get(field)) for item in claims]
    if any(not value for value in values):
        raise SemanticEvidenceError(f"claim operation has an empty {field} value")
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(value)
    if operation_type == "date_order":
        def parsed_date(value: str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
            except ValueError as exc:
                match = re.search(r"\b20\d{2}-\d{2}-\d{2}\b", value)
                if not match:
                    raise SemanticEvidenceError(f"date_order has an invalid date: {value}") from exc
                return datetime.strptime(match.group(0), "%Y-%m-%d").date()

        ordered = sorted(zip(claims, values), key=lambda item: parsed_date(item[1]))
        result = {
            "ordered_claim_ids": [item[0]["claim_id"] for item in ordered],
            "ordered_values": [item[1] for item in ordered],
        }
    elif operation_type == "date_difference":
        if len(values) != 2:
            raise SemanticEvidenceError("date_difference requires exactly two claim operands")

        def parsed_date(value: str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
            except ValueError as exc:
                match = re.search(r"\b20\d{2}-\d{2}-\d{2}\b", value)
                if not match:
                    raise SemanticEvidenceError(f"date_difference has an invalid date: {value}") from exc
                return datetime.strptime(match.group(0), "%Y-%m-%d").date()

        result = {"value": abs((parsed_date(values[1]) - parsed_date(values[0])).days), "unit": "days"}
    elif operation_type in {"numeric_sum", "numeric_difference", "numeric_average"}:
        numbers: list[Decimal] = []
        for value in values:
            match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", value)
            if not match:
                raise SemanticEvidenceError(f"{operation_type} has a non-numeric claim value: {value}")
            try:
                numbers.append(Decimal(match.group(0).replace(",", "")))
            except InvalidOperation as exc:
                raise SemanticEvidenceError(f"{operation_type} has an invalid number: {value}") from exc
        if operation_type == "numeric_sum":
            number = sum(numbers, Decimal(0))
        elif operation_type == "numeric_difference":
            if len(numbers) != 2:
                raise SemanticEvidenceError("numeric_difference requires exactly two claim operands")
            number = numbers[0] - numbers[1]
        else:
            number = sum(numbers, Decimal(0)) / Decimal(len(numbers))
        result = {
            "value": int(number) if number == number.to_integral_value() else float(number),
            "unit": _text(parameters.get("unit")),
        }
    elif operation_type == "count_distinct":
        result: dict[str, Any] = {"value": len(unique), "values": unique}
    elif operation_type == "ordered_unique_list":
        result = {"values": unique}
    elif operation_type == "latest_state":
        latest = max(claims, key=lambda item: _text(item.get("valid_time")))
        if not _text(latest.get("valid_time")):
            raise SemanticEvidenceError("latest_state claim operation lacks valid_time")
        result = {
            "claim_id": latest["claim_id"],
            "subject": latest["subject"],
            "predicate": latest["predicate"],
            "object": latest["object"],
            "valid_time": latest["valid_time"],
        }
    elif operation_type in {"entity_exact_match", "entity_mismatch"}:
        if len(values) != 2:
            raise SemanticEvidenceError(f"{operation_type} requires exactly two claim operands")
        equal = values[0].casefold() == values[1].casefold()
        result = {"value": equal if operation_type == "entity_exact_match" else not equal}
    else:
        raise SemanticEvidenceError(f"unsupported claim operation: {operation_type}")
    return {
        "operation_id": operation["operation_id"],
        "operation_type": operation_type,
        "status": "completed",
        "input_claim_ids": list(operation["input_claim_ids"]),
        "support_ids": list(operation.get("input_evidence_ids") or []),
        "result": result,
    }


def build_answerability_certificate(
    contract: Mapping[str, Any],
    resolution: Mapping[str, Any],
    operation_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    operation_status = {_text(item.get("operation_id")): _text(item.get("status")) for item in operation_results}
    binding_by_premise = {item["premise_id"]: item for item in resolution["bindings"]}
    statuses: list[dict[str, Any]] = []
    for premise in contract["premises"]:
        source = premise["source"]
        if source == "memory":
            binding = binding_by_premise[premise["premise_id"]]
            effective = binding["coverage"]
            relation = binding["relation"]
        elif source == "query_context":
            effective, relation = "complete", "context"
        elif source == "model_knowledge":
            effective, relation = "available", "context"
        else:
            effective, relation = "pending", "context"
        statuses.append(
            {
                "premise_id": premise["premise_id"],
                "necessity": premise["necessity"],
                "source": source,
                "coverage": effective,
                "relation": relation,
            }
        )
    required_memory = [
        item
        for item in statuses
        if item["necessity"] == "required" and item["source"] == "memory"
    ]
    required_coverages = [item["coverage"] for item in required_memory]
    if any(value == "conflicting" for value in required_coverages):
        memory_coverage = "conflicting"
    elif required_coverages and all(value == "absent" for value in required_coverages):
        memory_coverage = "absent"
    elif any(value in {"partial", "absent"} for value in required_coverages):
        memory_coverage = "partial"
    else:
        memory_coverage = "complete"
    output_origin = contract["output"]["origin"]
    target_operation_ids = [
        operation["operation_id"]
        for operation in resolution["operations"]
        if "TARGET" in operation["output_refs"]
    ]
    target_operations_completed = all(
        operation_status.get(operation_id) == "completed"
        for operation_id in target_operation_ids
    )
    failed_operation_ids = [
        operation_id
        for operation_id in target_operation_ids
        if operation_status.get(operation_id) != "completed"
    ]
    if not target_operation_ids:
        target_operation_status = "not_needed"
    elif failed_operation_ids:
        target_operation_status = "failed"
    else:
        target_operation_status = "completed"
    pending_external = [
        item["premise_id"]
        for item in statuses
        if item["necessity"] == "required" and item["source"] == "external_tool"
    ]
    if memory_coverage == "conflicting":
        executability, policy = "requires_clarification", "clarify"
    elif memory_coverage == "absent":
        executability, policy = "not_answerable", "abstain"
    elif memory_coverage == "partial" and output_origin in {
        "memory_direct",
        "memory_conditioned",
    }:
        executability, policy = "partially_answerable", "answer_partial"
    elif memory_coverage == "partial":
        executability, policy = "not_answerable", "abstain"
    elif pending_external:
        executability, policy = "requires_external_tool", "call_tool"
    elif output_origin == "memory_direct":
        executability, policy = "directly_answerable", "answer"
    elif output_origin == "memory_derived" and target_operations_completed:
        executability, policy = "derivable", "answer"
    elif output_origin == "memory_derived" and "semantic_composition" in contract["allowed_derivations"]:
        executability, policy = "derivable", "answer"
    elif output_origin == "memory_derived":
        executability, policy = "not_answerable", "abstain"
    elif output_origin == "memory_conditioned":
        executability, policy = "memory_conditioned_generation", "answer_with_memory_context"
    else:
        executability, policy = "requires_external_tool", "call_tool"
    unresolved = [
        item["premise_id"]
        for item in required_memory
        if item["coverage"] in {"partial", "absent", "conflicting"}
    ] + pending_external
    allowed_origins = ["memory_fact", "memory_derived"]
    if policy == "answer_with_memory_context" or any(
        item["source"] == "model_knowledge" for item in statuses
    ):
        allowed_origins.append("model_knowledge")
    if policy == "call_tool":
        allowed_origins.append("tool_result")
    return {
        "memory_coverage": memory_coverage,
        "operation_status": target_operation_status,
        "failed_operation_ids": failed_operation_ids,
        "unresolved_reasons": [
            f"operator_failure:{operation_id}" for operation_id in failed_operation_ids
        ],
        "task_executability": executability,
        "answer_policy": policy,
        "output_origin": output_origin,
        "target_operation_ids": target_operation_ids,
        "premise_status": statuses,
        "unresolved_premise_ids": unresolved,
        "allowed_claim_origins": allowed_origins,
    }


def compile_semantic_evidence_packet(
    row: Mapping[str, Any],
    contract: Mapping[str, Any],
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    query_context = task_contract_payload(row)
    contract = validate_task_contract(contract, query_context)
    catalog = build_evidence_catalog(row)
    resolution = validate_resolution_plan(resolution, contract, catalog)
    operation_plan = _operation_plan_from_resolution(contract, resolution)
    standard_results = {
        item["operation_id"]: item for item in execute_operation_plan(operation_plan, catalog)
    }
    claims_by_id = {item["claim_id"]: item for item in resolution["claims"]}
    operation_results: list[dict[str, Any]] = []
    for operation in resolution["operations"]:
        if operation["input_claim_ids"]:
            try:
                operation_results.append(_execute_claim_operation(operation, claims_by_id))
            except SemanticEvidenceError as exc:
                operation_results.append(
                    {
                        "operation_id": operation["operation_id"],
                        "operation_type": operation["operation_type"],
                        "status": "error",
                        "input_claim_ids": list(operation["input_claim_ids"]),
                        "support_ids": list(operation.get("input_evidence_ids") or []),
                        "error": str(exc),
                    }
                )
        else:
            operation_results.append(standard_results[operation["operation_id"]])
    certificate = build_answerability_certificate(contract, resolution, operation_results)
    evidence_by_id = {item["evidence_id"]: item for item in catalog["evidence"]}
    return {
        "schema_version": SEMANTIC_PACKET_SCHEMA,
        "question_id": catalog["question_id"],
        "query_context": query_context,
        "task_contract": contract,
        "claim_ledger": resolution["claims"],
        "premise_bindings": resolution["bindings"],
        "resolution_program": {
            "schema_version": RESOLUTION_PLAN_SCHEMA,
            "operations": resolution["operations"],
            "operation_results": operation_results,
        },
        "answerability_certificate": certificate,
        "bound_evidence": [
            {
                **binding,
                "claims": [
                    claim
                    for claim in resolution["claims"]
                    if claim["claim_id"] in binding["claim_ids"]
                ],
                "evidence": [evidence_by_id[item] for item in binding["evidence_ids"]],
            }
            for binding in resolution["bindings"]
        ],
        "raw_evidence_reservoir": catalog["evidence"],
        "lexical_anchor_ids": list(catalog.get("lexical_anchor_ids") or []),
        "atom_catalog": catalog,
    }


def semantic_packet_to_advisory_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Expose semantic planning as annotations over the complete Source reservoir."""
    if packet.get("schema_version") != SEMANTIC_PACKET_SCHEMA:
        raise SemanticEvidenceError("semantic evidence packet schema is invalid")
    contract = packet.get("task_contract")
    certificate = packet.get("answerability_certificate")
    reservoir = packet.get("raw_evidence_reservoir")
    if (
        not isinstance(contract, Mapping)
        or not isinstance(certificate, Mapping)
        or not isinstance(reservoir, list)
        or not reservoir
    ):
        raise SemanticEvidenceError("semantic packet lacks contract, certificate, or Source reservoir")
    evidence_ids = {
        _text(item.get("evidence_id"))
        for item in reservoir
        if isinstance(item, Mapping) and _text(item.get("evidence_id"))
    }
    bindings = {
        _text(item.get("premise_id")): item
        for item in packet.get("premise_bindings") or []
        if isinstance(item, Mapping) and _text(item.get("premise_id"))
    }
    requirements: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    for premise in contract.get("premises") or []:
        if not isinstance(premise, Mapping):
            continue
        premise_id = _text(premise.get("premise_id"))
        if not premise_id:
            continue
        binding = bindings.get(premise_id, {})
        bound_ids = [
            item
            for item in binding.get("evidence_ids") or []
            if isinstance(item, str) and item in evidence_ids
        ]
        coverage = _text(binding.get("coverage")) or (
            "available" if premise.get("source") != "memory" else "unresolved"
        )
        requirements.append(
            {
                "requirement_id": premise_id,
                "description": _text(premise.get("description")) or premise_id,
                "evidence_ids": bound_ids,
                "state": coverage,
                "planner_advisory": True,
            }
        )
        if bound_ids:
            bundles.append(
                {
                    "bundle_id": f"B_{premise_id}",
                    "role": _text(premise.get("role")) or "advisory",
                    "evidence_ids": bound_ids,
                    "evidence": [
                        item
                        for item in reservoir
                        if isinstance(item, Mapping) and item.get("evidence_id") in bound_ids
                    ],
                }
            )
    query_context = dict(packet.get("query_context") or {})
    typed_input = packet.get("typed_semantics")
    typed_evaluation: dict[str, Any] = {
        "status": "not_provided",
        "advisory": True,
        "authoritative": False,
        "accepted": [],
        "rejected": [],
    }
    if typed_input is not None:
        if not isinstance(typed_input, Mapping):
            typed_evaluation = {
                "status": "invalid_container",
                "advisory": True,
                "authoritative": False,
                "accepted": [],
                "rejected": [
                    {
                        "candidate_id": "typed_semantics",
                        "status": "rejected",
                        "authoritative": False,
                        "source_evidence_ids": [],
                        "evidence_ids": [],
                        "diagnostics": [
                            {
                                "code": "UNTYPED_CONTAINER",
                                "message": "typed_semantics must contain observations and proposals",
                                "path": "typed_semantics",
                            }
                        ],
                    }
                ],
            }
        else:
            typed_evaluation = evaluate_proposals(
                typed_input.get("observations", []),
                typed_input.get("proposals", []),
            )

    accepted_typed_results: list[dict[str, Any]] = []
    rejected_typed_results = list(typed_evaluation.get("rejected") or [])
    for candidate in typed_evaluation.get("accepted") or []:
        support_ids = [
            item
            for item in candidate.get("source_evidence_ids") or []
            if isinstance(item, str)
        ]
        unknown_ids = sorted(set(support_ids) - evidence_ids)
        if unknown_ids:
            rejected_typed_results.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "status": "rejected",
                    "authoritative": False,
                    "source_evidence_ids": support_ids,
                    "evidence_ids": support_ids,
                    "diagnostics": [
                        {
                            "code": "UNKNOWN_SOURCE_EVIDENCE",
                            "message": "typed result cites evidence outside the current Source reservoir",
                            "path": "typed_semantics.observations.evidence_ids",
                            "unknown_evidence_ids": unknown_ids,
                        }
                    ],
                }
            )
            continue
        candidate_id = _text(candidate.get("candidate_id"))
        accepted_typed_results.append(
            {
                "operation_id": f"TS_{candidate_id}",
                "operation_type": "typed_semantic_proposal",
                "status": "completed",
                "input_atom_ids": [],
                "support_ids": support_ids,
                "result": {
                    "value": candidate.get("value"),
                    "value_kind": candidate.get("value_kind"),
                    "unit": candidate.get("unit"),
                },
                "advisory": True,
                "authoritative": False,
            }
        )
    typed_diagnostics = {
        **typed_evaluation,
        "accepted": list(typed_evaluation.get("accepted") or []),
        "rejected": rejected_typed_results,
    }
    return {
        "schema_version": PACKET_SCHEMA,
        "question_id": packet.get("question_id"),
        "question_contract": {
            "question": query_context.get("question"),
            "question_date": query_context.get("question_date"),
            "requirement_count": len(requirements),
            "operation_count": len(accepted_typed_results),
            "semantic_authority": "advisory",
        },
        "requirement_coverage": requirements,
        # Legacy semantic operations are intentionally withheld until a typed
        # proposal validator proves their dimensions and entity constraints.
        "operation_results": accepted_typed_results,
        "evidence_bundles": bundles,
        "raw_evidence_reservoir": list(reservoir),
        "lexical_anchor_ids": list(packet.get("lexical_anchor_ids") or []),
        "operation_result_ids": [item["operation_id"] for item in accepted_typed_results],
        "semantic_advisory": {
            "authority": "advisory_not_answerability",
            "task_contract": dict(contract),
            "claim_ledger": list(packet.get("claim_ledger") or []),
            "planner_certificate": dict(certificate),
            "legacy_operation_status": "withheld_untyped",
            "typed_semantics": typed_diagnostics,
        },
    }


def normalize_semantic_answer_output(
    value: Mapping[str, Any], packet: Mapping[str, Any]
) -> dict[str, Any]:
    """Repair only mechanically verifiable answer metadata and support references."""
    if not isinstance(value, Mapping):
        return dict(value)
    normalized = dict(value)
    certificate = packet.get("answerability_certificate")
    if not isinstance(certificate, Mapping):
        return normalized

    expected_status = {
        "answer": "answered",
        "answer_partial": "partial",
        "answer_with_memory_context": "answered_with_context",
        "call_tool": "requires_tool",
        "clarify": "clarification",
        "abstain": "not_answerable",
    }.get(_text(certificate.get("answer_policy")))
    current_status = _text(normalized.get("response_status"))
    if {current_status, expected_status}.issubset({"answered", "answered_with_context"}):
        normalized["response_status"] = expected_status

    premise_bindings = {
        _text(item.get("premise_id")): item
        for item in packet.get("premise_bindings") or []
        if isinstance(item, Mapping) and _text(item.get("premise_id"))
    }
    task_premise_ids = {
        _text(item.get("premise_id"))
        for item in (packet.get("task_contract") or {}).get("premises") or []
        if isinstance(item, Mapping) and _text(item.get("premise_id"))
    }
    ledger_ids = {
        _text(item.get("claim_id"))
        for item in packet.get("claim_ledger") or []
        if isinstance(item, Mapping) and _text(item.get("claim_id"))
    }
    completed_operation_ids = {
        _text(item.get("operation_id"))
        for item in (packet.get("resolution_program") or {}).get("operation_results") or []
        if isinstance(item, Mapping)
        and item.get("status") == "completed"
        and _text(item.get("operation_id"))
    }
    operation_specs = {
        _text(item.get("operation_id")): item
        for item in (packet.get("resolution_program") or {}).get("operations") or []
        if isinstance(item, Mapping) and _text(item.get("operation_id"))
    }
    target_operation_ids = set(certificate.get("target_operation_ids") or [])
    allowed_origins = set(certificate.get("allowed_claim_origins") or [])

    claims = normalized.get("claims")
    if isinstance(claims, list):
        normalized_claims: list[Any] = []
        for raw_claim in claims:
            if not isinstance(raw_claim, Mapping):
                normalized_claims.append(raw_claim)
                continue
            claim = dict(raw_claim)
            origin = _text(claim.get("origin"))
            raw_premises = claim.get("premise_ids")
            current_premises = (
                [item for item in raw_premises if isinstance(item, str) and item in task_premise_ids]
                if isinstance(raw_premises, list)
                else raw_premises
            )
            if origin.startswith("memory_") and isinstance(current_premises, list):
                current_premises = [item for item in current_premises if item in premise_bindings]
            claim["premise_ids"] = current_premises

            if origin.startswith("memory_") and isinstance(current_premises, list):
                allowed_source_claims = {
                    item
                    for premise_id in current_premises
                    for item in premise_bindings[premise_id].get("claim_ids") or []
                }
                premise_source_claims = set(allowed_source_claims)
                premise_evidence = {
                    item
                    for premise_id in current_premises
                    for item in premise_bindings[premise_id].get("evidence_ids") or []
                }
                allowed_computations = {
                    item
                    for premise_id in current_premises
                    for item in premise_bindings[premise_id].get("operation_ids") or []
                }
                allowed_computations.update(
                    operation_id
                    for operation_id in target_operation_ids
                    if operation_id in operation_specs
                    and set(operation_specs[operation_id].get("input_claim_ids") or []).issubset(
                        premise_source_claims
                    )
                    and set(operation_specs[operation_id].get("input_evidence_ids") or []).issubset(
                        premise_evidence
                    )
                )
                raw_sources = claim.get("source_claim_ids")
                if isinstance(raw_sources, list):
                    claim["source_claim_ids"] = [
                        item
                        for item in raw_sources
                        if isinstance(item, str)
                        and item in ledger_ids
                        and item in allowed_source_claims
                    ]
                raw_computations = claim.get("computation_ids")
                if isinstance(raw_computations, list):
                    claim["computation_ids"] = [
                        item
                        for item in raw_computations
                        if isinstance(item, str)
                        and item in completed_operation_ids
                        and item in allowed_computations
                    ]
                if origin == "memory_fact":
                    claim["computation_ids"] = []
                elif (
                    origin == "memory_derived"
                    and not claim.get("computation_ids")
                    and claim.get("source_claim_ids")
                    and "memory_fact" in allowed_origins
                ):
                    claim["origin"] = "memory_fact"
            elif origin in {"model_knowledge", "tool_result"}:
                claim["source_claim_ids"] = []
                claim["computation_ids"] = []
            normalized_claims.append(claim)
        normalized["claims"] = normalized_claims

    unresolved = normalized.get("unresolved_premise_ids")
    expected_unresolved = list(certificate.get("unresolved_premise_ids") or [])
    if isinstance(unresolved, list) and set(unresolved) == set(expected_unresolved):
        normalized["unresolved_premise_ids"] = expected_unresolved
    return normalized


def validate_semantic_answer(value: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, Any]:
    root = {"schema_version", "response_status", "claims", "unresolved_premise_ids", "answer"}
    if not isinstance(value, Mapping) or not root.issubset(value):
        raise SemanticEvidenceError("semantic answer root fields are invalid")
    if value.get("schema_version") != SEMANTIC_ANSWER_SCHEMA:
        raise SemanticEvidenceError("semantic answer schema_version is invalid")
    status = _text(value.get("response_status"))
    if status not in ANSWER_STATUSES:
        raise SemanticEvidenceError("semantic answer response_status is invalid")
    certificate = packet.get("answerability_certificate")
    if not isinstance(certificate, Mapping):
        raise SemanticEvidenceError("semantic packet lacks an answerability certificate")
    expected_status = {
        "answer": "answered",
        "answer_partial": "partial",
        "answer_with_memory_context": "answered_with_context",
        "call_tool": "requires_tool",
        "clarify": "clarification",
        "abstain": "not_answerable",
    }.get(_text(certificate.get("answer_policy")))
    if status != expected_status:
        raise SemanticEvidenceError("semantic answer contradicts the answerability certificate")
    premise_bindings = {item["premise_id"]: item for item in packet.get("premise_bindings") or []}
    premise_ids = {
        item["premise_id"] for item in (packet.get("task_contract") or {}).get("premises") or []
    }
    ledger = {item["claim_id"]: item for item in packet.get("claim_ledger") or []}
    operation_results = {
        item["operation_id"]: item
        for item in (packet.get("resolution_program") or {}).get("operation_results") or []
        if item.get("status") == "completed"
    }
    operation_specs = {
        item["operation_id"]: item
        for item in (packet.get("resolution_program") or {}).get("operations") or []
    }
    target_operation_ids = set(certificate.get("target_operation_ids") or [])
    allowed_origins = set(certificate.get("allowed_claim_origins") or [])
    claims_value = value.get("claims")
    if not isinstance(claims_value, list):
        raise SemanticEvidenceError("semantic answer claims must be an array")
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(claims_value):
        required_fields = {"claim_id", "text", "origin", "premise_ids"}
        if not isinstance(item, Mapping) or not required_fields.issubset(item):
            raise SemanticEvidenceError(f"answer claims[{index}] fields are invalid")
        claim_id, text, origin = _text(item.get("claim_id")), _text(item.get("text")), _text(item.get("origin"))
        if not claim_id or claim_id in seen or not text or origin not in CLAIM_ORIGINS or origin not in allowed_origins:
            raise SemanticEvidenceError(f"answer claims[{index}] identity or origin is invalid")
        current_premises = _string_list(item.get("premise_ids"), path=f"answer claims[{index}].premise_ids", allowed=premise_ids)
        source_claims = _string_list(item.get("source_claim_ids", []), path=f"answer claims[{index}].source_claim_ids", allowed=set(ledger))
        computations = _string_list(item.get("computation_ids", []), path=f"answer claims[{index}].computation_ids", allowed=set(operation_results))
        if origin == "memory_fact" and not source_claims:
            raise SemanticEvidenceError(f"answer claims[{index}] memory_fact lacks Source claims")
        if origin == "memory_derived" and not computations:
            raise SemanticEvidenceError(f"answer claims[{index}] memory_derived lacks computations")
        if origin in {"model_knowledge", "tool_result"} and (source_claims or computations):
            raise SemanticEvidenceError(f"answer claims[{index}] non-memory origin has memory bindings")
        if origin.startswith("memory_") and not current_premises:
            raise SemanticEvidenceError(f"answer claims[{index}] memory claim lacks premise bindings")
        if origin.startswith("memory_") and any(
            premise_id not in premise_bindings for premise_id in current_premises
        ):
            raise SemanticEvidenceError(f"answer claims[{index}] memory claim cites a non-memory premise")
        allowed_source_claims = {
            claim
            for premise_id in current_premises
            if premise_id in premise_bindings
            for claim in premise_bindings[premise_id]["claim_ids"]
        }
        allowed_computations = {
            operation
            for premise_id in current_premises
            if premise_id in premise_bindings
            for operation in premise_bindings[premise_id]["operation_ids"]
        }
        premise_source_claims = {
            source_claim
            for premise_id in current_premises
            if premise_id in premise_bindings
            for source_claim in premise_bindings[premise_id]["claim_ids"]
        }
        premise_evidence = {
            evidence_id
            for premise_id in current_premises
            if premise_id in premise_bindings
            for evidence_id in premise_bindings[premise_id]["evidence_ids"]
        }
        allowed_computations.update(
            operation_id
            for operation_id in target_operation_ids
            if operation_id in operation_specs
            and set(operation_specs[operation_id].get("input_claim_ids") or []).issubset(
                premise_source_claims
            )
            and set(operation_specs[operation_id].get("input_evidence_ids") or []).issubset(
                premise_evidence
            )
        )
        if not set(source_claims).issubset(allowed_source_claims) or not set(computations).issubset(allowed_computations):
            raise SemanticEvidenceError(f"answer claims[{index}] cites support outside its premises")
        support_ids = sorted(
            {ledger[item]["evidence_id"] for item in source_claims}
            | {
                support_id
                for operation_id in computations
                for support_id in operation_results[operation_id].get("support_ids") or []
            }
        )
        seen.add(claim_id)
        claims.append(
            {
                "claim_id": claim_id,
                "text": text,
                "origin": origin,
                "premise_ids": current_premises,
                "source_claim_ids": source_claims,
                "support_ids": support_ids,
                "computation_ids": computations,
            }
        )
    unresolved = _string_list(
        value.get("unresolved_premise_ids"),
        path="unresolved_premise_ids",
        allowed=premise_ids,
    )
    expected_unresolved = list(certificate.get("unresolved_premise_ids") or [])
    if unresolved != expected_unresolved:
        raise SemanticEvidenceError("semantic answer unresolved premises do not match the certificate")
    if status in {"answered", "answered_with_context"} and (unresolved or not claims):
        raise SemanticEvidenceError("answering status requires claims and no unresolved premises")
    if status == "partial" and (not unresolved or not claims):
        raise SemanticEvidenceError("partial status requires supported claims and unresolved premises")
    if (
        status in {"answered_with_context", "partial"}
        and certificate.get("output_origin") == "memory_conditioned"
        and not any(
        claim["origin"] in {"memory_fact", "memory_derived"} for claim in claims
        )
    ):
        raise SemanticEvidenceError("memory-conditioned answer does not use any memory claim")
    answer = _text(value.get("answer"))
    if not answer:
        raise SemanticEvidenceError("semantic answer text is empty")
    return {
        "schema_version": SEMANTIC_ANSWER_SCHEMA,
        "response_status": status,
        "claims": claims,
        "unresolved_premise_ids": unresolved,
        "answer": answer,
    }
