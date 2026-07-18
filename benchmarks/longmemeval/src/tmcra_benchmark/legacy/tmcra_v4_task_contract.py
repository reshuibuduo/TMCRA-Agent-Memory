"""Strict, benchmark-independent task contracts for TMCRA V4.

The module is deliberately self-contained.  A contract describes what a task
needs, where its premises may come from, and how its result is shaped.  It does
not contain a benchmark route, question type, answer, or model/network code.

Canonical contract shape::

    {
        "schema_version": "tmcra.task-contract.v4",
        "output_origin": "memory_direct|memory_derived|...",
        "target": {
            "subject": "...",
            "relation": "...",
            "entity_constraints": ["..."],
            "temporal_constraints": ["..."],
            "state_constraints": ["..."]
        },
        "output": {
            "shape": "scalar|list|set|count|boolean|date|duration|structured|free_text",
            "cardinality": "one|zero_or_one|one_or_more|zero_or_more",
            "order": "none|input_order|chronological|reverse_chronological|recency|ranked|question_order"
        },
        "premises": [
            {
                "premise_id": "P01",
                "description": "...",
                "role": "fact|operand|constraint|scope|counterevidence|inventory|state",
                "necessity": "required|optional",
                "source": "memory|query_context|model_knowledge|external_tool",
                "grounded_constraints": ["..."],
                "context_quote": ""
            }
        ],
        "operations": [
            {
                "operation_id": "O01",
                "operation_type": "date_difference",
                "input_premise_ids": ["P01"],
                "output_ref": "TARGET",
                "parameters": {}
            }
        ]
    }

``operations`` is optional in the input and normalizes to an empty list.  All
other fields shown above are required.  ``risk_signals`` are intentionally
computed by :func:`structural_risk_signals`, rather than trusted from model
output.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


TASK_CONTRACT_SCHEMA = "tmcra.task-contract.v4"
SCHEMA_VERSION = TASK_CONTRACT_SCHEMA

OUTPUT_ORIGINS = frozenset(
    {
        "memory_direct",
        "memory_derived",
        "memory_conditioned_generation",
        "external_required",
    }
)
OUTPUT_SHAPES = frozenset(
    {
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
)
OUTPUT_CARDINALITIES = frozenset(
    {"one", "zero_or_one", "one_or_more", "zero_or_more"}
)
OUTPUT_ORDERS = frozenset(
    {
        "none",
        "input_order",
        "chronological",
        "reverse_chronological",
        "recency",
        "ranked",
        "question_order",
    }
)
PREMISE_SOURCES = frozenset(
    {"memory", "query_context", "model_knowledge", "external_tool"}
)
PREMISE_ROLES = frozenset(
    {"fact", "operand", "constraint", "scope", "counterevidence", "inventory", "state"}
)
PREMISE_NECESSITY = frozenset({"required", "optional"})
OPERATION_TYPES = frozenset(
    {
        "aggregate",
        "count",
        "sum",
        "average",
        "min",
        "max",
        "difference",
        "date_difference",
        "date_order",
        "latest",
        "latest_state",
        "ordered_unique_list",
        "entity_exact_match",
        "entity_mismatch",
        "set_difference",
        "sort",
        "semantic_composition",
        "constraint_application",
        "numeric_sum",
        "numeric_multiply",
        "numeric_average",
        "numeric_difference",
        "duration_difference",
        "relative_numeric_offset",
        "count_distinct",
    }
)

PREMISE_ROLE_ALIASES = {
    "preference": "constraint",
    "condition": "constraint",
    "requirement": "constraint",
    "context": "scope",
}

RISK_AGGREGATE_WITHOUT_TYPED_INVENTORY = "aggregate_without_typed_inventory"
RISK_TEMPORAL_WITHOUT_OPERATION = "temporal_without_operation"
RISK_MULTI_STATE_WITHOUT_LATEST = "multi_state_without_latest"
RISK_MEMORY_CONDITIONED_WITHOUT_GROUNDED_CONSTRAINTS = (
    "memory_conditioned_without_grounded_constraints"
)
RISK_PLANNER_MISSING_WITH_PLAUSIBLE_SOURCE = (
    "planner_missing_with_plausible_source"
)
RISK_SIGNALS = frozenset(
    {
        RISK_AGGREGATE_WITHOUT_TYPED_INVENTORY,
        RISK_TEMPORAL_WITHOUT_OPERATION,
        RISK_MULTI_STATE_WITHOUT_LATEST,
        RISK_MEMORY_CONDITIONED_WITHOUT_GROUNDED_CONSTRAINTS,
        RISK_PLANNER_MISSING_WITH_PLAUSIBLE_SOURCE,
    }
)

_ROOT_REQUIRED = frozenset(
    {"schema_version", "output_origin", "target", "output", "premises"}
)
_ROOT_OPTIONAL = frozenset({"operations", "risk_signals"})
_TARGET_REQUIRED = frozenset({"subject", "relation", "entity_constraints"})
_TARGET_OPTIONAL = frozenset({"temporal_constraints", "state_constraints"})
_OUTPUT_FIELDS = frozenset({"shape", "cardinality", "order"})
_PREMISE_FIELDS = frozenset(
    {
        "premise_id",
        "description",
        "role",
        "necessity",
        "source",
        "grounded_constraints",
        "context_quote",
    }
)
_OPERATION_FIELDS = frozenset(
    {"operation_id", "operation_type", "input_premise_ids", "output_ref", "parameters"}
)

_RECOMMENDATION_PATTERN = re.compile(
    r"\b(?:recommend(?:ation|ations|ed)?|advice|advise|suggest(?:ion|ions|ed)?)\b",
    re.IGNORECASE,
)
_HISTORICAL_RECOMMENDATION_PATTERN = re.compile(
    r"\b(?:past|previous|previously|earlier|historical|last|remember|remembered|remind|before|was|were|did)\b",
    re.IGNORECASE,
)
_HISTORICAL_RECOMMENDATION_RELATION_PATTERN = re.compile(
    r"\b(?:recommended|suggested|advised)\b", re.IGNORECASE
)
_TEMPORAL_OPERATION_PATTERN = re.compile(
    r"\b(?:between|elapsed|duration|difference|how long|how many (?:day|days|week|weeks|month|months|year|years)|before|after|earlier|later|chronological|date order)\b",
    re.IGNORECASE,
)
_INVENTORY_PATTERN = re.compile(
    r"\b(?:count|all|every|list|set|inventory|items|distinct)\b", re.IGNORECASE
)
_MULTI_STATE_PATTERN = re.compile(
    r"\b(?:multi[-_ ]?state|state history|state transition|transitions?|history|"
    r"changed|changes|previous|prior|current and|old and new|before and after)\b",
    re.IGNORECASE,
)


class TaskContractError(ValueError):
    """Raised when a task contract violates its schema or invariants."""


ContractValidationError = TaskContractError


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskContractError(f"{path} must be an object")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    path: str,
) -> None:
    fields = set(value)
    allowed = set(required) | set(optional)
    if not required.issubset(fields) or not fields.issubset(allowed):
        raise TaskContractError(f"{path} fields are invalid")


def _string(value: Any, path: str) -> str:
    result = _text(value)
    if not result:
        raise TaskContractError(f"{path} must be a non-empty string")
    return result


def _string_list(value: Any, path: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list):
        raise TaskContractError(f"{path} must be an array")
    if not allow_empty and not value:
        raise TaskContractError(f"{path} must not be empty")
    result: list[str] = []
    for index, item in enumerate(value):
        item_text = _string(item, f"{path}[{index}]")
        if item_text in result:
            raise TaskContractError(f"{path} contains a duplicate value")
        result.append(item_text)
    return result


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise TaskContractError(f"{path} must be a boolean")
    return value


def _recommendation_like(target: Mapping[str, Any]) -> bool:
    values = [target.get("subject"), target.get("relation")]
    for key in ("entity_constraints", "temporal_constraints", "state_constraints"):
        values.extend(target.get(key) or [])
    text = " ".join(_text(value) for value in values if _text(value))
    historical = bool(
        _HISTORICAL_RECOMMENDATION_RELATION_PATTERN.search(text)
        or _HISTORICAL_RECOMMENDATION_PATTERN.search(text)
    )
    return bool(_RECOMMENDATION_PATTERN.search(text) and not historical)


def _validate_target(value: Any) -> dict[str, Any]:
    target = _mapping(value, "target")
    _exact_fields(target, required=_TARGET_REQUIRED, optional=_TARGET_OPTIONAL, path="target")
    normalized = {
        "subject": _string(target.get("subject"), "target.subject"),
        "relation": _string(target.get("relation"), "target.relation"),
        "entity_constraints": _string_list(
            target.get("entity_constraints"), "target.entity_constraints"
        ),
        "temporal_constraints": _string_list(
            target.get("temporal_constraints", []), "target.temporal_constraints"
        ),
        "state_constraints": _string_list(
            target.get("state_constraints", []), "target.state_constraints"
        ),
    }
    return normalized


def _validate_output(value: Any) -> dict[str, str]:
    output = _mapping(value, "output")
    _exact_fields(output, required=_OUTPUT_FIELDS, path="output")
    normalized = {
        "shape": _string(output.get("shape"), "output.shape"),
        "cardinality": _string(output.get("cardinality"), "output.cardinality"),
        "order": _string(output.get("order"), "output.order"),
    }
    if normalized["shape"] not in OUTPUT_SHAPES:
        raise TaskContractError("output.shape is invalid")
    if normalized["cardinality"] not in OUTPUT_CARDINALITIES:
        raise TaskContractError("output.cardinality is invalid")
    if normalized["order"] not in OUTPUT_ORDERS:
        raise TaskContractError("output.order is invalid")
    if normalized["shape"] == "count" and normalized["order"] != "none":
        raise TaskContractError("count output must have order=none")
    return normalized


def _validate_premises(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TaskContractError("premises must be an array")
    if not value:
        raise TaskContractError("premises must not be empty")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        path = f"premises[{index}]"
        premise = _mapping(raw, path)
        _exact_fields(premise, required=_PREMISE_FIELDS, path=path)
        premise_id = _string(premise.get("premise_id"), f"{path}.premise_id")
        if premise_id in seen:
            raise TaskContractError(f"{path}.premise_id is duplicated")
        seen.add(premise_id)
        role = _string(premise.get("role"), f"{path}.role")
        role = PREMISE_ROLE_ALIASES.get(role, role)
        necessity = _string(premise.get("necessity"), f"{path}.necessity")
        source = _string(premise.get("source"), f"{path}.source")
        if role not in PREMISE_ROLES:
            raise TaskContractError(f"{path}.role is invalid")
        if necessity not in PREMISE_NECESSITY:
            raise TaskContractError(f"{path}.necessity is invalid")
        if source not in PREMISE_SOURCES:
            raise TaskContractError(f"{path}.source is invalid")
        context_quote = premise.get("context_quote")
        if not isinstance(context_quote, str):
            raise TaskContractError(f"{path}.context_quote must be a string")
        context_quote = context_quote.strip()
        result.append(
            {
                "premise_id": premise_id,
                "description": _string(premise.get("description"), f"{path}.description"),
                "role": role,
                "necessity": necessity,
                "source": source,
                "grounded_constraints": _string_list(
                    premise.get("grounded_constraints"),
                    f"{path}.grounded_constraints",
                ),
                "context_quote": context_quote,
            }
        )
    if not any(item["necessity"] == "required" for item in result):
        raise TaskContractError("premises needs at least one required item")
    return result


def _validate_operations(value: Any, premise_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TaskContractError("operations must be an array")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        path = f"operations[{index}]"
        operation = _mapping(raw, path)
        _exact_fields(operation, required=_OPERATION_FIELDS, path=path)
        operation_id = _string(operation.get("operation_id"), f"{path}.operation_id")
        if operation_id in seen:
            raise TaskContractError(f"{path}.operation_id is duplicated")
        seen.add(operation_id)
        operation_type = _string(operation.get("operation_type"), f"{path}.operation_type")
        if operation_type not in OPERATION_TYPES:
            raise TaskContractError(f"{path}.operation_type is invalid")
        inputs = _string_list(
            operation.get("input_premise_ids"),
            f"{path}.input_premise_ids",
            allow_empty=False,
        )
        if not set(inputs).issubset(premise_ids):
            raise TaskContractError(f"{path}.input_premise_ids contains an unknown premise")
        output_ref = _string(operation.get("output_ref"), f"{path}.output_ref")
        parameters = operation.get("parameters")
        if not isinstance(parameters, Mapping):
            raise TaskContractError(f"{path}.parameters must be an object")
        result.append(
            {
                "operation_id": operation_id,
                "operation_type": operation_type,
                "input_premise_ids": inputs,
                "output_ref": output_ref,
                "parameters": dict(parameters),
            }
        )
    return result


def _validate_memory_semantics(
    output_origin: str,
    target: Mapping[str, Any],
    premises: Sequence[Mapping[str, Any]],
) -> None:
    required_memory = [
        item
        for item in premises
        if item["source"] == "memory" and item["necessity"] == "required"
    ]
    if output_origin in {"memory_direct", "memory_derived", "memory_conditioned_generation"}:
        if not required_memory:
            raise TaskContractError("memory output origin needs a required memory premise")
    if output_origin == "external_required" and not any(
        item["source"] == "external_tool" and item["necessity"] == "required"
        for item in premises
    ):
        raise TaskContractError("external_required needs a required external_tool premise")
    if _recommendation_like(target):
        if output_origin != "memory_conditioned_generation":
            raise TaskContractError(
                "recommendation/advice must use memory_conditioned_generation"
            )
        if not required_memory:
            raise TaskContractError(
                "recommendation/advice needs a required memory constraint premise"
            )


def validate_task_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one contract without I/O or model calls.

    Unknown fields are rejected.  The returned object is a fresh plain-dict
    normalization and never mutates ``value``.
    """

    if not isinstance(value, Mapping):
        raise TaskContractError("task contract must be an object")
    _exact_fields(value, required=_ROOT_REQUIRED, optional=_ROOT_OPTIONAL, path="task contract")
    if value.get("schema_version") != TASK_CONTRACT_SCHEMA:
        raise TaskContractError("schema_version is invalid")
    output_origin = _string(value.get("output_origin"), "output_origin")
    if output_origin not in OUTPUT_ORIGINS:
        raise TaskContractError("output_origin is invalid")
    target = _validate_target(value.get("target"))
    output = _validate_output(value.get("output"))
    premises = _validate_premises(value.get("premises"))
    operations = (
        _validate_operations(
            value["operations"], {item["premise_id"] for item in premises}
        )
        if "operations" in value
        else []
    )
    _validate_memory_semantics(output_origin, target, premises)
    if "risk_signals" in value:
        signals = _string_list(value.get("risk_signals"), "risk_signals")
        if any(signal not in RISK_SIGNALS for signal in signals):
            raise TaskContractError("risk_signals contains an unknown signal")
    normalized = {
        "schema_version": TASK_CONTRACT_SCHEMA,
        "output_origin": output_origin,
        "target": target,
        "output": output,
        "premises": premises,
        "operations": operations,
    }
    if "risk_signals" in value:
        normalized["risk_signals"] = list(value["risk_signals"])
    return normalized


def _target_text(contract: Mapping[str, Any]) -> str:
    target = contract.get("target")
    if not isinstance(target, Mapping):
        return ""
    values: list[str] = [_text(target.get("subject")), _text(target.get("relation"))]
    for key in ("entity_constraints", "temporal_constraints", "state_constraints"):
        raw = target.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values.extend(_text(item) for item in raw)
    return " ".join(value for value in values if value)


def _premise_items(contract: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = contract.get("premises")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _has_typed_inventory(premises: Sequence[Mapping[str, Any]]) -> bool:
    for premise in premises:
        if premise.get("role") != "inventory":
            continue
        constraints = premise.get("grounded_constraints")
        if isinstance(constraints, list) and constraints:
            return True
        if re.search(r"\b(?:typed|type|entity|inventory)\b", _text(premise.get("description")), re.I):
            return True
    return False


def _has_temporal_operation(operations: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        _text(operation.get("operation_type"))
        in {"date_difference", "date_order", "latest", "latest_state", "sort"}
        for operation in operations
    )


def _has_latest_operation(operations: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        _text(operation.get("operation_type")) in {"latest", "latest_state"}
        for operation in operations
    )


def _is_aggregate(contract: Mapping[str, Any], target_text: str) -> bool:
    output = contract.get("output")
    shape = output.get("shape") if isinstance(output, Mapping) else ""
    return shape in {"count", "list", "set"} or bool(_INVENTORY_PATTERN.search(target_text))


def _is_temporal(contract: Mapping[str, Any], target_text: str) -> bool:
    output = contract.get("output")
    shape = output.get("shape") if isinstance(output, Mapping) else ""
    return shape in {"date", "duration"} or bool(
        _TEMPORAL_OPERATION_PATTERN.search(target_text)
    )


def _is_multi_state(contract: Mapping[str, Any], target_text: str) -> bool:
    target = contract.get("target")
    if isinstance(target, Mapping):
        constraints = target.get("state_constraints")
        if isinstance(constraints, list) and len(constraints) >= 2:
            return True
    state_premises = [item for item in _premise_items(contract) if item.get("role") == "state"]
    return len(state_premises) >= 2 or bool(_MULTI_STATE_PATTERN.search(target_text))


def _has_plausible_source(contract: Mapping[str, Any]) -> bool:
    if any(item.get("source") in PREMISE_SOURCES for item in _premise_items(contract)):
        return True
    target = contract.get("target")
    return isinstance(target, Mapping) and bool(
        _text(target.get("subject")) and _text(target.get("relation"))
    )


def structural_risk_signals(
    contract: Mapping[str, Any],
    *,
    planner_present: bool = True,
    plausible_source: bool | None = None,
) -> list[str]:
    """Return deterministic structural risk codes for a raw or valid contract.

    This function intentionally does not raise for malformed input: callers can
    inspect risks before deciding whether to invoke the strict validator.  Set
    ``planner_present=False`` when a planner stage was skipped or its artifact is
    missing.  ``plausible_source`` can override the local source heuristic.
    """

    target_text = _target_text(contract)
    premises = _premise_items(contract)
    operations_raw = contract.get("operations")
    operations = (
        [item for item in operations_raw if isinstance(item, Mapping)]
        if isinstance(operations_raw, list)
        else []
    )
    output_origin = _text(contract.get("output_origin"))
    risks: list[str] = []
    if _is_aggregate(contract, target_text) and not _has_typed_inventory(premises):
        risks.append(RISK_AGGREGATE_WITHOUT_TYPED_INVENTORY)
    if _is_temporal(contract, target_text) and not _has_temporal_operation(operations):
        risks.append(RISK_TEMPORAL_WITHOUT_OPERATION)
    if _is_multi_state(contract, target_text) and not _has_latest_operation(operations):
        risks.append(RISK_MULTI_STATE_WITHOUT_LATEST)
    if output_origin == "memory_conditioned_generation":
        grounded = any(
            item.get("source") == "memory"
            and item.get("necessity") == "required"
            and isinstance(item.get("grounded_constraints"), list)
            and bool(item.get("grounded_constraints"))
            for item in premises
        )
        if not grounded:
            risks.append(RISK_MEMORY_CONDITIONED_WITHOUT_GROUNDED_CONSTRAINTS)
    if not planner_present and (
        plausible_source if plausible_source is not None else _has_plausible_source(contract)
    ):
        risks.append(RISK_PLANNER_MISSING_WITH_PLAUSIBLE_SOURCE)
    return risks


def assess_structural_risks(
    contract: Mapping[str, Any],
    *,
    planner_present: bool = True,
    plausible_source: bool | None = None,
) -> list[str]:
    """Descriptive alias for :func:`structural_risk_signals`."""

    return structural_risk_signals(
        contract,
        planner_present=planner_present,
        plausible_source=plausible_source,
    )


validate_contract = validate_task_contract
get_structural_risk_signals = structural_risk_signals


__all__ = [
    "TASK_CONTRACT_SCHEMA",
    "SCHEMA_VERSION",
    "OUTPUT_ORIGINS",
    "OUTPUT_SHAPES",
    "OUTPUT_CARDINALITIES",
    "OUTPUT_ORDERS",
    "PREMISE_SOURCES",
    "PREMISE_ROLES",
    "OPERATION_TYPES",
    "RISK_SIGNALS",
    "RISK_AGGREGATE_WITHOUT_TYPED_INVENTORY",
    "RISK_TEMPORAL_WITHOUT_OPERATION",
    "RISK_MULTI_STATE_WITHOUT_LATEST",
    "RISK_MEMORY_CONDITIONED_WITHOUT_GROUNDED_CONSTRAINTS",
    "RISK_PLANNER_MISSING_WITH_PLAUSIBLE_SOURCE",
    "TaskContractError",
    "ContractValidationError",
    "validate_task_contract",
    "validate_contract",
    "structural_risk_signals",
    "get_structural_risk_signals",
    "assess_structural_risks",
]
