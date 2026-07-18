"""Strict, local typed semantic proposal validation and execution.

This module deliberately has no model, network, or repository dependencies. It
is an advisory computation layer: accepted results are useful deterministic
derivations, but neither accepted nor rejected proposals establish absence or
any other authoritative conclusion.

Input schema (mapping form)
---------------------------
An observation must contain ``observation_id``, ``evidence_ids``,
``entity_key``, ``value_kind``, ``value``, ``unit``, ``temporal_kind``, and
``polarity``. Event and entity observations may also carry an explicit
``event_status`` (``actual``, ``planned``, ``hypothetical``, or ``mentioned``).
``time`` is optional for ordinary observations and required by ``latest``. A candidate contains ``candidate_id`` and ``operations``. Each
operation contains ``operation_id``, ``operation``, and ``input_ids`` (the
aliases ``input_observation_ids`` and ``operands`` are also accepted).

The public evaluator accepts dictionaries or the dataclasses below. It returns
plain dictionaries to keep the result easy to serialize and inspect.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import math
import re
from typing import Any, Mapping, Sequence


VALUE_KINDS = {
    "entity_instance",
    "event",
    "state_snapshot",
    "cumulative_snapshot",
    "delta",
    "aggregated_quantity",
    "rate",
    "scalar",
    "date",
}
TEMPORAL_KINDS = {"absolute", "relative_anchored", "relative_unresolved", "none"}
POLARITIES = {"positive", "negative", "unknown"}
OPERATIONS = {
    "latest",
    "numeric_sum",
    "numeric_multiply",
    "numeric_average",
    "numeric_difference",
    "duration_difference",
    "relative_numeric_offset",
    "count_distinct",
    "date_order",
    "date_difference",
    "entity_exact_match",
}
SNAPSHOT_KINDS = {"state_snapshot", "cumulative_snapshot"}
EVENT_STATUSES = {"actual", "planned", "hypothetical", "mentioned"}
NUMERIC_KINDS = {"delta", "scalar", "aggregated_quantity"}


@dataclass(frozen=True)
class Observation:
    """Typed source observation.

    ``time`` is intentionally separate from ``value``. The required schema
    fields describe the semantic value; temporal operations need an explicit
    sortable anchor in addition to ``temporal_kind``.
    """

    observation_id: str
    evidence_ids: Sequence[str]
    entity_key: str
    value_kind: str
    value: Any
    unit: str | None
    temporal_kind: str
    polarity: str
    time: Any = None
    event_status: str | None = None


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    path: str = ""

    def to_dict(self) -> dict[str, str]:
        result = {"code": self.code, "message": self.message}
        if self.path:
            result["path"] = self.path
        return result


@dataclass(frozen=True)
class _TypedValue:
    ref: str
    value_kind: str
    value: Any
    unit: str | None
    entity_key: str
    temporal_kind: str
    time: Any
    source_evidence_ids: tuple[str, ...]
    is_observation: bool
    event_status: str | None = None


class _Rejected(Exception):
    def __init__(self, diagnostic: Diagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


def _diag(code: str, message: str, path: str = "") -> _Rejected:
    return _Rejected(Diagnostic(code, message, path))


def _as_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _diag("UNTYPED_VALUE", "expected an object with an explicit type", path)
    return value


def _nonempty_text(value: Any, *, code: str, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _diag(code, "expected a non-empty string", path)
    return value


def _evidence_ids(value: Any, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise _diag("UNTYPED_EVIDENCE", "evidence_ids must be a non-empty list", path)
    result: list[str] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        text = _nonempty_text(item, code="UNTYPED_EVIDENCE", path=item_path)
        if text not in result:
            result.append(text)
    return tuple(result)


def _observation_mapping(raw: Any, *, path: str) -> Mapping[str, Any]:
    if isinstance(raw, Observation):
        result = asdict(raw)
        if result.get("event_status") is None:
            result.pop("event_status", None)
        return result
    return _as_mapping(raw, path=path)


def _normalize_observation(raw: Any, *, index: int) -> _TypedValue:
    path = f"observations[{index}]"
    value = _observation_mapping(raw, path=path)
    required = (
        "observation_id",
        "evidence_ids",
        "entity_key",
        "value_kind",
        "value",
        "unit",
        "temporal_kind",
        "polarity",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise _diag("UNTYPED_OBSERVATION", f"missing typed fields: {', '.join(missing)}", path)

    observation_id = _nonempty_text(value["observation_id"], code="UNTYPED_OBSERVATION", path=f"{path}.observation_id")
    entity_key = _nonempty_text(value["entity_key"], code="UNTYPED_OBSERVATION", path=f"{path}.entity_key")
    value_kind = _nonempty_text(value["value_kind"], code="UNTYPED_OBSERVATION", path=f"{path}.value_kind")
    temporal_kind = _nonempty_text(value["temporal_kind"], code="UNTYPED_OBSERVATION", path=f"{path}.temporal_kind")
    polarity = _nonempty_text(value["polarity"], code="UNTYPED_OBSERVATION", path=f"{path}.polarity")
    if value_kind not in VALUE_KINDS:
        raise _diag("UNKNOWN_VALUE_KIND", f"unsupported value_kind: {value_kind!r}", f"{path}.value_kind")
    if temporal_kind not in TEMPORAL_KINDS:
        raise _diag("UNKNOWN_TEMPORAL_KIND", f"unsupported temporal_kind: {temporal_kind!r}", f"{path}.temporal_kind")
    if polarity not in POLARITIES:
        raise _diag("UNKNOWN_POLARITY", f"unsupported polarity: {polarity!r}", f"{path}.polarity")
    unit = value["unit"]
    if unit is not None and (not isinstance(unit, str) or not unit.strip()):
        raise _diag("UNTYPED_UNIT", "unit must be a non-empty string or null", f"{path}.unit")
    evidence_ids = _evidence_ids(value["evidence_ids"], path=f"{path}.evidence_ids")
    status_fields = [field for field in ("event_status", "event_state", "status") if field in value]
    statuses = [value[field] for field in status_fields]
    if any(status != statuses[0] for status in statuses[1:]):
        raise _diag("CONFLICTING_EVENT_STATUS", "event status aliases disagree", f"{path}.event_status")
    event_status = None
    if statuses:
        event_status = _nonempty_text(statuses[0], code="UNTYPED_EVENT_STATUS", path=f"{path}.{status_fields[0]}").strip().lower()
        if event_status not in EVENT_STATUSES:
            raise _diag("UNKNOWN_EVENT_STATUS", f"unsupported event_status: {event_status!r}", f"{path}.{status_fields[0]}")
    if value_kind == "event" and event_status is None:
        raise _diag("MISSING_EVENT_STATUS", "event observations require an explicit event_status", f"{path}.event_status")
    time_value = value.get("time", value.get("temporal_value", value.get("timestamp", value.get("observed_at"))))
    return _TypedValue(
        ref=observation_id,
        value_kind=value_kind,
        value=value["value"],
        unit=unit.strip() if isinstance(unit, str) else None,
        entity_key=entity_key,
        temporal_kind=temporal_kind,
        time=time_value,
        source_evidence_ids=evidence_ids,
        is_observation=True,
        event_status=event_status,
    )


def _iter_values(raw: Any, *, singular_fields: set[str]) -> list[Any]:
    if isinstance(raw, Mapping):
        if singular_fields.intersection(raw):
            return [raw]
        return list(raw.values())
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def _normalize_observations(raw: Any) -> tuple[dict[str, _TypedValue], list[Diagnostic]]:
    observations: dict[str, _TypedValue] = {}
    diagnostics: list[Diagnostic] = []
    items = _iter_values(raw, singular_fields={"observation_id", "value_kind"})
    for index, item in enumerate(items):
        try:
            normalized = _normalize_observation(item, index=index)
            if normalized.ref in observations:
                diagnostics.append(Diagnostic("DUPLICATE_OBSERVATION_ID", f"duplicate observation_id: {normalized.ref}", f"observations[{index}].observation_id"))
            else:
                observations[normalized.ref] = normalized
        except _Rejected as exc:
            diagnostics.append(exc.diagnostic)
    return observations, diagnostics


def _decimal(value: Any, *, path: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        raise _diag("NON_NUMERIC_VALUE", "operation requires numeric values", path)
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise _diag("NON_NUMERIC_VALUE", "operation requires numeric values", path)
    if not result.is_finite():
        raise _diag("NON_NUMERIC_VALUE", "numeric values must be finite", path)
    return result


def _json_number(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _unit(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    normalized = normalized.replace(" per ", "/")
    normalized = normalized.replace("usd", "$" ).replace("us dollars", "$")
    return normalized.replace(" ", "")


def _rate_parts(value: str | None) -> tuple[str, str] | None:
    normalized = _unit(value)
    if normalized is None or normalized.count("/") != 1:
        return None
    numerator, denominator = normalized.split("/", 1)
    if not numerator or not denominator:
        return None
    return numerator, denominator


def _join_sources(values: Sequence[_TypedValue]) -> tuple[str, ...]:
    result: list[str] = []
    for item in values:
        for evidence_id in item.source_evidence_ids:
            if evidence_id not in result:
                result.append(evidence_id)
    return tuple(result)


def _date_value(value: Any, *, path: str) -> date | datetime:
    if isinstance(value, (date, datetime)):
        return value
    if not isinstance(value, str) or not value.strip():
        raise _diag("INVALID_DATE", "date operation requires an ISO date or datetime value", path)
    text = value.strip()
    try:
        if "T" in text or " " in text:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        return date.fromisoformat(text)
    except ValueError:
        raise _diag("INVALID_DATE", "date operation requires an ISO date or datetime value", path)


def _sortable_time(value: _TypedValue, *, path: str) -> tuple[str, Any]:
    if value.time is None:
        raise _diag("MISSING_TIME", "latest requires an explicit sortable time", path)
    if isinstance(value.time, (int, float, Decimal)) and not isinstance(value.time, bool):
        return "number", Decimal(str(value.time))
    if isinstance(value.time, datetime):
        return "datetime", value.time
    if isinstance(value.time, date):
        return "date", value.time
    if isinstance(value.time, str):
        text = value.time.strip()
        try:
            if "T" in text or " " in text:
                return "datetime", datetime.fromisoformat(text.replace("Z", "+00:00"))
            return "date", date.fromisoformat(text)
        except ValueError:
            raise _diag("UNORDERABLE_TIME", "latest requires an ISO or numeric time anchor", path)
    raise _diag("UNORDERABLE_TIME", "latest requires an ISO or numeric time anchor", path)


def _resolve_inputs(
    raw_ids: Any,
    *,
    observations: Mapping[str, _TypedValue],
    results: Mapping[str, _TypedValue],
    path: str,
) -> list[_TypedValue]:
    if not isinstance(raw_ids, (list, tuple)) or not raw_ids:
        raise _diag("UNTYPED_OPERANDS", "operation inputs must be a non-empty list", path)
    resolved: list[_TypedValue] = []
    for index, ref in enumerate(raw_ids):
        ref_path = f"{path}[{index}]"
        if not isinstance(ref, str) or not ref.strip():
            raise _diag("UNTYPED_OPERAND", "operation input must be an explicit reference", ref_path)
        if ref in observations:
            resolved.append(observations[ref])
        elif ref in results:
            resolved.append(results[ref])
        else:
            raise _diag("UNKNOWN_REFERENCE", f"unknown observation or operation reference: {ref}", ref_path)
    return resolved


def _operation_name(raw: Mapping[str, Any], *, path: str) -> str:
    values = [raw.get(field) for field in ("operation", "operation_type", "op") if field in raw]
    if not values or not isinstance(values[0], str) or not values[0].strip():
        raise _diag("UNTYPED_OPERATION", "operation must name a typed deterministic operation", path)
    if any(value != values[0] for value in values[1:]):
        raise _diag("CONFLICTING_OPERATION_TYPE", "operation aliases disagree", path)
    operation = values[0]
    if operation not in OPERATIONS:
        raise _diag("UNKNOWN_OPERATION", f"unsupported operation: {operation!r}", path)
    return operation


def _input_field(raw: Mapping[str, Any], *, path: str) -> Any:
    fields = ("input_ids", "input_observation_ids", "observation_ids", "operands", "inputs")
    present = [field for field in fields if field in raw]
    if not present:
        raise _diag("UNTYPED_OPERANDS", "operation must explicitly list its inputs", path)
    first = raw[present[0]]
    if any(raw[field] != first for field in present[1:]):
        raise _diag("CONFLICTING_OPERANDS", "operation input aliases disagree", path)
    return first


def _derived(
    operation_id: str,
    value_kind: str,
    value: Any,
    unit: str | None,
    source_values: Sequence[_TypedValue],
    *,
    entity_key: str = "derived",
    temporal_kind: str = "none",
    time: Any = None,
) -> _TypedValue:
    return _TypedValue(
        ref=operation_id,
        value_kind=value_kind,
        value=value,
        unit=unit,
        entity_key=entity_key,
        temporal_kind=temporal_kind,
        time=time,
        source_evidence_ids=_join_sources(source_values),
        is_observation=False,
        event_status=None,
    )


def _numeric_operands(
    operation: str,
    inputs: Sequence[_TypedValue],
    *,
    path: str,
    allowed_kinds: set[str] = NUMERIC_KINDS,
) -> tuple[list[Decimal], str]:
    if any(item.value_kind not in allowed_kinds for item in inputs):
        allowed = ", ".join(sorted(allowed_kinds))
        raise _diag("INVALID_NUMERIC_ROLE", f"{operation} accepts only {allowed} values", path)
    units = {_unit(item.unit) for item in inputs}
    if len(units) != 1 or None in units:
        raise _diag("INCOMPATIBLE_UNITS", f"{operation} requires non-null, identical units", path)
    return [_decimal(item.value, path=path) for item in inputs], inputs[0].unit


def _execute_operation(
    operation: str,
    operation_id: str,
    inputs: Sequence[_TypedValue],
    parameters: Mapping[str, Any],
    *,
    path: str,
) -> _TypedValue:
    if operation == "count_distinct":
        if any(item.value_kind not in {"entity_instance", "event"} for item in inputs):
            raise _diag("INVALID_COUNT_ROLE", "count_distinct accepts only entity_instance or event values; snapshots and aggregated quantities are not countable", path)
        countable: list[_TypedValue] = []
        for index, item in enumerate(inputs):
            if item.event_status is None:
                raise _diag("MISSING_EVENT_STATUS", "count_distinct requires an explicit event_status on every event/entity input", f"{path}.inputs[{index}].event_status")
            if item.event_status == "actual":
                countable.append(item)
        return _derived(operation_id, "scalar", len({item.entity_key for item in countable}), "count", countable)

    if operation == "numeric_sum":
        if any(item.value_kind not in {"delta", "scalar"} for item in inputs):
            raise _diag("INVALID_SUM_ROLE", "numeric_sum accepts only additive delta or scalar values, never rates, snapshots, or aggregated quantities", path)
        values, unit = _numeric_operands(operation, inputs, path=path, allowed_kinds={"delta", "scalar"})
        total = sum(values, Decimal(0))
        return _derived(operation_id, "scalar", _json_number(total), unit, inputs)

    if operation in {"numeric_average", "numeric_difference", "duration_difference", "relative_numeric_offset"}:
        if operation in {"numeric_difference", "duration_difference", "relative_numeric_offset"} and len(inputs) != 2:
            raise _diag("INVALID_NUMERIC_ARITY", f"{operation} requires exactly two numeric values", path)
        values, unit = _numeric_operands(operation, inputs, path=path)
        if operation == "numeric_average":
            result = sum(values, Decimal(0)) / Decimal(len(values))
        else:
            result = values[0] - values[1]
        return _derived(operation_id, "scalar", _json_number(result), unit, inputs)

    if operation == "numeric_multiply":
        if len(inputs) != 2:
            raise _diag("INVALID_MULTIPLY_ARITY", "numeric_multiply requires exactly one rate and one quantity", path)
        rates = [item for item in inputs if item.value_kind == "rate"]
        quantities = [item for item in inputs if item.value_kind != "rate"]
        if len(rates) != 1 or len(quantities) != 1:
            raise _diag("INVALID_MULTIPLY_ROLE", "numeric_multiply requires exactly one rate and one compatible quantity", path)
        numerator_denominator = _rate_parts(rates[0].unit)
        if numerator_denominator is None:
            raise _diag("INVALID_RATE_UNIT", "rate unit must have the form numerator/denominator", path)
        numerator, denominator = numerator_denominator
        quantity = quantities[0]
        if quantity.value_kind not in {"scalar", "aggregated_quantity", "delta"}:
            raise _diag("INVALID_QUANTITY_ROLE", "the non-rate operand must be a scalar, aggregated quantity, or delta", path)
        if _unit(quantity.unit) != denominator:
            raise _diag("INCOMPATIBLE_RATE_QUANTITY", "quantity unit must exactly match the rate denominator", path)
        product = _decimal(rates[0].value, path=path) * _decimal(quantity.value, path=path)
        return _derived(operation_id, "scalar", _json_number(product), numerator, inputs)

    if operation == "latest":
        if any(item.value_kind not in SNAPSHOT_KINDS for item in inputs):
            raise _diag("INVALID_LATEST_ROLE", "latest requires state_snapshot or cumulative_snapshot values", path)
        if any(item.temporal_kind not in {"absolute", "relative_anchored"} for item in inputs):
            raise _diag("INVALID_LATEST_TIME", "latest requires absolute or relative_anchored temporal kinds", path)
        entities = {item.entity_key for item in inputs}
        if len(entities) != 1:
            raise _diag("MIXED_ENTITIES", "latest requires snapshots for one exact entity_key", path)
        sortable = [_sortable_time(item, path=f"{path}.inputs[{index}]") for index, item in enumerate(inputs)]
        kinds = {kind for kind, _ in sortable}
        if len(kinds) != 1:
            raise _diag("INCOMPARABLE_TIMES", "latest requires mutually comparable time anchors", path)
        latest_index = max(range(len(inputs)), key=lambda index: sortable[index][1])
        selected = inputs[latest_index]
        return _derived(
            operation_id,
            selected.value_kind,
            selected.value,
            selected.unit,
            inputs,
            entity_key=selected.entity_key,
            temporal_kind=selected.temporal_kind,
            time=selected.time,
        )

    if operation in {"date_order", "date_difference"}:
        if len(inputs) != 2:
            raise _diag("INVALID_DATE_ARITY", f"{operation} requires exactly two dates", path)
        if any(item.value_kind != "date" for item in inputs):
            raise _diag("INVALID_DATE_ROLE", f"{operation} accepts only date values", path)
        if any(item.temporal_kind == "relative_unresolved" for item in inputs):
            raise _diag("UNRESOLVED_RELATIVE_DATE", "date operations reject unresolved relative dates", path)
        first = _date_value(inputs[0].value, path=f"{path}.inputs[0]")
        second = _date_value(inputs[1].value, path=f"{path}.inputs[1]")
        try:
            difference = first - second
        except TypeError:
            raise _diag("INCOMPARABLE_DATES", "date values must use compatible timezone information", path)
        if operation == "date_difference":
            days = difference.total_seconds() / 86400
            if isinstance(first, date) and isinstance(second, date) and not isinstance(first, datetime) and not isinstance(second, datetime):
                days = difference.days
            return _derived(operation_id, "scalar", _json_number(Decimal(str(days))), "day", inputs)
        relation = parameters.get("relation", parameters.get("comparison", "before"))
        if relation not in {"before", "lt", "earlier", "after", "gt", "later", "equal", "eq", "same"}:
            raise _diag("UNKNOWN_DATE_RELATION", "date_order relation must be before, after, or equal", f"{path}.parameters")
        try:
            if relation in {"before", "lt", "earlier"}:
                result = first < second
            elif relation in {"after", "gt", "later"}:
                result = first > second
            else:
                result = first == second
        except TypeError:
            raise _diag("INCOMPARABLE_DATES", "date values must use compatible timezone information", path)
        return _derived(operation_id, "scalar", result, None, inputs)

    if operation == "entity_exact_match":
        if len(inputs) != 2:
            raise _diag("INVALID_ENTITY_ARITY", "entity_exact_match requires exactly two entities", path)
        if any(item.value_kind != "entity_instance" for item in inputs):
            raise _diag("INVALID_ENTITY_ROLE", "entity_exact_match requires entity_instance values", path)
        return _derived(operation_id, "scalar", inputs[0].entity_key == inputs[1].entity_key, None, inputs)

    raise _diag("UNKNOWN_OPERATION", f"unsupported operation: {operation!r}", path)


def _candidate_id(raw: Any, *, index: int) -> str:
    if isinstance(raw, Mapping) and isinstance(raw.get("candidate_id"), str) and raw["candidate_id"].strip():
        return raw["candidate_id"]
    return f"candidate_{index}"


def _candidate_mapping(raw: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise _diag("UNTYPED_PROPOSAL", "candidate program must be an object with typed operations", path)
    if "candidate_id" not in raw:
        raise _diag("UNTYPED_PROPOSAL", "candidate program requires candidate_id", path)
    if not isinstance(raw.get("candidate_id"), str) or not raw["candidate_id"].strip():
        raise _diag("UNTYPED_PROPOSAL", "candidate_id must be a non-empty string", f"{path}.candidate_id")
    if not isinstance(raw.get("operations"), (list, tuple)) or not raw["operations"]:
        raise _diag("UNTYPED_PROPOSAL", "candidate program requires a non-empty operations list", f"{path}.operations")
    return raw


def _candidate_source_ids(raw: Mapping[str, Any]) -> tuple[str, ...]:
    value = raw.get("source_evidence_ids", raw.get("evidence_ids", []))
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item.strip())


def _evaluate_candidate(raw: Any, *, index: int, observations: Mapping[str, _TypedValue]) -> dict[str, Any]:
    candidate_id = _candidate_id(raw, index=index)
    sources = list(_candidate_source_ids(raw) if isinstance(raw, Mapping) else ())
    base = {"candidate_id": candidate_id, "authoritative": False, "source_evidence_ids": sources}
    try:
        candidate = _candidate_mapping(raw, path=f"candidates[{index}]")
        results: dict[str, _TypedValue] = {}
        operation_results: list[dict[str, Any]] = []
        operation_ids: set[str] = set()
        for operation_index, raw_operation in enumerate(candidate["operations"]):
            path = f"candidates[{index}].operations[{operation_index}]"
            operation_map = _as_mapping(raw_operation, path=path)
            operation_id = _nonempty_text(operation_map.get("operation_id"), code="UNTYPED_OPERATION", path=f"{path}.operation_id")
            if operation_id in operation_ids:
                raise _diag("DUPLICATE_OPERATION_ID", f"duplicate operation_id: {operation_id}", f"{path}.operation_id")
            operation_ids.add(operation_id)
            operation = _operation_name(operation_map, path=f"{path}.operation")
            raw_inputs = _input_field(operation_map, path=f"{path}.input_ids")
            inputs = _resolve_inputs(raw_inputs, observations=observations, results=results, path=f"{path}.input_ids")
            for evidence_id in _join_sources(inputs):
                if evidence_id not in sources:
                    sources.append(evidence_id)
            parameters = operation_map.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise _diag("UNTYPED_PARAMETERS", "operation parameters must be an object", f"{path}.parameters")
            result = _execute_operation(operation, operation_id, inputs, parameters, path=path)
            results[operation_id] = result
            for evidence_id in result.source_evidence_ids:
                if evidence_id not in sources:
                    sources.append(evidence_id)
            operation_results.append({
                "operation_id": operation_id,
                "operation": operation,
                "value": result.value,
                "value_kind": result.value_kind,
                "unit": result.unit,
                "source_evidence_ids": list(result.source_evidence_ids),
            })

        output_ref = candidate.get("output_ref", candidate.get("output_operation_id"))
        if output_ref is None:
            output_ref = candidate["operations"][-1].get("operation_id") if isinstance(candidate["operations"][-1], Mapping) else None
        if not isinstance(output_ref, str) or output_ref not in results:
            raise _diag("UNKNOWN_OUTPUT_REFERENCE", "candidate output_ref must name an operation result", f"candidates[{index}].output_ref")
        output = results[output_ref]
        return {
            **base,
            "status": "accepted",
            "value": output.value,
            "value_kind": output.value_kind,
            "unit": output.unit,
            "result": operation_results[-1] if output_ref == operation_results[-1]["operation_id"] else next(item for item in operation_results if item["operation_id"] == output_ref),
            "operation_results": operation_results,
            "source_evidence_ids": sources,
            "evidence_ids": list(sources),
        }
    except _Rejected as exc:
        return {
            **base,
            "status": "rejected",
            "diagnostics": [exc.diagnostic.to_dict()],
            "source_evidence_ids": sources,
            "evidence_ids": list(sources),
        }


def evaluate_proposals(observations: Any, proposals: Any) -> dict[str, Any]:
    """Validate and execute typed candidate programs without authoritative claims.

    Each candidate is evaluated independently. Malformed observations make
    candidates that reference them reject with diagnostics; they are never
    converted into an absence or ``not found`` result.
    """

    normalized, observation_diagnostics = _normalize_observations(observations)
    candidates = _iter_values(proposals, singular_fields={"candidate_id", "operations"})
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        result = _evaluate_candidate(candidate, index=index, observations=normalized)
        if observation_diagnostics:
            references = set()
            if isinstance(candidate, Mapping):
                for operation in candidate.get("operations", []) or []:
                    if isinstance(operation, Mapping):
                        for key in ("input_ids", "input_observation_ids", "observation_ids", "operands", "inputs"):
                            value = operation.get(key, [])
                            if isinstance(value, (list, tuple)):
                                references.update(item for item in value if isinstance(item, str))
            if references.intersection(normalized) or not normalized:
                result.setdefault("diagnostics", []).extend(item.to_dict() for item in observation_diagnostics)
                if result["status"] == "accepted":
                    result["status"] = "rejected"
                    result.pop("value", None)
                    result.pop("value_kind", None)
                    result.pop("unit", None)
                    result.pop("result", None)
                    result.pop("operation_results", None)
                    rejected.append(result)
                    continue
        if result["status"] == "accepted":
            accepted.append(result)
        else:
            rejected.append(result)
    return {
        "status": "evaluated",
        "advisory": True,
        "authoritative": False,
        "accepted": accepted,
        "rejected": rejected,
    }


__all__ = [
    "Diagnostic",
    "EVENT_STATUSES",
    "Observation",
    "OPERATIONS",
    "POLARITIES",
    "TEMPORAL_KINDS",
    "VALUE_KINDS",
    "evaluate_proposals",
]
