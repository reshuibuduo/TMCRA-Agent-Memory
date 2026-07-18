"""DeepSeek Pro operation planner for compiled TMCRA evidence packets."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from tmcra_v4_evidence_operations import OPERATION_TYPES, PLAN_SCHEMA, validate_operation_plan
from tmcra_v4_task_contract import (
    OUTPUT_CARDINALITIES,
    OUTPUT_ORIGINS,
    OUTPUT_ORDERS,
    OUTPUT_SHAPES,
    PREMISE_NECESSITY,
    PREMISE_ROLES,
    PREMISE_SOURCES,
    TASK_CONTRACT_SCHEMA,
)
from tmcra_v4_typed_semantics import (
    EVENT_STATUSES,
    OPERATIONS as TYPED_OPERATIONS,
    POLARITIES,
    TEMPORAL_KINDS,
    VALUE_KINDS,
)


MODEL = "deepseek-v4-pro"
PROMPT_VERSION = "tmcra-evidence-operation-planner-2026-07-13.8"
SYSTEM_PROMPT = f"""You compile one memory question into an evidence operation plan.
Return exactly one JSON object using schema {PLAN_SCHEMA}:
{{"schema_version":"{PLAN_SCHEMA}","requirements":[{{"requirement_id":"R01","description":"required answer field","evidence_ids":["E01"]}}],"operations":[{{"operation_id":"O01","operation_type":"one supported operation","input_atom_ids":["D001"],"input_evidence_ids":["E01"],"parameters":{{}}}}],"bundles":[{{"bundle_id":"B01","role":"direct|operand|temporal_sequence|historical_state|current_state|preference|counterevidence|context","evidence_ids":["E01"]}}]}}

Supported deterministic operations: {', '.join(sorted(OPERATION_TYPES))}.
Legacy root fields are always required, even when arrays are empty. Every new planner output must also include task_contract and typed_semantics; typed_semantics is exactly {{"observations":[],"proposals":[]}} and its arrays may be empty. The runtime accepts old packets without these fields only for backward-compatible replay.
Use only supplied evidence IDs and atom IDs. Never invent an operand, date, number, entity, source ID, answer, fact, summary, or benchmark field.
Requirements describe every field or operand needed to answer. Use an empty evidence_ids array when the reservoir does not satisfy a requirement.
Operations are optional. Use them only when supplied atoms exactly support the calculation. Dates, differences, order, sums, averages, counts, and exact comparisons must use operations instead of mental arithmetic.
Atoms with derivation=source_timestamp_minus_1_day or source_timestamp_same_day are deterministic calendar dates denoted by relative language in that evidence. Use those derived atoms as event dates. When a question describes two remembered events, create separate requirements for both events, find evidence that explicitly supports each description, and calculate between those event dates. Do not assume one source supports both event descriptions unless its text explicitly contains both. lexical_anchor_ids and question_overlap_terms are deterministic search hints, not proof; inspect distinct anchors because separate phrases may identify separate events. In a relative question, words such as "ago" may be anchored by the other remembered event named in the question; use the question date only when the question clearly asks for elapsed time up to the question date. QUESTION atoms may be operation inputs but QUESTION is never an evidence ID.
For count or list questions, inventory every matching event in the reservoir, bind all supporting evidence IDs, and distinguish separate items from duplicate mentions. Use count_distinct or ordered_unique_list only when entity atoms are supplied; otherwise organize the matching sources and leave operations empty so the answer layer can enumerate grounded items. Do not mark the requirement missing merely because there is no pre-extracted entity atom.
Treat named entities and activities exactly. Evidence for a shorter, broader, or merely related term does not satisfy a multiword target unless the supplied source explicitly establishes equivalence. Preserve such evidence as counterevidence rather than silently substituting it.
Bundles organize source evidence without deleting it. For preferences retain specific prior experiences. For updates retain old and current states. For ambiguous or false-premise questions retain counterevidence and entity distinctions.
Each evidence item may include memory_contexts and attachments. A slow_context is an auditable Slow-graph claim that organizes durable identity, preference, routine, relationship, or standing state around that Source item. A fast_context is a current atomic Fast memory. An attachment with role=override may supersede a same-slot slow_context only when precedence=newer_fast_evidence. Use these graph views to bind the relevant Source evidence IDs and organize bundles, but never cite a capsule ID, claim ID, or memory ID as evidence and never let a graph view replace inspection of the immutable Source text and source_group_context.
task_contract must use this exact {TASK_CONTRACT_SCHEMA} shape: {{"schema_version":"{TASK_CONTRACT_SCHEMA}","output_origin":"{'|'.join(sorted(OUTPUT_ORIGINS))}","target":{{"subject":"...","relation":"...","entity_constraints":["..."],"temporal_constraints":["..."],"state_constraints":["..."]}},"output":{{"shape":"{'|'.join(sorted(OUTPUT_SHAPES))}","cardinality":"{'|'.join(sorted(OUTPUT_CARDINALITIES))}","order":"{'|'.join(sorted(OUTPUT_ORDERS))}"}},"premises":[{{"premise_id":"R01","description":"...","role":"{'|'.join(sorted(PREMISE_ROLES))}","necessity":"{'|'.join(sorted(PREMISE_NECESSITY))}","source":"{'|'.join(sorted(PREMISE_SOURCES))}","grounded_constraints":["..."],"context_quote":""}}],"operations":[{{"operation_id":"O01","operation_type":"date_difference","input_premise_ids":["R01"],"output_ref":"TARGET","parameters":{{}}}}]}}. The task_contract operations array is optional; all other shown root, target, output, and premise fields are required. Do not emit risk_signals because the runtime computes structural risks locally. A memory premise must use the same premise_id as its corresponding legacy requirement_id. Memory grounded_constraints must be explicit facts or preferences supported by that premise's bound evidence, never an invented recommendation.
For recommendation or advice requests, set output_origin to memory_conditioned_generation and bind memory requirements to preference or constraint premises with grounded_constraints. The contract describes how memory constrains a new generation, not a ready-made recommendation. Do not mark such a requirement missing merely because the reservoir contains no historical recommendation or advice.
When present, typed_semantics must be exactly {{"observations":[],"proposals":[]}}. Each observation uses the module schema fields observation_id, evidence_ids, entity_key, value_kind, value, unit, temporal_kind, and polarity; allowed value_kind values are {', '.join(sorted(VALUE_KINDS))}, temporal_kind values are {', '.join(sorted(TEMPORAL_KINDS))}, and polarity values are {', '.join(sorted(POLARITIES))}. Every typed evidence_ids entry must be an evidence ID from the supplied reservoir, never QUESTION or an invented ID. Event observations require event_status in {{{', '.join(sorted(EVENT_STATUSES))}}}; every event or entity input to count_distinct must carry one of those explicit statuses. A proposal contains candidate_id and a non-empty operations array. Each operation contains operation_id, one typed operation name ({', '.join(sorted(TYPED_OPERATIONS))}), explicit input_ids (or one supported alias), and parameters when needed; inputs may reference only prior observations or prior operation results. Use typed operations for latest state selection, aggregate/count, temporal/date arithmetic, and numeric arithmetic whenever applicable. Typed proposals are advisory and never establish absence.
Do not answer the question and do not rewrite source evidence."""

REVIEW_INSTRUCTION = """A review_context is supplied. Audit the initial plan against every lexical anchor, source atom, slow_context, and fast_context or override. Correct omitted event evidence, wrong temporal anchors, related-entity substitution, incomplete count/list inventories, ignored durable Slow claims, and missed newer Fast overrides. Review both optional structures: check task_contract fields, premise_id alignment with legacy requirement_id, grounded memory constraints, and structural risks including aggregate without typed inventory, temporal reasoning without an operation, multiple states without latest, memory-conditioned generation without grounded constraints, and planner-missing with a plausible source. For memory-conditioned recommendation/advice, distinguish missing preferences or constraints from the absence of a historical recommendation; do not mark the requirement missing solely for the latter. Return one complete replacement plan in the original schema. Preserve a genuinely missing requirement when the reservoir only contains a related but different entity."""


class EvidencePlannerError(RuntimeError):
    def __init__(self, message: str, *, metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = dict(metadata or {})


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _typed_evidence_ids(
    raw: Any,
    allowed: set[str],
    path: str,
    warnings: list[str],
    *,
    allow_empty: bool = False,
) -> list[str] | None:
    if not isinstance(raw, list) or (not raw and not allow_empty):
        warnings.append(f"{path}:quarantined_invalid_evidence_ids")
        return None
    output: list[str] = []
    invalid = False
    for item in raw:
        evidence_id = _text(item)
        if not evidence_id or evidence_id not in allowed:
            invalid = True
            warnings.append(f"{path}:dropped_unknown_id")
            continue
        if evidence_id not in output:
            output.append(evidence_id)
    if invalid or (not output and not allow_empty):
        warnings.append(f"{path}:quarantined_unknown_evidence_id")
        return None
    return output


def _normalize_typed_observations(
    raw: Any,
    *,
    evidence_ids: set[str],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(raw, list):
        warnings.append("typed_semantics.observations:quarantined_invalid_array")
        return [], {}
    observations: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    required = {
        "observation_id",
        "evidence_ids",
        "entity_key",
        "value_kind",
        "value",
        "unit",
        "temporal_kind",
        "polarity",
    }
    for index, item in enumerate(raw):
        path = f"typed_semantics.observations[{index}]"
        if not isinstance(item, Mapping):
            warnings.append(f"{path}:quarantined_invalid_observation")
            continue
        current = dict(item)
        if not required.issubset(current):
            warnings.append(f"{path}:quarantined_missing_typed_fields")
            continue
        observation_id = _text(current.get("observation_id"))
        if not observation_id or observation_id in by_id:
            warnings.append(f"{path}:quarantined_invalid_observation_id")
            continue
        typed_ids = _typed_evidence_ids(current.get("evidence_ids"), evidence_ids, f"{path}.evidence_ids", warnings)
        if typed_ids is None:
            warnings.append(f"{path}:quarantined_invalid_evidence")
            continue
        if _text(current.get("entity_key")) == "":
            warnings.append(f"{path}:quarantined_invalid_entity_key")
            continue
        if any(
            not isinstance(current.get(field), str) or current[field] not in allowed
            for field, allowed in (
                ("value_kind", VALUE_KINDS),
                ("temporal_kind", TEMPORAL_KINDS),
                ("polarity", POLARITIES),
            )
        ):
            warnings.append(f"{path}:quarantined_invalid_typed_enum")
            continue
        unit = current.get("unit")
        if unit is not None and (not isinstance(unit, str) or not unit.strip()):
            warnings.append(f"{path}:quarantined_invalid_unit")
            continue
        status_fields = [field for field in ("event_status", "event_state", "status") if field in current]
        statuses = [current[field] for field in status_fields]
        if statuses and any(status != statuses[0] for status in statuses[1:]):
            warnings.append(f"{path}:quarantined_conflicting_event_status")
            continue
        status = statuses[0].strip().lower() if statuses and isinstance(statuses[0], str) else None
        if statuses and status not in EVENT_STATUSES:
            warnings.append(f"{path}:quarantined_invalid_event_status")
            continue
        if current.get("value_kind") == "event" and not statuses:
            warnings.append(f"{path}:quarantined_missing_event_status")
            continue
        current["evidence_ids"] = typed_ids
        if status is not None:
            current["event_status"] = status
        observations.append(current)
        by_id[observation_id] = current
    return observations, by_id


def _normalize_typed_proposals(
    raw: Any,
    *,
    evidence_ids: set[str],
    observations: Mapping[str, Mapping[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        warnings.append("typed_semantics.proposals:quarantined_invalid_array")
        return []
    proposals: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for index, item in enumerate(raw):
        path = f"typed_semantics.proposals[{index}]"
        if not isinstance(item, Mapping):
            warnings.append(f"{path}:quarantined_invalid_proposal")
            continue
        current = dict(item)
        candidate_id = _text(current.get("candidate_id"))
        operations = current.get("operations")
        if (
            not candidate_id
            or candidate_id in candidate_ids
            or not isinstance(operations, list)
            or not operations
        ):
            warnings.append(f"{path}:quarantined_invalid_proposal_shape")
            continue
        candidate_ids.add(candidate_id)

        source_fields = [field for field in ("source_evidence_ids", "evidence_ids") if field in current]
        if len(source_fields) == 2 and current[source_fields[0]] != current[source_fields[1]]:
            warnings.append(f"{path}:quarantined_conflicting_source_evidence_ids")
            continue
        invalid = False
        if source_fields:
            source_ids = _typed_evidence_ids(
                current[source_fields[0]],
                evidence_ids,
                f"{path}.{source_fields[0]}",
                warnings,
                allow_empty=True,
            )
            if source_ids is None:
                invalid = True
            else:
                for field in source_fields:
                    current[field] = list(source_ids)
        if invalid:
            warnings.append(f"{path}:quarantined_invalid_evidence")
            continue

        known_refs = set(observations)
        operation_ids: set[str] = set()
        for operation_index, raw_operation in enumerate(operations):
            operation_path = f"{path}.operations[{operation_index}]"
            if not isinstance(raw_operation, Mapping):
                invalid = True
                warnings.append(f"{operation_path}:quarantined_invalid_operation")
                break
            operation = dict(raw_operation)
            operation_id = _text(operation.get("operation_id"))
            if not operation_id or operation_id in operation_ids:
                invalid = True
                warnings.append(f"{operation_path}:quarantined_invalid_operation_id")
                break
            operation_ids.add(operation_id)
            operation_names = [operation[field] for field in ("operation", "operation_type", "op") if field in operation]
            if not operation_names or not isinstance(operation_names[0], str) or operation_names[0] not in TYPED_OPERATIONS or any(name != operation_names[0] for name in operation_names[1:]):
                invalid = True
                warnings.append(f"{operation_path}:quarantined_invalid_operation_type")
                break
            input_fields = [field for field in ("input_ids", "input_observation_ids", "observation_ids", "operands", "inputs") if field in operation]
            if not input_fields or any(operation[field] != operation[input_fields[0]] for field in input_fields[1:]):
                invalid = True
                warnings.append(f"{operation_path}:quarantined_invalid_input_reference")
                break
            inputs = operation[input_fields[0]]
            if not isinstance(inputs, list) or not inputs or any(not isinstance(ref, str) or not ref.strip() or ref not in known_refs for ref in inputs):
                invalid = True
                warnings.append(f"{operation_path}:quarantined_invalid_input_reference")
                break
            if operation_names[0] == "count_distinct":
                if any(ref not in observations or observations[ref].get("value_kind") not in {"event", "entity_instance"} or observations[ref].get("event_status") not in EVENT_STATUSES for ref in inputs):
                    invalid = True
                    warnings.append(f"{operation_path}:quarantined_missing_count_event_status")
                    break
            parameters = operation.get("parameters", {})
            if not isinstance(parameters, Mapping):
                invalid = True
                warnings.append(f"{operation_path}:quarantined_invalid_parameters")
                break
            known_refs.add(operation_id)
        if invalid:
            warnings.append(f"{path}:quarantined_invalid_reference")
            continue
        output_fields = [field for field in ("output_ref", "output_operation_id") if field in current]
        if len(output_fields) == 2 and current[output_fields[0]] != current[output_fields[1]]:
            warnings.append(f"{path}:quarantined_conflicting_output_reference")
            continue
        if output_fields and current[output_fields[0]] not in operation_ids:
            warnings.append(f"{path}:quarantined_invalid_output_reference")
            continue
        current["operations"] = [dict(operation) for operation in operations]
        proposals.append(current)
    return proposals


def _normalize_typed_semantics(value: Any, evidence_ids: set[str], warnings: list[str]) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, Mapping):
        warnings.append("typed_semantics:quarantined_invalid_shape")
        return {"observations": [], "proposals": []}
    if set(value) != {"observations", "proposals"}:
        warnings.append("typed_semantics:normalized_to_exact_shape")
    observations, by_id = _normalize_typed_observations(value.get("observations"), evidence_ids=evidence_ids, warnings=warnings)
    proposals = _normalize_typed_proposals(value.get("proposals"), evidence_ids=evidence_ids, observations=by_id, warnings=warnings)
    return {"observations": observations, "proposals": proposals}


def _normalize_plan_ids(value: Mapping[str, Any], catalog: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Fail closed on invented IDs while tolerating harmless list defects."""
    evidence_ids = {_text(item.get("evidence_id")) for item in catalog.get("evidence") or []}
    atom_ids = {_text(item.get("atom_id")) for item in catalog.get("atoms") or []}
    normalized = dict(value)
    warnings: list[str] = []

    def clean_ids(raw: Any, allowed: set[str], path: str) -> tuple[list[str], bool]:
        if isinstance(raw, str):
            warnings.append(f"{path}:coerced_scalar_id")
            raw = [raw]
        if not isinstance(raw, list):
            return raw, False
        output: list[str] = []
        invalid = False
        for item in raw:
            item_id = _text(item)
            if not item_id or item_id not in allowed:
                invalid = True
                warnings.append(f"{path}:dropped_unknown_id")
                continue
            if item_id in output:
                warnings.append(f"{path}:deduplicated_id")
                continue
            output.append(item_id)
        return output, invalid

    requirements = []
    for index, item in enumerate(value.get("requirements") or []):
        current = dict(item) if isinstance(item, Mapping) else item
        if isinstance(current, dict):
            current["evidence_ids"], _ = clean_ids(current.get("evidence_ids"), evidence_ids, f"requirements[{index}].evidence_ids")
        requirements.append(current)
    normalized["requirements"] = requirements

    operations = []
    for index, item in enumerate(value.get("operations") or []):
        current = dict(item) if isinstance(item, Mapping) else item
        if not isinstance(current, dict):
            operations.append(current)
            continue
        if _text(current.get("operation_type")) not in OPERATION_TYPES:
            warnings.append(
                f"operations[{index}]:quarantined_unsupported_legacy_operation"
            )
            continue
        current["input_atom_ids"], bad_atoms = clean_ids(current.get("input_atom_ids"), atom_ids, f"operations[{index}].input_atom_ids")
        current["input_evidence_ids"], bad_evidence = clean_ids(current.get("input_evidence_ids"), evidence_ids, f"operations[{index}].input_evidence_ids")
        if bad_atoms or bad_evidence or not current["input_atom_ids"] or not current["input_evidence_ids"]:
            warnings.append(f"operations[{index}]:quarantined_invalid_operands")
            continue
        operations.append(current)
    normalized["operations"] = operations

    bundles = []
    for index, item in enumerate(value.get("bundles") or []):
        current = dict(item) if isinstance(item, Mapping) else item
        if isinstance(current, dict):
            current["evidence_ids"], _ = clean_ids(current.get("evidence_ids"), evidence_ids, f"bundles[{index}].evidence_ids")
            if not current["evidence_ids"]:
                warnings.append(f"bundles[{index}]:quarantined_empty_bundle")
                continue
        bundles.append(current)
    normalized["bundles"] = bundles
    if "typed_semantics" in value:
        normalized["typed_semantics"] = _normalize_typed_semantics(value.get("typed_semantics"), evidence_ids, warnings)
    return normalized, warnings


def normalize_planner_output(
    value: Mapping[str, Any], catalog: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Normalize harmless model-shape variance and enforce the production plan."""
    if not isinstance(value, Mapping):
        raise ValueError("planner output must be an object")
    normalized_plan, warnings = _normalize_plan_ids(value, catalog)
    if "task_contract" not in normalized_plan or "typed_semantics" not in normalized_plan:
        raise ValueError("new planner output requires task_contract and typed_semantics")
    return validate_operation_plan(normalized_plan, catalog), warnings


def planner_payload(catalog: Mapping[str, Any]) -> dict[str, Any]:
    evidence = [
        {
            "evidence_id": item["evidence_id"],
            "session_id": item["session_id"],
            "session_index": item["session_index"],
            "parent_chunk_index": item["parent_chunk_index"],
            "text": item["text"],
            "question_overlap_terms": item.get("question_overlap_terms") or [],
            "question_overlap_score": int(item.get("question_overlap_score", 0)),
            "source_group_id": item.get("source_group_id"),
            "source_group_context": list(item.get("source_group_context") or []),
            "memory_contexts": list(item.get("memory_contexts") or []),
            "attachments": list(item.get("attachments") or []),
        }
        for item in sorted(
            catalog.get("evidence") or [],
            key=lambda candidate: (
                int(candidate.get("session_index", 0)),
                int(candidate.get("parent_chunk_index", 0)),
                int(str(candidate.get("evidence_id"))[1:]),
            ),
        )
    ]
    atoms = [
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
    ]
    return {
        "question": catalog.get("question"),
        "question_date": catalog.get("question_date") or "unknown",
        "lexical_anchor_ids": list(catalog.get("lexical_anchor_ids") or []),
        "evidence": evidence,
        "atoms": atoms,
    }


def _usage(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise EvidencePlannerError("planner response lacks usage")

    def count(name: str, *aliases: str) -> int:
        raw = next((value.get(key) for key in (name, *aliases) if value.get(key) is not None), 0)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw < 0:
            raise EvidencePlannerError(f"usage.{name} is invalid")
        return int(raw)

    prompt = count("prompt_tokens", "input_tokens")
    completion = count("completion_tokens", "output_tokens")
    hit = count("prompt_cache_hit_tokens", "cache_read_input_tokens", "cached_tokens")
    miss_present = any(value.get(key) is not None for key in ("prompt_cache_miss_tokens", "cache_miss_input_tokens"))
    miss = count("prompt_cache_miss_tokens", "cache_miss_input_tokens")
    if hit > prompt or (miss_present and hit + miss != prompt):
        raise EvidencePlannerError("planner cache usage is inconsistent")
    if not miss_present:
        miss = prompt - hit
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "prompt_cache_hit_tokens": hit,
        "prompt_cache_miss_tokens": miss,
        "total_tokens": count("total_tokens") or prompt + completion,
    }


class DeepSeekEvidenceOperationPlanner:
    def __init__(self, *, base_url: str, api_keys: Sequence[str], timeout: float = 180.0, max_tokens: int = 8192, model: str = MODEL, provider: str = "deepseek") -> None:
        self.base_url = _text(base_url).rstrip("/")
        self.api_keys = list(dict.fromkeys(_text(key) for key in api_keys if _text(key)))
        self.timeout = float(timeout)
        self.max_tokens = int(max_tokens)
        self.model = _text(model)
        self.provider = _text(provider)
        self.call_index = 0
        if not self.base_url or not self.api_keys or not self.model or self.provider not in {"deepseek", "xiaomi_mimo"} or self.timeout <= 0 or self.max_tokens <= 0:
            raise EvidencePlannerError("planner base URL, key pool, timeout, and max tokens are required")

    def plan(self, catalog: Mapping[str, Any], *, review_context: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = planner_payload(catalog)
        if review_context is not None:
            payload["review_context"] = dict(review_context)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + ("\n\n" + REVIEW_INSTRUCTION if review_context is not None else "")},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            **({"thinking": {"type": "disabled"}, "enable_thinking": False} if self.provider == "deepseek" else {"stream": False}),
        }
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request_sha256 = hashlib.sha256(encoded).hexdigest()
        key_index = self.call_index % len(self.api_keys)
        self.call_index += 1
        call_id = "eop_" + uuid.uuid4().hex
        started = time.time()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_keys[key_index]}"},
            method="POST",
        )
        base_metadata = {
            "physical_call_id": call_id,
            "physical_api_call": True,
            "physical_api_calls": 1,
            "stage": "evidence_operation_planner",
            "model": self.model,
            "provider": self.provider,
            "prompt_version": PROMPT_VERSION,
            "review_call": review_context is not None,
            "api_key_index": key_index,
            "request_sha256": request_sha256,
        }
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = int(response.getcode())
                raw_http = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise EvidencePlannerError(
                f"planner HTTP {exc.code}: {detail}",
                metadata={**base_metadata, "status": "http_error", "http_status": int(exc.code), "latency_seconds": round(time.time() - started, 3)},
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EvidencePlannerError(
                f"planner request failed: {exc}",
                metadata={**base_metadata, "status": "request_error", "latency_seconds": round(time.time() - started, 3)},
            ) from exc
        try:
            response_body = json.loads(raw_http)
            choice = response_body["choices"][0]
            content = choice["message"]["content"]
            finish_reason = _text(choice.get("finish_reason"))
            raw_plan = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise EvidencePlannerError(
                "planner returned malformed JSON",
                metadata={**base_metadata, "status": "invalid_response", "http_status": status, "latency_seconds": round(time.time() - started, 3)},
            ) from exc
        usage = _usage(response_body.get("usage"))
        metadata = {
            **base_metadata,
            "status": "completed",
            "http_status": status,
            "latency_seconds": round(time.time() - started, 3),
            "finish_reason": finish_reason,
            "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "usage": usage,
        }
        if status != 200 or finish_reason != "stop" or not isinstance(raw_plan, Mapping):
            raise EvidencePlannerError("planner did not return one complete JSON object", metadata=metadata)
        try:
            plan, warnings = normalize_planner_output(raw_plan, catalog)
        except Exception as exc:
            raise EvidencePlannerError(
                f"planner returned an invalid operation plan: {exc}",
                metadata={
                    **metadata,
                    "normalization_warnings": warnings if "warnings" in locals() else [],
                    "raw_plan": dict(raw_plan),
                },
            ) from exc
        metadata["normalization_warnings"] = warnings
        return plan, metadata
