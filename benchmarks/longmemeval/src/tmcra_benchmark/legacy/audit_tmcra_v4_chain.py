"""Strict post-run audit for the TMCRA V4 writer/slow/retrieval chain.

The audit deliberately reads SQLite and JSON artifacts instead of importing the
controllers.  This keeps it useful after a process has failed and lets it fail
closed when a future V4 schema is not recognizable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


KNOWN_DBS = {"native_memory.sqlite3", "memory.sqlite3", "writer.sqlite3", "graph.sqlite3"}
SOURCE_TABLES = ("records",)
REQUIRED_GRAPH_TABLES = {"records", "memory_edges", "slot_heads"}
FORBIDDEN_SHADOW_TABLES = {"v4_source_records", "v4_fast_assertion_leaves"}
FORBIDDEN_INPUT_KEYS = {
    "answer", "gold_answer", "expected_answer", "answer_session_ids", "labels",
    "label", "supervision", "judge", "judge_output", "benchmark", "gold",
}
FORBIDDEN_ARTIFACT_KEY_PARTS = ("fallback", "retry", "alternate_model", "silent_repair")
OPERATIONAL_MARKER_VALUE_PATH_PARTS = (
    "route",
    "reason",
    "status",
    "error",
    "policy",
    "mode",
    "strategy",
    "recovery",
    "handler",
    "model",
)
SEMANTIC_IDENTIFIER_PATH_PARTS = (
    "evidence_id",
    "memory_id",
    "source_record_id",
    "capsule_id",
    "claim_id",
    "interaction_id",
    "region_key",
)
FORBIDDEN_RETRIEVAL_KEY_PARTS = ("hard_negative", "hard_mode", "gold", "label", "supervision", "judge")
MARKER_RE = re.compile(r"\b(?:fallback|retry|retryable|alternate\s+model|silent\s+repair)\b", re.I)
CONFLICT_RE = re.compile(r"same[_ -]?slot|counterevidence|unresolved[_ -]?challenge|conflict", re.I)
SLOW_OPERATIONAL_SUMMARY_RE = re.compile(
    r"^\s*deterministic\s+(?:create|revise|retire|cleanup|migration)\b|"
    r"^\s*(?:initial\s+)?consolidat(?:e|ed|es|ing|ion)\b[^.]{0,120}\b(?:evidence|claims?|facts?|preferences?|routines?|memory)\b|"
    r"^\s*(?:add|added|adding|challenge|challenged|challenging)\b[^.]{0,80}\bevidence\b|"
    r"^\s*add\b|"
    r"\b(?:supplied|required|current\s+fast)\s+evidence\b|"
    r"^\s*create\s+(?:an?\s+|the\s+)?(?:initial\s+)?[^.]{0,80}\b(?:memory|claims?|record)\b|"
    r"^\s*(?:memory|record)\s+(?:revision|cleanup|migration)\b|"
    r"^\s*(?:创建|新增|更新|合并|整理).{0,24}(?:记忆|胶囊|证据|声明|记录)|"
    r"^\s*(?:控制器|确定性创建|确定性修订).{0,24}(?:记忆|证据|声明|记录|胶囊)|"
    r"(?:快速图证据)",
    re.I,
)
SLOW_GENERIC_SUMMARY_RE = re.compile(
    r"^user(?:'s)?\s+(?:fitness\s+)?(?:goals?\s+and\s+facts?|commute\s+details?|"
    r"schedule\s+routine|preferences?\s+and\s+facts?)(?:\.)?$|"
    r"^user\s+(?:has\s+)?social\s+media\s+goals?\s+and\s+routines?(?:\.)?$|"
    r"^user\s+opinions?\s+(?:on|about)\s+[^.]+(?:\.)?$|"
    r"^user\s+preferences?\s+and\s+facts?\s+about\s+[^.]+(?:\.)?$|"
    r"^user(?:'s)?\s+[^.]+\s+preferences?\s+and\s+wearing\s+frequency(?:\.)?$",
    re.I,
)
CURRENT_FAST_INDEX_POLICY = (
    "current-fast-states-v1:active,challenged,parallel_active,promoted"
)
SOURCE_COVERAGE_TRACE_K = 24
RETRIEVAL_SCHEMA_VERSION = "tmcra.v4.online-retrieval.6"
RETRIEVAL_REPORT_SCHEMA_VERSION = "tmcra.v4.online-retrieval-report.6"
RETRIEVAL_CONTRACT_SCHEMA_VERSION = "tmcra.v4.layered-retrieval-contract.3"
ONLINE_INDEX_SCHEMA_VERSION = "tmcra.v4.online-index.3"
SLOW_INVENTORY_SCHEMA_VERSION = "tmcra.v4.slow-inventory.1"
SLOW_SUMMARY_CONTRACT_VERSION = "tmcra.v4.slow-lossless-summary.2"
CURRENT_SLOW_PROMPT_VERSION = "tmcra-v4-slow-graph-2026-07-14.16"
CURRENT_SLOW_PARTITION_CONTRACT_VERSION = "tmcra.v4.slow-semantic-partition.2"
SESSION_ORDERING_POLICY = "session_rrf_then_chronological_v1"
KEEP_PARALLEL_MIGRATION_SCHEMA_VERSION = "tmcra.v4.keep-parallel-migration.1"
KEEP_PARALLEL_MIGRATION_VERSION = "keep-parallel-authority-2026-07-12.1"
CHALLENGE_MIGRATION_SCHEMA_VERSION = "tmcra.v4.challenge-lifecycle-migration.1"
CHALLENGE_MIGRATION_VERSION = "challenge-lifecycle-authority-2026-07-15.1"
ACTIVE_RECORD_STATES = {"active", "parallel_active", "promoted"}
CURRENT_FAST_RECORD_STATES = {
    "active",
    "parallel_active",
    "promoted",
    "challenged",
}
PRODUCTION_FAST_NODE_KIND = "atomic_user_assertion"
PRODUCTION_FAST_VARIANT = "product_semantic_memory"
SUBJECT_ATTRIBUTION_PROMPT_VERSION = "tmcra-v4-subject-attribution-2026-07-14.3"
FAST_SOURCE_PARENT_KEYS = {
    "session_index",
    "parent_chunk_index",
    "source_record_id",
    "evidence_char_start",
    "evidence_char_end",
}
SLOW_SOURCE_PARENT_KEYS = {
    *FAST_SOURCE_PARENT_KEYS,
    "message_index",
    "event_id",
}

Identity = tuple[str, str, str]
SourceMessageKey = tuple[str, str, str]
SourceScopeMessageKey = tuple[str, str]
VALID_SUPERSESSION_LIFECYCLE_REASONS = {
    "same_state_revision",
    "slot_disallows_parallel",
    "v4_reconciliation_replace_current",
}


class AuditError(RuntimeError):
    pass


def _json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None or value == "":
        return {}
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid JSON artifact value: {exc}") from exc


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _normal_text(value: Any) -> str:
    return " ".join(_text(value).casefold().split())


def _leaf_metadata(leaf: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = leaf.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _leaf_slot(leaf: Mapping[str, Any]) -> str:
    metadata = _leaf_metadata(leaf)
    return _text(
        metadata.get("canonical_slot_key") or metadata.get("canonical_slot")
    )


def _leaf_text(leaf: Mapping[str, Any]) -> str:
    metadata = _leaf_metadata(leaf)
    return _text(
        leaf.get("value") or metadata.get("source_span") or metadata.get("raw_content")
    )


def _leaf_state(leaf: Mapping[str, Any]) -> str:
    metadata = _leaf_metadata(leaf)
    return _text(
        leaf.get("record_state")
        or leaf.get("state")
        or metadata.get("record_state")
    ).casefold()


def _is_current_durable_leaf(leaf: Mapping[str, Any]) -> bool:
    metadata = _leaf_metadata(leaf)
    durability = _text(
        metadata.get("durability") or metadata.get("durability_class")
    ).casefold()
    return durability in {
        "durable",
        "long_term",
        "long-term",
        "hard",
        "persistent",
    } and _leaf_state(leaf) in {"active", "parallel_active", "promoted"}


def _is_counterevidence_leaf(leaf: Mapping[str, Any]) -> bool:
    metadata = _leaf_metadata(leaf)
    return bool(metadata.get("counterevidence")) or bool(
        metadata.get("is_counterevidence")
    )


def _controlled_complementary_support_bundle(
    claim_slot: str,
    evidence_ids: Iterable[Identity],
    evidence_by_id: Mapping[Identity, Mapping[str, Any]],
) -> bool:
    """Recognize one durable parent fact plus compatible subslot refinements."""
    unique_ids = sorted(set(evidence_ids))
    if len(unique_ids) < 2:
        return False
    leaves = [evidence_by_id.get(evidence_id) for evidence_id in unique_ids]
    if any(
        leaf is None
        or not _is_current_durable_leaf(leaf)
        or _is_counterevidence_leaf(leaf)
        for leaf in leaves
    ):
        return False
    typed_leaves = [leaf for leaf in leaves if leaf is not None]
    slots = [_leaf_slot(leaf) for leaf in typed_leaves]
    if claim_slot not in slots or len(set(slots)) < 2:
        return False
    if any(
        slot != claim_slot and not slot.startswith(claim_slot + ".")
        for slot in slots
    ):
        return False

    texts_by_slot: dict[str, set[str]] = {}
    for leaf in typed_leaves:
        texts_by_slot.setdefault(_leaf_slot(leaf), set()).add(
            _normal_text(_leaf_text(leaf))
        )
    if any(len(texts) != 1 or "" in texts for texts in texts_by_slot.values()):
        return False

    def one_shared_metadata_value(key: str) -> bool:
        values = {
            _text(_leaf_metadata(leaf).get(key)).casefold()
            for leaf in typed_leaves
        }
        return len(values) == 1 and "" not in values

    if not all(
        one_shared_metadata_value(key)
        for key in ("subject_signature", "graph_entity_key", "memory_family")
    ):
        return False
    relations = {
        _text(
            leaf.get("relation") or _leaf_metadata(leaf).get("semantic_slot")
        ).casefold()
        for leaf in typed_leaves
    }
    polarities = {
        _text(_leaf_metadata(leaf).get("polarity")).casefold()
        for leaf in typed_leaves
    }
    return (
        len(relations) == 1
        and "" not in relations
        and len(polarities) == 1
        and "" not in polarities
    )


def _lossless_summary_projection(claims: Any) -> str:
    if not isinstance(claims, list) or not claims:
        raise AuditError("lossless Slow summary projection requires claims")
    texts: list[str] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            raise AuditError(f"lossless Slow summary claim {index} is not an object")
        text = " ".join(_text(claim.get("text")).split())
        if not text:
            raise AuditError(f"lossless Slow summary claim {index} lacks text")
        texts.append(text)
    return " ".join(texts)


def _current_summary_contract(
    record_metadata: Mapping[str, Any],
    patch_metadata: Mapping[str, Any] | None = None,
) -> bool:
    for value in (record_metadata, patch_metadata or {}):
        if not isinstance(value, Mapping):
            continue
        if (
            value.get("summary_contract_version") == SLOW_SUMMARY_CONTRACT_VERSION
            or value.get("prompt_version") == CURRENT_SLOW_PROMPT_VERSION
            or value.get("partition_contract_version")
            == "tmcra.v4.slow-semantic-partition.1"
        ):
            return True
        provenance = value.get("provenance")
        if isinstance(provenance, Mapping) and _current_summary_contract(provenance):
            return True
    return False


def _current_v47_artifact(value: Any) -> bool:
    """Return whether persisted metadata explicitly identifies current V4.7."""
    if isinstance(value, Mapping):
        if (
            value.get("prompt_version") == CURRENT_SLOW_PROMPT_VERSION
            or value.get("summary_contract_version") == SLOW_SUMMARY_CONTRACT_VERSION
            or value.get("partition_contract_version")
            == CURRENT_SLOW_PARTITION_CONTRACT_VERSION
        ):
            return True
        return any(_current_v47_artifact(item) for item in value.values())
    if isinstance(value, list):
        return any(_current_v47_artifact(item) for item in value)
    return False


def _current_v47_capsule_id(
    scope_id: Any, region_key: Any, capsule_key: Any
) -> str:
    """Mirror the V4.7 deterministic create-target mapping without importing runtime code."""
    normalized_key = _text(capsule_key).casefold()
    payload = {
        "scope_id": _text(scope_id),
        "region_key": _text(region_key),
        "capsule_key": normalized_key,
        "partition_contract_version": CURRENT_SLOW_PARTITION_CONTRACT_VERSION,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "cap_" + _sha(encoded)[:24]


def _audit_current_v47_patch_reconciliation(
    *,
    jobs: Sequence[tuple[Path, Mapping[str, Any]]],
    patches: Sequence[tuple[Path, Mapping[str, Any]]],
    operation_rows: Sequence[tuple[Path, Mapping[str, Any]]],
    slow_records: Sequence[Mapping[str, Any]],
    issues: list[str],
) -> None:
    """Reconcile current V4.7 patch envelopes, operation rows, and result records."""
    job_by_key = {
        (_db_path(path), _text(row.get("job_id"))): row
        for path, row in jobs
        if _text(row.get("job_id"))
    }
    patch_by_key = {
        (_db_path(path), _text(row.get("patch_id"))): row
        for path, row in patches
        if _text(row.get("patch_id"))
    }
    patches_by_job: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for path, row in patches:
        patch_id = _text(row.get("patch_id"))
        job_id = _text(row.get("job_id"))
        if patch_id and job_id:
            patches_by_job[(_db_path(path), job_id)].append(row)

    operations_by_patch: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for path, row in operation_rows:
        patch_id = _text(row.get("patch_id"))
        patch_key = (_db_path(path), patch_id)
        if patch_key not in patch_by_key:
            issues.append(
                f"{path}: orphan slow graph patch operation {_text(row.get('operation_id'))}"
            )
        else:
            operations_by_patch[patch_key].append(row)

    result_records: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    current_record_patch_ids: set[tuple[str, str]] = set()
    for record in slow_records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        patch_id = _text(metadata.get("patch_id"))
        capsule_id = _text(metadata.get("capsule_id"))
        if not patch_id or not capsule_id:
            continue
        revision_value = metadata.get("revision")
        if isinstance(revision_value, bool) or not isinstance(revision_value, int):
            continue
        revision = revision_value
        key = (_db_path(record.get("db")), patch_id)
        if _current_v47_artifact(metadata):
            current_record_patch_ids.add(key)
        result_records[(*key, capsule_id, revision)].append(record)

    current_jobs: set[tuple[str, str]] = set()
    for path, row in jobs:
        job_key = (_db_path(path), _text(row.get("job_id")))
        if job_key[1] and _current_v47_artifact(_json(row.get("metadata_json"))):
            current_jobs.add(job_key)

    current_patches: set[tuple[str, str]] = set(current_record_patch_ids)
    for path, row in patches:
        patch_key = (_db_path(path), _text(row.get("patch_id")))
        if not patch_key[1]:
            continue
        if _current_v47_artifact(_json(row.get("call_metadata_json"))):
            current_patches.add(patch_key)
        job_key = (_db_path(path), _text(row.get("job_id")))
        if job_key in current_jobs:
            current_patches.add(patch_key)

    for job_key in sorted(current_jobs):
        matching = patches_by_job.get(job_key, [])
        if len(matching) != 1:
            issues.append(
                f"{job_key[0]}: current V4.7 job {job_key[1]} has {len(matching)} patches; expected exactly one"
            )

    for patch_key in sorted(current_patches):
        patch = patch_by_key.get(patch_key)
        if patch is None:
            issues.append(f"{patch_key[0]}: current V4.7 result references missing patch {patch_key[1]}")
            continue
        job_id = _text(patch.get("job_id"))
        job_key = (patch_key[0], job_id)
        job = job_by_key.get(job_key)
        matching = patches_by_job.get(job_key, [])
        if job is None:
            issues.append(
                f"{patch_key[0]}: current V4.7 patch {patch_key[1]} references missing job {job_id}"
            )
        if len(matching) != 1:
            issues.append(
                f"{patch_key[0]}: current V4.7 patch {patch_key[1]} does not have exactly one patch for job {job_id}"
            )
        if job is not None and _text(patch.get("scope_id")) != _text(job.get("scope_id")):
            issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} scope differs from its job")
        if job is not None and _text(patch.get("region_key")) != _text(job.get("region_key")):
            issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} region differs from its job")

        try:
            patch_json = _json(patch.get("patch_json"))
        except AuditError as exc:
            issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} has invalid patch_json: {exc}")
            continue
        if not isinstance(patch_json, Mapping) or not isinstance(patch_json.get("operations"), list):
            issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operations are missing")
            continue
        operations = patch_json["operations"]
        rows = list(operations_by_patch.get(patch_key, ()))
        ordinals: list[int | None] = []
        for row in rows:
            try:
                ordinals.append(int(row.get("ordinal")))
            except (TypeError, ValueError):
                ordinals.append(None)
        duplicate_ordinals = sorted(
            ordinal for ordinal in set(item for item in ordinals if item is not None)
            if ordinals.count(ordinal) > 1
        )
        if duplicate_ordinals:
            issues.append(
                f"{patch_key[0]}: current V4.7 patch {patch_key[1]} has duplicate operation ordinals {duplicate_ordinals}"
            )
        operation_ids = [_text(row.get("operation_id")) for row in rows]
        duplicate_operation_ids = sorted(
            operation_id for operation_id in set(operation_ids) if operation_ids.count(operation_id) > 1
        )
        if duplicate_operation_ids:
            issues.append(
                f"{patch_key[0]}: current V4.7 patch {patch_key[1]} has duplicate operation rows {duplicate_operation_ids}"
            )
        def operation_row_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
            try:
                return (0, int(row.get("ordinal")), _text(row.get("operation_id")))
            except (TypeError, ValueError):
                return (1, 0, _text(row.get("operation_id")))

        rows.sort(key=operation_row_sort_key)
        if len(rows) != len(operations):
            issues.append(
                f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operation row count {len(rows)} differs from patch_json count {len(operations)}"
            )
        sorted_ordinals: list[int | None] = []
        for row in rows:
            try:
                sorted_ordinals.append(int(row.get("ordinal")))
            except (TypeError, ValueError):
                sorted_ordinals.append(None)
        if (
            any(ordinal is None for ordinal in sorted_ordinals)
            or len(rows) != len(operations)
            or sorted_ordinals != list(range(len(operations)))
        ):
            issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operation ordinals are not exact")

        represented_results: set[tuple[str, int]] = set()
        for ordinal, operation in enumerate(operations):
            if not isinstance(operation, Mapping):
                issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operation {ordinal} is not an object")
                continue
            if ordinal >= len(rows):
                continue
            row = rows[ordinal]
            try:
                stored_operation = _json(row.get("operation_json"))
            except AuditError as exc:
                issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operation {ordinal} has invalid operation_json: {exc}")
                stored_operation = None
            if stored_operation != operation:
                issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operation {ordinal} differs from patch_json")
            action = _text(operation.get("action"))
            if _text(row.get("action")) != action:
                issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operation {ordinal} action differs")
            row_capsule_id = _text(row.get("capsule_id"))
            region_key = _text(patch.get("region_key"))
            if action == "create":
                if not _text(operation.get("capsule_key")):
                    issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operation {ordinal} create capsule key is missing")
                expected_capsule_id = _current_v47_capsule_id(
                    patch.get("scope_id"), region_key, operation.get("capsule_key")
                )
                if operation.get("capsule_id") is not None or row_capsule_id != expected_capsule_id:
                    issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operation {ordinal} create target capsule differs")
                expected_base, expected_result = None, 1
            elif action == "noop":
                expected_target = _text(operation.get("capsule_id")) or "region:" + region_key
                if row_capsule_id != expected_target:
                    issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operation {ordinal} target capsule differs")
                expected_base = row.get("base_revision")
                expected_result = row.get("result_revision")
                if expected_base != expected_result:
                    issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operation {ordinal} noop revisions differ")
            else:
                expected_target = _text(operation.get("capsule_id"))
                if not expected_target or row_capsule_id != expected_target:
                    issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operation {ordinal} target capsule differs")
                expected_base = operation.get("base_revision")
                expected_result = expected_base + 1 if isinstance(expected_base, int) and not isinstance(expected_base, bool) else None
                if row.get("base_revision") != expected_base or row.get("result_revision") != expected_result:
                    issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operation {ordinal} revisions differ")
            if action == "create" and (
                row.get("base_revision") != expected_base or row.get("result_revision") != expected_result
            ):
                issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operation {ordinal} revisions differ")
            if action != "noop" and isinstance(expected_result, int) and row_capsule_id:
                result_key = (row_capsule_id, expected_result)
                represented_results.add(result_key)
                matches = result_records.get((*patch_key, *result_key), [])
                if len(matches) != 1:
                    issues.append(
                        f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operation {ordinal} lacks exactly one resulting record metadata match"
                    )
                elif _text(matches[0]["metadata"].get("action")) not in {"", action}:
                    issues.append(f"{patch_key[0]}: current V4.7 patch {patch_key[1]} operation {ordinal} result action differs")

        for result_key, matches in result_records.items():
            if result_key[:2] != patch_key or result_key[2:] in represented_results:
                continue
            issues.append(
                f"{patch_key[0]}: current V4.7 patch {patch_key[1]} has an unrepresented resulting record metadata row"
            )


def _active_slow_partition_issues(
    slow_records: Sequence[Mapping[str, Any]],
    patch_metadata: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Audit V4.7 partitioning without treating several heads as an error."""
    patch_metadata = patch_metadata or {}
    by_region: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
    issues: list[dict[str, Any]] = []
    for record in slow_records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping) or not _current_summary_contract(
            metadata,
            patch_metadata.get(
                (_db_path(record.get("db")), _text(metadata.get("patch_id"))),
            ),
        ):
            continue
        region_key = _text(metadata.get("region_key"))
        capsule_id = _text(metadata.get("capsule_id"))
        if (
            metadata.get("partition_contract_version")
            != CURRENT_SLOW_PARTITION_CONTRACT_VERSION
        ):
            issues.append(
                {
                    "code": "semantic_partition_migration_required",
                    "db": _db_path(record.get("db")),
                    "scope_id": _text(record.get("scope_id")),
                    "capsule_id": capsule_id,
                    "revision": record.get("slow_revision"),
                    "region_key": region_key,
                    "stored_contract_version": metadata.get(
                        "partition_contract_version"
                    ),
                }
            )
        if not region_key:
            issues.append(
                {
                    "code": "current_v4_7_missing_region_key",
                    "db": _db_path(record.get("db")),
                    "scope_id": _text(record.get("scope_id")),
                    "capsule_id": capsule_id,
                    "revision": record.get("slow_revision"),
                }
            )
            continue
        by_region.setdefault(
            (_db_path(record.get("db")), _text(record.get("scope_id")), region_key),
            {},
        )[capsule_id] = record

    for (db_path, scope_id, region_key), capsules in by_region.items():
        evidence_locations: dict[
            str, list[tuple[str, int, str, str, str]]
        ] = defaultdict(list)
        claim_identity_capsules: dict[tuple[str, str], set[str]] = defaultdict(set)
        for capsule_id, record in capsules.items():
            metadata = record["metadata"]
            for claim_index, claim in enumerate(list(metadata.get("claims") or [])):
                if not isinstance(claim, Mapping):
                    continue
                slot = _text(claim.get("canonical_slot"))
                claim_text = _normal_text(claim.get("text"))
                if slot and claim_text:
                    claim_identity_capsules[(slot, claim_text)].add(capsule_id)
                for role in ("support", "counterevidence"):
                    for evidence_id in list(claim.get(role) or []):
                        evidence_id = _text(evidence_id)
                        if evidence_id:
                            evidence_locations[evidence_id].append(
                                (capsule_id, claim_index, role, slot, claim_text)
                            )
        invalid_repeated_evidence: dict[str, list[str]] = {}
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
            invalid_repeated_evidence[evidence_id] = [
                f"{capsule}:{claim_index}:{role}"
                for capsule, claim_index, role, _, _ in locations
            ]
        if invalid_repeated_evidence:
            issues.append(
                {
                    "code": "duplicate_evidence_across_active_capsules",
                    "db": db_path,
                    "scope_id": scope_id,
                    "region_key": region_key,
                    "capsule_ids": sorted(capsules),
                    "evidence": invalid_repeated_evidence,
                }
            )
        duplicated_claim_identities = {
            f"{slot}\u241f{text}": sorted(capsule_ids)
            for (slot, text), capsule_ids in claim_identity_capsules.items()
            if len(capsule_ids) > 1
        }
        if duplicated_claim_identities:
            issues.append(
                {
                    "code": "semantic_claim_in_multiple_active_capsules",
                    "db": db_path,
                    "scope_id": scope_id,
                    "region_key": region_key,
                    "capsule_ids": sorted(capsules),
                    "claim_identities": duplicated_claim_identities,
                }
            )
    return issues


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _db_path(value: Any) -> str:
    """Normalize a persisted DB identity before using it as a graph key."""
    return str(Path(str(value)).resolve())


def _identity(db_path: Any, scope_id: Any, memory_id: Any) -> Identity:
    return (_db_path(db_path), _text(scope_id), _text(memory_id))


def _fast_quote(metadata: Mapping[str, Any]) -> str:
    return _text(
        metadata.get("evidence_quote")
        or metadata.get("raw_content")
        or metadata.get("source_span")
    )


def _fast_source_parent(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "session_index": metadata.get("session_index"),
        "parent_chunk_index": metadata.get(
            "parent_chunk_index", metadata.get("message_index")
        ),
        "source_record_id": metadata.get("source_record_id"),
        "evidence_char_start": metadata.get("evidence_char_start"),
        "evidence_char_end": metadata.get("evidence_char_end"),
    }


def _slow_source_parent(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **_fast_source_parent(metadata),
        "message_index": metadata.get("message_index"),
        "event_id": metadata.get("event_id"),
    }


def _expected_slow_context_provenance(
    record_metadata: Mapping[str, Any],
    claim: Mapping[str, Any],
    source_parents: list[Any],
    summary_candidate_id: str,
    *,
    patch_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = {
        "memory_layer": "slow",
        "content_variant": "slow_memory_capsule",
        "capsule_id": _text(record_metadata.get("capsule_id")),
        "revision": int(record_metadata.get("revision")),
        "claim_id": _text(claim.get("claim_id")),
        "canonical_slot": _text(claim.get("canonical_slot")),
        "patch_id": _text(record_metadata.get("patch_id")),
        "source_parents": source_parents,
        "candidate_kind": "capsule_claim",
        "capsule_summary_candidate_id": summary_candidate_id,
    }
    if _current_summary_contract(record_metadata, patch_metadata):
        provenance["summary_contract_version"] = SLOW_SUMMARY_CONTRACT_VERSION
    return provenance


def _production_fast_predicate(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    """Mirror the production semantic-leaf predicate used by Fast retrieval."""
    return (
        _text(metadata.get("memory_layer")) == "fast"
        and _text(metadata.get("content_variant")) == PRODUCTION_FAST_VARIANT
        and _text(metadata.get("node_kind")) == PRODUCTION_FAST_NODE_KIND
        and metadata.get("atomic_evidence_leaf") is True
        and _text(metadata.get("authority")) == "user_assertion"
        and bool(_text(row.get("value")))
    )


def _reference_identity(
    value: Any,
    *,
    db_path: Any,
    scope_id: Any,
    field: str = "memory_id",
) -> Identity | None:
    """Resolve a local reference without ever consulting a global bare-ID index."""
    if isinstance(value, Mapping):
        memory_id = _text(
            value.get(field)
            or value.get("memory_id")
            or value.get("leaf_id")
            or value.get("evidence_memory_id")
        )
        declared_scope = _text(value.get("scope_id"))
        declared_db = value.get("db_path")
        if declared_scope and declared_scope != _text(scope_id):
            return None
        if declared_db and _db_path(declared_db) != _db_path(db_path):
            return None
        reference_scope = _text(scope_id)
        reference_db = db_path
    else:
        memory_id = _text(value)
        reference_scope = _text(scope_id)
        reference_db = db_path
    if not memory_id or not reference_scope:
        return None
    return _identity(reference_db, reference_scope, memory_id)


def _source_for_reference(
    sources: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    db_path: Any,
    scope_id: Any,
    source_record_id: Any,
) -> Mapping[str, Any] | None:
    return sources.get(_identity(db_path, scope_id, source_record_id))


def _source_message_indices(
    sources: Mapping[Identity, Mapping[str, Any]],
) -> tuple[
    dict[SourceMessageKey, list[Mapping[str, Any]]],
    dict[SourceScopeMessageKey, list[Mapping[str, Any]]],
]:
    by_database: dict[SourceMessageKey, list[Mapping[str, Any]]] = defaultdict(list)
    by_scope: dict[SourceScopeMessageKey, list[Mapping[str, Any]]] = defaultdict(list)
    for identity, source in sources.items():
        message_id = _text(source.get("message_id"))
        by_database[(identity[0], identity[1], message_id)].append(source)
        by_scope[(identity[1], message_id)].append(source)
    return dict(by_database), dict(by_scope)


def _source_for_message(
    source_messages: Mapping[SourceMessageKey, Sequence[Mapping[str, Any]]],
    *,
    db_path: Any,
    scope_id: Any,
    message_id: Any,
) -> Mapping[str, Any] | None:
    key = (_db_path(db_path), _text(scope_id), _text(message_id))
    matches = list(source_messages.get(key, ()))
    return matches[0] if len(matches) == 1 else None


def _source_span_quote(
    source: Mapping[str, Any],
    start: Any,
    end: Any,
    quote: Any,
    *,
    label: str,
    issues: list[str],
) -> bool:
    if not isinstance(quote, str) or not quote:
        issues.append(f"{label}: quote is missing")
        return False
    if isinstance(start, bool) or isinstance(end, bool):
        issues.append(f"{label}: source character span is invalid")
        return False
    try:
        start_value, end_value = int(start), int(end)
    except (TypeError, ValueError):
        issues.append(f"{label}: source character span is invalid")
        return False
    content = str(source.get("content") or "")
    if start_value < 0 or end_value <= start_value or end_value > len(content):
        issues.append(f"{label}: source character span is out of bounds")
        return False
    actual = content[start_value:end_value]
    if actual != quote:
        issues.append(f"{label}: source slice does not equal quote exactly")
        return False
    return True


def _unique_source_quote_span(
    source: Mapping[str, Any],
    quote: Any,
    *,
    label: str,
    issues: list[str],
) -> tuple[int, int] | None:
    if not isinstance(quote, str) or not quote:
        issues.append(f"{label}: quote is missing")
        return None
    content = str(source.get("content") or "")
    start = content.find(quote)
    if start < 0:
        issues.append(f"{label}: quote is not an exact Source substring")
        return None
    if content.find(quote, start + 1) >= 0:
        issues.append(f"{label}: quote is not unique within Source")
        return None
    return start, start + len(quote)


def _validate_source_parent(
    parent: Any,
    *,
    db_path: Any,
    scope_id: Any,
    sources: Mapping[tuple[str, str, str], Mapping[str, Any]],
    issues: list[str],
    label: str,
    quote: Any = None,
) -> Mapping[str, Any] | None:
    parent_keys = frozenset(parent) if isinstance(parent, Mapping) else frozenset()
    if not isinstance(parent, Mapping) or parent_keys not in {
        frozenset(FAST_SOURCE_PARENT_KEYS),
        frozenset(SLOW_SOURCE_PARENT_KEYS),
    }:
        issues.append(f"{label}: source parent schema is invalid")
        return None
    source_id = _text(parent.get("source_record_id"))
    source = _source_for_reference(
        sources,
        db_path=db_path,
        scope_id=scope_id,
        source_record_id=source_id,
    )
    if source is None:
        issues.append(f"{label}: source parent is not the current immutable Source")
        return None
    source_meta = source.get("metadata") if isinstance(source.get("metadata"), Mapping) else {}
    if parent.get("session_index") != source_meta.get("session_index"):
        issues.append(f"{label}: source parent session_index is not bound to immutable Source")
    if parent.get("parent_chunk_index") != source_meta.get("message_index"):
        issues.append(f"{label}: source parent chunk coordinate is not bound to immutable Source")
    if "message_index" in parent and parent.get("message_index") != source_meta.get(
        "message_index"
    ):
        issues.append(f"{label}: source parent message_index is not bound to immutable Source")
    if "event_id" in parent and parent.get("event_id") != source_meta.get("event_id"):
        issues.append(f"{label}: source parent event_id is not bound to immutable Source")
    if quote is None:
        content = str(source.get("content") or "")
        try:
            start_value, end_value = int(parent.get("evidence_char_start")), int(parent.get("evidence_char_end"))
            quote = content[start_value:end_value]
        except (TypeError, ValueError):
            quote = ""
    _source_span_quote(
        source,
        parent.get("evidence_char_start"),
        parent.get("evidence_char_end"),
        quote,
        label=label,
        issues=issues,
    )
    return source


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _rows(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(f'SELECT * FROM "{table}"')]


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata_json", row.get("metadata", "{}"))
    value = _json(raw)
    if not isinstance(value, dict):
        raise AuditError("metadata_json must contain an object")
    return value


def _valid_active_to_superseded_lifecycle(
    connection: sqlite3.Connection,
    scope_id: str,
    memory_id: str,
) -> bool:
    seen: set[str] = set()
    expected_slot = ""
    prior_turn = -1
    while memory_id and memory_id not in seen:
        seen.add(memory_id)
        row = connection.execute(
            "SELECT slot_key,turn_index,state,metadata_json FROM records "
            "WHERE scope_id=? AND memory_id=?",
            (scope_id, memory_id),
        ).fetchone()
        if row is None:
            return False
        slot_key = _text(row[0])
        turn_index = int(row[1])
        state = _text(row[2])
        metadata = _json(row[3])
        if not isinstance(metadata, Mapping):
            return False
        if not expected_slot:
            expected_slot = slot_key
        if slot_key != expected_slot or turn_index < prior_turn:
            return False
        if state in ACTIVE_RECORD_STATES:
            return True
        if (
            state != "superseded"
            or _text(metadata.get("superseded_reason"))
            not in VALID_SUPERSESSION_LIFECYCLE_REASONS
        ):
            return False
        memory_id = _text(metadata.get("superseded_by"))
        prior_turn = turn_index
    return False


def _walk(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def _has_forbidden_input(value: Any) -> str | None:
    for path, item in _walk(value):
        if isinstance(item, Mapping):
            for key in item:
                lowered = str(key).casefold()
                if lowered in FORBIDDEN_INPUT_KEYS or any(token in lowered for token in ("gold_", "benchmark_", "judge_")):
                    return f"{path}.{key}"
    return None


def _artifact_markers(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    for current, item in _walk(value, path):
        if isinstance(item, Mapping):
            for key in item:
                if any(part in str(key).casefold() for part in FORBIDDEN_ARTIFACT_KEY_PARTS):
                    found.append(f"{current}.{key}")
        elif (
            isinstance(item, str)
            and not any(part in current.casefold() for part in SEMANTIC_IDENTIFIER_PATH_PARTS)
            and ".source_spans[" not in current.casefold()
            and (
                not current
                or any(
                    part in current.casefold()
                    for part in OPERATIONAL_MARKER_VALUE_PATH_PARTS
                )
            )
            and MARKER_RE.search(item)
        ):
            found.append(current)
    return found


def _slow_pro_route_is_justified(metadata: Mapping[str, Any]) -> bool:
    route = _text(metadata.get("route"))
    if route not in {"pro", "flash_to_pro"}:
        return True
    reason = _text(metadata.get("route_reason"))
    if CONFLICT_RE.search(reason):
        return True
    reason_codes = {item for item in reason.split("+") if item}
    partition_ids = metadata.get("semantic_partition_capsule_ids")
    migration = (
        route == "pro"
        and "semantic_partition_migration" in reason_codes
        and metadata.get("semantic_partition_contract_version")
        == CURRENT_SLOW_PARTITION_CONTRACT_VERSION
        and metadata.get("semantic_partition_mode") == "migrate"
        and isinstance(partition_ids, list)
        and bool(partition_ids)
        and all(isinstance(item, str) and bool(item.strip()) for item in partition_ids)
        and len(partition_ids) == len(set(partition_ids))
    )
    required_ids = metadata.get("required_operation_evidence_ids")
    semantic_management = (
        route == "pro"
        and bool(
            reason_codes
            & {
                "generic_region_semantic_management",
                "initial_multi_slot_semantic_partition",
            }
        )
        and metadata.get("semantic_partition_contract_version")
        == CURRENT_SLOW_PARTITION_CONTRACT_VERSION
        and metadata.get("semantic_partition_mode") == "manage"
        and not partition_ids
        and isinstance(required_ids, list)
        and bool(required_ids)
        and all(isinstance(item, str) and bool(item.strip()) for item in required_ids)
        and len(required_ids) == len(set(required_ids))
    )
    return migration or semantic_management


def _retrieval_forbidden(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    for current, item in _walk(value, path):
        if isinstance(item, Mapping):
            for key, child in item.items():
                lowered = str(key).casefold()
                if lowered == "runtime_input_has_gold" and child is False:
                    continue
                if any(part in lowered for part in FORBIDDEN_RETRIEVAL_KEY_PARTS):
                    found.append(f"{current}.{key}")
    return found


def discover_worker_databases(run_dir: Path) -> dict[str, list[Path]]:
    """Return the frozen manifest DBs, or recursively discover legacy runs."""
    run_dir = Path(run_dir).resolve()
    manifest = run_dir / "scope_manifest.jsonl"
    if manifest.is_file():
        result: dict[str, list[Path]] = defaultdict(list)
        seen_paths: set[Path] = set()
        try:
            lines = manifest.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise AuditError(f"cannot read frozen scope manifest: {manifest}") from exc
        if not lines:
            raise AuditError(f"frozen scope manifest is empty: {manifest}")
        for line_number, raw in enumerate(lines, start=1):
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AuditError(
                    f"frozen scope manifest line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(row, Mapping):
                raise AuditError(
                    f"frozen scope manifest line {line_number} is not an object"
                )
            raw_path = _text(row.get("db_path"))
            path = Path(raw_path)
            if not path.is_absolute():
                path = run_dir / path
            path = path.resolve()
            try:
                relative = path.relative_to(run_dir)
            except ValueError as exc:
                raise AuditError(
                    f"frozen worker database is outside the run directory: {path}"
                ) from exc
            if not path.is_file():
                raise AuditError(f"frozen worker database does not exist: {path}")
            if path in seen_paths:
                raise AuditError(f"frozen worker database is duplicated: {path}")
            worker = next(
                (part for part in relative.parts[:-1] if part.startswith("worker")),
                "",
            )
            if not worker:
                raise AuditError(f"frozen database has no worker directory: {path}")
            if result[worker]:
                raise AuditError(f"frozen manifest repeats worker {worker}")
            seen_paths.add(path)
            result[worker].append(path)
        return {key: values for key, values in sorted(result.items())}

    result: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in {".sqlite3", ".sqlite", ".db"}:
            continue
        if path.name.startswith(".") or path.name.startswith("tmp_"):
            continue
        relative = path.relative_to(run_dir)
        worker = next((part for part in relative.parts[:-1] if part.startswith("worker")), "root")
        result[worker].append(path)
    return {key: sorted(values) for key, values in sorted(result.items())}


def _parse_worker_db_specs(run_dir: Path, specs: Iterable[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for spec in specs:
        if "=" not in spec:
            raise AuditError(f"--worker-db must be WORKER=PATH: {spec!r}")
        worker, raw_path = spec.split("=", 1)
        worker = _text(worker)
        path = Path(raw_path)
        if not path.is_absolute():
            path = run_dir / path
        path = path.resolve()
        if not worker or not path.is_file():
            raise AuditError(f"worker DB does not exist: {spec!r}")
        result[worker].append(path)
    return {key: sorted(set(values)) for key, values in sorted(result.items())}


def _load_input(run_dir: Path) -> tuple[Path, list[Mapping[str, Any]]]:
    candidates = [
        run_dir / name for name in ("writer_input.json", "product_writer_input.json", "input.json")
    ]
    candidates += sorted(run_dir.rglob("*writer*input*.json"))
    candidates = list(dict.fromkeys(path for path in candidates if path.is_file()))
    if not candidates:
        raise AuditError("writer input artifact is missing")
    path = candidates[0]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read writer input {path}: {exc}") from exc
    rows = raw.get("rows") if isinstance(raw, Mapping) and isinstance(raw.get("rows"), list) else raw
    if not isinstance(rows, list) or not rows or not all(isinstance(row, Mapping) for row in rows):
        raise AuditError("writer input must be a non-empty JSON array of objects")
    forbidden = _has_forbidden_input(raw)
    if forbidden:
        raise AuditError(f"writer input contains non-history field: {forbidden}")
    return path, list(rows)


def _input_message_inventory(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    excluded: dict[tuple[str, str], dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        qid = _text(row.get("question_id")) or f"row{row_index:04d}"
        scope = _text(row.get("scope_id")) or f"tmcra_v4:{qid}"
        if "haystack_sessions" in row:
            sessions = list(row.get("haystack_sessions") or [])
            session_ids = list(row.get("haystack_session_ids") or [])
        elif "sessions" in row:
            sessions = list(row.get("sessions") or [])
            session_ids = []
        else:
            sessions = [list(row.get("messages") or [])]
            session_ids = []
        for session_index, messages in enumerate(sessions):
            if not isinstance(messages, list):
                raise AuditError(f"input row {row_index} session {session_index} is not a message list")
            session_id = (_text(session_ids[session_index]) if session_index < len(session_ids) else "") or _text(row.get("session_id")) or f"session-{session_index:03d}"
            for message_index, message in enumerate(messages):
                if not isinstance(message, Mapping):
                    raise AuditError(f"input message {row_index}/{session_index}/{message_index} is not an object")
                content = str(message.get("content") or "")
                role = _text(message.get("role")).casefold()
                message_id = _text(message.get("message_id")) or f"s{session_index:03d}_m{message_index:03d}"
                if role not in {"user", "assistant", "system", "tool"}:
                    raise AuditError(f"invalid input message {scope}/{message_id}")
                key = (scope, message_id)
                if not content.strip():
                    value = {
                        "scope_id": scope,
                        "session_id": session_id,
                        "session_index": session_index,
                        "message_index": message_index,
                        "message_id": message_id,
                        "message_role": role,
                        "reason": "empty_content",
                        "content_sha256": _sha(content),
                    }
                    if key in excluded and excluded[key] != value:
                        raise AuditError(
                            f"duplicate excluded input message with conflicting content: {key}"
                        )
                    excluded[key] = value
                    continue
                value = {"scope_id": scope, "session_id": session_id, "message_id": message_id, "content": content, "role": role, "session_index": session_index, "message_index": message_index}
                if key in expected and expected[key] != value:
                    raise AuditError(f"duplicate input message with conflicting content: {key}")
                expected[key] = value
    return expected, excluded


def _expected_messages(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return _input_message_inventory(rows)[0]


def _audit_source_exclusions(
    run_dir: Path,
    expected: Mapping[tuple[str, str], Mapping[str, Any]],
    issues: list[str],
) -> None:
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    paths = sorted(run_dir.rglob("source_exclusions.json"))
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"{path}: invalid source exclusion artifact: {exc}")
            continue
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != "tmcra.v4.source-exclusions.1"
            or not isinstance(payload.get("messages"), list)
            or payload.get("count") != len(payload["messages"])
        ):
            issues.append(f"{path}: invalid source exclusion schema")
            continue
        for item in payload["messages"]:
            if not isinstance(item, Mapping):
                issues.append(f"{path}: source exclusion entry is not an object")
                continue
            key = (_text(item.get("scope_id")), _text(item.get("message_id")))
            normalized = dict(item)
            old = observed.get(key)
            if old is not None and old != normalized:
                issues.append(f"{path}: conflicting source exclusion entry {key}")
            else:
                observed[key] = normalized
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        if missing:
            issues.append(f"source exclusion artifact lacks empty input messages: {missing}")
        if unexpected:
            issues.append(f"source exclusion artifact contains unexpected messages: {unexpected}")
    for key in sorted(set(observed) & set(expected)):
        if observed[key] != dict(expected[key]):
            issues.append(f"source exclusion metadata differs for {key}")


def _source_parent(meta: Mapping[str, Any], row: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    raw = meta.get("source_parent", meta.get("parent"))
    if isinstance(raw, Mapping):
        parent = dict(raw)
        parent.setdefault("source_record_id", source_id)
        return parent
    parent = {"source_record_id": _text(meta.get("source_record_id")) or source_id}
    for key in ("session_index", "parent_chunk_index", "message_index", "event_id", "evidence_char_start", "evidence_char_end"):
        if key in meta:
            parent[key] = meta[key]
    return parent


def _source_rows(db_paths: Iterable[Path], expected_scopes: set[str]) -> tuple[dict[Identity, dict[str, Any]], list[str], set[str]]:
    sources: dict[Identity, dict[str, Any]] = {}
    errors: list[str] = []
    seen_tables: set[str] = set()
    for path in db_paths:
        with sqlite3.connect(path) as con:
            con.row_factory = sqlite3.Row
            tables = _tables(con)
            for table in SOURCE_TABLES:
                if table not in tables:
                    continue
                seen_tables.add(table)
                for row in _rows(con, table):
                    scope = _text(row.get("scope_id"))
                    meta = _metadata(row)
                    if scope not in expected_scopes:
                        continue
                    if meta.get("content_variant") != "source_message" or meta.get("node_kind") != "immutable_source_message":
                        continue
                    message_id = _text(meta.get("message_id"))
                    raw_content = meta.get("raw_content")
                    if not isinstance(raw_content, str) or not raw_content:
                        errors.append(f"{path}:{table}: actual source lacks exact metadata.raw_content")
                        continue
                    content = raw_content
                    for field in ("source_span", "source_turn_text"):
                        if meta.get(field) != content:
                            errors.append(
                                f"{path}:{table}: actual source {scope}/{message_id} {field} differs from raw_content"
                            )
                    stored_value = str(row.get("value") or "")
                    if " ".join(stored_value.split()) != " ".join(content.split()):
                        errors.append(
                            f"{path}:{table}: actual source {scope}/{message_id} graph value differs from raw_content"
                        )
                    if not scope or not message_id or not content:
                        errors.append(f"{path}:{table}: actual source record lacks message/content")
                        continue
                    source_id = _text(meta.get("source_record_id")) or _text(row.get("memory_id"))
                    status = _text(meta.get("enrichment_status"))
                    if status not in {"pending", "failed", "enriched"}:
                        errors.append(f"{path}: actual source {scope}/{message_id} has unknown enrichment status")
                    item = {"source_record_id": source_id, "scope_id": scope, "message_id": message_id, "content": content, "role": _text(meta.get("role") or meta.get("speaker")).casefold(), "status": status, "turn_index": row.get("turn_index"), "metadata": meta, "db_path": _db_path(path)}
                    key = (_db_path(path), scope, source_id)
                    old = sources.get(key)
                    if not source_id:
                        errors.append(f"{path}: actual source {key} lacks source_record_id")
                    if old and (old["content"] != content or old["source_record_id"] != item["source_record_id"]):
                        errors.append(f"{path}:{table}: conflicting immutable source {key}")
                    else:
                        sources[key] = item
    return sources, errors, seen_tables


def _collect_jsonl(run_dir: Path, names: tuple[str, ...]) -> list[tuple[Path, dict[str, Any]]]:
    result: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(run_dir.rglob("*.jsonl")):
        if path.name not in names:
            continue
        relative = path.relative_to(run_dir)
        if len(relative.parts) > 1:
            canonical_worker = (
                re.fullmatch(r"worker_\d+", relative.parts[0]) is not None
                or (
                    len(relative.parts) >= 3
                    and relative.parts[0] == "writer"
                    and re.fullmatch(r"worker_\d+", relative.parts[1]) is not None
                )
            )
            if not canonical_worker:
                continue
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"invalid JSONL {path}:{index}: {exc}") from exc
            if not isinstance(value, dict):
                raise AuditError(f"JSONL object required at {path}:{index}")
            result.append((path, value))
    return result


def _audit_writer_input(run_dir: Path, expected: Mapping[tuple[str, str], Mapping[str, Any]], db_paths: list[Path], issues: list[str]) -> None:
    calls = _collect_jsonl(run_dir, ("product_writer_calls.jsonl", "writer_calls.jsonl"))
    raw_calls = _collect_jsonl(run_dir, ("product_writer_raw_responses.jsonl",))
    revalidations = _collect_jsonl(run_dir, ("product_writer_revalidations.jsonl",))
    reconciliation_revalidations = _collect_jsonl(
        run_dir, ("product_writer_reconciliation_revalidations.jsonl",)
    )
    validated_batch_recoveries = _collect_jsonl(
        run_dir, ("product_writer_validated_batch_recoveries.jsonl",)
    )
    historical_binding_recoveries = _collect_jsonl(
        run_dir, ("product_writer_historical_binding_recoveries.jsonl",)
    )
    keep_parallel_migrations = _collect_jsonl(
        run_dir, ("product_writer_keep_parallel_migrations.jsonl",)
    )
    challenge_migrations = _collect_jsonl(
        run_dir, ("product_writer_challenge_lifecycle_migrations.jsonl",)
    )
    interrupted_calls = _collect_jsonl(
        run_dir, ("product_writer_interrupted_calls.jsonl",)
    )
    revalidation_by_batch: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, record in revalidations:
        batch_id = _text(record.get("batch_id"))
        if not batch_id or batch_id in revalidation_by_batch:
            issues.append(f"writer revalidation has missing or duplicate batch ID: {path}")
            continue
        if int(record.get("physical_api_calls", -1)) != 0:
            issues.append(f"writer revalidation claims a physical API call: {path}:{batch_id}")
        if not _text(record.get("raw_response_sha256")) or not _text(
            record.get("validated_response_sha256")
        ):
            issues.append(f"writer revalidation hashes are incomplete: {path}:{batch_id}")
        revalidation_by_batch[batch_id] = (path, record)
    reconciliation_revalidation_by_job: dict[
        str, tuple[Path, dict[str, Any]]
    ] = {}
    for path, record in reconciliation_revalidations:
        job_id = _text(record.get("job_id"))
        if not job_id or job_id in reconciliation_revalidation_by_job:
            issues.append(
                f"reconciliation revalidation has missing or duplicate job ID: {path}"
            )
            continue
        if int(record.get("physical_api_calls", -1)) != 0:
            issues.append(
                f"reconciliation revalidation claims a physical API call: {path}:{job_id}"
            )
        if not _text(record.get("raw_response_sha256")) or not _text(
            record.get("normalized_adjudication_sha256")
        ):
            issues.append(
                f"reconciliation revalidation hashes are incomplete: {path}:{job_id}"
            )
        reconciliation_revalidation_by_job[job_id] = (path, record)
    validated_recovery_by_batch: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, record in validated_batch_recoveries:
        batch_id = _text(record.get("batch_id"))
        if not batch_id or batch_id in validated_recovery_by_batch:
            issues.append(
                f"validated batch recovery has missing or duplicate batch ID: {path}"
            )
            continue
        if int(record.get("physical_api_calls", -1)) != 0:
            issues.append(
                f"validated batch recovery claims a physical API call: {path}:{batch_id}"
            )
        if not _text(record.get("prior_error_sha256")) or not _text(
            record.get("response_sha256")
        ):
            issues.append(
                f"validated batch recovery hashes are incomplete: {path}:{batch_id}"
            )
        validated_recovery_by_batch[batch_id] = (path, record)
    historical_recovery_by_job: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, record in historical_binding_recoveries:
        job_id = _text(record.get("job_id"))
        if not job_id or job_id in historical_recovery_by_job:
            issues.append(
                f"historical binding recovery has missing or duplicate job ID: {path}"
            )
            continue
        binding_mode = _text(record.get("binding_mode")) or (
            "verified_historical_selected"
            if _text(record.get("resolved_memory_id"))
            in {"", _text(record.get("selected_memory_id"))}
            else ""
        )
        mode_invalid = False
        if binding_mode == "verified_historical_selected":
            mode_invalid = _text(record.get("resolved_memory_id")) not in {
                "",
                _text(record.get("selected_memory_id")),
            }
        elif binding_mode == "unique_active_semantic_equivalent":
            mode_invalid = (
                not _text(record.get("resolved_memory_id"))
                or _text(record.get("resolved_memory_id"))
                == _text(record.get("selected_memory_id"))
                or not _text(record.get("frozen_semantic_identity_sha256"))
                or _text(record.get("frozen_semantic_identity_sha256"))
                != _text(record.get("resolved_semantic_identity_sha256"))
            )
        else:
            mode_invalid = True
        if (
            int(record.get("physical_api_calls", -1)) != 0
            or _text(record.get("decision"))
            not in {"replace_current", "keep_parallel", "challenge", "quarantine"}
            or _text(record.get("superseded_reason"))
            != "v4_reconciliation_replace_current"
            or not _text(record.get("selected_memory_id"))
            or not _text(record.get("frozen_binding_identity_sha256"))
            or _text(record.get("frozen_binding_identity_sha256"))
            != _text(record.get("historical_binding_identity_sha256"))
            or mode_invalid
        ):
            issues.append(
                f"historical binding recovery contract is malformed: {path}:{job_id}"
            )
        historical_recovery_by_job[job_id] = (path, record)
    keep_parallel_migration_by_job: dict[
        str, tuple[Path, dict[str, Any]]
    ] = {}
    for path, record in keep_parallel_migrations:
        job_id = _text(record.get("job_id"))
        if not job_id or job_id in keep_parallel_migration_by_job:
            issues.append(
                f"keep_parallel migration has missing or duplicate job ID: {path}"
            )
            continue
        if (
            _text(record.get("schema_version"))
            != KEEP_PARALLEL_MIGRATION_SCHEMA_VERSION
            or _text(record.get("migration_version"))
            != KEEP_PARALLEL_MIGRATION_VERSION
            or _text(record.get("status")) != "completed"
            or int(record.get("physical_api_calls", -1)) != 0
            or not _text(record.get("migration_id"))
            or not _text(record.get("selected_memory_id"))
            or not _text(record.get("resolved_memory_id"))
            or not _text(record.get("incoming_memory_id"))
            or _text(record.get("assertion_binding_mode"))
            not in {
                "original_assertion_index",
                "exact_evidence_identity_after_commit_reindex",
            }
            or not re.fullmatch(r"[0-9a-f]{64}", _text(record.get("before_sha256")))
            or not re.fullmatch(r"[0-9a-f]{64}", _text(record.get("after_sha256")))
        ):
            issues.append(
                f"keep_parallel migration contract is malformed: {path}:{job_id}"
            )
        keep_parallel_migration_by_job[job_id] = (path, record)
    challenge_migration_by_job: dict[
        str, tuple[Path, dict[str, Any]]
    ] = {}
    for path, record in challenge_migrations:
        job_id = _text(record.get("job_id"))
        if not job_id or job_id in challenge_migration_by_job:
            issues.append(
                f"challenge migration has missing or duplicate job ID: {path}"
            )
            continue
        if (
            _text(record.get("schema_version"))
            != CHALLENGE_MIGRATION_SCHEMA_VERSION
            or _text(record.get("migration_version"))
            != CHALLENGE_MIGRATION_VERSION
            or _text(record.get("status")) != "completed"
            or int(record.get("physical_api_calls", -1)) != 0
            or not _text(record.get("migration_id"))
            or not _text(record.get("selected_memory_id"))
            or not _text(record.get("resolved_memory_id"))
            or not _text(record.get("incoming_memory_id"))
            or _text(record.get("assertion_binding_mode"))
            not in {
                "original_assertion_index",
                "exact_evidence_identity_after_commit_reindex",
            }
            or _text(record.get("restored_slot_head"))
            != _text(record.get("resolved_memory_id"))
            or not re.fullmatch(r"[0-9a-f]{64}", _text(record.get("before_sha256")))
            or not re.fullmatch(r"[0-9a-f]{64}", _text(record.get("after_sha256")))
        ):
            issues.append(
                f"challenge migration contract is malformed: {path}:{job_id}"
            )
        challenge_migration_by_job[job_id] = (path, record)
    interrupted_by_key: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, record in interrupted_calls:
        call_key = _text(record.get("call_key"))
        if not call_key or call_key in interrupted_by_key:
            issues.append(
                f"interrupted writer call has missing or duplicate call key: {path}"
            )
            continue
        for marker in _artifact_markers(record):
            issues.append(f"interrupted writer artifact marker {path}:{marker}")
        if (
            record.get("physical_api_call") is not True
            or int(record.get("physical_api_calls", 0) or 0) != 1
            or _text(record.get("status"))
            != "outcome_unknown_after_confirmed_process_loss"
            or record.get("same_model_replacement") is not True
            or _text(record.get("replacement_model")) != _text(record.get("model"))
            or not _text(record.get("physical_call_id"))
        ):
            issues.append(
                f"interrupted writer call recovery contract is malformed: {path}:{call_key}"
            )
        interrupted_by_key[call_key] = (path, record)
    raw_by_key: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, raw_call in raw_calls:
        call_key = _text(raw_call.get("call_key"))
        if not call_key or call_key in raw_by_key:
            issues.append(f"raw writer response has missing or duplicate call key: {path}")
            continue
        raw_response = raw_call.get("raw_response")
        if not isinstance(raw_response, str) or not raw_response:
            issues.append(f"raw writer response is empty: {path}:{call_key}")
            continue
        if _sha(raw_response) != _text(raw_call.get("raw_response_sha256")):
            issues.append(f"raw writer response hash differs: {path}:{call_key}")
        metadata_hash = _text(raw_call.get("metadata_response_sha256"))
        if metadata_hash and metadata_hash != _sha(raw_response):
            issues.append(f"raw writer response differs from API metadata hash: {path}:{call_key}")
        raw_by_key[call_key] = (path, raw_call)
    calls_by_key: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path, call in calls:
        for marker in _artifact_markers(call):
            issues.append(f"writer artifact marker {path}:{marker}")
        model = _text(call.get("model"))
        metadata = call.get("metadata") if isinstance(call.get("metadata"), Mapping) else {}
        call_key = _text(call.get("call_key"))
        calls_by_key[call_key].append((path, call))
        if _text(metadata.get("status")) == "completed" and call_key not in raw_by_key:
            issues.append(f"completed writer call lacks raw response: {path}:{call_key}")
        if "pro" in model.casefold():
            route = _text(call.get("route") or call.get("routing_reason") or call.get("stage") or metadata.get("route") or metadata.get("stage") or metadata.get("routing_reason"))
            if not re.search(r"reconcil|conflict|same[_ -]?slot", route, re.I):
                issues.append(f"Pro writer call is not a reconciliation route: {path}")
    for call_key, (path, interrupted) in interrupted_by_key.items():
        replacements = calls_by_key.get(call_key, [])
        if len(replacements) != 1:
            issues.append(
                f"interrupted writer call lacks exactly one replacement: {path}:{call_key}"
            )
            continue
        replacement_model = _text(replacements[0][1].get("model"))
        if replacement_model != _text(interrupted.get("model")):
            issues.append(
                f"interrupted writer call changed model: {path}:{call_key}"
            )
        if call_key not in raw_by_key:
            issues.append(
                f"interrupted writer replacement lacks raw response: {path}:{call_key}"
            )
    seen_historical_recovery_jobs: set[str] = set()
    seen_keep_parallel_migration_jobs: set[str] = set()
    seen_keep_parallel_journal_jobs: set[str] = set()
    seen_challenge_migration_jobs: set[str] = set()
    seen_challenge_journal_jobs: set[str] = set()
    for path in db_paths:
        with sqlite3.connect(path) as con:
            con.row_factory = sqlite3.Row
            tables = _tables(con)
            if "v4_batch_journal" not in tables:
                continue
            migration_journal_by_job: dict[str, dict[str, Any]] = {}
            challenge_journal_by_job: dict[str, dict[str, Any]] = {}
            if "v4_keep_parallel_migrations" in tables:
                for journal in _rows(con, "v4_keep_parallel_migrations"):
                    journal_job_id = _text(journal.get("job_id"))
                    if (
                        not journal_job_id
                        or journal_job_id in seen_keep_parallel_journal_jobs
                    ):
                        issues.append(
                            f"{path}: keep_parallel migration journal has missing or duplicate job ID"
                        )
                        continue
                    seen_keep_parallel_journal_jobs.add(journal_job_id)
                    migration_journal_by_job[journal_job_id] = journal
                    artifact = keep_parallel_migration_by_job.get(journal_job_id)
                    journal_artifact = _json(journal.get("artifact_json"))
                    before_json = _text(journal.get("before_json"))
                    after_json = _text(journal.get("after_json"))
                    if (
                        artifact is None
                        or _text(journal.get("status")) != "completed"
                        or not isinstance(journal_artifact, Mapping)
                        or dict(journal_artifact) != artifact[1]
                        or _sha(before_json)
                        != _text(journal_artifact.get("before_sha256"))
                        or _sha(after_json)
                        != _text(journal_artifact.get("after_sha256"))
                    ):
                        issues.append(
                            f"{path}: keep_parallel migration journal/artifact differs: {journal_job_id}"
                        )
            if "v4_challenge_lifecycle_migrations" in tables:
                for journal in _rows(con, "v4_challenge_lifecycle_migrations"):
                    journal_job_id = _text(journal.get("job_id"))
                    if (
                        not journal_job_id
                        or journal_job_id in seen_challenge_journal_jobs
                    ):
                        issues.append(
                            f"{path}: challenge migration journal has missing or duplicate job ID"
                        )
                        continue
                    seen_challenge_journal_jobs.add(journal_job_id)
                    challenge_journal_by_job[journal_job_id] = journal
                    artifact = challenge_migration_by_job.get(journal_job_id)
                    journal_artifact = _json(journal.get("artifact_json"))
                    before_json = _text(journal.get("before_json"))
                    after_json = _text(journal.get("after_json"))
                    if (
                        artifact is None
                        or _text(journal.get("status")) != "completed"
                        or not isinstance(journal_artifact, Mapping)
                        or dict(journal_artifact) != artifact[1]
                        or _sha(before_json)
                        != _text(journal_artifact.get("before_sha256"))
                        or _sha(after_json)
                        != _text(journal_artifact.get("after_sha256"))
                    ):
                        issues.append(
                            f"{path}: challenge migration journal/artifact differs: {journal_job_id}"
                        )
            if "v4_reconciliation_jobs" in tables:
                for job in _rows(con, "v4_reconciliation_jobs"):
                    job_id = _text(job.get("job_id"))
                    response = _json(job.get("response_json"))
                    migration_artifact = keep_parallel_migration_by_job.get(job_id)
                    if migration_artifact is not None:
                        seen_keep_parallel_migration_jobs.add(job_id)
                    if (
                        _text(job.get("status")) == "completed"
                        and _text(job.get("decision")) == "keep_parallel"
                        and isinstance(response, Mapping)
                    ):
                        selected_memory_id = _text(
                            response.get("selected_memory_id")
                        )
                        historical_artifact = historical_recovery_by_job.get(job_id)
                        resolved_memory_id = (
                            _text(
                                historical_artifact[1].get("resolved_memory_id")
                            )
                            if historical_artifact is not None
                            else ""
                        ) or selected_memory_id
                        scope_id = _text(job.get("scope_id"))
                        selected = con.execute(
                            "SELECT state,slot_key,turn_index,supersedes_json,metadata_json "
                            "FROM records WHERE scope_id=? AND memory_id=?",
                            (scope_id, resolved_memory_id),
                        ).fetchone() if "records" in tables else None
                        immediate_overwrite = False
                        selected_metadata: Mapping[str, Any] = {}
                        incoming = None
                        incoming_metadata: Mapping[str, Any] = {}
                        if selected is not None:
                            selected_metadata_value = _json(selected[4])
                            if isinstance(selected_metadata_value, Mapping):
                                selected_metadata = selected_metadata_value
                            incoming_id = _text(
                                selected_metadata.get("superseded_by")
                            )
                            if incoming_id:
                                incoming = con.execute(
                                    "SELECT state,slot_key,turn_index,supersedes_json,metadata_json "
                                    "FROM records WHERE scope_id=? AND memory_id=?",
                                    (scope_id, incoming_id),
                                ).fetchone()
                            if incoming is not None:
                                incoming_metadata_value = _json(incoming[4])
                                if isinstance(incoming_metadata_value, Mapping):
                                    incoming_metadata = incoming_metadata_value
                                immediate_overwrite = (
                                    _text(selected[0]) == "superseded"
                                    and _text(
                                        incoming_metadata.get("message_id")
                                    )
                                    == _text(job.get("message_id"))
                                    and _text(
                                        incoming_metadata.get(
                                            "reconciliation_decision"
                                        )
                                    )
                                    == "keep_parallel"
                                )
                        if immediate_overwrite:
                            issues.append(
                                f"{path}: keep_parallel decision was overwritten by graph policy: {job_id}"
                            )
                        if migration_artifact is not None:
                            artifact = migration_artifact[1]
                            journal = migration_journal_by_job.get(job_id)
                            incoming_id = _text(artifact.get("incoming_memory_id"))
                            migrated_incoming = con.execute(
                                "SELECT slot_key,supersedes_json,metadata_json FROM records "
                                "WHERE scope_id=? AND memory_id=?",
                                (scope_id, incoming_id),
                            ).fetchone() if "records" in tables else None
                            migrated_incoming_metadata = (
                                _json(migrated_incoming[2])
                                if migrated_incoming is not None
                                else {}
                            )
                            migrated_supersedes = (
                                _json(migrated_incoming[1])
                                if migrated_incoming is not None
                                else []
                            )
                            request = _json(job.get("request_json"))
                            cited = (
                                request.get("new_cited_assertion")
                                if isinstance(request, Mapping)
                                else None
                            )
                            binding_mode = _text(
                                artifact.get("assertion_binding_mode")
                            )
                            if binding_mode == "original_assertion_index":
                                binding_mode_valid = (
                                    isinstance(
                                        migrated_incoming_metadata, Mapping
                                    )
                                    and int(
                                        migrated_incoming_metadata.get(
                                            "llm_write_proposal_index", -1
                                        )
                                    )
                                    == int(job.get("assertion_index") or 0)
                                )
                            else:
                                binding_mode_valid = (
                                    migrated_incoming is not None
                                    and isinstance(request, Mapping)
                                    and isinstance(cited, Mapping)
                                    and _text(migrated_incoming[0])
                                    == _text(request.get("canonical_slot_key"))
                                    and _text(
                                        migrated_incoming_metadata.get(
                                            "source_span"
                                        )
                                    )
                                    == _text(cited.get("evidence_quote"))
                                    and _text(
                                        migrated_incoming_metadata.get(
                                            "memory_type"
                                        )
                                    )
                                    == _text(cited.get("memory_type"))
                                    and _text(
                                        migrated_incoming_metadata.get(
                                            "entity_key"
                                        )
                                    )
                                    == _text(cited.get("entity_key"))
                                    and _text(
                                        migrated_incoming_metadata.get(
                                            "attribute_key"
                                        )
                                    )
                                    == _text(cited.get("attribute_key"))
                                )
                            if (
                                journal is None
                                or _text(artifact.get("selected_memory_id"))
                                != selected_memory_id
                                or _text(artifact.get("resolved_memory_id"))
                                != resolved_memory_id
                                or _text(artifact.get("scope_id")) != scope_id
                                or _text(artifact.get("message_id"))
                                != _text(job.get("message_id"))
                                or int(artifact.get("assertion_index", -1))
                                != int(job.get("assertion_index") or 0)
                                or selected is None
                                or not binding_mode_valid
                                or not _valid_active_to_superseded_lifecycle(
                                    con, scope_id, resolved_memory_id
                                )
                                or migrated_incoming is None
                                or not isinstance(migrated_supersedes, list)
                                or resolved_memory_id
                                in {_text(item) for item in migrated_supersedes}
                                or not isinstance(
                                    migrated_incoming_metadata, Mapping
                                )
                                or _text(
                                    migrated_incoming_metadata.get(
                                        "conflict_action"
                                    )
                                )
                                != "keep_parallel"
                                or _text(
                                    migrated_incoming_metadata.get(
                                        "conflict_reason"
                                    )
                                )
                                != "v4_reconciliation_keep_parallel"
                            ):
                                issues.append(
                                    f"{path}: keep_parallel migration does not match graph/job state: {job_id}"
                                )
                    challenge_artifact = challenge_migration_by_job.get(job_id)
                    if challenge_artifact is not None:
                        seen_challenge_migration_jobs.add(job_id)
                        if not (
                            _text(job.get("status")) == "completed"
                            and _text(job.get("decision")) == "challenge"
                        ):
                            issues.append(
                                f"{path}: challenge migration does not reference a completed challenge job: {job_id}"
                            )
                    if (
                        _text(job.get("status")) == "completed"
                        and _text(job.get("decision")) == "challenge"
                        and isinstance(response, Mapping)
                    ):
                        selected_memory_id = _text(
                            response.get("selected_memory_id")
                        )
                        historical_artifact = historical_recovery_by_job.get(job_id)
                        resolved_memory_id = (
                            _text(
                                historical_artifact[1].get("resolved_memory_id")
                            )
                            if historical_artifact is not None
                            else ""
                        ) or selected_memory_id
                        scope_id = _text(job.get("scope_id"))
                        selected = con.execute(
                            "SELECT state,slot_key,turn_index,supersedes_json,metadata_json "
                            "FROM records WHERE scope_id=? AND memory_id=?",
                            (scope_id, resolved_memory_id),
                        ).fetchone() if "records" in tables else None
                        immediate_overwrite = False
                        if selected is not None:
                            selected_metadata = _json(selected[4])
                            selected_metadata = (
                                selected_metadata
                                if isinstance(selected_metadata, Mapping)
                                else {}
                            )
                            immediate_id = _text(
                                selected_metadata.get("superseded_by")
                            )
                            immediate = con.execute(
                                "SELECT state,metadata_json FROM records "
                                "WHERE scope_id=? AND memory_id=?",
                                (scope_id, immediate_id),
                            ).fetchone() if immediate_id else None
                            immediate_metadata = (
                                _json(immediate[1]) if immediate is not None else {}
                            )
                            immediate_overwrite = (
                                _text(selected[0]) == "superseded"
                                and immediate is not None
                                and _text(immediate[0]) == "challenged"
                                and isinstance(immediate_metadata, Mapping)
                                and _text(immediate_metadata.get("message_id"))
                                == _text(job.get("message_id"))
                                and _text(
                                    immediate_metadata.get(
                                        "reconciliation_decision"
                                    )
                                )
                                == "challenge"
                            )
                        if immediate_overwrite:
                            issues.append(
                                f"{path}: challenge decision was overwritten by graph policy: {job_id}"
                            )
                        if challenge_artifact is not None:
                            artifact = challenge_artifact[1]
                            journal = challenge_journal_by_job.get(job_id)
                            incoming_id = _text(
                                artifact.get("incoming_memory_id")
                            )
                            migrated_incoming = con.execute(
                                "SELECT state,slot_key,supersedes_json,metadata_json "
                                "FROM records WHERE scope_id=? AND memory_id=?",
                                (scope_id, incoming_id),
                            ).fetchone() if "records" in tables else None
                            migrated_incoming_metadata = (
                                _json(migrated_incoming[3])
                                if migrated_incoming is not None
                                else {}
                            )
                            migrated_supersedes = (
                                _json(migrated_incoming[2])
                                if migrated_incoming is not None
                                else []
                            )
                            request = _json(job.get("request_json"))
                            cited = (
                                request.get("new_cited_assertion")
                                if isinstance(request, Mapping)
                                else None
                            )
                            binding_mode = _text(
                                artifact.get("assertion_binding_mode")
                            )
                            if binding_mode == "original_assertion_index":
                                binding_mode_valid = (
                                    isinstance(
                                        migrated_incoming_metadata, Mapping
                                    )
                                    and int(
                                        migrated_incoming_metadata.get(
                                            "llm_write_proposal_index", -1
                                        )
                                    )
                                    == int(job.get("assertion_index") or 0)
                                )
                            else:
                                binding_mode_valid = (
                                    migrated_incoming is not None
                                    and isinstance(request, Mapping)
                                    and isinstance(cited, Mapping)
                                    and _text(migrated_incoming[1])
                                    == _text(request.get("canonical_slot_key"))
                                    and _text(
                                        migrated_incoming_metadata.get(
                                            "source_span"
                                        )
                                    )
                                    == _text(cited.get("evidence_quote"))
                                    and _text(
                                        migrated_incoming_metadata.get(
                                            "memory_type"
                                        )
                                    )
                                    == _text(cited.get("memory_type"))
                                    and _text(
                                        migrated_incoming_metadata.get(
                                            "entity_key"
                                        )
                                    )
                                    == _text(cited.get("entity_key"))
                                    and _text(
                                        migrated_incoming_metadata.get(
                                            "attribute_key"
                                        )
                                    )
                                    == _text(cited.get("attribute_key"))
                                )
                            if (
                                journal is None
                                or _text(response.get("slot_decision"))
                                != "bind_existing"
                                or _text(response.get("decision")) != "challenge"
                                or _text(artifact.get("selected_memory_id"))
                                != selected_memory_id
                                or _text(artifact.get("resolved_memory_id"))
                                != resolved_memory_id
                                or _text(artifact.get("scope_id")) != scope_id
                                or _text(artifact.get("message_id"))
                                != _text(job.get("message_id"))
                                or int(artifact.get("assertion_index", -1))
                                != int(job.get("assertion_index") or 0)
                                or selected is None
                                or not binding_mode_valid
                                or not _valid_active_to_superseded_lifecycle(
                                    con, scope_id, resolved_memory_id
                                )
                                or migrated_incoming is None
                                or _text(migrated_incoming[0]) != "challenged"
                                or not isinstance(migrated_supersedes, list)
                                or resolved_memory_id
                                in {_text(item) for item in migrated_supersedes}
                                or not isinstance(
                                    migrated_incoming_metadata, Mapping
                                )
                                or _text(
                                    migrated_incoming_metadata.get(
                                        "conflict_action"
                                    )
                                )
                                != "challenge"
                                or _text(
                                    migrated_incoming_metadata.get(
                                        "conflict_reason"
                                    )
                                )
                                != "v4_reconciliation_challenge"
                            ):
                                issues.append(
                                    f"{path}: challenge migration does not match graph/job state: {job_id}"
                                )
                    historical_artifact = historical_recovery_by_job.get(job_id)
                    if historical_artifact is not None:
                        seen_historical_recovery_jobs.add(job_id)
                        if (
                            not isinstance(response, Mapping)
                            or _text(job.get("batch_id"))
                            != _text(historical_artifact[1].get("batch_id"))
                            or _text(response.get("selected_memory_id"))
                            != _text(
                                historical_artifact[1].get("selected_memory_id")
                            )
                            or _text(response.get("decision"))
                            != _text(historical_artifact[1].get("decision"))
                        ):
                            issues.append(
                                f"{path}: historical binding recovery differs from frozen job: {job_id}"
                            )
                        if (
                            _text(historical_artifact[1].get("binding_mode"))
                            == "unique_active_semantic_equivalent"
                            and "records" in tables
                        ):
                            if not _valid_active_to_superseded_lifecycle(
                                con,
                                _text(job.get("scope_id")),
                                _text(
                                    historical_artifact[1].get(
                                        "resolved_memory_id"
                                    )
                                ),
                            ):
                                issues.append(
                                    f"{path}: historical binding recovery resolved leaf has no valid active-to-superseded lifecycle: {job_id}"
                                )
                    metadata = _json(job.get("response_metadata_json"))
                    if not (
                        isinstance(metadata, Mapping)
                        and metadata.get("raw_response_revalidated") is True
                    ):
                        continue
                    artifact = reconciliation_revalidation_by_job.get(job_id)
                    if artifact is None:
                        issues.append(
                            f"{path}: revalidated reconciliation lacks recovery artifact: {job_id}"
                        )
                        continue
                    response_json = _text(job.get("response_json"))
                    if (
                        not response_json
                        or _text(
                            artifact[1].get("normalized_adjudication_sha256")
                        )
                        != _sha(response_json)
                    ):
                        issues.append(
                            f"{path}: revalidated reconciliation response hash differs: {job_id}"
                        )
            for row in _rows(con, "v4_batch_journal"):
                request = _json(row.get("request_json"))
                response_metadata = _json(row.get("response_metadata_json"))
                for marker in _artifact_markers(request) + _artifact_markers(response_metadata):
                    issues.append(f"{path}: writer sidecar contains fallback/retry marker at {marker}")
                if isinstance(response_metadata, Mapping) and response_metadata.get("raw_response_revalidated") is True:
                    batch_id = _text(row.get("batch_id"))
                    artifact = revalidation_by_batch.get(batch_id)
                    if artifact is None:
                        issues.append(f"{path}: revalidated batch lacks recovery artifact: {batch_id}")
                    elif _text(artifact[1].get("validated_response_sha256")) != _text(row.get("response_sha256")):
                        issues.append(f"{path}: revalidated response hash differs: {batch_id}")
                if (
                    isinstance(response_metadata, Mapping)
                    and response_metadata.get("validated_batch_commit_recovered") is True
                ):
                    batch_id = _text(row.get("batch_id"))
                    artifact = validated_recovery_by_batch.get(batch_id)
                    if artifact is None:
                        issues.append(
                            f"{path}: validated batch recovery lacks a recovery artifact: {batch_id}"
                        )
                    elif _text(artifact[1].get("response_sha256")) != _text(
                        row.get("response_sha256")
                    ):
                        issues.append(
                            f"{path}: validated batch recovery response hash differs: {batch_id}"
                        )
                if not isinstance(request, Mapping):
                    issues.append(f"{path}: batch request is not an object")
                    continue
                forbidden = _has_forbidden_input(request)
                if forbidden:
                    issues.append(f"{path}: batch request contains benchmark field {forbidden}")
                messages = request.get("messages")
                if not isinstance(messages, list):
                    issues.append(f"{path}: batch request messages missing")
                    continue
                for item in messages:
                    if not isinstance(item, Mapping):
                        issues.append(f"{path}: batch request message is not an object")
                        continue
                    scope = _text(row.get("scope_id")); key = (scope, _text(item.get("message_id")))
                    source = expected.get(key)
                    if source is None:
                        issues.append(f"{path}: batch request message is not input history: {key}")
                        continue
                    spans = item.get("source_spans")
                    if (
                        not isinstance(spans, list)
                        or not spans
                        or any(
                            not isinstance(span, Mapping)
                            or set(span) != {"span_id", "text"}
                            or not _text(span.get("span_id"))
                            for span in spans
                        )
                    ):
                        issues.append(f"{path}: {key} lacks lossless source spans")
                        continue
                    if "content" in item or "source_tokens" in item or "token_strings" in item:
                        issues.append(f"{path}: {key} repeats full content/token strings")
                    text = "".join(str(span["text"]) for span in spans)
                    if text != source["content"]:
                        issues.append(f"{path}: source spans do not reproduce {key}")
                    span_ids = [_text(span.get("span_id")) for span in spans]
                    if len(set(span_ids)) != len(span_ids):
                        issues.append(f"{path}: source span IDs are not unique for {key}")
    for job_id, (path, _) in historical_recovery_by_job.items():
        if job_id not in seen_historical_recovery_jobs:
            issues.append(
                f"historical binding recovery does not match a persisted job: {path}:{job_id}"
            )
    for job_id, (path, _) in keep_parallel_migration_by_job.items():
        if job_id not in seen_keep_parallel_migration_jobs:
            issues.append(
                f"keep_parallel migration does not match a persisted job: {path}:{job_id}"
            )
        if job_id not in seen_keep_parallel_journal_jobs:
            issues.append(
                f"keep_parallel migration lacks a committed SQLite journal: {path}:{job_id}"
            )
    for job_id, (path, _) in challenge_migration_by_job.items():
        if job_id not in seen_challenge_migration_jobs:
            issues.append(
                f"challenge migration does not match a persisted job: {path}:{job_id}"
            )
        if job_id not in seen_challenge_journal_jobs:
            issues.append(
                f"challenge migration lacks a committed SQLite journal: {path}:{job_id}"
            )


def _audit_source_before_api(
    db_paths: list[Path],
    expected: Mapping[tuple[str, str], Mapping[str, Any]],
    source_messages: Mapping[SourceMessageKey, Sequence[Mapping[str, Any]]],
    issues: list[str],
) -> None:
    """Use actual graph turn events to establish source persistence before enrichment."""
    for path in db_paths:
        with sqlite3.connect(path) as con:
            con.row_factory = sqlite3.Row
            tables = _tables(con)
            if "v4_batch_journal" not in tables:
                continue
            if "audit_turn_log" not in tables:
                issues.append(f"{path}: audit_turn_log is missing; source-before-API is unproven")
                continue
            source_turn_events: set[tuple[str, str, str]] = set()
            for event in _rows(con, "audit_turn_log"):
                payload = _json(event.get("payload_json"))
                if isinstance(payload, Mapping):
                    metadata = (
                        payload.get("metadata")
                        if isinstance(payload.get("metadata"), Mapping)
                        else {}
                    )
                    source_record_id = _text(metadata.get("source_record_id"))
                    record_ids = {
                        _text(value)
                        for value in (
                            payload.get("record_ids")
                            if isinstance(payload.get("record_ids"), list)
                            else []
                        )
                    }
                    if source_record_id and source_record_id in record_ids:
                        source_turn_events.add(
                            (
                                _text(event.get("scope_id")),
                                _text(metadata.get("message_id")),
                                source_record_id,
                            )
                        )
            source_journal = {
                (_text(row.get("scope_id")), _text(row.get("message_id"))): row
                for row in _rows(con, "v4_source_journal")
            } if "v4_source_journal" in tables else {}
            for batch in _rows(con, "v4_batch_journal"):
                request = _json(batch.get("request_json"))
                if not isinstance(request, Mapping) or not batch.get("api_started_at") and _text(batch.get("status")) == "prepared":
                    continue
                scope = _text(batch.get("scope_id"))
                for item in list(request.get("messages") or []):
                    message_id = _text(item.get("message_id")) if isinstance(item, Mapping) else ""
                    key = (scope, message_id)
                    source = _source_for_message(
                        source_messages,
                        db_path=path,
                        scope_id=scope,
                        message_id=key[1],
                    )
                    if source is None:
                        issues.append(f"{path}: API batch lacks actual source record {key}")
                        continue
                    journal = source_journal.get(key)
                    if journal is None:
                        issues.append(f"{path}: source journal is missing for API batch message {key}")
                    elif _text(batch.get("api_started_at")):
                        persisted_at = _text(journal.get("source_persisted_at"))
                        if not persisted_at or persisted_at > _text(batch.get("api_started_at")):
                            issues.append(
                                f"{path}: source persistence was not recorded before API start for {key}"
                            )
                    if (
                        scope,
                        message_id,
                        _text(source.get("source_record_id")),
                    ) not in source_turn_events:
                        issues.append(f"{path}: actual source turn event is missing for API batch message {key}")


def audit_run(
    run_dir: Path,
    *,
    output: Path | None = None,
    worker_db_specs: Iterable[str] = (),
    retrieval_dir: Path | None = None,
    build_only: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    issues: list[str] = []
    try:
        input_path, input_rows = _load_input(run_dir)
        expected, excluded_empty = _input_message_inventory(input_rows)
    except AuditError as exc:
        report = {"schema_version": "tmcra.v4.chain-audit.1", "status": "failed", "passed": False, "issues": [str(exc)]}
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    worker_db_specs = list(worker_db_specs)
    _audit_source_exclusions(run_dir, excluded_empty, issues)
    db_groups = _parse_worker_db_specs(run_dir, worker_db_specs) if worker_db_specs else discover_worker_databases(run_dir)
    db_paths = sorted({path for paths in db_groups.values() for path in paths})
    if not db_paths:
        issues.append("no per-worker SQLite databases discovered")
    expected_scopes = {key[0] for key in expected}
    sources, source_errors, source_tables = _source_rows(db_paths, expected_scopes)
    source_messages, scoped_source_messages = _source_message_indices(sources)
    issues.extend(source_errors)
    graph_tables: set[str] = set()
    for path in db_paths:
        with sqlite3.connect(path) as con:
            con.row_factory = sqlite3.Row
            tables = _tables(con)
            graph_tables.update(tables)
            shadow_tables = sorted(FORBIDDEN_SHADOW_TABLES.intersection(tables))
            if shadow_tables:
                issues.append(
                    f"{path}: forbidden graph shadow tables exist: {','.join(shadow_tables)}"
                )
            if "v4_batch_journal" in tables:
                missing = sorted(REQUIRED_GRAPH_TABLES - tables)
                if missing:
                    issues.append(
                        f"{path}: writer database lacks actual graph tables: {','.join(missing)}"
                    )
                for batch_row in _rows(con, "v4_batch_journal"):
                    if _text(batch_row.get("status")) != "committed":
                        issues.append(
                            f"{path}: writer batch {_text(batch_row.get('batch_id'))} is not committed"
                        )
                if "v4_source_journal" not in tables:
                    issues.append(f"{path}: v4_source_journal is missing")
                else:
                    for journal in _rows(con, "v4_source_journal"):
                        key = (_text(journal.get("scope_id")), _text(journal.get("message_id")))
                        source = _source_for_message(
                            source_messages,
                            db_path=path,
                            scope_id=key[0],
                            message_id=key[1],
                        )
                        if _text(journal.get("status")) != "enriched":
                            issues.append(f"{path}: source journal is not enriched for {key}")
                        if source is None or _text(journal.get("source_record_id")) != _text(source.get("source_record_id")):
                            issues.append(f"{path}: source journal does not target actual graph source for {key}")
                        if not _text(journal.get("source_persisted_at")):
                            issues.append(f"{path}: source persistence timestamp is missing for {key}")
    missing_graph_tables = sorted(REQUIRED_GRAPH_TABLES - graph_tables)
    if missing_graph_tables:
        issues.append("actual TMCRA graph tables are missing: " + ",".join(missing_graph_tables))
    if not source_tables:
        issues.append("actual TMCRA records contain no immutable source records")
    if len(sources) != len(expected):
        issues.append(f"source record count {len(sources)} does not equal input message count {len(expected)}")
    for key, message in expected.items():
        source_matches = list(scoped_source_messages.get(key, ()))
        if len(source_matches) != 1:
            source = None
            if len(source_matches) > 1:
                issues.append(f"ambiguous immutable Source composite identity for {key}")
            issues.append(f"missing immutable source record {key}")
        else:
            source = source_matches[0]
        if source is not None and source["content"] != message["content"]:
            issues.append(f"immutable source content differs for {key}")
        elif source is not None and source["role"] and source["role"] != message["role"]:
            issues.append(f"immutable source role differs for {key}")
        elif source is not None and source["status"] != "enriched":
            issues.append(f"immutable source is not enriched for {key}")
    _audit_writer_input(run_dir, expected, db_paths, issues)
    _audit_source_before_api(db_paths, expected, source_messages, issues)

    fast: dict[Identity, dict[str, Any]] = {}
    all_fast: dict[Identity, dict[str, Any]] = {}
    records: dict[Identity, dict[str, Any]] = {}
    interactions: dict[Identity, dict[str, Any]] = {}
    jobs: list[dict[str, Any]] = []
    edges: list[tuple[Path, dict[str, Any]]] = []
    slow_candidates: list[dict[str, Any]] = []
    slow_provenance: list[tuple[Path, dict[str, Any]]] = []
    slow_patch_routes: dict[tuple[str, str], str] = {}
    slow_patch_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    slow_graph_jobs: list[tuple[Path, dict[str, Any]]] = []
    slow_graph_patches: list[tuple[Path, dict[str, Any]]] = []
    slow_graph_patch_operations: list[tuple[Path, dict[str, Any]]] = []
    subject_attribution_audits: list[tuple[Path, dict[str, Any]]] = []
    for path in db_paths:
        with sqlite3.connect(path) as con:
            con.row_factory = sqlite3.Row
            tables = _tables(con)
            if "v4_reconciliation_jobs" in tables:
                jobs.extend({**row, "db": _db_path(path)} for row in _rows(con, "v4_reconciliation_jobs"))
            if "slow_graph_jobs" in tables:
                slow_graph_jobs.extend((path, row) for row in _rows(con, "slow_graph_jobs"))
            if "v4_subject_attribution_audits" in tables:
                subject_attribution_audits.extend(
                    (path, row)
                    for row in _rows(con, "v4_subject_attribution_audits")
                )
            if "records" in tables:
                for row in _rows(con, "records"):
                    scope = _text(row.get("scope_id"))
                    memory_id = _text(row.get("memory_id"))
                    record_identity = _identity(path, scope, memory_id)
                    record = dict(row)
                    record["metadata"] = _metadata(row)
                    record["db"] = _db_path(path)
                    records[record_identity] = record
                    meta = record["metadata"]
                    variant = _text(meta.get("content_variant"))
                    fast_like = (
                        variant == PRODUCTION_FAST_VARIANT
                        or _text(meta.get("node_kind")) in {PRODUCTION_FAST_NODE_KIND, "fast_assertion"}
                    )
                    if fast_like:
                        if not _production_fast_predicate(row, meta):
                            issues.append(
                                f"{path}: Fast record {scope}/{memory_id} fails the production strict predicate"
                            )
                        else:
                            all_fast[record_identity] = record
                            if _text(row.get("state")) in CURRENT_FAST_RECORD_STATES:
                                fast[record_identity] = record
                    if variant == "product_interaction" or meta.get("node_kind") in {"atomic_interaction", "product_interaction"}:
                        interactions[record_identity] = record
                    if variant == "slow_memory_capsule" or meta.get("memory_layer") == "slow":
                        slow_candidates.append(record)
            if "memory_edges" in tables:
                edges.extend((path, row) for row in _rows(con, "memory_edges"))
            if "slow_graph_provenance" in tables:
                slow_provenance.extend((path, row) for row in _rows(con, "slow_graph_provenance"))
            if "slow_graph_patches" in tables:
                for row in _rows(con, "slow_graph_patches"):
                    slow_graph_patches.append((path, row))
                    patch_metadata = _json(row.get("call_metadata_json"))
                    patch_key = (_db_path(path), _text(row.get("patch_id")))
                    slow_patch_routes[patch_key] = _text(patch_metadata.get("route"))
                    if isinstance(patch_metadata, Mapping):
                        slow_patch_metadata[patch_key] = dict(patch_metadata)
            if "slow_graph_patch_operations" in tables:
                slow_graph_patch_operations.extend(
                    (path, row) for row in _rows(con, "slow_graph_patch_operations")
                )
            if "slow_graph_attempts" in tables:
                for row in _rows(con, "slow_graph_attempts"):
                    metadata = _json(row.get("call_metadata_json"))
                    if _artifact_markers(metadata):
                        issues.append(f"{path}: slow call contains fallback/retry marker")
                    route = _text(metadata.get("route"))
                    if not _slow_pro_route_is_justified(metadata):
                        issues.append(f"{path}: slow Pro call lacks a conflict route")
            if "slow_graph_jobs" in tables:
                for job in _rows(con, "slow_graph_jobs"):
                    if _text(job.get("status")) == "retryable" or _artifact_markers(job.get("last_error", "")):
                        issues.append(f"{path}: slow graph job contains a fallback/retry marker")

            if "slot_heads" in tables:
                for head in _rows(con, "slot_heads"):
                    scope = _text(head.get("scope_id")); memory_id = _text(head.get("memory_id"))
                    if not scope or not memory_id:
                        issues.append(f"{path}: slot_heads contains an incomplete head")
                    if "records" in tables:
                        target = con.execute(
                            "SELECT state FROM records WHERE scope_id=? AND memory_id=?",
                            (scope, memory_id),
                        ).fetchone()
                        if target is None:
                            issues.append(f"{path}: slot_heads targets missing actual record {memory_id}")
                        elif _text(target[0]) not in {
                            "active",
                            "parallel_active",
                            "promoted",
                        }:
                            issues.append(
                                f"{path}: slot_heads targets non-active record {memory_id}"
                            )

    _audit_current_v47_patch_reconciliation(
        jobs=slow_graph_jobs,
        patches=slow_graph_patches,
        operation_rows=slow_graph_patch_operations,
        slow_records=slow_candidates,
        issues=issues,
    )

    subject_attribution_decision_count = 0
    subject_attribution_quarantine_count = 0
    subject_attribution_superseded_count = 0
    subject_audit_index = {
        (_db_path(path), _text(row.get("audit_id"))): row
        for path, row in subject_attribution_audits
        if _text(row.get("audit_id"))
    }
    active_subject_messages: set[tuple[str, str, str]] = set()
    for path, row in subject_attribution_audits:
        scope_id = _text(row.get("scope_id"))
        audit_id = _text(row.get("audit_id"))
        status = _text(row.get("status"))
        if status == "superseded":
            prefix = "superseded_by:"
            error = _text(row.get("error"))
            target_id = error[len(prefix):] if error.startswith(prefix) else ""
            target = subject_audit_index.get((_db_path(path), target_id))
            if (
                not target_id
                or target is None
                or _text(target.get("status")) != "completed"
                or _text(target.get("scope_id")) != scope_id
                or _text(target.get("message_id")) != _text(row.get("message_id"))
            ):
                issues.append(
                    f"{path}: superseded subject-attribution audit has no active successor"
                )
            else:
                subject_attribution_superseded_count += 1
            continue
        if (
            status != "completed"
            or _text(row.get("prompt_version"))
            != SUBJECT_ATTRIBUTION_PROMPT_VERSION
            or _text(row.get("model")) != "deepseek-v4-pro"
            or not audit_id
            or not _text(row.get("request_sha256"))
            or not _text(row.get("response_sha256"))
        ):
            issues.append(f"{path}: subject-attribution journal is incomplete or drifted")
            continue
        message_identity = (
            _db_path(path),
            scope_id,
            _text(row.get("message_id")),
        )
        if message_identity in active_subject_messages:
            issues.append(f"{path}: multiple active subject-attribution audits for one message")
            continue
        active_subject_messages.add(message_identity)
        decisions = _json(row.get("decisions_json"))
        if not isinstance(decisions, list) or not decisions:
            issues.append(f"{path}: subject-attribution decisions are missing")
            continue
        for decision in decisions:
            if not isinstance(decision, Mapping):
                issues.append(f"{path}: subject-attribution decision is malformed")
                continue
            memory_id = _text(decision.get("memory_id"))
            value = _text(decision.get("decision"))
            bridge_quote = _text(decision.get("chat_user_bridge_quote"))
            record = records.get(_identity(path, scope_id, memory_id))
            if value not in {
                "keep_user",
                "quarantine_third_party",
                "quarantine_ambiguous",
            } or record is None:
                issues.append(f"{path}: subject-attribution decision identity is invalid")
                continue
            subject_attribution_decision_count += 1
            metadata = record["metadata"]
            if value == "keep_user":
                source_id = _text(metadata.get("source_record_id"))
                source = sources.get(_identity(path, scope_id, source_id))
                if (
                    not bridge_quote
                    or bridge_quote == _fast_quote(metadata)
                    or source is None
                    or bridge_quote not in _text(source.get("content"))
                ):
                    issues.append(
                        f"{path}: keep_user subject attribution lacks an exact Source bridge"
                    )
            elif bridge_quote:
                issues.append(
                    f"{path}: quarantined subject attribution carries a chat-user bridge"
                )
            if (
                _text(metadata.get("subject_attribution_audit_id")) != audit_id
                or _text(metadata.get("subject_attribution_decision")) != value
                or _text(
                    metadata.get("subject_attribution_chat_user_bridge_quote")
                )
                != bridge_quote
                or _text(metadata.get("subject_attribution_prompt_version"))
                != SUBJECT_ATTRIBUTION_PROMPT_VERSION
            ):
                issues.append(f"{path}: subject-attribution record binding changed")
            if value != "keep_user":
                subject_attribution_quarantine_count += 1
                if (
                    _text(record.get("state")) != "quarantined"
                    or metadata.get("excluded_from_retrieval") is not True
                ):
                    issues.append(
                        f"{path}: quarantined subject-attribution record is still current"
                    )

    edge_keys: set[tuple[str, str, str, str, str]] = set()
    for path, edge in edges:
        scope = _text(edge.get("scope_id"))
        edge_db = _text(edge.get("db_path"))
        if edge_db and _db_path(edge_db) != _db_path(path):
            issues.append(f"{path}: edge has a cross-database identity")
        source_identity = _identity(path, scope, edge.get("source_memory_id"))
        target_identity = _identity(path, scope, edge.get("target_memory_id"))
        edge_keys.add(
            (*source_identity, target_identity[2], _text(edge.get("edge_type")))
        )
        if source_identity not in records or target_identity not in records:
            issues.append(
                f"{path}: memory edge targets a missing composite record identity: {source_identity}->{target_identity}"
            )

    for leaf_identity, leaf in fast.items():
        db_path, scope_id, leaf_id = leaf_identity
        meta = leaf["metadata"]
        durability = _text(leaf.get("durability")) or _text(meta.get("durability"))
        if durability.casefold() not in {
            "durable",
            "long_term",
            "long-term",
            "hard",
            "persistent",
            "episodic",
            "uncertain",
        }:
            issues.append(f"fast leaf {leaf_id}: invalid or missing durability")
        source_id = _text(meta.get("source_record_id"))
        source = _source_for_reference(
            sources,
            db_path=db_path,
            scope_id=scope_id,
            source_record_id=source_id,
        )
        if not source or not source_id:
            issues.append(f"fast leaf {leaf_id}: source parent is not exact")
        elif _text(meta.get("message_id")) != _text(source.get("message_id")):
            issues.append(f"fast leaf {leaf_id}: message identity is not bound to immutable Source")
        quote = _fast_quote(meta)
        if not source or not _source_span_quote(
            source,
            meta.get("evidence_char_start"),
            meta.get("evidence_char_end"),
            quote,
            label=f"fast leaf {leaf_id}",
            issues=issues,
        ):
            issues.append(f"fast leaf {leaf_id}: evidence is not grounded in immutable source")
        parent = meta.get("source_parent") or _slow_source_parent(meta)
        validated_parent = _validate_source_parent(
            parent,
            db_path=db_path,
            scope_id=scope_id,
            sources=sources,
            issues=issues,
            label=f"fast leaf {leaf_id}",
            quote=quote,
        )
        expected_parent = (
            _slow_source_parent(meta)
            if isinstance(parent, Mapping)
            and set(parent) == SLOW_SOURCE_PARENT_KEYS
            else _fast_source_parent(meta)
        )
        if validated_parent is not None and dict(parent) != expected_parent:
            issues.append(f"fast leaf {leaf_id}: source_parent disagrees with record span or source identity")
        if (*leaf_identity, source_id, "grounded_in") not in edge_keys:
            issues.append(f"fast leaf {leaf_id}: missing grounded_in edge to exact source parent")
        raw_provenance = meta.get("provenance") or []
        if isinstance(raw_provenance, Mapping):
            provenance_entries = [raw_provenance]
        elif isinstance(raw_provenance, list):
            provenance_entries = raw_provenance
        else:
            provenance_entries = []
            issues.append(f"fast leaf {leaf_id}: provenance is not an object or array")
        for provenance in provenance_entries:
            if not isinstance(provenance, Mapping):
                issues.append(f"fast leaf {leaf_id}: malformed provenance")
                continue
            p_source = _text(provenance.get("source_record_id"))
            declared_db = provenance.get("db_path")
            declared_scope = _text(provenance.get("scope_id"))
            if declared_db and _db_path(declared_db) != db_path:
                issues.append(f"fast leaf {leaf_id}: provenance crosses database identity")
                continue
            if declared_scope and declared_scope != scope_id:
                issues.append(f"fast leaf {leaf_id}: provenance crosses scope identity")
                continue
            p_source_row = _source_for_reference(
                sources,
                db_path=db_path,
                scope_id=scope_id,
                source_record_id=p_source,
            )
            if not p_source_row:
                issues.append(f"fast leaf {leaf_id}: provenance target is invalid")
                continue
            if _text(provenance.get("message_id")) != _text(p_source_row.get("message_id")):
                issues.append(f"fast leaf {leaf_id}: provenance message identity is invalid")
            source_turn_index = provenance.get("source_turn_index")
            if source_turn_index is not None:
                try:
                    turn_matches = int(source_turn_index) == int(
                        p_source_row.get("turn_index")
                    )
                except (TypeError, ValueError):
                    turn_matches = False
                if not turn_matches:
                    issues.append(f"fast leaf {leaf_id}: provenance turn identity is invalid")
            start = provenance.get("source_char_start")
            end = provenance.get("source_char_end")
            if start is None and end is None:
                issues.append(
                    f"fast leaf {leaf_id}: provenance exact source offsets are missing"
                )
            elif start is None or end is None:
                issues.append(
                    f"fast leaf {leaf_id}: provenance exact source offsets are incomplete"
                )
            else:
                _source_span_quote(
                    p_source_row,
                    start,
                    end,
                    provenance.get("evidence_quote"),
                    label=f"fast leaf {leaf_id} provenance",
                    issues=issues,
                )
            if (*leaf_identity, p_source, "grounded_in") not in edge_keys:
                issues.append(f"fast leaf {leaf_id}: provenance grounded_in edge is missing")

    for interaction_identity, row in interactions.items():
        _, interaction_scope, interaction_id = interaction_identity
        meta = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else _metadata(row)
        source = _source_for_message(
            source_messages,
            db_path=interaction_identity[0],
            scope_id=interaction_scope,
            message_id=meta.get("message_id"),
        )
        if not interaction_id or not source:
            issues.append(f"interaction target is invalid: {interaction_id}")
        interaction = meta
        if isinstance(interaction, Mapping) and _identity(interaction_identity[0], interaction_scope, interaction.get("source_record_id")) not in {None, _identity(interaction_identity[0], interaction_scope, source.get("source_record_id") if source else "") }:
            issues.append(f"interaction {interaction_id}: source target is invalid")
        history = meta.get("resolution_history", [])
        if not isinstance(history, list):
            issues.append(f"interaction {interaction_id}: resolution history is not a list")
        else:
            for item in history:
                if not isinstance(item, Mapping):
                    issues.append(f"interaction {interaction_id}: resolution target is invalid")
                elif not _source_for_message(source_messages, db_path=interaction_identity[0], scope_id=interaction_scope, message_id=item.get("message_id")):
                    issues.append(f"interaction {interaction_id}: resolution message target is invalid")
                elif _source_for_reference(sources, db_path=interaction_identity[0], scope_id=interaction_scope, source_record_id=item.get("source_record_id")) is None:
                    issues.append(f"interaction {interaction_id}: resolution source target is invalid")
    for job in jobs:
        request = _json(job.get("request_json"))
        if _artifact_markers(request):
            issues.append("writer reconciliation request contains fallback/retry marker")
        job_db = _db_path(job.get("db"))
        job_scope = _text(job.get("scope_id"))
        candidate_ids: set[Identity] = set()
        for current in list(request.get("candidate_cited_leaves") or []) if isinstance(request, Mapping) else []:
            if isinstance(current, Mapping):
                reference = _reference_identity(
                    current,
                    db_path=job_db,
                    scope_id=job_scope,
                )
                if reference is not None:
                    candidate_ids.add(reference)
            else:
                reference = None
            if reference is None or reference not in records:
                issues.append(
                    f"reconciliation job {_text(job.get('job_id'))}: historical candidate record is missing"
                )
        if _text(job.get("status")) == "completed" and not _text(job.get("decision")):
            issues.append(f"reconciliation job {_text(job.get('job_id'))}: completed without decision")
        elif _text(job.get("status")) != "completed":
            issues.append(
                f"reconciliation job {_text(job.get('job_id'))}: status is not completed"
            )
        if _text(job.get("status")) == "completed":
            response = _json(job.get("response_json"))
            if not isinstance(response, Mapping):
                issues.append(f"reconciliation job {_text(job.get('job_id'))}: response is not an object")
            elif (
                _text(response.get("slot_decision")) == "bind_existing"
                and _reference_identity(
                    response.get("selected_memory_id"),
                    db_path=job_db,
                    scope_id=job_scope,
                ) not in candidate_ids
            ):
                issues.append(f"reconciliation job {_text(job.get('job_id'))}: selected slot was not a supplied candidate")

    # Capsules are append-only. Only one row may represent the latest revision;
    # older revisions are historical data and must not satisfy current retrieval.
    slow_by_capsule: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in slow_candidates:
        db_path = _db_path(record["db"])
        scope_id = _text(record.get("scope_id"))
        metadata = record["metadata"]
        capsule_id = _text(metadata.get("capsule_id"))
        try:
            revision = int(metadata.get("revision"))
        except (TypeError, ValueError):
            revision = -1
        if not capsule_id or revision < 1:
            issues.append(f"slow record {_text(record.get('memory_id'))}: capsule identity/revision is invalid")
            continue
        record["slow_group"] = (db_path, scope_id, capsule_id)
        record["slow_revision"] = revision
        slow_by_capsule[record["slow_group"]].append(record)

    slow_records: list[dict[str, Any]] = []
    for group, revisions in slow_by_capsule.items():
        latest_revision = max(record["slow_revision"] for record in revisions)
        latest = [record for record in revisions if record["slow_revision"] == latest_revision]
        if len(latest) != 1:
            issues.append(f"slow capsule {group[2]} does not have a unique latest revision")
            continue
        record = latest[0]
        if (
            _text(record.get("state")) != "active"
            or _text(record["metadata"].get("status"))
            not in {"active", "challenged"}
        ):
            continue
        slow_records.append(record)

    slow_current_groups = {record["slow_group"] for record in slow_records}
    slow_semantic_integrity_issues: list[dict[str, Any]] = []
    slow_semantic_integrity_issues.extend(
        _active_slow_partition_issues(slow_records, slow_patch_metadata)
    )
    slow_support_durability = {
        "durable",
        "long_term",
        "long-term",
        "hard",
        "persistent",
    }
    slow_support_state = {
        "active",
        "parallel_active",
        "promoted",
        "challenged",
    }
    for record in slow_records:
        db_path = _db_path(record["db"])
        scope_id = _text(record.get("scope_id"))
        meta = record["metadata"]
        summary = " ".join(_text(record.get("value")).split())
        structured_summary = False
        if summary[:1] in "[{":
            try:
                structured_summary = isinstance(
                    json.loads(summary), (dict, list)
                )
            except json.JSONDecodeError:
                structured_summary = False
        if (
            not summary
            or len(summary) > 4096
            or structured_summary
            or SLOW_OPERATIONAL_SUMMARY_RE.search(summary)
            or SLOW_GENERIC_SUMMARY_RE.fullmatch(summary)
        ):
            slow_semantic_integrity_issues.append(
                {
                    "code": "invalid_semantic_summary",
                    "db": db_path,
                    "scope_id": scope_id,
                    "capsule_id": _text(meta.get("capsule_id")),
                    "revision": meta.get("revision"),
                }
            )
        route = slow_patch_routes.get(
            (db_path, _text(meta.get("patch_id"))), ""
        )
        prior_counterevidence: set[Identity] = set()
        prior_revision = int(record["slow_revision"]) - 1
        if prior_revision >= 1:
            prior = [
                candidate
                for candidate in slow_by_capsule[record["slow_group"]]
                if int(candidate["slow_revision"]) == prior_revision
            ]
            if len(prior) == 1:
                for prior_claim in list(
                    prior[0]["metadata"].get("claims") or []
                ):
                    if not isinstance(prior_claim, Mapping):
                        continue
                    for raw_reference in list(
                        prior_claim.get("counterevidence") or []
                    ):
                        reference = _reference_identity(
                            raw_reference,
                            db_path=db_path,
                            scope_id=scope_id,
                        )
                        if reference is not None:
                            prior_counterevidence.add(reference)
        claims = meta.get("claims")
        if not isinstance(claims, list) or not claims:
            issues.append(f"slow record {_text(record.get('memory_id'))}: claims are missing")
            continue
        patch_metadata = slow_patch_metadata.get(
            (db_path, _text(meta.get("patch_id"))), {}
        )
        if _current_summary_contract(meta, patch_metadata):
            try:
                expected_summary = _lossless_summary_projection(claims)
            except AuditError as exc:
                slow_semantic_integrity_issues.append(
                    {
                        "code": "invalid_lossless_summary_claims",
                        "db": db_path,
                        "scope_id": scope_id,
                        "capsule_id": _text(meta.get("capsule_id")),
                        "revision": record["slow_revision"],
                        "error": str(exc),
                    }
                )
            else:
                if _text(record.get("value")) != expected_summary:
                    slow_semantic_integrity_issues.append(
                        {
                            "code": "stored_summary_not_lossless_projection",
                            "db": db_path,
                            "scope_id": scope_id,
                            "capsule_id": _text(meta.get("capsule_id")),
                            "revision": record["slow_revision"],
                            "expected_summary": expected_summary,
                            "stored_summary": _text(record.get("value")),
                        }
                    )
        claim_roles: list[tuple[str, set[Identity], set[Identity]]] = []
        for claim in claims:
            if not isinstance(claim, Mapping):
                issues.append(f"slow record {_text(record.get('memory_id'))}: malformed claim")
                continue
            claim_id = _text(claim.get("claim_id"))
            claim_slot = _text(claim.get("canonical_slot"))
            claim_text = _normal_text(claim.get("text"))
            support_references = {
                reference
                for raw_reference in list(claim.get("support") or [])
                if (
                    reference := _reference_identity(
                        raw_reference, db_path=db_path, scope_id=scope_id
                    )
                )
                is not None
            }
            counter_references = {
                reference
                for raw_reference in list(claim.get("counterevidence") or [])
                if (
                    reference := _reference_identity(
                        raw_reference, db_path=db_path, scope_id=scope_id
                    )
                )
                is not None
            }
            complementary_support_bundle = _controlled_complementary_support_bundle(
                claim_slot,
                support_references,
                all_fast,
            )
            shared_roles = support_references & counter_references
            if shared_roles:
                slow_semantic_integrity_issues.append(
                    {
                        "code": "same_evidence_support_and_counterevidence",
                        "db": db_path,
                        "scope_id": scope_id,
                        "capsule_id": _text(meta.get("capsule_id")),
                        "revision": record["slow_revision"],
                        "claim_id": claim_id,
                        "evidence_ids": sorted(item[2] for item in shared_roles),
                    }
                )
            if meta.get("action") == "create" and counter_references:
                slow_semantic_integrity_issues.append(
                    {
                        "code": "active_create_contains_counterevidence",
                        "db": db_path,
                        "scope_id": scope_id,
                        "capsule_id": _text(meta.get("capsule_id")),
                        "revision": record["slow_revision"],
                        "claim_id": claim_id,
                        "evidence_ids": sorted(
                            item[2] for item in counter_references
                        ),
                    }
                )
            support_text_groups: dict[str, list[str]] = {}
            for reference in sorted(support_references):
                leaf = all_fast.get(reference)
                if leaf is None:
                    continue
                leaf_meta = leaf["metadata"]
                evidence_text = _normal_text(
                    leaf.get("value")
                    or leaf_meta.get("source_span")
                    or leaf_meta.get("raw_content")
                )
                if evidence_text:
                    support_text_groups.setdefault(evidence_text, []).append(
                        reference[2]
                    )
            if len(support_text_groups) > 1 and not complementary_support_bundle:
                slow_semantic_integrity_issues.append(
                    {
                        "code": "support_distinct_fast_values_merged",
                        "db": db_path,
                        "scope_id": scope_id,
                        "capsule_id": _text(meta.get("capsule_id")),
                        "revision": record["slow_revision"],
                        "claim_id": claim_id,
                        "evidence_groups": [
                            {
                                "normalized_text": text,
                                "evidence_ids": evidence_ids,
                            }
                            for text, evidence_ids in sorted(
                                support_text_groups.items()
                            )
                        ],
                    }
                )
            claim_roles.append(
                (claim_id, support_references, counter_references)
            )
            cited: list[Identity] = []
            for raw_reference in [*(claim.get("support") or []), *(claim.get("counterevidence") or [])]:
                reference = _reference_identity(
                    raw_reference,
                    db_path=db_path,
                    scope_id=scope_id,
                )
                if reference is None or reference not in all_fast:
                    issues.append(
                        f"slow claim {_text(claim.get('claim_id'))}: cited fast leaf does not exist under composite identity: {raw_reference}"
                    )
                else:
                    cited.append(reference)
            for raw_reference in list(claim.get("support") or []):
                reference = _reference_identity(
                    raw_reference,
                    db_path=db_path,
                    scope_id=scope_id,
                )
                leaf = all_fast.get(reference) if reference is not None else None
                if leaf is None:
                    continue
                evidence_slot = _text(
                    leaf["metadata"].get("canonical_slot_key")
                    or leaf["metadata"].get("canonical_slot")
                )
                if evidence_slot != claim_slot and not complementary_support_bundle:
                    slow_semantic_integrity_issues.append(
                        {
                            "code": "support_canonical_slot_mismatch",
                            "db": db_path,
                            "scope_id": scope_id,
                            "capsule_id": _text(meta.get("capsule_id")),
                            "revision": record["slow_revision"],
                            "claim_id": claim_id,
                            "claim_slot": claim_slot,
                            "evidence_id": reference[2],
                            "evidence_slot": evidence_slot,
                        }
                    )
                leaf_meta = leaf["metadata"]
                if not (
                    _text(
                        leaf_meta.get("durability")
                        or leaf_meta.get("durability_class")
                    ).casefold()
                    in slow_support_durability
                    and _text(leaf.get("state")).casefold()
                    in slow_support_state
                ):
                    slow_semantic_integrity_issues.append(
                        {
                            "code": "support_noncurrent_fast_leaf",
                            "db": db_path,
                            "scope_id": scope_id,
                            "capsule_id": _text(meta.get("capsule_id")),
                            "revision": record["slow_revision"],
                            "claim_id": claim_id,
                            "evidence_id": reference[2],
                            "evidence_state": _text(leaf.get("state")),
                        }
                    )
            for raw_reference in list(claim.get("counterevidence") or []):
                reference = _reference_identity(
                    raw_reference,
                    db_path=db_path,
                    scope_id=scope_id,
                )
                leaf = all_fast.get(reference) if reference is not None else None
                if leaf is None:
                    continue
                if _normal_text(leaf.get("value")) == claim_text:
                    slow_semantic_integrity_issues.append(
                        {
                            "code": "counterevidence_identical_to_claim",
                            "db": db_path,
                            "scope_id": scope_id,
                            "capsule_id": _text(meta.get("capsule_id")),
                            "revision": record["slow_revision"],
                            "claim_id": claim_id,
                            "evidence_id": reference[2],
                        }
                    )
                leaf_meta = leaf["metadata"]
                if (
                    route == "flash"
                    and reference not in prior_counterevidence
                    and not bool(leaf_meta.get("counterevidence"))
                    and not bool(leaf_meta.get("is_counterevidence"))
                ):
                    slow_semantic_integrity_issues.append(
                        {
                            "code": "flash_invented_counterevidence",
                            "db": db_path,
                            "scope_id": scope_id,
                            "capsule_id": _text(meta.get("capsule_id")),
                            "revision": record["slow_revision"],
                            "claim_id": claim_id,
                            "evidence_id": reference[2],
                            "route": route,
                        }
                    )
            parents = claim.get("source_parents", meta.get("source_parents"))
            if not isinstance(parents, list) or not parents:
                issues.append(f"slow claim {_text(claim.get('claim_id'))}: source parents missing")
                continue
            for parent in parents:
                _validate_source_parent(
                    parent,
                    db_path=db_path,
                    scope_id=scope_id,
                    sources=sources,
                    issues=issues,
                    label=f"slow claim {_text(claim.get('claim_id'))}",
                )
            for leaf_identity in cited:
                leaf = all_fast[leaf_identity]
                leaf_meta = leaf["metadata"]
                leaf_parent = _slow_source_parent(leaf_meta)
                matching_parents = [
                    parent for parent in parents
                    if isinstance(parent, Mapping)
                    and dict(parent) == dict(leaf_parent)
                ]
                if not matching_parents:
                    issues.append(
                        f"slow claim {_text(claim.get('claim_id'))}: cited Fast source parent coordinates changed"
                    )
                else:
                    _validate_source_parent(
                        matching_parents[0],
                        db_path=db_path,
                        scope_id=scope_id,
                        sources=sources,
                        issues=issues,
                        label=f"slow claim {_text(claim.get('claim_id'))} citation",
                        quote=_fast_quote(leaf_meta),
                    )
        for left_index, (
            left_claim_id,
            left_support,
            left_counter,
        ) in enumerate(claim_roles):
            for (
                right_claim_id,
                right_support,
                right_counter,
            ) in claim_roles[left_index + 1 :]:
                if left_support & right_counter and right_support & left_counter:
                    slow_semantic_integrity_issues.append(
                        {
                            "code": "reciprocal_counterevidence_cycle",
                            "db": db_path,
                            "scope_id": scope_id,
                            "capsule_id": _text(meta.get("capsule_id")),
                            "revision": record["slow_revision"],
                            "claim_ids": sorted(
                                [left_claim_id, right_claim_id]
                            ),
                        }
                    )
    for path, row in slow_provenance:
        db_path = _db_path(path)
        scope_id = _text(row.get("scope_id"))
        group = (db_path, scope_id, _text(row.get("capsule_id")))
        try:
            revision = int(row.get("revision"))
        except (TypeError, ValueError):
            revision = -1
        if group not in slow_current_groups:
            continue
        current = next(record for record in slow_records if record["slow_group"] == group)
        if revision != current["slow_revision"]:
            continue
        reference = _reference_identity(
            row.get("evidence_memory_id"),
            db_path=db_path,
            scope_id=scope_id,
        )
        leaf = all_fast.get(reference) if reference is not None else None
        if leaf is None:
            issues.append(f"slow provenance cites missing Fast leaf under composite identity: {row.get('evidence_memory_id')}")
            continue
        parent = _json(row.get("source_parent_json"))
        _validate_source_parent(
            parent,
            db_path=db_path,
            scope_id=scope_id,
            sources=sources,
            issues=issues,
            label="slow provenance",
            quote=_fast_quote(leaf["metadata"]),
        )
        expected_parent = _slow_source_parent(leaf["metadata"])
        if isinstance(parent, Mapping) and dict(parent) != dict(expected_parent):
            issues.append("slow provenance source parent does not match cited Fast leaf")

    eligible_durability = {"durable", "long_term", "long-term", "hard", "persistent"}
    eligible_state = {"active", "parallel_active", "promoted"}
    eligible_fast_ids = {
        leaf_identity
        for leaf_identity, leaf in fast.items()
        if _text(
            leaf["metadata"].get("durability")
            or leaf["metadata"].get("durability_class")
        ).casefold()
        in eligible_durability
        and _text(leaf.get("state")).casefold() in eligible_state
    }
    cited_by_current_slow: set[Identity] = set()
    for record in slow_records:
        metadata = record["metadata"]
        if (
            _text(record.get("state")) != "active"
            or _text(metadata.get("status")).casefold() not in {"active", "challenged"}
        ):
            continue
        for claim in list(metadata.get("claims") or []):
            if isinstance(claim, Mapping):
                for raw_reference in [*(claim.get("support") or []), *(claim.get("counterevidence") or [])]:
                    reference = _reference_identity(
                        raw_reference,
                        db_path=record["db"],
                        scope_id=record.get("scope_id"),
                    )
                    if reference is not None:
                        cited_by_current_slow.add(reference)
    cited_eligible = eligible_fast_ids & cited_by_current_slow
    uncited_eligible = sorted(eligible_fast_ids - cited_by_current_slow)
    slow_semantic_integrity_issues.sort(
        key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True)
    )
    for semantic_issue in slow_semantic_integrity_issues:
        issues.append(
            "active Slow claim semantic integrity failed: "
            + json.dumps(semantic_issue, ensure_ascii=False, sort_keys=True)
        )
    enforce_slow_promotion = build_only or retrieval_dir is not None
    slow_promotion_coverage = {
        "schema_version": "tmcra.v4.slow-promotion-coverage.3",
        "enforced": enforce_slow_promotion,
        "complete": not uncited_eligible and not slow_semantic_integrity_issues,
        "eligible_current_durable_count": len(eligible_fast_ids),
        "cited_current_durable_count": len(cited_eligible),
        "coverage_ratio": (
            round(len(cited_eligible) / len(eligible_fast_ids), 6)
            if eligible_fast_ids
            else 1.0
        ),
        "uncited_current_durable_ids": uncited_eligible,
        "semantic_integrity_issue_count": len(slow_semantic_integrity_issues),
        "semantic_integrity_issues": slow_semantic_integrity_issues,
    }
    if enforce_slow_promotion and uncited_eligible:
        issues.append(
            "current durable Fast evidence is missing from active Slow claims: "
            + ",".join(identity[2] for identity in uncited_eligible[:20])
        )

    if build_only:
        retrieval = {
            "present": any(run_dir.rglob("retrieval_debug.jsonl")),
            "passed": None,
            "skipped": True,
            "reason": "build_only_audit",
        }
    else:
        retrieval = _audit_retrieval(
            run_dir,
            retrieval_dir,
            sources,
            issues,
            fast=fast,
            slow_records=slow_records,
            slow_patch_metadata=slow_patch_metadata,
            records=records,
        )
    report = {"schema_version": "tmcra.v4.chain-audit.1", "status": "passed" if not issues else "failed", "passed": not issues, "run_dir": str(run_dir), "input": str(input_path), "workers": {key: [str(path) for path in values] for key, values in db_groups.items()}, "counts": {"input_messages": len(expected) + len(excluded_empty), "nonempty_input_messages": len(expected), "excluded_empty_input_messages": len(excluded_empty), "source_records": len(sources), "fast_leaves": len(fast), "slow_records": len(slow_records), "interactions": len(interactions), "edges": len(edges)}, "subject_attribution": {"journal_count": len(subject_attribution_audits), "superseded_count": subject_attribution_superseded_count, "decision_count": subject_attribution_decision_count, "quarantine_count": subject_attribution_quarantine_count, "prompt_version": SUBJECT_ATTRIBUTION_PROMPT_VERSION, "model": "deepseek-v4-pro"}, "slow_promotion_coverage": slow_promotion_coverage, "source_tables": sorted(source_tables), "retrieval": retrieval, "issues": sorted(set(issues))}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _text_list(value: Any) -> list[str]:
    return [_text(item) for item in value] if isinstance(value, list) else []


def _audit_session_ordering(
    windows: list[Any], row_index: int, issues: list[str]
) -> None:
    pre_ranks: list[int] = []
    session_sequence: list[int] = []
    session_entries: dict[int, list[tuple[int, int, Mapping[str, Any]]]] = {}
    closed_sessions: set[int] = set()
    current_session: int | None = None
    malformed = False
    for window in windows:
        if not isinstance(window, Mapping):
            malformed = True
            continue
        metadata = window.get("retrieval_metadata")
        try:
            session_index = int(window["session_index"])
            parent_index = int(window["parent_chunk_index"])
            subchunk_index = int(window["subchunk_index"])
        except (KeyError, TypeError, ValueError):
            malformed = True
            continue
        if not isinstance(metadata, Mapping):
            malformed = True
            continue
        pre_rank = metadata.get("pre_session_order_rank")
        if type(pre_rank) is not int or pre_rank <= 0:
            malformed = True
            continue
        pre_ranks.append(pre_rank)
        if session_index != current_session:
            if current_session is not None:
                closed_sessions.add(current_session)
            if session_index in closed_sessions:
                issues.append(
                    f"retrieval row {row_index}: a session is interleaved after grouping"
                )
            current_session = session_index
            session_sequence.append(session_index)
        session_entries.setdefault(session_index, []).append(
            (parent_index, subchunk_index, metadata)
        )
    if malformed:
        issues.append(
            f"retrieval row {row_index}: session ordering coordinates are malformed"
        )
        return
    if sorted(pre_ranks) != list(range(1, len(windows) + 1)):
        issues.append(
            f"retrieval row {row_index}: pre-session ranks are not a permutation"
        )
    for expected_session_rank, session_index in enumerate(session_sequence, start=1):
        entries = session_entries[session_index]
        coordinates = [(parent, subchunk) for parent, subchunk, _ in entries]
        if coordinates != sorted(coordinates):
            issues.append(
                f"retrieval row {row_index}: a grouped session is not chronological"
            )
        expected_count = len(entries)
        expected_rrf = round(
            sum(1.0 / float(metadata["pre_session_order_rank"]) for _, _, metadata in entries),
            8,
        )
        for _, _, metadata in entries:
            observed_rrf = metadata.get("session_support_rrf")
            if (
                metadata.get("session_order_rank") != expected_session_rank
                or metadata.get("session_selected_window_count") != expected_count
                or isinstance(observed_rrf, bool)
                or not isinstance(observed_rrf, (int, float))
                or abs(float(observed_rrf) - expected_rrf) > 1e-8
            ):
                issues.append(
                    f"retrieval row {row_index}: session ordering aggregate is inconsistent"
                )
                break


def _audit_retrieval_contract(
    evidence_row: Mapping[str, Any],
    debug_row: Mapping[str, Any],
    windows: list[Any],
    row_index: int,
    issues: list[str],
) -> None:
    contract = evidence_row.get("retrieval_contract")
    if not isinstance(contract, Mapping):
        issues.append(f"retrieval row {row_index}: retrieval contract is missing")
        return
    if contract.get("schema_version") != RETRIEVAL_CONTRACT_SCHEMA_VERSION:
        issues.append(f"retrieval row {row_index}: retrieval contract schema is stale")
    lane = _text(contract.get("execution_lane"))
    mode = _text(contract.get("composition_mode"))
    if lane not in {"production", "diagnostic"}:
        issues.append(f"retrieval row {row_index}: execution lane is invalid")
    if lane == "production" and mode != "layered":
        issues.append(
            f"retrieval row {row_index}: production retrieval is not layered"
        )
    packing_mode = _text(contract.get("packing_budget_mode"))
    packing_budget = contract.get("packing_budget")
    if (
        isinstance(packing_budget, bool)
        or not isinstance(packing_budget, int)
        or packing_budget <= 0
    ):
        issues.append(f"retrieval row {row_index}: packing budget is invalid")
    if lane == "production" and (
        packing_mode != "fixed" or packing_budget != 8
    ):
        issues.append(
            f"retrieval row {row_index}: production packing is not fixed Top8"
        )
    if contract.get("source_coverage_trace_k") != SOURCE_COVERAGE_TRACE_K:
        issues.append(
            f"retrieval row {row_index}: retrieval contract is not Source Top24"
        )
    if contract.get("final_window_count") != len(windows):
        issues.append(
            f"retrieval row {row_index}: final window count disagrees with evidence"
        )
    if lane == "production" and len(windows) > 8:
        issues.append(f"retrieval row {row_index}: final evidence exceeds Top8")
    budget_decision = debug_row.get("packing_budget_decision")
    if (
        not isinstance(budget_decision, Mapping)
        or _text(budget_decision.get("mode")) != packing_mode
        or budget_decision.get("budget") != packing_budget
        or debug_row.get("packing_budget_top_k") != packing_budget
    ):
        issues.append(
            f"retrieval row {row_index}: packing contract disagrees with runtime debug"
        )

    actual = {"source": 0, "fast": 0, "slow": 0}
    for window in windows:
        metadata = window.get("retrieval_metadata") if isinstance(window, Mapping) else None
        contributions = (
            metadata.get("layer_contributions") if isinstance(metadata, Mapping) else []
        )
        layers = {
            _text(item.get("layer"))
            for item in contributions
            if isinstance(item, Mapping)
        }
        for layer in layers:
            if layer in actual:
                actual[layer] += 1
    inventory = contract.get("inventory_counts")
    inventory_counts: dict[str, int] = {}
    expected_inventory_keys = {
        "source",
        "fast",
        "fast_semantic",
        "slow",
        "slow_capsule_heads",
        "slow_summaries",
        "slow_claims",
        "slow_ranked_claims",
    }
    if (
        not isinstance(inventory, Mapping)
        or set(inventory) != expected_inventory_keys
        or any(
            type(inventory.get(layer)) is not int or inventory.get(layer) < 0
            for layer in expected_inventory_keys
        )
    ):
        issues.append(f"retrieval row {row_index}: inventory contract is invalid")
    else:
        inventory_counts = {layer: int(inventory[layer]) for layer in inventory}
        if inventory_counts["fast_semantic"] > inventory_counts["fast"]:
            issues.append(
                f"retrieval row {row_index}: semantic Fast inventory exceeds Fast"
            )
        if (
            inventory_counts["slow_capsule_heads"]
            != inventory_counts["slow_summaries"]
            or inventory_counts["slow_ranked_claims"] != inventory_counts["slow"]
            or inventory_counts["slow_claims"] < inventory_counts["slow_capsule_heads"]
        ):
            issues.append(
                f"retrieval row {row_index}: Slow summary/claim inventory counts are inconsistent"
            )
    selected = contract.get("selected_layer_window_counts")
    if not isinstance(selected, Mapping) or set(selected) != set(actual) or any(
        type(selected.get(layer)) is not int or selected.get(layer) != count
        for layer, count in actual.items()
    ):
        issues.append(
            f"retrieval row {row_index}: selected layer counts do not match final evidence"
        )
    required = contract.get("required_selected_layers")
    if (
        not isinstance(required, list)
        or not required
        or len(required) != len(set(map(_text, required)))
        or any(_text(layer) not in actual for layer in required)
    ):
        issues.append(f"retrieval row {row_index}: required layer contract is invalid")
    elif lane == "production":
        expected_required = {
            layer
            for layer in ("source", "fast", "slow")
            if inventory_counts.get(layer, 0) > 0
        }
        if set(map(_text, required)) != expected_required:
            issues.append(
                f"retrieval row {row_index}: required layers disagree with inventories"
            )
        missing = sorted(_text(layer) for layer in required if actual[_text(layer)] <= 0)
        if not required or missing:
            issues.append(
                f"retrieval row {row_index}: production layers are missing: {missing}"
            )
    paths = contract.get("candidate_paths_executed")
    if (
        not isinstance(paths, Mapping)
        or set(paths) != set(actual)
        or any(type(paths.get(layer)) is not bool for layer in actual)
        or paths != debug_row.get("candidate_paths_executed")
    ):
        issues.append(
            f"retrieval row {row_index}: path execution contract disagrees with debug"
        )


def _audit_retrieval(
    run_dir: Path,
    retrieval_dir: Path | None,
    sources: Mapping[Identity, Mapping[str, Any]],
    issues: list[str],
    *,
    fast: Mapping[Identity, Mapping[str, Any]] | None = None,
    slow_records: Iterable[Mapping[str, Any]] = (),
    slow_patch_metadata: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    records: Mapping[Identity, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    candidates = [Path(retrieval_dir)] if retrieval_dir else []
    if not candidates:
        candidates = sorted({path.parent for path in run_dir.rglob("retrieval_debug.jsonl")})
    if not candidates:
        return None
    directory = candidates[0]
    debug_path, evidence_path, report_path = directory / "retrieval_debug.jsonl", directory / "evidence_windows.jsonl", directory / "report.json"
    if not all(path.is_file() for path in (debug_path, evidence_path, report_path)):
        issues.append(f"retrieval output is incomplete: {directory}")
        return {"present": True, "passed": False}
    debug = [json.loads(line) for line in debug_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    evidence = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for item in [*debug, *evidence, report]:
        forbidden = _retrieval_forbidden(item)
        if forbidden:
            issues.extend(f"retrieval contains forbidden hard/evaluation field: {path}" for path in forbidden)
    if report.get("schema_version") != RETRIEVAL_REPORT_SCHEMA_VERSION:
        issues.append("retrieval report schema is missing or stale")
    if (
        report.get("runtime_schema_version") != RETRIEVAL_SCHEMA_VERSION
        or report.get("online_index_schema_version")
        != ONLINE_INDEX_SCHEMA_VERSION
        or report.get("slow_inventory_schema_version")
        != SLOW_INVENTORY_SCHEMA_VERSION
    ):
        issues.append("retrieval report index/inventory contract is missing or stale")
    if report.get("source_coverage_trace_k") != SOURCE_COVERAGE_TRACE_K:
        issues.append("retrieval report source coverage trace policy is missing or stale")
    if (
        report.get("answer_attachment_contract") != "object-list-v1"
        or report.get("ranking_metadata_field") != "retrieval_metadata"
    ):
        issues.append("retrieval report answer attachment contract is missing or stale")
    if (
        report.get("session_coherent_ordering") is not True
        or report.get("session_ordering_policy") != SESSION_ORDERING_POLICY
    ):
        issues.append("retrieval report session ordering contract is missing or stale")
    if report.get("status") != "complete" or len(debug) != len(evidence):
        issues.append("retrieval report/debug/evidence counts are not complete")
    fast = fast or {}
    slow_patch_metadata = slow_patch_metadata or {}
    slow_lookup: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    for record in slow_records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        if (
            _text(record.get("state")) != "active"
            or _text(metadata.get("status")) not in {"active", "challenged"}
        ):
            continue
        try:
            revision = int(metadata.get("revision", -1))
        except (TypeError, ValueError):
            continue
        slow_lookup[
            (
                _db_path(record.get("db")),
                _text(record.get("scope_id")),
                _text(metadata.get("capsule_id")),
                revision,
            )
        ] = record
    executed = 0
    source_candidate_id_collisions: set[tuple[int, str]] = set()
    for index, (debug_row, evidence_row) in enumerate(zip(debug, evidence)):
        if evidence_row.get("runtime_schema_version") != RETRIEVAL_SCHEMA_VERSION:
            issues.append(f"retrieval row {index}: runtime schema is missing or stale")
        if (
            debug_row.get("session_coherent_ordering") is not True
            or debug_row.get("session_ordering_policy") != SESSION_ORDERING_POLICY
        ):
            issues.append(
                f"retrieval row {index}: debug session ordering contract is missing or stale"
            )
        if debug_row.get("fast_semantic_state_policy") != CURRENT_FAST_INDEX_POLICY:
            issues.append(
                f"retrieval row {index}: fast semantic state policy is missing or stale"
            )
        inventories = {"source": int(debug_row.get("source_inventory_count", 0) or 0), "fast": int(debug_row.get("fast_inventory_count", 0) or 0), "slow": int(debug_row.get("slow_capsule_count", 0) or 0)}
        slow_count_fields = {
            "heads": debug_row.get("slow_capsule_head_count"),
            "summaries": debug_row.get("slow_summary_candidate_count"),
            "claims": debug_row.get("slow_claim_candidate_count"),
            "shortlist": debug_row.get("slow_shortlist_count"),
            "summary_hits": debug_row.get("slow_summary_hit_count"),
            "direct_hits": debug_row.get("slow_direct_claim_hit_count"),
        }
        if any(
            type(value) is not int or value < 0
            for value in slow_count_fields.values()
        ) or (
            slow_count_fields["heads"] != slow_count_fields["summaries"]
            or inventories["slow"] != slow_count_fields["heads"]
            or slow_count_fields["claims"] < slow_count_fields["heads"]
            or slow_count_fields["summary_hits"] > slow_count_fields["shortlist"]
            or slow_count_fields["direct_hits"] > slow_count_fields["shortlist"]
        ):
            issues.append(
                f"retrieval row {index}: Slow summary/claim debug counts are invalid"
            )
        slow_trace = debug_row.get("slow_retrieval_trace")
        trace_keys = {
            "inventory_schema_version",
            "direct_claim_hit",
            "claim_candidate_id",
            "claim_dense_rank",
            "claim_dense_score",
            "summary_expansion_hit",
            "summary_hit",
            "final_claim_cross_score",
            "source_parents",
        }
        if (
            not isinstance(slow_trace, list)
            or len(slow_trace) != slow_count_fields["shortlist"]
        ):
            issues.append(f"retrieval row {index}: Slow retrieval trace is incomplete")
        else:
            for trace_index, item in enumerate(slow_trace):
                if (
                    not isinstance(item, Mapping)
                    or set(item) != trace_keys
                    or item.get("inventory_schema_version")
                    != SLOW_INVENTORY_SCHEMA_VERSION
                    or type(item.get("direct_claim_hit")) is not bool
                    or type(item.get("summary_expansion_hit")) is not bool
                    or not (
                        item.get("direct_claim_hit")
                        or item.get("summary_expansion_hit")
                    )
                    or not _text(item.get("claim_candidate_id"))
                    or type(item.get("claim_dense_rank")) is not int
                    or item.get("claim_dense_rank") < 0
                    or not isinstance(item.get("claim_dense_score"), (int, float))
                    or not isinstance(
                        item.get("final_claim_cross_score"), (int, float)
                    )
                    or not isinstance(item.get("source_parents"), list)
                    or not item.get("source_parents")
                ):
                    issues.append(
                        f"retrieval row {index}: Slow trace {trace_index} is malformed"
                    )
                    continue
                summary_hit = item.get("summary_hit")
                if item.get("summary_expansion_hit"):
                    if (
                        not isinstance(summary_hit, Mapping)
                        or not _text(summary_hit.get("summary_candidate_id"))
                        or not _text(summary_hit.get("summary_text"))
                        or type(summary_hit.get("summary_dense_rank")) is not int
                        or summary_hit.get("summary_dense_rank") < 0
                        or not isinstance(
                            summary_hit.get("summary_dense_score"), (int, float)
                        )
                    ):
                        issues.append(
                            f"retrieval row {index}: Slow trace {trace_index} summary hit is malformed"
                        )
                elif summary_hit is not None:
                    issues.append(
                        f"retrieval row {index}: Slow trace {trace_index} has an unexecuted summary hit"
                    )
        paths = debug_row.get("candidate_paths_executed")
        if not isinstance(paths, Mapping):
            issues.append(f"retrieval row {index}: candidate path execution audit is missing")
        else:
            for layer, count in inventories.items():
                if count > 0 and paths.get(layer) is not True:
                    issues.append(f"retrieval row {index}: {layer} inventory exists but path did not execute")
            executed += int(all(paths.get(layer) is True for layer, count in inventories.items() if count > 0))
        source_count_value = debug_row.get("source_candidate_count")
        source_count = (
            source_count_value
            if type(source_count_value) is int and source_count_value > 0
            else 0
        )
        if source_count == 0:
            issues.append(f"retrieval row {index}: source candidate count is invalid")
        trace = debug_row.get("source_top24_candidates")
        pool = debug_row.get("source_candidate_pool_trace")
        if debug_row.get("source_coverage_trace_k") != SOURCE_COVERAGE_TRACE_K:
            issues.append(
                f"retrieval row {index}: source coverage trace policy is missing or stale"
            )
        if not isinstance(trace, list) or len(trace) != min(
            SOURCE_COVERAGE_TRACE_K, source_count
        ) or not isinstance(pool, list) or len(pool) != source_count:
            issues.append(f"retrieval row {index}: source Top24 trace is incomplete")
        else:
            required = {
                "rank",
                "candidate_id",
                "session_id",
                "session_index",
                "parent_chunk_index",
                "subchunk_index",
            }
            def trace_identities(
                rows: list[Any], label: str
            ) -> list[tuple[str, str, int, int, int]]:
                identities: list[tuple[str, str, int, int, int]] = []
                candidate_ids: set[str] = set()
                locations: set[tuple[str, int, int, int]] = set()
                for rank, item in enumerate(rows, start=1):
                    if not isinstance(item, Mapping) or set(item) != required:
                        issues.append(
                            f"retrieval row {index}: {label} entry is malformed"
                        )
                        continue
                    coordinates = (
                        item.get("session_index"),
                        item.get("parent_chunk_index"),
                        item.get("subchunk_index"),
                    )
                    candidate_id = _text(item.get("candidate_id"))
                    session_id = _text(item.get("session_id"))
                    if (
                        item.get("rank") != rank
                        or not candidate_id
                        or not session_id
                        or any(type(value) is not int or value < 0 for value in coordinates)
                    ):
                        issues.append(
                            f"retrieval row {index}: {label} rank or identity is invalid"
                        )
                        continue
                    location = (session_id, *coordinates)
                    if candidate_id in candidate_ids:
                        source_candidate_id_collisions.add((index, candidate_id))
                    if location in locations:
                        issues.append(
                            f"retrieval row {index}: {label} contains duplicate evidence"
                        )
                    candidate_ids.add(candidate_id)
                    locations.add(location)
                    identities.append((candidate_id, session_id, *coordinates))
                return identities

            trace_id = trace_identities(trace, "source Top24 trace")
            pool_id = trace_identities(pool, "source candidate pool trace")
            if trace_id != pool_id[: len(trace)]:
                issues.append(
                    f"retrieval row {index}: source Top24 trace is not the exact pool prefix"
                )
        windows = evidence_row.get("evidence_windows")
        if not isinstance(windows, list):
            issues.append(f"retrieval row {index}: evidence_windows missing")
            continue
        _audit_retrieval_contract(evidence_row, debug_row, windows, index, issues)
        _audit_session_ordering(windows, index, issues)
        for rank, window in enumerate(windows, start=1):
            if not isinstance(window, Mapping) or not _text(window.get("text")):
                issues.append(f"retrieval row {index}: final evidence text missing")
                continue
            if window.get("rank") != rank:
                issues.append(f"retrieval row {index}: final evidence ranks are not contiguous")
            attachments = window.get("attachments")
            if not isinstance(attachments, list) or any(
                not isinstance(item, Mapping) for item in attachments
            ):
                issues.append(
                    f"retrieval row {index}: attachments must be an object list"
                )
            contexts = window.get("memory_contexts")
            if not isinstance(contexts, list) or any(
                not isinstance(item, Mapping) for item in contexts
            ):
                issues.append(
                    f"retrieval row {index}: memory_contexts must be an object list"
                )
            provenance = window.get("provenance")
            if not isinstance(provenance, list) or any(
                not isinstance(item, Mapping) for item in provenance
            ):
                issues.append(
                    f"retrieval row {index}: provenance must be an object list"
                )
            retrieval_metadata = window.get("retrieval_metadata")
            contributions = (
                retrieval_metadata.get("layer_contributions")
                if isinstance(retrieval_metadata, Mapping)
                else None
            )
            if (
                not isinstance(retrieval_metadata, Mapping)
                or not isinstance(contributions, list)
                or not contributions
                or any(not isinstance(item, Mapping) for item in contributions)
            ):
                issues.append(
                    f"retrieval row {index}: retrieval_metadata is missing or malformed"
                )
            elif (
                retrieval_metadata.get("session_ordering_policy")
                != SESSION_ORDERING_POLICY
                or not isinstance(
                    retrieval_metadata.get("pre_session_order_rank"), int
                )
                or not isinstance(retrieval_metadata.get("session_order_rank"), int)
                or not isinstance(
                    retrieval_metadata.get("session_selected_window_count"), int
                )
                or not isinstance(
                    retrieval_metadata.get("session_support_rrf"), (int, float)
                )
            ):
                issues.append(
                    f"retrieval row {index}: session ordering audit is missing or malformed"
                )
            text = str(window["text"])
            memory_id = _text(window.get("memory_id"))
            source_id = _text(window.get("source_record_id")) or memory_id
            declared_scope = _text(window.get("scope_id"))
            declared_db = window.get("db_path")
            if not source_id or not declared_scope or not _text(declared_db):
                issues.append(
                    f"retrieval row {index}: final evidence lacks composite Source identity"
                )
                continue
            source_identity = _identity(declared_db, declared_scope, source_id)
            source = sources.get(source_identity)
            if source is None:
                issues.append(
                    f"retrieval row {index}: final evidence source identity is missing or ambiguous: {source_id}"
                )
                continue
            owner_db, owner_scope = source_identity[0], source_identity[1]

            def retrieval_fast_reference(value: Any, label: str) -> Identity | None:
                reference = _reference_identity(
                    value,
                    db_path=owner_db,
                    scope_id=owner_scope,
                )
                if reference is None or reference not in fast:
                    issues.append(
                        f"retrieval row {index}: {label} does not map to current Fast DB identity: {value}"
                    )
                    return None
                if records and reference not in records:
                    issues.append(
                        f"retrieval row {index}: {label} does not map to current DB record identity: {reference}"
                    )
                    return None
                return reference

            for attachment_index, attachment in enumerate(attachments if isinstance(attachments, list) else []):
                if not isinstance(attachment, Mapping):
                    continue
                attachment_identity = retrieval_fast_reference(
                    attachment,
                    f"Fast attachment {attachment_index}",
                )
                if attachment_identity is None:
                    continue
                leaf = fast[attachment_identity]
                leaf_meta = leaf["metadata"]
                if _text(attachment.get("text")) != _text(leaf.get("value")):
                    issues.append(
                        f"retrieval row {index}: Fast attachment {attachment_index} text differs from current DB record"
                    )
                if _text(attachment.get("record_state")) != _text(leaf.get("state")):
                    issues.append(
                        f"retrieval row {index}: Fast attachment {attachment_index} record state differs from current DB record"
                    )
                expected_slot = _text(
                    leaf_meta.get("canonical_slot")
                    or leaf_meta.get("canonical_slot_key")
                )
                if _text(attachment.get("canonical_slot")) != expected_slot:
                    issues.append(
                        f"retrieval row {index}: Fast attachment {attachment_index} slot differs from current DB record"
                    )
                expected_fast_fields = {
                    "memory_type": _text(leaf_meta.get("memory_type")),
                    "durability": _text(
                        leaf_meta.get("durability")
                        or leaf_meta.get("durability_class")
                    ),
                    "temporal_status": _text(
                        leaf_meta.get("temporal_status")
                        or leaf_meta.get("target_status")
                    ),
                }
                if any(
                    _text(attachment.get(field)) != expected
                    for field, expected in expected_fast_fields.items()
                ):
                    issues.append(
                        f"retrieval row {index}: Fast attachment {attachment_index} typed state differs from current DB record"
                    )
                parent = attachment.get("source_parent")
                expected_parent = _fast_source_parent(leaf_meta)
                if not isinstance(parent, Mapping) or dict(parent) != expected_parent:
                    issues.append(
                        f"retrieval row {index}: Fast attachment {attachment_index} source parent is not current"
                    )
                _validate_source_parent(
                    parent,
                    db_path=owner_db,
                    scope_id=owner_scope,
                    sources=sources,
                    issues=issues,
                    label=f"retrieval row {index} Fast attachment {attachment_index}",
                    quote=_fast_quote(leaf_meta),
                )
                attachment_provenance = attachment.get("provenance")
                expected_provenance = {
                    "memory_layer": "fast",
                    "content_variant": PRODUCTION_FAST_VARIANT,
                    "source_record_id": _text(leaf_meta.get("source_record_id")),
                    "semantic_memory_id": attachment_identity[2],
                }
                if (
                    not isinstance(attachment_provenance, Mapping)
                    or dict(attachment_provenance) != expected_provenance
                ):
                    issues.append(
                        f"retrieval row {index}: Fast attachment {attachment_index} provenance is invalid"
                    )

            for context_index, context in enumerate(contexts if isinstance(contexts, list) else []):
                if not isinstance(context, Mapping):
                    continue
                try:
                    context_revision = int(context.get("revision"))
                except (TypeError, ValueError):
                    context_revision = -1
                context_key = (
                    owner_db,
                    owner_scope,
                    _text(context.get("capsule_id")),
                    context_revision,
                )
                slow_record = slow_lookup.get(context_key)
                if slow_record is None:
                    issues.append(
                        f"retrieval row {index}: Slow context {context_index} is not the current DB revision"
                    )
                    continue
                if _text(context.get("status")) != _text(slow_record["metadata"].get("status")):
                    issues.append(
                        f"retrieval row {index}: Slow context {context_index} status differs from current DB state"
                    )
                current_claims = list(slow_record["metadata"].get("claims") or [])
                current_claim_index = next(
                    (
                        claim_index
                        for claim_index, claim in enumerate(current_claims)
                        if isinstance(claim, Mapping)
                        and _text(claim.get("claim_id"))
                        == _text(context.get("claim_id"))
                    ),
                    -1,
                )
                current_claim = next(
                    (
                        claim for claim in current_claims
                        if isinstance(claim, Mapping)
                        and _text(claim.get("claim_id")) == _text(context.get("claim_id"))
                    ),
                    None,
                )
                if current_claim is None:
                    issues.append(
                        f"retrieval row {index}: Slow context {context_index} claim is not current DB state"
                    )
                elif (
                    _text(context.get("canonical_slot")) != _text(current_claim.get("canonical_slot"))
                    or _text(context.get("claim_text")) != _text(current_claim.get("text"))
                ):
                    issues.append(
                        f"retrieval row {index}: Slow context {context_index} claim text/slot differs from current DB state"
                    )
                expected_summary_id = (
                    f"capsule-summary::{_text(slow_record['metadata'].get('capsule_id'))}:"
                    f"r{int(slow_record['metadata'].get('revision'))}"
                )
                expected_claim_candidate_id = (
                    f"capsule::{_text(slow_record['metadata'].get('capsule_id'))}:"
                    f"r{int(slow_record['metadata'].get('revision'))}:c{current_claim_index}"
                )
                if (
                    _text(context.get("capsule_summary"))
                    != _text(slow_record.get("value"))
                    or _text(context.get("capsule_summary_candidate_id"))
                    != expected_summary_id
                ):
                    issues.append(
                        f"retrieval row {index}: Slow context {context_index} capsule summary differs from current DB state"
                    )
                if current_claim is not None and (
                    list(context.get("support") or [])
                    != list(current_claim.get("support") or [])
                    or list(context.get("counterevidence") or [])
                    != list(current_claim.get("counterevidence") or [])
                ):
                    issues.append(
                        f"retrieval row {index}: Slow context {context_index} citations differ from current DB claim"
                    )
                context_refs: list[Identity] = []
                for raw_reference in [*(context.get("support") or []), *(context.get("counterevidence") or [])]:
                    reference = retrieval_fast_reference(
                        raw_reference,
                        f"Slow context {context_index} citation",
                    )
                    if reference is not None:
                        context_refs.append(reference)
                context_parents = context.get("source_parents")
                expected_context_parents = (
                    list(
                        current_claim.get(
                            "source_parents",
                            slow_record["metadata"].get("source_parents"),
                        )
                        or []
                    )
                    if current_claim is not None
                    else []
                )
                if not isinstance(context_parents, list) or not context_parents:
                    issues.append(
                        f"retrieval row {index}: Slow context {context_index} source parents are missing"
                    )
                else:
                    if context_parents != expected_context_parents:
                        issues.append(
                            f"retrieval row {index}: Slow context {context_index} source parents differ from current DB claim"
                        )
                    for parent in context_parents:
                        _validate_source_parent(
                            parent,
                            db_path=owner_db,
                            scope_id=owner_scope,
                            sources=sources,
                            issues=issues,
                            label=f"retrieval row {index} Slow context {context_index}",
                        )
                    for reference in context_refs:
                        leaf_meta = fast[reference]["metadata"]
                        expected_parent = _slow_source_parent(leaf_meta)
                        if not any(isinstance(parent, Mapping) and dict(parent) == dict(expected_parent) for parent in context_parents):
                            issues.append(
                                f"retrieval row {index}: Slow context {context_index} citation parent is not current"
                            )
                context_provenance = context.get("provenance")
                expected_context_provenance = (
                    _expected_slow_context_provenance(
                        slow_record["metadata"],
                        current_claim,
                        expected_context_parents,
                        expected_summary_id,
                        patch_metadata=slow_patch_metadata.get(
                            (
                                owner_db,
                                _text(slow_record["metadata"].get("patch_id")),
                            )
                        ),
                    )
                    if current_claim is not None
                    else None
                )
                if (
                    not isinstance(context_provenance, Mapping)
                    or expected_context_provenance is None
                    or dict(context_provenance) != expected_context_provenance
                ):
                    issues.append(
                        f"retrieval row {index}: Slow context {context_index} provenance is invalid"
                    )
                context_trace = context.get("retrieval_trace")
                if (
                    not isinstance(context_trace, Mapping)
                    or context_trace.get("inventory_schema_version")
                    != SLOW_INVENTORY_SCHEMA_VERSION
                    or _text(context_trace.get("claim_candidate_id"))
                    != expected_claim_candidate_id
                    or type(context_trace.get("direct_claim_hit")) is not bool
                    or type(context_trace.get("summary_expansion_hit")) is not bool
                    or not (
                        context_trace.get("direct_claim_hit")
                        or context_trace.get("summary_expansion_hit")
                    )
                    or list(context_trace.get("source_parents") or [])
                    != expected_context_parents
                ):
                    issues.append(
                        f"retrieval row {index}: Slow context {context_index} summary-to-claim-to-Source trace is invalid"
                    )
                elif context_trace.get("summary_expansion_hit"):
                    summary_hit = context_trace.get("summary_hit")
                    if (
                        not isinstance(summary_hit, Mapping)
                        or _text(summary_hit.get("summary_candidate_id"))
                        != expected_summary_id
                        or _text(summary_hit.get("summary_text"))
                        != _text(slow_record.get("value"))
                    ):
                        issues.append(
                            f"retrieval row {index}: Slow context {context_index} summary expansion is not current DB state"
                        )

            for citation_index, citation in enumerate(window.get("citations") or []):
                retrieval_fast_reference(citation, f"final citation {citation_index}")
            for provenance_index, item in enumerate(provenance if isinstance(provenance, list) else []):
                if not isinstance(item, Mapping):
                    continue
                raw_reference = item.get("evidence_memory_id") or item.get("memory_id") or item.get("leaf_id")
                if raw_reference:
                    retrieval_fast_reference(raw_reference, f"final provenance {provenance_index}")

            if window.get("source_char_start") is None or window.get("source_char_end") is None:
                issues.append(
                    f"retrieval row {index}: final evidence lacks auditable source offsets: {source_id}"
                )
                continue
            if not _source_span_quote(
                source,
                window.get("source_char_start"),
                window.get("source_char_end"),
                text,
                label=f"retrieval row {index} final evidence",
                issues=issues,
            ):
                issues.append(
                    f"retrieval row {index}: final evidence text/offsets do not match current immutable Source"
                )
    return {
        "present": True,
        "passed": not any(item.startswith("retrieval") for item in issues),
        "rows": len(debug),
        "all_layer_path_rows": executed,
        "source_candidate_id_collision_count": len(source_candidate_id_collisions),
        "source_candidate_id_collision_rows": sorted(
            {row_index for row_index, _ in source_candidate_id_collisions}
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict TMCRA V4 post-run chain audit")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-db", action="append", default=[], metavar="WORKER=PATH", help="explicit per-worker SQLite path; repeatable")
    parser.add_argument("--retrieval-dir", type=Path)
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="audit source/writer/fast/slow build state without selecting historical retrieval output",
    )
    args = parser.parse_args(argv)
    try:
        if args.build_only and args.retrieval_dir is not None:
            raise AuditError("--build-only and --retrieval-dir are mutually exclusive")
        report = audit_run(
            args.run_dir,
            output=args.output,
            worker_db_specs=args.worker_db,
            retrieval_dir=args.retrieval_dir,
            build_only=args.build_only,
        )
    except (AuditError, OSError, sqlite3.Error) as exc:
        report = {"schema_version": "tmcra.v4.chain-audit.1", "status": "failed", "passed": False, "issues": [str(exc)]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
