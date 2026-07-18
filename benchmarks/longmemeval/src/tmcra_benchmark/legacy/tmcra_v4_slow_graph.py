#!/usr/bin/env python3
"""TMCRA V4 tiered slow-graph controller.

The append-only store, provenance checks, leases, and CLI job lifecycle are
reused from the V3 implementation.  This module owns only V4 route selection
and the strict Flash/Pro transport policy.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import tmcra_v3_slow_graph as _v3


# Re-export the V3 store machinery instead of copying or changing it.
AuditError = _v3.AuditError
DeepSeekCallError = _v3.DeepSeekCallError
EvidencePolicyError = _v3.EvidencePolicyError
JobClaim = _v3.JobClaim
PatchValidationError = _v3.PatchValidationError
SlowGraphError = _v3.SlowGraphError
StaleRevisionError = _v3.StaleRevisionError
load_graph_schema = _v3.load_graph_schema
validate_patch = _v3.validate_patch

CAPSULE_VARIANT = _v3.CAPSULE_VARIANT
LEAF_VARIANT = _v3.LEAF_VARIANT
PATCH_ACTIONS = _v3.PATCH_ACTIONS
SCHEMA_VERSION = _v3.SCHEMA_VERSION
SLOW_PROMPT_MIGRATION_SOURCE_VERSION = "tmcra-v4-slow-graph-2026-07-14.13"
SLOW_PROMPT_MIGRATION_SOURCE_VERSIONS = (
    "tmcra-v4-slow-graph-2026-07-14.12",
    SLOW_PROMPT_MIGRATION_SOURCE_VERSION,
)
SLOW_PROMPT_VERSION = "tmcra-v4-slow-graph-2026-07-14.16"
SLOW_SUMMARY_CONTRACT_VERSION = "tmcra.v4.slow-lossless-summary.2"
SLOW_PARTITION_CONTRACT_VERSION = "tmcra.v4.slow-semantic-partition.2"
SLOW_EVIDENCE_BINDING_CONTRACT_VERSION = "tmcra.v4.slow-evidence-binding.3"
SUPPORTED_SLOW_EVIDENCE_BINDING_CONTRACT_VERSIONS = frozenset(
    {
        None,
        "tmcra.v4.slow-evidence-binding.2",
        SLOW_EVIDENCE_BINDING_CONTRACT_VERSION,
    }
)
SLOW_PROCESS_LOSS_RECOVERY_VERSION = "tmcra.v4.slow-process-loss-recovery.1"
SLOW_PROCESS_LOSS_PHYSICAL_CALLS_MAX = 3
PROCESS_LOSS_INTERRUPTION_ERROR = (
    "claim lease expired; external call outcome uncertain; explicit resume required"
)
SLOW_SUMMARY_MAX_CHARS = 4096
SLOW_MAX_REGION_OPERATIONS = 32
FLASH_ESCALATION_REASON = "cross_slot_conflict"
_CAPSULE_KEY_PATTERN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")
_GENERIC_REGION_KEYS = frozenset(
    {
        "activities",
        "activity",
        "belief",
        "beliefs",
        "communication",
        "goals",
        "interest",
        "interests",
        "learning",
        "opinion",
        "opinions",
        "preference",
        "preferences",
        "routine",
        "routines",
        "skills",
    }
)
_CAPSULE_KEY_SCAFFOLD_TOKENS = frozenset({"memory", "user"})
GENERIC_MULTI_SLOT_CAPSULE_KEY_ERROR = (
    "capsule_key must name a concrete semantic topic for a generic-region "
    "capsule containing multiple canonical slots"
)
LEGACY_SINGLE_BINDING_ERROR_PREFIX = (
    "atomic Fast evidence may belong to only one resulting claim: "
)
ZERO_CALL_CONFIGURATION_ERRORS = {
    "flash": "flash client is not configured; no fallback is allowed",
    "pro": "pro client is not configured; no fallback is allowed",
}


class TieredAPIError(DeepSeekCallError):
    """A physical API failure that is never eligible for automatic retry."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


_OPERATIONAL_SUMMARY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*deterministic\s+(?:create|revise|retire|cleanup|migration)\b",
        r"^\s*(?:initial\s+)?consolidat(?:e|ed|es|ing|ion)\b[^.]{0,120}\b(?:evidence|claims?|facts?|preferences?|routines?|memory)\b",
        r"^\s*(?:add|added|adding|challenge|challenged|challenging)\b[^.]{0,80}\bevidence\b",
        r"^\s*add\b",
        r"\b(?:supplied|required|current\s+fast)\s+evidence\b",
        r"^\s*create\s+(?:an?\s+|the\s+)?(?:initial\s+)?[^.]{0,80}\b(?:memory|claims?|record)\b",
        r"^\s*(?:memory|record)\s+(?:revision|cleanup|migration)\b",
        r"^\s*(?:创建|新增|更新|合并|整理).{0,24}(?:记忆|胶囊|证据|声明|记录)",
        r"^\s*(?:控制器|确定性创建|确定性修订).{0,24}(?:记忆|证据|声明|记录|胶囊)",
        r"(?:快速图证据)",
    )
)
_GENERIC_SUMMARY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^user(?:'s)?\s+(?:fitness\s+)?(?:goals?\s+and\s+facts?|commute\s+details?|schedule\s+routine|preferences?\s+and\s+facts?)(?:\.)?$",
        r"^user\s+(?:has\s+)?social\s+media\s+goals?\s+and\s+routines?(?:\.)?$",
        r"^user\s+opinions?\s+(?:on|about)\s+[^.]+(?:\.)?$",
        r"^user\s+preferences?\s+and\s+facts?\s+about\s+[^.]+(?:\.)?$",
        r"^user(?:'s)?\s+[^.]+\s+preferences?\s+and\s+wearing\s+frequency(?:\.)?$",
        r"^(?:用户|使用者)的?(?:健身目标和事实|通勤详情|日程惯例|偏好和事实)[。.]?$",
    )
)


def _semantic_summary_projection(claims: Any) -> str:
    if not isinstance(claims, list) or not claims:
        raise PatchValidationError("semantic summary projection requires claims")
    texts: list[str] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            raise PatchValidationError(
                f"semantic summary projection claim {index} is not an object"
            )
        text = _required_text(claim.get("text"), f"claim {index} text")
        texts.append(text)
    summary = " ".join(texts)
    if len(summary) > SLOW_SUMMARY_MAX_CHARS:
        raise PatchValidationError(
            "lossless semantic summary projection exceeds the summary limit"
        )
    return summary


def validate_semantic_summary(
    summary: Any,
    claims: Any,
    *,
    label: str = "slow GraphPatch summary",
) -> str:
    """Validate only high-confidence summary contract violations.

    Semantic quality remains a model responsibility and is reviewed separately;
    this gate rejects empty, structured, operational, and known heading-only text.
    """
    value = " ".join(_clean(summary).split())
    if not value:
        raise PatchValidationError(f"{label} is required")
    if len(value) > SLOW_SUMMARY_MAX_CHARS:
        raise PatchValidationError(
            f"{label} exceeds {SLOW_SUMMARY_MAX_CHARS} characters"
        )
    if not isinstance(claims, list) or not claims:
        raise PatchValidationError(f"{label} requires non-empty claims")
    if value[:1] in "[{":
        try:
            structured = json.loads(value)
        except json.JSONDecodeError:
            structured = None
        if isinstance(structured, (dict, list)):
            raise PatchValidationError(f"{label} must not be JSON")
    for pattern in _OPERATIONAL_SUMMARY_PATTERNS:
        if pattern.search(value):
            raise PatchValidationError(
                f"{label} describes a graph operation instead of user memory"
            )
    for pattern in _GENERIC_SUMMARY_PATTERNS:
        if pattern.fullmatch(value):
            raise PatchValidationError(
                f"{label} is a generic heading instead of a semantic memory statement"
            )
    if sum(character.isalnum() for character in value) < 4:
        raise PatchValidationError(f"{label} has no substantive semantic content")
    return value


def _validate_patch_summary_contract(patch: Mapping[str, Any]) -> None:
    operations = patch.get("operations")
    if not isinstance(operations, list):
        raise PatchValidationError("summary contract requires an operations list")
    for index, operation in enumerate(operations):
        if not isinstance(operation, Mapping):
            raise PatchValidationError("summary contract operation must be an object")
        if _clean(operation.get("action")) == "noop":
            continue
        summary = validate_semantic_summary(
            operation.get("summary"),
            operation.get("claims"),
            label=f"operations[{index}].summary",
        )
        expected = _semantic_summary_projection(operation.get("claims"))
        if summary != expected:
            raise PatchValidationError(
                f"operations[{index}].summary must equal the lossless final-claim projection"
            )


def _normalize_transport_patch(
    patch: Any,
    capsules: list[Mapping[str, Any]],
    region: Mapping[str, Any] | None = None,
) -> tuple[Any, list[dict[str, str]]]:
    """Normalize only JSON transport blemishes with one unambiguous meaning."""
    if not isinstance(patch, Mapping):
        return patch, []
    operations = patch.get("operations")
    normalizations: list[dict[str, str]] = []
    if isinstance(operations, Mapping):
        operations = [operations]
        normalizations.append(
            {
                "code": "single_operation_object_wrapped_as_list",
                "field": "operations",
                "reason": "one operation object is an unambiguous singleton operations list",
            }
        )
    if isinstance(operations, list) and len(operations) > 1 and not capsules:
        evidence = region.get("evidence", []) if isinstance(region, Mapping) else []
        evidence_ids = {
            _clean(item.get("memory_id"))
            for item in evidence
            if isinstance(item, Mapping) and _clean(item.get("memory_id"))
        }
        pure_evidence_noops = all(
            isinstance(operation, Mapping)
            and set(operation).issubset({"action", "capsule_id"})
            and operation.get("action") == "noop"
            and (
                "capsule_id" not in operation
                or _clean(operation.get("capsule_id")) in evidence_ids
            )
            for operation in operations
        )
        if pure_evidence_noops:
            operations = [{"action": "noop"}]
            normalizations.append(
                {
                    "code": "multiple_evidence_keyed_noops_collapsed",
                    "field": "operations",
                    "reason": "multiple capsule-free noops keyed only by supplied evidence IDs have one no-op effect",
                }
            )
    if not isinstance(operations, list):
        return patch, []
    evidence = region.get("evidence", []) if isinstance(region, Mapping) else []
    normalized_operations: list[Any] = []
    for operation_index, raw_operation in enumerate(operations):
        if not isinstance(raw_operation, Mapping):
            normalized_operations.append(raw_operation)
            continue
        operation = dict(raw_operation)
        if (
            operation.get("action") == "noop"
            and "capsule_id" in operation
            and operation.get("capsule_id") is None
            and len(operations) == 1
        ):
            del operation["capsule_id"]
            normalizations.append(
                {
                    "code": "noop_null_capsule_id_ignored",
                    "field": f"operations[{operation_index}].capsule_id",
                    "reason": "a null optional noop identity has the same meaning as an omitted identity",
                }
            )
        elif (
            operation.get("action") == "noop"
            and "capsule_id" in operation
            and not capsules
            and len(operations) == 1
        ):
            del operation["capsule_id"]
            normalizations.append(
                {
                    "code": "noop_identity_ignored_without_supplied_capsule",
                    "field": f"operations[{operation_index}].capsule_id",
                    "reason": "a capsule-free region noop cannot target a model-invented capsule identity",
                }
            )

        if operation.get("action") == "create" and _clean(
            operation.get("capsule_key")
        ):
            raw_capsule_key = _clean(operation.get("capsule_key"))
            normalized_capsule_key = _normalize_transport_capsule_key(
                raw_capsule_key
            )
            if (
                normalized_capsule_key is not None
                and normalized_capsule_key != raw_capsule_key
            ):
                operation["capsule_key"] = normalized_capsule_key
                normalizations.append(
                    {
                        "code": "create_capsule_key_ascii_separators_normalized",
                        "field": f"operations[{operation_index}].capsule_key",
                        "reason": "ASCII case and word separators have one stable dot-separated identifier representation",
                    }
                )

        claims = operation.get("claims")
        if isinstance(claims, list) and isinstance(evidence, list):
            normalized_claims: list[Any] = []
            for claim_index, raw_claim in enumerate(claims):
                if not isinstance(raw_claim, Mapping):
                    normalized_claims.append(raw_claim)
                    continue
                claim = dict(raw_claim)
                if claim.get("support") == [] and claim.get("counterevidence") == []:
                    slot = _clean(claim.get("canonical_slot"))
                    text = _normal_text(claim.get("text"))
                    matches = [
                        item
                        for item in evidence
                        if isinstance(item, Mapping)
                        and _clean(item.get("canonical_slot")) == slot
                        and _normal_text(item.get("value")) == text
                        and _clean(item.get("memory_id"))
                    ]
                    if slot and text and len(matches) == 1:
                        evidence_id = _clean(matches[0].get("memory_id"))
                        claim["support"] = [evidence_id]
                        normalizations.append(
                            {
                                "code": "empty_support_bound_to_unique_exact_evidence",
                                "field": f"operations[{operation_index}].claims[{claim_index}].support",
                                "reason": "canonical_slot and normalized text uniquely match one supplied evidence leaf",
                            }
                        )
                normalized_claims.append(claim)
            operation["claims"] = normalized_claims
            if operation.get("action") == "create" and not _clean(
                operation.get("capsule_key")
            ):
                slots = {
                    _clean(item.get("canonical_slot"))
                    for item in normalized_claims
                    if isinstance(item, Mapping)
                    and _clean(item.get("canonical_slot"))
                }
                if len(slots) == 1:
                    operation["capsule_key"] = _capsule_key_from_slot(
                        next(iter(slots))
                    )
                    normalizations.append(
                        {
                            "code": "create_capsule_key_bound_to_unique_claim_slot",
                            "field": f"operations[{operation_index}].capsule_key",
                            "reason": "all create claims share one authoritative canonical slot",
                        }
                    )
        normalized_operations.append(operation)

    if not normalizations:
        return patch, []
    normalized = dict(patch)
    normalized["operations"] = normalized_operations
    return normalized, normalizations


def _required_text(value: Any, label: str) -> str:
    result = _clean(value)
    if not result:
        raise PatchValidationError(f"{label} is required")
    return result


def _normal_text(value: Any) -> str:
    return " ".join(_clean(value).casefold().split())


def _normalize_capsule_key(value: Any, label: str = "capsule_key") -> str:
    key = _clean(value).casefold()
    if not key or len(key) > 96 or _CAPSULE_KEY_PATTERN.fullmatch(key) is None:
        raise PatchValidationError(
            f"{label} must be a lowercase dot-separated identifier of at most 96 characters"
        )
    return key


def _normalize_transport_capsule_key(value: Any) -> str | None:
    """Return a strict key only when ASCII spelling cleanup is lossless."""
    raw = _clean(value)
    if not raw:
        return None
    key = re.sub(r"[\s_-]+", ".", raw.casefold())
    key = re.sub(r"\.+", ".", key).strip(".")
    if not key or len(key) > 96 or _CAPSULE_KEY_PATTERN.fullmatch(key) is None:
        return None
    return key


def _capsule_key_from_slot(slot: Any) -> str:
    value = _required_text(slot, "canonical_slot").casefold()
    if len(value) <= 96 and _CAPSULE_KEY_PATTERN.fullmatch(value) is not None:
        return value
    return "slot." + _digest({"canonical_slot": value})[:24]


def _is_generic_region_key(value: Any) -> bool:
    return _clean(value).casefold() in _GENERIC_REGION_KEYS


def _capsule_key_is_generic_for_region(region_key: Any, capsule_key: Any) -> bool:
    key_tokens = {
        item
        for item in re.split(r"[^a-z0-9]+", _clean(capsule_key).casefold())
        if item
    }
    region_tokens = {
        item
        for item in re.split(r"[^a-z0-9]+", _clean(region_key).casefold())
        if item
    }
    # This gate is intentionally structural. Pro owns semantic grouping; the
    # controller rejects only identities that add nothing beyond the region
    # name and storage scaffolding.
    return bool(key_tokens) and key_tokens <= (
        _CAPSULE_KEY_SCAFFOLD_TOKENS | region_tokens
    )


def _validate_generic_create_partition_keys(
    region_key: Any, patch: Mapping[str, Any]
) -> None:
    """Reject only high-confidence generic multi-topic create identities."""
    if not _is_generic_region_key(region_key):
        return
    for index, operation in enumerate(patch.get("operations", [])):
        if not isinstance(operation, Mapping) or operation.get("action") != "create":
            continue
        claims = operation.get("claims")
        if not isinstance(claims, list):
            continue
        slots = {
            _clean(claim.get("canonical_slot"))
            for claim in claims
            if isinstance(claim, Mapping) and _clean(claim.get("canonical_slot"))
        }
        if len(slots) > 1 and _capsule_key_is_generic_for_region(
            region_key, operation.get("capsule_key")
        ):
            raise PatchValidationError(
                f"operations[{index}].{GENERIC_MULTI_SLOT_CAPSULE_KEY_ERROR}"
            )


def _generic_region_requires_semantic_management(
    region_key: Any,
    evidence: list[Mapping[str, Any]],
    capsules: list[Mapping[str, Any]],
) -> bool:
    if not _is_generic_region_key(region_key):
        return False
    slots = {
        _leaf_slot(item)
        for item in evidence
        if _is_current_durable(item) or _is_challenged_durable(item)
    }
    for capsule in capsules:
        if _clean(capsule.get("status")).casefold() not in {"active", "challenged"}:
            continue
        for claim in capsule.get("claims") or []:
            if isinstance(claim, Mapping) and _clean(claim.get("canonical_slot")):
                slots.add(_clean(claim.get("canonical_slot")))
    return len(slots) > 1


def _semantic_partition_targets(
    region_key: Any, capsules: list[Mapping[str, Any]]
) -> set[str]:
    targets: set[str] = set()
    for capsule in capsules:
        if _clean(capsule.get("status")).casefold() not in {"active", "challenged"}:
            continue
        claims = capsule.get("claims")
        projected_length = (
            sum(
                len(" ".join(_clean(claim.get("text")).split()))
                for claim in claims
                if isinstance(claim, Mapping)
            )
            + max(0, len(claims) - 1)
            if isinstance(claims, list)
            else 0
        )
        if (
            isinstance(claims, list)
            and claims
            and (
                projected_length > SLOW_SUMMARY_MAX_CHARS
                or capsule.get("partition_contract_version")
                != SLOW_PARTITION_CONTRACT_VERSION
            )
        ):
            targets.add(_required_text(capsule.get("capsule_id"), "capsule_id"))
    return targets


def _partition_targets_require_model(
    capsules: list[Mapping[str, Any]], targets: set[str]
) -> bool:
    for capsule in capsules:
        capsule_id = _clean(capsule.get("capsule_id"))
        if capsule_id not in targets:
            continue
        claims = capsule.get("claims")
        if not isinstance(claims, list) or len(claims) != 1:
            return True
        try:
            _semantic_summary_projection(claims)
        except PatchValidationError:
            return True
    return False


def _canonical_patch_claims(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise PatchValidationError("claims must be a non-empty list")
    claims = [
        _patch_claim_projection(claim)
        for claim in value
        if isinstance(claim, Mapping)
    ]
    if len(claims) != len(value):
        raise PatchValidationError("each claim must be an object")
    claims.sort(
        key=lambda claim: (
            claim["canonical_slot"],
            _normal_text(claim["text"]),
            tuple(claim["support"]),
            tuple(claim["counterevidence"]),
        )
    )
    return claims


def _materialize_lossless_summaries(patch: Mapping[str, Any]) -> dict[str, Any]:
    operations = patch.get("operations")
    if not isinstance(operations, list):
        raise PatchValidationError("lossless summary materialization requires operations")
    normalized: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise PatchValidationError("GraphPatch operation must be an object")
        current = dict(operation)
        if _clean(current.get("action")) != "noop":
            claims = _canonical_patch_claims(current.get("claims"))
            current["claims"] = claims
            current["summary"] = _semantic_summary_projection(claims)
        normalized.append(current)
    return {"operations": normalized}


def validate_v4_patch(
    patch: Mapping[str, Any], *, require_lossless_summary: bool = False
) -> None:
    if (
        not isinstance(patch, Mapping)
        or set(patch) != {"operations"}
        or not isinstance(patch["operations"], list)
    ):
        raise PatchValidationError("GraphPatch must contain exactly an operations list")
    operations = patch["operations"]
    if not operations:
        raise PatchValidationError("GraphPatch must contain at least one operation")
    if len(operations) > SLOW_MAX_REGION_OPERATIONS:
        raise PatchValidationError(
            f"GraphPatch exceeds {SLOW_MAX_REGION_OPERATIONS} region operations"
        )

    create_keys: set[str] = set()
    capsule_targets: set[str] = set()
    noop_count = 0
    for index, operation in enumerate(operations):
        if not isinstance(operation, Mapping):
            raise PatchValidationError("GraphPatch operation must be an object")
        action = _required_text(operation.get("action"), "action")
        if action not in PATCH_ACTIONS:
            raise PatchValidationError("unknown GraphPatch action")
        if "confidence" in operation:
            raise PatchValidationError(
                "model confidence is not an authoritative graph field"
            )
        allowed = {"action", "summary", "claims"}
        if action == "create":
            allowed.update({"capsule_key", "capsule_id"})
            capsule_key = _normalize_capsule_key(
                operation.get("capsule_key"),
                f"operations[{index}].capsule_key",
            )
            if capsule_key in create_keys:
                raise PatchValidationError(
                    f"duplicate create capsule_key in one patch: {capsule_key}"
                )
            create_keys.add(capsule_key)
            if "capsule_id" in operation and operation.get("capsule_id") is not None:
                raise PatchValidationError(
                    "create capsule identity is controller-derived from capsule_key"
                )
        elif action == "noop":
            noop_count += 1
            allowed.add("capsule_id")
            if "capsule_id" in operation:
                capsule_id = _required_text(operation.get("capsule_id"), "capsule_id")
                if capsule_id in capsule_targets:
                    raise PatchValidationError(
                        f"duplicate capsule target in one patch: {capsule_id}"
                    )
                capsule_targets.add(capsule_id)
        else:
            allowed.update({"capsule_id", "base_revision"})
            capsule_id = _required_text(operation.get("capsule_id"), "capsule_id")
            if capsule_id in capsule_targets:
                raise PatchValidationError(
                    f"duplicate capsule target in one patch: {capsule_id}"
                )
            capsule_targets.add(capsule_id)
            if (
                isinstance(operation.get("base_revision"), bool)
                or not isinstance(operation.get("base_revision"), int)
                or operation["base_revision"] < 1
            ):
                raise PatchValidationError("base_revision must be a positive integer")
        if set(operation) - allowed:
            raise PatchValidationError("unexpected GraphPatch operation fields")
        if action != "noop":
            _v3._validate_claims(operation.get("claims"))

    if noop_count and len(operations) != 1:
        raise PatchValidationError("noop must be the only operation in a GraphPatch")
    if require_lossless_summary:
        _validate_patch_summary_contract(patch)


# V4 owns a multi-capsule GraphPatch schema; V3 remains single-capsule.
validate_patch = validate_v4_patch


def _forbidden_field(name: str) -> bool:
    key = _clean(name).casefold()
    if key in {"qid", "query_id", "answer_id", "answer_session_id"}:
        return True
    return any(token in key for token in ("benchmark", "question", "answer", "judge", "gold", "label"))


def _assert_no_benchmark_fields(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _forbidden_field(str(key)):
                raise EvidencePolicyError(f"benchmark field is forbidden in slow-graph request: {path}.{key}")
            _assert_no_benchmark_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_benchmark_fields(item, f"{path}[{index}]")


@dataclass(frozen=True)
class DeepSeekTierConfig:
    base_url: str
    key_pool: tuple[str, ...]
    max_tokens: int
    model: str = "deepseek-v4-flash"
    prompt_cost_per_million: float = 0.0
    completion_cost_per_million: float = 0.0
    cache_cost_per_million: float = 0.0

    @classmethod
    def from_env(cls, prefix: str, *, model: str) -> "DeepSeekTierConfig":
        base_url = _clean(os.getenv(prefix + "_BASE_URL"))
        keys = tuple(
            item.strip()
            for item in _clean(os.getenv(prefix + "_KEY_POOL")).split(",")
            if item.strip()
        )
        try:
            max_tokens = int(_clean(os.getenv(prefix + "_MAX_TOKENS")))
        except ValueError as exc:
            raise SlowGraphError(f"{prefix}_MAX_TOKENS must be an integer") from exc
        if not base_url or not keys or max_tokens <= 0:
            raise SlowGraphError(f"{prefix} requires BASE_URL, KEY_POOL, and positive MAX_TOKENS")
        return cls(
            base_url.rstrip("/"),
            keys,
            max_tokens,
            model=model,
            prompt_cost_per_million=float(os.getenv(prefix + "_PROMPT_COST_PER_MILLION", "0")),
            completion_cost_per_million=float(os.getenv(prefix + "_COMPLETION_COST_PER_MILLION", "0")),
            cache_cost_per_million=float(os.getenv(prefix + "_CACHE_COST_PER_MILLION", "0")),
        )


DeepSeekFlashConfig = DeepSeekTierConfig


@dataclass(frozen=True)
class DeepSeekProConfig(DeepSeekTierConfig):
    model: str = "deepseek-v4-pro"


def _optional_config(prefix: str, model: str) -> DeepSeekTierConfig | None:
    values = [os.getenv(prefix + suffix) for suffix in ("_BASE_URL", "_KEY_POOL", "_MAX_TOKENS")]
    if not any(_clean(value) for value in values):
        return None
    return DeepSeekTierConfig.from_env(prefix, model=model)


class _DeepSeekTierClient:
    def __init__(self, config: DeepSeekTierConfig, *, route: str) -> None:
        expected_model = {
            "flash": "deepseek-v4-flash",
            "pro": "deepseek-v4-pro",
        }.get(route)
        if expected_model is None or config.model != expected_model:
            raise SlowGraphError(
                f"slow-graph route {route!r} requires model {expected_model!r}, "
                f"got {config.model!r}"
            )
        self.config = config
        self.route = route
        self._key_index = 0
        self.last_call_metadata: Mapping[str, Any] = {}

    def _messages(
        self,
        region: Mapping[str, Any],
        capsules: list[Mapping[str, Any]],
        *,
        correction: Mapping[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        required_evidence_ids = sorted(
            {
                _required_text(item, "required evidence ID")
                for item in region.get("required_evidence_ids", [])
            }
        )
        partition_required = region.get("semantic_partition_required") is True
        partition_mode = _clean(region.get("semantic_partition_mode"))
        partition_capsule_ids = sorted(
            {
                _required_text(item, "partition capsule ID")
                for item in region.get("partition_capsule_ids", [])
            }
        )
        if not partition_required and (partition_mode or partition_capsule_ids):
            raise EvidencePolicyError(
                "semantic partition metadata requires semantic_partition_required"
            )
        if partition_required and partition_mode not in {"manage", "migrate"}:
            raise EvidencePolicyError("semantic partition mode is invalid")
        if partition_mode == "migrate" and not partition_capsule_ids:
            raise EvidencePolicyError(
                "semantic partition migration requires explicit partition_capsule_ids"
            )
        if partition_mode == "manage" and partition_capsule_ids:
            raise EvidencePolicyError(
                "generic semantic management cannot name legacy partition targets"
            )
        route_instruction = (
            "The controller selected compatible consolidation. Assign each new atomic "
            "claim to one coherent existing or new capsule."
            if self.route == "flash"
            else
            "The controller selected full-state adjudication. Resolve genuine same-property "
            "conflicts while preserving uncertainty when the supplied evidence does not "
            "establish one current value."
        )
        if capsules and required_evidence_ids:
            allowed_actions = (
                "revise or create"
                if self.route == "flash"
                else "revise, create, challenge, resolve_challenge, or retire"
            )
            identity_instruction = (
                "For revise, challenge, resolve_challenge, or retire, copy capsule_id and "
                "base_revision exactly from one supplied capsule. For create, emit a stable "
                "lowercase dot-separated capsule_key and omit capsule_id/base_revision. "
                "noop is forbidden because uncited durable evidence is pending."
            )
        elif not capsules and required_evidence_ids:
            allowed_actions = "one or more create operations"
            identity_instruction = (
                "Every create operation contains a unique stable lowercase dot-separated "
                "capsule_key and omits capsule_id/base_revision; the controller derives "
                "capsule identity. noop is forbidden because uncited durable evidence is pending."
            )
        elif capsules:
            allowed_actions = (
                "revise, create, or noop"
                if self.route == "flash"
                else "revise, create, challenge, resolve_challenge, retire, or noop"
            )
            identity_instruction = (
                "For operations on supplied capsules, copy capsule_id and base_revision "
                "exactly. For create, provide a unique stable lowercase dot-separated "
                "capsule_key and omit capsule_id/base_revision."
            )
        else:
            allowed_actions = "one or more create operations, or one noop"
            identity_instruction = (
                "Every create operation contains a unique stable lowercase dot-separated "
                "capsule_key and omits capsule_id/base_revision."
            )
        if self.route == "flash" and required_evidence_ids:
            allowed_actions += " or escalate"
        delta_instruction = (
            "This is an additive delta proposal. Every operation claim must cite only IDs in "
            "region.required_evidence_ids and describe only those new evidence items. Use revise "
            "when a delta belongs in one supplied capsule and create when it forms a different "
            "semantic topic. Do not repeat or cite existing capsule claims; the controller will "
            "merge each revise delta after validation. "
            if self.route == "flash" and capsules and required_evidence_ids
            else
            "For each operated existing capsule, claims are its complete next revision. Preserve "
            "every still-current supplied claim unless supplied evidence resolves it or the claim "
            "is moved intact to a newly created coherent capsule. "
            if capsules
            else ""
        )
        migration_partition_instruction = (
            "semantic_partition_required is true. The supplied legacy capsule mixes semantic "
            "topics. Operate on every capsule ID listed in region.partition_capsule_ids and "
            "repartition all of its current claims "
            "across coherent next capsules. Keep one coherent group on a revised supplied "
            "capsule and create additional capsules as needed; never leave a supplied capsule "
            "listed for partition untouched, duplicate a claim, or group unrelated topics merely "
            "because they share a generic region name. Other supplied capsules are context and "
            "may remain untouched unless one must be revised to avoid duplicating the same "
            "self-contained semantic claim across capsules. "
        )
        managed_partition_instruction = (
            "semantic_partition_required is true because this region needs explicit multi-slot "
            "semantic management. Manage the complete supplied durable state as coherent "
            "real-world topics, revising coherent existing capsules and creating additional "
            "capsules only when the real-world topics differ. Never group claims merely because "
            "they share the region name, but do not split related claims merely because their "
            "canonical slots or claim types differ. "
            "Every multi-slot create must use a concrete topic-specific capsule_key; keys made "
            "only from user, region, or claim-type words are forbidden. "
        )
        topic_granularity_instruction = (
            "A capsule is one reusable retrieval concept centered on the same real-world person, "
            "object, activity, project, relationship, decision, or behavioral objective; it is "
            "not a schema field or claim-type bucket. First cluster by that shared referent and "
            "intent. Keep facts, preferences, goals, constraints, plans, and routines together "
            "when they jointly describe that concept, including a goal and the plan or routine "
            "used to achieve it. Different canonical slots, claim types, or timestamps alone do "
            "not justify separate capsules. Conversely, a shared broad region such as business, "
            "work, schedule, goals, or reading does not justify grouping independent concrete "
            "referents. Prefer real-world co-reference over taxonomy: details about one product "
            "and a model used for that product belong together, while an unrelated second "
            "business remains separate; never regroup them as identity versus possession. For "
            "each proposed capsule, there must be a natural memory-retrieval question for which "
            "every claim in that capsule is useful evidence. Name capsule_key after the concrete "
            "referent or retrieval use-case, never after a broad region or schema role. Use a "
            "singleton capsule only when no other supplied current claim shares its real-world "
            "topic, and use multiple operations only when the groups would normally be retrieved "
            "independently. "
        )
        partition_instruction = (
            migration_partition_instruction
            if partition_mode == "migrate"
            else managed_partition_instruction
            if partition_mode == "manage"
            else
            "Use multiple operations only when the claims form genuinely different semantic "
            "topics; keep related facts, preferences, goals, and routines together. "
        )
        conflict_instruction = (
            "If support-role evidence with different canonical slots may describe mutually "
            "exclusive values of the same changing property, do not choose a winner. Return "
            "exactly {\"operations\":[{\"action\":\"escalate\","
            "\"reason\":\"cross_slot_conflict\"}]} so the controller can route to Pro. "
            if self.route == "flash" and required_evidence_ids
            else "First decide whether the evidence actually conflicts. Two statements are "
            "compatible unless they cannot both be true of the same subject and changing property "
            "at the same time. A shared topic, different canonical slots, negative wording, or "
            "different details about one topic is not a conflict. Compatible statements must be "
            "separate support-only claims. Never emit reciprocal claims that use each other's "
            "support as counterevidence. Use turn_index and temporal_status only for a genuine "
            "same-property state change. Prefer a later explicit current observation or correction "
            "only when it resolves that same property; otherwise preserve uncertainty with one "
            "challenge claim rather than mirrored alternatives. A create operation cannot contain "
            "counterevidence because a new unresolved capsule must not be committed as active. "
        )
        system = (
            "Return exactly one JSON GraphPatch and no prose. The top-level object must contain "
            "exactly one key named operations; never echo the user envelope, region, capsules, "
            f"route, schema, or schema_version. operations contains 1 to {SLOW_MAX_REGION_OPERATIONS} "
            "atomic capsule operations. "
            f"The only allowed actions for this request are {allowed_actions}. {identity_instruction} "
            "A noop operation contains only action and optionally one supplied capsule_id, and "
            "must be the only operation. Every non-noop operation contains a non-empty claims list. "
            "Do not return summary; the controller deterministically derives the committed lossless "
            "summary from the final claims after merge. "
            "Each claim contains only canonical_slot, text, support, and counterevidence. "
            "support and counterevidence are arrays of supplied fast evidence IDs. "
            "A claim may cite multiple support IDs only when their normalized evidence texts "
            "are identical. When same-slot evidence texts differ, preserve each evidence meaning "
            "as a separate claim in the same coherent capsule; never compress distinct Fast "
            "evidence texts into one claim. "
            "One indivisible Fast evidence ID may itself name multiple parallel concrete "
            "referents. Never duplicate or split that evidence ID across claims or capsules. "
            "Preserve its complete compound statement as one self-contained claim in one "
            "concrete shared retrieval-use-case capsule that naturally covers every named "
            "referent. "
            "Claim text must be a self-contained user-memory statement: name the subject and "
            "object needed to understand it without neighboring claims, resolve pronouns or "
            "deictic phrases only when supplied evidence makes the referent explicit, and never "
            "invent a referent. Preserve epistemic force exactly (for example, heard, suspects, "
            "plans, prefers, and knows are not interchangeable), preserve quantities and temporal "
            "qualifiers, and do not broaden or weaken the atomic fact. "
            "Use only supplied fast evidence. polarity describes the statement's content; it does "
            "not make a negative statement counterevidence. evidence_role is authoritative: every "
            "Fast evidence whose record_state is challenged, superseded, or otherwise non-current "
            "must never become a new support binding or create a new active capsule. It may remain "
            "as historical support only when an existing claim is explicitly adjudicated against "
            "current replacement evidence in counterevidence. "
            "support-role required ID belongs in support, and every support ID must be attached to "
            "a claim whose canonical_slot exactly equals that evidence item's canonical_slot. "
            "Never consume a distinct slot by attaching its ID to another slot's claim. "
            "Every non-noop claim must cite supplied evidence IDs and preserve canonical slots. "
            "Every ID in region.required_evidence_ids must appear exactly once as support or "
            "counterevidence across all operation claims. canonical_slot is an attribute type, "
            "not a global entity identity: it may appear in different capsules when the claims "
            "name different concrete real-world referents. Claims about the same referent and "
            "retrieval topic belong in one capsule, and an identical self-contained semantic "
            "claim must never be duplicated across capsules. "
            + delta_instruction
            + topic_granularity_instruction
            + partition_instruction
            + conflict_instruction
            +
            "Do not invent capsule IDs, source IDs, evidence, confidence, benchmark fields, or fields "
            "outside this GraphPatch contract. " + route_instruction + " "
            "Do not change routes or repair invalid input locally."
        )
        if correction is not None:
            if self.route != "pro":
                raise EvidencePolicyError("semantic correction requires the Pro route")
            rejected_patch = correction.get("rejected_patch")
            validation_error = _clean(correction.get("validation_error"))
            if not isinstance(rejected_patch, Mapping) or not validation_error:
                raise EvidencePolicyError("semantic correction context is incomplete")
            _assert_no_benchmark_fields(rejected_patch)
            system += (
                " This is the single allowed correction pass. The previous Pro GraphPatch "
                "was rejected by the deterministic controller validator. Return a complete "
                "replacement GraphPatch, not a commentary or partial edit. Correct the "
                "underlying semantic partition when topics differ; do not merely rename a "
                "generic capsule that still mixes unrelated claims, and do not react by putting "
                "each related claim into its own singleton capsule. The rejected patch was "
                + _json(rejected_patch)
                + " The exact validator error was: "
                + validation_error[:2000]
                + "."
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": _json({"region": region, "capsules": capsules})},
        ]

    @staticmethod
    def _usage(raw: Mapping[str, Any]) -> dict[str, int]:
        def integer(*names: str, required: bool = False) -> tuple[int, bool]:
            for name in names:
                if raw.get(name) is None:
                    continue
                value = raw.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) < 0:
                    raise TieredAPIError(f"usage.{name} is invalid")
                return int(value), True
            if required:
                raise TieredAPIError(f"usage.{names[0]} is missing")
            return 0, False

        prompt, _ = integer("prompt_tokens", "input_tokens", required=True)
        completion, _ = integer("completion_tokens", "output_tokens", required=True)
        cached, has_cached = integer(
            "prompt_cache_hit_tokens", "cache_read_input_tokens", "cached_tokens"
        )
        cache_miss, has_miss = integer(
            "prompt_cache_miss_tokens", "cache_miss_input_tokens"
        )
        if cached > prompt or cache_miss > prompt:
            raise TieredAPIError("cache usage exceeds prompt tokens")
        if has_cached and has_miss and cached + cache_miss != prompt:
            raise TieredAPIError("cache hit and miss usage does not balance prompt tokens")
        if not has_miss:
            cache_miss = prompt - cached
        if not has_cached:
            cached = prompt - cache_miss
        total, has_total = integer("total_tokens")
        if has_total and total < prompt + completion:
            raise TieredAPIError("usage.total_tokens is smaller than prompt plus completion")
        if not has_total:
            total = prompt + completion
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cache_read_input_tokens": cached,
            "cache_hit_tokens": cached,
            "cache_miss_tokens": cache_miss,
            "total_tokens": total,
        }

    def _metadata(self, **values: Any) -> dict[str, Any]:
        return {
            "route": self.route,
            "prompt_version": SLOW_PROMPT_VERSION,
            "physical_api_call": True,
            "physical_api_calls": 1,
            "api_provider": "deepseek",
            "model": self.config.model,
            "attempt_count": 1,
            **values,
        }

    def propose(
        self, region: Mapping[str, Any], capsules: list[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        return self._propose(region, capsules, correction=None)

    def correct(
        self,
        region: Mapping[str, Any],
        capsules: list[Mapping[str, Any]],
        *,
        rejected_patch: Mapping[str, Any],
        validation_error: str,
    ) -> Mapping[str, Any]:
        if self.route != "pro":
            raise TieredAPIError("semantic correction is available only on the Pro route")
        return self._propose(
            region,
            capsules,
            correction={
                "rejected_patch": dict(rejected_patch),
                "validation_error": _required_text(
                    validation_error, "semantic correction validation error"
                ),
            },
        )

    def _propose(
        self,
        region: Mapping[str, Any],
        capsules: list[Mapping[str, Any]],
        *,
        correction: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        _assert_no_benchmark_fields(region)
        _assert_no_benchmark_fields(capsules)
        key_index = self._key_index % len(self.config.key_pool)
        self._key_index += 1
        body = {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "thinking": {"type": "disabled"},
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
            "messages": self._messages(region, capsules, correction=correction),
        }
        saved_request = {**body, "headers": {"authorization": "redacted"}}
        request_sha256 = _digest(saved_request)
        physical_call_id = "dsc_" + uuid.uuid4().hex
        started = time.time()
        self.last_call_metadata = self._metadata(
            physical_call_id=physical_call_id,
            key_index=key_index,
            started_at=started,
            status="started",
            request=saved_request,
            request_sha256=request_sha256,
        )
        request = urllib.request.Request(
            self.config.base_url + "/chat/completions",
            data=_json(body).encode("utf-8"),
            headers={"Authorization": "Bearer " + self.config.key_pool[key_index], "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:  # nosec B310
                status = response.getcode()
                raw_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")
            except (AttributeError, OSError):
                detail = ""
            self.last_call_metadata = self._metadata(
                physical_call_id=physical_call_id,
                key_index=key_index,
                started_at=started,
                completed_at=time.time(),
                latency_ms=round((time.time() - started) * 1000, 3),
                status="http_error",
                http_status=exc.code,
                error=detail,
                request=saved_request,
                request_sha256=request_sha256,
            )
            raise TieredAPIError(f"{self.route} HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.last_call_metadata = self._metadata(
                physical_call_id=physical_call_id,
                key_index=key_index,
                started_at=started,
                completed_at=time.time(),
                latency_ms=round((time.time() - started) * 1000, 3),
                status="request_error",
                error=f"{exc.__class__.__name__}: {exc}",
                request=saved_request,
                request_sha256=request_sha256,
            )
            raise TieredAPIError(f"{self.route} transport failure: {exc}") from exc
        completed = time.time()
        self.last_call_metadata = self._metadata(
            physical_call_id=physical_call_id,
            key_index=key_index,
            started_at=started,
            completed_at=completed,
            latency_ms=round((completed - started) * 1000, 3),
            status="response_received_unvalidated",
            http_status=status,
            raw_response=raw_text,
            request=saved_request,
            request_sha256=request_sha256,
        )
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            self.last_call_metadata = self._metadata(
                physical_call_id=physical_call_id,
                started_at=started,
                completed_at=completed,
                latency_ms=round((completed - started) * 1000, 3),
                status="invalid_json",
                http_status=status,
                raw_response=raw_text,
                request=saved_request,
                request_sha256=request_sha256,
            )
            raise TieredAPIError(f"{self.route} response is not JSON") from exc
        if not isinstance(raw, Mapping):
            raise TieredAPIError(f"{self.route} response must be an object")
        choices = raw.get("choices")
        usage = raw.get("usage")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(usage, Mapping):
            raise TieredAPIError(f"{self.route} response missing strict choices/usage")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, Mapping) else None
        finish_reason = _clean(choice.get("finish_reason")) if isinstance(choice, Mapping) else ""
        content = _clean(message.get("content")) if isinstance(message, Mapping) else ""
        normalized_usage = self._usage(usage)
        cost = {
            "prompt_tokens": normalized_usage["prompt_tokens"],
            "cache_hit_tokens": normalized_usage["cache_hit_tokens"],
            "cache_miss_tokens": normalized_usage["cache_miss_tokens"],
            "completion_tokens": normalized_usage["completion_tokens"],
            "cache_read_input_tokens": normalized_usage["cache_read_input_tokens"],
            "prompt_cost_per_million": self.config.prompt_cost_per_million,
            "completion_cost_per_million": self.config.completion_cost_per_million,
            "cache_cost_per_million": self.config.cache_cost_per_million,
            "estimated_cost": (
                normalized_usage["cache_miss_tokens"] * self.config.prompt_cost_per_million
                + normalized_usage["cache_hit_tokens"] * self.config.cache_cost_per_million
                + normalized_usage["completion_tokens"] * self.config.completion_cost_per_million
            )
            / 1_000_000,
        }
        self.last_call_metadata = self._metadata(
            physical_call_id=physical_call_id,
            started_at=started,
            completed_at=completed,
            latency_ms=round((completed - started) * 1000, 3),
            status="response_received",
            http_status=status,
            response_id=_clean(raw.get("id")),
            finish_reason=finish_reason,
            content=content,
            usage=normalized_usage,
            provider_usage=dict(usage),
            cost_audit=cost,
            raw_response=raw_text,
            request=saved_request,
            request_sha256=request_sha256,
        )
        if status < 200 or status >= 300:
            raise TieredAPIError(f"{self.route} returned HTTP {status}")
        if finish_reason != "stop":
            self.last_call_metadata = {**self.last_call_metadata, "status": "incomplete_response"}
            raise TieredAPIError(f"{self.route} finish_reason must be stop, got {finish_reason!r}")
        if not content:
            raise TieredAPIError(f"{self.route} response content is empty")
        try:
            raw_patch = json.loads(content)
        except json.JSONDecodeError as exc:
            raise TieredAPIError(f"{self.route} content is not JSON") from exc
        if self.route == "flash" and _flash_escalation_patch(raw_patch):
            required_evidence_ids = region.get("required_evidence_ids")
            if not isinstance(required_evidence_ids, list) or not required_evidence_ids:
                raise TieredAPIError(
                    "flash escalation requires pending durable evidence"
                )
            self.last_call_metadata = {
                **self.last_call_metadata,
                "status": "completed",
                "escalation_requested": True,
                "escalation_reason": FLASH_ESCALATION_REASON,
                "raw_patch_sha256": _digest(raw_patch),
            }
            return raw_patch
        patch, transport_normalizations = _normalize_transport_patch(
            raw_patch, capsules, region
        )
        if transport_normalizations:
            self.last_call_metadata = {
                **self.last_call_metadata,
                "transport_normalizations": transport_normalizations,
                "raw_patch_sha256": _digest(raw_patch),
                "normalized_patch_sha256": _digest(patch),
            }
        try:
            validate_patch(patch)
        except PatchValidationError as exc:
            raise TieredAPIError(f"{self.route} returned an invalid GraphPatch: {exc}") from exc
        self.last_call_metadata = {**self.last_call_metadata, "status": "completed"}
        return patch


def _leaf_metadata(leaf: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = leaf.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _leaf_id(leaf: Mapping[str, Any]) -> str:
    return _required_text(leaf.get("memory_id"), "fast evidence memory_id")


def _leaf_slot(leaf: Mapping[str, Any]) -> str:
    metadata = _leaf_metadata(leaf)
    return _required_text(metadata.get("canonical_slot_key") or metadata.get("canonical_slot"), "fast evidence canonical slot")


def _leaf_text(leaf: Mapping[str, Any]) -> str:
    metadata = _leaf_metadata(leaf)
    return _required_text(leaf.get("value") or metadata.get("source_span") or metadata.get("raw_content"), "fast evidence value")


def _leaf_state(leaf: Mapping[str, Any]) -> str:
    metadata = _leaf_metadata(leaf)
    state = _clean(
        leaf.get("record_state") or leaf.get("state") or metadata.get("record_state")
    ).casefold()
    if not state:
        raise EvidencePolicyError("fast evidence record state is missing")
    return state


def _is_current_durable(leaf: Mapping[str, Any]) -> bool:
    """Return whether the durable memory record is currently authoritative.

    target_status describes when the remembered fact, goal, or plan applies.  It
    is not a lifecycle state: an active durable memory about the past, future,
    or a planned action remains eligible until the Writer supersedes it.
    """
    metadata = _leaf_metadata(leaf)
    durability = _clean(metadata.get("durability") or metadata.get("durability_class")).casefold()
    if durability in {"episodic", "uncertain", "", "none"}:
        return False
    if durability not in {"durable", "long_term", "long-term", "hard", "persistent"}:
        return False
    return _leaf_state(leaf) in {"active", "parallel_active", "promoted"}


def _is_challenged_durable(leaf: Mapping[str, Any]) -> bool:
    metadata = _leaf_metadata(leaf)
    durability = _clean(
        metadata.get("durability") or metadata.get("durability_class")
    ).casefold()
    return (
        durability in {"durable", "long_term", "long-term", "hard", "persistent"}
        and _leaf_state(leaf) == "challenged"
    )


def _is_uncertain(leaf: Mapping[str, Any]) -> bool:
    metadata = _leaf_metadata(leaf)
    return _clean(metadata.get("durability") or metadata.get("durability_class")).casefold() == "uncertain"


def _is_episodic(leaf: Mapping[str, Any]) -> bool:
    metadata = _leaf_metadata(leaf)
    return _clean(metadata.get("durability") or metadata.get("durability_class")).casefold() == "episodic"


def _is_counterevidence(leaf: Mapping[str, Any]) -> bool:
    metadata = _leaf_metadata(leaf)
    return bool(metadata.get("counterevidence")) or bool(
        metadata.get("is_counterevidence")
    )


def _flash_escalation_patch(value: Any) -> bool:
    return value == {
        "operations": [
            {"action": "escalate", "reason": FLASH_ESCALATION_REASON}
        ]
    }


def _prior_counterevidence_ids(capsules: list[Mapping[str, Any]]) -> set[str]:
    output: set[str] = set()
    for capsule in capsules:
        claims = capsule.get("claims")
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if not isinstance(claim, Mapping):
                continue
            values = claim.get("counterevidence")
            if isinstance(values, list):
                output.update(_clean(item) for item in values if _clean(item))
    return output


def _support_text_groups(
    evidence_ids: Iterable[str],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for evidence_id in sorted(set(evidence_ids)):
        leaf = evidence_by_id.get(evidence_id)
        if leaf is None:
            continue
        text = _normal_text(_leaf_text(leaf))
        if text:
            groups.setdefault(text, []).append(evidence_id)
    return groups


def _controlled_complementary_support_bundle(
    claim_slot: str,
    evidence_ids: Iterable[str],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Recognize one parent fact plus compatible durable subslot refinements."""
    unique_ids = sorted(set(evidence_ids))
    if len(unique_ids) < 2:
        return False
    leaves = [evidence_by_id.get(evidence_id) for evidence_id in unique_ids]
    if any(
        leaf is None
        or not _is_current_durable(leaf)
        or _is_counterevidence(leaf)
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
    if any(len(texts) != 1 for texts in texts_by_slot.values()):
        return False

    def one_shared_metadata_value(key: str) -> bool:
        values = {
            _clean(_leaf_metadata(leaf).get(key)).casefold()
            for leaf in typed_leaves
        }
        return len(values) == 1 and "" not in values

    if not all(
        one_shared_metadata_value(key)
        for key in ("subject_signature", "graph_entity_key", "memory_family")
    ):
        return False
    relations = {
        _clean(leaf.get("relation") or _leaf_metadata(leaf).get("semantic_slot")).casefold()
        for leaf in typed_leaves
    }
    polarities = {
        _clean(_leaf_metadata(leaf).get("polarity")).casefold()
        for leaf in typed_leaves
    }
    return (
        len(relations) == 1
        and "" not in relations
        and len(polarities) == 1
        and "" not in polarities
    )


def _validate_repeated_evidence_bindings(
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    bindings_by_id: Mapping[str, list[tuple[str, str, str, str]]],
) -> None:
    """Allow compound current support to fan out without weakening provenance rules."""
    for evidence_id, bindings in sorted(bindings_by_id.items()):
        if len(bindings) <= 1:
            continue
        roles = {role for role, _, _, _ in bindings}
        if roles != {"support"}:
            raise PatchValidationError(
                "repeated Fast evidence bindings must be support-only; "
                "counterevidence and mixed-role fan-out are forbidden: "
                + evidence_id
            )
        leaf = evidence_by_id.get(evidence_id)
        if (
            leaf is None
            or not _is_current_durable(leaf)
            or _is_counterevidence(leaf)
        ):
            raise PatchValidationError(
                "only supplied current durable support may bind multiple Slow claims: "
                + evidence_id
            )
        claim_identities = [(slot, text) for _, slot, text, _ in bindings]
        if len(set(claim_identities)) != len(claim_identities):
            locations = [location for _, _, _, location in bindings]
            raise PatchValidationError(
                "one semantic claim cannot duplicate a Fast evidence binding: "
                + _json({evidence_id: locations})
            )


def _validate_claim_evidence_contract(
    region: Mapping[str, Any],
    capsules: list[Mapping[str, Any]],
    patch: Mapping[str, Any],
    *,
    route: str = "",
) -> None:
    """Preserve exact provenance while keeping Flash from inventing conflict."""
    evidence = [
        item for item in region.get("evidence", []) if isinstance(item, Mapping)
    ]
    evidence_by_id = {_leaf_id(item): item for item in evidence}
    operations = patch.get("operations")
    if not isinstance(operations, list) or not operations:
        raise PatchValidationError("claim evidence contract requires operations")
    if len(operations) == 1 and isinstance(operations[0], Mapping) and operations[0].get("action") == "noop":
        return
    prior_counterevidence = _prior_counterevidence_ids(capsules)
    prior_support_bindings: set[tuple[str, str, str]] = set()
    prior_counter_bindings: set[tuple[str, str, str]] = set()
    for capsule in capsules:
        prior_claims = capsule.get("claims")
        if not isinstance(prior_claims, list):
            continue
        for prior_claim in prior_claims:
            if not isinstance(prior_claim, Mapping):
                continue
            prior_slot = _clean(prior_claim.get("canonical_slot"))
            prior_text = _normal_text(prior_claim.get("text"))
            if not prior_slot or not prior_text:
                continue
            for evidence_id in prior_claim.get("support") or []:
                if _clean(evidence_id):
                    prior_support_bindings.add(
                        (prior_slot, prior_text, _clean(evidence_id))
                    )
            for evidence_id in prior_claim.get("counterevidence") or []:
                if _clean(evidence_id):
                    prior_counter_bindings.add(
                        (prior_slot, prior_text, _clean(evidence_id))
                    )
    support_ids: set[str] = set()
    citation_bindings: dict[str, list[tuple[str, str, str, str]]] = {}
    claim_roles: list[tuple[set[str], set[str]]] = []
    for operation_index, operation in enumerate(operations):
        if not isinstance(operation, Mapping):
            raise PatchValidationError("claim evidence contract received malformed operation")
        action = _clean(operation.get("action"))
        if action == "noop":
            continue
        claims = operation.get("claims")
        if not isinstance(claims, list):
            raise PatchValidationError("claim evidence contract requires claims")
        for claim_index, claim in enumerate(claims):
            if not isinstance(claim, Mapping):
                raise PatchValidationError("claim evidence contract received malformed claim")
            claim_slot = _required_text(
                claim.get("canonical_slot"), "claim canonical slot"
            )
            claim_text = _normal_text(claim.get("text"))
            claim_support: set[str] = set()
            claim_counter: set[str] = set()
            normalized_support_ids = [
                _required_text(evidence_id, "claim support evidence ID")
                for evidence_id in claim.get("support") or []
            ]
            complementary_support_bundle = _controlled_complementary_support_bundle(
                claim_slot, normalized_support_ids, evidence_by_id
            )
            for normalized_id in normalized_support_ids:
                leaf = evidence_by_id.get(normalized_id)
                if leaf is None:
                    if (claim_slot, claim_text, normalized_id) not in prior_support_bindings:
                        raise PatchValidationError(
                            f"claim support is absent from supplied evidence: {normalized_id}"
                        )
                else:
                    evidence_slot = _leaf_slot(leaf)
                    if (
                        evidence_slot != claim_slot
                        and not complementary_support_bundle
                        and (claim_slot, claim_text, normalized_id)
                        not in prior_support_bindings
                    ):
                        raise PatchValidationError(
                            "claim support canonical slot mismatch: "
                            f"claim={claim_slot} evidence={evidence_slot} id={normalized_id}"
                        )
                    if (
                        action != "retire"
                        and
                        not _is_current_durable(leaf)
                        and (claim_slot, claim_text, normalized_id)
                        not in prior_support_bindings
                    ):
                        raise PatchValidationError(
                            "non-current Fast evidence cannot become a new claim support: "
                            + normalized_id
                        )
                support_ids.add(normalized_id)
                claim_support.add(normalized_id)
                citation_bindings.setdefault(normalized_id, []).append(
                    (
                        "support",
                        claim_slot,
                        claim_text,
                        f"operations[{operation_index}].claims[{claim_index}].support",
                    )
                )
            support_text_groups = _support_text_groups(
                claim_support, evidence_by_id
            )
            unchanged_prior_support = bool(claim_support) and all(
                (claim_slot, claim_text, evidence_id) in prior_support_bindings
                for evidence_id in claim_support
            )
            if (
                len(support_text_groups) > 1
                and not complementary_support_bundle
                and not unchanged_prior_support
            ):
                raise PatchValidationError(
                    "distinct supplied Fast evidence texts cannot share one claim; "
                    "split these support IDs into separate claims within the same capsule "
                    "(multiple support IDs are allowed only for identical normalized evidence "
                    "text): "
                    + _json(
                        sorted(
                            evidence_id
                            for evidence_group in support_text_groups.values()
                            for evidence_id in evidence_group
                        )
                    )
                )
            for evidence_id in claim.get("counterevidence") or []:
                normalized_id = _required_text(
                    evidence_id, "claim counterevidence ID"
                )
                leaf = evidence_by_id.get(normalized_id)
                if leaf is None:
                    if (
                        claim_slot,
                        claim_text,
                        normalized_id,
                    ) not in prior_counter_bindings:
                        raise PatchValidationError(
                            "claim counterevidence is absent from supplied evidence: "
                            + normalized_id
                        )
                elif _normal_text(_leaf_text(leaf)) == claim_text:
                    raise PatchValidationError(
                        "claim text cannot be identical to its counterevidence: "
                        + normalized_id
                    )
                if (
                    route == "flash"
                    and normalized_id not in prior_counterevidence
                    and (leaf is None or not _is_counterevidence(leaf))
                ):
                    raise PatchValidationError(
                        "Flash cannot create new counterevidence; explicit Pro escalation is "
                        "required: "
                        + normalized_id
                    )
                claim_counter.add(normalized_id)
                citation_bindings.setdefault(normalized_id, []).append(
                    (
                        "counterevidence",
                        claim_slot,
                        claim_text,
                        f"operations[{operation_index}].claims[{claim_index}].counterevidence",
                    )
                )
            if claim_support & claim_counter:
                raise PatchValidationError(
                    "one claim cannot use the same evidence as support and counterevidence: "
                    + _json(sorted(claim_support & claim_counter))
                )
            stale_prior_support = {
                evidence_id
                for evidence_id in claim_support
                if evidence_id in evidence_by_id
                and not _is_current_durable(evidence_by_id[evidence_id])
            }
            current_replacements = {
                evidence_id
                for evidence_id, candidate in evidence_by_id.items()
                if _leaf_slot(candidate) == claim_slot
                and _is_current_durable(candidate)
            }
            if (
                action != "retire"
                and
                stale_prior_support
                and current_replacements
                and not current_replacements.intersection(claim_counter)
            ):
                raise PatchValidationError(
                    "historical non-current support requires current replacement "
                    "counterevidence: "
                    + _json(sorted(stale_prior_support))
                )
            if action == "create" and claim_counter:
                raise PatchValidationError(
                    "create cannot commit unresolved counterevidence as an active capsule: "
                    + _json(sorted(claim_counter))
                )
            if not claim_support and not claim_counter:
                raise PatchValidationError(
                    f"operations[{operation_index}].claims[{claim_index}] has no evidence"
                )
            claim_roles.append((claim_support, claim_counter))
    _validate_repeated_evidence_bindings(evidence_by_id, citation_bindings)
    for left_index, (left_support, left_counter) in enumerate(claim_roles):
        for right_support, right_counter in claim_roles[left_index + 1 :]:
            if left_support & right_counter and right_support & left_counter:
                raise PatchValidationError(
                    "reciprocal counterevidence claims are forbidden; emit one adjudicated "
                    "or challenged claim"
                )
    if route == "flash":
        _, required_ids = _required_promotion_ids(region, capsules)
        required_support = {
            evidence_id
            for evidence_id in required_ids
            if not _is_counterevidence(evidence_by_id[evidence_id])
        }
        missing_support = required_support - support_ids
        if missing_support:
            raise PatchValidationError(
                "Flash must represent every non-conflict durable delta as canonical-slot "
                "support: "
                + _json(sorted(missing_support))
            )


def _claim_evidence_ids(claims: Any) -> set[str]:
    cited: set[str] = set()
    if not isinstance(claims, list):
        return cited
    for claim in claims:
        if not isinstance(claim, Mapping):
            continue
        for field in ("support", "counterevidence"):
            values = claim.get(field)
            if not isinstance(values, list):
                continue
            cited.update(_clean(item) for item in values if _clean(item))
    return cited


def _capsule_evidence_ids(capsules: list[Mapping[str, Any]]) -> set[str]:
    cited: set[str] = set()
    for capsule in capsules:
        if _clean(capsule.get("status")).casefold() not in {"active", "challenged"}:
            continue
        cited.update(_claim_evidence_ids(capsule.get("claims")))
    return cited


def _patch_claim_projection(claim: Mapping[str, Any]) -> dict[str, Any]:
    support = claim.get("support")
    counter = claim.get("counterevidence")
    if not isinstance(support, list) or not isinstance(counter, list):
        raise PatchValidationError("capsule claim evidence must be arrays")
    return {
        "canonical_slot": _required_text(
            claim.get("canonical_slot"), "capsule claim canonical slot"
        ),
        "text": " ".join(
            _required_text(claim.get("text"), "capsule claim text").split()
        ),
        "support": sorted(
            {_required_text(item, "claim support evidence ID") for item in support}
        ),
        "counterevidence": sorted(
            {
                _required_text(item, "claim counterevidence ID")
                for item in counter
            }
        ),
    }


def _validate_flash_delta_patch(
    patch: Mapping[str, Any],
    capsules: list[Mapping[str, Any]],
    required_evidence_ids: set[str],
) -> None:
    """Keep additive Flash operations delta-only while allowing new topics."""
    operations = patch.get("operations")
    if not isinstance(operations, list) or not operations:
        raise PatchValidationError("Flash delta patch requires operations")
    existing = {
        _required_text(capsule.get("capsule_id"), "existing capsule_id"): capsule
        for capsule in capsules
    }
    cited: set[str] = set()
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise PatchValidationError("Flash delta operation must be an object")
        action = _clean(operation.get("action"))
        if action not in ({"revise", "create"} if capsules else {"create"}):
            raise PatchValidationError(
                "Flash delta operations must revise an existing capsule or create a new topic"
            )
        if action == "revise":
            capsule_id = _required_text(operation.get("capsule_id"), "capsule_id")
            capsule = existing.get(capsule_id)
            if capsule is None:
                raise PatchValidationError("Flash delta targeted an unknown capsule")
            if operation.get("base_revision") != capsule.get("revision"):
                raise PatchValidationError("Flash delta base_revision changed")
        claims = operation.get("claims")
        if not isinstance(claims, list) or not claims:
            raise PatchValidationError("Flash delta patch requires non-empty delta claims")
        cited.update(_claim_evidence_ids(claims))
        for claim in claims:
            if not isinstance(claim, Mapping):
                raise PatchValidationError("Flash delta claim must be an object")
            if not _claim_evidence_ids([claim]):
                raise PatchValidationError(
                    "every Flash delta claim must cite required delta evidence"
                )
    outside_delta = cited - required_evidence_ids
    if outside_delta:
        raise PatchValidationError(
            "Flash delta patch cited existing or non-delta evidence: "
            + _json(sorted(outside_delta))
        )
    missing = required_evidence_ids - cited
    if missing:
        raise PatchValidationError(
            "Flash delta patch omitted required evidence: " + _json(sorted(missing))
        )


def _merge_flash_delta_patch(
    patch: Mapping[str, Any], capsules: list[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build full next revisions for every Flash revise plus independent creates."""
    capsules_by_id = {
        _required_text(capsule.get("capsule_id"), "existing capsule_id"): capsule
        for capsule in capsules
    }
    full_operations: list[dict[str, Any]] = []
    appended_claims = 0
    merged_claims = 0
    prior_claim_count = 0
    model_delta_claim_count = 0
    result_claim_count = 0
    for operation in patch.get("operations", []):
        if not isinstance(operation, Mapping):
            raise PatchValidationError("Flash delta operation must be an object")
        action = _clean(operation.get("action"))
        raw_delta = operation.get("claims")
        if not isinstance(raw_delta, list):
            raise PatchValidationError("Flash delta merge requires claim arrays")
        model_delta_claim_count += len(raw_delta)
        if action == "create":
            full_operations.append(
                {
                    "action": "create",
                    "capsule_key": _normalize_capsule_key(
                        operation.get("capsule_key")
                    ),
                    "claims": _canonical_patch_claims(raw_delta),
                }
            )
            appended_claims += len(raw_delta)
            continue
        capsule_id = _required_text(operation.get("capsule_id"), "capsule_id")
        capsule = capsules_by_id.get(capsule_id)
        if capsule is None:
            raise PatchValidationError("Flash delta merge targeted an unknown capsule")
        expected_revision = capsule.get("revision")
        if operation.get("base_revision") != expected_revision:
            raise PatchValidationError("Flash delta base_revision changed")
        raw_prior = capsule.get("claims")
        if not isinstance(raw_prior, list):
            raise PatchValidationError("existing capsule claim array is missing")
        prior_claim_count += len(raw_prior)
        merged: list[dict[str, Any]] = []
        index_by_identity: dict[tuple[str, str], int] = {}
        for source_claims, is_delta in ((raw_prior, False), (raw_delta, True)):
            for raw_claim in source_claims:
                if not isinstance(raw_claim, Mapping):
                    raise PatchValidationError("Flash delta claim must be an object")
                claim = _patch_claim_projection(raw_claim)
                identity = (claim["canonical_slot"], _normal_text(claim["text"]))
                if identity in index_by_identity:
                    target = merged[index_by_identity[identity]]
                    target["support"] = sorted(set(target["support"] + claim["support"]))
                    target["counterevidence"] = sorted(
                        set(target["counterevidence"] + claim["counterevidence"])
                    )
                    if is_delta:
                        merged_claims += 1
                    continue
                index_by_identity[identity] = len(merged)
                merged.append(claim)
                if is_delta:
                    appended_claims += 1
        result_claim_count += len(merged)
        full_operations.append(
            {
                "action": "revise",
                "capsule_id": capsule_id,
                "base_revision": expected_revision,
                "claims": merged,
            }
        )
    full_patch = _materialize_lossless_summaries({"operations": full_operations})
    return full_patch, {
        "schema_version": "tmcra.v4.slow-flash-delta-merge.2",
        "operation_count": len(full_operations),
        "prior_claim_count": prior_claim_count,
        "model_delta_claim_count": model_delta_claim_count,
        "appended_claim_count": appended_claims,
        "merged_claim_count": merged_claims,
        "result_claim_count": result_claim_count,
        "model_delta_patch_sha256": _digest(patch),
        "committed_patch_sha256": _digest(full_patch),
    }


def _sanitize_capsules_for_current_support(
    capsules: list[Mapping[str, Any]],
    current_support_ids: set[str],
    challenged_support_ids: set[str],
    known_evidence_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove claims whose positive support is no longer current Fast evidence."""
    sanitized: list[dict[str, Any]] = []
    removed_support_ids: set[str] = set()
    removed_claim_count = 0
    changed_claim_count = 0
    for capsule in capsules:
        if _clean(capsule.get("status")).casefold() not in {"active", "challenged"}:
            sanitized.append(dict(capsule))
            continue
        raw_claims = capsule.get("claims")
        if not isinstance(raw_claims, list):
            raise EvidencePolicyError("capsule claims are not auditable")
        next_claims: list[dict[str, Any]] = []
        for raw_claim in raw_claims:
            if not isinstance(raw_claim, Mapping):
                raise EvidencePolicyError("capsule claim is not an object")
            claim = dict(raw_claim)
            support = claim.get("support")
            if not isinstance(support, list):
                raise EvidencePolicyError("capsule claim support is not a list")
            normalized_support = [
                _required_text(item, "capsule claim support evidence ID")
                for item in support
            ]
            counterevidence = claim.get("counterevidence")
            if not isinstance(counterevidence, list):
                raise EvidencePolicyError(
                    "capsule claim counterevidence is not a list"
                )
            has_current_counterevidence = bool(
                current_support_ids.intersection(
                    _required_text(item, "capsule claim counterevidence ID")
                    for item in counterevidence
                )
            )
            preserve_adjudicated_history = (
                _clean(capsule.get("status")).casefold() == "challenged"
                and has_current_counterevidence
            )
            current_support = [
                item
                for item in normalized_support
                if item not in known_evidence_ids
                or item in current_support_ids
                or (
                    preserve_adjudicated_history
                    and item in challenged_support_ids
                )
            ]
            removed_support_ids.update(set(normalized_support) - set(current_support))
            if not current_support:
                removed_claim_count += 1
                continue
            if current_support != normalized_support:
                changed_claim_count += 1
            claim["support"] = current_support
            next_claims.append(claim)
        next_capsule = dict(capsule)
        next_capsule["claims"] = next_claims
        sanitized.append(next_capsule)
    return sanitized, {
        "schema_version": "tmcra.v4.slow-current-support-cleanup.1",
        "changed": bool(removed_support_ids or removed_claim_count),
        "removed_support_ids": sorted(removed_support_ids),
        "removed_claim_count": removed_claim_count,
        "changed_claim_count": changed_claim_count,
        "remaining_claim_count": sum(
            len(capsule.get("claims") or []) for capsule in sanitized
        ),
    }


def _deterministic_support_cleanup_patch(
    original_capsules: list[Mapping[str, Any]],
    sanitized_capsules: list[Mapping[str, Any]],
) -> dict[str, Any]:
    original_by_id = {
        _required_text(capsule.get("capsule_id"), "capsule_id"): capsule
        for capsule in original_capsules
    }
    sanitized_by_id = {
        _required_text(capsule.get("capsule_id"), "capsule_id"): capsule
        for capsule in sanitized_capsules
    }
    if set(original_by_id) != set(sanitized_by_id):
        raise PatchValidationError("support cleanup changed capsule identity")
    operations: list[dict[str, Any]] = []
    for capsule_id in sorted(original_by_id):
        original = original_by_id[capsule_id]
        sanitized = sanitized_by_id[capsule_id]
        revision = original.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise PatchValidationError("capsule revision must be positive")
        original_claims = _canonical_patch_claims(original.get("claims"))
        raw_sanitized_claims = sanitized.get("claims")
        if not isinstance(raw_sanitized_claims, list):
            raise PatchValidationError("sanitized capsule claims are missing")
        sanitized_claims = (
            _canonical_patch_claims(raw_sanitized_claims)
            if raw_sanitized_claims
            else []
        )
        if sanitized_claims == original_claims:
            continue
        operations.append(
            {
                "action": "revise" if sanitized_claims else "retire",
                "capsule_id": capsule_id,
                "base_revision": revision,
                "claims": sanitized_claims or original_claims,
            }
        )
    if not operations:
        raise PatchValidationError("support cleanup produced no changed capsule")
    return _materialize_lossless_summaries({"operations": operations})


def _deterministic_summary_migration_patch(
    capsules: list[Mapping[str, Any]],
) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    for capsule in sorted(capsules, key=lambda item: _clean(item.get("capsule_id"))):
        if _clean(capsule.get("status")).casefold() not in {"active", "challenged"}:
            continue
        if not _capsule_requires_summary_migration([capsule]):
            continue
        capsule_id = _required_text(capsule.get("capsule_id"), "capsule_id")
        revision = capsule.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise PatchValidationError("capsule revision must be positive")
        operations.append(
            {
                "action": "revise",
                "capsule_id": capsule_id,
                "base_revision": revision,
                "claims": _canonical_patch_claims(capsule.get("claims")),
            }
        )
    if not operations:
        raise PatchValidationError("summary migration found no invalid capsule")
    patch = _materialize_lossless_summaries({"operations": operations})
    _validate_patch_summary_contract(patch)
    return patch


def _deterministic_contract_migration_patch(
    capsules: list[Mapping[str, Any]], partition_targets: set[str]
) -> dict[str, Any]:
    """Stamp unambiguous single-claim partitions and repair summaries together."""
    operations: list[dict[str, Any]] = []
    operated_targets: set[str] = set()
    for capsule in sorted(capsules, key=lambda item: _clean(item.get("capsule_id"))):
        if _clean(capsule.get("status")).casefold() not in {"active", "challenged"}:
            continue
        capsule_id = _required_text(capsule.get("capsule_id"), "capsule_id")
        targeted = capsule_id in partition_targets
        if not targeted and not _capsule_requires_summary_migration([capsule]):
            continue
        claims = _canonical_patch_claims(capsule.get("claims"))
        if targeted and len(claims) != 1:
            raise PatchValidationError(
                "deterministic partition migration requires exactly one claim per target"
            )
        revision = capsule.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise PatchValidationError("capsule revision must be positive")
        operations.append(
            {
                "action": "revise",
                "capsule_id": capsule_id,
                "base_revision": revision,
                "claims": claims,
            }
        )
        if targeted:
            operated_targets.add(capsule_id)
    if operated_targets != partition_targets:
        raise PatchValidationError(
            "deterministic partition migration target set is incomplete"
        )
    if not operations:
        raise PatchValidationError("contract migration found no capsules to revise")
    patch = _materialize_lossless_summaries({"operations": operations})
    _validate_patch_summary_contract(patch)
    return patch


def _capsule_requires_summary_migration(capsules: list[Mapping[str, Any]]) -> bool:
    for capsule in capsules:
        if _clean(capsule.get("status")).casefold() not in {"active", "challenged"}:
            continue
        try:
            claims = _canonical_patch_claims(capsule.get("claims"))
            summary = validate_semantic_summary(
                capsule.get("value"),
                claims,
                label="stored Slow capsule summary",
            )
            if (
                summary != _semantic_summary_projection(claims)
                or capsule.get("summary_contract_version")
                != SLOW_SUMMARY_CONTRACT_VERSION
            ):
                return True
        except PatchValidationError:
            return True
    return False


def _required_promotion_ids(
    region: Mapping[str, Any], capsules: list[Mapping[str, Any]]
) -> tuple[set[str], set[str]]:
    evidence = [
        item for item in region.get("evidence", []) if isinstance(item, Mapping)
    ]
    eligible = {_leaf_id(item) for item in evidence if _is_current_durable(item)}
    return eligible, eligible - _capsule_evidence_ids(capsules)


def _next_active_capsule_claims(
    capsules: list[Mapping[str, Any]], patch: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Project the full active Slow state after one atomic region patch."""
    next_claims: dict[str, list[dict[str, Any]]] = {}
    known_capsules: set[str] = set()
    for capsule in capsules:
        capsule_id = _required_text(capsule.get("capsule_id"), "capsule_id")
        if capsule_id in known_capsules:
            raise PatchValidationError(
                f"duplicate supplied capsule identity: {capsule_id}"
            )
        known_capsules.add(capsule_id)
        if _clean(capsule.get("status")).casefold() in {"active", "challenged"}:
            raw_claims = capsule.get("claims")
            if not isinstance(raw_claims, list):
                raise PatchValidationError("supplied capsule claims must be a list")
            next_claims[capsule_id] = (
                _canonical_patch_claims(raw_claims) if raw_claims else []
            )

    for operation in patch.get("operations", []):
        if not isinstance(operation, Mapping):
            raise PatchValidationError("promotion patch operation must be an object")
        action = _clean(operation.get("action"))
        if action == "noop":
            continue
        if action == "create":
            capsule_key = _normalize_capsule_key(operation.get("capsule_key"))
            target = "create:" + capsule_key
            if target in next_claims:
                raise PatchValidationError(
                    f"duplicate resulting capsule target: {capsule_key}"
                )
            next_claims[target] = _canonical_patch_claims(operation.get("claims"))
            continue
        capsule_id = _required_text(operation.get("capsule_id"), "capsule_id")
        if capsule_id not in known_capsules:
            raise PatchValidationError(
                f"GraphPatch targeted an unknown supplied capsule: {capsule_id}"
            )
        if action == "retire":
            next_claims.pop(capsule_id, None)
        else:
            next_claims[capsule_id] = _canonical_patch_claims(
                operation.get("claims")
            )
    empty_active = sorted(
        capsule_id for capsule_id, claims in next_claims.items() if not claims
    )
    if empty_active:
        raise PatchValidationError(
            "active Slow capsules cannot have zero claims: " + _json(empty_active)
        )
    return next_claims


def _validate_promotion_patch(
    region: Mapping[str, Any],
    capsules: list[Mapping[str, Any]],
    patch: Mapping[str, Any],
    *,
    required_evidence_ids: set[str] | None = None,
) -> None:
    """Require the committed next Slow revision to cover every current durable leaf."""
    _validate_claim_evidence_contract(region, capsules, patch)
    eligible_ids, required_ids = _required_promotion_ids(region, capsules)
    operations = patch.get("operations")
    if not isinstance(operations, list) or not operations:
        raise PatchValidationError("promotion patch must contain operations")
    if (
        required_ids
        and len(operations) == 1
        and isinstance(operations[0], Mapping)
        and _clean(operations[0].get("action")) == "noop"
    ):
        raise PatchValidationError(
            "noop cannot consume uncited current durable Fast evidence: "
            + _json(sorted(required_ids))
        )

    next_capsules = _next_active_capsule_claims(capsules, patch)
    citation_locations: dict[str, list[str]] = {}
    citation_bindings: dict[str, list[tuple[str, str, str, str]]] = {}
    claim_identity_capsules: dict[tuple[str, str], set[str]] = {}
    for capsule_id, claims in next_capsules.items():
        for claim_index, claim in enumerate(claims):
            slot = claim["canonical_slot"]
            claim_text = _normal_text(claim["text"])
            claim_identity = (slot, claim_text)
            claim_identity_capsules.setdefault(claim_identity, set()).add(capsule_id)
            for role in ("support", "counterevidence"):
                for evidence_id in claim[role]:
                    location = f"{capsule_id}:{claim_index}:{role}"
                    citation_locations.setdefault(evidence_id, []).append(location)
                    citation_bindings.setdefault(evidence_id, []).append(
                        (role, slot, claim_text, location)
                    )

    evidence_by_id = {
        _leaf_id(item): item
        for item in region.get("evidence", [])
        if isinstance(item, Mapping)
    }
    _validate_repeated_evidence_bindings(evidence_by_id, citation_bindings)
    duplicated_claim_identities = {
        f"{slot}\u241f{text}": sorted(capsule_ids)
        for (slot, text), capsule_ids in claim_identity_capsules.items()
        if len(capsule_ids) > 1
    }
    if duplicated_claim_identities:
        raise PatchValidationError(
            "one self-contained semantic claim cannot span multiple active Slow capsules: "
            + _json(duplicated_claim_identities)
        )

    cited_ids = set(citation_locations)
    missing = eligible_ids - cited_ids
    if missing:
        raise PatchValidationError(
            "next Slow revision omits current durable Fast evidence: "
            + _json(sorted(missing))
        )
    required = set(required_evidence_ids or ())
    missing_required = required - cited_ids
    if missing_required:
        raise PatchValidationError(
            "next Slow revision omits required Fast evidence: "
            + _json(sorted(missing_required))
        )


def _public_leaf(leaf: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _leaf_metadata(leaf)
    result = {
        "memory_id": _leaf_id(leaf),
        "value": _leaf_text(leaf),
        "turn_index": leaf.get("turn_index"),
        "record_state": leaf.get("record_state"),
        "canonical_slot": _leaf_slot(leaf),
        "durability": metadata.get("durability"),
        "temporal_status": metadata.get("temporal_status") or metadata.get("target_status"),
        "polarity": metadata.get("polarity"),
        "write_operation": metadata.get("write_operation"),
        "evidence_role": (
            "counterevidence" if _is_counterevidence(leaf) else "support"
        ),
    }
    _assert_no_benchmark_fields(result)
    return result


def _public_capsule(capsule: Mapping[str, Any]) -> dict[str, Any]:
    claims = capsule.get("claims")
    if not isinstance(claims, list):
        raise EvidencePolicyError("Slow capsule claims are not public-request ready")
    result = {
        "capsule_id": _required_text(capsule.get("capsule_id"), "capsule_id"),
        "revision": capsule.get("revision"),
        "status": _required_text(capsule.get("status") or "active", "capsule status"),
        "summary": _clean(capsule.get("value")),
        "claims": _canonical_patch_claims(claims) if claims else [],
    }
    capsule_key = _clean(capsule.get("capsule_key"))
    if capsule_key:
        result["capsule_key"] = _normalize_capsule_key(capsule_key)
    if capsule.get("partition_contract_version"):
        result["partition_contract_version"] = capsule.get(
            "partition_contract_version"
        )
    _assert_no_benchmark_fields(result)
    return result


class V4SlowGraphStore(_v3.SlowGraphStore):
    """V3 ledger with atomic multi-capsule region commits."""

    def _init_schema(self) -> None:
        super()._init_schema()
        with self.connection() as con:
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "idx_v4_slow_patch_one_per_job ON slow_graph_patches(job_id)"
            )
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "idx_v4_slow_patch_operation_ordinal "
                "ON slow_graph_patch_operations(patch_id,ordinal)"
            )
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_v4_slow_provenance_fact "
                "ON slow_graph_provenance("
                "patch_id,scope_id,capsule_id,revision,evidence_memory_id,claim_id,polarity)"
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS slow_graph_process_loss_recoveries(
                    recovery_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL UNIQUE,
                    scope_id TEXT NOT NULL,
                    recovery_source TEXT NOT NULL,
                    claim_token TEXT NOT NULL,
                    claim_owner TEXT NOT NULL,
                    lease_expires_at INTEGER,
                    attempt_created_at INTEGER NOT NULL,
                    job_attempts_before INTEGER NOT NULL,
                    attempt_metadata_sha256 TEXT NOT NULL,
                    interruption_error_sha256 TEXT NOT NULL,
                    external_call_outcome TEXT NOT NULL,
                    potential_duplicate_physical_calls_min INTEGER NOT NULL,
                    potential_duplicate_physical_calls_max INTEGER NOT NULL,
                    recovered_at INTEGER NOT NULL
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_slow_process_loss_job "
                "ON slow_graph_process_loss_recoveries(job_id,recovered_at)"
            )

    @staticmethod
    def _claim_owner_pid(owner: Any) -> int:
        parts = _clean(owner).split(":", 2)
        if len(parts) != 3 or parts[0] != "pid" or not parts[1].isdigit():
            raise SlowGraphError("interrupted Slow claim owner is invalid")
        pid = int(parts[1])
        if pid <= 0:
            raise SlowGraphError("interrupted Slow claim owner PID is invalid")
        return pid

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError as exc:
            if exc.errno in {errno.ESRCH, errno.EINVAL}:
                return False
            if exc.errno == errno.EPERM:
                return True
            raise
        return True

    def interrupted_process_loss_attempts(self) -> list[dict[str, Any]]:
        """List unjournaled attempts whose external outcome needs explicit review."""
        with self.connection() as con:
            rows = con.execute(
                "SELECT j.job_id,j.scope_id,j.region_key,j.status,j.attempts,"
                "j.last_error,j.claim_token AS job_claim_token,"
                "j.claim_owner AS job_claim_owner,j.lease_expires_at,"
                "a.attempt_id,a.status AS attempt_status,a.call_metadata_json,"
                "a.error AS attempt_error,a.created_at AS attempt_created_at,"
                "a.completed_at,a.claim_token AS attempt_claim_token,"
                "a.claim_owner AS attempt_claim_owner "
                "FROM slow_graph_jobs j JOIN slow_graph_attempts a "
                "ON a.job_id=j.job_id "
                "LEFT JOIN slow_graph_process_loss_recoveries r "
                "ON r.attempt_id=a.attempt_id "
                "WHERE r.attempt_id IS NULL AND ("
                "(j.status='pending' AND j.claim_token IS NOT NULL "
                "AND a.status='started' AND a.claim_token=j.claim_token "
                "AND a.claim_owner=j.claim_owner) OR "
                "(j.status='failed' AND j.claim_token IS NULL "
                "AND j.last_error=? AND a.status='expired' AND a.error=?)) "
                "ORDER BY a.created_at,a.attempt_id",
                (PROCESS_LOSS_INTERRUPTION_ERROR, PROCESS_LOSS_INTERRUPTION_ERROR),
            ).fetchall()
        return [dict(row) for row in rows]

    def recover_interrupted_process_loss(
        self, job_id: str, *, expected_attempt_id: str | None = None
    ) -> dict[str, Any]:
        """Atomically journal and reopen one reviewed process-loss attempt."""
        now = _v3._now()
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            job = con.execute(
                "SELECT * FROM slow_graph_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise SlowGraphError("unknown interrupted Slow job")
            patch_count = int(
                con.execute(
                    "SELECT count(*) FROM slow_graph_patches WHERE job_id=?",
                    (job_id,),
                ).fetchone()[0]
            )
            if patch_count:
                raise SlowGraphError("interrupted Slow job already has a patch")

            recovery_source: str
            lease_expires_at: int | None
            if job["status"] == "pending" and job["claim_token"] is not None:
                lease = job["lease_expires_at"]
                if lease is None or int(lease) >= now:
                    raise SlowGraphError("interrupted Slow claim has not expired")
                attempts = con.execute(
                    "SELECT * FROM slow_graph_attempts WHERE job_id=? "
                    "AND status='started' AND claim_token=? AND claim_owner=?",
                    (job_id, job["claim_token"], job["claim_owner"]),
                ).fetchall()
                recovery_source = "expired_claimed_started"
                lease_expires_at = int(lease)
                job_attempts_before = int(job["attempts"] or 0)
            elif (
                job["status"] == "failed"
                and job["claim_token"] is None
                and _clean(job["last_error"]) == PROCESS_LOSS_INTERRUPTION_ERROR
            ):
                attempts = con.execute(
                    "SELECT a.* FROM slow_graph_attempts a "
                    "LEFT JOIN slow_graph_process_loss_recoveries r "
                    "ON r.attempt_id=a.attempt_id "
                    "WHERE a.job_id=? AND a.status='expired' AND a.error=? "
                    "AND r.attempt_id IS NULL ORDER BY a.created_at,a.attempt_id",
                    (job_id, PROCESS_LOSS_INTERRUPTION_ERROR),
                ).fetchall()
                recovery_source = "legacy_expired_failed"
                lease_expires_at = None
                job_attempts_before = int(job["attempts"] or 0) - 1
            else:
                raise SlowGraphError(
                    "process-loss recovery requires one claimed started or legacy expired job"
                )
            if len(attempts) != 1:
                raise SlowGraphError(
                    "interrupted Slow job does not have exactly one recoverable attempt"
                )
            attempt = attempts[0]
            if (
                expected_attempt_id is not None
                and str(attempt["attempt_id"]) != expected_attempt_id
            ):
                raise SlowGraphError("interrupted Slow attempt changed after review")
            raw_metadata = _required_text(
                attempt["call_metadata_json"], "interrupted call metadata"
            )
            metadata = _v3._strict_json(
                raw_metadata, label="interrupted call metadata", expected=dict
            )
            if metadata or (
                recovery_source == "expired_claimed_started"
                and (
                    _clean(attempt["error"])
                    or attempt["completed_at"] is not None
                )
            ):
                raise SlowGraphError(
                    "interrupted Slow attempt already contains a durable outcome"
                )
            if (
                recovery_source == "legacy_expired_failed"
                and (
                    _clean(attempt["error"]) != PROCESS_LOSS_INTERRUPTION_ERROR
                    or attempt["completed_at"] is None
                )
            ):
                raise SlowGraphError("legacy expired Slow attempt is inconsistent")
            claim_token = _required_text(
                attempt["claim_token"], "interrupted claim token"
            )
            claim_owner = _required_text(
                attempt["claim_owner"], "interrupted claim owner"
            )
            owner_pid = self._claim_owner_pid(claim_owner)
            if self._pid_is_alive(owner_pid):
                raise SlowGraphError(
                    f"interrupted Slow claim owner is still alive: {owner_pid}"
                )
            metadata_sha256 = hashlib.sha256(
                raw_metadata.encode("utf-8")
            ).hexdigest()
            error_sha256 = hashlib.sha256(
                PROCESS_LOSS_INTERRUPTION_ERROR.encode("utf-8")
            ).hexdigest()
            recovery_id = "sgr_" + _digest(
                {
                    "job_id": job_id,
                    "attempt_id": attempt["attempt_id"],
                    "claim_token": claim_token,
                    "claim_owner": claim_owner,
                    "attempt_metadata_sha256": metadata_sha256,
                }
            )[:32]
            con.execute(
                "INSERT INTO slow_graph_process_loss_recoveries VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    recovery_id,
                    job_id,
                    attempt["attempt_id"],
                    job["scope_id"],
                    recovery_source,
                    claim_token,
                    claim_owner,
                    lease_expires_at,
                    int(attempt["created_at"]),
                    job_attempts_before,
                    metadata_sha256,
                    error_sha256,
                    "uncertain",
                    0,
                    SLOW_PROCESS_LOSS_PHYSICAL_CALLS_MAX,
                    now,
                ),
            )
            if recovery_source == "expired_claimed_started":
                expired = con.execute(
                    "UPDATE slow_graph_attempts SET status='expired',error=?,"
                    "completed_at=? WHERE attempt_id=? AND job_id=? "
                    "AND status='started' AND claim_token=? AND claim_owner=?",
                    (
                        PROCESS_LOSS_INTERRUPTION_ERROR,
                        now,
                        attempt["attempt_id"],
                        job_id,
                        claim_token,
                        claim_owner,
                    ),
                )
                if expired.rowcount != 1:
                    raise SlowGraphError(
                        "interrupted Slow attempt changed during recovery"
                    )
                reopened = con.execute(
                    "UPDATE slow_graph_jobs SET status='pending',attempts=attempts+1,"
                    "last_error='',updated_at=?,claim_token=NULL,claim_owner=NULL,"
                    "lease_expires_at=NULL WHERE job_id=? AND status='pending' "
                    "AND claim_token=? AND claim_owner=? AND lease_expires_at<?",
                    (now, job_id, claim_token, claim_owner, now),
                )
            else:
                reopened = con.execute(
                    "UPDATE slow_graph_jobs SET status='pending',last_error='',"
                    "updated_at=? WHERE job_id=? AND status='failed' "
                    "AND claim_token IS NULL AND last_error=?",
                    (now, job_id, PROCESS_LOSS_INTERRUPTION_ERROR),
                )
            if reopened.rowcount != 1:
                raise SlowGraphError("interrupted Slow job changed during recovery")
        return {
            "schema_version": SLOW_PROCESS_LOSS_RECOVERY_VERSION,
            "recovery_id": recovery_id,
            "job_id": job_id,
            "attempt_id": str(attempt["attempt_id"]),
            "scope_id": str(job["scope_id"]),
            "recovery_source": recovery_source,
            "external_call_outcome": "uncertain",
            "potential_duplicate_physical_calls_min": 0,
            "potential_duplicate_physical_calls_max": (
                SLOW_PROCESS_LOSS_PHYSICAL_CALLS_MAX
            ),
            "physical_api_calls_during_recovery": 0,
            "status": "pending",
            "recovered_at": now,
        }

    def recover_interrupted_attempts(self) -> int:
        """Fail closed so process loss cannot be silently converted into retry state."""
        now = _v3._now()
        with self.connection() as con:
            rows = con.execute(
                "SELECT job_id FROM slow_graph_jobs WHERE status='pending' "
                "AND claim_token IS NOT NULL AND lease_expires_at<? "
                "ORDER BY created_at,job_id",
                (now,),
            ).fetchall()
        if rows:
            raise SlowGraphError(
                "expired Slow attempts require explicit process-loss journal recovery: "
                + _json([str(row["job_id"]) for row in rows])
            )
        return 0

    def _capsule_id(
        self, scope_id: str, region_key: str, capsule_key: str | None = None
    ) -> str:
        if capsule_key is None:
            return super()._capsule_id(scope_id, region_key)
        return "cap_" + _digest(
            {
                "scope_id": _required_text(scope_id, "scope_id"),
                "region_key": _required_text(region_key, "region_key"),
                "capsule_key": _normalize_capsule_key(capsule_key),
                "partition_contract_version": SLOW_PARTITION_CONTRACT_VERSION,
            }
        )[:24]

    def _capsules(
        self, con: Any, scope_id: str, region_key: str
    ) -> list[dict[str, Any]]:
        rows = con.execute(
            "SELECT memory_id,state,value,metadata_json FROM records WHERE scope_id=?",
            (scope_id,),
        ).fetchall()
        by_capsule: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            metadata = self._metadata(row, "capsule metadata")
            if (
                metadata.get("content_variant") != CAPSULE_VARIANT
                or metadata.get("region_key") != region_key
            ):
                continue
            capsule_id = _required_text(metadata.get("capsule_id"), "capsule_id")
            revision = metadata.get("revision")
            if (
                isinstance(revision, bool)
                or not isinstance(revision, int)
                or revision < 1
            ):
                raise AuditError("capsule revision is invalid")
            by_capsule.setdefault(capsule_id, []).append(
                {
                    "memory_id": row["memory_id"],
                    "record_state": row["state"],
                    "value": row["value"],
                    **metadata,
                }
            )
        latest: list[dict[str, Any]] = []
        for capsule_id, revisions in sorted(by_capsule.items()):
            latest_revision = max(int(item["revision"]) for item in revisions)
            candidates = [
                item for item in revisions if int(item["revision"]) == latest_revision
            ]
            if len(candidates) != 1:
                raise AuditError(
                    f"capsule {capsule_id} lacks one latest revision"
                )
            latest.append(candidates[0])
        return latest

    def _job_metadata(
        self,
        con: Any,
        scope_id: str,
        region_key: str,
        evidence_ids: list[str],
        manager: Any | None,
    ) -> dict[str, Any]:
        metadata = dict(
            super()._job_metadata(
                con, scope_id, region_key, evidence_ids, manager
            )
        )
        # V3 intentionally excludes capsule_revision_hash from idempotency. V4.7
        # needs a changed Slow head to produce a fresh job after a stale claim.
        metadata["capsule_state_idempotency_hash"] = metadata[
            "capsule_revision_hash"
        ]
        return metadata

    def _assert_job_snapshot(
        self, con: Any, job: Any
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        metadata = self._metadata(job, "job")
        evidence_ids = _v3._strict_json(
            job["evidence_ids_json"], label="job evidence IDs", expected=list
        )
        region = {
            "region_key": job["region_key"],
            "evidence": self._evidence(con, job["scope_id"], evidence_ids),
        }
        capsules = self._capsules(con, job["scope_id"], job["region_key"])
        if metadata.get("evidence_content_hash") != _digest(region["evidence"]):
            raise StaleRevisionError(
                "Fast evidence changed after the Slow job was enqueued"
            )
        if metadata.get("capsule_revision_hash") != _digest(capsules):
            raise StaleRevisionError(
                "Slow capsules changed after the Slow job was enqueued"
            )
        return region, capsules

    def _claim_context(
        self, claim: JobClaim
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        with self.connection() as con:
            job = con.execute(
                "SELECT * FROM slow_graph_jobs WHERE job_id=?", (claim.job_id,)
            ).fetchone()
            if job is None:
                raise SlowGraphError("unknown slow graph job")
            if (
                job["status"] != "pending"
                or job["claim_token"] != claim.token
                or job["claim_owner"] != claim.owner
                or job["lease_expires_at"] is None
                or int(job["lease_expires_at"]) < _v3._now()
            ):
                raise SlowGraphError("job claim is no longer active")
            return self._assert_job_snapshot(con, job)

    def _annotate_revision(
        self,
        con: Any,
        *,
        scope_id: str,
        record_id: str,
        operation: Mapping[str, Any],
        old_memory_id: str | None,
        call_metadata: Mapping[str, Any],
    ) -> None:
        row = con.execute(
            "SELECT value,metadata_json FROM records "
            "WHERE scope_id=? AND memory_id=?",
            (scope_id, record_id),
        ).fetchone()
        if row is None:
            raise SlowGraphError("new Slow revision is missing before annotation")
        metadata = self._metadata(row, "new Slow revision")
        claims = _canonical_patch_claims(metadata.get("claims"))
        expected_summary = _semantic_summary_projection(claims)
        if row["value"] != expected_summary:
            raise AuditError("stored Slow summary differs from final claims")
        old_metadata: dict[str, Any] = {}
        if old_memory_id:
            old = con.execute(
                "SELECT metadata_json FROM records "
                "WHERE scope_id=? AND memory_id=?",
                (scope_id, old_memory_id),
            ).fetchone()
            if old is None:
                raise SlowGraphError("prior Slow revision disappeared during commit")
            old_metadata = self._metadata(old, "prior Slow revision")
        capsule_key = _clean(operation.get("capsule_key")) or _clean(
            old_metadata.get("capsule_key")
        )
        if not capsule_key:
            capsule_key = "legacy"
        metadata.update(
            {
                "capsule_key": _normalize_capsule_key(capsule_key),
                "summary_contract_version": SLOW_SUMMARY_CONTRACT_VERSION,
                "evidence_binding_contract_version": (
                    SLOW_EVIDENCE_BINDING_CONTRACT_VERSION
                ),
                "summary_projection_sha256": hashlib.sha256(
                    expected_summary.encode("utf-8")
                ).hexdigest(),
            }
        )
        partition_targets = set(
            call_metadata.get("semantic_partition_capsule_ids") or ()
        )
        if (
            operation.get("action") == "create"
            or old_metadata.get("partition_contract_version")
            == SLOW_PARTITION_CONTRACT_VERSION
            or metadata.get("capsule_id") in partition_targets
        ):
            metadata["partition_contract_version"] = (
                SLOW_PARTITION_CONTRACT_VERSION
            )
        con.execute(
            "UPDATE records SET metadata_json=? WHERE scope_id=? AND memory_id=?",
            (_json(metadata), scope_id, record_id),
        )

    def apply_patch(
        self,
        job_id: str,
        patch: Mapping[str, Any],
        *,
        manager_model: str,
        call_metadata: Mapping[str, Any] | None = None,
        claim: JobClaim,
    ) -> str:
        validate_patch(patch, require_lossless_summary=True)
        if claim.job_id != job_id:
            raise SlowGraphError("claim does not belong to job")
        patch_id = "sgp_" + uuid.uuid4().hex
        metadata_for_call = dict(call_metadata or {})
        metadata_for_call.setdefault(
            "evidence_binding_contract_version",
            SLOW_EVIDENCE_BINDING_CONTRACT_VERSION,
        )
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            now = _v3._now()
            job = con.execute(
                "SELECT * FROM slow_graph_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise SlowGraphError("unknown slow graph job")
            if (
                job["status"] != "pending"
                or job["claim_token"] != claim.token
                or job["claim_owner"] != claim.owner
                or job["lease_expires_at"] is None
                or int(job["lease_expires_at"]) < now
            ):
                raise SlowGraphError("job claim is no longer active")
            self._records_table_exists(con)
            job_metadata = self._metadata(job, "job")
            if job_metadata["schema_version"] != SCHEMA_VERSION:
                raise SlowGraphError("job schema metadata drift")
            _, current_capsules = self._assert_job_snapshot(con, job)
            capsules_by_id = {
                _required_text(item.get("capsule_id"), "capsule_id"): item
                for item in current_capsules
            }
            if con.execute(
                "SELECT 1 FROM slow_graph_patches WHERE job_id=?", (job_id,)
            ).fetchone() is not None:
                raise SlowGraphError("slow graph job already has a committed patch")

            planned: list[dict[str, Any]] = []
            planned_targets: set[str] = set()
            for operation in patch["operations"]:
                action = operation["action"]
                if action == "create":
                    capsule_id = self._capsule_id(
                        job["scope_id"],
                        job["region_key"],
                        _normalize_capsule_key(operation.get("capsule_key")),
                    )
                else:
                    capsule_id = _clean(operation.get("capsule_id"))
                    if not capsule_id:
                        capsule_id = "region:" + job["region_key"]
                if capsule_id in planned_targets:
                    raise PatchValidationError(
                        f"duplicate resolved capsule target: {capsule_id}"
                    )
                planned_targets.add(capsule_id)
                head = (
                    None
                    if capsule_id.startswith("region:")
                    else self._head(con, job["scope_id"], capsule_id)
                )
                if action == "create":
                    if head is not None:
                        raise StaleRevisionError("capsule already exists")
                elif action == "noop":
                    if operation.get("capsule_id") and capsule_id not in capsules_by_id:
                        raise StaleRevisionError("noop capsule does not exist in region")
                else:
                    current = capsules_by_id.get(capsule_id)
                    if current is None:
                        raise StaleRevisionError(
                            "capsule is not a current revision in this region"
                        )
                    if (
                        head is None
                        or operation["base_revision"] != head[0]
                        or operation["base_revision"] != current.get("revision")
                    ):
                        raise StaleRevisionError(
                            "base_revision is stale for " + capsule_id
                        )
                planned.append(
                    {
                        "operation": operation,
                        "action": action,
                        "capsule_id": capsule_id,
                        "head": head,
                    }
                )

            con.execute(
                "INSERT INTO slow_graph_patches VALUES(?,?,?,?,?,?,?,?)",
                (
                    patch_id,
                    job_id,
                    job["scope_id"],
                    job["region_key"],
                    manager_model,
                    _json(patch),
                    _json(metadata_for_call),
                    now,
                ),
            )
            for ordinal, item in enumerate(planned):
                operation = item["operation"]
                action = item["action"]
                capsule_id = item["capsule_id"]
                head = item["head"]
                old_memory_id: str | None = None
                if action == "create":
                    base, revision = None, 1
                    record_id = super()._insert_revision(
                        con,
                        job=job,
                        patch_id=patch_id,
                        operation=operation,
                        capsule_id=capsule_id,
                        revision=revision,
                        action=action,
                    )
                    self._annotate_revision(
                        con,
                        scope_id=job["scope_id"],
                        record_id=record_id,
                        operation=operation,
                        old_memory_id=None,
                        call_metadata=metadata_for_call,
                    )
                elif action == "noop":
                    if head is None:
                        record_id, base, revision = "", None, None
                    else:
                        record_id, (revision, _) = head[1], head
                        base = revision
                else:
                    if head is None:
                        raise StaleRevisionError("capsule head disappeared")
                    base, revision = head[0], head[0] + 1
                    old_memory_id = head[1]
                    record_id = super()._insert_revision(
                        con,
                        job=job,
                        patch_id=patch_id,
                        operation=operation,
                        capsule_id=capsule_id,
                        revision=revision,
                        action=action,
                        old_memory_id=old_memory_id,
                    )
                    self._annotate_revision(
                        con,
                        scope_id=job["scope_id"],
                        record_id=record_id,
                        operation=operation,
                        old_memory_id=old_memory_id,
                        call_metadata=metadata_for_call,
                    )
                    if action == "challenge":
                        self._write_edge(
                            con,
                            scope_id=job["scope_id"],
                            source=record_id,
                            target=old_memory_id,
                            edge_type="challenges",
                            patch_id=patch_id,
                            evidence_refs=[],
                            action=action,
                            turn=now,
                        )
                    if action == "retire":
                        self._write_edge(
                            con,
                            scope_id=job["scope_id"],
                            source=record_id,
                            target=old_memory_id,
                            edge_type="invalidates",
                            patch_id=patch_id,
                            evidence_refs=[],
                            action=action,
                            turn=now,
                        )
                con.execute(
                    "INSERT INTO slow_graph_patch_operations VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        "sgo_" + uuid.uuid4().hex,
                        patch_id,
                        ordinal,
                        capsule_id,
                        action,
                        base,
                        revision,
                        _json(operation),
                        now,
                    ),
                )

            completed_attempt = con.execute(
                "UPDATE slow_graph_attempts SET status='completed',"
                "call_metadata_json=?,completed_at=? WHERE attempt_id=? AND job_id=? "
                "AND claim_token=? AND claim_owner=? AND status='started'",
                (
                    _json(metadata_for_call),
                    now,
                    claim.attempt_id,
                    job_id,
                    claim.token,
                    claim.owner,
                ),
            )
            if completed_attempt.rowcount != 1:
                raise SlowGraphError("claimed attempt is no longer active")
            completed_job = con.execute(
                "UPDATE slow_graph_jobs SET status='completed',attempts=attempts+1,"
                "last_error='',updated_at=?,claim_token=NULL,claim_owner=NULL,"
                "lease_expires_at=NULL WHERE job_id=? AND status='pending' "
                "AND claim_token=? AND claim_owner=? AND lease_expires_at>=?",
                (now, job_id, claim.token, claim.owner, now),
            )
            if completed_job.rowcount != 1:
                raise SlowGraphError("job claim expired before completion")
            self._audit_transaction(con, job["scope_id"])
        return patch_id

    def _audit_transaction(self, con: Any, scope_id: str) -> None:
        super()._audit_transaction(con, scope_id)
        record_rows = con.execute(
            "SELECT memory_id,value,metadata_json FROM records WHERE scope_id=?",
            (scope_id,),
        ).fetchall()
        current_patch_ids: set[str] = set()
        result_records: dict[tuple[str, str, int], list[tuple[Any, dict[str, Any]]]] = {}
        for row in record_rows:
            metadata = self._metadata(row, "V4 transaction record")
            if metadata.get("content_variant") != CAPSULE_VARIANT:
                continue
            patch_id = _clean(metadata.get("patch_id"))
            capsule_id = _required_text(metadata.get("capsule_id"), "capsule_id")
            revision = metadata.get("revision")
            if (
                isinstance(revision, bool)
                or not isinstance(revision, int)
                or revision < 1
            ):
                raise AuditError("V4 capsule revision is invalid")
            result_records.setdefault(
                (patch_id, capsule_id, revision), []
            ).append((row, metadata))
            if metadata.get("summary_contract_version") != SLOW_SUMMARY_CONTRACT_VERSION:
                continue
            if (
                metadata.get("evidence_binding_contract_version")
                not in SUPPORTED_SLOW_EVIDENCE_BINDING_CONTRACT_VERSIONS
            ):
                raise AuditError("stored Slow evidence-binding contract has drifted")
            current_patch_ids.add(patch_id)
            claims = _canonical_patch_claims(metadata.get("claims"))
            expected_summary = _semantic_summary_projection(claims)
            if row["value"] != expected_summary:
                raise AuditError(
                    "current V4 Slow summary differs from final claims projection"
                )
            expected_digest = hashlib.sha256(
                expected_summary.encode("utf-8")
            ).hexdigest()
            if metadata.get("summary_projection_sha256") != expected_digest:
                raise AuditError("current V4 Slow summary digest is inconsistent")
            _normalize_capsule_key(
                metadata.get("capsule_key"), "stored capsule_key"
            )

        orphan_count = int(
            con.execute(
                "SELECT count(*) FROM slow_graph_patch_operations o "
                "LEFT JOIN slow_graph_patches p ON p.patch_id=o.patch_id "
                "WHERE p.patch_id IS NULL"
            ).fetchone()[0]
        )
        if orphan_count:
            raise AuditError("orphan Slow patch operations exist")

        patches = con.execute(
            "SELECT * FROM slow_graph_patches WHERE scope_id=?", (scope_id,)
        ).fetchall()
        for patch_row in patches:
            patch_id = str(patch_row["patch_id"])
            call_metadata = _v3._strict_json(
                patch_row["call_metadata_json"],
                label="V4 patch call metadata",
                expected=dict,
            )
            if (
                call_metadata.get("evidence_binding_contract_version")
                not in SUPPORTED_SLOW_EVIDENCE_BINDING_CONTRACT_VERSIONS
            ):
                raise AuditError("Slow patch evidence-binding contract has drifted")
            is_current = (
                patch_id in current_patch_ids
                or call_metadata.get("summary_contract_version")
                == SLOW_SUMMARY_CONTRACT_VERSION
                or call_metadata.get("prompt_version") == SLOW_PROMPT_VERSION
            )
            if not is_current:
                continue
            patch = _v3._strict_json(
                patch_row["patch_json"], label="V4 patch", expected=dict
            )
            validate_patch(patch, require_lossless_summary=True)
            operation_rows = con.execute(
                "SELECT * FROM slow_graph_patch_operations "
                "WHERE patch_id=? ORDER BY ordinal,operation_id",
                (patch_id,),
            ).fetchall()
            operations = patch["operations"]
            if len(operation_rows) != len(operations):
                raise AuditError("V4 patch operation row count is inconsistent")
            if [int(row["ordinal"]) for row in operation_rows] != list(
                range(len(operations))
            ):
                raise AuditError("V4 patch operation ordinals are inconsistent")
            for ordinal, (operation, operation_row) in enumerate(
                zip(operations, operation_rows, strict=True)
            ):
                stored_operation = _v3._strict_json(
                    operation_row["operation_json"],
                    label="V4 patch operation",
                    expected=dict,
                )
                if stored_operation != operation:
                    raise AuditError(
                        f"V4 patch operation {ordinal} differs from patch_json"
                    )
                action = operation["action"]
                if operation_row["action"] != action:
                    raise AuditError("V4 patch operation action is inconsistent")
                capsule_id = _required_text(
                    operation_row["capsule_id"], "operation capsule_id"
                )
                base_revision = operation_row["base_revision"]
                result_revision = operation_row["result_revision"]
                if action == "create":
                    expected_capsule_id = self._capsule_id(
                        scope_id,
                        patch_row["region_key"],
                        operation["capsule_key"],
                    )
                    if (
                        capsule_id != expected_capsule_id
                        or base_revision is not None
                        or result_revision != 1
                    ):
                        raise AuditError("V4 create operation identity is inconsistent")
                elif action == "noop":
                    expected_target = _clean(operation.get("capsule_id"))
                    if expected_target and capsule_id != expected_target:
                        raise AuditError("V4 noop operation identity is inconsistent")
                    if base_revision != result_revision:
                        raise AuditError("V4 noop revision mapping is inconsistent")
                    continue
                else:
                    if capsule_id != operation["capsule_id"]:
                        raise AuditError("V4 operation capsule target is inconsistent")
                    if (
                        base_revision != operation["base_revision"]
                        or result_revision != operation["base_revision"] + 1
                    ):
                        raise AuditError("V4 operation revision mapping is inconsistent")
                matches = result_records.get(
                    (patch_id, capsule_id, int(result_revision)), []
                )
                if len(matches) != 1:
                    raise AuditError(
                        "V4 patch operation lacks one matching result revision"
                    )
                result_metadata = matches[0][1]
                if result_metadata.get("action") != action:
                    raise AuditError("V4 result revision action is inconsistent")

    def fast_regions(self, scope_id: str) -> dict[str, list[dict[str, Any]]]:
        regions = super().fast_regions(scope_id)
        with self.connection() as con:
            state_by_id = {
                str(row["memory_id"]): _clean(row["state"])
                for row in con.execute(
                    "SELECT memory_id,state FROM records WHERE scope_id=?", (scope_id,)
                ).fetchall()
            }
        for leaves in regions.values():
            for leaf in leaves:
                memory_id = _leaf_id(leaf)
                state = _required_text(
                    state_by_id.get(memory_id), f"fast evidence {memory_id} record state"
                )
                leaf["record_state"] = state
                leaf["metadata"] = {**dict(_leaf_metadata(leaf)), "record_state": state}
        return regions

    def promotion_coverage(self, scope_id: str) -> dict[str, Any]:
        regions = self.fast_regions(scope_id)
        leaves_by_id = {
            _leaf_id(leaf): leaf
            for leaves in regions.values()
            for leaf in leaves
        }
        eligible_ids = {
            memory_id
            for memory_id, leaf in leaves_by_id.items()
            if _is_current_durable(leaf)
        }
        challenged_ids = {
            memory_id
            for memory_id, leaf in leaves_by_id.items()
            if _is_challenged_durable(leaf)
        }
        uncertain_ids = {
            memory_id
            for memory_id, leaf in leaves_by_id.items()
            if _is_uncertain(leaf)
        }
        episodic_ids = {
            memory_id
            for memory_id, leaf in leaves_by_id.items()
            if _is_episodic(leaf)
        }
        classified = eligible_ids | challenged_ids | uncertain_ids | episodic_ids
        inactive_ids = set(leaves_by_id) - classified
        cited_ids: set[str] = set()
        active_claim_count = 0
        invalid_capsule_ids: set[str] = set()
        semantic_integrity_issues: list[dict[str, Any]] = []
        evidence_locations: dict[str, list[dict[str, Any]]] = {}
        citation_bindings: dict[str, list[tuple[str, str, str, str]]] = {}
        claim_identity_capsules: dict[tuple[str, str], set[str]] = {}
        capsule_key_owners: dict[tuple[str, str], set[str]] = {}
        with self.connection() as con:
            patch_routes: dict[str, str] = {}
            patch_table = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='slow_graph_patches'"
            ).fetchone()
            if patch_table is not None:
                for patch_row in con.execute(
                    "SELECT patch_id,call_metadata_json FROM slow_graph_patches "
                    "WHERE scope_id=?",
                    (scope_id,),
                ).fetchall():
                    patch_metadata = _v3._strict_json(
                        patch_row["call_metadata_json"],
                        label="slow patch call metadata",
                        expected=dict,
                    )
                    patch_routes[str(patch_row["patch_id"])] = _clean(
                        patch_metadata.get("route")
                    )
            rows = con.execute(
                "SELECT memory_id,state,value,metadata_json FROM records WHERE scope_id=?",
                (scope_id,),
            ).fetchall()
            capsules_by_id: dict[str, list[tuple[int, sqlite3.Row, dict[str, Any]]]] = {}
            for row in rows:
                metadata = self._metadata(row, "record")
                if metadata.get("content_variant") != CAPSULE_VARIANT:
                    continue
                capsule_id = _clean(metadata.get("capsule_id"))
                revision = metadata.get("revision")
                if (
                    not capsule_id
                    or isinstance(revision, bool)
                    or not isinstance(revision, int)
                    or revision < 1
                ):
                    invalid_capsule_ids.add(capsule_id or f"memory:{row['memory_id']}")
                    continue
                capsules_by_id.setdefault(capsule_id, []).append(
                    (revision, row, metadata)
                )

            latest_capsules: list[
                tuple[Any, dict[str, Any], str, set[str]]
            ] = []
            for capsule_id, revisions in capsules_by_id.items():
                latest_revision = max(item[0] for item in revisions)
                latest = [item for item in revisions if item[0] == latest_revision]
                if len(latest) != 1:
                    invalid_capsule_ids.add(capsule_id)
                    continue
                row, metadata = latest[0][1], latest[0][2]
                if (
                    row["state"] != "active"
                    or metadata.get("status") not in {"active", "challenged"}
                ):
                    continue
                prior_counterevidence: set[str] = set()
                prior = [
                    item for item in revisions if item[0] == latest_revision - 1
                ]
                if len(prior) == 1:
                    prior_counterevidence = _prior_counterevidence_ids(
                        [prior[0][2]]
                    )
                route = patch_routes.get(_clean(metadata.get("patch_id")), "")
                latest_capsules.append(
                    (row, metadata, route, prior_counterevidence)
                )

            for capsule_row, metadata, route, prior_counterevidence in latest_capsules:
                claims = _v3._validate_claims(metadata.get("claims"), stored=True)
                capsule_id = _clean(metadata.get("capsule_id"))
                revision = metadata.get("revision")
                try:
                    summary = validate_semantic_summary(
                        capsule_row["value"],
                        claims,
                        label="active Slow capsule summary",
                    )
                    expected_summary = _semantic_summary_projection(
                        _canonical_patch_claims(claims)
                    )
                    if summary != expected_summary:
                        raise PatchValidationError(
                            "active Slow capsule summary differs from final claims"
                        )
                except PatchValidationError as exc:
                    semantic_integrity_issues.append(
                        {
                            "code": "invalid_semantic_summary",
                            "capsule_id": capsule_id,
                            "revision": revision,
                            "error": str(exc),
                        }
                    )
                if metadata.get("summary_contract_version") != SLOW_SUMMARY_CONTRACT_VERSION:
                    semantic_integrity_issues.append(
                        {
                            "code": "missing_lossless_summary_contract",
                            "capsule_id": capsule_id,
                            "revision": revision,
                            "stored_contract_version": metadata.get(
                                "summary_contract_version"
                            ),
                        }
                    )
                capsule_key = _clean(metadata.get("capsule_key")).casefold()
                if metadata.get("summary_contract_version") == SLOW_SUMMARY_CONTRACT_VERSION:
                    try:
                        capsule_key = _normalize_capsule_key(
                            capsule_key, "stored capsule_key"
                        )
                    except PatchValidationError as exc:
                        semantic_integrity_issues.append(
                            {
                                "code": "invalid_capsule_key",
                                "capsule_id": capsule_id,
                                "revision": revision,
                                "error": str(exc),
                            }
                        )
                if capsule_key:
                    capsule_key_owners.setdefault(
                        (_clean(metadata.get("region_key")), capsule_key), set()
                    ).add(capsule_id)
                if (
                    metadata.get("partition_contract_version")
                    != SLOW_PARTITION_CONTRACT_VERSION
                ):
                    semantic_integrity_issues.append(
                        {
                            "code": "semantic_partition_migration_required",
                            "capsule_id": capsule_id,
                            "revision": revision,
                            "region_key": metadata.get("region_key"),
                            "claim_count": len(claims),
                        }
                    )
                active_claim_count += len(claims)
                claim_roles: list[tuple[str, set[str], set[str]]] = []
                for claim in claims:
                    cited_ids.update(claim["support"])
                    cited_ids.update(claim["counterevidence"])
                    claim_id = _clean(claim.get("claim_id"))
                    claim_slot = _clean(claim.get("canonical_slot"))
                    claim_text = _normal_text(claim.get("text"))
                    claim_support = set(claim["support"])
                    claim_counter = set(claim["counterevidence"])
                    complementary_support_bundle = (
                        _controlled_complementary_support_bundle(
                            claim_slot, claim_support, leaves_by_id
                        )
                    )
                    claim_identity_capsules.setdefault(
                        (claim_slot, claim_text), set()
                    ).add(capsule_id)
                    support_text_groups = _support_text_groups(
                        claim_support, leaves_by_id
                    )
                    if (
                        len(support_text_groups) > 1
                        and not complementary_support_bundle
                    ):
                        semantic_integrity_issues.append(
                            {
                                "code": "support_distinct_fast_values_merged",
                                "capsule_id": capsule_id,
                                "revision": revision,
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
                    for role, evidence_values in (
                        ("support", claim_support),
                        ("counterevidence", claim_counter),
                    ):
                        for evidence_id in evidence_values:
                            location = {
                                "capsule_id": capsule_id,
                                "revision": revision,
                                "claim_id": claim_id,
                                "role": role,
                            }
                            evidence_locations.setdefault(evidence_id, []).append(
                                location
                            )
                            citation_bindings.setdefault(evidence_id, []).append(
                                (
                                    role,
                                    claim_slot,
                                    claim_text,
                                    _json(location),
                                )
                            )
                    shared_roles = claim_support & claim_counter
                    if shared_roles:
                        semantic_integrity_issues.append(
                            {
                                "code": "same_evidence_support_and_counterevidence",
                                "capsule_id": _clean(metadata.get("capsule_id")),
                                "revision": metadata.get("revision"),
                                "claim_id": claim_id,
                                "evidence_ids": sorted(shared_roles),
                            }
                        )
                    if metadata.get("action") == "create" and claim_counter:
                        semantic_integrity_issues.append(
                            {
                                "code": "active_create_contains_counterevidence",
                                "capsule_id": _clean(metadata.get("capsule_id")),
                                "revision": metadata.get("revision"),
                                "claim_id": claim_id,
                                "evidence_ids": sorted(claim_counter),
                            }
                        )
                    claim_roles.append((claim_id, claim_support, claim_counter))
                    for evidence_id in claim["support"]:
                        leaf = leaves_by_id.get(evidence_id)
                        if leaf is None:
                            semantic_integrity_issues.append(
                                {
                                    "code": "support_missing_fast_leaf",
                                    "capsule_id": _clean(metadata.get("capsule_id")),
                                    "revision": metadata.get("revision"),
                                    "claim_id": claim_id,
                                    "evidence_id": evidence_id,
                                }
                            )
                        elif (
                            _leaf_slot(leaf) != claim_slot
                            and not complementary_support_bundle
                        ):
                            semantic_integrity_issues.append(
                                {
                                    "code": "support_canonical_slot_mismatch",
                                    "capsule_id": _clean(metadata.get("capsule_id")),
                                    "revision": metadata.get("revision"),
                                    "claim_id": claim_id,
                                    "claim_slot": claim_slot,
                                    "evidence_id": evidence_id,
                                    "evidence_slot": _leaf_slot(leaf),
                                }
                            )
                        elif not (
                            _is_current_durable(leaf)
                            or _is_challenged_durable(leaf)
                        ):
                            semantic_integrity_issues.append(
                                {
                                    "code": "support_noncurrent_fast_leaf",
                                    "capsule_id": _clean(metadata.get("capsule_id")),
                                    "revision": metadata.get("revision"),
                                    "claim_id": claim_id,
                                    "evidence_id": evidence_id,
                                    "evidence_state": _leaf_state(leaf),
                                }
                            )
                    for evidence_id in claim["counterevidence"]:
                        leaf = leaves_by_id.get(evidence_id)
                        if leaf is None:
                            semantic_integrity_issues.append(
                                {
                                    "code": "counterevidence_missing_fast_leaf",
                                    "capsule_id": _clean(metadata.get("capsule_id")),
                                    "revision": metadata.get("revision"),
                                    "claim_id": claim_id,
                                    "evidence_id": evidence_id,
                                }
                            )
                            continue
                        if _normal_text(_leaf_text(leaf)) == claim_text:
                            semantic_integrity_issues.append(
                                {
                                    "code": "counterevidence_identical_to_claim",
                                    "capsule_id": _clean(metadata.get("capsule_id")),
                                    "revision": metadata.get("revision"),
                                    "claim_id": claim_id,
                                    "evidence_id": evidence_id,
                                }
                            )
                        if (
                            route == "flash"
                            and evidence_id not in prior_counterevidence
                            and not _is_counterevidence(leaf)
                        ):
                            semantic_integrity_issues.append(
                                {
                                    "code": "flash_invented_counterevidence",
                                    "capsule_id": _clean(metadata.get("capsule_id")),
                                    "revision": metadata.get("revision"),
                                    "claim_id": claim_id,
                                    "evidence_id": evidence_id,
                                    "route": route,
                                }
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
                        if (
                            left_support & right_counter
                            and right_support & left_counter
                        ):
                            semantic_integrity_issues.append(
                                {
                                    "code": "reciprocal_counterevidence_cycle",
                                    "capsule_id": _clean(metadata.get("capsule_id")),
                                    "revision": metadata.get("revision"),
                                    "claim_ids": sorted(
                                        [left_claim_id, right_claim_id]
                                    ),
                                }
                            )
        for evidence_id, locations in evidence_locations.items():
            if len(locations) > 1:
                try:
                    _validate_repeated_evidence_bindings(
                        leaves_by_id,
                        {evidence_id: citation_bindings[evidence_id]},
                    )
                except PatchValidationError as exc:
                    semantic_integrity_issues.append(
                        {
                            "code": "fast_evidence_assigned_multiple_times",
                            "evidence_id": evidence_id,
                            "locations": locations,
                            "error": str(exc),
                        }
                    )
        for (slot, claim_text), capsule_ids in claim_identity_capsules.items():
            if len(capsule_ids) > 1:
                semantic_integrity_issues.append(
                    {
                        "code": "semantic_claim_split_across_capsules",
                        "canonical_slot": slot,
                        "normalized_claim_text": claim_text,
                        "capsule_ids": sorted(capsule_ids),
                    }
                )
        for (region_key, capsule_key), capsule_ids in capsule_key_owners.items():
            if len(capsule_ids) > 1:
                semantic_integrity_issues.append(
                    {
                        "code": "duplicate_capsule_key_in_region",
                        "region_key": region_key,
                        "capsule_key": capsule_key,
                        "capsule_ids": sorted(capsule_ids),
                    }
                )
        cited_eligible = eligible_ids & cited_ids
        uncited_eligible = sorted(eligible_ids - cited_ids)
        eligible_count = len(eligible_ids)
        semantic_integrity_issues.sort(key=_json)
        return {
            "schema_version": "tmcra.v4.slow-promotion-coverage.3",
            "complete": (
                not uncited_eligible
                and not invalid_capsule_ids
                and not semantic_integrity_issues
            ),
            "eligible_current_durable_count": eligible_count,
            "cited_current_durable_count": len(cited_eligible),
            "coverage_ratio": (
                round(len(cited_eligible) / eligible_count, 6)
                if eligible_count
                else 1.0
            ),
            "uncited_current_durable_ids": uncited_eligible,
            "challenged_durable_count": len(challenged_ids),
            "uncertain_count": len(uncertain_ids),
            "episodic_count": len(episodic_ids),
            "inactive_or_other_count": len(inactive_ids),
            "active_or_challenged_claim_count": active_claim_count,
            "invalid_capsule_identity_count": len(invalid_capsule_ids),
            "invalid_capsule_ids": sorted(invalid_capsule_ids),
            "semantic_integrity_issue_count": len(semantic_integrity_issues),
            "semantic_integrity_issues": semantic_integrity_issues,
        }

    def drain(
        self, manager: Any, *, batch_size: int | None = None
    ) -> list[str]:
        """Drain independent jobs while retaining complete invalid model responses."""
        self.recover_interrupted_attempts()
        if batch_size is not None and batch_size <= 0:
            raise SlowGraphError("batch_size must be positive")
        owner = self._claim_owner()
        results: list[str] = []
        failures: list[dict[str, str]] = []
        processed = 0
        while batch_size is None or processed < batch_size:
            claim = self._claim_pending_job(None, owner=owner)
            if claim is None:
                break
            processed += 1
            try:
                results.append(self._run_claimed_job(claim, manager))
            except Exception as exc:
                metadata = dict(getattr(manager, "last_call_metadata", {}) or {})
                complete_model_response = (
                    metadata.get("physical_api_call") is True
                    and int(metadata.get("physical_api_calls", 0) or 0) >= 1
                    and int(metadata.get("http_status", 0) or 0) == 200
                    and metadata.get("finish_reason") == "stop"
                    and metadata.get("status") in {"response_received", "completed"}
                    and _clean(metadata.get("raw_response"))
                )
                if not complete_model_response:
                    raise
                failures.append(
                    {
                        "job_id": claim.job_id,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    }
                )
        if failures:
            raise SlowGraphError(
                "slow graph retained complete invalid model responses: "
                + _json(failures)
            )
        return results

    def _propose_with_lease_heartbeat(
        self,
        claim: JobClaim,
        manager: Any,
        region: Mapping[str, Any],
        capsules: list[dict[str, Any]],
    ) -> dict[str, Any]:
        patch = super()._propose_with_lease_heartbeat(
            claim, manager, region, capsules
        )
        _validate_promotion_patch(region, capsules, patch)
        return patch

    def audit(
        self, scope_id: str, *, require_promotion_coverage: bool = False
    ) -> dict[str, Any]:
        result = dict(super().audit(scope_id))
        promotion_coverage = self.promotion_coverage(scope_id)
        if require_promotion_coverage and not promotion_coverage["complete"]:
            failures: list[str] = []
            if promotion_coverage["uncited_current_durable_ids"]:
                failures.append(
                    "current durable Fast evidence is missing from active Slow claims: "
                    + _json(
                        promotion_coverage["uncited_current_durable_ids"][:20]
                    )
                )
            if promotion_coverage["invalid_capsule_ids"]:
                failures.append(
                    "active Slow capsule identity is invalid: "
                    + _json(promotion_coverage["invalid_capsule_ids"][:20])
                )
            if promotion_coverage["semantic_integrity_issues"]:
                failures.append(
                    "active Slow claim semantic integrity failed: "
                    + _json(
                        promotion_coverage["semantic_integrity_issues"][:20]
                    )
                )
            raise AuditError("; ".join(failures))
        route_counts: dict[str, int] = {}
        usage = {
            "physical_api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
        }
        with self.connection() as con:
            rows = con.execute(
                "SELECT call_metadata_json FROM slow_graph_attempts WHERE scope_id=?",
                (scope_id,),
            ).fetchall()
        for row in rows:
            metadata = _v3._strict_json(
                row["call_metadata_json"],
                label="patch call metadata",
                expected=dict,
            )
            route = _clean(metadata.get("route")) or "unknown"
            route_counts[route] = route_counts.get(route, 0) + 1
            usage["physical_api_calls"] += int(metadata.get("physical_api_calls", 0) or 0)
            raw_usage = metadata.get("usage")
            if isinstance(raw_usage, Mapping):
                for key in ("prompt_tokens", "completion_tokens", "cache_read_input_tokens", "cache_hit_tokens", "cache_miss_tokens", "total_tokens"):
                    usage[key] += int(raw_usage.get(key, 0) or 0)
            cost_audit = metadata.get("cost_audit")
            if isinstance(cost_audit, Mapping):
                usage["estimated_cost"] += float(cost_audit.get("estimated_cost", 0.0) or 0.0)
        zero_call_recoveries = 0
        zero_call_promotion_recoveries = 0
        zero_call_projection_recoveries = 0
        process_loss_recoveries = 0
        process_loss_potential_min = 0
        process_loss_potential_max = 0
        with self.connection() as con:
            recovery_table = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='slow_graph_zero_call_recoveries'"
            ).fetchone()
            if recovery_table is not None:
                recoveries = con.execute(
                    "SELECT * FROM slow_graph_zero_call_recoveries "
                    "WHERE scope_id=? ORDER BY created_at,recovery_id",
                    (scope_id,),
                ).fetchall()
                for recovery in recoveries:
                    attempt = con.execute(
                        "SELECT * FROM slow_graph_attempts "
                        "WHERE attempt_id=? AND job_id=?",
                        (recovery["attempt_id"], recovery["job_id"]),
                    ).fetchone()
                    if attempt is None:
                        raise AuditError(
                            "zero-call configuration recovery references a missing attempt"
                        )
                    raw_metadata = _clean(attempt["call_metadata_json"])
                    metadata = _v3._strict_json(
                        raw_metadata,
                        label="zero-call recovery metadata",
                        expected=dict,
                    )
                    error = _clean(attempt["error"])
                    route = _clean(metadata.get("route"))
                    if (
                        int(
                            recovery["physical_api_calls"]
                            if recovery["physical_api_calls"] is not None
                            else -1
                        )
                        != 0
                        or route != _clean(recovery["route"])
                        or error != ZERO_CALL_CONFIGURATION_ERRORS.get(route)
                        or metadata.get("physical_api_call") is not False
                        or int(metadata.get("physical_api_calls", -1)) != 0
                        or hashlib.sha256(error.encode("utf-8")).hexdigest()
                        != _clean(recovery["error_sha256"])
                        or hashlib.sha256(raw_metadata.encode("utf-8")).hexdigest()
                        != _clean(recovery["call_metadata_sha256"])
                    ):
                        raise AuditError(
                            "zero-call configuration recovery contract is invalid"
                        )
                zero_call_recoveries = len(recoveries)
            promotion_table = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='slow_graph_zero_call_promotion_recoveries'"
            ).fetchone()
            if promotion_table is not None:
                promotion_recoveries = con.execute(
                    "SELECT * FROM slow_graph_zero_call_promotion_recoveries "
                    "WHERE scope_id=? ORDER BY created_at,recovery_id",
                    (scope_id,),
                ).fetchall()
                for recovery in promotion_recoveries:
                    attempt = con.execute(
                        "SELECT * FROM slow_graph_attempts "
                        "WHERE attempt_id=? AND job_id=?",
                        (recovery["attempt_id"], recovery["job_id"]),
                    ).fetchone()
                    if attempt is None:
                        raise AuditError(
                            "zero-call promotion recovery references a missing attempt"
                        )
                    raw_metadata = _clean(attempt["call_metadata_json"])
                    metadata = _v3._strict_json(
                        raw_metadata,
                        label="zero-call promotion recovery metadata",
                        expected=dict,
                    )
                    error = _clean(attempt["error"])
                    eligible_ids = metadata.get("eligible_evidence_ids")
                    challenged_ids = metadata.get("challenged_evidence_ids")
                    delta_ids = metadata.get("delta_evidence_ids")
                    if (
                        int(
                            recovery["physical_api_calls"]
                            if recovery["physical_api_calls"] is not None
                            else -1
                        )
                        != 0
                        or not error.startswith(
                            "noop cannot consume uncited current durable Fast evidence: "
                        )
                        or _clean(metadata.get("route")) != "deterministic_noop"
                        or _clean(metadata.get("route_reason"))
                        != "new capsule blocked by unresolved fast challenge"
                        or metadata.get("physical_api_call") is not False
                        or int(metadata.get("physical_api_calls", -1)) != 0
                        or not isinstance(eligible_ids, list)
                        or not eligible_ids
                        or not isinstance(challenged_ids, list)
                        or not challenged_ids
                        or not isinstance(delta_ids, list)
                        or not set(eligible_ids) <= set(delta_ids)
                        or hashlib.sha256(error.encode("utf-8")).hexdigest()
                        != _clean(recovery["error_sha256"])
                        or hashlib.sha256(raw_metadata.encode("utf-8")).hexdigest()
                        != _clean(recovery["call_metadata_sha256"])
                    ):
                        raise AuditError(
                            "zero-call promotion recovery contract is invalid"
                        )
                zero_call_promotion_recoveries = len(promotion_recoveries)
            projection_table = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='slow_graph_zero_call_projection_recoveries'"
            ).fetchone()
            if projection_table is not None:
                projection_recoveries = con.execute(
                    "SELECT * FROM slow_graph_zero_call_projection_recoveries "
                    "WHERE scope_id=? ORDER BY created_at,recovery_id",
                    (scope_id,),
                ).fetchall()
                for recovery in projection_recoveries:
                    attempt = con.execute(
                        "SELECT * FROM slow_graph_attempts "
                        "WHERE attempt_id=? AND job_id=?",
                        (recovery["attempt_id"], recovery["job_id"]),
                    ).fetchone()
                    job = con.execute(
                        "SELECT * FROM slow_graph_jobs WHERE job_id=?",
                        (recovery["job_id"],),
                    ).fetchone()
                    if attempt is None or job is None:
                        raise AuditError(
                            "zero-call projection recovery references missing state"
                        )
                    raw_metadata = _clean(attempt["call_metadata_json"])
                    metadata = _v3._strict_json(
                        raw_metadata,
                        label="zero-call projection recovery metadata",
                        expected=dict,
                    )
                    error = _clean(attempt["error"])
                    evidence_ids = _v3._strict_json(
                        job["evidence_ids_json"],
                        label="projection recovery evidence IDs",
                        expected=list,
                    )
                    evidence = self._evidence(
                        con, job["scope_id"], evidence_ids
                    )
                    offending_paths = []
                    for index, leaf in enumerate(evidence):
                        for key, value in _leaf_metadata(leaf).items():
                            if not _forbidden_field(key):
                                continue
                            if key != "origin_answer_ids" or value != []:
                                raise AuditError(
                                    "projection recovery contains non-empty benchmark metadata"
                                )
                            offending_paths.append(
                                f"payload.evidence[{index}].metadata.origin_answer_ids"
                            )
                    stored_paths = _v3._strict_json(
                        recovery["offending_paths_json"],
                        label="projection recovery offending paths",
                        expected=list,
                    )
                    public_region = {
                        "region_key": _required_text(
                            job["region_key"], "region key"
                        ),
                        "evidence": [_public_leaf(item) for item in evidence],
                    }
                    _assert_no_benchmark_fields(public_region)
                    if (
                        int(
                            recovery["physical_api_calls"]
                            if recovery["physical_api_calls"] is not None
                            else -1
                        )
                        != 0
                        or metadata.get("physical_api_call") is not False
                        or int(metadata.get("physical_api_calls", -1)) != 0
                        or hashlib.sha256(error.encode("utf-8")).hexdigest()
                        != _clean(recovery["error_sha256"])
                        or hashlib.sha256(raw_metadata.encode("utf-8")).hexdigest()
                        != _clean(recovery["call_metadata_sha256"])
                        or _digest(public_region)
                        != _clean(recovery["public_projection_sha256"])
                        or offending_paths != stored_paths
                    ):
                        raise AuditError(
                            "zero-call projection recovery contract is invalid"
                        )
                zero_call_projection_recoveries = len(projection_recoveries)
            process_loss_rows = con.execute(
                "SELECT * FROM slow_graph_process_loss_recoveries "
                "WHERE scope_id=? ORDER BY recovered_at,recovery_id",
                (scope_id,),
            ).fetchall()
            for recovery in process_loss_rows:
                attempt = con.execute(
                    "SELECT * FROM slow_graph_attempts WHERE attempt_id=? "
                    "AND job_id=? AND scope_id=?",
                    (
                        recovery["attempt_id"],
                        recovery["job_id"],
                        recovery["scope_id"],
                    ),
                ).fetchone()
                job = con.execute(
                    "SELECT * FROM slow_graph_jobs WHERE job_id=? AND scope_id=?",
                    (recovery["job_id"], recovery["scope_id"]),
                ).fetchone()
                if attempt is None or job is None:
                    raise AuditError(
                        "process-loss recovery references missing Slow state"
                    )
                raw_metadata = _required_text(
                    attempt["call_metadata_json"],
                    "process-loss attempt metadata",
                )
                metadata = _v3._strict_json(
                    raw_metadata,
                    label="process-loss attempt metadata",
                    expected=dict,
                )
                expected_recovery_id = "sgr_" + _digest(
                    {
                        "job_id": recovery["job_id"],
                        "attempt_id": recovery["attempt_id"],
                        "claim_token": recovery["claim_token"],
                        "claim_owner": recovery["claim_owner"],
                        "attempt_metadata_sha256": recovery[
                            "attempt_metadata_sha256"
                        ],
                    }
                )[:32]
                source = _clean(recovery["recovery_source"])
                if (
                    metadata
                    or attempt["status"] != "expired"
                    or _clean(attempt["error"])
                    != PROCESS_LOSS_INTERRUPTION_ERROR
                    or attempt["completed_at"] is None
                    or _clean(attempt["claim_token"])
                    != _clean(recovery["claim_token"])
                    or _clean(attempt["claim_owner"])
                    != _clean(recovery["claim_owner"])
                    or int(attempt["created_at"])
                    != int(recovery["attempt_created_at"])
                    or hashlib.sha256(raw_metadata.encode("utf-8")).hexdigest()
                    != _clean(recovery["attempt_metadata_sha256"])
                    or hashlib.sha256(
                        PROCESS_LOSS_INTERRUPTION_ERROR.encode("utf-8")
                    ).hexdigest()
                    != _clean(recovery["interruption_error_sha256"])
                    or _clean(recovery["external_call_outcome"]) != "uncertain"
                    or int(recovery["potential_duplicate_physical_calls_min"])
                    != 0
                    or int(recovery["potential_duplicate_physical_calls_max"])
                    != SLOW_PROCESS_LOSS_PHYSICAL_CALLS_MAX
                    or int(job["attempts"])
                    < int(recovery["job_attempts_before"]) + 1
                    or _clean(recovery["recovery_id"]) != expected_recovery_id
                    or source
                    not in {"expired_claimed_started", "legacy_expired_failed"}
                    or (
                        source == "expired_claimed_started"
                        and recovery["lease_expires_at"] is None
                    )
                    or (
                        source == "legacy_expired_failed"
                        and recovery["lease_expires_at"] is not None
                    )
                ):
                    raise AuditError(
                        "process-loss recovery contract is invalid"
                    )
            process_loss_recoveries = len(process_loss_rows)
            process_loss_potential_min = sum(
                int(row["potential_duplicate_physical_calls_min"])
                for row in process_loss_rows
            )
            process_loss_potential_max = sum(
                int(row["potential_duplicate_physical_calls_max"])
                for row in process_loss_rows
            )
        result["route_counts"] = dict(sorted(route_counts.items()))
        result["usage"] = usage
        result["zero_call_configuration_recoveries"] = zero_call_recoveries
        result["zero_call_promotion_recoveries"] = (
            zero_call_promotion_recoveries
        )
        result["zero_call_projection_recoveries"] = (
            zero_call_projection_recoveries
        )
        result["process_loss_recoveries"] = process_loss_recoveries
        result["process_loss_unknown_external_outcomes"] = (
            process_loss_recoveries
        )
        result["process_loss_potential_duplicate_physical_calls_min"] = (
            process_loss_potential_min
        )
        result["process_loss_potential_duplicate_physical_calls_max"] = (
            process_loss_potential_max
        )
        result["promotion_coverage"] = promotion_coverage
        return result


def _saved_request_context(
    metadata: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    request = metadata.get("request")
    if not isinstance(request, Mapping):
        raise SlowGraphError("failed attempt has no saved request")
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise SlowGraphError("failed attempt request has no messages")
    user_messages = [
        item
        for item in messages
        if isinstance(item, Mapping) and item.get("role") == "user"
    ]
    if len(user_messages) != 1:
        raise SlowGraphError("failed attempt request must have exactly one user message")
    try:
        payload = json.loads(_required_text(user_messages[0].get("content"), "saved user content"))
    except json.JSONDecodeError as exc:
        raise SlowGraphError("failed attempt user content is not JSON") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"region", "capsules"}:
        raise SlowGraphError("failed attempt user payload has drifted")
    region = payload.get("region")
    capsules = payload.get("capsules")
    if not isinstance(region, Mapping) or not isinstance(capsules, list) or not all(
        isinstance(item, Mapping) for item in capsules
    ):
        raise SlowGraphError("failed attempt request context is invalid")
    return region, capsules


class _FailedRawPatchReplayManager:
    def __init__(
        self,
        *,
        patch: Mapping[str, Any],
        original_metadata: Mapping[str, Any],
        original_attempt_id: str,
        original_region: Mapping[str, Any],
        original_capsules: list[Mapping[str, Any]],
        transport_normalizations: list[dict[str, str]],
        controller_processing: Mapping[str, Any],
        context_source: str,
        evidence_snapshot_sha256: str,
        revalidation_route: str = "raw_response_revalidation",
        revalidation_reason: str = "unambiguous_transport_normalization",
        revalidation_details: Mapping[str, Any] | None = None,
    ) -> None:
        model = _required_text(original_metadata.get("model"), "saved model")
        self.model_config = {"model": model}
        self.prompt_hash = _digest(original_metadata.get("request", {}).get("messages", []))
        self.last_call_metadata: Mapping[str, Any] = {}
        self._patch = dict(patch)
        self._original_metadata = dict(original_metadata)
        self._original_attempt_id = original_attempt_id
        self._original_region = dict(original_region)
        self._original_capsules = [dict(item) for item in original_capsules]
        self._transport_normalizations = list(transport_normalizations)
        self._controller_processing = dict(controller_processing)
        self._context_source = context_source
        self._evidence_snapshot_sha256 = evidence_snapshot_sha256
        self._revalidation_route = _required_text(
            revalidation_route, "revalidation route"
        )
        self._revalidation_reason = _required_text(
            revalidation_reason, "revalidation reason"
        )
        self._revalidation_details = dict(revalidation_details or {})
        self._used = False

    @staticmethod
    def _capsule_identity(capsules: list[Mapping[str, Any]]) -> list[tuple[str, Any, str]]:
        return [
            (
                _clean(item.get("capsule_id")),
                item.get("revision"),
                _clean(item.get("status")),
            )
            for item in capsules
        ]

    def propose(
        self, region: Mapping[str, Any], capsules: list[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        if self._used:
            raise SlowGraphError("saved raw response replay may be consumed only once")
        if _clean(region.get("region_key")) != _clean(self._original_region.get("region_key")):
            raise SlowGraphError("slow graph region changed before raw response replay")
        if self._capsule_identity(capsules) != self._capsule_identity(
            self._original_capsules
        ):
            raise SlowGraphError("slow graph capsules changed before raw response replay")
        original_evidence = {
            _leaf_id(item)
            for item in self._original_region.get("evidence", [])
            if isinstance(item, Mapping)
        }
        current_evidence = {
            _leaf_id(item)
            for item in region.get("evidence", [])
            if isinstance(item, Mapping)
        }
        if not original_evidence.issubset(current_evidence):
            raise SlowGraphError("slow graph evidence changed before raw response replay")
        self._used = True
        raw_response = _required_text(
            self._original_metadata.get("raw_response"), "saved raw response"
        )
        self.last_call_metadata = {
            "route": self._revalidation_route,
            "route_reason": self._revalidation_reason,
            "status": "completed",
            "evidence_binding_contract_version": (
                SLOW_EVIDENCE_BINDING_CONTRACT_VERSION
            ),
            "physical_api_call": False,
            "physical_api_calls": 0,
            "api_provider": self._original_metadata.get("api_provider"),
            "model": self.model_config["model"],
            "prompt_version": self._original_metadata.get("prompt_version"),
            "attempt_count": 0,
            "original_attempt_id": self._original_attempt_id,
            "original_physical_call_id": self._original_metadata.get(
                "physical_call_id"
            ),
            "original_physical_api_calls": int(
                self._original_metadata.get("physical_api_calls", 0) or 0
            ),
            "original_attempt_count": int(
                self._original_metadata.get("attempt_count", 0) or 0
            ),
            "original_call_metadata_sha256": _digest(self._original_metadata),
            "raw_response_sha256": hashlib.sha256(
                raw_response.encode("utf-8")
            ).hexdigest(),
            "normalized_patch_sha256": _digest(self._patch),
            "transport_normalizations": self._transport_normalizations,
            "revalidation_context_source": self._context_source,
            "evidence_snapshot_sha256": self._evidence_snapshot_sha256,
            "revalidation_details": self._revalidation_details,
            **self._controller_processing,
        }
        return self._patch


def _saved_response_patch(
    metadata: Mapping[str, Any], *, label: str
) -> tuple[Mapping[str, Any], str, str]:
    raw_response_text = _required_text(
        metadata.get("raw_response"), f"{label} raw response"
    )
    content = _required_text(metadata.get("content"), f"{label} response content")
    try:
        raw_response = json.loads(raw_response_text)
        raw_patch = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SlowGraphError(f"{label} response is not valid JSON") from exc
    choices = raw_response.get("choices") if isinstance(raw_response, Mapping) else None
    choice = (
        choices[0]
        if isinstance(choices, list)
        and len(choices) == 1
        and isinstance(choices[0], Mapping)
        else None
    )
    message = choice.get("message") if isinstance(choice, Mapping) else None
    if (
        not isinstance(message, Mapping)
        or _clean(message.get("content")) != content
        or _clean(choice.get("finish_reason")) != "stop"
    ):
        raise SlowGraphError(f"{label} response envelope does not match saved content")
    if not isinstance(raw_patch, Mapping):
        raise SlowGraphError(f"{label} GraphPatch is not an object")
    return raw_patch, raw_response_text, content


def _prepare_revalidated_patch(
    raw_patch: Mapping[str, Any],
    *,
    current_region: Mapping[str, Any],
    current_capsules: list[Mapping[str, Any]],
    original_region: Mapping[str, Any],
    original_capsules: list[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    validation_route: str,
    context_source: str,
    transport_policy: str,
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    normalized_patch, normalizations = _normalize_transport_patch(
        raw_patch, original_capsules, original_region
    )
    if transport_policy == "required" and not normalizations:
        raise SlowGraphError("saved GraphPatch has no approved transport normalization")
    if transport_policy == "forbidden" and normalizations:
        raise SlowGraphError(
            "semantic-policy revalidation cannot include transport normalization"
        )
    if transport_policy not in {"required", "forbidden"}:
        raise SlowGraphError("saved GraphPatch transport policy is invalid")
    validate_patch(normalized_patch)
    _validate_generic_create_partition_keys(
        current_region.get("region_key"), normalized_patch
    )
    if context_source != "saved_request" and normalized_patch != {
        "operations": [{"action": "noop"}]
    }:
        raise SlowGraphError(
            "legacy response without saved request may revalidate only a pure noop"
        )
    _, required_ids = _required_promotion_ids(current_region, current_capsules)
    partition_capsule_ids = {
        _clean(item)
        for item in metadata.get("semantic_partition_capsule_ids") or ()
        if _clean(item)
    }
    TieredGraphPatchManager._validate_route_actions(
        validation_route,
        normalized_patch,
        original_capsules,
        required_evidence_ids=required_ids,
        partition_capsule_ids=partition_capsule_ids,
    )
    _validate_claim_evidence_contract(
        current_region,
        current_capsules,
        normalized_patch,
        route=validation_route,
    )
    merge_audit: dict[str, Any] | None = None
    if validation_route == "flash" and required_ids:
        _validate_flash_delta_patch(
            normalized_patch, current_capsules, required_ids
        )
        if current_capsules:
            committed_patch, merge_audit = _merge_flash_delta_patch(
                normalized_patch, current_capsules
            )
        else:
            committed_patch = _materialize_lossless_summaries(normalized_patch)
    else:
        committed_patch = _materialize_lossless_summaries(normalized_patch)
    validate_patch(committed_patch, require_lossless_summary=True)
    _validate_generic_create_partition_keys(
        current_region.get("region_key"), committed_patch
    )
    TieredGraphPatchManager._validate_route_actions(
        validation_route,
        committed_patch,
        current_capsules,
        required_evidence_ids=required_ids,
        partition_capsule_ids=partition_capsule_ids,
    )
    _validate_claim_evidence_contract(
        current_region,
        current_capsules,
        committed_patch,
        route=validation_route,
    )
    _validate_promotion_patch(
        current_region,
        current_capsules,
        committed_patch,
        required_evidence_ids=required_ids,
    )
    controller_processing: dict[str, Any] = {
        "controller_summary_materialization": {
            "schema_version": SLOW_SUMMARY_CONTRACT_VERSION,
            "model_patch_sha256": _digest(normalized_patch),
            "committed_patch_sha256": _digest(committed_patch),
        }
    }
    if merge_audit is not None:
        controller_processing["controller_delta_merge"] = merge_audit
    return committed_patch, normalizations, controller_processing


def revalidate_failed_raw_response(
    store: V4SlowGraphStore, job_id: str
) -> str:
    """Apply a saved, complete API response after a narrow transport normalization."""
    with store.connection() as con:
        job = con.execute(
            "SELECT * FROM slow_graph_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if job is None or job["status"] != "failed" or job["claim_token"] is not None:
            raise SlowGraphError("raw response revalidation requires one unclaimed failed job")
        attempts = con.execute(
            "SELECT rowid,* FROM slow_graph_attempts WHERE job_id=? AND status='failed' "
            "ORDER BY created_at DESC,rowid DESC",
            (job_id,),
        ).fetchall()
        attempt_count = int(
            con.execute(
                "SELECT count(*) FROM slow_graph_attempts WHERE job_id=?", (job_id,)
            ).fetchone()[0]
        )
        patch_count = int(
            con.execute(
                "SELECT count(*) FROM slow_graph_patches WHERE job_id=?", (job_id,)
            ).fetchone()[0]
        )
        evidence_ids = _v3._strict_json(
            job["evidence_ids_json"], label="job evidence IDs", expected=list
        )
        current_region = {
            "region_key": job["region_key"],
            "evidence": store._evidence(con, job["scope_id"], evidence_ids),
        }
        current_capsules = store._capsules(
            con, job["scope_id"], job["region_key"]
        )
    if patch_count != 0:
        raise SlowGraphError("failed job already has an applied patch")
    attempt = None
    metadata: dict[str, Any] | None = None
    for candidate in attempts:
        candidate_metadata = _v3._strict_json(
            candidate["call_metadata_json"],
            label="failed call metadata",
            expected=dict,
        )
        if (
            candidate_metadata.get("physical_api_call") is True
            and int(candidate_metadata.get("physical_api_calls", 0) or 0) == 1
            and candidate_metadata.get("status") in {"response_received", "completed"}
            and int(candidate_metadata.get("http_status", 0) or 0) == 200
            and candidate_metadata.get("finish_reason") == "stop"
            and _clean(candidate_metadata.get("raw_response"))
            and _clean(candidate_metadata.get("content"))
        ):
            attempt = candidate
            metadata = candidate_metadata
            break
    if attempt is None or metadata is None or not _clean(attempt["error"]):
        raise SlowGraphError(
            "failed job has no complete HTTP 200 physical response to revalidate"
        )
    raw_patch, _, _ = _saved_response_patch(metadata, label="saved failed")
    context_source = "saved_request"
    if isinstance(metadata.get("request"), Mapping):
        original_region, original_capsules = _saved_request_context(metadata)
    else:
        if attempt_count != 1 or patch_count != 0 or current_capsules:
            raise SlowGraphError(
                "legacy response without saved request is not a pristine capsule-free job"
            )
        original_region = current_region
        original_capsules = []
        context_source = "immutable_job_snapshot_for_legacy_noop"
    route = _required_text(metadata.get("route"), "saved route")
    if route not in {"flash", "pro", "flash_to_pro"}:
        raise SlowGraphError("saved route is not an API GraphPatch route")
    validation_route = "pro" if route in {"pro", "flash_to_pro"} else "flash"
    committed_patch, normalizations, controller_processing = (
        _prepare_revalidated_patch(
            raw_patch,
            current_region=current_region,
            current_capsules=current_capsules,
            original_region=original_region,
            original_capsules=original_capsules,
            metadata=metadata,
            validation_route=validation_route,
            context_source=context_source,
            transport_policy="required",
        )
    )
    manager = _FailedRawPatchReplayManager(
        patch=committed_patch,
        original_metadata=metadata,
        original_attempt_id=str(attempt["attempt_id"]),
        original_region=original_region,
        original_capsules=original_capsules,
        transport_normalizations=normalizations,
        controller_processing=controller_processing,
        context_source=context_source,
        evidence_snapshot_sha256=_digest(evidence_ids),
    )
    store.resume(job_id)
    return store.run_job(job_id, manager)


def _semantic_policy_failure_class(error: str) -> str:
    generic_pattern = re.compile(
        r"operations\[\d+\]\." + re.escape(GENERIC_MULTI_SLOT_CAPSULE_KEY_ERROR)
    )
    if generic_pattern.fullmatch(error):
        return "generic_capsule_key_policy"
    if error.startswith(LEGACY_SINGLE_BINDING_ERROR_PREFIX):
        raw_ids = error[len(LEGACY_SINGLE_BINDING_ERROR_PREFIX) :]
        try:
            evidence_ids = json.loads(raw_ids)
        except json.JSONDecodeError:
            return ""
        if (
            isinstance(evidence_ids, list)
            and evidence_ids
            and all(isinstance(item, str) and item.strip() for item in evidence_ids)
            and len(evidence_ids) == len(set(evidence_ids))
        ):
            return "compound_support_binding_policy"
    cross_slot_match = re.fullmatch(
        r"claim support canonical slot mismatch: "
        r"claim=(?P<claim>\S+) evidence=(?P<evidence>\S+) id=(?P<evidence_id>\S+)",
        error,
    )
    if cross_slot_match and cross_slot_match.group("evidence").startswith(
        cross_slot_match.group("claim") + "."
    ):
        return "complementary_subslot_support_policy"
    return ""


def revalidate_failed_semantic_policy_response(
    store: V4SlowGraphStore, job_id: str
) -> str:
    """Replay the final complete two-call Pro result after a reviewed policy change."""
    with store.connection() as con:
        job = con.execute(
            "SELECT * FROM slow_graph_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if (
            job is None
            or job["status"] != "failed"
            or job["claim_token"] is not None
            or int(job["attempts"] or 0) != 1
        ):
            raise SlowGraphError(
                "semantic-policy revalidation requires one unclaimed failed job"
            )
        attempts = con.execute(
            "SELECT * FROM slow_graph_attempts WHERE job_id=? "
            "ORDER BY created_at,attempt_id",
            (job_id,),
        ).fetchall()
        patch_count = int(
            con.execute(
                "SELECT count(*) FROM slow_graph_patches WHERE job_id=?", (job_id,)
            ).fetchone()[0]
        )
        evidence_ids = _v3._strict_json(
            job["evidence_ids_json"], label="job evidence IDs", expected=list
        )
        current_region = {
            "region_key": job["region_key"],
            "evidence": store._evidence(con, job["scope_id"], evidence_ids),
        }
        current_capsules = store._capsules(
            con, job["scope_id"], job["region_key"]
        )
    if len(attempts) != 1 or patch_count != 0:
        raise SlowGraphError(
            "semantic-policy revalidation requires one attempt and no patch"
        )
    attempt = attempts[0]
    raw_metadata = _required_text(
        attempt["call_metadata_json"], "failed call metadata"
    )
    metadata = _v3._strict_json(
        raw_metadata, label="failed call metadata", expected=dict
    )
    error = _clean(attempt["error"])
    policy_failure_class = _semantic_policy_failure_class(error)
    initial_hash = _clean(metadata.get("initial_rejected_patch_sha256"))
    corrected_hash = _clean(metadata.get("corrected_patch_sha256"))
    tier_calls = metadata.get("tier_calls")
    if (
        attempt["status"] != "failed"
        or not policy_failure_class
        or _clean(job["last_error"]) != error
        or _clean(metadata.get("semantic_correction_validation_error")) != error
        or _clean(metadata.get("semantic_correction_rejection_error")) != error
        or metadata.get("semantic_correction_attempted") is not True
        or metadata.get("semantic_correction_applied") is not False
        or metadata.get("status") != "semantic_correction_rejected"
        or metadata.get("route") != "pro"
        or metadata.get("prompt_version") != SLOW_PROMPT_VERSION
        or metadata.get("physical_api_call") is not True
        or int(metadata.get("physical_api_calls", 0) or 0) != 2
        or int(metadata.get("attempt_count", 0) or 0) != 2
        or int(metadata.get("http_status", 0) or 0) != 200
        or metadata.get("finish_reason") != "stop"
        or len(initial_hash) != 64
        or len(corrected_hash) != 64
        or not isinstance(tier_calls, list)
        or len(tier_calls) != 2
    ):
        raise SlowGraphError(
            "failed attempt is not a reviewed complete two-call Pro policy failure"
        )

    call_hashes = (initial_hash, corrected_hash)
    call_ids: list[str] = []
    for index, (call, expected_hash) in enumerate(
        zip(tier_calls, call_hashes, strict=True)
    ):
        if not isinstance(call, Mapping):
            raise SlowGraphError("semantic-policy tier call metadata is invalid")
        expected_stage = "initial_pro" if index == 0 else "semantic_correction"
        request = call.get("request")
        call_id = _clean(call.get("physical_call_id"))
        if (
            call.get("tier_stage") != expected_stage
            or call.get("route") != "pro"
            or call.get("prompt_version") != SLOW_PROMPT_VERSION
            or call.get("physical_api_call") is not True
            or int(call.get("physical_api_calls", 0) or 0) != 1
            or int(call.get("attempt_count", 0) or 0) != 1
            or call.get("status") != "completed"
            or int(call.get("http_status", 0) or 0) != 200
            or call.get("finish_reason") != "stop"
            or not call_id
            or not isinstance(request, Mapping)
            or _clean(call.get("request_sha256")) != _digest(request)
        ):
            raise SlowGraphError(
                "semantic-policy tier call is not one complete durable Pro response"
            )
        call_patch, _, _ = _saved_response_patch(
            call, label=f"semantic-policy tier call {index}"
        )
        if _digest(call_patch) != expected_hash:
            raise SlowGraphError("semantic-policy tier call patch hash differs")
        call_ids.append(call_id)
    if len(set(call_ids)) != 2:
        raise SlowGraphError("semantic-policy physical call IDs are not unique")

    raw_patch, _, _ = _saved_response_patch(
        metadata, label="semantic-policy corrected"
    )
    if _digest(raw_patch) != corrected_hash:
        raise SlowGraphError("semantic-policy corrected patch hash differs")
    request = metadata.get("request")
    if (
        not isinstance(request, Mapping)
        or _clean(metadata.get("request_sha256")) != _digest(request)
        or request != tier_calls[-1].get("request")
    ):
        raise SlowGraphError("semantic-policy corrected request has drifted")
    original_region, original_capsules = _saved_request_context(metadata)

    def metadata_ids(name: str) -> set[str]:
        raw = metadata.get(name)
        if not isinstance(raw, list):
            raise SlowGraphError(f"semantic-policy {name} is not a list")
        values = [_required_text(item, name) for item in raw]
        if len(values) != len(set(values)):
            raise SlowGraphError(f"semantic-policy {name} contains duplicates")
        return set(values)

    current_evidence = [
        item
        for item in current_region.get("evidence", [])
        if isinstance(item, Mapping)
    ]
    current_by_id = {_leaf_id(item): item for item in current_evidence}
    current_ids = set(current_by_id)
    current_eligible = {
        evidence_id
        for evidence_id, item in current_by_id.items()
        if _is_current_durable(item)
    }
    current_challenged = {
        evidence_id
        for evidence_id, item in current_by_id.items()
        if _is_challenged_durable(item)
    }
    current_uncertain = {
        evidence_id
        for evidence_id, item in current_by_id.items()
        if _is_uncertain(item)
    }
    current_episodic = {
        evidence_id
        for evidence_id, item in current_by_id.items()
        if _is_episodic(item)
    }
    current_inactive = (
        current_ids
        - current_eligible
        - current_challenged
        - current_uncertain
        - current_episodic
    )
    current_visible = current_eligible | current_challenged
    saved_evidence = {
        _leaf_id(item): item
        for item in original_region.get("evidence", [])
        if isinstance(item, Mapping)
    }
    saved_evidence_ids = set(saved_evidence)
    saved_required_ids = {
        _required_text(item, "saved required evidence ID")
        for item in original_region.get("required_evidence_ids", [])
    }
    metadata_required_ids = metadata_ids("required_operation_evidence_ids")
    if (
        _clean(original_region.get("region_key"))
        != _clean(current_region.get("region_key"))
        or current_ids != set(evidence_ids)
        or current_eligible != metadata_ids("eligible_evidence_ids")
        or current_challenged != metadata_ids("challenged_evidence_ids")
        or current_uncertain != metadata_ids("uncertain_evidence_ids")
        or current_episodic != metadata_ids("episodic_evidence_ids")
        or current_inactive != metadata_ids("inactive_evidence_ids")
        or (current_uncertain | current_episodic | current_inactive)
        != metadata_ids("ignored_evidence_ids")
        or saved_evidence_ids != current_visible
        or saved_required_ids != metadata_required_ids
        or not metadata_required_ids.issubset(current_visible)
        or saved_evidence
        != {
            evidence_id: _public_leaf(current_by_id[evidence_id])
            for evidence_id in sorted(current_visible)
        }
        or _FailedRawPatchReplayManager._capsule_identity(original_capsules)
        != _FailedRawPatchReplayManager._capsule_identity(current_capsules)
    ):
        raise SlowGraphError("semantic-policy saved request context has drifted")
    committed_patch, normalizations, controller_processing = (
        _prepare_revalidated_patch(
            raw_patch,
            current_region=current_region,
            current_capsules=current_capsules,
            original_region=original_region,
            original_capsules=original_capsules,
            metadata=metadata,
            validation_route="pro",
            context_source="saved_request",
            transport_policy="forbidden",
        )
    )
    manager = _FailedRawPatchReplayManager(
        patch=committed_patch,
        original_metadata=metadata,
        original_attempt_id=str(attempt["attempt_id"]),
        original_region=original_region,
        original_capsules=original_capsules,
        transport_normalizations=normalizations,
        controller_processing=controller_processing,
        context_source="saved_request",
        evidence_snapshot_sha256=_digest(evidence_ids),
        revalidation_route="semantic_policy_revalidation",
        revalidation_reason={
            "generic_capsule_key_policy": (
                "generic_capsule_key_structural_policy_narrowing"
            ),
            "compound_support_binding_policy": (
                "compound_support_many_to_many_binding_policy"
            ),
            "complementary_subslot_support_policy": (
                "controlled_parent_subslot_support_synthesis_policy"
            ),
        }[policy_failure_class],
        revalidation_details={
            "schema_version": "tmcra.v4.slow-semantic-policy-revalidation.1",
            "policy_failure_class": policy_failure_class,
            "evidence_binding_contract_version": (
                SLOW_EVIDENCE_BINDING_CONTRACT_VERSION
            ),
            "failed_error_sha256": hashlib.sha256(
                error.encode("utf-8")
            ).hexdigest(),
            "failed_call_metadata_sha256": hashlib.sha256(
                raw_metadata.encode("utf-8")
            ).hexdigest(),
            "initial_rejected_patch_sha256": initial_hash,
            "corrected_patch_sha256": corrected_hash,
            "semantic_correction_changed_patch": initial_hash != corrected_hash,
            "physical_call_ids": call_ids,
        },
    )
    store.resume(job_id)
    return store.run_job(job_id, manager)


def resume_failed_model_validation(
    store: V4SlowGraphStore,
    job_id: str,
    manager: "TieredGraphPatchManager",
) -> str:
    """Explicitly reopen one complete Pro response rejected by local validation."""
    with store.connection() as con:
        job = con.execute(
            "SELECT * FROM slow_graph_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if (
            job is None
            or job["status"] != "failed"
            or job["claim_token"] is not None
            or int(job["attempts"] or 0) != 1
        ):
            raise SlowGraphError(
                "model-validation recovery requires one unclaimed failed attempt"
            )
        attempts = con.execute(
            "SELECT * FROM slow_graph_attempts WHERE job_id=? "
            "ORDER BY created_at,attempt_id",
            (job_id,),
        ).fetchall()
        patch_count = int(
            con.execute(
                "SELECT count(*) FROM slow_graph_patches WHERE job_id=?", (job_id,)
            ).fetchone()[0]
        )
    if len(attempts) != 1 or patch_count != 0:
        raise SlowGraphError(
            "model-validation recovery requires one attempt and no applied patch"
        )
    attempt = attempts[0]
    metadata = _v3._strict_json(
        _required_text(attempt["call_metadata_json"], "failed call metadata"),
        label="failed call metadata",
        expected=dict,
    )
    error = _clean(attempt["error"])
    if (
        attempt["status"] != "failed"
        or not error
        or _clean(job["last_error"]) != error
        or metadata.get("physical_api_call") is not True
        or int(metadata.get("physical_api_calls", 0) or 0) < 1
        or _clean(metadata.get("route")) not in {"pro", "flash_to_pro"}
        or _clean(metadata.get("status"))
        not in {"completed", "response_received", "semantic_correction_rejected"}
        or int(metadata.get("http_status", 0) or 0) != 200
        or _clean(metadata.get("finish_reason")) != "stop"
        or not _clean(metadata.get("raw_response"))
        or not _clean(metadata.get("content"))
    ):
        raise SlowGraphError(
            "failed attempt is not a complete Pro model-validation response"
        )
    store.resume(job_id)
    return store.run_job(job_id, manager)


def resume_failed_model_validation_after_prompt_migration(
    store: V4SlowGraphStore,
    job_id: str,
    manager: "TieredGraphPatchManager",
) -> str:
    """Reopen one twice-rejected compound-leaf job after the reviewed prompt migration."""
    with store.connection() as con:
        job = con.execute(
            "SELECT * FROM slow_graph_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if (
            job is None
            or job["status"] != "failed"
            or job["claim_token"] is not None
            or int(job["attempts"] or 0) != 2
        ):
            raise SlowGraphError(
                "prompt-migration recovery requires two unclaimed failed attempts"
            )
        attempts = con.execute(
            "SELECT * FROM slow_graph_attempts WHERE job_id=? "
            "ORDER BY created_at,attempt_id",
            (job_id,),
        ).fetchall()
        patch_count = int(
            con.execute(
                "SELECT count(*) FROM slow_graph_patches WHERE job_id=?", (job_id,)
            ).fetchone()[0]
        )
    if len(attempts) != 2 or patch_count != 0:
        raise SlowGraphError(
            "prompt-migration recovery requires two attempts and no applied patch"
        )
    duplicate_evidence_error = (
        "atomic Fast evidence may belong to only one resulting claim: "
    )
    attempt_prompt_versions: list[str] = []
    for attempt in attempts:
        metadata = _v3._strict_json(
            _required_text(attempt["call_metadata_json"], "failed call metadata"),
            label="failed call metadata",
            expected=dict,
        )
        if (
            attempt["status"] != "failed"
            or not _clean(attempt["error"]).startswith(duplicate_evidence_error)
            or metadata.get("physical_api_call") is not True
            or int(metadata.get("physical_api_calls", 0) or 0) < 1
            or _clean(metadata.get("route")) not in {"pro", "flash_to_pro"}
            or _clean(metadata.get("status")) != "semantic_correction_rejected"
            or int(metadata.get("http_status", 0) or 0) != 200
            or _clean(metadata.get("finish_reason")) != "stop"
            or _clean(metadata.get("prompt_version"))
            not in SLOW_PROMPT_MIGRATION_SOURCE_VERSIONS
            or not _clean(metadata.get("raw_response"))
            or not _clean(metadata.get("content"))
        ):
            raise SlowGraphError(
                "failed attempts are not the reviewed compound-leaf prompt-migration class"
            )
        attempt_prompt_versions.append(_clean(metadata.get("prompt_version")))
    if attempt_prompt_versions[-1] != SLOW_PROMPT_MIGRATION_SOURCE_VERSION:
        raise SlowGraphError(
            "prompt-migration recovery requires the final failed attempt on the source version"
        )
    if _clean(job["last_error"]) != _clean(attempts[-1]["error"]):
        raise SlowGraphError("prompt-migration job and final attempt errors differ")
    if SLOW_PROMPT_VERSION == SLOW_PROMPT_MIGRATION_SOURCE_VERSION:
        raise SlowGraphError("prompt-migration recovery requires a new prompt version")
    store.resume(job_id)
    return store.run_job(job_id, manager)


def resume_zero_call_configuration_failure(
    store: V4SlowGraphStore, job_id: str
) -> dict[str, Any]:
    """Reopen one preflight configuration failure proven to have made no call."""
    with store.connection() as con:
        con.execute("BEGIN IMMEDIATE")
        job = con.execute(
            "SELECT * FROM slow_graph_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if (
            job is None
            or job["status"] != "failed"
            or job["claim_token"] is not None
            or int(job["attempts"] or 0) != 1
        ):
            raise SlowGraphError(
                "zero-call configuration recovery requires one unclaimed failed job"
            )
        attempts = con.execute(
            "SELECT * FROM slow_graph_attempts WHERE job_id=? ORDER BY created_at,attempt_id",
            (job_id,),
        ).fetchall()
        patch_count = int(
            con.execute(
                "SELECT count(*) FROM slow_graph_patches WHERE job_id=?", (job_id,)
            ).fetchone()[0]
        )
        if len(attempts) != 1 or patch_count != 0:
            raise SlowGraphError(
                "zero-call configuration recovery requires one attempt and no patch"
            )
        attempt = attempts[0]
        raw_metadata = _required_text(
            attempt["call_metadata_json"], "failed call metadata"
        )
        metadata = _v3._strict_json(
            raw_metadata, label="failed call metadata", expected=dict
        )
        route = _clean(metadata.get("route"))
        error = _clean(attempt["error"])
        usage = metadata.get("usage")
        zero_usage = isinstance(usage, Mapping) and all(
            int(value or 0) == 0 for value in usage.values()
        )
        if (
            attempt["status"] != "failed"
            or error != ZERO_CALL_CONFIGURATION_ERRORS.get(route)
            or _clean(job["last_error"]) != error
            or metadata.get("physical_api_call") is not False
            or int(metadata.get("physical_api_calls", -1)) != 0
            or int(metadata.get("attempt_count", -1)) != 0
            or _clean(metadata.get("status")) != "unavailable"
            or _clean(metadata.get("physical_call_id"))
            or _clean(metadata.get("raw_response"))
            or _clean(metadata.get("content"))
            or not zero_usage
        ):
            raise SlowGraphError(
                "failed attempt is not a proven zero-call configuration failure"
            )
        error_sha256 = hashlib.sha256(error.encode("utf-8")).hexdigest()
        metadata_sha256 = hashlib.sha256(raw_metadata.encode("utf-8")).hexdigest()
        recovery_id = "sgz_" + _digest(
            {
                "job_id": job_id,
                "attempt_id": attempt["attempt_id"],
                "error_sha256": error_sha256,
                "call_metadata_sha256": metadata_sha256,
            }
        )[:32]
        created_at = _v3._now()
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS slow_graph_zero_call_recoveries(
                recovery_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL UNIQUE,
                attempt_id TEXT NOT NULL UNIQUE,
                scope_id TEXT NOT NULL,
                route TEXT NOT NULL,
                error_sha256 TEXT NOT NULL,
                call_metadata_sha256 TEXT NOT NULL,
                physical_api_calls INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        con.execute(
            "INSERT INTO slow_graph_zero_call_recoveries VALUES(?,?,?,?,?,?,?,?,?)",
            (
                recovery_id,
                job_id,
                attempt["attempt_id"],
                job["scope_id"],
                route,
                error_sha256,
                metadata_sha256,
                0,
                created_at,
            ),
        )
        reopened = con.execute(
            "UPDATE slow_graph_jobs SET status='pending',last_error='',updated_at=?,"
            "claim_token=NULL,claim_owner=NULL,lease_expires_at=NULL "
            "WHERE job_id=? AND status='failed' AND claim_token IS NULL",
            (created_at, job_id),
        )
        if reopened.rowcount != 1:
            raise SlowGraphError(
                "slow graph job changed during zero-call configuration recovery"
            )
    return {
        "schema_version": "tmcra.v4.slow-zero-call-recovery.1",
        "recovery_id": recovery_id,
        "job_id": job_id,
        "attempt_id": str(attempt["attempt_id"]),
        "scope_id": str(job["scope_id"]),
        "route": route,
        "error_sha256": error_sha256,
        "call_metadata_sha256": metadata_sha256,
        "physical_api_calls": 0,
        "status": "pending",
        "created_at": created_at,
    }


def resume_zero_call_promotion_failure(
    store: V4SlowGraphStore, job_id: str
) -> dict[str, Any]:
    """Reopen the reviewed no-op routing bug without hiding any model failure."""
    with store.connection() as con:
        con.execute("BEGIN IMMEDIATE")
        job = con.execute(
            "SELECT * FROM slow_graph_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if (
            job is None
            or job["status"] != "failed"
            or job["claim_token"] is not None
            or int(job["attempts"] or 0) != 1
        ):
            raise SlowGraphError(
                "zero-call promotion recovery requires one unclaimed failed job"
            )
        attempts = con.execute(
            "SELECT * FROM slow_graph_attempts WHERE job_id=? "
            "ORDER BY created_at,attempt_id",
            (job_id,),
        ).fetchall()
        patch_count = int(
            con.execute(
                "SELECT count(*) FROM slow_graph_patches WHERE job_id=?", (job_id,)
            ).fetchone()[0]
        )
        if len(attempts) != 1 or patch_count != 0:
            raise SlowGraphError(
                "zero-call promotion recovery requires one attempt and no patch"
            )
        attempt = attempts[0]
        raw_metadata = _required_text(
            attempt["call_metadata_json"], "failed call metadata"
        )
        metadata = _v3._strict_json(
            raw_metadata, label="failed call metadata", expected=dict
        )
        error = _clean(attempt["error"])
        eligible_ids = metadata.get("eligible_evidence_ids")
        challenged_ids = metadata.get("challenged_evidence_ids")
        delta_ids = metadata.get("delta_evidence_ids")
        usage = metadata.get("usage")
        zero_usage = isinstance(usage, Mapping) and all(
            int(value or 0) == 0 for value in usage.values()
        )
        if (
            attempt["status"] != "failed"
            or not error.startswith(
                "noop cannot consume uncited current durable Fast evidence: "
            )
            or _clean(job["last_error"]) != error
            or _clean(metadata.get("route")) != "deterministic_noop"
            or _clean(metadata.get("route_reason"))
            != "new capsule blocked by unresolved fast challenge"
            or metadata.get("physical_api_call") is not False
            or int(metadata.get("physical_api_calls", -1)) != 0
            or int(metadata.get("attempt_count", -1)) != 0
            or _clean(metadata.get("physical_call_id"))
            or _clean(metadata.get("raw_response"))
            or _clean(metadata.get("content"))
            or not zero_usage
            or not isinstance(eligible_ids, list)
            or not eligible_ids
            or not isinstance(challenged_ids, list)
            or not challenged_ids
            or not isinstance(delta_ids, list)
            or not set(eligible_ids) <= set(delta_ids)
        ):
            raise SlowGraphError(
                "failed attempt is not the proven zero-call promotion routing failure"
            )
        error_sha256 = hashlib.sha256(error.encode("utf-8")).hexdigest()
        metadata_sha256 = hashlib.sha256(raw_metadata.encode("utf-8")).hexdigest()
        recovery_id = "sgp_" + _digest(
            {
                "job_id": job_id,
                "attempt_id": attempt["attempt_id"],
                "error_sha256": error_sha256,
                "call_metadata_sha256": metadata_sha256,
            }
        )[:32]
        created_at = _v3._now()
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS slow_graph_zero_call_promotion_recoveries(
                recovery_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL UNIQUE,
                attempt_id TEXT NOT NULL UNIQUE,
                scope_id TEXT NOT NULL,
                error_sha256 TEXT NOT NULL,
                call_metadata_sha256 TEXT NOT NULL,
                physical_api_calls INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        con.execute(
            "INSERT INTO slow_graph_zero_call_promotion_recoveries "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                recovery_id,
                job_id,
                attempt["attempt_id"],
                job["scope_id"],
                error_sha256,
                metadata_sha256,
                0,
                created_at,
            ),
        )
        reopened = con.execute(
            "UPDATE slow_graph_jobs SET status='pending',last_error='',updated_at=?,"
            "claim_token=NULL,claim_owner=NULL,lease_expires_at=NULL "
            "WHERE job_id=? AND status='failed' AND claim_token IS NULL",
            (created_at, job_id),
        )
        if reopened.rowcount != 1:
            raise SlowGraphError(
                "slow graph job changed during zero-call promotion recovery"
            )
    return {
        "schema_version": "tmcra.v4.slow-zero-call-promotion-recovery.1",
        "recovery_id": recovery_id,
        "job_id": job_id,
        "attempt_id": str(attempt["attempt_id"]),
        "scope_id": str(job["scope_id"]),
        "error_sha256": error_sha256,
        "call_metadata_sha256": metadata_sha256,
        "physical_api_calls": 0,
        "status": "pending",
        "created_at": created_at,
    }


def resume_zero_call_projection_failure(
    store: V4SlowGraphStore, job_id: str
) -> dict[str, Any]:
    """Reopen one zero-call failure caused only by an empty internal origin field."""
    with store.connection() as con:
        con.execute("BEGIN IMMEDIATE")
        job = con.execute(
            "SELECT * FROM slow_graph_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if (
            job is None
            or job["status"] != "failed"
            or job["claim_token"] is not None
            or int(job["attempts"] or 0) != 1
        ):
            raise SlowGraphError(
                "zero-call projection recovery requires one unclaimed failed job"
            )
        attempts = con.execute(
            "SELECT * FROM slow_graph_attempts WHERE job_id=? ORDER BY created_at,attempt_id",
            (job_id,),
        ).fetchall()
        patch_count = int(
            con.execute(
                "SELECT count(*) FROM slow_graph_patches WHERE job_id=?", (job_id,)
            ).fetchone()[0]
        )
        if len(attempts) != 1 or patch_count != 0:
            raise SlowGraphError(
                "zero-call projection recovery requires one attempt and no patch"
            )
        attempt = attempts[0]
        raw_metadata = _required_text(
            attempt["call_metadata_json"], "failed call metadata"
        )
        metadata = _v3._strict_json(
            raw_metadata, label="failed call metadata", expected=dict
        )
        error = _clean(attempt["error"])
        if (
            attempt["status"] != "failed"
            or not re.fullmatch(
                r"benchmark field is forbidden in slow-graph request: "
                r"payload\.evidence\[\d+\]\.metadata\.origin_answer_ids",
                error,
            )
            or _clean(job["last_error"]) != error
            or metadata.get("physical_api_call") is not False
            or int(metadata.get("physical_api_calls", -1)) != 0
            or _clean(metadata.get("physical_call_id"))
            or _clean(metadata.get("raw_response"))
            or _clean(metadata.get("content"))
        ):
            raise SlowGraphError(
                "failed attempt is not a proven zero-call projection failure"
            )
        evidence_ids = _v3._strict_json(
            job["evidence_ids_json"], label="job evidence IDs", expected=list
        )
        evidence = store._evidence(con, job["scope_id"], evidence_ids)
        offending_paths: list[str] = []
        for index, leaf in enumerate(evidence):
            for key, value in _leaf_metadata(leaf).items():
                if not _forbidden_field(key):
                    continue
                if key != "origin_answer_ids" or value != []:
                    raise SlowGraphError(
                        "projection recovery found non-empty or unsupported benchmark metadata"
                    )
                offending_paths.append(
                    f"payload.evidence[{index}].metadata.origin_answer_ids"
                )
        if error.rsplit(": ", 1)[-1] not in offending_paths:
            raise SlowGraphError(
                "projection recovery error does not match current internal evidence"
            )
        public_region = {
            "region_key": _required_text(job["region_key"], "region key"),
            "evidence": [_public_leaf(item) for item in evidence],
        }
        _assert_no_benchmark_fields(public_region)
        error_sha256 = hashlib.sha256(error.encode("utf-8")).hexdigest()
        metadata_sha256 = hashlib.sha256(raw_metadata.encode("utf-8")).hexdigest()
        projection_sha256 = _digest(public_region)
        recovery_id = "sgp0_" + _digest(
            {
                "job_id": job_id,
                "attempt_id": attempt["attempt_id"],
                "error_sha256": error_sha256,
                "projection_sha256": projection_sha256,
            }
        )[:32]
        created_at = _v3._now()
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS slow_graph_zero_call_projection_recoveries(
                recovery_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL UNIQUE,
                attempt_id TEXT NOT NULL UNIQUE,
                scope_id TEXT NOT NULL,
                offending_paths_json TEXT NOT NULL,
                error_sha256 TEXT NOT NULL,
                call_metadata_sha256 TEXT NOT NULL,
                public_projection_sha256 TEXT NOT NULL,
                physical_api_calls INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        con.execute(
            "INSERT INTO slow_graph_zero_call_projection_recoveries "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                recovery_id,
                job_id,
                attempt["attempt_id"],
                job["scope_id"],
                _json(offending_paths),
                error_sha256,
                metadata_sha256,
                projection_sha256,
                0,
                created_at,
            ),
        )
        reopened = con.execute(
            "UPDATE slow_graph_jobs SET status='pending',last_error='',updated_at=?,"
            "claim_token=NULL,claim_owner=NULL,lease_expires_at=NULL "
            "WHERE job_id=? AND status='failed' AND claim_token IS NULL",
            (created_at, job_id),
        )
        if reopened.rowcount != 1:
            raise SlowGraphError(
                "slow graph job changed during zero-call projection recovery"
            )
    return {
        "schema_version": "tmcra.v4.slow-zero-call-projection-recovery.1",
        "recovery_id": recovery_id,
        "job_id": job_id,
        "attempt_id": str(attempt["attempt_id"]),
        "scope_id": str(job["scope_id"]),
        "offending_paths": offending_paths,
        "public_projection_sha256": projection_sha256,
        "physical_api_calls": 0,
        "status": "pending",
        "created_at": created_at,
    }


class TieredGraphPatchManager:
    """Controller-owned route selection for V4 slow graph jobs."""

    def __init__(
        self,
        *,
        flash_config: DeepSeekTierConfig | None = None,
        pro_config: DeepSeekTierConfig | None = None,
        flash: Any | None = None,
        pro: Any | None = None,
    ) -> None:
        self.flash = flash or (_DeepSeekTierClient(flash_config, route="flash") if flash_config else None)
        self.pro = pro or (_DeepSeekTierClient(pro_config, route="pro") if pro_config else None)
        self.model_config = {
            "model": "deepseek-v4-tiered-slow-graph",
            "temperature": 0,
            "route_policy": (
                "deterministic-create-noop/flash-incremental/"
                "pro-initial-partition-conflict"
            ),
            "prompt_version": SLOW_PROMPT_VERSION,
        }
        self.prompt_hash = _digest(
            {
                "schema": SCHEMA_VERSION,
                "policy": self.model_config["route_policy"],
                "prompt_version": SLOW_PROMPT_VERSION,
            }
        )
        self.last_call_metadata: Mapping[str, Any] = {}

    @classmethod
    def from_env(cls) -> "TieredGraphPatchManager":
        return cls(
            flash_config=_optional_config("TMCRA_DEEPSEEK_FLASH", "deepseek-v4-flash"),
            pro_config=_optional_config("TMCRA_DEEPSEEK_PRO", "deepseek-v4-pro"),
        )

    @staticmethod
    def _capsule_claims(
        capsules: list[Mapping[str, Any]],
    ) -> tuple[set[str], dict[str, list[Mapping[str, Any]]], set[str]]:
        cited: set[str] = set()
        by_slot: dict[str, list[Mapping[str, Any]]] = {}
        statuses: set[str] = set()
        for capsule in capsules:
            raw_claims = capsule.get("claims")
            if not isinstance(raw_claims, list):
                raise EvidencePolicyError("capsule claims are not auditable")
            status = _clean(capsule.get("status") or "active").casefold()
            statuses.add(status)
            if status not in {"active", "challenged"}:
                continue
            for claim in raw_claims:
                if not isinstance(claim, Mapping):
                    raise EvidencePolicyError("capsule claim is not an object")
                slot = _required_text(claim.get("canonical_slot"), "capsule claim canonical slot")
                support = claim.get("support", [])
                counter = claim.get("counterevidence", [])
                if not isinstance(support, list) or not isinstance(counter, list):
                    raise EvidencePolicyError("capsule claim evidence is not a list")
                cited.update(_clean(item) for item in [*support, *counter] if _clean(item))
                by_slot.setdefault(slot, []).append(claim)
        return cited, by_slot, statuses

    @staticmethod
    def _base_metadata(
        route: str,
        *,
        reason: str,
        evidence: list[Mapping[str, Any]],
        eligible: list[Mapping[str, Any]],
        challenged: list[Mapping[str, Any]],
        uncertain: list[str],
        episodic: list[str],
        inactive: list[str],
        delta: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "route": route,
            "route_reason": reason,
            "summary_contract_version": SLOW_SUMMARY_CONTRACT_VERSION,
            "evidence_binding_contract_version": (
                SLOW_EVIDENCE_BINDING_CONTRACT_VERSION
            ),
            "physical_api_call": False,
            "physical_api_calls": 0,
            "attempt_count": 0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cache_read_input_tokens": 0, "cache_hit_tokens": 0, "cache_miss_tokens": 0, "total_tokens": 0},
            "cost_audit": {"estimated_cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "cache_read_input_tokens": 0, "cache_hit_tokens": 0, "cache_miss_tokens": 0},
            "eligible_evidence_ids": [_leaf_id(item) for item in eligible],
            "challenged_evidence_ids": [_leaf_id(item) for item in challenged],
            "delta_evidence_ids": [_leaf_id(item) for item in delta],
            "uncertain_evidence_ids": uncertain,
            "episodic_evidence_ids": episodic,
            "inactive_evidence_ids": inactive,
            "ignored_evidence_ids": sorted(set(uncertain + episodic + inactive)),
            "supplied_evidence_count": len(evidence),
        }

    @staticmethod
    def _create_patch(leaf: Mapping[str, Any]) -> dict[str, Any]:
        text = _leaf_text(leaf)
        return _materialize_lossless_summaries({
            "operations": [
                {
                    "action": "create",
                    "capsule_key": _capsule_key_from_slot(_leaf_slot(leaf)),
                    "claims": [
                        {
                            "canonical_slot": _leaf_slot(leaf),
                            "text": text,
                            "support": [_leaf_id(leaf)],
                            "counterevidence": [],
                        }
                    ],
                }
            ]
        })

    @staticmethod
    def _noop_patch(capsules: list[Mapping[str, Any]]) -> dict[str, Any]:
        operation: dict[str, Any] = {"action": "noop"}
        if capsules:
            capsule_id = _clean(capsules[0].get("capsule_id"))
            if capsule_id:
                operation["capsule_id"] = capsule_id
        return {"operations": [operation]}

    @staticmethod
    def _public_request(
        route: str,
        region: Mapping[str, Any],
        required_evidence_ids: set[str],
        *,
        partition_capsule_ids: set[str] | None = None,
        semantic_partition_mode: str | None = None,
    ) -> dict[str, Any]:
        selected_evidence = [
            item
            for item in region.get("evidence", [])
            if isinstance(item, Mapping)
            and (
                _leaf_id(item) in required_evidence_ids
                or (
                    route == "pro"
                    and (
                        _is_current_durable(item)
                        or _is_challenged_durable(item)
                    )
                )
            )
        ]
        public_region = {
            "region_key": _required_text(region.get("region_key"), "region key"),
            "evidence": [_public_leaf(item) for item in selected_evidence],
            "required_evidence_ids": sorted(required_evidence_ids),
        }
        partition_ids = sorted(partition_capsule_ids or ())
        partition_mode = _clean(semantic_partition_mode)
        if partition_mode:
            if partition_mode not in {"manage", "migrate"}:
                raise EvidencePolicyError("semantic partition mode is invalid")
            public_region["semantic_partition_required"] = True
            public_region["semantic_partition_mode"] = partition_mode
            if partition_mode == "migrate":
                if not partition_ids:
                    raise EvidencePolicyError(
                        "semantic partition migration requires capsule targets"
                    )
                public_region["partition_capsule_ids"] = partition_ids
            elif partition_ids:
                raise EvidencePolicyError(
                    "generic semantic management cannot name partition targets"
                )
        elif partition_ids:
            raise EvidencePolicyError(
                "partition capsule targets require semantic partition mode"
            )
        _assert_no_benchmark_fields(public_region)
        return public_region

    @staticmethod
    def _completed_client_metadata(client: Any, route: str) -> dict[str, Any]:
        raw = dict(getattr(client, "last_call_metadata", {}) or {})
        physical_calls = int(raw.get("physical_api_calls", 1) or 0)
        if physical_calls < 1:
            physical_calls = 1
        return {
            **raw,
            "route": route,
            "physical_api_call": True,
            "physical_api_calls": physical_calls,
            "attempt_count": int(raw.get("attempt_count", physical_calls) or physical_calls),
        }

    @staticmethod
    def _aggregate_escalation_metadata(
        metadata: Mapping[str, Any],
        flash_metadata: Mapping[str, Any],
        pro_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        tier_calls = [dict(flash_metadata)]
        if pro_metadata is not None:
            tier_calls.append(dict(pro_metadata))
        usage_keys = (
            "prompt_tokens",
            "completion_tokens",
            "cache_read_input_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
            "total_tokens",
        )
        usage = {key: 0 for key in usage_keys}
        estimated_cost = 0.0
        physical_calls = 0
        attempt_count = 0
        latency_ms = 0.0
        for call in tier_calls:
            physical_calls += int(call.get("physical_api_calls", 0) or 0)
            attempt_count += int(call.get("attempt_count", 0) or 0)
            call_usage = call.get("usage")
            if isinstance(call_usage, Mapping):
                for key in usage_keys:
                    usage[key] += int(call_usage.get(key, 0) or 0)
            call_cost = call.get("cost_audit")
            if isinstance(call_cost, Mapping):
                estimated_cost += float(call_cost.get("estimated_cost", 0.0) or 0.0)
            try:
                latency_ms += float(call.get("latency_ms", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
        final_call = dict(pro_metadata or flash_metadata)
        result = {
            **dict(metadata),
            **final_call,
            "route": "flash_to_pro",
            "route_reason": f"flash_escalation:{FLASH_ESCALATION_REASON}",
            "initial_route_reason": metadata.get("route_reason"),
            "escalation_requested": True,
            "escalation_reason": FLASH_ESCALATION_REASON,
            "physical_api_call": physical_calls > 0,
            "physical_api_calls": physical_calls,
            "attempt_count": attempt_count,
            "usage": usage,
            "cost_audit": {
                **usage,
                "estimated_cost": estimated_cost,
            },
            "tier_calls": tier_calls,
        }
        if latency_ms:
            result["latency_ms"] = round(latency_ms, 3)
        if flash_metadata.get("started_at") is not None:
            result["started_at"] = flash_metadata.get("started_at")
        return result

    @staticmethod
    def _aggregate_semantic_correction_metadata(
        metadata: Mapping[str, Any],
        tier_calls: list[Mapping[str, Any]],
        *,
        route: str,
        route_reason: str,
        validation_error: str,
    ) -> dict[str, Any]:
        if len(tier_calls) < 2:
            raise SlowGraphError("semantic correction metadata requires two calls")
        usage_keys = (
            "prompt_tokens",
            "completion_tokens",
            "cache_read_input_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
            "total_tokens",
        )
        usage = {key: 0 for key in usage_keys}
        estimated_cost = 0.0
        physical_calls = 0
        attempt_count = 0
        latency_ms = 0.0
        labeled_calls: list[dict[str, Any]] = []
        for index, raw_call in enumerate(tier_calls):
            call = dict(raw_call)
            if index == len(tier_calls) - 1:
                call["tier_stage"] = "semantic_correction"
            elif route == "flash_to_pro" and index == 0:
                call["tier_stage"] = "initial_flash"
            else:
                call["tier_stage"] = "initial_pro"
            labeled_calls.append(call)
            physical_calls += int(call.get("physical_api_calls", 0) or 0)
            attempt_count += int(call.get("attempt_count", 0) or 0)
            call_usage = call.get("usage")
            if isinstance(call_usage, Mapping):
                for key in usage_keys:
                    usage[key] += int(call_usage.get(key, 0) or 0)
            call_cost = call.get("cost_audit")
            if isinstance(call_cost, Mapping):
                estimated_cost += float(call_cost.get("estimated_cost", 0.0) or 0.0)
            try:
                latency_ms += float(call.get("latency_ms", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
        final_call = dict(labeled_calls[-1])
        result = {
            **dict(metadata),
            **final_call,
            "route": route,
            "route_reason": route_reason,
            "semantic_correction_attempted": True,
            "semantic_correction_validation_error": validation_error,
            "physical_api_call": physical_calls > 0,
            "physical_api_calls": physical_calls,
            "attempt_count": attempt_count,
            "usage": usage,
            "cost_audit": {**usage, "estimated_cost": estimated_cost},
            "tier_calls": labeled_calls,
        }
        if latency_ms:
            result["latency_ms"] = round(latency_ms, 3)
        first_started = labeled_calls[0].get("started_at")
        if first_started is not None:
            result["started_at"] = first_started
        return result

    def _validate_and_materialize_patch(
        self,
        *,
        route: str,
        patch: Mapping[str, Any],
        region: Mapping[str, Any],
        capsules: list[Mapping[str, Any]],
        required_evidence_ids: set[str],
        partition_capsule_ids: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        validate_patch(patch)
        _validate_generic_create_partition_keys(region.get("region_key"), patch)
        self._validate_route_actions(
            route,
            patch,
            capsules,
            required_evidence_ids=required_evidence_ids,
            partition_capsule_ids=partition_capsule_ids,
        )
        _validate_claim_evidence_contract(region, capsules, patch, route=route)
        merge_audit: dict[str, Any] | None = None
        if route == "flash" and required_evidence_ids:
            _validate_flash_delta_patch(patch, capsules, required_evidence_ids)
            if capsules:
                committed_patch, merge_audit = _merge_flash_delta_patch(
                    patch, capsules
                )
            else:
                committed_patch = _materialize_lossless_summaries(patch)
        else:
            committed_patch = _materialize_lossless_summaries(patch)
        validate_patch(committed_patch, require_lossless_summary=True)
        _validate_generic_create_partition_keys(
            region.get("region_key"), committed_patch
        )
        self._validate_route_actions(
            route,
            committed_patch,
            capsules,
            required_evidence_ids=required_evidence_ids,
            partition_capsule_ids=partition_capsule_ids,
        )
        _validate_claim_evidence_contract(
            region, capsules, committed_patch, route=route
        )
        _validate_promotion_patch(
            region,
            capsules,
            committed_patch,
            required_evidence_ids=required_evidence_ids,
        )
        return committed_patch, merge_audit

    def _validate_pro_with_optional_correction(
        self,
        *,
        client: Any,
        public_region: Mapping[str, Any],
        public_capsules: list[Mapping[str, Any]],
        region: Mapping[str, Any],
        capsules: list[Mapping[str, Any]],
        patch: Mapping[str, Any],
        metadata: Mapping[str, Any],
        tier_calls: list[Mapping[str, Any]],
        route: str,
        route_reason: str,
        required_evidence_ids: set[str],
        partition_capsule_ids: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any] | None, Mapping[str, Any]]:
        try:
            committed, merge_audit = self._validate_and_materialize_patch(
                route="pro",
                patch=patch,
                region=region,
                capsules=capsules,
                required_evidence_ids=required_evidence_ids,
                partition_capsule_ids=partition_capsule_ids,
            )
            return committed, merge_audit, patch
        except PatchValidationError as initial_error:
            correct = getattr(client, "correct", None)
            if not callable(correct):
                raise
            initial_error_text = str(initial_error)

        previous_call_id = _clean(
            dict(getattr(client, "last_call_metadata", {}) or {}).get(
                "physical_call_id"
            )
        )
        try:
            corrected_patch = correct(
                public_region,
                public_capsules,
                rejected_patch=patch,
                validation_error=initial_error_text,
            )
        except Exception:
            raw_correction = dict(
                getattr(client, "last_call_metadata", {}) or {}
            )
            correction_call_id = _clean(raw_correction.get("physical_call_id"))
            if correction_call_id and correction_call_id != previous_call_id:
                correction_metadata = self._completed_client_metadata(client, "pro")
                self.last_call_metadata = {
                    **self._aggregate_semantic_correction_metadata(
                        metadata,
                        [*tier_calls, correction_metadata],
                        route=route,
                        route_reason=route_reason,
                        validation_error=initial_error_text,
                    ),
                    "status": "semantic_correction_call_failed",
                    "semantic_correction_applied": False,
                }
            else:
                self.last_call_metadata = {
                    **dict(self.last_call_metadata),
                    "status": "semantic_correction_preflight_failed",
                    "semantic_correction_attempted": True,
                    "semantic_correction_applied": False,
                    "semantic_correction_validation_error": initial_error_text,
                }
            raise
        correction_metadata = self._completed_client_metadata(client, "pro")
        corrected_patch, correction_normalizations = _normalize_transport_patch(
            corrected_patch, public_capsules, public_region
        )
        if correction_normalizations:
            correction_metadata = {
                **correction_metadata,
                "controller_transport_normalizations": correction_normalizations,
            }
        self.last_call_metadata = self._aggregate_semantic_correction_metadata(
            metadata,
            [*tier_calls, correction_metadata],
            route=route,
            route_reason=route_reason,
            validation_error=initial_error_text,
        )
        try:
            committed, merge_audit = self._validate_and_materialize_patch(
                route="pro",
                patch=corrected_patch,
                region=region,
                capsules=capsules,
                required_evidence_ids=required_evidence_ids,
                partition_capsule_ids=partition_capsule_ids,
            )
        except PatchValidationError as correction_error:
            self.last_call_metadata = {
                **dict(self.last_call_metadata),
                "status": "semantic_correction_rejected",
                "semantic_correction_applied": False,
                "semantic_correction_rejection_error": str(correction_error),
                "initial_rejected_patch_sha256": _digest(patch),
                "corrected_patch_sha256": _digest(corrected_patch),
            }
            raise
        self.last_call_metadata = {
            **dict(self.last_call_metadata),
            "status": "completed",
            "semantic_correction_applied": True,
            "initial_rejected_patch_sha256": _digest(patch),
            "corrected_patch_sha256": _digest(corrected_patch),
        }
        return committed, merge_audit, corrected_patch

    def _invoke(
        self,
        route: str,
        reason: str,
        client: Any,
        region: Mapping[str, Any],
        capsules: list[Mapping[str, Any]],
        metadata: dict[str, Any],
    ) -> Mapping[str, Any]:
        if client is None:
            self.last_call_metadata = {**metadata, "route": route, "route_reason": reason, "status": "unavailable"}
            raise TieredAPIError(f"{route} client is not configured; no fallback is allowed")
        delta_ids = set(metadata["delta_evidence_ids"])
        required_ids = set(
            metadata.get("required_operation_evidence_ids") or delta_ids
        )
        partition_capsule_ids = set(
            metadata.get("semantic_partition_capsule_ids") or ()
        )
        semantic_partition_mode = _clean(
            metadata.get("semantic_partition_mode")
        ) or None
        public_region = self._public_request(
            route,
            region,
            required_ids,
            partition_capsule_ids=partition_capsule_ids,
            semantic_partition_mode=semantic_partition_mode,
        )
        public_capsules = [_public_capsule(capsule) for capsule in capsules]
        _assert_no_benchmark_fields(public_capsules)
        try:
            patch = client.propose(public_region, public_capsules)
        except Exception:
            client_metadata = getattr(client, "last_call_metadata", {})
            self.last_call_metadata = {**metadata, **dict(client_metadata), "route": route, "route_reason": reason}
            raise
        client_metadata = self._completed_client_metadata(client, route)
        patch, controller_transport_normalizations = _normalize_transport_patch(
            patch, public_capsules, public_region
        )
        if controller_transport_normalizations:
            client_metadata = {
                **client_metadata,
                "controller_transport_normalizations": controller_transport_normalizations,
            }

        if route == "flash" and _flash_escalation_patch(patch):
            if self.pro is None:
                self.last_call_metadata = {
                    **self._aggregate_escalation_metadata(
                        metadata, client_metadata, None
                    ),
                    "status": "unavailable",
                }
                raise TieredAPIError(
                    "pro client is not configured after explicit Flash escalation; "
                    "no fallback is allowed"
                )
            pro_region = self._public_request(
                "pro",
                region,
                required_ids,
                partition_capsule_ids=partition_capsule_ids,
                semantic_partition_mode=semantic_partition_mode,
            )
            try:
                pro_patch = self.pro.propose(pro_region, public_capsules)
            except Exception:
                pro_metadata = self._completed_client_metadata(self.pro, "pro")
                self.last_call_metadata = self._aggregate_escalation_metadata(
                    metadata, client_metadata, pro_metadata
                )
                raise
            pro_metadata = self._completed_client_metadata(self.pro, "pro")
            pro_patch, pro_transport_normalizations = _normalize_transport_patch(
                pro_patch, public_capsules, pro_region
            )
            if pro_transport_normalizations:
                pro_metadata = {
                    **pro_metadata,
                    "controller_transport_normalizations": pro_transport_normalizations,
                }
            self.last_call_metadata = self._aggregate_escalation_metadata(
                metadata, client_metadata, pro_metadata
            )
            escalation_reason = f"flash_escalation:{FLASH_ESCALATION_REASON}"
            committed_patch, merge_audit, model_patch = (
                self._validate_pro_with_optional_correction(
                    client=self.pro,
                    public_region=pro_region,
                    public_capsules=public_capsules,
                    region=region,
                    capsules=capsules,
                    patch=pro_patch,
                    metadata=metadata,
                    tier_calls=[client_metadata, pro_metadata],
                    route="flash_to_pro",
                    route_reason=escalation_reason,
                    required_evidence_ids=required_ids,
                    partition_capsule_ids=partition_capsule_ids,
                )
            )
            self.last_call_metadata = {
                **dict(self.last_call_metadata),
                "controller_summary_materialization": {
                    "schema_version": SLOW_SUMMARY_CONTRACT_VERSION,
                    "model_patch_sha256": _digest(model_patch),
                    "committed_patch_sha256": _digest(committed_patch),
                },
            }
            if merge_audit is not None:
                self.last_call_metadata = {
                    **dict(self.last_call_metadata),
                    "controller_delta_merge": merge_audit,
                }
            return committed_patch

        self.last_call_metadata = {
            **metadata,
            **client_metadata,
            "route": route,
            "route_reason": reason,
        }
        if route == "pro":
            committed_patch, merge_audit, model_patch = (
                self._validate_pro_with_optional_correction(
                    client=client,
                    public_region=public_region,
                    public_capsules=public_capsules,
                    region=region,
                    capsules=capsules,
                    patch=patch,
                    metadata=metadata,
                    tier_calls=[client_metadata],
                    route="pro",
                    route_reason=reason,
                    required_evidence_ids=required_ids,
                    partition_capsule_ids=partition_capsule_ids,
                )
            )
        else:
            committed_patch, merge_audit = self._validate_and_materialize_patch(
                route=route,
                patch=patch,
                region=region,
                capsules=capsules,
                required_evidence_ids=required_ids,
                partition_capsule_ids=partition_capsule_ids,
            )
            model_patch = patch
        summary_audit = {
            "schema_version": SLOW_SUMMARY_CONTRACT_VERSION,
            "model_patch_sha256": _digest(model_patch),
            "committed_patch_sha256": _digest(committed_patch),
        }
        if merge_audit is not None:
            self.last_call_metadata = {
                **dict(self.last_call_metadata),
                "controller_delta_merge": merge_audit,
                "controller_summary_materialization": summary_audit,
            }
        else:
            self.last_call_metadata = {
                **dict(self.last_call_metadata),
                "controller_summary_materialization": summary_audit,
            }
        return committed_patch

    @staticmethod
    def _validate_route_actions(
        route: str,
        patch: Mapping[str, Any],
        capsules: list[Mapping[str, Any]],
        *,
        required_evidence_ids: set[str] | None = None,
        partition_capsule_ids: set[str] | None = None,
    ) -> None:
        required = bool(required_evidence_ids)
        existing = {
            _required_text(capsule.get("capsule_id"), "capsule_id"): capsule
            for capsule in capsules
        }
        existing_keys = {
            _clean(capsule.get("capsule_key")).casefold()
            for capsule in capsules
            if _clean(capsule.get("capsule_key"))
        }
        allowed = (
            {"revise", "create"}
            if route == "flash"
            else {"revise", "create", "challenge", "resolve_challenge", "retire"}
        )
        if not required:
            allowed.add("noop")
        operated_capsules: set[str] = set()
        for operation in patch["operations"]:
            action = operation["action"]
            if action not in allowed:
                raise PatchValidationError(
                    f"{route} route does not allow action {action!r} for {'existing' if capsules else 'new'} capsule"
                )
            if action == "create":
                capsule_key = _normalize_capsule_key(operation.get("capsule_key"))
                if capsule_key in existing_keys:
                    raise PatchValidationError(
                        f"create capsule_key already exists in this region: {capsule_key}"
                    )
                continue
            if action == "noop":
                capsule_id = _clean(operation.get("capsule_id"))
                if capsule_id and capsule_id not in existing:
                    raise PatchValidationError("noop targeted an unknown capsule")
                continue
            capsule_id = _required_text(operation.get("capsule_id"), "capsule_id")
            capsule = existing.get(capsule_id)
            if capsule is None:
                raise PatchValidationError(
                    f"{route} route targeted an unknown capsule: {capsule_id}"
                )
            if operation.get("base_revision") != capsule.get("revision"):
                raise PatchValidationError(
                    f"{route} base_revision is stale for capsule {capsule_id}"
                )
            operated_capsules.add(capsule_id)
        missing_partition_targets = set(partition_capsule_ids or ()) - operated_capsules
        if missing_partition_targets:
            raise PatchValidationError(
                "semantic partition left required legacy capsules untouched: "
                + _json(sorted(missing_partition_targets))
            )

    def propose(self, region: Mapping[str, Any], capsules: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        if not isinstance(region, Mapping) or not isinstance(capsules, list):
            raise EvidencePolicyError("slow-graph region and capsules have invalid schemas")
        if set(region) != {"region_key", "evidence"}:
            raise EvidencePolicyError("slow-graph internal region envelope is not exact")
        evidence = [item for item in region.get("evidence", []) if isinstance(item, Mapping)]
        evidence.sort(key=lambda item: _leaf_id(item))
        _assert_no_benchmark_fields(
            {
                "region_key": _required_text(region.get("region_key"), "region key"),
                "evidence": [_public_leaf(item) for item in evidence],
            }
        )
        _assert_no_benchmark_fields(capsules)
        eligible = [item for item in evidence if _is_current_durable(item)]
        challenged = [item for item in evidence if _is_challenged_durable(item)]
        uncertain = sorted(_leaf_id(item) for item in evidence if _is_uncertain(item))
        episodic = sorted(_leaf_id(item) for item in evidence if _is_episodic(item))
        current_support_ids = {_leaf_id(item) for item in eligible}
        challenged_ids = {_leaf_id(item) for item in challenged}
        active_ids = current_support_ids | challenged_ids
        inactive = sorted(
            _leaf_id(item)
            for item in evidence
            if _leaf_id(item) not in active_ids
            and _leaf_id(item) not in set(uncertain)
            and _leaf_id(item) not in set(episodic)
        )
        original_capsules = capsules
        capsules, support_cleanup = _sanitize_capsules_for_current_support(
            original_capsules,
            current_support_ids,
            challenged_ids,
            {_leaf_id(item) for item in evidence},
        )
        cited, claims_by_slot, capsule_statuses = self._capsule_claims(capsules)
        delta = [item for item in [*eligible, *challenged] if _leaf_id(item) not in cited]
        partition_targets = _semantic_partition_targets(
            region.get("region_key"), capsules
        )
        generic_semantic_management = bool(delta) and (
            not partition_targets
            and _generic_region_requires_semantic_management(
                region.get("region_key"), [*eligible, *challenged], capsules
            )
        )
        uncited_current_support = [
            item for item in eligible if _leaf_id(item) not in cited
        ]
        if (
            support_cleanup["changed"]
            and not uncited_current_support
            and not partition_targets
        ):
            reason = "existing Slow claims lost all current Fast support"
            self.last_call_metadata = {
                **self._base_metadata(
                    "deterministic_support_cleanup",
                    reason=reason,
                    evidence=evidence,
                    eligible=eligible,
                    challenged=challenged,
                    uncertain=uncertain,
                    episodic=episodic,
                    inactive=inactive,
                    delta=delta,
                ),
                "support_cleanup": support_cleanup,
            }
            patch = _deterministic_support_cleanup_patch(
                original_capsules, capsules
            )
            validate_patch(patch)
            _validate_patch_summary_contract(patch)
            _validate_claim_evidence_contract(region, capsules, patch)
            _validate_promotion_patch(region, capsules, patch)
            return patch
        if not delta and partition_targets:
            if not _partition_targets_require_model(capsules, partition_targets):
                reason = "unambiguous_single_claim_partition_migration"
                self.last_call_metadata = {
                    **self._base_metadata(
                        "deterministic_contract_migration",
                        reason=reason,
                        evidence=evidence,
                        eligible=eligible,
                        challenged=challenged,
                        uncertain=uncertain,
                        episodic=episodic,
                        inactive=inactive,
                        delta=delta,
                    ),
                    "semantic_partition_contract_version": SLOW_PARTITION_CONTRACT_VERSION,
                    "semantic_partition_capsule_ids": sorted(partition_targets),
                    "semantic_partition_mode": "migrate",
                }
                if support_cleanup["changed"]:
                    self.last_call_metadata["support_cleanup"] = support_cleanup
                patch = _deterministic_contract_migration_patch(
                    capsules, partition_targets
                )
                validate_patch(patch, require_lossless_summary=True)
                _validate_claim_evidence_contract(region, capsules, patch)
                _validate_promotion_patch(region, capsules, patch)
                return patch
            reason = "semantic_partition_migration"
            required_operation_ids = {
                _leaf_id(item) for item in [*eligible, *challenged]
            }
            metadata = self._base_metadata(
                "pro",
                reason=reason,
                evidence=evidence,
                eligible=eligible,
                challenged=challenged,
                uncertain=uncertain,
                episodic=episodic,
                inactive=inactive,
                delta=delta,
            )
            metadata.update(
                {
                    "semantic_partition_contract_version": SLOW_PARTITION_CONTRACT_VERSION,
                    "semantic_partition_capsule_ids": sorted(partition_targets),
                    "semantic_partition_mode": "migrate",
                    "required_operation_evidence_ids": sorted(
                        required_operation_ids
                    ),
                }
            )
            if support_cleanup["changed"]:
                metadata["support_cleanup"] = support_cleanup
            return self._invoke(
                "pro", reason, self.pro, region, capsules, metadata
            )
        if not delta:
            if _capsule_requires_summary_migration(capsules):
                reason = "stored Slow summary violates the semantic summary contract"
                self.last_call_metadata = self._base_metadata(
                    "deterministic_summary_migration",
                    reason=reason,
                    evidence=evidence,
                    eligible=eligible,
                    challenged=challenged,
                    uncertain=uncertain,
                    episodic=episodic,
                    inactive=inactive,
                    delta=delta,
                )
                patch = _deterministic_summary_migration_patch(capsules)
                validate_patch(patch)
                _validate_patch_summary_contract(patch)
                _validate_claim_evidence_contract(region, capsules, patch)
                _validate_promotion_patch(region, capsules, patch)
                return patch
            reason = "no new eligible durable evidence"
            self.last_call_metadata = self._base_metadata("deterministic_noop", reason=reason, evidence=evidence, eligible=eligible, challenged=challenged, uncertain=uncertain, episodic=episodic, inactive=inactive, delta=delta)
            patch = self._noop_patch(capsules)
            validate_patch(patch)
            _validate_promotion_patch(region, capsules, patch)
            return patch

        if not capsules and challenged and not eligible:
            reason = "new capsule blocked by unresolved fast challenge"
            self.last_call_metadata = self._base_metadata("deterministic_noop", reason=reason, evidence=evidence, eligible=eligible, challenged=challenged, uncertain=uncertain, episodic=episodic, inactive=inactive, delta=delta)
            patch = self._noop_patch(capsules)
            validate_patch(patch)
            _validate_promotion_patch(region, capsules, patch)
            return patch

        reasons: list[str] = []
        delta_texts_by_slot: dict[str, set[str]] = {}
        for leaf in delta:
            delta_texts_by_slot.setdefault(_leaf_slot(leaf), set()).add(
                _normal_text(_leaf_text(leaf))
            )
        if not capsules and len(delta_texts_by_slot) > 1:
            reasons.append("initial_multi_slot_semantic_partition")
        if any(len(texts) > 1 for texts in delta_texts_by_slot.values()):
            reasons.append("same_slot_distinct_support_semantics")
        for leaf in delta:
            slot = _leaf_slot(leaf)
            if _is_challenged_durable(leaf):
                reasons.append("unresolved_fast_challenge")
            if _is_counterevidence(leaf):
                reasons.append("counterevidence")
            if slot in claims_by_slot:
                leaf_value = _normal_text(_leaf_text(leaf))
                if not any(_normal_text(claim.get("text")) == leaf_value for claim in claims_by_slot[slot]):
                    reasons.append("same_slot_correction")
        if capsule_statuses & {"challenged", "quarantined"}:
            reasons.append("unresolved_challenge")
        if partition_targets:
            reasons.append("semantic_partition_migration")
        if generic_semantic_management:
            reasons.append("generic_region_semantic_management")
        reason = "+".join(sorted(set(reasons)))
        metadata = self._base_metadata("pro" if reason else "flash", reason=reason or "compatible_consolidation", evidence=evidence, eligible=eligible, challenged=challenged, uncertain=uncertain, episodic=episodic, inactive=inactive, delta=delta)
        required_operation_ids = {
            _leaf_id(item) for item in [*eligible, *challenged]
        }
        if not capsules and eligible and challenged:
            # A challenged sibling is context, not a reason to drop an unrelated
            # authoritative fact or promote the unresolved sibling as active.
            required_operation_ids = {_leaf_id(item) for item in eligible}
            metadata["required_operation_evidence_ids"] = sorted(
                required_operation_ids
            )
        if support_cleanup["changed"]:
            metadata["support_cleanup"] = support_cleanup
        if partition_targets:
            metadata.update(
                {
                    "semantic_partition_contract_version": SLOW_PARTITION_CONTRACT_VERSION,
                    "semantic_partition_capsule_ids": sorted(partition_targets),
                    "semantic_partition_mode": "migrate",
                    "required_operation_evidence_ids": sorted(
                        required_operation_ids
                    ),
                }
            )
        elif generic_semantic_management or (
            not capsules and "initial_multi_slot_semantic_partition" in reasons
        ):
            metadata.update(
                {
                    "semantic_partition_contract_version": SLOW_PARTITION_CONTRACT_VERSION,
                    "semantic_partition_mode": "manage",
                    "required_operation_evidence_ids": sorted(
                        required_operation_ids
                    ),
                }
            )
        if not capsules and len(delta) == 1 and not reasons:
            self.last_call_metadata = self._base_metadata("deterministic_create", reason="one current durable leaf", evidence=evidence, eligible=eligible, challenged=challenged, uncertain=uncertain, episodic=episodic, inactive=inactive, delta=delta)
            patch = self._create_patch(delta[0])
            validate_patch(patch)
            _validate_patch_summary_contract(patch)
            _validate_promotion_patch(region, capsules, patch)
            return patch
        if reasons:
            return self._invoke("pro", reason, self.pro, region, capsules, metadata)
        return self._invoke("flash", "compatible_consolidation", self.flash, region, capsules, metadata)


class DeepSeekFlashGraphPatchManager(_DeepSeekTierClient):
    def __init__(self, config: DeepSeekTierConfig) -> None:
        super().__init__(config, route="flash")


class DeepSeekProGraphPatchManager(_DeepSeekTierClient):
    def __init__(self, config: DeepSeekTierConfig | DeepSeekProConfig) -> None:
        super().__init__(config, route="pro")


TieredSlowGraphPatchManager = TieredGraphPatchManager
DeepSeekTieredGraphPatchManager = TieredGraphPatchManager
TieredPatchManager = TieredGraphPatchManager
V4SlowGraphPatchManager = TieredGraphPatchManager
SlowGraphStore = V4SlowGraphStore


def main() -> None:
    parser = argparse.ArgumentParser(description="TMCRA V4 tiered slow graph controller")
    parser.add_argument("database", type=Path)
    parser.add_argument("--repo", type=Path, required=True, help="TMCRA repository containing the real graph schema")
    sub = parser.add_subparsers(dest="command", required=True)
    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("scope_id")
    enqueue.add_argument("--region")
    drain = sub.add_parser("drain")
    drain.add_argument("--batch-size", type=int)
    run = sub.add_parser("run")
    run.add_argument("job_id")
    revalidate = sub.add_parser("revalidate-failed")
    revalidate.add_argument("job_id")
    semantic_policy_revalidation = sub.add_parser(
        "revalidate-failed-semantic-policy"
    )
    semantic_policy_revalidation.add_argument("job_id")
    model_validation_recovery = sub.add_parser(
        "resume-failed-model-validation"
    )
    model_validation_recovery.add_argument("job_id")
    prompt_migration_recovery = sub.add_parser(
        "resume-failed-prompt-migration"
    )
    prompt_migration_recovery.add_argument("job_id")
    zero_call_recovery = sub.add_parser("resume-zero-call-config-failure")
    zero_call_recovery.add_argument("job_id")
    promotion_recovery = sub.add_parser(
        "resume-zero-call-promotion-failure"
    )
    promotion_recovery.add_argument("job_id")
    projection_recovery = sub.add_parser("resume-zero-call-projection-failure")
    projection_recovery.add_argument("job_id")
    audit = sub.add_parser("audit")
    audit.add_argument("scope_id")
    audit.add_argument("--require-promotion-coverage", action="store_true")
    args = parser.parse_args()
    store = SlowGraphStore(args.database, schema=load_graph_schema(args.repo))
    if args.command == "revalidate-failed":
        result = revalidate_failed_raw_response(store, args.job_id)
    elif args.command == "revalidate-failed-semantic-policy":
        result = revalidate_failed_semantic_policy_response(store, args.job_id)
    elif args.command == "resume-failed-model-validation":
        result = resume_failed_model_validation(
            store, args.job_id, TieredGraphPatchManager.from_env()
        )
    elif args.command == "resume-failed-prompt-migration":
        result = resume_failed_model_validation_after_prompt_migration(
            store, args.job_id, TieredGraphPatchManager.from_env()
        )
    elif args.command == "resume-zero-call-config-failure":
        result = resume_zero_call_configuration_failure(store, args.job_id)
    elif args.command == "resume-zero-call-promotion-failure":
        result = resume_zero_call_promotion_failure(store, args.job_id)
    elif args.command == "resume-zero-call-projection-failure":
        result = resume_zero_call_projection_failure(store, args.job_id)
    else:
        manager = TieredGraphPatchManager.from_env()
        if args.command == "enqueue":
            if args.region:
                region = store.fast_regions(args.scope_id).get(args.region, [])
                result = [store.enqueue(args.scope_id, args.region, (item["memory_id"] for item in region), manager=manager)]
            else:
                result = store.enqueue_regions(args.scope_id, manager=manager)
        elif args.command == "audit":
            result = store.audit(
                args.scope_id,
                require_promotion_coverage=args.require_promotion_coverage,
            )
        elif args.command == "drain":
            result = store.drain(manager, batch_size=args.batch_size)
        elif args.command == "run":
            result = store.run_job(args.job_id, manager)
    print(_json(result))


if __name__ == "__main__":
    main()
