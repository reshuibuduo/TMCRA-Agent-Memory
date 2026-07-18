#!/usr/bin/env python3
"""Strict, append-only slow-memory graph controller."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

LEAF_VARIANT = "product_semantic_memory"
CAPSULE_VARIANT = "slow_memory_capsule"
PATCH_ACTIONS = frozenset(
    {
        "create",
        "revise",
        "challenge",
        "resolve_challenge",
        "retire",
        "noop",
    }
)
EDGE_TYPES = frozenset(
    {
        "supports",
        "contradicts",
        "derived_from",
        "supersedes",
        "challenges",
        "invalidates",
    }
)
SCHEMA_VERSION = "slow-graph-patch/v4"
SLOW_EDGE_SOURCE = "slow_graph_control_plane"
DEFAULT_CLAIM_LEASE_SECONDS = 15 * 60


class SlowGraphError(RuntimeError):
    pass


class PatchValidationError(SlowGraphError):
    pass


class EvidencePolicyError(SlowGraphError):
    pass


class StaleRevisionError(SlowGraphError):
    pass


class AuditError(SlowGraphError):
    pass


class DeepSeekCallError(SlowGraphError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class JobClaim:
    job_id: str
    attempt_id: str
    token: str
    owner: str


def _now() -> int:
    return int(time.time())


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _strict_json(value: str, *, label: str, expected: type) -> Any:
    try:
        result = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SlowGraphError(f"invalid {label} JSON") from exc
    if not isinstance(result, expected):
        raise SlowGraphError(f"{label} JSON must be {expected.__name__}")
    return result


def _required_text(value: Any, label: str) -> str:
    text = _clean(value)
    if not text:
        raise PatchValidationError(f"{label} is required")
    return text


def load_graph_schema(repo: str | Path) -> tuple[type[Any], type[Any]]:
    """Load the real graph record and edge types from an explicit TMCRA repo."""
    root = Path(repo).resolve()
    candidates = [root, root / "tmcra_code"]
    package_roots = [
        candidate
        for candidate in candidates
        if (candidate / "experiments" / "replacement" / "memory_graph.py").is_file()
    ]
    if len(package_roots) != 1:
        raise SlowGraphError(
            "--repo must resolve exactly one experiments/replacement/memory_graph.py "
            f"at the repo root or repo/tmcra_code: {root}"
        )
    package_root = package_roots[0]
    module_path = package_root / "experiments" / "replacement" / "memory_graph.py"
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("experiments.replacement.memory_graph")
        return module.SessionMemoryRecordV2, module.SessionMemoryEdgeV2
    except (ImportError, AttributeError) as exc:
        raise SlowGraphError(
            "unable to import real SessionMemoryRecordV2/SessionMemoryEdgeV2"
        ) from exc


class PatchManager(Protocol):
    model_config: Mapping[str, Any]
    prompt_hash: str
    last_call_metadata: Mapping[str, Any]

    def propose(
        self, region: Mapping[str, Any], capsules: list[Mapping[str, Any]]
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class DeepSeekProConfig:
    base_url: str
    key_pool: tuple[str, ...]
    max_tokens: int
    model: str = "deepseek-v4-pro"

    @classmethod
    def from_env(cls) -> "DeepSeekProConfig":
        base_url = _clean(os.getenv("TMCRA_DEEPSEEK_PRO_BASE_URL"))
        keys = tuple(
            item.strip()
            for item in _clean(os.getenv("TMCRA_DEEPSEEK_PRO_KEY_POOL")).split(",")
            if item.strip()
        )
        try:
            max_tokens = int(_clean(os.getenv("TMCRA_DEEPSEEK_PRO_MAX_TOKENS")))
        except ValueError as exc:
            raise SlowGraphError(
                "TMCRA_DEEPSEEK_PRO_MAX_TOKENS must be an integer"
            ) from exc
        if not base_url or not keys or max_tokens <= 0:
            raise SlowGraphError(
                "DeepSeek requires BASE_URL, KEY_POOL, and positive MAX_TOKENS"
            )
        return cls(base_url.rstrip("/"), keys, max_tokens)


class DeepSeekProGraphPatchManager:
    """The only production patch manager: no fallback and no hidden defaults."""

    def __init__(self, config: DeepSeekProConfig) -> None:
        if config.model != "deepseek-v4-pro":
            raise SlowGraphError("DeepSeek model must be deepseek-v4-pro")
        self.config = config
        self._key_index = 0
        self.model_config = {
            "model": config.model,
            "temperature": 0,
            "thinking": "disabled",
            "max_tokens": config.max_tokens,
        }
        self.prompt_hash = _digest(self._messages("<region>", []))
        self.last_call_metadata: Mapping[str, Any] = {}

    def _messages(self, region: Any, capsules: Any) -> list[dict[str, str]]:
        schema = {
            "operations": [
                {
                    "action": "create|revise|challenge|resolve_challenge|retire|noop",
                    "capsule_id": "required for non-create operations; omit for create because controller derives it from region_key",
                    "base_revision": "required for mutations of an existing capsule",
                    "claims": [
                        {
                            "canonical_slot": "stable semantic slot shared by corrections of the same durable fact",
                            "text": "string",
                            "support": ["fast id"],
                            "counterevidence": ["fast id"],
                        }
                    ],
                }
            ]
        }
        return [
            {
                "role": "system",
                "content": (
                    "You manage only the slow durable layer above immutable fast evidence leaves. "
                    "Return exactly one JSON GraphPatch and no prose. Create or revise a capsule only for "
                    "a durable preference, identity fact, routine, relationship, standing constraint, or stable "
                    "long-running state. A single explicit durable assertion is sufficient; never require an "
                    "arbitrary repetition count. Transient events, one-off tasks, quoted third-party claims, "
                    "and evidence that does not establish durable memory must produce an empty operations list. "
                    "Use challenge when new fast evidence conflicts but does not resolve the old claim; use "
                    "resolve_challenge when the supplied evidence resolves an existing challenge; use retire only "
                    "when evidence establishes that a capsule is no longer applicable. Topology merge and split "
                    "are not part of this regional V1 controller. canonical_slot must remain identical across corrections "
                    "of the same fact. Every claim canonical_slot must exactly copy canonical_slot_key from at "
                    "least one fast evidence leaf cited by that claim. Do not invent evidence IDs, capsule IDs "
                    "for create, confidence, source "
                    "parents, claim_id, or fields outside the supplied schema. The controller assigns claim_id. "
                    "Always emit support and counterevidence arrays, using [] for the empty side; the controller "
                    "also accepts an omitted empty side as a transport-level normalization. Every non-noop claim "
                    "must cite supplied fast leaf evidence. record_state and slow_graph_evidence_role are "
                    "controller-owned. Never promote historical_noncurrent evidence as a current claim when "
                    "current_authoritative evidence exists for the same canonical_slot; use noncurrent evidence "
                    "only as history or counterevidence. Each region has exactly one controller-owned capsule: "
                    "emit create only when "
                    "capsules is empty; otherwise mutate the supplied capsule_id or emit an empty operations list. "
                    "Return at most one operation. Empty operations is the canonical region-level noop."
                ),
            },
            {
                "role": "user",
                "content": _json(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "schema": schema,
                        "region": region,
                        "capsules": capsules,
                    }
                ),
            },
        ]

    def propose(
        self, region: Mapping[str, Any], capsules: list[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        key_index = self._key_index % len(self.config.key_pool)
        key = self.config.key_pool[key_index]
        self._key_index += 1
        body = {
            "model": "deepseek-v4-pro",
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "thinking": {"type": "disabled"},
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
            "messages": self._messages(region, capsules),
        }
        request = urllib.request.Request(
            self.config.base_url + "/chat/completions",
            data=_json(body).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        physical_call_id = "dsc_" + uuid.uuid4().hex
        started_at = _now()
        self.last_call_metadata = {
            "request": body,
            "key_index": key_index,
            "started_at": started_at,
            "physical_call_id": physical_call_id,
            "status": "started",
        }
        try:
            with urllib.request.urlopen(request, timeout=90) as response:  # nosec B310
                status = response.getcode()
                raw_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")
            except (AttributeError, OSError):
                detail = ""
            self.last_call_metadata = {
                **self.last_call_metadata,
                "status": "http_error",
                "http_status": exc.code,
                "error": detail,
                "completed_at": _now(),
            }
            raise DeepSeekCallError(
                f"DeepSeek HTTP {exc.code}: {detail}",
                retryable=exc.code == 429 or exc.code >= 500,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.last_call_metadata = {
                **self.last_call_metadata,
                "status": "request_error",
                "error": f"{exc.__class__.__name__}: {exc}",
                "completed_at": _now(),
            }
            raise DeepSeekCallError(
                f"DeepSeek transport failure: {exc}", retryable=True
            ) from exc
        if status < 200 or status >= 300:
            self.last_call_metadata = {
                **self.last_call_metadata,
                "status": "unexpected_http_status",
                "http_status": status,
                "raw_response": raw_text,
                "completed_at": _now(),
            }
            raise DeepSeekCallError(
                f"DeepSeek returned HTTP {status}", retryable=status >= 500
            )
        self.last_call_metadata = {
            **self.last_call_metadata,
            "status": "response_received",
            "http_status": status,
            "raw_response": raw_text,
            "completed_at": _now(),
        }
        raw = _strict_json(raw_text, label="DeepSeek response", expected=dict)
        choices = raw.get("choices")
        usage = raw.get("usage")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(usage, Mapping)
        ):
            raise DeepSeekCallError(
                "DeepSeek response missing strict choices/usage", retryable=False
            )
        choice = choices[0]
        if not isinstance(choice, Mapping) or not _clean(choice.get("finish_reason")):
            raise DeepSeekCallError(
                "DeepSeek response missing finish_reason", retryable=False
            )
        message = choice.get("message")
        if not isinstance(message, Mapping) or not _clean(message.get("content")):
            raise DeepSeekCallError(
                "DeepSeek response missing choices[0].message.content", retryable=False
            )
        finish_reason = _clean(choice["finish_reason"])
        self.last_call_metadata = {
            **self.last_call_metadata,
            "finish_reason": finish_reason,
            "content": _clean(message["content"]),
            "usage": dict(usage),
        }
        if finish_reason != "stop":
            self.last_call_metadata = {
                **self.last_call_metadata,
                "status": "incomplete_response",
            }
            raise DeepSeekCallError(
                f"DeepSeek response finish_reason must be stop, got {finish_reason!r}",
                retryable=False,
            )
        self.last_call_metadata = {
            **self.last_call_metadata,
            "request": body,
            "response_id": _required_text(raw.get("id"), "DeepSeek response id"),
            "finish_reason": finish_reason,
            "choices": choices,
            "content": _clean(message["content"]),
            "usage": dict(usage),
            "http_status": status,
            "physical_call_id": physical_call_id,
            "status": "completed",
            "completed_at": _now(),
        }
        patch = _strict_json(
            _clean(message["content"]), label="DeepSeek content", expected=dict
        )
        transport_normalizations = []
        for operation in patch.get("operations", []):
            if (
                isinstance(operation, Mapping)
                and operation.get("action") == "create"
                and "capsule_id" in operation
                and operation.get("capsule_id") is None
            ):
                transport_normalizations.append(
                    {
                        "code": "create_null_capsule_id_ignored",
                        "field": "capsule_id",
                    }
                )
        if transport_normalizations:
            self.last_call_metadata = {
                **self.last_call_metadata,
                "transport_normalizations": transport_normalizations,
            }
        validate_patch(patch)
        return patch


def _ids(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise PatchValidationError(f"{label} must be a non-empty list")
    result = [_required_text(item, label) for item in value]
    if len(set(result)) != len(result):
        raise PatchValidationError(f"{label} contains duplicates")
    return result


def _validate_source_parents(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise PatchValidationError("source_parents must be a non-empty list")
    expected = {
        "session_index",
        "parent_chunk_index",
        "message_index",
        "source_record_id",
        "event_id",
        "evidence_char_start",
        "evidence_char_end",
    }
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for parent in value:
        if not isinstance(parent, Mapping) or set(parent) != expected:
            raise PatchValidationError("source_parent has an invalid schema")
        normalized = dict(parent)
        for key in ("session_index", "parent_chunk_index", "message_index"):
            if not isinstance(normalized[key], int) or normalized[key] < 0:
                raise PatchValidationError(f"source_parent {key} must be non-negative integer")
        if normalized["parent_chunk_index"] != normalized["message_index"]:
            raise PatchValidationError("source_parent chunk/message coordinates disagree")
        if (
            not isinstance(normalized["evidence_char_start"], int)
            or not isinstance(normalized["evidence_char_end"], int)
            or normalized["evidence_char_start"] < 0
            or normalized["evidence_char_end"] <= normalized["evidence_char_start"]
        ):
            raise PatchValidationError("source_parent evidence character span is invalid")
        _required_text(normalized["source_record_id"], "source_record_id")
        _required_text(normalized["event_id"], "event_id")
        digest = _json(normalized)
        if digest not in seen:
            seen.add(digest)
            result.append(normalized)
    return result


def _validate_claims(
    value: Any, *, stored: bool = False
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise PatchValidationError("claims must be a non-empty list")
    claims: list[dict[str, Any]] = []
    seen = set()
    for claim in value:
        expected = {"canonical_slot", "text", "support", "counterevidence"}
        if stored:
            expected.add("claim_id")
            expected.add("source_parents")
        if not isinstance(claim, Mapping):
            raise PatchValidationError("each claim must be an object")
        if stored:
            valid_shape = set(claim) == expected
        else:
            valid_shape = (
                {"canonical_slot", "text"}.issubset(claim)
                and set(claim).issubset(expected)
            )
        if not valid_shape:
            raise PatchValidationError(
                "each claim has an invalid schema"
            )
        support = claim.get("support", [])
        counter = claim.get("counterevidence", [])
        if not isinstance(support, list) or not isinstance(counter, list):
            raise PatchValidationError("claim evidence must be lists")
        normalized_support = _ids(support, "claim support") if support else []
        normalized_counter = (
            _ids(counter, "claim counterevidence") if counter else []
        )
        if not normalized_support and not normalized_counter:
            raise PatchValidationError("each claim needs support or counterevidence")
        canonical_slot = _required_text(
            claim.get("canonical_slot"), "canonical_slot"
        )
        claim_text = _required_text(claim.get("text"), "claim text")
        claim_id = (
            _required_text(claim.get("claim_id"), "claim_id")
            if stored
            else "clm_"
            + _digest(
                {
                    "canonical_slot": canonical_slot,
                    "text": claim_text,
                    "support": normalized_support,
                    "counterevidence": normalized_counter,
                }
            )[:24]
        )
        if claim_id in seen:
            raise PatchValidationError("claim_id must be unique")
        seen.add(claim_id)
        claims.append(
            {
                "claim_id": claim_id,
                "canonical_slot": canonical_slot,
                "text": claim_text,
                "support": normalized_support,
                "counterevidence": normalized_counter,
                **(
                    {
                        "source_parents": _validate_source_parents(
                            claim.get("source_parents")
                        )
                    }
                    if stored
                    else {}
                ),
            }
        )
    return claims


def validate_patch(patch: Mapping[str, Any]) -> None:
    if (
        not isinstance(patch, Mapping)
        or set(patch) != {"operations"}
        or not isinstance(patch["operations"], list)
    ):
        raise PatchValidationError("GraphPatch must contain exactly an operations list")
    if len(patch["operations"]) > 1:
        raise PatchValidationError("GraphPatch may contain at most one region operation")
    for operation in patch["operations"]:
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
            if "capsule_key" in operation:
                raise PatchValidationError(
                    "create capsule identity is controller-derived from region_key"
                )
            if "capsule_id" in operation:
                if operation.get("capsule_id") is not None:
                    raise PatchValidationError(
                        "create capsule identity is controller-derived from region_key"
                    )
                allowed.add("capsule_id")
        elif action == "noop":
            allowed.add("capsule_id")
            if "capsule_id" in operation:
                _required_text(operation.get("capsule_id"), "capsule_id")
        else:
            allowed.update({"capsule_id", "base_revision"})
            _required_text(operation.get("capsule_id"), "capsule_id")
            if (
                not isinstance(operation.get("base_revision"), int)
                or operation["base_revision"] < 1
            ):
                raise PatchValidationError("base_revision must be a positive integer")
        if set(operation) - allowed:
            raise PatchValidationError("unexpected GraphPatch operation fields")
        if action != "noop":
            _validate_claims(operation.get("claims"))


class SlowGraphStore:
    def __init__(
        self,
        database: str | Path,
        *,
        schema: tuple[type[Any], type[Any]],
        claim_lease_seconds: int = DEFAULT_CLAIM_LEASE_SECONDS,
    ) -> None:
        if claim_lease_seconds <= 0:
            raise SlowGraphError("claim lease duration must be positive")
        self.database = Path(database)
        self.record_type, self.edge_type = schema
        self.claim_lease_seconds = claim_lease_seconds
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.database)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        return con

    @contextmanager
    def connection(self):
        con = self.connect()
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_schema(self) -> None:
        with self.connection() as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS slow_graph_jobs (
              job_id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE, scope_id TEXT NOT NULL, region_key TEXT NOT NULL,
              evidence_ids_json TEXT NOT NULL, metadata_json TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','retryable','failed','completed')),
              attempts INTEGER NOT NULL, last_error TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
              claim_token TEXT, claim_owner TEXT, lease_expires_at INTEGER);
            CREATE TABLE IF NOT EXISTS slow_graph_attempts (
              attempt_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, scope_id TEXT NOT NULL, status TEXT NOT NULL,
              call_metadata_json TEXT NOT NULL, error TEXT NOT NULL, created_at INTEGER NOT NULL, completed_at INTEGER,
              claim_token TEXT, claim_owner TEXT);
            CREATE TABLE IF NOT EXISTS slow_graph_batches (
              batch_id TEXT PRIMARY KEY, batch_key TEXT NOT NULL UNIQUE, scope_id TEXT NOT NULL,
              evidence_snapshot_hash TEXT NOT NULL, job_ids_json TEXT NOT NULL, manager_metadata_json TEXT NOT NULL,
              created_at INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS slow_graph_patches (
              patch_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, scope_id TEXT NOT NULL, region_key TEXT NOT NULL, manager_model TEXT NOT NULL,
              patch_json TEXT NOT NULL, call_metadata_json TEXT NOT NULL, applied_at INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS slow_graph_patch_operations (
              operation_id TEXT PRIMARY KEY, patch_id TEXT NOT NULL, ordinal INTEGER NOT NULL, capsule_id TEXT NOT NULL, action TEXT NOT NULL,
              base_revision INTEGER, result_revision INTEGER, operation_json TEXT NOT NULL, created_at INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS slow_graph_provenance (
              provenance_id TEXT PRIMARY KEY, patch_id TEXT NOT NULL, scope_id TEXT NOT NULL, capsule_id TEXT NOT NULL, revision INTEGER NOT NULL,
              evidence_memory_id TEXT NOT NULL, claim_id TEXT NOT NULL, polarity TEXT NOT NULL, source_parent_json TEXT NOT NULL, created_at INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_slow_graph_jobs_pending ON slow_graph_jobs(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_slow_graph_provenance_capsule ON slow_graph_provenance(scope_id, capsule_id, revision);
            CREATE TABLE IF NOT EXISTS memory_edges (
              scope_id TEXT NOT NULL, edge_id TEXT NOT NULL, source_memory_id TEXT NOT NULL, target_memory_id TEXT NOT NULL, edge_type TEXT NOT NULL,
              score REAL NOT NULL, model_score REAL NOT NULL, evidence_turn INTEGER NOT NULL, evidence TEXT NOT NULL, metadata_json TEXT NOT NULL,
              PRIMARY KEY(scope_id, edge_id));
            CREATE TABLE IF NOT EXISTS slot_heads (
              scope_id TEXT NOT NULL, slot_key TEXT NOT NULL, memory_id TEXT NOT NULL,
              PRIMARY KEY(scope_id,slot_key));
            CREATE TABLE IF NOT EXISTS slot_history (
              scope_id TEXT NOT NULL, slot_key TEXT NOT NULL, ordinal INTEGER NOT NULL, memory_id TEXT NOT NULL,
              PRIMARY KEY(scope_id,slot_key,ordinal));
            """)
            self._add_column_if_missing(con, "slow_graph_jobs", "claim_token TEXT")
            self._add_column_if_missing(con, "slow_graph_jobs", "claim_owner TEXT")
            self._add_column_if_missing(
                con, "slow_graph_jobs", "lease_expires_at INTEGER"
            )
            self._add_column_if_missing(
                con, "slow_graph_attempts", "claim_token TEXT"
            )
            self._add_column_if_missing(
                con, "slow_graph_attempts", "claim_owner TEXT"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_slow_graph_jobs_claimable "
                "ON slow_graph_jobs(status, claim_token, created_at)"
            )

    @staticmethod
    def _add_column_if_missing(
        con: sqlite3.Connection, table: str, declaration: str
    ) -> None:
        column = declaration.split()[0]
        columns = {
            str(row["name"])
            for row in con.execute("PRAGMA table_info(" + table + ")")
        }
        if column not in columns:
            con.execute("ALTER TABLE " + table + " ADD COLUMN " + declaration)

    def _records_table_exists(self, con: sqlite3.Connection) -> None:
        if (
            con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='records'"
            ).fetchone()
            is None
        ):
            raise SlowGraphError("real graph records table is missing")

    def _metadata(self, row: sqlite3.Row, label: str) -> dict[str, Any]:
        return _strict_json(row["metadata_json"], label=label, expected=dict)

    def fast_regions(self, scope_id: str) -> dict[str, list[dict[str, Any]]]:
        regions: dict[str, list[dict[str, Any]]] = {}
        with self.connection() as con:
            self._records_table_exists(con)
            rows = con.execute(
                "SELECT memory_id,value,relation,turn_index,metadata_json FROM records WHERE scope_id=?",
                (scope_id,),
            ).fetchall()
        for row in rows:
            metadata = self._metadata(row, "fast evidence metadata")
            if (
                metadata.get("content_variant") != LEAF_VARIANT
                or metadata.get("memory_layer") != "fast"
                or metadata.get("node_kind") != "atomic_user_assertion"
                or metadata.get("atomic_evidence_leaf") is not True
                or metadata.get("authority") != "user_assertion"
            ):
                continue
            key = _clean(
                metadata.get("graph_entity_key")
                or metadata.get("entity_key")
                or metadata.get("domain")
            )
            if not key:
                raise SlowGraphError("fast evidence is missing region key")
            regions.setdefault(key, []).append(
                {
                    "memory_id": row["memory_id"],
                    "value": row["value"],
                    "relation": row["relation"],
                    "turn_index": row["turn_index"],
                    "metadata": metadata,
                }
            )
        return regions

    def _capsules(
        self, con: sqlite3.Connection, scope_id: str, region_key: str
    ) -> list[dict[str, Any]]:
        rows = con.execute(
            "SELECT memory_id,state,value,metadata_json FROM records WHERE scope_id=?",
            (scope_id,),
        ).fetchall()
        revisions: list[dict[str, Any]] = []
        for row in rows:
            meta = self._metadata(row, "capsule metadata")
            if (
                meta.get("content_variant") == CAPSULE_VARIANT
                and meta.get("region_key") == region_key
            ):
                revisions.append(
                    {
                        "memory_id": row["memory_id"],
                        "record_state": row["state"],
                        "value": row["value"],
                        **meta,
                    }
                )
        if not revisions:
            return []
        latest_revision = max(int(item["revision"]) for item in revisions)
        latest = [
            item for item in revisions if int(item["revision"]) == latest_revision
        ]
        if len(latest) != 1:
            raise AuditError("region capsule lacks one latest revision")
        return latest

    def _job_metadata(
        self,
        con: sqlite3.Connection,
        scope_id: str,
        region_key: str,
        evidence_ids: list[str],
        manager: PatchManager | None,
    ) -> dict[str, Any]:
        evidence = self._evidence(con, scope_id, evidence_ids)
        normalized_region = _required_text(region_key, "region key")
        foreign_evidence = [
            item["memory_id"]
            for item in evidence
            if _clean(item["metadata"].get("graph_entity_key"))
            != normalized_region
        ]
        if foreign_evidence:
            raise EvidencePolicyError(
                "fast evidence graph_entity_key does not match job region: "
                + ",".join(foreign_evidence)
            )
        model = dict(manager.model_config) if manager else {"model": "unbound"}
        prompt_hash = manager.prompt_hash if manager else "unbound"
        capsules = self._capsules(con, scope_id, region_key)
        return {
            "schema_version": SCHEMA_VERSION,
            "schema_hash": _digest(
                {"version": SCHEMA_VERSION, "actions": sorted(PATCH_ACTIONS)}
            ),
            "prompt_hash": prompt_hash,
            "model_config": model,
            "evidence_content_hash": _digest(evidence),
            "capsule_revision_hash": _digest(capsules),
        }

    def _enqueue_in_connection(
        self,
        con: sqlite3.Connection,
        scope_id: str,
        region_key: str,
        evidence_ids: list[str],
        *,
        manager: PatchManager | None = None,
    ) -> str:
        now = _now()
        metadata = self._job_metadata(
            con, scope_id, region_key, evidence_ids, manager
        )
        idem = _digest(
            {
                "scope_id": scope_id,
                "region_key": region_key,
                "evidence_ids": evidence_ids,
                **{
                    key: value
                    for key, value in metadata.items()
                    if key != "capsule_revision_hash"
                },
            }
        )
        job_id = "sgj_" + uuid.uuid4().hex
        con.execute(
            "INSERT OR IGNORE INTO slow_graph_jobs("
            "job_id,idempotency_key,scope_id,region_key,evidence_ids_json,metadata_json,"
            "status,attempts,last_error,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                job_id,
                idem,
                scope_id,
                region_key,
                _json(evidence_ids),
                _json(metadata),
                "pending",
                0,
                "",
                now,
                now,
            ),
        )
        row = con.execute(
            "SELECT job_id FROM slow_graph_jobs WHERE idempotency_key=?", (idem,)
        ).fetchone()
        return str(row["job_id"])

    def enqueue(
        self,
        scope_id: str,
        region_key: str,
        evidence_ids: Iterable[str],
        *,
        manager: PatchManager | None = None,
    ) -> str:
        normalized_ids = sorted(
            set(_required_text(item, "evidence id") for item in evidence_ids)
        )
        if not normalized_ids:
            raise SlowGraphError("cannot enqueue without fast evidence")
        with self.connection() as con:
            self._records_table_exists(con)
            return self._enqueue_in_connection(
                con,
                scope_id,
                region_key,
                normalized_ids,
                manager=manager,
            )

    def enqueue_regions(
        self, scope_id: str, *, manager: PatchManager | None = None
    ) -> list[str]:
        regions = self.fast_regions(scope_id)
        manager_metadata = {
            "schema_version": SCHEMA_VERSION,
            "prompt_hash": manager.prompt_hash if manager else "unbound",
            "model_config": dict(manager.model_config)
            if manager
            else {"model": "unbound"},
        }
        snapshot = {
            key: [
                {
                    "memory_id": item["memory_id"],
                    "value": item["value"],
                    "metadata": item["metadata"],
                }
                for item in values
            ]
            for key, values in sorted(regions.items())
        }
        evidence_snapshot_hash = _digest(snapshot)
        batch_key = _digest(
            {
                "scope_id": scope_id,
                "evidence_snapshot_hash": evidence_snapshot_hash,
                **manager_metadata,
            }
        )
        with self.connection() as con:
            self._records_table_exists(con)
            existing = con.execute(
                "SELECT job_ids_json FROM slow_graph_batches WHERE batch_key=?",
                (batch_key,),
            ).fetchone()
            if existing is not None:
                return [
                    str(item)
                    for item in _strict_json(
                        existing["job_ids_json"], label="batch job IDs", expected=list
                    )
                ]
            job_ids = [
                self._enqueue_in_connection(
                    con,
                    scope_id,
                    key,
                    sorted(str(item["memory_id"]) for item in values),
                    manager=manager,
                )
                for key, values in sorted(regions.items())
            ]
            con.execute(
                "INSERT INTO slow_graph_batches VALUES(?,?,?,?,?,?,?)",
                (
                    "sgb_" + uuid.uuid4().hex,
                    batch_key,
                    scope_id,
                    evidence_snapshot_hash,
                    _json(job_ids),
                    _json(manager_metadata),
                    _now(),
                ),
            )
            return job_ids

    def _job(self, job_id: str) -> sqlite3.Row:
        with self.connection() as con:
            row = con.execute(
                "SELECT * FROM slow_graph_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise SlowGraphError("unknown slow graph job: " + job_id)
        self._metadata(row, "job")
        return row

    def _evidence(
        self, con: sqlite3.Connection, scope_id: str, ids: Iterable[str]
    ) -> list[dict[str, Any]]:
        result = []
        for evidence_id in sorted(set(ids)):
            row = con.execute(
                "SELECT memory_id,value,turn_index,state,metadata_json FROM records WHERE scope_id=? AND memory_id=?",
                (scope_id, evidence_id),
            ).fetchone()
            if row is None:
                raise EvidencePolicyError("unknown evidence: " + evidence_id)
            meta = self._metadata(row, "fast evidence")
            if (
                meta.get("content_variant") != LEAF_VARIANT
                or meta.get("memory_layer") != "fast"
                or meta.get("node_kind") != "atomic_user_assertion"
                or meta.get("atomic_evidence_leaf") is not True
                or meta.get("authority") != "user_assertion"
            ):
                raise EvidencePolicyError(
                    "only fast product_semantic_memory leaves may support capsules: "
                    + evidence_id
                )
            parent = {
                key: meta.get(key)
                for key in (
                    "session_index",
                    "message_index",
                    "source_record_id",
                    "event_id",
                    "evidence_char_start",
                    "evidence_char_end",
                )
            }
            if any(value is None or _clean(value) == "" for value in parent.values()):
                raise EvidencePolicyError(
                    "fast evidence lacks structured source_parent fields: "
                    + evidence_id
                )
            parent["parent_chunk_index"] = parent["message_index"]
            record_state = _clean(row["state"])
            evidence_role = (
                "current_authoritative"
                if record_state in {"active", "parallel_active", "promoted"}
                else "historical_noncurrent"
            )
            result.append(
                {
                    "memory_id": row["memory_id"],
                    "value": row["value"],
                    "turn_index": row["turn_index"],
                    "record_state": record_state,
                    "slow_graph_evidence_role": evidence_role,
                    "metadata": {
                        **meta,
                        "record_state": record_state,
                        "slow_graph_evidence_role": evidence_role,
                    },
                    "source_parent": parent,
                }
            )
        return result

    def _head(
        self, con: sqlite3.Connection, scope_id: str, capsule_id: str
    ) -> tuple[int, str] | None:
        rows = con.execute(
            "SELECT memory_id,metadata_json FROM records WHERE scope_id=?", (scope_id,)
        ).fetchall()
        candidates = []
        for row in rows:
            meta = self._metadata(row, "capsule metadata")
            if (
                meta.get("content_variant") == CAPSULE_VARIANT
                and meta.get("capsule_id") == capsule_id
            ):
                candidates.append((int(meta["revision"]), str(row["memory_id"])))
        return max(candidates) if candidates else None

    def _capsule_id(self, scope_id: str, region_key: str) -> str:
        return (
            "cap_"
            + _digest(
                {
                    "scope_id": scope_id,
                    "region_key": _required_text(region_key, "region_key"),
                }
            )[:24]
        )

    def _mark_superseded(
        self, con: sqlite3.Connection, scope_id: str, memory_id: str
    ) -> None:
        row = con.execute(
            "SELECT slot_key,metadata_json FROM records WHERE scope_id=? AND memory_id=?",
            (scope_id, memory_id),
        ).fetchone()
        if row is None:
            raise SlowGraphError("cannot supersede missing record: " + memory_id)
        metadata = self._metadata(row, "superseded record")
        if metadata.get("content_variant") == CAPSULE_VARIANT:
            metadata["status"] = "superseded"
        con.execute(
            "UPDATE records SET state='superseded',metadata_json=? WHERE scope_id=? AND memory_id=?",
            (_json(metadata), scope_id, memory_id),
        )
        con.execute(
            "DELETE FROM slot_heads WHERE scope_id=? AND slot_key=? AND memory_id=?",
            (scope_id, row["slot_key"], memory_id),
        )

    def _write_edge(
        self,
        con: sqlite3.Connection,
        *,
        scope_id: str,
        source: str,
        target: str,
        edge_type: str,
        patch_id: str,
        evidence_refs: list[str],
        action: str,
        turn: int,
    ) -> None:
        if edge_type not in EDGE_TYPES or source == target:
            raise AuditError("invalid graph edge")
        edge = self.edge_type(
            edge_id="sge_"
            + _digest(
                {
                    "patch": patch_id,
                    "source": source,
                    "target": target,
                    "type": edge_type,
                }
            )[:24],
            source_memory_id=source,
            target_memory_id=target,
            edge_type=edge_type,
            score=0.0,
            model_score=0.0,
            evidence_turn=turn,
            evidence="slow graph patch",
            metadata={
                "edge_source": SLOW_EDGE_SOURCE,
                "patch_id": patch_id,
                "evidence_refs": evidence_refs,
                "action": action,
            },
        )
        con.execute(
            "INSERT INTO memory_edges VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                scope_id,
                edge.edge_id,
                edge.source_memory_id,
                edge.target_memory_id,
                edge.edge_type,
                edge.score,
                edge.model_score,
                edge.evidence_turn,
                edge.evidence,
                _json(edge.metadata),
            ),
        )

    def _insert_revision(
        self,
        con: sqlite3.Connection,
        *,
        job: sqlite3.Row,
        patch_id: str,
        operation: Mapping[str, Any],
        capsule_id: str,
        revision: int,
        action: str,
        old_memory_id: str | None = None,
    ) -> str:
        claims = _validate_claims(operation["claims"])
        evidence_ids = sorted(
            {
                item
                for claim in claims
                for key in ("support", "counterevidence")
                for item in claim[key]
            }
        )
        allowed_evidence = set(
            _strict_json(
                job["evidence_ids_json"], label="job evidence IDs", expected=list
            )
        )
        if not set(evidence_ids).issubset(allowed_evidence):
            raise EvidencePolicyError(
                "GraphPatch cited fast evidence that was not supplied to this job"
        )
        evidence = self._evidence(con, job["scope_id"], evidence_ids)
        evidence_by_id = {item["memory_id"]: item for item in evidence}
        controller_normalizations: list[dict[str, Any]] = []
        for index, claim in enumerate(claims):
            cited_slots = {
                _clean(evidence_by_id[evidence_id]["metadata"].get("canonical_slot_key"))
                for evidence_id in [*claim["support"], *claim["counterevidence"]]
            }
            cited_slots.discard("")
            if claim["canonical_slot"] not in cited_slots:
                if len(cited_slots) != 1:
                    raise EvidencePolicyError(
                        "claim canonical_slot is ambiguous across cited fast leaf slots"
                    )
                authoritative_slot = next(iter(cited_slots))
                original_slot = claim["canonical_slot"]
                normalized_claim = {
                    **claim,
                    "canonical_slot": authoritative_slot,
                }
                normalized_claim["claim_id"] = "clm_" + _digest(
                    {
                        "canonical_slot": authoritative_slot,
                        "text": normalized_claim["text"],
                        "support": normalized_claim["support"],
                        "counterevidence": normalized_claim["counterevidence"],
                    }
                )[:24]
                claims[index] = normalized_claim
                controller_normalizations.append(
                    {
                        "code": "canonical_slot_bound_to_unique_cited_leaf",
                        "claim_index": index,
                        "model_value": original_slot,
                        "authoritative_value": authoritative_slot,
                    }
                )
        claim_ids = [claim["claim_id"] for claim in claims]
        if len(set(claim_ids)) != len(claim_ids):
            raise EvidencePolicyError(
                "canonical slot binding produced duplicate claims"
            )
        status = {
            "challenge": "challenged",
            "retire": "retired",
            "resolve_challenge": "active",
        }.get(action, "active")
        stored_claims: list[dict[str, Any]] = []
        for claim in claims:
            claim_parents = [
                evidence_by_id[evidence_id]["source_parent"]
                for evidence_id in [*claim["support"], *claim["counterevidence"]]
            ]
            stored_claims.append(
                {**claim, "source_parents": _validate_source_parents(claim_parents)}
            )
        source_parents = _validate_source_parents(
            [parent for claim in stored_claims for parent in claim["source_parents"]]
        )
        record_id = f"slow.{capsule_id}.r{revision}"
        metadata = {
            "memory_layer": "slow",
            "content_variant": CAPSULE_VARIANT,
            "capsule_id": capsule_id,
            "revision": revision,
            "status": status,
            "claims": stored_claims,
            "source_parents": source_parents,
            "canonical_slots": sorted(
                {claim["canonical_slot"] for claim in stored_claims}
            ),
            "patch_id": patch_id,
            "region_key": job["region_key"],
            "action": action,
            "controller_normalizations": controller_normalizations,
        }
        record_state = "active" if status in {"active", "challenged"} else status
        record = self.record_type(
            memory_id=record_id,
            category=CAPSULE_VARIANT,
            slot_key="slow." + capsule_id,
            value=_clean(operation.get("summary")) or _json(claims),
            relation="capsule_revision",
            anchor_concepts=[job["region_key"]],
            evidence_anchors=evidence_ids,
            salience=0.7,
            confidence=0.0,
            source_kind="slow_graph",
            turn_index=max(item["turn_index"] for item in evidence),
            state=record_state,
            supersedes=[old_memory_id] if old_memory_id else [],
            metadata=metadata,
        )
        payload = record.to_dict()
        con.execute(
            "INSERT INTO records(scope_id,memory_id,category,slot_key,value,relation,anchor_concepts_json,evidence_anchors_json,salience,confidence,source_kind,turn_index,state,supersedes_json,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                job["scope_id"],
                payload["memory_id"],
                payload["category"],
                payload["slot_key"],
                payload["value"],
                payload["relation"],
                _json(payload["anchor_concepts"]),
                _json(payload["evidence_anchors"]),
                payload["salience"],
                payload["confidence"],
                payload["source_kind"],
                payload["turn_index"],
                payload["state"],
                _json(payload["supersedes"]),
                _json(payload["metadata"]),
            ),
        )
        slot_key = payload["slot_key"]
        ordinal = int(
            con.execute(
                "SELECT COALESCE(MAX(ordinal),-1)+1 FROM slot_history WHERE scope_id=? AND slot_key=?",
                (job["scope_id"], slot_key),
            ).fetchone()[0]
        )
        con.execute(
            "INSERT INTO slot_history(scope_id,slot_key,ordinal,memory_id) VALUES(?,?,?,?)",
            (job["scope_id"], slot_key, ordinal, record_id),
        )
        if record_state == "active":
            con.execute(
                "INSERT INTO slot_heads(scope_id,slot_key,memory_id) VALUES(?,?,?) "
                "ON CONFLICT(scope_id,slot_key) DO UPDATE SET memory_id=excluded.memory_id",
                (job["scope_id"], slot_key, record_id),
            )
        else:
            con.execute(
                "DELETE FROM slot_heads WHERE scope_id=? AND slot_key=?",
                (job["scope_id"], slot_key),
            )
        for claim in stored_claims:
            for polarity, edge_kind in (
                ("support", "supports"),
                ("counterevidence", "contradicts"),
            ):
                for evidence_id in claim[polarity]:
                    item = evidence_by_id[evidence_id]
                    con.execute(
                        "INSERT INTO slow_graph_provenance VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (
                            "sgv_" + uuid.uuid4().hex,
                            patch_id,
                            job["scope_id"],
                            capsule_id,
                            revision,
                            evidence_id,
                            claim["claim_id"],
                            polarity,
                            _json(item["source_parent"]),
                            _now(),
                        ),
                    )
                    self._write_edge(
                        con,
                        scope_id=job["scope_id"],
                        source=evidence_id,
                        target=record_id,
                        edge_type=edge_kind,
                        patch_id=patch_id,
                        evidence_refs=[evidence_id],
                        action=action,
                        turn=item["turn_index"],
                    )
        if old_memory_id:
            self._mark_superseded(con, job["scope_id"], old_memory_id)
            self._write_edge(
                con,
                scope_id=job["scope_id"],
                source=old_memory_id,
                target=record_id,
                edge_type="supersedes",
                patch_id=patch_id,
                evidence_refs=evidence_ids,
                action=action,
                turn=payload["turn_index"],
            )
        return record_id

    def _audit_transaction(self, con: sqlite3.Connection, scope_id: str) -> None:
        """Validate slow-layer invariants against the uncommitted write transaction."""
        rows = con.execute(
            "SELECT memory_id,state,metadata_json FROM records WHERE scope_id=?",
            (scope_id,),
        ).fetchall()
        capsules: dict[str, list[tuple[str, str, int, str]]] = {}
        known_records = {str(row["memory_id"]) for row in rows}
        for row in rows:
            meta = self._metadata(row, "transaction record")
            if meta.get("content_variant") != CAPSULE_VARIANT:
                continue
            capsule_id = _required_text(meta.get("capsule_id"), "capsule_id")
            revision = meta.get("revision")
            status = _required_text(meta.get("status"), "capsule status")
            if not isinstance(revision, int) or revision < 1:
                raise AuditError("capsule revision is invalid")
            if status not in {"active", "challenged", "retired", "superseded"}:
                raise AuditError("capsule status is invalid")
            claims = _validate_claims(meta.get("claims"), stored=True)
            capsules.setdefault(capsule_id, []).append(
                (str(row["memory_id"]), str(row["state"]), revision, status)
            )
            for claim in claims:
                expected = set(claim["support"] + claim["counterevidence"])
                actual = {
                    str(item["evidence_memory_id"])
                    for item in con.execute(
                        "SELECT evidence_memory_id FROM slow_graph_provenance "
                        "WHERE scope_id=? AND capsule_id=? AND revision=? AND claim_id=?",
                        (scope_id, capsule_id, revision, claim["claim_id"]),
                    )
                }
                if actual != expected:
                    raise AuditError("claim provenance does not match claim evidence")
        for capsule_id, revisions in capsules.items():
            ordered = sorted(revisions, key=lambda item: item[2])
            if [item[2] for item in ordered] != list(range(1, len(ordered) + 1)):
                raise AuditError("capsule revisions are not contiguous")
            memory_id, state, _, status = ordered[-1]
            slot_key = "slow." + capsule_id
            head = con.execute(
                "SELECT memory_id FROM slot_heads WHERE scope_id=? AND slot_key=?",
                (scope_id, slot_key),
            ).fetchone()
            if status in {"active", "challenged"}:
                if state != "active" or head is None or head["memory_id"] != memory_id:
                    raise AuditError("active/challenged capsule head is inconsistent")
            elif head is not None:
                raise AuditError("inactive capsule must not retain a slot head")
            history = [
                str(item["memory_id"])
                for item in con.execute(
                    "SELECT memory_id FROM slot_history WHERE scope_id=? AND slot_key=? ORDER BY ordinal",
                    (scope_id, slot_key),
                )
            ]
            if history != [item[0] for item in ordered]:
                raise AuditError("capsule slot history is inconsistent")
        for edge in con.execute(
            "SELECT source_memory_id,target_memory_id,edge_type,metadata_json "
            "FROM memory_edges WHERE scope_id=?",
            (scope_id,),
        ):
            meta = _strict_json(
                edge["metadata_json"], label="edge metadata", expected=dict
            )
            if meta.get("edge_source") != SLOW_EDGE_SOURCE:
                continue
            if (
                edge["source_memory_id"] == edge["target_memory_id"]
                or edge["edge_type"] not in EDGE_TYPES
                or edge["source_memory_id"] not in known_records
                or edge["target_memory_id"] not in known_records
                or not isinstance(meta.get("evidence_refs"), list)
                or not _clean(meta.get("patch_id"))
            ):
                raise AuditError("slow graph edge is inconsistent")

    def apply_patch(
        self,
        job_id: str,
        patch: Mapping[str, Any],
        *,
        manager_model: str,
        call_metadata: Mapping[str, Any] | None = None,
        claim: JobClaim,
    ) -> str:
        validate_patch(patch)
        if claim.job_id != job_id:
            raise SlowGraphError("claim does not belong to job")
        patch_id = "sgp_" + uuid.uuid4().hex
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            now = _now()
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
            metadata = self._metadata(job, "job")
            if metadata["schema_version"] != SCHEMA_VERSION:
                raise SlowGraphError("job schema metadata drift")
            con.execute(
                "INSERT INTO slow_graph_patches VALUES(?,?,?,?,?,?,?,?)",
                (
                    patch_id,
                    job_id,
                    job["scope_id"],
                    job["region_key"],
                    manager_model,
                    _json(patch),
                    _json(dict(call_metadata or {})),
                    now,
                ),
            )
            ordinal = 0
            for operation in patch["operations"]:
                action = operation["action"]
                region_capsule_id = self._capsule_id(
                    job["scope_id"], job["region_key"]
                )
                capsule_id = (
                    region_capsule_id
                    if action == "create"
                    else operation.get("capsule_id")
                    or "region:" + job["region_key"]
                )
                if action != "create" and operation.get("capsule_id"):
                    if capsule_id != region_capsule_id:
                        raise EvidencePolicyError(
                            "GraphPatch capsule_id does not belong to the current region"
                        )
                head = self._head(con, job["scope_id"], capsule_id)
                if action == "create":
                    if head is not None:
                        raise StaleRevisionError("capsule already exists")
                    record_id = self._insert_revision(
                        con,
                        job=job,
                        patch_id=patch_id,
                        operation=operation,
                        capsule_id=capsule_id,
                        revision=1,
                        action=action,
                    )
                    base, revision = None, 1
                elif action == "noop":
                    if operation.get("capsule_id") and head is None:
                        raise StaleRevisionError("noop capsule does not exist")
                    if head is None:
                        record_id, base, revision = "", None, None
                    else:
                        record_id, (revision, _) = head[1], head
                        base = revision
                else:
                    if head is None or operation["base_revision"] != head[0]:
                        raise StaleRevisionError(
                            "base_revision is stale for " + capsule_id
                        )
                    base, revision = head[0], head[0] + 1
                    record_id = self._insert_revision(
                        con,
                        job=job,
                        patch_id=patch_id,
                        operation=operation,
                        capsule_id=capsule_id,
                        revision=revision,
                        action=action,
                        old_memory_id=head[1],
                    )
                    if action == "challenge":
                        self._write_edge(
                            con,
                            scope_id=job["scope_id"],
                            source=record_id,
                            target=head[1],
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
                            target=head[1],
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
                ordinal += 1
            completed_attempt = con.execute(
                "UPDATE slow_graph_attempts SET status='completed',"
                "call_metadata_json=?,completed_at=? WHERE attempt_id=? AND job_id=? "
                "AND claim_token=? AND claim_owner=? AND status='started'",
                (
                    _json(dict(call_metadata or {})),
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

    def _claim_pending_job(
        self, job_id: str | None, *, owner: str
    ) -> JobClaim | None:
        token = "sgc_" + uuid.uuid4().hex
        attempt_id = "sga_" + uuid.uuid4().hex
        now = _now()
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            if job_id is None:
                row = con.execute(
                    "SELECT job_id,scope_id FROM slow_graph_jobs WHERE status='pending' "
                    "AND claim_token IS NULL ORDER BY created_at LIMIT 1"
                ).fetchone()
            else:
                row = con.execute(
                    "SELECT job_id,scope_id FROM slow_graph_jobs WHERE job_id=? "
                    "AND status='pending' AND claim_token IS NULL",
                    (job_id,),
                ).fetchone()
            if row is None:
                return None
            claimed = con.execute(
                "UPDATE slow_graph_jobs SET claim_token=?,claim_owner=?,"
                "lease_expires_at=?,updated_at=? WHERE job_id=? AND status='pending' "
                "AND claim_token IS NULL",
                (
                    token,
                    owner,
                    now + self.claim_lease_seconds,
                    now,
                    str(row["job_id"]),
                ),
            )
            if claimed.rowcount != 1:
                return None
            con.execute(
                "INSERT INTO slow_graph_attempts("
                "attempt_id,job_id,scope_id,status,call_metadata_json,error,created_at,"
                "completed_at,claim_token,claim_owner) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    attempt_id,
                    str(row["job_id"]),
                    str(row["scope_id"]),
                    "started",
                    _json({}),
                    "",
                    now,
                    None,
                    token,
                    owner,
                ),
            )
        return JobClaim(str(row["job_id"]), attempt_id, token, owner)

    def _claim_owner(self) -> str:
        return "pid:" + str(os.getpid()) + ":" + uuid.uuid4().hex

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
                or int(job["lease_expires_at"]) < _now()
            ):
                raise SlowGraphError("job claim is no longer active")
            evidence_ids = _strict_json(
                job["evidence_ids_json"], label="job evidence IDs", expected=list
            )
            return (
                {
                    "region_key": job["region_key"],
                    "evidence": self._evidence(con, job["scope_id"], evidence_ids),
                },
                self._capsules(con, job["scope_id"], job["region_key"]),
            )

    def _renew_claim(self, claim: JobClaim) -> None:
        now = _now()
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            renewed = con.execute(
                "UPDATE slow_graph_jobs SET lease_expires_at=?,updated_at=? "
                "WHERE job_id=? AND status='pending' AND claim_token=? "
                "AND claim_owner=? AND lease_expires_at>=?",
                (
                    now + self.claim_lease_seconds,
                    now,
                    claim.job_id,
                    claim.token,
                    claim.owner,
                    now,
                ),
            )
            if renewed.rowcount != 1:
                raise SlowGraphError("job claim could not be renewed")

    def _propose_with_lease_heartbeat(
        self,
        claim: JobClaim,
        manager: PatchManager,
        region: Mapping[str, Any],
        capsules: list[dict[str, Any]],
    ) -> dict[str, Any]:
        stop = threading.Event()
        errors: list[Exception] = []
        interval = max(0.1, min(30.0, self.claim_lease_seconds / 3.0))

        def heartbeat() -> None:
            while not stop.wait(interval):
                try:
                    self._renew_claim(claim)
                except Exception as exc:
                    errors.append(exc)
                    stop.set()

        thread = threading.Thread(
            target=heartbeat,
            name=f"slow-graph-lease-{claim.job_id}",
            daemon=True,
        )
        thread.start()
        try:
            patch = manager.propose(region, capsules)
        finally:
            stop.set()
            thread.join(timeout=max(1.0, interval + 1.0))
        if thread.is_alive():
            raise SlowGraphError("slow graph claim heartbeat did not stop")
        if errors:
            raise SlowGraphError(f"slow graph claim heartbeat failed: {errors[0]}")
        self._renew_claim(claim)
        return patch

    def _finish_claim_failure(
        self, claim: JobClaim, manager: PatchManager, exc: Exception
    ) -> bool:
        status = (
            "retryable"
            if isinstance(exc, DeepSeekCallError) and exc.retryable
            else "failed"
        )
        now = _now()
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            failed_job = con.execute(
                "UPDATE slow_graph_jobs SET status=?,attempts=attempts+1,last_error=?,"
                "updated_at=?,claim_token=NULL,claim_owner=NULL,lease_expires_at=NULL "
                "WHERE job_id=? AND status='pending' AND claim_token=? "
                "AND claim_owner=? AND lease_expires_at>=?",
                (status, str(exc), now, claim.job_id, claim.token, claim.owner, now),
            )
            if failed_job.rowcount != 1:
                return False
            failed_attempt = con.execute(
                "UPDATE slow_graph_attempts SET status=?,call_metadata_json=?,error=?,"
                "completed_at=? WHERE attempt_id=? AND job_id=? AND claim_token=? "
                "AND claim_owner=? AND status='started'",
                (
                    status,
                    _json(dict(manager.last_call_metadata)),
                    str(exc),
                    now,
                    claim.attempt_id,
                    claim.job_id,
                    claim.token,
                    claim.owner,
                ),
            )
            if failed_attempt.rowcount != 1:
                raise SlowGraphError("claimed attempt is no longer active")
        return True

    def _run_claimed_job(self, claim: JobClaim, manager: PatchManager) -> str:
        try:
            region, capsules = self._claim_context(claim)
            patch = self._propose_with_lease_heartbeat(
                claim, manager, region, capsules
            )
            patch_id = self.apply_patch(
                claim.job_id,
                patch,
                manager_model=_required_text(
                    manager.model_config.get("model"), "manager model"
                ),
                call_metadata=manager.last_call_metadata,
                claim=claim,
            )
            return patch_id
        except Exception as exc:
            self._finish_claim_failure(claim, manager, exc)
            raise

    def run_job(self, job_id: str, manager: PatchManager) -> str:
        self.recover_interrupted_attempts()
        claim = self._claim_pending_job(job_id, owner=self._claim_owner())
        if claim is None:
            raise SlowGraphError("job is not pending or is already claimed")
        return self._run_claimed_job(claim, manager)

    def resume(self, job_id: str) -> None:
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT status,claim_token FROM slow_graph_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if (
                row is None
                or row["status"] not in {"failed", "retryable"}
                or row["claim_token"] is not None
            ):
                raise SlowGraphError(
                    "only failed or retryable jobs can be explicitly reopened"
                )
            reopened = con.execute(
                "UPDATE slow_graph_jobs SET status='pending',last_error='',updated_at=?,"
                "claim_token=NULL,claim_owner=NULL,lease_expires_at=NULL WHERE job_id=? "
                "AND status IN ('failed','retryable') AND claim_token IS NULL",
                (_now(), job_id),
            )
            if reopened.rowcount != 1:
                raise SlowGraphError("slow graph job changed while reopening")

    def recover_interrupted_attempts(self) -> int:
        now = _now()
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            rows = con.execute(
                "SELECT job_id,claim_token,claim_owner FROM slow_graph_jobs "
                "WHERE status='pending' AND claim_token IS NOT NULL "
                "AND lease_expires_at<?",
                (now,),
            ).fetchall()
            for row in rows:
                released = con.execute(
                    "UPDATE slow_graph_jobs SET status='failed',attempts=attempts+1,"
                    "last_error='claim lease expired; external call outcome uncertain; explicit resume required',"
                    "updated_at=?,claim_token=NULL,claim_owner=NULL,lease_expires_at=NULL "
                    "WHERE job_id=? AND status='pending' AND claim_token=? "
                    "AND claim_owner=? AND lease_expires_at<?",
                    (
                        now,
                        row["job_id"],
                        row["claim_token"],
                        row["claim_owner"],
                        now,
                    ),
                )
                if released.rowcount != 1:
                    raise SlowGraphError("expired job claim changed during recovery")
                expired_attempt = con.execute(
                    "UPDATE slow_graph_attempts SET status='expired',"
                    "error='claim lease expired; external call outcome uncertain; explicit resume required',completed_at=? "
                    "WHERE job_id=? AND claim_token=? AND claim_owner=? "
                    "AND status='started'",
                    (now, row["job_id"], row["claim_token"], row["claim_owner"]),
                )
                if expired_attempt.rowcount != 1:
                    raise SlowGraphError("expired job has no active claimed attempt")
        return len(rows)

    def drain(
        self, manager: PatchManager, *, batch_size: int | None = None
    ) -> list[str]:
        self.recover_interrupted_attempts()
        if batch_size is not None:
            if batch_size <= 0:
                raise SlowGraphError("batch_size must be positive")
        owner = self._claim_owner()
        result: list[str] = []
        while batch_size is None or len(result) < batch_size:
            claim = self._claim_pending_job(None, owner=owner)
            if claim is None:
                break
            result.append(self._run_claimed_job(claim, manager))
        return result

    def audit(self, scope_id: str) -> dict[str, int]:
        with self.connection() as con:
            self._records_table_exists(con)
            unfinished = con.execute(
                "SELECT job_id,status,last_error FROM slow_graph_jobs WHERE scope_id=? AND status!='completed' ORDER BY created_at",
                (scope_id,),
            ).fetchall()
            if unfinished:
                raise AuditError(
                    "scope has unfinished slow graph jobs: "
                    + _json([dict(item) for item in unfinished])
                )
            rows = con.execute(
                "SELECT memory_id,state,metadata_json FROM records WHERE scope_id=?",
                (scope_id,),
            ).fetchall()
            fast_state: dict[str, tuple[str, str]] = {}
            active_fast_by_slot: dict[str, set[str]] = {}
            for row in rows:
                row_meta = self._metadata(row, "record")
                if (
                    row_meta.get("content_variant") == LEAF_VARIANT
                    and row_meta.get("memory_layer") == "fast"
                ):
                    slot = _clean(row_meta.get("canonical_slot_key"))
                    state = _clean(row["state"])
                    fast_state[str(row["memory_id"])] = (state, slot)
                    if state in {"active", "parallel_active", "promoted"}:
                        active_fast_by_slot.setdefault(slot, set()).add(
                            str(row["memory_id"])
                        )
            capsules: dict[str, list[tuple[str, str, int, str]]] = {}
            for row in rows:
                meta = self._metadata(row, "record")
                if meta.get("content_variant") != CAPSULE_VARIANT:
                    continue
                capsule_id = _required_text(meta.get("capsule_id"), "capsule_id")
                revision = meta.get("revision")
                if not isinstance(revision, int) or revision < 1:
                    raise AuditError("capsule revision is invalid")
                status = _required_text(meta.get("status"), "capsule status")
                if status not in {"active", "challenged", "retired", "superseded"}:
                    raise AuditError("capsule status is invalid")
                capsules.setdefault(capsule_id, []).append(
                    (str(row["memory_id"]), str(row["state"]), revision, status)
                )
                claims = _validate_claims(meta.get("claims"), stored=True)
                for claim in claims:
                    if status in {"active", "challenged"}:
                        stale_support = {
                            evidence_id
                            for evidence_id in claim["support"]
                            if fast_state.get(evidence_id, ("", ""))[0]
                            not in {"active", "parallel_active", "promoted"}
                        }
                        current_replacements = active_fast_by_slot.get(
                            claim["canonical_slot"], set()
                        )
                        if (
                            stale_support
                            and current_replacements
                            and not current_replacements.intersection(
                                claim["counterevidence"]
                            )
                        ):
                            raise AuditError(
                                "active capsule claim promotes superseded fast evidence"
                            )
                    evidence = claim["support"] + claim["counterevidence"]
                    provenance = con.execute(
                        "SELECT evidence_memory_id,source_parent_json FROM slow_graph_provenance WHERE scope_id=? AND capsule_id=? AND revision=? AND claim_id=?",
                        (scope_id, capsule_id, revision, claim["claim_id"]),
                    ).fetchall()
                    if {item["evidence_memory_id"] for item in provenance} != set(
                        evidence
                    ):
                        raise AuditError(
                            "claim provenance does not match claim evidence"
                        )
                    for item in provenance:
                        parent = _strict_json(
                            item["source_parent_json"],
                            label="provenance source_parent",
                            expected=dict,
                        )
                        if set(parent) != {
                            "session_index",
                            "parent_chunk_index",
                            "message_index",
                            "source_record_id",
                            "event_id",
                            "evidence_char_start",
                            "evidence_char_end",
                        }:
                            raise AuditError("provenance is not a leaf source_parent")
                        leaf = con.execute(
                            "SELECT metadata_json FROM records WHERE scope_id=? AND memory_id=?",
                            (scope_id, item["evidence_memory_id"]),
                        ).fetchone()
                        if leaf is None:
                            raise AuditError("provenance references missing evidence")
                        leaf_meta = self._metadata(leaf, "provenance leaf")
                        expected_parent = {
                            "session_index": leaf_meta.get("session_index"),
                            "parent_chunk_index": leaf_meta.get("message_index"),
                            "message_index": leaf_meta.get("message_index"),
                            "source_record_id": leaf_meta.get("source_record_id"),
                            "event_id": leaf_meta.get("event_id"),
                            "evidence_char_start": leaf_meta.get(
                                "evidence_char_start"
                            ),
                            "evidence_char_end": leaf_meta.get("evidence_char_end"),
                        }
                        if (
                            leaf_meta.get("content_variant") != LEAF_VARIANT
                            or leaf_meta.get("memory_layer") != "fast"
                            or leaf_meta.get("node_kind") != "atomic_user_assertion"
                            or leaf_meta.get("atomic_evidence_leaf") is not True
                            or leaf_meta.get("authority") != "user_assertion"
                            or expected_parent != parent
                        ):
                            raise AuditError(
                                "provenance does not resolve to the cited fast leaf"
                            )
            for capsule_id, revisions in capsules.items():
                revision_numbers = sorted(item[2] for item in revisions)
                if revision_numbers != list(range(1, max(revision_numbers) + 1)):
                    raise AuditError("capsule revisions are not contiguous")
                latest = [item for item in revisions if item[2] == max(revision_numbers)]
                if len(latest) != 1:
                    raise AuditError("capsule lacks one latest revision")
                memory_id, state, _, status = latest[0]
                slot_key = "slow." + capsule_id
                head = con.execute(
                    "SELECT memory_id FROM slot_heads WHERE scope_id=? AND slot_key=?",
                    (scope_id, slot_key),
                ).fetchone()
                if status in {"active", "challenged"}:
                    if state != "active" or head is None or head["memory_id"] != memory_id:
                        raise AuditError("active/challenged capsule head is inconsistent")
                elif head is not None:
                    raise AuditError("inactive capsule must not retain a slot head")
                history = [
                    item["memory_id"]
                    for item in con.execute(
                        "SELECT memory_id FROM slot_history WHERE scope_id=? AND slot_key=? ORDER BY ordinal",
                        (scope_id, slot_key),
                    )
                ]
                expected_history = [
                    item[0] for item in sorted(revisions, key=lambda item: item[2])
                ]
                if history != expected_history:
                    raise AuditError("capsule slot history is inconsistent")
            edge_rows = con.execute(
                "SELECT source_memory_id,target_memory_id,edge_type,metadata_json FROM memory_edges WHERE scope_id=?",
                (scope_id,),
            ).fetchall()
            for edge in edge_rows:
                meta = _strict_json(
                    edge["metadata_json"], label="edge metadata", expected=dict
                )
                if meta.get("edge_source") != SLOW_EDGE_SOURCE:
                    continue
                if (
                    edge["source_memory_id"] == edge["target_memory_id"]
                    or edge["edge_type"] not in EDGE_TYPES
                    or not isinstance(meta.get("evidence_refs"), list)
                    or not _clean(meta.get("patch_id"))
                ):
                    raise AuditError("edge is inconsistent")
            return {
                "slow_graph_batches": int(
                    con.execute(
                        "SELECT COUNT(*) FROM slow_graph_batches WHERE scope_id=?",
                        (scope_id,),
                    ).fetchone()[0]
                ),
                "slow_graph_jobs": int(
                    con.execute(
                        "SELECT COUNT(*) FROM slow_graph_jobs WHERE scope_id=?",
                        (scope_id,),
                    ).fetchone()[0]
                ),
                "slow_graph_patches": int(
                    con.execute(
                        "SELECT COUNT(*) FROM slow_graph_patches WHERE scope_id=?",
                        (scope_id,),
                    ).fetchone()[0]
                ),
                "slow_graph_attempts": int(
                    con.execute(
                        "SELECT COUNT(*) FROM slow_graph_attempts WHERE scope_id=?",
                        (scope_id,),
                    ).fetchone()[0]
                ),
                "slow_graph_patch_operations": int(
                    con.execute(
                        "SELECT COUNT(*) FROM slow_graph_patch_operations o JOIN slow_graph_patches p ON p.patch_id=o.patch_id WHERE p.scope_id=?",
                        (scope_id,),
                    ).fetchone()[0]
                ),
                "slow_graph_provenance": int(
                    con.execute(
                        "SELECT COUNT(*) FROM slow_graph_provenance WHERE scope_id=?",
                        (scope_id,),
                    ).fetchone()[0]
                ),
                "memory_edges": sum(
                    1
                    for edge in edge_rows
                    if _strict_json(
                        edge["metadata_json"], label="edge metadata", expected=dict
                    ).get("edge_source")
                    == SLOW_EDGE_SOURCE
                ),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="TMCRA slow graph controller")
    parser.add_argument("database", type=Path)
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="TMCRA repository containing the real graph schema",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("scope_id")
    enqueue.add_argument("--region")
    drain = sub.add_parser("drain")
    drain.add_argument("--batch-size", type=int)
    run = sub.add_parser("run")
    run.add_argument("job_id")
    resume = sub.add_parser("resume")
    resume.add_argument("job_id")
    audit = sub.add_parser("audit")
    audit.add_argument("scope_id")
    args = parser.parse_args()
    store = SlowGraphStore(args.database, schema=load_graph_schema(args.repo))
    if args.command == "enqueue":
        manager = DeepSeekProGraphPatchManager(DeepSeekProConfig.from_env())
        if args.region:
            region = store.fast_regions(args.scope_id).get(args.region, [])
            result = [
                store.enqueue(
                    args.scope_id,
                    args.region,
                    (item["memory_id"] for item in region),
                    manager=manager,
                )
            ]
        else:
            result = store.enqueue_regions(args.scope_id, manager=manager)
    elif args.command == "resume":
        store.resume(args.job_id)
        result = {"job_id": args.job_id, "status": "pending"}
    elif args.command == "audit":
        result = store.audit(args.scope_id)
    else:
        manager = DeepSeekProGraphPatchManager(DeepSeekProConfig.from_env())
        result = (
            store.drain(manager, batch_size=args.batch_size)
            if args.command == "drain"
            else store.run_job(args.job_id, manager)
        )
    print(_json(result))


if __name__ == "__main__":
    main()
