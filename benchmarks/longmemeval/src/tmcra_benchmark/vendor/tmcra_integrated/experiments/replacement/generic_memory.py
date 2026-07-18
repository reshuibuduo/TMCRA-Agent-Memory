from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import re
import sqlite3
import time
import uuid
from typing import Any, Dict, Iterable, List, Sequence


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize(value: Any) -> str:
    return _clean_text(value).lower()


def _dedupe(items: Iterable[Any]) -> List[str]:
    results: List[str] = []
    seen = set()
    for item in items:
        text = _clean_text(item)
        if not text:
            continue
        key = _normalize(text)
        if key in seen:
            continue
        seen.add(key)
        results.append(text)
    return results


def _tokenize(value: Any) -> List[str]:
    text = _normalize(value)
    if not text:
        return []
    english = re.findall(r"[a-z0-9_.-]+", text)
    cjk = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    if english or cjk:
        return _dedupe([*english, *cjk])
    return [char for char in text if char.strip()]


def _estimate_tokens(value: Any) -> int:
    text = _clean_text(value)
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


@dataclass(slots=True)
class MemorySessionScope:
    scope_id: str
    namespace: str = "default"
    profile_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "namespace": self.namespace,
            "profile_name": self.profile_name,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class MemoryPolicy:
    overwrite_mode: str = "slot_latest"
    dedupe_mode: str = "key_value_active"
    retention_mode: str = "bounded_audit"
    writeback_mode: str = "explicit_and_derived"
    conflict_mode: str = "keep_history"
    audit_retention: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overwrite_mode": self.overwrite_mode,
            "dedupe_mode": self.dedupe_mode,
            "retention_mode": self.retention_mode,
            "writeback_mode": self.writeback_mode,
            "conflict_mode": self.conflict_mode,
            "audit_retention": int(self.audit_retention),
        }


@dataclass(slots=True)
class MemoryRecord:
    record_id: str
    namespace: str
    type: str
    key: str
    value: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    state: str = "active"
    supersedes: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    timestamps: Dict[str, Any] = field(default_factory=dict)
    scope_id: str = ""

    def search_text(self) -> str:
        payload = {
            "namespace": self.namespace,
            "type": self.type,
            "key": self.key,
            "value": self.value,
            "attributes": self.attributes,
            "provenance": self.provenance,
        }
        return _json_dumps(payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "namespace": self.namespace,
            "type": self.type,
            "key": self.key,
            "value": self.value,
            "attributes": dict(self.attributes),
            "state": self.state,
            "supersedes": list(self.supersedes),
            "provenance": dict(self.provenance),
            "timestamps": dict(self.timestamps),
            "scope_id": self.scope_id,
        }


@dataclass(slots=True)
class MemoryRetrievalEntry:
    record: MemoryRecord
    score: float = 0.0
    matched_terms: List[str] = field(default_factory=list)
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "score": round(float(self.score), 6),
            "matched_terms": list(self.matched_terms),
            "rank": int(self.rank),
        }


@dataclass(slots=True)
class MemoryRetrievalResult:
    records: List[MemoryRetrievalEntry] = field(default_factory=list)
    active_records: List[MemoryRetrievalEntry] = field(default_factory=list)
    historical_records: List[MemoryRetrievalEntry] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "records": [item.to_dict() for item in self.records],
            "active_records": [item.to_dict() for item in self.active_records],
            "historical_records": [item.to_dict() for item in self.historical_records],
            "relations": list(self.relations),
            "retrieval_metadata": dict(self.retrieval_metadata),
        }


@dataclass(slots=True)
class MemoryWritebackRequest:
    writeback_type: str = "derived_writeback"
    namespace: str = "default"
    query_text: str = ""
    answer_text: str = ""
    records: List[Dict[str, Any]] = field(default_factory=list)
    supports: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "writeback_type": self.writeback_type,
            "namespace": self.namespace,
            "query_text": self.query_text,
            "answer_text": self.answer_text,
            "records": [dict(item) for item in self.records],
            "supports": [dict(item) for item in self.supports],
            "metadata": dict(self.metadata),
        }


class MemoryProfile(ABC):
    profile_name: str = "generic"

    def policy(self) -> MemoryPolicy:
        return MemoryPolicy()

    def normalize_event(self, event_kind: str, payload: Dict[str, Any] | None, scope: MemorySessionScope) -> Dict[str, Any]:
        _ = event_kind, scope
        normalized = dict(payload or {})
        raw_records = normalized.get("records")
        if raw_records is None:
            raw_records = normalized.get("writeback_records")
        if raw_records is None:
            raw_records = normalized.get("replacement_memory_records")
        if isinstance(raw_records, dict):
            raw_records = [raw_records]
        normalized["records"] = [dict(item) for item in list(raw_records or []) if isinstance(item, dict)]
        normalized["metadata"] = dict(normalized.get("metadata") or {})
        return normalized

    @abstractmethod
    def derive_records(self, event_kind: str, payload: Dict[str, Any], scope: MemorySessionScope) -> List[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def resolve_key(
        self,
        record_type: str,
        value: str,
        attributes: Dict[str, Any],
        *,
        namespace: str,
        provided_key: str = "",
        relation: str = "",
        provenance: Dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError

    def infer_query_types(self, query: str) -> List[str]:
        _ = query
        return []

    def record_search_text(self, record: MemoryRecord) -> str:
        return record.search_text()

    def score_bonus(self, query: str, record: MemoryRecord) -> float:
        _ = query, record
        return 0.0

    def build_relations(self, retrieval: MemoryRetrievalResult) -> List[Dict[str, Any]]:
        relations: List[Dict[str, Any]] = []
        seen = set()
        for entry in retrieval.records:
            record = entry.record
            for superseded_id in record.supersedes:
                key = ("supersedes", superseded_id, record.record_id)
                if key in seen:
                    continue
                seen.add(key)
                relations.append(
                    {
                        "from": superseded_id,
                        "to": record.record_id,
                        "relation": "supersedes",
                        "score": round(float(entry.score), 6),
                    }
                )
            relation = _clean_text(record.attributes.get("relation", ""))
            anchors = [anchor for anchor in list(record.attributes.get("anchors", []) or []) if _clean_text(anchor)]
            if relation and anchors:
                key = (relation, anchors[0], record.value)
                if key in seen:
                    continue
                seen.add(key)
                relations.append(
                    {
                        "from": anchors[0],
                        "to": record.value,
                        "relation": relation,
                        "score": round(float(entry.score), 6),
                    }
                )
        return relations

    def render_context(self, query: str, retrieval: MemoryRetrievalResult, scope: MemorySessionScope) -> Dict[str, Any]:
        lines = [f"[{item.record.type}] {item.record.key}: {item.record.value}" for item in retrieval.records[:8]]
        return {
            "profile_name": self.profile_name,
            "query": query,
            "scope": scope.to_dict(),
            "lines": lines,
            "record_ids": [item.record.record_id for item in retrieval.records],
        }


class MemoryStore(ABC):
    @abstractmethod
    def next_turn_index(self, scope: MemorySessionScope) -> int:
        raise NotImplementedError

    @abstractmethod
    def save_records(self, scope: MemorySessionScope, records: Sequence[MemoryRecord], policy: MemoryPolicy) -> List[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def list_records(self, scope: MemorySessionScope) -> List[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def append_support(self, scope: MemorySessionScope, *, answer_id: str, record_ids: Sequence[str], payload: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_audit(self, scope: MemorySessionScope, *, event_kind: str, payload: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def scope_stats(self, scope: MemorySessionScope) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def storage_bytes(self, scope: MemorySessionScope) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset_scope(self, scope: MemorySessionScope) -> None:
        raise NotImplementedError


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, storage_path: str | Path, *, audit_retention: int = 256) -> None:
        self.storage_path = Path(storage_path).expanduser().resolve()
        self.audit_retention = max(1, int(audit_retention))
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.storage_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    @contextmanager
    def _managed_connection(self) -> Iterable[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self._managed_connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_scope_meta (
                    scope_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    profile_name TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    last_turn_index INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_records (
                    scope_id TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    attributes_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    supersedes_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    timestamps_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (scope_id, record_id)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_records_scope_key_state
                    ON memory_records(scope_id, namespace, key, state, updated_at);
                CREATE INDEX IF NOT EXISTS idx_memory_records_scope_type
                    ON memory_records(scope_id, type, updated_at);
                CREATE TABLE IF NOT EXISTS memory_support_log (
                    scope_id TEXT NOT NULL,
                    support_id TEXT NOT NULL,
                    answer_id TEXT NOT NULL,
                    record_ids_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (scope_id, support_id)
                );
                CREATE TABLE IF NOT EXISTS memory_audit (
                    scope_id TEXT NOT NULL,
                    event_kind TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (scope_id, event_id)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_audit_scope_kind_created
                    ON memory_audit(scope_id, event_kind, created_at);
                CREATE TABLE IF NOT EXISTS memory_audit_meta (
                    scope_id TEXT NOT NULL,
                    event_kind TEXT NOT NULL,
                    total_count INTEGER NOT NULL DEFAULT 0,
                    trimmed_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (scope_id, event_kind)
                );
                """
            )

    def _ensure_scope_meta(self, connection: sqlite3.Connection, scope: MemorySessionScope) -> None:
        now = time.time()
        connection.execute(
            """
            INSERT INTO memory_scope_meta(scope_id, namespace, profile_name, metadata_json, last_turn_index, created_at, updated_at)
            VALUES(?, ?, ?, ?, 0, ?, ?)
            ON CONFLICT(scope_id) DO UPDATE SET
                namespace = excluded.namespace,
                profile_name = CASE
                    WHEN excluded.profile_name != '' THEN excluded.profile_name
                    ELSE memory_scope_meta.profile_name
                END,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                scope.scope_id,
                _clean_text(scope.namespace) or "default",
                _clean_text(scope.profile_name),
                _json_dumps(scope.metadata),
                float(now),
                float(now),
            ),
        )

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            record_id=str(row["record_id"]),
            namespace=str(row["namespace"]),
            type=str(row["type"]),
            key=str(row["key"]),
            value=str(row["value"]),
            attributes=json.loads(row["attributes_json"] or "{}"),
            state=str(row["state"]),
            supersedes=list(json.loads(row["supersedes_json"] or "[]")),
            provenance=json.loads(row["provenance_json"] or "{}"),
            timestamps=json.loads(row["timestamps_json"] or "{}"),
            scope_id=str(row["scope_id"]),
        )

    def next_turn_index(self, scope: MemorySessionScope) -> int:
        with self._managed_connection() as connection:
            self._ensure_scope_meta(connection, scope)
            row = connection.execute(
                "SELECT last_turn_index FROM memory_scope_meta WHERE scope_id = ?",
                (scope.scope_id,),
            ).fetchone()
            next_turn = int(row["last_turn_index"] or 0) + 1 if row is not None else 1
            connection.execute(
                "UPDATE memory_scope_meta SET last_turn_index = ?, updated_at = ? WHERE scope_id = ?",
                (int(next_turn), float(time.time()), scope.scope_id),
            )
            return int(next_turn)

    def save_records(self, scope: MemorySessionScope, records: Sequence[MemoryRecord], policy: MemoryPolicy) -> List[MemoryRecord]:
        stored: List[MemoryRecord] = []
        with self._managed_connection() as connection:
            self._ensure_scope_meta(connection, scope)
            for original in records:
                record = replace(
                    original,
                    scope_id=scope.scope_id,
                    namespace=_clean_text(original.namespace) or _clean_text(scope.namespace) or "default",
                    state=_clean_text(original.state) or "active",
                )
                if _normalize(policy.dedupe_mode) == "key_value_active":
                    existing = connection.execute(
                        """
                        SELECT * FROM memory_records
                        WHERE scope_id = ? AND namespace = ? AND key = ? AND value = ? AND state = 'active'
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        (scope.scope_id, record.namespace, record.key, record.value),
                    ).fetchone()
                    if existing is not None and _normalize(existing["type"]) == _normalize(record.type):
                        stored.append(self._row_to_record(existing))
                        continue
                superseded_ids: List[str] = list(record.supersedes)
                if _normalize(record.state) == "active" and _normalize(policy.overwrite_mode) in {"latest", "slot_latest"}:
                    active_rows = connection.execute(
                        """
                        SELECT record_id FROM memory_records
                        WHERE scope_id = ? AND namespace = ? AND key = ? AND state = 'active'
                        ORDER BY updated_at DESC
                        """,
                        (scope.scope_id, record.namespace, record.key),
                    ).fetchall()
                    active_ids = [str(row["record_id"]) for row in active_rows if _clean_text(row["record_id"])]
                    superseded_ids.extend(active_ids)
                    if active_ids:
                        placeholders = ", ".join("?" for _ in active_ids)
                        connection.execute(
                            f"""
                            UPDATE memory_records
                            SET state = 'superseded', updated_at = ?
                            WHERE scope_id = ? AND record_id IN ({placeholders})
                            """,
                            (float(time.time()), scope.scope_id, *active_ids),
                        )
                created_at = _to_float(record.timestamps.get("created_at"), time.time())
                updated_at = _to_float(record.timestamps.get("updated_at"), created_at)
                record = replace(
                    record,
                    supersedes=_dedupe(superseded_ids),
                    timestamps={
                        **dict(record.timestamps),
                        "created_at": created_at,
                        "updated_at": updated_at,
                    },
                )
                connection.execute(
                    """
                    INSERT OR REPLACE INTO memory_records(
                        scope_id, record_id, namespace, type, key, value,
                        attributes_json, state, supersedes_json, provenance_json,
                        timestamps_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scope.scope_id,
                        record.record_id,
                        record.namespace,
                        record.type,
                        record.key,
                        record.value,
                        _json_dumps(record.attributes),
                        record.state,
                        _json_dumps(record.supersedes),
                        _json_dumps(record.provenance),
                        _json_dumps(record.timestamps),
                        float(created_at),
                        float(updated_at),
                    ),
                )
                stored.append(record)
            connection.execute(
                "UPDATE memory_scope_meta SET profile_name = ?, updated_at = ? WHERE scope_id = ?",
                (_clean_text(scope.profile_name), float(time.time()), scope.scope_id),
            )
        return stored

    def list_records(self, scope: MemorySessionScope) -> List[MemoryRecord]:
        with self._managed_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_records WHERE scope_id = ? ORDER BY created_at ASC, record_id ASC",
                (scope.scope_id,),
            ).fetchall()
        records = [self._row_to_record(row) for row in rows]
        records.sort(
            key=lambda item: (
                _to_int(item.timestamps.get("turn_index"), 0),
                _to_float(item.timestamps.get("created_at"), 0.0),
                item.record_id,
            )
        )
        return records

    def append_support(self, scope: MemorySessionScope, *, answer_id: str, record_ids: Sequence[str], payload: Dict[str, Any]) -> None:
        with self._managed_connection() as connection:
            self._ensure_scope_meta(connection, scope)
            connection.execute(
                """
                INSERT INTO memory_support_log(scope_id, support_id, answer_id, record_ids_json, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    scope.scope_id,
                    uuid.uuid4().hex,
                    _clean_text(answer_id),
                    _json_dumps(_dedupe(record_ids)),
                    _json_dumps(payload),
                    float(time.time()),
                ),
            )

    def append_audit(self, scope: MemorySessionScope, *, event_kind: str, payload: Dict[str, Any]) -> None:
        normalized_kind = _clean_text(event_kind) or "event"
        with self._managed_connection() as connection:
            self._ensure_scope_meta(connection, scope)
            connection.execute(
                """
                INSERT INTO memory_audit(scope_id, event_kind, event_id, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (scope.scope_id, normalized_kind, uuid.uuid4().hex, _json_dumps(payload), float(time.time())),
            )
            connection.execute(
                """
                INSERT INTO memory_audit_meta(scope_id, event_kind, total_count, trimmed_count)
                VALUES (?, ?, 1, 0)
                ON CONFLICT(scope_id, event_kind) DO UPDATE SET
                    total_count = memory_audit_meta.total_count + 1
                """,
                (scope.scope_id, normalized_kind),
            )
            retained = connection.execute(
                "SELECT COUNT(*) AS retained FROM memory_audit WHERE scope_id = ? AND event_kind = ?",
                (scope.scope_id, normalized_kind),
            ).fetchone()
            overflow = max(0, int(retained["retained"] if retained is not None else 0) - self.audit_retention)
            if overflow:
                rows = connection.execute(
                    """
                    SELECT event_id FROM memory_audit
                    WHERE scope_id = ? AND event_kind = ?
                    ORDER BY created_at ASC, event_id ASC
                    LIMIT ?
                    """,
                    (scope.scope_id, normalized_kind, int(overflow)),
                ).fetchall()
                event_ids = [str(row["event_id"]) for row in rows]
                if event_ids:
                    placeholders = ", ".join("?" for _ in event_ids)
                    connection.execute(
                        f"DELETE FROM memory_audit WHERE scope_id = ? AND event_id IN ({placeholders})",
                        (scope.scope_id, *event_ids),
                    )
                    connection.execute(
                        """
                        UPDATE memory_audit_meta
                        SET trimmed_count = trimmed_count + ?
                        WHERE scope_id = ? AND event_kind = ?
                        """,
                        (int(overflow), scope.scope_id, normalized_kind),
                    )

    def scope_stats(self, scope: MemorySessionScope) -> Dict[str, Any]:
        records = self.list_records(scope)
        active_records = [record for record in records if _normalize(record.state) == "active"]
        historical_records = [record for record in records if _normalize(record.state) != "active"]
        with self._managed_connection() as connection:
            support_row = connection.execute(
                "SELECT COUNT(*) AS count FROM memory_support_log WHERE scope_id = ?",
                (scope.scope_id,),
            ).fetchone()
            meta_rows = connection.execute(
                "SELECT event_kind, total_count, trimmed_count FROM memory_audit_meta WHERE scope_id = ?",
                (scope.scope_id,),
            ).fetchall()
            retained_rows = connection.execute(
                "SELECT event_kind, COUNT(*) AS retained FROM memory_audit WHERE scope_id = ? GROUP BY event_kind",
                (scope.scope_id,),
            ).fetchall()
            meta_row = connection.execute(
                "SELECT profile_name, last_turn_index FROM memory_scope_meta WHERE scope_id = ?",
                (scope.scope_id,),
            ).fetchone()
        retained_lookup = {str(row["event_kind"]): int(row["retained"]) for row in retained_rows}
        audit_totals = {str(row["event_kind"]): int(row["total_count"]) for row in meta_rows}
        audit_trimmed = {str(row["event_kind"]): int(row["trimmed_count"]) for row in meta_rows}
        support_count = int(support_row["count"] if support_row is not None else 0)
        payload = {
            "records": [record.to_dict() for record in records],
            "support_events": support_count,
            "audit_totals": audit_totals,
            "audit_trimmed": audit_trimmed,
        }
        return {
            "records": len(records),
            "active_records": len(active_records),
            "historical_records": len(historical_records),
            "support_events": support_count,
            "audit_totals": audit_totals,
            "audit_trimmed": audit_trimmed,
            "audit_retained": retained_lookup,
            "last_turn_index": int(meta_row["last_turn_index"] if meta_row is not None else 0),
            "profile_name": _clean_text(meta_row["profile_name"] if meta_row is not None else scope.profile_name),
            "state_token_estimate": _estimate_tokens(_json_dumps(payload)),
        }

    def storage_bytes(self, scope: MemorySessionScope) -> int:
        payload = {
            "records": [record.to_dict() for record in self.list_records(scope)],
            "scope": scope.to_dict(),
        }
        return len(_json_dumps(payload).encode("utf-8"))

    def reset_scope(self, scope: MemorySessionScope) -> None:
        with self._managed_connection() as connection:
            connection.execute("DELETE FROM memory_records WHERE scope_id = ?", (scope.scope_id,))
            connection.execute("DELETE FROM memory_support_log WHERE scope_id = ?", (scope.scope_id,))
            connection.execute("DELETE FROM memory_audit WHERE scope_id = ?", (scope.scope_id,))
            connection.execute("DELETE FROM memory_audit_meta WHERE scope_id = ?", (scope.scope_id,))
            connection.execute("DELETE FROM memory_scope_meta WHERE scope_id = ?", (scope.scope_id,))


class MemoryRuntime:
    def __init__(
        self,
        *,
        profile: MemoryProfile,
        store: MemoryStore | None = None,
        storage_backend: str = "sqlite",
        storage_path: str = "",
        scope: MemorySessionScope | None = None,
        audit_retention: int = 256,
    ) -> None:
        self.profile = profile
        self.storage_backend = _normalize(storage_backend) or "sqlite"
        if store is not None:
            self.store = store
        elif self.storage_backend == "sqlite":
            resolved_path = _clean_text(storage_path) or str((Path.cwd() / "outputs" / "generic_memory.sqlite3").resolve())
            self.store = SQLiteMemoryStore(resolved_path, audit_retention=audit_retention)
        else:
            raise ValueError(f"Unsupported storage backend: {storage_backend}")
        self.storage_path = str(getattr(self.store, "storage_path", _clean_text(storage_path)))
        self.default_scope = self._resolve_scope(scope or MemorySessionScope(scope_id="default", namespace="default", profile_name=self.profile.profile_name))
        self._last_retrieval_tokens: Dict[str, int] = {}

    def _resolve_scope(self, session_scope: MemorySessionScope | str | None) -> MemorySessionScope:
        if isinstance(session_scope, MemorySessionScope):
            scope_id = _clean_text(session_scope.scope_id) or "default"
            return MemorySessionScope(
                scope_id=scope_id,
                namespace=_clean_text(session_scope.namespace) or "default",
                profile_name=_clean_text(session_scope.profile_name) or self.profile.profile_name,
                metadata=dict(session_scope.metadata),
            )
        if isinstance(session_scope, str) and _clean_text(session_scope):
            return MemorySessionScope(scope_id=_clean_text(session_scope), namespace="default", profile_name=self.profile.profile_name)
        if hasattr(self, "default_scope"):
            return MemorySessionScope(
                scope_id=self.default_scope.scope_id,
                namespace=self.default_scope.namespace,
                profile_name=self.profile.profile_name,
                metadata=dict(self.default_scope.metadata),
            )
        return MemorySessionScope(scope_id="default", namespace="default", profile_name=self.profile.profile_name)

    def ingest_event(self, event_kind: str, payload: Dict[str, Any] | None, session_scope: MemorySessionScope | str | None = None) -> List[MemoryRecord]:
        scope = self._resolve_scope(session_scope)
        turn_index = self.store.next_turn_index(scope)
        created_at = time.time()
        normalized = self.profile.normalize_event(event_kind, payload or {}, scope)
        normalized.update({"turn_index": int(turn_index), "created_at": float(created_at)})
        records = self.profile.derive_records(event_kind, normalized, scope)
        stored = self.store.save_records(scope, records, self.profile.policy())
        self.store.append_audit(
            scope,
            event_kind=event_kind,
            payload={
                "event_kind": event_kind,
                "turn_index": int(turn_index),
                "record_ids": [record.record_id for record in stored],
                "metadata": dict(normalized.get("metadata") or {}),
            },
        )
        return stored

    def retrieve(self, query: str, session_scope: MemorySessionScope | str | None = None, *, top_k: int = 6) -> MemoryRetrievalResult:
        scope = self._resolve_scope(session_scope)
        records = self.store.list_records(scope)
        query_tokens = set(_tokenize(query))
        query_types = {_normalize(item) for item in self.profile.infer_query_types(query)}
        scored: List[MemoryRetrievalEntry] = []
        for record in records:
            record_tokens = set(_tokenize(self.profile.record_search_text(record)))
            overlap = len(query_tokens & record_tokens) if query_tokens and record_tokens else 0
            union = len(query_tokens | record_tokens) if query_tokens and record_tokens else 0
            score = overlap / max(1, union) if union else 0.0
            if query_types and _normalize(record.type) in query_types:
                score += 0.18
            if _normalize(record.state) == "active":
                score += 0.12
            if record.key and _normalize(record.key) in _normalize(query):
                score += 0.08
            score += float(self.profile.score_bonus(query, record) or 0.0)
            score += min(0.08, float(_to_int(record.timestamps.get("turn_index"), 0)) * 0.0005)
            if score > 0:
                scored.append(
                    MemoryRetrievalEntry(
                        record=record,
                        score=float(score),
                        matched_terms=sorted(query_tokens & record_tokens)[:8],
                    )
                )
        if not scored:
            fallback = list(records)
            fallback.sort(
                key=lambda item: (
                    _normalize(item.state) == "active",
                    _to_int(item.timestamps.get("turn_index"), 0),
                    _to_float(item.timestamps.get("created_at"), 0.0),
                ),
                reverse=True,
            )
            scored = [
                MemoryRetrievalEntry(record=item, score=(0.15 if _normalize(item.state) == "active" else 0.05), matched_terms=[])
                for item in fallback[: max(1, int(top_k))]
            ]
        scored.sort(
            key=lambda item: (
                float(item.score),
                _normalize(item.record.state) == "active",
                _to_int(item.record.timestamps.get("turn_index"), 0),
                item.record.record_id,
            ),
            reverse=True,
        )
        selected = [replace(item, rank=index + 1) for index, item in enumerate(scored[: max(1, int(top_k))])]
        retrieval = MemoryRetrievalResult(
            records=selected,
            active_records=[item for item in selected if _normalize(item.record.state) == "active"],
            historical_records=[item for item in selected if _normalize(item.record.state) != "active"],
            retrieval_metadata={
                "profile_name": self.profile.profile_name,
                "scope_id": scope.scope_id,
                "namespace": scope.namespace,
                "query": query,
                "query_types": sorted(query_types),
                "total_records": len(records),
                "selected_records": len(selected),
                "context_token_estimate": _estimate_tokens(_json_dumps([item.to_dict() for item in selected])),
                "persistence_backend": self.storage_backend,
                "persistence_path": self.storage_path,
            },
        )
        retrieval.relations = self.profile.build_relations(retrieval)
        self._last_retrieval_tokens[scope.scope_id] = int(retrieval.retrieval_metadata.get("context_token_estimate", 0) or 0)
        self.store.append_audit(
            scope,
            event_kind="retrieve",
            payload={
                "query": query,
                "query_types": sorted(query_types),
                "selected_record_ids": [item.record.record_id for item in selected],
            },
        )
        return retrieval

    def register_support(
        self,
        *,
        answer_id: str,
        supports: Sequence[Any],
        session_scope: MemorySessionScope | str | None = None,
        answer_text: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        scope = self._resolve_scope(session_scope)
        record_ids: List[str] = []
        for item in supports:
            if isinstance(item, MemoryRecord):
                record_ids.append(item.record_id)
            elif isinstance(item, MemoryRetrievalEntry):
                record_ids.append(item.record.record_id)
            elif isinstance(item, dict):
                record_id = _clean_text(item.get("record_id", "")) or _clean_text(item.get("memory_id", ""))
                if record_id:
                    record_ids.append(record_id)
            else:
                record_id = _clean_text(item)
                if record_id:
                    record_ids.append(record_id)
        payload = {
            "answer_id": _clean_text(answer_id),
            "record_ids": _dedupe(record_ids),
            "answer_text": _clean_text(answer_text),
            "metadata": dict(metadata or {}),
        }
        self.store.append_support(scope, answer_id=answer_id, record_ids=record_ids, payload=payload)
        self.store.append_audit(scope, event_kind="support", payload=payload)

    def writeback(self, request: MemoryWritebackRequest, session_scope: MemorySessionScope | str | None = None) -> List[MemoryRecord]:
        payload = request.to_dict()
        payload["metadata"] = {
            **dict(request.metadata),
            "writeback_type": request.writeback_type,
            "supports": [dict(item) for item in request.supports],
            "query_text": request.query_text,
            "answer_text": request.answer_text,
        }
        return self.ingest_event(request.writeback_type, payload, session_scope=session_scope)

    def reset_scope(self, session_scope: MemorySessionScope | str | None = None) -> None:
        scope = self._resolve_scope(session_scope)
        self.store.reset_scope(scope)
        self._last_retrieval_tokens.pop(scope.scope_id, None)

    def stats(self, session_scope: MemorySessionScope | str | None = None) -> Dict[str, Any]:
        scope = self._resolve_scope(session_scope)
        stats = self.store.scope_stats(scope)
        return {
            "profile_name": self.profile.profile_name,
            "scope_id": scope.scope_id,
            "namespace": scope.namespace,
            "persistence_backend": self.storage_backend,
            "persistence_path": self.storage_path,
            "storage_bytes": int(self.store.storage_bytes(scope)),
            "context_token_estimate": int(self._last_retrieval_tokens.get(scope.scope_id, 0)),
            "total_state_token_estimate": int(stats.get("state_token_estimate", 0)),
            **stats,
        }

    def before_llm(self, query: str, session_scope: MemorySessionScope | str | None = None, *, top_k: int = 6) -> Dict[str, Any]:
        scope = self._resolve_scope(session_scope)
        retrieval = self.retrieve(query, session_scope=scope, top_k=top_k)
        return {
            "query": query,
            "scope": scope.to_dict(),
            "retrieval": retrieval.to_dict(),
            "context": self.profile.render_context(query, retrieval, scope),
        }

    def after_user_turn(self, turn_payload: Dict[str, Any] | None, session_scope: MemorySessionScope | str | None = None) -> List[MemoryRecord]:
        return self.ingest_event("user_turn", turn_payload or {}, session_scope=session_scope)

    def after_llm_answer(
        self,
        answer_payload: Dict[str, Any] | None,
        supports: Sequence[Any] | None = None,
        session_scope: MemorySessionScope | str | None = None,
    ) -> Dict[str, Any]:
        payload = dict(answer_payload or {})
        answer_id = _clean_text(payload.get("answer_id", "")) or f"answer:{uuid.uuid4().hex}"
        support_items = list(supports or payload.get("supports", []) or [])
        if support_items:
            self.register_support(
                answer_id=answer_id,
                supports=support_items,
                session_scope=session_scope,
                answer_text=_clean_text(payload.get("answer_text", "")),
                metadata=dict(payload.get("metadata") or {}),
            )
        raw_records = payload.get("writeback_records")
        if raw_records is None:
            raw_records = payload.get("records")
        if isinstance(raw_records, dict):
            raw_records = [raw_records]
        stored: List[MemoryRecord] = []
        if raw_records:
            request = MemoryWritebackRequest(
                writeback_type=_clean_text(payload.get("writeback_type", "")) or "derived_writeback",
                namespace=_clean_text(payload.get("namespace", "")) or self._resolve_scope(session_scope).namespace,
                query_text=_clean_text(payload.get("query_text", "")),
                answer_text=_clean_text(payload.get("answer_text", "")),
                records=[dict(item) for item in list(raw_records or []) if isinstance(item, dict)],
                supports=[dict(item) for item in list(payload.get("supports", []) or []) if isinstance(item, dict)],
                metadata=dict(payload.get("metadata") or {}),
            )
            stored = self.writeback(request, session_scope=session_scope)
        return {
            "answer_id": answer_id,
            "stored_records": [record.to_dict() for record in stored],
            "support_count": len(support_items),
        }
