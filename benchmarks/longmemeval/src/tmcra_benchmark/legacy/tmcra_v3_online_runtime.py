#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from build_v3_runtime_dataset import covered_windows, rrank
from tmcra_v2_lme_pipeline import BgeM3DenseVectorizer
from tmcra_v3_reranker import ChannelAwareMemoryReranker
from tmcra_model_assets import load_optional_hf_model_manifest
from tmcra_v3_schema import CHANNEL_NAMES, SCHEMA_VERSION, clean_text, write_jsonl
from tmcra_v3_recall_planner import (
    DEEPSEEK_FLASH_MODEL,
    DeepSeekFlashRecallPlanner,
    RecallPlannerError,
    apply_recall_plan,
)


LEGACY_CHUNK_ID_RE = re.compile(r"^s(?P<session>\d+)_c(?P<parent>\d+)$")
MESSAGE_ID_RE = re.compile(r"^s(?P<session>\d+)_m(?P<parent>\d+)$")
EVENT_PARENT_RE = re.compile(r":s(?P<session>\d+)_[cm](?P<parent>\d+)$")
CURRENT_FAST_STATES = frozenset(
    {"active", "parallel_active", "promoted", "challenged"}
)
ONLINE_INDEX_SCHEMA_VERSION = "tmcra.v3.online-index.3"
FAST_SEMANTIC_STATE_POLICY = (
    "current-fast-states-v1:" + ",".join(sorted(CURRENT_FAST_STATES))
)
INGEST_PREFIX_RE = re.compile(r"^\[[^\]\n]+\]\s+user:\s*", flags=re.IGNORECASE)
SESSION_HEADER_RE = re.compile(
    r"^LongMemEval session_id=(?P<session_id>\S+) date=(?P<date>.+?)(?: continued=true)? \[",
    flags=re.DOTALL,
)


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


def atomic_torch_save(payload: Mapping[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, target)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_native_harness(path: Path, repo: Path):
    resolved_repo = str(repo.resolve())
    if resolved_repo not in sys.path:
        sys.path.insert(0, resolved_repo)
    module_name = "tmcra_v3_native_runtime_harness"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import native TMCRA harness: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def graph_runtime_env(args: argparse.Namespace) -> dict[str, str]:
    values = {
        "TMCRA_RETRIEVAL_MODE": "hybrid_node_scored",
        "TMCRA_NODE_MODEL_PATH": str(Path(args.node_model).resolve()),
        "TMCRA_PATH_MODEL_PATH": str(Path(args.path_model).resolve()),
        "TMCRA_NODE_MODEL_DEVICE": args.graph_device,
        "TMCRA_SUPPORT_PATH_K": str(args.support_path_k),
        "TMCRA_PATH_TUNNEL_RESCUE_K": str(args.path_tunnel_rescue_k),
        "TMCRA_CANDIDATE_EVENT_K": str(args.candidate_event_k),
        "TMCRA_DEEPSEEK_GRAPH_MODEL_MODE": "off",
        "TMCRA_TOPIC_BUCKET_MODE": "off",
        "TMCRA_LLM_CHANNEL_PLANNER_MODE": "off",
        "TMCRA_LLM_EVIDENCE_SELECTOR_MODE": "off",
        "TMCRA_EVIDENCE_UNIT_PLANNER_MODE": "off",
        "TMCRA_UNIFIED_OPERATION_PLANNER_MODE": "off",
    }
    for name, value in values.items():
        os.environ[name] = value
    return values


def scope_counts(db_path: Path, scope_id: str) -> dict[str, int]:
    with sqlite3.connect(db_path) as connection:
        output = {}
        for table in ("records", "memory_edges", "audit_turn_log", "audit_retrieval_log"):
            output[table] = int(
                connection.execute(f'SELECT COUNT(*) FROM "{table}" WHERE scope_id=?', (scope_id,)).fetchone()[0]
            )
    return output


def scope_fingerprint(db_path: Path, scope_id: str) -> str:
    relevant_tables = ("records", "memory_edges", "slot_heads", "slot_history")
    snapshot: dict[str, Any] = {}
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        for table in relevant_tables:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if exists is None:
                snapshot[table] = None
                continue
            rows = []
            for row in connection.execute(
                f'SELECT * FROM "{table}" WHERE scope_id=?', (scope_id,)
            ):
                normalized: dict[str, Any] = {}
                for key in row.keys():
                    value = row[key]
                    if key.endswith("_json") and isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(
                                f"invalid JSON in {table}.{key} while fingerprinting scope"
                            ) from exc
                    normalized[key] = value
                rows.append(normalized)
            snapshot[table] = sorted(
                rows,
                key=lambda item: json.dumps(
                    item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            )
    return hashlib.sha256(
        json.dumps(
            snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def load_recent_dialogue_context(
    db_path: Path,
    scope_id: str,
    *,
    current_query: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Load a small, metadata-free dialogue tail for planner reference resolution."""
    if limit <= 0 or limit > 8:
        raise RuntimeError("recent dialogue limit must be between 1 and 8")
    with sqlite3.connect(db_path) as connection:
        columns = {
            str(row[1]) for row in connection.execute('PRAGMA table_info("audit_turn_log")')
        }
        required = {"scope_id", "event_index", "payload_json"}
        if not required.issubset(columns):
            raise RuntimeError(
                "audit_turn_log lacks required production dialogue columns: "
                + ",".join(sorted(required - columns))
            )
        rows = connection.execute(
            'SELECT event_index,payload_json FROM "audit_turn_log" '
            "WHERE scope_id=? ORDER BY event_index DESC LIMIT ?",
            (scope_id, limit + 1),
        ).fetchall()
    dialogue: list[dict[str, Any]] = []
    for event_index, raw_payload in reversed(rows):
        try:
            payload = json.loads(raw_payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid audit_turn_log payload at event_index={event_index}") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"audit_turn_log payload is not an object at event_index={event_index}")
        speaker = clean_text(payload.get("speaker")).lower()
        text = clean_text(payload.get("text"))
        if speaker not in {"user", "assistant"} or not text:
            raise RuntimeError(f"audit_turn_log turn is not planner-safe at event_index={event_index}")
        dialogue.append(
            {"turn_index": int(event_index), "speaker": speaker, "text": text}
        )
    current = clean_text(current_query)
    if dialogue and dialogue[-1]["speaker"] == "user" and dialogue[-1]["text"] == current:
        dialogue.pop()
    return dialogue[-limit:]


def append_layered_retrieval_audit(
    *,
    repo: Path,
    db_path: Path,
    scope_id: str,
    operation_id: str,
    evidence: Mapping[str, Any],
    debug: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the final layered retrieval decision without rewriting graph state."""
    resolved_repo = str(repo.resolve())
    if resolved_repo not in sys.path:
        sys.path.insert(0, resolved_repo)
    from experiments.replacement.memory_graph import SQLiteSessionMemoryStore

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT value_json FROM meta WHERE scope_id=? AND key='audit_retention'",
            (scope_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"{scope_id}: missing audit_retention metadata")
    audit_retention = int(json.loads(row[0]) or 0)
    if audit_retention <= 0:
        raise RuntimeError(f"{scope_id}: invalid audit_retention metadata: {audit_retention}")

    windows = list(evidence.get("evidence_windows") or [])
    memory_contexts = [
        context
        for window in windows
        for context in list(window.get("memory_contexts") or [])
    ]
    graph = dict(debug.get("graph") or {})
    payload = {
        "event_kind": "tmcra.v3.layered_retrieval",
        "schema_version": "tmcra.v3.layered-retrieval-audit.1",
        "operation_id": clean_text(operation_id),
        "question_id": clean_text(evidence.get("question_id")),
        "query": clean_text(evidence.get("question")),
        "question_date": clean_text(evidence.get("question_date")),
        "recall_plan": dict(evidence.get("recall_plan") or {}),
        "selected_memory_ids": [
            clean_text(window.get("memory_id"))
            for window in windows
            if clean_text(window.get("memory_id"))
        ],
        "selected_session_ids": list(evidence.get("selected_session_ids") or []),
        "selected_capsule_ids": sorted(
            {
                clean_text(context.get("capsule_id"))
                for context in memory_contexts
                if clean_text(context.get("capsule_id"))
            }
        ),
        "selected_claim_ids": sorted(
            {
                clean_text(dict(context.get("provenance") or {}).get("claim_id"))
                for context in memory_contexts
                if clean_text(dict(context.get("provenance") or {}).get("claim_id"))
            }
        ),
        "selected_graph_event_ids": list(graph.get("selected_event_ids") or []),
        "runtime_input_has_gold": bool(debug.get("runtime_input_has_gold")),
        "graph_fingerprint": clean_text(debug.get("graph_fingerprint")),
        "evidence_sha256": hashlib.sha256(
            json.dumps(
                dict(evidence),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    if payload["runtime_input_has_gold"]:
        raise RuntimeError(f"{scope_id}: refusing to audit retrieval with evaluation labels")
    if not payload["operation_id"]:
        raise RuntimeError(f"{scope_id}: layered retrieval audit lacks operation_id")
    store = SQLiteSessionMemoryStore(str(db_path), audit_retention=audit_retention)
    return dict(
        store.append_audit_event(
            scope_id,
            "retrieval_log",
            payload,
            idempotency_key=payload["operation_id"],
        )
    )


def load_persisted_parent_chunks(db_path: Path, scope_id: str) -> list[dict[str, Any]]:
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    with sqlite3.connect(db_path) as connection:
        record_rows = connection.execute(
            "SELECT memory_id, turn_index, metadata_json FROM records WHERE scope_id=?",
            (scope_id,),
        ).fetchall()
        if not record_rows:
            raise RuntimeError(f"scope has no graph records: {scope_id}")
        product_parents: list[dict[str, Any]] = []
        product_message_ids: set[str] = set()
        for memory_id, turn_index, raw_metadata in record_rows:
            metadata = json.loads(raw_metadata)
            if clean_text(metadata.get("content_variant")) != "source_message":
                continue
            message_id = clean_text(metadata.get("message_id"))
            match = MESSAGE_ID_RE.fullmatch(message_id)
            if match is None:
                raise RuntimeError(f"{scope_id}: malformed persisted product message id: {message_id!r}")
            session_index = int(metadata.get("session_index", -1))
            message_index = int(metadata.get("message_index", -1))
            if session_index != int(match.group("session")) or message_index != int(match.group("parent")):
                raise RuntimeError(f"{scope_id}: product message location metadata disagrees with {message_id}")
            raw_content = metadata.get("raw_content")
            if not isinstance(raw_content, str) or not raw_content:
                raise RuntimeError(f"{scope_id}: product source {message_id} has no persisted raw content")
            if message_id in product_message_ids:
                raise RuntimeError(f"{scope_id}: duplicate immutable product source message: {message_id}")
            product_message_ids.add(message_id)
            role = clean_text(metadata.get("speaker") or dict(metadata.get("sidecar_hint_metadata") or {}).get("role"))
            session_id = clean_text(metadata.get("session_id") or dict(metadata.get("sidecar_hint_metadata") or {}).get("session_id"))
            date = clean_text(metadata.get("historical_date") or dict(metadata.get("sidecar_hint_metadata") or {}).get("historical_date"))
            timestamp = clean_text(metadata.get("timestamp"))
            if role not in {"user", "assistant", "system", "tool"} or not session_id or not date or not timestamp:
                raise RuntimeError(f"{scope_id}: incomplete product source metadata for {message_id}")
            product_parents.append(
                {
                    "chunk_id": message_id,
                    "parent_kind": "message",
                    "session_index": session_index,
                    "parent_chunk_index": message_index,
                    "message_index": message_index,
                    "session_id": session_id,
                    "date": date,
                    "timestamp": timestamp,
                    "role": role,
                    "text": raw_content,
                    "turn_index": int(turn_index),
                    "source_record_id": clean_text(memory_id),
                }
            )

        audit_rows = connection.execute(
            "SELECT event_index, payload_json FROM audit_turn_log WHERE scope_id=? ORDER BY event_index",
            (scope_id,),
        ).fetchall()
        if product_parents:
            audit_message_ids = {
                clean_text(dict(json.loads(raw_payload).get("metadata") or {}).get("message_id"))
                for _, raw_payload in audit_rows
                if clean_text(dict(json.loads(raw_payload).get("metadata") or {}).get("message_id"))
            }
            if audit_message_ids != product_message_ids:
                raise RuntimeError(
                    f"{scope_id}: immutable source/audit message sets differ: "
                    f"source={len(product_message_ids)} audit={len(audit_message_ids)}"
                )
            product_parents.sort(key=lambda row: (row["session_index"], row["message_index"]))
            return product_parents

        chunk_by_record: dict[str, str] = {}
        chunks_by_turn: dict[int, set[str]] = {}
        for memory_id, turn_index, raw_metadata in record_rows:
            metadata = json.loads(raw_metadata)
            sidecar = dict(metadata.get("sidecar_hint_metadata") or {})
            chunk_id = clean_text(sidecar.get("chunk_id"))
            if not chunk_id:
                dia_id = clean_text(metadata.get("dia_id"))
                chunk_id = dia_id.rsplit(":", 1)[-1] if dia_id else ""
            if not LEGACY_CHUNK_ID_RE.match(chunk_id):
                continue
            chunk_by_record[clean_text(memory_id)] = chunk_id
            chunks_by_turn.setdefault(int(turn_index), set()).add(chunk_id)

    if not audit_rows:
        raise RuntimeError(f"scope has no persisted write audit: {scope_id}")

    parents: list[dict[str, Any]] = []
    seen_locations: set[tuple[int, int]] = set()
    for event_index, raw_payload in audit_rows:
        payload = json.loads(raw_payload)
        if clean_text(payload.get("kind")) != "memory_write":
            continue
        turn_index = int(payload.get("turn_index", 0) or 0)
        record_ids = [clean_text(value) for value in list(payload.get("record_ids") or [])]
        candidate_chunks = {chunk_by_record[value] for value in record_ids if value in chunk_by_record}
        candidate_chunks.update(chunks_by_turn.get(turn_index, set()))
        if len(candidate_chunks) != 1:
            raise RuntimeError(
                f"{scope_id}: persisted turn {turn_index} maps to {len(candidate_chunks)} chunk ids: {sorted(candidate_chunks)}"
            )
        chunk_id = next(iter(candidate_chunks))
        match = LEGACY_CHUNK_ID_RE.match(chunk_id)
        assert match is not None
        session_index = int(match.group("session"))
        parent_index = int(match.group("parent"))
        location = (session_index, parent_index)
        if location in seen_locations:
            raise RuntimeError(f"{scope_id}: duplicate persisted parent location: {location}")
        seen_locations.add(location)
        persisted_text = str(payload.get("text", ""))
        parent_text = INGEST_PREFIX_RE.sub("", persisted_text, count=1)
        if parent_text == persisted_text or not parent_text:
            raise RuntimeError(f"{scope_id}: cannot recover raw parent text from audit turn {turn_index}")
        header = SESSION_HEADER_RE.match(parent_text)
        if header is None:
            raise RuntimeError(f"{scope_id}: malformed persisted LongMemEval parent header at {chunk_id}")
        parents.append(
            {
                "chunk_id": chunk_id,
                "parent_kind": "legacy_chunk",
                "session_index": session_index,
                "parent_chunk_index": parent_index,
                "session_id": clean_text(header.group("session_id")),
                "date": clean_text(header.group("date")),
                "text": parent_text,
                "turn_index": turn_index,
                "audit_event_index": int(event_index),
            }
        )
    parents.sort(key=lambda row: (row["session_index"], row["parent_chunk_index"]))
    if len(parents) != len(audit_rows):
        raise RuntimeError(
            f"{scope_id}: not every persisted audit turn became a parent chunk: parents={len(parents)} audit={len(audit_rows)}"
        )
    return parents


def parent_subchunks(
    parents: Sequence[Mapping[str, Any]],
    *,
    scope_id: str,
    subchunk_chars: int,
    subchunk_overlap: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for parent in parents:
        if clean_text(parent.get("parent_kind")) == "message":
            prefix = (
                f"TMCRA conversation_id={parent['session_id']} timestamp={parent['timestamp']} "
                f"message={int(parent['message_index']):03d} role={parent['role']}"
            )
        else:
            prefix = (
                f"LongMemEval session_id={parent['session_id']} date={parent['date']} "
                f"parent_chunk={int(parent['parent_chunk_index']):02d}"
            )
        payload_chars = int(subchunk_chars) - len(prefix) - 1
        spans = covered_windows(str(parent["text"]), payload_chars, int(subchunk_overlap))
        for subchunk_index, (char_start, char_end) in enumerate(spans, start=1):
            text = f"{prefix}\n{str(parent['text'])[char_start:char_end]}"
            if len(text) > subchunk_chars:
                raise RuntimeError("generated online subchunk exceeds the strict character limit")
            candidates.append(
                {
                    "candidate_id": f"chunk::{scope_id}:{parent['chunk_id']}_p{subchunk_index:02d}",
                    "text": text,
                    "session_id": parent["session_id"],
                    "session_index": int(parent["session_index"]),
                    "parent_chunk_index": int(parent["parent_chunk_index"]),
                    "parent_kind": clean_text(parent.get("parent_kind")) or "legacy_chunk",
                    "role": clean_text(parent.get("role")),
                    "message_role": clean_text(parent.get("role")),
                    "historical_date": clean_text(parent.get("date")),
                    "timestamp": clean_text(parent.get("timestamp")),
                    "subchunk_index": subchunk_index,
                    "source_char_start": char_start,
                    "source_char_end": char_end,
                    "source_record_id": clean_text(parent.get("source_record_id")),
                }
            )
    if not candidates:
        raise RuntimeError(f"no online candidates from persisted scope: {scope_id}")
    return candidates


def _metadata_json(value: Any, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label} metadata JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"invalid {label} metadata object")
    return parsed


def _canonical_slot(value: Any, *, label: str) -> str:
    slot = clean_text(value)
    if not slot:
        raise RuntimeError(f"{label} lacks canonical_slot")
    return slot


def _normalize_source_parents(value: Any, *, valid_locations: set[tuple[int, int]], label: str) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise RuntimeError(f"{label} lacks structured source_parents")
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"{label} source_parent is not an object")
        try:
            session = int(raw["session_index"])
            parent = int(raw.get("parent_chunk_index", raw.get("message_index")))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{label} source_parent lacks integer session/parent coordinates") from exc
        message_index: int | None = None
        if "message_index" in raw:
            try:
                message_index = int(raw["message_index"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{label} source_parent has a non-integer message_index"
                ) from exc
            if message_index != parent:
                raise RuntimeError(
                    f"{label} source_parent message_index differs from parent_chunk_index"
                )
        location = (session, parent)
        if location not in valid_locations:
            raise RuntimeError(f"{label} source_parent cannot map to a persisted parent: {location}")
        try:
            char_start = int(raw["evidence_char_start"])
            char_end = int(raw["evidence_char_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{label} source_parent lacks integer evidence character span"
            ) from exc
        source_record_id = clean_text(raw.get("source_record_id"))
        if char_start < 0 or char_end <= char_start or not source_record_id:
            raise RuntimeError(f"{label} source_parent has invalid evidence provenance")
        identity = (session, parent, char_start, char_end, source_record_id)
        if identity not in seen:
            seen.add(identity)
            item = dict(raw)
            item["session_index"] = session
            item["parent_chunk_index"] = parent
            item["evidence_char_start"] = char_start
            item["evidence_char_end"] = char_end
            item["source_record_id"] = source_record_id
            if message_index is not None:
                item["message_index"] = message_index
            output.append(item)
    if not output:
        raise RuntimeError(f"{label} has no mapped source_parents")
    return output


def load_layered_inventory(
    db_path: Path, scope_id: str, parents: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load current slow heads and map immutable fast semantic leaves to parents."""
    valid_locations = {(int(item["session_index"]), int(item["parent_chunk_index"])) for item in parents}
    slow_rows: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    semantic: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT memory_id, value, state, metadata_json FROM records WHERE scope_id=?", (scope_id,)
        ).fetchall()
    for memory_id, value, state, raw_metadata in rows:
        metadata = _metadata_json(raw_metadata, label=str(memory_id))
        layer = clean_text(metadata.get("memory_layer"))
        variant = clean_text(metadata.get("content_variant"))
        if layer == "slow" or variant == "slow_memory_capsule":
            if layer != "slow" or variant != "slow_memory_capsule":
                raise RuntimeError(f"{memory_id}: malformed slow capsule layer/variant")
            capsule_id = clean_text(metadata.get("capsule_id"))
            revision = metadata.get("revision")
            if not capsule_id or not isinstance(revision, int) or revision < 1:
                raise RuntimeError(f"{memory_id}: invalid capsule_id or revision")
            slow_rows.setdefault(capsule_id, []).append((str(memory_id), str(state), {**metadata, "value": str(value)}))
            continue
        if layer != "fast" or variant != "product_semantic_memory":
            continue
        if clean_text(state) not in CURRENT_FAST_STATES:
            continue
        if (
            clean_text(metadata.get("node_kind")) != "atomic_user_assertion"
            or metadata.get("atomic_evidence_leaf") is not True
            or clean_text(metadata.get("authority")) != "user_assertion"
        ):
            raise RuntimeError(f"{memory_id}: malformed fast semantic evidence leaf")
        slot = clean_text(metadata.get("canonical_slot") or metadata.get("canonical_slot_key"))
        if not slot:
            # Non-slot fast leaves still remain retrievable through their source parent.
            continue
        try:
            location = (int(metadata["session_index"]), int(metadata.get("parent_chunk_index", metadata.get("message_index"))))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{memory_id}: fast semantic record lacks source parent location") from exc
        if location not in valid_locations:
            raise RuntimeError(f"{memory_id}: fast semantic record cannot map to persisted parent {location}")
        source_record_id = clean_text(metadata.get("source_record_id"))
        if not source_record_id:
            raise RuntimeError(f"{memory_id}: fast semantic leaf lacks metadata.source_record_id")
        try:
            evidence_char_start = int(metadata["evidence_char_start"])
            evidence_char_end = int(metadata["evidence_char_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{memory_id}: fast semantic leaf lacks evidence character span"
            ) from exc
        if evidence_char_start < 0 or evidence_char_end <= evidence_char_start:
            raise RuntimeError(f"{memory_id}: fast semantic leaf has invalid evidence character span")
        semantic.append({
            "memory_id": str(memory_id), "record_state": clean_text(state), "canonical_slot": slot, "source_parent": {
                "session_index": location[0], "parent_chunk_index": location[1],
                "source_record_id": source_record_id,
                "evidence_char_start": evidence_char_start,
                "evidence_char_end": evidence_char_end,
            }, "provenance": {"memory_layer": "fast", "content_variant": variant, "source_record_id": source_record_id, "semantic_memory_id": str(memory_id)},
        })
    capsules: list[dict[str, Any]] = []
    for capsule_id, revisions in slow_rows.items():
        max_revision = max(int(meta["revision"]) for _, _, meta in revisions)
        current = [(memory_id, state, meta) for memory_id, state, meta in revisions if int(meta["revision"]) == max_revision]
        if len(current) != 1:
            raise RuntimeError(f"{scope_id}: capsule {capsule_id} lacks a unique latest active/challenged revision")
        memory_id, state, meta = current[0]
        if state != "active" or clean_text(meta.get("status")) not in {"active", "challenged"}:
            continue
        claims = meta.get("claims")
        if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)) or not claims:
            raise RuntimeError(f"{memory_id}: capsule lacks claims")
        capsule_parents = meta.get("source_parents")
        for claim_index, claim in enumerate(claims):
            if not isinstance(claim, Mapping):
                raise RuntimeError(f"{memory_id}: claim {claim_index} is not an object")
            slot = _canonical_slot(claim.get("canonical_slot"), label=f"{memory_id}: claim {claim_index}")
            source_parents = _normalize_source_parents(
                claim.get("source_parents", capsule_parents), valid_locations=valid_locations,
                label=f"{memory_id}: claim {claim_index}",
            )
            claim_text = clean_text(claim.get("text"))
            claim_id = clean_text(claim.get("claim_id"))
            if not claim_text or not claim_id:
                raise RuntimeError(f"{memory_id}: claim {claim_index} lacks claim_id or text")
            capsules.append({
                "candidate_id": f"capsule::{capsule_id}:r{meta['revision']}:c{claim_index}",
                "memory_id": memory_id, "capsule_id": capsule_id, "revision": int(meta["revision"]),
                "status": clean_text(meta.get("status")), "canonical_slot": slot, "claims": [dict(claim)],
                "source_parents": source_parents, "text": claim_text,
                "provenance": {"memory_layer": "slow", "content_variant": "slow_memory_capsule", "capsule_id": capsule_id, "revision": int(meta["revision"]), "claim_id": claim_id, "canonical_slot": slot, "patch_id": clean_text(meta.get("patch_id")), "source_parents": source_parents},
            })
    return capsules, semantic


def command_build_index(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.scope_manifest))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    vectorizer: BgeM3DenseVectorizer | None = None
    started = time.time()
    report_rows: list[dict[str, Any]] = []
    reused_index_count = 0
    for row_index, row in enumerate(rows, start=1):
        db_path = Path(row["db_path"]).resolve()
        scope_id = clean_text(row.get("scope_id"))
        index_path = Path(row["index_path"]).resolve()
        if index_path.exists():
            payload = torch.load(index_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping):
                raise RuntimeError(f"existing online index is not an object: {index_path}")
            expected = {
                "schema_version": ONLINE_INDEX_SCHEMA_VERSION,
                "fast_semantic_state_policy": FAST_SEMANTIC_STATE_POLICY,
                "scope_id": scope_id,
                "db_path": str(db_path),
                "subchunk_chars": int(args.subchunk_chars),
                "subchunk_overlap": int(args.subchunk_overlap),
                "embedding_model": str(Path(args.embedding_model).resolve()),
                "embedding_max_length": int(args.embedding_max_length),
                "strict_no_truncation": True,
            }
            mismatches = {
                key: {"expected": value, "actual": payload.get(key)}
                for key, value in expected.items()
                if payload.get(key) != value
            }
            current_fingerprint = scope_fingerprint(db_path, scope_id)
            if clean_text(payload.get("graph_fingerprint")) != current_fingerprint:
                mismatches["graph_fingerprint"] = {
                    "expected": current_fingerprint,
                    "actual": payload.get("graph_fingerprint"),
                }
            if mismatches:
                raise RuntimeError(
                    f"existing online index is incompatible with current scope: "
                    f"{index_path} mismatches={json.dumps(mismatches, sort_keys=True)}"
                )
            report_row = {
                "question_id": clean_text(row.get("question_id")),
                "scope_id": scope_id,
                "db_path": str(db_path),
                "index_path": str(index_path),
                "parent_count": int(payload["parent_count"]),
                "candidate_count": int(payload["candidate_count"]),
                "slow_capsule_count": int(payload["slow_capsule_count"]),
                "fast_semantic_record_count": int(payload["fast_semantic_record_count"]),
                "fast_semantic_state_policy": payload["fast_semantic_state_policy"],
                "graph_counts": dict(payload["graph_counts_at_index"]),
                "graph_fingerprint": current_fingerprint,
                "reused_existing_index": True,
            }
            reused_index_count += 1
            report_rows.append(report_row)
            print(
                json.dumps(
                    {"status": "index_reused", "row": row_index, "total": len(rows), **report_row}
                ),
                flush=True,
            )
            continue
        if vectorizer is None:
            vectorizer = BgeM3DenseVectorizer(
                dim=args.text_dim,
                model_path=args.embedding_model,
                device=str(device),
                max_length=args.embedding_max_length,
                strict_max_length=True,
            )
        graph_fingerprint = scope_fingerprint(db_path, scope_id)
        parents = load_persisted_parent_chunks(db_path, scope_id)
        candidates = parent_subchunks(
            parents,
            scope_id=scope_id,
            subchunk_chars=args.subchunk_chars,
            subchunk_overlap=args.subchunk_overlap,
        )
        slow_capsules, fast_semantic_records = load_layered_inventory(db_path, scope_id, parents)
        fast_vectors = vectorizer.encode_batch([candidate["text"] for candidate in candidates], batch_size=args.batch_size)
        slow_vectors = (
            vectorizer.encode_batch([capsule["text"] for capsule in slow_capsules], batch_size=args.batch_size)
            if slow_capsules else torch.empty((0, args.text_dim), dtype=torch.float32)
        )
        fast_vectors = fast_vectors.to(torch.float16).contiguous()
        slow_vectors = slow_vectors.to(torch.float16).contiguous()
        counts = scope_counts(db_path, scope_id)
        if scope_fingerprint(db_path, scope_id) != graph_fingerprint:
            raise RuntimeError(
                f"{scope_id}: graph changed while the online index was being built"
            )
        payload = {
            "schema_version": ONLINE_INDEX_SCHEMA_VERSION,
            "fast_semantic_state_policy": FAST_SEMANTIC_STATE_POLICY,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "scope_id": scope_id,
            "db_path": str(db_path),
            "graph_counts_at_index": counts,
            "graph_fingerprint": graph_fingerprint,
            "parent_count": len(parents),
            "candidate_count": len(candidates),
            "slow_capsule_count": len(slow_capsules),
            "fast_semantic_record_count": len(fast_semantic_records),
            "subchunk_chars": args.subchunk_chars,
            "subchunk_overlap": args.subchunk_overlap,
            "embedding_model": str(Path(args.embedding_model).resolve()),
            "embedding_max_length": args.embedding_max_length,
            "strict_no_truncation": True,
            "fast_candidates": candidates,
            "fast_vectors": fast_vectors,
            "slow_capsules": slow_capsules,
            "slow_vectors": slow_vectors,
            "fast_semantic_records": fast_semantic_records,
        }
        atomic_torch_save(payload, index_path)
        report_row = {
            "question_id": clean_text(row.get("question_id")),
            "scope_id": scope_id,
            "db_path": str(db_path),
            "index_path": str(index_path),
            "parent_count": len(parents),
            "candidate_count": len(candidates),
            "slow_capsule_count": len(slow_capsules),
            "fast_semantic_record_count": len(fast_semantic_records),
            "fast_semantic_state_policy": FAST_SEMANTIC_STATE_POLICY,
            "graph_counts": counts,
            "graph_fingerprint": graph_fingerprint,
            "reused_existing_index": False,
        }
        report_rows.append(report_row)
        print(json.dumps({"status": "indexed", "row": row_index, "total": len(rows), **report_row}), flush=True)
    report = {
        "status": "complete",
        "schema_version": "tmcra.v3.online-index-report.2",
        "row_count": len(report_rows),
        "parent_count": sum(row["parent_count"] for row in report_rows),
        "candidate_count": sum(row["candidate_count"] for row in report_rows),
        "slow_capsule_count": sum(row["slow_capsule_count"] for row in report_rows),
        "reused_index_count": reused_index_count,
        "elapsed_sec": round(time.time() - started, 3),
        "rows": report_rows,
    }
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(out_report, json.dumps(report, indent=2, sort_keys=True) + "\n")


def ordered_graph_parents(
    event_ids: Iterable[Any],
    *,
    valid_locations: set[tuple[int, int]],
    strict_prefix: bool = False,
) -> tuple[list[tuple[int, int]], list[str]]:
    output: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    unmapped: list[str] = []
    for raw in event_ids:
        event_id = clean_text(raw)
        match = EVENT_PARENT_RE.search(event_id)
        if match is None:
            if strict_prefix and event_id.startswith(("event::longmemeval:", "event::tmcra:")):
                unmapped.append(event_id)
            continue
        location = (int(match.group("session")), int(match.group("parent")))
        if location not in valid_locations:
            unmapped.append(event_id)
            continue
        if location not in seen:
            seen.add(location)
            output.append(location)
    return output, unmapped


def expand_parent_locations(
    locations: Iterable[tuple[int, int]],
    parent_candidates: Mapping[tuple[int, int], Sequence[int]],
) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for location in locations:
        for candidate_index in parent_candidates[location]:
            if candidate_index not in seen:
                seen.add(candidate_index)
                output.append(candidate_index)
    return output


class OnlineModels:
    def __init__(self, args: argparse.Namespace):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.dense = BgeM3DenseVectorizer(
            dim=args.text_dim,
            model_path=args.embedding_model,
            device=str(self.device),
            max_length=args.embedding_max_length,
            strict_max_length=True,
        )
        cross_path = Path(args.cross_model).resolve()
        self.cross_manifest = load_optional_hf_model_manifest(cross_path)
        self.cross_tokenizer = AutoTokenizer.from_pretrained(
            str(cross_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.cross_model = AutoModelForSequenceClassification.from_pretrained(
            str(cross_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.cross_model.eval()
        hidden_size = int(getattr(self.cross_model.config, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            raise RuntimeError("cross encoder hidden size is unavailable")
        checkpoint_path = Path(args.checkpoint).resolve()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if checkpoint.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("online checkpoint schema mismatch")
        if tuple(checkpoint.get("channel_names") or ()) != CHANNEL_NAMES:
            raise RuntimeError("online checkpoint channel mismatch")
        config = dict(checkpoint.get("model_config") or {})
        self.fusion = ChannelAwareMemoryReranker(
            representation_dim=hidden_size,
            channel_dim=len(CHANNEL_NAMES),
            hidden_dim=int(config["hidden_dim"]),
            layers=int(config["layers"]),
        ).to(self.device)
        self.fusion.load_state_dict(checkpoint["model_state"])
        self.fusion.eval()
        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        self.cross_max_length = int(args.cross_max_length)
        self.cross_batch_size = int(args.cross_batch_size)

    def encode_cross(self, query: str, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        representations: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.cross_batch_size):
                batch_texts = list(texts[start : start + self.cross_batch_size])
                encoded = self.cross_tokenizer(
                    [query] * len(batch_texts),
                    batch_texts,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                token_lengths = encoded["attention_mask"].sum(dim=1)
                longest = int(token_lengths.max())
                if longest > self.cross_max_length:
                    offending = int(torch.argmax(token_lengths))
                    raise RuntimeError(
                        f"online cross pair has {longest} tokens, exceeding strict max={self.cross_max_length}; "
                        f"candidate_batch_index={offending}"
                    )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.cross_model(**encoded, output_hidden_states=True, return_dict=True)
                if output.hidden_states is None:
                    raise RuntimeError("online cross encoder returned no hidden states")
                representations.append(output.hidden_states[-1][:, 0].float())
                semantic = output.logits.float()
                if semantic.ndim == 2 and semantic.shape[1] == 1:
                    semantic = semantic[:, 0]
                elif semantic.ndim != 1:
                    raise RuntimeError(f"online cross logits have invalid shape: {tuple(semantic.shape)}")
                logits.append(semantic)
        return torch.cat(representations, dim=0), torch.cat(logits, dim=0)


def load_online_index(path: Path, expected_db: Path, expected_scope: str) -> tuple[list[dict[str, Any]], torch.Tensor, list[dict[str, Any]], torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != ONLINE_INDEX_SCHEMA_VERSION:
        raise RuntimeError(f"online index schema mismatch: {path}")
    if payload.get("fast_semantic_state_policy") != FAST_SEMANTIC_STATE_POLICY:
        raise RuntimeError(f"online index fast semantic state policy mismatch: {path}")
    if clean_text(payload.get("scope_id")) != expected_scope:
        raise RuntimeError(f"online index scope mismatch: {path}")
    if Path(payload.get("db_path", "")).resolve() != expected_db.resolve():
        raise RuntimeError(f"online index database mismatch: {path}")
    candidates = list(payload.get("fast_candidates") or [])
    vectors = payload.get("fast_vectors")
    slow_capsules = list(payload.get("slow_capsules") or [])
    slow_vectors = payload.get("slow_vectors")
    semantic_records = list(payload.get("fast_semantic_records") or [])
    if not candidates or vectors is None or tuple(vectors.shape) != (len(candidates), 1024):
        raise RuntimeError(f"online index payload is incomplete: {path}")
    if slow_vectors is None or tuple(slow_vectors.shape) != (len(slow_capsules), 1024):
        raise RuntimeError(f"online slow index payload is incomplete: {path}")
    if any("labels" in candidate for candidate in candidates):
        raise RuntimeError("runtime index must not contain benchmark labels")
    return candidates, vectors.float().contiguous(), slow_capsules, slow_vectors.float().contiguous(), semantic_records, payload


def choose_diverse(
    candidates: Sequence[Mapping[str, Any]],
    scores: torch.Tensor,
    *,
    top_k: int,
    max_per_parent: int,
    max_per_session: int,
) -> list[int]:
    order = torch.argsort(scores, descending=True).tolist()
    selected: list[int] = []
    parent_counts: Counter[tuple[int, int]] = Counter()
    session_counts: Counter[str] = Counter()
    for index in order:
        candidate = candidates[index]
        parent = (int(candidate["session_index"]), int(candidate["parent_chunk_index"]))
        session_id = clean_text(candidate.get("session_id"))
        if max_per_parent > 0 and parent_counts[parent] >= max_per_parent:
            continue
        if max_per_session > 0 and session_counts[session_id] >= max_per_session:
            continue
        selected.append(index)
        parent_counts[parent] += 1
        session_counts[session_id] += 1
        if len(selected) >= top_k:
            break
    if len(selected) != top_k:
        raise RuntimeError(f"online diversity policy produced only {len(selected)} of {top_k} required windows")
    return selected


def _deprecated_legacy_retrieve_one_v1(
    row: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    harness: Any,
    models: OnlineModels,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retained only for forensic comparison; the v2 command never calls this schema .1 path."""
    started = time.time()
    qid = clean_text(row.get("question_id"))
    question = clean_text(row.get("question"))
    question_date = clean_text(row.get("question_date"))
    runtime_question = f"{question}\nQuestion date: {question_date}" if question_date else question
    db_path = Path(row["db_path"]).resolve()
    scope_id = clean_text(row.get("scope_id"))
    index_path = Path(row["index_path"]).resolve()
    if not qid or not question or not scope_id:
        raise RuntimeError("online retrieval manifest row lacks qid, question, or scope")
    candidates, dense_vectors, index_payload = load_online_index(index_path, db_path, scope_id)
    counts_before = scope_counts(db_path, scope_id)
    if counts_before["records"] != int(index_payload["graph_counts_at_index"]["records"]):
        raise RuntimeError(f"{qid}: graph records changed after online index creation")

    adapter = harness.build_adapter(scope_id, db_path)
    graph_started = time.time()
    retrieval = adapter.retrieve(runtime_question, top_k=args.graph_top_k)
    graph_elapsed = time.time() - graph_started
    metadata = dict(getattr(retrieval, "metadata", {}) or {})
    if clean_text(metadata.get("retrieval_mode")) != "hybrid_node_scored":
        raise RuntimeError(f"{qid}: graph retrieval did not use hybrid_node_scored")
    if not bool(metadata.get("hybrid_enabled")):
        raise RuntimeError(f"{qid}: graph hybrid model path is not active")
    selected_event_ids = list(metadata.get("selected_event_ids") or [])
    recall_event_ids = list(metadata.get("recall_event_ids") or [])
    final_event_ids = list(metadata.get("final_hit_event_ids") or [])
    if not selected_event_ids:
        raise RuntimeError(f"{qid}: graph model selected no events")

    parent_candidates: dict[tuple[int, int], list[int]] = {}
    for index, candidate in enumerate(candidates):
        location = (int(candidate["session_index"]), int(candidate["parent_chunk_index"]))
        parent_candidates.setdefault(location, []).append(index)
    valid_locations = set(parent_candidates)
    selected_parents, selected_unmapped = ordered_graph_parents(
        selected_event_ids, valid_locations=valid_locations, strict_prefix=True
    )
    recall_parents, recall_unmapped = ordered_graph_parents(recall_event_ids, valid_locations=valid_locations)
    final_parents, final_unmapped = ordered_graph_parents(final_event_ids, valid_locations=valid_locations)
    critical_unmapped = [*selected_unmapped, *final_unmapped]
    if critical_unmapped:
        raise RuntimeError(f"{qid}: selected/final graph events cannot map to persisted chunks: {critical_unmapped[:8]}")
    graph_parents: list[tuple[int, int]] = []
    graph_parent_seen: set[tuple[int, int]] = set()
    for location in [*selected_parents, *recall_parents]:
        if location not in graph_parent_seen:
            graph_parent_seen.add(location)
            graph_parents.append(location)
    graph_parents = graph_parents[: args.graph_k]
    graph_parent_rank = {location: rank for rank, location in enumerate(graph_parents)}
    graph_runtime_order = expand_parent_locations(graph_parents, parent_candidates)
    graph_rank = {
        index: graph_parent_rank[(int(candidates[index]["session_index"]), int(candidates[index]["parent_chunk_index"]))]
        for index in graph_runtime_order
    }
    selected_indexes = set(expand_parent_locations(selected_parents, parent_candidates))
    final_indexes = set(expand_parent_locations(final_parents, parent_candidates))

    dense_started = time.time()
    query_vector = models.dense.encode_one(runtime_question)
    dense_scores = dense_vectors @ query_vector
    dense_order = sorted(range(len(candidates)), key=lambda index: (-float(dense_scores[index]), index))
    dense_rank = {index: rank for rank, index in enumerate(dense_order)}
    dense_elapsed = time.time() - dense_started
    runtime_indexes: list[int] = []
    runtime_seen: set[int] = set()
    for index in [*dense_order[: args.dense_k], *graph_runtime_order]:
        if index not in runtime_seen:
            runtime_seen.add(index)
            runtime_indexes.append(index)
    if not runtime_indexes:
        raise RuntimeError(f"{qid}: graph+dense union is empty")

    session_count = max(int(candidate["session_index"]) for candidate in candidates) + 1
    runtime_candidates: list[dict[str, Any]] = []
    for index in runtime_indexes:
        source = dict(candidates[index])
        channels = {
            "dense_score": float(dense_scores[index]),
            "dense_rank_rr": rrank(dense_rank[index]),
            "graph_rank_rr": rrank(graph_rank.get(index)),
            "graph_selected": float(index in selected_indexes),
            "graph_final": float(index in final_indexes),
            "recency_norm": float(source["session_index"]) / float(max(1, session_count - 1)),
        }
        if tuple(channels) != CHANNEL_NAMES:
            raise AssertionError("online channel order changed")
        source["channels"] = channels
        runtime_candidates.append(source)

    cross_started = time.time()
    representations, semantic_logits = models.encode_cross(
        runtime_question, [candidate["text"] for candidate in runtime_candidates]
    )
    channel_tensor = torch.tensor(
        [[candidate["channels"][name] for name in CHANNEL_NAMES] for candidate in runtime_candidates],
        dtype=torch.float32,
        device=models.device,
    )
    mask = torch.ones((1, len(runtime_candidates)), dtype=torch.bool, device=models.device)
    with torch.inference_mode():
        fusion_scores = models.fusion(
            representations.unsqueeze(0),
            semantic_logits.unsqueeze(0),
            channel_tensor.unsqueeze(0),
            mask,
            ablation="full",
        )[0].detach().cpu()
    semantic_cpu = semantic_logits.detach().cpu()
    cross_elapsed = time.time() - cross_started
    selected = choose_diverse(
        runtime_candidates,
        fusion_scores,
        top_k=args.top_k,
        max_per_parent=args.max_per_parent,
        max_per_session=args.max_per_session,
    )
    evidence_windows = []
    for rank, index in enumerate(selected, start=1):
        candidate = runtime_candidates[index]
        evidence_windows.append(
            {
                "memory_id": candidate["candidate_id"],
                "session_id": candidate["session_id"],
                "session_index": candidate["session_index"],
                "parent_chunk_index": candidate["parent_chunk_index"],
                "subchunk_index": candidate["subchunk_index"],
                "rank": rank,
                "score": round(float(fusion_scores[index]), 6),
                "semantic_logit": round(float(semantic_cpu[index]), 6),
                "channels": candidate["channels"],
                "text": candidate["text"],
            }
        )
    evidence_row = {
        "schema_version": SCHEMA_VERSION,
        "runtime_schema_version": "tmcra.v3.online-retrieval.1",
        "question_id": qid,
        "question": question,
        "question_date": question_date,
        "question_type": clean_text(row.get("question_type")),
        "selected_session_ids": [window["session_id"] for window in evidence_windows],
        "evidence_windows": evidence_windows,
    }
    debug_row = {
        "question_id": qid,
        "scope_id": scope_id,
        "db_path": str(db_path),
        "index_path": str(index_path),
        "runtime_input_has_gold": any(
            key in row for key in ("answer", "gold_answer", "answer_session_ids", "labels", "supervision")
        ),
        "graph": {
            "retrieval_mode": metadata.get("retrieval_mode"),
            "hybrid_enabled": metadata.get("hybrid_enabled"),
            "decision_score_source": metadata.get("decision_score_source"),
            "selected_event_count": len(selected_event_ids),
            "recall_event_count": len(recall_event_ids),
            "final_event_count": len(final_event_ids),
            "selected_parent_count": len(selected_parents),
            "recall_parent_count": len(recall_parents),
            "final_parent_count": len(final_parents),
            "unmapped_recall_event_ids": recall_unmapped,
            "selected_event_ids": selected_event_ids,
            "recall_event_ids": recall_event_ids,
            "final_hit_event_ids": final_event_ids,
        },
        "inventory_count": len(candidates),
        "dense_k": args.dense_k,
        "graph_k": args.graph_k,
        "union_count": len(runtime_candidates),
        "selected_count": len(evidence_windows),
        "checkpoint": str(models.checkpoint_path),
        "checkpoint_sha256": models.checkpoint_sha256,
        "cross_model_revision": models.cross_manifest.get("revision"),
        "strict_no_truncation": True,
        "restart_boundary_verified": True,
        "graph_counts_before_query": counts_before,
        "latency_sec": {
            "graph": round(graph_elapsed, 4),
            "dense": round(dense_elapsed, 4),
            "cross_and_fusion": round(cross_elapsed, 4),
            "total": round(time.time() - started, 4),
        },
        "ranked_union": [
            {
                "candidate_id": runtime_candidates[index]["candidate_id"],
                "session_id": runtime_candidates[index]["session_id"],
                "parent_chunk_index": runtime_candidates[index]["parent_chunk_index"],
                "subchunk_index": runtime_candidates[index]["subchunk_index"],
                "fusion_score": round(float(fusion_scores[index]), 6),
                "semantic_logit": round(float(semantic_cpu[index]), 6),
                "channels": runtime_candidates[index]["channels"],
            }
            for index in torch.argsort(fusion_scores, descending=True).tolist()
        ],
    }
    if debug_row["runtime_input_has_gold"]:
        raise RuntimeError(f"{qid}: runtime retrieval manifest contains forbidden evaluation labels")
    return evidence_row, debug_row


def planner_from_env() -> DeepSeekFlashRecallPlanner:
    pool = [part.strip() for part in os.environ.get("TMCRA_RECALL_PLANNER_API_KEY_POOL", "").split(",") if part.strip()]
    return DeepSeekFlashRecallPlanner(
        base_url=os.environ.get("TMCRA_RECALL_PLANNER_BASE_URL", ""),
        model=os.environ.get("TMCRA_RECALL_PLANNER_MODEL", DEEPSEEK_FLASH_MODEL), api_keys=pool,
    )


def _fast_candidates_with_slots(candidates: Sequence[Mapping[str, Any]], semantic_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_parent: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for record in semantic_records:
        parent = dict(record["source_parent"])
        by_parent.setdefault((int(parent["session_index"]), int(parent["parent_chunk_index"])), []).append(record)
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        location = (int(candidate["session_index"]), int(candidate["parent_chunk_index"]))
        records = [
            record
            for record in by_parent.get(location, [])
            if int(candidate["source_char_end"])
            > int(record["source_parent"]["evidence_char_start"])
            and int(candidate["source_char_start"])
            < int(record["source_parent"]["evidence_char_end"])
        ]
        by_slot: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            by_slot.setdefault(_canonical_slot(record.get("canonical_slot"), label="fast semantic record"), []).append(record)
        if not by_slot:
            by_slot[f"fast.parent.s{location[0]}.p{location[1]}"] = []
        for slot, slot_records in by_slot.items():
            item = dict(candidate)
            item["canonical_slot"] = slot
            item["semantic_record_ids"] = [str(record["memory_id"]) for record in slot_records]
            item["provenance"] = {
                "memory_layer": "fast",
                "content_variant": "source_subchunk",
                "source_parent": {
                    "session_index": location[0],
                    "parent_chunk_index": location[1],
                    "source_record_ids": sorted(
                        {
                            clean_text(record["source_parent"].get("source_record_id"))
                            for record in slot_records
                            if clean_text(record["source_parent"].get("source_record_id"))
                        }
                    )
                    or ([clean_text(candidate.get("source_record_id"))] if clean_text(candidate.get("source_record_id")) else []),
                },
            }
            output.append(item)
    return output


def _unit_windows(unit: Mapping[str, Any], fast_candidates: Sequence[Mapping[str, Any]], *, qid: str) -> list[dict[str, Any]]:
    """Fully descend one composed unit, merging duplicate physical source windows."""
    by_parent: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for candidate in fast_candidates:
        by_parent.setdefault((int(candidate["session_index"]), int(candidate["parent_chunk_index"])), []).append(candidate)
    output: dict[tuple[int, int, int], dict[str, Any]] = {}

    def add(candidate: Mapping[str, Any], *, role: str, capsule: Mapping[str, Any] | None = None) -> None:
        key = (int(candidate["session_index"]), int(candidate["parent_chunk_index"]), int(candidate["subchunk_index"]))
        item = output.setdefault(key, {**dict(candidate), "roles": [], "capsules": []})
        if role not in item["roles"]:
            item["roles"].append(role)
        if capsule is not None and capsule not in item["capsules"]:
            item["capsules"].append(dict(capsule))

    def descend(capsule: Mapping[str, Any], role: str) -> None:
        for parent in capsule["source_parents"]:
            location = (int(parent["session_index"]), int(parent["parent_chunk_index"]))
            char_start = int(parent["evidence_char_start"])
            char_end = int(parent["evidence_char_end"])
            matches = [
                candidate
                for candidate in by_parent.get(location, [])
                if int(candidate["source_char_end"]) > char_start
                and int(candidate["source_char_start"]) < char_end
            ]
            if not matches:
                raise RuntimeError(f"{qid}: slow source_parent is unmapped during descent: {location}")
            for candidate in matches:
                add(candidate, role=role, capsule=capsule)

    kind = str(unit["unit_type"])
    if kind.startswith("fast_primary"):
        add(unit["fast_candidate"], role="primary")
    elif kind == "slow_primary_with_fast_override":
        descend(unit["slow_capsule"], "primary")
        for candidate in unit.get("fast_overrides", []):
            add(candidate, role="override")
    elif kind.startswith("slow_primary"):
        descend(unit["slow_capsule"], "primary")
    elif kind == "conflict_group":
        for capsule in unit["slow_capsules"]:
            descend(capsule, "slow_conflict_candidate")
        for candidate in unit["fast_candidates"]:
            add(candidate, role="fast_conflict_candidate")
    else:
        raise RuntimeError(f"{qid}: unsupported planned unit type {kind}")
    return list(output.values())


def _unit_attachments(unit: Mapping[str, Any]) -> list[dict[str, Any]]:
    if unit.get("slow_context"):
        return [{"role": "context_only", "capsule_id": item["capsule_id"], "canonical_slot": item["canonical_slot"], "summary": item["text"], "source_parents": item["source_parents"], "provenance": item["provenance"]} for item in unit["slow_context"]]
    if unit.get("fast_overrides"):
        return [{"role": "override", "memory_id": item["candidate_id"], "canonical_slot": item["canonical_slot"], "text": item["text"], "provenance": item["provenance"]} for item in unit["fast_overrides"]]
    return []


def pack_recall_units(units: Sequence[Mapping[str, Any]], fast_candidates: Sequence[Mapping[str, Any]], *, top_k: int, qid: str) -> list[tuple[Mapping[str, Any], list[dict[str, Any]]]]:
    if top_k <= 0:
        raise RuntimeError("top_k must be positive")
    packed: list[tuple[Mapping[str, Any], list[dict[str, Any]]]] = []
    used: set[tuple[int, int, int]] = set()
    for unit in units:
        windows = _unit_windows(unit, fast_candidates, qid=qid)
        unique = [item for item in windows if (int(item["session_index"]), int(item["parent_chunk_index"]), int(item["subchunk_index"])) not in used]
        if not unique:
            if windows:
                packed.append((unit, windows))
            continue
        if len(unique) > top_k and not packed:
            raise RuntimeError(f"{qid}: first atomic recall unit requires {len(unique)} windows, exceeding strict packing budget {top_k}")
        if len(used) + len(unique) > top_k:
            break
        packed.append((unit, windows))
        used.update((int(item["session_index"]), int(item["parent_chunk_index"]), int(item["subchunk_index"])) for item in unique)
    if not packed:
        raise RuntimeError(f"{qid}: recall plan produced no packable evidence units")
    return packed


def retrieve_one(
    row: Mapping[str, Any], *, args: argparse.Namespace, harness: Any, models: OnlineModels,
    planner: DeepSeekFlashRecallPlanner,
    graph_adapter_cache: dict[tuple[str, str], Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.time()
    qid, question, question_date = clean_text(row.get("question_id")), clean_text(row.get("question")), clean_text(row.get("question_date"))
    if not qid or not question:
        raise RuntimeError("online retrieval manifest row lacks qid or question")
    if int(args.slow_dense_k) <= 0:
        raise RuntimeError("slow_dense_k must be positive")
    db_path, scope_id, index_path = Path(row["db_path"]).resolve(), clean_text(row.get("scope_id")), Path(row["index_path"]).resolve()
    fast, fast_vectors, slow, slow_vectors, semantic_records, payload = load_online_index(index_path, db_path, scope_id)
    counts_before = scope_counts(db_path, scope_id)
    if counts_before["records"] != int(payload["graph_counts_at_index"]["records"]):
        raise RuntimeError(f"{qid}: graph records changed after online index creation")
    graph_fingerprint = scope_fingerprint(db_path, scope_id)
    if graph_fingerprint != clean_text(payload.get("graph_fingerprint")):
        raise RuntimeError(f"{qid}: graph fingerprint changed after online index creation")
    recent_dialogue = load_recent_dialogue_context(
        db_path, scope_id, current_query=question, limit=8
    )
    raw_plan, planner_metadata = planner.plan(
        query=question, question_date=question_date or "unknown",
        recent_dialogue=recent_dialogue,
        available_layers={"fast": {"available": bool(fast), "candidate_count": len(fast)}, "slow": {"available": bool(slow), "capsule_count": len(slow)}},
    )
    plan = dict(raw_plan)
    planner_decision_reason = clean_text(plan.get("decision_reason"))
    if not planner_decision_reason:
        raise RuntimeError(f"{qid}: Flash recall plan lacks a decision reason")
    resolved_query = clean_text(plan.get("resolved_query"))
    if not resolved_query:
        raise RuntimeError(f"{qid}: Flash recall plan lacks a resolved query")
    runtime_question = (
        f"{resolved_query}\nQuestion date: {question_date}"
        if question_date
        else resolved_query
    )
    authoritative_plan = dict(plan)
    authoritative_plan.pop("decision_reason", None)
    if not slow and plan["mode"] != "FAST_ONLY":
        raise RuntimeError(f"{qid}: Flash must select FAST_ONLY when no slow capsules are available")

    graph_metadata: dict[str, Any] = {"skipped": plan["mode"] == "SLOW_ONLY"}
    graph_elapsed = 0.0
    graph_runtime_order: list[int] = []
    selected_indexes: set[int] = set()
    final_indexes: set[int] = set()
    graph_rank: dict[int, int] = {}
    parent_candidates: dict[tuple[int, int], list[int]] = {}
    graph_adapter_reused = False
    for index, candidate in enumerate(fast):
        parent_candidates.setdefault((int(candidate["session_index"]), int(candidate["parent_chunk_index"])), []).append(index)
    if plan["mode"] != "SLOW_ONLY":
        adapter_key = (scope_id, str(db_path))
        adapter = graph_adapter_cache.get(adapter_key) if graph_adapter_cache is not None else None
        if adapter is None:
            adapter = harness.build_adapter(scope_id, db_path)
            if graph_adapter_cache is not None:
                graph_adapter_cache[adapter_key] = adapter
        else:
            graph_adapter_reused = True
        graph_started = time.time()
        retrieval = adapter.retrieve(runtime_question, top_k=args.graph_top_k)
        graph_elapsed = time.time() - graph_started
        metadata = dict(getattr(retrieval, "metadata", {}) or {})
        if clean_text(metadata.get("retrieval_mode")) != "hybrid_node_scored" or not bool(metadata.get("hybrid_enabled")):
            raise RuntimeError(f"{qid}: graph hybrid_node_scored path is not active")
        selected_events, recall_events, final_events = list(metadata.get("selected_event_ids") or []), list(metadata.get("recall_event_ids") or []), list(metadata.get("final_hit_event_ids") or [])
        slow_graph_events = [str(event_id) for event_id in [*selected_events, *recall_events, *final_events] if str(event_id).startswith("slow.")]
        if slow_graph_events:
            raise RuntimeError(f"{qid}: fast graph execution crossed the slow-layer boundary: " + ",".join(dict.fromkeys(slow_graph_events)))
        if not selected_events:
            raise RuntimeError(f"{qid}: graph model selected no events")
        valid_locations = set(parent_candidates)
        selected_parents, selected_unmapped = ordered_graph_parents(selected_events, valid_locations=valid_locations, strict_prefix=True)
        recall_parents, recall_unmapped = ordered_graph_parents(recall_events, valid_locations=valid_locations)
        final_parents, final_unmapped = ordered_graph_parents(final_events, valid_locations=valid_locations)
        if [*selected_unmapped, *final_unmapped]:
            raise RuntimeError(f"{qid}: graph events cannot map to persisted chunks")
        graph_parents = list(dict.fromkeys([*selected_parents, *recall_parents]))[:args.graph_k]
        graph_runtime_order = expand_parent_locations(graph_parents, parent_candidates)
        graph_rank = {index: rank for rank, location in enumerate(graph_parents) for index in parent_candidates[location]}
        selected_indexes, final_indexes = set(expand_parent_locations(selected_parents, parent_candidates)), set(expand_parent_locations(final_parents, parent_candidates))
        graph_metadata = {"skipped": False, "adapter_reused": graph_adapter_reused, "selected_event_ids": selected_events, "recall_event_ids": recall_events, "final_hit_event_ids": final_events, "unmapped_recall_event_ids": recall_unmapped, "retrieval_mode": metadata.get("retrieval_mode"), "runtime_graph_cache_hit": bool(metadata.get("runtime_graph_cache_hit", False)), "hybrid_candidate_union_rescored": bool(metadata.get("hybrid_candidate_union_rescored", False)), "node_runtime_profile": dict(metadata.get("node_runtime_profile", {}) or {}), "hybrid_runtime_profile": dict(metadata.get("hybrid_runtime_profile", {}) or {}), "adapter_runtime_profile": dict(metadata.get("adapter_runtime_profile", {}) or {})}

    fast_ranked: list[dict[str, Any]] = []
    fast_scores: dict[str, float] = {}
    dense_elapsed = 0.0
    cross_elapsed = 0.0
    if plan["mode"] != "SLOW_ONLY":
        dense_started = time.time()
        dense_scores = fast_vectors @ models.dense.encode_one(runtime_question)
        dense_order = sorted(range(len(fast)), key=lambda i: (-float(dense_scores[i]), i))
        dense_rank = {index: rank for rank, index in enumerate(dense_order)}
        dense_elapsed = time.time() - dense_started
        indexes = list(dict.fromkeys([*dense_order[:args.dense_k], *graph_runtime_order]))
        if not indexes:
            raise RuntimeError(f"{qid}: graph+dense union is empty")
        session_count = max(int(candidate["session_index"]) for candidate in fast) + 1
        runtime = []
        for index in indexes:
            item = dict(fast[index])
            item["channels"] = {"dense_score": float(dense_scores[index]), "dense_rank_rr": rrank(dense_rank[index]), "graph_rank_rr": rrank(graph_rank.get(index)), "graph_selected": float(index in selected_indexes), "graph_final": float(index in final_indexes), "recency_norm": float(item["session_index"]) / max(1, session_count - 1)}
            runtime.append(item)
        cross_started = time.time()
        reps, logits = models.encode_cross(runtime_question, [item["text"] for item in runtime])
        channel_tensor = torch.tensor([[item["channels"][name] for name in CHANNEL_NAMES] for item in runtime], dtype=torch.float32, device=models.device)
        with torch.inference_mode():
            scores = models.fusion(reps.unsqueeze(0), logits.unsqueeze(0), channel_tensor.unsqueeze(0), torch.ones((1, len(runtime)), dtype=torch.bool, device=models.device), ablation="full")[0].detach().cpu()
        cross_elapsed += time.time() - cross_started
        for item, score, semantic in zip(runtime, scores.tolist(), logits.detach().cpu().tolist()):
            item["score"], item["semantic_logit"] = float(score), float(semantic)
        fast_ranked = _fast_candidates_with_slots(sorted(runtime, key=lambda item: -float(item["score"])), semantic_records)
        fast_scores = {str(item["candidate_id"]): float(item["score"]) for item in fast_ranked}

    slow_ranked: list[dict[str, Any]] = []
    slow_dense_elapsed = 0.0
    if slow and plan["mode"] != "FAST_ONLY":
        slow_dense_started = time.time()
        slow_dense_scores = slow_vectors @ models.dense.encode_one(runtime_question)
        slow_dense_order = sorted(range(len(slow)), key=lambda i: (-float(slow_dense_scores[i]), i))
        slow_dense_elapsed = time.time() - slow_dense_started
        slow_indexes = slow_dense_order[: min(len(slow_dense_order), int(args.slow_dense_k))]
        if not slow_indexes:
            raise RuntimeError(f"{qid}: slow inventory is nonempty but dense shortlist is empty")
        slow_ranked = [dict(slow[index]) for index in slow_indexes]
        for index, item in zip(slow_indexes, slow_ranked):
            item["slow_dense_score"] = float(slow_dense_scores[index])
            item["slow_dense_rank"] = int(slow_dense_order.index(index))
        cross_started = time.time()
        _, logits = models.encode_cross(runtime_question, [item["text"] for item in slow_ranked])
        cross_elapsed += time.time() - cross_started
        for item, score in zip(slow_ranked, logits.detach().cpu().tolist()):
            item["semantic_logit"] = float(score)
        slow_ranked.sort(key=lambda item: -float(item["semantic_logit"]))
    try:
        units = apply_recall_plan(plan, fast_ranked, slow_ranked)
    except RecallPlannerError as exc:
        raise RuntimeError(f"{qid}: invalid planned evidence composition: {exc}") from exc
    if plan["mode"] == "CONFLICT_COMPARE":
        units.sort(
            key=lambda item: (
                -max(
                    (
                        fast_scores.get(str(candidate.get("candidate_id", "")), float("-inf"))
                        for candidate in item.get("fast_candidates", [])
                    ),
                    default=float("-inf"),
                ),
                str(item["canonical_slot"]),
            )
        )
    elif plan["primary_layer"] == "fast":
        units.sort(key=lambda item: -fast_scores.get(str(item.get("fast_candidate", {}).get("candidate_id", "")), float("-inf")))
    else:
        units.sort(key=lambda item: -float(item.get("slow_capsule", {}).get("semantic_logit", float("-inf"))))

    packed_units = pack_recall_units(units, fast, top_k=args.top_k, qid=qid)
    evidence_by_location: dict[tuple[int, int, int], dict[str, Any]] = {}
    for unit_rank, (unit, entries) in enumerate(packed_units, start=1):
        attachments = _unit_attachments(unit)
        for entry in entries:
            capsules = list(entry.get("capsules") or [])
            memory_contexts = [
                {
                    "role": next(
                        (
                            role
                            for role in entry["roles"]
                            if role in {"primary", "slow_conflict_candidate"}
                        ),
                        "slow_context",
                    ),
                    "capsule_id": item["capsule_id"],
                    "canonical_slot": item["canonical_slot"],
                    "claim_text": item["text"],
                    "provenance": item["provenance"],
                }
                for item in capsules
            ]
            location = (
                int(entry["session_index"]),
                int(entry["parent_chunk_index"]),
                int(entry["subchunk_index"]),
            )
            candidate = {
                "memory_id": entry.get("candidate_id"),
                "session_id": entry["session_id"],
                "session_index": entry["session_index"],
                "parent_chunk_index": entry["parent_chunk_index"],
                "subchunk_index": entry["subchunk_index"],
                "score": entry.get("score", entry.get("semantic_logit")),
                "semantic_logit": entry.get("semantic_logit"),
                "channels": entry.get("channels"),
                "text": entry["text"],
                "unit_type": unit["unit_type"],
                "unit_types": [unit["unit_type"]],
                "canonical_slot": unit["canonical_slot"],
                "canonical_slots": [unit["canonical_slot"]],
                "capsule_id": [item["capsule_id"] for item in capsules],
                "role": list(entry["roles"]),
                "provenance": [item["provenance"] for item in capsules]
                or [entry.get("provenance")],
                "memory_contexts": memory_contexts,
                "attachments": attachments,
            }
            existing = evidence_by_location.get(location)
            if existing is None:
                evidence_by_location[location] = candidate
                continue
            for field in (
                "unit_types",
                "canonical_slots",
                "capsule_id",
                "role",
                "provenance",
                "memory_contexts",
                "attachments",
            ):
                for item in candidate[field]:
                    if item not in existing[field]:
                        existing[field].append(item)
    evidence_windows = list(evidence_by_location.values())
    for rank, item in enumerate(evidence_windows, start=1):
        item["rank"] = rank
    selected_session_ids = list(
        dict.fromkeys(str(item["session_id"]) for item in evidence_windows)
    )
    evidence = {"schema_version": SCHEMA_VERSION, "runtime_schema_version": "tmcra.v3.online-retrieval.3", "question_id": qid, "question": question, "question_date": question_date, "question_type": clean_text(row.get("question_type")), "selected_session_ids": selected_session_ids, "recall_plan": authoritative_plan, "evidence_windows": evidence_windows}
    runtime_input_has_gold = any(key in row for key in ("answer", "gold_answer", "answer_session_ids", "labels", "supervision"))
    debug = {"question_id": qid, "scope_id": scope_id, "db_path": str(db_path), "index_path": str(index_path), "recall_plan": authoritative_plan, "planner_decision_reason": planner_decision_reason, "planner_resolved_query": resolved_query, "recent_dialogue_count": len(recent_dialogue), "planner": planner_metadata, "cross_layer_weighted_fusion": False, "graph": graph_metadata, "runtime_input_has_gold": runtime_input_has_gold, "inventory_count": len(fast) + len(slow), "union_count": len(fast_ranked) + len(slow_ranked), "fast_inventory_count": len(fast), "fast_shortlist_count": len(fast_ranked), "slow_capsule_count": len(slow), "slow_shortlist_count": len(slow_ranked), "slow_dense_k": int(args.slow_dense_k), "fast_semantic_record_count": len(semantic_records), "planned_unit_count": len(units), "packed_unit_count": len(packed_units), "budget_excluded_unit_count": max(0, len(units) - len(packed_units)), "selected_count": len(evidence_windows), "atomic_unit_packing": True, "strict_no_truncation": True, "restart_boundary_verified": True, "graph_counts_before_query": counts_before, "graph_fingerprint": graph_fingerprint, "latency_sec": {"graph": round(graph_elapsed, 4), "dense": round(dense_elapsed, 4), "slow_dense": round(slow_dense_elapsed, 4), "cross": round(cross_elapsed, 4), "total": round(time.time() - started, 4)}}
    if any(key in row for key in ("answer", "gold_answer", "answer_session_ids", "labels", "supervision")):
        raise RuntimeError(f"{qid}: runtime retrieval manifest contains forbidden evaluation labels")
    return evidence, debug


def layered_retrieval_operation_id(out_dir: Path, scope_id: str, question_id: str) -> str:
    identity = "\n".join(
        (
            "tmcra.v3.layered-retrieval",
            str(out_dir.absolute()),
            clean_text(scope_id),
            clean_text(question_id),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _reconcile_completed_retrieval_output(
    *,
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    repo: Path,
) -> dict[str, Any] | None:
    """Reuse a committed retrieval directory and repair only missing idempotent audit state."""
    if not out_dir.exists():
        return None
    required = {
        "evidence": out_dir / "evidence_windows.jsonl",
        "debug": out_dir / "retrieval_debug.jsonl",
        "report": out_dir / "report.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"retrieval output directory is incomplete and will not be overwritten: "
            f"{out_dir} missing={','.join(missing)}"
        )
    evidence_rows = read_jsonl(required["evidence"])
    debug_rows = read_jsonl(required["debug"])
    report = json.loads(required["report"].read_text(encoding="utf-8"))
    expected_qids = [clean_text(row.get("question_id")) for row in rows]
    evidence_qids = [clean_text(row.get("question_id")) for row in evidence_rows]
    debug_qids = [clean_text(row.get("question_id")) for row in debug_rows]
    if (
        report.get("status") != "complete"
        or evidence_qids != expected_qids
        or debug_qids != expected_qids
    ):
        raise RuntimeError(f"existing retrieval output does not match its query manifest: {out_dir}")

    newly_appended = 0
    query_ids: list[str] = []
    for evidence, debug in zip(evidence_rows, debug_rows):
        plan = dict(evidence.get("recall_plan") or {})
        legacy_reason = clean_text(plan.pop("decision_reason", ""))
        if legacy_reason:
            observed_reason = clean_text(debug.get("planner_decision_reason"))
            if observed_reason and observed_reason != legacy_reason:
                raise RuntimeError(
                    f"{evidence['question_id']}: persisted planner reason disagrees with debug output"
                )
            debug["planner_decision_reason"] = legacy_reason
            evidence["recall_plan"] = plan
        operation_id = layered_retrieval_operation_id(
            out_dir,
            str(debug["scope_id"]),
            str(evidence["question_id"]),
        )
        persisted = append_layered_retrieval_audit(
            repo=repo,
            db_path=Path(debug["db_path"]),
            scope_id=str(debug["scope_id"]),
            operation_id=operation_id,
            evidence=evidence,
            debug=debug,
        )
        audit_payload = dict(persisted.get("payload") or {})
        query_id = clean_text(audit_payload.get("query_id"))
        if not query_id:
            raise RuntimeError(f"{evidence['question_id']}: persisted retrieval audit lacks query_id")
        newly_appended += int(bool(persisted.get("appended")))
        query_ids.append(query_id)
        debug["layered_retrieval_audit"] = {
            "operation_id": operation_id,
            "query_id": query_id,
            "event_total": int(persisted.get("event_total") or 0),
            "trimmed_total": int(persisted.get("trimmed_total") or 0),
            "newly_appended": bool(persisted.get("appended")),
        }

    report["layered_retrieval_audit_confirmed_count"] = len(query_ids)
    report["layered_retrieval_audit_newly_appended_count"] = newly_appended
    report["layered_retrieval_audit_query_ids"] = query_ids
    report["reused_completed_output"] = True
    _atomic_write_text(
        required["evidence"],
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in evidence_rows),
    )
    _atomic_write_text(
        required["debug"],
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in debug_rows),
    )
    _atomic_write_text(
        required["report"], json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def command_retrieve(args: argparse.Namespace) -> None:
    query_manifest = Path(args.query_manifest)
    try:
        rows = read_jsonl(query_manifest)
    except RuntimeError as exc:
        if str(exc) == f"no rows: {query_manifest}":
            raise RuntimeError(f"query manifest is empty: {query_manifest}") from exc
        raise
    out_dir = Path(args.out_dir).absolute()
    completed_report = _reconcile_completed_retrieval_output(
        out_dir=out_dir,
        rows=rows,
        repo=Path(args.repo),
    )
    if completed_report is not None:
        print(json.dumps(completed_report, indent=2, sort_keys=True))
        return
    runtime_env = graph_runtime_env(args)
    for path in (Path(args.node_model), Path(args.path_model), Path(args.checkpoint)):
        if not path.exists():
            raise FileNotFoundError(path)
    planner = planner_from_env()
    harness = load_native_harness(Path(args.harness), Path(args.repo))
    harness.disable_topic_bucket_runtime()
    models = OnlineModels(args)
    started = time.time()
    evidence_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    graph_adapter_cache: dict[tuple[str, str], Any] = {}
    newly_appended_audits = 0
    for row_index, row in enumerate(rows, start=1):
        evidence, debug = retrieve_one(
            row,
            args=args,
            harness=harness,
            models=models,
            planner=planner,
            graph_adapter_cache=graph_adapter_cache,
        )
        operation_id = layered_retrieval_operation_id(
            out_dir,
            str(debug["scope_id"]),
            str(evidence["question_id"]),
        )
        persisted_audit = append_layered_retrieval_audit(
            repo=Path(args.repo),
            db_path=Path(debug["db_path"]),
            scope_id=str(debug["scope_id"]),
            operation_id=operation_id,
            evidence=evidence,
            debug=debug,
        )
        newly_appended_audits += int(bool(persisted_audit.get("appended")))
        debug["layered_retrieval_audit"] = {
            "operation_id": operation_id,
            "query_id": clean_text(dict(persisted_audit.get("payload") or {}).get("query_id")),
            "event_total": int(persisted_audit.get("event_total") or 0),
            "trimmed_total": int(persisted_audit.get("trimmed_total") or 0),
            "newly_appended": bool(persisted_audit.get("appended")),
        }
        evidence_rows.append(evidence)
        debug_rows.append(debug)
        print(
            json.dumps(
                {
                    "status": "retrieved",
                    "row": row_index,
                    "total": len(rows),
                    "question_id": evidence["question_id"],
                    "inventory": debug["inventory_count"],
                    "union": debug["union_count"],
                    "selected": debug["selected_count"],
                    "latency_sec": debug["latency_sec"]["total"],
                }
            ),
            flush=True,
        )
    latency = [float(row["latency_sec"]["total"]) for row in debug_rows]
    report = {
        "status": "complete",
        "schema_version": "tmcra.v3.online-retrieval-report.2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "query_count": len(rows),
        "runtime_input_with_gold_count": sum(int(row["runtime_input_has_gold"]) for row in debug_rows),
        "restart_boundary_verified_count": sum(int(row["restart_boundary_verified"]) for row in debug_rows),
        "layered_retrieval_audit_confirmed_count": sum(
            int(bool(row.get("layered_retrieval_audit"))) for row in debug_rows
        ),
        "layered_retrieval_audit_newly_appended_count": newly_appended_audits,
        "layered_retrieval_audit_query_ids": [
            row["layered_retrieval_audit"]["query_id"] for row in debug_rows
        ],
        "checkpoint": str(models.checkpoint_path),
        "checkpoint_sha256": models.checkpoint_sha256,
        "cross_model_revision": models.cross_manifest.get("revision"),
        "graph_runtime_env": runtime_env,
        "strict_no_truncation": True,
        "cross_layer_weighted_fusion": False,
        "graph_adapter_cache_size": len(graph_adapter_cache),
        "planner": {
            "base_url_configured": bool(os.environ.get("TMCRA_RECALL_PLANNER_BASE_URL")),
            "model": os.environ.get("TMCRA_RECALL_PLANNER_MODEL", DEEPSEEK_FLASH_MODEL),
            "api_key_pool_size": len(planner.api_keys),
        },
        "top_k": args.top_k,
        "dense_k": args.dense_k,
        "graph_k": args.graph_k,
        "slow_dense_k": args.slow_dense_k,
        "atomic_unit_packing": True,
        "avg_planned_unit_count": round(sum(row["planned_unit_count"] for row in debug_rows) / len(debug_rows), 4),
        "avg_packed_unit_count": round(sum(row["packed_unit_count"] for row in debug_rows) / len(debug_rows), 4),
        "avg_budget_excluded_unit_count": round(sum(row["budget_excluded_unit_count"] for row in debug_rows) / len(debug_rows), 4),
        "avg_inventory_count": round(sum(row["inventory_count"] for row in debug_rows) / len(debug_rows), 4),
        "avg_union_count": round(sum(row["union_count"] for row in debug_rows) / len(debug_rows), 4),
        "avg_latency_sec": round(sum(latency) / len(latency), 4),
        "max_latency_sec": round(max(latency), 4),
        "elapsed_sec": round(time.time() - started, 3),
        "evidence": str(out_dir / "evidence_windows.jsonl"),
        "debug": str(out_dir / "retrieval_debug.jsonl"),
        "reused_completed_output": False,
    }
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = out_dir.with_name(
        f".{out_dir.name}.staging.{os.getpid()}.{time.time_ns()}"
    )
    staging.mkdir(parents=False, exist_ok=False)
    write_jsonl(staging / "evidence_windows.jsonl", evidence_rows)
    write_jsonl(staging / "retrieval_debug.jsonl", debug_rows)
    (staging / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging, out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-model", default=os.getenv("TMCRA_BGE_M3_PATH", "BAAI/bge-m3"))
    parser.add_argument("--text-dim", type=int, default=1024)
    parser.add_argument("--embedding-max-length", type=int, default=8192)
    parser.add_argument("--device", default="cuda")


def main() -> int:
    parser = argparse.ArgumentParser(description="TMCRA V3 production online index and retrieval runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("build-index")
    index.add_argument("--scope-manifest", required=True)
    index.add_argument("--out-report", required=True)
    index.add_argument("--subchunk-chars", type=int, default=1800)
    index.add_argument("--subchunk-overlap", type=int, default=200)
    index.add_argument("--batch-size", type=int, default=16)
    add_common_model_args(index)

    retrieve = sub.add_parser("retrieve")
    retrieve.add_argument("--query-manifest", required=True)
    retrieve.add_argument("--out-dir", required=True)
    retrieve.add_argument("--checkpoint", required=True)
    retrieve.add_argument("--cross-model", default=os.getenv("TMCRA_BGE_RERANKER_PATH", "BAAI/bge-reranker-v2-m3"))
    retrieve.add_argument("--cross-max-length", type=int, default=1280)
    retrieve.add_argument("--cross-batch-size", type=int, default=24)
    retrieve.add_argument("--repo", required=True)
    retrieve.add_argument("--harness", required=True)
    retrieve.add_argument("--node-model", required=True)
    retrieve.add_argument("--path-model", required=True)
    retrieve.add_argument("--graph-device", default="cuda")
    retrieve.add_argument("--candidate-event-k", type=int, default=24)
    retrieve.add_argument("--support-path-k", type=int, default=3)
    retrieve.add_argument("--path-tunnel-rescue-k", type=int, default=2)
    retrieve.add_argument("--graph-top-k", type=int, default=12)
    retrieve.add_argument("--dense-k", type=int, default=32)
    retrieve.add_argument("--slow-dense-k", type=int, default=24)
    retrieve.add_argument("--graph-k", type=int, default=24)
    retrieve.add_argument("--top-k", type=int, default=8)
    retrieve.add_argument("--max-per-parent", type=int, default=2)
    retrieve.add_argument("--max-per-session", type=int, default=4)
    add_common_model_args(retrieve)
    args = parser.parse_args()
    if args.command == "build-index":
        command_build_index(args)
    else:
        command_retrieve(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
