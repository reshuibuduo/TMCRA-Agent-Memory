#!/usr/bin/env python3
"""Audit user-memory ownership inside pasted or forwarded documents.

The deterministic router only selects document-shaped source messages. DeepSeek
Pro makes every semantic keep/quarantine decision from exact Source excerpts.
Original Source records remain immutable; quarantined Fast records stay in the
append-only graph but are excluded from current retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tmcra_v4_batch_writer import DeepSeekBatchClient


PROMPT_VERSION = "tmcra-v4-subject-attribution-2026-07-14.3"
MODEL = "deepseek-v4-pro"
CURRENT_STATES = {"active", "parallel_active", "promoted", "challenged"}
DECISIONS = {"keep_user", "quarantine_third_party", "quarantine_ambiguous"}

SYSTEM_PROMPT = """You audit subject ownership for a production personal-memory system.
Return exactly one JSON object and no prose:
{"decisions":[{"memory_id":"exact supplied ID","decision":"keep_user|quarantine_third_party|quarantine_ambiguous","actual_subject":"chat_user for keep_user; named local subject for quarantine_third_party; otherwise empty","chat_user_bridge_quote":"exact outside-artifact identity bridge for keep_user; otherwise empty","reason":"concise source-grounded reason"}]}.
Return every supplied candidate exactly once and never invent an ID.

The chat user is the person who submitted the outer message to this memory system. An author, sender,
signatory, quoted speaker, recipient, company, or document subject inside an embedded artifact is a separate
identity and must never be equated with the chat user merely because the artifact appears in a user-role
message. Text inside a pasted email thread, article, resume, log, transcript, or document is not the chat
user's conversational voice. A sender or signatory inside the artifact is third party by default.

keep_user is allowed only when text outside the embedded artifact explicitly bridges the chat user to the
local author or subject, for example "I wrote the email below" or "this is my resume". A sender name,
signature, first-person wording inside the artifact, or mailbox label is not an identity bridge. If the
source starts directly with mailbox UI or document text and contains no outside bridge, do not use
keep_user. A mailbox line such as "to me" identifies the mailbox owner as a recipient, not as the sender.
For keep_user, actual_subject must be exactly "chat_user". Use quarantine_third_party when the source
locally attributes the fact to a named author, sender, signatory, quoted speaker, company, or document
subject. Use quarantine_ambiguous when no actual subject can be established. Useful document facts remain
in immutable Source; do not force them into the chat user's Fast profile.
Every keep_user decision must cite chat_user_bridge_quote as an exact Source substring outside the artifact.
The candidate evidence quote itself cannot be used as that bridge. For both quarantine decisions,
chat_user_bridge_quote must be empty. Candidate objects intentionally contain no Writer claim or slot;
judge ownership from Source only.
Do not use outside knowledge, model confidence, benchmark labels, or the assertion wording as authority.
Judge against the exact source excerpts and offsets."""

HEADER_RE = re.compile(
    r"(?im)^(?:from|to|subject|sent|de|para|asunto|enviado):\s*\S"
)
MAILBOX_RE = re.compile(r"(?im)^\s*to\s+(?:me|[A-Z][^\n,]{0,40})(?:,|\s*$)")
SIGNATURE_RE = re.compile(r"(?im)^\s*(?:regards|best regards|sincerely),?\s*$")
LEGAL_RE = re.compile(r"(?i)confidentiality notice|electronic communications privacy act")
THREAD_RE = re.compile(r"(?i)view entire message|scanned by gmail|\bon .{0,120} wrote:\s*$", re.M)


class AttributionError(RuntimeError):
    pass


class AttributionClient(Protocol):
    def complete(self, payload: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
        ...


class DeepSeekProAttributionClient:
    def __init__(self) -> None:
        base_url = (
            os.getenv("TMCRA_DEEPSEEK_PRO_BASE_URL")
            or os.getenv("TMCRA_WRITER_BASE_URL")
            or "https://api.deepseek.com/v1"
        )
        raw_keys = (
            os.getenv("TMCRA_DEEPSEEK_PRO_KEY_POOL")
            or os.getenv("TMCRA_WRITER_API_KEY_POOL")
            or os.getenv("TMCRA_DEEPSEEK_WRITER_KEY_POOL")
            or ""
        )
        keys = [item.strip() for item in raw_keys.split(",") if item.strip()]
        if not keys:
            raise AttributionError("DeepSeek Pro key pool is empty; no fallback is allowed")
        max_tokens = int(os.getenv("TMCRA_DEEPSEEK_PRO_MAX_TOKENS", "16384"))
        self.client = DeepSeekBatchClient(
            base_url=base_url,
            model=MODEL,
            api_keys=keys,
            timeout=float(os.getenv("TMCRA_DEEPSEEK_PRO_TIMEOUT", "180")),
            max_tokens=max_tokens,
        )

    def complete(self, payload: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
        return self.client._complete(
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
            payload=payload,
            stage="subject_attribution_pro",
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def document_route_reasons(content: str) -> list[str]:
    """Route only; these signals never decide whether an assertion is valid."""
    reasons: list[str] = []
    if HEADER_RE.search(content):
        reasons.append("mail_header")
    if MAILBOX_RE.search(content):
        reasons.append("mailbox_recipient_line")
    if SIGNATURE_RE.search(content):
        reasons.append("signature_block")
    if LEGAL_RE.search(content):
        reasons.append("mail_legal_notice")
    if THREAD_RE.search(content):
        reasons.append("quoted_thread")
    return reasons if len(reasons) >= 2 else []


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(str(row["metadata_json"]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise AttributionError("record metadata is invalid JSON") from exc
    if not isinstance(value, dict):
        raise AttributionError("record metadata must be an object")
    return value


def _initialize(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS v4_subject_attribution_audits (
          audit_id TEXT PRIMARY KEY,
          scope_id TEXT NOT NULL,
          message_id TEXT NOT NULL,
          prompt_version TEXT NOT NULL,
          model TEXT NOT NULL,
          request_json TEXT NOT NULL,
          request_sha256 TEXT NOT NULL,
          status TEXT NOT NULL,
          response_json TEXT NOT NULL DEFAULT '',
          response_sha256 TEXT NOT NULL DEFAULT '',
          call_metadata_json TEXT NOT NULL DEFAULT '{}',
          decisions_json TEXT NOT NULL DEFAULT '[]',
          error TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(scope_id,message_id,request_sha256)
        )
        """
    )


def _source_rows(connection: sqlite3.Connection, scope_id: str) -> list[sqlite3.Row]:
    return connection.execute(
        "SELECT scope_id,message_id,session_index,message_index,message_role,"
        "source_turn_index,content,content_sha256 FROM v4_source_journal "
        "WHERE scope_id=? AND message_role='user' AND status='enriched' "
        "ORDER BY session_index,message_index",
        (scope_id,),
    ).fetchall()


def _candidate_records(
    connection: sqlite3.Connection, scope_id: str, source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rows = connection.execute(
        "SELECT memory_id,value,slot_key,turn_index,state,metadata_json FROM records "
        "WHERE scope_id=? AND turn_index=? ORDER BY memory_id",
        (scope_id, int(source["source_turn_index"])),
    ).fetchall()
    for row in rows:
        metadata = _metadata(row)
        if (
            metadata.get("content_variant") != "product_semantic_memory"
            or metadata.get("memory_layer") != "fast"
            or metadata.get("node_kind") != "atomic_user_assertion"
            or _text(row["state"]) not in CURRENT_STATES
            or _text(metadata.get("message_id")) != _text(source["message_id"])
        ):
            continue
        start = metadata.get("evidence_char_start")
        end = metadata.get("evidence_char_end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(str(source["content"]))
        ):
            raise AttributionError(f"{row['memory_id']}: evidence offsets are invalid")
        quote = str(source["content"])[start:end]
        expected_quote = _text(metadata.get("source_span") or metadata.get("raw_content"))
        if quote != expected_quote:
            raise AttributionError(f"{row['memory_id']}: Source quote drift")
        output.append(
            {
                "memory_id": str(row["memory_id"]),
                "claim_text": str(row["value"]),
                "canonical_slot": _text(
                    metadata.get("canonical_slot_key") or row["slot_key"]
                ),
                "evidence_quote": quote,
                "evidence_char_start": start,
                "evidence_char_end": end,
            }
        )
    return output


def _source_segments(content: str, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranges = [(0, min(len(content), 700))]
    for candidate in candidates:
        start = max(0, int(candidate["evidence_char_start"]) - 420)
        end = min(len(content), int(candidate["evidence_char_end"]) + 420)
        ranges.append((start, end))
    ranges.sort()
    merged: list[list[int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 80:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [
        {"char_start": start, "char_end": end, "text": content[start:end]}
        for start, end in merged
    ]


def scan_database(database: Path, scope_id: str) -> list[dict[str, Any]]:
    with closing(sqlite3.connect(database)) as connection:
        connection.row_factory = sqlite3.Row
        jobs: list[dict[str, Any]] = []
        for source in _source_rows(connection, scope_id):
            reasons = document_route_reasons(str(source["content"]))
            if not reasons:
                continue
            review_candidates = _candidate_records(connection, scope_id, source)
            if not review_candidates:
                continue
            candidates = [
                {
                    "memory_id": item["memory_id"],
                    "evidence_quote": item["evidence_quote"],
                    "evidence_char_start": item["evidence_char_start"],
                    "evidence_char_end": item["evidence_char_end"],
                }
                for item in review_candidates
            ]
            payload = {
                "scope_id": scope_id,
                "message_id": str(source["message_id"]),
                "message_role": "user",
                "route_reasons": reasons,
                "source_segments": _source_segments(
                    str(source["content"]), review_candidates
                ),
                "candidates": candidates,
            }
            jobs.append(
                {
                    "database": str(database),
                    "scope_id": scope_id,
                    "message_id": str(source["message_id"]),
                    "session_index": int(source["session_index"]),
                    "message_index": int(source["message_index"]),
                    "source_turn_index": int(source["source_turn_index"]),
                    "route_reasons": reasons,
                    "payload": payload,
                    "review_candidates": review_candidates,
                    "request_sha256": _digest(
                        {
                            "prompt_version": PROMPT_VERSION,
                            "model": MODEL,
                            "payload": payload,
                        }
                    ),
                }
            )
        return jobs


def validate_decisions(
    raw: Any, payload: Mapping[str, Any]
) -> list[dict[str, str]]:
    candidates = payload.get("candidates")
    source_segments = payload.get("source_segments")
    if not isinstance(candidates, list) or not isinstance(source_segments, list):
        raise AttributionError("attribution request payload is malformed")
    candidate_ids = [
        _text(item.get("memory_id"))
        for item in candidates
        if isinstance(item, Mapping)
    ]
    if len(candidate_ids) != len(candidates) or not all(candidate_ids):
        raise AttributionError("attribution request candidate identity is malformed")
    candidate_by_id = {
        _text(item["memory_id"]): item
        for item in candidates
        if isinstance(item, Mapping)
    }
    source_texts = [
        _text(item.get("text"))
        for item in source_segments
        if isinstance(item, Mapping) and _text(item.get("text"))
    ]
    if not isinstance(raw, Mapping) or set(raw) != {"decisions"}:
        raise AttributionError("attribution response must contain exactly decisions")
    decisions = raw.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(candidate_ids):
        raise AttributionError("attribution response decision count changed")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(decisions):
        if not isinstance(item, Mapping) or set(item) != {
            "memory_id",
            "decision",
            "actual_subject",
            "chat_user_bridge_quote",
            "reason",
        }:
            raise AttributionError(f"decisions[{index}] has an invalid shape")
        memory_id = _text(item.get("memory_id"))
        decision = _text(item.get("decision"))
        actual_subject = _text(item.get("actual_subject"))
        bridge_quote = _text(item.get("chat_user_bridge_quote"))
        reason = _text(item.get("reason"))
        if memory_id not in candidate_ids or memory_id in seen:
            raise AttributionError(f"decisions[{index}] changed candidate identity")
        if decision not in DECISIONS or not reason:
            raise AttributionError(f"decisions[{index}] has an invalid decision")
        if decision == "keep_user" and actual_subject != "chat_user":
            raise AttributionError(
                f"decisions[{index}] keep_user must bind actual_subject to chat_user"
            )
        if decision == "keep_user":
            evidence_quote = _text(candidate_by_id[memory_id].get("evidence_quote"))
            if (
                not bridge_quote
                or bridge_quote == evidence_quote
                or not any(bridge_quote in text for text in source_texts)
            ):
                raise AttributionError(
                    f"decisions[{index}] keep_user lacks an exact outside-artifact bridge"
                )
        elif bridge_quote:
            raise AttributionError(
                f"decisions[{index}] quarantine decision cannot cite a chat-user bridge"
            )
        if decision == "quarantine_third_party" and not actual_subject:
            raise AttributionError(
                f"decisions[{index}] must identify the third-party subject"
            )
        if decision == "quarantine_ambiguous" and actual_subject:
            raise AttributionError(
                f"decisions[{index}] ambiguous subject must remain empty"
            )
        seen.add(memory_id)
        normalized.append(
            {
                "memory_id": memory_id,
                "decision": decision,
                "actual_subject": actual_subject,
                "chat_user_bridge_quote": bridge_quote,
                "reason": reason,
            }
        )
    if seen != set(candidate_ids):
        raise AttributionError("attribution response omitted candidate identities")
    normalized.sort(key=lambda item: candidate_ids.index(item["memory_id"]))
    return normalized


def _replacement_head(
    connection: sqlite3.Connection,
    scope_id: str,
    slot_key: str,
    excluded_ids: set[str],
) -> str:
    rows = connection.execute(
        "SELECT h.memory_id,r.state FROM slot_history h JOIN records r "
        "ON r.scope_id=h.scope_id AND r.memory_id=h.memory_id "
        "WHERE h.scope_id=? AND h.slot_key=? ORDER BY h.ordinal DESC",
        (scope_id, slot_key),
    ).fetchall()
    return next(
        (
            str(row["memory_id"])
            for row in rows
            if str(row["memory_id"]) not in excluded_ids
            and _text(row["state"]) in CURRENT_STATES
        ),
        "",
    )


def _repair_slot_head(
    connection: sqlite3.Connection,
    scope_id: str,
    slot_key: str,
    memory_id: str,
    excluded_ids: set[str],
) -> None:
    head = connection.execute(
        "SELECT memory_id FROM slot_heads WHERE scope_id=? AND slot_key=?",
        (scope_id, slot_key),
    ).fetchone()
    if head is None or str(head["memory_id"]) != memory_id:
        return
    replacement = _replacement_head(connection, scope_id, slot_key, excluded_ids)
    if replacement:
        connection.execute(
            "UPDATE slot_heads SET memory_id=? WHERE scope_id=? AND slot_key=?",
            (replacement, scope_id, slot_key),
        )
    else:
        connection.execute(
            "DELETE FROM slot_heads WHERE scope_id=? AND slot_key=?",
            (scope_id, slot_key),
        )


def _apply_decisions(
    connection: sqlite3.Connection,
    job: Mapping[str, Any],
    audit_id: str,
    decisions: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    candidate_ids = [item["memory_id"] for item in job["payload"]["candidates"]]
    rows = connection.execute(
        "SELECT memory_id,slot_key,state,metadata_json FROM records WHERE scope_id=? "
        f"AND memory_id IN ({','.join('?' for _ in candidate_ids)})",
        (job["scope_id"], *candidate_ids),
    ).fetchall()
    if {str(row["memory_id"]) for row in rows} != set(candidate_ids):
        raise AttributionError("candidate records changed before attribution commit")
    decision_by_id = {item["memory_id"]: item for item in decisions}
    quarantine_ids = {
        item["memory_id"]
        for item in decisions
        if item["decision"] != "keep_user"
    }
    parent_signature_map: dict[str, tuple[str, Mapping[str, str]]] = {}
    for row in rows:
        memory_id = str(row["memory_id"])
        if memory_id not in quarantine_ids:
            continue
        signature = _text(_metadata(row).get("event_signature"))
        if not signature:
            continue
        if signature in parent_signature_map:
            raise AttributionError("quarantined parent event signature is not unique")
        parent_signature_map[signature] = (memory_id, decision_by_id[memory_id])

    dependent_rows: list[sqlite3.Row] = []
    if parent_signature_map:
        for row in connection.execute(
            "SELECT memory_id,slot_key,state,metadata_json FROM records WHERE scope_id=?",
            (job["scope_id"],),
        ).fetchall():
            if str(row["memory_id"]) in candidate_ids:
                continue
            parent_signature = _text(
                _metadata(row).get("facet_parent_event_signature")
            )
            if parent_signature in parent_signature_map:
                dependent_rows.append(row)

    dependent_ids = {str(row["memory_id"]) for row in dependent_rows}
    all_quarantine_ids = quarantine_ids | dependent_ids
    changed_ids: list[str] = []
    for row in rows:
        memory_id = str(row["memory_id"])
        decision = decision_by_id[memory_id]
        metadata = _metadata(row)
        metadata["subject_attribution_audit_id"] = audit_id
        metadata["subject_attribution_prompt_version"] = PROMPT_VERSION
        metadata["subject_attribution_model"] = MODEL
        metadata["subject_attribution_decision"] = decision["decision"]
        metadata["subject_attribution_actual_subject"] = decision["actual_subject"]
        metadata["subject_attribution_chat_user_bridge_quote"] = decision[
            "chat_user_bridge_quote"
        ]
        metadata["subject_attribution_reason"] = decision["reason"]
        state = str(row["state"])
        if memory_id in quarantine_ids:
            state = "quarantined"
            metadata["excluded_from_retrieval"] = True
            metadata["conflict_action"] = "subject_attribution_quarantine"
        connection.execute(
            "UPDATE records SET state=?,metadata_json=? WHERE scope_id=? AND memory_id=?",
            (state, _json(metadata), job["scope_id"], memory_id),
        )
        changed_ids.append(memory_id)
        if memory_id not in quarantine_ids:
            continue
        slot_key = str(row["slot_key"])
        _repair_slot_head(
            connection,
            job["scope_id"],
            slot_key,
            memory_id,
            all_quarantine_ids,
        )

    for row in dependent_rows:
        memory_id = str(row["memory_id"])
        metadata = _metadata(row)
        parent_signature = _text(metadata.get("facet_parent_event_signature"))
        parent_memory_id, parent_decision = parent_signature_map[parent_signature]
        metadata["subject_attribution_audit_id"] = audit_id
        metadata["subject_attribution_prompt_version"] = PROMPT_VERSION
        metadata["subject_attribution_model"] = MODEL
        metadata["subject_attribution_decision"] = "quarantine_with_parent"
        metadata["subject_attribution_parent_memory_id"] = parent_memory_id
        metadata["subject_attribution_parent_decision"] = parent_decision["decision"]
        metadata["subject_attribution_actual_subject"] = parent_decision[
            "actual_subject"
        ]
        metadata["subject_attribution_chat_user_bridge_quote"] = ""
        metadata["subject_attribution_reason"] = (
            "structural facet of quarantined parent assertion"
        )
        metadata["excluded_from_retrieval"] = True
        metadata["conflict_action"] = "subject_attribution_parent_quarantine"
        connection.execute(
            "UPDATE records SET state='quarantined',metadata_json=? "
            "WHERE scope_id=? AND memory_id=?",
            (_json(metadata), job["scope_id"], memory_id),
        )
        changed_ids.append(memory_id)
        _repair_slot_head(
            connection,
            job["scope_id"],
            str(row["slot_key"]),
            memory_id,
            all_quarantine_ids,
        )
    return {
        "changed_memory_ids": sorted(changed_ids),
        "quarantined_memory_ids": sorted(quarantine_ids),
        "cascaded_quarantined_memory_ids": sorted(dependent_ids),
        "kept_memory_ids": sorted(set(candidate_ids) - quarantine_ids),
    }


def _reused_application_state(
    connection: sqlite3.Connection,
    job: Mapping[str, Any],
    audit_id: str,
    decisions: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    candidate_ids = {item["memory_id"] for item in job["payload"]["candidates"]}
    quarantine_ids = {
        item["memory_id"] for item in decisions if item["decision"] != "keep_user"
    }
    dependent_ids: set[str] = set()
    for row in connection.execute(
        "SELECT memory_id,metadata_json FROM records WHERE scope_id=?",
        (job["scope_id"],),
    ).fetchall():
        metadata = _metadata(row)
        if (
            _text(metadata.get("subject_attribution_audit_id")) == audit_id
            and _text(metadata.get("subject_attribution_decision"))
            == "quarantine_with_parent"
            and _text(metadata.get("subject_attribution_parent_memory_id"))
            in quarantine_ids
        ):
            dependent_ids.add(str(row["memory_id"]))
    return {
        "changed_memory_ids": sorted(candidate_ids | dependent_ids),
        "quarantined_memory_ids": sorted(quarantine_ids),
        "cascaded_quarantined_memory_ids": sorted(dependent_ids),
        "kept_memory_ids": sorted(candidate_ids - quarantine_ids),
    }


def _cost(metadata: Mapping[str, Any]) -> float:
    usage = metadata.get("usage")
    usage = usage if isinstance(usage, Mapping) else metadata
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
    miss = int(usage.get("prompt_cache_miss_tokens", prompt - hit) or 0)
    prompt_rate = float(os.getenv("TMCRA_DEEPSEEK_PRO_PROMPT_COST_PER_MILLION", "3"))
    completion_rate = float(os.getenv("TMCRA_DEEPSEEK_PRO_COMPLETION_COST_PER_MILLION", "6"))
    cache_rate = float(os.getenv("TMCRA_DEEPSEEK_PRO_CACHE_COST_PER_MILLION", "0.025"))
    return (miss * prompt_rate + hit * cache_rate + completion * completion_rate) / 1_000_000


def execute_job(
    database: Path,
    job: Mapping[str, Any],
    client: AttributionClient,
) -> dict[str, Any]:
    audit_id = "saa_" + job["request_sha256"][:24]
    request_json = _json(job["payload"])
    with closing(sqlite3.connect(database, timeout=30)) as connection:
        connection.row_factory = sqlite3.Row
        _initialize(connection)
        existing = connection.execute(
            "SELECT status,decisions_json,call_metadata_json FROM "
            "v4_subject_attribution_audits WHERE audit_id=?",
            (audit_id,),
        ).fetchone()
        if existing is not None:
            if str(existing["status"]) != "completed":
                raise AttributionError(
                    f"{audit_id}: prior attribution call is not safely reusable"
                )
            decisions = json.loads(str(existing["decisions_json"]))
            applied = _reused_application_state(
                connection, job, audit_id, decisions
            )
            return {
                "audit_id": audit_id,
                "status": "reused",
                "physical_api_calls": 0,
                "decisions": decisions,
                "call_metadata": json.loads(str(existing["call_metadata_json"])),
                **applied,
            }
        now = _now()
        connection.execute(
            "INSERT INTO v4_subject_attribution_audits VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                audit_id,
                job["scope_id"],
                job["message_id"],
                PROMPT_VERSION,
                MODEL,
                request_json,
                job["request_sha256"],
                "api_started",
                "",
                "",
                "{}",
                "[]",
                "",
                now,
                now,
            ),
        )
        connection.commit()

    try:
        content, call_metadata = client.complete(job["payload"])
    except Exception as exc:
        with closing(sqlite3.connect(database, timeout=30)) as connection:
            connection.execute(
                "UPDATE v4_subject_attribution_audits SET status='failed',error=?,"
                "call_metadata_json=?,updated_at=? "
                "WHERE audit_id=? AND status='api_started'",
                (
                    str(exc),
                    _json(dict(getattr(exc, "metadata", {}) or {})),
                    _now(),
                    audit_id,
                ),
            )
            connection.commit()
        raise

    with closing(sqlite3.connect(database, timeout=30)) as connection:
        received = connection.execute(
            "UPDATE v4_subject_attribution_audits SET status='response_received',"
            "response_json=?,response_sha256=?,call_metadata_json=?,updated_at=? "
            "WHERE audit_id=? AND status='api_started'",
            (
                content,
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
                _json(dict(call_metadata)),
                _now(),
                audit_id,
            ),
        )
        if received.rowcount != 1:
            raise AttributionError("attribution journal changed before response save")
        connection.commit()
    try:
        raw = json.loads(content)
        decisions = validate_decisions(raw, job["payload"])
    except (json.JSONDecodeError, AttributionError) as exc:
        error = (
            "attribution response content is not JSON"
            if isinstance(exc, json.JSONDecodeError)
            else str(exc)
        )
        with closing(sqlite3.connect(database, timeout=30)) as connection:
            connection.execute(
                "UPDATE v4_subject_attribution_audits SET status='failed',error=?,"
                "updated_at=? WHERE audit_id=? AND status='response_received'",
                (error, _now(), audit_id),
            )
            connection.commit()
        raise AttributionError(error) from exc
    with closing(sqlite3.connect(database, timeout=30)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN IMMEDIATE")
        applied = _apply_decisions(connection, job, audit_id, decisions)
        superseded = connection.execute(
            "UPDATE v4_subject_attribution_audits SET status='superseded',error=?,"
            "updated_at=? WHERE scope_id=? AND message_id=? "
            "AND status IN ('completed','superseded') "
            "AND audit_id<>?",
            (
                f"superseded_by:{audit_id}",
                _now(),
                job["scope_id"],
                job["message_id"],
                audit_id,
            ),
        ).rowcount
        updated = connection.execute(
            "UPDATE v4_subject_attribution_audits SET status='completed',response_json=?,"
            "response_sha256=?,call_metadata_json=?,decisions_json=?,updated_at=? "
            "WHERE audit_id=? AND status='response_received'",
            (
                content,
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
                _json(dict(call_metadata)),
                _json(decisions),
                _now(),
                audit_id,
            ),
        )
        if updated.rowcount != 1:
            raise AttributionError("attribution journal changed before commit")
        connection.commit()
    return {
        "audit_id": audit_id,
        "status": "completed",
        "physical_api_calls": 1,
        "estimated_cost_cny": _cost(call_metadata),
        "decisions": decisions,
        "call_metadata": dict(call_metadata),
        "superseded_audit_count": int(superseded),
        **applied,
    }


def _worker_databases(run_dir: Path, workers: Sequence[str]) -> list[tuple[str, Path, str]]:
    selected = set(workers)
    output: list[tuple[str, Path, str]] = []
    for database in sorted((run_dir / "writer").glob("worker_*/native_memory.sqlite3")):
        worker = database.parent.name
        if selected and worker not in selected:
            continue
        with closing(sqlite3.connect(database)) as connection:
            row = connection.execute(
                "SELECT scope_id FROM v4_source_journal LIMIT 1"
            ).fetchone()
        if row is None or not _text(row[0]):
            raise AttributionError(f"{worker}: scope is missing")
        output.append((worker, database, _text(row[0])))
    if selected - {item[0] for item in output}:
        raise AttributionError(
            "unknown workers: " + ",".join(sorted(selected - {item[0] for item in output}))
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", default="")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    workers = [item.strip() for item in args.workers.split(",") if item.strip()]
    scanned: list[dict[str, Any]] = []
    worker_by_database: dict[str, str] = {}
    for worker, database, scope_id in _worker_databases(run_dir, workers):
        worker_by_database[str(database)] = worker
        for job in scan_database(database, scope_id):
            scanned.append({"worker": worker, **job})
    results: list[dict[str, Any]] = []
    if args.apply and scanned:
        client = DeepSeekProAttributionClient()
        for job in scanned:
            results.append(
                {
                    "worker": job["worker"],
                    "message_id": job["message_id"],
                    **execute_job(Path(job["database"]), job, client),
                }
            )
    report = {
        "schema_version": "tmcra.v4.subject-attribution-report.1",
        "status": "complete",
        "mode": "apply" if args.apply else "scan_only",
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "run_dir": str(run_dir),
        "scanned_worker_count": len(_worker_databases(run_dir, workers)),
        "routed_message_count": len(scanned),
        "routed_candidate_count": sum(
            len(job["payload"]["candidates"]) for job in scanned
        ),
        "physical_api_calls": sum(
            int(item.get("physical_api_calls", 0) or 0) for item in results
        ),
        "estimated_cost_cny": round(
            sum(float(item.get("estimated_cost_cny", 0.0) or 0.0) for item in results),
            8,
        ),
        "decision_quarantined_count": sum(
            len(item.get("quarantined_memory_ids", [])) for item in results
        ),
        "cascaded_quarantined_count": sum(
            len(item.get("cascaded_quarantined_memory_ids", []))
            for item in results
        ),
        "quarantined_count": sum(
            len(item.get("quarantined_memory_ids", []))
            + len(item.get("cascaded_quarantined_memory_ids", []))
            for item in results
        ),
        "routed": [
            {
                "worker": job["worker"],
                "database": job["database"],
                "scope_id": job["scope_id"],
                "message_id": job["message_id"],
                "session_index": job["session_index"],
                "message_index": job["message_index"],
                "source_turn_index": job["source_turn_index"],
                "route_reasons": job["route_reasons"],
                "request_sha256": job["request_sha256"],
                "candidate_count": len(job["payload"]["candidates"]),
                "candidates": job["review_candidates"],
            }
            for job in scanned
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
