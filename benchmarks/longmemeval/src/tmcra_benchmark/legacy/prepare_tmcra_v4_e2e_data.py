#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


WRITER_FIELDS = (
    "question_id",
    "haystack_sessions",
    "haystack_session_ids",
    "haystack_dates",
)
FORBIDDEN_WRITER_FIELDS = frozenset(
    {
        "question",
        "question_date",
        "question_type",
        "answer",
        "gold_answer",
        "answer_session_ids",
        "labels",
        "supervision",
    }
)


class PreparationError(RuntimeError):
    pass


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def validate_source_row(row: Mapping[str, Any]) -> None:
    qid = clean_text(row.get("question_id"))
    sessions = row.get("haystack_sessions")
    session_ids = row.get("haystack_session_ids")
    dates = row.get("haystack_dates")
    if not qid or not isinstance(sessions, list) or not sessions:
        raise PreparationError(f"{qid or '<missing-qid>'}: missing history sessions")
    if not isinstance(session_ids, list) or not isinstance(dates, list):
        raise PreparationError(f"{qid}: session IDs and dates must be arrays")
    if len(sessions) != len(session_ids) or len(sessions) != len(dates):
        raise PreparationError(f"{qid}: history/session/date lengths differ")
    for session_index, session in enumerate(sessions):
        if not isinstance(session, list) or not session:
            raise PreparationError(f"{qid}: session {session_index} is empty")
        for message_index, message in enumerate(session):
            if not isinstance(message, Mapping):
                raise PreparationError(
                    f"{qid}: session {session_index} message {message_index} is not an object"
                )
            role = clean_text(message.get("role")).lower()
            if role not in {"user", "assistant", "system", "tool"}:
                raise PreparationError(
                    f"{qid}: session {session_index} message {message_index} is invalid"
                )


def prepare(
    *, data_path: Path, qid_path: Path, out_dir: Path
) -> dict[str, Any]:
    qids = [
        clean_text(value)
        for value in qid_path.read_text(encoding="utf-8").splitlines()
        if clean_text(value)
    ]
    if not qids or len(qids) != len(set(qids)):
        raise PreparationError("qid list must be non-empty and unique")
    source = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise PreparationError("source dataset must be a JSON array")
    source_rows: dict[str, Mapping[str, Any]] = {}
    for raw_row in source:
        if not isinstance(raw_row, Mapping):
            raise PreparationError("source dataset contains a non-object row")
        validate_source_row(raw_row)
        source_qid = clean_text(raw_row.get("question_id"))
        if source_qid in source_rows:
            raise PreparationError(f"duplicate source question_id: {source_qid}")
        source_rows[source_qid] = raw_row
    missing = [qid for qid in qids if qid not in source_rows]
    if missing:
        raise PreparationError(f"missing qids in source data: {missing}")

    out_dir.mkdir(parents=True, exist_ok=False)
    scope_manifest: list[dict[str, Any]] = []
    query_manifest: list[dict[str, Any]] = []
    evaluation_refs: list[dict[str, Any]] = []
    workers: list[dict[str, Any]] = []
    writer_rows: list[dict[str, Any]] = []
    total_input_messages = 0
    total_nonempty_messages = 0
    total_empty_messages = 0
    total_duplicate_session_id_occurrences = 0
    duplicate_session_id_qids: list[str] = []
    for worker_index, qid in enumerate(qids):
        source_row = source_rows[qid]
        session_id_counts = Counter(
            clean_text(value)
            for value in source_row.get("haystack_session_ids") or []
            if clean_text(value)
        )
        duplicate_session_ids = {
            session_id: count
            for session_id, count in sorted(session_id_counts.items())
            if count > 1
        }
        duplicate_occurrences = sum(
            count - 1 for count in duplicate_session_ids.values()
        )
        if duplicate_occurrences:
            duplicate_session_id_qids.append(qid)
            total_duplicate_session_id_occurrences += duplicate_occurrences
        writer_row = {field: source_row.get(field) for field in WRITER_FIELDS}
        if FORBIDDEN_WRITER_FIELDS.intersection(writer_row):
            raise AssertionError("writer sanitizer retained a forbidden field")
        worker_dir = out_dir / "writer" / f"worker_{worker_index:03d}"
        worker_dir.mkdir(parents=True, exist_ok=False)
        writer_input = worker_dir / "input.json"
        writer_input.write_text(
            json.dumps([writer_row], ensure_ascii=False) + "\n", encoding="utf-8"
        )
        writer_rows.append(writer_row)
        database = worker_dir / "native_memory.sqlite3"
        scope_id = f"tmcra_v4:{qid}"
        index_path = out_dir / "indexes" / f"{qid}.pt"
        scope_manifest.append(
            {
                "question_id": qid,
                "db_path": str(database),
                "scope_id": scope_id,
                "index_path": str(index_path),
            }
        )
        query_manifest.append(
            {
                "question_id": qid,
                "question": clean_text(source_row.get("question")),
                "question_date": clean_text(source_row.get("question_date")),
                "question_type": clean_text(source_row.get("question_type")),
                "db_path": str(database),
                "scope_id": scope_id,
                "index_path": str(index_path),
            }
        )
        evaluation_refs.append(
            {
                "question_id": qid,
                "answer": source_row.get("answer"),
                "answer_session_ids": source_row.get("answer_session_ids"),
            }
        )
        workers.append(
            {
                "worker_index": worker_index,
                "question_id": qid,
                "worker_dir": str(worker_dir),
                "input": str(writer_input),
                "input_sha256": sha256_file(writer_input),
                "scope_id": scope_id,
                "session_count": len(source_row.get("haystack_sessions") or []),
                "message_count": sum(
                    len(session)
                    for session in source_row.get("haystack_sessions") or []
                ),
                "nonempty_message_count": sum(
                    1
                    for session in source_row.get("haystack_sessions") or []
                    for message in session
                    if str(message.get("content") or "").strip()
                ),
                "empty_message_count": sum(
                    1
                    for session in source_row.get("haystack_sessions") or []
                    for message in session
                    if not str(message.get("content") or "").strip()
                ),
                "duplicate_session_ids": duplicate_session_ids,
                "duplicate_session_id_occurrence_count": duplicate_occurrences,
            }
        )
        total_input_messages += workers[-1]["message_count"]
        total_nonempty_messages += workers[-1]["nonempty_message_count"]
        total_empty_messages += workers[-1]["empty_message_count"]
    (out_dir / "indexes").mkdir(parents=True, exist_ok=False)
    (out_dir / "writer_input.json").write_text(
        json.dumps(writer_rows, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_jsonl(out_dir / "scope_manifest.jsonl", scope_manifest)
    write_jsonl(out_dir / "query_manifest.jsonl", query_manifest)
    evaluation_dir = out_dir / "evaluation_only"
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(evaluation_dir / "references.jsonl", evaluation_refs)
    (out_dir / "qids.txt").write_text("\n".join(qids) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "tmcra.v4.e2e-input.1",
        "status": "prepared",
        "source_data": str(data_path.resolve()),
        "source_data_sha256": sha256_file(data_path),
        "row_count": len(qids),
        "input_message_count": total_input_messages,
        "nonempty_message_count": total_nonempty_messages,
        "empty_message_count": total_empty_messages,
        "duplicate_session_id_occurrence_count": total_duplicate_session_id_occurrences,
        "duplicate_session_id_qids": duplicate_session_id_qids,
        "qids": qids,
        "writer_fields": list(WRITER_FIELDS),
        "forbidden_writer_fields": sorted(FORBIDDEN_WRITER_FIELDS),
        "writer_inputs_have_query_or_evaluation_fields": False,
        "gold_isolation_dir": str(evaluation_dir),
        "workers": workers,
        "combined_writer_input": str(out_dir / "writer_input.json"),
        "scope_manifest": str(out_dir / "scope_manifest.jsonl"),
        "query_manifest": str(out_dir / "query_manifest.jsonl"),
    }
    (out_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare gold-isolated TMCRA V4 E2E inputs")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--qid-list", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    report = prepare(
        data_path=args.data.resolve(),
        qid_path=args.qid_list.resolve(),
        out_dir=args.out_dir.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
