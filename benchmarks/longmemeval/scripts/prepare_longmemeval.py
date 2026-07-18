#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


QUESTION_TYPES = {
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
}


def load_and_validate(path: Path) -> list[dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("LongMemEval input must be a non-empty JSON array")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} is not an object")
        qid = str(row.get("question_id") or "").strip()
        if not qid or qid in seen:
            raise ValueError(f"invalid or duplicate question_id at row {index}: {qid!r}")
        seen.add(qid)
        question_type = str(row.get("question_type") or "").strip()
        if question_type not in QUESTION_TYPES:
            raise ValueError(f"{qid}: unexpected question_type {question_type!r}")
        for field in ("question", "answer", "haystack_sessions", "haystack_dates"):
            if field not in row:
                raise ValueError(f"{qid}: missing {field}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--qid-output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = load_and_validate(args.data)
    selected = rows[: args.limit] if args.limit else rows
    qids = [str(row["question_id"]) for row in selected]
    args.qid_output.parent.mkdir(parents=True, exist_ok=True)
    args.qid_output.write_text("\n".join(qids) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "tmcra.longmemeval-input.1",
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "source_count": len(rows),
        "selected_count": len(selected),
        "question_types": dict(sorted(Counter(str(row["question_type"]) for row in selected).items())),
        "qid_output": str(args.qid_output),
    }
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
