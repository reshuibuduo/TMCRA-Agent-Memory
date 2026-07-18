#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label(row: dict[str, Any]) -> bool:
    value = row.get("autoeval_label")
    if isinstance(value, dict):
        value = value.get("label")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"invalid autoeval label: {value!r}")


def summarize(dataset: list[dict[str, Any]], judged: list[dict[str, Any]]) -> dict[str, Any]:
    type_by_qid = {str(row["question_id"]): str(row["question_type"]) for row in dataset}
    seen: set[str] = set()
    buckets: dict[str, list[bool]] = defaultdict(list)
    all_labels: list[bool] = []
    for row in judged:
        qid = str(row.get("question_id") or "")
        if qid not in type_by_qid or qid in seen:
            raise ValueError(f"unknown or duplicate question_id: {qid!r}")
        seen.add(qid)
        result = label(row)
        buckets[type_by_qid[qid]].append(result)
        all_labels.append(result)
    if len(seen) != len(dataset):
        missing = sorted(set(type_by_qid) - seen)
        raise ValueError(f"judge results are incomplete: {len(seen)}/{len(dataset)}; first missing={missing[:5]}")
    by_type = {}
    for name, values in sorted(buckets.items()):
        correct = sum(values)
        by_type[name] = {
            "correct": correct,
            "total": len(values),
            "accuracy": round(correct / len(values), 6),
            "accuracy_percent": round(correct * 100 / len(values), 1),
        }
    correct = sum(all_labels)
    return {
        "schema_version": "tmcra.longmemeval-scorecard.1",
        "benchmark": "LongMemEval",
        "correct": correct,
        "total": len(all_labels),
        "accuracy": round(correct / len(all_labels), 6),
        "accuracy_percent": round(correct * 100 / len(all_labels), 1),
        "by_question_type": by_type,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-results", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    scorecard = summarize(dataset, read_jsonl(args.judge_results))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(scorecard, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
