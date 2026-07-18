#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PRODUCTION_ANSWER_PROTOCOL = "evidence_operation_bound_v6"


def fail(message: str) -> None:
    raise SystemExit(message)


def require(path: Path) -> Path:
    if not path.is_file():
        fail(f"missing required artifact: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    require(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid JSON artifact {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON artifact must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"invalid JSONL row {path}:{line_number}: {exc}")
        if not isinstance(value, dict):
            fail(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(value)
    return rows


def qids(path: Path) -> list[str]:
    require(path)
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        fail(f"QID list is empty: {path}")
    if len(values) != len(set(values)):
        fail(f"QID list contains duplicates: {path}")
    return values


def row_qids(path: Path) -> list[str]:
    values = [str(row.get("question_id") or row.get("qid") or "").strip() for row in read_jsonl(path)]
    if any(not value for value in values):
        fail(f"artifact contains an empty QID: {path}")
    if len(values) != len(set(values)):
        fail(f"artifact contains duplicate QIDs: {path}")
    return values


def assert_qid_order(path: Path, expected: list[str]) -> None:
    actual = row_qids(path)
    if actual != expected:
        fail(f"QID mismatch in {path}: got {len(actual)} ordered rows, expected {len(expected)}")


def assert_complete_report(
    path: Path,
    *,
    expected_count: int,
    count_field: str,
) -> dict[str, Any]:
    report = read_json(path)
    if report.get("status") != "complete":
        fail(f"report is not complete: {path}")
    try:
        actual_count = int(report.get(count_field, -1))
    except (TypeError, ValueError):
        fail(f"report has an invalid {count_field}: {path}")
    if actual_count != expected_count:
        fail(f"report {count_field} mismatch in {path}: {actual_count}/{expected_count}")
    return report


def validate_build(run_dir: Path, expected: list[str]) -> None:
    require(run_dir / "BUILD_COMPLETE")
    require(run_dir / "qids.txt")
    if qids(run_dir / "qids.txt") != expected:
        fail("build qids.txt does not match the requested QID order")

    manifest = read_json(run_dir / "input_manifest.json")
    if manifest.get("status") != "prepared":
        fail("build input manifest is not in prepared state")
    if list(manifest.get("qids") or []) != expected:
        fail("build input manifest QIDs do not match the requested QID order")
    if int(manifest.get("row_count", -1)) != len(expected):
        fail("build input manifest row_count is inconsistent")
    workers = manifest.get("workers")
    if not isinstance(workers, list):
        fail("build input manifest workers must be a list")
    worker_qids = [str(row.get("question_id") or "").strip() for row in workers if isinstance(row, Mapping)]
    if len(worker_qids) != len(workers) or worker_qids != expected:
        fail("build worker QIDs do not match the requested QID order")

    assert_qid_order(run_dir / "scope_manifest.jsonl", expected)
    assert_qid_order(run_dir / "query_manifest.jsonl", expected)
    assert_complete_report(
        run_dir / "build_report.json",
        expected_count=len(expected),
        count_field="row_count",
    )
    assert_complete_report(
        run_dir / "index_report.json",
        expected_count=len(expected),
        count_field="row_count",
    )


def validate_retrieve(run_dir: Path, tag: str, expected: list[str]) -> None:
    directory = run_dir / tag
    require(directory / "RETRIEVAL_COMPLETE")
    assert_complete_report(
        directory / "report.json",
        expected_count=len(expected),
        count_field="query_count",
    )
    run_report = assert_complete_report(
        directory / "retrieval_run_report.json",
        expected_count=len(expected),
        count_field="query_count",
    )
    if str(run_report.get("tag") or "") != tag:
        fail(f"retrieval tag mismatch in {directory / 'retrieval_run_report.json'}")
    assert_qid_order(directory / "evidence_windows.jsonl", expected)
    assert_qid_order(directory / "retrieval_debug.jsonl", expected)


def validate_compile(run_dir: Path, expected: list[str]) -> None:
    require(run_dir / "COMPILE_COMPLETE")
    assert_complete_report(
        run_dir / "report.json",
        expected_count=len(expected),
        count_field="row_count",
    )
    assert_qid_order(run_dir / "evidence_windows.jsonl", expected)
    assert_qid_order(run_dir / "retrieval_debug.jsonl", expected)


def validate_answer(run_dir: Path, expected: list[str]) -> None:
    report = assert_complete_report(
        run_dir / "report.json",
        expected_count=len(expected),
        count_field="requested",
    )
    if int(report.get("completed", -1)) != len(expected):
        fail(f"answer report completed count mismatch: {report.get('completed')}/{len(expected)}")
    if int(report.get("failure_count", -1)) != 0:
        fail(f"answer report contains failures: {report.get('failure_count')}")
    if report.get("answer_model") != "gpt-5.4":
        fail(f"answer report used an unexpected model: {report.get('answer_model')!r}")
    if report.get("answer_protocol") != PRODUCTION_ANSWER_PROTOCOL:
        fail(f"answer report used an unexpected protocol: {report.get('answer_protocol')!r}")
    if report.get("lane") != "production" or report.get("promotion_eligible") is not True:
        fail("answer report is not a promotion-eligible production run")
    if report.get("compiled_evidence_packet_required") is not True:
        fail("answer report did not require compiled evidence packets")
    answers_path = run_dir / "answers.jsonl"
    assert_qid_order(answers_path, expected)
    for row in read_jsonl(answers_path):
        if row.get("answer_model") != "gpt-5.4":
            fail(f"answer row used an unexpected model: {row.get('question_id')}")
        if row.get("answer_protocol") != PRODUCTION_ANSWER_PROTOCOL:
            fail(f"answer row used an unexpected protocol: {row.get('question_id')}")


def validate_judge(results: Path, expected: list[str]) -> None:
    assert_qid_order(results, expected)
    rows = read_jsonl(results)
    for row in rows:
        label = row.get("autoeval_label")
        if not isinstance(label, Mapping) or type(label.get("label")) is not bool:
            fail(f"judge row has no valid boolean label: {row.get('question_id')}")
        if label.get("model") != "gpt-4o-2024-08-06":
            fail(f"judge row used an unexpected model: {label.get('model')!r}")

    report_path = results.with_name(f"{results.name}.isolated_report.json")
    report = read_json(report_path)
    if report.get("status") != "complete":
        fail(f"judge report is not complete: {report_path}")
    if int(report.get("row_count", -1)) != len(expected):
        fail(f"judge row_count mismatch in {report_path}")
    if int(report.get("committed_count", -1)) != len(expected):
        fail(f"judge committed_count mismatch in {report_path}")
    if int(report.get("failure_count", -1)) != 0:
        fail(f"judge report contains failures: {report_path}")
    if report.get("metric_model") not in {"gpt-4o", "gpt-4o-2024-08-06"}:
        fail(f"judge report used an unexpected model alias: {report.get('metric_model')!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("build", "retrieve", "compile", "answer", "judge"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-qids", required=True, type=Path)
    parser.add_argument("--tag", default="retrieval_s500")
    parser.add_argument("--judge-results", type=Path)
    args = parser.parse_args()
    expected = qids(args.expected_qids)

    if args.stage == "build":
        validate_build(args.run_dir, expected)
    elif args.stage == "retrieve":
        validate_retrieve(args.run_dir, args.tag, expected)
    elif args.stage == "compile":
        validate_compile(args.run_dir, expected)
    elif args.stage == "answer":
        validate_answer(args.run_dir, expected)
    else:
        if args.judge_results is None:
            fail("--judge-results is required for the judge stage")
        validate_judge(args.judge_results, expected)

    print(f"{args.stage} validation: OK ({len(expected)} QIDs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
