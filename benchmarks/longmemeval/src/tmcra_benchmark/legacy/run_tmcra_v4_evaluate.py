#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from run_tmcra_v4_build import BASE, DEFAULT_DATA, _now, _resume_log, _run
from run_tmcra_v4_retrieve import DEFAULT_HARNESS
from tmcra_v4_route_policy import (
    POLICY_SCHEMA,
    PRODUCTION_ANSWER_MODEL,
    PRODUCTION_ANSWER_PROTOCOL,
    PRODUCTION_ANSWER_RUNNER,
    PRODUCTION_LANE,
    SOURCE_COVERAGE_TRACE_K,
    RoutePolicyError,
    assert_production_answer_runner,
    validate_production_answers,
    validate_production_evidence,
)
from tmcra_v4_regression_gate import (
    DEFAULT_CASES as DEFAULT_REGRESSION_CASES,
    evaluate_gate as evaluate_regression_gate,
)


PROJECT_ROOT = BASE.parents[2]
PACKAGE_ROOT = BASE.parent
DEFAULT_ANSWER_ENV = PROJECT_ROOT / "configs" / "answer.env"
DEFAULT_JUDGE = PACKAGE_ROOT / "harness" / "evaluate_qa_official_judge.py"
DEFAULT_ANSWER_RUNNER = BASE / PRODUCTION_ANSWER_RUNNER


class EvaluationError(RuntimeError):
    pass


def _load_environment(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise EvaluationError(f"answer environment file is missing: {path}")
    result = subprocess.run(
        ["bash", "-c", 'set -a; source "$1"; env -0', "tmcra-v4-answer-env", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output: dict[str, str] = {}
    for entry in result.stdout.decode("utf-8").split("\0"):
        if "=" in entry:
            key, value = entry.split("=", 1)
            output[key] = value
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source_top24_session_ids(
    debug_row: Mapping[str, Any], qid: str
) -> list[str]:
    trace = debug_row.get("source_top24_candidates")
    pool = debug_row.get("source_candidate_pool_trace")
    candidate_count = debug_row.get("source_candidate_count")
    trace_k = debug_row.get("source_coverage_trace_k")
    if type(candidate_count) is not int or candidate_count <= 0:
        raise EvaluationError(f"{qid}: Source candidate count is invalid")
    if trace_k != SOURCE_COVERAGE_TRACE_K:
        raise EvaluationError(
            f"{qid}: Source coverage trace contract is not Top{SOURCE_COVERAGE_TRACE_K}"
        )
    if not isinstance(trace, list) or not trace:
        raise EvaluationError(f"{qid}: Source Top24 coverage trace is missing")
    if not isinstance(pool, list):
        raise EvaluationError(f"{qid}: Source candidate pool trace is missing")
    expected_trace_count = min(SOURCE_COVERAGE_TRACE_K, candidate_count)
    if len(trace) != expected_trace_count:
        raise EvaluationError(
            f"{qid}: Source Top24 coverage trace has {len(trace)} rows; "
            f"expected {expected_trace_count}"
        )
    if len(pool) != candidate_count:
        raise EvaluationError(
            f"{qid}: Source candidate pool trace has {len(pool)} rows; "
            f"expected {candidate_count}"
        )

    def validate_trace_rows(
        rows: list[Any], *, label: str
    ) -> list[tuple[str, str, int, int, int]]:
        identities: list[tuple[str, str, int, int, int]] = []
        locations: set[tuple[str, int, int, int]] = set()
        for expected_rank, item in enumerate(rows, 1):
            if not isinstance(item, Mapping):
                raise EvaluationError(f"{qid}: {label} row must be an object")
            rank = item.get("rank")
            if type(rank) is not int or rank != expected_rank:
                raise EvaluationError(f"{qid}: {label} is malformed")
            candidate_id = str(item.get("candidate_id") or "").strip()
            session_id = str(item.get("session_id") or "").strip()
            coordinates = (
                item.get("session_index"),
                item.get("parent_chunk_index"),
                item.get("subchunk_index"),
            )
            if (
                not candidate_id
                or not session_id
                or any(type(value) is not int or value < 0 for value in coordinates)
            ):
                raise EvaluationError(f"{qid}: {label} has an invalid identity")
            location = (session_id, *coordinates)
            # Candidate IDs are content-derived and can collide when a dataset
            # repeats the same session/chunk at a different session_index. The
            # physical occurrence coordinates remain the authoritative identity.
            if location in locations:
                raise EvaluationError(f"{qid}: {label} contains duplicate evidence")
            locations.add(location)
            identities.append((candidate_id, session_id, *coordinates))
        return identities

    top_identities = validate_trace_rows(trace, label="Source Top24 coverage trace")
    pool_identities = validate_trace_rows(pool, label="Source candidate pool trace")
    if top_identities != pool_identities[:expected_trace_count]:
        raise EvaluationError(
            f"{qid}: Source Top24 trace is not the exact candidate-pool prefix"
        )
    session_ids: list[str] = []
    for item in trace:
        session_ids.append(str(item["session_id"]).strip())
    return list(dict.fromkeys(session_ids))


def _validate_retrieval_debug_rows(
    expected_qids: list[str], retrieval_debug: list[dict[str, Any]]
) -> None:
    actual_qids = [
        str(row.get("question_id") or row.get("qid") or "").strip()
        if isinstance(row, Mapping)
        else ""
        for row in retrieval_debug
    ]
    if actual_qids != expected_qids:
        raise EvaluationError("retrieval debug rows do not match frozen qid order")
    for qid, row in zip(expected_qids, retrieval_debug, strict=True):
        _source_top24_session_ids(row, qid)


def _reference_rows(value: Any, source: str = "reference") -> list[dict[str, Any]]:
    if isinstance(value, list):
        raw_rows = value
    elif isinstance(value, Mapping):
        raw_rows: Any = None
        for key in ("questions", "data", "rows", "references"):
            if key in value:
                raw_rows = value[key]
                break
        if raw_rows is None and value and all(isinstance(item, Mapping) for item in value.values()):
            raw_rows = [dict(item, qid=key) for key, item in value.items()]
        if raw_rows is None and ("qid" in value or "question_id" in value):
            raw_rows = [value]
        if not isinstance(raw_rows, list):
            raise EvaluationError(f"{source} must contain a list of rows")
    else:
        raise EvaluationError(f"{source} must be a JSON object or list")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(raw_rows, 1):
        if not isinstance(raw_row, Mapping):
            raise EvaluationError(f"{source} row {index} must be an object")
        qid_value = raw_row.get("qid")
        question_id = raw_row.get("question_id")
        if qid_value is not None and question_id is not None and str(qid_value).strip() != str(question_id).strip():
            raise EvaluationError(f"{source} row {index} has disagreeing qid and question_id")
        qid = str(qid_value or question_id or "").strip()
        if not qid:
            raise EvaluationError(f"{source} row {index} is missing qid")
        if qid in seen:
            raise EvaluationError(f"{source} contains duplicate qid {qid}")
        session_ids = raw_row.get("answer_session_ids")
        if not isinstance(session_ids, list) or not session_ids:
            raise EvaluationError(f"{source} row {index} ({qid}) lacks answer_session_ids")
        normalized_sessions = [str(item).strip() for item in session_ids]
        if any(not item for item in normalized_sessions) or len(set(normalized_sessions)) != len(normalized_sessions):
            raise EvaluationError(
                f"{source} row {index} ({qid}) has blank or duplicate answer_session_ids"
            )
        gold_key = next(
            (key for key in ("gold_answer", "answer", "expected_answer") if key in raw_row),
            None,
        )
        if gold_key is None:
            raise EvaluationError(f"{source} row {index} ({qid}) lacks a gold answer")
        rows.append(
            {
                "qid": qid,
                "gold_answer": raw_row[gold_key],
                "answer_session_ids": normalized_sessions,
                "question_type": str(
                    raw_row.get("question_type") or raw_row.get("type") or "unknown"
                ).strip()
                or "unknown",
            }
        )
        seen.add(qid)
    if not rows:
        raise EvaluationError(f"{source} is empty")
    return rows


def _read_reference(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationError(f"cannot read reference data: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError("reference data is not valid JSON") from exc
    return _reference_rows(value)


def _build_evaluation_audit(
    expected_qids: list[str],
    reference: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    retrieval_debug: list[dict[str, Any]],
    judged: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _validate_retrieval_debug_rows(expected_qids, retrieval_debug)
    reference_by_qid = {row["qid"]: row for row in reference}
    evidence_by_qid = {str(row.get("question_id") or row.get("qid") or "").strip(): row for row in evidence}
    judged_by_qid = {str(row.get("question_id") or row.get("qid") or "").strip(): row for row in judged}
    debug_by_qid = {
        str(row.get("question_id") or row.get("qid") or "").strip(): row
        for row in retrieval_debug
    }
    missing_reference = [qid for qid in expected_qids if qid not in reference_by_qid]
    missing_evidence = [qid for qid in expected_qids if qid not in evidence_by_qid]
    missing_judged = [qid for qid in expected_qids if qid not in judged_by_qid]
    missing_debug = [qid for qid in expected_qids if qid not in debug_by_qid]
    if missing_reference:
        raise EvaluationError(f"reference is missing evaluated qids: {missing_reference[:10]}")
    if missing_evidence:
        raise EvaluationError(f"evidence is missing evaluated qids: {missing_evidence[:10]}")
    if missing_judged:
        raise EvaluationError(f"judge is missing evaluated qids: {missing_judged[:10]}")
    if missing_debug:
        raise EvaluationError(
            f"retrieval debug is missing evaluated qids: {missing_debug[:10]}"
        )

    audit: list[dict[str, Any]] = []
    for qid in expected_qids:
        reference_row = reference_by_qid[qid]
        selected = evidence_by_qid[qid].get("selected_session_ids")
        if not isinstance(selected, list):
            raise EvaluationError(f"{qid}: selected_session_ids must be a list")
        selected_ids = [str(item).strip() for item in selected]
        if any(not item for item in selected_ids) or len(set(selected_ids)) != len(selected_ids):
            raise EvaluationError(f"{qid}: selected_session_ids has blank or duplicate values")
        answer_ids = list(reference_row["answer_session_ids"])
        source_top24_session_ids = _source_top24_session_ids(debug_by_qid[qid], qid)

        def coverage(candidate_session_ids: list[str]) -> dict[str, Any]:
            missing_ids = [
                session_id
                for session_id in answer_ids
                if session_id not in set(candidate_session_ids)
            ]
            return {
                "complete": not missing_ids,
                "expected_session_count": len(answer_ids),
                "selected_session_count": len(candidate_session_ids),
                "covered_session_count": len(answer_ids) - len(missing_ids),
                "hit_rate": round(
                    (len(answer_ids) - len(missing_ids)) / len(answer_ids), 6
                ),
                "missing_session_ids": missing_ids,
            }

        source_top24_coverage = coverage(source_top24_session_ids)
        final_evidence_coverage = coverage(selected_ids)
        official_label = official_judge_correct(judged_by_qid[qid])
        audit.append(
            {
                "qid": qid,
                "question_id": qid,
                "question_type": reference_row["question_type"],
                "gold_answer": reference_row["gold_answer"],
                "gold_answer_session_ids": answer_ids,
                "answer_session_ids": answer_ids,
                "source_top24_session_ids": source_top24_session_ids,
                "selected_session_ids": selected_ids,
                "source_top24_coverage": source_top24_coverage,
                "final_evidence_coverage": final_evidence_coverage,
                "session_coverage": final_evidence_coverage,
                "source_coverage_complete": source_top24_coverage["complete"],
                "source_top24_coverage_complete": source_top24_coverage["complete"],
                "final_evidence_coverage_complete": final_evidence_coverage["complete"],
                "missing_source_top24_session_ids": source_top24_coverage[
                    "missing_session_ids"
                ],
                "missing_session_ids": final_evidence_coverage["missing_session_ids"],
                "missing_answer_session_ids": final_evidence_coverage[
                    "missing_session_ids"
                ],
                "official_label": official_label,
                "official_judge_label": official_label,
                "official_correct": official_label,
            }
        )
    return audit


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_answer_facing_evidence(rows: list[dict[str, Any]]) -> None:
    forbidden_fields = {
        "answer",
        "gold_answer",
        "expected_answer",
        "answer_session_ids",
        "labels",
        "supervision",
    }
    for row in rows:
        qid = str(row.get("question_id") or "").strip() or "<unknown>"
        leaked = sorted(forbidden_fields.intersection(row))
        if leaked:
            raise EvaluationError(
                f"{qid}: answer-facing evidence contains benchmark fields: {', '.join(leaked)}"
            )


def official_judge_correct(row: Mapping[str, Any]) -> bool:
    if str(row.get("error") or "").strip():
        raise EvaluationError("official judge row contains an error")
    autoeval = row.get("autoeval_label")
    if not isinstance(autoeval, Mapping) or type(autoeval.get("label")) is not bool:
        raise EvaluationError(
            "official judge row lacks boolean autoeval_label.label"
        )
    return bool(autoeval["label"])


def _validate_judge_resume_rows(
    answers: list[dict[str, Any]], judged: list[dict[str, Any]]
) -> None:
    answer_qids = [str(row.get("question_id") or "").strip() for row in answers]
    judged_qids = [str(row.get("question_id") or "").strip() for row in judged]
    if judged_qids != answer_qids[: len(judged_qids)]:
        raise EvaluationError("existing judge rows are not a frozen answer prefix")
    answer_by_qid = {str(row["question_id"]): row for row in answers}
    for row in judged:
        qid = str(row.get("question_id") or "").strip()
        answer = answer_by_qid[qid]
        for key in ("question", "hypothesis", "evidence_sha256", "answer_model"):
            if row.get(key) != answer.get(key):
                raise EvaluationError(
                    f"{qid}: existing judge row is bound to different answer evidence"
                )


def _archive_failure_marker(run_dir: Path, retrieval_tag: str) -> None:
    marker = run_dir / f"FAILED.answer_{retrieval_tag}"
    if not marker.is_file():
        return
    history = run_dir / f"answer_{retrieval_tag}.failure_history.jsonl"
    with history.open("a", encoding="utf-8") as handle:
        for line in marker.read_text(encoding="utf-8").splitlines():
            if line.strip():
                handle.write(line.rstrip() + "\n")
    marker.unlink()


def _record_failure(run_dir: Path, retrieval_tag: str, exc: Exception) -> None:
    _archive_failure_marker(run_dir, retrieval_tag)
    marker = run_dir / f"FAILED.answer_{retrieval_tag}"
    temporary = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps({"at": _now(), "error": f"{exc.__class__.__name__}: {exc}"})
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    retrieval_dir = run_dir / args.retrieval_tag
    if not any(
        (retrieval_dir / marker).is_file()
        for marker in ("RETRIEVAL_COMPLETE", "COMPILE_COMPLETE")
    ):
        raise EvaluationError("retrieval or evidence compilation run is incomplete")
    answer_dir = run_dir / f"answer_{args.retrieval_tag}"
    resume = bool(getattr(args, "resume", False))
    failure_marker = run_dir / f"FAILED.answer_{args.retrieval_tag}"
    judge_path = run_dir / f"{args.retrieval_tag}.official_judge.jsonl"
    if answer_dir.exists() and not resume:
        raise EvaluationError(f"answer output already exists: {answer_dir}")
    if failure_marker.exists() and not resume:
        raise EvaluationError("prior answer failure exists; use explicit --resume")
    if resume and not (
        answer_dir.exists() or failure_marker.exists() or judge_path.exists()
    ):
        raise EvaluationError("resume requested but no prior answer state exists")
    try:
        qid_list = Path(
            getattr(args, "qid_list", None) or run_dir / "qids.txt"
        ).resolve()
        if not qid_list.is_file():
            raise EvaluationError(f"answer qid list is missing: {qid_list}")
        expected_qids = [
            line.strip()
            for line in qid_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not expected_qids or len(expected_qids) != len(set(expected_qids)):
            raise EvaluationError("answer qid list is empty or contains duplicates")
        baseline_judge = getattr(args, "baseline_judge", None)
        if baseline_judge is not None:
            baseline_judge = Path(baseline_judge).resolve()
            if not baseline_judge.is_file():
                raise EvaluationError(
                    f"frozen baseline judge is missing: {baseline_judge}"
                )

        try:
            assert_production_answer_runner(args.answer_runner.resolve())
            answer_facing_evidence = _read_jsonl(retrieval_dir / "evidence_windows.jsonl")
            _validate_answer_facing_evidence(answer_facing_evidence)
            route_report = validate_production_evidence(answer_facing_evidence)
        except RoutePolicyError as exc:
            raise EvaluationError(f"production route policy rejected the run: {exc}") from exc
        if [item.get("question_id") for item in answer_facing_evidence] != expected_qids:
            raise EvaluationError("evidence rows do not match frozen qid order")
        retrieval_debug = _read_jsonl(retrieval_dir / "retrieval_debug.jsonl")
        _validate_retrieval_debug_rows(expected_qids, retrieval_debug)

        loaded = _load_environment(args.answer_env.resolve())
        environment = {**os.environ, **loaded}
        if environment.get("TMCRA_ANSWER_MODEL") != PRODUCTION_ANSWER_MODEL:
            raise EvaluationError(
                f"answer layer must remain fixed to {PRODUCTION_ANSWER_MODEL}"
            )
        _run(
            [
                sys.executable,
                str(args.answer_runner.resolve()),
                "--evidence",
                str(retrieval_dir / "evidence_windows.jsonl"),
                "--harness",
                str(args.harness.resolve()),
                "--out-dir",
                str(answer_dir),
                "--workers",
                str(args.answer_workers),
                "--attempts",
                str(args.answer_attempts),
                "--answer-window-limit",
                str(args.answer_window_limit),
                "--qid-list",
                str(qid_list),
                "--lane",
                "production",
            ],
            (
                _resume_log(run_dir, f"answer_{args.retrieval_tag}")
                if resume
                else run_dir / f"answer_{args.retrieval_tag}.log"
            ),
            environment,
        )
        answers = _read_jsonl(answer_dir / "answers.jsonl")
        try:
            validate_production_answers(answers)
        except RoutePolicyError as exc:
            raise EvaluationError(f"production route policy rejected answers: {exc}") from exc
        if [item.get("question_id") for item in answers] != expected_qids:
            raise EvaluationError("answer rows do not match frozen qid order")
        judge_resume_options: list[str] = []
        if judge_path.is_file():
            if not resume:
                raise EvaluationError("judge output already exists without --resume")
            existing_judged = _read_jsonl(judge_path)
            _validate_judge_resume_rows(answers, existing_judged)
            judge_resume_options = ["--resume"]
        _run(
            [
                sys.executable,
                str(args.judge.resolve()),
                "--metric-model",
                "gpt-5.4",
                "--hyp-file",
                str(answer_dir / "answers.jsonl"),
                "--ref-file",
                str(args.data.resolve()),
                "--result-file",
                str(judge_path),
                "--base-url",
                environment["TMCRA_ANSWER_BASE_URL"],
                "--api-key-env",
                "TMCRA_ANSWER_API_KEY",
                "--timeout",
                "180",
                "--max-retries",
                "0",
                *judge_resume_options,
                "--quiet",
            ],
            (
                _resume_log(run_dir, f"{args.retrieval_tag}.judge")
                if resume
                else run_dir / f"{args.retrieval_tag}.judge.log"
            ),
            environment,
        )
        judged = _read_jsonl(judge_path)
        if [item.get("question_id") for item in answers] != expected_qids:
            raise EvaluationError("answer rows do not match frozen qid order")
        if [item.get("question_id") for item in judged] != expected_qids:
            raise EvaluationError("judge rows do not match frozen qid order")
        if any(item.get("answer_model") != PRODUCTION_ANSWER_MODEL for item in answers):
            raise EvaluationError(f"answer output did not use {PRODUCTION_ANSWER_MODEL}")
        labels = [official_judge_correct(item) for item in judged]
        correct_count = sum(labels)
        reference = _read_reference(args.data.resolve())
        evaluation_audit = _build_evaluation_audit(
            expected_qids,
            reference,
            answer_facing_evidence,
            retrieval_debug,
            judged,
        )
        evaluation_audit_path = answer_dir / "evaluation_audit.jsonl"
        _write_jsonl(evaluation_audit_path, evaluation_audit)
        source_complete_count = sum(
            bool(row["source_top24_coverage_complete"]) for row in evaluation_audit
        )
        final_complete_count = sum(
            bool(row["final_evidence_coverage_complete"]) for row in evaluation_audit
        )
        expected_gold_sessions = sum(
            int(row["source_top24_coverage"]["expected_session_count"])
            for row in evaluation_audit
        )
        source_covered_sessions = sum(
            int(row["source_top24_coverage"]["covered_session_count"])
            for row in evaluation_audit
        )
        final_covered_sessions = sum(
            int(row["final_evidence_coverage"]["covered_session_count"])
            for row in evaluation_audit
        )
        source_top24_hit_rate = (
            source_covered_sessions / expected_gold_sessions
            if expected_gold_sessions
            else 0.0
        )
        final_evidence_hit_rate = (
            final_covered_sessions / expected_gold_sessions
            if expected_gold_sessions
            else 0.0
        )
        accuracy = correct_count / len(judged) if judged else 0.0
        evaluation_quality_gate_passed = (
            source_top24_hit_rate == 1.0
            and final_evidence_hit_rate >= 0.9
            and accuracy >= 0.9
        )
        regression_gate_report: dict[str, Any] = {
            "schema_version": "tmcra.v4.regression-gate-reference.1",
            "status": "not_run",
            "promotion_eligible": False,
            "reason": "frozen baseline judge was not supplied",
        }
        if baseline_judge is not None:
            regression_evidence_path = answer_dir / "source_top24_regression_evidence.jsonl"
            _write_jsonl(
                regression_evidence_path,
                [
                    {
                        "qid": row["qid"],
                        "question_id": row["question_id"],
                        "selected_session_ids": row["source_top24_session_ids"],
                    }
                    for row in evaluation_audit
                ],
            )
            regression_gate_report = evaluate_regression_gate(
                baseline_judge,
                judge_path,
                regression_evidence_path,
                args.data.resolve(),
                cases_path=Path(
                    getattr(args, "regression_cases", DEFAULT_REGRESSION_CASES)
                ).resolve(),
                min_candidate_accuracy=None,
                max_regressions=0,
                min_improvements=0,
            )
            (answer_dir / "v4_regression_gate_report.json").write_text(
                json.dumps(regression_gate_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        regression_gate_passed = bool(
            regression_gate_report.get("promotion_eligible")
        )
        promotion_gate_passed = (
            evaluation_quality_gate_passed and regression_gate_passed
        )
        summary = {
            "schema_version": "tmcra.v4.evaluation.3",
            "status": "complete",
            "route_policy_schema": POLICY_SCHEMA,
            "lane": PRODUCTION_LANE,
            "answer_protocol": PRODUCTION_ANSWER_PROTOCOL,
            "promotion_eligible": promotion_gate_passed,
            "route_report": route_report,
            "retrieval_tag": args.retrieval_tag,
            "evaluated_count": len(judged),
            "answer_model": PRODUCTION_ANSWER_MODEL,
            "answer_attempt_limit": int(args.answer_attempts),
            "judge_retry_limit": 0,
            "judge_label_field": "autoeval_label.label",
            "correct_count": correct_count,
            "accuracy": round(accuracy, 6),
            "evaluation_quality_gate_passed": evaluation_quality_gate_passed,
            "promotion_gate": {
                "passed": promotion_gate_passed,
                "evaluation_quality_gate_passed": evaluation_quality_gate_passed,
                "zero_api_regression_gate_passed": regression_gate_passed,
                "source_top24_gold_session_hit_rate_min": 1.0,
                "final_evidence_gold_session_hit_rate_min": 0.9,
                "official_accuracy_min": 0.9,
                "max_regressions": 0,
            },
            "regression_gate": regression_gate_report,
            "evaluation_audit": {
                "path": str(evaluation_audit_path),
                "row_count": len(evaluation_audit),
                "reference_row_count": len(reference),
                "reference_is_superset": len(reference) >= len(expected_qids)
                and set(expected_qids).issubset({row["qid"] for row in reference}),
                "source_top24_coverage_complete_count": source_complete_count,
                "source_top24_coverage_complete_rate": round(
                    source_complete_count / len(evaluation_audit), 6
                )
                if evaluation_audit
                else 0.0,
                "source_top24_gold_session_hit_rate": round(
                    source_top24_hit_rate, 6
                ),
                "final_evidence_coverage_complete_count": final_complete_count,
                "final_evidence_coverage_complete_rate": round(
                    final_complete_count / len(evaluation_audit), 6
                )
                if evaluation_audit
                else 0.0,
                "final_evidence_gold_session_hit_rate": round(
                    final_evidence_hit_rate, 6
                ),
                "session_coverage_complete_count": final_complete_count,
                "session_coverage_complete_rate": round(
                    final_complete_count / len(evaluation_audit), 6
                )
                if evaluation_audit
                else 0.0,
            },
        }
        (answer_dir / "v4_evaluation_report.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (answer_dir / "ANSWER_COMPLETE").write_text(_now() + "\n", encoding="utf-8")
        _archive_failure_marker(run_dir, args.retrieval_tag)
        return summary
    except Exception as exc:
        _record_failure(run_dir, args.retrieval_tag, exc)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a completed TMCRA V4 retrieval run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--retrieval-tag", default="retrieval_1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--answer-env", type=Path, default=DEFAULT_ANSWER_ENV)
    parser.add_argument("--answer-runner", type=Path, default=DEFAULT_ANSWER_RUNNER)
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--qid-list", type=Path)
    parser.add_argument("--answer-workers", type=int, default=1)
    parser.add_argument("--answer-attempts", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--answer-window-limit", type=int, default=8)
    parser.add_argument("--baseline-judge", type=Path)
    parser.add_argument(
        "--regression-cases", type=Path, default=DEFAULT_REGRESSION_CASES
    )
    args = parser.parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
