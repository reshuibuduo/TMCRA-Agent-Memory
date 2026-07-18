#!/usr/bin/env python3
"""Run one isolated official LongMemEval judge process per question."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping


DEFAULT_OFFICIAL_JUDGE = (
    Path(__file__).resolve().parent.parent / "harness" / "evaluate_qa_official_judge.py"
)
METRIC_MODEL_ALIASES = {"gpt-4o-2024-08-06": "gpt-4o"}
BOUND_FIELDS = ("question", "hypothesis", "evidence_sha256", "answer_model")


class ParallelJudgeError(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl_atomic(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_result(row: Mapping[str, Any], answer: Mapping[str, Any]) -> None:
    qid = str(answer.get("question_id") or "").strip()
    if str(row.get("question_id") or "").strip() != qid:
        raise ParallelJudgeError(f"{qid}: judge result has a different question ID")
    for key in BOUND_FIELDS:
        if row.get(key) != answer.get(key):
            raise ParallelJudgeError(f"{qid}: judge result is bound to different {key}")
    label = row.get("autoeval_label")
    if (
        not isinstance(label, Mapping)
        or type(label.get("label")) is not bool
        or not str(label.get("model") or "").strip()
        or not str(label.get("raw_response") or "").strip()
    ):
        raise ParallelJudgeError(f"{qid}: judge result has no valid official label")


def _question_dir(work_dir: Path, index: int, qid: str) -> Path:
    digest = hashlib.sha256(qid.encode("utf-8")).hexdigest()[:16]
    return work_dir / f"{index:04d}_{digest}"


def _judge_one(
    *,
    index: int,
    answer: Mapping[str, Any],
    args: argparse.Namespace,
    environment: Mapping[str, str],
    work_dir: Path,
) -> tuple[int, dict[str, Any], bool]:
    qid = str(answer["question_id"])
    question_dir = _question_dir(work_dir, index, qid)
    question_dir.mkdir(parents=True, exist_ok=True)
    hypothesis_path = question_dir / "hypothesis.jsonl"
    result_path = question_dir / "official_judge.jsonl"
    log_path = question_dir / "official_judge.log"
    _write_jsonl_atomic(hypothesis_path, [answer])

    cached = _read_jsonl(result_path)
    if cached:
        if len(cached) != 1:
            raise ParallelJudgeError(f"{qid}: cached judge result is not one row")
        _validate_result(cached[0], answer)
        return index, cached[0], True

    command = [
        sys.executable,
        str(args.judge.resolve()),
        "--metric-model",
        METRIC_MODEL_ALIASES.get(args.metric_model, args.metric_model),
        "--hyp-file",
        str(hypothesis_path),
        "--ref-file",
        str(args.ref_file.resolve()),
        "--result-file",
        str(result_path),
        "--base-url",
        args.base_url,
        "--api-key-env",
        args.api_key_env,
        "--timeout",
        str(args.timeout),
        "--max-retries",
        str(args.max_retries),
        "--quiet",
    ]
    completed = subprocess.run(
        command,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=args.process_timeout,
        check=False,
    )
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise ParallelJudgeError(
            f"{qid}: official judge process exited {completed.returncode}; log={log_path}"
        )
    rows = _read_jsonl(result_path)
    if len(rows) != 1:
        raise ParallelJudgeError(f"{qid}: official judge process did not write one row")
    _validate_result(rows[0], answer)
    return index, rows[0], False


def run_parallel_judge(
    args: argparse.Namespace, *, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    environment = dict(environment or os.environ)
    if not environment.get(args.api_key_env):
        raise ParallelJudgeError(f"missing API key env: {args.api_key_env}")
    if not args.judge.is_file():
        raise ParallelJudgeError(f"official judge is missing: {args.judge}")

    hypotheses = _read_jsonl(args.hyp_file)
    if args.limit and args.limit > 0:
        hypotheses = hypotheses[: args.limit]
    expected_qids = [str(item.get("question_id") or "").strip() for item in hypotheses]
    if not expected_qids or any(not qid for qid in expected_qids):
        raise ParallelJudgeError("hypotheses are empty or contain empty question IDs")
    if len(expected_qids) != len(set(expected_qids)):
        raise ParallelJudgeError("hypotheses contain duplicate question IDs")

    result_file = args.result_file or Path(
        str(args.hyp_file) + f".eval-results-{args.metric_model}"
    )
    if result_file.exists() and not args.resume:
        raise ParallelJudgeError("judge result exists; use --resume")
    existing = _read_jsonl(result_file)
    if len(existing) > len(hypotheses):
        raise ParallelJudgeError("existing judge result is longer than hypotheses")
    for index, row in enumerate(existing):
        _validate_result(row, hypotheses[index])

    committed: dict[int, dict[str, Any]] = {
        index: row for index, row in enumerate(existing)
    }
    work_dir = args.work_dir or result_file.with_name(f"{result_file.name}.work")
    work_dir.mkdir(parents=True, exist_ok=True)
    pending_indices = list(range(len(existing), len(hypotheses)))
    failures: list[str] = []
    cached_count = 0
    called_count = 0

    def commit_contiguous_prefix() -> int:
        prefix: list[dict[str, Any]] = []
        for index in range(len(hypotheses)):
            if index not in committed:
                break
            prefix.append(committed[index])
        _write_jsonl_atomic(result_file, prefix)
        return len(prefix)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _judge_one,
                index=index,
                answer=hypotheses[index],
                args=args,
                environment=environment,
                work_dir=work_dir,
            ): index
            for index in pending_indices
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                completed_index, row, cached = future.result()
                committed[completed_index] = row
                cached_count += int(cached)
                called_count += int(not cached)
                commit_contiguous_prefix()
            except Exception as exc:
                failures.append(f"{expected_qids[index]}: {exc}")

    committed_count = commit_contiguous_prefix()
    report = {
        "status": "complete" if committed_count == len(hypotheses) else "incomplete",
        "metric_model": args.metric_model,
        "row_count": len(hypotheses),
        "committed_count": committed_count,
        "existing_count": len(existing),
        "cached_count": cached_count,
        "physical_call_count": called_count,
        "failure_count": len(failures),
        "failures": failures,
        "workers": args.workers,
        "work_dir": str(work_dir),
    }
    if failures or committed_count != len(hypotheses):
        report_path = result_file.with_name(f"{result_file.name}.isolated_report.json")
        temporary = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, report_path)
        raise ParallelJudgeError(
            f"official judge incomplete: committed={committed_count}/{len(hypotheses)}, "
            f"failures={len(failures)}; report={report_path}"
        )
    labels = [bool((committed[index]["autoeval_label"])["label"]) for index in range(len(hypotheses))]
    report["correct_count"] = sum(labels)
    report["accuracy"] = round(sum(labels) / len(labels), 6)
    report_path = result_file.with_name(f"{result_file.name}.isolated_report.json")
    temporary = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", type=Path, default=DEFAULT_OFFICIAL_JUDGE)
    parser.add_argument("--metric-model", default="gpt-4o")
    parser.add_argument("--hyp-file", required=True, type=Path)
    parser.add_argument("--ref-file", required=True, type=Path)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--process-timeout", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("TMCRA_OFFICIAL_JUDGE_WORKERS", "8")),
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ParallelJudgeError("workers must be positive")
    if args.process_timeout <= args.timeout:
        raise ParallelJudgeError("process timeout must exceed request timeout")
    summary = run_parallel_judge(args)
    if not args.quiet:
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
