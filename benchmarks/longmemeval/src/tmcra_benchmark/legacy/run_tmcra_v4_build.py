#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from prepare_tmcra_v4_e2e_data import prepare
from tmcra_v4_cost_report import build_report, collect_calls
from tmcra_v4_slow_graph import (
    PROCESS_LOSS_INTERRUPTION_ERROR,
    SlowGraphStore,
    load_graph_schema,
)


BASE = Path(__file__).resolve().parent
PACKAGE_ROOT = BASE.parent
PROJECT_ROOT = BASE.parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "longmemeval_s_cleaned.json"
DEFAULT_REPO = PACKAGE_ROOT / "vendor" / "tmcra_integrated"
DEFAULT_WRITER_ENV = PROJECT_ROOT / "configs" / "writer.env"
DEFAULT_EMBEDDING = PROJECT_ROOT / "models" / "BAAI" / "bge-m3"
SUBJECT_ATTRIBUTION_PROMPT_VERSION = "tmcra-v4-subject-attribution-2026-07-14.3"


class BuildError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stage(root: Path, name: str, **values: Any) -> None:
    record = {"at": _now(), "stage": name, **values}
    with (root / "build.log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def _load_shell_environment(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise BuildError(f"writer environment file is missing: {path}")
    command = 'set -a; source "$1"; env -0'
    result = subprocess.run(
        ["bash", "-c", command, "tmcra-v4-env", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    loaded: dict[str, str] = {}
    for entry in result.stdout.decode("utf-8").split("\0"):
        if "=" in entry:
            key, value = entry.split("=", 1)
            loaded[key] = value
    return loaded


def _key_pool(environment: Mapping[str, str]) -> list[str]:
    raw = environment.get("TMCRA_DEEPSEEK_WRITER_KEY_POOL", "")
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    if not keys or len(keys) != len(set(keys)):
        raise BuildError("DeepSeek writer key pool must be non-empty and unique")
    configured_count = environment.get("TMCRA_DEEPSEEK_WRITER_KEY_POOL_COUNT", "")
    if configured_count and int(configured_count) != len(keys):
        raise BuildError("DeepSeek writer key pool count does not match actual keys")
    return keys


def _rotated(keys: Sequence[str], worker_index: int) -> str:
    offset = worker_index % len(keys)
    return ",".join([*keys[offset:], *keys[:offset]])


def _worker_environment(base: Mapping[str, str], keys: Sequence[str], worker_index: int) -> dict[str, str]:
    environment = dict(base)
    pool = _rotated(keys, worker_index)
    base_url = environment.get("TMCRA_DEEPSEEK_WRITER_BASE_URL") or environment.get("TMCRA_WRITER_BASE_URL") or "https://api.deepseek.com/v1"
    max_tokens = environment.get("TMCRA_WRITER_MAX_TOKENS", "16384")
    environment.update(
        {
            "TMCRA_WRITER_MAX_TOKENS": max_tokens,
            "TMCRA_WRITER_BASE_URL": base_url,
            "TMCRA_WRITER_MODEL": "deepseek-v4-flash",
            "TMCRA_WRITER_API_KEY_POOL": pool,
            "TMCRA_DEEPSEEK_FLASH_BASE_URL": base_url,
            "TMCRA_DEEPSEEK_FLASH_KEY_POOL": pool,
            "TMCRA_DEEPSEEK_FLASH_MAX_TOKENS": max_tokens,
            "TMCRA_DEEPSEEK_FLASH_PROMPT_COST_PER_MILLION": "1",
            "TMCRA_DEEPSEEK_FLASH_COMPLETION_COST_PER_MILLION": "2",
            "TMCRA_DEEPSEEK_FLASH_CACHE_COST_PER_MILLION": "0.02",
            "TMCRA_DEEPSEEK_PRO_BASE_URL": base_url,
            "TMCRA_DEEPSEEK_PRO_KEY_POOL": pool,
            "TMCRA_DEEPSEEK_PRO_MAX_TOKENS": max_tokens,
            "TMCRA_DEEPSEEK_PRO_PROMPT_COST_PER_MILLION": "3",
            "TMCRA_DEEPSEEK_PRO_COMPLETION_COST_PER_MILLION": "6",
            "TMCRA_DEEPSEEK_PRO_CACHE_COST_PER_MILLION": "0.025",
        }
    )
    return environment


def _run(command: Sequence[str], log_path: Path, environment: Mapping[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"command": list(command)}, sort_keys=True) + "\n")
        log.flush()
        subprocess.run(
            list(command),
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(environment),
        )


def _resume_log(directory: Path, stem: str) -> Path:
    return directory / f"{stem}.resume.{time.time_ns()}.log"


def _load_resume_manifest(out_dir: Path) -> dict[str, Any]:
    manifest_path = out_dir / "input_manifest.json"
    if not manifest_path.is_file():
        raise BuildError("resume run has no input_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError("resume input manifest is unreadable") from exc
    if not isinstance(manifest, Mapping):
        raise BuildError("resume input manifest must be an object")
    workers = manifest.get("workers")
    if (
        manifest.get("status") != "prepared"
        or not isinstance(workers, list)
        or not workers
        or int(manifest.get("row_count", 0) or 0) != len(workers)
    ):
        raise BuildError("resume input manifest is incomplete or stale")
    return dict(manifest)


def _verify_resume_writer(worker: Mapping[str, Any]) -> None:
    worker_dir = Path(str(worker["worker_dir"])).resolve()
    database = worker_dir / "native_memory.sqlite3"
    report_path = worker_dir / "product_writer_report.json"
    audit_path = worker_dir / "writer_chain_audit.json"
    if not all(path.is_file() for path in (database, report_path, audit_path)):
        raise BuildError(f"resume writer artifacts are incomplete: {worker_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if report.get("completed") is not True or audit.get("passed") is not True:
        raise BuildError(f"resume writer audit is not complete: {worker_dir}")
    with closing(sqlite3.connect(database)) as con:
        statuses = dict(
            con.execute("SELECT status,count(*) FROM v4_batch_journal GROUP BY status")
        )
    if statuses != {"committed": int(report.get("batches", -1))}:
        raise BuildError(
            f"resume writer journal is not fully committed: {worker_dir}: {statuses}"
        )


def _interrupted_writer_calls(worker: Mapping[str, Any]) -> list[dict[str, str]]:
    worker_dir = Path(str(worker["worker_dir"])).resolve()
    database = worker_dir / "native_memory.sqlite3"
    if not database.is_file():
        return []
    output: list[dict[str, str]] = []
    with closing(sqlite3.connect(database)) as con:
        con.row_factory = sqlite3.Row
        for row in con.execute(
            "SELECT batch_id FROM v4_batch_journal WHERE status='api_started' ORDER BY batch_index"
        ):
            output.append(
                {
                    "worker": str(worker["question_id"]),
                    "stage": "batch_flash",
                    "call_key": f"flash:{row['batch_id']}",
                }
            )
        if con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='v4_reconciliation_jobs'"
        ).fetchone():
            for row in con.execute(
                "SELECT job_id FROM v4_reconciliation_jobs WHERE status='pro_started' ORDER BY created_at,job_id"
            ):
                output.append(
                    {
                        "worker": str(worker["question_id"]),
                        "stage": "reconciliation_pro",
                        "call_key": f"pro:{row['job_id']}",
                    }
                )
    return output


def _interrupted_slow_calls(worker: Mapping[str, Any]) -> list[dict[str, Any]]:
    worker_dir = Path(str(worker["worker_dir"])).resolve()
    database = worker_dir / "native_memory.sqlite3"
    if not database.is_file():
        return []
    output: list[dict[str, Any]] = []
    with closing(sqlite3.connect(database)) as con:
        con.row_factory = sqlite3.Row
        tables = {
            str(row[0])
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not {"slow_graph_jobs", "slow_graph_attempts"} <= tables:
            return []
        recovered_attempts: set[str] = set()
        if "slow_graph_process_loss_recoveries" in tables:
            recovered_attempts = {
                str(row[0])
                for row in con.execute(
                    "SELECT attempt_id FROM slow_graph_process_loss_recoveries"
                )
            }
        rows = con.execute(
            "SELECT j.job_id,j.scope_id,j.region_key,j.status,j.claim_owner,"
            "j.lease_expires_at,a.attempt_id,a.status AS attempt_status,"
            "a.created_at FROM slow_graph_jobs j JOIN slow_graph_attempts a "
            "ON a.job_id=j.job_id WHERE ("
            "(j.status='pending' AND j.claim_token IS NOT NULL "
            "AND a.status='started' AND a.claim_token=j.claim_token "
            "AND a.claim_owner=j.claim_owner) OR "
            "(j.status='failed' AND j.claim_token IS NULL AND j.last_error=? "
            "AND a.status='expired' AND a.error=?)) "
            "ORDER BY a.created_at,a.attempt_id",
            (PROCESS_LOSS_INTERRUPTION_ERROR, PROCESS_LOSS_INTERRUPTION_ERROR),
        ).fetchall()
        for row in rows:
            if str(row["attempt_id"]) in recovered_attempts:
                continue
            output.append(
                {
                    "worker": str(worker.get("question_id") or worker_dir.name),
                    **dict(row),
                }
            )
    return output


def _recover_interrupted_slow_calls(
    worker: Mapping[str, Any], *, repo: Path
) -> list[dict[str, Any]]:
    worker_dir = Path(str(worker["worker_dir"])).resolve()
    database = worker_dir / "native_memory.sqlite3"
    store = SlowGraphStore(database, schema=load_graph_schema(repo))
    reviewed = store.interrupted_process_loss_attempts()
    reports = [
        store.recover_interrupted_process_loss(
            str(item["job_id"]),
            expected_attempt_id=str(item["attempt_id"]),
        )
        for item in reviewed
    ]
    if reports:
        report_path = _resume_log(worker_dir, "slow_process_loss_recovery").with_suffix(
            ".json"
        )
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": "tmcra.v4.slow-process-loss-worker-report.1",
                    "worker": str(worker.get("question_id") or worker_dir.name),
                    "physical_api_calls_during_recovery": 0,
                    "recoveries": reports,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return reports


def _failed_slow_jobs(database: Path) -> list[dict[str, str]]:
    with closing(sqlite3.connect(database)) as con:
        con.row_factory = sqlite3.Row
        if con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='slow_graph_jobs'"
        ).fetchone() is None:
            return []
        return [
            {
                "job_id": str(row["job_id"]),
                "region_key": str(row["region_key"]),
                "last_error": str(row["last_error"]),
            }
            for row in con.execute(
                "SELECT job_id,region_key,last_error FROM slow_graph_jobs "
                "WHERE status IN ('failed','retryable') ORDER BY created_at,job_id"
            )
        ]


def _slow_worker_resume(
    worker: Mapping[str, Any],
    *,
    repo: Path,
    environment: Mapping[str, str],
    enqueue: bool = True,
    recover_interrupted_slow_calls: bool = False,
) -> None:
    worker_dir = Path(str(worker["worker_dir"])).resolve()
    database = worker_dir / "native_memory.sqlite3"
    scope_id = str(worker["scope_id"])
    prefix = [
        sys.executable,
        str(BASE / "tmcra_v4_slow_graph.py"),
        str(database),
        "--repo",
        str(repo),
    ]
    interrupted = _interrupted_slow_calls(worker)
    if interrupted and not recover_interrupted_slow_calls:
        raise BuildError(
            "resume has Slow calls without durable responses; explicit process-loss "
            "recovery flag is required: " + json.dumps(interrupted, sort_keys=True)
        )
    if interrupted:
        recovered = _recover_interrupted_slow_calls(worker, repo=repo)
        if len(recovered) != len(interrupted):
            raise BuildError(
                "Slow process-loss recovery count drifted after preflight"
            )
    if enqueue:
        _run(
            [*prefix, "enqueue", scope_id],
            _resume_log(worker_dir, "slow_enqueue"),
            environment,
        )
    failed = _failed_slow_jobs(database)
    if failed:
        raise BuildError(
            "resume requires explicit revalidation of failed slow jobs: "
            + json.dumps(failed, sort_keys=True)
        )
    _run([*prefix, "drain"], _resume_log(worker_dir, "slow_drain"), environment)
    _run(
        [*prefix, "audit", scope_id, "--require-promotion-coverage"],
        _resume_log(worker_dir, "slow_audit"),
        environment,
    )


def _subject_attribution_stage(
    out_dir: Path, environment: Mapping[str, str]
) -> dict[str, Any]:
    report_path = out_dir / "subject_attribution_report.json"
    if not report_path.exists():
        _run(
            [
                sys.executable,
                str(BASE / "ops" / "audit_tmcra_v4_subject_attribution.py"),
                "--run-dir",
                str(out_dir),
                "--apply",
                "--output",
                str(report_path),
            ],
            out_dir / "subject_attribution.log",
            environment,
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError("subject-attribution report is unreadable") from exc
    if (
        not isinstance(report, Mapping)
        or report.get("status") != "complete"
        or report.get("mode") != "apply"
        or report.get("prompt_version") != SUBJECT_ATTRIBUTION_PROMPT_VERSION
        or report.get("model") != "deepseek-v4-pro"
    ):
        raise BuildError("subject-attribution stage is incomplete or drifted")
    _stage(
        out_dir,
        "subject_attribution_complete",
        routed_messages=int(report.get("routed_message_count", 0) or 0),
        quarantined=int(report.get("quarantined_count", 0) or 0),
        physical_api_calls=int(report.get("physical_api_calls", 0) or 0),
    )
    return dict(report)


def _slow_process_loss_cost_uncertainty(
    databases: Sequence[Path],
) -> dict[str, int]:
    recoveries = 0
    potential_min = 0
    potential_max = 0
    for database in databases:
        with closing(sqlite3.connect(database)) as con:
            if con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='slow_graph_process_loss_recoveries'"
            ).fetchone() is None:
                continue
            row = con.execute(
                "SELECT count(*),"
                "coalesce(sum(potential_duplicate_physical_calls_min),0),"
                "coalesce(sum(potential_duplicate_physical_calls_max),0) "
                "FROM slow_graph_process_loss_recoveries"
            ).fetchone()
            recoveries += int(row[0])
            potential_min += int(row[1])
            potential_max += int(row[2])
    return {
        "unknown_external_call_outcomes": recoveries,
        "potential_duplicate_physical_calls_min": potential_min,
        "potential_duplicate_physical_calls_max": potential_max,
    }


def _finalize_build(
    *,
    out_dir: Path,
    workers: Sequence[Mapping[str, Any]],
    writer_concurrency: int,
    slow_concurrency: int,
    recovered: bool,
) -> dict[str, Any]:
    databases = [
        Path(str(worker["worker_dir"])) / "native_memory.sqlite3"
        for worker in workers
    ]
    interrupted_call_logs = [
        path
        for worker in workers
        if (
            path := Path(str(worker["worker_dir"]))
            / "product_writer_interrupted_calls.jsonl"
        ).is_file()
    ]
    cost = build_report(collect_calls(interrupted_call_logs, databases))
    process_loss = _slow_process_loss_cost_uncertainty(databases)
    cost["slow_process_loss_uncertainty"] = process_loss
    cost["cost_is_fully_observed"] = (
        process_loss["unknown_external_call_outcomes"] == 0
    )
    (out_dir / "build_cost_report.json").write_text(
        json.dumps(cost, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    quality = _writer_quality_report(workers)
    (out_dir / "writer_quality_report.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": "tmcra.v4.build.1",
        "status": "complete",
        "row_count": len(workers),
        "writer_concurrency": writer_concurrency,
        "slow_concurrency": slow_concurrency,
        "physical_api_call_count": cost["physical_call_count"],
        "exact_cost_cny": cost["exact_cost_cny"],
        "min_cost_cny": cost["min_cost_cny"],
        "max_cost_cny": cost["max_cost_cny"],
        "cost_is_fully_observed": cost["cost_is_fully_observed"],
        "slow_process_loss_unknown_external_outcomes": process_loss[
            "unknown_external_call_outcomes"
        ],
        "slow_process_loss_potential_duplicate_physical_calls_min": (
            process_loss["potential_duplicate_physical_calls_min"]
        ),
        "slow_process_loss_potential_duplicate_physical_calls_max": (
            process_loss["potential_duplicate_physical_calls_max"]
        ),
        "interrupted_calls_without_usage": sum(
            int(item.get("calls_without_usage") or 0)
            for item in cost["by_stage_model"]
            if str(item.get("stage") or "").endswith("_interrupted")
        ),
        "resumed": recovered,
        "writer_quality_requires_review": quality["requires_review"],
        "writer_quality_warning_count": quality["warning_count"],
        "writer_quality_dropped_count": quality["dropped_count"],
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failed = out_dir / "FAILED"
    if failed.is_file():
        history = {
            "resolved_at": _now(),
            "resolved_by_resume": recovered,
            "failure": json.loads(failed.read_text(encoding="utf-8")),
        }
        with (out_dir / "build_failure_history.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(history, sort_keys=True) + "\n")
        failed.unlink()
    (out_dir / "BUILD_COMPLETE").write_text(_now() + "\n", encoding="utf-8")
    _stage(out_dir, "complete", resumed=recovered)
    return report


def _writer_quality_report(
    workers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    warning_counts: dict[str, int] = {}
    dropped_counts: dict[str, int] = {}
    message_count = 0
    messages_with_warnings = 0
    semantic_proposals = 0
    semantic_committed = 0
    for worker in workers:
        path = Path(str(worker["worker_dir"])) / "product_write_messages.jsonl"
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            message_count += 1
            semantic_proposals += int(row.get("semantic_proposals") or 0)
            semantic_committed += int(row.get("semantic_committed") or 0)
            warnings = list(row.get("validation_warnings") or [])
            if warnings:
                messages_with_warnings += 1
            for warning in warnings:
                code = str(warning.get("code") or "unknown_warning")
                warning_counts[code] = warning_counts.get(code, 0) + 1
                dropped = int(warning.get("dropped_count") or 0)
                if dropped:
                    dropped_counts[code] = dropped_counts.get(code, 0) + dropped
    review_codes = {
        "assistant_assertions_dropped",
        "invalid_assertion_quarantined",
        "invalid_interaction_quarantined",
        "invalid_item_collection_defaulted_empty",
        "durability_defaulted_uncertain",
        "temporal_durability_defaulted_uncertain",
        "conflicting_durability_defaulted_uncertain",
    }
    review_events = sum(warning_counts.get(code, 0) for code in review_codes)
    return {
        "schema_version": "tmcra.v4.writer-quality.1",
        "message_count": message_count,
        "messages_with_warnings": messages_with_warnings,
        "warning_count": sum(warning_counts.values()),
        "warnings_by_code": dict(sorted(warning_counts.items())),
        "dropped_count": sum(dropped_counts.values()),
        "dropped_by_code": dict(sorted(dropped_counts.items())),
        "review_event_count": review_events,
        "review_codes": sorted(review_codes),
        "requires_review": review_events > 0,
        "semantic_proposals": semantic_proposals,
        "semantic_committed": semantic_committed,
    }


def _record_failure(out_dir: Path, exc: Exception) -> None:
    failed = out_dir / "FAILED"
    if failed.is_file():
        with (out_dir / "build_failure_history.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(
                json.dumps(
                    {
                        "superseded_at": _now(),
                        "resolved": False,
                        "failure": json.loads(failed.read_text(encoding="utf-8")),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    failed.write_text(
        json.dumps({"at": _now(), "error": f"{exc.__class__.__name__}: {exc}"})
        + "\n",
        encoding="utf-8",
    )


def _writer_worker(worker: Mapping[str, Any], *, repo: Path, environment: Mapping[str, str]) -> None:
    worker_dir = Path(str(worker["worker_dir"]))
    _run(
        [
            sys.executable,
            str(BASE / "tmcra_v4_batch_writer.py"),
            "--input",
            str(worker["input"]),
            "--out-dir",
            str(worker_dir),
            "--repo",
            str(repo),
        ],
        worker_dir / "writer.log",
        environment,
    )
    _run(
        [
            sys.executable,
            str(BASE / "audit_tmcra_v4_chain.py"),
            "--run-dir",
            str(worker_dir),
            "--output",
            str(worker_dir / "writer_chain_audit.json"),
            "--worker-db",
            f"worker={worker_dir / 'native_memory.sqlite3'}",
        ],
        worker_dir / "writer_audit.log",
        environment,
    )


def _writer_worker_resume(
    worker: Mapping[str, Any],
    *,
    repo: Path,
    environment: Mapping[str, str],
    recover_interrupted_api_calls: bool,
) -> None:
    worker_dir = Path(str(worker["worker_dir"])).resolve()
    command = [
        sys.executable,
        str(BASE / "tmcra_v4_batch_writer.py"),
        "--input",
        str(worker["input"]),
        "--out-dir",
        str(worker_dir),
        "--repo",
        str(repo),
        "--revalidate-failed-raw-response",
    ]
    if recover_interrupted_api_calls:
        command.append("--recover-interrupted-api-calls")
    _run(
        command,
        _resume_log(worker_dir, "writer"),
        environment,
    )
    _run(
        [
            sys.executable,
            str(BASE / "audit_tmcra_v4_chain.py"),
            "--run-dir",
            str(worker_dir),
            "--output",
            str(worker_dir / "writer_chain_audit.json"),
            "--worker-db",
            f"worker={worker_dir / 'native_memory.sqlite3'}",
        ],
        _resume_log(worker_dir, "writer_audit"),
        environment,
    )


def _slow_worker(worker: Mapping[str, Any], *, repo: Path, environment: Mapping[str, str]) -> None:
    worker_dir = Path(str(worker["worker_dir"]))
    database = worker_dir / "native_memory.sqlite3"
    scope_id = str(worker["scope_id"])
    prefix = [sys.executable, str(BASE / "tmcra_v4_slow_graph.py"), str(database), "--repo", str(repo)]
    _run([*prefix, "enqueue", scope_id], worker_dir / "slow_enqueue.log", environment)
    _run([*prefix, "drain"], worker_dir / "slow_drain.log", environment)
    _run(
        [*prefix, "audit", scope_id, "--require-promotion-coverage"],
        worker_dir / "slow_audit.log",
        environment,
    )


def _parallel(
    workers: Sequence[Mapping[str, Any]],
    concurrency: int,
    task: Any,
    environments: Sequence[Mapping[str, str]],
) -> None:
    if concurrency <= 0:
        raise BuildError("concurrency must be positive")
    with ThreadPoolExecutor(max_workers=min(concurrency, len(workers))) as executor:
        futures = {
            executor.submit(task, worker, environment=environments[index]): str(worker["question_id"])
            for index, worker in enumerate(workers)
        }
        for future in as_completed(futures):
            future.result()


def resume_build(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    if not out_dir.is_dir():
        raise BuildError(f"resume output directory does not exist: {out_dir}")
    if (out_dir / "BUILD_COMPLETE").exists():
        raise BuildError("resume output is already complete")
    manifest = _load_resume_manifest(out_dir)
    expected_qids = [
        line.strip()
        for line in args.qid_list.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if expected_qids != list(manifest.get("qids") or []):
        raise BuildError("resume qid list does not match the frozen input manifest")
    shell_environment = _load_shell_environment(args.writer_env.resolve())
    base_environment = {**os.environ, **shell_environment}
    keys = _key_pool(base_environment)
    workers = list(manifest["workers"])
    environments = [
        _worker_environment(base_environment, keys, index)
        for index in range(len(workers))
    ]
    try:
        incomplete_workers: list[Mapping[str, Any]] = []
        for worker in workers:
            try:
                _verify_resume_writer(worker)
            except BuildError:
                incomplete_workers.append(worker)
        if incomplete_workers:
            if not getattr(args, "revalidate_failed_writer_raw_response", False):
                raise BuildError(
                    "resume has incomplete writer workers; explicit raw-response "
                    "revalidation flag is required: "
                    + json.dumps(
                        [str(worker["question_id"]) for worker in incomplete_workers]
                    )
                )
            interrupted = [
                call
                for worker in incomplete_workers
                for call in _interrupted_writer_calls(worker)
            ]
            if interrupted and not getattr(
                args, "recover_interrupted_api_calls", False
            ):
                raise BuildError(
                    "resume has started calls without durable responses; explicit "
                    "process-loss recovery flag is required: "
                    + json.dumps(interrupted, sort_keys=True)
                )
            _stage(
                out_dir,
                "writer_resume_started",
                workers=len(incomplete_workers),
                interrupted_calls=len(interrupted),
            )
            selected_environments = [
                environments[workers.index(worker)] for worker in incomplete_workers
            ]
            _parallel(
                incomplete_workers,
                args.writer_concurrency,
                lambda worker, environment: _writer_worker_resume(
                    worker,
                    repo=args.repo.resolve(),
                    environment=environment,
                    recover_interrupted_api_calls=bool(interrupted),
                ),
                selected_environments,
            )
            for worker in workers:
                _verify_resume_writer(worker)
            _stage(out_dir, "writer_resume_complete")
        _stage(out_dir, "subject_attribution_started", resumed=True)
        _subject_attribution_stage(out_dir, base_environment)
        interrupted_slow = [
            call for worker in workers for call in _interrupted_slow_calls(worker)
        ]
        if interrupted_slow and not getattr(
            args, "recover_interrupted_slow_calls", False
        ):
            raise BuildError(
                "resume has Slow calls without durable responses; explicit "
                "--recover-interrupted-slow-calls is required: "
                + json.dumps(interrupted_slow, sort_keys=True)
            )
        _stage(
            out_dir,
            "resume_started",
            row_count=len(workers),
            interrupted_slow_calls=len(interrupted_slow),
        )
        _parallel(
            workers,
            args.slow_concurrency,
            lambda worker, environment: _slow_worker_resume(
                worker,
                repo=args.repo.resolve(),
                environment=environment,
                recover_interrupted_slow_calls=bool(interrupted_slow),
            ),
            environments,
        )
        _stage(out_dir, "slow_graph_complete", resumed=True)
        runtime_environment = dict(base_environment)
        runtime_environment["TMCRA_NODE_MODEL_DEVICE"] = args.device
        _run(
            [
                sys.executable,
                str(BASE / "tmcra_v4_online_runtime.py"),
                "build-index",
                "--scope-manifest",
                str(out_dir / "scope_manifest.jsonl"),
                "--out-report",
                str(out_dir / "index_report.json"),
                "--embedding-model",
                str(args.embedding_model.resolve()),
                "--device",
                args.device,
                "--batch-size",
                str(args.index_batch_size),
            ],
            _resume_log(out_dir, "index"),
            runtime_environment,
        )
        _stage(out_dir, "index_complete", resumed=True)
        _run(
            [
                sys.executable,
                str(BASE / "audit_tmcra_v4_chain.py"),
                "--run-dir",
                str(out_dir),
                "--output",
                str(out_dir / "build_chain_audit.json"),
                "--build-only",
            ],
            _resume_log(out_dir, "build_chain_audit"),
            runtime_environment,
        )
        _stage(out_dir, "build_audit_complete", resumed=True)
        return _finalize_build(
            out_dir=out_dir,
            workers=workers,
            writer_concurrency=args.writer_concurrency,
            slow_concurrency=args.slow_concurrency,
            recovered=True,
        )
    except Exception as exc:
        _record_failure(out_dir, exc)
        raise


def build(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "resume", False):
        return resume_build(args)
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise BuildError(f"output directory already exists: {out_dir}")
    manifest = prepare(
        data_path=args.data.resolve(),
        qid_path=args.qid_list.resolve(),
        out_dir=out_dir,
    )
    _stage(out_dir, "prepared", rows=manifest["row_count"])
    shell_environment = _load_shell_environment(args.writer_env.resolve())
    base_environment = {**os.environ, **shell_environment}
    keys = _key_pool(base_environment)
    workers = list(manifest["workers"])
    environments = [
        _worker_environment(base_environment, keys, index) for index in range(len(workers))
    ]
    try:
        _stage(out_dir, "writer_started", concurrency=args.writer_concurrency)
        _parallel(
            workers,
            args.writer_concurrency,
            lambda worker, environment: _writer_worker(
                worker, repo=args.repo.resolve(), environment=environment
            ),
            environments,
        )
        _stage(out_dir, "writer_complete")
        _stage(out_dir, "subject_attribution_started")
        _subject_attribution_stage(out_dir, base_environment)
        _stage(out_dir, "slow_graph_started", concurrency=args.slow_concurrency)
        _parallel(
            workers,
            args.slow_concurrency,
            lambda worker, environment: _slow_worker(
                worker, repo=args.repo.resolve(), environment=environment
            ),
            environments,
        )
        _stage(out_dir, "slow_graph_complete")
        runtime_environment = dict(base_environment)
        runtime_environment["TMCRA_NODE_MODEL_DEVICE"] = args.device
        _run(
            [
                sys.executable,
                str(BASE / "tmcra_v4_online_runtime.py"),
                "build-index",
                "--scope-manifest",
                str(out_dir / "scope_manifest.jsonl"),
                "--out-report",
                str(out_dir / "index_report.json"),
                "--embedding-model",
                str(args.embedding_model.resolve()),
                "--device",
                args.device,
                "--batch-size",
                str(args.index_batch_size),
            ],
            out_dir / "index.log",
            runtime_environment,
        )
        _stage(out_dir, "index_complete")
        _run(
            [
                sys.executable,
                str(BASE / "audit_tmcra_v4_chain.py"),
                "--run-dir",
                str(out_dir),
                "--output",
                str(out_dir / "build_chain_audit.json"),
                "--build-only",
            ],
            out_dir / "build_chain_audit.log",
            runtime_environment,
        )
        return _finalize_build(
            out_dir=out_dir,
            workers=workers,
            writer_concurrency=args.writer_concurrency,
            slow_concurrency=args.slow_concurrency,
            recovered=False,
        )
    except Exception as exc:
        _record_failure(out_dir, exc)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a frozen TMCRA V4 memory corpus")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qid-list", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--writer-env", type=Path, default=DEFAULT_WRITER_ENV)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--writer-concurrency", type=int, default=1)
    parser.add_argument("--slow-concurrency", type=int, default=1)
    parser.add_argument("--index-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a frozen incomplete build after explicit failed-job review",
    )
    parser.add_argument(
        "--revalidate-failed-writer-raw-response",
        action="store_true",
        help="explicitly revalidate one saved clean writer response per failed worker",
    )
    parser.add_argument(
        "--recover-interrupted-api-calls",
        action="store_true",
        help=(
            "after explicit process-loss review, replace started writer calls "
            "that have no durable response using the same model"
        ),
    )
    parser.add_argument(
        "--recover-interrupted-slow-calls",
        action="store_true",
        help=(
            "after explicit review, journal expired Slow attempts with unknown "
            "external outcomes and reopen only those jobs"
        ),
    )
    args = parser.parse_args()
    report = build(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
