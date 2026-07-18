#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import tmcra_v4_online_runtime as online
from run_tmcra_v4_build import (
    BASE,
    DEFAULT_REPO,
    DEFAULT_WRITER_ENV,
    _key_pool,
    _load_shell_environment,
    _now,
    _resume_log,
    _run,
)
from tmcra_v4_cost_report import build_report, collect_calls
from tmcra_v4_route_policy import (
    RoutePolicyError,
    validate_diagnostic_retrieval_rows,
    validate_production_packing_budget,
    validate_production_retrieval_mode,
    validate_production_retrieval_rows,
)


PROJECT_ROOT = BASE.parents[2]
PACKAGE_ROOT = BASE.parent
DEFAULT_HARNESS = PACKAGE_ROOT / "harness" / "run_lme_s10_native_tmcra.py"
DEFAULT_NODE = PROJECT_ROOT / "models" / "tmcra" / "node_scorer.pt"
DEFAULT_PATH_MODEL = PROJECT_ROOT / "models" / "tmcra" / "path_scorer.pt"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "tmcra" / "tmcra_v3_reranker.pt"
DEFAULT_EMBEDDING = PROJECT_ROOT / "models" / "BAAI" / "bge-m3"
DEFAULT_CROSS = PROJECT_ROOT / "models" / "BAAI" / "bge-reranker-v2-m3"


class RetrievalError(RuntimeError):
    pass


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fingerprints(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(row["question_id"]): online.scope_fingerprint(
            Path(str(row["db_path"])).resolve(), str(row["scope_id"])
        )
        for row in rows
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _prepare_graph_boundary(
    *,
    run_dir: Path,
    tag: str,
    fingerprints: dict[str, str],
    resume: bool,
) -> None:
    before_path = run_dir / f"{tag}.graph_before.json"
    failed_path = run_dir / f"FAILED.{tag}"
    staging_path = run_dir / f".{tag}.staging"
    if resume:
        if not failed_path.is_file():
            raise RetrievalError("resume requires the existing FAILED marker")
        if not before_path.is_file():
            raise RetrievalError("resume requires the original graph_before boundary")
        try:
            original = json.loads(before_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RetrievalError("original graph_before boundary is unreadable") from exc
        if original != fingerprints:
            raise RetrievalError("graph changed since the failed retrieval began")
        return
    if failed_path.exists() or before_path.exists() or staging_path.exists():
        raise RetrievalError(
            "prior retrieval state exists; inspect it and use explicit --resume"
        )
    _atomic_write_text(
        before_path,
        json.dumps(fingerprints, indent=2, sort_keys=True) + "\n",
    )


def _archive_failed_marker(run_dir: Path, tag: str) -> None:
    failed_path = run_dir / f"FAILED.{tag}"
    if not failed_path.is_file():
        return
    history_path = run_dir / f"{tag}.failure_history.jsonl"
    content = failed_path.read_text(encoding="utf-8")
    with history_path.open("a", encoding="utf-8") as handle:
        for line in content.splitlines():
            if line.strip():
                handle.write(line.rstrip() + "\n")
    failed_path.unlink()


def _record_failure(run_dir: Path, tag: str, exc: Exception) -> None:
    _archive_failed_marker(run_dir, tag)
    _atomic_write_text(
        run_dir / f"FAILED.{tag}",
        json.dumps(
            {"at": _now(), "error": f"{exc.__class__.__name__}: {exc}"}
        )
        + "\n",
    )


def _validate_manifest_pair(
    scope_rows: list[dict[str, Any]], query_rows: list[dict[str, Any]]
) -> None:
    def inventory(rows: list[dict[str, Any]], label: str) -> dict[str, tuple[str, str, str]]:
        result: dict[str, tuple[str, str, str]] = {}
        for row in rows:
            qid = str(row.get("question_id") or "").strip()
            identity = tuple(
                str(Path(str(row[key])).resolve()) if key != "scope_id" else str(row[key]).strip()
                for key in ("db_path", "scope_id", "index_path")
            )
            if not qid or qid in result or not all(identity):
                raise RetrievalError(f"{label} has a missing or duplicate retrieval identity")
            result[qid] = identity
        return result

    scope_inventory = inventory(scope_rows, "scope manifest")
    query_inventory = inventory(query_rows, "query manifest")
    if scope_inventory != query_inventory:
        raise RetrievalError("scope/query manifests do not reference the same frozen indexes")


def _select_manifest_rows(
    scope_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    qid_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qids = [
        line.strip()
        for line in qid_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not qids or len(qids) != len(set(qids)):
        raise RetrievalError("qid list is empty or duplicated")
    scope_by_qid = {str(row.get("question_id") or "").strip(): row for row in scope_rows}
    query_by_qid = {str(row.get("question_id") or "").strip(): row for row in query_rows}
    missing = [
        qid
        for qid in qids
        if qid not in scope_by_qid or qid not in query_by_qid
    ]
    if missing:
        raise RetrievalError(f"qid list contains unknown questions: {missing[:10]}")
    return (
        [scope_by_qid[qid] for qid in qids],
        [query_by_qid[qid] for qid in qids],
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def _run_production_build_preflight(
    run_dir: Path, tag: str, *, resume: bool
) -> dict[str, Any]:
    output = run_dir / f"{tag}.production_build_preflight.json"
    _run(
        [
            sys.executable,
            str(BASE / "audit_tmcra_v4_chain.py"),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
            "--build-only",
        ],
        (
            _resume_log(run_dir, f"{tag}.production_build_preflight")
            if resume
            else run_dir / f"{tag}.production_build_preflight.log"
        ),
        dict(os.environ),
    )
    if not output.is_file():
        raise RetrievalError("production build preflight did not emit an audit report")
    try:
        report = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalError("production build preflight report is unreadable") from exc
    if report.get("passed") is not True:
        raise RetrievalError("production build preflight did not pass")
    return report


def _validate_committed_retrieval_output(
    out_dir: Path,
    *,
    expected_qids: list[str],
    execution_lane: str,
) -> dict[str, Any]:
    evidence_rows = _jsonl(out_dir / "evidence_windows.jsonl")
    debug_rows = _jsonl(out_dir / "retrieval_debug.jsonl")
    try:
        route_report = (
            validate_production_retrieval_rows(evidence_rows)
            if execution_lane == "production"
            else validate_diagnostic_retrieval_rows(evidence_rows)
        )
    except RoutePolicyError as exc:
        raise RetrievalError(
            f"committed retrieval output does not match the requested lane: {exc}"
        ) from exc
    evidence_qids = [str(row.get("question_id") or "").strip() for row in evidence_rows]
    debug_qids = [
        str(row.get("question_id") or row.get("qid") or "").strip()
        for row in debug_rows
    ]
    if evidence_qids != expected_qids or debug_qids != expected_qids:
        raise RetrievalError(
            "committed retrieval output does not match the selected query manifest"
        )
    try:
        report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalError("committed retrieval report is unreadable") from exc
    if report.get("query_count") != len(expected_qids):
        raise RetrievalError("committed retrieval report query count is inconsistent")
    return route_report


def retrieve(args: argparse.Namespace) -> dict[str, Any]:
    # The logical run path is part of durable retrieval identity. Following a
    # storage symlink here would invalidate resumable checkpoints after a move.
    run_dir = args.run_dir.absolute()
    execution_lane = "diagnostic" if bool(getattr(args, "diagnostic", False)) else "production"
    try:
        validate_production_retrieval_mode(
            getattr(args, "composition_mode", None), execution_lane=execution_lane
        )
        validate_production_packing_budget(
            getattr(args, "packing_budget_mode", "fixed"),
            getattr(args, "top_k", 8),
            execution_lane=execution_lane,
        )
    except RoutePolicyError as exc:
        raise RetrievalError(f"retrieval route policy rejected the run: {exc}") from exc
    if not (run_dir / "BUILD_COMPLETE").is_file():
        raise RetrievalError("frozen V4 build is incomplete")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        raise RetrievalError("tag contains unsupported characters")
    out_dir = run_dir / args.tag
    resume = bool(getattr(args, "resume", False))
    reuse_committed_output = False
    if out_dir.exists():
        required = (
            out_dir / "evidence_windows.jsonl",
            out_dir / "retrieval_debug.jsonl",
            out_dir / "report.json",
        )
        if not resume:
            raise RetrievalError(f"retrieval output already exists: {out_dir}")
        if not out_dir.is_dir() or not all(path.is_file() for path in required):
            raise RetrievalError(
                f"resume found an incomplete committed retrieval output: {out_dir}"
            )
        reuse_committed_output = True
    query_manifest = Path(
        getattr(args, "query_manifest", None) or run_dir / "query_manifest.jsonl"
    ).resolve()
    scope_manifest = Path(
        getattr(args, "scope_manifest", None) or run_dir / "scope_manifest.jsonl"
    ).resolve()
    if not query_manifest.is_file() or not scope_manifest.is_file():
        raise RetrievalError("scope/query manifest is missing")
    rows = _jsonl(scope_manifest)
    query_rows = _jsonl(query_manifest)
    if not rows or not query_rows:
        raise RetrievalError("scope/query manifest is empty")
    _validate_manifest_pair(rows, query_rows)
    qid_list = getattr(args, "qid_list", None)
    if qid_list is not None:
        if not Path(qid_list).resolve().is_file():
            raise RetrievalError("qid list is missing")
        rows, query_rows = _select_manifest_rows(
            rows, query_rows, Path(qid_list).resolve()
        )
        scope_manifest = run_dir / f"{args.tag}.scope_manifest.selected.jsonl"
        query_manifest = run_dir / f"{args.tag}.query_manifest.selected.jsonl"
        _write_jsonl(scope_manifest, rows)
        _write_jsonl(query_manifest, query_rows)
        _validate_manifest_pair(rows, query_rows)

    if reuse_committed_output:
        _validate_committed_retrieval_output(
            out_dir,
            expected_qids=[str(row["question_id"]).strip() for row in query_rows],
            execution_lane=execution_lane,
        )

    if execution_lane == "production":
        _run_production_build_preflight(run_dir, args.tag, resume=resume)

    shell_environment = _load_shell_environment(args.writer_env.resolve())
    environment = {**os.environ, **shell_environment}
    keys = _key_pool(environment)
    base_url = environment.get("TMCRA_DEEPSEEK_WRITER_BASE_URL") or environment.get("TMCRA_WRITER_BASE_URL") or "https://api.deepseek.com/v1"
    environment.update(
        {
            "TMCRA_RECALL_PLANNER_BASE_URL": base_url,
            "TMCRA_RECALL_PLANNER_MODEL": "deepseek-v4-flash",
            "TMCRA_RECALL_PLANNER_API_KEY_POOL": ",".join(keys),
            "TMCRA_NODE_MODEL_PATH": str(args.node_model.resolve()),
            "TMCRA_PATH_MODEL_PATH": str(args.path_model.resolve()),
            "TMCRA_NODE_MODEL_DEVICE": args.graph_device,
        }
    )
    before = _fingerprints(rows)
    _prepare_graph_boundary(
        run_dir=run_dir,
        tag=args.tag,
        fingerprints=before,
        resume=resume,
    )
    try:
        if not reuse_committed_output:
            runtime_options = ["--resume"] if resume else []
            if args.planner_replay_dir is not None:
                runtime_options.extend(
                    ["--planner-replay-dir", str(args.planner_replay_dir.resolve())]
                )
            _run(
                [
                    sys.executable,
                    str(BASE / "tmcra_v4_online_runtime.py"),
                    "retrieve",
                    *runtime_options,
                    "--query-manifest",
                    str(query_manifest),
                    "--out-dir",
                    str(out_dir),
                    "--checkpoint",
                    str(args.checkpoint.resolve()),
                    "--cross-model",
                    str(args.cross_model.resolve()),
                    "--repo",
                    str(args.repo.resolve()),
                    "--harness",
                    str(args.harness.resolve()),
                    "--node-model",
                    str(args.node_model.resolve()),
                    "--path-model",
                    str(args.path_model.resolve()),
                    "--graph-device",
                    args.graph_device,
                    "--embedding-model",
                    str(args.embedding_model.resolve()),
                    "--slow-dense-k",
                    str(args.slow_dense_k),
                    "--dense-k",
                    str(args.dense_k),
                    "--graph-k",
                    str(args.graph_k),
                    "--composition-mode",
                    str(args.composition_mode),
                    "--execution-lane",
                    execution_lane,
                    "--packing-budget-mode",
                    str(args.packing_budget_mode),
                    "--top-k",
                    str(args.top_k),
                    "--adaptive-simple-k",
                    str(args.adaptive_simple_k),
                    "--adaptive-standard-k",
                    str(args.adaptive_standard_k),
                    "--adaptive-complex-k",
                    str(args.adaptive_complex_k),
                    "--cross-batch-size",
                    str(args.cross_batch_size),
                    "--device",
                    args.device,
                ],
                (
                    _resume_log(run_dir, args.tag)
                    if resume
                    else run_dir / f"{args.tag}.log"
                ),
                environment,
            )
        after = _fingerprints(rows)
        _atomic_write_text(
            run_dir / f"{args.tag}.graph_after.json",
            json.dumps(after, indent=2, sort_keys=True) + "\n",
        )
        if after != before:
            raise RetrievalError("retrieval changed graph structure or graph records")
        _run(
            [
                sys.executable,
                str(BASE / "audit_tmcra_v4_chain.py"),
                "--run-dir",
                str(run_dir),
                "--retrieval-dir",
                str(out_dir),
                "--output",
                str(run_dir / f"{args.tag}.chain_audit.json"),
            ],
            (
                _resume_log(run_dir, f"{args.tag}.chain_audit")
                if resume
                else run_dir / f"{args.tag}.chain_audit.log"
            ),
            environment,
        )
        # Retrieval cost is incremental. Frozen graph databases contain the
        # historical writer ledger and must not be charged to every query run.
        calls = collect_calls([out_dir / "retrieval_debug.jsonl"], [])
        cost = build_report(calls)
        (run_dir / f"{args.tag}.cost_report.json").write_text(
            json.dumps(cost, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        summary = {
            "schema_version": "tmcra.v4.retrieval-run.1",
            "status": "complete",
            "tag": args.tag,
            "execution_lane": execution_lane,
            "composition_mode": str(args.composition_mode),
            "query_count": report["query_count"],
            "selected_output": str(out_dir / "evidence_windows.jsonl"),
            "scope_manifest": str(scope_manifest),
            "scope_manifest_sha256": _file_sha256(scope_manifest),
            "query_manifest": str(query_manifest),
            "query_manifest_sha256": _file_sha256(query_manifest),
            "qid_list": str(Path(qid_list).resolve()) if qid_list is not None else None,
            "qid_list_sha256": _file_sha256(Path(qid_list).resolve()) if qid_list is not None else None,
            "graph_structure_unchanged": True,
            "resumed": resume,
            "reused_committed_retrieval_output": reuse_committed_output,
            "physical_api_call_count_total": cost["physical_call_count"],
            "exact_cost_cny_total": cost["exact_cost_cny"],
            "min_cost_cny_total": cost["min_cost_cny"],
            "max_cost_cny_total": cost["max_cost_cny"],
        }
        (out_dir / "retrieval_run_report.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (out_dir / "RETRIEVAL_COMPLETE").write_text(_now() + "\n", encoding="utf-8")
        _archive_failed_marker(run_dir, args.tag)
        return summary
    except Exception as exc:
        _record_failure(run_dir, args.tag, exc)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one TMCRA V4 retrieval pass over a frozen build")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tag", default="retrieval_1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scope-manifest", type=Path)
    parser.add_argument("--query-manifest", type=Path)
    parser.add_argument("--qid-list", type=Path)
    parser.add_argument("--planner-replay-dir", type=Path)
    parser.add_argument("--writer-env", type=Path, default=DEFAULT_WRITER_ENV)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--node-model", type=Path, default=DEFAULT_NODE)
    parser.add_argument("--path-model", type=Path, default=DEFAULT_PATH_MODEL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--cross-model", type=Path, default=DEFAULT_CROSS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graph-device", default="cuda")
    parser.add_argument("--dense-k", type=int, default=32)
    parser.add_argument("--slow-dense-k", type=int, default=24)
    parser.add_argument("--graph-k", type=int, default=24)
    parser.add_argument("--composition-mode", choices=("layered", "source-only-diagnostic"), default="layered")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="explicitly allow a non-production retrieval composition",
    )
    parser.add_argument("--packing-budget-mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--adaptive-simple-k", type=int, default=8)
    parser.add_argument("--adaptive-standard-k", type=int, default=12)
    parser.add_argument("--adaptive-complex-k", type=int, default=16)
    parser.add_argument("--cross-batch-size", type=int, default=24)
    args = parser.parse_args()
    summary = retrieve(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
