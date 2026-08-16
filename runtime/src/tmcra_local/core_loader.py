from __future__ import annotations

import os
import sys
from pathlib import Path


class LocalCoreError(RuntimeError):
    pass


def release_root() -> Path:
    override = os.getenv("TMCRA_LOCAL_RELEASE_ROOT", "").strip()
    candidates = [Path(override).expanduser().resolve()] if override else []
    candidates.extend(Path(__file__).resolve().parents)
    marker = Path("benchmarks/longmemeval/src/tmcra_benchmark/vendor/tmcra_integrated")
    for candidate in candidates:
        if (candidate / marker / "experiments/replacement/memory_graph.py").is_file():
            return candidate
    raise LocalCoreError(
        "cannot locate the TMCRA release root; run from a complete repository clone "
        "or set TMCRA_LOCAL_RELEASE_ROOT"
    )


def graph_core_root() -> Path:
    root = (
        release_root()
        / "benchmarks"
        / "longmemeval"
        / "src"
        / "tmcra_benchmark"
        / "vendor"
        / "tmcra_integrated"
    ).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def graph_scorer_paths() -> tuple[Path, Path]:
    root = release_root() / "models" / "tmcra_v4_longmemeval_s500_20260715"
    return (root / "node_scorer.pt").resolve(), (root / "path_scorer.pt").resolve()


__all__ = ["LocalCoreError", "graph_core_root", "graph_scorer_paths", "release_root"]
