from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "summarize_official_judge.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_score_breakdown() -> None:
    module = load_script()
    dataset = [
        {"question_id": "a", "question_type": "single-session-user"},
        {"question_id": "b", "question_type": "knowledge-update"},
    ]
    judged = [
        {"question_id": "a", "autoeval_label": {"label": True}},
        {"question_id": "b", "autoeval_label": {"label": False}},
    ]
    result = module.summarize(dataset, judged)
    assert result["correct"] == 1
    assert result["total"] == 2
    assert result["accuracy_percent"] == 50.0
