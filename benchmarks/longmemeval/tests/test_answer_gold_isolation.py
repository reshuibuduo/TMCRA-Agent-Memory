from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANSWER_RUNNER = (
    ROOT
    / "src"
    / "tmcra_benchmark"
    / "legacy"
    / "run_tmcra_v4_gpt54_answers.py"
)


def test_answer_runner_rejects_evaluation_only_fields() -> None:
    tree = ast.parse(ANSWER_RUNNER.read_text(encoding="utf-8"), filename=str(ANSWER_RUNNER))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    rejection = functions["reject_evaluation_fields"]
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "gold_answer" in literals
    assert "answer_session_ids" in literals
    assert any(
        isinstance(node, ast.Raise)
        for node in ast.walk(rejection)
    )


def test_answer_output_does_not_persist_gold_or_gold_scores() -> None:
    source = ANSWER_RUNNER.read_text(encoding="utf-8")
    assert 'row.get("gold_answer")' not in source
    assert '"deterministic_scores"' not in source
    assert 'row.get("answer_session_ids")' not in source
