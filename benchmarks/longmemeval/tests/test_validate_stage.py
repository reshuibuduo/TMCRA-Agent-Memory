from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "validate_stage.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_answer_validation_requires_complete_report(tmp_path: Path) -> None:
    module = load_script()
    write_json(
        tmp_path / "report.json",
        {
            "status": "incomplete",
            "requested": 1,
            "completed": 1,
            "failure_count": 0,
            "answer_model": "gpt-5.4",
        },
    )
    write_jsonl(tmp_path / "answers.jsonl", [{"question_id": "q1"}])
    with pytest.raises(SystemExit, match="report is not complete"):
        module.validate_answer(tmp_path, ["q1"])


def test_answer_validation_requires_hardened_production_protocol(tmp_path: Path) -> None:
    module = load_script()
    report = {
        "status": "complete",
        "requested": 1,
        "completed": 1,
        "failure_count": 0,
        "answer_model": "gpt-5.4",
        "answer_protocol": "evidence_operation_bound_v6",
        "lane": "production",
        "promotion_eligible": True,
        "compiled_evidence_packet_required": True,
    }
    write_json(tmp_path / "report.json", report)
    write_jsonl(
        tmp_path / "answers.jsonl",
        [
            {
                "question_id": "q1",
                "answer_model": "gpt-5.4",
                "answer_protocol": "evidence_operation_bound_v6",
            }
        ],
    )
    module.validate_answer(tmp_path, ["q1"])

    report["answer_protocol"] = "evidence_operation_bound_v5"
    write_json(tmp_path / "report.json", report)
    with pytest.raises(SystemExit, match="unexpected protocol"):
        module.validate_answer(tmp_path, ["q1"])


def test_judge_validation_requires_fixed_resolved_model(tmp_path: Path) -> None:
    module = load_script()
    results = tmp_path / "judge.jsonl"
    write_jsonl(
        results,
        [
            {
                "question_id": "q1",
                "autoeval_label": {
                    "model": "gpt-4o-2024-08-06",
                    "label": True,
                    "raw_response": "yes",
                },
            }
        ],
    )
    write_json(
        tmp_path / "judge.jsonl.isolated_report.json",
        {
            "status": "complete",
            "metric_model": "gpt-4o",
            "row_count": 1,
            "committed_count": 1,
            "failure_count": 0,
        },
    )
    module.validate_judge(results, ["q1"])

    rows = [json.loads(results.read_text(encoding="utf-8"))]
    rows[0]["autoeval_label"]["model"] = "gpt-4o"
    write_jsonl(results, rows)
    with pytest.raises(SystemExit, match="unexpected model"):
        module.validate_judge(results, ["q1"])
