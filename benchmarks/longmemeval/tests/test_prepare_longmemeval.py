from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_fixture_is_valid_and_gold_is_not_exported(tmp_path: Path) -> None:
    module = load_script("prepare_longmemeval.py")
    rows = module.load_and_validate(ROOT / "fixtures" / "tiny_longmemeval.json")
    assert [row["question_id"] for row in rows] == ["tmcra_fixture_user", "tmcra_fixture_update"]
    output = tmp_path / "qids.txt"
    output.write_text("\n".join(str(row["question_id"]) for row in rows) + "\n", encoding="utf-8")
    text = output.read_text(encoding="utf-8")
    assert "Tea" not in text
    assert "Shanghai" not in text
