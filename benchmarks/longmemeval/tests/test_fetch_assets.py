from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "fetch_assets.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_git_lfs_checkpoint_is_verified_in_place(tmp_path: Path) -> None:
    module = load_script()
    payload = b"checkpoint-bytes"
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(payload)
    module.fetch_file(
        tmp_path,
        {
            "id": "weights",
            "kind": "checkpoint",
            "storage": "git_lfs",
            "destination": "weights.pt",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    )
    assert checkpoint.read_bytes() == payload


def test_git_lfs_pointer_fails_with_pull_guidance(tmp_path: Path) -> None:
    module = load_script()
    pointer = tmp_path / "weights.pt"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "size 10\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="git lfs pull"):
        module.fetch_file(
            tmp_path,
            {
                "id": "weights",
                "kind": "checkpoint",
                "storage": "git_lfs",
                "destination": "weights.pt",
                "bytes": 10,
                "sha256": "0" * 64,
            },
        )
