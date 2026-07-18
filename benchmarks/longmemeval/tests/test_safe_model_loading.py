from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "tmcra_benchmark"
REPOSITORY_ROOT = ROOT.parents[1]
SECURITY_SOURCE_ROOTS = (SOURCE_ROOT, REPOSITORY_ROOT / "code")


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def test_every_torch_load_is_weights_only() -> None:
    violations: list[str] = []
    for source_root in SECURITY_SOURCE_ROOTS:
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            torch_aliases = {
                alias.asname or alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
                if alias.name == "torch"
            }
            torch_aliases.update({"torch", "torch_module"})
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr != "load":
                    continue
                receiver = _dotted_name(node.func.value)
                if receiver not in torch_aliases and not receiver.endswith(".torch"):
                    continue
                value = _keyword(node, "weights_only")
                if not isinstance(value, ast.Constant) or value.value is not True:
                    violations.append(f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}")
    assert not violations, "torch.load must use weights_only=True: " + ", ".join(violations)


def test_local_transformer_models_disable_remote_code() -> None:
    violations: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "from_pretrained":
                continue
            owner = _dotted_name(node.func.value)
            if not (owner.startswith("AutoTokenizer") or owner.startswith("AutoModel")):
                continue
            local_only = _keyword(node, "local_files_only")
            remote_code = _keyword(node, "trust_remote_code")
            if not isinstance(local_only, ast.Constant) or local_only.value is not True:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:local_files_only")
            if not isinstance(remote_code, ast.Constant) or remote_code.value is not False:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:trust_remote_code")
    assert not violations, "local model loading is not hardened: " + ", ".join(violations)


def test_public_code_never_enables_remote_code() -> None:
    violations: list[str] = []
    for source_root in SECURITY_SOURCE_ROOTS:
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or node.func is None:
                    continue
                remote_code = _keyword(node, "trust_remote_code")
                if isinstance(remote_code, ast.Constant) and remote_code.value is True:
                    violations.append(f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}")
    assert not violations, "public code enables trust_remote_code: " + ", ".join(violations)


def _load_model_assets_module():
    path = SOURCE_ROOT / "legacy" / "tmcra_model_assets.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_hf_model_does_not_require_tmcra_manifest(tmp_path: Path) -> None:
    module = _load_model_assets_module()
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")

    assert module.load_optional_hf_model_manifest(tmp_path) == {}

    for relative in ("tmcra_v3_reranker.py", "tmcra_v3_online_runtime.py"):
        source = (SOURCE_ROOT / "legacy" / relative).read_text(encoding="utf-8")
        assert "load_optional_hf_model_manifest" in source
        assert "pinned cross model manifest is required" not in source
        assert "pinned model manifest is required" not in source


def test_optional_tmcra_manifest_is_validated(tmp_path: Path) -> None:
    module = _load_model_assets_module()
    manifest = tmp_path / "TMCRA_MODEL_MANIFEST.json"
    manifest.write_text(json.dumps({"repo_id": "BAAI/bge-reranker-v2-m3", "revision": "abc"}), encoding="utf-8")
    assert module.load_optional_hf_model_manifest(tmp_path)["revision"] == "abc"

    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        module.load_optional_hf_model_manifest(tmp_path)
