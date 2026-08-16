from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tmcra_local.auth import ensure_local_token
from tmcra_local.config import LocalConfigError, build_local_config
from tmcra_local.engine import LocalMemoryEngine
from tmcra_local.model_catalog import catalog_payload
from tmcra_local.runtime_env import (
    LocalRuntimeEnvironmentError,
    _validate_graph_assets,
    build_service_environment,
    load_local_runtime_config,
)


def test_config_is_secret_free_and_loopback_only(config_path: Path) -> None:
    raw = config_path.read_text(encoding="utf-8")
    payload = load_local_runtime_config(config_path)
    assert "test-only-key-never-sent" not in raw
    assert payload["network"] == {
        "allow_non_loopback": False,
        "bind_host": "127.0.0.1",
        "bind_port": 2009,
        "silent_cloud_fallback": False,
    }
    environment = build_service_environment(
        payload, config_path=config_path, require_model=False
    )
    assert environment["TMCRA_SERVICE_BIND_HOST"] == "127.0.0.1"
    assert environment["TMCRA_WRITER_API_KEY_POOL"] == "test-only-key-never-sent"
    assert environment["TMCRA_LOCAL_GENERATION_SOURCE"] == "byok"
    assert set(payload["generation"]["task_routes"]) == {
        "memory_writer",
        "personal_knowledge",
    }
    assert "TMCRA_RECALL_PLANNER_MODEL" not in environment
    assert "TMCRA_SLOW_GRAPH_MODEL" not in environment


def test_local_api_token_is_stable_and_not_in_config(config_path: Path) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    root = payload["installation"]["config_root"]
    first_path, first, created = ensure_local_token(root)
    second_path, second, created_again = ensure_local_token(root)
    assert created is True
    assert created_again is False
    assert first_path == second_path
    assert first == second
    assert len(first) >= 32
    assert first not in config_path.read_text(encoding="utf-8")


def test_local_engine_does_not_export_provider_key_to_process_environment(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_names = (
        "TMCRA_WRITER_API_KEY_POOL",
        "TMCRA_WRITER_REVIEWER_API_KEY_POOL",
    )
    for name in secret_names:
        monkeypatch.delenv(name, raising=False)
    engine = LocalMemoryEngine(config_path, verify_models=False)
    assert engine.llm.api_key == "test-only-key-never-sent"
    assert all(name not in os.environ for name in secret_names)


def test_rejects_credentials_embedded_in_base_url(tmp_path: Path) -> None:
    with pytest.raises(LocalConfigError, match="credential-free"):
        build_local_config(
            embedding_profile_id="compact-zh",
            llm_policy_id="byok",
            config_root=tmp_path / "config",
            byok_provider="unsafe",
            byok_base_url="https://name:secret@example.com/v1",  # public-audit: allow-test-fixture
            byok_model="model",
        )


def test_graph_assets_are_verified_against_manifest(tmp_path: Path) -> None:
    root = tmp_path / "graph-assets"
    root.mkdir()
    entries = []
    for name, payload in (
        ("node_scorer.pt", b"node-test-weights"),
        ("path_scorer.pt", b"path-test-weights"),
    ):
        (root / name).write_bytes(payload)
        entries.append(
            {
                "name": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    (root / "TMCRA_MODEL_MANIFEST.json").write_text(
        json.dumps({"files": entries}), encoding="utf-8"
    )
    node, path = _validate_graph_assets(root)
    assert node.name == "node_scorer.pt"
    assert path.name == "path_scorer.pt"

    (root / "node_scorer.pt").write_bytes(b"tampered")
    with pytest.raises(LocalRuntimeEnvironmentError, match="failed manifest"):
        _validate_graph_assets(root)


def test_public_model_catalog_hides_failed_preview_and_explains_statuses() -> None:
    public = catalog_payload()
    assert [item["id"] for item in public["generation_profiles"]] == [
        "recommended-qwen36"
    ]
    assert "production" not in json.dumps(public, ensure_ascii=False).casefold()

    complete = catalog_payload(include_preview=True)
    statuses = {
        str(item["effect"]["status"])
        for group in (
            complete["embedding_profiles"],
            complete["generation_profiles"],
            complete["reranker_policies"],
        )
        for item in group
    }
    statuses.update(str(item["effect_status"]) for item in complete["llm_policies"])
    assert statuses <= set(complete["effect_contract"])
