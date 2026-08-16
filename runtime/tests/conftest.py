from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tmcra_local.auth import write_secret_file
from tmcra_local.config import build_local_config, write_local_config
from tmcra_local.core import BATCH_SCHEMA_VERSION
from tmcra_local.engine import LocalMemoryEngine


class FakeMemoryModel:
    """Deterministic model boundary used only by the local runtime tests."""

    model = "fake-memory-model"
    max_tokens = 8192

    def __init__(self, usage_sink=None) -> None:
        self.usage_sink = usage_sink

    def _metadata(self, stage: str) -> dict[str, Any]:
        metadata = {
            "physical_call_id": "test_" + uuid.uuid4().hex,
            "physical_api_call": True,
            "physical_api_calls": 1,
            "stage": stage,
            "provider": "test-local",
            "model": self.model,
            "status": "completed",
            "http_status": 200,
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "usage_reported": True,
            "latency_seconds": 0.001,
        }
        if self.usage_sink is not None:
            self.usage_sink(metadata)
        return metadata

    def complete(self, payload: Mapping[str, Any]):
        response = {
            "schema_version": BATCH_SCHEMA_VERSION,
            "batch_id": payload["batch_id"],
            "messages": [
                {
                    "message_id": message["message_id"],
                    "message_role": message["message_role"],
                    "assertions": [],
                    "interactions": [],
                    "resolutions": [],
                }
                for message in payload.get("messages", [])
                if message.get("message_role") in {"user", "assistant"}
            ],
        }
        return json.dumps(response, ensure_ascii=False), self._metadata("memory_writer")

    def reconcile(self, payload: Mapping[str, Any]):
        response = {
            "slot_decision": "keep_proposed",
            "selected_memory_id": "",
            "decision": "insert",
        }
        return json.dumps(response), self._metadata("memory_reconciliation")

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: Mapping[str, Any],
        stage: str,
        max_tokens: int,
    ):
        del system_prompt, max_tokens
        evidence = list(payload.get("evidence") or [])
        if not evidence:
            raise AssertionError("personal-knowledge test batch has no evidence")
        evidence_id = str(evidence[0]["id"])
        evidence_text = str(
            evidence[0].get("summary") or evidence[0].get("label") or evidence_id
        )
        domain = dict(payload.get("domain") or {})
        title = str(domain.get("label") or "Project knowledge")
        description = str(domain.get("summary") or evidence_text)
        response = {
            "schema_version": "tmcra.personal-knowledge.domain.1",
            "domain_id": payload["domain_id"],
            "batch_id": payload["batch_id"],
            "title": title,
            "description": description,
            "display": {
                "zh": {"title": title, "description": description},
                "en": {"title": title, "description": description},
            },
            "pages": [
                {
                    "collection": "project",
                    "page_type": "overview",
                    "title": title,
                    "abstract": description,
                    "display": {
                        "zh": {"title": title, "abstract": description},
                        "en": {"title": title, "abstract": description},
                    },
                    "claims": [
                        {
                            "text": evidence_text,
                            "status": "confirmed",
                            "evidence_ids": [evidence_id],
                            "display": {
                                "zh": {"text": evidence_text},
                                "en": {"text": evidence_text},
                            },
                        }
                    ],
                    "sections": [],
                }
            ],
            "excluded_evidence_ids": [
                str(item["id"]) for item in evidence[1:]
            ],
        }
        return json.dumps(response, ensure_ascii=False), self._metadata(stage)


def local_config(tmp_path: Path, *, knowledge_enabled: bool = False) -> Path:
    config_root = tmp_path / "config"
    key_path = config_root / "runtime" / "secrets" / "byok-api.key"
    payload = build_local_config(
        embedding_profile_id="compact-zh",
        reranker_profile_id="local-dense-only",
        llm_policy_id="byok",
        config_root=config_root,
        models_root=tmp_path / "models",
        byok_provider="test-local",
        byok_base_url="http://127.0.0.1:9/v1",
        byok_model="fake-memory-model",
        byok_api_key_file=str(key_path),
        generation_task_overrides=(
            {} if knowledge_enabled else {"personal_knowledge": "disabled"}
        ),
    )
    path = write_local_config(payload, config_root=config_root)
    write_secret_file(key_path, "test-only-key-never-sent")
    return path


def build_test_engine(config_path: Path) -> LocalMemoryEngine:
    engine = LocalMemoryEngine(config_path, verify_models=False)
    # Test the complete storage/write/recall contract without loading release
    # model weights. Production startup never applies these overrides.
    engine.graph_factory.retrieval_mode = "heuristic"
    engine.graph_factory.node_model_path = ""
    engine.graph_factory.path_model_path = ""
    engine.graph_factory.graph_environment.update(
        {
            "TMCRA_WRITE_EMBEDDER_INDEX_MODE": "off",
            "TMCRA_EMBEDDER_INDEX_RECALL_MODE": "off",
            "TMCRA_EMBEDDER_PRE_RECALL_MODE": "off",
            "TMCRA_EMBEDDER_FUSION_MODE": "off",
        }
    )
    fake = FakeMemoryModel(engine._record_usage)
    engine.llm = fake
    engine.writer.flash_client = fake
    engine.writer.pro_client = fake
    return engine


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return local_config(tmp_path)


@pytest.fixture
def engine(config_path: Path) -> LocalMemoryEngine:
    return build_test_engine(config_path)
