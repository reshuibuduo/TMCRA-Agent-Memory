from __future__ import annotations

import json
from pathlib import Path

from tmcra_local.generation_runtime import (
    managed_local_generation,
    probe_generation_engine,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    @staticmethod
    def getcode() -> int:
        return 200

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_byok_probe_reads_local_key_file_without_exposing_it(config_path: Path) -> None:
    observed = {}

    def opener(request, timeout):
        observed["authorization"] = request.headers.get("Authorization")
        body = json.loads(request.data.decode("utf-8"))
        requested = json.loads(
            body["messages"][1]["content"].split("object exactly: ", 1)[1]
        )
        return FakeResponse(
            {
                "choices": [{"message": {"content": json.dumps(requested)}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            }
        )

    result = probe_generation_engine(config_path, opener=opener)
    assert result["status"] == "passed"
    assert result["credential_exposed"] is False
    assert observed["authorization"] == "Bearer test-only-key-never-sent"  # public-audit: allow-test-fixture
    assert "test-only-key-never-sent" not in json.dumps(result)


def test_byok_start_does_not_spawn_a_local_model(config_path: Path) -> None:
    with managed_local_generation(config_path) as state:
        assert state == {
            "source": "byok",
            "managed": False,
            "already_running": False,
        }
