from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/smoke_local_api.py"
SPEC = importlib.util.spec_from_file_location("tmcra_release_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def test_smoke_client_accepts_only_loopback_urls() -> None:
    assert SMOKE._loopback_base_url("http://127.0.0.1:2009") == (
        "http://127.0.0.1:2009"
    )
    assert SMOKE._loopback_base_url("http://[::1]:2009/") == "http://[::1]:2009"
    with pytest.raises(SMOKE.SmokeFailure, match="non-loopback"):
        SMOKE._loopback_base_url("https://example.com")
    with pytest.raises(SMOKE.SmokeFailure, match="credentials"):
        SMOKE._loopback_base_url("http://name:secret@127.0.0.1:2009")  # public-audit: allow-test-fixture


def test_smoke_marker_scan_is_structural() -> None:
    assert SMOKE._contains({"windows": [{"text": "safe marker"}]}, "marker")
    assert not SMOKE._contains({"windows": [{"text": "safe"}]}, "marker")
