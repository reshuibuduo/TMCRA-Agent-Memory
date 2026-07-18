from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_optional_hf_model_manifest(model_dir: str | Path) -> dict[str, Any]:
    """Read optional provenance metadata for a locally cached Hugging Face model.

    ``TMCRA_MODEL_MANIFEST.json`` is a TMCRA sidecar, not part of the Hugging
    Face model layout.  Official BGE snapshots therefore remain valid without
    it.  When a sidecar is supplied, validate it before using its provenance
    fields in reports.
    """

    path = Path(model_dir) / "TMCRA_MODEL_MANIFEST.json"
    if not path.exists():
        return {}
    if not path.is_file():
        raise ValueError(f"model manifest must be a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid model manifest: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"model manifest must contain a JSON object: {path}")
    return dict(payload)
