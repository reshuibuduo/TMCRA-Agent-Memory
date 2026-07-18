from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError:
    torch = None
    Tensor = Any
    nn = None

from experiments.replacement.temporal_modeling_types import clean_text


TEMPORAL_ROUTER_ARCHITECTURE = "hashed_text_multihead_classifier_v1"


def _tokens(text: str) -> list[str]:
    normalized = clean_text(text).lower()
    english = re.findall(r"[a-z0-9_.-]+", normalized)
    cjk = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
    return [*english, *cjk]


def _stable_hash(value: str) -> int:
    h = 2166136261
    for byte in value.encode("utf-8"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _vectorize_text(text: str, *, buckets: int) -> Tensor:
    if torch is None:
        raise RuntimeError("torch is required for temporal router runtime")
    vector = torch.zeros((buckets,), dtype=torch.float32)
    tokens = _tokens(text)
    if not tokens:
        return vector
    features: list[str] = []
    features.extend(tokens)
    features.extend(f"{tokens[index]}__{tokens[index + 1]}" for index in range(len(tokens) - 1))
    for item in features:
        vector[_stable_hash(item) % buckets] += 1.0
    norm = float(vector.norm().item())
    if norm > 0.0:
        vector /= norm
    return vector


class MultiHeadClassifier(nn.Module if nn is not None else object):
    def __init__(self, *, input_dim: int, hidden_dim: int, heads: Mapping[str, int]) -> None:
        if nn is None:
            raise RuntimeError("torch is required for temporal router runtime")
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict({key: nn.Linear(hidden_dim, int(size)) for key, size in heads.items()})

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        hidden = self.encoder(features)
        return {key: head(hidden) for key, head in self.heads.items()}


class LoadedTemporalRouter:
    """Runtime wrapper for the trained temporal writer/query router checkpoints."""

    def __init__(
        self,
        *,
        writer_model_path: str = "",
        query_model_path: str = "",
        device: str = "cpu",
    ) -> None:
        if torch is None:
            raise RuntimeError("torch is required for temporal router runtime")
        self.device = torch.device(device or "cpu")
        self.writer = self._load_model(writer_model_path, expected_task="writer") if writer_model_path else None
        self.query = self._load_model(query_model_path, expected_task="query") if query_model_path else None

    @classmethod
    def from_dir(cls, model_dir: str | Path, *, device: str = "cpu") -> "LoadedTemporalRouter":
        root = Path(model_dir)
        return cls(
            writer_model_path=str(root / "writer_temporal_router.pt"),
            query_model_path=str(root / "query_temporal_router.pt"),
            device=device,
        )

    def writer_available(self) -> bool:
        return self.writer is not None

    def query_available(self) -> bool:
        return self.query is not None

    def predict_writer_frame(
        self,
        *,
        current_turn: str,
        previous_turn: str = "",
        session_timestamp: str = "",
    ) -> Dict[str, Any]:
        if self.writer is None:
            return {}
        text = "\n".join(
            part
            for part in [
                f"previous: {clean_text(previous_turn)}",
                f"current: {clean_text(current_turn)}",
                f"timestamp: {clean_text(session_timestamp)}",
            ]
            if clean_text(part)
        )
        prediction = self._predict(self.writer, text)
        labels = dict(prediction.get("labels", {}) or {})
        confidences = dict(prediction.get("confidences", {}) or {})
        confidence = self._mean_confidence(confidences)
        return {
            "temporal_intent": clean_text(labels.get("temporal_intent", "")),
            "anchor_type": clean_text(labels.get("anchor_type", "")),
            "granularity": clean_text(labels.get("granularity", "")),
            "state_operation": clean_text(labels.get("state_operation", "")),
            "should_create_timeline_edge": labels.get("should_create_timeline_edge") == "true",
            "confidence": confidence,
            "metadata": {
                "temporal_router_runtime": True,
                "temporal_router_task": "writer",
                "temporal_router_confidences": confidences,
            },
        }

    def predict_query_plan(self, *, query: str, session_timestamp: str = "") -> Dict[str, Any]:
        if self.query is None:
            return {}
        text = "\n".join(
            part
            for part in [
                f"query: {clean_text(query)}",
                f"timestamp: {clean_text(session_timestamp)}",
            ]
            if clean_text(part)
        )
        prediction = self._predict(self.query, text)
        labels = dict(prediction.get("labels", {}) or {})
        confidences = dict(prediction.get("confidences", {}) or {})
        confidence = self._mean_confidence(confidences)
        return {
            "query_temporal_intent": clean_text(labels.get("query_temporal_intent", "")),
            "timeline_operation": clean_text(labels.get("timeline_operation", "")),
            "prefer_current_state": labels.get("prefer_current_state") == "true",
            "prefer_previous_state": labels.get("prefer_previous_state") == "true",
            "requires_ordered_chain": labels.get("requires_ordered_chain") == "true",
            "requires_comparison": labels.get("requires_comparison") == "true",
            "confidence": confidence,
            "metadata": {
                "temporal_router_runtime": True,
                "temporal_router_task": "query",
                "temporal_router_confidences": confidences,
            },
        }

    def _load_model(self, model_path: str, *, expected_task: str) -> Dict[str, Any]:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if clean_text(payload.get("architecture", "")) != TEMPORAL_ROUTER_ARCHITECTURE:
            raise ValueError(f"Unsupported temporal router architecture: {payload.get('architecture', '')}")
        task = clean_text(payload.get("task", ""))
        if task and task != expected_task:
            raise ValueError(f"Expected temporal router task {expected_task}, got {task}")
        buckets = int(payload.get("buckets", 4096) or 4096)
        hidden_dim = int(payload.get("hidden_dim", 256) or 256)
        label_maps = {key: dict(value) for key, value in dict(payload.get("label_maps", {}) or {}).items()}
        heads = {key: len(value) for key, value in label_maps.items()}
        model = MultiHeadClassifier(input_dim=buckets, hidden_dim=hidden_dim, heads=heads).to(self.device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return {
            "path": str(path),
            "model": model,
            "buckets": buckets,
            "label_maps": label_maps,
            "metrics": {
                "val": dict(payload.get("val_metrics", {}) or {}),
                "test": dict(payload.get("test_metrics", {}) or {}),
            },
        }

    def _predict(self, loaded: Mapping[str, Any], text: str) -> Dict[str, Any]:
        model = loaded["model"]
        buckets = int(loaded["buckets"])
        label_maps = dict(loaded["label_maps"])
        features = _vectorize_text(text, buckets=buckets).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = model(features)
        labels: Dict[str, str] = {}
        confidences: Dict[str, float] = {}
        for key, logits in dict(outputs).items():
            probs = torch.softmax(logits[0].detach().float().cpu(), dim=-1)
            index = int(torch.argmax(probs).item())
            reverse = {int(value): str(label) for label, value in dict(label_maps.get(key, {}) or {}).items()}
            labels[key] = reverse.get(index, "")
            confidences[key] = round(float(probs[index].item()), 6)
        return {"labels": labels, "confidences": confidences}

    def _mean_confidence(self, confidences: Mapping[str, Any]) -> float:
        values = [float(value) for value in dict(confidences or {}).values()]
        if not values:
            return 0.0
        return round(sum(values) / len(values), 6)
