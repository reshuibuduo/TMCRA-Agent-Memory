from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .core_loader import graph_core_root
from .runtime_env import build_service_environment, load_local_runtime_config


class LocalInferenceProbeError(RuntimeError):
    pass


def _finite_numbers(value: Any) -> list[float]:
    numbers: list[float] = []
    if isinstance(value, bool):
        return numbers
    if isinstance(value, (int, float)):
        numbers.append(float(value))
    elif isinstance(value, Mapping):
        for nested in value.values():
            numbers.extend(_finite_numbers(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            numbers.extend(_finite_numbers(nested))
    return numbers


def probe_local_models(config_path: Path) -> dict[str, Any]:
    """Load the selected embedding and graph scorers and run real tensors."""

    try:
        import torch
        import transformers
    except (ImportError, RuntimeError) as exc:
        raise LocalInferenceProbeError(
            "local inference dependencies are unavailable or incompatible: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    payload = load_local_runtime_config(config_path)
    environment = build_service_environment(
        payload, config_path=config_path, require_model=True
    )
    os.environ.update(environment)
    graph_core_root()
    try:
        from experiments.replacement.adapters.memory_adapters import (
            _embedder_dense_vectors_for_texts,
        )
        from experiments.replacement.node_memory import LoadedNodeMemoryScorer
    except Exception as exc:
        raise LocalInferenceProbeError(
            "TMCRA graph runtime dependency closure is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    query = "TMCRA 本地安装器增加了哪些模型选择？"
    candidates = [
        "TMCRA 本地安装器已经增加 Embedding 与 Reranker 模型选择。",
        "用户今天午饭吃了米饭。",
    ]
    try:
        query_vectors, query_metadata = _embedder_dense_vectors_for_texts(
            [query], mode="local_transformers", text_kind="query"
        )
        document_vectors, document_metadata = _embedder_dense_vectors_for_texts(
            candidates, mode="local_transformers", text_kind="document"
        )
        if not query_vectors or not query_vectors[0] or len(document_vectors) != 2:
            raise RuntimeError(
                "embedding backend returned no vector: "
                + str(
                    query_metadata.get("write_embedder_dense_error")
                    or document_metadata.get("write_embedder_dense_error")
                    or "unknown"
                )
            )
        query_tensor = torch.tensor(query_vectors[0], dtype=torch.float32)
        document_tensor = torch.tensor(document_vectors, dtype=torch.float32)
        similarities = torch.mv(document_tensor, query_tensor)
        if similarities.numel() != 2 or not bool(torch.isfinite(similarities).all()):
            raise RuntimeError("embedding probe returned invalid similarity tensors")
        if float(similarities[0]) <= float(similarities[1]):
            raise RuntimeError("embedding relevance-order sanity check failed")

        scorer = LoadedNodeMemoryScorer(
            node_model_path=Path(environment["TMCRA_GRAPH_NODE_MODEL_PATH"]),
            path_model_path=Path(environment["TMCRA_GRAPH_PATH_MODEL_PATH"]),
            device=environment["TMCRA_SERVICE_GRAPH_DEVICE"],
        )
        event_text = candidates[0]
        graph = {
            "conversation_id": "tmcra:local-model-probe",
            "nodes": [
                {"id": "speaker:user", "type": "speaker", "text": "user"},
                {
                    "id": "event:models",
                    "type": "event",
                    "text": event_text,
                    "turn_index": 1,
                    "confidence": 1.0,
                    "salience": 1.0,
                },
                {
                    "id": "source:models",
                    "type": "source_turn",
                    "text": event_text,
                    "turn_index": 1,
                },
            ],
            "edges": [
                {"source": "speaker:user", "target": "event:models", "type": "speaker_event"},
                {"source": "event:models", "target": "source:models", "type": "event_source_turn"},
            ],
            "paths": [
                {
                    "id": "path:models",
                    "event_id": "event:models",
                    "type": "speaker_event_source_turn",
                    "node_ids": ["speaker:user", "event:models", "source:models"],
                }
            ],
        }
        graph_scores = scorer.score_runtime(
            graph=graph,
            question=query,
            candidate_event_ids=["event:models"],
            rerank_top_k=1,
            top_k=1,
        )
        finite_values = _finite_numbers(graph_scores)
        if not finite_values or not all(torch.isfinite(torch.tensor(finite_values))):
            raise RuntimeError("graph scorer returned no finite numeric outputs")
    except Exception as exc:
        raise LocalInferenceProbeError(
            "selected local model stack failed inference: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return {
        "status": "passed",
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "device": str(environment["TMCRA_SERVICE_DEVICE"]),
        "embedding_profile_id": str(payload.get("embedding", {}).get("profile_id") or ""),
        "embedding_dimension": int(query_tensor.numel()),
        "embedding_similarity_values": [
            round(float(value), 6) for value in similarities.detach().cpu().tolist()
        ],
        "embedding_relevant_candidate_ranked_first": True,
        "graph_node_checkpoint_loaded": True,
        "graph_path_checkpoint_loaded": True,
        "graph_probe_numeric_outputs": len(finite_values),
        "finite": True,
    }


__all__ = ["LocalInferenceProbeError", "probe_local_models"]
