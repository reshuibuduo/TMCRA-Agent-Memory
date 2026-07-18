"""
Neural policy for Tri-Maze exploration (non-LLM).
Provides a backward-compatible EdgePolicy wrapper that can load the original
v1 shallow MLP checkpoints or the new v2 structured ranking model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import random

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None

from .policy_dataset import (
    BASE_FEATURE_DIM,
    CANDIDATE_FLAG_DIM,
    FEATURE_SCHEMA_VERSION,
    PATH_CONTEXT_DIM,
    PolicyStepDataset,
    PolicyVocabulary,
    build_runtime_step_record,
    policy_collate_fn,
)


CHECKPOINT_VERSION_V2 = "tmcra_tri_maze_policy_v2"


def _mask_fill_value(tensor: torch.Tensor) -> float:
    if torch is None:
        return -1e4
    try:
        return float(torch.finfo(tensor.dtype).min)
    except Exception:
        return -1e4


@dataclass(slots=True)
class PolicyModelConfig:
    model_version: str = "v2"
    base_feature_dim: int = BASE_FEATURE_DIM
    path_context_dim: int = PATH_CONTEXT_DIM
    candidate_flag_dim: int = CANDIDATE_FLAG_DIM
    history_size: int = 4
    concept_embedding_dim: int = 64
    relation_embedding_dim: int = 16
    domain_embedding_dim: int = 8
    hash_bucket_size: int = 8192
    trunk_dims: tuple[int, ...] = (128, 256, 128)
    dropout: float = 0.1
    feature_attention: bool = True
    multitask: bool = True
    domain_adapt: bool = False
    contrastive_dim: int = 128
    path_length_buckets: int = 4
    concept_vocab_size: int = 2
    relation_vocab_size: int = 2
    source_vocab_size: int = 2
    query_type_vocab_size: int = 2
    mode_vocab_size: int = 6
    task_vocab_size: int = 2

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trunk_dims"] = list(self.trunk_dims)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PolicyModelConfig":
        if not payload:
            return cls()
        data = dict(payload)
        if "trunk_dims" in data:
            data["trunk_dims"] = tuple(int(item) for item in data.get("trunk_dims") or ())
        return cls(**data)

    def apply_vocabulary(self, vocabulary: PolicyVocabulary | None) -> "PolicyModelConfig":
        if vocabulary is None:
            return self
        updated = self.to_dict()
        updated.update(
            {
                "hash_bucket_size": int(vocabulary.hash_bucket_size),
                "concept_vocab_size": int(vocabulary.concept_vocab_size),
                "relation_vocab_size": int(vocabulary.relation_vocab_size),
                "source_vocab_size": int(vocabulary.source_vocab_size),
                "query_type_vocab_size": int(vocabulary.query_type_vocab_size),
                "mode_vocab_size": int(vocabulary.mode_vocab_size),
                "task_vocab_size": int(vocabulary.task_vocab_size),
            }
        )
        return PolicyModelConfig.from_dict(updated)


if nn is not None:
    class PolicyNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):  # x: [N, D]
            return self.net(x).squeeze(-1)
else:  # pragma: no cover
    class PolicyNet:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch not available; install torch to use PolicyNet")


if nn is not None:
    class _GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, coeff):
            ctx.coeff = coeff
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.coeff, None


    def grad_reverse(x: torch.Tensor, coeff: float) -> torch.Tensor:
        return _GradReverse.apply(x, coeff)


    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        weights = mask.float()
        total = (values * weights.unsqueeze(-1)).sum(dim=dim)
        denom = weights.sum(dim=dim, keepdim=True).clamp_min(1.0)
        return total / denom


    class PolicyNetV2(nn.Module):
        def __init__(self, config: PolicyModelConfig):
            super().__init__()
            self.config = config
            self.concept_embedding = nn.Embedding(
                config.concept_vocab_size,
                config.concept_embedding_dim,
                padding_idx=0,
            )
            self.hash_embedding = nn.Embedding(
                config.hash_bucket_size + 1,
                config.concept_embedding_dim,
                padding_idx=0,
            )
            self.relation_embedding = nn.Embedding(
                config.relation_vocab_size,
                config.relation_embedding_dim,
                padding_idx=0,
            )
            self.source_embedding = nn.Embedding(
                config.source_vocab_size,
                config.domain_embedding_dim,
                padding_idx=0,
            )
            self.query_type_embedding = nn.Embedding(
                config.query_type_vocab_size,
                config.domain_embedding_dim,
                padding_idx=0,
            )
            self.mode_embedding = nn.Embedding(
                config.mode_vocab_size,
                config.domain_embedding_dim,
                padding_idx=0,
            )
            self.task_embedding = nn.Embedding(
                config.task_vocab_size,
                config.domain_embedding_dim,
                padding_idx=0,
            )

            self.sample_context_dim = (
                config.concept_embedding_dim * 2
                + config.path_context_dim
                + config.domain_embedding_dim * 4
            )
            self.candidate_input_dim = (
                config.base_feature_dim
                + config.path_context_dim
                + config.candidate_flag_dim
                + config.relation_embedding_dim
                + config.concept_embedding_dim * 4
                + config.domain_embedding_dim * 4
                + config.trunk_dims[0]
            )
            self.sample_context_proj = nn.Sequential(
                nn.Linear(self.sample_context_dim, config.trunk_dims[0]),
                nn.LayerNorm(config.trunk_dims[0]),
                nn.SiLU(),
            )
            if config.feature_attention:
                self.feature_gate = nn.Sequential(
                    nn.Linear(self.candidate_input_dim, self.candidate_input_dim),
                    nn.LayerNorm(self.candidate_input_dim),
                    nn.Sigmoid(),
                )
            else:
                self.feature_gate = None

            layers: list[nn.Module] = []
            prev_dim = self.candidate_input_dim
            for hidden_dim in config.trunk_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.SiLU(),
                        nn.Dropout(config.dropout),
                    ]
                )
                prev_dim = hidden_dim
            self.trunk = nn.Sequential(*layers)
            self.score_head = nn.Linear(prev_dim, 1)
            self.path_length_head = nn.Linear(prev_dim, config.path_length_buckets)
            self.tunnel_head = nn.Linear(prev_dim, 1)
            self.high_value_head = nn.Linear(prev_dim, 1)
            self.context_projection = nn.Linear(prev_dim, config.contrastive_dim)
            self.candidate_projection = nn.Linear(prev_dim, config.contrastive_dim)
            self.domain_head = nn.Linear(prev_dim, config.source_vocab_size) if config.domain_adapt else None

        def _concept_repr(self, token_ids: torch.Tensor, hash_ids: torch.Tensor) -> torch.Tensor:
            return self.concept_embedding(token_ids) + self.hash_embedding(hash_ids)

        def forward(self, batch: PolicyBatch | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            if isinstance(batch, dict):
                batch = PolicyBatch(**batch)

            current_embed = self._concept_repr(batch.current_concept_ids, batch.current_hash_ids)
            history_embed = self._concept_repr(batch.history_concept_ids, batch.history_hash_ids)
            history_mask = (batch.history_concept_ids != 0) | (batch.history_hash_ids != 0)
            history_summary = _masked_mean(history_embed, history_mask, dim=1)

            candidate_embed = self._concept_repr(batch.candidate_concept_ids, batch.candidate_hash_ids)
            relation_embed = self.relation_embedding(batch.relation_ids)
            source_embed = self.source_embedding(batch.source_ids)
            query_embed = self.query_type_embedding(batch.query_type_ids)
            mode_embed = self.mode_embedding(batch.mode_ids)
            task_embed = self.task_embedding(batch.task_ids)

            sample_context = torch.cat(
                [
                    current_embed,
                    history_summary,
                    batch.path_context,
                    source_embed,
                    query_embed,
                    mode_embed,
                    task_embed,
                ],
                dim=-1,
            )
            sample_context = self.sample_context_proj(sample_context)
            sample_context_expanded = sample_context.unsqueeze(1).expand(-1, batch.base_features.shape[1], -1)
            path_context_expanded = batch.path_context.unsqueeze(1).expand(-1, batch.base_features.shape[1], -1)
            domain_context = torch.cat([source_embed, query_embed, mode_embed, task_embed], dim=-1)
            domain_context = domain_context.unsqueeze(1).expand(-1, batch.base_features.shape[1], -1)

            candidate_input = torch.cat(
                [
                    batch.base_features,
                    path_context_expanded,
                    batch.candidate_flags,
                    relation_embed,
                    sample_context_expanded,
                    candidate_embed,
                    current_embed.unsqueeze(1).expand_as(candidate_embed),
                    candidate_embed - current_embed.unsqueeze(1),
                    candidate_embed * current_embed.unsqueeze(1),
                    domain_context,
                ],
                dim=-1,
            )
            if self.feature_gate is not None:
                candidate_input = candidate_input * self.feature_gate(candidate_input)

            hidden = self.trunk(candidate_input)
            logits = self.score_head(hidden).squeeze(-1)
            logits = logits.masked_fill(~batch.candidate_mask, _mask_fill_value(logits))
            pooled = _masked_mean(hidden, batch.candidate_mask, dim=1)

            outputs = {
                "logits": logits,
                "hidden": hidden,
                "pooled": pooled,
                "path_length_logits": self.path_length_head(pooled),
                "tunnel_logits": self.tunnel_head(pooled).squeeze(-1),
                "high_value_logits": self.high_value_head(pooled).squeeze(-1),
                "contrastive_context": F.normalize(self.context_projection(pooled), dim=-1),
                "contrastive_candidates": F.normalize(self.candidate_projection(hidden), dim=-1),
            }
            if self.domain_head is not None:
                outputs["domain_logits"] = self.domain_head(grad_reverse(pooled, 1.0))
            return outputs


def masked_cross_entropy(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if F is None:
        raise RuntimeError("torch not available")
    masked_logits = logits.masked_fill(~candidate_mask, _mask_fill_value(logits))
    loss = F.cross_entropy(masked_logits, target_index, reduction="none")
    if sample_weights is not None:
        loss = loss * sample_weights
    return loss.mean()


def hard_negative_margin_loss(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    margin: float = 0.2,
) -> torch.Tensor:
    if F is None:
        raise RuntimeError("torch not available")
    batch_index = torch.arange(logits.shape[0], device=logits.device)
    positive = logits[batch_index, target_index]
    negative_mask = candidate_mask.clone()
    negative_mask[batch_index, target_index] = False
    hardest_negative = logits.masked_fill(~negative_mask, _mask_fill_value(logits)).max(dim=-1).values
    loss = F.relu(margin - (positive - hardest_negative))
    valid = negative_mask.any(dim=-1)
    if valid.any():
        return loss[valid].mean()
    return loss.new_tensor(0.0)


def candidate_contrastive_loss(
    context_repr: torch.Tensor,
    candidate_repr: torch.Tensor,
    target_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    if F is None:
        raise RuntimeError("torch not available")
    logits = torch.einsum("bd,bcd->bc", context_repr, candidate_repr) / max(temperature, 1e-4)
    logits = logits.masked_fill(~candidate_mask, _mask_fill_value(logits))
    return F.cross_entropy(logits, target_index)


def _copy_embedding_rows_by_token(
    target_tensor: torch.Tensor,
    source_tensor: torch.Tensor | None,
    target_mapping: dict[str, int],
    source_mapping: dict[str, int],
) -> None:
    if source_tensor is None:
        return
    rows = int(target_tensor.shape[0])
    source_rows = int(source_tensor.shape[0])
    cols = min(int(target_tensor.shape[1]), int(source_tensor.shape[1])) if target_tensor.ndim == 2 and source_tensor.ndim == 2 else 0
    if cols <= 0:
        return
    with torch.no_grad():
        for token, target_index in target_mapping.items():
            source_index = source_mapping.get(token)
            if source_index is None:
                continue
            if not (0 <= int(target_index) < rows and 0 <= int(source_index) < source_rows):
                continue
            target_tensor[int(target_index), :cols].copy_(source_tensor[int(source_index), :cols])


def _copy_prefix_tensor(target_tensor: torch.Tensor, source_tensor: torch.Tensor | None) -> None:
    if source_tensor is None or target_tensor.ndim != source_tensor.ndim:
        return
    common_shape = tuple(min(int(a), int(b)) for a, b in zip(target_tensor.shape, source_tensor.shape))
    if not common_shape:
        return
    target_slices = tuple(slice(0, size) for size in common_shape)
    source_slices = tuple(slice(0, size) for size in common_shape)
    with torch.no_grad():
        target_tensor[target_slices].copy_(source_tensor[source_slices])


class EdgePolicy:
    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        temperature: float = 1.0,
        branch_factor: int = 2,
        revisit_probability: float = 0.2,
        seed: int = 42,
        *,
        model_version: str = "v2",
        model_config: PolicyModelConfig | None = None,
        vocabulary: PolicyVocabulary | None = None,
        weight_decay: float = 1e-4,
    ):
        self.enabled = torch is not None
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.temperature = max(0.1, float(temperature))
        self.branch_factor = max(1, int(branch_factor))
        self.revisit_probability = max(0.0, min(1.0, float(revisit_probability)))
        self.max_degree = 1
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.model_version = str(model_version or "v2").lower()
        self.feature_schema = {"version": FEATURE_SCHEMA_VERSION}
        self.vocabulary = vocabulary or PolicyVocabulary.empty()
        self.model_config = (model_config or PolicyModelConfig()).apply_vocabulary(self.vocabulary)

        if not self.enabled:
            self.model = None
            self.optimizer = None
            return

        torch.manual_seed(seed)
        self.model = None
        self.optimizer = None
        self._build_model()

    def _build_model(self) -> None:
        if not self.enabled:
            return
        if self.model_version == "v1":
            self.model = PolicyNet(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.model_version = "v2"
            self.model_config = self.model_config.apply_vocabulary(self.vocabulary)
            self.model = PolicyNetV2(self.model_config)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

    def _model_device(self) -> torch.device:
        if self.model is None:
            return torch.device("cpu")
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def set_max_degree(self, max_degree: int) -> None:
        self.max_degree = max(1, int(max_degree))

    def allow_revisit(self) -> bool:
        return self.rng.random() < self.revisit_probability

    def config_dict(self) -> dict[str, Any]:
        base = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "temperature": self.temperature,
            "branch_factor": self.branch_factor,
            "revisit_probability": self.revisit_probability,
            "seed": self.seed,
            "max_degree": self.max_degree,
            "model_version": self.model_version,
        }
        if self.model_version == "v2":
            base["model_config"] = self.model_config.to_dict()
        return base

    def set_vocabulary(self, vocabulary: PolicyVocabulary | None) -> None:
        if vocabulary is None:
            return
        self.vocabulary = vocabulary
        if self.model_version == "v2":
            self.model_config = self.model_config.apply_vocabulary(vocabulary)
            self._build_model()

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        scheduler_state: dict[str, Any] | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            raise RuntimeError("torch not available; cannot save EdgePolicy checkpoint")
        payload = {
            "format_version": CHECKPOINT_VERSION_V2 if self.model_version == "v2" else "tmcra_tri_maze_policy_v1",
            "config": self.config_dict(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "metadata": metadata or {},
            "scheduler_state": scheduler_state,
            "extra_state": extra_state or {},
        }
        if self.model_version == "v2":
            payload["vocabulary"] = self.vocabulary.to_metadata()
            payload["feature_schema"] = self.feature_schema
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)

    def _load_v1_checkpoint(
        self,
        payload: dict[str, Any],
        *,
        load_optimizer: bool,
        strict: bool,
    ) -> dict[str, Any]:
        config = payload.get("config") or {}
        required = {"input_dim", "hidden_dim", "lr", "temperature", "branch_factor", "revisit_probability", "seed"}
        if not required.issubset(config):
            raise ValueError("invalid EdgePolicy checkpoint: missing config keys")
        self.model_version = "v1"
        self.input_dim = int(config["input_dim"])
        self.hidden_dim = int(config["hidden_dim"])
        self.lr = float(config["lr"])
        self.temperature = float(config["temperature"])
        self.branch_factor = int(config["branch_factor"])
        self.revisit_probability = float(config["revisit_probability"])
        self.seed = int(config["seed"])
        self.max_degree = int(config.get("max_degree", self.max_degree))
        self.rng = random.Random(self.seed)
        self._build_model()
        self.model.load_state_dict(payload["model_state"], strict=strict)
        if load_optimizer and payload.get("optimizer_state") is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        return payload.get("metadata") or {}

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizer: bool = False,
        strict: bool = True,
        preserve_vocabulary: bool = False,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("torch not available; cannot load EdgePolicy checkpoint")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        config = payload.get("config") or {}
        if payload.get("format_version") == CHECKPOINT_VERSION_V2 or config.get("model_version") == "v2":
            self.model_version = "v2"
            if not preserve_vocabulary:
                self.lr = float(config.get("lr", self.lr))
                self.weight_decay = float(config.get("weight_decay", self.weight_decay))
                self.temperature = float(config.get("temperature", self.temperature))
                self.branch_factor = int(config.get("branch_factor", self.branch_factor))
                self.revisit_probability = float(config.get("revisit_probability", self.revisit_probability))
                self.seed = int(config.get("seed", self.seed))
                self.max_degree = int(config.get("max_degree", self.max_degree))
                self.rng = random.Random(self.seed)
            checkpoint_vocabulary = PolicyVocabulary.from_metadata(payload.get("vocabulary"))
            if preserve_vocabulary:
                current_vocabulary = self.vocabulary or PolicyVocabulary.empty()
                self.vocabulary = current_vocabulary
            else:
                self.vocabulary = checkpoint_vocabulary
            self.feature_schema = dict(payload.get("feature_schema") or {"version": FEATURE_SCHEMA_VERSION})
            self.model_config = PolicyModelConfig.from_dict(config.get("model_config")).apply_vocabulary(self.vocabulary)
            self._build_model()
            if preserve_vocabulary:
                target_state = self.model.state_dict()
                source_state = payload["model_state"]
                for name, source_tensor in source_state.items():
                    target_tensor = target_state.get(name)
                    if target_tensor is None:
                        continue
                    if target_tensor.shape == source_tensor.shape:
                        target_state[name] = source_tensor
                _copy_embedding_rows_by_token(
                    target_state["concept_embedding.weight"],
                    source_state.get("concept_embedding.weight"),
                    self.vocabulary.concept_to_id,
                    checkpoint_vocabulary.concept_to_id,
                )
                _copy_embedding_rows_by_token(
                    target_state["relation_embedding.weight"],
                    source_state.get("relation_embedding.weight"),
                    self.vocabulary.relation_to_id,
                    checkpoint_vocabulary.relation_to_id,
                )
                _copy_embedding_rows_by_token(
                    target_state["source_embedding.weight"],
                    source_state.get("source_embedding.weight"),
                    self.vocabulary.source_to_id,
                    checkpoint_vocabulary.source_to_id,
                )
                _copy_embedding_rows_by_token(
                    target_state["query_type_embedding.weight"],
                    source_state.get("query_type_embedding.weight"),
                    self.vocabulary.query_type_to_id,
                    checkpoint_vocabulary.query_type_to_id,
                )
                _copy_embedding_rows_by_token(
                    target_state["mode_embedding.weight"],
                    source_state.get("mode_embedding.weight"),
                    self.vocabulary.mode_to_id,
                    checkpoint_vocabulary.mode_to_id,
                )
                _copy_embedding_rows_by_token(
                    target_state["task_embedding.weight"],
                    source_state.get("task_embedding.weight"),
                    self.vocabulary.task_to_id,
                    checkpoint_vocabulary.task_to_id,
                )
                _copy_prefix_tensor(
                    target_state["hash_embedding.weight"],
                    source_state.get("hash_embedding.weight"),
                )
                self.model.load_state_dict(target_state, strict=False)
            else:
                self.model.load_state_dict(payload["model_state"], strict=strict)
            if load_optimizer and not preserve_vocabulary and payload.get("optimizer_state") is not None and self.optimizer is not None:
                self.optimizer.load_state_dict(payload["optimizer_state"])
            return payload.get("metadata") or {}
        return self._load_v1_checkpoint(payload, load_optimizer=load_optimizer, strict=strict)

    def _encode_runtime_batch(
        self,
        engine,
        current_node,
        candidate_edges,
        path,
        visited: set[str],
        mode: str,
        *,
        target_index: int = -1,
        sample_id: str | None = None,
        weight: float = 1.0,
    ) -> PolicyBatch:
        record = build_runtime_step_record(
            engine=engine,
            current_node=current_node,
            candidate_edges=candidate_edges,
            path=path,
            visited=visited,
            mode=mode,
            target_index=target_index,
            sample_id=sample_id,
            weight=weight,
            source_kind="runtime_update" if target_index >= 0 else "runtime",
            source_dataset="runtime",
            query_type="runtime",
            task_key=f"runtime|runtime|{mode}",
        )
        dataset = PolicyStepDataset(
            [record],
            vocabulary=self.vocabulary,
            history_size=self.model_config.history_size,
        )
        batch = policy_collate_fn([dataset[0]])
        return batch.to(self._model_device())

    def evaluate_candidates(self, engine, current_node, candidate_edges, path, visited: set, mode: str = "forward"):
        if not self.enabled or not candidate_edges:
            return None
        self.model.eval()
        with torch.no_grad():
            if self.model_version == "v1":
                features = torch.tensor(
                    [
                        row
                        for row in [
                            build_runtime_step_record(
                                engine,
                                current_node,
                                [edge],
                                path=path,
                                visited=visited,
                                mode=mode,
                                sample_id=f"single::{index}",
                            ).candidate_base_features[0]
                            for index, edge in enumerate(candidate_edges)
                        ]
                    ],
                    dtype=torch.float32,
                    device=self._model_device(),
                )
                logits = self.model(features)
                probs = torch.softmax(logits / self.temperature, dim=0)
                return {"features": features, "logits": logits, "probs": probs}

            batch = self._encode_runtime_batch(
                engine=engine,
                current_node=current_node,
                candidate_edges=candidate_edges,
                path=path,
                visited=visited,
                mode=mode,
            )
            outputs = self.model(batch)
            logits = outputs["logits"][0, : len(candidate_edges)]
            probs = torch.softmax(logits / self.temperature, dim=0)
            return {"batch": batch, "outputs": outputs, "logits": logits, "probs": probs}

    def score_edges(self, engine, current_node, edges, path, visited: set, mode: str):
        if not self.enabled or not edges:
            return None
        evaluated = self.evaluate_candidates(engine, current_node, edges, path, visited, mode)
        if evaluated is None:
            return None
        return evaluated["logits"]

    def select_edges(
        self,
        engine,
        current_node,
        candidate_edges,
        path,
        visited: set,
        mode: str = "forward",
        k: Optional[int] = None,
        deterministic: bool = False,
    ):
        if not self.enabled or not candidate_edges:
            return []

        evaluated = self.evaluate_candidates(engine, current_node, candidate_edges, path, visited, mode)
        if evaluated is None:
            return []
        probs = evaluated["probs"]

        k = k or self.branch_factor
        k = max(1, min(int(k), len(candidate_edges)))

        if deterministic:
            top_idx = torch.topk(probs, k=k).indices.tolist()
            return [candidate_edges[i] for i in top_idx]

        if k == 1:
            idx = torch.multinomial(probs, num_samples=1).item()
            return [candidate_edges[idx]]

        idxs = torch.multinomial(probs, num_samples=k, replacement=False).tolist()
        return [candidate_edges[i] for i in idxs]

    def _online_supervised_step(self, batch: PolicyBatch) -> float:
        if self.optimizer is None:
            return 0.0
        self.model.train()
        outputs = self.model(batch)
        loss = masked_cross_entropy(
            outputs["logits"],
            batch.target_index,
            batch.candidate_mask,
            sample_weights=batch.weights,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def warm_start_from_memory(
        self,
        engine,
        max_paths: int = 200,
        max_steps: int = 800,
    ) -> int:
        if not self.enabled:
            return 0
        memory = getattr(engine, "memory", None)
        if not memory:
            return 0
        paths = memory.get_all_paths()
        if not paths:
            return 0

        steps = 0
        self.rng.shuffle(paths)
        for path_item in paths[:max_paths]:
            concept_list = [str(item).strip() for item in (path_item.get("path") or []) if str(item).strip()]
            if len(concept_list) < 2:
                continue
            weight = max(0.1, min(1.0, float(path_item.get("score", 0.5) or 0.5)))
            visited: set[str] = set()
            for index in range(len(concept_list) - 1):
                if steps >= max_steps:
                    return steps
                current = concept_list[index]
                target = concept_list[index + 1]
                current_node = engine.nodes.get(current)
                if not current_node:
                    continue
                edges = list(current_node.connections)
                target_idx = next((item for item, edge in enumerate(edges) if edge.to_node.concept == target), None)
                if target_idx is None:
                    continue
                visited.add(current)
                path_stub = type("PathStub", (), {"length": index, "nodes": [engine.nodes[c] for c in concept_list[: index + 1] if c in engine.nodes], "edges": []})()
                if self.model_version == "v1":
                    features = torch.tensor(
                        [build_runtime_step_record(engine, current_node, [edge], path=path_stub, visited=visited, mode="forward").candidate_base_features[0] for edge in edges],
                        dtype=torch.float32,
                        device=self._model_device(),
                    )
                    logits = self.model(features).unsqueeze(0)
                    target_tensor = torch.tensor([target_idx], dtype=torch.long, device=self._model_device())
                    loss = F.cross_entropy(logits, target_tensor) * weight
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    batch = self._encode_runtime_batch(
                        engine=engine,
                        current_node=current_node,
                        candidate_edges=edges,
                        path=path_stub,
                        visited=visited,
                        mode="forward",
                        target_index=target_idx,
                        sample_id=f"warm::{current}::{target}::{index}",
                        weight=weight,
                    )
                    self._online_supervised_step(batch)
                steps += 1
        return steps

    def update_from_path(self, engine, path, mode: str = "forward") -> int:
        if not self.enabled or not path or not getattr(path, "edges", None):
            return 0

        reward = 1.0
        if hasattr(path, "score") and hasattr(engine, "length_penalty"):
            reward = max(0.05, 1.0 - float(path.score(engine.length_penalty)))

        visited: set[str] = set()
        steps = 0
        for index, edge in enumerate(path.edges):
            current_node = path.nodes[index]
            visited.add(current_node.concept)
            candidate_edges = list(current_node.connections)
            if not candidate_edges:
                continue
            try:
                target_idx = candidate_edges.index(edge)
            except ValueError:
                continue
            if self.model_version == "v1":
                features = torch.tensor(
                    [build_runtime_step_record(engine, current_node, [candidate], path=path, visited=visited, mode=mode).candidate_base_features[0] for candidate in candidate_edges],
                    dtype=torch.float32,
                    device=self._model_device(),
                )
                logits = self.model(features).unsqueeze(0)
                target_tensor = torch.tensor([target_idx], dtype=torch.long, device=self._model_device())
                loss = F.cross_entropy(logits, target_tensor) * reward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                batch = self._encode_runtime_batch(
                    engine=engine,
                    current_node=current_node,
                    candidate_edges=candidate_edges,
                    path=path,
                    visited=visited,
                    mode=mode,
                    target_index=target_idx,
                    sample_id=f"update::{current_node.concept}::{index}",
                    weight=reward,
                )
                self._online_supervised_step(batch)
            steps += 1
        return steps
