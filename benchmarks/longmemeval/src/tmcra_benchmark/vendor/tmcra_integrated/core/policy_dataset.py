from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

try:
    import torch
    from torch.utils.data import Dataset, Sampler
except Exception as exc:  # pragma: no cover - torch is required for trainer usage
    raise RuntimeError(f"torch is required for Tri-Maze policy datasets: {exc}")


BASE_FEATURE_DIM = 13
PATH_CONTEXT_DIM = 6
CANDIDATE_FLAG_DIM = 4
DEFAULT_HISTORY_SIZE = 4
FEATURE_SCHEMA_VERSION = "tri_maze_policy_v2.0"
PAD_ID = 0
UNK_ID = 1


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\u00a0", " ").strip().split())


def _stable_hash(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:8], "big")


def _hash_bucket(text: str, bucket_count: int) -> int:
    if bucket_count <= 0:
        return 0
    normalized = _normalize_text(text)
    if not normalized:
        return 0
    return 1 + (_stable_hash(normalized.casefold()) % bucket_count)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return float(default)


def _mode_one_hot(mode: str) -> list[float]:
    normalized = (_normalize_text(mode) or "forward").lower()
    if normalized == "reverse":
        return [0.0, 1.0, 0.0]
    if normalized == "boundary":
        return [0.0, 0.0, 1.0]
    return [1.0, 0.0, 0.0]


def compute_candidate_base_features(
    engine: Any,
    current_node: Any,
    edge: Any,
    *,
    path: Any = None,
    visited: set[str] | None = None,
    mode: str = "forward",
    max_degree: int | None = None,
) -> list[float]:
    visited = visited or set()
    next_node = edge.to_node
    degree_cur = len(getattr(current_node, "connections", []) or [])
    degree_next = len(getattr(next_node, "connections", []) or [])
    max_degree = max(1, int(max_degree or getattr(engine, "_max_degree", 1) or 1))

    resistance = float(getattr(edge, "resistance", 0.5) or 0.5)
    memory_reinf = 0.0
    if getattr(engine, "memory", None):
        try:
            memory_reinf = float(
                engine.memory.get_edge_reinforcement(current_node.concept, next_node.concept)
            )
        except Exception:
            memory_reinf = 0.0

    semantic = 0.0
    if hasattr(engine, "_semantic_similarity"):
        try:
            semantic = float(engine._semantic_similarity(current_node.concept, next_node.concept))
        except Exception:
            semantic = 0.0

    visited_flag = 1.0 if next_node.concept in visited else 0.0
    path_len = float(getattr(path, "length", 0) or 0)
    max_steps = float(getattr(engine, "max_exploration_steps", 1) or 1)

    features = [
        resistance,
        memory_reinf,
        semantic,
        min(1.0, degree_cur / max_degree),
        min(1.0, degree_next / max_degree),
        min(1.0, path_len / max_steps),
        visited_flag,
        1.0 if getattr(edge, "is_memory", False) else 0.0,
        1.0 if getattr(edge, "is_expanded", False) else 0.0,
        1.0 if getattr(edge, "is_tunneling", False) else 0.0,
    ]
    features.extend(_mode_one_hot(mode))
    return features


def build_path_context_features(
    *,
    path_length: int,
    visited_count: int,
    candidate_count: int,
    max_degree: int,
    history_count: int,
    revisit_ratio: float,
    max_steps: int | None = None,
) -> list[float]:
    max_degree = max(1, int(max_degree or 1))
    max_steps = max(1, int(max_steps or 80))
    path_length_norm = min(1.0, max(0, path_length) / max_steps)
    candidate_density = min(1.0, max(0, candidate_count) / max_degree)
    visited_density = min(1.0, max(0, visited_count) / max(1, path_length + candidate_count + 1))
    branch_density = min(1.0, max(0, candidate_count) / max(1, visited_count + candidate_count))
    history_fill = min(1.0, max(0, history_count) / max(1, DEFAULT_HISTORY_SIZE))
    revisit_ratio = _clamp01(revisit_ratio)
    return [
        path_length_norm,
        candidate_density,
        visited_density,
        branch_density,
        history_fill,
        revisit_ratio,
    ]


def _path_length_bucket(path_length: int) -> int:
    if path_length <= 1:
        return 0
    if path_length <= 3:
        return 1
    if path_length <= 5:
        return 2
    return 3


@dataclass(slots=True)
class CurriculumConfig:
    enabled: bool = False
    min_fraction: float = 0.35
    max_fraction: float = 1.0
    include_tunneling_after: float = 0.45
    include_hard_after: float = 0.55

    def fraction_for_epoch(self, epoch: int, total_epochs: int) -> float:
        if not self.enabled:
            return self.max_fraction
        total_epochs = max(1, int(total_epochs))
        progress = min(1.0, max(0.0, float(epoch) / max(1, total_epochs - 1)))
        return self.min_fraction + (self.max_fraction - self.min_fraction) * progress


@dataclass(slots=True)
class PolicyStepRecord:
    sample_id: str
    source_kind: str
    source_dataset: str
    query_type: str
    mode: str
    task_key: str
    current_concept: str
    recent_concepts: tuple[str, ...]
    visited_concepts: tuple[str, ...]
    candidate_concepts: tuple[str, ...]
    candidate_relations: tuple[str, ...]
    candidate_base_features: tuple[tuple[float, ...], ...]
    candidate_is_memory: tuple[int, ...]
    candidate_is_expanded: tuple[int, ...]
    candidate_is_tunneling: tuple[int, ...]
    target_index: int
    weight: float
    path_length: int
    path_length_bucket: int
    difficulty: float
    high_value_target: int
    has_tunneling_path: int
    source_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "recent_concepts",
            "visited_concepts",
            "candidate_concepts",
            "candidate_relations",
            "candidate_is_memory",
            "candidate_is_expanded",
            "candidate_is_tunneling",
        ):
            payload[key] = list(payload[key])
        payload["candidate_base_features"] = [list(item) for item in self.candidate_base_features]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PolicyStepRecord":
        return cls(
            sample_id=_normalize_text(payload.get("sample_id") or ""),
            source_kind=_normalize_text(payload.get("source_kind") or "unknown"),
            source_dataset=_normalize_text(payload.get("source_dataset") or "unknown"),
            query_type=_normalize_text(payload.get("query_type") or "query"),
            mode=_normalize_text(payload.get("mode") or "forward"),
            task_key=_normalize_text(payload.get("task_key") or "default"),
            current_concept=_normalize_text(payload.get("current_concept") or ""),
            recent_concepts=tuple(_normalize_text(item) for item in payload.get("recent_concepts") or [] if _normalize_text(item)),
            visited_concepts=tuple(_normalize_text(item) for item in payload.get("visited_concepts") or [] if _normalize_text(item)),
            candidate_concepts=tuple(_normalize_text(item) for item in payload.get("candidate_concepts") or [] if _normalize_text(item)),
            candidate_relations=tuple(_normalize_text(item) for item in payload.get("candidate_relations") or [] if _normalize_text(item)),
            candidate_base_features=tuple(
                tuple(float(value) for value in feature_row)
                for feature_row in payload.get("candidate_base_features") or []
            ),
            candidate_is_memory=tuple(int(value) for value in payload.get("candidate_is_memory") or []),
            candidate_is_expanded=tuple(int(value) for value in payload.get("candidate_is_expanded") or []),
            candidate_is_tunneling=tuple(int(value) for value in payload.get("candidate_is_tunneling") or []),
            target_index=int(payload.get("target_index", -1)),
            weight=float(payload.get("weight", 1.0) or 1.0),
            path_length=int(payload.get("path_length", 0) or 0),
            path_length_bucket=int(payload.get("path_length_bucket", _path_length_bucket(int(payload.get("path_length", 0) or 0)))),
            difficulty=float(payload.get("difficulty", 0.0) or 0.0),
            high_value_target=int(payload.get("high_value_target", 0) or 0),
            has_tunneling_path=int(payload.get("has_tunneling_path", 0) or 0),
            source_score=float(payload.get("source_score", 0.0) or 0.0),
            metadata=dict(payload.get("metadata") or {}),
        )

    def all_concepts(self) -> set[str]:
        concepts = {self.current_concept}
        concepts.update(self.recent_concepts)
        concepts.update(self.visited_concepts)
        concepts.update(self.candidate_concepts)
        return {item for item in concepts if item}


@dataclass(slots=True)
class PolicyVocabulary:
    concept_to_id: dict[str, int]
    relation_to_id: dict[str, int]
    source_to_id: dict[str, int]
    query_type_to_id: dict[str, int]
    mode_to_id: dict[str, int]
    task_to_id: dict[str, int]
    hash_bucket_size: int = 8192

    @classmethod
    def empty(cls, *, hash_bucket_size: int = 8192) -> "PolicyVocabulary":
        return cls(
            concept_to_id={"<pad>": PAD_ID, "<unk>": UNK_ID},
            relation_to_id={"<pad>": PAD_ID, "<unk>": UNK_ID},
            source_to_id={"<pad>": PAD_ID, "<unk>": UNK_ID},
            query_type_to_id={"<pad>": PAD_ID, "<unk>": UNK_ID},
            mode_to_id={"<pad>": PAD_ID, "<unk>": UNK_ID, "forward": 2, "reverse": 3, "boundary": 4, "runtime": 5},
            task_to_id={"<pad>": PAD_ID, "<unk>": UNK_ID},
            hash_bucket_size=max(32, int(hash_bucket_size)),
        )

    @classmethod
    def build(
        cls,
        records: Sequence[PolicyStepRecord],
        *,
        hash_bucket_size: int = 8192,
    ) -> "PolicyVocabulary":
        vocab = cls.empty(hash_bucket_size=hash_bucket_size)
        for record in records:
            vocab._add(vocab.concept_to_id, record.current_concept)
            for concept in record.recent_concepts:
                vocab._add(vocab.concept_to_id, concept)
            for concept in record.visited_concepts:
                vocab._add(vocab.concept_to_id, concept)
            for concept in record.candidate_concepts:
                vocab._add(vocab.concept_to_id, concept)
            for relation in record.candidate_relations:
                vocab._add(vocab.relation_to_id, relation)
            vocab._add(vocab.source_to_id, record.source_dataset)
            vocab._add(vocab.query_type_to_id, record.query_type)
            vocab._add(vocab.mode_to_id, record.mode)
            vocab._add(vocab.task_to_id, record.task_key)
        return vocab

    @staticmethod
    def _key(value: str) -> str:
        normalized = _normalize_text(value)
        return normalized.casefold() if normalized else ""

    def _add(self, mapping: dict[str, int], value: str) -> None:
        key = self._key(value)
        if not key or key in mapping:
            return
        mapping[key] = len(mapping)

    def encode_token(self, mapping: dict[str, int], value: str) -> int:
        key = self._key(value)
        if not key:
            return PAD_ID
        return int(mapping.get(key, UNK_ID))

    def encode_concept(self, value: str) -> tuple[int, int]:
        return (
            self.encode_token(self.concept_to_id, value),
            _hash_bucket(value, self.hash_bucket_size),
        )

    def encode_relation(self, value: str) -> int:
        return self.encode_token(self.relation_to_id, value)

    def encode_source(self, value: str) -> int:
        return self.encode_token(self.source_to_id, value)

    def encode_query_type(self, value: str) -> int:
        return self.encode_token(self.query_type_to_id, value)

    def encode_mode(self, value: str) -> int:
        return self.encode_token(self.mode_to_id, value)

    def encode_task(self, value: str) -> int:
        return self.encode_token(self.task_to_id, value)

    @property
    def concept_vocab_size(self) -> int:
        return len(self.concept_to_id)

    @property
    def relation_vocab_size(self) -> int:
        return len(self.relation_to_id)

    @property
    def source_vocab_size(self) -> int:
        return len(self.source_to_id)

    @property
    def query_type_vocab_size(self) -> int:
        return len(self.query_type_to_id)

    @property
    def mode_vocab_size(self) -> int:
        return len(self.mode_to_id)

    @property
    def task_vocab_size(self) -> int:
        return len(self.task_to_id)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "concept_to_id": self.concept_to_id,
            "relation_to_id": self.relation_to_id,
            "source_to_id": self.source_to_id,
            "query_type_to_id": self.query_type_to_id,
            "mode_to_id": self.mode_to_id,
            "task_to_id": self.task_to_id,
            "hash_bucket_size": self.hash_bucket_size,
        }

    @classmethod
    def from_metadata(cls, payload: dict[str, Any] | None) -> "PolicyVocabulary":
        if not payload:
            return cls.empty()
        return cls(
            concept_to_id={str(k): int(v) for k, v in dict(payload.get("concept_to_id") or {}).items()},
            relation_to_id={str(k): int(v) for k, v in dict(payload.get("relation_to_id") or {}).items()},
            source_to_id={str(k): int(v) for k, v in dict(payload.get("source_to_id") or {}).items()},
            query_type_to_id={str(k): int(v) for k, v in dict(payload.get("query_type_to_id") or {}).items()},
            mode_to_id={str(k): int(v) for k, v in dict(payload.get("mode_to_id") or {}).items()},
            task_to_id={str(k): int(v) for k, v in dict(payload.get("task_to_id") or {}).items()},
            hash_bucket_size=max(32, int(payload.get("hash_bucket_size", 8192) or 8192)),
        )


@dataclass(slots=True)
class PolicyBatch:
    sample_ids: list[str]
    source_kinds: list[str]
    current_concept_ids: torch.Tensor
    current_hash_ids: torch.Tensor
    history_concept_ids: torch.Tensor
    history_hash_ids: torch.Tensor
    candidate_concept_ids: torch.Tensor
    candidate_hash_ids: torch.Tensor
    relation_ids: torch.Tensor
    source_ids: torch.Tensor
    query_type_ids: torch.Tensor
    mode_ids: torch.Tensor
    task_ids: torch.Tensor
    base_features: torch.Tensor
    path_context: torch.Tensor
    candidate_flags: torch.Tensor
    candidate_mask: torch.Tensor
    target_index: torch.Tensor
    weights: torch.Tensor
    path_length_bucket: torch.Tensor
    tunnel_label: torch.Tensor
    high_value_label: torch.Tensor
    difficulty: torch.Tensor
    candidate_count: torch.Tensor

    def to(self, device: torch.device | str) -> "PolicyBatch":
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if torch.is_tensor(value):
                setattr(self, field_name, value.to(device))
        return self

    @property
    def batch_size(self) -> int:
        return int(self.current_concept_ids.shape[0])


class PolicyStepDataset(Dataset):
    def __init__(
        self,
        records: Sequence[PolicyStepRecord],
        *,
        vocabulary: PolicyVocabulary | None = None,
        history_size: int = DEFAULT_HISTORY_SIZE,
        feature_schema_version: str = FEATURE_SCHEMA_VERSION,
    ) -> None:
        self.records = list(records)
        self.history_size = max(1, int(history_size))
        self.feature_schema_version = feature_schema_version
        self.vocabulary = vocabulary or PolicyVocabulary.build(self.records)
        self._encoded = [self._encode_record(record) for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._encoded[index]

    def _encode_record(self, record: PolicyStepRecord) -> dict[str, Any]:
        vocab = self.vocabulary
        history = list(record.recent_concepts[-self.history_size :])
        while len(history) < self.history_size:
            history.insert(0, "")

        current_id, current_hash = vocab.encode_concept(record.current_concept)
        history_ids: list[int] = []
        history_hash_ids: list[int] = []
        for concept in history:
            concept_id, hash_id = vocab.encode_concept(concept)
            history_ids.append(concept_id)
            history_hash_ids.append(hash_id)

        candidate_ids: list[int] = []
        candidate_hash_ids: list[int] = []
        relation_ids: list[int] = []
        candidate_flags: list[list[float]] = []
        candidate_features: list[list[float]] = []
        candidate_count = len(record.candidate_concepts)
        visited_set = {item.casefold() for item in record.visited_concepts}
        revisit_hits = 0
        for index, concept in enumerate(record.candidate_concepts):
            concept_id, hash_id = vocab.encode_concept(concept)
            candidate_ids.append(concept_id)
            candidate_hash_ids.append(hash_id)
            relation_ids.append(vocab.encode_relation(record.candidate_relations[index] if index < len(record.candidate_relations) else "related_to"))
            visited_flag = 1.0 if concept.casefold() in visited_set else 0.0
            revisit_hits += int(visited_flag > 0.0)
            candidate_flags.append(
                [
                    float(record.candidate_is_memory[index] if index < len(record.candidate_is_memory) else 0),
                    float(record.candidate_is_expanded[index] if index < len(record.candidate_is_expanded) else 0),
                    float(record.candidate_is_tunneling[index] if index < len(record.candidate_is_tunneling) else 0),
                    visited_flag,
                ]
            )
            feature_row = list(record.candidate_base_features[index] if index < len(record.candidate_base_features) else ())
            if len(feature_row) < BASE_FEATURE_DIM:
                feature_row.extend([0.0] * (BASE_FEATURE_DIM - len(feature_row)))
            candidate_features.append(feature_row[:BASE_FEATURE_DIM])

        path_context = build_path_context_features(
            path_length=record.path_length,
            visited_count=len(record.visited_concepts),
            candidate_count=candidate_count,
            max_degree=max(candidate_count, 1),
            history_count=len(record.recent_concepts),
            revisit_ratio=float(revisit_hits / max(1, candidate_count)),
        )

        return {
            "sample_id": record.sample_id,
            "source_kind": record.source_kind,
            "current_concept_id": current_id,
            "current_hash_id": current_hash,
            "history_concept_ids": history_ids,
            "history_hash_ids": history_hash_ids,
            "candidate_concept_ids": candidate_ids,
            "candidate_hash_ids": candidate_hash_ids,
            "relation_ids": relation_ids,
            "source_id": vocab.encode_source(record.source_dataset),
            "query_type_id": vocab.encode_query_type(record.query_type),
            "mode_id": vocab.encode_mode(record.mode),
            "task_id": vocab.encode_task(record.task_key),
            "base_features": candidate_features,
            "path_context": path_context,
            "candidate_flags": candidate_flags,
            "target_index": int(record.target_index),
            "weight": float(record.weight),
            "path_length_bucket": int(record.path_length_bucket),
            "tunnel_label": int(record.has_tunneling_path),
            "high_value_label": int(record.high_value_target),
            "difficulty": float(record.difficulty),
            "candidate_count": int(candidate_count),
        }


def policy_collate_fn(batch: Sequence[dict[str, Any]]) -> PolicyBatch:
    if not batch:
        raise ValueError("cannot collate empty policy batch")

    batch_size = len(batch)
    max_candidates = max(1, max(int(item["candidate_count"]) for item in batch))
    history_size = max(1, len(batch[0]["history_concept_ids"]))

    current_concept_ids = torch.zeros(batch_size, dtype=torch.long)
    current_hash_ids = torch.zeros(batch_size, dtype=torch.long)
    history_concept_ids = torch.zeros(batch_size, history_size, dtype=torch.long)
    history_hash_ids = torch.zeros(batch_size, history_size, dtype=torch.long)
    candidate_concept_ids = torch.zeros(batch_size, max_candidates, dtype=torch.long)
    candidate_hash_ids = torch.zeros(batch_size, max_candidates, dtype=torch.long)
    relation_ids = torch.zeros(batch_size, max_candidates, dtype=torch.long)
    source_ids = torch.zeros(batch_size, dtype=torch.long)
    query_type_ids = torch.zeros(batch_size, dtype=torch.long)
    mode_ids = torch.zeros(batch_size, dtype=torch.long)
    task_ids = torch.zeros(batch_size, dtype=torch.long)
    base_features = torch.zeros(batch_size, max_candidates, BASE_FEATURE_DIM, dtype=torch.float32)
    path_context = torch.zeros(batch_size, PATH_CONTEXT_DIM, dtype=torch.float32)
    candidate_flags = torch.zeros(batch_size, max_candidates, CANDIDATE_FLAG_DIM, dtype=torch.float32)
    candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    target_index = torch.full((batch_size,), -1, dtype=torch.long)
    weights = torch.ones(batch_size, dtype=torch.float32)
    path_length_bucket = torch.zeros(batch_size, dtype=torch.long)
    tunnel_label = torch.zeros(batch_size, dtype=torch.float32)
    high_value_label = torch.zeros(batch_size, dtype=torch.float32)
    difficulty = torch.zeros(batch_size, dtype=torch.float32)
    candidate_count = torch.zeros(batch_size, dtype=torch.long)

    sample_ids: list[str] = []
    source_kinds: list[str] = []

    for row_index, row in enumerate(batch):
        count = int(row["candidate_count"])
        sample_ids.append(str(row["sample_id"]))
        source_kinds.append(str(row["source_kind"]))
        current_concept_ids[row_index] = int(row["current_concept_id"])
        current_hash_ids[row_index] = int(row["current_hash_id"])
        history_concept_ids[row_index] = torch.tensor(row["history_concept_ids"], dtype=torch.long)
        history_hash_ids[row_index] = torch.tensor(row["history_hash_ids"], dtype=torch.long)
        if count > 0:
            candidate_concept_ids[row_index, :count] = torch.tensor(row["candidate_concept_ids"], dtype=torch.long)
            candidate_hash_ids[row_index, :count] = torch.tensor(row["candidate_hash_ids"], dtype=torch.long)
            relation_ids[row_index, :count] = torch.tensor(row["relation_ids"], dtype=torch.long)
            base_features[row_index, :count] = torch.tensor(row["base_features"], dtype=torch.float32)
            candidate_flags[row_index, :count] = torch.tensor(row["candidate_flags"], dtype=torch.float32)
            candidate_mask[row_index, :count] = True
        source_ids[row_index] = int(row["source_id"])
        query_type_ids[row_index] = int(row["query_type_id"])
        mode_ids[row_index] = int(row["mode_id"])
        task_ids[row_index] = int(row["task_id"])
        path_context[row_index] = torch.tensor(row["path_context"], dtype=torch.float32)
        target_index[row_index] = int(row["target_index"])
        weights[row_index] = float(row["weight"])
        path_length_bucket[row_index] = int(row["path_length_bucket"])
        tunnel_label[row_index] = float(row["tunnel_label"])
        high_value_label[row_index] = float(row["high_value_label"])
        difficulty[row_index] = float(row["difficulty"])
        candidate_count[row_index] = int(count)

    return PolicyBatch(
        sample_ids=sample_ids,
        source_kinds=source_kinds,
        current_concept_ids=current_concept_ids,
        current_hash_ids=current_hash_ids,
        history_concept_ids=history_concept_ids,
        history_hash_ids=history_hash_ids,
        candidate_concept_ids=candidate_concept_ids,
        candidate_hash_ids=candidate_hash_ids,
        relation_ids=relation_ids,
        source_ids=source_ids,
        query_type_ids=query_type_ids,
        mode_ids=mode_ids,
        task_ids=task_ids,
        base_features=base_features,
        path_context=path_context,
        candidate_flags=candidate_flags,
        candidate_mask=candidate_mask,
        target_index=target_index,
        weights=weights,
        path_length_bucket=path_length_bucket,
        tunnel_label=tunnel_label,
        high_value_label=high_value_label,
        difficulty=difficulty,
        candidate_count=candidate_count,
    )


class EpisodicBatchSampler(Sampler[list[int]]):
    def __init__(self, records: Sequence[PolicyStepRecord], *, batch_size: int, seed: int = 42) -> None:
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        task_groups: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            task_groups.setdefault(record.task_key or "default", []).append(index)
        self.task_groups = {key: value for key, value in task_groups.items() if value}
        self.task_keys = list(self.task_groups.keys())

    def __iter__(self) -> Iterator[list[int]]:
        if not self.task_keys:
            return iter(())
        generator = torch.Generator().manual_seed(self.seed)
        task_order = torch.randperm(len(self.task_keys), generator=generator).tolist()
        batches: list[list[int]] = []
        for task_index in task_order:
            indices = self.task_groups[self.task_keys[task_index]].copy()
            if len(indices) > 1:
                perm = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[item] for item in perm]
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start : start + self.batch_size])
        return iter(batches)

    def __len__(self) -> int:
        total = 0
        for indices in self.task_groups.values():
            total += math.ceil(len(indices) / self.batch_size)
        return total


def build_domain_sampling_weights(records: Sequence[PolicyStepRecord]) -> list[float]:
    if not records:
        return []
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (
            _normalize_text(record.source_dataset).casefold() or "unknown",
            _normalize_text(record.query_type).casefold() or "query",
        )
        counts[key] = counts.get(key, 0) + 1
    weights: list[float] = []
    for record in records:
        key = (
            _normalize_text(record.source_dataset).casefold() or "unknown",
            _normalize_text(record.query_type).casefold() or "query",
        )
        weights.append(1.0 / max(1, counts.get(key, 1)))
    return weights


def filter_curriculum_records(
    records: Sequence[PolicyStepRecord],
    *,
    epoch: int,
    total_epochs: int,
    curriculum: CurriculumConfig,
) -> list[PolicyStepRecord]:
    if not curriculum.enabled or not records:
        return list(records)
    ordered = sorted(records, key=lambda item: (item.difficulty, item.path_length, item.sample_id))
    cutoff = max(1, int(math.ceil(len(ordered) * curriculum.fraction_for_epoch(epoch, total_epochs))))
    allowed = ordered[:cutoff]
    progress = min(1.0, max(0.0, float(epoch) / max(1, total_epochs - 1)))
    if progress >= curriculum.include_tunneling_after:
        tunneling_records = [record for record in records if record.has_tunneling_path and record not in allowed]
        allowed.extend(tunneling_records)
    if progress >= curriculum.include_hard_after:
        hard_records = [record for record in records if len(record.candidate_concepts) > 3 and record not in allowed]
        allowed.extend(hard_records)
    seen = set()
    deduped: list[PolicyStepRecord] = []
    for record in allowed:
        if record.sample_id in seen:
            continue
        seen.add(record.sample_id)
        deduped.append(record)
    return deduped


def serialize_policy_records(path: str | Path, records: Sequence[PolicyStepRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_policy_records(path: str | Path) -> list[PolicyStepRecord]:
    records: list[PolicyStepRecord] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(PolicyStepRecord.from_dict(payload))
    return records


def build_runtime_step_record(
    engine: Any,
    current_node: Any,
    candidate_edges: Sequence[Any],
    *,
    path: Any = None,
    visited: set[str] | None = None,
    mode: str = "forward",
    target_index: int = -1,
    sample_id: str | None = None,
    source_kind: str = "runtime",
    source_dataset: str = "runtime",
    query_type: str = "runtime",
    task_key: str = "runtime|runtime|forward",
    weight: float = 1.0,
    source_score: float = 0.0,
) -> PolicyStepRecord:
    visited = visited or set()
    candidate_edges = list(candidate_edges)
    candidate_concepts = tuple(getattr(edge.to_node, "concept", "") for edge in candidate_edges)
    candidate_relations = tuple(_normalize_text(getattr(edge, "relation", "") or "related_to") for edge in candidate_edges)
    base_features = tuple(
        tuple(
            compute_candidate_base_features(
                engine,
                current_node,
                edge,
                path=path,
                visited=visited,
                mode=mode,
                max_degree=getattr(engine, "_max_degree", 1),
            )
        )
        for edge in candidate_edges
    )
    candidate_is_memory = tuple(1 if getattr(edge, "is_memory", False) else 0 for edge in candidate_edges)
    candidate_is_expanded = tuple(1 if getattr(edge, "is_expanded", False) else 0 for edge in candidate_edges)
    candidate_is_tunneling = tuple(1 if getattr(edge, "is_tunneling", False) else 0 for edge in candidate_edges)

    path_nodes = [getattr(node, "concept", "") for node in getattr(path, "nodes", []) if getattr(node, "concept", "")]
    recent_concepts = tuple(path_nodes[-DEFAULT_HISTORY_SIZE:])
    visited_concepts = tuple(sorted(_normalize_text(item) for item in visited if _normalize_text(item)))
    candidate_count = len(candidate_edges)
    target_index = int(target_index)
    selected_edge = candidate_edges[target_index] if 0 <= target_index < len(candidate_edges) else None
    high_value_target = 0
    if selected_edge is not None:
        heuristic_score = 1.0 - float(getattr(selected_edge, "resistance", 0.5) or 0.5)
        heuristic_score += 0.25 * float(getattr(selected_edge, "is_memory", False))
        heuristic_score += 0.15 * float(getattr(selected_edge, "is_expanded", False))
        high_value_target = int(heuristic_score >= 0.75)
    has_tunneling_path = int(any(getattr(edge, "is_tunneling", False) for edge in getattr(path, "edges", []) or []))
    difficulty = (
        float(getattr(path, "length", 0) or 0)
        + 0.5 * max(0, candidate_count - 1)
        + 1.25 * float(has_tunneling_path)
        + 0.5 * float(target_index >= 0 and candidate_count > 3)
    )
    return PolicyStepRecord(
        sample_id=sample_id or f"runtime::{current_node.concept}::{mode}::{_stable_hash('|'.join(candidate_concepts))}",
        source_kind=source_kind,
        source_dataset=source_dataset or "runtime",
        query_type=query_type or "runtime",
        mode=mode or "forward",
        task_key=task_key or f"{source_dataset}|{query_type}|{mode}",
        current_concept=_normalize_text(getattr(current_node, "concept", "")),
        recent_concepts=recent_concepts,
        visited_concepts=visited_concepts,
        candidate_concepts=candidate_concepts,
        candidate_relations=candidate_relations,
        candidate_base_features=base_features,
        candidate_is_memory=candidate_is_memory,
        candidate_is_expanded=candidate_is_expanded,
        candidate_is_tunneling=candidate_is_tunneling,
        target_index=target_index,
        weight=float(weight),
        path_length=int(getattr(path, "length", 0) or 0),
        path_length_bucket=_path_length_bucket(int(getattr(path, "length", 0) or 0)),
        difficulty=float(difficulty),
        high_value_target=high_value_target,
        has_tunneling_path=has_tunneling_path,
        source_score=float(source_score),
        metadata={"candidate_count": candidate_count},
    )


def build_path_stub(engine: Any, concepts: Sequence[str]) -> Any:
    nodes = [engine.nodes[concept] for concept in concepts if concept in engine.nodes]
    return SimpleNamespace(length=max(0, len(concepts) - 1), nodes=nodes, edges=[])
