from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # Allows corpus tooling to run on machines without torch.
    torch = None
    nn = None
    F = None
    Dataset = object


TEMPORAL_STATES = ("current", "stable", "historical", "temporary", "superseded", "irrelevant")
LOGIC_ROLES = ("preference", "constraint", "resource", "goal", "negative", "evidence", "noise")
EVIDENCE_ROLES = (
    "direct_answer",
    "initial_value",
    "current_value",
    "updated_value",
    "supporting_context",
    "bridge_context",
    "negative_evidence",
    "noise",
)
INJECTION_MODES = ("none", "direct_fact", "profile_bridge", "temporal_state", "logic_summary")
MEMORY_LAYERS = ("event", "profile", "resource", "temporal", "path_tunnel", "topic_tunnel")
PLANNER_ARCHITECTURE = "hashed_candidate_injection_planner_v2"
PLANNER_ARCHITECTURE_SEMANTIC_V3 = "hashed_candidate_injection_planner_v3_semantic_scalar"

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_ISO_TIMESTAMP_RE = re.compile(r"\[\d{4}-\d{2}-\d{2}T[^\]]+\]")
_SESSION_SOURCE_RE = re.compile(r"\b[\w.-]+\s+session_id\s*=", re.IGNORECASE)
_SESSION_ID_RE = re.compile(r"session_id\s*=\s*[\w:./-]+", re.IGNORECASE)
_SESSION_DATE_RE = re.compile(
    r"date\s*=\s*\d{4}/\d{2}/\d{2}(?:\s*\([A-Za-z]{3,9}\))?(?:\s+\d{1,2}:\d{2})?",
    re.IGNORECASE,
)
_TURN_ROLE_RE = re.compile(
    r"\[[^\]]*?\bturn\s*=\s*(\d+)\s+role\s*=\s*([A-Za-z_]+)[^\]]*\]",
    re.IGNORECASE,
)


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("torch is required for InjectionPlannerDataset and InjectionPlannerModel")


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.strip().split())


def tokens(text: str) -> List[str]:
    return [item.lower() for item in _TOKEN_RE.findall(clean_text(text))]


def canonicalize_runtime_text(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = _ISO_TIMESTAMP_RE.sub("[timestamp]", text)
    text = _SESSION_SOURCE_RE.sub("MemoryRuntime session_id=", text)
    text = _SESSION_ID_RE.sub("session_id=session", text)
    text = _SESSION_DATE_RE.sub("date=DATE", text)
    text = _TURN_ROLE_RE.sub(lambda match: f"[turn={match.group(1)} role={match.group(2).lower()}]", text)
    return clean_text(text)


def stable_hash(value: str, *, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    digest = hashlib.blake2b(clean_text(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) % modulo


def normalize_label(value: Any, *, choices: Sequence[str], default: str) -> str:
    text = clean_text(value).lower().replace("-", "_").replace(" ", "_")
    allowed = set(choices)
    return text if text in allowed else default


def _float_feature(value: Any, *, default: float = 0.0, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if math.isnan(parsed) or math.isinf(parsed):
        parsed = default
    return max(lo, min(hi, parsed))


def vectorize_text(text: str, *, dim: int) -> List[float]:
    if dim <= 0:
        raise ValueError("dim must be positive")
    values = [0.0] * dim
    for token in tokens(text):
        index = stable_hash(token, modulo=dim)
        sign = -1.0 if stable_hash("sign:" + token, modulo=2) == 0 else 1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values))
    if norm > 0.0:
        values = [value / norm for value in values]
    return values


@dataclass(frozen=True)
class InjectionPlannerConfig:
    query_hash_dim: int = 256
    candidate_hash_dim: int = 256
    scalar_dim: int = 22
    hidden_dim: int = 192
    dropout: float = 0.10

    @property
    def input_dim(self) -> int:
        return int(self.query_hash_dim) + int(self.candidate_hash_dim) + int(self.scalar_dim)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "InjectionPlannerConfig":
        if not payload:
            return cls()
        values = dict(payload)
        legacy_text_buckets = values.get("text_buckets")
        if legacy_text_buckets is not None:
            values.setdefault("query_hash_dim", legacy_text_buckets)
            values.setdefault("candidate_hash_dim", legacy_text_buckets)
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in values.items() if key in allowed})


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _first_nonempty(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if clean_text(value):
            return value
    return ""


def normalize_candidate_row(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    candidate_id = _first_nonempty(candidate, ("id", "candidate_id", "memory_id", "event_id"))
    score = candidate.get("retrieval_score", candidate.get("score", 0.0))
    path_score = candidate.get("tunnel_score", candidate.get("path_score", 0.0))
    temporal_state = candidate.get("temporal_state", candidate.get("time_state", "irrelevant"))
    layer = candidate.get("layer", candidate.get("memory_layer", "event"))
    normalized = dict(candidate)
    normalized.update(
        {
            "id": clean_text(candidate_id),
            "text": _first_nonempty(candidate, ("text", "value", "summary", "content")),
            "summary": clean_text(candidate.get("summary", "")),
            "topic": clean_text(candidate.get("topic", candidate.get("profile_domain", ""))),
            "profile_key": clean_text(candidate.get("profile_key", candidate.get("profile_domain", ""))),
            "resource_key": clean_text(candidate.get("resource_key", "")),
            "layer": normalize_label(layer, choices=MEMORY_LAYERS, default="event"),
            "temporal_state": normalize_label(temporal_state, choices=TEMPORAL_STATES, default="irrelevant"),
            "logic_roles": [
                normalize_label(item, choices=LOGIC_ROLES, default="noise")
                for item in _as_list(candidate.get("logic_roles", candidate.get("roles", [])))
            ]
            or ["noise"],
            "evidence_role": normalize_label(
                candidate.get("evidence_role", candidate.get("evidence_role_hint", "")),
                choices=EVIDENCE_ROLES,
                default="",
            ),
            "retrieval_score": score,
            "graph_score": candidate.get("graph_score", score),
            "tunnel_score": path_score,
            "topic_similarity": candidate.get("topic_similarity", candidate.get("topic_score", 0.0)),
            "confidence": candidate.get("confidence", score),
            "rank_score": candidate.get("rank_score", candidate.get("rank_prior", 0.0)),
            "branch_depth": candidate.get("branch_depth", candidate.get("chain_depth", 0)),
            "contradicts_current": bool(
                candidate.get("contradicts_current")
                or candidate.get("superseded")
                or normalize_label(temporal_state, choices=TEMPORAL_STATES, default="irrelevant") == "superseded"
            ),
            "is_current": bool(candidate.get("is_current"))
            or normalize_label(temporal_state, choices=TEMPORAL_STATES, default="irrelevant") in {"current", "stable"},
        }
    )
    return normalized


def normalize_gold_row(gold: Mapping[str, Any] | None) -> Dict[str, Any]:
    source = dict(gold or {})
    selected = source.get("selected_candidate_ids", source.get("selected_memory_ids", []))
    temporal = source.get("temporal_state_by_candidate_id", source.get("temporal_labels", {}))
    logic = source.get("logic_roles_by_candidate_id", source.get("logic_labels", {}))
    evidence_roles = source.get("evidence_role_by_candidate_id", source.get("evidence_roles", {}))
    protected = source.get("protected_candidate_ids", source.get("protected_memory_ids", []))
    suppressed = source.get("suppressed_candidate_ids", source.get("suppressed_memory_ids", []))
    normalized = dict(source)
    normalized.update(
        {
            "selected_candidate_ids": [clean_text(item) for item in _as_list(selected) if clean_text(item)],
            "temporal_state_by_candidate_id": dict(temporal or {}),
            "logic_roles_by_candidate_id": dict(logic or {}),
            "evidence_role_by_candidate_id": dict(evidence_roles or {}),
            "protected_candidate_ids": [clean_text(item) for item in _as_list(protected) if clean_text(item)],
            "suppressed_candidate_ids": [clean_text(item) for item in _as_list(suppressed) if clean_text(item)],
            "expand_needed": bool(source.get("expand_needed", False)),
            "expand_scope": clean_text(source.get("expand_scope", "")),
            "profile_domain": clean_text(source.get("profile_domain", "")),
            "support_chain": list(source.get("support_chain", []) or []),
        }
    )
    return normalized


def normalize_training_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = row.get("candidates", row.get("retrieved_candidates", row.get("candidate_events", [])))
    normalized_candidates = [
        normalize_candidate_row(candidate)
        for candidate in list(candidates or [])
        if isinstance(candidate, Mapping)
    ]
    return {
        **dict(row),
        "id": clean_text(_first_nonempty(row, ("id", "sample_id", "row_id"))) or f"row_{stable_hash(str(row), modulo=10**12)}",
        "query": clean_text(_first_nonempty(row, ("query", "current_turn", "user_message", "question"))),
        "candidates": normalized_candidates,
        "gold": normalize_gold_row(dict(row.get("gold") or {})),
    }


def candidate_text(candidate: Mapping[str, Any]) -> str:
    pieces = [
        candidate.get("text"),
        candidate.get("summary"),
        candidate.get("topic"),
        candidate.get("profile_key"),
        candidate.get("resource_key"),
    ]
    return canonicalize_runtime_text(" ".join(clean_text(piece) for piece in pieces if clean_text(piece)))


def candidate_scalar_features(candidate: Mapping[str, Any], config: InjectionPlannerConfig | None = None) -> List[float]:
    resolved_config = config or InjectionPlannerConfig()
    layer = normalize_label(candidate.get("layer"), choices=MEMORY_LAYERS, default="")
    temporal_state = normalize_label(candidate.get("temporal_state"), choices=TEMPORAL_STATES, default="")
    overlap = _float_feature(candidate.get("query_overlap"), hi=1.0)
    retrieval_score = _float_feature(candidate.get("retrieval_score"), hi=1.0)
    graph_score = _float_feature(candidate.get("graph_score"), hi=1.0)
    tunnel_score = _float_feature(candidate.get("tunnel_score"), hi=1.0)
    topic_similarity = _float_feature(candidate.get("topic_similarity"), hi=1.0)
    confidence = _float_feature(candidate.get("confidence"), default=0.5, hi=1.0)
    rank_score = _float_feature(candidate.get("rank_score", candidate.get("rank_prior")), default=0.0, hi=1.0)
    semantic_similarity = _float_feature(
        candidate.get(
            "semantic_similarity",
            candidate.get(
                "answer_window_semantic_similarity",
                candidate.get("embedder_similarity", candidate.get("dense_similarity", 0.0)),
            ),
        ),
        default=0.0,
        hi=1.0,
    )
    age_turns = _float_feature(candidate.get("age_turns"), default=0.0, lo=0.0, hi=500.0) / 500.0
    branch_depth = _float_feature(candidate.get("branch_depth"), default=0.0, lo=0.0, hi=20.0) / 20.0
    contradiction = 1.0 if bool(candidate.get("contradicts_current")) else 0.0
    explicit_current = 1.0 if bool(candidate.get("is_current")) else 0.0
    has_profile = 1.0 if clean_text(candidate.get("profile_key")) else 0.0
    has_resource = 1.0 if clean_text(candidate.get("resource_key")) else 0.0
    layer_flags = [1.0 if layer == item else 0.0 for item in MEMORY_LAYERS]
    temporal_flags = [1.0 if temporal_state == item else 0.0 for item in ("current", "stable", "superseded")]
    features = [
        overlap,
        retrieval_score,
        graph_score,
        tunnel_score,
        topic_similarity,
        confidence,
        age_turns,
        branch_depth,
        contradiction,
        explicit_current,
        has_profile,
        has_resource,
        *layer_flags,
        *temporal_flags,
    ]
    if int(resolved_config.scalar_dim) >= 22:
        features = [
            overlap,
            retrieval_score,
            graph_score,
            tunnel_score,
            topic_similarity,
            confidence,
            rank_score,
            age_turns,
            branch_depth,
            contradiction,
            explicit_current,
            has_profile,
            has_resource,
            *layer_flags,
            *temporal_flags,
        ]
    if int(resolved_config.scalar_dim) >= 23:
        features.append(semantic_similarity)
    if len(features) != int(resolved_config.scalar_dim):
        raise AssertionError(f"scalar feature dim mismatch: {len(features)} != {resolved_config.scalar_dim}")
    return features


def _candidate_targets(candidate: Mapping[str, Any], gold: Mapping[str, Any]) -> Dict[str, Any]:
    selected_ids = {clean_text(item) for item in gold.get("selected_candidate_ids", [])}
    candidate_id = clean_text(candidate.get("id"))
    temporal_by_id = dict(gold.get("temporal_state_by_candidate_id") or {})
    logic_by_id = dict(gold.get("logic_roles_by_candidate_id") or {})
    evidence_role_by_id = dict(gold.get("evidence_role_by_candidate_id") or {})
    temporal = temporal_by_id.get(candidate_id, candidate.get("temporal_state", "irrelevant"))
    logic_values = logic_by_id.get(candidate_id, candidate.get("logic_roles", []))
    if isinstance(logic_values, str):
        logic_values = [logic_values]
    temporal_state = normalize_label(temporal, choices=TEMPORAL_STATES, default="irrelevant")
    logic_roles = [
        normalize_label(item, choices=LOGIC_ROLES, default="noise")
        for item in list(logic_values or [])
    ] or ["noise"]
    selected = candidate_id in selected_ids
    explicit_evidence_role = normalize_label(
        evidence_role_by_id.get(candidate_id, candidate.get("evidence_role", "")),
        choices=EVIDENCE_ROLES,
        default="",
    )
    if explicit_evidence_role:
        evidence_role = explicit_evidence_role
    elif not selected:
        evidence_role = "negative_evidence" if ("negative" in logic_roles or temporal_state == "superseded") else "noise"
    elif "negative" in logic_roles or temporal_state == "superseded":
        evidence_role = "negative_evidence"
    elif normalize_label(candidate.get("layer"), choices=MEMORY_LAYERS, default="event") in {"path_tunnel", "topic_tunnel"}:
        evidence_role = "bridge_context"
    elif temporal_state in {"current", "stable"}:
        evidence_role = "current_value"
    elif temporal_state == "historical":
        evidence_role = "initial_value"
    elif temporal_state == "temporary":
        evidence_role = "updated_value"
    else:
        evidence_role = "direct_answer"
    return {
        "selected": 1.0 if selected else 0.0,
        "temporal_state": temporal_state,
        "logic_roles": logic_roles,
        "evidence_role": evidence_role,
    }


def flatten_training_rows(rows: Iterable[Mapping[str, Any]], config: InjectionPlannerConfig | None = None) -> List[Dict[str, Any]]:
    resolved_config = config or InjectionPlannerConfig()
    output: List[Dict[str, Any]] = []
    for raw_row in rows:
        row = normalize_training_row(raw_row)
        query = clean_text(row.get("query") or row.get("user_message"))
        gold = dict(row.get("gold") or {})
        candidates = list(row.get("candidates") or [])
        query_vec = vectorize_text(query, dim=resolved_config.query_hash_dim)
        candidate_items = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            text = candidate_text(candidate)
            targets = _candidate_targets(candidate, gold)
            logic_target = [0.0] * len(LOGIC_ROLES)
            for role in targets["logic_roles"]:
                logic_target[LOGIC_ROLES.index(role)] = 1.0
            candidate_items.append(
                {
                    "id": clean_text(candidate.get("id")),
                    "text": text,
                    "features": query_vec
                    + vectorize_text(text, dim=resolved_config.candidate_hash_dim)
                    + candidate_scalar_features(candidate, resolved_config),
                    "selected_target": targets["selected"],
                    "temporal_target": TEMPORAL_STATES.index(targets["temporal_state"]),
                    "logic_target": logic_target,
                    "evidence_role_target": EVIDENCE_ROLES.index(targets["evidence_role"]),
                    "raw": dict(candidate),
                }
            )
        injection_mode = normalize_label(gold.get("injection_mode"), choices=INJECTION_MODES, default="none")
        output.append(
            {
                "id": clean_text(row.get("id")),
                "query": query,
                "candidates": candidate_items,
                "should_inject_target": 1.0 if bool(gold.get("should_inject")) else 0.0,
                "injection_mode_target": INJECTION_MODES.index(injection_mode),
                "raw": dict(row),
            }
        )
    return output


class InjectionPlannerDataset(Dataset):  # type: ignore[misc]
    def __init__(self, rows: Sequence[Mapping[str, Any]], config: InjectionPlannerConfig | None = None):
        _require_torch()
        self.config = config or InjectionPlannerConfig()
        self.examples = flatten_training_rows(rows, self.config)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        candidates = example["candidates"]
        features = [item["features"] for item in candidates]
        if not features:
            features = [[0.0] * self.config.input_dim]
            candidates = [
                {
                    "id": "__empty__",
                    "selected_target": 0.0,
                    "temporal_target": TEMPORAL_STATES.index("irrelevant"),
                    "logic_target": [0.0] * len(LOGIC_ROLES),
                    "evidence_role_target": EVIDENCE_ROLES.index("noise"),
                    "raw": {},
                }
            ]
        return {
            "id": example["id"],
            "features": torch.tensor(features, dtype=torch.float32),
            "valid_mask": torch.tensor([item["id"] != "__empty__" for item in candidates], dtype=torch.bool),
            "selected_target": torch.tensor([item["selected_target"] for item in candidates], dtype=torch.float32),
            "temporal_target": torch.tensor([item["temporal_target"] for item in candidates], dtype=torch.long),
            "logic_target": torch.tensor([item["logic_target"] for item in candidates], dtype=torch.float32),
            "evidence_role_target": torch.tensor([item["evidence_role_target"] for item in candidates], dtype=torch.long),
            "should_inject_target": torch.tensor(float(example["should_inject_target"]), dtype=torch.float32),
            "injection_mode_target": torch.tensor(int(example["injection_mode_target"]), dtype=torch.long),
        }


def collate_injection_batch(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    _require_torch()
    max_candidates = max(int(item["features"].shape[0]) for item in batch)
    input_dim = int(batch[0]["features"].shape[-1])
    batch_size = len(batch)
    features = torch.zeros((batch_size, max_candidates, input_dim), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    selected_target = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    temporal_target = torch.full(
        (batch_size, max_candidates),
        TEMPORAL_STATES.index("irrelevant"),
        dtype=torch.long,
    )
    logic_target = torch.zeros((batch_size, max_candidates, len(LOGIC_ROLES)), dtype=torch.float32)
    evidence_role_target = torch.full(
        (batch_size, max_candidates),
        EVIDENCE_ROLES.index("noise"),
        dtype=torch.long,
    )
    for row_index, item in enumerate(batch):
        count = int(item["features"].shape[0])
        features[row_index, :count] = item["features"]
        valid_mask[row_index, :count] = item["valid_mask"]
        selected_target[row_index, :count] = item["selected_target"]
        temporal_target[row_index, :count] = item["temporal_target"]
        logic_target[row_index, :count] = item["logic_target"]
        evidence_role_target[row_index, :count] = item["evidence_role_target"]
    return {
        "ids": [str(item["id"]) for item in batch],
        "features": features,
        "valid_mask": valid_mask,
        "selected_target": selected_target,
        "temporal_target": temporal_target,
        "logic_target": logic_target,
        "evidence_role_target": evidence_role_target,
        "should_inject_target": torch.stack([item["should_inject_target"] for item in batch]),
        "injection_mode_target": torch.stack([item["injection_mode_target"] for item in batch]),
    }


class InjectionPlannerModel(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, config: InjectionPlannerConfig | Mapping[str, Any] | None = None):
        _require_torch()
        super().__init__()
        self.config = config if isinstance(config, InjectionPlannerConfig) else InjectionPlannerConfig.from_dict(config)
        hidden = int(self.config.hidden_dim)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.config.input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(float(self.config.dropout)),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.selection_head = nn.Linear(hidden, 1)
        self.temporal_head = nn.Linear(hidden, len(TEMPORAL_STATES))
        self.logic_head = nn.Linear(hidden, len(LOGIC_ROLES))
        self.evidence_role_head = nn.Linear(hidden, len(EVIDENCE_ROLES))
        self.row_encoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Dropout(float(self.config.dropout)),
        )
        self.should_inject_head = nn.Linear(hidden, 1)
        self.injection_mode_head = nn.Linear(hidden, len(INJECTION_MODES))

    def forward(self, features: Any, valid_mask: Any | None = None) -> Dict[str, Any]:
        encoded = self.candidate_encoder(features)
        if valid_mask is None:
            valid_mask = torch.ones(encoded.shape[:2], dtype=torch.bool, device=encoded.device)
        mask_float = valid_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        masked_encoded = encoded * mask_float
        count = mask_float.sum(dim=1).clamp_min(1.0)
        mean_pool = masked_encoded.sum(dim=1) / count
        max_source = encoded.masked_fill(~valid_mask.unsqueeze(-1), -1e4)
        max_pool = torch.max(max_source, dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        row_hidden = self.row_encoder(torch.cat([mean_pool, max_pool], dim=-1))
        return {
            "candidate_hidden": encoded,
            "selection_logits": self.selection_head(encoded).squeeze(-1),
            "temporal_logits": self.temporal_head(encoded),
            "logic_logits": self.logic_head(encoded),
            "evidence_role_logits": self.evidence_role_head(encoded),
            "should_inject_logits": self.should_inject_head(row_hidden).squeeze(-1),
            "injection_mode_logits": self.injection_mode_head(row_hidden),
        }


def loss_for_batch(outputs: Mapping[str, Any], batch: Mapping[str, Any], *, weights: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    _require_torch()
    raw_weights = dict(weights or {})
    logic_pos_weight = raw_weights.pop("logic_pos_weight", None)
    temporal_class_weight = raw_weights.pop("temporal_class_weight", None)
    mode_class_weight = raw_weights.pop("mode_class_weight", None)
    resolved_weights = {
        "selection": 1.0,
        "temporal": 0.5,
        "logic": 0.5,
        "evidence_role": 0.8,
        "should_inject": 0.75,
        "mode": 0.5,
        **raw_weights,
    }
    valid_mask = batch["valid_mask"]
    valid_count = int(valid_mask.sum().item())
    if valid_count > 0:
        selection_loss = F.binary_cross_entropy_with_logits(
            outputs["selection_logits"][valid_mask],
            batch["selected_target"][valid_mask],
        )
        temporal_loss = F.cross_entropy(
            outputs["temporal_logits"][valid_mask],
            batch["temporal_target"][valid_mask],
            weight=temporal_class_weight.to(outputs["temporal_logits"].device) if hasattr(temporal_class_weight, "to") else temporal_class_weight,
        )
        logic_loss = F.binary_cross_entropy_with_logits(
            outputs["logic_logits"][valid_mask],
            batch["logic_target"][valid_mask],
            pos_weight=logic_pos_weight.to(outputs["logic_logits"].device) if hasattr(logic_pos_weight, "to") else logic_pos_weight,
        )
        evidence_role_loss = F.cross_entropy(
            outputs["evidence_role_logits"][valid_mask],
            batch["evidence_role_target"][valid_mask],
        )
    else:
        zero = outputs["should_inject_logits"].sum() * 0.0
        selection_loss = zero
        temporal_loss = zero
        logic_loss = zero
        evidence_role_loss = zero
    should_inject_loss = F.binary_cross_entropy_with_logits(
        outputs["should_inject_logits"],
        batch["should_inject_target"],
    )
    mode_loss = F.cross_entropy(
        outputs["injection_mode_logits"],
        batch["injection_mode_target"],
        weight=mode_class_weight.to(outputs["injection_mode_logits"].device) if hasattr(mode_class_weight, "to") else mode_class_weight,
    )
    total = (
        resolved_weights["selection"] * selection_loss
        + resolved_weights["temporal"] * temporal_loss
        + resolved_weights["logic"] * logic_loss
        + resolved_weights["evidence_role"] * evidence_role_loss
        + resolved_weights["should_inject"] * should_inject_loss
        + resolved_weights["mode"] * mode_loss
    )
    return {
        "loss": total,
        "selection_loss": selection_loss.detach(),
        "temporal_loss": temporal_loss.detach(),
        "logic_loss": logic_loss.detach(),
        "evidence_role_loss": evidence_role_loss.detach(),
        "should_inject_loss": should_inject_loss.detach(),
        "mode_loss": mode_loss.detach(),
    }


def summarize_candidate_plan(
    row: Mapping[str, Any],
    *,
    selected_candidate_ids: Sequence[str] | None = None,
    temporal_state_by_candidate_id: Mapping[str, str] | None = None,
    logic_roles_by_candidate_id: Mapping[str, Sequence[str]] | None = None,
    injection_mode: str | None = None,
) -> Dict[str, Any]:
    gold = dict(row.get("gold") or {})
    selected_ids = {
        clean_text(item)
        for item in (
            selected_candidate_ids
            if selected_candidate_ids is not None
            else gold.get("selected_candidate_ids", [])
        )
    }
    temporal_by_id = dict(temporal_state_by_candidate_id or gold.get("temporal_state_by_candidate_id") or {})
    logic_by_id = dict(logic_roles_by_candidate_id or gold.get("logic_roles_by_candidate_id") or {})
    evidence_role_by_id = dict(gold.get("evidence_role_by_candidate_id") or {})
    selected_candidates = []
    for candidate in list(row.get("candidates") or []):
        candidate_id = clean_text(candidate.get("id"))
        if candidate_id not in selected_ids:
            continue
        selected_candidates.append(
            {
                "id": candidate_id,
                "text": candidate_text(candidate),
                "layer": clean_text(candidate.get("layer")),
                "temporal_state": normalize_label(
                    temporal_by_id.get(candidate_id, candidate.get("temporal_state")),
                    choices=TEMPORAL_STATES,
                    default="irrelevant",
                ),
                "logic_roles": [
                    normalize_label(item, choices=LOGIC_ROLES, default="noise")
                    for item in logic_by_id.get(candidate_id, candidate.get("logic_roles", []))
                ],
                "evidence_role": normalize_label(
                    evidence_role_by_id.get(candidate_id, candidate.get("evidence_role", "")),
                    choices=EVIDENCE_ROLES,
                    default="direct_answer",
                ),
            }
        )
    mode = normalize_label(
        injection_mode if injection_mode is not None else gold.get("injection_mode"),
        choices=INJECTION_MODES,
        default="none",
    )
    text_lines = []
    for item in selected_candidates:
        roles = ",".join(role for role in item["logic_roles"] if role != "noise") or "evidence"
        text_lines.append(f"[{item['temporal_state']}|{item['evidence_role']}|{roles}] {item['text']}")
    return {
        "architecture": PLANNER_ARCHITECTURE,
        "should_inject": bool(selected_candidates) and mode != "none",
        "injection_mode": mode,
        "selected_candidates": selected_candidates,
        "inject_text": "\n".join(text_lines),
    }
