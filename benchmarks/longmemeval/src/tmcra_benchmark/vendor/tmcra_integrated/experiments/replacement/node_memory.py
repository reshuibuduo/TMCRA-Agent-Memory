from __future__ import annotations

from collections import Counter, OrderedDict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
import errno
from functools import lru_cache
import gc
import hashlib
import inspect
import itertools
import json
from json import encoder as json_encoder
import math
import multiprocessing
import os
import random
import re
import threading
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


QUESTION_HASH_BUCKETS = 4096
NODE_HASH_BUCKETS = 4096
QUESTION_EMBED_DIM = 256
NODE_EMBED_DIM = 256
NODE_TYPE_EMBED_DIM = 16
QUESTION_CATEGORICAL_EMBED_DIM = 16
EDGE_TYPE_EMBED_DIM = 16
NODE_SCALAR_DIM = 8
NODE_SCALAR_HIDDEN_DIM = 32
QUESTION_OUTPUT_DIM = 288
NODE_OUTPUT_DIM = 304
MESSAGE_HIDDEN_DIM = 448
MESSAGE_DROPOUT = 0.1
MODEL_ARCH_VERSION = "node_memory_v11_tmcra_scale49m_trimaze"
QUESTION_ENCODER_VARIANT = "hashed_tmcra_token_attn_pool_6l_w256"
NODE_ENCODER_VARIANT = "hashed_tmcra_token_attn_pool_6l_w256"
EVENT_SUBGRAPH_REFINER_VARIANT = "event_centered_subgraph_attn_6l_h448"
EVENT_DISTRACTOR_VARIANT = "candidate_distractor_delta_bce_v1"
MEMORY_TUNNEL_VARIANT = "chain_depth_tunnel_support_delta_v1"
ENCODER_STRUCTURAL_BIAS_VARIANT = "tmcra_token_role_relbias_v1"
PAIR_FEATURE_ADAPTER_VARIANT = "residual_pair_feature_adapter_v1"
ANSWER_CALIBRATION_VARIANT = "competition_alignment_v3_trimaze"
QUESTION_INTENT_VARIANT = "text_pooled_multihead_v1"
MEMORY_ROUTER_VARIANT = "question_pooled_multilabel_v1"
MESSAGE_PASSING_VARIANT = "support_to_event_bidirectional_shared_typed_v1"
ANSWER_PLAN_VARIANT = "candidate_event_answer_plan_v1"
QUESTION_MAX_TOKENS = 48
NODE_MAX_TOKENS = 32
ENCODER_ATTENTION_LAYERS = 6
ENCODER_ATTENTION_HEADS = 8
EVENT_SUBGRAPH_ATTENTION_LAYERS = 6
EVENT_SUBGRAPH_ATTENTION_HEADS = 8
TMCRA_TOKEN_ROLE_COUNT = 8
TMCRA_TOKEN_RELATION_FEATURE_DIM = 9
PAIR_FEATURE_DIM = 29
PATH_PAIR_FEATURE_DIM = 22
ANSWER_CALIBRATION_FEATURE_NAMES = (
    "event_prob_top1",
    "event_prob_margin",
    "event_prob_entropy",
    "event_prob_top3_mass",
    "path_prob_top1",
    "path_prob_margin",
    "path_prob_entropy",
    "path_prob_top3_mass",
    "temporal_prob_top1",
    "temporal_prob_margin",
    "temporal_prob_entropy",
    "temporal_prob_top3_mass",
    "event_score_top1",
    "event_score_margin",
    "path_score_top1",
    "path_score_margin",
    "temporal_score_top1",
    "temporal_score_margin",
    "is_temporal_question",
    "has_event_candidates",
    "has_path_candidates",
    "has_temporal_candidates",
    "speaker_conflict_rate",
    "time_conflict_rate",
    "temporal_distractor_rate",
    "best_exact_signature_cover",
    "best_source_turn_support",
    "best_preferred_path",
    "best_support_anchor",
    "top_event_path_support",
    "top_event_temporal_support",
    "top_path_aligns_top_event",
    "event_reverse_prob_top1",
    "event_reverse_relation_top1",
    "event_boundary_gap_top1",
    "event_reverse_available_top1",
    "path_reverse_prob_top1",
    "path_reverse_relation_top1",
    "path_boundary_gap_top1",
    "path_reverse_available_top1",
)
ANSWER_CALIBRATION_FEATURE_DIM = len(ANSWER_CALIBRATION_FEATURE_NAMES)
ANSWER_CALIBRATION_FEATURE_INDEX = {
    name: index for index, name in enumerate(ANSWER_CALIBRATION_FEATURE_NAMES)
}
EVENT_DISTRACTOR_FEATURE_DIM = 17
EVENT_RUNTIME_CALIBRATION_FEATURE_DIM = 22
PATH_RUNTIME_CALIBRATION_FEATURE_DIM = 27
EVENT_TUNNEL_FEATURE_DIM = 20
PATH_TUNNEL_FEATURE_DIM = 14
EVENT_TUNNEL_DELTA_LOGIT_LIMIT = 3.0
PATH_TUNNEL_DELTA_LOGIT_LIMIT = 3.0
FINAL_EVENT_FUSION_FEATURE_DIM = 40
FINAL_PATH_FUSION_FEATURE_DIM = 46
ANSWER_PLAN_OUTPUTS = ("selected", "current", "historical", "suppressed")
ANSWER_PLAN_OUTPUT_TO_ID = {value: index for index, value in enumerate(ANSWER_PLAN_OUTPUTS)}
ANSWER_PLAN_FEATURE_DIM = 12
DEFAULT_RUNTIME_RECALL_TOP_K = 24
DEFAULT_MATRIX_EVENT_TOP_K = 16
DEFAULT_MATRIX_EVENT_RECALL_SEED_TOP_K = 12
DEFAULT_MATRIX_EVENT_HARD_NEGATIVE_LIMIT = 4
MATRIX_EVENT_DELTA_LOGIT_LIMIT = 4.0
MATRIX_RELATION_FEATURE_DIM = 6
DEFAULT_MATRIX_PATH_TOP_K = 12
DEFAULT_MATRIX_PATH_RECALL_SEED_TOP_K = 8
DEFAULT_MATRIX_PATH_HARD_NEGATIVE_LIMIT = 4
MATRIX_PATH_DELTA_LOGIT_LIMIT = 4.0
PATH_MATRIX_RELATION_FEATURE_DIM = 6
TRI_MAZE_EVENT_RELATION_WEIGHTS = (0.32, 0.16, 0.12, 0.12, 0.08, 0.20)
TRI_MAZE_PATH_RELATION_WEIGHTS = (0.34, 0.10, 0.20, 0.10, 0.16, 0.10)
TRI_MAZE_RELATION_THRESHOLD = 0.15
GRAPH_TENSOR_CACHE_VERSION = 2
DEFAULT_BATCH_PREPARE_WORKER_MAX_TASKS_PER_CHILD = 0
DEFAULT_BATCH_PREPARE_THREAD_FALLBACK_WORKERS_MULTIPLIER = 2
DEFAULT_BATCH_PREPARE_THREAD_FALLBACK_MIN_WORKERS = 2
DEFAULT_BATCH_PREPARE_PENDING_MULTIPLIER = 3
DEFAULT_BATCH_PREPARE_PREFER_COMPLETION_ORDER = True
DEFAULT_BATCH_PREPARE_PROCESS_RECOVERY_ATTEMPTS = 3
DEFAULT_BATCH_PREPARE_PROCESS_WARMUP_PENDING_MULTIPLIER = 2
MAX_LAZY_GRAPH_PATHS_MATERIALIZE = 100000
GRAPH_TENSOR_VALUE_KEYS = frozenset(
    {
        "node_type_ids",
        "node_scalar_features",
        "edge_src",
        "edge_dst",
        "edge_type_ids",
    }
)
_GRAPH_TENSOR_CACHE_RECOVERABLE_ERRNOS = frozenset(
    int(value)
    for value in (
        getattr(errno, "EUCLEAN", None),
        117,
    )
    if isinstance(value, int)
)
_graph_tensor_cache_write_warning_lock = threading.Lock()
_graph_tensor_cache_write_warning_keys: set[tuple[str, str, int | None]] = set()


NODE_TYPES = ("speaker", "event", "time", "profile", "status", "source_turn")
EDGE_TYPES = (
    "speaker_of",
    "time_of",
    "profile_of",
    "status_of",
    "supported_by_turn",
    "same_session_next",
)
PATH_TYPES = (
    "speaker_event_time",
    "speaker_event_profile",
    "speaker_event_status",
    "speaker_event_source_turn",
)
EVENT_SUPPORT_PATH_TYPE_ORDER = PATH_TYPES
ANSWER_TYPES = ("time", "profile", "event_text", "multi_evidence", "abstain")
SEMANTIC_SLOTS = ("event", "identity", "research_topic", "education", "occupation", "event_time", "profile")
TARGET_STATUSES = ("", "past", "current", "planned")
TIME_GRANULARITIES = ("", "day", "month", "year", "relative_day_reference", "none", "day_or_coarse")
EVENT_PAIR_FEATURE_MODES = ("full", "zero")
MEMORY_ROUTER_LAYERS = ("event", "profile", "resource", "temporal", "path_tunnel", "topic_tunnel")
MEMORY_ROUTER_LAYER_TO_ID = {value: index for index, value in enumerate(MEMORY_ROUTER_LAYERS)}
DEPTH_LAYERS = ("core_view", "deep_view", "mechanism", "risk", "metric", "product_view", "evidence")
CHAIN_HINT_TOKENS = frozenset(
    {
        "chain",
        "context",
        "depth",
        "graph",
        "memory",
        "mechanism",
        "metric",
        "risk",
        "tunnel",
        "view",
    }
)

TMCRA_TOKEN_ROLE_CONTENT = 0
TMCRA_TOKEN_ROLE_ANCHOR = 1
TMCRA_TOKEN_ROLE_SPEAKER = 2
TMCRA_TOKEN_ROLE_TEMPORAL = 3
TMCRA_TOKEN_ROLE_PROFILE = 4
TMCRA_TOKEN_ROLE_STATUS = 5
TMCRA_TOKEN_ROLE_SOURCE = 6
TMCRA_TOKEN_ROLE_EMPTY = 7

NODE_TYPE_TO_ID = {value: index for index, value in enumerate(NODE_TYPES)}
EDGE_TYPE_TO_ID = {value: index for index, value in enumerate(EDGE_TYPES)}
PATH_TYPE_TO_ID = {value: index for index, value in enumerate(PATH_TYPES)}
ANSWER_TYPE_TO_ID = {value: index for index, value in enumerate(ANSWER_TYPES)}
SEMANTIC_SLOT_TO_ID = {value: index for index, value in enumerate(SEMANTIC_SLOTS)}
TARGET_STATUS_TO_ID = {value: index for index, value in enumerate(TARGET_STATUSES)}
TIME_GRANULARITY_TO_ID = {value: index for index, value in enumerate(TIME_GRANULARITIES)}
DEPTH_LAYER_TO_ID = {value: index for index, value in enumerate(DEPTH_LAYERS)}


def node_memory_arch_metadata() -> Dict[str, Any]:
    return {
        "model_arch_version": MODEL_ARCH_VERSION,
        "question_encoder_variant": QUESTION_ENCODER_VARIANT,
        "node_encoder_variant": NODE_ENCODER_VARIANT,
        "event_subgraph_refiner_variant": EVENT_SUBGRAPH_REFINER_VARIANT,
        "event_distractor_variant": EVENT_DISTRACTOR_VARIANT,
        "memory_tunnel_variant": MEMORY_TUNNEL_VARIANT,
        "encoder_structural_bias_variant": ENCODER_STRUCTURAL_BIAS_VARIANT,
        "pair_feature_adapter_variant": PAIR_FEATURE_ADAPTER_VARIANT,
        "answer_calibration_variant": ANSWER_CALIBRATION_VARIANT,
        "question_intent_variant": QUESTION_INTENT_VARIANT,
        "memory_router_variant": MEMORY_ROUTER_VARIANT,
        "message_passing_variant": MESSAGE_PASSING_VARIANT,
        "answer_plan_variant": ANSWER_PLAN_VARIANT,
        "question_max_tokens": QUESTION_MAX_TOKENS,
        "node_max_tokens": NODE_MAX_TOKENS,
        "encoder_attention_layers": ENCODER_ATTENTION_LAYERS,
        "encoder_attention_heads": ENCODER_ATTENTION_HEADS,
        "event_subgraph_attention_layers": EVENT_SUBGRAPH_ATTENTION_LAYERS,
        "event_subgraph_attention_heads": EVENT_SUBGRAPH_ATTENTION_HEADS,
    }


def extract_node_memory_checkpoint_arch(payload: Mapping[str, Any]) -> Dict[str, Any]:
    config = dict(payload.get("config", {}) or {})
    metadata = dict(payload.get("metadata", {}) or {})
    extracted: Dict[str, Any] = {}
    for key, default_value in node_memory_arch_metadata().items():
        if key in config:
            value = config.get(key)
        elif key in metadata:
            value = metadata.get(key)
        else:
            value = default_value if key != "model_arch_version" else ""
        if value is None or value == "":
            value = default_value if key != "model_arch_version" else ""
        extracted[key] = value
    return extracted


def validate_node_memory_checkpoint_payload(payload: Mapping[str, Any], *, context: str) -> Dict[str, Any]:
    arch = extract_node_memory_checkpoint_arch(payload)
    found_version = clean_text(arch.get("model_arch_version", ""))
    if found_version != MODEL_ARCH_VERSION:
        found_label = found_version or "<missing>"
        raise ValueError(
            f"{context}: incompatible checkpoint architecture; expected "
            f"model_arch_version='{MODEL_ARCH_VERSION}', found '{found_label}'. "
            "Old architecture checkpoints can be archived, but cannot be resumed or loaded into the new scorer."
        )
    return arch


DEFAULT_TRAINING_CONFIG: Dict[str, Any] = {
    "epochs": 12,
    "batch_size": 16,
    "max_train_steps": 0,
    "lr": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "grad_clip": 1.0,
    "early_stopping_patience": 3,
    "log_every_steps": 100,
    "checkpoint_every_steps": 0,
    "amp": True,
    "graph_prefetch_lookahead_batches": 3,
    "batch_prepare_workers": 0,
    "batch_prepare_lookahead_batches": 0,
    "recall_loss_weight": 0.6,
    "event_loss_weight": 1.1,
    "path_loss_weight": 0.55,
    "temporal_loss_weight": 0.3,
    "answer_type_loss_weight": 0.2,
    "answer_plan_loss_weight": 0.18,
    "token_role_loss_weight": 0.08,
    "question_understanding_loss_weight": 0.12,
    "memory_router_loss_weight": 0.18,
    "event_distractor_loss_weight": 0.15,
    "event_tunnel_loss_weight": 0.12,
    "path_tunnel_loss_weight": 0.06,
    "path_tunnel_delta_loss_weight": 0.12,
    "event_tunnel_selection_loss_weight": 0.08,
    "path_tunnel_selection_loss_weight": 0.04,
    "event_hard_negative_loss_weight": 0.35,
    "path_hard_negative_loss_weight": 0.25,
    "recall_selection_loss_weight": 0.3,
    "event_selection_loss_weight": 0.35,
    "path_selection_loss_weight": 0.3,
    "final_event_set_loss_weight": 0.35,
    "event_matrix_delta_loss_weight": 0.08,
    "path_matrix_delta_loss_weight": 0.08,
    "event_matrix_delta_margin": 0.15,
    "path_matrix_delta_margin": 0.12,
    "event_tunnel_margin": 0.16,
    "path_tunnel_margin": 0.16,
    "online_event_hard_negative_limit": 6,
    "online_path_hard_negative_limit": 8,
    "trainable_stage": "all",
    "event_hard_negative_margin": 0.35,
    "path_hard_negative_margin": 0.28,
    "recall_selection_margin": 0.12,
    "event_selection_margin": 0.2,
    "path_selection_margin": 0.15,
    "recall_selection_top_k": DEFAULT_RUNTIME_RECALL_TOP_K,
    "event_selection_top_k": 5,
    "path_selection_top_k": 3,
    "recall_selection_positive_coverage_count": 3,
    "event_selection_positive_coverage_count": 3,
    "path_selection_positive_coverage_count": 2,
    "final_event_set_margin": 0.08,
    "final_event_set_top_k": 10,
    "final_event_set_support_path_k": 3,
    "final_event_set_positive_coverage_count": 3,
    "multi_positive_coverage_fraction": 0.6,
    "multi_positive_recall_coverage_count": 6,
    "multi_positive_event_coverage_count": 5,
    "multi_positive_path_coverage_count": 3,
    "multi_positive_final_event_set_coverage_count": 5,
    "answer_refusal_loss_weight": 0.08,
    "answer_refusal_margin": 0.15,
    "answer_plan_selection_margin": 0.12,
    "answer_plan_selection_top_k": 5,
    "answer_plan_current_old_margin_loss_weight": 0.0,
    "answer_plan_current_old_margin": 0.2,
    "answer_plan_selected_negative_margin_loss_weight": 0.0,
    "answer_plan_selected_negative_margin": 0.3,
    "train_sampling_mode": "source_aware_balanced",
    "sampling_source_alpha": 0.35,
    "sampling_blend_uniform_ratio": 0.35,
    "sampling_time_boost": 1.2,
    "sampling_multi_evidence_boost": 1.45,
    "sampling_temporal_positive_boost": 1.15,
    "sampling_max_conversation_multiplier": 2.0,
    "sampling_max_group_repeat": 2,
    "loss_source_alpha": 0.4,
    "loss_blend_uniform_ratio": 0.25,
    "loss_weight_power": 0.5,
    "loss_time_boost": 1.45,
    "loss_multi_evidence_boost": 1.75,
    "loss_temporal_positive_boost": 1.2,
    "loss_min_example_weight": 0.65,
    "loss_max_example_weight": 1.85,
    "loss_group_balancing_mode": "supervision_bucket",
    "l2sp_loss_weight": 0.0,
    **node_memory_arch_metadata(),
}

_METRIC_WEIGHT_KEYS: Dict[str, str] = {
    "loss": "samples",
    "recall_loss": "recall_loss_count",
    "event_loss": "event_loss_count",
    "path_loss": "path_loss_count",
    "temporal_loss": "temporal_loss_count",
    "answer_type_loss": "answer_type_loss_count",
    "answer_plan_loss": "answer_plan_loss_count",
    "token_role_loss": "token_role_loss_count",
    "question_understanding_loss": "question_understanding_loss_count",
    "memory_router_loss": "memory_router_loss_count",
    "memory_router_exact_match": "memory_router_total",
    "memory_router_f1": "memory_router_total",
    "event_distractor_loss": "event_distractor_loss_count",
    "event_tunnel_loss": "event_tunnel_loss_count",
    "path_tunnel_loss": "path_tunnel_loss_count",
    "path_tunnel_delta_loss": "path_tunnel_delta_loss_count",
    "event_tunnel_selection_loss": "event_tunnel_selection_loss_count",
    "path_tunnel_selection_loss": "path_tunnel_selection_loss_count",
    "event_hard_negative_loss": "event_hard_negative_loss_count",
    "path_hard_negative_loss": "path_hard_negative_loss_count",
    "recall_selection_loss": "recall_selection_loss_count",
    "event_selection_loss": "event_selection_loss_count",
    "path_selection_loss": "path_selection_loss_count",
    "final_event_set_loss": "final_event_set_loss_count",
    "event_matrix_delta_loss": "event_matrix_delta_loss_count",
    "path_matrix_delta_loss": "path_matrix_delta_loss_count",
    "answer_refusal_loss": "answer_refusal_loss_count",
    "recall_event_recall_at_24": "recall_event_recall_total",
    "recall_event_positive_coverage_at_24": "recall_event24_positive_total",
    "event_recall_at_1": "event_recall_total",
    "event_recall_at_5": "event_recall_total",
    "event_positive_coverage_at_5": "event5_positive_total",
    "path_recall_at_3": "path_recall_total",
    "path_positive_coverage_at_3": "path3_positive_total",
    "path_tunnel_support_recall_at_3": "path_tunnel_support_recall_total",
    "path_tunnel_support_positive_coverage_at_3": "path_tunnel_support_positive_total",
    "path_tunnel_delta_recall_at_3": "path_tunnel_delta_recall_total",
    "path_tunnel_delta_positive_coverage_at_3": "path_tunnel_delta_positive_total",
    "answer_plan_selected_recall_at_5": "answer_plan_selected_total",
    "answer_plan_selected_positive_coverage_at_5": "answer_plan_selected_positive_total",
    "answer_plan_current_top1_accuracy": "answer_plan_current_total",
    "path_tunnel_rescue025_recall_at_3": "path_tunnel_rescue025_recall_total",
    "path_tunnel_rescue050_recall_at_3": "path_tunnel_rescue050_recall_total",
    "path_tunnel_rescue100_recall_at_3": "path_tunnel_rescue100_recall_total",
    "temporal_accuracy": "temporal_total",
}
_METRIC_COUNT_KEYS = {"samples", "graph_error_count", "loss_group_count", *set(_METRIC_WEIGHT_KEYS.values())}


_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_MONTH_MARKERS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_NON_SPEAKER_CAPITALIZED_WORDS = {
    "What",
    "When",
    "Which",
    "Where",
    "Who",
    "Why",
    "How",
    "Today",
    "Tomorrow",
    "Yesterday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
}
_QUESTION_SPEAKER_PREPOSITIONS = {
    "at",
    "about",
    "around",
    "by",
    "during",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "near",
    "of",
    "on",
    "over",
    "through",
    "to",
    "under",
    "with",
}
_QUESTION_SPEAKER_AUXILIARIES = (
    "did",
    "does",
    "do",
    "is",
    "was",
    "were",
    "has",
    "have",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
)
_WEEKDAY_MARKERS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "mon",
    "tue",
    "wed",
    "thu",
    "fri",
    "sat",
    "sun",
}
_TMCRA_TEMPORAL_TOKEN_HINTS = {
    "today",
    "tomorrow",
    "yesterday",
    "tonight",
    "morning",
    "afternoon",
    "evening",
    "night",
    "week",
    "weekend",
    "month",
    "year",
    "day",
    *_MONTH_MARKERS,
    *_WEEKDAY_MARKERS,
}
_QUESTION_EDGE_GLUE_TOKENS = {
    "a",
    "an",
    "the",
    "to",
    "as",
    "at",
    "in",
    "on",
    "of",
    "for",
    "with",
    "from",
    "about",
    "around",
    "into",
    "over",
    "under",
}
_QUESTION_LEADING_FUNCTION_RE = re.compile(
    rf"^\s*(?:what|when|which|where|who|why|how)\b(?:\s+(?:{'|'.join(_QUESTION_SPEAKER_AUXILIARIES)}))?",
    re.IGNORECASE,
)
_TEMPORAL_REFERENCE_RE = re.compile(
    r"\b(?:when|date|day|month|year|today|tomorrow|yesterday|tonight|morning|afternoon|evening|night|week(?:end)?|last|next|before|after)\b",
    re.IGNORECASE,
)
_TEMPORAL_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:when|since\s+when)\b", re.IGNORECASE)
_TEMPORAL_ANSWER_WH_RE = re.compile(r"^\s*(?:what|which)\s+(?:date|day|month|year|time)\b", re.IGNORECASE)
_TEMPORAL_ANSWER_DURATION_RE = re.compile(
    r"^\s*how\s+(?:long|many\s+(?:days?|weeks?|months?|years?))\b",
    re.IGNORECASE,
)
_EDUCATION_YEAR_RE = re.compile(
    r"\b(?:year|semester|grade)\s+(?:in|of|at)\s+(?:\w+\s+){0,3}(?:school|college|university|uni)\b",
    re.IGNORECASE,
)
_PROFILE_COPULAR_RE = re.compile(r"^\s*who\s+is\b", re.IGNORECASE)
_PROFILE_ATTRIBUTE_RE = re.compile(r"^\s*(?:what|where)\s+(?:does|do|is|are|was|were|has|have|had)\b", re.IGNORECASE)
_PROFILE_QUERY_HINT_RE = re.compile(
    r"\b(?:occupation|job|work(?:ing)?|study|studying|research|major(?:ing)?|profession|career|school|college|university)\b",
    re.IGNORECASE,
)
_PROFILE_POSSESSIVE_RE = re.compile(r"\b[A-Za-z][A-Za-z'-]+(?:\s+[A-Za-z][A-Za-z'-]+)?'s\b", re.IGNORECASE)
_PLANNED_STATUS_RE = re.compile(
    r"\b(?:will|going\s+to|plan(?:ning)?\s+to|scheduled\s+to|hoping\s+to|preparing\s+to)\b",
    re.IGNORECASE,
)
_CURRENT_STATUS_RE = re.compile(r"\b(?:currently|now|still)\b|\b(?:is|are|am)\s+\w+ing\b", re.IGNORECASE)
_PAST_STATUS_RE = re.compile(r"\b(?:did|was|were|had|has)\b", re.IGNORECASE)


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(text.split())


def _env_flag(name: str, *, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    normalized = clean_text(raw).lower()
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    return bool(default)


def normalize_text(value: Any) -> str:
    return clean_text(value).lower()


@lru_cache(maxsize=32768)
def _tokenize_normalized_text_cached(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    english = tuple(_TOKEN_RE.findall(text))
    cjk = tuple(char for char in text if "\u4e00" <= char <= "\u9fff")
    if english or cjk:
        return tuple([*english, *cjk])
    return tuple(char for char in text if char.strip())


@lru_cache(maxsize=32768)
def _normalized_token_set_cached(text: str) -> frozenset[str]:
    return frozenset(_tokenize_normalized_text_cached(text))


@lru_cache(maxsize=32768)
def _normalized_token_bigram_set_cached(text: str) -> frozenset[tuple[str, str]]:
    tokens = _tokenize_normalized_text_cached(text)
    if len(tokens) < 2:
        return frozenset()
    return frozenset((tokens[index], tokens[index + 1]) for index in range(len(tokens) - 1))


def tokenize_text(value: Any) -> List[str]:
    text = normalize_text(value)
    if not text:
        return []
    return list(_tokenize_normalized_text_cached(text))


def dedupe_texts(values: Iterable[Any], *, max_items: int | None = None) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        key = normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if max_items is not None and len(result) >= max_items:
            break
    return result


class GraphTensorValidationError(RuntimeError):
    pass


def _format_debug_context(context: Mapping[str, Any] | None = None) -> str:
    if not context:
        return ""
    parts: List[str] = []
    for key, value in context.items():
        normalized_key = clean_text(key)
        if not normalized_key:
            continue
        if isinstance(value, (list, tuple)):
            preview = [clean_text(item) for item in list(value)[:8]]
            suffix = "" if len(value) <= 8 else "..."
            rendered = f"[{', '.join(item for item in preview if item)}{suffix}]"
        else:
            rendered = clean_text(value)
        parts.append(f"{normalized_key}={rendered}")
    return ", ".join(part for part in parts if part)


def _raise_index_validation_error(
    index_name: str,
    *,
    context: str,
    invalid_value: int,
    lower_bound: int,
    upper_bound: int,
    extra: Mapping[str, Any] | None = None,
) -> None:
    context_payload = _format_debug_context(extra)
    suffix = f" ({context_payload})" if context_payload else ""
    raise GraphTensorValidationError(
        f"{context}: {index_name}={int(invalid_value)} outside [{int(lower_bound)}, {int(upper_bound)}){suffix}"
    )


def _validate_index_values(
    values: Sequence[int],
    *,
    upper_bound: int,
    index_name: str,
    context: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    resolved_upper_bound = int(upper_bound)
    if resolved_upper_bound <= 0:
        if values:
            _raise_index_validation_error(
                index_name,
                context=context,
                invalid_value=int(values[0]),
                lower_bound=0,
                upper_bound=resolved_upper_bound,
                extra=extra,
            )
        return
    for value in values:
        resolved_value = int(value)
        if 0 <= resolved_value < resolved_upper_bound:
            continue
        _raise_index_validation_error(
            index_name,
            context=context,
            invalid_value=resolved_value,
            lower_bound=0,
            upper_bound=resolved_upper_bound,
            extra=extra,
        )


def _normalize_event_rerank_mode(value: Any) -> str:
    normalized = normalize_text(value)
    return "single" if normalized == "single" else "matrix"


def _normalize_event_pair_feature_mode(value: Any) -> str:
    normalized = normalize_text(value)
    return "zero" if normalized == "zero" else "full"


def _apply_event_pair_feature_mode(features: Tensor, mode: str) -> Tensor:
    resolved_mode = _normalize_event_pair_feature_mode(mode)
    if resolved_mode == "zero" and isinstance(features, Tensor) and features.numel() > 0:
        return torch.zeros_like(features)
    return features


def _call_with_supported_kwargs(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return func(*args, **kwargs)
    supported_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return func(*args, **supported_kwargs)


def _extract_question_speakers(question: str) -> List[str]:
    clean = clean_text(question)
    if not clean:
        return []

    candidates: List[str] = []
    seen = set()

    def add_candidate(raw_value: str) -> None:
        candidate = clean_text(raw_value).removesuffix("'s")
        if not candidate:
            return
        parts = [part for part in candidate.split() if clean_text(part)]
        if not parts or any(part in _NON_SPEAKER_CAPITALIZED_WORDS for part in parts):
            return
        key = normalize_text(candidate)
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    name_pattern = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
    anchored_patterns = (
        rf"\b({name_pattern})'s\b",
        rf"\b(?:{'|'.join(_QUESTION_SPEAKER_AUXILIARIES)})\s+({name_pattern})\b",
        rf"^\s*(?:What|When|Which|Where|Why|How)\s+({name_pattern})\b",
    )
    for pattern in anchored_patterns:
        for match in re.finditer(pattern, clean):
            add_candidate(match.group(1))

    for match in re.finditer(rf"\b({name_pattern})\b", clean):
        prefix = clean[: match.start()].rstrip()
        previous_word_match = re.search(r"([A-Za-z]+)$", prefix)
        previous_word = normalize_text(previous_word_match.group(1)) if previous_word_match else ""
        if previous_word in _QUESTION_SPEAKER_PREPOSITIONS:
            continue
        add_candidate(match.group(1))
    return candidates


def _question_has_subject_reference(question: str, speaker_candidates: Sequence[str]) -> bool:
    if bool(list(speaker_candidates or [])) or bool(_PROFILE_POSSESSIVE_RE.search(question)):
        return True
    clean = clean_text(question)
    if not clean:
        return False
    content_text = clean
    prefix_match = _QUESTION_LEADING_FUNCTION_RE.match(content_text)
    if prefix_match:
        content_text = content_text[prefix_match.end() :]
    content_tokens = _trim_question_anchor_edge_tokens(_normalized_token_list(content_text))
    if not content_tokens:
        return False
    first_token = normalize_text(content_tokens[0])
    if not first_token or first_token in _TMCRA_TEMPORAL_TOKEN_HINTS:
        return False
    return True


def _question_requests_profile_detail(question: str, lowered: str, speaker_candidates: Sequence[str]) -> bool:
    has_subject_reference = _question_has_subject_reference(question, speaker_candidates)
    if not has_subject_reference:
        return False
    if _PROFILE_COPULAR_RE.search(question):
        return True
    if _PROFILE_QUERY_HINT_RE.search(lowered):
        return True
    return bool(_PROFILE_ATTRIBUTE_RE.search(question) and _PROFILE_QUERY_HINT_RE.search(lowered))


def _question_has_temporal_reference(question: str, lowered: str) -> bool:
    first_token = normalize_text(tokenize_text(question)[0]) if tokenize_text(question) else ""
    if first_token == "when":
        return True
    if _TEMPORAL_REFERENCE_RE.search(lowered):
        return True
    if any(marker in lowered for marker in _MONTH_MARKERS):
        return True
    if re.search(r"\b(?:mon|tue|wed|thu|fri|sat|sun)(?:day)?\b", lowered):
        return True
    if re.search(r"\b\d{4}\b", lowered):
        return True
    return False


def _question_requests_temporal_answer(question: str, lowered: str) -> bool:
    clean = clean_text(question)
    if not clean:
        return False
    if _EDUCATION_YEAR_RE.search(clean):
        return False
    if _TEMPORAL_ANSWER_PREFIX_RE.search(clean):
        return True
    if _TEMPORAL_ANSWER_WH_RE.search(clean):
        return True
    if _TEMPORAL_ANSWER_DURATION_RE.search(clean):
        return True
    return False


def _question_is_temporal(question: str, lowered: str) -> bool:
    return _question_requests_temporal_answer(question, lowered)


def _question_time_granularity(question: str, lowered: str) -> str:
    if not _question_requests_temporal_answer(question, lowered):
        return ""
    if "month" in lowered or any(marker in lowered for marker in _MONTH_MARKERS):
        return "month"
    if "year" in lowered or re.search(r"\b\d{4}\b", lowered):
        return "year"
    return "day_or_coarse"


def _question_target_status(question: str, lowered: str) -> str:
    if _PLANNED_STATUS_RE.search(question):
        return "planned"
    if _CURRENT_STATUS_RE.search(question):
        return "current"
    if _PAST_STATUS_RE.search(lowered):
        return "past"
    return ""


def _question_semantic_slot(question: str, lowered: str, speaker_candidates: Sequence[str]) -> str:
    if _question_is_temporal(question, lowered):
        return "event_time"
    if _question_requests_profile_detail(question, lowered, speaker_candidates):
        return "profile"
    return "event"


def _positive_event_payloads_have_profile(payloads: Sequence[Mapping[str, Any]]) -> bool:
    for payload in payloads:
        metadata = dict(payload.get("metadata", {}) or {})
        if any(
            clean_text(
                payload.get(key, metadata.get(key, ""))
            )
            for key in ("profile_type", "profile_value")
        ):
            return True
    return False


def stable_hash_int(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def stable_hash_bucket(value: str, buckets: int) -> int:
    return stable_hash_int(value or "__empty__") % max(1, int(buckets))


def stable_token_identity(value: str) -> int:
    return stable_hash_int(value or "__empty__") & ((1 << 63) - 1)


def _text_hash_indices(value: Any, buckets: int) -> List[int]:
    tokens = tokenize_text(value) or ["__empty__"]
    return [stable_hash_bucket(token, buckets) for token in tokens]


def _resolved_text_tokens(value: Any) -> List[str]:
    return tokenize_text(value) or ["__empty__"]


def _align_sequence_length(values: Sequence[Any], *, target_length: int, fill_value: Any) -> List[Any]:
    resolved_target_length = max(0, int(target_length))
    if resolved_target_length <= 0:
        return []
    normalized = list(values[:resolved_target_length])
    if len(normalized) < resolved_target_length:
        normalized.extend([fill_value] * (resolved_target_length - len(normalized)))
    return normalized


def _padded_long_batch(
    value_batches: Sequence[Sequence[int]],
    *,
    device: torch.device,
    fill_value: int = 0,
    max_tokens: int | None = None,
) -> tuple[Tensor, Tensor]:
    if not value_batches:
        return (
            torch.full((0, 1), int(fill_value), dtype=torch.long, device=device),
            torch.zeros((0, 1), dtype=torch.bool, device=device),
        )
    max_allowed_tokens = max(1, int(max_tokens or 0)) if max_tokens is not None else None
    normalized_batches: List[List[int]] = []
    max_length = 1
    for values in value_batches:
        normalized = [int(item) for item in list(values or [])]
        if max_allowed_tokens is not None:
            normalized = normalized[:max_allowed_tokens]
        if not normalized:
            normalized = [int(fill_value)]
        normalized_batches.append(normalized)
        max_length = max(max_length, len(normalized))
    padded = torch.full((len(normalized_batches), max_length), int(fill_value), dtype=torch.long, device=device)
    valid_mask = torch.zeros((len(normalized_batches), max_length), dtype=torch.bool, device=device)
    for batch_index, values in enumerate(normalized_batches):
        value_count = len(values)
        padded[batch_index, :value_count] = torch.tensor(values, dtype=torch.long, device=device)
        valid_mask[batch_index, :value_count] = True
    return padded, valid_mask


def _padded_hashed_index_batch(
    index_batches: Sequence[Sequence[int]],
    *,
    embedding: nn.Embedding,
    max_tokens: int | None = None,
) -> tuple[Tensor, Tensor]:
    device = embedding.weight.device
    empty_index = stable_hash_bucket("__empty__", int(embedding.num_embeddings))
    if not index_batches:
        return (
            torch.full((0, 1), empty_index, dtype=torch.long, device=device),
            torch.zeros((0, 1), dtype=torch.bool, device=device),
        )
    max_allowed_tokens = max(1, int(max_tokens or 0)) if max_tokens is not None else None
    normalized_batches: List[List[int]] = []
    max_length = 1
    for index_values in index_batches:
        values = [int(item) for item in list(index_values or [])]
        if max_allowed_tokens is not None:
            values = values[:max_allowed_tokens]
        if not values:
            values = [empty_index]
        normalized_batches.append(values)
        max_length = max(max_length, len(values))
    batch_size = len(normalized_batches)
    padded_indices = torch.full((batch_size, max_length), empty_index, dtype=torch.long, device=device)
    valid_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
    for batch_index, values in enumerate(normalized_batches):
        value_count = len(values)
        _validate_index_values(
            values,
            upper_bound=int(embedding.num_embeddings),
            index_name="hashed_index",
            context="_padded_hashed_index_batch",
            extra={"batch_index": batch_index, "value_count": value_count},
        )
        padded_indices[batch_index, :value_count] = torch.tensor(values, dtype=torch.long, device=device)
        valid_mask[batch_index, :value_count] = True
    return padded_indices, valid_mask


def _hashed_token_embedding_batch(
    index_batches: Sequence[Sequence[int]],
    embedding: nn.Embedding,
    *,
    max_tokens: int | None = None,
) -> tuple[Tensor, Tensor]:
    padded_indices, valid_mask = _padded_hashed_index_batch(
        index_batches,
        embedding=embedding,
        max_tokens=max_tokens,
    )
    return embedding(padded_indices), valid_mask


def _hashed_index_embedding_batch(index_batches: Sequence[Sequence[int]], embedding: nn.Embedding) -> Tensor:
    device = embedding.weight.device
    if not index_batches:
        return torch.zeros((0, embedding.embedding_dim), dtype=embedding.weight.dtype, device=device)
    embedded, valid_mask = _hashed_token_embedding_batch(index_batches, embedding)
    valid_mask_float = valid_mask.unsqueeze(-1).to(dtype=embedded.dtype)
    pooled = (embedded * valid_mask_float).sum(dim=1)
    denom = valid_mask_float.sum(dim=1).clamp_min(1.0)
    return pooled / denom


def _question_token_roles(question: str, question_features: Mapping[str, Any]) -> tuple[List[str], List[int]]:
    prepared = _prepare_question_scoring_features(question, question_features)
    tokens = _resolved_text_tokens(question)
    anchor_tokens = set(prepared.get("question_anchor_tokens", []) or [])
    speaker_tokens = set(prepared.get("speaker_candidates", set()) or set())
    semantic_target = clean_text(prepared.get("semantic_target", ""))
    has_status_target = bool(clean_text(prepared.get("target_status_target", "")))
    temporal_mode = bool(prepared.get("question_is_temporal", False)) or bool(clean_text(prepared.get("time_target", "")))
    profile_mode = semantic_target in {"identity", "research_topic", "education", "occupation", "profile"}
    roles: List[int] = []
    for token in tokens:
        if token == "__empty__":
            roles.append(TMCRA_TOKEN_ROLE_EMPTY)
        elif token in speaker_tokens:
            roles.append(TMCRA_TOKEN_ROLE_SPEAKER)
        elif temporal_mode and (token in anchor_tokens or token in _TMCRA_TEMPORAL_TOKEN_HINTS):
            roles.append(TMCRA_TOKEN_ROLE_TEMPORAL)
        elif profile_mode and token in anchor_tokens:
            roles.append(TMCRA_TOKEN_ROLE_PROFILE)
        elif has_status_target and token in anchor_tokens:
            roles.append(TMCRA_TOKEN_ROLE_STATUS)
        elif token in anchor_tokens:
            roles.append(TMCRA_TOKEN_ROLE_ANCHOR)
        else:
            roles.append(TMCRA_TOKEN_ROLE_CONTENT)
    return tokens, roles


def _node_type_token_role(node_type: str) -> int:
    normalized_type = clean_text(node_type)
    if normalized_type == "speaker":
        return TMCRA_TOKEN_ROLE_SPEAKER
    if normalized_type == "time":
        return TMCRA_TOKEN_ROLE_TEMPORAL
    if normalized_type == "profile":
        return TMCRA_TOKEN_ROLE_PROFILE
    if normalized_type == "status":
        return TMCRA_TOKEN_ROLE_STATUS
    if normalized_type == "source_turn":
        return TMCRA_TOKEN_ROLE_SOURCE
    return TMCRA_TOKEN_ROLE_CONTENT


def _node_token_roles(text: str, node_type_id: int) -> tuple[List[str], List[int]]:
    tokens = _resolved_text_tokens(text)
    node_type = NODE_TYPES[int(node_type_id)] if 0 <= int(node_type_id) < len(NODE_TYPES) else ""
    default_role = _node_type_token_role(node_type)
    roles: List[int] = []
    for token in tokens:
        if token == "__empty__":
            roles.append(TMCRA_TOKEN_ROLE_EMPTY)
        elif token in {"past", "current", "planned"}:
            roles.append(TMCRA_TOKEN_ROLE_STATUS)
        elif token in _TMCRA_TEMPORAL_TOKEN_HINTS:
            roles.append(TMCRA_TOKEN_ROLE_TEMPORAL)
        else:
            roles.append(default_role)
    return tokens, roles


def _tmcra_text_relation_features(
    token_id_batches: Sequence[Sequence[int]],
    token_role_batches: Sequence[Sequence[int]],
    *,
    max_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    role_ids, token_mask = _padded_long_batch(
        token_role_batches,
        device=device,
        fill_value=TMCRA_TOKEN_ROLE_EMPTY,
        max_tokens=max_tokens,
    )
    token_ids, _ = _padded_long_batch(
        token_id_batches,
        device=device,
        fill_value=0,
        max_tokens=max_tokens,
    )
    features = _tmcra_text_relation_features_from_tensors(
        token_ids,
        role_ids,
        token_mask=token_mask,
        dtype=dtype,
    )
    return features, token_mask


def _tmcra_text_relation_features_from_tensors(
    token_ids: Tensor,
    role_ids: Tensor,
    *,
    token_mask: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    batch_size, token_count = role_ids.shape
    device = token_ids.device
    valid_pair = token_mask.unsqueeze(1) & token_mask.unsqueeze(2)
    left_roles = role_ids.unsqueeze(2)
    right_roles = role_ids.unsqueeze(1)
    non_content_left = (left_roles != TMCRA_TOKEN_ROLE_CONTENT) & (left_roles != TMCRA_TOKEN_ROLE_EMPTY)
    non_content_right = (right_roles != TMCRA_TOKEN_ROLE_CONTENT) & (right_roles != TMCRA_TOKEN_ROLE_EMPTY)
    position_ids = torch.arange(token_count, dtype=torch.long, device=device)
    position_distance = (position_ids.view(1, token_count, 1) - position_ids.view(1, 1, token_count)).abs()
    features = torch.stack(
        [
            (token_ids.unsqueeze(2) == token_ids.unsqueeze(1)) & valid_pair,
            (left_roles == right_roles) & non_content_left & non_content_right & valid_pair,
            (left_roles == TMCRA_TOKEN_ROLE_ANCHOR) & (right_roles == TMCRA_TOKEN_ROLE_ANCHOR) & valid_pair,
            (left_roles == TMCRA_TOKEN_ROLE_SPEAKER) & (right_roles == TMCRA_TOKEN_ROLE_SPEAKER) & valid_pair,
            (left_roles == TMCRA_TOKEN_ROLE_TEMPORAL) & (right_roles == TMCRA_TOKEN_ROLE_TEMPORAL) & valid_pair,
            (left_roles == TMCRA_TOKEN_ROLE_PROFILE) & (right_roles == TMCRA_TOKEN_ROLE_PROFILE) & valid_pair,
            (left_roles == TMCRA_TOKEN_ROLE_STATUS) & (right_roles == TMCRA_TOKEN_ROLE_STATUS) & valid_pair,
            (left_roles == TMCRA_TOKEN_ROLE_SOURCE) & (right_roles == TMCRA_TOKEN_ROLE_SOURCE) & valid_pair,
            (position_distance <= 1) & valid_pair,
        ],
        dim=-1,
    ).to(dtype=dtype)
    return features


def _move_structure_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_structure_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_structure_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_structure_to_device(item, device) for item in value)
    return value


def _tensor_to_device(value: Tensor, device: torch.device, *, non_blocking: bool = False) -> Tensor:
    if value.device == device:
        return value
    return value.to(device=device, non_blocking=bool(non_blocking))


def _pin_tensor_memory(value: Tensor) -> Tensor:
    if value.device.type != "cpu" or not torch.cuda.is_available():
        return value
    try:
        if value.is_pinned():
            return value
    except RuntimeError:
        return value
    try:
        return value.pin_memory()
    except RuntimeError:
        return value


def _graph_tensor_mapping_to_device(
    graph_tensors: Mapping[str, Any],
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> Dict[str, Any]:
    moved = dict(graph_tensors)
    for key in GRAPH_TENSOR_VALUE_KEYS:
        value = moved.get(key)
        if isinstance(value, Tensor):
            moved[key] = _tensor_to_device(value, device, non_blocking=non_blocking)
    return moved


def _pin_graph_tensor_mapping(graph_tensors: Mapping[str, Any]) -> Dict[str, Any]:
    pinned = dict(graph_tensors)
    for key in GRAPH_TENSOR_VALUE_KEYS:
        value = pinned.get(key)
        if isinstance(value, Tensor):
            pinned[key] = _pin_tensor_memory(value)
    return pinned


def _graph_source_signature(path: Path) -> Dict[str, int]:
    stats = path.stat()
    return {
        "size": int(stats.st_size),
        "mtime_ns": int(stats.st_mtime_ns),
        "cache_version": int(GRAPH_TENSOR_CACHE_VERSION),
    }


def _graph_tensor_cache_path(cache_dir: Path, graph_path: Path) -> Path:
    return cache_dir / f"{graph_path.stem}.pt"


def _load_graph_tensor_cache(
    cache_path: Path,
    *,
    source_signature: Mapping[str, Any],
) -> Dict[str, Any] | None:
    def _invalidate_cache_file() -> None:
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            pass

    if not cache_path.exists():
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except Exception:
        _invalidate_cache_file()
        return None
    if not isinstance(payload, Mapping):
        _invalidate_cache_file()
        return None
    cached_signature = dict(payload.get("source_signature", {}) or {})
    if cached_signature != dict(source_signature):
        _invalidate_cache_file()
        return None
    graph = payload.get("graph")
    tensors = payload.get("tensors")
    if not isinstance(graph, Mapping) or not isinstance(tensors, Mapping):
        _invalidate_cache_file()
        return None
    return {"graph": dict(graph), "tensors": dict(tensors)}


def _write_graph_tensor_cache(
    cache_path: Path,
    *,
    source_signature: Mapping[str, Any],
    graph: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> bool:
    def _report_cache_write_failure(error: Exception) -> None:
        error_no = getattr(error, "errno", None)
        warning_key = (str(cache_path.parent), type(error).__name__, error_no if isinstance(error_no, int) else None)
        with _graph_tensor_cache_write_warning_lock:
            if warning_key in _graph_tensor_cache_write_warning_keys:
                return
            _graph_tensor_cache_write_warning_keys.add(warning_key)
        details = f" errno={error_no}" if isinstance(error_no, int) else ""
        print(
            f"[node_memory] graph tensor cache write skipped for {cache_path}{details}: {error}",
            flush=True,
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=str(cache_path.parent),
            prefix=f"{cache_path.stem}.",
            suffix=".tmp",
        ) as handle:
            temp_path = Path(handle.name)
            torch.save(
                {
                    "source_signature": dict(source_signature),
                    "graph": dict(graph),
                    "tensors": dict(tensors),
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temp_path, cache_path)
            temp_path = None
            return True
        except OSError as error:
            error_no = getattr(error, "errno", None)
            if isinstance(error_no, int) and error_no in _GRAPH_TENSOR_CACHE_RECOVERABLE_ERRNOS:
                try:
                    cache_path.unlink(missing_ok=True)
                except OSError:
                    pass
                try:
                    os.replace(temp_path, cache_path)
                    temp_path = None
                    return True
                except Exception as retry_error:
                    _report_cache_write_failure(retry_error)
                    return False
            _report_cache_write_failure(error)
            return False
    except Exception as error:
        _report_cache_write_failure(error)
        return False
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


class _PurePythonJSONEncoder(json.JSONEncoder):
    def encode(self, o: Any) -> str:
        chunks = self.iterencode(o, _one_shot=False)
        if not isinstance(chunks, (list, tuple)):
            chunks = list(chunks)
        return "".join(chunks)

    def iterencode(self, o: Any, _one_shot: bool = False):  # type: ignore[override]
        if self.check_circular:
            markers: dict[int, Any] | None = {}
        else:
            markers = None
        if self.ensure_ascii:
            string_encoder = json_encoder.encode_basestring_ascii
        else:
            string_encoder = json_encoder.encode_basestring

        def floatstr(
            value: float,
            allow_nan: bool = self.allow_nan,
            _repr: Callable[[float], str] = float.__repr__,
            _inf: float = float("inf"),
            _neginf: float = -float("inf"),
        ) -> str:
            if value != value:
                text = "NaN"
            elif value == _inf:
                text = "Infinity"
            elif value == _neginf:
                text = "-Infinity"
            else:
                return _repr(value)
            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " + repr(value)
                )
            return text

        _iterencode = json_encoder._make_iterencode(
            markers,
            self.default,
            string_encoder,
            self.indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            _one_shot=False,
        )
        return _iterencode(o, 0)

def json_dumps(payload: Any) -> str:
    # Reusing a single pure-Python encoder instance has shown state corruption on
    # Python 3.12 during repeated large graph exports. Build a fresh encoder per
    # call so markers and iterencode internals never leak across payloads.
    return _PurePythonJSONEncoder(ensure_ascii=False, sort_keys=True).encode(payload)


def read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Failed to decode JSON file {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse JSON file {path}: line {exc.lineno} column {exc.colno} char {exc.pos}: {exc.msg}"
        ) from exc


def iter_jsonl_with_offsets(path: Path) -> Iterator[tuple[int, Dict[str, Any]]]:
    if not path.exists():
        return
    try:
        with path.open("rb") as handle:
            for file_line_number in itertools.count(start=1):
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    decoded_line = line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise RuntimeError(
                        f"Failed to decode JSONL file {path} at source line {file_line_number}: {exc}"
                    ) from exc
                try:
                    yield offset, dict(json.loads(decoded_line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Failed to parse JSONL file {path} at source line {file_line_number}: "
                        f"line {exc.lineno} column {exc.colno} char {exc.pos}: {exc.msg}"
                    ) from exc
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Failed to decode JSONL file {path}: {exc}") from exc


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    for _, row in iter_jsonl_with_offsets(path):
        yield row


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [row for row in iter_jsonl(path)]


def write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json_dumps(payload))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "\n".join(json_dumps(dict(row)) for row in rows)
    _atomic_write_text(path, text + ("\n" if text else ""))


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def answer_type_from_query(
    question: str,
    *,
    category: Any = "",
    positive_event_ids: Sequence[str] | None = None,
    question_features: Mapping[str, Any] | None = None,
    positive_event_payloads: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    positives = [clean_text(item) for item in list(positive_event_ids or []) if clean_text(item)]
    if str(category or "").strip() == "5" or not positives:
        return "abstain"
    resolved_question_features = dict(question_features or extract_question_features(question))
    if len(positives) > 1:
        return "multi_evidence"
    if bool(resolved_question_features.get("is_temporal", False)):
        return "time"
    if _positive_event_payloads_have_profile(list(positive_event_payloads or [])):
        return "profile"
    if clean_text(resolved_question_features.get("semantic_slot_target", "")) == "profile":
        return "profile"
    return "event_text"


def extract_question_features(question: str) -> Dict[str, Any]:
    clean = clean_text(question)
    lowered = normalize_text(clean)
    speaker_candidates = _extract_question_speakers(clean)
    semantic_slot = _question_semantic_slot(clean, lowered, speaker_candidates)
    target_status = _question_target_status(clean, lowered)
    time_granularity = _question_time_granularity(clean, lowered)
    is_temporal = _question_is_temporal(clean, lowered)
    has_temporal_reference = _question_has_temporal_reference(clean, lowered)
    question_anchor_tokens = _question_anchor_tokens(
        clean,
        {"speaker_candidates": list(speaker_candidates)},
    )
    return {
        "speaker_candidates": list(speaker_candidates),
        "semantic_slot_target": semantic_slot,
        "target_status_target": target_status,
        "time_granularity_target": time_granularity,
        "is_temporal": is_temporal,
        "has_temporal_reference": has_temporal_reference,
        "question_anchor_tokens": list(question_anchor_tokens),
    }


def _token_overlap_ratio(left: Sequence[str] | str, right: Sequence[str] | str) -> float:
    left_tokens = (
        _normalized_token_set_cached(normalize_text(left))
        if isinstance(left, str)
        else {clean_text(item) for item in left if clean_text(item)}
    )
    right_tokens = (
        _normalized_token_set_cached(normalize_text(right))
        if isinstance(right, str)
        else {clean_text(item) for item in right if clean_text(item)}
    )
    if not left_tokens or not right_tokens:
        return 0.0
    return float(len(left_tokens & right_tokens)) / float(max(1, len(left_tokens | right_tokens)))


def _max_token_overlap_ratio(left: Sequence[str] | str, values: Sequence[Any]) -> float:
    best = 0.0
    for value in values:
        best = max(best, _token_overlap_ratio(left, clean_text(value)))
    return best


def _normalized_speaker_candidates(features: Mapping[str, Any]) -> set[str]:
    return {
        normalize_text(clean_text(item).removesuffix("'s"))
        for item in list(features.get("speaker_candidates", []) or [])
        if clean_text(item)
    }


def _normalized_required_tokens(required_tokens: Sequence[str]) -> List[str]:
    normalized_required: List[str] = []
    for token in required_tokens:
        normalized = normalize_text(token)
        if not normalized:
            normalized = normalize_text(clean_text(token))
        if normalized:
            normalized_required.append(normalized)
    return normalized_required


def _normalized_token_list(value: Sequence[str] | str) -> List[str]:
    if isinstance(value, str):
        return list(_tokenize_normalized_text_cached(normalize_text(value)))
    tokens: List[str] = []
    for item in value:
        for token in tokenize_text(item):
            normalized = normalize_text(token)
            if normalized:
                tokens.append(normalized)
    return tokens


def _normalized_token_set(value: Sequence[str] | str) -> set[str]:
    if isinstance(value, str):
        return set(_normalized_token_set_cached(normalize_text(value)))
    return set(_normalized_token_list(value))


def _strip_question_speaker_mentions(text: str, speaker_candidates: Sequence[str]) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    for candidate in sorted(
        (clean_text(item) for item in list(speaker_candidates or []) if clean_text(item)),
        key=len,
        reverse=True,
    ):
        cleaned = re.sub(rf"\b{re.escape(candidate)}\b", " ", cleaned, flags=re.IGNORECASE)
    return clean_text(cleaned)


def _trim_question_anchor_edge_tokens(tokens: Sequence[str]) -> List[str]:
    trimmed = [normalize_text(clean_text(token)) for token in list(tokens or []) if clean_text(token)]
    while trimmed and trimmed[0] in _QUESTION_EDGE_GLUE_TOKENS:
        trimmed.pop(0)
    while trimmed and trimmed[-1] in _QUESTION_EDGE_GLUE_TOKENS:
        trimmed.pop()
    return trimmed


def _question_anchor_tokens(question: str, question_features: Mapping[str, Any]) -> List[str]:
    existing = [
        normalize_text(clean_text(token))
        for token in list(question_features.get("question_anchor_tokens", []) or [])
        if clean_text(token)
    ]
    if existing:
        return dedupe_texts(existing)
    clean = clean_text(question)
    if not clean:
        return []
    speaker_tokens = _normalized_token_set(list(question_features.get("speaker_candidates", []) or []))
    content_text = clean
    prefix_match = _QUESTION_LEADING_FUNCTION_RE.match(content_text)
    if prefix_match:
        content_text = content_text[prefix_match.end() :]
    content_text = _strip_question_speaker_mentions(
        content_text,
        list(question_features.get("speaker_candidates", []) or []),
    )
    if bool(question_features.get("is_temporal", False)):
        content_text = _TEMPORAL_REFERENCE_RE.sub(" ", content_text)
        content_text = re.sub(r"\b\d{1,4}\b", " ", content_text)
    target_status = clean_text(question_features.get("target_status_target", ""))
    if target_status == "planned":
        content_text = _PLANNED_STATUS_RE.sub(" ", content_text)
    elif target_status == "current":
        content_text = _CURRENT_STATUS_RE.sub(" ", content_text)
    elif target_status == "past":
        content_text = _PAST_STATUS_RE.sub(" ", content_text)
    anchors = _trim_question_anchor_edge_tokens(
        [
            token
            for token in _normalized_token_list(content_text)
            if token and token not in speaker_tokens and not re.fullmatch(r"\d+", token)
        ]
    )
    if anchors:
        return dedupe_texts(anchors)
    fallback_tokens = _trim_question_anchor_edge_tokens(
        [
            token
            for token in _normalized_token_list(clean)
            if token and token not in speaker_tokens and not re.fullmatch(r"\d+", token)
        ]
    )
    return dedupe_texts(fallback_tokens)


def _speaker_candidates_from_predicted_roles(tokens: Sequence[str], role_ids: Sequence[int], token_mask: Sequence[bool]) -> List[str]:
    candidates: List[str] = []
    current_tokens: List[str] = []
    for token, role_id, is_valid in zip(tokens, role_ids, token_mask):
        normalized_token = normalize_text(token)
        if not is_valid or not normalized_token or normalized_token == "__empty__":
            break
        if int(role_id) == TMCRA_TOKEN_ROLE_SPEAKER:
            current_tokens.append(normalized_token)
            continue
        if current_tokens:
            candidates.append(" ".join(current_tokens))
            current_tokens = []
    if current_tokens:
        candidates.append(" ".join(current_tokens))
    return dedupe_texts(candidates)


def _anchor_tokens_from_predicted_roles(tokens: Sequence[str], role_ids: Sequence[int], token_mask: Sequence[bool]) -> List[str]:
    anchor_like_roles = {
        TMCRA_TOKEN_ROLE_ANCHOR,
        TMCRA_TOKEN_ROLE_PROFILE,
        TMCRA_TOKEN_ROLE_STATUS,
    }
    anchors = [
        normalize_text(token)
        for token, role_id, is_valid in zip(tokens, role_ids, token_mask)
        if is_valid
        and int(role_id) in anchor_like_roles
        and normalize_text(token)
        and normalize_text(token) != "__empty__"
        and not re.fullmatch(r"\d+", normalize_text(token))
    ]
    return dedupe_texts(_trim_question_anchor_edge_tokens(anchors))


def _merge_learned_question_features_from_token_roles(
    question: str,
    base_features: Mapping[str, Any],
    *,
    tokens: Sequence[str],
    predicted_role_ids: Sequence[int],
    token_mask: Sequence[bool],
) -> Dict[str, Any]:
    merged = dict(base_features or {})
    learned_speakers = _speaker_candidates_from_predicted_roles(tokens, predicted_role_ids, token_mask)
    learned_anchors = _anchor_tokens_from_predicted_roles(tokens, predicted_role_ids, token_mask)
    if learned_speakers:
        merged["speaker_candidates"] = learned_speakers
    if learned_anchors:
        merged["question_anchor_tokens"] = learned_anchors
    return merged


def _required_token_coverage(required_tokens: Sequence[str], candidate: Sequence[str] | str) -> float:
    normalized_required = _normalized_required_tokens(required_tokens)
    if not normalized_required:
        return 0.0
    candidate_tokens = (
        _normalized_token_set_cached(normalize_text(candidate))
        if isinstance(candidate, str)
        else _normalized_token_set(candidate)
    )
    if not candidate_tokens:
        return 0.0
    covered = 0
    for token in normalized_required:
        if token in candidate_tokens:
            covered += 1
    return float(covered) / float(len(normalized_required))


def _max_required_token_coverage(required_tokens: Sequence[str], values: Sequence[Any]) -> float:
    best = 0.0
    for value in values:
        coverage = _required_token_coverage(required_tokens, value)
        if coverage > best:
            best = coverage
    return best


def _required_bigram_coverage(required_tokens: Sequence[str], candidate: Sequence[str] | str) -> float:
    normalized_required = _normalized_required_tokens(required_tokens)
    if len(normalized_required) < 2:
        return 0.0
    required_bigrams = set()
    for index in range(len(normalized_required) - 1):
        required_bigrams.add((normalized_required[index], normalized_required[index + 1]))
    if isinstance(candidate, str):
        candidate_bigrams = _normalized_token_bigram_set_cached(normalize_text(candidate))
    else:
        candidate_tokens = _normalized_token_list(candidate)
        if len(candidate_tokens) < 2:
            return 0.0
        candidate_bigrams = set()
        for index in range(len(candidate_tokens) - 1):
            candidate_bigrams.add((candidate_tokens[index], candidate_tokens[index + 1]))
    if not required_bigrams:
        return 0.0
    return float(len(required_bigrams & candidate_bigrams)) / float(len(required_bigrams))


def _text_feature_payload(value: Any) -> Dict[str, Any]:
    text = clean_text(value)
    normalized = normalize_text(text)
    token_set = set(_normalized_token_set_cached(normalized)) if normalized else set()
    bigram_set = set(_normalized_token_bigram_set_cached(normalized)) if normalized else set()
    return {
        "text": text,
        "normalized": normalized,
        "token_set": token_set,
        "bigram_set": bigram_set,
    }


def _text_feature_payloads(values: Sequence[Any]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for value in values:
        payload = _text_feature_payload(value)
        if payload["text"]:
            payloads.append(payload)
    return payloads


def _token_overlap_ratio_from_token_sets(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    return float(len(left_tokens & right_tokens)) / float(max(1, len(left_tokens | right_tokens)))


def _max_token_overlap_ratio_from_payloads(left_tokens: set[str], payloads: Sequence[Mapping[str, Any]]) -> float:
    best = 0.0
    for payload in payloads:
        best = max(best, _token_overlap_ratio_from_token_sets(left_tokens, set(payload.get("token_set", set()))))
    return best


def _required_bigram_set(required_tokens: Sequence[str]) -> set[tuple[str, str]]:
    normalized_required = _normalized_required_tokens(required_tokens)
    return _required_bigram_set_from_normalized_tokens(normalized_required)


def _required_bigram_set_from_normalized_tokens(normalized_required: Sequence[str]) -> set[tuple[str, str]]:
    if len(normalized_required) < 2:
        return set()
    return {
        (normalized_required[index], normalized_required[index + 1])
        for index in range(len(normalized_required) - 1)
    }


def _required_token_coverage_from_tokens(required_tokens: Sequence[str], candidate_tokens: set[str]) -> float:
    normalized_required = _normalized_required_tokens(required_tokens)
    return _required_token_coverage_from_normalized_tokens(normalized_required, candidate_tokens)


def _required_token_coverage_from_normalized_tokens(
    normalized_required: Sequence[str],
    candidate_tokens: set[str],
) -> float:
    if not normalized_required or not candidate_tokens:
        return 0.0
    covered = 0
    for token in normalized_required:
        if token in candidate_tokens:
            covered += 1
    return float(covered) / float(len(normalized_required))


def _max_required_token_coverage_from_payloads(required_tokens: Sequence[str], payloads: Sequence[Mapping[str, Any]]) -> float:
    normalized_required = _normalized_required_tokens(required_tokens)
    return _max_required_token_coverage_from_normalized_tokens(normalized_required, payloads)


def _max_required_token_coverage_from_normalized_tokens(
    normalized_required: Sequence[str],
    payloads: Sequence[Mapping[str, Any]],
) -> float:
    best = 0.0
    for payload in payloads:
        coverage = _required_token_coverage_from_normalized_tokens(
            normalized_required,
            set(payload.get("token_set", set())),
        )
        if coverage > best:
            best = coverage
    return best


def _required_bigram_coverage_from_bigrams(
    required_bigrams: set[tuple[str, str]],
    candidate_bigrams: set[tuple[str, str]],
) -> float:
    if not required_bigrams or not candidate_bigrams:
        return 0.0
    return float(len(required_bigrams & candidate_bigrams)) / float(len(required_bigrams))


def _max_required_bigram_coverage_from_payloads(
    required_bigrams: set[tuple[str, str]],
    payloads: Sequence[Mapping[str, Any]],
) -> float:
    best = 0.0
    for payload in payloads:
        coverage = _required_bigram_coverage_from_bigrams(
            required_bigrams,
            set(payload.get("bigram_set", set())),
        )
        if coverage > best:
            best = coverage
    return best


def _prepare_question_scoring_features(question: str, question_features: Mapping[str, Any]) -> Dict[str, Any]:
    question_text = clean_text(question)
    question_payload = _text_feature_payload(question_text)
    question_anchor_tokens = _normalized_required_tokens(_question_anchor_tokens(question_text, question_features))
    return {
        "question_text": question_text,
        "question_token_set": set(question_payload.get("token_set", set())),
        "speaker_candidates": _normalized_speaker_candidates(question_features),
        "semantic_target": clean_text(question_features.get("semantic_slot_target", "")),
        "target_status_target": clean_text(question_features.get("target_status_target", "")),
        "time_target": clean_text(question_features.get("time_granularity_target", "")),
        "question_is_temporal": bool(question_features.get("is_temporal", False)),
        "question_anchor_tokens": question_anchor_tokens,
        "question_anchor_bigrams": _required_bigram_set_from_normalized_tokens(question_anchor_tokens),
    }


def _prepare_event_scoring_features(
    event_node: Mapping[str, Any],
    *,
    support_payload: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = dict(event_node.get("metadata", {}) or {})
    support = dict(support_payload or {})
    depth_layer = clean_text(
        event_node.get(
            "depth_layer",
            metadata.get(
                "depth_layer",
                metadata.get("memory_chain_depth_layer", metadata.get("chain_depth_layer", "")),
            ),
        )
    )
    time_display_value = clean_text(event_node.get("time_display_value", metadata.get("time_display_value", "")))
    time_value = clean_text(event_node.get("time_value", metadata.get("time_value", "")))
    event_text_payload = _text_feature_payload(event_node.get("text", ""))
    typed_signature_text = clean_text(metadata.get("tmcra_typed_event_signature", ""))
    typed_signature_text = " ".join(
        dedupe_texts(
            [
                typed_signature_text,
                " ".join(str(item) for item in list(metadata.get("tmcra_node_tags", []) or [])),
                " ".join(str(item) for item in list(metadata.get("tmcra_path_tags", []) or [])),
                clean_text(metadata.get("tmcra_tunnel_group_key", "")),
            ],
            max_items=8,
        )
    )
    event_signature_payload = _text_feature_payload(
        " ".join(
            item
            for item in (
                clean_text(event_node.get("event_signature", metadata.get("event_signature", ""))),
                typed_signature_text,
            )
            if item
        )
    )
    profile_value_payload = _text_feature_payload(event_node.get("profile_value", metadata.get("profile_value", "")))
    time_payload = _text_feature_payload(time_display_value or time_value)
    source_turn_payloads = _text_feature_payloads(list(support.get("source_turn_texts", []) or []))
    profile_text_payloads = _text_feature_payloads(list(support.get("profile_texts", []) or []))
    time_text_payloads = _text_feature_payloads(list(support.get("time_texts", []) or []))
    return {
        "speaker": normalize_text(clean_text(event_node.get("speaker", metadata.get("speaker", ""))).removesuffix("'s")),
        "target_status": clean_text(event_node.get("target_status", metadata.get("target_status", ""))),
        "time_granularity": clean_text(event_node.get("time_granularity", metadata.get("time_granularity", ""))),
        "profile_type": clean_text(event_node.get("profile_type", metadata.get("profile_type", ""))),
        "profile_value": profile_value_payload,
        "event_signature": event_signature_payload,
        "event_text": event_text_payload,
        "time_text": time_payload,
        "time_compare_value": normalize_text(time_value or time_display_value),
        "session_name": normalize_text(clean_text(event_node.get("session_name", metadata.get("session_name", "")))),
        "turn_index": int(event_node.get("turn_index", metadata.get("turn_index", 0)) or 0),
        "depth_layer": depth_layer,
        "memory_chain_role": clean_text(
            event_node.get(
                "memory_chain_role",
                metadata.get("memory_chain_role", metadata.get("chain_role", metadata.get("support_role", ""))),
            )
        ),
        "source_turn_texts": source_turn_payloads,
        "profile_texts": profile_text_payloads,
        "time_texts": time_text_payloads,
        "has_time": 1.0 if time_display_value or time_value else 0.0,
        "has_profile": 1.0 if clean_text(event_node.get("profile_type", metadata.get("profile_type", ""))) or profile_value_payload["text"] else 0.0,
        "has_status": 1.0 if clean_text(event_node.get("target_status", metadata.get("target_status", ""))) else 0.0,
        "relation_signature_tokens": (
            set(event_signature_payload.get("token_set", set()))
            or set(event_text_payload.get("token_set", set()))
        ),
    }


def _prepare_node_scoring_features(node: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = dict(node.get("metadata", {}) or {})
    time_display_value = clean_text(node.get("time_display_value", metadata.get("time_display_value", "")))
    time_value = clean_text(node.get("time_value", metadata.get("time_value", "")))
    profile_value = clean_text(node.get("profile_value", metadata.get("profile_value", "")))
    text_payload = _text_feature_payload(node.get("text", ""))
    time_payload = _text_feature_payload(time_display_value or time_value)
    profile_type_payload = _text_feature_payload(node.get("profile_type", metadata.get("profile_type", "")))
    return {
        "type": clean_text(node.get("type", "")),
        "text": text_payload,
        "time_granularity": clean_text(node.get("time_granularity", metadata.get("time_granularity", ""))),
        "time_text": time_payload,
        "time_compare_value": clean_text(time_value or time_display_value),
        "profile_type": clean_text(node.get("profile_type", metadata.get("profile_type", ""))),
        "profile_type_payload": profile_type_payload,
        "profile_value": profile_value,
        "status_or_text": clean_text(node.get("target_status", metadata.get("target_status", ""))) or text_payload["text"],
    }


def _prepare_path_scoring_features(
    path: Mapping[str, Any],
    *,
    node_scoring_features_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    support_node_id = _path_support_node_id(path)
    support_features = dict(node_scoring_features_by_id.get(support_node_id, {}) or {})
    path_metadata = dict(path.get("metadata", {}) or {})
    path_tag_payload = _text_feature_payload(
        " ".join(
            item
            for item in (
                " ".join(str(tag) for tag in list(path.get("tmcra_path_tags", path_metadata.get("tmcra_path_tags", [])) or [])),
                clean_text(path_metadata.get("tmcra_tunnel_group_key", "")),
            )
            if clean_text(item)
        )
    )
    support_text_tokens = set(dict(support_features.get("text", {}) or {}).get("token_set", set()))
    support_text_tokens.update(set(path_tag_payload.get("token_set", set())))
    return {
        "id": clean_text(path.get("id", "")),
        "event_id": clean_text(path.get("event_id", "")),
        "path_type": clean_text(path.get("_path_type", path.get("type", ""))),
        "support_node_id": support_node_id,
        "support_type": clean_text(support_features.get("type", "")),
        "support_time_value": clean_text(support_features.get("time_compare_value", "")),
        "support_profile": clean_text(support_features.get("profile_value", "")),
        "support_text_tokens": support_text_tokens,
        "path_tag_tokens": set(path_tag_payload.get("token_set", set())),
    }


def _ensure_graph_scoring_feature_cache(graph_tensors: Mapping[str, Any]) -> Mapping[str, Any]:
    if (
        "event_scoring_features_by_id" in graph_tensors
        and "node_scoring_features_by_id" in graph_tensors
        and "path_scoring_features_by_id" in graph_tensors
    ):
        return graph_tensors
    resolved_graph_tensors = graph_tensors if isinstance(graph_tensors, dict) else dict(graph_tensors)
    node_lookup = {
        clean_text(node_id): dict(node or {})
        for node_id, node in dict(resolved_graph_tensors.get("node_by_id", {}) or {}).items()
        if clean_text(node_id)
    }
    event_support_lookup = {
        clean_text(event_id): dict(payload or {})
        for event_id, payload in dict(resolved_graph_tensors.get("event_support_lookup", {}) or {}).items()
        if clean_text(event_id)
    }
    node_scoring_features_by_id = {
        node_id: _prepare_node_scoring_features(node)
        for node_id, node in node_lookup.items()
    }
    event_scoring_features_by_id = {
        event_id: _prepare_event_scoring_features(
            node_lookup.get(event_id, {}),
            support_payload=event_support_lookup.get(event_id, {}),
        )
        for event_id in [
            clean_text(item)
            for item in list(resolved_graph_tensors.get("event_node_ids", []) or [])
            if clean_text(item)
        ]
    }
    path_scoring_features_by_id = {
        clean_text(path.get("id", "")): _prepare_path_scoring_features(
            path,
            node_scoring_features_by_id=node_scoring_features_by_id,
        )
        for path in list(resolved_graph_tensors.get("paths", []) or [])
        if clean_text(path.get("id", ""))
    }
    resolved_graph_tensors["node_scoring_features_by_id"] = node_scoring_features_by_id
    resolved_graph_tensors["event_scoring_features_by_id"] = event_scoring_features_by_id
    resolved_graph_tensors["path_scoring_features_by_id"] = path_scoring_features_by_id
    return resolved_graph_tensors


@contextmanager
def _gc_suspended() -> Any:
    gc_enabled = gc.isenabled()
    if gc_enabled:
        gc.disable()
    try:
        yield
    finally:
        if gc_enabled:
            gc.enable()


def _time_granularity_compatible(target: Any, value: Any) -> bool:
    normalized_target = clean_text(target)
    normalized_value = clean_text(value)
    if not normalized_target:
        return True
    if not normalized_value:
        return False
    if normalized_target == "day_or_coarse":
        return normalized_value in {"day", "relative_day_reference", "month", "year"}
    return normalized_target == normalized_value


def _event_pair_feature_values(
    question: str,
    question_features: Mapping[str, Any],
    event_node: Mapping[str, Any],
    *,
    support_payload: Mapping[str, Any] | None = None,
    prepared_question: Mapping[str, Any] | None = None,
    prepared_event: Mapping[str, Any] | None = None,
) -> List[float]:
    question_cache = dict(prepared_question or _prepare_question_scoring_features(question, question_features))
    event_cache = dict(prepared_event or _prepare_event_scoring_features(event_node, support_payload=support_payload))
    question_token_set = set(question_cache.get("question_token_set", set()))
    question_anchor_tokens = list(question_cache.get("question_anchor_tokens", []) or [])
    question_anchor_bigrams = set(question_cache.get("question_anchor_bigrams", set()))
    speaker_candidates = set(question_cache.get("speaker_candidates", set()))
    question_is_temporal = bool(question_cache.get("question_is_temporal", False))
    semantic_target = clean_text(question_cache.get("semantic_target", ""))
    target_status_target = clean_text(question_cache.get("target_status_target", ""))
    time_target = clean_text(question_cache.get("time_target", ""))
    event_speaker = normalize_text(clean_text(event_cache.get("speaker", "")))
    event_speaker_tokens = _normalized_token_set(event_speaker)
    event_status = clean_text(event_cache.get("target_status", ""))
    event_time_granularity = clean_text(event_cache.get("time_granularity", ""))
    profile_type = clean_text(event_cache.get("profile_type", ""))
    profile_value_payload = dict(event_cache.get("profile_value", {}) or {})
    event_signature_payload = dict(event_cache.get("event_signature", {}) or {})
    time_payload = dict(event_cache.get("time_text", {}) or {})
    event_text_payload = dict(event_cache.get("event_text", {}) or {})
    source_turn_payloads = [dict(item or {}) for item in list(event_cache.get("source_turn_texts", []) or [])]
    profile_text_payloads = [dict(item or {}) for item in list(event_cache.get("profile_texts", []) or [])]
    time_text_payloads = [dict(item or {}) for item in list(event_cache.get("time_texts", []) or [])]
    has_time = float(event_cache.get("has_time", 0.0) or 0.0)
    has_profile = float(event_cache.get("has_profile", 0.0) or 0.0)
    has_status = float(event_cache.get("has_status", 0.0) or 0.0)
    semantic_match = 0.0
    if semantic_target == "event_time":
        semantic_match = has_time
    elif semantic_target in {"identity", "research_topic", "education", "occupation"}:
        semantic_match = 1.0 if profile_type == semantic_target else 0.0
    elif semantic_target == "profile":
        semantic_match = has_profile
    time_match = 1.0 if time_target and _time_granularity_compatible(time_target, event_time_granularity) else 0.0
    time_conflict = 1.0 if time_target and event_time_granularity and not _time_granularity_compatible(time_target, event_time_granularity) else 0.0
    status_match = 1.0 if target_status_target and event_status and target_status_target == event_status else 0.0
    status_conflict = 1.0 if target_status_target and event_status and target_status_target != event_status else 0.0
    speaker_token_overlap = _required_token_coverage_from_normalized_tokens(
        list(event_speaker_tokens),
        question_token_set,
    )
    explicit_speaker_match = 1.0 if speaker_candidates and event_speaker and event_speaker in speaker_candidates else 0.0
    speaker_match = max(explicit_speaker_match, speaker_token_overlap)
    speaker_conflict = 1.0 if (
        speaker_candidates
        and event_speaker
        and event_speaker not in speaker_candidates
        and speaker_token_overlap <= 0.0
    ) else 0.0
    source_turn_overlap = _max_token_overlap_ratio_from_payloads(question_token_set, source_turn_payloads)
    content_overlap = max(
        _token_overlap_ratio_from_token_sets(question_token_set, set(event_text_payload.get("token_set", set()))),
        _token_overlap_ratio_from_token_sets(question_token_set, set(event_signature_payload.get("token_set", set()))),
        source_turn_overlap,
        _max_token_overlap_ratio_from_payloads(question_token_set, profile_text_payloads),
        _max_token_overlap_ratio_from_payloads(question_token_set, time_text_payloads),
    )
    profile_conflict = 1.0 if semantic_target in {"identity", "research_topic", "education", "occupation"} and profile_type and profile_type != semantic_target else 0.0
    profile_missing = 1.0 if semantic_target in {"identity", "research_topic", "education", "occupation", "profile"} and not has_profile else 0.0
    time_missing = 1.0 if semantic_target == "event_time" and not has_time else 0.0
    source_turn_present = 1.0 if source_turn_payloads else 0.0
    signature_anchor_coverage = _required_token_coverage_from_normalized_tokens(
        question_anchor_tokens,
        set(event_signature_payload.get("token_set", set())),
    )
    signature_anchor_bigram_coverage = _required_bigram_coverage_from_bigrams(
        question_anchor_bigrams,
        set(event_signature_payload.get("bigram_set", set())),
    )
    source_anchor_coverage = _max_required_token_coverage_from_normalized_tokens(question_anchor_tokens, source_turn_payloads)
    event_anchor_coverage = max(
        signature_anchor_coverage,
        _required_token_coverage_from_normalized_tokens(question_anchor_tokens, set(event_text_payload.get("token_set", set()))),
        source_anchor_coverage,
    )
    exact_signature_anchor_cover = 1.0 if question_anchor_tokens and signature_anchor_coverage >= 0.999 else 0.0
    same_speaker_day_past_distractor = 1.0 if (
        question_is_temporal
        and speaker_match > 0.5
        and event_status == "past"
        and event_time_granularity in {"day", "relative_day_reference"}
        and event_anchor_coverage < 0.5
    ) else 0.0
    temporal_anchor_miss = 1.0 if (
        question_is_temporal
        and (has_time > 0.5 or bool(event_time_granularity))
        and event_anchor_coverage <= 0.0
    ) else 0.0
    return [
        max(1.0 if speaker_candidates else 0.0, speaker_token_overlap),
        speaker_match,
        semantic_match,
        status_match,
        status_conflict,
        time_match,
        time_conflict,
        _token_overlap_ratio_from_token_sets(question_token_set, set(event_text_payload.get("token_set", set()))),
        _token_overlap_ratio_from_token_sets(question_token_set, set(profile_value_payload.get("token_set", set()))),
        _token_overlap_ratio_from_token_sets(question_token_set, set(time_payload.get("token_set", set()))),
        has_time,
        has_profile,
        has_status,
        _token_overlap_ratio_from_token_sets(question_token_set, set(event_signature_payload.get("token_set", set()))),
        speaker_conflict,
        source_turn_overlap,
        content_overlap,
        profile_conflict,
        profile_missing,
        time_missing,
        source_turn_present,
        max(
            _token_overlap_ratio_from_token_sets(question_token_set, set(profile_value_payload.get("token_set", set()))),
            _max_token_overlap_ratio_from_payloads(question_token_set, profile_text_payloads),
        ),
        signature_anchor_coverage,
        signature_anchor_bigram_coverage,
        source_anchor_coverage,
        event_anchor_coverage,
        exact_signature_anchor_cover,
        same_speaker_day_past_distractor,
        temporal_anchor_miss,
    ]


def _event_relation_feature_values(
    left_event_node: Mapping[str, Any],
    right_event_node: Mapping[str, Any],
    *,
    prepared_left: Mapping[str, Any] | None = None,
    prepared_right: Mapping[str, Any] | None = None,
) -> List[float]:
    left_cache = dict(prepared_left or _prepare_event_scoring_features(left_event_node))
    right_cache = dict(prepared_right or _prepare_event_scoring_features(right_event_node))
    return _event_relation_feature_values_from_prepared(left_cache, right_cache)


def _event_relation_feature_values_from_prepared(
    left_cache: Mapping[str, Any],
    right_cache: Mapping[str, Any],
) -> List[float]:
    left_speaker = clean_text(left_cache.get("speaker", ""))
    right_speaker = clean_text(right_cache.get("speaker", ""))
    left_time_value = clean_text(left_cache.get("time_compare_value", ""))
    right_time_value = clean_text(right_cache.get("time_compare_value", ""))
    left_time_granularity = normalize_text(clean_text(left_cache.get("time_granularity", "")))
    right_time_granularity = normalize_text(clean_text(right_cache.get("time_granularity", "")))
    left_target_status = normalize_text(clean_text(left_cache.get("target_status", "")))
    right_target_status = normalize_text(clean_text(right_cache.get("target_status", "")))
    left_turn_index = int(left_cache.get("turn_index", 0) or 0)
    right_turn_index = int(right_cache.get("turn_index", 0) or 0)
    left_session = clean_text(left_cache.get("session_name", ""))
    right_session = clean_text(right_cache.get("session_name", ""))
    return [
        1.0 if left_speaker and right_speaker and left_speaker == right_speaker else 0.0,
        1.0 if left_time_value and right_time_value and left_time_value == right_time_value else 0.0,
        1.0 if left_time_granularity and right_time_granularity and left_time_granularity == right_time_granularity else 0.0,
        1.0 if left_target_status and right_target_status and left_target_status == right_target_status else 0.0,
        1.0 if left_turn_index > 0 and right_turn_index > 0 and left_session and right_session and left_session == right_session and abs(left_turn_index - right_turn_index) <= 2 else 0.0,
        _token_overlap_ratio_from_token_sets(
            set(left_cache.get("relation_signature_tokens", set())),
            set(right_cache.get("relation_signature_tokens", set())),
        ),
    ]


def _event_relation_feature_matrix(
    prepared_events: Sequence[Mapping[str, Any]],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    relation_rows = [
        [
            _event_relation_feature_values_from_prepared(left_cache, right_cache)
            for right_cache in prepared_events
        ]
        for left_cache in prepared_events
    ]
    if not relation_rows:
        return torch.zeros((0, 0, MATRIX_RELATION_FEATURE_DIM), dtype=dtype, device=device)
    return torch.tensor(relation_rows, dtype=dtype, device=device)


def _path_pair_feature_values(
    question: str,
    question_features: Mapping[str, Any],
    event_node: Mapping[str, Any],
    support_node: Mapping[str, Any],
    *,
    path_type: str,
    prepared_question: Mapping[str, Any] | None = None,
    prepared_event: Mapping[str, Any] | None = None,
    prepared_support: Mapping[str, Any] | None = None,
) -> List[float]:
    question_cache = dict(prepared_question or _prepare_question_scoring_features(question, question_features))
    event_cache = dict(prepared_event or _prepare_event_scoring_features(event_node))
    support_cache = dict(prepared_support or _prepare_node_scoring_features(support_node))
    question_token_set = set(question_cache.get("question_token_set", set()))
    question_anchor_tokens = list(question_cache.get("question_anchor_tokens", []) or [])
    question_anchor_bigrams = set(question_cache.get("question_anchor_bigrams", set()))
    semantic_target = clean_text(question_cache.get("semantic_target", ""))
    target_status_target = clean_text(question_cache.get("target_status_target", ""))
    time_target = clean_text(question_cache.get("time_target", ""))
    question_is_temporal = bool(question_cache.get("question_is_temporal", False))
    normalized_path_type = clean_text(path_type)
    support_text_payload = dict(support_cache.get("text", {}) or {})
    event_text_payload = dict(event_cache.get("event_text", {}) or {})
    event_signature_payload = dict(event_cache.get("event_signature", {}) or {})
    support_time_granularity = clean_text(support_cache.get("time_granularity", ""))
    support_time_payload = dict(support_cache.get("time_text", {}) or {})
    support_profile_type = clean_text(support_cache.get("profile_type", ""))
    support_profile_type_payload = dict(support_cache.get("profile_type_payload", {}) or {})
    support_status = clean_text(support_cache.get("status_or_text", ""))
    profile_like_question = semantic_target in {"identity", "research_topic", "education", "occupation", "profile"}
    preferred_path_type = "speaker_event_source_turn"
    if question_is_temporal:
        preferred_path_type = "speaker_event_time"
    elif profile_like_question:
        preferred_path_type = "speaker_event_profile"
    elif target_status_target:
        preferred_path_type = "speaker_event_status"
    preferred_match = 1.0 if normalized_path_type == preferred_path_type else 0.0
    dispreferred = 1.0 if normalized_path_type != preferred_path_type else 0.0
    time_match = 1.0 if time_target and support_time_granularity and _time_granularity_compatible(time_target, support_time_granularity) else 0.0
    time_conflict = 1.0 if time_target and support_time_granularity and not _time_granularity_compatible(time_target, support_time_granularity) else 0.0
    profile_match = 1.0 if semantic_target in {"identity", "research_topic", "education", "occupation"} and support_profile_type == semantic_target else 0.0
    profile_conflict = 1.0 if semantic_target in {"identity", "research_topic", "education", "occupation"} and support_profile_type and support_profile_type != semantic_target else 0.0
    status_match = 1.0 if target_status_target and support_status and support_status == target_status_target else 0.0
    status_conflict = 1.0 if target_status_target and support_status and support_status != target_status_target else 0.0
    event_anchor_coverage = max(
        _required_token_coverage_from_normalized_tokens(question_anchor_tokens, set(event_signature_payload.get("token_set", set()))),
        _required_token_coverage_from_normalized_tokens(question_anchor_tokens, set(event_text_payload.get("token_set", set()))),
    )
    event_anchor_bigram_coverage = max(
        _required_bigram_coverage_from_bigrams(question_anchor_bigrams, set(event_signature_payload.get("bigram_set", set()))),
        _required_bigram_coverage_from_bigrams(question_anchor_bigrams, set(event_text_payload.get("bigram_set", set()))),
    )
    support_anchor_coverage = max(
        _required_token_coverage_from_normalized_tokens(question_anchor_tokens, set(support_text_payload.get("token_set", set()))),
        _required_token_coverage_from_normalized_tokens(question_anchor_tokens, set(support_time_payload.get("token_set", set()))),
        _required_token_coverage_from_normalized_tokens(question_anchor_tokens, set(support_profile_type_payload.get("token_set", set()))),
    )
    support_anchor_bigram_coverage = max(
        _required_bigram_coverage_from_bigrams(question_anchor_bigrams, set(support_text_payload.get("bigram_set", set()))),
        _required_bigram_coverage_from_bigrams(question_anchor_bigrams, set(support_time_payload.get("bigram_set", set()))),
    )
    temporal_time_focus = 1.0 if (
        question_is_temporal
        and normalized_path_type == "speaker_event_time"
        and time_match > 0.5
        and event_anchor_coverage >= 0.5
    ) else 0.0
    temporal_non_time_penalty = 1.0 if question_is_temporal and normalized_path_type != "speaker_event_time" else 0.0
    temporal_profile_noise = 1.0 if question_is_temporal and normalized_path_type == "speaker_event_profile" else 0.0
    return [
        1.0 if normalized_path_type == "speaker_event_time" else 0.0,
        1.0 if normalized_path_type == "speaker_event_profile" else 0.0,
        1.0 if normalized_path_type == "speaker_event_status" else 0.0,
        1.0 if normalized_path_type == "speaker_event_source_turn" else 0.0,
        preferred_match,
        dispreferred,
        _token_overlap_ratio_from_token_sets(question_token_set, set(support_text_payload.get("token_set", set()))),
        max(
            _token_overlap_ratio_from_token_sets(question_token_set, set(event_text_payload.get("token_set", set()))),
            _token_overlap_ratio_from_token_sets(question_token_set, set(event_signature_payload.get("token_set", set()))),
        ),
        time_match,
        time_conflict,
        profile_match,
        profile_conflict,
        status_match,
        status_conflict,
        max(
            _token_overlap_ratio_from_token_sets(question_token_set, set(support_time_payload.get("token_set", set()))),
            _token_overlap_ratio_from_token_sets(question_token_set, set(support_profile_type_payload.get("token_set", set()))),
        ),
        event_anchor_coverage,
        event_anchor_bigram_coverage,
        support_anchor_coverage,
        support_anchor_bigram_coverage,
        temporal_time_focus,
        temporal_non_time_penalty,
        temporal_profile_noise,
    ]


def _path_relation_feature_values(
    left_path: Mapping[str, Any],
    right_path: Mapping[str, Any],
    *,
    node_lookup: Mapping[str, Mapping[str, Any]],
    prepared_left: Mapping[str, Any] | None = None,
    prepared_right: Mapping[str, Any] | None = None,
) -> List[float]:
    if prepared_left is None or prepared_right is None:
        node_scoring_features_by_id = {
            clean_text(node_id): _prepare_node_scoring_features(node)
            for node_id, node in dict(node_lookup or {}).items()
            if clean_text(node_id)
        }
    left_cache = dict(
        prepared_left
        or _prepare_path_scoring_features(
            left_path,
            node_scoring_features_by_id=node_scoring_features_by_id,
        )
    )
    right_cache = dict(
        prepared_right
        or _prepare_path_scoring_features(
            right_path,
            node_scoring_features_by_id=node_scoring_features_by_id,
        )
    )
    return _path_relation_feature_values_from_prepared(left_cache, right_cache)


def _path_relation_feature_values_from_prepared(
    left_cache: Mapping[str, Any],
    right_cache: Mapping[str, Any],
) -> List[float]:
    left_event_id = clean_text(left_cache.get("event_id", ""))
    right_event_id = clean_text(right_cache.get("event_id", ""))
    left_path_type = clean_text(left_cache.get("path_type", ""))
    right_path_type = clean_text(right_cache.get("path_type", ""))
    left_support_node_id = clean_text(left_cache.get("support_node_id", ""))
    right_support_node_id = clean_text(right_cache.get("support_node_id", ""))
    left_support_type = clean_text(left_cache.get("support_type", ""))
    right_support_type = clean_text(right_cache.get("support_type", ""))
    left_support_time_value = clean_text(left_cache.get("support_time_value", ""))
    right_support_time_value = clean_text(right_cache.get("support_time_value", ""))
    left_support_profile = clean_text(left_cache.get("support_profile", ""))
    right_support_profile = clean_text(right_cache.get("support_profile", ""))
    return [
        1.0 if left_event_id and right_event_id and left_event_id == right_event_id else 0.0,
        1.0 if left_path_type and right_path_type and left_path_type == right_path_type else 0.0,
        1.0 if left_support_node_id and right_support_node_id and left_support_node_id == right_support_node_id else 0.0,
        1.0 if left_support_type and right_support_type and left_support_type == right_support_type else 0.0,
        1.0 if (
            (left_support_time_value and right_support_time_value and left_support_time_value == right_support_time_value)
            or (left_support_profile and right_support_profile and left_support_profile == right_support_profile)
        ) else 0.0,
        _token_overlap_ratio_from_token_sets(
            set(left_cache.get("support_text_tokens", set())),
            set(right_cache.get("support_text_tokens", set())),
        ),
    ]


def _path_relation_feature_matrix(
    prepared_paths: Sequence[Mapping[str, Any]],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    relation_rows = [
        [
            _path_relation_feature_values_from_prepared(left_cache, right_cache)
            for right_cache in prepared_paths
        ]
        for left_cache in prepared_paths
    ]
    if not relation_rows:
        return torch.zeros((0, 0, PATH_MATRIX_RELATION_FEATURE_DIM), dtype=dtype, device=device)
    return torch.tensor(relation_rows, dtype=dtype, device=device)


def build_path_id(event_id: str, path_type: str, support_node_id: str) -> str:
    return f"{clean_text(event_id)}::{clean_text(path_type)}::{clean_text(support_node_id)}"


def parse_path_id(path_id: str) -> tuple[str, str, str]:
    normalized_path_id = clean_text(path_id)
    for path_type in PATH_TYPES:
        marker = f"::{path_type}::"
        if marker in normalized_path_id:
            event_id, support_node_id = normalized_path_id.split(marker, 1)
            return clean_text(event_id), path_type, clean_text(support_node_id)
    return "", "", ""


def build_default_path_templates(
    *,
    event_id: str,
    speaker_node_id: str,
    time_node_ids: Sequence[str] = (),
    profile_node_ids: Sequence[str] = (),
    status_node_ids: Sequence[str] = (),
    source_turn_node_ids: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    paths: List[Dict[str, Any]] = []
    for support_node_id in time_node_ids:
        paths.append(
            {
                "id": build_path_id(event_id, "speaker_event_time", support_node_id),
                "type": "speaker_event_time",
                "event_id": event_id,
                "node_ids": [speaker_node_id, event_id, support_node_id],
            }
        )
    for support_node_id in profile_node_ids:
        paths.append(
            {
                "id": build_path_id(event_id, "speaker_event_profile", support_node_id),
                "type": "speaker_event_profile",
                "event_id": event_id,
                "node_ids": [speaker_node_id, event_id, support_node_id],
            }
        )
    for support_node_id in status_node_ids:
        paths.append(
            {
                "id": build_path_id(event_id, "speaker_event_status", support_node_id),
                "type": "speaker_event_status",
                "event_id": event_id,
                "node_ids": [speaker_node_id, event_id, support_node_id],
            }
        )
    for support_node_id in source_turn_node_ids:
        paths.append(
            {
                "id": build_path_id(event_id, "speaker_event_source_turn", support_node_id),
                "type": "speaker_event_source_turn",
                "event_id": event_id,
                "node_ids": [speaker_node_id, event_id, support_node_id],
            }
        )
    return paths


@dataclass(slots=True)
class QueryTrainingExample:
    conversation_id: str
    question_id: str
    question: str
    question_features: Dict[str, Any]
    candidate_event_ids: List[str]
    positive_event_ids: List[str]
    positive_path_ids: List[str]
    positive_time_node_ids: List[str]
    negative_event_ids: List[str]
    answer_targets: Dict[str, Any]
    answer_plan_targets: Dict[str, Any] = field(default_factory=dict)
    temporal_target: Dict[str, Any] = field(default_factory=dict)
    event_catalog_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    hard_negative_event_ids: List[str] = field(default_factory=list, repr=False)
    easy_negative_event_ids: List[str] = field(default_factory=list, repr=False)
    question_anchor_tokens: List[str] = field(default_factory=list, repr=False)
    question_hash_indices: List[int] = field(default_factory=list, repr=False)
    question_feature_index: int | None = field(default=None, repr=False)
    question_query_type_index: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.candidate_event_ids = [clean_text(item) for item in list(self.candidate_event_ids or []) if clean_text(item)]
        self.positive_event_ids = [clean_text(item) for item in list(self.positive_event_ids or []) if clean_text(item)]
        self.positive_path_ids = [clean_text(item) for item in list(self.positive_path_ids or []) if clean_text(item)]
        self.positive_time_node_ids = [clean_text(item) for item in list(self.positive_time_node_ids or []) if clean_text(item)]
        legacy_negative_ids = [clean_text(item) for item in list(self.negative_event_ids or []) if clean_text(item)]
        explicit_hard_negative_ids = [clean_text(item) for item in list(self.hard_negative_event_ids or []) if clean_text(item)]
        explicit_easy_negative_ids = [clean_text(item) for item in list(self.easy_negative_event_ids or []) if clean_text(item)]
        positive_event_id_set = {item for item in self.positive_event_ids if item}
        if not explicit_hard_negative_ids and legacy_negative_ids:
            explicit_hard_negative_ids = list(legacy_negative_ids)
        combined_negative_ids = dedupe_texts([*legacy_negative_ids, *explicit_hard_negative_ids, *explicit_easy_negative_ids])
        self.negative_event_ids = [event_id for event_id in combined_negative_ids if event_id not in positive_event_id_set]
        negative_event_id_set = {item for item in self.negative_event_ids if item}
        self.hard_negative_event_ids = [
            event_id
            for event_id in dedupe_texts(explicit_hard_negative_ids)
            if event_id in negative_event_id_set
        ]
        hard_negative_event_id_set = {item for item in self.hard_negative_event_ids if item}
        self.easy_negative_event_ids = [
            event_id
            for event_id in dedupe_texts(explicit_easy_negative_ids)
            if event_id in negative_event_id_set and event_id not in hard_negative_event_id_set
        ]
        prepared_features = dict(self.question_features or {})
        cached_anchor_tokens = [normalize_text(clean_text(token)) for token in list(self.question_anchor_tokens or []) if clean_text(token)]
        if not cached_anchor_tokens:
            cached_anchor_tokens = _question_anchor_tokens(self.question, prepared_features)
        self.question_anchor_tokens = list(cached_anchor_tokens)
        if self.question_anchor_tokens:
            prepared_features["question_anchor_tokens"] = list(self.question_anchor_tokens)
        self.question_features = prepared_features
        if not self.question_hash_indices:
            self.question_hash_indices = _text_hash_indices(self.question, QUESTION_HASH_BUCKETS)
        feature_index, query_type_index = _question_feature_indices(self.question_features)
        if self.question_feature_index is None:
            self.question_feature_index = feature_index
        if self.question_query_type_index is None:
            self.question_query_type_index = query_type_index

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QueryTrainingExample":
        return cls(
            conversation_id=clean_text(payload.get("conversation_id", "")),
            question_id=clean_text(payload.get("question_id", "")),
            question=clean_text(payload.get("question", "")),
            question_features=dict(payload.get("question_features", {}) or {}),
            candidate_event_ids=[clean_text(item) for item in list(payload.get("candidate_event_ids", []) or []) if clean_text(item)],
            positive_event_ids=[clean_text(item) for item in list(payload.get("positive_event_ids", []) or []) if clean_text(item)],
            positive_path_ids=[clean_text(item) for item in list(payload.get("positive_path_ids", []) or []) if clean_text(item)],
            positive_time_node_ids=[clean_text(item) for item in list(payload.get("positive_time_node_ids", []) or []) if clean_text(item)],
            negative_event_ids=[clean_text(item) for item in list(payload.get("negative_event_ids", []) or []) if clean_text(item)],
            answer_targets=dict(payload.get("answer_targets", {}) or {}),
            answer_plan_targets=dict(payload.get("answer_plan_targets", {}) or {}),
            temporal_target=dict(payload.get("temporal_target", {}) or {}),
            event_catalog_size=int(payload.get("event_catalog_size", 0) or 0),
            metadata=dict(payload.get("metadata", {}) or {}),
            hard_negative_event_ids=[clean_text(item) for item in list(payload.get("hard_negative_event_ids", []) or []) if clean_text(item)],
            easy_negative_event_ids=[clean_text(item) for item in list(payload.get("easy_negative_event_ids", []) or []) if clean_text(item)],
            question_anchor_tokens=[clean_text(item) for item in list(payload.get("question_anchor_tokens", []) or []) if clean_text(item)],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "question_id": self.question_id,
            "question": self.question,
            "question_features": dict(self.question_features),
            "candidate_event_ids": list(self.candidate_event_ids),
            "positive_event_ids": list(self.positive_event_ids),
            "positive_path_ids": list(self.positive_path_ids),
            "positive_time_node_ids": list(self.positive_time_node_ids),
            "hard_negative_event_ids": list(self.hard_negative_event_ids),
            "easy_negative_event_ids": list(self.easy_negative_event_ids),
            "negative_event_ids": list(self.negative_event_ids),
            "answer_targets": dict(self.answer_targets),
            "answer_plan_targets": dict(self.answer_plan_targets),
            "temporal_target": dict(self.temporal_target),
            "event_catalog_size": int(self.event_catalog_size),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class _TrainingConversationSummary:
    conversation_id: str
    source_dataset: str
    row_count: int = 0
    time_example_count: int = 0
    multi_evidence_count: int = 0
    temporal_positive_count: int = 0
    answer_type_counts: Dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class _TrainingConversationGroup:
    conversation_id: str
    source_dataset: str
    row_count: int
    time_example_count: int
    multi_evidence_count: int
    temporal_positive_count: int
    answer_type_counts: Dict[str, int]
    repeat_count: int = 1
    sampling_multiplier: float = 1.0
    source_store: Any | None = field(default=None, repr=False)
    offsets: tuple[int, ...] = field(default_factory=tuple, repr=False)
    rows: tuple[QueryTrainingExample, ...] = field(default_factory=tuple, repr=False)


def _clone_query_training_example(
    example: QueryTrainingExample,
    *,
    metadata_updates: Mapping[str, Any] | None = None,
) -> QueryTrainingExample:
    cloned = QueryTrainingExample(
        conversation_id=example.conversation_id,
        question_id=example.question_id,
        question=example.question,
        question_features=dict(example.question_features or {}),
        candidate_event_ids=list(example.candidate_event_ids or []),
        positive_event_ids=list(example.positive_event_ids or []),
        positive_path_ids=list(example.positive_path_ids or []),
        positive_time_node_ids=list(example.positive_time_node_ids or []),
        negative_event_ids=list(example.negative_event_ids or []),
        answer_targets=dict(example.answer_targets or {}),
        answer_plan_targets=dict(example.answer_plan_targets or {}),
        temporal_target=dict(example.temporal_target or {}),
        event_catalog_size=int(example.event_catalog_size),
        metadata={
            **dict(example.metadata or {}),
            **dict(metadata_updates or {}),
        },
        hard_negative_event_ids=list(example.hard_negative_event_ids or []),
        easy_negative_event_ids=list(example.easy_negative_event_ids or []),
        question_anchor_tokens=list(example.question_anchor_tokens or []),
        question_hash_indices=list(example.question_hash_indices or []),
        question_feature_index=example.question_feature_index,
        question_query_type_index=example.question_query_type_index,
    )
    return cloned


def _training_source_dataset_name(*, example: QueryTrainingExample, fallback_source: str = "") -> str:
    metadata = dict(example.metadata or {})
    dataset_name = clean_text(
        metadata.get("source_dataset", "")
        or metadata.get("dataset", "")
        or metadata.get("source", "")
        or metadata.get("origin_dataset", "")
        or metadata.get("source_data", "")
    )
    if dataset_name:
        return dataset_name
    conversation_id = clean_text(example.conversation_id)
    if "__" in conversation_id:
        prefix = clean_text(conversation_id.split("__", 1)[0])
        if prefix:
            return prefix
    fallback = clean_text(fallback_source)
    return fallback or "unknown"


def _training_answer_type_name(example: QueryTrainingExample) -> str:
    answer_type = clean_text(dict(example.answer_targets or {}).get("answer_type", ""))
    if answer_type in ANSWER_TYPE_TO_ID:
        return answer_type
    inferred_index = int(_answer_type_label(example))
    if 0 <= inferred_index < len(ANSWER_TYPES):
        return ANSWER_TYPES[inferred_index]
    return "event_text"


def _positive_event_id_count(example: QueryTrainingExample) -> int:
    return len({clean_text(item) for item in list(example.positive_event_ids or []) if clean_text(item)})


def _training_supervision_bucket_name(example: QueryTrainingExample) -> str:
    metadata = dict(example.metadata or {})
    answer_type = _training_answer_type_name(example)
    positive_count = _positive_event_id_count(example)
    explicit_bucket = clean_text(metadata.get("supervision_bucket", ""))
    if explicit_bucket:
        # Old prepared rows can carry stale metadata.supervision_bucket values.
        # The provable multi-positive cases are safer to reclassify than to let
        # them hide inside ordinary event_text/time/profile metrics.
        if answer_type == "event_text" and positive_count > 1 and explicit_bucket == "event_text":
            return "list_event_text_multi_positive"
        if answer_type in {"time", "profile"} and positive_count > 1 and explicit_bucket == answer_type:
            return f"{answer_type}_multi_positive"
        return explicit_bucket
    if answer_type == "abstain":
        return "abstain"
    if positive_count > 1:
        if answer_type == "event_text":
            return "list_event_text_multi_positive"
        if answer_type in {"time", "profile"}:
            return f"{answer_type}_multi_positive"
        return "multi_evidence"
    return answer_type or "event_text"


def _loss_group_key_for_example(
    example: QueryTrainingExample,
    supervision: Mapping[str, Any] | None = None,
    *,
    mode: str = "answer_type",
) -> str:
    normalized_mode = clean_text(mode or "none").lower().replace("-", "_")
    if normalized_mode in {"", "none", "off", "disabled"}:
        return ""
    answer_type = _training_answer_type_name(example)
    if normalized_mode in {"answer_type", "type"}:
        return answer_type
    supervision_bucket = clean_text(dict(supervision or {}).get("supervision_bucket", ""))
    if not supervision_bucket:
        supervision_bucket = _training_supervision_bucket_name(example)
    if normalized_mode in {"supervision_bucket", "supervision", "task_bucket", "bucket"}:
        return supervision_bucket
    if normalized_mode in {"source_supervision_bucket", "source_bucket"}:
        source_dataset = clean_text(dict(supervision or {}).get("training_source_dataset", ""))
        if not source_dataset:
            source_dataset = _training_source_dataset_name(example=example, fallback_source="unknown")
        return f"{source_dataset}:{supervision_bucket}"
    if normalized_mode in {"source_answer_type", "source_type"}:
        source_dataset = clean_text(dict(supervision or {}).get("training_source_dataset", ""))
        if not source_dataset:
            source_dataset = _training_source_dataset_name(example=example, fallback_source="unknown")
        return f"{source_dataset}:{answer_type}"
    if normalized_mode in {"source", "dataset"}:
        source_dataset = clean_text(dict(supervision or {}).get("training_source_dataset", ""))
        if not source_dataset:
            source_dataset = _training_source_dataset_name(example=example, fallback_source="unknown")
        return source_dataset
    return answer_type


def _group_balanced_batch_loss(
    loss_entries: Sequence[tuple[Tensor, str]],
    *,
    mode: str = "answer_type",
) -> tuple[Tensor, int]:
    if not loss_entries:
        raise ValueError("loss_entries must not be empty")
    losses = [loss for loss, _ in loss_entries]
    normalized_mode = clean_text(mode or "none").lower().replace("-", "_")
    if normalized_mode in {"", "none", "off", "disabled"}:
        return torch.stack(losses).mean(), 1
    grouped_losses: "OrderedDict[str, List[Tensor]]" = OrderedDict()
    for loss, group_key in loss_entries:
        resolved_group_key = clean_text(group_key) or "unknown"
        grouped_losses.setdefault(resolved_group_key, []).append(loss)
    if len(grouped_losses) <= 1:
        return torch.stack(losses).mean(), len(grouped_losses) or 1
    group_means = [torch.stack(group_values).mean() for group_values in grouped_losses.values() if group_values]
    if not group_means:
        return torch.stack(losses).mean(), 1
    return torch.stack(group_means).mean(), len(group_means)


def _new_answer_type_metric_stats() -> Dict[str, int]:
    return {
        "samples": 0,
        "recall_event_recall_total": 0,
        "recall_event24_hits": 0,
        "recall_event24_positive_hits": 0,
        "recall_event24_positive_total": 0,
        "event_recall_total": 0,
        "event_recall1_hits": 0,
        "event_recall5_hits": 0,
        "event5_positive_hits": 0,
        "event5_positive_total": 0,
        "path_recall_total": 0,
        "path_recall3_hits": 0,
        "path3_positive_hits": 0,
        "path3_positive_total": 0,
        "answer_plan_selected_total": 0,
        "answer_plan_selected_recall5_hits": 0,
        "answer_plan_selected_positive_hits": 0,
        "answer_plan_selected_positive_total": 0,
        "answer_plan_current_total": 0,
        "answer_plan_current_top1_hits": 0,
        "temporal_total": 0,
        "temporal_hits": 0,
    }


def _finalize_answer_type_metric_stats(
    stats_by_answer_type: Mapping[str, Mapping[str, int]],
) -> Dict[str, Dict[str, Any]]:
    finalized: Dict[str, Dict[str, Any]] = {}
    for answer_type, stats in sorted(stats_by_answer_type.items()):
        recall_total = int(stats.get("recall_event_recall_total", 0) or 0)
        event_total = int(stats.get("event_recall_total", 0) or 0)
        path_total = int(stats.get("path_recall_total", 0) or 0)
        answer_plan_selected_total = int(stats.get("answer_plan_selected_total", 0) or 0)
        answer_plan_current_total = int(stats.get("answer_plan_current_total", 0) or 0)
        temporal_total = int(stats.get("temporal_total", 0) or 0)
        finalized[answer_type] = {
            "samples": int(stats.get("samples", 0) or 0),
            "recall_event_recall_at_24": round(float(stats.get("recall_event24_hits", 0) or 0) / max(1, recall_total), 6),
            "recall_event_positive_coverage_at_24": round(
                float(stats.get("recall_event24_positive_hits", 0) or 0)
                / max(1, int(stats.get("recall_event24_positive_total", 0) or 0)),
                6,
            ),
            "event_recall_at_1": round(float(stats.get("event_recall1_hits", 0) or 0) / max(1, event_total), 6),
            "event_recall_at_5": round(float(stats.get("event_recall5_hits", 0) or 0) / max(1, event_total), 6),
            "event_positive_coverage_at_5": round(
                float(stats.get("event5_positive_hits", 0) or 0)
                / max(1, int(stats.get("event5_positive_total", 0) or 0)),
                6,
            ),
            "path_recall_at_3": round(float(stats.get("path_recall3_hits", 0) or 0) / max(1, path_total), 6),
            "path_positive_coverage_at_3": round(
                float(stats.get("path3_positive_hits", 0) or 0)
                / max(1, int(stats.get("path3_positive_total", 0) or 0)),
                6,
            ),
            "answer_plan_selected_recall_at_5": round(
                float(stats.get("answer_plan_selected_recall5_hits", 0) or 0)
                / max(1, answer_plan_selected_total),
                6,
            ),
            "answer_plan_selected_positive_coverage_at_5": round(
                float(stats.get("answer_plan_selected_positive_hits", 0) or 0)
                / max(1, int(stats.get("answer_plan_selected_positive_total", 0) or 0)),
                6,
            ),
            "answer_plan_current_top1_accuracy": round(
                float(stats.get("answer_plan_current_top1_hits", 0) or 0)
                / max(1, answer_plan_current_total),
                6,
            ),
            "temporal_accuracy": round(float(stats.get("temporal_hits", 0) or 0) / max(1, temporal_total), 6),
            "recall_event_recall_total": recall_total,
            "recall_event24_positive_total": int(stats.get("recall_event24_positive_total", 0) or 0),
            "event_recall_total": event_total,
            "event5_positive_total": int(stats.get("event5_positive_total", 0) or 0),
            "path_recall_total": path_total,
            "path3_positive_total": int(stats.get("path3_positive_total", 0) or 0),
            "answer_plan_selected_total": answer_plan_selected_total,
            "answer_plan_selected_positive_total": int(stats.get("answer_plan_selected_positive_total", 0) or 0),
            "answer_plan_current_total": answer_plan_current_total,
            "temporal_total": temporal_total,
        }
    return finalized


def _merge_nested_metric_summaries(metrics_list: Sequence[Mapping[str, Any]], nested_key: str) -> Dict[str, Dict[str, Any]]:
    group_names: set[str] = set()
    for metrics in metrics_list:
        nested = metrics.get(nested_key, {})
        if isinstance(nested, Mapping):
            group_names.update(clean_text(group_name) for group_name in nested.keys() if clean_text(group_name))
    merged: Dict[str, Dict[str, Any]] = {}
    for group_name in sorted(group_names):
        group_metrics_list = [
            dict(dict(metrics.get(nested_key, {}) or {}).get(group_name, {}) or {})
            for metrics in metrics_list
            if isinstance(metrics.get(nested_key, {}), Mapping)
            and isinstance(dict(metrics.get(nested_key, {}) or {}).get(group_name, {}), Mapping)
        ]
        if not group_metrics_list:
            continue
        group_summary: Dict[str, Any] = {}
        metric_keys: set[str] = set()
        for group_metrics in group_metrics_list:
            metric_keys.update(clean_text(key) for key in group_metrics.keys() if clean_text(key))
        for metric_key in sorted(metric_keys):
            if metric_key in _METRIC_COUNT_KEYS:
                group_summary[metric_key] = int(sum(int(item.get(metric_key, 0) or 0) for item in group_metrics_list))
                continue
            weight_key = _METRIC_WEIGHT_KEYS.get(metric_key, "samples")
            total_weight = float(sum(item.get(weight_key, 0.0) or 0.0 for item in group_metrics_list))
            if total_weight <= 0.0:
                group_summary[metric_key] = 0.0
                continue
            weighted_total = sum(
                float(item.get(metric_key, 0.0) or 0.0) * float(item.get(weight_key, 0.0) or 0.0)
                for item in group_metrics_list
            )
            group_summary[metric_key] = round(weighted_total / total_weight, 6)
        merged[group_name] = group_summary
    return merged


def _has_positive_temporal_supervision(example: QueryTrainingExample) -> bool:
    if example.positive_time_node_ids:
        return True
    temporal_target = dict(example.temporal_target or {})
    return bool(temporal_target.get("has_positive_time_supervision", False))


def _has_multi_evidence_supervision(example: QueryTrainingExample) -> bool:
    if _training_answer_type_name(example) == "multi_evidence":
        return True
    return len({clean_text(item) for item in example.positive_event_ids if clean_text(item)}) > 1


def _smoothed_inverse_share_factor(
    *,
    count: int,
    total: int,
    unique_count: int,
    alpha: float,
    blend_uniform_ratio: float,
) -> float:
    resolved_total = max(1, int(total or 0))
    resolved_unique_count = max(1, int(unique_count or 0))
    resolved_count = max(1, int(count or 0))
    if resolved_unique_count <= 1:
        return 1.0
    source_share = float(resolved_count) / float(resolved_total)
    uniform_share = 1.0 / float(resolved_unique_count)
    raw_factor = math.pow(max(uniform_share / max(source_share, 1e-12), 1e-12), max(0.0, float(alpha)))
    blend = min(1.0, max(0.0, float(blend_uniform_ratio)))
    return blend + (1.0 - blend) * raw_factor


def _bounded_training_weight(value: float, *, minimum: float, maximum: float) -> float:
    resolved_minimum = float(minimum)
    resolved_maximum = float(maximum)
    if resolved_maximum < resolved_minimum:
        resolved_maximum = resolved_minimum
    return min(resolved_maximum, max(resolved_minimum, float(value)))


def _conversation_repeat_count(
    conversation_id: str,
    *,
    sampling_multiplier: float,
    max_group_repeat: int,
) -> int:
    resolved_max_group_repeat = max(1, int(max_group_repeat or 1))
    capped_multiplier = min(float(resolved_max_group_repeat), max(1.0, float(sampling_multiplier or 1.0)))
    base_repeat = 1 + int(math.floor(max(0.0, capped_multiplier - 1.0)))
    repeat_count = min(resolved_max_group_repeat, base_repeat)
    fractional_extra = max(0.0, capped_multiplier - float(base_repeat))
    if repeat_count < resolved_max_group_repeat and fractional_extra > 1e-9:
        threshold = float(stable_hash_bucket(f"train-repeat::{clean_text(conversation_id)}", 1_000_000)) / 1_000_000.0
        if threshold < fractional_extra:
            repeat_count += 1
    return max(1, min(resolved_max_group_repeat, repeat_count))


@dataclass(frozen=True, slots=True)
class QueryBatchSourceOffsets:
    path: str
    offsets: tuple[int, ...]
    skip_bad_rows: bool = False


@dataclass(frozen=True, slots=True)
class QueryBatchSpec:
    sources: tuple[QueryBatchSourceOffsets, ...]


@dataclass(slots=True)
class GraphCacheItem:
    graph: Dict[str, Any]
    tensors: Dict[str, Any]


class LazyGraphCache(Mapping[str, GraphCacheItem]):
    def __init__(
        self,
        *,
        graph_paths: Mapping[str, Path],
        conversation_ids: Sequence[str] | None = None,
        graph_dir: Path | None = None,
        known_graph_count: int | None = None,
        device: torch.device,
        cache_dir: Path | None = None,
        memory_cache_size: int = 64,
        prefetch_workers: int = 0,
        prefetch_window: int = 0,
        cache_write_enabled: bool = True,
        require_cache_hit: bool = False,
        progress_callback: Callable[[str, Dict[str, Any]], None] | None = None,
    ) -> None:
        self._graph_paths = {clean_text(key): Path(value) for key, value in dict(graph_paths or {}).items() if clean_text(key)}
        self._conversation_ids = [clean_text(item) for item in list(conversation_ids or []) if clean_text(item)]
        self._graph_dir = Path(graph_dir) if graph_dir is not None else None
        self._known_graph_count = max(len(self._graph_paths), len(self._conversation_ids), int(known_graph_count or 0))
        self._device = torch.device(device)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._memory_cache_size = max(0, int(memory_cache_size or 0))
        self._memory_cache: OrderedDict[str, GraphCacheItem] = OrderedDict()
        self._progress_callback = progress_callback
        self._disk_cache_hits = 0
        self._tensorized_count = 0
        self._memory_cache_hits = 0
        self._load_count = 0
        self._pin_host_tensors = self._device.type == "cpu" and torch.cuda.is_available()
        self._cache_write_enabled = bool(cache_write_enabled)
        self._require_cache_hit = bool(require_cache_hit)
        self._prefetch_workers = max(0, int(prefetch_workers or 0)) if self._device.type == "cpu" else 0
        self._prefetch_window = max(0, int(prefetch_window or 0))
        self._prefetch_submitted = 0
        self._prefetch_completed = 0
        self._prefetch_failed = 0
        self._prefetch_lock = threading.Lock()
        self._prefetch_futures: OrderedDict[str, Future[tuple[GraphCacheItem, str]]] = OrderedDict()
        self._prefetch_executor = (
            ThreadPoolExecutor(max_workers=self._prefetch_workers, thread_name_prefix="lazy-graph-cache")
            if self._prefetch_workers > 0 and self._prefetch_window > 0
            else None
        )
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return max(len(self._graph_paths), len(self._conversation_ids), int(self._known_graph_count))

    def __iter__(self):
        if self._conversation_ids:
            return iter(self._conversation_ids)
        return iter(self._graph_paths)

    def __getitem__(self, key: str) -> GraphCacheItem:
        conversation_id = clean_text(key)
        graph_path = self._graph_paths.get(conversation_id)
        if graph_path is None:
            if self._graph_dir is None:
                raise KeyError(key)
            graph_path = self._graph_dir / f"{conversation_id}.json"
            self._graph_paths[conversation_id] = graph_path
        self._collect_completed_prefetch()
        cached_item = self._memory_cache.get(conversation_id)
        if cached_item is not None:
            self._memory_cache.move_to_end(conversation_id)
            self._memory_cache_hits += 1
            return cached_item

        prefetched_item = self._consume_prefetched_item(conversation_id)
        if prefetched_item is not None:
            self._memory_cache_hits += 1
            return prefetched_item

        item, loaded_from = self._load_item(conversation_id)
        self._store_loaded_item(conversation_id, item, loaded_from=loaded_from)
        return item

    def close(self) -> None:
        executor = self._prefetch_executor
        self._prefetch_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def stats(self) -> Dict[str, int]:
        self._collect_completed_prefetch()
        return {
            "graph_count": int(len(self)),
            "loads": int(self._load_count),
            "memory_cache_hits": int(self._memory_cache_hits),
            "disk_cache_hits": int(self._disk_cache_hits),
            "tensorized": int(self._tensorized_count),
            "prefetch_submitted": int(self._prefetch_submitted),
            "prefetch_completed": int(self._prefetch_completed),
            "prefetch_failed": int(self._prefetch_failed),
            "prefetch_pending": int(len(self._prefetch_futures)),
            "memory_cache_size": int(len(self._memory_cache)),
        }

    def batch_prepare_context(self) -> Dict[str, Any]:
        worker_memory_cache_size = 0
        if self._memory_cache_size > 0:
            worker_memory_cache_size = min(32, max(2, int(self._memory_cache_size or 0)))
        if self._graph_dir is not None:
            return {
                "conversation_specs": {},
                "graph_dir": str(self._graph_dir),
                "cache_dir": str(self._cache_dir) if self._cache_dir is not None else "",
                "cache_write_enabled": bool(self._cache_write_enabled),
                "require_cache_hit": bool(self._require_cache_hit),
                "worker_memory_cache_size": int(worker_memory_cache_size),
            }
        conversation_specs: Dict[str, Dict[str, Any]] = {}
        for conversation_id in [clean_text(item) for item in list(self._graph_paths) if clean_text(item)]:
            graph_path = self._graph_paths.get(conversation_id)
            if graph_path is None:
                continue
            cache_path = _graph_tensor_cache_path(self._cache_dir, graph_path) if self._cache_dir is not None else None
            conversation_specs[conversation_id] = {
                "graph_path": str(graph_path),
                "cache_path": str(cache_path) if cache_path is not None else "",
            }
        return {
            "conversation_specs": conversation_specs,
            "graph_dir": "",
            "cache_dir": "",
            "cache_write_enabled": bool(self._cache_write_enabled),
            "require_cache_hit": bool(self._require_cache_hit),
            "worker_memory_cache_size": int(worker_memory_cache_size),
        }

    def prefetch(self, conversation_ids: Sequence[str]) -> None:
        if self._prefetch_executor is None or self._prefetch_window <= 0:
            return
        self._collect_completed_prefetch()
        normalized_ids: List[str] = []
        for item in list(conversation_ids or []):
            conversation_id = clean_text(item)
            if not conversation_id:
                continue
            if conversation_id not in self._graph_paths:
                if self._graph_dir is None:
                    continue
                self._graph_paths[conversation_id] = self._graph_dir / f"{conversation_id}.json"
            normalized_ids.append(conversation_id)
        if not normalized_ids:
            return
        with self._prefetch_lock:
            for conversation_id in normalized_ids:
                if conversation_id in self._memory_cache or conversation_id in self._prefetch_futures:
                    continue
                if len(self._prefetch_futures) >= self._prefetch_window:
                    break
                future = self._prefetch_executor.submit(self._load_item, conversation_id)
                self._prefetch_futures[conversation_id] = future
                self._prefetch_submitted += 1

    def _load_item(self, conversation_id: str) -> tuple[GraphCacheItem, str]:
        graph_path = self._graph_paths.get(conversation_id)
        if graph_path is None:
            if self._graph_dir is None:
                raise KeyError(conversation_id)
            graph_path = self._graph_dir / f"{conversation_id}.json"
            self._graph_paths[conversation_id] = graph_path
        source_signature = _graph_source_signature(graph_path)
        payload = None
        loaded_from = "tensorized"
        if self._cache_dir is not None:
            payload = _load_graph_tensor_cache(
                _graph_tensor_cache_path(self._cache_dir, graph_path),
                source_signature=source_signature,
            )
        if payload is not None:
            graph = dict(payload["graph"])
            tensors = _graph_tensor_mapping_to_device(payload["tensors"], self._device)
            loaded_from = "disk_cache"
        else:
            if self._cache_dir is not None and self._require_cache_hit:
                raise FileNotFoundError(f"Prebuilt graph tensor cache missing for conversation '{conversation_id}': {graph_path}")
            graph = dict(read_json(graph_path))
            cache_tensors = tensorize_graph(graph, device=torch.device("cpu"))
            if self._cache_dir is not None and self._cache_write_enabled:
                _write_graph_tensor_cache(
                    _graph_tensor_cache_path(self._cache_dir, graph_path),
                    source_signature=source_signature,
                    graph=graph,
                    tensors=cache_tensors,
                )
            tensors = _graph_tensor_mapping_to_device(cache_tensors, self._device)
        tensors = dict(_ensure_graph_scoring_feature_cache(tensors))
        if self._pin_host_tensors:
            tensors = _pin_graph_tensor_mapping(tensors)
        return GraphCacheItem(graph=graph, tensors=tensors), loaded_from

    def _store_loaded_item(self, conversation_id: str, item: GraphCacheItem, *, loaded_from: str) -> GraphCacheItem:
        self._load_count += 1
        if loaded_from == "disk_cache":
            self._disk_cache_hits += 1
        else:
            self._tensorized_count += 1
        if self._memory_cache_size > 0:
            self._memory_cache[conversation_id] = item
            self._memory_cache.move_to_end(conversation_id)
            while len(self._memory_cache) > self._memory_cache_size:
                self._memory_cache.popitem(last=False)
        self._maybe_report_progress(loaded_from=loaded_from, conversation_id=conversation_id)
        return item

    def _collect_completed_prefetch(self) -> None:
        if not self._prefetch_futures:
            return
        completed_ids: List[str] = []
        with self._prefetch_lock:
            for conversation_id, future in list(self._prefetch_futures.items()):
                if not future.done():
                    continue
                completed_ids.append(conversation_id)
                try:
                    item, loaded_from = future.result()
                except Exception as exc:
                    self._prefetch_failed += 1
                    self._report_prefetch_error(conversation_id=conversation_id, exc=exc)
                    continue
                self._store_loaded_item(conversation_id, item, loaded_from=loaded_from)
                self._prefetch_completed += 1
            for conversation_id in completed_ids:
                self._prefetch_futures.pop(conversation_id, None)

    def _consume_prefetched_item(self, conversation_id: str) -> GraphCacheItem | None:
        with self._prefetch_lock:
            future = self._prefetch_futures.pop(conversation_id, None)
        if future is None:
            return None
        try:
            item, loaded_from = future.result()
        except Exception as exc:
            self._prefetch_failed += 1
            self._report_prefetch_error(conversation_id=conversation_id, exc=exc)
            return None
        self._prefetch_completed += 1
        return self._store_loaded_item(conversation_id, item, loaded_from=loaded_from)

    def _report_prefetch_error(self, *, conversation_id: str, exc: Exception) -> None:
        if self._progress_callback is None:
            return
        self._progress_callback(
            "lazy_graph_cache_prefetch_error",
            {
                "conversation_id": clean_text(conversation_id),
                "error_type": type(exc).__name__,
                "error_message": clean_text(str(exc)),
                "prefetch_failed": int(self._prefetch_failed),
            },
        )

    def _maybe_report_progress(self, *, loaded_from: str, conversation_id: str) -> None:
        if self._progress_callback is None:
            return
        if self._load_count not in {1} and self._load_count % 1000 != 0:
            return
        self._progress_callback(
            "lazy_graph_cache_progress",
            {
                "graph_count": len(self._graph_paths),
                "known_graph_count": int(len(self)),
                "loads": int(self._load_count),
                "memory_cache_hits": int(self._memory_cache_hits),
                "disk_cache_hits": int(self._disk_cache_hits),
                "tensorized": int(self._tensorized_count),
                "memory_cache_size": int(len(self._memory_cache)),
                "prefetch_submitted": int(self._prefetch_submitted),
                "prefetch_completed": int(self._prefetch_completed),
                "prefetch_pending": int(len(self._prefetch_futures)),
                "loaded_from": clean_text(loaded_from),
                "last_conversation_id": clean_text(conversation_id),
            },
        )

    def register_conversation_ids(self, conversation_ids: Sequence[str], *, graph_dir: Path) -> None:
        resolved_graph_dir = Path(graph_dir)
        for conversation_id in conversation_ids:
            normalized_id = clean_text(conversation_id)
            if not normalized_id:
                continue
            if normalized_id not in self._graph_paths:
                self._graph_paths[normalized_id] = resolved_graph_dir / f"{normalized_id}.json"
            if self._conversation_ids and normalized_id in self._conversation_ids:
                continue
            if self._conversation_ids:
                self._conversation_ids.append(normalized_id)
        self._known_graph_count = max(int(self._known_graph_count), len(self))


def index_graph_paths(
    graph_dir: Path,
    *,
    progress_callback: Callable[[str, Dict[str, Any]], None] | None = None,
) -> Dict[str, Path]:
    graph_paths: Dict[str, Path] = {}
    for index, path in enumerate(_iter_graph_paths(graph_dir), start=1):
        conversation_id = clean_text(path.stem)
        if not conversation_id:
            raise RuntimeError(f"Unable to resolve conversation_id from graph filename: {path}")
        if conversation_id in graph_paths:
            raise RuntimeError(f"Duplicate conversation_id in graph directory {graph_dir}: {conversation_id}")
        graph_paths[conversation_id] = path
        if progress_callback is not None and (index == 1 or index % 10000 == 0):
            progress_callback(
                "graph_index_progress",
                {
                    "graph_dir": str(graph_dir),
                    "graphs_indexed": int(index),
                    "last_conversation_id": conversation_id,
                },
            )
    if progress_callback is not None:
        progress_callback(
            "graph_index_completed",
            {
                "graph_dir": str(graph_dir),
                "graph_count": int(len(graph_paths)),
            },
        )
    return graph_paths


def load_lazy_graph_cache(
    graph_dir: Path,
    *,
    device: torch.device,
    conversation_ids: Sequence[str] | None = None,
    cache_dir: Path | None = None,
    memory_cache_size: int = 64,
    prefetch_workers: int = 0,
    prefetch_window: int = 0,
    cache_write_enabled: bool = True,
    require_cache_hit: bool = False,
    progress_callback: Callable[[str, Dict[str, Any]], None] | None = None,
) -> LazyGraphCache:
    resolved_graph_dir = Path(graph_dir)
    resolved_conversation_ids = [clean_text(item) for item in list(conversation_ids or []) if clean_text(item)]
    if resolved_conversation_ids and len(resolved_conversation_ids) <= MAX_LAZY_GRAPH_PATHS_MATERIALIZE:
        graph_paths = {
            conversation_id: resolved_graph_dir / f"{conversation_id}.json"
            for conversation_id in resolved_conversation_ids
        }
        materialized_conversation_ids: Sequence[str] | None = None
        direct_graph_dir = None
        known_graph_count = len(graph_paths)
    elif resolved_conversation_ids:
        graph_paths = {}
        materialized_conversation_ids = resolved_conversation_ids
        direct_graph_dir = resolved_graph_dir
        known_graph_count = len({conversation_id for conversation_id in resolved_conversation_ids if conversation_id})
    else:
        graph_paths = index_graph_paths(resolved_graph_dir, progress_callback=progress_callback)
        materialized_conversation_ids = None
        direct_graph_dir = None
        known_graph_count = len(graph_paths)
    return LazyGraphCache(
        graph_paths=graph_paths,
        conversation_ids=materialized_conversation_ids,
        graph_dir=direct_graph_dir,
        known_graph_count=known_graph_count,
        device=device,
        cache_dir=cache_dir,
        memory_cache_size=memory_cache_size,
        prefetch_workers=prefetch_workers,
        prefetch_window=prefetch_window,
        cache_write_enabled=cache_write_enabled,
        require_cache_hit=require_cache_hit,
        progress_callback=progress_callback,
    )


def _build_graph_tensor_cache_entry(
    graph_path_str: str,
    cache_dir_str: str,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    torch.set_num_threads(1)
    graph_path = Path(graph_path_str)
    cache_dir = Path(cache_dir_str)
    source_signature = _graph_source_signature(graph_path)
    cache_path = _graph_tensor_cache_path(cache_dir, graph_path)
    try:
        if not force_rebuild:
            cached_payload = _load_graph_tensor_cache(cache_path, source_signature=source_signature)
            if cached_payload is not None:
                cached_graph = dict(cached_payload.get("graph", {}) or {})
                return {
                    "status": "cache_hit",
                    "conversation_id": clean_text(cached_graph.get("conversation_id", graph_path.stem)),
                    "cache_path": str(cache_path),
                    "graph_path": str(graph_path),
                }
        graph = dict(read_json(graph_path))
        cache_tensors = tensorize_graph(graph, device=torch.device("cpu"))
        wrote = _write_graph_tensor_cache(
            cache_path,
            source_signature=source_signature,
            graph=graph,
            tensors=cache_tensors,
        )
        return {
            "status": "built" if wrote else "write_skipped",
            "conversation_id": clean_text(graph.get("conversation_id", graph_path.stem)),
            "cache_path": str(cache_path),
            "graph_path": str(graph_path),
        }
    except Exception as exc:
        return {
            "status": "error",
            "conversation_id": clean_text(graph_path.stem),
            "cache_path": str(cache_path),
            "graph_path": str(graph_path),
            "error_type": type(exc).__name__,
            "error_message": clean_text(str(exc)),
        }


def build_graph_tensor_cache(
    graph_dir: Path,
    *,
    cache_dir: Path,
    conversation_ids: Sequence[str] | None = None,
    worker_count: int = 1,
    force_rebuild: bool = False,
    error_log_path: Path | None = None,
    progress_callback: Callable[[str, Dict[str, Any]], None] | None = None,
) -> Dict[str, int]:
    resolved_graph_dir = Path(graph_dir)
    resolved_cache_dir = Path(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    if conversation_ids:
        seen: set[str] = set()
        graph_paths = []
        for conversation_id in conversation_ids:
            normalized_id = clean_text(conversation_id)
            if not normalized_id or normalized_id in seen:
                continue
            seen.add(normalized_id)
            graph_paths.append(resolved_graph_dir / f"{normalized_id}.json")
    else:
        graph_paths = list(_iter_graph_paths(resolved_graph_dir))
    total_graphs = len(graph_paths)
    summary = {
        "graph_count": int(total_graphs),
        "cache_hits": 0,
        "built": 0,
        "write_skipped": 0,
        "errors": 0,
        "workers": max(1, int(worker_count or 1)),
    }
    if progress_callback is not None:
        progress_callback(
            "graph_cache_build_started",
            {
                "graph_dir": str(resolved_graph_dir),
                "cache_dir": str(resolved_cache_dir),
                "graph_count": int(total_graphs),
                "workers": int(summary["workers"]),
                "force_rebuild": bool(force_rebuild),
            },
        )
    resolved_error_log_path = Path(error_log_path) if error_log_path is not None else None
    if resolved_error_log_path is not None:
        resolved_error_log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_error_log_path.unlink(missing_ok=True)

    def _handle_result(result: Mapping[str, Any], completed_count: int) -> None:
        status = clean_text(result.get("status", "")) or "unknown"
        if status == "cache_hit":
            summary["cache_hits"] += 1
        elif status == "built":
            summary["built"] += 1
        elif status == "write_skipped":
            summary["write_skipped"] += 1
        else:
            summary["errors"] += 1
            if resolved_error_log_path is not None:
                with resolved_error_log_path.open("a", encoding="utf-8", newline="") as handle:
                    handle.write(
                        json_dumps(
                            {
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "stage": "graph_cache_prebuild",
                                "status": status,
                                "conversation_id": clean_text(result.get("conversation_id", "")),
                                "graph_path": clean_text(result.get("graph_path", "")),
                                "cache_path": clean_text(result.get("cache_path", "")),
                                "error_type": clean_text(result.get("error_type", "")),
                                "error_message": clean_text(result.get("error_message", "")),
                            }
                        )
                    )
                    handle.write("\n")
            if progress_callback is not None:
                progress_callback(
                    "graph_cache_build_error",
                    {
                        "graph_dir": str(resolved_graph_dir),
                        "cache_dir": str(resolved_cache_dir),
                        "completed": int(completed_count),
                        "graph_count": int(total_graphs),
                        "conversation_id": clean_text(result.get("conversation_id", "")),
                        "graph_path": clean_text(result.get("graph_path", "")),
                        "error_type": clean_text(result.get("error_type", "")),
                        "error_message": clean_text(result.get("error_message", "")),
                        "errors": int(summary["errors"]),
                    },
                )
        if progress_callback is not None and (
            completed_count == 1
            or completed_count == total_graphs
            or completed_count % 1000 == 0
        ):
            progress_callback(
                "graph_cache_build_progress",
                {
                    "graph_dir": str(resolved_graph_dir),
                    "cache_dir": str(resolved_cache_dir),
                    "completed": int(completed_count),
                    "graph_count": int(total_graphs),
                    "built": int(summary["built"]),
                    "cache_hits": int(summary["cache_hits"]),
                    "write_skipped": int(summary["write_skipped"]),
                    "errors": int(summary["errors"]),
                    "last_conversation_id": clean_text(result.get("conversation_id", "")),
                    "last_status": status,
                },
            )

    resolved_workers = max(1, int(worker_count or 1))
    if resolved_workers <= 1 or total_graphs <= 1:
        for completed_count, graph_path in enumerate(graph_paths, start=1):
            _handle_result(
                _build_graph_tensor_cache_entry(
                    str(graph_path),
                    str(resolved_cache_dir),
                    force_rebuild=force_rebuild,
                ),
                completed_count,
            )
    else:
        chunksize = max(1, min(128, total_graphs // max(1, resolved_workers * 8)))
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            iterator = executor.map(
                _build_graph_tensor_cache_entry,
                (str(path) for path in graph_paths),
                itertools.repeat(str(resolved_cache_dir), total_graphs),
                itertools.repeat(force_rebuild, total_graphs),
                chunksize=chunksize,
            )
            for completed_count, result in enumerate(iterator, start=1):
                _handle_result(result, completed_count)
    if progress_callback is not None:
        progress_callback(
            "graph_cache_build_completed",
            {
                "graph_dir": str(resolved_graph_dir),
                "cache_dir": str(resolved_cache_dir),
                **{key: int(value) for key, value in summary.items()},
            },
        )
    return {key: int(value) for key, value in summary.items()}


def _combine_graph_tensors_by_conversation(
    graph_tensors_by_conversation: Mapping[str, Mapping[str, Any]],
    *,
    device: torch.device,
) -> Dict[str, Any]:
    ordered_items = [
        (clean_text(conversation_id), dict(graph_tensors))
        for conversation_id, graph_tensors in graph_tensors_by_conversation.items()
        if clean_text(conversation_id)
    ]
    if not ordered_items:
        raise ValueError("graph_tensors_by_conversation must not be empty")
    if len(ordered_items) == 1:
        conversation_id, graph_tensors = ordered_items[0]
        merged = dict(graph_tensors)
        merged = dict(_ensure_graph_scoring_feature_cache(merged))
        merged["event_node_ids_by_conversation"] = {
            conversation_id: [event_id for event_id in list(graph_tensors.get("event_node_ids", []) or []) if clean_text(event_id)]
        }
        merged["temporal_paths_by_conversation"] = {
            conversation_id: [dict(path) for path in list(graph_tensors.get("temporal_paths", []) or [])]
        }
        return merged

    merged_node_ids: List[str] = []
    merged_node_texts: List[str] = []
    merged_node_text_hash_indices: List[List[int]] = []
    merged_node_lookup: Dict[str, Dict[str, Any]] = {}
    merged_event_node_ids: List[str] = []
    merged_time_node_ids: List[str] = []
    merged_paths: List[Dict[str, Any]] = []
    merged_temporal_paths: List[Dict[str, Any]] = []
    merged_paths_by_event_id: Dict[str, List[Dict[str, Any]]] = {}
    merged_event_support_lookup: Dict[str, Dict[str, List[str]]] = {}
    merged_support_node_ids_by_event: Dict[str, List[str]] = {}
    merged_support_node_ids_by_event_and_type: Dict[str, Dict[str, List[str]]] = {}
    merged_event_scoring_features_by_id: Dict[str, Dict[str, Any]] = {}
    merged_node_scoring_features_by_id: Dict[str, Dict[str, Any]] = {}
    merged_path_scoring_features_by_id: Dict[str, Dict[str, Any]] = {}
    event_node_ids_by_conversation: Dict[str, List[str]] = {}
    temporal_paths_by_conversation: Dict[str, List[Dict[str, Any]]] = {}
    node_type_tensors: List[Tensor] = []
    node_scalar_feature_tensors: List[Tensor] = []
    edge_src_tensors: List[Tensor] = []
    edge_dst_tensors: List[Tensor] = []
    edge_type_tensors: List[Tensor] = []
    node_offset = 0
    seen_node_ids: set[str] = set()
    non_blocking = device.type == "cuda"

    for conversation_id, graph_tensors in ordered_items:
        graph_tensors = dict(_ensure_graph_scoring_feature_cache(graph_tensors))
        local_node_type_ids = graph_tensors.get("node_type_ids")
        local_node_scalar_features = graph_tensors.get("node_scalar_features")
        local_edge_src = graph_tensors.get("edge_src")
        local_edge_dst = graph_tensors.get("edge_dst")
        local_edge_type_ids = graph_tensors.get("edge_type_ids")
        if not isinstance(local_node_type_ids, Tensor) or not isinstance(local_node_scalar_features, Tensor):
            raise TypeError("graph_tensors missing node tensors required for merge")
        if not isinstance(local_edge_src, Tensor) or not isinstance(local_edge_dst, Tensor) or not isinstance(local_edge_type_ids, Tensor):
            raise TypeError("graph_tensors missing edge tensors required for merge")
        local_node_texts = [clean_text(text) for text in list(graph_tensors.get("node_texts", []) or [])]
        local_node_text_hash_indices = [list(indices or []) for indices in list(graph_tensors.get("node_text_hash_indices", []) or [])]
        local_node_ids_raw = [clean_text(node_id) for node_id in list(graph_tensors.get("node_ids", []) or [])]
        local_node_count = int(local_node_type_ids.size(0))
        if len(local_node_ids_raw) != local_node_count:
            raise GraphTensorValidationError(
                f"_combine_graph_tensors_by_conversation: node_ids length mismatch for conversation '{conversation_id}'"
                f" (node_ids={len(local_node_ids_raw)}, node_type_ids={local_node_count})"
            )
        if len(local_node_texts) != local_node_count or len(local_node_text_hash_indices) != local_node_count:
            raise GraphTensorValidationError(
                f"_combine_graph_tensors_by_conversation: node text alignment mismatch for conversation '{conversation_id}'"
                f" (node_texts={len(local_node_texts)}, node_text_hash_indices={len(local_node_text_hash_indices)}, node_type_ids={local_node_count})"
            )
        local_node_ids: List[str] = []
        local_seen_node_ids: set[str] = set()
        for local_index, raw_node_id in enumerate(local_node_ids_raw):
            normalized_node_id = raw_node_id or f"__missing_node__:{conversation_id}:{local_index}"
            if normalized_node_id in local_seen_node_ids:
                raise GraphTensorValidationError(
                    f"_combine_graph_tensors_by_conversation: duplicate node_id within conversation '{conversation_id}': {normalized_node_id}"
                )
            local_seen_node_ids.add(normalized_node_id)
            local_node_ids.append(normalized_node_id)
        duplicate_node_ids = [node_id for node_id in local_node_ids if node_id in seen_node_ids]
        if duplicate_node_ids:
            raise RuntimeError(f"Duplicate node_id across merged graph tensors: {duplicate_node_ids[0]}")
        seen_node_ids.update(local_node_ids)
        merged_node_ids.extend(local_node_ids)
        merged_node_texts.extend(local_node_texts)
        merged_node_text_hash_indices.extend(local_node_text_hash_indices)
        for node_id, node in dict(graph_tensors.get("node_by_id", {}) or {}).items():
            normalized_node_id = clean_text(node_id)
            if normalized_node_id:
                merged_node_lookup[normalized_node_id] = dict(node)
        local_event_node_ids = [
            clean_text(event_id) for event_id in list(graph_tensors.get("event_node_ids", []) or []) if clean_text(event_id)
        ]
        merged_event_node_ids.extend(local_event_node_ids)
        event_node_ids_by_conversation[conversation_id] = list(local_event_node_ids)
        merged_time_node_ids.extend(
            [clean_text(node_id) for node_id in list(graph_tensors.get("time_node_ids", []) or []) if clean_text(node_id)]
        )
        local_paths = [dict(path) for path in list(graph_tensors.get("paths", []) or [])]
        merged_paths.extend(local_paths)
        for event_id, paths in dict(graph_tensors.get("paths_by_event_id", {}) or {}).items():
            normalized_event_id = clean_text(event_id)
            if not normalized_event_id:
                continue
            merged_paths_by_event_id.setdefault(normalized_event_id, []).extend(dict(path) for path in list(paths or []))
        local_temporal_paths = [dict(path) for path in list(graph_tensors.get("temporal_paths", []) or [])]
        merged_temporal_paths.extend(local_temporal_paths)
        temporal_paths_by_conversation[conversation_id] = list(local_temporal_paths)
        for event_id, payload in dict(graph_tensors.get("event_support_lookup", {}) or {}).items():
            normalized_event_id = clean_text(event_id)
            if not normalized_event_id:
                continue
            merged_event_support_lookup[normalized_event_id] = {
                clean_text(key): [clean_text(item) for item in list(values or []) if clean_text(item)]
                for key, values in dict(payload or {}).items()
                if clean_text(key)
            }
        for event_id, support_node_ids in dict(graph_tensors.get("support_node_ids_by_event", {}) or {}).items():
            normalized_event_id = clean_text(event_id)
            if normalized_event_id:
                merged_support_node_ids_by_event[normalized_event_id] = [
                    clean_text(node_id) for node_id in list(support_node_ids or []) if clean_text(node_id)
                ]
        for event_id, path_type_map in dict(graph_tensors.get("support_node_ids_by_event_and_type", {}) or {}).items():
            normalized_event_id = clean_text(event_id)
            if not normalized_event_id:
                continue
            merged_support_node_ids_by_event_and_type[normalized_event_id] = {
                clean_text(path_type): [clean_text(node_id) for node_id in list(node_ids or []) if clean_text(node_id)]
                for path_type, node_ids in dict(path_type_map or {}).items()
                if clean_text(path_type)
            }
        for event_id, payload in dict(graph_tensors.get("event_scoring_features_by_id", {}) or {}).items():
            normalized_event_id = clean_text(event_id)
            if normalized_event_id:
                merged_event_scoring_features_by_id[normalized_event_id] = dict(payload or {})
        for node_id, payload in dict(graph_tensors.get("node_scoring_features_by_id", {}) or {}).items():
            normalized_node_id = clean_text(node_id)
            if normalized_node_id:
                merged_node_scoring_features_by_id[normalized_node_id] = dict(payload or {})
        for path_id, payload in dict(graph_tensors.get("path_scoring_features_by_id", {}) or {}).items():
            normalized_path_id = clean_text(path_id)
            if normalized_path_id:
                merged_path_scoring_features_by_id[normalized_path_id] = dict(payload or {})
        node_type_tensors.append(_tensor_to_device(local_node_type_ids, device, non_blocking=non_blocking))
        node_scalar_feature_tensors.append(_tensor_to_device(local_node_scalar_features, device, non_blocking=non_blocking))
        edge_src_tensors.append(_tensor_to_device(local_edge_src, device, non_blocking=non_blocking) + node_offset)
        edge_dst_tensors.append(_tensor_to_device(local_edge_dst, device, non_blocking=non_blocking) + node_offset)
        edge_type_tensors.append(_tensor_to_device(local_edge_type_ids, device, non_blocking=non_blocking))
        node_offset += local_node_count

    merged_node_id_to_index = {node_id: index for index, node_id in enumerate(merged_node_ids)}
    return {
        "node_ids": merged_node_ids,
        "node_texts": merged_node_texts,
        "node_text_hash_indices": merged_node_text_hash_indices,
        "node_by_id": merged_node_lookup,
        "event_node_ids": merged_event_node_ids,
        "time_node_ids": merged_time_node_ids,
        "node_type_ids": torch.cat(node_type_tensors, dim=0) if node_type_tensors else torch.zeros((0,), dtype=torch.long, device=device),
        "node_scalar_features": (
            torch.cat(node_scalar_feature_tensors, dim=0)
            if node_scalar_feature_tensors
            else torch.zeros((0, NODE_SCALAR_DIM), dtype=torch.float32, device=device)
        ),
        "edge_src": torch.cat(edge_src_tensors, dim=0) if edge_src_tensors else torch.zeros((0,), dtype=torch.long, device=device),
        "edge_dst": torch.cat(edge_dst_tensors, dim=0) if edge_dst_tensors else torch.zeros((0,), dtype=torch.long, device=device),
        "edge_type_ids": (
            torch.cat(edge_type_tensors, dim=0) if edge_type_tensors else torch.zeros((0,), dtype=torch.long, device=device)
        ),
        "node_id_to_index": merged_node_id_to_index,
        "paths": merged_paths,
        "paths_by_event_id": merged_paths_by_event_id,
        "temporal_paths": merged_temporal_paths,
        "event_support_lookup": merged_event_support_lookup,
        "support_node_ids_by_event": merged_support_node_ids_by_event,
        "support_node_ids_by_event_and_type": merged_support_node_ids_by_event_and_type,
        "event_scoring_features_by_id": merged_event_scoring_features_by_id,
        "node_scoring_features_by_id": merged_node_scoring_features_by_id,
        "path_scoring_features_by_id": merged_path_scoring_features_by_id,
        "event_node_ids_by_conversation": event_node_ids_by_conversation,
        "temporal_paths_by_conversation": temporal_paths_by_conversation,
    }


def _hashed_text_embedding(texts: Sequence[str], embedding: nn.Embedding, buckets: int) -> Tensor:
    return _hashed_index_embedding_batch(
        [_text_hash_indices(text, buckets) for text in texts],
        embedding,
    )


def _question_feature_indices(features: Mapping[str, Any]) -> tuple[int, int]:
    feature_key = "|".join(
        [
            clean_text(features.get("semantic_slot_target", "")),
            clean_text(features.get("target_status_target", "")),
            clean_text(features.get("time_granularity_target", "")),
        ]
    )
    feature_index = stable_hash_bucket(feature_key or "__feature__", 256)
    query_type_index = 0
    if bool(features.get("is_temporal", False)):
        query_type_index = 1
    elif clean_text(features.get("semantic_slot_target", "")) in {"identity", "research_topic", "education", "occupation", "profile"}:
        query_type_index = 2
    elif bool(list(features.get("speaker_candidates", []) or [])):
        query_type_index = 3
    return feature_index, query_type_index


def _question_understanding_targets(features: Mapping[str, Any]) -> Dict[str, Any]:
    semantic_target = clean_text(features.get("semantic_slot_target", ""))
    status_target = clean_text(features.get("target_status_target", ""))
    time_target = clean_text(features.get("time_granularity_target", ""))
    is_temporal = bool(features.get("is_temporal", False))
    return {
        "semantic_slot_index": SEMANTIC_SLOT_TO_ID.get(semantic_target, SEMANTIC_SLOT_TO_ID["event"]),
        "target_status_index": TARGET_STATUS_TO_ID.get(status_target, TARGET_STATUS_TO_ID[""]),
        "time_granularity_index": TIME_GRANULARITY_TO_ID.get(time_target, TIME_GRANULARITY_TO_ID[""]),
        "is_temporal_value": 1.0 if is_temporal else 0.0,
    }


def _confident_label_from_logits(
    logits: Tensor,
    *,
    labels: Sequence[str],
    fallback: str,
    min_margin: float = 0.08,
) -> str:
    if not isinstance(logits, Tensor) or logits.numel() <= 0:
        return clean_text(fallback)
    probs = torch.softmax(logits.detach().float(), dim=-1)
    top_k = min(2, int(probs.numel()))
    top_values, top_indices = torch.topk(probs, k=top_k, dim=0)
    top_margin = float(top_values[0] - top_values[1]) if top_values.numel() > 1 else float(top_values[0])
    if top_margin < float(min_margin):
        return clean_text(fallback)
    top_index = int(top_indices[0].item())
    if 0 <= top_index < len(labels):
        return clean_text(labels[top_index])
    return clean_text(fallback)


def _confident_temporal_from_logit(logit: Tensor, *, fallback: bool, min_margin: float = 0.12) -> bool:
    if not isinstance(logit, Tensor) or logit.numel() <= 0:
        return bool(fallback)
    probability = float(torch.sigmoid(logit.detach().float()).item())
    if abs(probability - 0.5) < float(min_margin):
        return bool(fallback)
    return probability >= 0.5


def _merge_learned_question_features_from_intent_logits(
    base_features: Mapping[str, Any],
    *,
    semantic_logits: Tensor,
    status_logits: Tensor,
    time_granularity_logits: Tensor,
    temporal_logit: Tensor,
) -> Dict[str, Any]:
    merged = dict(base_features or {})
    merged["semantic_slot_target"] = _confident_label_from_logits(
        semantic_logits,
        labels=SEMANTIC_SLOTS,
        fallback=clean_text(merged.get("semantic_slot_target", "event")) or "event",
    )
    merged["target_status_target"] = _confident_label_from_logits(
        status_logits,
        labels=TARGET_STATUSES,
        fallback=clean_text(merged.get("target_status_target", "")),
    )
    merged["time_granularity_target"] = _confident_label_from_logits(
        time_granularity_logits,
        labels=TIME_GRANULARITIES,
        fallback=clean_text(merged.get("time_granularity_target", "")),
    )
    merged["is_temporal"] = _confident_temporal_from_logit(
        temporal_logit,
        fallback=bool(merged.get("is_temporal", False)),
    )
    if clean_text(merged.get("semantic_slot_target", "")) == "event_time":
        merged["is_temporal"] = True
    if clean_text(merged.get("time_granularity_target", "")) in {"day", "month", "year", "relative_day_reference", "day_or_coarse"}:
        merged["is_temporal"] = True
    return merged


class _LightweightSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        relation_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.relation_bias_projection = (
            nn.Linear(int(relation_feature_dim), self.num_heads, bias=False)
            if int(relation_feature_dim) > 0
            else None
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: Tensor, token_mask: Tensor, *, relation_features: Tensor | None = None) -> Tensor:
        key_padding_mask = ~token_mask
        attn_mask = None
        if relation_features is not None and self.relation_bias_projection is not None:
            batch_size, token_count, _ = hidden.shape
            relation_bias = self.relation_bias_projection(relation_features).permute(0, 3, 1, 2)
            relation_bias = relation_bias.masked_fill((~token_mask).unsqueeze(1).unsqueeze(1), -1e4)
            attn_mask = relation_bias.reshape(batch_size * self.num_heads, token_count, token_count)
            key_padding_mask = None
        attn_input = self.attn_norm(hidden)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        hidden = hidden + self.attn_dropout(attn_output)
        hidden = hidden * token_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        hidden = hidden * token_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        return hidden


class HashedSelfAttentionTextEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_embeddings: int,
        embed_dim: int,
        max_tokens: int,
        attention_layers: int = ENCODER_ATTENTION_LAYERS,
        attention_heads: int = ENCODER_ATTENTION_HEADS,
        dropout: float = MESSAGE_DROPOUT,
    ) -> None:
        super().__init__()
        self.max_tokens = max(1, int(max_tokens))
        self.embed_dim = int(embed_dim)
        self.token_embedding = nn.Embedding(num_embeddings, embed_dim)
        self.position_embedding = nn.Embedding(self.max_tokens, embed_dim)
        self.role_embedding = nn.Embedding(TMCRA_TOKEN_ROLE_COUNT, embed_dim)
        self.role_predictor = nn.Linear(embed_dim, TMCRA_TOKEN_ROLE_COUNT)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                _LightweightSelfAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=attention_heads,
                    dropout=dropout,
                    relation_feature_dim=TMCRA_TOKEN_RELATION_FEATURE_DIM,
                )
                for _ in range(max(1, int(attention_layers)))
            ]
        )
        for block in self.blocks:
            if block.relation_bias_projection is not None:
                nn.init.zeros_(block.relation_bias_projection.weight)
        nn.init.zeros_(self.role_embedding.weight)
        self.pool_norm = nn.LayerNorm(embed_dim)
        self.pool_projection = nn.Linear(embed_dim, 1)
        self._last_role_aux_loss: Tensor | None = None
        self._last_role_predictions: Dict[str, Tensor] | None = None

    def forward_indices(
        self,
        index_batches: Sequence[Sequence[int]],
        *,
        token_role_batches: Sequence[Sequence[int]] | None = None,
        token_identity_batches: Sequence[Sequence[int]] | None = None,
    ) -> Tensor:
        device = self.token_embedding.weight.device
        self._last_role_aux_loss = None
        self._last_role_predictions = None
        if not index_batches:
            return torch.zeros((0, self.embed_dim), dtype=self.token_embedding.weight.dtype, device=device)
        token_embeddings, token_mask = _hashed_token_embedding_batch(
            index_batches,
            self.token_embedding,
            max_tokens=self.max_tokens,
        )
        sequence_length = int(token_embeddings.size(1))
        position_ids = torch.arange(sequence_length, dtype=torch.long, device=device)
        if token_identity_batches is None:
            token_identity_batches = [
                [int(item) + 1 for item in list(index_values or [])] or [0]
                for index_values in index_batches
            ]
        padded_token_ids, token_mask = _padded_long_batch(
            token_identity_batches,
            device=device,
            fill_value=0,
            max_tokens=self.max_tokens,
        )
        base_hidden = token_embeddings + self.position_embedding(position_ids).unsqueeze(0)
        role_logits = self.role_predictor(base_hidden)
        role_probs = torch.softmax(role_logits, dim=-1)
        role_probs = role_probs * token_mask.unsqueeze(-1).to(dtype=role_probs.dtype)
        if token_role_batches is not None and self.training:
            padded_role_targets, _ = _padded_long_batch(
                token_role_batches,
                device=device,
                fill_value=TMCRA_TOKEN_ROLE_EMPTY,
                max_tokens=self.max_tokens,
            )
            valid_targets = token_mask.reshape(-1)
            if bool(torch.any(valid_targets)):
                self._last_role_aux_loss = F.cross_entropy(
                    role_logits.reshape(-1, TMCRA_TOKEN_ROLE_COUNT)[valid_targets],
                    padded_role_targets.reshape(-1)[valid_targets],
                )
        predicted_role_ids = torch.argmax(role_probs, dim=-1)
        predicted_role_ids = predicted_role_ids.masked_fill(~token_mask, TMCRA_TOKEN_ROLE_EMPTY)
        self._last_role_predictions = {
            "role_probs": role_probs.detach(),
            "predicted_role_ids": predicted_role_ids.detach(),
            "token_mask": token_mask.detach(),
        }
        relation_features = _tmcra_text_relation_features_from_tensors(
            padded_token_ids,
            predicted_role_ids,
            token_mask=token_mask,
            dtype=token_embeddings.dtype,
        )
        role_context = torch.matmul(role_probs, self.role_embedding.weight)
        hidden = base_hidden + role_context
        hidden = self.input_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden, token_mask, relation_features=relation_features)
        return self._attention_pool(hidden, token_mask)

    def pop_last_role_aux_loss(self) -> Tensor | None:
        value = self._last_role_aux_loss
        self._last_role_aux_loss = None
        return value

    def pop_last_role_predictions(self) -> Dict[str, Tensor] | None:
        value = self._last_role_predictions
        self._last_role_predictions = None
        return value

    def _attention_pool(self, hidden: Tensor, token_mask: Tensor) -> Tensor:
        pool_logits = self.pool_projection(self.pool_norm(hidden)).squeeze(-1)
        pool_logits = pool_logits.masked_fill(~token_mask, float("-inf"))
        pool_weights = torch.softmax(pool_logits, dim=-1)
        pool_weights = pool_weights.masked_fill(~token_mask, 0.0)
        pool_weights = pool_weights / pool_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (hidden * pool_weights.unsqueeze(-1)).sum(dim=1)


class QuestionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = HashedSelfAttentionTextEncoder(
            num_embeddings=QUESTION_HASH_BUCKETS,
            embed_dim=QUESTION_EMBED_DIM,
            max_tokens=QUESTION_MAX_TOKENS,
            attention_layers=ENCODER_ATTENTION_LAYERS,
            attention_heads=ENCODER_ATTENTION_HEADS,
            dropout=MESSAGE_DROPOUT,
        )
        self.token_embedding = self.text_encoder.token_embedding
        self.feature_embedding = nn.Embedding(256, QUESTION_CATEGORICAL_EMBED_DIM)
        self.query_type_embedding = nn.Embedding(8, QUESTION_CATEGORICAL_EMBED_DIM)
        self._last_text_embeddings: Tensor | None = None

    def _feature_indices(self, features: Mapping[str, Any]) -> tuple[int, int]:
        return _question_feature_indices(features)

    def pop_last_role_aux_loss(self) -> Tensor | None:
        return self.text_encoder.pop_last_role_aux_loss()

    def pop_last_role_predictions(self) -> Dict[str, Tensor] | None:
        return self.text_encoder.pop_last_role_predictions()

    def pop_last_text_embeddings(self) -> Tensor | None:
        value = self._last_text_embeddings
        self._last_text_embeddings = None
        return value

    def forward_batch(
        self,
        questions: Sequence[str],
        features_batch: Sequence[Mapping[str, Any]],
        *,
        token_hash_indices_batch: Sequence[Sequence[int]] | None = None,
        feature_indices: Sequence[int] | None = None,
        query_type_indices: Sequence[int] | None = None,
    ) -> Tensor:
        device = self.token_embedding.weight.device
        if not questions:
            self._last_text_embeddings = None
            return torch.zeros((0, QUESTION_OUTPUT_DIM), dtype=self.token_embedding.weight.dtype, device=device)
        resolved_token_hash_indices = list(token_hash_indices_batch or [_text_hash_indices(question, QUESTION_HASH_BUCKETS) for question in questions])
        token_role_batches = None
        question_token_payloads = [
            (_resolved_text_tokens(question), [])
            for question in questions
        ]
        if self.training:
            question_token_payloads = [
                _question_token_roles(question, features_batch[index] if index < len(features_batch) else {})
                for index, question in enumerate(questions)
            ]
            token_role_batches = [
                _align_sequence_length(
                    roles,
                    target_length=len(resolved_token_hash_indices[index]),
                    fill_value=TMCRA_TOKEN_ROLE_CONTENT,
                ) or [TMCRA_TOKEN_ROLE_EMPTY]
                for index, (_, roles) in enumerate(question_token_payloads)
            ]
        token_identity_batches = [
            [
                stable_token_identity(f"qtok:{token}") + 1
                for token in _align_sequence_length(
                    tokens,
                    target_length=len(resolved_token_hash_indices[index]),
                    fill_value="__empty__",
                )
            ] or [0]
            for index, (tokens, _) in enumerate(question_token_payloads)
        ]
        text_embeddings = self.text_encoder.forward_indices(
            resolved_token_hash_indices,
            token_role_batches=token_role_batches,
            token_identity_batches=token_identity_batches,
        )
        self._last_text_embeddings = text_embeddings
        resolved_feature_indices = list(feature_indices or [])
        resolved_query_type_indices = list(query_type_indices or [])
        if len(resolved_feature_indices) != len(questions) or len(resolved_query_type_indices) != len(questions):
            resolved_feature_indices = []
            resolved_query_type_indices = []
            for features in features_batch:
                feature_index, query_type_index = self._feature_indices(features)
                resolved_feature_indices.append(feature_index)
                resolved_query_type_indices.append(query_type_index)
        _validate_index_values(
            resolved_feature_indices,
            upper_bound=int(self.feature_embedding.num_embeddings),
            index_name="question_feature_index",
            context="QuestionEncoder.forward_batch",
            extra={"question_count": len(questions)},
        )
        _validate_index_values(
            resolved_query_type_indices,
            upper_bound=int(self.query_type_embedding.num_embeddings),
            index_name="question_query_type_index",
            context="QuestionEncoder.forward_batch",
            extra={"question_count": len(questions)},
        )
        feature_embedding = self.feature_embedding(torch.tensor(resolved_feature_indices, dtype=torch.long, device=device))
        query_type_embedding = self.query_type_embedding(torch.tensor(resolved_query_type_indices, dtype=torch.long, device=device))
        return torch.cat([text_embeddings, feature_embedding, query_type_embedding], dim=-1)

    def forward(self, question: str, features: Mapping[str, Any]) -> Tensor:
        return self.forward_batch([question], [features])[0]


class QuestionIntentHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(QUESTION_EMBED_DIM),
            nn.Linear(QUESTION_EMBED_DIM, MESSAGE_HIDDEN_DIM),
            nn.SiLU(),
        )
        self.semantic_head = nn.Linear(MESSAGE_HIDDEN_DIM, len(SEMANTIC_SLOTS))
        self.status_head = nn.Linear(MESSAGE_HIDDEN_DIM, len(TARGET_STATUSES))
        self.time_granularity_head = nn.Linear(MESSAGE_HIDDEN_DIM, len(TIME_GRANULARITIES))
        self.temporal_head = nn.Linear(MESSAGE_HIDDEN_DIM, 1)
        for layer in (self.semantic_head, self.status_head, self.time_granularity_head, self.temporal_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, question_text_embeddings: Tensor) -> Dict[str, Tensor]:
        hidden = self.shared(question_text_embeddings)
        return {
            "semantic_logits": self.semantic_head(hidden),
            "status_logits": self.status_head(hidden),
            "time_granularity_logits": self.time_granularity_head(hidden),
            "temporal_logits": self.temporal_head(hidden).squeeze(-1),
        }


class MemoryRouterHead(nn.Module):
    """Predict which memory layers should be opened before candidate ranking."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(QUESTION_OUTPUT_DIM),
            nn.Linear(QUESTION_OUTPUT_DIM, MESSAGE_HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(MESSAGE_DROPOUT),
            nn.Linear(MESSAGE_HIDDEN_DIM, len(MEMORY_ROUTER_LAYERS)),
        )
        output = self.net[-1]
        if isinstance(output, nn.Linear):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(self, question_embeddings: Tensor) -> Tensor:
        return self.net(question_embeddings)


class NodeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = HashedSelfAttentionTextEncoder(
            num_embeddings=NODE_HASH_BUCKETS,
            embed_dim=NODE_EMBED_DIM,
            max_tokens=NODE_MAX_TOKENS,
            attention_layers=ENCODER_ATTENTION_LAYERS,
            attention_heads=ENCODER_ATTENTION_HEADS,
            dropout=MESSAGE_DROPOUT,
        )
        self.text_embedding = self.text_encoder.token_embedding
        self.node_type_embedding = nn.Embedding(len(NODE_TYPE_TO_ID), NODE_TYPE_EMBED_DIM)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(NODE_SCALAR_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, NODE_SCALAR_HIDDEN_DIM),
        )

    def pop_last_role_aux_loss(self) -> Tensor | None:
        return self.text_encoder.pop_last_role_aux_loss()

    def forward(
        self,
        node_texts: Sequence[str],
        node_type_ids: Tensor,
        scalar_features: Tensor,
        *,
        token_hash_indices_batch: Sequence[Sequence[int]] | None = None,
    ) -> Tensor:
        resolved_token_hash_indices = list(token_hash_indices_batch or [_text_hash_indices(text, NODE_HASH_BUCKETS) for text in node_texts])
        node_type_values = node_type_ids.detach().cpu().tolist()
        token_role_batches = None
        node_token_payloads = [
            (_resolved_text_tokens(node_texts[index] if index < len(node_texts) else ""), [])
            for index in range(len(node_texts))
        ]
        if self.training:
            node_token_payloads = [
                _node_token_roles(
                    node_texts[index] if index < len(node_texts) else "",
                    int(node_type_values[index]) if index < len(node_type_values) else 0,
                )
                for index in range(len(node_texts))
            ]
            token_role_batches = [
                _align_sequence_length(
                    roles,
                    target_length=len(resolved_token_hash_indices[index]),
                    fill_value=TMCRA_TOKEN_ROLE_CONTENT,
                ) or [TMCRA_TOKEN_ROLE_EMPTY]
                for index, (_, roles) in enumerate(node_token_payloads)
            ]
        token_identity_batches = [
            [
                stable_token_identity(f"ntok:{token}") + 1
                for token in _align_sequence_length(
                    tokens,
                    target_length=len(resolved_token_hash_indices[index]),
                    fill_value="__empty__",
                )
            ] or [0]
            for index, (tokens, _) in enumerate(node_token_payloads)
        ]
        text_embedding = self.text_encoder.forward_indices(
            resolved_token_hash_indices,
            token_role_batches=token_role_batches,
            token_identity_batches=token_identity_batches,
        )
        _validate_index_values(
            node_type_ids.detach().cpu().tolist(),
            upper_bound=int(self.node_type_embedding.num_embeddings),
            index_name="node_type_id",
            context="NodeEncoder.forward",
            extra={"node_count": int(node_type_ids.numel())},
        )
        type_embedding = self.node_type_embedding(node_type_ids)
        scalar_embedding = self.scalar_mlp(scalar_features)
        return torch.cat([text_embedding, type_embedding, scalar_embedding], dim=-1)


class TypedMessagePassing(nn.Module):
    def __init__(self, *, hidden_dim: int = MESSAGE_HIDDEN_DIM, dropout: float = MESSAGE_DROPOUT) -> None:
        super().__init__()
        self.edge_embedding = nn.Embedding(len(EDGE_TYPE_TO_ID), EDGE_TYPE_EMBED_DIM)
        self.reverse_support_edge_type_ids = tuple(
            EDGE_TYPE_TO_ID[edge_type]
            for edge_type in (
                "time_of",
                "profile_of",
                "status_of",
                "supported_by_turn",
            )
            if edge_type in EDGE_TYPE_TO_ID
        )
        self.input_projection = nn.Linear(NODE_OUTPUT_DIM, hidden_dim)
        self.message_layers = nn.ModuleList(
            [nn.Linear(hidden_dim + EDGE_TYPE_EMBED_DIM, hidden_dim) for _ in range(2)]
        )
        self.update_layers = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(2)]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_inputs: Tensor, edge_src: Tensor, edge_dst: Tensor, edge_type_ids: Tensor) -> Tensor:
        hidden = self.input_projection(node_inputs)
        if edge_src.numel() == 0:
            return hidden
        if self.reverse_support_edge_type_ids and _env_flag("TMCRA_NODE_MEMORY_REVERSE_SUPPORT_MESSAGES", default=False):
            reverse_mask = torch.zeros_like(edge_type_ids, dtype=torch.bool)
            for edge_type_id in self.reverse_support_edge_type_ids:
                reverse_mask = reverse_mask | (edge_type_ids == int(edge_type_id))
            if bool(reverse_mask.any().item()):
                reverse_src = edge_dst.masked_select(reverse_mask)
                reverse_dst = edge_src.masked_select(reverse_mask)
                reverse_types = edge_type_ids.masked_select(reverse_mask)
                edge_src = torch.cat([edge_src, reverse_src], dim=0)
                edge_dst = torch.cat([edge_dst, reverse_dst], dim=0)
                edge_type_ids = torch.cat([edge_type_ids, reverse_types], dim=0)
        _validate_index_values(
            edge_src.detach().cpu().tolist(),
            upper_bound=int(hidden.size(0)),
            index_name="edge_src",
            context="TypedMessagePassing.forward",
            extra={"node_count": int(hidden.size(0)), "edge_count": int(edge_src.numel())},
        )
        _validate_index_values(
            edge_dst.detach().cpu().tolist(),
            upper_bound=int(hidden.size(0)),
            index_name="edge_dst",
            context="TypedMessagePassing.forward",
            extra={"node_count": int(hidden.size(0)), "edge_count": int(edge_dst.numel())},
        )
        _validate_index_values(
            edge_type_ids.detach().cpu().tolist(),
            upper_bound=int(self.edge_embedding.num_embeddings),
            index_name="edge_type_id",
            context="TypedMessagePassing.forward",
            extra={"edge_count": int(edge_type_ids.numel())},
        )
        for message_layer, update_layer, norm_layer in zip(self.message_layers, self.update_layers, self.norm_layers):
            edge_features = self.edge_embedding(edge_type_ids)
            src_hidden = hidden.index_select(0, edge_src)
            messages = F.silu(message_layer(torch.cat([src_hidden, edge_features], dim=-1)))
            messages = messages.to(dtype=hidden.dtype)
            aggregated = torch.zeros_like(hidden)
            aggregated.index_add_(0, edge_dst, messages)
            counts = torch.zeros((hidden.size(0), 1), dtype=hidden.dtype, device=hidden.device)
            counts.index_add_(0, edge_dst, torch.ones((edge_dst.size(0), 1), dtype=hidden.dtype, device=hidden.device))
            aggregated = aggregated / counts.clamp_min(1.0)
            update = F.silu(update_layer(torch.cat([hidden, aggregated], dim=-1)))
            hidden = norm_layer(hidden + self.dropout(update))
        return hidden


class ResidualPairFeatureAdapter(nn.Module):
    def __init__(self, feature_dim: int, *, dropout: float = MESSAGE_DROPOUT) -> None:
        super().__init__()
        hidden_dim = max(32, int(feature_dim) * 2)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.numel() == 0:
            return features
        gate_input = self.dropout(self.norm(features))
        delta = 0.5 * torch.tanh(self.net(gate_input))
        return features * (1.0 + delta)


class EventRecallHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(QUESTION_OUTPUT_DIM + MESSAGE_HIDDEN_DIM + MESSAGE_HIDDEN_DIM + PAIR_FEATURE_DIM, MESSAGE_HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(MESSAGE_HIDDEN_DIM, 1),
        )
        self.feature_residual = nn.Sequential(
            nn.LayerNorm(PAIR_FEATURE_DIM),
            nn.Linear(PAIR_FEATURE_DIM, MESSAGE_HIDDEN_DIM // 2),
            nn.SiLU(),
            nn.Linear(MESSAGE_HIDDEN_DIM // 2, 1),
        )

    def forward(self, question_embedding: Tensor, event_embeddings: Tensor, question_projected: Tensor, pair_features: Tensor) -> Tensor:
        if event_embeddings.dim() == 2:
            question_features = question_embedding.unsqueeze(0).expand(event_embeddings.size(0), -1)
            projected = question_projected.unsqueeze(0).expand(event_embeddings.size(0), -1)
            return (
                self.net(torch.cat([question_features, event_embeddings, projected * event_embeddings, pair_features], dim=-1))
                + self.feature_residual(pair_features)
            ).squeeze(-1)
        if event_embeddings.dim() == 3:
            question_features = question_embedding.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            projected = question_projected.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            return (
                self.net(torch.cat([question_features, event_embeddings, projected * event_embeddings, pair_features], dim=-1))
                + self.feature_residual(pair_features)
            ).squeeze(-1)
        raise ValueError(f"Unsupported event_embeddings rank: {event_embeddings.dim()}")


class EventScoreHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                QUESTION_OUTPUT_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + PAIR_FEATURE_DIM,
                MESSAGE_HIDDEN_DIM,
            ),
            nn.SiLU(),
            nn.Linear(MESSAGE_HIDDEN_DIM, 1),
        )
        self.feature_residual = nn.Sequential(
            nn.LayerNorm(PAIR_FEATURE_DIM),
            nn.Linear(PAIR_FEATURE_DIM, MESSAGE_HIDDEN_DIM // 2),
            nn.SiLU(),
            nn.Linear(MESSAGE_HIDDEN_DIM // 2, 1),
        )

    def forward(
        self,
        question_embedding: Tensor,
        event_embeddings: Tensor,
        support_embeddings: Tensor,
        question_projected: Tensor,
        pair_features: Tensor,
    ) -> Tensor:
        if event_embeddings.dim() == 2:
            question_features = question_embedding.unsqueeze(0).expand(event_embeddings.size(0), -1)
            projected = question_projected.unsqueeze(0).expand(event_embeddings.size(0), -1)
            return (
                self.net(
                    torch.cat(
                        [
                            question_features,
                            event_embeddings,
                            support_embeddings,
                            projected * event_embeddings,
                            projected * support_embeddings,
                            pair_features,
                        ],
                        dim=-1,
                    )
                )
                + self.feature_residual(pair_features)
            ).squeeze(-1)
        if event_embeddings.dim() == 3:
            question_features = question_embedding.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            projected = question_projected.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            return (
                self.net(
                    torch.cat(
                        [
                            question_features,
                            event_embeddings,
                            support_embeddings,
                            projected * event_embeddings,
                            projected * support_embeddings,
                            pair_features,
                        ],
                        dim=-1,
                    )
                )
                + self.feature_residual(pair_features)
            ).squeeze(-1)
        raise ValueError(f"Unsupported event_embeddings rank: {event_embeddings.dim()}")


class EventMatrixRerankBlock(nn.Module):
    def __init__(self, *, hidden_dim: int = MESSAGE_HIDDEN_DIM, num_heads: int = 4, dropout: float = MESSAGE_DROPOUT) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, *, attn_mask: Tensor, event_mask: Tensor) -> Tensor:
        masked_hidden = hidden * event_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        attended, _ = self.attention(
            self.norm1(masked_hidden),
            self.norm1(masked_hidden),
            self.norm1(masked_hidden),
            attn_mask=attn_mask,
            need_weights=False,
        )
        hidden = masked_hidden + self.dropout(attended)
        hidden = hidden * event_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        hidden = hidden + self.dropout(self.ffn(self.norm2(hidden)))
        return hidden * event_mask.unsqueeze(-1).to(dtype=hidden.dtype)


class EventMatrixRerankHead(nn.Module):
    def __init__(self, *, hidden_dim: int = MESSAGE_HIDDEN_DIM, num_heads: int = 4, dropout: float = MESSAGE_DROPOUT) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.pair_feature_projection = nn.Sequential(
            nn.LayerNorm(PAIR_FEATURE_DIM),
            nn.Linear(PAIR_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
        )
        self.token_projection = nn.Linear(QUESTION_OUTPUT_DIM + (hidden_dim * 5), hidden_dim)
        self.blocks = nn.ModuleList(
            [
                EventMatrixRerankBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout),
                EventMatrixRerankBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout),
            ]
        )
        self.relation_bias_projection = nn.Linear(MATRIX_RELATION_FEATURE_DIM, num_heads, bias=False)
        self.output_projection = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        question_embedding: Tensor,
        event_embeddings: Tensor,
        support_embeddings: Tensor,
        question_projected: Tensor,
        pair_features: Tensor,
        relation_features: Tensor,
        event_mask: Tensor,
    ) -> Tensor:
        squeeze_batch = False
        if event_embeddings.dim() == 2:
            question_embedding = question_embedding.unsqueeze(0)
            event_embeddings = event_embeddings.unsqueeze(0)
            support_embeddings = support_embeddings.unsqueeze(0)
            question_projected = question_projected.unsqueeze(0)
            pair_features = pair_features.unsqueeze(0)
            relation_features = relation_features.unsqueeze(0)
            event_mask = event_mask.unsqueeze(0)
            squeeze_batch = True
        if event_embeddings.dim() != 3:
            raise ValueError(f"Unsupported event_embeddings rank: {event_embeddings.dim()}")
        batch_size, event_count, _ = event_embeddings.shape
        pair_projected = self.pair_feature_projection(pair_features)
        question_features = question_embedding.unsqueeze(1).expand(-1, event_count, -1)
        projected = question_projected.unsqueeze(1).expand(-1, event_count, -1)
        hidden = self.token_projection(
            torch.cat(
                [
                    question_features,
                    event_embeddings,
                    support_embeddings,
                    projected * event_embeddings,
                    projected * support_embeddings,
                    pair_projected,
                ],
                dim=-1,
            )
        )
        hidden = hidden * event_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        relation_bias = self.relation_bias_projection(relation_features).permute(0, 3, 1, 2)
        relation_bias = relation_bias.masked_fill((~event_mask).unsqueeze(1).unsqueeze(1), -1e4)
        attn_mask = relation_bias.reshape(batch_size * self.num_heads, event_count, event_count)
        for block in self.blocks:
            hidden = block(hidden, attn_mask=attn_mask, event_mask=event_mask)
        delta_logits = self.output_projection(hidden).squeeze(-1)
        delta_logits = delta_logits * event_mask.to(dtype=delta_logits.dtype)
        return delta_logits[0] if squeeze_batch else delta_logits


class PathMatrixRerankHead(nn.Module):
    def __init__(self, *, hidden_dim: int = MESSAGE_HIDDEN_DIM, num_heads: int = 4, dropout: float = MESSAGE_DROPOUT) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.pair_feature_projection = nn.Sequential(
            nn.LayerNorm(PATH_PAIR_FEATURE_DIM),
            nn.Linear(PATH_PAIR_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
        )
        self.token_projection = nn.Linear(QUESTION_OUTPUT_DIM + (hidden_dim * 5), hidden_dim)
        self.blocks = nn.ModuleList(
            [
                EventMatrixRerankBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout),
                EventMatrixRerankBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout),
            ]
        )
        self.relation_bias_projection = nn.Linear(PATH_MATRIX_RELATION_FEATURE_DIM, num_heads, bias=False)
        self.output_projection = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        question_embedding: Tensor,
        event_embeddings: Tensor,
        support_embeddings: Tensor,
        question_projected: Tensor,
        pair_features: Tensor,
        relation_features: Tensor,
        path_mask: Tensor,
    ) -> Tensor:
        squeeze_batch = False
        if event_embeddings.dim() == 2:
            question_embedding = question_embedding.unsqueeze(0)
            event_embeddings = event_embeddings.unsqueeze(0)
            support_embeddings = support_embeddings.unsqueeze(0)
            question_projected = question_projected.unsqueeze(0)
            pair_features = pair_features.unsqueeze(0)
            relation_features = relation_features.unsqueeze(0)
            path_mask = path_mask.unsqueeze(0)
            squeeze_batch = True
        if event_embeddings.dim() != 3:
            raise ValueError(f"Unsupported event_embeddings rank: {event_embeddings.dim()}")
        batch_size, path_count, _ = event_embeddings.shape
        pair_projected = self.pair_feature_projection(pair_features)
        question_features = question_embedding.unsqueeze(1).expand(-1, path_count, -1)
        projected = question_projected.unsqueeze(1).expand(-1, path_count, -1)
        hidden = self.token_projection(
            torch.cat(
                [
                    question_features,
                    event_embeddings,
                    support_embeddings,
                    projected * event_embeddings,
                    projected * support_embeddings,
                    pair_projected,
                ],
                dim=-1,
            )
        )
        hidden = hidden * path_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        relation_bias = self.relation_bias_projection(relation_features).permute(0, 3, 1, 2)
        relation_bias = relation_bias.masked_fill((~path_mask).unsqueeze(1).unsqueeze(1), -1e4)
        attn_mask = relation_bias.reshape(batch_size * self.num_heads, path_count, path_count)
        for block in self.blocks:
            hidden = block(hidden, attn_mask=attn_mask, event_mask=path_mask)
        delta_logits = self.output_projection(hidden).squeeze(-1)
        delta_logits = delta_logits * path_mask.to(dtype=delta_logits.dtype)
        return delta_logits[0] if squeeze_batch else delta_logits


class PathScoreHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.path_type_embedding = nn.Embedding(len(PATH_TYPE_TO_ID), EDGE_TYPE_EMBED_DIM)
        self.net = nn.Sequential(
            nn.Linear(
                QUESTION_OUTPUT_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + EDGE_TYPE_EMBED_DIM
                + PATH_PAIR_FEATURE_DIM,
                MESSAGE_HIDDEN_DIM,
            ),
            nn.SiLU(),
            nn.Linear(MESSAGE_HIDDEN_DIM, 1),
        )
        self.feature_residual = nn.Sequential(
            nn.LayerNorm(PATH_PAIR_FEATURE_DIM),
            nn.Linear(PATH_PAIR_FEATURE_DIM, MESSAGE_HIDDEN_DIM // 2),
            nn.SiLU(),
            nn.Linear(MESSAGE_HIDDEN_DIM // 2, 1),
        )

    def forward(
        self,
        question_embedding: Tensor,
        event_embeddings: Tensor,
        support_embeddings: Tensor,
        question_projected: Tensor,
        path_type_ids: Tensor,
        pair_features: Tensor,
    ) -> Tensor:
        type_embeddings = self.path_type_embedding(path_type_ids)
        if event_embeddings.dim() == 2:
            question_features = question_embedding.unsqueeze(0).expand(event_embeddings.size(0), -1)
            projected = question_projected.unsqueeze(0).expand(event_embeddings.size(0), -1)
            return (
                self.net(
                    torch.cat(
                        [
                            question_features,
                            event_embeddings,
                            support_embeddings,
                            projected * event_embeddings,
                            projected * support_embeddings,
                            type_embeddings,
                            pair_features,
                        ],
                        dim=-1,
                    )
                )
                + self.feature_residual(pair_features)
            ).squeeze(-1)
        if event_embeddings.dim() == 3:
            question_features = question_embedding.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            projected = question_projected.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            return (
                self.net(
                    torch.cat(
                        [
                            question_features,
                            event_embeddings,
                            support_embeddings,
                            projected * event_embeddings,
                            projected * support_embeddings,
                            type_embeddings,
                            pair_features,
                        ],
                        dim=-1,
                    )
                )
                + self.feature_residual(pair_features)
            ).squeeze(-1)
        raise ValueError(f"Unsupported event_embeddings rank: {event_embeddings.dim()}")


class EventSubgraphAttentionRefiner(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = MESSAGE_HIDDEN_DIM,
        attention_layers: int = EVENT_SUBGRAPH_ATTENTION_LAYERS,
        attention_heads: int = EVENT_SUBGRAPH_ATTENTION_HEADS,
        dropout: float = MESSAGE_DROPOUT,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.role_embedding = nn.Embedding(2 + len(EVENT_SUPPORT_PATH_TYPE_ORDER), hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                _LightweightSelfAttentionBlock(
                    embed_dim=hidden_dim,
                    num_heads=attention_heads,
                    dropout=dropout,
                )
                for _ in range(max(1, int(attention_layers)))
            ]
        )
        self.event_output_norm = nn.LayerNorm(hidden_dim)
        self.support_pool_norm = nn.LayerNorm(hidden_dim)
        self.support_pool_projection = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        question_projected: Tensor,
        event_embeddings: Tensor,
        support_slot_embeddings: Tensor,
        support_slot_mask: Tensor,
        event_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        squeeze_batch = False
        if event_embeddings.dim() == 2:
            question_projected = question_projected.unsqueeze(0)
            event_embeddings = event_embeddings.unsqueeze(0)
            support_slot_embeddings = support_slot_embeddings.unsqueeze(0)
            support_slot_mask = support_slot_mask.unsqueeze(0)
            event_mask = event_mask.unsqueeze(0)
            squeeze_batch = True
        if event_embeddings.dim() != 3 or support_slot_embeddings.dim() != 4:
            raise ValueError(
                "EventSubgraphAttentionRefiner.forward expects "
                f"event_embeddings rank 3 and support_slot_embeddings rank 4, got "
                f"{event_embeddings.dim()} and {support_slot_embeddings.dim()}"
            )
        batch_size, event_count, hidden_dim = event_embeddings.shape
        slot_count = int(support_slot_embeddings.size(2))
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"EventSubgraphAttentionRefiner hidden_dim mismatch: expected {self.hidden_dim}, got {hidden_dim}"
            )
        question_token = question_projected.unsqueeze(1).expand(-1, event_count, -1).unsqueeze(2)
        event_token = event_embeddings.unsqueeze(2)
        hidden = torch.cat([question_token, event_token, support_slot_embeddings], dim=2)
        role_ids = torch.arange(2 + slot_count, dtype=torch.long, device=event_embeddings.device)
        hidden = hidden + self.role_embedding(role_ids).view(1, 1, 2 + slot_count, hidden_dim)
        hidden = self.input_dropout(self.input_norm(hidden))
        token_mask = torch.cat(
            [
                event_mask.unsqueeze(-1),
                event_mask.unsqueeze(-1),
                support_slot_mask,
            ],
            dim=-1,
        )
        flat_hidden = hidden.view(batch_size * event_count, 2 + slot_count, hidden_dim)
        flat_mask = token_mask.view(batch_size * event_count, 2 + slot_count)
        flat_valid = event_mask.reshape(batch_size * event_count)
        output_hidden = flat_hidden.clone()
        if bool(flat_valid.any().item()):
            valid_indices = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
            valid_hidden = flat_hidden.index_select(0, valid_indices)
            valid_mask = flat_mask.index_select(0, valid_indices)
            for block in self.blocks:
                valid_hidden = block(valid_hidden, valid_mask)
            output_hidden.index_copy_(0, valid_indices, valid_hidden)
        hidden = output_hidden.view(batch_size, event_count, 2 + slot_count, hidden_dim)
        refined_event_embeddings = self.event_output_norm(hidden[:, :, 1, :])
        support_hidden = hidden[:, :, 2:, :]
        support_logits = self.support_pool_projection(self.support_pool_norm(support_hidden)).squeeze(-1)
        support_logits = support_logits.masked_fill(~support_slot_mask, -1e4)
        has_support = support_slot_mask.any(dim=-1, keepdim=True)
        support_logits = torch.where(has_support, support_logits, torch.zeros_like(support_logits))
        support_weights = torch.softmax(support_logits, dim=-1)
        support_weights = support_weights.masked_fill(~support_slot_mask, 0.0)
        support_weights = support_weights / support_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        refined_support_embeddings = (support_hidden * support_weights.unsqueeze(-1)).sum(dim=2)
        has_support = support_slot_mask.any(dim=-1)
        refined_support_embeddings = refined_support_embeddings * has_support.unsqueeze(-1).to(
            dtype=refined_support_embeddings.dtype
        )
        refined_event_embeddings = refined_event_embeddings * event_mask.unsqueeze(-1).to(dtype=refined_event_embeddings.dtype)
        if squeeze_batch:
            return refined_event_embeddings[0], refined_support_embeddings[0]
        return refined_event_embeddings, refined_support_embeddings


class TemporalHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                QUESTION_OUTPUT_DIM + MESSAGE_HIDDEN_DIM + MESSAGE_HIDDEN_DIM + MESSAGE_HIDDEN_DIM + MESSAGE_HIDDEN_DIM,
                MESSAGE_HIDDEN_DIM,
            ),
            nn.SiLU(),
            nn.Linear(MESSAGE_HIDDEN_DIM, 1),
        )

    def forward(
        self,
        question_embedding: Tensor,
        event_embeddings: Tensor,
        time_embeddings: Tensor,
        question_projected: Tensor,
    ) -> Tensor:
        if event_embeddings.dim() == 2:
            question_features = question_embedding.unsqueeze(0).expand(event_embeddings.size(0), -1)
            projected = question_projected.unsqueeze(0).expand(event_embeddings.size(0), -1)
            return self.net(
                torch.cat(
                    [
                        question_features,
                        event_embeddings,
                        time_embeddings,
                        projected * event_embeddings,
                        projected * time_embeddings,
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
        if event_embeddings.dim() == 3:
            question_features = question_embedding.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            projected = question_projected.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            return self.net(
                torch.cat(
                    [
                        question_features,
                        event_embeddings,
                        time_embeddings,
                        projected * event_embeddings,
                        projected * time_embeddings,
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
        raise ValueError(f"Unsupported event_embeddings rank: {event_embeddings.dim()}")


class AnswerTypeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                QUESTION_OUTPUT_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + ANSWER_CALIBRATION_FEATURE_DIM,
                MESSAGE_HIDDEN_DIM,
            ),
            nn.SiLU(),
            nn.Linear(MESSAGE_HIDDEN_DIM, len(ANSWER_TYPE_TO_ID)),
        )

    def forward(
        self,
        question_embedding: Tensor,
        pooled_event_embedding: Tensor,
        pooled_support_embedding: Tensor,
        question_projected: Tensor,
        calibration_features: Tensor,
    ) -> Tensor:
        return self.net(
            torch.cat(
                [
                    question_embedding,
                    pooled_event_embedding,
                    pooled_support_embedding,
                    question_projected * pooled_event_embedding,
                    question_projected * pooled_support_embedding,
                    calibration_features,
                ],
                dim=-1,
            )
        )


class AnswerPlanHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                QUESTION_OUTPUT_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + PAIR_FEATURE_DIM
                + ANSWER_PLAN_FEATURE_DIM,
                MESSAGE_HIDDEN_DIM,
            ),
            nn.SiLU(),
            nn.Dropout(MESSAGE_DROPOUT),
            nn.Linear(MESSAGE_HIDDEN_DIM, len(ANSWER_PLAN_OUTPUTS)),
        )

    def forward(
        self,
        question_embedding: Tensor,
        event_embeddings: Tensor,
        support_embeddings: Tensor,
        question_projected: Tensor,
        event_pair_features: Tensor,
        answer_plan_features: Tensor,
    ) -> Tensor:
        if event_embeddings.dim() == 2:
            question_features = question_embedding.unsqueeze(0).expand(event_embeddings.size(0), -1)
            projected = question_projected.unsqueeze(0).expand(event_embeddings.size(0), -1)
        elif event_embeddings.dim() == 3:
            question_features = question_embedding.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            projected = question_projected.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
        else:
            raise ValueError(f"Unsupported event_embeddings rank: {event_embeddings.dim()}")
        return self.net(
            torch.cat(
                [
                    question_features,
                    event_embeddings,
                    support_embeddings,
                    projected * event_embeddings,
                    projected * support_embeddings,
                    event_pair_features,
                    answer_plan_features,
                ],
                dim=-1,
            )
        )


class EventDistractorHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                QUESTION_OUTPUT_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + PAIR_FEATURE_DIM
                + EVENT_DISTRACTOR_FEATURE_DIM,
                MESSAGE_HIDDEN_DIM,
            ),
            nn.SiLU(),
        )
        self.output_projection = nn.Linear(MESSAGE_HIDDEN_DIM, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        question_embedding: Tensor,
        event_embeddings: Tensor,
        support_embeddings: Tensor,
        question_projected: Tensor,
        pair_features: Tensor,
        distractor_features: Tensor,
    ) -> Tensor:
        if event_embeddings.dim() == 2:
            question_features = question_embedding.unsqueeze(0).expand(event_embeddings.size(0), -1)
            projected = question_projected.unsqueeze(0).expand(event_embeddings.size(0), -1)
            hidden = self.net(
                torch.cat(
                    [
                        question_features,
                        event_embeddings,
                        support_embeddings,
                        projected * event_embeddings,
                        projected * support_embeddings,
                        pair_features,
                        distractor_features,
                    ],
                    dim=-1,
                )
            )
            return self.output_projection(hidden).squeeze(-1)
        if event_embeddings.dim() == 3:
            question_features = question_embedding.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            projected = question_projected.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            hidden = self.net(
                torch.cat(
                    [
                        question_features,
                        event_embeddings,
                        support_embeddings,
                        projected * event_embeddings,
                        projected * support_embeddings,
                        pair_features,
                        distractor_features,
                    ],
                    dim=-1,
                )
            )
            return self.output_projection(hidden).squeeze(-1)
        raise ValueError(f"Unsupported event_embeddings rank: {event_embeddings.dim()}")


class EventCalibrationHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                QUESTION_OUTPUT_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + PAIR_FEATURE_DIM
                + EVENT_RUNTIME_CALIBRATION_FEATURE_DIM,
                MESSAGE_HIDDEN_DIM,
            ),
            nn.SiLU(),
        )
        self.output_projection = nn.Linear(MESSAGE_HIDDEN_DIM, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        question_embedding: Tensor,
        event_embeddings: Tensor,
        support_embeddings: Tensor,
        question_projected: Tensor,
        pair_features: Tensor,
        calibration_features: Tensor,
    ) -> Tensor:
        if event_embeddings.dim() == 2:
            question_features = question_embedding.unsqueeze(0).expand(event_embeddings.size(0), -1)
            projected = question_projected.unsqueeze(0).expand(event_embeddings.size(0), -1)
            hidden = self.net(
                torch.cat(
                    [
                        question_features,
                        event_embeddings,
                        support_embeddings,
                        projected * event_embeddings,
                        projected * support_embeddings,
                        pair_features,
                        calibration_features,
                    ],
                    dim=-1,
                )
            )
            return self.output_projection(hidden).squeeze(-1)
        if event_embeddings.dim() == 3:
            question_features = question_embedding.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            projected = question_projected.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            hidden = self.net(
                torch.cat(
                    [
                        question_features,
                        event_embeddings,
                        support_embeddings,
                        projected * event_embeddings,
                        projected * support_embeddings,
                        pair_features,
                        calibration_features,
                    ],
                    dim=-1,
                )
            )
            return self.output_projection(hidden).squeeze(-1)
        raise ValueError(f"Unsupported event_embeddings rank: {event_embeddings.dim()}")


class PathCalibrationHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                QUESTION_OUTPUT_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + PATH_PAIR_FEATURE_DIM
                + PATH_RUNTIME_CALIBRATION_FEATURE_DIM,
                MESSAGE_HIDDEN_DIM,
            ),
            nn.SiLU(),
        )
        self.output_projection = nn.Linear(MESSAGE_HIDDEN_DIM, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        question_embedding: Tensor,
        event_embeddings: Tensor,
        support_embeddings: Tensor,
        question_projected: Tensor,
        pair_features: Tensor,
        calibration_features: Tensor,
    ) -> Tensor:
        if event_embeddings.dim() == 2:
            question_features = question_embedding.unsqueeze(0).expand(event_embeddings.size(0), -1)
            projected = question_projected.unsqueeze(0).expand(event_embeddings.size(0), -1)
            hidden = self.net(
                torch.cat(
                    [
                        question_features,
                        event_embeddings,
                        support_embeddings,
                        projected * event_embeddings,
                        projected * support_embeddings,
                        pair_features,
                        calibration_features,
                    ],
                    dim=-1,
                )
            )
            return self.output_projection(hidden).squeeze(-1)
        if event_embeddings.dim() == 3:
            question_features = question_embedding.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            projected = question_projected.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            hidden = self.net(
                torch.cat(
                    [
                        question_features,
                        event_embeddings,
                        support_embeddings,
                        projected * event_embeddings,
                        projected * support_embeddings,
                        pair_features,
                        calibration_features,
                    ],
                    dim=-1,
                )
            )
            return self.output_projection(hidden).squeeze(-1)
        raise ValueError(f"Unsupported event_embeddings rank: {event_embeddings.dim()}")


class FinalEventFusionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                QUESTION_OUTPUT_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + PAIR_FEATURE_DIM
                + FINAL_EVENT_FUSION_FEATURE_DIM,
                MESSAGE_HIDDEN_DIM,
            ),
            nn.SiLU(),
        )
        self.output_projection = nn.Linear(MESSAGE_HIDDEN_DIM, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        question_embedding: Tensor,
        event_embeddings: Tensor,
        support_embeddings: Tensor,
        question_projected: Tensor,
        pair_features: Tensor,
        fusion_features: Tensor,
    ) -> Tensor:
        if event_embeddings.dim() == 2:
            question_features = question_embedding.unsqueeze(0).expand(event_embeddings.size(0), -1)
            projected = question_projected.unsqueeze(0).expand(event_embeddings.size(0), -1)
            hidden = self.net(
                torch.cat(
                    [
                        question_features,
                        event_embeddings,
                        support_embeddings,
                        projected * event_embeddings,
                        projected * support_embeddings,
                        pair_features,
                        fusion_features,
                    ],
                    dim=-1,
                )
            )
            return self.output_projection(hidden).squeeze(-1)
        if event_embeddings.dim() == 3:
            question_features = question_embedding.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            projected = question_projected.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            hidden = self.net(
                torch.cat(
                    [
                        question_features,
                        event_embeddings,
                        support_embeddings,
                        projected * event_embeddings,
                        projected * support_embeddings,
                        pair_features,
                        fusion_features,
                    ],
                    dim=-1,
                )
            )
            return self.output_projection(hidden).squeeze(-1)
        raise ValueError(f"Unsupported event_embeddings rank: {event_embeddings.dim()}")


class FinalPathFusionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                QUESTION_OUTPUT_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + PATH_PAIR_FEATURE_DIM
                + FINAL_PATH_FUSION_FEATURE_DIM,
                MESSAGE_HIDDEN_DIM,
            ),
            nn.SiLU(),
        )
        self.output_projection = nn.Linear(MESSAGE_HIDDEN_DIM, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        question_embedding: Tensor,
        event_embeddings: Tensor,
        support_embeddings: Tensor,
        question_projected: Tensor,
        pair_features: Tensor,
        fusion_features: Tensor,
    ) -> Tensor:
        if event_embeddings.dim() == 2:
            question_features = question_embedding.unsqueeze(0).expand(event_embeddings.size(0), -1)
            projected = question_projected.unsqueeze(0).expand(event_embeddings.size(0), -1)
            hidden = self.net(
                torch.cat(
                    [
                        question_features,
                        event_embeddings,
                        support_embeddings,
                        projected * event_embeddings,
                        projected * support_embeddings,
                        pair_features,
                        fusion_features,
                    ],
                    dim=-1,
                )
            )
            return self.output_projection(hidden).squeeze(-1)
        if event_embeddings.dim() == 3:
            question_features = question_embedding.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            projected = question_projected.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            hidden = self.net(
                torch.cat(
                    [
                        question_features,
                        event_embeddings,
                        support_embeddings,
                        projected * event_embeddings,
                        projected * support_embeddings,
                        pair_features,
                        fusion_features,
                    ],
                    dim=-1,
                )
            )
            return self.output_projection(hidden).squeeze(-1)
        raise ValueError(f"Unsupported event_embeddings rank: {event_embeddings.dim()}")


class MemoryTunnelHead(nn.Module):
    def __init__(self, pair_feature_dim: int, tunnel_feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                QUESTION_OUTPUT_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + MESSAGE_HIDDEN_DIM
                + int(pair_feature_dim)
                + int(tunnel_feature_dim),
                MESSAGE_HIDDEN_DIM,
            ),
            nn.SiLU(),
        )
        self.output_projection = nn.Linear(MESSAGE_HIDDEN_DIM, 2)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        question_embedding: Tensor,
        event_embeddings: Tensor,
        support_embeddings: Tensor,
        question_projected: Tensor,
        pair_features: Tensor,
        tunnel_features: Tensor,
    ) -> Tensor:
        if event_embeddings.dim() == 2:
            question_features = question_embedding.unsqueeze(0).expand(event_embeddings.size(0), -1)
            projected = question_projected.unsqueeze(0).expand(event_embeddings.size(0), -1)
            hidden = self.net(
                torch.cat(
                    [
                        question_features,
                        event_embeddings,
                        support_embeddings,
                        projected * event_embeddings,
                        projected * support_embeddings,
                        pair_features,
                        tunnel_features,
                    ],
                    dim=-1,
                )
            )
            return self.output_projection(hidden)
        if event_embeddings.dim() == 3:
            question_features = question_embedding.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            projected = question_projected.unsqueeze(1).expand(-1, event_embeddings.size(1), -1)
            hidden = self.net(
                torch.cat(
                    [
                        question_features,
                        event_embeddings,
                        support_embeddings,
                        projected * event_embeddings,
                        projected * support_embeddings,
                        pair_features,
                        tunnel_features,
                    ],
                    dim=-1,
                )
            )
            return self.output_projection(hidden)
        raise ValueError(f"Unsupported event_embeddings rank: {event_embeddings.dim()}")


class LocomoNodeMemoryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.question_encoder = QuestionEncoder()
        self.question_intent_head = QuestionIntentHead()
        self.memory_router_head = MemoryRouterHead()
        self.node_encoder = NodeEncoder()
        self.message_passing = TypedMessagePassing()
        self.question_projection = nn.Linear(QUESTION_OUTPUT_DIM, MESSAGE_HIDDEN_DIM)
        self.event_pair_adapter = ResidualPairFeatureAdapter(PAIR_FEATURE_DIM)
        self.path_pair_adapter = ResidualPairFeatureAdapter(PATH_PAIR_FEATURE_DIM)
        self.event_recall_head = EventRecallHead()
        self.event_subgraph_refiner = EventSubgraphAttentionRefiner()
        self.event_head = EventScoreHead()
        self.event_matrix_head = EventMatrixRerankHead()
        self.event_distractor_head = EventDistractorHead()
        self.path_head = PathScoreHead()
        self.path_matrix_head = PathMatrixRerankHead()
        self.temporal_head = TemporalHead()
        self.answer_type_head = AnswerTypeHead()
        self.answer_plan_head = AnswerPlanHead()
        self.event_calibration_head = EventCalibrationHead()
        self.path_calibration_head = PathCalibrationHead()
        self.event_tunnel_head = MemoryTunnelHead(PAIR_FEATURE_DIM, EVENT_TUNNEL_FEATURE_DIM)
        self.path_tunnel_head = MemoryTunnelHead(PATH_PAIR_FEATURE_DIM, PATH_TUNNEL_FEATURE_DIM)
        self.final_event_fusion_head = FinalEventFusionHead()
        self.final_path_fusion_head = FinalPathFusionHead()
        self._last_node_token_role_aux_loss: Tensor | None = None

    def encode_graph(self, graph_tensors: Mapping[str, Any]) -> Tensor:
        node_inputs = self.node_encoder(
            graph_tensors["node_texts"],
            graph_tensors["node_type_ids"],
            graph_tensors["node_scalar_features"],
            token_hash_indices_batch=graph_tensors.get("node_text_hash_indices"),
        )
        self._last_node_token_role_aux_loss = self.node_encoder.pop_last_role_aux_loss()
        return self.message_passing(
            node_inputs,
            graph_tensors["edge_src"],
            graph_tensors["edge_dst"],
            graph_tensors["edge_type_ids"],
        )

    def pop_last_node_token_role_aux_loss(self) -> Tensor | None:
        value = self._last_node_token_role_aux_loss
        self._last_node_token_role_aux_loss = None
        return value

    def score_example(
        self,
        graph_tensors: Mapping[str, Any],
        example: QueryTrainingExample,
        *,
        event_rerank_mode: str = "matrix",
        matrix_event_top_k: int = DEFAULT_MATRIX_EVENT_TOP_K,
        event_pair_feature_mode: str = "full",
    ) -> Dict[str, Any]:
        return self.score_examples(
            graph_tensors,
            [example],
            event_rerank_mode=event_rerank_mode,
            matrix_event_top_k=matrix_event_top_k,
            event_pair_feature_mode=event_pair_feature_mode,
        )[0]

    def score_example_with_graph_encoding(
        self,
        graph_tensors: Mapping[str, Any],
        node_hidden: Tensor,
        example: QueryTrainingExample,
        *,
        event_rerank_mode: str = "matrix",
        matrix_event_top_k: int = DEFAULT_MATRIX_EVENT_TOP_K,
        event_pair_feature_mode: str = "full",
        recall_only: bool = False,
        precomputed_recall: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return self.score_examples_with_graph_encoding(
            graph_tensors,
            node_hidden,
            [example],
            event_rerank_mode=event_rerank_mode,
            matrix_event_top_k=matrix_event_top_k,
            event_pair_feature_mode=event_pair_feature_mode,
            recall_only=recall_only,
            precomputed_recall_batch=[dict(precomputed_recall)] if precomputed_recall is not None else None,
        )[0]

    def score_examples(
        self,
        graph_tensors: Mapping[str, Any],
        examples: Sequence[QueryTrainingExample],
        *,
        event_rerank_mode: str = "matrix",
        matrix_event_top_k: int = DEFAULT_MATRIX_EVENT_TOP_K,
        event_pair_feature_mode: str = "full",
    ) -> List[Dict[str, Any]]:
        if not examples:
            return []
        node_hidden = self.encode_graph(graph_tensors)
        return self.score_examples_with_graph_encoding(
            graph_tensors,
            node_hidden,
            examples,
            event_rerank_mode=event_rerank_mode,
            matrix_event_top_k=matrix_event_top_k,
            event_pair_feature_mode=event_pair_feature_mode,
        )

    def score_examples_with_graph_encoding(
        self,
        graph_tensors: Mapping[str, Any],
        node_hidden: Tensor,
        examples: Sequence[QueryTrainingExample],
        *,
        event_rerank_mode: str = "matrix",
        matrix_event_top_k: int = DEFAULT_MATRIX_EVENT_TOP_K,
        event_pair_feature_mode: str = "full",
        recall_only: bool = False,
        precomputed_recall_batch: Sequence[Mapping[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        if not examples:
            return []
        graph_tensors = dict(_ensure_graph_scoring_feature_cache(graph_tensors))
        device = graph_tensors["node_type_ids"].device
        node_id_to_index = graph_tensors["node_id_to_index"]
        node_lookup = dict(graph_tensors.get("node_by_id", {}) or {})
        event_support_lookup = dict(graph_tensors.get("event_support_lookup", {}) or {})
        event_scoring_features_by_id = dict(graph_tensors.get("event_scoring_features_by_id", {}) or {})
        node_scoring_features_by_id = dict(graph_tensors.get("node_scoring_features_by_id", {}) or {})
        path_scoring_features_by_id = dict(graph_tensors.get("path_scoring_features_by_id", {}) or {})
        question_features_batch = [dict(example.question_features or {}) for example in examples]
        question_batch = self.question_encoder.forward_batch(
            [example.question for example in examples],
            question_features_batch,
            token_hash_indices_batch=[example.question_hash_indices for example in examples],
            feature_indices=[int(example.question_feature_index or 0) for example in examples],
            query_type_indices=[int(example.question_query_type_index or 0) for example in examples],
        )
        memory_router_logits_batch = self.memory_router_head(question_batch)
        question_text_embeddings = self.question_encoder.pop_last_text_embeddings()
        question_token_role_aux_loss = self.question_encoder.pop_last_role_aux_loss()
        question_role_predictions = self.question_encoder.pop_last_role_predictions() or {}
        question_intent_outputs = (
            self.question_intent_head(question_text_embeddings)
            if isinstance(question_text_embeddings, Tensor) and question_text_embeddings.numel() > 0
            else {
                "semantic_logits": question_batch.new_zeros((len(examples), len(SEMANTIC_SLOTS))),
                "status_logits": question_batch.new_zeros((len(examples), len(TARGET_STATUSES))),
                "time_granularity_logits": question_batch.new_zeros((len(examples), len(TIME_GRANULARITIES))),
                "temporal_logits": question_batch.new_zeros((len(examples),)),
            }
        )
        predicted_question_role_ids = question_role_predictions.get("predicted_role_ids")
        predicted_question_role_mask = question_role_predictions.get("token_mask")
        if isinstance(predicted_question_role_ids, Tensor) and isinstance(predicted_question_role_mask, Tensor):
            learned_question_features_batch: List[Dict[str, Any]] = []
            for batch_index, example in enumerate(examples):
                token_count = min(
                    max(1, len(list(example.question_hash_indices or []))),
                    QUESTION_MAX_TOKENS,
                )
                aligned_tokens = _align_sequence_length(
                    _resolved_text_tokens(example.question),
                    target_length=token_count,
                    fill_value="__empty__",
                ) or ["__empty__"]
                role_ids = predicted_question_role_ids[batch_index, : len(aligned_tokens)].detach().cpu().tolist()
                token_mask_values = predicted_question_role_mask[batch_index, : len(aligned_tokens)].detach().cpu().tolist()
                learned_question_features_batch.append(
                    _merge_learned_question_features_from_token_roles(
                        example.question,
                        question_features_batch[batch_index],
                        tokens=aligned_tokens,
                        predicted_role_ids=role_ids,
                        token_mask=token_mask_values,
                    )
                )
            question_features_batch = learned_question_features_batch
        learned_intent_question_features_batch: List[Dict[str, Any]] = []
        for batch_index in range(len(examples)):
            learned_intent_question_features_batch.append(
                _merge_learned_question_features_from_intent_logits(
                    question_features_batch[batch_index],
                    semantic_logits=question_intent_outputs["semantic_logits"][batch_index],
                    status_logits=question_intent_outputs["status_logits"][batch_index],
                    time_granularity_logits=question_intent_outputs["time_granularity_logits"][batch_index],
                    temporal_logit=question_intent_outputs["temporal_logits"][batch_index],
                )
            )
        question_features_batch = learned_intent_question_features_batch
        prepared_question_batch = [
            _prepare_question_scoring_features(example.question, question_features_batch[batch_index])
            for batch_index, example in enumerate(examples)
        ]
        node_token_role_aux_loss = self.pop_last_node_token_role_aux_loss()
        projected_batch = self.question_projection(question_batch)
        batch_size = len(examples)
        if precomputed_recall_batch is not None and len(precomputed_recall_batch) != batch_size:
            raise ValueError("precomputed recall batch size must match examples")
        resolved_event_rerank_mode = _normalize_event_rerank_mode(event_rerank_mode)
        resolved_event_pair_feature_mode = _normalize_event_pair_feature_mode(event_pair_feature_mode)
        resolved_matrix_event_top_k = max(1, int(matrix_event_top_k or DEFAULT_MATRIX_EVENT_TOP_K))

        all_event_ids = [event_id for event_id in list(graph_tensors.get("event_node_ids", []) or []) if event_id in node_id_to_index]
        event_node_ids_by_conversation = {
            clean_text(conversation_id): [
                event_id for event_id in list(event_ids or []) if clean_text(event_id) in node_id_to_index
            ]
            for conversation_id, event_ids in dict(graph_tensors.get("event_node_ids_by_conversation", {}) or {}).items()
            if clean_text(conversation_id)
        }
        recall_event_ids_batch: List[List[str]] = []
        recall_event_lengths: List[int] = []
        max_recall_event_count = 0
        for batch_index, example in enumerate(examples):
            precomputed_recall = (
                dict(precomputed_recall_batch[batch_index])
                if precomputed_recall_batch is not None
                else {}
            )
            recall_event_ids = [
                clean_text(event_id)
                for event_id in list(precomputed_recall.get("recall_event_ids", []) or [])
                if clean_text(event_id) in node_id_to_index
            ]
            if not recall_event_ids:
                recall_event_ids = list(event_node_ids_by_conversation.get(example.conversation_id, all_event_ids))
            if not recall_event_ids:
                recall_event_ids = list(all_event_ids)
            recall_event_ids_batch.append(recall_event_ids)
            recall_event_lengths.append(len(recall_event_ids))
            max_recall_event_count = max(max_recall_event_count, len(recall_event_ids))
        recall_event_logits_batch = node_hidden.new_zeros((batch_size, max_recall_event_count))
        recall_event_mask = torch.zeros((batch_size, max_recall_event_count), dtype=torch.bool, device=device)
        recall_pair_features_batch = node_hidden.new_zeros((batch_size, max_recall_event_count, PAIR_FEATURE_DIM))
        recall_event_index_lookup_batch = [
            {event_id: index for index, event_id in enumerate(recall_event_ids)}
            for recall_event_ids in recall_event_ids_batch
        ]
        if max_recall_event_count > 0 and precomputed_recall_batch is not None:
            for batch_index, recall_event_ids in enumerate(recall_event_ids_batch):
                recall_count = len(recall_event_ids)
                if recall_count <= 0:
                    continue
                precomputed_recall = dict(precomputed_recall_batch[batch_index])
                logits = precomputed_recall.get("recall_event_logits")
                pair_features = precomputed_recall.get("recall_pair_features")
                if not isinstance(logits, Tensor) or tuple(logits.shape) != (recall_count,):
                    raise ValueError("precomputed recall logits do not match recall event ids")
                if not isinstance(pair_features, Tensor) or tuple(pair_features.shape) != (recall_count, PAIR_FEATURE_DIM):
                    raise ValueError("precomputed recall pair features do not match recall event ids")
                recall_event_logits_batch[batch_index, :recall_count] = logits.to(
                    device=device,
                    dtype=node_hidden.dtype,
                )
                recall_pair_features_batch[batch_index, :recall_count] = pair_features.to(
                    device=device,
                    dtype=node_hidden.dtype,
                )
                recall_event_mask[batch_index, :recall_count] = True
        elif max_recall_event_count > 0:
            padded_recall_indices = torch.zeros((batch_size, max_recall_event_count), dtype=torch.long, device=device)
            for batch_index, recall_event_ids in enumerate(recall_event_ids_batch):
                if not recall_event_ids:
                    continue
                recall_index_values = [node_id_to_index[event_id] for event_id in recall_event_ids]
                _validate_index_values(
                    recall_index_values,
                    upper_bound=int(node_hidden.size(0)),
                    index_name="recall_event_index",
                    context="LocomoNodeMemoryModel.score_examples_with_graph_encoding",
                    extra={
                        "stage": "recall",
                        "conversation_id": examples[batch_index].conversation_id,
                        "question_id": examples[batch_index].question_id,
                        "event_count": len(recall_event_ids),
                    },
                )
                recall_indices = torch.tensor(
                    recall_index_values,
                    dtype=torch.long,
                    device=device,
                )
                recall_count = int(recall_indices.numel())
                padded_recall_indices[batch_index, :recall_count] = recall_indices
                recall_event_mask[batch_index, :recall_count] = True
                recall_pair_features_batch[batch_index, :recall_count] = torch.tensor(
                    [
                        _event_pair_feature_values(
                            examples[batch_index].question,
                            question_features_batch[batch_index],
                            node_lookup.get(event_id, {}),
                            support_payload=event_support_lookup.get(event_id, {}),
                            prepared_question=prepared_question_batch[batch_index],
                            prepared_event=event_scoring_features_by_id.get(event_id),
                        )
                        for event_id in recall_event_ids
                    ],
                    dtype=node_hidden.dtype,
                    device=device,
                )
            recall_pair_features_batch = _apply_event_pair_feature_mode(
                recall_pair_features_batch,
                resolved_event_pair_feature_mode,
            )
            recall_event_embeddings = node_hidden.index_select(0, padded_recall_indices.reshape(-1)).view(
                batch_size,
                max_recall_event_count,
                -1,
            )
            recall_event_logits_batch = self.event_recall_head(
                question_batch,
                recall_event_embeddings,
                projected_batch,
                recall_pair_features_batch,
            ).masked_fill(~recall_event_mask, 0.0)
        recall_event_prob_batch = (
            torch.sigmoid(recall_event_logits_batch).masked_fill(~recall_event_mask, 0.0)
            if max_recall_event_count > 0
            else node_hidden.new_zeros((batch_size, 0))
        )
        if recall_only:
            return [
                {
                    "recall_event_ids": list(recall_event_ids_batch[batch_index]),
                    "recall_event_logits": recall_event_logits_batch[
                        batch_index, : recall_event_lengths[batch_index]
                    ],
                    "recall_pair_features": recall_pair_features_batch[
                        batch_index, : recall_event_lengths[batch_index]
                    ],
                    "memory_router_logits": memory_router_logits_batch[batch_index],
                    "memory_router_layers": list(MEMORY_ROUTER_LAYERS),
                    "effective_question_features": dict(question_features_batch[batch_index]),
                }
                for batch_index in range(batch_size)
            ]

        candidate_event_ids_batch: List[List[str]] = []
        candidate_event_lengths: List[int] = []
        max_event_count = 0
        for batch_index, example in enumerate(examples):
            static_candidate_event_ids = [event_id for event_id in example.candidate_event_ids if event_id in node_id_to_index]
            candidate_limit = max(DEFAULT_RUNTIME_RECALL_TOP_K, len(static_candidate_event_ids))
            candidate_event_ids = (
                _select_rerank_event_ids(
                    recall_event_ids_batch[batch_index],
                    recall_event_logits_batch[batch_index, : recall_event_lengths[batch_index]],
                    positive_event_ids=example.positive_event_ids,
                    forced_event_ids=static_candidate_event_ids,
                    candidate_limit=candidate_limit,
                )
                if recall_event_lengths[batch_index] > 0
                else list(static_candidate_event_ids)
            )
            if not candidate_event_ids:
                candidate_event_ids = list(static_candidate_event_ids)
            candidate_event_ids_batch.append(candidate_event_ids)
            candidate_event_lengths.append(len(candidate_event_ids))
            max_event_count = max(max_event_count, len(candidate_event_ids))
        candidate_event_index_lookup_batch = [
            {event_id: index for index, event_id in enumerate(candidate_event_ids)}
            for candidate_event_ids in candidate_event_ids_batch
        ]
        candidate_event_recall_indices_batch = [
            [recall_event_index_lookup_batch[batch_index].get(event_id, -1) for event_id in candidate_event_ids]
            for batch_index, candidate_event_ids in enumerate(candidate_event_ids_batch)
        ]

        candidate_paths_batch: List[List[Dict[str, Any]]] = []
        candidate_path_ids_batch: List[List[str]] = []
        candidate_path_event_ids_batch: List[List[str]] = []
        candidate_path_support_node_ids_batch: List[List[str]] = []
        candidate_path_lengths: List[int] = []
        max_path_count = 0
        all_paths = list(graph_tensors.get("paths", []) or [])
        paths_by_event_id = dict(graph_tensors.get("paths_by_event_id", {}) or {})
        for candidate_event_ids in candidate_event_ids_batch:
            candidate_paths = (
                [path for event_id in candidate_event_ids for path in list(paths_by_event_id.get(event_id, []) or [])]
                if paths_by_event_id
                else _candidate_paths_for_event_ids(
                    all_paths,
                    candidate_event_ids,
                    node_id_to_index=node_id_to_index,
                )
            )
            candidate_paths_batch.append(candidate_paths)
            candidate_path_ids = [clean_text(path.get("id", "")) for path in candidate_paths]
            candidate_path_event_ids = [clean_text(path.get("event_id", "")) for path in candidate_paths]
            candidate_path_support_node_ids = [_path_support_node_id(path) for path in candidate_paths]
            candidate_path_ids_batch.append(candidate_path_ids)
            candidate_path_event_ids_batch.append(candidate_path_event_ids)
            candidate_path_support_node_ids_batch.append(candidate_path_support_node_ids)
            candidate_path_lengths.append(len(candidate_path_ids))
            max_path_count = max(max_path_count, len(candidate_path_ids))
        candidate_path_event_indices_batch = [
            [
                candidate_event_index_lookup_batch[batch_index].get(event_id, -1)
                for event_id in candidate_path_event_ids_batch[batch_index]
            ]
            for batch_index in range(batch_size)
        ]
        candidate_path_recall_indices_batch = [
            [
                candidate_event_recall_indices_batch[batch_index][event_index]
                if 0 <= event_index < len(candidate_event_recall_indices_batch[batch_index])
                else -1
                for event_index in candidate_path_event_indices_batch[batch_index]
            ]
            for batch_index in range(batch_size)
        ]

        base_event_logits_batch = node_hidden.new_zeros((batch_size, max_event_count))
        rerank_event_logits_batch = node_hidden.new_zeros((batch_size, max_event_count))
        distractor_event_logits_batch = node_hidden.new_zeros((batch_size, max_event_count))
        calibrated_event_logits_batch = node_hidden.new_zeros((batch_size, max_event_count))
        final_event_logits_batch = node_hidden.new_zeros((batch_size, max_event_count))
        matrix_event_delta_batch = node_hidden.new_zeros((batch_size, max_event_count))
        event_distractor_logits_batch = node_hidden.new_zeros((batch_size, max_event_count))
        event_distractor_delta_batch = node_hidden.new_zeros((batch_size, max_event_count))
        event_calibration_delta_batch = node_hidden.new_zeros((batch_size, max_event_count))
        event_tunnel_support_logits_batch = node_hidden.new_zeros((batch_size, max_event_count))
        event_tunnel_delta_batch = node_hidden.new_zeros((batch_size, max_event_count))
        event_fusion_delta_batch = node_hidden.new_zeros((batch_size, max_event_count))
        event_mask = torch.zeros((batch_size, max_event_count), dtype=torch.bool, device=device)
        event_support_embeddings_batch = node_hidden.new_zeros((batch_size, max_event_count, MESSAGE_HIDDEN_DIM))
        event_support_slot_embeddings_batch = node_hidden.new_zeros(
            (batch_size, max_event_count, len(EVENT_SUPPORT_PATH_TYPE_ORDER), MESSAGE_HIDDEN_DIM)
        )
        event_support_slot_mask_batch = torch.zeros(
            (batch_size, max_event_count, len(EVENT_SUPPORT_PATH_TYPE_ORDER)),
            dtype=torch.bool,
            device=device,
        )
        pooled_event_embeddings = node_hidden.new_zeros((batch_size, MESSAGE_HIDDEN_DIM))
        pooled_support_embeddings = node_hidden.new_zeros((batch_size, MESSAGE_HIDDEN_DIM))
        matrix_event_ids_batch: List[List[str]] = [[] for _ in range(batch_size)]
        matrix_event_lengths: List[int] = [0 for _ in range(batch_size)]
        matrix_local_index_batch: List[List[int]] = [[] for _ in range(batch_size)]
        event_relation_strength_batch = node_hidden.new_zeros((batch_size, max_event_count, max_event_count))
        event_pair_features_batch_raw = node_hidden.new_zeros((batch_size, max_event_count, PAIR_FEATURE_DIM))
        event_pair_features_batch = event_pair_features_batch_raw
        if max_event_count > 0:
            padded_event_indices = torch.zeros((batch_size, max_event_count), dtype=torch.long, device=device)
            for batch_index, candidate_event_ids in enumerate(candidate_event_ids_batch):
                if not candidate_event_ids:
                    continue
                event_index_values = [node_id_to_index[event_id] for event_id in candidate_event_ids]
                _validate_index_values(
                    event_index_values,
                    upper_bound=int(node_hidden.size(0)),
                    index_name="candidate_event_index",
                    context="LocomoNodeMemoryModel.score_examples_with_graph_encoding",
                    extra={
                        "stage": "event",
                        "conversation_id": examples[batch_index].conversation_id,
                        "question_id": examples[batch_index].question_id,
                        "candidate_event_count": len(candidate_event_ids),
                    },
                )
                indices = torch.tensor(event_index_values, dtype=torch.long, device=device)
                padded_event_indices[batch_index, : indices.numel()] = indices
                event_mask[batch_index, : indices.numel()] = True
                recall_indices_for_candidates = candidate_event_recall_indices_batch[batch_index]
                reusable_positions = [
                    position for position, recall_index in enumerate(recall_indices_for_candidates) if int(recall_index) >= 0
                ]
                if reusable_positions:
                    reusable_recall_indices = torch.tensor(
                        [int(recall_indices_for_candidates[position]) for position in reusable_positions],
                        dtype=torch.long,
                        device=device,
                    )
                    reusable_feature_positions = torch.tensor(
                        reusable_positions,
                        dtype=torch.long,
                        device=device,
                    )
                    event_pair_features_batch_raw[batch_index, : indices.numel()].index_copy_(
                        0,
                        reusable_feature_positions,
                        recall_pair_features_batch[batch_index].index_select(0, reusable_recall_indices),
                    )
                missing_positions = [
                    position for position, recall_index in enumerate(recall_indices_for_candidates) if int(recall_index) < 0
                ]
                if missing_positions:
                    missing_feature_positions = torch.tensor(
                        missing_positions,
                        dtype=torch.long,
                        device=device,
                    )
                    event_pair_features_batch_raw[batch_index, : indices.numel()].index_copy_(
                        0,
                        missing_feature_positions,
                        torch.tensor(
                            [
                                _event_pair_feature_values(
                                    examples[batch_index].question,
                                    question_features_batch[batch_index],
                                    node_lookup.get(candidate_event_ids[position], {}),
                                    support_payload=event_support_lookup.get(candidate_event_ids[position], {}),
                                    prepared_question=prepared_question_batch[batch_index],
                                    prepared_event=event_scoring_features_by_id.get(candidate_event_ids[position]),
                                )
                                for position in missing_positions
                            ],
                            dtype=node_hidden.dtype,
                            device=device,
                        ),
                    )
                relation_features = _event_relation_feature_matrix(
                    [
                        event_scoring_features_by_id.get(event_id, {})
                        for event_id in candidate_event_ids
                    ],
                    dtype=node_hidden.dtype,
                    device=device,
                )
                if relation_features.numel() > 0:
                    event_relation_strength_batch[batch_index, : indices.numel(), : indices.numel()] = _relation_strength_matrix(
                        relation_features,
                        TRI_MAZE_EVENT_RELATION_WEIGHTS,
                    )
            event_pair_features_batch_raw = _apply_event_pair_feature_mode(
                event_pair_features_batch_raw,
                resolved_event_pair_feature_mode,
            )
            event_pair_features_batch = self.event_pair_adapter(event_pair_features_batch_raw)
            gathered_event_embeddings = node_hidden.index_select(0, padded_event_indices.reshape(-1)).view(
                batch_size,
                max_event_count,
                -1,
            )
            raw_event_support_embeddings_batch = _aggregate_event_support_embeddings(
                node_hidden,
                node_id_to_index,
                candidate_event_ids_batch,
                candidate_paths_batch=candidate_paths_batch,
                support_node_ids_by_event=dict(graph_tensors.get("support_node_ids_by_event", {}) or {}),
                support_node_ids_by_event_and_type=dict(graph_tensors.get("support_node_ids_by_event_and_type", {}) or {}),
                question_features_batch=question_features_batch,
            )
            event_support_slot_embeddings_batch, event_support_slot_mask_batch = _aggregate_event_support_slot_embeddings(
                node_hidden,
                node_id_to_index,
                candidate_event_ids_batch,
                candidate_paths_batch=candidate_paths_batch,
                support_node_ids_by_event_and_type=dict(
                    graph_tensors.get("support_node_ids_by_event_and_type", {}) or {}
                ),
            )
            gathered_event_embeddings, event_support_embeddings_batch = self.event_subgraph_refiner(
                projected_batch,
                gathered_event_embeddings,
                event_support_slot_embeddings_batch,
                event_support_slot_mask_batch,
                event_mask,
            )
            missing_support_mask = ~event_support_slot_mask_batch.any(dim=-1)
            event_support_embeddings_batch = torch.where(
                missing_support_mask.unsqueeze(-1),
                raw_event_support_embeddings_batch,
                event_support_embeddings_batch,
            )
            base_event_logits_batch = self.event_head(
                question_batch,
                gathered_event_embeddings,
                event_support_embeddings_batch,
                projected_batch,
                event_pair_features_batch,
            )
            rerank_event_logits_batch = base_event_logits_batch
            if resolved_event_rerank_mode == "matrix":
                max_matrix_event_count = 0
                for batch_index, candidate_event_ids in enumerate(candidate_event_ids_batch):
                    recall_ranked_event_ids = []
                    candidate_event_id_set = {clean_text(item) for item in candidate_event_ids if clean_text(item)}
                    if recall_event_lengths[batch_index] > 0:
                        ranked_indices = torch.argsort(
                            recall_event_logits_batch[batch_index, : recall_event_lengths[batch_index]].detach(),
                            descending=True,
                        )
                        recall_ranked_event_ids = [
                            clean_text(recall_event_ids_batch[batch_index][int(index)])
                            for index in ranked_indices.tolist()
                            if clean_text(recall_event_ids_batch[batch_index][int(index)]) in candidate_event_id_set
                        ]
                    matrix_event_ids = _select_matrix_event_ids(
                        candidate_event_ids,
                        recall_ranked_event_ids,
                        positive_event_ids=examples[batch_index].positive_event_ids,
                        hard_negative_event_ids=_resolved_hard_negative_event_ids_for_example(examples[batch_index]),
                        matrix_top_k=resolved_matrix_event_top_k,
                    )
                    matrix_event_ids_batch[batch_index] = list(matrix_event_ids)
                    matrix_event_lengths[batch_index] = len(matrix_event_ids)
                    max_matrix_event_count = max(max_matrix_event_count, len(matrix_event_ids))
                    candidate_index_lookup = {event_id: index for index, event_id in enumerate(candidate_event_ids)}
                    matrix_local_index_batch[batch_index] = [
                        candidate_index_lookup[event_id] for event_id in matrix_event_ids if event_id in candidate_index_lookup
                    ]
                    _validate_index_values(
                        matrix_local_index_batch[batch_index],
                        upper_bound=len(candidate_event_ids),
                        index_name="matrix_event_local_index",
                        context="LocomoNodeMemoryModel.score_examples_with_graph_encoding",
                        extra={
                            "stage": "event_matrix",
                            "conversation_id": examples[batch_index].conversation_id,
                            "question_id": examples[batch_index].question_id,
                            "matrix_event_count": len(matrix_event_ids),
                            "candidate_event_count": len(candidate_event_ids),
                        },
                    )
                if max_matrix_event_count > 0:
                    matrix_event_embeddings = node_hidden.new_zeros((batch_size, max_matrix_event_count, MESSAGE_HIDDEN_DIM))
                    matrix_support_embeddings = node_hidden.new_zeros((batch_size, max_matrix_event_count, MESSAGE_HIDDEN_DIM))
                    matrix_pair_features = node_hidden.new_zeros((batch_size, max_matrix_event_count, PAIR_FEATURE_DIM))
                    matrix_relation_features = node_hidden.new_zeros(
                        (batch_size, max_matrix_event_count, max_matrix_event_count, MATRIX_RELATION_FEATURE_DIM)
                    )
                    matrix_event_mask = torch.zeros((batch_size, max_matrix_event_count), dtype=torch.bool, device=device)
                    for batch_index, local_indices in enumerate(matrix_local_index_batch):
                        if not local_indices:
                            continue
                        local_index_tensor = torch.tensor(local_indices, dtype=torch.long, device=device)
                        matrix_count = local_index_tensor.numel()
                        matrix_event_mask[batch_index, :matrix_count] = True
                        matrix_event_embeddings[batch_index, :matrix_count] = gathered_event_embeddings[batch_index].index_select(0, local_index_tensor)
                        matrix_support_embeddings[batch_index, :matrix_count] = event_support_embeddings_batch[batch_index].index_select(0, local_index_tensor)
                        matrix_pair_features[batch_index, :matrix_count] = event_pair_features_batch[batch_index].index_select(0, local_index_tensor)
                        matrix_node_features = [
                            event_scoring_features_by_id.get(event_id, {})
                            for event_id in matrix_event_ids_batch[batch_index]
                        ]
                        matrix_relation_features[batch_index, :matrix_count, :matrix_count] = _event_relation_feature_matrix(
                            matrix_node_features,
                            dtype=node_hidden.dtype,
                            device=device,
                        )
                    matrix_delta_logits = self.event_matrix_head(
                        question_batch,
                        matrix_event_embeddings,
                        matrix_support_embeddings,
                        projected_batch,
                        matrix_pair_features,
                        matrix_relation_features,
                        matrix_event_mask,
                    )
                    for batch_index, local_indices in enumerate(matrix_local_index_batch):
                        if not local_indices:
                            continue
                        local_index_tensor = torch.tensor(local_indices, dtype=torch.long, device=device)
                        matrix_count = int(local_index_tensor.numel())
                        matrix_event_delta_batch[batch_index].index_copy_(
                            0,
                            local_index_tensor,
                            MATRIX_EVENT_DELTA_LOGIT_LIMIT
                            * torch.tanh(
                                matrix_delta_logits[batch_index, :matrix_count].to(
                                    dtype=matrix_event_delta_batch.dtype
                                )
                                / MATRIX_EVENT_DELTA_LOGIT_LIMIT
                            ),
                        )
                    rerank_event_logits_batch = base_event_logits_batch + matrix_event_delta_batch
            rerank_event_prob_batch = torch.sigmoid(rerank_event_logits_batch).masked_fill(~event_mask, 0.0)
            event_distractor_features_batch = node_hidden.new_zeros(
                (batch_size, max_event_count, EVENT_DISTRACTOR_FEATURE_DIM)
            )
            for batch_index, candidate_event_ids in enumerate(candidate_event_ids_batch):
                if not candidate_event_ids:
                    continue
                event_count = candidate_event_lengths[batch_index]
                recall_scores_for_candidates = _gather_scores_by_indices(
                    recall_event_prob_batch[batch_index, : recall_event_lengths[batch_index]],
                    candidate_event_recall_indices_batch[batch_index],
                    device=device,
                )
                base_event_probs = torch.sigmoid(base_event_logits_batch[batch_index, :event_count])
                rerank_event_probs = rerank_event_prob_batch[batch_index, :event_count]
                support_slot_cover = (
                    event_support_slot_mask_batch[batch_index, :event_count]
                    .sum(dim=-1)
                    .to(dtype=node_hidden.dtype)
                    / max(1.0, float(len(EVENT_SUPPORT_PATH_TYPE_ORDER)))
                )
                is_temporal_tensor = node_hidden.new_full(
                    (event_count, 1),
                    1.0 if bool(question_features_batch[batch_index].get("is_temporal", False)) else 0.0,
                )
                event_pair_slice = event_pair_features_batch[batch_index, :event_count]
                event_distractor_features_batch[batch_index, :event_count] = torch.cat(
                    [
                        recall_scores_for_candidates.unsqueeze(-1),
                        base_event_probs.unsqueeze(-1),
                        rerank_event_probs.unsqueeze(-1),
                        (rerank_event_probs - _best_other_scores(rerank_event_probs)).unsqueeze(-1),
                        (recall_scores_for_candidates - _best_other_scores(recall_scores_for_candidates)).unsqueeze(-1),
                        _rank_ratio(rerank_event_probs).unsqueeze(-1),
                        support_slot_cover.unsqueeze(-1),
                        is_temporal_tensor,
                        event_pair_slice[:, [1, 5, 6, 14, 15, 25, 26, 27, 28]],
                    ],
                    dim=-1,
                )
            event_distractor_logits_batch = self.event_distractor_head(
                question_batch,
                gathered_event_embeddings,
                event_support_embeddings_batch,
                projected_batch,
                event_pair_features_batch,
                event_distractor_features_batch,
            ).masked_fill(~event_mask, 0.0)
            event_distractor_delta_batch = ((0.5 - torch.sigmoid(event_distractor_logits_batch)) * event_mask.to(
                dtype=node_hidden.dtype
            ))
            distractor_event_logits_batch = rerank_event_logits_batch + event_distractor_delta_batch
            calibrated_event_logits_batch = distractor_event_logits_batch
            masked_event_logits = distractor_event_logits_batch.masked_fill(~event_mask, float("-inf"))
            top_k = min(3, max_event_count)
            top_indices = torch.topk(masked_event_logits, k=top_k, dim=1).indices
            selected_event_embeddings = torch.gather(
                gathered_event_embeddings,
                1,
                top_indices.unsqueeze(-1).expand(-1, -1, gathered_event_embeddings.size(-1)),
            )
            selected_support_embeddings = torch.gather(
                event_support_embeddings_batch,
                1,
                top_indices.unsqueeze(-1).expand(-1, -1, event_support_embeddings_batch.size(-1)),
            )
            selected_event_mask = torch.gather(event_mask, 1, top_indices)
            pooled_event_embeddings = (
                (selected_event_embeddings + selected_support_embeddings) * 0.5
                * selected_event_mask.unsqueeze(-1).to(dtype=gathered_event_embeddings.dtype)
            ).sum(dim=1) / selected_event_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=gathered_event_embeddings.dtype)
            pooled_support_embeddings = (
                selected_support_embeddings
                * selected_event_mask.unsqueeze(-1).to(dtype=gathered_event_embeddings.dtype)
            ).sum(dim=1) / selected_event_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=gathered_event_embeddings.dtype)

        base_path_logits_batch = node_hidden.new_zeros((batch_size, max_path_count))
        rerank_path_logits_batch = node_hidden.new_zeros((batch_size, max_path_count))
        calibrated_path_logits_batch = node_hidden.new_zeros((batch_size, max_path_count))
        final_path_logits_batch = node_hidden.new_zeros((batch_size, max_path_count))
        matrix_path_delta_batch = node_hidden.new_zeros((batch_size, max_path_count))
        path_calibration_delta_batch = node_hidden.new_zeros((batch_size, max_path_count))
        path_tunnel_support_logits_batch = node_hidden.new_zeros((batch_size, max_path_count))
        path_tunnel_delta_batch = node_hidden.new_zeros((batch_size, max_path_count))
        path_fusion_delta_batch = node_hidden.new_zeros((batch_size, max_path_count))
        path_mask = torch.zeros((batch_size, max_path_count), dtype=torch.bool, device=device)
        path_event_embeddings = node_hidden.new_zeros((batch_size, max_path_count, MESSAGE_HIDDEN_DIM))
        path_support_embeddings = node_hidden.new_zeros((batch_size, max_path_count, MESSAGE_HIDDEN_DIM))
        matrix_path_ids_batch: List[List[str]] = [[] for _ in range(batch_size)]
        matrix_path_lengths: List[int] = [0 for _ in range(batch_size)]
        matrix_path_local_index_batch: List[List[int]] = [[] for _ in range(batch_size)]
        path_relation_strength_batch = node_hidden.new_zeros((batch_size, max_path_count, max_path_count))
        path_pair_features_batch_raw = node_hidden.new_zeros((batch_size, max_path_count, PATH_PAIR_FEATURE_DIM))
        path_pair_features_batch = path_pair_features_batch_raw
        if max_path_count > 0:
            padded_event_path_indices = torch.zeros((batch_size, max_path_count), dtype=torch.long, device=device)
            padded_support_indices = torch.zeros((batch_size, max_path_count), dtype=torch.long, device=device)
            padded_path_type_ids = torch.zeros((batch_size, max_path_count), dtype=torch.long, device=device)
            for batch_index, candidate_paths in enumerate(candidate_paths_batch):
                if not candidate_paths:
                    continue
                event_index_values = [node_id_to_index[clean_text(path.get("event_id", ""))] for path in candidate_paths]
                support_index_values = [node_id_to_index[_path_support_node_id(path)] for path in candidate_paths]
                _validate_index_values(
                    event_index_values,
                    upper_bound=int(node_hidden.size(0)),
                    index_name="path_event_index",
                    context="LocomoNodeMemoryModel.score_examples_with_graph_encoding",
                    extra={
                        "stage": "path",
                        "conversation_id": examples[batch_index].conversation_id,
                        "question_id": examples[batch_index].question_id,
                        "candidate_path_count": len(candidate_paths),
                    },
                )
                _validate_index_values(
                    support_index_values,
                    upper_bound=int(node_hidden.size(0)),
                    index_name="path_support_index",
                    context="LocomoNodeMemoryModel.score_examples_with_graph_encoding",
                    extra={
                        "stage": "path",
                        "conversation_id": examples[batch_index].conversation_id,
                        "question_id": examples[batch_index].question_id,
                        "candidate_path_count": len(candidate_paths),
                    },
                )
                event_indices = torch.tensor(
                    event_index_values,
                    dtype=torch.long,
                    device=device,
                )
                support_indices = torch.tensor(
                    support_index_values,
                    dtype=torch.long,
                    device=device,
                )
                path_type_ids = torch.tensor(
                    [PATH_TYPE_TO_ID.get(_path_type(path), 0) for path in candidate_paths],
                    dtype=torch.long,
                    device=device,
                )
                padded_event_path_indices[batch_index, : event_indices.numel()] = event_indices
                padded_support_indices[batch_index, : support_indices.numel()] = support_indices
                padded_path_type_ids[batch_index, : path_type_ids.numel()] = path_type_ids
                path_pair_features_batch_raw[batch_index, : path_type_ids.numel()] = torch.tensor(
                    [
                        _path_pair_feature_values(
                            examples[batch_index].question,
                            question_features_batch[batch_index],
                            node_lookup.get(clean_text(path.get("event_id", "")), {}),
                            node_lookup.get(_path_support_node_id(path), {}),
                            path_type=_path_type(path),
                            prepared_question=prepared_question_batch[batch_index],
                            prepared_event=event_scoring_features_by_id.get(clean_text(path.get("event_id", ""))),
                            prepared_support=node_scoring_features_by_id.get(_path_support_node_id(path)),
                        )
                        for path in candidate_paths
                    ],
                    dtype=node_hidden.dtype,
                    device=device,
                )
                path_relation_features = _path_relation_feature_matrix(
                    [
                        path_scoring_features_by_id.get(path_id, {})
                        for path_id in candidate_path_ids_batch[batch_index]
                    ],
                    dtype=node_hidden.dtype,
                    device=device,
                )
                if path_relation_features.numel() > 0:
                    path_relation_strength_batch[batch_index, : path_type_ids.numel(), : path_type_ids.numel()] = _relation_strength_matrix(
                        path_relation_features,
                        TRI_MAZE_PATH_RELATION_WEIGHTS,
                    )
                path_mask[batch_index, : path_type_ids.numel()] = True
            path_pair_features_batch = self.path_pair_adapter(path_pair_features_batch_raw)
            path_event_embeddings = node_hidden.new_zeros((batch_size, max_path_count, MESSAGE_HIDDEN_DIM))
            for batch_index, event_indices in enumerate(candidate_path_event_indices_batch):
                if not event_indices:
                    continue
                local_event_index_tensor = torch.tensor(event_indices, dtype=torch.long, device=device)
                _validate_index_values(
                    local_event_index_tensor.detach().cpu().tolist(),
                    upper_bound=int(candidate_event_lengths[batch_index]),
                    index_name="path_event_local_index",
                    context="LocomoNodeMemoryModel.score_examples_with_graph_encoding",
                    extra={
                        "stage": "path_event_refined",
                        "conversation_id": examples[batch_index].conversation_id,
                        "question_id": examples[batch_index].question_id,
                        "candidate_path_count": len(event_indices),
                        "candidate_event_count": int(candidate_event_lengths[batch_index]),
                    },
                )
                path_event_embeddings[batch_index, : local_event_index_tensor.numel()] = gathered_event_embeddings[
                    batch_index
                ].index_select(0, local_event_index_tensor)
            path_support_embeddings = node_hidden.index_select(0, padded_support_indices.reshape(-1)).view(batch_size, max_path_count, -1)
            base_path_logits_batch = self.path_head(
                question_batch,
                path_event_embeddings,
                path_support_embeddings,
                projected_batch,
                padded_path_type_ids,
                path_pair_features_batch,
            )
            base_path_logits_batch = base_path_logits_batch.masked_fill(~path_mask, 0.0)
            rerank_path_logits_batch = base_path_logits_batch
            max_matrix_path_count = 0
            for batch_index, candidate_paths in enumerate(candidate_paths_batch):
                if not candidate_paths:
                    continue
                ranked_indices = torch.argsort(base_path_logits_batch[batch_index, : candidate_path_lengths[batch_index]].detach(), descending=True)
                ranked_path_ids = [
                    candidate_path_ids_batch[batch_index][int(index)]
                    for index in ranked_indices.tolist()
                    if int(index) < len(candidate_path_ids_batch[batch_index])
                ]
                hard_negative_path_ids = _hard_negative_path_ids(examples[batch_index], candidate_path_ids_batch[batch_index])
                matrix_path_ids = _select_matrix_path_ids(
                    candidate_path_ids_batch[batch_index],
                    ranked_path_ids,
                    positive_path_ids=examples[batch_index].positive_path_ids,
                    hard_negative_path_ids=hard_negative_path_ids,
                    matrix_top_k=DEFAULT_MATRIX_PATH_TOP_K,
                )
                matrix_path_ids_batch[batch_index] = list(matrix_path_ids)
                matrix_path_lengths[batch_index] = len(matrix_path_ids)
                max_matrix_path_count = max(max_matrix_path_count, len(matrix_path_ids))
                candidate_index_lookup = {
                    path_id: index for index, path_id in enumerate(candidate_path_ids_batch[batch_index])
                }
                matrix_path_local_index_batch[batch_index] = [
                    candidate_index_lookup[path_id] for path_id in matrix_path_ids if path_id in candidate_index_lookup
                ]
                _validate_index_values(
                    matrix_path_local_index_batch[batch_index],
                    upper_bound=len(candidate_path_ids_batch[batch_index]),
                    index_name="matrix_path_local_index",
                    context="LocomoNodeMemoryModel.score_examples_with_graph_encoding",
                    extra={
                        "stage": "path_matrix",
                        "conversation_id": examples[batch_index].conversation_id,
                        "question_id": examples[batch_index].question_id,
                        "matrix_path_count": len(matrix_path_ids),
                        "candidate_path_count": len(candidate_path_ids_batch[batch_index]),
                    },
                )
            if max_matrix_path_count > 0:
                matrix_path_event_embeddings = node_hidden.new_zeros((batch_size, max_matrix_path_count, MESSAGE_HIDDEN_DIM))
                matrix_path_support_embeddings = node_hidden.new_zeros((batch_size, max_matrix_path_count, MESSAGE_HIDDEN_DIM))
                matrix_path_pair_features = node_hidden.new_zeros((batch_size, max_matrix_path_count, PATH_PAIR_FEATURE_DIM))
                matrix_path_relation_features = node_hidden.new_zeros(
                    (batch_size, max_matrix_path_count, max_matrix_path_count, PATH_MATRIX_RELATION_FEATURE_DIM)
                )
                matrix_path_mask = torch.zeros((batch_size, max_matrix_path_count), dtype=torch.bool, device=device)
                for batch_index, local_indices in enumerate(matrix_path_local_index_batch):
                    if not local_indices:
                        continue
                    local_index_tensor = torch.tensor(local_indices, dtype=torch.long, device=device)
                    matrix_count = local_index_tensor.numel()
                    matrix_path_mask[batch_index, :matrix_count] = True
                    matrix_path_event_embeddings[batch_index, :matrix_count] = path_event_embeddings[batch_index].index_select(0, local_index_tensor)
                    matrix_path_support_embeddings[batch_index, :matrix_count] = path_support_embeddings[batch_index].index_select(0, local_index_tensor)
                    matrix_path_pair_features[batch_index, :matrix_count] = path_pair_features_batch[batch_index].index_select(0, local_index_tensor)
                    matrix_path_features = [
                        path_scoring_features_by_id.get(path_id, {})
                        for path_id in matrix_path_ids_batch[batch_index]
                    ]
                    matrix_path_relation_features[batch_index, :matrix_count, :matrix_count] = _path_relation_feature_matrix(
                        matrix_path_features,
                        dtype=node_hidden.dtype,
                        device=device,
                    )
                matrix_path_delta_logits = self.path_matrix_head(
                    question_batch,
                    matrix_path_event_embeddings,
                    matrix_path_support_embeddings,
                    projected_batch,
                    matrix_path_pair_features,
                    matrix_path_relation_features,
                    matrix_path_mask,
                )
                for batch_index, local_indices in enumerate(matrix_path_local_index_batch):
                    if not local_indices:
                        continue
                    local_index_tensor = torch.tensor(local_indices, dtype=torch.long, device=device)
                    matrix_count = int(local_index_tensor.numel())
                    matrix_path_delta_batch[batch_index].index_copy_(
                        0,
                        local_index_tensor,
                        MATRIX_PATH_DELTA_LOGIT_LIMIT
                        * torch.tanh(
                            matrix_path_delta_logits[batch_index, :matrix_count].to(
                                dtype=matrix_path_delta_batch.dtype
                            )
                            / MATRIX_PATH_DELTA_LOGIT_LIMIT
                        ),
                    )
                rerank_path_logits_batch = (base_path_logits_batch + matrix_path_delta_batch).masked_fill(~path_mask, 0.0)
            calibrated_path_logits_batch = rerank_path_logits_batch

        temporal_paths_by_conversation = {
            clean_text(conversation_id): [dict(path) for path in list(paths or [])]
            for conversation_id, paths in dict(graph_tensors.get("temporal_paths_by_conversation", {}) or {}).items()
            if clean_text(conversation_id)
        }
        all_temporal_paths = [dict(path) for path in list(graph_tensors.get("temporal_paths", []) or [])]
        temporal_paths_batch: List[List[Dict[str, Any]]] = []
        candidate_temporal_path_ids_batch: List[List[str]] = []
        candidate_temporal_node_ids_batch: List[List[str]] = []
        candidate_temporal_event_ids_batch: List[List[str]] = []
        temporal_lengths: List[int] = []
        max_temporal_count = 0
        for example in examples:
            temporal_paths = list(temporal_paths_by_conversation.get(example.conversation_id, all_temporal_paths))
            temporal_paths_batch.append(temporal_paths)
            candidate_temporal_path_ids = [clean_text(path.get("id", "")) for path in temporal_paths]
            candidate_temporal_node_ids = [_path_support_node_id(path) for path in temporal_paths]
            candidate_temporal_event_ids = [clean_text(path.get("event_id", "")) for path in temporal_paths]
            candidate_temporal_path_ids_batch.append(candidate_temporal_path_ids)
            candidate_temporal_node_ids_batch.append(candidate_temporal_node_ids)
            candidate_temporal_event_ids_batch.append(candidate_temporal_event_ids)
            temporal_lengths.append(len(temporal_paths))
            max_temporal_count = max(max_temporal_count, len(temporal_paths))
        candidate_temporal_event_indices_batch = [
            [
                candidate_event_index_lookup_batch[batch_index].get(event_id, -1)
                for event_id in candidate_temporal_event_ids_batch[batch_index]
            ]
            for batch_index in range(batch_size)
        ]
        temporal_node_index_lookup_batch = [
            {node_id: index for index, node_id in enumerate(candidate_temporal_node_ids)}
            for candidate_temporal_node_ids in candidate_temporal_node_ids_batch
        ]
        candidate_path_temporal_support_indices_batch = [
            [
                temporal_node_index_lookup_batch[batch_index].get(support_node_id, -1)
                for support_node_id in candidate_path_support_node_ids_batch[batch_index]
            ]
            for batch_index in range(batch_size)
        ]
        temporal_logits_batch = node_hidden.new_zeros((batch_size, max_temporal_count))
        temporal_mask = torch.zeros((batch_size, max_temporal_count), dtype=torch.bool, device=device)
        if max_temporal_count > 0:
            padded_temporal_event_indices = torch.zeros((batch_size, max_temporal_count), dtype=torch.long, device=device)
            padded_temporal_time_indices = torch.zeros((batch_size, max_temporal_count), dtype=torch.long, device=device)
            for batch_index, temporal_paths in enumerate(temporal_paths_batch):
                if not temporal_paths:
                    continue
                temporal_event_index_values = [node_id_to_index[clean_text(path.get("event_id", ""))] for path in temporal_paths]
                temporal_time_index_values = [node_id_to_index[_path_support_node_id(path)] for path in temporal_paths]
                _validate_index_values(
                    temporal_event_index_values,
                    upper_bound=int(node_hidden.size(0)),
                    index_name="temporal_event_index",
                    context="LocomoNodeMemoryModel.score_examples_with_graph_encoding",
                    extra={
                        "stage": "temporal",
                        "conversation_id": examples[batch_index].conversation_id,
                        "question_id": examples[batch_index].question_id,
                        "temporal_path_count": len(temporal_paths),
                    },
                )
                _validate_index_values(
                    temporal_time_index_values,
                    upper_bound=int(node_hidden.size(0)),
                    index_name="temporal_time_index",
                    context="LocomoNodeMemoryModel.score_examples_with_graph_encoding",
                    extra={
                        "stage": "temporal",
                        "conversation_id": examples[batch_index].conversation_id,
                        "question_id": examples[batch_index].question_id,
                        "temporal_path_count": len(temporal_paths),
                    },
                )
                temporal_event_indices = torch.tensor(
                    temporal_event_index_values,
                    dtype=torch.long,
                    device=device,
                )
                temporal_time_indices = torch.tensor(
                    temporal_time_index_values,
                    dtype=torch.long,
                    device=device,
                )
                temporal_count = int(temporal_event_indices.numel())
                padded_temporal_event_indices[batch_index, :temporal_count] = temporal_event_indices
                padded_temporal_time_indices[batch_index, :temporal_count] = temporal_time_indices
                temporal_mask[batch_index, :temporal_count] = True
            temporal_event_embeddings = node_hidden.index_select(0, padded_temporal_event_indices.reshape(-1)).view(
                batch_size,
                max_temporal_count,
                -1,
            )
            temporal_time_embeddings = node_hidden.index_select(0, padded_temporal_time_indices.reshape(-1)).view(
                batch_size,
                max_temporal_count,
                -1,
            )
            temporal_logits_batch = self.temporal_head(
                question_batch,
                temporal_event_embeddings,
                temporal_time_embeddings,
                projected_batch,
            ).masked_fill(~temporal_mask, 0.0)

        answer_calibration_features = _build_answer_calibration_features(
            calibrated_event_logits_batch=calibrated_event_logits_batch,
            calibrated_path_logits_batch=calibrated_path_logits_batch,
            temporal_logits_batch=temporal_logits_batch,
            event_pair_features_batch_raw=event_pair_features_batch_raw,
            path_pair_features_batch_raw=path_pair_features_batch_raw,
            event_relation_strength_batch=event_relation_strength_batch,
            path_relation_strength_batch=path_relation_strength_batch,
            event_mask=event_mask,
            path_mask=path_mask,
            temporal_mask=temporal_mask,
            examples=examples,
            candidate_path_event_indices_batch=candidate_path_event_indices_batch,
            candidate_temporal_event_indices_batch=candidate_temporal_event_indices_batch,
        )
        answer_type_logits_batch = self.answer_type_head(
            question_batch,
            pooled_event_embeddings,
            pooled_support_embeddings,
            projected_batch,
            answer_calibration_features,
        )
        answer_type_probs_batch = F.softmax(answer_type_logits_batch, dim=-1)
        base_path_prob_batch = (
            torch.sigmoid(base_path_logits_batch)
            if max_path_count > 0
            else node_hidden.new_zeros((batch_size, 0))
        )
        rerank_path_prob_batch = (
            torch.sigmoid(rerank_path_logits_batch).masked_fill(~path_mask, 0.0)
            if max_path_count > 0
            else node_hidden.new_zeros((batch_size, 0))
        )
        temporal_prob_batch = (
            torch.sigmoid(temporal_logits_batch).masked_fill(~temporal_mask, 0.0)
            if max_temporal_count > 0
            else node_hidden.new_zeros((batch_size, 0))
        )
        if max_event_count > 0:
            event_runtime_features_batch = node_hidden.new_zeros(
                (batch_size, max_event_count, EVENT_RUNTIME_CALIBRATION_FEATURE_DIM)
            )
            for batch_index, candidate_event_ids in enumerate(candidate_event_ids_batch):
                if not candidate_event_ids:
                    continue
                event_count = candidate_event_lengths[batch_index]
                recall_scores_for_candidates = _gather_scores_by_indices(
                    recall_event_prob_batch[batch_index, : recall_event_lengths[batch_index]],
                    candidate_event_recall_indices_batch[batch_index],
                    device=device,
                )
                distractor_event_probs = torch.sigmoid(distractor_event_logits_batch[batch_index, :event_count])
                path_event_indices = candidate_path_event_indices_batch[batch_index]
                max_path_prob_by_event = _scatter_amax_by_indices(
                    rerank_path_prob_batch[batch_index, : candidate_path_lengths[batch_index]],
                    path_event_indices,
                    size=event_count,
                    device=device,
                )
                event_path_count = _scatter_sum_by_indices(
                    path_event_indices,
                    size=event_count,
                    device=device,
                    dtype=node_hidden.dtype,
                )
                temporal_event_indices = candidate_temporal_event_indices_batch[batch_index]
                max_temporal_prob_by_event = _scatter_amax_by_indices(
                    temporal_prob_batch[batch_index, : temporal_lengths[batch_index]],
                    temporal_event_indices,
                    size=event_count,
                    device=device,
                )
                is_temporal_tensor = node_hidden.new_full(
                    (event_count, 1),
                    1.0 if bool(question_features_batch[batch_index].get("is_temporal", False)) else 0.0,
                )
                event_pair_slice = event_pair_features_batch_raw[batch_index, :event_count]
                event_reverse_scores, event_boundary_gaps, event_reverse_relations, event_reverse_available, _ = _reverse_competition_features(
                    rerank_event_prob_batch[batch_index, :event_count],
                    event_relation_strength_batch[batch_index, :event_count, :event_count],
                    valid_mask=event_mask[batch_index, :event_count],
                )
                answer_features = answer_type_probs_batch[batch_index].unsqueeze(0).expand(event_count, -1).to(dtype=node_hidden.dtype)
                event_runtime_features_batch[batch_index, :event_count] = torch.cat(
                    [
                        recall_scores_for_candidates.unsqueeze(-1),
                        distractor_event_probs.unsqueeze(-1),
                        max_path_prob_by_event.unsqueeze(-1),
                        max_temporal_prob_by_event.unsqueeze(-1),
                        is_temporal_tensor,
                        (event_path_count / max(1.0, float(len(PATH_TYPES)))).clamp(max=1.0).unsqueeze(-1),
                        event_pair_slice[:, [25, 26, 1, 14, 5, 6, 27]],
                        event_reverse_scores.unsqueeze(-1),
                        event_boundary_gaps.unsqueeze(-1),
                        event_reverse_relations.unsqueeze(-1),
                        event_reverse_available.unsqueeze(-1),
                        answer_features,
                    ],
                    dim=-1,
                )
            event_calibration_delta_batch = self.event_calibration_head(
                question_batch,
                gathered_event_embeddings,
                event_support_embeddings_batch,
                projected_batch,
                event_pair_features_batch,
                event_runtime_features_batch,
            ).masked_fill(~event_mask, 0.0)
            calibrated_event_logits_batch = rerank_event_logits_batch + event_calibration_delta_batch
        if max_path_count > 0:
            calibrated_event_prob_batch = torch.sigmoid(calibrated_event_logits_batch)
            path_runtime_features_batch = node_hidden.new_zeros(
                (batch_size, max_path_count, PATH_RUNTIME_CALIBRATION_FEATURE_DIM)
            )
            for batch_index, candidate_paths in enumerate(candidate_paths_batch):
                if not candidate_paths:
                    continue
                path_count = candidate_path_lengths[batch_index]
                event_indices = candidate_path_event_indices_batch[batch_index]
                calibrated_event_scores_for_paths = _gather_scores_by_indices(
                    calibrated_event_prob_batch[batch_index, : candidate_event_lengths[batch_index]],
                    event_indices,
                    device=device,
                )
                recall_scores_for_paths = _gather_scores_by_indices(
                    recall_event_prob_batch[batch_index, : recall_event_lengths[batch_index]],
                    candidate_path_recall_indices_batch[batch_index],
                    device=device,
                )
                temporal_support_scores = _gather_scores_by_indices(
                    temporal_prob_batch[batch_index, : temporal_lengths[batch_index]],
                    candidate_path_temporal_support_indices_batch[batch_index],
                    device=device,
                )
                is_temporal_tensor = node_hidden.new_full(
                    (path_count, 1),
                    1.0 if bool(question_features_batch[batch_index].get("is_temporal", False)) else 0.0,
                )
                path_pair_slice = path_pair_features_batch_raw[batch_index, :path_count]
                path_reverse_scores, path_boundary_gaps, path_reverse_relations, path_reverse_available, _ = _reverse_competition_features(
                    rerank_path_prob_batch[batch_index, :path_count],
                    path_relation_strength_batch[batch_index, :path_count, :path_count],
                    valid_mask=path_mask[batch_index, :path_count],
                )
                answer_features = answer_type_probs_batch[batch_index].unsqueeze(0).expand(path_count, -1).to(dtype=node_hidden.dtype)
                path_runtime_features_batch[batch_index, :path_count] = torch.cat(
                    [
                        calibrated_event_scores_for_paths.unsqueeze(-1),
                        recall_scores_for_paths.unsqueeze(-1),
                        rerank_path_prob_batch[batch_index, :path_count].unsqueeze(-1),
                        temporal_support_scores.unsqueeze(-1),
                        is_temporal_tensor,
                        path_pair_slice[:, [4, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21]],
                        path_reverse_scores.unsqueeze(-1),
                        path_boundary_gaps.unsqueeze(-1),
                        path_reverse_relations.unsqueeze(-1),
                        path_reverse_available.unsqueeze(-1),
                        answer_features,
                    ],
                    dim=-1,
                )
            path_calibration_delta_batch = self.path_calibration_head(
                question_batch,
                path_event_embeddings,
                path_support_embeddings,
                projected_batch,
                path_pair_features_batch,
                path_runtime_features_batch,
            ).masked_fill(~path_mask, 0.0)
            calibrated_path_logits_batch = (rerank_path_logits_batch + path_calibration_delta_batch).masked_fill(~path_mask, 0.0)
        calibrated_event_prob_batch = (
            torch.sigmoid(calibrated_event_logits_batch)
            if max_event_count > 0
            else node_hidden.new_zeros((batch_size, 0))
        )
        calibrated_path_prob_batch = (
            torch.sigmoid(calibrated_path_logits_batch)
            if max_path_count > 0
            else node_hidden.new_zeros((batch_size, 0))
        )
        if max_event_count > 0:
            event_tunnel_features_batch = node_hidden.new_zeros(
                (batch_size, max_event_count, EVENT_TUNNEL_FEATURE_DIM)
            )
            for batch_index, candidate_event_ids in enumerate(candidate_event_ids_batch):
                if not candidate_event_ids:
                    continue
                event_count = candidate_event_lengths[batch_index]
                path_event_indices = candidate_path_event_indices_batch[batch_index]
                max_rerank_path_prob_by_event = _scatter_amax_by_indices(
                    rerank_path_prob_batch[batch_index, : candidate_path_lengths[batch_index]],
                    path_event_indices,
                    size=event_count,
                    device=device,
                )
                max_calibrated_path_prob_by_event = _scatter_amax_by_indices(
                    calibrated_path_prob_batch[batch_index, : candidate_path_lengths[batch_index]],
                    path_event_indices,
                    size=event_count,
                    device=device,
                )
                event_path_count = _scatter_sum_by_indices(
                    path_event_indices,
                    size=event_count,
                    device=device,
                    dtype=node_hidden.dtype,
                )
                temporal_event_indices = candidate_temporal_event_indices_batch[batch_index]
                max_temporal_prob_by_event = _scatter_amax_by_indices(
                    temporal_prob_batch[batch_index, : temporal_lengths[batch_index]],
                    temporal_event_indices,
                    size=event_count,
                    device=device,
                )
                event_tunnel_features_batch[batch_index, :event_count] = _build_event_tunnel_features(
                    prepared_events=[
                        event_scoring_features_by_id.get(event_id, {})
                        for event_id in candidate_event_ids
                    ],
                    calibrated_event_probs=calibrated_event_prob_batch[batch_index, :event_count],
                    max_rerank_path_prob_by_event=max_rerank_path_prob_by_event,
                    max_calibrated_path_prob_by_event=max_calibrated_path_prob_by_event,
                    max_temporal_prob_by_event=max_temporal_prob_by_event,
                    event_path_count=event_path_count,
                    event_relation_strength=event_relation_strength_batch[batch_index, :event_count, :event_count],
                    event_mask=event_mask[batch_index, :event_count],
                    question=examples[batch_index].question,
                    question_features=question_features_batch[batch_index],
                )
            event_tunnel_outputs = self.event_tunnel_head(
                question_batch,
                gathered_event_embeddings,
                event_support_embeddings_batch,
                projected_batch,
                event_pair_features_batch,
                event_tunnel_features_batch,
            ).masked_fill(~event_mask.unsqueeze(-1), 0.0)
            event_tunnel_support_logits_batch = event_tunnel_outputs[..., 0]
            event_tunnel_delta_batch = (
                EVENT_TUNNEL_DELTA_LOGIT_LIMIT
                * torch.tanh(event_tunnel_outputs[..., 1] / EVENT_TUNNEL_DELTA_LOGIT_LIMIT)
            ).masked_fill(~event_mask, 0.0)
        final_event_logits_batch = calibrated_event_logits_batch + event_tunnel_delta_batch
        answer_plan_prior_logits_batch = node_hidden.new_zeros((batch_size, max_event_count, len(ANSWER_PLAN_OUTPUTS)))
        if max_event_count > 0:
            answer_plan_prior_features_batch = node_hidden.new_zeros((batch_size, max_event_count, ANSWER_PLAN_FEATURE_DIM))
            preliminary_event_prob_batch = torch.sigmoid(final_event_logits_batch).masked_fill(~event_mask, 0.0)
            for batch_index, candidate_event_ids in enumerate(candidate_event_ids_batch):
                if not candidate_event_ids:
                    continue
                event_count = candidate_event_lengths[batch_index]
                path_event_indices = candidate_path_event_indices_batch[batch_index]
                max_calibrated_path_prob_by_event = _scatter_amax_by_indices(
                    calibrated_path_prob_batch[batch_index, : candidate_path_lengths[batch_index]],
                    path_event_indices,
                    size=event_count,
                    device=device,
                )
                temporal_event_indices = candidate_temporal_event_indices_batch[batch_index]
                max_temporal_prob_by_event = _scatter_amax_by_indices(
                    temporal_prob_batch[batch_index, : temporal_lengths[batch_index]],
                    temporal_event_indices,
                    size=event_count,
                    device=device,
                )
                question_features = dict(question_features_batch[batch_index] or {})
                semantic_target = clean_text(question_features.get("semantic_slot_target", ""))
                lowered_question = normalize_text(examples[batch_index].question)
                is_profile_query = bool(
                    question_features.get("asks_profile", False)
                    or semantic_target == "profile"
                    or "profile" in lowered_question
                    or "preference" in lowered_question
                    or "prefer" in lowered_question
                )
                is_temporal_query = bool(question_features.get("is_temporal", False))
                is_chain_query = bool(
                    question_features.get("asks_chain", False)
                    or "chain" in lowered_question
                    or "connected" in lowered_question
                    or "evidence" in lowered_question
                    or "changed" in lowered_question
                )
                answer_profile_prob = answer_type_probs_batch[batch_index, ANSWER_TYPE_TO_ID["profile"]]
                answer_multi_prob = answer_type_probs_batch[batch_index, ANSWER_TYPE_TO_ID["multi_evidence"]]
                answer_plan_prior_features_batch[batch_index, :event_count] = torch.cat(
                    [
                        preliminary_event_prob_batch[batch_index, :event_count].unsqueeze(-1),
                        calibrated_event_prob_batch[batch_index, :event_count].unsqueeze(-1),
                        torch.sigmoid(event_tunnel_support_logits_batch[batch_index, :event_count]).unsqueeze(-1),
                        torch.sigmoid(event_distractor_logits_batch[batch_index, :event_count]).unsqueeze(-1),
                        max_calibrated_path_prob_by_event.unsqueeze(-1),
                        max_calibrated_path_prob_by_event.unsqueeze(-1),
                        max_temporal_prob_by_event.unsqueeze(-1),
                        node_hidden.new_full((event_count, 1), 1.0 if is_profile_query else 0.0),
                        node_hidden.new_full((event_count, 1), 1.0 if is_temporal_query else 0.0),
                        node_hidden.new_full((event_count, 1), 1.0 if is_chain_query else 0.0),
                        answer_profile_prob.reshape(1, 1).expand(event_count, 1).to(dtype=node_hidden.dtype),
                        answer_multi_prob.reshape(1, 1).expand(event_count, 1).to(dtype=node_hidden.dtype),
                    ],
                    dim=-1,
                )
            answer_plan_prior_logits_batch = self.answer_plan_head(
                question_batch,
                gathered_event_embeddings,
                event_support_embeddings_batch,
                projected_batch,
                event_pair_features_batch,
                answer_plan_prior_features_batch,
            ).masked_fill(~event_mask.unsqueeze(-1), 0.0)
        answer_plan_prior_prob_batch = torch.sigmoid(answer_plan_prior_logits_batch).masked_fill(
            ~event_mask.unsqueeze(-1),
            0.0,
        )
        if max_event_count > 0:
            final_event_features_batch = node_hidden.new_zeros(
                (batch_size, max_event_count, FINAL_EVENT_FUSION_FEATURE_DIM)
            )
            for batch_index, candidate_event_ids in enumerate(candidate_event_ids_batch):
                if not candidate_event_ids:
                    continue
                event_count = candidate_event_lengths[batch_index]
                recall_scores_for_candidates = _gather_scores_by_indices(
                    recall_event_prob_batch[batch_index, : recall_event_lengths[batch_index]],
                    candidate_event_recall_indices_batch[batch_index],
                    device=device,
                )
                base_event_probs = torch.sigmoid(base_event_logits_batch[batch_index, :event_count])
                rerank_event_probs = torch.sigmoid(rerank_event_logits_batch[batch_index, :event_count])
                distractor_event_probs = torch.sigmoid(distractor_event_logits_batch[batch_index, :event_count])
                calibrated_event_probs = calibrated_event_prob_batch[batch_index, :event_count]
                path_event_indices = candidate_path_event_indices_batch[batch_index]
                max_rerank_path_prob_by_event = _scatter_amax_by_indices(
                    rerank_path_prob_batch[batch_index, : candidate_path_lengths[batch_index]],
                    path_event_indices,
                    size=event_count,
                    device=device,
                )
                max_calibrated_path_prob_by_event = _scatter_amax_by_indices(
                    calibrated_path_prob_batch[batch_index, : candidate_path_lengths[batch_index]],
                    path_event_indices,
                    size=event_count,
                    device=device,
                )
                event_path_count = _scatter_sum_by_indices(
                    path_event_indices,
                    size=event_count,
                    device=device,
                    dtype=node_hidden.dtype,
                )
                temporal_event_indices = candidate_temporal_event_indices_batch[batch_index]
                max_temporal_prob_by_event = _scatter_amax_by_indices(
                    temporal_prob_batch[batch_index, : temporal_lengths[batch_index]],
                    temporal_event_indices,
                    size=event_count,
                    device=device,
                )
                is_temporal_tensor = node_hidden.new_full(
                    (event_count, 1),
                    1.0 if bool(question_features_batch[batch_index].get("is_temporal", False)) else 0.0,
                )
                event_pair_slice = event_pair_features_batch_raw[batch_index, :event_count]
                event_reverse_scores, event_boundary_gaps, event_reverse_relations, event_reverse_available, event_reverse_indices = _reverse_competition_features(
                    calibrated_event_probs,
                    event_relation_strength_batch[batch_index, :event_count, :event_count],
                    valid_mask=event_mask[batch_index, :event_count],
                )
                reverse_event_path_support = _gather_1d_by_index(
                    max_calibrated_path_prob_by_event,
                    event_reverse_indices,
                    valid_mask=event_reverse_available > 0.0,
                )
                reverse_event_temporal_support = _gather_1d_by_index(
                    max_temporal_prob_by_event,
                    event_reverse_indices,
                    valid_mask=event_reverse_available > 0.0,
                )
                answer_plan_prior_probs = answer_plan_prior_prob_batch[batch_index, :event_count]
                answer_plan_prior_current_delta = (
                    answer_plan_prior_probs[:, ANSWER_PLAN_OUTPUT_TO_ID["current"]]
                    - answer_plan_prior_probs[:, ANSWER_PLAN_OUTPUT_TO_ID["suppressed"]]
                )
                answer_plan_prior_selected_delta = (
                    answer_plan_prior_probs[:, ANSWER_PLAN_OUTPUT_TO_ID["selected"]]
                    - answer_plan_prior_probs[:, ANSWER_PLAN_OUTPUT_TO_ID["suppressed"]]
                )
                answer_features = answer_type_probs_batch[batch_index].unsqueeze(0).expand(event_count, -1).to(dtype=node_hidden.dtype)
                final_event_features_batch[batch_index, :event_count] = torch.cat(
                    [
                        recall_scores_for_candidates.unsqueeze(-1),
                        base_event_probs.unsqueeze(-1),
                        rerank_event_probs.unsqueeze(-1),
                        distractor_event_probs.unsqueeze(-1),
                        calibrated_event_probs.unsqueeze(-1),
                        max_rerank_path_prob_by_event.unsqueeze(-1),
                        max_calibrated_path_prob_by_event.unsqueeze(-1),
                        max_temporal_prob_by_event.unsqueeze(-1),
                        is_temporal_tensor,
                        (event_path_count / max(1.0, float(len(PATH_TYPES)))).clamp(max=1.0).unsqueeze(-1),
                        event_pair_slice[:, [1, 2, 3, 4, 5, 6, 14, 25, 26, 27, 28]],
                        (calibrated_event_probs - _best_other_scores(calibrated_event_probs)).unsqueeze(-1),
                        (recall_scores_for_candidates - _best_other_scores(recall_scores_for_candidates)).unsqueeze(-1),
                        event_reverse_scores.unsqueeze(-1),
                        event_boundary_gaps.unsqueeze(-1),
                        event_reverse_relations.unsqueeze(-1),
                        event_reverse_available.unsqueeze(-1),
                        reverse_event_path_support.unsqueeze(-1),
                        reverse_event_temporal_support.unsqueeze(-1),
                        answer_plan_prior_probs,
                        answer_plan_prior_current_delta.unsqueeze(-1),
                        answer_plan_prior_selected_delta.unsqueeze(-1),
                        answer_features,
                    ],
                    dim=-1,
                )
            event_fusion_delta_batch = self.final_event_fusion_head(
                question_batch,
                gathered_event_embeddings,
                event_support_embeddings_batch,
                projected_batch,
                event_pair_features_batch,
                final_event_features_batch,
            ).masked_fill(~event_mask, 0.0)
            final_event_logits_batch = calibrated_event_logits_batch + event_tunnel_delta_batch + event_fusion_delta_batch
        final_event_prob_batch = (
            torch.sigmoid(final_event_logits_batch)
            if max_event_count > 0
            else node_hidden.new_zeros((batch_size, 0))
        )
        if max_path_count > 0:
            path_tunnel_features_batch = node_hidden.new_zeros(
                (batch_size, max_path_count, PATH_TUNNEL_FEATURE_DIM)
            )
            for batch_index, candidate_paths in enumerate(candidate_paths_batch):
                if not candidate_paths:
                    continue
                path_count = candidate_path_lengths[batch_index]
                event_count = candidate_event_lengths[batch_index]
                path_event_indices = candidate_path_event_indices_batch[batch_index]
                event_scores_for_paths = _gather_scores_by_indices(
                    final_event_prob_batch[batch_index, :event_count],
                    path_event_indices,
                    device=device,
                )
                temporal_support_scores = _gather_scores_by_indices(
                    temporal_prob_batch[batch_index, : temporal_lengths[batch_index]],
                    candidate_path_temporal_support_indices_batch[batch_index],
                    device=device,
                )
                calibrated_path_scores = calibrated_path_prob_batch[batch_index, :path_count]
                path_count_by_event = _scatter_sum_by_indices(
                    path_event_indices,
                    size=event_count,
                    device=device,
                    dtype=node_hidden.dtype,
                )
                path_coverage_by_event = (path_count_by_event / max(1.0, float(len(PATH_TYPES)))).clamp(max=1.0)
                path_coverage_for_paths = _gather_scores_by_indices(path_coverage_by_event, path_event_indices, device=device)
                event_rank_ratio_for_paths = _gather_scores_by_indices(
                    _rank_ratio(final_event_prob_batch[batch_index, :event_count]),
                    path_event_indices,
                    device=device,
                )
                path_reverse_scores, path_boundary_gaps, path_reverse_relations, path_reverse_available, _ = _reverse_competition_features(
                    calibrated_path_scores,
                    path_relation_strength_batch[batch_index, :path_count, :path_count],
                    valid_mask=path_mask[batch_index, :path_count],
                )
                path_tunnel_features_batch[batch_index, :path_count] = _build_path_tunnel_features(
                    event_scores_for_paths=event_scores_for_paths,
                    calibrated_path_scores=calibrated_path_scores,
                    temporal_support_scores=temporal_support_scores,
                    path_pair_features=path_pair_features_batch_raw[batch_index, :path_count],
                    path_reverse_scores=path_reverse_scores,
                    path_boundary_gaps=path_boundary_gaps,
                    path_reverse_relations=path_reverse_relations,
                    path_reverse_available=path_reverse_available,
                    path_coverage_for_paths=path_coverage_for_paths,
                    event_rank_ratio_for_paths=event_rank_ratio_for_paths,
                    question=examples[batch_index].question,
                    question_features=question_features_batch[batch_index],
                )
            path_tunnel_outputs = self.path_tunnel_head(
                question_batch,
                path_event_embeddings,
                path_support_embeddings,
                projected_batch,
                path_pair_features_batch,
                path_tunnel_features_batch,
            ).masked_fill(~path_mask.unsqueeze(-1), 0.0)
            path_tunnel_support_logits_batch = path_tunnel_outputs[..., 0]
            path_tunnel_delta_batch = (
                PATH_TUNNEL_DELTA_LOGIT_LIMIT
                * torch.tanh(path_tunnel_outputs[..., 1] / PATH_TUNNEL_DELTA_LOGIT_LIMIT)
            ).masked_fill(~path_mask, 0.0)
            path_tunnel_delta_batch = _zero_center_masked_logits(path_tunnel_delta_batch, path_mask)
        final_path_logits_batch = calibrated_path_logits_batch + path_tunnel_delta_batch
        if max_path_count > 0:
            final_path_features_batch = node_hidden.new_zeros(
                (batch_size, max_path_count, FINAL_PATH_FUSION_FEATURE_DIM)
            )
            for batch_index, candidate_paths in enumerate(candidate_paths_batch):
                if not candidate_paths:
                    continue
                path_count = candidate_path_lengths[batch_index]
                event_count = candidate_event_lengths[batch_index]
                path_event_indices = candidate_path_event_indices_batch[batch_index]
                event_scores_for_paths = _gather_scores_by_indices(
                    final_event_prob_batch[batch_index, :event_count],
                    path_event_indices,
                    device=device,
                )
                calibrated_event_scores_for_paths = _gather_scores_by_indices(
                    calibrated_event_prob_batch[batch_index, :event_count],
                    path_event_indices,
                    device=device,
                )
                recall_scores_for_paths = _gather_scores_by_indices(
                    recall_event_prob_batch[batch_index, : recall_event_lengths[batch_index]],
                    candidate_path_recall_indices_batch[batch_index],
                    device=device,
                )
                temporal_support_scores = _gather_scores_by_indices(
                    temporal_prob_batch[batch_index, : temporal_lengths[batch_index]],
                    candidate_path_temporal_support_indices_batch[batch_index],
                    device=device,
                )
                calibrated_path_scores = calibrated_path_prob_batch[batch_index, :path_count]
                path_count_by_event = _scatter_sum_by_indices(
                    path_event_indices,
                    size=event_count,
                    device=device,
                    dtype=node_hidden.dtype,
                )
                path_coverage_by_event = (path_count_by_event / max(1.0, float(len(PATH_TYPES)))).clamp(max=1.0)
                path_coverage_for_paths = _gather_scores_by_indices(path_coverage_by_event, path_event_indices, device=device)
                event_rank_ratio_for_paths = _gather_scores_by_indices(
                    _rank_ratio(final_event_prob_batch[batch_index, :event_count]),
                    path_event_indices,
                    device=device,
                )
                is_temporal_tensor = node_hidden.new_full(
                    (path_count, 1),
                    1.0 if bool(question_features_batch[batch_index].get("is_temporal", False)) else 0.0,
                )
                path_pair_slice = path_pair_features_batch_raw[batch_index, :path_count]
                path_reverse_scores, path_boundary_gaps, path_reverse_relations, path_reverse_available, path_reverse_indices = _reverse_competition_features(
                    calibrated_path_scores,
                    path_relation_strength_batch[batch_index, :path_count, :path_count],
                    valid_mask=path_mask[batch_index, :path_count],
                )
                reverse_path_event_scores = _gather_1d_by_index(
                    event_scores_for_paths,
                    path_reverse_indices,
                    valid_mask=path_reverse_available > 0.0,
                )
                reverse_path_temporal_scores = _gather_1d_by_index(
                    temporal_support_scores,
                    path_reverse_indices,
                    valid_mask=path_reverse_available > 0.0,
                )
                answer_plan_prior_for_paths = torch.stack(
                    [
                        _gather_scores_by_indices(
                            answer_plan_prior_prob_batch[batch_index, :event_count, role_index],
                            path_event_indices,
                            device=device,
                        )
                        for role_index in range(len(ANSWER_PLAN_OUTPUTS))
                    ],
                    dim=-1,
                )
                answer_plan_prior_current_delta_for_paths = (
                    answer_plan_prior_for_paths[:, ANSWER_PLAN_OUTPUT_TO_ID["current"]]
                    - answer_plan_prior_for_paths[:, ANSWER_PLAN_OUTPUT_TO_ID["suppressed"]]
                )
                answer_plan_prior_selected_delta_for_paths = (
                    answer_plan_prior_for_paths[:, ANSWER_PLAN_OUTPUT_TO_ID["selected"]]
                    - answer_plan_prior_for_paths[:, ANSWER_PLAN_OUTPUT_TO_ID["suppressed"]]
                )
                answer_features = answer_type_probs_batch[batch_index].unsqueeze(0).expand(path_count, -1).to(dtype=node_hidden.dtype)
                final_path_features_batch[batch_index, :path_count] = torch.cat(
                    [
                        event_scores_for_paths.unsqueeze(-1),
                        calibrated_event_scores_for_paths.unsqueeze(-1),
                        recall_scores_for_paths.unsqueeze(-1),
                        rerank_path_prob_batch[batch_index, :path_count].unsqueeze(-1),
                        calibrated_path_scores.unsqueeze(-1),
                        temporal_support_scores.unsqueeze(-1),
                        is_temporal_tensor,
                        path_pair_slice[:, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21]],
                        (calibrated_path_scores - _best_other_scores(calibrated_path_scores)).unsqueeze(-1),
                        (event_scores_for_paths - _gather_scores_by_indices(_best_other_scores(final_event_prob_batch[batch_index, :event_count]), path_event_indices, device=device)).unsqueeze(-1),
                        path_coverage_for_paths.unsqueeze(-1),
                        _rank_ratio(calibrated_path_scores).unsqueeze(-1),
                        event_rank_ratio_for_paths.unsqueeze(-1),
                        path_reverse_scores.unsqueeze(-1),
                        path_boundary_gaps.unsqueeze(-1),
                        path_reverse_relations.unsqueeze(-1),
                        path_reverse_available.unsqueeze(-1),
                        reverse_path_event_scores.unsqueeze(-1),
                        reverse_path_temporal_scores.unsqueeze(-1),
                        answer_plan_prior_for_paths,
                        answer_plan_prior_current_delta_for_paths.unsqueeze(-1),
                        answer_plan_prior_selected_delta_for_paths.unsqueeze(-1),
                        answer_features,
                    ],
                    dim=-1,
                )
            path_fusion_delta_batch = self.final_path_fusion_head(
                question_batch,
                path_event_embeddings,
                path_support_embeddings,
                projected_batch,
                path_pair_features_batch,
                final_path_features_batch,
            ).masked_fill(~path_mask, 0.0)
            final_path_logits_batch = (
                calibrated_path_logits_batch + path_tunnel_delta_batch + path_fusion_delta_batch
            ).masked_fill(~path_mask, 0.0)
        answer_plan_logits_batch = node_hidden.new_zeros((batch_size, max_event_count, len(ANSWER_PLAN_OUTPUTS)))
        if max_event_count > 0:
            answer_plan_features_batch = node_hidden.new_zeros((batch_size, max_event_count, ANSWER_PLAN_FEATURE_DIM))
            final_path_prob_batch = (
                torch.sigmoid(final_path_logits_batch).masked_fill(~path_mask, 0.0)
                if max_path_count > 0
                else node_hidden.new_zeros((batch_size, 0))
            )
            for batch_index, candidate_event_ids in enumerate(candidate_event_ids_batch):
                if not candidate_event_ids:
                    continue
                event_count = candidate_event_lengths[batch_index]
                path_event_indices = candidate_path_event_indices_batch[batch_index]
                max_calibrated_path_prob_by_event = _scatter_amax_by_indices(
                    calibrated_path_prob_batch[batch_index, : candidate_path_lengths[batch_index]],
                    path_event_indices,
                    size=event_count,
                    device=device,
                )
                max_final_path_prob_by_event = _scatter_amax_by_indices(
                    final_path_prob_batch[batch_index, : candidate_path_lengths[batch_index]],
                    path_event_indices,
                    size=event_count,
                    device=device,
                )
                temporal_event_indices = candidate_temporal_event_indices_batch[batch_index]
                max_temporal_prob_by_event = _scatter_amax_by_indices(
                    temporal_prob_batch[batch_index, : temporal_lengths[batch_index]],
                    temporal_event_indices,
                    size=event_count,
                    device=device,
                )
                question_features = dict(question_features_batch[batch_index] or {})
                semantic_target = clean_text(question_features.get("semantic_slot_target", ""))
                lowered_question = normalize_text(examples[batch_index].question)
                is_profile_query = bool(
                    question_features.get("asks_profile", False)
                    or semantic_target == "profile"
                    or "profile" in lowered_question
                    or "preference" in lowered_question
                    or "prefer" in lowered_question
                )
                is_temporal_query = bool(question_features.get("is_temporal", False))
                is_chain_query = bool(
                    question_features.get("asks_chain", False)
                    or "chain" in lowered_question
                    or "connected" in lowered_question
                    or "evidence" in lowered_question
                    or "changed" in lowered_question
                )
                answer_profile_prob = answer_type_probs_batch[batch_index, ANSWER_TYPE_TO_ID["profile"]]
                answer_multi_prob = answer_type_probs_batch[batch_index, ANSWER_TYPE_TO_ID["multi_evidence"]]
                answer_plan_features_batch[batch_index, :event_count] = torch.cat(
                    [
                        final_event_prob_batch[batch_index, :event_count].unsqueeze(-1),
                        calibrated_event_prob_batch[batch_index, :event_count].unsqueeze(-1),
                        torch.sigmoid(event_tunnel_support_logits_batch[batch_index, :event_count]).unsqueeze(-1),
                        torch.sigmoid(event_distractor_logits_batch[batch_index, :event_count]).unsqueeze(-1),
                        max_calibrated_path_prob_by_event.unsqueeze(-1),
                        max_final_path_prob_by_event.unsqueeze(-1),
                        max_temporal_prob_by_event.unsqueeze(-1),
                        node_hidden.new_full((event_count, 1), 1.0 if is_profile_query else 0.0),
                        node_hidden.new_full((event_count, 1), 1.0 if is_temporal_query else 0.0),
                        node_hidden.new_full((event_count, 1), 1.0 if is_chain_query else 0.0),
                        answer_profile_prob.reshape(1, 1).expand(event_count, 1).to(dtype=node_hidden.dtype),
                        answer_multi_prob.reshape(1, 1).expand(event_count, 1).to(dtype=node_hidden.dtype),
                    ],
                    dim=-1,
                )
            answer_plan_logits_batch = self.answer_plan_head(
                question_batch,
                gathered_event_embeddings,
                event_support_embeddings_batch,
                projected_batch,
                event_pair_features_batch,
                answer_plan_features_batch,
            ).masked_fill(~event_mask.unsqueeze(-1), 0.0)
        outputs: List[Dict[str, Any]] = []
        for batch_index, candidate_event_ids in enumerate(candidate_event_ids_batch):
            event_count = candidate_event_lengths[batch_index]
            path_count = candidate_path_lengths[batch_index]
            event_reverse_scores = node_hidden.new_zeros((event_count,))
            event_boundary_scores = node_hidden.new_zeros((event_count,))
            event_reverse_relations = node_hidden.new_zeros((event_count,))
            path_reverse_scores = node_hidden.new_zeros((path_count,))
            path_boundary_scores = node_hidden.new_zeros((path_count,))
            path_reverse_relations = node_hidden.new_zeros((path_count,))
            if event_count > 0:
                event_reverse_scores, event_boundary_scores, event_reverse_relations, _, _ = _reverse_competition_features(
                    final_event_prob_batch[batch_index, :event_count],
                    event_relation_strength_batch[batch_index, :event_count, :event_count],
                    valid_mask=event_mask[batch_index, :event_count],
                )
            if path_count > 0:
                path_reverse_scores, path_boundary_scores, path_reverse_relations, _, _ = _reverse_competition_features(
                    torch.sigmoid(final_path_logits_batch[batch_index, :path_count]),
                    path_relation_strength_batch[batch_index, :path_count, :path_count],
                    valid_mask=path_mask[batch_index, :path_count],
                )
            matrix_event_count = matrix_event_lengths[batch_index]
            matrix_index_tensor = torch.tensor(matrix_local_index_batch[batch_index], dtype=torch.long, device=device) if matrix_event_count > 0 else None
            matrix_path_count = matrix_path_lengths[batch_index]
            matrix_path_index_tensor = (
                torch.tensor(matrix_path_local_index_batch[batch_index], dtype=torch.long, device=device)
                if matrix_path_count > 0
                else None
            )
            outputs.append(
                {
                    "recall_event_ids": list(recall_event_ids_batch[batch_index]),
                    "recall_event_logits": recall_event_logits_batch[batch_index, : recall_event_lengths[batch_index]],
                    "candidate_event_ids": candidate_event_ids,
                    "event_pair_feature_mode": resolved_event_pair_feature_mode,
                    "base_event_logits": base_event_logits_batch[batch_index, :event_count],
                    "rerank_event_logits": rerank_event_logits_batch[batch_index, :event_count],
                    "matrix_event_ids": list(matrix_event_ids_batch[batch_index]),
                    "matrix_event_delta_logits": (
                        matrix_event_delta_batch[batch_index].index_select(0, matrix_index_tensor)
                        if matrix_index_tensor is not None
                        else node_hidden.new_zeros((0,))
                    ),
                    "event_distractor_logits": event_distractor_logits_batch[batch_index, :event_count],
                    "event_distractor_delta_logits": event_distractor_delta_batch[batch_index, :event_count],
                    "distractor_event_logits": distractor_event_logits_batch[batch_index, :event_count],
                    "event_calibration_delta_logits": event_calibration_delta_batch[batch_index, :event_count],
                    "calibrated_event_logits": calibrated_event_logits_batch[batch_index, :event_count],
                    "event_tunnel_support_logits": event_tunnel_support_logits_batch[batch_index, :event_count],
                    "event_tunnel_delta_logits": event_tunnel_delta_batch[batch_index, :event_count],
                    "event_fusion_delta_logits": event_fusion_delta_batch[batch_index, :event_count],
                    "event_logits": final_event_logits_batch[batch_index, :event_count],
                    "answer_plan_logits": answer_plan_logits_batch[batch_index, :event_count],
                    "answer_plan_output_names": list(ANSWER_PLAN_OUTPUTS),
                    "event_reverse_scores": event_reverse_scores,
                    "event_boundary_scores": event_boundary_scores,
                    "event_reverse_relations": event_reverse_relations,
                    "candidate_path_ids": candidate_path_ids_batch[batch_index],
                    "candidate_path_event_ids": candidate_path_event_ids_batch[batch_index],
                    "candidate_path_types": [
                        clean_text(path.get("type", ""))
                        for path in candidate_paths_batch[batch_index]
                    ],
                    "base_path_logits": base_path_logits_batch[batch_index, :path_count],
                    "rerank_path_logits": rerank_path_logits_batch[batch_index, :path_count],
                    "matrix_path_ids": list(matrix_path_ids_batch[batch_index]),
                    "matrix_path_delta_logits": (
                        matrix_path_delta_batch[batch_index].index_select(0, matrix_path_index_tensor)
                        if matrix_path_index_tensor is not None
                        else node_hidden.new_zeros((0,))
                    ),
                    "path_calibration_delta_logits": path_calibration_delta_batch[batch_index, :path_count],
                    "calibrated_path_logits": calibrated_path_logits_batch[batch_index, :path_count],
                    "path_tunnel_support_logits": path_tunnel_support_logits_batch[batch_index, :path_count],
                    "path_tunnel_delta_logits": path_tunnel_delta_batch[batch_index, :path_count],
                    "path_fusion_delta_logits": path_fusion_delta_batch[batch_index, :path_count],
                    "path_logits": final_path_logits_batch[batch_index, :path_count],
                    "path_reverse_scores": path_reverse_scores,
                    "path_boundary_scores": path_boundary_scores,
                    "path_reverse_relations": path_reverse_relations,
                    "candidate_temporal_path_ids": list(candidate_temporal_path_ids_batch[batch_index]),
                    "candidate_temporal_event_ids": list(candidate_temporal_event_ids_batch[batch_index]),
                    "candidate_temporal_node_ids": list(candidate_temporal_node_ids_batch[batch_index]),
                    "temporal_logits": temporal_logits_batch[batch_index, : temporal_lengths[batch_index]],
                    "answer_type_logits": answer_type_logits_batch[batch_index],
                    "memory_router_logits": memory_router_logits_batch[batch_index],
                    "memory_router_layers": list(MEMORY_ROUTER_LAYERS),
                    "question_semantic_logits": question_intent_outputs["semantic_logits"][batch_index],
                    "question_status_logits": question_intent_outputs["status_logits"][batch_index],
                    "question_time_granularity_logits": question_intent_outputs["time_granularity_logits"][batch_index],
                    "question_temporal_logit": question_intent_outputs["temporal_logits"][batch_index],
                    "effective_question_features": dict(question_features_batch[batch_index]),
                    "question_token_role_aux_loss": (
                        question_token_role_aux_loss
                        if batch_index == 0 and question_token_role_aux_loss is not None
                        else node_hidden.new_zeros(())
                    ),
                    "node_token_role_aux_loss": (
                        node_token_role_aux_loss
                        if batch_index == 0 and node_token_role_aux_loss is not None
                        else node_hidden.new_zeros(())
                    ),
                }
            )
        return outputs

    def checkpoint_payload(self, *, metadata: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        arch_metadata = node_memory_arch_metadata()
        return {
            "config": {
                "question_hash_buckets": QUESTION_HASH_BUCKETS,
                "node_hash_buckets": NODE_HASH_BUCKETS,
                "question_output_dim": QUESTION_OUTPUT_DIM,
                "node_output_dim": NODE_OUTPUT_DIM,
                "message_hidden_dim": MESSAGE_HIDDEN_DIM,
                **arch_metadata,
            },
            "state_dict": self.state_dict(),
            "metadata": {**dict(metadata or {}), **arch_metadata},
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any], *, device: torch.device | None = None) -> "LocomoNodeMemoryModel":
        validate_node_memory_checkpoint_payload(payload, context="LocomoNodeMemoryModel.from_checkpoint")
        model = cls()
        state_dict = dict(payload.get("state_dict", {}) or {})
        model._checkpoint_load_report = _load_compatible_state_dict(model, state_dict)
        if device is not None:
            model.to(device)
        return model


def _load_compatible_state_dict(module: nn.Module, state_dict: Mapping[str, Any]) -> Dict[str, List[str]]:
    narrowable_vocab_embeddings = {
        "node_encoder.node_type_embedding.weight",
        "message_passing.edge_embedding.weight",
        "event_subgraph_refiner.role_embedding.weight",
        "path_head.path_type_embedding.weight",
    }
    current_state = module.state_dict()
    compatible_state: Dict[str, Any] = {}
    skipped_keys: List[str] = []
    partially_loaded_keys: List[str] = []
    unexpected_keys: List[str] = []
    for key, value in dict(state_dict or {}).items():
        if key not in current_state:
            unexpected_keys.append(key)
            continue
        current_value = current_state[key]
        if tuple(current_value.shape) != tuple(value.shape):
            if (
                key in narrowable_vocab_embeddings
                and isinstance(current_value, Tensor)
                and isinstance(value, Tensor)
                and int(current_value.ndim) == int(value.ndim)
                and int(current_value.ndim) >= 1
                and int(value.shape[0]) >= int(current_value.shape[0])
                and tuple(value.shape[1:]) == tuple(current_value.shape[1:])
            ):
                slices = (slice(0, int(current_value.shape[0])),) + tuple(
                    slice(0, int(dim)) for dim in current_value.shape[1:]
                )
                compatible_state[key] = value[slices].to(
                    device=current_value.device,
                    dtype=current_value.dtype,
                )
                partially_loaded_keys.append(key)
                continue
            if (
                isinstance(current_value, Tensor)
                and isinstance(value, Tensor)
                and int(current_value.ndim) == int(value.ndim)
                and all(int(old_dim) <= int(new_dim) for old_dim, new_dim in zip(value.shape, current_value.shape))
            ):
                widened_value = torch.zeros_like(current_value)
                slices = tuple(slice(0, int(dim)) for dim in value.shape)
                widened_value[slices] = value.to(device=widened_value.device, dtype=widened_value.dtype)
                compatible_state[key] = widened_value
                partially_loaded_keys.append(key)
                continue
            skipped_keys.append(key)
            continue
        compatible_state[key] = value
    missing_keys, remaining_unexpected = module.load_state_dict(compatible_state, strict=False)
    return {
        "missing_keys": list(missing_keys),
        "unexpected_keys": [*unexpected_keys, *list(remaining_unexpected)],
        "skipped_keys": skipped_keys,
        "partially_loaded_keys": partially_loaded_keys,
    }


def _output_projection_has_signal(module: Any) -> bool:
    output_projection = getattr(module, "output_projection", None)
    if not isinstance(output_projection, nn.Linear):
        return False
    weight_nonzero = bool(torch.count_nonzero(output_projection.weight.detach()).item())
    bias_nonzero = bool(torch.count_nonzero(output_projection.bias.detach()).item()) if output_projection.bias is not None else False
    return weight_nonzero or bias_nonzero


def _sorted_score_ids(score_lookup: Mapping[str, Any], *, limit: int | None = None) -> List[str]:
    ranked = [
        event_id
        for event_id, _ in sorted(
            (
                (clean_text(item_id), float(score or 0.0))
                for item_id, score in dict(score_lookup or {}).items()
                if clean_text(item_id)
            ),
            key=lambda item: (-float(item[1]), item[0]),
        )
    ]
    if limit is None:
        return ranked
    return ranked[: max(0, int(limit))]


def _dominant_answer_type_from_scores(
    answer_type_scores: Mapping[str, Any],
    *,
    question_is_temporal: bool = False,
) -> str:
    normalized_scores = {
        clean_text(answer_type): float(value or 0.0)
        for answer_type, value in dict(answer_type_scores or {}).items()
        if clean_text(answer_type)
    }
    if not normalized_scores:
        return "time" if question_is_temporal else "event_text"
    return max(normalized_scores.items(), key=lambda item: (float(item[1]), item[0]))[0]


def _safe_probability_to_logit(value: Any) -> float:
    probability = min(1.0 - 1e-6, max(1e-6, float(value or 0.0)))
    return math.log(probability / (1.0 - probability))


def _safe_sigmoid_scalar(value: float) -> float:
    if value >= 0:
        z = math.exp(-float(value))
        return 1.0 / (1.0 + z)
    z = math.exp(float(value))
    return z / (1.0 + z)


def _event_turn_index_from_node(event_id: str, node: Mapping[str, Any]) -> int:
    metadata = dict(node.get("metadata", {}) or {})
    for value in (
        node.get("turn_index"),
        metadata.get("turn_index"),
        node.get("turn"),
        metadata.get("turn"),
    ):
        try:
            turn_index = int(value)
        except (TypeError, ValueError):
            continue
        if turn_index > 0:
            return turn_index
    for value in (
        clean_text(event_id),
        clean_text(node.get("dia_id", "")),
        clean_text(metadata.get("dia_id", "")),
    ):
        match = re.search(r"(?:realchat|turn|event)[^0-9]{0,4}(\d+)", value, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def _node_text_token_set(node: Mapping[str, Any]) -> set[str]:
    metadata = dict(node.get("metadata", {}) or {})
    texts = [
        clean_text(node.get("text", "")),
        clean_text(node.get("event_signature", metadata.get("event_signature", ""))),
        clean_text(metadata.get("tmcra_typed_event_signature", "")),
        " ".join(str(item) for item in list(metadata.get("tmcra_node_tags", []) or [])),
        " ".join(str(item) for item in list(metadata.get("tmcra_path_tags", []) or [])),
        clean_text(node.get("profile_value", metadata.get("profile_value", ""))),
        clean_text(node.get("time_value", metadata.get("time_value", ""))),
        clean_text(node.get("target_status", metadata.get("target_status", ""))),
    ]
    tokens: set[str] = set()
    for text in texts:
        tokens.update(_normalized_token_list(text))
    return {token for token in tokens if token}


def _node_scalar_features(node: Mapping[str, Any]) -> List[float]:
    metadata = dict(node.get("metadata", {}) or {})
    teacher_fields = dict(node.get("teacher_fields", {}) or {})
    turn_index = float(node.get("turn_index", metadata.get("turn_index", 0)) or 0.0)
    salience = float(node.get("salience", metadata.get("salience", 0.7)) or 0.7)
    confidence = float(node.get("confidence", metadata.get("confidence", 0.7)) or 0.7)
    time_present = 1.0 if clean_text(metadata.get("time_value", node.get("time_value", ""))) else 0.0
    profile_present = 1.0 if clean_text(metadata.get("profile_value", node.get("profile_value", ""))) else 0.0
    status_present = 1.0 if clean_text(metadata.get("target_status", node.get("target_status", ""))) else 0.0
    teacher_present = 1.0 if teacher_fields else 0.0
    token_count = min(32.0, float(len(tokenize_text(node.get("text", "")))))
    return [
        salience,
        confidence,
        min(1.0, turn_index / 64.0),
        time_present,
        profile_present,
        status_present,
        teacher_present,
        token_count / 32.0,
    ]


def _event_encoder_text(node: Mapping[str, Any], support_payload: Mapping[str, Any] | None = None) -> str:
    support = dict(support_payload or {})
    source_turn_texts = [clean_text(item) for item in list(support.get("source_turn_texts", []) or []) if clean_text(item)]
    profile_texts = [clean_text(item) for item in list(support.get("profile_texts", []) or []) if clean_text(item)]
    time_texts = [clean_text(item) for item in list(support.get("time_texts", []) or []) if clean_text(item)]
    status_texts = [clean_text(item) for item in list(support.get("status_texts", []) or []) if clean_text(item)]
    metadata = dict(node.get("metadata", {}) or {})
    typed_signature_text = " ".join(
        dedupe_texts(
            [
                clean_text(metadata.get("tmcra_typed_event_signature", "")),
                " ".join(str(item) for item in list(metadata.get("tmcra_node_tags", []) or [])),
                " ".join(str(item) for item in list(metadata.get("tmcra_path_tags", []) or [])),
                clean_text(metadata.get("tmcra_tunnel_group_key", "")),
            ],
            max_items=8,
        )
    )
    return " ".join(
        dedupe_texts(
            [
                *source_turn_texts[:1],
                clean_text(node.get("text", "")),
                clean_text(node.get("event_signature", metadata.get("event_signature", ""))),
                typed_signature_text,
                *profile_texts[:1],
                *time_texts[:1],
                *status_texts[:1],
            ],
            max_items=6,
        )
    )


def tensorize_graph(graph: Mapping[str, Any], *, device: torch.device) -> Dict[str, Any]:
    nodes = [dict(node) for node in list(graph.get("nodes", []) or [])]
    node_ids = [clean_text(node.get("id", "")) for node in nodes]
    node_texts = [clean_text(node.get("text", "")) for node in nodes]
    event_node_ids = [node_id for node_id, node in zip(node_ids, nodes) if clean_text(node.get("type", "")) == "event"]
    time_node_ids = [node_id for node_id, node in zip(node_ids, nodes) if clean_text(node.get("type", "")) == "time"]
    node_type_ids = torch.tensor(
        [NODE_TYPE_TO_ID.get(clean_text(node.get("type", "")), 0) for node in nodes],
        dtype=torch.long,
        device=device,
    )
    node_scalar_features = torch.zeros((0, NODE_SCALAR_DIM), dtype=torch.float32, device=device)
    node_id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    edge_src_values: List[int] = []
    edge_dst_values: List[int] = []
    edge_type_values: List[int] = []
    for edge in list(graph.get("edges", []) or []):
        source = clean_text(edge.get("source", ""))
        target = clean_text(edge.get("target", ""))
        if source not in node_id_to_index or target not in node_id_to_index:
            continue
        edge_src_values.append(node_id_to_index[source])
        edge_dst_values.append(node_id_to_index[target])
        edge_type_values.append(EDGE_TYPE_TO_ID.get(clean_text(edge.get("type", "")), 0))
    edge_src = torch.tensor(edge_src_values, dtype=torch.long, device=device)
    edge_dst = torch.tensor(edge_dst_values, dtype=torch.long, device=device)
    edge_type_ids = torch.tensor(edge_type_values, dtype=torch.long, device=device)
    node_lookup = {node_id: dict(node) for node_id, node in zip(node_ids, nodes)}
    normalized_paths: List[Dict[str, Any]] = []
    paths_by_event_id: Dict[str, List[Dict[str, Any]]] = {}
    temporal_paths: List[Dict[str, Any]] = []
    support_texts_by_event: Dict[str, Dict[str, List[str]]] = {}
    support_node_ids_by_event_and_type: Dict[str, Dict[str, List[str]]] = {}
    for path in list(graph.get("paths", []) or []):
        normalized_path = dict(path)
        event_id = clean_text(normalized_path.get("event_id", ""))
        normalized_node_ids = [clean_text(item) for item in list(normalized_path.get("node_ids", []) or []) if clean_text(item)]
        support_node_id = clean_text(normalized_node_ids[2]) if len(normalized_node_ids) >= 3 else ""
        path_type = clean_text(normalized_path.get("type", ""))
        if not event_id or event_id not in node_id_to_index or not support_node_id or support_node_id not in node_id_to_index:
            continue
        normalized_path["event_id"] = event_id
        normalized_path["node_ids"] = normalized_node_ids
        normalized_path["_support_node_id"] = support_node_id
        normalized_path["_path_type"] = path_type
        normalized_paths.append(normalized_path)
        paths_by_event_id.setdefault(event_id, []).append(normalized_path)
        support_node_ids_by_event_and_type.setdefault(event_id, {}).setdefault(path_type, []).append(support_node_id)
        support_node = dict(node_lookup.get(support_node_id, {}) or {})
        support_text = clean_text(support_node.get("text", ""))
        if support_text:
            event_support = support_texts_by_event.setdefault(
                event_id,
                {
                    "source_turn_texts": [],
                    "profile_texts": [],
                    "time_texts": [],
                    "status_texts": [],
                },
            )
            if path_type == "speaker_event_source_turn":
                event_support["source_turn_texts"].append(support_text)
            elif path_type == "speaker_event_profile":
                event_support["profile_texts"].append(support_text)
            elif path_type == "speaker_event_time":
                event_support["time_texts"].append(support_text)
                temporal_paths.append(normalized_path)
            elif path_type == "speaker_event_status":
                event_support["status_texts"].append(support_text)
    event_support_lookup = {
        event_id: {key: dedupe_texts(values, max_items=4) for key, values in payload.items()}
        for event_id, payload in support_texts_by_event.items()
    }
    if _env_flag("TMCRA_NODE_MEMORY_EVENT_SUPPORT_CONTEXT", default=True):
        for index, (node_id, node) in enumerate(zip(node_ids, nodes)):
            if clean_text(node.get("type", "")) != "event":
                continue
            encoder_text = _event_encoder_text(node, event_support_lookup.get(node_id, {}))
            if encoder_text:
                nodes[index]["text"] = encoder_text
                node_texts[index] = encoder_text
    node_text_hash_indices = [_text_hash_indices(text, NODE_HASH_BUCKETS) for text in node_texts]
    node_lookup = {node_id: dict(node) for node_id, node in zip(node_ids, nodes)}
    node_scalar_features = torch.tensor(
        [_node_scalar_features(node) for node in nodes],
        dtype=torch.float32,
        device=device,
    ) if nodes else torch.zeros((0, NODE_SCALAR_DIM), dtype=torch.float32, device=device)
    support_node_ids_by_event_and_type = {
        event_id: {path_type: dedupe_texts(values) for path_type, values in payload.items()}
        for event_id, payload in support_node_ids_by_event_and_type.items()
    }
    support_node_ids_by_event = {
        event_id: dedupe_texts(
            support_node_id
            for support_node_ids in path_type_map.values()
            for support_node_id in support_node_ids
        )
        for event_id, path_type_map in support_node_ids_by_event_and_type.items()
    }
    return {
        "node_ids": node_ids,
        "node_texts": node_texts,
        "node_text_hash_indices": node_text_hash_indices,
        "node_by_id": node_lookup,
        "event_node_ids": event_node_ids,
        "time_node_ids": time_node_ids,
        "node_type_ids": node_type_ids,
        "node_scalar_features": node_scalar_features,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_type_ids": edge_type_ids,
        "node_id_to_index": node_id_to_index,
        "paths": normalized_paths,
        "paths_by_event_id": paths_by_event_id,
        "temporal_paths": temporal_paths,
        "event_support_lookup": event_support_lookup,
        "support_node_ids_by_event": support_node_ids_by_event,
        "support_node_ids_by_event_and_type": support_node_ids_by_event_and_type,
    }


def _path_support_node_id(path: Mapping[str, Any]) -> str:
    cached = clean_text(path.get("_support_node_id", ""))
    if cached:
        return cached
    node_ids = [clean_text(item) for item in list(path.get("node_ids", []) or []) if clean_text(item)]
    return clean_text(node_ids[2]) if len(node_ids) >= 3 else ""


def _path_type(path: Mapping[str, Any]) -> str:
    return clean_text(path.get("_path_type", path.get("type", "")))


def _build_event_support_lookup(graph_tensors: Mapping[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    precomputed = graph_tensors.get("event_support_lookup")
    if isinstance(precomputed, Mapping):
        return {
            clean_text(event_id): {
                clean_text(key): [clean_text(item) for item in list(values or []) if clean_text(item)]
                for key, values in dict(payload or {}).items()
            }
            for event_id, payload in dict(precomputed).items()
            if clean_text(event_id)
        }
    node_lookup = dict(graph_tensors.get("node_by_id", {}) or {})
    support_lookup: Dict[str, Dict[str, List[str]]] = {}
    for path in list(graph_tensors.get("paths", []) or []):
        event_id = clean_text(path.get("event_id", ""))
        support_node_id = _path_support_node_id(path)
        if not event_id or not support_node_id:
            continue
        support_node = dict(node_lookup.get(support_node_id, {}) or {})
        support_text = clean_text(support_node.get("text", ""))
        if not support_text:
            continue
        event_support = support_lookup.setdefault(
            event_id,
            {
                "source_turn_texts": [],
                "profile_texts": [],
                "time_texts": [],
                "status_texts": [],
            },
        )
        normalized_path_type = _path_type(path)
        if normalized_path_type == "speaker_event_source_turn":
            event_support["source_turn_texts"].append(support_text)
        elif normalized_path_type == "speaker_event_profile":
            event_support["profile_texts"].append(support_text)
        elif normalized_path_type == "speaker_event_time":
            event_support["time_texts"].append(support_text)
        elif normalized_path_type == "speaker_event_status":
            event_support["status_texts"].append(support_text)
    for event_id, payload in support_lookup.items():
        support_lookup[event_id] = {
            key: dedupe_texts(values, max_items=4)
            for key, values in payload.items()
        }
    return support_lookup


def _preferred_support_path_types(question_features: Mapping[str, Any]) -> List[str]:
    semantic_target = clean_text(question_features.get("semantic_slot_target", ""))
    target_status_target = clean_text(question_features.get("target_status_target", ""))
    if bool(question_features.get("is_temporal", False)) or semantic_target == "event_time":
        return ["speaker_event_time", "speaker_event_source_turn", "speaker_event_status", "speaker_event_profile"]
    if semantic_target in {"identity", "research_topic", "education", "occupation", "profile"}:
        return ["speaker_event_profile", "speaker_event_source_turn", "speaker_event_status", "speaker_event_time"]
    if target_status_target:
        return ["speaker_event_status", "speaker_event_source_turn", "speaker_event_time", "speaker_event_profile"]
    return ["speaker_event_source_turn", "speaker_event_time", "speaker_event_profile", "speaker_event_status"]


def _candidate_paths_for_event_ids(
    all_paths: Sequence[Mapping[str, Any]],
    candidate_event_ids: Sequence[str],
    *,
    node_id_to_index: Mapping[str, int],
) -> List[Dict[str, Any]]:
    candidate_event_set = {clean_text(item) for item in candidate_event_ids if clean_text(item)}
    return [
        dict(path)
        for path in all_paths
        if clean_text(path.get("event_id", "")) in candidate_event_set
        and _path_support_node_id(path) in node_id_to_index
    ]


def _support_node_ids_by_event_and_type(candidate_paths: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    support_node_ids_by_event: Dict[str, Dict[str, List[str]]] = {}
    for path in candidate_paths:
        event_id = clean_text(path.get("event_id", ""))
        support_node_id = _path_support_node_id(path)
        path_type = _path_type(path)
        if not event_id or not support_node_id:
            continue
        support_node_ids_by_event.setdefault(event_id, {}).setdefault(path_type, []).append(support_node_id)
    return {
        event_id: {
            path_type: dedupe_texts(support_node_ids)
            for path_type, support_node_ids in path_type_map.items()
        }
        for event_id, path_type_map in support_node_ids_by_event.items()
    }


def _support_node_ids_by_event(candidate_paths: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    support_node_ids_by_event: Dict[str, List[str]] = {}
    typed_support = _support_node_ids_by_event_and_type(candidate_paths)
    return {
        event_id: dedupe_texts(
            support_node_id
            for support_node_ids in path_type_map.values()
            for support_node_id in support_node_ids
        )
        for event_id, path_type_map in typed_support.items()
    }


def _aggregate_event_support_embeddings(
    node_hidden: Tensor,
    node_id_to_index: Mapping[str, int],
    candidate_event_ids_batch: Sequence[Sequence[str]],
    candidate_paths_batch: Sequence[Sequence[Mapping[str, Any]]] | None = None,
    *,
    support_node_ids_by_event: Mapping[str, Sequence[str]] | None = None,
    support_node_ids_by_event_and_type: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    question_features_batch: Sequence[Mapping[str, Any]] | None = None,
) -> Tensor:
    batch_size = len(candidate_event_ids_batch)
    max_event_count = max((len(candidate_event_ids) for candidate_event_ids in candidate_event_ids_batch), default=0)
    hidden_dim = int(node_hidden.size(-1)) if node_hidden.dim() > 1 else 0
    support_embeddings = node_hidden.new_zeros((batch_size, max_event_count, hidden_dim))
    resolved_question_features = list(question_features_batch or [{} for _ in candidate_event_ids_batch])
    resolved_candidate_paths_batch = list(candidate_paths_batch or [[] for _ in candidate_event_ids_batch])
    for batch_index, candidate_event_ids in enumerate(candidate_event_ids_batch):
        if support_node_ids_by_event is not None and support_node_ids_by_event_and_type is not None:
            local_support_node_ids_by_event = support_node_ids_by_event
            local_support_node_ids_by_event_and_type = support_node_ids_by_event_and_type
        else:
            local_support_node_ids_by_event = _support_node_ids_by_event(resolved_candidate_paths_batch[batch_index])
            local_support_node_ids_by_event_and_type = _support_node_ids_by_event_and_type(resolved_candidate_paths_batch[batch_index])
        preferred_path_types = _preferred_support_path_types(
            resolved_question_features[batch_index] if batch_index < len(resolved_question_features) else {}
        )
        for event_index, event_id in enumerate(candidate_event_ids):
            event_support_by_type = dict(local_support_node_ids_by_event_and_type.get(event_id, {}) or {})
            support_node_ids: List[str] = []
            for path_type in preferred_path_types:
                preferred_support_ids = [
                    support_node_id
                    for support_node_id in event_support_by_type.get(path_type, [])
                    if support_node_id in node_id_to_index
                ]
                if preferred_support_ids:
                    support_node_ids = preferred_support_ids
                    break
            if not support_node_ids:
                support_node_ids = [
                    support_node_id
                    for support_node_id in local_support_node_ids_by_event.get(event_id, [])
                    if support_node_id in node_id_to_index
                ]
            if not support_node_ids:
                continue
            support_indices = torch.tensor(
                [node_id_to_index[support_node_id] for support_node_id in support_node_ids],
                dtype=torch.long,
                device=node_hidden.device,
            )
            support_embeddings[batch_index, event_index] = node_hidden.index_select(0, support_indices).mean(dim=0)
    return support_embeddings


def _aggregate_event_support_slot_embeddings(
    node_hidden: Tensor,
    node_id_to_index: Mapping[str, int],
    candidate_event_ids_batch: Sequence[Sequence[str]],
    candidate_paths_batch: Sequence[Sequence[Mapping[str, Any]]] | None = None,
    *,
    support_node_ids_by_event_and_type: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> tuple[Tensor, Tensor]:
    batch_size = len(candidate_event_ids_batch)
    max_event_count = max((len(candidate_event_ids) for candidate_event_ids in candidate_event_ids_batch), default=0)
    slot_count = len(EVENT_SUPPORT_PATH_TYPE_ORDER)
    hidden_dim = int(node_hidden.size(-1)) if node_hidden.dim() > 1 else 0
    support_slot_embeddings = node_hidden.new_zeros((batch_size, max_event_count, slot_count, hidden_dim))
    support_slot_mask = torch.zeros((batch_size, max_event_count, slot_count), dtype=torch.bool, device=node_hidden.device)
    resolved_candidate_paths_batch = list(candidate_paths_batch or [[] for _ in candidate_event_ids_batch])
    for batch_index, candidate_event_ids in enumerate(candidate_event_ids_batch):
        if support_node_ids_by_event_and_type is not None:
            local_support_node_ids_by_event_and_type = support_node_ids_by_event_and_type
        else:
            local_support_node_ids_by_event_and_type = _support_node_ids_by_event_and_type(
                resolved_candidate_paths_batch[batch_index]
            )
        for event_index, event_id in enumerate(candidate_event_ids):
            event_support_by_type = dict(local_support_node_ids_by_event_and_type.get(event_id, {}) or {})
            for slot_index, path_type in enumerate(EVENT_SUPPORT_PATH_TYPE_ORDER):
                support_node_ids = [
                    support_node_id
                    for support_node_id in event_support_by_type.get(path_type, [])
                    if support_node_id in node_id_to_index
                ]
                if not support_node_ids:
                    continue
                support_indices = torch.tensor(
                    [node_id_to_index[support_node_id] for support_node_id in support_node_ids],
                    dtype=torch.long,
                    device=node_hidden.device,
                )
                support_slot_embeddings[batch_index, event_index, slot_index] = node_hidden.index_select(
                    0,
                    support_indices,
                ).mean(dim=0)
                support_slot_mask[batch_index, event_index, slot_index] = True
    return support_slot_embeddings, support_slot_mask


def _select_rerank_event_ids(
    all_event_ids: Sequence[str],
    recall_logits: Tensor,
    *,
    positive_event_ids: Sequence[str] = (),
    forced_event_ids: Sequence[str] = (),
    candidate_limit: int = DEFAULT_RUNTIME_RECALL_TOP_K,
) -> List[str]:
    available_event_ids = {clean_text(item) for item in all_event_ids if clean_text(item)}
    ranked_indices = torch.argsort(recall_logits.detach(), descending=True)
    ranked_event_ids = [clean_text(all_event_ids[int(index)]) for index in ranked_indices.tolist() if clean_text(all_event_ids[int(index)])]
    positives = [clean_text(item) for item in positive_event_ids if clean_text(item) and clean_text(item) in available_event_ids]
    forced = [clean_text(item) for item in forced_event_ids if clean_text(item) and clean_text(item) in available_event_ids]
    resolved_limit = max(1, int(candidate_limit or DEFAULT_RUNTIME_RECALL_TOP_K))
    return dedupe_texts([*positives, *forced, *ranked_event_ids[:resolved_limit]])


def _resolved_hard_negative_event_ids_for_example(example: Any) -> List[str]:
    explicit = [
        clean_text(item)
        for item in list(getattr(example, "hard_negative_event_ids", []) or [])
        if clean_text(item)
    ]
    if explicit:
        return dedupe_texts(explicit)
    return [
        clean_text(item)
        for item in list(getattr(example, "negative_event_ids", []) or [])
        if clean_text(item)
    ]


def _resolved_hard_negative_event_ids_from_supervision(supervision: Mapping[str, Any]) -> frozenset[str]:
    explicit = frozenset(
        clean_text(item)
        for item in list(supervision.get("hard_negative_event_ids", []) or [])
        if clean_text(item)
    )
    if explicit:
        return explicit
    return frozenset(
        clean_text(item)
        for item in list(supervision.get("negative_event_ids", []) or [])
        if clean_text(item)
    )


def _select_matrix_event_ids(
    candidate_event_ids: Sequence[str],
    recall_ranked_event_ids: Sequence[str],
    *,
    positive_event_ids: Sequence[str] = (),
    hard_negative_event_ids: Sequence[str] = (),
    matrix_top_k: int = DEFAULT_MATRIX_EVENT_TOP_K,
    recall_seed_top_k: int = DEFAULT_MATRIX_EVENT_RECALL_SEED_TOP_K,
    hard_negative_limit: int = DEFAULT_MATRIX_EVENT_HARD_NEGATIVE_LIMIT,
) -> List[str]:
    available = {clean_text(item) for item in candidate_event_ids if clean_text(item)}
    recall_ranked = [clean_text(item) for item in recall_ranked_event_ids if clean_text(item) and clean_text(item) in available]
    if not recall_ranked:
        recall_ranked = [clean_text(item) for item in candidate_event_ids if clean_text(item)]
    positives = [clean_text(item) for item in positive_event_ids if clean_text(item) and clean_text(item) in available]
    positive_set = {clean_text(item) for item in positives if clean_text(item)}
    hard_negative_set = {clean_text(item) for item in hard_negative_event_ids if clean_text(item) and clean_text(item) in available}
    resolved_top_k = max(1, int(matrix_top_k or DEFAULT_MATRIX_EVENT_TOP_K))
    if not positives and not hard_negative_set:
        return dedupe_texts(recall_ranked, max_items=resolved_top_k)
    recall_seed = recall_ranked[: max(1, int(recall_seed_top_k or DEFAULT_MATRIX_EVENT_RECALL_SEED_TOP_K))]
    hard_negatives = [
        event_id
        for event_id in recall_ranked
        if event_id in hard_negative_set and event_id not in positive_set
    ][: max(0, int(hard_negative_limit or DEFAULT_MATRIX_EVENT_HARD_NEGATIVE_LIMIT))]
    return dedupe_texts([*positives, *hard_negatives, *recall_seed], max_items=resolved_top_k)


def _select_matrix_path_ids(
    candidate_path_ids: Sequence[str],
    ranked_path_ids: Sequence[str],
    *,
    positive_path_ids: Sequence[str] = (),
    hard_negative_path_ids: Sequence[str] = (),
    matrix_top_k: int = DEFAULT_MATRIX_PATH_TOP_K,
    ranked_seed_top_k: int = DEFAULT_MATRIX_PATH_RECALL_SEED_TOP_K,
    hard_negative_limit: int = DEFAULT_MATRIX_PATH_HARD_NEGATIVE_LIMIT,
) -> List[str]:
    available = {clean_text(item) for item in candidate_path_ids if clean_text(item)}
    ranked = [clean_text(item) for item in ranked_path_ids if clean_text(item) and clean_text(item) in available]
    if not ranked:
        ranked = [clean_text(item) for item in candidate_path_ids if clean_text(item)]
    positives = [clean_text(item) for item in positive_path_ids if clean_text(item) and clean_text(item) in available]
    positive_set = {clean_text(item) for item in positives if clean_text(item)}
    hard_negative_set = {clean_text(item) for item in hard_negative_path_ids if clean_text(item) and clean_text(item) in available}
    resolved_top_k = max(1, int(matrix_top_k or DEFAULT_MATRIX_PATH_TOP_K))
    if not positives and not hard_negative_set:
        return dedupe_texts(ranked, max_items=resolved_top_k)
    ranked_seed = ranked[: max(1, int(ranked_seed_top_k or DEFAULT_MATRIX_PATH_RECALL_SEED_TOP_K))]
    hard_negatives = [
        path_id
        for path_id in ranked
        if path_id in hard_negative_set and path_id not in positive_set
    ][: max(0, int(hard_negative_limit or DEFAULT_MATRIX_PATH_HARD_NEGATIVE_LIMIT))]
    return dedupe_texts([*positives, *hard_negatives, *ranked_seed], max_items=resolved_top_k)


def load_graph_cache(
    graph_dir: Path,
    *,
    device: torch.device,
    cache_dir: Path | None = None,
) -> Dict[str, GraphCacheItem]:
    cache: Dict[str, GraphCacheItem] = {}
    cpu_device = torch.device("cpu")
    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else None
    if resolved_cache_dir is not None:
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    gc_enabled = gc.isenabled()
    if gc_enabled:
        gc.disable()
    try:
        for path in _iter_graph_paths(graph_dir):
            cached_payload = None
            source_signature = _graph_source_signature(path)
            if resolved_cache_dir is not None:
                cached_payload = _load_graph_tensor_cache(
                    _graph_tensor_cache_path(resolved_cache_dir, path),
                    source_signature=source_signature,
                )
            if cached_payload is not None:
                graph = dict(cached_payload["graph"])
                tensors = _graph_tensor_mapping_to_device(cached_payload["tensors"], device, non_blocking=device.type == "cuda")
            else:
                graph = dict(read_json(path))
                cache_tensors = tensorize_graph(graph, device=cpu_device)
                if resolved_cache_dir is not None:
                    _write_graph_tensor_cache(
                        _graph_tensor_cache_path(resolved_cache_dir, path),
                        source_signature=source_signature,
                        graph=graph,
                        tensors=cache_tensors,
                    )
                tensors = _graph_tensor_mapping_to_device(cache_tensors, device, non_blocking=device.type == "cuda")
            tensors = dict(_ensure_graph_scoring_feature_cache(tensors))
            conversation_id = clean_text(graph.get("conversation_id", path.stem))
            cache[conversation_id] = GraphCacheItem(graph=graph, tensors=tensors)
    finally:
        if gc_enabled:
            gc.enable()
    return cache


def _iter_graph_paths(graph_dir: Path) -> Iterator[Path]:
    resolved_graph_dir = Path(graph_dir)
    names: List[str] = []
    with os.scandir(resolved_graph_dir) as iterator:
        for entry in iterator:
            if not entry.is_file():
                continue
            name = str(entry.name)
            if not name.endswith(".json"):
                continue
            names.append(name)
    names.sort()
    for name in names:
        yield resolved_graph_dir / name


def _load_query_example_from_raw_line(path: Path, offset: int, raw_line: bytes) -> QueryTrainingExample:
    if not raw_line:
        raise RuntimeError(f"Missing JSONL row at byte offset {offset} in {path}")
    try:
        payload = dict(json.loads(raw_line.decode("utf-8")))
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Failed to decode JSONL row at byte offset {offset} in {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse JSONL row at byte offset {offset} in {path}: "
            f"line {exc.lineno} column {exc.colno} char {exc.pos}: {exc.msg}"
        ) from exc
    try:
        return QueryTrainingExample.from_dict(payload)
    except Exception as exc:
        raise RuntimeError(f"Failed to normalize JSONL row at byte offset {offset} in {path}: {exc}") from exc


def iter_query_examples(
    path: Path,
    *,
    skip_bad_rows: bool = False,
    error_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> Iterator[QueryTrainingExample]:
    for row in load_query_examples(path, skip_bad_rows=skip_bad_rows, error_callback=error_callback):
        yield row


class IndexedQueryExampleStore(Sequence[QueryTrainingExample]):
    def __init__(
        self,
        path: Path,
        *,
        skip_bad_rows: bool = False,
        error_callback: Callable[[Dict[str, Any]], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self._skip_bad_rows = bool(skip_bad_rows)
        self._error_callback = error_callback
        self._row_offsets: List[int] = []
        self._conversation_offsets: "OrderedDict[str, List[int]]" = OrderedDict()
        self._build_index()

    def _build_index(self) -> None:
        if not self._skip_bad_rows:
            for offset, row in iter_jsonl_with_offsets(self.path):
                row_index = len(self._row_offsets)
                conversation_id = clean_text(row.get("conversation_id", "")) or f"__row_{row_index}"
                self._row_offsets.append(int(offset))
                self._conversation_offsets.setdefault(conversation_id, []).append(int(offset))
            return
        if not self.path.exists():
            return
        with self.path.open("rb") as handle:
            for file_line_number in itertools.count(start=1):
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                if not raw_line.strip():
                    continue
                try:
                    example = _load_query_example_from_raw_line(self.path, int(offset), raw_line)
                except Exception as exc:
                    self._report_bad_row(
                        offset=int(offset),
                        file_line_number=int(file_line_number),
                        exc=exc,
                    )
                    continue
                row_index = len(self._row_offsets)
                conversation_id = clean_text(example.conversation_id) or f"__row_{row_index}"
                self._row_offsets.append(int(offset))
                self._conversation_offsets.setdefault(conversation_id, []).append(int(offset))

    def _report_bad_row(self, *, offset: int, file_line_number: int, exc: Exception) -> None:
        if self._error_callback is None:
            return
        self._error_callback(
            {
                "path": str(self.path),
                "offset": int(offset),
                "file_line_number": int(file_line_number),
                "error_type": type(exc).__name__,
                "error_message": clean_text(str(exc)),
            }
        )

    def _load_batch_from_offsets(self, offsets: Sequence[int]) -> List[QueryTrainingExample]:
        resolved_offsets = [int(offset) for offset in offsets]
        if not resolved_offsets:
            return []
        rows: List[QueryTrainingExample] = []
        with self.path.open("rb") as handle:
            for offset in resolved_offsets:
                handle.seek(offset)
                raw_line = handle.readline()
                try:
                    rows.append(_load_query_example_from_raw_line(self.path, int(offset), raw_line))
                except Exception as exc:
                    if not self._skip_bad_rows:
                        raise
                    self._report_bad_row(offset=int(offset), file_line_number=0, exc=exc)
        return rows

    def _batch_spec_from_offsets(self, offsets: Sequence[int]) -> QueryBatchSpec:
        return QueryBatchSpec(
            sources=(
                QueryBatchSourceOffsets(
                    path=str(self.path),
                    offsets=tuple(int(offset) for offset in offsets),
                    skip_bad_rows=bool(self._skip_bad_rows),
                ),
            ),
        )

    def conversation_offset_items(self, *, shuffle: bool, seed: int) -> Iterator[tuple[str, List[int]]]:
        rng = random.Random(seed)
        conversation_ids = list(self._conversation_offsets)
        if shuffle:
            rng.shuffle(conversation_ids)
        for conversation_id in conversation_ids:
            offsets = list(self._conversation_offsets.get(conversation_id, []))
            if shuffle:
                rng.shuffle(offsets)
            yield conversation_id, offsets

    def conversation_ids(self) -> List[str]:
        return list(self._conversation_offsets)

    def iter_batches(
        self,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterator[List[QueryTrainingExample]]:
        resolved_batch_size = max(1, int(batch_size))
        current_offsets: List[int] = []
        for _, offsets in self.conversation_offset_items(shuffle=shuffle, seed=seed):
            if len(offsets) > resolved_batch_size:
                if current_offsets:
                    yield self._load_batch_from_offsets(current_offsets)
                    current_offsets = []
                for index in range(0, len(offsets), resolved_batch_size):
                    yield self._load_batch_from_offsets(offsets[index : index + resolved_batch_size])
                continue
            if current_offsets and len(current_offsets) + len(offsets) > resolved_batch_size:
                yield self._load_batch_from_offsets(current_offsets)
                current_offsets = []
            current_offsets.extend(offsets)
        if current_offsets:
            yield self._load_batch_from_offsets(current_offsets)

    def iter_batch_specs(
        self,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterator[QueryBatchSpec]:
        resolved_batch_size = max(1, int(batch_size))
        current_offsets: List[int] = []
        for _, offsets in self.conversation_offset_items(shuffle=shuffle, seed=seed):
            if len(offsets) > resolved_batch_size:
                if current_offsets:
                    yield self._batch_spec_from_offsets(current_offsets)
                    current_offsets = []
                for index in range(0, len(offsets), resolved_batch_size):
                    yield self._batch_spec_from_offsets(offsets[index : index + resolved_batch_size])
                continue
            if current_offsets and len(current_offsets) + len(offsets) > resolved_batch_size:
                yield self._batch_spec_from_offsets(current_offsets)
                current_offsets = []
            current_offsets.extend(offsets)
        if current_offsets:
            yield self._batch_spec_from_offsets(current_offsets)

    def __len__(self) -> int:
        return len(self._row_offsets)

    def __getitem__(self, index: int | slice) -> QueryTrainingExample | List[QueryTrainingExample]:
        if isinstance(index, slice):
            return self._load_batch_from_offsets(self._row_offsets[index])
        resolved_index = int(index)
        if resolved_index < 0:
            resolved_index += len(self._row_offsets)
        if resolved_index < 0 or resolved_index >= len(self._row_offsets):
            raise IndexError(resolved_index)
        return self._load_batch_from_offsets([self._row_offsets[resolved_index]])[0]

    def __iter__(self) -> Iterator[QueryTrainingExample]:
        with self.path.open("rb") as handle:
            for offset in self._row_offsets:
                handle.seek(offset)
                raw_line = handle.readline()
                try:
                    yield _load_query_example_from_raw_line(self.path, int(offset), raw_line)
                except Exception as exc:
                    if not self._skip_bad_rows:
                        raise
                    self._report_bad_row(offset=int(offset), file_line_number=0, exc=exc)


class CombinedQueryExampleStore(Sequence[QueryTrainingExample]):
    def __init__(self, sources: Sequence[IndexedQueryExampleStore]) -> None:
        self._sources = [source for source in sources if len(source) > 0]
        self._row_count = sum(len(source) for source in self._sources)

    def __len__(self) -> int:
        return self._row_count

    def __getitem__(self, index: int | slice) -> QueryTrainingExample | List[QueryTrainingExample]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return [self[position] for position in range(start, stop, step)]
            rows: List[QueryTrainingExample] = []
            for position in range(start, stop):
                rows.append(self[position])
            return rows
        resolved_index = int(index)
        if resolved_index < 0:
            resolved_index += len(self)
        if resolved_index < 0 or resolved_index >= len(self):
            raise IndexError(resolved_index)
        for source in self._sources:
            if resolved_index < len(source):
                return source[resolved_index]
            resolved_index -= len(source)
        raise IndexError(index)

    def __iter__(self) -> Iterator[QueryTrainingExample]:
        for source in self._sources:
            for row in source:
                yield row

    def iter_batches(
        self,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterator[List[QueryTrainingExample]]:
        resolved_batch_size = max(1, int(batch_size))
        rng = random.Random(seed)
        group_specs: List[tuple[IndexedQueryExampleStore, List[int]]] = []
        for source_index, source in enumerate(self._sources):
            source_seed = seed + ((source_index + 1) * 9973)
            for _, offsets in source.conversation_offset_items(shuffle=shuffle, seed=source_seed):
                group_specs.append((source, offsets))
        if shuffle:
            rng.shuffle(group_specs)
        current_batch: List[tuple[IndexedQueryExampleStore, List[int]]] = []
        current_batch_size = 0
        for source, offsets in group_specs:
            if len(offsets) > resolved_batch_size:
                if current_batch:
                    yield self._load_group_batch(current_batch)
                    current_batch = []
                    current_batch_size = 0
                for index in range(0, len(offsets), resolved_batch_size):
                    yield source._load_batch_from_offsets(offsets[index : index + resolved_batch_size])
                continue
            if current_batch and current_batch_size + len(offsets) > resolved_batch_size:
                yield self._load_group_batch(current_batch)
                current_batch = []
                current_batch_size = 0
            current_batch.append((source, offsets))
            current_batch_size += len(offsets)
        if current_batch:
            yield self._load_group_batch(current_batch)

    def iter_batch_specs(
        self,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterator[QueryBatchSpec]:
        resolved_batch_size = max(1, int(batch_size))
        rng = random.Random(seed)
        group_specs: List[tuple[IndexedQueryExampleStore, List[int]]] = []
        for source_index, source in enumerate(self._sources):
            source_seed = seed + ((source_index + 1) * 9973)
            for _, offsets in source.conversation_offset_items(shuffle=shuffle, seed=source_seed):
                group_specs.append((source, offsets))
        if shuffle:
            rng.shuffle(group_specs)
        current_batch: List[tuple[IndexedQueryExampleStore, List[int]]] = []
        current_batch_size = 0
        for source, offsets in group_specs:
            if len(offsets) > resolved_batch_size:
                if current_batch:
                    yield self._load_group_batch_spec(current_batch)
                    current_batch = []
                    current_batch_size = 0
                for index in range(0, len(offsets), resolved_batch_size):
                    yield source._batch_spec_from_offsets(offsets[index : index + resolved_batch_size])
                continue
            if current_batch and current_batch_size + len(offsets) > resolved_batch_size:
                yield self._load_group_batch_spec(current_batch)
                current_batch = []
                current_batch_size = 0
            current_batch.append((source, offsets))
            current_batch_size += len(offsets)
        if current_batch:
            yield self._load_group_batch_spec(current_batch)

    def conversation_ids(self) -> List[str]:
        conversation_ids: List[str] = []
        seen: set[str] = set()
        for source in self._sources:
            source_conversation_ids = getattr(source, "conversation_ids", None)
            if callable(source_conversation_ids):
                values = source_conversation_ids()
            else:
                values = [clean_text(getattr(row, "conversation_id", "")) for row in source]
            for conversation_id in values:
                normalized_id = clean_text(conversation_id)
                if not normalized_id or normalized_id in seen:
                    continue
                seen.add(normalized_id)
                conversation_ids.append(normalized_id)
        return conversation_ids

    def _load_group_batch(self, group_specs: Sequence[tuple[IndexedQueryExampleStore, List[int]]]) -> List[QueryTrainingExample]:
        rows: List[QueryTrainingExample] = []
        for source, offsets in group_specs:
            rows.extend(source._load_batch_from_offsets(offsets))
        return rows

    def _load_group_batch_spec(self, group_specs: Sequence[tuple[IndexedQueryExampleStore, List[int]]]) -> QueryBatchSpec:
        return QueryBatchSpec(
            sources=tuple(
                QueryBatchSourceOffsets(
                    path=str(source.path),
                    offsets=tuple(int(offset) for offset in offsets),
                    skip_bad_rows=bool(source._skip_bad_rows),
                )
                for source, offsets in group_specs
                if offsets
            )
        )


class SourceAwareTrainingStore:
    def __init__(
        self,
        base_rows: Sequence[QueryTrainingExample],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self._base_rows = base_rows
        self._config = {**DEFAULT_TRAINING_CONFIG, **dict(config or {})}
        self._groups: List[_TrainingConversationGroup] = []
        self._source_row_counts: Counter[str] = Counter()
        self._answer_type_counts: Counter[str] = Counter()
        self._supervision_bucket_counts: Counter[str] = Counter()
        self._source_answer_temporal_counts: Counter[tuple[str, str, bool]] = Counter()
        self._source_supervision_temporal_counts: Counter[tuple[str, str, bool]] = Counter()
        self._sampling_summary: Dict[str, Any] = {}
        self._effective_row_count = 0
        self._build()

    def __len__(self) -> int:
        return int(self._effective_row_count)

    def base_rows(self) -> Sequence[QueryTrainingExample]:
        return self._base_rows

    def sampling_summary(self) -> Dict[str, Any]:
        return dict(self._sampling_summary)

    def _iter_sources(self) -> List[Any]:
        if isinstance(self._base_rows, CombinedQueryExampleStore):
            return list(self._base_rows._sources)
        return [self._base_rows]

    def _collect_conversation_summaries(
        self,
        source: Any,
        *,
        fallback_source: str,
        in_memory_groups: OrderedDict[str, List[QueryTrainingExample]] | None = None,
    ) -> "OrderedDict[str, _TrainingConversationSummary]":
        conversation_summaries: "OrderedDict[str, _TrainingConversationSummary]" = OrderedDict()
        for example in source:
            if not isinstance(example, QueryTrainingExample):
                continue
            conversation_id = clean_text(example.conversation_id)
            if not conversation_id:
                continue
            source_dataset = _training_source_dataset_name(example=example, fallback_source=fallback_source)
            answer_type = _training_answer_type_name(example)
            supervision_bucket = _training_supervision_bucket_name(example)
            has_temporal_positive = _has_positive_temporal_supervision(example)
            summary = conversation_summaries.get(conversation_id)
            if summary is None:
                summary = _TrainingConversationSummary(
                    conversation_id=conversation_id,
                    source_dataset=source_dataset,
                )
                conversation_summaries[conversation_id] = summary
            summary.row_count += 1
            summary.answer_type_counts[answer_type] = int(summary.answer_type_counts.get(answer_type, 0) or 0) + 1
            if answer_type == "time":
                summary.time_example_count += 1
            if _has_multi_evidence_supervision(example):
                summary.multi_evidence_count += 1
            if has_temporal_positive:
                summary.temporal_positive_count += 1
            self._source_row_counts[source_dataset] += 1
            self._answer_type_counts[answer_type] += 1
            self._supervision_bucket_counts[supervision_bucket] += 1
            self._source_answer_temporal_counts[(source_dataset, answer_type, bool(has_temporal_positive))] += 1
            self._source_supervision_temporal_counts[(source_dataset, supervision_bucket, bool(has_temporal_positive))] += 1
            if in_memory_groups is not None:
                in_memory_groups.setdefault(conversation_id, []).append(example)
        return conversation_summaries

    def _source_fallback_name(self, source: Any) -> str:
        path_value = getattr(source, "path", None)
        if path_value is None:
            return "unknown"
        try:
            resolved_path = Path(path_value)
        except Exception:
            return "unknown"
        return clean_text(resolved_path.parent.parent.name) or "unknown"

    def _build(self) -> None:
        source_entries: List[tuple[Any, OrderedDict[str, _TrainingConversationSummary], OrderedDict[str, List[QueryTrainingExample]] | None]] = []
        for source in self._iter_sources():
            fallback_source = self._source_fallback_name(source)
            has_offset_loader = callable(getattr(source, "conversation_offset_items", None)) and callable(
                getattr(source, "_load_batch_from_offsets", None)
            )
            in_memory_groups: OrderedDict[str, List[QueryTrainingExample]] | None = None
            if not has_offset_loader:
                in_memory_groups = OrderedDict()
            conversation_summaries = self._collect_conversation_summaries(
                source,
                fallback_source=fallback_source,
                in_memory_groups=in_memory_groups,
            )
            source_entries.append((source, conversation_summaries, in_memory_groups))

        total_rows = sum(int(count or 0) for count in self._source_row_counts.values())
        unique_source_count = max(1, len(self._source_row_counts))
        sampling_source_factors = {
            source_dataset: max(
                1.0,
                _smoothed_inverse_share_factor(
                    count=row_count,
                    total=total_rows,
                    unique_count=unique_source_count,
                    alpha=float(self._config.get("sampling_source_alpha", 0.35) or 0.35),
                    blend_uniform_ratio=float(self._config.get("sampling_blend_uniform_ratio", 0.35) or 0.35),
                ),
            )
            for source_dataset, row_count in self._source_row_counts.items()
        }
        loss_source_factors = {
            source_dataset: _smoothed_inverse_share_factor(
                count=row_count,
                total=total_rows,
                unique_count=unique_source_count,
                alpha=float(self._config.get("loss_source_alpha", 0.4) or 0.4),
                blend_uniform_ratio=float(self._config.get("loss_blend_uniform_ratio", 0.25) or 0.25),
            )
            for source_dataset, row_count in self._source_row_counts.items()
        }

        for source, conversation_summaries, in_memory_groups in source_entries:
            has_offset_loader = callable(getattr(source, "conversation_offset_items", None)) and callable(
                getattr(source, "_load_batch_from_offsets", None)
            )
            if has_offset_loader:
                for conversation_id, offsets in source.conversation_offset_items(shuffle=False, seed=0):
                    summary = conversation_summaries.get(clean_text(conversation_id))
                    if summary is None:
                        continue
                    sampling_multiplier = self._conversation_sampling_multiplier(
                        summary,
                        sampling_source_factors=sampling_source_factors,
                    )
                    repeat_count = _conversation_repeat_count(
                        summary.conversation_id,
                        sampling_multiplier=sampling_multiplier,
                        max_group_repeat=int(self._config.get("sampling_max_group_repeat", 2) or 2),
                    )
                    self._groups.append(
                        _TrainingConversationGroup(
                            conversation_id=summary.conversation_id,
                            source_dataset=summary.source_dataset,
                            row_count=int(summary.row_count),
                            time_example_count=int(summary.time_example_count),
                            multi_evidence_count=int(summary.multi_evidence_count),
                            temporal_positive_count=int(summary.temporal_positive_count),
                            answer_type_counts=dict(summary.answer_type_counts),
                            repeat_count=repeat_count,
                            sampling_multiplier=float(sampling_multiplier),
                            source_store=source,
                            offsets=tuple(int(offset) for offset in list(offsets or []) if int(offset) >= 0),
                        )
                    )
            else:
                for conversation_id, rows in (in_memory_groups or {}).items():
                    summary = conversation_summaries.get(clean_text(conversation_id))
                    if summary is None:
                        continue
                    sampling_multiplier = self._conversation_sampling_multiplier(
                        summary,
                        sampling_source_factors=sampling_source_factors,
                    )
                    repeat_count = _conversation_repeat_count(
                        summary.conversation_id,
                        sampling_multiplier=sampling_multiplier,
                        max_group_repeat=int(self._config.get("sampling_max_group_repeat", 2) or 2),
                    )
                    self._groups.append(
                        _TrainingConversationGroup(
                            conversation_id=summary.conversation_id,
                            source_dataset=summary.source_dataset,
                            row_count=int(summary.row_count),
                            time_example_count=int(summary.time_example_count),
                            multi_evidence_count=int(summary.multi_evidence_count),
                            temporal_positive_count=int(summary.temporal_positive_count),
                            answer_type_counts=dict(summary.answer_type_counts),
                            repeat_count=repeat_count,
                            sampling_multiplier=float(sampling_multiplier),
                            rows=tuple(rows),
                        )
                    )
        self._groups.sort(key=lambda item: item.conversation_id)
        self._effective_row_count = sum(int(group.row_count) * int(group.repeat_count) for group in self._groups)
        self._sampling_summary = self._build_sampling_summary(
            total_rows=total_rows,
            sampling_source_factors=sampling_source_factors,
            loss_source_factors=loss_source_factors,
        )
        self._loss_source_factors = loss_source_factors

    def _conversation_sampling_multiplier(
        self,
        summary: _TrainingConversationSummary,
        *,
        sampling_source_factors: Mapping[str, float],
    ) -> float:
        source_factor = max(1.0, float(sampling_source_factors.get(summary.source_dataset, 1.0) or 1.0))
        boost = 1.0
        time_signal = min(1.0, float(summary.time_example_count) / 2.0)
        multi_signal = min(1.0, float(summary.multi_evidence_count))
        temporal_signal = min(1.0, float(summary.temporal_positive_count) / 2.0)
        boost += max(0.0, float(self._config.get("sampling_time_boost", 1.2) or 1.2) - 1.0) * time_signal
        boost += max(0.0, float(self._config.get("sampling_multi_evidence_boost", 1.45) or 1.45) - 1.0) * multi_signal
        boost += max(0.0, float(self._config.get("sampling_temporal_positive_boost", 1.15) or 1.15) - 1.0) * temporal_signal
        raw_multiplier = source_factor * boost
        return _bounded_training_weight(
            raw_multiplier,
            minimum=1.0,
            maximum=float(self._config.get("sampling_max_conversation_multiplier", 2.0) or 2.0),
        )

    def _example_training_weight(self, example: QueryTrainingExample, *, fallback_source: str) -> float:
        source_dataset = _training_source_dataset_name(example=example, fallback_source=fallback_source)
        answer_type = _training_answer_type_name(example)
        raw_weight = float(self._loss_source_factors.get(source_dataset, 1.0) or 1.0)
        if answer_type == "time":
            raw_weight *= float(self._config.get("loss_time_boost", 1.45) or 1.45)
        elif answer_type == "multi_evidence" or _has_multi_evidence_supervision(example):
            raw_weight *= float(self._config.get("loss_multi_evidence_boost", 1.75) or 1.75)
        if _has_positive_temporal_supervision(example):
            raw_weight *= float(self._config.get("loss_temporal_positive_boost", 1.2) or 1.2)
        resolved_power = max(0.0, float(self._config.get("loss_weight_power", 0.5) or 0.5))
        if resolved_power <= 0.0:
            softened_weight = 1.0
        elif abs(resolved_power - 1.0) <= 1e-9:
            softened_weight = raw_weight
        else:
            softened_weight = math.pow(max(raw_weight, 1e-12), resolved_power)
        return _bounded_training_weight(
            softened_weight,
            minimum=float(self._config.get("loss_min_example_weight", 0.65) or 0.65),
            maximum=float(self._config.get("loss_max_example_weight", 1.85) or 1.85),
        )

    def _annotate_rows_for_training(
        self,
        rows: Sequence[QueryTrainingExample],
        *,
        group: _TrainingConversationGroup,
        clone_rows: bool,
        shuffle: bool,
        rng: random.Random,
    ) -> List[QueryTrainingExample]:
        prepared_rows: List[QueryTrainingExample] = []
        for row in rows:
            example_weight = self._example_training_weight(row, fallback_source=group.source_dataset)
            metadata_updates = {
                "_training_enabled": True,
                "_training_source_dataset": group.source_dataset,
                "_training_sampling_multiplier": float(group.sampling_multiplier),
                "_training_repeat_count": int(group.repeat_count),
                "_training_example_weight": float(example_weight),
            }
            if clone_rows:
                prepared_rows.append(_clone_query_training_example(row, metadata_updates=metadata_updates))
            else:
                row.metadata = {**dict(row.metadata or {}), **metadata_updates}
                prepared_rows.append(row)
        if shuffle and len(prepared_rows) > 1:
            rng.shuffle(prepared_rows)
        return prepared_rows

    def _materialize_group_rows(
        self,
        group: _TrainingConversationGroup,
        *,
        shuffle: bool,
        rng: random.Random,
    ) -> List[QueryTrainingExample]:
        if group.source_store is not None and group.offsets:
            rows = group.source_store._load_batch_from_offsets(group.offsets)
            return self._annotate_rows_for_training(rows, group=group, clone_rows=False, shuffle=shuffle, rng=rng)
        rows = list(group.rows or ())
        return self._annotate_rows_for_training(rows, group=group, clone_rows=True, shuffle=shuffle, rng=rng)

    def _load_group_batch(
        self,
        group_specs: Sequence[_TrainingConversationGroup],
        *,
        shuffle: bool,
        rng: random.Random,
    ) -> List[QueryTrainingExample]:
        rows: List[QueryTrainingExample] = []
        for group in group_specs:
            rows.extend(self._materialize_group_rows(group, shuffle=shuffle, rng=rng))
        return rows

    def iter_batches(
        self,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterator[List[QueryTrainingExample]]:
        resolved_batch_size = max(1, int(batch_size))
        rng = random.Random(seed)
        expanded_groups: List[_TrainingConversationGroup] = []
        for group in self._groups:
            repeat_count = max(1, int(group.repeat_count))
            for _ in range(repeat_count):
                expanded_groups.append(group)
        if shuffle:
            rng.shuffle(expanded_groups)
        current_batch: List[_TrainingConversationGroup] = []
        current_batch_size = 0
        for group in expanded_groups:
            if int(group.row_count) > resolved_batch_size:
                if current_batch:
                    yield self._load_group_batch(current_batch, shuffle=shuffle, rng=rng)
                    current_batch = []
                    current_batch_size = 0
                oversized_rows = self._materialize_group_rows(group, shuffle=shuffle, rng=rng)
                for index in range(0, len(oversized_rows), resolved_batch_size):
                    yield oversized_rows[index : index + resolved_batch_size]
                continue
            if current_batch and current_batch_size + int(group.row_count) > resolved_batch_size:
                yield self._load_group_batch(current_batch, shuffle=shuffle, rng=rng)
                current_batch = []
                current_batch_size = 0
            current_batch.append(group)
            current_batch_size += int(group.row_count)
        if current_batch:
            yield self._load_group_batch(current_batch, shuffle=shuffle, rng=rng)

    def _build_sampling_summary(
        self,
        *,
        total_rows: int,
        sampling_source_factors: Mapping[str, float],
        loss_source_factors: Mapping[str, float],
    ) -> Dict[str, Any]:
        sampled_source_rows: Counter[str] = Counter()
        repeat_histogram: Counter[str] = Counter()
        weighted_answer_mass: Counter[str] = Counter()
        weighted_supervision_bucket_mass: Counter[str] = Counter()
        weighted_source_mass: Counter[str] = Counter()
        weight_floor = float(self._config.get("loss_min_example_weight", 0.65) or 0.65)
        weight_cap = float(self._config.get("loss_max_example_weight", 1.85) or 1.85)
        observed_min_weight = weight_cap
        observed_max_weight = weight_floor

        for group in self._groups:
            sampled_source_rows[group.source_dataset] += int(group.row_count) * int(group.repeat_count)
            repeat_histogram[str(int(group.repeat_count))] += 1

        for (source_dataset, answer_type, has_temporal_positive), count in self._source_answer_temporal_counts.items():
            if count <= 0:
                continue
            raw_weight = float(loss_source_factors.get(source_dataset, 1.0) or 1.0)
            if answer_type == "time":
                raw_weight *= float(self._config.get("loss_time_boost", 1.45) or 1.45)
            elif answer_type == "multi_evidence":
                raw_weight *= float(self._config.get("loss_multi_evidence_boost", 1.75) or 1.75)
            if has_temporal_positive:
                raw_weight *= float(self._config.get("loss_temporal_positive_boost", 1.2) or 1.2)
            resolved_power = max(0.0, float(self._config.get("loss_weight_power", 0.5) or 0.5))
            if resolved_power <= 0.0:
                example_weight = 1.0
            elif abs(resolved_power - 1.0) <= 1e-9:
                example_weight = raw_weight
            else:
                example_weight = math.pow(max(raw_weight, 1e-12), resolved_power)
            example_weight = _bounded_training_weight(example_weight, minimum=weight_floor, maximum=weight_cap)
            observed_min_weight = min(observed_min_weight, float(example_weight))
            observed_max_weight = max(observed_max_weight, float(example_weight))
            weighted_answer_mass[answer_type] += float(count) * float(example_weight)
            weighted_source_mass[source_dataset] += float(count) * float(example_weight)
        for (source_dataset, supervision_bucket, has_temporal_positive), count in self._source_supervision_temporal_counts.items():
            if count <= 0:
                continue
            raw_weight = float(loss_source_factors.get(source_dataset, 1.0) or 1.0)
            if supervision_bucket == "time":
                raw_weight *= float(self._config.get("loss_time_boost", 1.45) or 1.45)
            elif supervision_bucket in {"multi_evidence", "list_event_text_multi_positive"} or supervision_bucket.endswith("_multi_positive"):
                raw_weight *= float(self._config.get("loss_multi_evidence_boost", 1.75) or 1.75)
            if has_temporal_positive:
                raw_weight *= float(self._config.get("loss_temporal_positive_boost", 1.2) or 1.2)
            resolved_power = max(0.0, float(self._config.get("loss_weight_power", 0.5) or 0.5))
            if resolved_power <= 0.0:
                example_weight = 1.0
            elif abs(resolved_power - 1.0) <= 1e-9:
                example_weight = raw_weight
            else:
                example_weight = math.pow(max(raw_weight, 1e-12), resolved_power)
            example_weight = _bounded_training_weight(example_weight, minimum=weight_floor, maximum=weight_cap)
            weighted_supervision_bucket_mass[supervision_bucket] += float(count) * float(example_weight)

        source_details: Dict[str, Any] = {}
        for source_dataset, row_count in self._source_row_counts.items():
            sampled_rows = int(sampled_source_rows.get(source_dataset, row_count) or 0)
            weighted_loss_mass = float(weighted_source_mass.get(source_dataset, 0.0) or 0.0)
            source_details[source_dataset] = {
                "row_count": int(row_count),
                "row_share": round(float(row_count) / float(max(1, total_rows)), 6),
                "sampling_source_factor": round(float(sampling_source_factors.get(source_dataset, 1.0) or 1.0), 6),
                "loss_source_factor": round(float(loss_source_factors.get(source_dataset, 1.0) or 1.0), 6),
                "sampled_row_count_estimate": int(sampled_rows),
                "sampled_row_share_estimate": round(float(sampled_rows) / float(max(1, self._effective_row_count)), 6),
                "weighted_loss_mass": round(weighted_loss_mass, 6),
                "avg_example_weight_estimate": round(weighted_loss_mass / float(max(1, row_count)), 6),
            }

        return {
            "mode": clean_text(self._config.get("train_sampling_mode", "")) or "uniform",
            "base_row_count": int(total_rows),
            "effective_row_count": int(self._effective_row_count),
            "source_count": int(len(self._source_row_counts)),
            "source_row_counts": {key: int(value) for key, value in sorted(self._source_row_counts.items())},
            "answer_type_counts": {key: int(value) for key, value in sorted(self._answer_type_counts.items())},
            "supervision_bucket_counts": {key: int(value) for key, value in sorted(self._supervision_bucket_counts.items())},
            "repeat_histogram": {key: int(value) for key, value in sorted(repeat_histogram.items(), key=lambda item: int(item[0]))},
            "weighted_answer_mass_estimate": {key: round(float(value), 6) for key, value in sorted(weighted_answer_mass.items())},
            "weighted_supervision_bucket_mass_estimate": {
                key: round(float(value), 6) for key, value in sorted(weighted_supervision_bucket_mass.items())
            },
            "loss_weight_range_estimate": {
                "min": round(float(observed_min_weight if self._source_answer_temporal_counts else 1.0), 6),
                "max": round(float(observed_max_weight if self._source_answer_temporal_counts else 1.0), 6),
            },
            "source_details": source_details,
        }


def maybe_build_source_aware_training_store(
    rows: Sequence[QueryTrainingExample],
    *,
    config: Mapping[str, Any] | None = None,
) -> Sequence[QueryTrainingExample]:
    resolved_config = {**DEFAULT_TRAINING_CONFIG, **dict(config or {})}
    mode = clean_text(resolved_config.get("train_sampling_mode", ""))
    if not mode or mode == "uniform":
        return rows
    return SourceAwareTrainingStore(rows, config=resolved_config)


def load_query_examples(
    path: Path,
    *,
    skip_bad_rows: bool = False,
    error_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> IndexedQueryExampleStore:
    return IndexedQueryExampleStore(path, skip_bad_rows=skip_bad_rows, error_callback=error_callback)


def _materialize_query_batch_spec(batch_spec: QueryBatchSpec) -> List[QueryTrainingExample]:
    rows: List[QueryTrainingExample] = []
    for source in list(batch_spec.sources or ()):
        resolved_path = Path(clean_text(source.path))
        resolved_offsets = [int(offset) for offset in list(source.offsets or ()) if int(offset) >= 0]
        if not resolved_offsets or not str(resolved_path):
            continue
        with resolved_path.open("rb") as handle:
            for offset in resolved_offsets:
                handle.seek(offset)
                raw_line = handle.readline()
                try:
                    rows.append(_load_query_example_from_raw_line(resolved_path, int(offset), raw_line))
                except Exception:
                    if not bool(source.skip_bad_rows):
                        raise
    return rows


def _materialize_query_batch_payload(
    batch_payload: QueryBatchSpec | Sequence[QueryTrainingExample],
) -> List[QueryTrainingExample]:
    if isinstance(batch_payload, QueryBatchSpec):
        return _materialize_query_batch_spec(batch_payload)
    return list(batch_payload)


def _build_grad_scaler(*, enabled: bool, device: torch.device) -> Any:
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        return amp_module.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(*, enabled: bool, device: torch.device) -> Any:
    if not enabled:
        return nullcontext()
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "autocast"):
        return amp_module.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _event_labels(example: QueryTrainingExample, candidate_event_ids: Sequence[str], *, device: torch.device) -> Tensor:
    positive_ids = {clean_text(item) for item in example.positive_event_ids if clean_text(item)}
    return torch.tensor(
        [1.0 if event_id in positive_ids else 0.0 for event_id in candidate_event_ids],
        dtype=torch.float32,
        device=device,
    )


def _balanced_binary_loss(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    working_logits = logits.float()
    working_targets = targets.float()
    positive_count = int((working_targets > 0.5).sum().item())
    negative_count = int(working_targets.numel() - positive_count)
    if positive_count > 0 and negative_count > 0:
        pos_weight = torch.tensor(
            float(negative_count) / float(max(1, positive_count)),
            dtype=working_logits.dtype,
            device=working_logits.device,
        )
        return F.binary_cross_entropy_with_logits(working_logits, working_targets, pos_weight=pos_weight)
    return F.binary_cross_entropy_with_logits(working_logits, working_targets)


def _pairwise_margin_loss(logits: Tensor, targets: Tensor, *, margin: float = 0.2) -> Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    working_logits = logits.float()
    working_targets = targets.float()
    positive_logits = working_logits[working_targets > 0.5]
    negative_logits = working_logits[working_targets <= 0.5]
    if positive_logits.numel() == 0 or negative_logits.numel() == 0:
        return working_logits.sum() * 0.0
    margins = float(margin) - (positive_logits.unsqueeze(-1) - negative_logits.unsqueeze(0))
    return F.relu(margins).mean()


def _positive_mass_loss(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    working_logits = logits.float()
    positive_mask = targets.float() > 0.5
    if not bool(torch.any(positive_mask)):
        return working_logits.sum() * 0.0
    positive_logits = working_logits.masked_fill(~positive_mask, float("-inf"))
    return torch.logsumexp(working_logits, dim=-1) - torch.logsumexp(positive_logits, dim=-1)


def _recall_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return 0.5 * _balanced_binary_loss(logits, targets) + 0.5 * _positive_mass_loss(logits, targets)


def _ranking_binary_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    margin: float = 0.2,
    bce_weight: float = 0.4,
    positive_mass_weight: float = 0.3,
    pairwise_weight: float = 0.3,
) -> Tensor:
    return (
        float(bce_weight) * _balanced_binary_loss(logits, targets)
        + float(positive_mass_weight) * _positive_mass_loss(logits, targets)
        + float(pairwise_weight) * _pairwise_margin_loss(logits, targets, margin=margin)
    )


def _path_labels(example: QueryTrainingExample, candidate_path_ids: Sequence[str], *, device: torch.device) -> Tensor:
    positive_ids = {clean_text(item) for item in example.positive_path_ids if clean_text(item)}
    return torch.tensor(
        [1.0 if path_id in positive_ids else 0.0 for path_id in candidate_path_ids],
        dtype=torch.float32,
        device=device,
    )


def _apply_trainable_stage(model: nn.Module, stage: str | None) -> Dict[str, Any]:
    normalized_stage = clean_text(stage or "all").lower().replace("-", "_")
    if normalized_stage in {"", "all", "full"}:
        for parameter in model.parameters():
            parameter.requires_grad = True
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        return {
            "trainable_stage": "all",
            "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
            "frozen_parameter_count": 0,
            "trainable_name_prefixes": sorted({name.split(".", 1)[0] for name in trainable_names}),
        }
    if normalized_stage == "recall_only":
        allowed_prefixes = ("event_recall_head.",)
    elif normalized_stage == "recall_and_question_projection":
        allowed_prefixes = ("event_recall_head.", "question_projection.")
    elif normalized_stage == "selection_and_fusion":
        allowed_prefixes = (
            "answer_type_head.",
            "event_calibration_head.",
            "event_distractor_head.",
            "event_head.",
            "event_matrix_head.",
            "event_pair_adapter.",
            "event_subgraph_refiner.",
            "event_tunnel_head.",
            "final_event_fusion_head.",
            "final_path_fusion_head.",
            "memory_router_head.",
            "path_calibration_head.",
            "path_head.",
            "path_matrix_head.",
            "path_pair_adapter.",
            "path_tunnel_head.",
            "question_intent_head.",
            "temporal_head.",
        )
    elif normalized_stage == "profile_recall_selection_plan":
        allowed_prefixes = (
            "answer_plan_head.",
            "event_head.",
            "event_matrix_head.",
            "event_recall_head.",
            "event_tunnel_head.",
            "final_event_fusion_head.",
            "final_path_fusion_head.",
            "memory_router_head.",
            "path_head.",
            "path_matrix_head.",
            "path_tunnel_head.",
            "question_intent_head.",
            "temporal_head.",
        )
    elif normalized_stage == "tunnel_fusion_only":
        allowed_prefixes = (
            "event_tunnel_head.",
            "final_event_fusion_head.",
            "final_path_fusion_head.",
            "path_tunnel_head.",
        )
    elif normalized_stage == "path_tunnel_only":
        allowed_prefixes = (
            "final_path_fusion_head.",
            "path_calibration_head.",
            "path_head.",
            "path_matrix_head.",
            "path_pair_adapter.",
            "path_tunnel_head.",
        )
    elif normalized_stage == "answer_plan_only":
        allowed_prefixes = ("answer_plan_head.",)
    elif normalized_stage == "memory_router_only":
        allowed_prefixes = ("memory_router_head.",)
    else:
        raise ValueError(
            "Unsupported trainable_stage "
            f"{stage!r}; expected one of: all, recall_only, recall_and_question_projection, selection_and_fusion, profile_recall_selection_plan, tunnel_fusion_only, path_tunnel_only, answer_plan_only, memory_router_only"
        )
    trainable_parameter_count = 0
    frozen_parameter_count = 0
    trainable_prefixes: set[str] = set()
    for name, parameter in model.named_parameters():
        should_train = any(name.startswith(prefix) for prefix in allowed_prefixes)
        parameter.requires_grad = bool(should_train)
        if should_train:
            trainable_parameter_count += int(parameter.numel())
            trainable_prefixes.add(name.split(".", 1)[0])
        else:
            frozen_parameter_count += int(parameter.numel())
    if trainable_parameter_count <= 0:
        raise RuntimeError(f"trainable_stage={stage!r} did not leave any trainable parameters")
    return {
        "trainable_stage": normalized_stage,
        "trainable_parameter_count": int(trainable_parameter_count),
        "frozen_parameter_count": int(frozen_parameter_count),
        "trainable_name_prefixes": sorted(trainable_prefixes),
    }


def _temporal_path_labels(
    example: QueryTrainingExample,
    candidate_temporal_path_ids: Sequence[str],
    candidate_temporal_node_ids: Sequence[str],
    *,
    device: torch.device,
) -> Tensor:
    positive_event_ids = {clean_text(item) for item in example.positive_event_ids if clean_text(item)}
    positive_time_node_ids = {clean_text(item) for item in example.positive_time_node_ids if clean_text(item)}
    temporal_path_targets: List[float] = []
    for path_id, time_node_id in zip(candidate_temporal_path_ids, candidate_temporal_node_ids):
        event_id, _, _ = parse_path_id(path_id)
        is_positive = bool(event_id in positive_event_ids and time_node_id in positive_time_node_ids)
        temporal_path_targets.append(1.0 if is_positive else 0.0)
    return torch.tensor(temporal_path_targets, dtype=torch.float32, device=device)


def _is_temporal_example(example: QueryTrainingExample) -> bool:
    answer_type = clean_text(dict(example.answer_targets or {}).get("answer_type", ""))
    temporal_target = dict(example.temporal_target or {})
    if "use_temporal_head" in temporal_target:
        return bool(temporal_target.get("use_temporal_head", False))
    question_is_temporal = bool(
        temporal_target.get("question_is_temporal", temporal_target.get("is_temporal", False))
    ) or bool(dict(example.question_features or {}).get("is_temporal", False))
    has_positive_supervision = bool(
        temporal_target.get("has_positive_time_supervision", False)
    ) or bool(list(example.positive_time_node_ids or []))
    return question_is_temporal and (has_positive_supervision or answer_type == "abstain")


def _answer_type_label(example: QueryTrainingExample) -> int:
    answer_type = clean_text(dict(example.answer_targets or {}).get("answer_type", "")) or "abstain"
    return ANSWER_TYPE_TO_ID.get(answer_type, ANSWER_TYPE_TO_ID["abstain"])


def _coerce_memory_router_targets(value: Any) -> Dict[str, float]:
    targets = {layer: 0.0 for layer in MEMORY_ROUTER_LAYERS}
    if isinstance(value, Mapping):
        for key, raw in value.items():
            layer = clean_text(key).lower().replace("-", "_").replace(" ", "_")
            if layer in targets:
                targets[layer] = 1.0 if bool(raw) else 0.0
        return targets
    if isinstance(value, str):
        raw_items = re.split(r"[\s,;|/]+", value)
    else:
        raw_items = list(value or []) if isinstance(value, (list, tuple, set)) else []
    for item in raw_items:
        layer = clean_text(item).lower().replace("-", "_").replace(" ", "_")
        if layer in targets:
            targets[layer] = 1.0
    return targets


def _memory_router_targets_for_example(
    example: QueryTrainingExample,
    supervision: Mapping[str, Any] | None = None,
) -> Dict[str, float]:
    answer_targets = dict(example.answer_targets or {})
    metadata = dict(example.metadata or {})
    question_features = dict(example.question_features or {})
    explicit = (
        answer_targets.get("memory_router_targets")
        or metadata.get("memory_router_targets")
        or question_features.get("memory_router_targets")
    )
    if explicit:
        targets = _coerce_memory_router_targets(explicit)
        if any(value > 0.0 for value in targets.values()):
            return targets

    supervision_payload = dict(supervision or {})
    positive_event_ids = set(supervision_payload.get("positive_event_ids", example.positive_event_ids) or [])
    positive_path_ids = set(supervision_payload.get("positive_path_ids", example.positive_path_ids) or [])
    answer_type = _training_answer_type_name(example)
    semantic_slot = clean_text(question_features.get("semantic_slot_target", "")).lower()
    training_focus = clean_text(metadata.get("training_focus", "")).lower()
    source_dataset = clean_text(metadata.get("source_dataset", "")).lower()
    domain = clean_text(metadata.get("domain", "")).lower()
    targets = {layer: 0.0 for layer in MEMORY_ROUTER_LAYERS}
    if positive_event_ids or answer_type in {"event_text", "multi_evidence", "profile", "time"}:
        targets["event"] = 1.0
    if answer_type == "profile" or semantic_slot == "profile" or "profile" in training_focus:
        targets["profile"] = 1.0
    if (
        bool(question_features.get("implicit_profile_resource_query", False))
        or "resource" in training_focus
        or any(marker in domain for marker in ("resource", "tool", "gear", "device", "accessor", "ingredient"))
    ):
        targets["resource"] = 1.0
    if answer_type == "time" or _is_temporal_example(example):
        targets["temporal"] = 1.0
    if positive_path_ids and (len(positive_event_ids) > 1 or answer_type in {"profile", "multi_evidence"}):
        targets["path_tunnel"] = 1.0
    if len(positive_event_ids) > 1 or any(marker in source_dataset for marker in ("longmemeval", "multi_session", "locomo")):
        targets["topic_tunnel"] = 1.0
    if not any(value > 0.0 for value in targets.values()):
        targets["event"] = 1.0
    return targets


def _memory_router_target_tensor(
    example: QueryTrainingExample,
    supervision: Mapping[str, Any] | None,
    *,
    device: torch.device,
) -> Tensor:
    targets = _memory_router_targets_for_example(example, supervision)
    return torch.tensor(
        [float(targets.get(layer, 0.0) or 0.0) for layer in MEMORY_ROUTER_LAYERS],
        dtype=torch.float32,
        device=device,
    )


def _batched_logit_statistics(logits: Tensor, *, mask: Tensor | None = None) -> Tensor:
    if logits.dim() != 2:
        raise ValueError(f"Expected rank-2 logits, got {logits.dim()}")
    batch_size = int(logits.size(0))
    if logits.numel() == 0:
        return torch.zeros((batch_size, 2), dtype=logits.dtype, device=logits.device)
    scores = torch.sigmoid(logits.float())
    if mask is not None:
        scores = scores.masked_fill(~mask, 0.0)
        available = mask.any(dim=1).to(dtype=scores.dtype)
    else:
        available = torch.ones((batch_size,), dtype=scores.dtype, device=scores.device)
    top_k = min(2, int(scores.size(1)))
    if top_k <= 0:
        return torch.zeros((batch_size, 2), dtype=logits.dtype, device=logits.device)
    top_values = torch.topk(scores, k=top_k, dim=1).values
    top1 = top_values[:, 0] * available
    top2 = top_values[:, 1] * available if top_values.size(1) > 1 else torch.zeros_like(top1)
    return torch.stack([top1.to(dtype=logits.dtype), (top1 - top2).to(dtype=logits.dtype)], dim=-1)


def _batched_probability_competition_features(logits: Tensor, *, mask: Tensor | None = None) -> Tensor:
    if logits.dim() != 2:
        raise ValueError(f"Expected rank-2 logits, got {logits.dim()}")
    batch_size = int(logits.size(0))
    if logits.numel() == 0 or logits.size(1) <= 0:
        return torch.zeros((batch_size, 4), dtype=logits.dtype, device=logits.device)
    working_logits = logits.float()
    if mask is not None:
        valid_mask = mask.bool()
    else:
        valid_mask = torch.ones_like(working_logits, dtype=torch.bool)
    available = valid_mask.any(dim=1)
    masked_logits = working_logits.masked_fill(~valid_mask, float("-inf"))
    probs = torch.softmax(masked_logits, dim=1)
    probs = torch.where(valid_mask, probs, torch.zeros_like(probs))
    if probs.size(1) <= 0:
        return torch.zeros((batch_size, 4), dtype=logits.dtype, device=logits.device)
    top_k = min(3, int(probs.size(1)))
    top_values = torch.topk(probs, k=top_k, dim=1).values
    top1 = top_values[:, 0]
    top2 = top_values[:, 1] if top_values.size(1) > 1 else torch.zeros_like(top1)
    top3_mass = top_values.sum(dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)
    valid_counts = valid_mask.sum(dim=1).clamp_min(1).to(dtype=entropy.dtype)
    entropy = entropy / torch.log(valid_counts + 1e-8).clamp_min(1.0)
    features = torch.stack([top1, top1 - top2, entropy, top3_mass], dim=-1)
    return torch.where(available.unsqueeze(-1), features.to(dtype=logits.dtype), torch.zeros_like(features, dtype=logits.dtype))


def _masked_feature_mean(features: Tensor, mask: Tensor | None, feature_index: int) -> Tensor:
    batch_size = int(features.size(0))
    if features.numel() == 0:
        return torch.zeros((batch_size, 1), dtype=features.dtype, device=features.device)
    values = features[:, :, feature_index]
    if mask is None:
        return values.mean(dim=1, keepdim=True)
    valid_mask = mask.bool()
    denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=features.dtype)
    total = values.masked_fill(~valid_mask, 0.0).sum(dim=1, keepdim=True)
    available = valid_mask.any(dim=1, keepdim=True)
    return torch.where(available, total / denom, torch.zeros_like(total))


def _zero_center_masked_logits(values: Tensor, mask: Tensor | None) -> Tensor:
    if values.dim() != 2:
        raise ValueError(f"Expected rank-2 values, got {values.dim()}")
    if values.numel() == 0:
        return values
    if mask is None:
        if values.size(1) <= 1:
            return values
        return values - values.mean(dim=1, keepdim=True)
    valid_mask = mask.bool()
    valid_counts = valid_mask.sum(dim=1, keepdim=True)
    multi_candidate = valid_counts > 1
    denom = valid_counts.clamp_min(1).to(dtype=values.dtype)
    mean = values.masked_fill(~valid_mask, 0.0).sum(dim=1, keepdim=True) / denom
    centered = torch.where(multi_candidate, values - mean, values)
    return centered.masked_fill(~valid_mask, 0.0)


def _masked_feature_max(features: Tensor, mask: Tensor | None, feature_index: int) -> Tensor:
    batch_size = int(features.size(0))
    if features.numel() == 0:
        return torch.zeros((batch_size, 1), dtype=features.dtype, device=features.device)
    values = features[:, :, feature_index]
    if mask is None:
        return values.max(dim=1, keepdim=True).values
    valid_mask = mask.bool()
    masked_values = values.masked_fill(~valid_mask, float("-inf"))
    best = masked_values.max(dim=1, keepdim=True).values
    available = valid_mask.any(dim=1, keepdim=True)
    return torch.where(available, best, torch.zeros_like(best))


def _gather_scores_by_indices(source_scores: Tensor, indices: Sequence[int], *, device: torch.device) -> Tensor:
    count = len(indices)
    if count <= 0:
        return source_scores.new_zeros((0,))
    source_count = int(source_scores.numel())
    if source_count <= 0:
        return source_scores.new_zeros((count,))
    safe_indices = [index if 0 <= int(index) < source_count else 0 for index in indices]
    index_tensor = torch.tensor(safe_indices, dtype=torch.long, device=device)
    gathered = source_scores.index_select(0, index_tensor)
    valid_mask = torch.tensor(
        [0 <= int(index) < source_count for index in indices],
        dtype=torch.bool,
        device=device,
    )
    return gathered.masked_fill(~valid_mask, 0.0)


def _gather_scores_by_id(
    source_ids: Sequence[str],
    source_scores: Tensor,
    target_ids: Sequence[str],
    *,
    device: torch.device,
) -> Tensor:
    lookup = {clean_text(item_id): index for index, item_id in enumerate(source_ids) if clean_text(item_id)}
    return _gather_scores_by_indices(
        source_scores,
        [lookup.get(clean_text(item_id), -1) for item_id in target_ids],
        device=device,
    )


def _scatter_amax_by_indices(
    values: Tensor,
    indices: Sequence[int],
    *,
    size: int,
    device: torch.device,
) -> Tensor:
    result = values.new_zeros((max(0, int(size)),))
    if result.numel() == 0 or values.numel() == 0 or not indices:
        return result
    usable_count = min(int(values.numel()), len(indices))
    if usable_count <= 0:
        return result
    safe_indices = [int(index) if 0 <= int(index) < int(size) else 0 for index in indices[:usable_count]]
    valid_mask = torch.tensor(
        [0 <= int(index) < int(size) for index in indices[:usable_count]],
        dtype=torch.bool,
        device=device,
    )
    if not bool(torch.any(valid_mask)):
        return result
    index_tensor = torch.tensor(safe_indices, dtype=torch.long, device=device)
    value_tensor = values[:usable_count]
    if hasattr(result, "scatter_reduce_"):
        result.scatter_reduce_(0, index_tensor[valid_mask], value_tensor[valid_mask], reduce="amax", include_self=True)
        return result
    for index, value in zip(index_tensor[valid_mask].tolist(), value_tensor[valid_mask]):
        result[index] = torch.maximum(result[index], value)
    return result


def _scatter_sum_by_indices(
    indices: Sequence[int],
    *,
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    result = torch.zeros((max(0, int(size)),), dtype=dtype, device=device)
    if result.numel() == 0 or not indices:
        return result
    safe_indices = [int(index) if 0 <= int(index) < int(size) else 0 for index in indices]
    valid_mask = torch.tensor(
        [0 <= int(index) < int(size) for index in indices],
        dtype=torch.bool,
        device=device,
    )
    if not bool(torch.any(valid_mask)):
        return result
    index_tensor = torch.tensor(safe_indices, dtype=torch.long, device=device)
    ones = torch.ones((len(indices),), dtype=dtype, device=device)
    result.scatter_add_(0, index_tensor[valid_mask], ones[valid_mask])
    return result


def _best_other_scores(scores: Tensor) -> Tensor:
    if scores.numel() <= 1:
        return scores.new_zeros(scores.shape)
    top_values, top_indices = torch.topk(scores, k=min(2, int(scores.numel())), dim=0)
    best_other = top_values[0].expand_as(scores).clone()
    best_other[top_indices[0]] = top_values[1]
    return best_other


def _best_other_indices(scores: Tensor) -> Tensor:
    if scores.numel() <= 1:
        return torch.zeros(scores.shape, dtype=torch.long, device=scores.device)
    top_indices = torch.topk(scores, k=min(2, int(scores.numel())), dim=0).indices
    best_other = top_indices[0].expand(scores.shape).clone()
    best_other[top_indices[0]] = top_indices[1]
    return best_other.to(dtype=torch.long)


def _relation_strength_matrix(relation_features: Tensor, weights: Sequence[float]) -> Tensor:
    if relation_features.numel() == 0:
        return relation_features.new_zeros(relation_features.shape[:-1])
    weight_tensor = relation_features.new_tensor(list(weights), dtype=relation_features.dtype)
    if relation_features.size(-1) != int(weight_tensor.numel()):
        raise ValueError(
            f"Relation feature width mismatch: expected {int(weight_tensor.numel())}, "
            f"found {int(relation_features.size(-1))}"
        )
    normalizer = weight_tensor.sum().clamp_min(1e-8)
    return ((relation_features * weight_tensor).sum(dim=-1) / normalizer).clamp_(0.0, 1.0)


def _gather_1d_by_index(values: Tensor, indices: Tensor, *, valid_mask: Tensor | None = None) -> Tensor:
    if values.numel() <= 0 or indices.numel() <= 0:
        return values.new_zeros(indices.shape, dtype=values.dtype)
    safe_indices = indices.clamp(min=0, max=max(0, int(values.numel()) - 1)).to(dtype=torch.long)
    gathered = values.index_select(0, safe_indices)
    if valid_mask is not None:
        gathered = gathered.masked_fill(~valid_mask.bool(), 0.0)
    return gathered


def _reverse_competition_features(
    scores: Tensor,
    relation_strength: Tensor | None = None,
    *,
    valid_mask: Tensor | None = None,
    relation_threshold: float = TRI_MAZE_RELATION_THRESHOLD,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    score_count = int(scores.numel())
    if score_count <= 0:
        empty_float = scores.new_zeros((0,))
        empty_index = torch.zeros((0,), dtype=torch.long, device=scores.device)
        return empty_float, empty_float, empty_float, empty_float, empty_index
    active_mask = (
        valid_mask.reshape(-1).bool()
        if isinstance(valid_mask, Tensor)
        else torch.ones((score_count,), dtype=torch.bool, device=scores.device)
    )
    available = active_mask.to(dtype=scores.dtype)
    if int(active_mask.sum().item()) <= 1:
        zero_scores = scores.new_zeros(scores.shape)
        zero_indices = torch.zeros(scores.shape, dtype=torch.long, device=scores.device)
        return zero_scores, zero_scores, zero_scores, zero_scores, zero_indices
    working_scores = scores.masked_fill(~active_mask, float("-inf"))
    fallback_scores = _best_other_scores(working_scores)
    fallback_indices = _best_other_indices(working_scores)
    fallback_scores = torch.where(torch.isfinite(fallback_scores), fallback_scores, torch.zeros_like(fallback_scores))
    reverse_scores = fallback_scores
    reverse_indices = fallback_indices
    reverse_relations = scores.new_zeros(scores.shape)
    if isinstance(relation_strength, Tensor) and relation_strength.numel() > 0:
        relation = relation_strength[:score_count, :score_count].to(dtype=scores.dtype, device=scores.device)
        pair_mask = active_mask.unsqueeze(0) & active_mask.unsqueeze(1)
        pair_mask &= ~torch.eye(score_count, dtype=torch.bool, device=scores.device)
        relation = relation.masked_fill(~pair_mask, 0.0)
        related_mask = pair_mask & (relation >= float(relation_threshold))
        related_scores = working_scores.unsqueeze(0).expand(score_count, -1).masked_fill(~related_mask, float("-inf"))
        related_values, related_indices = related_scores.max(dim=1)
        has_related = torch.isfinite(related_values)
        reverse_scores = torch.where(has_related, related_values, fallback_scores)
        reverse_indices = torch.where(has_related, related_indices, fallback_indices)
        reverse_relations = relation.gather(1, reverse_indices.unsqueeze(-1)).squeeze(-1)
    reverse_available = available.clone()
    invalid_mask = ~active_mask
    reverse_scores = reverse_scores.masked_fill(invalid_mask, 0.0)
    reverse_relations = reverse_relations.masked_fill(invalid_mask, 0.0)
    reverse_available = reverse_available.masked_fill(invalid_mask, 0.0)
    boundary_gap = torch.where(
        active_mask,
        scores - reverse_scores,
        torch.zeros_like(scores),
    )
    reverse_indices = reverse_indices.masked_fill(invalid_mask, 0)
    return reverse_scores, boundary_gap, reverse_relations, reverse_available, reverse_indices


def _build_answer_calibration_features(
    *,
    calibrated_event_logits_batch: Tensor,
    calibrated_path_logits_batch: Tensor,
    temporal_logits_batch: Tensor,
    event_pair_features_batch_raw: Tensor,
    path_pair_features_batch_raw: Tensor,
    event_relation_strength_batch: Tensor | None = None,
    path_relation_strength_batch: Tensor | None = None,
    event_mask: Tensor,
    path_mask: Tensor,
    temporal_mask: Tensor,
    examples: Sequence[QueryTrainingExample],
    candidate_path_event_indices_batch: Sequence[Sequence[int]],
    candidate_temporal_event_indices_batch: Sequence[Sequence[int]],
) -> Tensor:
    batch_size = len(examples)
    device = calibrated_event_logits_batch.device
    dtype = calibrated_event_logits_batch.dtype
    event_competition_features = _batched_probability_competition_features(
        calibrated_event_logits_batch,
        mask=event_mask,
    )
    path_competition_features = _batched_probability_competition_features(
        calibrated_path_logits_batch,
        mask=path_mask,
    )
    temporal_competition_features = _batched_probability_competition_features(
        temporal_logits_batch,
        mask=temporal_mask,
    )
    event_logit_stats = _batched_logit_statistics(calibrated_event_logits_batch, mask=event_mask)
    path_logit_stats = _batched_logit_statistics(calibrated_path_logits_batch, mask=path_mask)
    temporal_logit_stats = _batched_logit_statistics(temporal_logits_batch, mask=temporal_mask)
    if event_pair_features_batch_raw.size(1) > 0:
        speaker_conflict_rate = _masked_feature_mean(event_pair_features_batch_raw, event_mask, 14)
        time_conflict_rate = _masked_feature_mean(event_pair_features_batch_raw, event_mask, 6)
        temporal_distractor_rate = _masked_feature_mean(event_pair_features_batch_raw, event_mask, 27)
        best_exact_signature_cover = _masked_feature_max(event_pair_features_batch_raw, event_mask, 26)
        best_source_turn_support = _masked_feature_max(event_pair_features_batch_raw, event_mask, 15)
    else:
        speaker_conflict_rate = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        time_conflict_rate = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        temporal_distractor_rate = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        best_exact_signature_cover = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        best_source_turn_support = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    if path_pair_features_batch_raw.size(1) > 0:
        best_preferred_path = _masked_feature_max(path_pair_features_batch_raw, path_mask, 4)
        best_support_anchor = _masked_feature_max(path_pair_features_batch_raw, path_mask, 17)
    else:
        best_preferred_path = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        best_support_anchor = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    has_event_candidates = event_mask.any(dim=1, keepdim=True).to(dtype=dtype)
    has_path_candidates = path_mask.any(dim=1, keepdim=True).to(dtype=dtype)
    has_temporal_candidates = temporal_mask.any(dim=1, keepdim=True).to(dtype=dtype)
    is_temporal_question = torch.tensor(
        [[1.0 if bool(dict(example.question_features or {}).get("is_temporal", False)) else 0.0] for example in examples],
        dtype=dtype,
        device=device,
    )
    alignment_features = torch.zeros((batch_size, 3), dtype=dtype, device=device)
    structured_reverse_features = torch.zeros((batch_size, 8), dtype=dtype, device=device)
    calibrated_path_prob_batch = torch.sigmoid(calibrated_path_logits_batch.float()).masked_fill(~path_mask, 0.0)
    temporal_prob_batch = torch.sigmoid(temporal_logits_batch.float()).masked_fill(~temporal_mask, 0.0)
    for batch_index in range(batch_size):
        event_count = int(event_mask[batch_index].sum().item())
        if event_count <= 0:
            continue
        event_probs = torch.sigmoid(calibrated_event_logits_batch[batch_index, :event_count].float())
        best_event_index = int(torch.argmax(calibrated_event_logits_batch[batch_index, :event_count].float()).item())
        event_reverse_scores, event_boundary_gaps, event_reverse_relations, event_reverse_available, _ = _reverse_competition_features(
            event_probs,
            event_relation_strength_batch[batch_index, :event_count, :event_count]
            if isinstance(event_relation_strength_batch, Tensor)
            else None,
            valid_mask=event_mask[batch_index, :event_count],
        )
        structured_reverse_features[batch_index, 0] = event_reverse_scores[best_event_index]
        structured_reverse_features[batch_index, 1] = event_reverse_relations[best_event_index]
        structured_reverse_features[batch_index, 2] = event_boundary_gaps[best_event_index]
        structured_reverse_features[batch_index, 3] = event_reverse_available[best_event_index]
        path_count = int(path_mask[batch_index].sum().item())
        if path_count > 0:
            path_event_indices = list(candidate_path_event_indices_batch[batch_index])[:path_count]
            max_path_prob_by_event = _scatter_amax_by_indices(
                calibrated_path_prob_batch[batch_index, :path_count],
                path_event_indices,
                size=event_count,
                device=device,
            )
            alignment_features[batch_index, 0] = max_path_prob_by_event[best_event_index]
            best_path_index = int(torch.argmax(calibrated_path_logits_batch[batch_index, :path_count].float()).item())
            if 0 <= best_path_index < len(path_event_indices):
                top_path_event_index = int(path_event_indices[best_path_index])
                if 0 <= top_path_event_index < event_count and top_path_event_index == best_event_index:
                    alignment_features[batch_index, 2] = 1.0
            path_reverse_scores, path_boundary_gaps, path_reverse_relations, path_reverse_available, _ = _reverse_competition_features(
                calibrated_path_prob_batch[batch_index, :path_count],
                path_relation_strength_batch[batch_index, :path_count, :path_count]
                if isinstance(path_relation_strength_batch, Tensor)
                else None,
                valid_mask=path_mask[batch_index, :path_count],
            )
            best_path_index = int(torch.argmax(calibrated_path_logits_batch[batch_index, :path_count].float()).item())
            structured_reverse_features[batch_index, 4] = path_reverse_scores[best_path_index]
            structured_reverse_features[batch_index, 5] = path_reverse_relations[best_path_index]
            structured_reverse_features[batch_index, 6] = path_boundary_gaps[best_path_index]
            structured_reverse_features[batch_index, 7] = path_reverse_available[best_path_index]
        temporal_count = int(temporal_mask[batch_index].sum().item())
        if temporal_count > 0:
            temporal_event_indices = list(candidate_temporal_event_indices_batch[batch_index])[:temporal_count]
            max_temporal_prob_by_event = _scatter_amax_by_indices(
                temporal_prob_batch[batch_index, :temporal_count],
                temporal_event_indices,
                size=event_count,
                device=device,
            )
            alignment_features[batch_index, 1] = max_temporal_prob_by_event[best_event_index]
    return torch.cat(
        [
            event_competition_features,
            path_competition_features,
            temporal_competition_features,
            event_logit_stats,
            path_logit_stats,
            temporal_logit_stats,
            is_temporal_question,
            has_event_candidates,
            has_path_candidates,
            has_temporal_candidates,
            speaker_conflict_rate,
            time_conflict_rate,
            temporal_distractor_rate,
            best_exact_signature_cover,
            best_source_turn_support,
            best_preferred_path,
            best_support_anchor,
            alignment_features,
            structured_reverse_features,
        ],
        dim=-1,
    )


def _rank_ratio(scores: Tensor) -> Tensor:
    score_count = int(scores.numel())
    if score_count <= 0:
        return scores.new_zeros((0,))
    if score_count == 1:
        return scores.new_ones((1,))
    higher_counts = (scores.unsqueeze(0) > scores.unsqueeze(1)).sum(dim=1).to(dtype=scores.dtype)
    return 1.0 - (higher_counts / float(score_count - 1))


def _question_chain_hint_value(question: str, question_features: Mapping[str, Any]) -> float:
    tokens = set(_normalized_token_list(question))
    tokens.update(normalize_text(clean_text(item)) for item in list(question_features.get("question_anchor_tokens", []) or []))
    return 1.0 if any(token in CHAIN_HINT_TOKENS for token in tokens if token) else 0.0


def _depth_layer_scalar(depth_layer: str) -> float:
    normalized_layer = clean_text(depth_layer)
    if not normalized_layer:
        return 0.0
    index = DEPTH_LAYER_TO_ID.get(normalized_layer)
    if index is None:
        return 1.0
    return float(index + 1) / float(max(1, len(DEPTH_LAYERS)))


def _event_status_feature_values(event_features: Mapping[str, Any]) -> tuple[float, float, float]:
    status = clean_text(event_features.get("target_status", ""))
    return (
        1.0 if status == "current" else 0.0,
        1.0 if status == "past" else 0.0,
        1.0 if status == "planned" else 0.0,
    )


def _turn_feature_values(
    prepared_events: Sequence[Mapping[str, Any]],
    *,
    scores: Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    event_count = len(prepared_events)
    if event_count <= 0:
        empty = torch.zeros((0,), dtype=dtype, device=device)
        return empty, empty, empty
    turn_values = torch.tensor(
        [max(0, int(dict(event or {}).get("turn_index", 0) or 0)) for event in prepared_events],
        dtype=dtype,
        device=device,
    )
    max_turn = torch.clamp(turn_values.max(), min=1.0)
    turn_norm = turn_values / max_turn
    if scores.numel() <= 0:
        top_turn = turn_values.new_zeros(())
    else:
        top_turn = turn_values[int(torch.argmax(scores).item())]
    older_than_top = (turn_values < top_turn).to(dtype=dtype)
    turn_distance_to_top = torch.abs(turn_values - top_turn) / max_turn
    return turn_norm, older_than_top, turn_distance_to_top


def _build_event_tunnel_features(
    *,
    prepared_events: Sequence[Mapping[str, Any]],
    calibrated_event_probs: Tensor,
    max_rerank_path_prob_by_event: Tensor,
    max_calibrated_path_prob_by_event: Tensor,
    max_temporal_prob_by_event: Tensor,
    event_path_count: Tensor,
    event_relation_strength: Tensor,
    event_mask: Tensor,
    question: str,
    question_features: Mapping[str, Any],
) -> Tensor:
    event_count = len(prepared_events)
    dtype = calibrated_event_probs.dtype
    device = calibrated_event_probs.device
    if event_count <= 0:
        return calibrated_event_probs.new_zeros((0, EVENT_TUNNEL_FEATURE_DIM))
    valid_mask = event_mask[:event_count].bool()
    event_scores = calibrated_event_probs[:event_count].float()
    rank_ratio = _rank_ratio(event_scores).to(dtype=dtype)
    best_other_gap = (event_scores - _best_other_scores(event_scores)).to(dtype=dtype)
    event_reverse_scores, event_boundary_gaps, event_reverse_relations, event_reverse_available, _ = _reverse_competition_features(
        event_scores,
        event_relation_strength[:event_count, :event_count],
        valid_mask=valid_mask,
    )
    if event_count > 0:
        top_index = int(torch.argmax(event_scores.masked_fill(~valid_mask, float("-inf"))).item()) if bool(valid_mask.any()) else 0
        relation_to_top = event_relation_strength[:event_count, top_index].to(dtype=dtype)
    else:
        relation_to_top = calibrated_event_probs.new_zeros((0,))
    turn_norm, older_than_top, turn_distance_to_top = _turn_feature_values(
        prepared_events,
        scores=event_scores,
        dtype=dtype,
        device=device,
    )
    status_rows = [_event_status_feature_values(event) for event in prepared_events]
    status_features = torch.tensor(status_rows, dtype=dtype, device=device)
    depth_scalar = torch.tensor(
        [_depth_layer_scalar(clean_text(dict(event or {}).get("depth_layer", ""))) for event in prepared_events],
        dtype=dtype,
        device=device,
    )
    depth_present = (depth_scalar > 0.0).to(dtype=dtype)
    deep_layer = (depth_scalar > (1.0 / float(max(1, len(DEPTH_LAYERS))))).to(dtype=dtype)
    chain_hint = calibrated_event_probs.new_full(
        (event_count,),
        _question_chain_hint_value(question, question_features),
        dtype=dtype,
    )
    features = torch.cat(
        [
            calibrated_event_probs[:event_count].to(dtype=dtype).unsqueeze(-1),
            rank_ratio.unsqueeze(-1),
            best_other_gap.unsqueeze(-1),
            max_rerank_path_prob_by_event[:event_count].to(dtype=dtype).unsqueeze(-1),
            max_calibrated_path_prob_by_event[:event_count].to(dtype=dtype).unsqueeze(-1),
            max_temporal_prob_by_event[:event_count].to(dtype=dtype).unsqueeze(-1),
            (event_path_count[:event_count].to(dtype=dtype) / max(1.0, float(len(PATH_TYPES)))).clamp(max=1.0).unsqueeze(-1),
            event_reverse_scores.to(dtype=dtype).unsqueeze(-1),
            event_boundary_gaps.to(dtype=dtype).unsqueeze(-1),
            event_reverse_relations.to(dtype=dtype).unsqueeze(-1),
            event_reverse_available.to(dtype=dtype).unsqueeze(-1),
            relation_to_top.unsqueeze(-1),
            turn_norm.unsqueeze(-1),
            older_than_top.unsqueeze(-1),
            turn_distance_to_top.unsqueeze(-1),
            status_features[:, [0, 1]],
            depth_present.unsqueeze(-1),
            deep_layer.unsqueeze(-1),
            chain_hint.unsqueeze(-1),
        ],
        dim=-1,
    )
    return features.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


def _build_path_tunnel_features(
    *,
    event_scores_for_paths: Tensor,
    calibrated_path_scores: Tensor,
    temporal_support_scores: Tensor,
    path_pair_features: Tensor,
    path_reverse_scores: Tensor,
    path_boundary_gaps: Tensor,
    path_reverse_relations: Tensor,
    path_reverse_available: Tensor,
    path_coverage_for_paths: Tensor,
    event_rank_ratio_for_paths: Tensor,
    question: str,
    question_features: Mapping[str, Any],
) -> Tensor:
    path_count = int(calibrated_path_scores.numel())
    if path_count <= 0:
        return calibrated_path_scores.new_zeros((0, PATH_TUNNEL_FEATURE_DIM))
    dtype = calibrated_path_scores.dtype
    chain_hint = calibrated_path_scores.new_full(
        (path_count,),
        _question_chain_hint_value(question, question_features),
        dtype=dtype,
    )
    pair_slice = path_pair_features[:path_count]
    return torch.cat(
        [
            event_scores_for_paths.to(dtype=dtype).unsqueeze(-1),
            calibrated_path_scores.to(dtype=dtype).unsqueeze(-1),
            temporal_support_scores.to(dtype=dtype).unsqueeze(-1),
            path_reverse_scores.to(dtype=dtype).unsqueeze(-1),
            path_boundary_gaps.to(dtype=dtype).unsqueeze(-1),
            path_reverse_relations.to(dtype=dtype).unsqueeze(-1),
            path_reverse_available.to(dtype=dtype).unsqueeze(-1),
            path_coverage_for_paths.to(dtype=dtype).unsqueeze(-1),
            event_rank_ratio_for_paths.to(dtype=dtype).unsqueeze(-1),
            pair_slice[:, [3, 4, 15, 17]],
            chain_hint.unsqueeze(-1),
        ],
        dim=-1,
    )


def _normalized_id_frozenset(values: Sequence[str] | frozenset[str]) -> frozenset[str]:
    if isinstance(values, frozenset):
        return values
    return frozenset(clean_text(item) for item in values if clean_text(item))


def _event_distractor_supervision_loss(
    logits: Tensor,
    *,
    candidate_ids: Sequence[str],
    positive_ids: Sequence[str],
    hard_negative_ids: Sequence[str],
) -> Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    positive_id_set = _normalized_id_frozenset(positive_ids)
    hard_negative_id_set = _normalized_id_frozenset(hard_negative_ids)
    if not positive_id_set and not hard_negative_id_set:
        return logits.sum() * 0.0
    supervised_indices: List[int] = []
    supervised_targets: List[float] = []
    for index, candidate_id in enumerate(candidate_ids):
        normalized_id = clean_text(candidate_id)
        if not normalized_id:
            continue
        if normalized_id in positive_id_set:
            supervised_indices.append(index)
            supervised_targets.append(0.0)
        elif normalized_id in hard_negative_id_set:
            supervised_indices.append(index)
            supervised_targets.append(1.0)
    if not supervised_indices:
        return logits.sum() * 0.0
    working_logits = logits.float()
    device = working_logits.device
    index_tensor = torch.tensor(supervised_indices, dtype=torch.long, device=device)
    target_tensor = torch.tensor(supervised_targets, dtype=working_logits.dtype, device=device)
    selected_logits = working_logits.index_select(0, index_tensor)
    return F.binary_cross_entropy_with_logits(selected_logits, target_tensor)


def _hard_negative_margin_loss(
    logits: Tensor,
    *,
    candidate_ids: Sequence[str],
    positive_ids: Sequence[str],
    hard_negative_ids: Sequence[str],
    margin: float,
) -> Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    positive_id_set = _normalized_id_frozenset(positive_ids)
    hard_negative_id_set = _normalized_id_frozenset(hard_negative_ids)
    if not positive_id_set or not hard_negative_id_set:
        return logits.sum() * 0.0
    positive_indices = [index for index, candidate_id in enumerate(candidate_ids) if clean_text(candidate_id) in positive_id_set]
    hard_negative_indices = [index for index, candidate_id in enumerate(candidate_ids) if clean_text(candidate_id) in hard_negative_id_set]
    if not positive_indices or not hard_negative_indices:
        return logits.sum() * 0.0
    working_logits = logits.float()
    device = working_logits.device
    positive_logits = working_logits.index_select(0, torch.tensor(positive_indices, dtype=torch.long, device=device))
    negative_logits = working_logits.index_select(0, torch.tensor(hard_negative_indices, dtype=torch.long, device=device))
    margins = float(margin) - (positive_logits.unsqueeze(-1) - negative_logits.unsqueeze(0))
    return F.relu(margins).mean()


def _online_hard_negative_ids(
    logits: Tensor,
    *,
    candidate_ids: Sequence[str],
    positive_ids: Sequence[str],
    limit: int,
    protected_ids: Sequence[str] = (),
) -> List[str]:
    resolved_limit = max(0, int(limit or 0))
    if resolved_limit <= 0 or logits.numel() == 0:
        return []
    positive_id_set = _normalized_id_frozenset(positive_ids)
    protected_id_set = _normalized_id_frozenset(protected_ids)
    scored_candidates: List[tuple[str, float]] = []
    detached_scores = logits.detach().float().cpu().tolist()
    for candidate_id, score in zip(candidate_ids, detached_scores):
        normalized_id = clean_text(candidate_id)
        if not normalized_id or normalized_id in positive_id_set or normalized_id in protected_id_set:
            continue
        scored_candidates.append((normalized_id, float(score)))
    scored_candidates.sort(key=lambda item: (-item[1], item[0]))
    return dedupe_texts([candidate_id for candidate_id, _ in scored_candidates], max_items=resolved_limit)


def _online_hard_negative_path_ids(
    logits: Tensor,
    *,
    candidate_path_ids: Sequence[str],
    positive_path_ids: Sequence[str],
    candidate_path_event_ids: Sequence[str] | None = None,
    positive_event_ids: Sequence[str] = (),
    limit: int,
) -> List[str]:
    resolved_limit = max(0, int(limit or 0))
    if resolved_limit <= 0 or logits.numel() == 0:
        return []
    positive_path_id_set = _normalized_id_frozenset(positive_path_ids)
    positive_event_id_set = _normalized_id_frozenset(positive_event_ids)
    event_ids = list(candidate_path_event_ids or [])
    scored_candidates: List[tuple[str, float]] = []
    detached_scores = logits.detach().float().cpu().tolist()
    for index, (path_id, score) in enumerate(zip(candidate_path_ids, detached_scores)):
        normalized_path_id = clean_text(path_id)
        if not normalized_path_id or normalized_path_id in positive_path_id_set:
            continue
        event_id = clean_text(event_ids[index]) if index < len(event_ids) else parse_path_id(normalized_path_id)[0]
        if event_id and event_id in positive_event_id_set:
            continue
        scored_candidates.append((normalized_path_id, float(score)))
    scored_candidates.sort(key=lambda item: (-item[1], item[0]))
    return dedupe_texts([path_id for path_id, _ in scored_candidates], max_items=resolved_limit)


def _multi_positive_supervision_active(supervision: Mapping[str, Any]) -> bool:
    bucket = clean_text(supervision.get("supervision_bucket", ""))
    answer_type = clean_text(supervision.get("answer_type", ""))
    positive_event_count = int(supervision.get("positive_event_count", 0) or 0)
    if bucket.endswith("_multi_positive") or bucket in {"list_event_text_multi_positive", "multi_evidence"}:
        return True
    return positive_event_count > 1 and answer_type in {"event_text", "multi_evidence"}


def _selection_positive_coverage_count(
    supervision: Mapping[str, Any],
    *,
    positive_count: int,
    top_k: int,
    base_count: int,
    multi_positive_count: int,
    multi_positive_fraction: float,
) -> int:
    resolved_positive_count = max(0, int(positive_count or 0))
    if resolved_positive_count <= 0:
        return 1
    resolved_top_k = max(1, int(top_k or 1))
    resolved_base_count = max(1, int(base_count or 1))
    if not _multi_positive_supervision_active(supervision):
        return max(1, min(resolved_positive_count, resolved_top_k, resolved_base_count))
    resolved_multi_count = max(resolved_base_count, int(multi_positive_count or resolved_base_count))
    fraction_target = int(math.ceil(float(resolved_positive_count) * max(0.0, float(multi_positive_fraction or 0.0))))
    target_count = max(resolved_base_count, fraction_target)
    return max(1, min(resolved_positive_count, resolved_top_k, resolved_multi_count, target_count))


def _positive_coverage_hit_count(
    ranked_ids: Sequence[str],
    positive_ids: Sequence[str],
    *,
    limit: int,
) -> int:
    positive_id_set = _normalized_id_frozenset(positive_ids)
    if not positive_id_set:
        return 0
    selected_ids = _normalized_id_frozenset(ranked_ids[: max(0, int(limit or 0))])
    return len(positive_id_set & selected_ids)


def _selection_margin_loss(
    logits: Tensor,
    *,
    candidate_ids: Sequence[str],
    positive_ids: Sequence[str],
    margin: float,
    top_k: int = 1,
    required_positive_count: int = 1,
) -> Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    positive_id_set = _normalized_id_frozenset(positive_ids)
    if not positive_id_set:
        return logits.sum() * 0.0
    positive_indices = [index for index, candidate_id in enumerate(candidate_ids) if clean_text(candidate_id) in positive_id_set]
    negative_indices = [index for index, candidate_id in enumerate(candidate_ids) if clean_text(candidate_id) not in positive_id_set]
    if not positive_indices or not negative_indices:
        return logits.sum() * 0.0
    working_logits = logits.float()
    device = working_logits.device
    positive_logits = working_logits.index_select(0, torch.tensor(positive_indices, dtype=torch.long, device=device))
    negative_logits = working_logits.index_select(0, torch.tensor(negative_indices, dtype=torch.long, device=device))
    resolved_top_k = max(1, int(top_k or 1))
    resolved_required_positive_count = max(
        1,
        min(
            int(required_positive_count or 1),
            int(positive_logits.numel()),
            resolved_top_k,
        ),
    )
    negative_rank = max(1, resolved_top_k - resolved_required_positive_count + 1)
    if negative_rank > int(negative_logits.numel()):
        # If the runtime selection window is wider than all negatives, any
        # required positive candidates are guaranteed to survive this stage.
        return logits.sum() * 0.0
    positive_threshold = torch.topk(
        positive_logits,
        k=resolved_required_positive_count,
        dim=0,
    ).values[-1]
    negative_threshold = torch.topk(negative_logits, k=negative_rank, dim=0).values[-1]
    return F.relu(torch.tensor(float(margin), dtype=working_logits.dtype, device=device) - (positive_threshold - negative_threshold))


def _positive_vs_negative_ids_margin_loss(
    logits: Tensor,
    *,
    candidate_ids: Sequence[str],
    positive_ids: Sequence[str] | frozenset[str],
    negative_ids: Sequence[str] | frozenset[str],
    margin: float,
) -> Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    positive_id_set = _normalized_id_frozenset(positive_ids)
    negative_id_set = _normalized_id_frozenset(negative_ids)
    if not positive_id_set or not negative_id_set:
        return logits.sum() * 0.0
    positive_indices = [
        index for index, candidate_id in enumerate(candidate_ids)
        if clean_text(candidate_id) in positive_id_set
    ]
    negative_indices = [
        index for index, candidate_id in enumerate(candidate_ids)
        if clean_text(candidate_id) in negative_id_set
    ]
    if not positive_indices or not negative_indices:
        return logits.sum() * 0.0
    working_logits = logits.float()
    device = working_logits.device
    positive_logits = working_logits.index_select(0, torch.tensor(positive_indices, dtype=torch.long, device=device))
    negative_logits = working_logits.index_select(0, torch.tensor(negative_indices, dtype=torch.long, device=device))
    positive_threshold = torch.max(positive_logits)
    negative_threshold = torch.max(negative_logits)
    return F.relu(torch.tensor(float(margin), dtype=working_logits.dtype, device=device) - (positive_threshold - negative_threshold))


def _event_decision_scores_for_final_set(
    event_logits: Tensor,
    *,
    candidate_event_ids: Sequence[str],
    path_logits: Tensor,
    candidate_path_event_ids: Sequence[str],
) -> Tensor:
    if event_logits.numel() == 0:
        return event_logits.sum().reshape(1)[:0]
    event_scores = torch.sigmoid(event_logits.float())
    if path_logits.numel() == 0 or not candidate_path_event_ids:
        return event_scores.to(dtype=event_logits.dtype)
    path_scores = torch.sigmoid(path_logits.float())
    event_to_path_scores: Dict[str, List[Tensor]] = {}
    for path_index, event_id in enumerate(candidate_path_event_ids):
        if path_index >= int(path_scores.numel()):
            break
        normalized_event_id = clean_text(event_id)
        if not normalized_event_id:
            continue
        event_to_path_scores.setdefault(normalized_event_id, []).append(path_scores[path_index])
    decision_scores: List[Tensor] = []
    for event_index, event_id in enumerate(candidate_event_ids):
        components = [event_scores[event_index]]
        components.extend(event_to_path_scores.get(clean_text(event_id), []))
        decision_scores.append(torch.stack(components).max())
    if not decision_scores:
        return event_scores.new_zeros((0,), dtype=event_logits.dtype)
    return torch.stack(decision_scores).to(dtype=event_logits.dtype)


def _runtime_selected_event_set_loss(
    event_logits: Tensor,
    *,
    candidate_event_ids: Sequence[str],
    path_logits: Tensor,
    candidate_path_event_ids: Sequence[str],
    positive_event_ids: Sequence[str],
    margin: float,
    top_k: int,
    support_path_k: int,
    required_positive_count: int = 1,
) -> Tensor:
    """Match the runtime selected-event construction more closely.

    Runtime prepends events from the highest scoring support paths before it
    fills the remaining prompt window with event scores. A loss that only sees
    event top-k can therefore pass while high-scoring negative paths consume
    the exact slots that would have preserved a gold event.
    """

    if event_logits.numel() == 0:
        return event_logits.sum() * 0.0
    positive_id_set = _normalized_id_frozenset(positive_event_ids)
    if not positive_id_set:
        return event_logits.sum() * 0.0
    resolved_top_k = max(1, int(top_k or 1))
    resolved_support_path_k = max(0, int(support_path_k or 0))
    resolved_required_positive_count = max(1, int(required_positive_count or 1))
    event_window_top_k = max(1, resolved_top_k - resolved_support_path_k)
    event_window_loss = _selection_margin_loss(
        event_logits,
        candidate_ids=candidate_event_ids,
        positive_ids=positive_event_ids,
        margin=margin,
        top_k=event_window_top_k,
        required_positive_count=resolved_required_positive_count,
    )
    if path_logits.numel() == 0 or not candidate_path_event_ids or resolved_support_path_k <= 0:
        return event_window_loss
    working_path_logits = path_logits.float()
    device = working_path_logits.device
    positive_path_scores_by_event: Dict[str, Tensor] = {}
    negative_path_scores_by_event: Dict[str, Tensor] = {}
    for path_index, event_id in enumerate(candidate_path_event_ids):
        if path_index >= int(working_path_logits.numel()):
            break
        normalized_event_id = clean_text(event_id)
        if not normalized_event_id:
            continue
        score = working_path_logits[path_index]
        if normalized_event_id in positive_id_set:
            previous = positive_path_scores_by_event.get(normalized_event_id)
            if previous is None:
                positive_path_scores_by_event[normalized_event_id] = score
            else:
                positive_path_scores_by_event[normalized_event_id] = torch.maximum(previous, score)
        else:
            previous = negative_path_scores_by_event.get(normalized_event_id)
            if previous is None:
                negative_path_scores_by_event[normalized_event_id] = score
            else:
                negative_path_scores_by_event[normalized_event_id] = torch.maximum(previous, score)
    if not positive_path_scores_by_event or not negative_path_scores_by_event:
        return event_window_loss
    positive_path_scores = torch.stack(list(positive_path_scores_by_event.values()))
    negative_path_scores = torch.stack(list(negative_path_scores_by_event.values()))
    required_path_positive_count = max(
        1,
        min(
            resolved_required_positive_count,
            int(positive_path_scores.numel()),
            resolved_support_path_k,
        ),
    )
    negative_rank = max(1, resolved_support_path_k - required_path_positive_count + 1)
    if negative_rank > int(negative_path_scores.numel()):
        return event_window_loss
    positive_path_threshold = torch.topk(
        positive_path_scores,
        k=required_path_positive_count,
        dim=0,
    ).values[-1]
    negative_prefix_count = min(negative_rank, int(negative_path_scores.numel()))
    negative_threshold = torch.topk(negative_path_scores, k=negative_prefix_count, dim=0).values[-1]
    path_prefix_loss = F.relu(
        torch.tensor(float(margin), dtype=working_path_logits.dtype, device=device)
        - (positive_path_threshold - negative_threshold)
    )
    return 0.5 * (event_window_loss + path_prefix_loss.to(dtype=event_window_loss.dtype))


def _matrix_delta_direction_loss(
    delta_logits: Tensor,
    *,
    matrix_ids: Sequence[str],
    positive_ids: Sequence[str],
    hard_negative_ids: Sequence[str],
    margin: float,
    positive_weight: float = 1.0,
    hard_negative_weight: float = 1.0,
    pair_weight: float = 1.0,
) -> Tensor:
    if delta_logits.numel() == 0:
        return delta_logits.sum() * 0.0
    positive_id_set = _normalized_id_frozenset(positive_ids)
    hard_negative_id_set = _normalized_id_frozenset(hard_negative_ids)
    if not positive_id_set and not hard_negative_id_set:
        return delta_logits.sum() * 0.0
    working_delta = delta_logits.float()
    device = working_delta.device
    weighted_losses: List[Tensor] = []
    weights: List[float] = []
    positive_indices = [index for index, item_id in enumerate(matrix_ids) if clean_text(item_id) in positive_id_set]
    hard_negative_indices = [index for index, item_id in enumerate(matrix_ids) if clean_text(item_id) in hard_negative_id_set]
    if positive_indices:
        positive_delta = working_delta.index_select(0, torch.tensor(positive_indices, dtype=torch.long, device=device))
        weighted_losses.append(
            float(positive_weight) * F.relu(torch.tensor(float(margin), dtype=working_delta.dtype, device=device) - positive_delta).mean()
        )
        weights.append(max(0.0, float(positive_weight)))
    if hard_negative_indices:
        hard_negative_delta = working_delta.index_select(0, torch.tensor(hard_negative_indices, dtype=torch.long, device=device))
        weighted_losses.append(
            float(hard_negative_weight)
            * F.relu(torch.tensor(float(margin), dtype=working_delta.dtype, device=device) + hard_negative_delta).mean()
        )
        weights.append(max(0.0, float(hard_negative_weight)))
    if positive_indices and hard_negative_indices:
        positive_delta = working_delta.index_select(0, torch.tensor(positive_indices, dtype=torch.long, device=device))
        hard_negative_delta = working_delta.index_select(0, torch.tensor(hard_negative_indices, dtype=torch.long, device=device))
        pair_margin = torch.tensor(float(margin), dtype=working_delta.dtype, device=device)
        weighted_losses.append(float(pair_weight) * F.relu(pair_margin - (positive_delta.unsqueeze(-1) - hard_negative_delta.unsqueeze(0))).mean())
        weights.append(max(0.0, float(pair_weight)))
    if not weighted_losses:
        return delta_logits.sum() * 0.0
    weight_sum = sum(weight for weight in weights if weight > 0.0)
    if weight_sum <= 0.0:
        return delta_logits.sum() * 0.0
    return torch.stack(weighted_losses).sum() / float(weight_sum)


def _answer_refusal_margin_loss(
    logits: Tensor,
    *,
    target_index: int,
    has_positive_supervision: bool,
    margin: float,
) -> Tensor:
    if logits.numel() == 0 or not has_positive_supervision:
        return logits.sum() * 0.0
    abstain_index = ANSWER_TYPE_TO_ID["abstain"]
    if target_index == abstain_index:
        return logits.sum() * 0.0
    working_logits = logits.float()
    gold_logit = working_logits[target_index]
    abstain_logit = working_logits[abstain_index]
    return F.relu(torch.tensor(float(margin), dtype=working_logits.dtype, device=working_logits.device) - (gold_logit - abstain_logit))


def _hard_negative_path_ids(example: QueryTrainingExample, candidate_path_ids: Sequence[str]) -> List[str]:
    question_features = dict(example.question_features or {})
    hard_negative_event_ids = {clean_text(item) for item in _resolved_hard_negative_event_ids_for_example(example) if clean_text(item)}
    positive_event_ids = {clean_text(item) for item in example.positive_event_ids if clean_text(item)}
    preferred_path_types = _preferred_support_path_types(question_features)
    primary_preferred_path_type = clean_text(preferred_path_types[0]) if preferred_path_types else ""
    semantic_target = clean_text(question_features.get("semantic_slot_target", ""))
    typed_query = bool(question_features.get("is_temporal", False)) or semantic_target in {
        "identity",
        "research_topic",
        "education",
        "occupation",
        "profile",
        "event_time",
    } or bool(clean_text(question_features.get("target_status_target", "")))
    hard_negative_path_ids: List[str] = []
    for path_id in candidate_path_ids:
        event_id, path_type, _ = parse_path_id(path_id)
        if not event_id or not path_type:
            continue
        if event_id in hard_negative_event_ids:
            hard_negative_path_ids.append(path_id)
            continue
        if typed_query and primary_preferred_path_type and event_id in positive_event_ids and path_type != primary_preferred_path_type:
            hard_negative_path_ids.append(path_id)
    return dedupe_texts(hard_negative_path_ids)


def _label_tensor_from_positive_ids(
    candidate_ids: Sequence[str],
    positive_ids: frozenset[str],
    *,
    device: torch.device,
) -> Tensor:
    return torch.tensor(
        [1.0 if candidate_id in positive_ids else 0.0 for candidate_id in candidate_ids],
        dtype=torch.float32,
        device=device,
    )


def _binary_plan_loss_for_ids(
    logits: Tensor,
    *,
    candidate_ids: Sequence[str],
    positive_ids: Sequence[str] | frozenset[str],
    device: torch.device,
) -> Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    target_ids = _normalized_id_frozenset(positive_ids)
    if not target_ids:
        return logits.sum() * 0.0
    targets = _label_tensor_from_positive_ids(candidate_ids, target_ids, device=device)
    return F.binary_cross_entropy_with_logits(logits.float(), targets)


def _build_example_supervision_payload(example: QueryTrainingExample) -> Dict[str, Any]:
    question_features = dict(example.question_features or {})
    metadata = dict(example.metadata or {})
    answer_plan_targets = dict(example.answer_plan_targets or {})
    selected_memory_ids = frozenset(
        clean_text(item)
        for item in list(answer_plan_targets.get("selected_memory_ids", []) or [])
        if clean_text(item)
    )
    protected_memory_ids = frozenset(
        clean_text(item)
        for item in list(answer_plan_targets.get("protected_memory_ids", []) or [])
        if clean_text(item)
    )
    current_memory_ids = frozenset(
        clean_text(item)
        for item in list(answer_plan_targets.get("current_memory_ids", []) or [])
        if clean_text(item)
    )
    historical_memory_ids = frozenset(
        clean_text(item)
        for item in list(answer_plan_targets.get("historical_memory_ids", []) or [])
        if clean_text(item)
    )
    suppressed_memory_ids = frozenset(
        clean_text(item)
        for item in list(answer_plan_targets.get("suppressed_memory_ids", []) or [])
        if clean_text(item)
    )
    preferred_path_types = _preferred_support_path_types(question_features)
    semantic_target = clean_text(question_features.get("semantic_slot_target", ""))
    typed_query = bool(question_features.get("is_temporal", False)) or semantic_target in {
        "identity",
        "research_topic",
        "education",
        "occupation",
        "profile",
        "event_time",
    } or bool(clean_text(question_features.get("target_status_target", "")))
    return {
        "answer_type_index": int(_answer_type_label(example)),
        "answer_type": _training_answer_type_name(example),
        "supervision_bucket": _training_supervision_bucket_name(example),
        "positive_event_count": _positive_event_id_count(example),
        "has_positive_supervision": bool(example.positive_event_ids or example.positive_path_ids),
        "is_temporal_example": bool(_is_temporal_example(example)),
        "positive_event_ids": frozenset(
            clean_text(item) for item in example.positive_event_ids if clean_text(item)
        ),
        "positive_path_ids": frozenset(
            clean_text(item) for item in example.positive_path_ids if clean_text(item)
        ),
        "positive_time_node_ids": frozenset(
            clean_text(item) for item in example.positive_time_node_ids if clean_text(item)
        ),
        "has_answer_plan_supervision": bool(
            selected_memory_ids
            or protected_memory_ids
            or current_memory_ids
            or historical_memory_ids
            or suppressed_memory_ids
        ),
        "answer_plan_selected_event_ids": frozenset(selected_memory_ids or protected_memory_ids),
        "answer_plan_current_event_ids": current_memory_ids,
        "answer_plan_historical_event_ids": historical_memory_ids,
        "answer_plan_suppressed_event_ids": suppressed_memory_ids,
        "negative_event_ids": frozenset(
            clean_text(item) for item in example.negative_event_ids if clean_text(item)
        ),
        "hard_negative_event_ids": frozenset(_resolved_hard_negative_event_ids_for_example(example)),
        "ancestor_event_ids": frozenset(
            clean_text(item) for item in list(metadata.get("ancestor_event_ids", []) or []) if clean_text(item)
        ),
        "answer_primary_event_ids": frozenset(
            clean_text(item) for item in list(metadata.get("answer_primary_event_ids", []) or []) if clean_text(item)
        ),
        "side_branch_event_ids": frozenset(
            clean_text(item) for item in list(metadata.get("side_branch_event_ids", []) or []) if clean_text(item)
        ),
        "distractor_event_ids": frozenset(
            clean_text(item) for item in list(metadata.get("distractor_event_ids", []) or []) if clean_text(item)
        ),
        "typed_query": bool(typed_query),
        "primary_preferred_path_type": clean_text(preferred_path_types[0]) if preferred_path_types else "",
        "training_example_weight": float(metadata.get("_training_example_weight", 1.0) or 1.0),
        "training_source_dataset": clean_text(metadata.get("_training_source_dataset", "")),
        "training_sampling_multiplier": float(metadata.get("_training_sampling_multiplier", 1.0) or 1.0),
    }


def _path_hard_negative_ids_from_supervision(
    supervision: Mapping[str, Any],
    *,
    candidate_path_ids: Sequence[str],
    candidate_path_event_ids: Sequence[str] | None = None,
    candidate_path_types: Sequence[str] | None = None,
) -> List[str]:
    hard_negative_event_ids = _resolved_hard_negative_event_ids_from_supervision(supervision)
    positive_event_ids = {
        clean_text(item) for item in list(supervision.get("positive_event_ids", []) or []) if clean_text(item)
    }
    typed_query = bool(supervision.get("typed_query", False))
    primary_preferred_path_type = clean_text(supervision.get("primary_preferred_path_type", ""))
    hard_negative_path_ids: List[str] = []
    for index, path_id in enumerate(candidate_path_ids):
        event_id = (
            clean_text(candidate_path_event_ids[index])
            if candidate_path_event_ids is not None and index < len(candidate_path_event_ids)
            else ""
        )
        path_type = (
            clean_text(candidate_path_types[index])
            if candidate_path_types is not None and index < len(candidate_path_types)
            else ""
        )
        if not event_id or not path_type:
            parsed_event_id, parsed_path_type, _ = parse_path_id(path_id)
            event_id = event_id or parsed_event_id
            path_type = path_type or parsed_path_type
        if not event_id or not path_type:
            continue
        if event_id in hard_negative_event_ids:
            hard_negative_path_ids.append(path_id)
            continue
        if typed_query and primary_preferred_path_type and event_id in positive_event_ids and path_type != primary_preferred_path_type:
            hard_negative_path_ids.append(path_id)
    return dedupe_texts(hard_negative_path_ids)


def _tunnel_positive_event_ids_from_supervision(supervision: Mapping[str, Any]) -> frozenset[str]:
    bucket = clean_text(supervision.get("supervision_bucket", ""))
    positive_ids = {
        clean_text(item)
        for item in list(supervision.get("positive_event_ids", []) or [])
        if clean_text(item)
    }
    if bucket in {"chain_retrieval", "current_chain_head_selection"} or bucket.startswith("depth_chain"):
        positive_ids.update(
            clean_text(item)
            for item in list(supervision.get("ancestor_event_ids", []) or [])
            if clean_text(item)
        )
        positive_ids.update(
            clean_text(item)
            for item in list(supervision.get("answer_primary_event_ids", []) or [])
            if clean_text(item)
        )
    return frozenset(positive_ids)


def _tunnel_hard_negative_event_ids_from_supervision(supervision: Mapping[str, Any]) -> frozenset[str]:
    positive_ids = _tunnel_positive_event_ids_from_supervision(supervision)
    hard_negative_ids = {
        clean_text(item)
        for item in list(supervision.get("hard_negative_event_ids", []) or [])
        if clean_text(item)
    }
    hard_negative_ids.update(
        clean_text(item)
        for item in list(supervision.get("distractor_event_ids", []) or [])
        if clean_text(item)
    )
    hard_negative_ids.update(
        clean_text(item)
        for item in list(supervision.get("side_branch_event_ids", []) or [])
        if clean_text(item)
    )
    return frozenset(item for item in hard_negative_ids if item and item not in positive_ids)


def _temporal_path_labels_from_supervision(
    supervision: Mapping[str, Any],
    candidate_temporal_path_ids: Sequence[str],
    candidate_temporal_node_ids: Sequence[str],
    *,
    candidate_temporal_event_ids: Sequence[str] | None = None,
    device: torch.device,
) -> Tensor:
    positive_event_ids = {
        clean_text(item) for item in list(supervision.get("positive_event_ids", []) or []) if clean_text(item)
    }
    positive_time_node_ids = {
        clean_text(item) for item in list(supervision.get("positive_time_node_ids", []) or []) if clean_text(item)
    }
    temporal_path_targets: List[float] = []
    for index, (path_id, time_node_id) in enumerate(zip(candidate_temporal_path_ids, candidate_temporal_node_ids)):
        event_id = (
            clean_text(candidate_temporal_event_ids[index])
            if candidate_temporal_event_ids is not None and index < len(candidate_temporal_event_ids)
            else ""
        )
        if not event_id:
            event_id, _, _ = parse_path_id(path_id)
        is_positive = bool(event_id in positive_event_ids and clean_text(time_node_id) in positive_time_node_ids)
        temporal_path_targets.append(1.0 if is_positive else 0.0)
    return torch.tensor(temporal_path_targets, dtype=torch.float32, device=device)


_BATCH_PREPARE_WORKER_CONTEXT: Dict[str, Any] = {}
_BATCH_PREPARE_WORKER_CACHE: OrderedDict[str, GraphCacheItem] = OrderedDict()
_BATCH_PREPARE_WORKER_MEMORY_CACHE_SIZE = 0
_BATCH_PREPARE_THREAD_STATE = threading.local()


class _PreparedBatchWorkerPool:
    def __init__(
        self,
        *,
        worker_context: Mapping[str, Any] | None,
        worker_count: int,
    ) -> None:
        self.worker_context = _normalize_batch_prepare_worker_context(worker_context)
        self.worker_count = max(0, int(worker_count or 0))
        self._executor: ProcessPoolExecutor | None = None
        self._thread_executor: ThreadPoolExecutor | None = None
        self._direct_fallback_mode = False
        self._thread_fallback_mode = False
        self._worker_exc: Exception | None = None
        self._process_recovery_attempts = 0
        self._max_process_recovery_attempts = max(0, int(DEFAULT_BATCH_PREPARE_PROCESS_RECOVERY_ATTEMPTS or 0))
        self._current_process_worker_count = self.worker_count
        self._process_success_count = 0
        self._max_tasks_per_child = max(0, int(DEFAULT_BATCH_PREPARE_WORKER_MAX_TASKS_PER_CHILD or 0))
        self._thread_worker_count = max(
            int(DEFAULT_BATCH_PREPARE_THREAD_FALLBACK_MIN_WORKERS),
            min(16, max(1, int(self.worker_count or 0) * int(DEFAULT_BATCH_PREPARE_THREAD_FALLBACK_WORKERS_MULTIPLIER))),
        )

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        self._close_process_executor()
        self._close_thread_executor()

    def _close_process_executor(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def _close_thread_executor(self) -> None:
        executor = self._thread_executor
        self._thread_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def _enable_direct_fallback(self, exc: Exception) -> None:
        self._worker_exc = exc
        self._direct_fallback_mode = True
        self.close()

    def _enable_thread_fallback(self, exc: Exception) -> None:
        self._worker_exc = exc
        self._thread_fallback_mode = True
        self._close_process_executor()

    def _record_worker_success(self, *, backend: str) -> None:
        self._worker_exc = None
        if backend == "process":
            self._process_recovery_attempts = 0
            self._process_success_count += 1

    def _process_worker_count(self) -> int:
        return max(1, int(self._current_process_worker_count or self.worker_count or 1))

    def _recover_process_pool(self, exc: Exception) -> Dict[str, Any]:
        previous_worker_count = self._process_worker_count()
        self._worker_exc = exc
        self._close_process_executor()
        self._process_success_count = 0
        self._process_recovery_attempts += 1
        shrunk_worker_count = previous_worker_count
        if previous_worker_count > 1:
            shrunk_worker_count = max(1, math.ceil(previous_worker_count / 2))
            self._current_process_worker_count = shrunk_worker_count
        if self._process_recovery_attempts > self._max_process_recovery_attempts:
            self._enable_thread_fallback(exc)
            return {
                "recovery_mode": "thread_fallback",
                "recovery_attempt": int(self._process_recovery_attempts),
                "previous_worker_count": int(previous_worker_count),
                "next_worker_count": int(self._thread_worker_count),
            }
        return {
            "recovery_mode": "process_retry",
            "recovery_attempt": int(self._process_recovery_attempts),
            "previous_worker_count": int(previous_worker_count),
            "next_worker_count": int(self._process_worker_count()),
        }

    def _ensure_executor(self) -> ProcessPoolExecutor | None:
        if self.worker_count <= 0 or self._direct_fallback_mode or self._thread_fallback_mode:
            return None
        if self._executor is not None:
            return self._executor
        try:
            executor_kwargs: Dict[str, Any] = {
                "max_workers": self._process_worker_count(),
                "mp_context": multiprocessing.get_context("spawn"),
                "initializer": _initialize_batch_prepare_worker,
                "initargs": (self.worker_context,),
            }
            executor_signature = inspect.signature(ProcessPoolExecutor)
            if "max_tasks_per_child" in executor_signature.parameters and self._max_tasks_per_child > 0:
                executor_kwargs["max_tasks_per_child"] = int(self._max_tasks_per_child)
            self._executor = ProcessPoolExecutor(
                **executor_kwargs,
            )
        except Exception as exc:
            self._enable_thread_fallback(exc)
        return self._executor

    def _ensure_thread_executor(self) -> ThreadPoolExecutor | None:
        if self._direct_fallback_mode:
            return None
        if self._thread_executor is not None:
            return self._thread_executor
        try:
            self._thread_executor = ThreadPoolExecutor(
                max_workers=max(1, int(self._thread_worker_count)),
                thread_name_prefix="prepared-batch-thread",
            )
        except Exception as exc:
            self._enable_direct_fallback(exc)
        return self._thread_executor

    def iter_prepared_batches(
        self,
        rows: Sequence[QueryTrainingExample],
        *,
        graph_cache: Mapping[str, GraphCacheItem],
        batch_size: int,
        shuffle: bool,
        seed: int,
        lookahead_batches: int,
        graph_error_stage: str,
        skip_batches: int = 0,
        prefer_completion_order: bool = DEFAULT_BATCH_PREPARE_PREFER_COMPLETION_ORDER,
        worker_event_callback: Callable[[Dict[str, Any]], None] | None = None,
    ) -> Iterator[Dict[str, Any]]:
        batch_iter = iter(
            _iter_batches_any(
                rows,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                skip_batches=skip_batches,
            )
        )
        resolved_worker_context = dict(self.worker_context or {})
        has_conversation_specs = bool(dict(resolved_worker_context).get("conversation_specs"))
        has_graph_dir = bool(clean_text(resolved_worker_context.get("graph_dir", "")))
        if self.worker_count <= 0 or (not has_conversation_specs and not has_graph_dir):
            for batch in batch_iter:
                yield _prepare_batch_payload_from_graph_cache(
                    batch,
                    graph_cache,
                    merge_device=torch.device("cpu"),
                    graph_error_stage=graph_error_stage,
                )
            return

        pending_items: deque[
            tuple[QueryBatchSpec | List[QueryTrainingExample], Future[Dict[str, Any]] | None, List[Dict[str, Any]], str]
        ] = deque()
        worker_parallelism = max(self.worker_count, self._thread_worker_count)
        pending_target = max(
            1,
            max(0, int(lookahead_batches or 0), int(worker_parallelism) * int(DEFAULT_BATCH_PREPARE_PENDING_MULTIPLIER)) + 1,
        )
        resolved_graph_error_stage = clean_text(graph_error_stage) or "prepare_batch"
        use_completion_order = bool(prefer_completion_order)

        def _emit_worker_event(event_type: str, **payload: Any) -> None:
            if worker_event_callback is None:
                return
            worker_event_callback(
                {
                    "event_type": clean_text(event_type),
                    "graph_error_stage": resolved_graph_error_stage,
                    **dict(payload),
                }
            )

        def _current_pending_target() -> int:
            if self._direct_fallback_mode:
                return 1
            if self._thread_fallback_mode:
                worker_parallelism = max(1, int(self._thread_worker_count))
                return max(
                    1,
                    max(0, int(lookahead_batches or 0), int(worker_parallelism) * int(DEFAULT_BATCH_PREPARE_PENDING_MULTIPLIER)) + 1,
                )
            process_parallelism = self._process_worker_count()
            steady_target = max(
                1,
                max(0, int(lookahead_batches or 0), int(process_parallelism) * int(DEFAULT_BATCH_PREPARE_PENDING_MULTIPLIER)) + 1,
            )
            if self._process_success_count > 0:
                return steady_target
            return max(
                2,
                min(
                    steady_target,
                    max(
                        process_parallelism + 1,
                        int(process_parallelism) * int(DEFAULT_BATCH_PREPARE_PROCESS_WARMUP_PENDING_MULTIPLIER),
                    ),
                ),
            )

        def _queue_local_batch(
            batch_payload: QueryBatchSpec | List[QueryTrainingExample],
            *,
            worker_exc: Exception | None = None,
        ) -> None:
            worker_error_payloads = (
                _batch_prepare_worker_error_payloads(
                    batch_payload,
                    worker_exc,
                    graph_error_stage=graph_error_stage,
                )
                if worker_exc is not None
                else []
            )
            pending_items.append((batch_payload, None, list(worker_error_payloads), "local"))

        def _submit_batch(
            batch_payload: QueryBatchSpec | List[QueryTrainingExample],
            *,
            worker_exc: Exception | None = None,
        ) -> None:
            worker_error_payloads = (
                _batch_prepare_worker_error_payloads(
                    batch_payload,
                    worker_exc,
                    graph_error_stage=graph_error_stage,
                )
                if worker_exc is not None
                else []
            )
            if self._direct_fallback_mode:
                pending_items.append((batch_payload, None, list(worker_error_payloads), "local"))
                return
            if not self._thread_fallback_mode:
                executor = self._ensure_executor()
                if executor is not None:
                    try:
                        future = executor.submit(_prepare_batch_worker_task, batch_payload, resolved_graph_error_stage)
                    except Exception as exc:
                        recovery_payload = self._recover_process_pool(exc)
                        _emit_worker_event(
                            "batch_prepare_process_pool_recovery",
                            error_type=type(exc).__name__,
                            error_message=clean_text(str(exc)),
                            backend="process_submit",
                            **dict(recovery_payload),
                        )
                        worker_error_payloads.extend(
                            _batch_prepare_worker_error_payloads(
                                batch_payload,
                                exc,
                                graph_error_stage=graph_error_stage,
                            )
                        )
                    else:
                        pending_items.append((batch_payload, future, list(worker_error_payloads), "process"))
                        return
            thread_executor = self._ensure_thread_executor()
            if thread_executor is not None:
                try:
                    future = thread_executor.submit(
                        _prepare_batch_thread_task,
                        batch_payload,
                        resolved_graph_error_stage,
                        resolved_worker_context,
                    )
                except Exception as exc:
                    self._enable_direct_fallback(exc)
                    worker_error_payloads.extend(
                        _batch_prepare_worker_error_payloads(
                            batch_payload,
                            exc,
                            graph_error_stage=graph_error_stage,
                        )
                    )
                else:
                    pending_items.append((batch_payload, future, list(worker_error_payloads), "thread"))
                    return
            pending_items.append((batch_payload, None, list(worker_error_payloads), "local"))

        def _retry_pending_with_fresh_executor() -> None:
            if not pending_items:
                return
            retry_batches = [batch_payload for batch_payload, _, _, _ in pending_items]
            for _, future, _, _ in pending_items:
                if future is not None:
                    try:
                        future.cancel()
                    except Exception:
                        pass
            pending_items.clear()
            for batch_payload in retry_batches:
                _submit_batch(batch_payload)

        def _submit_pending() -> None:
            while len(pending_items) < _current_pending_target():
                try:
                    raw_batch = next(batch_iter)
                except StopIteration:
                    break
                batch_payload = raw_batch if isinstance(raw_batch, QueryBatchSpec) else list(raw_batch)
                _submit_batch(batch_payload)

        def _pop_pending_item_at(
            index: int,
        ) -> tuple[QueryBatchSpec | List[QueryTrainingExample], Future[Dict[str, Any]] | None, List[Dict[str, Any]], str]:
            pending_items.rotate(-int(index))
            item = pending_items.popleft()
            pending_items.rotate(int(index))
            return item

        def _pop_next_pending_item(
        ) -> tuple[QueryBatchSpec | List[QueryTrainingExample], Future[Dict[str, Any]] | None, List[Dict[str, Any]], str]:
            if not use_completion_order or len(pending_items) <= 1:
                return pending_items.popleft()
            for index, item in enumerate(pending_items):
                if item[1] is None:
                    return _pop_pending_item_at(index)
            for index, item in enumerate(pending_items):
                future = item[1]
                future_done = getattr(future, "done", None)
                if future is not None and callable(future_done) and future_done():
                    return _pop_pending_item_at(index)
            pending_futures = []
            for _, future, _, _ in pending_items:
                future_done = getattr(future, "done", None)
                if future is None or not callable(future_done):
                    return pending_items.popleft()
                pending_futures.append(future)
            if pending_futures:
                done_futures, _ = wait(pending_futures, return_when=FIRST_COMPLETED)
                if done_futures:
                    for index, item in enumerate(pending_items):
                        future = item[1]
                        if future is not None and future in done_futures:
                            return _pop_pending_item_at(index)
            return pending_items.popleft()

        _submit_pending()
        while pending_items:
            batch_payload, future, queued_error_payloads, backend = _pop_next_pending_item()
            prepared_batch: Dict[str, Any]
            if future is None:
                prepared_batch = _prepare_batch_payload_from_graph_cache(
                    batch_payload,
                    graph_cache,
                    merge_device=torch.device("cpu"),
                    graph_error_stage=graph_error_stage,
                )
                if queued_error_payloads:
                    prepared_batch["graph_error_payloads"] = list(queued_error_payloads) + list(prepared_batch.get("graph_error_payloads", []) or [])
                    prepared_batch["graph_error_count"] = len(prepared_batch["graph_error_payloads"])
            else:
                try:
                    prepared_batch = dict(future.result())
                    self._record_worker_success(backend=backend)
                    if backend == "process" and self._process_success_count == 1:
                        _emit_worker_event(
                            "batch_prepare_process_pool_ready",
                            backend="process",
                            worker_count=int(self._process_worker_count()),
                        )
                    if queued_error_payloads:
                        prepared_batch["graph_error_payloads"] = list(queued_error_payloads) + list(prepared_batch.get("graph_error_payloads", []) or [])
                        prepared_batch["graph_error_count"] = len(prepared_batch["graph_error_payloads"])
                except (BrokenProcessPool, Exception) as exc:
                    if backend == "process":
                        recovery_payload = self._recover_process_pool(exc)
                        _emit_worker_event(
                            "batch_prepare_process_pool_recovery",
                            error_type=type(exc).__name__,
                            error_message=clean_text(str(exc)),
                            backend="process_result",
                            **dict(recovery_payload),
                        )
                        _retry_pending_with_fresh_executor()
                    elif backend == "thread":
                        self._worker_exc = exc
                        _emit_worker_event(
                            "batch_prepare_thread_fallback_error",
                            error_type=type(exc).__name__,
                            error_message=clean_text(str(exc)),
                            backend="thread_result",
                        )
                    prepared_batch = _prepare_batch_payload_from_graph_cache(
                        batch_payload,
                        graph_cache,
                        merge_device=torch.device("cpu"),
                        graph_error_stage=graph_error_stage,
                    )
                    worker_error_payloads = _batch_prepare_worker_error_payloads(
                        batch_payload,
                        exc,
                        graph_error_stage=graph_error_stage,
                    )
                    prepared_batch["graph_error_payloads"] = list(queued_error_payloads) + worker_error_payloads + list(prepared_batch.get("graph_error_payloads", []) or [])
                    prepared_batch["graph_error_count"] = len(prepared_batch["graph_error_payloads"])
            yield prepared_batch
            _submit_pending()


def _build_graph_error_payload(
    example: QueryTrainingExample,
    exc: Exception,
    *,
    stage: str,
    batch_conversation_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    payload = {
        "stage": clean_text(stage) or "score_batch",
        "conversation_id": clean_text(example.conversation_id),
        "question_id": clean_text(example.question_id),
        "question": clean_text(example.question),
        "error_type": type(exc).__name__,
        "error_message": clean_text(str(exc)),
    }
    if batch_conversation_ids:
        payload["batch_conversation_ids"] = [clean_text(item) for item in list(batch_conversation_ids) if clean_text(item)]
    return payload


def _emit_graph_error_payloads(
    graph_error_payloads: Sequence[Mapping[str, Any]],
    *,
    graph_error_callback: Callable[[Dict[str, Any]], None] | None,
) -> None:
    if graph_error_callback is None:
        return
    for payload in graph_error_payloads:
        graph_error_callback(dict(payload))


def _empty_score_batch_result(*, device: torch.device, graph_error_count: int) -> Dict[str, Any]:
    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    return {
        "loss": zero,
        "metrics": {
            "recall_loss": 0.0,
            "loss": 0.0,
            "event_loss": 0.0,
            "path_loss": 0.0,
            "temporal_loss": 0.0,
            "answer_type_loss": 0.0,
            "answer_plan_loss": 0.0,
            "token_role_loss": 0.0,
            "question_understanding_loss": 0.0,
            "memory_router_loss": 0.0,
            "memory_router_exact_match": 0.0,
            "memory_router_f1": 0.0,
            "event_distractor_loss": 0.0,
            "event_tunnel_loss": 0.0,
            "path_tunnel_loss": 0.0,
            "path_tunnel_delta_loss": 0.0,
            "event_tunnel_selection_loss": 0.0,
            "path_tunnel_selection_loss": 0.0,
            "event_hard_negative_loss": 0.0,
            "path_hard_negative_loss": 0.0,
            "recall_selection_loss": 0.0,
            "event_selection_loss": 0.0,
            "path_selection_loss": 0.0,
            "final_event_set_loss": 0.0,
            "event_matrix_delta_loss": 0.0,
            "path_matrix_delta_loss": 0.0,
            "answer_refusal_loss": 0.0,
            "recall_event_recall_at_24": 0.0,
            "event_recall_at_1": 0.0,
            "event_recall_at_5": 0.0,
            "path_recall_at_3": 0.0,
            "path_tunnel_support_recall_at_3": 0.0,
            "path_tunnel_support_positive_coverage_at_3": 0.0,
            "path_tunnel_delta_recall_at_3": 0.0,
            "path_tunnel_delta_positive_coverage_at_3": 0.0,
            "answer_plan_selected_recall_at_5": 0.0,
            "answer_plan_selected_positive_coverage_at_5": 0.0,
            "answer_plan_current_top1_accuracy": 0.0,
            "path_tunnel_rescue025_recall_at_3": 0.0,
            "path_tunnel_rescue050_recall_at_3": 0.0,
            "path_tunnel_rescue100_recall_at_3": 0.0,
            "temporal_accuracy": 0.0,
            "training_weight_mean": 1.0,
            "training_weight_min": 1.0,
            "training_weight_max": 1.0,
            "samples": 0,
            "loss_group_count": 0,
            "answer_type_metrics": {},
            "supervision_bucket_metrics": {},
            "recall_loss_count": 0,
            "event_loss_count": 0,
            "path_loss_count": 0,
            "temporal_loss_count": 0,
            "answer_type_loss_count": 0,
            "answer_plan_loss_count": 0,
            "token_role_loss_count": 0,
            "question_understanding_loss_count": 0,
            "memory_router_loss_count": 0,
            "memory_router_total": 0,
            "event_distractor_loss_count": 0,
            "event_tunnel_loss_count": 0,
            "path_tunnel_loss_count": 0,
            "path_tunnel_delta_loss_count": 0,
            "event_tunnel_selection_loss_count": 0,
            "path_tunnel_selection_loss_count": 0,
            "event_hard_negative_loss_count": 0,
            "path_hard_negative_loss_count": 0,
            "recall_selection_loss_count": 0,
            "event_selection_loss_count": 0,
            "path_selection_loss_count": 0,
            "final_event_set_loss_count": 0,
            "event_matrix_delta_loss_count": 0,
            "path_matrix_delta_loss_count": 0,
            "answer_refusal_loss_count": 0,
            "recall_event_recall_total": 0,
            "event_recall_total": 0,
            "path_recall_total": 0,
            "path_tunnel_support_recall_total": 0,
            "path_tunnel_support_positive_total": 0,
            "path_tunnel_delta_recall_total": 0,
            "path_tunnel_delta_positive_total": 0,
            "answer_plan_selected_total": 0,
            "answer_plan_selected_positive_total": 0,
            "answer_plan_current_total": 0,
            "path_tunnel_rescue025_recall_total": 0,
            "path_tunnel_rescue050_recall_total": 0,
            "path_tunnel_rescue100_recall_total": 0,
            "temporal_total": 0,
            "graph_error_count": int(graph_error_count),
        },
    }


def _prepare_batch_payload(
    batch: Sequence[QueryTrainingExample],
    *,
    graph_item_resolver: Callable[[str], GraphCacheItem | None],
    merge_device: torch.device,
    graph_error_stage: str,
) -> Dict[str, Any]:
    ordered_graph_tensors: OrderedDict[str, Mapping[str, Any]] = OrderedDict()
    scored_examples: List[QueryTrainingExample] = []
    prepared_supervision: List[Dict[str, Any]] = []
    graph_error_payloads: List[Dict[str, Any]] = []
    for example in batch:
        conversation_id = clean_text(example.conversation_id)
        try:
            graph_item = graph_item_resolver(conversation_id)
        except Exception as exc:
            graph_error_payloads.append(_build_graph_error_payload(example, exc, stage=graph_error_stage))
            continue
        if graph_item is None:
            continue
        if conversation_id not in ordered_graph_tensors:
            graph_tensors = graph_item.tensors
            if isinstance(graph_tensors.get("node_type_ids"), Tensor) and graph_tensors["node_type_ids"].device != merge_device:
                graph_tensors = _graph_tensor_mapping_to_device(
                    graph_tensors,
                    merge_device,
                    non_blocking=merge_device.type == "cuda",
                )
            ordered_graph_tensors[conversation_id] = graph_tensors
        scored_examples.append(example)
        prepared_supervision.append(_build_example_supervision_payload(example))

    merged_graph_tensors: Dict[str, Any] | None = None
    if scored_examples:
        try:
            if len(ordered_graph_tensors) == 1:
                merged_graph_tensors = dict(next(iter(ordered_graph_tensors.values())))
            else:
                merged_graph_tensors = _combine_graph_tensors_by_conversation(ordered_graph_tensors, device=merge_device)
        except Exception as exc:
            batch_conversation_ids = list(ordered_graph_tensors)
            graph_error_payloads.extend(
                _build_graph_error_payload(
                    example,
                    exc,
                    stage=graph_error_stage,
                    batch_conversation_ids=batch_conversation_ids,
                )
                for example in scored_examples
            )
            scored_examples = []
            merged_graph_tensors = None
    return {
        "scored_examples": list(scored_examples),
        "prepared_supervision": list(prepared_supervision),
        "merged_graph_tensors": merged_graph_tensors,
        "graph_error_payloads": graph_error_payloads,
        "graph_error_count": len(graph_error_payloads),
        "prepared_conversation_ids": [clean_text(item) for item in list(ordered_graph_tensors) if clean_text(item)],
    }


def _prepare_batch_payload_from_graph_cache(
    batch: QueryBatchSpec | Sequence[QueryTrainingExample],
    graph_cache: Mapping[str, GraphCacheItem],
    *,
    merge_device: torch.device,
    graph_error_stage: str,
) -> Dict[str, Any]:
    materialized_batch = _materialize_query_batch_payload(batch)
    return _prepare_batch_payload(
        materialized_batch,
        graph_item_resolver=lambda conversation_id: graph_cache.get(conversation_id),
        merge_device=merge_device,
        graph_error_stage=graph_error_stage,
    )


def _normalize_batch_prepare_worker_context(worker_context: Mapping[str, Any] | None) -> Dict[str, Any]:
    resolved_context = dict(worker_context or {})
    return {
        "conversation_specs": {
            clean_text(conversation_id): dict(payload or {})
            for conversation_id, payload in dict(resolved_context.get("conversation_specs", {}) or {}).items()
            if clean_text(conversation_id)
        },
        "graph_dir": clean_text(resolved_context.get("graph_dir", "")),
        "cache_dir": clean_text(resolved_context.get("cache_dir", "")),
        "cache_write_enabled": bool(resolved_context.get("cache_write_enabled", True)),
        "require_cache_hit": bool(resolved_context.get("require_cache_hit", False)),
        "worker_memory_cache_size": max(0, int(resolved_context.get("worker_memory_cache_size", 0) or 0)),
    }


def _load_graph_item_from_worker_context(
    conversation_id: str,
    *,
    worker_context: Mapping[str, Any],
    worker_cache: OrderedDict[str, GraphCacheItem],
    worker_memory_cache_size: int,
) -> GraphCacheItem | None:
    normalized_conversation_id = clean_text(conversation_id)
    if not normalized_conversation_id:
        return None
    cached_item = worker_cache.get(normalized_conversation_id)
    if cached_item is not None:
        worker_cache.move_to_end(normalized_conversation_id)
        return cached_item
    resolved_context = dict(worker_context or {})
    spec = dict(resolved_context.get("conversation_specs", {}) or {}).get(normalized_conversation_id)
    graph_path_raw = clean_text((spec or {}).get("graph_path", ""))
    cache_path_raw = clean_text((spec or {}).get("cache_path", ""))
    if not graph_path_raw:
        graph_dir_raw = clean_text(resolved_context.get("graph_dir", ""))
        if not graph_dir_raw:
            return None
        graph_path_raw = str(Path(graph_dir_raw) / f"{normalized_conversation_id}.json")
    graph_path = Path(graph_path_raw)
    if not cache_path_raw:
        cache_dir_raw = clean_text(resolved_context.get("cache_dir", ""))
        if cache_dir_raw:
            cache_path_raw = str(_graph_tensor_cache_path(Path(cache_dir_raw), graph_path))
    cache_path = Path(cache_path_raw) if cache_path_raw else None
    source_signature = _graph_source_signature(graph_path)
    payload = _load_graph_tensor_cache(cache_path, source_signature=source_signature) if cache_path is not None else None
    if payload is not None:
        graph = dict(payload.get("graph", {}) or {})
        tensors = _graph_tensor_mapping_to_device(payload.get("tensors", {}), torch.device("cpu"))
    else:
        if cache_path is not None and bool((spec or {}).get("require_cache_hit", resolved_context.get("require_cache_hit", False))):
            raise FileNotFoundError(f"Prebuilt graph tensor cache missing for conversation '{normalized_conversation_id}': {graph_path}")
        graph = dict(read_json(graph_path))
        cache_tensors = tensorize_graph(graph, device=torch.device("cpu"))
        if cache_path is not None and bool((spec or {}).get("cache_write_enabled", resolved_context.get("cache_write_enabled", True))):
            _write_graph_tensor_cache(
                cache_path,
                source_signature=source_signature,
                graph=graph,
                tensors=cache_tensors,
            )
        tensors = _graph_tensor_mapping_to_device(cache_tensors, torch.device("cpu"))
    tensors = dict(_ensure_graph_scoring_feature_cache(tensors))
    item = GraphCacheItem(graph=graph, tensors=tensors)
    if worker_memory_cache_size > 0:
        worker_cache[normalized_conversation_id] = item
        worker_cache.move_to_end(normalized_conversation_id)
        while len(worker_cache) > worker_memory_cache_size:
            worker_cache.popitem(last=False)
    return item


def _initialize_batch_prepare_worker(worker_context: Mapping[str, Any]) -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    global _BATCH_PREPARE_WORKER_CONTEXT, _BATCH_PREPARE_WORKER_CACHE, _BATCH_PREPARE_WORKER_MEMORY_CACHE_SIZE
    normalized_context = _normalize_batch_prepare_worker_context(worker_context)
    _BATCH_PREPARE_WORKER_CONTEXT = dict(normalized_context)
    _BATCH_PREPARE_WORKER_CACHE = OrderedDict()
    _BATCH_PREPARE_WORKER_MEMORY_CACHE_SIZE = max(0, int(normalized_context.get("worker_memory_cache_size", 0) or 0))


def _load_graph_item_from_batch_prepare_worker(conversation_id: str) -> GraphCacheItem | None:
    return _load_graph_item_from_worker_context(
        conversation_id,
        worker_context=_BATCH_PREPARE_WORKER_CONTEXT,
        worker_cache=_BATCH_PREPARE_WORKER_CACHE,
        worker_memory_cache_size=int(_BATCH_PREPARE_WORKER_MEMORY_CACHE_SIZE),
    )


def _thread_batch_prepare_state(worker_context: Mapping[str, Any]) -> Dict[str, Any]:
    normalized_context = _normalize_batch_prepare_worker_context(worker_context)
    context_key = (
        clean_text(normalized_context.get("graph_dir", "")),
        clean_text(normalized_context.get("cache_dir", "")),
        bool(normalized_context.get("cache_write_enabled", True)),
        bool(normalized_context.get("require_cache_hit", False)),
        int(normalized_context.get("worker_memory_cache_size", 0) or 0),
        len(dict(normalized_context.get("conversation_specs", {}) or {})),
    )
    state = getattr(_BATCH_PREPARE_THREAD_STATE, "state", None)
    if not isinstance(state, Mapping) or tuple(state.get("context_key", ())) != context_key:
        state = {
            "context_key": context_key,
            "context": dict(normalized_context),
            "cache": OrderedDict(),
            "worker_memory_cache_size": int(normalized_context.get("worker_memory_cache_size", 0) or 0),
        }
        _BATCH_PREPARE_THREAD_STATE.state = state
    return dict(state)


def _prepare_batch_thread_task(
    batch: QueryBatchSpec | Sequence[QueryTrainingExample],
    graph_error_stage: str,
    worker_context: Mapping[str, Any],
) -> Dict[str, Any]:
    state = _thread_batch_prepare_state(worker_context)
    thread_context = dict(state.get("context", {}) or {})
    thread_cache = state.get("cache")
    if not isinstance(thread_cache, OrderedDict):
        thread_cache = OrderedDict()
        _BATCH_PREPARE_THREAD_STATE.state = {
            **dict(state),
            "cache": thread_cache,
        }
    return _prepare_batch_payload(
        _materialize_query_batch_payload(batch),
        graph_item_resolver=lambda conversation_id: _load_graph_item_from_worker_context(
            conversation_id,
            worker_context=thread_context,
            worker_cache=thread_cache,
            worker_memory_cache_size=int(state.get("worker_memory_cache_size", 0) or 0),
        ),
        merge_device=torch.device("cpu"),
        graph_error_stage=graph_error_stage,
    )


def _prepare_batch_worker_task(
    batch: QueryBatchSpec | Sequence[QueryTrainingExample],
    graph_error_stage: str,
) -> Dict[str, Any]:
    return _prepare_batch_payload(
        _materialize_query_batch_payload(batch),
        graph_item_resolver=_load_graph_item_from_batch_prepare_worker,
        merge_device=torch.device("cpu"),
        graph_error_stage=graph_error_stage,
    )


def _batch_prepare_worker_error_payloads(
    batch: QueryBatchSpec | Sequence[QueryTrainingExample],
    exc: Exception,
    *,
    graph_error_stage: str,
) -> List[Dict[str, Any]]:
    worker_stage = f"{clean_text(graph_error_stage) or 'prepare_batch'}_worker"
    materialized_batch = _materialize_query_batch_payload(batch)
    return [
        _build_graph_error_payload(example, exc, stage=worker_stage)
        for example in materialized_batch
    ]


def _iter_prepared_batches_with_workers(
    rows: Sequence[QueryTrainingExample],
    *,
    graph_cache: Mapping[str, GraphCacheItem],
    batch_size: int,
    shuffle: bool,
    seed: int,
    worker_context: Mapping[str, Any] | None,
    worker_count: int,
    lookahead_batches: int,
    graph_error_stage: str,
    skip_batches: int = 0,
    worker_pool: _PreparedBatchWorkerPool | None = None,
    prefer_completion_order: bool = DEFAULT_BATCH_PREPARE_PREFER_COMPLETION_ORDER,
    worker_event_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> Iterator[Dict[str, Any]]:
    resolved_pool = worker_pool
    created_pool = False
    if resolved_pool is None:
        resolved_pool = _PreparedBatchWorkerPool(
            worker_context=worker_context,
            worker_count=worker_count,
        )
        created_pool = True
    try:
        yield from resolved_pool.iter_prepared_batches(
            rows,
            graph_cache=graph_cache,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            lookahead_batches=lookahead_batches,
            graph_error_stage=graph_error_stage,
            skip_batches=skip_batches,
            prefer_completion_order=prefer_completion_order,
            worker_event_callback=worker_event_callback,
        )
    finally:
        if created_pool and resolved_pool is not None:
            resolved_pool.close()


def score_prepared_batch(
    model: LocomoNodeMemoryModel,
    prepared_batch: Mapping[str, Any],
    *,
    device: torch.device,
    loss_config: Mapping[str, Any] | None = None,
    score_kwargs: Mapping[str, Any] | None = None,
    apply_example_weights: bool = False,
    graph_error_callback: Callable[[Dict[str, Any]], None] | None = None,
    graph_error_stage: str = "",
) -> Dict[str, Any]:
    resolved_loss_config = {**DEFAULT_TRAINING_CONFIG, **dict(loss_config or {})}
    losses: List[Tensor] = []
    loss_entries: List[tuple[Tensor, str]] = []
    recall_loss_values: List[float] = []
    event_loss_values: List[float] = []
    path_loss_values: List[float] = []
    temporal_loss_values: List[float] = []
    answer_type_loss_values: List[float] = []
    answer_plan_loss_values: List[float] = []
    token_role_loss_values: List[float] = []
    question_understanding_loss_values: List[float] = []
    memory_router_loss_values: List[float] = []
    event_distractor_loss_values: List[float] = []
    event_tunnel_loss_values: List[float] = []
    path_tunnel_loss_values: List[float] = []
    path_tunnel_delta_loss_values: List[float] = []
    event_tunnel_selection_loss_values: List[float] = []
    path_tunnel_selection_loss_values: List[float] = []
    event_hard_negative_loss_values: List[float] = []
    path_hard_negative_loss_values: List[float] = []
    recall_selection_loss_values: List[float] = []
    event_selection_loss_values: List[float] = []
    path_selection_loss_values: List[float] = []
    final_event_set_loss_values: List[float] = []
    event_matrix_delta_loss_values: List[float] = []
    path_matrix_delta_loss_values: List[float] = []
    answer_refusal_loss_values: List[float] = []
    training_weight_values: List[float] = []
    answer_type_metric_stats: Dict[str, Dict[str, int]] = {}
    supervision_bucket_metric_stats: Dict[str, Dict[str, int]] = {}
    recall_event24_hits = 0
    recall_event24_positive_hits = 0
    recall_event24_positive_total = 0
    recall_event_total = 0
    event_recall1_hits = 0
    event_recall5_hits = 0
    event5_positive_hits = 0
    event5_positive_total = 0
    path_recall3_hits = 0
    path3_positive_hits = 0
    path3_positive_total = 0
    path_tunnel_support_recall3_hits = 0
    path_tunnel_support_positive_hits = 0
    path_tunnel_support_positive_total = 0
    path_tunnel_support_recall_total = 0
    path_tunnel_delta_recall3_hits = 0
    path_tunnel_delta_positive_hits = 0
    path_tunnel_delta_positive_total = 0
    path_tunnel_delta_recall_total = 0
    answer_plan_selected_recall5_hits = 0
    answer_plan_selected_total = 0
    answer_plan_selected_positive_hits = 0
    answer_plan_selected_positive_total = 0
    answer_plan_current_top1_hits = 0
    answer_plan_current_total = 0
    path_tunnel_rescue025_recall3_hits = 0
    path_tunnel_rescue025_recall_total = 0
    path_tunnel_rescue050_recall3_hits = 0
    path_tunnel_rescue050_recall_total = 0
    path_tunnel_rescue100_recall3_hits = 0
    path_tunnel_rescue100_recall_total = 0
    temporal_hits = 0
    memory_router_exact_hits = 0
    memory_router_f1_sum = 0.0
    memory_router_total = 0
    recall_loss_count = 0
    event_loss_count = 0
    path_loss_count = 0
    temporal_loss_count = 0
    answer_type_loss_count = 0
    answer_plan_loss_count = 0
    token_role_loss_count = 0
    question_understanding_loss_count = 0
    memory_router_loss_count = 0
    event_distractor_loss_count = 0
    event_tunnel_loss_count = 0
    path_tunnel_loss_count = 0
    path_tunnel_delta_loss_count = 0
    event_tunnel_selection_loss_count = 0
    path_tunnel_selection_loss_count = 0
    event_hard_negative_loss_count = 0
    path_hard_negative_loss_count = 0
    recall_selection_loss_count = 0
    event_selection_loss_count = 0
    path_selection_loss_count = 0
    final_event_set_loss_count = 0
    event_matrix_delta_loss_count = 0
    path_matrix_delta_loss_count = 0
    answer_refusal_loss_count = 0
    graph_error_payloads = [dict(payload) for payload in list(prepared_batch.get("graph_error_payloads", []) or [])]
    _emit_graph_error_payloads(graph_error_payloads, graph_error_callback=graph_error_callback)
    graph_error_count = len(graph_error_payloads)
    event_recall_total = 0
    path_recall_total = 0
    temporal_total = 0
    scored_examples = list(prepared_batch.get("scored_examples", []) or [])
    prepared_supervision = list(prepared_batch.get("prepared_supervision", []) or [])
    merged_graph_tensors = prepared_batch.get("merged_graph_tensors")

    if not scored_examples or not isinstance(merged_graph_tensors, Mapping):
        return _empty_score_batch_result(device=device, graph_error_count=graph_error_count)

    prepared_conversation_ids = [
        clean_text(item) for item in list(prepared_batch.get("prepared_conversation_ids", []) or []) if clean_text(item)
    ]
    working_graph_tensors = dict(merged_graph_tensors)
    if isinstance(working_graph_tensors.get("node_type_ids"), Tensor) and working_graph_tensors["node_type_ids"].device != device:
        working_graph_tensors = _graph_tensor_mapping_to_device(
            working_graph_tensors,
            device,
            non_blocking=device.type == "cuda",
        )

    with _gc_suspended():
        try:
            outputs_batch = _call_with_supported_kwargs(
                model.score_examples,
                working_graph_tensors,
                scored_examples,
                **dict(score_kwargs or {}),
            )
        except Exception as exc:
            graph_error_count += len(scored_examples)
            if graph_error_callback is not None:
                for example in scored_examples:
                    graph_error_callback(
                        _build_graph_error_payload(
                            example,
                            exc,
                            stage=clean_text(graph_error_stage) or "score_prepared_batch",
                            batch_conversation_ids=prepared_conversation_ids,
                        )
                    )
            return _empty_score_batch_result(device=device, graph_error_count=graph_error_count)

        for example_index, (example, outputs) in enumerate(zip(scored_examples, outputs_batch)):
            supervision = (
                prepared_supervision[example_index]
                if example_index < len(prepared_supervision) and isinstance(prepared_supervision[example_index], Mapping)
                else _build_example_supervision_payload(example)
            )
            example_training_weight = max(0.0, float(supervision.get("training_example_weight", 1.0) or 1.0))
            training_weight_values.append(example_training_weight)
            answer_type_name = _training_answer_type_name(example)
            answer_stats = answer_type_metric_stats.setdefault(answer_type_name, _new_answer_type_metric_stats())
            answer_stats["samples"] += 1
            supervision_bucket_name = clean_text(supervision.get("supervision_bucket", "")) or _training_supervision_bucket_name(example)
            supervision_stats = supervision_bucket_metric_stats.setdefault(
                supervision_bucket_name,
                _new_answer_type_metric_stats(),
            )
            supervision_stats["samples"] += 1
            example_recall_total = 0
            example_recall24_hit = 0
            example_recall24_positive_hits = 0
            example_recall24_positive_total = 0
            example_event_total = 0
            example_event1_hit = 0
            example_event5_hit = 0
            example_event5_positive_hits = 0
            example_event5_positive_total = 0
            example_path_total = 0
            example_path3_hit = 0
            example_path3_positive_hits = 0
            example_path3_positive_total = 0
            example_temporal_total = 0
            example_temporal_hit = 0
            example_answer_plan_total = 0
            example_answer_plan_recall5_hit = 0
            example_answer_plan_positive_hits = 0
            example_answer_plan_positive_total = 0
            example_answer_plan_current_total = 0
            example_answer_plan_current_top1_hit = 0
            total_loss = outputs["answer_type_logits"].new_zeros(())
            answer_type_logits = outputs["answer_type_logits"].unsqueeze(0)
            answer_type_target = torch.tensor(
                [int(supervision.get("answer_type_index", _answer_type_label(example)))],
                dtype=torch.long,
                device=device,
            )
            question_understanding_targets = _question_understanding_targets(example.question_features or {})
            question_semantic_logits = outputs.get("question_semantic_logits")
            question_status_logits = outputs.get("question_status_logits")
            question_time_granularity_logits = outputs.get("question_time_granularity_logits")
            question_temporal_logit = outputs.get("question_temporal_logit")
            if all(
                isinstance(value, Tensor)
                for value in (
                    question_semantic_logits,
                    question_status_logits,
                    question_time_granularity_logits,
                    question_temporal_logit,
                )
            ):
                question_semantic_loss = F.cross_entropy(
                    question_semantic_logits.unsqueeze(0),
                    torch.tensor(
                        [int(question_understanding_targets["semantic_slot_index"])],
                        dtype=torch.long,
                        device=device,
                    ),
                )
                question_status_loss = F.cross_entropy(
                    question_status_logits.unsqueeze(0),
                    torch.tensor(
                        [int(question_understanding_targets["target_status_index"])],
                        dtype=torch.long,
                        device=device,
                    ),
                )
                question_time_granularity_loss = F.cross_entropy(
                    question_time_granularity_logits.unsqueeze(0),
                    torch.tensor(
                        [int(question_understanding_targets["time_granularity_index"])],
                        dtype=torch.long,
                        device=device,
                    ),
                )
                question_temporal_loss = F.binary_cross_entropy_with_logits(
                    question_temporal_logit.reshape(1),
                    torch.tensor(
                        [float(question_understanding_targets["is_temporal_value"])],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                question_understanding_loss = torch.stack(
                    [
                        question_semantic_loss,
                        question_status_loss,
                        question_time_granularity_loss,
                        question_temporal_loss,
                    ]
                ).mean()
                total_loss = (
                    total_loss
                    + float(resolved_loss_config["question_understanding_loss_weight"]) * question_understanding_loss
                )
                question_understanding_loss_values.append(float(question_understanding_loss.detach().cpu()))
                question_understanding_loss_count += 1
            answer_type_loss = F.cross_entropy(answer_type_logits, answer_type_target)
            total_loss = total_loss + float(resolved_loss_config["answer_type_loss_weight"]) * answer_type_loss
            answer_type_loss_values.append(float(answer_type_loss.detach().cpu()))
            answer_type_loss_count += 1
            memory_router_logits = outputs.get("memory_router_logits")
            if isinstance(memory_router_logits, Tensor) and memory_router_logits.numel() == len(MEMORY_ROUTER_LAYERS):
                memory_router_targets = _memory_router_target_tensor(example, supervision, device=device)
                memory_router_loss = F.binary_cross_entropy_with_logits(
                    memory_router_logits.reshape(-1),
                    memory_router_targets,
                )
                total_loss = total_loss + float(resolved_loss_config["memory_router_loss_weight"]) * memory_router_loss
                memory_router_loss_values.append(float(memory_router_loss.detach().cpu()))
                memory_router_loss_count += 1
                router_probs = torch.sigmoid(memory_router_logits.detach().reshape(-1))
                router_pred = router_probs >= 0.5
                router_target = memory_router_targets.detach() >= 0.5
                if bool(torch.equal(router_pred, router_target)):
                    memory_router_exact_hits += 1
                tp = int((router_pred & router_target).sum().item())
                fp = int((router_pred & ~router_target).sum().item())
                fn = int((~router_pred & router_target).sum().item())
                memory_router_f1_sum += (2.0 * tp) / max(1.0, float(2 * tp + fp + fn))
                memory_router_total += 1
            if example_index == 0:
                question_token_role_aux_loss = outputs.get("question_token_role_aux_loss")
                node_token_role_aux_loss = outputs.get("node_token_role_aux_loss")
                token_role_aux_losses = [
                    loss
                    for loss in (question_token_role_aux_loss, node_token_role_aux_loss)
                    if isinstance(loss, Tensor) and loss.numel() > 0
                ]
                if token_role_aux_losses:
                    token_role_loss = torch.stack([loss.float() for loss in token_role_aux_losses]).mean()
                    total_loss = total_loss + float(resolved_loss_config["token_role_loss_weight"]) * token_role_loss
                    token_role_loss_values.append(float(token_role_loss.detach().cpu()))
                    token_role_loss_count += 1
            answer_refusal_loss = _answer_refusal_margin_loss(
                outputs["answer_type_logits"],
                target_index=int(answer_type_target.item()),
                has_positive_supervision=bool(supervision.get("has_positive_supervision", False)),
                margin=float(resolved_loss_config["answer_refusal_margin"]),
            )
            total_loss = total_loss + float(resolved_loss_config["answer_refusal_loss_weight"]) * answer_refusal_loss
            answer_refusal_loss_values.append(float(answer_refusal_loss.detach().cpu()))
            answer_refusal_loss_count += 1

            answer_plan_logits = outputs.get("answer_plan_logits")
            if (
                bool(supervision.get("has_answer_plan_supervision", False))
                and isinstance(answer_plan_logits, Tensor)
                and answer_plan_logits.numel() > 0
            ):
                candidate_ids_for_plan = outputs["candidate_event_ids"]
                selected_plan_ids = supervision.get("answer_plan_selected_event_ids", frozenset())
                current_plan_ids = supervision.get("answer_plan_current_event_ids", frozenset())
                historical_plan_ids = supervision.get("answer_plan_historical_event_ids", frozenset())
                suppressed_plan_ids = supervision.get("answer_plan_suppressed_event_ids", frozenset())
                role_losses: List[Tensor] = []
                for role_name, supervision_key in (
                    ("selected", "answer_plan_selected_event_ids"),
                    ("current", "answer_plan_current_event_ids"),
                    ("historical", "answer_plan_historical_event_ids"),
                    ("suppressed", "answer_plan_suppressed_event_ids"),
                ):
                    role_ids = supervision.get(supervision_key, frozenset())
                    if not role_ids:
                        continue
                    role_index = ANSWER_PLAN_OUTPUT_TO_ID[role_name]
                    role_losses.append(
                        _binary_plan_loss_for_ids(
                            answer_plan_logits[:, role_index],
                            candidate_ids=candidate_ids_for_plan,
                            positive_ids=role_ids,
                            device=device,
                        )
                    )
                current_old_weight = float(resolved_loss_config.get("answer_plan_current_old_margin_loss_weight", 0.0) or 0.0)
                if current_old_weight > 0.0 and current_plan_ids:
                    old_or_suppressed_ids = _normalized_id_frozenset(historical_plan_ids) | _normalized_id_frozenset(suppressed_plan_ids)
                    current_margin_loss = _positive_vs_negative_ids_margin_loss(
                        answer_plan_logits[:, ANSWER_PLAN_OUTPUT_TO_ID["current"]],
                        candidate_ids=candidate_ids_for_plan,
                        positive_ids=current_plan_ids,
                        negative_ids=old_or_suppressed_ids,
                        margin=float(resolved_loss_config.get("answer_plan_current_old_margin", 0.2) or 0.2),
                    )
                    total_loss = total_loss + current_old_weight * current_margin_loss
                    answer_plan_loss_values.append(float(current_margin_loss.detach().cpu()))
                selected_negative_weight = float(
                    resolved_loss_config.get("answer_plan_selected_negative_margin_loss_weight", 0.0) or 0.0
                )
                if selected_negative_weight > 0.0 and selected_plan_ids:
                    selected_negative_ids = (
                        _normalized_id_frozenset(suppressed_plan_ids)
                        | _normalized_id_frozenset(supervision.get("hard_negative_event_ids", frozenset()))
                        | _normalized_id_frozenset(supervision.get("negative_event_ids", frozenset()))
                    )
                    selected_negative_margin_loss = _positive_vs_negative_ids_margin_loss(
                        answer_plan_logits[:, ANSWER_PLAN_OUTPUT_TO_ID["selected"]],
                        candidate_ids=candidate_ids_for_plan,
                        positive_ids=selected_plan_ids,
                        negative_ids=selected_negative_ids,
                        margin=float(
                            resolved_loss_config.get("answer_plan_selected_negative_margin", 0.3) or 0.3
                        ),
                    )
                    total_loss = total_loss + selected_negative_weight * selected_negative_margin_loss
                    answer_plan_loss_values.append(float(selected_negative_margin_loss.detach().cpu()))
                if selected_plan_ids:
                    selected_index = ANSWER_PLAN_OUTPUT_TO_ID["selected"]
                    role_losses.append(
                        _selection_margin_loss(
                            answer_plan_logits[:, selected_index],
                            candidate_ids=candidate_ids_for_plan,
                            positive_ids=selected_plan_ids,
                            margin=float(resolved_loss_config["answer_plan_selection_margin"]),
                            top_k=int(resolved_loss_config["answer_plan_selection_top_k"]),
                            required_positive_count=_selection_positive_coverage_count(
                                supervision,
                                positive_count=len(selected_plan_ids),
                                top_k=int(resolved_loss_config["answer_plan_selection_top_k"]),
                                base_count=1,
                                multi_positive_count=min(3, max(1, len(selected_plan_ids))),
                                multi_positive_fraction=0.5,
                            ),
                        )
                    )
                if role_losses:
                    answer_plan_loss = torch.stack([loss.float() for loss in role_losses]).mean()
                    total_loss = total_loss + float(resolved_loss_config["answer_plan_loss_weight"]) * answer_plan_loss
                    answer_plan_loss_values.append(float(answer_plan_loss.detach().cpu()))
                    answer_plan_loss_count += 1
                if selected_plan_ids:
                    selected_scores = torch.sigmoid(answer_plan_logits[:, ANSWER_PLAN_OUTPUT_TO_ID["selected"]]).detach()
                    selected_ranked_indices = torch.argsort(selected_scores, descending=True)
                    selected_ranked_event_ids = [
                        candidate_ids_for_plan[int(index)]
                        for index in selected_ranked_indices.tolist()
                        if int(index) < len(candidate_ids_for_plan)
                    ]
                    answer_plan_selected_total += 1
                    example_answer_plan_total = 1
                    if selected_ranked_event_ids[:5] and any(item in selected_plan_ids for item in selected_ranked_event_ids[:5]):
                        answer_plan_selected_recall5_hits += 1
                        example_answer_plan_recall5_hit = 1
                    example_answer_plan_positive_hits = _positive_coverage_hit_count(
                        selected_ranked_event_ids,
                        selected_plan_ids,
                        limit=5,
                    )
                    example_answer_plan_positive_total = min(len(selected_plan_ids), 5)
                    answer_plan_selected_positive_hits += example_answer_plan_positive_hits
                    answer_plan_selected_positive_total += example_answer_plan_positive_total
                if current_plan_ids:
                    current_scores = torch.sigmoid(answer_plan_logits[:, ANSWER_PLAN_OUTPUT_TO_ID["current"]]).detach()
                    current_ranked_indices = torch.argsort(current_scores, descending=True)
                    current_ranked_event_ids = [
                        candidate_ids_for_plan[int(index)]
                        for index in current_ranked_indices.tolist()
                        if int(index) < len(candidate_ids_for_plan)
                    ]
                    answer_plan_current_total += 1
                    example_answer_plan_current_total = 1
                    if current_ranked_event_ids and current_ranked_event_ids[0] in current_plan_ids:
                        answer_plan_current_top1_hits += 1
                        example_answer_plan_current_top1_hit = 1

            positive_event_ids = supervision.get("positive_event_ids", frozenset())
            positive_path_ids = supervision.get("positive_path_ids", frozenset())
            negative_event_ids = supervision.get("negative_event_ids", frozenset())
            hard_negative_event_ids = supervision.get("hard_negative_event_ids", negative_event_ids)
            if positive_event_ids and outputs["recall_event_logits"].numel() > 0:
                recall_targets = _label_tensor_from_positive_ids(
                    outputs["recall_event_ids"],
                    positive_event_ids,
                    device=device,
                )
                recall_loss = _recall_loss(outputs["recall_event_logits"], recall_targets)
                total_loss = total_loss + float(resolved_loss_config["recall_loss_weight"]) * recall_loss
                recall_loss_values.append(float(recall_loss.detach().cpu()))
                recall_loss_count += 1
                recall_selection_loss = _selection_margin_loss(
                    outputs["recall_event_logits"],
                    candidate_ids=outputs["recall_event_ids"],
                    positive_ids=positive_event_ids,
                    margin=float(resolved_loss_config["recall_selection_margin"]),
                    top_k=int(resolved_loss_config["recall_selection_top_k"]),
                    required_positive_count=_selection_positive_coverage_count(
                        supervision,
                        positive_count=len(positive_event_ids),
                        top_k=int(resolved_loss_config["recall_selection_top_k"]),
                        base_count=max(1, int(resolved_loss_config.get("recall_selection_positive_coverage_count", 1) or 1)),
                        multi_positive_count=max(
                            1,
                            int(resolved_loss_config.get("multi_positive_recall_coverage_count", 1) or 1),
                        ),
                        multi_positive_fraction=float(
                            resolved_loss_config.get("multi_positive_coverage_fraction", 0.0) or 0.0
                        ),
                    ),
                )
                total_loss = total_loss + float(resolved_loss_config["recall_selection_loss_weight"]) * recall_selection_loss
                recall_selection_loss_values.append(float(recall_selection_loss.detach().cpu()))
                recall_selection_loss_count += 1
                recall_scores = torch.sigmoid(outputs["recall_event_logits"]).detach()
                ranked_indices = torch.argsort(recall_scores, descending=True)
                ranked_event_ids = [outputs["recall_event_ids"][int(index)] for index in ranked_indices.tolist()]
                recall_event_total += 1
                example_recall_total = 1
                if ranked_event_ids[:24] and any(item in positive_event_ids for item in ranked_event_ids[:24]):
                    recall_event24_hits += 1
                    example_recall24_hit = 1
                example_recall24_positive_hits = _positive_coverage_hit_count(
                    ranked_event_ids,
                    positive_event_ids,
                    limit=24,
                )
                example_recall24_positive_total = min(len(positive_event_ids), 24)
                recall_event24_positive_hits += example_recall24_positive_hits
                recall_event24_positive_total += example_recall24_positive_total

            if positive_event_ids and outputs["event_logits"].numel() > 0:
                online_hard_negative_event_ids = _online_hard_negative_ids(
                    outputs["event_logits"],
                    candidate_ids=outputs["candidate_event_ids"],
                    positive_ids=positive_event_ids,
                    limit=int(resolved_loss_config.get("online_event_hard_negative_limit", 0) or 0),
                )
                combined_hard_negative_event_ids = dedupe_texts(
                    [*list(hard_negative_event_ids or []), *online_hard_negative_event_ids]
                )
                event_targets = _label_tensor_from_positive_ids(
                    outputs["candidate_event_ids"],
                    positive_event_ids,
                    device=device,
                )
                event_loss = _ranking_binary_loss(outputs["event_logits"], event_targets, margin=0.25)
                total_loss = total_loss + float(resolved_loss_config["event_loss_weight"]) * event_loss
                event_loss_values.append(float(event_loss.detach().cpu()))
                event_loss_count += 1
                event_distractor_loss = _event_distractor_supervision_loss(
                    outputs.get("event_distractor_logits", outputs["event_logits"]),
                    candidate_ids=outputs["candidate_event_ids"],
                    positive_ids=positive_event_ids,
                    hard_negative_ids=combined_hard_negative_event_ids,
                )
                total_loss = total_loss + float(resolved_loss_config["event_distractor_loss_weight"]) * event_distractor_loss
                event_distractor_loss_values.append(float(event_distractor_loss.detach().cpu()))
                event_distractor_loss_count += 1
                tunnel_positive_event_ids = _tunnel_positive_event_ids_from_supervision(supervision)
                tunnel_hard_negative_event_ids = _tunnel_hard_negative_event_ids_from_supervision(supervision)
                event_tunnel_logits = outputs.get("event_tunnel_support_logits", outputs["event_logits"].new_zeros((0,)))
                if tunnel_positive_event_ids and isinstance(event_tunnel_logits, Tensor) and event_tunnel_logits.numel() > 0:
                    event_tunnel_targets = _label_tensor_from_positive_ids(
                        outputs["candidate_event_ids"],
                        tunnel_positive_event_ids,
                        device=device,
                    )
                    event_tunnel_loss = _ranking_binary_loss(event_tunnel_logits, event_tunnel_targets, margin=0.2)
                    if tunnel_hard_negative_event_ids:
                        event_tunnel_loss = 0.5 * (
                            event_tunnel_loss
                            + _hard_negative_margin_loss(
                                event_tunnel_logits,
                                candidate_ids=outputs["candidate_event_ids"],
                                positive_ids=tunnel_positive_event_ids,
                                hard_negative_ids=tunnel_hard_negative_event_ids,
                                margin=float(resolved_loss_config["event_tunnel_margin"]),
                            )
                        )
                    total_loss = total_loss + float(resolved_loss_config["event_tunnel_loss_weight"]) * event_tunnel_loss
                    event_tunnel_loss_values.append(float(event_tunnel_loss.detach().cpu()))
                    event_tunnel_loss_count += 1
                    event_tunnel_selection_loss = _selection_margin_loss(
                        event_tunnel_logits,
                        candidate_ids=outputs["candidate_event_ids"],
                        positive_ids=tunnel_positive_event_ids,
                        margin=float(resolved_loss_config["event_tunnel_margin"]),
                        top_k=int(resolved_loss_config["final_event_set_top_k"]),
                        required_positive_count=_selection_positive_coverage_count(
                            supervision,
                            positive_count=len(tunnel_positive_event_ids),
                            top_k=int(resolved_loss_config["final_event_set_top_k"]),
                            base_count=max(1, int(resolved_loss_config.get("final_event_set_positive_coverage_count", 1) or 1)),
                            multi_positive_count=max(
                                1,
                                int(resolved_loss_config.get("multi_positive_final_event_set_coverage_count", 1) or 1),
                            ),
                            multi_positive_fraction=float(
                                resolved_loss_config.get("multi_positive_coverage_fraction", 0.0) or 0.0
                            ),
                        ),
                    )
                    total_loss = total_loss + float(resolved_loss_config["event_tunnel_selection_loss_weight"]) * event_tunnel_selection_loss
                    event_tunnel_selection_loss_values.append(float(event_tunnel_selection_loss.detach().cpu()))
                    event_tunnel_selection_loss_count += 1
                event_hard_negative_loss = _hard_negative_margin_loss(
                    outputs["event_logits"],
                    candidate_ids=outputs["candidate_event_ids"],
                    positive_ids=positive_event_ids,
                    hard_negative_ids=combined_hard_negative_event_ids,
                    margin=float(resolved_loss_config["event_hard_negative_margin"]),
                )
                total_loss = total_loss + float(resolved_loss_config["event_hard_negative_loss_weight"]) * event_hard_negative_loss
                event_hard_negative_loss_values.append(float(event_hard_negative_loss.detach().cpu()))
                event_hard_negative_loss_count += 1
                event_selection_loss = _selection_margin_loss(
                    outputs["event_logits"],
                    candidate_ids=outputs["candidate_event_ids"],
                    positive_ids=positive_event_ids,
                    margin=float(resolved_loss_config["event_selection_margin"]),
                    top_k=int(resolved_loss_config["event_selection_top_k"]),
                    required_positive_count=_selection_positive_coverage_count(
                        supervision,
                        positive_count=len(positive_event_ids),
                        top_k=int(resolved_loss_config["event_selection_top_k"]),
                        base_count=max(1, int(resolved_loss_config.get("event_selection_positive_coverage_count", 1) or 1)),
                        multi_positive_count=max(
                            1,
                            int(resolved_loss_config.get("multi_positive_event_coverage_count", 1) or 1),
                        ),
                        multi_positive_fraction=float(
                            resolved_loss_config.get("multi_positive_coverage_fraction", 0.0) or 0.0
                        ),
                    ),
                )
                total_loss = total_loss + float(resolved_loss_config["event_selection_loss_weight"]) * event_selection_loss
                event_selection_loss_values.append(float(event_selection_loss.detach().cpu()))
                event_selection_loss_count += 1
                final_event_set_loss = _runtime_selected_event_set_loss(
                    outputs["event_logits"],
                    candidate_event_ids=outputs["candidate_event_ids"],
                    path_logits=outputs.get("path_logits", outputs["event_logits"].new_zeros((0,))),
                    candidate_path_event_ids=outputs.get("candidate_path_event_ids", []),
                    positive_event_ids=positive_event_ids,
                    margin=float(resolved_loss_config["final_event_set_margin"]),
                    top_k=int(resolved_loss_config["final_event_set_top_k"]),
                    support_path_k=int(resolved_loss_config.get("final_event_set_support_path_k", 3) or 3),
                    required_positive_count=_selection_positive_coverage_count(
                        supervision,
                        positive_count=len(positive_event_ids),
                        top_k=int(resolved_loss_config["final_event_set_top_k"]),
                        base_count=max(1, int(resolved_loss_config.get("final_event_set_positive_coverage_count", 1) or 1)),
                        multi_positive_count=max(
                            1,
                            int(resolved_loss_config.get("multi_positive_final_event_set_coverage_count", 1) or 1),
                        ),
                        multi_positive_fraction=float(
                            resolved_loss_config.get("multi_positive_coverage_fraction", 0.0) or 0.0
                        ),
                    ),
                )
                total_loss = total_loss + float(resolved_loss_config["final_event_set_loss_weight"]) * final_event_set_loss
                final_event_set_loss_values.append(float(final_event_set_loss.detach().cpu()))
                final_event_set_loss_count += 1
                event_matrix_delta_loss = _matrix_delta_direction_loss(
                    outputs.get("matrix_event_delta_logits", outputs["event_logits"].new_zeros((0,))),
                    matrix_ids=outputs.get("matrix_event_ids", []),
                    positive_ids=positive_event_ids,
                    hard_negative_ids=combined_hard_negative_event_ids,
                    margin=float(resolved_loss_config["event_matrix_delta_margin"]),
                )
                total_loss = total_loss + float(resolved_loss_config["event_matrix_delta_loss_weight"]) * event_matrix_delta_loss
                event_matrix_delta_loss_values.append(float(event_matrix_delta_loss.detach().cpu()))
                event_matrix_delta_loss_count += 1
                scores = torch.sigmoid(outputs["event_logits"]).detach()
                ranked_indices = torch.argsort(scores, descending=True)
                ranked_event_ids = [outputs["candidate_event_ids"][int(index)] for index in ranked_indices.tolist()]
                event_recall_total += 1
                example_event_total = 1
                if ranked_event_ids[:1] and any(item in positive_event_ids for item in ranked_event_ids[:1]):
                    event_recall1_hits += 1
                    example_event1_hit = 1
                if ranked_event_ids[:5] and any(item in positive_event_ids for item in ranked_event_ids[:5]):
                    event_recall5_hits += 1
                    example_event5_hit = 1
                example_event5_positive_hits = _positive_coverage_hit_count(
                    ranked_event_ids,
                    positive_event_ids,
                    limit=5,
                )
                example_event5_positive_total = min(len(positive_event_ids), 5)
                event5_positive_hits += example_event5_positive_hits
                event5_positive_total += example_event5_positive_total

            if positive_path_ids and outputs["path_logits"].numel() > 0:
                path_hard_negative_ids = _path_hard_negative_ids_from_supervision(
                    supervision,
                    candidate_path_ids=outputs["candidate_path_ids"],
                    candidate_path_event_ids=outputs.get("candidate_path_event_ids"),
                    candidate_path_types=outputs.get("candidate_path_types"),
                )
                online_path_hard_negative_ids = _online_hard_negative_path_ids(
                    outputs["path_logits"],
                    candidate_path_ids=outputs["candidate_path_ids"],
                    positive_path_ids=positive_path_ids,
                    candidate_path_event_ids=outputs.get("candidate_path_event_ids"),
                    positive_event_ids=positive_event_ids,
                    limit=int(resolved_loss_config.get("online_path_hard_negative_limit", 0) or 0),
                )
                combined_path_hard_negative_ids = dedupe_texts(
                    [*path_hard_negative_ids, *online_path_hard_negative_ids]
                )
                path_targets = _label_tensor_from_positive_ids(
                    outputs["candidate_path_ids"],
                    positive_path_ids,
                    device=device,
                )
                path_loss = _ranking_binary_loss(outputs["path_logits"], path_targets, margin=0.2)
                total_loss = total_loss + float(resolved_loss_config["path_loss_weight"]) * path_loss
                path_loss_values.append(float(path_loss.detach().cpu()))
                path_loss_count += 1
                path_tunnel_logits = outputs.get("path_tunnel_support_logits", outputs["path_logits"].new_zeros((0,)))
                if isinstance(path_tunnel_logits, Tensor) and path_tunnel_logits.numel() > 0:
                    path_tunnel_targets = _label_tensor_from_positive_ids(
                        outputs["candidate_path_ids"],
                        positive_path_ids,
                        device=device,
                    )
                    path_tunnel_loss = _ranking_binary_loss(path_tunnel_logits, path_tunnel_targets, margin=0.16)
                    if combined_path_hard_negative_ids:
                        path_tunnel_loss = 0.5 * (
                            path_tunnel_loss
                            + _hard_negative_margin_loss(
                                path_tunnel_logits,
                                candidate_ids=outputs["candidate_path_ids"],
                                positive_ids=positive_path_ids,
                                hard_negative_ids=combined_path_hard_negative_ids,
                                margin=float(resolved_loss_config["path_tunnel_margin"]),
                            )
                        )
                    total_loss = total_loss + float(resolved_loss_config["path_tunnel_loss_weight"]) * path_tunnel_loss
                    path_tunnel_loss_values.append(float(path_tunnel_loss.detach().cpu()))
                    path_tunnel_loss_count += 1
                    path_tunnel_selection_loss = _selection_margin_loss(
                        path_tunnel_logits,
                        candidate_ids=outputs["candidate_path_ids"],
                        positive_ids=positive_path_ids,
                        margin=float(resolved_loss_config["path_tunnel_margin"]),
                        top_k=int(resolved_loss_config["path_selection_top_k"]),
                        required_positive_count=_selection_positive_coverage_count(
                            supervision,
                            positive_count=len(positive_path_ids),
                            top_k=int(resolved_loss_config["path_selection_top_k"]),
                            base_count=max(1, int(resolved_loss_config.get("path_selection_positive_coverage_count", 1) or 1)),
                            multi_positive_count=max(
                                1,
                                int(resolved_loss_config.get("multi_positive_path_coverage_count", 1) or 1),
                            ),
                            multi_positive_fraction=float(
                                resolved_loss_config.get("multi_positive_coverage_fraction", 0.0) or 0.0
                            ),
                        ),
                    )
                    total_loss = total_loss + float(resolved_loss_config["path_tunnel_selection_loss_weight"]) * path_tunnel_selection_loss
                    path_tunnel_selection_loss_values.append(float(path_tunnel_selection_loss.detach().cpu()))
                    path_tunnel_selection_loss_count += 1
                path_tunnel_delta_logits = outputs.get(
                    "path_tunnel_delta_logits",
                    outputs["path_logits"].new_zeros((0,)),
                )
                if (
                    isinstance(path_tunnel_delta_logits, Tensor)
                    and path_tunnel_delta_logits.numel() > 0
                    and float(resolved_loss_config.get("path_tunnel_delta_loss_weight", 0.0) or 0.0) > 0.0
                ):
                    path_tunnel_delta_loss = _matrix_delta_direction_loss(
                        path_tunnel_delta_logits,
                        matrix_ids=outputs["candidate_path_ids"],
                        positive_ids=positive_path_ids,
                        hard_negative_ids=combined_path_hard_negative_ids,
                        margin=float(resolved_loss_config["path_tunnel_margin"]),
                        positive_weight=0.45,
                        hard_negative_weight=3.0,
                        pair_weight=1.0,
                    )
                    total_loss = total_loss + float(resolved_loss_config["path_tunnel_delta_loss_weight"]) * path_tunnel_delta_loss
                    path_tunnel_delta_loss_values.append(float(path_tunnel_delta_loss.detach().cpu()))
                    path_tunnel_delta_loss_count += 1
                path_hard_negative_loss = _hard_negative_margin_loss(
                    outputs["path_logits"],
                    candidate_ids=outputs["candidate_path_ids"],
                    positive_ids=positive_path_ids,
                    hard_negative_ids=combined_path_hard_negative_ids,
                    margin=float(resolved_loss_config["path_hard_negative_margin"]),
                )
                total_loss = total_loss + float(resolved_loss_config["path_hard_negative_loss_weight"]) * path_hard_negative_loss
                path_hard_negative_loss_values.append(float(path_hard_negative_loss.detach().cpu()))
                path_hard_negative_loss_count += 1
                path_selection_loss = _selection_margin_loss(
                    outputs["path_logits"],
                    candidate_ids=outputs["candidate_path_ids"],
                    positive_ids=positive_path_ids,
                    margin=float(resolved_loss_config["path_selection_margin"]),
                    top_k=int(resolved_loss_config["path_selection_top_k"]),
                    required_positive_count=_selection_positive_coverage_count(
                        supervision,
                        positive_count=len(positive_path_ids),
                        top_k=int(resolved_loss_config["path_selection_top_k"]),
                        base_count=max(1, int(resolved_loss_config.get("path_selection_positive_coverage_count", 1) or 1)),
                        multi_positive_count=max(
                            1,
                            int(resolved_loss_config.get("multi_positive_path_coverage_count", 1) or 1),
                        ),
                        multi_positive_fraction=float(
                            resolved_loss_config.get("multi_positive_coverage_fraction", 0.0) or 0.0
                        ),
                    ),
                )
                total_loss = total_loss + float(resolved_loss_config["path_selection_loss_weight"]) * path_selection_loss
                path_selection_loss_values.append(float(path_selection_loss.detach().cpu()))
                path_selection_loss_count += 1
                path_matrix_delta_loss = _matrix_delta_direction_loss(
                    outputs.get("matrix_path_delta_logits", outputs["path_logits"].new_zeros((0,))),
                    matrix_ids=outputs.get("matrix_path_ids", []),
                    positive_ids=positive_path_ids,
                    hard_negative_ids=combined_path_hard_negative_ids,
                    margin=float(resolved_loss_config["path_matrix_delta_margin"]),
                )
                total_loss = total_loss + float(resolved_loss_config["path_matrix_delta_loss_weight"]) * path_matrix_delta_loss
                path_matrix_delta_loss_values.append(float(path_matrix_delta_loss.detach().cpu()))
                path_matrix_delta_loss_count += 1
                scores = torch.sigmoid(outputs["path_logits"]).detach()
                ranked_indices = torch.argsort(scores, descending=True)
                ranked_path_ids = [outputs["candidate_path_ids"][int(index)] for index in ranked_indices.tolist()]
                path_recall_total += 1
                example_path_total = 1
                if ranked_path_ids[:3] and any(item in positive_path_ids for item in ranked_path_ids[:3]):
                    path_recall3_hits += 1
                    example_path3_hit = 1
                example_path3_positive_hits = _positive_coverage_hit_count(
                    ranked_path_ids,
                    positive_path_ids,
                    limit=3,
                )
                example_path3_positive_total = min(len(positive_path_ids), 3)
                path3_positive_hits += example_path3_positive_hits
                path3_positive_total += example_path3_positive_total
                path_tunnel_support_logits = outputs.get(
                    "path_tunnel_support_logits",
                    outputs["path_logits"].new_zeros((0,)),
                )
                if isinstance(path_tunnel_support_logits, Tensor) and path_tunnel_support_logits.numel() > 0:
                    support_scores = torch.sigmoid(path_tunnel_support_logits).detach()
                    support_ranked_indices = torch.argsort(support_scores, descending=True)
                    support_ranked_path_ids = [
                        outputs["candidate_path_ids"][int(index)]
                        for index in support_ranked_indices.tolist()
                    ]
                    path_tunnel_support_recall_total += 1
                    if support_ranked_path_ids[:3] and any(
                        item in positive_path_ids for item in support_ranked_path_ids[:3]
                    ):
                        path_tunnel_support_recall3_hits += 1
                    path_tunnel_support_positive_hits += _positive_coverage_hit_count(
                        support_ranked_path_ids,
                        positive_path_ids,
                        limit=3,
                    )
                    path_tunnel_support_positive_total += min(len(positive_path_ids), 3)
                    for alpha, hit_attr in (
                        (0.25, "025"),
                        (0.50, "050"),
                        (1.00, "100"),
                    ):
                        rescue_scores = scores + (support_scores * float(alpha))
                        rescue_ranked_indices = torch.argsort(rescue_scores, descending=True)
                        rescue_ranked_path_ids = [
                            outputs["candidate_path_ids"][int(index)]
                            for index in rescue_ranked_indices.tolist()
                        ]
                        if hit_attr == "025":
                            path_tunnel_rescue025_recall_total += 1
                            if rescue_ranked_path_ids[:3] and any(
                                item in positive_path_ids for item in rescue_ranked_path_ids[:3]
                            ):
                                path_tunnel_rescue025_recall3_hits += 1
                        elif hit_attr == "050":
                            path_tunnel_rescue050_recall_total += 1
                            if rescue_ranked_path_ids[:3] and any(
                                item in positive_path_ids for item in rescue_ranked_path_ids[:3]
                            ):
                                path_tunnel_rescue050_recall3_hits += 1
                        else:
                            path_tunnel_rescue100_recall_total += 1
                            if rescue_ranked_path_ids[:3] and any(
                                item in positive_path_ids for item in rescue_ranked_path_ids[:3]
                            ):
                                path_tunnel_rescue100_recall3_hits += 1
                path_tunnel_delta_logits = outputs.get(
                    "path_tunnel_delta_logits",
                    outputs["path_logits"].new_zeros((0,)),
                )
                if isinstance(path_tunnel_delta_logits, Tensor) and path_tunnel_delta_logits.numel() > 0:
                    delta_scores = path_tunnel_delta_logits.detach()
                    delta_ranked_indices = torch.argsort(delta_scores, descending=True)
                    delta_ranked_path_ids = [
                        outputs["candidate_path_ids"][int(index)]
                        for index in delta_ranked_indices.tolist()
                    ]
                    path_tunnel_delta_recall_total += 1
                    if delta_ranked_path_ids[:3] and any(
                        item in positive_path_ids for item in delta_ranked_path_ids[:3]
                    ):
                        path_tunnel_delta_recall3_hits += 1
                    path_tunnel_delta_positive_hits += _positive_coverage_hit_count(
                        delta_ranked_path_ids,
                        positive_path_ids,
                        limit=3,
                    )
                    path_tunnel_delta_positive_total += min(len(positive_path_ids), 3)

            if bool(supervision.get("is_temporal_example", _is_temporal_example(example))) and outputs["temporal_logits"].numel() > 0:
                temporal_targets = _temporal_path_labels_from_supervision(
                    supervision,
                    outputs["candidate_temporal_path_ids"],
                    outputs["candidate_temporal_node_ids"],
                    candidate_temporal_event_ids=outputs.get("candidate_temporal_event_ids"),
                    device=device,
                )
                temporal_loss = _ranking_binary_loss(
                    outputs["temporal_logits"],
                    temporal_targets,
                    margin=0.15,
                    bce_weight=0.5,
                    positive_mass_weight=0.0,
                    pairwise_weight=0.5,
                )
                total_loss = total_loss + float(resolved_loss_config["temporal_loss_weight"]) * temporal_loss
                temporal_loss_values.append(float(temporal_loss.detach().cpu()))
                temporal_loss_count += 1
                temporal_scores = torch.sigmoid(outputs["temporal_logits"]).detach()
                temporal_total += 1
                example_temporal_total = 1
                if temporal_scores.numel() > 0:
                    top_index = int(torch.argmax(temporal_scores).item())
                    target_values = temporal_targets.detach().cpu().tolist()
                    if any(value > 0.5 for value in target_values):
                        if target_values[top_index] > 0.5:
                            temporal_hits += 1
                            example_temporal_hit = 1
                    elif float(torch.max(temporal_scores).item()) < 0.5:
                        temporal_hits += 1
                        example_temporal_hit = 1

            if apply_example_weights and example_training_weight > 0.0 and abs(example_training_weight - 1.0) > 1e-9:
                total_loss = total_loss * outputs["answer_type_logits"].new_tensor(float(example_training_weight))
            for stats in (answer_stats, supervision_stats):
                stats["recall_event_recall_total"] += example_recall_total
                stats["recall_event24_hits"] += example_recall24_hit
                stats["recall_event24_positive_hits"] += example_recall24_positive_hits
                stats["recall_event24_positive_total"] += example_recall24_positive_total
                stats["event_recall_total"] += example_event_total
                stats["event_recall1_hits"] += example_event1_hit
                stats["event_recall5_hits"] += example_event5_hit
                stats["event5_positive_hits"] += example_event5_positive_hits
                stats["event5_positive_total"] += example_event5_positive_total
                stats["path_recall_total"] += example_path_total
                stats["path_recall3_hits"] += example_path3_hit
                stats["path3_positive_hits"] += example_path3_positive_hits
                stats["path3_positive_total"] += example_path3_positive_total
                stats["answer_plan_selected_total"] += example_answer_plan_total
                stats["answer_plan_selected_recall5_hits"] += example_answer_plan_recall5_hit
                stats["answer_plan_selected_positive_hits"] += example_answer_plan_positive_hits
                stats["answer_plan_selected_positive_total"] += example_answer_plan_positive_total
                stats["answer_plan_current_total"] += example_answer_plan_current_total
                stats["answer_plan_current_top1_hits"] += example_answer_plan_current_top1_hit
                stats["temporal_total"] += example_temporal_total
                stats["temporal_hits"] += example_temporal_hit
            losses.append(total_loss)
            loss_entries.append(
                (
                    total_loss,
                    _loss_group_key_for_example(
                        example,
                        supervision,
                        mode=clean_text(resolved_loss_config.get("loss_group_balancing_mode", "answer_type")),
                    ),
                )
            )

    if not losses:
        return _empty_score_batch_result(device=device, graph_error_count=graph_error_count)
    batch_loss, loss_group_count = _group_balanced_batch_loss(
        loss_entries,
        mode=clean_text(resolved_loss_config.get("loss_group_balancing_mode", "answer_type")),
    )
    return {
        "loss": batch_loss,
        "metrics": {
            "loss": float(batch_loss.detach().cpu()),
            "recall_loss": round(sum(recall_loss_values) / max(1, len(recall_loss_values)), 6),
            "event_loss": round(sum(event_loss_values) / max(1, len(event_loss_values)), 6),
            "path_loss": round(sum(path_loss_values) / max(1, len(path_loss_values)), 6),
            "temporal_loss": round(sum(temporal_loss_values) / max(1, len(temporal_loss_values)), 6),
            "answer_type_loss": round(sum(answer_type_loss_values) / max(1, len(answer_type_loss_values)), 6),
            "answer_plan_loss": round(sum(answer_plan_loss_values) / max(1, len(answer_plan_loss_values)), 6),
            "token_role_loss": round(sum(token_role_loss_values) / max(1, len(token_role_loss_values)), 6),
            "question_understanding_loss": round(
                sum(question_understanding_loss_values) / max(1, len(question_understanding_loss_values)),
                6,
            ),
            "memory_router_loss": round(sum(memory_router_loss_values) / max(1, len(memory_router_loss_values)), 6),
            "memory_router_exact_match": round(memory_router_exact_hits / max(1, memory_router_total), 6),
            "memory_router_f1": round(memory_router_f1_sum / max(1, memory_router_total), 6),
            "event_distractor_loss": round(sum(event_distractor_loss_values) / max(1, len(event_distractor_loss_values)), 6),
            "event_tunnel_loss": round(sum(event_tunnel_loss_values) / max(1, len(event_tunnel_loss_values)), 6),
            "path_tunnel_loss": round(sum(path_tunnel_loss_values) / max(1, len(path_tunnel_loss_values)), 6),
            "path_tunnel_delta_loss": round(
                sum(path_tunnel_delta_loss_values) / max(1, len(path_tunnel_delta_loss_values)),
                6,
            ),
            "event_tunnel_selection_loss": round(
                sum(event_tunnel_selection_loss_values) / max(1, len(event_tunnel_selection_loss_values)),
                6,
            ),
            "path_tunnel_selection_loss": round(
                sum(path_tunnel_selection_loss_values) / max(1, len(path_tunnel_selection_loss_values)),
                6,
            ),
            "event_hard_negative_loss": round(sum(event_hard_negative_loss_values) / max(1, len(event_hard_negative_loss_values)), 6),
            "path_hard_negative_loss": round(sum(path_hard_negative_loss_values) / max(1, len(path_hard_negative_loss_values)), 6),
            "recall_selection_loss": round(sum(recall_selection_loss_values) / max(1, len(recall_selection_loss_values)), 6),
            "event_selection_loss": round(sum(event_selection_loss_values) / max(1, len(event_selection_loss_values)), 6),
            "path_selection_loss": round(sum(path_selection_loss_values) / max(1, len(path_selection_loss_values)), 6),
            "final_event_set_loss": round(sum(final_event_set_loss_values) / max(1, len(final_event_set_loss_values)), 6),
            "event_matrix_delta_loss": round(sum(event_matrix_delta_loss_values) / max(1, len(event_matrix_delta_loss_values)), 6),
            "path_matrix_delta_loss": round(sum(path_matrix_delta_loss_values) / max(1, len(path_matrix_delta_loss_values)), 6),
            "answer_refusal_loss": round(sum(answer_refusal_loss_values) / max(1, len(answer_refusal_loss_values)), 6),
            "recall_event_recall_at_24": round(recall_event24_hits / max(1, recall_event_total), 6),
            "recall_event_positive_coverage_at_24": round(
                recall_event24_positive_hits / max(1, recall_event24_positive_total),
                6,
            ),
            "event_recall_at_1": round(event_recall1_hits / max(1, event_recall_total), 6),
            "event_recall_at_5": round(event_recall5_hits / max(1, event_recall_total), 6),
            "event_positive_coverage_at_5": round(event5_positive_hits / max(1, event5_positive_total), 6),
            "path_recall_at_3": round(path_recall3_hits / max(1, path_recall_total), 6),
            "path_positive_coverage_at_3": round(path3_positive_hits / max(1, path3_positive_total), 6),
            "path_tunnel_support_recall_at_3": round(
                path_tunnel_support_recall3_hits / max(1, path_tunnel_support_recall_total),
                6,
            ),
            "path_tunnel_support_positive_coverage_at_3": round(
                path_tunnel_support_positive_hits / max(1, path_tunnel_support_positive_total),
                6,
            ),
            "path_tunnel_delta_recall_at_3": round(
                path_tunnel_delta_recall3_hits / max(1, path_tunnel_delta_recall_total),
                6,
            ),
            "path_tunnel_delta_positive_coverage_at_3": round(
                path_tunnel_delta_positive_hits / max(1, path_tunnel_delta_positive_total),
                6,
            ),
            "answer_plan_selected_recall_at_5": round(
                answer_plan_selected_recall5_hits / max(1, answer_plan_selected_total),
                6,
            ),
            "answer_plan_selected_positive_coverage_at_5": round(
                answer_plan_selected_positive_hits / max(1, answer_plan_selected_positive_total),
                6,
            ),
            "answer_plan_current_top1_accuracy": round(
                answer_plan_current_top1_hits / max(1, answer_plan_current_total),
                6,
            ),
            "path_tunnel_rescue025_recall_at_3": round(
                path_tunnel_rescue025_recall3_hits / max(1, path_tunnel_rescue025_recall_total),
                6,
            ),
            "path_tunnel_rescue050_recall_at_3": round(
                path_tunnel_rescue050_recall3_hits / max(1, path_tunnel_rescue050_recall_total),
                6,
            ),
            "path_tunnel_rescue100_recall_at_3": round(
                path_tunnel_rescue100_recall3_hits / max(1, path_tunnel_rescue100_recall_total),
                6,
            ),
            "temporal_accuracy": round(temporal_hits / max(1, temporal_total), 6),
            "samples": len(losses),
            "loss_group_count": int(loss_group_count),
            "recall_event24_positive_total": int(recall_event24_positive_total),
            "event5_positive_total": int(event5_positive_total),
            "path3_positive_total": int(path3_positive_total),
            "path_tunnel_support_recall_total": int(path_tunnel_support_recall_total),
            "path_tunnel_support_positive_total": int(path_tunnel_support_positive_total),
            "path_tunnel_delta_recall_total": int(path_tunnel_delta_recall_total),
            "path_tunnel_delta_positive_total": int(path_tunnel_delta_positive_total),
            "answer_plan_selected_total": int(answer_plan_selected_total),
            "answer_plan_selected_positive_total": int(answer_plan_selected_positive_total),
            "answer_plan_current_total": int(answer_plan_current_total),
            "path_tunnel_rescue025_recall_total": int(path_tunnel_rescue025_recall_total),
            "path_tunnel_rescue050_recall_total": int(path_tunnel_rescue050_recall_total),
            "path_tunnel_rescue100_recall_total": int(path_tunnel_rescue100_recall_total),
            "answer_type_metrics": _finalize_answer_type_metric_stats(answer_type_metric_stats),
            "supervision_bucket_metrics": _finalize_answer_type_metric_stats(supervision_bucket_metric_stats),
            "training_weight_mean": round(sum(training_weight_values) / max(1, len(training_weight_values)), 6),
            "training_weight_min": round(min(training_weight_values) if training_weight_values else 1.0, 6),
            "training_weight_max": round(max(training_weight_values) if training_weight_values else 1.0, 6),
            "recall_loss_count": recall_loss_count,
            "event_loss_count": event_loss_count,
            "path_loss_count": path_loss_count,
            "temporal_loss_count": temporal_loss_count,
            "answer_type_loss_count": answer_type_loss_count,
            "answer_plan_loss_count": answer_plan_loss_count,
            "token_role_loss_count": token_role_loss_count,
            "question_understanding_loss_count": question_understanding_loss_count,
            "memory_router_loss_count": memory_router_loss_count,
            "memory_router_total": memory_router_total,
            "event_distractor_loss_count": event_distractor_loss_count,
            "event_tunnel_loss_count": event_tunnel_loss_count,
            "path_tunnel_loss_count": path_tunnel_loss_count,
            "path_tunnel_delta_loss_count": path_tunnel_delta_loss_count,
            "event_tunnel_selection_loss_count": event_tunnel_selection_loss_count,
            "path_tunnel_selection_loss_count": path_tunnel_selection_loss_count,
            "event_hard_negative_loss_count": event_hard_negative_loss_count,
            "path_hard_negative_loss_count": path_hard_negative_loss_count,
            "recall_selection_loss_count": recall_selection_loss_count,
            "event_selection_loss_count": event_selection_loss_count,
            "path_selection_loss_count": path_selection_loss_count,
            "final_event_set_loss_count": final_event_set_loss_count,
            "event_matrix_delta_loss_count": event_matrix_delta_loss_count,
            "path_matrix_delta_loss_count": path_matrix_delta_loss_count,
            "answer_refusal_loss_count": answer_refusal_loss_count,
            "recall_event_recall_total": recall_event_total,
            "event_recall_total": event_recall_total,
            "path_recall_total": path_recall_total,
            "temporal_total": temporal_total,
            "graph_error_count": graph_error_count,
        },
    }


def score_batch(
    model: LocomoNodeMemoryModel,
    batch: Sequence[QueryTrainingExample],
    graph_cache: Mapping[str, GraphCacheItem],
    *,
    device: torch.device,
    loss_config: Mapping[str, Any] | None = None,
    score_kwargs: Mapping[str, Any] | None = None,
    apply_example_weights: bool = False,
    graph_error_callback: Callable[[Dict[str, Any]], None] | None = None,
    graph_error_stage: str = "",
) -> Dict[str, Any]:
    prepared_batch = _prepare_batch_payload_from_graph_cache(
        batch,
        graph_cache,
        merge_device=device,
        graph_error_stage=graph_error_stage,
    )
    return score_prepared_batch(
        model,
        prepared_batch,
        device=device,
        loss_config=loss_config,
        score_kwargs=score_kwargs,
        apply_example_weights=apply_example_weights,
        graph_error_callback=graph_error_callback,
        graph_error_stage=graph_error_stage,
    )


def iter_batches(
    rows: Sequence[QueryTrainingExample],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> List[List[QueryTrainingExample]]:
    resolved_batch_size = max(1, int(batch_size))
    if any(not hasattr(row, "conversation_id") for row in rows):
        working = list(rows)
        if shuffle:
            random.Random(seed).shuffle(working)
        return [
            working[index : index + resolved_batch_size]
            for index in range(0, len(working), resolved_batch_size)
        ]
    grouped: "OrderedDict[str, List[QueryTrainingExample]]" = OrderedDict()
    for row in rows:
        grouped.setdefault(row.conversation_id, []).append(row)
    conversation_ids = list(grouped)
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(conversation_ids)
    batches: List[List[QueryTrainingExample]] = []
    current_batch: List[QueryTrainingExample] = []
    for conversation_id in conversation_ids:
        group_rows = list(grouped[conversation_id])
        if shuffle:
            rng.shuffle(group_rows)
        if len(group_rows) > resolved_batch_size:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
            for index in range(0, len(group_rows), resolved_batch_size):
                batches.append(group_rows[index : index + resolved_batch_size])
            continue
        if current_batch and len(current_batch) + len(group_rows) > resolved_batch_size:
            batches.append(current_batch)
            current_batch = []
        current_batch.extend(group_rows)
    if current_batch:
        batches.append(current_batch)
    return batches


def _iter_batches_any(
    rows: Sequence[QueryTrainingExample],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    skip_batches: int = 0,
) -> Iterator[QueryBatchSpec | Sequence[QueryTrainingExample]]:
    custom_iter_batches = getattr(rows, "iter_batches", None)
    custom_iter_batch_specs = getattr(rows, "iter_batch_specs", None)
    if callable(custom_iter_batch_specs):
        batch_iter = iter(custom_iter_batch_specs(batch_size=batch_size, shuffle=shuffle, seed=seed))
    elif callable(custom_iter_batches):
        batch_iter = iter(custom_iter_batches(batch_size=batch_size, shuffle=shuffle, seed=seed))
    else:
        batch_iter = iter(iter_batches(rows, batch_size=batch_size, shuffle=shuffle, seed=seed))
    resolved_skip_batches = max(0, int(skip_batches or 0))
    for _ in range(resolved_skip_batches):
        try:
            next(batch_iter)
        except StopIteration:
            return
    yield from batch_iter


def _prefetch_graph_cache_for_pending_batches(
    graph_cache: Mapping[str, GraphCacheItem],
    pending_batches: Sequence[QueryBatchSpec | Sequence[QueryTrainingExample]],
) -> None:
    prefetch = getattr(graph_cache, "prefetch", None)
    if not callable(prefetch):
        return
    conversation_ids: List[str] = []
    seen: set[str] = set()
    for batch in pending_batches:
        for example in _materialize_query_batch_payload(batch):
            conversation_id = clean_text(example.conversation_id)
            if not conversation_id or conversation_id in seen:
                continue
            seen.add(conversation_id)
            conversation_ids.append(conversation_id)
    if conversation_ids:
        prefetch(conversation_ids)


def _iter_batches_with_prefetch(
    graph_cache: Mapping[str, GraphCacheItem],
    rows: Sequence[QueryTrainingExample],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    lookahead_batches: int,
    skip_batches: int = 0,
) -> Iterator[Sequence[QueryTrainingExample]]:
    batch_iter = iter(
        _iter_batches_any(
            rows,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            skip_batches=skip_batches,
        )
    )
    pending_batches: deque[Sequence[QueryTrainingExample]] = deque()
    pending_target = max(1, max(0, int(lookahead_batches or 0)) + 1)

    def _fill_pending() -> None:
        while len(pending_batches) < pending_target:
            try:
                pending_batches.append(next(batch_iter))
            except StopIteration:
                break

    _fill_pending()
    while pending_batches:
        _prefetch_graph_cache_for_pending_batches(graph_cache, list(pending_batches))
        yield pending_batches.popleft()
        _fill_pending()


def _prefetch_graph_cache_for_batches(
    graph_cache: Mapping[str, GraphCacheItem],
    batches: Sequence[QueryBatchSpec | Sequence[QueryTrainingExample]],
    *,
    batch_index: int,
    lookahead_batches: int,
) -> None:
    prefetch = getattr(graph_cache, "prefetch", None)
    if not callable(prefetch):
        return
    resolved_lookahead = max(0, int(lookahead_batches or 0))
    if resolved_lookahead <= 0:
        return
    conversation_ids: List[str] = []
    seen: set[str] = set()
    end_index = min(len(batches), int(batch_index) + resolved_lookahead + 1)
    for future_batch in batches[int(batch_index) : end_index]:
        for example in _materialize_query_batch_payload(future_batch):
            conversation_id = clean_text(example.conversation_id)
            if not conversation_id or conversation_id in seen:
                continue
            seen.add(conversation_id)
            conversation_ids.append(conversation_id)
    if conversation_ids:
        prefetch(conversation_ids)


def _select_eval_rows(
    rows: Sequence[QueryTrainingExample],
    *,
    max_rows: int = 0,
    seed: int = 0,
) -> tuple[Sequence[QueryTrainingExample], int]:
    total_rows = len(rows)
    resolved_max_rows = max(0, int(max_rows or 0))
    if resolved_max_rows <= 0 or total_rows <= resolved_max_rows:
        return rows, total_rows
    rng = random.Random(int(seed))
    selected_indices = sorted(rng.sample(range(total_rows), resolved_max_rows))
    return [rows[index] for index in selected_indices], total_rows


def evaluate_examples(
    model: LocomoNodeMemoryModel,
    rows: Sequence[QueryTrainingExample],
    graph_cache: Mapping[str, GraphCacheItem],
    *,
    device: torch.device,
    batch_size: int,
    loss_config: Mapping[str, Any] | None = None,
    score_kwargs: Mapping[str, Any] | None = None,
    prefetch_lookahead_batches: int = 0,
    graph_error_callback: Callable[[Dict[str, Any]], None] | None = None,
    worker_event_callback: Callable[[Dict[str, Any]], None] | None = None,
    graph_error_stage: str = "",
    max_rows: int = 0,
    sample_seed: int = 0,
    batch_prepare_context: Mapping[str, Any] | None = None,
    batch_prepare_workers: int = 0,
    batch_prepare_lookahead_batches: int = 0,
    batch_prepare_worker_pool: _PreparedBatchWorkerPool | None = None,
) -> Dict[str, Any]:
    model.eval()
    metrics_list: List[Dict[str, Any]] = []
    eval_rows, total_row_count = _select_eval_rows(rows, max_rows=max_rows, seed=sample_seed)
    evaluated_row_count = len(eval_rows)
    evaluation_limited = evaluated_row_count < total_row_count
    with torch.no_grad():
        if batch_prepare_worker_pool is not None or (
            max(0, int(batch_prepare_workers or 0)) > 0 and isinstance(batch_prepare_context, Mapping)
        ):
            for prepared_batch in _iter_prepared_batches_with_workers(
                eval_rows,
                graph_cache=graph_cache,
                batch_size=batch_size,
                shuffle=False,
                seed=int(sample_seed),
                worker_context=batch_prepare_context,
                worker_count=max(0, int(batch_prepare_workers or 0)),
                lookahead_batches=max(0, int(batch_prepare_lookahead_batches or 0)),
                graph_error_stage=f"{clean_text(graph_error_stage) or 'eval'}_batch_prepare",
                worker_pool=batch_prepare_worker_pool,
                prefer_completion_order=False,
                worker_event_callback=worker_event_callback,
            ):
                metrics_list.append(
                    score_prepared_batch(
                        model,
                        prepared_batch,
                        device=device,
                        loss_config=loss_config,
                        score_kwargs=score_kwargs,
                        graph_error_callback=graph_error_callback,
                        graph_error_stage=graph_error_stage,
                    )["metrics"]
                )
        else:
            for batch in _iter_batches_with_prefetch(
                graph_cache,
                eval_rows,
                batch_size=batch_size,
                shuffle=False,
                seed=int(sample_seed),
                lookahead_batches=prefetch_lookahead_batches,
            ):
                metrics_list.append(
                    score_batch(
                        model,
                        batch,
                        graph_cache,
                        device=device,
                        loss_config=loss_config,
                        score_kwargs=score_kwargs,
                        graph_error_callback=graph_error_callback,
                        graph_error_stage=graph_error_stage,
                    )["metrics"]
                )
    if not metrics_list:
        return {
            "loss": 0.0,
            "recall_loss": 0.0,
            "event_loss": 0.0,
            "path_loss": 0.0,
            "temporal_loss": 0.0,
            "answer_type_loss": 0.0,
            "answer_plan_loss": 0.0,
            "token_role_loss": 0.0,
            "memory_router_loss": 0.0,
            "memory_router_exact_match": 0.0,
            "memory_router_f1": 0.0,
            "event_distractor_loss": 0.0,
            "event_tunnel_loss": 0.0,
            "path_tunnel_loss": 0.0,
            "path_tunnel_delta_loss": 0.0,
            "event_tunnel_selection_loss": 0.0,
            "path_tunnel_selection_loss": 0.0,
            "event_hard_negative_loss": 0.0,
            "path_hard_negative_loss": 0.0,
            "recall_selection_loss": 0.0,
            "event_selection_loss": 0.0,
            "path_selection_loss": 0.0,
            "final_event_set_loss": 0.0,
            "event_matrix_delta_loss": 0.0,
            "path_matrix_delta_loss": 0.0,
            "answer_refusal_loss": 0.0,
            "recall_event_recall_at_24": 0.0,
            "event_recall_at_1": 0.0,
            "event_recall_at_5": 0.0,
            "path_recall_at_3": 0.0,
            "answer_plan_selected_recall_at_5": 0.0,
            "answer_plan_selected_positive_coverage_at_5": 0.0,
            "answer_plan_current_top1_accuracy": 0.0,
            "temporal_accuracy": 0.0,
            "training_weight_mean": 1.0,
            "training_weight_min": 1.0,
            "training_weight_max": 1.0,
            "samples": 0,
            "loss_group_count": 0,
            "answer_type_metrics": {},
            "supervision_bucket_metrics": {},
            "recall_loss_count": 0,
            "event_loss_count": 0,
            "path_loss_count": 0,
            "temporal_loss_count": 0,
            "answer_type_loss_count": 0,
            "answer_plan_loss_count": 0,
            "token_role_loss_count": 0,
            "memory_router_loss_count": 0,
            "memory_router_total": 0,
            "event_distractor_loss_count": 0,
            "event_tunnel_loss_count": 0,
            "path_tunnel_loss_count": 0,
            "path_tunnel_delta_loss_count": 0,
            "event_tunnel_selection_loss_count": 0,
            "path_tunnel_selection_loss_count": 0,
            "event_hard_negative_loss_count": 0,
            "path_hard_negative_loss_count": 0,
            "recall_selection_loss_count": 0,
            "event_selection_loss_count": 0,
            "path_selection_loss_count": 0,
            "final_event_set_loss_count": 0,
            "event_matrix_delta_loss_count": 0,
            "path_matrix_delta_loss_count": 0,
            "answer_refusal_loss_count": 0,
            "recall_event_recall_total": 0,
            "event_recall_total": 0,
            "path_recall_total": 0,
            "answer_plan_selected_total": 0,
            "answer_plan_selected_positive_total": 0,
            "answer_plan_current_total": 0,
            "temporal_total": 0,
            "graph_error_count": 0,
            "rows_total": int(total_row_count),
            "rows_evaluated": int(evaluated_row_count),
            "evaluation_limited": bool(evaluation_limited),
        }
    summary: Dict[str, Any] = {}
    for key in metrics_list[0].keys():
        if isinstance(metrics_list[0].get(key), Mapping):
            summary[key] = _merge_nested_metric_summaries(metrics_list, key)  # type: ignore[assignment]
            continue
        if key in _METRIC_COUNT_KEYS:
            summary[key] = int(sum(int(item.get(key, 0) or 0) for item in metrics_list))
            continue
        weight_key = _METRIC_WEIGHT_KEYS.get(key, "samples")
        total_weight = float(sum(item.get(weight_key, 0.0) or 0.0 for item in metrics_list))
        if total_weight <= 0.0:
            summary[key] = 0.0
            continue
        weighted_total = sum(
            float(item.get(key, 0.0) or 0.0) * float(item.get(weight_key, 0.0) or 0.0)
            for item in metrics_list
        )
        summary[key] = round(weighted_total / total_weight, 6)
    summary["rows_total"] = int(total_row_count)
    summary["rows_evaluated"] = int(evaluated_row_count)
    summary["evaluation_limited"] = bool(evaluation_limited)
    return summary


def _skipped_eval_summary(sample_count: int) -> Dict[str, Any]:
    return {
        "loss": 0.0,
        "recall_loss": 0.0,
        "event_loss": 0.0,
        "path_loss": 0.0,
        "temporal_loss": 0.0,
        "answer_type_loss": 0.0,
        "answer_plan_loss": 0.0,
        "token_role_loss": 0.0,
        "memory_router_loss": 0.0,
        "memory_router_exact_match": 0.0,
        "memory_router_f1": 0.0,
        "event_distractor_loss": 0.0,
        "event_tunnel_loss": 0.0,
        "path_tunnel_loss": 0.0,
        "path_tunnel_delta_loss": 0.0,
        "event_tunnel_selection_loss": 0.0,
        "path_tunnel_selection_loss": 0.0,
        "event_hard_negative_loss": 0.0,
        "path_hard_negative_loss": 0.0,
        "recall_selection_loss": 0.0,
        "event_selection_loss": 0.0,
        "path_selection_loss": 0.0,
        "final_event_set_loss": 0.0,
        "event_matrix_delta_loss": 0.0,
        "path_matrix_delta_loss": 0.0,
        "answer_refusal_loss": 0.0,
        "recall_event_recall_at_24": 0.0,
        "event_recall_at_1": 0.0,
        "event_recall_at_5": 0.0,
        "path_recall_at_3": 0.0,
        "answer_plan_selected_recall_at_5": 0.0,
        "answer_plan_selected_positive_coverage_at_5": 0.0,
        "answer_plan_current_top1_accuracy": 0.0,
        "temporal_accuracy": 0.0,
        "training_weight_mean": 1.0,
        "training_weight_min": 1.0,
        "training_weight_max": 1.0,
        "samples": int(sample_count),
        "loss_group_count": 0,
        "answer_type_metrics": {},
        "supervision_bucket_metrics": {},
        "final_event_set_loss_count": 0,
        "path_tunnel_delta_loss_count": 0,
        "answer_plan_loss_count": 0,
        "memory_router_loss_count": 0,
        "memory_router_total": 0,
        "answer_plan_selected_total": 0,
        "answer_plan_selected_positive_total": 0,
        "answer_plan_current_total": 0,
        "graph_error_count": 0,
        "skipped": True,
    }


def _answer_type_metric_floor(
    summary: Mapping[str, Any],
    metric_key: str,
    total_key: str,
    *,
    fallback: float,
    min_total: int = 3,
    nested_key: str = "answer_type_metrics",
) -> float:
    answer_type_metrics = summary.get(nested_key, {})
    if not isinstance(answer_type_metrics, Mapping):
        return float(fallback)
    values: List[float] = []
    for metrics in answer_type_metrics.values():
        if not isinstance(metrics, Mapping):
            continue
        if int(metrics.get(total_key, 0) or 0) < int(min_total):
            continue
        values.append(float(metrics.get(metric_key, 0.0) or 0.0))
    if not values:
        return float(fallback)
    return min(values)


def _checkpoint_selection_key(summary: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    event_recall_at_5 = float(summary.get("event_recall_at_5", 0.0) or 0.0)
    path_recall_at_3 = float(summary.get("path_recall_at_3", 0.0) or 0.0)
    event_type_floor = _answer_type_metric_floor(
        summary,
        "event_recall_at_5",
        "event_recall_total",
        fallback=event_recall_at_5,
    )
    path_type_floor = _answer_type_metric_floor(
        summary,
        "path_recall_at_3",
        "path_recall_total",
        fallback=path_recall_at_3,
    )
    event_supervision_floor = _answer_type_metric_floor(
        summary,
        "event_recall_at_5",
        "event_recall_total",
        fallback=event_recall_at_5,
        nested_key="supervision_bucket_metrics",
    )
    path_supervision_floor = _answer_type_metric_floor(
        summary,
        "path_recall_at_3",
        "path_recall_total",
        fallback=path_recall_at_3,
        nested_key="supervision_bucket_metrics",
    )
    return (
        float(min(event_type_floor, event_supervision_floor)),
        float(summary.get("recall_event_recall_at_24", 0.0) or 0.0),
        float(min(path_type_floor, path_supervision_floor)),
        float(summary.get("temporal_accuracy", 0.0) or 0.0),
        -float(summary.get("loss", 0.0) or 0.0),
    )


def _checkpoint_selection_score(summary: Mapping[str, Any]) -> float:
    selection_key = _checkpoint_selection_key(summary)
    return round(
        selection_key[0] * 100.0
        + selection_key[1] * 10.0
        + selection_key[2] * 10.0
        + selection_key[3] * 5.0
        + selection_key[4],
        6,
    )


def train_locomo_node_memory(
    *,
    train_rows: Sequence[QueryTrainingExample],
    val_rows: Sequence[QueryTrainingExample],
    train_eval_rows: Sequence[QueryTrainingExample] | None = None,
    graph_cache: Mapping[str, GraphCacheItem],
    device: torch.device,
    config: Mapping[str, Any] | None = None,
    initial_model_state: Mapping[str, Any] | None = None,
    resume_training_state: Mapping[str, Any] | None = None,
    epoch_train_eval: bool = True,
    epoch_val_eval: bool = True,
    epoch_callback: Callable[[Dict[str, Any]], None] | None = None,
    step_callback: Callable[[Dict[str, Any]], None] | None = None,
    checkpoint_callback: Callable[[Dict[str, Any], LocomoNodeMemoryModel, Any, Any, Any], None] | None = None,
    step_checkpoint_callback: Callable[[Dict[str, Any], LocomoNodeMemoryModel, Any, Any, Any], None] | None = None,
    graph_error_callback: Callable[[Dict[str, Any]], None] | None = None,
    worker_event_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    resolved_config = {**DEFAULT_TRAINING_CONFIG, **dict(config or {})}
    resolved_train_eval_rows = train_eval_rows if train_eval_rows is not None else train_rows
    model = LocomoNodeMemoryModel().to(device)
    if initial_model_state:
        _load_compatible_state_dict(model, initial_model_state)
    trainable_stage_summary = _apply_trainable_stage(
        model,
        clean_text(resolved_config.get("trainable_stage", "all")) or "all",
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters resolved for node-memory training")
    l2sp_loss_weight = max(0.0, float(resolved_config.get("l2sp_loss_weight", 0.0) or 0.0))
    l2sp_reference_state: Dict[str, Tensor] = {}
    if l2sp_loss_weight > 0.0 and initial_model_state:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            reference_value = initial_model_state.get(name)
            if not isinstance(reference_value, Tensor):
                continue
            if tuple(reference_value.shape) != tuple(parameter.shape):
                continue
            l2sp_reference_state[name] = reference_value.detach().to(device=device, dtype=parameter.dtype).clone()
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(resolved_config["lr"]),
        weight_decay=float(resolved_config["weight_decay"]),
    )
    planned_total_steps = max(1, math.ceil(len(train_rows) / max(1, int(resolved_config["batch_size"]))) * max(1, int(resolved_config["epochs"])))
    configured_max_train_steps = max(0, int(resolved_config.get("max_train_steps", 0) or 0))
    total_steps = min(planned_total_steps, configured_max_train_steps) if configured_max_train_steps > 0 else planned_total_steps
    warmup_steps = max(1, int(total_steps * float(resolved_config["warmup_ratio"])))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        return max(0.05, float(total_steps - step) / float(remaining))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    amp_enabled = bool(resolved_config["amp"]) and device.type in {"cuda", "cpu"}
    scaler = _build_grad_scaler(enabled=amp_enabled and device.type == "cuda", device=device)
    log_every_steps = max(1, int(resolved_config.get("log_every_steps", 100) or 100))
    checkpoint_every_steps = max(0, int(resolved_config.get("checkpoint_every_steps", 0) or 0))
    prefetch_lookahead_batches = max(0, int(resolved_config.get("graph_prefetch_lookahead_batches", 0) or 0))
    batch_prepare_workers = max(0, int(resolved_config.get("batch_prepare_workers", 0) or 0))
    batch_prepare_lookahead_batches = max(0, int(resolved_config.get("batch_prepare_lookahead_batches", 0) or 0))
    epoch_train_eval_max_rows = max(0, int(resolved_config.get("epoch_train_eval_max_rows", 0) or 0))
    epoch_val_eval_max_rows = max(0, int(resolved_config.get("epoch_val_eval_max_rows", 0) or 0))
    batch_prepare_context: Dict[str, Any] | None = None
    batch_prepare_context_getter = getattr(graph_cache, "batch_prepare_context", None)
    if batch_prepare_workers > 0 and callable(batch_prepare_context_getter):
        candidate_context = batch_prepare_context_getter()
        if isinstance(candidate_context, Mapping):
            resolved_candidate_context = dict(candidate_context or {})
            has_conversation_specs = bool(dict(resolved_candidate_context).get("conversation_specs"))
            has_graph_dir = bool(clean_text(resolved_candidate_context.get("graph_dir", "")))
            if has_conversation_specs or has_graph_dir:
                batch_prepare_context = resolved_candidate_context
    batch_prepare_worker_pool = (
        _PreparedBatchWorkerPool(
            worker_context=batch_prepare_context,
            worker_count=batch_prepare_workers,
        )
        if batch_prepare_workers > 0 and batch_prepare_context is not None
        else None
    )

    resolved_resume_training_state = dict(resume_training_state or {})
    optimizer_state_dict = resolved_resume_training_state.get("optimizer_state_dict")
    if isinstance(optimizer_state_dict, Mapping):
        optimizer.load_state_dict(dict(optimizer_state_dict))
    scheduler_state_dict = resolved_resume_training_state.get("scheduler_state_dict")
    if isinstance(scheduler_state_dict, Mapping):
        scheduler.load_state_dict(dict(scheduler_state_dict))
    scaler_state_dict = resolved_resume_training_state.get("scaler_state_dict")
    if isinstance(scaler_state_dict, Mapping) and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(dict(scaler_state_dict))

    history = list(resolved_resume_training_state.get("history", []) or [])
    val_selection_enabled = bool(epoch_val_eval)
    best_val_loss = float(resolved_resume_training_state.get("best_val_loss", float("inf")) or float("inf"))
    best_val_selection_key: tuple[float, float, float, float, float] | None = None
    raw_best_val_selection_key = resolved_resume_training_state.get("best_val_selection_key")
    if isinstance(raw_best_val_selection_key, (list, tuple)) and len(raw_best_val_selection_key) == 5:
        best_val_selection_key = tuple(float(value or 0.0) for value in raw_best_val_selection_key)
    best_val_selection_score = float(
        resolved_resume_training_state.get("best_val_selection_score", float("-inf")) or float("-inf")
    )
    best_state: Dict[str, Any] | None = None
    patience = max(0, int(resolved_resume_training_state.get("patience", 0) or 0))
    if not val_selection_enabled:
        best_val_loss = float("inf")
        best_val_selection_key = None
        best_val_selection_score = float("-inf")
        patience = 0
    global_step = max(0, int(resolved_resume_training_state.get("global_step", 0) or 0))
    stop_training = global_step >= total_steps
    resolved_batch_size = max(1, int(resolved_config["batch_size"]))
    steps_per_epoch = max(1, math.ceil(len(train_rows) / resolved_batch_size))
    start_epoch = max(1, int(resolved_resume_training_state.get("epoch", 1) or 1))
    start_epoch_step = max(0, int(resolved_resume_training_state.get("epoch_step", 0) or 0))
    if start_epoch_step >= steps_per_epoch:
        start_epoch += start_epoch_step // steps_per_epoch
        start_epoch_step = start_epoch_step % steps_per_epoch

    for epoch in range(start_epoch, max(1, int(resolved_config["epochs"])) + 1):
        if stop_training:
            break
        model.train()
        skip_batches = start_epoch_step if epoch == start_epoch else 0
        epoch_step = skip_batches
        if batch_prepare_workers > 0 and batch_prepare_context is not None:
            batch_iter: Iterator[Any] = _iter_prepared_batches_with_workers(
                train_rows,
                graph_cache=graph_cache,
                batch_size=resolved_batch_size,
                shuffle=True,
                seed=epoch,
                worker_context=batch_prepare_context,
                worker_count=batch_prepare_workers,
                lookahead_batches=batch_prepare_lookahead_batches,
                graph_error_stage="train_batch_prepare",
                skip_batches=skip_batches,
                worker_pool=batch_prepare_worker_pool,
                prefer_completion_order=True,
                worker_event_callback=worker_event_callback,
            )
        else:
            batch_iter = _iter_batches_with_prefetch(
                graph_cache,
                train_rows,
                batch_size=resolved_batch_size,
                shuffle=True,
                seed=epoch,
                lookahead_batches=prefetch_lookahead_batches,
                skip_batches=skip_batches,
            )
        for batch_payload in batch_iter:
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(enabled=amp_enabled and device.type == "cuda", device=device):
                if isinstance(batch_payload, Mapping) and "scored_examples" in batch_payload:
                    batch_result = score_prepared_batch(
                        model,
                        batch_payload,
                        device=device,
                        loss_config=resolved_config,
                        apply_example_weights=True,
                        graph_error_callback=graph_error_callback,
                        graph_error_stage="train_batch",
                    )
                else:
                    batch_result = score_batch(
                        model,
                        batch_payload,
                        graph_cache,
                        device=device,
                        loss_config=resolved_config,
                        apply_example_weights=True,
                        graph_error_callback=graph_error_callback,
                        graph_error_stage="train_batch",
                    )
                loss = batch_result["loss"]
                if l2sp_reference_state:
                    l2sp_terms = [
                        (parameter.float() - l2sp_reference_state[name].float()).pow(2).mean()
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad and name in l2sp_reference_state
                    ]
                    if l2sp_terms:
                        l2sp_loss = torch.stack(l2sp_terms).mean()
                        loss = loss + loss.new_tensor(float(l2sp_loss_weight)) * l2sp_loss
                        batch_result["loss"] = loss
                        metrics = dict(batch_result.get("metrics", {}) or {})
                        metrics["l2sp_loss"] = round(float(l2sp_loss.detach().cpu()), 8)
                        metrics["l2sp_loss_weight"] = float(l2sp_loss_weight)
                        metrics["l2sp_parameter_count"] = int(len(l2sp_terms))
                        batch_result["metrics"] = metrics
            if int(dict(batch_result.get("metrics", {}) or {}).get("samples", 0) or 0) <= 0:
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(resolved_config["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            epoch_step += 1
            if step_callback is not None and (global_step == 1 or global_step % log_every_steps == 0 or global_step == total_steps):
                step_callback(
                    {
                        "epoch": epoch,
                        "epoch_step": epoch_step,
                        "global_step": global_step,
                        "total_steps": total_steps,
                        "batch_metrics": dict(batch_result.get("metrics", {}) or {}),
                    }
                )
            if step_checkpoint_callback is not None and checkpoint_every_steps > 0 and (
                global_step % checkpoint_every_steps == 0 or global_step == total_steps
            ):
                step_checkpoint_callback(
                    {
                        "epoch": epoch,
                        "epoch_step": epoch_step,
                        "global_step": global_step,
                        "total_steps": total_steps,
                        "batch_metrics": dict(batch_result.get("metrics", {}) or {}),
                    },
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                )
            if global_step >= total_steps:
                stop_training = True
                break
        if epoch_train_eval:
            train_summary = evaluate_examples(
                model,
                resolved_train_eval_rows,
                graph_cache,
                device=device,
                batch_size=max(1, int(resolved_config["batch_size"])),
                loss_config=resolved_config,
                prefetch_lookahead_batches=prefetch_lookahead_batches,
                graph_error_callback=graph_error_callback,
                graph_error_stage="train_eval",
                max_rows=epoch_train_eval_max_rows,
                sample_seed=epoch,
                batch_prepare_context=batch_prepare_context,
                batch_prepare_workers=batch_prepare_workers,
                batch_prepare_lookahead_batches=batch_prepare_lookahead_batches,
                batch_prepare_worker_pool=batch_prepare_worker_pool,
                worker_event_callback=worker_event_callback,
            )
        else:
            train_summary = _skipped_eval_summary(len(resolved_train_eval_rows))
        if epoch_val_eval:
            val_summary = evaluate_examples(
                model,
                val_rows,
                graph_cache,
                device=device,
                batch_size=max(1, int(resolved_config["batch_size"])),
                loss_config=resolved_config,
                prefetch_lookahead_batches=prefetch_lookahead_batches,
                graph_error_callback=graph_error_callback,
                graph_error_stage="val_eval",
                max_rows=epoch_val_eval_max_rows,
                sample_seed=epoch,
                batch_prepare_context=batch_prepare_context,
                batch_prepare_workers=batch_prepare_workers,
                batch_prepare_lookahead_batches=batch_prepare_lookahead_batches,
                batch_prepare_worker_pool=batch_prepare_worker_pool,
                worker_event_callback=worker_event_callback,
            )
        else:
            val_summary = _skipped_eval_summary(len(val_rows))
        epoch_summary = {
            "epoch": epoch,
            "global_step": global_step,
            "total_steps": total_steps,
            "train": train_summary,
            "val": val_summary,
        }
        current_val_loss = float(val_summary.get("loss", 0.0) or 0.0)
        current_val_selection_key = _checkpoint_selection_key(val_summary) if val_selection_enabled else None
        current_val_selection_score = _checkpoint_selection_score(val_summary) if val_selection_enabled else float("-inf")
        is_best = False
        if val_selection_enabled and (
            best_val_selection_key is None or current_val_selection_key is not None and current_val_selection_key > best_val_selection_key
        ):
            best_val_loss = current_val_loss
            best_val_selection_key = current_val_selection_key
            best_val_selection_score = current_val_selection_score
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            patience = 0
            is_best = True
        elif val_selection_enabled:
            patience += 1
        epoch_summary["is_best"] = is_best
        epoch_summary["val_selection_enabled"] = val_selection_enabled
        epoch_summary["patience"] = patience
        epoch_summary["best_val_loss"] = round(float(best_val_loss), 6)
        epoch_summary["val_selection_score"] = round(float(current_val_selection_score), 6)
        epoch_summary["best_val_selection_score"] = round(float(best_val_selection_score if best_val_selection_score != float("-inf") else 0.0), 6)
        epoch_summary["best_val_selection_key"] = list(best_val_selection_key) if best_val_selection_key is not None else []
        history.append(epoch_summary)
        if epoch_callback is not None:
            epoch_callback(dict(epoch_summary))
        if checkpoint_callback is not None:
            checkpoint_callback(dict(epoch_summary), model, optimizer, scheduler, scaler)
        if val_selection_enabled and not is_best and patience >= max(1, int(resolved_config["early_stopping_patience"])):
            break
        if stop_training:
            break

    if batch_prepare_worker_pool is not None:
        batch_prepare_worker_pool.close()

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    return {
        "model": model,
        "history": history,
        "best_val_loss": round(float(best_val_loss if best_val_loss != float("inf") else 0.0), 6),
        "best_val_selection_score": round(float(best_val_selection_score if best_val_selection_score != float("-inf") else 0.0), 6),
        "config": resolved_config,
        "trainable_stage": dict(trainable_stage_summary),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if hasattr(scaler, "state_dict") else {},
    }


class LoadedNodeMemoryScorer:
    def __init__(
        self,
        *,
        node_model_path: Path,
        path_model_path: Path | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        requested_device = device if device is not None else os.environ.get("TMCRA_NODE_MEMORY_DEVICE", "")
        self.device = torch.device(str(requested_device)) if str(requested_device or "").strip() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        node_payload = torch.load(Path(node_model_path), map_location=self.device, weights_only=True)
        validate_node_memory_checkpoint_payload(node_payload, context=f"LoadedNodeMemoryScorer(node_model_path={node_model_path})")
        self.model = LocomoNodeMemoryModel.from_checkpoint(node_payload, device=self.device)
        node_load_report = dict(getattr(self.model, "_checkpoint_load_report", {}) or {})
        node_load_failures = {
            key: list(node_load_report.get(key, []) or [])
            for key in ("missing_keys", "unexpected_keys", "skipped_keys")
            if node_load_report.get(key)
        }
        if node_load_failures:
            raise ValueError(f"node checkpoint is not fully compatible: {node_load_failures}")
        if path_model_path:
            path_payload = torch.load(Path(path_model_path), map_location=self.device, weights_only=True)
            validate_node_memory_checkpoint_payload(path_payload, context=f"LoadedNodeMemoryScorer(path_model_path={path_model_path})")
            path_load_report = _load_compatible_state_dict(
                self.model,
                dict(path_payload.get("state_dict", {}) or {}),
            )
            path_load_failures = {
                key: list(path_load_report.get(key, []) or [])
                for key in ("missing_keys", "unexpected_keys", "skipped_keys")
                if path_load_report.get(key)
            }
            if path_load_failures:
                raise ValueError(f"path checkpoint is not fully compatible: {path_load_failures}")
        self.model.eval()
        self._runtime_graph_cache_key = ""
        self._runtime_graph_cache_model_id = id(self.model)
        self._runtime_graph_tensors: Dict[str, Any] | None = None
        self._runtime_graph_hidden: Tensor | None = None
        self._runtime_recall_cache_key = ""
        self._runtime_recall_cache: Dict[str, Any] | None = None

    def _runtime_graph_signature(self, graph: Mapping[str, Any]) -> str:
        payload = {
            "conversation_id": clean_text(graph.get("conversation_id", "")) or "runtime",
            "nodes": [dict(node) for node in list(graph.get("nodes", []) or [])],
            "edges": [dict(edge) for edge in list(graph.get("edges", []) or [])],
            "paths": [dict(path) for path in list(graph.get("paths", []) or [])],
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _runtime_graph_state(self, graph: Mapping[str, Any]) -> tuple[Dict[str, Any], Tensor | None]:
        model_id = id(self.model)
        cache_key = self._runtime_graph_signature(graph)
        if (
            self._runtime_graph_tensors is not None
            and self._runtime_graph_cache_key == cache_key
            and self._runtime_graph_cache_model_id == model_id
        ):
            return self._runtime_graph_tensors, self._runtime_graph_hidden
        graph_tensors = tensorize_graph(graph, device=self.device)
        node_hidden: Tensor | None = None
        encode_graph = getattr(self.model, "encode_graph", None)
        if callable(encode_graph):
            with torch.no_grad():
                node_hidden = encode_graph(graph_tensors)
        self._runtime_graph_cache_key = cache_key
        self._runtime_graph_cache_model_id = model_id
        self._runtime_graph_tensors = graph_tensors
        self._runtime_graph_hidden = node_hidden
        self._runtime_recall_cache_key = ""
        self._runtime_recall_cache = None
        return graph_tensors, node_hidden

    def score_runtime(
        self,
        *,
        graph: Mapping[str, Any],
        question: str,
        question_features: Mapping[str, Any] | None = None,
        candidate_event_ids: Sequence[str] | None = None,
        rerank_top_k: int | None = None,
        event_rerank_mode: str = "matrix",
        matrix_event_top_k: int = DEFAULT_MATRIX_EVENT_TOP_K,
        event_pair_feature_mode: str = "full",
        support_path_k: int | None = None,
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        profile_enabled = str(os.environ.get("TMCRA_NODE_RUNTIME_PROFILE", "")).strip().lower() in {
            "1", "true", "yes", "on",
        }

        def profile_clock() -> float:
            if profile_enabled and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            return time.perf_counter()

        def materialize(values: Tensor, *, sigmoid: bool = False) -> List[float]:
            resolved = torch.sigmoid(values) if sigmoid else values
            return [float(value) for value in resolved.detach().float().cpu().tolist()]

        def score_map(ids: Sequence[str], values: Tensor, *, sigmoid: bool = False) -> Dict[str, float]:
            return dict(zip(ids, materialize(values, sigmoid=sigmoid)))

        runtime_started = profile_clock()
        graph_tensors, node_hidden = self._runtime_graph_state(graph)
        graph_state_finished = profile_clock()
        explicit_runtime_candidates = candidate_event_ids is not None
        runtime_candidate_event_ids = [
            clean_text(item)
            for item in (candidate_event_ids if candidate_event_ids is not None else graph_tensors.get("event_node_ids", []))
            if clean_text(item)
        ]
        question_feature_payload = dict(question_features or extract_question_features(question))
        resolved_event_rerank_mode = _normalize_event_rerank_mode(event_rerank_mode)
        resolved_matrix_event_top_k = max(1, int(matrix_event_top_k or DEFAULT_MATRIX_EVENT_TOP_K))
        recall_cache_payload = {
            "graph": self._runtime_graph_cache_key,
            "question": clean_text(question),
            "question_features": question_feature_payload,
            "event_pair_feature_mode": _normalize_event_pair_feature_mode(event_pair_feature_mode),
        }
        recall_cache_key = hashlib.sha256(
            json.dumps(
                recall_cache_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        recall_cache_hit = bool(
            self._runtime_recall_cache is not None
            and self._runtime_recall_cache_key == recall_cache_key
        )
        first_pass_example = QueryTrainingExample(
            conversation_id=clean_text(graph.get("conversation_id", "runtime")),
            question_id="runtime:recall",
            question=question,
            question_features=question_feature_payload,
            candidate_event_ids=runtime_candidate_event_ids,
            positive_event_ids=[],
            positive_path_ids=[],
            positive_time_node_ids=[],
            negative_event_ids=[],
            answer_targets={"answer_type": "event_text"},
            answer_plan_targets={},
            temporal_target={
                "is_temporal": bool(question_feature_payload.get("is_temporal", False)),
                "question_is_temporal": bool(question_feature_payload.get("is_temporal", False)),
                "use_temporal_head": bool(question_feature_payload.get("is_temporal", False)),
            },
            event_catalog_size=len(runtime_candidate_event_ids),
            metadata={},
        )
        recall_started = profile_clock()
        if recall_cache_hit:
            recall_outputs = dict(self._runtime_recall_cache or {})
        else:
            with torch.no_grad():
                if node_hidden is not None and hasattr(self.model, "score_example_with_graph_encoding"):
                    recall_outputs = _call_with_supported_kwargs(
                        self.model.score_example_with_graph_encoding,
                        graph_tensors,
                        node_hidden,
                        first_pass_example,
                        event_rerank_mode="single",
                        matrix_event_top_k=resolved_matrix_event_top_k,
                        event_pair_feature_mode=event_pair_feature_mode,
                        recall_only=True,
                    )
                else:
                    recall_outputs = _call_with_supported_kwargs(
                        self.model.score_example,
                        graph_tensors,
                        first_pass_example,
                        event_rerank_mode="single",
                        matrix_event_top_k=resolved_matrix_event_top_k,
                        event_pair_feature_mode=event_pair_feature_mode,
                    )
            if isinstance(recall_outputs.get("recall_pair_features"), Tensor):
                self._runtime_recall_cache_key = recall_cache_key
                self._runtime_recall_cache = dict(recall_outputs)
        recall_finished = profile_clock()
        recall_event_scores = score_map(
            recall_outputs["recall_event_ids"],
            recall_outputs["recall_event_logits"],
            sigmoid=True,
        )
        recall_event_ids = [
            event_id
            for event_id, _ in sorted(recall_event_scores.items(), key=lambda item: (-float(item[1]), item[0]))
        ]
        resolved_rerank_top_k = max(
            1,
            min(
                len(runtime_candidate_event_ids),
                int(rerank_top_k or DEFAULT_RUNTIME_RECALL_TOP_K),
            ),
        )
        if explicit_runtime_candidates:
            # A caller-provided candidate set is a contract, not just a hint. The
            # oracle/probe path relies on this to distinguish recall failure from
            # rerank failure; silently dropping explicit candidates makes the
            # diagnostic look worse than the actual reranker.
            rerank_candidate_event_ids = dedupe_texts(
                [
                    *runtime_candidate_event_ids,
                    *(recall_event_ids[:resolved_rerank_top_k] if recall_event_ids else []),
                ]
            )
        else:
            rerank_candidate_event_ids = (
                recall_event_ids[:resolved_rerank_top_k]
                if recall_event_ids
                else list(runtime_candidate_event_ids[:resolved_rerank_top_k])
            )
        rerank_example = QueryTrainingExample(
            conversation_id=clean_text(graph.get("conversation_id", "runtime")),
            question_id="runtime:rerank",
            question=question,
            question_features=question_feature_payload,
            candidate_event_ids=rerank_candidate_event_ids,
            positive_event_ids=[],
            positive_path_ids=[],
            positive_time_node_ids=[],
            negative_event_ids=[],
            answer_targets={"answer_type": "event_text"},
            answer_plan_targets={},
            temporal_target={
                "is_temporal": bool(question_feature_payload.get("is_temporal", False)),
                "question_is_temporal": bool(question_feature_payload.get("is_temporal", False)),
                "use_temporal_head": bool(question_feature_payload.get("is_temporal", False)),
            },
            event_catalog_size=len(runtime_candidate_event_ids),
            metadata={},
        )
        rerank_started = profile_clock()
        with torch.no_grad():
            if node_hidden is not None and hasattr(self.model, "score_example_with_graph_encoding"):
                outputs = _call_with_supported_kwargs(
                    self.model.score_example_with_graph_encoding,
                    graph_tensors,
                    node_hidden,
                    rerank_example,
                    event_rerank_mode=resolved_event_rerank_mode,
                    matrix_event_top_k=resolved_matrix_event_top_k,
                    event_pair_feature_mode=event_pair_feature_mode,
                    precomputed_recall=(
                        recall_outputs
                        if isinstance(recall_outputs.get("recall_pair_features"), Tensor)
                        else None
                    ),
                )
            else:
                outputs = _call_with_supported_kwargs(
                    self.model.score_example,
                    graph_tensors,
                    rerank_example,
                    event_rerank_mode=resolved_event_rerank_mode,
                    matrix_event_top_k=resolved_matrix_event_top_k,
                    event_pair_feature_mode=event_pair_feature_mode,
                )
        rerank_finished = profile_clock()
        materialization_started = profile_clock()
        base_event_logits = outputs.get("base_event_logits", outputs.get("event_logits", torch.tensor([], dtype=torch.float32, device=self.device)))
        rerank_event_logits = outputs.get("rerank_event_logits", outputs.get("event_logits", torch.tensor([], dtype=torch.float32, device=self.device)))
        distractor_event_logits = outputs.get("distractor_event_logits", rerank_event_logits)
        calibrated_event_logits = outputs.get("calibrated_event_logits", outputs.get("event_logits", torch.tensor([], dtype=torch.float32, device=self.device)))
        matrix_event_ids = list(outputs.get("matrix_event_ids", []) or [])
        matrix_event_delta_logits = outputs.get(
            "matrix_event_delta_logits",
            outputs["event_logits"].new_zeros((0,)) if isinstance(outputs.get("event_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        event_distractor_logits = outputs.get(
            "event_distractor_logits",
            outputs["event_logits"].new_zeros((0,)) if isinstance(outputs.get("event_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        event_distractor_delta_logits = outputs.get(
            "event_distractor_delta_logits",
            outputs["event_logits"].new_zeros((0,)) if isinstance(outputs.get("event_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        event_fusion_delta_logits = outputs.get(
            "event_fusion_delta_logits",
            outputs["event_logits"].new_zeros((0,)) if isinstance(outputs.get("event_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        event_tunnel_support_logits = outputs.get(
            "event_tunnel_support_logits",
            outputs["event_logits"].new_zeros((0,)) if isinstance(outputs.get("event_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        event_tunnel_delta_logits = outputs.get(
            "event_tunnel_delta_logits",
            outputs["event_logits"].new_zeros((0,)) if isinstance(outputs.get("event_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        base_path_logits = outputs.get("base_path_logits", outputs.get("path_logits", torch.tensor([], dtype=torch.float32, device=self.device)))
        rerank_path_logits = outputs.get("rerank_path_logits", outputs.get("path_logits", torch.tensor([], dtype=torch.float32, device=self.device)))
        calibrated_path_logits = outputs.get("calibrated_path_logits", outputs.get("path_logits", torch.tensor([], dtype=torch.float32, device=self.device)))
        matrix_path_ids = list(outputs.get("matrix_path_ids", []) or [])
        matrix_path_delta_logits = outputs.get(
            "matrix_path_delta_logits",
            outputs["path_logits"].new_zeros((0,)) if isinstance(outputs.get("path_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        path_fusion_delta_logits = outputs.get(
            "path_fusion_delta_logits",
            outputs["path_logits"].new_zeros((0,)) if isinstance(outputs.get("path_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        path_tunnel_support_logits = outputs.get(
            "path_tunnel_support_logits",
            outputs["path_logits"].new_zeros((0,)) if isinstance(outputs.get("path_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        path_tunnel_delta_logits = outputs.get(
            "path_tunnel_delta_logits",
            outputs["path_logits"].new_zeros((0,)) if isinstance(outputs.get("path_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        event_reverse_values = outputs.get(
            "event_reverse_scores",
            outputs["event_logits"].new_zeros((0,)) if isinstance(outputs.get("event_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        event_boundary_values = outputs.get(
            "event_boundary_scores",
            outputs["event_logits"].new_zeros((0,)) if isinstance(outputs.get("event_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        event_reverse_relation_values = outputs.get(
            "event_reverse_relations",
            outputs["event_logits"].new_zeros((0,)) if isinstance(outputs.get("event_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        path_reverse_values = outputs.get(
            "path_reverse_scores",
            outputs["path_logits"].new_zeros((0,)) if isinstance(outputs.get("path_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        path_boundary_values = outputs.get(
            "path_boundary_scores",
            outputs["path_logits"].new_zeros((0,)) if isinstance(outputs.get("path_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        path_reverse_relation_values = outputs.get(
            "path_reverse_relations",
            outputs["path_logits"].new_zeros((0,)) if isinstance(outputs.get("path_logits"), Tensor) else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        candidate_event_ids = outputs["candidate_event_ids"]
        base_event_scores = score_map(candidate_event_ids, base_event_logits, sigmoid=True)
        rerank_event_scores = score_map(candidate_event_ids, rerank_event_logits, sigmoid=True)
        distractor_event_scores = score_map(candidate_event_ids, event_distractor_logits, sigmoid=True)
        distractor_adjusted_event_scores = score_map(candidate_event_ids, distractor_event_logits, sigmoid=True)
        calibrated_event_scores = score_map(candidate_event_ids, calibrated_event_logits, sigmoid=True)
        event_scores = score_map(candidate_event_ids, outputs["event_logits"], sigmoid=True)
        matrix_event_scores = score_map(matrix_event_ids, matrix_event_delta_logits, sigmoid=True)
        event_distractor_delta_scores = score_map(candidate_event_ids, event_distractor_delta_logits)
        event_fusion_delta_scores = score_map(candidate_event_ids, event_fusion_delta_logits, sigmoid=True)
        event_tunnel_support_scores = score_map(candidate_event_ids, event_tunnel_support_logits, sigmoid=True)
        event_tunnel_delta_scores = score_map(candidate_event_ids, event_tunnel_delta_logits)
        tri_maze_event_reverse_scores = score_map(candidate_event_ids, event_reverse_values)
        tri_maze_event_boundary_scores = score_map(candidate_event_ids, event_boundary_values)
        tri_maze_event_reverse_relations = score_map(candidate_event_ids, event_reverse_relation_values)
        candidate_path_ids = outputs["candidate_path_ids"]
        base_path_scores = score_map(candidate_path_ids, base_path_logits, sigmoid=True)
        rerank_path_scores = score_map(candidate_path_ids, rerank_path_logits, sigmoid=True)
        calibrated_path_scores = score_map(candidate_path_ids, calibrated_path_logits, sigmoid=True)
        path_scores = score_map(candidate_path_ids, outputs["path_logits"], sigmoid=True)
        matrix_path_scores = score_map(matrix_path_ids, matrix_path_delta_logits, sigmoid=True)
        path_fusion_delta_scores = score_map(candidate_path_ids, path_fusion_delta_logits, sigmoid=True)
        path_tunnel_support_scores = score_map(candidate_path_ids, path_tunnel_support_logits, sigmoid=True)
        path_tunnel_delta_scores = score_map(candidate_path_ids, path_tunnel_delta_logits)
        tri_maze_path_reverse_scores = score_map(candidate_path_ids, path_reverse_values)
        tri_maze_path_boundary_scores = score_map(candidate_path_ids, path_boundary_values)
        tri_maze_path_reverse_relations = score_map(candidate_path_ids, path_reverse_relation_values)
        temporal_node_scores: Dict[str, float] = {}
        for node_id, score in zip(
            outputs["candidate_temporal_node_ids"],
            materialize(outputs["temporal_logits"], sigmoid=True),
        ):
            temporal_node_scores[node_id] = max(score, float(temporal_node_scores.get(node_id, 0.0)))
        answer_type_values = materialize(F.softmax(outputs["answer_type_logits"], dim=-1))
        answer_type_scores = {
            answer_type: answer_type_values[index]
            for answer_type, index in ANSWER_TYPE_TO_ID.items()
        }
        answer_plan_logits = outputs.get(
            "answer_plan_logits",
            outputs["event_logits"].new_zeros((0, len(ANSWER_PLAN_OUTPUTS)))
            if isinstance(outputs.get("event_logits"), Tensor)
            else torch.tensor([], dtype=torch.float32, device=self.device),
        )
        answer_plan_scores: Dict[str, Dict[str, float]] = {}
        if isinstance(answer_plan_logits, Tensor) and answer_plan_logits.numel() > 0:
            for role_name, role_index in ANSWER_PLAN_OUTPUT_TO_ID.items():
                answer_plan_scores[role_name] = score_map(
                    candidate_event_ids,
                    answer_plan_logits[:, role_index],
                    sigmoid=True,
                )
        memory_router_logits = outputs.get("memory_router_logits", torch.tensor([], dtype=torch.float32, device=self.device))
        memory_router_scores = score_map(MEMORY_ROUTER_LAYERS, memory_router_logits, sigmoid=True)
        memory_router_top_layers = [
            layer
            for layer, score in sorted(memory_router_scores.items(), key=lambda item: (-float(item[1]), item[0]))
            if float(score) >= 0.5
        ]
        if not memory_router_top_layers and memory_router_scores:
            memory_router_top_layers = [
                max(memory_router_scores.items(), key=lambda item: float(item[1]))[0]
            ]
        event_distractor_enabled = _output_projection_has_signal(getattr(self.model, "event_distractor_head", None))
        event_calibration_enabled = _output_projection_has_signal(getattr(self.model, "event_calibration_head", None))
        path_calibration_enabled = _output_projection_has_signal(getattr(self.model, "path_calibration_head", None))
        event_tunnel_enabled = _output_projection_has_signal(getattr(self.model, "event_tunnel_head", None))
        path_tunnel_enabled = _output_projection_has_signal(getattr(self.model, "path_tunnel_head", None))
        final_event_fusion_enabled = _output_projection_has_signal(getattr(self.model, "final_event_fusion_head", None))
        final_path_fusion_enabled = _output_projection_has_signal(getattr(self.model, "final_path_fusion_head", None))
        event_fusion_enabled = bool(event_calibration_enabled or event_tunnel_enabled or final_event_fusion_enabled)
        path_fusion_enabled = bool(path_calibration_enabled or path_tunnel_enabled or final_path_fusion_enabled)
        decision_fusion_enabled = bool(event_tunnel_enabled or path_tunnel_enabled or final_event_fusion_enabled or final_path_fusion_enabled)
        resolved_support_path_k = max(1, int(support_path_k or 3))
        resolved_top_k = max(1, int(top_k or resolved_support_path_k))
        path_lookup = {
            clean_text(path.get("id", "")): dict(path)
            for path in list(graph_tensors.get("paths", []) or [])
            if clean_text(path.get("id", ""))
        }
        path_model_scores = dict(path_scores)
        path_chain_extended_scores: Dict[str, float] = {}
        path_chain_extension_delta_scores: Dict[str, float] = {}
        ranked_final_path_ids = _sorted_score_ids(path_scores)
        ranked_final_event_ids = _sorted_score_ids(event_scores)
        selected_path_ids: List[str] = []
        selected_event_ids: List[str] = []
        decision_score_source = ""
        if decision_fusion_enabled:
            selected_path_limit = min(len(ranked_final_path_ids), max(1, min(resolved_support_path_k, resolved_top_k)))
            selected_path_ids = list(ranked_final_path_ids[:selected_path_limit])
            selected_event_ids = dedupe_texts(
                [
                    *[
                        clean_text(path_lookup.get(path_id, {}).get("event_id", ""))
                        for path_id in selected_path_ids
                        if clean_text(path_lookup.get(path_id, {}).get("event_id", ""))
                    ],
                    *_sorted_score_ids(event_scores, limit=max(resolved_support_path_k * 2, resolved_top_k)),
                ],
                max_items=max(resolved_support_path_k * 2, resolved_top_k),
            )
            if final_path_fusion_enabled and selected_path_ids:
                decision_score_source = "learned_final_path_fusion"
            elif final_event_fusion_enabled:
                decision_score_source = "learned_final_event_fusion"
            elif path_tunnel_enabled and selected_path_ids:
                decision_score_source = "learned_path_tunnel"
            elif event_tunnel_enabled:
                decision_score_source = "learned_event_tunnel"
            if path_chain_extension_delta_scores and any(
                float(path_chain_extension_delta_scores.get(path_id, 0.0) or 0.0) > 0.0
                for path_id in selected_path_ids
            ):
                decision_score_source = (
                    f"{decision_score_source}+chain_extension"
                    if decision_score_source
                    else "chain_extension"
                )
        focused_answer_type = _dominant_answer_type_from_scores(
            answer_type_scores,
            question_is_temporal=bool(question_feature_payload.get("is_temporal", False)),
        )
        materialization_finished = profile_clock()
        return {
            "runtime_profile": {
                "enabled": bool(profile_enabled),
                "graph_state_sec": round(graph_state_finished - runtime_started, 6),
                "recall_sec": round(recall_finished - recall_started, 6),
                "rerank_sec": round(rerank_finished - rerank_started, 6),
                "materialization_sec": round(materialization_finished - materialization_started, 6),
                "total_sec": round(materialization_finished - runtime_started, 6),
                "recall_cache_hit": bool(recall_cache_hit),
                "runtime_candidate_event_count": len(runtime_candidate_event_ids),
                "rerank_candidate_event_count": len(rerank_candidate_event_ids),
                "recall_event_count": len(recall_outputs.get("recall_event_ids", []) or []),
                "candidate_path_count": len(outputs.get("candidate_path_ids", []) or []),
            },
            "recall_event_scores": recall_event_scores,
            "recall_event_ids": recall_event_ids,
            "rerank_candidate_event_ids": rerank_candidate_event_ids,
            "base_event_scores": base_event_scores,
            "rerank_event_scores": rerank_event_scores,
            "event_distractor_scores": distractor_event_scores,
            "distractor_event_scores": distractor_adjusted_event_scores,
            "calibrated_event_scores": calibrated_event_scores,
            "matrix_event_scores": matrix_event_scores,
            "event_distractor_delta_scores": event_distractor_delta_scores,
            "event_fusion_delta_scores": event_fusion_delta_scores,
            "event_tunnel_support_scores": event_tunnel_support_scores,
            "event_tunnel_delta_scores": event_tunnel_delta_scores,
            "tri_maze_event_reverse_scores": tri_maze_event_reverse_scores,
            "tri_maze_event_boundary_scores": tri_maze_event_boundary_scores,
            "tri_maze_event_reverse_relations": tri_maze_event_reverse_relations,
            "matrix_rerank_event_ids": matrix_event_ids,
            "matrix_enabled": bool(resolved_event_rerank_mode == "matrix" and matrix_event_ids),
            "matrix_path_enabled": bool(matrix_path_ids),
            "event_distractor_enabled": bool(event_distractor_enabled),
            "event_calibration_enabled": bool(event_calibration_enabled),
            "path_calibration_enabled": bool(path_calibration_enabled),
            "event_tunnel_enabled": bool(event_tunnel_enabled),
            "path_tunnel_enabled": bool(path_tunnel_enabled),
            "final_event_fusion_enabled": bool(final_event_fusion_enabled),
            "final_path_fusion_enabled": bool(final_path_fusion_enabled),
            "decision_fusion_enabled": bool(decision_fusion_enabled),
            "decision_score_source": decision_score_source,
            "event_fusion_enabled": bool(event_fusion_enabled),
            "path_fusion_enabled": bool(path_fusion_enabled),
            "fusion_enabled": bool(event_fusion_enabled or path_fusion_enabled),
            "path_chain_extension_enabled": bool(path_chain_extension_delta_scores),
            "selected_event_ids": list(selected_event_ids),
            "selected_path_ids": list(selected_path_ids),
            "event_scores": event_scores,
            "path_model_scores": dict(path_model_scores),
            "base_path_scores": base_path_scores,
            "rerank_path_scores": rerank_path_scores,
            "calibrated_path_scores": calibrated_path_scores,
            "matrix_path_scores": matrix_path_scores,
            "matrix_path_rerank_ids": matrix_path_ids,
            "path_fusion_delta_scores": path_fusion_delta_scores,
            "path_tunnel_support_scores": path_tunnel_support_scores,
            "path_tunnel_delta_scores": path_tunnel_delta_scores,
            "path_chain_extension_delta_scores": path_chain_extension_delta_scores,
            "path_chain_extended_scores": path_chain_extended_scores,
            "tri_maze_path_reverse_scores": tri_maze_path_reverse_scores,
            "tri_maze_path_boundary_scores": tri_maze_path_boundary_scores,
            "tri_maze_path_reverse_relations": tri_maze_path_reverse_relations,
            "path_scores": path_scores,
            "temporal_scores": temporal_node_scores,
            "answer_type_scores": answer_type_scores,
            "answer_plan_scores": answer_plan_scores,
            "memory_router_scores": memory_router_scores,
            "memory_router_top_layers": memory_router_top_layers,
            "memory_router_layers": list(MEMORY_ROUTER_LAYERS),
            "focused_answer_type": focused_answer_type,
            "event_pair_feature_mode": _normalize_event_pair_feature_mode(event_pair_feature_mode),
        }
