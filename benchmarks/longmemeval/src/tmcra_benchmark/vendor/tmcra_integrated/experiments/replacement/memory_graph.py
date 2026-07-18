from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence
import urllib.error
import urllib.request

from .memory_profiles import TMCRAProfile
from .profile_layer import (
    PROFILE_AGGREGATE_CATEGORY,
    PROFILE_AGGREGATE_SOURCE_KIND,
    PROFILE_CLUSTER_CATEGORY,
    PROFILE_CLUSTER_SOURCE_KIND,
    build_profile_aggregate_metadata,
    build_profile_cluster_metadata,
    is_profile_layer_record,
    profile_aggregate_slot_key,
    profile_cluster_similarity,
    profile_cluster_slot_key,
    profile_edge_score,
    profile_query_score_delta,
)


_HISTORY_MARKERS = (
    "previous",
    "earlier",
    "before",
    "used to",
    "history",
    "historical",
    "old",
    "prior",
    "earliest",
    "之前",
    "以前",
    "历史",
    "上一版",
    "旧",
)

_GOAL_MARKERS = ("goal", "mission", "objective", "target", "primary goal", "目标")
_CONSTRAINT_MARKERS = ("constraint", "must", "forbid", "policy", "限制", "约束", "必须")
_PREFERENCE_MARKERS = ("preference", "prefer", "default", "mode", "偏好", "默认")
_TERM_MARKERS = ("term", "terminology", "alias", "mean", "definition", "术语", "定义")
_STAGE_MARKERS = ("stage", "phase", "status", "state", "阶段", "状态")
_PATH_MARKERS = ("path", "route", "connect", "why", "tied", "链路", "路径", "连接")
_NOISE_MARKERS = ("noise turn", "noise note", "routine chatter", "random artifact", "distractor")
_NEGATION_MARKERS = ("not", "never", "no ", "should not", "不要", "不应", "不是", "无")


_PUBLIC_MONTH_MARKERS = (
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
_PUBLIC_QUERY_FRAME_TOKENS = {
    "what",
    "when",
    "which",
    "where",
    "who",
    "why",
    "how",
    "did",
    "does",
    "do",
    "is",
    "are",
    "am",
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
}
_PUBLIC_TEMPORAL_REFERENCE_RE = re.compile(
    r"\b(?:when|date|day|month|year|today|tomorrow|yesterday|tonight|morning|afternoon|evening|night|week(?:end)?|last|next|before|after)\b",
    re.IGNORECASE,
)
_PUBLIC_PROFILE_QUERY_RE = re.compile(
    r"^\s*(?:who\s+is|what\s+(?:is|are|was|were)|(?:what|where)\s+(?:does|do|is|are|was|were|has|have|had)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'s\b)",
    re.IGNORECASE,
)
_PUBLIC_PLANNED_STATUS_RE = re.compile(
    r"\b(?:will|going\s+to|plan(?:ning)?\s+to|scheduled\s+to|hoping\s+to|preparing\s+to)\b",
    re.IGNORECASE,
)
_PUBLIC_CURRENT_STATUS_RE = re.compile(r"\b(?:currently|now|still)\b|\b(?:is|are|am)\s+\w+ing\b", re.IGNORECASE)
_PUBLIC_PAST_STATUS_RE = re.compile(r"\b(?:did|was|were|had|has)\b", re.IGNORECASE)
_PUBLIC_MEMORY_SUBJECT_QUERY_PATTERNS = (
    re.compile(
        r"\bmy\s+(?P<subject>[a-z0-9][a-z0-9\s'/_-]{1,100}?)\s+(?:is|are|was|were)\s+(?:now|currently)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+(?:is|are|was|were)\s+my\s+(?P<subject>[a-z0-9][a-z0-9\s'/_-]{1,100}?)\s+(?:right\s+now|now|currently)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmy\s+(?P<subject>[a-z0-9][a-z0-9\s'/_-]{1,100}?)\s+(?:right\s+now|now|currently)\b",
        re.IGNORECASE,
    ),
)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


_DEFAULT_TMCRA_PROFILE = TMCRAProfile()


# Override legacy mojibake markers with clean Chinese variants.
_HISTORY_MARKERS = (
    "previous",
    "earlier",
    "before",
    "used to",
    "history",
    "historical",
    "old",
    "prior",
    "earliest",
    "之前",
    "以前",
    "历史",
    "上一版",
    "旧",
    "原来",
)
_GOAL_MARKERS = ("goal", "mission", "objective", "target", "primary goal", "目标")
_CONSTRAINT_MARKERS = ("constraint", "must", "forbid", "policy", "限制", "约束", "必须", "不要", "不能")
_PREFERENCE_MARKERS = ("preference", "prefer", "default", "mode", "偏好", "默认", "优先")
_TERM_MARKERS = ("term", "terminology", "alias", "mean", "definition", "术语", "定义", "别名", "叫做", "指的是")
_STAGE_MARKERS = ("stage", "phase", "status", "state", "阶段", "状态", "进度")
_PATH_MARKERS = ("path", "route", "connect", "why", "tied", "链路", "路径", "连接")
_NEGATION_MARKERS = ("not", "never", "no ", "should not", "不要", "不应", "不是", "不能")


_INACTIVE_HISTORY_MARKERS = (
    "noise",
    "just noise",
    "noise note",
    "inactive",
    "stay inactive",
    "should stay inactive",
    "keep inactive",
    "remain inactive",
    "should be ignored",
    "ignore this",
    "ignore that",
    "\u566a\u58f0",
    "\u4e0d\u6fc0\u6d3b",
    "\u4fdd\u6301\u4e0d\u6fc0\u6d3b",
    "\u4e0d\u5e94\u751f\u6548",
    "\u5ffd\u7565",
)

def _normalize(value: Any) -> str:
    return _clean_text(value).lower()


def _tokenize(value: Any) -> List[str]:
    text = _normalize(value)
    if not text:
        return []
    english = re.findall(r"[a-z0-9_]+", text)
    if english:
        return english
    return [char for char in text if char.strip()]


def _tokenize(value: Any) -> List[str]:
    text = _normalize(value)
    if not text:
        return []
    english = re.findall(r"[a-z0-9_]+", text)
    cjk = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    if english or cjk:
        return _dedupe([*english, *cjk])
    return [char for char in text if char.strip()]


def _dedupe(items: Iterable[Any], *, max_items: int | None = None) -> List[str]:
    values: List[str] = []
    seen = set()
    for item in items:
        text = _clean_text(item)
        if not text:
            continue
        key = _normalize(text)
        if key in seen:
            continue
        seen.add(key)
        values.append(text)
        if max_items is not None and len(values) >= max_items:
            break
    return values


def _contains_any(text: str, markers: Sequence[str]) -> bool:
    lowered = _normalize(text)
    return any(marker in lowered for marker in markers)


def _public_query_requests_temporal_fact(query: str) -> bool:
    lowered = _normalize(query)
    first_token = _tokenize(query)[:1]
    if first_token and first_token[0] == "when":
        return True
    if _PUBLIC_TEMPORAL_REFERENCE_RE.search(lowered):
        return True
    if any(marker in lowered for marker in _PUBLIC_MONTH_MARKERS):
        return True
    if re.search(r"\b(?:mon|tue|wed|thu|fri|sat|sun)(?:day)?\b", lowered):
        return True
    return bool(re.search(r"\b\d{4}\b", lowered))


def _public_event_tokens(tokens: Iterable[str], *, speaker_candidates: Iterable[str] = ()) -> set[str]:
    normalized_speakers = {_normalize(item) for item in speaker_candidates if _clean_text(item)}
    content_tokens: set[str] = set()
    skipped_frame_tokens = 0
    for token in tokens:
        normalized = _normalize(token)
        if not normalized or normalized.isdigit() or len(normalized) <= 2 or normalized in normalized_speakers:
            continue
        if skipped_frame_tokens < 2 and normalized in _PUBLIC_QUERY_FRAME_TOKENS:
            skipped_frame_tokens += 1
            continue
        content_tokens.add(normalized)
    return content_tokens


def _public_query_requests_profile_fact(query: str, speaker_candidates: Iterable[str]) -> bool:
    return bool(list(speaker_candidates or [])) and bool(_PUBLIC_PROFILE_QUERY_RE.search(_clean_text(query)))


def _public_query_semantic_slot(query: str) -> str:
    speaker_candidates = _public_query_speakers(query)
    if _public_query_requests_temporal_fact(query):
        return "event_time"
    if _public_query_requests_profile_fact(query, speaker_candidates):
        return "profile"
    return "event"


def _public_query_target_status(query: str) -> str:
    clean_query = _clean_text(query)
    lowered = _normalize(query)
    if _PUBLIC_PLANNED_STATUS_RE.search(clean_query):
        return "planned"
    if _PUBLIC_CURRENT_STATUS_RE.search(clean_query):
        return "current"
    if _PUBLIC_PAST_STATUS_RE.search(lowered):
        return "past"
    return ""


def _public_query_time_granularity(query: str) -> str:
    lowered = _normalize(query)
    if "month" in lowered or any(marker in lowered for marker in _PUBLIC_MONTH_MARKERS):
        return "month"
    if "year" in lowered:
        return "year"
    if any(marker in lowered for marker in ("date", "day", "when")):
        return "day_or_coarse"
    return ""


def _public_query_speakers(query: str) -> set[str]:
    candidates = {
        _normalize(match.group(1).rstrip("'s"))
        for match in re.finditer(r"\b([A-Z][a-z]+(?:'s)?)\b", _clean_text(query))
    }
    return {
        item
        for item in candidates
        if item not in {"what", "when", "which", "who", "where", "why", "how"}
    }


def _public_subject_signature(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _normalize(value)).strip("_")


def _public_query_subject(query: str) -> str:
    clean_query = _clean_text(query)
    for pattern in _PUBLIC_MEMORY_SUBJECT_QUERY_PATTERNS:
        match = pattern.search(clean_query)
        if not match:
            continue
        subject = _clean_text(match.group("subject")).strip(" .,:;!?\"'")
        subject = re.sub(r"\b(?:right|now|currently|current)\b$", "", subject, flags=re.IGNORECASE).strip(" .,:;!?\"'")
        if len(_tokenize(subject)) >= 2:
            return subject
    return ""


_DEPTH_LAYERS = {
    "core_view",
    "profile",
    "deep_view",
    "mechanism",
    "risk",
    "metric",
    "product_view",
    "evidence",
    "question",
}
_DEPTH_LAYER_ORDER = {
    "core_view": 0,
    "profile": 0,
    "deep_view": 1,
    "risk": 2,
    "mechanism": 3,
    "metric": 4,
    "product_view": 5,
    "evidence": 6,
    "question": 7,
}


def _record_subject_signature_from_parts(slot_key: Any, metadata: Mapping[str, Any]) -> str:
    explicit = _clean_text(metadata.get("subject_signature", ""))
    if explicit:
        return _public_subject_signature(explicit)
    subject = _clean_text(metadata.get("subject", ""))
    if subject:
        return _public_subject_signature(subject)
    canonical_slot_key = _clean_text(metadata.get("canonical_slot_key", "") or slot_key)
    if ".subject." in canonical_slot_key:
        return _public_subject_signature(canonical_slot_key.split(".subject.", 1)[-1])
    if ".subject." in _clean_text(slot_key):
        return _public_subject_signature(_clean_text(slot_key).split(".subject.", 1)[-1])
    return _public_subject_signature(_slot_group_key(_clean_text(slot_key)))


def _infer_depth_layer(*, category: Any, relation: Any, slot_key: Any, value: Any, metadata: Mapping[str, Any]) -> str:
    explicit = _normalize(metadata.get("depth_layer", "") or metadata.get("memory_depth_layer", ""))
    if explicit in _DEPTH_LAYERS:
        return explicit
    semantic_slot = _normalize(metadata.get("semantic_slot", "") or metadata.get("profile_type", ""))
    category_text = _normalize(category)
    relation_text = _normalize(relation)
    slot_text = _normalize(slot_key)
    if is_profile_layer_record(category=category_text, semantic_slot=semantic_slot, metadata=metadata):
        return "profile"
    combined = _normalize(
        " ".join(
            [
                str(value or ""),
                str(metadata.get("source_span", "")),
                str(metadata.get("raw_text", "")),
                str(metadata.get("source_turn_text", "")),
                str(semantic_slot),
                str(category_text),
                str(relation_text),
                str(slot_text),
            ]
        )
    )
    if category_text == "question" or "question" in relation_text or semantic_slot in {"goal", "information_needed"}:
        return "question"
    if any(marker in combined for marker in ("deeper", "deep view", "deep safety", "accountable memory", "mature", "refined")):
        if any(marker in combined for marker in ("dashboard", "product", "enterprise adoption", "adoption view")):
            return "product_view"
        return "deep_view"
    if any(marker in combined for marker in ("metric", "measure", "score", "benchmark", "evaluation_metric", "recovery rate", "success rate")):
        return "metric"
    if any(marker in combined for marker in ("dashboard", "product", "enterprise adoption", "adoption view", "deployment cost")):
        return "product_view"
    if any(marker in combined for marker in ("risk", "failure", "outdated", "unverified", "corrupt", "silent failure", "safety violation")):
        return "risk"
    if any(marker in combined for marker in ("mechanism", "checksum", "retrieval", "filter", "pipeline", "interface", "state transition")):
        return "mechanism"
    if re.search(r"\bmy\b.{0,80}\bview\s+is\s+now\b", combined) or re.search(r"\bmy\s+view\s+is\s+now\b", combined):
        return "core_view"
    if any(marker in combined for marker in ("mechanism", "checksum", "logging", "audit trail", "metadata", "retrieval", "filter", "pipeline", "interface", "tool call", "write", "state transition")):
        return "mechanism"
    if category_text in {"fact", "profile", "evidence"} or "source_turn" in relation_text:
        return "evidence"
    return "core_view"


def _depth_edge_type(source_layer: str, target_layer: str, source_text: str, target_text: str) -> str:
    combined = _normalize(f"{source_text} {target_text}")
    if any(marker in combined for marker in ("instead", "no longer", "not anymore", "replace", "rather than")):
        return "contrasts"
    if source_layer == "deep_view" and target_layer in {"core_view", "evidence", "mechanism", "product_view"}:
        return "refines"
    if source_layer == "mechanism" and target_layer in {"core_view", "deep_view", "risk", "metric"}:
        return "operationalizes"
    if source_layer == "metric":
        return "measures"
    if source_layer == "product_view":
        return "productizes"
    if source_layer == "risk":
        return "qualifies"
    if source_layer == target_layer:
        return "reframes"
    return "co_mentions"


def _public_signature_tokens(value: Any) -> set[str]:
    return _public_event_tokens(_tokenize(_clean_text(value)))


def _public_slot_root(slot_key: str) -> str:
    clean = _clean_text(slot_key)
    match = re.match(r"^(.*?\.turn_\d+)", clean)
    return match.group(1) if match else clean


def _public_query_analysis(query: str, query_tokens: set[str]) -> Dict[str, Any]:
    speaker_candidates = _public_query_speakers(query)
    subject_text = _public_query_subject(query)
    subject_tokens = set(_tokenize(subject_text))
    event_query_tokens = _public_event_tokens(
        [token for token in query_tokens if token not in subject_tokens],
        speaker_candidates=speaker_candidates,
    )
    return {
        "is_temporal": _public_query_requests_temporal_fact(query),
        "semantic_slot_target": _public_query_semantic_slot(query),
        "target_status": _public_query_target_status(query),
        "time_granularity_target": _public_query_time_granularity(query),
        "speaker_candidates": speaker_candidates,
        "event_tokens": event_query_tokens,
        "query_subject": subject_text,
        "query_subject_signature": _public_subject_signature(subject_text),
        "query_subject_tokens": subject_tokens,
    }


def _public_semantic_slot_match(target: str, value: str) -> bool:
    normalized_target = _normalize(target)
    normalized_value = _normalize(value)
    if not normalized_target or not normalized_value:
        return False
    if normalized_target == "profile":
        return normalized_value in {"profile", "identity", "research_topic", "education", "occupation"}
    return normalized_target == normalized_value


def _public_time_granularity_compatible(target: str, record_value: str) -> bool:
    normalized_target = _normalize(target)
    normalized_value = _normalize(record_value)
    if not normalized_target:
        return True
    if normalized_target == "day_or_coarse":
        return normalized_value in {"day", "relative_day_reference", "month", "year"}
    return normalized_target == normalized_value


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _weighted_token_coverage(
    expected_tokens: Iterable[str],
    available_tokens: Iterable[str],
    *,
    min_token_length: int = 1,
) -> float:
    expected = [
        _normalize(token)
        for token in list(expected_tokens or [])
        if _normalize(token) and len(_normalize(token)) >= max(1, int(min_token_length))
    ]
    if not expected:
        return 0.0
    available = {
        _normalize(token)
        for token in list(available_tokens or [])
        if _normalize(token)
    }
    if not available:
        return 0.0
    total_weight = sum(max(1, len(token)) for token in expected)
    matched_weight = sum(max(1, len(token)) for token in expected if token in available)
    return matched_weight / max(1, total_weight)


def _public_match_profile(
    record: "SessionMemoryRecordV2",
    analysis: Dict[str, Any] | None,
    *,
    query_text: str,
    query_tokens: set[str],
    token_set: set[str],
) -> Dict[str, Any]:
    metadata = dict(record.metadata or {})
    source_kind = _normalize(record.source_kind)
    speaker = _normalize(metadata.get("speaker", ""))
    resolved_date = _normalize(metadata.get("resolved_date", ""))
    temporal_query = bool((analysis or {}).get("is_temporal", _public_query_requests_temporal_fact(query_text)))
    speaker_candidates = {
        _normalize(item)
        for item in set((analysis or {}).get("speaker_candidates", set()) or set())
        if _clean_text(item)
    }
    event_query_tokens = set((analysis or {}).get("event_tokens", set()) or set())
    event_focus_tokens = {token for token in event_query_tokens if len(_normalize(token)) >= 4}
    query_subject_signature = _normalize((analysis or {}).get("query_subject_signature", "")).replace("-", "_")
    query_subject_tokens = {
        _normalize(token)
        for token in set((analysis or {}).get("query_subject_tokens", set()) or set())
        if _normalize(token)
    }
    record_subject_text = _clean_text(metadata.get("subject", ""))
    record_subject_signature = _normalize(metadata.get("subject_signature", "")).replace("-", "_")
    canonical_slot_key = _normalize(metadata.get("canonical_slot_key", "") or record.slot_key)
    slot_subject_signature = _public_subject_signature(canonical_slot_key.split(".subject.", 1)[-1]) if ".subject." in canonical_slot_key else ""
    record_subject_signatures = {
        item
        for item in {
            record_subject_signature,
            _public_subject_signature(record_subject_text),
            slot_subject_signature,
        }
        if item
    }
    record_subject_tokens = set(
        _tokenize(" ".join([record_subject_text, record_subject_signature.replace("_", " "), canonical_slot_key.replace("_", " ")]))
    )
    subject_exact_match = bool(query_subject_signature and query_subject_signature in record_subject_signatures)
    subject_overlap = (
        _weighted_token_coverage(query_subject_tokens, record_subject_tokens, min_token_length=3)
        if query_subject_tokens
        else 0.0
    )
    subject_match = bool(subject_exact_match or (query_subject_signature and subject_overlap >= 0.72))
    record_signature_tokens = (
        _public_signature_tokens(metadata.get("event_signature", ""))
        or _public_event_tokens(token_set, speaker_candidates=speaker_candidates)
        or _public_signature_tokens(record.value)
    )
    record_content_tokens = _public_event_tokens(token_set, speaker_candidates=speaker_candidates)
    record_focus_tokens = record_signature_tokens | record_content_tokens
    event_shared_tokens = event_query_tokens & record_signature_tokens
    event_overlap = len(event_shared_tokens) / max(1, len(event_query_tokens)) if event_query_tokens else 0.0
    event_focus_coverage = _weighted_token_coverage(event_focus_tokens, record_focus_tokens, min_token_length=4)
    exact_speaker_match = bool(speaker and (speaker in speaker_candidates or speaker in query_text))
    semantic_target = _normalize((analysis or {}).get("semantic_slot_target", ""))
    semantic_slot = _normalize(metadata.get("semantic_slot", ""))
    semantic_match = _public_semantic_slot_match(semantic_target, semantic_slot)
    target_status_target = _normalize((analysis or {}).get("target_status", ""))
    target_status = _normalize(metadata.get("target_status", ""))
    target_status_match = bool(target_status_target and target_status and target_status_target == target_status)
    time_target = _normalize((analysis or {}).get("time_granularity_target", ""))
    time_granularity = _normalize(metadata.get("time_granularity", ""))
    time_match = bool(time_granularity and _public_time_granularity_compatible(time_target, time_granularity))
    shared_tokens = query_tokens & token_set if query_tokens and token_set else set()
    shared_content_tokens = _public_event_tokens(shared_tokens, speaker_candidates=speaker_candidates)
    shared_content_ratio = (
        len(shared_content_tokens) / max(1, len(event_query_tokens))
        if event_query_tokens
        else len(shared_content_tokens) / max(1, len(query_tokens))
        if query_tokens
        else 0.0
    )
    event_signal = max(event_overlap, shared_content_ratio, event_focus_coverage)

    support_weight = 0.0
    support_score = 0.0
    if speaker_candidates and speaker:
        support_weight += 0.18
        support_score += 0.18 * (1.0 if exact_speaker_match else 0.0)
    if semantic_target:
        support_weight += 0.16
        support_score += 0.16 * (1.0 if semantic_match else 0.0)
    if query_subject_signature:
        support_weight += 0.42
        support_score += 0.42 * (1.0 if subject_match else subject_overlap * 0.55)
    if event_query_tokens:
        support_weight += 0.34
        support_score += 0.34 * event_signal
    if target_status_target and target_status:
        support_weight += 0.08
        support_score += 0.08 * (1.0 if target_status_match else 0.0)
    if temporal_query:
        support_weight += 0.10
        support_score += 0.10 * (1.0 if resolved_date else 0.0)
        if time_target or time_granularity:
            support_weight += 0.10
            support_score += 0.10 * (1.0 if time_match else 0.0)
        support_weight += 0.08
        support_score += 0.08 * (
            1.0
            if source_kind == "public_dialog_time"
            else 0.55
            if source_kind == "public_dialog_event"
            else 0.0
        )
    elif semantic_target == "profile":
        support_weight += 0.08
        support_score += 0.08 * (1.0 if source_kind == "public_dialog_profile" else 0.0)
    else:
        support_weight += 0.06
        support_score += 0.06 * (1.0 if source_kind == "public_dialog_event" else 0.0)
    positive_signal = support_score / max(1e-6, support_weight) if support_weight > 0 else event_signal

    conflict = 0.0
    if speaker_candidates and speaker and not exact_speaker_match:
        conflict += 0.14
    if semantic_target and semantic_slot and not semantic_match:
        conflict += 0.10 if semantic_target == "profile" else 0.05
    if query_subject_signature and not subject_match:
        conflict += 0.28 * max(0.0, 1.0 - subject_overlap)
    if target_status_target and target_status and not target_status_match:
        conflict += 0.08
    if temporal_query and source_kind == "public_dialog_profile":
        conflict += 0.12
    elif temporal_query and source_kind == "public_dialog_turn":
        conflict += 0.06
    if event_query_tokens and source_kind in {"public_dialog_time", "public_dialog_event"}:
        conflict += 0.22 * max(0.0, 1.0 - event_focus_coverage)
    if temporal_query and source_kind == "public_dialog_time" and not resolved_date and not time_match:
        conflict += 0.16
    if shared_tokens and not shared_content_tokens and not exact_speaker_match and event_signal <= 0 and not semantic_match and not resolved_date:
        conflict += 0.14
    compatibility = _clamp01(positive_signal - conflict)
    return {
        "cluster_root": _public_slot_root(record.slot_key),
        "signature": _normalize(metadata.get("event_signature", "")),
        "source_kind": source_kind,
        "temporal_query": temporal_query,
        "speaker_match": exact_speaker_match,
        "semantic_match": semantic_match,
        "subject_match": subject_match,
        "subject_overlap": subject_overlap,
        "query_subject_signature": query_subject_signature,
        "target_status_match": target_status_match,
        "time_match": time_match,
        "resolved_date": bool(resolved_date),
        "event_overlap": event_overlap,
        "event_focus_coverage": event_focus_coverage,
        "shared_content_ratio": shared_content_ratio,
        "event_signal": event_signal,
        "positive_signal": positive_signal,
        "conflict": conflict,
        "compatibility": compatibility,
    }


def _public_match_reason(
    record: "SessionMemoryRecordV2",
    analysis: Dict[str, Any] | None,
    *,
    speaker_match: bool,
    semantic_match: bool,
    subject_match: bool,
    event_overlap: float,
    target_status_match: bool,
    time_match: bool,
) -> str:
    reasons: List[str] = []
    if speaker_match:
        reasons.append("speaker")
    if semantic_match:
        reasons.append("semantic_slot")
    if subject_match:
        reasons.append("subject")
    if event_overlap > 0:
        reasons.append("event_signature")
    if target_status_match:
        reasons.append("target_status")
    if time_match:
        reasons.append("time_granularity")
    if not reasons and analysis and _public_query_requests_temporal_fact(_clean_text(analysis.get("query", ""))):
        reasons.append("temporal_backoff")
    if not reasons and _clean_text(record.source_kind).startswith("public_dialog"):
        reasons.append("public_lexical")
    return ",".join(reasons[:3])


def infer_category_hints(query: str) -> List[str]:
    return _DEFAULT_TMCRA_PROFILE.infer_category_hints(query)


def infer_history_mode(query: str) -> bool:
    return _contains_any(query, _HISTORY_MARKERS)


def infer_history_mode_clean(query: str) -> bool:
    clean_markers = (
        "\u4e4b\u524d",
        "\u4ee5\u524d",
        "\u5386\u53f2",
        "\u8986\u76d6\u524d",
        "\u4e0a\u4e00\u7248",
    )
    return infer_history_mode(query) or _contains_any(query, clean_markers) or _query_requests_inactive_history(query)


def infer_noise(text: str) -> bool:
    lowered = _normalize(text)
    return any(marker in lowered for marker in _NOISE_MARKERS)


def _query_requests_inactive_history(query: str) -> bool:
    lowered = _normalize(query)
    return any(marker in lowered for marker in _INACTIVE_HISTORY_MARKERS)


def _extract_noise_value(text: str) -> str:
    clean = _clean_text(text)
    if ":" in clean:
        prefix, suffix = clean.split(":", 1)
        if _normalize(prefix) in {"noise note", "noise turn"} and _clean_text(suffix):
            return _clean_text(suffix).rstrip(".")
    return clean.rstrip(".")


def _query_requests_current_pair(query: str) -> bool:
    lowered = _normalize(query)
    explicit_pair_markers = (
        "historical",
        "history",
        "both",
        "previous",
        "old",
        "prior",
        "which one is active",
        "current and previous",
        "current vs previous",
        "current versus previous",
        "鍘嗗彶",
        "之前",
        "旧",
    )
    if any(marker in lowered for marker in explicit_pair_markers):
        return True
    current_markers = ("current", "active", "now", "right now", "当前", "现在")
    history_markers = ("previous", "historical", "history", "old", "prior", "before", "之前", "历史", "旧")
    return any(marker in lowered for marker in current_markers) and any(marker in lowered for marker in history_markers)


def _query_requests_current_pair_clean(query: str) -> bool:
    lowered = _normalize(query)
    clean_markers = (
        "\u5386\u53f2",
        "\u4e4b\u524d",
        "\u65e7",
        "\u4e0a\u4e00\u7248",
    )
    return _query_requests_current_pair(query) or any(marker in lowered for marker in clean_markers)


def _query_requests_current_pair(query: str) -> bool:
    lowered = _normalize(query)
    explicit_pair_markers = (
        "historical",
        "history",
        "both",
        "previous",
        "old",
        "prior",
        "which one is active",
        "current and previous",
        "current vs previous",
        "current versus previous",
        "历史",
        "之前",
        "旧",
    )
    if any(marker in lowered for marker in explicit_pair_markers):
        return True
    current_markers = ("current", "active", "now", "right now", "当前", "现在")
    history_markers = ("previous", "historical", "history", "old", "prior", "before", "之前", "历史", "旧")
    return any(marker in lowered for marker in current_markers) and any(marker in lowered for marker in history_markers)


def _query_requests_current_pair_clean(query: str) -> bool:
    lowered = _normalize(query)
    clean_markers = (
        "历史",
        "之前",
        "旧",
        "上一版",
    )
    return _query_requests_current_pair(query) or any(marker in lowered for marker in clean_markers)


def _slot_slug(value: Any, *, fallback: str = "default", max_length: int = 24) -> str:
    text = _normalize(value)
    if not text:
        return fallback
    parts = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)
    if not parts:
        return fallback
    slug = ".".join(parts[: max(1, max_length // 2)])
    slug = re.sub(r"\.+", ".", slug).strip(".")
    if not slug:
        return fallback
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip(".")
    return slug or fallback


def stable_slot_key(
    *,
    category: str,
    value: str,
    anchors: Sequence[str],
    slot_key: str = "",
    relation: str = "",
    metadata: Dict[str, Any] | None = None,
) -> str:
    return _DEFAULT_TMCRA_PROFILE.stable_slot_key(
        category=category,
        value=value,
        anchors=anchors,
        slot_key=slot_key,
        relation=relation,
        metadata=metadata,
    )


def guess_slot_key(*, category: str, value: str, anchors: Sequence[str]) -> str:
    return _DEFAULT_TMCRA_PROFILE.stable_slot_key(category=category, value=value, anchors=anchors)


def _record_tokens(category: str, slot_key: str, value: str, anchors: Sequence[str], relation: str) -> List[str]:
    return _tokenize(" ".join([category, slot_key, value, relation, *anchors]))


_ACTIVE_RECORD_STATES = {"active", "parallel_active"}
_SUSPECT_RECORD_STATES = {"suspect"}
_PASSIVE_RECORD_STATES = {"evidence", "historical", "inactive", "superseded", "promoted", "false"}
_SINGLETON_MEMORY_CATEGORIES = {"stage_state"}
_PARALLEL_MEMORY_CATEGORIES = {"goal", "constraint", "preference", "terminology", "path"}


def _is_active_record_state(state: Any) -> bool:
    return _normalize(state) in _ACTIVE_RECORD_STATES


def _is_suspect_record_state(state: Any) -> bool:
    return _normalize(state) in _SUSPECT_RECORD_STATES


def _is_passive_record_state(state: Any) -> bool:
    return _normalize(state) in _PASSIVE_RECORD_STATES


def _slot_relation_for_state(state: Any) -> str:
    normalized = _normalize(state)
    if normalized == "parallel_active":
        return "parallel_active_in_slot"
    if normalized == "active":
        return "active_in_slot"
    return "historical_in_slot"


def _metadata_text(metadata: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = _clean_text(metadata.get(key, ""))
        if value:
            return value
    return ""


def _slot_group_key(slot_key: str) -> str:
    clean_slot = _clean_text(slot_key)
    if not clean_slot:
        return ""
    parts = [part for part in clean_slot.split(".") if part]
    if len(parts) <= 2:
        return clean_slot
    if parts[-1].isdigit():
        return ".".join(parts[:-1])
    return ".".join(parts[:-1])


def _record_state_signature(record: "SessionMemoryRecordV2") -> str:
    metadata = dict(record.metadata or {})
    components = [
        _normalize(record.category),
        _normalize(record.relation),
        _normalize(
            _metadata_text(metadata, "parallel_group_key")
            or _metadata_text(metadata, "canonical_slot_key")
            or _slot_group_key(record.slot_key)
            or record.slot_key
        ),
        _normalize(_metadata_text(metadata, "speaker")),
        _normalize(_metadata_text(metadata, "semantic_slot", "profile_type")),
        _normalize(_metadata_text(metadata, "target_status")),
        _normalize(_metadata_text(metadata, "polarity")),
        _normalize(_metadata_text(metadata, "time_granularity")),
        _normalize(
            _metadata_text(metadata, "subject_signature")
            or _metadata_text(metadata, "event_id")
            or _metadata_text(metadata, "dia_id")
            or _metadata_text(metadata, "speaker")
            or (record.anchor_concepts[0] if record.anchor_concepts else "")
        ),
    ]
    return "|".join(component for component in components if component)


def _record_memory_signature(record: "SessionMemoryRecordV2") -> str:
    metadata = dict(record.metadata or {})
    value_signature = _normalize(
        _metadata_text(
            metadata,
            "resolved_time_value",
            "resolved_date",
            "profile_value",
            "time_value",
            "time_display_value",
            "event_signature",
        )
        or record.value
    )
    components = [_record_state_signature(record), value_signature]
    return "|".join(component for component in components if component)


def _record_value_signature(record: "SessionMemoryRecordV2") -> str:
    metadata = dict(record.metadata or {})
    return _normalize(
        _metadata_text(
            metadata,
            "resolved_time_value",
            "resolved_date",
            "profile_value",
            "time_value",
            "time_display_value",
            "event_signature",
            "target_status",
        )
        or record.value
    )


def _record_semantic_family(record: "SessionMemoryRecordV2") -> str:
    metadata = dict(record.metadata or {})
    return _normalize(
        _metadata_text(metadata, "semantic_slot", "profile_type", "canonical_slot_key")
        or record.relation
        or record.category
    )


def _record_singleton_like(record: "SessionMemoryRecordV2") -> bool:
    metadata = dict(record.metadata or {})
    category = _normalize(record.category)
    if category in _SINGLETON_MEMORY_CATEGORIES:
        return True
    semantic_slot = _normalize(_metadata_text(metadata, "semantic_slot", "profile_type"))
    if semantic_slot in {"identity", "research_topic", "education", "occupation"}:
        return True
    canonical_slot_key = _normalize(_metadata_text(metadata, "canonical_slot_key"))
    if canonical_slot_key.startswith("stage.") or canonical_slot_key.endswith(".current"):
        return True
    relation = _normalize(record.relation)
    return relation in {"event_date", "profile_of", "status_of"}


def _records_version_conflict(existing: "SessionMemoryRecordV2", incoming: "SessionMemoryRecordV2") -> bool:
    existing_metadata = dict(existing.metadata or {})
    incoming_metadata = dict(incoming.metadata or {})
    if _normalize(_metadata_text(existing_metadata, "state_signature")) != _normalize(_metadata_text(incoming_metadata, "state_signature")):
        return False
    existing_value_signature = _record_value_signature(existing)
    incoming_value_signature = _record_value_signature(incoming)
    existing_singleton = _record_singleton_like(existing)
    incoming_singleton = _record_singleton_like(incoming)
    if existing_singleton or incoming_singleton:
        if existing_value_signature and incoming_value_signature:
            return existing_value_signature != incoming_value_signature
        return _normalize(existing.value) != _normalize(incoming.value)
    if _record_allows_parallel(existing) and _record_allows_parallel(incoming):
        return False
    existing_time = _normalize(_metadata_text(existing_metadata, "resolved_time_value", "resolved_date", "time_value", "time_display_value"))
    incoming_time = _normalize(_metadata_text(incoming_metadata, "resolved_time_value", "resolved_date", "time_value", "time_display_value"))
    if existing_time and incoming_time and existing_time != incoming_time:
        return True
    existing_profile = _normalize(_metadata_text(existing_metadata, "profile_value"))
    incoming_profile = _normalize(_metadata_text(incoming_metadata, "profile_value"))
    if existing_profile and incoming_profile and existing_profile != incoming_profile:
        return True
    existing_status = _normalize(_metadata_text(existing_metadata, "target_status"))
    incoming_status = _normalize(_metadata_text(incoming_metadata, "target_status"))
    if existing_status and incoming_status and existing_status != incoming_status:
        return True
    existing_family = _record_semantic_family(existing)
    incoming_family = _record_semantic_family(incoming)
    if existing_family and incoming_family and existing_family == incoming_family and existing_value_signature and incoming_value_signature:
        if existing_value_signature != incoming_value_signature:
            return True
    return False


def _classify_conflict_action(
    incoming: "SessionMemoryRecordV2",
    *,
    active_records: Sequence["SessionMemoryRecordV2"],
    active_by_state_signature: Sequence["SessionMemoryRecordV2"],
) -> tuple[str, List["SessionMemoryRecordV2"], str]:
    if not active_records:
        return "insert", [], "empty_slot"
    supersede_targets = [candidate for candidate in active_by_state_signature if _records_version_conflict(candidate, incoming)]
    if supersede_targets:
        return "supersede", supersede_targets, "same_state_revision"
    if not _record_allows_parallel(incoming):
        return "supersede", list(active_records), "slot_disallows_parallel"
    return "parallel_active", [], "parallel_fact"


def _record_allows_parallel(record: "SessionMemoryRecordV2") -> bool:
    metadata = dict(record.metadata or {})
    if "allow_parallel_state" in metadata:
        return bool(metadata.get("allow_parallel_state"))
    category = _normalize(record.category)
    if category in _SINGLETON_MEMORY_CATEGORIES:
        return False
    if category in _PARALLEL_MEMORY_CATEGORIES:
        return True
    semantic_slot = _normalize(_metadata_text(metadata, "semantic_slot"))
    if semantic_slot in {"identity", "research_topic", "education", "occupation"}:
        return False
    canonical_slot_key = _normalize(_metadata_text(metadata, "canonical_slot_key"))
    if canonical_slot_key.startswith("stage.") or canonical_slot_key.endswith(".current"):
        return False
    relation = _normalize(record.relation)
    if relation in {"event_date", "profile_of", "status_of"}:
        return False
    return bool(_metadata_text(metadata, "parallel_group_key")) or category not in {"profile", "status"}


def _prepare_record_signatures(record: "SessionMemoryRecordV2") -> None:
    metadata = dict(record.metadata or {})
    subject_signature = _record_subject_signature_from_parts(record.slot_key, metadata)
    if subject_signature:
        metadata["subject_signature"] = subject_signature
    metadata["depth_layer"] = _infer_depth_layer(
        category=record.category,
        relation=record.relation,
        slot_key=record.slot_key,
        value=record.value,
        metadata=metadata,
    )
    metadata["depth_node_enabled"] = True
    metadata["parallel_group_key"] = _clean_text(
        metadata.get("parallel_group_key")
        or metadata.get("canonical_slot_key")
        or _slot_group_key(record.slot_key)
        or record.slot_key
    )
    record.metadata = metadata
    record.metadata["state_signature"] = _record_state_signature(record)
    record.metadata["memory_signature"] = _record_memory_signature(record)
    record.metadata["value_signature"] = _record_value_signature(record)


def _merge_record_payload(existing: "SessionMemoryRecordV2", incoming: "SessionMemoryRecordV2") -> None:
    previous_metadata = dict(existing.metadata or {})
    incoming_metadata = dict(incoming.metadata or {})
    existing.turn_index = max(existing.turn_index, incoming.turn_index)
    existing.salience = max(existing.salience, incoming.salience)
    existing.confidence = max(existing.confidence, incoming.confidence)
    existing.anchor_concepts = _dedupe([*existing.anchor_concepts, *incoming.anchor_concepts])
    existing.evidence_anchors = _dedupe([*existing.evidence_anchors, *incoming.evidence_anchors])
    merged_metadata = dict(previous_metadata)
    merged_metadata.update(incoming_metadata)
    merged_metadata["origin_answer_ids"] = _dedupe(
        [
            *list(previous_metadata.get("origin_answer_ids", []) or []),
            _clean_text(previous_metadata.get("origin_answer_id", "")),
            *list(incoming_metadata.get("origin_answer_ids", []) or []),
            _clean_text(incoming_metadata.get("origin_answer_id", "")),
        ]
    )
    merged_metadata["support_memory_ids"] = _dedupe(
        [*list(previous_metadata.get("support_memory_ids", []) or []), *list(incoming_metadata.get("support_memory_ids", []) or [])]
    )
    merged_metadata["support_fact_refs"] = _dedupe(
        [*list(previous_metadata.get("support_fact_refs", []) or []), *list(incoming_metadata.get("support_fact_refs", []) or [])]
    )
    merged_metadata["support_path_refs"] = _dedupe(
        [*list(previous_metadata.get("support_path_refs", []) or []), *list(incoming_metadata.get("support_path_refs", []) or [])]
    )
    existing.metadata = merged_metadata
    _prepare_record_signatures(existing)


def _record_suspect_support_turns(record: "SessionMemoryRecordV2") -> List[int]:
    metadata = dict(record.metadata or {})
    turns = []
    for item in list(metadata.get("suspect_support_turns", []) or []):
        try:
            turn = int(item)
        except Exception:
            continue
        if turn > 0 and turn not in turns:
            turns.append(turn)
    if int(record.turn_index or 0) > 0 and int(record.turn_index) not in turns:
        turns.append(int(record.turn_index))
    return sorted(turns)


def _estimate_tokens(text: Any) -> int:
    clean = _clean_text(text)
    if not clean:
        return 0
    return max(1, math.ceil(len(clean) / 4))


@dataclass(slots=True)
class SessionMemoryRecordV2:
    memory_id: str
    category: str
    slot_key: str
    value: str
    relation: str
    anchor_concepts: List[str] = field(default_factory=list)
    evidence_anchors: List[str] = field(default_factory=list)
    salience: float = 0.7
    confidence: float = 0.7
    source_kind: str = "session_memory"
    turn_index: int = 0
    state: str = "active"
    supersedes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def token_set(self) -> set[str]:
        return set(_record_tokens(self.category, self.slot_key, self.value, self.anchor_concepts, self.relation))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "category": self.category,
            "slot_key": self.slot_key,
            "value": self.value,
            "relation": self.relation,
            "anchor_concepts": list(self.anchor_concepts),
            "evidence_anchors": list(self.evidence_anchors),
            "salience": round(float(self.salience), 6),
            "confidence": round(float(self.confidence), 6),
            "source_kind": self.source_kind,
            "turn_index": int(self.turn_index),
            "state": self.state,
            "supersedes": list(self.supersedes),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SessionMemoryEdgeV2:
    edge_id: str
    source_memory_id: str
    target_memory_id: str
    edge_type: str
    score: float = 0.0
    model_score: float = 0.0
    evidence_turn: int = 0
    evidence: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_memory_id": self.source_memory_id,
            "target_memory_id": self.target_memory_id,
            "edge_type": self.edge_type,
            "score": round(float(self.score), 6),
            "model_score": round(float(self.model_score), 6),
            "evidence_turn": int(self.evidence_turn),
            "evidence": self.evidence,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class MemoryScalingRecord:
    turn_count: int
    records: int
    active_slots: int
    superseded_records: int
    graph_nodes: int
    graph_edges: int
    storage_bytes: int
    context_token_estimate: int
    ingest_seconds_total: float
    ingest_us_per_turn: float
    retrieval_ms_p50: float
    retrieval_ms_p95: float
    retrieval_ms_p99: float
    python_rss_bytes: int
    python_peak_bytes: int
    cpu_percent: float
    disk_bytes_written: int
    exploded: bool = False
    guard_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_count": self.turn_count,
            "records": self.records,
            "active_slots": self.active_slots,
            "superseded_records": self.superseded_records,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "storage_bytes": self.storage_bytes,
            "context_token_estimate": self.context_token_estimate,
            "ingest_seconds_total": round(float(self.ingest_seconds_total), 6),
            "ingest_us_per_turn": round(float(self.ingest_us_per_turn), 6),
            "retrieval_ms_p50": round(float(self.retrieval_ms_p50), 6),
            "retrieval_ms_p95": round(float(self.retrieval_ms_p95), 6),
            "retrieval_ms_p99": round(float(self.retrieval_ms_p99), 6),
            "python_rss_bytes": self.python_rss_bytes,
            "python_peak_bytes": self.python_peak_bytes,
            "cpu_percent": round(float(self.cpu_percent), 6),
            "disk_bytes_written": self.disk_bytes_written,
            "exploded": self.exploded,
            "guard_reason": self.guard_reason,
        }


_AUDIT_LOG_FIELDS = ("turn_log", "retrieval_log", "answer_support_log")


class SessionMemoryGraphV2:
    def __init__(
        self,
        *,
        audit_retention: int = 256,
        persistence_backend: str = "memory",
        persistence_path: str = "",
    ) -> None:
        self.turn_index = 0
        self.records_by_id: Dict[str, SessionMemoryRecordV2] = {}
        self.slot_heads: Dict[str, str] = {}
        self.slot_history: Dict[str, List[str]] = defaultdict(list)
        self.memory_edges: Dict[str, SessionMemoryEdgeV2] = {}
        self.subject_depth_heads: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.turn_log: List[Dict[str, Any]] = []
        self.retrieval_log: List[Dict[str, Any]] = []
        self.answer_support_log: List[Dict[str, Any]] = []
        self.noise_turn_count = 0
        self.audit_retention = max(1, int(audit_retention))
        self.persistence_backend = _normalize(persistence_backend) or "memory"
        self.persistence_path = _clean_text(persistence_path)
        self.audit_event_totals: Dict[str, int] = {field: 0 for field in _AUDIT_LOG_FIELDS}
        self.audit_trimmed_counts: Dict[str, int] = {field: 0 for field in _AUDIT_LOG_FIELDS}

    def next_turn(self) -> int:
        self.turn_index += 1
        return self.turn_index

    def configure_persistence(self, *, backend: str, path: str = "", audit_retention: int | None = None) -> None:
        self.persistence_backend = _normalize(backend) or "memory"
        self.persistence_path = _clean_text(path)
        if audit_retention is not None:
            self.audit_retention = max(1, int(audit_retention))

    def _append_audit_event(self, field_name: str, payload: Dict[str, Any]) -> None:
        if field_name not in _AUDIT_LOG_FIELDS:
            raise KeyError(f"Unknown audit field: {field_name}")
        events = getattr(self, field_name)
        events.append(payload)
        self.audit_event_totals[field_name] = int(self.audit_event_totals.get(field_name, 0) or 0) + 1
        overflow = max(0, len(events) - self.audit_retention)
        if overflow:
            del events[:overflow]
            self.audit_trimmed_counts[field_name] = int(self.audit_trimmed_counts.get(field_name, 0) or 0) + overflow

    def _slot_active_records(self, slot_key: str) -> List[SessionMemoryRecordV2]:
        active_records: List[SessionMemoryRecordV2] = []
        for memory_id in reversed(list(self.slot_history.get(slot_key, []) or [])):
            record = self.records_by_id.get(memory_id)
            if record is None or not _is_active_record_state(record.state):
                continue
            active_records.append(record)
        return active_records

    def _record_subject_signature(self, record: SessionMemoryRecordV2) -> str:
        return _record_subject_signature_from_parts(record.slot_key, dict(record.metadata or {}))

    def _same_subject_records(self, subject_signature: str) -> List[SessionMemoryRecordV2]:
        normalized_subject = _public_subject_signature(subject_signature)
        if not normalized_subject:
            return []
        records = [
            record
            for record in self.records_by_id.values()
            if self._record_subject_signature(record) == normalized_subject
        ]
        return sorted(records, key=lambda item: (int(item.turn_index), item.memory_id))

    def _refresh_subject_depth_head(self, record: SessionMemoryRecordV2) -> None:
        if not _is_active_record_state(record.state):
            return
        subject_signature = self._record_subject_signature(record)
        depth_layer = _normalize(dict(record.metadata or {}).get("depth_layer", "")) or "core_view"
        if not subject_signature or depth_layer not in _DEPTH_LAYERS:
            return
        current_id = self.subject_depth_heads[subject_signature].get(depth_layer, "")
        current = self.records_by_id.get(current_id)
        if current is None or int(record.turn_index) >= int(current.turn_index):
            self.subject_depth_heads[subject_signature][depth_layer] = record.memory_id

    def _upsert_memory_edge(self, edge: SessionMemoryEdgeV2) -> None:
        if not edge.source_memory_id or not edge.target_memory_id or edge.source_memory_id == edge.target_memory_id:
            return
        edge.edge_type = _normalize(edge.edge_type) or "co_mentions"
        edge.edge_id = edge.edge_id or f"{edge.source_memory_id}->{edge.target_memory_id}:{edge.edge_type}"
        existing = self.memory_edges.get(edge.edge_id)
        if existing is not None and float(existing.score) >= float(edge.score):
            return
        self.memory_edges[edge.edge_id] = edge

    def _active_profile_aggregate_metadata(self, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        filtered = dict(metadata or {})
        raw_support_ids = [_clean_text(item) for item in filtered.get("profile_support_ids", []) or [] if _clean_text(item)]
        if not raw_support_ids:
            return filtered
        active_supports: List[SessionMemoryRecordV2] = []
        for support_id in raw_support_ids:
            support = self.records_by_id.get(support_id)
            if support is None or not _is_active_record_state(support.state):
                continue
            active_supports.append(support)
        route_terms: List[str] = []
        for support in active_supports:
            support_metadata = dict(support.metadata or {})
            route_terms.extend(support_metadata.get("profile_route_terms", []) or [])
            route_terms.extend(support.anchor_concepts or [])
        filtered["profile_support_ids"] = _dedupe([support.memory_id for support in active_supports], max_items=64)
        filtered["profile_support_turns"] = sorted({int(support.turn_index) for support in active_supports})
        filtered["profile_support_values"] = _dedupe([support.value for support in active_supports], max_items=12)
        filtered["profile_support_count"] = len(filtered["profile_support_ids"])
        if route_terms:
            filtered["profile_route_terms"] = _dedupe(route_terms, max_items=24)
        return filtered

    def _active_profile_cluster_metadata(self, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        filtered = dict(metadata or {})
        raw_profile_ids = [_clean_text(item) for item in filtered.get("profile_support_profile_ids", []) or [] if _clean_text(item)]
        if not raw_profile_ids:
            return filtered
        active_profiles: List[SessionMemoryRecordV2] = []
        for profile_id in raw_profile_ids:
            profile_record = self.records_by_id.get(profile_id)
            if profile_record is None or not _is_active_record_state(profile_record.state):
                continue
            if _normalize(profile_record.source_kind) != PROFILE_AGGREGATE_SOURCE_KIND:
                continue
            active_profiles.append(profile_record)
        support_ids: List[str] = []
        support_turns: List[int] = []
        support_values: List[str] = []
        route_terms: List[str] = []
        profile_types: List[str] = []
        domains: List[str] = []
        for profile_record in active_profiles:
            profile_metadata = dict(profile_record.metadata or {})
            support_ids.extend(profile_metadata.get("profile_support_ids", []) or [])
            support_turns.extend(profile_metadata.get("profile_support_turns", []) or [])
            support_values.extend(profile_metadata.get("profile_support_values", []) or [])
            route_terms.extend(profile_metadata.get("profile_route_terms", []) or [])
            route_terms.extend(profile_metadata.get("profile_cluster_route_terms", []) or [])
            route_terms.extend(profile_record.anchor_concepts or [])
            profile_types.append(_clean_text(profile_metadata.get("profile_type", "")))
            domains.extend(
                [
                    _clean_text(profile_metadata.get("profile_domain", "")),
                    _clean_text(profile_metadata.get("profile_domain_label", "")),
                ]
            )
        filtered["profile_support_profile_ids"] = _dedupe([profile.memory_id for profile in active_profiles], max_items=32)
        filtered["profile_support_ids"] = _dedupe(support_ids, max_items=96)
        filtered["profile_support_turns"] = sorted({int(item) for item in support_turns if str(item).strip().lstrip("-").isdigit()})
        filtered["profile_support_values"] = _dedupe(support_values, max_items=16)
        filtered["profile_support_count"] = len(filtered["profile_support_ids"])
        filtered["profile_cluster_profile_count"] = len(filtered["profile_support_profile_ids"])
        filtered["profile_cluster_types"] = _dedupe(profile_types, max_items=8)
        filtered["profile_cluster_domains"] = _dedupe(domains, max_items=16)
        if route_terms:
            clean_terms = _dedupe(route_terms, max_items=32)
            filtered["profile_cluster_route_terms"] = clean_terms
            filtered["profile_route_terms"] = clean_terms
        return filtered

    def _profile_consolidator_api_key(self) -> str:
        explicit = _clean_text(os.getenv("TMCRA_PROFILE_CONSOLIDATOR_API_KEY", ""))
        if explicit:
            return explicit
        pool = _clean_text(os.getenv("TMCRA_DEEPSEEK_WRITER_KEY_POOL", ""))
        if not pool:
            pool_files = [
                _clean_text(os.getenv("TMCRA_PROFILE_CONSOLIDATOR_KEY_POOL_FILE", "")),
                _clean_text(os.getenv("TMCRA_DEEPSEEK_WRITER_KEY_POOL_FILE", "")),
            ]
            for path_text in pool_files:
                if not path_text:
                    continue
                try:
                    path = Path(path_text)
                    if not path.exists():
                        continue
                    for line in path.read_text(errors="replace").splitlines():
                        clean_line = line.strip()
                        if not clean_line or clean_line.startswith("#") or "=" not in clean_line:
                            continue
                        name, value = clean_line.split("=", 1)
                        if name.strip() == "TMCRA_DEEPSEEK_WRITER_KEY_POOL":
                            pool = value.strip().strip('"').strip("'")
                            break
                    if pool:
                        break
                except Exception:
                    continue
        if not pool:
            return ""
        for candidate in re.split(r"[\s,;]+", pool):
            candidate = _clean_text(candidate)
            if candidate:
                return candidate
        return ""

    def _profile_consolidator_enabled(self, metadata: Mapping[str, Any]) -> bool:
        flag = _normalize(os.getenv("TMCRA_PROFILE_CONSOLIDATOR_ENABLED", "0"))
        if flag not in {"1", "true", "yes", "on"}:
            return False
        try:
            min_support = int(os.getenv("TMCRA_PROFILE_CONSOLIDATOR_MIN_SUPPORT_COUNT", "2") or "2")
        except ValueError:
            min_support = 2
        support_count = int(metadata.get("profile_support_count", 0) or 0)
        if support_count < max(1, min_support):
            return False
        return bool(self._profile_consolidator_api_key())

    def _extract_profile_consolidator_json(self, content: str) -> Dict[str, Any]:
        text = _clean_text(content)
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return dict(parsed) if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _apply_profile_llm_consolidator(self, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        consolidated = dict(metadata or {})
        if not self._profile_consolidator_enabled(consolidated):
            return consolidated
        base_url = _clean_text(os.getenv("TMCRA_PROFILE_CONSOLIDATOR_BASE_URL", "https://api.deepseek.com/v1")).rstrip("/")
        model = _clean_text(os.getenv("TMCRA_PROFILE_CONSOLIDATOR_MODEL", "deepseek-chat"))
        api_key = self._profile_consolidator_api_key()
        if not base_url or not model or not api_key:
            return consolidated
        try:
            max_tokens = int(os.getenv("TMCRA_PROFILE_CONSOLIDATOR_MAX_TOKENS", "256") or "256")
        except ValueError:
            max_tokens = 256
        try:
            timeout_seconds = float(os.getenv("TMCRA_PROFILE_CONSOLIDATOR_TIMEOUT_SECONDS", "45") or "45")
        except ValueError:
            timeout_seconds = 45.0
        system_prompt = (
            "Return JSON only. Consolidate a user profile cluster from evidence. "
            "Do not invent facts. Do not expose memory ids, retrieval, scores, or internal metadata. "
            "Keep profile_summary concise and directly usable as long-term memory. "
            "Return keys: profile_summary, profile_domain_label, profile_output_kind, profile_update_policy."
        )
        payload = {
            "profile_types": list(consolidated.get("profile_cluster_types", []) or []),
            "domain_candidates": list(consolidated.get("profile_cluster_domains", []) or []),
            "support_values": list(consolidated.get("profile_support_values", []) or [])[:10],
            "deterministic_summary": _clean_text(consolidated.get("profile_summary", "")),
            "output_kind": _clean_text(consolidated.get("profile_output_kind", "")),
            "update_policy": _clean_text(consolidated.get("profile_update_policy", "")),
        }
        request_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": max(64, min(max_tokens, 768)),
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        url = f"{base_url}/chat/completions"
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=max(1.0, timeout_seconds)) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            choices = list(response_payload.get("choices", []) or [])
            message = dict(dict(choices[0] if choices else {}).get("message", {}) or {})
            parsed = self._extract_profile_consolidator_json(_clean_text(message.get("content", "")))
            summary = _clean_text(parsed.get("profile_summary", ""))
            if summary:
                consolidated["profile_summary"] = summary[:900]
                consolidated["profile_consolidator_model"] = model
                consolidated["profile_consolidator_status"] = "llm_ok"
                consolidated["profile_consolidator_source"] = "deepseek_openai_compat"
                for key in ("profile_domain_label", "profile_output_kind", "profile_update_policy"):
                    value = _clean_text(parsed.get(key, ""))
                    if value:
                        consolidated[key] = value[:160]
                usage = dict(response_payload.get("usage", {}) or {})
                if usage:
                    consolidated["profile_consolidator_usage"] = {
                        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                        "total_tokens": int(usage.get("total_tokens", 0) or 0),
                    }
            else:
                consolidated["profile_consolidator_status"] = "llm_empty"
        except Exception as exc:
            consolidated["profile_consolidator_status"] = "llm_failed"
            consolidated["profile_consolidator_error"] = f"{exc.__class__.__name__}: {_clean_text(str(exc))[:180]}"
        return consolidated

    def _link_depth_edges_for_record(self, record: SessionMemoryRecordV2, previous_records: Sequence[SessionMemoryRecordV2]) -> None:
        metadata = dict(record.metadata or {})
        subject_signature = self._record_subject_signature(record)
        source_layer = _normalize(metadata.get("depth_layer", "")) or "core_view"
        if not subject_signature or source_layer not in _DEPTH_LAYERS:
            return
        candidates: List[tuple[float, SessionMemoryRecordV2, str]] = []
        source_tokens = set(_tokenize(record.value))
        for previous in previous_records:
            if previous.memory_id == record.memory_id:
                continue
            if self._record_subject_signature(previous) != subject_signature:
                continue
            target_layer = _normalize(dict(previous.metadata or {}).get("depth_layer", "")) or "core_view"
            if target_layer not in _DEPTH_LAYERS:
                target_layer = "core_view"
            target_tokens = set(_tokenize(previous.value))
            overlap = len(source_tokens & target_tokens) / max(1, len(source_tokens | target_tokens)) if source_tokens or target_tokens else 0.0
            turn_gap = max(0, int(record.turn_index) - int(previous.turn_index))
            recency = max(0.0, 1.0 - (turn_gap / max(1.0, float(max(1, int(self.turn_index or record.turn_index or 1))))))
            depth_bonus = 0.24 if source_layer != target_layer else 0.08
            state_bonus = 0.10 if _is_active_record_state(previous.state) else 0.02
            edge_type = _depth_edge_type(source_layer, target_layer, record.value, previous.value)
            score = 0.38 + (0.24 * overlap) + (0.18 * recency) + depth_bonus + state_bonus
            if edge_type in {"refines", "operationalizes", "measures", "productizes", "qualifies"}:
                score += 0.12
            candidates.append((round(score, 6), previous, edge_type))
        candidates.sort(
            key=lambda item: (
                float(item[0]),
                _DEPTH_LAYER_ORDER.get(_normalize(dict(item[1].metadata or {}).get("depth_layer", "")), 99) != _DEPTH_LAYER_ORDER.get(source_layer, 99),
                int(item[1].turn_index),
            ),
            reverse=True,
        )
        for score, previous, edge_type in candidates[:4]:
            edge_id = f"{record.memory_id}->{previous.memory_id}:{edge_type}"
            self._upsert_memory_edge(
                SessionMemoryEdgeV2(
                    edge_id=edge_id,
                    source_memory_id=record.memory_id,
                    target_memory_id=previous.memory_id,
                    edge_type=edge_type,
                    score=score,
                    model_score=0.0,
                    evidence_turn=int(record.turn_index),
                    evidence=_clean_text(dict(record.metadata or {}).get("source_span", "")) or record.value,
                    metadata={
                        "subject_signature": subject_signature,
                        "source_depth_layer": source_layer,
                        "target_depth_layer": _normalize(dict(previous.metadata or {}).get("depth_layer", "")) or "core_view",
                        "edge_source": "depth_heuristic_v1",
                    },
                )
            )

    def _link_profile_edges_for_record(self, record: SessionMemoryRecordV2) -> None:
        metadata = dict(record.metadata or {})
        if not is_profile_layer_record(
            category=record.category,
            source_kind=record.source_kind,
            semantic_slot=metadata.get("semantic_slot", ""),
            metadata=metadata,
        ):
            return
        subject_signature = self._record_subject_signature(record)
        candidates: List[tuple[float, SessionMemoryRecordV2, str]] = []
        for previous in self.records_by_id.values():
            if previous.memory_id == record.memory_id:
                continue
            previous_metadata = dict(previous.metadata or {})
            score, edge_type = profile_edge_score(
                metadata,
                previous_metadata,
                source_value=record.value,
                target_value=previous.value,
            )
            if score <= 0:
                continue
            if subject_signature and self._record_subject_signature(previous) == subject_signature:
                score += 0.04
            candidates.append((round(min(score, 0.96), 6), previous, edge_type))
        candidates.sort(key=lambda item: (float(item[0]), int(item[1].turn_index)), reverse=True)
        for score, previous, edge_type in candidates[:6]:
            previous_metadata = dict(previous.metadata or {})
            self._upsert_memory_edge(
                SessionMemoryEdgeV2(
                    edge_id=f"{record.memory_id}->{previous.memory_id}:{edge_type}",
                    source_memory_id=record.memory_id,
                    target_memory_id=previous.memory_id,
                    edge_type=edge_type,
                    score=score,
                    model_score=0.0,
                    evidence_turn=int(record.turn_index),
                    evidence=_clean_text(metadata.get("source_span", "")) or record.value,
                    metadata={
                        "subject_signature": subject_signature,
                        "source_profile_type": _clean_text(metadata.get("profile_type", "")),
                        "target_profile_type": _clean_text(previous_metadata.get("profile_type", "")),
                        "source_profile_domain": _clean_text(metadata.get("profile_domain", "")),
                        "target_profile_domain": _clean_text(previous_metadata.get("profile_domain", "")),
                        "edge_source": "profile_layer_v1",
                    },
                )
            )

    def _link_facet_edges_for_record(self, record: SessionMemoryRecordV2) -> None:
        metadata = dict(record.metadata or {})
        facet_type = _normalize(metadata.get("facet_type", ""))
        if _normalize(metadata.get("content_variant", "")) != "event_facet_write" and not facet_type:
            return
        parent_slot_key = _clean_text(metadata.get("facet_parent_slot_key", ""))
        parent_event_signature = _normalize(metadata.get("facet_parent_event_signature", ""))
        if not parent_slot_key and not parent_event_signature:
            return
        candidates: List[tuple[float, SessionMemoryRecordV2]] = []
        for candidate in self.records_by_id.values():
            if candidate.memory_id == record.memory_id:
                continue
            candidate_metadata = dict(candidate.metadata or {})
            if _normalize(candidate_metadata.get("content_variant", "")) == "event_facet_write":
                continue
            score = 0.0
            if parent_slot_key and _clean_text(candidate.slot_key).lower() == parent_slot_key.lower():
                score += 0.66
            if parent_event_signature and _normalize(candidate_metadata.get("event_signature", "")) == parent_event_signature:
                score += 0.28
            if int(candidate.turn_index) == int(record.turn_index):
                score += 0.08
            if score > 0:
                candidates.append((round(min(score, 0.98), 6), candidate))
        if not candidates:
            return
        candidates.sort(key=lambda item: (float(item[0]), int(item[1].turn_index)), reverse=True)
        score, parent = candidates[0]
        edge_type = f"has_{facet_type or 'attribute'}_facet"
        evidence = _clean_text(metadata.get("facet_source_span", "")) or _clean_text(metadata.get("source_span", "")) or record.value
        shared_metadata = {
            "edge_source": "event_facet_layer_v1",
            "facet_type": facet_type,
            "facet_role": _clean_text(metadata.get("facet_role", "")),
            "facet_value": _clean_text(metadata.get("facet_value", "")) or record.value,
            "parent_slot_key": parent_slot_key,
            "parent_event_signature": parent_event_signature,
            "subject_signature": self._record_subject_signature(parent) or self._record_subject_signature(record),
        }
        self._upsert_memory_edge(
            SessionMemoryEdgeV2(
                edge_id=f"{parent.memory_id}->{record.memory_id}:{edge_type}",
                source_memory_id=parent.memory_id,
                target_memory_id=record.memory_id,
                edge_type=edge_type,
                score=score,
                model_score=0.0,
                evidence_turn=int(record.turn_index),
                evidence=evidence,
                metadata=shared_metadata,
            )
        )
        self._upsert_memory_edge(
            SessionMemoryEdgeV2(
                edge_id=f"{record.memory_id}->{parent.memory_id}:facet_of",
                source_memory_id=record.memory_id,
                target_memory_id=parent.memory_id,
                edge_type="facet_of",
                score=score,
                model_score=0.0,
                evidence_turn=int(record.turn_index),
                evidence=evidence,
                metadata=shared_metadata,
            )
        )

    def _upsert_profile_cluster_for_aggregate(self, aggregate: SessionMemoryRecordV2) -> str:
        metadata = dict(aggregate.metadata or {})
        if _normalize(aggregate.source_kind) != PROFILE_AGGREGATE_SOURCE_KIND:
            return ""
        if not bool(metadata.get("profile_aggregate_node")):
            return ""
        if bool(metadata.get("profile_cluster_node")):
            return ""
        if not _is_active_record_state(aggregate.state):
            return ""

        candidates: List[tuple[float, SessionMemoryRecordV2]] = []
        for candidate in self.records_by_id.values():
            candidate_metadata = dict(candidate.metadata or {})
            if candidate.memory_id == aggregate.memory_id:
                continue
            if _normalize(candidate.source_kind) != PROFILE_CLUSTER_SOURCE_KIND and not bool(candidate_metadata.get("profile_cluster_node")):
                continue
            if not _is_active_record_state(candidate.state):
                continue
            support_profile_ids = {
                _normalize(item)
                for item in candidate_metadata.get("profile_support_profile_ids", []) or []
                if _clean_text(item)
            }
            if _normalize(aggregate.memory_id) in support_profile_ids:
                candidates.append((1.0, candidate))
                continue
            similarity = profile_cluster_similarity(metadata, candidate_metadata)
            if similarity >= 0.16:
                candidates.append((similarity, candidate))

        candidates.sort(key=lambda item: (float(item[0]), int(item[1].turn_index)), reverse=True)
        cluster = candidates[0][1] if candidates else None
        existing_metadata = self._active_profile_cluster_metadata(dict(cluster.metadata or {})) if cluster is not None else {}
        cluster_metadata = build_profile_cluster_metadata(
            support_profile_id=aggregate.memory_id,
            support_metadata=metadata,
            existing_metadata=existing_metadata,
        )
        cluster_metadata = self._apply_profile_llm_consolidator(cluster_metadata)
        slot_key = cluster.slot_key if cluster is not None else profile_cluster_slot_key(cluster_metadata)
        if not slot_key:
            return ""
        cluster_value = _clean_text(cluster_metadata.get("profile_value", "")) or aggregate.value
        cluster_anchors = _dedupe(
            [
                *list(cluster_metadata.get("profile_cluster_route_terms", []) or []),
                *list(cluster_metadata.get("profile_route_terms", []) or []),
                *list(aggregate.anchor_concepts or []),
            ],
            max_items=16,
        )
        if cluster is None:
            cluster = SessionMemoryRecordV2(
                memory_id=f"{slot_key}:cluster:{int(aggregate.turn_index)}",
                category=PROFILE_CLUSTER_CATEGORY,
                slot_key=slot_key,
                value=cluster_value,
                relation="profile_cluster",
                anchor_concepts=cluster_anchors,
                evidence_anchors=list(aggregate.evidence_anchors or aggregate.anchor_concepts),
                salience=min(0.99, max(float(aggregate.salience), 0.90)),
                confidence=min(0.97, max(float(aggregate.confidence), 0.84)),
                source_kind=PROFILE_CLUSTER_SOURCE_KIND,
                turn_index=int(aggregate.turn_index),
                state="active",
                metadata={
                    **cluster_metadata,
                    "canonical_slot_key": slot_key,
                    "memory_role": "user",
                    "authority": "derived_profile_cluster",
                    "source": "profile_layer",
                },
            )
            _prepare_record_signatures(cluster)
            self.records_by_id[cluster.memory_id] = cluster
            self.slot_heads[slot_key] = cluster.memory_id
            self.slot_history[slot_key].append(cluster.memory_id)
            self._refresh_subject_depth_head(cluster)
        else:
            cluster.value = cluster_value
            cluster.anchor_concepts = cluster_anchors
            cluster.evidence_anchors = _dedupe(
                [*list(cluster.evidence_anchors or []), *list(aggregate.evidence_anchors or aggregate.anchor_concepts)],
                max_items=16,
            )
            cluster.salience = min(0.99, max(float(cluster.salience), float(aggregate.salience), 0.90))
            cluster.confidence = min(0.97, max(float(cluster.confidence), float(aggregate.confidence), 0.84))
            cluster.turn_index = min(int(cluster.turn_index or aggregate.turn_index), int(aggregate.turn_index))
            cluster.state = "active"
            cluster.metadata = {
                **cluster_metadata,
                "canonical_slot_key": slot_key,
                "memory_role": "user",
                "authority": "derived_profile_cluster",
                "source": "profile_layer",
            }
            _prepare_record_signatures(cluster)
            self._refresh_subject_depth_head(cluster)

        self._upsert_memory_edge(
            SessionMemoryEdgeV2(
                edge_id=f"{aggregate.memory_id}->{cluster.memory_id}:profile_cluster_supports",
                source_memory_id=aggregate.memory_id,
                target_memory_id=cluster.memory_id,
                edge_type="profile_cluster_supports",
                score=0.95,
                model_score=0.0,
                evidence_turn=int(aggregate.turn_index),
                evidence=aggregate.value,
                metadata={
                    "edge_source": "profile_layer_cluster_v1",
                    "profile_support_key": _clean_text(cluster.metadata.get("profile_support_key", "")),
                    "profile_cluster_profile_count": int(cluster.metadata.get("profile_cluster_profile_count", 0) or 0),
                },
            )
        )
        self._upsert_memory_edge(
            SessionMemoryEdgeV2(
                edge_id=f"{cluster.memory_id}->{aggregate.memory_id}:profile_cluster_supported_by",
                source_memory_id=cluster.memory_id,
                target_memory_id=aggregate.memory_id,
                edge_type="profile_cluster_supported_by",
                score=0.90,
                model_score=0.0,
                evidence_turn=int(aggregate.turn_index),
                evidence=aggregate.value,
                metadata={
                    "edge_source": "profile_layer_cluster_v1",
                    "profile_support_key": _clean_text(cluster.metadata.get("profile_support_key", "")),
                },
            )
        )
        self._link_profile_edges_for_record(cluster)
        return cluster.memory_id

    def _upsert_profile_aggregate_for_record(self, record: SessionMemoryRecordV2) -> str:
        legacy_profile_layer = _normalize(os.getenv("TMCRA_LEGACY_PROFILE_LAYER_ENABLED", "1"))
        if legacy_profile_layer in {"0", "false", "no", "off", "disabled"}:
            return ""
        metadata = dict(record.metadata or {})
        if _normalize(record.source_kind) in {PROFILE_AGGREGATE_SOURCE_KIND, PROFILE_CLUSTER_SOURCE_KIND}:
            return ""
        if bool(metadata.get("profile_aggregate_node")):
            return ""
        if bool(metadata.get("profile_cluster_node")):
            return ""
        if not _is_active_record_state(record.state):
            return ""
        if not is_profile_layer_record(
            category=record.category,
            source_kind=record.source_kind,
            semantic_slot=metadata.get("semantic_slot", ""),
            metadata=metadata,
        ):
            return ""
        slot_key = profile_aggregate_slot_key(metadata)
        if not slot_key:
            return ""
        aggregate_id = self.slot_heads.get(slot_key, "")
        aggregate = self.records_by_id.get(aggregate_id) if aggregate_id else None
        existing_metadata = self._active_profile_aggregate_metadata(dict(aggregate.metadata or {})) if aggregate is not None else {}
        aggregate_metadata = build_profile_aggregate_metadata(
            support_record_id=record.memory_id,
            support_turn_index=int(record.turn_index),
            support_value=record.value,
            support_anchors=record.anchor_concepts,
            support_metadata=metadata,
            existing_metadata=existing_metadata,
        )
        aggregate_value = _clean_text(aggregate_metadata.get("profile_value", "")) or record.value
        aggregate_anchors = _dedupe(
            [
                *list(aggregate_metadata.get("profile_route_terms", []) or []),
                *list(record.anchor_concepts or []),
            ],
            max_items=12,
        )
        if aggregate is None:
            aggregate = SessionMemoryRecordV2(
                memory_id=f"{slot_key}:profile:{int(record.turn_index)}",
                category=PROFILE_AGGREGATE_CATEGORY,
                slot_key=slot_key,
                value=aggregate_value,
                relation="profile_aggregate",
                anchor_concepts=aggregate_anchors,
                evidence_anchors=list(record.evidence_anchors or record.anchor_concepts),
                salience=min(0.98, max(float(record.salience), 0.86)),
                confidence=min(0.96, max(float(record.confidence), 0.82)),
                source_kind=PROFILE_AGGREGATE_SOURCE_KIND,
                turn_index=int(record.turn_index),
                state="active",
                metadata={
                    **aggregate_metadata,
                    "canonical_slot_key": slot_key,
                    "memory_role": "user",
                    "authority": "derived_profile",
                    "source": "profile_layer",
                },
            )
            _prepare_record_signatures(aggregate)
            self.records_by_id[aggregate.memory_id] = aggregate
            self.slot_heads[slot_key] = aggregate.memory_id
            self.slot_history[slot_key].append(aggregate.memory_id)
            self._refresh_subject_depth_head(aggregate)
        else:
            aggregate.value = aggregate_value
            aggregate.anchor_concepts = aggregate_anchors
            aggregate.evidence_anchors = _dedupe(
                [*list(aggregate.evidence_anchors or []), *list(record.evidence_anchors or record.anchor_concepts)],
                max_items=12,
            )
            aggregate.salience = min(0.98, max(float(aggregate.salience), float(record.salience), 0.86))
            aggregate.confidence = min(0.96, max(float(aggregate.confidence), float(record.confidence), 0.82))
            aggregate.turn_index = min(int(aggregate.turn_index or record.turn_index), int(record.turn_index))
            aggregate.state = "active"
            aggregate.metadata = {
                **aggregate_metadata,
                "canonical_slot_key": slot_key,
                "memory_role": "user",
                "authority": "derived_profile",
                "source": "profile_layer",
            }
            _prepare_record_signatures(aggregate)
            self._refresh_subject_depth_head(aggregate)
        self._upsert_memory_edge(
            SessionMemoryEdgeV2(
                edge_id=f"{record.memory_id}->{aggregate.memory_id}:profile_supports",
                source_memory_id=record.memory_id,
                target_memory_id=aggregate.memory_id,
                edge_type="profile_supports",
                score=0.94,
                model_score=0.0,
                evidence_turn=int(record.turn_index),
                evidence=record.value,
                metadata={
                    "edge_source": "profile_layer_aggregate_v1",
                    "profile_support_key": _clean_text(aggregate.metadata.get("profile_support_key", "")),
                    "profile_type": _clean_text(aggregate.metadata.get("profile_type", "")),
                    "profile_domain": _clean_text(aggregate.metadata.get("profile_domain", "")),
                },
            )
        )
        self._upsert_memory_edge(
            SessionMemoryEdgeV2(
                edge_id=f"{aggregate.memory_id}->{record.memory_id}:profile_supported_by",
                source_memory_id=aggregate.memory_id,
                target_memory_id=record.memory_id,
                edge_type="profile_supported_by",
                score=0.88,
                model_score=0.0,
                evidence_turn=int(record.turn_index),
                evidence=record.value,
                metadata={
                    "edge_source": "profile_layer_aggregate_v1",
                    "profile_support_key": _clean_text(aggregate.metadata.get("profile_support_key", "")),
                },
            )
        )
        self._link_profile_edges_for_record(aggregate)
        self._upsert_profile_cluster_for_aggregate(aggregate)
        return aggregate.memory_id

    def depth_chain_for_query(
        self,
        query: str,
        *,
        seed_memory_ids: Sequence[str] | None = None,
        top_k: int = 6,
    ) -> Dict[str, Any]:
        query_text = _clean_text(query)
        query_tokens = set(_tokenize(query_text))
        analysis = _public_query_analysis(query_text, query_tokens)
        subject_signature = _public_subject_signature(analysis.get("query_subject", ""))
        seed_ids = [_clean_text(item) for item in list(seed_memory_ids or []) if _clean_text(item)]
        if not subject_signature:
            for memory_id in seed_ids:
                record = self.records_by_id.get(memory_id)
                if record is None:
                    continue
                subject_signature = self._record_subject_signature(record)
                if subject_signature:
                    break
        if not subject_signature:
            return {"enabled": False, "reason": "no_subject_signature", "nodes": [], "edges": []}
        subject_records = self._same_subject_records(subject_signature)
        if not subject_records:
            return {
                "enabled": True,
                "reason": "no_subject_records",
                "subject_signature": subject_signature,
                "nodes": [],
                "edges": [],
            }
        seed_set = set(seed_ids)
        head_ids = set(self.subject_depth_heads.get(subject_signature, {}).values())
        edge_neighbor_ids = set()
        for edge in self.memory_edges.values():
            if _normalize(dict(edge.metadata or {}).get("subject_signature", "")) != subject_signature:
                continue
            if edge.source_memory_id in seed_set or edge.target_memory_id in seed_set or edge.source_memory_id in head_ids or edge.target_memory_id in head_ids:
                edge_neighbor_ids.add(edge.source_memory_id)
                edge_neighbor_ids.add(edge.target_memory_id)
        scored: List[tuple[float, SessionMemoryRecordV2]] = []
        for record in subject_records:
            layer = _normalize(dict(record.metadata or {}).get("depth_layer", "")) or "core_view"
            layer_rank = _DEPTH_LAYER_ORDER.get(layer, 99)
            token_overlap = 0.0
            value_tokens = set(_tokenize(record.value))
            if query_tokens or value_tokens:
                token_overlap = len(query_tokens & value_tokens) / max(1, len(query_tokens | value_tokens))
            score = 0.30 + (0.18 * token_overlap)
            if record.memory_id in seed_set:
                score += 0.50
            if record.memory_id in head_ids:
                score += 0.32
            if record.memory_id in edge_neighbor_ids:
                score += 0.20
            if _is_active_record_state(record.state):
                score += 0.12
            score += max(0.0, 0.18 - (0.018 * layer_rank))
            scored.append((round(score, 6), record))
        scored.sort(key=lambda item: (item[0], int(item[1].turn_index)), reverse=True)
        selected: List[SessionMemoryRecordV2] = []
        seen_layers: set[str] = set()
        for _, record in scored:
            layer = _normalize(dict(record.metadata or {}).get("depth_layer", "")) or "core_view"
            if layer not in seen_layers or record.memory_id in seed_set or record.memory_id in head_ids:
                selected.append(record)
                seen_layers.add(layer)
            if len(selected) >= max(1, int(top_k)):
                break
        if len(selected) < max(1, min(3, int(top_k))):
            selected_ids = {record.memory_id for record in selected}
            for _, record in scored:
                if record.memory_id in selected_ids:
                    continue
                selected.append(record)
                selected_ids.add(record.memory_id)
                if len(selected) >= max(1, int(top_k)):
                    break
        selected_ids = {record.memory_id for record in selected}
        chain_edges = [
            edge
            for edge in self.memory_edges.values()
            if edge.source_memory_id in selected_ids
            and edge.target_memory_id in selected_ids
            and _normalize(dict(edge.metadata or {}).get("subject_signature", "")) == subject_signature
        ]
        chain_edges.sort(key=lambda item: (float(item.score), int(item.evidence_turn)), reverse=True)
        ordered_nodes = sorted(
            selected,
            key=lambda item: (
                _DEPTH_LAYER_ORDER.get(_normalize(dict(item.metadata or {}).get("depth_layer", "")) or "core_view", 99),
                int(item.turn_index),
            ),
        )
        return {
            "enabled": True,
            "reason": "subject_depth_chain",
            "subject_signature": subject_signature,
            "subject": _clean_text(analysis.get("query_subject", "")),
            "node_count": len(ordered_nodes),
            "edge_count": len(chain_edges),
            "depth_layers": _dedupe(
                _normalize(dict(record.metadata or {}).get("depth_layer", "")) or "core_view"
                for record in ordered_nodes
            ),
            "nodes": [record.to_dict() for record in ordered_nodes],
            "edges": [edge.to_dict() for edge in chain_edges[: max(0, int(top_k) * 2)]],
        }

    def _core_payload(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "slot_heads": dict(self.slot_heads),
            "slot_history": {slot: list(values) for slot, values in self.slot_history.items()},
            "subject_depth_heads": {subject: dict(heads) for subject, heads in self.subject_depth_heads.items()},
            "records": [record.to_dict() for record in sorted(self.records_by_id.values(), key=lambda item: (item.turn_index, item.slot_key, item.state))],
            "memory_edges": [edge.to_dict() for edge in sorted(self.memory_edges.values(), key=lambda item: (item.evidence_turn, item.edge_id))],
        }

    def _audit_payload(self) -> Dict[str, Any]:
        return {
            "turn_log": list(self.turn_log),
            "retrieval_log": list(self.retrieval_log),
            "answer_support_log": list(self.answer_support_log),
            "audit_event_totals": dict(self.audit_event_totals),
            "audit_trimmed_counts": dict(self.audit_trimmed_counts),
            "audit_retention": int(self.audit_retention),
        }

    def record_turn(
        self,
        *,
        turn_kind: str,
        text: str,
        turn_index: int,
        record_ids: Sequence[str],
        speaker: str = "user",
        assistant_text: str = "",
        metadata: Dict[str, Any] | None = None,
        writeback_class: str = "",
    ) -> None:
        if not record_ids and infer_noise(text):
            self.noise_turn_count += 1
        self._append_audit_event(
            "turn_log",
            {
                "turn_id": f"turn:{turn_index}",
                "turn_index": int(turn_index),
                "kind": turn_kind,
                "text": _clean_text(text),
                "speaker": _clean_text(speaker) or "user",
                "assistant_text": _clean_text(assistant_text),
                "writeback_class": _clean_text(writeback_class),
                "record_ids": list(record_ids),
                "metadata": dict(metadata or {}),
            },
        )

    def add_records(self, records: Iterable[SessionMemoryRecordV2]) -> List[str]:
        stored_ids: List[str] = []
        for record in records:
            record.value = _clean_text(record.value)
            record.category = _clean_text(record.category) or "memory"
            record.slot_key = _clean_text(record.slot_key) or guess_slot_key(category=record.category, value=record.value, anchors=record.anchor_concepts)
            record.relation = _clean_text(record.relation) or "related_to"
            record.anchor_concepts = _dedupe(record.anchor_concepts)
            record.evidence_anchors = _dedupe(record.evidence_anchors or record.anchor_concepts)
            record.state = _normalize(record.state) or "active"
            _prepare_record_signatures(record)
            previous_subject_records = self._same_subject_records(self._record_subject_signature(record))
            if not record.memory_id:
                record.memory_id = f"{record.slot_key}:{record.turn_index}:{len(self.records_by_id)}"

            if _is_suspect_record_state(record.state):
                record.metadata["conflict_action"] = _clean_text(record.metadata.get("conflict_action", "")) or "suspect_buffer"
                record.metadata["promotion_state"] = _clean_text(record.metadata.get("promotion_state", "")) or "suspect_buffered"
                record.metadata["suspect_support_turns"] = _record_suspect_support_turns(record)
                record.metadata["suspect_support_count"] = len(record.metadata["suspect_support_turns"])
                suspect_duplicate = next(
                    (
                        candidate
                        for candidate in self.records_by_id.values()
                        if _is_suspect_record_state(candidate.state)
                        and _normalize(candidate.slot_key) == _normalize(record.slot_key)
                        and _normalize(dict(candidate.metadata or {}).get("memory_signature", "")) == _normalize(dict(record.metadata or {}).get("memory_signature", ""))
                    ),
                    None,
                )
                if suspect_duplicate is not None:
                    _merge_record_payload(suspect_duplicate, record)
                    support_turns = sorted(
                        set(_record_suspect_support_turns(suspect_duplicate)) | set(_record_suspect_support_turns(record))
                    )
                    suspect_duplicate.metadata["suspect_support_turns"] = support_turns
                    suspect_duplicate.metadata["suspect_support_count"] = len(support_turns)
                    suspect_duplicate.metadata["conflict_action"] = "suspect_duplicate_reinforced"
                    stored_ids.append(suspect_duplicate.memory_id)
                    continue
                self.records_by_id[record.memory_id] = record
                self.slot_history[record.slot_key].append(record.memory_id)
                self._refresh_subject_depth_head(record)
                self._link_depth_edges_for_record(record, previous_subject_records)
                self._link_profile_edges_for_record(record)
                self._link_facet_edges_for_record(record)
                stored_ids.append(record.memory_id)
                continue

            if _is_passive_record_state(record.state):
                record.metadata["conflict_action"] = _clean_text(record.metadata.get("conflict_action", "")) or f"{record.state}_buffer"
                self.records_by_id[record.memory_id] = record
                self.slot_history[record.slot_key].append(record.memory_id)
                self._refresh_subject_depth_head(record)
                self._link_depth_edges_for_record(record, previous_subject_records)
                self._link_profile_edges_for_record(record)
                self._link_facet_edges_for_record(record)
                stored_ids.append(record.memory_id)
                continue

            active_records = self._slot_active_records(record.slot_key)
            duplicate_target = next(
                (
                    candidate
                    for candidate in active_records
                    if _normalize(dict(candidate.metadata or {}).get("memory_signature", "")) == _normalize(dict(record.metadata or {}).get("memory_signature", ""))
                ),
                None,
            )
            if duplicate_target is not None:
                _merge_record_payload(duplicate_target, record)
                duplicate_target.metadata["conflict_action"] = "duplicate"
                stored_ids.append(duplicate_target.memory_id)
                aggregate_id = self._upsert_profile_aggregate_for_record(duplicate_target)
                if aggregate_id and aggregate_id not in stored_ids:
                    stored_ids.append(aggregate_id)
                continue

            state_signature = _normalize(dict(record.metadata or {}).get("state_signature", ""))
            active_by_state_signature = [
                candidate
                for candidate in active_records
                if _normalize(dict(candidate.metadata or {}).get("state_signature", "")) == state_signature
            ]
            if active_records:
                conflict_action, supersede_targets, conflict_reason = _classify_conflict_action(
                    record,
                    active_records=active_records,
                    active_by_state_signature=active_by_state_signature,
                )
                record.metadata["conflict_reason"] = conflict_reason
                if conflict_action == "supersede":
                    for active_head in supersede_targets:
                        if not _is_active_record_state(active_head.state):
                            continue
                        active_head.state = "superseded"
                        active_head.metadata["superseded_by"] = record.memory_id
                        active_head.metadata["superseded_reason"] = conflict_reason
                        record.supersedes.append(active_head.memory_id)
                    record.state = "active"
                    record.metadata["conflict_action"] = "supersede"
                else:
                    record.state = "parallel_active"
                    record.metadata["conflict_action"] = "parallel_active"
            else:
                record.state = "active"
                record.metadata["conflict_action"] = "insert"

            self.records_by_id[record.memory_id] = record
            self.slot_heads[record.slot_key] = record.memory_id
            self.slot_history[record.slot_key].append(record.memory_id)
            self._refresh_subject_depth_head(record)
            self._link_depth_edges_for_record(record, previous_subject_records)
            self._link_profile_edges_for_record(record)
            self._link_facet_edges_for_record(record)
            aggregate_id = self._upsert_profile_aggregate_for_record(record)
            if aggregate_id and aggregate_id not in stored_ids:
                stored_ids.append(aggregate_id)
            for suspect in list(self.records_by_id.values()):
                if not _is_suspect_record_state(suspect.state):
                    continue
                if _normalize(suspect.slot_key) != _normalize(record.slot_key):
                    continue
                suspect_signature = _normalize(dict(suspect.metadata or {}).get("memory_signature", ""))
                record_signature = _normalize(dict(record.metadata or {}).get("memory_signature", ""))
                if not suspect_signature or suspect_signature != record_signature:
                    continue
                suspect.state = "promoted"
                suspect.metadata["promotion_state"] = "promoted_by_formal_evidence"
                suspect.metadata["promoted_to_memory_id"] = record.memory_id
                suspect.metadata["promoted_turn_index"] = int(record.turn_index)
            stored_ids.append(record.memory_id)
        return stored_ids

    def _score_record(
        self,
        record: SessionMemoryRecordV2,
        *,
        query: str,
        query_tokens: set[str],
        hints: set[str],
        history_mode: bool,
        public_analysis: Dict[str, Any] | None = None,
    ) -> tuple[float, str, Dict[str, Any] | None]:
        token_set = record.token_set()
        overlap = len(query_tokens & token_set) if query_tokens and token_set else 0
        lexical = overlap / max(1, len(query_tokens | token_set)) if query_tokens and token_set else 0.0
        base_score = lexical
        match_reason = ""
        public_profile: Dict[str, Any] | None = None
        query_text = _normalize(query)
        if hints and record.category in hints:
            base_score += 0.2
        if hints and "path" in hints and record.relation == "path_edge":
            base_score += 0.15
        if record.slot_key and record.slot_key.lower() in query_text:
            base_score += 0.12
        canonical_slot_key = _normalize(record.metadata.get("canonical_slot_key", "")) if isinstance(record.metadata, dict) else ""
        if canonical_slot_key and canonical_slot_key in query_text:
            base_score += 0.12
        anchor_tokens = set()
        for anchor in record.anchor_concepts[:6]:
            anchor_text = _normalize(anchor)
            if not anchor_text:
                continue
            if anchor_text in query_text:
                base_score += 0.18
            anchor_tokens.update(_tokenize(anchor_text))
        if anchor_tokens and query_tokens:
            base_score += 0.1 * (len(query_tokens & anchor_tokens) / max(1, len(query_tokens)))
        value_text = _normalize(record.value)
        if value_text and value_text in query_text:
            base_score += 0.16
        if hints and record.category not in hints and record.category in {"goal", "constraint", "preference", "terminology", "stage_state"}:
            base_score -= 0.04
        recency = 1.0 if self.turn_index <= 0 else max(0.0, 1.0 - ((self.turn_index - record.turn_index) / max(1.0, float(self.turn_index))))
        generic_recency_prior = 0.12 * recency
        generic_salience_prior = 0.18 * float(record.salience)
        generic_confidence_prior = 0.12 * float(record.confidence)
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        memory_role = _normalize(metadata.get("memory_role", ""))
        authority = _normalize(metadata.get("authority", ""))
        generic_authority_prior = 0.0
        if memory_role == "user" and authority == "source":
            generic_authority_prior += 0.18
        elif memory_role == "assistant" and authority == "promoted":
            generic_authority_prior += 0.08
        elif memory_role == "assistant" and authority == "derived":
            generic_authority_prior -= 0.18
        source_kind = _normalize(record.source_kind)
        speaker = _normalize(metadata.get("speaker", ""))
        resolved_date = _normalize(metadata.get("resolved_date", ""))
        is_public_benchmark_record = source_kind.startswith("public_dialog") or _normalize(metadata.get("source", "")) == "public_benchmark"
        score = base_score + generic_recency_prior + generic_salience_prior + generic_confidence_prior + generic_authority_prior
        if is_public_benchmark_record:
            analysis = dict(public_analysis or {})
            public_profile = _public_match_profile(
                record,
                analysis,
                query_text=query_text,
                query_tokens=query_tokens,
                token_set=token_set,
            )
            temporal_query = bool(public_profile.get("temporal_query", False))
            exact_speaker_match = bool(public_profile.get("speaker_match", False))
            semantic_match = bool(public_profile.get("semantic_match", False))
            target_status_match = bool(public_profile.get("target_status_match", False))
            time_match = bool(public_profile.get("time_match", False))
            event_overlap = float(public_profile.get("event_overlap", 0.0) or 0.0)
            event_focus_coverage = float(public_profile.get("event_focus_coverage", 0.0) or 0.0)
            event_signal = float(public_profile.get("event_signal", 0.0) or 0.0)
            subject_match = bool(public_profile.get("subject_match", False))
            subject_overlap = float(public_profile.get("subject_overlap", 0.0) or 0.0)
            query_subject_signature = _normalize(public_profile.get("query_subject_signature", ""))
            public_compatibility = float(public_profile.get("compatibility", 0.0) or 0.0)
            public_conflict = float(public_profile.get("conflict", 0.0) or 0.0)
            public_prior_gate = _clamp01(0.12 + (0.96 * public_compatibility) - (0.50 * public_conflict))
            structured_score = base_score
            structured_score *= 0.64 + (0.42 * public_compatibility) + (0.12 * event_signal)
            structured_score += 0.56 * public_compatibility
            structured_score += 0.16 * event_focus_coverage
            structured_score += 0.08 * event_overlap
            if query_subject_signature:
                structured_score += 0.62 if subject_match else 0.18 * subject_overlap
                if not subject_match and subject_overlap < 0.55:
                    structured_score -= 0.30
            if temporal_query:
                if resolved_date:
                    structured_score += 0.14 * public_compatibility
                if time_match:
                    structured_score += 0.10 * public_compatibility
                if record.relation == "event_date":
                    structured_score += 0.10
                if source_kind == "public_dialog_time":
                    structured_score += 0.08 * max(public_compatibility, event_signal)
                elif source_kind == "public_dialog_profile":
                    structured_score -= 0.08 * max(0.0, 1.0 - public_compatibility)
            elif source_kind == "public_dialog_profile" and semantic_match:
                structured_score += 0.08 * public_compatibility
            elif source_kind == "public_dialog_event":
                structured_score += 0.05 * event_signal
            generic_prior_score = (
                (generic_recency_prior * (0.18 + (0.22 * public_prior_gate)))
                + (generic_salience_prior * (0.04 + (0.16 * public_prior_gate)))
                + (generic_confidence_prior * (0.04 + (0.12 * public_prior_gate)))
                + (generic_authority_prior * (0.08 + (0.12 * public_prior_gate)))
            )
            score = structured_score + generic_prior_score
            score -= 0.22 * public_conflict
            if public_compatibility < 0.12:
                score -= 0.14
            public_profile = {
                **dict(public_profile or {}),
                "prior_gate": public_prior_gate,
                "structured_score": structured_score,
                "generic_prior_score": generic_prior_score,
                "base_score": base_score,
            }
            match_reason = _public_match_reason(
                record,
                {**analysis, "query": query},
                speaker_match=exact_speaker_match,
                semantic_match=semantic_match,
                subject_match=subject_match,
                event_overlap=event_overlap,
                target_status_match=target_status_match,
                time_match=time_match,
            )
        profile_delta, profile_reason = profile_query_score_delta(
            query=query_text,
            query_tokens=query_tokens,
            category=record.category,
            source_kind=record.source_kind,
            semantic_slot=metadata.get("semantic_slot", ""),
            value=record.value,
            anchors=record.anchor_concepts,
            metadata=metadata,
        )
        if profile_delta > 0:
            score += profile_delta
            if profile_reason:
                match_reason = ",".join(_dedupe([match_reason, profile_reason], max_items=4))
        if memory_role == "assistant" and canonical_slot_key:
            source_head = next(iter(self._slot_active_records(canonical_slot_key)), None)
            if (
                source_head is not None
                and isinstance(source_head.metadata, dict)
                and _normalize(source_head.metadata.get("memory_role", "")) == "user"
                and _normalize(source_head.metadata.get("authority", "")) == "source"
                and _normalize(source_head.value) != _normalize(record.value)
            ):
                score -= 0.25
        if history_mode:
            if _is_active_record_state(record.state):
                score -= 0.08
            else:
                score += 0.26
        elif _is_active_record_state(record.state):
            score += 0.24
            if _normalize(record.state) == "parallel_active":
                score += 0.06
        else:
            score -= 0.25
        memory_signature_tokens = set(_tokenize(metadata.get("memory_signature", "")))
        state_signature_tokens = set(_tokenize(metadata.get("state_signature", "")))
        if query_tokens and memory_signature_tokens:
            score += 0.12 * (len(query_tokens & memory_signature_tokens) / max(1, len(query_tokens)))
        if query_tokens and state_signature_tokens:
            score += 0.08 * (len(query_tokens & state_signature_tokens) / max(1, len(query_tokens)))
        lowered = _normalize(record.value)
        if any(marker in lowered for marker in _NEGATION_MARKERS) and not hints.intersection({"constraint", "history"}):
            score -= 0.08
        return score, match_reason, public_profile

    def _expanded_selection(
        self,
        *,
        selected: Sequence[tuple[float, SessionMemoryRecordV2]],
        scored: Sequence[tuple[float, SessionMemoryRecordV2]],
        query: str,
        hints: set[str],
        history_mode: bool,
        top_k: int,
        public_analysis: Dict[str, Any] | None = None,
        public_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> List[tuple[float, SessionMemoryRecordV2]]:
        expanded = list(selected)
        seen = {record.memory_id for _, record in expanded}
        query_text = _normalize(query)
        analysis = dict(public_analysis or {})

        if analysis:
            cluster_roots = {
                _public_slot_root(record.slot_key)
                for _, record in expanded
                if _clean_text(record.source_kind).startswith("public_dialog")
            }
            cluster_signatures = {
                _normalize(dict(record.metadata or {}).get("event_signature", ""))
                for _, record in expanded
                if _clean_text(record.source_kind).startswith("public_dialog")
            }
            cluster_signatures.discard("")
            per_root: Dict[str, int] = defaultdict(int)
            for _, record in expanded:
                if _clean_text(record.source_kind).startswith("public_dialog"):
                    per_root[_public_slot_root(record.slot_key)] += 1
            public_limit = min(max(top_k + 4, top_k), 12)
            per_event_cap = 3 if bool(analysis.get("is_temporal", False)) else 2
            for score, record in scored:
                if record.memory_id in seen or not _clean_text(record.source_kind).startswith("public_dialog"):
                    continue
                profile = dict((public_profiles or {}).get(record.memory_id, {}) or {})
                root = _clean_text(profile.get("cluster_root", "")) or _public_slot_root(record.slot_key)
                signature = _normalize(profile.get("signature", "")) or _normalize(dict(record.metadata or {}).get("event_signature", ""))
                compatibility = _clamp01(profile.get("compatibility", 0.0))
                event_focus_coverage = _clamp01(profile.get("event_focus_coverage", profile.get("event_overlap", 0.0)))
                semantic_match = bool(profile.get("semantic_match", False))
                time_match = bool(profile.get("time_match", False))
                cluster_bonus = 0.0
                if root and root in cluster_roots:
                    cluster_bonus += 0.12
                if signature and signature in cluster_signatures:
                    cluster_bonus += 0.14
                expansion_score = compatibility + cluster_bonus + (0.06 if (semantic_match or time_match) else 0.0)
                if event_focus_coverage >= 0.8:
                    expansion_score += 0.06
                if expansion_score < 0.18:
                    continue
                if root and per_root.get(root, 0) >= per_event_cap and expansion_score < 0.34:
                    continue
                boost = 0.02 + (0.06 * compatibility) + (0.04 * event_focus_coverage)
                semantic_slot = _normalize(dict(record.metadata or {}).get("semantic_slot", ""))
                if bool(analysis.get("is_temporal", False)) and semantic_slot == "event_time":
                    boost += 0.05
                elif _normalize(record.source_kind) in {"public_dialog_event", "public_dialog_profile"}:
                    boost += 0.03
                expanded.append((score + boost, record))
                seen.add(record.memory_id)
                if root:
                    cluster_roots.add(root)
                    per_root[root] += 1
                if signature:
                    cluster_signatures.add(signature)
                if len(expanded) >= public_limit:
                    break

        if "path" in hints:
            frontier = {
                _clean_text(anchor)
                for _, record in expanded
                for anchor in record.anchor_concepts[:6]
                if _clean_text(anchor)
            }
            mentioned = {
                _clean_text(anchor)
                for _, record in scored
                for anchor in record.anchor_concepts[:6]
                if _clean_text(anchor) and _normalize(anchor) in query_text
            }
            frontier.update(mentioned)
            path_limit = min(max(top_k + 4, top_k), 12)
            for _ in range(4):
                added = False
                for score, record in scored:
                    if record.memory_id in seen or record.relation != "path_edge":
                        continue
                    anchors = {_clean_text(anchor) for anchor in record.anchor_concepts[:6] if _clean_text(anchor)}
                    if frontier and not frontier.intersection(anchors):
                        continue
                    expanded.append((score + 0.03, record))
                    seen.add(record.memory_id)
                    frontier.update(anchors)
                    added = True
                    if len(expanded) >= path_limit:
                        break
                if not added or len(expanded) >= path_limit:
                    break

        if history_mode or _query_requests_current_pair_clean(query):
            pair_limit = min(max(top_k + 2, top_k), 10)
            for score, record in list(expanded):
                if not _is_active_record_state(record.state):
                    for current in self._slot_active_records(record.slot_key)[:2]:
                        if current.memory_id in seen or current.value == record.value:
                            continue
                        expanded.append((max(0.05, score - 0.01), current))
                        seen.add(current.memory_id)
                        if len(expanded) >= pair_limit:
                            break
                if _is_active_record_state(record.state):
                    for previous_id in record.supersedes[:2]:
                        previous = self.records_by_id.get(previous_id)
                        if previous and previous.memory_id not in seen:
                            expanded.append((max(0.05, score - 0.03), previous))
                            seen.add(previous.memory_id)
                            if len(expanded) >= pair_limit:
                                break
                    for parallel in self._slot_active_records(record.slot_key):
                        if parallel.memory_id in seen or parallel.memory_id == record.memory_id:
                            continue
                        expanded.append((max(0.05, score - 0.015), parallel))
                        seen.add(parallel.memory_id)
                        if len(expanded) >= pair_limit:
                            break
                if len(expanded) >= pair_limit:
                    break

        expanded.sort(key=lambda item: (item[0], _is_active_record_state(item[1].state), item[1].turn_index), reverse=True)
        selection_limit = min(
            max(
                top_k
                + (
                    4
                    if "path" in hints
                    else 3
                    if analysis
                    else 2
                    if (history_mode or _query_requests_current_pair_clean(query))
                    else 0
                ),
                top_k,
            ),
            12,
        )
        return expanded[:selection_limit]

    def _noise_false_candidates(
        self,
        *,
        query: str,
        query_tokens: set[str],
        top_k: int,
    ) -> List[tuple[float, SessionMemoryRecordV2]]:
        candidates: List[tuple[float, SessionMemoryRecordV2]] = []
        if not _query_requests_inactive_history(query):
            return candidates
        for turn in reversed(self.turn_log):
            if _normalize(turn.get("kind", "")) != "noise":
                continue
            text = _clean_text(turn.get("text", ""))
            if not text:
                continue
            value = _extract_noise_value(text)
            if not value:
                continue
            turn_index = int(turn.get("turn_index", 0) or 0)
            value_tokens = set(_tokenize(value))
            overlap = len(query_tokens & value_tokens) / max(1, len(query_tokens | value_tokens)) if query_tokens or value_tokens else 0.0
            recency = 1.0 if self.turn_index <= 0 else max(0.0, 1.0 - ((self.turn_index - turn_index) / max(1.0, float(self.turn_index))))
            score = 1.28 + (0.08 * overlap) + (0.1 * recency)
            anchor = next((token for token in _tokenize(value) if token not in {"should", "not", "become", "active", "requirement", "noise"}), "noise")
            record = SessionMemoryRecordV2(
                memory_id=f"noise.turn.{turn_index}",
                category="noise",
                slot_key=f"noise.turn.{turn_index}",
                value=value,
                relation="inactive_note",
                anchor_concepts=_dedupe([anchor, "noise"], max_items=4),
                evidence_anchors=_dedupe([anchor, "noise"], max_items=4),
                salience=0.92,
                confidence=0.88,
                source_kind="noise_turn",
                turn_index=turn_index,
                state="false",
                metadata={"turn_kind": "noise", "inactive_candidate": True, "source_text": text},
            )
            candidates.append((round(score, 6), record))
            if len(candidates) >= max(1, top_k):
                break
        return candidates

    def retrieve(self, query: str, *, top_k: int = 6) -> Dict[str, Any]:
        query_text = _clean_text(query)
        query_tokens = set(_tokenize(query_text))
        hints = set(infer_category_hints(query_text))
        history_mode = infer_history_mode_clean(query_text)
        public_analysis = _public_query_analysis(query_text, query_tokens)
        scored: List[tuple[float, SessionMemoryRecordV2]] = []
        suppressed: List[SessionMemoryRecordV2] = []
        match_reasons: Dict[str, str] = {}
        public_profiles: Dict[str, Dict[str, Any]] = {}
        for record in self.records_by_id.values():
            if _normalize(dict(record.metadata or {}).get("memory_layer", "")) == "slow":
                continue
            score, match_reason, public_profile = self._score_record(
                record,
                query=query_text,
                query_tokens=query_tokens,
                hints=hints,
                history_mode=history_mode,
                public_analysis=public_analysis,
            )
            if score <= 0:
                continue
            if not _is_active_record_state(record.state) and not history_mode:
                suppressed.append(record)
                continue
            scored.append((score, record))
            if match_reason:
                match_reasons[record.memory_id] = match_reason
            if public_profile:
                public_profiles[record.memory_id] = dict(public_profile)
        if _query_requests_inactive_history(query_text):
            scored.extend(self._noise_false_candidates(query=query_text, query_tokens=query_tokens, top_k=max(1, top_k)))
        if not scored:
            fast_records = [
                record
                for record in self.records_by_id.values()
                if _normalize(dict(record.metadata or {}).get("memory_layer", "")) != "slow"
            ]
            for record in sorted(fast_records, key=lambda item: item.turn_index, reverse=True)[: top_k * 2]:
                fallback_score = 0.05 + (0.2 if _is_active_record_state(record.state) else 0.0)
                if not _is_active_record_state(record.state) and not history_mode:
                    suppressed.append(record)
                    continue
                scored.append((fallback_score, record))
        scored.sort(key=lambda item: (item[0], _is_active_record_state(item[1].state), item[1].turn_index), reverse=True)
        if history_mode:
            history_scored = [item for item in scored if not _is_active_record_state(item[1].state)]
            active_scored = [item for item in scored if _is_active_record_state(item[1].state)]
            selected = [*history_scored[: max(1, top_k)], *active_scored[: max(0, top_k - len(history_scored[: max(1, top_k)]))]]
            selected = selected[: max(1, top_k)]
        else:
            selected = scored[: max(1, top_k)]
        selected = self._expanded_selection(
            selected=selected,
            scored=scored,
            query=query_text,
            hints=hints,
            history_mode=history_mode,
            top_k=max(1, top_k),
            public_analysis=public_analysis,
            public_profiles=public_profiles,
        )
        seed_memory_ids = [record.memory_id for _, record in selected]
        depth_chain = self.depth_chain_for_query(
            query_text,
            seed_memory_ids=seed_memory_ids,
            top_k=max(3, min(8, top_k)),
        )
        if depth_chain.get("enabled") and depth_chain.get("nodes"):
            seen_selected = {record.memory_id for _, record in selected}
            chain_limit = min(max(top_k + 4, top_k), 12)
            for node in depth_chain.get("nodes", []) or []:
                memory_id = _clean_text(dict(node).get("memory_id", ""))
                record = self.records_by_id.get(memory_id)
                if (
                    record is None
                    or memory_id in seen_selected
                    or _normalize(dict(record.metadata or {}).get("memory_layer", "")) == "slow"
                ):
                    continue
                layer = _normalize(dict(record.metadata or {}).get("depth_layer", "")) or "core_view"
                chain_score = 0.58 + (0.04 if _is_active_record_state(record.state) else 0.0)
                chain_score += max(0.0, 0.12 - (0.012 * _DEPTH_LAYER_ORDER.get(layer, 8)))
                selected.append((round(chain_score, 6), record))
                seen_selected.add(memory_id)
                if len(selected) >= chain_limit:
                    break
        active_hits = [record for _, record in selected if _is_active_record_state(record.state)]
        history_hits = [record for _, record in selected if not _is_active_record_state(record.state) and record.state != "false"]
        overwrite_hits = [record for record in history_hits if record.state == "superseded"]
        stale_hits = overwrite_hits if not history_mode else []
        false_hits: List[SessionMemoryRecordV2] = [record for _, record in selected if record.state == "false"]

        concepts: Dict[str, Dict[str, Any]] = {}
        relations: List[Dict[str, Any]] = []
        seen_relations = set()
        selected_payload: List[Dict[str, Any]] = []
        for score, record in selected:
            record_payload = record.to_dict()
            payload_metadata = dict(record_payload.get("metadata", {}) or {})
            payload_metadata["match_reason"] = match_reasons.get(record.memory_id, "")
            if record.memory_id in public_profiles:
                profile = dict(public_profiles.get(record.memory_id, {}) or {})
                payload_metadata["public_match_compatibility"] = round(float(profile.get("compatibility", 0.0) or 0.0), 6)
                payload_metadata["public_match_conflict"] = round(float(profile.get("conflict", 0.0) or 0.0), 6)
                payload_metadata["public_event_focus_coverage"] = round(float(profile.get("event_focus_coverage", 0.0) or 0.0), 6)
                payload_metadata["public_event_signal"] = round(float(profile.get("event_signal", 0.0) or 0.0), 6)
                payload_metadata["public_subject_match"] = bool(profile.get("subject_match", False))
                payload_metadata["public_subject_overlap"] = round(float(profile.get("subject_overlap", 0.0) or 0.0), 6)
                payload_metadata["public_prior_gate"] = round(float(profile.get("prior_gate", 0.0) or 0.0), 6)
                payload_metadata["public_structured_score"] = round(float(profile.get("structured_score", 0.0) or 0.0), 6)
                payload_metadata["public_generic_prior_score"] = round(float(profile.get("generic_prior_score", 0.0) or 0.0), 6)
                payload_metadata["public_base_score"] = round(float(profile.get("base_score", 0.0) or 0.0), 6)
            record_payload["metadata"] = payload_metadata
            selected_payload.append(
                {
                    "score": round(float(score), 6),
                    **record_payload,
                }
            )
            concepts.setdefault(record.value, {"concept": record.value, "type": record.category, "source_kind": record.source_kind})
            slot_node = f"slot::{record.slot_key}"
            concepts.setdefault(slot_node, {"concept": slot_node, "type": "slot_head", "source_kind": "memory_slot"})
            slot_relation = _slot_relation_for_state(record.state)
            rel_key = (slot_node, record.value, slot_relation)
            if rel_key not in seen_relations:
                seen_relations.add(rel_key)
                relations.append(
                    {
                        "from": slot_node,
                        "to": record.value,
                        "relation": slot_relation,
                        "weight": round(max(0.3, min(0.98, 0.5 + score * 0.4)), 6),
                        "source_kind": record.source_kind,
                        "memory_id": record.memory_id,
                    }
                )
            for anchor in record.anchor_concepts[:4]:
                concepts.setdefault(anchor, {"concept": anchor, "type": "concept", "source_kind": record.source_kind})
                anchor_key = (anchor, record.value, record.relation)
                if anchor_key in seen_relations:
                    continue
                seen_relations.add(anchor_key)
                relations.append(
                    {
                        "from": anchor,
                        "to": record.value,
                        "relation": record.relation,
                        "weight": round(max(0.32, min(0.98, 0.42 + score * 0.4)), 6),
                        "source_kind": record.source_kind,
                        "memory_id": record.memory_id,
                    }
                )
            if history_mode:
                for previous_id in record.supersedes:
                    previous = self.records_by_id.get(previous_id)
                    if not previous:
                        continue
                    concepts.setdefault(previous.value, {"concept": previous.value, "type": previous.category, "source_kind": previous.source_kind})
                    relation_key = (record.value, previous.value, "supersedes")
                    if relation_key not in seen_relations:
                        seen_relations.add(relation_key)
                        relations.append(
                            {
                                "from": record.value,
                                "to": previous.value,
                                "relation": "supersedes",
                                "weight": 0.91,
                                "source_kind": record.source_kind,
                                "memory_id": record.memory_id,
                            }
                        )

        query_id = f"query:{int(self.audit_event_totals.get('retrieval_log', 0) or 0) + 1}"
        self._append_audit_event(
            "retrieval_log",
            {
                "query_id": query_id,
                "turn_index": int(self.turn_index),
                "query": query_text,
                "history_mode": history_mode,
                "memory_ids": [record["memory_id"] for record in selected_payload],
            },
        )
        return {
            "query_id": query_id,
            "concepts": list(concepts.values()),
            "relations": relations,
            "hits": selected_payload,
            "active_hits": [record.to_dict() for record in active_hits],
            "history_hits": [record.to_dict() for record in history_hits],
            "stale_hits": [record.to_dict() for record in stale_hits],
            "overwrite_hits": [record.to_dict() for record in overwrite_hits],
            "false_hits": [record.to_dict() for record in false_hits],
            "suppressed_hits": [record.to_dict() for record in sorted(suppressed, key=lambda item: item.turn_index, reverse=True)[:top_k]],
            "context_token_estimate": sum(_estimate_tokens(item["value"]) + sum(_estimate_tokens(anchor) for anchor in item.get("anchor_concepts", [])) for item in selected_payload),
            "metadata": {
                "history_mode": history_mode,
                "inactive_history_mode": _query_requests_inactive_history(query_text),
                "slot_heads": len(self.slot_heads),
                "records": len(self.records_by_id),
                "noise_turn_count": self.noise_turn_count,
                "public_query_analysis": public_analysis,
                "memory_chain": depth_chain,
                "memory_chain_enabled": bool(depth_chain.get("enabled", False)),
                "memory_chain_node_count": int(depth_chain.get("node_count", 0) or 0),
                "memory_chain_edge_count": int(depth_chain.get("edge_count", 0) or 0),
            },
        }

    def register_answer_support(self, *, answer_id: str, memory_ids: Sequence[str], query_id: str = "", answer_text: str = "") -> None:
        self._append_audit_event(
            "answer_support_log",
            {
                "answer_id": answer_id or f"answer:{int(self.audit_event_totals.get('answer_support_log', 0) or 0) + 1}",
                "query_id": query_id,
                "memory_ids": list(memory_ids),
                "answer_text": _clean_text(answer_text),
                "turn_index": int(self.turn_index),
            },
        )

    def summary(self) -> Dict[str, Any]:
        active_records = [record for record in self.records_by_id.values() if _is_active_record_state(record.state)]
        parallel_active = [record for record in active_records if _normalize(record.state) == "parallel_active"]
        superseded = [record for record in self.records_by_id.values() if record.state == "superseded"]
        suspect_records = [record for record in self.records_by_id.values() if _is_suspect_record_state(record.state)]
        promoted_suspect_records = [record for record in self.records_by_id.values() if _normalize(record.state) == "promoted"]
        return {
            "turn_index": int(self.turn_index),
            "records": len(self.records_by_id),
            "memory_edges": len(self.memory_edges),
            "subject_depth_heads": sum(len(heads) for heads in self.subject_depth_heads.values()),
            "active_slots": len(self.slot_heads),
            "active_records": len(active_records),
            "parallel_active_records": len(parallel_active),
            "superseded_records": len(superseded),
            "suspect_records": len(suspect_records),
            "promoted_suspect_records": len(promoted_suspect_records),
            "noise_turn_count": self.noise_turn_count,
            "turn_events": len(self.turn_log),
            "retrieval_events": len(self.retrieval_log),
            "answer_support_events": len(self.answer_support_log),
            "audit_turn_events": int(self.audit_event_totals.get("turn_log", 0) or 0),
            "audit_retrieval_events": int(self.audit_event_totals.get("retrieval_log", 0) or 0),
            "audit_answer_support_events": int(self.audit_event_totals.get("answer_support_log", 0) or 0),
            "audit_trimmed_counts": dict(self.audit_trimmed_counts),
            "audit_retention": int(self.audit_retention),
            "persistence_backend": self.persistence_backend,
            "persistence_path": self.persistence_path,
        }

    def to_dict(self, *, mode: str = "full") -> Dict[str, Any]:
        payload = self._core_payload()
        normalized_mode = _normalize(mode) or "full"
        if normalized_mode == "light":
            payload["audit"] = {
                "retained": {
                    "turn_log": len(self.turn_log),
                    "retrieval_log": len(self.retrieval_log),
                    "answer_support_log": len(self.answer_support_log),
                },
                "totals": dict(self.audit_event_totals),
                "trimmed": dict(self.audit_trimmed_counts),
                "audit_retention": int(self.audit_retention),
            }
            return payload
        payload.update(self._audit_payload())
        return payload

    def export_graph(self, *, snapshot_points: Sequence[int] | None = None, mode: str = "full") -> Dict[str, Any]:
        snapshot_set = {int(point) for point in (snapshot_points or []) if int(point) > 0}
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        node_ids = set()
        edge_ids = set()

        def add_node(node_id: str, node_type: str, **payload: Any) -> None:
            if node_id in node_ids:
                return
            node_ids.add(node_id)
            nodes.append({"id": node_id, "type": node_type, **payload})

        def add_edge(source: str, target: str, relation: str, **payload: Any) -> None:
            key = (source, target, relation)
            if key in edge_ids:
                return
            edge_ids.add(key)
            edges.append({"from": source, "to": target, "relation": relation, **payload})

        for event in self.turn_log:
            turn_id = event["turn_id"]
            add_node(
                turn_id,
                "turn",
                turn_index=event["turn_index"],
                kind=event["kind"],
                text=event["text"],
                speaker=event.get("speaker", "user"),
                assistant_text=event.get("assistant_text", ""),
                writeback_class=event.get("writeback_class", ""),
                metadata=dict(event.get("metadata", {}) or {}),
            )
            for record_id in event.get("record_ids", []) or []:
                add_edge(turn_id, record_id, "writes")

        for record in self.records_by_id.values():
            add_node(
                record.memory_id,
                "memory_record",
                category=record.category,
                slot_key=record.slot_key,
                value=record.value,
                state=record.state,
                turn_index=record.turn_index,
                depth_layer=_normalize(dict(record.metadata or {}).get("depth_layer", "")) or "core_view",
                subject_signature=self._record_subject_signature(record),
            )
            slot_node = f"slot::{record.slot_key}"
            add_node(slot_node, "slot_head", slot_key=record.slot_key)
            add_edge(slot_node, record.memory_id, _slot_relation_for_state(record.state))
            for previous_id in record.supersedes:
                add_edge(record.memory_id, previous_id, "supersedes")
            for anchor in record.anchor_concepts:
                concept_id = f"concept::{anchor}"
                add_node(concept_id, "concept", concept=anchor)
                add_edge(record.memory_id, concept_id, "anchors_to")

        for edge in self.memory_edges.values():
            if edge.source_memory_id in self.records_by_id and edge.target_memory_id in self.records_by_id:
                add_edge(
                    edge.source_memory_id,
                    edge.target_memory_id,
                    edge.edge_type,
                    weight=round(float(edge.score), 6),
                    edge_id=edge.edge_id,
                    metadata=dict(edge.metadata or {}),
                )

        for event in self.retrieval_log:
            query_id = event["query_id"]
            add_node(query_id, "query", query=event["query"], turn_index=event["turn_index"], history_mode=event["history_mode"])
            for record_id in event.get("memory_ids", []) or []:
                add_edge(record_id, query_id, "retrieved_by")

        for event in self.answer_support_log:
            answer_id = event["answer_id"]
            add_node(answer_id, "answer", answer_text=event["answer_text"], turn_index=event["turn_index"])
            if event.get("query_id"):
                add_edge(event["query_id"], answer_id, "answered_as")
            for record_id in event.get("memory_ids", []) or []:
                add_edge(record_id, answer_id, "supports_answer")

        snapshots: List[Dict[str, Any]] = []
        if snapshot_set:
            turn_events = sorted(self.turn_log, key=lambda item: item["turn_index"])
            for point in sorted(snapshot_set):
                visible_turns = [event for event in turn_events if int(event["turn_index"]) <= point]
                visible_turn_ids = {event["turn_id"] for event in visible_turns}
                visible_record_ids = {record_id for event in visible_turns for record_id in event.get("record_ids", []) or []}
                active_records = [
                    record.to_dict()
                    for record in self.records_by_id.values()
                    if record.turn_index <= point and (_is_active_record_state(record.state) or record.memory_id in visible_record_ids)
                ]
                snapshots.append(
                    {
                        "turn_index": point,
                        "visible_turns": len(visible_turns),
                        "visible_records": len(active_records),
                        "slot_heads": {
                            slot: record_id
                            for slot, record_id in self.slot_heads.items()
                            if self.records_by_id.get(record_id) and self.records_by_id[record_id].turn_index <= point
                        },
                        "records": active_records,
                    }
                )

        return {
            "summary": {
                **self.summary(),
                "graph_nodes": len(nodes),
                "graph_edges": len(edges),
            },
            "nodes": nodes,
            "edges": edges,
            "snapshots": snapshots if _normalize(mode) != "light" else [],
        }

    def export_mermaid(self, *, max_records: int = 96) -> str:
        lines = ["graph TD"]
        records = sorted(self.records_by_id.values(), key=lambda item: (not _is_active_record_state(item.state), -item.turn_index))[:max_records]
        for record in records:
            record_id = record.memory_id.replace(":", "_").replace(".", "_").replace("-", "_")
            slot_id = f"slot_{record.slot_key}".replace(":", "_").replace(".", "_").replace("-", "_")
            lines.append(f'    {slot_id}["{record.slot_key}"]')
            lines.append(f'    {record_id}["{record.category}: {record.value[:48]}"]')
            relation = _slot_relation_for_state(record.state)
            lines.append(f"    {slot_id} -->|{relation}| {record_id}")
            for previous_id in record.supersedes[:2]:
                prev = previous_id.replace(":", "_").replace(".", "_").replace("-", "_")
                lines.append(f"    {record_id} -->|supersedes| {prev}")
            for anchor in record.anchor_concepts[:2]:
                anchor_id = f"concept_{anchor}".replace(":", "_").replace(".", "_").replace("-", "_").replace(" ", "_")
                lines.append(f'    {anchor_id}["{anchor[:36]}"]')
                lines.append(f"    {record_id} -->|anchors_to| {anchor_id}")
        return "\n".join(lines) + "\n"

    def storage_bytes(self) -> int:
        return len(json.dumps(self.to_dict(mode="full"), ensure_ascii=False).encode("utf-8"))


class StaleGraphSnapshotError(RuntimeError):
    pass


class SQLiteSessionMemoryStore:
    def __init__(self, storage_path: str | Path, *, audit_retention: int = 256) -> None:
        self.storage_path = Path(storage_path).expanduser().resolve()
        self.audit_retention = max(1, int(audit_retention))
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.storage_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    @contextmanager
    def _managed_connection(self) -> Iterable[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self._managed_connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS records (
                    scope_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    slot_key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    anchor_concepts_json TEXT NOT NULL,
                    evidence_anchors_json TEXT NOT NULL,
                    salience REAL NOT NULL,
                    confidence REAL NOT NULL,
                    source_kind TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    supersedes_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (scope_id, memory_id)
                );
                CREATE INDEX IF NOT EXISTS idx_records_scope_slot ON records(scope_id, slot_key);
                CREATE INDEX IF NOT EXISTS idx_records_scope_turn ON records(scope_id, turn_index);
                CREATE TABLE IF NOT EXISTS slot_heads (
                    scope_id TEXT NOT NULL,
                    slot_key TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    PRIMARY KEY (scope_id, slot_key)
                );
                CREATE TABLE IF NOT EXISTS slot_history (
                    scope_id TEXT NOT NULL,
                    slot_key TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    memory_id TEXT NOT NULL,
                    PRIMARY KEY (scope_id, slot_key, ordinal)
                );
                CREATE TABLE IF NOT EXISTS memory_edges (
                    scope_id TEXT NOT NULL,
                    edge_id TEXT NOT NULL,
                    source_memory_id TEXT NOT NULL,
                    target_memory_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    score REAL NOT NULL,
                    model_score REAL NOT NULL,
                    evidence_turn INTEGER NOT NULL,
                    evidence TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (scope_id, edge_id)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_edges_scope_source ON memory_edges(scope_id, source_memory_id);
                CREATE INDEX IF NOT EXISTS idx_memory_edges_scope_target ON memory_edges(scope_id, target_memory_id);
                CREATE TABLE IF NOT EXISTS subject_depth_heads (
                    scope_id TEXT NOT NULL,
                    subject_signature TEXT NOT NULL,
                    depth_layer TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    PRIMARY KEY (scope_id, subject_signature, depth_layer)
                );
                CREATE TABLE IF NOT EXISTS audit_turn_log (
                    scope_id TEXT NOT NULL,
                    event_index INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (scope_id, event_index)
                );
                CREATE TABLE IF NOT EXISTS audit_retrieval_log (
                    scope_id TEXT NOT NULL,
                    event_index INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (scope_id, event_index)
                );
                CREATE TABLE IF NOT EXISTS audit_answer_support (
                    scope_id TEXT NOT NULL,
                    event_index INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (scope_id, event_index)
                );
                CREATE TABLE IF NOT EXISTS meta (
                    scope_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    PRIMARY KEY (scope_id, key)
                );
                """
            )

    def clear_scope(self, scope_id: str) -> None:
        normalized_scope = _clean_text(scope_id)
        if not normalized_scope:
            return
        with self._managed_connection() as connection:
            for table in (
                "records",
                "slot_heads",
                "slot_history",
                "memory_edges",
                "subject_depth_heads",
                "audit_turn_log",
                "audit_retrieval_log",
                "audit_answer_support",
                "meta",
            ):
                connection.execute(f"DELETE FROM {table} WHERE scope_id = ?", (normalized_scope,))

    def _refresh_authoritative_slow_state(
        self,
        connection: sqlite3.Connection,
        scope_id: str,
        graph: SessionMemoryGraphV2,
    ) -> None:
        """Keep control-plane slow revisions authoritative over stale graph snapshots."""
        record_rows = connection.execute(
            """
            SELECT memory_id, category, slot_key, value, relation, anchor_concepts_json,
                   evidence_anchors_json, salience, confidence, source_kind, turn_index,
                   state, supersedes_json, metadata_json
            FROM records WHERE scope_id = ?
            """,
            (scope_id,),
        ).fetchall()
        slow_records: Dict[str, SessionMemoryRecordV2] = {}
        slow_slots: set[str] = set()
        for row in record_rows:
            metadata = dict(json.loads(row["metadata_json"]) or {})
            if (
                _normalize(metadata.get("memory_layer")) != "slow"
                or _normalize(metadata.get("content_variant"))
                != "slow_memory_capsule"
            ):
                continue
            record = SessionMemoryRecordV2(
                memory_id=str(row["memory_id"]),
                category=str(row["category"]),
                slot_key=str(row["slot_key"]),
                value=str(row["value"]),
                relation=str(row["relation"]),
                anchor_concepts=[
                    str(item) for item in json.loads(row["anchor_concepts_json"]) or []
                ],
                evidence_anchors=[
                    str(item) for item in json.loads(row["evidence_anchors_json"]) or []
                ],
                salience=float(row["salience"] or 0.0),
                confidence=float(row["confidence"] or 0.0),
                source_kind=str(row["source_kind"]),
                turn_index=int(row["turn_index"] or 0),
                state=str(row["state"]),
                supersedes=[
                    str(item) for item in json.loads(row["supersedes_json"]) or []
                ],
                metadata=metadata,
            )
            slow_records[record.memory_id] = record
            slow_slots.add(record.slot_key)

        graph.records_by_id = {
            memory_id: record
            for memory_id, record in graph.records_by_id.items()
            if _normalize(dict(record.metadata or {}).get("memory_layer")) != "slow"
        }
        graph.records_by_id.update(slow_records)
        for slot_key in list(graph.slot_heads):
            if slot_key in slow_slots or _normalize(slot_key).startswith("slow."):
                graph.slot_heads.pop(slot_key, None)
        for slot_key in list(graph.slot_history):
            if slot_key in slow_slots or _normalize(slot_key).startswith("slow."):
                graph.slot_history.pop(slot_key, None)
        if slow_slots:
            head_rows = connection.execute(
                "SELECT slot_key,memory_id FROM slot_heads WHERE scope_id=?",
                (scope_id,),
            ).fetchall()
            for row in head_rows:
                if str(row["slot_key"]) in slow_slots:
                    graph.slot_heads[str(row["slot_key"])] = str(row["memory_id"])
            history_rows = connection.execute(
                "SELECT slot_key,memory_id FROM slot_history WHERE scope_id=? ORDER BY slot_key,ordinal",
                (scope_id,),
            ).fetchall()
            for row in history_rows:
                if str(row["slot_key"]) in slow_slots:
                    graph.slot_history[str(row["slot_key"])].append(
                        str(row["memory_id"])
                    )

        graph.memory_edges = {
            edge_id: edge
            for edge_id, edge in graph.memory_edges.items()
            if _normalize(dict(edge.metadata or {}).get("edge_source"))
            != "slow_graph_control_plane"
        }
        edge_rows = connection.execute(
            """
            SELECT edge_id,source_memory_id,target_memory_id,edge_type,score,model_score,
                   evidence_turn,evidence,metadata_json
            FROM memory_edges WHERE scope_id=?
            """,
            (scope_id,),
        ).fetchall()
        for row in edge_rows:
            metadata = dict(json.loads(row["metadata_json"]) or {})
            if _normalize(metadata.get("edge_source")) != "slow_graph_control_plane":
                continue
            edge = SessionMemoryEdgeV2(
                edge_id=str(row["edge_id"]),
                source_memory_id=str(row["source_memory_id"]),
                target_memory_id=str(row["target_memory_id"]),
                edge_type=str(row["edge_type"]),
                score=float(row["score"] or 0.0),
                model_score=float(row["model_score"] or 0.0),
                evidence_turn=int(row["evidence_turn"] or 0),
                evidence=str(row["evidence"]),
                metadata=metadata,
            )
            graph.memory_edges[edge.edge_id] = edge

    def append_audit_event(
        self,
        scope_id: str,
        field_name: str,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str = "",
    ) -> Dict[str, Any]:
        """Append one audit event without rewriting the graph snapshot."""
        normalized_scope = _clean_text(scope_id)
        audit_fields = {
            "turn_log": (
                "audit_turn_log",
                "audit_turn_events",
                "audit_trimmed_turn_log",
            ),
            "retrieval_log": (
                "audit_retrieval_log",
                "audit_retrieval_events",
                "audit_trimmed_retrieval_log",
            ),
            "answer_support_log": (
                "audit_answer_support",
                "audit_answer_support_events",
                "audit_trimmed_answer_support_log",
            ),
        }
        if not normalized_scope:
            raise ValueError("scope_id must be non-empty for SQLite persistence")
        if field_name not in audit_fields:
            raise KeyError(f"Unknown audit field: {field_name}")
        table, total_key, trimmed_key = audit_fields[field_name]
        stored_payload = dict(payload)
        normalized_idempotency_key = _clean_text(idempotency_key)
        payload_idempotency_key = _clean_text(stored_payload.get("idempotency_key"))
        if (
            normalized_idempotency_key
            and payload_idempotency_key
            and normalized_idempotency_key != payload_idempotency_key
        ):
            raise ValueError("audit payload idempotency_key conflicts with argument")
        if normalized_idempotency_key:
            stored_payload["idempotency_key"] = normalized_idempotency_key

        with self._managed_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")

            def meta_int(key: str) -> int:
                row = connection.execute(
                    "SELECT value_json FROM meta WHERE scope_id=? AND key=?",
                    (normalized_scope, key),
                ).fetchone()
                return int(json.loads(row["value_json"]) or 0) if row else 0

            if normalized_idempotency_key:
                for existing in connection.execute(
                    f"SELECT payload_json FROM {table} WHERE scope_id=? ORDER BY event_index",
                    (normalized_scope,),
                ):
                    existing_payload = dict(json.loads(existing["payload_json"]) or {})
                    if (
                        _clean_text(existing_payload.get("idempotency_key"))
                        == normalized_idempotency_key
                    ):
                        return {
                            "payload": existing_payload,
                            "event_total": meta_int(total_key),
                            "trimmed_total": meta_int(trimmed_key),
                            "appended": False,
                        }

            event_total = meta_int(total_key) + 1
            if field_name == "retrieval_log":
                stored_payload["query_id"] = f"query:{event_total}"
            event_index = int(
                connection.execute(
                    f"SELECT COALESCE(MAX(event_index),-1)+1 FROM {table} "
                    "WHERE scope_id=?",
                    (normalized_scope,),
                ).fetchone()[0]
            )
            connection.execute(
                f"INSERT INTO {table}(scope_id,event_index,payload_json) VALUES(?,?,?)",
                (
                    normalized_scope,
                    event_index,
                    json.dumps(stored_payload, ensure_ascii=False),
                ),
            )
            retained = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE scope_id=?",
                    (normalized_scope,),
                ).fetchone()[0]
            )
            overflow = max(0, retained - self.audit_retention)
            if overflow:
                expired = connection.execute(
                    f"SELECT event_index FROM {table} WHERE scope_id=? "
                    "ORDER BY event_index LIMIT ?",
                    (normalized_scope, overflow),
                ).fetchall()
                connection.executemany(
                    f"DELETE FROM {table} WHERE scope_id=? AND event_index=?",
                    [(normalized_scope, int(row["event_index"])) for row in expired],
                )
            trimmed_total = meta_int(trimmed_key) + overflow
            for key, value in (
                (total_key, event_total),
                (trimmed_key, trimmed_total),
                ("audit_retention", int(self.audit_retention)),
            ):
                connection.execute(
                    "INSERT INTO meta(scope_id,key,value_json) VALUES(?,?,?) "
                    "ON CONFLICT(scope_id,key) DO UPDATE SET value_json=excluded.value_json",
                    (normalized_scope, key, json.dumps(value, ensure_ascii=False)),
                )
        return {
            "payload": stored_payload,
            "event_total": event_total,
            "trimmed_total": trimmed_total,
            "appended": True,
        }

    def _refresh_authoritative_audit_state(
        self,
        connection: sqlite3.Connection,
        scope_id: str,
        graph: SessionMemoryGraphV2,
    ) -> None:
        meta_rows = connection.execute(
            "SELECT key,value_json FROM meta WHERE scope_id=?",
            (scope_id,),
        ).fetchall()
        meta = {
            str(row["key"]): json.loads(row["value_json"])
            for row in meta_rows
        }
        fields = {
            "turn_log": (
                "audit_turn_log",
                "audit_turn_events",
                "audit_trimmed_turn_log",
            ),
            "retrieval_log": (
                "audit_retrieval_log",
                "audit_retrieval_events",
                "audit_trimmed_retrieval_log",
            ),
            "answer_support_log": (
                "audit_answer_support",
                "audit_answer_support_events",
                "audit_trimmed_answer_support_log",
            ),
        }

        def event_key(event: Mapping[str, Any]) -> str:
            return json.dumps(
                dict(event),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        for field_name, (table, total_key, trimmed_key) in fields.items():
            persisted = [
                dict(json.loads(row["payload_json"]) or {})
                for row in connection.execute(
                    f"SELECT payload_json FROM {table} WHERE scope_id=? "
                    "ORDER BY event_index",
                    (scope_id,),
                )
            ]
            persisted_counts: Dict[str, int] = {}
            for event in persisted:
                key = event_key(event)
                persisted_counts[key] = persisted_counts.get(key, 0) + 1
            observed: Dict[str, int] = {}
            merged = list(persisted)
            appended = 0
            for event in list(getattr(graph, field_name)):
                normalized = dict(event)
                key = event_key(normalized)
                observed[key] = observed.get(key, 0) + 1
                if observed[key] > persisted_counts.get(key, 0):
                    merged.append(normalized)
                    appended += 1
            if len(merged) > self.audit_retention:
                merged = merged[-self.audit_retention :]
            event_total = int(meta.get(total_key, 0) or 0) + appended
            trimmed_total = max(
                int(meta.get(trimmed_key, 0) or 0),
                int(graph.audit_trimmed_counts.get(field_name, 0) or 0),
                event_total - len(merged),
            )
            setattr(graph, field_name, merged)
            graph.audit_event_totals[field_name] = event_total
            graph.audit_trimmed_counts[field_name] = trimmed_total

    def save_graph(
        self,
        scope_id: str,
        graph: SessionMemoryGraphV2,
        *,
        transaction_hook: Callable[[sqlite3.Connection], None] | None = None,
    ) -> None:
        normalized_scope = _clean_text(scope_id)
        if not normalized_scope:
            raise ValueError("scope_id must be non-empty for SQLite persistence")
        graph.configure_persistence(backend="sqlite", path=str(self.storage_path), audit_retention=self.audit_retention)
        raw_expected_revision = getattr(graph, "_storage_revision", None)
        expected_revision = (
            int(raw_expected_revision or 0)
            if raw_expected_revision is not None
            else None
        )
        with self._managed_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            revision_row = connection.execute(
                "SELECT value_json FROM meta WHERE scope_id=? AND key='storage_revision'",
                (normalized_scope,),
            ).fetchone()
            current_revision = (
                int(json.loads(revision_row["value_json"]) or 0)
                if revision_row
                else 0
            )
            if expected_revision is None:
                existing_record_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM records WHERE scope_id=?",
                        (normalized_scope,),
                    ).fetchone()[0]
                )
                if current_revision or existing_record_count:
                    raise StaleGraphSnapshotError(
                        "unversioned graph cannot replace an existing scope; "
                        "load_graph must establish the snapshot revision first"
                    )
                expected_revision = 0
            if expected_revision != current_revision:
                raise StaleGraphSnapshotError(
                    "graph snapshot revision is stale: "
                    f"expected={expected_revision}, current={current_revision}"
                )
            self._refresh_authoritative_slow_state(connection, normalized_scope, graph)
            self._refresh_authoritative_audit_state(
                connection, normalized_scope, graph
            )
            next_revision = current_revision + 1
            audit_tables = {
                "audit_turn_log": list(graph.turn_log),
                "audit_retrieval_log": list(graph.retrieval_log),
                "audit_answer_support": list(graph.answer_support_log),
            }
            meta_entries = {
                "turn_index": int(graph.turn_index),
                "noise_turn_count": int(graph.noise_turn_count),
                "audit_retention": int(graph.audit_retention),
                "audit_turn_events": int(graph.audit_event_totals.get("turn_log", 0) or 0),
                "audit_retrieval_events": int(graph.audit_event_totals.get("retrieval_log", 0) or 0),
                "audit_answer_support_events": int(graph.audit_event_totals.get("answer_support_log", 0) or 0),
                "audit_trimmed_turn_log": int(graph.audit_trimmed_counts.get("turn_log", 0) or 0),
                "audit_trimmed_retrieval_log": int(graph.audit_trimmed_counts.get("retrieval_log", 0) or 0),
                "audit_trimmed_answer_support_log": int(graph.audit_trimmed_counts.get("answer_support_log", 0) or 0),
                "schema_version": 3,
                "storage_revision": next_revision,
            }
            for table in (
                "records",
                "slot_heads",
                "slot_history",
                "memory_edges",
                "subject_depth_heads",
                "audit_turn_log",
                "audit_retrieval_log",
                "audit_answer_support",
                "meta",
            ):
                connection.execute(f"DELETE FROM {table} WHERE scope_id = ?", (normalized_scope,))
            connection.executemany(
                """
                INSERT INTO records (
                    scope_id, memory_id, category, slot_key, value, relation,
                    anchor_concepts_json, evidence_anchors_json, salience, confidence,
                    source_kind, turn_index, state, supersedes_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        normalized_scope,
                        record.memory_id,
                        record.category,
                        record.slot_key,
                        record.value,
                        record.relation,
                        json.dumps(list(record.anchor_concepts), ensure_ascii=False),
                        json.dumps(list(record.evidence_anchors), ensure_ascii=False),
                        float(record.salience),
                        float(record.confidence),
                        record.source_kind,
                        int(record.turn_index),
                        record.state,
                        json.dumps(list(record.supersedes), ensure_ascii=False),
                        json.dumps(dict(record.metadata), ensure_ascii=False),
                    )
                    for record in graph.records_by_id.values()
                ],
            )
            connection.executemany(
                "INSERT INTO slot_heads (scope_id, slot_key, memory_id) VALUES (?, ?, ?)",
                [(normalized_scope, slot_key, memory_id) for slot_key, memory_id in graph.slot_heads.items()],
            )
            connection.executemany(
                "INSERT INTO slot_history (scope_id, slot_key, ordinal, memory_id) VALUES (?, ?, ?, ?)",
                [
                    (normalized_scope, slot_key, ordinal, memory_id)
                    for slot_key, memory_ids in graph.slot_history.items()
                    for ordinal, memory_id in enumerate(memory_ids)
                ],
            )
            connection.executemany(
                """
                INSERT INTO memory_edges (
                    scope_id, edge_id, source_memory_id, target_memory_id, edge_type,
                    score, model_score, evidence_turn, evidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        normalized_scope,
                        edge.edge_id,
                        edge.source_memory_id,
                        edge.target_memory_id,
                        edge.edge_type,
                        float(edge.score),
                        float(edge.model_score),
                        int(edge.evidence_turn),
                        edge.evidence,
                        json.dumps(dict(edge.metadata), ensure_ascii=False),
                    )
                    for edge in graph.memory_edges.values()
                ],
            )
            connection.executemany(
                "INSERT INTO subject_depth_heads (scope_id, subject_signature, depth_layer, memory_id) VALUES (?, ?, ?, ?)",
                [
                    (normalized_scope, subject_signature, depth_layer, memory_id)
                    for subject_signature, heads in graph.subject_depth_heads.items()
                    for depth_layer, memory_id in heads.items()
                ],
            )
            for table, rows in audit_tables.items():
                connection.executemany(
                    f"INSERT INTO {table} (scope_id, event_index, payload_json) VALUES (?, ?, ?)",
                    [
                        (normalized_scope, event_index, json.dumps(dict(payload), ensure_ascii=False))
                        for event_index, payload in enumerate(rows)
                    ],
                )
            connection.executemany(
                "INSERT INTO meta (scope_id, key, value_json) VALUES (?, ?, ?)",
                [(normalized_scope, key, json.dumps(value, ensure_ascii=False)) for key, value in meta_entries.items()],
            )
            if transaction_hook is not None:
                transaction_hook(connection)
        graph._storage_revision = next_revision

    def load_graph(self, scope_id: str) -> SessionMemoryGraphV2:
        normalized_scope = _clean_text(scope_id)
        graph = SessionMemoryGraphV2(
            audit_retention=self.audit_retention,
            persistence_backend="sqlite",
            persistence_path=str(self.storage_path),
        )
        graph._storage_revision = 0
        if not normalized_scope:
            return graph
        with self._managed_connection() as connection:
            meta_rows = connection.execute("SELECT key, value_json FROM meta WHERE scope_id = ?", (normalized_scope,)).fetchall()
            meta = {str(row["key"]): json.loads(row["value_json"]) for row in meta_rows}
            graph._storage_revision = int(meta.get("storage_revision", 0) or 0)
            graph.audit_retention = int(meta.get("audit_retention", graph.audit_retention) or graph.audit_retention)
            graph.turn_index = int(meta.get("turn_index", 0) or 0)
            graph.noise_turn_count = int(meta.get("noise_turn_count", 0) or 0)
            graph.audit_event_totals = {
                "turn_log": int(meta.get("audit_turn_events", 0) or 0),
                "retrieval_log": int(meta.get("audit_retrieval_events", 0) or 0),
                "answer_support_log": int(meta.get("audit_answer_support_events", 0) or 0),
            }
            graph.audit_trimmed_counts = {
                "turn_log": int(meta.get("audit_trimmed_turn_log", 0) or 0),
                "retrieval_log": int(meta.get("audit_trimmed_retrieval_log", 0) or 0),
                "answer_support_log": int(meta.get("audit_trimmed_answer_support_log", 0) or 0),
            }
            record_rows = connection.execute(
                """
                SELECT memory_id, category, slot_key, value, relation, anchor_concepts_json,
                       evidence_anchors_json, salience, confidence, source_kind, turn_index,
                       state, supersedes_json, metadata_json
                FROM records
                WHERE scope_id = ?
                ORDER BY turn_index, slot_key, memory_id
                """,
                (normalized_scope,),
            ).fetchall()
            for row in record_rows:
                record = SessionMemoryRecordV2(
                    memory_id=str(row["memory_id"]),
                    category=str(row["category"]),
                    slot_key=str(row["slot_key"]),
                    value=str(row["value"]),
                    relation=str(row["relation"]),
                    anchor_concepts=[str(item) for item in json.loads(row["anchor_concepts_json"]) or []],
                    evidence_anchors=[str(item) for item in json.loads(row["evidence_anchors_json"]) or []],
                    salience=float(row["salience"] or 0.0),
                    confidence=float(row["confidence"] or 0.0),
                    source_kind=str(row["source_kind"]),
                    turn_index=int(row["turn_index"] or 0),
                    state=str(row["state"]),
                    supersedes=[str(item) for item in json.loads(row["supersedes_json"]) or []],
                    metadata=dict(json.loads(row["metadata_json"]) or {}),
                )
                graph.records_by_id[record.memory_id] = record
            head_rows = connection.execute(
                "SELECT slot_key, memory_id FROM slot_heads WHERE scope_id = ? ORDER BY slot_key",
                (normalized_scope,),
            ).fetchall()
            graph.slot_heads = {str(row["slot_key"]): str(row["memory_id"]) for row in head_rows}
            history_rows = connection.execute(
                "SELECT slot_key, ordinal, memory_id FROM slot_history WHERE scope_id = ? ORDER BY slot_key, ordinal",
                (normalized_scope,),
            ).fetchall()
            graph.slot_history = defaultdict(list)
            for row in history_rows:
                graph.slot_history[str(row["slot_key"])].append(str(row["memory_id"]))
            edge_rows = connection.execute(
                """
                SELECT edge_id, source_memory_id, target_memory_id, edge_type,
                       score, model_score, evidence_turn, evidence, metadata_json
                FROM memory_edges
                WHERE scope_id = ?
                ORDER BY evidence_turn, edge_id
                """,
                (normalized_scope,),
            ).fetchall()
            graph.memory_edges = {}
            for row in edge_rows:
                edge = SessionMemoryEdgeV2(
                    edge_id=str(row["edge_id"]),
                    source_memory_id=str(row["source_memory_id"]),
                    target_memory_id=str(row["target_memory_id"]),
                    edge_type=str(row["edge_type"]),
                    score=float(row["score"] or 0.0),
                    model_score=float(row["model_score"] or 0.0),
                    evidence_turn=int(row["evidence_turn"] or 0),
                    evidence=str(row["evidence"] or ""),
                    metadata=dict(json.loads(row["metadata_json"]) or {}),
                )
                graph.memory_edges[edge.edge_id] = edge
            head_depth_rows = connection.execute(
                """
                SELECT subject_signature, depth_layer, memory_id
                FROM subject_depth_heads
                WHERE scope_id = ?
                ORDER BY subject_signature, depth_layer
                """,
                (normalized_scope,),
            ).fetchall()
            graph.subject_depth_heads = defaultdict(dict)
            for row in head_depth_rows:
                graph.subject_depth_heads[str(row["subject_signature"])][str(row["depth_layer"])] = str(row["memory_id"])
            if not graph.subject_depth_heads:
                for record in sorted(graph.records_by_id.values(), key=lambda item: int(item.turn_index)):
                    graph._refresh_subject_depth_head(record)
            graph.turn_log = self._load_audit_table(connection, "audit_turn_log", normalized_scope)
            graph.retrieval_log = self._load_audit_table(connection, "audit_retrieval_log", normalized_scope)
            graph.answer_support_log = self._load_audit_table(connection, "audit_answer_support", normalized_scope)
        if graph.records_by_id and graph.turn_index <= 0:
            graph.turn_index = max(record.turn_index for record in graph.records_by_id.values())
        for field_name, log_rows in (
            ("turn_log", graph.turn_log),
            ("retrieval_log", graph.retrieval_log),
            ("answer_support_log", graph.answer_support_log),
        ):
            graph.audit_event_totals[field_name] = max(int(graph.audit_event_totals.get(field_name, 0) or 0), len(log_rows))
        return graph

    def _load_audit_table(self, connection: sqlite3.Connection, table: str, scope_id: str) -> List[Dict[str, Any]]:
        rows = connection.execute(
            f"SELECT payload_json FROM {table} WHERE scope_id = ? ORDER BY event_index",
            (scope_id,),
        ).fetchall()
        return [dict(json.loads(row["payload_json"]) or {}) for row in rows]
