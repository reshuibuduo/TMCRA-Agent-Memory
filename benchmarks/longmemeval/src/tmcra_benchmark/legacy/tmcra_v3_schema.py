#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "tmcra.memory_recall.v3.0"
CHANNEL_NAMES = (
    "dense_score",
    "dense_rank_rr",
    "graph_rank_rr",
    "graph_selected",
    "graph_final",
    "recency_norm",
)
FORBIDDEN_CHANNEL_TOKENS = ("label", "positive", "negative", "hard", "gold", "answer")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid jsonl at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"jsonl row is not an object at {path}:{line_no}")
            rows.append(value)
    if not rows:
        raise RuntimeError(f"jsonl is empty: {path}")
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def channel_vector(candidate: Mapping[str, Any]) -> list[float]:
    channels = candidate.get("channels")
    if not isinstance(channels, Mapping):
        raise RuntimeError("candidate.channels must be an object")
    return [float(channels[name]) for name in CHANNEL_NAMES]


def _require_finite(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{context} is not finite: {number!r}")
    return number


def validate_candidate(candidate: Mapping[str, Any], *, context: str) -> None:
    candidate_id = clean_text(candidate.get("candidate_id"))
    text = clean_text(candidate.get("text"))
    if not candidate_id:
        raise RuntimeError(f"{context}: candidate_id is required")
    if not text:
        raise RuntimeError(f"{context}: text is required")
    channels = candidate.get("channels")
    if not isinstance(channels, Mapping):
        raise RuntimeError(f"{context}: channels must be an object")
    if len(channels) != len(CHANNEL_NAMES) or set(channels) != set(CHANNEL_NAMES):
        raise RuntimeError(
            f"{context}: channel keys must exactly equal {CHANNEL_NAMES}, got {tuple(channels.keys())}"
        )
    for name, value in channels.items():
        lowered = name.lower()
        if any(token in lowered for token in FORBIDDEN_CHANNEL_TOKENS):
            raise RuntimeError(f"{context}: label-derived channel name is forbidden: {name}")
        _require_finite(value, f"{context}.channels.{name}")
    labels = candidate.get("labels")
    if not isinstance(labels, Mapping):
        raise RuntimeError(f"{context}: labels must be an object")
    relevance = bool(labels.get("relevance", False))
    hard_negative = bool(labels.get("hard_negative", False))
    if relevance and hard_negative:
        raise RuntimeError(f"{context}: a positive candidate cannot be a hard negative")
    role = clean_text(labels.get("evidence_role"))
    expected_role = "positive" if relevance else ("hard_negative" if hard_negative else "negative")
    if role != expected_role:
        raise RuntimeError(f"{context}: evidence_role={role!r}, expected {expected_role!r}")
    target_scope = clean_text(labels.get("target_scope"))
    if target_scope and target_scope != "answer_session_bag":
        raise RuntimeError(f"{context}: unsupported target_scope={target_scope!r}")


def validate_sample(
    sample: Mapping[str, Any],
    *,
    require_positive: bool,
    allowed_splits: Sequence[str] = ("train", "holdout", "full_eval"),
) -> None:
    if clean_text(sample.get("schema_version")) != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported schema_version: {sample.get('schema_version')!r}")
    qid = clean_text(sample.get("question_id"))
    query = clean_text(sample.get("query_text"))
    split = clean_text(sample.get("split"))
    if not qid or not query:
        raise RuntimeError("question_id and query_text are required")
    if split not in set(allowed_splits):
        raise RuntimeError(f"invalid split for {qid}: {split!r}")
    supervision = sample.get("supervision")
    if supervision is not None:
        if not isinstance(supervision, Mapping):
            raise RuntimeError(f"{qid}: supervision must be an object")
        if clean_text(supervision.get("target_type")) not in {
            "multi_instance_answer_session_bag",
            "teacher_aligned_turn_bag",
        }:
            raise RuntimeError(f"{qid}: unsupported supervision target_type")
        if clean_text(supervision.get("loss")) != "negative_log_positive_probability_mass":
            raise RuntimeError(f"{qid}: unsupported supervision loss")
        weight = _require_finite(supervision.get("training_weight", 1.0), f"{qid}.supervision.training_weight")
        if weight <= 0.0 or weight > 1.0:
            raise RuntimeError(f"{qid}: supervision training_weight must be in (0, 1]")
    candidates = sample.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"{qid}: candidates must be a non-empty list")
    seen: set[str] = set()
    positive_count = 0
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise RuntimeError(f"{qid}: candidate {index} is not an object")
        validate_candidate(candidate, context=f"{qid}.candidates[{index}]")
        candidate_id = clean_text(candidate.get("candidate_id"))
        if candidate_id in seen:
            raise RuntimeError(f"{qid}: duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        positive_count += int(bool(candidate["labels"].get("relevance", False)))
    if require_positive and positive_count == 0:
        raise RuntimeError(f"{qid}: no positive candidate")


def validate_split_isolation(train_rows: Sequence[Mapping[str, Any]], holdout_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    train_qids = {clean_text(row.get("question_id")) for row in train_rows}
    holdout_qids = {clean_text(row.get("question_id")) for row in holdout_rows}
    train_queries = {clean_text(row.get("query_text")) for row in train_rows}
    holdout_queries = {clean_text(row.get("query_text")) for row in holdout_rows}
    qid_overlap = train_qids & holdout_qids
    query_overlap = train_queries & holdout_queries
    if qid_overlap:
        raise RuntimeError(f"train/holdout qid overlap: {sorted(qid_overlap)[:10]}")
    if query_overlap:
        raise RuntimeError(f"train/holdout query overlap: {sorted(query_overlap)[:3]}")
    return {
        "train_qids": len(train_qids),
        "holdout_qids": len(holdout_qids),
        "qid_overlap": 0,
        "query_text_overlap": 0,
    }
