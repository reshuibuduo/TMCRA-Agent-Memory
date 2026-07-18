#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from tmcra_v2_lme_pipeline import EmbeddingCacheVectorizer, session_text_chunks
from tmcra_v3_schema import (
    CHANNEL_NAMES,
    SCHEMA_VERSION,
    clean_text,
    read_jsonl,
    validate_sample,
    validate_split_isolation,
    write_jsonl,
)


EVENT_INDEX_RE = re.compile(
    r"^event::longmemeval:(?P<qid>[^:]+):s(?P<session>\d+)_c(?P<parent>\d+)$"
)
TEACHER_EVENT_INDEX_RE = re.compile(r":s(?P<session>\d+)_c(?P<parent>\d+)$")
RAW_TEACHER_EVENT_RE = re.compile(
    r"^lme:[^:]+:session:(?P<session>[^:]+):turn:(?P<turn>\d+)$"
)
TURN_BOUNDARY_RE = re.compile(r"\n\[[^\]]+ turn=\d+ role=")


def stable_int(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def normalized_question(value: Any) -> str:
    return clean_text(value).casefold()


def load_teacher_supervision(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        question_key = normalized_question(row.get("question"))
        if not question_key:
            raise RuntimeError("teacher row is missing question text")
        if question_key in output:
            raise RuntimeError(f"duplicate teacher question after exact normalization: {row.get('question')!r}")
        if not bool(row.get("usable_for_training", False)):
            continue
        if not list(row.get("positive_event_ids") or []):
            raise RuntimeError(f"teacher row has no positive events: {row.get('question_id')}")
        output[question_key] = row
    if not output:
        raise RuntimeError(f"no usable teacher supervision rows: {path}")
    return output


def teacher_parent_locations(event_ids: Iterable[Any], valid_locations: set[tuple[int, int]], qid: str) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw in event_ids:
        event_id = clean_text(raw)
        match = TEACHER_EVENT_INDEX_RE.search(event_id)
        if match is None:
            raise RuntimeError(f"{qid}: teacher event is not a runtime parent event: {event_id}")
        location = (int(match.group("session")), int(match.group("parent")))
        if location not in valid_locations:
            raise RuntimeError(f"{qid}: teacher event has no parent in the current inventory: {event_id}")
        if location not in seen:
            seen.add(location)
            output.append(location)
    return output


def reconstruct_parent_text(candidates: Sequence[Mapping[str, Any]], indexes: Sequence[int], qid: str) -> str:
    if not indexes:
        raise RuntimeError(f"{qid}: cannot reconstruct an empty parent")
    total = max(int(candidates[index]["source_char_end"]) for index in indexes)
    characters: list[str | None] = [None] * total
    for index in indexes:
        candidate = candidates[index]
        text = str(candidate["text"])
        if "\n" not in text:
            raise RuntimeError(f"{qid}: inventory candidate is missing its payload separator")
        payload = text.split("\n", 1)[1]
        start = int(candidate["source_char_start"])
        end = int(candidate["source_char_end"])
        if len(payload) != end - start:
            raise RuntimeError(f"{qid}: inventory candidate payload length does not match its source span")
        for offset, character in enumerate(payload, start=start):
            existing = characters[offset]
            if existing is not None and existing != character:
                raise RuntimeError(f"{qid}: overlapping inventory subchunks disagree")
            characters[offset] = character
    if any(character is None for character in characters):
        raise RuntimeError(f"{qid}: reconstructed parent contains a character coverage gap")
    return "".join(character for character in characters if character is not None)


def teacher_turn_candidate_indexes(
    event_ids: Iterable[Any],
    *,
    candidates: Sequence[Mapping[str, Any]],
    parent_candidates: Mapping[tuple[int, int], Sequence[int]],
    qid: str,
) -> set[int]:
    output: set[int] = set()
    session_locations: dict[str, list[tuple[int, int]]] = {}
    for location, indexes in parent_candidates.items():
        session_id = clean_text(candidates[indexes[0]].get("session_id"))
        session_locations.setdefault(session_id, []).append(location)
    parent_text_cache: dict[tuple[int, int], str] = {}
    for raw in event_ids:
        event_id = clean_text(raw)
        match = RAW_TEACHER_EVENT_RE.match(event_id)
        if match is None:
            raise RuntimeError(f"{qid}: teacher raw event is malformed: {event_id}")
        session_id = clean_text(match.group("session"))
        turn = int(match.group("turn"))
        marker = f"[{session_id} turn={turn} role="
        matches: list[tuple[tuple[int, int], int, int]] = []
        for location in session_locations.get(session_id, []):
            if location not in parent_text_cache:
                parent_text_cache[location] = reconstruct_parent_text(
                    candidates, parent_candidates[location], qid
                )
            parent_text = parent_text_cache[location]
            turn_start = parent_text.find(marker)
            if turn_start < 0:
                continue
            next_turn = TURN_BOUNDARY_RE.search(parent_text, turn_start + len(marker))
            turn_end = next_turn.start() if next_turn else len(parent_text)
            matches.append((location, turn_start, turn_end))
        if len(matches) != 1:
            raise RuntimeError(
                f"{qid}: teacher event must map to exactly one current parent turn, got {len(matches)}: {event_id}"
            )
        location, turn_start, turn_end = matches[0]
        matched_subchunks = {
            index
            for index in parent_candidates[location]
            if int(candidates[index]["source_char_start"]) < turn_end
            and int(candidates[index]["source_char_end"]) > turn_start
        }
        if not matched_subchunks:
            raise RuntimeError(f"{qid}: teacher turn has no overlapping subchunk: {event_id}")
        output.update(matched_subchunks)
    return output


def covered_windows(text: str, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    if max_chars <= 0:
        raise RuntimeError("subchunk payload size must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise RuntimeError("subchunk overlap must be non-negative and smaller than the payload")
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + max_chars)
        end = hard_end
        if hard_end < len(text):
            floor = max(start + max_chars // 2, hard_end - 160)
            split_at = text.rfind(" ", floor, hard_end)
            if split_at > start:
                end = split_at
        if end <= start:
            end = hard_end
        spans.append((start, end))
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    if spans[0][0] != 0 or spans[-1][1] != len(text):
        raise AssertionError("subchunk coverage does not reach both text boundaries")
    if any(right_start > left_end for (_, left_end), (right_start, _) in zip(spans, spans[1:])):
        raise AssertionError("subchunk coverage contains a gap")
    return spans


def load_lme(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"LongMemEval data must be a non-empty list: {path}")
    rows = [row for row in value if isinstance(row, dict)]
    qids = [clean_text(row.get("question_id")) for row in rows]
    if len(qids) != len(set(qids)):
        raise RuntimeError("LongMemEval question_id values are not unique")
    return rows


def choose_holdout(rows: Sequence[Mapping[str, Any]], count: int) -> list[str]:
    if count <= 0 or count >= len(rows):
        raise RuntimeError(f"holdout count must be between 1 and {len(rows) - 1}")
    by_type: dict[str, list[str]] = {}
    for row in rows:
        qid = clean_text(row.get("question_id"))
        qtype = clean_text(row.get("question_type")) or "unknown"
        by_type.setdefault(qtype, []).append(qid)
    selected: list[str] = []
    for qtype, qids in sorted(by_type.items()):
        qids.sort(key=lambda qid: (stable_int(f"tmcra-v3-holdout:{qid}"), qid))
        quota = int(len(qids) * count / len(rows))
        selected.extend(qids[:quota])
    selected_set = set(selected)
    remaining = [clean_text(row.get("question_id")) for row in rows if clean_text(row.get("question_id")) not in selected_set]
    remaining.sort(key=lambda qid: (stable_int(f"tmcra-v3-holdout-fill:{qid}"), qid))
    selected.extend(remaining[: max(0, count - len(selected))])
    selected = sorted(selected[:count])
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("failed to construct an exact unique holdout")
    return selected


def make_inventory_row(
    row: Mapping[str, Any],
    split: str,
    parent_chars: int,
    subchunk_chars: int,
    subchunk_overlap: int,
) -> dict[str, Any]:
    qid = clean_text(row.get("question_id"))
    question = clean_text(row.get("question"))
    question_date = clean_text(row.get("question_date"))
    query_text = f"{question}\nQuestion date: {question_date}" if question_date else question
    sessions = list(row.get("haystack_sessions") or [])
    session_ids = [clean_text(value) for value in list(row.get("haystack_session_ids") or [])]
    dates = [clean_text(value) for value in list(row.get("haystack_dates") or [])]
    answer_session_ids = {clean_text(value) for value in list(row.get("answer_session_ids") or []) if clean_text(value)}
    if not sessions or len(sessions) != len(session_ids):
        raise RuntimeError(f"{qid}: haystack session/id mismatch")
    candidates: list[dict[str, Any]] = []
    found_gold: set[str] = set()
    parent_chunk_count = 0
    source_parent_chars = 0
    for index, session in enumerate(sessions):
        session_id = session_ids[index]
        date = dates[index] if index < len(dates) else ""
        if not isinstance(session, Sequence) or isinstance(session, (str, bytes, bytearray)):
            raise RuntimeError(f"{qid}: session {index} is not a turn list")
        turns = list(session)
        if any(not isinstance(turn, Mapping) for turn in turns):
            raise RuntimeError(f"{qid}: session {index} contains a non-object turn")
        parent_chunks = session_text_chunks(
            session_id=session_id,
            date=date,
            turns=turns,
            max_chars=parent_chars,
            max_chunks=0,
        )
        if not parent_chunks:
            parent_chunks = [f"LongMemEval session_id={session_id} date={date}"]
        relevant = session_id in answer_session_ids
        if relevant:
            found_gold.add(session_id)
        for parent_index, parent_text in enumerate(parent_chunks, start=1):
            parent_chunk_count += 1
            source_parent_chars += len(parent_text)
            prefix = (
                f"LongMemEval session_id={session_id} date={date} "
                f"parent_chunk={parent_index:02d}"
            )
            payload_chars = subchunk_chars - len(prefix) - 1
            spans = covered_windows(parent_text, payload_chars, subchunk_overlap)
            for subchunk_index, (char_start, char_end) in enumerate(spans, start=1):
                text = f"{prefix}\n{parent_text[char_start:char_end]}"
                if len(text) > subchunk_chars:
                    raise AssertionError(f"{qid}: generated subchunk exceeds configured character cap")
                candidates.append(
                    {
                        "candidate_id": (
                            f"chunk::longmemeval:{qid}:s{index:03d}_c{parent_index:02d}_p{subchunk_index:02d}"
                        ),
                        "text": text,
                        "session_id": session_id,
                        "session_index": index,
                        "parent_chunk_index": parent_index,
                        "subchunk_index": subchunk_index,
                        "source_char_start": char_start,
                        "source_char_end": char_end,
                        "labels": {
                            "relevance": relevant,
                            "hard_negative": False,
                            "evidence_role": "positive" if relevant else "negative",
                            "target_scope": "answer_session_bag",
                        },
                    }
                )
    if found_gold != answer_session_ids:
        raise RuntimeError(f"{qid}: answer sessions missing from haystack: {sorted(answer_session_ids - found_gold)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "question_id": qid,
        "question": question,
        "question_date": question_date,
        "query_text": query_text,
        "question_type": clean_text(row.get("question_type")),
        "split": split,
        "gold_answer": clean_text(row.get("answer")),
        "answer_session_ids": sorted(answer_session_ids),
        "supervision": {
            "target_type": "multi_instance_answer_session_bag",
            "loss": "negative_log_positive_probability_mass",
        },
        "inventory_metadata": {
            "session_count": len(sessions),
            "parent_chunk_count": parent_chunk_count,
            "candidate_subchunk_count": len(candidates),
            "source_parent_chars": source_parent_chars,
            "parent_chars": parent_chars,
            "subchunk_chars": subchunk_chars,
            "subchunk_overlap": subchunk_overlap,
            "truncated": False,
        },
        "candidates": candidates,
    }


def command_inventory(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rows = load_lme(Path(args.data))
    holdout = set(choose_holdout(rows, args.holdout_count))
    inventory_rows = [
        make_inventory_row(
            row,
            "holdout" if clean_text(row.get("question_id")) in holdout else "train",
            args.parent_chars,
            args.subchunk_chars,
            args.subchunk_overlap,
        )
        for row in rows
    ]
    holdout_rows = [row for row in inventory_rows if row["split"] == "holdout"]
    train_rows = [row for row in inventory_rows if row["split"] == "train"]
    isolation = validate_split_isolation(train_rows, holdout_rows)
    write_jsonl(out_dir / "inventory.jsonl", inventory_rows)
    (out_dir / "holdout_qids.txt").write_text("\n".join(sorted(holdout)) + "\n", encoding="utf-8")
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data": str(Path(args.data).resolve()),
        "row_count": len(inventory_rows),
        "train_count": len(train_rows),
        "holdout_count": len(holdout_rows),
        "candidate_count": sum(len(row["candidates"]) for row in inventory_rows),
        "candidate_count_avg": round(sum(len(row["candidates"]) for row in inventory_rows) / len(inventory_rows), 4),
        "parent_chunk_count": sum(row["inventory_metadata"]["parent_chunk_count"] for row in inventory_rows),
        "session_count": sum(row["inventory_metadata"]["session_count"] for row in inventory_rows),
        "positive_bag_candidate_count": sum(
            int(candidate["labels"]["relevance"])
            for row in inventory_rows
            for candidate in row["candidates"]
        ),
        "max_candidate_chars": max(len(candidate["text"]) for row in inventory_rows for candidate in row["candidates"]),
        "truncated_candidate_count": 0,
        "parent_chars": args.parent_chars,
        "subchunk_chars": args.subchunk_chars,
        "subchunk_overlap": args.subchunk_overlap,
        "question_types": dict(Counter(row["question_type"] for row in inventory_rows)),
        "holdout_question_types": dict(Counter(row["question_type"] for row in holdout_rows)),
        "isolation": isolation,
    }
    (out_dir / "inventory_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def load_graph_debug(run_dir: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    paths = sorted(run_dir.glob("shard_*_out/samples_debug.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no graph debug shards under {run_dir}")
    for path in paths:
        for row in read_jsonl(path):
            qid = clean_text(row.get("question_id"))
            if qid:
                output[qid] = row
    return output


def ordered_parent_locations(
    event_ids: Iterable[Any],
    *,
    qid: str,
    valid_locations: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw in event_ids:
        event_id = clean_text(raw)
        match = EVENT_INDEX_RE.match(event_id)
        if match is None:
            continue
        if clean_text(match.group("qid")) != qid:
            raise RuntimeError(f"{qid}: graph debug contains an event for another question: {event_id}")
        location = (int(match.group("session")), int(match.group("parent")))
        if location not in valid_locations:
            raise RuntimeError(f"{qid}: graph event has no exact parent chunk mapping: {event_id}")
        if location in seen:
            continue
        seen.add(location)
        output.append(location)
    return output


def expand_parent_locations(
    locations: Iterable[tuple[int, int]],
    parent_candidates: Mapping[tuple[int, int], Sequence[int]],
) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for location in locations:
        for index in parent_candidates[location]:
            if index not in seen:
                seen.add(index)
                output.append(index)
    return output


def rrank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / float(rank + 1)


def canonical_candidate(
    raw: Mapping[str, Any],
    *,
    relevant: bool,
    session_relevant: bool,
    teacher_relevant: bool | None,
    dense_score: float,
    dense_rank: int,
    graph_rank: int | None,
    graph_selected: bool,
    graph_final: bool,
    session_count: int,
    forced_positive: bool,
    hard_negative: bool,
) -> dict[str, Any]:
    relevant = bool(relevant)
    hard_negative = bool(hard_negative and not relevant)
    channels = {
        "dense_score": float(dense_score),
        "dense_rank_rr": rrank(dense_rank),
        "graph_rank_rr": rrank(graph_rank),
        "graph_selected": float(bool(graph_selected)),
        "graph_final": float(bool(graph_final)),
        "recency_norm": float(raw["session_index"]) / float(max(1, session_count - 1)),
    }
    if tuple(channels.keys()) != CHANNEL_NAMES:
        raise AssertionError("channel construction order changed")
    return {
        "candidate_id": raw["candidate_id"],
        "text": raw["text"],
        "session_id": raw["session_id"],
        "session_index": int(raw["session_index"]),
        "parent_chunk_index": int(raw["parent_chunk_index"]),
        "subchunk_index": int(raw["subchunk_index"]),
        "source_char_start": int(raw["source_char_start"]),
        "source_char_end": int(raw["source_char_end"]),
        "channels": channels,
        "labels": {
            "relevance": relevant,
            "session_relevance": bool(session_relevant),
            "teacher_evidence_relevance": teacher_relevant,
            "hard_negative": hard_negative,
            "evidence_role": "positive" if relevant else ("hard_negative" if hard_negative else "negative"),
            "target_scope": "answer_session_bag",
        },
        "provenance": {
            "dense_retrieved": dense_rank >= 0,
            "graph_retrieved": graph_rank is not None,
            "graph_selected": bool(graph_selected),
            "graph_final": bool(graph_final),
            "forced_positive_for_training": bool(forced_positive),
        },
    }


def metric_hit(order: Sequence[int], positives: set[int], k: int) -> int:
    return int(any(index in positives for index in order[:k]))


def command_finalize(args: argparse.Namespace) -> None:
    inventory_rows = read_jsonl(Path(args.inventory))
    graph_debug = load_graph_debug(Path(args.graph_debug_run))
    teacher_by_question = load_teacher_supervision(Path(args.teacher_labels))
    vectorizer = EmbeddingCacheVectorizer(
        cache_dir=args.embedding_cache,
        dim=args.text_dim,
        expected_backend="hf",
        expected_model=args.embedding_model,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    full_inventory_rows: list[dict[str, Any]] = []
    provider = Counter()
    teacher_provider = Counter()
    supervision_counts = Counter()
    for source in inventory_rows:
        qid = clean_text(source["question_id"])
        candidates = list(source["candidates"])
        session_count = int(source["inventory_metadata"]["session_count"])
        candidate_count = len(candidates)
        parent_candidates: dict[tuple[int, int], list[int]] = {}
        for index, candidate in enumerate(candidates):
            location = (int(candidate["session_index"]), int(candidate["parent_chunk_index"]))
            parent_candidates.setdefault(location, []).append(index)
        valid_parent_locations = set(parent_candidates)
        qvec = vectorizer.encode_one(source["query_text"])
        evec = torch.stack([vectorizer.encode_one(candidate["text"]) for candidate in candidates])
        dense_scores = (evec @ qvec).tolist()
        dense_order = sorted(range(candidate_count), key=lambda index: (-dense_scores[index], index))
        dense_rank = {index: rank for rank, index in enumerate(dense_order)}
        debug = graph_debug.get(qid, {})
        retrieval = dict(debug.get("retrieval") or {})
        selected_parents = ordered_parent_locations(
            retrieval.get("selected_event_ids") or [], qid=qid, valid_locations=valid_parent_locations
        )
        final_parents = ordered_parent_locations(
            retrieval.get("final_hit_event_ids") or [], qid=qid, valid_locations=valid_parent_locations
        )
        recall_parents = ordered_parent_locations(
            retrieval.get("recall_event_ids") or [], qid=qid, valid_locations=valid_parent_locations
        )
        graph_parents: list[tuple[int, int]] = []
        graph_parent_seen: set[tuple[int, int]] = set()
        for location in [*selected_parents, *recall_parents]:
            if location not in graph_parent_seen:
                graph_parent_seen.add(location)
                graph_parents.append(location)
        graph_parent_rank = {location: rank for rank, location in enumerate(graph_parents)}
        graph_order = expand_parent_locations(graph_parents, parent_candidates)
        graph_rank = {
            index: graph_parent_rank[(int(candidates[index]["session_index"]), int(candidates[index]["parent_chunk_index"]))]
            for index in graph_order
        }
        selected_indexes = set(expand_parent_locations(selected_parents, parent_candidates))
        final_indexes = set(expand_parent_locations(final_parents, parent_candidates))
        session_positives = {index for index, candidate in enumerate(candidates) if candidate["labels"]["relevance"]}
        teacher_row = teacher_by_question.get(normalized_question(source["question"]))
        teacher_positives: set[int] | None = None
        teacher_hard: set[int] = set()
        if teacher_row is not None:
            teacher_metadata = dict(teacher_row.get("metadata") or {})
            raw_teacher_positive = list(teacher_metadata.get("raw_teacher_positive_event_ids") or [])
            raw_teacher_hard = list(teacher_metadata.get("raw_teacher_hard_negative_event_ids") or [])
            if not raw_teacher_positive:
                raise RuntimeError(f"{qid}: aligned teacher row is missing raw positive turn ids")
            teacher_positives = teacher_turn_candidate_indexes(
                raw_teacher_positive,
                candidates=candidates,
                parent_candidates=parent_candidates,
                qid=qid,
            )
            teacher_hard = teacher_turn_candidate_indexes(
                raw_teacher_hard,
                candidates=candidates,
                parent_candidates=parent_candidates,
                qid=qid,
            )
            if not teacher_positives:
                raise RuntimeError(f"{qid}: teacher supervision produced no positive subchunks")
            mixed_teacher_subchunks = teacher_positives & teacher_hard
            if mixed_teacher_subchunks:
                supervision_counts["teacher_mixed_positive_hard_subchunks"] += len(mixed_teacher_subchunks)
                teacher_hard -= teacher_positives
            positives = teacher_positives
            supervision = {
                "target_type": "teacher_aligned_turn_bag",
                "loss": "negative_log_positive_probability_mass",
                "training_weight": float(teacher_row.get("training_weight_hint", 1.0) or 1.0),
                "teacher_label_model": clean_text((teacher_row.get("metadata") or {}).get("teacher_label_model")),
                "teacher_verify_model": clean_text((teacher_row.get("metadata") or {}).get("teacher_verify_model")),
                "teacher_confidence": clean_text(teacher_row.get("label_confidence")),
            }
        else:
            positives = session_positives
            supervision = {
                "target_type": "multi_instance_answer_session_bag",
                "loss": "negative_log_positive_probability_mass",
                "training_weight": float(args.weak_supervision_weight),
            }
        supervision_counts[supervision["target_type"]] += 1
        provider["n"] += 1
        for k in (1, 5, 10, 20, 32):
            provider[f"primary_dense_r@{k}"] += metric_hit(dense_order, positives, k)
            provider[f"primary_graph_r@{k}"] += metric_hit(graph_order, positives, k)
            provider[f"session_dense_r@{k}"] += metric_hit(dense_order, session_positives, k)
            provider[f"session_graph_r@{k}"] += metric_hit(graph_order, session_positives, k)
            if teacher_positives is not None:
                teacher_provider[f"dense_r@{k}"] += metric_hit(dense_order, teacher_positives, k)
                teacher_provider[f"graph_r@{k}"] += metric_hit(graph_order, teacher_positives, k)
        if teacher_positives is not None:
            teacher_provider["n"] += 1
        runtime_indexes: list[int] = []
        runtime_seen: set[int] = set()
        graph_runtime_order = expand_parent_locations(graph_parents[: args.graph_k], parent_candidates)
        for index in [*dense_order[: args.dense_k], *graph_runtime_order]:
            if index not in runtime_seen:
                runtime_seen.add(index)
                runtime_indexes.append(index)
        provider["union_gold_hit"] += int(bool(positives & runtime_seen))
        provider["union_session_hit"] += int(bool(session_positives & runtime_seen))
        if teacher_positives is not None:
            teacher_provider["union_hit"] += int(bool(teacher_positives & runtime_seen))
        pre_force_hit = bool(positives & runtime_seen)
        training_indexes = list(runtime_indexes)
        forced_indexes: set[int] = set()
        if source["split"] == "train" and not pre_force_hit:
            best_positive = next(index for index in dense_order if index in positives)
            training_indexes.append(best_positive)
            runtime_seen.add(best_positive)
            forced_indexes.add(best_positive)
        def build_row(indexes: Sequence[int], split: str, allow_forced: bool) -> dict[str, Any]:
            output_candidates = []
            for index in indexes:
                hard = index not in positives and (
                    index in teacher_hard
                    or dense_rank[index] < args.hard_negative_k
                    or graph_rank.get(index, 10**9) < args.hard_negative_k
                )
                output_candidates.append(
                    canonical_candidate(
                        candidates[index],
                        relevant=index in positives,
                        session_relevant=index in session_positives,
                        teacher_relevant=None if teacher_positives is None else index in teacher_positives,
                        dense_score=dense_scores[index],
                        dense_rank=dense_rank[index],
                        graph_rank=graph_rank.get(index),
                        graph_selected=index in selected_indexes,
                        graph_final=index in final_indexes,
                        session_count=session_count,
                        forced_positive=allow_forced and index in forced_indexes,
                        hard_negative=hard,
                    )
                )
            return {
                "schema_version": SCHEMA_VERSION,
                "question_id": qid,
                "question": source["question"],
                "question_date": source["question_date"],
                "query_text": source["query_text"],
                "question_type": source["question_type"],
                "split": split,
                "gold_answer": source["gold_answer"],
                "answer_session_ids": source["answer_session_ids"],
                "supervision": supervision,
                "candidates": output_candidates,
                "pool_metadata": {
                    "session_count": session_count,
                    "inventory_count": candidate_count,
                    "graph_parent_event_count": len(graph_parents),
                    "graph_runtime_candidate_count": len(graph_runtime_order),
                    "runtime_pool_count": len(runtime_indexes),
                    "pre_force_gold_hit": pre_force_hit,
                    "pre_force_session_hit": bool(session_positives & set(runtime_indexes)),
                    "pre_force_teacher_hit": None if teacher_positives is None else bool(teacher_positives & set(runtime_indexes)),
                    "forced_positive_count": len(forced_indexes) if allow_forced else 0,
                    "dense_k": args.dense_k,
                    "graph_k": args.graph_k,
                    "graph_debug_available": bool(debug),
                    "graph_retrieval_mode": retrieval.get("retrieval_mode"),
                },
            }
        runtime_row = build_row(runtime_indexes, "full_eval", False)
        validate_sample(runtime_row, require_positive=False)
        full_rows.append(runtime_row)
        all_row = build_row(list(range(candidate_count)), "full_eval", False)
        validate_sample(all_row, require_positive=True)
        full_inventory_rows.append(all_row)
        if source["split"] == "train":
            train_row = build_row(training_indexes, "train", True)
            validate_sample(train_row, require_positive=True)
            train_rows.append(train_row)
        else:
            holdout_row = build_row(runtime_indexes, "holdout", False)
            validate_sample(holdout_row, require_positive=False)
            holdout_rows.append(holdout_row)
    isolation = validate_split_isolation(train_rows, holdout_rows)
    if len(train_rows) + len(holdout_rows) != len(inventory_rows):
        raise RuntimeError("finalized row count mismatch")
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "holdout.jsonl", holdout_rows)
    write_jsonl(out_dir / "full_eval.jsonl", full_rows)
    write_jsonl(out_dir / "full_inventory_eval.jsonl", full_inventory_rows)
    if args.dev_count <= 0 or args.dev_count >= len(train_rows):
        raise RuntimeError("dev_count must leave non-empty optimization train and dev sets")
    ordered_train_qids = sorted(
        (row["question_id"] for row in train_rows),
        key=lambda qid: (stable_int(f"tmcra-v3-dev:{qid}"), qid),
    )
    dev_qids = set(ordered_train_qids[: args.dev_count])
    model_train_rows = [row for row in train_rows if row["question_id"] not in dev_qids]
    full_by_qid = {row["question_id"]: row for row in full_rows}
    model_dev_rows = [full_by_qid[qid] for qid in ordered_train_qids[: args.dev_count]]
    if any(row["pool_metadata"]["forced_positive_count"] for row in model_dev_rows):
        raise RuntimeError("runtime dev rows must never contain forced positives")
    write_jsonl(out_dir / "model_train.jsonl", model_train_rows)
    write_jsonl(out_dir / "model_dev_runtime.jsonl", model_dev_rows)
    (out_dir / "model_dev_qids.txt").write_text("\n".join(ordered_train_qids[: args.dev_count]) + "\n", encoding="utf-8")
    cross_cache_rows = [*train_rows, *holdout_rows]
    if len({row["question_id"] for row in cross_cache_rows}) != len(inventory_rows):
        raise RuntimeError("cross-cache input must contain exactly one row per question")
    write_jsonl(out_dir / "cross_cache_samples.jsonl", cross_cache_rows)
    n = max(1, int(provider["n"]))
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_count": len(train_rows),
        "holdout_count": len(holdout_rows),
        "full_eval_count": len(full_rows),
        "full_inventory_eval_count": len(full_inventory_rows),
        "cross_cache_sample_count": len(cross_cache_rows),
        "cross_cache_pair_count": sum(len(row["candidates"]) for row in cross_cache_rows),
        "model_train_count": len(model_train_rows),
        "model_dev_runtime_count": len(model_dev_rows),
        "model_dev_runtime_candidate_miss_count": sum(
            int(not any(candidate["labels"]["relevance"] for candidate in row["candidates"]))
            for row in model_dev_rows
        ),
        "isolation": isolation,
        "candidate_pool": {
            "dense_k": args.dense_k,
            "graph_k": args.graph_k,
            "hard_negative_k": args.hard_negative_k,
            "train_candidate_avg": round(sum(len(row["candidates"]) for row in train_rows) / max(1, len(train_rows)), 4),
            "holdout_candidate_avg": round(sum(len(row["candidates"]) for row in holdout_rows) / max(1, len(holdout_rows)), 4),
            "full_inventory_candidate_avg": round(sum(len(row["candidates"]) for row in full_inventory_rows) / max(1, len(full_inventory_rows)), 4),
            "train_forced_positive_rows": sum(int(row["pool_metadata"]["forced_positive_count"] > 0) for row in train_rows),
        },
        "provider_metrics": {
            key: round(value / n, 6) for key, value in sorted(provider.items()) if key != "n"
        },
        "teacher_provider_metrics": {
            key: round(value / max(1, int(teacher_provider["n"])), 6)
            for key, value in sorted(teacher_provider.items())
            if key != "n"
        },
        "teacher_supervision_count": int(teacher_provider["n"]),
        "supervision_counts": dict(supervision_counts),
        "teacher_labels": str(Path(args.teacher_labels).resolve()),
        "weak_supervision_weight": args.weak_supervision_weight,
        "embedding_cache": str(Path(args.embedding_cache).resolve()),
        "graph_debug_run": str(Path(args.graph_debug_run).resolve()),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--data", required=True)
    inventory.add_argument("--out-dir", required=True)
    inventory.add_argument("--holdout-count", type=int, default=100)
    inventory.add_argument("--parent-chars", type=int, default=7000)
    inventory.add_argument("--subchunk-chars", type=int, default=1800)
    inventory.add_argument("--subchunk-overlap", type=int, default=200)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--inventory", required=True)
    finalize.add_argument("--embedding-cache", required=True)
    finalize.add_argument("--embedding-model", required=True)
    finalize.add_argument("--graph-debug-run", required=True)
    finalize.add_argument("--teacher-labels", required=True)
    finalize.add_argument("--out-dir", required=True)
    finalize.add_argument("--text-dim", type=int, default=1024)
    finalize.add_argument("--dense-k", type=int, default=32)
    finalize.add_argument("--graph-k", type=int, default=24)
    finalize.add_argument("--hard-negative-k", type=int, default=12)
    finalize.add_argument("--dev-count", type=int, default=40)
    finalize.add_argument("--weak-supervision-weight", type=float, default=0.35)
    args = parser.parse_args()
    if args.command == "inventory":
        command_inventory(args)
    else:
        command_finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
