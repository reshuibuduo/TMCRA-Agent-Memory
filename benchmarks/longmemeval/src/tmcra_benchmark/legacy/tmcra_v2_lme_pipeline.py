#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
EVENT_RE = re.compile(r"^event::longmemeval:(?P<qid>[^:]+):s(?P<session>\d+)_c(?P<chunk>\d+)$")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def norm_tokens(text: Any) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(clean_text(text))]


def stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def embedding_text_key(text: Any) -> str:
    cleaned = clean_text(text)
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def iter_sample_texts(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in samples:
        qid = clean_text(sample.get("question_id"))
        question = clean_text(sample.get("question", ""))
        if question:
            key = embedding_text_key(question)
            if key not in seen:
                seen.add(key)
                items.append({"key": key, "kind": "question", "question_id": qid, "text": question})
        for cand in list(sample.get("candidates") or []):
            text = clean_text(cand.get("text", ""))
            if not text:
                continue
            key = embedding_text_key(text)
            if key not in seen:
                seen.add(key)
                items.append({
                    "key": key,
                    "kind": "candidate",
                    "question_id": qid,
                    "event_id": clean_text(cand.get("event_id", "")),
                    "text": text,
                })
    if not items:
        raise RuntimeError("no texts found for embedding precompute")
    return items


def split_name(qid: str, val_ratio: float = 0.2) -> str:
    bucket = stable_hash(qid) % 10000
    return "val" if bucket < int(val_ratio * 10000) else "train"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"required jsonl file not found: {path}")
    rows=[]
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid jsonl at {path}:{lineno}: {exc}") from exc
    if not rows:
        raise RuntimeError(f"required jsonl file is empty: {path}")
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_shard_rows(shard_glob: str) -> dict[str, list[dict[str, Any]]]:
    base = Path("/") if shard_glob.startswith("/") else Path()
    pattern = shard_glob[1:] if shard_glob.startswith("/") else shard_glob
    shard_paths = sorted(base.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"no shard files matched --shard-glob: {shard_glob}")
    rows_by_qid: dict[str, list[dict[str, Any]]] = {}
    for shard in shard_paths:
        data = json.loads(shard.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(data, list):
            raise RuntimeError(f"shard is not a JSON list: {shard}")
        for idx, row in enumerate(data):
            if not isinstance(row, dict):
                raise RuntimeError(f"shard row is not an object: {shard}[{idx}]")
            qid = clean_text(row.get("question_id"))
            if qid.endswith("_abs"):
                qid = qid[:-4]
            if not qid:
                raise RuntimeError(f"shard row missing question_id: {shard}[{idx}]")
            rows_by_qid.setdefault(qid, []).append(row)
    if not rows_by_qid:
        raise RuntimeError(f"no usable rows loaded from --shard-glob: {shard_glob}")
    return rows_by_qid


def session_text_chunks(*, session_id: str, date: str, turns: list[Mapping[str, Any]], max_chars: int = 7000, max_chunks: int = 0) -> list[str]:
    chunks: list[str] = []
    current: list[str] = [f"LongMemEval session_id={session_id} date={date}"]
    current_len = len(current[0])
    for index, turn in enumerate(turns, start=1):
        role = clean_text(turn.get("role", "unknown"))
        content = clean_text(turn.get("content", ""))
        if not content:
            continue
        part = f"[{session_id} turn={index} role={role}] {content}"
        if current_len + len(part) + 2 > max_chars and len(current) > 1:
            chunks.append("\n".join(current))
            if max_chunks > 0 and len(chunks) >= max_chunks:
                return chunks
            current = [f"LongMemEval session_id={session_id} date={date} continued=true"]
            current_len = len(current[0])
        current.append(part)
        current_len += len(part) + 1
    if len(current) > 1 and (max_chunks <= 0 or len(chunks) < max_chunks):
        chunks.append("\n".join(current))
    return chunks


def event_text_map_for_row(row: Mapping[str, Any], qid: str) -> dict[str, dict[str, Any]]:
    sessions = list(row.get("haystack_sessions") or [])
    session_ids = [clean_text(x) for x in list(row.get("haystack_session_ids") or [])]
    dates = [clean_text(x) for x in list(row.get("haystack_dates") or [])]
    out: dict[str, dict[str, Any]] = {}
    for sidx, sid in enumerate(session_ids):
        if sidx >= len(sessions):
            continue
        chunks = session_text_chunks(
            session_id=sid,
            date=dates[sidx] if sidx < len(dates) else "",
            turns=list(sessions[sidx] or []),
            max_chars=7000,
            max_chunks=0,
        )
        for cidx, text in enumerate(chunks, start=1):
            eid = f"event::longmemeval:{qid}:s{sidx:03d}_c{cidx:02d}"
            out[eid] = {
                "event_id": eid,
                "text": text,
                "session_id": sid,
                "session_index": sidx,
                "chunk_index": cidx,
                "date": dates[sidx] if sidx < len(dates) else "",
            }
    return out


def choose_row_for_query(rows: list[dict[str, Any]], candidate_ids: Sequence[str], positive_ids: Sequence[str]) -> dict[str, Any] | None:
    if not rows:
        return None
    wanted = set(candidate_ids or []) | set(positive_ids or [])
    if not wanted:
        return None
    qid = clean_text(rows[0].get("question_id"))
    if qid.endswith("_abs"):
        qid = qid[:-4]
    best = None
    best_hits = -1
    for row in rows:
        rqid = clean_text(row.get("question_id"))
        if rqid.endswith("_abs"):
            rqid = rqid[:-4]
        events = set(event_text_map_for_row(row, rqid))
        hits = len(wanted & events)
        if hits > best_hits:
            best = row
            best_hits = hits
    return best


def token_overlap(a: str, b: str) -> float:
    at, bt = set(norm_tokens(a)), set(norm_tokens(b))
    if not at or not bt:
        return 0.0
    return len(at & bt) / max(1, len(at | bt))


def contains_number_overlap(a: str, b: str) -> float:
    nums_a = {t for t in norm_tokens(a) if any(ch.isdigit() for ch in t)}
    nums_b = {t for t in norm_tokens(b) if any(ch.isdigit() for ch in t)}
    if not nums_a or not nums_b:
        return 0.0
    return len(nums_a & nums_b) / max(1, len(nums_a | nums_b))


def build_dataset(args: argparse.Namespace) -> None:
    label_path = Path(args.aligned_queries)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = read_jsonl(label_path)
    rows_by_qid = load_shard_rows(args.shard_glob)
    samples=[]; rejected=[]
    for row in labels:
        qid = clean_text(row.get("question_id"))
        if qid.endswith("_abs"):
            qid = qid[:-4]
        question = clean_text(row.get("question"))
        positives = list(row.get("positive_event_ids") or [])
        hard_negs = set(row.get("hard_negative_event_ids") or [])
        candidate_ids = list(row.get("candidate_event_ids") or [])
        source_rows = rows_by_qid.get(qid, [])
        source_row = choose_row_for_query(source_rows, candidate_ids, positives)
        if not source_row:
            rejected.append({"question_id": qid, "reason": "missing_source_row"})
            continue
        event_map = event_text_map_for_row(source_row, qid)
        if not candidate_ids:
            rejected.append({"question_id": qid, "reason": "missing_candidate_event_ids"})
            continue
        before_filter_count = len(candidate_ids)
        candidate_ids = [eid for eid in candidate_ids if eid in event_map]
        if len(candidate_ids) != before_filter_count:
            rejected.append({"question_id": qid, "reason": "candidate_ids_not_in_event_map", "candidate_count": before_filter_count, "mapped_count": len(candidate_ids)})
            continue
        if not candidate_ids:
            rejected.append({"question_id": qid, "reason": "no_candidate_events"})
            continue
        if not any(eid in event_map for eid in positives):
            rejected.append({"question_id": qid, "reason": "positive_not_in_event_map", "positive_event_ids": positives[:10]})
            continue
        q_tokens = set(norm_tokens(question))
        candidates=[]
        max_session = max(1, len(source_row.get("haystack_session_ids") or []))
        for eid in candidate_ids:
            info = event_map[eid]
            text = clean_text(info["text"])
            pos = eid in positives
            hard = eid in hard_negs
            candidates.append({
                "event_id": eid,
                "text": text,
                "session_id": info.get("session_id", ""),
                "session_index": int(info.get("session_index", 0)),
                "chunk_index": int(info.get("chunk_index", 0)),
                "features": [
                    token_overlap(question, text),
                    contains_number_overlap(question, text),
                    min(1.0, len(set(norm_tokens(text)) & q_tokens) / max(1, len(q_tokens))),
                    float(info.get("session_index", 0)) / float(max_session),
                    float(info.get("chunk_index", 0)) / 10.0,
                    1.0 if hard else 0.0,
                ],
                "label": 1 if pos else 0,
                "role_label": 1 if pos else (2 if hard else 0),
            })
        if not any(c["label"] for c in candidates):
            rejected.append({"question_id": qid, "reason": "positive_filtered_out", "positive_event_ids": positives[:10]})
            continue
        metadata = dict(row.get("metadata") or {})
        metadata.setdefault("question_type", clean_text(source_row.get("question_type", "")))
        metadata.setdefault("gold_answer", clean_text(source_row.get("answer", "")))
        metadata.setdefault("answer_session_ids", list(source_row.get("answer_session_ids") or []))
        samples.append({
            "question_id": qid,
            "question": question,
            "question_type": clean_text(source_row.get("question_type", "")),
            "split": split_name(qid, args.val_ratio),
            "positive_event_ids": positives,
            "positive_path_ids": list(row.get("positive_path_ids") or []),
            "metadata": metadata,
            "candidates": candidates,
        })
    write_jsonl(out_dir / "samples.jsonl", samples)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "aligned_queries": str(label_path),
        "sample_count": len(samples),
        "rejected_count": len(rejected),
        "train_count": sum(1 for s in samples if s["split"] == "train"),
        "val_count": sum(1 for s in samples if s["split"] == "val"),
        "candidate_count_avg": round(sum(len(s["candidates"]) for s in samples) / max(1, len(samples)), 3),
        "positive_count_avg": round(sum(sum(c["label"] for c in s["candidates"]) for s in samples) / max(1, len(samples)), 3),
        "outputs": {"samples": str(out_dir / "samples.jsonl"), "rejected": str(out_dir / "rejected.jsonl")},
    }
    (out_dir / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


class OpenAIEmbeddingVectorizer:
    def __init__(self, *, dim: int, base_url: str, model: str, api_key: str = ""):
        self.dim = int(dim)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.cache: dict[str, torch.Tensor] = {}

    def encode_one(self, text: str) -> torch.Tensor:
        text = clean_text(text)
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        payload = json.dumps({"model": self.model, "input": text}, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + "/embeddings", data=payload, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
        data = body.get("data") or []
        if not data or "embedding" not in data[0]:
            raise RuntimeError(f"embedding response missing data[0].embedding from {self.base_url}")
        vec = torch.tensor(list(data[0]["embedding"]), dtype=torch.float32)
        if vec.numel() != self.dim:
            raise RuntimeError(f"embedding dim mismatch: got {vec.numel()} expected {self.dim}")
        vec = vec / vec.norm().clamp_min(1e-6)
        self.cache[text] = vec
        return vec


class BgeM3DenseVectorizer:
    def __init__(
        self,
        *,
        dim: int,
        model_path: str,
        device: str = "cpu",
        max_length: int = 8192,
        strict_max_length: bool = False,
    ):
        from transformers import AutoModel, AutoTokenizer
        self.dim = int(dim)
        self.cache: dict[str, torch.Tensor] = {}
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.strict_max_length = bool(strict_max_length)
        if self.max_length <= 0:
            raise RuntimeError("--embedding-max-length must be positive")
        resolved = Path(model_path).resolve()
        if "bge-m3" not in resolved.name.lower() and "bge-m3" not in str(resolved).lower():
            raise RuntimeError(f"hf backend is currently wired only for local BAAI/bge-m3, got: {model_path}")
        self.model_path = str(resolved)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(resolved),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            str(resolved),
            local_files_only=True,
            trust_remote_code=False,
        ).to(self.device)
        self.model.eval()
        hidden_size = int(getattr(self.model.config, "hidden_size", 0) or 0)
        if hidden_size != self.dim:
            raise RuntimeError(f"bge-m3 hidden_size mismatch: model has {hidden_size}, --text-dim is {self.dim}")

    def encode_batch(self, texts: Sequence[str], batch_size: int = 4) -> torch.Tensor:
        cleaned = [clean_text(t) for t in texts]
        if not cleaned:
            raise RuntimeError("encode_batch got no texts")
        out_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(cleaned), max(1, int(batch_size))):
                batch_texts = cleaned[start:start + max(1, int(batch_size))]
                encoded = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=not self.strict_max_length,
                    max_length=self.max_length if not self.strict_max_length else None,
                )
                if self.strict_max_length:
                    token_lengths = encoded["attention_mask"].sum(dim=1)
                    longest = int(token_lengths.max())
                    if longest > self.max_length:
                        offending = int(torch.argmax(token_lengths))
                        raise RuntimeError(
                            f"BGE input has {longest} tokens, exceeding strict max_length={self.max_length}; "
                            f"batch_index={offending} chars={len(batch_texts[offending])}"
                        )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.device.type == "cuda",
                ):
                    model_out = self.model(**encoded)
                if not hasattr(model_out, "last_hidden_state"):
                    raise RuntimeError("bge-m3 model output missing last_hidden_state")
                vecs = model_out.last_hidden_state[:, 0].detach().cpu().float()
                if vecs.shape[1] != self.dim:
                    raise RuntimeError(f"bge-m3 embedding dim mismatch: got {vecs.shape[1]} expected {self.dim}")
                vecs = vecs / vecs.norm(dim=1, keepdim=True).clamp_min(1e-6)
                out_chunks.append(vecs)
        return torch.cat(out_chunks, dim=0)

    def encode_one(self, text: str) -> torch.Tensor:
        text = clean_text(text)
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        vec = self.encode_batch([text], batch_size=1)[0]
        self.cache[text] = vec
        return vec


class EmbeddingCacheVectorizer:
    def __init__(self, *, cache_dir: str, dim: int, expected_backend: str = "", expected_model: str = ""):
        root = Path(cache_dir)
        manifest_path = root / "manifest.json"
        tensor_path = root / "embeddings.pt"
        if not manifest_path.exists():
            raise FileNotFoundError(f"embedding cache manifest not found: {manifest_path}")
        if not tensor_path.exists():
            raise FileNotFoundError(f"embedding cache tensor not found: {tensor_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.dim = int(dim)
        if int(self.manifest.get("text_dim", 0) or 0) != self.dim:
            raise RuntimeError(f"embedding cache dim mismatch: cache has {self.manifest.get('text_dim')} expected {self.dim}")
        backend = clean_text(self.manifest.get("embedding_backend", ""))
        model = clean_text(self.manifest.get("embedding_model", ""))
        if expected_backend and backend != expected_backend:
            raise RuntimeError(f"embedding cache backend mismatch: cache has {backend} expected {expected_backend}")
        if expected_model and model != expected_model:
            raise RuntimeError(f"embedding cache model mismatch: cache has {model} expected {expected_model}")
        payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
        keys = list(payload.get("keys") or [])
        vectors = payload.get("vectors")
        if not keys or vectors is None:
            raise RuntimeError(f"invalid embedding cache payload: {tensor_path}")
        if len(keys) != int(vectors.shape[0]):
            raise RuntimeError(f"embedding cache row mismatch: {len(keys)} keys vs {vectors.shape[0]} vectors")
        if int(vectors.shape[1]) != self.dim:
            raise RuntimeError(f"embedding cache tensor dim mismatch: {vectors.shape[1]} expected {self.dim}")
        self.keys = keys
        self.index = {k: i for i, k in enumerate(keys)}
        if len(self.index) != len(keys):
            raise RuntimeError("embedding cache contains duplicate keys")
        self.vectors = vectors.float().contiguous()

    def encode_one(self, text: str) -> torch.Tensor:
        key = embedding_text_key(text)
        idx = self.index.get(key)
        if idx is None:
            preview = clean_text(text)[:120]
            raise RuntimeError(f"embedding cache miss: key={key} preview={preview!r}")
        return self.vectors[idx]


def build_vectorizer(args: argparse.Namespace, device: torch.device):
    backend = clean_text(getattr(args, "embedding_backend", ""))
    cache_dir = clean_text(getattr(args, "embedding_cache", ""))
    if not backend:
        raise RuntimeError("embedding backend is required; pass --embedding-backend openai or --embedding-backend hf")
    if cache_dir:
        return EmbeddingCacheVectorizer(
            cache_dir=cache_dir,
            dim=args.text_dim,
            expected_backend=backend,
            expected_model=clean_text(getattr(args, "embedding_model", "")),
        )
    if backend == "openai":
        base_url = clean_text(getattr(args, "embedding_base_url", ""))
        model = clean_text(getattr(args, "embedding_model", ""))
        api_key = clean_text(getattr(args, "embedding_api_key", ""))
        if not base_url or not model:
            raise RuntimeError("openai embedding backend requires explicit --embedding-base-url and --embedding-model")
        return OpenAIEmbeddingVectorizer(dim=args.text_dim, base_url=base_url, model=model, api_key=api_key)
    if backend == "hf":
        model_path = clean_text(getattr(args, "embedding_model", ""))
        if not model_path:
            raise RuntimeError("hf embedding backend requires explicit --embedding-model local path")
        if not os.path.exists(model_path):
            raise RuntimeError(f"hf embedding backend requires a local model path; not found: {model_path}")
        return BgeM3DenseVectorizer(dim=args.text_dim, model_path=model_path, device=str(device), max_length=args.embedding_max_length)
    raise RuntimeError(f"unsupported embedding backend: {backend}")


class V2EvidenceScorer(nn.Module):
    def __init__(self, text_dim: int, feature_dim: int, hidden_dim: int = 256, layers: int = 2, roles: int = 3):
        super().__init__()
        self.query_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.SiLU())
        self.event_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.SiLU())
        self.feature_proj = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim // 2), nn.SiLU())
        self.session_embedding = nn.Embedding(256, hidden_dim)
        self.chunk_embedding = nn.Embedding(32, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.graph_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(layers)))
        pair_dim = hidden_dim * 4 + hidden_dim // 2
        self.event_head = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.pack_head = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.role_head = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, roles))

    def forward(self, q_vec: torch.Tensor, e_vec: torch.Tensor, features: torch.Tensor, session_idx: torch.Tensor, chunk_idx: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        q = self.query_proj(q_vec)
        e = self.event_proj(e_vec)
        f = self.feature_proj(features)
        session_idx = session_idx.clamp_min(0).clamp_max(255)
        chunk_idx = chunk_idx.clamp_min(0).clamp_max(31)
        h = e + self.session_embedding(session_idx) + self.chunk_embedding(chunk_idx)
        h = self.graph_encoder(h, src_key_padding_mask=~mask)
        qx = q.unsqueeze(1).expand_as(h)
        pair = torch.cat([qx, h, qx * h, torch.abs(qx - h), f], dim=-1)
        event_logits = self.event_head(pair).squeeze(-1).masked_fill(~mask, -1e4)
        pack_logits = self.pack_head(pair).squeeze(-1).masked_fill(~mask, -1e4)
        role_logits = self.role_head(pair).masked_fill(~mask.unsqueeze(-1), 0.0)
        return {"event_logits": event_logits, "pack_logits": pack_logits, "role_logits": role_logits}


@dataclass
class Batch:
    q_vec: torch.Tensor
    e_vec: torch.Tensor
    features: torch.Tensor
    session_idx: torch.Tensor
    chunk_idx: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor
    role_labels: torch.Tensor
    qids: list[str]
    event_ids: list[list[str]]


def make_batch(samples: Sequence[Mapping[str, Any]], vectorizer: Any, text_dim: int, feature_dim: int, device: torch.device) -> Batch:
    max_c = max(len(s["candidates"]) for s in samples)
    b = len(samples)
    q_vec = torch.zeros((b, text_dim), dtype=torch.float32)
    e_vec = torch.zeros((b, max_c, text_dim), dtype=torch.float32)
    features = torch.zeros((b, max_c, feature_dim), dtype=torch.float32)
    session_idx = torch.zeros((b, max_c), dtype=torch.long)
    chunk_idx = torch.zeros((b, max_c), dtype=torch.long)
    mask = torch.zeros((b, max_c), dtype=torch.bool)
    labels = torch.zeros((b, max_c), dtype=torch.float32)
    role_labels = torch.zeros((b, max_c), dtype=torch.long)
    qids=[]; event_ids=[]
    for i, sample in enumerate(samples):
        qids.append(str(sample["question_id"])); event_ids.append([])
        q_vec[i] = vectorizer.encode_one(str(sample.get("question", "")))
        for j, cand in enumerate(sample["candidates"]):
            e_vec[i, j] = vectorizer.encode_one(str(cand.get("text", "")))
            feat = list(cand.get("features") or [])
            features[i, j, : min(feature_dim, len(feat))] = torch.tensor(feat[:feature_dim], dtype=torch.float32)
            session_idx[i, j] = int(cand.get("session_index", 0) or 0)
            chunk_idx[i, j] = int(cand.get("chunk_index", 0) or 0)
            mask[i, j] = True
            labels[i, j] = float(cand.get("label", 0) or 0)
            role_labels[i, j] = int(cand.get("role_label", 0) or 0)
            event_ids[-1].append(str(cand.get("event_id", "")))
    return Batch(q_vec.to(device), e_vec.to(device), features.to(device), session_idx.to(device), chunk_idx.to(device), mask.to(device), labels.to(device), role_labels.to(device), qids, event_ids)


def ranking_loss(event_logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses=[]
    for logits, y, m in zip(event_logits, labels, mask):
        valid_logits = logits[m]
        valid_y = y[m]
        pos = valid_y > 0.5
        if not bool(pos.any()):
            continue
        log_probs = F.log_softmax(valid_logits, dim=0)
        target = valid_y / valid_y.sum().clamp_min(1.0)
        losses.append(-(target * log_probs).sum())
    if not losses:
        return event_logits.sum() * 0.0
    return torch.stack(losses).mean()


def eval_model(model: V2EvidenceScorer, samples: Sequence[Mapping[str, Any]], vectorizer: Any, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model.eval()
    totals={"n":0,"event_recall_at_1":0,"event_recall_at_5":0,"event_recall_at_10":0,"pack_recall_at_10":0,"mrr":0.0}
    with torch.no_grad():
        for start in range(0, len(samples), args.batch_size):
            batch_samples=samples[start:start+args.batch_size]
            batch=make_batch(batch_samples, vectorizer, args.text_dim, args.feature_dim, device)
            out=model(batch.q_vec,batch.e_vec,batch.features,batch.session_idx,batch.chunk_idx,batch.mask)
            event_scores=out["event_logits"].detach().cpu()
            pack_scores=out["pack_logits"].detach().cpu()
            labels=batch.labels.detach().cpu()
            mask=batch.mask.detach().cpu()
            for i in range(len(batch_samples)):
                valid=torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
                if valid.numel()==0: continue
                pos=set(torch.nonzero(labels[i][valid]>0.5, as_tuple=False).squeeze(-1).tolist())
                if not pos: continue
                order=torch.argsort(event_scores[i][valid], descending=True).tolist()
                pack_order=torch.argsort(pack_scores[i][valid], descending=True).tolist()
                totals["n"]+=1
                totals["event_recall_at_1"] += int(any(idx in pos for idx in order[:1]))
                totals["event_recall_at_5"] += int(any(idx in pos for idx in order[:5]))
                totals["event_recall_at_10"] += int(any(idx in pos for idx in order[:10]))
                totals["pack_recall_at_10"] += int(any(idx in pos for idx in pack_order[:10]))
                first_rank=next((r+1 for r,idx in enumerate(order) if idx in pos), None)
                totals["mrr"] += 0.0 if first_rank is None else 1.0/first_rank
    n=max(1, totals["n"])
    return {k:(round(v/n,6) if k!="n" else v) for k,v in totals.items()}


def precompute_embeddings(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = out_dir / "embeddings.pt"
    manifest_path = out_dir / "manifest.json"
    items_path = out_dir / "items.jsonl"
    if (tensor_path.exists() or manifest_path.exists() or items_path.exists()) and not args.overwrite:
        raise RuntimeError(f"embedding cache output already exists; pass --overwrite explicitly: {out_dir}")
    samples = read_jsonl(Path(args.samples))
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError("no samples for embedding precompute")
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --cpu explicitly for CPU precompute")
    if args.embedding_backend != "hf":
        raise RuntimeError("precompute currently supports only --embedding-backend hf for local bge-m3")
    device = torch.device("cpu" if args.cpu else "cuda")
    vectorizer = BgeM3DenseVectorizer(
        dim=args.text_dim,
        model_path=args.embedding_model,
        device=str(device),
        max_length=args.embedding_max_length,
    )
    items = iter_sample_texts(samples)
    keys = [item["key"] for item in items]
    texts = [item["text"] for item in items]
    vectors: list[torch.Tensor] = []
    total = len(texts)
    started = time.time()
    for start in range(0, total, args.batch_size):
        batch_texts = texts[start:start + args.batch_size]
        vecs = vectorizer.encode_batch(batch_texts, batch_size=args.batch_size)
        vectors.append(vecs.to(dtype=torch.float16 if args.dtype == "float16" else torch.float32))
        done = min(total, start + len(batch_texts))
        if done == total or done % max(1, args.log_every) == 0:
            elapsed = time.time() - started
            print(json.dumps({"done": done, "total": total, "elapsed_sec": round(elapsed, 3)}, ensure_ascii=False), flush=True)
    matrix = torch.cat(vectors, dim=0).contiguous()
    if matrix.shape != (len(keys), args.text_dim):
        raise RuntimeError(f"embedding matrix shape mismatch: got {tuple(matrix.shape)} expected {(len(keys), args.text_dim)}")
    torch.save({"keys": keys, "vectors": matrix}, tensor_path)
    item_rows=[]
    for item in items:
        item_rows.append({
            "key": item["key"],
            "kind": item.get("kind", ""),
            "question_id": item.get("question_id", ""),
            "event_id": item.get("event_id", ""),
            "text_len": len(item.get("text", "")),
            "preview": clean_text(item.get("text", ""))[:200],
        })
    write_jsonl(items_path, item_rows)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "samples": str(Path(args.samples).resolve()),
        "sample_count": len(samples),
        "unique_text_count": len(keys),
        "text_dim": int(args.text_dim),
        "dtype": args.dtype,
        "embedding_backend": args.embedding_backend,
        "embedding_model": clean_text(args.embedding_model),
        "embedding_max_length": int(args.embedding_max_length),
        "device": str(device),
        "outputs": {"tensor": str(tensor_path), "items": str(items_path), "manifest": str(manifest_path)},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def smoke(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(Path(args.samples))
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError("no samples for smoke")
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --cpu explicitly for CPU smoke")
    device = torch.device("cpu" if args.cpu else "cuda")
    vectorizer = build_vectorizer(args, device)
    model = V2EvidenceScorer(args.text_dim, args.feature_dim, args.hidden_dim, args.layers).to(device)
    model.eval()
    batch = make_batch(samples, vectorizer, args.text_dim, args.feature_dim, device)
    with torch.no_grad():
        out = model(batch.q_vec, batch.e_vec, batch.features, batch.session_idx, batch.chunk_idx, batch.mask)
        loss_rank = ranking_loss(out["event_logits"], batch.labels, batch.mask)
        pack_loss = F.binary_cross_entropy_with_logits(out["pack_logits"][batch.mask], batch.labels[batch.mask])
        role_loss = F.cross_entropy(out["role_logits"][batch.mask], batch.role_labels[batch.mask])
    metrics = eval_model(model, samples, vectorizer, args, device)
    report = {
        "status": "ok",
        "mode": "forward_only_no_optimizer_step",
        "device": str(device),
        "sample_count": len(samples),
        "candidate_count_max": max(len(s.get("candidates", [])) for s in samples),
        "candidate_count_avg": round(sum(len(s.get("candidates", [])) for s in samples) / max(1, len(samples)), 3),
        "losses": {
            "ranking_loss": round(float(loss_rank.detach().cpu()), 6),
            "pack_loss": round(float(pack_loss.detach().cpu()), 6),
            "role_loss": round(float(role_loss.detach().cpu()), 6),
        },
        "metrics": metrics,
        "embedding_backend": clean_text(getattr(args, "embedding_backend", "")),
        "embedding_model": clean_text(getattr(args, "embedding_model", "")),
        "embedding_cache": clean_text(getattr(args, "embedding_cache", "")),
        "embedding_max_length": int(getattr(args, "embedding_max_length", 0) or 0),
        "text_dim": int(args.text_dim),
        "hidden_dim": int(args.hidden_dim),
        "layers": int(args.layers),
    }
    (out_dir / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def train(args: argparse.Namespace) -> None:
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    samples=read_jsonl(Path(args.samples))
    if args.limit>0:
        samples=samples[:args.limit]
    train_samples=[s for s in samples if s.get("split") == "train"]
    val_samples=[s for s in samples if s.get("split") == "val"]
    if not train_samples:
        raise RuntimeError("no train samples")
    if not val_samples:
        raise RuntimeError("no val samples; rebuild dataset with a non-empty validation split")
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --cpu explicitly for CPU train")
    device=torch.device("cpu" if args.cpu else "cuda")
    vectorizer=build_vectorizer(args, device)
    model=V2EvidenceScorer(args.text_dim,args.feature_dim,args.hidden_dim,args.layers).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best=None
    history=[]
    rng=random.Random(args.seed)
    for epoch in range(1,args.epochs+1):
        model.train(); rng.shuffle(train_samples)
        losses=[]
        for start in range(0,len(train_samples),args.batch_size):
            batch_samples=train_samples[start:start+args.batch_size]
            batch=make_batch(batch_samples, vectorizer, args.text_dim, args.feature_dim, device)
            out=model(batch.q_vec,batch.e_vec,batch.features,batch.session_idx,batch.chunk_idx,batch.mask)
            loss_rank=ranking_loss(out["event_logits"],batch.labels,batch.mask)
            bce=F.binary_cross_entropy_with_logits(out["pack_logits"][batch.mask], batch.labels[batch.mask])
            role=F.cross_entropy(out["role_logits"][batch.mask], batch.role_labels[batch.mask])
            loss=loss_rank + args.pack_loss_weight*bce + args.role_loss_weight*role
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        val=eval_model(model,val_samples,vectorizer,args,device)
        train_eval=eval_model(model,train_samples[:min(len(train_samples),100)],vectorizer,args,device)
        summary={"epoch":epoch,"loss":round(sum(losses)/max(1,len(losses)),6),"train":train_eval,"val":val}
        history.append(summary)
        print(json.dumps(summary,ensure_ascii=False), flush=True)
        score=val.get("event_recall_at_5",0.0)+0.1*val.get("mrr",0.0)
        if best is None or score > best[0]:
            best=(score,epoch)
            torch.save({"model_state":model.state_dict(),"args":vars(args),"epoch":epoch,"val":val}, out_dir/"tmcra_v2_scorer.pt")
    (out_dir/"train_history.json").write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding="utf-8")
    report={"best": {"score": best[0] if best else None, "epoch": best[1] if best else None}, "final": history[-1] if history else {}, "checkpoint": str(out_dir/"tmcra_v2_scorer.pt")}
    (out_dir/"train_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True))


def main() -> int:
    p=argparse.ArgumentParser(description="TMCRA v2 LongMemEval semantic-graph scorer pipeline")
    sub=p.add_subparsers(dest="cmd", required=True)
    b=sub.add_parser("build")
    b.add_argument("--aligned-queries", required=True)
    b.add_argument("--shard-glob", default=os.getenv("TMCRA_SHARD_GLOB", "runs/shard_*.json"))
    b.add_argument("--out-dir", required=True)
    b.add_argument("--val-ratio", type=float, default=0.2)
    pc=sub.add_parser("precompute")
    pc.add_argument("--samples", required=True)
    pc.add_argument("--out-dir", required=True)
    pc.add_argument("--text-dim", type=int, default=1024)
    pc.add_argument("--embedding-backend", choices=["hf"], required=True)
    pc.add_argument("--embedding-model", required=True)
    pc.add_argument("--embedding-max-length", type=int, default=8192)
    pc.add_argument("--batch-size", type=int, default=2)
    pc.add_argument("--log-every", type=int, default=100)
    pc.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    pc.add_argument("--limit", type=int, default=0)
    pc.add_argument("--overwrite", action="store_true")
    pc.add_argument("--cpu", action="store_true")
    t=sub.add_parser("train")
    t.add_argument("--samples", required=True)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--text-dim", type=int, default=1024)
    t.add_argument("--embedding-backend", choices=["openai", "hf"], required=True)
    t.add_argument("--embedding-base-url", default="")
    t.add_argument("--embedding-model", default="")
    t.add_argument("--embedding-cache", default="")
    t.add_argument("--embedding-api-key", default="")
    t.add_argument("--embedding-max-length", type=int, default=8192)
    t.add_argument("--feature-dim", type=int, default=6)
    t.add_argument("--hidden-dim", type=int, default=256)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--pack-loss-weight", type=float, default=0.3)
    t.add_argument("--role-loss-weight", type=float, default=0.15)
    t.add_argument("--seed", type=int, default=13)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--cpu", action="store_true")
    sm=sub.add_parser("smoke")
    sm.add_argument("--samples", required=True)
    sm.add_argument("--out-dir", required=True)
    sm.add_argument("--text-dim", type=int, default=1024)
    sm.add_argument("--embedding-backend", choices=["openai", "hf"], required=True)
    sm.add_argument("--embedding-base-url", default="")
    sm.add_argument("--embedding-model", default="")
    sm.add_argument("--embedding-cache", default="")
    sm.add_argument("--embedding-api-key", default="")
    sm.add_argument("--embedding-max-length", type=int, default=8192)
    sm.add_argument("--feature-dim", type=int, default=6)
    sm.add_argument("--hidden-dim", type=int, default=96)
    sm.add_argument("--layers", type=int, default=1)
    sm.add_argument("--batch-size", type=int, default=8)
    sm.add_argument("--limit", type=int, default=4)
    sm.add_argument("--cpu", action="store_true")
    args=p.parse_args()
    if args.cmd == "build": build_dataset(args)
    elif args.cmd == "precompute": precompute_embeddings(args)
    elif args.cmd == "train": train(args)
    elif args.cmd == "smoke": smoke(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
