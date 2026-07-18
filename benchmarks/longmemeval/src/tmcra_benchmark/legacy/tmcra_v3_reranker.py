#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmcra_v3_schema import (
    CHANNEL_NAMES,
    SCHEMA_VERSION,
    channel_vector,
    clean_text,
    read_jsonl,
    validate_sample,
    write_jsonl,
)
from tmcra_model_assets import load_optional_hf_model_manifest


def pair_key(sample: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    payload = "\0".join(
        (
            clean_text(sample.get("question_id")),
            clean_text(candidate.get("candidate_id")),
            clean_text(sample.get("query_text")),
            clean_text(candidate.get("text")),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_int(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def atomic_torch_save(payload: Mapping[str, Any], target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, target)


def command_precompute_cross(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA is unavailable; pass --cpu explicitly")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    sample_paths = [Path(value) for value in args.samples]
    rows = [row for path in sample_paths for row in read_jsonl(path)]
    for row in rows:
        validate_sample(
            row,
            require_positive=False,
            allowed_splits=("full_eval", "train", "holdout", "aux_train", "aux_dev"),
        )
    model_dir = Path(args.model).resolve()
    model_manifest = load_optional_hf_model_manifest(model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_pair_count = 0
    duplicate_pair_count = 0
    for row in rows:
        for candidate in row["candidates"]:
            raw_pair_count += 1
            key = pair_key(row, candidate)
            if key in seen:
                duplicate_pair_count += 1
                continue
            seen.add(key)
            items.append(
                {
                    "key": key,
                    "question_id": row["question_id"],
                    "candidate_id": candidate["candidate_id"],
                    "query_text": row["query_text"],
                    "candidate_text": candidate["text"],
                }
            )
    write_jsonl(
        out_dir / "items.jsonl",
        [
            {
                "key": item["key"],
                "question_id": item["question_id"],
                "candidate_id": item["candidate_id"],
                "query_len": len(item["query_text"]),
                "candidate_len": len(item["candidate_text"]),
            }
            for item in items
        ],
    )
    device = torch.device("cpu" if args.cpu else "cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32 if args.cpu else torch.float16,
    ).to(device)
    model.eval()
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        raise RuntimeError("reranker config.hidden_size is missing")
    started = time.time()
    shard_paths: list[Path] = []
    for start in range(0, len(items), args.shard_size):
        stop = min(len(items), start + args.shard_size)
        shard_index = start // args.shard_size
        shard_path = shard_dir / f"shard_{shard_index:05d}.pt"
        expected_keys = [item["key"] for item in items[start:stop]]
        if shard_path.exists():
            cached = torch.load(shard_path, map_location="cpu", weights_only=True)
            if list(cached.get("keys") or []) != expected_keys:
                raise RuntimeError(f"cross-cache resume key mismatch: {shard_path}")
            representations = cached.get("representations")
            logits = cached.get("semantic_logits")
            if representations is None or logits is None:
                raise RuntimeError(f"cross-cache resume payload invalid: {shard_path}")
            if tuple(representations.shape) != (len(expected_keys), hidden_size):
                raise RuntimeError(f"cross-cache resume representation shape mismatch: {shard_path}")
            print(json.dumps({"status": "resume", "shard": shard_index, "done": stop, "total": len(items)}), flush=True)
            shard_paths.append(shard_path)
            continue
        rep_chunks: list[torch.Tensor] = []
        logit_chunks: list[torch.Tensor] = []
        shard_items = items[start:stop]
        with torch.inference_mode():
            for batch_start in range(0, len(shard_items), args.batch_size):
                batch = shard_items[batch_start : batch_start + args.batch_size]
                encoded = tokenizer(
                    [item["query_text"] for item in batch],
                    [item["candidate_text"] for item in batch],
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                token_lengths = encoded["attention_mask"].sum(dim=1)
                longest = int(token_lengths.max())
                if longest > args.max_length:
                    offending = int(torch.argmax(token_lengths))
                    raise RuntimeError(
                        f"cross-encoder pair has {longest} tokens, exceeding strict max_length={args.max_length}; "
                        f"qid={batch[offending]['question_id']} candidate={batch[offending]['candidate_id']}"
                    )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                output = model(**encoded, output_hidden_states=True, return_dict=True)
                if output.hidden_states is None:
                    raise RuntimeError("reranker did not return hidden states")
                representation = output.hidden_states[-1][:, 0].detach().cpu().float()
                semantic = output.logits.detach().cpu().float()
                if semantic.ndim == 2 and semantic.shape[1] == 1:
                    semantic = semantic[:, 0]
                elif semantic.ndim != 1:
                    raise RuntimeError(f"expected one reranker logit per pair, got {tuple(semantic.shape)}")
                rep_chunks.append(representation.to(torch.float16))
                logit_chunks.append(semantic)
        representations = torch.cat(rep_chunks, dim=0).contiguous()
        logits = torch.cat(logit_chunks, dim=0).contiguous()
        atomic_torch_save(
            {"keys": expected_keys, "representations": representations, "semantic_logits": logits},
            shard_path,
        )
        print(
            json.dumps(
                {
                    "status": "encoded",
                    "shard": shard_index,
                    "done": stop,
                    "total": len(items),
                    "elapsed_sec": round(time.time() - started, 3),
                }
            ),
            flush=True,
        )
        shard_paths.append(shard_path)
    all_keys: list[str] = []
    all_representations: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    for path in shard_paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        all_keys.extend(list(payload["keys"]))
        all_representations.append(payload["representations"])
        all_logits.append(payload["semantic_logits"])
    expected_all_keys = [item["key"] for item in items]
    representations = torch.cat(all_representations, dim=0).contiguous()
    logits = torch.cat(all_logits, dim=0).contiguous()
    if all_keys != expected_all_keys:
        raise RuntimeError("cross-cache final key order mismatch")
    if tuple(representations.shape) != (len(items), hidden_size) or tuple(logits.shape) != (len(items),):
        raise RuntimeError("cross-cache final tensor shape mismatch")
    atomic_torch_save(
        {"keys": all_keys, "representations": representations, "semantic_logits": logits},
        out_dir / "cross_cache.pt",
    )
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema_version": SCHEMA_VERSION,
        "samples": [str(path.resolve()) for path in sample_paths],
        "sample_count": len(rows),
        "pair_count": len(items),
        "raw_pair_count": raw_pair_count,
        "duplicate_pair_count": duplicate_pair_count,
        "hidden_size": hidden_size,
        "max_length": args.max_length,
        "strict_no_truncation": True,
        "dtype": "float16",
        "model": str(model_dir),
        "model_repo_id": model_manifest.get("repo_id"),
        "model_revision": model_manifest.get("revision"),
        "elapsed_sec": round(time.time() - started, 3),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


class CrossRepresentationCache:
    def __init__(self, root: str):
        directory = Path(root)
        manifest_path = directory / "manifest.json"
        cache_path = directory / "cross_cache.pt"
        if not manifest_path.exists() or not cache_path.exists():
            raise FileNotFoundError(f"complete cross cache is required: {directory}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        keys = list(payload.get("keys") or [])
        representations = payload.get("representations")
        semantic_logits = payload.get("semantic_logits")
        if not keys or representations is None or semantic_logits is None:
            raise RuntimeError(f"invalid cross cache: {cache_path}")
        if len(keys) != len(set(keys)) or len(keys) != representations.shape[0] or len(keys) != semantic_logits.shape[0]:
            raise RuntimeError(f"cross cache row mismatch: {cache_path}")
        self.index = {key: index for index, key in enumerate(keys)}
        self.representations = representations.float().contiguous()
        self.semantic_logits = semantic_logits.float().contiguous()
        self.hidden_size = int(representations.shape[1])

    def get(self, sample: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        key = pair_key(sample, candidate)
        index = self.index.get(key)
        if index is None:
            raise RuntimeError(
                f"cross cache miss: qid={sample.get('question_id')} candidate={candidate.get('candidate_id')} key={key}"
            )
        return self.representations[index], self.semantic_logits[index]


class ChannelAwareMemoryReranker(nn.Module):
    def __init__(
        self,
        representation_dim: int,
        channel_dim: int,
        hidden_dim: int = 256,
        layers: int = 2,
        channel_mean: torch.Tensor | None = None,
        channel_scale: torch.Tensor | None = None,
    ):
        super().__init__()
        mean = torch.zeros(channel_dim, dtype=torch.float32) if channel_mean is None else channel_mean.float()
        scale = torch.ones(channel_dim, dtype=torch.float32) if channel_scale is None else channel_scale.float()
        if tuple(mean.shape) != (channel_dim,) or tuple(scale.shape) != (channel_dim,):
            raise RuntimeError("channel normalization tensors have the wrong shape")
        self.register_buffer("channel_mean", mean.contiguous())
        self.register_buffer("channel_scale", scale.clamp_min(1e-4).contiguous())
        self.representation_proj = nn.Sequential(
            nn.LayerNorm(representation_dim),
            nn.Linear(representation_dim, hidden_dim),
            nn.GELU(),
        )
        self.semantic_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU())
        self.channel_proj = nn.Sequential(nn.Linear(channel_dim, hidden_dim), nn.GELU())
        self.channel_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 3,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(layers)))
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self,
        representations: torch.Tensor,
        semantic_logits: torch.Tensor,
        channels: torch.Tensor,
        mask: torch.Tensor,
        *,
        ablation: str = "full",
    ) -> torch.Tensor:
        if ablation not in {"full", "no_language", "no_channels", "no_graph"}:
            raise RuntimeError(f"unknown ablation: {ablation}")
        reps = representations
        semantic = semantic_logits
        feature_values = (channels - self.channel_mean) / self.channel_scale
        if ablation == "no_language":
            reps = torch.zeros_like(reps)
            semantic = torch.zeros_like(semantic)
        if ablation == "no_channels":
            feature_values = torch.zeros_like(feature_values)
        elif ablation == "no_graph":
            feature_values = feature_values.clone()
            feature_values[..., 2:5] = 0.0
        rep_hidden = self.representation_proj(reps)
        semantic_hidden = self.semantic_proj(semantic.unsqueeze(-1))
        channel_hidden = self.channel_proj(feature_values)
        gate = self.channel_gate(torch.cat((rep_hidden, semantic_hidden), dim=-1))
        hidden = rep_hidden + semantic_hidden + gate * channel_hidden
        hidden = self.set_encoder(hidden, src_key_padding_mask=~mask)
        delta = self.delta_head(hidden).squeeze(-1)
        scores = semantic + delta
        return scores.masked_fill(~mask, -1e4)


def select_training_candidates(sample: Mapping[str, Any], max_candidates: int, epoch: int) -> list[dict[str, Any]]:
    candidates = list(sample["candidates"])
    positives = [candidate for candidate in candidates if candidate["labels"]["relevance"]]
    negatives = [candidate for candidate in candidates if not candidate["labels"]["relevance"]]
    if not positives:
        raise RuntimeError(f"training sample has no positive: {sample.get('question_id')}")
    if max_candidates < 2:
        raise RuntimeError("max_train_candidates must be at least 2")
    positive_budget = min(len(positives), max(1, max_candidates // 2))
    positives.sort(
        key=lambda candidate: (
            -float(candidate["channels"]["graph_rank_rr"]),
            -float(candidate["channels"]["dense_score"]),
            clean_text(candidate["candidate_id"]),
        )
    )
    if len(positives) > positive_budget:
        anchor_count = min(2, positive_budget)
        anchors = positives[:anchor_count]
        rotating = positives[anchor_count:]
        rng = random.Random(stable_int(f"positive-bag:{epoch}:{sample['question_id']}"))
        rng.shuffle(rotating)
        positives = anchors + rotating[: positive_budget - anchor_count]
    negatives.sort(
        key=lambda candidate: (
            -int(bool(candidate["labels"]["hard_negative"])),
            -float(candidate["channels"]["graph_rank_rr"]),
            -float(candidate["channels"]["dense_score"]),
            clean_text(candidate["candidate_id"]),
        )
    )
    selected = positives + negatives[: max(0, max_candidates - len(positives))]
    rng = random.Random(stable_int(f"candidate-shuffle:{epoch}:{sample['question_id']}"))
    rng.shuffle(selected)
    return selected


def make_batch(
    samples: Sequence[Mapping[str, Any]],
    cache: CrossRepresentationCache,
    device: torch.device,
    *,
    training: bool,
    max_candidates: int,
    epoch: int,
) -> dict[str, Any]:
    selected_by_sample = [
        select_training_candidates(sample, max_candidates, epoch) if training else list(sample["candidates"])
        for sample in samples
    ]
    max_count = max(len(candidates) for candidates in selected_by_sample)
    batch_size = len(samples)
    representations = torch.zeros((batch_size, max_count, cache.hidden_size), dtype=torch.float32)
    semantic = torch.zeros((batch_size, max_count), dtype=torch.float32)
    channels = torch.zeros((batch_size, max_count, len(CHANNEL_NAMES)), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_count), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_count), dtype=torch.bool)
    candidate_ids: list[list[str]] = []
    sample_weights = torch.tensor(
        [float((sample.get("supervision") or {}).get("training_weight", 1.0) or 1.0) for sample in samples],
        dtype=torch.float32,
    )
    for row_index, (sample, candidates) in enumerate(zip(samples, selected_by_sample)):
        ids: list[str] = []
        for candidate_index, candidate in enumerate(candidates):
            representation, semantic_logit = cache.get(sample, candidate)
            representations[row_index, candidate_index] = representation
            semantic[row_index, candidate_index] = semantic_logit
            channels[row_index, candidate_index] = torch.tensor(channel_vector(candidate), dtype=torch.float32)
            labels[row_index, candidate_index] = float(bool(candidate["labels"]["relevance"]))
            mask[row_index, candidate_index] = True
            ids.append(clean_text(candidate["candidate_id"]))
        candidate_ids.append(ids)
    return {
        "representations": representations.to(device),
        "semantic_logits": semantic.to(device),
        "channels": channels.to(device),
        "labels": labels.to(device),
        "mask": mask.to(device),
        "candidate_ids": candidate_ids,
        "sample_weights": sample_weights.to(device),
    }


def listwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    if sample_weights is None:
        sample_weights = torch.ones(scores.shape[0], dtype=scores.dtype, device=scores.device)
    for row_scores, row_labels, row_mask, row_weight in zip(scores, labels, mask, sample_weights):
        valid_scores = row_scores[row_mask]
        valid_labels = row_labels[row_mask]
        if not bool((valid_labels > 0.5).any()):
            continue
        log_probabilities = F.log_softmax(valid_scores, dim=0)
        positive_mask = valid_labels > 0.5
        losses.append(-torch.logsumexp(log_probabilities[positive_mask], dim=0))
        weights.append(row_weight)
    if not losses:
        return scores.sum() * 0.0
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.stack(weights).to(dtype=loss_tensor.dtype).clamp_min(1e-6)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum()


def rank_metrics_update(target: dict[str, float], scores: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> None:
    valid_scores = scores[valid]
    valid_labels = labels[valid]
    positives = set(torch.nonzero(valid_labels > 0.5, as_tuple=False).squeeze(-1).tolist())
    target["n"] += 1
    if not positives:
        target["candidate_miss"] += 1
        return
    order = torch.argsort(valid_scores, descending=True).tolist()
    rank = next((position + 1 for position, index in enumerate(order) if index in positives), None)
    for cutoff in (1, 3, 5, 10, 20):
        target[f"recall@{cutoff}"] += int(rank is not None and rank <= cutoff)
    target["mrr"] += 0.0 if rank is None else 1.0 / rank


def finalize_metrics(values: Mapping[str, float]) -> dict[str, Any]:
    n = max(1.0, float(values.get("n", 0.0)))
    output: dict[str, Any] = {"n": int(values.get("n", 0.0))}
    for key, value in sorted(values.items()):
        if key == "n":
            continue
        output[key] = round(float(value) / n, 6)
    return output


def evaluate_model(
    model: ChannelAwareMemoryReranker,
    rows: Sequence[Mapping[str, Any]],
    cache: CrossRepresentationCache,
    device: torch.device,
    *,
    batch_size: int,
    ablation: str,
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            samples = rows[start : start + batch_size]
            batch = make_batch(samples, cache, device, training=False, max_candidates=0, epoch=0)
            scores = model(
                batch["representations"],
                batch["semantic_logits"],
                batch["channels"],
                batch["mask"],
                ablation=ablation,
            ).cpu()
            labels = batch["labels"].cpu()
            mask = batch["mask"].cpu()
            for index in range(len(samples)):
                rank_metrics_update(totals, scores[index], labels[index], mask[index])
    return finalize_metrics(totals)


def split_train_dev(rows: Sequence[Mapping[str, Any]], dev_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (stable_int(f"tmcra-v3-dev:{row['question_id']}"), row["question_id"]))
    if dev_count <= 0 or dev_count >= len(ordered):
        raise RuntimeError("dev_count must leave non-empty train and dev sets")
    dev = list(ordered[:dev_count])
    train = list(ordered[dev_count:])
    return train, dev


def compute_channel_statistics(rows: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    values = [channel_vector(candidate) for row in rows for candidate in row["candidates"]]
    if not values:
        raise RuntimeError("cannot compute channel statistics from an empty candidate set")
    matrix = torch.tensor(values, dtype=torch.float32)
    mean = matrix.mean(dim=0)
    scale = matrix.std(dim=0, unbiased=False)
    scale = torch.where(scale < 1e-4, torch.ones_like(scale), scale)
    return mean, scale


def load_model_checkpoint(path: Path, cache: CrossRepresentationCache, device: torch.device) -> tuple[ChannelAwareMemoryReranker, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("checkpoint schema version mismatch")
    if tuple(checkpoint.get("channel_names") or ()) != CHANNEL_NAMES:
        raise RuntimeError("checkpoint channel contract mismatch")
    config = checkpoint["model_config"]
    model = ChannelAwareMemoryReranker(
        representation_dim=cache.hidden_size,
        channel_dim=len(CHANNEL_NAMES),
        hidden_dim=int(config["hidden_dim"]),
        layers=int(config["layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def command_train(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.train_samples))
    for row in rows:
        validate_sample(row, require_positive=True, allowed_splits=("train", "aux_train"))
    if args.dev_samples:
        train_rows = list(rows)
        dev_rows = read_jsonl(Path(args.dev_samples))
        for row in dev_rows:
            validate_sample(row, require_positive=False, allowed_splits=("train", "full_eval", "aux_dev"))
    else:
        train_rows, dev_rows = split_train_dev(rows, args.dev_count)
    cache = CrossRepresentationCache(args.cross_cache)
    device = torch.device("cpu" if args.cpu else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --cpu explicitly")
    channel_mean, channel_scale = compute_channel_statistics(train_rows)
    model = ChannelAwareMemoryReranker(
        representation_dim=cache.hidden_size,
        channel_dim=len(CHANNEL_NAMES),
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        channel_mean=channel_mean,
        channel_scale=channel_scale,
    ).to(device)
    if args.init_checkpoint:
        initial = torch.load(Path(args.init_checkpoint), map_location="cpu", weights_only=True)
        if initial.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("initial checkpoint schema version mismatch")
        if tuple(initial.get("channel_names") or ()) != CHANNEL_NAMES:
            raise RuntimeError("initial checkpoint channel contract mismatch")
        initial_config = initial.get("model_config") or {}
        if int(initial_config.get("hidden_dim", -1)) != args.hidden_dim or int(initial_config.get("layers", -1)) != args.layers:
            raise RuntimeError("initial checkpoint architecture does not match requested hidden_dim/layers")
        transferable = {
            key: value
            for key, value in initial["model_state"].items()
            if key not in {"channel_mean", "channel_scale"}
        }
        missing, unexpected = model.load_state_dict(transferable, strict=False)
        if set(missing) != {"channel_mean", "channel_scale"} or unexpected:
            raise RuntimeError(f"unexpected initial checkpoint keys: missing={missing} unexpected={unexpected}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    best_score: float | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_rows)
        losses: list[float] = []
        for start in range(0, len(train_rows), args.batch_size):
            samples = train_rows[start : start + args.batch_size]
            batch = make_batch(
                samples,
                cache,
                device,
                training=True,
                max_candidates=args.max_train_candidates,
                epoch=epoch,
            )
            scores = model(
                batch["representations"],
                batch["semantic_logits"],
                batch["channels"],
                batch["mask"],
            )
            ranking_loss = listwise_loss(scores, batch["labels"], batch["mask"], batch["sample_weights"])
            valid_delta = (scores - batch["semantic_logits"])[batch["mask"]]
            regularization = valid_delta.square().mean() if valid_delta.numel() else scores.sum() * 0.0
            loss = ranking_loss + args.delta_l2 * regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_metrics = evaluate_model(
            model,
            dev_rows,
            cache,
            device,
            batch_size=args.batch_size,
            ablation="full",
        )
        teacher_dev_rows = [
            row
            for row in dev_rows
            if clean_text((row.get("supervision") or {}).get("target_type")) == "teacher_aligned_turn_bag"
        ]
        dev_teacher_metrics = (
            evaluate_model(
                model,
                teacher_dev_rows,
                cache,
                device,
                batch_size=args.batch_size,
                ablation="full",
            )
            if teacher_dev_rows
            else {"n": 0}
        )
        summary = {
            "epoch": epoch,
            "loss": round(sum(losses) / max(1, len(losses)), 6),
            "dev": dev_metrics,
            "dev_teacher_aligned": dev_teacher_metrics,
        }
        history.append(summary)
        print(json.dumps(summary), flush=True)
        score_metrics = dev_teacher_metrics if int(dev_teacher_metrics.get("n", 0)) >= 10 else dev_metrics
        score = (
            float(score_metrics.get("recall@1", 0.0))
            + 0.25 * float(score_metrics.get("mrr", 0.0))
            + 0.05 * float(score_metrics.get("recall@5", 0.0))
        )
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            atomic_torch_save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "channel_names": CHANNEL_NAMES,
                    "model_config": {"hidden_dim": args.hidden_dim, "layers": args.layers},
                    "model_state": model.state_dict(),
                    "channel_statistics": {
                        "mean": channel_mean.tolist(),
                        "scale": channel_scale.tolist(),
                    },
                    "epoch": epoch,
                    "dev_metrics": dev_metrics,
                    "dev_teacher_aligned_metrics": dev_teacher_metrics,
                    "cross_cache_manifest": cache.manifest,
                    "train_args": vars(args),
                },
                out_dir / "tmcra_v3_reranker.pt",
            )
        else:
            stale_epochs += 1
        if args.patience > 0 and stale_epochs >= args.patience:
            print(json.dumps({"status": "early_stop", "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break
    report = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else "",
        "channel_statistics": {"mean": channel_mean.tolist(), "scale": channel_scale.tolist()},
        "selection_metric_scope": "teacher_aligned_dev" if any(
            int(item.get("dev_teacher_aligned", {}).get("n", 0)) >= 10 for item in history
        ) else "all_dev",
        "checkpoint": str(out_dir / "tmcra_v3_reranker.pt"),
        "history": history,
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def direct_scores(sample: Mapping[str, Any], method: str) -> list[float]:
    candidates = list(sample["candidates"])
    if method == "dense":
        return [float(candidate["channels"]["dense_score"]) for candidate in candidates]
    if method == "graph":
        return [float(candidate["channels"]["graph_rank_rr"]) for candidate in candidates]
    raise RuntimeError(method)


def command_eval(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.samples))
    if args.qid_list:
        qids = {clean_text(value) for value in Path(args.qid_list).read_text(encoding="utf-8").splitlines() if clean_text(value)}
        rows = [row for row in rows if clean_text(row.get("question_id")) in qids]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("no evaluation rows")
    for row in rows:
        validate_sample(row, require_positive=False)
    cache = CrossRepresentationCache(args.cross_cache)
    device = torch.device("cpu" if args.cpu else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --cpu explicitly")
    model, checkpoint = load_model_checkpoint(Path(args.checkpoint), cache, device)
    methods = ("dense", "graph", "base_cross", "fusion", "no_language", "no_channels", "no_graph")
    totals = {method: defaultdict(float) for method in methods}
    by_type: dict[str, dict[str, dict[str, float]]] = {
        method: defaultdict(lambda: defaultdict(float)) for method in methods
    }
    by_supervision: dict[str, dict[str, dict[str, float]]] = {
        method: defaultdict(lambda: defaultdict(float)) for method in methods
    }
    predictions: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            samples = rows[start : start + args.batch_size]
            batch = make_batch(samples, cache, device, training=False, max_candidates=0, epoch=0)
            model_scores = {
                "base_cross": batch["semantic_logits"].cpu(),
                "fusion": model(batch["representations"], batch["semantic_logits"], batch["channels"], batch["mask"], ablation="full").cpu(),
                "no_language": model(batch["representations"], batch["semantic_logits"], batch["channels"], batch["mask"], ablation="no_language").cpu(),
                "no_channels": model(batch["representations"], batch["semantic_logits"], batch["channels"], batch["mask"], ablation="no_channels").cpu(),
                "no_graph": model(batch["representations"], batch["semantic_logits"], batch["channels"], batch["mask"], ablation="no_graph").cpu(),
            }
            labels = batch["labels"].cpu()
            mask = batch["mask"].cpu()
            for row_index, sample in enumerate(samples):
                qtype = clean_text(sample.get("question_type")) or "unknown"
                supervision_type = clean_text((sample.get("supervision") or {}).get("target_type")) or "unspecified"
                valid = mask[row_index]
                score_map: dict[str, torch.Tensor] = {
                    "dense": torch.tensor(direct_scores(sample, "dense"), dtype=torch.float32),
                    "graph": torch.tensor(direct_scores(sample, "graph"), dtype=torch.float32),
                }
                score_map.update({method: values[row_index][valid] for method, values in model_scores.items()})
                ranks: dict[str, int | None] = {}
                top_ids: dict[str, list[str]] = {}
                valid_labels = labels[row_index][valid]
                positives = set(torch.nonzero(valid_labels > 0.5, as_tuple=False).squeeze(-1).tolist())
                for method in methods:
                    scores = score_map[method]
                    method_valid = torch.ones_like(scores, dtype=torch.bool)
                    rank_metrics_update(totals[method], scores, valid_labels, method_valid)
                    rank_metrics_update(by_type[method][qtype], scores, valid_labels, method_valid)
                    rank_metrics_update(by_supervision[method][supervision_type], scores, valid_labels, method_valid)
                    order = torch.argsort(scores, descending=True).tolist()
                    ranks[method] = next((position + 1 for position, index in enumerate(order) if index in positives), None)
                    top_ids[method] = [batch["candidate_ids"][row_index][index] for index in order[:10]]
                predictions.append(
                    {
                        "question_id": sample["question_id"],
                        "question_type": qtype,
                        "supervision_type": supervision_type,
                        "candidate_count": int(valid.sum()),
                        "gold_candidate_count": len(positives),
                        "ranks": ranks,
                        "top10": top_ids,
                    }
                )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(out_dir / "predictions.jsonl", predictions)
    report = {
        "status": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema_version": SCHEMA_VERSION,
        "samples": str(Path(args.samples).resolve()),
        "sample_count": len(rows),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "cross_cache": str(Path(args.cross_cache).resolve()),
        "metrics": {method: finalize_metrics(totals[method]) for method in methods},
        "metrics_by_question_type": {
            method: {qtype: finalize_metrics(values) for qtype, values in sorted(by_type[method].items())}
            for method in methods
        },
        "metrics_by_supervision": {
            method: {name: finalize_metrics(values) for name, values in sorted(by_supervision[method].items())}
            for method in methods
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def command_export_evidence(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.samples))
    if args.qid_list:
        requested = [
            clean_text(value)
            for value in Path(args.qid_list).read_text(encoding="utf-8").splitlines()
            if clean_text(value)
        ]
        by_qid = {clean_text(row.get("question_id")): row for row in rows}
        missing = [qid for qid in requested if qid not in by_qid]
        if missing:
            raise RuntimeError(f"requested qids are missing from samples: {missing[:10]}")
        rows = [by_qid[qid] for qid in requested]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("no rows for evidence export")
    for row in rows:
        validate_sample(row, require_positive=False)
    cache = CrossRepresentationCache(args.cross_cache)
    device = torch.device("cpu" if args.cpu else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --cpu explicitly")
    model, checkpoint = load_model_checkpoint(Path(args.checkpoint), cache, device)
    model.eval()
    output_rows: list[dict[str, Any]] = []
    recall = Counter()
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            samples = rows[start : start + args.batch_size]
            batch = make_batch(samples, cache, device, training=False, max_candidates=0, epoch=0)
            if args.method == "fusion":
                scores = model(
                    batch["representations"],
                    batch["semantic_logits"],
                    batch["channels"],
                    batch["mask"],
                    ablation="full",
                ).cpu()
            elif args.method == "base_cross":
                scores = batch["semantic_logits"].cpu().masked_fill(~batch["mask"].cpu(), -1e4)
            elif args.method in {"dense", "graph"}:
                channel_name = "dense_score" if args.method == "dense" else "graph_rank_rr"
                scores = torch.full(batch["mask"].shape, -1e4, dtype=torch.float32)
                for row_index, sample in enumerate(samples):
                    values = [float(candidate["channels"][channel_name]) for candidate in sample["candidates"]]
                    scores[row_index, : len(values)] = torch.tensor(values, dtype=torch.float32)
            else:
                raise RuntimeError(f"unsupported evidence method: {args.method}")
            labels = batch["labels"].cpu()
            mask = batch["mask"].cpu()
            for row_index, sample in enumerate(samples):
                candidates = list(sample["candidates"])
                valid_count = int(mask[row_index].sum())
                order = torch.argsort(scores[row_index, :valid_count], descending=True).tolist()
                selected_indexes: list[int] = []
                parent_counts: Counter[tuple[int, int]] = Counter()
                session_counts: Counter[str] = Counter()
                for candidate_index in order:
                    candidate = candidates[candidate_index]
                    parent_key = (int(candidate["session_index"]), int(candidate.get("parent_chunk_index", 0)))
                    session_id = clean_text(candidate.get("session_id"))
                    if args.max_per_parent > 0 and parent_counts[parent_key] >= args.max_per_parent:
                        continue
                    if args.max_per_session > 0 and session_counts[session_id] >= args.max_per_session:
                        continue
                    selected_indexes.append(candidate_index)
                    parent_counts[parent_key] += 1
                    session_counts[session_id] += 1
                    if len(selected_indexes) >= args.top_k:
                        break
                positives = set(
                    torch.nonzero(labels[row_index, :valid_count] > 0.5, as_tuple=False).squeeze(-1).tolist()
                )
                recall["n"] += 1
                recall["hit"] += int(any(index in positives for index in selected_indexes))
                supervision_type = clean_text((sample.get("supervision") or {}).get("target_type")) or "unspecified"
                recall[f"n:{supervision_type}"] += 1
                recall[f"hit:{supervision_type}"] += int(any(index in positives for index in selected_indexes))
                evidence_windows = []
                for rank, candidate_index in enumerate(selected_indexes, start=1):
                    candidate = candidates[candidate_index]
                    evidence_windows.append(
                        {
                            "memory_id": candidate["candidate_id"],
                            "session_id": candidate["session_id"],
                            "session_index": candidate["session_index"],
                            "parent_chunk_index": candidate.get("parent_chunk_index", 0),
                            "subchunk_index": candidate.get("subchunk_index", 0),
                            "score": round(float(scores[row_index, candidate_index]), 6),
                            "rank": rank,
                            "text": candidate["text"],
                            "channels": candidate["channels"],
                        }
                    )
                output_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "question_id": sample["question_id"],
                        "question": sample["question"],
                        "question_date": sample.get("question_date", ""),
                        "question_type": sample.get("question_type", ""),
                        "gold_answer": sample.get("gold_answer", ""),
                        "answer_session_ids": sample.get("answer_session_ids", []),
                        "selected_session_ids": [window["session_id"] for window in evidence_windows],
                        "evidence_windows": evidence_windows,
                    }
                )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(out_dir / "evidence_windows.jsonl", output_rows)
    report = {
        "status": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema_version": SCHEMA_VERSION,
        "samples": str(Path(args.samples).resolve()),
        "sample_count": len(output_rows),
        "top_k": args.top_k,
        "method": args.method,
        "max_per_parent": args.max_per_parent,
        "max_per_session": args.max_per_session,
        "evidence_recall": round(float(recall["hit"]) / max(1, int(recall["n"])), 6),
        "evidence_recall_by_supervision": {
            name.removeprefix("n:"): round(
                float(recall[f"hit:{name.removeprefix('n:')}"]) / max(1, int(count)), 6
            )
            for name, count in sorted(recall.items())
            if name.startswith("n:")
        },
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "output": str(out_dir / "evidence_windows.jsonl"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    precompute = sub.add_parser("precompute-cross")
    precompute.add_argument("--samples", required=True, nargs="+")
    precompute.add_argument("--model", required=True)
    precompute.add_argument("--out-dir", required=True)
    precompute.add_argument("--max-length", type=int, default=1280)
    precompute.add_argument("--batch-size", type=int, default=4)
    precompute.add_argument("--shard-size", type=int, default=500)
    precompute.add_argument("--cpu", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--train-samples", required=True)
    train.add_argument("--dev-samples", default="")
    train.add_argument("--cross-cache", required=True)
    train.add_argument("--out-dir", required=True)
    train.add_argument("--init-checkpoint", default="")
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--max-train-candidates", type=int, default=32)
    train.add_argument("--dev-count", type=int, default=40)
    train.add_argument("--epochs", type=int, default=16)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--delta-l2", type=float, default=0.002)
    train.add_argument("--patience", type=int, default=4)
    train.add_argument("--seed", type=int, default=31)
    train.add_argument("--cpu", action="store_true")
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--samples", required=True)
    evaluate.add_argument("--cross-cache", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--out-dir", required=True)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--limit", type=int, default=0)
    evaluate.add_argument("--qid-list", default="")
    evaluate.add_argument("--cpu", action="store_true")
    export = sub.add_parser("export-evidence")
    export.add_argument("--samples", required=True)
    export.add_argument("--cross-cache", required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--out-dir", required=True)
    export.add_argument("--top-k", type=int, default=8)
    export.add_argument("--method", choices=("dense", "graph", "base_cross", "fusion"), default="fusion")
    export.add_argument("--max-per-parent", type=int, default=0)
    export.add_argument("--max-per-session", type=int, default=0)
    export.add_argument("--batch-size", type=int, default=8)
    export.add_argument("--limit", type=int, default=0)
    export.add_argument("--qid-list", default="")
    export.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.command == "precompute-cross":
        command_precompute_cross(args)
    elif args.command == "train":
        command_train(args)
    elif args.command == "eval":
        command_eval(args)
    else:
        command_export_evidence(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
