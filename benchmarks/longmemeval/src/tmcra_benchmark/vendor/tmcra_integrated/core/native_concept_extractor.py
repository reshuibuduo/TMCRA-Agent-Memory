"""
Native concept extraction based on character n-grams and co-occurrence.
No rules/KB/embeddings/LLM are used.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class NativeExtractorConfig:
    ngram_min: int = 2
    ngram_max: int = 4
    window_size: int = 10
    min_freq: int = 2
    top_k: int = 80
    entropy_threshold: float = 0.8
    alpha: float = 0.5  # PMI weight
    beta: float = 0.3   # Jaccard weight
    gamma: float = 0.2  # Frequency ratio weight


class NativeConceptExtractor:
    """Extracts concept nodes and directed relations using only statistics."""

    def __init__(self, config: NativeExtractorConfig | None = None) -> None:
        self.config = config or NativeExtractorConfig()

    def _normalize_text(self, text: str) -> str:
        # Keep characters but remove whitespace for stable n-gram extraction.
        return "".join(ch for ch in text if not ch.isspace())

    def _entropy(self, counter: Counter) -> float:
        total = sum(counter.values())
        if total <= 0:
            return 0.0
        ent = 0.0
        for count in counter.values():
            p = count / total
            ent -= p * math.log(p + 1e-12)
        return ent

    def _collect_ngrams(self, text: str) -> Tuple[Counter, Dict[str, Counter], Dict[str, Counter]]:
        counts: Counter = Counter()
        left_ctx: Dict[str, Counter] = defaultdict(Counter)
        right_ctx: Dict[str, Counter] = defaultdict(Counter)
        length = len(text)
        for i in range(length):
            for n in range(self.config.ngram_min, self.config.ngram_max + 1):
                j = i + n
                if j > length:
                    break
                gram = text[i:j]
                counts[gram] += 1
                if i > 0:
                    left_ctx[gram][text[i - 1]] += 1
                if j < length:
                    right_ctx[gram][text[j]] += 1
        return counts, left_ctx, right_ctx

    def _select_concepts(
        self,
        counts: Counter,
        left_ctx: Dict[str, Counter],
        right_ctx: Dict[str, Counter],
    ) -> List[str]:
        scored = []
        for gram, freq in counts.items():
            if freq < self.config.min_freq:
                continue
            l_ent = self._entropy(left_ctx.get(gram, Counter()))
            r_ent = self._entropy(right_ctx.get(gram, Counter()))
            ent = (l_ent + r_ent) / 2.0
            if ent < self.config.entropy_threshold:
                continue
            score = freq * (1.0 + ent)
            scored.append((score, gram))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [gram for _, gram in scored[: self.config.top_k]]

    def _scan_positions(self, text: str, concepts: set[str]) -> Tuple[Dict[int, List[str]], Counter]:
        positions: Dict[int, List[str]] = defaultdict(list)
        occ_counts: Counter = Counter()
        length = len(text)
        for i in range(length):
            for n in range(self.config.ngram_min, self.config.ngram_max + 1):
                j = i + n
                if j > length:
                    break
                gram = text[i:j]
                if gram in concepts:
                    positions[i].append(gram)
                    occ_counts[gram] += 1
        return positions, occ_counts

    def _build_cooccurrence(
        self, positions: Dict[int, List[str]], occ_counts: Counter
    ) -> Tuple[Dict[Tuple[str, str], int], Dict[str, Dict[str, int]]]:
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        context_profile: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        indices = sorted(positions.keys())
        idx_set = set(indices)
        for i in indices:
            current = positions[i]
            if not current:
                continue
            for j in range(i + 1, i + self.config.window_size + 1):
                if j not in idx_set:
                    continue
                future = positions[j]
                if not future:
                    continue
                for src in current:
                    for dst in future:
                        if src == dst:
                            continue
                        pair_counts[(src, dst)] += 1
                        context_profile[src][dst] += 1
        return pair_counts, context_profile

    def _normalize_scores(self, values: List[float]) -> List[float]:
        if not values:
            return []
        v_min = min(values)
        v_max = max(values)
        if abs(v_max - v_min) < 1e-12:
            return [0.0 for _ in values]
        return [(v - v_min) / (v_max - v_min) for v in values]

    def _compute_weights(
        self,
        pair_counts: Dict[Tuple[str, str], int],
        occ_counts: Counter,
    ) -> Dict[Tuple[str, str], float]:
        total_pairs = sum(pair_counts.values()) or 1
        pmi_vals: Dict[Tuple[str, str], float] = {}
        for (src, dst), count in pair_counts.items():
            pmi = math.log((count * total_pairs) / ((occ_counts[src] * occ_counts[dst]) + 1e-9) + 1e-9)
            pmi_vals[(src, dst)] = max(0.0, pmi)

        max_pmi = max(pmi_vals.values()) if pmi_vals else 1.0
        weights: Dict[Tuple[str, str], float] = {}
        for (src, dst), count in pair_counts.items():
            pmi_norm = (pmi_vals.get((src, dst), 0.0) / max_pmi) if max_pmi > 0 else 0.0
            jaccard = count / (occ_counts[src] + occ_counts[dst] - count + 1e-9)
            ratio = count / (min(occ_counts[src], occ_counts[dst]) + 1e-9)
            weight = (
                self.config.alpha * pmi_norm
                + self.config.beta * jaccard
                + self.config.gamma * ratio
            )
            weights[(src, dst)] = max(0.0, min(1.0, weight))
        return weights

    def extract(self, text: str) -> Dict:
        cleaned = self._normalize_text(text)
        if not cleaned:
            return {"concepts": [], "relations": [], "contexts": {}}

        counts, left_ctx, right_ctx = self._collect_ngrams(cleaned)
        concepts = self._select_concepts(counts, left_ctx, right_ctx)
        if not concepts:
            return {"concepts": [], "relations": [], "contexts": {}}

        concept_set = set(concepts)
        positions, occ_counts = self._scan_positions(cleaned, concept_set)
        pair_counts, context_profile = self._build_cooccurrence(positions, occ_counts)
        weights = self._compute_weights(pair_counts, occ_counts)

        relations = []
        for (src, dst), weight in weights.items():
            relations.append({
                "from": src,
                "to": dst,
                "relation": "co_occurs",
                "weight": float(weight),
            })

        concept_records = [{"concept": c, "type": "ngram"} for c in concepts]
        return {
            "concepts": concept_records,
            "relations": relations,
            "contexts": {k: dict(v) for k, v in context_profile.items()},
        }
