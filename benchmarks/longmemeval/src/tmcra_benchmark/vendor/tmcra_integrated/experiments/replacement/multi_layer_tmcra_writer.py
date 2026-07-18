from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Mapping, Sequence

from experiments.replacement.semantic_memory_writer import (
    DeterministicMemoryWriteGate,
    OpenAICompatSemanticMemoryWriter,
    SemanticMemoryWriterError,
    _dedupe_texts,
    _extract_memory_subject,
    _safe_text,
    _slug,
    _tokens,
)


def _clip_text(value: Any, limit: int) -> str:
    text = _safe_text(value)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip(" ,.;:") + "..."


def _category_weight(category: str) -> float:
    category = _safe_text(category).lower()
    if category in {"profile", "status", "preference", "goal", "constraint"}:
        return 0.05
    if category in {"time", "event"}:
        return 0.03
    return 0.0


def _semantic_slot_for(proposal: Mapping[str, Any]) -> str:
    category = _safe_text(proposal.get("category", "")).lower() or "fact"
    slot = _safe_text(proposal.get("semantic_slot", "")).lower()
    if slot:
        return slot
    if category == "profile":
        return "profile"
    if category == "preference":
        return "preference"
    if category == "goal":
        return "goal"
    if category == "status":
        return "status"
    if category == "time":
        return "event_time"
    return category


_UNCERTAINTY_CUE_RE = re.compile(
    r"\b("
    r"maybe|might|could|perhaps|possibly|probably|i wonder|i think|i guess|not sure|"
    r"sort of|kind of|feels like|seems like|leaning toward|drawn to|what if"
    r")\b",
    re.IGNORECASE,
)

_ABSTRACT_MEMORY_TERMS = {
    "memory",
    "identity",
    "meaning",
    "self",
    "alive",
    "aliveness",
    "experience",
    "truth",
    "freedom",
    "agency",
    "resonance",
    "dissonance",
    "pattern",
    "structure",
    "story",
    "narrative",
    "body",
    "biology",
    "transcendence",
}

_LOW_SIGNAL_TOKENS = {
    "about",
    "again",
    "already",
    "also",
    "because",
    "before",
    "being",
    "could",
    "does",
    "feel",
    "feels",
    "from",
    "have",
    "just",
    "like",
    "maybe",
    "more",
    "that",
    "this",
    "think",
    "what",
    "when",
    "where",
    "with",
    "would",
    "your",
}


def _token_set(*values: Any) -> set[str]:
    return {token for value in values for token in _tokens(_safe_text(value)) if token not in _LOW_SIGNAL_TOKENS}


def _abstract_ratio(tokens: set[str]) -> float:
    if not tokens:
        return 0.0
    return len(tokens & _ABSTRACT_MEMORY_TERMS) / max(1, len(tokens))


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


@dataclass(slots=True)
class TMCRACell:
    category: str
    semantic_slot: str
    value: str
    source_span: str
    anchors: List[str] = field(default_factory=list)
    subject: str = ""
    slot_key: str = ""
    recurrence_count: int = 1
    previous_value: str = ""
    confidence: float = 0.86
    salience: float = 0.9
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_proposal(self) -> Dict[str, Any]:
        payload = {
            "category": self.category,
            "value": self.value,
            "source_span": self.source_span,
            "semantic_slot": self.semantic_slot,
            "anchors": list(self.anchors),
            "confidence": self.confidence,
            "salience": self.salience,
            "slot_key": self.slot_key,
            "event_signature": self.metadata.get("event_signature", ""),
            "relation": self.metadata.get("relation", f"{self.category}_memory"),
            "state": self.metadata.get("state", "active"),
            "allow_parallel_state": self.metadata.get("allow_parallel_state", True),
            "tmcra_stack_metadata": dict(self.metadata),
        }
        return {
            key: value
            for key, value in payload.items()
            if value is not None and value != "" and value != []
        }


class MultiLayerTMCRASemanticMemoryWriter:
    """A local multi-layer TMCRA writer stack around one delta LLM call.

    The LLM remains L0 and only sees the current turn. Later TMCRA layers are
    deterministic/local so stacking does not multiply prompt tokens.
    """

    architecture = "multi_layer_tmcra_writer_v1"

    def __init__(
        self,
        base_writer: OpenAICompatSemanticMemoryWriter,
        *,
        max_cells: int = 2,
        value_char_budget: int = 120,
        source_span_char_budget: int = 180,
    ) -> None:
        self.base_writer = base_writer
        self.gate: DeterministicMemoryWriteGate = base_writer.gate
        self.max_proposals = max(1, int(max_cells or 2))
        self.value_char_budget = max(40, int(value_char_budget or 120))
        self.source_span_char_budget = max(60, int(source_span_char_budget or 180))
        self._slot_state: Dict[str, TMCRACell] = {}

    def available(self) -> bool:
        return self.base_writer.available()

    def propose_public_turn(
        self,
        *,
        current_turn: str,
        previous_turn: str = "",
        next_turn: str = "",
        speaker: str = "",
        session_timestamp: str = "",
        sidecar_hints: Mapping[str, Any] | None = None,
        auxiliary_evidence_texts: Sequence[Any] | None = None,
        input_mode: str = "delta",
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.available():
            raise SemanticMemoryWriterError("multi-layer TMCRA writer requires an available base writer")
        base_proposals, base_metadata = self.base_writer.propose_public_turn(
            current_turn=current_turn,
            previous_turn="",
            next_turn="",
            speaker=speaker,
            session_timestamp=session_timestamp,
            sidecar_hints={},
            auxiliary_evidence_texts=[],
            input_mode="delta",
        )
        turn_cells, turn_meta = self._turn_layer(
            base_proposals,
            current_turn=current_turn,
            speaker=speaker,
            session_timestamp=session_timestamp,
        )
        memory_cells, memory_meta = self._memory_layer(turn_cells, speaker=speaker)
        compressed_cells, compression_meta = self._compression_layer(memory_cells)
        recurrent_cells, recurrence_meta = self._recurrence_layer(compressed_cells, speaker=speaker)
        activated_cells, activation_meta = self._activation_layer(recurrent_cells)
        proposals = [cell.to_proposal() for cell in activated_cells[: self.max_proposals]]
        metadata = {
            **dict(base_metadata or {}),
            "architecture": self.architecture,
            "input_mode": "delta",
            "base_input_mode": dict(base_metadata or {}).get("input_mode", "delta"),
            "system_prompt_profile": "multi_layer_tmcra_delta_l0",
            "user_payload_keys": ["current_turn", "max_write_proposals", "speaker", "timestamp"],
            "tmcra_layers": [
                turn_meta,
                memory_meta,
                compression_meta,
                recurrence_meta,
                activation_meta,
            ],
            "tmcra_stack_state_size": len(self._slot_state),
            "tmcra_stack_input_ignored": {
                "previous_turn": bool(_safe_text(previous_turn)),
                "next_turn": bool(_safe_text(next_turn)),
                "sidecar_hints": bool(sidecar_hints),
                "auxiliary_evidence_texts": bool(auxiliary_evidence_texts),
            },
            "proposal_count_after_stack": len(proposals),
        }
        return proposals, metadata

    def _turn_layer(
        self,
        proposals: Sequence[Mapping[str, Any]],
        *,
        current_turn: str,
        speaker: str,
        session_timestamp: str,
    ) -> tuple[List[TMCRACell], Dict[str, Any]]:
        cells: List[TMCRACell] = []
        for index, proposal in enumerate(proposals):
            category = _safe_text(proposal.get("category", "")).lower().replace("-", "_") or "fact"
            value = _clip_text(proposal.get("value", ""), self.value_char_budget)
            source_span = _clip_text(proposal.get("source_span", ""), self.source_span_char_budget)
            if not value or not source_span:
                continue
            semantic_slot = _semantic_slot_for(proposal)
            anchors = _dedupe_texts(
                [
                    speaker,
                    session_timestamp,
                    *list(proposal.get("anchors", []) or []),
                    *(_tokens(value)[:4]),
                ],
                max_items=8,
            )
            cells.append(
                TMCRACell(
                    category=category,
                    semantic_slot=semantic_slot,
                    value=value,
                    source_span=source_span,
                    anchors=anchors,
                    confidence=min(1.0, max(0.0, float(proposal.get("confidence", 0.86) or 0.86))),
                    salience=min(1.0, max(0.0, float(proposal.get("salience", 0.9) or 0.9))),
                    metadata={
                        "tmcra_layers": ["T"],
                        "tmcra_turn_layer_index": index,
                        "tmcra_turn_signature": _slug(" ".join(_tokens(source_span)[:8])),
                        "target_status": _safe_text(proposal.get("target_status", "")),
                    },
                )
            )
        return cells, {
            "layer": "T",
            "name": "turn_delta_projection",
            "input_count": len(list(proposals)),
            "output_count": len(cells),
        }

    def _memory_layer(self, cells: Sequence[TMCRACell], *, speaker: str) -> tuple[List[TMCRACell], Dict[str, Any]]:
        output: List[TMCRACell] = []
        for cell in cells:
            subject = _extract_memory_subject(cell.source_span, cell.value)
            if not subject:
                subject = self._subject_hint(cell.value, cell.semantic_slot)
            subject_signature = _slug(subject or cell.semantic_slot or cell.category)
            slot_key = ".".join(
                part
                for part in [
                    _slug(speaker or "speaker"),
                    "tmcra",
                    _slug(cell.category),
                    _slug(cell.semantic_slot),
                    subject_signature,
                ]
                if part
            )
            cell.subject = subject
            cell.slot_key = slot_key
            cell.metadata = {
                **cell.metadata,
                "tmcra_layers": [*list(cell.metadata.get("tmcra_layers", [])), "M"],
                "tmcra_memory_subject": subject,
                "tmcra_subject_signature": subject_signature,
                "canonical_slot_key": slot_key,
                "event_signature": _slug(f"{cell.category} {cell.semantic_slot} {subject_signature} {' '.join(_tokens(cell.value)[:6])}"),
            }
            output.append(cell)
        return output, {
            "layer": "M",
            "name": "memory_cell_projection",
            "input_count": len(cells),
            "output_count": len(output),
        }

    def _compression_layer(self, cells: Sequence[TMCRACell]) -> tuple[List[TMCRACell], Dict[str, Any]]:
        merged: Dict[str, TMCRACell] = {}
        for cell in cells:
            dedupe_key = f"{cell.slot_key}:{_slug(' '.join(_tokens(cell.value)[:10]))}"
            existing = merged.get(dedupe_key)
            if existing is None:
                merged[dedupe_key] = cell
                continue
            if len(cell.value) < len(existing.value):
                cell.anchors = _dedupe_texts([*existing.anchors, *cell.anchors], max_items=8)
                merged[dedupe_key] = cell
            else:
                existing.anchors = _dedupe_texts([*existing.anchors, *cell.anchors], max_items=8)
        output = sorted(
            merged.values(),
            key=lambda item: (item.salience + item.confidence + _category_weight(item.category), -len(item.value)),
            reverse=True,
        )[: self.max_proposals]
        for cell in output:
            cell.metadata = {
                **cell.metadata,
                "tmcra_layers": [*list(cell.metadata.get("tmcra_layers", [])), "C"],
                "tmcra_compressed": True,
                "tmcra_value_char_budget": self.value_char_budget,
                "tmcra_source_span_char_budget": self.source_span_char_budget,
            }
        return output, {
            "layer": "C",
            "name": "cell_compression_dedupe_budget",
            "input_count": len(cells),
            "output_count": len(output),
        }

    def _recurrence_layer(self, cells: Sequence[TMCRACell], *, speaker: str) -> tuple[List[TMCRACell], Dict[str, Any]]:
        updated = 0
        output: List[TMCRACell] = []
        for cell in cells:
            previous = self._slot_state.get(cell.slot_key)
            if previous is not None:
                cell.previous_value = previous.value
                cell.recurrence_count = previous.recurrence_count + 1
                updated += 1
            self._slot_state[cell.slot_key] = cell
            cell.metadata = {
                **cell.metadata,
                "tmcra_layers": [*list(cell.metadata.get("tmcra_layers", [])), "R"],
                "tmcra_recurrence_count": cell.recurrence_count,
                "tmcra_previous_value": cell.previous_value,
                "relation": "updates_memory" if cell.previous_value and cell.previous_value != cell.value else f"{cell.category}_memory",
            }
            output.append(cell)
        return output, {
            "layer": "R",
            "name": "recurrent_slot_state",
            "input_count": len(cells),
            "output_count": len(output),
            "updated_slots": updated,
            "state_size": len(self._slot_state),
        }

    def _activation_layer(self, cells: Sequence[TMCRACell]) -> tuple[List[TMCRACell], Dict[str, Any]]:
        output: List[TMCRACell] = []
        for cell in cells:
            source_tokens = len(set(_tokens(cell.source_span)))
            confidence = min(0.98, max(cell.confidence, 0.72 + min(0.16, source_tokens / 80.0)))
            salience = min(0.98, max(cell.salience, 0.82 + _category_weight(cell.category)))
            cell.confidence = confidence
            cell.salience = salience
            cell.metadata = {
                **cell.metadata,
                "tmcra_layers": [*list(cell.metadata.get("tmcra_layers", [])), "A"],
                "tmcra_activation_score": round((confidence + salience) / 2.0, 6),
                "state": "active",
                "allow_parallel_state": cell.category not in {"profile", "status", "preference", "goal", "constraint"},
                "source_layer": self.architecture,
            }
            output.append(cell)
        return output, {
            "layer": "A",
            "name": "activation_write_budget",
            "input_count": len(cells),
            "output_count": len(output),
        }

    @staticmethod
    def _subject_hint(value: str, semantic_slot: str) -> str:
        tokens = [token for token in _tokens(value) if token not in {"maya", "user", "the", "and", "that", "with"}]
        if len(tokens) >= 2:
            return " ".join(tokens[:4])
        return _safe_text(semantic_slot)


class TMCRATransformerSemanticMemoryWriter(MultiLayerTMCRASemanticMemoryWriter):
    """Transformer-inspired TMCRA stack over memory cells, not raw tokens.

    This keeps the online LLM cost at one delta writer call, then applies local
    cell attention, residual updates, normalization, and feed-forward slot
    refinement before the normal TMCRA compression/recurrent/activation path.
    """

    architecture = "tmcra_transformer_stack_v1"

    def __init__(
        self,
        base_writer: OpenAICompatSemanticMemoryWriter,
        *,
        max_cells: int = 2,
        value_char_budget: int = 120,
        source_span_char_budget: int = 180,
        transformer_layers: int = 2,
        state_attention_k: int = 16,
    ) -> None:
        super().__init__(
            base_writer,
            max_cells=max_cells,
            value_char_budget=value_char_budget,
            source_span_char_budget=source_span_char_budget,
        )
        self.transformer_layers = max(1, int(transformer_layers or 2))
        self.state_attention_k = max(0, int(state_attention_k or 16))

    def propose_public_turn(
        self,
        *,
        current_turn: str,
        previous_turn: str = "",
        next_turn: str = "",
        speaker: str = "",
        session_timestamp: str = "",
        sidecar_hints: Mapping[str, Any] | None = None,
        auxiliary_evidence_texts: Sequence[Any] | None = None,
        input_mode: str = "delta",
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.available():
            raise SemanticMemoryWriterError("TMCRA transformer writer requires an available base writer")
        base_proposals, base_metadata = self.base_writer.propose_public_turn(
            current_turn=current_turn,
            previous_turn="",
            next_turn="",
            speaker=speaker,
            session_timestamp=session_timestamp,
            sidecar_hints={},
            auxiliary_evidence_texts=[],
            input_mode="delta",
        )
        turn_cells, turn_meta = self._turn_layer(
            base_proposals,
            current_turn=current_turn,
            speaker=speaker,
            session_timestamp=session_timestamp,
        )
        memory_cells, memory_meta = self._memory_layer(turn_cells, speaker=speaker)
        transformer_cells, transformer_meta = self._transformer_cell_layers(memory_cells)
        compressed_cells, compression_meta = self._compression_layer(transformer_cells)
        recurrent_cells, recurrence_meta = self._recurrence_layer(compressed_cells, speaker=speaker)
        activated_cells, activation_meta = self._activation_layer(recurrent_cells)
        proposals = [cell.to_proposal() for cell in activated_cells[: self.max_proposals]]
        metadata = {
            **dict(base_metadata or {}),
            "architecture": self.architecture,
            "input_mode": "delta",
            "base_input_mode": dict(base_metadata or {}).get("input_mode", "delta"),
            "system_prompt_profile": "tmcra_transformer_delta_l0",
            "user_payload_keys": ["current_turn", "max_write_proposals", "speaker", "timestamp"],
            "tmcra_layers": [
                turn_meta,
                memory_meta,
                *transformer_meta,
                compression_meta,
                recurrence_meta,
                activation_meta,
            ],
            "tmcra_stack_state_size": len(self._slot_state),
            "tmcra_transformer_layers": self.transformer_layers,
            "tmcra_state_attention_k": self.state_attention_k,
            "tmcra_stack_input_ignored": {
                "previous_turn": bool(_safe_text(previous_turn)),
                "next_turn": bool(_safe_text(next_turn)),
                "sidecar_hints": bool(sidecar_hints),
                "auxiliary_evidence_texts": bool(auxiliary_evidence_texts),
            },
            "proposal_count_after_stack": len(proposals),
        }
        return proposals, metadata

    def _transformer_cell_layers(self, cells: Sequence[TMCRACell]) -> tuple[List[TMCRACell], List[Dict[str, Any]]]:
        active_cells = list(cells)
        metas: List[Dict[str, Any]] = []
        for layer_index in range(self.transformer_layers):
            attended, attention_meta = self._cell_attention_layer(active_cells, layer_index=layer_index)
            normalized, norm_meta = self._residual_norm_layer(attended, layer_index=layer_index)
            active_cells, ff_meta = self._feed_forward_layer(normalized, layer_index=layer_index)
            metas.extend([attention_meta, norm_meta, ff_meta])
        return active_cells, metas

    def _cell_attention_layer(
        self,
        cells: Sequence[TMCRACell],
        *,
        layer_index: int,
    ) -> tuple[List[TMCRACell], Dict[str, Any]]:
        state_cells = self._state_context_cells()
        output: List[TMCRACell] = []
        attended_edges = 0
        for cell in cells:
            head_scores = self._multi_head_scores(cell, state_cells)
            best_key = ""
            best_score = 0.0
            if head_scores:
                best_key, best_score = max(head_scores.items(), key=lambda item: item[1])
            if best_key:
                attended_edges += 1
            cell.metadata = {
                **cell.metadata,
                "tmcra_layers": [*list(cell.metadata.get("tmcra_layers", [])), f"X{layer_index}.attn"],
                "tmcra_attention_heads": {key: round(value, 6) for key, value in head_scores.items()},
                "tmcra_attention_best_slot": best_key,
                "tmcra_attention_best_score": round(best_score, 6),
                "tmcra_attention_context_count": len(state_cells),
            }
            if best_key and best_score >= 0.42 and best_key in self._slot_state:
                previous = self._slot_state[best_key]
                cell.previous_value = previous.value
                cell.recurrence_count = max(cell.recurrence_count, previous.recurrence_count + 1)
                if not cell.subject and previous.subject:
                    cell.subject = previous.subject
                if previous.slot_key and best_score >= 0.62:
                    cell.slot_key = previous.slot_key
                cell.confidence = min(0.98, cell.confidence + min(0.08, best_score * 0.08))
            output.append(cell)
        return output, {
            "layer": f"X{layer_index}.attn",
            "name": "tmcra_cell_multi_head_attention",
            "input_count": len(cells),
            "output_count": len(output),
            "state_context_count": len(state_cells),
            "attended_edges": attended_edges,
        }

    def _residual_norm_layer(
        self,
        cells: Sequence[TMCRACell],
        *,
        layer_index: int,
    ) -> tuple[List[TMCRACell], Dict[str, Any]]:
        if not cells:
            return [], {
                "layer": f"X{layer_index}.norm",
                "name": "tmcra_residual_layer_norm",
                "input_count": 0,
                "output_count": 0,
            }
        mean_conf = sum(cell.confidence for cell in cells) / len(cells)
        mean_sal = sum(cell.salience for cell in cells) / len(cells)
        output: List[TMCRACell] = []
        for cell in cells:
            attention_score = float(cell.metadata.get("tmcra_attention_best_score", 0.0) or 0.0)
            # Residual path: preserve original L0 confidence/salience, then add
            # a bounded attention-derived correction and normalize toward the
            # turn mean so one noisy cell does not dominate the write budget.
            cell.confidence = min(0.98, max(0.52, (cell.confidence * 0.82) + (mean_conf * 0.12) + (attention_score * 0.06)))
            cell.salience = min(0.98, max(0.52, (cell.salience * 0.84) + (mean_sal * 0.10) + _category_weight(cell.category)))
            cell.metadata = {
                **cell.metadata,
                "tmcra_layers": [*list(cell.metadata.get("tmcra_layers", [])), f"X{layer_index}.norm"],
                "tmcra_residual_norm_mean_confidence": round(mean_conf, 6),
                "tmcra_residual_norm_mean_salience": round(mean_sal, 6),
            }
            output.append(cell)
        return output, {
            "layer": f"X{layer_index}.norm",
            "name": "tmcra_residual_layer_norm",
            "input_count": len(cells),
            "output_count": len(output),
        }

    def _feed_forward_layer(
        self,
        cells: Sequence[TMCRACell],
        *,
        layer_index: int,
    ) -> tuple[List[TMCRACell], Dict[str, Any]]:
        output: List[TMCRACell] = []
        slot_refined = 0
        for cell in cells:
            original_slot = cell.slot_key
            if not cell.subject:
                cell.subject = self._subject_hint(cell.value, cell.semantic_slot)
            subject_signature = _slug(cell.subject or cell.semantic_slot or cell.category)
            refined_slot = ".".join(
                part
                for part in [
                    "tmcra",
                    _slug(cell.category),
                    _slug(cell.semantic_slot),
                    subject_signature,
                ]
                if part
            )
            if refined_slot and refined_slot != cell.slot_key:
                cell.slot_key = refined_slot
                slot_refined += int(bool(original_slot))
            cell.value = _clip_text(cell.value, self.value_char_budget)
            cell.source_span = _clip_text(cell.source_span, self.source_span_char_budget)
            cell.metadata = {
                **cell.metadata,
                "tmcra_layers": [*list(cell.metadata.get("tmcra_layers", [])), f"X{layer_index}.ffn"],
                "tmcra_ffn_refined_slot": cell.slot_key,
                "tmcra_ffn_subject_signature": subject_signature,
                "event_signature": _slug(f"{cell.category} {cell.semantic_slot} {subject_signature} {' '.join(_tokens(cell.value)[:6])}"),
            }
            output.append(cell)
        return output, {
            "layer": f"X{layer_index}.ffn",
            "name": "tmcra_feed_forward_slot_refine",
            "input_count": len(cells),
            "output_count": len(output),
            "slot_refined": slot_refined,
        }

    def _state_context_cells(self) -> List[TMCRACell]:
        if self.state_attention_k <= 0:
            return []
        values = list(self._slot_state.values())
        return sorted(
            values,
            key=lambda cell: (cell.recurrence_count, cell.confidence + cell.salience),
            reverse=True,
        )[: self.state_attention_k]

    def _multi_head_scores(self, cell: TMCRACell, state_cells: Sequence[TMCRACell]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        cell_tokens = set(_tokens(" ".join([cell.value, cell.source_span, " ".join(cell.anchors)])))
        for previous in state_cells:
            previous_tokens = set(_tokens(" ".join([previous.value, previous.source_span, " ".join(previous.anchors)])))
            union = max(1, len(cell_tokens | previous_tokens))
            token_overlap = len(cell_tokens & previous_tokens) / union
            slot_score = 1.0 if previous.slot_key == cell.slot_key and previous.slot_key else 0.0
            category_score = 1.0 if previous.category == cell.category else 0.0
            semantic_score = 1.0 if previous.semantic_slot == cell.semantic_slot else 0.0
            subject_score = 1.0 if previous.subject and previous.subject == cell.subject else 0.0
            recurrence_score = min(1.0, previous.recurrence_count / 4.0)
            novelty_penalty = 0.18 if _safe_text(previous.value).lower() == _safe_text(cell.value).lower() else 0.0
            head_score = max(
                0.0,
                (0.30 * slot_score)
                + (0.18 * semantic_score)
                + (0.12 * category_score)
                + (0.15 * subject_score)
                + (0.20 * token_overlap)
                + (0.05 * recurrence_score)
                - novelty_penalty,
            )
            if head_score > 0:
                scores[previous.slot_key] = max(scores.get(previous.slot_key, 0.0), head_score)
        return scores


class TMCRASuspectAnchoredTransformerMemoryWriter(TMCRATransformerSemanticMemoryWriter):
    """Current-turn anchored transformer stack with a suspect memory buffer.

    The suspect layer quarantines tentative or abstract cells before they can
    become active recurrent state. Repeated compatible evidence can promote a
    suspect cell back into the active path.
    """

    architecture = "tmcra_suspect_anchored_transformer_v1"

    def __init__(
        self,
        base_writer: OpenAICompatSemanticMemoryWriter,
        *,
        max_cells: int = 2,
        value_char_budget: int = 120,
        source_span_char_budget: int = 180,
        transformer_layers: int = 2,
        state_attention_k: int = 16,
        suspect_threshold: float = 0.48,
        suspect_promote_count: int = 2,
    ) -> None:
        super().__init__(
            base_writer,
            max_cells=max_cells,
            value_char_budget=value_char_budget,
            source_span_char_budget=source_span_char_budget,
            transformer_layers=transformer_layers,
            state_attention_k=state_attention_k,
        )
        self.suspect_threshold = min(0.95, max(0.05, float(suspect_threshold or 0.48)))
        self.suspect_promote_count = max(2, int(suspect_promote_count or 2))
        self._suspect_state: Dict[str, TMCRACell] = {}
        self._suspect_counts: Dict[str, int] = {}

    def propose_public_turn(
        self,
        *,
        current_turn: str,
        previous_turn: str = "",
        next_turn: str = "",
        speaker: str = "",
        session_timestamp: str = "",
        sidecar_hints: Mapping[str, Any] | None = None,
        auxiliary_evidence_texts: Sequence[Any] | None = None,
        input_mode: str = "delta",
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.available():
            raise SemanticMemoryWriterError("TMCRA suspect anchored writer requires an available base writer")
        base_proposals, base_metadata = self.base_writer.propose_public_turn(
            current_turn=current_turn,
            previous_turn="",
            next_turn="",
            speaker=speaker,
            session_timestamp=session_timestamp,
            sidecar_hints={},
            auxiliary_evidence_texts=[],
            input_mode="delta",
        )
        turn_cells, turn_meta = self._turn_layer(
            base_proposals,
            current_turn=current_turn,
            speaker=speaker,
            session_timestamp=session_timestamp,
        )
        memory_cells, memory_meta = self._memory_layer(turn_cells, speaker=speaker)
        active_cells, suspect_cells, suspect_meta = self._suspect_uncertainty_layer(
            memory_cells,
            current_turn=current_turn,
        )
        transformer_cells, transformer_meta = self._anchored_transformer_cell_layers(
            active_cells,
            current_turn=current_turn,
        )
        compressed_cells, compression_meta = self._compression_layer(transformer_cells)
        recurrent_cells, recurrence_meta = self._recurrence_layer(compressed_cells, speaker=speaker)
        activated_cells, activation_meta = self._activation_layer(recurrent_cells)
        active_proposals = [cell.to_proposal() for cell in activated_cells[: self.max_proposals]]
        suspect_budget = max(0, self.max_proposals - len(active_proposals))
        suspect_proposals = [cell.to_proposal() for cell in suspect_cells[:suspect_budget]]
        proposals = [*active_proposals, *suspect_proposals]
        metadata = {
            **dict(base_metadata or {}),
            "architecture": self.architecture,
            "input_mode": "delta",
            "base_input_mode": dict(base_metadata or {}).get("input_mode", "delta"),
            "system_prompt_profile": "tmcra_suspect_anchored_delta_l0",
            "user_payload_keys": ["current_turn", "max_write_proposals", "speaker", "timestamp"],
            "tmcra_layers": [
                turn_meta,
                memory_meta,
                suspect_meta,
                *transformer_meta,
                compression_meta,
                recurrence_meta,
                activation_meta,
            ],
            "tmcra_stack_state_size": len(self._slot_state),
            "tmcra_suspect_buffer_size": len(self._suspect_state),
            "tmcra_transformer_layers": self.transformer_layers,
            "tmcra_state_attention_k": self.state_attention_k,
            "tmcra_suspect_threshold": self.suspect_threshold,
            "tmcra_suspect_promote_count": self.suspect_promote_count,
            "tmcra_stack_input_ignored": {
                "previous_turn": bool(_safe_text(previous_turn)),
                "next_turn": bool(_safe_text(next_turn)),
                "sidecar_hints": bool(sidecar_hints),
                "auxiliary_evidence_texts": bool(auxiliary_evidence_texts),
            },
            "proposal_count_after_stack": len(proposals),
            "active_proposal_count_after_stack": len(active_proposals),
            "suspect_proposal_count_after_stack": len(suspect_proposals),
        }
        return proposals, metadata

    def _suspect_uncertainty_layer(
        self,
        cells: Sequence[TMCRACell],
        *,
        current_turn: str,
    ) -> tuple[List[TMCRACell], List[TMCRACell], Dict[str, Any]]:
        active: List[TMCRACell] = []
        suspect: List[TMCRACell] = []
        buffered = 0
        promoted = 0
        for cell in cells:
            score, reasons = self._suspect_score(cell, current_turn=current_turn)
            signature = _slug(
                " ".join(
                    [
                        cell.category,
                        cell.semantic_slot,
                        cell.subject,
                        " ".join(_tokens(cell.value)[:8]),
                    ]
                )
            )
            support_count = self._suspect_counts.get(signature, 0) + 1
            should_buffer = score >= self.suspect_threshold and support_count < self.suspect_promote_count
            layer_metadata = {
                "tmcra_layers": [*list(cell.metadata.get("tmcra_layers", [])), "S"],
                "tmcra_suspect_score": round(score, 6),
                "tmcra_suspect_reasons": reasons,
                "tmcra_suspect_signature": signature,
                "tmcra_suspect_support_count": support_count,
                "tmcra_suspect_threshold": self.suspect_threshold,
            }
            if should_buffer:
                self._suspect_state[signature] = cell
                self._suspect_counts[signature] = support_count
                cell.confidence = min(cell.confidence, 0.56)
                cell.salience = min(cell.salience, 0.58)
                cell.metadata = {
                    **cell.metadata,
                    **layer_metadata,
                    "state": "suspect",
                    "relation": "suspect_memory",
                    "write_decision": "suspect_buffer",
                    "memory_gate_decision": "suspect_buffer",
                    "tmcra_suspect_buffered": True,
                    "suspicion_reason": ",".join(reasons) or "low_stability",
                    "source_layer": self.architecture,
                }
                suspect.append(cell)
                buffered += 1
                continue
            if signature in self._suspect_state and support_count >= self.suspect_promote_count:
                previous = self._suspect_state.pop(signature)
                cell.previous_value = previous.value
                cell.recurrence_count = max(cell.recurrence_count, support_count)
                cell.confidence = min(0.9, max(cell.confidence, 0.74))
                cell.salience = min(0.92, max(cell.salience, 0.76))
                self._suspect_counts[signature] = support_count
                promoted += 1
                layer_metadata["tmcra_suspect_promoted"] = True
            else:
                self._suspect_counts[signature] = support_count
                layer_metadata["tmcra_suspect_promoted"] = False
            cell.metadata = {
                **cell.metadata,
                **layer_metadata,
                "tmcra_suspect_buffered": False,
            }
            active.append(cell)
        if not active and len(suspect) > 1:
            release_index, release_cell = min(
                enumerate(suspect),
                key=lambda item: float(item[1].metadata.get("tmcra_suspect_score", 1.0) or 1.0),
            )
            suspect.pop(release_index)
            release_cell.confidence = max(release_cell.confidence, 0.68)
            release_cell.salience = max(release_cell.salience, 0.72)
            release_cell.metadata = {
                **release_cell.metadata,
                "state": "active",
                "relation": f"{release_cell.category}_memory",
                "write_decision": "active_fallback",
                "memory_gate_decision": "active_fallback",
                "tmcra_suspect_buffered": False,
                "tmcra_suspect_release_reason": "avoid_all_suspect_turn",
            }
            active.append(release_cell)
        return active, suspect, {
            "layer": "S",
            "name": "suspect_uncertainty_buffer",
            "input_count": len(cells),
            "output_count": len(active),
            "suspect_output_count": len(suspect),
            "buffered": buffered,
            "promoted": promoted,
            "buffer_size": len(self._suspect_state),
            "threshold": self.suspect_threshold,
        }

    def _suspect_score(self, cell: TMCRACell, *, current_turn: str) -> tuple[float, List[str]]:
        del current_turn
        text = " ".join([cell.value, cell.source_span])
        cell_tokens = _token_set(cell.value, cell.source_span)
        reasons: List[str] = []
        score = 0.0
        if _UNCERTAINTY_CUE_RE.search(text):
            score += 0.24
            reasons.append("uncertainty_cue")
        abstract_ratio = _abstract_ratio(cell_tokens)
        if abstract_ratio >= 0.28:
            score += min(0.14, abstract_ratio * 0.6)
            reasons.append("abstract_high_level_claim")
        source_token_count = len(_token_set(cell.source_span))
        if source_token_count <= 4:
            score += 0.12
            reasons.append("short_source_span")
        if cell.category in {"profile", "preference", "goal"} and _UNCERTAINTY_CUE_RE.search(text):
            score += 0.10
            reasons.append("tentative_stable_category")
        if cell.semantic_slot in {"preference", "identity_view", "exploration_topic"} and not cell.subject:
            score += 0.08
            reasons.append("weak_subject_anchor")
        if cell.slot_key not in self._slot_state and cell.category in {"profile", "preference", "goal"}:
            score += 0.04
            reasons.append("first_observation")
        if cell.category in {"time", "event", "constraint"}:
            score = max(0.0, score - 0.12)
        return min(1.0, score), reasons

    def _anchored_transformer_cell_layers(
        self,
        cells: Sequence[TMCRACell],
        *,
        current_turn: str,
    ) -> tuple[List[TMCRACell], List[Dict[str, Any]]]:
        active_cells = list(cells)
        metas: List[Dict[str, Any]] = []
        current_tokens = _token_set(current_turn)
        for layer_index in range(self.transformer_layers):
            attended, attention_meta = self._anchored_cell_attention_layer(
                active_cells,
                layer_index=layer_index,
                current_tokens=current_tokens,
            )
            normalized, norm_meta = self._residual_norm_layer(attended, layer_index=layer_index)
            active_cells, ff_meta = self._feed_forward_layer(normalized, layer_index=layer_index)
            metas.extend([attention_meta, norm_meta, ff_meta])
        return active_cells, metas

    def _anchored_cell_attention_layer(
        self,
        cells: Sequence[TMCRACell],
        *,
        layer_index: int,
        current_tokens: set[str],
    ) -> tuple[List[TMCRACell], Dict[str, Any]]:
        state_cells = self._state_context_cells()
        output: List[TMCRACell] = []
        attended_edges = 0
        gated_out_edges = 0
        for cell in cells:
            head_scores, gate_reasons = self._anchored_multi_head_scores(cell, state_cells, current_tokens)
            best_key = ""
            best_score = 0.0
            if head_scores:
                best_key, best_score = max(head_scores.items(), key=lambda item: item[1])
            if best_key:
                attended_edges += 1
            else:
                gated_out_edges += len(gate_reasons)
            cell.metadata = {
                **cell.metadata,
                "tmcra_layers": [*list(cell.metadata.get("tmcra_layers", [])), f"X{layer_index}.attn"],
                "tmcra_attention_heads": {key: round(value, 6) for key, value in head_scores.items()},
                "tmcra_attention_best_slot": best_key,
                "tmcra_attention_best_score": round(best_score, 6),
                "tmcra_attention_context_count": len(state_cells),
                "tmcra_current_anchor_tokens": len(current_tokens),
                "tmcra_attention_gated_out": gate_reasons[:8],
            }
            if best_key and best_score >= 0.50 and best_key in self._slot_state:
                previous = self._slot_state[best_key]
                cell.previous_value = previous.value
                cell.recurrence_count = max(cell.recurrence_count, previous.recurrence_count + 1)
                if not cell.subject and previous.subject:
                    cell.subject = previous.subject
                if previous.slot_key and best_score >= 0.70:
                    cell.slot_key = previous.slot_key
                cell.confidence = min(0.96, cell.confidence + min(0.05, best_score * 0.05))
            output.append(cell)
        return output, {
            "layer": f"X{layer_index}.attn",
            "name": "tmcra_current_anchored_multi_head_attention",
            "input_count": len(cells),
            "output_count": len(output),
            "state_context_count": len(state_cells),
            "attended_edges": attended_edges,
            "gated_out_edges": gated_out_edges,
        }

    def _anchored_multi_head_scores(
        self,
        cell: TMCRACell,
        state_cells: Sequence[TMCRACell],
        current_tokens: set[str],
    ) -> tuple[Dict[str, float], List[str]]:
        scores: Dict[str, float] = {}
        gate_reasons: List[str] = []
        cell_tokens = _token_set(cell.value, cell.source_span, " ".join(cell.anchors))
        for previous in state_cells:
            previous_tokens = _token_set(previous.value, previous.source_span, " ".join(previous.anchors))
            previous_current_overlap = _overlap_ratio(previous_tokens, current_tokens)
            cell_previous_overlap = _overlap_ratio(cell_tokens, previous_tokens)
            slot_score = 1.0 if previous.slot_key == cell.slot_key and previous.slot_key else 0.0
            semantic_score = 1.0 if previous.semantic_slot == cell.semantic_slot else 0.0
            category_score = 1.0 if previous.category == cell.category else 0.0
            subject_score = 1.0 if previous.subject and previous.subject == cell.subject else 0.0
            recurrence_score = min(1.0, previous.recurrence_count / 4.0)
            generic_penalty = min(0.22, _abstract_ratio(previous_tokens) * 0.5)
            novelty_penalty = 0.18 if _safe_text(previous.value).lower() == _safe_text(cell.value).lower() else 0.0
            anchor_ok = previous_current_overlap >= 0.08 or slot_score > 0 or subject_score > 0
            if not anchor_ok:
                gate_reasons.append(previous.slot_key)
                continue
            head_score = max(
                0.0,
                (0.28 * previous_current_overlap)
                + (0.22 * slot_score)
                + (0.16 * semantic_score)
                + (0.10 * category_score)
                + (0.10 * subject_score)
                + (0.10 * cell_previous_overlap)
                + (0.04 * recurrence_score)
                - generic_penalty
                - novelty_penalty,
            )
            if head_score > 0:
                scores[previous.slot_key] = max(scores.get(previous.slot_key, 0.0), head_score)
        return scores, gate_reasons


__all__ = [
    "MultiLayerTMCRASemanticMemoryWriter",
    "TMCRACell",
    "TMCRASuspectAnchoredTransformerMemoryWriter",
    "TMCRATransformerSemanticMemoryWriter",
]
