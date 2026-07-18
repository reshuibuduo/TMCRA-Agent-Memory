from __future__ import annotations

from typing import Any

from experiments.replacement.temporal_modeling_types import TemporalQueryPlan, TimelineEvidencePack, TimelineState, clean_text
from experiments.replacement.timeline_state_layer import TimelineStateLayer


class TimelineEvidencePackBuilder:
    """Build structured temporal evidence for the answer model."""

    version = "timeline_evidence_pack_v1"

    def build(self, *, plan: TemporalQueryPlan, timeline_layer: TimelineStateLayer) -> TimelineEvidencePack:
        subject_key = clean_text(plan.target_subject_key)
        states = timeline_layer.timeline(subject_key) if subject_key else []
        selected: TimelineState | None = None
        if plan.timeline_operation == "query_current":
            selected = timeline_layer.current_state(subject_key)
        elif plan.timeline_operation == "query_previous":
            selected = timeline_layer.previous_state(subject_key)
        elif plan.timeline_operation == "query_compare":
            selected = timeline_layer.current_state(subject_key)
        elif states:
            selected = states[-1]
        timeline = [self._state_payload(state) for state in states]
        evidence = self._selected_payload(selected, plan=plan)
        confidence = selected.confidence if selected is not None else 0.0
        return TimelineEvidencePack(
            mode=plan.timeline_operation,
            subject=plan.target_subject,
            subject_key=subject_key,
            timeline=timeline,
            selected_evidence=evidence,
            evidence_confidence=confidence,
        )

    def _state_payload(self, state: TimelineState) -> dict[str, Any]:
        return {
            "time": clean_text(state.valid_from or state.event_time or state.utterance_time),
            "state": clean_text(state.state_value),
            "state_id": state.state_id,
            "source_event_id": state.source_event_id,
            "source_turn_id": state.source_turn_id,
            "source_text": clean_text(state.metadata.get("source_text", "")),
            "is_current": bool(state.is_current),
            "valid_to": clean_text(state.valid_to),
        }

    def _selected_payload(self, state: TimelineState | None, *, plan: TemporalQueryPlan) -> dict[str, Any]:
        if state is None:
            return {
                "answer_value": "",
                "state_id": "",
                "source_event_id": "",
                "reason": "no_timeline_state_found",
            }
        reason = {
            "query_current": "selected current state",
            "query_previous": "selected previous state before current state",
            "query_timeline": "selected latest state in ordered timeline",
            "query_compare": "selected current state for comparison",
        }.get(plan.timeline_operation, "selected timeline state")
        return {
            "answer_value": clean_text(state.state_value),
            "state_id": state.state_id,
            "source_event_id": state.source_event_id,
            "source_turn_id": state.source_turn_id,
            "reason": reason,
        }
