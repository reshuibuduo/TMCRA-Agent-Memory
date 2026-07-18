from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from experiments.replacement.temporal_modeling_types import (
    TemporalFrame,
    TimelineEdge,
    TimelineState,
    clean_text,
    normalize_key,
)


class TimelineStateLayer:
    """In-memory timeline state layer for temporal smoke tests and runtime wiring."""

    version = "timeline_state_layer_v1"

    def __init__(self) -> None:
        self._states: Dict[str, List[TimelineState]] = defaultdict(list)
        self._edges: List[TimelineEdge] = []
        self._counter = 0

    def apply_frame(
        self,
        frame: TemporalFrame,
        *,
        source_event_id: str = "",
        source_turn_id: str = "",
        state_type: str = "profile",
    ) -> TimelineState | None:
        subject_key = clean_text(frame.subject_key) or normalize_key(frame.subject)
        if not subject_key or not clean_text(frame.new_state):
            return None
        valid_from = clean_text(frame.state_valid_from or frame.event_time or frame.resolved_start or frame.utterance_time)
        previous = self.current_state(subject_key)
        if frame.old_state and previous is None:
            previous = self._create_state(
                subject_key=subject_key,
                subject_label=frame.subject,
                value=frame.old_state,
                valid_from="",
                valid_to=valid_from,
                event_time="",
                utterance_time=frame.utterance_time,
                is_current=False,
                source_event_id=source_event_id,
                source_turn_id=source_turn_id,
                state_type=state_type,
                confidence=frame.confidence,
                metadata={
                    "synthetic_previous_from_frame": True,
                    "temporal_intent": frame.temporal_intent,
                    "state_operation": frame.state_operation,
                    "time_expression": frame.time_expression,
                    "source_text": clean_text(frame.evidence_span),
                    "timeline_state_layer": self.version,
                },
            )
            self._states[subject_key].append(previous)
        if previous is not None and previous.is_current:
            previous.is_current = False
            previous.valid_to = previous.valid_to or valid_from
        state = self._create_state(
            subject_key=subject_key,
            subject_label=frame.subject,
            value=frame.new_state,
            valid_from=valid_from,
            valid_to=clean_text(frame.state_valid_to),
            event_time=clean_text(frame.event_time),
            utterance_time=clean_text(frame.utterance_time),
            is_current=True,
            supersedes_state_id=previous.state_id if previous is not None else "",
            source_event_id=source_event_id,
            source_turn_id=source_turn_id,
            state_type=state_type,
            confidence=frame.confidence,
            metadata={
                "temporal_intent": frame.temporal_intent,
                "state_operation": frame.state_operation,
                "time_expression": frame.time_expression,
                "source_text": clean_text(frame.evidence_span),
                "timeline_state_layer": self.version,
            },
        )
        self._states[subject_key].append(state)
        if previous is not None:
            self._edges.append(
                TimelineEdge(
                    edge_id=self._next_id("edge"),
                    subject_key=subject_key,
                    from_state_id=previous.state_id,
                    to_state_id=state.state_id,
                    edge_type="supersedes" if frame.state_operation == "supersede" else "updates",
                    transition_time=valid_from,
                    transition_reason=frame.evidence_span,
                    source_event_id=source_event_id,
                    confidence=frame.confidence,
                )
            )
        return state

    def current_state(self, subject_key: str) -> TimelineState | None:
        states = self._states.get(clean_text(subject_key), [])
        for state in reversed(states):
            if state.is_current:
                return state
        return states[-1] if states else None

    def previous_state(self, subject_key: str) -> TimelineState | None:
        states = self.timeline(subject_key)
        if len(states) < 2:
            return None
        current = self.current_state(subject_key)
        for state in reversed(states):
            if current is None or state.state_id != current.state_id:
                return state
        return None

    def timeline(self, subject_key: str) -> List[TimelineState]:
        states = list(self._states.get(clean_text(subject_key), []))
        return sorted(states, key=lambda item: (clean_text(item.valid_from), clean_text(item.event_time), item.state_id))

    def edges(self, subject_key: str = "") -> List[TimelineEdge]:
        key = clean_text(subject_key)
        if not key:
            return list(self._edges)
        return [edge for edge in self._edges if edge.subject_key == key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "subjects": {
                subject: [state.to_dict() for state in states]
                for subject, states in sorted(self._states.items())
            },
            "edges": [edge.to_dict() for edge in self._edges],
        }

    def _create_state(
        self,
        *,
        subject_key: str,
        subject_label: str,
        value: str,
        valid_from: str,
        valid_to: str,
        event_time: str,
        utterance_time: str,
        is_current: bool,
        source_event_id: str,
        source_turn_id: str,
        state_type: str,
        confidence: float,
        supersedes_state_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TimelineState:
        return TimelineState(
            state_id=self._next_id("state"),
            subject_key=subject_key,
            subject_label=clean_text(subject_label),
            state_value=clean_text(value),
            state_type=clean_text(state_type) or "profile",
            valid_from=valid_from,
            valid_to=valid_to,
            event_time=event_time,
            utterance_time=utterance_time,
            is_current=bool(is_current),
            supersedes_state_id=supersedes_state_id,
            source_event_id=clean_text(source_event_id),
            source_turn_id=clean_text(source_turn_id),
            confidence=confidence,
            metadata=dict(metadata or {}),
        )

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:06d}"
