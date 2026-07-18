from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


TEMPORAL_INTENTS = {
    "non_temporal",
    "explicit_time",
    "relative_time",
    "current_state",
    "previous_state",
    "timeline",
    "compare_before_after",
    "state_evolution",
}

ANCHOR_TYPES = {"none", "absolute", "relative", "session_time", "event_time", "turn_order"}

GRANULARITIES = {
    "none",
    "turn",
    "day",
    "week",
    "month",
    "year",
    "range",
    "relative_day_reference",
}

STATE_OPERATIONS = {
    "none",
    "create",
    "update_current",
    "supersede",
    "query_current",
    "query_previous",
    "query_timeline",
    "query_compare",
}


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_key(value: Any, *, fallback: str = "general") -> str:
    text = clean_text(value).lower()
    if not text:
        return fallback
    allowed = []
    previous_sep = False
    for char in text:
        if char.isalnum() or "\u4e00" <= char <= "\u9fff":
            allowed.append(char)
            previous_sep = False
        elif not previous_sep:
            allowed.append("_")
            previous_sep = True
    key = "".join(allowed).strip("_")
    return key or fallback


def coerce_choice(value: Any, allowed: set[str], default: str) -> str:
    text = clean_text(value).lower().replace("-", "_").replace(" ", "_")
    return text if text in allowed else default


@dataclass(slots=True)
class TemporalFrame:
    temporal_intent: str = "non_temporal"
    time_expression: str = ""
    anchor_type: str = "none"
    anchor_base: str = ""
    resolved_start: str = ""
    resolved_end: str = ""
    granularity: str = "none"
    utterance_time: str = ""
    event_time: str = ""
    state_valid_from: str = ""
    state_valid_to: str = ""
    state_operation: str = "none"
    subject: str = ""
    subject_key: str = ""
    old_state: str = ""
    new_state: str = ""
    should_create_timeline_edge: bool = False
    confidence: float = 0.0
    evidence_span: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.temporal_intent = coerce_choice(self.temporal_intent, TEMPORAL_INTENTS, "non_temporal")
        self.anchor_type = coerce_choice(self.anchor_type, ANCHOR_TYPES, "none")
        self.granularity = coerce_choice(self.granularity, GRANULARITIES, "none")
        self.state_operation = coerce_choice(self.state_operation, STATE_OPERATIONS, "none")
        if not self.subject_key and self.subject:
            self.subject_key = normalize_key(self.subject)
        self.confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "TemporalFrame":
        data = dict(payload or {})
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass(slots=True)
class TimelineState:
    state_id: str
    subject_key: str
    subject_label: str = ""
    state_value: str = ""
    state_type: str = "profile"
    valid_from: str = ""
    valid_to: str = ""
    event_time: str = ""
    utterance_time: str = ""
    is_current: bool = True
    supersedes_state_id: str = ""
    source_event_id: str = ""
    source_turn_id: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TimelineEdge:
    edge_id: str
    subject_key: str
    from_state_id: str = ""
    to_state_id: str = ""
    edge_type: str = "updates"
    transition_time: str = ""
    transition_reason: str = ""
    source_event_id: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TemporalQueryPlan:
    query_temporal_intent: str = "non_temporal"
    target_subject: str = ""
    target_subject_key: str = ""
    time_filter: dict[str, Any] = field(default_factory=dict)
    timeline_operation: str = "none"
    prefer_current_state: bool = False
    prefer_previous_state: bool = False
    requires_ordered_chain: bool = False
    requires_comparison: bool = False
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.query_temporal_intent = coerce_choice(self.query_temporal_intent, TEMPORAL_INTENTS, "non_temporal")
        if not self.target_subject_key and self.target_subject:
            self.target_subject_key = normalize_key(self.target_subject)
        self.confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TimelineEvidencePack:
    mode: str
    subject: str = ""
    subject_key: str = ""
    timeline: list[dict[str, Any]] = field(default_factory=list)
    selected_evidence: dict[str, Any] = field(default_factory=dict)
    evidence_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
