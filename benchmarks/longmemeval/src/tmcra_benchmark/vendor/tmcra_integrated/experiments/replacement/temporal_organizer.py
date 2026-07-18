from __future__ import annotations

import re
from typing import Any, Mapping

from experiments.replacement.temporal_modeling_types import TemporalFrame, clean_text, normalize_key
from experiments.replacement.temporal_normalizer import TemporalNormalizer


_TIME_MARKERS = (
    "yesterday",
    "today",
    "last week",
    "this week",
    "last month",
    "may",
    "june",
    "july",
    "昨天",
    "今天",
    "上周",
    "本周",
    "上个月",
    "月",
)

_CURRENT_MARKERS = ("now", "currently", "current", "right now", "现在", "目前", "当前")
_PREVIOUS_MARKERS = ("previous", "before", "used to", "last time", "earlier", "之前", "以前", "上次", "原来")
_TIMELINE_MARKERS = ("timeline", "over time", "how did", "history", "按时间", "时间线", "怎么变", "变化")
_COMPARE_MARKERS = ("before and after", "compare", "versus", "vs", "对比", "前后")


class TemporalOrganizer:
    """Create a temporal frame for a turn before writer/retrieval logic.

    This class is intentionally model-ready: callers may pass a teacher/model
    payload later and convert it with TemporalFrame.from_mapping. The fallback
    implementation is conservative and only supplies a valid baseline frame.
    """

    version = "temporal_organizer_v1"

    def __init__(self, *, normalizer: TemporalNormalizer | None = None) -> None:
        self.normalizer = normalizer or TemporalNormalizer()

    def organize_turn(
        self,
        *,
        current_turn: str,
        session_timestamp: str = "",
        previous_turn: str = "",
        speaker: str = "user",
        model_frame: Mapping[str, Any] | None = None,
    ) -> TemporalFrame:
        if model_frame:
            frame = TemporalFrame.from_mapping(model_frame)
            return self.normalizer.normalize(frame, session_timestamp=session_timestamp)
        text = clean_text(current_turn)
        lowered = text.lower()
        subject, old_state, new_state = self._extract_subject_state(text)
        time_expression = self._extract_time_expression(text)
        has_time = bool(time_expression)
        state_change = bool(new_state and (old_state or self._contains_any(lowered, ("switched", "changed", "换成", "改成", "开始"))))
        if state_change:
            temporal_intent = "state_evolution"
            state_operation = "supersede" if old_state else "update_current"
        elif subject and self._contains_any(lowered, _CURRENT_MARKERS):
            temporal_intent = "current_state"
            state_operation = "update_current"
        elif has_time:
            temporal_intent = "relative_time" if self._is_relative_time(time_expression) else "explicit_time"
            state_operation = "update_current" if new_state else "create"
        else:
            temporal_intent = "non_temporal"
            state_operation = "none"
        frame = TemporalFrame(
            temporal_intent=temporal_intent,
            time_expression=time_expression,
            anchor_type="relative" if self._is_relative_time(time_expression) else ("absolute" if has_time else "none"),
            anchor_base="session_time" if self._is_relative_time(time_expression) else "",
            granularity="day" if has_time else "none",
            utterance_time=session_timestamp,
            state_operation=state_operation,
            subject=subject,
            subject_key=normalize_key(subject) if subject else "",
            old_state=old_state,
            new_state=new_state,
            should_create_timeline_edge=state_operation in {"update_current", "supersede"},
            confidence=0.62 if temporal_intent != "non_temporal" else 0.25,
            evidence_span=text,
            metadata={
                "source": self.version,
                "speaker": clean_text(speaker),
                "previous_turn_available": bool(clean_text(previous_turn)),
                "fallback_mode": "heuristic_bootstrap",
            },
        )
        return self.normalizer.normalize(frame, session_timestamp=session_timestamp)

    def _extract_time_expression(self, text: str) -> str:
        lowered = clean_text(text).lower()
        for marker in _TIME_MARKERS:
            if marker in lowered:
                if marker == "月":
                    match = re.search(r"(?:1[0-2]|0?[1-9])\s*月", text)
                    return match.group(0) if match else ""
                return marker
        month = re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            lowered,
        )
        return month.group(1) if month else ""

    def _extract_subject_state(self, text: str) -> tuple[str, str, str]:
        compact = clean_text(text)
        lowered = compact.lower()
        subject = self._infer_subject(lowered)
        old_state = ""
        new_state = ""
        switch_match = re.search(r"\bfrom\s+(?P<old>[^,.!?]+?)\s+to\s+(?P<new>[^,.!?]+)", compact, flags=re.IGNORECASE)
        if switch_match:
            old_state = clean_text(switch_match.group("old"))
            new_state = clean_text(switch_match.group("new"))
        cjk_switch = re.search(r"从(?P<old>[^，。！？]+?)换成(?:了)?(?P<new>[^，。！？]+)", compact)
        if cjk_switch:
            old_state = clean_text(cjk_switch.group("old"))
            new_state = clean_text(cjk_switch.group("new"))
        if not new_state:
            use_match = re.search(r"\b(?:use|using|currently use|now use)\s+(?P<new>[A-Za-z0-9][^,.!?]*)", compact, flags=re.IGNORECASE)
            if use_match:
                new_state = clean_text(use_match.group("new"))
        if not new_state:
            cjk_now = re.search(r"(?:现在|目前|当前).*?(?:用|使用|换成了?|改成了?)(?P<new>[^，。！？]+)", compact)
            if cjk_now:
                new_state = clean_text(cjk_now.group("new"))
        if not old_state:
            used_to = re.search(r"\bused to use\s+(?P<old>[^,.!?]+)", compact, flags=re.IGNORECASE)
            if used_to:
                old_state = clean_text(used_to.group("old"))
        return subject, old_state, new_state

    def _infer_subject(self, lowered: str) -> str:
        if any(token in lowered for token in ("phone", "iphone", "android", "手机")):
            return "phone"
        if any(token in lowered for token in ("project", "api", "benchmark", "论文", "项目")):
            return "project_focus"
        if any(token in lowered for token in ("diet", "meal", "food", "ingredient", "饮食", "食物", "食材")):
            return "food"
        if any(token in lowered for token in ("plan", "schedule", "calendar", "计划", "日程")):
            return "plan"
        return ""

    def _is_relative_time(self, expression: str) -> bool:
        text = clean_text(expression).lower()
        return text in {"yesterday", "today", "last week", "this week", "last month", "昨天", "今天", "上周", "本周", "上个月"}

    def _contains_any(self, text: str, markers: tuple[str, ...]) -> bool:
        return any(marker in text for marker in markers)


def infer_query_temporal_intent(query: str) -> tuple[str, str, dict[str, Any]]:
    text = clean_text(query)
    lowered = text.lower()
    if any(marker in lowered for marker in _TIMELINE_MARKERS):
        return "timeline", "query_timeline", {"requires_ordered_chain": True}
    if any(marker in lowered for marker in _COMPARE_MARKERS):
        return "compare_before_after", "query_compare", {"requires_comparison": True}
    if any(marker in lowered for marker in _PREVIOUS_MARKERS):
        return "previous_state", "query_previous", {"prefer_previous_state": True}
    if any(marker in lowered for marker in _CURRENT_MARKERS):
        return "current_state", "query_current", {"prefer_current_state": True}
    if any(marker in lowered for marker in _TIME_MARKERS):
        return "explicit_time", "query_timeline", {"requires_ordered_chain": False}
    return "non_temporal", "none", {}
