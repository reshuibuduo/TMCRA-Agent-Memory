from __future__ import annotations

from experiments.replacement.temporal_modeling_types import TemporalQueryPlan, clean_text, normalize_key
from experiments.replacement.temporal_organizer import infer_query_temporal_intent


class TemporalQueryPlanner:
    """Plan temporal retrieval before graph/rerank/answer stages."""

    version = "temporal_query_planner_v1"

    def plan(self, query: str, *, target_subject: str = "", session_timestamp: str = "") -> TemporalQueryPlan:
        intent, operation, flags = infer_query_temporal_intent(query)
        subject = clean_text(target_subject) or self._infer_subject(query)
        return TemporalQueryPlan(
            query_temporal_intent=intent,
            target_subject=subject,
            target_subject_key=normalize_key(subject) if subject else "",
            timeline_operation=operation,
            prefer_current_state=bool(flags.get("prefer_current_state", False)),
            prefer_previous_state=bool(flags.get("prefer_previous_state", False)),
            requires_ordered_chain=bool(flags.get("requires_ordered_chain", False)),
            requires_comparison=bool(flags.get("requires_comparison", False)),
            confidence=0.66 if intent != "non_temporal" else 0.25,
            metadata={
                "source": self.version,
                "session_timestamp": clean_text(session_timestamp),
            },
        )

    def _infer_subject(self, query: str) -> str:
        lowered = clean_text(query).lower()
        if any(token in lowered for token in ("phone", "iphone", "android", "手机")):
            return "phone"
        if any(token in lowered for token in ("project", "api", "benchmark", "论文", "项目")):
            return "project_focus"
        if any(token in lowered for token in ("diet", "meal", "food", "ingredient", "饮食", "食物", "食材")):
            return "food"
        if any(token in lowered for token in ("plan", "schedule", "calendar", "计划", "日程")):
            return "plan"
        return ""
