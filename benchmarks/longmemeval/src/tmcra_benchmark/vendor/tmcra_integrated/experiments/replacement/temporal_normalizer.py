from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta
from typing import Any

from experiments.replacement.temporal_modeling_types import TemporalFrame, clean_text


_MONTH_NAMES = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

_CJK_MONTH_RE = re.compile(r"(?P<month>1[0-2]|0?[1-9])\s*月")


def _parse_datetime(value: Any) -> datetime | None:
    text = clean_text(value)
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _date_text(value: datetime) -> str:
    return value.strftime("%Y-%m-%d")


def _month_range(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}"


class TemporalNormalizer:
    """Resolve a TemporalFrame against session time without mutating memory."""

    version = "temporal_normalizer_v1"

    def normalize(self, frame: TemporalFrame, *, session_timestamp: str = "") -> TemporalFrame:
        resolved = TemporalFrame.from_mapping(frame.to_dict())
        session_time = _parse_datetime(session_timestamp or resolved.utterance_time) or datetime.now()
        if not resolved.utterance_time:
            resolved.utterance_time = _date_text(session_time)
        if resolved.resolved_start or resolved.resolved_end:
            self._fill_event_state_time(resolved)
            return resolved
        start, end, granularity = self._resolve_expression(
            resolved.time_expression,
            session_time=session_time,
            fallback_granularity=resolved.granularity,
        )
        if start:
            resolved.resolved_start = start
            resolved.resolved_end = end or start
            resolved.event_time = resolved.event_time or start
            resolved.granularity = granularity or resolved.granularity
        elif resolved.state_operation in {"create", "update_current", "supersede"}:
            resolved.resolved_start = _date_text(session_time)
            resolved.resolved_end = _date_text(session_time)
            resolved.event_time = resolved.event_time or resolved.resolved_start
            if resolved.granularity == "none":
                resolved.granularity = "day"
        self._fill_event_state_time(resolved)
        resolved.metadata = {
            **dict(resolved.metadata or {}),
            "temporal_normalizer": self.version,
            "session_timestamp": clean_text(session_timestamp),
        }
        return resolved

    def _fill_event_state_time(self, frame: TemporalFrame) -> None:
        if frame.state_operation in {"create", "update_current", "supersede"}:
            frame.state_valid_from = frame.state_valid_from or frame.event_time or frame.resolved_start
        if not frame.event_time and frame.resolved_start:
            frame.event_time = frame.resolved_start

    def _resolve_expression(
        self,
        expression: str,
        *,
        session_time: datetime,
        fallback_granularity: str = "none",
    ) -> tuple[str, str, str]:
        text = clean_text(expression).lower()
        if not text:
            return "", "", fallback_granularity
        if text in {"today", "今天"} or "今天" in text:
            day = _date_text(session_time)
            return day, day, "day"
        if text in {"yesterday", "昨天"} or "昨天" in text:
            day = _date_text(session_time - timedelta(days=1))
            return day, day, "day"
        if "last week" in text or "上周" in text:
            start = session_time - timedelta(days=session_time.weekday() + 7)
            end = start + timedelta(days=6)
            return _date_text(start), _date_text(end), "week"
        if "this week" in text or "本周" in text:
            start = session_time - timedelta(days=session_time.weekday())
            end = start + timedelta(days=6)
            return _date_text(start), _date_text(end), "week"
        if "last month" in text or "上个月" in text:
            month = session_time.month - 1
            year = session_time.year
            if month <= 0:
                month = 12
                year -= 1
            start, end = _month_range(year, month)
            return start, end, "month"
        cjk_month = _CJK_MONTH_RE.search(text)
        if cjk_month:
            month = int(cjk_month.group("month"))
            start, end = _month_range(session_time.year, month)
            return start, end, "month"
        for name, month in _MONTH_NAMES.items():
            if name in text:
                start, end = _month_range(session_time.year, month)
                return start, end, "month"
        parsed = _parse_datetime(text)
        if parsed is not None:
            day = _date_text(parsed)
            return day, day, "day"
        return "", "", fallback_granularity
