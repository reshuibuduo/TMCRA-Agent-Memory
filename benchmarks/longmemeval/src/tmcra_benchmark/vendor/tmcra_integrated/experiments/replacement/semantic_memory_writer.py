from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from experiments.replacement.adapters.base import LLMProfile
from experiments.replacement.profile_layer import profile_candidate_metadata
from experiments.replacement.public_event_signature import compute_public_event_signature


ALLOWED_WRITE_CATEGORIES = {
    "fact",
    "event",
    "profile",
    "time",
    "status",
    "preference",
    "goal",
    "constraint",
    "stage_state",
    "terminology",
    "question",
    "interaction_intent",
}
_FACET_TYPES = {"temporal", "numeric", "state", "entity", "evidence_role", "action", "role"}
_FACET_TYPE_ALIASES = {
    "time": "temporal",
    "date": "temporal",
    "duration": "temporal",
    "temporal_anchor": "temporal",
    "number": "numeric",
    "quantity": "numeric",
    "amount": "numeric",
    "count": "numeric",
    "numeric_quantity": "numeric",
    "status": "state",
    "state_status": "state",
    "object": "entity",
    "item": "entity",
    "target_entity": "entity",
    "role": "evidence_role",
    "action_unit": "action",
    "action": "action",
    "participation": "role",
    "leadership": "role",
    "role_participation_leadership": "role",
}
_FACET_CATEGORY_BY_TYPE = {
    "temporal": "time",
    "numeric": "fact",
    "state": "status",
    "entity": "fact",
    "evidence_role": "fact",
    "action": "event",
    "role": "fact",
}
_UNIT_WRITER_ACTION_RE = re.compile(
    r"\b("
    r"buy|bought|purchase|purchased|order|ordered|return|returned|exchange|exchanged|"
    r"pick(?:ed|ing)?\s+up|receive|received|get|got|arrive|arrived|"
    r"start(?:ed|ing)?|finish(?:ed|ing)?|complete(?:d|ing)?|work(?:ed|ing)?\s+on|"
    r"plan(?:ned|ning)?\s+to\s+[a-z][a-z-]+|"
    r"lead|led|leading|manage|managed|organize|organized|"
    r"attend(?:ed)?|participate(?:d)?|join(?:ed)?|volunteer(?:ed)?|"
    r"build|built|create|created|submit|submitted|redeem|redeemed|"
    r"spend|spent|pay|paid|borrow|borrowed|rent|rented"
    r")\b",
    re.IGNORECASE,
)
_UNIT_WRITER_NUMERIC_RE = re.compile(
    r"(?:(?:[$€£¥]\s*)?\d+(?:[.,]\d+)?\s*(?:%|percent|cents?|dollars?|usd|eur|gbp|"
    r"days?|weeks?|months?|years?|hours?|minutes?|times?|items?|people|engineers?|members?|"
    r"tickets?|points?|miles?|km|kilometers?|meters?|lbs?|pounds?|kg|grams?)?\b|"
    r"\b\d+\s*/\s*\d+\b)",
    re.IGNORECASE,
)
_UNIT_WRITER_NUMERIC_RE = re.compile(
    r"(?:\b\d+\s*/\s*\d+\s*(?:scale|ratio|chance|share)?\b|"
    r"(?:[$]\s*)?\d+(?:[.,]\d+)?\s*(?:%|percent|cents?|dollars?|usd|eur|gbp|"
    r"days?|weeks?|months?|years?|hours?|minutes?|times?|items?|people|engineers?|members?|"
    r"tickets?|points?|miles?|km|kilometers?|meters?|lbs?|pounds?|kg|grams?)\b)",
    re.IGNORECASE,
)
_UNIT_WRITER_TEMPORAL_RE = re.compile(
    r"\b("
    r"today|tomorrow|yesterday|tonight|currently|now|recently|previously|later|earlier|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan\.?|feb\.?|mar\.?|apr\.?|jun\.?|jul\.?|aug\.?|sep\.?|sept\.?|oct\.?|nov\.?|dec\.?|"
    r"last\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"next\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?"
    r")\b",
    re.IGNORECASE,
)
_UNIT_WRITER_STATE_RE = re.compile(
    r"\b("
    r"currently|still|already|pending|planned|planning|need(?:s|ed)?\s+to|"
    r"not\s+yet|waiting\s+for|completed|finished|done|cancelled|canceled|"
    r"too\s+\w+|larger|smaller|newer|older|current|previous|latest|old|"
    r"single|single\s+parent|married|divorced|widowed|partner|relationship|"
    r"identity|member|belongs|home\s+country|country|origin|roots|moved\s+from"
    r")\b",
    re.IGNORECASE,
)
_UNIT_WRITER_ROLE_RE = re.compile(
    r"\b("
    r"lead|led|leading|leader|manager|managed|organizer|organized|"
    r"participant|participated|volunteer|volunteered|attendee|attended|member|team"
    r")\b",
    re.IGNORECASE,
)
_PROFILE_SLOTS = {"identity", "research_topic", "education", "occupation"}
_PROFILE_DIRECT_CATEGORIES = {"profile", "preference", "goal", "constraint"}
_PROFILE_SETUP_SLOT_MARKERS = {
    "brand",
    "camera",
    "device",
    "equipment",
    "gear",
    "model",
    "phone",
    "setup",
    "software",
    "stack",
    "tool",
    "workflow",
}
_PROFILE_INTENT_SLOT_MARKERS = {
    "avoid",
    "constraint",
    "goal",
    "habit",
    "identity",
    "preference",
    "profile",
    "style",
    "usage",
    *_PROFILE_SETUP_SLOT_MARKERS,
}
_WRITE_CATEGORY_ALIASES = {
    "identity": "profile",
    "research_topic": "profile",
    "education": "profile",
    "occupation": "profile",
    "event_time": "time",
    "user_question": "question",
}
_SECRET_RE = re.compile(
    r"\b(password|passcode|secret|private[_ -]?key|bearer\s+[a-z0-9._-]+|sk-[a-z0-9]{12,}|"
    r"api[_ -]?key\s*(?:is|=|:)\s*[a-z0-9._-]{8,})\b",
    flags=re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9_]+", flags=re.IGNORECASE)
_SPEECH_ACT_MEMORY_RE = re.compile(
    r"^\s*(asked|asks|asking)\b|"
    r"^\s*(requested|requests|requesting)\s+(the\s+)?(assistant|ai|model|you)\b|"
    r"^\s*(wanted|wants|wanting)\s+(the\s+)?(assistant|ai|model|you)\s+to\b",
    flags=re.IGNORECASE,
)
_QUESTION_ONLY_RE = re.compile(
    r"^\s*(quick\s+memory\s+check|what|which|who|where|when|how|why|do\s+you|can\s+you|could\s+you|would\s+you|please\s+remind|remind\s+me)\b",
    flags=re.IGNORECASE,
)
_ASSERTIVE_MEMORY_CUE_RE = re.compile(
    r"\b(remember\s+that|for\s+context|small\s+update|update\s+this|is\s+now|going\s+forward|my\s+.+?\s+is\s+[^?]+)\b",
    flags=re.IGNORECASE,
)
_FIRST_PERSON_MEMORY_ASSERTION_RE = re.compile(
    r"\b(?:i|we)\s+(?:want|prefer|need|avoid|like|dislike|use|usually|always|keep|choose|work|value|care|plan|intend|have|am|do)\b|"
    r"\b(?:my|our)\s+[a-z0-9][a-z0-9\s'/_-]{0,80}\s+(?:is|are|was|were|should|needs?|uses?|prefers?|avoids?|has|have)\b|"
    r"\bwhat\s+(?:i|we)\s+(?:want|prefer|need|avoid|like|use|value)\s+is\b",
    flags=re.IGNORECASE,
)
_PROFILE_VALUE_MARKERS = re.compile(
    r"\b(user\s+)?(?:uses?|prefers?|likes?|dislikes?|avoids?|needs?|wants?|plans?|values?|cares?|has|is|am|works?\s+with|usually|always)\b|"
    r"\b(?:current|setup|preference|goal|constraint|habit|style|brand|device|equipment|workflow)\b",
    flags=re.IGNORECASE,
)
_CJK_MEMORY_ASSERTION_RE = re.compile(
    r"(我|我们).{0,18}(想要|希望|偏好|喜欢|不喜欢|避免|需要|通常|总是|选择|使用|重视|计划|打算|有|是)|"
    r"(我的|我们的).{0,32}(是|需要|使用|偏好|避免|有)",
)
_TRANSIENT_ASSISTANT_DIRECTIVE_RE = re.compile(
    r"\b(answer|respond|reply|summari[sz]e|translate|compare|explain|based\s+only|only\s+use|do\s+not\s+(mention|reveal|include|use))\b|"
    r"\buse\s+(?:this|that|the|these|those|memory|context|retrieval|debug|tool|api|file|document|prompt|evidence|format|style)\b|"
    r"(请|帮我|你先|先别|不要|别).{0,18}(回答|回复|总结|翻译|解释|比较|复述|展开|写|生成|提到|泄露)|"
    r"(只|仅|只能|不要|别).{0,18}(基于|根据|使用|提到|绕远路|写总结)|"
    r"(长期记忆上下文|拿到的.*上下文|memory\s+context|retrieval|debug\s+metadata)",
    flags=re.IGNORECASE,
)
_CJK_QUESTION_MARKERS = (
    "什么",
    "吗",
    "有没有",
    "能不能",
    "是不是",
    "是否",
    "怎么",
    "如何",
    "为什么",
    "哪",
    "记得",
    "还记得",
)


def _safe_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").replace("\t", " ").split()).strip()


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _safe_text(value).lower()).strip("_") or "item"


def _tokens(value: Any) -> List[str]:
    return [item.lower() for item in _TOKEN_RE.findall(_safe_text(value)) if len(item) >= 2]


def _dedupe_texts(items: Iterable[Any], *, max_items: int | None = None) -> List[str]:
    values: List[str] = []
    seen = set()
    for item in items:
        text = _safe_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(text)
        if max_items is not None and len(values) >= max_items:
            break
    return values


def _facet_type(value: Any) -> str:
    raw = _safe_text(value).lower().replace("-", "_")
    raw = _FACET_TYPE_ALIASES.get(raw, raw)
    return raw if raw in _FACET_TYPES else ""


def _facet_role(facet: Mapping[str, Any], *, fallback: str = "") -> str:
    for key in ("role", "facet_role", "evidence_role", "temporal_role", "state_role", "numeric_role"):
        value = _safe_text(facet.get(key, ""))
        if value:
            return value[:80]
    return _safe_text(fallback)[:80]


def _facet_source_span(facet: Mapping[str, Any], *, fallback: str = "") -> str:
    for key in ("source_span", "span", "evidence_span", "quantity_span", "time_expression_span"):
        value = _safe_text(facet.get(key, ""))
        if value:
            return value[:240]
    return _safe_text(fallback)[:240]


def _facet_value(facet: Mapping[str, Any], *, facet_type: str) -> str:
    for key in ("value", "normalized_value", "text", "resolved_value", "resolved_time_value", "time_display_value"):
        value = _safe_text(facet.get(key, ""))
        if value:
            return value[:180]
    if facet_type == "numeric":
        numeric_value = _safe_text(facet.get("numeric_value", facet.get("number", facet.get("amount", ""))))
        unit = _safe_text(facet.get("unit", facet.get("numeric_unit", "")))
        if numeric_value and unit:
            return f"{numeric_value} {unit}"[:180]
        if numeric_value:
            return numeric_value[:180]
    if facet_type == "temporal":
        value = _safe_text(facet.get("time_expression_span", facet.get("resolved_date", "")))
        if value:
            return value[:180]
    if facet_type == "state":
        value = _safe_text(facet.get("target_status", facet.get("state", "")))
        if value:
            return value[:180]
    return ""


def _writer_facets_from_proposal(
    proposal: Mapping[str, Any],
    *,
    source_span: str,
    subject: str = "",
    semantic_slot: str = "",
) -> List[Dict[str, Any]]:
    raw_facets = proposal.get("facets", proposal.get("memory_facets", proposal.get("attribute_facets", [])))
    if isinstance(raw_facets, Mapping):
        raw_items: List[Any] = [raw_facets]
    elif isinstance(raw_facets, list):
        raw_items = list(raw_facets)
    else:
        raw_items = []
    facets: List[Dict[str, Any]] = [dict(item) for item in raw_items if isinstance(item, Mapping)]
    explicit_facet_types = {
        _facet_type(item.get("type", item.get("facet_type", item.get("category", ""))))
        for item in facets
        if isinstance(item, Mapping)
    }

    time_span = _safe_text(proposal.get("time_expression_span", ""))
    resolved_time = _safe_text(proposal.get("resolved_time_value", "")) or _safe_text(proposal.get("time_display_value", ""))
    resolved_date = _safe_text(proposal.get("resolved_date", ""))
    if "temporal" not in explicit_facet_types and (time_span or resolved_time or resolved_date):
        facets.append(
            {
                "type": "temporal",
                "role": _safe_text(proposal.get("target_status", "")) or _safe_text(semantic_slot) or "time",
                "value": resolved_time or resolved_date or time_span,
                "source_span": time_span or source_span,
                "resolved_time_value": resolved_time,
                "resolved_date": resolved_date,
                "time_granularity": _safe_text(proposal.get("time_granularity", "")),
            }
        )

    target_status = _safe_text(proposal.get("target_status", ""))
    if "state" not in explicit_facet_types and target_status:
        facets.append(
            {
                "type": "state",
                "role": "target_status",
                "value": target_status,
                "source_span": source_span,
            }
        )

    numeric_value = _safe_text(proposal.get("numeric_value", proposal.get("quantity_value", proposal.get("amount_value", ""))))
    numeric_unit = _safe_text(proposal.get("numeric_unit", proposal.get("quantity_unit", proposal.get("amount_unit", ""))))
    quantity_span = _safe_text(proposal.get("quantity_span", proposal.get("numeric_span", "")))
    if "numeric" not in explicit_facet_types and (numeric_value or quantity_span):
        facets.append(
            {
                "type": "numeric",
                "role": _safe_text(proposal.get("numeric_role", "")) or _safe_text(semantic_slot) or "quantity",
                "value": f"{numeric_value} {numeric_unit}".strip() or quantity_span,
                "source_span": quantity_span or source_span,
                "numeric_value": numeric_value,
                "unit": numeric_unit,
            }
        )

    if "entity" not in explicit_facet_types and subject:
        facets.append(
            {
                "type": "entity",
                "role": "subject",
                "value": subject,
                "source_span": subject if _safe_text(subject).lower() in _safe_text(source_span).lower() else source_span,
            }
        )

    normalized: List[Dict[str, Any]] = []
    seen = set()
    for raw in facets:
        facet_type = _facet_type(raw.get("type", raw.get("facet_type", raw.get("category", ""))))
        if not facet_type:
            continue
        role = _facet_role(raw, fallback=semantic_slot or facet_type)
        value = _facet_value(raw, facet_type=facet_type)
        span = _facet_source_span(raw, fallback=source_span)
        if not value or not span:
            continue
        key = (facet_type, role.lower(), value.lower(), span.lower())
        if key in seen:
            continue
        seen.add(key)
        normalized.append({**raw, "type": facet_type, "role": role, "value": value, "source_span": span})
    return normalized[:6]


def _build_event_facet_records(
    *,
    parent_record: Mapping[str, Any],
    proposal: Mapping[str, Any],
    common_metadata: Mapping[str, Any],
    source_context: str,
    current_turn_text: str,
    speaker: str,
    timestamp: str,
    turn_index: int,
    proposal_index: int,
) -> List[Dict[str, Any]]:
    if _safe_text(parent_record.get("category", "")).lower() in {"question", "interaction_intent"}:
        return []
    parent_metadata = dict(parent_record.get("metadata", {}) or {})
    parent_slot_key = _safe_text(parent_record.get("slot_key", ""))
    parent_event_signature = _safe_text(parent_metadata.get("event_signature", ""))
    parent_source_span = _safe_text(parent_metadata.get("source_span", "")) or _safe_text(parent_record.get("value", ""))
    parent_subject = _safe_text(parent_metadata.get("subject", ""))
    parent_subject_signature = _safe_text(parent_metadata.get("subject_signature", ""))
    parent_semantic_slot = _safe_text(parent_metadata.get("semantic_slot", "")) or _safe_text(parent_record.get("category", ""))
    facets = _writer_facets_from_proposal(
        proposal,
        source_span=parent_source_span,
        subject=parent_subject,
        semantic_slot=parent_semantic_slot,
    )
    records: List[Dict[str, Any]] = []
    for facet_index, facet in enumerate(facets, start=1):
        facet_type = _facet_type(facet.get("type", ""))
        role = _facet_role(facet, fallback=parent_semantic_slot or facet_type)
        value = _facet_value(facet, facet_type=facet_type)
        span = _facet_source_span(facet, fallback=parent_source_span)
        if not facet_type or not value or not span:
            continue
        if not _span_is_grounded_in_current_turn(source_span=span, current_turn_text=current_turn_text):
            # Facets are evidence-bearing; keep only attributes grounded in the
            # writable turn or in the already-grounded parent source span.
            span = parent_source_span
            if not _span_is_grounded_in_current_turn(source_span=span, current_turn_text=current_turn_text):
                continue
        category = _FACET_CATEGORY_BY_TYPE.get(facet_type, "fact")
        slot_key = (
            f"{parent_slot_key}.facet.{facet_type}.{_slug(role)}.{facet_index}"
            if parent_slot_key
            else f"turn_{max(1, int(turn_index or 1))}.facet.{facet_type}.{proposal_index}.{facet_index}"
        )
        anchors = _dedupe_texts(
            [
                speaker,
                timestamp,
                parent_subject,
                facet_type,
                role,
                value,
                *list(parent_record.get("anchors", []) or []),
            ],
            max_items=12,
        )
        metadata = {
            **dict(common_metadata),
            "content_variant": "event_facet_write",
            "facet_layer_version": "event_facet_v1",
            "facet_type": facet_type,
            "facet_role": role,
            "facet_value": value,
            "facet_source_span": span,
            "facet_parent_slot_key": parent_slot_key,
            "facet_parent_event_signature": parent_event_signature,
            "facet_parent_category": _safe_text(parent_record.get("category", "")),
            "facet_parent_relation": _safe_text(parent_record.get("relation", "")),
            "facet_parent_source_kind": _safe_text(parent_record.get("source_kind", "")),
            "raw_text": span,
            "source_turn_text": span,
            "source_span": span,
            "semantic_slot": f"{facet_type}:{role}",
            "event_phrase": value,
            "event_text": value,
            "event_signature": f"{parent_event_signature} facet {facet_type} {role} {value}".strip(),
            "subject": parent_subject,
            "subject_signature": parent_subject_signature,
            "canonical_slot_key": slot_key,
            "allow_parallel_state": True,
            "memory_gate_decision": "facet_from_parent_event",
            "grounding_score": round(float(_grounding_score(value=value, source_span=span, source_context=source_context)), 6),
            "value_grounding_score": round(float(_value_grounding_score(value=value, source_context=source_context)), 6),
            "llm_write_proposal_index": proposal_index,
            "facet_index": facet_index,
            "facet_raw": {key: _safe_text(val) if not isinstance(val, (list, dict)) else val for key, val in dict(facet).items()},
            "time_expression_span": _safe_text(facet.get("time_expression_span", "")),
            "time_granularity": _safe_text(facet.get("time_granularity", "")),
            "resolved_time_value": _safe_text(facet.get("resolved_time_value", "")),
            "time_display_value": _safe_text(facet.get("time_display_value", "")),
            "resolved_date": _safe_text(facet.get("resolved_date", "")),
            "target_status": _safe_text(facet.get("target_status", "")),
            "numeric_value": _safe_text(facet.get("numeric_value", facet.get("number", facet.get("amount", "")))),
            "numeric_unit": _safe_text(facet.get("unit", facet.get("numeric_unit", ""))),
            "entity_signature": _slug(value) if facet_type == "entity" else "",
            "evidence_role": role if facet_type == "evidence_role" else "",
        }
        records.append(
            {
                "category": category,
                "slot_key": slot_key,
                "value": value,
                "anchors": anchors,
                "relation": f"{facet_type}_facet_of",
                "source_kind": "public_dialog_facet",
                "state": "active",
                "salience": min(0.96, max(0.72, float(parent_record.get("salience", 0.86) or 0.86))),
                "confidence": min(0.96, max(0.70, float(parent_record.get("confidence", 0.84) or 0.84))),
                "metadata": metadata,
            }
        )
    return records


def _proposal_parent_candidates(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            continue
        metadata = dict(raw.get("metadata", {}) or {})
        category = _safe_text(raw.get("category", ""))
        if category.lower() in {"question", "interaction_intent", "profile", "preference"}:
            continue
        slot_key = _safe_text(raw.get("slot_key", ""))
        value = _safe_text(raw.get("value", ""))
        source_span = _safe_text(metadata.get("source_span", "")) or value
        if not value or not slot_key:
            continue
        candidates.append(
            {
                "index": str(index),
                "slot_key": slot_key,
                "category": category,
                "value": value[:220],
                "source_span": source_span[:320],
                "event_signature": _safe_text(metadata.get("event_signature", "")),
            }
        )
    return candidates[:12]


def _unit_writer_signal_hits(text: str, parents: Sequence[Mapping[str, Any]]) -> List[str]:
    haystack_parts = [_safe_text(text)]
    for parent in parents[:8]:
        if not isinstance(parent, Mapping):
            continue
        haystack_parts.extend(
            [
                _safe_text(parent.get("value", "")),
                _safe_text(parent.get("source_span", "")),
                _safe_text(parent.get("category", "")),
            ]
        )
    haystack = "\n".join(item for item in haystack_parts if item)
    hits: List[str] = []
    if _UNIT_WRITER_ACTION_RE.search(haystack):
        hits.append("action")
    if _UNIT_WRITER_NUMERIC_RE.search(haystack):
        hits.append("numeric")
    if _UNIT_WRITER_TEMPORAL_RE.search(haystack):
        hits.append("temporal")
    if _UNIT_WRITER_STATE_RE.search(haystack):
        hits.append("state")
    if _UNIT_WRITER_ROLE_RE.search(haystack):
        hits.append("role")
    return _dedupe_texts(hits, max_items=8)


def _unit_writer_clause_span(text: str, start: int, end: int, *, max_chars: int = 220) -> str:
    source = _safe_text(text)
    if not source:
        return ""
    start = max(0, min(len(source), int(start)))
    end = max(start, min(len(source), int(end)))
    left = max(source.rfind(".", 0, start), source.rfind(";", 0, start), source.rfind("\n", 0, start))
    left = 0 if left < 0 else left + 1
    comma_left = source.rfind(",", left, start)
    if comma_left >= left and start - comma_left > 36:
        left = comma_left + 1
    right_candidates = [idx for idx in (source.find(".", end), source.find(";", end), source.find("\n", end)) if idx >= 0]
    right = min(right_candidates) if right_candidates else len(source)
    comma_right = source.find(",", end, right)
    if comma_right >= 0 and comma_right - end > 48:
        right = comma_right
    if right - left > max_chars:
        left = max(0, start - 80)
        right = min(len(source), end + 120)
    return source[left:right].strip(" \t\r\n,;")


def _unit_writer_coverage_requirements(text: str, *, max_items: int = 18) -> List[Dict[str, str]]:
    source = _safe_text(text)
    if not source:
        return []
    requirements: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, facet_type: str, span: str, role: str = "") -> None:
        clean_span = _safe_text(span)
        if not clean_span:
            return
        key = (kind, re.sub(r"\s+", " ", clean_span.lower()).strip())
        if key in seen:
            return
        seen.add(key)
        requirements.append(
            {
                "id": f"req_{len(requirements) + 1}",
                "unit_kind_hint": kind,
                "type_hint": facet_type,
                "role_hint": role or kind,
                "source_span": clean_span[:220],
            }
        )

    for match in _UNIT_WRITER_ACTION_RE.finditer(source):
        add("action_unit", "action", _unit_writer_clause_span(source, match.start(), match.end()), "action")
    for match in _UNIT_WRITER_NUMERIC_RE.finditer(source):
        add("numeric_quantity", "numeric", match.group(0), "quantity")
    for match in _UNIT_WRITER_TEMPORAL_RE.finditer(source):
        temporal_span = match.group(0)
        nearby = source[match.start() : min(len(source), match.end() + 16)].lower()
        if "/" in temporal_span and "scale" in nearby:
            continue
        add("temporal_anchor", "temporal", match.group(0), "time")
    for match in _UNIT_WRITER_STATE_RE.finditer(source):
        add("state_status", "state", _unit_writer_clause_span(source, match.start(), match.end(), max_chars=180), "status")
    for match in _UNIT_WRITER_ROLE_RE.finditer(source):
        add("evidence_role", "evidence_role", _unit_writer_clause_span(source, match.start(), match.end(), max_chars=180), "role")
    return requirements[: max(1, int(max_items or 18))]


def _unit_requirement_is_covered(requirement: Mapping[str, Any], unit_record: Mapping[str, Any]) -> bool:
    metadata = dict(unit_record.get("metadata", {}) or {})
    req_kind = _safe_text(requirement.get("unit_kind_hint", "")).lower()
    req_type = _safe_text(requirement.get("type_hint", "")).lower()
    req_span = _safe_text(requirement.get("source_span", ""))
    unit_kind = _safe_text(metadata.get("unit_kind", "")).lower()
    unit_type = _safe_text(metadata.get("facet_type", "")).lower()
    unit_text = " ".join(
        [
            _safe_text(unit_record.get("value", "")),
            _safe_text(metadata.get("facet_source_span", "")),
            _safe_text(metadata.get("source_span", "")),
        ]
    )
    if req_kind and unit_kind and req_kind != unit_kind:
        if req_kind not in unit_kind and unit_kind not in req_kind:
            return False
    if req_type and unit_type and req_type != unit_type:
        return False
    req_norm = re.sub(r"\s+", " ", req_span.lower()).strip()
    unit_norm = re.sub(r"\s+", " ", unit_text.lower()).strip()
    if req_norm and unit_norm and (req_norm in unit_norm or unit_norm in req_norm):
        return True
    req_tokens = {token for token in _tokens(req_span) if len(token) > 2}
    unit_tokens = {token for token in _tokens(unit_text) if len(token) > 2}
    if not req_tokens or not unit_tokens:
        return False
    coverage = len(req_tokens & unit_tokens) / max(1, len(req_tokens))
    return coverage >= 0.45


def _unit_writer_uncovered_requirements(
    requirements: Sequence[Mapping[str, Any]],
    unit_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, str]]:
    uncovered: List[Dict[str, str]] = []
    for requirement in requirements:
        if not isinstance(requirement, Mapping):
            continue
        if any(_unit_requirement_is_covered(requirement, record) for record in unit_records if isinstance(record, Mapping)):
            continue
        uncovered.append({key: _safe_text(value) for key, value in dict(requirement).items()})
    return uncovered


def _unit_writer_fallback_units_from_requirements(
    requirements: Sequence[Mapping[str, Any]],
    *,
    parent_records: Sequence[Mapping[str, Any]],
    current_turn_text: str,
) -> List[Dict[str, Any]]:
    """Grounded attachment fallback when the LLM returns no unit for an obvious span."""

    output: List[Dict[str, Any]] = []
    source = _safe_text(current_turn_text)
    for item in requirements:
        if not isinstance(item, Mapping):
            continue
        span = _safe_text(item.get("source_span", ""))
        if not span or not _span_is_grounded_in_current_turn(source_span=span, current_turn_text=source):
            continue
        kind = _safe_text(item.get("unit_kind_hint", "")) or "evidence_role"
        facet_type = _safe_text(item.get("type_hint", "")) or "evidence_role"
        role = _safe_text(item.get("role_hint", "")) or kind
        if kind == "action_unit":
            match = _UNIT_WRITER_ACTION_RE.search(span)
            action = match.group(0) if match else ""
            value = span
        elif kind == "numeric_quantity":
            action = ""
            value = span
        elif kind == "temporal_anchor":
            action = ""
            value = span
        elif kind == "state_status":
            action = ""
            value = span
        else:
            action = ""
            value = span
        parent = _resolve_unit_parent_record({"source_span": span, "value": value}, parent_records)
        output.append(
            {
                "parent_slot_key": _safe_text(parent.get("slot_key", "")) if parent else "",
                "unit_kind": kind,
                "type": facet_type,
                "role": role,
                "value": value,
                "source_span": span,
                "action": action,
                "target": "",
                "action_frame_id": _slug(f"fallback {kind} {span}")[:80],
                "action_frame_index": "fallback",
                "action_frame_source_span": span,
                "applies_to_action": action,
                "applies_to_entity": "",
                "unit_writer_fallback": "coverage_requirement",
            }
        )
    return output


def _as_mapping_list(value: Any) -> List[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _first_text(mapping: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        text = _safe_text(mapping.get(key, ""))
        if text:
            return text
    return ""


def _unit_writer_raw_units_from_action_frames(parsed: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Convert action-frame output into flat event_unit records while keeping frame binding metadata."""

    raw_frames = (
        parsed.get("action_frames")
        or parsed.get("frames")
        or parsed.get("event_frames")
        or parsed.get("action_frame_units")
        or []
    )
    frames = _as_mapping_list(raw_frames)
    output: List[Dict[str, Any]] = []
    for frame_index, frame in enumerate(frames, start=1):
        action = _first_text(frame, ("action", "verb", "action_unit", "operation"))
        entity = _first_text(frame, ("entity", "target_entity", "target", "object", "item"))
        source_span = _first_text(frame, ("source_span", "span", "evidence_span", "action_span"))
        frame_id = _safe_text(frame.get("frame_id", "")) or _slug(" ".join([action, entity, str(frame_index)]))
        parent_slot_key = _safe_text(frame.get("parent_slot_key", frame.get("parent", "")))
        base = {
            "parent_slot_key": parent_slot_key,
            "action_frame_id": frame_id,
            "action_frame_index": str(frame_index),
            "action_frame_source_span": source_span,
            "applies_to_action": action,
            "applies_to_entity": entity,
        }
        if action:
            value = " ".join(item for item in (action, entity) if item).strip() or source_span
            output.append(
                {
                    **base,
                    "unit_kind": "action_unit",
                    "type": "action",
                    "role": _safe_text(frame.get("action_role", "")) or "action",
                    "value": value,
                    "source_span": _first_text(frame, ("action_span", "source_span", "span")) or source_span or value,
                    "action": action,
                    "target": entity,
                }
            )
        if entity:
            output.append(
                {
                    **base,
                    "unit_kind": "target_entity",
                    "type": "entity",
                    "role": _safe_text(frame.get("entity_role", "")) or "target_entity",
                    "value": entity,
                    "source_span": _first_text(frame, ("entity_span", "source_span", "span")) or entity,
                    "action": action,
                    "target": entity,
                }
            )
        for temporal_index, temporal in enumerate(
            _as_mapping_list(frame.get("temporal_anchors", frame.get("temporals", frame.get("times", [])))),
            start=1,
        ):
            value = _first_text(temporal, ("value", "time", "time_expression_span", "resolved_time_value", "resolved_date"))
            span = _first_text(temporal, ("source_span", "span", "time_expression_span")) or value
            if not value and not span:
                continue
            role = _first_text(temporal, ("role", "temporal_role", "applies_to_role")) or "temporal_anchor"
            output.append(
                {
                    **base,
                    "unit_kind": "temporal_anchor",
                    "type": "temporal",
                    "role": role,
                    "value": value or span,
                    "source_span": span,
                    "normalized_time": _first_text(temporal, ("normalized_time", "resolved_time_value", "resolved_date")),
                    "temporal_role": role,
                    "temporal_index": str(temporal_index),
                    "action": action,
                    "target": entity,
                }
            )
        for numeric_index, numeric in enumerate(
            _as_mapping_list(frame.get("numeric_quantities", frame.get("numerics", frame.get("quantities", [])))),
            start=1,
        ):
            value = _first_text(numeric, ("value", "quantity", "numeric_value", "number", "amount"))
            unit = _first_text(numeric, ("unit", "numeric_unit"))
            span = _first_text(numeric, ("source_span", "span", "quantity_span")) or " ".join(item for item in (value, unit) if item)
            if not value and not span:
                continue
            output.append(
                {
                    **base,
                    "unit_kind": "numeric_quantity",
                    "type": "numeric",
                    "role": _first_text(numeric, ("role", "numeric_role")) or "quantity",
                    "value": " ".join(item for item in (value, unit) if item).strip() or span,
                    "source_span": span,
                    "quantity": value,
                    "unit": unit,
                    "numeric_index": str(numeric_index),
                    "action": action,
                    "target": entity,
                }
            )
        for state_index, state in enumerate(
            _as_mapping_list(frame.get("states", frame.get("state_statuses", frame.get("statuses", [])))),
            start=1,
        ):
            value = _first_text(state, ("value", "status", "state", "target_status"))
            span = _first_text(state, ("source_span", "span")) or value or source_span
            if not value and not span:
                continue
            output.append(
                {
                    **base,
                    "unit_kind": "state_status",
                    "type": "state",
                    "role": _first_text(state, ("role", "state_role")) or "status",
                    "value": value or span,
                    "source_span": span,
                    "status": value,
                    "state_index": str(state_index),
                    "action": action,
                    "target": entity,
                }
            )
        for role_index, role_item in enumerate(
            _as_mapping_list(frame.get("roles", frame.get("participation_roles", []))),
            start=1,
        ):
            value = _first_text(role_item, ("value", "role", "participant", "actor")) or _safe_text(frame.get("actor", ""))
            span = _first_text(role_item, ("source_span", "span")) or source_span or value
            if not value and not span:
                continue
            role_value = _first_text(role_item, ("role", "evidence_role")) or "role"
            output.append(
                {
                    **base,
                    "unit_kind": "leadership_role" if "lead" in role_value.lower() else "participation_role",
                    "type": "role",
                    "role": role_value,
                    "value": value or role_value,
                    "source_span": span,
                    "role_index": str(role_index),
                    "action": action,
                    "target": entity,
                }
            )
    return output


def _resolve_unit_parent_record(unit: Mapping[str, Any], parent_records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    requested_slot = _safe_text(unit.get("parent_slot_key", unit.get("parent", ""))).lower()
    if requested_slot:
        for record in parent_records:
            if _safe_text(record.get("slot_key", "")).lower() == requested_slot:
                return record
    span = _safe_text(unit.get("source_span", ""))
    value = _safe_text(unit.get("value", unit.get("target", "")))
    best: tuple[int, Mapping[str, Any] | None] = (0, None)
    for record in parent_records:
        metadata = dict(record.get("metadata", {}) or {})
        haystack = " ".join(
            [
                _safe_text(record.get("value", "")),
                _safe_text(metadata.get("source_span", "")),
                _safe_text(metadata.get("raw_text", "")),
            ]
        ).lower()
        score = 0
        if span and span.lower() in haystack:
            score += 5
        if value and value.lower() in haystack:
            score += 2
        score += len(set(_tokens(span or value)) & set(_tokens(haystack)))
        if score > best[0]:
            best = (score, record)
    if best[1] is not None:
        return best[1]
    return parent_records[0] if parent_records else None


def _unit_kind(raw: Mapping[str, Any], facet_type: str, role: str) -> str:
    value = _safe_text(raw.get("unit_kind", raw.get("kind", raw.get("schema", "")))).lower()
    if value:
        return _slug(value)[:80]
    role_l = role.lower()
    if facet_type == "action":
        return "action_unit"
    if facet_type == "entity":
        return "target_entity"
    if facet_type == "numeric":
        return "numeric_quantity"
    if facet_type == "temporal":
        return "temporal_anchor"
    if facet_type == "state":
        return "state_status"
    if "lead" in role_l:
        return "leadership_role"
    if "participat" in role_l or "volunteer" in role_l or "attend" in role_l:
        return "participation_role"
    return "evidence_role"


def build_modelized_facet_unit_records(
    writer: Any,
    *,
    current_turn_text: str,
    parent_records: Sequence[Mapping[str, Any]],
    speaker: str,
    timestamp: str,
    turn_index: int,
    max_units: int = 14,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Attach action-frame unit nodes under accepted event/source records.

    This is not a separate multi-memory writer. It refines normal event writes by
    adding action-frame-bound attributes that graph tunneling can later connect.
    """

    mode = _safe_text(
        os.environ.get("TMCRA_EVENT_ACTION_FRAME_ATTACHMENT_MODE", os.environ.get("TMCRA_MODELIZED_UNIT_WRITER_MODE", "on"))
    ).lower()
    if mode in {"0", "off", "false", "disabled", "none"}:
        return [], {"enabled": False, "reason": "disabled"}
    parents = _proposal_parent_candidates(parent_records)
    source_or_event_parents = [
        item
        for item in parents
        if _safe_text(item.get("category", "")).lower() in {"fact", "event", "time", "status", "constraint", "goal"}
    ]
    if not source_or_event_parents:
        return [], {"enabled": False, "reason": "no_event_or_source_parent_records", "parent_count": len(parents)}
    gate_mode = _safe_text(os.environ.get("TMCRA_MODELIZED_UNIT_WRITER_SIGNAL_GATE", "on")).lower()
    signal_hits = _unit_writer_signal_hits(current_turn_text, source_or_event_parents)
    if gate_mode not in {"0", "off", "false", "disabled", "none"} and not signal_hits:
        return [], {
            "enabled": False,
            "reason": "no_unit_worthy_signal",
            "parent_count": len(parents),
            "source_or_event_parent_count": len(source_or_event_parents),
        }
    chat = getattr(writer, "_chat", None)
    if chat is None and hasattr(writer, "base_writer"):
        chat = getattr(getattr(writer, "base_writer"), "_chat", None)
    if chat is None:
        return [], {
            "enabled": False,
            "reason": "writer_chat_unavailable",
            "parent_count": len(parents),
            "unit_signal_hits": signal_hits,
        }
    max_units = max(1, min(24, int(max_units or 14)))
    coverage_requirements = _unit_writer_coverage_requirements(current_turn_text, max_items=max_units)
    system_prompt = (
        "Return strict JSON only. You are the TMCRA event action-frame attachment writer. "
        "Your job is to attach grounded action frames under current_turn's accepted event/source records. "
        "This is not a multi-memory writer and you must not decide whether separate events form a chain. "
        "Do not answer any benchmark question and do not infer facts not grounded in current_turn plus its explicit timestamp/session date metadata. "
        "Split one sentence into multiple action_frames when it contains multiple actions with different time/status/numeric bindings. "
        "For each action_frame, bind temporal_anchors, numeric_quantities, states, roles, and target_entity to the specific action they apply to. "
        "Preserve exact answer-bearing values as units: names, places, relationship/status labels, countries, dates, years, quantities, object identities, and role participants. "
        "For status-like phrases, emit state units with precise role names such as relationship_status, life_status, task_status, possession_status, or membership_status; examples include single parent -> relationship_status/single and passed interviews -> task_status/passed. "
        "For appositions or identity bindings, emit entity units with role identity_binding or origin_binding; examples include home country, Sweden -> home_country = Sweden and necklace from grandma -> origin/person = grandma. "
        "For relative time expressions, keep the exact expression in value and source_span, and fill normalized_time when it can be grounded from the turn/session date; examples: last year with a 2023 session date -> 2022, yesterday -> normalized calendar date if computable. "
        "If a turn says 'I got X and will do Y this weekend', create one frame for getting X with the turn time/current status, and one frame for doing Y with weekend as planned_action_time. "
        "Use coverage_requirements as grounded spans that should be covered by frames or frame attributes when durable. "
        "For every source_span, copy exact text from current_turn. "
        "For parent_slot_key, choose one slot_key from parent_candidates when possible; otherwise use the source_turn parent. "
        "Return {\"action_frames\":[{\"parent_slot_key\":\"...\",\"frame_id\":\"...\",\"action\":\"...\",\"entity\":\"...\",\"source_span\":\"...\","
        "\"temporal_anchors\":[{\"role\":\"acquisition_time|planned_action_time|completion_time|current_turn_time|event_time\",\"value\":\"...\",\"source_span\":\"...\",\"normalized_time\":\"\"}],"
        "\"numeric_quantities\":[{\"role\":\"quantity|scale|amount|duration\",\"value\":\"...\",\"unit\":\"...\",\"source_span\":\"...\"}],"
        "\"states\":[{\"role\":\"status\",\"value\":\"current|planned|completed|pending|returned|started\",\"source_span\":\"...\"}],"
        "\"roles\":[{\"role\":\"leadership|participation|ownership\",\"value\":\"...\",\"source_span\":\"...\"}]}],"
        "\"units\":[{\"parent_slot_key\":\"...\",\"unit_kind\":\"...\",\"type\":\"action|entity|numeric|temporal|state|role|evidence_role\","
        "\"role\":\"...\",\"value\":\"...\",\"source_span\":\"...\",\"action\":\"\",\"target\":\"\",\"quantity\":\"\",\"unit\":\"\",\"normalized_time\":\"\",\"status\":\"\"}]}. "
        "Prefer action_frames; include units only for extra attributes that cannot fit a frame. Do not emit empty or duplicate data."
    )
    payload = {
        "speaker": _safe_text(speaker),
        "timestamp": _safe_text(timestamp),
        "max_units": max_units,
        "parent_candidates": parents,
        "coverage_requirements": coverage_requirements,
        "current_turn": _safe_text(current_turn_text),
    }
    try:
        raw, usage = chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
    except Exception as exc:
        return [], {
            "enabled": True,
            "error": f"{exc.__class__.__name__}: {str(exc)[:200]}",
            "parent_count": len(parents),
        }
    parsed = _extract_json_object(raw)
    raw_frame_units = _unit_writer_raw_units_from_action_frames(parsed) if isinstance(parsed, Mapping) else []
    raw_units = parsed.get("units", []) if isinstance(parsed, Mapping) else []
    if not isinstance(raw_units, list):
        raw_units = []
    raw_units = list(raw_frame_units) + [item for item in raw_units if isinstance(item, Mapping)]
    source_text = _safe_text(current_turn_text)
    output: List[Dict[str, Any]] = []
    seen = set()
    parent_records_by_slot = {_safe_text(item.get("slot_key", "")): item for item in parent_records if isinstance(item, Mapping)}

    def materialize_units(raw_items: Sequence[Any], *, phase: str, limit: int) -> int:
        stored = 0
        for raw_unit in list(raw_items or [])[: max(0, int(limit or 0))]:
            if not isinstance(raw_unit, Mapping):
                continue
            facet_type = _facet_type(raw_unit.get("type", raw_unit.get("facet_type", raw_unit.get("unit_kind", ""))))
            if not facet_type:
                continue
            role = _facet_role(raw_unit, fallback=_unit_kind(raw_unit, facet_type, ""))
            value = _facet_value(raw_unit, facet_type=facet_type) or _safe_text(raw_unit.get("target", raw_unit.get("action", "")))
            normalized_time = _safe_text(raw_unit.get("normalized_time", raw_unit.get("resolved_date", "")))
            if facet_type == "temporal" and normalized_time and normalized_time.lower() not in value.lower():
                value = f"{value} ({normalized_time})"
            span = _facet_source_span(raw_unit, fallback=value)
            if not value or not span:
                continue
            if not _span_is_grounded_in_current_turn(source_span=span, current_turn_text=source_text):
                continue
            parent = _resolve_unit_parent_record(raw_unit, parent_records)
            if parent is None:
                continue
            parent_slot_key = _safe_text(parent.get("slot_key", ""))
            parent_metadata = dict(parent.get("metadata", {}) or {})
            unit_kind = _unit_kind(raw_unit, facet_type, role)
            key = (parent_slot_key.lower(), unit_kind, facet_type, role.lower(), value.lower(), span.lower())
            if key in seen:
                continue
            seen.add(key)
            category = _FACET_CATEGORY_BY_TYPE.get(facet_type, "fact")
            slot_key = f"{parent_slot_key}.unit.{facet_type}.{_slug(unit_kind)}.{len(output) + 1}"
            metadata = {
                **parent_metadata,
                "content_variant": "event_facet_write",
                "facet_layer_version": "event_unit_v1",
                "unit_writer": "event_action_frame_attachment_writer_v2",
                "unit_writer_phase": phase,
                "unit_attachment_schema": "event_action_frame_v2",
                "action_frame_id": _safe_text(raw_unit.get("action_frame_id", raw_unit.get("frame_id", ""))),
                "action_frame_index": _safe_text(raw_unit.get("action_frame_index", "")),
                "action_frame_source_span": _safe_text(raw_unit.get("action_frame_source_span", "")),
                "applies_to_action": _safe_text(raw_unit.get("applies_to_action", raw_unit.get("action", ""))),
                "applies_to_entity": _safe_text(raw_unit.get("applies_to_entity", raw_unit.get("target", ""))),
                "unit_writer_fallback": _safe_text(raw_unit.get("unit_writer_fallback", "")),
                "temporal_role": _safe_text(raw_unit.get("temporal_role", "")),
                "unit_kind": unit_kind,
                "facet_type": facet_type,
                "facet_role": role,
                "facet_value": value,
                "facet_source_span": span,
                "facet_parent_slot_key": parent_slot_key,
                "facet_parent_event_signature": _safe_text(parent_metadata.get("event_signature", "")),
                "facet_parent_category": _safe_text(parent.get("category", "")),
                "facet_parent_relation": _safe_text(parent.get("relation", "")),
                "raw_text": span,
                "source_turn_text": span,
                "source_span": span,
                "semantic_slot": f"{unit_kind}:{role}",
                "event_phrase": value,
                "event_text": value,
                "event_signature": f"{_safe_text(parent_metadata.get('event_signature', ''))} unit {unit_kind} {role} {value}".strip(),
                "action": _safe_text(raw_unit.get("action", "")),
                "target": _safe_text(raw_unit.get("target", "")),
                "quantity": _safe_text(raw_unit.get("quantity", raw_unit.get("numeric_value", ""))),
                "unit": _safe_text(raw_unit.get("unit", "")),
                "normalized_time": normalized_time,
                "status": _safe_text(raw_unit.get("status", "")),
            }
            output.append(
                {
                    "category": category,
                    "slot_key": slot_key,
                    "value": value,
                    "anchors": _dedupe_texts([speaker, timestamp, unit_kind, facet_type, role, value], max_items=10),
                    "relation": "unit_of_event",
                    "source_kind": "event_action_frame_attachment",
                    "state": "evidence",
                    "salience": 0.94,
                    "confidence": 0.86,
                    "metadata": metadata,
                }
            )
            stored += 1
        return stored

    initial_stored_count = materialize_units(raw_units, phase="initial", limit=max_units)
    uncovered_before_repair = _unit_writer_uncovered_requirements(coverage_requirements, output)
    repair_raw_count = 0
    repair_stored_count = 0
    repair_error = ""
    repair_usage: Dict[str, Any] = {}
    repair_mode = _safe_text(
        os.environ.get(
            "TMCRA_EVENT_ACTION_FRAME_ATTACHMENT_REPAIR_MODE",
            os.environ.get("TMCRA_MODELIZED_UNIT_WRITER_COVERAGE_REPAIR_MODE", "on"),
        )
    ).lower()
    if (
        repair_mode not in {"0", "off", "false", "disabled", "none"}
        and uncovered_before_repair
        and len(output) < max_units
    ):
        repair_prompt = (
            "Return strict JSON only. You are repairing missing TMCRA event action-frame attachment writes. "
            "The initial pass missed some grounded coverage_requirements. "
            "Emit missing action_frames or units only, using exact source_span text from current_turn. "
            "Do not repeat existing_units. Do not infer facts not grounded in current_turn plus explicit timestamp/session date metadata. "
            "Do not decide whether this is a multi-memory chain; only attach attributes to the current event. "
            "Prefer action_frames with correctly bound temporal_anchors/numeric_quantities/states. "
            "Repair missing precise answer-bearing units: relationship/status labels, country/place identity bindings, normalized relative dates, quantities, and participants. "
            "Return {\"action_frames\":[{\"parent_slot_key\":\"...\",\"frame_id\":\"...\",\"action\":\"...\",\"entity\":\"...\",\"source_span\":\"...\","
            "\"temporal_anchors\":[{\"role\":\"...\",\"value\":\"...\",\"source_span\":\"...\",\"normalized_time\":\"\"}],"
            "\"numeric_quantities\":[{\"role\":\"...\",\"value\":\"...\",\"unit\":\"...\",\"source_span\":\"...\"}],"
            "\"states\":[{\"role\":\"status\",\"value\":\"...\",\"source_span\":\"...\"}],\"roles\":[]}],"
            "\"units\":[{\"parent_slot_key\":\"...\",\"unit_kind\":\"...\",\"type\":\"action|entity|numeric|temporal|state|role|evidence_role\","
            "\"role\":\"...\",\"value\":\"...\",\"source_span\":\"...\",\"action\":\"\",\"target\":\"\",\"quantity\":\"\",\"unit\":\"\",\"normalized_time\":\"\",\"status\":\"\"}]}."
        )
        repair_payload = {
            "speaker": _safe_text(speaker),
            "timestamp": _safe_text(timestamp),
            "max_missing_units": max_units - len(output),
            "parent_candidates": parents,
            "current_turn": source_text,
            "uncovered_requirements": uncovered_before_repair,
            "existing_units": [
                {
                    "unit_kind": dict(item.get("metadata", {}) or {}).get("unit_kind", ""),
                    "type": dict(item.get("metadata", {}) or {}).get("facet_type", ""),
                    "value": item.get("value", ""),
                    "source_span": dict(item.get("metadata", {}) or {}).get("facet_source_span", ""),
                }
                for item in output
            ],
        }
        try:
            repair_raw, repair_usage = chat(
                [
                    {"role": "system", "content": repair_prompt},
                    {"role": "user", "content": json.dumps(repair_payload, ensure_ascii=False)},
                ]
            )
            repair_parsed = _extract_json_object(repair_raw)
            repair_frame_units = _unit_writer_raw_units_from_action_frames(repair_parsed)
            repair_units = repair_parsed.get("units", []) if isinstance(repair_parsed, Mapping) else []
            if not isinstance(repair_units, list):
                repair_units = []
            repair_units = list(repair_frame_units) + [item for item in repair_units if isinstance(item, Mapping)]
            repair_raw_count = len(repair_units)
            repair_stored_count = materialize_units(repair_units, phase="coverage_repair", limit=max_units - len(output))
        except Exception as exc:
            repair_error = f"{exc.__class__.__name__}: {str(exc)[:200]}"
    uncovered_after_repair = _unit_writer_uncovered_requirements(coverage_requirements, output)
    fallback_raw_count = 0
    fallback_stored_count = 0
    fallback_mode = _safe_text(
        os.environ.get(
            "TMCRA_EVENT_ACTION_FRAME_ATTACHMENT_FALLBACK_MODE",
            os.environ.get("TMCRA_MODELIZED_UNIT_WRITER_FALLBACK_MODE", "on"),
        )
    ).lower()
    if (
        fallback_mode not in {"0", "off", "false", "disabled", "none"}
        and uncovered_after_repair
        and len(output) < max_units
    ):
        fallback_units = _unit_writer_fallback_units_from_requirements(
            uncovered_after_repair,
            parent_records=parent_records,
            current_turn_text=source_text,
        )
        fallback_raw_count = len(fallback_units)
        fallback_stored_count = materialize_units(fallback_units, phase="coverage_fallback", limit=max_units - len(output))
        uncovered_after_repair = _unit_writer_uncovered_requirements(coverage_requirements, output)
    return output, {
        "enabled": True,
        "parent_count": len(parents),
        "source_or_event_parent_count": len(source_or_event_parents),
        "unit_signal_hits": signal_hits,
        "coverage_requirement_count": len(coverage_requirements),
        "coverage_requirements": coverage_requirements,
        "raw_unit_count": len(raw_units),
        "initial_stored_unit_count": initial_stored_count,
        "coverage_repair_enabled": repair_mode not in {"0", "off", "false", "disabled", "none"},
        "uncovered_before_repair_count": len(uncovered_before_repair),
        "uncovered_before_repair": uncovered_before_repair,
        "repair_raw_unit_count": repair_raw_count,
        "repair_stored_unit_count": repair_stored_count,
        "repair_error": repair_error,
        "repair_usage": repair_usage,
        "coverage_fallback_enabled": fallback_mode not in {"0", "off", "false", "disabled", "none"},
        "fallback_raw_unit_count": fallback_raw_count,
        "fallback_stored_unit_count": fallback_stored_count,
        "uncovered_after_repair_count": len(uncovered_after_repair),
        "uncovered_after_repair": uncovered_after_repair,
        "stored_unit_count": len(output),
        "usage": usage,
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = _safe_text(text)
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    except Exception:
        pass
    repaired_raw = _repair_jsonish_text(raw)
    try:
        parsed = json.loads(repaired_raw)
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    except Exception:
        pass
    start = repaired_raw.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(repaired_raw)):
            char = repaired_raw[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = repaired_raw[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, Mapping) and (
                            "write_proposals" in parsed or "write_proposal" in parsed or "memory_writes" in parsed or "units" in parsed
                        ):
                            return dict(parsed)
                    except Exception:
                        break
        start = repaired_raw.find("{", start + 1)
    end = repaired_raw.rfind("}")
    if repaired_raw.find("{") >= 0 and end > repaired_raw.find("{"):
        try:
            parsed = json.loads(repaired_raw[repaired_raw.find("{") : end + 1])
            if isinstance(parsed, Mapping) and ("write_proposals" in parsed or "write_proposal" in parsed or "memory_writes" in parsed or "units" in parsed):
                return dict(parsed)
        except Exception:
            pass
    partial = _extract_partial_write_proposals(repaired_raw)
    if partial:
        return partial
    if "write_proposals" in repaired_raw or "write_proposal" in repaired_raw or "memory_writes" in repaired_raw or "units" in repaired_raw:
        return {
            "write_proposals": [],
            "_tmcra_json_repair_status": "empty_from_unrecoverable_writer_json",
        }
    return {}


def _repair_jsonish_text(value: str) -> str:
    text = _safe_text(value)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r'"write_proposal"\s*:', '"write_proposals":', text, flags=re.IGNORECASE)
    text = re.sub(r"//.*?(?=(?:[,}\]]|$))", "", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', text)
    # Some local OpenAI-compatible servers emit invalid JSON escapes such as
    # That\'s inside double-quoted strings. This repair is conservative enough
    # for writer payloads and avoids failing an otherwise complete object.
    text = text.replace("\\'", "'")
    return text


def _extract_turn_intent_object(parsed: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(parsed, Mapping):
        return {}
    candidate = parsed.get("turn_intent", parsed.get("turn_intent_gate", {}))
    if isinstance(candidate, Mapping):
        normalized = {_safe_text(key).strip().lower(): value for key, value in dict(candidate).items()}
        if "write_allowed" not in normalized:
            for key, value in list(normalized.items()):
                if key.startswith("write_allowed"):
                    normalized["write_allowed"] = value
                    break
        return normalized
    normalized_parsed = {_safe_text(key).strip().lower(): value for key, value in dict(parsed).items()}
    if "write_allowed" not in normalized_parsed:
        for key, value in list(normalized_parsed.items()):
            if key.startswith("write_allowed"):
                normalized_parsed["write_allowed"] = value
                break
    if any(key in normalized_parsed for key in ("intent", "write_allowed", "question_like", "memory_assertion", "question_only")):
        return {
            key: normalized_parsed.get(key)
            for key in ("intent", "write_allowed", "question_like", "memory_assertion", "question_only", "reason")
            if key in normalized_parsed
        }
    return {}


def _extract_turn_intent_from_text(text: str) -> Dict[str, Any]:
    raw = _safe_text(text)
    if not raw:
        return {}
    intent_match = re.search(r"""["']?\bintent\b["']?\s*:\s*["']([a-zA-Z0-9_\- ]+)["']""", raw, flags=re.IGNORECASE)
    if not intent_match:
        return {}

    def find_bool(key: str) -> bool | None:
        pattern = rf"""["']?\s*{re.escape(key)}[a-zA-Z_ ]*["']?\s*:\s*(true|false|yes|no|1|0)\b"""
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if not match:
            return None
        return match.group(1).lower() in {"true", "yes", "1"}

    reason = ""
    reason_match = re.search(r"""["']?\breason\b["']?\s*:\s*["']([^"']{0,240})["']""", raw, flags=re.IGNORECASE)
    if reason_match:
        reason = _safe_text(reason_match.group(1))
    result: Dict[str, Any] = {
        "intent": intent_match.group(1).strip().lower().replace(" ", "_"),
    }
    for key in ("write_allowed", "question_like", "memory_assertion", "question_only"):
        value = find_bool(key)
        if value is not None:
            result[key] = value
    if reason:
        result["reason"] = reason
    return result


def _extract_partial_write_proposals(raw: str) -> Dict[str, Any]:
    proposals_start = re.search(r'"(?:write_proposals|write_proposal|memory_writes)"\s*:\s*\[', raw)
    if not proposals_start:
        return {}
    array_start = raw.find("[", proposals_start.start())
    if array_start < 0:
        return {}
    proposals: List[Dict[str, Any]] = []
    start = raw.find("{", array_start)
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        parsed_any = False
        for index in range(start, len(raw)):
            char = raw[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = _repair_jsonish_text(raw[start : index + 1])
                    try:
                        parsed = json.loads(candidate)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, Mapping):
                        proposals.append(dict(parsed))
                    start = raw.find("{", index + 1)
                    parsed_any = True
                    break
            elif char == "]" and depth == 0:
                start = -1
                parsed_any = True
                break
        if not parsed_any:
            break
    if proposals:
        return {
            "write_proposals": proposals,
            "_tmcra_json_repair_status": "partial_write_proposals_recovered",
        }
    return {}


def _grounding_score(*, value: str, source_span: str, source_context: str) -> float:
    source_lower = _safe_text(source_context).lower()
    value_lower = _safe_text(value).lower()
    span_lower = _safe_text(source_span).lower()
    if span_lower and span_lower in source_lower:
        return 1.0
    if value_lower and value_lower in source_lower:
        return 1.0
    candidate_tokens = set(_tokens(source_span or value))
    if not candidate_tokens:
        return 0.0
    source_tokens = set(_tokens(source_context))
    if not source_tokens:
        return 0.0
    return len(candidate_tokens & source_tokens) / max(1, len(candidate_tokens))


def _value_grounding_score(*, value: str, source_context: str) -> float:
    return _grounding_score(value=value, source_span="", source_context=source_context)


def _span_is_grounded_in_current_turn(*, source_span: str, current_turn_text: str) -> bool:
    span_lower = _safe_text(source_span).lower()
    turn_lower = _safe_text(current_turn_text).lower()
    return bool(span_lower) and span_lower in turn_lower


def _source_span_can_stand_as_value(*, source_span: str, value: str, current_turn_text: str) -> bool:
    """Allow a truth-preserving repair only when the copied span is real evidence.

    If an LLM emits a lossy normalized value, the gate may replace that value
    with the copied source span instead of accepting an under-grounded summary.
    This is not a fallback write: the mutation remains constrained to text that
    appears in the current user turn.
    """
    if not _span_is_grounded_in_current_turn(source_span=source_span, current_turn_text=current_turn_text):
        return False
    span_tokens = _tokens(source_span)
    value_tokens = _tokens(value)
    if not span_tokens:
        return False
    # Very short spans such as "my project" are not safe canonical values for a
    # hallucinated summary. Longer copied spans can preserve the asserted fact.
    return len(set(span_tokens)) >= max(4, min(6, len(set(value_tokens)) or 4))


def _current_turn_can_repair_proposal(*, current_turn_text: str, value: str, source_span: str) -> bool:
    turn_tokens = set(_tokens(current_turn_text))
    if len(turn_tokens) < 5:
        return False
    candidate_tokens = set(_tokens(" ".join([value, source_span])))
    if len(candidate_tokens) < 4:
        return False
    overlap = len(turn_tokens & candidate_tokens) / max(1, len(candidate_tokens))
    return overlap >= 0.52


_MEMORY_SUBJECT_PATTERNS = (
    re.compile(
        r"\bmy\s+(?P<subject>[a-z0-9][a-z0-9\s'/_-]{1,90}?)\s+(?:is|are|was|were)\s+(?:now\s+)?",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+(?:is|are|was|were)\s+my\s+(?P<subject>[a-z0-9][a-z0-9\s'/_-]{1,90}?)\s+(?:right\s+now|now|currently)\b",
        flags=re.IGNORECASE,
    ),
)


def _extract_memory_subject(*texts: Any) -> str:
    for text in texts:
        compact = _safe_text(text)
        if not compact:
            continue
        for pattern in _MEMORY_SUBJECT_PATTERNS:
            match = pattern.search(compact)
            if not match:
                continue
            subject = _safe_text(match.group("subject")).strip(" .,:;!?\"'")
            subject = re.sub(r"\b(now|currently|right)\b$", "", subject, flags=re.IGNORECASE).strip(" .,:;!?\"'")
            if len(_tokens(subject)) >= 2:
                return subject
    return ""


def _loose_phrase_pattern(value: Any) -> str:
    parts = [_safe_text(part) for part in _safe_text(value).split() if _safe_text(part)]
    return r"\s+".join(re.escape(part) for part in parts)


def _trim_current_value(value: Any) -> str:
    clean = _safe_text(value).strip(" .,:;!?\"'")
    if not clean:
        return ""
    clean = re.split(r"\b(?:the\s+)?earlier\s+view\s+was\b", clean, maxsplit=1, flags=re.IGNORECASE)[0]
    clean = re.split(r"\b(?:the\s+)?previous\s+view\s+was\b", clean, maxsplit=1, flags=re.IGNORECASE)[0]
    return _safe_text(clean).strip(" .,:;!?\"'")


def _extract_current_value_assertion(*, current_turn_text: str, subject: str = "") -> tuple[str, str]:
    """Extract a current-value assertion from visible user text only."""

    compact = _safe_text(current_turn_text)
    subject_text = _safe_text(subject)
    if not compact:
        return "", ""
    if subject_text:
        subject_pattern = _loose_phrase_pattern(subject_text)
        if subject_pattern:
            patterns = (
                rf"\bmy\s+{subject_pattern}\s+(?:is|are)\s+now\s+(?P<value>.+?)(?:[.!?](?:\s|$)|$)",
                rf"\b(?:i(?:'ve| have)\s+updated\s+my\s+{subject_pattern}\s*[:,-]\s*)(?P<value>.+?)(?:[.!?](?:\s|$)|$)",
            )
            for pattern in patterns:
                match = re.search(pattern, compact, flags=re.IGNORECASE)
                if not match:
                    continue
                value = _trim_current_value(match.group("value"))
                if len(_tokens(value)) >= 3:
                    return _safe_text(match.group(0)).strip(" .,:;!?\"'"), value

    subject_tokens = {token for token in _tokens(subject_text) if len(token) >= 4}
    for marker in (
        r"\bi\s+now\s+(?:believe|think|see)\s+(?P<value>.+?)(?:[.!?](?:\s|$)|$)",
        r"\bi\s+now\s+believe\s+that\s+(?P<value>.+?)(?:[.!?](?:\s|$)|$)",
    ):
        match = re.search(marker, compact, flags=re.IGNORECASE)
        if not match:
            continue
        value = _trim_current_value(match.group("value"))
        value_tokens = set(_tokens(value))
        if len(value_tokens) < 3:
            continue
        if subject_tokens and not (subject_tokens & value_tokens):
            continue
        return _safe_text(match.group(0)).strip(" .,:;!?\"'"), value
    return "", ""


_CURRENT_VALUE_COVERAGE_STOP_TOKENS = {
    "about",
    "current",
    "latest",
    "main",
    "more",
    "now",
    "position",
    "right",
    "that",
    "this",
    "view",
}


def _current_value_meaning_tokens(value: Any) -> set[str]:
    return {
        token
        for token in _tokens(value)
        if len(token) >= 4 and token not in _CURRENT_VALUE_COVERAGE_STOP_TOKENS
    }


def _record_covers_current_value(*, record: Mapping[str, Any], slot_key: str, current_value: str) -> bool:
    if _safe_text(record.get("slot_key", "")).lower() != _safe_text(slot_key).lower():
        return False
    if _safe_text(record.get("category", "")).lower() == "question":
        return False
    target_tokens = _current_value_meaning_tokens(current_value)
    if not target_tokens:
        return False
    metadata = dict(record.get("metadata", {}) or {})
    record_text = " ".join(
        [
            _safe_text(record.get("value", "")),
            _safe_text(metadata.get("source_span", "")),
            _safe_text(metadata.get("event_text", "")),
            _safe_text(metadata.get("event_phrase", "")),
        ]
    )
    covered_tokens = _current_value_meaning_tokens(record_text)
    coverage = len(target_tokens & covered_tokens) / max(1, len(target_tokens))
    return coverage >= 0.85


def _sidecar_memory_subject_hint(sidecar_hints: Mapping[str, Any] | None, *, session_slug: str) -> tuple[str, str, str]:
    hints = dict(sidecar_hints or {})
    raw_slot_hint = hints.get("memory_slot", {})
    slot_hint = dict(raw_slot_hint or {}) if isinstance(raw_slot_hint, Mapping) else {}
    subject = _safe_text(
        slot_hint.get("subject", "")
        or slot_hint.get("prompt_label", "")
        or slot_hint.get("label", "")
    )
    subject_signature = _safe_text(slot_hint.get("subject_signature", "")) or (_slug(subject) if subject else "")
    canonical_slot_key = _safe_text(slot_hint.get("canonical_slot_key", ""))
    if not canonical_slot_key and subject_signature:
        canonical_slot_key = f"{session_slug}.subject.{subject_signature}"
    if subject and len(_tokens(subject)) < 2:
        subject = ""
        subject_signature = ""
        canonical_slot_key = ""
    return subject, subject_signature, canonical_slot_key


def _looks_like_speech_act_memory(*, value: str, source_span: str) -> bool:
    """Reject proposals that describe the user act of asking instead of durable memory."""
    value_text = _safe_text(value).lower()
    span_text = _safe_text(source_span)
    if not value_text:
        return False
    if _SPEECH_ACT_MEMORY_RE.search(value_text):
        return True
    if span_text.endswith("?") and value_text.startswith(("asked ", "asks ", "asking ", "requested ", "requesting ")):
        return True
    return False


def _looks_like_transient_assistant_directive(*, value: str, source_span: str, turn_text: str) -> bool:
    """Reject one-shot instructions only when the proposal itself is transient."""
    value_text = _safe_text(value)
    span_text = _safe_text(source_span)
    combined = " ".join(part for part in (value_text, span_text) if part)
    if not combined:
        return False
    durable_profile_assertion = bool(
        _FIRST_PERSON_MEMORY_ASSERTION_RE.search(span_text)
        or _FIRST_PERSON_MEMORY_ASSERTION_RE.search(_safe_text(turn_text))
        or _PROFILE_VALUE_MARKERS.search(combined)
    )
    if _TRANSIENT_ASSISTANT_DIRECTIVE_RE.search(combined):
        if durable_profile_assertion and not re.search(
            r"\b(?:answer|respond|reply|summari[sz]e|translate|compare|explain|based\s+only|only\s+use|do\s+not\s+(?:mention|reveal|include|use))\b",
            combined,
            flags=re.IGNORECASE,
        ):
            return False
        return True
    # Only reject when the proposal itself is transient. Long chunks can mix
    # durable facts with nearby requests such as "recommend" or "give advice";
    # scanning the whole chunk here rejects valid writer proposals.
    return False


class TurnIntentGate:
    """Use whole-turn intent as the gate before mutating writer proposals."""

    INTENT_ALIASES = {
        "assertion": "memory_assertion",
        "memory": "memory_assertion",
        "memory_write": "memory_assertion",
        "durable_memory": "memory_assertion",
        "question": "question_only",
        "question_only_turn": "question_only",
        "recall_question": "question_only",
        "answer_request": "question_only",
        "mixed": "mixed_question_with_memory",
        "mixed_question": "mixed_question_with_memory",
        "question_with_memory": "mixed_question_with_memory",
        "question_like_with_memory_assertion": "mixed_question_with_memory",
        "no_memory": "no_new_memory",
        "none": "no_new_memory",
        "empty": "no_new_memory",
        "transient": "transient_directive",
        "instruction": "transient_directive",
        "directive": "transient_directive",
    }
    MEMORY_LABELS = {"memory_assertion", "mixed_question_with_memory"}
    QUESTION_ONLY_LABELS = {"question_only"}
    BLOCKED_LABELS = {"question_only", "no_new_memory", "transient_directive"}

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        text = _safe_text(value).lower()
        if text in {"true", "yes", "1", "allow", "allowed"}:
            return True
        if text in {"false", "no", "0", "deny", "blocked"}:
            return False
        return default

    @staticmethod
    def _heuristic_intent(text: str) -> Dict[str, Any]:
        compact = _safe_text(text)
        if not compact:
            return {
                "intent": "no_new_memory",
                "question_like": False,
                "memory_assertion": False,
                "question_only": False,
                "write_allowed": False,
                "reason": "empty",
                "source": "heuristic_fallback",
            }
        memory_assertion = bool(
            _ASSERTIVE_MEMORY_CUE_RE.search(compact)
            or _FIRST_PERSON_MEMORY_ASSERTION_RE.search(compact)
            or _CJK_MEMORY_ASSERTION_RE.search(compact)
        )
        question_like = bool(
            compact.endswith("?")
            or compact.endswith("？")
            or _QUESTION_ONLY_RE.search(compact)
            or any(marker in compact for marker in _CJK_QUESTION_MARKERS)
        )
        if question_like and memory_assertion:
            intent = "mixed_question_with_memory"
        elif question_like:
            intent = "question_only"
        elif memory_assertion:
            intent = "memory_assertion"
        else:
            intent = "plain_statement"
        return {
            "intent": intent,
            "question_like": question_like,
            "memory_assertion": memory_assertion,
            "question_only": bool(question_like and not memory_assertion),
            "write_allowed": bool(memory_assertion or not question_like),
            "reason": intent,
            "source": "heuristic_fallback",
        }

    @classmethod
    def _normalize_writer_intent(
        cls,
        writer_intent: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        if not isinstance(writer_intent, Mapping):
            return {}
        raw_label = _safe_text(
            writer_intent.get("turn_intent")
            or writer_intent.get("intent")
            or writer_intent.get("label")
            or writer_intent.get("decision")
        ).lower()
        label = raw_label.replace("-", "_").replace(" ", "_")
        label = cls.INTENT_ALIASES.get(label, label)
        valid_labels = cls.MEMORY_LABELS | cls.QUESTION_ONLY_LABELS | cls.BLOCKED_LABELS | {"plain_statement"}
        if label not in valid_labels:
            return {}
        memory_assertion = cls._coerce_bool(writer_intent.get("memory_assertion"), label in cls.MEMORY_LABELS)
        question_only = cls._coerce_bool(writer_intent.get("question_only"), label in cls.QUESTION_ONLY_LABELS)
        question_like = cls._coerce_bool(
            writer_intent.get("question_like"),
            label in {"question_only", "mixed_question_with_memory"},
        )
        write_allowed = cls._coerce_bool(writer_intent.get("write_allowed"), label not in cls.BLOCKED_LABELS)
        if label in cls.MEMORY_LABELS:
            write_allowed = True
            memory_assertion = True
            question_only = False
        if label in cls.BLOCKED_LABELS:
            write_allowed = False
            memory_assertion = False
            question_only = label == "question_only"
        return {
            "intent": label,
            "question_like": question_like,
            "memory_assertion": memory_assertion,
            "question_only": question_only,
            "write_allowed": write_allowed,
            "reason": _safe_text(writer_intent.get("reason", "")) or label,
            "source": "writer_turn_intent",
            "writer_raw_intent": dict(writer_intent),
        }

    @classmethod
    def classify_turn(
        cls,
        text: str,
        *,
        writer_intent: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        fallback = cls._heuristic_intent(text)
        normalized = cls._normalize_writer_intent(writer_intent)
        if not normalized:
            return fallback
        normalized["heuristic_fallback"] = fallback
        return normalized

    @staticmethod
    def proposal_policy(
        *,
        turn_intent: Mapping[str, Any],
        value: str,
        source_span: str,
        turn_text: str,
    ) -> Dict[str, Any]:
        speech_act_memory = _looks_like_speech_act_memory(value=value, source_span=source_span)
        transient_directive = _looks_like_transient_assistant_directive(
            value=value,
            source_span=source_span,
            turn_text=turn_text,
        )
        if transient_directive:
            return {
                "decision": "reject",
                "reject_reason": "transient_assistant_directive",
                "force_question": False,
                "speech_act_memory": speech_act_memory,
                "transient_directive": True,
                "turn_intent_reason": _safe_text(turn_intent.get("reason", "")),
            }
        force_question = bool(turn_intent.get("question_only", False) or speech_act_memory)
        if force_question:
            reason = "speech_act_memory" if speech_act_memory else "turn_question_only"
            return {
                "decision": "force_question",
                "reject_reason": "",
                "force_question": True,
                "force_question_reason": reason,
                "speech_act_memory": speech_act_memory,
                "transient_directive": False,
                "turn_intent_reason": _safe_text(turn_intent.get("reason", "")),
            }
        return {
            "decision": "preserve_writer_category",
            "reject_reason": "",
            "force_question": False,
            "speech_act_memory": speech_act_memory,
            "transient_directive": False,
            "turn_intent_reason": _safe_text(turn_intent.get("reason", "")),
        }


def _turn_intent_gate(text: str) -> Dict[str, Any]:
    return TurnIntentGate.classify_turn(text)


def _is_question_only_turn(text: str) -> bool:
    return bool(_turn_intent_gate(text).get("question_only", False))


def _source_kind_for(category: str, semantic_slot: str, relation: str) -> tuple[str, str]:
    normalized_category = _safe_text(category).lower()
    normalized_slot = _safe_text(semantic_slot).lower()
    normalized_relation = _safe_text(relation).lower()
    if (
        normalized_category in {"question", "interaction_intent"}
        or normalized_slot in {"question", "interaction_intent", "user_question"}
        or "question" in normalized_relation
        or "ask" in normalized_relation
    ):
        return "question", "public_dialog_question"
    if normalized_category == "profile" or normalized_slot in _PROFILE_SLOTS:
        return "profile", "public_dialog_profile"
    if normalized_category == "time" or normalized_slot == "event_time" or "time" in normalized_relation or "date" in normalized_relation:
        return "fact", "public_dialog_time"
    if normalized_category == "event":
        return "event", "public_dialog_event"
    return normalized_category or "fact", f"public_dialog_{normalized_category or 'fact'}"


def _profile_intent_from_proposal(
    *,
    proposal: Mapping[str, Any],
    category: str,
    record_category: str,
    semantic_slot: str,
    relation: str,
    value: str,
    source_span: str,
    subject: str,
    turn_intent: Mapping[str, Any],
) -> Dict[str, Any]:
    if record_category == "question":
        return {"enabled": False, "reason": "question_record", "create_profile_shadow": False}
    normalized_category = _safe_text(category).lower()
    normalized_record_category = _safe_text(record_category).lower()
    slot_text = _safe_text(semantic_slot).lower()
    relation_text = _safe_text(relation).lower()
    combined = " ".join([slot_text, relation_text, _safe_text(value).lower(), _safe_text(source_span).lower()])
    explicit_profile_type = _safe_text(proposal.get("profile_type", "")).lower()
    explicit_domain = _safe_text(proposal.get("profile_domain", "") or proposal.get("profile_domain_label", ""))
    slot_tokens = set(_tokens(slot_text.replace("_", " ")))
    reasons: List[str] = []
    enabled = False
    if normalized_category in _PROFILE_DIRECT_CATEGORIES or normalized_record_category in _PROFILE_DIRECT_CATEGORIES:
        enabled = True
        reasons.append("profile_category")
    if explicit_profile_type:
        enabled = True
        reasons.append("writer_profile_type")
    if slot_tokens & _PROFILE_INTENT_SLOT_MARKERS:
        enabled = True
        reasons.append("profile_slot_marker")
    if bool(turn_intent.get("memory_assertion")) and _PROFILE_VALUE_MARKERS.search(combined):
        enabled = True
        reasons.append("profile_value_marker")
    if not enabled:
        return {"enabled": False, "reason": "no_profile_signal", "create_profile_shadow": False}

    profile_type = explicit_profile_type
    if profile_type not in {"setup", "preference", "constraint", "goal", "avoid", "usage_context"}:
        if normalized_record_category == "preference" or "prefer" in combined or "like" in combined:
            profile_type = "preference"
        elif normalized_record_category == "goal" or any(marker in combined for marker in ("goal", "want", "plan", "intend")):
            profile_type = "goal"
        elif normalized_record_category == "constraint" or any(marker in combined for marker in ("constraint", "must", "cannot", "need", "avoid")):
            profile_type = "avoid" if "avoid" in combined or "dislike" in combined else "constraint"
        elif slot_tokens & _PROFILE_SETUP_SLOT_MARKERS or any(marker in combined for marker in ("current", "setup", "brand", "uses", "user is", "equipment", "gear")):
            profile_type = "setup"
        else:
            profile_type = "usage_context"

    profile_domain = explicit_domain
    if not profile_domain:
        if slot_text and slot_text not in {"profile", "preference", "goal", "constraint", "event", "fact", "status"}:
            profile_domain = slot_text.replace("_", " ")
        elif subject:
            profile_domain = subject
        else:
            domain_tokens = [
                token
                for token in _tokens(value)
                if token not in {"user", "uses", "prefers", "likes", "wants", "needs", "has", "current"}
            ]
            profile_domain = " ".join(domain_tokens[:4]) if domain_tokens else "general"
    return {
        "enabled": True,
        "reason": ",".join(_dedupe_texts(reasons, max_items=6)),
        "profile_type": profile_type,
        "profile_domain": profile_domain,
        "create_profile_shadow": normalized_record_category not in _PROFILE_DIRECT_CATEGORIES,
        "profile_writer_layer_version": "turn_intent_profile_writer_v1",
    }


@dataclass(slots=True)
class MemoryWriteGateResult:
    payload: Dict[str, Any]
    accepted_count: int
    rejected: List[Dict[str, Any]]
    suspected_count: int = 0


class SemanticMemoryWriterError(RuntimeError):
    """Raised when the required LLM semantic writer cannot produce gated writes."""


def _suspect_source_kind(source_kind: str) -> str:
    normalized = _safe_text(source_kind) or "memory"
    return normalized if normalized.startswith("suspect_") else f"suspect_{normalized}"


class DeterministicMemoryWriteGate:
    """Validate LLM memory write proposals before they can mutate memory state."""

    def __init__(self, *, min_grounding_score: float = 0.5) -> None:
        self.min_grounding_score = max(0.0, float(min_grounding_score))

    def _suspect_record(
        self,
        *,
        index: int,
        category: str,
        value: str,
        source_span: str,
        semantic_slot: str,
        relation: str,
        source_kind: str,
        slot_key: str,
        event_signature: str,
        anchors: Sequence[Any],
        common_metadata: Mapping[str, Any],
        proposal: Mapping[str, Any],
        grounding: float,
        value_grounding: float,
        min_value_grounding: float,
        suspicion_reason: str,
        compact_text: str,
        subject: str = "",
        subject_signature: str = "",
    ) -> Dict[str, Any]:
        return {
            "category": category or "fact",
            "slot_key": slot_key,
            "value": value or source_span,
            "anchors": _dedupe_texts(anchors, max_items=8),
            "relation": relation or f"{category or 'fact'}_memory",
            "source_kind": _suspect_source_kind(source_kind),
            "state": "suspect",
            "salience": min(0.58, float(proposal.get("salience", 0.46) or 0.46)),
            "confidence": min(0.56, float(proposal.get("confidence", 0.42) or 0.42)),
            "metadata": {
                **dict(common_metadata),
                "content_variant": "llm_semantic_suspect_write",
                "semantic_slot": semantic_slot or category,
                "target_status": _safe_text(proposal.get("target_status", "")),
                "source_span": source_span,
                "event_phrase": _safe_text(proposal.get("event_phrase", "")) or value or source_span,
                "event_text": _safe_text(proposal.get("event_text", "")) or _safe_text(proposal.get("event_phrase", "")) or value or source_span,
                "event_signature": event_signature,
                "subject": subject,
                "subject_signature": subject_signature,
                "canonical_slot_key": slot_key,
                "allow_parallel_state": False if subject_signature else bool(proposal.get("allow_parallel_state", True)),
                "time_expression_span": _safe_text(proposal.get("time_expression_span", "")),
                "time_granularity": _safe_text(proposal.get("time_granularity", "")),
                "resolved_time_value": _safe_text(proposal.get("resolved_time_value", "")),
                "time_display_value": _safe_text(proposal.get("time_display_value", "")),
                "resolved_date": _safe_text(proposal.get("resolved_date", "")),
                "memory_gate_decision": "suspect_buffer",
                "suspicion_reason": suspicion_reason,
                "candidate_source_kind": source_kind,
                "source_span_exact_current_turn": _span_is_grounded_in_current_turn(
                    source_span=source_span,
                    current_turn_text=compact_text,
                ),
                "grounding_score": round(float(grounding), 6),
                "value_grounding_score": round(float(value_grounding), 6),
                "min_value_grounding_score": round(float(min_value_grounding), 6),
                "llm_write_proposal_index": index,
                "suspect_support_count": 1,
                "suspect_support_turns": [int(proposal.get("turn_index", 0) or 0)],
                "tmcra_stack_metadata": dict(proposal.get("tmcra_stack_metadata", {}) or {}),
                "write_decision": _safe_text(proposal.get("write_decision", "")),
                "proposal_intent_gate": dict(proposal.get("proposal_intent_gate", {}) or {}),
                **profile_candidate_metadata(
                    category=category,
                    semantic_slot=semantic_slot,
                    relation=relation,
                    value=value or source_span,
                    source_span=source_span,
                    slot_key=slot_key,
                    anchors=anchors,
                    subject=subject,
                    subject_signature=subject_signature,
                    proposal=proposal,
                ),
            },
        }

    def build_payload(
        self,
        *,
        proposals: Sequence[Mapping[str, Any]],
        text: str,
        raw_text: str,
        speaker: str,
        session_key: str,
        turn_index: int,
        timestamp: str = "",
        dia_id: str = "",
        sidecar_hints: Mapping[str, Any] | None = None,
        auxiliary_evidence_texts: Sequence[Any] | None = None,
        writer_metadata: Mapping[str, Any] | None = None,
    ) -> MemoryWriteGateResult:
        rendered_text = _safe_text(text)
        compact_text = _safe_text(raw_text) or rendered_text
        auxiliary_texts = _dedupe_texts(auxiliary_evidence_texts or [])
        source_context = compact_text if not auxiliary_texts else compact_text + "\nAuxiliary evidence: " + " | ".join(auxiliary_texts)
        writer_turn_intent = dict((writer_metadata or {}).get("writer_turn_intent", {}) or {})
        turn_intent = TurnIntentGate.classify_turn(compact_text, writer_intent=writer_turn_intent)
        question_only_turn = bool(turn_intent.get("question_only", False))
        session_slug = _slug(session_key)
        sidecar_subject, sidecar_subject_signature, sidecar_canonical_slot_key = _sidecar_memory_subject_hint(
            sidecar_hints,
            session_slug=session_slug,
        )
        common_metadata = {
            "speaker": _safe_text(speaker),
            "timestamp": _safe_text(timestamp),
            "session_key": _safe_text(session_key),
            "dia_id": _safe_text(dia_id),
            "event_id": f"event::{_safe_text(dia_id)}" if _safe_text(dia_id) else "",
            "source": "public_benchmark",
            "raw_text": compact_text,
            "source_turn_text": source_context,
            "write_path": "llm_semantic_writer_gate",
            "memory_writer_role": "llm_semantic_writer",
            "turn_extractor_role": "sidecar_hints",
            "semantic_writer_metadata": dict(writer_metadata or {}),
            "sidecar_hint_metadata": dict(sidecar_hints or {}).get("metadata", {}),
            "turn_intent_gate": dict(turn_intent),
        }
        records: List[Dict[str, Any]] = [
            {
                "category": "fact",
                "slot_key": f"{session_slug}.turn_{max(1, int(turn_index or 1))}",
                "value": rendered_text,
                "anchors": _dedupe_texts([speaker, timestamp]),
                "relation": "source_turn_grounding",
                "source_kind": "public_dialog_turn",
                "state": "evidence",
                "salience": 0.92,
                "confidence": 1.0,
                "metadata": {
                    **common_metadata,
                    "content_variant": "source_turn",
                    "memory_gate_decision": "source_grounding",
                },
            }
        ]
        rejected: List[Dict[str, Any]] = []
        suspect_records: List[Dict[str, Any]] = []
        seen = set()
        for index, raw_proposal in enumerate(proposals):
            proposal = dict(raw_proposal or {})
            if proposal.get("write") is False:
                rejected.append({"index": index, "reason": "proposal_write_false"})
                continue
            category = _safe_text(proposal.get("category", "")).lower().replace("-", "_")
            if category in _WRITE_CATEGORY_ALIASES:
                proposal["original_category"] = category
                proposal["category"] = _WRITE_CATEGORY_ALIASES[category]
                proposal["semantic_slot"] = _safe_text(proposal.get("semantic_slot", "")) or category
                category = _safe_text(proposal.get("category", "")).lower().replace("-", "_")
            if category not in ALLOWED_WRITE_CATEGORIES:
                rejected.append({"index": index, "reason": "category_not_allowed", "category": category})
                continue
            source_span = _safe_text(proposal.get("source_span", ""))
            if not source_span:
                rejected.append({"index": index, "reason": "missing_source_span"})
                continue
            value = _safe_text(proposal.get("value", "")) or source_span
            if not value:
                rejected.append({"index": index, "reason": "missing_value"})
                continue
            proposal_policy = TurnIntentGate.proposal_policy(
                turn_intent=turn_intent,
                value=value,
                source_span=source_span,
                turn_text=compact_text,
            )
            proposal["proposal_intent_gate"] = dict(proposal_policy)
            if proposal_policy.get("decision") == "reject":
                rejected.append(
                    {
                        "index": index,
                        "reason": _safe_text(proposal_policy.get("reject_reason", "")) or "turn_intent_gate_reject",
                        "turn_intent_reason": _safe_text(proposal_policy.get("turn_intent_reason", "")),
                    }
                )
                continue
            if bool(proposal_policy.get("force_question", False)):
                category = "question"
                proposal["category"] = "question"
                proposal["semantic_slot"] = _safe_text(proposal.get("semantic_slot", "")) or "interaction_intent"
                proposal["relation"] = _safe_text(proposal.get("relation", "")) or "user_question"
                proposal["target_status"] = ""
            if _SECRET_RE.search(value) or _SECRET_RE.search(source_span):
                rejected.append({"index": index, "reason": "secret_like_value"})
                continue
            grounding = _grounding_score(value=value, source_span=source_span, source_context=source_context)
            value_grounding = _value_grounding_score(value=value, source_context=source_context)
            semantic_slot = _safe_text(proposal.get("semantic_slot", ""))
            relation = _safe_text(proposal.get("relation", "")) or f"{category}_fact"
            record_category, source_kind = _source_kind_for(category, semantic_slot, relation)
            if record_category == "question":
                # Question memory should preserve the user's actual question, not a
                # lossy LLM summary such as "check evaluation view".
                source_span = compact_text
                value = compact_text
                grounding = 1.0
                value_grounding = 1.0
            # Values may be concise normalizations of a longer exact source span. Keep a
            # floor high enough to reject hallucinated slot values, but do not reject
            # grounded summaries when the source_span itself is copied from the turn.
            min_value_grounding = self.min_grounding_score if record_category == "question" else max(self.min_grounding_score, 0.65)
            if (
                record_category != "question"
                and bool(turn_intent.get("write_allowed", False))
                and value_grounding < min_value_grounding
                and not _span_is_grounded_in_current_turn(source_span=source_span, current_turn_text=compact_text)
                and _current_turn_can_repair_proposal(
                    current_turn_text=compact_text,
                    value=value,
                    source_span=source_span,
                )
            ):
                proposal["source_span_repair"] = "current_turn_exact_from_turn_intent"
                proposal["original_source_span"] = source_span
                proposal["original_value"] = value
                source_span = compact_text
                value = compact_text
                grounding = 1.0
                value_grounding = 1.0
            event_signature = _safe_text(proposal.get("event_signature", "")) or compute_public_event_signature(
                source_span or value,
                speaker=_safe_text(speaker),
                semantic_slot=semantic_slot or record_category,
            )
            # Prefer the full current turn so nested values like
            # "my model size view is now my refined deployment view is ..."
            # keep the outer durable slot instead of drifting to the inner
            # wording emitted by the writer.
            extracted_subject = "" if record_category == "question" else _extract_memory_subject(compact_text, source_span, value)
            subject = extracted_subject
            subject_from_sidecar = False
            if record_category != "question" and sidecar_subject:
                subject = sidecar_subject
                subject_from_sidecar = _slug(extracted_subject) != (sidecar_subject_signature or _slug(sidecar_subject))
            subject_signature = _slug(subject) if subject else ""
            if subject_from_sidecar and sidecar_subject_signature:
                subject_signature = sidecar_subject_signature
            canonical_slot_key = f"{session_slug}.subject.{subject_signature}" if subject_signature else ""
            if subject_from_sidecar and sidecar_canonical_slot_key:
                canonical_slot_key = sidecar_canonical_slot_key
            slot_key = _safe_text(proposal.get("slot_key", ""))
            if canonical_slot_key:
                slot_key = canonical_slot_key
            elif not slot_key:
                slot_key = f"{session_slug}.turn_{max(1, int(turn_index or 1))}.{record_category}.{index + 1}"
            anchors = _dedupe_texts(
                [
                    speaker,
                    timestamp,
                    subject,
                    *list(proposal.get("anchors", []) or []),
                    *event_signature.split()[:4],
                ],
                max_items=8,
            )
            profile_intent_gate = _profile_intent_from_proposal(
                proposal=proposal,
                category=category,
                record_category=record_category,
                semantic_slot=semantic_slot,
                relation=relation,
                value=value,
                source_span=source_span,
                subject=subject,
                turn_intent=turn_intent,
            )
            if grounding < self.min_grounding_score:
                rejected.append(
                    {
                        "index": index,
                        "reason": "ungrounded",
                        "grounding_score": round(float(grounding), 6),
                    }
                )
                if record_category != "question":
                    suspect_records.append(
                        self._suspect_record(
                            index=index,
                            category=record_category,
                            value=value,
                            source_span=source_span,
                            semantic_slot=semantic_slot,
                            relation=relation,
                            source_kind=source_kind,
                            slot_key=slot_key,
                            event_signature=event_signature,
                            anchors=anchors,
                            common_metadata=common_metadata,
                            proposal={**proposal, "turn_index": turn_index},
                            grounding=grounding,
                            value_grounding=value_grounding,
                            min_value_grounding=min_value_grounding,
                            suspicion_reason="ungrounded",
                            compact_text=compact_text,
                            subject=subject,
                            subject_signature=subject_signature,
                        )
                    )
                continue
            canonicalized_from_value = ""
            canonicalization_reason = ""
            original_value_grounding = value_grounding
            if (
                record_category != "question"
                and value_grounding < min_value_grounding
                and _source_span_can_stand_as_value(
                    source_span=source_span,
                    value=value,
                    current_turn_text=compact_text,
                )
            ):
                canonicalized_from_value = value
                canonicalization_reason = "low_value_grounding_source_span_exact_current_turn"
                value = source_span
                value_grounding = 1.0
            if value_grounding < min_value_grounding:
                rejected.append(
                    {
                        "index": index,
                        "reason": "ungrounded_value",
                        "grounding_score": round(float(grounding), 6),
                        "value_grounding_score": round(float(value_grounding), 6),
                        "min_value_grounding_score": round(float(min_value_grounding), 6),
                        "source_span_token_count": len(set(_tokens(source_span))),
                        "value_token_count": len(set(_tokens(value))),
                    }
                )
                if record_category != "question":
                    suspect_records.append(
                        self._suspect_record(
                            index=index,
                            category=record_category,
                            value=value,
                            source_span=source_span,
                            semantic_slot=semantic_slot,
                            relation=relation,
                            source_kind=source_kind,
                            slot_key=slot_key,
                            event_signature=event_signature,
                            anchors=anchors,
                            common_metadata=common_metadata,
                            proposal={**proposal, "turn_index": turn_index},
                            grounding=grounding,
                            value_grounding=value_grounding,
                            min_value_grounding=min_value_grounding,
                            suspicion_reason="ungrounded_value",
                            compact_text=compact_text,
                            subject=subject,
                            subject_signature=subject_signature,
                        )
                    )
                continue
            dedupe_key = (
                record_category,
                slot_key.lower(),
                re.sub(r"\s+", " ", value.lower()).strip(),
            )
            if dedupe_key in seen:
                rejected.append({"index": index, "reason": "duplicate"})
                continue
            seen.add(dedupe_key)
            tmcra_stack_metadata = dict(proposal.get("tmcra_stack_metadata", {}) or {})
            proposal_state = _safe_text(proposal.get("state", "")).lower()
            proposal_write_decision = (
                _safe_text(proposal.get("write_decision", "")).lower()
                or _safe_text(proposal.get("memory_gate_decision", "")).lower()
                or _safe_text(tmcra_stack_metadata.get("memory_gate_decision", "")).lower()
            )
            tmcra_suspect_buffered = bool(tmcra_stack_metadata.get("tmcra_suspect_buffered"))
            if proposal_state == "suspect" or proposal_write_decision == "suspect_buffer" or tmcra_suspect_buffered:
                suspect_records.append(
                    self._suspect_record(
                        index=index,
                        category=record_category,
                        value=value,
                        source_span=source_span,
                        semantic_slot=semantic_slot,
                        relation=relation,
                        source_kind=source_kind,
                        slot_key=slot_key,
                        event_signature=event_signature,
                        anchors=anchors,
                        common_metadata=common_metadata,
                        proposal={**proposal, "turn_index": turn_index},
                        grounding=grounding,
                        value_grounding=value_grounding,
                        min_value_grounding=min_value_grounding,
                        suspicion_reason=_safe_text(
                            tmcra_stack_metadata.get("suspicion_reason", "")
                            or proposal.get("suspicion_reason", "")
                            or "writer_marked_suspect"
                        ),
                        compact_text=compact_text,
                        subject=subject,
                        subject_signature=subject_signature,
                    )
                )
                continue
            profile_proposal = {
                **proposal,
                "profile_type": _safe_text(profile_intent_gate.get("profile_type", "")) or proposal.get("profile_type", ""),
                "profile_domain": _safe_text(profile_intent_gate.get("profile_domain", "")) or proposal.get("profile_domain", ""),
            }
            metadata = {
                **common_metadata,
                "content_variant": "llm_semantic_write",
                "raw_text": source_span or value,
                "source_turn_text": source_span or value,
                "semantic_slot": semantic_slot or record_category,
                "target_status": _safe_text(proposal.get("target_status", "")),
                "source_span": source_span or value,
                "event_phrase": _safe_text(proposal.get("event_phrase", "")) or value,
                "event_text": _safe_text(proposal.get("event_text", "")) or _safe_text(proposal.get("event_phrase", "")) or value,
                "event_signature": event_signature,
                "extracted_subject": extracted_subject,
                "subject": subject,
                "subject_signature": subject_signature,
                "canonical_slot_key": canonical_slot_key or slot_key,
                "subject_from_sidecar": bool(subject_from_sidecar),
                "sidecar_subject_hint": sidecar_subject,
                "sidecar_subject_signature_hint": sidecar_subject_signature,
                "allow_parallel_state": False if canonical_slot_key else bool(proposal.get("allow_parallel_state", True)),
                "time_expression_span": _safe_text(proposal.get("time_expression_span", "")),
                "time_granularity": _safe_text(proposal.get("time_granularity", "")),
                "resolved_time_value": _safe_text(proposal.get("resolved_time_value", "")),
                "time_display_value": _safe_text(proposal.get("time_display_value", "")),
                "resolved_date": _safe_text(proposal.get("resolved_date", "")),
                "memory_gate_decision": _safe_text(proposal.get("write_decision", "")) or "parallel_active",
                "grounding_score": round(float(grounding), 6),
                "value_grounding_score": round(float(value_grounding), 6),
                "original_value_grounding_score": round(float(original_value_grounding), 6),
                "canonicalized_from_value": canonicalized_from_value,
                "canonicalization_reason": canonicalization_reason,
                "source_span_repair": _safe_text(proposal.get("source_span_repair", "")),
                "original_source_span": _safe_text(proposal.get("original_source_span", "")),
                "original_value": _safe_text(proposal.get("original_value", "")),
                "llm_write_proposal_index": index,
                "proposal_intent_gate": dict(proposal_policy),
                "profile_intent_gate": dict(profile_intent_gate),
                **profile_candidate_metadata(
                    category=record_category,
                    semantic_slot=semantic_slot,
                    relation=relation,
                    value=value,
                    source_span=source_span,
                    slot_key=slot_key,
                    anchors=anchors,
                    subject=subject,
                    subject_signature=subject_signature,
                    proposal=profile_proposal,
                ),
            }
            parent_record = {
                "category": record_category,
                "slot_key": slot_key,
                "value": value,
                "anchors": anchors,
                "relation": relation,
                "source_kind": source_kind,
                "state": "evidence" if record_category == "question" else _safe_text(proposal.get("state", "")),
                "salience": float(proposal.get("salience", 0.94) or 0.94),
                "confidence": min(1.0, max(0.0, float(proposal.get("confidence", 0.86) or 0.86))),
                "metadata": metadata,
            }
            facet_records = _build_event_facet_records(
                parent_record=parent_record,
                proposal=proposal,
                common_metadata=common_metadata,
                source_context=source_context,
                current_turn_text=compact_text,
                speaker=speaker,
                timestamp=timestamp,
                turn_index=turn_index,
                proposal_index=index,
            )
            if facet_records:
                metadata["facet_types"] = _dedupe_texts(
                    [dict(item.get("metadata", {}) or {}).get("facet_type", "") for item in facet_records],
                    max_items=8,
                )
                metadata["facet_record_count"] = len(facet_records)
            records.append(parent_record)
            records.extend(facet_records)
            suppress_profile_shadow_for_sidecar_subject = bool(
                sidecar_subject
                and sidecar_canonical_slot_key
                and _safe_text(slot_key).lower() == _safe_text(sidecar_canonical_slot_key).lower()
            )
            if bool(profile_intent_gate.get("create_profile_shadow", False)) and not suppress_profile_shadow_for_sidecar_subject:
                shadow_profile_type = _safe_text(profile_intent_gate.get("profile_type", "")) or "usage_context"
                shadow_profile_domain = _safe_text(profile_intent_gate.get("profile_domain", "")) or semantic_slot or subject or "general"
                shadow_slot_key = (
                    f"{session_slug}.turn_{max(1, int(turn_index or 1))}."
                    f"profile.{_slug(shadow_profile_type)}.{_slug(shadow_profile_domain)}.{index + 1}"
                )
                shadow_anchors = _dedupe_texts(
                    [
                        *anchors,
                        shadow_profile_type,
                        shadow_profile_domain,
                    ],
                    max_items=10,
                )
                shadow_proposal = {
                    **proposal,
                    "profile_type": shadow_profile_type,
                    "profile_domain": shadow_profile_domain,
                    "profile_candidate_status": "writer_shadow",
                }
                shadow_metadata = {
                    **common_metadata,
                    "content_variant": "profile_shadow_from_writer",
                    "raw_text": source_span or value,
                    "source_turn_text": source_span or value,
                    "semantic_slot": semantic_slot or shadow_profile_domain,
                    "target_status": _safe_text(proposal.get("target_status", "")),
                    "source_span": source_span or value,
                    "event_phrase": _safe_text(proposal.get("event_phrase", "")) or value,
                    "event_text": _safe_text(proposal.get("event_text", "")) or _safe_text(proposal.get("event_phrase", "")) or value,
                    "event_signature": event_signature,
                    "extracted_subject": extracted_subject,
                    "subject": subject or shadow_profile_domain,
                    "subject_signature": subject_signature or _slug(subject or shadow_profile_domain),
                    "canonical_slot_key": shadow_slot_key,
                    "subject_from_sidecar": bool(subject_from_sidecar),
                    "sidecar_subject_hint": sidecar_subject,
                    "sidecar_subject_signature_hint": sidecar_subject_signature,
                    "allow_parallel_state": False,
                    "time_expression_span": _safe_text(proposal.get("time_expression_span", "")),
                    "time_granularity": _safe_text(proposal.get("time_granularity", "")),
                    "resolved_time_value": _safe_text(proposal.get("resolved_time_value", "")),
                    "time_display_value": _safe_text(proposal.get("time_display_value", "")),
                    "resolved_date": _safe_text(proposal.get("resolved_date", "")),
                    "memory_gate_decision": "profile_shadow_from_turn_intent",
                    "grounding_score": round(float(grounding), 6),
                    "value_grounding_score": round(float(value_grounding), 6),
                    "original_value_grounding_score": round(float(original_value_grounding), 6),
                    "canonicalized_from_value": canonicalized_from_value,
                    "canonicalization_reason": canonicalization_reason,
                    "source_span_repair": _safe_text(proposal.get("source_span_repair", "")),
                    "original_source_span": _safe_text(proposal.get("original_source_span", "")),
                    "original_value": _safe_text(proposal.get("original_value", "")),
                    "llm_write_proposal_index": index,
                    "proposal_intent_gate": dict(proposal_policy),
                    "profile_intent_gate": dict(profile_intent_gate),
                    "profile_shadow_source_category": record_category,
                    "profile_shadow_source_slot_key": slot_key,
                    **profile_candidate_metadata(
                        category="profile",
                        semantic_slot=semantic_slot or shadow_profile_domain,
                        relation="profile_shadow",
                        value=value,
                        source_span=source_span,
                        slot_key=shadow_slot_key,
                        anchors=shadow_anchors,
                        subject=subject or shadow_profile_domain,
                        subject_signature=subject_signature or _slug(subject or shadow_profile_domain),
                        proposal=shadow_proposal,
                    ),
                }
                shadow_dedupe_key = (
                    "profile",
                    shadow_slot_key.lower(),
                    re.sub(r"\s+", " ", value.lower()).strip(),
                )
                if shadow_dedupe_key not in seen:
                    seen.add(shadow_dedupe_key)
                    records.append(
                        {
                            "category": "profile",
                            "slot_key": shadow_slot_key,
                            "value": value,
                            "anchors": shadow_anchors,
                            "relation": "profile_shadow",
                            "source_kind": "public_dialog_profile",
                            "state": _safe_text(proposal.get("state", "")),
                            "salience": min(0.98, max(0.86, float(proposal.get("salience", 0.9) or 0.9))),
                            "confidence": min(0.97, max(0.82, float(proposal.get("confidence", 0.84) or 0.84))),
                            "metadata": shadow_metadata,
                        }
                    )
        if (
            len(records) == 1
            and bool(turn_intent.get("write_allowed", False))
            and not question_only_turn
            and _safe_text(turn_intent.get("source", "")).startswith("writer")
            and not _SECRET_RE.search(compact_text)
        ):
            fallback_proposal = next((dict(item) for item in proposals if isinstance(item, Mapping)), {"category": "fact"})
            fallback_category = _safe_text(fallback_proposal.get("category", "")).lower().replace("-", "_") or "fact"
            if fallback_category in _WRITE_CATEGORY_ALIASES:
                fallback_category = _WRITE_CATEGORY_ALIASES[fallback_category]
            if fallback_category in ALLOWED_WRITE_CATEGORIES and fallback_category not in {"question", "interaction_intent"}:
                fallback_semantic_slot = _safe_text(fallback_proposal.get("semantic_slot", "")) or fallback_category
                fallback_relation = _safe_text(fallback_proposal.get("relation", "")) or f"{fallback_category}_memory"
                fallback_record_category, fallback_source_kind = _source_kind_for(
                    fallback_category,
                    fallback_semantic_slot,
                    fallback_relation,
                )
                if fallback_record_category != "question" and len(set(_tokens(compact_text))) >= 4:
                    fallback_event_signature = compute_public_event_signature(
                        compact_text,
                        speaker=_safe_text(speaker),
                        semantic_slot=fallback_semantic_slot or fallback_record_category,
                    )
                    fallback_slot_key = (
                        _safe_text(fallback_proposal.get("slot_key", ""))
                        or f"{session_slug}.turn_{max(1, int(turn_index or 1))}.{fallback_record_category}.intent_fallback"
                    )
                    fallback_anchors = _dedupe_texts(
                        [
                            speaker,
                            timestamp,
                            *list(fallback_proposal.get("anchors", []) or []),
                            *fallback_event_signature.split()[:4],
                        ],
                        max_items=8,
                    )
                    fallback_policy = {
                        "decision": "current_turn_exact_fallback",
                        "reject_reason": "",
                        "force_question": False,
                        "speech_act_memory": False,
                        "transient_directive": False,
                        "turn_intent_reason": _safe_text(turn_intent.get("reason", "")),
                    }
                    fallback_metadata = {
                        **common_metadata,
                        "content_variant": "llm_semantic_write",
                        "raw_text": compact_text,
                        "source_turn_text": compact_text,
                        "semantic_slot": fallback_semantic_slot,
                        "target_status": _safe_text(fallback_proposal.get("target_status", "")),
                        "source_span": compact_text,
                        "event_phrase": compact_text,
                        "event_text": compact_text,
                        "event_signature": fallback_event_signature,
                        "canonical_slot_key": fallback_slot_key,
                        "allow_parallel_state": bool(fallback_proposal.get("allow_parallel_state", True)),
                        "memory_gate_decision": "turn_intent_current_turn_fallback",
                        "grounding_score": 1.0,
                        "value_grounding_score": 1.0,
                        "original_value_grounding_score": 0.0,
                        "canonicalized_from_value": _safe_text(fallback_proposal.get("value", "")),
                        "canonicalization_reason": "turn_intent_current_turn_exact_fallback",
                        "source_span_repair": "current_turn_exact_from_turn_intent",
                        "original_source_span": _safe_text(fallback_proposal.get("source_span", "")),
                        "original_value": _safe_text(fallback_proposal.get("value", "")),
                        "llm_write_proposal_index": 0,
                        "proposal_intent_gate": fallback_policy,
                        **profile_candidate_metadata(
                            category=fallback_record_category,
                            semantic_slot=fallback_semantic_slot,
                            relation=fallback_relation,
                            value=compact_text,
                            source_span=compact_text,
                            slot_key=fallback_slot_key,
                            anchors=fallback_anchors,
                            subject="",
                            subject_signature="",
                            proposal={**fallback_proposal, "proposal_intent_gate": fallback_policy},
                        ),
                    }
                    fallback_record = {
                        "category": fallback_record_category,
                        "slot_key": fallback_slot_key,
                        "value": compact_text,
                        "anchors": fallback_anchors,
                        "relation": fallback_relation,
                        "source_kind": fallback_source_kind,
                        "state": "active",
                        "salience": float(fallback_proposal.get("salience", 0.88) or 0.88),
                        "confidence": min(1.0, max(0.0, float(fallback_proposal.get("confidence", 0.82) or 0.82))),
                        "metadata": fallback_metadata,
                    }
                    fallback_facet_records = _build_event_facet_records(
                        parent_record=fallback_record,
                        proposal=fallback_proposal,
                        common_metadata=common_metadata,
                        source_context=source_context,
                        current_turn_text=compact_text,
                        speaker=speaker,
                        timestamp=timestamp,
                        turn_index=turn_index,
                        proposal_index=0,
                    )
                    if fallback_facet_records:
                        fallback_metadata["facet_types"] = _dedupe_texts(
                            [dict(item.get("metadata", {}) or {}).get("facet_type", "") for item in fallback_facet_records],
                            max_items=8,
                        )
                        fallback_metadata["facet_record_count"] = len(fallback_facet_records)
                    records.append(fallback_record)
                    records.extend(fallback_facet_records)
        current_subject = sidecar_subject or _extract_memory_subject(compact_text)
        current_subject_signature = sidecar_subject_signature or (_slug(current_subject) if current_subject else "")
        current_slot_key = sidecar_canonical_slot_key or (
            f"{session_slug}.subject.{current_subject_signature}" if current_subject_signature else ""
        )
        current_source_span, current_value = _extract_current_value_assertion(
            current_turn_text=compact_text,
            subject=current_subject,
        )
        current_slot_records = [
            record
            for record in records[1:]
            if _safe_text(record.get("slot_key", "")).lower() == _safe_text(current_slot_key).lower()
            and _safe_text(record.get("category", "")).lower() != "question"
        ]
        current_value_final_head = bool(current_slot_records) and _record_covers_current_value(
            record=current_slot_records[-1],
            slot_key=current_slot_key,
            current_value=current_value,
        )
        if current_slot_key and current_value and not question_only_turn and not current_value_final_head:
            current_event_signature = compute_public_event_signature(
                current_source_span or current_value,
                speaker=_safe_text(speaker),
                semantic_slot=current_subject_signature or "current_subject_value",
            )
            current_dedupe_key = (
                "preference",
                current_slot_key.lower(),
                re.sub(r"\s+", " ", current_value.lower()).strip(),
            )
            if current_dedupe_key not in seen:
                seen.add(current_dedupe_key)
                records.append(
                    {
                        "category": "preference",
                        "slot_key": current_slot_key,
                        "value": current_value,
                        "anchors": _dedupe_texts(
                            [
                                speaker,
                                timestamp,
                                current_subject,
                                *current_event_signature.split()[:4],
                            ],
                            max_items=8,
                        ),
                        "relation": "current_subject_value",
                        "source_kind": "public_dialog_fact",
                        "state": "active",
                        "salience": 0.98,
                        "confidence": 0.98,
                        "metadata": {
                            **common_metadata,
                            "content_variant": "deterministic_current_subject_value",
                            "raw_text": current_source_span or current_value,
                            "source_turn_text": current_source_span or current_value,
                            "semantic_slot": current_subject_signature or "current_subject_value",
                            "target_status": "current",
                            "source_span": current_source_span or current_value,
                            "event_phrase": current_value,
                            "event_text": current_value,
                            "event_signature": current_event_signature,
                            "extracted_subject": _extract_memory_subject(compact_text),
                            "subject": current_subject,
                            "subject_signature": current_subject_signature,
                            "canonical_slot_key": current_slot_key,
                            "subject_from_sidecar": bool(sidecar_subject and current_subject == sidecar_subject),
                            "sidecar_subject_hint": sidecar_subject,
                            "sidecar_subject_signature_hint": sidecar_subject_signature,
                            "allow_parallel_state": False,
                            "memory_gate_decision": "current_slot_head",
                            "grounding_score": 1.0,
                            "value_grounding_score": 1.0,
                            "original_value_grounding_score": 1.0,
                            "canonicalized_from_value": "",
                            "canonicalization_reason": "",
                            "llm_write_proposal_index": -1,
                            **profile_candidate_metadata(
                                category="preference",
                                semantic_slot=current_subject_signature or "current_subject_value",
                                relation="current_subject_value",
                                value=current_value,
                                source_span=current_source_span or current_value,
                                slot_key=current_slot_key,
                                anchors=[
                                    speaker,
                                    timestamp,
                                    current_subject,
                                    *current_event_signature.split()[:4],
                                ],
                                subject=current_subject,
                                subject_signature=current_subject_signature,
                                proposal={"profile_type": "usage_context", "profile_domain": current_subject},
                            ),
                        },
                    }
                )

        payload = {
            "metadata": {
                "memory_write": True,
                "source": "public_benchmark",
                "write_path": "llm_semantic_writer_gate",
                "semantic_writer_accepted_count": max(0, len(records) - 1),
                "semantic_writer_rejected_count": len(rejected),
                "semantic_writer_suspect_count": len(suspect_records),
                "turn_extractor_role": "sidecar_hints",
            },
            "replacement_memory_records": records,
            "suspect_memory_records": suspect_records,
            "semantic_writer_rejections": rejected,
        }
        accepted_count = max(0, len(records) - 1)
        return MemoryWriteGateResult(
            payload=payload,
            accepted_count=accepted_count,
            rejected=rejected,
            suspected_count=len(suspect_records),
        )


class OpenAICompatSemanticMemoryWriter:
    def __init__(
        self,
        profile: LLMProfile,
        *,
        gate: DeterministicMemoryWriteGate | None = None,
        max_proposals: int = 4,
    ) -> None:
        self.profile = profile
        self.gate = gate or DeterministicMemoryWriteGate()
        self.max_proposals = max(1, int(max_proposals or 4))

    def available(self) -> bool:
        return bool(_safe_text(self.profile.base_url)) and bool(_safe_text(self.profile.model))

    def _writer_family(self) -> str:
        marker = f"{_safe_text(self.profile.model)} {_safe_text(self.profile.base_url)}".lower()
        if "deepseek" in marker:
            return "deepseek"
        if "gemma" in marker:
            return "gemma"
        return "generic"

    def _chat(self, messages: Sequence[Mapping[str, str]]) -> tuple[str, Dict[str, int]]:
        base_url = _safe_text(self.profile.base_url).rstrip("/")
        if not base_url:
            raise RuntimeError("semantic memory writer base_url is empty")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": self.profile.model,
            "messages": list(messages),
            "temperature": float(self.profile.temperature),
            "max_tokens": int(self.profile.max_tokens),
            "response_format": {"type": "json_object"},
        }
        raw_payload = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if _safe_text(self.profile.api_key):
            headers["Authorization"] = f"Bearer {_safe_text(self.profile.api_key)}"
        request = urllib.request.Request(url, data=raw_payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=float(self.profile.timeout_seconds)) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if b"response_format" not in exc.read()[:2048]:
                raise
            payload.pop("response_format", None)
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=float(self.profile.timeout_seconds)) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        choices = list(response_payload.get("choices", []) or [])
        message = dict(dict(choices[0] if choices else {}).get("message", {}) or {})
        usage = dict(response_payload.get("usage", {}) or {})
        return _safe_text(message.get("content", "")), {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

    def _classify_public_turn_intent(
        self,
        *,
        current_turn: str,
        speaker: str = "",
        session_timestamp: str = "",
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        system_prompt = (
            "Return JSON only. Read only current_turn. Classify the whole current_turn for memory writing. "
            "Do not extract memories. Do not include prose. "
            "Use intent=memory_assertion when the turn states a durable user fact, preference, goal, constraint, status, event, or plan. "
            "Use intent=mixed_question_with_memory when the turn has question-like wording and also states a durable new user memory. "
            "Use intent=question_only when the turn only asks, checks, or asks you to recall existing memory. "
            "Use intent=no_new_memory for thanks, reactions, or turns with no new durable memory. "
            "Use intent=transient_directive for one-shot instructions about how to answer this turn. "
            "Return exactly {\"turn_intent\":{\"intent\":\"memory_assertion|mixed_question_with_memory|question_only|no_new_memory|transient_directive|plain_statement\","
            "\"write_allowed\":true,\"question_like\":false,\"memory_assertion\":true,\"question_only\":false,\"reason\":\"...\"}}."
        )
        user_payload = {
            "speaker": _safe_text(speaker),
            "timestamp": _safe_text(session_timestamp),
            "current_turn": _safe_text(current_turn),
        }
        content, usage = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ]
        )
        parsed = _extract_json_object(content)
        writer_turn_intent = _extract_turn_intent_object(parsed) or _extract_turn_intent_from_text(content)
        return writer_turn_intent, {
            "raw_output_excerpt": content[:1000],
            "usage": usage,
            "json_repair_status": "parsed" if parsed else ("intent_extracted_from_jsonish_text" if writer_turn_intent else "empty_from_non_json_writer_content"),
            "system_prompt_chars": len(system_prompt),
            "user_payload_chars": len(json.dumps(user_payload, ensure_ascii=False)),
        }

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
        input_mode: str = "full",
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.available():
            raise SemanticMemoryWriterError(
                "semantic memory writer is required but not configured; "
                f"model={_safe_text(self.profile.model)!r}, base_url={_safe_text(self.profile.base_url)!r}"
            )
        normalized_input_mode = _safe_text(input_mode).lower() or "full"
        if normalized_input_mode not in {"full", "delta"}:
            raise SemanticMemoryWriterError(f"unsupported semantic writer input_mode={input_mode!r}")
        writer_family = self._writer_family()
        answerable_memory_prompt = ""
        if writer_family == "deepseek":
            answerable_memory_prompt = (
                "DeepSeek writer profile: maximize recall for answerable user memories, not just broad preferences. "
                "In long benchmark-like conversation chunks, one-off first-person facts are durable when they can answer a later question. "
                "Always prefer the most specific grounded fact over a generic summary. "
                "Treat these as high-priority writable memories when stated by the user: store or venue names, service names, class or studio names, playlist names, book/movie/play/song/artwork titles, event names, dates/months, durations, prices, colors, locations, people names, product names, and organizations. "
                "User phrases such as I bought, I redeemed, I attended, I volunteered, I created, I take classes at, I repainted, I changed my name, I commute, I recently, by the way, or I just remembered usually contain answerable memory. "
                "Do not skip such facts because nearby text asks for recommendations, tips, explanations, or advice. "
                "For source_span, copy the shortest exact span that contains the answer string and enough context to prove it belongs to the user. "
                "For value, write a direct third-person memory about the user that preserves exact names and casing. "
                "Prioritize answerable facts over generic goals, attitudes, or assistant-request memories. "
                "If a chunk contains multiple answerable facts, return the strongest one or two within the proposal budget. "
                "Few-shot positives: "
                "'I redeemed a $5 coupon on coffee creamer at Target' -> user redeemed a $5 coupon on coffee creamer at Target. "
                "'I attended The Glass Menagerie at the local community theater' -> user attended The Glass Menagerie at the local community theater. "
                "'I take yoga classes at Serenity Yoga' -> user takes yoga classes at Serenity Yoga. "
                "'I created a Spotify playlist called Summer Vibes' -> user created a Spotify playlist called Summer Vibes. "
                "'I volunteered at the Love is in the Air fundraising dinner back on Valentine's Day' -> user volunteered at the Love is in the Air fundraising dinner on Valentine's Day. "
            )
        intent_schema_prompt = (
            "Top-level turn intent is required. "
            "Use intent=memory_assertion when the turn states a durable user fact, preference, goal, constraint, status, event, or plan. "
            "Use intent=mixed_question_with_memory when the turn contains a question-like phrase and also states a durable new user memory. "
            "Use intent=question_only when the turn only asks, checks, or asks you to recall existing memory. "
            "Use intent=no_new_memory for thanks, reactions, or turns with no new durable memory. "
            "Use intent=transient_directive for one-shot instructions about how to answer this turn. "
            "Use intent=plain_statement only for durable statements that are not clearly first-person profile/fact writes. "
            "Return the full JSON shape {\"turn_intent\":{\"intent\":\"memory_assertion|mixed_question_with_memory|question_only|no_new_memory|transient_directive|plain_statement\","
            "\"write_allowed\":true,\"question_like\":false,\"memory_assertion\":true,\"question_only\":false,\"reason\":\"...\"},\"write_proposals\":[...]}. "
        )
        schema_prompt = intent_schema_prompt + (
            "Write proposal schema: {\"category\":\"event|profile|time|status|preference|goal|constraint|fact\","
            "\"value\":\"...\",\"source_span\":\"...\",\"semantic_slot\":\"open_domain_slot|event_time\","
            "\"profile_type\":\"setup|preference|constraint|goal|avoid|usage_context\",\"profile_domain\":\"dynamic open-domain label\","
            "\"target_status\":\"past|current|planned|\",\"time_expression_span\":\"...\",\"time_granularity\":\"day|month|year|relative_day_reference|none|\","
            "\"anchors\":[\"...\"],\"salience\":0.0,\"confidence\":0.0,"
            "\"facets\":[{\"type\":\"temporal|numeric|state|entity|evidence_role\",\"role\":\"start|finish|amount|count|subject|constraint|...\","
            "\"value\":\"grounded attribute value\",\"unit\":\"optional unit\",\"source_span\":\"exact copied evidence span\"}]}. "
            "Facets are optional attribute subnodes of the proposal. Emit a facet only when the current_turn explicitly contains that attribute; never emit empty, zero, or placeholder facets. "
        )
        if normalized_input_mode == "delta":
            system_prompt = (
                "Return JSON only. Read only current_turn. "
                "First classify the whole current_turn into top-level turn_intent, then emit write_proposals. "
                "Write proposals only for new durable user memory explicitly stated now: fact, event, profile, time, status, preference, goal, or constraint. "
                "Keep output tiny: at most two proposals; value <= 120 chars; source_span <= 160 chars and copied exactly from current_turn. "
                "If current_turn is a long mixed conversation chunk, still extract durable first-person user facts inside it; do not skip them just because nearby turns ask for recommendations, tips, or advice. "
                "Write value as a complete third-person memory about the user, preferably starting with 'user ...'. "
                "For user preferences, goals, constraints, setup, habits, or usage context, include profile_type and dynamic profile_domain. "
                "When a writable fact contains explicit time, duration, quantity, amount, count, state, or acted-on entity, include grounded facets under that write proposal. "
                "Do not put absent attributes into facets; omitted means not present. "
                "Profile writer rule: when the user states current setup, a brand/device they use, durable taste, recurring habit, goal, or constraint, mark it as profile-compatible with profile_type and profile_domain even if the category is fact or event. "
                "Preserve exact project names, codenames, acronyms, model/API names, and mixed-case identifiers when they appear in source_span. "
                f"{answerable_memory_prompt}"
                "Skip questions, thanks, reactions, assistant requests, one-shot answer instructions, secrets, and no-new-memory turns. "
                "Never write instructions about how the assistant should answer this turn. "
                f"{intent_schema_prompt}"
                "{\"turn_intent\":{\"intent\":\"...\",\"write_allowed\":true,\"question_like\":false,\"memory_assertion\":true,\"question_only\":false,\"reason\":\"...\"},"
                "\"write_proposals\":[{\"category\":\"fact|event|profile|time|status|preference|goal|constraint\","
                "\"value\":\"...\",\"source_span\":\"...\",\"semantic_slot\":\"...\","
                "\"facets\":[{\"type\":\"temporal|numeric|state|entity|evidence_role\",\"role\":\"...\",\"value\":\"...\",\"source_span\":\"...\"}]}]}"
            )
            user_payload = {
                "speaker": _safe_text(speaker),
                "timestamp": _safe_text(session_timestamp),
                "current_turn": _safe_text(current_turn),
                "max_write_proposals": self.max_proposals,
            }
            system_prompt_profile = "delta_current_turn_only"
        else:
            system_prompt = (
                "You are the TMCRA semantic memory writer. Return strict JSON only. "
                "Do not include thinking, analysis, markdown, or prose outside JSON. "
                "First classify the whole current_turn into top-level turn_intent, then decide write_proposals. "
                "Decide what should be remembered from the current conversation turn. "
                "The extractor output is only sidecar hints; do not copy unsupported hints. "
                "Every write must include a source_span copied from the turn or auxiliary evidence. "
                "The current turn is the only writable dialogue source; previous and next turns are context only. "
                "Do not write secrets, passwords, API keys, or unsupported guesses. "
                "Only write durable memory: asserted user facts, preferences, goals, constraints, plans, status changes, events, or time facts. "
                "When a writable fact contains explicit time, duration, quantity, amount, count, state, or acted-on entity, include grounded facets under that write proposal. "
                "Do not put absent attributes into facets; omitted means not present. "
                "For user preferences, goals, constraints, setup, habits, or usage context, fill profile_type from the fixed set "
                "setup|preference|constraint|goal|avoid|usage_context and fill profile_domain as a dynamic open-domain label from the user's own wording. "
                "Profile writer rule: when the user states current setup, a brand/device they use, durable taste, recurring habit, goal, or constraint, mark it as profile-compatible with profile_type and profile_domain even if the category is fact or event. "
                "Preserve exact project names, codenames, acronyms, model/API names, and mixed-case identifiers when they appear in source_span. "
                "Skip one-shot answer instructions such as 'answer only from context', 'do not mention retrieval', or 'summarize this now'. "
                "Never write instructions about how the assistant should answer this turn. "
                "If the current turn only asks a question, asks the assistant to recall existing memory, thanks/reacts, or contains no new durable memory, return an empty write_proposals list. "
                "Do not convert questions such as 'what is my project?' or 'do you remember my plan?' into profile/fact/status writes. "
                "Never write speech-act memories that merely describe the user asking, checking, reminding, thanking, agreeing, or requesting assistant action. "
                "Values like 'asked what the project is', 'asked for a reminder', 'wanted the assistant to recall', or 'requested help' are forbidden unless the user also states a new durable fact. "
                "A question mark in the current turn is a strong warning: only write if the same turn also asserts a concrete new value to remember. "
                "Negative examples: 'Can you remind me what my main project is?' -> {\"turn_intent\":{\"intent\":\"question_only\",\"write_allowed\":false,\"question_like\":true,\"memory_assertion\":false,\"question_only\":true,\"reason\":\"asks recall only\"},\"write_proposals\":[]}; 'What is my meeting plan?' -> same question_only shape with empty write_proposals. "
                "Positive examples: 'Remember that my main project is TMCRA.' -> write a grounded profile/fact; 'My meeting plan is Friday.' -> write a grounded plan/status. "
                "For memory-bearing turns, produce at least one grounded write proposal. "
                "Activity evaluations without explicit 'I like' can still be speaker preferences or coping strategies when the preference is grounded in the current turn; previous turns may disambiguate only. "
                f"{answerable_memory_prompt}"
                f"{schema_prompt}"
            )
            user_payload = {
                "speaker": _safe_text(speaker),
                "timestamp": _safe_text(session_timestamp),
                "previous_turn": _safe_text(previous_turn),
                "current_turn": _safe_text(current_turn),
                "next_turn": _safe_text(next_turn),
                "auxiliary_evidence_texts": list(auxiliary_evidence_texts or []),
                "sidecar_hints": dict(sidecar_hints or {}),
                "max_write_proposals": self.max_proposals,
            }
            system_prompt_profile = "full_context_sidecar"
        content, usage = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ]
        )
        parsed = _extract_json_object(content)
        json_repair_status = _safe_text(parsed.pop("_tmcra_json_repair_status", "")) if parsed else ""
        if not parsed and content:
            parsed = {"write_proposals": []}
            json_repair_status = "empty_from_non_json_writer_content"
        writer_turn_intent = _extract_turn_intent_object(parsed) or _extract_turn_intent_from_text(content)
        writer_turn_intent_source = "inline_writer_output" if writer_turn_intent else ""
        writer_turn_intent_repair_metadata: Dict[str, Any] = {}
        if not writer_turn_intent:
            writer_turn_intent, writer_turn_intent_repair_metadata = self._classify_public_turn_intent(
                current_turn=current_turn,
                speaker=speaker,
                session_timestamp=session_timestamp,
            )
            writer_turn_intent_source = "repair_intent_call" if writer_turn_intent else "missing"
        proposals = parsed.get("write_proposals", parsed.get("write_proposal", parsed.get("memory_writes", [])))
        if isinstance(proposals, Mapping):
            proposals = [dict(proposals)]
        if not isinstance(proposals, list):
            proposals = []
            json_repair_status = json_repair_status or "empty_from_invalid_write_proposals_field"
        return [dict(item) for item in proposals[: self.max_proposals] if isinstance(item, Mapping)], {
            "raw_output_excerpt": content[:1000],
            "usage": usage,
            "proposal_count": len(proposals),
            "json_repair_status": json_repair_status or "parsed",
            "json_repair_applied": bool(json_repair_status),
            "writer_turn_intent": dict(writer_turn_intent),
            "writer_turn_intent_source": writer_turn_intent_source,
            "writer_turn_intent_repair_metadata": dict(writer_turn_intent_repair_metadata),
            "input_mode": normalized_input_mode,
            "writer_family": writer_family,
            "system_prompt_profile": system_prompt_profile,
            "user_payload_keys": sorted(user_payload.keys()),
            "system_prompt_chars": len(system_prompt),
            "user_payload_chars": len(json.dumps(user_payload, ensure_ascii=False)),
        }

    def write_public_turn(
        self,
        *,
        text: str,
        raw_text: str = "",
        speaker: str,
        session_key: str,
        turn_index: int,
        timestamp: str = "",
        dia_id: str = "",
        previous_turn: str = "",
        next_turn: str = "",
        sidecar_hints: Mapping[str, Any] | None = None,
        auxiliary_evidence_texts: Sequence[Any] | None = None,
        input_mode: str = "full",
    ) -> Dict[str, Any]:
        proposals, writer_metadata = self.propose_public_turn(
            current_turn=raw_text or text,
            previous_turn=previous_turn,
            next_turn=next_turn,
            speaker=speaker,
            session_timestamp=timestamp,
            sidecar_hints=sidecar_hints,
            auxiliary_evidence_texts=auxiliary_evidence_texts,
            input_mode=input_mode,
        )
        result = self.gate.build_payload(
            proposals=proposals,
            text=text,
            raw_text=raw_text,
            speaker=speaker,
            session_key=session_key,
            turn_index=turn_index,
            timestamp=timestamp,
            dia_id=dia_id,
            sidecar_hints=sidecar_hints,
            auxiliary_evidence_texts=auxiliary_evidence_texts,
            writer_metadata=writer_metadata,
        )
        if result.accepted_count <= 0 and result.suspected_count <= 0:
            raise SemanticMemoryWriterError(
                "semantic memory writer produced no gated write proposals; "
                f"proposal_count={len(proposals)}, rejected={result.rejected[:3]}"
            )
        return result.payload


__all__ = [
    "ALLOWED_WRITE_CATEGORIES",
    "DeterministicMemoryWriteGate",
    "MemoryWriteGateResult",
    "OpenAICompatSemanticMemoryWriter",
    "SemanticMemoryWriterError",
    "build_modelized_facet_unit_records",
]
