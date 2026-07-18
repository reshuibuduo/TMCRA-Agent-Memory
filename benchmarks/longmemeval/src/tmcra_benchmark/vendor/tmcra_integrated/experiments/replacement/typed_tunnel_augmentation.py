from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence


TYPED_TUNNEL_VERSION = "typed_tunnel_v1"

NODE_TAG_KEY = "tmcra_node_tags"
EDGE_TAG_KEY = "tmcra_edge_tags"
PATH_TAG_KEY = "tmcra_path_tags"
TUNNEL_ROLE_KEY = "tmcra_tunnel_roles"
TUNNEL_GROUP_KEY = "tmcra_tunnel_group_key"
TEMPORAL_RELATION_KEY = "tmcra_temporal_relation"
MEASURE_SIGNATURE_KEY = "tmcra_measure_signature"
ACTION_SIGNATURE_KEY = "tmcra_action_signature"
ENTITY_SIGNATURE_KEY = "tmcra_entity_signature"

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.I)
_MONEY_RE = re.compile(
    r"(?:[$€£]\s*\d+(?:[\d,]*)(?:\.\d+)?|\b\d+(?:[\d,]*)(?:\.\d+)?\s*(?:dollars?|usd|eur|pounds?|yuan|rmb)\b)",
    re.I,
)
_QUANTITY_RE = re.compile(
    r"\b\d+(?:[\d,]*)(?:\.\d+)?\s*(?:items?|tickets?|orders?|times?|days?|weeks?|months?|years?|people|users?|engineers?|hours?|minutes?|eggs?|cars?|bikes?|books?|meals?|servings?|units?)\b",
    re.I,
)
_DATE_RE = re.compile(
    r"\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]20\d{2}|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|yesterday|last|next|before|after|earlier|later|then|finally)\b",
    re.I,
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "the",
    "their",
    "this",
    "that",
    "to",
    "was",
    "we",
    "with",
    "you",
}

_ACTION_MARKERS = (
    "bought",
    "buy",
    "purchased",
    "purchase",
    "ordered",
    "order",
    "sold",
    "sell",
    "paid",
    "pay",
    "returned",
    "return",
    "exchanged",
    "exchange",
    "picked",
    "pick",
    "visited",
    "visit",
    "serviced",
    "service",
    "cleaned",
    "clean",
    "made",
    "make",
    "ate",
    "eat",
    "drank",
    "drink",
    "met",
    "meet",
    "called",
    "call",
    "scheduled",
    "schedule",
    "cancelled",
    "canceled",
    "cancel",
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\x00", " ")).strip()


def _normalize(value: Any) -> str:
    return _clean_text(value).lower()


def _tokens(value: Any) -> list[str]:
    return [item.lower() for item in _TOKEN_RE.findall(_clean_text(value))]


def _dedupe(items: Iterable[Any], *, max_items: int | None = None) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _clean_text(item)
        if not text:
            continue
        key = _normalize(text)
        if key in seen:
            continue
        seen.add(key)
        values.append(text)
        if max_items is not None and len(values) >= max_items:
            break
    return values


def _first_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return _clean_text(match.group(0)) if match else ""


def _signature_slug(parts: Sequence[Any], *, max_tokens: int = 6) -> str:
    tokens: list[str] = []
    for part in parts:
        for token in _tokens(part):
            if token in _STOPWORDS:
                continue
            tokens.append(token)
            if len(tokens) >= max_tokens:
                break
        if len(tokens) >= max_tokens:
            break
    return ".".join(_dedupe(tokens, max_items=max_tokens))


def _infer_entity_signature(
    *,
    text: str,
    metadata: Mapping[str, Any],
    anchors: Sequence[Any],
    slot_key: Any,
    category: Any,
) -> str:
    explicit = (
        _clean_text(metadata.get(ENTITY_SIGNATURE_KEY, ""))
        or _clean_text(metadata.get("subject_signature", ""))
        or _clean_text(metadata.get("memory_chain_subject_signature", ""))
        or _clean_text(metadata.get("profile_subject_signature", ""))
        or _clean_text(metadata.get("topic_bucket_id", ""))
    )
    if explicit:
        return explicit
    anchor_signature = _signature_slug(list(anchors or []), max_tokens=5)
    if anchor_signature:
        return anchor_signature
    slot_signature = _signature_slug([slot_key, category], max_tokens=5)
    if slot_signature:
        return slot_signature
    return _signature_slug([text], max_tokens=5)


def _infer_action_signature(text: str, metadata: Mapping[str, Any]) -> str:
    explicit = _clean_text(metadata.get(ACTION_SIGNATURE_KEY, ""))
    if explicit:
        return explicit
    tokens = _tokens(text)
    for index, token in enumerate(tokens):
        if token in _ACTION_MARKERS:
            tail = [item for item in tokens[index + 1 : index + 4] if item not in _STOPWORDS]
            return ".".join(_dedupe([token, *tail], max_items=4))
    return ""


def _infer_temporal_relation(text: str, metadata: Mapping[str, Any]) -> str:
    explicit = _clean_text(metadata.get(TEMPORAL_RELATION_KEY, ""))
    if explicit:
        return explicit
    normalized = _normalize(text)
    if "immediately before" in normalized or "right before" in normalized:
        return "immediate_before"
    if "immediately after" in normalized or "right after" in normalized:
        return "immediate_after"
    if "before" in normalized or "earlier" in normalized:
        return "before"
    if "after" in normalized or "later" in normalized or "then" in normalized:
        return "after"
    if "latest" in normalized or "current" in normalized or "now" in normalized:
        return "current"
    if "first" in normalized or "earliest" in normalized:
        return "first"
    if "last" in normalized or "finally" in normalized:
        return "last"
    return ""


def infer_typed_tunnel_metadata(
    *,
    value: Any,
    category: Any = "",
    relation: Any = "",
    slot_key: Any = "",
    anchors: Sequence[Any] = (),
    metadata: Mapping[str, Any] | None = None,
    source_text: Any = "",
) -> dict[str, Any]:
    base = dict(metadata or {})
    text = _clean_text(" ".join([_clean_text(source_text), _clean_text(value), _clean_text(relation), _clean_text(category)]))
    normalized = _normalize(text)
    node_tags = list(base.get(NODE_TAG_KEY, []) or [])
    path_tags = list(base.get(PATH_TAG_KEY, []) or [])
    tunnel_roles = list(base.get(TUNNEL_ROLE_KEY, []) or [])
    action_frame_schema = _clean_text(base.get("unit_attachment_schema", ""))
    action_frame_id = _clean_text(base.get("action_frame_id", ""))
    applies_to_action = _clean_text(base.get("applies_to_action", "")) or _clean_text(base.get("action", ""))
    applies_to_entity = _clean_text(base.get("applies_to_entity", "")) or _clean_text(base.get("target", ""))
    unit_kind = _clean_text(base.get("unit_kind", ""))
    facet_type = _clean_text(base.get("facet_type", ""))
    temporal_role = _clean_text(base.get("temporal_role", ""))
    numeric_unit = _clean_text(base.get("unit", "")) or _clean_text(base.get("numeric_unit", ""))

    money_signature = _clean_text(base.get(MEASURE_SIGNATURE_KEY, "")) or _first_match(_MONEY_RE, text)
    quantity_signature = _first_match(_QUANTITY_RE, text)
    measure_signature = _clean_text(base.get(MEASURE_SIGNATURE_KEY, "")) or numeric_unit or money_signature or quantity_signature
    temporal_relation = _clean_text(base.get(TEMPORAL_RELATION_KEY, "")) or temporal_role or _infer_temporal_relation(text, base)
    action_signature = _clean_text(base.get(ACTION_SIGNATURE_KEY, "")) or _signature_slug([applies_to_action], max_tokens=4) or _infer_action_signature(text, base)
    entity_signature = _clean_text(base.get(ENTITY_SIGNATURE_KEY, "")) or _signature_slug([applies_to_entity], max_tokens=6)
    if not entity_signature:
        entity_signature = _infer_entity_signature(
            text=text,
            metadata=base,
            anchors=anchors,
            slot_key=slot_key,
            category=category,
        )

    if action_frame_schema == "event_action_frame_v2" or action_frame_id:
        node_tags.append("event_action_frame_unit")
        path_tags.append("unit_tunnel_path")
        path_tags.append("multi_support_path")
        tunnel_roles.append("unit_positive_candidate")
        if unit_kind:
            node_tags.append(unit_kind)
        if facet_type:
            node_tags.append(f"{facet_type}_unit")
        if action_frame_id:
            node_tags.append("action_frame_bound")
        if applies_to_action:
            node_tags.append("action_instance")
        if applies_to_entity:
            node_tags.append("entity_bound")
        if temporal_role:
            node_tags.append("time_anchor")
            node_tags.append("timeline_event")
            path_tags.append("temporal_order_path")
        if facet_type == "numeric" or unit_kind == "numeric_quantity":
            node_tags.append("measure_fact")
            path_tags.append("aggregation_support_path")
            tunnel_roles.append("multi_positive_candidate")

    if money_signature:
        node_tags.append("money_fact")
        node_tags.append("measure_fact")
        path_tags.append("aggregation_support_path")
        tunnel_roles.append("multi_positive_candidate")
    if quantity_signature:
        node_tags.append("quantity_fact")
        node_tags.append("measure_fact")
        path_tags.append("aggregation_support_path")
        tunnel_roles.append("multi_positive_candidate")
    if action_signature:
        node_tags.append("action_instance")
        path_tags.append("multi_support_path")
    if _DATE_RE.search(text) or _clean_text(base.get("resolved_date", "")) or _clean_text(base.get("time_value", "")):
        node_tags.append("time_anchor")
        node_tags.append("timeline_event")
        path_tags.append("temporal_order_path")
    if temporal_relation:
        path_tags.append("temporal_order_path")
        if temporal_relation in {"before", "after", "immediate_before", "immediate_after"}:
            tunnel_roles.append("temporal_anchor_candidate")
    if any(marker in normalized for marker in ("current", "now", "latest", "updated", "changed to")):
        node_tags.append("current_state")
        path_tags.append("current_value_path")
    if any(marker in normalized for marker in ("previous", "old", "former", "used to", "earlier", "historical")):
        node_tags.append("historical_state")
    if any(marker in normalized for marker in ("plan", "planning", "might", "maybe", "consider", "could", "would", "if ")):
        node_tags.append("planned_state")
        node_tags.append("hypothetical_state")
    if any(marker in normalized for marker in ("not ", "never", "cancel", "exclude", "returned", "refund")):
        node_tags.append("negative_or_excluded")
    if _normalize(category) in {"profile", "preference", "constraint", "goal"}:
        path_tags.append("profile_bridge_path")

    group_parts = [entity_signature, action_signature, measure_signature]
    if action_frame_schema == "event_action_frame_v2" and facet_type in {"temporal", "state"}:
        group_parts.append(f"frame:{action_frame_id or applies_to_action}")
    if temporal_relation:
        group_parts.append(f"time:{temporal_relation}")
    tunnel_group_key = _clean_text(base.get(TUNNEL_GROUP_KEY, "")) or _signature_slug(group_parts, max_tokens=8)

    typed_signature_terms = _dedupe(
        [
            *node_tags,
            *path_tags,
            *tunnel_roles,
            entity_signature and f"entity {entity_signature}",
            action_signature and f"action {action_signature}",
            measure_signature and f"measure {measure_signature}",
            temporal_relation and f"temporal {temporal_relation}",
        ],
        max_items=24,
    )

    return {
        "tmcra_typed_tunnel_version": TYPED_TUNNEL_VERSION,
        NODE_TAG_KEY: _dedupe(node_tags, max_items=16),
        PATH_TAG_KEY: _dedupe(path_tags, max_items=16),
        TUNNEL_ROLE_KEY: _dedupe(tunnel_roles, max_items=16),
        TUNNEL_GROUP_KEY: tunnel_group_key,
        TEMPORAL_RELATION_KEY: temporal_relation,
        MEASURE_SIGNATURE_KEY: measure_signature,
        ACTION_SIGNATURE_KEY: action_signature,
        ENTITY_SIGNATURE_KEY: entity_signature,
        "tmcra_typed_event_signature": " ".join(typed_signature_terms),
    }


def annotate_memory_record(record: Any, *, source_text: Any = "") -> Any:
    metadata = dict(getattr(record, "metadata", {}) or {})
    typed = infer_typed_tunnel_metadata(
        value=getattr(record, "value", ""),
        category=getattr(record, "category", ""),
        relation=getattr(record, "relation", ""),
        slot_key=getattr(record, "slot_key", ""),
        anchors=list(getattr(record, "anchor_concepts", []) or []),
        metadata=metadata,
        source_text=source_text,
    )
    metadata.update({key: value for key, value in typed.items() if value not in ("", [], {})})
    record.metadata = metadata
    anchors = list(getattr(record, "anchor_concepts", []) or [])
    evidence_anchors = list(getattr(record, "evidence_anchors", []) or [])
    tag_anchors = [f"tag:{tag}" for tag in list(metadata.get(NODE_TAG_KEY, []) or [])[:6]]
    if tag_anchors:
        record.anchor_concepts = _dedupe([*anchors, *tag_anchors], max_items=24)
        record.evidence_anchors = _dedupe([*evidence_anchors, *tag_anchors], max_items=24)
    return record


def typed_tunnel_signature_text(metadata: Mapping[str, Any] | None) -> str:
    data = dict(metadata or {})
    parts = [
        data.get("tmcra_typed_event_signature", ""),
        " ".join(str(item) for item in list(data.get(NODE_TAG_KEY, []) or [])),
        " ".join(str(item) for item in list(data.get(PATH_TAG_KEY, []) or [])),
        data.get(TUNNEL_GROUP_KEY, ""),
    ]
    return _clean_text(" ".join(_clean_text(item) for item in parts if _clean_text(item)))


def typed_edge_tags_between(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> list[str]:
    left_data = dict(left or {})
    right_data = dict(right or {})
    tags: list[str] = []
    if _clean_text(left_data.get(ENTITY_SIGNATURE_KEY, "")) and _clean_text(left_data.get(ENTITY_SIGNATURE_KEY, "")) == _clean_text(right_data.get(ENTITY_SIGNATURE_KEY, "")):
        tags.append("same_entity")
    if _clean_text(left_data.get(ACTION_SIGNATURE_KEY, "")) and _clean_text(left_data.get(ACTION_SIGNATURE_KEY, "")) == _clean_text(right_data.get(ACTION_SIGNATURE_KEY, "")):
        tags.append("same_action")
    if _clean_text(left_data.get(MEASURE_SIGNATURE_KEY, "")) and _clean_text(left_data.get(MEASURE_SIGNATURE_KEY, "")) == _clean_text(right_data.get(MEASURE_SIGNATURE_KEY, "")):
        tags.append("same_measure")
    if _clean_text(left_data.get(TUNNEL_GROUP_KEY, "")) and _clean_text(left_data.get(TUNNEL_GROUP_KEY, "")) == _clean_text(right_data.get(TUNNEL_GROUP_KEY, "")):
        tags.append("same_tunnel_group")
    left_temporal = _clean_text(left_data.get(TEMPORAL_RELATION_KEY, ""))
    right_temporal = _clean_text(right_data.get(TEMPORAL_RELATION_KEY, ""))
    left_tags = set(left_data.get(NODE_TAG_KEY, []) or [])
    right_tags = set(right_data.get(NODE_TAG_KEY, []) or [])
    if left_temporal or right_temporal or ("timeline_event" in left_tags and "timeline_event" in right_tags):
        tags.append("same_timeline")
    if "current_state" in left_tags and "historical_state" in right_tags:
        tags.append("current_replaces_old")
    return _dedupe(tags, max_items=8)


def merge_typed_metadata(items: Iterable[Mapping[str, Any] | None]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        data = dict(item or {})
        for key in (NODE_TAG_KEY, PATH_TAG_KEY, TUNNEL_ROLE_KEY):
            merged[key] = _dedupe([*list(merged.get(key, []) or []), *list(data.get(key, []) or [])], max_items=24)
        for key in (
            TUNNEL_GROUP_KEY,
            TEMPORAL_RELATION_KEY,
            MEASURE_SIGNATURE_KEY,
            ACTION_SIGNATURE_KEY,
            ENTITY_SIGNATURE_KEY,
            "tmcra_typed_event_signature",
        ):
            if not _clean_text(merged.get(key, "")) and _clean_text(data.get(key, "")):
                merged[key] = _clean_text(data.get(key, ""))
    if merged:
        merged["tmcra_typed_tunnel_version"] = TYPED_TUNNEL_VERSION
    return merged
