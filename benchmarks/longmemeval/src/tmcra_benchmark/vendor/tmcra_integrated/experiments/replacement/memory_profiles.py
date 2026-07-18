from __future__ import annotations

from dataclasses import dataclass
import time
import re
from typing import Any, Dict, Iterable, List, Sequence

from .generic_memory import MemoryPolicy, MemoryProfile, MemoryRecord, MemoryRetrievalResult, MemorySessionScope
from .generic_memory import _clean_text, _dedupe, _normalize


def _slot_slug(value: Any, *, fallback: str = "default", max_length: int = 24) -> str:
    text = _normalize(value)
    if not text:
        return fallback
    parts = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)
    if not parts:
        return fallback
    slug = ".".join(parts[: max(1, max_length // 2)])
    slug = re.sub(r"\.+", ".", slug).strip(".")
    if not slug:
        return fallback
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip(".")
    return slug or fallback


def _query_tokens(value: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", _normalize(value))


def _strip_record_label_prefix(label: str) -> str:
    normalized = _normalize(label)
    if not normalized:
        return ""
    normalized = re.sub(
        r"^(?:current|previous|historical)\s+",
        "",
        normalized,
    )
    normalized = re.sub(r"^(?:当前|之前|历史)", "", normalized)
    return normalized.strip()


def _record_type_aliases(record_type: str, labels: "_TypeLabels") -> List[str]:
    aliases = _dedupe(
        [
            _normalize(record_type),
            _normalize(record_type.replace("_", " ")),
            _strip_record_label_prefix(labels.current_en),
            _strip_record_label_prefix(labels.previous_en),
            _strip_record_label_prefix(labels.current_zh),
            _strip_record_label_prefix(labels.previous_zh),
        ]
    )
    return [alias for alias in aliases if alias]


@dataclass(slots=True)
class _TypeLabels:
    current_en: str
    previous_en: str
    current_zh: str
    previous_zh: str


class StructuredMemoryProfile(MemoryProfile):
    profile_name = "structured"
    record_labels: Dict[str, _TypeLabels] = {}
    record_markers: Dict[str, Sequence[str]] = {}
    render_order: Dict[str, int] = {}

    def policy(self) -> MemoryPolicy:
        return MemoryPolicy()

    def normalize_event(self, event_kind: str, payload: Dict[str, Any] | None, scope: MemorySessionScope) -> Dict[str, Any]:
        normalized = super().normalize_event(event_kind, payload, scope)
        if normalized["records"]:
            return normalized
        return {**normalized, "records": self._records_from_payload(event_kind, normalized, scope)}

    def _records_from_payload(self, event_kind: str, payload: Dict[str, Any], scope: MemorySessionScope) -> List[Dict[str, Any]]:
        _ = event_kind, payload, scope
        return []

    def _build_record(
        self,
        raw: Dict[str, Any],
        *,
        event_kind: str,
        turn_index: int,
        created_at: float,
        scope: MemorySessionScope,
        index: int,
    ) -> MemoryRecord | None:
        record_type = _clean_text(raw.get("type", "")) or _clean_text(raw.get("category", "")) or "memory"
        value = _clean_text(raw.get("value", ""))
        if not value:
            return None
        namespace = _clean_text(raw.get("namespace", "")) or _clean_text(scope.namespace) or "default"
        attributes = dict(raw.get("attributes") or {})
        anchors = _dedupe(raw.get("anchors", []) or [])
        if anchors and not attributes.get("anchors"):
            attributes["anchors"] = anchors
        relation = _clean_text(raw.get("relation", ""))
        if relation and not attributes.get("relation"):
            attributes["relation"] = relation
        provenance = {
            **dict(raw.get("provenance") or {}),
            **dict(raw.get("metadata") or {}),
            "event_kind": _clean_text(event_kind),
        }
        key = self.resolve_key(
            record_type,
            value,
            attributes,
            namespace=namespace,
            provided_key=_clean_text(raw.get("key", "")) or _clean_text(raw.get("slot_key", "")) or _clean_text(raw.get("slot", "")),
            relation=relation,
            provenance=provenance,
        )
        state = _clean_text(raw.get("state", "")) or ("active" if bool(raw.get("active", True)) else "historical")
        record_id = _clean_text(raw.get("record_id", "")) or f"{scope.scope_id}:{namespace}:{key}:{turn_index}:{index}"
        timestamps = {
            **dict(raw.get("timestamps") or {}),
            "turn_index": int(turn_index),
            "created_at": float(raw.get("created_at", created_at) or created_at),
            "updated_at": float(raw.get("updated_at", created_at) or created_at),
        }
        return MemoryRecord(
            record_id=record_id,
            namespace=namespace,
            type=record_type,
            key=key,
            value=value,
            attributes=attributes,
            state=state,
            supersedes=_dedupe(raw.get("supersedes", []) or []),
            provenance=provenance,
            timestamps=timestamps,
            scope_id=scope.scope_id,
        )

    def derive_records(self, event_kind: str, payload: Dict[str, Any], scope: MemorySessionScope) -> List[MemoryRecord]:
        turn_index = int(payload.get("turn_index", 0) or 0)
        created_at = float(payload.get("created_at", time.time()) or time.time())
        records: List[MemoryRecord] = []
        for index, raw in enumerate(list(payload.get("records", []) or [])):
            if not isinstance(raw, dict):
                continue
            record = self._build_record(raw, event_kind=event_kind, turn_index=turn_index, created_at=created_at, scope=scope, index=index)
            if record is not None:
                records.append(record)
        return records

    def infer_query_types(self, query: str) -> List[str]:
        lowered = _normalize(query)
        results: List[str] = []
        for record_type, markers in self.record_markers.items():
            if any(_normalize(marker) in lowered for marker in markers):
                results.append(record_type)
        return _dedupe(results)

    def requested_category_order(self, query: str, hinted_types: Sequence[str] = ()) -> List[str]:
        ordered: List[str] = []
        for item in hinted_types:
            key = _normalize(item)
            if key and key not in ordered:
                ordered.append(key)
        lowered = _normalize(query)
        for record_type, markers in self.record_markers.items():
            if any(_normalize(marker) in lowered for marker in markers) and record_type not in ordered:
                ordered.append(record_type)
        return ordered

    def label(self, record_type: str, *, prefer_chinese: bool, historical: bool = False, variant: str = "compare") -> str:
        labels = self.record_labels.get(
            record_type,
            _TypeLabels(
                current_en=f"current {record_type}",
                previous_en=f"previous {record_type}",
                current_zh=f"当前{record_type}",
                previous_zh=f"之前{record_type}",
            ),
        )
        if prefer_chinese:
            return labels.previous_zh if historical else labels.current_zh
        if variant == "history" and historical:
            return labels.previous_en.replace("previous ", "historical ")
        return labels.previous_en if historical else labels.current_en

    def render_position(self, record_type: str) -> int:
        return int(self.render_order.get(_normalize(record_type), 99))

    def render_context(self, query: str, retrieval: MemoryRetrievalResult, scope: MemorySessionScope) -> Dict[str, Any]:
        lines = [
            f"{item.record.key} = {item.record.value}"
            for item in retrieval.records[:8]
        ]
        return {
            "profile_name": self.profile_name,
            "query": query,
            "scope": scope.to_dict(),
            "lines": lines,
            "record_ids": [item.record.record_id for item in retrieval.records],
        }


class TMCRAProfile(StructuredMemoryProfile):
    profile_name = "tmcra"
    record_labels = {
        "goal": _TypeLabels("current goal", "previous goal", "当前目标", "之前目标"),
        "constraint": _TypeLabels("current constraint", "previous constraint", "当前约束", "之前约束"),
        "preference": _TypeLabels("current preference", "previous preference", "当前偏好", "之前偏好"),
        "terminology": _TypeLabels("current terminology", "previous terminology", "当前术语", "之前术语"),
        "stage_state": _TypeLabels("current stage", "previous stage", "当前阶段", "之前阶段"),
        "path": _TypeLabels("current path", "previous path", "当前路径", "之前路径"),
    }
    record_markers = {
        "goal": ("goal", "mission", "objective", "target", "primary goal", "目标"),
        "constraint": ("constraint", "must", "forbid", "policy", "约束", "限制", "不要", "不能"),
        "preference": ("preference", "prefer", "default", "mode", "偏好", "优先"),
        "terminology": ("term", "terminology", "alias", "definition", "术语", "别名", "定义"),
        "stage_state": ("stage", "phase", "status", "state", "阶段", "进度", "状态"),
        "path": ("path", "route", "connect", "chain", "路径", "链路", "连接"),
    }
    render_order = {
        "goal": 0,
        "constraint": 1,
        "stage_state": 2,
        "preference": 3,
        "terminology": 4,
        "path": 5,
    }

    def policy(self) -> MemoryPolicy:
        return MemoryPolicy(writeback_mode="shadow")

    def infer_category_hints(self, query: str) -> List[str]:
        return self.infer_query_types(query)

    def stable_slot_key(
        self,
        *,
        category: str,
        value: str,
        anchors: Sequence[str],
        slot_key: str = "",
        relation: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        category_key = _normalize(category) or "memory"
        relation_key = _normalize(relation)
        raw_slot = _normalize(slot_key or (metadata or {}).get("slot_key") or (metadata or {}).get("slot") or "")
        anchor_slug = _slot_slug(anchors[0] if anchors else "", fallback="")
        value_slug = _slot_slug(value, fallback="value")
        if relation_key == "path_edge" and len(anchors) >= 2:
            left = _slot_slug(anchors[0], fallback="left")
            right = _slot_slug(anchors[1], fallback="right")
            return f"path.{left}.{right}"
        if raw_slot:
            raw_slot = re.sub(r"\s+", "", raw_slot)
            raw_slot = raw_slot.replace("terminology.", "term.").replace("terms.", "term.")
            raw_slot = raw_slot.replace("preferences.", "preference.").replace("constraints.", "constraint.")
            raw_slot = raw_slot.replace("stage_state.", "stage.").replace("stages.", "stage.")
            raw_slot = re.sub(r"\.+", ".", raw_slot).strip(".")
            if category_key == "goal" and raw_slot in {"goal", "goal.current", "goal.main", "goal.default"}:
                return "goal.primary"
            if category_key == "stage_state" and raw_slot in {"stage", "stage.current", "status", "state", "phase"}:
                return "stage.current"
            if category_key == "terminology" and raw_slot in {"term", "terminology", "term.current"}:
                return f"term.{anchor_slug or value_slug}"
            if category_key == "preference" and raw_slot in {"preference", "preference.current"}:
                return f"preference.{anchor_slug or 'default'}"
            if category_key == "constraint" and raw_slot in {"constraint", "constraint.current"}:
                return f"constraint.{anchor_slug or 'policy'}"
            return raw_slot
        if category_key == "goal":
            return "goal.primary"
        if category_key == "preference":
            return f"preference.{anchor_slug or 'default'}"
        if category_key == "constraint":
            return f"constraint.{anchor_slug or 'policy'}"
        if category_key == "terminology":
            return f"term.{anchor_slug or value_slug}"
        if category_key == "stage_state":
            return "stage.current"
        if category_key == "fact" and anchor_slug:
            return f"fact.{anchor_slug}.{value_slug}"
        return f"{category_key}.{anchor_slug or value_slug}"

    def resolve_key(
        self,
        record_type: str,
        value: str,
        attributes: Dict[str, Any],
        *,
        namespace: str,
        provided_key: str = "",
        relation: str = "",
        provenance: Dict[str, Any] | None = None,
    ) -> str:
        _ = namespace, provenance
        return self.stable_slot_key(
            category=record_type,
            value=value,
            anchors=list(attributes.get("anchors", []) or []),
            slot_key=provided_key,
            relation=relation or _clean_text(attributes.get("relation", "")),
            metadata=attributes,
        )

    def record_search_text(self, record: MemoryRecord) -> str:
        payload = {
            "type": record.type,
            "key": record.key,
            "value": record.value,
            "anchors": list(record.attributes.get("anchors", []) or []),
            "relation": _clean_text(record.attributes.get("relation", "")),
            "provenance": record.provenance,
        }
        return str(payload)


class AttributeMemoryProfile(StructuredMemoryProfile):
    profile_name = "attribute"
    record_labels = {
        "attribute": _TypeLabels("current attribute", "previous attribute", "当前属性", "之前属性"),
        "preference": _TypeLabels("current preference", "previous preference", "当前偏好", "之前偏好"),
    }
    record_markers = {
        "attribute": ("attribute", "profile", "email", "name", "role", "属性", "资料"),
        "preference": ("preference", "prefer", "setting", "偏好", "设置"),
    }
    render_order = {"preference": 0, "attribute": 1}

    def _records_from_payload(self, event_kind: str, payload: Dict[str, Any], scope: MemorySessionScope) -> List[Dict[str, Any]]:
        _ = event_kind, scope
        attributes = dict(payload.get("attributes") or {})
        if not attributes:
            return []
        entity = _clean_text(payload.get("entity", "")) or _clean_text(payload.get("subject", "")) or "user"
        record_type = _clean_text(payload.get("type", "")) or "attribute"
        return [
            {"type": record_type, "value": _clean_text(value), "attributes": {"entity": entity, "attribute": _clean_text(name)}}
            for name, value in attributes.items()
            if _clean_text(name) and _clean_text(value)
        ]

    def resolve_key(
        self,
        record_type: str,
        value: str,
        attributes: Dict[str, Any],
        *,
        namespace: str,
        provided_key: str = "",
        relation: str = "",
        provenance: Dict[str, Any] | None = None,
    ) -> str:
        _ = value, namespace, relation, provenance
        if _clean_text(provided_key):
            return _clean_text(provided_key)
        entity = _slot_slug(attributes.get("entity", "") or attributes.get("subject", ""), fallback="global")
        attribute_name = _slot_slug(attributes.get("attribute", "") or attributes.get("name", ""), fallback=record_type or "attribute")
        return f"attribute.{entity}.{attribute_name}"

    def render_context(self, query: str, retrieval: MemoryRetrievalResult, scope: MemorySessionScope) -> Dict[str, Any]:
        lines = [
            f"{item.record.attributes.get('entity', 'user')}.{item.record.attributes.get('attribute', item.record.key)} = {item.record.value}"
            for item in retrieval.records[:8]
        ]
        return {"profile_name": self.profile_name, "query": query, "scope": scope.to_dict(), "lines": lines}


class StateMemoryProfile(StructuredMemoryProfile):
    profile_name = "state"
    record_labels = {
        "state": _TypeLabels("current state", "previous state", "当前状态", "之前状态"),
        "stage": _TypeLabels("current stage", "previous stage", "当前阶段", "之前阶段"),
    }
    record_markers = {
        "state": ("state", "status", "workflow", "任务状态", "状态"),
        "stage": ("stage", "phase", "阶段"),
    }
    render_order = {"state": 0, "stage": 1}

    def _records_from_payload(self, event_kind: str, payload: Dict[str, Any], scope: MemorySessionScope) -> List[Dict[str, Any]]:
        _ = event_kind, scope
        state_value = _clean_text(payload.get("state", "")) or _clean_text(payload.get("status", ""))
        if not state_value:
            return []
        entity = _clean_text(payload.get("entity", "")) or _clean_text(payload.get("machine", "")) or "workflow"
        state_name = _clean_text(payload.get("state_name", "")) or "status"
        return [{"type": "state", "value": state_value, "attributes": {"entity": entity, "state_name": state_name}}]

    def resolve_key(
        self,
        record_type: str,
        value: str,
        attributes: Dict[str, Any],
        *,
        namespace: str,
        provided_key: str = "",
        relation: str = "",
        provenance: Dict[str, Any] | None = None,
    ) -> str:
        _ = value, namespace, relation, provenance
        if _clean_text(provided_key):
            return _clean_text(provided_key)
        entity = _slot_slug(attributes.get("entity", "") or attributes.get("machine", ""), fallback="workflow")
        state_name = _slot_slug(attributes.get("state_name", "") or attributes.get("name", ""), fallback="status")
        return f"state.{entity}.{state_name}"


class AliasMemoryProfile(StructuredMemoryProfile):
    profile_name = "alias"
    record_labels = {
        "alias": _TypeLabels("current alias", "previous alias", "当前别名", "之前别名"),
    }
    record_markers = {
        "alias": ("alias", "also called", "nickname", "别名", "叫做"),
    }
    render_order = {"alias": 0}

    def _records_from_payload(self, event_kind: str, payload: Dict[str, Any], scope: MemorySessionScope) -> List[Dict[str, Any]]:
        _ = event_kind, scope
        canonical = _clean_text(payload.get("canonical", "")) or _clean_text(payload.get("entity", ""))
        aliases = list(payload.get("aliases", []) or [])
        alias = _clean_text(payload.get("alias", ""))
        if alias:
            aliases.append(alias)
        if not canonical or not aliases:
            return []
        return [
            {"type": "alias", "value": _clean_text(item), "attributes": {"canonical": canonical, "alias": _clean_text(item)}}
            for item in aliases
            if _clean_text(item)
        ]

    def normalize_event(self, event_kind: str, payload: Dict[str, Any] | None, scope: MemorySessionScope) -> Dict[str, Any]:
        normalized = super().normalize_event(event_kind, payload, scope)
        expanded: List[Dict[str, Any]] = []
        for raw in list(normalized.get("records", []) or []):
            if not isinstance(raw, dict):
                continue
            aliases = list(raw.get("aliases", []) or [])
            alias = _clean_text(raw.get("alias", ""))
            if alias:
                aliases.append(alias)
            canonical = _clean_text(raw.get("canonical", "")) or _clean_text((raw.get("attributes") or {}).get("canonical", ""))
            if canonical and aliases:
                for item in aliases:
                    item_text = _clean_text(item)
                    if not item_text:
                        continue
                    expanded.append(
                        {
                            **raw,
                            "type": _clean_text(raw.get("type", "")) or "alias",
                            "value": item_text,
                            "attributes": {**dict(raw.get("attributes") or {}), "canonical": canonical, "alias": item_text},
                        }
                    )
                continue
            expanded.append(dict(raw))
        if expanded:
            normalized["records"] = expanded
        return normalized

    def resolve_key(
        self,
        record_type: str,
        value: str,
        attributes: Dict[str, Any],
        *,
        namespace: str,
        provided_key: str = "",
        relation: str = "",
        provenance: Dict[str, Any] | None = None,
    ) -> str:
        _ = record_type, namespace, relation, provenance
        if _clean_text(provided_key):
            return _clean_text(provided_key)
        canonical = _slot_slug(attributes.get("canonical", ""), fallback="entity")
        alias = _slot_slug(attributes.get("alias", "") or value, fallback="alias")
        return f"alias.{canonical}.{alias}"

    def record_search_text(self, record: MemoryRecord) -> str:
        canonical = _clean_text(record.attributes.get("canonical", ""))
        alias = _clean_text(record.attributes.get("alias", "")) or record.value
        return f"{record.type} {record.key} {alias} {canonical}"

    def build_relations(self, retrieval: MemoryRetrievalResult) -> List[Dict[str, Any]]:
        relations = super().build_relations(retrieval)
        for entry in retrieval.records:
            canonical = _clean_text(entry.record.attributes.get("canonical", ""))
            alias = _clean_text(entry.record.attributes.get("alias", "")) or entry.record.value
            if canonical and alias:
                relations.append(
                    {
                        "from": alias,
                        "to": canonical,
                        "relation": "alias_of",
                        "score": round(float(entry.score), 6),
                    }
                )
        return relations
