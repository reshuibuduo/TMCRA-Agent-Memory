from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List


_GOAL_MARKERS = (
    "目标",
    "想要",
    "希望",
    "实现",
    "做成",
    "达成",
    "验证",
    "评估",
    "提升",
    "重构",
    "升级",
    "训练",
    "支持",
    "切换",
)
_PREFERENCE_MARKERS = (
    "默认",
    "优先",
    "保留",
    "保持",
    "透明",
    "自然",
    "可解释",
    "开源",
    "本地",
)
_CONSTRAINT_MARKERS = (
    "不要",
    "必须",
    "不能",
    "仅",
    "只做",
    "只用",
    "禁止",
)
_VAGUE_QUERY_MARKERS = (
    "继续",
    "接着",
    "然后",
    "当前计划",
    "这个体系",
    "这个系统",
    "方案",
    "当前",
)
_RELATION_BY_CATEGORY = {
    "goal": "session_goal",
    "preference": "prefers",
    "terminology": "uses_term",
    "constraint": "constrained_by",
    "stage_state": "stage_state",
}
_TYPE_BY_CATEGORY = {
    "goal": "session_goal",
    "preference": "session_preference",
    "terminology": "session_term",
    "constraint": "session_constraint",
    "stage_state": "session_stage",
}


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_key(value: Any) -> str:
    return _clean_text(value).lower()


def _clip_text(value: Any, max_len: int = 72) -> str:
    text = _clean_text(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _dedupe_texts(values: Iterable[Any], max_items: int | None = None) -> List[str]:
    items: List[str] = []
    seen = set()
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        key = _normalize_key(text)
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
        if max_items is not None and len(items) >= max_items:
            break
    return items


def _split_fragments(text: str) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    fragments = re.split(r"[，,。；;！!？?\n]+", text)
    return _dedupe_texts(fragments)


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    return any(marker in text for marker in markers)


def _match_fragments(text: str, markers: Iterable[str], max_items: int = 2) -> List[str]:
    matched = [fragment for fragment in _split_fragments(text) if _contains_any(fragment, markers)]
    return _dedupe_texts(matched, max_items=max_items)


def _concept_order(
    understanding_result: Dict[str, Any] | None,
    extraction_result: Dict[str, Any] | None,
    answer_bundle: Dict[str, Any] | None,
) -> List[str]:
    concepts: List[str] = []
    if isinstance(answer_bundle, dict):
        concepts.extend(answer_bundle.get("core_concepts", []) or [])
        focus = answer_bundle.get("focus_concept")
        if focus:
            concepts.insert(0, focus)
    if isinstance(understanding_result, dict):
        focus = understanding_result.get("focus_concept")
        if focus:
            concepts.insert(0, focus)
    if isinstance(extraction_result, dict):
        for item in extraction_result.get("concepts", []) or []:
            if isinstance(item, dict):
                concepts.append(item.get("concept", ""))
    return _dedupe_texts(concepts, max_items=12)


_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)


def _token_set(value: Any) -> set[str]:
    text = _normalize_key(value)
    if not text:
        return set()
    return {token for token in _TOKEN_RE.findall(text) if token}


def _category_slot_key(category: str, value: str) -> str:
    normalized_category = _normalize_key(category)
    if normalized_category == "goal":
        return "goal.primary"
    if normalized_category == "preference":
        return "preference.active"
    if normalized_category == "constraint":
        return "constraint.active"
    if normalized_category == "stage_state":
        return "stage.current"
    if normalized_category == "terminology":
        return f"terminology.{_normalize_key(value)[:24]}"
    return f"{normalized_category}.{_normalize_key(value)[:24]}"


def _anchor_signature(anchors: Iterable[Any]) -> str:
    normalized = sorted({_normalize_key(anchor) for anchor in anchors if _clean_text(anchor)})
    return "|".join(normalized)


def _state_signature(*, category: str, slot_key: str, anchors: Iterable[Any]) -> str:
    components = [_normalize_key(category), _normalize_key(slot_key), _anchor_signature(anchors)]
    return "|".join(component for component in components if component)


def _memory_signature(*, category: str, value: str, slot_key: str, anchors: Iterable[Any]) -> str:
    components = [
        _state_signature(category=category, slot_key=slot_key, anchors=anchors),
        _normalize_key(value),
    ]
    return "|".join(component for component in components if component)


def _event_signature(*, category: str, value: str, slot_key: str, anchors: Iterable[Any]) -> str:
    category_label = _normalize_key(category).replace("_", " ")
    slot_label = _normalize_key(slot_key).replace(".", " ").replace("_", " ")
    parts = [category_label, slot_label]
    parts.extend(_clean_text(anchor) for anchor in anchors if _clean_text(anchor))
    parts.append(_clean_text(value))
    return " ".join(_dedupe_texts(parts, max_items=8))


@dataclass(slots=True)
class SessionMemoryRecord:
    memory_id: str
    category: str
    value: str
    relation: str
    anchor_concepts: List[str] = field(default_factory=list)
    salience: float = 0.6
    confidence: float = 0.6
    source_kind: str = "session_memory"
    turn_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str, str]:
        anchor_key = "|".join(sorted(_normalize_key(item) for item in self.anchor_concepts))
        return (_normalize_key(self.category), _normalize_key(self.value), anchor_key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "category": self.category,
            "value": self.value,
            "relation": self.relation,
            "anchor_concepts": list(self.anchor_concepts),
            "salience": round(float(self.salience), 4),
            "confidence": round(float(self.confidence), 4),
            "source_kind": self.source_kind,
            "turn_index": int(self.turn_index),
            "metadata": dict(self.metadata),
        }

    def to_hit(self, relevance: float) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "category": self.category,
            "value": self.value,
            "relation": self.relation,
            "anchors": list(self.anchor_concepts),
            "salience": round(float(self.salience), 4),
            "confidence": round(float(self.confidence), 4),
            "relevance": round(float(relevance), 4),
            "source_kind": self.source_kind,
            "turn_index": int(self.turn_index),
            "metadata": dict(self.metadata),
        }


class SessionMemoryExtractor:
    def extract(
        self,
        query: str,
        understanding_result: Dict[str, Any] | None = None,
        extraction_result: Dict[str, Any] | None = None,
        answer_bundle: Dict[str, Any] | None = None,
        *,
        answer_mode: str = "natural",
        turn_index: int = 0,
    ) -> List[SessionMemoryRecord]:
        query_text = _clean_text(query)
        if not query_text:
            return []

        anchors = _concept_order(understanding_result, extraction_result, answer_bundle)
        records: List[SessionMemoryRecord] = []

        for constraint in self._extract_constraints(query_text, understanding_result):
            records.append(
                self._make_record(
                    category="constraint",
                    value=constraint,
                    anchors=anchors,
                    salience=0.88,
                    confidence=0.82,
                    turn_index=turn_index,
                )
            )

        for preference in self._extract_preferences(query_text, answer_mode):
            records.append(
                self._make_record(
                    category="preference",
                    value=preference,
                    anchors=anchors,
                    salience=0.82,
                    confidence=0.8,
                    turn_index=turn_index,
                )
            )

        for goal in self._extract_goals(query_text, anchors):
            records.append(
                self._make_record(
                    category="goal",
                    value=goal,
                    anchors=anchors,
                    salience=0.9,
                    confidence=0.78,
                    turn_index=turn_index,
                )
            )

        for stage_state in self._extract_stage_states(query_text):
            records.append(
                self._make_record(
                    category="stage_state",
                    value=stage_state,
                    anchors=anchors,
                    salience=0.74,
                    confidence=0.72,
                    turn_index=turn_index,
                )
            )

        for term in self._extract_terms(query_text, anchors):
            records.append(
                self._make_record(
                    category="terminology",
                    value=term,
                    anchors=[term],
                    salience=0.64,
                    confidence=0.68,
                    turn_index=turn_index,
                )
            )

        merged: Dict[tuple[str, str, str], SessionMemoryRecord] = {}
        for record in records:
            key = record.key()
            existing = merged.get(key)
            if existing is None:
                merged[key] = record
                continue
            existing.salience = max(existing.salience, record.salience)
            existing.confidence = max(existing.confidence, record.confidence)
            existing.turn_index = max(existing.turn_index, record.turn_index)
            existing.anchor_concepts = _dedupe_texts([*existing.anchor_concepts, *record.anchor_concepts], max_items=8)
        return list(merged.values())

    def _make_record(
        self,
        *,
        category: str,
        value: str,
        anchors: List[str],
        salience: float,
        confidence: float,
        turn_index: int,
    ) -> SessionMemoryRecord:
        clean_value = _clip_text(value)
        clean_anchors = _dedupe_texts(anchors, max_items=6)
        slot_key = _category_slot_key(category, clean_value)
        state_signature = _state_signature(category=category, slot_key=slot_key, anchors=clean_anchors)
        memory_signature = _memory_signature(category=category, value=clean_value, slot_key=slot_key, anchors=clean_anchors)
        event_signature = _event_signature(category=category, value=clean_value, slot_key=slot_key, anchors=clean_anchors)
        memory_id = f"{category}:{_normalize_key(clean_value)}"
        return SessionMemoryRecord(
            memory_id=memory_id,
            category=category,
            value=clean_value,
            relation=_RELATION_BY_CATEGORY.get(category, "related_to"),
            anchor_concepts=clean_anchors,
            salience=max(0.0, min(1.0, float(salience))),
            confidence=max(0.0, min(1.0, float(confidence))),
            turn_index=int(turn_index),
            metadata={
                "slot_key": slot_key,
                "state_signature": state_signature,
                "memory_signature": memory_signature,
                "event_signature": event_signature,
                "anchor_signature": _anchor_signature(clean_anchors),
            },
        )

    def _extract_goals(self, query_text: str, anchors: List[str]) -> List[str]:
        goals = _match_fragments(query_text, _GOAL_MARKERS, max_items=2)
        if goals:
            return goals
        if _contains_any(query_text, ("继续", "接着")) and anchors:
            return [f"围绕 {anchors[0]} 持续推进"]
        return []

    def _extract_preferences(self, query_text: str, answer_mode: str) -> List[str]:
        preferences: List[str] = []
        lowered = query_text.lower()
        if "transparent" in lowered or "透明" in query_text or "可解释" in query_text or answer_mode == "transparent":
            preferences.append("transparent_mode")
        if "natural" in lowered or "自然" in query_text or answer_mode == "natural":
            preferences.append("natural_mode")
        if "开源" in query_text:
            preferences.append("prefer_open_models")
        if any(marker in query_text for marker in ("本地", "离线", "私有部署")):
            preferences.append("prefer_local_modules")
        if _contains_any(query_text, _PREFERENCE_MARKERS):
            preferences.extend(_match_fragments(query_text, _PREFERENCE_MARKERS, max_items=2))
        return _dedupe_texts(preferences, max_items=4)

    def _extract_constraints(self, query_text: str, understanding_result: Dict[str, Any] | None) -> List[str]:
        constraints: List[str] = []
        if isinstance(understanding_result, dict):
            constraints.extend(understanding_result.get("constraints", []) or [])
        constraints.extend(_match_fragments(query_text, _CONSTRAINT_MARKERS, max_items=3))
        return [_clip_text(item, max_len=64) for item in _dedupe_texts(constraints, max_items=4)]

    def _extract_terms(self, query_text: str, anchors: List[str]) -> List[str]:
        terms: List[str] = []
        for anchor in anchors[:6]:
            if len(anchor) >= 2 and anchor in query_text:
                terms.append(anchor)
        english_terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", query_text)
        terms.extend(english_terms[:4])
        return _dedupe_texts(terms, max_items=6)

    def _extract_stage_states(self, query_text: str) -> List[str]:
        states: List[str] = []
        if "继续" in query_text or "接着" in query_text:
            states.append("continue_current_plan")
        if "开始" in query_text or "启动" in query_text:
            states.append("execution_started")
        if "验证" in query_text or "测试" in query_text or "smoke" in query_text.lower():
            states.append("validation_phase")
        if "训练" in query_text:
            states.append("training_phase")
        if "实施" in query_text or "落地" in query_text or "重构" in query_text:
            states.append("implementation_phase")
        return _dedupe_texts(states, max_items=3)


class SessionMemoryGraph:
    def __init__(self, records: Iterable[SessionMemoryRecord] | None = None, turn_index: int = 0):
        self.turn_index = int(turn_index)
        self._records: Dict[tuple[str, str, str], SessionMemoryRecord] = {}
        for record in records or []:
            self.add_records([record])

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "SessionMemoryGraph":
        if not isinstance(payload, dict):
            return cls()
        records = []
        for item in payload.get("records", []) or []:
            if not isinstance(item, dict):
                continue
            records.append(
                SessionMemoryRecord(
                    memory_id=_clean_text(item.get("memory_id")),
                    category=_clean_text(item.get("category")),
                    value=_clean_text(item.get("value")),
                    relation=_clean_text(item.get("relation")) or "related_to",
                    anchor_concepts=_dedupe_texts(item.get("anchor_concepts", []) or [], max_items=8),
                    salience=float(item.get("salience", 0.6) or 0.6),
                    confidence=float(item.get("confidence", 0.6) or 0.6),
                    source_kind=_clean_text(item.get("source_kind")) or "session_memory",
                    turn_index=int(item.get("turn_index", 0) or 0),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        return cls(records=records, turn_index=int(payload.get("turn_index", 0) or 0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": int(self.turn_index),
            "records": [record.to_dict() for record in self.records()],
        }

    def next_turn(self) -> int:
        self.turn_index += 1
        return self.turn_index

    def records(self) -> List[SessionMemoryRecord]:
        ordered = sorted(
            self._records.values(),
            key=lambda item: (item.turn_index, item.salience, item.confidence),
            reverse=True,
        )
        return ordered

    def record_count(self) -> int:
        return len(self._records)

    def add_records(self, records: Iterable[SessionMemoryRecord]) -> int:
        stored = 0
        for record in records:
            key = record.key()
            existing = self._records.get(key)
            if existing is None:
                self._records[key] = record
                stored += 1
                continue
            existing.salience = max(existing.salience, record.salience)
            existing.confidence = max(existing.confidence, record.confidence)
            existing.turn_index = max(existing.turn_index, record.turn_index)
            existing.anchor_concepts = _dedupe_texts([*existing.anchor_concepts, *record.anchor_concepts], max_items=8)
            existing.metadata.update(record.metadata)
        return stored

    def build_context(
        self,
        query: str,
        understanding_result: Dict[str, Any] | None = None,
        extraction_result: Dict[str, Any] | None = None,
        answer_bundle: Dict[str, Any] | None = None,
        *,
        max_records: int = 8,
    ) -> Dict[str, Any]:
        anchors = _concept_order(understanding_result, extraction_result, answer_bundle)
        query_text = _clean_text(query)
        vague_query = not anchors or _contains_any(query_text, _VAGUE_QUERY_MARKERS)
        ranked: List[tuple[float, SessionMemoryRecord]] = []
        for record in self.records():
            relevance = self._relevance_score(record, query=query_text, anchors=anchors, vague_query=vague_query)
            if relevance <= 0.0:
                continue
            ranked.append((relevance, record))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[: max(0, int(max_records))]

        concepts: Dict[str, Dict[str, Any]] = {}
        relations: List[Dict[str, Any]] = []
        memory_hits: List[Dict[str, Any]] = []
        relation_seen = set()

        for relevance, record in selected:
            target = record.value
            concepts.setdefault(
                target,
                {
                    "concept": target,
                    "type": _TYPE_BY_CATEGORY.get(record.category, "session_memory"),
                    "source_kind": "session_memory",
                },
            )

            relation_added = False
            for anchor in record.anchor_concepts[:4]:
                if not anchor or anchor == target:
                    continue
                concepts.setdefault(anchor, {"concept": anchor, "type": "general"})
                relation_key = (anchor, target, record.relation)
                if relation_key not in relation_seen:
                    relation_seen.add(relation_key)
                    relations.append(
                        {
                            "from": anchor,
                            "to": target,
                            "relation": record.relation,
                            "weight": round(max(0.35, min(0.95, 0.45 + record.salience * 0.4)), 4),
                            "source_kind": "session_memory",
                            "memory_id": record.memory_id,
                        }
                    )
                back_key = (target, anchor, "context_for")
                if back_key not in relation_seen:
                    relation_seen.add(back_key)
                    relations.append(
                        {
                            "from": target,
                            "to": anchor,
                            "relation": "context_for",
                            "weight": round(max(0.25, min(0.85, 0.3 + record.confidence * 0.35)), 4),
                            "source_kind": "session_memory",
                            "memory_id": record.memory_id,
                        }
                    )
                relation_added = True

            if not relation_added and anchors:
                anchor = anchors[0]
                if anchor and anchor != target:
                    relation_key = (anchor, target, record.relation)
                    if relation_key not in relation_seen:
                        relation_seen.add(relation_key)
                        relations.append(
                            {
                                "from": anchor,
                                "to": target,
                                "relation": record.relation,
                                "weight": round(max(0.35, min(0.95, 0.45 + record.salience * 0.4)), 4),
                                "source_kind": "session_memory",
                                "memory_id": record.memory_id,
                            }
                        )

            memory_hits.append(record.to_hit(relevance))

        return {
            "concepts": list(concepts.values()),
            "relations": relations,
            "memory_hits": memory_hits,
        }

    def _relevance_score(self, record: SessionMemoryRecord, *, query: str, anchors: List[str], vague_query: bool) -> float:
        query_tokens = _token_set(query)
        anchor_set = {_normalize_key(item) for item in anchors}
        record_anchor_set = {_normalize_key(item) for item in record.anchor_concepts}
        overlap = len(anchor_set & record_anchor_set)
        value_overlap = len(query_tokens & _token_set(record.value))
        metadata = dict(record.metadata or {})
        slot_key_tokens = _token_set(metadata.get("slot_key", ""))
        state_signature_tokens = _token_set(metadata.get("state_signature", ""))
        memory_signature_tokens = _token_set(metadata.get("memory_signature", ""))
        event_signature_tokens = _token_set(metadata.get("event_signature", ""))
        recency = 1.0 if self.turn_index <= 0 else max(0.0, 1.0 - ((self.turn_index - record.turn_index) / max(1.0, float(self.turn_index))))
        score = record.salience * 0.5 + record.confidence * 0.25 + recency * 0.15
        if overlap > 0:
            score += min(0.4, overlap * 0.15)
        if value_overlap > 0:
            score += min(0.22, value_overlap * 0.08)
        if query_tokens and event_signature_tokens:
            score += min(0.22, (len(query_tokens & event_signature_tokens) / max(1, len(query_tokens))) * 0.22)
        if query_tokens and slot_key_tokens:
            score += min(0.16, (len(query_tokens & slot_key_tokens) / max(1, len(query_tokens))) * 0.16)
        if query_tokens and state_signature_tokens:
            score += min(0.18, (len(query_tokens & state_signature_tokens) / max(1, len(query_tokens))) * 0.18)
        if query_tokens and memory_signature_tokens:
            score += min(0.24, (len(query_tokens & memory_signature_tokens) / max(1, len(query_tokens))) * 0.24)
        elif not vague_query and overlap <= 0 and value_overlap <= 0:
            score -= 0.18
        if vague_query and record.category in {"goal", "preference", "stage_state"}:
            score += 0.12
        return max(0.0, min(1.5, score))


def create_tmcra_reasoning_adapter(*, flags: Any | None = None) -> Any:
    from core.tmcra_reasoning_runtime import create_reasoning_v2_shadow_adapter

    return create_reasoning_v2_shadow_adapter(flags=flags)
