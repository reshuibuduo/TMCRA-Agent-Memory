from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=lambda item: str(item))]
    return value


@dataclass(slots=True)
class EvalCase:
    case_id: str
    query: str
    answer_mode: str = "transparent"
    category: str = "general"
    expected_keywords: List[str] = field(default_factory=list)
    expected_answer_keywords: List[str] = field(default_factory=list)
    expected_memory_values: List[str] = field(default_factory=list)
    expected_absent_values: List[str] = field(default_factory=list)
    expected_fact_phrases: List[str] = field(default_factory=list)
    expected_path_concepts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScenarioProfile:
    profile_id: str
    title: str
    description: str = ""


@dataclass(slots=True)
class LongDialogProbe:
    probe_id: str
    slot: str
    prompt: str
    expected_values: List[str] = field(default_factory=list)
    stale_values: List[str] = field(default_factory=list)
    false_values: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "slot": self.slot,
            "prompt": self.prompt,
            "expected_values": list(self.expected_values),
            "stale_values": list(self.stale_values),
            "false_values": list(self.false_values),
        }


@dataclass(slots=True)
class LongDialogProfile:
    profile_id: str
    title: str
    description: str = ""


@dataclass(slots=True)
class LLMProfile:
    name: str
    model: str
    base_url: str = ""
    api_key: str = ""
    system_prompt: str = "Use only the supplied evidence and memory."
    timeout_seconds: float = 60.0
    temperature: float = 0.1
    max_tokens: int = 256


@dataclass(slots=True)
class LeaderboardRecord:
    reasoner: str
    memory: str
    reasoning_quality_score: float
    memory_quality_score: float
    efficiency_score: float
    total_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoner": self.reasoner,
            "memory": self.memory,
            "reasoning_quality_score": round(float(self.reasoning_quality_score), 6),
            "memory_quality_score": round(float(self.memory_quality_score), 6),
            "efficiency_score": round(float(self.efficiency_score), 6),
            "total_score": round(float(self.total_score), 6),
            "metadata": _json_safe(self.metadata),
        }


@dataclass(slots=True)
class FailureRecord:
    benchmark: str
    reasoner: str = ""
    memory: str = ""
    case_id: str = ""
    probe_id: str = ""
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "reasoner": self.reasoner,
            "memory": self.memory,
            "case_id": self.case_id,
            "probe_id": self.probe_id,
            "reason": self.reason,
            "details": _json_safe(self.details),
        }


@dataclass(slots=True)
class MemoryHit:
    memory_id: str
    category: str
    value: str
    relation: str = "related_to"
    anchors: List[str] = field(default_factory=list)
    score: float = 0.0
    source_kind: str = "memory"
    slot_key: str = ""
    state: str = "active"
    turn_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "category": self.category,
            "value": self.value,
            "relation": self.relation,
            "anchors": list(self.anchors),
            "score": round(float(self.score), 6),
            "source_kind": self.source_kind,
            "slot_key": self.slot_key,
            "state": self.state,
            "turn_index": int(self.turn_index),
            "metadata": _json_safe(self.metadata),
        }


@dataclass(slots=True)
class MemoryRetrieval:
    concepts: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    hits: List[MemoryHit] = field(default_factory=list)
    active_hits: List[MemoryHit] = field(default_factory=list)
    history_hits: List[MemoryHit] = field(default_factory=list)
    stale_hits: List[MemoryHit] = field(default_factory=list)
    overwrite_hits: List[MemoryHit] = field(default_factory=list)
    false_hits: List[MemoryHit] = field(default_factory=list)
    retrieval_seconds: float = 0.0
    context_token_estimate: int = 0
    retrieval_context_token_estimate: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concepts": _json_safe(self.concepts),
            "relations": _json_safe(self.relations),
            "hits": [hit.to_dict() for hit in self.hits],
            "active_hits": [hit.to_dict() for hit in self.active_hits],
            "history_hits": [hit.to_dict() for hit in self.history_hits],
            "stale_hits": [hit.to_dict() for hit in self.stale_hits],
            "overwrite_hits": [hit.to_dict() for hit in self.overwrite_hits],
            "false_hits": [hit.to_dict() for hit in self.false_hits],
            "retrieval_seconds": round(float(self.retrieval_seconds), 6),
            "context_token_estimate": int(self.context_token_estimate),
            "retrieval_context_token_estimate": int(self.retrieval_context_token_estimate or self.context_token_estimate),
            "metadata": _json_safe(self.metadata),
        }


@dataclass(slots=True)
class AdapterResponse:
    answer: str
    answer_mode: str
    reasoner_name: str
    memory_name: str
    confidence: float = 0.0
    paths: List[Dict[str, Any]] = field(default_factory=list)
    facts: List[Dict[str, Any]] = field(default_factory=list)
    candidate_scores: List[Dict[str, Any]] = field(default_factory=list)
    memory_hits: List[Dict[str, Any]] = field(default_factory=list)
    evidence_consistent: bool = False
    unsupported_claims: List[str] = field(default_factory=list)
    pillar_scores: Dict[str, float] = field(default_factory=dict)
    latency_seconds: float = 0.0
    trace: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "answer_mode": self.answer_mode,
            "reasoner_name": self.reasoner_name,
            "memory_name": self.memory_name,
            "confidence": round(float(self.confidence), 6),
            "paths": _json_safe(self.paths),
            "facts": _json_safe(self.facts),
            "candidate_scores": _json_safe(self.candidate_scores),
            "memory_hits": _json_safe(self.memory_hits),
            "evidence_consistent": bool(self.evidence_consistent),
            "unsupported_claims": _json_safe(self.unsupported_claims),
            "pillar_scores": {key: round(float(value), 6) for key, value in self.pillar_scores.items()},
            "latency_seconds": round(float(self.latency_seconds), 6),
            "trace": _json_safe(self.trace),
            "metadata": _json_safe(self.metadata),
        }


class MemoryAdapter(ABC):
    name: str = "memory"

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def ingest_turn(
        self,
        user_text: str,
        assistant_text: str = "",
        *,
        answer_payload: Dict[str, Any] | None = None,
        extraction_result: Dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, *, top_k: int = 6) -> MemoryRetrieval:
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def storage_bytes(self) -> int:
        raise NotImplementedError

    def export_dialog_graph(self) -> Dict[str, Any]:
        return {"summary": {"supported": False}}

    def export_dialog_graph_mermaid(self) -> str:
        return "graph TD\n"

    def register_answer_support(
        self,
        *,
        answer_id: str,
        memory_ids: List[str],
        query_id: str = "",
        answer_text: str = "",
    ) -> None:
        _ = answer_id, memory_ids, query_id, answer_text

    def ingest_answer_writeback(
        self,
        *,
        query_text: str,
        answer_text: str,
        answer_id: str,
        writeback_records: List[Dict[str, Any]],
        trace: Dict[str, Any] | None = None,
    ) -> List[str]:
        _ = query_text, answer_text, answer_id, trace
        if not writeback_records:
            return []
        self.ingest_turn(
            query_text,
            answer_text,
            answer_payload={"replacement_memory_records": list(writeback_records), "metadata": {"memory_write": True, "source": "assistant_writeback"}},
            extraction_result={},
        )
        return [str(item.get("memory_id", "")) for item in writeback_records if isinstance(item, dict) and str(item.get("memory_id", "")).strip()]

    def telemetry_snapshot(self) -> Dict[str, Any]:
        return {}

    def build_prompt_context(self, query: str, *, top_k: int = 8) -> Dict[str, Any]:
        return {
            "mode": "retrieval",
            "query": query,
            "retrieval": self.retrieve(query, top_k=top_k).to_dict(),
            "stats": self.stats(),
        }


class ReasoningAdapter(ABC):
    name: str = "reasoner"

    @abstractmethod
    async def answer(
        self,
        query: str,
        *,
        answer_mode: str,
        memory_adapter: MemoryAdapter,
    ) -> AdapterResponse:
        raise NotImplementedError
