"""User-facing narrative projections for committed TMCRA memory graphs.

The production memory graph remains the retrieval and audit substrate.  This
module derives a smaller, stable view for people: semantic records become key
moments, immutable Source records remain evidence-only, and explicit temporal
links make each topic readable as a storyline.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


NARRATIVE_SCHEMA_VERSION = "tmcra.narrative-graph.1"
NARRATIVE_FOCI = frozenset(
    {"all", "decision", "milestone", "goal", "issue", "preference", "relationship", "fact"}
)
_OPAQUE_KEY = re.compile(r"(?:^[0-9a-f]{20,}$|^[0-9a-f-]{32,}$|^personal-[0-9a-f-]+$)", re.I)
_NON_WORD = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)

_KIND_SIGNALS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("decision", ("decision", "decide", "chosen", "choose", "selected", "approved", "adopt", "决定", "选择", "采用", "批准")),
    ("milestone", ("milestone", "complete", "completed", "finish", "finished", "launch", "release", "deploy", "result", "outcome", "完成", "上线", "发布", "部署", "结果", "进展")),
    ("issue", ("issue", "problem", "error", "failure", "failed", "risk", "block", "bug", "问题", "错误", "失败", "风险", "阻塞", "故障")),
    ("goal", ("goal", "plan", "task", "requirement", "request", "intent", "objective", "目标", "计划", "任务", "需求", "要求")),
    ("preference", ("preference", "prefer", "like", "style", "habit", "偏好", "喜欢", "习惯", "风格")),
    ("relationship", ("person", "people", "relationship", "team", "family", "friend", "colleague", "联系人", "人物", "关系", "团队", "家人", "朋友", "同事")),
)
_KIND_PRIORITY = {
    "decision": 1.7,
    "milestone": 1.5,
    "issue": 1.35,
    "goal": 1.2,
    "preference": 1.0,
    "relationship": 0.9,
    "fact": 0.5,
}


class NarrativeGraphError(ValueError):
    pass


def build_narrative_graph(
    graph: Mapping[str, Any],
    *,
    limit: int = 36,
    focus: str = "all",
) -> dict[str, Any]:
    """Project one raw graph page into an evidence-bound narrative view."""

    focus = str(focus or "all").strip().lower()
    if focus not in NARRATIVE_FOCI:
        raise NarrativeGraphError("focus must be all or a supported narrative type")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 60:
        raise NarrativeGraphError("limit must be an integer between 1 and 60")

    raw_nodes = [dict(item) for item in _items(graph.get("nodes"))]
    raw_edges = [dict(item) for item in _items(graph.get("edges"))]
    semantic_nodes = [item for item in raw_nodes if _text(item.get("layer")) != "source"]
    typed_nodes = [(item, _narrative_kind(item)) for item in semantic_nodes]
    if focus != "all":
        typed_nodes = [(item, kind) for item, kind in typed_nodes if kind == focus]

    grouped: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for node, kind in typed_nodes:
        grouped[_thread_key(node)].append((node, kind))

    raw_degree = Counter()
    for edge in raw_edges:
        source, target = _text(edge.get("source")), _text(edge.get("target"))
        if source:
            raw_degree[source] += 1
        if target:
            raw_degree[target] += 1

    ranked_threads = sorted(
        grouped.items(),
        key=lambda item: (
            -max((_node_score(node, kind, raw_degree) for node, kind in item[1]), default=0.0),
            item[0],
        ),
    )[:8]
    selected_by_thread: dict[str, list[tuple[dict[str, Any], str]]] = {}
    remaining: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for key, records in ranked_threads:
        ranked = sorted(records, key=lambda item: (-_node_score(item[0], item[1], raw_degree), _node_id(item[0])))
        selected_by_thread[key] = ranked[:1]
        remaining[key] = ranked[1:]

    while sum(len(items) for items in selected_by_thread.values()) < limit:
        changed = False
        for key, _ in ranked_threads:
            if not remaining[key]:
                continue
            selected_by_thread[key].append(remaining[key].pop(0))
            changed = True
            if sum(len(items) for items in selected_by_thread.values()) >= limit:
                break
        if not changed:
            break

    thread_metadata: dict[str, dict[str, Any]] = {}
    narrative_nodes: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for thread_index, (key, all_records) in enumerate(ranked_threads):
        chosen = selected_by_thread.get(key, [])
        if not chosen:
            continue
        ordered = sorted(chosen, key=lambda item: _temporal_key(item[0]))
        representative, representative_kind = max(
            all_records,
            key=lambda item: (_node_score(item[0], item[1], raw_degree), _node_id(item[0])),
        )
        thread_id = _stable_id("thread", key)
        title = _thread_title(key, representative)
        full_ordered = sorted(all_records, key=lambda item: _temporal_key(item[0]))
        occurred = [_text(node.get("occurred_at")) for node, _ in full_ordered]
        occurred = [value for value in occurred if value]
        kinds = Counter(kind for _, kind in all_records)
        dominant_kind = kinds.most_common(1)[0][0] if kinds else representative_kind
        thread_node_ids: list[str] = []
        for sequence_index, (node, kind) in enumerate(ordered):
            identifier = _node_id(node)
            if not identifier or identifier in selected_ids:
                continue
            selected_ids.add(identifier)
            thread_node_ids.append(identifier)
            attributes = dict(node.get("attributes")) if isinstance(node.get("attributes"), Mapping) else {}
            attributes.update(
                {
                    "narrative_type": kind,
                    "thread_id": thread_id,
                    "thread_title": title,
                    "thread_index": thread_index,
                    "sequence_index": sequence_index,
                    "is_key_moment": kind in {"decision", "milestone", "issue"} or _text(node.get("layer")) == "slow",
                    "projection_source": "committed_slow_fast_graph",
                    "evidence_memory_id": identifier,
                }
            )
            projected = dict(node)
            projected.update(
                {
                    "kind": kind,
                    "label": _short(_text(node.get("label") or node.get("summary")) or title, 96),
                    "summary": _short(_text(node.get("summary") or node.get("label")) or title, 1_200),
                    "attributes": attributes,
                }
            )
            narrative_nodes.append(projected)
        thread_metadata[key] = {
            "id": thread_id,
            "title": title,
            "summary": _short(_text(representative.get("summary") or representative.get("label")) or title, 420),
            "kind": dominant_kind,
            "status": _text(full_ordered[-1][0].get("status") or full_ordered[-1][0].get("state")) or "active",
            "node_ids": thread_node_ids,
            "memory_count": len(all_records),
            "evidence_count": sum(max(0, _integer(node.get("evidence_count"))) for node, _ in all_records),
            "started_at": min(occurred) if occurred else None,
            "updated_at": max(occurred) if occurred else None,
        }

    selected = {node["id"]: node for node in narrative_nodes}
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def add_edge(source: str, target: str, relation: str, weight: float, provenance: Mapping[str, Any]) -> None:
        if source not in selected or target not in selected or source == target:
            return
        key = (source, target, relation)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append(
            {
                "id": _stable_id("narrative-edge", "|".join(key)),
                "source": source,
                "target": target,
                "type": relation,
                "weight": max(0.0, min(1.0, float(weight))),
                "origin": "derived",
                "provenance": dict(provenance),
            }
        )

    for edge in raw_edges:
        source, target = _text(edge.get("source")), _text(edge.get("target"))
        relation = _text(edge.get("type")) or "related"
        add_edge(
            source,
            target,
            relation,
            _number(edge.get("weight"), 0.6),
            {
                "source": "production_memory_edge",
                "source_edge_id": _text(edge.get("id")) or None,
                "source_origin": _text(edge.get("origin")) or None,
            },
        )

    for key, records in selected_by_thread.items():
        ordered_ids = [
            _node_id(node)
            for node, _ in sorted(records, key=lambda item: _temporal_key(item[0]))
            if _node_id(node) in selected
        ]
        for source, target in zip(ordered_ids, ordered_ids[1:]):
            add_edge(
                source,
                target,
                "followed_by",
                0.72,
                {"source": "narrative_chronology", "thread_id": thread_metadata[key]["id"]},
            )

    threads = [thread_metadata[key] for key, _ in ranked_threads if key in thread_metadata]
    times = [_text(node.get("occurred_at")) for node in narrative_nodes]
    times = [value for value in times if value]
    top_titles = [item["title"] for item in threads[:3]]
    source_page = graph.get("page") if isinstance(graph.get("page"), Mapping) else {}
    source_truncated = bool(source_page.get("truncated"))
    return {
        "schema_version": NARRATIVE_SCHEMA_VERSION,
        "scope_name": _text(graph.get("scope_name")),
        "snapshot_id": _text(graph.get("snapshot_id")),
        "snapshot_state": _text(graph.get("snapshot_state")) or "committed",
        "provisional": bool(graph.get("provisional")),
        "view": "narrative",
        "requested_layers": ["slow", "fast"],
        "resolved_layers": list(graph.get("resolved_layers") or ["slow", "fast"]),
        "fallback_layer": graph.get("fallback_layer"),
        "nodes": narrative_nodes,
        "edges": edges,
        "counts": {
            "nodes": len(narrative_nodes),
            "edges": len(edges),
            "slow": sum(_text(node.get("layer")) == "slow" for node in narrative_nodes),
            "fast": sum(_text(node.get("layer")) == "fast" for node in narrative_nodes),
            "source": 0,
        },
        "page": {
            "limit": limit,
            "offset": 0,
            "truncated": source_truncated or len(typed_nodes) > len(narrative_nodes),
            "next_cursor": None,
        },
        "threads": threads,
        "narrative": {
            "headline": top_titles[0] if top_titles else "Memory storyline",
            "summary": " · ".join(top_titles),
            "thread_count": len(threads),
            "key_moment_count": len(narrative_nodes),
            "evidence_count": sum(item["evidence_count"] for item in threads),
            "started_at": min(times) if times else None,
            "updated_at": max(times) if times else None,
            "focus": focus,
            "source_schema_version": _text(graph.get("schema_version")),
            "source_node_count": len(raw_nodes),
            "source_truncated": source_truncated,
            "projection_strategy": "slow_first_evidence_bound_v1",
            "semantic_source": "model_curated_slow_graph_with_fast_graph_fallback",
        },
    }


def _items(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def _integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _short(value: str, maximum: int) -> str:
    value = " ".join(value.split())
    return value if len(value) <= maximum else value[: maximum - 1].rstrip() + "…"


def _node_id(node: Mapping[str, Any]) -> str:
    return _text(node.get("id"))


def _narrative_kind(node: Mapping[str, Any]) -> str:
    attributes = node.get("attributes") if isinstance(node.get("attributes"), Mapping) else {}
    for value in (
        node.get("category"),
        node.get("kind"),
        attributes.get("memory_type"),
        attributes.get("memory_family"),
    ):
        explicit_kind = _text(value).lower()
        if explicit_kind in NARRATIVE_FOCI and explicit_kind != "all":
            return explicit_kind
    haystack = " ".join(
        _text(value).lower()
        for value in (
            node.get("kind"), node.get("category"), node.get("relation"),
            node.get("label"), attributes.get("memory_type"), attributes.get("memory_family"),
        )
    )
    for kind, signals in _KIND_SIGNALS:
        if any(signal in haystack for signal in signals):
            return kind
    return "fact"


def _thread_key(node: Mapping[str, Any]) -> str:
    attributes = node.get("attributes") if isinstance(node.get("attributes"), Mapping) else {}
    candidates = (
        node.get("cluster_id"), node.get("subject_id"), attributes.get("graph_entity_key"),
        attributes.get("memory_family"), node.get("category"), node.get("kind"),
    )
    for value in candidates:
        clean = _text(value)
        if clean:
            normalized = _NON_WORD.sub("-", clean.lower()).strip("-")
            if normalized:
                return normalized[:120]
    return "general-memory"


def _thread_title(key: str, representative: Mapping[str, Any]) -> str:
    label = _short(_text(representative.get("label") or representative.get("summary")), 64)
    readable_key = " ".join(part for part in re.split(r"[-_.:]+", key) if part)
    if readable_key and not _OPAQUE_KEY.match(key) and len(readable_key) <= 42:
        return _short(readable_key, 56)
    return label or "Memory thread"


def _node_score(node: Mapping[str, Any], kind: str, degree: Mapping[str, int]) -> float:
    return (
        (2.5 if _text(node.get("layer")) == "slow" else 0.6)
        + max(0.0, min(1.0, _number(node.get("salience")))) * 2.0
        + max(0.0, min(1.0, _number(node.get("confidence"))))
        + min(6, degree.get(_node_id(node), 0)) * 0.18
        + _KIND_PRIORITY.get(kind, 0.5)
    )


def _temporal_key(node: Mapping[str, Any]) -> tuple[str, int, str]:
    return (_text(node.get("occurred_at")) or "9999", _integer(node.get("turn_index")), _node_id(node))


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}.{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"
