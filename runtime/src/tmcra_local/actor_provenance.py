from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any


ACTOR_PROVENANCE_SCHEMA_VERSION = "tmcra.service.actor-provenance.1"
ACTOR_ROLES = frozenset({"user", "assistant", "system", "tool"})
AGENT_FIELDS = (
    "agent_id",
    "agent_name",
    "agent_role",
    "agent_specialty",
    "agent_team",
)
ACTOR_FIELDS = ("actor_role", *AGENT_FIELDS)
ROUTING_FIELDS = ("target_agent_id",)
AGENT_FIELD_LIMITS = {
    "agent_id": 200,
    "agent_name": 200,
    "agent_role": 120,
    "agent_specialty": 200,
    "agent_team": 200,
}
AGENT_ALIASES = {
    "agent_id": ("agent_id", "id"),
    "agent_name": ("agent_name", "name"),
    "agent_role": ("agent_role", "team_role", "role"),
    "agent_specialty": ("agent_specialty", "specialty"),
    "agent_team": ("agent_team", "team"),
}
AGENT_PLURAL_FIELDS = {
    "agent_id": "agent_ids",
    "agent_name": "agent_names",
    "agent_role": "agent_roles",
    "agent_specialty": "agent_specialties",
    "agent_team": "agent_teams",
}


class ActorProvenanceError(RuntimeError):
    pass


def _text(value: Any, *, field: str, limit: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ActorProvenanceError(f"message metadata {field} must be a string")
    normalized = value.strip()
    if len(normalized) > limit:
        raise ActorProvenanceError(
            f"message metadata {field} must be at most {limit} characters"
        )
    return normalized


def _alias_value(
    metadata: Mapping[str, Any],
    nested: Mapping[str, Any],
    canonical: str,
) -> str:
    values: list[str] = []
    for alias in AGENT_ALIASES[canonical]:
        sources = (
            (nested,)
            if alias in {"id", "name", "role"}
            else (metadata, nested)
        )
        for source in sources:
            if alias not in source or source.get(alias) is None:
                continue
            value = _text(
                source.get(alias),
                field=canonical,
                limit=AGENT_FIELD_LIMITS[canonical],
            )
            if value and value not in values:
                values.append(value)
    if len(values) > 1:
        raise ActorProvenanceError(
            f"message metadata has conflicting aliases for {canonical}"
        )
    return values[0] if values else ""


def normalize_message_actor_metadata(
    role: Any,
    metadata: Any,
) -> dict[str, str]:
    """Return the bounded producer identity persisted for one message.

    Arbitrary integration metadata may still travel with an API request, but it
    is never copied into the memory graph.  Only this allowlisted, size-bounded
    identity becomes immutable provenance.  The message role is authoritative:
    a caller cannot label assistant output as a user statement (or vice versa).
    """

    actor_role = str(role or "").strip().lower()
    if actor_role not in ACTOR_ROLES:
        raise ActorProvenanceError("message role is invalid")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ActorProvenanceError("message metadata must be an object")
    declared_role = metadata.get("actor_role")
    if declared_role is not None:
        normalized_declared = _text(
            declared_role, field="actor_role", limit=16
        ).lower()
        if normalized_declared != actor_role:
            raise ActorProvenanceError(
                "message metadata actor_role differs from message role"
            )
    nested = metadata.get("agent")
    if nested is None:
        nested = {}
    if not isinstance(nested, Mapping):
        raise ActorProvenanceError("message metadata agent must be an object")
    result = {
        "actor_provenance_schema": ACTOR_PROVENANCE_SCHEMA_VERSION,
        "actor_role": actor_role,
    }
    for field in AGENT_FIELDS:
        value = _alias_value(metadata, nested, field)
        if value:
            result[field] = value
    target_agent_id = _text(
        metadata.get("target_agent_id"),
        field="target_agent_id",
        limit=200,
    )
    if target_agent_id:
        result["target_agent_id"] = target_agent_id
    if actor_role == "user" and any(field in result for field in AGENT_FIELDS):
        raise ActorProvenanceError(
            "user messages cannot declare an assistant agent producer"
        )
    return result


def actor_metadata_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            field: value[field]
            for field in (*ACTOR_FIELDS, *ROUTING_FIELDS)
            if value.get(field)
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def actor_metadata_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(actor_metadata_json(value).encode("utf-8")).hexdigest()


def graph_actor_metadata(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        field: str(value[field])
        for field in ("actor_provenance_schema", *ACTOR_FIELDS, *ROUTING_FIELDS)
        if value.get(field)
    }


def _source_ids(value: Mapping[str, Any]) -> list[str]:
    result: list[str] = []

    def add(raw: Any) -> None:
        if isinstance(raw, Sequence) and not isinstance(
            raw, (str, bytes, bytearray)
        ):
            for item in raw:
                add(item)
            return
        text = str(raw or "").strip()
        if text and text not in result:
            result.append(text)

    add(value.get("source_record_id"))
    add(value.get("source_record_ids"))
    return result


def _actor_identity(metadata: Mapping[str, Any]) -> dict[str, str]:
    role = str(
        metadata.get("actor_role")
        or metadata.get("message_role")
        or metadata.get("speaker")
        or metadata.get("role")
        or ""
    ).strip().lower()
    if role not in ACTOR_ROLES:
        return {}
    result = {"actor_role": role}
    for field in AGENT_FIELDS:
        value = metadata.get(field)
        if isinstance(value, str) and value.strip():
            result[field] = value.strip()
    target = metadata.get("target_agent_id")
    if isinstance(target, str) and target.strip():
        result["target_agent_id"] = target.strip()
    return result


def load_source_actor_index(
    database: Path | str,
    scope_id: str,
) -> dict[str, dict[str, str]]:
    path = Path(database).resolve()
    if not path.is_file():
        # Actor labels are additive answer metadata.  A legacy/mock snapshot
        # without a materialized graph must keep its pre-provenance recall
        # behavior instead of turning a successful recall into a conflict.
        return {}
    output: dict[str, dict[str, str]] = {}
    try:
        with closing(sqlite3.connect(path, timeout=30.0)) as connection:
            rows = connection.execute(
                "SELECT memory_id,metadata_json FROM records WHERE scope_id=?",
                (scope_id,),
            ).fetchall()
    except sqlite3.Error as exc:
        raise ActorProvenanceError("actor provenance database is unreadable") from exc
    for memory_id, raw_metadata in rows:
        try:
            metadata = json.loads(str(raw_metadata or "{}"))
        except json.JSONDecodeError as exc:
            raise ActorProvenanceError("actor provenance metadata is invalid JSON") from exc
        if not isinstance(metadata, Mapping):
            raise ActorProvenanceError("actor provenance metadata is not an object")
        if str(metadata.get("content_variant") or "").strip() != "source_message":
            continue
        identity = _actor_identity(metadata)
        if identity:
            output[str(memory_id)] = identity
    return output


def _merge_identities(identities: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    roles: list[str] = []
    for identity in identities:
        role = str(identity.get("actor_role") or "").strip().lower()
        if role in ACTOR_ROLES and role not in roles:
            roles.append(role)
    result: dict[str, Any] = {}
    if len(roles) == 1:
        result["actor_role"] = roles[0]
    elif roles:
        result["actor_roles"] = roles
    for field in AGENT_FIELDS:
        values: list[str] = []
        for identity in identities:
            value = str(identity.get(field) or "").strip()
            if value and value not in values:
                values.append(value)
        if len(values) == 1:
            result[field] = values[0]
        elif values:
            result[AGENT_PLURAL_FIELDS[field]] = values
    targets: list[str] = []
    for identity in identities:
        value = str(identity.get("target_agent_id") or "").strip()
        if value and value not in targets:
            targets.append(value)
    if len(targets) == 1:
        result["target_agent_id"] = targets[0]
    elif targets:
        result["target_agent_ids"] = targets
    return result


def enrich_evidence_actor_provenance(
    evidence: Mapping[str, Any],
    *,
    database: Path | str,
    scope_id: str,
) -> dict[str, Any]:
    """Attach producer labels to answer-facing evidence without changing rank.

    The walk copies the evidence tree and resolves immutable source IDs back to
    source records.  It does not add, remove, reorder, score, or select evidence.
    """

    index = load_source_actor_index(database, scope_id)

    def visit(value: Any) -> Any:
        if isinstance(value, Mapping):
            row = {str(key): visit(item) for key, item in value.items()}
            identities = [index[item] for item in _source_ids(row) if item in index]
            if not identities:
                for field in ("source_parent", "source_parents"):
                    nested = row.get(field)
                    candidates = (
                        nested
                        if isinstance(nested, Sequence)
                        and not isinstance(nested, (str, bytes, bytearray))
                        else [nested]
                    )
                    for candidate in candidates:
                        if isinstance(candidate, Mapping):
                            identity = _actor_identity(candidate)
                            if identity:
                                identities.append(identity)
            resolved = _merge_identities(identities)
            for field, item in resolved.items():
                existing = row.get(field)
                if existing in (None, "", []):
                    row[field] = item
            return row
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [visit(item) for item in value]
        return copy.deepcopy(value)

    return visit(evidence)
