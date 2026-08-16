from __future__ import annotations

from pathlib import Path

import pytest

from tmcra_local.storage import LocalStore


def _register(store: LocalStore, *, project: str, session: str, native_id: str):
    return store.register_message(
        project_id=project,
        project_title=project,
        session_id=session,
        session_title=session,
        role="user",
        content="stable content",
        occurred_at="2026-08-16T00:00:00Z",
        source_app="codex",
        native_thread_id=session,
        native_message_id=native_id,
        actor={"actor_role": "user"},
    )


def test_message_identity_is_idempotent_and_project_scoped(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "local.sqlite3")
    first, created = _register(
        store, project="project-a", session="session-a", native_id="m1"
    )
    repeated, created_again = _register(
        store, project="project-a", session="session-a", native_id="m1"
    )
    assert created is True
    assert created_again is False
    assert first["message_id"] == repeated["message_id"]
    with pytest.raises(ValueError, match="different project"):
        _register(
            store, project="project-b", session="session-a", native_id="m2"
        )


def test_secure_delete_removes_message_and_rewrites_database(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "local.sqlite3")
    message, _ = _register(
        store, project="project-a", session="session-a", native_id="m1"
    )
    assert store.erase_message_metadata(message["message_id"])
    store.secure_compact()
    assert store.message(message["message_id"]) is None
