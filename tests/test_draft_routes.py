from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

from codoxear import server
from codoxear.draft_routes import DraftRouteDeps
from codoxear.draft_routes import handle_draft_get_route
from codoxear.draft_routes import handle_draft_post_route
from codoxear.draft_store import DraftStore
from codoxear.launch_ledger import launch_attempt_row
from codoxear.server import _match_session_route
from codoxear.session_cleanup import SessionCleanupCoordinator
from codoxear.session_listing import build_active_session_rows_snapshot
from codoxear.session_listing import build_launch_attempt_rows
from codoxear.session_listing import build_orphan_recovery_rows
from codoxear.session_model import Session
from codoxear.session_store import SessionStore
from codoxear.session_store import SessionStorePaths


class _FakeVoicePushCoordinator:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def _build_manager_patches() -> contextlib.ExitStack:
    stack = contextlib.ExitStack()
    stack.enter_context(patch.object(SessionStore, "load_persistent_state", lambda self: None))
    stack.enter_context(patch.object(server.SessionManager, "_backfill_recent_cwds_from_logs", lambda self: None))
    stack.enter_context(patch.object(server.SessionManager, "_discover_existing", lambda self, force=True: None))
    stack.enter_context(patch.object(server, "VoicePushCoordinator", _FakeVoicePushCoordinator))
    stack.enter_context(patch("threading.Thread.start", lambda self: None))
    return stack


class Handler:
    def __init__(self, body: dict[str, Any] | None = None) -> None:
        self.body = body or {}
        self.unauthorized = False

    def _unauthorized(self) -> None:
        self.unauthorized = True


class Manager:
    def __init__(self, drafts: dict[str, dict[str, Any]] | None = None, *, now_ts: float = 50.0) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.drafts = drafts if drafts is not None else {}
        self._store = DraftStore(None)
        self.now_ts = now_ts

    def draft_get(self, session_id: str) -> dict[str, Any]:
        self.calls.append(("get", session_id))
        entry = self.drafts.get(session_id)
        if not isinstance(entry, dict):
            return {"text": "", "updated_ts": 0.0}
        return {"text": str(entry.get("text") or ""), "updated_ts": float(entry.get("updated_ts") or 0.0)}

    def draft_set(self, session_id: str, text: str) -> float:
        self.calls.append(("set", session_id, text))
        if session_id not in self.drafts:
            raise KeyError("unknown session")
        # Mirrors SessionDraftCoordinator: real store write semantics with
        # the server wall clock, so blank text records a tombstone.
        return self._store.set(self.drafts, session_id, text, now_ts=self.now_ts)


class OversizeManager(Manager):
    def draft_set(self, session_id: str, text: str) -> float:
        self.calls.append(("set", session_id, text))
        raise ValueError("draft text exceeds the 262144 byte limit")


def _deps(responses: list[tuple[int, dict[str, Any]]], *, auth: bool = True) -> DraftRouteDeps:
    return DraftRouteDeps(
        require_auth=lambda _handler: auth,
        json_response=lambda _handler, status, obj: responses.append((status, obj)),
        read_json_body=lambda handler: handler.body,
    )


def _get(path: str, manager: Any, responses: list[tuple[int, dict[str, Any]]], *, auth: bool = True) -> bool:
    return handle_draft_get_route(
        Handler(),
        path=path,
        manager=manager,
        deps=_deps(responses, auth=auth),
        match_session_route=_match_session_route,
    )


def _post(path: str, manager: Any, body: dict[str, Any], responses: list[tuple[int, dict[str, Any]]], *, auth: bool = True) -> bool:
    return handle_draft_post_route(
        Handler(body),
        path=path,
        manager=manager,
        deps=_deps(responses, auth=auth),
        match_session_route=_match_session_route,
    )


def test_draft_get_returns_stored_text_and_timestamp() -> None:
    responses: list[tuple[int, dict[str, Any]]] = []
    manager = Manager({"s1": {"text": "hello draft", "updated_ts": 12.5}})

    assert _get("/api/sessions/s1/draft", manager, responses) is True

    assert responses == [(200, {"ok": True, "text": "hello draft", "updated_ts": 12.5})]
    assert manager.calls == [("get", "s1")]


def test_draft_get_returns_tombstone_projection() -> None:
    responses: list[tuple[int, dict[str, Any]]] = []
    manager = Manager({"s1": {"text": "", "updated_ts": 12.5}})

    assert _get("/api/sessions/s1/draft", manager, responses) is True

    assert responses == [(200, {"ok": True, "text": "", "updated_ts": 12.5})]


def test_draft_get_without_draft_returns_empty_projection() -> None:
    responses: list[tuple[int, dict[str, Any]]] = []
    manager = Manager()

    assert _get("/api/sessions/s1/draft", manager, responses) is True

    assert responses == [(200, {"ok": True, "text": "", "updated_ts": 0.0})]


def test_draft_get_unknown_session_maps_to_404() -> None:
    responses: list[tuple[int, dict[str, Any]]] = []

    class Missing(Manager):
        def draft_get(self, session_id: str) -> dict[str, Any]:
            self.calls.append(("get", session_id))
            raise KeyError("unknown session")

    assert _get("/api/sessions/ghost/draft", Missing(), responses) is True

    assert responses == [(404, {"error": "unknown session"})]


def test_draft_get_requires_auth() -> None:
    responses: list[tuple[int, dict[str, Any]]] = []
    manager = Manager({"s1": {"text": "secret", "updated_ts": 1.0}})
    handler = Handler()

    handled = handle_draft_get_route(
        handler,
        path="/api/sessions/s1/draft",
        manager=manager,
        deps=_deps(responses, auth=False),
        match_session_route=_match_session_route,
    )

    assert handled is True
    assert handler.unauthorized is True
    assert responses == []
    assert manager.calls == []


def test_draft_routes_ignore_other_paths() -> None:
    responses: list[tuple[int, dict[str, Any]]] = []
    manager = Manager()

    assert _get("/api/sessions/s1/queue", manager, responses) is False
    assert _post("/api/sessions/s1/enqueue", manager, {"text": "x"}, responses) is False
    assert _get("/api/sessions/s1/draft/extra", manager, responses) is False
    assert responses == []
    assert manager.calls == []


def test_draft_post_stores_text_and_returns_server_timestamp() -> None:
    responses: list[tuple[int, dict[str, Any]]] = []
    manager = Manager({"s1": {"text": "hello draft", "updated_ts": 1.0}}, now_ts=77.25)

    assert _post("/api/sessions/s1/draft", manager, {"text": "hello draft"}, responses) is True

    assert responses == [(200, {"ok": True, "updated_ts": 77.25})]
    assert manager.calls == [("set", "s1", "hello draft")]
    assert manager.drafts["s1"] == {"text": "hello draft", "updated_ts": 77.25}


def test_draft_post_empty_text_records_tombstone_and_reports_its_timestamp() -> None:
    responses: list[tuple[int, dict[str, Any]]] = []
    manager = Manager({"s1": {"text": "sent from this client", "updated_ts": 1.0}}, now_ts=88.5)

    assert _post("/api/sessions/s1/draft", manager, {"text": ""}, responses) is True

    assert responses == [(200, {"ok": True, "updated_ts": 88.5})]
    assert manager.calls == [("set", "s1", "")]
    assert manager.drafts["s1"] == {"text": "", "updated_ts": 88.5}

    # The tombstone stays visible through GET with its own timestamp.
    get_responses: list[tuple[int, dict[str, Any]]] = []
    assert _get("/api/sessions/s1/draft", manager, get_responses) is True
    assert get_responses == [(200, {"ok": True, "text": "", "updated_ts": 88.5})]


def test_draft_post_unknown_session_maps_to_404() -> None:
    responses: list[tuple[int, dict[str, Any]]] = []
    manager = Manager()

    assert _post("/api/sessions/ghost/draft", manager, {"text": "hi"}, responses) is True

    assert responses == [(404, {"error": "unknown session"})]


def test_draft_post_oversize_text_maps_to_413() -> None:
    responses: list[tuple[int, dict[str, Any]]] = []
    manager = OversizeManager()

    assert _post("/api/sessions/s1/draft", manager, {"text": "a" * (256 * 1024 + 1)}, responses) is True

    assert len(responses) == 1
    status, payload = responses[0]
    assert status == 413
    assert "byte limit" in str(payload.get("error"))


def test_draft_post_rejects_non_string_text() -> None:
    responses: list[tuple[int, dict[str, Any]]] = []
    manager = Manager()

    assert _post("/api/sessions/s1/draft", manager, {"text": 42}, responses) is True
    assert _post("/api/sessions/s1/draft", manager, {}, responses) is True

    assert responses == [
        (400, {"error": "text required"}),
        (400, {"error": "text required"}),
    ]
    assert manager.calls == []


def test_draft_post_rejects_lone_surrogate_text() -> None:
    responses: list[tuple[int, dict[str, Any]]] = []
    manager = Manager()

    assert _post("/api/sessions/s1/draft", manager, {"text": "lone \ud800 surrogate"}, responses) is True

    assert responses == [(400, {"error": "text must be valid UTF-8"})]
    assert manager.calls == []


def test_draft_post_requires_auth() -> None:
    responses: list[tuple[int, dict[str, Any]]] = []
    manager = Manager()
    handler = Handler({"text": "hello"})

    handled = handle_draft_post_route(
        handler,
        path="/api/sessions/s1/draft",
        manager=manager,
        deps=_deps(responses, auth=False),
        match_session_route=_match_session_route,
    )

    assert handled is True
    assert handler.unauthorized is True
    assert responses == []
    assert manager.calls == []


# ---------------------------------------------------------------------------
# Session-list projection and delete-cleanup behavior.
# ---------------------------------------------------------------------------


def _store(tmp_path: Path) -> SessionStore:
    return SessionStore(
        paths=SessionStorePaths(
            aliases=tmp_path / "aliases.json",
            sidebar_meta=tmp_path / "sidebar.json",
            hidden_sessions=tmp_path / "hidden.json",
            files=tmp_path / "files.json",
            queues=tmp_path / "queues.json",
            pending_attachments=tmp_path / "pending.json",
            commit_unknown_sends=tmp_path / "commit.json",
            recent_cwds=tmp_path / "recent.json",
            unattended=tmp_path / "unattended.json",
            drafts=tmp_path / "session_drafts.json",
        ),
        file_history_max=5,
        recent_cwd_max=5,
        unattended_default_idle_minutes=5,
        unattended_default_max_injections=10,
        clean_alias=lambda value: value if isinstance(value, str) else "",
        clean_priority_offset=lambda value: float(value or 0.0),
        clean_snooze_until=lambda value: float(value) if value not in (None, "", 0) else None,
        clean_dependency_session_id=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
        clean_recent_cwd=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
        clean_commit_unknown_send_record=lambda value: value if isinstance(value, dict) else None,
    )


def _session(sid: str, tmp_path: Path) -> Session:
    return Session(
        session_id=sid,
        thread_id=f"t-{sid}",
        broker_pid=2,
        codex_pid=1,
        agent_backend="codex",
        owned=True,
        start_ts=5.0,
        cwd=str(tmp_path),
        log_path=None,
        sock_path=tmp_path / f"{sid}.sock",
    )


def _rows(store: SessionStore, sessions: list[Session]) -> list[dict[str, Any]]:
    snapshot = build_active_session_rows_snapshot(
        sessions=sessions,
        queues={},
        unattended={},
        aliases={},
        store=store,
        now_ts=100.0,
        unattended_default_idle_minutes=5,
        unattended_default_max_injections=10,
        clean_unattended_cooldown_minutes=lambda value: 5,
        clean_unattended_remaining_injections=lambda value, **kwargs: 10,
        provider_choice_for_settings=lambda model_provider, preferred_auth_method: "openai-api",
        resolve_session_cwd=lambda cwd: Path(cwd),
        priority_half_life_seconds=8.0 * 3600.0,
        priority_bucket_seconds=30.0,
        subagent_runs={},
    )
    return snapshot.rows


def test_session_list_rows_expose_draft_updated_ts(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.drafts = {
        "s1": {"text": "unsent", "updated_ts": 42.5},
        "s2": {"text": "broken ts", "updated_ts": "not-a-number"},
        "s4": {"text": "", "updated_ts": 51.5},
        "other": {"text": "not listed", "updated_ts": 9.0},
    }
    rows = _rows(store, [_session("s1", tmp_path), _session("s2", tmp_path), _session("s3", tmp_path), _session("s4", tmp_path)])

    by_sid = {row["session_id"]: row for row in rows}
    assert by_sid["s1"]["draft_updated_ts"] == 42.5
    assert by_sid["s2"]["draft_updated_ts"] == 0.0
    # A session that never had a draft still reports 0.0.
    assert by_sid["s3"]["draft_updated_ts"] == 0.0
    # A tombstone reports its own timestamp so polling clients can see the
    # deletion through the same last-writer-wins channel as edits.
    assert by_sid["s4"]["draft_updated_ts"] == 51.5


def test_orphan_recovery_and_launch_attempt_rows_carry_zero_draft_timestamp() -> None:
    orphan_rows = build_orphan_recovery_rows(
        active_session_ids={"live"},
        commit_unknown_sends={"dead": {"text": "maybe", "created_ts": 1.0}},
        queues={"dead": [{"id": "q1", "text": "queued", "created_ts": 1.0}]},
        existing_session_ids=set(),
        now_ts=10.0,
        unattended_default_idle_minutes=5,
        unattended_default_max_injections=10,
    )
    launch_rows = build_launch_attempt_rows(
        records=[{"launch_id": "L1", "state": "failed", "cwd": "/tmp", "created_ts": 1.0}],
        hidden_failure_ids=set(),
        active_launch_ids=set(),
        active_spawn_nonces=set(),
        row_from_record=lambda record: launch_attempt_row(
            record,
            default_agent_backend="codex",
            unattended_default_idle_minutes=5,
            unattended_default_max_injections=10,
        ),
    )

    assert orphan_rows and orphan_rows[0]["draft_updated_ts"] == 0.0
    assert launch_rows and launch_rows[0]["draft_updated_ts"] == 0.0


def _cleanup_coordinator(store: SessionStore, sessions: dict[str, Session], saved: list[str]) -> SessionCleanupCoordinator:
    return SessionCleanupCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        store=lambda: store,
        input_locks=lambda: {},
        unlink_quiet=lambda path: None,
        save_pending_attachments=lambda: None,
        save_commit_unknown_sends=lambda: None,
        save_aliases=lambda: None,
        save_sidebar_meta=lambda: None,
        save_hidden_sessions=lambda: None,
        save_unattended=lambda: None,
        save_files=lambda: None,
        save_queues=lambda: None,
        save_drafts=lambda: saved.append("drafts") or store.save_drafts(store.drafts),
    )


def test_delete_session_chain_removes_only_the_deleted_sessions_draft(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.load_persistent_state()
    sessions = {"s1": _session("s1", tmp_path), "s2": _session("s2", tmp_path)}
    store.drafts = {
        "s1": {"text": "delete me", "updated_ts": 1.0},
        "s2": {"text": "keep me", "updated_ts": 2.0},
    }
    saved: list[str] = []
    coordinator = _cleanup_coordinator(store, sessions, saved)

    coordinator.clear_deleted_session_state("s1")

    assert "s1" not in store.drafts
    assert store.drafts["s2"] == {"text": "keep me", "updated_ts": 2.0}
    assert saved == ["drafts"]
    on_disk = (tmp_path / "session_drafts.json").read_text(encoding="utf-8")
    assert "delete me" not in on_disk
    assert "keep me" in on_disk

    # Deleting again with no draft present must not re-save.
    coordinator.clear_deleted_session_state("s1")
    assert saved == ["drafts"]


def test_delete_session_chain_removes_tombstone_entry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.load_persistent_state()
    sessions = {"s1": _session("s1", tmp_path), "s2": _session("s2", tmp_path)}
    store.drafts = {
        "s1": {"text": "", "updated_ts": 9.0},
        "s2": {"text": "keep me", "updated_ts": 2.0},
    }
    saved: list[str] = []
    coordinator = _cleanup_coordinator(store, sessions, saved)

    coordinator.clear_deleted_session_state("s1")

    # Entry removal happens only here: a tombstone outlives clears but not
    # its session.
    assert store.drafts == {"s2": {"text": "keep me", "updated_ts": 2.0}}
    assert saved == ["drafts"]
    assert json.loads((tmp_path / "session_drafts.json").read_text(encoding="utf-8")) == {
        "s2": {"text": "keep me", "updated_ts": 2.0}
    }


def test_manager_draft_methods_round_trip_through_disk(tmp_path: Path, monkeypatch) -> None:
    # DRAFT_PATH must stay patched for the whole test: the store rebinds when
    # the paths it was built with no longer match, which would redirect draft
    # writes back to the real app dir.
    draft_path = tmp_path / "session_drafts.json"
    monkeypatch.setattr(server, "DRAFT_PATH", draft_path)
    import pytest

    with _build_manager_patches():
        manager = server.SessionManager()
        sid = "s1"
        manager._sessions[sid] = _session(sid, tmp_path)
        before = __import__("time").time()

        assert manager.draft_get(sid) == {"text": "", "updated_ts": 0.0}

        updated_ts = manager.draft_set(sid, "composing")
        assert updated_ts >= before
        assert manager.draft_get(sid) == {"text": "composing", "updated_ts": updated_ts}
        saved = json.loads(draft_path.read_text(encoding="utf-8"))
        assert saved == {sid: {"text": "composing", "updated_ts": updated_ts}}

        # Clearing after a send is a timestamped tombstone (never 0): other
        # clients' companion timestamps must see the deletion as newer.
        clear_before = __import__("time").time()
        clear_ts = manager.draft_set(sid, "")
        assert clear_ts >= clear_before
        assert clear_ts >= updated_ts
        assert manager.draft_get(sid) == {"text": "", "updated_ts": clear_ts}
        assert json.loads(draft_path.read_text(encoding="utf-8")) == {sid: {"text": "", "updated_ts": clear_ts}}

        with pytest.raises(KeyError):
            manager.draft_get("ghost")
        with pytest.raises(KeyError):
            manager.draft_set("ghost", "text")
        with pytest.raises(ValueError):
            manager.draft_set(sid, "a" * (256 * 1024 + 1))
        # The rejected oversize write leaves the tombstone untouched.
        assert manager._drafts[sid] == {"text": "", "updated_ts": clear_ts}


def test_clear_deleted_session_state_flags_draft_changes_for_save(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.drafts = {"s1": {"text": "x", "updated_ts": 1.0}}

    changes = store.clear_deleted_session_state("s1")

    assert changes.drafts is True
    assert store.drafts == {}

    called: list[str] = []
    store.save_deleted_session_state_changes(
        changes,
        save_aliases=lambda: called.append("aliases"),
        save_sidebar_meta=lambda: called.append("sidebar"),
        save_hidden_sessions=lambda: called.append("hidden"),
        save_unattended=lambda: called.append("unattended"),
        save_files=lambda: called.append("files"),
        save_queues=lambda: called.append("queues"),
        save_pending_attachments=lambda: called.append("pending"),
        save_commit_unknown_sends=lambda: called.append("unknown"),
        save_drafts=lambda: called.append("drafts"),
    )
    assert called == ["drafts"]

    # No draft -> no draft change flag, no save.
    empty_changes = store.clear_deleted_session_state("s2")
    assert empty_changes.drafts is False
