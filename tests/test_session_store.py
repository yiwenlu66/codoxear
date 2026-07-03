import json
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear import server
from codoxear.server import SessionManager
from codoxear.session_manager_store import create_session_store
from codoxear.session_manager_store import session_store_paths
from codoxear.session_recent_cwd import SessionRecentCwdCoordinator


def test_session_manager_persistent_maps_are_store_owned() -> None:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()

    aliases = {"s1": "alpha"}
    mgr._aliases = aliases
    assert mgr._session_store_for_manager().aliases is aliases
    assert mgr._aliases is aliases

    queues = {"s1": [{"id": "q1", "text": "queued"}]}
    mgr._queues = queues
    assert mgr._session_store_for_manager().queues is queues
    assert mgr._queues is queues

    pending = {"s1"}
    mgr._pending_attachment_ids = pending
    assert mgr._session_store_for_manager().pending_attachment_ids is pending
    assert mgr._pending_attachment_ids is pending


def test_session_store_file_history_migrates_legacy_session_key() -> None:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    store = mgr._session_store_for_manager()
    store.files = {"s1": ["a.py", "b.py"]}

    out, dirty = store.file_history_for_keys("sid:s1", ["s1"])

    assert out == ["a.py", "b.py"]
    assert dirty is True
    assert store.files == {"sid:s1": ["a.py", "b.py"]}


def test_session_store_file_history_adds_from_legacy_and_caps() -> None:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    store = mgr._session_store_for_manager()
    store.file_history_max = 3
    store.files = {"s1": ["a.py", "b.py", "c.py"]}

    out = store.add_file_history_entry("sid:s1", ["s1"], "b.py")

    assert out == ["b.py", "a.py", "c.py"]
    assert store.files == {"sid:s1": ["b.py", "a.py", "c.py"]}


def test_session_store_file_history_clear_removes_session_and_cwd_legacy_keys() -> None:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    store = mgr._session_store_for_manager()
    store.files = {"sid:s1": ["new.py"], "s1": ["old.py"], "cwd:/repo": ["leak.py"], "sid:other": ["keep.py"]}

    dirty = store.clear_file_history_for_keys("sid:s1", ["s1"], cwd="/repo")

    assert dirty is True
    assert store.files == {"sid:other": ["keep.py"]}


def test_session_store_clear_deleted_session_state_cleans_persistent_maps_and_cwd_history() -> None:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    store = mgr._session_store_for_manager()
    store.aliases = {"s1": "alpha", "keep": "beta"}
    store.hidden_sessions = {"s1", "keep"}
    store.sidebar_meta = {
        "s1": {"priority_offset": 0.5},
        "dependent": {"priority_offset": 0.0, "dependency_session_id": "s1"},
        "keep": {"dependency_session_id": "other"},
    }
    store.unattended = {"s1": {"enabled": True}, "keep": {"enabled": True}}
    store.files = {"sid:s1": ["new.py"], "s1": ["old.py"], "cwd:/repo": ["leak.py"], "sid:keep": ["keep.py"]}
    store.pending_attachment_ids = {"s1", "keep"}
    store.commit_unknown_sends = {"s1": {"text": "maybe", "created_ts": 1.0}, "keep": {"text": "keep", "created_ts": 2.0}}
    store.queues = {"s1": [{"id": "q1", "text": "later"}], "keep": [{"id": "q2", "text": "keep"}]}

    changes = store.clear_deleted_session_state("s1", clear_recovery=True, cwd="/repo")

    assert changes.aliases is True
    assert changes.hidden_sessions is True
    assert changes.sidebar_meta is True
    assert changes.unattended is True
    assert changes.files is True
    assert changes.pending_attachments is True
    assert changes.commit_unknown_sends is True
    assert changes.queues is True
    assert store.aliases == {"keep": "beta"}
    assert store.hidden_sessions == {"keep"}
    assert store.sidebar_meta == {"dependent": {"priority_offset": 0.0}, "keep": {"dependency_session_id": "other"}}
    assert store.unattended == {"keep": {"enabled": True}}
    assert store.files == {"sid:keep": ["keep.py"]}
    assert store.pending_attachment_ids == {"keep"}
    assert store.commit_unknown_sends == {"keep": {"text": "keep", "created_ts": 2.0}}
    assert store.queues == {"keep": [{"id": "q2", "text": "keep"}]}


def test_session_store_clear_deleted_session_state_preserves_recovery_queue_until_explicit() -> None:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    store = mgr._session_store_for_manager()
    store.commit_unknown_sends = {"s1": {"text": "maybe direct", "created_ts": 1.0}}
    store.queues = {"s1": [{"id": "q1", "text": "later"}]}

    changes = store.clear_deleted_session_state("s1")

    assert changes.queues is True
    assert changes.commit_unknown_sends is False
    assert store.commit_unknown_sends == {"s1": {"text": "maybe direct", "created_ts": 1.0}}
    assert store.queues == {"s1": [{"id": "q1", "text": "later", "orphan_recovery": True}]}

    changes = store.clear_deleted_session_state("s1", clear_recovery=True)

    assert changes.queues is True
    assert changes.commit_unknown_sends is True
    assert store.commit_unknown_sends == {}
    assert store.queues == {}


def test_session_store_prunes_stale_direct_unknowns_and_marks_queues_for_recovery() -> None:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    store = mgr._session_store_for_manager()
    store.commit_unknown_sends = {
        "active": {"text": "keep active", "created_ts": 1.0},
        "fresh": {"text": "keep fresh", "created_ts": 95.0},
        "stale": {"text": "drop stale", "created_ts": 1.0},
        "bad": {"text": "drop bad", "created_ts": 0.0},
    }
    store.queues = {"stale": [{"id": "q1", "text": "later"}], "bad": [{"id": "q2", "text": "later"}]}

    changes = store.prune_missing_commit_unknown_sends(active_session_ids={"active"}, now_ts=100.0, max_age_seconds=10.0)

    assert changes.commit_unknown_sends is True
    assert changes.queues is True
    assert store.commit_unknown_sends == {
        "active": {"text": "keep active", "created_ts": 1.0},
        "fresh": {"text": "keep fresh", "created_ts": 95.0},
    }
    assert store.queues == {
        "stale": [{"id": "q1", "text": "later", "orphan_recovery": True}],
        "bad": [{"id": "q2", "text": "later", "orphan_recovery": True}],
    }


def test_session_store_deleted_state_save_order_uses_changed_maps_only() -> None:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    store = mgr._session_store_for_manager()
    calls: list[str] = []
    changes = store.clear_deleted_session_state("missing")

    store.save_deleted_session_state_changes(
        changes,
        save_pending_attachments=lambda: calls.append("pending"),
        save_commit_unknown_sends=lambda: calls.append("unknown"),
        save_aliases=lambda: calls.append("aliases"),
        save_sidebar_meta=lambda: calls.append("sidebar"),
        save_hidden_sessions=lambda: calls.append("hidden"),
        save_unattended=lambda: calls.append("unattended"),
        save_files=lambda: calls.append("files"),
        save_queues=lambda: calls.append("queues"),
    )
    assert calls == []

    store.aliases = {"s1": "alpha"}
    store.hidden_sessions = {"s1"}
    store.pending_attachment_ids = {"s1"}
    changes = store.clear_deleted_session_state("s1")
    store.save_deleted_session_state_changes(
        changes,
        save_pending_attachments=lambda: calls.append("pending"),
        save_commit_unknown_sends=lambda: calls.append("unknown"),
        save_aliases=lambda: calls.append("aliases"),
        save_sidebar_meta=lambda: calls.append("sidebar"),
        save_hidden_sessions=lambda: calls.append("hidden"),
        save_unattended=lambda: calls.append("unattended"),
        save_files=lambda: calls.append("files"),
        save_queues=lambda: calls.append("queues"),
    )
    assert calls == ["pending", "aliases", "hidden"]


def test_session_store_sidebar_state_repairs_invalid_dependency_and_expired_snooze() -> None:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    store = mgr._session_store_for_manager()
    store.sidebar_meta = {"s1": {"priority_offset": 0.25, "dependency_session_id": "missing", "snooze_until": 5.0}}

    state = store.sidebar_state_for_session("s1", active_session_ids={"s1"}, now_ts=10.0)

    assert state.priority_offset == 0.25
    assert state.dependency_session_id is None
    assert state.snooze_until is None
    assert state.dirty is True
    assert store.sidebar_meta == {"s1": {"priority_offset": 0.25}}


def test_session_store_recent_cwd_records_newer_timestamps_only() -> None:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    store = mgr._session_store_for_manager()
    store.recent_cwds = {"/repo": 10.0}

    assert store.note_recent_cwd("/repo", 9.0) is False
    assert store.recent_cwds == {"/repo": 10.0}
    assert store.note_recent_cwd("/repo", 11.0) is True
    assert store.recent_cwds == {"/repo": 11.0}
    assert store.note_recent_cwd("", 12.0) is False


def test_session_store_recent_cwd_remember_normalizes_trims_and_lists() -> None:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    store = mgr._session_store_for_manager()
    store.recent_cwd_max = 2
    now_calls = iter([100.0, 101.0])

    assert store.remember_recent_cwd("/repo", ts="bad", now=lambda: next(now_calls)) is True
    assert store.remember_recent_cwd("/repo", ts=99.0, now=lambda: 102.0) is False
    assert store.remember_recent_cwd("/other", ts=True, now=lambda: next(now_calls)) is True
    assert store.remember_recent_cwd("/third", ts=103.0, now=lambda: 104.0) is True
    assert store.remember_recent_cwd("/fourth", ts=104.0, now=lambda: 105.0) is True
    assert store.remember_recent_cwd("/fifth", ts=105.0, now=lambda: 106.0) is True

    assert store.list_recent_cwds(limit=3) == ["/fifth", "/fourth"]
    assert store.recent_cwds == {"/fourth": 104.0, "/fifth": 105.0}


def test_session_store_load_persistent_state_owns_bootstrap_order() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        paths = session_store_paths(
            aliases=root / "session_aliases.json",
            sidebar_meta=root / "session_sidebar.json",
            hidden_sessions=root / "hidden_sessions.json",
            files=root / "session_files.json",
            queues=root / "session_queues.json",
            pending_attachments=root / "pending_attachments.json",
            commit_unknown_sends=root / "commit_unknown_sends.json",
            recent_cwds=root / "recent_cwds.json",
            unattended=root / "unattended.json",
        )
        paths.aliases.write_text(json.dumps({"s1": " alpha "}), encoding="utf-8")
        paths.sidebar_meta.write_text(json.dumps({"s1": {"priority_offset": 0.5, "dependency_session_id": "dep"}}), encoding="utf-8")
        paths.hidden_sessions.write_text(json.dumps(["s1", " ", 5]), encoding="utf-8")
        paths.files.write_text(json.dumps({"s1": ["a.py", "a.py", "b.py"], "cwd:/repo": ["leak.py"]}), encoding="utf-8")
        paths.queues.write_text(json.dumps({"s1": [{"id": "q1", "text": "queued", "created_ts": 1.0}]}), encoding="utf-8")
        paths.pending_attachments.write_text(json.dumps(["s1", "", 4]), encoding="utf-8")
        paths.commit_unknown_sends.write_text(json.dumps({"s1": {"text": "maybe", "created_ts": 1.0}}), encoding="utf-8")
        paths.recent_cwds.write_text(json.dumps({"/repo": 2.0}), encoding="utf-8")
        paths.unattended.write_text(json.dumps({"s1": {"enabled": True, "request": "continue"}}), encoding="utf-8")
        store = create_session_store(
            paths=paths,
            file_history_max=3,
            recent_cwd_max=5,
            unattended_default_idle_minutes=15,
            unattended_default_max_injections=7,
            clean_alias=server._clean_alias,
            clean_priority_offset=server._clean_priority_offset,
            clean_snooze_until=server._clean_snooze_until,
            clean_dependency_session_id=server._clean_dependency_session_id,
            clean_recent_cwd=server._clean_recent_cwd,
            clean_commit_unknown_send_record=lambda raw: dict(raw) if isinstance(raw, dict) and isinstance(raw.get("text"), str) else None,
        )
        store.aliases = {"junk": "value"}
        store.pending_attachment_ids = {"junk"}

        store.reset_in_memory_state()
        assert store.aliases == {}
        assert store.pending_attachment_ids == set()

        store.load_persistent_state()

        assert store.aliases == {"s1": "alpha"}
        assert store.sidebar_meta == {"s1": {"priority_offset": 0.5, "dependency_session_id": "dep"}}
        assert store.hidden_sessions == {"s1"}
        assert store.files == {"sid:s1": ["a.py", "b.py"]}
        assert store.queues == {"s1": [{"id": "q1", "text": "queued", "created_ts": 1.0}]}
        assert store.pending_attachment_ids == {"s1"}
        assert store.commit_unknown_sends == {"s1": {"text": "maybe", "created_ts": 1.0}}
        assert store.recent_cwds == {"/repo": 2.0}
        assert store.unattended == {"s1": {"enabled": True, "request": "continue", "cooldown_minutes": 15, "remaining_injections": 7}}


def test_recent_cwd_coordinator_lists_store_default_limit_when_route_omits_limit() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        store = create_session_store(
            paths=session_store_paths(
                aliases=root / "session_aliases.json",
                sidebar_meta=root / "session_sidebar.json",
                hidden_sessions=root / "hidden_sessions.json",
                files=root / "session_files.json",
                queues=root / "session_queues.json",
                pending_attachments=root / "pending_attachments.json",
                commit_unknown_sends=root / "commit_unknown_sends.json",
                recent_cwds=root / "recent_cwds.json",
                unattended=root / "unattended.json",
            ),
            file_history_max=3,
            recent_cwd_max=2,
            unattended_default_idle_minutes=15,
            unattended_default_max_injections=7,
            clean_alias=server._clean_alias,
            clean_priority_offset=server._clean_priority_offset,
            clean_snooze_until=server._clean_snooze_until,
            clean_dependency_session_id=server._clean_dependency_session_id,
            clean_recent_cwd=server._clean_recent_cwd,
            clean_commit_unknown_send_record=lambda raw: dict(raw) if isinstance(raw, dict) and isinstance(raw.get("text"), str) else None,
        )
        store.recent_cwds = {"/old": 1.0, "/new": 3.0, "/mid": 2.0}
        coordinator = SessionRecentCwdCoordinator(
            lock=threading.Lock(),
            store=lambda: store,
            iter_session_logs=lambda: [],
            resume_candidate_from_log=lambda _path: None,
            save_recent_cwds=lambda: None,
            now=lambda: 4.0,
        )

        assert coordinator.list_recent() == ["/new", "/mid"]
        assert coordinator.list_recent(limit=1) == ["/new"]


def test_session_store_rebinds_paths_without_losing_in_memory_state() -> None:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    mgr._aliases = {"s1": "alpha"}
    original_store = mgr._session_store_for_manager()
    with patch.object(server, "ALIAS_PATH", Path("/tmp/codoxear-test-aliases.json")):
        rebound_store = mgr._session_store_for_manager()
    assert rebound_store is not original_store
    assert rebound_store.paths.aliases == Path("/tmp/codoxear-test-aliases.json")
    assert rebound_store.aliases == {"s1": "alpha"}
    assert mgr._aliases == {"s1": "alpha"}
