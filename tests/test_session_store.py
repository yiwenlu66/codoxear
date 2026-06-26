import threading
from pathlib import Path
from unittest.mock import patch

from codoxear import server
from codoxear.server import SessionManager


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
