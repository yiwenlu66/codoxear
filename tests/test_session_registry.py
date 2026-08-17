from __future__ import annotations

import threading
import time
from unittest.mock import patch

from codoxear import server
from codoxear.session_registry import SessionRegistry, session_registry_for_manager


def test_session_manager_private_registry_attrs_remain_compatibility_seams() -> None:
    manager = server.SessionManager.__new__(server.SessionManager)
    lock = threading.Lock()
    sessions = {"session-a": object()}
    stop_event = threading.Event()
    input_locks = {"session-a": threading.RLock()}
    store = object()

    manager._lock = lock
    manager._sessions = sessions
    manager._stop = stop_event
    manager._last_discover_ts = 123.5
    manager._input_locks = input_locks
    manager._store = store

    registry = session_registry_for_manager(manager)
    assert isinstance(registry, SessionRegistry)
    assert registry.lock is lock
    assert registry.sessions is sessions
    assert registry.stop_event is stop_event
    assert registry.last_discover_ts == 123.5
    assert registry.input_locks is input_locks
    assert registry.store is store
    assert manager._lock is lock
    assert manager._sessions is sessions
    assert manager._stop is stop_event
    assert manager._last_discover_ts == 123.5
    assert manager._input_locks is input_locks
    assert manager._store is store


def test_registry_lazy_initialization_builds_once_across_threads() -> None:
    manager = server.SessionManager.__new__(server.SessionManager)
    barrier = threading.Barrier(2)
    constructor_calls = 0
    constructor_lock = threading.Lock()
    real_init = SessionRegistry.__init__
    returned: list[SessionRegistry] = []

    def initialize_registry(self, *args, **kwargs):
        nonlocal constructor_calls
        with constructor_lock:
            constructor_calls += 1
        time.sleep(0.05)
        real_init(self, *args, **kwargs)

    def get_registry() -> None:
        barrier.wait()
        returned.append(session_registry_for_manager(manager))

    with patch.object(SessionRegistry, "__init__", initialize_registry):
        threads = [threading.Thread(target=get_registry) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)

    assert all(not thread.is_alive() for thread in threads)
    assert constructor_calls == 1
    assert len(returned) == 2
    assert returned[0] is returned[1]


def test_get_session_reads_registry_sessions_for_new_fixtures() -> None:
    manager = server.SessionManager.__new__(server.SessionManager)
    session = object()
    session_registry_for_manager(manager).sessions["session-a"] = session

    assert manager.get_session("session-a") is session
    assert manager.get_session("missing") is None
