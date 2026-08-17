from __future__ import annotations

from dataclasses import fields
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from codoxear import server
from codoxear.session_manager_factories import QueueFactoryDeps
from codoxear.session_manager_factories import SessionManagerCoordinatorDeps
from codoxear.session_manager_factories import session_manager_coordinator_deps
from codoxear.session_store import SessionStore


class _FakeVoicePushCoordinator:
    def __init__(self, *args, **kwargs) -> None:
        pass


def _build_manager() -> server.SessionManager:
    with (
        patch.object(SessionStore, "load_persistent_state", lambda self: None),
        patch.object(server.SessionManager, "_backfill_recent_cwds_from_logs", lambda self: None),
        patch.object(server.SessionManager, "_discover_existing", lambda self, force=True: None),
        patch.object(server, "VoicePushCoordinator", _FakeVoicePushCoordinator),
        patch("threading.Thread.start", lambda self: None),
    ):
        return server.SessionManager()


def test_manager_dependencies_are_focused_per_coordinator() -> None:
    deps = session_manager_coordinator_deps(server)

    assert isinstance(deps, SessionManagerCoordinatorDeps)
    assert isinstance(deps.queue_coordinator, QueueFactoryDeps)
    assert {field.name for field in fields(deps.queue_coordinator)} == {
        "commit_unknown_error",
        "injection_error",
        "not_ready_error",
        "now",
        "queue_idle_grace_seconds",
    }
    assert not hasattr(deps.queue_coordinator, "tmux_session_name")
    assert deps.web_launch_coordinator.homes == {
        "codex": server.CODEX_HOME,
        "pi": server.PI_HOME,
        "cc": server.CC_HOME,
    }


def test_manager_retains_one_coordinator_graph() -> None:
    manager = _build_manager()

    assert manager._queue_coordinator_for_manager() is manager._queue_coordinator_for_manager()
    assert manager._log_runtime_for_manager() is manager._log_runtime_for_manager()
    assert manager._queue_coordinator_for_manager() is manager._coordinators.queue


def test_compatibility_lazy_graph_initialization_builds_once_across_threads() -> None:
    manager = server.SessionManager.__new__(server.SessionManager)
    barrier = threading.Barrier(2)
    builder_calls = 0
    builder_lock = threading.Lock()
    returned: list[object] = []

    def build_graph(_manager, _deps):
        nonlocal builder_calls
        with builder_lock:
            builder_calls += 1
        time.sleep(0.05)
        return SimpleNamespace(queue=object())

    def get_queue() -> None:
        barrier.wait()
        returned.append(manager._queue_coordinator_for_manager())

    with (
        patch.object(server, "_session_manager_coordinator_deps", lambda _module: object()),
        patch.object(server, "_build_session_manager_coordinator_graph", build_graph),
    ):
        threads = [threading.Thread(target=get_queue) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)

    assert all(not thread.is_alive() for thread in threads)
    assert builder_calls == 1
    assert len(returned) == 2
    assert returned[0] is returned[1]


def test_queue_sweep_cursor_lives_on_retained_coordinator() -> None:
    manager = _build_manager()
    coordinator = manager._queue_sweep_coordinator_for_manager()

    coordinator.cursor = 7

    assert manager._queue_sweep_coordinator_for_manager() is coordinator
    assert manager._queue_sweep_coordinator_for_manager().cursor == 7
    assert not hasattr(manager, "_queue_sweep_cursor")
