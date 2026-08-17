from __future__ import annotations

from dataclasses import fields
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


def test_queue_sweep_cursor_lives_on_retained_coordinator() -> None:
    manager = _build_manager()
    coordinator = manager._queue_sweep_coordinator_for_manager()

    coordinator.cursor = 7

    assert manager._queue_sweep_coordinator_for_manager() is coordinator
    assert manager._queue_sweep_coordinator_for_manager().cursor == 7
    assert not hasattr(manager, "_queue_sweep_cursor")
