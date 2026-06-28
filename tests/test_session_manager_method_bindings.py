from pathlib import Path

from codoxear.session_manager_method_bindings import bind_session_manager_forwarders


ROOT = Path(__file__).resolve().parents[1]


class _UiCoordinator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def alias_set(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.calls.append(("alias_set", args, kwargs))
        return {"ok": True, "args": args, "kwargs": kwargs}


class _UnattendedCoordinator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def set(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.calls.append(("set", args, kwargs))
        return {"ok": True, "args": args, "kwargs": kwargs}


@bind_session_manager_forwarders
class _BoundManager:
    def __init__(self) -> None:
        self.ui = _UiCoordinator()
        self.unattended = _UnattendedCoordinator()

    def _ui_state_coordinator_for_manager(self) -> _UiCoordinator:
        return self.ui

    def _unattended_config_coordinator_for_manager(self) -> _UnattendedCoordinator:
        return self.unattended


def test_forwarder_preserves_public_name_and_delegates_args() -> None:
    manager = _BoundManager()

    assert _BoundManager.alias_set.__name__ == "alias_set"
    assert manager.alias_set("session-a", "label") == {
        "ok": True,
        "args": ("session-a", "label"),
        "kwargs": {},
    }
    assert manager.ui.calls == [("alias_set", ("session-a", "label"), {})]


def test_forwarder_preserves_public_name_when_target_method_differs() -> None:
    manager = _BoundManager()

    assert _BoundManager.unattended_set.__name__ == "unattended_set"
    assert manager.unattended_set("session-a", enabled=True) == {
        "ok": True,
        "args": ("session-a",),
        "kwargs": {"enabled": True},
    }
    assert manager.unattended.calls == [("set", ("session-a",), {"enabled": True})]


def test_session_manager_compatibility_forwards_live_outside_server() -> None:
    server_source = (ROOT / "codoxear" / "server.py").read_text(encoding="utf-8")
    binding_source = (ROOT / "codoxear" / "session_manager_method_bindings.py").read_text(encoding="utf-8")

    assert "@_bind_session_manager_forwarders\nclass SessionManager" in server_source
    assert "def alias_set(" not in server_source
    assert "def queue_delete(" not in server_source
    assert "def inject_attachment_keys(" not in server_source
    assert '("alias_set", "_ui_state_coordinator_for_manager", "alias_set")' in binding_source
    assert '("queue_delete", "_queue_coordinator_for_manager", "delete_local")' in binding_source
    assert '("inject_attachment_keys", "_attachment_coordinator_for_manager", "inject_attachment_keys")' in binding_source
    assert "def inject_keys(self, session_id: str, seq: str, *, track_request_sent: bool = False, interrupt: bool = False)" in server_source
