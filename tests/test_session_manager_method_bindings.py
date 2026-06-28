import sys
import types
from pathlib import Path

from codoxear import session_manager_method_bindings as bindings
from codoxear.session_manager_method_bindings import bind_session_manager_forwarders
from codoxear.session_manager_method_bindings import bind_session_manager_server_factories
from codoxear.session_manager_method_bindings import core_method


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


def test_server_factory_binding_uses_live_server_module_lookup() -> None:
    module_name = "_codoxear_test_server_binding"
    module = types.SimpleNamespace()
    calls: list[tuple[object, object]] = []

    def impl(manager: object, server_module: object) -> dict[str, object]:
        calls.append((manager, server_module))
        return {"server": server_module}

    caps = types.SimpleNamespace(name="caps")
    module._session_manager_factory_caps_impl = lambda server_module: caps
    module._queue_coordinator_for_manager_impl = impl
    sys.modules[module_name] = module  # type: ignore[assignment]
    try:
        class FactoryBound:
            pass

        bind_session_manager_server_factories(FactoryBound, server_module_name=module_name)
        manager = FactoryBound()

        assert FactoryBound._queue_coordinator_for_manager.__name__ == "_queue_coordinator_for_manager"
        assert manager._queue_coordinator_for_manager() == {"server": caps}
        assert calls == [(manager, caps)]

        replacement_caps = types.SimpleNamespace(name="replacement-caps")
        replacement = types.SimpleNamespace()
        replacement._session_manager_factory_caps_impl = lambda server_module: replacement_caps
        replacement._queue_coordinator_for_manager_impl = lambda manager, factory_caps: {"server": factory_caps, "manager": manager}
        sys.modules[module_name] = replacement  # type: ignore[assignment]
        assert manager._queue_coordinator_for_manager() == {"server": replacement_caps, "manager": manager}
    finally:
        sys.modules.pop(module_name, None)


def test_core_method_binding_uses_live_server_module_lookup() -> None:
    module_name = "_codoxear_test_server_core_binding"
    calls: list[tuple[object, object, tuple[object, ...], dict[str, object]]] = []

    def impl(manager: object, server_module: object, *args: object, **kwargs: object) -> dict[str, object]:
        calls.append((manager, server_module, args, kwargs))
        return {"server": server_module, "args": args, "kwargs": kwargs}

    setattr(bindings._core_methods, "_fake_core_impl", impl)
    first_server = types.SimpleNamespace(name="first")
    sys.modules[module_name] = first_server  # type: ignore[assignment]
    try:
        manager = object()
        method = core_method("fake_core", "_fake_core_impl", module_name)

        assert method.__name__ == "fake_core"
        assert method(manager, "a", flag=True) == {"server": first_server, "args": ("a",), "kwargs": {"flag": True}}

        second_server = types.SimpleNamespace(name="second")
        sys.modules[module_name] = second_server  # type: ignore[assignment]
        assert method(manager, "b") == {"server": second_server, "args": ("b",), "kwargs": {}}
        assert calls[0][1] is first_server
        assert calls[1][1] is second_server
    finally:
        sys.modules.pop(module_name, None)
        delattr(bindings._core_methods, "_fake_core_impl")


def test_session_manager_compatibility_forwards_live_outside_server() -> None:
    server_source = (ROOT / "codoxear" / "server.py").read_text(encoding="utf-8")
    binding_source = (ROOT / "codoxear" / "session_manager_method_bindings.py").read_text(encoding="utf-8")

    assert "@_bind_session_manager_methods(__name__)\nclass SessionManager" in server_source
    assert "def __init__(self)" not in server_source
    assert "def _session_run_settings(" not in server_source
    assert "def _sock_call(" not in server_source
    assert "def alias_set(" not in server_source
    assert "def queue_delete(" not in server_source
    assert "def inject_attachment_keys(" not in server_source
    assert "def _queue_coordinator_for_manager(" not in server_source
    assert '("alias_set", "_ui_state_coordinator_for_manager", "alias_set")' in binding_source
    assert '("queue_delete", "_queue_coordinator_for_manager", "delete_local")' in binding_source
    assert '("inject_attachment_keys", "_attachment_coordinator_for_manager", "inject_attachment_keys")' in binding_source
    assert '("__init__", "init_for_manager")' in binding_source
    assert '("_sock_call", "sock_call_for_manager")' in binding_source
    assert '("_queue_coordinator_for_manager", "_queue_coordinator_for_manager_impl")' in binding_source
    assert "_session_manager_factory_caps_impl(server_module)" in binding_source
    assert "getattr(_core_methods, impl_name)(manager, server_module, *args, **kwargs)" in binding_source
    assert "def inject_keys(self, session_id: str, seq: str, *, track_request_sent: bool = False, interrupt: bool = False)" in server_source
