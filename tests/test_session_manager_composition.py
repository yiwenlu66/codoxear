from __future__ import annotations

from codoxear import server


def _recording_coordinator(method_name: str, result: object, calls: list[tuple[tuple[object, ...], dict[str, object]]]):
    class Coordinator:
        pass

    coordinator = Coordinator()

    def record(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return result

    setattr(coordinator, method_name, record)
    return coordinator


def test_session_manager_routes_operations_to_their_runtime_coordinators() -> None:
    """Public manager operations preserve arguments and return their owner result."""
    manager = object.__new__(server.SessionManager)
    cases = (
        ("_ui_state_coordinator_for_manager", "hide_session", "_hide_session", ("sid",), {}, "hidden"),
        ("_ui_state_coordinator_for_manager", "alias_set", "alias_set", ("sid", "focus"), {}, "aliased"),
        ("_queue_coordinator_for_manager", "delete_local", "queue_delete", ("sid", "item"), {"allow_commit_unknown": True}, "deleted"),
        ("_unattended_config_coordinator_for_manager", "set", "unattended_set", ("sid",), {"enabled": True}, "saved"),
        ("_control_coordinator_for_manager", "get_state", "get_state", ("sid",), {}, {"busy": False}),
    )

    for factory_name, coordinator_method, manager_method, args, kwargs, expected in cases:
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        setattr(manager, factory_name, lambda name=coordinator_method, value=expected, seen=calls: _recording_coordinator(name, value, seen))

        assert getattr(manager, manager_method)(*args, **kwargs) == expected
        assert calls == [(args, kwargs)]
