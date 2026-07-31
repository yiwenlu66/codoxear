import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"


CORE_METHODS = {
    "__init__": "init_for_manager",
    "stop": "stop_for_manager",
    "_reset_log_caches": "reset_log_caches_for_manager",
    "_session_run_settings": "session_run_settings_for_manager",
    "_session_transport": "session_transport_for_manager",
    "_discover_existing_if_stale": "discover_existing_if_stale_for_manager",
    "_new_session_store_for_manager": "new_session_store_for_manager",
    "_session_store_for_manager": "session_store_for_manager",
    "_queue_store_for_manager": "queue_store_for_manager",
    "_input_lock_for_session": "input_lock_for_session",
    "_broker_busy_queue_from_state": "broker_busy_queue_from_state_for_manager",
    "_log_size_or_none": "log_size_or_none_for_manager",
    "_clear_confirmed_send_boundary_locked": "clear_confirmed_send_boundary_locked_for_manager",
    "_confirmed_send_boundary_unresolved_for_session": "confirmed_send_boundary_unresolved_for_manager",
    "_voice_push_scan_loop": "voice_push_scan_loop_for_manager",
    "_unattended_loop": "unattended_loop_for_manager",
    "_queue_loop": "queue_loop_for_manager",
    "_maybe_drain_session_queue": "maybe_drain_session_queue_for_manager",
    "_discover_existing": "discover_existing_for_manager",
    "get_session": "get_session_for_manager",
    "_sock_call": "sock_call_for_manager",
}

FACTORY_METHODS = {
    "_discovery_deps": "discovery_deps_for_manager",
    "_queue_coordinator_for_manager": "queue_coordinator_for_manager",
    "_control_coordinator_for_manager": "control_coordinator_for_manager",
    "_list_coordinator_for_manager": "list_coordinator_for_manager",
    "_refresh_coordinator_for_manager": "refresh_coordinator_for_manager",
    "_readiness_coordinator_for_manager": "readiness_coordinator_for_manager",
    "_unattended_sweep_coordinator_for_manager": "unattended_sweep_coordinator_for_manager",
    "_queue_sweep_coordinator_for_manager": "queue_sweep_coordinator_for_manager",
    "_voice_runtime_for_manager": "voice_runtime_for_manager",
    "_log_runtime_for_manager": "log_runtime_for_manager",
    "_files_coordinator_for_manager": "files_coordinator_for_manager",
    "_ui_state_coordinator_for_manager": "ui_state_coordinator_for_manager",
    "_unattended_config_coordinator_for_manager": "unattended_config_coordinator_for_manager",
    "_cleanup_coordinator_for_manager": "cleanup_coordinator_for_manager",
    "_pending_state_coordinator_for_manager": "pending_state_coordinator_for_manager",
    "_recent_cwd_coordinator_for_manager": "recent_cwd_coordinator_for_manager",
    "_lifecycle_coordinator_for_manager": "lifecycle_coordinator_for_manager",
    "_discovery_registry_for_manager": "discovery_registry_for_manager",
    "_prune_coordinator_for_manager": "prune_coordinator_for_manager",
    "_send_coordinator_for_manager": "send_coordinator_for_manager",
    "_prelog_user_message_recorder_for_manager": "prelog_user_message_recorder_for_manager",
    "_web_launch_coordinator_for_manager": "web_launch_coordinator_for_manager",
}

FORWARD_METHODS = {
    "_hide_session": ("_ui_state_coordinator_for_manager", "hide_session"),
    "alias_set": ("_ui_state_coordinator_for_manager", "alias_set"),
    "queue_delete": ("_queue_coordinator_for_manager", "delete_local"),
    "unattended_set": ("_unattended_config_coordinator_for_manager", "set"),
    "get_state": ("_control_coordinator_for_manager", "get_state"),
}


def _is_vararg_coordinator_forward(method: ast.FunctionDef) -> bool:
    if method.args.vararg is None or method.args.vararg.arg != "args":
        return False
    if method.args.kwarg is None or method.args.kwarg.arg != "kwargs":
        return False
    if len(method.body) != 1 or not isinstance(method.body[0], ast.Return):
        return False
    value = method.body[0].value
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Call)
        and isinstance(value.func.value.func, ast.Attribute)
        and isinstance(value.func.value.func.value, ast.Name)
        and value.func.value.func.value.id == "self"
    )


def _session_manager_methods() -> dict[str, ast.FunctionDef]:
    module = ast.parse(SERVER_PY.read_text(encoding="utf-8"))
    session_manager = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "SessionManager"
    )
    return {node.name: node for node in session_manager.body if isinstance(node, ast.FunctionDef)}


def test_session_manager_core_methods_are_explicit_composition() -> None:
    methods = _session_manager_methods()

    for public_name, implementation_name in CORE_METHODS.items():
        method = methods[public_name]
        assert ast.unparse(method.body[0]) == (
            f"return _core_methods.{implementation_name}(self, sys.modules[__name__], *args, **kwargs)"
        )


def test_session_manager_factory_methods_are_explicit_composition() -> None:
    methods = _session_manager_methods()

    for public_name, factory_name in FACTORY_METHODS.items():
        method = methods[public_name]
        assert ast.unparse(method.body[0]) == (
            f"return _factories.{factory_name}(self, _session_manager_factory_caps_impl(sys.modules[__name__]))"
        )


def test_session_manager_forward_methods_are_explicit_composition() -> None:
    methods = _session_manager_methods()

    forwarders = [method for method in methods.values() if _is_vararg_coordinator_forward(method)]
    assert len(forwarders) == 73
    for public_name, (factory_name, coordinator_method) in FORWARD_METHODS.items():
        method = methods[public_name]
        assert ast.unparse(method.body[0]) == f"return self.{factory_name}().{coordinator_method}(*args, **kwargs)"


def test_session_manager_has_no_runtime_method_binding() -> None:
    source = SERVER_PY.read_text(encoding="utf-8")

    assert "_bind_session_manager_methods" not in source
    assert not (ROOT / "codoxear" / "session_manager_method_bindings.py").exists()
    assert "__server_module" not in source
    assert "def inject_keys(self, session_id: str, seq: str, *, track_request_sent: bool = False, interrupt: bool = False)" in source
