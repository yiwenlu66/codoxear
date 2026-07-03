from __future__ import annotations

import sys
from typing import Any

from . import session_manager_core_methods as _core_methods


SESSION_MANAGER_CORE_METHODS: tuple[tuple[str, str], ...] = (
    ("__init__", "init_for_manager"),
    ("stop", "stop_for_manager"),
    ("_reset_log_caches", "reset_log_caches_for_manager"),
    ("_session_run_settings", "session_run_settings_for_manager"),
    ("_session_transport", "session_transport_for_manager"),
    ("_discover_existing_if_stale", "discover_existing_if_stale_for_manager"),
    ("_new_session_store_for_manager", "new_session_store_for_manager"),
    ("_session_store_for_manager", "session_store_for_manager"),
    ("_queue_store_for_manager", "queue_store_for_manager"),
    ("_input_lock_for_session", "input_lock_for_session"),
    ("_broker_busy_queue_from_state", "broker_busy_queue_from_state_for_manager"),
    ("_log_size_or_none", "log_size_or_none_for_manager"),
    ("_clear_confirmed_send_boundary_locked", "clear_confirmed_send_boundary_locked_for_manager"),
    ("_confirmed_send_boundary_unresolved_for_session", "confirmed_send_boundary_unresolved_for_manager"),
    ("_voice_push_scan_loop", "voice_push_scan_loop_for_manager"),
    ("_unattended_loop", "unattended_loop_for_manager"),
    ("_queue_loop", "queue_loop_for_manager"),
    ("_maybe_drain_session_queue", "maybe_drain_session_queue_for_manager"),
    ("_discover_existing", "discover_existing_for_manager"),
    ("get_session", "get_session_for_manager"),
    ("_sock_call", "sock_call_for_manager"),
)


SESSION_MANAGER_SERVER_FACTORY_METHODS: tuple[tuple[str, str], ...] = (
    ("_discovery_deps", "_discovery_deps_for_manager_impl"),
    ("_queue_coordinator_for_manager", "_queue_coordinator_for_manager_impl"),
    ("_control_coordinator_for_manager", "_control_coordinator_for_manager_impl"),
    ("_attachment_coordinator_for_manager", "_attachment_coordinator_for_manager_impl"),
    ("_list_coordinator_for_manager", "_list_coordinator_for_manager_impl"),
    ("_refresh_coordinator_for_manager", "_refresh_coordinator_for_manager_impl"),
    ("_readiness_coordinator_for_manager", "_readiness_coordinator_for_manager_impl"),
    ("_unattended_sweep_coordinator_for_manager", "_unattended_sweep_coordinator_for_manager_impl"),
    ("_queue_sweep_coordinator_for_manager", "_queue_sweep_coordinator_for_manager_impl"),
    ("_voice_runtime_for_manager", "_voice_runtime_for_manager_impl"),
    ("_log_runtime_for_manager", "_log_runtime_for_manager_impl"),
    ("_files_coordinator_for_manager", "_files_coordinator_for_manager_impl"),
    ("_ui_state_coordinator_for_manager", "_ui_state_coordinator_for_manager_impl"),
    ("_unattended_config_coordinator_for_manager", "_unattended_config_coordinator_for_manager_impl"),
    ("_cleanup_coordinator_for_manager", "_cleanup_coordinator_for_manager_impl"),
    ("_pending_state_coordinator_for_manager", "_pending_state_coordinator_for_manager_impl"),
    ("_recent_cwd_coordinator_for_manager", "_recent_cwd_coordinator_for_manager_impl"),
    ("_lifecycle_coordinator_for_manager", "_lifecycle_coordinator_for_manager_impl"),
    ("_discovery_registry_for_manager", "_discovery_registry_for_manager_impl"),
    ("_prune_coordinator_for_manager", "_prune_coordinator_for_manager_impl"),
    ("_send_coordinator_for_manager", "_send_coordinator_for_manager_impl"),
    ("_prelog_user_message_recorder_for_manager", "_prelog_user_message_recorder_for_manager_impl"),
    ("_web_launch_coordinator_for_manager", "_web_launch_coordinator_for_manager_impl"),
)


SESSION_MANAGER_FORWARD_METHODS: tuple[tuple[str, str, str], ...] = (
    ("_hide_session", "_ui_state_coordinator_for_manager", "hide_session"),
    ("_unhide_session", "_ui_state_coordinator_for_manager", "unhide_session"),
    ("alias_set", "_ui_state_coordinator_for_manager", "alias_set"),
    ("alias_get", "_ui_state_coordinator_for_manager", "alias_get"),
    ("alias_clear", "_ui_state_coordinator_for_manager", "alias_clear"),
    ("sidebar_meta_get", "_ui_state_coordinator_for_manager", "sidebar_meta_get"),
    ("sidebar_meta_set", "_ui_state_coordinator_for_manager", "sidebar_meta_set"),
    ("edit_session", "_ui_state_coordinator_for_manager", "edit_session"),
    ("_prune_stale_socket_without_metadata", "_cleanup_coordinator_for_manager", "prune_stale_socket_without_metadata"),
    ("_clear_deleted_session_state", "_cleanup_coordinator_for_manager", "clear_deleted_session_state"),
    ("_set_pending_attachment", "_pending_state_coordinator_for_manager", "set_pending_attachment"),
    ("clear_pending_attachment", "_pending_state_coordinator_for_manager", "clear_pending_attachment"),
    ("_clean_commit_unknown_send_record", "_pending_state_coordinator_for_manager", "clean_commit_unknown_send_record"),
    ("_set_commit_unknown_send", "_pending_state_coordinator_for_manager", "set_commit_unknown_send"),
    ("clear_commit_unknown_send", "_pending_state_coordinator_for_manager", "clear_commit_unknown_send"),
    ("_prune_missing_commit_unknown_sends", "_pending_state_coordinator_for_manager", "prune_missing_commit_unknown_sends"),
    ("_remember_recent_cwd", "_recent_cwd_coordinator_for_manager", "remember"),
    ("_backfill_recent_cwds_from_logs", "_recent_cwd_coordinator_for_manager", "backfill_from_logs"),
    ("recent_cwds", "_recent_cwd_coordinator_for_manager", "list_recent"),
    ("_queue_len", "_queue_coordinator_for_manager", "queue_len"),
    ("_mark_queue_orphan_recovery_locked", "_queue_coordinator_for_manager", "mark_orphan_recovery_locked"),
    ("_queue_has_recovery_items_locked", "_queue_coordinator_for_manager", "has_recovery_items_locked"),
    ("_queue_list_local", "_queue_coordinator_for_manager", "list_local"),
    ("_queue_append_item_local", "_queue_coordinator_for_manager", "append_item_local"),
    ("_queue_enqueue_local", "_queue_coordinator_for_manager", "enqueue_local"),
    ("_queue_delete_local", "_queue_coordinator_for_manager", "delete_local"),
    ("_queue_update_local", "_queue_coordinator_for_manager", "update_local"),
    ("_queue_move_local", "_queue_coordinator_for_manager", "move_local"),
    ("_queue_session_state", "_queue_coordinator_for_manager", "session_state"),
    ("_promote_queue_head_if_sendable", "_queue_coordinator_for_manager", "promote_head_if_sendable"),
    ("_runtime_status_from_state_and_log", "_readiness_coordinator_for_manager", "runtime_status_from_state_and_log"),
    ("_remote_ready_from_state_and_log", "_readiness_coordinator_for_manager", "remote_ready_from_state_and_log"),
    ("_remote_state_after_metadata_probe", "_readiness_coordinator_for_manager", "remote_state_after_metadata_probe"),
    ("_send_remote_ready", "_readiness_coordinator_for_manager", "send_remote_ready"),
    ("_queue_remote_ready", "_readiness_coordinator_for_manager", "queue_remote_ready"),
    ("attachment_injection_ready", "_readiness_coordinator_for_manager", "attachment_injection_ready"),
    ("_files_key_for_session", "_files_coordinator_for_manager", "files_key_for_session"),
    ("files_get", "_files_coordinator_for_manager", "get"),
    ("files_add", "_files_coordinator_for_manager", "add"),
    ("files_clear", "_files_coordinator_for_manager", "clear"),
    ("unattended_get", "_unattended_config_coordinator_for_manager", "get"),
    ("unattended_set", "_unattended_config_coordinator_for_manager", "set"),
    ("_session_display_name", "_voice_runtime_for_manager", "session_display_name"),
    ("_observe_rollout_delta", "_voice_runtime_for_manager", "observe_rollout_delta"),
    ("_voice_push_scan_sweep", "_voice_runtime_for_manager", "scan_sweep"),
    ("_unattended_sweep", "_unattended_sweep_coordinator_for_manager", "sweep"),
    ("_queue_sweep", "_queue_sweep_coordinator_for_manager", "sweep"),
    ("_apply_discovery_result", "_discovery_registry_for_manager", "apply_result"),
    ("_upsert_discovery_registration", "_discovery_registry_for_manager", "upsert_registration"),
    ("_refresh_session_state", "_prune_coordinator_for_manager", "refresh_session_state"),
    ("_prune_dead_sessions", "_prune_coordinator_for_manager", "prune_dead_sessions"),
    ("_update_meta_counters", "_log_runtime_for_manager", "update_meta_counters"),
    ("list_sessions", "_list_coordinator_for_manager", "list_sessions"),
    ("_attach_notification_texts", "_voice_runtime_for_manager", "attach_notification_texts"),
    ("mark_log_delta", "_log_runtime_for_manager", "mark_log_delta"),
    ("idle_from_log", "_log_runtime_for_manager", "idle_from_log"),
    ("idle_from_log_path", "_log_runtime_for_manager", "idle_from_log_path"),
    ("_kill_session_via_pids", "_lifecycle_coordinator_for_manager", "kill_session_via_pids"),
    ("kill_session", "_lifecycle_coordinator_for_manager", "kill_session"),
    ("_live_session_for_resume_target", "_lifecycle_coordinator_for_manager", "live_session_for_resume_target"),
    ("delete_session", "_lifecycle_coordinator_for_manager", "delete_session"),
    ("_record_prelog_user_message", "_prelog_user_message_recorder_for_manager", "record"),
    ("enqueue", "_queue_coordinator_for_manager", "enqueue"),
    ("queue_list", "_queue_coordinator_for_manager", "list_local"),
    ("queue_delete", "_queue_coordinator_for_manager", "delete_local"),
    ("queue_update", "_queue_coordinator_for_manager", "update_local"),
    ("queue_move", "_queue_coordinator_for_manager", "move_local"),
    ("get_state", "_control_coordinator_for_manager", "get_state"),
    ("get_tail", "_control_coordinator_for_manager", "get_tail"),
    ("inject_attachment_keys", "_attachment_coordinator_for_manager", "inject_attachment_keys"),
)


def coordinator_forwarder(public_name: str, coordinator_factory_name: str, method_name: str) -> Any:
    def method(manager: Any, *args: Any, **kwargs: Any) -> Any:
        coordinator = getattr(manager, coordinator_factory_name)()
        return getattr(coordinator, method_name)(*args, **kwargs)

    method.__name__ = public_name
    method.__qualname__ = public_name
    return method


def server_factory_method(public_name: str, impl_name: str, server_module_name: str) -> Any:
    def method(manager: Any) -> Any:
        server_module = sys.modules[server_module_name]
        factory_caps = server_module._session_manager_factory_caps_impl(server_module)
        return getattr(server_module, impl_name)(manager, factory_caps)

    method.__name__ = public_name
    method.__qualname__ = public_name
    return method


def core_method(public_name: str, impl_name: str, server_module_name: str) -> Any:
    def method(manager: Any, *args: Any, **kwargs: Any) -> Any:
        server_module = sys.modules[server_module_name]
        return getattr(_core_methods, impl_name)(manager, server_module, *args, **kwargs)

    method.__name__ = public_name
    method.__qualname__ = public_name
    return method


def bind_session_manager_core_methods(cls: type[Any], *, server_module_name: str) -> type[Any]:
    for public_name, impl_name in SESSION_MANAGER_CORE_METHODS:
        method = core_method(public_name, impl_name, server_module_name)
        method.__qualname__ = f"{cls.__qualname__}.{public_name}"
        setattr(cls, public_name, method)
    return cls


def bind_session_manager_forwarders(cls: type[Any]) -> type[Any]:
    for public_name, coordinator_factory_name, method_name in SESSION_MANAGER_FORWARD_METHODS:
        method = coordinator_forwarder(public_name, coordinator_factory_name, method_name)
        method.__qualname__ = f"{cls.__qualname__}.{public_name}"
        setattr(cls, public_name, method)
    return cls


def bind_session_manager_server_factories(cls: type[Any], *, server_module_name: str) -> type[Any]:
    for public_name, impl_name in SESSION_MANAGER_SERVER_FACTORY_METHODS:
        method = server_factory_method(public_name, impl_name, server_module_name)
        method.__qualname__ = f"{cls.__qualname__}.{public_name}"
        setattr(cls, public_name, method)
    return cls


def bind_session_manager_methods(server_module_name: str) -> Any:
    def decorator(cls: type[Any]) -> type[Any]:
        bind_session_manager_core_methods(cls, server_module_name=server_module_name)
        bind_session_manager_forwarders(cls)
        bind_session_manager_server_factories(cls, server_module_name=server_module_name)
        return cls

    return decorator
