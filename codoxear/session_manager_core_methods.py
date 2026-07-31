from __future__ import annotations

from pathlib import Path
from typing import Any

from .session_registry import SessionRegistry
from .session_registry import session_registry_for_manager


def session_store_paths_for_server(server: Any) -> Any:
    return server._session_store_paths_impl(
        aliases=server.ALIAS_PATH,
        sidebar_meta=server.SIDEBAR_META_PATH,
        hidden_sessions=server.HIDDEN_SESSIONS_PATH,
        files=server.FILE_HISTORY_PATH,
        queues=server.QUEUE_PATH,
        pending_attachments=server.PENDING_ATTACHMENTS_PATH,
        staged_attachments=server.STAGED_ATTACHMENTS_PATH,
        commit_unknown_sends=server.COMMIT_UNKNOWN_SENDS_PATH,
        recent_cwds=server.RECENT_CWD_PATH,
        unattended=server.UNATTENDED_PATH,
        uploads_root=server.UPLOAD_DIR,
    )


def init_for_manager(manager: Any, server: Any) -> None:
    registry = SessionRegistry()
    manager._registry = registry
    manager._store = manager._new_session_store_for_manager(session_store_paths_for_server(server))
    server._seed_manager_in_memory_state_impl(manager)
    server._load_manager_persistent_state_impl(manager)
    manager._voice_push = server._create_voice_push_coordinator_impl(
        voice_push_factory=server.VoicePushCoordinator,
        app_dir=server.APP_DIR,
        stop_event=registry.stop_event,
        settings_path=server.VOICE_SETTINGS_PATH,
        subscriptions_path=server.PUSH_SUBSCRIPTIONS_PATH,
        delivery_ledger_path=server.DELIVERY_LEDGER_PATH,
        vapid_private_key_path=server.VAPID_PRIVATE_KEY_PATH,
    )
    manager._discover_existing(force=True)
    manager._prune_missing_commit_unknown_sends()
    server._start_manager_worker_threads_impl(manager=manager, thread_factory=server.threading.Thread)


def stop_for_manager(manager: Any, server: Any) -> None:
    session_registry_for_manager(manager).stop_event.set()


def reset_log_caches_for_manager(manager: Any, server: Any, session: Any, *, meta_log_off: int) -> None:
    return server._reset_session_log_caches_impl(session, meta_log_off=meta_log_off)


def session_run_settings_for_manager(
    manager: Any,
    server: Any,
    *,
    meta: dict[str, Any],
    log_path: Path | None,
    agent_backend: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    return server._session_run_settings_from_meta_impl(
        meta=meta,
        log_path=log_path,
        agent_backend=agent_backend,
        clean_optional_text=server._clean_optional_text,
        normalize_requested_preferred_auth_method=server._normalize_requested_preferred_auth_method,
        display_reasoning_effort=server._display_reasoning_effort,
        display_pi_reasoning_effort=server._display_pi_reasoning_effort,
        normalize_requested_cc_reasoning_effort=server._normalize_requested_cc_reasoning_effort,
        read_run_settings_from_log=server._read_run_settings_from_log,
    )


def session_transport_for_manager(manager: Any, server: Any, *, meta: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    return server._session_transport_from_meta_impl(meta=meta, clean_optional_text=server._clean_optional_text)


def discover_existing_if_stale_for_manager(manager: Any, server: Any, *, force: bool = False) -> None:
    return server._discover_existing_if_stale_for_manager_impl(
        manager,
        force=force,
        discover_min_interval_seconds=server.DISCOVER_MIN_INTERVAL_SECONDS,
        now=server.time.time,
    )


def new_session_store_for_manager(manager: Any, server: Any, paths: Any) -> Any:
    return server._create_session_store_impl(
        paths=paths,
        file_history_max=server.FILE_HISTORY_MAX,
        recent_cwd_max=server.RECENT_CWD_MAX,
        unattended_default_idle_minutes=server.UNATTENDED_DEFAULT_IDLE_MINUTES,
        unattended_default_max_injections=server.UNATTENDED_DEFAULT_MAX_INJECTIONS,
        clean_alias=server._clean_alias,
        clean_priority_offset=server._clean_priority_offset,
        clean_snooze_until=server._clean_snooze_until,
        clean_dependency_session_id=server._clean_dependency_session_id,
        clean_recent_cwd=server._clean_recent_cwd,
        clean_commit_unknown_send_record=manager._clean_commit_unknown_send_record,
    )


def session_store_for_manager(manager: Any, server: Any) -> Any:
    store = server._session_store_for_manager_impl(
        existing=getattr(manager, "_store", None),
        paths=session_store_paths_for_server(server),
        create_store=manager._new_session_store_for_manager,
    )
    manager._store = store
    return store


def queue_store_for_manager(manager: Any, server: Any) -> Any:
    return manager._session_store_for_manager().queue_store


def input_lock_for_session(manager: Any, server: Any, session_id: str) -> Any:
    return server._input_lock_for_session_impl(manager, session_id)


def broker_busy_queue_from_state_for_manager(manager: Any, server: Any, state: dict[str, Any]) -> tuple[bool, int]:
    return server._broker_busy_queue_from_state(state)


def log_size_or_none_for_manager(manager: Any, server: Any, log_path: Path | None) -> int | None:
    return server._log_path_size_or_none(log_path)


def clear_confirmed_send_boundary_locked_for_manager(manager: Any, server: Any, session: Any) -> None:
    server._clear_session_confirmed_send_boundary(session)


def confirmed_send_boundary_unresolved_for_manager(manager: Any, server: Any, session_id: str, log_path: Path | None, log_size: int | None) -> bool:
    registry = session_registry_for_manager(manager)
    with registry.lock:
        session = registry.sessions.get(session_id)
        return server._consume_session_confirmed_send_boundary(session, log_path, log_size)


def voice_push_scan_loop_for_manager(manager: Any, server: Any) -> None:
    return server._voice_push_scan_loop_impl(manager, wait_seconds=server.VOICE_PUSH_SWEEP_SECONDS, stderr=server.sys.stderr, print_exc=server.traceback.print_exc)


def unattended_loop_for_manager(manager: Any, server: Any) -> None:
    return server._unattended_loop_impl(manager, wait_seconds=server.UNATTENDED_SWEEP_SECONDS, stderr=server.sys.stderr, print_exc=server.traceback.print_exc)


def queue_loop_for_manager(manager: Any, server: Any) -> None:
    return server._queue_loop_impl(manager, wait_seconds=server.QUEUE_SWEEP_SECONDS, stderr=server.sys.stderr)


def maybe_drain_session_queue_for_manager(manager: Any, server: Any, session_id: str, *, now_ts: float | None = None) -> bool:
    response = manager._promote_queue_head_if_sendable(session_id, require_idle_grace=True, now_ts=now_ts)
    return isinstance(response, dict)


def discover_existing_for_manager(manager: Any, server: Any, *, force: bool = False) -> None:
    return server._discover_existing_for_manager_impl(
        manager,
        force=force,
        discover_min_interval_seconds=server.DISCOVER_MIN_INTERVAL_SECONDS,
        sock_dir=server.SOCK_DIR,
        proc_root=server.PROC_ROOT,
        discover_sessions=server._discover_sessions,
        now=server.time.time,
    )


def get_session_for_manager(manager: Any, server: Any, session_id: str) -> Any | None:
    registry = session_registry_for_manager(manager)
    with registry.lock:
        return registry.sessions.get(session_id)


def sock_call_for_manager(
    manager: Any,
    server: Any,
    sock_path: Path,
    req: dict[str, Any],
    timeout_s: float | None = 2.0,
    *,
    track_request_sent: bool = False,
) -> dict[str, Any]:
    return server._call_control_socket_impl(sock_path, req, timeout_s=timeout_s, track_request_sent=track_request_sent)
