from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from .queue_sweep import QueueSweepCoordinator
from .session_discovery import DiscoveryDeps
from .session_attachment import SessionAttachmentCoordinator
from .session_cleanup import SessionCleanupCoordinator
from .session_control import SessionControlCoordinator
from .session_discovery_registry import SessionDiscoveryRegistryCoordinator
from .session_files import SessionFilesCoordinator
from .session_lifecycle import SessionLifecycleCoordinator
from .session_list import SessionListCoordinator
from .session_log_runtime import SessionLogRuntimeCoordinator
from .session_pending_state import SessionPendingStateCoordinator
from .session_prune import SessionPruneCoordinator
from .session_queue import SessionQueueCoordinator
from .session_readiness import SessionReadinessCoordinator
from .session_recent_cwd import SessionRecentCwdCoordinator
from .session_refresh import SessionRefreshCoordinator
from .session_runtime import ListingRuntimeProbes
from .session_send import PrelogUserMessageRecorder
from .session_send import SessionSendCoordinator
from .session_ui_state import SessionUiStateCoordinator
from .session_unattended_config import SessionUnattendedConfigCoordinator
from .session_web_launch import SessionWebLaunchCoordinator
from .unattended_sweep import UnattendedSweepCoordinator
from .voice_runtime import VoiceRuntimeCoordinator


@dataclass(frozen=True)
class SessionManagerFactoryCaps:
    analyze_log_chunk: Any
    broker_interrupted_idle_from_state: Any
    broker_tail_has_session_detach_marker: Any
    cc_pending_tool_ids_before: Any
    clean_alias: Any
    clean_dependency_session_id: Any
    clean_optional_text: Any
    clean_priority_offset: Any
    clean_recent_cwd: Any
    clean_snooze_until: Any
    clean_unattended_cooldown_minutes: Any
    clean_unattended_remaining_injections: Any
    coerce_main_thread_log: Any
    codex_trust_override_for_path: Any
    commit_unknown_error: Any
    commit_unknown_orphan_prune_seconds: float
    compute_idle_from_log: Any
    control_socket_call_error: Any
    create_git_worktree: Any
    current_git_branch: Any
    dotenv_path: Any
    drain_stream: Any
    environ: Any
    extract_delivery_messages: Any
    find_latest_token_update: Any
    homes: dict[str, Any]
    injection_error: Any
    iter_session_logs: Any
    last_chat_role_ts_from_tail: Any
    last_conversation_ts_from_tail: Any
    latest_launch_attempt: Any
    launch_attempt_row: Any
    launch_attempts_path: Any
    launch_error: Any
    launch_failure_tail: Any
    list_resume_candidates_for_cwd: Any
    load_env_file: Any
    log_invalid_sidecar_metadata: Any
    max_scan_bytes: int
    normalize_requested_service_tier: Any
    not_ready_error: Any
    now: Any
    pid_alive: Any
    popen: Any
    priority_bucket_seconds: float
    priority_half_life_seconds: float
    proc_find_open_rollout_log: Any
    proc_root: Any
    process_group_alive: Any
    prompt_prefix: str
    provider_choice_for_settings: Any
    python_executable: str
    queue_idle_grace_seconds: float
    read_jsonl_from_offset: Any
    read_launch_attempts: Any
    read_run_settings_from_log: Any
    read_session_meta_or_none: Any
    recent_cwd_max: int
    record_launch_attempt: Any
    repo_root: Any
    resolve_dir_target: Any
    resolve_session_cwd: Any
    resume_candidate_from_log: Any
    run: Any
    send_commit_timeout_seconds: float
    sock_error_definitely_stale: Any
    socket_timeout: Any
    stderr: Any
    submitted_user_messages: Any
    terminate_process: Any
    terminate_process_group: Any
    tmux_pane_snapshot: Any
    tmux_session_name: str
    turn_context_run_settings: Any
    unattended_default_idle_minutes: int
    unattended_default_max_injections: int
    unlink_quiet: Any
    wait_for_spawned_broker_meta: Any
    wait_or_raise: Any
    which_tmux: Any


def session_manager_factory_caps(server: Any) -> SessionManagerFactoryCaps:
    return SessionManagerFactoryCaps(
        analyze_log_chunk=server._analyze_log_chunk,
        broker_interrupted_idle_from_state=server._broker_interrupted_idle_from_state,
        broker_tail_has_session_detach_marker=server._broker_tail_has_session_detach_marker,
        cc_pending_tool_ids_before=server._rollout_log._cc_pending_tool_ids_before,
        clean_alias=server._clean_alias,
        clean_dependency_session_id=server._clean_dependency_session_id,
        clean_optional_text=server._clean_optional_text,
        clean_priority_offset=server._clean_priority_offset,
        clean_recent_cwd=server._clean_recent_cwd,
        clean_snooze_until=server._clean_snooze_until,
        clean_unattended_cooldown_minutes=server._clean_unattended_cooldown_minutes,
        clean_unattended_remaining_injections=server._clean_unattended_remaining_injections,
        coerce_main_thread_log=server._coerce_main_thread_log,
        codex_trust_override_for_path=server._codex_trust_override_for_path,
        commit_unknown_error=server.SessionCommitUnknownError,
        commit_unknown_orphan_prune_seconds=server.COMMIT_UNKNOWN_ORPHAN_PRUNE_SECONDS,
        compute_idle_from_log=server._compute_idle_from_log,
        control_socket_call_error=server.ControlSocketCallError,
        create_git_worktree=server._create_git_worktree,
        current_git_branch=server._current_git_branch,
        dotenv_path=server._DOTENV,
        drain_stream=server._drain_stream,
        environ=server.os.environ,
        extract_delivery_messages=server._extract_delivery_messages,
        find_latest_token_update=server._rollout_log._find_latest_token_update,
        homes={"codex": server.CODEX_HOME, "pi": server.PI_HOME, "cc": server.CC_HOME},
        injection_error=server.SessionInjectionError,
        iter_session_logs=server._iter_session_logs,
        last_chat_role_ts_from_tail=server._last_chat_role_ts_from_tail,
        last_conversation_ts_from_tail=server._last_conversation_ts_from_tail,
        latest_launch_attempt=server._latest_launch_attempt,
        launch_attempt_row=server._launch_attempt_row,
        launch_attempts_path=server.LAUNCH_ATTEMPTS_PATH,
        launch_error=server.SessionLaunchError,
        launch_failure_tail=server._launch_failure_tail,
        list_resume_candidates_for_cwd=server._list_resume_candidates_for_cwd,
        load_env_file=server._load_env_file,
        log_invalid_sidecar_metadata=server._log_invalid_sidecar_metadata,
        max_scan_bytes=server.UNATTENDED_MAX_SCAN_BYTES,
        normalize_requested_service_tier=server._normalize_requested_service_tier,
        not_ready_error=server.SessionNotReadyError,
        now=server.time.time,
        pid_alive=server._pid_alive,
        popen=server.subprocess.Popen,
        priority_bucket_seconds=server.SIDEBAR_PRIORITY_BUCKET_SECONDS,
        priority_half_life_seconds=server.SIDEBAR_PRIORITY_HALF_LIFE_SECONDS,
        proc_find_open_rollout_log=server._proc_find_open_rollout_log,
        proc_root=server.PROC_ROOT,
        process_group_alive=server._process_group_alive,
        prompt_prefix=server.UNATTENDED_PROMPT_PREFIX,
        provider_choice_for_settings=server._provider_choice_for_settings,
        python_executable=server.sys.executable,
        queue_idle_grace_seconds=server.QUEUE_IDLE_GRACE_SECONDS,
        read_jsonl_from_offset=server._read_jsonl_from_offset,
        read_launch_attempts=server._read_launch_attempts,
        read_run_settings_from_log=server._read_run_settings_from_log,
        read_session_meta_or_none=server._read_session_meta_or_none,
        recent_cwd_max=server.RECENT_CWD_MAX,
        record_launch_attempt=server._record_launch_attempt,
        repo_root=server.Path(__file__).resolve().parent.parent,
        resolve_dir_target=server._resolve_dir_target,
        resolve_session_cwd=server._resolve_session_cwd,
        resume_candidate_from_log=server._resume_candidate_from_log,
        run=server.subprocess.run,
        send_commit_timeout_seconds=server.SEND_COMMIT_TIMEOUT_SECONDS,
        sock_error_definitely_stale=server._sock_error_definitely_stale,
        socket_timeout=server.socket.timeout,
        stderr=server.sys.stderr,
        submitted_user_messages=server._submitted_user_messages,
        terminate_process=server._terminate_process,
        terminate_process_group=server._terminate_process_group,
        tmux_pane_snapshot=server._tmux_pane_snapshot,
        tmux_session_name=server.TMUX_SESSION_NAME,
        turn_context_run_settings=server._turn_context_run_settings,
        unattended_default_idle_minutes=server.UNATTENDED_DEFAULT_IDLE_MINUTES,
        unattended_default_max_injections=server.UNATTENDED_DEFAULT_MAX_INJECTIONS,
        unlink_quiet=server._unlink_quiet,
        wait_for_spawned_broker_meta=server._wait_for_spawned_broker_meta,
        wait_or_raise=server._wait_or_raise,
        which_tmux=server.shutil.which,
    )


def discovery_deps_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> DiscoveryDeps:
    return DiscoveryDeps(
        pid_alive=caps.pid_alive,
        proc_find_open_rollout_log=lambda proc_root, root_pid, agent_backend, cwd, ignored_paths: caps.proc_find_open_rollout_log(
            proc_root=proc_root,
            root_pid=root_pid,
            agent_backend=agent_backend,
            cwd=cwd,
            ignored_paths=ignored_paths,
        ),
        read_session_meta_or_none=lambda log_path, agent_backend, context: caps.read_session_meta_or_none(
            log_path,
            agent_backend=agent_backend,
            context=context,
        ),
        coerce_main_thread_log=lambda thread_id, log_path: caps.coerce_main_thread_log(thread_id=thread_id, log_path=log_path),
        session_transport=lambda meta: manager._session_transport(meta=meta),
        session_run_settings=lambda meta, log_path, agent_backend: manager._session_run_settings(
            meta=meta,
            log_path=log_path,
            agent_backend=agent_backend,
        ),
        sock_call=lambda sock, req, timeout_s: manager._sock_call(sock, req, timeout_s=timeout_s),
        broker_busy_queue_from_state=manager._broker_busy_queue_from_state,
        broker_interrupted_idle_from_state=caps.broker_interrupted_idle_from_state,
        sock_error_definitely_stale=caps.sock_error_definitely_stale,
        token_update_finder=caps.find_latest_token_update,
    )


def queue_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionQueueCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        queues=lambda: manager._queues,
        queue_store=manager._queue_store_for_manager,
        commit_unknown_sends=lambda: manager._commit_unknown_sends,
        save_queues=manager._save_queues,
        input_lock_for_session=manager._input_lock_for_session,
        remote_ready=lambda session_id, log_path: manager._queue_remote_ready(session_id, log_path=log_path),
        send=manager.send,
        not_ready_error=caps.not_ready_error,
        retryable_send_errors=(caps.not_ready_error, caps.injection_error),
        commit_unknown_error=caps.commit_unknown_error,
        queue_idle_grace_seconds=caps.queue_idle_grace_seconds,
        now=caps.now,
        recovery_items_locked=lambda session_id: manager._queue_has_recovery_items_locked(session_id),
    )


def control_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionControlCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        sock_call=lambda sock, req, **kwargs: manager._sock_call(sock, req, **kwargs),
        pid_alive=caps.pid_alive,
        unlink_quiet=caps.unlink_quiet,
        clear_deleted_session_state=manager._clear_deleted_session_state,
        broker_busy_queue=manager._broker_busy_queue_from_state,
        broker_interrupted_idle=caps.broker_interrupted_idle_from_state,
        control_socket_call_error=caps.control_socket_call_error,
        commit_unknown_error=caps.commit_unknown_error,
    )


def attachment_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionAttachmentCoordinator(
        input_lock_for_session=manager._input_lock_for_session,
        attachment_injection_ready=manager.attachment_injection_ready,
        inject_keys=manager.inject_keys,
        set_pending_attachment=manager._set_pending_attachment,
        not_ready_error=caps.not_ready_error,
        injection_error=caps.injection_error,
        commit_unknown_error=caps.commit_unknown_error,
    )


def list_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionListCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        queues=lambda: manager._queues,
        unattended=lambda: manager._unattended,
        aliases=lambda: manager._aliases,
        hidden_sessions=lambda: manager._hidden_sessions,
        commit_unknown_sends=lambda: manager._commit_unknown_sends,
        store=manager._session_store_for_manager(),
        discover_existing_if_stale=manager._discover_existing_if_stale,
        prune_dead_sessions=manager._prune_dead_sessions,
        update_meta_counters=manager._update_meta_counters,
        save_files=manager._save_files,
        save_sidebar_meta=manager._save_sidebar_meta,
        save_recent_cwds=manager._save_recent_cwds,
        now=caps.now,
        runtime_probes=ListingRuntimeProbes(
            last_conversation_ts_from_tail=lambda path: caps.last_conversation_ts_from_tail(path),
            read_run_settings_from_log=lambda path, agent_backend: caps.read_run_settings_from_log(path, agent_backend=agent_backend),
            log_size_or_none=manager._log_size_or_none,
            send_boundary_unresolved=manager._confirmed_send_boundary_unresolved_for_session,
            idle_from_log_path=manager.idle_from_log_path,
            current_git_branch=caps.current_git_branch,
        ),
        include_launch_attempts=lambda: bool(getattr(manager, "_include_launch_attempts", False)),
        read_launch_attempts=lambda: caps.read_launch_attempts(path=caps.launch_attempts_path, max_records=100, max_age_s=24 * 3600),
        launch_attempt_row=caps.launch_attempt_row,
        clean_unattended_cooldown_minutes=caps.clean_unattended_cooldown_minutes,
        clean_unattended_remaining_injections=caps.clean_unattended_remaining_injections,
        provider_choice_for_settings=caps.provider_choice_for_settings,
        resolve_session_cwd=caps.resolve_session_cwd,
        unattended_default_idle_minutes=caps.unattended_default_idle_minutes,
        unattended_default_max_injections=caps.unattended_default_max_injections,
        priority_half_life_seconds=caps.priority_half_life_seconds,
        priority_bucket_seconds=caps.priority_bucket_seconds,
    )


def refresh_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionRefreshCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        prune_stale_socket_without_metadata=manager._prune_stale_socket_without_metadata,
        log_invalid_sidecar_metadata=caps.log_invalid_sidecar_metadata,
        session_transport=manager._session_transport,
        sock_call=lambda sock, req, **kwargs: manager._sock_call(sock, req, **kwargs),
        broker_tail_has_session_detach_marker=caps.broker_tail_has_session_detach_marker,
        pid_alive=caps.pid_alive,
        proc_find_open_rollout_log=caps.proc_find_open_rollout_log,
        proc_root=caps.proc_root,
        read_session_meta_or_none=caps.read_session_meta_or_none,
        coerce_main_thread_log=caps.coerce_main_thread_log,
        clean_optional_text=caps.clean_optional_text,
        session_run_settings=manager._session_run_settings,
        normalize_requested_service_tier=caps.normalize_requested_service_tier,
        reset_log_caches=lambda session, log_off: manager._reset_log_caches(session, meta_log_off=log_off),
        queue_len=manager._queue_len,
        maybe_drain_session_queue=manager._maybe_drain_session_queue,
    )


def readiness_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionReadinessCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        refresh_session_meta_if_sidecar_exists=manager._refresh_session_meta_if_sidecar_exists,
        get_state=manager.get_state,
        log_size_or_none=manager._log_size_or_none,
        confirmed_send_boundary_unresolved_for_session=manager._confirmed_send_boundary_unresolved_for_session,
        idle_from_log=manager.idle_from_log,
        queue_len=lambda session_id: manager._queue_store_for_manager().queue_len(manager._queues, session_id),
        not_ready_error=caps.not_ready_error,
    )


def unattended_sweep_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return UnattendedSweepCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        unattended=lambda: manager._unattended,
        unattended_last_injected=lambda: manager._unattended_last_injected,
        unattended_last_injected_scope=lambda: manager._unattended_last_injected_scope,
        discover_existing_if_stale=manager._discover_existing_if_stale,
        prune_dead_sessions=manager._prune_dead_sessions,
        input_lock_for_session=manager._input_lock_for_session,
        save_unattended=manager._save_unattended,
        get_state=manager.get_state,
        broker_busy_queue_from_state=manager._broker_busy_queue_from_state,
        queue_len=manager._queue_len,
        last_chat_role_ts_from_tail=caps.last_chat_role_ts_from_tail,
        send=manager.send,
        now=caps.now,
        prompt_prefix=caps.prompt_prefix,
        default_idle_minutes=caps.unattended_default_idle_minutes,
        default_max_injections=caps.unattended_default_max_injections,
        max_scan_bytes=caps.max_scan_bytes,
    )


def queue_sweep_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return QueueSweepCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        queues=lambda: manager._queues,
        commit_unknown_sends=lambda: manager._commit_unknown_sends,
        queue_store=manager._queue_store_for_manager(),
        discover_existing_if_stale=manager._discover_existing_if_stale,
        prune_dead_sessions=manager._prune_dead_sessions,
        mark_queue_orphan_recovery_locked=manager._mark_queue_orphan_recovery_locked,
        save_queues=manager._save_queues,
        maybe_drain_session_queue=manager._maybe_drain_session_queue,
    )


def voice_runtime_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return VoiceRuntimeCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        aliases=lambda: manager._aliases,
        voice_push=lambda: getattr(manager, "_voice_push", None),
        discover_existing_if_stale=manager._discover_existing_if_stale,
        prune_dead_sessions=manager._prune_dead_sessions,
        refresh_session_meta=lambda session_id: manager.refresh_session_meta(session_id),
        read_jsonl_from_offset=caps.read_jsonl_from_offset,
        extract_delivery_messages=lambda objs, **kwargs: caps.extract_delivery_messages(objs, **kwargs),
        cc_pending_tool_ids_before=caps.cc_pending_tool_ids_before,
    )


def log_runtime_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionLogRuntimeCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        analyze_log_chunk=caps.analyze_log_chunk,
        turn_context_run_settings=caps.turn_context_run_settings,
        compute_idle_from_log=caps.compute_idle_from_log,
        read_jsonl_from_offset=caps.read_jsonl_from_offset,
        find_latest_token_update=caps.find_latest_token_update,
    )


def files_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionFilesCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        store=manager._session_store_for_manager(),
        save_files=manager._save_files,
    )


def ui_state_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionUiStateCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        aliases=lambda: manager._aliases,
        set_aliases=lambda value: setattr(manager, "_aliases", value),
        sidebar_meta=lambda: manager._sidebar_meta,
        set_sidebar_meta=lambda value: setattr(manager, "_sidebar_meta", value),
        hidden_sessions=lambda: manager._hidden_sessions,
        set_hidden_sessions=lambda value: setattr(manager, "_hidden_sessions", value),
        save_aliases=manager._save_aliases,
        save_sidebar_meta=manager._save_sidebar_meta,
        save_hidden_sessions=manager._save_hidden_sessions,
        clean_alias=caps.clean_alias,
        clean_priority_offset=caps.clean_priority_offset,
        clean_snooze_until=caps.clean_snooze_until,
        clean_dependency_session_id=caps.clean_dependency_session_id,
    )


def unattended_config_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionUnattendedConfigCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        unattended=lambda: manager._unattended,
        unattended_last_injected=lambda: manager._unattended_last_injected,
        input_lock_for_session=manager._input_lock_for_session,
        save_unattended=manager._save_unattended,
        clean_unattended_cooldown_minutes=caps.clean_unattended_cooldown_minutes,
        clean_unattended_remaining_injections=caps.clean_unattended_remaining_injections,
    )


def cleanup_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionCleanupCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        aliases=lambda: manager._aliases,
        sidebar_meta=lambda: manager._sidebar_meta,
        unattended=lambda: manager._unattended,
        files=lambda: manager._files,
        queues=lambda: manager._queues,
        commit_unknown_sends=lambda: manager._commit_unknown_sends,
        input_locks=lambda: getattr(manager, "_input_locks", {}),
        pending_attachment_ids=lambda: getattr(manager, "_pending_attachment_ids", set()),
        unhide_session=manager._unhide_session,
        mark_queue_orphan_recovery_locked=manager._mark_queue_orphan_recovery_locked,
        unlink_quiet=caps.unlink_quiet,
        save_pending_attachments=manager._save_pending_attachments,
        save_commit_unknown_sends=manager._save_commit_unknown_sends,
        save_aliases=manager._save_aliases,
        save_sidebar_meta=manager._save_sidebar_meta,
        save_unattended=manager._save_unattended,
        save_files=manager._save_files,
        save_queues=manager._save_queues,
    )


def pending_state_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionPendingStateCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        pending_attachment_ids=lambda: getattr(manager, "_pending_attachment_ids", None),
        set_pending_attachment_ids=lambda value: setattr(manager, "_pending_attachment_ids", value),
        commit_unknown_sends=lambda: getattr(manager, "_commit_unknown_sends", None),
        set_commit_unknown_sends=lambda value: setattr(manager, "_commit_unknown_sends", value),
        mark_queue_orphan_recovery_locked=manager._mark_queue_orphan_recovery_locked,
        save_pending_attachments=manager._save_pending_attachments,
        save_commit_unknown_sends=manager._save_commit_unknown_sends,
        save_queues=manager._save_queues,
        now=caps.now,
        commit_unknown_orphan_prune_seconds=caps.commit_unknown_orphan_prune_seconds,
    )


def recent_cwd_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionRecentCwdCoordinator(
        lock=manager._lock,
        recent_cwds=lambda: getattr(manager, "_recent_cwds", None),
        set_recent_cwds=lambda value: setattr(manager, "_recent_cwds", value),
        clean_recent_cwd=caps.clean_recent_cwd,
        iter_session_logs=caps.iter_session_logs,
        resume_candidate_from_log=caps.resume_candidate_from_log,
        save_recent_cwds=manager._save_recent_cwds,
        now=caps.now,
        max_recent_cwds=caps.recent_cwd_max,
    )


def lifecycle_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionLifecycleCoordinator(
        lock=getattr(manager, "_lock", threading.RLock()),
        sessions=lambda: getattr(manager, "_sessions", {}),
        sock_call=lambda sock, req, **kwargs: manager._sock_call(sock, req, **kwargs),
        process_group_alive=caps.process_group_alive,
        pid_alive=caps.pid_alive,
        terminate_process_group=caps.terminate_process_group,
        terminate_process=caps.terminate_process,
        unlink_quiet=caps.unlink_quiet,
        commit_unknown_sends=lambda: getattr(manager, "_commit_unknown_sends", {}),
        queue_has_recovery_items_locked=manager._queue_has_recovery_items_locked,
        clear_deleted_session_state=manager._clear_deleted_session_state,
        read_launch_attempts=lambda: caps.read_launch_attempts(path=caps.launch_attempts_path, max_records=100, max_age_s=24 * 3600),
        launch_attempt_row=caps.launch_attempt_row,
        hide_session=manager._hide_session,
        files_clear=manager.files_clear,
        clean_optional_text=caps.clean_optional_text,
        kill_session_via_pids_fallback=manager._kill_session_via_pids,
    )


def discovery_registry_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionDiscoveryRegistryCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        pending_attachment_ids=lambda: getattr(manager, "_pending_attachment_ids", set()),
        commit_unknown_sends=lambda: getattr(manager, "_commit_unknown_sends", {}),
        reset_log_caches=lambda session, log_off: manager._reset_log_caches(session, meta_log_off=log_off),
        record_launch_attempt=caps.record_launch_attempt,
        prune_stale_socket_without_metadata=manager._prune_stale_socket_without_metadata,
        unhide_session=manager._unhide_session,
        unlink_quiet=caps.unlink_quiet,
        remember_recent_cwd=manager._remember_recent_cwd,
        save_recent_cwds=manager._save_recent_cwds,
        stderr=caps.stderr,
    )


def prune_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionPruneCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        sock_call=lambda sock, req, **kwargs: manager._sock_call(sock, req, **kwargs),
        broker_busy_queue_from_state=manager._broker_busy_queue_from_state,
        broker_interrupted_idle_from_state=caps.broker_interrupted_idle_from_state,
        sock_error_definitely_stale=caps.sock_error_definitely_stale,
        pid_alive=caps.pid_alive,
        latest_launch_attempt=caps.latest_launch_attempt,
        submitted_user_messages=caps.submitted_user_messages,
        launch_failure_tail=lambda record: caps.launch_failure_tail(record or {}),
        which_tmux=caps.which_tmux,
        tmux_pane_snapshot=caps.tmux_pane_snapshot,
        clean_optional_text=caps.clean_optional_text,
        record_launch_attempt=caps.record_launch_attempt,
        clear_deleted_session_state=manager._clear_deleted_session_state,
        unlink_quiet=caps.unlink_quiet,
        stderr=caps.stderr,
    )


def send_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionSendCoordinator(
        lock=manager._lock,
        sessions=lambda: manager._sessions,
        input_lock_for_session=manager._input_lock_for_session,
        queue_len=lambda session_id: manager._queue_store_for_manager().queue_len(getattr(manager, "_queues", {}), session_id),
        send_remote_ready=manager._send_remote_ready,
        log_size_or_none=manager._log_size_or_none,
        call_confirmed_send=lambda session_id, **kwargs: manager._control_coordinator_for_manager().call_confirmed_send(session_id, **kwargs),
        set_pending_attachment=manager._set_pending_attachment,
        set_commit_unknown_send=manager._set_commit_unknown_send,
        record_prelog_user_message=lambda session, text: manager._record_prelog_user_message(session, text, source="send"),
        now=caps.now,
        send_commit_timeout_seconds=caps.send_commit_timeout_seconds,
        not_ready_error=caps.not_ready_error,
        commit_unknown_error=caps.commit_unknown_error,
        injection_error=caps.injection_error,
        timeout_errors=(TimeoutError, caps.socket_timeout),
    )


def prelog_user_message_recorder_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return PrelogUserMessageRecorder(
        latest_launch_attempt=caps.latest_launch_attempt,
        submitted_user_messages=caps.submitted_user_messages,
        clean_optional_text=caps.clean_optional_text,
        record_launch_attempt=caps.record_launch_attempt,
        now=caps.now,
    )


def web_launch_coordinator_for_manager(manager: Any, caps: SessionManagerFactoryCaps) -> Any:
    return SessionWebLaunchCoordinator(
        resolve_dir_target=caps.resolve_dir_target,
        create_git_worktree=caps.create_git_worktree,
        codex_trust_override_for_path=caps.codex_trust_override_for_path,
        list_resume_candidates_for_cwd=caps.list_resume_candidates_for_cwd,
        live_session_for_resume_target=manager._live_session_for_resume_target,
        load_env_file=caps.load_env_file,
        environ=caps.environ,
        dotenv_path=caps.dotenv_path,
        homes={"codex": caps.homes["codex"], "pi": caps.homes["pi"], "cc": caps.homes["cc"]},
        python_executable=caps.python_executable,
        tmux_session_name=caps.tmux_session_name,
        repo_root=caps.repo_root,
        record_launch_attempt=caps.record_launch_attempt,
        now=caps.now,
        stderr=caps.stderr,
        which_tmux=caps.which_tmux,
        run=caps.run,
        popen=caps.popen,
        wait_or_raise=caps.wait_or_raise,
        wait_for_spawned_broker_meta=caps.wait_for_spawned_broker_meta,
        tmux_pane_snapshot=caps.tmux_pane_snapshot,
        drain_stream=caps.drain_stream,
        launch_error=caps.launch_error,
    )
