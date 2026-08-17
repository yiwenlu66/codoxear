from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .broker_watchdog import BrokerWatchdogCoordinator
from .queue_sweep import QueueSweepCoordinator
from .session_discovery import DiscoveryDeps
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
from .session_registry import session_registry_for_manager
from .session_runtime import ListingRuntimeProbes
from .session_send import PrelogUserMessageRecorder
from .session_send import SessionSendCoordinator
from .session_ui_state import SessionUiStateCoordinator
from .session_unattended_config import SessionUnattendedConfigCoordinator
from .session_web_launch import SessionWebLaunchCoordinator
from .unattended_sweep import UnattendedSweepCoordinator
from .voice_runtime import VoiceRuntimeCoordinator


def _registry_lock(manager: Any) -> Any:
    return session_registry_for_manager(manager).lock


def _registry_sessions(manager: Any) -> dict[str, Any]:
    return session_registry_for_manager(manager).sessions


def _registry_input_locks(manager: Any) -> dict[str, Any]:
    return session_registry_for_manager(manager).input_locks


@dataclass(frozen=True)
class DiscoveryFactoryDeps:
    broker_interrupted_idle_from_state: Any
    coerce_main_thread_log: Any
    compute_idle_from_log: Any
    find_latest_token_update: Any
    pid_alive: Any
    proc_find_open_rollout_log: Any
    read_session_meta_or_none: Any
    sock_error_definitely_stale: Any


@dataclass(frozen=True)
class QueueFactoryDeps:
    commit_unknown_error: Any
    injection_error: Any
    not_ready_error: Any
    now: Any
    queue_idle_grace_seconds: float


@dataclass(frozen=True)
class ControlFactoryDeps:
    broker_interrupted_idle_from_state: Any
    commit_unknown_error: Any
    compute_idle_from_log: Any
    control_socket_call_error: Any
    latest_launch_attempt: Any
    pid_alive: Any
    record_launch_attempt: Any
    stderr: Any
    unlink_quiet: Any


@dataclass(frozen=True)
class ListFactoryDeps:
    clean_unattended_cooldown_minutes: Any
    clean_unattended_remaining_injections: Any
    current_git_branch: Any
    last_conversation_ts_from_tail: Any
    launch_attempt_row: Any
    launch_attempts_path: Any
    now: Any
    priority_bucket_seconds: float
    priority_half_life_seconds: float
    provider_choice_for_settings: Any
    read_launch_attempts: Any
    read_run_settings_from_log: Any
    resolve_session_cwd: Any
    unattended_default_idle_minutes: int
    unattended_default_max_injections: int


@dataclass(frozen=True)
class RefreshFactoryDeps:
    broker_tail_has_session_detach_marker: Any
    clean_optional_text: Any
    coerce_main_thread_log: Any
    log_invalid_sidecar_metadata: Any
    normalize_requested_service_tier: Any
    pid_alive: Any
    proc_find_open_rollout_log: Any
    proc_root: Any
    read_session_meta_or_none: Any


@dataclass(frozen=True)
class ReadinessFactoryDeps:
    not_ready_error: Any


@dataclass(frozen=True)
class UnattendedSweepFactoryDeps:
    last_chat_role_ts_from_tail: Any
    max_scan_bytes: int
    now: Any
    prompt_prefix: str | Callable[[], str]
    unattended_default_idle_minutes: int
    unattended_default_max_injections: int


@dataclass(frozen=True)
class QueueSweepFactoryDeps:
    queue_sweep_max_attempts: int
    queue_sweep_max_drains: int


@dataclass(frozen=True)
class VoiceRuntimeFactoryDeps:
    cc_pending_tool_ids_before: Any
    extract_delivery_messages: Any
    read_jsonl_from_offset: Any


@dataclass(frozen=True)
class LogRuntimeFactoryDeps:
    analyze_log_chunk: Any
    compute_idle_from_log: Any
    find_latest_token_update: Any
    read_jsonl_from_offset: Any
    turn_context_run_settings: Any


@dataclass(frozen=True)
class UiStateFactoryDeps:
    clean_alias: Any
    clean_dependency_session_id: Any
    clean_priority_offset: Any
    clean_snooze_until: Any


@dataclass(frozen=True)
class UnattendedConfigFactoryDeps:
    clean_unattended_cooldown_minutes: Any
    clean_unattended_remaining_injections: Any


@dataclass(frozen=True)
class CleanupFactoryDeps:
    unlink_quiet: Any


@dataclass(frozen=True)
class PendingStateFactoryDeps:
    commit_unknown_orphan_prune_seconds: float
    now: Any


@dataclass(frozen=True)
class RecentCwdFactoryDeps:
    iter_session_logs: Any
    now: Any
    resume_candidate_from_log: Any


@dataclass(frozen=True)
class LifecycleFactoryDeps:
    clean_optional_text: Any
    launch_attempt_row: Any
    launch_attempts_path: Any
    pid_alive: Any
    process_group_alive: Any
    read_launch_attempts: Any
    terminate_process: Any
    terminate_process_group: Any
    unlink_quiet: Any


@dataclass(frozen=True)
class DiscoveryRegistryFactoryDeps:
    record_launch_attempt: Any
    stderr: Any
    unlink_quiet: Any


@dataclass(frozen=True)
class PruneFactoryDeps:
    broker_interrupted_idle_from_state: Any
    clean_optional_text: Any
    compute_idle_from_log: Any
    latest_launch_attempt: Any
    launch_failure_tail: Any
    pid_alive: Any
    record_launch_attempt: Any
    sock_error_definitely_stale: Any
    stderr: Any
    submitted_user_messages: Any
    tmux_pane_snapshot: Any
    unlink_quiet: Any
    which_tmux: Any


@dataclass(frozen=True)
class BrokerWatchdogFactoryDeps:
    broker_watchdog_grace_seconds: float
    now: Any
    pid_alive: Any
    unlink_quiet: Any


@dataclass(frozen=True)
class SendFactoryDeps:
    commit_unknown_error: Any
    injection_error: Any
    not_ready_error: Any
    now: Any
    send_commit_timeout_seconds: float
    socket_timeout: Any


@dataclass(frozen=True)
class PrelogRecorderFactoryDeps:
    clean_optional_text: Any
    latest_launch_attempt: Any
    now: Any
    record_launch_attempt: Any
    submitted_user_messages: Any


@dataclass(frozen=True)
class WebLaunchFactoryDeps:
    codex_trust_override_for_path: Any
    create_git_worktree: Any
    dotenv_path: Any
    drain_stream: Any
    environ: Any
    homes: dict[str, Any]
    launch_error: Any
    list_resume_candidates_for_cwd: Any
    load_env_file: Any
    now: Any
    popen: Any
    python_executable: str
    record_launch_attempt: Any
    repo_root: Any
    resolve_dir_target: Any
    run: Any
    stderr: Any
    tmux_pane_snapshot: Any
    tmux_session_name: str
    wait_for_spawned_broker_meta: Any
    wait_or_raise: Any
    which_tmux: Any


@dataclass(frozen=True)
class SessionManagerCoordinatorDeps:
    discovery_deps: DiscoveryFactoryDeps
    queue_coordinator: QueueFactoryDeps
    control_coordinator: ControlFactoryDeps
    list_coordinator: ListFactoryDeps
    refresh_coordinator: RefreshFactoryDeps
    readiness_coordinator: ReadinessFactoryDeps
    unattended_sweep_coordinator: UnattendedSweepFactoryDeps
    queue_sweep_coordinator: QueueSweepFactoryDeps
    voice_runtime: VoiceRuntimeFactoryDeps
    log_runtime: LogRuntimeFactoryDeps
    ui_state_coordinator: UiStateFactoryDeps
    unattended_config_coordinator: UnattendedConfigFactoryDeps
    cleanup_coordinator: CleanupFactoryDeps
    pending_state_coordinator: PendingStateFactoryDeps
    recent_cwd_coordinator: RecentCwdFactoryDeps
    lifecycle_coordinator: LifecycleFactoryDeps
    discovery_registry: DiscoveryRegistryFactoryDeps
    prune_coordinator: PruneFactoryDeps
    broker_watchdog_coordinator: BrokerWatchdogFactoryDeps
    send_coordinator: SendFactoryDeps
    prelog_user_message_recorder: PrelogRecorderFactoryDeps
    web_launch_coordinator: WebLaunchFactoryDeps


def session_manager_coordinator_deps(server: Any) -> SessionManagerCoordinatorDeps:
    """Capture focused coordinator dependencies once at the composition root."""
    return SessionManagerCoordinatorDeps(
        discovery_deps=DiscoveryFactoryDeps(
            broker_interrupted_idle_from_state=server._broker_interrupted_idle_from_state,
            coerce_main_thread_log=server._coerce_main_thread_log,
            compute_idle_from_log=server._compute_idle_from_log,
            find_latest_token_update=server._rollout_log._find_latest_token_update,
            pid_alive=server._pid_alive,
            proc_find_open_rollout_log=server._proc_find_open_rollout_log,
            read_session_meta_or_none=server._read_session_meta_or_none,
            sock_error_definitely_stale=server._sock_error_definitely_stale,
        ),
        queue_coordinator=QueueFactoryDeps(
            commit_unknown_error=server.SessionCommitUnknownError,
            injection_error=server.SessionInjectionError,
            not_ready_error=server.SessionNotReadyError,
            now=server.time.time,
            queue_idle_grace_seconds=server.QUEUE_IDLE_GRACE_SECONDS,
        ),
        control_coordinator=ControlFactoryDeps(
            broker_interrupted_idle_from_state=server._broker_interrupted_idle_from_state,
            commit_unknown_error=server.SessionCommitUnknownError,
            compute_idle_from_log=server._compute_idle_from_log,
            control_socket_call_error=server.ControlSocketCallError,
            latest_launch_attempt=server._latest_launch_attempt,
            pid_alive=server._pid_alive,
            record_launch_attempt=server._record_launch_attempt,
            stderr=server.sys.stderr,
            unlink_quiet=server._unlink_quiet,
        ),
        list_coordinator=ListFactoryDeps(
            clean_unattended_cooldown_minutes=server._clean_unattended_cooldown_minutes,
            clean_unattended_remaining_injections=server._clean_unattended_remaining_injections,
            current_git_branch=server._current_git_branch,
            last_conversation_ts_from_tail=server._last_conversation_ts_from_tail,
            launch_attempt_row=server._launch_attempt_row,
            launch_attempts_path=server.LAUNCH_ATTEMPTS_PATH,
            now=server.time.time,
            priority_bucket_seconds=server.SIDEBAR_PRIORITY_BUCKET_SECONDS,
            priority_half_life_seconds=server.SIDEBAR_PRIORITY_HALF_LIFE_SECONDS,
            provider_choice_for_settings=server._provider_choice_for_settings,
            read_launch_attempts=server._read_launch_attempts,
            read_run_settings_from_log=server._read_run_settings_from_log,
            resolve_session_cwd=server._resolve_session_cwd,
            unattended_default_idle_minutes=server.UNATTENDED_DEFAULT_IDLE_MINUTES,
            unattended_default_max_injections=server.UNATTENDED_DEFAULT_MAX_INJECTIONS,
        ),
        refresh_coordinator=RefreshFactoryDeps(
            broker_tail_has_session_detach_marker=server._broker_tail_has_session_detach_marker,
            clean_optional_text=server._clean_optional_text,
            coerce_main_thread_log=server._coerce_main_thread_log,
            log_invalid_sidecar_metadata=server._log_invalid_sidecar_metadata,
            normalize_requested_service_tier=server._normalize_requested_service_tier,
            pid_alive=server._pid_alive,
            proc_find_open_rollout_log=server._proc_find_open_rollout_log,
            proc_root=server.PROC_ROOT,
            read_session_meta_or_none=server._read_session_meta_or_none,
        ),
        readiness_coordinator=ReadinessFactoryDeps(
            not_ready_error=server.SessionNotReadyError,
        ),
        unattended_sweep_coordinator=UnattendedSweepFactoryDeps(
            last_chat_role_ts_from_tail=server._last_chat_role_ts_from_tail,
            max_scan_bytes=server.UNATTENDED_MAX_SCAN_BYTES,
            now=server.time.time,
            prompt_prefix=lambda: server._load_unattended_prompt(server.UNATTENDED_PROMPT_PATH),
            unattended_default_idle_minutes=server.UNATTENDED_DEFAULT_IDLE_MINUTES,
            unattended_default_max_injections=server.UNATTENDED_DEFAULT_MAX_INJECTIONS,
        ),
        queue_sweep_coordinator=QueueSweepFactoryDeps(
            queue_sweep_max_attempts=server.QUEUE_SWEEP_MAX_ATTEMPTS,
            queue_sweep_max_drains=server.QUEUE_SWEEP_MAX_DRAINS,
        ),
        voice_runtime=VoiceRuntimeFactoryDeps(
            cc_pending_tool_ids_before=server._rollout_log._cc_pending_tool_ids_before,
            extract_delivery_messages=server._extract_delivery_messages,
            read_jsonl_from_offset=server._read_jsonl_from_offset,
        ),
        log_runtime=LogRuntimeFactoryDeps(
            analyze_log_chunk=server._analyze_log_chunk,
            compute_idle_from_log=server._compute_idle_from_log,
            find_latest_token_update=server._rollout_log._find_latest_token_update,
            read_jsonl_from_offset=server._read_jsonl_from_offset,
            turn_context_run_settings=server._turn_context_run_settings,
        ),
        ui_state_coordinator=UiStateFactoryDeps(
            clean_alias=server._clean_alias,
            clean_dependency_session_id=server._clean_dependency_session_id,
            clean_priority_offset=server._clean_priority_offset,
            clean_snooze_until=server._clean_snooze_until,
        ),
        unattended_config_coordinator=UnattendedConfigFactoryDeps(
            clean_unattended_cooldown_minutes=server._clean_unattended_cooldown_minutes,
            clean_unattended_remaining_injections=server._clean_unattended_remaining_injections,
        ),
        cleanup_coordinator=CleanupFactoryDeps(
            unlink_quiet=server._unlink_quiet,
        ),
        pending_state_coordinator=PendingStateFactoryDeps(
            commit_unknown_orphan_prune_seconds=server.COMMIT_UNKNOWN_ORPHAN_PRUNE_SECONDS,
            now=server.time.time,
        ),
        recent_cwd_coordinator=RecentCwdFactoryDeps(
            iter_session_logs=server._iter_session_logs,
            now=server.time.time,
            resume_candidate_from_log=server._resume_candidate_from_log,
        ),
        lifecycle_coordinator=LifecycleFactoryDeps(
            clean_optional_text=server._clean_optional_text,
            launch_attempt_row=server._launch_attempt_row,
            launch_attempts_path=server.LAUNCH_ATTEMPTS_PATH,
            pid_alive=server._pid_alive,
            process_group_alive=server._process_group_alive,
            read_launch_attempts=server._read_launch_attempts,
            terminate_process=server._terminate_process,
            terminate_process_group=server._terminate_process_group,
            unlink_quiet=server._unlink_quiet,
        ),
        discovery_registry=DiscoveryRegistryFactoryDeps(
            record_launch_attempt=server._record_launch_attempt,
            stderr=server.sys.stderr,
            unlink_quiet=server._unlink_quiet,
        ),
        prune_coordinator=PruneFactoryDeps(
            broker_interrupted_idle_from_state=server._broker_interrupted_idle_from_state,
            clean_optional_text=server._clean_optional_text,
            compute_idle_from_log=server._compute_idle_from_log,
            latest_launch_attempt=server._latest_launch_attempt,
            launch_failure_tail=server._launch_failure_tail,
            pid_alive=server._pid_alive,
            record_launch_attempt=server._record_launch_attempt,
            sock_error_definitely_stale=server._sock_error_definitely_stale,
            stderr=server.sys.stderr,
            submitted_user_messages=server._submitted_user_messages,
            tmux_pane_snapshot=server._tmux_pane_snapshot,
            unlink_quiet=server._unlink_quiet,
            which_tmux=server.shutil.which,
        ),
        broker_watchdog_coordinator=BrokerWatchdogFactoryDeps(
            broker_watchdog_grace_seconds=server.BROKER_WATCHDOG_GRACE_SECONDS,
            now=server.time.time,
            pid_alive=server._pid_alive,
            unlink_quiet=server._unlink_quiet,
        ),
        send_coordinator=SendFactoryDeps(
            commit_unknown_error=server.SessionCommitUnknownError,
            injection_error=server.SessionInjectionError,
            not_ready_error=server.SessionNotReadyError,
            now=server.time.time,
            send_commit_timeout_seconds=server.SEND_COMMIT_TIMEOUT_SECONDS,
            socket_timeout=server.socket.timeout,
        ),
        prelog_user_message_recorder=PrelogRecorderFactoryDeps(
            clean_optional_text=server._clean_optional_text,
            latest_launch_attempt=server._latest_launch_attempt,
            now=server.time.time,
            record_launch_attempt=server._record_launch_attempt,
            submitted_user_messages=server._submitted_user_messages,
        ),
        web_launch_coordinator=WebLaunchFactoryDeps(
            codex_trust_override_for_path=server._codex_trust_override_for_path,
            create_git_worktree=server._create_git_worktree,
            dotenv_path=server._DOTENV,
            drain_stream=server._drain_stream,
            environ=server.os.environ,
            homes={'codex': server.CODEX_HOME, 'pi': server.PI_HOME, 'cc': server.CC_HOME},
            launch_error=server.SessionLaunchError,
            list_resume_candidates_for_cwd=server._list_resume_candidates_for_cwd,
            load_env_file=server._load_env_file,
            now=server.time.time,
            popen=server.subprocess.Popen,
            python_executable=server.sys.executable,
            record_launch_attempt=server._record_launch_attempt,
            repo_root=server.Path(__file__).resolve().parent.parent,
            resolve_dir_target=server._resolve_dir_target,
            run=server.subprocess.run,
            stderr=server.sys.stderr,
            tmux_pane_snapshot=server._tmux_pane_snapshot,
            tmux_session_name=server.TMUX_SESSION_NAME,
            wait_for_spawned_broker_meta=server._wait_for_spawned_broker_meta,
            wait_or_raise=server._wait_or_raise,
            which_tmux=server.shutil.which,
        ),
    )




def discovery_deps_for_manager(manager: Any, deps: DiscoveryFactoryDeps) -> DiscoveryDeps:
    return DiscoveryDeps(
        pid_alive=deps.pid_alive,
        proc_find_open_rollout_log=lambda proc_root, root_pid, agent_backend, cwd, ignored_paths: deps.proc_find_open_rollout_log(
            proc_root=proc_root,
            root_pid=root_pid,
            agent_backend=agent_backend,
            cwd=cwd,
            ignored_paths=ignored_paths,
        ),
        read_session_meta_or_none=lambda log_path, agent_backend, context: deps.read_session_meta_or_none(
            log_path,
            agent_backend=agent_backend,
            context=context,
        ),
        coerce_main_thread_log=lambda thread_id, log_path: deps.coerce_main_thread_log(thread_id=thread_id, log_path=log_path),
        session_transport=lambda meta: manager._session_transport(meta=meta),
        session_run_settings=lambda meta, log_path, agent_backend: manager._session_run_settings(
            meta=meta,
            log_path=log_path,
            agent_backend=agent_backend,
        ),
        sock_call=lambda sock, req, timeout_s: manager._sock_call(sock, req, timeout_s=timeout_s),
        broker_busy_queue_from_state=manager._broker_busy_queue_from_state,
        broker_interrupted_idle_from_state=deps.broker_interrupted_idle_from_state,
        sock_error_definitely_stale=deps.sock_error_definitely_stale,
        token_update_finder=deps.find_latest_token_update,
        compute_idle_from_log=deps.compute_idle_from_log,
    )


def queue_coordinator_for_manager(manager: Any, deps: QueueFactoryDeps) -> Any:
    return SessionQueueCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        queues=lambda: manager._queues,
        queue_store=manager._queue_store_for_manager,
        commit_unknown_sends=lambda: manager._commit_unknown_sends,
        save_queues=manager._save_queues,
        input_lock_for_session=manager._input_lock_for_session,
        remote_ready=lambda session_id, log_path: manager._queue_remote_ready(session_id, log_path=log_path),
        send=manager.send,
        not_ready_error=deps.not_ready_error,
        retryable_send_errors=(deps.not_ready_error, deps.injection_error),
        commit_unknown_error=deps.commit_unknown_error,
        queue_idle_grace_seconds=deps.queue_idle_grace_seconds,
        now=deps.now,
        recovery_items_locked=lambda session_id: manager._queue_has_recovery_items_locked(session_id),
    )


def control_coordinator_for_manager(manager: Any, deps: ControlFactoryDeps) -> Any:
    return SessionControlCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        sock_call=lambda sock, req, **kwargs: manager._sock_call(sock, req, **kwargs),
        pid_alive=deps.pid_alive,
        unlink_quiet=deps.unlink_quiet,
        clear_deleted_session_state=manager._clear_deleted_session_state,
        broker_busy_queue=manager._broker_busy_queue_from_state,
        broker_interrupted_idle=deps.broker_interrupted_idle_from_state,
        control_socket_call_error=deps.control_socket_call_error,
        commit_unknown_error=deps.commit_unknown_error,
        latest_launch_attempt=deps.latest_launch_attempt,
        record_launch_attempt=deps.record_launch_attempt,
        compute_idle_from_log=deps.compute_idle_from_log,
        stderr=deps.stderr,
    )



def list_coordinator_for_manager(manager: Any, deps: ListFactoryDeps) -> Any:
    return SessionListCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
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
        now=deps.now,
        runtime_probes=ListingRuntimeProbes(
            last_conversation_ts_from_tail=lambda path: deps.last_conversation_ts_from_tail(path),
            read_run_settings_from_log=lambda path, agent_backend: deps.read_run_settings_from_log(path, agent_backend=agent_backend),
            commit_log_observation=manager.commit_log_observation,
            log_size_or_none=manager._log_size_or_none,
            send_boundary_unresolved=manager._confirmed_send_boundary_unresolved_for_session,
            idle_from_log_path=manager.idle_from_log_path,
            current_git_branch=deps.current_git_branch,
        ),
        include_launch_attempts=lambda: bool(getattr(manager, "_include_launch_attempts", False)),
        read_launch_attempts=lambda: deps.read_launch_attempts(path=deps.launch_attempts_path, max_records=100, max_age_s=24 * 3600),
        launch_attempt_row=deps.launch_attempt_row,
        clean_unattended_cooldown_minutes=deps.clean_unattended_cooldown_minutes,
        clean_unattended_remaining_injections=deps.clean_unattended_remaining_injections,
        provider_choice_for_settings=deps.provider_choice_for_settings,
        resolve_session_cwd=deps.resolve_session_cwd,
        unattended_default_idle_minutes=deps.unattended_default_idle_minutes,
        unattended_default_max_injections=deps.unattended_default_max_injections,
        priority_half_life_seconds=deps.priority_half_life_seconds,
        priority_bucket_seconds=deps.priority_bucket_seconds,
    )


def refresh_coordinator_for_manager(manager: Any, deps: RefreshFactoryDeps) -> Any:
    return SessionRefreshCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        prune_stale_socket_without_metadata=manager._prune_stale_socket_without_metadata,
        log_invalid_sidecar_metadata=deps.log_invalid_sidecar_metadata,
        session_transport=manager._session_transport,
        sock_call=lambda sock, req, **kwargs: manager._sock_call(sock, req, **kwargs),
        broker_tail_has_session_detach_marker=deps.broker_tail_has_session_detach_marker,
        pid_alive=deps.pid_alive,
        proc_find_open_rollout_log=deps.proc_find_open_rollout_log,
        proc_root=deps.proc_root,
        read_session_meta_or_none=deps.read_session_meta_or_none,
        coerce_main_thread_log=deps.coerce_main_thread_log,
        clean_optional_text=deps.clean_optional_text,
        session_run_settings=manager._session_run_settings,
        normalize_requested_service_tier=deps.normalize_requested_service_tier,
        reset_log_caches=lambda session, log_off: manager._reset_log_caches(session, meta_log_off=log_off),
        commit_log_observation=manager.commit_log_observation,
        queue_len=manager._queue_len,
        maybe_drain_session_queue=manager._maybe_drain_session_queue,
    )


def readiness_coordinator_for_manager(manager: Any, deps: ReadinessFactoryDeps) -> Any:
    return SessionReadinessCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        refresh_session_meta_if_sidecar_exists=manager._refresh_session_meta_if_sidecar_exists,
        get_state=manager.get_state,
        log_size_or_none=manager._log_size_or_none,
        confirmed_send_boundary_unresolved_for_session=manager._confirmed_send_boundary_unresolved_for_session,
        idle_from_log=manager.idle_from_log,
        queue_len=lambda session_id: manager._queue_store_for_manager().queue_len(manager._queues, session_id),
        not_ready_error=deps.not_ready_error,
    )


def unattended_sweep_coordinator_for_manager(manager: Any, deps: UnattendedSweepFactoryDeps) -> Any:
    return UnattendedSweepCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        unattended=lambda: manager._unattended,
        unattended_last_injected=lambda: manager._unattended_last_injected,
        unattended_last_injected_scope=lambda: manager._unattended_last_injected_scope,
        discover_existing_if_stale=manager._discover_existing_if_stale,
        prune_dead_sessions=manager._prune_dead_sessions,
        input_lock_for_session=manager._input_lock_for_session,
        save_unattended=manager._save_unattended,
        get_state=manager.get_state,
        runtime_status_from_state=manager._runtime_status_from_state_and_log,
        queue_len=manager._queue_len,
        last_chat_role_ts_from_tail=deps.last_chat_role_ts_from_tail,
        send=manager.send,
        now=deps.now,
        prompt_prefix=deps.prompt_prefix,
        default_idle_minutes=deps.unattended_default_idle_minutes,
        default_max_injections=deps.unattended_default_max_injections,
        max_scan_bytes=deps.max_scan_bytes,
    )


def queue_sweep_coordinator_for_manager(manager: Any, deps: QueueSweepFactoryDeps) -> Any:
    return QueueSweepCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        queues=lambda: manager._queues,
        commit_unknown_sends=lambda: manager._commit_unknown_sends,
        queue_store=manager._queue_store_for_manager(),
        discover_existing_if_stale=manager._discover_existing_if_stale,
        prune_dead_sessions=manager._prune_dead_sessions,
        mark_queue_orphan_recovery_locked=manager._mark_queue_orphan_recovery_locked,
        save_queues=manager._save_queues,
        maybe_drain_session_queue=manager._maybe_drain_session_queue,
        max_drains_per_sweep=deps.queue_sweep_max_drains,
        max_attempts_per_sweep=deps.queue_sweep_max_attempts,
    )


def voice_runtime_for_manager(manager: Any, deps: VoiceRuntimeFactoryDeps) -> Any:
    return VoiceRuntimeCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        aliases=lambda: manager._aliases,
        voice_push=lambda: getattr(manager, "_voice_push", None),
        discover_existing_if_stale=manager._discover_existing_if_stale,
        prune_dead_sessions=manager._prune_dead_sessions,
        refresh_session_meta=lambda session_id: manager.refresh_session_meta(session_id),
        read_jsonl_from_offset=deps.read_jsonl_from_offset,
        extract_delivery_messages=lambda objs, **kwargs: deps.extract_delivery_messages(objs, **kwargs),
        cc_pending_tool_ids_before=deps.cc_pending_tool_ids_before,
    )


def log_runtime_for_manager(manager: Any, deps: LogRuntimeFactoryDeps) -> Any:
    return SessionLogRuntimeCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        analyze_log_chunk=deps.analyze_log_chunk,
        turn_context_run_settings=deps.turn_context_run_settings,
        compute_idle_from_log=deps.compute_idle_from_log,
        read_jsonl_from_offset=deps.read_jsonl_from_offset,
        find_latest_token_update=deps.find_latest_token_update,
    )


def files_coordinator_for_manager(manager: Any) -> Any:
    return SessionFilesCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        store=manager._session_store_for_manager(),
        save_files=manager._save_files,
    )


def ui_state_coordinator_for_manager(manager: Any, deps: UiStateFactoryDeps) -> Any:
    return SessionUiStateCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        aliases=lambda: manager._aliases,
        set_aliases=lambda value: setattr(manager, "_aliases", value),
        sidebar_meta=lambda: manager._sidebar_meta,
        set_sidebar_meta=lambda value: setattr(manager, "_sidebar_meta", value),
        hidden_sessions=lambda: manager._hidden_sessions,
        set_hidden_sessions=lambda value: setattr(manager, "_hidden_sessions", value),
        save_aliases=manager._save_aliases,
        save_sidebar_meta=manager._save_sidebar_meta,
        save_hidden_sessions=manager._save_hidden_sessions,
        clean_alias=deps.clean_alias,
        clean_priority_offset=deps.clean_priority_offset,
        clean_snooze_until=deps.clean_snooze_until,
        clean_dependency_session_id=deps.clean_dependency_session_id,
    )


def unattended_config_coordinator_for_manager(manager: Any, deps: UnattendedConfigFactoryDeps) -> Any:
    return SessionUnattendedConfigCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        unattended=lambda: manager._unattended,
        unattended_last_injected=lambda: manager._unattended_last_injected,
        input_lock_for_session=manager._input_lock_for_session,
        save_unattended=manager._save_unattended,
        clean_unattended_cooldown_minutes=deps.clean_unattended_cooldown_minutes,
        clean_unattended_remaining_injections=deps.clean_unattended_remaining_injections,
    )


def cleanup_coordinator_for_manager(manager: Any, deps: CleanupFactoryDeps) -> Any:
    return SessionCleanupCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        store=manager._session_store_for_manager,
        input_locks=lambda: _registry_input_locks(manager),
        unlink_quiet=deps.unlink_quiet,
        save_pending_attachments=manager._save_pending_attachments,
        save_staged_attachments=manager._save_staged_attachments,
        save_commit_unknown_sends=manager._save_commit_unknown_sends,
        save_aliases=manager._save_aliases,
        save_sidebar_meta=manager._save_sidebar_meta,
        save_hidden_sessions=manager._save_hidden_sessions,
        save_unattended=manager._save_unattended,
        save_files=manager._save_files,
        save_queues=manager._save_queues,
        clear_unread=lambda session_id: getattr(manager, "_unread_store", None).clear(session_id) if getattr(manager, "_unread_store", None) is not None else None,
    )


def pending_state_coordinator_for_manager(manager: Any, deps: PendingStateFactoryDeps) -> Any:
    return SessionPendingStateCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        store=manager._session_store_for_manager,
        pending_attachment_ids=lambda: getattr(manager, "_pending_attachment_ids", None),
        set_pending_attachment_ids=lambda value: setattr(manager, "_pending_attachment_ids", value),
        commit_unknown_sends=lambda: getattr(manager, "_commit_unknown_sends", None),
        set_commit_unknown_sends=lambda value: setattr(manager, "_commit_unknown_sends", value),
        mark_queue_orphan_recovery_locked=manager._mark_queue_orphan_recovery_locked,
        save_pending_attachments=manager._save_pending_attachments,
        save_staged_attachments=manager._save_staged_attachments,
        save_commit_unknown_sends=manager._save_commit_unknown_sends,
        save_queues=manager._save_queues,
        now=deps.now,
        commit_unknown_orphan_prune_seconds=deps.commit_unknown_orphan_prune_seconds,
    )


def recent_cwd_coordinator_for_manager(manager: Any, deps: RecentCwdFactoryDeps) -> Any:
    return SessionRecentCwdCoordinator(
        lock=_registry_lock(manager),
        store=manager._session_store_for_manager,
        iter_session_logs=deps.iter_session_logs,
        resume_candidate_from_log=deps.resume_candidate_from_log,
        save_recent_cwds=manager._save_recent_cwds,
        now=deps.now,
    )


def lifecycle_coordinator_for_manager(manager: Any, deps: LifecycleFactoryDeps) -> Any:
    return SessionLifecycleCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        sock_call=lambda sock, req, **kwargs: manager._sock_call(sock, req, **kwargs),
        process_group_alive=deps.process_group_alive,
        pid_alive=deps.pid_alive,
        terminate_process_group=deps.terminate_process_group,
        terminate_process=deps.terminate_process,
        unlink_quiet=deps.unlink_quiet,
        commit_unknown_sends=lambda: getattr(manager, "_commit_unknown_sends", {}),
        queue_has_recovery_items_locked=manager._queue_has_recovery_items_locked,
        clear_deleted_session_state=manager._clear_deleted_session_state,
        read_launch_attempts=lambda: deps.read_launch_attempts(path=deps.launch_attempts_path, max_records=100, max_age_s=24 * 3600),
        launch_attempt_row=deps.launch_attempt_row,
        hide_session=manager._hide_session,
        clean_optional_text=deps.clean_optional_text,
        kill_session_via_pids_fallback=manager._kill_session_via_pids,
    )


def discovery_registry_for_manager(manager: Any, deps: DiscoveryRegistryFactoryDeps) -> Any:
    return SessionDiscoveryRegistryCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        pending_attachment_ids=lambda: getattr(manager, "_pending_attachment_ids", set()),
        commit_unknown_sends=lambda: getattr(manager, "_commit_unknown_sends", {}),
        reset_log_caches=lambda session, log_off: manager._reset_log_caches(session, meta_log_off=log_off),
        record_launch_attempt=deps.record_launch_attempt,
        prune_stale_socket_without_metadata=manager._prune_stale_socket_without_metadata,
        unhide_session=manager._unhide_session,
        unlink_quiet=deps.unlink_quiet,
        remember_recent_cwd=manager._remember_recent_cwd,
        save_recent_cwds=manager._save_recent_cwds,
        commit_log_observation=manager.commit_log_observation,
        stderr=deps.stderr,
    )


def prune_coordinator_for_manager(manager: Any, deps: PruneFactoryDeps) -> Any:
    return SessionPruneCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        sock_call=lambda sock, req, **kwargs: manager._sock_call(sock, req, **kwargs),
        broker_busy_queue_from_state=manager._broker_busy_queue_from_state,
        broker_interrupted_idle_from_state=deps.broker_interrupted_idle_from_state,
        sock_error_definitely_stale=deps.sock_error_definitely_stale,
        pid_alive=deps.pid_alive,
        latest_launch_attempt=deps.latest_launch_attempt,
        submitted_user_messages=deps.submitted_user_messages,
        launch_failure_tail=lambda record: deps.launch_failure_tail(record or {}),
        which_tmux=deps.which_tmux,
        tmux_pane_snapshot=deps.tmux_pane_snapshot,
        clean_optional_text=deps.clean_optional_text,
        record_launch_attempt=deps.record_launch_attempt,
        clear_deleted_session_state=manager._clear_deleted_session_state,
        unlink_quiet=deps.unlink_quiet,
        compute_idle_from_log=deps.compute_idle_from_log,
        stderr=deps.stderr,
    )


def broker_watchdog_coordinator_for_manager(manager: Any, deps: BrokerWatchdogFactoryDeps) -> Any:
    return BrokerWatchdogCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        pid_alive=deps.pid_alive,
        unlink_quiet=deps.unlink_quiet,
        now=deps.now,
        grace_seconds=deps.broker_watchdog_grace_seconds,
    )


def send_coordinator_for_manager(manager: Any, deps: SendFactoryDeps) -> Any:
    return SessionSendCoordinator(
        lock=_registry_lock(manager),
        sessions=lambda: _registry_sessions(manager),
        input_lock_for_session=manager._input_lock_for_session,
        queue_len=lambda session_id: manager._queue_store_for_manager().queue_len(getattr(manager, "_queues", {}), session_id),
        send_remote_ready=manager._send_remote_ready,
        refresh_session_meta_if_sidecar_exists=lambda session_id: manager._refresh_session_meta_if_sidecar_exists(session_id),
        log_size_or_none=manager._log_size_or_none,
        call_confirmed_send=lambda session_id, **kwargs: manager._control_coordinator_for_manager().call_confirmed_send(session_id, **kwargs),
        staged_attachments_for_session=lambda session_id: manager._session_store_for_manager().staged_attachments_for_session(session_id),
        clear_staged_attachments=manager.clear_staged_attachments,
        enqueue=manager.enqueue,
        set_pending_attachment=manager._set_pending_attachment,
        set_commit_unknown_send=manager._set_commit_unknown_send,
        record_prelog_user_message=lambda session, text: manager._record_prelog_user_message(session, text, source="send"),
        now=deps.now,
        send_commit_timeout_seconds=deps.send_commit_timeout_seconds,
        not_ready_error=deps.not_ready_error,
        commit_unknown_error=deps.commit_unknown_error,
        injection_error=deps.injection_error,
        timeout_errors=(TimeoutError, deps.socket_timeout),
    )


def prelog_user_message_recorder_for_manager(manager: Any, deps: PrelogRecorderFactoryDeps) -> Any:
    return PrelogUserMessageRecorder(
        latest_launch_attempt=deps.latest_launch_attempt,
        submitted_user_messages=deps.submitted_user_messages,
        clean_optional_text=deps.clean_optional_text,
        record_launch_attempt=deps.record_launch_attempt,
        now=deps.now,
    )


def web_launch_coordinator_for_manager(manager: Any, deps: WebLaunchFactoryDeps) -> Any:
    return SessionWebLaunchCoordinator(
        resolve_dir_target=deps.resolve_dir_target,
        create_git_worktree=deps.create_git_worktree,
        codex_trust_override_for_path=deps.codex_trust_override_for_path,
        list_resume_candidates_for_cwd=deps.list_resume_candidates_for_cwd,
        live_session_for_resume_target=manager._live_session_for_resume_target,
        load_env_file=deps.load_env_file,
        environ=deps.environ,
        dotenv_path=deps.dotenv_path,
        homes={"codex": deps.homes["codex"], "pi": deps.homes["pi"], "cc": deps.homes["cc"]},
        python_executable=deps.python_executable,
        tmux_session_name=deps.tmux_session_name,
        repo_root=deps.repo_root,
        record_launch_attempt=deps.record_launch_attempt,
        now=deps.now,
        stderr=deps.stderr,
        which_tmux=deps.which_tmux,
        run=deps.run,
        popen=deps.popen,
        wait_or_raise=deps.wait_or_raise,
        wait_for_spawned_broker_meta=deps.wait_for_spawned_broker_meta,
        tmux_pane_snapshot=deps.tmux_pane_snapshot,
        drain_stream=deps.drain_stream,
        launch_error=deps.launch_error,
    )


@dataclass(frozen=True)
class SessionManagerCoordinatorGraph:
    discovery: DiscoveryDeps
    queue: SessionQueueCoordinator
    control: SessionControlCoordinator
    listing: SessionListCoordinator
    refresh: SessionRefreshCoordinator
    readiness: SessionReadinessCoordinator
    unattended_sweep: UnattendedSweepCoordinator
    queue_sweep: QueueSweepCoordinator
    voice_runtime: VoiceRuntimeCoordinator
    log_runtime: SessionLogRuntimeCoordinator
    files: SessionFilesCoordinator
    ui_state: SessionUiStateCoordinator
    unattended_config: SessionUnattendedConfigCoordinator
    cleanup: SessionCleanupCoordinator
    pending_state: SessionPendingStateCoordinator
    recent_cwd: SessionRecentCwdCoordinator
    lifecycle: SessionLifecycleCoordinator
    discovery_registry: SessionDiscoveryRegistryCoordinator
    prune: SessionPruneCoordinator
    broker_watchdog: BrokerWatchdogCoordinator
    send: SessionSendCoordinator
    prelog_recorder: PrelogUserMessageRecorder
    web_launch: SessionWebLaunchCoordinator


def build_session_manager_coordinator_graph(
    manager: Any,
    deps: SessionManagerCoordinatorDeps,
) -> SessionManagerCoordinatorGraph:
    """Construct the manager's coordinator graph exactly once."""
    return SessionManagerCoordinatorGraph(
        discovery=discovery_deps_for_manager(manager, deps.discovery_deps),
        queue=queue_coordinator_for_manager(manager, deps.queue_coordinator),
        control=control_coordinator_for_manager(manager, deps.control_coordinator),
        listing=list_coordinator_for_manager(manager, deps.list_coordinator),
        refresh=refresh_coordinator_for_manager(manager, deps.refresh_coordinator),
        readiness=readiness_coordinator_for_manager(manager, deps.readiness_coordinator),
        unattended_sweep=unattended_sweep_coordinator_for_manager(manager, deps.unattended_sweep_coordinator),
        queue_sweep=queue_sweep_coordinator_for_manager(manager, deps.queue_sweep_coordinator),
        voice_runtime=voice_runtime_for_manager(manager, deps.voice_runtime),
        log_runtime=log_runtime_for_manager(manager, deps.log_runtime),
        files=files_coordinator_for_manager(manager),
        ui_state=ui_state_coordinator_for_manager(manager, deps.ui_state_coordinator),
        unattended_config=unattended_config_coordinator_for_manager(manager, deps.unattended_config_coordinator),
        cleanup=cleanup_coordinator_for_manager(manager, deps.cleanup_coordinator),
        pending_state=pending_state_coordinator_for_manager(manager, deps.pending_state_coordinator),
        recent_cwd=recent_cwd_coordinator_for_manager(manager, deps.recent_cwd_coordinator),
        lifecycle=lifecycle_coordinator_for_manager(manager, deps.lifecycle_coordinator),
        discovery_registry=discovery_registry_for_manager(manager, deps.discovery_registry),
        prune=prune_coordinator_for_manager(manager, deps.prune_coordinator),
        broker_watchdog=broker_watchdog_coordinator_for_manager(manager, deps.broker_watchdog_coordinator),
        send=send_coordinator_for_manager(manager, deps.send_coordinator),
        prelog_recorder=prelog_user_message_recorder_for_manager(manager, deps.prelog_user_message_recorder),
        web_launch=web_launch_coordinator_for_manager(manager, deps.web_launch_coordinator),
    )
