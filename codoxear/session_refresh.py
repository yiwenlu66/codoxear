from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from .agent_backend import normalize_agent_backend
from .session_model import Session
from .sidecar_metadata import detaches_current_log as metadata_detaches_current_log
from .sidecar_metadata import ignored_rollout_paths as metadata_ignored_rollout_paths
from .sidecar_metadata import key_write_errors_supported as metadata_key_write_errors_supported
from .sidecar_metadata import log_path as metadata_log_path
from .sidecar_metadata import read_metadata as read_sidecar_metadata
from .sidecar_metadata import required_int as metadata_required_int
from .sidecar_metadata import required_text as metadata_required_text
from .sidecar_metadata import start_ts as metadata_start_ts
from .sidecar_metadata import sync_send_supported as metadata_sync_send_supported


@dataclass(frozen=True)
class SessionRefreshCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    prune_stale_socket_without_metadata: Callable[[str, Path], None]
    log_invalid_sidecar_metadata: Callable[[str, Path, Exception], None]
    session_transport: Callable[..., tuple[str | None, str | None, str | None]]
    sock_call: Callable[..., dict[str, Any]]
    broker_tail_has_session_detach_marker: Callable[[str, Any], bool]
    pid_alive: Callable[[int], bool]
    proc_find_open_rollout_log: Callable[..., Path | None]
    proc_root: Path
    read_session_meta_or_none: Callable[..., dict[str, Any] | None]
    coerce_main_thread_log: Callable[..., tuple[str, Path]]
    clean_optional_text: Callable[[Any], str | None]
    session_run_settings: Callable[..., tuple[str | None, str | None, str | None, str | None]]
    normalize_requested_service_tier: Callable[[Any], str | None]
    reset_log_caches: Callable[[Session, int], None]
    queue_len: Callable[[str], int]
    maybe_drain_session_queue: Callable[[str], None]

    def refresh_session_meta(self, session_id: str, *, drain_queue: bool = False) -> None:
        with self.lock:
            session = self.sessions().get(session_id)
            if not session:
                return
            sock = session.sock_path
            current_log_path = session.log_path
            current_thread_id = session.thread_id
            current_owned = session.owned
            current_agent_backend = session.agent_backend
            current_codex_pid = session.codex_pid
        meta_path = sock.with_suffix(".json")
        if not meta_path.exists():
            self.prune_stale_socket_without_metadata(session_id, sock)
            return
        try:
            meta = read_sidecar_metadata(meta_path, sock=sock)
            metadata_required_int(meta, "codex_pid", sock=sock)
            metadata_required_int(meta, "broker_pid", sock=sock)
            metadata_start_ts(meta, sock=sock)
            cwd = metadata_required_text(meta, "cwd", sock=sock)
            log_path = metadata_log_path(meta, sock=sock)
            ignored_rollout_paths = metadata_ignored_rollout_paths(meta, sock=sock)
        except ValueError as exc:
            self.log_invalid_sidecar_metadata("refresh", sock, exc)
            return

        thread_id = meta.get("session_id") if isinstance(meta.get("session_id"), str) and meta.get("session_id") else current_thread_id
        owned = (meta.get("owner") == "web") if isinstance(meta.get("owner"), str) else current_owned
        agent_backend = normalize_agent_backend(meta.get("agent_backend"), default=current_agent_backend)
        transport, tmux_session, tmux_window = self.session_transport(meta=meta)
        sync_send_supported = metadata_sync_send_supported(meta)
        key_write_errors_supported = metadata_key_write_errors_supported(meta)
        if log_path is not None and not log_path.exists():
            log_path = None
        if metadata_detaches_current_log(meta, current_log_path):
            try:
                tail_state = self.sock_call(sock, {"cmd": "tail"}, timeout_s=0.4)
            except Exception:
                tail_state = {}
            tail = tail_state.get("tail") if isinstance(tail_state, dict) else None
            if self.broker_tail_has_session_detach_marker(agent_backend, tail):
                ignored_rollout_paths.add(current_log_path)
        if log_path is None and agent_backend in {"codex", "cc"} and self.pid_alive(int(current_codex_pid)):
            discovered_log_path = self.proc_find_open_rollout_log(
                proc_root=self.proc_root,
                root_pid=current_codex_pid,
                agent_backend=agent_backend,
                cwd=cwd,
                ignored_paths=ignored_rollout_paths,
            )
            if discovered_log_path is not None and discovered_log_path.exists():
                log_path = discovered_log_path
        if log_path is not None and agent_backend == "codex":
            session_meta = self.read_session_meta_or_none(log_path, agent_backend="codex", context="session refresh")
            meta_session_id = session_meta.get("id") if session_meta else None
            if isinstance(meta_session_id, str) and meta_session_id:
                thread_id = meta_session_id
            thread_id, log_path = self.coerce_main_thread_log(thread_id=thread_id, log_path=log_path)

        resume_session_id = self.clean_optional_text(meta.get("resume_session_id"))
        model_provider, preferred_auth_method, model, reasoning_effort = self.session_run_settings(
            meta=meta,
            log_path=log_path,
            agent_backend=agent_backend,
        )
        service_tier = self.normalize_requested_service_tier(meta.get("service_tier")) if agent_backend == "codex" else None

        with self.lock:
            current = self.sessions().get(session_id)
            if not current:
                return
            current.thread_id = thread_id
            current.agent_backend = agent_backend
            current.cwd = str(cwd)
            current.owned = bool(owned)
            current.transport = transport
            if current.log_path != log_path:
                current.log_path = log_path
                current.interrupted_idle = False
                if log_path is not None:
                    log_off = int(log_path.stat().st_size)
                else:
                    log_off = 0
                self.reset_log_caches(current, log_off)
            current.model_provider = model_provider
            current.preferred_auth_method = preferred_auth_method
            current.model = model
            current.reasoning_effort = reasoning_effort
            current.service_tier = service_tier
            current.tmux_session = tmux_session
            current.tmux_window = tmux_window
            current.resume_session_id = resume_session_id
            current.sync_send_supported = sync_send_supported
            current.key_write_errors_supported = key_write_errors_supported
        if drain_queue and self.queue_len(session_id) > 0:
            self.maybe_drain_session_queue(session_id)
