from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping
import sys

from .session_discovery import DiscoveryRegistration, DiscoveryResult
from .session_model import Session


@dataclass(frozen=True)
class SessionDiscoveryRegistryCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    pending_attachment_ids: Callable[[], set[str]]
    commit_unknown_sends: Callable[[], MutableMapping[str, dict[str, Any]]]
    reset_log_caches: Callable[[Session, int], None]
    record_launch_attempt: Callable[[dict[str, Any]], None]
    prune_stale_socket_without_metadata: Callable[[str, Path], None]
    unhide_session: Callable[[str], None]
    unlink_quiet: Callable[[Path], None]
    remember_recent_cwd: Callable[..., bool]
    save_recent_cwds: Callable[[], None]
    stderr: Any = sys.stderr

    def apply_result(self, result: DiscoveryResult) -> None:
        recent_cwd_dirty = False
        for action in result.stale_actions:
            if action.failure_record is not None:
                try:
                    self.record_launch_attempt(action.failure_record)
                except Exception as exc:
                    self.stderr.write(f"error: failed to record launch failure for {action.sock_path}: {type(exc).__name__}: {exc}\n")
                    self.stderr.flush()
            if action.clear_session_state:
                self.prune_stale_socket_without_metadata(action.session_id, action.sock_path)
                continue
            if action.unhide_session:
                self.unhide_session(action.session_id)
            self.unlink_quiet(action.sock_path)
            self.unlink_quiet(action.meta_path)

        for recent in result.recent_cwds:
            if self.remember_recent_cwd(recent.cwd, ts=recent.ts):
                recent_cwd_dirty = True

        for registration in result.registrations:
            self.upsert_registration(registration)

        if recent_cwd_dirty:
            self.save_recent_cwds()

    def upsert_registration(self, registration: DiscoveryRegistration) -> None:
        pending_ids = self.pending_attachment_ids()
        unknown_sends = self.commit_unknown_sends()
        session = Session(
            session_id=registration.session_id,
            thread_id=registration.thread_id,
            broker_pid=registration.broker_pid,
            codex_pid=registration.codex_pid,
            agent_backend=registration.agent_backend,
            owned=registration.owned,
            transport=registration.transport,
            start_ts=float(registration.start_ts),
            cwd=str(registration.cwd),
            log_path=registration.log_path,
            sock_path=registration.sock_path,
            busy=registration.busy,
            queue_len=registration.queue_len,
            token=registration.token,
            meta_thinking=0,
            meta_tools=0,
            meta_system=0,
            meta_log_off=registration.meta_log_off,
            model_provider=registration.model_provider,
            preferred_auth_method=registration.preferred_auth_method,
            model=registration.model,
            reasoning_effort=registration.reasoning_effort,
            service_tier=registration.service_tier,
            tmux_session=registration.tmux_session,
            tmux_window=registration.tmux_window,
            launch_id=registration.launch_id,
            spawn_nonce=registration.spawn_nonce,
            resume_session_id=registration.resume_session_id,
            pending_attachment=registration.session_id in pending_ids,
            commit_unknown_send=dict(unknown_sends.get(registration.session_id) or {}) or None,
            sync_send_supported=registration.sync_send_supported,
            key_write_errors_supported=registration.key_write_errors_supported,
            interrupted_idle=registration.interrupted_idle,
        )
        with self.lock:
            previous = self.sessions().get(registration.session_id)
            if not previous:
                self.reset_log_caches(session, registration.meta_log_off)
                session.model_provider = registration.model_provider
                session.preferred_auth_method = registration.preferred_auth_method
                session.model = registration.model
                session.reasoning_effort = registration.reasoning_effort
                session.service_tier = registration.service_tier
                self.sessions()[registration.session_id] = session
            else:
                previous.sock_path = session.sock_path
                previous.thread_id = session.thread_id
                previous.broker_pid = session.broker_pid
                previous.codex_pid = session.codex_pid
                previous.agent_backend = session.agent_backend
                previous.owned = session.owned
                previous.transport = session.transport
                previous.start_ts = session.start_ts
                previous.cwd = session.cwd
                previous.busy = session.busy
                previous.queue_len = session.queue_len
                previous.interrupted_idle = session.interrupted_idle
                previous.token = session.token
                if previous.log_path != session.log_path:
                    previous.log_path = session.log_path
                    self.reset_log_caches(previous, registration.meta_log_off)
                previous.model_provider = registration.model_provider
                previous.preferred_auth_method = registration.preferred_auth_method
                previous.model = registration.model
                previous.reasoning_effort = registration.reasoning_effort
                previous.service_tier = registration.service_tier
                previous.tmux_session = registration.tmux_session
                previous.tmux_window = registration.tmux_window
                previous.launch_id = registration.launch_id
                previous.spawn_nonce = registration.spawn_nonce
                previous.resume_session_id = registration.resume_session_id
                previous.pending_attachment = bool(previous.pending_attachment or registration.session_id in self.pending_attachment_ids())
                previous.commit_unknown_send = dict(self.commit_unknown_sends().get(registration.session_id) or {}) or None
                previous.sync_send_supported = registration.sync_send_supported
                previous.key_write_errors_supported = registration.key_write_errors_supported
