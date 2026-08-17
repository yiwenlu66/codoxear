from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping
import sys

from .session_discovery import DiscoveryRegistration, DiscoveryResult
from .session_log_projection import LogDerivedSessionObservation
from .session_model import Session
from .session_runtime import set_session_interrupted_idle
from .token_signal import coerce_token_observation


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
    commit_log_observation: Callable[[str, LogDerivedSessionObservation], bool] = lambda _sid, _observation: False
    stderr: Any = sys.stderr

    def apply_result(self, result: DiscoveryResult) -> None:
        recent_cwd_dirty = False
        for action in result.stale_actions:
            # A discovered sidecar can outlive a crashed broker. Preserve an
            # already-active session as a lost tombstone so the watchdog can
            # apply its grace period; missing metadata remains ordinary stale
            # state and is still cleaned immediately.
            with self.lock:
                current = self.sessions().get(action.session_id)
                if current is not None and (not action.clear_session_state) and action.meta_path.exists():
                    if not current.lost:
                        current.lost = True
                        current.lost_since = None
                    continue
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
            token=registration.token if registration.log_path is None else None,
            meta_thinking=0,
            meta_thinking_tokens=0,
            meta_tools=0,
            meta_system=0,
            meta_log_off=registration.meta_log_off,
            model_provider=registration.model_provider if registration.log_path is None else None,
            preferred_auth_method=registration.preferred_auth_method,
            model=registration.model if registration.log_path is None else None,
            reasoning_effort=registration.reasoning_effort if registration.log_path is None else None,
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
            pi_thinking_command=bool(registration.pi_thinking_command),
            slash_commands=list(registration.slash_commands),
            interrupted_idle=registration.interrupted_idle,
            interrupted_idle_log_off=(registration.meta_log_off if registration.interrupted_idle else 0),
            lost=bool(registration.lost),
        )
        should_commit_log = False
        with self.lock:
            previous = self.sessions().get(registration.session_id)
            if not previous:
                self.reset_log_caches(session, registration.meta_log_off)
                # reset_log_caches() clears interrupted_idle/log_off/suppression.
                # Route the broker's interrupted-idle truth through the same
                # helper the broker, refresh, and prune paths use so a fresh
                # discovery of an interrupted stopped turn records an active
                # baseline (current log size) instead of being lost to the
                # reset. One helper owns semantics for new and existing
                # sessions alike; only re-establish when the broker reports
                # true so the false/clearing path stays untouched.
                if registration.interrupted_idle:
                    set_session_interrupted_idle(session, registration.interrupted_idle)
                session.preferred_auth_method = registration.preferred_auth_method
                should_commit_log = registration.log_revision is not None and registration.log_path is not None
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
                # Route the interrupted-idle flag through the same
                # baseline/suppression helper the broker and prune paths use.
                # Direct assignment of ``interrupted_idle_log_off`` to
                # ``registration.meta_log_off`` would re-baseline the override
                # to the current log size, moving it past any post-interrupt
                # resumed activity that arrived before this discovery refresh;
                # the log watcher would then skip that activity as pre-baseline
                # and never clear the stale override. The helper preserves an
                # existing baseline, records a fresh one only for a genuinely
                # new interrupt, respects stale-true suppression, and clears
                # suppression when the broker reports false.
                set_session_interrupted_idle(previous, registration.interrupted_idle)
                same_bound_log = previous.log_path == session.log_path
                if previous.log_path != session.log_path:
                    previous.log_path = session.log_path
                    self.reset_log_caches(previous, registration.meta_log_off)
                if not same_bound_log:
                    previous.token = session.token
                    previous.model_provider = session.model_provider
                    previous.model = session.model
                    previous.reasoning_effort = session.reasoning_effort
                # Discovery has already reconciled live bridge/sidecar settings
                # ahead of delayed JSONL evidence. Route that effective result
                # through the same ordered log boundary even when the bound path
                # is unchanged: equal-end commits allow a live setting change
                # without inventing a second registry writer, while a discovery
                # snapshot behind a newer cursor observation is still rejected.
                should_commit_log = registration.log_revision is not None and registration.log_path is not None
                previous.preferred_auth_method = registration.preferred_auth_method
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
                previous.lost = bool(registration.lost)
                if not previous.lost:
                    previous.lost_since = None
                previous.pi_thinking_command = bool(registration.pi_thinking_command)
        if should_commit_log and registration.log_revision is not None and registration.log_path is not None:
            self.commit_log_observation(
                registration.session_id,
                LogDerivedSessionObservation(
                    log_path=registration.log_path,
                    revision=registration.log_revision,
                    start_off=0,
                    end_off=registration.log_revision[2],
                    token=coerce_token_observation(registration.token),
                    model_provider=registration.model_provider,
                    model=registration.model,
                    reasoning_effort=registration.reasoning_effort,
                    settings_revision=registration.log_revision,
                    effective_settings=True,
                    replace_settings=True,
                ),
            )
