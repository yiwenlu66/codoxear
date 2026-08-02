from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping
import sys

from .post_log_recovery import compose_post_log_bound_failure_record
from .post_log_recovery import log_needs_post_log_bound_recovery
from .session_model import Session
from .session_runtime import set_session_interrupted_idle


@dataclass(frozen=True)
class SessionPruneCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    sock_call: Callable[..., dict[str, Any]]
    broker_busy_queue_from_state: Callable[[dict[str, Any]], tuple[bool, int]]
    broker_interrupted_idle_from_state: Callable[[dict[str, Any]], bool]
    sock_error_definitely_stale: Callable[[BaseException], bool]
    pid_alive: Callable[[int], bool]
    latest_launch_attempt: Callable[[str], dict[str, Any] | None]
    submitted_user_messages: Callable[[dict[str, Any] | None], list[dict[str, Any]]]
    launch_failure_tail: Callable[[dict[str, Any] | None], str]
    which_tmux: Callable[[str], str | None]
    tmux_pane_snapshot: Callable[..., dict[str, Any]]
    clean_optional_text: Callable[[Any], str | None]
    record_launch_attempt: Callable[[dict[str, Any]], None]
    clear_deleted_session_state: Callable[[str], None]
    unlink_quiet: Callable[[Path], None]
    compute_idle_from_log: Callable[[Path], bool | None] | None = None
    stderr: Any = sys.stderr

    def refresh_session_state(self, session_id: str, sock_path: Path, timeout_s: float = 0.4) -> tuple[bool, BaseException | None]:
        try:
            response = self.sock_call(sock_path, {"cmd": "state"}, timeout_s=timeout_s)
        except Exception as exc:
            return False, exc
        try:
            busy_val, queue_len = self.broker_busy_queue_from_state(response)
            interrupted_idle = self.broker_interrupted_idle_from_state(response)
        except ValueError as exc:
            return False, exc
        with self.lock:
            session = self.sessions().get(session_id)
            if session:
                session.busy = busy_val
                session.queue_len = queue_len
                set_session_interrupted_idle(session, interrupted_idle)
                session.pi_thinking_command = session.agent_backend == "pi" and response.get("pi_thinking_command") is True
                if "token" in response:
                    token = response.get("token")
                    if isinstance(token, dict) or token is None:
                        log_available = session.log_path is not None and session.log_path.exists()
                        if not log_available:
                            session.token = token
        return True, None

    def prune_dead_sessions(self) -> None:
        with self.lock:
            items = list(self.sessions().items())
        dead: list[tuple[str, Path, Session]] = []
        for sid, session in items:
            if not session.sock_path.exists():
                dead.append((sid, session.sock_path, session))
                continue
            ok, err = self.refresh_session_state(sid, session.sock_path, timeout_s=0.4)
            if ok:
                continue
            if err is not None and self.sock_error_definitely_stale(err):
                dead.append((sid, session.sock_path, session))
                continue
            if self.pid_alive(session.broker_pid) or self.pid_alive(session.codex_pid):
                continue
            dead.append((sid, session.sock_path, session))
        if not dead:
            return
        with self.lock:
            for sid, _sock, _session in dead:
                self.sessions().pop(sid, None)
        for sid, sock, session in dead:
            self._record_pruned_launch_failure(sid, session)
            self.clear_deleted_session_state(sid)
            self.unlink_quiet(sock)
            self.unlink_quiet(sock.with_suffix(".json"))

    def _record_pruned_launch_failure(self, sid: str, session: Session) -> None:
        existing_launch_failed = False
        latest_launch_record: dict[str, Any] | None = None
        if session.launch_id:
            latest_launch_record = self.latest_launch_attempt(session.launch_id)
            existing_launch_failed = bool(latest_launch_record and latest_launch_record.get("state") == "failed")
        if existing_launch_failed or not session.owned:
            return
        if session.log_path is not None and session.log_path.exists():
            if not log_needs_post_log_bound_recovery(session.log_path, compute_idle_from_log=self.compute_idle_from_log):
                return
            try:
                self.record_launch_attempt(
                    compose_post_log_bound_failure_record(
                        session_id=sid,
                        thread_id=session.thread_id,
                        launch_id=session.launch_id,
                        stage="session_pruned_after_log_bind",
                        error="web-owned session process disappeared before completing the bound transcript turn",
                        agent_backend=session.agent_backend,
                        cwd=session.cwd,
                        log_path=session.log_path,
                        created_ts=session.start_ts,
                        broker_pid=session.broker_pid,
                        agent_pid=session.codex_pid,
                        transport=session.transport,
                        tmux_session=session.tmux_session,
                        tmux_window=session.tmux_window,
                        spawn_nonce=session.spawn_nonce,
                        model_provider=session.model_provider,
                        preferred_auth_method=session.preferred_auth_method,
                        model=session.model,
                        reasoning_effort=session.reasoning_effort,
                        service_tier=session.service_tier,
                    )
                )
            except Exception as exc:
                self.stderr.write(f"error: failed to record post-log pruned recovery for {sid}: {type(exc).__name__}: {exc}\n")
                self.stderr.flush()
            return
        if session.log_path is not None:
            return
        try:
            tmux_snapshot: dict[str, Any] = {}
            if session.transport == "tmux":
                tmux_bin = self.which_tmux("tmux")
                if tmux_bin is not None:
                    pane_id = (
                        self.clean_optional_text(latest_launch_record.get("tmux_pane_id"))
                        if isinstance(latest_launch_record, dict)
                        else None
                    )
                    tmux_snapshot = self.tmux_pane_snapshot(tmux_bin, pane_id=pane_id, window=session.tmux_window)
            submitted_messages = self.submitted_user_messages(latest_launch_record)
            prior_tail = self.launch_failure_tail(latest_launch_record) if isinstance(latest_launch_record, dict) else ""
            snapshot_tail = self.launch_failure_tail(tmux_snapshot)
            agent_status = None
            broker_status = None
            if isinstance(latest_launch_record, dict):
                previous_agent_status = latest_launch_record.get("agent_exit_status", latest_launch_record.get("exit_code"))
                previous_broker_status = latest_launch_record.get("broker_exit_status")
                if isinstance(previous_agent_status, int):
                    agent_status = previous_agent_status
                if isinstance(previous_broker_status, int):
                    broker_status = previous_broker_status
            failure_record: dict[str, Any] = {
                "launch_id": session.launch_id,
                "state": "failed",
                "stage": "session_pruned_before_log_bind",
                "error": "web-owned session process disappeared before a session log was bound",
                "agent_backend": session.agent_backend,
                "cwd": session.cwd,
                "created_ts": session.start_ts,
                "broker_pid": session.broker_pid,
                "agent_pid": session.codex_pid,
                "transport": session.transport,
                "tmux_session": session.tmux_session,
                "tmux_window": session.tmux_window,
                "spawn_nonce": session.spawn_nonce,
                "model_provider": session.model_provider,
                "preferred_auth_method": session.preferred_auth_method,
                "model": session.model,
                "reasoning_effort": session.reasoning_effort,
                "service_tier": session.service_tier,
            }
            if submitted_messages:
                failure_record["submitted_user_messages"] = submitted_messages
            if prior_tail:
                failure_record["pty_tail"] = prior_tail
            if snapshot_tail:
                failure_record["tmux_pane_tail"] = snapshot_tail
            if agent_status is not None:
                failure_record["agent_exit_status"] = agent_status
            if broker_status is not None:
                failure_record["broker_exit_status"] = broker_status
            failure_record.update(tmux_snapshot)
            self.record_launch_attempt(failure_record)
        except Exception as exc:
            self.stderr.write(f"error: failed to record pruned launch failure for {sid}: {type(exc).__name__}: {exc}\n")
            self.stderr.flush()
