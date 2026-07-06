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
class SessionControlCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    sock_call: Callable[..., dict[str, Any]]
    pid_alive: Callable[[int], bool]
    unlink_quiet: Callable[[Path], None]
    clear_deleted_session_state: Callable[[str], None]
    broker_busy_queue: Callable[[dict[str, Any]], tuple[bool, int]]
    broker_interrupted_idle: Callable[[dict[str, Any]], bool]
    control_socket_call_error: type[BaseException]
    commit_unknown_error: type[BaseException]
    latest_launch_attempt: Callable[[str], dict[str, Any] | None] | None = None
    record_launch_attempt: Callable[[dict[str, Any]], None] | None = None
    compute_idle_from_log: Callable[[Path], bool | None] | None = None
    stderr: Any = sys.stderr

    def _session_and_sock(self, session_id: str) -> tuple[Session, Path]:
        with self.lock:
            session = self.sessions().get(session_id)
            if not session:
                raise KeyError("unknown session")
            return session, session.sock_path

    def _dead_processes(self, session: Session) -> bool:
        return (not self.pid_alive(int(session.broker_pid))) and (not self.pid_alive(int(session.codex_pid)))

    def _record_post_log_bound_failure(self, session_id: str, session: Session, *, stage: str) -> None:
        if self.record_launch_attempt is None:
            return
        if not session.owned:
            return
        if session.log_path is None or not session.log_path.exists():
            return
        if session.launch_id and self.latest_launch_attempt is not None:
            latest = self.latest_launch_attempt(session.launch_id)
            if isinstance(latest, dict) and latest.get("state") == "failed":
                return
        if not log_needs_post_log_bound_recovery(session.log_path, compute_idle_from_log=self.compute_idle_from_log):
            return
        try:
            self.record_launch_attempt(
                compose_post_log_bound_failure_record(
                    session_id=session_id,
                    thread_id=session.thread_id,
                    launch_id=session.launch_id,
                    stage=stage,
                    error="web-owned session control socket became unreachable after binding a transcript log before the turn completed",
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
            self.stderr.write(f"error: failed to record post-log control recovery for {session_id}: {type(exc).__name__}: {exc}\n")
            self.stderr.flush()

    def _drop_dead_session(self, session_id: str, session: Session, sock: Path, *, clear_deleted_state: bool, stage: str) -> None:
        self._record_post_log_bound_failure(session_id, session, stage=stage)
        with self.lock:
            self.sessions().pop(session_id, None)
        if clear_deleted_state:
            self.clear_deleted_session_state(session_id)
        self.unlink_quiet(sock)
        self.unlink_quiet(sock.with_suffix(".json"))

    def get_state(self, session_id: str) -> dict[str, Any]:
        session, sock = self._session_and_sock(session_id)
        try:
            response = self.sock_call(sock, {"cmd": "state"}, timeout_s=1.5)
        except Exception:
            if self._dead_processes(session):
                self._drop_dead_session(session_id, session, sock, clear_deleted_state=True, stage="session_control_state_after_log_bind")
                raise KeyError("unknown session")
            raise
        with self.lock:
            current = self.sessions().get(session_id)
            if current:
                busy_val, queue_len = self.broker_busy_queue(response)
                interrupted_idle = self.broker_interrupted_idle(response)
                current.busy = busy_val
                current.queue_len = queue_len
                set_session_interrupted_idle(current, interrupted_idle)
                if "token" in response:
                    token = response.get("token")
                    if isinstance(token, dict) or token is None:
                        log_available = current.log_path is not None and current.log_path.exists()
                        if not log_available:
                            current.token = token
        return response

    def get_tail(self, session_id: str) -> str:
        session, sock = self._session_and_sock(session_id)
        try:
            response = self.sock_call(sock, {"cmd": "tail"}, timeout_s=1.5)
        except Exception:
            if self._dead_processes(session):
                self._drop_dead_session(session_id, session, sock, clear_deleted_state=False, stage="session_control_tail_after_log_bind")
                raise KeyError("unknown session")
            raise
        if "tail" not in response:
            raise ValueError("invalid broker tail response")
        tail = response.get("tail")
        if not isinstance(tail, str):
            raise ValueError("invalid broker tail response")
        return tail

    def call_confirmed_send(
        self,
        session_id: str,
        *,
        session: Session,
        sock: Path,
        text: str,
        timeout_s: float | None,
        raise_commit_unknown: Callable[[str, BaseException | None], None],
        not_ready_error: type[BaseException],
        timeout_errors: tuple[type[BaseException], ...],
    ) -> dict[str, Any]:
        try:
            return self.sock_call(
                sock,
                {"cmd": "send", "text": text, "sync": True},
                timeout_s=timeout_s,
                track_request_sent=True,
            )
        except self.control_socket_call_error as exc:
            if bool(getattr(exc, "request_sent", False)):
                raise_commit_unknown("send commit status unknown; broker response failed", exc)
            if self._dead_processes(session):
                self._drop_dead_session(session_id, session, sock, clear_deleted_state=True, stage="session_control_send_after_log_bind")
                raise KeyError("unknown session")
            raise not_ready_error("session control socket unavailable") from exc
        except timeout_errors as exc:
            raise_commit_unknown("send commit status unknown; broker did not reply before timeout", exc)

    def inject_keys(self, session_id: str, seq: str, *, track_request_sent: bool = False, interrupt: bool = False) -> dict[str, Any]:
        session, sock = self._session_and_sock(session_id)
        try:
            request: dict[str, Any] = {"cmd": "keys", "seq": seq}
            if interrupt:
                request["interrupt"] = True
            response = self.sock_call(sock, request, timeout_s=2.0, track_request_sent=track_request_sent)
        except self.control_socket_call_error as exc:
            if track_request_sent and bool(getattr(exc, "request_sent", False)):
                raise self.commit_unknown_error("attachment commit status unknown; broker response failed") from exc
            if self._dead_processes(session):
                self._drop_dead_session(session_id, session, sock, clear_deleted_state=False, stage="session_control_keys_after_log_bind")
                raise KeyError("unknown session")
            raise
        except Exception:
            if self._dead_processes(session):
                self._drop_dead_session(session_id, session, sock, clear_deleted_state=False, stage="session_control_keys_after_log_bind")
                raise KeyError("unknown session")
            raise
        return response
