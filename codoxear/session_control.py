from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from .session_model import Session


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

    def _session_and_sock(self, session_id: str) -> tuple[Session, Path]:
        with self.lock:
            session = self.sessions().get(session_id)
            if not session:
                raise KeyError("unknown session")
            return session, session.sock_path

    def _dead_processes(self, session: Session) -> bool:
        return (not self.pid_alive(int(session.broker_pid))) and (not self.pid_alive(int(session.codex_pid)))

    def _drop_dead_session(self, session_id: str, sock: Path, *, clear_deleted_state: bool) -> None:
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
                self._drop_dead_session(session_id, sock, clear_deleted_state=True)
                raise KeyError("unknown session")
            raise
        with self.lock:
            current = self.sessions().get(session_id)
            if current:
                busy_val, queue_len = self.broker_busy_queue(response)
                interrupted_idle = self.broker_interrupted_idle(response)
                current.busy = busy_val
                current.queue_len = queue_len
                current.interrupted_idle = interrupted_idle
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
                self._drop_dead_session(session_id, sock, clear_deleted_state=False)
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
                self._drop_dead_session(session_id, sock, clear_deleted_state=True)
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
                self._drop_dead_session(session_id, sock, clear_deleted_state=False)
                raise KeyError("unknown session")
            raise
        except Exception:
            if self._dead_processes(session):
                self._drop_dead_session(session_id, sock, clear_deleted_state=False)
                raise KeyError("unknown session")
            raise
        return response
