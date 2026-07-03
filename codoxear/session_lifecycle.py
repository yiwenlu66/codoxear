from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, MutableMapping

from .session_model import Session


@dataclass(frozen=True)
class SessionLifecycleCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    sock_call: Callable[..., dict[str, Any]]
    process_group_alive: Callable[[int], bool]
    pid_alive: Callable[[int], bool]
    terminate_process_group: Callable[..., bool]
    terminate_process: Callable[..., bool]
    unlink_quiet: Callable[[Path], None]
    commit_unknown_sends: Callable[[], MutableMapping[str, Any]]
    queue_has_recovery_items_locked: Callable[[str], bool]
    clear_deleted_session_state: Callable[..., None]
    read_launch_attempts: Callable[[], Iterable[dict[str, Any]]]
    launch_attempt_row: Callable[[dict[str, Any]], dict[str, Any] | None]
    hide_session: Callable[[str], None]
    clean_optional_text: Callable[[Any], str | None] = lambda value: value if isinstance(value, str) and value.strip() else None
    kill_session_via_pids_fallback: Callable[[Session], bool] | None = None

    def live_session_for_resume_target(self, resume_id: str, resume_row: dict[str, Any] | None) -> Session | None:
        sessions = self.sessions()
        if not isinstance(sessions, dict):
            return None
        target_log_raw = self.clean_optional_text(resume_row.get("log_path") if isinstance(resume_row, dict) else None)
        try:
            target_log = str(Path(target_log_raw).expanduser().resolve(strict=False)) if target_log_raw is not None else None
        except OSError:
            target_log = target_log_raw

        with self.lock:
            values = list(sessions.values())
        for session in values:
            if not isinstance(session, Session):
                continue
            same_thread = session.session_id == resume_id or session.thread_id == resume_id
            same_log = False
            if target_log is not None and session.log_path is not None:
                try:
                    session_log = str(session.log_path.expanduser().resolve(strict=False))
                except OSError:
                    session_log = str(session.log_path)
                same_log = session_log == target_log
            if (same_thread or same_log) and (self.pid_alive(session.broker_pid) or self.pid_alive(session.codex_pid)):
                return session
        return None

    def kill_session_via_pids(self, session: Session) -> bool:
        group_alive = self.process_group_alive(int(session.codex_pid))
        broker_alive = self.pid_alive(int(session.broker_pid))
        if not group_alive and not broker_alive:
            self.unlink_quiet(session.sock_path)
            self.unlink_quiet(session.sock_path.with_suffix(".json"))
            return True
        if group_alive and (not self.terminate_process_group(int(session.codex_pid), wait_seconds=1.0)):
            return False
        if self.pid_alive(int(session.broker_pid)) and (not self.terminate_process(int(session.broker_pid), wait_seconds=1.0)):
            return False
        group_dead = not self.process_group_alive(int(session.codex_pid))
        broker_dead = not self.pid_alive(int(session.broker_pid))
        if group_dead and broker_dead:
            self.unlink_quiet(session.sock_path)
            self.unlink_quiet(session.sock_path.with_suffix(".json"))
            return True
        return False

    def kill_session(self, session_id: str) -> bool:
        with self.lock:
            session = self.sessions().get(session_id)
        if not session:
            return False
        try:
            response = self.sock_call(session.sock_path, {"cmd": "shutdown"}, timeout_s=1.0)
        except Exception:
            return (self.kill_session_via_pids_fallback or self.kill_session_via_pids)(session)
        if response.get("ok") is True:
            return True
        return (self.kill_session_via_pids_fallback or self.kill_session_via_pids)(session)

    def delete_session(self, session_id: str) -> bool:
        with self.lock:
            session = self.sessions().get(session_id)
        if not session:
            with self.lock:
                has_direct_unknown = session_id in self.commit_unknown_sends()
                has_queue_recovery = self.queue_has_recovery_items_locked(session_id)
            if has_direct_unknown or has_queue_recovery:
                self.clear_deleted_session_state(session_id, clear_recovery=True)
                return True
            for record in self.read_launch_attempts():
                row = self.launch_attempt_row(record)
                if row is not None and row.get("session_id") == session_id:
                    self.hide_session(session_id)
                    return True
            return False

        ok = self.kill_session(session_id)
        if ok:
            launch_id = session.launch_id
            cwd = session.cwd
            with self.lock:
                self.sessions().pop(session_id, None)
            if launch_id:
                self.hide_session(launch_id)
            self.clear_deleted_session_state(session_id, clear_recovery=True, cwd=cwd)
        return ok
