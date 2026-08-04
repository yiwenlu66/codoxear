from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, MutableMapping

from .session_model import Session


@dataclass(frozen=True)
class BrokerWatchdogCoordinator:
    """Detect broker death independently of backend-child liveness.

    The broker owns the PTY/control socket.  Once its PID is gone, the session
    cannot accept browser control even when its child agent briefly survives.
    The session remains in the registry as a lost tombstone; only the stale
    socket and metadata are removed after the grace period.
    """

    lock: object
    sessions: Callable[[], MutableMapping[str, Session]]
    pid_alive: Callable[[int], bool]
    unlink_quiet: Callable[[Path], None]
    now: Callable[[], float]
    grace_seconds: float

    def sweep(self) -> None:
        with self.lock:
            sessions = list(self.sessions().values())

        prune_paths: list[Path] = []
        now_ts = self.now()
        for session in sessions:
            meta_path = session.sock_path.with_suffix(".json")
            # A no-sidecar session is handled by normal explicit deletion or
            # discovery cleanup. The watchdog's contract is stale sidecars.
            if not meta_path.exists():
                continue
            if self.pid_alive(int(session.broker_pid)):
                self._clear_lost_if_current(session.session_id)
                continue

            should_prune = self._mark_lost_if_current(session.session_id, now_ts)
            if should_prune:
                prune_paths.append(session.sock_path)

        for sock_path in prune_paths:
            self.unlink_quiet(sock_path)
            self.unlink_quiet(sock_path.with_suffix(".json"))

    def _clear_lost_if_current(self, session_id: str) -> None:
        with self.lock:
            current = self.sessions().get(session_id)
            if current is None or not current.lost:
                return
            current.lost = False
            current.lost_since = None

    def _mark_lost_if_current(self, session_id: str, now_ts: float) -> bool:
        with self.lock:
            current = self.sessions().get(session_id)
            if current is None:
                return False
            if not current.lost:
                current.lost = True
                current.lost_since = now_ts
                return False
            since = current.lost_since
            if not isinstance(since, (int, float)):
                current.lost_since = now_ts
                return False
            return (now_ts - float(since)) >= self.grace_seconds
