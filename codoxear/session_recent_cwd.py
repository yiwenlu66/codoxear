from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .session_store import SessionStore


@dataclass(frozen=True)
class SessionRecentCwdCoordinator:
    lock: Any
    store: Callable[[], SessionStore]
    iter_session_logs: Callable[[], Iterable[Path]]
    resume_candidate_from_log: Callable[[Path], dict[str, Any] | None]
    save_recent_cwds: Callable[[], None]
    now: Callable[[], float]

    def remember(self, cwd: Any, *, ts: Any = None) -> bool:
        with self.lock:
            return self.store().remember_recent_cwd(cwd, ts=ts, now=self.now)

    def backfill_from_logs(self) -> None:
        changed = False
        seen: set[str] = set()
        max_recent_cwds = int(self.store().recent_cwd_max)
        for log_path in self.iter_session_logs():
            try:
                row = self.resume_candidate_from_log(log_path)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            cwd = row.get("cwd")
            if not isinstance(cwd, str) or not cwd or cwd in seen:
                continue
            seen.add(cwd)
            if self.remember(cwd, ts=row.get("updated_ts")):
                changed = True
            if len(seen) >= max_recent_cwds:
                break
        if changed:
            self.save_recent_cwds()

    def list_recent(self, *, limit: int) -> list[str]:
        with self.lock:
            return self.store().list_recent_cwds(limit=limit)
