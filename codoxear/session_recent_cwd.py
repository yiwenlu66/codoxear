from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Iterable, MutableMapping


@dataclass(frozen=True)
class SessionRecentCwdCoordinator:
    lock: Any
    recent_cwds: Callable[[], MutableMapping[str, float]]
    set_recent_cwds: Callable[[dict[str, float]], None]
    clean_recent_cwd: Callable[[Any], str | None]
    iter_session_logs: Callable[[], Iterable[Path]]
    resume_candidate_from_log: Callable[[Path], dict[str, Any] | None]
    save_recent_cwds: Callable[[], None]
    now: Callable[[], float]
    max_recent_cwds: int

    def remember(self, cwd: Any, *, ts: Any = None) -> bool:
        cleaned = self.clean_recent_cwd(cwd)
        if cleaned is None:
            return False
        if isinstance(ts, bool):
            ts_value = self.now()
        else:
            try:
                ts_value = float(ts) if ts is not None else self.now()
            except (TypeError, ValueError, OverflowError):
                ts_value = self.now()
        if not math.isfinite(ts_value) or ts_value <= 0:
            ts_value = self.now()
        with self.lock:
            recent = self.recent_cwds()
            if not isinstance(recent, dict):
                self.set_recent_cwds({})
                recent = self.recent_cwds()
            previous = recent.get(cleaned)
            if previous is not None and previous >= ts_value:
                return False
            recent[cleaned] = ts_value
            if len(recent) > self.max_recent_cwds * 2:
                keep = dict(sorted(recent.items(), key=lambda item: (-float(item[1]), item[0]))[: self.max_recent_cwds])
                recent.clear()
                recent.update(keep)
        return True

    def backfill_from_logs(self) -> None:
        changed = False
        seen: set[str] = set()
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
            if len(seen) >= self.max_recent_cwds:
                break
        if changed:
            self.save_recent_cwds()

    def list_recent(self, *, limit: int) -> list[str]:
        with self.lock:
            items = list(self.recent_cwds().items())
        return [cwd for cwd, _ts in sorted(items, key=lambda item: (-float(item[1]), item[0]))[: max(0, int(limit))]]
