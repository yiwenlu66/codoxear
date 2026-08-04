from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

from .util import atomic_write_json, load_json_file


class UnreadStore:
    """Persistent per-session read watermark for transcript projections."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        self._watermarks: dict[str, str] = {}
        self.load()

    def load(self) -> None:
        raw = load_json_file(self.path, default={})
        if not isinstance(raw, dict):
            raise ValueError("invalid session_unread.json (expected object)")
        self._watermarks = {
            str(sid): str(event_id)
            for sid, event_id in raw.items()
            if isinstance(sid, str) and sid.strip() and isinstance(event_id, str) and event_id.strip()
        }

    def watermark(self, session_id: str) -> str | None:
        with self._lock:
            return self._watermarks.get(str(session_id))

    def mark_read(self, session_id: str, event_id: str) -> None:
        sid = str(session_id).strip()
        eid = str(event_id).strip()
        if not sid or not eid:
            raise ValueError("session_id and event_id are required")
        with self._lock:
            self._watermarks[sid] = eid
            atomic_write_json(self.path, dict(self._watermarks))

    def clear(self, session_id: str) -> None:
        with self._lock:
            if self._watermarks.pop(str(session_id), None) is not None:
                atomic_write_json(self.path, dict(self._watermarks))

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._watermarks)
