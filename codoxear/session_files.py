from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, MutableMapping

from .session_model import Session
from .session_store import SessionStore


@dataclass(frozen=True)
class SessionFilesCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    store: SessionStore
    save_files: Callable[[], None]

    def files_key_for_session(self, session_id: str) -> tuple[str, list[str], Session]:
        session = self.sessions().get(session_id)
        if not session:
            raise KeyError("unknown session")
        sid_key = f"sid:{session_id}"
        return sid_key, [session_id], session

    def get(self, session_id: str) -> list[str]:
        dirty = False
        out: list[str] = []
        with self.lock:
            key, legacy_keys, _session = self.files_key_for_session(session_id)
            out, dirty = self.store.file_history_for_keys(key, legacy_keys)
        if dirty:
            self.save_files()
        return list(out)

    def add(self, session_id: str, path: str) -> list[str]:
        value = str(path)
        if value == "":
            return self.get(session_id)
        with self.lock:
            key, legacy_keys, _session = self.files_key_for_session(session_id)
            current = self.store.add_file_history_entry(key, legacy_keys, value)
        self.save_files()
        return list(current)

    def clear(self, session_id: str) -> None:
        dirty = False
        with self.lock:
            key, legacy_keys, session = self.files_key_for_session(session_id)
            cwd = str(getattr(session, "cwd", "") or "")
            dirty = self.store.clear_file_history_for_keys(key, legacy_keys, cwd=cwd)
        if dirty:
            self.save_files()
