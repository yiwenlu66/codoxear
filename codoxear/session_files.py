from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, MutableMapping

from .session_model import Session
from .session_store import SessionStore
from .session_store import _file_entry_path as _entry_path


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
        # ``files_get`` is consumed by callers that treat entries as filesystem
        # path strings (e.g. basename-based tracked-file resolution). Return
        # only the path component so tokenized (dict) entries remain
        # transparent to those callers; the token channel reaches the UI through
        # the listing layer which reads the polymorphic store entries directly.
        dirty = False
        out: list[str] = []
        with self.lock:
            key, legacy_keys, _session = self.files_key_for_session(session_id)
            entries, dirty = self.store.file_history_for_keys(key, legacy_keys)
            out = [_entry_path(entry) for entry in entries]
        if dirty:
            self.save_files()
        return list(out)

    def add(self, session_id: str, path: str, api_path: str | None = None) -> list[str]:
        value = str(path)
        if value == "":
            return self.get(session_id)
        token = str(api_path or "")
        with self.lock:
            key, legacy_keys, _session = self.files_key_for_session(session_id)
            self.store.add_file_history_entry(key, legacy_keys, value, api_path=token)
        self.save_files()
        return self.get(session_id)

    def clear(self, session_id: str) -> None:
        dirty = False
        with self.lock:
            key, legacy_keys, session = self.files_key_for_session(session_id)
            cwd = str(getattr(session, "cwd", "") or "")
            dirty = self.store.clear_file_history_for_keys(key, legacy_keys, cwd=cwd)
        if dirty:
            self.save_files()
