from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, MutableMapping

from .draft_store import DraftStore
from .session_model import Session


@dataclass(frozen=True)
class SessionDraftCoordinator:
    """Server-side composer-draft get/set for known sessions.

    Drafts are backend-neutral UI state keyed by session id. ``draft_get``
    projects the stored entry (empty text, timestamp 0 when absent);
    ``draft_set`` writes with the server wall clock — empty/
    whitespace-only text writes a timestamped tombstone entry so the
    deletion propagates through the same last-writer-wins channel as
    edits (entry removal is reserved for session deletion).
    """

    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    drafts: Callable[[], MutableMapping[str, Any]]
    draft_store: Callable[[], DraftStore]
    save_drafts: Callable[[], None]
    now: Callable[[], float] = time.time

    def draft_get(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            entry = self.drafts().get(session_id)
        return self.draft_store().public_entry(entry)

    def draft_set(self, session_id: str, text: str) -> float:
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            updated_ts = self.draft_store().set(self.drafts(), session_id, text, now_ts=self.now())
        self.save_drafts()
        return updated_ts
