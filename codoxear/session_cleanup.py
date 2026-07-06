from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from .session_model import Session
from .session_store import SessionStore


@dataclass(frozen=True)
class SessionCleanupCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    store: Callable[[], SessionStore]
    input_locks: Callable[[], MutableMapping[str, Any]]
    unlink_quiet: Callable[[Path], None]
    save_pending_attachments: Callable[[], None]
    save_commit_unknown_sends: Callable[[], None]
    save_aliases: Callable[[], None]
    save_sidebar_meta: Callable[[], None]
    save_hidden_sessions: Callable[[], None]
    save_unattended: Callable[[], None]
    save_files: Callable[[], None]
    save_queues: Callable[[], None]
    save_staged_attachments: Callable[[], None] = lambda: None

    def prune_stale_socket_without_metadata(self, session_id: str, sock: Path) -> None:
        with self.lock:
            self.sessions().pop(session_id, None)
        self.clear_deleted_session_state(session_id)
        self.unlink_quiet(sock)
        self.unlink_quiet(sock.with_suffix(".json"))

    def clear_deleted_session_state(self, session_id: str, *, clear_recovery: bool = False, cwd: str = "") -> None:
        with self.lock:
            changes = self.store().clear_deleted_session_state(session_id, clear_recovery=clear_recovery, cwd=cwd)
            input_locks = self.input_locks()
            if isinstance(input_locks, dict):
                input_locks.pop(session_id, None)
        self.store().save_deleted_session_state_changes(
            changes,
            save_pending_attachments=self.save_pending_attachments,
            save_staged_attachments=self.save_staged_attachments,
            save_commit_unknown_sends=self.save_commit_unknown_sends,
            save_aliases=self.save_aliases,
            save_sidebar_meta=self.save_sidebar_meta,
            save_hidden_sessions=self.save_hidden_sessions,
            save_unattended=self.save_unattended,
            save_files=self.save_files,
            save_queues=self.save_queues,
        )
