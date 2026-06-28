from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from .session_model import Session


@dataclass(frozen=True)
class SessionCleanupCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    aliases: Callable[[], MutableMapping[str, str]]
    sidebar_meta: Callable[[], MutableMapping[str, dict[str, Any]]]
    unattended: Callable[[], MutableMapping[str, dict[str, Any]]]
    files: Callable[[], MutableMapping[str, list[str]]]
    queues: Callable[[], MutableMapping[str, list[dict[str, Any]]]]
    commit_unknown_sends: Callable[[], MutableMapping[str, dict[str, Any]]]
    input_locks: Callable[[], MutableMapping[str, Any]]
    pending_attachment_ids: Callable[[], set[str]]
    unhide_session: Callable[[str], None]
    mark_queue_orphan_recovery_locked: Callable[[str], bool]
    unlink_quiet: Callable[[Path], None]
    save_pending_attachments: Callable[[], None]
    save_commit_unknown_sends: Callable[[], None]
    save_aliases: Callable[[], None]
    save_sidebar_meta: Callable[[], None]
    save_unattended: Callable[[], None]
    save_files: Callable[[], None]
    save_queues: Callable[[], None]

    def prune_stale_socket_without_metadata(self, session_id: str, sock: Path) -> None:
        with self.lock:
            self.sessions().pop(session_id, None)
        self.unhide_session(session_id)
        self.clear_deleted_session_state(session_id)
        self.unlink_quiet(sock)
        self.unlink_quiet(sock.with_suffix(".json"))

    def clear_deleted_session_state(self, session_id: str, *, clear_recovery: bool = False) -> None:
        changed_sidebar = False
        changed_unattended = False
        changed_files = False
        changed_queues = False
        changed_unknown_sends = False
        with self.lock:
            aliases = self.aliases()
            if isinstance(aliases, dict):
                aliases.pop(session_id, None)
            meta_map = self.sidebar_meta()
            if isinstance(meta_map, dict) and session_id in meta_map:
                meta_map.pop(session_id, None)
                changed_sidebar = True
            if isinstance(meta_map, dict):
                for entry in meta_map.values():
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("dependency_session_id") != session_id:
                        continue
                    entry.pop("dependency_session_id", None)
                    changed_sidebar = True
            unattended = self.unattended()
            if isinstance(unattended, dict) and session_id in unattended:
                unattended.pop(session_id, None)
                changed_unattended = True
            files = self.files()
            if isinstance(files, dict):
                for key in [f"sid:{session_id}", session_id]:
                    if key in files:
                        files.pop(key, None)
                        changed_files = True
            unknown_sends = self.commit_unknown_sends()
            has_direct_unknown = isinstance(unknown_sends, dict) and session_id in unknown_sends
            queues = self.queues()
            if isinstance(queues, dict) and session_id in queues:
                queue = queues.get(session_id)
                has_queued_recovery = isinstance(queue, list) and any(
                    isinstance(item, dict) and (bool(item.get("commit_unknown")) or bool(item.get("orphan_recovery")))
                    for item in queue
                )
                if isinstance(queue, list) and queue and has_direct_unknown:
                    if self.mark_queue_orphan_recovery_locked(session_id):
                        changed_queues = True
                    has_queued_recovery = True
                if clear_recovery or not has_queued_recovery:
                    queues.pop(session_id, None)
                    changed_queues = True
            input_locks = self.input_locks()
            if isinstance(input_locks, dict):
                input_locks.pop(session_id, None)
            pending_attachment_ids = self.pending_attachment_ids()
            if isinstance(pending_attachment_ids, set):
                pending_attachment_ids.discard(session_id)
            if clear_recovery and isinstance(unknown_sends, dict) and session_id in unknown_sends:
                unknown_sends.pop(session_id, None)
                changed_unknown_sends = True
        self.save_pending_attachments()
        if changed_unknown_sends:
            self.save_commit_unknown_sends()
        self.save_aliases()
        if changed_sidebar:
            self.save_sidebar_meta()
        if changed_unattended:
            self.save_unattended()
        if changed_files:
            self.save_files()
        if changed_queues:
            self.save_queues()
