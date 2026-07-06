from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, MutableMapping
from uuid import uuid4

from .session_model import Session
from .session_store import SessionStore


@dataclass(frozen=True)
class SessionPendingStateCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    store: Callable[[], SessionStore]
    pending_attachment_ids: Callable[[], set[str]]
    set_pending_attachment_ids: Callable[[set[str]], None]
    commit_unknown_sends: Callable[[], MutableMapping[str, dict[str, Any]]]
    set_commit_unknown_sends: Callable[[dict[str, dict[str, Any]]], None]
    mark_queue_orphan_recovery_locked: Callable[[str], bool]
    save_pending_attachments: Callable[[], None]
    save_commit_unknown_sends: Callable[[], None]
    save_queues: Callable[[], None]
    now: Callable[[], float]
    commit_unknown_orphan_prune_seconds: float
    save_staged_attachments: Callable[[], None] = lambda: None
    uuid_hex: Callable[[], str] = lambda: uuid4().hex

    def set_pending_attachment(self, session_id: str, value: bool) -> None:
        with self.lock:
            ids = self.pending_attachment_ids()
            if not isinstance(ids, set):
                self.set_pending_attachment_ids(set())
                ids = self.pending_attachment_ids()
            session = self.sessions().get(session_id)
            if session:
                session.pending_attachment = bool(value)
            if value:
                ids.add(session_id)
            else:
                ids.discard(session_id)
        self.save_pending_attachments()

    def _sync_pending_projection_locked(self, session_id: str) -> bool:
        has_staged = bool(self.store().staged_attachments.get(session_id))
        ids = self.pending_attachment_ids()
        if not isinstance(ids, set):
            self.set_pending_attachment_ids(set())
            ids = self.pending_attachment_ids()
        session = self.sessions().get(session_id)
        if session:
            session.pending_attachment = has_staged
        if has_staged:
            ids.add(session_id)
        else:
            ids.discard(session_id)
        return has_staged

    def list_staged_attachments(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            attachments = self.store().staged_attachments_for_session(session_id)
            pending = bool(attachments)
        return {"ok": True, "attachments": attachments, "pending_attachment": pending}

    def add_staged_attachment(self, session_id: str, *, display_name: str, filename: str, path: Path, size: int, created_ts: float) -> dict[str, Any]:
        entry = {
            "id": self.uuid_hex(),
            "display_name": str(display_name or filename or Path(path).name or "file"),
            "filename": str(filename or display_name or Path(path).name or "file"),
            "path": str(path),
            "size": int(size),
            "created_ts": float(created_ts),
        }
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            staged = self.store().add_staged_attachment(session_id, entry)
            self._sync_pending_projection_locked(session_id)
            attachments = self.store().staged_attachments_for_session(session_id)
        self.save_staged_attachments()
        self.save_pending_attachments()
        return {"ok": True, "attachment": staged, "attachments": attachments, "pending_attachment": True}

    def remove_staged_attachment(self, session_id: str, attachment_id: str) -> dict[str, Any]:
        if not isinstance(attachment_id, str) or not attachment_id.strip():
            raise ValueError("attachment id required")
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            removed, attachments = self.store().remove_staged_attachment(session_id, attachment_id.strip())
            pending = self._sync_pending_projection_locked(session_id)
        self.save_staged_attachments()
        self.save_pending_attachments()
        return {"ok": True, "removed": removed, "attachments": attachments, "pending_attachment": pending}

    def clear_staged_attachments(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            removed = self.store().clear_staged_attachments(session_id)
            self._sync_pending_projection_locked(session_id)
        self.save_staged_attachments()
        self.save_pending_attachments()
        return {"ok": True, "removed_count": len(removed), "attachments": [], "pending_attachment": False}

    def clear_pending_attachment(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
        self.clear_staged_attachments(session_id)
        return {"ok": True, "pending_attachment": False}

    def clean_commit_unknown_send_record(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        timestamp_raw = raw.get("created_ts")
        try:
            created_ts = float(timestamp_raw) if timestamp_raw is not None else self.now()
        except (TypeError, ValueError):
            created_ts = self.now()
        if not math.isfinite(created_ts) or created_ts <= 0:
            created_ts = self.now()
        error = raw.get("error")
        record: dict[str, Any] = {"text": text, "created_ts": created_ts}
        if isinstance(error, str) and error.strip():
            record["error"] = error.strip()
        return record

    def set_commit_unknown_send(self, session_id: str, record: dict[str, Any] | None) -> None:
        cleaned = self.clean_commit_unknown_send_record(record) if record is not None else None
        with self.lock:
            unknown_sends = self.commit_unknown_sends()
            if not isinstance(unknown_sends, dict):
                self.set_commit_unknown_sends({})
                unknown_sends = self.commit_unknown_sends()
            session = self.sessions().get(session_id)
            if cleaned is None:
                unknown_sends.pop(session_id, None)
                if session:
                    session.commit_unknown_send = None
            else:
                unknown_sends[session_id] = dict(cleaned)
                if session:
                    session.commit_unknown_send = dict(cleaned)
        self.save_commit_unknown_sends()

    def clear_commit_unknown_send(self, session_id: str) -> dict[str, Any]:
        queue_changed = False
        with self.lock:
            unknown_sends = self.commit_unknown_sends()
            has_orphan_marker = isinstance(unknown_sends, dict) and session_id in unknown_sends
            if session_id not in self.sessions() and not has_orphan_marker:
                raise KeyError("unknown session")
            if has_orphan_marker:
                queue_changed = self.mark_queue_orphan_recovery_locked(session_id)
        if queue_changed:
            self.save_queues()
        self.set_commit_unknown_send(session_id, None)
        return {"ok": True, "commit_unknown_send": False}

    def prune_missing_commit_unknown_sends(self, *, max_age_seconds: float | None = None) -> bool:
        now_ts = self.now()
        age_limit = self.commit_unknown_orphan_prune_seconds if max_age_seconds is None else float(max_age_seconds)
        with self.lock:
            unknown_sends = self.commit_unknown_sends()
            if not isinstance(unknown_sends, dict):
                self.set_commit_unknown_sends({})
                return False
            changes = self.store().prune_missing_commit_unknown_sends(
                active_session_ids=self.sessions().keys(),
                now_ts=now_ts,
                max_age_seconds=age_limit,
            )
        if changes.queues:
            self.save_queues()
        if changes.commit_unknown_sends:
            self.save_commit_unknown_sends()
        return changes.commit_unknown_sends
