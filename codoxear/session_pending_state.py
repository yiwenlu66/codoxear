from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, MutableMapping

from .session_model import Session


@dataclass(frozen=True)
class SessionPendingStateCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
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

    def clear_pending_attachment(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
        self.set_pending_attachment(session_id, False)
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
        changed = False
        queue_changed = False
        now_ts = self.now()
        age_limit = self.commit_unknown_orphan_prune_seconds if max_age_seconds is None else float(max_age_seconds)
        with self.lock:
            unknown_sends = self.commit_unknown_sends()
            if not isinstance(unknown_sends, dict):
                self.set_commit_unknown_sends({})
                return False
            active_ids = set(self.sessions().keys())
            for sid, record in list(unknown_sends.items()):
                if sid in active_ids:
                    continue
                created_raw = record.get("created_ts") if isinstance(record, dict) else None
                try:
                    created_ts = float(created_raw)
                except (TypeError, ValueError):
                    created_ts = now_ts
                if math.isfinite(created_ts) and created_ts > 0 and (now_ts - created_ts) < age_limit:
                    continue
                if self.mark_queue_orphan_recovery_locked(str(sid)):
                    queue_changed = True
                unknown_sends.pop(sid, None)
                changed = True
        if queue_changed:
            self.save_queues()
        if changed:
            self.save_commit_unknown_sends()
        return changed
