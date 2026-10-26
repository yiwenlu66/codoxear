from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, MutableMapping
from uuid import uuid4

from .attachment_resource import upload_id_value
from .file_upload import remove_staged_attachment_file
from .session_model import Session
from .session_store import SessionStore
from .session_store import public_staged_attachment
from .session_store import public_staged_attachments


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

    @contextmanager
    def _attachment_mutation(self, session_id: str):
        # Publication, revision and commit identity share one persistence boundary.
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            store = self.store()
            before = deepcopy((store.staged_attachments, store.attachment_resources, store.pending_attachment_ids))
            try:
                yield store
                store.save_staged_attachments()
            except Exception:
                store.staged_attachments, store.attachment_resources, store.pending_attachment_ids = before
                raise
            finally:
                session = self.sessions().get(session_id)
                if session:
                    session.pending_attachment = session_id in store.pending_attachment_ids

    def set_pending_attachment(self, session_id: str, value: bool) -> None:
        with self._attachment_mutation(session_id) as store:
            if store.attachment_resource(session_id)["pending"] != bool(value):
                store.advance_attachments(session_id, pending=bool(value))

    def list_staged_attachments(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            return self.store().attachment_snapshot(session_id)

    def add_staged_attachment(self, session_id: str, *, display_name: str, filename: str, path: Path, size: int, created_ts: float) -> dict[str, Any]:
        entry = {"id": self.uuid_hex(), "display_name": display_name or filename,
                 "filename": filename or display_name, "path": str(path), "size": int(size), "created_ts": float(created_ts)}
        with self._attachment_mutation(session_id) as store:
            staged = store.add_staged_attachment(session_id, entry)
            result = dict(store.attachment_snapshot(session_id), attachment=public_staged_attachment(staged))
        return result

    def _delete_files(self, session_id: str, entries: list[dict[str, Any]]) -> None:
        root = self.store().paths.uploads_root
        if root is not None:
            for entry in entries:
                remove_staged_attachment_file(root, session_id, entry["path"])

    def remove_staged_attachment(self, session_id: str, attachment_id: str) -> dict[str, Any]:
        if not isinstance(attachment_id, str) or not attachment_id.strip():
            raise ValueError("attachment id required")
        with self._attachment_mutation(session_id) as store:
            removed, _ = store.remove_staged_attachment(session_id, attachment_id.strip(), delete_files=False)
            result = dict(store.attachment_snapshot(session_id), removed=public_staged_attachment(removed))
        self._delete_files(session_id, [removed])
        return result

    def clear_staged_attachments(self, session_id: str, *, delete_files: bool = True, attachment_ids: list[str] | None = None) -> dict[str, Any]:
        with self._attachment_mutation(session_id) as store:
            if attachment_ids is None:
                removed = store.clear_staged_attachments(session_id, delete_files=False)
            else:
                # A confirmed send consumes exactly its captured set. Uploads
                # committed while broker dispatch ran belong to the next send.
                removed = []
                for entry in store.staged_attachments_for_session(session_id):
                    if entry["id"] in attachment_ids:
                        item, _ = store.remove_staged_attachment(session_id, entry["id"], delete_files=False)
                        removed.append(item)
            result = dict(store.attachment_snapshot(session_id), removed_count=len(removed))
        if delete_files:
            self._delete_files(session_id, removed)
        return result

    def clear_pending_attachment(self, session_id: str) -> dict[str, Any]:
        return self.clear_staged_attachments(session_id)

    def upload_status(self, session_id: str, upload_id: str) -> dict[str, Any]:
        upload_id_value(upload_id)
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            store = self.store()
            record = store.attachment_resource(session_id)["uploads"].get(upload_id, {})
            attachment = next((entry for entry in store.staged_attachments_for_session(session_id)
                               if entry["id"] == record.get("attachment_id")), None)
            result = dict(store.attachment_snapshot(session_id), upload_id=upload_id,
                          state=record.get("state", "absent"), active=attachment is not None)
            if attachment is not None:
                result["attachment"] = public_staged_attachment(attachment)
            if record.get("state") == "failed":
                result["retryable"] = bool(record.get("retryable"))
            return result

    def begin_upload(self, session_id: str, upload_id: str, *, filename: str, size: int) -> dict[str, Any]:
        upload_id_value(upload_id)
        with self._attachment_mutation(session_id) as store:
            uploads = store.attachment_resource(session_id)["uploads"]
            record = uploads.get(upload_id)
            if record and record.get("filename") is not None and (record["filename"] != filename or record["size"] != size):
                raise ValueError("upload_id reused with incompatible file")
            owner = record is None or (record["state"] == "failed" and record.get("retryable"))
            if owner:
                uploads[upload_id] = {"state": "pending", "filename": filename, "size": size}
                store.advance_attachments(session_id)
            # Do not reacquire self.lock (test doubles may use a non-reentrant lock).
            result = dict(store.attachment_snapshot(session_id), upload_id=upload_id,
                          state=uploads[upload_id]["state"], writer=bool(owner))
        return result

    def finish_upload(self, session_id: str, upload_id: str, *, filename: str, raw: bytes, stage: Callable[[], Path]) -> dict[str, Any]:
        digest = hashlib.sha256(raw).hexdigest()
        path = None
        try:
            with self._attachment_mutation(session_id) as store:
                record = store.attachment_resource(session_id)["uploads"][upload_id]
                if record.get("digest") is not None and record["digest"] != digest:
                    raise ValueError("upload_id reused with incompatible file")
                if record["state"] == "pending":
                    path = stage()
                    entry = store.add_staged_attachment(session_id, {
                        "id": self.uuid_hex(), "display_name": filename, "filename": path.name,
                        "path": str(path), "size": len(raw), "created_ts": self.now(),
                    })
                    record.update(state="committed", attachment_id=entry["id"], digest=digest)
        except Exception:
            if path is not None:
                path.unlink(missing_ok=True)
            raise
        return self.upload_status(session_id, upload_id)

    def fail_upload(self, session_id: str, upload_id: str) -> None:
        with self._attachment_mutation(session_id) as store:
            record = store.attachment_resource(session_id)["uploads"].get(upload_id)
            if record and record["state"] == "pending":
                record.update(state="failed", retryable=True)
                store.advance_attachments(session_id)

    def cancel_upload(self, session_id: str, upload_id: str) -> dict[str, Any]:
        upload_id_value(upload_id)
        removed = []
        with self._attachment_mutation(session_id) as store:
            uploads = store.attachment_resource(session_id)["uploads"]
            record = uploads.get(upload_id, {})
            for entry in store.staged_attachments_for_session(session_id):
                if entry["id"] == record.get("attachment_id"):
                    item, _ = store.remove_staged_attachment(session_id, entry["id"], delete_files=False)
                    removed.append(item)
            # A consumed committed ID stays committed: cancel must not unlink a
            # path already handed to the agent. Only an active chip is removed.
            if record.get("state") != "committed":
                uploads[upload_id] = dict(record, state="cancelled", retryable=False)
            store.advance_attachments(session_id)
        self._delete_files(session_id, removed)
        return self.upload_status(session_id, upload_id)

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
        display_text = raw.get("display_text")
        record: dict[str, Any] = {"text": text, "created_ts": created_ts}
        if isinstance(display_text, str):
            record["display_text"] = display_text
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
