from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Mapping, MutableMapping

from .queue_runtime import clear_queue_promotion
from .queue_runtime import queue_idle_grace_ready
from .queue_runtime import reset_queue_idle
from .queue_runtime import start_queue_promotion
from .queue_store import QueueMap
from .queue_store import QueueStore
from .session_model import Session


@dataclass(frozen=True)
class SessionQueueCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    queues: Callable[[], QueueMap]
    queue_store: Callable[[], QueueStore]
    commit_unknown_sends: Callable[[], Mapping[str, Any]]
    save_queues: Callable[[], None]
    remote_ready: Callable[[str, Path | None], bool]
    send: Callable[..., dict[str, Any]]
    not_ready_error: type[BaseException]
    retryable_send_errors: tuple[type[BaseException], ...]
    commit_unknown_error: type[BaseException]
    queue_idle_grace_seconds: float
    now: Callable[[], float] = time.time

    def queue_len(self, session_id: str) -> int:
        with self.lock:
            qmap = self.queues()
            if not isinstance(qmap, dict):
                return 0
            return self.queue_store().queue_len(qmap, session_id)

    def mark_orphan_recovery_locked(self, session_id: str) -> bool:
        qmap = self.queues()
        if not isinstance(qmap, dict):
            return False
        q = qmap.get(session_id)
        if not isinstance(q, list) or not q:
            return False
        changed = False
        for item in q:
            if isinstance(item, dict) and not bool(item.get("orphan_recovery")):
                item["orphan_recovery"] = True
                changed = True
        return changed

    def has_recovery_items_locked(self, session_id: str) -> bool:
        qmap = self.queues()
        if not isinstance(qmap, dict):
            return False
        has_direct_unknown = session_id in self.commit_unknown_sends()
        return has_direct_unknown or self.queue_store().has_recovery_items(qmap, session_id)

    def list_local(self, session_id: str) -> list[dict[str, Any]]:
        with self.lock:
            qmap = self.queues()
            if not isinstance(qmap, dict):
                return []
            session = self.sessions().get(session_id)
            if session is None and not self.has_recovery_items_locked(session_id):
                raise KeyError("unknown session")
            sending_id = session.queue_sending_item_id if session else None
            items = self.queue_store().list_items(qmap, session_id, sending_item_id=sending_id)
            if self.has_recovery_items_locked(session_id):
                for item in items:
                    if not bool(item.get("sending")) and not bool(item.get("commit_unknown")):
                        item["orphan_recovery"] = True
            return items

    def append_item_local(self, session_id: str, text: str, *, reject_recovery_barrier: bool = False) -> tuple[dict[str, Any], int]:
        value = str(text)
        if not value.strip():
            raise ValueError("text required")
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            if reject_recovery_barrier and self.has_recovery_items_locked(session_id):
                raise self.not_ready_error("resolve the recovery queue before queueing another prompt")
            item, ql = self.queue_store().append(self.queues(), session_id, text)
        self.save_queues()
        return item, int(ql)

    def enqueue_local(self, session_id: str, text: str) -> dict[str, Any]:
        item, ql = self.append_item_local(session_id, text, reject_recovery_barrier=True)
        return {"queued": True, "queue_len": int(ql), "item": item}

    def delete_local(
        self,
        session_id: str,
        item_id: str,
        *,
        allow_commit_unknown: bool = False,
        allow_orphan_recovery: bool = False,
    ) -> dict[str, Any]:
        item_id_clean = str(item_id).strip()
        if not item_id_clean:
            raise ValueError("id required")
        with self.lock:
            session = self.sessions().get(session_id)
            queue_recovery = self.has_recovery_items_locked(session_id)
            if session is None and not queue_recovery:
                raise KeyError("unknown session")
            target_item = None
            q_before = self.queues().get(session_id)
            if isinstance(q_before, list):
                target_item = next((item for item in q_before if isinstance(item, dict) and item.get("id") == item_id_clean), None)
            if queue_recovery and isinstance(target_item, dict):
                if not bool(target_item.get("commit_unknown")) and not bool(target_item.get("orphan_recovery")):
                    if not allow_orphan_recovery:
                        raise ValueError("orphan recovery item requires explicit confirmation")
                    target_item["orphan_recovery"] = True
            sending_id = session.queue_sending_item_id if session else None
            ql = self.queue_store().delete(
                self.queues(),
                session_id,
                item_id_clean,
                sending_item_id=sending_id,
                allow_commit_unknown=allow_commit_unknown,
                allow_orphan_recovery=allow_orphan_recovery,
            )
            deleted_recovery = isinstance(target_item, dict) and (
                bool(target_item.get("commit_unknown")) or bool(target_item.get("orphan_recovery"))
            )
            if deleted_recovery and (allow_commit_unknown or allow_orphan_recovery):
                self.mark_orphan_recovery_locked(session_id)
        self.save_queues()
        return {"ok": True, "queue_len": int(ql)}

    def update_local(self, session_id: str, item_id: str, text: str) -> dict[str, Any]:
        item_id_clean = str(item_id).strip()
        value = str(text)
        if not item_id_clean:
            raise ValueError("id required")
        if not value.strip():
            raise ValueError("text required")
        with self.lock:
            session = self.sessions().get(session_id)
            if self.has_recovery_items_locked(session_id):
                raise ValueError("item is preserved for recovery")
            if session is None:
                raise KeyError("unknown session")
            sending_id = session.queue_sending_item_id if session else None
            item, ql = self.queue_store().update(self.queues(), session_id, item_id_clean, value, sending_item_id=sending_id)
        self.save_queues()
        return {"ok": True, "queue_len": int(ql), "item": item}

    def move_local(self, session_id: str, item_id: str, to_index: int) -> dict[str, Any]:
        item_id_clean = str(item_id).strip()
        if not item_id_clean:
            raise ValueError("id required")
        if isinstance(to_index, bool):
            raise ValueError("to_index must be an integer")
        target = int(to_index)
        with self.lock:
            session = self.sessions().get(session_id)
            if self.has_recovery_items_locked(session_id):
                raise ValueError("item is preserved for recovery")
            if session is None:
                raise KeyError("unknown session")
            sending_id = session.queue_sending_item_id if session else None
            ql = self.queue_store().move(self.queues(), session_id, item_id_clean, target, sending_item_id=sending_id)
        self.save_queues()
        return {"ok": True, "queue_len": int(ql)}

    def session_state(self, session_id: str) -> tuple[Session, Path | None]:
        with self.lock:
            session = self.sessions().get(session_id)
            if not session:
                raise KeyError("unknown session")
            return session, session.log_path

    def promote_head_if_sendable(
        self,
        session_id: str,
        *,
        require_idle_grace: bool,
        now_ts: float | None = None,
        expected_item_id: str | None = None,
    ) -> dict[str, Any] | None:
        if now_ts is None:
            now_ts = self.now()
        _session, log_path = self.session_state(session_id)
        with self.lock:
            session = self.sessions().get(session_id)
            if not session:
                return None
            queue_store = self.queue_store()
            if queue_store.queue_len(self.queues(), session_id) <= 0:
                reset_queue_idle(session)
                return None
            if session.queue_sending_item_id is not None:
                return None
            if queue_store.has_recovery_items(self.queues(), session_id):
                reset_queue_idle(session)
                return None
            head = queue_store.promotion_head(self.queues(), session_id, expected_item_id=expected_item_id)
            if head is None:
                return None
            head_id = head.item_id
        try:
            ready = self.remote_ready(session_id, log_path)
        except Exception:
            with self.lock:
                session = self.sessions().get(session_id)
                if session:
                    reset_queue_idle(session)
            return None
        with self.lock:
            session = self.sessions().get(session_id)
            if not session:
                return None
            queue_store = self.queue_store()
            if queue_store.queue_len(self.queues(), session_id) <= 0:
                reset_queue_idle(session)
                return None
            if session.queue_sending_item_id is not None:
                return None
            if queue_store.has_recovery_items(self.queues(), session_id):
                reset_queue_idle(session)
                return None
            head = queue_store.promotion_head(self.queues(), session_id, expected_item_id=expected_item_id)
            if head is None:
                return None
            head_id = head.item_id
            if not ready:
                reset_queue_idle(session)
                return None
            if not queue_idle_grace_ready(
                session,
                now_ts=float(now_ts),
                grace_seconds=self.queue_idle_grace_seconds,
                require_idle_grace=require_idle_grace,
            ):
                return None
            start_queue_promotion(session, head_id)
            text = queue_store.mark_promotion_commit_unknown(
                self.queues(),
                session_id,
                head_id,
                ts=self.now(),
            )
            if text is None:
                clear_queue_promotion(session, head_id)
                return None
        self.save_queues()
        try:
            resp = self.send(session_id, text, queue_item_id=head_id)
        except self.retryable_send_errors:
            with self.lock:
                session = self.sessions().get(session_id)
                if session:
                    clear_queue_promotion(session, head_id)
                self.queue_store().clear_commit_unknown_marker(self.queues(), session_id, head_id)
            self.save_queues()
            return None
        except self.commit_unknown_error:
            unknown_item: dict[str, Any] | None = None
            queue_len = 0
            with self.lock:
                session = self.sessions().get(session_id)
                if session:
                    clear_queue_promotion(session, head_id)
                unknown_item, queue_len = self.queue_store().preserve_commit_unknown_marker(
                    self.queues(),
                    session_id,
                    head_id,
                    ts=self.now(),
                )
            self.save_queues()
            return {"queued": True, "queue_len": int(queue_len), "item": unknown_item, "commit_unknown": True}
        except Exception:
            with self.lock:
                session = self.sessions().get(session_id)
                if session:
                    clear_queue_promotion(session, head_id)
                self.queue_store().clear_commit_unknown_marker(self.queues(), session_id, head_id)
            self.save_queues()
            return None
        with self.lock:
            session = self.sessions().get(session_id)
            if session:
                clear_queue_promotion(session, head_id)
            self.queue_store().pop_sent(self.queues(), session_id, head_id)
        self.save_queues()
        return resp
