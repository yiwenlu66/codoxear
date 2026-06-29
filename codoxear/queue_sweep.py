from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, MutableMapping

from .session_model import Session
from .queue_store import QueueStore


@dataclass(frozen=True)
class QueueSweepCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    queues: Callable[[], MutableMapping[str, list[dict[str, Any]]]]
    commit_unknown_sends: Callable[[], MutableMapping[str, Any]]
    queue_store: QueueStore
    discover_existing_if_stale: Callable[[], None]
    prune_dead_sessions: Callable[[], None]
    mark_queue_orphan_recovery_locked: Callable[[str], bool]
    save_queues: Callable[[], None]
    maybe_drain_session_queue: Callable[[str], bool]
    max_drains_per_sweep: int = 1

    def sweep(self) -> None:
        self.discover_existing_if_stale()
        self.prune_dead_sessions()
        with self.lock:
            active_ids = set(self.sessions().keys())
            direct_unknown_ids = set(self.commit_unknown_sends().keys())
            marked_recovery = False
            for sid, queue in list(self.queues().items()):
                if sid in active_ids or sid not in direct_unknown_ids or not isinstance(queue, list):
                    continue
                if self.mark_queue_orphan_recovery_locked(str(sid)):
                    marked_recovery = True
            dropped = self.queue_store.drop_missing_sessions(self.queues(), active_ids)
            session_ids = [sid for sid in self.queue_store.nonempty_session_ids(self.queues()) if sid in active_ids]
        if dropped or marked_recovery:
            self.save_queues()
        max_drains = max(1, int(self.max_drains_per_sweep))
        drained = 0
        for sid in session_ids:
            if self.maybe_drain_session_queue(sid):
                drained += 1
                if drained >= max_drains:
                    break
