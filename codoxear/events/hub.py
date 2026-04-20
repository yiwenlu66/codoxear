from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EventPollResult:
    events: list[dict[str, Any]]
    cursor_expired: bool
    closed: bool
    latest_seq: int


class EventHub:
    def __init__(self, *, max_events: int = 512) -> None:
        self._max_events = max(32, int(max_events))
        self._events: deque[dict[str, Any]] = deque()
        self._cond = threading.Condition()
        self._next_seq = 1
        self._closed = False
        self._last_emit_ts_by_key: dict[tuple[str, str], float] = {}

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def latest_seq(self) -> int:
        with self._cond:
            if not self._events:
                return max(0, self._next_seq - 1)
            return int(self._events[-1].get("seq") or 0)

    def publish(self, event: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(event)
        event_type = str(payload.get("type") or "").strip()
        if not event_type:
            raise ValueError("event type required")
        now_ts = float(payload.get("ts") or time.time())
        coalesce_ms = int(payload.pop("_coalesce_ms", 0) or 0)
        raw_key = payload.pop("_coalesce_key", None)
        if isinstance(raw_key, tuple) and len(raw_key) == 2:
            coalesce_key = (str(raw_key[0]), str(raw_key[1]))
        else:
            session_key = str(payload.get("session_id") or "")
            coalesce_key = (event_type, session_key)
        with self._cond:
            if self._closed:
                return None
            if coalesce_ms > 0:
                last_emit_ts = self._last_emit_ts_by_key.get(coalesce_key)
                if last_emit_ts is not None and ((now_ts - last_emit_ts) * 1000.0) < float(coalesce_ms):
                    return None
            stamped = dict(payload)
            stamped["seq"] = self._next_seq
            self._next_seq += 1
            stamped["type"] = event_type
            stamped["ts"] = now_ts
            self._events.append(stamped)
            while len(self._events) > self._max_events:
                self._events.popleft()
            if coalesce_ms > 0:
                self._last_emit_ts_by_key[coalesce_key] = now_ts
            self._cond.notify_all()
            return stamped

    def poll(self, after_seq: int, *, timeout_s: float) -> EventPollResult:
        target_seq = max(0, int(after_seq))
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._cond:
            while True:
                if self._closed:
                    return EventPollResult(events=[], cursor_expired=False, closed=True, latest_seq=self._latest_seq_locked())
                if self._events:
                    first_seq = int(self._events[0].get("seq") or 0)
                    latest_seq = int(self._events[-1].get("seq") or 0)
                    if target_seq < (first_seq - 1):
                        return EventPollResult(events=[], cursor_expired=True, closed=False, latest_seq=latest_seq)
                    if target_seq < latest_seq:
                        return EventPollResult(
                            events=[dict(item) for item in self._events if int(item.get("seq") or 0) > target_seq],
                            cursor_expired=False,
                            closed=False,
                            latest_seq=latest_seq,
                        )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return EventPollResult(events=[], cursor_expired=False, closed=False, latest_seq=self._latest_seq_locked())
                self._cond.wait(timeout=remaining)

    def _latest_seq_locked(self) -> int:
        if not self._events:
            return max(0, self._next_seq - 1)
        return int(self._events[-1].get("seq") or 0)
