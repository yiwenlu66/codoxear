from __future__ import annotations

import math
import secrets
import time
from pathlib import Path
from typing import Any, Iterable

from .util import atomic_write_json
from .util import load_json_file

QueueMap = dict[str, list[dict[str, Any]]]


def new_queue_item_id() -> str:
    return f"queue-{secrets.token_hex(8)}"


def new_queue_item(text: str, *, created_ts: float | None = None) -> dict[str, Any]:
    ts = float(created_ts) if created_ts is not None else time.time()
    if not math.isfinite(ts) or ts <= 0:
        ts = time.time()
    return {"id": new_queue_item_id(), "text": str(text), "created_ts": ts}


def copy_queue_item(item: dict[str, Any]) -> dict[str, Any]:
    created_ts = item.get("created_ts")
    try:
        ts = float(created_ts)
    except (TypeError, ValueError):
        ts = time.time()
    if not math.isfinite(ts) or ts <= 0:
        ts = time.time()
    out: dict[str, Any] = {"id": str(item.get("id") or ""), "text": str(item.get("text") or ""), "created_ts": ts}
    if bool(item.get("commit_unknown")):
        out["commit_unknown"] = True
        commit_unknown_ts = item.get("commit_unknown_ts")
        try:
            unknown_ts = float(commit_unknown_ts)
        except (TypeError, ValueError):
            unknown_ts = time.time()
        if not math.isfinite(unknown_ts) or unknown_ts <= 0:
            unknown_ts = time.time()
        out["commit_unknown_ts"] = unknown_ts
    return out


def coerce_queue_item(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        return new_queue_item(raw)
    if not isinstance(raw, dict):
        return None
    item_id = raw.get("id")
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(item_id, str) or not item_id.strip():
        item_id = new_queue_item_id()
    created_ts = raw.get("created_ts")
    try:
        ts = float(created_ts)
    except (TypeError, ValueError):
        ts = time.time()
    if not math.isfinite(ts) or ts <= 0:
        ts = time.time()
    out: dict[str, Any] = {"id": item_id.strip(), "text": text, "created_ts": ts}
    if bool(raw.get("commit_unknown")):
        out["commit_unknown"] = True
        commit_unknown_ts = raw.get("commit_unknown_ts")
        try:
            unknown_ts = float(commit_unknown_ts)
        except (TypeError, ValueError):
            unknown_ts = time.time()
        if not math.isfinite(unknown_ts) or unknown_ts <= 0:
            unknown_ts = time.time()
        out["commit_unknown_ts"] = unknown_ts
    return out


class QueueStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> QueueMap:
        obj = load_json_file(self.path, default=None)
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise ValueError("invalid session_queues.json (expected object)")
        cleaned: QueueMap = {}
        for sid, arr in obj.items():
            if not isinstance(sid, str) or not sid:
                continue
            if not isinstance(arr, list):
                continue
            out: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for value in arr:
                item = coerce_queue_item(value)
                if item is None:
                    continue
                item_id = str(item["id"])
                if item_id in seen_ids:
                    item["id"] = new_queue_item_id()
                    item_id = str(item["id"])
                seen_ids.add(item_id)
                out.append(item)
            if out:
                cleaned[sid] = out
        return cleaned

    def save(self, queues: QueueMap) -> None:
        obj = {
            sid: [copy_queue_item(item) for item in items]
            for sid, items in queues.items()
            if isinstance(items, list) and items
        }
        atomic_write_json(self.path, obj)

    def queue_len(self, queues: QueueMap, session_id: str) -> int:
        q = queues.get(session_id)
        return int(len(q)) if isinstance(q, list) else 0

    def list_items(self, queues: QueueMap, session_id: str, *, sending_item_id: str | None = None) -> list[dict[str, Any]]:
        q = queues.get(session_id)
        if not isinstance(q, list) or not q:
            return []
        out: list[dict[str, Any]] = []
        for item in q:
            copied = copy_queue_item(item)
            copied["sending"] = bool(sending_item_id and copied["id"] == sending_item_id)
            copied["commit_unknown"] = bool(copied.get("commit_unknown")) and not bool(copied["sending"])
            out.append(copied)
        return out

    def append(self, queues: QueueMap, session_id: str, text: str) -> tuple[dict[str, Any], int]:
        value = str(text)
        if not value.strip():
            raise ValueError("text required")
        item = new_queue_item(value)
        q = queues.get(session_id)
        if not isinstance(q, list):
            q = []
            queues[session_id] = q
        q.append(item)
        return copy_queue_item(item), len(q)

    def delete(self, queues: QueueMap, session_id: str, item_id: str, *, sending_item_id: str | None = None) -> int:
        item_id_clean = str(item_id).strip()
        if not item_id_clean:
            raise ValueError("id required")
        if sending_item_id == item_id_clean:
            raise ValueError("item is already sending")
        q = queues.get(session_id)
        if not isinstance(q, list):
            q = []
            queues[session_id] = q
        idx = next((i for i, item in enumerate(q) if item.get("id") == item_id_clean), -1)
        if idx < 0:
            raise ValueError("item not found")
        q.pop(idx)
        ql = len(q)
        if not q:
            queues.pop(session_id, None)
        return ql

    def update(self, queues: QueueMap, session_id: str, item_id: str, text: str, *, sending_item_id: str | None = None) -> tuple[dict[str, Any], int]:
        item_id_clean = str(item_id).strip()
        value = str(text)
        if not item_id_clean:
            raise ValueError("id required")
        if not value.strip():
            raise ValueError("text required")
        if sending_item_id == item_id_clean:
            raise ValueError("item is already sending")
        q = queues.get(session_id)
        if not isinstance(q, list):
            q = []
            queues[session_id] = q
        idx = next((i for i, item in enumerate(q) if item.get("id") == item_id_clean), -1)
        if idx < 0:
            raise ValueError("item not found")
        q[idx]["text"] = value
        return copy_queue_item(q[idx]), len(q)

    def move(self, queues: QueueMap, session_id: str, item_id: str, to_index: int, *, sending_item_id: str | None = None) -> int:
        item_id_clean = str(item_id).strip()
        if not item_id_clean:
            raise ValueError("id required")
        if isinstance(to_index, bool):
            raise ValueError("to_index must be an integer")
        target = int(to_index)
        q = queues.get(session_id)
        if not isinstance(q, list):
            q = []
            queues[session_id] = q
        if not q:
            raise ValueError("item not found")
        idx = next((i for i, item in enumerate(q) if item.get("id") == item_id_clean), -1)
        if idx < 0:
            raise ValueError("item not found")
        if sending_item_id == item_id_clean:
            raise ValueError("item is already sending")
        if bool(q[idx].get("commit_unknown")):
            raise ValueError("item commit status is unknown")
        min_index = 1 if sending_item_id else 0
        barrier_idx = next((i for i, item in enumerate(q) if bool(item.get("commit_unknown"))), None)
        if barrier_idx is not None and idx > barrier_idx and target <= barrier_idx:
            raise ValueError("commit-unknown item blocks reordering")
        if target < min_index or target >= len(q):
            raise ValueError("to_index out of range")
        item = q.pop(idx)
        q.insert(target, item)
        return len(q)

    def pop_sent(self, queues: QueueMap, session_id: str, item_id: str) -> None:
        q = queues.get(session_id)
        if not isinstance(q, list):
            return
        idx = next((i for i, item in enumerate(q) if item.get("id") == item_id), -1)
        if idx >= 0:
            q.pop(idx)
        if not q:
            queues.pop(session_id, None)

    def drop_missing_sessions(self, queues: QueueMap, active_session_ids: Iterable[str]) -> bool:
        active = set(active_session_ids)
        drop = [sid for sid in queues.keys() if sid not in active]
        for sid in drop:
            queues.pop(sid, None)
        return bool(drop)

    def nonempty_session_ids(self, queues: QueueMap) -> list[str]:
        return [sid for sid, q in queues.items() if isinstance(q, list) and q]
