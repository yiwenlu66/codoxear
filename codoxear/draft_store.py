from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from .util import atomic_write_json
from .util import load_json_file


DraftMap = dict[str, dict[str, Any]]

DRAFT_MAX_BYTES = 256 * 1024


def draft_byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def coerce_draft_entry(raw: Any) -> dict[str, Any] | None:
    """Validate one persisted draft entry, preserving tombstones.

    An entry is ``{text: str, updated_ts: float}``. Empty text is a
    tombstone — a first-class deletion event that must survive load/save so
    it propagates through the same last-writer-wins channel as edits.
    Whitespace-only text normalizes to the canonical empty-text tombstone.
    """
    if not isinstance(raw, dict):
        return None
    text = raw.get("text")
    if not isinstance(text, str):
        return None
    updated_raw = raw.get("updated_ts")
    if isinstance(updated_raw, bool):
        updated_ts = time.time()
    else:
        try:
            updated_ts = float(updated_raw)
        except (TypeError, ValueError):
            updated_ts = time.time()
    if not math.isfinite(updated_ts) or updated_ts <= 0:
        updated_ts = time.time()
    return {"text": text if text.strip() else "", "updated_ts": updated_ts}


def _resolve_updated_ts(now_ts: float | None) -> float:
    ts = time.time() if now_ts is None or isinstance(now_ts, bool) else float(now_ts)
    if not math.isfinite(ts) or ts <= 0:
        ts = time.time()
    return ts


class DraftStore:
    """Path-bound persistence for per-session composer drafts.

    State shape: ``{session_id: {"text": str, "updated_ts": float}}``.
    Both text and tombstone writes persist entries; the write boundary
    (``set``) enforces the 256 KiB cap (blank clears bypass it) and records
    deletion as a timestamped empty-text tombstone. Entry removal happens
    only via session-deletion cleanup — a vanished entry cannot propagate a
    deletion through the last-writer-wins channel.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def load(self) -> DraftMap:
        if self.path is None:
            return {}
        obj = load_json_file(self.path, default=None)
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise ValueError("invalid session_drafts.json (expected object)")
        cleaned: DraftMap = {}
        for sid, raw in obj.items():
            if not isinstance(sid, str) or not sid.strip():
                continue
            entry = coerce_draft_entry(raw)
            if entry is not None:
                cleaned[sid.strip()] = entry
        return cleaned

    def save(self, drafts: DraftMap) -> None:
        if self.path is None:
            return
        obj: DraftMap = {}
        for sid, raw in drafts.items():
            if not isinstance(sid, str) or not sid.strip():
                continue
            entry = coerce_draft_entry(raw)
            if entry is not None:
                obj[sid.strip()] = entry
        atomic_write_json(self.path, obj)

    def public_entry(self, entry: Any) -> dict[str, Any]:
        if not isinstance(entry, dict):
            return {"text": "", "updated_ts": 0.0}
        text = entry.get("text")
        updated_raw = entry.get("updated_ts")
        if isinstance(updated_raw, bool):
            updated_ts = 0.0
        else:
            try:
                updated_ts = float(updated_raw)
            except (TypeError, ValueError):
                updated_ts = 0.0
        if not math.isfinite(updated_ts) or updated_ts <= 0:
            updated_ts = 0.0
        return {"text": text if isinstance(text, str) else "", "updated_ts": updated_ts}

    def updated_ts(self, drafts: Any, session_id: str) -> float:
        if not isinstance(drafts, dict):
            return 0.0
        return self.public_entry(drafts.get(session_id))["updated_ts"]

    def tombstone(self, drafts: DraftMap, session_id: str, *, now_ts: float | None = None) -> float:
        """Record a session's draft deletion and return its timestamp.

        Deletion is a first-class event: other clients hold companion
        server timestamps > 0, so removing the entry would look older than
        their local copy and the cleared draft would be re-pushed
        (resurrection). The empty-text entry with the server wall clock
        wins over any earlier edit timestamp.
        """
        ts = _resolve_updated_ts(now_ts)
        drafts[session_id] = {"text": "", "updated_ts": ts}
        return ts

    def set(self, drafts: DraftMap, session_id: str, text: str, *, now_ts: float | None = None) -> float:
        """Write a session's draft in place and return its timestamp.

        Empty/whitespace-only text writes a tombstone entry with ``now_ts``
        (server wall clock) — bypassing the byte cap, since clearing can
        never exceed it — and returns that timestamp; otherwise the text is
        stored under ``now_ts`` and that timestamp is returned.
        """
        value = str(text)
        if not value.strip():
            return self.tombstone(drafts, session_id, now_ts=now_ts)
        try:
            encoded_len = draft_byte_len(value)
        except UnicodeEncodeError as exc:
            raise ValueError("draft text must be valid UTF-8") from exc
        if encoded_len > DRAFT_MAX_BYTES:
            raise ValueError(f"draft text exceeds the {DRAFT_MAX_BYTES} byte limit")
        ts = _resolve_updated_ts(now_ts)
        drafts[session_id] = {"text": value, "updated_ts": ts}
        return ts
