from __future__ import annotations

import datetime
import hashlib
import json
import re
from typing import Any


_OAI_MEM_CITATION_TAIL_RE = re.compile(r"\s*<oai-mem-citation>\s*.*?</oai-mem-citation>\s*\Z", re.DOTALL)


def _parse_iso8601_to_epoch(ts: str) -> float | None:
    t = ts.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(t).timestamp()
    except ValueError:
        return None


def _event_ts(obj: dict[str, Any]) -> float | None:
    ts = obj.get("ts")
    if isinstance(ts, (int, float)):
        return float(ts)
    ts2 = obj.get("timestamp")
    if isinstance(ts2, (int, float)):
        return float(ts2)
    if isinstance(ts2, str):
        v = _parse_iso8601_to_epoch(ts2)
        if v is not None:
            return float(v)
    return None


def _strip_oai_mem_citation_tail(text: str) -> str:
    # Delivery notifications should follow the assistant reply itself, not the appended memory-citation envelope.
    return _OAI_MEM_CITATION_TAIL_RE.sub("", text)


def _codex_error_affects_turn_status(payload: dict[str, Any]) -> bool:
    info = payload.get("codex_error_info")
    if info == "thread_rollback_failed":
        return False
    return not (isinstance(info, dict) and "thread_rollback_failed" in info)


def _codex_event_text(payload: dict[str, Any]) -> str | None:
    msg = payload.get("message")
    if not isinstance(msg, str) or not msg.strip():
        return None
    details = payload.get("additional_details")
    if isinstance(details, str) and details.strip() and details.strip() != msg.strip():
        return f"{msg.strip()}\n\n{details.strip()}"
    return msg.strip()


def _codex_item_completed_item(obj: dict[str, Any]) -> dict[str, Any] | None:
    """Return a turn item from a persisted Codex ``item_completed`` row."""
    if obj.get("type") != "event_msg":
        return None
    payload = obj.get("payload")
    if not isinstance(payload, dict) or payload.get("type") != "item_completed":
        return None
    item = payload.get("item")
    return item if isinstance(item, dict) else None


def _codex_user_item_text(obj: dict[str, Any]) -> str | None:
    """Join the text chunks in a paginated Codex ``UserMessage`` item.

    Non-text inputs such as image and audio items do not appear as transcript
    text. Returning an empty string for a valid but text-empty user item lets
    callers distinguish it from other completed turn-item types while keeping
    it out of turn-boundary detection.
    """
    item = _codex_item_completed_item(obj)
    if item is None or item.get("type") != "UserMessage":
        return None
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        part["text"]
        for part in content
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
    )


def _text_message_id(*, message_class: str, text: str, ts: float | None) -> str:
    ts_ms = int(round(ts * 1000.0)) if isinstance(ts, (int, float)) else None
    payload = json.dumps({"class": message_class, "text": " ".join(text.split()), "ts_ms": ts_ms}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# User-facing text for a backend turn that was explicitly interrupted/aborted
# before completion. This is distinct from _NO_RESPONSE_TEXT (a turn that
# closed with no answer): an interruption means the turn was actively stopped.
# The word "interrupted" makes interruption rows searchable and keeps them
# distinct from generic no-response completions.
_INTERRUPTED_TEXT = "The backend turn was interrupted before completion."


def _build_interrupted_event(obj: dict[str, Any], *, partial_text: str | None = None) -> dict[str, Any]:
    """Build a persistent assistant transcript event for an interrupted turn.

    Used by both the Pi and Codex backend adapters so aborted/interrupted
    turns render the same normalized outcome across tail/history/live/search/
    export. ``message_class`` is "error" so the renderer surfaces it with error
    styling. When ``partial_text`` is present (a Pi aborted turn that already
    streamed some content) it is appended under a clear label so the partial
    output stays visible and searchable instead of being discarded.
    """
    ts = _event_ts(obj)
    text = _INTERRUPTED_TEXT
    if isinstance(partial_text, str) and partial_text.strip():
        text = f"{_INTERRUPTED_TEXT}\n\nPartial output before interruption:\n{partial_text.strip()}"
    event: dict[str, Any] = {
        "role": "assistant",
        "text": text,
        "message_class": "error",
        "message_id": _text_message_id(message_class="error", text=text, ts=ts),
    }
    if ts is not None:
        event["ts"] = ts
    return event
