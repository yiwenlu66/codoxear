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


def _text_message_id(*, message_class: str, text: str, ts: float | None) -> str:
    ts_ms = int(round(ts * 1000.0)) if isinstance(ts, (int, float)) else None
    payload = json.dumps({"class": message_class, "text": " ".join(text.split()), "ts_ms": ts_ms}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
