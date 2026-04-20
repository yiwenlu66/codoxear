from __future__ import annotations

from typing import Any

from .model import AttentionItem


def compact_notification_state(row: dict[str, Any], compact_text) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    message_id = str(row.get("message_id") or "").strip()
    if not message_id:
        return None
    return {
        "message_id": message_id,
        "message_class": row.get("message_class"),
        "summary_status": row.get("summary_status"),
        "push_status": row.get("push_status"),
        "notification_text": compact_text(row.get("notification_text") or ""),
    }



def final_response_attention_feed(
    rows: list[dict[str, Any]], *, since_ts: float, compact_text
) -> list[dict[str, Any]]:
    out: list[AttentionItem] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("message_class") != "final_response":
            continue
        updated_ts = float(row.get("updated_ts") or 0.0)
        if updated_ts <= float(since_ts):
            continue
        summary_status = str(row.get("summary_status") or "")
        if summary_status not in {"sent", "skipped", "error"}:
            continue
        text = compact_text(row.get("notification_text") or "")
        if not text:
            continue
        out.append(
            AttentionItem(
                message_id=str(row.get("message_id") or ""),
                session_id=str(row.get("session_id") or ""),
                session_display_name=str(row.get("session_display_name") or "").strip() or "Session",
                notification_text=text,
                updated_ts=updated_ts,
                message_class="final_response",
                summary_status=summary_status,
            )
        )
    out.sort(key=lambda item: (float(item.updated_ts), item.message_id))
    return [
        {
            "message_id": item.message_id,
            "session_id": item.session_id,
            "session_display_name": item.session_display_name,
            "notification_text": item.notification_text,
            "updated_ts": item.updated_ts,
        }
        for item in out
    ]
