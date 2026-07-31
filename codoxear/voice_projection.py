from __future__ import annotations

from typing import Any

from .voice_push_state import _compact_text


def voice_settings_snapshot_payload(
    *,
    voice_settings: dict[str, Any],
    subscriptions: dict[str, dict[str, Any]],
    queue_depth: int,
    active_listener_count: int,
    audio_state: dict[str, Any],
    vapid_public_key: str,
    redact_secrets: bool = False,
) -> dict[str, Any]:
    settings = dict(voice_settings)
    has_tts_api_key = bool(settings.get("tts_api_key"))
    if redact_secrets:
        settings["tts_api_key"] = ""
    settings["has_tts_api_key"] = has_tts_api_key
    enabled_devices = sum(
        1
        for item in subscriptions.values()
        if item.get("notifications_enabled") and item.get("device_class") == "mobile"
    )
    total_devices = sum(1 for item in subscriptions.values() if item.get("device_class") == "mobile")
    return {
        **settings,
        "audio": {
            "queue_depth": queue_depth,
            "active_listener_count": active_listener_count,
            "stream_url": "/api/audio/live.m3u8",
            **audio_state,
        },
        "notifications": {
            "enabled_devices": enabled_devices,
            "total_devices": total_devices,
            "vapid_public_key": vapid_public_key,
        },
    }


def subscriptions_snapshot_payload(
    *,
    subscriptions: dict[str, dict[str, Any]],
    vapid_public_key: str,
) -> dict[str, Any]:
    items = [
        {
            "id": record["id"],
            "endpoint": record["subscription"]["endpoint"],
            "notifications_enabled": bool(record.get("notifications_enabled")),
            "device_class": str(record.get("device_class") or "desktop"),
            "created_ts": record.get("created_ts"),
            "updated_ts": record.get("updated_ts"),
            "last_success_ts": record.get("last_success_ts"),
            "last_failure_ts": record.get("last_failure_ts"),
            "last_error": record.get("last_error"),
            "user_agent": record.get("user_agent"),
            "device_label": record.get("device_label"),
        }
        for record in subscriptions.values()
    ]
    items.sort(key=lambda item: float(item.get("updated_ts") or 0.0), reverse=True)
    return {"vapid_public_key": vapid_public_key, "subscriptions": items}


def notification_text_for_message(
    ledger: dict[str, dict[str, Any]],
    message_id: str,
) -> str | None:
    row = ledger.get(message_id)
    if not isinstance(row, dict):
        return None
    text = _compact_text(row.get("notification_text") or "")
    return text or None


def notification_state_for_message(
    ledger: dict[str, dict[str, Any]],
    message_id: str,
) -> dict[str, Any] | None:
    row = ledger.get(message_id)
    if not isinstance(row, dict):
        return None
    return {
        "message_id": message_id,
        "message_class": row.get("message_class"),
        "summary_status": row.get("summary_status"),
        "push_status": row.get("push_status"),
        "notification_text": _compact_text(row.get("notification_text") or ""),
    }


def notification_feed_since(
    ledger: dict[str, dict[str, Any]],
    since_ts: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in ledger.values():
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
        text = _compact_text(row.get("notification_text") or "")
        if not text:
            continue
        out.append(
            {
                "message_id": str(row.get("message_id") or ""),
                "session_id": str(row.get("session_id") or ""),
                "session_display_name": str(row.get("session_display_name") or "").strip() or "Session",
                "notification_text": text,
                "updated_ts": updated_ts,
            }
        )
    out.sort(key=lambda item: (float(item.get("updated_ts") or 0.0), str(item.get("message_id") or "")))
    return out
