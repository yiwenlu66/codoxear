from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SUMMARIZATION_MODEL = "gpt-4.1-mini"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_TTS_BASE_URL = "https://api.openai.com/v1"
DEFAULT_VAPID_SUBJECT = "https://localhost"
DEFAULT_VOICES = ("alloy", "ash", "ballad", "cedar", "coral", "echo", "fable", "marin", "nova", "onyx", "sage", "shimmer", "verse")
DELIVERY_LEDGER_MAX = 4000


def _sha256_hex(raw: str | bytes) -> str:
    data = raw.encode("utf-8") if isinstance(raw, str) else raw
    return hashlib.sha256(data).hexdigest()


def _clip_text(raw: str, *, limit: int) -> str:
    text = " ".join(str(raw or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "..."


def _compact_text(raw: str) -> str:
    return " ".join(str(raw or "").split()).strip()


def _normalize_base_url(raw: Any) -> str:
    value = str(raw or "").strip() or DEFAULT_TTS_BASE_URL
    if not value.startswith(("http://", "https://")):
        raise ValueError("tts_base_url must start with http:// or https://")
    return value.rstrip("/")


def _normalize_vapid_subject(raw: Any) -> str:
    value = str(raw or "").strip()
    if not value:
        raise ValueError("empty vapid subject")
    if value.startswith("mailto:"):
        return value
    if value.startswith(("http://", "https://")):
        return value.rstrip("/")
    raise ValueError("vapid subject must start with https://, http://, or mailto:")


def _chmod_private_file(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _clean_voice_settings(raw: Any) -> dict[str, Any]:
    obj = dict(raw) if isinstance(raw, dict) else {}
    narration = bool(obj.get("tts_enabled_for_narration"))
    final_response = bool(obj.get("tts_enabled_for_final_response"))
    base_url = _normalize_base_url(obj.get("tts_base_url"))
    api_key = str(obj.get("tts_api_key") or "").strip()
    summarization_model = str(obj.get("summarization_model") or DEFAULT_SUMMARIZATION_MODEL).strip() or DEFAULT_SUMMARIZATION_MODEL
    tts_model = str(obj.get("tts_model") or DEFAULT_TTS_MODEL).strip() or DEFAULT_TTS_MODEL
    return {
        "tts_enabled_for_narration": narration,
        "tts_enabled_for_final_response": final_response,
        "tts_base_url": base_url,
        "tts_api_key": api_key,
        "summarization_model": summarization_model,
        "tts_model": tts_model,
    }


def _subscription_id(subscription: dict[str, Any]) -> str:
    endpoint = str(subscription.get("endpoint") or "").strip()
    return _sha256_hex(endpoint)[:24]


def _clean_subscription(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("subscription must be an object")
    endpoint = str(raw.get("endpoint") or "").strip()
    keys = raw.get("keys")
    if not endpoint:
        raise ValueError("subscription endpoint required")
    if not isinstance(keys, dict):
        raise ValueError("subscription keys required")
    p256dh = str(keys.get("p256dh") or "").strip()
    auth = str(keys.get("auth") or "").strip()
    if not p256dh or not auth:
        raise ValueError("subscription keys.p256dh and keys.auth required")
    return {"endpoint": endpoint, "keys": {"p256dh": p256dh, "auth": auth}}


def _device_class_from_user_agent(raw: Any) -> str:
    ua = str(raw or "").strip().lower()
    if "mobile" in ua or "android" in ua or "iphone" in ua or "ipad" in ua or "ipod" in ua:
        return "mobile"
    return "desktop"


def _clean_device_class(raw: Any, *, user_agent: str) -> str:
    value = str(raw or "").strip().lower()
    if value in {"mobile", "desktop"}:
        return value
    return _device_class_from_user_agent(user_agent)


def _clean_subscription_record(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    try:
        subscription = _clean_subscription(raw.get("subscription"))
    except ValueError:
        return None
    now_ts = float(time.time())
    enabled = bool(raw.get("notifications_enabled", True))
    created_ts = float(raw.get("created_ts", now_ts))
    updated_ts = float(raw.get("updated_ts", created_ts))
    last_success_ts = raw.get("last_success_ts")
    last_failure_ts = raw.get("last_failure_ts")
    last_error = str(raw.get("last_error") or "").strip()
    user_agent = str(raw.get("user_agent") or "").strip()
    device_label = str(raw.get("device_label") or "").strip()
    device_class = _clean_device_class(raw.get("device_class"), user_agent=user_agent)
    return {
        "id": _subscription_id(subscription),
        "subscription": subscription,
        "notifications_enabled": enabled,
        "created_ts": created_ts,
        "updated_ts": updated_ts,
        "last_success_ts": float(last_success_ts) if isinstance(last_success_ts, (int, float)) else None,
        "last_failure_ts": float(last_failure_ts) if isinstance(last_failure_ts, (int, float)) else None,
        "last_error": last_error,
        "user_agent": user_agent,
        "device_label": device_label,
        "device_class": device_class,
    }


def _clean_ledger(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, dict[str, Any]] = {}
    for message_id, row in raw.items():
        if not isinstance(message_id, str) or not message_id:
            continue
        if not isinstance(row, dict):
            continue
        session_id = str(row.get("session_id") or "").strip()
        message_class = str(row.get("message_class") or "").strip()
        if not session_id or message_class not in {"narration", "final_response"}:
            continue
        cleaned[message_id] = {
            "message_id": message_id,
            "session_id": session_id,
            "session_display_name": str(row.get("session_display_name") or "").strip(),
            "message_class": message_class,
            "preview_text": str(row.get("preview_text") or "").strip(),
            "notification_text": str(row.get("notification_text") or "").strip(),
            "summary_text": str(row.get("summary_text") or "").strip(),
            "summary_status": str(row.get("summary_status") or "pending"),
            "narrated_status": str(row.get("narrated_status") or "pending"),
            "push_status": str(row.get("push_status") or "pending"),
            "voice": str(row.get("voice") or "").strip(),
            "created_ts": float(row.get("created_ts") or time.time()),
            "updated_ts": float(row.get("updated_ts") or time.time()),
            "last_error": str(row.get("last_error") or "").strip(),
        }
    return cleaned


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


@dataclass(frozen=True)
class ClassifiedAssistantMessage:
    message_id: str
    message_class: str
    text: str
    ts: float | None


@dataclass(frozen=True)
class AnnouncementTask:
    message_id: str
    source_message_ids: tuple[str, ...]
    session_id: str
    session_display_name: str
    message_class: str
    source_text: str
    spoken_text: str
    notification_text: str
    voice: str
    ts: float | None
    summary_word_target: int | None
    listener_epoch: int


@dataclass(frozen=True)
class GeneratedAnnouncement:
    task: AnnouncementTask
    audio_bytes: bytes
