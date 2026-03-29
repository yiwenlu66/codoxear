from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from py_vapid import Vapid
from pywebpush import WebPushException
from pywebpush import webpush


DEFAULT_SUMMARIZATION_MODEL = "gpt-4.1-mini"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_TTS_BASE_URL = "https://api.openai.com/v1"
DEFAULT_VAPID_SUBJECT = "https://localhost"
DEFAULT_VOICES = ("alloy", "ash", "ballad", "cedar", "coral", "echo", "fable", "marin", "nova", "onyx", "sage", "shimmer", "verse")
HLS_TARGET_DURATION_SECONDS = 12
HLS_MAX_SEGMENTS = 18
HLS_KEEPALIVE_SECONDS = 6.0
HLS_SILENCE_SECONDS = 6.0
LISTENER_TTL_SECONDS = 45.0
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


def _tailscale_https_subject() -> str | None:
    try:
        raw = subprocess.check_output(["tailscale", "status", "--json"], text=True, timeout=5.0)
        obj = json.loads(raw)
    except Exception:
        return None
    self_node = obj.get("Self")
    if not isinstance(self_node, dict):
        return None
    dns_name = str(self_node.get("DNSName") or "").strip().rstrip(".")
    if not dns_name:
        return None
    return f"https://{dns_name}"


def _default_vapid_subject() -> str:
    env_value = os.environ.get("CODEX_WEB_PUSH_VAPID_SUBJECT")
    if env_value:
        return _normalize_vapid_subject(env_value)
    tailscale_subject = _tailscale_https_subject()
    if tailscale_subject:
        return tailscale_subject
    return DEFAULT_VAPID_SUBJECT


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
    session_id: str
    session_display_name: str
    message_class: str
    spoken_text: str
    notification_text: str
    voice: str
    ts: float | None


@dataclass(frozen=True)
class GeneratedAnnouncement:
    task: AnnouncementTask
    audio_bytes: bytes


class OpenAICompatibleClient:
    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        self._timeout_seconds = float(timeout_seconds)

    def _request_json(self, *, base_url: str, api_key: str, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not api_key:
            raise ValueError("tts_api_key is required")
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            base_url.rstrip("/") + route,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{route} failed with {e.code}: {detail}") from e
        obj = json.loads(raw.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"{route} returned non-object json")
        return obj

    def _request_bytes(self, *, base_url: str, api_key: str, route: str, payload: dict[str, Any]) -> bytes:
        if not api_key:
            raise ValueError("tts_api_key is required")
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            base_url.rstrip("/") + route,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/octet-stream",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{route} failed with {e.code}: {detail}") from e

    def summarize(self, *, base_url: str, api_key: str, model: str, session_name: str, text: str) -> str:
        obj = self._request_json(
            base_url=base_url,
            api_key=api_key,
            route="/chat/completions",
            payload={
                "model": model,
                "temperature": 0.2,
                "max_completion_tokens": 90,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You write spoken mobile notifications. "
                            "Return one plain sentence of about 30 words. "
                            "Aim for roughly 24 to 36 words, no markdown, no quotes, no prefixes."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Session name: {session_name}\nFinal assistant response:\n{text}",
                    },
                ],
            },
        )
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("chat completions response missing choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError("chat completions response missing message")
        content = message.get("content")
        if isinstance(content, str):
            summary = " ".join(content.split()).strip()
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"} and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            summary = " ".join("".join(parts).split()).strip()
        else:
            raise ValueError("chat completions response missing content")
        if not summary:
            raise ValueError("empty summary response")
        return summary

    def synthesize(self, *, base_url: str, api_key: str, model: str, voice: str, text: str) -> bytes:
        audio = self._request_bytes(
            base_url=base_url,
            api_key=api_key,
            route="/audio/speech",
            payload={
                "model": model,
                "voice": voice,
                "input": text,
                "response_format": "aac",
            },
        )
        if not audio:
            raise ValueError("audio/speech returned empty body")
        return audio


class MergedHLSStream:
    def __init__(self, *, root_dir: Path) -> None:
        self._root_dir = Path(root_dir)
        self._segments_dir = self._root_dir / "segments"
        self._playlist_path = self._root_dir / "live.m3u8"
        self._lock = threading.Lock()
        self._segments: list[dict[str, Any]] = []
        self._next_seq = 1
        self._last_error = ""
        self._last_append_ts = 0.0
        os.makedirs(self._segments_dir, exist_ok=True)
        self._rewrite_playlist()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "segment_count": len(self._segments),
                "last_error": self._last_error,
                "media_sequence": self._segments[0]["seq"] if self._segments else self._next_seq,
            }

    def playlist_bytes(self) -> bytes:
        with self._lock:
            return self._playlist_path.read_bytes() if self._playlist_path.exists() else b"#EXTM3U\n"

    def segment_path(self, segment_name: str) -> Path:
        name = Path(segment_name).name
        if name != segment_name or not name.endswith(".ts"):
            raise FileNotFoundError(segment_name)
        path = (self._segments_dir / name).resolve()
        if not str(path).startswith(str(self._segments_dir.resolve())) or not path.exists():
            raise FileNotFoundError(segment_name)
        return path

    def append_audio(self, *, message_id: str, audio_bytes: bytes) -> float:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("ffmpeg and ffprobe are required for merged HLS output")
        os.makedirs(self._segments_dir, exist_ok=True)
        input_path = self._segments_dir / f"{message_id[:12] or 'audio'}.aac"
        input_path.write_bytes(audio_bytes)
        tmp_pattern = self._segments_dir / f"{message_id[:12] or 'audio'}-part-%03d.ts"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(input_path),
                    "-vn",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-f",
                    "segment",
                    "-segment_time",
                    "6",
                    "-segment_format",
                    "mpegts",
                    "-reset_timestamps",
                    "1",
                    str(tmp_pattern),
                ],
                check=True,
                capture_output=True,
            )
            total_duration = 0.0
            chunk_paths = sorted(self._segments_dir.glob(f"{message_id[:12] or 'audio'}-part-*.ts"))
            if not chunk_paths:
                raise RuntimeError("ffmpeg produced no HLS segments")
            for chunk_path in chunk_paths:
                try:
                    duration = self._segment_duration_seconds(chunk_path)
                except RuntimeError as e:
                    if "invalid ffprobe duration: N/A" not in str(e):
                        raise
                    try:
                        chunk_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                seq, segment_name, segment_path = self._reserve_segment(f"{message_id[:12] or 'audio'}")
                chunk_path.replace(segment_path)
                total_duration += duration
                self._store_segment(seq=seq, segment_name=segment_name, segment_path=segment_path, duration=duration)
            if total_duration <= 0.0:
                raise RuntimeError("ffmpeg produced no valid HLS segments")
        except subprocess.CalledProcessError as e:
            detail = e.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed: {detail}") from e
        finally:
            try:
                input_path.unlink()
            except FileNotFoundError:
                pass
            for chunk_path in self._segments_dir.glob(f"{message_id[:12] or 'audio'}-part-*.ts"):
                try:
                    chunk_path.unlink()
                except FileNotFoundError:
                    pass

        return total_duration

    def append_silence(self, *, force: bool = False) -> bool:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("ffmpeg and ffprobe are required for merged HLS output")
        with self._lock:
            if (not force) and self._last_append_ts and (time.time() - self._last_append_ts) < HLS_KEEPALIVE_SECONDS:
                return False
        os.makedirs(self._segments_dir, exist_ok=True)
        seq, segment_name, segment_path = self._reserve_segment("silence")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=24000:cl=mono",
                    "-t",
                    str(HLS_SILENCE_SECONDS),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "32k",
                    "-f",
                    "mpegts",
                    str(segment_path),
                ],
                check=True,
                capture_output=True,
            )
            duration = self._segment_duration_seconds(segment_path)
        except subprocess.CalledProcessError as e:
            detail = e.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed: {detail}") from e
        self._store_segment(seq=seq, segment_name=segment_name, segment_path=segment_path, duration=duration)
        return True

    def _reserve_segment(self, prefix: str) -> tuple[int, str, Path]:
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
        segment_name = f"{seq:06d}-{prefix[:12]}.ts"
        segment_path = self._segments_dir / segment_name
        return seq, segment_name, segment_path

    def _store_segment(self, *, seq: int, segment_name: str, segment_path: Path, duration: float) -> None:
        with self._lock:
            self._segments.append({"seq": seq, "name": segment_name, "duration": duration, "path": segment_path})
            self._segments.sort(key=lambda item: int(item["seq"]))
            while len(self._segments) > HLS_MAX_SEGMENTS:
                old = self._segments.pop(0)
                try:
                    Path(old["path"]).unlink()
                except FileNotFoundError:
                    pass
            self._last_append_ts = time.time()
            self._rewrite_playlist()

    def set_last_error(self, message: str) -> None:
        with self._lock:
            self._last_error = str(message or "").strip()

    def _segment_duration_seconds(self, segment_path: Path) -> float:
        try:
            raw = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(segment_path),
                ],
                text=True,
            ).strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffprobe failed: {e}") from e
        try:
            value = float(raw)
        except ValueError as e:
            raise RuntimeError(f"invalid ffprobe duration: {raw}") from e
        return max(0.2, value)

    def _rewrite_playlist(self) -> None:
        target_duration = max(
            HLS_TARGET_DURATION_SECONDS,
            int(math.ceil(max((float(item["duration"]) for item in self._segments), default=0.0))),
        )
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{target_duration}",
            f"#EXT-X-MEDIA-SEQUENCE:{self._segments[0]['seq'] if self._segments else self._next_seq}",
        ]
        for item in self._segments:
            lines.append(f"#EXTINF:{item['duration']:.3f},")
            lines.append(f"segments/{item['name']}")
        self._playlist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class VoicePushCoordinator:
    def __init__(
        self,
        *,
        app_dir: Path,
        stop_event: threading.Event,
        settings_path: Path,
        subscriptions_path: Path,
        delivery_ledger_path: Path,
        vapid_private_key_path: Path,
    ) -> None:
        self._app_dir = Path(app_dir)
        self._stop = stop_event
        self._settings_path = Path(settings_path)
        self._subscriptions_path = Path(subscriptions_path)
        self._delivery_ledger_path = Path(delivery_ledger_path)
        self._vapid_private_key_path = Path(vapid_private_key_path)
        self._hls = MergedHLSStream(root_dir=self._app_dir / "audio")
        self._client = OpenAICompatibleClient()
        self._lock = threading.Lock()
        self._queue_ready = threading.Condition(self._lock)
        self._queue: list[AnnouncementTask] = []
        self._listeners: dict[str, float] = {}
        self._generating_task: AnnouncementTask | None = None
        self._prepared: GeneratedAnnouncement | None = None
        self._playing_task: AnnouncementTask | None = None
        self._playing_until_monotonic = 0.0
        self._observed_serial = 0
        self._latest_observed_serial_by_session: dict[str, int] = {}
        self._voice_settings = _clean_voice_settings({})
        self._subscriptions: dict[str, dict[str, Any]] = {}
        self._delivery_ledger: dict[str, dict[str, Any]] = {}
        self._vapid_public_key = ""
        self._vapid_subject = _default_vapid_subject()
        self._load_settings()
        self._load_subscriptions()
        self._load_delivery_ledger()
        self._ensure_vapid_keys()
        self._worker = threading.Thread(target=self._worker_loop, name="voice-push", daemon=True)
        self._worker.start()
        self._keepalive = threading.Thread(target=self._keepalive_loop, name="voice-push-keepalive", daemon=True)
        self._keepalive.start()

    def settings_snapshot(self) -> dict[str, Any]:
        with self._lock:
            settings = dict(self._voice_settings)
            queue_depth = len(self._queue)
            enabled_devices = sum(1 for item in self._subscriptions.values() if item.get("notifications_enabled"))
            total_devices = len(self._subscriptions)
            active_listener_count = self._active_listener_count_locked(now_ts=time.time())
        audio_state = self._hls.snapshot()
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
                "vapid_public_key": self._vapid_public_key,
            },
        }

    def listener_heartbeat(self, *, client_id: str, enabled: bool) -> dict[str, Any]:
        cid = str(client_id or "").strip()
        if not cid:
            raise ValueError("client_id required")
        now_ts = time.time()
        with self._lock:
            self._prune_listeners_locked(now_ts=now_ts)
            if enabled:
                self._listeners[cid] = now_ts
            else:
                self._listeners.pop(cid, None)
            count = self._active_listener_count_locked(now_ts=now_ts)
            self._queue_ready.notify_all()
        return {"active_listener_count": count}

    def set_settings(self, raw: Any) -> dict[str, Any]:
        settings = _clean_voice_settings(raw)
        with self._lock:
            self._voice_settings = settings
            self._queue_ready.notify_all()
        self._save_settings()
        return self.settings_snapshot()

    def subscriptions_snapshot(self) -> dict[str, Any]:
        with self._lock:
            items = [
                {
                    "id": record["id"],
                    "endpoint": record["subscription"]["endpoint"],
                    "notifications_enabled": bool(record.get("notifications_enabled")),
                    "created_ts": record.get("created_ts"),
                    "updated_ts": record.get("updated_ts"),
                    "last_success_ts": record.get("last_success_ts"),
                    "last_failure_ts": record.get("last_failure_ts"),
                    "last_error": record.get("last_error"),
                    "user_agent": record.get("user_agent"),
                    "device_label": record.get("device_label"),
                }
                for record in self._subscriptions.values()
            ]
        items.sort(key=lambda item: float(item.get("updated_ts") or 0.0), reverse=True)
        return {"vapid_public_key": self._vapid_public_key, "subscriptions": items}

    def upsert_subscription(self, *, subscription: Any, user_agent: str, device_label: str | None = None) -> dict[str, Any]:
        cleaned = _clean_subscription(subscription)
        now_ts = float(time.time())
        sid = _subscription_id(cleaned)
        with self._lock:
            current = dict(self._subscriptions.get(sid) or {})
            record = {
                "id": sid,
                "subscription": cleaned,
                "notifications_enabled": bool(current.get("notifications_enabled", True)),
                "created_ts": float(current.get("created_ts", now_ts)),
                "updated_ts": now_ts,
                "last_success_ts": current.get("last_success_ts"),
                "last_failure_ts": current.get("last_failure_ts"),
                "last_error": str(current.get("last_error") or "").strip(),
                "user_agent": str(user_agent or "").strip(),
                "device_label": str(device_label or "").strip(),
            }
            self._subscriptions[sid] = record
        self._save_subscriptions()
        return self.subscriptions_snapshot()

    def toggle_subscription(self, *, endpoint: str, enabled: bool) -> dict[str, Any]:
        endpoint_clean = str(endpoint or "").strip()
        if not endpoint_clean:
            raise ValueError("endpoint required")
        target_id = _subscription_id({"endpoint": endpoint_clean, "keys": {"p256dh": "x", "auth": "x"}})
        with self._lock:
            record = self._subscriptions.get(target_id)
            if not isinstance(record, dict) or record.get("subscription", {}).get("endpoint") != endpoint_clean:
                raise KeyError("unknown subscription")
            record["notifications_enabled"] = bool(enabled)
            record["updated_ts"] = float(time.time())
            self._subscriptions[target_id] = record
        self._save_subscriptions()
        return self.subscriptions_snapshot()

    def observe_messages(
        self,
        *,
        session_id: str,
        session_display_name: str,
        messages: list[ClassifiedAssistantMessage],
    ) -> None:
        for msg in messages:
            task: AnnouncementTask | None = None
            now_ts = float(time.time())
            observed_serial = 0
            with self._lock:
                if msg.message_id in self._delivery_ledger:
                    continue
                self._observed_serial += 1
                observed_serial = self._observed_serial
                self._latest_observed_serial_by_session[session_id] = observed_serial
                self._delivery_ledger[msg.message_id] = {
                    "message_id": msg.message_id,
                    "session_id": session_id,
                    "session_display_name": session_display_name,
                    "message_class": msg.message_class,
                    "preview_text": _clip_text(msg.text, limit=160),
                    "notification_text": "",
                    "summary_text": "",
                    "summary_status": "pending" if msg.message_class == "final_response" else "skipped",
                    "narrated_status": "pending",
                    "push_status": "pending" if msg.message_class == "final_response" else "skipped",
                    "voice": "",
                    "created_ts": now_ts,
                    "updated_ts": now_ts,
                    "last_error": "",
                }
                if msg.message_class != "final_response" and (not self._voice_settings.get("tts_enabled_for_narration")):
                    row = self._delivery_ledger.get(msg.message_id)
                    if isinstance(row, dict):
                        row["narrated_status"] = "skipped"
                        row["updated_ts"] = now_ts
                        self._delivery_ledger[msg.message_id] = row
                self._trim_locked()
            self._save_delivery_ledger()
            if msg.message_class == "final_response":
                task = self._prepare_final_response(
                    message=msg,
                    session_id=session_id,
                    session_display_name=session_display_name,
                )
            elif self._voice_settings.get("tts_enabled_for_narration"):
                task = AnnouncementTask(
                    message_id=msg.message_id,
                    session_id=session_id,
                    session_display_name=session_display_name,
                    message_class=msg.message_class,
                    spoken_text=f"From {session_display_name}. {_compact_text(msg.text)}",
                    notification_text="",
                    voice=self._voice_for_session(session_id, session_display_name),
                    ts=msg.ts,
                )
            if task is None:
                continue
            with self._lock:
                if msg.message_class == "final_response" and self._latest_observed_serial_by_session.get(task.session_id) != observed_serial:
                    self._mark_task_replaced_locked(task)
                else:
                    self._replace_queued_session_tasks_locked(session_id=task.session_id)
                    self._queue.append(task)
                self._trim_locked()
                self._queue_ready.notify_all()
            self._save_delivery_ledger()

    def playlist_bytes(self) -> bytes:
        return self._hls.playlist_bytes()

    def segment_path(self, segment_name: str) -> Path:
        return self._hls.segment_path(segment_name)

    def notification_text_for_message(self, message_id: str) -> str | None:
        with self._lock:
            row = self._delivery_ledger.get(message_id)
            if not isinstance(row, dict):
                return None
            text = _compact_text(row.get("notification_text") or "")
            return text or None

    def notification_state_for_message(self, message_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._delivery_ledger.get(message_id)
            if not isinstance(row, dict):
                return None
            return {
                "message_id": message_id,
                "message_class": row.get("message_class"),
                "summary_status": row.get("summary_status"),
                "push_status": row.get("push_status"),
                "notification_text": _compact_text(row.get("notification_text") or ""),
            }

    def notification_feed_since(self, since_ts: float) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._delivery_ledger.values())
        out: list[dict[str, Any]] = []
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

    def _keepalive_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._keepalive_sweep()
            except Exception as e:
                self._hls.set_last_error(str(e))
            self._stop.wait(1.0)

    def _keepalive_sweep(self) -> None:
        with self._lock:
            listener_count = self._active_listener_count_locked(now_ts=time.time())
            should_keepalive = (
                listener_count > 0
                and not self._queue
                and self._generating_task is None
                and self._prepared is None
                and self._playing_task is None
            )
        if not should_keepalive:
            return
        self._hls.append_silence(force=False)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            action = ""
            task: AnnouncementTask | None = None
            prepared: GeneratedAnnouncement | None = None
            with self._lock:
                while not self._stop.is_set():
                    now_wall = time.time()
                    now_mono = time.monotonic()
                    listener_count = self._active_listener_count_locked(now_ts=now_wall)
                    if self._playing_task is not None and now_mono >= self._playing_until_monotonic:
                        self._playing_task = None
                        self._playing_until_monotonic = 0.0
                    if listener_count > 0 and self._prepared is not None and self._playing_task is None:
                        prepared = self._prepared
                        self._prepared = None
                        action = "append"
                        break
                    if listener_count > 0 and self._generating_task is None and self._prepared is None and self._queue:
                        task = self._queue.pop(0)
                        self._generating_task = task
                        action = "generate"
                        break
                    timeout = 0.25
                    if self._playing_task is not None:
                        timeout = min(timeout, max(0.0, self._playing_until_monotonic - now_mono))
                    self._queue_ready.wait(timeout=timeout)
                if self._stop.is_set():
                    return
            if action == "append" and prepared is not None:
                self._append_prepared(prepared)
                continue
            if action != "generate" or task is None:
                continue
            try:
                self._process_task(task)
            except Exception as e:
                self._set_ledger_error(task.message_id, str(e))
                self._hls.set_last_error(str(e))
            finally:
                with self._lock:
                    if self._generating_task is not None and self._generating_task.message_id == task.message_id:
                        self._generating_task = None
                    self._queue_ready.notify_all()

    def _process_task(self, task: AnnouncementTask) -> None:
        settings = self.settings_snapshot()
        self._set_ledger_field(task.message_id, "voice", task.voice)
        audio = self._client.synthesize(
            base_url=settings["tts_base_url"],
            api_key=settings["tts_api_key"],
            model=settings["tts_model"],
            voice=task.voice,
            text=task.spoken_text,
        )
        with self._lock:
            self._prepared = GeneratedAnnouncement(task=task, audio_bytes=audio)
            self._queue_ready.notify_all()

    def _append_prepared(self, prepared: GeneratedAnnouncement) -> None:
        duration = self._hls.append_audio(message_id=prepared.task.message_id, audio_bytes=prepared.audio_bytes)
        self._hls.set_last_error("")
        with self._lock:
            self._playing_task = prepared.task
            self._playing_until_monotonic = time.monotonic() + max(0.2, float(duration))
            self._queue_ready.notify_all()
        self._set_ledger_field(prepared.task.message_id, "narrated_status", "sent")

    def _prepare_final_response(
        self,
        *,
        message: ClassifiedAssistantMessage,
        session_id: str,
        session_display_name: str,
    ) -> AnnouncementTask | None:
        source_text = _compact_text(message.text)
        settings = self.settings_snapshot()
        summary_text = ""
        notification_text = _clip_text(source_text, limit=120)
        if settings.get("tts_api_key"):
            try:
                summary_text = self._client.summarize(
                    base_url=settings["tts_base_url"],
                    api_key=settings["tts_api_key"],
                    model=settings["summarization_model"],
                    session_name=session_display_name,
                    text=message.text,
                )
            except Exception as e:
                self._set_ledger_fields(
                    message.message_id,
                    {
                        "summary_status": "error",
                        "push_status": "error",
                        "narrated_status": "error" if settings.get("tts_enabled_for_final_response") else "skipped",
                        "last_error": _clip_text(str(e), limit=400),
                    },
                )
                self._hls.set_last_error(str(e))
                return None
            self._set_ledger_fields(
                message.message_id,
                {
                    "notification_text": notification_text,
                    "summary_status": "sent",
                    "summary_text": summary_text,
                },
            )
            notification_text = _clip_text(_compact_text(summary_text), limit=120)
            self._set_ledger_field(message.message_id, "notification_text", notification_text)
        else:
            self._set_ledger_fields(
                message.message_id,
                {
                    "summary_status": "skipped",
                    "notification_text": notification_text,
                },
            )
        self._send_push_notifications(
            session_id=session_id,
            session_display_name=session_display_name,
            message_id=message.message_id,
            notification_text=notification_text,
            timestamp=message.ts,
        )
        if not settings.get("tts_enabled_for_final_response"):
            self._set_ledger_field(message.message_id, "narrated_status", "skipped")
            return None
        if not settings.get("tts_api_key"):
            self._set_ledger_fields(
                message.message_id,
                {
                    "narrated_status": "error",
                    "last_error": "tts_api_key is required",
                },
            )
            return None
        spoken_basis = summary_text or source_text
        return AnnouncementTask(
            message_id=message.message_id,
            session_id=session_id,
            session_display_name=session_display_name,
            message_class="final_response",
            spoken_text=f"Turn summary from {session_display_name}. {spoken_basis}",
            notification_text=notification_text,
            voice=self._voice_for_session(session_id, session_display_name),
            ts=message.ts,
        )

    def _send_push_notifications(self, *, session_id: str, session_display_name: str, message_id: str, notification_text: str, timestamp: float | None) -> None:
        with self._lock:
            subscriptions = [dict(item) for item in self._subscriptions.values() if item.get("notifications_enabled")]
        if not subscriptions:
            self._set_ledger_field(message_id, "push_status", "skipped")
            return
        vapid = Vapid.from_file(str(self._vapid_private_key_path))
        payload = json.dumps(
            {
                "session_id": session_id,
                "session_display_name": session_display_name,
                "message_id": message_id,
                "notification_text": notification_text,
                "timestamp": timestamp or time.time(),
            }
        )
        any_success = False
        for record in subscriptions:
            try:
                response = webpush(
                    subscription_info=record["subscription"],
                    data=payload,
                    vapid_private_key=vapid,
                    vapid_claims={"sub": self._vapid_subject},
                    ttl=300,
                    timeout=10.0,
                )
                now_ts = float(time.time())
                with self._lock:
                    current = self._subscriptions.get(record["id"])
                    if isinstance(current, dict):
                        current["last_success_ts"] = now_ts
                        current["last_error"] = ""
                        current["updated_ts"] = now_ts
                        self._subscriptions[record["id"]] = current
                any_success = True
                _ = response
            except WebPushException as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                self._mark_subscription_failure(record_id=record["id"], error=str(e))
                if status in {404, 410}:
                    self._drop_subscription(record["id"])
        self._save_subscriptions()
        self._set_ledger_field(message_id, "push_status", "sent" if any_success else "error")

    def _voice_for_session(self, session_id: str, session_name: str) -> str:
        token = _sha256_hex(session_id)
        return DEFAULT_VOICES[int(token[:8], 16) % len(DEFAULT_VOICES)]

    def _mark_task_replaced_locked(self, task: AnnouncementTask) -> None:
        row = self._delivery_ledger.get(task.message_id)
        if not isinstance(row, dict):
            return
        row["last_error"] = "replaced by newer message"
        row["updated_ts"] = float(time.time())
        row["narrated_status"] = "skipped"
        if task.message_class == "final_response":
            if row.get("summary_status") == "pending":
                row["summary_status"] = "skipped"
            if row.get("push_status") == "pending":
                row["push_status"] = "skipped"
        self._delivery_ledger[task.message_id] = row

    def _replace_queued_session_tasks_locked(self, *, session_id: str) -> None:
        if not self._queue:
            return
        kept: list[AnnouncementTask] = []
        for task in self._queue:
            if task.session_id == session_id:
                self._mark_task_replaced_locked(task)
                continue
            kept.append(task)
        self._queue = kept

    def _set_ledger_error(self, message_id: str, error: str) -> None:
        with self._lock:
            row = self._delivery_ledger.get(message_id)
            if not isinstance(row, dict):
                return
            row["last_error"] = _clip_text(error, limit=400)
            row["updated_ts"] = float(time.time())
            row["narrated_status"] = "error"
            if row.get("message_class") == "final_response":
                if row.get("summary_status") == "pending":
                    row["summary_status"] = "error"
                if row.get("push_status") == "pending":
                    row["push_status"] = "error"
            self._delivery_ledger[message_id] = row
        self._save_delivery_ledger()

    def _set_ledger_field(self, message_id: str, key: str, value: Any) -> None:
        self._set_ledger_fields(message_id, {key: value})

    def _set_ledger_fields(self, message_id: str, patch: dict[str, Any]) -> None:
        with self._lock:
            row = self._delivery_ledger.get(message_id)
            if not isinstance(row, dict):
                return
            row.update(patch)
            row["updated_ts"] = float(time.time())
            self._delivery_ledger[message_id] = row
        self._save_delivery_ledger()

    def _mark_subscription_failure(self, *, record_id: str, error: str) -> None:
        with self._lock:
            record = self._subscriptions.get(record_id)
            if not isinstance(record, dict):
                return
            now_ts = float(time.time())
            record["last_failure_ts"] = now_ts
            record["last_error"] = _clip_text(error, limit=400)
            record["updated_ts"] = now_ts
            self._subscriptions[record_id] = record

    def _drop_subscription(self, record_id: str) -> None:
        with self._lock:
            self._subscriptions.pop(record_id, None)

    def _prune_listeners_locked(self, *, now_ts: float) -> None:
        stale = [cid for cid, seen_at in self._listeners.items() if (now_ts - float(seen_at)) > LISTENER_TTL_SECONDS]
        for cid in stale:
            self._listeners.pop(cid, None)

    def _active_listener_count_locked(self, *, now_ts: float) -> int:
        self._prune_listeners_locked(now_ts=now_ts)
        return len(self._listeners)

    def _ensure_vapid_keys(self) -> None:
        if self._vapid_private_key_path.exists():
            vapid = Vapid.from_file(str(self._vapid_private_key_path))
        else:
            vapid = Vapid()
            vapid.generate_keys()
            os.makedirs(self._vapid_private_key_path.parent, exist_ok=True)
            self._vapid_private_key_path.write_bytes(vapid.private_pem())
        public_bytes = vapid.public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint,
        )
        self._vapid_public_key = _b64u(public_bytes)

    def _trim_locked(self) -> None:
        if len(self._delivery_ledger) <= DELIVERY_LEDGER_MAX:
            return
        doomed = sorted(
            self._delivery_ledger.values(),
            key=lambda row: float(row.get("updated_ts") or row.get("created_ts") or 0.0),
        )[: len(self._delivery_ledger) - DELIVERY_LEDGER_MAX]
        for row in doomed:
            message_id = row.get("message_id")
            if isinstance(message_id, str):
                self._delivery_ledger.pop(message_id, None)

    def _load_settings(self) -> None:
        try:
            raw = json.loads(self._settings_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raw = {}
        self._voice_settings = _clean_voice_settings(raw)

    def _save_settings(self) -> None:
        os.makedirs(self._settings_path.parent, exist_ok=True)
        with self._lock:
            payload = dict(self._voice_settings)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self._settings_path.parent, prefix=self._settings_path.name + ".", suffix=".tmp", delete=False) as tmp:
            tmp.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, self._settings_path)

    def _load_subscriptions(self) -> None:
        try:
            raw = json.loads(self._subscriptions_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raw = []
        cleaned: dict[str, dict[str, Any]] = {}
        if isinstance(raw, list):
            for item in raw:
                record = _clean_subscription_record(item)
                if record is not None:
                    cleaned[record["id"]] = record
        self._subscriptions = cleaned

    def _save_subscriptions(self) -> None:
        os.makedirs(self._subscriptions_path.parent, exist_ok=True)
        with self._lock:
            payload = list(self._subscriptions.values())
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self._subscriptions_path.parent, prefix=self._subscriptions_path.name + ".", suffix=".tmp", delete=False) as tmp:
            tmp.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, self._subscriptions_path)

    def _load_delivery_ledger(self) -> None:
        try:
            raw = json.loads(self._delivery_ledger_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raw = {}
        self._delivery_ledger = _clean_ledger(raw)

    def _save_delivery_ledger(self) -> None:
        os.makedirs(self._delivery_ledger_path.parent, exist_ok=True)
        with self._lock:
            self._trim_locked()
            payload = dict(self._delivery_ledger)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self._delivery_ledger_path.parent, prefix=self._delivery_ledger_path.name + ".", suffix=".tmp", delete=False) as tmp:
            tmp.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, self._delivery_ledger_path)
