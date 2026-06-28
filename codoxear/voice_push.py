from __future__ import annotations

import json
import os
import shutil  # retained as a module-level patch seam for voice_push.MergedHLSStream compatibility
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .voice_persistence import load_subscription_records
from .voice_persistence import load_voice_delivery_ledger
from .voice_persistence import load_voice_settings
from .voice_persistence import save_subscription_records
from .voice_persistence import save_voice_delivery_ledger
from .voice_persistence import save_voice_settings
from .voice_projection import notification_feed_since as _notification_feed_since_payload
from .voice_projection import notification_state_for_message as _notification_state_for_message_payload
from .voice_projection import notification_text_for_message as _notification_text_for_message_payload
from .voice_projection import subscriptions_snapshot_payload
from .voice_projection import voice_settings_snapshot_payload
from .voice_push_state import AnnouncementTask
from .voice_push_state import ClassifiedAssistantMessage
from .voice_push_state import DEFAULT_SUMMARIZATION_MODEL
from .voice_push_state import DEFAULT_TTS_BASE_URL
from .voice_push_state import DEFAULT_TTS_MODEL
from .voice_push_state import DEFAULT_VAPID_SUBJECT
from .voice_push_state import DEFAULT_VOICES
from .voice_push_state import DELIVERY_LEDGER_MAX
from .voice_push_state import GeneratedAnnouncement
from .voice_push_state import _b64u
from .voice_push_state import _chmod_private_file
from .voice_push_state import _clean_device_class
from .voice_push_state import _clean_ledger
from .voice_push_state import _clean_subscription
from .voice_push_state import _clean_subscription_record
from .voice_push_state import _clean_voice_settings
from .voice_push_state import _clip_text
from .voice_push_state import _compact_text
from .voice_push_state import _normalize_vapid_subject
from .voice_push_state import _sha256_hex
from .voice_push_state import _subscription_id
from .voice_openai_client import OpenAICompatibleClient
from .voice_hls import HLS_KEEPALIVE_SECONDS
from .voice_hls import HLS_MAX_SEGMENTS
from .voice_hls import HLS_SILENCE_SECONDS
from .voice_hls import HLS_TARGET_DURATION_SECONDS
from .voice_hls import MergedHLSStream
from .voice_ledger import mark_task_replaced
from .voice_ledger import mark_tasks_skipped_no_listener
from .voice_ledger import set_ledger_fields_many
from .voice_ledger import set_task_error
from .voice_ledger import trim_delivery_ledger
from .voice_task_queue import enqueue_announcement_task
from .voice_task_queue import voice_for_session
from .voice_webpush import ensure_vapid_public_key
from .voice_webpush import push_payload_json
from .voice_webpush import send_web_push_notifications


LISTENER_TTL_SECONDS = 45.0


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
        self._listener_epoch = 0
        self._observed_serial = 0
        self._latest_observed_serial_by_slot: dict[tuple[str, str], int] = {}
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

    def settings_snapshot(self, *, redact_secrets: bool = False) -> dict[str, Any]:
        with self._lock:
            settings = dict(self._voice_settings)
            subscriptions = dict(self._subscriptions)
            queue_depth = len(self._queue)
            active_listener_count = self._active_listener_count_locked(now_ts=time.time())
        return voice_settings_snapshot_payload(
            voice_settings=settings,
            subscriptions=subscriptions,
            queue_depth=queue_depth,
            active_listener_count=active_listener_count,
            audio_state=self._hls.snapshot(),
            vapid_public_key=self._vapid_public_key,
            redact_secrets=redact_secrets,
        )

    def listener_heartbeat(self, *, client_id: str, enabled: bool) -> dict[str, Any]:
        cid = str(client_id or "").strip()
        if not cid:
            raise ValueError("client_id required")
        now_ts = time.time()
        dropped_tasks: list[AnnouncementTask] = []
        should_reset_hls = False
        with self._lock:
            self._prune_listeners_locked(now_ts=now_ts)
            previous_count = len(self._listeners)
            if enabled:
                self._listeners[cid] = now_ts
            else:
                self._listeners.pop(cid, None)
            count = self._active_listener_count_locked(now_ts=now_ts)
            if previous_count > 0 and count == 0:
                self._listener_epoch += 1
                dropped_tasks.extend(self._queue)
                self._queue = []
                if self._prepared is not None:
                    dropped_tasks.append(self._prepared.task)
                    self._prepared = None
                if self._generating_task is not None:
                    dropped_tasks.append(self._generating_task)
                    self._generating_task = None
                self._playing_task = None
                self._playing_until_monotonic = 0.0
                should_reset_hls = True
            self._queue_ready.notify_all()
        if dropped_tasks:
            self._mark_tasks_skipped_no_listener(dropped_tasks)
        if should_reset_hls:
            self._hls.reset()
        return {"active_listener_count": count}

    def set_settings(
        self,
        raw: Any,
        *,
        preserve_blank_api_key: bool = False,
        redact_response: bool = False,
    ) -> dict[str, Any]:
        obj = dict(raw) if isinstance(raw, dict) else {}
        clear_api_key = bool(obj.get("tts_api_key_clear"))
        if clear_api_key:
            obj["tts_api_key"] = ""
        elif preserve_blank_api_key:
            candidate = str(obj.get("tts_api_key") or "").strip()
            if not candidate:
                with self._lock:
                    obj["tts_api_key"] = str(self._voice_settings.get("tts_api_key") or "")
        settings = _clean_voice_settings(obj)
        with self._lock:
            self._voice_settings = settings
            self._queue_ready.notify_all()
        self._save_settings()
        return self.settings_snapshot(redact_secrets=redact_response)

    def subscriptions_snapshot(self) -> dict[str, Any]:
        with self._lock:
            subscriptions = dict(self._subscriptions)
        return subscriptions_snapshot_payload(subscriptions=subscriptions, vapid_public_key=self._vapid_public_key)

    def upsert_subscription(
        self,
        *,
        subscription: Any,
        user_agent: str,
        device_label: str | None = None,
        device_class: str | None = None,
    ) -> dict[str, Any]:
        cleaned = _clean_subscription(subscription)
        now_ts = float(time.time())
        sid = _subscription_id(cleaned)
        user_agent_clean = str(user_agent or "").strip()
        device_class_clean = _clean_device_class(device_class, user_agent=user_agent_clean)
        with self._lock:
            current = dict(self._subscriptions.get(sid) or {})
            record = {
                "id": sid,
                "subscription": cleaned,
                "notifications_enabled": True,
                "created_ts": float(current.get("created_ts", now_ts)),
                "updated_ts": now_ts,
                "last_success_ts": current.get("last_success_ts"),
                "last_failure_ts": current.get("last_failure_ts"),
                "last_error": str(current.get("last_error") or "").strip(),
                "user_agent": user_agent_clean,
                "device_label": str(device_label or "").strip(),
                "device_class": device_class_clean,
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
            slot_key = (session_id, msg.message_class)
            narration_enabled = bool(self._voice_settings.get("tts_enabled_for_narration"))
            listener_epoch = 0
            listener_count = 0
            with self._lock:
                if msg.message_id in self._delivery_ledger:
                    continue
                self._observed_serial += 1
                observed_serial = self._observed_serial
                self._latest_observed_serial_by_slot[slot_key] = observed_serial
                listener_epoch = self._listener_epoch
                listener_count = self._active_listener_count_locked(now_ts=now_ts)
                self._delivery_ledger[msg.message_id] = {
                    "message_id": msg.message_id,
                    "session_id": session_id,
                    "session_display_name": session_display_name,
                    "message_class": msg.message_class,
                    "preview_text": _clip_text(msg.text, limit=160),
                    "notification_text": "",
                    "summary_text": "",
                    "summary_status": "pending" if (msg.message_class == "final_response" or narration_enabled) else "skipped",
                    "narrated_status": "pending" if (msg.message_class == "final_response" or narration_enabled) else "skipped",
                    "push_status": "pending" if msg.message_class == "final_response" else "skipped",
                    "voice": "",
                    "created_ts": now_ts,
                    "updated_ts": now_ts,
                    "last_error": "",
                }
                self._trim_locked()
            self._save_delivery_ledger()
            if msg.message_class == "final_response":
                task = self._prepare_final_response(
                    message=msg,
                    session_id=session_id,
                    session_display_name=session_display_name,
                    listener_epoch=listener_epoch,
                )
            elif narration_enabled:
                task = AnnouncementTask(
                    message_id=msg.message_id,
                    source_message_ids=(msg.message_id,),
                    session_id=session_id,
                    session_display_name=session_display_name,
                    message_class=msg.message_class,
                    source_text=_compact_text(msg.text),
                    spoken_text="",
                    notification_text="",
                    voice=self._voice_for_session(session_id, session_display_name),
                    ts=msg.ts,
                    summary_word_target=15,
                    listener_epoch=listener_epoch,
                )
            if task is None:
                continue
            if listener_count <= 0:
                self._mark_tasks_skipped_no_listener([task])
                continue
            with self._lock:
                current_listener_count = self._active_listener_count_locked(now_ts=time.time())
                if current_listener_count <= 0 or task.listener_epoch != self._listener_epoch:
                    drop_for_listener = True
                else:
                    drop_for_listener = False
                if drop_for_listener:
                    pass
                elif msg.message_class == "final_response" and self._latest_observed_serial_by_slot.get(slot_key) != observed_serial:
                    self._mark_task_replaced_locked(task)
                else:
                    self._enqueue_task_locked(task)
                self._trim_locked()
                self._queue_ready.notify_all()
            if drop_for_listener:
                self._mark_tasks_skipped_no_listener([task])
            self._save_delivery_ledger()

    def playlist_bytes(self) -> bytes:
        return self._hls.playlist_bytes()

    def segment_path(self, segment_name: str) -> Path:
        return self._hls.segment_path(segment_name)

    def notification_text_for_message(self, message_id: str) -> str | None:
        with self._lock:
            ledger = dict(self._delivery_ledger)
        return _notification_text_for_message_payload(ledger, message_id)

    def notification_state_for_message(self, message_id: str) -> dict[str, Any] | None:
        with self._lock:
            ledger = dict(self._delivery_ledger)
        return _notification_state_for_message_payload(ledger, message_id)

    def notification_feed_since(self, since_ts: float) -> list[dict[str, Any]]:
        with self._lock:
            ledger = dict(self._delivery_ledger)
        return _notification_feed_since_payload(ledger, since_ts)

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
            stale_task: AnnouncementTask | None = None
            with self._lock:
                while not self._stop.is_set():
                    now_wall = time.time()
                    now_mono = time.monotonic()
                    listener_count = self._active_listener_count_locked(now_ts=now_wall)
                    if self._playing_task is not None and now_mono >= self._playing_until_monotonic:
                        self._playing_task = None
                        self._playing_until_monotonic = 0.0
                    if (
                        listener_count > 0
                        and self._prepared is not None
                        and self._prepared.task.listener_epoch == self._listener_epoch
                        and self._playing_task is None
                    ):
                        prepared = self._prepared
                        self._prepared = None
                        action = "append"
                        break
                    if listener_count > 0 and self._prepared is not None and self._prepared.task.listener_epoch != self._listener_epoch:
                        stale_task = self._prepared.task
                        self._prepared = None
                        self._queue_ready.notify_all()
                        continue
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
            if stale_task is not None:
                self._mark_tasks_skipped_no_listener([stale_task])
                continue
            if action != "generate" or task is None:
                continue
            try:
                self._process_task(task)
            except Exception as e:
                self._set_task_error(task, str(e))
                self._hls.set_last_error(str(e))
            finally:
                with self._lock:
                    if self._generating_task is not None and self._generating_task.message_id == task.message_id:
                        self._generating_task = None
                    self._queue_ready.notify_all()

    def _process_task(self, task: AnnouncementTask) -> None:
        settings = self.settings_snapshot()
        self._set_task_ledger_fields(task, {"voice": task.voice})
        spoken_text = task.spoken_text
        if task.summary_word_target is not None:
            compact_source = _compact_text(task.source_text)
            source_word_count = len(compact_source.split())
            if source_word_count and source_word_count < int(task.summary_word_target):
                self._set_task_ledger_fields(task, {"summary_status": "skipped", "summary_text": ""})
                spoken_text = f"From {task.session_display_name}. {compact_source}"
            else:
                summary_text = self._client.summarize(
                    base_url=settings["tts_base_url"],
                    api_key=settings["tts_api_key"],
                    model=settings["summarization_model"],
                    session_name=task.session_display_name,
                    source_label="Narration updates",
                    text=task.source_text,
                    target_words=task.summary_word_target,
                )
                self._set_task_ledger_fields(task, {"summary_status": "sent", "summary_text": summary_text})
                spoken_text = f"From {task.session_display_name}. {summary_text}"
        with self._lock:
            if task.listener_epoch != self._listener_epoch or self._active_listener_count_locked(now_ts=time.time()) <= 0:
                stale = True
            else:
                stale = False
            self._queue_ready.notify_all()
        if stale:
            self._mark_tasks_skipped_no_listener([task])
            return
        audio = self._client.synthesize(
            base_url=settings["tts_base_url"],
            api_key=settings["tts_api_key"],
            model=settings["tts_model"],
            voice=task.voice,
            text=spoken_text,
        )
        with self._lock:
            if task.listener_epoch != self._listener_epoch or self._active_listener_count_locked(now_ts=time.time()) <= 0:
                self._queue_ready.notify_all()
                stale = True
            else:
                self._prepared = GeneratedAnnouncement(task=task, audio_bytes=audio)
                self._queue_ready.notify_all()
                stale = False
        if stale:
            self._mark_tasks_skipped_no_listener([task])
            return

    def _append_prepared(self, prepared: GeneratedAnnouncement) -> None:
        with self._lock:
            if prepared.task.listener_epoch != self._listener_epoch or self._active_listener_count_locked(now_ts=time.time()) <= 0:
                stale = True
            else:
                stale = False
        if stale:
            self._mark_tasks_skipped_no_listener([prepared.task])
            return
        duration = self._hls.append_audio(message_id=prepared.task.message_id, audio_bytes=prepared.audio_bytes)
        self._hls.set_last_error("")
        with self._lock:
            self._playing_task = prepared.task
            self._playing_until_monotonic = time.monotonic() + max(0.2, float(duration))
            self._queue_ready.notify_all()
        self._set_task_ledger_fields(prepared.task, {"narrated_status": "sent"})

    def _prepare_final_response(
        self,
        *,
        message: ClassifiedAssistantMessage,
        session_id: str,
        session_display_name: str,
        listener_epoch: int,
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
                    source_label="Final assistant response",
                    text=message.text,
                    target_words=30,
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
            source_message_ids=(message.message_id,),
            session_id=session_id,
            session_display_name=session_display_name,
            message_class="final_response",
            source_text=source_text,
            spoken_text=f"Turn summary from {session_display_name}. {spoken_basis}",
            notification_text=notification_text,
            voice=self._voice_for_session(session_id, session_display_name),
            ts=message.ts,
            summary_word_target=None,
            listener_epoch=listener_epoch,
        )

    def _send_push_notifications(self, *, session_id: str, session_display_name: str, message_id: str, notification_text: str, timestamp: float | None) -> None:
        with self._lock:
            subscriptions = [
                dict(item)
                for item in self._subscriptions.values()
                if item.get("notifications_enabled") and item.get("device_class") == "mobile"
            ]
        if not subscriptions:
            self._set_ledger_field(message_id, "push_status", "skipped")
            return
        payload = push_payload_json(
            session_id=session_id,
            session_display_name=session_display_name,
            message_id=message_id,
            notification_text=notification_text,
            timestamp=timestamp,
        )
        outcomes = send_web_push_notifications(
            subscriptions=subscriptions,
            private_key_path=self._vapid_private_key_path,
            vapid_subject=self._vapid_subject,
            payload_json=payload,
        )
        any_success = False
        for outcome in outcomes:
            if outcome.success:
                self._mark_subscription_success(record_id=outcome.record_id, now_ts=outcome.timestamp)
                any_success = True
                continue
            self._mark_subscription_failure(record_id=outcome.record_id, error=outcome.error)
            if outcome.drop_subscription:
                self._drop_subscription(outcome.record_id)
        self._save_subscriptions()
        self._set_ledger_field(message_id, "push_status", "sent" if any_success else "error")

    def _voice_for_session(self, session_id: str, session_name: str) -> str:
        return voice_for_session(session_id)

    def _mark_task_replaced_locked(self, task: AnnouncementTask) -> None:
        mark_task_replaced(self._delivery_ledger, task, now_ts=float(time.time()))

    def _enqueue_task_locked(self, new_task: AnnouncementTask) -> None:
        update = enqueue_announcement_task(self._queue, new_task)
        for replaced_task in update.replaced_tasks:
            self._mark_task_replaced_locked(replaced_task)
        self._queue = update.queue

    def _set_task_error(self, task: AnnouncementTask, error: str) -> None:
        with self._lock:
            set_task_error(self._delivery_ledger, task, error, now_ts=float(time.time()))
        self._save_delivery_ledger()

    def _mark_tasks_skipped_no_listener(self, tasks: list[AnnouncementTask]) -> None:
        with self._lock:
            dirty = mark_tasks_skipped_no_listener(self._delivery_ledger, tasks, now_ts=float(time.time()))
        if dirty:
            self._save_delivery_ledger()

    def _set_ledger_field(self, message_id: str, key: str, value: Any) -> None:
        self._set_ledger_fields(message_id, {key: value})

    def _set_task_ledger_fields(self, task: AnnouncementTask, patch: dict[str, Any]) -> None:
        self._set_ledger_fields_many(task.source_message_ids, patch)

    def _set_ledger_fields(self, message_id: str, patch: dict[str, Any]) -> None:
        self._set_ledger_fields_many((message_id,), patch)

    def _set_ledger_fields_many(self, message_ids: tuple[str, ...], patch: dict[str, Any]) -> None:
        with self._lock:
            dirty = set_ledger_fields_many(self._delivery_ledger, message_ids, patch, now_ts=float(time.time()))
        if not dirty:
            return
        self._save_delivery_ledger()

    def _mark_subscription_success(self, *, record_id: str, now_ts: float) -> None:
        with self._lock:
            record = self._subscriptions.get(record_id)
            if not isinstance(record, dict):
                return
            record["last_success_ts"] = now_ts
            record["last_error"] = ""
            record["updated_ts"] = now_ts
            self._subscriptions[record_id] = record

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
        self._vapid_public_key = ensure_vapid_public_key(self._vapid_private_key_path)

    def _trim_locked(self) -> None:
        trim_delivery_ledger(self._delivery_ledger, limit=DELIVERY_LEDGER_MAX)

    def _load_settings(self) -> None:
        self._voice_settings = load_voice_settings(self._settings_path)

    def _save_settings(self) -> None:
        with self._lock:
            payload = dict(self._voice_settings)
        save_voice_settings(self._settings_path, payload)

    def _load_subscriptions(self) -> None:
        self._subscriptions = load_subscription_records(self._subscriptions_path)

    def _save_subscriptions(self) -> None:
        with self._lock:
            payload = dict(self._subscriptions)
        save_subscription_records(self._subscriptions_path, payload)

    def _load_delivery_ledger(self) -> None:
        self._delivery_ledger = load_voice_delivery_ledger(self._delivery_ledger_path)

    def _save_delivery_ledger(self) -> None:
        with self._lock:
            self._trim_locked()
            payload = dict(self._delivery_ledger)
        save_voice_delivery_ledger(self._delivery_ledger_path, payload)
