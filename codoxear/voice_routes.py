from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import urllib.parse


JsonResponse = Callable[[Any, int, dict[str, Any]], None]
ReadJsonBody = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class VoiceRouteDeps:
    require_auth: Callable[[Any], bool]
    json_response: JsonResponse
    read_json_body: ReadJsonBody
    load_unattended_prompt: Callable[[], str]
    save_unattended_prompt: Callable[[str], str]
    default_unattended_prompt: str


def handle_voice_get_route(
    handler: Any,
    *,
    path: str,
    query: str,
    voice_push: Any,
    deps: VoiceRouteDeps,
) -> bool:
    if path == "/api/settings/voice":
        if not _authorized(handler, deps):
            return True
        deps.json_response(handler, 200, {"ok": True, **voice_push.settings_snapshot(redact_secrets=True)})
        return True

    if path == "/api/settings/unattended-prompt":
        if not _authorized(handler, deps):
            return True
        deps.json_response(
            handler,
            200,
            {"ok": True, "prompt": deps.load_unattended_prompt(), "default_prompt": deps.default_unattended_prompt},
        )
        return True

    if path == "/api/notifications/subscription":
        if not _authorized(handler, deps):
            return True
        deps.json_response(handler, 200, {"ok": True, **voice_push.subscriptions_snapshot()})
        return True

    if path == "/api/notifications/message":
        if not _authorized(handler, deps):
            return True
        qs = urllib.parse.parse_qs(query)
        message_id = (qs.get("message_id") or [""])[0].strip()
        if not message_id:
            deps.json_response(handler, 400, {"error": "message_id required"})
            return True
        state = voice_push.notification_state_for_message(message_id)
        if state is None:
            deps.json_response(handler, 404, {"error": "unknown message"})
            return True
        deps.json_response(handler, 200, {"ok": True, **state})
        return True

    if path == "/api/notifications/feed":
        if not _authorized(handler, deps):
            return True
        qs = urllib.parse.parse_qs(query)
        since_raw = (qs.get("since") or ["0"])[0].strip()
        try:
            since_ts = float(since_raw or "0")
        except ValueError:
            deps.json_response(handler, 400, {"error": "invalid since"})
            return True
        items = voice_push.notification_feed_since(since_ts)
        deps.json_response(handler, 200, {"ok": True, "items": items})
        return True

    if path == "/api/audio/live.m3u8":
        if not _authorized(handler, deps):
            return True
        _write_no_store_bytes(handler, content_type="application/vnd.apple.mpegurl", body=voice_push.playlist_bytes())
        return True

    if path.startswith("/api/audio/segments/"):
        if not _authorized(handler, deps):
            return True
        segment_name = path.split("/api/audio/segments/", 1)[1]
        try:
            segment_path = voice_push.segment_path(segment_name)
        except FileNotFoundError:
            handler.send_error(404)
            return True
        raw = Path(segment_path).read_bytes()
        _write_no_store_bytes(handler, content_type="video/mp2t", body=raw)
        return True

    return False


def handle_voice_post_route(
    handler: Any,
    *,
    path: str,
    voice_push: Any,
    deps: VoiceRouteDeps,
) -> bool:
    if path == "/api/settings/voice":
        if not _authorized(handler, deps):
            return True
        obj = deps.read_json_body(handler)
        try:
            payload = voice_push.set_settings(obj, preserve_blank_api_key=True, redact_response=True)
        except ValueError as e:
            deps.json_response(handler, 400, {"error": str(e)})
            return True
        deps.json_response(handler, 200, {"ok": True, **payload})
        return True

    if path == "/api/settings/unattended-prompt":
        if not _authorized(handler, deps):
            return True
        obj = deps.read_json_body(handler)
        prompt = obj.get("prompt")
        if not isinstance(prompt, str):
            deps.json_response(handler, 400, {"error": "prompt must be a string"})
            return True
        try:
            saved = deps.save_unattended_prompt(prompt)
        except ValueError as e:
            deps.json_response(handler, 400, {"error": str(e)})
            return True
        deps.json_response(handler, 200, {"ok": True, "prompt": saved, "default_prompt": deps.default_unattended_prompt})
        return True

    if path == "/api/notifications/subscription":
        if not _authorized(handler, deps):
            return True
        obj = deps.read_json_body(handler)
        try:
            payload = voice_push.upsert_subscription(
                subscription=obj.get("subscription"),
                user_agent=str(obj.get("user_agent") or ""),
                device_label=str(obj.get("device_label") or ""),
                device_class=str(obj.get("device_class") or ""),
            )
        except ValueError as e:
            deps.json_response(handler, 400, {"error": str(e)})
            return True
        deps.json_response(handler, 200, {"ok": True, **payload})
        return True

    if path == "/api/notifications/subscription/toggle":
        if not _authorized(handler, deps):
            return True
        obj = deps.read_json_body(handler)
        endpoint = obj.get("endpoint")
        enabled = obj.get("enabled")
        if not isinstance(endpoint, str) or not endpoint.strip():
            deps.json_response(handler, 400, {"error": "endpoint required"})
            return True
        if not isinstance(enabled, bool):
            deps.json_response(handler, 400, {"error": "enabled must be a boolean"})
            return True
        try:
            payload = voice_push.toggle_subscription(endpoint=endpoint, enabled=enabled)
        except KeyError:
            deps.json_response(handler, 404, {"error": "unknown subscription"})
            return True
        except ValueError as e:
            deps.json_response(handler, 400, {"error": str(e)})
            return True
        deps.json_response(handler, 200, {"ok": True, **payload})
        return True

    if path == "/api/audio/listener":
        if not _authorized(handler, deps):
            return True
        obj = deps.read_json_body(handler)
        client_id = obj.get("client_id")
        enabled = obj.get("enabled")
        if not isinstance(client_id, str) or not client_id.strip():
            deps.json_response(handler, 400, {"error": "client_id required"})
            return True
        if not isinstance(enabled, bool):
            deps.json_response(handler, 400, {"error": "enabled must be a boolean"})
            return True
        payload = voice_push.listener_heartbeat(client_id=client_id, enabled=enabled)
        deps.json_response(handler, 200, {"ok": True, **payload})
        return True

    return False


def _authorized(handler: Any, deps: VoiceRouteDeps) -> bool:
    if deps.require_auth(handler):
        return True
    handler._unauthorized()
    return False


def _write_no_store_bytes(handler: Any, *, content_type: str, body: bytes) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Expires", "0")
    handler.end_headers()
    handler.wfile.write(body)
