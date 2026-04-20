from __future__ import annotations

import json
import urllib.parse

_SERVER = None


def bind_server_runtime(runtime) -> None:
    global _SERVER
    _SERVER = runtime



def _sv():
    if _SERVER is None:
        raise RuntimeError("server runtime not bound")
    return _SERVER


def handle_get(handler, path: str, u) -> bool:
    sv = _sv()
    if path == "/api/settings/voice":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        sv._json_response(handler, 200, {"ok": True, **sv.MANAGER._voice_push.settings_snapshot()})
        return True
    if path == "/api/notifications/subscription":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        sv._json_response(
            handler,
            200,
            {"ok": True, **sv.MANAGER._voice_push.subscriptions_snapshot()},
        )
        return True
    if path == "/api/notifications/message":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        qs = urllib.parse.parse_qs(u.query)
        message_id = (qs.get("message_id") or [""])[0].strip()
        if not message_id:
            sv._json_response(handler, 400, {"error": "message_id required"})
            return True
        state = sv.MANAGER._voice_push.notification_state_for_message(message_id)
        if state is None:
            sv._json_response(handler, 404, {"error": "unknown message"})
            return True
        sv._json_response(handler, 200, {"ok": True, **state})
        return True
    if path == "/api/notifications/feed":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        qs = urllib.parse.parse_qs(u.query)
        since_raw = (qs.get("since") or ["0"])[0].strip()
        try:
            since_ts = float(since_raw or "0")
        except ValueError:
            sv._json_response(handler, 400, {"error": "invalid since"})
            return True
        items = sv.MANAGER._voice_push.notification_feed_since(since_ts)
        sv._json_response(handler, 200, {"ok": True, "items": items})
        return True
    if path == "/api/audio/live.m3u8":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        body = sv.MANAGER._voice_push.playlist_bytes()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/vnd.apple.mpegurl")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Pragma", "no-cache")
        handler.send_header("Expires", "0")
        handler.end_headers()
        handler.wfile.write(body)
        return True
    if path.startswith("/api/audio/segments/"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        segment_name = path.split("/api/audio/segments/", 1)[1]
        try:
            segment_path = sv.MANAGER._voice_push.segment_path(segment_name)
        except FileNotFoundError:
            handler.send_error(404)
            return True
        raw = segment_path.read_bytes()
        handler.send_response(200)
        handler.send_header("Content-Type", "video/mp2t")
        handler.send_header("Content-Length", str(len(raw)))
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Pragma", "no-cache")
        handler.send_header("Expires", "0")
        handler.end_headers()
        handler.wfile.write(raw)
        return True
    return False



def handle_post(handler, path: str, _u) -> bool:
    sv = _sv()
    if path == "/api/settings/voice":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        try:
            payload = sv.MANAGER._voice_push.set_settings(obj)
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        sv._json_response(handler, 200, {"ok": True, **payload})
        return True
    if path == "/api/notifications/subscription":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        try:
            payload = sv.MANAGER._voice_push.upsert_subscription(
                subscription=obj.get("subscription"),
                user_agent=str(obj.get("user_agent") or ""),
                device_label=str(obj.get("device_label") or ""),
                device_class=str(obj.get("device_class") or ""),
            )
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        sv._json_response(handler, 200, {"ok": True, **payload})
        return True
    if path == "/api/notifications/subscription/toggle":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        endpoint = obj.get("endpoint")
        enabled = obj.get("enabled")
        if not isinstance(endpoint, str) or not endpoint.strip():
            sv._json_response(handler, 400, {"error": "endpoint required"})
            return True
        if not isinstance(enabled, bool):
            sv._json_response(handler, 400, {"error": "enabled must be a boolean"})
            return True
        try:
            payload = sv.MANAGER._voice_push.toggle_subscription(
                endpoint=endpoint,
                enabled=enabled,
            )
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown subscription"})
            return True
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        sv._json_response(handler, 200, {"ok": True, **payload})
        return True
    if path == "/api/notifications/test_push":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        try:
            payload = sv.MANAGER._voice_push.send_test_push_notification(
                session_display_name="Codoxear test"
            )
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        sv._json_response(handler, 200, {"ok": True, **payload})
        return True
    if path == "/api/audio/listener":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        client_id = obj.get("client_id")
        enabled = obj.get("enabled")
        if not isinstance(client_id, str) or not client_id.strip():
            sv._json_response(handler, 400, {"error": "client_id required"})
            return True
        if not isinstance(enabled, bool):
            sv._json_response(handler, 400, {"error": "enabled must be a boolean"})
            return True
        payload = sv.MANAGER._voice_push.listener_heartbeat(
            client_id=client_id,
            enabled=enabled,
        )
        sv._json_response(handler, 200, {"ok": True, **payload})
        return True
    if path == "/api/audio/test_announcement":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        try:
            payload = sv.MANAGER._voice_push.enqueue_test_announcement(
                session_display_name="Codoxear test"
            )
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        sv._json_response(handler, 200, {"ok": True, **payload})
        return True
    if path == "/api/hooks/notify":
        sv._read_body(handler)
        sv._json_response(handler, 200, {"ignored": True})
        return True
    return False
