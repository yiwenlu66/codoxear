from __future__ import annotations

import json
import socket
import time
import urllib.parse
from typing import Any

_SERVER = None
SSE_HEARTBEAT_SECONDS = 15.0
SSE_RETRY_MS = 3000


def bind_server_runtime(runtime: Any) -> None:
    global _SERVER
    _SERVER = runtime



def _sv() -> Any:
    if _SERVER is None:
        raise RuntimeError("server runtime not bound")
    return _SERVER



def _cursor_from_request(handler: Any, query: dict[str, list[str]]) -> int:
    header_value = ""
    headers = getattr(handler, "headers", None)
    if headers is not None:
        header_value = str(headers.get("Last-Event-ID") or "").strip()
    cursor_value = str((query.get("cursor") or [header_value or "0"])[0] or header_value or "0").strip()
    try:
        return max(0, int(cursor_value or "0"))
    except ValueError:
        return 0



def _write_sse_event(handler: Any, event: dict[str, Any]) -> None:
    payload = json.dumps(event, ensure_ascii=True, separators=(",", ":"))
    data = (
        f"id: {int(event.get('seq') or 0)}\n"
        f"event: {str(event.get('type') or 'message')}\n"
        f"data: {payload}\n\n"
    ).encode("utf-8")
    handler.wfile.write(data)
    handler.wfile.flush()



def _write_sse_comment(handler: Any, text: str) -> None:
    handler.wfile.write(f": {text}\n\n".encode("utf-8"))
    handler.wfile.flush()



def handle_get(handler: Any, path: str, u: Any) -> bool:
    sv = _sv()
    if path != "/api/events":
        return False
    if not sv._require_auth(handler):
        handler._unauthorized()
        return True

    query = urllib.parse.parse_qs(u.query)
    after_seq = _cursor_from_request(handler, query)
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()

    try:
        handler.wfile.write(f"retry: {SSE_RETRY_MS}\n\n".encode("utf-8"))
        handler.wfile.flush()
        while True:
            result = sv.EVENT_HUB.poll(after_seq, timeout_s=SSE_HEARTBEAT_SECONDS)
            if result.closed:
                return True
            if result.cursor_expired:
                resync_event = {
                    "seq": result.latest_seq,
                    "type": "stream.resync",
                    "ts": time.time(),
                }
                _write_sse_event(handler, resync_event)
                after_seq = int(result.latest_seq)
                continue
            if result.events:
                for event in result.events:
                    _write_sse_event(handler, event)
                    after_seq = max(after_seq, int(event.get("seq") or 0))
                continue
            _write_sse_comment(handler, "heartbeat")
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout, ValueError, OSError):
        return True
