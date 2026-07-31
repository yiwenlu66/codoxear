from __future__ import annotations

import json
import socket
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any


ControlAction = Callable[[], None]
ControlHandler = Callable[[dict[str, Any]], tuple[dict[str, Any], ControlAction | None]]
SendJsonLine = Callable[[socket.socket, dict[str, Any]], None]
PeerDisconnected = Callable[[BaseException], bool]


class ControlSocketCallError(RuntimeError):
    def __init__(self, message: str, *, request_sent: bool):
        super().__init__(message)
        self.request_sent = bool(request_sent)


def call_control_socket(
    sock_path: Path,
    req: dict[str, Any],
    *,
    timeout_s: float | None = 2.0,
    track_request_sent: bool = False,
) -> dict[str, Any]:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    request_sent = False
    try:
        sock.connect(str(sock_path))
        sock.sendall((json.dumps(req) + "\n").encode("utf-8"))
        request_sent = True
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
        line = buf.split(b"\n", 1)[0]
        if not line:
            return {"error": "empty response"}
        return json.loads(line.decode("utf-8"))
    except Exception as exc:
        if track_request_sent:
            raise ControlSocketCallError(str(exc), request_sent=request_sent) from exc
        raise
    finally:
        sock.close()


def handle_control_socket_connection(
    conn: socket.socket,
    *,
    handlers: dict[str, ControlHandler],
    send_json_line: SendJsonLine,
    socket_peer_disconnected: PeerDisconnected,
) -> None:
    f = None
    try:
        f = conn.makefile("rb")
        line = f.readline()
        if not line:
            return
        req = json.loads(line.decode("utf-8"))
        if not isinstance(req, dict):
            send_json_line(conn, {"error": "invalid request"})
            return
        cmd = req.get("cmd")
        handler = handlers.get(cmd) if isinstance(cmd, str) else None
        if handler is None:
            send_json_line(conn, {"error": "unknown cmd"})
            return
        resp, after_reply = handler(req)
        send_json_line(conn, resp)
        if after_reply is not None:
            after_reply()
    except Exception as exc:
        if socket_peer_disconnected(exc):
            return
        try:
            send_json_line(conn, {"error": "exception", "trace": traceback.format_exc()})
        except Exception as send_exc:
            if not socket_peer_disconnected(send_exc):
                traceback.print_exc()
    finally:
        if f is not None:
            try:
                f.close()
            except Exception as close_exc:
                if not socket_peer_disconnected(close_exc):
                    traceback.print_exc()
        try:
            conn.close()
        except Exception as close_exc:
            if not socket_peer_disconnected(close_exc):
                traceback.print_exc()
