from __future__ import annotations

"""Control channel into the running Pi bridge extension.

The bridge (``codoxear/pi_active_session_bridge.ts``) owns the only in-process
handle on Pi's session state. It opens a Unix socket next to its marker file;
the broker speaks one request per connection over it. Everything is fail-closed:
a missing socket, a stale peer, or an unexpected reply is a capability failure,
never a session failure.
"""

import json
import socket
from pathlib import Path
from typing import Any


_MAX_RESPONSE_BYTES = 64 * 1024


def bridge_socket_path(marker_path: Path) -> Path:
    return Path(f"{marker_path}.bridge.sock")


def call_bridge(
    marker_path: Path,
    request: dict[str, Any],
    *,
    timeout_s: float = 1.5,
) -> dict[str, Any]:
    """Send one request to the bridge and return its reply object."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(max(timeout_s, 0.1))
    try:
        sock.connect(str(bridge_socket_path(marker_path)))
        sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
            if len(buf) > _MAX_RESPONSE_BYTES:
                raise ValueError("bridge response too large")
    finally:
        sock.close()
    line = buf.split(b"\n", 1)[0]
    if not line:
        raise ValueError("empty bridge response")
    reply = json.loads(line.decode("utf-8"))
    if not isinstance(reply, dict):
        raise ValueError("invalid bridge response")
    return reply


def bridge_session_name(marker_path: Path, *, expected_pid: int | None = None, timeout_s: float = 1.5) -> str | None:
    """The bridge's current session name, or ``None`` when it cannot be read."""
    try:
        reply = call_bridge(marker_path, {"op": "get_session_name"}, timeout_s=timeout_s)
    except Exception:
        return None
    if expected_pid is not None and reply.get("pid") != expected_pid:
        return None
    if reply.get("ok") is not True:
        return None
    name = reply.get("name")
    return name if isinstance(name, str) else ""


def set_bridge_session_name(
    marker_path: Path,
    name: str,
    *,
    expected_pid: int | None = None,
    timeout_s: float = 1.5,
) -> dict[str, Any]:
    """Write the Pi session display name through the bridge.

    Returns ``{"ok": True, "name": <effective>}`` on success. A pid mismatch
    means the socket belongs to a different Pi process than the one this broker
    bound, which must never be renamed.
    """
    reply = call_bridge(marker_path, {"op": "set_session_name", "name": name}, timeout_s=timeout_s)
    if expected_pid is not None and reply.get("pid") != expected_pid:
        return {"ok": False, "error": "bridge pid mismatch"}
    if reply.get("ok") is not True:
        return {"ok": False, "error": str(reply.get("error") or "bridge rejected the request")}
    effective = reply.get("name")
    return {"ok": True, "name": effective if isinstance(effective, str) else ""}
