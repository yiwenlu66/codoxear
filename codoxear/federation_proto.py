"""Wire protocol primitives for codoxear federation (node <-> hub).

Frames are 4-byte big-endian length-prefixed UTF-8 JSON objects over TCP.
Authentication is a shared-secret token: ``sha256(secret + ":" + node_name)``
hex, presented by the node in its ``hello`` frame and by the hub on proxied
HTTP requests via the ``X-Federation-Token`` header. See docs/federation.md.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import struct
from typing import Any

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 16 * 1024 * 1024
FEDERATION_AUTH_HEADER = "X-Federation-Token"

# Message types.
HELLO = "hello"
WELCOME = "welcome"
REGISTER = "register"
UNREGISTER = "unregister"
SOCK_RPC = "sock_rpc"
SOCK_RESULT = "sock_result"
LAUNCH = "launch"
LAUNCH_RESULT = "launch_result"


class FederationProtocolError(RuntimeError):
    """Malformed or oversized frame on a federation connection."""


def federation_token(secret: str, node_name: str) -> str:
    """Return the hex auth token for *node_name* under *secret*."""
    return hashlib.sha256(f"{secret}:{node_name}".encode("utf-8")).hexdigest()


def token_matches(secret: str, node_name: str, token: Any) -> bool:
    """Constant-time check of *token* against the expected node token."""
    if not isinstance(token, str) or not token:
        return False
    return hmac.compare_digest(token, federation_token(secret, node_name))


def _recv_exact(conn: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("federation peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(conn: socket.socket, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    conn.sendall(struct.pack(">I", len(body)) + body)


def recv_frame(conn: socket.socket, *, max_bytes: int = MAX_FRAME_BYTES) -> dict[str, Any]:
    """Read one frame; raise FederationProtocolError on bad framing.

    Raises ConnectionError on clean or mid-frame EOF.
    """
    header = _recv_exact(conn, 4)
    (length,) = struct.unpack(">I", header)
    if length <= 0 or length > max_bytes:
        raise FederationProtocolError(f"invalid frame length {length} (max {max_bytes})")
    body = _recv_exact(conn, length)
    try:
        obj = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FederationProtocolError(f"invalid frame payload: {exc}") from exc
    if not isinstance(obj, dict):
        raise FederationProtocolError("frame payload must be a JSON object")
    return obj


def hello_message(*, node_name: str, secret: str, http_port: int) -> dict[str, Any]:
    return {
        "type": HELLO,
        "name": node_name,
        "token": federation_token(secret, node_name),
        "http_port": int(http_port),
        "version": PROTOCOL_VERSION,
    }


def sock_rpc_message(*, rid: int, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
    return {"type": SOCK_RPC, "rid": int(rid), "session_id": session_id, "request": request}


def sock_result_message(*, rid: int, ok: bool, response: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
    if ok:
        return {"type": SOCK_RESULT, "rid": int(rid), "ok": True, "response": dict(response or {})}
    return {"type": SOCK_RESULT, "rid": int(rid), "ok": False, "error": str(error or "unknown error")}
