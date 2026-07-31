from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Callable, Protocol


class MessageCursorError(ValueError):
    pass


class MessageCursorSession(Protocol):
    thread_id: str
    log_path: Path | None


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64u_dec(s: str) -> bytes:
    pad = "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))


def sign_message_cursor(payload: dict[str, Any], *, secret: bytes) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sig = hmac.new(secret, raw, hashlib.sha256).digest()
    return f"{_b64u(raw)}.{_b64u(sig)}"


def verify_message_cursor(token: str, *, secret: bytes) -> dict[str, Any]:
    try:
        a, b = token.split(".", 1)
        raw = _b64u_dec(a)
        sig = _b64u_dec(b)
        want = hmac.new(secret, raw, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, want):
            raise MessageCursorError("cursor_invalid")
        payload = json.loads(raw.decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError):
        raise MessageCursorError("cursor_invalid")
    if not isinstance(payload, dict):
        raise MessageCursorError("cursor_invalid")
    return payload


def encode_message_cursor(*, kind: str, session: MessageCursorSession, pos: int, secret: bytes) -> str:
    return sign_message_cursor(
        {
            "v": 1,
            "kind": kind,
            "thread_id": session.thread_id,
            "log_path": str(session.log_path) if session.log_path is not None else None,
            "pos": int(pos),
        },
        secret=secret,
    )


def decode_message_cursor(token: str, *, kind: str, session: MessageCursorSession, secret: bytes) -> int:
    payload = verify_message_cursor(token, secret=secret)
    if payload.get("v") != 1 or payload.get("kind") != kind:
        raise MessageCursorError("cursor_invalid")
    if payload.get("thread_id") != session.thread_id:
        raise MessageCursorError("cursor_invalid")
    expected_log_path = str(session.log_path) if session.log_path is not None else None
    if payload.get("log_path") != expected_log_path:
        raise MessageCursorError("cursor_invalid")
    pos = payload.get("pos")
    if not isinstance(pos, int) or pos < 0:
        raise MessageCursorError("cursor_invalid")
    if session.log_path is not None and session.log_path.exists():
        size = int(session.log_path.stat().st_size)
        if pos > size:
            raise MessageCursorError("cursor_invalid")
    return int(pos)


def attach_history_cursors(
    events: list[dict[str, Any]],
    *,
    session: MessageCursorSession,
    encode_cursor: Callable[..., str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            out.append(ev)
            continue
        pos = ev.get("_before_byte")
        ev2 = {k: v for k, v in ev.items() if k != "_before_byte"}
        if isinstance(pos, int) and pos >= 0:
            ev2["history_cursor"] = encode_cursor(kind="history", session=session, pos=pos)
        out.append(ev2)
    return out
