#!/usr/bin/env python3
from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import http.server
import io
import json
import os
import signal
import socket
import socketserver
import subprocess
import struct
import sys
import threading
import time
import traceback
import urllib.parse
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _default_app_dir() -> Path:
    base = Path.home() / ".local" / "share"
    new = base / "codoxear"
    old = base / "codex-web"
    if old.exists() and not new.exists():
        return old
    return new


def _load_env_file(path: Path) -> dict[str, str]:
    try:
        data = path.read_text("utf-8")
    except Exception:
        return {}

    out: dict[str, str] = {}
    for raw in data.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1]
        if k:
            out[k] = v
    return out


APP_DIR = _default_app_dir()
STATIC_DIR = Path(__file__).resolve().parent / "static"
SOCK_DIR = APP_DIR / "socks"
STATE_PATH = APP_DIR / "state.json"
HMAC_SECRET_PATH = APP_DIR / "hmac_secret"
UPLOAD_DIR = APP_DIR / "uploads"

_DOTENV = (Path.cwd() / ".env").resolve()
if _DOTENV.exists():
    for _k, _v in _load_env_file(_DOTENV).items():
        os.environ.setdefault(_k, _v)

BOOTSTRAP_PROMPT = os.environ.get("CODEX_WEB_BOOTSTRAP_PROMPT", "Reply with the single word READY.")

CONTEXT_WINDOW_BASELINE_TOKENS = 12000

COOKIE_NAME = "codoxear_auth"
COOKIE_TTL_SECONDS = int(os.environ.get("CODEX_WEB_COOKIE_TTL_SECONDS", str(30 * 24 * 3600)))
COOKIE_SECURE = os.environ.get("CODEX_WEB_COOKIE_SECURE", "0") == "1"

CODEX_HOME = Path(os.environ.get("CODEX_HOME") or str(Path.home() / ".codex"))
CODEX_SESSIONS_DIR = CODEX_HOME / "sessions"

DEFAULT_HOST = os.environ.get("CODEX_WEB_HOST", "::")
DEFAULT_PORT = int(os.environ.get("CODEX_WEB_PORT", "8743"))


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # The PID exists but is owned by another user.
        return True


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except Exception:
        return


def _now() -> float:
    return time.time()


def _context_percent_remaining(*, tokens_in_context: int, context_window: int) -> int:
    if context_window <= CONTEXT_WINDOW_BASELINE_TOKENS:
        return 0
    effective = context_window - CONTEXT_WINDOW_BASELINE_TOKENS
    used = max(tokens_in_context - CONTEXT_WINDOW_BASELINE_TOKENS, 0)
    remaining = max(effective - used, 0)
    return int(round((remaining / effective) * 100.0))


def _extract_token_update(objs: list[dict[str, Any]]) -> dict[str, Any] | None:
    # Prefer the newest token_count in this batch.
    for obj in reversed(objs):
        if obj.get("type") != "event_msg":
            continue
        p = obj.get("payload") or {}
        if p.get("type") != "token_count":
            continue
        info = p.get("info")
        if not isinstance(info, dict) or not isinstance(info.get("total_token_usage"), dict):
            continue
        ctx = info.get("model_context_window")
        last = info.get("last_token_usage")
        if not isinstance(ctx, int) or not isinstance(last, dict):
            continue
        tt = last.get("total_tokens")
        if not isinstance(tt, int):
            continue
        return {
            "context_window": ctx,
            "tokens_in_context": tt,
            "tokens_remaining": max(ctx - tt, 0),
            "percent_remaining": _context_percent_remaining(tokens_in_context=tt, context_window=ctx),
            "baseline_tokens": CONTEXT_WINDOW_BASELINE_TOKENS,
            "as_of": obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None,
        }
    return None


def _find_latest_token_update(log_path: Path, max_scan_bytes: int = 32 * 1024 * 1024) -> dict[str, Any] | None:
    scan = 256 * 1024
    while scan <= max_scan_bytes:
        token = _extract_token_update(_read_jsonl_tail(log_path, scan))
        if token is not None:
            return token
        scan *= 2
    return None


def _sniff_image_ext(raw: bytes) -> str | None:
    if len(raw) >= 8 and raw[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if len(raw) >= 3 and raw[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return ".webp"
    return None


def _repair_png_crc(raw: bytes) -> bytes:
    if len(raw) < 8 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        return raw
    out = bytearray(raw)
    o = 8
    while o + 12 <= len(out):
        ln = struct.unpack(">I", out[o : o + 4])[0]
        typ = bytes(out[o + 4 : o + 8])
        data_start = o + 8
        data_end = data_start + ln
        crc_start = data_end
        crc_end = crc_start + 4
        if data_end > len(out) or crc_end > len(out):
            break
        data = bytes(out[data_start:data_end])
        calc = zlib.crc32(typ)
        calc = zlib.crc32(data, calc) & 0xFFFFFFFF
        cur = struct.unpack(">I", out[crc_start:crc_end])[0]
        if cur != calc:
            out[crc_start:crc_end] = struct.pack(">I", calc)
        o = crc_end
        if typ == b"IEND":
            break
    return bytes(out)


def _validate_image(raw: bytes) -> None:
    try:
        from PIL import Image
    except Exception:
        ext = _sniff_image_ext(raw)
        if ext is None:
            raise ValueError("unsupported image format")
        return
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            if not im.size or im.size[0] <= 0 or im.size[1] <= 0:
                raise ValueError("invalid image dimensions")
    except Exception as e:
        raise ValueError(str(e)) from e


def _json_response(handler: http.server.BaseHTTPRequestHandler, status: int, obj: Any) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: http.server.BaseHTTPRequestHandler, limit: int = 2 * 1024 * 1024) -> bytes:
    n = int(handler.headers.get("Content-Length", "0") or "0")
    if n < 0 or n > limit:
        raise ValueError(f"invalid content-length: {n}")
    return handler.rfile.read(n)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_or_create_hmac_secret() -> bytes:
    try:
        b = HMAC_SECRET_PATH.read_bytes()
        if len(b) >= 32:
            return b[:64]
    except FileNotFoundError:
        pass
    secret = os.urandom(64)
    HMAC_SECRET_PATH.write_bytes(secret)
    os.chmod(HMAC_SECRET_PATH, 0o600)
    return secret


HMAC_SECRET = _load_or_create_hmac_secret()


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _b64u_dec(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))


def _sign_cookie(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sig = hmac.new(HMAC_SECRET, raw, hashlib.sha256).digest()
    return f"{_b64u(raw)}.{_b64u(sig)}"


def _verify_cookie(value: str) -> dict[str, Any] | None:
    try:
        a, b = value.split(".", 1)
        raw = _b64u_dec(a)
        sig = _b64u_dec(b)
        want = hmac.new(HMAC_SECRET, raw, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, want):
            return None
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        exp = int(payload.get("exp", 0))
        if exp <= int(_now()):
            return None
        return payload
    except Exception:
        return None


def _parse_cookies(header: str | None) -> dict[str, str]:
    if not header:
        return {}
    out: dict[str, str] = {}
    parts = header.split(";")
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _require_auth(handler: http.server.BaseHTTPRequestHandler) -> bool:
    cookies = _parse_cookies(handler.headers.get("Cookie"))
    token = cookies.get(COOKIE_NAME)
    if not token:
        return False
    return _verify_cookie(token) is not None


def _set_auth_cookie(handler: http.server.BaseHTTPRequestHandler) -> None:
    exp = int(_now()) + COOKIE_TTL_SECONDS
    token = _sign_cookie({"exp": exp})
    attrs = [
        f"{COOKIE_NAME}={token}",
        "Path=/",
        "HttpOnly",
        "SameSite=Strict",
        f"Max-Age={COOKIE_TTL_SECONDS}",
    ]
    forwarded_proto = (handler.headers.get("X-Forwarded-Proto") or "").lower()
    if COOKIE_SECURE or forwarded_proto == "https":
        attrs.append("Secure")
    handler.send_header("Set-Cookie", "; ".join(attrs))

_PASSWORD_CACHE: str | None = None


def _require_password() -> str:
    global _PASSWORD_CACHE
    if _PASSWORD_CACHE is not None:
        return _PASSWORD_CACHE
    pw = (os.environ.get("CODEX_WEB_PASSWORD") or "").strip()
    if not pw:
        raise RuntimeError("CODEX_WEB_PASSWORD is required (set it in .env)")
    _PASSWORD_CACHE = pw
    return pw


def _password_hash() -> str:
    return _sha256_hex(_require_password().encode("utf-8"))


def _is_same_password(pw: str) -> bool:
    return hmac.compare_digest(_sha256_hex(pw.encode("utf-8")), _password_hash())


def _safe_read_text(path: Path, max_bytes: int = 512 * 1024) -> str:
    try:
        b = path.read_bytes()
        if len(b) > max_bytes:
            b = b[-max_bytes:]
        return b.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _safe_filename(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            out.append(ch)
    s = "".join(out).strip().replace(" ", "_")
    if not s:
        return "image"
    return s[:96]


def _iter_session_logs() -> list[Path]:
    if not CODEX_SESSIONS_DIR.exists():
        return []
    return sorted(CODEX_SESSIONS_DIR.rglob("rollout-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def _find_session_log_for_session_id(session_id: str) -> Path | None:
    for p in _iter_session_logs():
        if session_id in p.name:
            return p
    return None


def _find_new_session_log(
    *,
    after_ts: float,
    preexisting: set[Path],
    timeout_s: float = 15.0,
) -> tuple[str, Path] | None:
    deadline = _now() + timeout_s
    while _now() < deadline:
        for p in _iter_session_logs():
            if p in preexisting:
                continue
            try:
                if p.stat().st_mtime < after_ts - 2:
                    continue
            except FileNotFoundError:
                continue
            try:
                with p.open("r", encoding="utf-8") as f:
                    first = f.readline().strip()
                obj = json.loads(first)
                if obj.get("type") == "session_meta":
                    sid = obj.get("payload", {}).get("id")
                    if isinstance(sid, str) and sid:
                        return sid, p
            except Exception:
                continue
        time.sleep(0.2)
    return None


def _read_jsonl_from_offset(path: Path, offset: int, max_bytes: int = 2 * 1024 * 1024) -> tuple[list[dict[str, Any]], int]:
    try:
        with path.open("rb") as f:
            f.seek(offset)
            data = f.read(max_bytes)
            new_off = f.tell()
    except FileNotFoundError:
        return [], offset

    lines = data.splitlines()
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out, new_off


def _parse_iso8601_to_epoch(ts: str) -> float | None:
    try:
        t = ts.strip()
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        return datetime.datetime.fromisoformat(t).timestamp()
    except Exception:
        return None


def _discover_log_for_session_id(session_id: str) -> Path | None:
    return _find_session_log_for_session_id(session_id)


def _read_session_meta(log_path: Path) -> dict[str, Any]:
    try:
        with log_path.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
        obj = json.loads(first) if first else {}
        if obj.get("type") == "session_meta":
            return obj.get("payload") or {}
    except Exception:
        pass
    return {}


def _extract_chat_events(
    objs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, bool], dict[str, Any]]:
    """
    Produces a stream of chat-like events:
      - user: from event_msg.user_message
      - assistant: from response_item.message role=assistant output_text (preferred)
      - meta counts are returned separately (no meta events)
    """
    events: list[dict[str, Any]] = []
    total_thinking = 0
    total_tools = 0
    total_system = 0
    turn_start = False
    turn_end = False
    turn_aborted = False
    tool_names: set[str] = set()
    last_tool: str | None = None
    skip_next_assistant = 0

    def event_ts(o: dict[str, Any]) -> float | None:
        ts = o.get("ts")
        if isinstance(ts, (int, float)):
            return float(ts)
        ts2 = o.get("timestamp")
        if isinstance(ts2, (int, float)):
            return float(ts2)
        if isinstance(ts2, str):
            v = _parse_iso8601_to_epoch(ts2)
            if v is not None:
                return float(v)
        return None

    for obj in objs:
        typ = obj.get("type")
        if typ == "event_msg":
            p = obj.get("payload") or {}
            pt = p.get("type")
            if pt == "user_message":
                msg = p.get("message")
                if isinstance(msg, str):
                    if msg == BOOTSTRAP_PROMPT:
                        skip_next_assistant = 1
                        continue
                    turn_start = True
                    ets = event_ts(obj)
                    ev: dict[str, Any] = {"role": "user", "text": msg}
                    if ets is not None:
                        ev["ts"] = ets
                    events.append(ev)
                continue
            if pt == "agent_reasoning":
                # Count, do not show content.
                total_thinking += 1
                continue
            if pt == "turn_aborted":
                turn_aborted = True
                continue
            if pt == "token_count":
                turn_end = True
                continue

        if typ == "response_item":
            p = obj.get("payload") or {}
            pt = p.get("type")
            if pt == "message":
                role = p.get("role")
                if role in ("developer", "system"):
                    total_system += 1
                    continue
                if role == "assistant":
                    # Extract output_text parts.
                    content = p.get("content") or []
                    out_text_parts: list[str] = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                            out_text_parts.append(part["text"])
                    if out_text_parts:
                        text = "".join(out_text_parts)
                        if skip_next_assistant > 0:
                            skip_next_assistant -= 1
                            continue
                        ets = event_ts(obj)
                        ev2: dict[str, Any] = {"role": "assistant", "text": text}
                        if ets is not None:
                            ev2["ts"] = ets
                        events.append(ev2)
                    continue

            if pt == "reasoning":
                total_thinking += 1
                continue
            if pt == "function_call":
                nm = p.get("name")
                if isinstance(nm, str) and nm:
                    tool_names.add(nm)
                    last_tool = nm
                total_tools += 1
                continue
            if pt == "function_call_output":
                total_tools += 1
                continue

        # Other top-level entries: ignore.

    return (
        events,
        {"thinking": total_thinking, "tool": total_tools, "system": total_system},
        {"turn_start": turn_start, "turn_end": turn_end, "turn_aborted": turn_aborted},
        {"tool_names": sorted(tool_names), "last_tool": last_tool},
    )


def _read_jsonl_tail(path: Path, max_bytes: int) -> list[dict[str, Any]]:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - max_bytes)
            f.seek(start)
            data = f.read()
    except FileNotFoundError:
        return []
    except Exception:
        return []

    if not data:
        return []
    if start > 0:
        nl = data.find(b"\n")
        if nl >= 0:
            data = data[nl + 1 :]

    out: list[dict[str, Any]] = []
    for line in data.splitlines():
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def _read_chat_events_from_tail(
    log_path: Path,
    min_events: int = 120,
    max_scan_bytes: int = 128 * 1024 * 1024,
) -> list[dict[str, Any]]:
    scan = 256 * 1024
    best_events: list[dict[str, Any]] = []
    while scan <= max_scan_bytes:
        objs = _read_jsonl_tail(log_path, scan)
        events, _meta, _flags, _diag = _extract_chat_events(objs)
        best_events = events
        if len(events) >= min_events:
            break
        scan *= 2
    return best_events


def _compute_idle_from_log(path: Path, max_scan_bytes: int = 8 * 1024 * 1024) -> bool | None:
    """
    "Thread idle" (user spec; fail-busy):
    - idle only if:
      1) fresh session (no user/assistant/aborted events), or
      2) the last message is an assistant response, or
      3) a turn_aborted event occurred most recently (manual interrupt).
    - otherwise busy.
    """
    try:
        sz = int(path.stat().st_size)
    except Exception:
        sz = 0

    scan = 256 * 1024
    objs: list[dict[str, Any]] = []
    last_user_idx: int | None = None
    last_assistant_idx: int | None = None
    last_aborted_idx: int | None = None

    def has_assistant_text(obj: dict[str, Any]) -> bool:
        p = obj.get("payload") or {}
        if p.get("type") != "message" or p.get("role") != "assistant":
            return False
        content = p.get("content") or []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str) and part.get("text"):
                return True
        return False

    while scan <= max_scan_bytes:
        objs = _read_jsonl_tail(path, scan)
        last_user_idx = None
        last_assistant_idx = None
        last_aborted_idx = None
        for i, obj in enumerate(objs):
            typ = obj.get("type")
            if typ == "event_msg":
                p = obj.get("payload") or {}
                pt = p.get("type")
                if pt == "user_message":
                    msg = p.get("message")
                    if isinstance(msg, str) and msg != BOOTSTRAP_PROMPT:
                        last_user_idx = i
                        continue
                if pt == "agent_message":
                    msg = p.get("message")
                    if isinstance(msg, str) and msg.strip():
                        last_assistant_idx = i
                        continue
                if pt == "turn_aborted":
                    last_aborted_idx = i
                    continue
            if typ == "response_item" and has_assistant_text(obj):
                last_assistant_idx = i

        if (last_user_idx is not None) or (last_assistant_idx is not None) or (last_aborted_idx is not None) or scan >= max_scan_bytes:
            break
        scan *= 2

    if not objs:
        return None

    # Fresh session: very small log and no user/assistant/aborted events in the scanned window.
    if last_user_idx is None and last_assistant_idx is None and last_aborted_idx is None:
        return True if sz <= 128 * 1024 else False

    # Determine the last relevant event.
    end_idx: tuple[str, int] | None = None
    if last_user_idx is not None:
        end_idx = ("user", last_user_idx)
    if last_assistant_idx is not None:
        if end_idx is None or last_assistant_idx > end_idx[1]:
            end_idx = ("assistant", last_assistant_idx)
    if last_aborted_idx is not None:
        if end_idx is None or last_aborted_idx > end_idx[1]:
            end_idx = ("aborted", last_aborted_idx)
    if end_idx is None:
        return False
    return end_idx[0] in ("assistant", "aborted")


@dataclass
class Session:
    session_id: str
    thread_id: str
    broker_pid: int
    codex_pid: int
    owned: bool
    start_ts: float
    cwd: str
    log_path: Path
    sock_path: Path
    busy: bool = False
    queue_len: int = 0
    token: dict[str, Any] | None = None
    last_turn_id: str | None = None
    meta_thinking: int = 0
    meta_tools: int = 0
    meta_system: int = 0
    meta_log_off: int = 0


class SessionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._stop = threading.Event()
        self._discover_existing()

    def stop(self) -> None:
        self._stop.set()

    def _discover_existing(self) -> None:
        SOCK_DIR.mkdir(parents=True, exist_ok=True)
        for sock in sorted(SOCK_DIR.glob("*.sock")):
            session_id = sock.stem
            # Prefer metadata file written by sessiond.
            meta_path = sock.with_suffix(".json")
            meta: dict[str, Any] = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}

            thread_id = meta.get("session_id") if isinstance(meta.get("session_id"), str) and meta.get("session_id") else session_id
            codex_pid = int(meta.get("codex_pid") or 0)
            broker_pid = int(meta.get("broker_pid") or meta.get("sessiond_pid") or 0)
            owned = (meta.get("owner") == "web") if isinstance(meta.get("owner"), str) else False

            log_path = None
            if isinstance(meta.get("log_path"), str) and meta["log_path"]:
                log_path = Path(meta["log_path"])
            if not log_path or not log_path.exists():
                log_path = _discover_log_for_session_id(thread_id)
            if not log_path or not log_path.exists():
                if not _pid_alive(codex_pid) and not _pid_alive(broker_pid):
                    _unlink_quiet(sock)
                    _unlink_quiet(meta_path)
                continue

            cwd = meta.get("cwd") if isinstance(meta.get("cwd"), str) else ""
            if not cwd:
                sm = _read_session_meta(log_path)
                cwd = sm.get("cwd") if isinstance(sm.get("cwd"), str) else ""
            if not cwd:
                cwd = "?"

            start_ts = None
            if isinstance(meta.get("start_ts"), (int, float)):
                start_ts = float(meta["start_ts"])
            if start_ts is None:
                sm = _read_session_meta(log_path)
                if isinstance(sm.get("timestamp"), str):
                    start_ts = _parse_iso8601_to_epoch(sm["timestamp"])
            if start_ts is None:
                try:
                    start_ts = log_path.stat().st_mtime
                except Exception:
                    start_ts = _now()

            try:
                # Validate socket is responsive.
                resp = self._sock_call(sock, {"cmd": "state"}, timeout_s=0.5)
            except Exception:
                if not _pid_alive(codex_pid) and not _pid_alive(broker_pid):
                    _unlink_quiet(sock)
                    _unlink_quiet(meta_path)
                continue

            try:
                meta_log_off = int(log_path.stat().st_size)
            except Exception:
                meta_log_off = 0

            s = Session(
                session_id=session_id,
                thread_id=thread_id,
                broker_pid=broker_pid,
                codex_pid=codex_pid,
                owned=owned,
                start_ts=float(start_ts),
                cwd=str(cwd),
                log_path=log_path,
                sock_path=sock,
                busy=bool(resp.get("busy")) if isinstance(resp, dict) and "busy" in resp else False,
                queue_len=int(resp.get("queue_len")) if isinstance(resp, dict) and "queue_len" in resp else 0,
                token=resp.get("token") if isinstance(resp, dict) and ("token" in resp) and isinstance(resp.get("token"), (dict, type(None))) else None,
                meta_thinking=0,
                meta_tools=0,
                meta_system=0,
                meta_log_off=meta_log_off,
            )
            with self._lock:
                prev = self._sessions.get(session_id)
                if not prev:
                    self._sessions[session_id] = s
                else:
                    prev.sock_path = s.sock_path
                    prev.thread_id = s.thread_id
                    prev.broker_pid = s.broker_pid
                    prev.codex_pid = s.codex_pid
                    prev.owned = s.owned
                    prev.start_ts = s.start_ts
                    prev.cwd = s.cwd
                    prev.busy = s.busy
                    prev.queue_len = s.queue_len
                    prev.token = s.token
                    if prev.log_path != s.log_path:
                        prev.log_path = s.log_path
                        prev.meta_thinking = 0
                        prev.meta_tools = 0
                        prev.meta_system = 0
                        prev.meta_log_off = s.meta_log_off

    def _refresh_session_state(self, session_id: str, sock_path: Path, timeout_s: float = 0.4) -> bool:
        try:
            resp = self._sock_call(sock_path, {"cmd": "state"}, timeout_s=timeout_s)
        except Exception:
            return False
        with self._lock:
            s2 = self._sessions.get(session_id)
            if s2 and "busy" in resp:
                s2.busy = bool(resp.get("busy"))
                s2.queue_len = int(resp.get("queue_len", s2.queue_len))
                if "token" in resp:
                    tok = resp.get("token")
                    if isinstance(tok, dict) or tok is None:
                        s2.token = tok
        # Broker busy can be stale/incorrect if a log chunk contains both a prior assistant
        # response and a new user_message. When the rollout log was written very recently,
        # cross-check with a fail-busy log scan.
        if isinstance(resp, dict) and ("busy" in resp) and (not bool(resp.get("busy"))):
            with self._lock:
                s3 = self._sessions.get(session_id)
                lp = s3.log_path if s3 else None
            if lp and lp.exists():
                try:
                    mtime = float(lp.stat().st_mtime)
                except Exception:
                    mtime = 0.0
                if mtime and (_now() - mtime) < 120.0:
                    idle = _compute_idle_from_log(lp, max_scan_bytes=2 * 1024 * 1024)
                    if idle is False:
                        with self._lock:
                            s4 = self._sessions.get(session_id)
                            if s4:
                                s4.busy = True
        return True

    def _prune_dead_sessions(self) -> None:
        with self._lock:
            items = list(self._sessions.items())
        dead: list[tuple[str, Path]] = []
        for sid, s in items:
            if not s.sock_path.exists():
                dead.append((sid, s.sock_path))
                continue
            if self._refresh_session_state(sid, s.sock_path, timeout_s=0.4):
                continue
            if _pid_alive(s.broker_pid) or _pid_alive(s.codex_pid):
                continue
            dead.append((sid, s.sock_path))
        if not dead:
            return
        with self._lock:
            for sid, _sock in dead:
                self._sessions.pop(sid, None)
        for _sid, sock in dead:
            _unlink_quiet(sock)
            _unlink_quiet(sock.with_suffix(".json"))

    def _update_meta_counters(self) -> None:
        with self._lock:
            items = list(self._sessions.items())
        for sid, s in items:
            if not s.log_path.exists():
                continue

            if not s.busy:
                try:
                    off = int(s.log_path.stat().st_size)
                except Exception:
                    off = s.meta_log_off
                with self._lock:
                    s2 = self._sessions.get(sid)
                    if s2:
                        s2.meta_thinking = 0
                        s2.meta_tools = 0
                        s2.meta_system = 0
                        s2.meta_log_off = off
                continue

            objs, new_off = _read_jsonl_from_offset(s.log_path, s.meta_log_off, max_bytes=256 * 1024)
            if new_off == s.meta_log_off:
                continue

            d_th = 0
            d_tools = 0
            d_sys = 0
            for obj in objs:
                typ = obj.get("type")
                if typ == "event_msg":
                    p = obj.get("payload") or {}
                    pt = p.get("type")
                    if pt == "agent_reasoning":
                        d_th += 1
                    if pt == "user_message":
                        d_th = 0
                        d_tools = 0
                        d_sys = 0
                if typ == "response_item":
                    p = obj.get("payload") or {}
                    pt = p.get("type")
                    if pt == "reasoning":
                        d_th += 1
                    if pt in ("function_call", "function_call_output"):
                        d_tools += 1
                    if pt == "message" and p.get("role") in ("developer", "system"):
                        d_sys += 1

            with self._lock:
                s2 = self._sessions.get(sid)
                if s2:
                    s2.meta_thinking += d_th
                    s2.meta_tools += d_tools
                    s2.meta_system += d_sys
                    s2.meta_log_off = new_off

    def list_sessions(self) -> list[dict[str, Any]]:
        # Rescan sockets to pick up sessions created before the server started.
        self._discover_existing()
        self._prune_dead_sessions()
        with self._lock:
            items = list(self._sessions.items())
        for sid, s in items:
            with self._lock:
                s2 = self._sessions.get(sid)
            if s2 and s2.token is None:
                s2.token = _find_latest_token_update(s2.log_path, max_scan_bytes=8 * 1024 * 1024)
        self._update_meta_counters()
        with self._lock:
            out = []
            for s in self._sessions.values():
                out.append(
                    {
                        "session_id": s.session_id,
                        "thread_id": s.thread_id,
                        "pid": s.codex_pid,
                        "broker_pid": s.broker_pid,
                        "owned": s.owned,
                        "cwd": s.cwd,
                        "start_ts": s.start_ts,
                        "log_path": str(s.log_path),
                        "busy": s.busy,
                        "queue_len": s.queue_len,
                        "token": s.token,
                        "thinking": s.meta_thinking,
                        "tools": s.meta_tools,
                        "system": s.meta_system,
                    }
                )
            return out

    def get_session(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def refresh_session_meta(self, session_id: str) -> None:
        # The broker may rewrite the sock .json when Codex switches threads (/new, /resume).
        # Refresh the log path and thread id without requiring the UI to poll /api/sessions.
        self._discover_existing()
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return
            sock = s.sock_path
        meta_path = sock.with_suffix(".json")
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        thread_id = meta.get("session_id") if isinstance(meta.get("session_id"), str) and meta.get("session_id") else s.thread_id
        owned = (meta.get("owner") == "web") if isinstance(meta.get("owner"), str) else s.owned
        log_path = None
        if isinstance(meta.get("log_path"), str) and meta["log_path"]:
            log_path = Path(meta["log_path"])
        if not log_path or not log_path.exists():
            log_path = _discover_log_for_session_id(thread_id)
        if not log_path or not log_path.exists():
            return

        cwd = meta.get("cwd") if isinstance(meta.get("cwd"), str) else s.cwd
        if not cwd:
            sm = _read_session_meta(log_path)
            cwd = sm.get("cwd") if isinstance(sm.get("cwd"), str) else ""
        if not cwd:
            cwd = "?"

        with self._lock:
            s2 = self._sessions.get(session_id)
            if not s2:
                return
            s2.thread_id = thread_id
            s2.cwd = str(cwd)
            s2.owned = bool(owned)
            if s2.log_path != log_path:
                s2.log_path = log_path
                s2.meta_thinking = 0
                s2.meta_tools = 0
                s2.meta_system = 0
                try:
                    s2.meta_log_off = int(log_path.stat().st_size)
                except Exception:
                    s2.meta_log_off = 0

    def _sock_call(self, sock_path: Path, req: dict[str, Any], timeout_s: float = 2.0) -> dict[str, Any]:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout_s)
        try:
            s.connect(str(sock_path))
            s.sendall((json.dumps(req) + "\n").encode("utf-8"))
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
            line = buf.split(b"\n", 1)[0]
            if not line:
                return {"error": "empty response"}
            return json.loads(line.decode("utf-8"))
        finally:
            try:
                s.close()
            except Exception:
                pass

    def kill_session(self, session_id: str) -> bool:
        with self._lock:
            s = self._sessions.get(session_id)
        if not s:
            return False
        try:
            self._sock_call(s.sock_path, {"cmd": "shutdown"}, timeout_s=1.0)
        except Exception:
            try:
                if s.broker_pid:
                    os.kill(s.broker_pid, signal.SIGTERM)
            except Exception:
                pass
        return True

    def spawn_web_session(self, *, cwd: str, args: list[str] | None = None) -> dict[str, Any]:
        argv = [sys.executable, "-m", "codoxear.broker", "--cwd", cwd, "--"]
        if args:
            argv.extend(args)

        env = dict(os.environ)
        if _DOTENV.exists():
            for k, v in _load_env_file(_DOTENV).items():
                env.setdefault(k, v)
        env["CODEX_WEB_OWNER"] = "web"
        env.setdefault("CODEX_HOME", str(CODEX_HOME))

        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
        except Exception as e:
            raise RuntimeError(f"spawn failed: {e}") from e

        # Prevent zombies when the broker exits.
        threading.Thread(target=proc.wait, daemon=True).start()
        return {"broker_pid": int(proc.pid)}

    def delete_web_session(self, session_id: str) -> bool:
        with self._lock:
            s = self._sessions.get(session_id)
        if not s:
            return False
        if not s.owned:
            raise PermissionError("not owned by web")
        return self.kill_session(session_id)

    def send(self, session_id: str, text: str) -> dict[str, Any]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            sock = s.sock_path
        try:
            resp = self._sock_call(sock, {"cmd": "send", "text": text}, timeout_s=3.0)
        except Exception:
            if not _pid_alive(s.broker_pid) and not _pid_alive(s.codex_pid):
                with self._lock:
                    self._sessions.pop(session_id, None)
                _unlink_quiet(sock)
                _unlink_quiet(sock.with_suffix(".json"))
                raise KeyError("unknown session")
            raise
        with self._lock:
            s2 = self._sessions.get(session_id)
            if s2:
                if "busy" in resp:
                    s2.busy = bool(resp.get("busy"))
                if "queue_len" in resp:
                    s2.queue_len = int(resp.get("queue_len", s2.queue_len))
        return resp

    def get_state(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            sock = s.sock_path
        try:
            resp = self._sock_call(sock, {"cmd": "state"}, timeout_s=1.5)
        except Exception:
            if not _pid_alive(s.broker_pid) and not _pid_alive(s.codex_pid):
                with self._lock:
                    self._sessions.pop(session_id, None)
                _unlink_quiet(sock)
                _unlink_quiet(sock.with_suffix(".json"))
                raise KeyError("unknown session")
            raise
        with self._lock:
            s2 = self._sessions.get(session_id)
            if s2 and "busy" in resp:
                s2.busy = bool(resp.get("busy"))
                s2.queue_len = int(resp.get("queue_len", s2.queue_len))
                if "token" in resp:
                    tok = resp.get("token")
                    if isinstance(tok, dict) or tok is None:
                        s2.token = tok
        return resp

    def get_tail(self, session_id: str) -> str:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            sock = s.sock_path
        try:
            resp = self._sock_call(sock, {"cmd": "tail"}, timeout_s=1.5)
        except Exception:
            if not _pid_alive(s.broker_pid) and not _pid_alive(s.codex_pid):
                with self._lock:
                    self._sessions.pop(session_id, None)
                _unlink_quiet(sock)
                _unlink_quiet(sock.with_suffix(".json"))
                raise KeyError("unknown session")
            raise
        return str(resp.get("tail") or "")

    def inject_keys(self, session_id: str, seq: str) -> dict[str, Any]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            sock = s.sock_path
        try:
            resp = self._sock_call(sock, {"cmd": "keys", "seq": seq}, timeout_s=2.0)
        except Exception:
            if not _pid_alive(s.broker_pid) and not _pid_alive(s.codex_pid):
                with self._lock:
                    self._sessions.pop(session_id, None)
                _unlink_quiet(sock)
                _unlink_quiet(sock.with_suffix(".json"))
                raise KeyError("unknown session")
            raise
        return resp

    def mark_turn_complete(self, session_id: str, payload: dict[str, Any]) -> None:
        return


MANAGER = SessionManager()


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "codoxear/0.1"

    def _send_static(self, rel: str) -> None:
        path = (STATIC_DIR / rel.lstrip("/")).resolve()
        if not str(path).startswith(str(STATIC_DIR.resolve())):
            self.send_error(404)
            return
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        if path.suffix == ".html":
            ctype = "text/html; charset=utf-8"
        elif path.suffix == ".js":
            ctype = "text/javascript; charset=utf-8"
        elif path.suffix == ".css":
            ctype = "text/css; charset=utf-8"
        else:
            ctype = "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # UI is used for interactive debugging; serve assets without caching so changes
        # (including inline JS) show up immediately on refresh.
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(data)

    def _unauthorized(self) -> None:
        _json_response(self, 401, {"error": "unauthorized"})

    def do_GET(self) -> None:
        try:
            u = urllib.parse.urlparse(self.path)
            path = u.path
            if path == "/":
                self._send_static("index.html")
                return
            if path.startswith("/static/"):
                self._send_static(path[len("/static/") :])
                return

            if path == "/api/me":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                _json_response(self, 200, {"ok": True})
                return

            if path == "/api/sessions":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                _json_response(self, 200, {"sessions": MANAGER.list_sessions()})
                return

            if path.startswith("/api/sessions/") and path.endswith("/messages"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                if len(parts) < 4:
                    self.send_error(404)
                    return
                session_id = parts[3]
                MANAGER.refresh_session_meta(session_id)
                s = MANAGER.get_session(session_id)
                if not s or not s.log_path:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                qs = urllib.parse.parse_qs(u.query)
                offset = int((qs.get("offset") or ["0"])[0])
                init = (qs.get("init") or ["0"])[0] == "1"
                log_path_str = str(s.log_path)

                if init and offset == 0:
                    limit_raw = (qs.get("limit") or ["160"])[0]
                    try:
                        limit = int(limit_raw)
                    except Exception:
                        limit = 160
                    limit = max(20, min(400, limit))
                    events = _read_chat_events_from_tail(s.log_path, min_events=limit)
                    meta_delta = {"thinking": 0, "tool": 0, "system": 0}
                    flags = {"turn_start": False, "turn_end": False, "turn_aborted": False}
                    diag = {"tool_names": [], "last_tool": None}
                    try:
                        new_off = int(s.log_path.stat().st_size)
                    except Exception:
                        new_off = offset
                    token_update = _find_latest_token_update(s.log_path)
                else:
                    objs, new_off = _read_jsonl_from_offset(s.log_path, offset)
                    events, meta_delta, flags, diag = _extract_chat_events(objs)
                    token_update = _extract_token_update(objs)
                try:
                    state = MANAGER.get_state(session_id)
                except Exception:
                    state = None
                s2 = MANAGER.get_session(session_id)
                if token_update is not None and s2 is not None:
                    s2.token = token_update
                idle = _compute_idle_from_log(s.log_path)
                state_busy: bool | None = None
                state_queue: int | None = None
                token_sentinel = object()
                state_token: dict[str, Any] | None | object = token_sentinel
                if isinstance(state, dict) and ("busy" in state):
                    state_busy = bool(state.get("busy"))
                if isinstance(state, dict) and ("queue_len" in state):
                    try:
                        state_queue = int(state.get("queue_len"))
                    except Exception:
                        state_queue = None
                if isinstance(state, dict) and ("token" in state):
                    state_token = state.get("token")

                if state_busy is True:
                    busy_val = True
                else:
                    busy_val = ((not idle) if idle is not None else (s2.busy if s2 else True))

                queue_val = state_queue if state_queue is not None else (s2.queue_len if s2 else 0)

                if state_token is not token_sentinel and (isinstance(state_token, dict) or state_token is None):
                    token_val = state_token
                else:
                    token_val = s2.token if s2 else None
                _json_response(
                    self,
                    200,
                    {
                        "thread_id": s.thread_id,
                        "log_path": str(s.log_path),
                        "offset": new_off,
                        "events": events,
                        "meta_delta": meta_delta,
                        "turn_start": bool(flags.get("turn_start")),
                        "turn_end": bool(flags.get("turn_end")),
                        "turn_aborted": bool(flags.get("turn_aborted")),
                        "diag": diag,
                        "busy": bool(busy_val),
                        "queue_len": int(queue_val),
                        "token": token_val,
                    },
                )
                return

            if path.startswith("/api/sessions/") and path.endswith("/tail"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                try:
                    tail = MANAGER.get_tail(session_id)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, {"tail": tail})
                return

            self.send_error(404)
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})

    def do_POST(self) -> None:
        try:
            u = urllib.parse.urlparse(self.path)
            path = u.path

            if path == "/api/login":
                body = _read_body(self)
                obj = json.loads(body.decode("utf-8") or "{}")
                pw = obj.get("password")
                if not isinstance(pw, str) or not _is_same_password(pw):
                    _json_response(self, 403, {"error": "bad password"})
                    return
                self.send_response(200)
                _set_auth_cookie(self)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
                return

            if path == "/api/logout":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                self.send_response(200)
                self.send_header(
                    "Set-Cookie",
                    f"{COOKIE_NAME}=deleted; Path=/; Max-Age=0; HttpOnly; SameSite=Strict",
                )
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
                return

            if path == "/api/sessions":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                body = _read_body(self)
                obj = json.loads(body.decode("utf-8") or "{}")
                cwd = obj.get("cwd")
                if not isinstance(cwd, str) or not cwd.strip():
                    _json_response(self, 400, {"error": "cwd required"})
                    return
                args = obj.get("args")
                if args is None:
                    args_list = None
                elif isinstance(args, list) and all(isinstance(x, str) for x in args):
                    args_list = [x for x in args if x]
                else:
                    _json_response(self, 400, {"error": "args must be a list of strings"})
                    return
                res = MANAGER.spawn_web_session(cwd=cwd, args=args_list)
                _json_response(self, 200, {"ok": True, **res})
                return

            if path.startswith("/api/sessions/") and path.endswith("/delete"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                _read_body(self)
                try:
                    ok = MANAGER.delete_web_session(session_id)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                except PermissionError:
                    _json_response(self, 403, {"error": "session not owned by web"})
                    return
                if not ok:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, {"ok": True})
                return

            if path.startswith("/api/sessions/") and path.endswith("/send"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                body = _read_body(self)
                obj = json.loads(body.decode("utf-8") or "{}")
                text = obj.get("text")
                if not isinstance(text, str) or not text.strip():
                    _json_response(self, 400, {"error": "text required"})
                    return
                res = MANAGER.send(session_id, text)
                _json_response(self, 200, res)
                return

            if path.startswith("/api/sessions/") and path.endswith("/interrupt"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                _read_body(self)
                try:
                    # Send a literal ESC byte. Older brokers may not recognize "ESC" but will
                    # decode "\\x1b" via unicode_escape into a single 0x1b byte.
                    resp = MANAGER.inject_keys(session_id, "\\x1b")
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, {"ok": True, "broker": resp})
                return

            if path.startswith("/api/sessions/") and path.endswith("/inject_image"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                body = _read_body(self, limit=20 * 1024 * 1024)
                obj = json.loads(body.decode("utf-8") or "{}")
                data_b64 = obj.get("data_b64")
                filename = obj.get("filename") or "image"
                if not isinstance(data_b64, str) or not data_b64:
                    _json_response(self, 400, {"error": "data_b64 required"})
                    return
                if not isinstance(filename, str):
                    filename = "image"
                try:
                    raw = base64.b64decode(data_b64.encode("ascii"), validate=True)
                except Exception:
                    _json_response(self, 400, {"error": "invalid base64"})
                    return
                if len(raw) > 10 * 1024 * 1024:
                    _json_response(self, 413, {"error": "image too large"})
                    return

                sniffed = _sniff_image_ext(raw)
                if sniffed is None:
                    _json_response(self, 400, {"error": "unsupported image format"})
                    return
                if sniffed == ".png":
                    raw = _repair_png_crc(raw)
                try:
                    _validate_image(raw)
                except Exception as e:
                    _json_response(self, 400, {"error": f"invalid image: {e}"})
                    return

                stem = Path(_safe_filename(filename)).stem or "image"
                safe = stem + sniffed

                subdir = UPLOAD_DIR / session_id
                subdir.mkdir(parents=True, exist_ok=True)
                ts = int(_now() * 1000)
                out_path = (subdir / f"{ts}_{safe}").resolve()
                if not str(out_path).startswith(str(subdir.resolve())):
                    _json_response(self, 400, {"error": "bad path"})
                    return
                out_path.write_bytes(raw)
                try:
                    os.chmod(out_path, 0o600)
                except Exception:
                    pass

                # Bracketed paste: inject the image path; Codex TUI attaches if it exists and is an image.
                seq = f"\x1b[200~{str(out_path)}\x1b[201~"
                try:
                    resp = MANAGER.inject_keys(session_id, seq)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, {"ok": True, "path": str(out_path), "broker": resp})
                return

            if path == "/api/hooks/notify":
                # Optional integration point. Current design does not rely on this.
                _read_body(self)
                _json_response(self, 200, {"ignored": True})
                return

            self.send_error(404)
        except KeyError:
            _json_response(self, 404, {"error": "unknown session"})
        except Exception as e:
            _json_response(self, 500, {"error": str(e), "trace": traceback.format_exc()})

    def log_message(self, fmt: str, *args: Any) -> None:
        # Quiet default logging to keep terminal usable.
        return


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class ThreadingHTTPServerV6(ThreadingHTTPServer):
    address_family = socket.AF_INET6

    def server_bind(self) -> None:
        try:
            v6only = getattr(socket, "IPV6_V6ONLY", None)
            if v6only is not None:
                self.socket.setsockopt(socket.IPPROTO_IPV6, v6only, 0)
        except Exception:
            pass
        super().server_bind()


def main() -> None:
    os.makedirs(APP_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    try:
        _require_password()
    except Exception as e:
        sys.stderr.write(f"error: {e}\n")
        raise SystemExit(2)

    host = DEFAULT_HOST
    server: ThreadingHTTPServer
    try:
        if ":" in host:
            server = ThreadingHTTPServerV6((host, DEFAULT_PORT), Handler)
        else:
            server = ThreadingHTTPServer((host, DEFAULT_PORT), Handler)
    except Exception:
        # Fallback to dual-stack bind on all addresses when host was unset/default.
        if host == "::":
            try:
                server = ThreadingHTTPServerV6(("::", DEFAULT_PORT), Handler)
            except Exception:
                server = ThreadingHTTPServer(("0.0.0.0", DEFAULT_PORT), Handler)
        else:
            raise

    def _sigterm(_signo: int, _frame: Any) -> None:
        # BaseServer.shutdown() must not run in the serve_forever thread.
        MANAGER.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    server.serve_forever()


if __name__ == "__main__":
    main()
