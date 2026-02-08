#!/usr/bin/env python3
from __future__ import annotations

import base64
import datetime
import errno
import hashlib
import hmac
import http.server
import io
import json
import os
import re
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
from shutil import which
from typing import Any

from .util import default_app_dir as _default_app_dir
from .util import classify_session_log as _classify_session_log
from .util import find_new_session_log as _find_new_session_log_impl
from .util import find_session_log_for_session_id as _find_session_log_for_session_id_impl
from .util import is_subagent_session_meta as _is_subagent_session_meta
from .util import iter_session_logs as _iter_session_logs_impl
from .util import now as _now
from .util import read_jsonl_from_offset as _read_jsonl_from_offset_impl
from .util import subagent_parent_thread_id as _subagent_parent_thread_id


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
HARNESS_PATH = APP_DIR / "harness.json"

_DOTENV = (Path.cwd() / ".env").resolve()
if _DOTENV.exists():
    for _k, _v in _load_env_file(_DOTENV).items():
        os.environ.setdefault(_k, _v)

CONTEXT_WINDOW_BASELINE_TOKENS = 12000

COOKIE_NAME = "codoxear_auth"
COOKIE_TTL_SECONDS = int(os.environ.get("CODEX_WEB_COOKIE_TTL_SECONDS", str(30 * 24 * 3600)))
COOKIE_SECURE = os.environ.get("CODEX_WEB_COOKIE_SECURE", "0") == "1"

CODEX_HOME = Path(os.environ.get("CODEX_HOME") or str(Path.home() / ".codex"))
CODEX_SESSIONS_DIR = CODEX_HOME / "sessions"

DEFAULT_HOST = os.environ.get("CODEX_WEB_HOST", "::")
DEFAULT_PORT = int(os.environ.get("CODEX_WEB_PORT", "8743"))
HARNESS_IDLE_SECONDS = int(os.environ.get("CODEX_WEB_HARNESS_IDLE_SECONDS", "300"))
HARNESS_SWEEP_SECONDS = float(os.environ.get("CODEX_WEB_HARNESS_SWEEP_SECONDS", "2.5"))
HARNESS_MAX_SCAN_BYTES = int(os.environ.get("CODEX_WEB_HARNESS_MAX_SCAN_BYTES", str(8 * 1024 * 1024)))
DISCOVER_MIN_INTERVAL_SECONDS = float(os.environ.get("CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS", "1.0"))
CHAT_INIT_SEED_SCAN_BYTES = int(os.environ.get("CODEX_WEB_CHAT_INIT_SEED_SCAN_BYTES", str(512 * 1024)))
CHAT_INIT_MAX_SCAN_BYTES = int(os.environ.get("CODEX_WEB_CHAT_INIT_MAX_SCAN_BYTES", str(128 * 1024 * 1024)))
CHAT_INDEX_INCREMENT_BYTES = int(os.environ.get("CODEX_WEB_CHAT_INDEX_INCREMENT_BYTES", str(2 * 1024 * 1024)))
CHAT_INDEX_RESEED_THRESHOLD_BYTES = int(os.environ.get("CODEX_WEB_CHAT_INDEX_RESEED_THRESHOLD_BYTES", str(16 * 1024 * 1024)))
CHAT_INDEX_MAX_EVENTS = int(os.environ.get("CODEX_WEB_CHAT_INDEX_MAX_EVENTS", "12000"))
METRICS_WINDOW = int(os.environ.get("CODEX_WEB_METRICS_WINDOW", "256"))
HARNESS_DEFAULT_TEXT = (
    "[Automated message from Agent Harness] the user is currently away, and you should continue with your previous task. "
    "Review whether what you have done fully complies with user intention. Self-reflect and make improvements when possible. "
    "If you believe you are done with all of the tasks, spawn a critic sub-agent to review your work and examine whether if fully addresses the user request. "
    "Continue monitoring experiments running in the background to locate any issues.\n\n---\n"
)

_ROLLOUT_PATH_RE = re.compile(r'"([^"]*rollout-[^"]*\.jsonl)"')
_SESSION_ID_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)

_STRACE_USABLE: bool | None = None
_STRACE_ERR: str | None = None
_METRICS_LOCK = threading.Lock()
_METRICS: dict[str, list[float]] = {}


def _record_metric(name: str, value_ms: float) -> None:
    if not isinstance(name, str) or not name:
        return
    try:
        v = float(value_ms)
    except Exception:
        return
    if not (v >= 0):
        return
    with _METRICS_LOCK:
        arr = _METRICS.get(name)
        if arr is None:
            arr = []
            _METRICS[name] = arr
        arr.append(v)
        if len(arr) > METRICS_WINDOW:
            del arr[: len(arr) - METRICS_WINDOW]


def _metric_percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = max(0.0, min(1.0, float(p))) * float(len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - float(lo)
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _metrics_snapshot() -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    with _METRICS_LOCK:
        items = list(_METRICS.items())
    for name, samples in items:
        if not samples:
            continue
        srt = sorted(float(x) for x in samples)
        out[name] = {
            "count": len(srt),
            "last_ms": float(samples[-1]),
            "p50_ms": _metric_percentile(srt, 0.50),
            "p95_ms": _metric_percentile(srt, 0.95),
            "max_ms": float(srt[-1]),
        }
    return out


def _strace_usable() -> bool:
    global _STRACE_USABLE, _STRACE_ERR
    if _STRACE_USABLE is not None:
        return bool(_STRACE_USABLE)
    binp = os.environ.get("CODEX_WEB_STRACE_BIN", "strace")
    if which(binp) is None:
        _STRACE_USABLE = False
        _STRACE_ERR = f"{binp} not found in PATH"
        return False
    try:
        proc = subprocess.run(
            [binp, "-qq", "-o", "/dev/null", "--", "/bin/true"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=1.5,
            check=False,
        )
        _STRACE_USABLE = proc.returncode == 0
        if not _STRACE_USABLE:
            err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
            _STRACE_ERR = err[-400:] if err else f"rc={proc.returncode}"
    except Exception as e:
        _STRACE_USABLE = False
        _STRACE_ERR = str(e)
    return bool(_STRACE_USABLE)


def _wait_or_raise(proc: subprocess.Popen[bytes], *, label: str, timeout_s: float = 1.5) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        rc = proc.poll()
        if rc is None:
            time.sleep(0.05)
            continue
        try:
            _out, err = proc.communicate(timeout=0.5)
        except Exception:
            err = b""
        msg = (err or b"").decode("utf-8", errors="replace").strip()
        msg = msg[-4000:] if msg else ""
        raise RuntimeError(f"{label} exited early (rc={rc}): {msg}")


def _drain_stream(f: Any) -> None:
    try:
        while True:
            b = f.read(65536)
            if not b:
                break
    except Exception:
        pass
    try:
        f.close()
    except Exception:
        pass


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


def _sock_error_definitely_stale(exc: BaseException) -> bool:
    if isinstance(exc, (FileNotFoundError, ConnectionRefusedError)):
        return True
    if isinstance(exc, OSError):
        return exc.errno in (errno.ENOENT, errno.ECONNREFUSED, errno.ENOTSOCK)
    return False


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
    scan = min(256 * 1024, max_scan_bytes)
    if scan <= 0:
        return None
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
    return _iter_session_logs_impl(CODEX_SESSIONS_DIR)


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
    return _find_new_session_log_impl(
        sessions_dir=CODEX_SESSIONS_DIR,
        after_ts=after_ts,
        preexisting=preexisting,
        timeout_s=timeout_s,
    )


def _read_jsonl_from_offset(path: Path, offset: int, max_bytes: int = 2 * 1024 * 1024) -> tuple[list[dict[str, Any]], int]:
    return _read_jsonl_from_offset_impl(path, offset, max_bytes=max_bytes)


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

def _session_id_from_rollout_path(log_path: Path) -> str | None:
    try:
        name = log_path.name
    except Exception:
        name = str(log_path)
    m = _SESSION_ID_RE.findall(name)
    return m[-1] if m else None


def _is_write_open_line(line: str) -> bool:
    if ("O_WRONLY" in line) or ("O_RDWR" in line) or ("O_APPEND" in line):
        return True
    if "O_RDONLY" in line:
        return False
    if "O_CREAT" in line:
        return True

    m = re.search(r",\s*(0x[0-9a-fA-F]+|\d+)(?:\s*,|\s*\))", line)
    if not m:
        return False
    try:
        v = int(m.group(1), 16 if m.group(1).lower().startswith("0x") else 10)
    except Exception:
        return False
    acc = v & 3
    return acc in (1, 2)


def _read_text_tail(path: Path, *, max_bytes: int) -> str:
    try:
        with path.open("rb") as f:
            try:
                sz = int(f.seek(0, os.SEEK_END))
            except Exception:
                sz = 0
            start = max(0, sz - int(max_bytes))
            drop_partial = False
            if start > 0:
                try:
                    f.seek(start - 1)
                    prev = f.read(1)
                    drop_partial = (prev != b"\n")
                except Exception:
                    drop_partial = True
            try:
                f.seek(start)
            except Exception:
                f.seek(0)
                start = 0
                drop_partial = False
            b = f.read(int(max_bytes))
    except FileNotFoundError:
        return ""
    except Exception:
        return ""

    if start > 0 and drop_partial:
        nl = b.find(b"\n")
        if nl >= 0:
            b = b[nl + 1 :]
    try:
        return b.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _discover_log_from_trace(trace_path: Path) -> Path | None:
    data = _read_text_tail(trace_path, max_bytes=8 * 1024 * 1024)
    if not data:
        return None

    sessions_dir = CODEX_SESSIONS_DIR.resolve()
    best: Path | None = None
    for line in data.splitlines():
        if "rollout-" not in line or ".jsonl" not in line:
            continue
        m = _ROLLOUT_PATH_RE.findall(line)
        if not m:
            continue
        raw = line.strip()
        is_open = (
            (" open(" in raw)
            or (" openat(" in raw)
            or (" openat2(" in raw)
            or (" creat(" in raw)
            or raw.startswith("open(")
            or raw.startswith("openat")
            or raw.startswith("openat2")
            or raw.startswith("creat")
        )
        is_rename = (
            (" rename(" in raw)
            or (" renameat(" in raw)
            or (" renameat2(" in raw)
            or raw.startswith("rename")
            or raw.startswith("renameat")
            or raw.startswith("renameat2")
        )
        if is_open and (not _is_write_open_line(raw)):
            continue

        p = Path(m[-1])
        try:
            p.resolve().relative_to(sessions_dir)
        except Exception:
            continue
        if not (p.name.startswith("rollout-") and p.name.endswith(".jsonl")):
            continue
        if _classify_session_log(p, timeout_s=0.0) != "main":
            continue

        if is_open or is_rename:
            best = p

    return best


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


def _coerce_main_thread_log(*, thread_id: str, log_path: Path) -> tuple[str, Path]:
    sm = _read_session_meta(log_path)
    if not sm:
        return thread_id, log_path
    if not _is_subagent_session_meta(sm):
        return thread_id, log_path
    parent = _subagent_parent_thread_id(sm)
    if not parent:
        return thread_id, log_path
    parent_log = _find_session_log_for_session_id_impl(CODEX_SESSIONS_DIR, parent)
    if parent_log is None or not parent_log.exists():
        return thread_id, log_path
    return parent, parent_log


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
    events, _token, _scan_bytes, _scan_complete, _size = _read_chat_tail_snapshot(
        log_path,
        min_events=min_events,
        initial_scan_bytes=min(256 * 1024, max_scan_bytes),
        max_scan_bytes=max_scan_bytes,
    )
    return events


def _read_chat_tail_snapshot(
    log_path: Path,
    *,
    min_events: int,
    initial_scan_bytes: int,
    max_scan_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int, bool, int]:
    try:
        size = int(log_path.stat().st_size)
    except Exception:
        size = 0
    scan = min(max(256 * 1024, int(initial_scan_bytes)), int(max_scan_bytes))
    if scan <= 0:
        return [], None, 0, True, size

    best_events: list[dict[str, Any]] = []
    best_token: dict[str, Any] | None = None
    while True:
        objs = _read_jsonl_tail(log_path, scan)
        events, _meta, _flags, _diag = _extract_chat_events(objs)
        best_events = events
        tok = _extract_token_update(objs)
        if tok is not None:
            best_token = tok
        if len(events) >= min_events or scan >= max_scan_bytes:
            break
        next_scan = min(scan * 2, max_scan_bytes)
        if next_scan <= scan:
            break
        scan = next_scan

    scan_complete = (size <= scan)
    return best_events, best_token, scan, scan_complete, size


def _event_ts(obj: dict[str, Any]) -> float | None:
    ts = obj.get("ts")
    if isinstance(ts, (int, float)):
        return float(ts)
    ts2 = obj.get("timestamp")
    if isinstance(ts2, (int, float)):
        return float(ts2)
    if isinstance(ts2, str):
        v = _parse_iso8601_to_epoch(ts2)
        if v is not None:
            return float(v)
    return None


def _has_assistant_output_text(obj: dict[str, Any]) -> bool:
    p = obj.get("payload") or {}
    if p.get("type") != "message" or p.get("role") != "assistant":
        return False
    content = p.get("content") or []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str) and part.get("text"):
            return True
    return False


def _analyze_log_chunk(
    objs: list[dict[str, Any]],
) -> tuple[int, int, int, float | None, dict[str, Any] | None, list[dict[str, Any]]]:
    d_th = 0
    d_tools = 0
    d_sys = 0
    last_chat_ts: float | None = None
    token_update = _extract_token_update(objs)
    chat_events, _meta, _flags, _diag = _extract_chat_events(objs)

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
                if isinstance(p.get("message"), str):
                    last_chat_ts = _event_ts(obj)
            if pt == "agent_message":
                msg = p.get("message")
                if isinstance(msg, str) and msg.strip():
                    last_chat_ts = _event_ts(obj)
        if typ == "response_item":
            p = obj.get("payload") or {}
            pt = p.get("type")
            if pt == "reasoning":
                d_th += 1
            if pt in ("function_call", "function_call_output"):
                d_tools += 1
            if pt == "message" and p.get("role") in ("developer", "system"):
                d_sys += 1
            if _has_assistant_output_text(obj):
                last_chat_ts = _event_ts(obj)

    return d_th, d_tools, d_sys, last_chat_ts, token_update, chat_events


def _last_conversation_ts_from_tail(
    log_path: Path,
    *,
    max_scan_bytes: int,
) -> float | None:
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

    def has_assistant_text(obj: dict[str, Any]) -> bool:
        p = obj.get("payload") or {}
        if p.get("type") != "message" or p.get("role") != "assistant":
            return False
        content = p.get("content") or []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str) and part.get("text"):
                return True
        return False

    scan = 256 * 1024
    while True:
        objs = _read_jsonl_tail(log_path, scan)
        last_idx: int | None = None
        last_ts: float | None = None
        for i, obj in enumerate(objs):
            typ = obj.get("type")
            if typ == "event_msg":
                p = obj.get("payload") or {}
                pt = p.get("type")
                if pt == "user_message" and isinstance(p.get("message"), str):
                    last_idx = i
                    last_ts = event_ts(obj)
                    continue
                if pt == "agent_message":
                    msg = p.get("message")
                    if isinstance(msg, str) and msg.strip():
                        last_idx = i
                        last_ts = event_ts(obj)
                        continue
            if typ == "response_item" and has_assistant_text(obj):
                last_idx = i
                last_ts = event_ts(obj)
                continue

        if last_idx is not None:
            return last_ts
        if scan >= max_scan_bytes:
            return None
        scan *= 2


def _compute_idle_from_log(path: Path, max_scan_bytes: int = 8 * 1024 * 1024) -> bool | None:
    """
    "Thread idle" (user spec; fail-busy):
    - idle only if:
      1) fresh session (no turn activity in scanned window), or
      2) current turn has an assistant completion candidate and no later work,
      3) latest terminal event is turn_aborted/thread_rolled_back.
    - otherwise busy.
    """
    try:
        sz = int(path.stat().st_size)
    except Exception:
        sz = 0

    scan = min(256 * 1024, max_scan_bytes)
    if scan <= 0:
        return None
    objs: list[dict[str, Any]] = []
    saw_user = False
    turn_open = False
    turn_has_completion_candidate = False
    last_terminal_event: str | None = None

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
        saw_user = False
        turn_open = False
        turn_has_completion_candidate = False
        last_terminal_event = None
        for obj in objs:
            typ = obj.get("type")
            if typ == "event_msg":
                p = obj.get("payload") or {}
                pt = p.get("type")
                if pt == "user_message":
                    msg = p.get("message")
                    if isinstance(msg, str):
                        saw_user = True
                        turn_open = True
                        turn_has_completion_candidate = False
                        last_terminal_event = "user"
                        continue
                if pt == "agent_message":
                    msg = p.get("message")
                    if isinstance(msg, str) and msg.strip():
                        if turn_open:
                            turn_has_completion_candidate = True
                        last_terminal_event = "assistant"
                        continue
                if pt in ("turn_aborted", "thread_rolled_back"):
                    turn_open = False
                    turn_has_completion_candidate = False
                    last_terminal_event = "aborted"
                    continue
                if pt == "agent_reasoning" and turn_open:
                    turn_has_completion_candidate = False
                    continue
            if typ == "response_item":
                p = obj.get("payload") or {}
                pt = p.get("type")
                if has_assistant_text(obj):
                    if turn_open:
                        turn_has_completion_candidate = True
                    last_terminal_event = "assistant"
                    continue
                if pt in (
                    "reasoning",
                    "function_call",
                    "function_call_output",
                    "custom_tool_call",
                    "custom_tool_call_output",
                    "web_search_call",
                ) and turn_open:
                    turn_has_completion_candidate = False
                    continue

        if saw_user or (last_terminal_event is not None) or scan >= max_scan_bytes:
            break
        scan *= 2

    if not objs:
        return None

    if (not saw_user) and (last_terminal_event is None):
        return True if sz <= 128 * 1024 else False

    if turn_open:
        return bool(turn_has_completion_candidate)

    if last_terminal_event in ("assistant", "aborted"):
        return True
    if last_terminal_event == "user":
        return False
    return False


def _last_chat_role_ts_from_tail(
    path: Path,
    *,
    max_scan_bytes: int,
) -> tuple[str, float] | None:
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

    def has_assistant_text(obj: dict[str, Any]) -> bool:
        p = obj.get("payload") or {}
        if p.get("type") != "message" or p.get("role") != "assistant":
            return False
        content = p.get("content") or []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str) and part.get("text"):
                return True
        return False

    scan = 256 * 1024
    while scan <= max_scan_bytes:
        objs = _read_jsonl_tail(path, scan)
        last_user: tuple[int, float | None] | None = None
        last_assistant: tuple[int, float | None] | None = None
        for i, obj in enumerate(objs):
            typ = obj.get("type")
            if typ == "event_msg":
                p = obj.get("payload") or {}
                pt = p.get("type")
                if pt == "user_message" and isinstance(p.get("message"), str):
                    last_user = (i, event_ts(obj))
                    continue
                if pt == "agent_message":
                    msg = p.get("message")
                    if isinstance(msg, str) and msg.strip():
                        last_assistant = (i, event_ts(obj))
                        continue
            if typ == "response_item" and has_assistant_text(obj):
                last_assistant = (i, event_ts(obj))

        best: tuple[str, tuple[int, float | None]] | None = None
        if last_user is not None:
            best = ("user", last_user)
        if last_assistant is not None:
            if best is None or last_assistant[0] > best[1][0]:
                best = ("assistant", last_assistant)
        if best is not None:
            role, (_i, ts) = best
            if ts is None:
                return None
            return (role, float(ts))
        scan *= 2
    return None


@dataclass
class Session:
    session_id: str
    thread_id: str
    broker_pid: int
    codex_pid: int
    owned: bool
    start_ts: float
    cwd: str
    log_path: Path | None
    sock_path: Path
    busy: bool = False
    queue_len: int = 0
    token: dict[str, Any] | None = None
    last_turn_id: str | None = None
    last_chat_ts: float | None = None
    meta_thinking: int = 0
    meta_tools: int = 0
    meta_system: int = 0
    meta_log_off: int = 0
    chat_index_events: list[dict[str, Any]] = field(default_factory=list)
    chat_index_scan_bytes: int = 0
    chat_index_scan_complete: bool = False
    chat_index_log_off: int = 0
    idle_cache_log_off: int = -1
    idle_cache_value: bool | None = None


class SessionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._stop = threading.Event()
        self._last_discover_ts = 0.0
        self._harness: dict[str, dict[str, Any]] = {}
        self._harness_last_injected: dict[str, float] = {}
        self._harness_last_injected_scope: dict[str, float] = {}
        self._discover_existing(force=True)
        self._load_harness()
        self._harness_thr = threading.Thread(target=self._harness_loop, name="harness", daemon=True)
        self._harness_thr.start()

    def stop(self) -> None:
        self._stop.set()

    def _reset_log_caches(self, s: Session, *, meta_log_off: int) -> None:
        s.meta_thinking = 0
        s.meta_tools = 0
        s.meta_system = 0
        s.last_chat_ts = None
        s.meta_log_off = int(meta_log_off)
        s.chat_index_events = []
        s.chat_index_scan_bytes = 0
        s.chat_index_scan_complete = False
        s.chat_index_log_off = int(meta_log_off)
        s.idle_cache_log_off = -1
        s.idle_cache_value = None

    def _discover_existing_if_stale(self, *, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            last = float(getattr(self, "_last_discover_ts", 0.0))
        if (not force) and ((now - last) < DISCOVER_MIN_INTERVAL_SECONDS):
            return
        try:
            self._discover_existing(force=force)
        except TypeError:
            self._discover_existing()

    def _load_harness(self) -> None:
        try:
            raw = HARNESS_PATH.read_text(encoding="utf-8")
        except Exception:
            return
        try:
            obj = json.loads(raw)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        cleaned: dict[str, dict[str, Any]] = {}
        for sid, v in obj.items():
            if not isinstance(sid, str) or not sid:
                continue
            if not isinstance(v, dict):
                continue
            enabled = bool(v.get("enabled")) if "enabled" in v else False
            text = v.get("text")
            if not isinstance(text, str):
                text = ""
            cleaned[sid] = {"enabled": enabled, "text": text}
        with self._lock:
            self._harness = cleaned

    def _save_harness(self) -> None:
        with self._lock:
            obj = dict(self._harness)
        try:
            os.makedirs(APP_DIR, exist_ok=True)
            tmp = HARNESS_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, HARNESS_PATH)
        except Exception:
            return

    def harness_get(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            cfg = dict(self._harness.get(session_id) or {})
        enabled = bool(cfg.get("enabled"))
        text = cfg.get("text")
        if not isinstance(text, str):
            text = ""
        if not text.strip():
            text = HARNESS_DEFAULT_TEXT
        return {"enabled": enabled, "text": text}

    def harness_set(self, session_id: str, *, enabled: bool | None = None, text: str | None = None) -> dict[str, Any]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            cur = dict(self._harness.get(session_id) or {})
            if enabled is not None:
                cur["enabled"] = bool(enabled)
            if text is not None:
                cur["text"] = str(text)
            if bool(cur.get("enabled")) and (not isinstance(cur.get("text"), str) or (not str(cur.get("text")).strip())):
                cur["text"] = HARNESS_DEFAULT_TEXT
            self._harness[session_id] = cur
            if enabled is not None and bool(enabled) is False:
                self._harness_last_injected.pop(session_id, None)
        self._save_harness()
        return self.harness_get(session_id)

    def _harness_loop(self) -> None:
        # Persist across browser disconnects: server is the scheduler.
        while not self._stop.is_set():
            try:
                self._harness_sweep()
            except Exception:
                pass
            self._stop.wait(HARNESS_SWEEP_SECONDS)

    def _harness_sweep(self) -> None:
        now = time.time()
        # Keep discovery fresh; sessions can appear/disappear without UI polling.
        self._discover_existing_if_stale()
        self._prune_dead_sessions()
        with self._lock:
            items = [(sid, s, dict(self._harness.get(sid) or {}), float(self._harness_last_injected.get(sid, 0.0))) for sid, s in self._sessions.items()]

        for sid, s, cfg, last_inj in items:
            if not bool(cfg.get("enabled")):
                continue
            text = cfg.get("text")
            if not isinstance(text, str) or not text.strip():
                text = HARNESS_DEFAULT_TEXT
            lp = s.log_path
            if lp is None or (not lp.exists()):
                continue
            scope_key = f"thread:{s.thread_id}" if s.thread_id else f"log:{str(lp)}"
            with self._lock:
                scope_last = float(self._harness_last_injected_scope.get(scope_key, 0.0))
            if (last_inj and (now - last_inj) < float(HARNESS_IDLE_SECONDS)) or (scope_last and (now - scope_last) < float(HARNESS_IDLE_SECONDS)):
                continue
            try:
                st = self.get_state(sid)
                busy = bool(st.get("busy"))
                ql = int(st.get("queue_len", 0))
            except Exception:
                # Fail-busy: do not inject when state cannot be queried.
                continue
            if busy or ql > 0:
                continue
            last = _last_chat_role_ts_from_tail(lp, max_scan_bytes=HARNESS_MAX_SCAN_BYTES)
            if not last:
                continue
            role, ts = last
            if role != "assistant":
                continue
            if (now - float(ts)) < float(HARNESS_IDLE_SECONDS):
                continue
            try:
                with self._lock:
                    scope_last = float(self._harness_last_injected_scope.get(scope_key, 0.0))
                if scope_last and (now - scope_last) < float(HARNESS_IDLE_SECONDS):
                    continue
                self.send(sid, text)
            except Exception:
                continue
            with self._lock:
                self._harness_last_injected[sid] = now
                self._harness_last_injected_scope[scope_key] = now

    def _discover_existing(self, *, force: bool = False) -> None:
        if not force:
            now = time.time()
            with self._lock:
                last = float(self._last_discover_ts)
            if (now - last) < DISCOVER_MIN_INTERVAL_SECONDS:
                return
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

            log_path: Path | None = None
            if isinstance(meta.get("log_path"), str) and meta["log_path"]:
                log_path = Path(meta["log_path"])
            trace_path = None
            if isinstance(meta.get("trace_path"), str) and meta["trace_path"]:
                trace_path = Path(meta["trace_path"])
                if not trace_path.exists():
                    trace_path = None
            if trace_path is not None:
                lp2 = _discover_log_from_trace(trace_path)
                if lp2 is not None:
                    log_path = lp2
                    sid2 = _session_id_from_rollout_path(lp2)
                    if sid2:
                        thread_id = sid2
            if (log_path is None) or (not log_path.exists()):
                lp3 = _discover_log_for_session_id(thread_id)
                if lp3 is not None and lp3.exists():
                    log_path = lp3
            if log_path is not None and log_path.exists():
                thread_id, log_path = _coerce_main_thread_log(thread_id=thread_id, log_path=log_path)
            else:
                log_path = None

            if (log_path is None) and (not _pid_alive(codex_pid)) and (not _pid_alive(broker_pid)):
                _unlink_quiet(sock)
                _unlink_quiet(meta_path)
                continue

            cwd = meta.get("cwd") if isinstance(meta.get("cwd"), str) else ""
            if (not cwd) and (log_path is not None):
                sm = _read_session_meta(log_path)
                cwd = sm.get("cwd") if isinstance(sm.get("cwd"), str) else ""
            if not cwd:
                cwd = "?"

            start_ts = None
            if isinstance(meta.get("start_ts"), (int, float)):
                start_ts = float(meta["start_ts"])
            if start_ts is None and (log_path is not None):
                sm = _read_session_meta(log_path)
                if isinstance(sm.get("timestamp"), str):
                    start_ts = _parse_iso8601_to_epoch(sm["timestamp"])
            if start_ts is None and (log_path is not None):
                try:
                    start_ts = log_path.stat().st_mtime
                except Exception:
                    start_ts = None
            if start_ts is None:
                start_ts = _now()

            try:
                # Validate socket is responsive.
                resp = self._sock_call(sock, {"cmd": "state"}, timeout_s=0.5)
            except Exception as e:
                if _sock_error_definitely_stale(e):
                    _unlink_quiet(sock)
                    _unlink_quiet(meta_path)
                    continue
                if not _pid_alive(codex_pid) and not _pid_alive(broker_pid):
                    _unlink_quiet(sock)
                    _unlink_quiet(meta_path)
                resp = {}

            if log_path is not None:
                try:
                    meta_log_off = int(log_path.stat().st_size)
                except Exception:
                    meta_log_off = 0
            else:
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
                    self._reset_log_caches(s, meta_log_off=meta_log_off)
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
                        self._reset_log_caches(prev, meta_log_off=meta_log_off)
        with self._lock:
            self._last_discover_ts = time.time()

    def _refresh_session_state(self, session_id: str, sock_path: Path, timeout_s: float = 0.4) -> tuple[bool, BaseException | None]:
        try:
            resp = self._sock_call(sock_path, {"cmd": "state"}, timeout_s=timeout_s)
        except Exception as e:
            return False, e
        with self._lock:
            s2 = self._sessions.get(session_id)
            if s2 and "busy" in resp:
                s2.busy = bool(resp.get("busy"))
                s2.queue_len = int(resp.get("queue_len", s2.queue_len))
                if "token" in resp:
                    tok = resp.get("token")
                    if isinstance(tok, dict) or tok is None:
                        s2.token = tok
        return True, None

    def _prune_dead_sessions(self) -> None:
        with self._lock:
            items = list(self._sessions.items())
        dead: list[tuple[str, Path]] = []
        for sid, s in items:
            if not s.sock_path.exists():
                dead.append((sid, s.sock_path))
                continue
            ok, err = self._refresh_session_state(sid, s.sock_path, timeout_s=0.4)
            if ok:
                continue
            if err is not None and _sock_error_definitely_stale(err):
                dead.append((sid, s.sock_path))
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
            lp = s.log_path
            if lp is None or (not lp.exists()):
                continue

            try:
                sz = int(lp.stat().st_size)
            except Exception:
                sz = s.meta_log_off
            off = int(s.meta_log_off)
            if sz < off:
                off = 0

            total_th = 0
            total_tools = 0
            total_sys = 0
            latest_chat_ts: float | None = None
            latest_token: dict[str, Any] | None = None
            loops = 0
            while off < sz and loops < 16:
                objs, new_off = _read_jsonl_from_offset(lp, off, max_bytes=256 * 1024)
                if new_off <= off:
                    break
                d_th, d_tools, d_sys, chunk_chat_ts, token_update, _chat_events = _analyze_log_chunk(objs)
                total_th += d_th
                total_tools += d_tools
                total_sys += d_sys
                if chunk_chat_ts is not None:
                    latest_chat_ts = chunk_chat_ts if latest_chat_ts is None else max(latest_chat_ts, chunk_chat_ts)
                if token_update is not None:
                    latest_token = token_update
                off = new_off
                loops += 1

            with self._lock:
                s2 = self._sessions.get(sid)
                if not s2:
                    continue
                if latest_chat_ts is not None:
                    s2.last_chat_ts = latest_chat_ts if s2.last_chat_ts is None else max(s2.last_chat_ts, latest_chat_ts)
                if latest_token is not None:
                    s2.token = latest_token
                if s2.busy:
                    s2.meta_thinking += total_th
                    s2.meta_tools += total_tools
                    s2.meta_system += total_sys
                else:
                    s2.meta_thinking = 0
                    s2.meta_tools = 0
                    s2.meta_system = 0
                s2.meta_log_off = off if off >= 0 else s2.meta_log_off

    def list_sessions(self) -> list[dict[str, Any]]:
        # Rescan sockets to pick up sessions created before the server started.
        self._discover_existing_if_stale()
        self._prune_dead_sessions()
        self._update_meta_counters()
        with self._lock:
            out = []
            for s in self._sessions.values():
                h_enabled = bool((self._harness.get(s.session_id) or {}).get("enabled"))
                if s.last_chat_ts is None and s.log_path is not None and s.log_path.exists():
                    try:
                        s.last_chat_ts = float(s.log_path.stat().st_mtime)
                    except Exception:
                        s.last_chat_ts = None
                updated_ts = float(s.last_chat_ts) if isinstance(s.last_chat_ts, (int, float)) else float(s.start_ts)
                out.append(
                    {
                        "session_id": s.session_id,
                        "thread_id": s.thread_id,
                        "pid": s.codex_pid,
                        "broker_pid": s.broker_pid,
                        "owned": s.owned,
                        "cwd": s.cwd,
                        "start_ts": s.start_ts,
                        "updated_ts": updated_ts,
                        "log_path": (str(s.log_path) if s.log_path is not None else None),
                        "busy": s.busy,
                        "queue_len": s.queue_len,
                        "token": s.token,
                        "thinking": s.meta_thinking,
                        "tools": s.meta_tools,
                        "system": s.meta_system,
                        "harness_enabled": h_enabled,
                    }
                )
            return out

    def get_session(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def refresh_session_meta(self, session_id: str) -> None:
        # The broker may rewrite the sock .json when Codex switches threads (/new, /resume).
        # Refresh the log path and thread id without requiring the UI to poll /api/sessions.
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
        trace_path = None
        if isinstance(meta.get("trace_path"), str) and meta["trace_path"]:
            trace_path = Path(meta["trace_path"])
            if not trace_path.exists():
                trace_path = None
        if trace_path is not None:
            lp2 = _discover_log_from_trace(trace_path)
            if lp2 is not None and lp2.exists():
                log_path = lp2
                sid2 = _session_id_from_rollout_path(lp2)
                if sid2:
                    thread_id = sid2
        if not log_path or not log_path.exists():
            log_path = _discover_log_for_session_id(thread_id)
        if not log_path or not log_path.exists():
            return

        thread_id, log_path = _coerce_main_thread_log(thread_id=thread_id, log_path=log_path)

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
                try:
                    log_off = int(log_path.stat().st_size)
                except Exception:
                    log_off = 0
                self._reset_log_caches(s2, meta_log_off=log_off)

    def _set_chat_index_snapshot(
        self,
        *,
        session_id: str,
        events: list[dict[str, Any]],
        token_update: dict[str, Any] | None,
        scan_bytes: int,
        scan_complete: bool,
        log_off: int,
    ) -> None:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return
            s.chat_index_events = list(events[-CHAT_INDEX_MAX_EVENTS:])
            s.chat_index_scan_bytes = int(scan_bytes)
            s.chat_index_scan_complete = bool(scan_complete) and (len(events) <= CHAT_INDEX_MAX_EVENTS)
            s.chat_index_log_off = int(log_off)
            if token_update is not None:
                s.token = token_update

    def _append_chat_events(self, session_id: str, new_events: list[dict[str, Any]], *, new_off: int, latest_token: dict[str, Any] | None) -> None:
        if not new_events and latest_token is None:
            with self._lock:
                s = self._sessions.get(session_id)
                if s:
                    s.chat_index_log_off = int(new_off)
            return
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return
            if new_events:
                merged = s.chat_index_events + new_events
                if len(merged) > CHAT_INDEX_MAX_EVENTS:
                    merged = merged[-CHAT_INDEX_MAX_EVENTS:]
                    s.chat_index_scan_complete = False
                s.chat_index_events = merged
                for ev in new_events:
                    ts = ev.get("ts")
                    if isinstance(ts, (int, float)):
                        tsf = float(ts)
                        s.last_chat_ts = tsf if s.last_chat_ts is None else max(s.last_chat_ts, tsf)
            s.chat_index_log_off = int(new_off)
            if latest_token is not None:
                s.token = latest_token

    def _ensure_chat_index(self, session_id: str, *, min_events: int, before: int) -> tuple[list[dict[str, Any]], int, bool, int, dict[str, Any] | None]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return [], 0, False, 0, None
            lp = s.log_path
            scan_bytes = int(s.chat_index_scan_bytes) if s.chat_index_scan_bytes > 0 else CHAT_INIT_SEED_SCAN_BYTES
            idx_off = int(s.chat_index_log_off)
        if lp is None or (not lp.exists()):
            return [], 0, False, 0, None

        try:
            sz = int(lp.stat().st_size)
        except Exception:
            sz = idx_off

        if sz < idx_off:
            idx_off = 0
            self._set_chat_index_snapshot(
                session_id=session_id,
                events=[],
                token_update=None,
                scan_bytes=CHAT_INIT_SEED_SCAN_BYTES,
                scan_complete=False,
                log_off=0,
            )

        with self._lock:
            s2 = self._sessions.get(session_id)
            ready = bool(s2 and ((s2.chat_index_events is not None)))
            cached_count = len(s2.chat_index_events) if s2 else 0
            scan_complete = bool(s2.chat_index_scan_complete) if s2 else False

        target_events = max(0, int(min_events) + max(0, int(before)))
        if (not ready) or ((target_events > cached_count) and (not scan_complete)):
            events, token_update, used_scan, complete, log_size = _read_chat_tail_snapshot(
                lp,
                min_events=max(20, target_events),
                initial_scan_bytes=max(CHAT_INIT_SEED_SCAN_BYTES, scan_bytes),
                max_scan_bytes=CHAT_INIT_MAX_SCAN_BYTES,
            )
            self._set_chat_index_snapshot(
                session_id=session_id,
                events=events,
                token_update=token_update,
                scan_bytes=used_scan,
                scan_complete=complete,
                log_off=log_size,
            )

        with self._lock:
            s3 = self._sessions.get(session_id)
            if not s3:
                return [], 0, False, 0, None
            lp3 = s3.log_path
            off3 = int(s3.chat_index_log_off)
            prev_events = list(s3.chat_index_events)
        if lp3 is None or (not lp3.exists()):
            return [], off3, False, 0, None

        try:
            sz2 = int(lp3.stat().st_size)
        except Exception:
            sz2 = off3

        if sz2 > off3:
            delta = sz2 - off3
            if delta >= CHAT_INDEX_RESEED_THRESHOLD_BYTES:
                events, token_update, used_scan, complete, log_size = _read_chat_tail_snapshot(
                    lp3,
                    min_events=max(20, target_events),
                    initial_scan_bytes=max(CHAT_INIT_SEED_SCAN_BYTES, scan_bytes),
                    max_scan_bytes=CHAT_INIT_MAX_SCAN_BYTES,
                )
                self._set_chat_index_snapshot(
                    session_id=session_id,
                    events=events,
                    token_update=token_update,
                    scan_bytes=used_scan,
                    scan_complete=complete,
                    log_off=log_size,
                )
            else:
                cur = off3
                loops = 0
                latest_token: dict[str, Any] | None = None
                aggregated_events: list[dict[str, Any]] = []
                while cur < sz2 and loops < 16:
                    objs, new_off = _read_jsonl_from_offset(lp3, cur, max_bytes=CHAT_INDEX_INCREMENT_BYTES)
                    if new_off <= cur:
                        break
                    _th, _tools, _sys, _last_ts, token_update, new_events = _analyze_log_chunk(objs)
                    if token_update is not None:
                        latest_token = token_update
                    if new_events:
                        aggregated_events.extend(new_events)
                    cur = new_off
                    loops += 1
                self._append_chat_events(session_id, aggregated_events, new_off=cur, latest_token=latest_token)

        with self._lock:
            s4 = self._sessions.get(session_id)
            if not s4:
                return prev_events, off3, False, 0, None
            events2 = list(s4.chat_index_events)
            log_off2 = int(s4.chat_index_log_off)
            scan_complete2 = bool(s4.chat_index_scan_complete)
            token2 = s4.token if isinstance(s4.token, dict) or s4.token is None else None

        n = len(events2)
        b = max(0, int(before))
        end = max(0, n - b)
        start = max(0, end - max(20, int(min_events)))
        page = events2[start:end]
        has_older = (start > 0) or ((not scan_complete2) and bool(page))
        next_before = b + len(page) if has_older else 0
        return page, log_off2, has_older, next_before, token2

    def mark_log_delta(self, session_id: str, *, objs: list[dict[str, Any]], new_off: int) -> None:
        _th, _tools, _sys, _last_ts, token_update, new_events = _analyze_log_chunk(objs)
        self._append_chat_events(session_id, new_events, new_off=new_off, latest_token=token_update)
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s.idle_cache_log_off = -1

    def busy_from_log_fallback(self, session_id: str) -> bool:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return True
            lp = s.log_path
            cached_off = int(s.idle_cache_log_off)
            cached_idle = s.idle_cache_value
        if lp is None or (not lp.exists()):
            return True
        try:
            sz = int(lp.stat().st_size)
        except Exception:
            sz = -1
        if (sz >= 0) and (cached_off == sz) and isinstance(cached_idle, bool):
            return not bool(cached_idle)
        idle = _compute_idle_from_log(lp)
        with self._lock:
            s2 = self._sessions.get(session_id)
            if s2:
                s2.idle_cache_log_off = sz
                s2.idle_cache_value = idle
        if idle is None:
            with self._lock:
                s3 = self._sessions.get(session_id)
                return bool(s3.busy) if s3 else True
        return not bool(idle)

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
        use_broker = _strace_usable()
        mod = "codoxear.broker" if use_broker else "codoxear.sessiond"
        argv = [sys.executable, "-m", mod, "--cwd", cwd, "--"]
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
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
        except Exception as e:
            raise RuntimeError(f"spawn failed: {e}") from e

        _wait_or_raise(proc, label=("broker" if use_broker else "sessiond"), timeout_s=1.5)
        if proc.stderr is not None:
            threading.Thread(target=_drain_stream, args=(proc.stderr,), daemon=True).start()

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
        elif path.suffix == ".png":
            ctype = "image/png"
        elif path.suffix in (".jpg", ".jpeg"):
            ctype = "image/jpeg"
        elif path.suffix == ".webp":
            ctype = "image/webp"
        elif path.suffix == ".svg":
            ctype = "image/svg+xml; charset=utf-8"
        elif path.suffix == ".ico":
            ctype = "image/x-icon"
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
            if path == "/favicon.ico":
                self._send_static("favicon.png")
                return
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
                t0 = time.perf_counter()
                sessions = MANAGER.list_sessions()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                _record_metric("api_sessions_ms", dt_ms)
                _json_response(self, 200, {"sessions": sessions})
                return

            if path == "/api/metrics":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                _json_response(self, 200, {"metrics": _metrics_snapshot()})
                return

            if path.startswith("/api/sessions/") and path.endswith("/messages"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                t0_total = time.perf_counter()
                parts = path.split("/")
                if len(parts) < 4:
                    self.send_error(404)
                    return
                session_id = parts[3]
                t0_meta = time.perf_counter()
                MANAGER.refresh_session_meta(session_id)
                dt_meta_ms = (time.perf_counter() - t0_meta) * 1000.0
                s = MANAGER.get_session(session_id)
                if not s:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                qs = urllib.parse.parse_qs(u.query)
                try:
                    offset = int((qs.get("offset") or ["0"])[0])
                except Exception:
                    offset = 0
                if offset < 0:
                    offset = 0
                init = (qs.get("init") or ["0"])[0] == "1"
                try:
                    before = int((qs.get("before") or ["0"])[0])
                except Exception:
                    before = 0
                before = max(0, before)
                if s.log_path is None or (not s.log_path.exists()):
                    try:
                        state = MANAGER.get_state(session_id)
                    except Exception:
                        state = None
                    state_busy = bool(state.get("busy")) if isinstance(state, dict) and ("busy" in state) else None
                    state_queue = int(state.get("queue_len")) if isinstance(state, dict) and ("queue_len" in state) else None
                    state_token = state.get("token") if isinstance(state, dict) and ("token" in state) else None
                    busy_val = state_busy if state_busy is not None else bool(s.busy)
                    queue_val = state_queue if state_queue is not None else int(s.queue_len)
                    token_val = state_token if (isinstance(state_token, dict) or state_token is None) else None
                    _json_response(
                        self,
                        200,
                        {
                            "thread_id": s.thread_id,
                            "log_path": None,
                            "offset": 0,
                            "events": [],
                            "meta_delta": {"thinking": 0, "tool": 0, "system": 0},
                            "turn_start": False,
                            "turn_end": False,
                            "turn_aborted": False,
                            "diag": {"pending_log": True},
                            "busy": bool(busy_val),
                            "queue_len": int(queue_val),
                            "token": token_val,
                            "has_older": False,
                            "next_before": 0,
                        },
                    )
                    dt_total_ms = (time.perf_counter() - t0_total) * 1000.0
                    _record_metric("api_messages_init_ms" if init else "api_messages_poll_ms", dt_total_ms)
                    return

                if init and offset == 0:
                    limit_raw = (qs.get("limit") or ["80"])[0]
                    try:
                        limit = int(limit_raw)
                    except Exception:
                        limit = 80
                    limit = max(20, min(200, limit))
                    t0_index = time.perf_counter()
                    events, new_off, has_older, next_before, token_update = MANAGER._ensure_chat_index(
                        session_id,
                        min_events=limit,
                        before=before,
                    )
                    dt_index_ms = (time.perf_counter() - t0_index) * 1000.0
                    meta_delta = {"thinking": 0, "tool": 0, "system": 0}
                    flags = {"turn_start": False, "turn_end": False, "turn_aborted": False}
                    diag = {"tool_names": [], "last_tool": None, "init_index_ms": round(dt_index_ms, 3)}
                else:
                    has_older = False
                    next_before = 0
                    objs, new_off = _read_jsonl_from_offset(s.log_path, offset)
                    events, meta_delta, flags, diag = _extract_chat_events(objs)
                    token_update = _extract_token_update(objs)
                    MANAGER.mark_log_delta(session_id, objs=objs, new_off=new_off)
                t0_state = time.perf_counter()
                try:
                    state = MANAGER.get_state(session_id)
                except Exception:
                    state = None
                dt_state_ms = (time.perf_counter() - t0_state) * 1000.0
                s2 = MANAGER.get_session(session_id)
                if token_update is not None and s2 is not None:
                    s2.token = token_update
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

                if state_busy is not None:
                    busy_val = bool(state_busy)
                else:
                    t0_idle = time.perf_counter()
                    busy_val = MANAGER.busy_from_log_fallback(session_id)
                    dt_idle_ms = (time.perf_counter() - t0_idle) * 1000.0
                    diag["busy_fallback_ms"] = round(dt_idle_ms, 3)

                queue_val = state_queue if state_queue is not None else (s2.queue_len if s2 else 0)

                if state_token is not token_sentinel and (isinstance(state_token, dict) or state_token is None):
                    token_val = state_token
                else:
                    token_val = s2.token if s2 else None
                diag["state_ms"] = round(dt_state_ms, 3)
                diag["meta_refresh_ms"] = round(dt_meta_ms, 3)
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
                        "has_older": bool(has_older),
                        "next_before": int(next_before),
                    },
                )
                dt_total_ms = (time.perf_counter() - t0_total) * 1000.0
                _record_metric("api_messages_init_ms" if init else "api_messages_poll_ms", dt_total_ms)
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

            if path.startswith("/api/sessions/") and path.endswith("/harness"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                if len(parts) < 4:
                    self.send_error(404)
                    return
                session_id = parts[3]
                try:
                    cfg = MANAGER.harness_get(session_id)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, {"ok": True, **cfg})
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

            if path.startswith("/api/sessions/") and path.endswith("/harness"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                body = _read_body(self)
                obj = json.loads(body.decode("utf-8") or "{}")
                enabled_raw = obj.get("enabled", None)
                text_raw = obj.get("text", None)
                enabled: bool | None
                if enabled_raw is None:
                    enabled = None
                else:
                    enabled = bool(enabled_raw)
                text: str | None
                if text_raw is None:
                    text = None
                elif isinstance(text_raw, str):
                    text = text_raw
                else:
                    _json_response(self, 400, {"error": "text must be a string"})
                    return
                cfg = MANAGER.harness_set(session_id, enabled=enabled, text=text)
                _json_response(self, 200, {"ok": True, **cfg})
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
