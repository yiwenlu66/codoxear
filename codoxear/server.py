#!/usr/bin/env python3
from __future__ import annotations

import base64
import errno
import hashlib
import hmac
import http.server
import io
import json
import math
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
from typing import Any

from . import rollout_log as _rollout_log
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
    data = path.read_text("utf-8")

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


def _normalize_url_prefix(raw: str | None) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s == "/":
        return ""
    if "://" in s:
        raise ValueError("CODEX_WEB_URL_PREFIX must be a path prefix (not a URL)")
    if "?" in s or "#" in s:
        raise ValueError("CODEX_WEB_URL_PREFIX must not include '?' or '#'")
    if not s.startswith("/"):
        raise ValueError("CODEX_WEB_URL_PREFIX must start with '/'")
    while len(s) > 1 and s.endswith("/"):
        s = s[:-1]
    if s == "/":
        return ""
    return s


def _strip_url_prefix(prefix: str, path: str) -> str | None:
    if not prefix:
        return path
    if path == prefix:
        return "/"
    if path.startswith(prefix + "/"):
        return path[len(prefix) :]
    return None


APP_DIR = _default_app_dir()
STATIC_DIR = Path(__file__).resolve().parent / "static"
SOCK_DIR = APP_DIR / "socks"
STATE_PATH = APP_DIR / "state.json"
HMAC_SECRET_PATH = APP_DIR / "hmac_secret"
UPLOAD_DIR = APP_DIR / "uploads"
HARNESS_PATH = APP_DIR / "harness.json"
ALIAS_PATH = APP_DIR / "session_aliases.json"
SIDEBAR_META_PATH = APP_DIR / "session_sidebar.json"
HIDDEN_SESSIONS_PATH = APP_DIR / "hidden_sessions.json"
FILE_HISTORY_PATH = APP_DIR / "session_files.json"
QUEUE_PATH = APP_DIR / "session_queues.json"

_DOTENV = (Path.cwd() / ".env").resolve()
if _DOTENV.exists():
    for _k, _v in _load_env_file(_DOTENV).items():
        os.environ.setdefault(_k, _v)

COOKIE_NAME = "codoxear_auth"
COOKIE_TTL_SECONDS = int(os.environ.get("CODEX_WEB_COOKIE_TTL_SECONDS", str(30 * 24 * 3600)))
COOKIE_SECURE = os.environ.get("CODEX_WEB_COOKIE_SECURE", "0") == "1"
URL_PREFIX = _normalize_url_prefix(os.environ.get("CODEX_WEB_URL_PREFIX"))
COOKIE_PATH = (URL_PREFIX + "/") if URL_PREFIX else "/"

_CODEX_HOME_ENV = os.environ.get("CODEX_HOME")
if _CODEX_HOME_ENV is None or (not _CODEX_HOME_ENV.strip()):
    CODEX_HOME = Path.home() / ".codex"
else:
    CODEX_HOME = Path(_CODEX_HOME_ENV)
CODEX_SESSIONS_DIR = CODEX_HOME / "sessions"

DEFAULT_HOST = os.environ.get("CODEX_WEB_HOST", "::")
DEFAULT_PORT = int(os.environ.get("CODEX_WEB_PORT", "8743"))
HARNESS_IDLE_SECONDS = int(os.environ.get("CODEX_WEB_HARNESS_IDLE_SECONDS", "300"))
HARNESS_SWEEP_SECONDS = float(os.environ.get("CODEX_WEB_HARNESS_SWEEP_SECONDS", "2.5"))
QUEUE_SWEEP_SECONDS = float(os.environ.get("CODEX_WEB_QUEUE_SWEEP_SECONDS", "1.0"))
QUEUE_IDLE_GRACE_SECONDS = float(os.environ.get("CODEX_WEB_QUEUE_IDLE_GRACE_SECONDS", "10.0"))
HARNESS_MAX_SCAN_BYTES = int(os.environ.get("CODEX_WEB_HARNESS_MAX_SCAN_BYTES", str(8 * 1024 * 1024)))
DISCOVER_MIN_INTERVAL_SECONDS = float(os.environ.get("CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS", "1.0"))
CHAT_INIT_SEED_SCAN_BYTES = int(os.environ.get("CODEX_WEB_CHAT_INIT_SEED_SCAN_BYTES", str(512 * 1024)))
CHAT_INIT_MAX_SCAN_BYTES = int(os.environ.get("CODEX_WEB_CHAT_INIT_MAX_SCAN_BYTES", str(128 * 1024 * 1024)))
CHAT_INDEX_INCREMENT_BYTES = int(os.environ.get("CODEX_WEB_CHAT_INDEX_INCREMENT_BYTES", str(2 * 1024 * 1024)))
CHAT_INDEX_RESEED_THRESHOLD_BYTES = int(os.environ.get("CODEX_WEB_CHAT_INDEX_RESEED_THRESHOLD_BYTES", str(16 * 1024 * 1024)))
CHAT_INDEX_MAX_EVENTS = int(os.environ.get("CODEX_WEB_CHAT_INDEX_MAX_EVENTS", "12000"))
METRICS_WINDOW = int(os.environ.get("CODEX_WEB_METRICS_WINDOW", "256"))
FILE_READ_MAX_BYTES = int(os.environ.get("CODEX_WEB_FILE_READ_MAX_BYTES", str(2 * 1024 * 1024)))
FILE_HISTORY_MAX = int(os.environ.get("CODEX_WEB_FILE_HISTORY_MAX", "20"))
GIT_DIFF_MAX_BYTES = int(os.environ.get("CODEX_WEB_GIT_DIFF_MAX_BYTES", str(800 * 1024)))
GIT_DIFF_TIMEOUT_SECONDS = float(os.environ.get("CODEX_WEB_GIT_DIFF_TIMEOUT_SECONDS", "4.0"))
GIT_WORKTREE_TIMEOUT_SECONDS = float(os.environ.get("CODEX_WEB_GIT_WORKTREE_TIMEOUT_SECONDS", "10.0"))
GIT_CHANGED_FILES_MAX = int(os.environ.get("CODEX_WEB_GIT_CHANGED_FILES_MAX", "400"))
SIDEBAR_PRIORITY_HALF_LIFE_SECONDS = 8.0 * 3600.0
ATTACH_UPLOAD_MAX_BYTES = int(os.environ.get("CODEX_WEB_ATTACH_MAX_BYTES", str(10 * 1024 * 1024)))
SIDEBAR_PRIORITY_LAMBDA = math.log(2.0) / SIDEBAR_PRIORITY_HALF_LIFE_SECONDS
HARNESS_PROMPT_PREFIX = """Unattended-mode instructions (optimize for 8+ hours, minimal turns, minimal repetition, maximal progress)

- Maintain three live lists: Deliverables, Next actions, Parked questions.
- Default is to keep working in the same turn; do not yield while any Next actions remain.

- Question handling:
  - If blocked on a decision, write it to Parked questions (question, why it matters, what is blocked).
  - Immediately continue with unblocked work; do not wait for an answer.
  - Surface parked questions only when truly blocked on user-only input, preferably at the end as a compact list.

- Progress loop (repeat):
  - Choose the highest-leverage unblocked action.
  - Execute it.
  - Produce evidence using the strongest available verification that matches the request (full tests/builds/real runs/logs over minimal checks).
  - Update lists and continue.

- Long-running work:
  - If you start a long task, actively monitor it (poll status/output/logs), diagnose stalls, and either fix, restart, or reroute to other unblocked work.
  - Never claim background monitoring without returning observed state.

- Anti-repetition:
  - Do not repeat the same command/edit/analysis unless you can name the new evidence you expect.
  - If you detect a loop, change strategy: new hypothesis, new subsystem boundary, new tool, or new verification path.

- Turn minimization:
  - Avoid mid-flight questions, progress check-ins, and "want me to do X next" prompts.
  - Yield only when all deliverables are evidenced complete, or every remaining action is blocked by user-only input, or the next step is irreversible/high-risk.

- End-of-turn gate (only when yielding is necessary):
  - Run an extensive clean-room adversarial review via a dedicated subagent.
  - Prompt the subagent with: original user intent, project architectural constraints/invariants, deliverables, objective evidence, changed artifacts (no approach narrative).
  - Apply findings before yielding (or surface a concrete blocker/risk).
"""


def _render_harness_prompt(request: str | None) -> str:
    base = HARNESS_PROMPT_PREFIX.rstrip()
    r = (request or "").strip()
    if not r:
        return base + "\n"
    return base + "\n\n---\n\nAdditional request from user: " + r + "\n"

_SESSION_ID_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)
_METRICS_LOCK = threading.Lock()
_METRICS: dict[str, list[float]] = {}


def _record_metric(name: str, value_ms: float) -> None:
    if not isinstance(name, str) or not name:
        return
    v = float(value_ms)
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


def _wait_or_raise(proc: subprocess.Popen[bytes], *, label: str, timeout_s: float = 1.5) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        rc = proc.poll()
        if rc is None:
            time.sleep(0.05)
            continue
        _out, err = proc.communicate(timeout=0.5)
        err2 = err if isinstance(err, (bytes, bytearray)) else b""
        msg = bytes(err2).decode("utf-8", errors="replace").strip()
        msg = msg[-4000:] if msg else ""
        raise RuntimeError(f"{label} exited early (rc={rc}): {msg}")


def _drain_stream(f: Any) -> None:
    while True:
        b = f.read(65536)
        if not b:
            break
    f.close()


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


def _sock_error_definitely_stale(exc: BaseException) -> bool:
    if isinstance(exc, (FileNotFoundError, ConnectionRefusedError)):
        return True
    if isinstance(exc, OSError):
        return exc.errno in (errno.ENOENT, errno.ECONNREFUSED, errno.ENOTSOCK)
    return False


def _extract_token_update(objs: list[dict[str, Any]]) -> dict[str, Any] | None:
    return _rollout_log._extract_token_update(objs)


def _sniff_image_ext(raw: bytes) -> str | None:
    if len(raw) >= 8 and raw[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if len(raw) >= 3 and raw[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return ".webp"
    return None


def _image_content_type(path: Path, raw: bytes) -> str | None:
    if path.suffix.lower() == ".svg":
        return "image/svg+xml; charset=utf-8"
    ext = _sniff_image_ext(raw)
    if ext == ".png":
        return "image/png"
    if ext == ".jpg":
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return None


def _file_kind(path: Path, raw: bytes) -> tuple[str, str | None]:
    ctype = _image_content_type(path, raw)
    if ctype is not None:
        return "image", ctype
    return "text", None


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
    cl = handler.headers.get("Content-Length")
    if cl is None:
        cl = "0"
    cl2 = str(cl).strip()
    if not cl2:
        cl2 = "0"
    n = int(cl2)
    if n < 0 or n > limit:
        raise ValueError(f"invalid content-length: {n}")
    return handler.rfile.read(n)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_or_create_hmac_secret() -> bytes:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    if HMAC_SECRET_PATH.exists():
        b = HMAC_SECRET_PATH.read_bytes()
        if len(b) < 32:
            raise ValueError(f"invalid hmac secret (too short): {HMAC_SECRET_PATH}")
        return b[:64]
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
        exp_raw = payload.get("exp")
        if exp_raw is None:
            return None
        exp = int(exp_raw)
        if exp <= int(_now()):
            return None
        return payload
    except (TypeError, ValueError, json.JSONDecodeError):
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
        f"Path={COOKIE_PATH}",
        "HttpOnly",
        "SameSite=Strict",
        f"Max-Age={COOKIE_TTL_SECONDS}",
    ]
    forwarded_proto_raw = handler.headers.get("X-Forwarded-Proto")
    forwarded_proto = str(forwarded_proto_raw).lower() if forwarded_proto_raw is not None else ""
    if COOKIE_SECURE or forwarded_proto == "https":
        attrs.append("Secure")
    handler.send_header("Set-Cookie", "; ".join(attrs))

_PASSWORD_CACHE: str | None = None


def _require_password() -> str:
    global _PASSWORD_CACHE
    if _PASSWORD_CACHE is not None:
        return _PASSWORD_CACHE
    pw_raw = os.environ.get("CODEX_WEB_PASSWORD")
    pw = str(pw_raw).strip() if pw_raw is not None else ""
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


def _read_text_file_strict(path: Path, *, max_bytes: int) -> tuple[str, int]:
    st = path.stat()
    size = int(st.st_size)
    if size > max_bytes:
        raise ValueError(f"file too large (max {max_bytes} bytes)")
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError("binary file not supported")
    text = data.decode("utf-8", errors="replace")
    return text, size


def _resolve_under(base: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("path required")
    if "\x00" in rel:
        raise ValueError("invalid path")
    p = Path(rel)
    if p.is_absolute():
        raise ValueError("path must be relative")
    resolved_base = base.resolve()
    resolved = (resolved_base / p).resolve()
    if not str(resolved).startswith(str(resolved_base) + os.sep) and resolved != resolved_base:
        raise ValueError("path escapes session cwd")
    return resolved


def _resolve_session_path(base: Path, raw_path: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("path required")
    if "\x00" in raw_path:
        raise ValueError("invalid path")
    p = Path(raw_path)
    if p.is_absolute():
        return p.expanduser().resolve()
    resolved_base = base.expanduser()
    if not resolved_base.is_absolute():
        resolved_base = resolved_base.resolve()
    return (resolved_base / p).resolve()


def _resolve_git_path(cwd: Path, raw_path: str) -> tuple[Path, Path, str]:
    repo_root = Path(_run_git(cwd, ["rev-parse", "--show-toplevel"], timeout_s=GIT_DIFF_TIMEOUT_SECONDS, max_bytes=64 * 1024).strip()).resolve()
    target = _resolve_session_path(cwd, raw_path)
    try:
        rel = str(target.relative_to(repo_root))
    except ValueError as e:
        raise ValueError("path is outside git repo") from e
    return target, repo_root, rel


def _resolve_unique_bare_filename(search_root: Path, raw_path: str) -> Path | None:
    name = str(raw_path).strip()
    if not name or "/" in name or "\\" in name or "\x00" in name:
        return None
    if "." not in Path(name).name:
        return None
    root = search_root.resolve()
    match: Path | None = None
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", ".hg", ".svn", "__pycache__", "node_modules", "build", "dist"}]
        if name not in filenames:
            continue
        candidate = (Path(current_root) / name).resolve()
        if match is None:
            match = candidate
            continue
        if candidate != match:
            return None
    return match


def _resolve_tracked_file_by_basename(session_id: str, raw_path: str) -> Path | None:
    name = str(raw_path).strip()
    if not name or "/" in name or "\\" in name or "\x00" in name:
        return None
    try:
        tracked = MANAGER.files_get(session_id)
    except KeyError:
        return None
    match: Path | None = None
    for raw in tracked:
        candidate = Path(raw).expanduser().resolve()
        if candidate.name != name:
            continue
        if match is None:
            match = candidate
            continue
        if candidate != match:
            return None
    return match


def _run_git(cwd: Path, args: list[str], *, timeout_s: float, max_bytes: int) -> str:
    cmd = ["git", *args]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(err or f"git failed with code {proc.returncode}")
    if len(proc.stdout) > max_bytes:
        raise ValueError(f"git output too large (max {max_bytes} bytes)")
    return proc.stdout.decode("utf-8", errors="replace")


def _expand_user_path(raw: str) -> Path:
    home = str(Path.home())
    expanded = raw.strip().replace("${HOME}", home)
    expanded = re.sub(r"\$HOME(?![A-Za-z0-9_])", home, expanded)
    return Path(os.path.expanduser(os.path.expandvars(expanded)))


def _resolve_existing_dir(raw: str, *, field_name: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field_name} required")
    path = _expand_user_path(raw)
    if not path.is_dir():
        raise ValueError(f"{field_name} is not a directory: {path}")
    return path.resolve()


def _resolve_new_path(raw: str, *, field_name: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field_name} required")
    path = _expand_user_path(raw).resolve()
    if path.exists():
        raise ValueError(f"{field_name} already exists: {path}")
    return path


def _clean_worktree_branch(raw: str) -> str:
    if not isinstance(raw, str):
        raise ValueError("worktree_branch must be a string")
    branch = raw.strip()
    if not branch:
        raise ValueError("worktree_branch required")
    return branch


def _require_git_repo(cwd: Path) -> None:
    _run_git(cwd, ["rev-parse", "--is-inside-work-tree"], timeout_s=GIT_DIFF_TIMEOUT_SECONDS, max_bytes=4096)


def _git_repo_root(cwd: Path) -> Path | None:
    try:
        root = _run_git(cwd, ["rev-parse", "--show-toplevel"], timeout_s=GIT_DIFF_TIMEOUT_SECONDS, max_bytes=64 * 1024).strip()
    except (RuntimeError, FileNotFoundError):
        return None
    if not root:
        return None
    return Path(root).resolve()


def _describe_session_cwd(cwd: Path) -> dict[str, Any]:
    repo_root = _git_repo_root(cwd)
    git_branch = _current_git_branch(cwd) or ""
    return {
        "cwd": str(cwd),
        "git_repo": repo_root is not None,
        "git_root": str(repo_root) if repo_root is not None else "",
        "git_branch": git_branch,
    }


def _worktree_path_slug(branch: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", branch).strip(".-")
    return slug or "worktree"


def _default_worktree_path(source_cwd: Path, branch: str) -> Path:
    slug = _worktree_path_slug(branch)
    return (source_cwd.parent / f"{source_cwd.name}-{slug}").resolve()


def _create_git_worktree(source_cwd: Path, worktree_branch: str) -> Path:
    repo_root = _git_repo_root(source_cwd)
    if repo_root is None:
        raise ValueError("cwd is not inside a git worktree")
    branch = _clean_worktree_branch(worktree_branch)
    target = _default_worktree_path(source_cwd, branch)
    if target.exists():
        raise ValueError(f"derived worktree path already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            ["git", "worktree", "add", "-b", branch, str(target)],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=GIT_WORKTREE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise ValueError("git worktree add timed out") from e
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        out = proc.stdout.decode("utf-8", errors="replace").strip()
        raise ValueError(err or out or f"git worktree add failed with code {proc.returncode}")
    return target.resolve()


def _parse_git_numstat(text: str) -> dict[str, dict[str, int | None]]:
    out: dict[str, dict[str, int | None]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        add_raw, del_raw, path = parts
        path_s = path.strip()
        if not path_s:
            continue
        add_v = None if add_raw == "-" else int(add_raw)
        del_v = None if del_raw == "-" else int(del_raw)
        prev = out.get(path_s)
        if prev is None:
            out[path_s] = {"additions": add_v, "deletions": del_v}
            continue
        if add_v is None or prev["additions"] is None:
            prev["additions"] = None
        else:
            prev["additions"] = int(prev["additions"]) + add_v
        if del_v is None or prev["deletions"] is None:
            prev["deletions"] = None
        else:
            prev["deletions"] = int(prev["deletions"]) + del_v
    return out


def _safe_filename(name: str, *, default: str = "file") -> str:
    out = []
    base = Path(str(name or "")).name
    for ch in base:
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            out.append(ch)
    s = "".join(out).strip().replace(" ", "_")
    if not s:
        return default
    return s[:96]


def _clean_alias(name: str) -> str:
    if not isinstance(name, str):
def _stage_uploaded_file(session_id: str, filename: str, raw: bytes, *, max_bytes: int = ATTACH_UPLOAD_MAX_BYTES) -> Path:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id required")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename required")
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError("file bytes required")
    data = bytes(raw)
    if len(data) > int(max_bytes):
        raise ValueError("file too large")
    safe_name = _safe_filename(filename, default="file")
    subdir = (UPLOAD_DIR / session_id).resolve()
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = (subdir / f"{int(_now() * 1000)}_{safe_name}").resolve()
    if not str(out_path).startswith(str(subdir) + os.sep):
        raise ValueError("bad path")
    out_path.write_bytes(data)
    os.chmod(out_path, 0o600)
    return out_path


def _attachment_inject_text(attachment_index: int, path: Path) -> str:
    idx = int(attachment_index)
    if idx <= 0:
        raise ValueError("attachment_index must be >= 1")
    return f"Attachment {idx}: {path}\n"


        return ""
    # Collapse whitespace and cap length to keep titles readable.
    cleaned = " ".join(name.split()).strip()
    if not cleaned:
        return ""
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rstrip()
    return cleaned


def _clip01(v: float) -> float:
    if v <= 0.0:
        return 0.0
    if v >= 1.0:
        return 1.0
    return float(v)


def _clean_priority_offset(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        raise ValueError("priority_offset must be a number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("priority_offset must be finite")
    if out < -1.0 or out > 1.0:
        raise ValueError("priority_offset must be within [-1, 1]")
    return out


def _clean_snooze_until(value: Any) -> float | None:
    if value in (None, "", 0):
        return None
    if isinstance(value, bool):
        raise ValueError("snooze_until must be a unix timestamp or null")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("snooze_until must be finite")
    if out <= 0:
        return None
    return out


def _clean_dependency_session_id(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("dependency_session_id must be a string or null")
    out = value.strip()
    return out or None


def _priority_from_elapsed_seconds(elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 1.0
    return _clip01(math.exp(-SIDEBAR_PRIORITY_LAMBDA * float(elapsed_s)))


def _current_git_branch(cwd: Path) -> str | None:
    try:
        branch = _run_git(cwd, ["rev-parse", "--abbrev-ref", "HEAD"], timeout_s=GIT_DIFF_TIMEOUT_SECONDS, max_bytes=64 * 1024).strip()
    except (RuntimeError, FileNotFoundError):
        return None
    if not branch:
        return None
    return branch


def _resolve_client_file_path(*, session_id: str, raw_path: str) -> Path:
    path_obj = Path(raw_path).expanduser()
    if not path_obj.is_absolute():
        if session_id:
            MANAGER.refresh_session_meta(session_id)
            s = MANAGER.get_session(session_id)
            if s:
                base = Path(s.cwd).expanduser()
                if not base.is_absolute():
                    base = base.resolve()
                direct = (base / path_obj).resolve()
                if direct.exists():
                    path_obj = direct
                else:
                    tracked = _resolve_tracked_file_by_basename(session_id, raw_path)
                    if tracked is not None:
                        path_obj = tracked
                        return path_obj
                    try:
                        repo_root = Path(
                            _run_git(base, ["rev-parse", "--show-toplevel"], timeout_s=GIT_DIFF_TIMEOUT_SECONDS, max_bytes=64 * 1024).strip()
                        ).resolve()
                    except RuntimeError:
                        repo_root = base.resolve()
                    path_obj = _resolve_unique_bare_filename(repo_root, raw_path) or direct
            else:
                path_obj = (Path.cwd() / path_obj).resolve()
        else:
            path_obj = (Path.cwd() / path_obj).resolve()
    else:
        path_obj = path_obj.resolve()
    return path_obj


def _inspect_openable_file(path_obj: Path) -> tuple[bytes, int, str, str | None]:
    if not path_obj.exists():
        raise FileNotFoundError("file not found")
    if not path_obj.is_file():
        raise ValueError("path is not a file")
    try:
        raw = path_obj.read_bytes()
    except PermissionError as e:
        raise PermissionError("permission denied") from e
    size = len(raw)
    if size > FILE_READ_MAX_BYTES:
        raise ValueError(f"file too large (max {FILE_READ_MAX_BYTES} bytes)")
    kind, image_ctype = _file_kind(path_obj, raw)
    if kind != "image" and b"\x00" in raw:
        raise ValueError("binary file not supported")
    return raw, size, kind, image_ctype


def _read_downloadable_file(path_obj: Path) -> tuple[bytes, int]:
    if not path_obj.exists():
        raise FileNotFoundError("file not found")
    if not path_obj.is_file():
        raise ValueError("path is not a file")
    try:
        raw = path_obj.read_bytes()
    except PermissionError as e:
        raise PermissionError("permission denied") from e
    return raw, len(raw)


def _inspect_client_path(path_obj: Path) -> tuple[int, str, str | None]:
    if not path_obj.exists():
        raise FileNotFoundError("file not found")
    if path_obj.is_dir():
        return 0, "directory", None
    _raw, size, kind, image_ctype = _inspect_openable_file(path_obj)
    return size, kind, image_ctype


def _download_disposition(path_obj: Path) -> str:
    return f"attachment; filename*=UTF-8''{urllib.parse.quote(path_obj.name, safe='')}"


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


def _discover_log_for_session_id(session_id: str) -> Path | None:
    return _find_session_log_for_session_id(session_id)

def _session_id_from_rollout_path(log_path: Path) -> str | None:
    name = log_path.name
    m = _SESSION_ID_RE.findall(name)
    return m[-1] if m else None


def _read_session_meta(log_path: Path) -> dict[str, Any]:
    with log_path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    if not first:
        raise ValueError(f"missing session_meta in {log_path}")
    obj = json.loads(first)
    if not isinstance(obj, dict) or obj.get("type") != "session_meta":
        raise ValueError(f"invalid session_meta record in {log_path}")
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid session_meta payload in {log_path}")
    return payload


def _resume_candidate_from_log(log_path: Path) -> dict[str, Any] | None:
    meta = _read_session_meta(log_path)
    if _is_subagent_session_meta(meta):
        return None
    session_id = meta.get("id")
    cwd = meta.get("cwd")
    if not isinstance(session_id, str) or not session_id:
        return None
    if not isinstance(cwd, str) or not cwd:
        return None
    try:
        stat = log_path.stat()
        updated_ts = float(stat.st_mtime)
    except FileNotFoundError:
        return None
    except Exception:
        updated_ts = 0.0
    git_info = meta.get("git")
    git_branch = ""
    if isinstance(git_info, dict):
        branch_raw = git_info.get("branch")
        if isinstance(branch_raw, str):
            git_branch = branch_raw
    return {
        "session_id": session_id,
        "cwd": cwd,
        "log_path": str(log_path),
        "updated_ts": updated_ts,
        "timestamp": meta.get("timestamp"),
        "git_branch": git_branch,
    }


def _list_resume_candidates_for_cwd(cwd: str, *, limit: int = 12) -> list[dict[str, Any]]:
    cwd2 = str(Path(cwd).expanduser().resolve())
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for log_path in _iter_session_logs():
        try:
            row = _resume_candidate_from_log(log_path)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        session_id = row.get("session_id")
        row_cwd = row.get("cwd")
        if not (isinstance(session_id, str) and session_id):
            continue
        if not (isinstance(row_cwd, str) and row_cwd == cwd2):
            continue
        if session_id in seen:
            continue
        out.append(row)
        seen.add(session_id)
        if len(out) >= limit:
            break
    return out


def _resume_preview_from_text(text: str, *, max_chars: int = 120) -> str:
    lines = [line.strip() for line in text.splitlines()]
    compact = " ".join(line for line in lines if line)
    compact = re.sub(r"\s+", " ", compact).strip()
    if len(compact) <= max_chars:
        return compact
    head = compact[: max_chars - 1].rstrip()
    cut = head.rfind(" ")
    if cut >= max_chars * 0.6:
        head = head[:cut].rstrip()
    return head + "..."


def _user_message_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type not in ("input_text", "output_text", "text"):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return "\n".join(parts).strip()


def _is_scaffold_user_text(text: str) -> bool:
    s = text.strip()
    return s.startswith("# AGENTS.md instructions") or s.startswith("<environment_context>")


def _first_user_message_preview_from_log(log_path: Path, *, max_scan_bytes: int = 256 * 1024) -> str:
    try:
        with log_path.open("rb") as f:
            total = 0
            for raw in f:
                total += len(raw)
                if total > max_scan_bytes:
                    break
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(obj, dict) or obj.get("type") != "response_item":
                    continue
                payload = obj.get("payload")
                if not isinstance(payload, dict):
                    continue
                if payload.get("type") != "message" or payload.get("role") != "user":
                    continue
                text = _user_message_text(payload)
                if not text or _is_scaffold_user_text(text):
                    continue
                return _resume_preview_from_text(text)
    except FileNotFoundError:
        return ""
    return ""


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
    return _rollout_log._extract_chat_events(objs)


def _read_jsonl_tail(path: Path, max_bytes: int) -> list[dict[str, Any]]:
    return _rollout_log._read_jsonl_tail(path, max_bytes)


def _read_chat_events_from_tail(
    log_path: Path,
    min_events: int = 120,
    max_scan_bytes: int = 128 * 1024 * 1024,
) -> list[dict[str, Any]]:
    return _rollout_log._read_chat_events_from_tail(log_path, min_events=min_events, max_scan_bytes=max_scan_bytes)


def _read_chat_tail_snapshot(
    log_path: Path,
    *,
    min_events: int,
    initial_scan_bytes: int,
    max_scan_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int, bool, int]:
    return _rollout_log._read_chat_tail_snapshot(
        log_path,
        min_events=min_events,
        initial_scan_bytes=initial_scan_bytes,
        max_scan_bytes=max_scan_bytes,
    )


def _event_ts(obj: dict[str, Any]) -> float | None:
    return _rollout_log._event_ts(obj)


def _has_assistant_output_text(obj: dict[str, Any]) -> bool:
    return _rollout_log._has_assistant_output_text(obj)


def _analyze_log_chunk(
    objs: list[dict[str, Any]],
) -> tuple[int, int, int, float | None, dict[str, Any] | None, list[dict[str, Any]]]:
    return _rollout_log._analyze_log_chunk(objs)


def _last_conversation_ts_from_tail(
    log_path: Path,
    *,
    max_scan_bytes: int,
) -> float | None:
    return _rollout_log._last_conversation_ts_from_tail(log_path, max_scan_bytes=max_scan_bytes)


def _compute_idle_from_log(path: Path, max_scan_bytes: int = 8 * 1024 * 1024) -> bool | None:
    return _rollout_log._compute_idle_from_log(path, max_scan_bytes=max_scan_bytes)


def _last_chat_role_ts_from_tail(
    path: Path,
    *,
    max_scan_bytes: int,
) -> tuple[str, float] | None:
    return _rollout_log._last_chat_role_ts_from_tail(path, max_scan_bytes=max_scan_bytes)


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
    queue_idle_since: float | None = None


class SessionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._stop = threading.Event()
        self._last_discover_ts = 0.0
        self._harness: dict[str, dict[str, Any]] = {}
        self._aliases: dict[str, str] = {}
        self._sidebar_meta: dict[str, dict[str, Any]] = {}
        self._hidden_sessions: set[str] = set()
        self._files: dict[str, list[str]] = {}
        self._queues: dict[str, list[str]] = {}
        self._harness_last_injected: dict[str, float] = {}
        self._harness_last_injected_scope: dict[str, float] = {}
        self._load_harness()
        self._load_aliases()
        self._load_sidebar_meta()
        self._load_hidden_sessions()
        self._load_files()
        self._load_queues()
        self._discover_existing(force=True)
        self._harness_thr = threading.Thread(target=self._harness_loop, name="harness", daemon=True)
        self._harness_thr.start()
        self._queue_thr = threading.Thread(target=self._queue_loop, name="queue", daemon=True)
        self._queue_thr.start()

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
        s.queue_idle_since = None

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
        except FileNotFoundError:
            return
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("invalid harness.json (expected object)")
        cleaned: dict[str, dict[str, Any]] = {}
        for sid, v in obj.items():
            if not isinstance(sid, str) or not sid:
                continue
            if not isinstance(v, dict):
                continue
            enabled = bool(v.get("enabled")) if "enabled" in v else False
            if "text" in v:
                raise ValueError(f"invalid harness config for session {sid!r} (use 'request', not 'text')")
            request = v.get("request")
            if request is None:
                request = ""
            if not isinstance(request, str):
                raise ValueError(f"invalid harness request for session {sid!r}")
            cleaned[sid] = {"enabled": enabled, "request": request}
        with self._lock:
            self._harness = cleaned

    def _save_harness(self) -> None:
        with self._lock:
            obj = dict(self._harness)
        os.makedirs(APP_DIR, exist_ok=True)
        tmp = HARNESS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, HARNESS_PATH)

    def _load_aliases(self) -> None:
        try:
            raw = ALIAS_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("invalid session_aliases.json (expected object)")
        cleaned: dict[str, str] = {}
        for sid, v in obj.items():
            if not isinstance(sid, str) or not sid:
                continue
            if not isinstance(v, str):
                continue
            alias = _clean_alias(v)
            if alias:
                cleaned[sid] = alias
        with self._lock:
            self._aliases = cleaned

    def _save_aliases(self) -> None:
        with self._lock:
            obj = dict(self._aliases)
        os.makedirs(APP_DIR, exist_ok=True)
        tmp = ALIAS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, ALIAS_PATH)

    def _load_sidebar_meta(self) -> None:
        try:
            raw = SIDEBAR_META_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("invalid session_sidebar.json (expected object)")
        cleaned: dict[str, dict[str, Any]] = {}
        for sid, value in obj.items():
            if not isinstance(sid, str) or not sid:
                continue
            if not isinstance(value, dict):
                continue
            offset = _clean_priority_offset(value.get("priority_offset"))
            snooze_until = _clean_snooze_until(value.get("snooze_until"))
            dependency_session_id = _clean_dependency_session_id(value.get("dependency_session_id"))
            entry: dict[str, Any] = {"priority_offset": offset}
            if snooze_until is not None:
                entry["snooze_until"] = snooze_until
            if dependency_session_id is not None:
                entry["dependency_session_id"] = dependency_session_id
            cleaned[sid] = entry
        with self._lock:
            self._sidebar_meta = cleaned

    def _save_sidebar_meta(self) -> None:
        with self._lock:
            obj = dict(self._sidebar_meta)
        os.makedirs(APP_DIR, exist_ok=True)
        tmp = SIDEBAR_META_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, SIDEBAR_META_PATH)

    def _load_hidden_sessions(self) -> None:
        try:
            raw = HIDDEN_SESSIONS_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        obj = json.loads(raw)
        if not isinstance(obj, list):
            raise ValueError("invalid hidden_sessions.json (expected list)")
        cleaned = {sid.strip() for sid in obj if isinstance(sid, str) and sid.strip()}
        with self._lock:
            self._hidden_sessions = cleaned

    def _save_hidden_sessions(self) -> None:
        with self._lock:
            obj = sorted(getattr(self, "_hidden_sessions", set()))
        os.makedirs(APP_DIR, exist_ok=True)
        tmp = HIDDEN_SESSIONS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, HIDDEN_SESSIONS_PATH)

    def _hide_session(self, session_id: str) -> None:
        with self._lock:
            hidden = getattr(self, "_hidden_sessions", None)
            if not isinstance(hidden, set):
                self._hidden_sessions = set()
                hidden = self._hidden_sessions
            hidden.add(session_id)
        self._save_hidden_sessions()

    def _unhide_session(self, session_id: str) -> None:
        changed = False
        with self._lock:
            hidden = getattr(self, "_hidden_sessions", None)
            if isinstance(hidden, set) and session_id in hidden:
                hidden.remove(session_id)
                changed = True
        if changed:
            self._save_hidden_sessions()

    def alias_set(self, session_id: str, name: str) -> str:
        alias = _clean_alias(name)
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("unknown session")
            if alias:
                self._aliases[session_id] = alias
            else:
                self._aliases.pop(session_id, None)
        self._save_aliases()
        return alias

    def alias_get(self, session_id: str) -> str:
        with self._lock:
            alias = self._aliases.get(session_id)
        return alias if isinstance(alias, str) else ""

    def alias_clear(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._aliases:
                return
            self._aliases.pop(session_id, None)
        self._save_aliases()

    def sidebar_meta_get(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("unknown session")
            meta_map = getattr(self, "_sidebar_meta", None)
            entry = meta_map.get(session_id) if isinstance(meta_map, dict) else None
        if not isinstance(entry, dict):
            return {"priority_offset": 0.0, "snooze_until": None, "dependency_session_id": None}
        return {
            "priority_offset": _clean_priority_offset(entry.get("priority_offset")),
            "snooze_until": _clean_snooze_until(entry.get("snooze_until")),
            "dependency_session_id": _clean_dependency_session_id(entry.get("dependency_session_id")),
        }

    def sidebar_meta_set(
        self,
        session_id: str,
        *,
        priority_offset: Any,
        snooze_until: Any,
        dependency_session_id: Any,
    ) -> dict[str, Any]:
        offset = _clean_priority_offset(priority_offset)
        snooze_until_clean = _clean_snooze_until(snooze_until)
        dependency_clean = _clean_dependency_session_id(dependency_session_id)
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("unknown session")
            if dependency_clean == session_id:
                raise ValueError("session cannot depend on itself")
            if dependency_clean is not None and dependency_clean not in self._sessions:
                raise ValueError("dependency session not found")
            entry = {"priority_offset": offset}
            if snooze_until_clean is not None:
                entry["snooze_until"] = snooze_until_clean
            if dependency_clean is not None:
                entry["dependency_session_id"] = dependency_clean
            meta_map = getattr(self, "_sidebar_meta", None)
            if not isinstance(meta_map, dict):
                self._sidebar_meta = {}
                meta_map = self._sidebar_meta
            meta_map[session_id] = entry
        self._save_sidebar_meta()
        return {"priority_offset": offset, "snooze_until": snooze_until_clean, "dependency_session_id": dependency_clean}

    def edit_session(
        self,
        session_id: str,
        *,
        name: str,
        priority_offset: Any,
        snooze_until: Any,
        dependency_session_id: Any,
    ) -> tuple[str, dict[str, Any]]:
        alias = _clean_alias(name)
        offset = _clean_priority_offset(priority_offset)
        snooze_until_clean = _clean_snooze_until(snooze_until)
        dependency_clean = _clean_dependency_session_id(dependency_session_id)
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("unknown session")
            if dependency_clean == session_id:
                raise ValueError("session cannot depend on itself")
            if dependency_clean is not None and dependency_clean not in self._sessions:
                raise ValueError("dependency session not found")
            aliases = getattr(self, "_aliases", None)
            if not isinstance(aliases, dict):
                self._aliases = {}
                aliases = self._aliases
            if alias:
                aliases[session_id] = alias
            else:
                aliases.pop(session_id, None)
            meta_map = getattr(self, "_sidebar_meta", None)
            if not isinstance(meta_map, dict):
                self._sidebar_meta = {}
                meta_map = self._sidebar_meta
            entry = {"priority_offset": offset}
            if snooze_until_clean is not None:
                entry["snooze_until"] = snooze_until_clean
            if dependency_clean is not None:
                entry["dependency_session_id"] = dependency_clean
            meta_map[session_id] = entry
        self._save_aliases()
        self._save_sidebar_meta()
        return alias, {"priority_offset": offset, "snooze_until": snooze_until_clean, "dependency_session_id": dependency_clean}

    def _clear_deleted_session_state(self, session_id: str) -> None:
        changed_sidebar = False
        changed_harness = False
        changed_files = False
        changed_queues = False
        with self._lock:
            aliases = getattr(self, "_aliases", None)
            if isinstance(aliases, dict):
                aliases.pop(session_id, None)
            meta_map = getattr(self, "_sidebar_meta", None)
            if isinstance(meta_map, dict) and session_id in meta_map:
                meta_map.pop(session_id, None)
                changed_sidebar = True
            if isinstance(meta_map, dict):
                for entry in meta_map.values():
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("dependency_session_id") != session_id:
                        continue
                    entry.pop("dependency_session_id", None)
                    changed_sidebar = True
            harness = getattr(self, "_harness", None)
            if isinstance(harness, dict) and session_id in harness:
                harness.pop(session_id, None)
                changed_harness = True
            files = getattr(self, "_files", None)
            if isinstance(files, dict):
                for key in [f"sid:{session_id}", session_id]:
                    if key in files:
                        files.pop(key, None)
                        changed_files = True
            queues = getattr(self, "_queues", None)
            if isinstance(queues, dict) and session_id in queues:
                queues.pop(session_id, None)
                changed_queues = True
        self._save_aliases()
        if changed_sidebar:
            self._save_sidebar_meta()
        if changed_harness:
            self._save_harness()
        if changed_files:
            self._save_files()
        if changed_queues:
            self._save_queues()

    def _load_files(self) -> None:
        try:
            raw = FILE_HISTORY_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("invalid session_files.json (expected object)")
        cleaned: dict[str, list[str]] = {}
        for sid, arr in obj.items():
            if not isinstance(sid, str) or not sid:
                continue
            key = sid if (sid.startswith("cwd:") or sid.startswith("sid:")) else f"sid:{sid}"
            if not isinstance(arr, list):
                continue
            out: list[str] = []
            for v in arr:
                if not isinstance(v, str):
                    continue
                p = v.strip()
                if not p or p in out:
                    continue
                out.append(p)
                if len(out) >= FILE_HISTORY_MAX:
                    break
            if out:
                cleaned[key] = out
        with self._lock:
            self._files = cleaned

    def _save_files(self) -> None:
        with self._lock:
            obj = dict(self._files)
        os.makedirs(APP_DIR, exist_ok=True)
        tmp = FILE_HISTORY_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, FILE_HISTORY_PATH)

    def _load_queues(self) -> None:
        try:
            raw = QUEUE_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("invalid session_queues.json (expected object)")
        cleaned: dict[str, list[str]] = {}
        for sid, arr in obj.items():
            if not isinstance(sid, str) or not sid:
                continue
            if not isinstance(arr, list):
                continue
            out: list[str] = []
            for v in arr:
                if not isinstance(v, str):
                    continue
                t = v.strip()
                if not t:
                    continue
                out.append(v)
            if out:
                cleaned[sid] = out
        with self._lock:
            self._queues = cleaned

    def _save_queues(self) -> None:
        with self._lock:
            obj = dict(self._queues)
        os.makedirs(APP_DIR, exist_ok=True)
        tmp = QUEUE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, QUEUE_PATH)

    def _compat_queue_len(self, session_id: str) -> int:
        with self._lock:
            qmap = getattr(self, "_queues", None)
            if not isinstance(qmap, dict):
                return 0
            q = qmap.get(session_id)
            return int(len(q)) if isinstance(q, list) else 0

    def _compat_queue_list(self, session_id: str) -> list[str]:
        with self._lock:
            qmap = getattr(self, "_queues", None)
            if not isinstance(qmap, dict):
                return []
            q = qmap.get(session_id)
            if not isinstance(q, list) or not q:
                return []
            return list(q)

    def _compat_queue_enqueue(self, session_id: str, text: str) -> dict[str, Any]:
        t = str(text)
        if not t.strip():
            raise ValueError("text required")
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("unknown session")
            q = self._queues.get(session_id)
            if not isinstance(q, list):
                q = []
                self._queues[session_id] = q
            q.append(t)
            ql = len(q)
        self._save_queues()
        return {"queued": True, "queue_len": int(ql), "backend": "server_compat"}

    def _compat_queue_delete(self, session_id: str, index: int) -> dict[str, Any]:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("unknown session")
            q = self._queues.get(session_id)
            if not isinstance(q, list):
                q = []
                self._queues[session_id] = q
            if index < 0 or index >= len(q):
                raise ValueError("index out of range")
            q.pop(int(index))
            ql = len(q)
        self._save_queues()
        return {"ok": True, "queue_len": int(ql), "backend": "server_compat"}

    def _compat_queue_update(self, session_id: str, index: int, text: str) -> dict[str, Any]:
        t = str(text)
        if not t.strip():
            raise ValueError("text required")
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("unknown session")
            q = self._queues.get(session_id)
            if not isinstance(q, list):
                q = []
                self._queues[session_id] = q
            if index < 0 or index >= len(q):
                raise ValueError("index out of range")
            q[int(index)] = t
            ql = len(q)
        self._save_queues()
        return {"ok": True, "queue_len": int(ql), "backend": "server_compat"}

    def _files_key_for_session(self, session_id: str) -> tuple[str, list[str], "Session"]:
        s = self._sessions.get(session_id)
        if not s:
            raise KeyError("unknown session")
        cwd_key: str | None = None
        cwd_raw = s.cwd if isinstance(s.cwd, str) else ""
        if cwd_raw and cwd_raw != "?":
            try:
                cwd_norm = str(Path(cwd_raw).expanduser().resolve())
            except Exception:
                cwd_norm = cwd_raw.strip()
            if cwd_norm:
                cwd_key = f"cwd:{cwd_norm}"
        sid_key = f"sid:{session_id}"
        if cwd_key:
            legacy = [sid_key, session_id]
            return cwd_key, legacy, s
        return sid_key, [session_id], s

    def files_get(self, session_id: str) -> list[str]:
        dirty = False
        out: list[str] = []
        with self._lock:
            key, legacy_keys, _s = self._files_key_for_session(session_id)
            arr = self._files.get(key)
            if isinstance(arr, list) and arr:
                out = list(arr)
            else:
                for lk in legacy_keys:
                    arr2 = self._files.get(lk)
                    if isinstance(arr2, list) and arr2:
                        out = list(arr2)
                        if lk != key:
                            self._files[key] = list(arr2)
                            self._files.pop(lk, None)
                            dirty = True
                        break
        if dirty:
            self._save_files()
        return list(out)

    def files_add(self, session_id: str, path: str) -> list[str]:
        p = str(path).strip()
        if not p:
            return self.files_get(session_id)
        dirty = False
        with self._lock:
            key, legacy_keys, _s = self._files_key_for_session(session_id)
            cur = list(self._files.get(key, []))
            if not cur:
                for lk in legacy_keys:
                    legacy = self._files.get(lk)
                    if isinstance(legacy, list) and legacy:
                        cur = list(legacy)
                        if lk != key:
                            self._files.pop(lk, None)
                            dirty = True
                        break
            cur = [x for x in cur if x != p]
            cur.insert(0, p)
            if len(cur) > FILE_HISTORY_MAX:
                cur = cur[:FILE_HISTORY_MAX]
            self._files[key] = cur
        self._save_files()
        return list(cur)

    def files_clear(self, session_id: str) -> None:
        dirty = False
        with self._lock:
            key, legacy_keys, s = self._files_key_for_session(session_id)
            for lk in legacy_keys:
                if lk in self._files:
                    self._files.pop(lk, None)
                    dirty = True
            if key.startswith("cwd:"):
                keep = False
                for other in self._sessions.values():
                    if other.session_id == s.session_id:
                        continue
                    try:
                        other_key, _legacy, _s2 = self._files_key_for_session(other.session_id)
                    except KeyError:
                        continue
                    if other_key == key:
                        keep = True
                        break
                if not keep and key in self._files:
                    self._files.pop(key, None)
                    dirty = True
            else:
                if key in self._files:
                    self._files.pop(key, None)
                    dirty = True
        if dirty:
            self._save_files()

    def harness_get(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            cfg0 = self._harness.get(session_id)
            cfg = dict(cfg0) if isinstance(cfg0, dict) else {}
        enabled = bool(cfg.get("enabled"))
        request = cfg.get("request")
        if not isinstance(request, str):
            request = ""
        return {"enabled": enabled, "request": request}

    def harness_set(self, session_id: str, *, enabled: bool | None = None, request: str | None = None) -> dict[str, Any]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            cur0 = self._harness.get(session_id)
            cur = dict(cur0) if isinstance(cur0, dict) else {}
            if enabled is not None:
                cur["enabled"] = bool(enabled)
            if request is not None:
                cur["request"] = str(request)
            self._harness[session_id] = cur
            if enabled is not None and bool(enabled) is False:
                self._harness_last_injected.pop(session_id, None)
        self._save_harness()
        return self.harness_get(session_id)

    def _harness_loop(self) -> None:
        # Persist across browser disconnects: server is the scheduler.
        while not self._stop.is_set():
            self._harness_sweep()
            self._stop.wait(HARNESS_SWEEP_SECONDS)

    def _harness_sweep(self) -> None:
        now = time.time()
        # Keep discovery fresh; sessions can appear/disappear without UI polling.
        self._discover_existing_if_stale()
        self._prune_dead_sessions()
        with self._lock:
            items: list[tuple[str, Session, dict[str, Any], float]] = []
            for sid, s in self._sessions.items():
                cfg0 = self._harness.get(sid)
                cfg = dict(cfg0) if isinstance(cfg0, dict) else {}
                last_inj = float(self._harness_last_injected.get(sid, 0.0))
                items.append((sid, s, cfg, last_inj))

        for sid, s, cfg, last_inj in items:
            if not bool(cfg.get("enabled")):
                continue
            request = cfg.get("request")
            if not isinstance(request, str):
                request = ""
            prompt = _render_harness_prompt(request)
            lp = s.log_path
            if lp is None or (not lp.exists()):
                continue
            scope_key = f"thread:{s.thread_id}" if s.thread_id else f"log:{str(lp)}"
            with self._lock:
                scope_last = float(self._harness_last_injected_scope.get(scope_key, 0.0))
            if (last_inj and (now - last_inj) < float(HARNESS_IDLE_SECONDS)) or (scope_last and (now - scope_last) < float(HARNESS_IDLE_SECONDS)):
                continue
            st = self.get_state(sid)
            if not isinstance(st, dict):
                raise ValueError("invalid broker state response")
            if "busy" not in st or "queue_len" not in st:
                raise ValueError("invalid broker state response")
            busy = bool(st.get("busy"))
            ql = int(st.get("queue_len"))
            if busy or ql > 0 or self._compat_queue_len(sid) > 0:
                continue
            last = _last_chat_role_ts_from_tail(lp, max_scan_bytes=HARNESS_MAX_SCAN_BYTES)
            if not last:
                continue
            role, ts = last
            if role != "assistant":
                continue
            if (now - float(ts)) < float(HARNESS_IDLE_SECONDS):
                continue
            with self._lock:
                scope_last = float(self._harness_last_injected_scope.get(scope_key, 0.0))
            if scope_last and (now - scope_last) < float(HARNESS_IDLE_SECONDS):
                continue
            self.send(sid, prompt)
            with self._lock:
                self._harness_last_injected[sid] = now
                self._harness_last_injected_scope[scope_key] = now

    def _queue_loop(self) -> None:
        while not self._stop.is_set():
            self._queue_sweep()
            self._stop.wait(QUEUE_SWEEP_SECONDS)

    def _queue_sweep(self) -> None:
        self._discover_existing_if_stale()
        self._prune_dead_sessions()
        with self._lock:
            # Drop queues for sessions that no longer exist.
            drop = [sid for sid in self._queues.keys() if sid not in self._sessions]
            for sid in drop:
                self._queues.pop(sid, None)
            items = [(sid, q[0]) for sid, q in self._queues.items() if isinstance(q, list) and q]
            log_paths = {sid: (self._sessions.get(sid).log_path if sid in self._sessions else None) for sid, _ in items}
        if drop:
            self._save_queues()
        # At most one injection per sweep.
        for sid, text in items:
            now_ts = time.time()
            try:
                st = self.get_state(sid)
            except Exception:
                with self._lock:
                    s0 = self._sessions.get(sid)
                    if s0:
                        s0.queue_idle_since = None
                continue
            if not isinstance(st, dict):
                with self._lock:
                    s0 = self._sessions.get(sid)
                    if s0:
                        s0.queue_idle_since = None
                continue
            if "busy" not in st or "queue_len" not in st:
                with self._lock:
                    s0 = self._sessions.get(sid)
                    if s0:
                        s0.queue_idle_since = None
                continue
            busy = bool(st.get("busy"))
            ql = int(st.get("queue_len"))
            if busy or ql > 0:
                with self._lock:
                    s0 = self._sessions.get(sid)
                    if s0:
                        s0.queue_idle_since = None
                continue
            # Guardrail: only inject server-queued messages when the rollout log indicates the
            # session is idle. This avoids injecting "next" prompts mid-turn when the broker's
            # busy bit is momentarily false due to a quiet window.
            lp = log_paths.get(sid)
            try:
                if isinstance(lp, Path) and lp.exists():
                    if not self.idle_from_log(sid):
                        with self._lock:
                            s0 = self._sessions.get(sid)
                            if s0:
                                s0.queue_idle_since = None
                        continue
            except Exception:
                with self._lock:
                    s0 = self._sessions.get(sid)
                    if s0:
                        s0.queue_idle_since = None
                continue
            with self._lock:
                s0 = self._sessions.get(sid)
                if not s0:
                    continue
                idle_since = s0.queue_idle_since
                if idle_since is None:
                    s0.queue_idle_since = now_ts
                    continue
                if (now_ts - idle_since) < QUEUE_IDLE_GRACE_SECONDS:
                    continue
            try:
                self.send(sid, text)
            except Exception:
                with self._lock:
                    s0 = self._sessions.get(sid)
                    if s0:
                        s0.queue_idle_since = None
                continue
            with self._lock:
                q = self._queues.get(sid)
                s0 = self._sessions.get(sid)
                if s0:
                    s0.queue_idle_since = None
                if isinstance(q, list) and q and q[0] == text:
                    q.pop(0)
                    if not q:
                        self._queues.pop(sid, None)
            self._save_queues()
            break

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
            if not meta_path.exists():
                raise RuntimeError(f"missing metadata sidecar for socket {sock}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError(f"invalid metadata json for socket {sock}")

            thread_id = meta.get("session_id") if isinstance(meta.get("session_id"), str) and meta.get("session_id") else session_id
            codex_pid_raw = meta.get("codex_pid")
            broker_pid_raw = meta.get("broker_pid")
            if not isinstance(codex_pid_raw, int):
                raise ValueError(f"invalid codex_pid in metadata for socket {sock}")
            if not isinstance(broker_pid_raw, int):
                raise ValueError(f"invalid broker_pid in metadata for socket {sock}")
            codex_pid = int(codex_pid_raw)
            broker_pid = int(broker_pid_raw)
            owned = (meta.get("owner") == "web") if isinstance(meta.get("owner"), str) else False

            log_path: Path | None = None
            if "log_path" not in meta:
                raise ValueError(f"missing log_path in metadata for socket {sock}")
            if meta.get("log_path") is None:
                log_path = None
            else:
                log_path_raw = meta.get("log_path")
                if not isinstance(log_path_raw, str) or (not log_path_raw.strip()):
                    raise ValueError(f"invalid log_path in metadata for socket {sock}")
                log_path = Path(log_path_raw)
            if log_path is not None and log_path.exists():
                thread_id, log_path = _coerce_main_thread_log(thread_id=thread_id, log_path=log_path)
            else:
                log_path = None

            if (log_path is None) and (not _pid_alive(codex_pid)) and (not _pid_alive(broker_pid)):
                self._unhide_session(session_id)
                _unlink_quiet(sock)
                _unlink_quiet(meta_path)
                continue
            with self._lock:
                hidden_sessions = set(getattr(self, "_hidden_sessions", set()))
            if session_id in hidden_sessions:
                if (not _pid_alive(codex_pid)) and (not _pid_alive(broker_pid)):
                    self._unhide_session(session_id)
                    _unlink_quiet(sock)
                    _unlink_quiet(meta_path)
                continue

            cwd_raw = meta.get("cwd")
            if not isinstance(cwd_raw, str) or (not cwd_raw.strip()):
                raise ValueError(f"invalid cwd in metadata for socket {sock}")
            cwd = cwd_raw

            start_ts_raw = meta.get("start_ts")
            if not isinstance(start_ts_raw, (int, float)):
                raise ValueError(f"invalid start_ts in metadata for socket {sock}")
            start_ts = float(start_ts_raw)

            # Validate socket is responsive.
            try:
                resp = self._sock_call(sock, {"cmd": "state"}, timeout_s=0.5)
            except Exception as e:
                # Socket discovery should not take down the sessions listing. Treat
                # definitely-stale sockets as runtime artifacts and prune them, but
                # avoid unlinking sockets for live processes (startup races).
                sys.stderr.write(f"error: discover: sock state call failed for {sock}: {type(e).__name__}: {e}\n")
                sys.stderr.flush()
                if _sock_error_definitely_stale(e) and (not _pid_alive(codex_pid)) and (not _pid_alive(broker_pid)):
                    _unlink_quiet(sock)
                    _unlink_quiet(meta_path)
                continue

            if log_path is not None:
                meta_log_off = int(log_path.stat().st_size)
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
                busy=bool(resp.get("busy")),
                queue_len=int(resp.get("queue_len")),
                token=(resp.get("token") if isinstance(resp.get("token"), (dict, type(None))) else None),
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
            if s2:
                if "busy" not in resp or "queue_len" not in resp:
                    raise ValueError("invalid broker state response")
                s2.busy = bool(resp.get("busy"))
                s2.queue_len = int(resp.get("queue_len"))
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
        for sid, sock in dead:
            self._clear_deleted_session_state(sid)
            _unlink_quiet(sock)
            _unlink_quiet(sock.with_suffix(".json"))

    def _update_meta_counters(self) -> None:
        with self._lock:
            items = list(self._sessions.items())
        for sid, s in items:
            lp = s.log_path
            if lp is None or (not lp.exists()):
                continue
            sz = int(lp.stat().st_size)
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
        files_dirty = False
        sidebar_dirty = False
        now_ts = time.time()
        with self._lock:
            items: list[dict[str, Any]] = []
            qmap = getattr(self, "_queues", None)
            meta_map = getattr(self, "_sidebar_meta", None)
            active_ids = set(self._sessions.keys())
            for s in self._sessions.values():
                cfg0 = self._harness.get(s.session_id)
                h_enabled = bool(cfg0.get("enabled")) if isinstance(cfg0, dict) else False
                alias = self._aliases.get(s.session_id)
                if not isinstance(alias, str):
                    alias = ""
                files: list[str] = []
                try:
                    key, legacy_keys, _sref = self._files_key_for_session(s.session_id)
                except KeyError:
                    key = ""
                    legacy_keys = []
                if key:
                    cur = self._files.get(key)
                    if isinstance(cur, list) and cur:
                        files = list(cur)
                    else:
                        for lk in legacy_keys:
                            legacy = self._files.get(lk)
                            if isinstance(legacy, list) and legacy:
                                files = list(legacy)
                                if lk != key:
                                    self._files[key] = list(legacy)
                                    self._files.pop(lk, None)
                                    files_dirty = True
                                break
                log_exists = bool(s.log_path is not None and s.log_path.exists())
                if s.last_chat_ts is None and log_exists and s.log_path is not None:
                    conv_ts = _last_conversation_ts_from_tail(s.log_path, max_scan_bytes=256 * 1024)
                    if isinstance(conv_ts, (int, float)):
                        s.last_chat_ts = float(conv_ts)
                updated_ts = float(s.last_chat_ts) if isinstance(s.last_chat_ts, (int, float)) else float(s.start_ts)
                compat_ql = 0
                if isinstance(qmap, dict):
                    q0 = qmap.get(s.session_id)
                    if isinstance(q0, list):
                        compat_ql = len(q0)
                meta0 = meta_map.get(s.session_id) if isinstance(meta_map, dict) else None
                if not isinstance(meta0, dict):
                    meta0 = {}
                priority_offset = _clean_priority_offset(meta0.get("priority_offset"))
                snooze_until = _clean_snooze_until(meta0.get("snooze_until"))
                dependency_session_id = _clean_dependency_session_id(meta0.get("dependency_session_id"))
                if dependency_session_id == s.session_id or (dependency_session_id is not None and dependency_session_id not in active_ids):
                    dependency_session_id = None
                    if isinstance(meta_map, dict) and isinstance(meta0, dict):
                        meta0.pop("dependency_session_id", None)
                        sidebar_dirty = True
                if snooze_until is not None and snooze_until <= now_ts:
                    snooze_until = None
                    if isinstance(meta_map, dict) and isinstance(meta0, dict):
                        meta0.pop("snooze_until", None)
                        sidebar_dirty = True
                elapsed_s = max(0.0, now_ts - updated_ts)
                time_priority = _priority_from_elapsed_seconds(elapsed_s)
                base_priority = _clip01(time_priority + priority_offset)
                blocked = dependency_session_id is not None
                snoozed = snooze_until is not None and snooze_until > now_ts
                final_priority = 0.0 if (snoozed or blocked) else base_priority
                cwd_path = Path(s.cwd).expanduser()
                if not cwd_path.is_absolute():
                    cwd_path = cwd_path.resolve()
                git_branch = _current_git_branch(cwd_path)
                items.append(
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
                        "log_exists": log_exists,
                        "state_busy": bool(s.busy),
                        "queue_len": int(s.queue_len) + int(compat_ql),
                        "token": s.token,
                        "thinking": int(s.meta_thinking),
                        "tools": int(s.meta_tools),
                        "system": int(s.meta_system),
                        "harness_enabled": h_enabled,
                        "alias": alias,
                        "files": list(files),
                        "git_branch": git_branch,
                        "priority_offset": priority_offset,
                        "snooze_until": snooze_until,
                        "dependency_session_id": dependency_session_id,
                        "time_priority": time_priority,
                        "base_priority": base_priority,
                        "final_priority": final_priority,
                        "blocked": blocked,
                        "snoozed": snoozed,
                    }
                )

        out: list[dict[str, Any]] = []
        for it in items:
            sid = str(it["session_id"])
            log_exists = bool(it.get("log_exists"))
            state_busy = bool(it.get("state_busy"))
            if not log_exists:
                busy_out = False
            else:
                # When a log exists, unify semantics with /messages:
                # busy if broker says busy OR log-derived idle is false.
                busy_out = state_busy or (not bool(self.idle_from_log(sid)))
            it2 = dict(it)
            it2.pop("log_exists", None)
            it2.pop("state_busy", None)
            it2["busy"] = bool(busy_out)
            out.append(it2)
        if files_dirty:
            self._save_files()
        if sidebar_dirty:
            self._save_sidebar_meta()
        out.sort(
            key=lambda item: (
                -float(item.get("final_priority", 0.0)),
                -float(item.get("updated_ts", item.get("start_ts", 0.0))),
                -float(item.get("start_ts", 0.0)),
                str(item.get("session_id", "")),
            )
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
        if not meta_path.exists():
            raise RuntimeError(f"missing metadata sidecar for socket {sock}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError(f"invalid metadata json for socket {sock}")

        thread_id = meta.get("session_id") if isinstance(meta.get("session_id"), str) and meta.get("session_id") else s.thread_id
        owned = (meta.get("owner") == "web") if isinstance(meta.get("owner"), str) else s.owned
        if "log_path" not in meta:
            raise ValueError(f"missing log_path in metadata for socket {sock}")
        log_path: Path | None
        if meta.get("log_path") is None:
            log_path = None
        else:
            log_path_raw = meta.get("log_path")
            if not isinstance(log_path_raw, str) or (not log_path_raw.strip()):
                raise ValueError(f"invalid log_path in metadata for socket {sock}")
            log_path = Path(log_path_raw)
        if log_path is not None and log_path.exists():
            thread_id, log_path = _coerce_main_thread_log(thread_id=thread_id, log_path=log_path)

        cwd_raw = meta.get("cwd")
        if not isinstance(cwd_raw, str) or (not cwd_raw.strip()):
            raise ValueError(f"invalid cwd in metadata for socket {sock}")
        cwd = cwd_raw

        with self._lock:
            s2 = self._sessions.get(session_id)
            if not s2:
                return
            s2.thread_id = thread_id
            s2.cwd = str(cwd)
            s2.owned = bool(owned)
            if s2.log_path != log_path:
                s2.log_path = log_path
                if log_path is not None:
                    log_off = int(log_path.stat().st_size)
                else:
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
        def event_key(ev: dict[str, Any]) -> tuple[str, int, str] | None:
            role = ev.get("role")
            if role not in ("user", "assistant"):
                return None
            text = ev.get("text")
            if not isinstance(text, str):
                return None
            ts = ev.get("ts")
            if not isinstance(ts, (int, float)):
                return None
            return role, int(round(float(ts) * 1000.0)), text

        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return
            # Deduplicate to avoid re-appending overlapping tail events across incremental scans.
            # Also guard against runtimes emitting the same assistant message twice with slightly
            # different timestamps by deduping assistant text within the current assistant stretch
            # (resetting when we see a user message).
            tail = list(events[-CHAT_INDEX_MAX_EVENTS:])
            uniq_rev: list[dict[str, Any]] = []
            seen_exact: set[tuple[str, int, str]] = set()
            seen_assistant_stretch: set[str] = set()
            for ev in reversed(tail):
                k = event_key(ev)
                if k is not None and k in seen_exact:
                    continue
                if k is not None:
                    seen_exact.add(k)
                role = ev.get("role")
                if role == "user":
                    seen_assistant_stretch.clear()
                elif role == "assistant":
                    text = ev.get("text")
                    if isinstance(text, str):
                        if text in seen_assistant_stretch:
                            continue
                        seen_assistant_stretch.add(text)
                uniq_rev.append(ev)
            s.chat_index_events = list(reversed(uniq_rev))
            s.chat_index_scan_bytes = int(scan_bytes)
            s.chat_index_scan_complete = bool(scan_complete) and (len(events) <= CHAT_INDEX_MAX_EVENTS)
            s.chat_index_log_off = int(log_off)
            if token_update is not None:
                s.token = token_update

    def _append_chat_events(self, session_id: str, new_events: list[dict[str, Any]], *, new_off: int, latest_token: dict[str, Any] | None) -> None:
        def event_key(ev: dict[str, Any]) -> tuple[str, int, str] | None:
            role = ev.get("role")
            if role not in ("user", "assistant"):
                return None
            text = ev.get("text")
            if not isinstance(text, str):
                return None
            ts = ev.get("ts")
            if not isinstance(ts, (int, float)):
                return None
            return role, int(round(float(ts) * 1000.0)), text

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
                merged = list(s.chat_index_events)
                recent = merged[-256:] if len(merged) > 256 else merged
                seen_exact: set[tuple[str, int, str]] = set()
                for ev in recent:
                    k = event_key(ev)
                    if k is not None:
                        seen_exact.add(k)
                # Build assistant stretch state from the end of the merged list.
                # If the current tail already has assistant messages (after the last user),
                # avoid appending the same assistant text again.
                assistant_stretch: set[str] = set()
                for ev in reversed(merged):
                    role = ev.get("role")
                    if role == "user":
                        break
                    if role == "assistant":
                        text = ev.get("text")
                        if isinstance(text, str):
                            assistant_stretch.add(text)
                appended: list[dict[str, Any]] = []
                for ev in new_events:
                    k = event_key(ev)
                    if k is not None and k in seen_exact:
                        continue
                    if k is not None:
                        seen_exact.add(k)
                    role = ev.get("role")
                    if role == "user":
                        assistant_stretch.clear()
                    elif role == "assistant":
                        text = ev.get("text")
                        if isinstance(text, str):
                            if text in assistant_stretch:
                                continue
                            assistant_stretch.add(text)
                    merged.append(ev)
                    appended.append(ev)
                if len(merged) > CHAT_INDEX_MAX_EVENTS:
                    merged = merged[-CHAT_INDEX_MAX_EVENTS:]
                    s.chat_index_scan_complete = False
                s.chat_index_events = merged
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

        sz = int(lp.stat().st_size)

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

        sz2 = int(lp3.stat().st_size)

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
        _th, _tools, _sys, last_ts, token_update, new_events = _analyze_log_chunk(objs)
        self._append_chat_events(session_id, new_events, new_off=new_off, latest_token=token_update)
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                if isinstance(last_ts, (int, float)):
                    tsf = float(last_ts)
                    s.last_chat_ts = tsf if s.last_chat_ts is None else max(s.last_chat_ts, tsf)
                s.idle_cache_log_off = -1

    def idle_from_log(self, session_id: str) -> bool:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            lp = s.log_path
            cached_off = int(s.idle_cache_log_off)
            cached_idle = s.idle_cache_value
        if lp is None or (not lp.exists()):
            raise FileNotFoundError(f"missing rollout log for session {session_id}")
        sz = int(lp.stat().st_size)
        if (sz >= 0) and (cached_off == sz) and isinstance(cached_idle, bool):
            return bool(cached_idle)
        idle = _compute_idle_from_log(lp)
        with self._lock:
            s2 = self._sessions.get(session_id)
            if s2:
                s2.idle_cache_log_off = sz
                s2.idle_cache_value = idle
        if idle is None:
            raise RuntimeError("unable to compute idle state from log")
        return bool(idle)

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
            s.close()

    def kill_session(self, session_id: str) -> bool:
        with self._lock:
            s = self._sessions.get(session_id)
        if not s:
            return False
        self._sock_call(s.sock_path, {"cmd": "shutdown"}, timeout_s=1.0)
        return True

    def spawn_web_session(
        self,
        *,
        cwd: str,
        args: list[str] | None = None,
        resume_session_id: str | None = None,
        worktree_branch: str | None = None,
    ) -> dict[str, Any]:
        cwd_path = _resolve_existing_dir(cwd, field_name="cwd")
        cwd3 = str(cwd_path)
        if resume_session_id is not None and worktree_branch is not None:
            raise ValueError("worktree_branch cannot be used when resuming a session")
        spawn_cwd = cwd_path
        if worktree_branch is not None:
            spawn_cwd = _create_git_worktree(cwd_path, worktree_branch)

        argv = [sys.executable, "-m", "codoxear.broker", "--cwd", str(spawn_cwd), "--"]
        codex_args = list(args or [])
        if resume_session_id is not None:
            resume_id = str(resume_session_id).strip()
            if not resume_id:
                raise ValueError("resume_session_id must be a non-empty string")
            found = False
            for row in _list_resume_candidates_for_cwd(cwd3, limit=1000):
                if row.get("session_id") == resume_id:
                    found = True
                    break
            if not found:
                raise ValueError(f"resume session not found for cwd: {resume_id}")
            argv.extend(["resume", resume_id])
        argv.extend(codex_args)

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

        _wait_or_raise(proc, label="broker", timeout_s=1.5)
        if proc.stderr is not None:
            threading.Thread(target=_drain_stream, args=(proc.stderr,), daemon=True).start()

        # Prevent zombies when the broker exits.
        threading.Thread(target=proc.wait, daemon=True).start()
        return {"broker_pid": int(proc.pid)}

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            s = self._sessions.get(session_id)
        if not s:
            return False
        ok = self.kill_session(session_id)
        if ok:
            self.files_clear(session_id)
            self._clear_deleted_session_state(session_id)
        return ok

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
                self._clear_deleted_session_state(session_id)
                _unlink_quiet(sock)
                _unlink_quiet(sock.with_suffix(".json"))
                raise KeyError("unknown session")
            raise
        with self._lock:
            s2 = self._sessions.get(session_id)
            if s2:
                if "busy" in resp:
                    s2.busy = bool(resp.get("busy"))
                if "queue_len" not in resp:
                    raise ValueError("invalid broker send response")
                s2.queue_len = int(resp.get("queue_len"))
        return resp

    def enqueue(self, session_id: str, text: str) -> dict[str, Any]:
        # The web queue is durability-oriented. Persist to the server-side queue so
        # queued messages survive broker restarts.
        resp = self._compat_queue_enqueue(session_id, text)
        with self._lock:
            s = self._sessions.get(session_id)
            broker_ql = int(s.queue_len) if s else 0
        resp["queue_len_total"] = broker_ql + int(resp.get("queue_len") or 0)
        return resp

    def queue_list(self, session_id: str) -> list[str]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            sock = s.sock_path
        resp = self._sock_call(sock, {"cmd": "queue_list"}, timeout_s=1.5)
        if isinstance(resp, dict) and resp.get("error") == "unknown cmd":
            return self._compat_queue_list(session_id)
        if isinstance(resp, dict) and isinstance(resp.get("error"), str) and resp.get("error"):
            raise ValueError(f"broker queue_list error: {resp.get('error')}")
        q = resp.get("queue")
        if not isinstance(q, list) or not all(isinstance(x, str) for x in q):
            raise ValueError("invalid broker queue_list response")
        compat = self._compat_queue_list(session_id)
        return list(compat) + list(q)

    def queue_delete(self, session_id: str, index: int) -> dict[str, Any]:
        compat = self._compat_queue_list(session_id)
        compat_len = len(compat)
        if compat_len and int(index) < compat_len:
            resp = self._compat_queue_delete(session_id, int(index))
            with self._lock:
                s2 = self._sessions.get(session_id)
                broker_ql = int(s2.queue_len) if s2 is not None else 0
            if isinstance(resp, dict) and "queue_len" in resp:
                resp["queue_len_total"] = int(resp.get("queue_len")) + int(broker_ql)
            return resp

        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            sock = s.sock_path
        broker_index = int(index) - int(compat_len)
        resp = self._sock_call(sock, {"cmd": "queue_delete", "index": int(broker_index)}, timeout_s=1.5)
        if isinstance(resp, dict) and resp.get("error") == "unknown cmd":
            return self._compat_queue_delete(session_id, int(index))
        if isinstance(resp, dict) and isinstance(resp.get("error"), str) and resp.get("error"):
            raise ValueError(f"broker queue_delete error: {resp.get('error')}")
        if "queue_len" not in resp:
            raise ValueError("invalid broker queue_delete response")
        resp["backend"] = "broker"
        resp["queue_len_total"] = int(resp.get("queue_len")) + int(compat_len)
        with self._lock:
            s2 = self._sessions.get(session_id)
            if s2 and "queue_len" in resp:
                s2.queue_len = int(resp.get("queue_len"))
        return resp

    def queue_update(self, session_id: str, index: int, text: str) -> dict[str, Any]:
        compat = self._compat_queue_list(session_id)
        compat_len = len(compat)
        if compat_len and int(index) < compat_len:
            resp = self._compat_queue_update(session_id, int(index), text)
            with self._lock:
                s2 = self._sessions.get(session_id)
                broker_ql = int(s2.queue_len) if s2 is not None else 0
            if isinstance(resp, dict) and "queue_len" in resp:
                resp["queue_len_total"] = int(resp.get("queue_len")) + int(broker_ql)
            return resp

        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            sock = s.sock_path
        broker_index = int(index) - int(compat_len)
        resp = self._sock_call(sock, {"cmd": "queue_update", "index": int(broker_index), "text": text}, timeout_s=1.5)
        if isinstance(resp, dict) and resp.get("error") == "unknown cmd":
            return self._compat_queue_update(session_id, int(index), text)
        if isinstance(resp, dict) and isinstance(resp.get("error"), str) and resp.get("error"):
            raise ValueError(f"broker queue_update error: {resp.get('error')}")
        if "queue_len" not in resp:
            raise ValueError("invalid broker queue_update response")
        resp["backend"] = "broker"
        resp["queue_len_total"] = int(resp.get("queue_len")) + int(compat_len)
        with self._lock:
            s2 = self._sessions.get(session_id)
            if s2 and "queue_len" in resp:
                s2.queue_len = int(resp.get("queue_len"))
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
                self._clear_deleted_session_state(session_id)
                _unlink_quiet(sock)
                _unlink_quiet(sock.with_suffix(".json"))
                raise KeyError("unknown session")
            raise
        with self._lock:
            s2 = self._sessions.get(session_id)
            if s2:
                if "busy" not in resp or "queue_len" not in resp:
                    raise ValueError("invalid broker state response")
                s2.busy = bool(resp.get("busy"))
                s2.queue_len = int(resp.get("queue_len"))
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
        if "tail" not in resp:
            raise ValueError("invalid broker tail response")
        tail = resp.get("tail")
        if not isinstance(tail, str):
            raise ValueError("invalid broker tail response")
        return tail

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
            if URL_PREFIX:
                if path == URL_PREFIX:
                    loc = URL_PREFIX + "/"
                    if u.query:
                        loc = loc + "?" + u.query
                    self.send_response(308)
                    self.send_header("Location", loc)
                    self.end_headers()
                    return
                stripped = _strip_url_prefix(URL_PREFIX, path)
                if stripped is None:
                    self.send_error(404)
                    return
                path = stripped
            if path == "/favicon.ico":
                self._send_static("favicon.png")
                return
            if path == "/app.js":
                self._send_static("app.js")
                return
            if path == "/app.css":
                self._send_static("app.css")
                return
            if path == "/favicon.png":
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

            if path == "/api/session_resume_candidates":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                qs = urllib.parse.parse_qs(u.query)
                cwd_raw = qs.get("cwd", [""])[0]
                try:
                    cwd_path = _resolve_existing_dir(str(cwd_raw), field_name="cwd")
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                info = _describe_session_cwd(cwd_path)
                rows = _list_resume_candidates_for_cwd(info["cwd"])
                for row in rows:
                    sid = row.get("session_id")
                    log_path_raw = row.get("log_path")
                    alias = MANAGER.alias_get(sid) if isinstance(sid, str) and sid else ""
                    preview = ""
                    if isinstance(log_path_raw, str) and log_path_raw:
                        preview = _first_user_message_preview_from_log(Path(log_path_raw))
                    row["alias"] = alias
                    row["first_user_message"] = preview
                _json_response(self, 200, {"ok": True, **info, "sessions": rows})
                return

            if path == "/api/metrics":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                _json_response(self, 200, {"metrics": _metrics_snapshot()})
                return

            if path.startswith("/api/sessions/") and path.endswith("/diagnostics"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                if not session_id:
                    self.send_error(404)
                    return
                MANAGER.refresh_session_meta(session_id)
                s = MANAGER.get_session(session_id)
                if not s:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                state = MANAGER.get_state(session_id)
                if not isinstance(state, dict):
                    raise ValueError("invalid broker state response")
                if "busy" not in state:
                    raise ValueError("missing busy from broker state response")
                if "queue_len" not in state:
                    raise ValueError("missing queue_len from broker state response")
                token_val: dict[str, Any] | None = None
                st_token = state.get("token")
                if isinstance(st_token, dict) or st_token is None:
                    token_val = st_token
                meta = None
                if s.log_path is not None and s.log_path.exists():
                    meta = _read_session_meta(s.log_path)
                model_provider = meta.get("model_provider") if isinstance(meta, dict) else None
                if not isinstance(model_provider, str) or not model_provider.strip():
                    model_provider = None
                model = meta.get("model") if isinstance(meta, dict) else None
                if not isinstance(model, str) or not model.strip():
                    model = None
                reasoning_effort = meta.get("reasoning_effort") if isinstance(meta, dict) else None
                if not isinstance(reasoning_effort, str) or not reasoning_effort.strip():
                    reasoning_effort = None
                if (model is None or reasoning_effort is None) and s.log_path is not None and s.log_path.exists():
                    tc = _rollout_log._find_latest_turn_context(s.log_path, max_scan_bytes=8 * 1024 * 1024)
                    if isinstance(tc, dict):
                        if model is None:
                            m2 = tc.get("model")
                            if isinstance(m2, str) and m2.strip():
                                model = m2.strip()
                        if reasoning_effort is None:
                            eff = tc.get("reasoning_effort")
                            if not (isinstance(eff, str) and eff.strip()):
                                eff = tc.get("effort")
                            if isinstance(eff, str) and eff.strip():
                                reasoning_effort = eff.strip()
                sidebar_meta = MANAGER.sidebar_meta_get(session_id)
                cwd_path = Path(s.cwd).expanduser()
                if not cwd_path.is_absolute():
                    cwd_path = cwd_path.resolve()
                git_branch = _current_git_branch(cwd_path)
                updated_ts = float(s.last_chat_ts) if isinstance(s.last_chat_ts, (int, float)) else float(s.start_ts)
                elapsed_s = max(0.0, time.time() - updated_ts)
                time_priority = _priority_from_elapsed_seconds(elapsed_s)
                base_priority = _clip01(time_priority + float(sidebar_meta["priority_offset"]))
                blocked = sidebar_meta["dependency_session_id"] is not None
                snoozed = sidebar_meta["snooze_until"] is not None and float(sidebar_meta["snooze_until"]) > time.time()
                final_priority = 0.0 if (snoozed or blocked) else base_priority
                _json_response(
                    self,
                    200,
                    {
                        "session_id": s.session_id,
                        "thread_id": s.thread_id,
                        "owned": bool(s.owned),
                        "cwd": s.cwd,
                        "start_ts": float(s.start_ts),
                        "updated_ts": float(s.last_chat_ts) if isinstance(s.last_chat_ts, (int, float)) else float(s.start_ts),
                        "log_path": (str(s.log_path) if s.log_path is not None else None),
                        "broker_pid": int(s.broker_pid),
                        "codex_pid": int(s.codex_pid),
                        "busy": bool(state.get("busy")),
                        "queue_len": int(state.get("queue_len")) + MANAGER._compat_queue_len(session_id),
                        "token": token_val,
                        "model_provider": model_provider,
                        "model": model,
                        "reasoning_effort": reasoning_effort,
                        "git_branch": git_branch,
                        "time_priority": time_priority,
                        "base_priority": base_priority,
                        "final_priority": final_priority,
                        "priority_offset": sidebar_meta["priority_offset"],
                        "snooze_until": sidebar_meta["snooze_until"],
                        "dependency_session_id": sidebar_meta["dependency_session_id"],
                    },
                )
                return

            if path.startswith("/api/sessions/") and path.endswith("/queue"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                if not session_id:
                    self.send_error(404)
                    return
                try:
                    q = MANAGER.queue_list(session_id)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                except ValueError as e:
                    _json_response(self, 502, {"error": str(e)})
                    return
                _json_response(self, 200, {"ok": True, "queue": q})
                return

            if path.startswith("/api/sessions/") and path.endswith("/file/read"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                if not session_id:
                    self.send_error(404)
                    return
                MANAGER.refresh_session_meta(session_id)
                s = MANAGER.get_session(session_id)
                if not s:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                qs = urllib.parse.parse_qs(u.query)
                path_q = qs.get("path")
                if not path_q or not path_q[0]:
                    _json_response(self, 400, {"error": "path required"})
                    return
                rel = path_q[0]
                base = Path(s.cwd).expanduser()
                if not base.is_absolute():
                    base = base.resolve()
                p = _resolve_session_path(base, rel)
                if not p.exists():
                    _json_response(self, 404, {"error": "file not found"})
                    return
                if not p.is_file():
                    _json_response(self, 400, {"error": "path is not a file"})
                    return
                raw = p.read_bytes()
                size = len(raw)
                if size > FILE_READ_MAX_BYTES:
                    _json_response(self, 400, {"error": f"file too large (max {FILE_READ_MAX_BYTES} bytes)"})
                    return
                kind, image_ctype = _file_kind(p, raw)
                try:
                    MANAGER.files_add(session_id, str(p))
                except KeyError:
                    pass
                if kind == "image":
                    _json_response(
                        self,
                        200,
                        {
                            "ok": True,
                            "kind": "image",
                            "content_type": image_ctype,
                            "path": str(p),
                            "rel": str(rel),
                            "size": int(size),
                            "image_url": f"/api/sessions/{session_id}/file/blob?path={urllib.parse.quote(rel)}",
                        },
                    )
                    return
                if b"\x00" in raw:
                    _json_response(self, 400, {"error": "binary file not supported"})
                    return
                text = raw.decode("utf-8", errors="replace")
                _json_response(self, 200, {"ok": True, "kind": "text", "path": str(p), "rel": str(rel), "size": int(size), "text": text})
                return

            if path.startswith("/api/sessions/") and path.endswith("/file/blob"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                if not session_id:
                    self.send_error(404)
                    return
                MANAGER.refresh_session_meta(session_id)
                s = MANAGER.get_session(session_id)
                if not s:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                qs = urllib.parse.parse_qs(u.query)
                path_q = qs.get("path")
                if not path_q or not path_q[0]:
                    _json_response(self, 400, {"error": "path required"})
                    return
                rel = path_q[0]
                base = Path(s.cwd).expanduser()
                if not base.is_absolute():
                    base = base.resolve()
                p = _resolve_session_path(base, rel)
                if not p.exists():
                    _json_response(self, 404, {"error": "file not found"})
                    return
                if not p.is_file():
                    _json_response(self, 400, {"error": "path is not a file"})
                    return
                raw = p.read_bytes()
                _kind, ctype = _file_kind(p, raw)
                if ctype is None:
                    _json_response(self, 400, {"error": "file is not a supported image"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(raw)
                return

            if path.startswith("/api/sessions/") and path.endswith("/file/download"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                if not session_id:
                    self.send_error(404)
                    return
                MANAGER.refresh_session_meta(session_id)
                s = MANAGER.get_session(session_id)
                if not s:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                qs = urllib.parse.parse_qs(u.query)
                path_q = qs.get("path")
                if not path_q or not path_q[0]:
                    _json_response(self, 400, {"error": "path required"})
                    return
                rel = path_q[0]
                base = Path(s.cwd).expanduser()
                if not base.is_absolute():
                    base = base.resolve()
                p = _resolve_session_path(base, rel)
                try:
                    raw, size = _read_downloadable_file(p)
                except FileNotFoundError as e:
                    _json_response(self, 404, {"error": str(e)})
                    return
                except PermissionError as e:
                    _json_response(self, 403, {"error": str(e)})
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.send_header("Content-Disposition", _download_disposition(p))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(raw)
                return

            if path.startswith("/api/sessions/") and path.endswith("/git/changed_files"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                if not session_id:
                    self.send_error(404)
                    return
                MANAGER.refresh_session_meta(session_id)
                s = MANAGER.get_session(session_id)
                if not s:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                cwd = Path(s.cwd).expanduser()
                if not cwd.is_absolute():
                    cwd = cwd.resolve()
                try:
                    _require_git_repo(cwd)
                except RuntimeError as e:
                    _json_response(self, 409, {"error": str(e)})
                    return
                unstaged = _run_git(
                    cwd,
                    ["diff", "--name-only"],
                    timeout_s=GIT_DIFF_TIMEOUT_SECONDS,
                    max_bytes=64 * 1024,
                ).splitlines()
                staged = _run_git(
                    cwd,
                    ["diff", "--name-only", "--cached"],
                    timeout_s=GIT_DIFF_TIMEOUT_SECONDS,
                    max_bytes=64 * 1024,
                ).splitlines()
                unstaged_numstat = _run_git(
                    cwd,
                    ["diff", "--numstat"],
                    timeout_s=GIT_DIFF_TIMEOUT_SECONDS,
                    max_bytes=128 * 1024,
                )
                staged_numstat = _run_git(
                    cwd,
                    ["diff", "--numstat", "--cached"],
                    timeout_s=GIT_DIFF_TIMEOUT_SECONDS,
                    max_bytes=128 * 1024,
                )
                def _norm_list(xs: list[str]) -> list[str]:
                    out: list[str] = []
                    for x in xs:
                        t = x.strip()
                        if not t:
                            continue
                        out.append(t)
                        if len(out) >= GIT_CHANGED_FILES_MAX:
                            break
                    return out
                unstaged2 = _norm_list(unstaged)
                staged2 = _norm_list(staged)
                seen: set[str] = set()
                merged: list[str] = []
                for x in [*unstaged2, *staged2]:
                    if x in seen:
                        continue
                    seen.add(x)
                    merged.append(x)
                stats = _parse_git_numstat(unstaged_numstat)
                for path_key, vals in _parse_git_numstat(staged_numstat).items():
                    prev = stats.get(path_key)
                    if prev is None:
                        stats[path_key] = vals
                        continue
                    add_prev = prev.get("additions")
                    del_prev = prev.get("deletions")
                    add_new = vals.get("additions")
                    del_new = vals.get("deletions")
                    prev["additions"] = None if add_prev is None or add_new is None else int(add_prev) + int(add_new)
                    prev["deletions"] = None if del_prev is None or del_new is None else int(del_prev) + int(del_new)
                entries: list[dict[str, Any]] = []
                for path_key in merged:
                    vals = stats.get(path_key, {})
                    entries.append(
                        {
                            "path": path_key,
                            "additions": vals.get("additions"),
                            "deletions": vals.get("deletions"),
                            "changed": True,
                        }
                    )
                _json_response(
                    self,
                    200,
                    {"ok": True, "cwd": str(cwd), "files": merged, "entries": entries, "unstaged": unstaged2, "staged": staged2},
                )
                return

            if path.startswith("/api/sessions/") and path.endswith("/git/diff"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                if not session_id:
                    self.send_error(404)
                    return
                MANAGER.refresh_session_meta(session_id)
                s = MANAGER.get_session(session_id)
                if not s:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                qs = urllib.parse.parse_qs(u.query)
                path_q = qs.get("path")
                if not path_q or not path_q[0]:
                    _json_response(self, 400, {"error": "path required"})
                    return
                rel = path_q[0]
                staged_q = qs.get("staged")
                staged = bool(staged_q and staged_q[0] == "1")
                cwd = Path(s.cwd).expanduser()
                if not cwd.is_absolute():
                    cwd = cwd.resolve()
                try:
                    _require_git_repo(cwd)
                except RuntimeError as e:
                    _json_response(self, 409, {"error": str(e)})
                    return
                try:
                    _target, _repo_root, rel = _resolve_git_path(cwd, rel)
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                args = ["diff", "-U3"]
                if staged:
                    args.append("--cached")
                args.extend(["--", rel])
                diff = _run_git(
                    cwd,
                    args,
                    timeout_s=GIT_DIFF_TIMEOUT_SECONDS,
                    max_bytes=GIT_DIFF_MAX_BYTES,
                )
                _json_response(self, 200, {"ok": True, "cwd": str(cwd), "path": rel, "staged": staged, "diff": diff})
                return

            if path.startswith("/api/sessions/") and path.endswith("/git/file_versions"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                if not session_id:
                    self.send_error(404)
                    return
                MANAGER.refresh_session_meta(session_id)
                s = MANAGER.get_session(session_id)
                if not s:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                qs = urllib.parse.parse_qs(u.query)
                path_q = qs.get("path")
                if not path_q or not path_q[0]:
                    _json_response(self, 400, {"error": "path required"})
                    return
                rel = path_q[0]
                cwd = Path(s.cwd).expanduser()
                if not cwd.is_absolute():
                    cwd = cwd.resolve()
                try:
                    _require_git_repo(cwd)
                except RuntimeError as e:
                    _json_response(self, 409, {"error": str(e)})
                    return
                try:
                    p, _repo_root, rel = _resolve_git_path(cwd, rel)
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                current_text = ""
                current_size = 0
                current_exists = bool(p.exists() and p.is_file())
                if current_exists:
                    current_text, current_size = _read_text_file_strict(p, max_bytes=FILE_READ_MAX_BYTES)
                try:
                    MANAGER.files_add(session_id, str(p))
                except KeyError:
                    pass
                base_exists = False
                base_text = ""
                try:
                    base_text = _run_git(
                        cwd,
                        ["show", f"HEAD:{rel}"],
                        timeout_s=GIT_DIFF_TIMEOUT_SECONDS,
                        max_bytes=FILE_READ_MAX_BYTES,
                    )
                    base_exists = True
                except RuntimeError:
                    base_exists = False
                    base_text = ""
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "cwd": str(cwd),
                        "path": rel,
                        "abs_path": str(p),
                        "current_exists": current_exists,
                        "current_size": int(current_size),
                        "current_text": current_text,
                        "base_exists": base_exists,
                        "base_text": base_text,
                    },
                )
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
                offset_q = qs.get("offset")
                if offset_q is None:
                    offset = 0
                else:
                    if not offset_q:
                        raise ValueError("invalid offset")
                    offset = int(offset_q[0])
                if offset < 0:
                    offset = 0
                init_q = qs.get("init")
                init = bool(init_q and init_q[0] == "1")
                before_q = qs.get("before")
                if before_q is None:
                    before = 0
                else:
                    if not before_q:
                        raise ValueError("invalid before")
                    before = int(before_q[0])
                before = max(0, before)
                if s.log_path is None or (not s.log_path.exists()):
                    state = MANAGER.get_state(session_id)
                    if not isinstance(state, dict):
                        raise ValueError("invalid broker state response")
                    if "busy" not in state:
                        raise ValueError("missing busy from broker state response")
                    if "queue_len" not in state:
                        raise ValueError("missing queue_len from broker state response")
                    queue_val = int(state.get("queue_len")) + MANAGER._compat_queue_len(session_id)
                    state_token = state.get("token")
                    if not (isinstance(state_token, dict) or (state_token is None)):
                        raise ValueError("invalid token from broker state response")
                    token_val = state_token
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
                            # Definition: a session with no associated rollout log is idle.
                            "busy": False,
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
                    limit_q = qs.get("limit")
                    if limit_q is None:
                        limit = 80
                    else:
                        if not limit_q:
                            raise ValueError("invalid limit")
                        limit = int(limit_q[0])
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
                state = MANAGER.get_state(session_id)
                dt_state_ms = (time.perf_counter() - t0_state) * 1000.0
                s2 = MANAGER.get_session(session_id)
                if token_update is not None and s2 is not None:
                    s2.token = token_update
                if not isinstance(state, dict):
                    raise ValueError("invalid broker state response")
                if "busy" not in state:
                    raise ValueError("missing busy from broker state response")
                if "queue_len" not in state:
                    raise ValueError("missing queue_len from broker state response")
                state_busy = bool(state.get("busy"))
                state_queue = int(state.get("queue_len"))

                t0_idle = time.perf_counter()
                idle_val = MANAGER.idle_from_log(session_id)
                dt_idle_ms = (time.perf_counter() - t0_idle) * 1000.0
                diag["idle_from_log_ms"] = round(dt_idle_ms, 3)

                busy_val = bool(state_busy) or (not bool(idle_val))
                queue_val = state_queue + MANAGER._compat_queue_len(session_id)

                token_val: dict[str, Any] | None = None
                if "token" in state:
                    state_token = state.get("token")
                    if not (isinstance(state_token, dict) or (state_token is None)):
                        raise ValueError("invalid token from broker state response")
                    token_val = state_token
                elif isinstance(token_update, dict):
                    token_val = token_update
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
            traceback.print_exc()
            _json_response(self, 500, {"error": str(e), "trace": traceback.format_exc()})

    def do_POST(self) -> None:
        try:
            u = urllib.parse.urlparse(self.path)
            path = u.path
            if URL_PREFIX:
                if path == URL_PREFIX:
                    loc = URL_PREFIX + "/"
                    if u.query:
                        loc = loc + "?" + u.query
                    self.send_response(308)
                    self.send_header("Location", loc)
                    self.end_headers()
                    return
                stripped = _strip_url_prefix(URL_PREFIX, path)
                if stripped is None:
                    self.send_error(404)
                    return
                path = stripped

            if path == "/api/login":
                body = _read_body(self)
                body_text = body.decode("utf-8")
                if not body_text.strip():
                    raise ValueError("empty request body")
                obj = json.loads(body_text)
                if not isinstance(obj, dict):
                    raise ValueError("invalid json body (expected object)")
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
                    f"{COOKIE_NAME}=deleted; Path={COOKIE_PATH}; Max-Age=0; HttpOnly; SameSite=Strict",
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
                body_text = body.decode("utf-8")
                if not body_text.strip():
                    raise ValueError("empty request body")
                obj = json.loads(body_text)
                if not isinstance(obj, dict):
                    raise ValueError("invalid json body (expected object)")
                cwd = obj.get("cwd")
                if not isinstance(cwd, str) or not cwd.strip():
                    _json_response(self, 400, {"error": "cwd required"})
                    return
                resume_session_id_raw = obj.get("resume_session_id")
                if resume_session_id_raw is None:
                    resume_session_id = None
                elif isinstance(resume_session_id_raw, str):
                    resume_session_id = resume_session_id_raw.strip() or None
                else:
                    _json_response(self, 400, {"error": "resume_session_id must be a string"})
                    return
                worktree_branch_raw = obj.get("worktree_branch")
                if worktree_branch_raw is None:
                    worktree_branch = None
                elif isinstance(worktree_branch_raw, str):
                    worktree_branch = worktree_branch_raw.strip() or None
                else:
                    _json_response(self, 400, {"error": "worktree_branch must be a string"})
                    return
                args = obj.get("args")
                if args is None:
                    args_list = None
                elif isinstance(args, list) and all(isinstance(x, str) for x in args):
                    args_list = [x for x in args if x]
                else:
                    _json_response(self, 400, {"error": "args must be a list of strings"})
                    return
                try:
                    res = MANAGER.spawn_web_session(cwd=cwd, args=args_list, resume_session_id=resume_session_id, worktree_branch=worktree_branch)
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                _json_response(self, 200, {"ok": True, **res})
                return

            if path == "/api/files/read":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                body = _read_body(self)
                body_text = body.decode("utf-8")
                if not body_text.strip():
                    raise ValueError("empty request body")
                obj = json.loads(body_text)
                if not isinstance(obj, dict):
                    raise ValueError("invalid json body (expected object)")
                path_raw = obj.get("path")
                if not isinstance(path_raw, str) or not path_raw.strip():
                    _json_response(self, 400, {"error": "path required"})
                    return
                session_id_raw = obj.get("session_id")
                session_id = session_id_raw if isinstance(session_id_raw, str) and session_id_raw else ""
                try:
                    path_obj = _resolve_client_file_path(session_id=session_id, raw_path=path_raw)
                    raw, size, kind, image_ctype = _inspect_openable_file(path_obj)
                except FileNotFoundError as e:
                    _json_response(self, 404, {"error": str(e)})
                    return
                except PermissionError as e:
                    _json_response(self, 403, {"error": str(e)})
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                if session_id:
                    try:
                        MANAGER.files_add(session_id, str(path_obj))
                    except KeyError:
                        pass
                if kind == "image":
                    _json_response(
                        self,
                        200,
                        {
                            "ok": True,
                            "kind": "image",
                            "content_type": image_ctype,
                            "path": str(path_obj),
                            "size": int(size),
                            "image_url": f"/api/files/blob?path={urllib.parse.quote(str(path_obj))}",
                        },
                    )
                    return
                text = raw.decode("utf-8", errors="replace")
                _json_response(self, 200, {"ok": True, "kind": "text", "path": str(path_obj), "size": int(size), "text": text})
                return

            if path == "/api/files/inspect":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                body = _read_body(self)
                body_text = body.decode("utf-8")
                if not body_text.strip():
                    raise ValueError("empty request body")
                obj = json.loads(body_text)
                if not isinstance(obj, dict):
                    raise ValueError("invalid json body (expected object)")
                path_raw = obj.get("path")
                if not isinstance(path_raw, str) or not path_raw.strip():
                    _json_response(self, 400, {"error": "path required"})
                    return
                session_id_raw = obj.get("session_id")
                session_id = session_id_raw if isinstance(session_id_raw, str) and session_id_raw else ""
                try:
                    path_obj = _resolve_client_file_path(session_id=session_id, raw_path=path_raw)
                    size, kind, image_ctype = _inspect_client_path(path_obj)
                except FileNotFoundError as e:
                    _json_response(self, 404, {"error": str(e)})
                    return
                except PermissionError as e:
                    _json_response(self, 403, {"error": str(e)})
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "path": str(path_obj),
                        "kind": kind,
                        "content_type": image_ctype,
                        "size": int(size),
                    },
                )
                return

            if path == "/api/files/blob":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                qs = urllib.parse.parse_qs(u.query)
                path_q = qs.get("path")
                if not path_q or not path_q[0]:
                    _json_response(self, 400, {"error": "path required"})
                    return
                path_obj = Path(path_q[0]).expanduser().resolve()
                if not path_obj.exists():
                    _json_response(self, 404, {"error": "file not found"})
                    return
                if not path_obj.is_file():
                    _json_response(self, 400, {"error": "path is not a file"})
                    return
                raw = path_obj.read_bytes()
                _kind, ctype = _file_kind(path_obj, raw)
                if ctype is None:
                    _json_response(self, 400, {"error": "file is not a supported image"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(raw)
                return

            if path.startswith("/api/sessions/") and path.endswith("/delete"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                _read_body(self)
                ok = MANAGER.delete_session(session_id)
                if not ok:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, {"ok": True})
                return

            if path.startswith("/api/sessions/") and path.endswith("/edit"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                body = _read_body(self)
                body_text = body.decode("utf-8")
                if not body_text.strip():
                    raise ValueError("empty request body")
                obj = json.loads(body_text)
                if not isinstance(obj, dict):
                    raise ValueError("invalid json body (expected object)")
                name = obj.get("name")
                if not isinstance(name, str):
                    _json_response(self, 400, {"error": "name required"})
                    return
                try:
                    alias, sidebar_meta = MANAGER.edit_session(
                        session_id,
                        name=name,
                        priority_offset=obj.get("priority_offset"),
                        snooze_until=obj.get("snooze_until"),
                        dependency_session_id=obj.get("dependency_session_id"),
                    )
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                _json_response(self, 200, {"ok": True, "alias": alias, **sidebar_meta})
                return

            if path.startswith("/api/sessions/") and path.endswith("/rename"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                body = _read_body(self)
                body_text = body.decode("utf-8")
                if not body_text.strip():
                    raise ValueError("empty request body")
                obj = json.loads(body_text)
                if not isinstance(obj, dict):
                    raise ValueError("invalid json body (expected object)")
                name = obj.get("name")
                if not isinstance(name, str):
                    _json_response(self, 400, {"error": "name required"})
                    return
                try:
                    alias = MANAGER.alias_set(session_id, name)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, {"ok": True, "alias": alias})
                return

            if path.startswith("/api/sessions/") and path.endswith("/send"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                body = _read_body(self)
                body_text = body.decode("utf-8")
                if not body_text.strip():
                    raise ValueError("empty request body")
                obj = json.loads(body_text)
                if not isinstance(obj, dict):
                    raise ValueError("invalid json body (expected object)")
                text = obj.get("text")
                if not isinstance(text, str) or not text.strip():
                    _json_response(self, 400, {"error": "text required"})
                    return
                res = MANAGER.send(session_id, text)
                _json_response(self, 200, res)
                return

            if path.startswith("/api/sessions/") and path.endswith("/enqueue"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                body = _read_body(self)
                body_text = body.decode("utf-8")
                if not body_text.strip():
                    raise ValueError("empty request body")
                obj = json.loads(body_text)
                if not isinstance(obj, dict):
                    raise ValueError("invalid json body (expected object)")
                text = obj.get("text")
                if not isinstance(text, str) or not text.strip():
                    _json_response(self, 400, {"error": "text required"})
                    return
                try:
                    res = MANAGER.enqueue(session_id, text)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                except ValueError as e:
                    _json_response(self, 502, {"error": str(e)})
                    return
                _json_response(self, 200, res)
                return

            if path.startswith("/api/sessions/") and path.endswith("/queue/delete"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                body = _read_body(self)
                body_text = body.decode("utf-8")
                if not body_text.strip():
                    raise ValueError("empty request body")
                obj = json.loads(body_text)
                if not isinstance(obj, dict):
                    raise ValueError("invalid json body (expected object)")
                idx = obj.get("index")
                if not isinstance(idx, int):
                    _json_response(self, 400, {"error": "index required"})
                    return
                try:
                    res = MANAGER.queue_delete(session_id, idx)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                except ValueError as e:
                    _json_response(self, 502, {"error": str(e)})
                    return
                _json_response(self, 200, res)
                return

            if path.startswith("/api/sessions/") and path.endswith("/queue/update"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                body = _read_body(self)
                body_text = body.decode("utf-8")
                if not body_text.strip():
                    raise ValueError("empty request body")
                obj = json.loads(body_text)
                if not isinstance(obj, dict):
                    raise ValueError("invalid json body (expected object)")
                idx = obj.get("index")
                text = obj.get("text")
                if not isinstance(idx, int):
                    _json_response(self, 400, {"error": "index required"})
                    return
                if not isinstance(text, str) or not text.strip():
                    _json_response(self, 400, {"error": "text required"})
                    return
                try:
                    res = MANAGER.queue_update(session_id, idx, text)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                except ValueError as e:
                    _json_response(self, 502, {"error": str(e)})
                    return
                _json_response(self, 200, res)
                return

            if path.startswith("/api/sessions/") and path.endswith("/harness"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                body = _read_body(self)
                body_text = body.decode("utf-8")
                if not body_text.strip():
                    raise ValueError("empty request body")
                obj = json.loads(body_text)
                if not isinstance(obj, dict):
                    raise ValueError("invalid json body (expected object)")
                enabled_raw = obj.get("enabled", None)
                request_raw = obj.get("request", None)
                if "text" in obj:
                    _json_response(self, 400, {"error": "unknown field: text (use request)"})
                    return
                enabled: bool | None
                if enabled_raw is None:
                    enabled = None
                else:
                    enabled = bool(enabled_raw)

                if request_raw is not None and (not isinstance(request_raw, str)):
                    _json_response(self, 400, {"error": "request must be a string"})
                    return
                request: str | None
                if request_raw is not None:
                    request = request_raw
                else:
                    request = None

                cfg = MANAGER.harness_set(session_id, enabled=enabled, request=request)
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

            if path.startswith("/api/sessions/") and (path.endswith("/inject_file") or path.endswith("/inject_image")):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                body = _read_body(self, limit=20 * 1024 * 1024)
                body_text = body.decode("utf-8")
                if not body_text.strip():
                    raise ValueError("empty request body")
                obj = json.loads(body_text)
                if not isinstance(obj, dict):
                    raise ValueError("invalid json body (expected object)")
                data_b64 = obj.get("data_b64")
                filename = obj.get("filename")
                if not isinstance(filename, str) or (not filename.strip()):
                    raise ValueError("filename required")
                if not isinstance(data_b64, str) or not data_b64:
                    _json_response(self, 400, {"error": "data_b64 required"})
                    return
                try:
                    raw = base64.b64decode(data_b64.encode("ascii"), validate=True)
                except Exception:
                    _json_response(self, 400, {"error": "invalid base64"})
                    return
                try:
                    out_path = _stage_uploaded_file(session_id, filename, raw)
                except ValueError as e:
                    status = 413 if str(e) == "file too large" else 400
                    _json_response(self, status, {"error": str(e)})
                    return

                try:
                    inject_text = _attachment_inject_text(attachment_index, out_path)
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return

                # Bracketed paste: inject the staged attachment line into the active broker input.
                seq = f"\x1b[200~{inject_text}\x1b[201~"
                try:
                    resp = MANAGER.inject_keys(session_id, seq)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, {"ok": True, "path": str(out_path), "inject_text": inject_text, "broker": resp})
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
            traceback.print_exc()
            _json_response(self, 500, {"error": str(e), "trace": traceback.format_exc()})

    def log_message(self, fmt: str, *args: Any) -> None:
        # Quiet default logging to keep terminal usable.
        return


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class ThreadingHTTPServerV6(ThreadingHTTPServer):
    address_family = socket.AF_INET6

    def server_bind(self) -> None:
        v6only = getattr(socket, "IPV6_V6ONLY", None)
        if v6only is not None:
            self.socket.setsockopt(socket.IPPROTO_IPV6, v6only, 0)
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
    if ":" in host:
        server = ThreadingHTTPServerV6((host, DEFAULT_PORT), Handler)
    else:
        server = ThreadingHTTPServer((host, DEFAULT_PORT), Handler)

    def _sigterm(_signo: int, _frame: Any) -> None:
        # BaseServer.shutdown() must not run in the serve_forever thread.
        MANAGER.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    server.serve_forever()


if __name__ == "__main__":
    main()
                attachment_index = obj.get("attachment_index")
                if isinstance(attachment_index, bool) or not isinstance(attachment_index, int):
                    _json_response(self, 400, {"error": "attachment_index must be an integer"})
                    return
