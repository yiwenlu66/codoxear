#!/usr/bin/env python3
from __future__ import annotations

import base64
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
import shutil
import threading
import time
import traceback
import urllib.parse
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cli_support import cli_bin as _cli_bin
from .cli_support import cli_home as _cli_home
from .cli_support import infer_cli_from_log_path as _infer_cli_from_log_path
from .cli_support import normalize_cli_name as _normalize_cli_name
from .cli_support import parse_cli_name as _parse_cli_name
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
FILE_HISTORY_PATH = APP_DIR / "session_files.json"

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
DEFAULT_SPAWN_CLI = _normalize_cli_name(os.environ.get("CODEX_WEB_DEFAULT_CLI"), default="codex")

DEFAULT_HOST = os.environ.get("CODEX_WEB_HOST", "::")
DEFAULT_PORT = int(os.environ.get("CODEX_WEB_PORT", "8743"))
HARNESS_IDLE_SECONDS = int(os.environ.get("CODEX_WEB_HARNESS_IDLE_SECONDS", "300"))
HARNESS_SWEEP_SECONDS = float(os.environ.get("CODEX_WEB_HARNESS_SWEEP_SECONDS", "2.5"))
HARNESS_MAX_SCAN_BYTES = int(os.environ.get("CODEX_WEB_HARNESS_MAX_SCAN_BYTES", str(8 * 1024 * 1024)))
DISCOVER_MIN_INTERVAL_SECONDS = float(os.environ.get("CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS", "1.0"))
LOG_BUSY_FROM_LOG_STALE_SECONDS = float(os.environ.get("CODEX_WEB_LOG_BUSY_FROM_LOG_STALE_SECONDS", "45.0"))
CHAT_INIT_SEED_SCAN_BYTES = int(os.environ.get("CODEX_WEB_CHAT_INIT_SEED_SCAN_BYTES", str(512 * 1024)))
CHAT_INIT_MAX_SCAN_BYTES = int(os.environ.get("CODEX_WEB_CHAT_INIT_MAX_SCAN_BYTES", str(128 * 1024 * 1024)))
CHAT_INDEX_INCREMENT_BYTES = int(os.environ.get("CODEX_WEB_CHAT_INDEX_INCREMENT_BYTES", str(2 * 1024 * 1024)))
CHAT_INDEX_RESEED_THRESHOLD_BYTES = int(os.environ.get("CODEX_WEB_CHAT_INDEX_RESEED_THRESHOLD_BYTES", str(16 * 1024 * 1024)))
CHAT_INDEX_MAX_EVENTS = int(os.environ.get("CODEX_WEB_CHAT_INDEX_MAX_EVENTS", "12000"))
METRICS_WINDOW = int(os.environ.get("CODEX_WEB_METRICS_WINDOW", "256"))
FILE_READ_MAX_BYTES = int(os.environ.get("CODEX_WEB_FILE_READ_MAX_BYTES", str(2 * 1024 * 1024)))
FILE_WRITE_MAX_BYTES = int(os.environ.get("CODEX_WEB_FILE_WRITE_MAX_BYTES", str(FILE_READ_MAX_BYTES)))
FILE_HISTORY_MAX = int(os.environ.get("CODEX_WEB_FILE_HISTORY_MAX", "20"))
UPDATE_CHECK_TTL_SECONDS = float(os.environ.get("CODEX_WEB_UPDATE_CHECK_TTL_SECONDS", "600"))
UPDATE_CHECK_TIMEOUT_SECONDS = float(os.environ.get("CODEX_WEB_UPDATE_CHECK_TIMEOUT_SECONDS", "2.0"))
UPDATE_CHECK_REMOTE = str(os.environ.get("CODEX_WEB_UPDATE_REMOTE", "")).strip()
UPDATE_CHECK_BRANCH = str(os.environ.get("CODEX_WEB_UPDATE_BRANCH", "")).strip()
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


def _normalize_queue_list(raw: list[Any]) -> list[str]:
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        if not item.strip():
            continue
        out.append(item)
    return out


def _normalize_outgoing_text_for_cli(text: str, cli: str) -> str:
    raw = text if isinstance(text, str) else ""
    if _normalize_cli_name(cli, default="codex") != "claude":
        return raw
    # Claude CLI treats a leading "!" as local shell command. Escape markdown
    # image prefix so `![...]` stays literal text in chat prompts.
    return re.sub(r"^(\s*)!\[", r"\1\\![", raw, count=1)

_SESSION_ID_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.I)
_TAIL_SHORT_SHARD_RE = re.compile(r"^[A-Za-z0-9·✢✶✻✽*\.]+$")
_METRICS_LOCK = threading.Lock()
_METRICS: dict[str, list[float]] = {}
_UPDATE_CHECK_LOCK = threading.Lock()
_UPDATE_CHECK_CACHE: dict[str, Any] | None = None

def _strip_ansi_sequences(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "\x1b":
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= n:
            break
        nxt = text[i]
        if nxt == "[":
            i += 1
            while i < n:
                c = text[i]
                if 0x40 <= ord(c) <= 0x7E:
                    i += 1
                    break
                i += 1
            continue
        if nxt == "]":
            i += 1
            while i < n:
                c = text[i]
                if c == "\x07":
                    i += 1
                    break
                if c == "\x1b" and (i + 1) < n and text[i + 1] == "\\":
                    i += 2
                    break
                i += 1
            continue
        if nxt in ("P", "^", "_"):
            i += 1
            while i < n:
                if text[i] == "\x1b" and (i + 1) < n and text[i + 1] == "\\":
                    i += 2
                    break
                i += 1
            continue
        if nxt in ("(", ")", "*", "+", "-", ".", "/"):
            i += 1
            if i < n:
                i += 1
            continue
        i += 1
    return "".join(out)


def _sanitize_tail_text(text: str) -> str:
    cleaned = _strip_ansi_sequences(text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_RE.sub("", cleaned)


def _has_cjk(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF):
            return True
    return False


def _sanitize_claude_tail_text(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    blank = False
    box_chars = "│─╭╮╰╯▐▛▜▝▘"
    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            if not blank:
                out.append("")
            blank = True
            continue
        blank = False
        flat = "".join(stripped.split()).lower()
        if flat in ("esctointerrupt", "?forshortcuts"):
            continue
        if ("flowing…" in flat) or ("flowing..." in flat) or ("brewedfor" in flat):
            continue
        if len(stripped) >= 8:
            box_count = sum(1 for ch in stripped if ch in box_chars)
            if (box_count / float(len(stripped))) > 0.35:
                continue
        if stripped and all(not ch.isalnum() for ch in stripped):
            if stripped not in ("❯", "↯", "───"):
                continue
        if _has_cjk(stripped):
            out.append(line)
            continue
        if (" " not in stripped) and len(stripped) <= 6:
            up = stripped.upper()
            if up not in ("OK", "DONE", "YES", "NO") and not stripped.startswith(("❯", "●", "⎿", "↯")):
                continue
        if (
            len(stripped) <= 8
            and bool(_TAIL_SHORT_SHARD_RE.fullmatch(stripped))
            and any(ch.isdigit() for ch in stripped)
        ):
            continue
        if (
            re.search(r"[A-Za-z]{3,}", stripped)
            or ("/" in stripped)
            or ("\\" in stripped)
            or stripped.startswith(("●", "⎿", "❯", "↯", "───"))
        ):
            out.append(line)
            continue
        if len(stripped) >= 10:
            out.append(line)
            continue
    compact: list[str] = []
    prev_blank = False
    for line in out:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        compact.append(line)
        prev_blank = is_blank
    while compact and (not compact[0].strip()):
        compact.pop(0)
    while compact and (not compact[-1].strip()):
        compact.pop()
    return "\n".join(compact)


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


def _git_run(args: list[str], *, required: bool = True, timeout_s: float | None = None) -> str:
    timeout = UPDATE_CHECK_TIMEOUT_SECONDS if timeout_s is None else float(timeout_s)
    timeout = max(0.2, timeout)
    repo_dir = Path(__file__).resolve().parent.parent
    cmd = ["git", *args]
    try:
        res = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as e:
        if required:
            raise RuntimeError("git is not installed") from e
        return ""
    except subprocess.TimeoutExpired as e:
        if required:
            raise RuntimeError(f"git {' '.join(args)} timed out after {timeout:.1f}s") from e
        return ""
    if res.returncode != 0:
        if required:
            detail = (res.stderr or res.stdout or "").strip()
            if detail:
                raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
            raise RuntimeError(f"git {' '.join(args)} failed (rc={res.returncode})")
        return ""
    return (res.stdout or "").strip()


def _parse_upstream_ref(raw: str) -> tuple[str, str] | None:
    text = str(raw or "").strip()
    if not text or text == "HEAD" or "/" not in text:
        return None
    remote, branch = text.split("/", 1)
    remote = remote.strip()
    branch = branch.strip()
    if not remote or not branch:
        return None
    return remote, branch


def _select_update_remote_branch(local_branch: str) -> tuple[str, str]:
    remote = UPDATE_CHECK_REMOTE
    branch = UPDATE_CHECK_BRANCH
    if not (remote and branch):
        upstream_raw = _git_run(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"], required=False)
        upstream = _parse_upstream_ref(upstream_raw)
        if upstream is not None:
            up_remote, up_branch = upstream
            if not remote:
                remote = up_remote
            if not branch:
                branch = up_branch
    if not remote:
        remote = "origin"
    if not branch:
        branch = local_branch
    if not branch or branch == "HEAD":
        branch = "main"
    return remote, branch


def _parse_ls_remote_head(raw: str, *, branch: str, remote: str) -> str:
    target_ref = f"refs/heads/{branch}"
    for line in str(raw or "").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        sha, ref = parts[0], parts[1]
        if ref != target_ref:
            continue
        if _GIT_SHA_RE.fullmatch(sha):
            return sha.lower()
    raise RuntimeError(f"remote branch not found: {remote}/{branch}")


def _git_divergence(local_commit: str, remote_commit: str) -> tuple[int, int]:
    raw = _git_run(["rev-list", "--left-right", "--count", f"{local_commit}...{remote_commit}"])
    parts = raw.split()
    if len(parts) < 2:
        raise RuntimeError(f"unexpected rev-list output: {raw!r}")
    try:
        local_only = int(parts[0])
        remote_only = int(parts[1])
    except ValueError as e:
        raise RuntimeError(f"unexpected rev-list output: {raw!r}") from e
    return local_only, remote_only


def _github_repo_base(remote_url: str) -> str | None:
    raw = str(remote_url or "").strip()
    if not raw:
        return None
    path = ""
    if raw.startswith("git@github.com:"):
        path = raw[len("git@github.com:") :]
    elif raw.startswith("ssh://git@github.com/"):
        path = raw[len("ssh://git@github.com/") :]
    elif raw.startswith("https://github.com/"):
        path = raw[len("https://github.com/") :]
    elif raw.startswith("http://github.com/"):
        path = raw[len("http://github.com/") :]
    elif raw.startswith("git://github.com/"):
        path = raw[len("git://github.com/") :]
    else:
        return None
    path = path.strip().lstrip("/").rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if "/" not in path:
        return None
    return f"https://github.com/{path}"


def _github_compare_url(remote_url: str, local_commit: str, remote_commit: str) -> str | None:
    base = _github_repo_base(remote_url)
    if not base:
        return None
    return f"{base}/compare/{local_commit}...{remote_commit}"


def _check_update_status_now() -> dict[str, Any]:
    checked_at = int(time.time())
    try:
        _git_run(["rev-parse", "--is-inside-work-tree"])
        local_commit = _git_run(["rev-parse", "HEAD"]).strip().lower()
        if not _GIT_SHA_RE.fullmatch(local_commit):
            raise RuntimeError(f"invalid local commit: {local_commit!r}")
        local_branch = _git_run(["rev-parse", "--abbrev-ref", "HEAD"], required=False).strip() or "HEAD"
        remote, branch = _select_update_remote_branch(local_branch)
        remote_head_raw = _git_run(["ls-remote", "--heads", remote, branch])
        remote_commit = _parse_ls_remote_head(remote_head_raw, branch=branch, remote=remote)
        if remote_commit == local_commit:
            local_only, remote_only = 0, 0
        else:
            try:
                local_only, remote_only = _git_divergence(local_commit, remote_commit)
            except Exception:
                # The remote head object may not exist in the local object database
                # yet (no fetch). A commit mismatch still means a newer remote state
                # is visible from ls-remote, so surface it as update-available.
                local_only, remote_only = 0, 1
        remote_url = _git_run(["remote", "get-url", remote], required=False)
        out: dict[str, Any] = {
            "ok": True,
            "checked_at": checked_at,
            "update_available": bool(remote_only > 0),
            "remote": remote,
            "branch": branch,
            "local_commit": local_commit,
            "remote_commit": remote_commit,
            "local_only_commits": int(local_only),
            "remote_only_commits": int(remote_only),
        }
        compare_url = _github_compare_url(remote_url, local_commit, remote_commit)
        if compare_url:
            out["compare_url"] = compare_url
        return out
    except Exception as e:
        return {
            "ok": False,
            "checked_at": checked_at,
            "update_available": False,
            "error": str(e),
        }


def _update_status(force: bool = False) -> dict[str, Any]:
    global _UPDATE_CHECK_CACHE
    now = float(time.time())
    ttl = max(5.0, float(UPDATE_CHECK_TTL_SECONDS))
    if not force:
        with _UPDATE_CHECK_LOCK:
            cached = _UPDATE_CHECK_CACHE
            if isinstance(cached, dict):
                ts = float(cached.get("ts", 0.0))
                if ts > 0 and (now - ts) < ttl:
                    data = cached.get("data")
                    if isinstance(data, dict):
                        return dict(data)
    data = _check_update_status_now()
    with _UPDATE_CHECK_LOCK:
        _UPDATE_CHECK_CACHE = {"ts": now, "data": dict(data)}
    return data


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

def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _claude_args_override_session(args: list[str]) -> bool:
    if not args:
        return False
    for a in args:
        if not isinstance(a, str):
            continue
        if a in ("-c", "--continue", "-r", "--resume", "--session-id", "--from-pr"):
            return True
        if a.startswith("--resume=") or a.startswith("--session-id=") or a.startswith("--from-pr="):
            return True
    return False

def _tmux_pane_pid(tmux_bin: str, session_name: str, env: dict[str, str]) -> int | None:
    try:
        res = subprocess.run(
            [tmux_bin, "list-panes", "-t", session_name, "-F", "#{pane_pid}"],
            capture_output=True,
            text=True,
            env=env,
            timeout=1.5,
        )
    except Exception:
        return None
    if res.returncode != 0:
        return None
    for line in (res.stdout or "").splitlines():
        try:
            pid = int(line.strip())
        except Exception:
            continue
        if pid > 0:
            return pid
    return None


def _tmux_global_env_has_nonempty(tmux_bin: str, key: str, env: dict[str, str]) -> bool:
    if not key:
        return False
    try:
        res = subprocess.run(
            [tmux_bin, "show-environment", "-g", key],
            capture_output=True,
            text=True,
            env=env,
            timeout=1.5,
            check=False,
        )
    except Exception:
        return False
    if res.returncode != 0:
        return False
    line = (res.stdout or "").strip()
    if not line or line.startswith("-"):
        return False
    prefix = key + "="
    if line.startswith(prefix):
        return bool(line[len(prefix) :].strip())
    return line == key


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
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Expires", "0")
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


def _safe_filename(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            out.append(ch)
    s = "".join(out).strip().replace(" ", "_")
    if not s:
        return "image"
    return s[:96]


def _clean_alias(name: str) -> str:
    if not isinstance(name, str):
        return ""
    # Collapse whitespace and cap length to keep titles readable.
    cleaned = " ".join(name.split()).strip()
    if not cleaned:
        return ""
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rstrip()
    return cleaned


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


def _coerce_main_thread_log(*, thread_id: str, log_path: Path) -> tuple[str, Path]:
    try:
        sm = _read_session_meta(log_path)
    except Exception:
        # Non-Codex logs (for example Claude project logs) do not start with
        # session_meta; keep the original path untouched.
        return thread_id, log_path
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
) -> tuple[int, int, int, float | None, float | None, dict[str, Any] | None, list[dict[str, Any]]]:
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


def _last_assistant_ts_from_tail(
    path: Path,
    *,
    max_scan_bytes: int,
) -> float | None:
    return _rollout_log._last_assistant_ts_from_tail(path, max_scan_bytes=max_scan_bytes)


def _busy_from_state_and_log_idle(*, state_busy: bool, idle_from_log: bool, log_path: Path | None) -> bool:
    # Broker runtime state is primary. Log-based busy is only used as a short
    # fallback window to avoid stale "busy" when no new events arrive.
    if state_busy:
        return True
    if idle_from_log:
        return False
    if log_path is None:
        return False
    try:
        age = max(0.0, time.time() - float(log_path.stat().st_mtime))
    except Exception:
        return True
    return age <= max(float(LOG_BUSY_FROM_LOG_STALE_SECONDS), 0.0)


def _read_cli_config() -> dict[str, Any]:
    """Read configuration for all three CLIs from files and environment."""
    config: dict[str, Any] = {
        "codex": {},
        "claude": {},
        "gemini": {},
        "env": {},
    }

    # Read Codex config
    try:
        codex_config_path = Path.home() / ".codex" / "config.toml"
        if codex_config_path.exists():
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore
                except ImportError:
                    config["codex"]["error"] = "toml library not available"
                    tomllib = None

            if tomllib:
                with open(codex_config_path, "rb") as f:
                    codex_data = tomllib.load(f)
                    config["codex"]["config_toml"] = codex_data
                    # Extract commonly used fields
                    if "model_providers" in codex_data:
                        for provider_name, provider_data in codex_data["model_providers"].items():
                            if isinstance(provider_data, dict) and "base_url" in provider_data:
                                config["codex"]["base_url"] = provider_data["base_url"]
                                break
                    if "model" in codex_data:
                        config["codex"]["model"] = codex_data["model"]
    except Exception as e:
        config["codex"]["error"] = str(e)

    try:
        codex_auth_path = Path.home() / ".codex" / "auth.json"
        if codex_auth_path.exists():
            with open(codex_auth_path, "r") as f:
                auth_data = json.load(f)
                config["codex"]["auth_json"] = auth_data
                if "OPENAI_API_KEY" in auth_data:
                    config["codex"]["api_key"] = auth_data["OPENAI_API_KEY"]
    except Exception as e:
        config["codex"]["auth_error"] = str(e)

    # Read Claude config
    try:
        claude_settings_path = Path.home() / ".claude" / "settings.json"
        if claude_settings_path.exists():
            with open(claude_settings_path, "r") as f:
                claude_data = json.load(f)
                config["claude"]["settings_json"] = claude_data
                if "model" in claude_data:
                    config["claude"]["model"] = claude_data["model"]
    except Exception as e:
        config["claude"]["error"] = str(e)

    # Read Gemini config
    try:
        gemini_settings_path = Path.home() / ".gemini" / "settings.json"
        if gemini_settings_path.exists():
            with open(gemini_settings_path, "r") as f:
                gemini_data = json.load(f)
                config["gemini"]["settings_json"] = gemini_data
    except Exception as e:
        config["gemini"]["error"] = str(e)

    # Read environment variables
    env_vars = [
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
        "GEMINI_API_KEY", "GOOGLE_GEMINI_BASE_URL", "GEMINI_MODEL",
    ]
    for var in env_vars:
        val = os.environ.get(var)
        if val:
            config["env"][var] = val

    return config


def _save_cli_config(updates: dict[str, Any]) -> dict[str, Any]:
    """Save configuration updates for CLIs."""
    result = {"ok": True, "updated": [], "note": "Configuration saved. Restart CLI sessions for changes to take effect."}

    # Update Codex config
    if "codex" in updates:
        codex_updates = updates["codex"]

        # Update config.toml using simple text replacement
        if "base_url" in codex_updates or "model" in codex_updates:
            try:
                codex_config_path = Path.home() / ".codex" / "config.toml"
                if codex_config_path.exists():
                    content = codex_config_path.read_text()

                    # Update base_url
                    if "base_url" in codex_updates and codex_updates["base_url"]:
                        import re
                        content = re.sub(
                            r'(base_url\s*=\s*")[^"]*(")',
                            r'\1' + codex_updates["base_url"] + r'\2',
                            content
                        )

                    # Update model
                    if "model" in codex_updates and codex_updates["model"]:
                        import re
                        content = re.sub(
                            r'^(model\s*=\s*")[^"]*(")',
                            r'\1' + codex_updates["model"] + r'\2',
                            content,
                            flags=re.MULTILINE
                        )

                    codex_config_path.write_text(content)
                    result["updated"].append("codex_config")
            except Exception as e:
                result["codex_config_error"] = str(e)

        # Update auth.json
        if "api_key" in codex_updates and codex_updates["api_key"]:
            try:
                codex_auth_path = Path.home() / ".codex" / "auth.json"
                auth_data = {}
                if codex_auth_path.exists():
                    with open(codex_auth_path, "r") as f:
                        auth_data = json.load(f)

                auth_data["OPENAI_API_KEY"] = codex_updates["api_key"]
                if "auth_mode" not in auth_data:
                    auth_data["auth_mode"] = "apikey"

                with open(codex_auth_path, "w") as f:
                    json.dump(auth_data, f, indent=2)

                result["updated"].append("codex_auth")
            except Exception as e:
                result["codex_auth_error"] = str(e)

    # Update Claude config
    if "claude" in updates:
        claude_updates = updates["claude"]

        # Update settings.json
        if "model" in claude_updates and claude_updates["model"]:
            try:
                claude_settings_path = Path.home() / ".claude" / "settings.json"
                claude_data = {}
                if claude_settings_path.exists():
                    with open(claude_settings_path, "r") as f:
                        claude_data = json.load(f)

                claude_data["model"] = claude_updates["model"]

                with open(claude_settings_path, "w") as f:
                    json.dump(claude_data, f, indent=2)

                result["updated"].append("claude_settings")
            except Exception as e:
                result["claude_error"] = str(e)

        # Note: Claude API key and base URL are typically set via environment variables
        # We'll add a note about this
        if "api_key" in claude_updates or "base_url" in claude_updates:
            result["claude_env_note"] = "Claude API key and base URL should be set via ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL environment variables"

    # Update Gemini config
    if "gemini" in updates:
        gemini_updates = updates["gemini"]

        # Gemini settings.json doesn't typically store API keys or base URLs
        # These are usually in environment variables
        if "api_key" in gemini_updates or "base_url" in gemini_updates or "model" in gemini_updates:
            result["gemini_env_note"] = "Gemini API key, base URL, and model should be set via GEMINI_API_KEY, GOOGLE_GEMINI_BASE_URL, and GEMINI_MODEL environment variables"

    return result


@dataclass
class Session:
    session_id: str
    thread_id: str
    broker_pid: int
    codex_pid: int
    cli: str
    owned: bool
    start_ts: float
    cwd: str
    log_path: Path | None
    sock_path: Path
    tmux_name: str | None = None
    busy: bool = False
    queue_len: int = 0
    token: dict[str, Any] | None = None
    last_turn_id: str | None = None
    last_chat_ts: float | None = None
    last_assistant_ts: float | None = None
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
        self._aliases: dict[str, str] = {}
        self._files: dict[str, list[str]] = {}
        self._harness_last_injected: dict[str, float] = {}
        self._harness_last_injected_scope: dict[str, float] = {}
        self._discover_existing(force=True)
        self._load_harness()
        self._load_aliases()
        self._load_files()
        self._harness_thr = threading.Thread(target=self._harness_loop, name="harness", daemon=True)
        self._harness_thr.start()

    def stop(self) -> None:
        self._stop.set()

    def _reset_log_caches(self, s: Session, *, meta_log_off: int) -> None:
        s.meta_thinking = 0
        s.meta_tools = 0
        s.meta_system = 0
        s.last_chat_ts = None
        s.last_assistant_ts = None
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

    def _cwd_key_from_value(self, cwd_raw: str) -> str | None:
        if not isinstance(cwd_raw, str):
            return None
        cwd = cwd_raw.strip()
        if not cwd or cwd == "?":
            return None
        try:
            cwd_norm = str(Path(cwd).expanduser().resolve())
        except Exception:
            cwd_norm = cwd
        if not cwd_norm:
            return None
        return f"cwd:{cwd_norm}"

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

    def files_remove(self, session_id: str, path: str) -> list[str]:
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
            if not cur:
                return []
            next_list = [x for x in cur if x != p]
            if next_list:
                self._files[key] = next_list
            else:
                if key in self._files:
                    self._files.pop(key, None)
            for lk in legacy_keys:
                if lk == key:
                    continue
                legacy = self._files.get(lk)
                if isinstance(legacy, list):
                    legacy_next = [x for x in legacy if x != p]
                    if legacy_next:
                        self._files[lk] = legacy_next
                    else:
                        self._files.pop(lk, None)
            dirty = True
        if dirty:
            self._save_files()
        return list(next_list)

    def files_remove_cwd(self, cwd_raw: str, path: str) -> list[str]:
        p = str(path).strip()
        if not p:
            return []
        cwd_key = self._cwd_key_from_value(cwd_raw)
        if not cwd_key:
            return []
        dirty = False
        out: list[str] = []
        with self._lock:
            keys = {cwd_key}
            for s in self._sessions.values():
                try:
                    key, legacy_keys, _sref = self._files_key_for_session(s.session_id)
                except KeyError:
                    continue
                if key != cwd_key:
                    continue
                for lk in legacy_keys:
                    keys.add(lk)
            for key in keys:
                cur = self._files.get(key)
                if not isinstance(cur, list) or not cur:
                    continue
                next_list = [x for x in cur if x != p]
                if next_list == cur:
                    continue
                dirty = True
                if next_list:
                    self._files[key] = next_list
                else:
                    self._files.pop(key, None)
            cur = self._files.get(cwd_key)
            if isinstance(cur, list) and cur:
                out = list(cur)
        if dirty:
            self._save_files()
        return list(out)

    def files_remove_all(self, path: str) -> int:
        p = str(path).strip()
        if not p:
            return 0
        dirty = False
        removed = 0
        with self._lock:
            for key, cur in list(self._files.items()):
                if not isinstance(cur, list) or not cur:
                    continue
                next_list = [x for x in cur if x != p]
                if next_list == cur:
                    continue
                removed += len(cur) - len(next_list)
                dirty = True
                if next_list:
                    self._files[key] = next_list
                else:
                    self._files.pop(key, None)
        if dirty:
            self._save_files()
        return removed

    def files_clear_scope(self, session_id: str) -> None:
        dirty = False
        with self._lock:
            key, legacy_keys, _s = self._files_key_for_session(session_id)
            for lk in legacy_keys:
                if lk in self._files:
                    self._files.pop(lk, None)
                    dirty = True
            if key in self._files:
                self._files.pop(key, None)
                dirty = True
        if dirty:
            self._save_files()

    def files_clear_cwd(self, cwd_raw: str) -> None:
        cwd_key = self._cwd_key_from_value(cwd_raw)
        if not cwd_key:
            return
        dirty = False
        with self._lock:
            keys = {cwd_key}
            for s in self._sessions.values():
                try:
                    key, legacy_keys, _sref = self._files_key_for_session(s.session_id)
                except KeyError:
                    continue
                if key != cwd_key:
                    continue
                for lk in legacy_keys:
                    keys.add(lk)
            for key in keys:
                if key in self._files:
                    self._files.pop(key, None)
                    dirty = True
        if dirty:
            self._save_files()

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
            with self._lock:
                scope_last = float(self._harness_last_injected_scope.get(scope_key, 0.0))
            if scope_last and (now - scope_last) < float(HARNESS_IDLE_SECONDS):
                continue
            self.send(sid, prompt)
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
            cli_raw = meta.get("cli") if isinstance(meta.get("cli"), str) else ""
            cli = _normalize_cli_name(cli_raw, default=DEFAULT_SPAWN_CLI)
            tmux_raw = meta.get("tmux_name")
            tmux_name = tmux_raw.strip() if isinstance(tmux_raw, str) and tmux_raw.strip() else None

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
            if log_path is not None and not cli_raw:
                inferred = _infer_cli_from_log_path(log_path)
                if isinstance(inferred, str):
                    cli = _normalize_cli_name(inferred, default=cli)
            if log_path is not None and log_path.exists():
                if cli == "codex":
                    thread_id, log_path = _coerce_main_thread_log(thread_id=thread_id, log_path=log_path)
            else:
                log_path = None

            if (log_path is None) and (not _pid_alive(codex_pid)) and (not _pid_alive(broker_pid)):
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
            prev: Session | None
            with self._lock:
                prev = self._sessions.get(session_id)
            last_assistant_ts: float | None = None
            if log_path is not None and log_path.exists():
                if prev and prev.log_path == log_path and prev.last_assistant_ts is not None:
                    last_assistant_ts = prev.last_assistant_ts
                else:
                    last_assistant_ts = _last_assistant_ts_from_tail(log_path, max_scan_bytes=CHAT_INIT_MAX_SCAN_BYTES)

            s = Session(
                session_id=session_id,
                thread_id=thread_id,
                broker_pid=broker_pid,
                codex_pid=codex_pid,
                cli=cli,
                owned=owned,
                start_ts=float(start_ts),
                cwd=str(cwd),
                log_path=log_path,
                sock_path=sock,
                tmux_name=tmux_name,
                busy=bool(resp.get("busy")),
                queue_len=int(resp.get("queue_len")),
                token=(resp.get("token") if isinstance(resp.get("token"), (dict, type(None))) else None),
                last_assistant_ts=last_assistant_ts,
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
                    prev.cli = s.cli
                    prev.owned = s.owned
                    prev.start_ts = s.start_ts
                    prev.cwd = s.cwd
                    prev.busy = s.busy
                    prev.queue_len = s.queue_len
                    prev.token = s.token
                    prev.tmux_name = s.tmux_name
                    if prev.log_path != s.log_path:
                        prev.log_path = s.log_path
                        self._reset_log_caches(prev, meta_log_off=meta_log_off)
                    if s.last_assistant_ts is not None:
                        prev.last_assistant_ts = s.last_assistant_ts
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
            sz = int(lp.stat().st_size)
            off = int(s.meta_log_off)
            if sz < off:
                off = 0

            total_th = 0
            total_tools = 0
            total_sys = 0
            latest_chat_ts: float | None = None
            latest_assistant_ts: float | None = None
            latest_token: dict[str, Any] | None = None
            loops = 0
            while off < sz and loops < 16:
                objs, new_off = _read_jsonl_from_offset(lp, off, max_bytes=256 * 1024)
                if new_off <= off:
                    break
                d_th, d_tools, d_sys, chunk_chat_ts, chunk_assistant_ts, token_update, _chat_events = _analyze_log_chunk(objs)
                total_th += d_th
                total_tools += d_tools
                total_sys += d_sys
                if chunk_chat_ts is not None:
                    latest_chat_ts = chunk_chat_ts if latest_chat_ts is None else max(latest_chat_ts, chunk_chat_ts)
                if chunk_assistant_ts is not None:
                    latest_assistant_ts = (
                        chunk_assistant_ts if latest_assistant_ts is None else max(latest_assistant_ts, chunk_assistant_ts)
                    )
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
                if latest_assistant_ts is not None:
                    s2.last_assistant_ts = (
                        latest_assistant_ts if s2.last_assistant_ts is None else max(s2.last_assistant_ts, latest_assistant_ts)
                    )
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
        with self._lock:
            items: list[dict[str, Any]] = []
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
                    s.last_chat_ts = float(s.log_path.stat().st_mtime)
                updated_ts = float(s.last_chat_ts) if isinstance(s.last_chat_ts, (int, float)) else float(s.start_ts)
                items.append(
                    {
                        "session_id": s.session_id,
                        "thread_id": s.thread_id,
                        "pid": s.codex_pid,
                        "broker_pid": s.broker_pid,
                        "cli": s.cli,
                        "owned": s.owned,
                        "cwd": s.cwd,
                        "start_ts": s.start_ts,
                        "updated_ts": updated_ts,
                        "log_path": (str(s.log_path) if s.log_path is not None else None),
                        "log_exists": log_exists,
                        "state_busy": bool(s.busy),
                        "queue_len": int(s.queue_len),
                        "token": s.token,
                        "thinking": int(s.meta_thinking),
                        "tools": int(s.meta_tools),
                        "system": int(s.meta_system),
                        "last_assistant_ts": (float(s.last_assistant_ts) if isinstance(s.last_assistant_ts, (int, float)) else None),
                        "harness_enabled": h_enabled,
                        "alias": alias,
                        "files": list(files),
                        "tmux_name": s.tmux_name if isinstance(getattr(s, "tmux_name", None), str) else None,
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
                log_path_raw = it.get("log_path")
                lp = Path(log_path_raw) if isinstance(log_path_raw, str) and log_path_raw else None
                idle_val = bool(self.idle_from_log(sid))
                busy_out = _busy_from_state_and_log_idle(
                    state_busy=state_busy,
                    idle_from_log=idle_val,
                    log_path=lp,
                )
            it2 = dict(it)
            it2.pop("log_exists", None)
            it2.pop("state_busy", None)
            it2["busy"] = bool(busy_out)
            out.append(it2)
        if files_dirty:
            self._save_files()
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
        cli_raw = meta.get("cli") if isinstance(meta.get("cli"), str) else ""
        cli = _normalize_cli_name(cli_raw, default=s.cli)
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
        if log_path is not None and not cli_raw:
            inferred = _infer_cli_from_log_path(log_path)
            if isinstance(inferred, str):
                cli = _normalize_cli_name(inferred, default=cli)
        if log_path is not None and log_path.exists():
            if cli == "codex":
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
            s2.cli = cli
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
            # Deduplicate to avoid replaying overlap between repeated tail snapshots.
            tail = list(events[-CHAT_INDEX_MAX_EVENTS:])
            uniq_rev: list[dict[str, Any]] = []
            seen: set[tuple[str, int, str]] = set()
            for ev in reversed(tail):
                k = event_key(ev)
                if k is not None and k in seen:
                    continue
                if k is not None:
                    seen.add(k)
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
                seen: set[tuple[str, int, str]] = set()
                for ev in recent:
                    k = event_key(ev)
                    if k is not None:
                        seen.add(k)
                appended: list[dict[str, Any]] = []
                for ev in new_events:
                    k = event_key(ev)
                    if k is not None and k in seen:
                        continue
                    if k is not None:
                        seen.add(k)
                    merged.append(ev)
                    appended.append(ev)
                if len(merged) > CHAT_INDEX_MAX_EVENTS:
                    merged = merged[-CHAT_INDEX_MAX_EVENTS:]
                    s.chat_index_scan_complete = False
                s.chat_index_events = merged
                for ev in appended:
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
                    _th, _tools, _sys, _last_ts, _last_assistant_ts, token_update, new_events = _analyze_log_chunk(objs)
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
        _th, _tools, _sys, _last_ts, _last_assistant_ts, token_update, new_events = _analyze_log_chunk(objs)
        self._append_chat_events(session_id, new_events, new_off=new_off, latest_token=token_update)
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
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
        cli: str | None = None,
    ) -> dict[str, Any]:
        cli_name = _parse_cli_name(cli, default=DEFAULT_SPAWN_CLI)
        cli_args = list(args) if isinstance(args, list) else []
        if cli_name == "claude" and (not _claude_args_override_session(cli_args)):
            # Keep "new web session" semantics stable even when another Claude
            # session already exists in the same workspace.
            cli_args.extend(["--session-id", str(uuid.uuid4())])
        argv = [sys.executable, "-m", "codoxear.broker", "--cwd", cwd, "--"]
        if cli_args:
            argv.extend(cli_args)

        env = dict(os.environ)
        if _DOTENV.exists():
            for k, v in _load_env_file(_DOTENV).items():
                env.setdefault(k, v)
        env["CODEX_WEB_OWNER"] = "web"
        env["CODEX_WEB_CLI"] = cli_name
        env.pop("CODEX_WEB_UNSET_ANTHROPIC_AUTH_TOKEN", None)
        use_tmux = _env_flag("CODEX_WEB_TMUX", True)
        tmux_bin = shutil.which("tmux") if use_tmux else None
        child_env_unset: list[str] = []
        if cli_name == "claude":
            env.setdefault("CLAUDE_HOME", str(_cli_home("claude")))
            env.setdefault("CLAUDE_BIN", _cli_bin("claude"))
            prefer_api_key_raw = env.get("CODEX_WEB_CLAUDE_PREFER_API_KEY")
            prefer_api_key = (
                True
                if prefer_api_key_raw is None
                else str(prefer_api_key_raw).strip().lower() not in ("0", "false", "no", "off")
            )
            api_key = env.get("ANTHROPIC_API_KEY")
            auth_token = env.get("ANTHROPIC_AUTH_TOKEN")
            has_api_key = isinstance(api_key, str) and bool(api_key.strip())
            has_auth_token = isinstance(auth_token, str) and bool(auth_token.strip())
            if use_tmux and tmux_bin:
                if not has_api_key:
                    has_api_key = _tmux_global_env_has_nonempty(tmux_bin, "ANTHROPIC_API_KEY", env)
                if not has_auth_token:
                    has_auth_token = _tmux_global_env_has_nonempty(tmux_bin, "ANTHROPIC_AUTH_TOKEN", env)
            # Claude CLI can stall at auth prompts when both auth modes are set.
            # Default behavior prefers API key for headless web-owned sessions.
            if prefer_api_key and has_api_key and has_auth_token:
                env.pop("ANTHROPIC_AUTH_TOKEN", None)
                if "ANTHROPIC_AUTH_TOKEN" not in child_env_unset:
                    child_env_unset.append("ANTHROPIC_AUTH_TOKEN")
                env["CODEX_WEB_UNSET_ANTHROPIC_AUTH_TOKEN"] = "1"
        elif cli_name == "gemini":
            env.setdefault("GEMINI_HOME", str(_cli_home("gemini")))
            env.setdefault("GEMINI_BIN", _cli_bin("gemini"))
        else:
            env.setdefault("CODEX_HOME", str(_cli_home("codex")))
            env.setdefault("CODEX_BIN", _cli_bin("codex"))

        if use_tmux and tmux_bin:
            tmux_name = f"codoxear-web-{uuid.uuid4().hex[:8]}"
            env.setdefault("CODEX_WEB_TMUX_INTERACTIVE", "1")
            env["CODEX_WEB_TMUX_NAME"] = tmux_name
            env_args = [f"{k}={v}" for k, v in env.items() if isinstance(k, str) and v is not None]
            env_unset_args: list[str] = []
            for key in child_env_unset:
                if isinstance(key, str) and key:
                    env_unset_args.extend(["-u", key])
            tmux_cmd = [tmux_bin, "new-session", "-d", "-s", tmux_name, "--", "env", *env_unset_args, *env_args, *argv]
            try:
                proc = subprocess.run(
                    tmux_cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=False,
                    text=True,
                )
            except Exception as e:
                raise RuntimeError(f"tmux spawn failed: {e}") from e
            if proc.returncode != 0:
                msg = (proc.stderr or "").strip()
                msg = msg[-4000:] if msg else ""
                raise RuntimeError(f"tmux spawn failed (rc={proc.returncode}): {msg}")
            broker_pid = _tmux_pane_pid(tmux_bin, tmux_name, env) or 0
            return {"broker_pid": int(broker_pid), "tmux_name": tmux_name, "cli": cli_name}
        if use_tmux and not tmux_bin:
            sys.stderr.write("warning: CODEX_WEB_TMUX enabled but tmux not found; falling back to direct broker spawn.\n")
            sys.stderr.flush()

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
        return {"broker_pid": int(proc.pid), "cli": cli_name}

    def delete_web_session(self, session_id: str) -> bool:
        with self._lock:
            s = self._sessions.get(session_id)
        if not s:
            return False
        if not s.owned:
            raise PermissionError("not owned by web")
        ok = self.kill_session(session_id)
        if ok:
            self.alias_clear(session_id)
            self.files_clear(session_id)
        return ok


    def send(self, session_id: str, text: str) -> dict[str, Any]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            sock = s.sock_path
            cli = _normalize_cli_name(s.cli, default="codex")
        text_out = _normalize_outgoing_text_for_cli(text, cli)
        try:
            resp = self._sock_call(sock, {"cmd": "send", "text": text_out}, timeout_s=3.0)
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
                if "queue_len" not in resp:
                    raise ValueError("invalid broker send response")
                s2.queue_len = int(resp.get("queue_len"))
        return resp

    def _queue_call(self, session_id: str, req: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            sock = s.sock_path
        try:
            resp = self._sock_call(sock, req, timeout_s=2.5)
        except Exception:
            if not _pid_alive(s.broker_pid) and not _pid_alive(s.codex_pid):
                with self._lock:
                    self._sessions.pop(session_id, None)
                _unlink_quiet(sock)
                _unlink_quiet(sock.with_suffix(".json"))
                raise KeyError("unknown session")
            raise
        q_raw = resp.get("queue")
        if not isinstance(q_raw, list):
            raise ValueError("invalid broker queue response")
        q = _normalize_queue_list(q_raw)
        qlen = len(q)
        with self._lock:
            s2 = self._sessions.get(session_id)
            if s2:
                s2.queue_len = int(qlen)
        return {"queue": q, "queue_len": int(qlen)}

    def queue_get(self, session_id: str) -> dict[str, Any]:
        return self._queue_call(session_id, {"cmd": "queue", "op": "get"})

    def queue_set(self, session_id: str, queue: list[str]) -> dict[str, Any]:
        cleaned = _normalize_queue_list(queue)
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            cli = _normalize_cli_name(s.cli, default="codex")
        cleaned = [_normalize_outgoing_text_for_cli(item, cli) for item in cleaned]
        return self._queue_call(session_id, {"cmd": "queue", "op": "set", "queue": cleaned})

    def queue_push(self, session_id: str, text: str, *, front: bool = False) -> dict[str, Any]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            cli = _normalize_cli_name(s.cli, default="codex")
        text_out = _normalize_outgoing_text_for_cli(text, cli)
        return self._queue_call(session_id, {"cmd": "queue", "op": "push", "text": text_out, "front": bool(front)})

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
            cli = _normalize_cli_name(getattr(s, "cli", ""), default="codex")
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
        cleaned = _sanitize_tail_text(tail)
        if cli == "claude":
            return _sanitize_claude_tail_text(cleaned)
        return cleaned

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

            if path == "/api/metrics":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                _json_response(self, 200, {"metrics": _metrics_snapshot()})
                return

            if path == "/api/update":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                qs = urllib.parse.parse_qs(u.query)
                force_raw = qs.get("force")
                force = bool(force_raw and force_raw[0] == "1")
                _json_response(self, 200, _update_status(force=force))
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
                    queue_val = int(state.get("queue_len"))
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

                busy_val = _busy_from_state_and_log_idle(
                    state_busy=bool(state_busy),
                    idle_from_log=bool(idle_val),
                    log_path=s.log_path,
                )
                queue_val = state_queue

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

            if path.startswith("/api/sessions/") and path.endswith("/queue"):
                if not _require_auth(self):
                    self._unauthorized()
                    return
                parts = path.split("/")
                session_id = parts[3] if len(parts) >= 4 else ""
                try:
                    resp = MANAGER.queue_get(session_id)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, resp)
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

            if path == "/api/config":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                config = _read_cli_config()
                _json_response(self, 200, {"ok": True, "config": config})
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
                args = obj.get("args")
                if args is None:
                    args_list = None
                elif isinstance(args, list) and all(isinstance(x, str) for x in args):
                    args_list = [x for x in args if x]
                else:
                    _json_response(self, 400, {"error": "args must be a list of strings"})
                    return
                cli_raw = obj.get("cli")
                if cli_raw is None:
                    cli = DEFAULT_SPAWN_CLI
                elif isinstance(cli_raw, str):
                    try:
                        cli = _parse_cli_name(cli_raw, default=DEFAULT_SPAWN_CLI)
                    except ValueError:
                        _json_response(self, 400, {"error": "unsupported cli (use codex, claude, or gemini)"})
                        return
                else:
                    _json_response(self, 400, {"error": "cli must be a string"})
                    return
                res = MANAGER.spawn_web_session(cwd=cwd, args=args_list, cli=cli)
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
                record_history = obj.get("record_history")
                record_history = True if record_history is None else bool(record_history)
                path_obj = Path(path_raw).expanduser()
                if not path_obj.is_absolute():
                    path_obj = (Path.cwd() / path_obj).resolve()
                else:
                    path_obj = path_obj.resolve()
                if not path_obj.exists():
                    _json_response(self, 404, {"error": "file not found"})
                    return
                if not path_obj.is_file():
                    _json_response(self, 400, {"error": "path is not a file"})
                    return
                try:
                    text, size = _read_text_file_strict(path_obj, max_bytes=FILE_READ_MAX_BYTES)
                except PermissionError:
                    _json_response(self, 403, {"error": "permission denied"})
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                if session_id and record_history:
                    try:
                        MANAGER.files_add(session_id, str(path_obj))
                    except KeyError:
                        pass
                _json_response(self, 200, {"ok": True, "path": str(path_obj), "size": int(size), "text": text})
                return

            if path == "/api/files/write":
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
                text = obj.get("text")
                if not isinstance(text, str):
                    _json_response(self, 400, {"error": "text required"})
                    return
                session_id_raw = obj.get("session_id")
                session_id = session_id_raw if isinstance(session_id_raw, str) and session_id_raw else ""
                path_obj = Path(path_raw).expanduser()
                if not path_obj.is_absolute():
                    path_obj = (Path.cwd() / path_obj).resolve()
                else:
                    path_obj = path_obj.resolve()
                if not path_obj.exists():
                    _json_response(self, 404, {"error": "file not found"})
                    return
                if not path_obj.is_file():
                    _json_response(self, 400, {"error": "path is not a file"})
                    return
                data = text.encode("utf-8")
                if len(data) > FILE_WRITE_MAX_BYTES:
                    _json_response(self, 400, {"error": f"file too large (max {FILE_WRITE_MAX_BYTES} bytes)"})
                    return
                tmp = path_obj.with_name(path_obj.name + ".tmp")
                try:
                    tmp.write_bytes(data)
                    os.replace(tmp, path_obj)
                except PermissionError:
                    _json_response(self, 403, {"error": "permission denied"})
                    return
                except Exception as e:
                    _json_response(self, 500, {"error": f"write failed: {e}"})
                    return
                if session_id:
                    try:
                        MANAGER.files_add(session_id, str(path_obj))
                    except KeyError:
                        pass
                _json_response(self, 200, {"ok": True, "path": str(path_obj), "size": int(len(data))})
                return

            if path == "/api/files/remove":
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
                cwd_raw = obj.get("cwd")
                cwd = cwd_raw if isinstance(cwd_raw, str) and cwd_raw.strip() else ""
                scope_raw = obj.get("scope")
                scope = scope_raw if isinstance(scope_raw, str) else ""
                session_id_raw = obj.get("session_id")
                session_id = session_id_raw if isinstance(session_id_raw, str) and session_id_raw else ""
                if scope and scope not in ("cwd", "session", "all"):
                    _json_response(self, 400, {"error": "invalid scope"})
                    return
                if scope == "all":
                    removed = MANAGER.files_remove_all(path_raw)
                    _json_response(self, 200, {"ok": True, "removed": int(removed), "files": []})
                    return
                if cwd:
                    files = MANAGER.files_remove_cwd(cwd, path_raw)
                    _json_response(self, 200, {"ok": True, "files": list(files)})
                    return
                if not session_id:
                    _json_response(self, 400, {"error": "session_id required"})
                    return
                try:
                    files = MANAGER.files_remove(session_id, path_raw)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, {"ok": True, "files": list(files)})
                return

            if path == "/api/files/clear":
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
                cwd_raw = obj.get("cwd")
                cwd = cwd_raw if isinstance(cwd_raw, str) and cwd_raw.strip() else ""
                session_id_raw = obj.get("session_id")
                session_id = session_id_raw if isinstance(session_id_raw, str) and session_id_raw else ""
                if cwd:
                    MANAGER.files_clear_cwd(cwd)
                    _json_response(self, 200, {"ok": True, "files": []})
                    return
                if not session_id:
                    _json_response(self, 400, {"error": "session_id required"})
                    return
                try:
                    MANAGER.files_clear_scope(session_id)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, {"ok": True, "files": []})
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

            if path.startswith("/api/sessions/") and path.endswith("/queue"):
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
                if "queue" in obj and "text" in obj:
                    _json_response(self, 400, {"error": "use queue or text, not both"})
                    return
                if "queue" in obj:
                    q_raw = obj.get("queue")
                    if not isinstance(q_raw, list):
                        _json_response(self, 400, {"error": "queue must be a list"})
                        return
                    try:
                        resp = MANAGER.queue_set(session_id, _normalize_queue_list(q_raw))
                    except KeyError:
                        _json_response(self, 404, {"error": "unknown session"})
                        return
                    _json_response(self, 200, resp)
                    return
                text = obj.get("text")
                if not isinstance(text, str) or not text.strip():
                    _json_response(self, 400, {"error": "text required"})
                    return
                front = bool(obj.get("front"))
                try:
                    resp = MANAGER.queue_push(session_id, text, front=front)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, resp)
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

            if path.startswith("/api/sessions/") and path.endswith("/inject_image"):
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

                stem = Path(_safe_filename(filename)).stem
                if not stem:
                    raise ValueError("invalid filename")
                safe = stem + sniffed

                subdir = UPLOAD_DIR / session_id
                subdir.mkdir(parents=True, exist_ok=True)
                ts = int(_now() * 1000)
                out_path = (subdir / f"{ts}_{safe}").resolve()
                if not str(out_path).startswith(str(subdir.resolve())):
                    _json_response(self, 400, {"error": "bad path"})
                    return
                out_path.write_bytes(raw)
                os.chmod(out_path, 0o600)

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

            if path == "/api/config":
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
                updates = obj.get("updates")
                if not isinstance(updates, dict):
                    _json_response(self, 400, {"error": "updates must be an object"})
                    return
                result = _save_cli_config(updates)
                _json_response(self, 200, result)
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
