#!/usr/bin/env python3
from __future__ import annotations

import base64
import copy
import errno
import fnmatch
import gzip
import hashlib
import heapq
import hmac
import http.server
import io
import json
import logging
import math
import mimetypes
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time
import tomllib
import traceback
import urllib.parse
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import pi_messages as _pi_messages
from . import rollout_log as _rollout_log
from .agent_backend import (
    get_agent_backend,
    infer_agent_backend_from_log_path,
    normalize_agent_backend,
)
from .events.hub import EventHub
from .http.routes import assets as _http_assets_routes
from .http.routes import auth as _http_auth_routes
from .http.routes import events as _http_events_routes
from .http.routes import files as _http_file_routes
from .http.routes import notifications as _http_notification_routes
from .http.routes import sessions_read as _http_session_read_routes
from .http.routes import sessions_write as _http_session_write_routes
from .page_state_sqlite import (
    DurableSessionRecord,
    PageStateDB,
    SessionRef,
    import_legacy_app_dir_to_db,
)
from .pi import ui_bridge as _pi_ui_bridge
from .pi_log import pi_model_context_window as _pi_model_context_window_impl
from .pi_log import pi_user_text as _pi_user_text
from .pi_log import read_pi_run_settings as _read_pi_run_settings
from .pi_log import read_pi_session_id as _read_pi_session_id
from .sessions import live_payloads as _session_live_payloads
from .sessions import payloads as _session_payloads
from .sessions import sidebar_state as _sidebar_state_module
from .sessions.sidebar_state import SidebarStateFacade
from .util import default_app_dir as _default_app_dir
from .util import find_new_session_log as _find_new_session_log_impl
from .util import (
    find_session_log_for_session_id as _find_session_log_for_session_id_impl,
)
from .util import is_subagent_session_meta as _is_subagent_session_meta
from .util import iter_session_logs as _iter_session_logs_impl
from .util import now as _now
from .util import proc_find_open_rollout_log as _proc_find_open_rollout_log
from .util import read_jsonl_from_offset as _read_jsonl_from_offset_impl
from .util import read_session_meta_payload as _read_session_meta_payload_impl
from .util import subagent_parent_thread_id as _subagent_parent_thread_id
from .voice_push import VoicePushCoordinator

SessionStateKey = SessionRef | str

for _seam_module in (
    _http_assets_routes,
    _http_auth_routes,
    _http_events_routes,
    _http_file_routes,
    _http_notification_routes,
    _http_session_read_routes,
    _http_session_write_routes,
    _session_live_payloads,
    _session_payloads,
    _sidebar_state_module,
    _pi_ui_bridge,
):
    _seam_module.bind_server_runtime(sys.modules[__name__])


LOG = logging.getLogger(__name__)
EVENT_HUB = EventHub(max_events=1024)


def _publish_invalidate_event(
    event_type: str,
    *,
    session_id: str | None = None,
    runtime_id: str | None = None,
    reason: str,
    hints: dict[str, Any] | None = None,
    coalesce_ms: int = 300,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {
        "type": event_type,
        "reason": str(reason).strip() or "update",
        "_coalesce_ms": int(coalesce_ms),
        "_coalesce_key": (str(event_type), str(session_id or "")),
    }
    clean_session_id = _clean_optional_text(session_id)
    clean_runtime_id = _clean_optional_text(runtime_id)
    if clean_session_id is not None:
        payload["session_id"] = clean_session_id
    if clean_runtime_id is not None:
        payload["runtime_id"] = clean_runtime_id
    if isinstance(hints, dict) and hints:
        payload["hints"] = dict(hints)
    return EVENT_HUB.publish(payload)



def _publish_sessions_invalidate(*, reason: str, coalesce_ms: int = 500) -> dict[str, Any] | None:
    return _publish_invalidate_event(
        "sessions.invalidate",
        reason=reason,
        coalesce_ms=coalesce_ms,
    )



def _publish_session_live_invalidate(
    session_id: str,
    *,
    runtime_id: str | None = None,
    reason: str,
    hints: dict[str, Any] | None = None,
    coalesce_ms: int = 300,
) -> dict[str, Any] | None:
    return _publish_invalidate_event(
        "session.live.invalidate",
        session_id=session_id,
        runtime_id=runtime_id,
        reason=reason,
        hints=hints,
        coalesce_ms=coalesce_ms,
    )



def _publish_session_workspace_invalidate(
    session_id: str,
    *,
    runtime_id: str | None = None,
    reason: str,
    coalesce_ms: int = 300,
) -> dict[str, Any] | None:
    return _publish_invalidate_event(
        "session.workspace.invalidate",
        session_id=session_id,
        runtime_id=runtime_id,
        reason=reason,
        coalesce_ms=coalesce_ms,
    )



def _publish_session_transport_invalidate(
    session_id: str,
    *,
    runtime_id: str | None = None,
    reason: str,
    coalesce_ms: int = 300,
) -> dict[str, Any] | None:
    return _publish_invalidate_event(
        "session.transport.invalidate",
        session_id=session_id,
        runtime_id=runtime_id,
        reason=reason,
        coalesce_ms=coalesce_ms,
    )



def _publish_notifications_invalidate(*, reason: str, coalesce_ms: int = 500) -> dict[str, Any] | None:
    return _publish_invalidate_event(
        "notifications.invalidate",
        reason=reason,
        coalesce_ms=coalesce_ms,
    )



def _publish_attention_invalidate(
    *,
    reason: str,
    session_id: str | None = None,
    coalesce_ms: int = 500,
) -> dict[str, Any] | None:
    return _publish_invalidate_event(
        "attention.invalidate",
        session_id=session_id,
        reason=reason,
        coalesce_ms=coalesce_ms,
    )



def _voice_push_publish_callback(event: dict[str, Any]) -> None:
    _publish_invalidate_event(
        str(event.get("type") or "notifications.invalidate"),
        session_id=_clean_optional_text(event.get("session_id")),
        runtime_id=_clean_optional_text(event.get("runtime_id")),
        reason=str(event.get("reason") or "update"),
        hints=event.get("hints") if isinstance(event.get("hints"), dict) else None,
        coalesce_ms=int(event.get("coalesce_ms") or 500),
    )



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


def _match_session_route(path: str, *suffix: str) -> str | None:
    parts = path.split("/")
    if len(parts) != 4 + len(suffix):
        return None
    if parts[:3] != ["", "api", "sessions"]:
        return None
    session_id = urllib.parse.unquote(parts[3])
    if not session_id:
        return None
    if tuple(parts[4:]) != tuple(suffix):
        return None
    return session_id


def _strip_url_prefix(prefix: str, path: str) -> str | None:
    if not prefix:
        return path
    if path == prefix:
        return "/"
    if path.startswith(prefix + "/"):
        return path[len(prefix) :]
    return None


ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
WEB_DIST_DIR = WEB_DIR / "dist"
LEGACY_STATIC_DIR = ROOT_DIR / "codoxear" / "static"
PACKAGED_WEB_DIST_DIR = LEGACY_STATIC_DIR / "dist"

APP_DIR = _default_app_dir()
STATIC_DIR = LEGACY_STATIC_DIR
STATIC_ASSET_VERSION_PLACEHOLDER = "__CODOXEAR_ASSET_VERSION__"
STATIC_ATTACH_MAX_BYTES_PLACEHOLDER = "__CODOXEAR_ATTACH_MAX_BYTES__"
STATIC_ASSET_VERSION_FILES = ("app.js", "app.css")
SOCK_DIR = APP_DIR / "socks"
PROC_ROOT = Path("/proc")
STATE_PATH = APP_DIR / "state.json"
HMAC_SECRET_PATH = APP_DIR / "hmac_secret"
UPLOAD_DIR = APP_DIR / "uploads"
HARNESS_PATH = APP_DIR / "harness.json"
ALIAS_PATH = APP_DIR / "session_aliases.json"
SIDEBAR_META_PATH = APP_DIR / "session_sidebar.json"
HIDDEN_SESSIONS_PATH = APP_DIR / "hidden_sessions.json"
FILE_HISTORY_PATH = APP_DIR / "session_files.json"
QUEUE_PATH = APP_DIR / "session_queues.json"
RECENT_CWD_PATH = APP_DIR / "recent_cwds.json"
CWD_GROUPS_PATH = APP_DIR / "cwd_groups.json"
PAGE_STATE_DB_PATH = APP_DIR / "sqlite.db"
VOICE_SETTINGS_PATH = APP_DIR / "voice_settings.json"
PUSH_SUBSCRIPTIONS_PATH = APP_DIR / "push_subscriptions.json"
DELIVERY_LEDGER_PATH = APP_DIR / "voice_delivery_ledger.json"
VAPID_PRIVATE_KEY_PATH = APP_DIR / "webpush_vapid_private.pem"

_DOTENV = (Path.cwd() / ".env").resolve()
if _DOTENV.exists():
    for _k, _v in _load_env_file(_DOTENV).items():
        os.environ.setdefault(_k, _v)

COOKIE_NAME = "codoxear_auth"
COOKIE_TTL_SECONDS = int(
    os.environ.get("CODEX_WEB_COOKIE_TTL_SECONDS", str(30 * 24 * 3600))
)
COOKIE_SECURE = os.environ.get("CODEX_WEB_COOKIE_SECURE", "0") == "1"
URL_PREFIX = _normalize_url_prefix(os.environ.get("CODEX_WEB_URL_PREFIX"))
COOKIE_PATH = (URL_PREFIX + "/") if URL_PREFIX else "/"
TMUX_SESSION_NAME = (
    os.environ.get("CODEX_WEB_TMUX_SESSION") or "codoxear"
).strip() or "codoxear"
TMUX_META_WAIT_SECONDS = 3.0
TMUX_SHORT_APP_DIR = Path("/tmp/codoxear")

_CODEX_HOME_ENV = os.environ.get("CODEX_HOME")
if _CODEX_HOME_ENV is None or (not _CODEX_HOME_ENV.strip()):
    CODEX_HOME = Path.home() / ".codex"
else:
    CODEX_HOME = Path(_CODEX_HOME_ENV)
CODEX_SESSIONS_DIR = CODEX_HOME / "sessions"
CODEX_CONFIG_PATH = CODEX_HOME / "config.toml"
PI_NATIVE_SESSIONS_DIR = Path.home() / ".pi" / "agent" / "sessions"
MODELS_CACHE_PATH = CODEX_HOME / "models_cache.json"
PI_HOME = get_agent_backend("pi").home()
PI_SESSIONS_DIR = get_agent_backend("pi").sessions_dir()
PI_SETTINGS_PATH = PI_HOME / "agent" / "settings.json"
PI_MODELS_PATH = PI_HOME / "agent" / "models.json"
PI_AUTH_PATH = PI_HOME / "agent" / "auth.json"
DEFAULT_AGENT_BACKEND = normalize_agent_backend(
    os.environ.get("CODEX_WEB_DEFAULT_AGENT_BACKEND"), default="pi"
)
SUPPORTED_REASONING_EFFORTS = ("xhigh", "high", "medium", "low")
SUPPORTED_PI_REASONING_EFFORTS = ("off", "minimal", "low", "medium", "high", "xhigh")
PI_COMMANDS_CACHE_TTL_SECONDS = float(
    os.environ.get("CODEX_WEB_PI_COMMANDS_CACHE_TTL_SECONDS", "45.0")
)

DEFAULT_HOST = os.environ.get("CODEX_WEB_HOST", "::")
DEFAULT_PORT = int(os.environ.get("CODEX_WEB_PORT", "8743"))
USE_LEGACY_WEB = os.environ.get("CODOXEAR_USE_LEGACY_WEB", "0") == "1"
HARNESS_DEFAULT_IDLE_MINUTES = 5
HARNESS_DEFAULT_MAX_INJECTIONS = 10
HARNESS_SWEEP_SECONDS = float(os.environ.get("CODEX_WEB_HARNESS_SWEEP_SECONDS", "2.5"))
QUEUE_SWEEP_SECONDS = float(os.environ.get("CODEX_WEB_QUEUE_SWEEP_SECONDS", "1.0"))
VOICE_PUSH_SWEEP_SECONDS = float(
    os.environ.get("CODEX_WEB_VOICE_PUSH_SWEEP_SECONDS", "1.0")
)
QUEUE_IDLE_GRACE_SECONDS = float(
    os.environ.get("CODEX_WEB_QUEUE_IDLE_GRACE_SECONDS", "10.0")
)
BRIDGE_TRANSPORT_PROBE_STALE_SECONDS = float(
    os.environ.get("CODEX_WEB_BRIDGE_TRANSPORT_PROBE_STALE_SECONDS", "2.0")
)
BRIDGE_TRANSPORT_RPC_TIMEOUT_SECONDS = float(
    os.environ.get("CODEX_WEB_BRIDGE_TRANSPORT_RPC_TIMEOUT_SECONDS", "0.35")
)
BRIDGE_OUTBOUND_FAILURE_MAX_ATTEMPTS = int(
    os.environ.get("CODEX_WEB_BRIDGE_OUTBOUND_FAILURE_MAX_ATTEMPTS", "3")
)
BRIDGE_OUTBOUND_FAILURE_MAX_AGE_SECONDS = float(
    os.environ.get("CODEX_WEB_BRIDGE_OUTBOUND_FAILURE_MAX_AGE_SECONDS", "8.0")
)
HARNESS_MAX_SCAN_BYTES = int(
    os.environ.get("CODEX_WEB_HARNESS_MAX_SCAN_BYTES", str(8 * 1024 * 1024))
)
DISCOVER_MIN_INTERVAL_SECONDS = float(
    os.environ.get("CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS", "60.0")
)
CHAT_INIT_SEED_SCAN_BYTES = int(
    os.environ.get("CODEX_WEB_CHAT_INIT_SEED_SCAN_BYTES", str(512 * 1024))
)
CHAT_INIT_MAX_SCAN_BYTES = int(
    os.environ.get("CODEX_WEB_CHAT_INIT_MAX_SCAN_BYTES", str(128 * 1024 * 1024))
)
CHAT_INDEX_INCREMENT_BYTES = int(
    os.environ.get("CODEX_WEB_CHAT_INDEX_INCREMENT_BYTES", str(2 * 1024 * 1024))
)
CHAT_INDEX_RESEED_THRESHOLD_BYTES = int(
    os.environ.get("CODEX_WEB_CHAT_INDEX_RESEED_THRESHOLD_BYTES", str(16 * 1024 * 1024))
)
CHAT_INDEX_MAX_EVENTS = int(os.environ.get("CODEX_WEB_CHAT_INDEX_MAX_EVENTS", "12000"))
METRICS_WINDOW = int(os.environ.get("CODEX_WEB_METRICS_WINDOW", "256"))
FILE_READ_MAX_BYTES = int(
    os.environ.get("CODEX_WEB_FILE_READ_MAX_BYTES", str(2 * 1024 * 1024))
)
FILE_HISTORY_MAX = int(os.environ.get("CODEX_WEB_FILE_HISTORY_MAX", "20"))
FILE_SEARCH_LIMIT = int(os.environ.get("CODEX_WEB_FILE_SEARCH_LIMIT", "120"))
FILE_SEARCH_TIMEOUT_SECONDS = float(
    os.environ.get("CODEX_WEB_FILE_SEARCH_TIMEOUT_SECONDS", "0.75")
)
FILE_SEARCH_MAX_CANDIDATES = int(
    os.environ.get("CODEX_WEB_FILE_SEARCH_MAX_CANDIDATES", "200000")
)
GIT_DIFF_MAX_BYTES = int(
    os.environ.get("CODEX_WEB_GIT_DIFF_MAX_BYTES", str(800 * 1024))
)
GIT_DIFF_TIMEOUT_SECONDS = float(
    os.environ.get("CODEX_WEB_GIT_DIFF_TIMEOUT_SECONDS", "4.0")
)
GIT_WORKTREE_TIMEOUT_SECONDS = float(
    os.environ.get("CODEX_WEB_GIT_WORKTREE_TIMEOUT_SECONDS", "10.0")
)
GIT_CHANGED_FILES_MAX = int(os.environ.get("CODEX_WEB_GIT_CHANGED_FILES_MAX", "400"))
ATTACH_UPLOAD_MAX_BYTES = int(
    os.environ.get("CODEX_WEB_ATTACH_MAX_BYTES", str(16 * 1024 * 1024))
)
ATTACH_UPLOAD_BODY_MAX_BYTES = int(
    os.environ.get(
        "CODEX_WEB_ATTACH_BODY_MAX_BYTES",
        str((4 * ((ATTACH_UPLOAD_MAX_BYTES + 2) // 3)) + (64 * 1024)),
    )
)
FILE_LIST_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".svn",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
        ".venv",
    }
)
MARKDOWN_EXTENSIONS = frozenset({"md", "markdown", "mdown", "mkd"})
TEXTUAL_EXTENSIONS = frozenset(
    {
        "bash",
        "c",
        "cc",
        "cfg",
        "conf",
        "cpp",
        "css",
        "csv",
        "diff",
        "go",
        "h",
        "hpp",
        "htm",
        "html",
        "ini",
        "java",
        "js",
        "json",
        "jsonl",
        "log",
        "md",
        "markdown",
        "mdown",
        "mkd",
        "patch",
        "py",
        "rs",
        "scss",
        "sh",
        "sql",
        "svg",
        "toml",
        "ts",
        "tsx",
        "txt",
        "xml",
        "yaml",
        "yml",
        "zsh",
    }
)
TEXTUAL_FILENAMES = frozenset({"dockerfile", "license", "makefile", "readme"})
SIDEBAR_PRIORITY_HALF_LIFE_SECONDS = 8.0 * 3600.0
SIDEBAR_PRIORITY_LAMBDA = math.log(2.0) / SIDEBAR_PRIORITY_HALF_LIFE_SECONDS
RECENT_CWD_MAX = int(os.environ.get("CODEX_WEB_RECENT_CWD_MAX", "256"))
HARNESS_PROMPT_PREFIX = """Unattended-mode instructions (optimize for 8+ hours, minimal turns, minimal repetition, maximal progress)

- Maintain four internal sections:
  1. Deliverables
     - The concrete outputs the agent owes the user by the end of the task.
     - Stable unless the user changes the request.
  2. Completed
     - Verified facts already established while producing the Deliverables.
  3. Next actions
     - Ordered concrete steps from the current state toward the Deliverables.
  4. Parked user decisions
     - Decisions or inputs that only the user can provide.

- Working rules:
  - Keep these sections internal. Surface them only when yielding is necessary.
  - Default to continuing in the same turn.
  - Before each action, reason until the approach, failure modes, and verification path are clear.
  - Exploration should happen through reading, tracing, inspection, and reasoning.
  - Avoid trial and error.
  - Resolve crashes, bugs, and design mistakes yourself unless a true user decision is required.
  - Use the strongest available verification.
  - Do not repeat the same command, edit, or analysis without a concrete new reason.

- Yield only when:
  - all Deliverables are finished and supported by Completed;
  - the only remaining gap is a Parked user decision;
  - or the next step is irreversible or high-risk and needs explicit user confirmation.

- End-of-turn gate (only when yielding is necessary):
  - Run a clean-room adversarial review via a dedicated subagent.
  - Give it: user intent, Deliverables, Completed, remaining Next actions, Parked user decisions, constraints, and changed artifacts.
  - Apply findings before yielding, or surface the exact remaining user decision or risk.
"""


def _render_harness_prompt(request: str | None) -> str:
    base = HARNESS_PROMPT_PREFIX.rstrip()
    r = (request or "").strip()
    if not r:
        return base + "\n"
    return base + "\n\n---\n\nAdditional request from user: " + r + "\n"


def _clean_harness_cooldown_minutes(raw: Any) -> int:
    if raw is None:
        return HARNESS_DEFAULT_IDLE_MINUTES
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError("harness cooldown_minutes must be an integer")
    if raw < 1:
        raise ValueError("harness cooldown_minutes must be at least 1")
    return int(raw)


def _clean_harness_remaining_injections(raw: Any, *, allow_zero: bool) -> int:
    if raw is None:
        return HARNESS_DEFAULT_MAX_INJECTIONS
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError("harness remaining_injections must be an integer")
    minimum = 0 if allow_zero else 1
    if raw < minimum:
        lower = "0" if allow_zero else "1"
        raise ValueError(f"harness remaining_injections must be at least {lower}")
    return int(raw)


_SESSION_ID_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I
)
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


def _wait_or_raise(
    proc: subprocess.Popen[bytes], *, label: str, timeout_s: float = 1.5
) -> None:
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


def _tmux_available() -> bool:
    return shutil.which("tmux") is not None


def _ensure_tmux_short_app_dir() -> str:
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        return str(TMUX_SHORT_APP_DIR)

    alias = TMUX_SHORT_APP_DIR
    try:
        if alias.is_symlink():
            if alias.resolve() == APP_DIR.resolve():
                return str(alias)
            alias.unlink()
        elif alias.exists():
            if alias.resolve() == APP_DIR.resolve():
                return str(alias)
            return str(alias)
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(APP_DIR, target_is_directory=True)
        return str(alias)
    except Exception:
        return str(alias)


def _wait_for_spawned_broker_meta(
    spawn_nonce: str, *, timeout_s: float = TMUX_META_WAIT_SECONDS
) -> dict[str, Any]:
    deadline = time.time() + max(timeout_s, 0.0)
    last_meta: dict[str, Any] | None = None
    while time.time() <= deadline:
        for meta_path in sorted(SOCK_DIR.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue
            if not isinstance(meta, dict):
                continue
            if _clean_optional_text(meta.get("spawn_nonce")) != spawn_nonce:
                continue
            broker_pid = meta.get("broker_pid")
            if not isinstance(broker_pid, int):
                continue
            last_meta = meta
            backend = normalize_agent_backend(meta.get("backend"), default="codex")
            session_id = _clean_optional_text(meta.get("session_id"))
            if backend == "pi" and session_id is None:
                continue
            return meta
        time.sleep(0.05)
    if last_meta is not None:
        return last_meta
    raise RuntimeError(
        f"tmux launch did not publish broker metadata within {timeout_s:.1f}s"
    )


def _spawn_result_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    broker_pid = meta.get("broker_pid")
    if not isinstance(broker_pid, int):
        raise RuntimeError("spawn metadata is missing broker_pid")
    sock_path = _clean_optional_text(meta.get("sock_path"))
    runtime_id = Path(sock_path).stem if sock_path else None
    session_id = _clean_optional_text(meta.get("session_id")) or runtime_id
    payload: dict[str, Any] = {"broker_pid": int(broker_pid)}
    if session_id:
        payload["session_id"] = session_id
    if runtime_id:
        payload["runtime_id"] = runtime_id
    backend = _clean_optional_text(meta.get("backend"))
    if backend:
        payload["backend"] = backend
    return payload


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


def _process_group_alive(root_pid: int) -> bool:
    if root_pid <= 0:
        return False
    try:
        os.killpg(root_pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_process_group(root_pid: int, *, wait_seconds: float = 1.0) -> bool:
    if not _process_group_alive(root_pid):
        return True
    try:
        os.killpg(root_pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    deadline = _now() + max(wait_seconds, 0.0)
    while _process_group_alive(root_pid):
        if _now() >= deadline:
            break
        time.sleep(0.05)
    if not _process_group_alive(root_pid):
        return True
    try:
        os.killpg(root_pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    deadline = _now() + 0.2
    while _process_group_alive(root_pid):
        if _now() >= deadline:
            break
        time.sleep(0.05)
    return not _process_group_alive(root_pid)


def _terminate_process(pid: int, *, wait_seconds: float = 1.0) -> bool:
    if not _pid_alive(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    deadline = _now() + max(wait_seconds, 0.0)
    while _pid_alive(pid):
        if _now() >= deadline:
            break
        time.sleep(0.05)
    if not _pid_alive(pid):
        return True
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    deadline = _now() + 0.2
    while _pid_alive(pid):
        if _now() >= deadline:
            break
        time.sleep(0.05)
    return not _pid_alive(pid)


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


def _probe_failure_safe_to_prune(*, broker_pid: int, codex_pid: int) -> bool:
    # Probe timeouts can be normal during startup; only prune runtime artifacts
    # once both tracked processes are gone.
    return (not _pid_alive(codex_pid)) and (not _pid_alive(broker_pid))


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


def _pdf_content_type(path: Path, raw: bytes) -> str | None:
    if path.suffix.lower() == ".pdf" or raw.startswith(b"%PDF-"):
        return "application/pdf"
    return None


def _file_kind(path: Path, raw: bytes) -> tuple[str, str | None]:
    ctype = _image_content_type(path, raw)
    if ctype is not None:
        return "image", ctype
    ctype = _pdf_content_type(path, raw)
    if ctype is not None:
        return "pdf", ctype
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


def _json_response(
    handler: http.server.BaseHTTPRequestHandler, status: int, obj: Any
) -> None:
    body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    accept_encoding = str(handler.headers.get("Accept-Encoding") or "").lower()
    use_gzip = "gzip" in accept_encoding
    if use_gzip:
        body = gzip.compress(body)
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    if use_gzip:
        handler.send_header("Content-Encoding", "gzip")
        handler.send_header("Vary", "Accept-Encoding")
    handler.send_header("Content-Length", str(len(body)))
    try:
        handler.end_headers()
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        # Client disconnected during transmission.
        pass


def _read_body(
    handler: http.server.BaseHTTPRequestHandler, limit: int = 2 * 1024 * 1024
) -> bytes:
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
    forwarded_proto = (
        str(forwarded_proto_raw).lower() if forwarded_proto_raw is not None else ""
    )
    if COOKIE_SECURE or forwarded_proto == "https":
        attrs.append("Secure")
    handler.send_header("Set-Cookie", "; ".join(attrs))


_PASSWORD_CACHE: str | None = None


@dataclass(frozen=True)
class ClientFileView:
    kind: str
    size: int
    content_type: str | None = None
    text: str | None = None
    editable: bool = False
    version: str | None = None
    blocked_reason: str | None = None
    viewer_max_bytes: int | None = None


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


def _file_content_version(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_extension(path: Path) -> str:
    suffix = str(path.suffix or "").lower()
    if not suffix.startswith("."):
        return ""
    return suffix[1:]


def _markdown_kind(path: Path) -> str:
    return "markdown" if _file_extension(path) in MARKDOWN_EXTENSIONS else "text"


def _path_looks_textual(path: Path) -> bool:
    ext = _file_extension(path)
    if ext in TEXTUAL_EXTENSIONS:
        return True
    return str(path.name or "").strip().lower() in TEXTUAL_FILENAMES


def _looks_like_text_bytes(raw: bytes) -> bool:
    if b"\x00" in raw:
        return False
    for b in raw:
        if b < 32 and b not in (9, 10, 12, 13, 27):
            return False
    return True


def _decode_text_for_client(raw: bytes) -> tuple[str, bool]:
    try:
        return raw.decode("utf-8"), True
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), False


def _decode_text_view_for_client(
    path: Path, raw: bytes
) -> tuple[str, bool, str] | None:
    if b"\x00" in raw:
        return None
    try:
        text = raw.decode("utf-8")
        editable = True
    except UnicodeDecodeError:
        if not _path_looks_textual(path) and not _looks_like_text_bytes(raw):
            return None
        text = raw.decode("utf-8", errors="replace")
        editable = False
    return text, editable, _file_content_version(raw)


def _read_text_file_for_client(
    path: Path, *, max_bytes: int
) -> tuple[str, int, bool, str]:
    st = path.stat()
    size = int(st.st_size)
    if size > max_bytes:
        raise ValueError(f"file too large (max {max_bytes} bytes)")
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError("binary file not supported")
    text, editable = _decode_text_for_client(data)
    return text, size, editable, _file_content_version(data)


def _read_text_file_for_write(path: Path, *, max_bytes: int) -> tuple[str, int, str]:
    st = path.stat()
    size = int(st.st_size)
    if size > max_bytes:
        raise ValueError(f"file too large (max {max_bytes} bytes)")
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError("binary file not supported")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("file is not editable as utf-8 text") from e
    return text, size, _file_content_version(data)


def _write_text_file_atomic(path: Path, *, text: str) -> tuple[int, str]:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if path.is_symlink():
        raise ValueError("symlink file not supported")
    data = text.encode("utf-8")
    size = len(data)
    if size > FILE_READ_MAX_BYTES:
        raise ValueError(f"file too large (max {FILE_READ_MAX_BYTES} bytes)")
    st = path.stat()
    tmp = path.with_name(f".{path.name}.codoxear-tmp-{secrets.token_hex(6)}")
    try:
        tmp.write_bytes(data)
        os.chmod(tmp, st.st_mode & 0o777)
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return size, _file_content_version(data)


def _write_new_text_file_atomic(path: Path, *, text: str) -> tuple[int, str]:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if path.is_symlink():
        raise ValueError("symlink file not supported")
    parent = path.parent
    if not parent.exists():
        raise FileNotFoundError("parent directory not found")
    if not parent.is_dir():
        raise ValueError("parent path is not a directory")
    if parent.is_symlink():
        raise ValueError("symlink parent directory not supported")
    if path.exists():
        raise FileExistsError("file already exists")
    data = text.encode("utf-8")
    size = len(data)
    if size > FILE_READ_MAX_BYTES:
        raise ValueError(f"file too large (max {FILE_READ_MAX_BYTES} bytes)")
    tmp = path.with_name(f".{path.name}.codoxear-tmp-{secrets.token_hex(6)}")
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.link(str(tmp), str(path))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return size, _file_content_version(data)


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
    if (
        not str(resolved).startswith(str(resolved_base) + os.sep)
        and resolved != resolved_base
    ):
        raise ValueError("path escapes session cwd")
    return resolved


def _safe_expanduser(p: Path) -> Path:
    try:
        return p.expanduser()
    except RuntimeError:
        return p


def _resolve_session_path(base: Path, raw_path: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("path required")
    if "\x00" in raw_path:
        raise ValueError("invalid path")
    p = Path(raw_path)
    if p.is_absolute():
        return _safe_expanduser(p).resolve()
    resolved_base = _safe_expanduser(base)
    if not resolved_base.is_absolute():
        resolved_base = resolved_base.resolve()
    return (resolved_base / p).resolve()


def _resolve_git_path(cwd: Path, raw_path: str) -> tuple[Path, Path, str]:
    repo_root = Path(
        _run_git(
            cwd,
            ["rev-parse", "--show-toplevel"],
            timeout_s=GIT_DIFF_TIMEOUT_SECONDS,
            max_bytes=64 * 1024,
        ).strip()
    ).resolve()
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
        dirnames[:] = [
            d
            for d in dirnames
            if d
            not in {
                ".git",
                ".hg",
                ".svn",
                "__pycache__",
                "node_modules",
                "build",
                "dist",
            }
        ]
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
        candidate = _safe_expanduser(Path(raw)).resolve()
        if candidate.name != name:
            continue
        if match is None:
            match = candidate
            continue
        if candidate != match:
            return None
    return match


def _resolve_session_relative_child(base: Path, raw_path: str) -> Path:
    rel = str(raw_path or "").strip()
    if not rel:
        return base.resolve()
    if "\x00" in rel:
        raise ValueError("invalid path")
    p = Path(rel)
    if p.is_absolute():
        raise ValueError("path must be relative")
    resolved_base = base.resolve()
    resolved = (resolved_base / p).resolve()
    if (
        not str(resolved).startswith(str(resolved_base) + os.sep)
        and resolved != resolved_base
    ):
        raise ValueError("path escapes session cwd")
    return resolved


def _load_root_gitignore_patterns(root: Path) -> list[str]:
    path = root / ".gitignore"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError:
        return []
    patterns: list[str] = []
    for line in raw.splitlines():
        pattern = line.strip()
        if not pattern or pattern.startswith("#") or pattern.startswith("!"):
            continue
        patterns.append(pattern)
    return patterns


def _gitignore_matches(rel_path: str, *, is_dir: bool, pattern: str) -> bool:
    candidate = rel_path.strip("/")
    if not candidate:
        return False
    rule = pattern.strip()
    if not rule:
        return False
    dir_only = rule.endswith("/")
    if dir_only and not is_dir:
        return False
    rule = rule.rstrip("/")
    if not rule:
        return False
    anchored = rule.startswith("/")
    rule = rule.lstrip("/")
    if not rule:
        return False

    if "/" in rule:
        return fnmatch.fnmatchcase(candidate, rule)

    parts = candidate.split("/")
    if anchored:
        return fnmatch.fnmatchcase(parts[0], rule)
    return any(fnmatch.fnmatchcase(part, rule) for part in parts)


def _is_ignored_session_relpath(
    rel_path: str, *, is_dir: bool, patterns: list[str]
) -> bool:
    return any(
        _gitignore_matches(rel_path, is_dir=is_dir, pattern=pattern)
        for pattern in patterns
    )


def _session_entry_sort_key(entry: dict[str, str]) -> tuple[int, str]:
    return (0 if entry.get("kind") == "dir" else 1, entry.get("name", ""))


def _list_session_directory_entries(
    base: Path, raw_path: str = ""
) -> list[dict[str, str]]:
    root = _safe_expanduser(base).resolve()
    if not root.exists():
        raise FileNotFoundError("session cwd not found")
    if not root.is_dir():
        raise ValueError("session cwd is not a directory")
    target = _resolve_session_relative_child(root, raw_path)
    if not target.exists():
        raise FileNotFoundError("path not found")
    if not target.is_dir():
        raise ValueError("path is not a directory")

    patterns = _load_root_gitignore_patterns(root)
    out: list[dict[str, str]] = []
    for child in target.iterdir():
        rel = child.relative_to(root).as_posix()
        if child.is_dir() and child.name in FILE_LIST_IGNORED_DIRS:
            continue
        if _is_ignored_session_relpath(rel, is_dir=child.is_dir(), patterns=patterns):
            continue
        out.append(
            {
                "name": child.name,
                "path": rel,
                "kind": "dir" if child.is_dir() else "file",
            }
        )
    out.sort(key=_session_entry_sort_key)
    return out


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


def _resolve_dir_target(raw: str, *, field_name: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field_name} required")
    path = _expand_user_path(raw).resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"{field_name} is not a directory: {path}")
    return path


def _codex_trust_override_for_path(path: Path) -> str:
    return f'projects={{ {json.dumps(str(path.resolve()))} = {{ trust_level = "trusted" }} }}'


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
    _run_git(
        cwd,
        ["rev-parse", "--is-inside-work-tree"],
        timeout_s=GIT_DIFF_TIMEOUT_SECONDS,
        max_bytes=4096,
    )


def _git_repo_root(cwd: Path) -> Path | None:
    try:
        root = _run_git(
            cwd,
            ["rev-parse", "--show-toplevel"],
            timeout_s=GIT_DIFF_TIMEOUT_SECONDS,
            max_bytes=64 * 1024,
        ).strip()
    except (RuntimeError, FileNotFoundError):
        return None
    if not root:
        return None
    return Path(root).resolve()


def _file_search_score(candidate: str, query: str) -> int:
    text = str(candidate or "")
    raw = str(query or "").strip().lower()
    if not raw:
        return 0
    lower = text.lower()
    if lower == raw:
        return 12000
    base = Path(text).name.lower()
    if base == raw:
        return 10000
    total = 0
    for token in [part for part in raw.split() if part]:
        exact_idx = lower.find(token)
        if exact_idx >= 0:
            prev = lower[exact_idx - 1] if exact_idx > 0 else ""
            boundary_bonus = 24 if (not prev or prev in "/._-") else 0
            base_idx = base.find(token)
            total += (
                240
                - exact_idx * 2
                + boundary_bonus
                + (44 - base_idx if base_idx >= 0 else 0)
            )
            continue
        pos = -1
        first = -1
        last = -1
        consecutive = 0
        boundaries = 0
        for ch in token:
            pos = lower.find(ch, pos + 1)
            if pos < 0:
                return -1
            if first < 0:
                first = pos
            if last >= 0 and pos == last + 1:
                consecutive += 1
            if pos == 0 or lower[pos - 1] in "/._-":
                boundaries += 1
            last = pos
        span = last - first + 1
        total += (
            120
            - first
            - max(0, span - len(token)) * 4
            + consecutive * 10
            + boundaries * 8
        )
    return total


def _push_file_search_match(
    heap: list[tuple[int, str]], *, path: str, score: int, limit: int
) -> None:
    item = (score, path)
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    if item > heap[0]:
        heapq.heapreplace(heap, item)


def _finish_file_search(
    heap: list[tuple[int, str]], *, mode: str, query: str, scanned: int, truncated: bool
) -> dict[str, Any]:
    matches = [
        {"path": path, "score": score}
        for score, path in sorted(heap, key=lambda item: (-item[0], item[1]))
    ]
    return {
        "mode": mode,
        "query": query,
        "matches": matches,
        "scanned": scanned,
        "truncated": truncated,
    }


def _search_walk_relative_files(
    root: Path, *, query: str, limit: int
) -> dict[str, Any]:
    deadline = time.monotonic() + FILE_SEARCH_TIMEOUT_SECONDS
    heap: list[tuple[int, str]] = []
    scanned = 0
    truncated = False

    def _onerror(err: OSError) -> None:
        raise err

    for current_root, dirnames, filenames in os.walk(
        root, topdown=True, onerror=_onerror, followlinks=False
    ):
        dirnames[:] = [
            name for name in sorted(dirnames) if name not in FILE_LIST_IGNORED_DIRS
        ]
        current_path = Path(current_root)
        for name in sorted(filenames):
            scanned += 1
            if scanned > FILE_SEARCH_MAX_CANDIDATES or time.monotonic() > deadline:
                truncated = True
                return _finish_file_search(
                    heap,
                    mode="walk",
                    query=query,
                    scanned=scanned - 1,
                    truncated=truncated,
                )
            rel = (current_path / name).relative_to(root).as_posix()
            score = _file_search_score(rel, query)
            if score < 0:
                continue
            _push_file_search_match(heap, path=rel, score=score, limit=limit)
    return _finish_file_search(
        heap, mode="walk", query=query, scanned=scanned, truncated=truncated
    )


def _search_git_relative_files(cwd: Path, *, query: str, limit: int) -> dict[str, Any]:
    deadline = time.monotonic() + FILE_SEARCH_TIMEOUT_SECONDS
    heap: list[tuple[int, str]] = []
    scanned = 0
    truncated = False
    proc = subprocess.Popen(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            path = raw_line.rstrip("\n")
            if not path:
                continue
            scanned += 1
            if scanned > FILE_SEARCH_MAX_CANDIDATES or time.monotonic() > deadline:
                truncated = True
                proc.kill()
                break
            score = _file_search_score(path, query)
            if score < 0:
                continue
            _push_file_search_match(heap, path=path, score=score, limit=limit)
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        return_code = proc.wait()
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
    if truncated:
        return _finish_file_search(
            heap, mode="git", query=query, scanned=scanned - 1, truncated=True
        )
    if return_code != 0:
        err = stderr.strip()
        raise RuntimeError(err or f"git ls-files failed with code {return_code}")
    return _finish_file_search(
        heap, mode="git", query=query, scanned=scanned, truncated=False
    )


def _search_session_relative_files(
    base: Path, *, query: str, limit: int = FILE_SEARCH_LIMIT
) -> dict[str, Any]:
    root = _safe_expanduser(base)
    if not root.is_absolute():
        root = root.resolve()
    if not root.exists():
        raise FileNotFoundError("session cwd not found")
    if not root.is_dir():
        raise ValueError("session cwd is not a directory")
    raw_query = str(query).strip()
    if not raw_query:
        raise ValueError("query required")
    clamped_limit = max(1, min(int(limit), FILE_SEARCH_LIMIT))
    repo_root = _git_repo_root(root)
    if repo_root is not None:
        return _search_git_relative_files(root, query=raw_query, limit=clamped_limit)
    return _search_walk_relative_files(root, query=raw_query, limit=clamped_limit)


def _describe_session_cwd(cwd: Path) -> dict[str, Any]:
    exists = cwd.exists()
    if exists and not cwd.is_dir():
        raise ValueError(f"cwd is not a directory: {cwd}")
    repo_root = _git_repo_root(cwd) if exists else None
    git_branch = (_current_git_branch(cwd) or "") if exists else ""
    return {
        "cwd": str(cwd),
        "exists": exists,
        "will_create": not exists,
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
        raise ValueError(
            err or out or f"git worktree add failed with code {proc.returncode}"
        )
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


def _stage_uploaded_file(
    session_id: str,
    filename: str,
    raw: bytes,
    *,
    max_bytes: int = ATTACH_UPLOAD_MAX_BYTES,
) -> Path:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id required")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename required")
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError("file bytes required")
    data = bytes(raw)
    if len(data) > int(max_bytes):
        raise ValueError(f"file too large (max {int(max_bytes)} bytes)")
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


def _normalize_cwd_group_key(cwd: Any) -> str:
    if not isinstance(cwd, str) or not cwd.strip():
        raise ValueError("cwd must be a non-empty string")
    trimmed = cwd.strip()
    return str(Path(trimmed).expanduser().resolve(strict=False))


def _existing_workspace_dir(cwd: Any) -> str | None:
    try:
        normalized = _normalize_cwd_group_key(cwd)
    except ValueError:
        return None
    try:
        if not Path(normalized).is_dir():
            return None
    except OSError:
        return None
    return normalized


def _canonical_session_cwd(cwd: Any) -> str | None:
    if not isinstance(cwd, str):
        return None
    trimmed = cwd.strip()
    if not trimmed:
        return None
    try:
        return _normalize_cwd_group_key(trimmed)
    except ValueError:
        return trimmed


SESSION_LIST_ROW_KEYS = (
    "session_id",
    "runtime_id",
    "thread_id",
    "display_name",
    "title",
    "alias",
    "first_user_message",
    "cwd",
    "agent_backend",
    "owned",
    "busy",
    "queue_len",
    "git_branch",
    "transport",
    "blocked",
    "snoozed",
    "historical",
    "pending_startup",
    "focused",
)
SESSION_LIST_PAGE_SIZE = 50
SESSION_LIST_GROUP_PAGE_SIZE = 12
SESSION_LIST_RECENT_GROUP_LIMIT = 12
SESSION_LIST_FALLBACK_GROUP_KEY = "__no_working_directory__"


def _session_row_display_name(row: dict[str, Any], *, fallback: str = "Session") -> str:
    if not isinstance(row, dict):
        return fallback
    for key in ("alias", "title", "first_user_message", "session_id"):
        value = row.get(key)
        if not isinstance(value, str):
            continue
        out = value.strip()
        if out:
            return out
    return fallback


def _clean_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _normalize_session_cwd_row(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        return row
    normalized = dict(row)
    if "cwd" in normalized:
        canonical_cwd = _canonical_session_cwd(normalized.get("cwd"))
        if canonical_cwd is not None:
            normalized["cwd"] = canonical_cwd
    normalized["display_name"] = _session_row_display_name(normalized)
    return normalized


def _frontend_session_list_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_session_cwd_row(row)
    if not isinstance(normalized, dict):
        return normalized
    return {key: normalized[key] for key in SESSION_LIST_ROW_KEYS if key in normalized}


def _session_list_group_key(row: dict[str, Any]) -> str:
    cwd = _canonical_session_cwd(row.get("cwd"))
    return cwd or SESSION_LIST_FALLBACK_GROUP_KEY


def _session_list_payload(
    rows: list[dict[str, Any]],
    *,
    group_key: str | None = None,
    offset: int = 0,
    limit: int = SESSION_LIST_PAGE_SIZE,
    group_offset: int = 0,
    group_limit: int = SESSION_LIST_RECENT_GROUP_LIMIT,
) -> dict[str, Any]:
    start = max(0, int(offset))
    stop = start + max(1, int(limit))

    if group_key is None and group_offset <= 0 and group_limit == SESSION_LIST_RECENT_GROUP_LIMIT:
        page_rows = [_frontend_session_list_row(row) for row in rows[start:stop]]
        remaining = max(0, len(rows) - stop)
        payload: dict[str, Any] = {"sessions": page_rows}
        if remaining > 0:
            payload["remaining_count"] = remaining
        return payload

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = _session_list_group_key(row)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)

    def _page_rows_for_group(group_rows: list[dict[str, Any]], page_size: int) -> list[dict[str, Any]]:
        if len(group_rows) <= page_size:
            return list(group_rows)
        page_rows = list(group_rows[:page_size])
        focused_rows = [row for row in group_rows if bool(row.get("focused"))]
        if not focused_rows:
            return page_rows
        existing_ids = {id(row) for row in page_rows}
        extras = [row for row in focused_rows if id(row) not in existing_ids]
        if not extras:
            return page_rows
        keep = [row for row in page_rows if bool(row.get("focused"))]
        keep.extend(extras)
        keep = keep[:page_size]
        keep_ids = {id(item) for item in keep}
        for row in group_rows:
            if len(keep) >= page_size:
                break
            if id(row) in keep_ids:
                continue
            keep.append(row)
            keep_ids.add(id(row))
        return keep

    def _group_sort_key(key: str) -> tuple[int, float]:
        group_rows = grouped[key]
        busy = any(bool(row.get("busy")) for row in group_rows)
        latest_updated = max(float(row.get("updated_ts") or 0.0) for row in group_rows)
        return (0 if busy else 1, -latest_updated)

    group_order = sorted(grouped.keys(), key=_group_sort_key)

    if group_key is not None:
        group_rows = grouped.get(group_key, [])
        page_rows = [_frontend_session_list_row(row) for row in group_rows[start:stop]]
        remaining = max(0, len(group_rows) - stop)
        return {
            "sessions": page_rows,
            "remaining_by_group": {group_key: remaining} if remaining > 0 else {},
        }

    selected_group_keys = set(group_order[:SESSION_LIST_RECENT_GROUP_LIMIT])
    omitted_group_count = 0
    for key, group_rows in grouped.items():
        if any(bool(row.get("busy")) for row in group_rows) or any(bool(row.get("focused")) for row in group_rows):
            selected_group_keys.add(key)

    if group_offset > 0 or group_limit != SESSION_LIST_RECENT_GROUP_LIMIT:
        group_stop = max(group_offset, 0) + max(1, int(group_limit))
        extra_group_order = group_order[group_offset:group_stop]
        selected_group_keys = set(extra_group_order)
        omitted_group_count = max(0, len(group_order) - group_stop)

    sessions: list[dict[str, Any]] = []
    remaining_by_group: dict[str, int] = {}
    for key in group_order:
        if key not in selected_group_keys:
            continue
        group_rows = grouped[key]
        page_rows = _page_rows_for_group(group_rows, SESSION_LIST_GROUP_PAGE_SIZE)
        sessions.extend(_frontend_session_list_row(row) for row in page_rows)
        remaining = len(group_rows) - len(page_rows)
        if remaining > 0:
            remaining_by_group[key] = remaining
    result: dict[str, Any] = {
        "sessions": sessions,
        "remaining_by_group": remaining_by_group,
    }
    if group_offset <= 0 and group_limit == SESSION_LIST_RECENT_GROUP_LIMIT:
        omitted_group_count = max(0, len(group_order) - len(selected_group_keys))
    result["omitted_group_count"] = omitted_group_count
    return result


def _listed_session_row(manager: "SessionManager", session_id: str) -> dict[str, Any] | None:
    for row in manager.list_sessions():
        if str(row.get("session_id") or "") == session_id:
            return dict(row)
    return None


def _session_details_payload(
    manager: "SessionManager", session_id: str
) -> dict[str, Any]:
    return _session_payloads.session_details_payload(manager, session_id)


def _clean_recent_cwd(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


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


def _clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


def _clean_optional_resume_session_id(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("resume_session_id must be a string")
    out = value.strip()
    return out or None


def _normalize_requested_model(value: Any) -> str | None:
    out = _clean_optional_text(value)
    if out is None:
        return None
    return None if out.lower() == "default" else out


def _display_reasoning_effort(value: Any) -> str | None:
    out = _clean_optional_text(value)
    if out is None:
        return None
    lowered = out.lower()
    return lowered if lowered in SUPPORTED_REASONING_EFFORTS else None


def _display_pi_reasoning_effort(value: Any) -> str | None:
    out = _clean_optional_text(value)
    if out is None:
        return None
    lowered = out.lower()
    return lowered if lowered in SUPPORTED_PI_REASONING_EFFORTS else None


def _normalize_requested_reasoning_effort(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("reasoning_effort must be a string")
    out = value.strip().lower()
    if not out:
        return None
    if out not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort must be one of {', '.join(SUPPORTED_REASONING_EFFORTS)}"
        )
    return out


def _normalize_requested_pi_reasoning_effort(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("reasoning_effort must be a string")
    out = value.strip().lower()
    if not out:
        return None
    if out not in SUPPORTED_PI_REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort must be one of {', '.join(SUPPORTED_PI_REASONING_EFFORTS)}"
        )
    return out


def _priority_from_elapsed_seconds(elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 1.0
    return _clip01(math.exp(-SIDEBAR_PRIORITY_LAMBDA * float(elapsed_s)))


def _current_git_branch(cwd: Path) -> str | None:
    try:
        branch = _run_git(
            cwd,
            ["rev-parse", "--abbrev-ref", "HEAD"],
            timeout_s=GIT_DIFF_TIMEOUT_SECONDS,
            max_bytes=64 * 1024,
        ).strip()
    except (RuntimeError, FileNotFoundError):
        return None
    if not branch:
        return None
    return branch


def _todo_snapshot_payload_for_session(s: Session) -> dict[str, Any]:
    empty = {"available": False, "error": False, "items": []}
    read_error = {"available": False, "error": True, "items": []}
    if s.backend != "pi" or s.session_path is None:
        return empty
    try:
        snapshot = _pi_messages.read_latest_pi_todo_snapshot(s.session_path)
    except FileNotFoundError:
        return empty
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return empty
        return read_error
    if snapshot is None:
        return empty
    return {
        "available": True,
        "error": False,
        "items": snapshot.get("items", []),
        "counts": snapshot.get("counts", {}),
        "progress_text": snapshot.get("progress_text", ""),
    }


_PI_DIALOG_UI_METHODS = {"select", "confirm", "input", "editor"}


def _sanitize_pi_ui_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
    requests = payload.get("requests")
    if not isinstance(requests, list):
        return {"requests": []}
    filtered = []
    for item in requests:
        if not isinstance(item, dict):
            continue
        method = item.get("method")
        if not isinstance(method, str) or method not in _PI_DIALOG_UI_METHODS:
            continue
        filtered.append(item)
    return {"requests": filtered}


def _ui_requests_version(requests: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        requests,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()[:12]
    return _b64u(digest)


def _sanitize_pi_commands_payload(payload: dict[str, Any]) -> dict[str, Any]:
    commands = payload.get("commands")
    if not isinstance(commands, list):
        return {"commands": []}
    filtered: list[dict[str, Any]] = []
    for item in commands:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        clean_name = name.strip()
        if not clean_name:
            continue
        clean_item: dict[str, Any] = {"name": clean_name}
        description = item.get("description")
        if isinstance(description, str) and description.strip():
            clean_item["description"] = description.strip()
        source = item.get("source")
        if isinstance(source, str) and source.strip():
            clean_item["source"] = source.strip()
        filtered.append(clean_item)
    return {"commands": filtered}


def _legacy_pi_ui_response_text(payload: dict[str, Any]) -> str | None:
    if payload.get("cancelled") is True:
        return None
    confirmed = payload.get("confirmed")
    if isinstance(confirmed, bool):
        return "yes" if confirmed else "no"
    value = payload.get("value")
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        parts = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text:
                continue
            parts.append(text)
        if parts:
            return ", ".join(parts)
    return None


def _resolve_client_file_path(*, session_id: str, raw_path: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("empty path")
    if "\n" in raw_path or len(raw_path) > 1024:
        # Likely a code snippet or pasted content, not a path.
        # Rejecting here avoids OSError: File name too long in Path.exists().
        raise ValueError("invalid path format")

    try:
        path_obj = _safe_expanduser(Path(raw_path))
        for part in path_obj.parts:
            if len(part.encode("utf-8", errors="ignore")) > 255:
                raise ValueError("invalid path format (name too long)")
    except ValueError:
        raise
    except Exception:
        raise ValueError("invalid path format")

    if not path_obj.is_absolute():
        if session_id:
            MANAGER.refresh_session_meta(session_id, strict=False)
            s = MANAGER.get_session(session_id)
            if s:
                base = _safe_expanduser(Path(s.cwd))
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
                            _run_git(
                                base,
                                ["rev-parse", "--show-toplevel"],
                                timeout_s=GIT_DIFF_TIMEOUT_SECONDS,
                                max_bytes=64 * 1024,
                            ).strip()
                        ).resolve()
                    except RuntimeError:
                        repo_root = base.resolve()
                    path_obj = (
                        _resolve_unique_bare_filename(repo_root, raw_path) or direct
                    )
            else:
                path_obj = (Path.cwd() / path_obj).resolve()
        else:
            path_obj = (Path.cwd() / path_obj).resolve()
    else:
        path_obj = path_obj.resolve()
    return path_obj


def _inspect_openable_file(path_obj: Path) -> tuple[bytes, int, str, str | None]:
    view = _read_client_file_view(path_obj)
    if view.kind == "directory":
        raise ValueError("path is not a file")
    if view.kind == "download_only":
        if view.blocked_reason == "too_large":
            raise ValueError(f"file too large (max {FILE_READ_MAX_BYTES} bytes)")
        raise ValueError("binary file not supported")
    raw = path_obj.read_bytes()
    return raw, view.size, view.kind, view.content_type


def _inspect_path_metadata(path_obj: Path) -> tuple[int, str, str | None]:
    view = _read_client_file_view(path_obj)
    return view.size, view.kind, view.content_type


def _read_client_file_view(path_obj: Path) -> ClientFileView:
    if not path_obj.exists():
        raise FileNotFoundError("file not found")
    if path_obj.is_dir():
        return ClientFileView(kind="directory", size=0)
    if not path_obj.is_file():
        raise ValueError("path is not a file")
    try:
        size = int(path_obj.stat().st_size)
        with path_obj.open("rb") as f:
            prefix = f.read(4096)
    except PermissionError as e:
        raise PermissionError("permission denied") from e
    kind, content_type = _file_kind(path_obj, prefix)
    if kind in {"image", "pdf"}:
        return ClientFileView(kind=kind, size=size, content_type=content_type)
    if size > FILE_READ_MAX_BYTES:
        return ClientFileView(
            kind="download_only",
            size=size,
            blocked_reason="too_large",
            viewer_max_bytes=FILE_READ_MAX_BYTES,
        )
    raw = path_obj.read_bytes()
    text_payload = _decode_text_view_for_client(path_obj, raw)
    if text_payload is None:
        return ClientFileView(kind="download_only", size=size, blocked_reason="binary")
    text, editable, version = text_payload
    return ClientFileView(
        kind=_markdown_kind(path_obj),
        size=size,
        text=text,
        editable=editable,
        version=version,
    )


def _read_text_or_image(path_obj: Path) -> tuple[str, int, str | None, bytes | None]:
    view = _read_client_file_view(path_obj)
    if view.kind in {"image", "pdf", "download_only", "directory"}:
        return view.kind, view.size, view.content_type, None
    raw = path_obj.read_bytes()
    return view.kind, view.size, view.content_type, raw


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
    view = _read_client_file_view(path_obj)
    return view.size, view.kind, view.content_type


def _download_disposition(path_obj: Path) -> str:
    return f"attachment; filename*=UTF-8''{urllib.parse.quote(path_obj.name, safe='')}"


def _iter_session_logs(*, agent_backend: str = "codex") -> list[Path]:
    backend_name = normalize_agent_backend(agent_backend)
    sessions_dir = CODEX_SESSIONS_DIR if backend_name == "codex" else PI_SESSIONS_DIR
    return _iter_session_logs_impl(sessions_dir, agent_backend=backend_name)


def _find_session_log_for_session_id(
    session_id: str, *, agent_backend: str = "codex"
) -> Path | None:
    backend_name = normalize_agent_backend(agent_backend)
    sessions_dir = CODEX_SESSIONS_DIR if backend_name == "codex" else PI_SESSIONS_DIR
    return _find_session_log_for_session_id_impl(
        sessions_dir, session_id, agent_backend=backend_name
    )


def _find_new_session_log(
    *,
    agent_backend: str = "codex",
    after_ts: float,
    preexisting: set[Path],
    timeout_s: float = 15.0,
) -> tuple[str, Path] | None:
    backend_name = normalize_agent_backend(agent_backend)
    sessions_dir = CODEX_SESSIONS_DIR if backend_name == "codex" else PI_SESSIONS_DIR
    return _find_new_session_log_impl(
        sessions_dir=sessions_dir,
        agent_backend=backend_name,
        after_ts=after_ts,
        preexisting=preexisting,
        timeout_s=timeout_s,
    )


def _read_jsonl_from_offset(
    path: Path, offset: int, max_bytes: int = 2 * 1024 * 1024
) -> tuple[list[dict[str, Any]], int]:
    return _read_jsonl_from_offset_impl(path, offset, max_bytes=max_bytes)


def _discover_log_for_session_id(
    session_id: str, *, agent_backend: str = "codex"
) -> Path | None:
    return _find_session_log_for_session_id(session_id, agent_backend=agent_backend)


def _session_id_from_rollout_path(log_path: Path) -> str | None:
    name = log_path.name
    m = _SESSION_ID_RE.findall(name)
    return m[-1] if m else None


def _read_session_meta(
    log_path: Path, *, agent_backend: str | None = None
) -> dict[str, Any]:
    if agent_backend is None:
        try:
            log_path.resolve().relative_to(PI_SESSIONS_DIR.resolve())
            inferred = "pi"
        except Exception:
            inferred = "codex"
        backend_name = inferred
    else:
        backend_name = normalize_agent_backend(agent_backend)
    payload = _read_session_meta_payload_impl(
        log_path, agent_backend=backend_name, timeout_s=0.0
    )
    if payload is None:
        raise ValueError(f"missing session metadata in {log_path}")
    return payload


def _turn_context_run_settings(payload: Any) -> tuple[str | None, str | None]:
    if not isinstance(payload, dict):
        return None, None
    return (
        _clean_optional_text(payload.get("model")),
        _display_reasoning_effort(
            payload.get("reasoning_effort") or payload.get("effort")
        ),
    )


def _read_run_settings_from_log(
    log_path: Path, *, agent_backend: str = "codex"
) -> tuple[str | None, str | None, str | None]:
    backend_name = normalize_agent_backend(agent_backend)
    if backend_name == "pi":
        return _read_pi_run_settings(log_path)
    meta = _read_session_meta(log_path, agent_backend="codex")
    model_provider = _clean_optional_text(meta.get("model_provider"))
    model = _clean_optional_text(meta.get("model"))
    reasoning_effort = _display_reasoning_effort(meta.get("reasoning_effort"))
    if model is None or reasoning_effort is None:
        ctx_model, ctx_effort = _turn_context_run_settings(
            _rollout_log._find_latest_turn_context(
                log_path, max_scan_bytes=8 * 1024 * 1024
            )
        )
        if model is None:
            model = ctx_model
        if reasoning_effort is None:
            reasoning_effort = ctx_effort
    return model_provider, model, reasoning_effort


def _normalize_requested_model_provider(
    value: Any, *, allowed: set[str] | None = None
) -> str | None:
    provider = _clean_optional_text(value)
    if provider is None:
        return None
    if allowed is not None and provider not in allowed:
        allowed_txt = ", ".join(sorted(allowed))
        raise ValueError(f"model_provider must be one of {allowed_txt}")
    return provider


def _normalize_requested_service_tier(value: Any) -> str | None:
    tier = _clean_optional_text(value)
    if tier is None:
        return None
    if tier not in {"fast", "flex"}:
        raise ValueError("service_tier must be one of fast, flex")
    return tier


def _normalize_requested_preferred_auth_method(value: Any) -> str | None:
    method = _clean_optional_text(value)
    if method is None:
        return None
    if method not in {"chatgpt", "apikey"}:
        raise ValueError("preferred_auth_method must be one of chatgpt, apikey")
    return method


def _normalize_requested_backend(raw: Any) -> str:
    if raw is None:
        return "codex"
    if not isinstance(raw, str):
        raise ValueError("backend must be a string")
    backend = raw.strip().lower()
    if not backend:
        return "codex"
    if backend not in {"codex", "pi"}:
        raise ValueError("backend must be one of codex, pi")
    return backend


def _parse_create_session_request(obj: dict[str, Any]) -> dict[str, Any]:
    cwd = obj.get("cwd")
    if not isinstance(cwd, str) or not cwd.strip():
        raise ValueError("cwd required")

    name_raw = obj.get("name")
    if name_raw is None:
        name = None
    elif isinstance(name_raw, str):
        name = name_raw
    else:
        raise ValueError("name must be a string")

    backend = normalize_agent_backend(
        obj.get("backend"),
        default=normalize_agent_backend(obj.get("agent_backend"), default="codex"),
    )
    args = obj.get("args")
    if args is None:
        args_list = None
    elif isinstance(args, list) and all(isinstance(x, str) for x in args):
        args_list = [x for x in args if x]
    else:
        raise ValueError("args must be a list of strings")

    resume_session_id = _clean_optional_resume_session_id(obj.get("resume_session_id"))

    if backend == "pi":
        create_in_tmux_raw = obj.get("create_in_tmux")
        if create_in_tmux_raw is None:
            create_in_tmux = False
        elif isinstance(create_in_tmux_raw, bool):
            create_in_tmux = create_in_tmux_raw
        else:
            raise ValueError("create_in_tmux must be a boolean")

        pi_provider_choices = {
            str(value)
            for value in (_read_pi_launch_defaults().get("provider_choices") or [])
            if isinstance(value, str) and value.strip()
        }
        model_provider = _normalize_requested_model_provider(
            obj.get("model_provider"),
            allowed=pi_provider_choices or None,
        )
        model = _normalize_requested_model(obj.get("model"))
        reasoning_effort = _normalize_requested_pi_reasoning_effort(
            obj.get("reasoning_effort")
        )
        return {
            "cwd": cwd,
            "name": name,
            "backend": backend,
            "args": args_list,
            "resume_session_id": resume_session_id,
            "worktree_branch": None,
            "model_provider": model_provider,
            "preferred_auth_method": None,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "service_tier": None,
            "create_in_tmux": create_in_tmux,
        }

    allowed_providers = set(
        _read_codex_launch_defaults().get("model_providers") or ["openai"]
    )
    model_provider = _normalize_requested_model_provider(
        obj.get("model_provider"),
        allowed=set(
            [
                "openai",
                *[p for p in allowed_providers if p not in {"chatgpt", "openai-api"}],
            ]
        ),
    )
    preferred_auth_method = _normalize_requested_preferred_auth_method(
        obj.get("preferred_auth_method")
    )
    model = _normalize_requested_model(obj.get("model"))
    reasoning_effort = _normalize_requested_reasoning_effort(
        obj.get("reasoning_effort")
    )
    service_tier = _normalize_requested_service_tier(obj.get("service_tier"))

    create_in_tmux_raw = obj.get("create_in_tmux")
    if create_in_tmux_raw is None:
        create_in_tmux = False
    elif isinstance(create_in_tmux_raw, bool):
        create_in_tmux = create_in_tmux_raw
    else:
        raise ValueError("create_in_tmux must be a boolean")

    worktree_branch_raw = obj.get("worktree_branch")
    if worktree_branch_raw is None:
        worktree_branch = None
    elif isinstance(worktree_branch_raw, str):
        worktree_branch = worktree_branch_raw.strip() or None
    else:
        raise ValueError("worktree_branch must be a string")

    return {
        "cwd": cwd,
        "name": name,
        "backend": backend,
        "args": args_list,
        "resume_session_id": resume_session_id,
        "worktree_branch": worktree_branch,
        "model_provider": model_provider,
        "preferred_auth_method": preferred_auth_method,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "service_tier": service_tier,
        "create_in_tmux": create_in_tmux,
    }


def _configured_model_providers(data: dict[str, Any]) -> list[str]:
    providers = ["openai"]
    seen = {"openai"}
    raw = data.get("model_providers")
    if not isinstance(raw, dict):
        return providers
    for key in raw.keys():
        if not isinstance(key, str):
            continue
        name = key.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        providers.append(name)
    return providers


def _provider_choice_for_settings(
    *, model_provider: str | None, preferred_auth_method: str | None
) -> str:
    provider = model_provider or "openai"
    if provider == "openai":
        return "chatgpt" if preferred_auth_method == "chatgpt" else "openai-api"
    return provider


def _provider_choice_for_backend(
    *, backend: str, model_provider: str | None, preferred_auth_method: str | None
) -> str | None:
    if backend == "pi":
        return None
    return _provider_choice_for_settings(
        model_provider=model_provider, preferred_auth_method=preferred_auth_method
    )


def _metadata_log_path(
    *, meta: dict[str, Any], backend: str, sock: Path
) -> Path | None:
    if backend == "pi":
        return None
    if "log_path" not in meta:
        raise ValueError(f"missing log_path in metadata for socket {sock}")
    if meta.get("log_path") is None:
        return None
    log_path_raw = meta.get("log_path")
    if not isinstance(log_path_raw, str) or (not log_path_raw.strip()):
        raise ValueError(f"invalid log_path in metadata for socket {sock}")
    return Path(log_path_raw)


def _metadata_session_path(
    *, meta: dict[str, Any], backend: str, sock: Path
) -> Path | None:
    if backend != "pi":
        return None
    if "session_path" not in meta:
        raise ValueError(f"missing session_path in metadata for socket {sock}")
    session_path_raw = meta.get("session_path")
    if not isinstance(session_path_raw, str) or (not session_path_raw.strip()):
        raise ValueError(f"invalid session_path in metadata for socket {sock}")
    return Path(session_path_raw)


def _patch_metadata_session_path(
    sock: Path, session_path: Path, *, force: bool = False
) -> None:
    """Write discovered session_path back to the broker metadata file.

    The PTY broker (broker.py) does not include session_path in its metadata
    for pi sessions.  Once the server discovers the correct file, writing it
    back makes the mapping persistent across server restarts.

    When *force* is True, overwrite even if session_path is already present
    (used when the session has switched via /resume).
    """
    meta_path = sock.with_suffix(".json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            return
        if not force and "session_path" in meta:
            return  # already present (e.g. pi_broker wrote it)
        meta["session_path"] = str(session_path)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    except Exception:
        pass  # best-effort; discovery will re-run next cycle


def _patch_metadata_pi_binding(sock: Path, session_path: Path) -> None:
    """Persist a recovered Pi binding for brokers that were launched ambiguously."""
    meta_path = sock.with_suffix(".json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            return
        changed = False
        if meta.get("backend") != "pi":
            meta["backend"] = "pi"
            changed = True
        if meta.get("agent_backend") != "pi":
            meta["agent_backend"] = "pi"
            changed = True
        if meta.get("session_path") != str(session_path):
            meta["session_path"] = str(session_path)
            changed = True
        if not changed:
            return
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    except Exception:
        pass


def _read_codex_launch_defaults() -> dict[str, Any]:
    configured_model = None
    configured_effort = None
    configured_provider = "openai"
    configured_auth_method = "apikey"
    configured_service_tier = "flex"
    configured_providers = ["chatgpt", "openai-api"]
    if CODEX_CONFIG_PATH.exists():
        data = tomllib.loads(CODEX_CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid Codex config in {CODEX_CONFIG_PATH}")
        configured_model = _clean_optional_text(data.get("model"))
        configured_effort = _display_reasoning_effort(
            data.get("model_reasoning_effort")
        )
        configured_auth_method = (
            _normalize_requested_preferred_auth_method(
                data.get("preferred_auth_method")
            )
            or configured_auth_method
        )
        configured_providers = [
            "chatgpt",
            "openai-api",
            *[p for p in _configured_model_providers(data) if p != "openai"],
        ]
        configured_provider = (
            _normalize_requested_model_provider(
                data.get("model_provider") or data.get("model_provider_id"),
                allowed=set(
                    [
                        "openai",
                        *[
                            p
                            for p in configured_providers
                            if p not in {"chatgpt", "openai-api"}
                        ],
                    ]
                ),
            )
            or configured_provider
        )
        configured_service_tier = (
            _normalize_requested_service_tier(
                data.get("service_tier"),
            )
            or configured_service_tier
        )
    defaults: dict[str, Any] = {
        "model_provider": configured_provider,
        "preferred_auth_method": configured_auth_method,
        "provider_choice": _provider_choice_for_settings(
            model_provider=configured_provider,
            preferred_auth_method=configured_auth_method,
        ),
        "model": configured_model,
        "model_providers": configured_providers,
        "service_tier": configured_service_tier,
    }
    if configured_effort is not None:
        defaults["reasoning_effort"] = configured_effort
        return defaults
    if not MODELS_CACHE_PATH.exists():
        defaults["reasoning_effort"] = None
        return defaults
    cache = json.loads(MODELS_CACHE_PATH.read_text(encoding="utf-8"))
    models = cache.get("models") if isinstance(cache, dict) else None
    if not isinstance(models, list):
        raise ValueError(f"invalid models cache in {MODELS_CACHE_PATH}")
    rows: list[dict[str, Any]] = [row for row in models if isinstance(row, dict)]
    if not rows:
        defaults["reasoning_effort"] = None
        return defaults
    if configured_model is not None:
        for row in rows:
            names = {
                _clean_optional_text(row.get("slug")),
                _clean_optional_text(row.get("display_name")),
            }
            if configured_model in names:
                defaults["reasoning_effort"] = _display_reasoning_effort(
                    row.get("default_reasoning_level")
                )
                return defaults
    ranked = sorted(
        rows,
        key=lambda row: (
            row.get("priority") if isinstance(row.get("priority"), int) else 999999,
            _clean_optional_text(row.get("slug")) or "",
        ),
    )
    defaults["reasoning_effort"] = _display_reasoning_effort(
        ranked[0].get("default_reasoning_level")
    )
    return defaults


def _read_pi_launch_defaults() -> dict[str, Any]:
    configured_provider: str | None = None
    configured_model: str | None = None
    configured_effort: str | None = "high"
    provider_choices: list[str] = []
    provider_models: dict[str, list[str]] = {}

    if PI_SETTINGS_PATH.exists():
        data = json.loads(PI_SETTINGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid Pi settings in {PI_SETTINGS_PATH}")
        configured_provider = _clean_optional_text(data.get("defaultProvider"))
        configured_model = _clean_optional_text(data.get("defaultModel"))

    if PI_MODELS_PATH.exists():
        data = json.loads(PI_MODELS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid Pi models config in {PI_MODELS_PATH}")
        providers = data.get("providers")
        if isinstance(providers, dict):
            for key, value in providers.items():
                if not isinstance(key, str):
                    continue
                name = key.strip()
                if not name or name in provider_choices:
                    continue
                provider_choices.append(name)
                model_choices: list[str] = []
                if isinstance(value, dict):
                    models = value.get("models")
                    if isinstance(models, list):
                        for row in models:
                            if not isinstance(row, dict):
                                continue
                            model_id = _clean_optional_text(row.get("id"))
                            if model_id is None or model_id in model_choices:
                                continue
                            model_choices.append(model_id)
                provider_models[name] = model_choices

    fallback_provider = next(
        (name for name in provider_choices if provider_models.get(name)),
        provider_choices[0] if provider_choices else None,
    )
    selected_provider = (
        configured_provider
        if configured_provider in provider_choices
        else fallback_provider
    )
    selected_models = provider_models.get(selected_provider or "", [])
    selected_model = (
        configured_model
        if configured_model in selected_models
        else (selected_models[0] if selected_models else None)
    )

    return {
        "agent_backend": "pi",
        "model_provider": selected_provider,
        "preferred_auth_method": None,
        "provider_choice": selected_provider,
        "provider_choices": provider_choices,
        "model": selected_model,
        "models": selected_models,
        "provider_models": provider_models,
        "reasoning_effort": configured_effort,
        "reasoning_efforts": list(SUPPORTED_PI_REASONING_EFFORTS),
        "service_tier": None,
        "supports_fast": False,
    }


def _read_new_session_defaults() -> dict[str, Any]:
    codex = _read_codex_launch_defaults()
    codex["agent_backend"] = "codex"
    codex["provider_choices"] = list(codex.get("model_providers") or [])
    codex["reasoning_efforts"] = list(SUPPORTED_REASONING_EFFORTS)
    codex["supports_fast"] = True
    pi = _read_pi_launch_defaults()
    return {
        "default_backend": DEFAULT_AGENT_BACKEND,
        "backends": {
            "codex": codex,
            "pi": pi,
        },
    }


def _fallback_path_mtime(path: Path) -> float | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    except Exception:
        return 0.0
    return float(stat.st_mtime)


def _last_pi_conversation_ts(path: Path) -> float | None:
    try:
        for entry in _rollout_log._iter_jsonl_objects_reverse(path):
            if entry.get("type") != "message":
                continue
            message = entry.get("message")
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role not in {"user", "assistant", "toolResult"}:
                continue
            ts = _pi_messages._entry_ts(message)
            if ts is None:
                ts = _pi_messages._entry_ts(entry)
            if isinstance(ts, (int, float)) and math.isfinite(float(ts)) and float(ts) > 0:
                return float(ts)
    except FileNotFoundError:
        return None
    except Exception:
        return 0.0
    return None


def _resume_candidate_updated_ts(path: Path, *, agent_backend: str) -> float | None:
    backend_name = normalize_agent_backend(agent_backend)
    if backend_name == "pi":
        ts = _last_pi_conversation_ts(path)
    else:
        ts = _last_conversation_ts_from_tail(path)
    if isinstance(ts, (int, float)) and math.isfinite(float(ts)) and float(ts) > 0:
        return float(ts)
    return _fallback_path_mtime(path)


def _resume_candidate_from_log(
    log_path: Path, *, agent_backend: str = "codex"
) -> dict[str, Any] | None:
    backend_name = normalize_agent_backend(agent_backend)
    meta = _read_session_meta(log_path, agent_backend=backend_name)
    if backend_name == "codex" and _is_subagent_session_meta(meta):
        return None
    session_id = meta.get("id")
    cwd = meta.get("cwd")
    if not isinstance(session_id, str) or not session_id:
        return None
    if not isinstance(cwd, str) or not cwd:
        return None
    updated_ts = _resume_candidate_updated_ts(log_path, agent_backend=backend_name)
    if updated_ts is None:
        return None
    git_branch = ""
    if backend_name == "codex":
        git_info = meta.get("git")
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
        "agent_backend": backend_name,
    }


def _pi_native_session_dir_for_cwd(cwd: str | Path) -> Path:
    cwd_path = _safe_expanduser(Path(cwd)).resolve()
    slug = str(cwd_path).strip("/").replace("/", "-")
    return PI_NATIVE_SESSIONS_DIR / f"--{slug}--"


def _pi_new_session_file_for_cwd(cwd: str | Path) -> Path:
    now = float(_now())
    millis = int(round((now - math.floor(now)) * 1000))
    if millis >= 1000:
        now = math.floor(now) + 1.0
        millis = 0
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime(now))
    name = f"{stamp}-{millis:03d}Z_{uuid.uuid4()}.jsonl"
    return _pi_native_session_dir_for_cwd(cwd) / name


def _write_pi_session_header(
    session_path: Path,
    *,
    session_id: str,
    cwd: str,
    parent_session: str | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    thinking_level: str | None = None,
) -> None:
    session_path.parent.mkdir(parents=True, exist_ok=True)
    now = float(_now())
    millis = int(round((now - math.floor(now)) * 1000))
    if millis >= 1000:
        now = math.floor(now) + 1.0
        millis = 0
    timestamp = f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(now))}.{millis:03d}Z"
    header: dict[str, Any] = {
        "type": "session",
        "version": 3,
        "id": session_id,
        "timestamp": timestamp,
        "cwd": cwd,
    }
    if isinstance(parent_session, str) and parent_session.strip():
        header["parentSession"] = parent_session.strip()
    if isinstance(provider, str) and provider.strip():
        header["provider"] = provider.strip()
    if isinstance(model_id, str) and model_id.strip():
        header["modelId"] = model_id.strip()
    if isinstance(thinking_level, str) and thinking_level.strip():
        header["thinkingLevel"] = thinking_level.strip()
    with session_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")


def _pi_session_name_from_session_file(
    session_path: Path, *, max_scan_bytes: int = 512 * 1024
) -> str:
    try:
        objs = _read_jsonl_tail(session_path, max_scan_bytes)
    except Exception:
        return ""
    for obj in reversed(objs):
        if not isinstance(obj, dict) or obj.get("type") != "session_info":
            continue
        name = obj.get("name")
        if isinstance(name, str):
            return name.strip()
    return ""



def _pi_resume_candidate_from_session_file(session_path: Path) -> dict[str, Any] | None:
    try:
        with session_path.open("rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(obj, dict) or obj.get("type") != "session":
                    continue
                session_id = obj.get("id") or obj.get("session_id")
                cwd = obj.get("cwd")
                if not (isinstance(session_id, str) and session_id):
                    return None
                if not (isinstance(cwd, str) and cwd):
                    return None
                updated_ts = _resume_candidate_updated_ts(session_path, agent_backend="pi")
                if updated_ts is None:
                    return None
                return {
                    "session_id": session_id,
                    "cwd": cwd,
                    "session_path": str(session_path),
                    "updated_ts": updated_ts,
                    "timestamp": obj.get("timestamp"),
                    "git_branch": None,
                    "agent_backend": "pi",
                    "backend": "pi",
                    "title": _pi_session_name_from_session_file(session_path),
                }
    except OSError:
        return None
    return None


def _discover_pi_session_for_cwd(
    cwd: str, start_ts: float, *, exclude: set[Path] | None = None
) -> Path | None:
    """Auto-discover the pi session file for a PTY-wrapped pi broker.

    When pi runs inside the PTY broker (piox), pi writes its own session file
    to ~/.pi/agent/sessions/ but the broker sidecar has no session_path.
    This function finds the most recently modified session file that was
    created around/after the broker start time.

    *exclude* contains session paths already claimed by other active sessions
    so that two brokers sharing the same CWD pick different files.
    """
    session_dir = _pi_native_session_dir_for_cwd(cwd)
    if not session_dir.is_dir():
        return None
    best: Path | None = None
    best_mtime: float = 0
    for f in session_dir.glob("*.jsonl"):
        if exclude and f in exclude:
            continue
        try:
            mtime = f.stat().st_mtime
        except OSError:
            continue
        if mtime < start_ts - 10:
            continue
        if mtime > best_mtime:
            best = f
            best_mtime = mtime
    return best


def _resolve_pi_session_path(
    *,
    thread_id: str | None,
    cwd: str,
    start_ts: float,
    preferred: Path | None = None,
    exclude: set[Path] | None = None,
) -> tuple[Path | None, str | None]:
    """Resolve the Pi session file, preferring the current backend thread id.

    Returns ``(path, source)`` where ``source`` is one of ``exact``,
    ``preferred``, ``discovered``, or ``None``.
    """

    clean_thread_id = str(thread_id or "").strip()
    if preferred is not None:
        try:
            preferred_exists = preferred.exists()
        except OSError:
            preferred_exists = False
        if preferred_exists:
            if (not clean_thread_id) or (
                _read_pi_session_id(preferred) == clean_thread_id
            ):
                return preferred, "preferred"
    if clean_thread_id:
        exact = _find_session_log_for_session_id(clean_thread_id, agent_backend="pi")
        if exact is not None:
            return exact, "exact"
    if preferred is not None:
        return preferred, "preferred"
    discovered = _discover_pi_session_for_cwd(cwd, start_ts, exclude=exclude)
    if discovered is not None:
        return discovered, "discovered"
    return None, None


def _safe_path_mtime(path: Path) -> float | None:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return None


def _list_resume_candidates_for_cwd(
    cwd: str,
    *,
    limit: int = 12,
    offset: int = 0,
    backend: str | None = None,
    agent_backend: str | None = None,
) -> list[dict[str, Any]]:
    cwd2 = str(_safe_expanduser(Path(cwd)).resolve())
    backend_raw = backend if backend is not None else agent_backend
    backend2 = normalize_agent_backend(backend_raw, default="codex")
    limit2 = max(1, int(limit))
    offset2 = max(0, int(offset))
    if backend2 == "pi":
        rows: list[dict[str, Any]] = []
        session_dir = _pi_native_session_dir_for_cwd(cwd2)
        if not session_dir.exists():
            return rows
        for session_path in session_dir.glob("*.jsonl"):
            row = _pi_resume_candidate_from_session_file(session_path)
            if not isinstance(row, dict):
                continue
            if row.get("cwd") != cwd2:
                continue
            rows.append(row)
        rows.sort(key=lambda row: -float(row.get("updated_ts") or 0.0))
        return rows[offset2 : offset2 + limit2]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for log_path in _iter_session_logs(agent_backend=backend2):
        try:
            row = _resume_candidate_from_log(log_path, agent_backend=backend2)
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
    out.sort(key=lambda row: -float(row.get("updated_ts") or 0.0))
    return out[offset2 : offset2 + limit2]


def _iter_all_resume_candidates(*, limit: int = 200) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()

    ranked_rows: list[tuple[float, dict[str, Any]]] = []

    if PI_NATIVE_SESSIONS_DIR.exists():
        for session_path in PI_NATIVE_SESSIONS_DIR.glob("--*--/*.jsonl"):
            row = _pi_resume_candidate_from_session_file(session_path)
            if not isinstance(row, dict):
                continue
            session_id = row.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                continue
            key = ("pi", session_id)
            if key in seen:
                continue
            seen.add(key)
            ranked_rows.append((float(row.get("updated_ts") or 0.0), row))

    for log_path in _iter_session_logs(agent_backend="codex"):
        try:
            row = _resume_candidate_from_log(log_path, agent_backend="codex")
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        session_id = row.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            continue
        key = ("codex", session_id)
        if key in seen:
            continue
        seen.add(key)
        ranked_rows.append((float(row.get("updated_ts") or 0.0), row))

    ranked_rows.sort(key=lambda item: -item[0])
    return [row for _updated_ts, row in ranked_rows[:limit]]


def _historical_session_id(backend: str, resume_session_id: str) -> str:
    return f"history:{backend}:{resume_session_id}"


def _parse_historical_session_id(session_id: str) -> tuple[str, str] | None:
    raw = str(session_id or "").strip()
    if not raw.startswith("history:"):
        return None
    _prefix, backend, resume_session_id = (
        raw.split(":", 2) if raw.count(":") >= 2 else ("", "", "")
    )
    backend_clean = normalize_agent_backend(backend, default="codex")
    resume_clean = _clean_optional_text(resume_session_id)
    if not resume_clean:
        return None
    return backend_clean, resume_clean


def _historical_session_row(session_id: str) -> dict[str, Any] | None:
    parsed = _parse_historical_session_id(session_id)
    if parsed is None:
        return None
    backend, resume_session_id = parsed
    for row in _iter_all_resume_candidates():
        if (
            normalize_agent_backend(
                row.get("agent_backend", row.get("backend")), default="codex"
            )
            != backend
        ):
            continue
        if _clean_optional_text(row.get("session_id")) != resume_session_id:
            continue
        out = dict(row)
        out["session_id"] = _historical_session_id(backend, resume_session_id)
        out["resume_session_id"] = resume_session_id
        out["historical"] = True
        return out
    return None


def _historical_sidebar_items(
    *, live_resume_keys: set[tuple[str, str]], now_ts: float
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _iter_all_resume_candidates():
        resume_session_id = row.get("session_id")
        cwd = row.get("cwd")
        if not isinstance(resume_session_id, str) or not resume_session_id:
            continue
        if not isinstance(cwd, str) or not cwd:
            continue
        backend = normalize_agent_backend(
            row.get("agent_backend", row.get("backend")), default="codex"
        )
        live_key = (backend, resume_session_id)
        if live_key in live_resume_keys:
            continue
        updated_ts_raw = row.get("updated_ts")
        updated_ts = (
            float(updated_ts_raw)
            if isinstance(updated_ts_raw, (int, float))
            else float(now_ts)
        )
        elapsed_s = max(0.0, now_ts - updated_ts)
        time_priority = _priority_from_elapsed_seconds(elapsed_s)
        first_user_message = ""
        try:
            if backend == "pi":
                session_path_raw = row.get("session_path")
                if isinstance(session_path_raw, str) and session_path_raw:
                    first_user_message = _first_user_message_preview_from_pi_session(
                        Path(session_path_raw)
                    )
            else:
                log_path_raw = row.get("log_path")
                if isinstance(log_path_raw, str) and log_path_raw:
                    first_user_message = _first_user_message_preview_from_log(
                        Path(log_path_raw)
                    )
        except Exception:
            first_user_message = ""
        out.append(
            {
                "session_id": _historical_session_id(backend, resume_session_id),
                "runtime_id": None,
                "thread_id": resume_session_id,
                "backend": backend,
                "pid": None,
                "broker_pid": None,
                "agent_backend": backend,
                "owned": False,
                "transport": None,
                "cwd": cwd,
                "start_ts": updated_ts,
                "updated_ts": updated_ts,
                "log_path": row.get("log_path")
                if isinstance(row.get("log_path"), str)
                else row.get("session_path")
                if isinstance(row.get("session_path"), str)
                else None,
                "queue_len": 0,
                "token": None,
                "thinking": 0,
                "tools": 0,
                "system": 0,
                "harness_enabled": False,
                "harness_cooldown_minutes": HARNESS_DEFAULT_IDLE_MINUTES,
                "harness_remaining_injections": HARNESS_DEFAULT_MAX_INJECTIONS,
                "alias": "",
                "first_user_message": first_user_message,
                "files": [],
                "git_branch": row.get("git_branch"),
                "model_provider": None,
                "preferred_auth_method": None,
                "provider_choice": None,
                "model": None,
                "reasoning_effort": None,
                "service_tier": None,
                "tmux_session": None,
                "tmux_window": None,
                "priority_offset": 0.0,
                "snooze_until": None,
                "dependency_session_id": None,
                "time_priority": time_priority,
                "base_priority": time_priority,
                "final_priority": time_priority,
                "blocked": False,
                "snoozed": False,
                "busy": False,
                "historical": True,
                "resume_session_id": resume_session_id,
            }
        )
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
    return s.startswith("# AGENTS.md instructions") or s.startswith(
        "<environment_context>"
    )


def _first_user_message_preview_from_log(
    log_path: Path, *, max_scan_bytes: int = 256 * 1024
) -> str:
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
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") == "message":
                    text = _pi_user_text(obj) or ""
                elif obj.get("type") == "response_item":
                    payload = obj.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    if (
                        payload.get("type") != "message"
                        or payload.get("role") != "user"
                    ):
                        continue
                    text = _user_message_text(payload)
                else:
                    continue
                if not text or _is_scaffold_user_text(text):
                    continue
                return _resume_preview_from_text(text)
    except FileNotFoundError:
        return ""
    return ""


def _first_user_message_preview_from_pi_session(
    session_path: Path, *, max_scan_bytes: int = 256 * 1024
) -> str:
    try:
        with session_path.open("rb") as f:
            total = 0
            for raw in f:
                total += len(raw)
                if total > max_scan_bytes:
                    break
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(obj, dict) or obj.get("type") != "message":
                    continue
                payload = obj.get("message")
                if not isinstance(payload, dict):
                    payload = obj.get("payload")
                if not isinstance(payload, dict):
                    payload = obj
                if payload.get("role") != "user":
                    continue
                text = _user_message_text(payload)
                if not text:
                    content = payload.get("content")
                    if isinstance(content, str):
                        text = content.strip()
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


def _extract_delivery_messages(objs: list[dict[str, Any]]) -> list[Any]:
    return _rollout_log._extract_delivery_messages(objs)


def _read_jsonl_tail(path: Path, max_bytes: int) -> list[dict[str, Any]]:
    return _rollout_log._read_jsonl_tail(path, max_bytes)


def _read_chat_events_from_tail(
    log_path: Path,
    min_events: int = 120,
    max_scan_bytes: int = 128 * 1024 * 1024,
) -> list[dict[str, Any]]:
    return _rollout_log._read_chat_events_from_tail(
        log_path, min_events=min_events, max_scan_bytes=max_scan_bytes
    )


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
    max_scan_bytes: int | None = None,
) -> float | None:
    return _rollout_log._last_conversation_ts_from_tail(
        log_path, max_scan_bytes=max_scan_bytes
    )


def _compute_idle_from_log(
    path: Path, max_scan_bytes: int = 8 * 1024 * 1024
) -> bool | None:
    return _rollout_log._compute_idle_from_log(path, max_scan_bytes=max_scan_bytes)


def _last_chat_role_ts_from_tail(
    path: Path,
    *,
    max_scan_bytes: int,
) -> tuple[str, float] | None:
    return _rollout_log._last_chat_role_ts_from_tail(
        path, max_scan_bytes=max_scan_bytes
    )


def _session_file_activity_ts(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    try:
        ts = float(path.stat().st_mtime)
    except OSError:
        return None
    if not math.isfinite(ts) or ts <= 0:
        return None
    return ts


def _touch_session_file(path: Path | None) -> float | None:
    """Best-effort touch used to mark local prompt activity for Pi sessions."""
    if path is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    except OSError:
        return _session_file_activity_ts(path)
    return _session_file_activity_ts(path)


@dataclass
class Session:
    session_id: str
    thread_id: str
    broker_pid: int
    codex_pid: int
    agent_backend: str
    owned: bool
    start_ts: float
    cwd: str
    log_path: Path | None
    sock_path: Path
    session_path: Path | None = None
    backend: str = "codex"
    busy: bool = False
    queue_len: int = 0
    token: dict[str, Any] | None = None
    last_turn_id: str | None = None
    last_chat_ts: float | None = None
    last_chat_history_scanned: bool = False
    meta_thinking: int = 0
    meta_tools: int = 0
    meta_system: int = 0
    meta_log_off: int = 0
    chat_index_events: list[dict[str, Any]] = field(default_factory=list)
    chat_index_scan_bytes: int = 0
    chat_index_scan_complete: bool = False
    chat_index_log_off: int = 0
    delivery_log_off: int = 0
    idle_cache_log_off: int = -1
    idle_cache_value: bool | None = None
    queue_idle_since: float | None = None
    model_provider: str | None = None
    preferred_auth_method: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None
    transport: str | None = None
    supports_live_ui: bool | None = None
    ui_protocol_version: int | None = None
    tmux_session: str | None = None
    tmux_window: str | None = None
    resume_session_id: str | None = None
    title: str | None = None
    first_user_message: str | None = None
    pi_idle_activity_ts: float | None = None
    pi_busy_activity_floor: float | None = None
    pi_session_path_discovered: bool = False
    pi_attention_scan_activity_ts: float | None = None
    bridge_transport_state: str = "unknown"
    bridge_transport_error: str | None = None
    bridge_transport_checked_ts: float = 0.0


@dataclass
class BridgeOutboundRequest:
    request_id: str
    runtime_id: str
    durable_session_id: str
    text: str
    created_ts: float
    state: str = "queued"
    attempts: int = 0
    last_error: str | None = None
    last_attempt_ts: float = 0.0


def _session_supports_live_pi_ui(session: Session) -> bool:
    if session.backend != "pi":
        return False
    transport = (session.transport or "").strip().lower()
    if transport != "pi-rpc":
        return False
    if session.supports_live_ui is not True:
        return False
    if (
        not isinstance(session.ui_protocol_version, int)
        or session.ui_protocol_version < 1
    ):
        return False
    return True


def _is_attention_worthy_session_event(event: dict[str, Any]) -> bool:
    if not isinstance(event, dict) or event.get("display") is False:
        return False
    event_type = str(event.get("type") or "").strip()
    return bool(
        event.get("role") in {"user", "assistant"}
        or bool(event.get("is_error"))
        or event_type == "ask_user"
    )



def _attention_updated_ts_from_events(events: list[dict[str, Any]]) -> float | None:
    latest_ts: float | None = None
    for event in events:
        if not _is_attention_worthy_session_event(event):
            continue
        ts = event.get("ts")
        if not isinstance(ts, (int, float)) or not math.isfinite(float(ts)):
            continue
        latest_ts = float(ts) if latest_ts is None else max(latest_ts, float(ts))
    return latest_ts



def _last_attention_ts_from_pi_tail(
    session_path: Path | None, *, max_scan_bytes: int = 8 * 1024 * 1024
) -> float | None:
    if session_path is None or not session_path.exists():
        return None
    try:
        events, _token_update, _off, _scan_bytes, _complete, _diag = (
            _pi_messages.read_pi_message_tail_snapshot(
                session_path,
                min_events=80,
                initial_scan_bytes=256 * 1024,
                max_scan_bytes=max_scan_bytes,
            )
        )
    except Exception:
        return None
    if any(_is_attention_worthy_session_event(event) for event in events):
        activity_ts = _session_file_activity_ts(session_path)
        if activity_ts is not None:
            return activity_ts
    return _attention_updated_ts_from_events(events)



def _display_updated_ts(s: Session) -> float:
    return (
        float(s.last_chat_ts)
        if isinstance(s.last_chat_ts, (int, float))
        else float(s.start_ts)
    )


def _session_row_dedupe_key(row: dict[str, Any]) -> str:
    if row.get("historical"):
        backend = normalize_agent_backend(
            row.get("agent_backend"), default=str(row.get("backend", "codex"))
        )
        return f"historical:{backend}:{str(row.get('session_id', '')).strip()}"
    thread_id = str(row.get("thread_id", "")).strip()
    if thread_id:
        backend = normalize_agent_backend(
            row.get("agent_backend"), default=str(row.get("backend", "codex"))
        )
        return f"thread:{backend}:{thread_id}"
    return f"session:{str(row.get('session_id', '')).strip()}"


def _display_source_path(s: Session) -> str | None:
    if s.backend == "pi":
        return str(s.session_path) if s.session_path is not None else None
    return str(s.log_path) if s.log_path is not None else None


def _durable_session_id_for_live_session(s: Session) -> str:
    return _clean_optional_text(s.thread_id) or _clean_optional_text(s.session_id) or ""


def _display_pi_busy(s: Session, *, broker_busy: bool) -> bool:
    if not broker_busy:
        activity_ts = _session_file_activity_ts(s.session_path)
        if activity_ts is not None:
            s.pi_idle_activity_ts = activity_ts
        s.pi_busy_activity_floor = None
        return False
    session_path = s.session_path
    if session_path is None or (not session_path.exists()):
        return True
    activity_ts = _session_file_activity_ts(session_path)
    if activity_ts is None:
        return True
    floor = s.pi_busy_activity_floor
    if isinstance(floor, (int, float)) and activity_ts <= float(floor):
        return True
    idle_marker = s.pi_idle_activity_ts
    if isinstance(idle_marker, (int, float)) and activity_ts <= float(idle_marker):
        return False
    idle = _pi_messages.is_pi_session_idle(session_path)
    if idle is True:
        s.pi_idle_activity_ts = activity_ts
        s.pi_busy_activity_floor = None
        return False
    if idle is False:
        s.pi_idle_activity_ts = None
    return True


def _validated_session_state(state: dict[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise ValueError("invalid broker state response")
    _state_busy_value(state)
    _state_queue_len_value(state)
    return state


def _state_busy_value(state: dict[str, Any]) -> bool:
    busy_raw = state.get("busy")
    if not isinstance(busy_raw, bool):
        raise ValueError("invalid busy from broker state response")
    return busy_raw


def _state_queue_len_value(state: dict[str, Any]) -> int:
    queue_len_raw = state.get("queue_len")
    if type(queue_len_raw) is not int or int(queue_len_raw) < 0:
        raise ValueError("invalid queue_len from broker state response")
    return int(queue_len_raw)


def _display_session_busy(
    manager: "SessionManager", session_id: str, s: Session, state: dict[str, Any]
) -> tuple[bool, bool]:
    broker_busy = _state_busy_value(state)
    busy = (
        _display_pi_busy(s, broker_busy=broker_busy)
        if s.backend == "pi"
        else broker_busy
    )
    if s.backend != "pi" and s.log_path is not None and s.log_path.exists():
        idle_val = manager.idle_from_log(session_id)
        busy = broker_busy or (not bool(idle_val))
    return bool(busy), broker_busy


def _resolved_session_run_settings(
    s: Session,
) -> tuple[str | None, str | None, str | None, str | None]:
    model_provider = s.model_provider
    preferred_auth_method = s.preferred_auth_method
    model = s.model
    reasoning_effort = s.reasoning_effort
    if (
        (model_provider is None or model is None or reasoning_effort is None)
        and s.backend == "pi"
        and s.session_path is not None
        and s.session_path.exists()
    ):
        pi_provider, pi_model, pi_effort = _read_pi_run_settings(s.session_path)
        if model_provider is None:
            model_provider = pi_provider
        if model is None:
            model = pi_model
        if reasoning_effort is None:
            reasoning_effort = pi_effort
    if (
        (model_provider is None or model is None or reasoning_effort is None)
        and s.log_path is not None
        and s.log_path.exists()
    ):
        log_provider, log_model, log_effort = _read_run_settings_from_log(
            s.log_path, agent_backend=s.agent_backend
        )
        if model_provider is None:
            model_provider = log_provider
        if model is None:
            model = log_model
        if reasoning_effort is None:
            reasoning_effort = log_effort
    return model_provider, preferred_auth_method, model, reasoning_effort


def _resolved_session_token(
    s: Session, token: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    if isinstance(token, dict):
        return token
    if isinstance(s.token, dict):
        return s.token
    source_path: Path | None = None
    if s.backend == "pi" and s.session_path is not None and s.session_path.exists():
        source_path = s.session_path
    elif s.log_path is not None and s.log_path.exists():
        source_path = s.log_path
    if source_path is None:
        return None
    token_update = _rollout_log._find_latest_token_update(source_path)
    return token_update if isinstance(token_update, dict) else None


def _session_context_usage_payload(
    s: Session, token_val: dict[str, Any] | None
) -> dict[str, Any] | None:
    return _session_payloads.session_context_usage_payload(s, token_val)



def _session_turn_timing_payload(
    s: Session,
    events: list[dict[str, Any]],
    *,
    busy: bool,
) -> dict[str, Any] | None:
    return _session_payloads.session_turn_timing_payload(s, events, busy=busy)


def _session_diagnostics_payload(
    manager: "SessionManager", session_id: str, s: Session, state: dict[str, Any]
) -> dict[str, Any]:
    return _session_payloads.session_diagnostics_payload(manager, session_id, s, state)


def _session_workspace_payload(
    manager: "SessionManager", session_id: str
) -> dict[str, Any]:
    return _session_payloads.session_workspace_payload(manager, session_id)


def _session_live_payload(
    manager: "SessionManager",
    session_id: str,
    *,
    offset: int = 0,
    live_offset: int = 0,
    bridge_offset: int = 0,
    requests_version: str | None = None,
) -> dict[str, Any]:
    return _session_live_payloads.session_live_payload(
        manager,
        session_id,
        offset=offset,
        live_offset=live_offset,
        bridge_offset=bridge_offset,
        requests_version=requests_version,
    )


def _pi_live_messages_payload(
    manager: "SessionManager", session: Session, *, offset: int = 0
) -> dict[str, Any]:
    return _session_live_payloads.pi_live_messages_payload(manager, session, offset=offset)


def _merge_pi_live_message_events(
    durable_events: list[dict[str, Any]], streamed_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return _session_live_payloads.merge_pi_live_message_events(durable_events, streamed_events)


def _supports_web_control(meta: dict[str, Any]) -> bool:
    return meta.get("supports_web_control") is True


class SessionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Sidebar should reflect broker-visible live sessions only.
        self._include_historical_sessions = False
        self._bad_sidecars: dict[str, tuple[bool, int, int]] = {}
        self._sessions: dict[str, Session] = {}
        self._stop = threading.Event()
        self._last_discover_ts = 0.0
        self._last_session_catalog_refresh_ts = 0.0
        self._page_state_db = PageStateDB(PAGE_STATE_DB_PATH)
        if self._page_state_db.is_empty():
            import_legacy_app_dir_to_db(source_app_dir=APP_DIR, db_path=PAGE_STATE_DB_PATH)
        self._harness: dict[str, dict[str, Any]] = {}
        self._aliases: dict[SessionStateKey, str] = {}
        self._sidebar_meta: dict[SessionStateKey, dict[str, Any]] = {}
        self._hidden_sessions: set[str] = set()
        self._files: dict[SessionStateKey, list[str]] = {}
        self._queues: dict[SessionStateKey, list[str]] = {}
        self._bridge_events: dict[str, list[dict[str, Any]]] = {}
        self._bridge_event_offsets: dict[str, int] = {}
        self._outbound_requests: dict[str, list[BridgeOutboundRequest]] = {}
        self._queue_wakeup = threading.Event()
        self._pi_commands_cache: dict[str, dict[str, Any]] = {}
        self._recent_cwds: dict[str, float] = {}
        self._cwd_groups: dict[str, dict[str, Any]] = {}
        self._prune_missing_workspace_dirs = True
        self._sidebar_state = SidebarStateFacade(self)
        self._harness_last_injected: dict[str, float] = {}
        self._harness_last_injected_scope: dict[str, float] = {}
        self._load_harness()
        self._load_aliases()
        self._load_sidebar_meta()
        self._load_hidden_sessions()
        self._load_files()
        self._load_queues()
        self._load_recent_cwds()
        self._load_cwd_groups()
        self._voice_push = VoicePushCoordinator(
            app_dir=APP_DIR,
            stop_event=self._stop,
            settings_path=VOICE_SETTINGS_PATH,
            subscriptions_path=PUSH_SUBSCRIPTIONS_PATH,
            delivery_ledger_path=DELIVERY_LEDGER_PATH,
            vapid_private_key_path=VAPID_PRIVATE_KEY_PATH,
            page_state_db=self._page_state_db,
            publish_callback=_voice_push_publish_callback,
        )
        self._discover_existing(force=True, skip_invalid_sidecars=True)
        self._refresh_durable_session_catalog(force=True)
        self._harness_thr = threading.Thread(
            target=self._harness_loop, name="harness", daemon=True
        )
        self._harness_thr.start()
        self._queue_thr = threading.Thread(
            target=self._queue_loop, name="queue", daemon=True
        )
        self._queue_thr.start()
        self._voice_push_scan_thr = threading.Thread(
            target=self._voice_push_scan_loop, name="voice-push-scan", daemon=True
        )
        self._voice_push_scan_thr.start()

    def _sidebar_state_facade(self) -> SidebarStateFacade:
        facade = getattr(self, "_sidebar_state", None)
        if isinstance(facade, SidebarStateFacade):
            return facade
        facade = SidebarStateFacade(self)
        self._sidebar_state = facade
        return facade

    def stop(self) -> None:
        self._stop.set()
        EVENT_HUB.close()

    def _page_state_ref_for_session(self, session: Session) -> SessionRef | None:
        durable_id = _clean_optional_text(session.thread_id) or _clean_optional_text(session.session_id)
        if durable_id is None:
            return None
        backend = normalize_agent_backend(session.agent_backend, default=session.backend or "codex")
        return backend, durable_id

    def _durable_session_id_for_session(self, session: Session) -> str:
        ref = self._page_state_ref_for_session(session)
        if ref is not None:
            return ref[1]
        return str(session.session_id)

    def _runtime_session_id_for_identifier(self, session_id: str) -> str | None:
        target = _clean_optional_text(session_id)
        if target is None:
            return None
        with self._lock:
            if target in self._sessions:
                return target
            matches: list[tuple[float, str]] = []
            for runtime_id, session in self._sessions.items():
                ref = self._page_state_ref_for_session(session)
                if ref is not None and ref[1] == target:
                    matches.append((float(session.start_ts or 0.0), runtime_id))
                    continue
                thread_id = _clean_optional_text(session.thread_id)
                if thread_id == target:
                    matches.append((float(session.start_ts or 0.0), runtime_id))
            if not matches:
                return None
            matches.sort(key=lambda item: (-item[0], item[1]))
            return matches[0][1]

    def _durable_session_id_for_identifier(self, session_id: str) -> str | None:
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is not None:
            with self._lock:
                session = self._sessions.get(runtime_id)
            if session is not None:
                return self._durable_session_id_for_session(session)
        target = _clean_optional_text(session_id)
        return target if target is not None else None

    def _append_bridge_event(self, durable_session_id: str, event: dict[str, Any]) -> dict[str, Any]:
        key = _clean_optional_text(durable_session_id)
        if key is None:
            raise ValueError("durable session id required")
        with self._lock:
            offsets = getattr(self, "_bridge_event_offsets", None)
            if not isinstance(offsets, dict):
                self._bridge_event_offsets = {}
                offsets = self._bridge_event_offsets
            rows_by_session = getattr(self, "_bridge_events", None)
            if not isinstance(rows_by_session, dict):
                self._bridge_events = {}
                rows_by_session = self._bridge_events
            next_offset = int(offsets.get(key, 0)) + 1
            offsets[key] = next_offset
            stamped = dict(event)
            stamped["event_id"] = str(stamped.get("event_id") or f"bridge:{key}:{next_offset}")
            stamped["ts"] = float(stamped.get("ts") or time.time())
            rows_by_session.setdefault(key, []).append({"offset": next_offset, "event": stamped})
            rows = rows_by_session[key]
            if len(rows) > 64:
                rows_by_session[key] = rows[-64:]
        _publish_session_live_invalidate(key, reason="bridge_event")
        return stamped

    def _bridge_events_since(self, durable_session_id: str, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        key = _clean_optional_text(durable_session_id)
        if key is None:
            return [], max(0, int(offset))
        with self._lock:
            rows_by_session = getattr(self, "_bridge_events", None)
            offsets = getattr(self, "_bridge_event_offsets", None)
            rows = list(rows_by_session.get(key, [])) if isinstance(rows_by_session, dict) else []
            latest = int(offsets.get(key, 0)) if isinstance(offsets, dict) else 0
        since = max(0, int(offset))
        events: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if int(row.get("offset", 0) or 0) <= since:
                continue
            event = row.get("event")
            if isinstance(event, dict):
                events.append(dict(event))
        return events, latest

    def _set_bridge_transport_state(
        self,
        runtime_id: str,
        *,
        state: str,
        error: str | None = None,
        checked_ts: float | None = None,
    ) -> None:
        publish = False
        durable_session_id: str | None = None
        with self._lock:
            session = self._sessions.get(runtime_id)
            if session is None:
                return
            next_error = _clean_optional_text(error)
            publish = (
                session.bridge_transport_state != state
                or session.bridge_transport_error != next_error
            )
            session.bridge_transport_state = state
            session.bridge_transport_error = next_error
            session.bridge_transport_checked_ts = float(checked_ts if checked_ts is not None else time.time())
            durable_session_id = self._durable_session_id_for_session(session)
        if publish and durable_session_id is not None:
            _publish_session_transport_invalidate(
                durable_session_id,
                runtime_id=runtime_id,
                reason="transport_state",
            )
            _publish_session_live_invalidate(
                durable_session_id,
                runtime_id=runtime_id,
                reason="transport_state",
            )

    def _probe_bridge_transport(self, session_id: str, *, force_rpc: bool = False) -> tuple[str, str | None]:
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is None:
            return "dead", "unknown session"
        with self._lock:
            session = self._sessions.get(runtime_id)
        if session is None:
            return "dead", "unknown session"
        processes_dead = (not _pid_alive(session.broker_pid)) and (not _pid_alive(session.codex_pid))
        now = time.time()
        last_checked = float(session.bridge_transport_checked_ts or 0.0)
        if (
            not force_rpc
            and session.bridge_transport_state in {"alive", "degraded"}
            and (now - last_checked) < BRIDGE_TRANSPORT_PROBE_STALE_SECONDS
        ):
            return session.bridge_transport_state, session.bridge_transport_error
        try:
            resp = self._sock_call(
                session.sock_path,
                {"cmd": "state"},
                timeout_s=BRIDGE_TRANSPORT_RPC_TIMEOUT_SECONDS,
            )
            if not isinstance(resp, dict):
                raise ValueError("invalid state probe response")
        except Exception as exc:
            if processes_dead:
                self._set_bridge_transport_state(runtime_id, state="dead", error="broker exited", checked_ts=now)
                return "dead", "broker exited"
            error = str(exc).strip() or type(exc).__name__
            self._set_bridge_transport_state(runtime_id, state="degraded", error=error, checked_ts=now)
            return "degraded", error
        self._set_bridge_transport_state(runtime_id, state="alive", error=None, checked_ts=now)
        return "alive", None

    def _enqueue_outbound_request(self, runtime_id: str, text: str) -> BridgeOutboundRequest:
        with self._lock:
            session = self._sessions.get(runtime_id)
            if session is None:
                raise KeyError("unknown session")
            durable_session_id = self._durable_session_id_for_session(session)
            requests_by_runtime = getattr(self, "_outbound_requests", None)
            if not isinstance(requests_by_runtime, dict):
                self._outbound_requests = {}
                requests_by_runtime = self._outbound_requests
            request = BridgeOutboundRequest(
                request_id=f"bridge-send-{uuid.uuid4().hex}",
                runtime_id=runtime_id,
                durable_session_id=durable_session_id,
                text=str(text),
                created_ts=time.time(),
            )
            requests_by_runtime.setdefault(runtime_id, []).append(request)
        queue_wakeup = getattr(self, "_queue_wakeup", None)
        if isinstance(queue_wakeup, threading.Event):
            queue_wakeup.set()
        return request

    def _fail_outbound_request(self, request: BridgeOutboundRequest, error: str) -> None:
        event = {
            "type": "pi_event",
            "summary": "Bridge send failed",
            "text": f"Bridge could not deliver the queued prompt: {error}\n\nOriginal prompt:\n{request.text}",
            "is_error": True,
            "request_id": request.request_id,
            "request_state": "failed",
            "pending_text": request.text,
            "ts": time.time(),
        }
        self._append_bridge_event(request.durable_session_id, event)

    def _mark_outbound_request_buffered_for_compaction(
        self, request: BridgeOutboundRequest
    ) -> None:
        if request.state == "buffered":
            return
        request.state = "buffered"
        event = {
            "type": "pi_event",
            "summary": "Bridge buffered prompt during compaction",
            "text": "Waiting for Pi compaction to finish before delivering this prompt.\n\nQueued prompt:\n"
            f"{request.text}",
            "request_id": request.request_id,
            "request_state": "buffered",
            "pending_text": request.text,
            "ts": time.time(),
        }
        self._append_bridge_event(request.durable_session_id, event)

    def _maybe_drain_outbound_request(self, runtime_id: str) -> bool:
        with self._lock:
            requests_by_runtime = getattr(self, "_outbound_requests", None)
            session = self._sessions.get(runtime_id)
            queue = requests_by_runtime.get(runtime_id) if isinstance(requests_by_runtime, dict) else None
            request = queue[0] if session is not None and isinstance(queue, list) and queue else None
        if session is None or request is None:
            return False
        if request.runtime_id != runtime_id:
            with self._lock:
                queue2 = self._outbound_requests.get(runtime_id)
                if isinstance(queue2, list) and queue2 and queue2[0] is request:
                    queue2.pop(0)
            self._fail_outbound_request(request, "stale runtime")
            return True
        state, transport_error = self._probe_bridge_transport(runtime_id)
        if state == "dead":
            with self._lock:
                queue2 = self._outbound_requests.get(runtime_id)
                if isinstance(queue2, list) and queue2 and queue2[0] is request:
                    queue2.pop(0)
                    if not queue2:
                        self._outbound_requests.pop(runtime_id, None)
            self._fail_outbound_request(request, transport_error or "broker exited")
            return True
        try:
            st = self.get_state(runtime_id)
        except Exception as exc:
            request.attempts += 1
            request.last_attempt_ts = time.time()
            request.last_error = str(exc).strip() or type(exc).__name__
            if request.attempts >= BRIDGE_OUTBOUND_FAILURE_MAX_ATTEMPTS or (time.time() - request.created_ts) >= BRIDGE_OUTBOUND_FAILURE_MAX_AGE_SECONDS:
                with self._lock:
                    queue2 = self._outbound_requests.get(runtime_id)
                    if isinstance(queue2, list) and queue2 and queue2[0] is request:
                        queue2.pop(0)
                        if not queue2:
                            self._outbound_requests.pop(runtime_id, None)
                self._fail_outbound_request(request, request.last_error)
                return True
            return False
        if not isinstance(st, dict):
            return False
        if bool(st.get("isCompacting")):
            self._mark_outbound_request_buffered_for_compaction(request)
            return False
        if bool(st.get("busy")) or int(st.get("queue_len") or 0) > 0:
            return False
        request.state = "sending"
        request.attempts += 1
        request.last_attempt_ts = time.time()
        try:
            resp = self._sock_call(session.sock_path, {"cmd": "send", "text": request.text}, timeout_s=1.0)
        except Exception as exc:
            request.last_error = str(exc).strip() or type(exc).__name__
            state2, transport_error2 = self._probe_bridge_transport(runtime_id, force_rpc=True)
            if state2 == "dead" or request.attempts >= BRIDGE_OUTBOUND_FAILURE_MAX_ATTEMPTS or (time.time() - request.created_ts) >= BRIDGE_OUTBOUND_FAILURE_MAX_AGE_SECONDS:
                with self._lock:
                    queue2 = self._outbound_requests.get(runtime_id)
                    if isinstance(queue2, list) and queue2 and queue2[0] is request:
                        queue2.pop(0)
                        if not queue2:
                            self._outbound_requests.pop(runtime_id, None)
                self._fail_outbound_request(request, transport_error2 or request.last_error)
                return True
            return False
        error = resp.get("error") if isinstance(resp, dict) else None
        if isinstance(error, str) and error:
            with self._lock:
                queue2 = self._outbound_requests.get(runtime_id)
                if isinstance(queue2, list) and queue2 and queue2[0] is request:
                    queue2.pop(0)
                    if not queue2:
                        self._outbound_requests.pop(runtime_id, None)
            self._fail_outbound_request(request, error)
            return True
        with self._lock:
            queue2 = self._outbound_requests.get(runtime_id)
            if isinstance(queue2, list) and queue2 and queue2[0] is request:
                queue2.pop(0)
                if not queue2:
                    self._outbound_requests.pop(runtime_id, None)
            session2 = self._sessions.get(runtime_id)
            if session2 is not None:
                if isinstance(resp, dict) and isinstance(resp.get("busy"), bool):
                    session2.busy = _state_busy_value(resp)
                if isinstance(resp, dict):
                    queue_len_raw = resp.get("queue_len")
                    if type(queue_len_raw) is int and int(queue_len_raw) >= 0:
                        session2.queue_len = _state_queue_len_value(resp)
                session2.pi_idle_activity_ts = None
                if session2.backend == "pi":
                    activity_ts = _touch_session_file(session2.session_path)
                    session2.pi_busy_activity_floor = activity_ts if session2.busy else None
                else:
                    session2.pi_busy_activity_floor = None
        return True

    def _catalog_record_for_ref(self, ref: SessionRef) -> DurableSessionRecord | None:
        backend, durable_session_id = ref
        source_path = _find_session_log_for_session_id(
            durable_session_id, agent_backend=backend
        )
        if source_path is None or (not source_path.exists()):
            return None
        if backend == "pi":
            row = _pi_resume_candidate_from_session_file(source_path)
            if not isinstance(row, dict):
                return None
            title = _clean_optional_text(row.get("title")) or ""
            first_user_message = _clean_optional_text(
                _first_user_message_preview_from_pi_session(source_path)
            )
        else:
            row = _resume_candidate_from_log(source_path, agent_backend=backend)
            if not isinstance(row, dict):
                return None
            title = _clean_optional_text(row.get("title")) or ""
            first_user_message = _clean_optional_text(
                _first_user_message_preview_from_log(source_path)
            )
        cwd = _clean_optional_text(row.get("cwd"))
        updated_ts = row.get("updated_ts")
        updated_at = float(updated_ts) if isinstance(updated_ts, (int, float)) else _safe_path_mtime(source_path)
        return DurableSessionRecord(
            backend=backend,
            session_id=durable_session_id,
            cwd=cwd,
            source_path=str(source_path),
            title=title,
            first_user_message=first_user_message,
            created_at=updated_at,
            updated_at=updated_at,
        )

    def _refresh_durable_session_catalog(self, *, force: bool = False) -> None:
        db = getattr(self, "_page_state_db", None)
        if not isinstance(db, PageStateDB):
            return
        now = time.time()
        last_refresh = float(getattr(self, "_last_session_catalog_refresh_ts", 0.0) or 0.0)
        if (not force) and (now - last_refresh) < 5.0:
            return
        refs = set(db.known_session_refs())
        with self._lock:
            for session in self._sessions.values():
                ref = self._page_state_ref_for_session(session)
                if ref is not None:
                    refs.add(ref)
        existing = db.load_sessions()
        rows: dict[SessionRef, DurableSessionRecord] = {}
        for ref in sorted(refs):
            record = self._catalog_record_for_ref(ref)
            if record is None:
                record = existing.get(ref)
            if record is not None:
                rows[ref] = record
        db.save_sessions(rows)
        self._last_session_catalog_refresh_ts = now

    def _page_state_ref_for_session_id(self, session_id: str) -> SessionRef | None:
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is not None:
            with self._lock:
                session = self._sessions.get(runtime_id)
            if session is not None:
                return self._page_state_ref_for_session(session)
        parsed = _parse_historical_session_id(session_id)
        if parsed is not None:
            backend, durable_id = parsed
            return backend, durable_id
        target = _clean_optional_text(session_id)
        db = getattr(self, "_page_state_db", None)
        if target is not None and isinstance(db, PageStateDB):
            matches = [ref for ref in db.known_session_refs() if ref[1] == target]
            if len(matches) == 1:
                return matches[0]
        return None

    def _persist_durable_session_record(self, row: DurableSessionRecord) -> None:
        db = getattr(self, "_page_state_db", None)
        if isinstance(db, PageStateDB):
            db.upsert_session(row)

    def _delete_durable_session_record(self, ref: SessionRef | None) -> None:
        db = getattr(self, "_page_state_db", None)
        if ref is not None and isinstance(db, PageStateDB):
            db.delete_session(ref)

    def _finalize_pending_pi_spawn(
        self,
        *,
        spawn_nonce: str,
        durable_session_id: str,
        cwd: str,
        session_path: Path,
        proc: subprocess.Popen[bytes] | None = None,
    ) -> None:
        ref = ("pi", durable_session_id)
        try:
            if proc is not None:
                _wait_or_raise(proc, label="pi broker", timeout_s=0.25)
            meta = _wait_for_spawned_broker_meta(spawn_nonce)
            live_session_id = _clean_optional_text(meta.get("session_id")) or durable_session_id
            if live_session_id != durable_session_id:
                raise RuntimeError(
                    f"pi session id mismatch: expected {durable_session_id}, got {live_session_id}"
                )
            self._discover_existing(force=True, skip_invalid_sidecars=True)
            self._refresh_durable_session_catalog(force=True)
            db = getattr(self, "_page_state_db", None)
            current = (
                db.load_sessions().get(ref)
                if isinstance(db, PageStateDB)
                else None
            )
            self._persist_durable_session_record(
                DurableSessionRecord(
                    backend="pi",
                    session_id=durable_session_id,
                    cwd=(current.cwd if current is not None else cwd),
                    source_path=(current.source_path if current is not None else str(session_path)),
                    title=current.title if current is not None else None,
                    first_user_message=current.first_user_message if current is not None else None,
                    created_at=(current.created_at if current is not None else _safe_path_mtime(session_path)),
                    updated_at=(current.updated_at if current is not None else _safe_path_mtime(session_path)),
                    pending_startup=False,
                )
            )
            _publish_sessions_invalidate(reason="session_created")
        except Exception:
            self._delete_durable_session_record(ref)
            self._clear_deleted_session_state(durable_session_id)
            _publish_sessions_invalidate(reason="session_removed")

    def _persist_session_ui_state(self) -> None:
        self._sidebar_state_facade().persist_session_ui_state()

    def _persist_files(self) -> None:
        db = getattr(self, "_page_state_db", None)
        if db is None:
            return
        with self._lock:
            files_src = dict(self._files)
        rows: dict[SessionRef, list[str]] = {}
        for key, value in files_src.items():
            ref = key if isinstance(key, tuple) and len(key) == 2 else self._page_state_ref_for_session_id(str(key))
            if ref is None or not isinstance(value, list):
                continue
            rows[ref] = [row for row in value if isinstance(row, str) and row.strip()]
        db.save_files(rows)

    def _persist_queues(self) -> None:
        db = getattr(self, "_page_state_db", None)
        if db is None:
            return
        with self._lock:
            queues_src = dict(self._queues)
        rows: dict[SessionRef, list[str]] = {}
        for key, value in queues_src.items():
            ref = key if isinstance(key, tuple) and len(key) == 2 else self._page_state_ref_for_session_id(str(key))
            if ref is None or not isinstance(value, list):
                continue
            rows[ref] = [row for row in value if isinstance(row, str) and row.strip()]
        db.save_queues(rows)

    def _persist_recent_cwds(self) -> None:
        db = getattr(self, "_page_state_db", None)
        if db is None:
            return
        with self._lock:
            recent_cwds = dict(self._recent_cwds)
        db.save_recent_cwds(recent_cwds)

    def _persist_cwd_groups(self) -> None:
        db = getattr(self, "_page_state_db", None)
        if db is None:
            return
        with self._lock:
            cwd_groups = dict(self._cwd_groups)
        db.save_cwd_groups(cwd_groups)

    def _reset_log_caches(self, s: Session, *, meta_log_off: int) -> None:
        s.meta_thinking = 0
        s.meta_tools = 0
        s.meta_system = 0
        s.last_chat_ts = None
        s.last_chat_history_scanned = False
        s.pi_idle_activity_ts = None
        s.pi_busy_activity_floor = None
        s.meta_log_off = int(meta_log_off)
        s.chat_index_events = []
        s.chat_index_scan_bytes = 0
        s.chat_index_scan_complete = False
        s.chat_index_log_off = int(meta_log_off)
        s.delivery_log_off = int(meta_log_off)
        s.idle_cache_log_off = -1
        s.idle_cache_value = None
        s.queue_idle_since = None
        s.model_provider = None
        s.preferred_auth_method = None
        s.model = None
        s.reasoning_effort = None
        s.service_tier = None
        s.first_user_message = None

    def _session_source_changed(
        self, s: Session, *, log_path: Path | None, session_path: Path | None
    ) -> bool:
        if s.log_path != log_path:
            return True
        if s.backend == "pi" and s.session_path != session_path:
            return True
        return False

    def _claimed_pi_session_paths(self, *, exclude_sid: str = "") -> set[Path]:
        """Return session_path values already assigned to active pi sessions."""
        with self._lock:
            out: set[Path] = set()
            for s in self._sessions.values():
                if (
                    s.backend == "pi"
                    and s.session_path is not None
                    and s.session_id != exclude_sid
                ):
                    out.add(s.session_path)
            return out

    def _apply_session_source(
        self, s: Session, *, log_path: Path | None, session_path: Path | None
    ) -> None:
        # For PTY-wrapped pi sessions, preserve a previously discovered
        # session_path when the sidecar doesn't provide one.
        if s.backend == "pi" and session_path is None and s.session_path is not None:
            session_path = s.session_path
        source_changed = self._session_source_changed(
            s, log_path=log_path, session_path=session_path
        )
        s.log_path = log_path
        s.session_path = session_path
        if source_changed:
            log_off = (
                int(log_path.stat().st_size)
                if log_path is not None and log_path.exists()
                else 0
            )
            self._reset_log_caches(s, meta_log_off=log_off)

    def _session_run_settings(
        self,
        *,
        meta: dict[str, Any],
        log_path: Path | None,
        backend: str | None = None,
        agent_backend: str | None = None,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        backend_name = normalize_agent_backend(
            backend if backend is not None else agent_backend, default="codex"
        )
        model_provider = _clean_optional_text(meta.get("model_provider"))
        preferred_auth_method = _normalize_requested_preferred_auth_method(
            meta.get("preferred_auth_method")
        )
        model = _clean_optional_text(meta.get("model"))
        reasoning_effort = (
            _display_reasoning_effort(meta.get("reasoning_effort"))
            if backend_name == "codex"
            else _display_pi_reasoning_effort(meta.get("reasoning_effort"))
        )
        if log_path is not None and log_path.exists():
            log_provider, log_model, log_effort = _read_run_settings_from_log(
                log_path, agent_backend=backend_name
            )
            if log_provider is not None:
                model_provider = log_provider
            if log_model is not None:
                model = log_model
            if log_effort is not None:
                reasoning_effort = log_effort
        return model_provider, preferred_auth_method, model, reasoning_effort

    def _session_transport(
        self, *, meta: dict[str, Any]
    ) -> tuple[str | None, str | None, str | None]:
        transport = _clean_optional_text(meta.get("transport"))
        tmux_session = _clean_optional_text(meta.get("tmux_session"))
        tmux_window = _clean_optional_text(meta.get("tmux_window"))
        if transport is None and (tmux_session is not None or tmux_window is not None):
            transport = "tmux"
        return transport, tmux_session, tmux_window

    def _discover_existing_if_stale(self, *, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            last = float(getattr(self, "_last_discover_ts", 0.0))
        if (not force) and ((now - last) < DISCOVER_MIN_INTERVAL_SECONDS):
            return
        try:
            self._discover_existing(force=force, skip_invalid_sidecars=True)
        except TypeError:
            try:
                self._discover_existing(force=force)
            except TypeError:
                self._discover_existing()

    def _sidecar_quarantine_signature(self, sock: Path) -> tuple[bool, int, int]:
        meta_path = sock.with_suffix(".json")
        try:
            st = meta_path.stat()
        except FileNotFoundError:
            return (False, 0, 0)
        return (True, int(st.st_mtime_ns), int(st.st_size))

    def _sidecar_is_quarantined(self, sock: Path) -> bool:
        bad_sidecars = getattr(self, "_bad_sidecars", None)
        if not isinstance(bad_sidecars, dict):
            self._bad_sidecars = {}
            bad_sidecars = self._bad_sidecars
        key = str(sock)
        prev_sig = bad_sidecars.get(key)
        if prev_sig is None:
            return False
        cur_sig = self._sidecar_quarantine_signature(sock)
        if cur_sig == prev_sig:
            return True
        bad_sidecars.pop(key, None)
        return False

    def _quarantine_sidecar(
        self,
        sock: Path,
        exc: BaseException,
        *,
        reason: str = "invalid sidecar",
        log: bool = True,
    ) -> None:
        bad_sidecars = getattr(self, "_bad_sidecars", None)
        if not isinstance(bad_sidecars, dict):
            self._bad_sidecars = {}
            bad_sidecars = self._bad_sidecars
        bad_sidecars[str(sock)] = self._sidecar_quarantine_signature(sock)
        if log:
            sys.stderr.write(
                f"error: discover: quarantining {reason} for {sock}: {type(exc).__name__}: {exc}\n"
            )
            sys.stderr.flush()

    def _clear_sidecar_quarantine(self, sock: Path) -> None:
        bad_sidecars = getattr(self, "_bad_sidecars", None)
        if isinstance(bad_sidecars, dict):
            bad_sidecars.pop(str(sock), None)

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
                raise ValueError(
                    f"invalid harness config for session {sid!r} (use 'request', not 'text')"
                )
            request = v.get("request")
            if request is None:
                request = ""
            if not isinstance(request, str):
                raise ValueError(f"invalid harness request for session {sid!r}")
            cooldown_minutes = _clean_harness_cooldown_minutes(
                v.get("cooldown_minutes")
            )
            remaining_injections = _clean_harness_remaining_injections(
                v.get("remaining_injections"), allow_zero=True
            )
            cleaned[sid] = {
                "enabled": enabled,
                "request": request,
                "cooldown_minutes": cooldown_minutes,
                "remaining_injections": remaining_injections,
            }
        with self._lock:
            self._harness = cleaned

    def _save_harness(self) -> None:
        with self._lock:
            obj = dict(self._harness)
        os.makedirs(APP_DIR, exist_ok=True)
        tmp = HARNESS_PATH.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, HARNESS_PATH)

    def _load_aliases(self) -> None:
        self._sidebar_state_facade().load_aliases()

    def _save_aliases(self) -> None:
        self._persist_session_ui_state()

    def _load_sidebar_meta(self) -> None:
        self._sidebar_state_facade().load_sidebar_meta()

    def _save_sidebar_meta(self) -> None:
        self._persist_session_ui_state()

    def _load_hidden_sessions(self) -> None:
        self._sidebar_state_facade().load_hidden_sessions()

    def _save_hidden_sessions(self) -> None:
        self._persist_session_ui_state()

    def _hidden_session_keys(
        self,
        session_id: str | None,
        thread_id: str | None,
        resume_session_id: str | None,
        backend: str | None,
    ) -> set[str]:
        return self._sidebar_state_facade().hidden_session_keys(
            session_id, thread_id, resume_session_id, backend
        )

    def _session_is_hidden(
        self,
        session_id: str | None,
        thread_id: str | None,
        resume_session_id: str | None,
        backend: str | None,
    ) -> bool:
        return self._sidebar_state_facade().session_is_hidden(
            session_id, thread_id, resume_session_id, backend
        )

    def _hide_session(self, session_id: str) -> None:
        self._sidebar_state_facade().hide_session(session_id)

    def _hide_session_identity_values(
        self,
        session_id: str | None,
        thread_id: str | None,
        resume_session_id: str | None,
        backend: str | None,
    ) -> None:
        self._sidebar_state_facade().hide_session_identity_values(
            session_id, thread_id, resume_session_id, backend
        )

    def _hide_session_identity(self, s: Session) -> None:
        self._sidebar_state_facade().hide_session_identity(s)

    def _unhide_session(self, session_id: str) -> None:
        self._sidebar_state_facade().unhide_session(session_id)

    def set_created_session_name(
        self,
        *,
        session_id: Any,
        runtime_id: Any = None,
        backend: Any = None,
        name: Any,
    ) -> str:
        return self._sidebar_state_facade().set_created_session_name(
            session_id=session_id,
            runtime_id=runtime_id,
            backend=backend,
            name=name,
        )

    def alias_set(self, session_id: str, name: str) -> str:
        alias = self._sidebar_state_facade().alias_set(session_id, name)
        _publish_sessions_invalidate(reason="alias_changed")
        return alias

    def alias_get(self, session_id: str) -> str:
        return self._sidebar_state_facade().alias_get(session_id)

    def alias_clear(self, session_id: str) -> None:
        self._sidebar_state_facade().alias_clear(session_id)
        _publish_sessions_invalidate(reason="alias_cleared")

    def sidebar_meta_get(self, session_id: str) -> dict[str, Any]:
        return self._sidebar_state_facade().sidebar_meta_get(session_id)

    def sidebar_meta_set(
        self,
        session_id: str,
        *,
        priority_offset: Any,
        snooze_until: Any,
        dependency_session_id: Any,
    ) -> dict[str, Any]:
        payload = self._sidebar_state_facade().sidebar_meta_set(
            session_id,
            priority_offset=priority_offset,
            snooze_until=snooze_until,
            dependency_session_id=dependency_session_id,
        )
        _publish_sessions_invalidate(reason="sidebar_meta_changed")
        return payload

    def focus_set(self, session_id: str, focused: Any) -> bool:
        value = self._sidebar_state_facade().focus_set(session_id, focused)
        _publish_sessions_invalidate(reason="focus_changed")
        return value

    def edit_session(
        self,
        session_id: str,
        *,
        name: str,
        priority_offset: Any,
        snooze_until: Any,
        dependency_session_id: Any,
    ) -> tuple[str, dict[str, Any]]:
        payload = self._sidebar_state_facade().edit_session(
            session_id,
            name=name,
            priority_offset=priority_offset,
            snooze_until=snooze_until,
            dependency_session_id=dependency_session_id,
        )
        _publish_sessions_invalidate(reason="session_edited")
        return payload

    def _clear_deleted_session_state(self, session_id: str) -> None:
        changed_sidebar = False
        changed_harness = False
        changed_files = False
        changed_queues = False
        ref = self._page_state_ref_for_session_id(session_id)
        with self._lock:
            aliases = getattr(self, "_aliases", None)
            if isinstance(aliases, dict):
                aliases.pop(session_id, None)
                if ref is not None:
                    aliases.pop(ref, None)
            meta_map = getattr(self, "_sidebar_meta", None)
            if isinstance(meta_map, dict):
                if session_id in meta_map:
                    meta_map.pop(session_id, None)
                    changed_sidebar = True
                if ref is not None and ref in meta_map:
                    meta_map.pop(ref, None)
                    changed_sidebar = True
            if isinstance(meta_map, dict) and ref is not None:
                for entry in meta_map.values():
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("dependency_session_id") != ref[1]:
                        continue
                    entry.pop("dependency_session_id", None)
                    changed_sidebar = True
            harness = getattr(self, "_harness", None)
            if isinstance(harness, dict) and session_id in harness:
                harness.pop(session_id, None)
                changed_harness = True
            files = getattr(self, "_files", None)
            if isinstance(files, dict):
                for legacy_key in (session_id, f"sid:{session_id}"):
                    if legacy_key in files:
                        files.pop(legacy_key, None)
                        changed_files = True
                if ref is not None and ref in files:
                    files.pop(ref, None)
                    changed_files = True
            queues = getattr(self, "_queues", None)
            if isinstance(queues, dict):
                if session_id in queues:
                    queues.pop(session_id, None)
                    changed_queues = True
                if ref is not None and ref in queues:
                    queues.pop(ref, None)
                    changed_queues = True
            command_cache = getattr(self, "_pi_commands_cache", None)
            if isinstance(command_cache, dict):
                command_cache.pop(session_id, None)
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
        db = getattr(self, "_page_state_db", None)
        if db is None:
            try:
                raw = FILE_HISTORY_PATH.read_text(encoding="utf-8")
            except FileNotFoundError:
                return
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("invalid session_files.json (expected object)")
            cleaned: dict[SessionStateKey, list[str]] = {}
            for sid, arr in obj.items():
                if not isinstance(sid, str) or not sid:
                    continue
                if sid.startswith("cwd:"):
                    continue
                key = sid if sid.startswith("sid:") else f"sid:{sid}"
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
            return
        with self._lock:
            self._files = db.load_files()

    def _save_files(self) -> None:
        db = getattr(self, "_page_state_db", None)
        if db is None:
            with self._lock:
                obj = dict(self._files)
            os.makedirs(APP_DIR, exist_ok=True)
            tmp = FILE_HISTORY_PATH.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, FILE_HISTORY_PATH)
            return
        self._persist_files()

    def _load_queues(self) -> None:
        db = getattr(self, "_page_state_db", None)
        if db is None:
            try:
                raw = QUEUE_PATH.read_text(encoding="utf-8")
            except FileNotFoundError:
                return
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("invalid session_queues.json (expected object)")
            cleaned: dict[SessionStateKey, list[str]] = {}
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
            return
        with self._lock:
            self._queues = db.load_queues()

    def _save_queues(self) -> None:
        db = getattr(self, "_page_state_db", None)
        if db is None:
            with self._lock:
                obj = dict(self._queues)
            os.makedirs(APP_DIR, exist_ok=True)
            tmp = QUEUE_PATH.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, QUEUE_PATH)
            return
        self._persist_queues()

    def _load_recent_cwds(self) -> None:
        db = getattr(self, "_page_state_db", None)
        if db is None:
            try:
                raw = RECENT_CWD_PATH.read_text(encoding="utf-8")
            except FileNotFoundError:
                return
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("invalid recent_cwds.json (expected object)")
            source_items = obj.items()
        else:
            source_items = db.load_recent_cwds().items()
        cleaned: dict[str, float] = {}
        for raw_cwd, raw_ts in source_items:
            cwd = _clean_recent_cwd(raw_cwd)
            if cwd is None or isinstance(raw_ts, bool):
                continue
            try:
                ts = float(raw_ts)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(ts) or ts <= 0:
                continue
            prev = cleaned.get(cwd)
            if prev is None or ts > prev:
                cleaned[cwd] = ts
        top = sorted(cleaned.items(), key=lambda item: (-item[1], item[0]))[:RECENT_CWD_MAX]
        with self._lock:
            self._recent_cwds = dict(top)

    def _save_recent_cwds(self) -> None:
        db = getattr(self, "_page_state_db", None)
        with self._lock:
            items = sorted(
                getattr(self, "_recent_cwds", {}).items(),
                key=lambda item: (-float(item[1]), item[0]),
            )[:RECENT_CWD_MAX]
            self._recent_cwds = dict(items)
        if db is not None:
            self._persist_recent_cwds()
            return
        obj = {cwd: ts for cwd, ts in items}
        os.makedirs(APP_DIR, exist_ok=True)
        tmp = RECENT_CWD_PATH.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, RECENT_CWD_PATH)

    def _load_cwd_groups(self) -> None:
        db = getattr(self, "_page_state_db", None)
        source_items: Any
        if db is None:
            try:
                raw = CWD_GROUPS_PATH.read_text(encoding="utf-8")
            except FileNotFoundError:
                with self._lock:
                    self._cwd_groups = {}
                return
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    raise ValueError("invalid cwd_groups.json (expected object)")
                source_items = obj.items()
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                LOG.warning("recovering malformed cwd_groups.json as empty state: %s", e)
                source_items = []
        else:
            source_items = db.load_cwd_groups().items()
        cleaned: dict[str, dict[str, Any]] = {}
        for cwd, v in source_items:
            try:
                normalized_cwd = _normalize_cwd_group_key(cwd)
            except ValueError:
                continue
            if not isinstance(v, dict):
                continue
            label = _clean_alias(v.get("label", ""))
            persisted_collapsed = v.get("collapsed", False)
            collapsed = (
                persisted_collapsed if isinstance(persisted_collapsed, bool) else False
            )
            if label or collapsed:
                cleaned[normalized_cwd] = {"label": label, "collapsed": collapsed}
        with self._lock:
            self._cwd_groups = cleaned

    def _save_cwd_groups(self) -> None:
        db = getattr(self, "_page_state_db", None)
        if db is not None:
            self._persist_cwd_groups()
            return
        with self._lock:
            obj = dict(self._cwd_groups)
        os.makedirs(APP_DIR, exist_ok=True)
        tmp = CWD_GROUPS_PATH.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, CWD_GROUPS_PATH)

    def cwd_groups_get(self) -> dict[str, dict[str, Any]]:
        self._prune_stale_workspace_dirs()
        with self._lock:
            return copy.deepcopy(self._cwd_groups)

    def _prune_stale_workspace_dirs(self) -> None:
        if not bool(getattr(self, "_prune_missing_workspace_dirs", False)):
            return
        active_cwds: set[str] = set()
        with self._lock:
            sessions = list(getattr(self, "_sessions", {}).values())
            recent_items = list(getattr(self, "_recent_cwds", {}).keys())
            grouped_items = list(getattr(self, "_cwd_groups", {}).keys())
        for session in sessions:
            try:
                active_cwds.add(_normalize_cwd_group_key(getattr(session, "cwd", None)))
            except ValueError:
                continue
        stale_recent = {
            cwd
            for cwd in recent_items
            if cwd not in active_cwds and _existing_workspace_dir(cwd) is None
        }
        stale_groups = {
            cwd
            for cwd in grouped_items
            if cwd not in active_cwds and _existing_workspace_dir(cwd) is None
        }
        save_recent = False
        save_groups = False
        if stale_recent or stale_groups:
            with self._lock:
                recent_map = getattr(self, "_recent_cwds", None)
                if isinstance(recent_map, dict):
                    for cwd in stale_recent:
                        if recent_map.pop(cwd, None) is not None:
                            save_recent = True
                group_map = getattr(self, "_cwd_groups", None)
                if isinstance(group_map, dict):
                    for cwd in stale_groups:
                        if group_map.pop(cwd, None) is not None:
                            save_groups = True
        if save_recent:
            self._save_recent_cwds()
        if save_groups:
            self._save_cwd_groups()

    def _known_cwd_group_keys(self) -> set[str]:
        self._prune_stale_workspace_dirs()
        known: set[str] = set()
        with self._lock:
            sessions = list(getattr(self, "_sessions", {}).values())
            recent_items = list(getattr(self, "_recent_cwds", {}).keys())
            grouped_items = list(getattr(self, "_cwd_groups", {}).keys())
        for session in sessions:
            try:
                normalized = _normalize_cwd_group_key(getattr(session, "cwd", None))
            except ValueError:
                continue
            known.add(normalized)
        for cwd in recent_items:
            try:
                normalized = _normalize_cwd_group_key(cwd)
            except ValueError:
                continue
            known.add(normalized)
        for cwd in grouped_items:
            try:
                normalized = _normalize_cwd_group_key(cwd)
            except ValueError:
                continue
            known.add(normalized)
        return known

    def cwd_group_set(
        self, cwd: str, label: str | None = None, collapsed: bool | None = None
    ) -> tuple[str, dict[str, Any]]:
        normalized_cwd = _normalize_cwd_group_key(cwd)
        if label is not None and not isinstance(label, str):
            raise ValueError("label must be a string")

        requested_label = _clean_alias(label) if label is not None else None
        requested_collapsed = collapsed
        if requested_collapsed is not None and not isinstance(
            requested_collapsed, bool
        ):
            raise ValueError("collapsed must be a boolean")

        self._prune_stale_workspace_dirs()
        with self._lock:
            existing = self._cwd_groups.get(
                normalized_cwd, {"label": "", "collapsed": False}
            )
        known_cwds = self._known_cwd_group_keys()
        if normalized_cwd not in known_cwds:
            effective_label = (
                requested_label if requested_label is not None else existing["label"]
            )
            effective_collapsed = (
                requested_collapsed
                if requested_collapsed is not None
                else existing["collapsed"]
            )
            if not effective_label and not effective_collapsed:
                return normalized_cwd, {"label": "", "collapsed": False}
            raise ValueError("cwd is not a known session working directory")

        with self._lock:
            existing = self._cwd_groups.get(normalized_cwd, existing)
            new_label = (
                requested_label if requested_label is not None else existing["label"]
            )
            new_collapsed = (
                requested_collapsed
                if requested_collapsed is not None
                else existing["collapsed"]
            )

            entry = {"label": new_label, "collapsed": new_collapsed}

            if not new_label and not new_collapsed:
                self._cwd_groups.pop(normalized_cwd, None)
            else:
                self._cwd_groups[normalized_cwd] = entry

        self._save_cwd_groups()
        return normalized_cwd, dict(entry)

    def _remember_recent_cwd(self, cwd: Any, *, ts: Any = None) -> bool:
        cleaned = _clean_recent_cwd(cwd)
        if cleaned is None:
            return False
        if isinstance(ts, bool):
            ts_value = time.time()
        else:
            try:
                ts_value = float(ts) if ts is not None else time.time()
            except (TypeError, ValueError):
                ts_value = time.time()
        if not math.isfinite(ts_value) or ts_value <= 0:
            ts_value = time.time()
        with self._lock:
            recent = getattr(self, "_recent_cwds", None)
            if not isinstance(recent, dict):
                self._recent_cwds = {}
                recent = self._recent_cwds
            prev = recent.get(cleaned)
            if prev is not None and prev >= ts_value:
                return False
            recent[cleaned] = ts_value
            if len(recent) > RECENT_CWD_MAX * 2:
                keep = dict(
                    sorted(recent.items(), key=lambda item: (-float(item[1]), item[0]))[
                        :RECENT_CWD_MAX
                    ]
                )
                recent.clear()
                recent.update(keep)
        return True

    def _backfill_recent_cwds_from_logs(self) -> None:
        changed = False
        seen: set[str] = set()
        for log_path in _iter_session_logs():
            try:
                row = _resume_candidate_from_log(log_path)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            cwd = row.get("cwd")
            if not isinstance(cwd, str) or not cwd or cwd in seen:
                continue
            seen.add(cwd)
            if self._remember_recent_cwd(cwd, ts=row.get("updated_ts")):
                changed = True
            if len(seen) >= RECENT_CWD_MAX:
                break
        if changed:
            self._save_recent_cwds()

    def recent_cwds(self, *, limit: int = RECENT_CWD_MAX) -> list[str]:
        self._prune_stale_workspace_dirs()
        with self._lock:
            items = sorted(
                getattr(self, "_recent_cwds", {}).items(),
                key=lambda item: (-float(item[1]), item[0]),
            )
        return [cwd for cwd, _ts in items[: max(0, int(limit))]]

    def _queue_len(self, session_id: str) -> int:
        ref = self._page_state_ref_for_session_id(session_id)
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if ref is None:
            return 0
        with self._lock:
            qmap = getattr(self, "_queues", None)
            if not isinstance(qmap, dict):
                return 0
            q = qmap.get(ref)
            if not isinstance(q, list) and runtime_id is not None:
                q = qmap.get(runtime_id)
            if not isinstance(q, list):
                q = qmap.get(session_id)
            return int(len(q)) if isinstance(q, list) else 0

    def _queue_list_local(self, session_id: str) -> list[str]:
        ref = self._page_state_ref_for_session_id(session_id)
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if ref is None:
            return []
        with self._lock:
            qmap = getattr(self, "_queues", None)
            if not isinstance(qmap, dict):
                return []
            q = qmap.get(ref)
            if not isinstance(q, list) and runtime_id is not None:
                q = qmap.get(runtime_id)
            if not isinstance(q, list):
                q = qmap.get(session_id)
            if not isinstance(q, list) or not q:
                return []
            return list(q)

    def _queue_enqueue_local(self, session_id: str, text: str) -> dict[str, Any]:
        t = str(text)
        if not t.strip():
            raise ValueError("text required")
        touched_ts: float | None = None
        ref = self._page_state_ref_for_session_id(session_id)
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        durable_session_id = _clean_optional_text(session_id)
        with self._lock:
            if runtime_id is not None:
                s0 = self._sessions.get(runtime_id)
                if s0 is not None:
                    durable_session_id = self._durable_session_id_for_session(s0)
        if ref is None:
            raise KeyError("unknown session")
        with self._lock:
            s = self._sessions.get(runtime_id) if runtime_id is not None else None
            q = self._queues.get(runtime_id) if runtime_id is not None else None
            if not isinstance(q, list):
                q = self._queues.get(ref)
            if not isinstance(q, list):
                q = []
                if runtime_id is not None:
                    self._queues[runtime_id] = q
                else:
                    self._queues[ref] = q
            q.append(t)
            ql = len(q)
            if s is not None and s.backend == "pi":
                touched_ts = _touch_session_file(s.session_path)
                s.pi_idle_activity_ts = None
                s.pi_busy_activity_floor = touched_ts
        self._save_queues()
        if durable_session_id is not None:
            _publish_session_workspace_invalidate(
                durable_session_id,
                runtime_id=runtime_id,
                reason="queue_changed",
            )
            _publish_session_live_invalidate(
                durable_session_id,
                runtime_id=runtime_id,
                reason="queue_changed",
            )
        return {"queued": True, "queue_len": int(ql)}

    def _queue_delete_local(self, session_id: str, index: int) -> dict[str, Any]:
        ref = self._page_state_ref_for_session_id(session_id)
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if ref is None or runtime_id is None:
            raise KeyError("unknown session")
        with self._lock:
            session = self._sessions.get(runtime_id)
            if session is None:
                raise KeyError("unknown session")
            durable_session_id = self._durable_session_id_for_session(session)
            q = self._queues.get(runtime_id)
            if not isinstance(q, list):
                q = []
                self._queues[runtime_id] = q
            if index < 0 or index >= len(q):
                raise ValueError("index out of range")
            q.pop(int(index))
            ql = len(q)
            if not q:
                self._queues.pop(runtime_id, None)
                self._queues.pop(ref, None)
        self._save_queues()
        _publish_session_workspace_invalidate(
            durable_session_id,
            runtime_id=runtime_id,
            reason="queue_changed",
        )
        _publish_session_live_invalidate(
            durable_session_id,
            runtime_id=runtime_id,
            reason="queue_changed",
        )
        return {"ok": True, "queue_len": int(ql)}

    def _queue_update_local(
        self, session_id: str, index: int, text: str
    ) -> dict[str, Any]:
        t = str(text)
        if not t.strip():
            raise ValueError("text required")
        ref = self._page_state_ref_for_session_id(session_id)
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if ref is None or runtime_id is None:
            raise KeyError("unknown session")
        with self._lock:
            session = self._sessions.get(runtime_id)
            if session is None:
                raise KeyError("unknown session")
            durable_session_id = self._durable_session_id_for_session(session)
            q = self._queues.get(runtime_id)
            if not isinstance(q, list):
                q = []
                self._queues[runtime_id] = q
            if index < 0 or index >= len(q):
                raise ValueError("index out of range")
            q[int(index)] = t
            ql = len(q)
        self._save_queues()
        _publish_session_workspace_invalidate(
            durable_session_id,
            runtime_id=runtime_id,
            reason="queue_changed",
        )
        return {"ok": True, "queue_len": int(ql)}

    def _files_key_for_session(self, session_id: str) -> tuple[str, SessionRef, "Session"]:
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is None:
            raise KeyError("unknown session")
        s = self._sessions.get(runtime_id)
        if not s:
            raise KeyError("unknown session")
        ref = self._page_state_ref_for_session(s)
        if ref is None:
            raise KeyError("unknown session")
        return runtime_id, ref, s

    def files_get(self, session_id: str) -> list[str]:
        runtime_id, key, _s = self._files_key_for_session(session_id)
        with self._lock:
            arr = self._files.get(runtime_id)
            if not isinstance(arr, list):
                arr = self._files.get(key)
            if not isinstance(arr, list):
                arr = self._files.get(session_id)
            return list(arr) if isinstance(arr, list) else []

    def files_add(self, session_id: str, path: str) -> list[str]:
        p = str(path).strip()
        if not p:
            return self.files_get(session_id)
        runtime_id, key, _s = self._files_key_for_session(session_id)
        with self._lock:
            cur = list(self._files.get(runtime_id, self._files.get(key, self._files.get(session_id, []))))
            cur = [x for x in cur if x != p]
            cur.insert(0, p)
            if len(cur) > FILE_HISTORY_MAX:
                cur = cur[:FILE_HISTORY_MAX]
            self._files[runtime_id] = cur
        self._save_files()
        return list(cur)

    def files_clear(self, session_id: str) -> None:
        dirty = False
        runtime_id, key, _s = self._files_key_for_session(session_id)
        with self._lock:
            if runtime_id in self._files:
                self._files.pop(runtime_id, None)
                dirty = True
            if session_id in self._files:
                self._files.pop(session_id, None)
                dirty = True
            if key in self._files:
                self._files.pop(key, None)
                dirty = True
        if dirty:
            self._save_files()

    def harness_get(self, session_id: str) -> dict[str, Any]:
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is None:
            raise KeyError("unknown session")
        with self._lock:
            s = self._sessions.get(runtime_id)
            if not s:
                raise KeyError("unknown session")
            cfg0 = self._harness.get(runtime_id)
            cfg = dict(cfg0) if isinstance(cfg0, dict) else {}
        enabled = bool(cfg.get("enabled"))
        request = cfg.get("request")
        if not isinstance(request, str):
            request = ""
        cooldown_minutes = _clean_harness_cooldown_minutes(cfg.get("cooldown_minutes"))
        remaining_injections = _clean_harness_remaining_injections(
            cfg.get("remaining_injections"), allow_zero=True
        )
        return {
            "enabled": enabled,
            "request": request,
            "cooldown_minutes": cooldown_minutes,
            "remaining_injections": remaining_injections,
        }

    def harness_set(
        self,
        session_id: str,
        *,
        enabled: bool | None = None,
        request: str | None = None,
        cooldown_minutes: int | None = None,
        remaining_injections: int | None = None,
    ) -> dict[str, Any]:
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is None:
            raise KeyError("unknown session")
        with self._lock:
            s = self._sessions.get(runtime_id)
            if not s:
                raise KeyError("unknown session")
            cur0 = self._harness.get(runtime_id)
            cur = dict(cur0) if isinstance(cur0, dict) else {}
            if enabled is not None:
                cur["enabled"] = bool(enabled)
            if request is not None:
                cur["request"] = str(request)
            if cooldown_minutes is not None:
                cur["cooldown_minutes"] = _clean_harness_cooldown_minutes(
                    cooldown_minutes
                )
            if remaining_injections is not None:
                cur["remaining_injections"] = _clean_harness_remaining_injections(
                    remaining_injections, allow_zero=True
                )
            cur["cooldown_minutes"] = _clean_harness_cooldown_minutes(
                cur.get("cooldown_minutes")
            )
            cur["remaining_injections"] = _clean_harness_remaining_injections(
                cur.get("remaining_injections"), allow_zero=True
            )
            self._harness[runtime_id] = cur
            if enabled is not None and bool(enabled) is False:
                self._harness_last_injected.pop(runtime_id, None)
        self._save_harness()
        return self.harness_get(session_id)

    def _session_display_name(self, session_id: str) -> str:
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is None:
            return "Session"
        with self._lock:
            s = self._sessions.get(runtime_id)
            if not s:
                return "Session"
            ref = self._page_state_ref_for_session(s)
            alias = self._aliases.get(ref) if ref is not None else None
            return _session_row_display_name(
                {
                    "session_id": self._durable_session_id_for_session(s),
                    "alias": alias,
                    "title": s.title or "",
                    "first_user_message": s.first_user_message or "",
                }
            )

    def _observe_rollout_delta(
        self, session_id: str, *, objs: list[dict[str, Any]], new_off: int
    ) -> None:
        voice_push = getattr(self, "_voice_push", None)
        if voice_push is None:
            with self._lock:
                s = self._sessions.get(session_id)
                if s is not None:
                    s.delivery_log_off = max(int(s.delivery_log_off), int(new_off))
            return
        with self._lock:
            s0 = self._sessions.get(session_id)
            resume_muted = bool(s0 and s0.resume_session_id)
        messages = _extract_delivery_messages(objs)
        if (not messages) or resume_muted:
            with self._lock:
                s = self._sessions.get(session_id)
                if s is not None:
                    s.delivery_log_off = max(int(s.delivery_log_off), int(new_off))
            return
        session_name = self._session_display_name(session_id)
        voice_push.observe_messages(
            session_id=session_id, session_display_name=session_name, messages=messages
        )
        with self._lock:
            s = self._sessions.get(session_id)
            if s is not None:
                s.delivery_log_off = max(int(s.delivery_log_off), int(new_off))

    def _voice_push_scan_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._voice_push_scan_sweep()
            except Exception as e:
                sys.stderr.write(
                    f"error: voice-push scan failed: {type(e).__name__}: {e}\n"
                )
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
            self._stop.wait(VOICE_PUSH_SWEEP_SECONDS)

    def _voice_push_scan_sweep(self) -> None:
        self._discover_existing_if_stale()
        self._prune_dead_sessions()
        with self._lock:
            session_ids = list(self._sessions.keys())
        for sid in session_ids:
            try:
                self.refresh_session_meta(sid)
            except Exception:
                continue
            with self._lock:
                s = self._sessions.get(sid)
                if s is None:
                    continue
                log_path = s.log_path
                delivery_off = int(s.delivery_log_off)
            if log_path is None or (not log_path.exists()):
                continue
            try:
                size = int(log_path.stat().st_size)
            except FileNotFoundError:
                continue
            off = 0 if size < delivery_off else int(delivery_off)
            loops = 0
            while off < size and loops < 16:
                objs, new_off = _read_jsonl_from_offset(
                    log_path, off, max_bytes=256 * 1024
                )
                if new_off <= off:
                    break
                self._observe_rollout_delta(sid, objs=objs, new_off=new_off)
                off = new_off
                loops += 1

    def _harness_loop(self) -> None:
        # Persist across browser disconnects: server is the scheduler.
        while not self._stop.is_set():
            try:
                self._harness_sweep()
            except Exception as e:
                sys.stderr.write(
                    f"error: harness sweep failed: {type(e).__name__}: {e}\n"
                )
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
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
            try:
                cooldown_minutes = _clean_harness_cooldown_minutes(
                    cfg.get("cooldown_minutes")
                )
                cooldown_seconds = float(cooldown_minutes * 60)
                remaining_injections = _clean_harness_remaining_injections(
                    cfg.get("remaining_injections"), allow_zero=True
                )
                if remaining_injections <= 0:
                    with self._lock:
                        cur0 = self._harness.get(sid)
                        cur = dict(cur0) if isinstance(cur0, dict) else {}
                        cur["enabled"] = False
                        cur["remaining_injections"] = 0
                        self._harness[sid] = cur
                        self._harness_last_injected.pop(sid, None)
                    self._save_harness()
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
                    scope_last = float(
                        self._harness_last_injected_scope.get(scope_key, 0.0)
                    )
                if (last_inj and (now - last_inj) < cooldown_seconds) or (
                    scope_last and (now - scope_last) < cooldown_seconds
                ):
                    continue
                st = self.get_state(sid)
                if not isinstance(st, dict):
                    raise ValueError("invalid broker state response")
                if "busy" not in st or "queue_len" not in st:
                    raise ValueError("invalid broker state response")
                busy = _state_busy_value(st)
                ql = _state_queue_len_value(st)
                if busy or ql > 0 or self._queue_len(sid) > 0:
                    continue
                last = _last_chat_role_ts_from_tail(
                    lp, max_scan_bytes=HARNESS_MAX_SCAN_BYTES
                )
                if not last:
                    continue
                role, ts = last
                if role != "assistant":
                    continue
                if (now - float(ts)) < cooldown_seconds:
                    continue
                with self._lock:
                    scope_last = float(
                        self._harness_last_injected_scope.get(scope_key, 0.0)
                    )
                if scope_last and (now - scope_last) < cooldown_seconds:
                    continue
                self.send(sid, prompt)
                with self._lock:
                    self._harness_last_injected[sid] = now
                    self._harness_last_injected_scope[scope_key] = now
                    cur0 = self._harness.get(sid)
                    cur = dict(cur0) if isinstance(cur0, dict) else {}
                    next_remaining = max(0, remaining_injections - 1)
                    cur["remaining_injections"] = next_remaining
                    if next_remaining <= 0:
                        cur["enabled"] = False
                        self._harness_last_injected.pop(sid, None)
                    self._harness[sid] = cur
                self._save_harness()
            except Exception as e:
                sys.stderr.write(
                    f"error: harness session {sid} skipped: {type(e).__name__}: {e}\n"
                )
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()

    def _queue_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._queue_sweep()
            except Exception:
                sys.stderr.write("error: queue sweep crashed; continuing\n")
                sys.stderr.flush()
            queue_wakeup = getattr(self, "_queue_wakeup", None)
            if isinstance(queue_wakeup, threading.Event):
                queue_wakeup.wait(QUEUE_SWEEP_SECONDS)
                queue_wakeup.clear()
            else:
                self._stop.wait(QUEUE_SWEEP_SECONDS)

    def _maybe_drain_session_queue(
        self, session_id: str, *, now_ts: float | None = None
    ) -> bool:
        if now_ts is None:
            now_ts = time.time()
        with self._lock:
            s0 = self._sessions.get(session_id)
            if not s0:
                return False
            q = self._queues.get(session_id)
            if not isinstance(q, list) or not q:
                s0.queue_idle_since = None
                return False
            text = q[0]
            log_path = s0.log_path
        try:
            st = self.get_state(session_id)
        except Exception:
            with self._lock:
                s0 = self._sessions.get(session_id)
                if s0:
                    s0.queue_idle_since = None
            return False
        if not isinstance(st, dict) or "busy" not in st or "queue_len" not in st:
            with self._lock:
                s0 = self._sessions.get(session_id)
                if s0:
                    s0.queue_idle_since = None
            return False
        queue_len_raw = st.get("queue_len")
        if not isinstance(queue_len_raw, int):
            with self._lock:
                s0 = self._sessions.get(session_id)
                if s0:
                    s0.queue_idle_since = None
            return False
        if _state_busy_value(st) or int(queue_len_raw) > 0:
            with self._lock:
                s0 = self._sessions.get(session_id)
                if s0:
                    s0.queue_idle_since = None
            return False
        try:
            if (
                isinstance(log_path, Path)
                and log_path.exists()
                and (not self.idle_from_log(session_id))
            ):
                with self._lock:
                    s0 = self._sessions.get(session_id)
                    if s0:
                        s0.queue_idle_since = None
                return False
        except Exception:
            with self._lock:
                s0 = self._sessions.get(session_id)
                if s0:
                    s0.queue_idle_since = None
            return False
        with self._lock:
            s0 = self._sessions.get(session_id)
            if not s0:
                return False
            idle_since = s0.queue_idle_since
            if idle_since is None:
                s0.queue_idle_since = float(now_ts)
                return False
            if (float(now_ts) - idle_since) < QUEUE_IDLE_GRACE_SECONDS:
                return False
        try:
            self.send(session_id, text)
        except Exception:
            with self._lock:
                s0 = self._sessions.get(session_id)
                if s0:
                    s0.queue_idle_since = None
            return False
        ref = self._page_state_ref_for_session_id(session_id)
        with self._lock:
            q = self._queues.get(session_id)
            if not isinstance(q, list) and ref is not None:
                q = self._queues.get(ref)
            s0 = self._sessions.get(session_id)
            if s0:
                s0.queue_idle_since = None
            if isinstance(q, list) and q and q[0] == text:
                q.pop(0)
                if not q:
                    self._queues.pop(session_id, None)
                    if ref is not None:
                        self._queues.pop(ref, None)
        self._save_queues()
        return True

    def _queue_sweep(self) -> None:
        self._discover_existing_if_stale()
        self._prune_dead_sessions()
        with self._lock:
            active_runtime_ids: list[str] = []
            active_outbound_runtime_ids: list[str] = []
            outbound_requests = getattr(self, "_outbound_requests", None)
            for runtime_id, session in self._sessions.items():
                ref = self._page_state_ref_for_session(session)
                q = self._queues.get(runtime_id)
                if (not isinstance(q, list)) and ref is not None:
                    q = self._queues.get(ref)
                if isinstance(q, list) and q:
                    active_runtime_ids.append(runtime_id)
                outbound = outbound_requests.get(runtime_id) if isinstance(outbound_requests, dict) else None
                if isinstance(outbound, list) and outbound:
                    active_outbound_runtime_ids.append(runtime_id)
        for sid in active_outbound_runtime_ids:
            try:
                if self._maybe_drain_outbound_request(sid):
                    break
            except Exception:
                sys.stderr.write(
                    f"error: outbound sweep failed for session {sid}; skipping\n"
                )
                sys.stderr.flush()
        for sid in active_runtime_ids:
            try:
                if self._maybe_drain_session_queue(sid):
                    break
            except Exception:
                sys.stderr.write(
                    f"error: queue sweep failed for session {sid}; skipping\n"
                )
                sys.stderr.flush()

    def _discover_existing(
        self, *, force: bool = False, skip_invalid_sidecars: bool = False
    ) -> None:
        if not force:
            now = time.time()
            with self._lock:
                last = float(self._last_discover_ts)
            if (now - last) < DISCOVER_MIN_INTERVAL_SECONDS:
                return
        SOCK_DIR.mkdir(parents=True, exist_ok=True)
        for sock in sorted(SOCK_DIR.glob("*.sock")):
            if skip_invalid_sidecars and self._sidecar_is_quarantined(sock):
                continue
            session_id = sock.stem
            try:
                # Prefer metadata file written by sessiond.
                meta_path = sock.with_suffix(".json")
                if not meta_path.exists():
                    # The broker creates the Unix socket before atomically
                    # writing the JSON sidecar, so this can be a transient
                    # startup race rather than a broken session.
                    continue
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if not isinstance(meta, dict):
                    raise ValueError(f"invalid metadata json for socket {sock}")

                thread_id = _clean_optional_text(meta.get("session_id")) or session_id
                backend = normalize_agent_backend(
                    meta.get("backend"),
                    default=normalize_agent_backend(
                        meta.get("agent_backend"), default="codex"
                    ),
                )
                agent_backend = normalize_agent_backend(
                    meta.get("agent_backend"), default=backend
                )
                codex_pid_raw = meta.get("codex_pid")
                broker_pid_raw = meta.get("broker_pid")
                if not isinstance(codex_pid_raw, int):
                    raise ValueError(f"invalid codex_pid in metadata for socket {sock}")
                if not isinstance(broker_pid_raw, int):
                    raise ValueError(
                        f"invalid broker_pid in metadata for socket {sock}"
                    )
                codex_pid = int(codex_pid_raw)
                broker_pid = int(broker_pid_raw)
                owned = (
                    (meta.get("owner") == "web")
                    if isinstance(meta.get("owner"), str)
                    else False
                )
                transport, tmux_session, tmux_window = self._session_transport(
                    meta=meta
                )
                supports_live_ui = (
                    meta.get("supports_live_ui")
                    if isinstance(meta.get("supports_live_ui"), bool)
                    else None
                )
                ui_protocol_version_raw = meta.get("ui_protocol_version")
                ui_protocol_version = (
                    ui_protocol_version_raw
                    if type(ui_protocol_version_raw) is int
                    else None
                )
                # Older Pi sidecars predate explicit live-ui metadata. Infer the
                # RPC capability only when the sidecar already claims web control.
                if backend == "pi" and transport is None and (owned or _supports_web_control(meta)):
                    transport = "pi-rpc"
                if backend == "pi" and transport == "pi-rpc" and supports_live_ui is None:
                    supports_live_ui = True
                if (
                    backend == "pi"
                    and transport == "pi-rpc"
                    and supports_live_ui is True
                    and ui_protocol_version is None
                ):
                    ui_protocol_version = 1

                cwd_raw = meta.get("cwd")
                if not isinstance(cwd_raw, str) or (not cwd_raw.strip()):
                    raise ValueError(f"invalid cwd in metadata for socket {sock}")
                cwd = cwd_raw

                start_ts_raw = meta.get("start_ts")
                if not isinstance(start_ts_raw, (int, float)):
                    raise ValueError(f"invalid start_ts in metadata for socket {sock}")
                start_ts = float(start_ts_raw)

                session_path_discovered = False
                inferred_pi_session_path: Path | None = None
                if backend == "codex" and agent_backend == "codex":
                    for key in ("session_path", "log_path"):
                        raw_path = meta.get(key)
                        if not isinstance(raw_path, str) or not raw_path.strip():
                            continue
                        candidate = Path(raw_path)
                        if infer_agent_backend_from_log_path(candidate) != "pi":
                            continue
                        inferred_pi_session_path = candidate
                        break
                    if inferred_pi_session_path is None and _pid_alive(codex_pid):
                        ignored_paths = self._claimed_pi_session_paths(exclude_sid=session_id)
                        inferred_pi_session_path = _proc_find_open_rollout_log(
                            proc_root=PROC_ROOT,
                            root_pid=codex_pid,
                            agent_backend="pi",
                            cwd=cwd,
                            ignored_paths=ignored_paths,
                        )
                if inferred_pi_session_path is not None:
                    backend = "pi"
                    agent_backend = "pi"
                    session_path_discovered = True
                    if transport is None and (owned or _supports_web_control(meta)):
                        transport = "pi-rpc"
                    if supports_live_ui is None and transport == "pi-rpc":
                        supports_live_ui = True
                    if ui_protocol_version is None and supports_live_ui is True:
                        ui_protocol_version = 1
                    _patch_metadata_pi_binding(sock, inferred_pi_session_path)

                if backend == "pi":
                    # Only Pi RPC brokers support the live web control path.
                    if transport != "pi-rpc":
                        continue
                    if supports_live_ui is not True:
                        continue
                    if not isinstance(ui_protocol_version, int) or ui_protocol_version < 1:
                        continue
                    if (not owned) and (not _supports_web_control(meta)):
                        continue

                log_path = _metadata_log_path(meta=meta, backend=backend, sock=sock)
                session_path: Path | None
                if inferred_pi_session_path is not None:
                    session_path = inferred_pi_session_path
                else:
                    preferred_session_path: Path | None = None
                    if backend == "pi":
                        try:
                            preferred_session_path = _metadata_session_path(
                                meta=meta, backend=backend, sock=sock
                            )
                        except ValueError as exc:
                            if "missing session_path" not in str(exc):
                                raise
                        claimed: set[Path] | None = (
                            self._claimed_pi_session_paths(exclude_sid=session_id)
                            if preferred_session_path is None
                            else None
                        )
                        session_path, session_path_source = _resolve_pi_session_path(
                            thread_id=thread_id,
                            cwd=cwd,
                            start_ts=start_ts,
                            preferred=preferred_session_path,
                            exclude=claimed,
                        )
                        if session_path is not None and session_path_source in {
                            "exact",
                            "discovered",
                        }:
                            session_path_discovered = True
                            _patch_metadata_session_path(
                                sock,
                                session_path,
                                force=(
                                    preferred_session_path is not None
                                    and preferred_session_path != session_path
                                ),
                            )
                    else:
                        session_path = _metadata_session_path(
                            meta=meta, backend=backend, sock=sock
                        )
                if log_path is not None and log_path.exists():
                    thread_id, log_path = _coerce_main_thread_log(
                        thread_id=thread_id, log_path=log_path
                    )
                else:
                    log_path = None
            except Exception as exc:
                if skip_invalid_sidecars:
                    self._quarantine_sidecar(sock, exc)
                    continue
                raise
            self._clear_sidecar_quarantine(sock)

            if (
                (log_path is None)
                and (not _pid_alive(codex_pid))
                and (not _pid_alive(broker_pid))
            ):
                self._unhide_session(session_id)
                _unlink_quiet(sock)
                _unlink_quiet(meta_path)
                continue
            resume_session_id = _clean_optional_text(meta.get("resume_session_id"))
            if self._session_is_hidden(
                session_id,
                thread_id,
                resume_session_id,
                agent_backend,
            ):
                if (not _pid_alive(codex_pid)) and (not _pid_alive(broker_pid)):
                    self._unhide_session(session_id)
                    _unlink_quiet(sock)
                    _unlink_quiet(meta_path)
                continue

            try:
                model_provider, preferred_auth_method, model, reasoning_effort = (
                    self._session_run_settings(
                        backend=backend, meta=meta, log_path=log_path
                    )
                )
                service_tier = _normalize_requested_service_tier(
                    meta.get("service_tier")
                )
            except Exception as exc:
                if skip_invalid_sidecars:
                    self._quarantine_sidecar(sock, exc)
                    continue
                raise
            # Validate socket is responsive.
            # If the broker is still bootstrapping, keep the session visible with
            # a degraded state so UI registration does not depend on a ready RPC socket.
            try:
                resp = self._sock_call(sock, {"cmd": "state"}, timeout_s=0.5)
            except Exception as e:
                if _probe_failure_safe_to_prune(
                    broker_pid=broker_pid, codex_pid=codex_pid
                ):
                    _unlink_quiet(sock)
                    _unlink_quiet(meta_path)
                    continue
                if (not _sock_error_definitely_stale(e)) and (
                    not skip_invalid_sidecars
                ):
                    sys.stderr.write(
                        f"error: discover: sock state call failed for {sock}: {type(e).__name__}: {e}\n"
                    )
                    sys.stderr.flush()
                resp = {"busy": False, "queue_len": 0, "token": None}
            queue_len_raw = resp.get("queue_len") if isinstance(resp, dict) else None
            if (
                not isinstance(resp, dict)
                or not isinstance(resp.get("busy"), bool)
                or type(queue_len_raw) is not int
                or int(queue_len_raw) < 0
            ):
                state_error = ValueError(
                    f"invalid broker state response for socket {sock}"
                )
                if skip_invalid_sidecars:
                    continue
                raise state_error

            if log_path is not None:
                meta_log_off = int(log_path.stat().st_size)
            else:
                meta_log_off = 0

            queue_len_raw = resp.get("queue_len")
            queue_len = int(queue_len_raw) if type(queue_len_raw) is int and int(queue_len_raw) >= 0 else 0
            s = Session(
                session_id=session_id,
                thread_id=thread_id,
                broker_pid=broker_pid,
                codex_pid=codex_pid,
                agent_backend=agent_backend,
                owned=owned,
                backend=backend,
                transport=transport,
                supports_live_ui=supports_live_ui,
                ui_protocol_version=ui_protocol_version,
                start_ts=float(start_ts),
                cwd=str(cwd),
                log_path=log_path,
                sock_path=sock,
                session_path=session_path,
                busy=_state_busy_value(resp),
                queue_len=queue_len,
                token=(
                    resp.get("token")
                    if isinstance(resp.get("token"), (dict, type(None)))
                    else None
                ),
                meta_thinking=0,
                meta_tools=0,
                meta_system=0,
                meta_log_off=meta_log_off,
                model_provider=model_provider,
                preferred_auth_method=preferred_auth_method,
                model=model,
                reasoning_effort=reasoning_effort,
                service_tier=service_tier,
                tmux_session=tmux_session,
                tmux_window=tmux_window,
                resume_session_id=resume_session_id,
                pi_session_path_discovered=session_path_discovered,
            )
            with self._lock:
                prev = self._sessions.get(session_id)
                if not prev:
                    self._reset_log_caches(s, meta_log_off=meta_log_off)
                    s.model_provider = model_provider
                    s.preferred_auth_method = preferred_auth_method
                    s.model = model
                    s.reasoning_effort = reasoning_effort
                    s.service_tier = service_tier
                    self._sessions[session_id] = s
                else:
                    prev.sock_path = s.sock_path
                    prev.thread_id = s.thread_id
                    prev.backend = s.backend
                    prev.broker_pid = s.broker_pid
                    prev.codex_pid = s.codex_pid
                    prev.agent_backend = s.agent_backend
                    prev.owned = s.owned
                    prev.transport = s.transport
                    prev.supports_live_ui = s.supports_live_ui
                    prev.ui_protocol_version = s.ui_protocol_version
                    prev.start_ts = s.start_ts
                    prev.cwd = s.cwd
                    prev.busy = s.busy
                    prev.queue_len = s.queue_len
                    prev.token = s.token
                    self._apply_session_source(
                        prev, log_path=s.log_path, session_path=s.session_path
                    )
                    prev.model_provider = model_provider
                    prev.preferred_auth_method = preferred_auth_method
                    prev.model = model
                    prev.reasoning_effort = reasoning_effort
                    prev.service_tier = service_tier
                    prev.tmux_session = tmux_session
                    prev.tmux_window = tmux_window
                    prev.resume_session_id = resume_session_id
                    prev.pi_session_path_discovered = (
                        s.pi_session_path_discovered or prev.pi_session_path_discovered
                    )
        with self._lock:
            self._last_discover_ts = time.time()

    def _refresh_session_state(
        self, session_id: str, sock_path: Path, timeout_s: float = 0.4
    ) -> tuple[bool, BaseException | None]:
        try:
            resp = self._sock_call(sock_path, {"cmd": "state"}, timeout_s=timeout_s)
            _validated_session_state(resp)
        except Exception as e:
            return False, e
        publish_sessions = False
        publish_live = False
        publish_workspace = False
        durable_session_id: str | None = None
        with self._lock:
            s2 = self._sessions.get(session_id)
            if s2:
                next_busy = _state_busy_value(resp)
                next_queue_len = _state_queue_len_value(resp)
                next_token = resp.get("token") if isinstance(resp.get("token"), dict) else s2.token
                durable_session_id = self._durable_session_id_for_session(s2)
                publish_sessions = s2.busy != next_busy
                publish_live = publish_sessions or s2.queue_len != next_queue_len or next_token != s2.token
                publish_workspace = s2.queue_len != next_queue_len
                s2.busy = next_busy
                s2.queue_len = next_queue_len
                if isinstance(resp.get("token"), dict):
                    s2.token = resp.get("token")
        if durable_session_id is not None:
            if publish_sessions:
                _publish_sessions_invalidate(reason="session_state_changed")
            if publish_live:
                _publish_session_live_invalidate(
                    durable_session_id,
                    runtime_id=session_id,
                    reason="session_state_changed",
                )
            if publish_workspace:
                _publish_session_workspace_invalidate(
                    durable_session_id,
                    runtime_id=session_id,
                    reason="session_state_changed",
                )
        return True, None

    def _prune_dead_sessions(self) -> None:
        with self._lock:
            items = list(self._sessions.items())
        dead: list[tuple[str, Path]] = []
        for sid, s in items:
            if not s.sock_path.exists():
                dead.append((sid, s.sock_path))
                continue
            ok, _ = self._refresh_session_state(sid, s.sock_path, timeout_s=0.4)
            if ok:
                continue
            if not _probe_failure_safe_to_prune(
                broker_pid=s.broker_pid, codex_pid=s.codex_pid
            ):
                continue
            dead.append((sid, s.sock_path))
        if not dead:
            return
        dead_events: list[tuple[str, str]] = []
        with self._lock:
            for sid, _sock in dead:
                session = self._sessions.pop(sid, None)
                if session is not None:
                    dead_events.append((self._durable_session_id_for_session(session), sid))
        for sid, sock in dead:
            self._clear_deleted_session_state(sid)
            _unlink_quiet(sock)
            _unlink_quiet(sock.with_suffix(".json"))
        _publish_sessions_invalidate(reason="session_removed")
        for durable_session_id, runtime_id in dead_events:
            _publish_session_live_invalidate(
                durable_session_id,
                runtime_id=runtime_id,
                reason="session_removed",
            )
            _publish_session_workspace_invalidate(
                durable_session_id,
                runtime_id=runtime_id,
                reason="session_removed",
            )

    def _update_meta_counters(self) -> None:
        with self._lock:
            items = list(self._sessions.items())
        for sid, s in items:
            lp = s.log_path
            if (lp is None or (not lp.exists())) and s.backend == "pi":
                lp = s.session_path
            if lp is None or (not lp.exists()):
                continue
            sz = int(lp.stat().st_size)
            off = int(s.meta_log_off)
            reset_last_chat = False
            if sz < off:
                off = 0
                reset_last_chat = True

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
                d_th, d_tools, d_sys, chunk_chat_ts, token_update, _chat_events = (
                    _analyze_log_chunk(objs)
                )
                total_th += d_th
                total_tools += d_tools
                total_sys += d_sys
                if chunk_chat_ts is not None:
                    latest_chat_ts = (
                        chunk_chat_ts
                        if latest_chat_ts is None
                        else max(latest_chat_ts, chunk_chat_ts)
                    )
                if token_update is not None:
                    latest_token = token_update
                off = new_off
                loops += 1

            if latest_token is None and s.token is None:
                latest_token = _rollout_log._find_latest_token_update(lp)

            with self._lock:
                s2 = self._sessions.get(sid)
                if not s2:
                    continue
                if reset_last_chat:
                    s2.last_chat_ts = None
                    s2.last_chat_history_scanned = False
                if latest_chat_ts is not None:
                    s2.last_chat_ts = (
                        latest_chat_ts
                        if s2.last_chat_ts is None
                        else max(s2.last_chat_ts, latest_chat_ts)
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
        recovered_catalog = {}
        db = getattr(self, "_page_state_db", None)
        if isinstance(db, PageStateDB):
            recovered_catalog = db.load_sessions()
        if float(getattr(self, "_last_discover_ts", 0.0) or 0.0) <= 0.0:
            self._discover_existing_if_stale(force=True)
        self._update_meta_counters()
        files_dirty = False
        sidebar_dirty = False
        now_ts = time.time()
        with self._lock:
            items: list[dict[str, Any]] = []
            qmap = getattr(self, "_queues", None)
            meta_map = getattr(self, "_sidebar_meta", None)
            hidden_sessions = set(getattr(self, "_hidden_sessions", set()))
            active_durable_ids = {
                (_clean_optional_text(v.thread_id) or _clean_optional_text(v.session_id) or "")
                for v in self._sessions.values()
            }
            live_resume_keys: set[tuple[str, str]] = set()
            for s in self._sessions.values():
                thread_id = str(s.thread_id or "").strip()
                if thread_id:
                    live_resume_keys.add(
                        (
                            normalize_agent_backend(
                                s.agent_backend, default=s.backend or "codex"
                            ),
                            thread_id,
                        )
                    )
                cfg0 = self._harness.get(s.session_id)
                h_enabled = (
                    bool(cfg0.get("enabled")) if isinstance(cfg0, dict) else False
                )
                h_cooldown_minutes = (
                    _clean_harness_cooldown_minutes(cfg0.get("cooldown_minutes"))
                    if isinstance(cfg0, dict)
                    else HARNESS_DEFAULT_IDLE_MINUTES
                )
                h_remaining_injections = (
                    _clean_harness_remaining_injections(
                        cfg0.get("remaining_injections"), allow_zero=True
                    )
                    if isinstance(cfg0, dict)
                    else HARNESS_DEFAULT_MAX_INJECTIONS
                )
                log_exists = bool(s.log_path is not None and s.log_path.exists())
                if (
                    log_exists
                    and s.log_path is not None
                    and (
                        s.model_provider is None
                        or s.model is None
                        or s.reasoning_effort is None
                    )
                ):
                    try:
                        log_provider, log_model, log_effort = (
                            _read_run_settings_from_log(
                                s.log_path, agent_backend=s.agent_backend
                            )
                        )
                    except (FileNotFoundError, ValueError):
                        log_provider = log_model = log_effort = None
                    if s.model_provider is None:
                        s.model_provider = log_provider
                    if s.model is None:
                        s.model = log_model
                    if s.reasoning_effort is None:
                        s.reasoning_effort = log_effort
                if (
                    s.last_chat_ts is None
                    and log_exists
                    and s.log_path is not None
                    and (not s.last_chat_history_scanned)
                ):
                    # Discovery seeds offsets at EOF, so recover preexisting chat history once.
                    conv_ts = _last_conversation_ts_from_tail(s.log_path)
                    s.last_chat_history_scanned = True
                    if isinstance(conv_ts, (int, float)):
                        s.last_chat_ts = float(conv_ts)
                if s.backend == "pi" and s.session_path is not None and s.session_path.exists():
                    activity_ts = _session_file_activity_ts(s.session_path)
                    scanned_activity_ts = s.pi_attention_scan_activity_ts
                    should_refresh_attention = bool(
                        activity_ts is not None
                        and (
                            scanned_activity_ts is None
                            or float(activity_ts) > float(scanned_activity_ts)
                        )
                    )
                    if should_refresh_attention or (
                        s.last_chat_ts is None and (not s.last_chat_history_scanned)
                    ):
                        conv_ts = _last_attention_ts_from_pi_tail(s.session_path)
                        s.last_chat_history_scanned = True
                        s.pi_attention_scan_activity_ts = activity_ts
                        if isinstance(conv_ts, (int, float)):
                            s.last_chat_ts = (
                                float(conv_ts)
                                if s.last_chat_ts is None
                                else max(float(s.last_chat_ts), float(conv_ts))
                            )
                updated_ts = _display_updated_ts(s)
                canonical_cwd = _canonical_session_cwd(s.cwd)
                cwd_recent = _clean_recent_cwd(canonical_cwd)
                recent_map = getattr(self, "_recent_cwds", None)
                if cwd_recent is not None:
                    if not isinstance(recent_map, dict):
                        self._recent_cwds = {}
                        recent_map = self._recent_cwds
                    prev_recent_ts = recent_map.get(cwd_recent)
                    if prev_recent_ts is None or prev_recent_ts < updated_ts:
                        recent_map[cwd_recent] = updated_ts
                ref = self._page_state_ref_for_session(s)
                queue_len = 0
                if isinstance(qmap, dict):
                    q0 = qmap.get(s.session_id)
                    if not isinstance(q0, list) and ref is not None:
                        q0 = qmap.get(ref)
                    if isinstance(q0, list):
                        queue_len = len(q0)
                meta0 = None
                if isinstance(meta_map, dict):
                    meta0 = meta_map.get(s.session_id)
                    if meta0 is None and ref is not None:
                        meta0 = meta_map.get(ref)
                if not isinstance(meta0, dict):
                    meta0 = {}
                priority_offset = _clean_priority_offset(meta0.get("priority_offset"))
                snooze_until = _clean_snooze_until(meta0.get("snooze_until"))
                dependency_session_id = _clean_dependency_session_id(
                    meta0.get("dependency_session_id")
                )
                active_durable_ids = {
                    (_clean_optional_text(v.thread_id) or _clean_optional_text(v.session_id) or "")
                    for v in self._sessions.values()
                }
                if dependency_session_id == (_clean_optional_text(s.thread_id) or _clean_optional_text(s.session_id)) or (
                    dependency_session_id is not None
                    and dependency_session_id not in active_durable_ids
                ):
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
                cwd_path = _safe_expanduser(Path(canonical_cwd or s.cwd))
                if not cwd_path.is_absolute():
                    cwd_path = cwd_path.resolve()
                git_branch = _current_git_branch(cwd_path)
                if s.title is None:
                    try:
                        if (
                            s.backend == "pi"
                            and s.session_path is not None
                            and s.session_path.exists()
                        ):
                            title = _pi_session_name_from_session_file(s.session_path)
                            if title:
                                s.title = title
                    except Exception:
                        pass
                if s.first_user_message is None:
                    try:
                        preview = ""
                        if (
                            s.backend == "pi"
                            and s.session_path is not None
                            and s.session_path.exists()
                        ):
                            preview = _first_user_message_preview_from_pi_session(
                                s.session_path
                            )
                        elif log_exists and s.log_path is not None:
                            preview = _first_user_message_preview_from_log(s.log_path)
                        if preview:
                            s.first_user_message = preview
                    except Exception:
                        pass
                durable_session_id = ref[1] if ref is not None else self._durable_session_id_for_session(s)
                items.append(
                    {
                        "session_id": durable_session_id,
                        "runtime_id": s.session_id,
                        "thread_id": s.thread_id,
                        "backend": s.backend,
                        "pid": s.codex_pid,
                        "broker_pid": s.broker_pid,
                        "agent_backend": s.agent_backend,
                        "owned": s.owned,
                        "transport": s.transport,
                        "cwd": canonical_cwd,
                        "start_ts": s.start_ts,
                        "updated_ts": updated_ts,
                        "log_path": (
                            str(s.log_path) if s.log_path is not None else None
                        ),
                        "log_exists": log_exists,
                        "state_busy": bool(s.busy),
                        "queue_len": int(queue_len),
                        "token": s.token,
                        "thinking": int(s.meta_thinking),
                        "tools": int(s.meta_tools),
                        "system": int(s.meta_system),
                        "harness_enabled": h_enabled,
                        "harness_cooldown_minutes": h_cooldown_minutes,
                        "harness_remaining_injections": h_remaining_injections,
                        "alias": (self._aliases.get(s.session_id) if self._aliases.get(s.session_id) is not None else (self._aliases.get(ref) if ref is not None else None)),
                        "title": s.title or "",
                        "first_user_message": s.first_user_message or "",
                        "files": list(self._files.get(s.session_id, self._files.get(ref, []))) if ref is not None else list(self._files.get(s.session_id, [])),
                        "git_branch": git_branch,
                        "model_provider": s.model_provider,
                        "preferred_auth_method": s.preferred_auth_method,
                        "provider_choice": _provider_choice_for_backend(
                            backend=s.backend,
                            model_provider=s.model_provider,
                            preferred_auth_method=s.preferred_auth_method,
                        ),
                        "model": s.model,
                        "reasoning_effort": s.reasoning_effort,
                        "service_tier": s.service_tier,
                        "tmux_session": s.tmux_session,
                        "tmux_window": s.tmux_window,
                        "priority_offset": priority_offset,
                        "snooze_until": snooze_until,
                        "dependency_session_id": dependency_session_id,
                        "time_priority": time_priority,
                        "base_priority": base_priority,
                        "final_priority": final_priority,
                        "blocked": blocked,
                        "snoozed": snoozed,
                        "focused": bool(meta0.get("focused")),
                    }
                )

            for ref, record in recovered_catalog.items():
                backend, durable_session_id = ref
                if backend != "pi":
                    continue
                if (backend, durable_session_id) in live_resume_keys:
                    continue
                session_row_id = durable_session_id if record.pending_startup else _historical_session_id(backend, durable_session_id)
                if hidden_sessions.intersection(
                    self._hidden_session_keys(
                        session_row_id,
                        durable_session_id,
                        durable_session_id,
                        backend,
                    )
                ):
                    continue
                meta0 = meta_map.get(ref) if isinstance(meta_map, dict) else None
                if not isinstance(meta0, dict):
                    meta0 = {}
                priority_offset = _clean_priority_offset(meta0.get("priority_offset"))
                snooze_until = _clean_snooze_until(meta0.get("snooze_until"))
                dependency_session_id = _clean_dependency_session_id(
                    meta0.get("dependency_session_id")
                )
                if dependency_session_id is not None and dependency_session_id not in active_durable_ids:
                    dependency_session_id = None
                    if isinstance(meta_map, dict):
                        meta0.pop("dependency_session_id", None)
                        sidebar_dirty = True
                if snooze_until is not None and snooze_until <= now_ts:
                    snooze_until = None
                    if isinstance(meta_map, dict):
                        meta0.pop("snooze_until", None)
                        sidebar_dirty = True
                updated_ts = float(record.updated_at or record.created_at or now_ts)
                elapsed_s = max(0.0, now_ts - updated_ts)
                time_priority = _priority_from_elapsed_seconds(elapsed_s)
                base_priority = _clip01(time_priority + priority_offset)
                blocked = dependency_session_id is not None
                snoozed = snooze_until is not None and snooze_until > now_ts
                final_priority = 0.0 if (snoozed or blocked) else base_priority
                alias = None
                if isinstance(self._aliases, dict):
                    alias = self._aliases.get(ref)
                queue_rows = self._queues.get(ref, []) if isinstance(self._queues, dict) else []
                file_rows = self._files.get(ref, []) if isinstance(self._files, dict) else []
                cwd = record.cwd or ""
                history_cwd_path: Path | None = _safe_expanduser(Path(cwd)).resolve() if cwd else None
                git_branch = _current_git_branch(history_cwd_path) if history_cwd_path is not None else None
                items.append(
                    {
                        "session_id": session_row_id,
                        "runtime_id": None,
                        "thread_id": durable_session_id,
                        "resume_session_id": durable_session_id,
                        "backend": backend,
                        "pid": None,
                        "broker_pid": None,
                        "agent_backend": backend,
                        "owned": False,
                        "transport": None,
                        "cwd": cwd,
                        "start_ts": float(record.created_at or updated_ts),
                        "updated_ts": updated_ts,
                        "busy": False,
                        "queue_len": len(queue_rows) if isinstance(queue_rows, list) else 0,
                        "token": None,
                        "thinking": 0,
                        "tools": 0,
                        "system": 0,
                        "harness_enabled": False,
                        "harness_cooldown_minutes": HARNESS_DEFAULT_IDLE_MINUTES,
                        "harness_remaining_injections": HARNESS_DEFAULT_MAX_INJECTIONS,
                        "alias": alias,
                        "title": record.title or "",
                        "first_user_message": record.first_user_message or "",
                        "focused": bool(meta0.get("focused")),
                        "files": list(file_rows) if isinstance(file_rows, list) else [],
                        "git_branch": git_branch,
                        "model_provider": None,
                        "preferred_auth_method": None,
                        "provider_choice": None,
                        "model": None,
                        "reasoning_effort": None,
                        "service_tier": None,
                        "tmux_session": None,
                        "tmux_window": None,
                        "priority_offset": priority_offset,
                        "snooze_until": snooze_until,
                        "dependency_session_id": dependency_session_id,
                        "time_priority": time_priority,
                        "base_priority": base_priority,
                        "final_priority": final_priority,
                        "blocked": blocked,
                        "snoozed": snoozed,
                        "historical": not record.pending_startup,
                        "pending_startup": bool(record.pending_startup),
                        "source_path": record.source_path,
                        "session_path": record.source_path,
                    }
                )

            if bool(getattr(self, "_include_historical_sessions", False)):
                for hist in _historical_sidebar_items(
                    live_resume_keys=live_resume_keys,
                    now_ts=now_ts,
                ):
                    if hidden_sessions.intersection(
                        self._hidden_session_keys(
                            hist.get("session_id"),
                            hist.get("thread_id"),
                            hist.get("resume_session_id"),
                            hist.get("agent_backend"),
                        )
                    ):
                        continue
                    items.append(hist)

        out: list[dict[str, Any]] = []
        for it in items:
            sid = str(it["session_id"])
            agent_backend = normalize_agent_backend(
                it.get("agent_backend"), default="codex"
            )
            if it.get("historical"):
                out.append(_normalize_session_cwd_row(dict(it)))
                continue
            log_exists = bool(it.get("log_exists"))
            state_busy = bool(it.get("state_busy"))
            if not log_exists and it.get("backend") == "pi":
                s_obj = self._sessions.get(sid)
                busy_out = (
                    _display_pi_busy(s_obj, broker_busy=state_busy)
                    if s_obj is not None
                    else state_busy
                )
            elif not log_exists:
                busy_out = False
            else:
                idle_val = bool(self.idle_from_log(sid))
                if agent_backend == "pi":
                    busy_out = not idle_val
                else:
                    # When a log exists, unify semantics with /messages:
                    # busy if broker says busy OR log-derived idle is false.
                    busy_out = state_busy or (not idle_val)
            it2 = dict(it)
            it2.pop("log_exists", None)
            it2.pop("state_busy", None)
            it2["busy"] = bool(busy_out)
            out.append(_normalize_session_cwd_row(it2))
        for item in out:
            if item.get("busy") or int(item.get("queue_len", 0)) <= 0:
                continue
            self._maybe_drain_session_queue(str(item["session_id"]))
        if files_dirty:
            self._save_files()
        if sidebar_dirty:
            self._save_sidebar_meta()
        out.sort(
            key=lambda item: (
                -float(item.get("final_priority", 0.0)),
                -float(item.get("updated_ts", item.get("start_ts", 0.0))),
                -float(item.get("start_ts", 0.0)),
                0 if normalize_agent_backend(item.get("agent_backend"), default="codex") == "pi" else 1,
                str(item.get("session_id", "")),
            )
        )
        deduped: list[dict[str, Any]] = []
        seen_row_keys: set[str] = set()
        for item in out:
            row_key = _session_row_dedupe_key(item)
            if row_key in seen_row_keys:
                continue
            seen_row_keys.add(row_key)
            deduped.append(item)
        return deduped

    def get_session(self, session_id: str) -> Session | None:
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is None:
            return None
        with self._lock:
            return self._sessions.get(runtime_id)

    def refresh_session_meta(self, session_id: str, *, strict: bool = True) -> None:
        # The broker may rewrite the sock .json when Codex switches threads (/new, /resume).
        # Refresh the log path and thread id without requiring the UI to poll /api/sessions.
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is None:
            return
        with self._lock:
            s = self._sessions.get(runtime_id)
            if not s:
                return
            sock = s.sock_path
        try:
            meta_path = sock.with_suffix(".json")
            if not meta_path.exists():
                raise RuntimeError(f"missing metadata sidecar for socket {sock}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError(f"invalid metadata json for socket {sock}")

            thread_id = _clean_optional_text(meta.get("session_id")) or s.thread_id
            backend = normalize_agent_backend(
                meta.get("backend"),
                default=normalize_agent_backend(
                    meta.get("agent_backend"), default=s.backend
                ),
            )
            agent_backend = normalize_agent_backend(
                meta.get("agent_backend"), default=backend
            )
            owned = (
                (meta.get("owner") == "web")
                if isinstance(meta.get("owner"), str)
                else s.owned
            )
            transport, tmux_session, tmux_window = self._session_transport(meta=meta)
            supports_live_ui = (
                meta.get("supports_live_ui")
                if isinstance(meta.get("supports_live_ui"), bool)
                else None
            )
            ui_protocol_version_raw = meta.get("ui_protocol_version")
            ui_protocol_version = (
                ui_protocol_version_raw
                if type(ui_protocol_version_raw) is int
                else None
            )
            log_path = _metadata_log_path(meta=meta, backend=backend, sock=sock)
            session_path_discovered = False
            if backend == "pi":
                preferred_session_path: Path | None = s.session_path
                if strict or ("session_path" in meta):
                    preferred_session_path = _metadata_session_path(
                        meta=meta, backend=backend, sock=sock
                    )
                claimed: set[Path] | None = (
                    self._claimed_pi_session_paths(exclude_sid=session_id)
                    if preferred_session_path is None
                    else None
                )
                session_path, session_path_source = _resolve_pi_session_path(
                    thread_id=thread_id,
                    cwd=str(meta.get("cwd") or s.cwd),
                    start_ts=float(meta.get("start_ts") or s.start_ts),
                    preferred=preferred_session_path,
                    exclude=claimed,
                )
                if session_path is not None and session_path_source in {
                    "exact",
                    "discovered",
                }:
                    session_path_discovered = True
                    _patch_metadata_session_path(
                        sock,
                        session_path,
                        force=(
                            preferred_session_path is not None
                            and preferred_session_path != session_path
                        ),
                    )
            else:
                session_path = _metadata_session_path(
                    meta=meta, backend=backend, sock=sock
                )
            if log_path is not None and log_path.exists():
                thread_id, log_path = _coerce_main_thread_log(
                    thread_id=thread_id, log_path=log_path
                )

            cwd_raw = meta.get("cwd")
            if not isinstance(cwd_raw, str) or (not cwd_raw.strip()):
                raise ValueError(f"invalid cwd in metadata for socket {sock}")
            cwd = cwd_raw

            start_ts_raw = meta.get("start_ts")
            start_ts = (
                float(start_ts_raw)
                if isinstance(start_ts_raw, (int, float))
                else s.start_ts
            )
            resume_session_id = _clean_optional_text(meta.get("resume_session_id"))
            model_provider, preferred_auth_method, model, reasoning_effort = (
                self._session_run_settings(
                    backend=backend, meta=meta, log_path=log_path
                )
            )
            service_tier = _normalize_requested_service_tier(meta.get("service_tier"))
        except Exception as exc:
            if strict:
                raise
            self._quarantine_sidecar(sock, exc)
            return
        self._clear_sidecar_quarantine(sock)

        # Detect pi session switch: if thread_id changed, the underlying
        # session file has switched and we must re-discover session_path.
        pi_session_switched = False
        old_session_path: Path | None = None
        with self._lock:
            s2 = self._sessions.get(session_id)
            if (
                s2
                and backend == "pi"
                and s2.thread_id
                and thread_id
                and s2.thread_id != thread_id
                and s2.session_path is not None
            ):
                pi_session_switched = True
                old_session_path = s2.session_path

        if pi_session_switched and old_session_path is not None:
            claimed = self._claimed_pi_session_paths(exclude_sid=session_id)
            claimed.add(old_session_path)
            new_sp, new_sp_source = _resolve_pi_session_path(
                thread_id=thread_id,
                cwd=cwd,
                start_ts=start_ts,
                preferred=None,
                exclude=claimed,
            )
            if new_sp is not None and new_sp != old_session_path:
                session_path = new_sp
                if new_sp_source in {"exact", "discovered"}:
                    session_path_discovered = True
                _patch_metadata_session_path(sock, new_sp, force=True)

        with self._lock:
            s2 = self._sessions.get(session_id)
            if not s2:
                return
            if pi_session_switched:
                s2.session_path = None
                s2.pi_attention_scan_activity_ts = None
                self._reset_log_caches(s2, meta_log_off=0)
            s2.thread_id = str(thread_id)
            s2.agent_backend = agent_backend
            s2.backend = backend
            s2.cwd = str(cwd)
            s2.owned = bool(owned)
            s2.transport = transport
            s2.supports_live_ui = supports_live_ui
            s2.ui_protocol_version = ui_protocol_version
            self._apply_session_source(s2, log_path=log_path, session_path=session_path)
            s2.model_provider = model_provider
            s2.preferred_auth_method = preferred_auth_method
            s2.model = model
            s2.reasoning_effort = reasoning_effort
            s2.service_tier = service_tier
            s2.tmux_session = tmux_session
            s2.tmux_window = tmux_window
            s2.resume_session_id = resume_session_id
            s2.pi_session_path_discovered = bool(
                s2.pi_session_path_discovered or session_path_discovered
            )
        if self._queue_len(session_id) > 0:
            self._maybe_drain_session_queue(session_id)

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
            s.chat_index_scan_complete = bool(scan_complete) and (
                len(events) <= CHAT_INDEX_MAX_EVENTS
            )
            s.chat_index_log_off = int(log_off)
            if token_update is not None:
                s.token = token_update

    def _append_chat_events(
        self,
        session_id: str,
        new_events: list[dict[str, Any]],
        *,
        new_off: int,
        latest_token: dict[str, Any] | None,
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

    def _attach_notification_texts(
        self, events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        voice_push = getattr(self, "_voice_push", None)
        if voice_push is None:
            return list(events)
        out: list[dict[str, Any]] = []
        for ev in events:
            if not isinstance(ev, dict):
                out.append(ev)
                continue
            if (
                ev.get("role") != "assistant"
                or ev.get("message_class") != "final_response"
            ):
                out.append(ev)
                continue
            message_id = ev.get("message_id")
            if not isinstance(message_id, str) or not message_id:
                out.append(ev)
                continue
            notification_text = voice_push.notification_text_for_message(message_id)
            if not notification_text:
                out.append(ev)
                continue
            ev2 = dict(ev)
            ev2["notification_text"] = notification_text
            out.append(ev2)
        return out

    def _update_pi_last_chat_ts(
        self,
        session_id: str,
        events: list[dict[str, Any]],
        *,
        session_path: Path | None,
    ) -> None:
        if not events:
            return
        if not any(_is_attention_worthy_session_event(event) for event in events):
            return
        latest_chat_ts = _session_file_activity_ts(session_path)
        if latest_chat_ts is None:
            latest_chat_ts = _attention_updated_ts_from_events(events)
        if latest_chat_ts is None:
            return
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return
            s.last_chat_ts = (
                latest_chat_ts
                if s.last_chat_ts is None
                else max(s.last_chat_ts, latest_chat_ts)
            )

    def _ensure_pi_chat_index(
        self, session_id: str, *, min_events: int, before: int
    ) -> tuple[list[dict[str, Any]], int, bool, int, dict[str, Any]]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return [], 0, False, 0, {"tool_names": [], "last_tool": None}
            session_path = s.session_path
            scan_bytes = int(s.chat_index_scan_bytes) if s.chat_index_scan_bytes > 0 else (256 * 1024)
            idx_off = int(s.chat_index_log_off)
        if session_path is None or (not session_path.exists()):
            return [], 0, False, 0, {"tool_names": [], "last_tool": None}

        size = int(session_path.stat().st_size)
        if size < idx_off:
            self._set_chat_index_snapshot(
                session_id=session_id,
                events=[],
                token_update=None,
                scan_bytes=256 * 1024,
                scan_complete=False,
                log_off=0,
            )
            idx_off = 0

        with self._lock:
            s2 = self._sessions.get(session_id)
            ready = bool(s2 and (s2.chat_index_events is not None))
            cached_count = len(s2.chat_index_events) if s2 else 0
            scan_complete = bool(s2.chat_index_scan_complete) if s2 else False
        target_events = max(0, int(min_events) + max(0, int(before)))
        seed_diag: dict[str, Any] = {"tool_names": [], "last_tool": None}
        if (not ready) or ((target_events > cached_count) and (not scan_complete)):
            events, token_update, new_off, used_scan, complete, diag = _pi_messages.read_pi_message_tail_snapshot(
                session_path,
                min_events=max(20, target_events),
                initial_scan_bytes=max(256 * 1024, scan_bytes),
                max_scan_bytes=64 * 1024 * 1024,
            )
            seed_diag = diag
            self._set_chat_index_snapshot(
                session_id=session_id,
                events=events,
                token_update=token_update,
                scan_bytes=used_scan,
                scan_complete=complete,
                log_off=new_off,
            )
        with self._lock:
            s3 = self._sessions.get(session_id)
            if not s3:
                return [], 0, False, 0, {"tool_names": [], "last_tool": None}
            session_path2 = s3.session_path
            off3 = int(s3.chat_index_log_off)
            prev_events = list(s3.chat_index_events)
        if session_path2 is None or (not session_path2.exists()):
            return prev_events, off3, False, 0, {"tool_names": [], "last_tool": None}
        size2 = int(session_path2.stat().st_size)
        latest_diag = seed_diag
        if size2 > off3:
            events_delta, new_off, _meta_delta, _flags, latest_diag = _pi_messages.read_pi_message_delta(
                session_path2,
                offset=off3,
            )
            if events_delta:
                self._append_chat_events(
                    session_id,
                    events_delta,
                    new_off=new_off,
                    latest_token=None,
                )
            elif new_off > off3:
                self._append_chat_events(session_id, [], new_off=new_off, latest_token=None)
        with self._lock:
            s4 = self._sessions.get(session_id)
            if not s4:
                return prev_events, off3, False, 0, latest_diag
            events2 = list(s4.chat_index_events)
            off4 = int(s4.chat_index_log_off)
            scan_complete3 = bool(s4.chat_index_scan_complete)
        n = len(events2)
        b = max(0, int(before))
        end = max(0, n - b)
        start = max(0, end - max(20, int(min_events)))
        page = self._attach_notification_texts(events2[start:end])
        has_older = start > 0 or ((not scan_complete3) and bool(page))
        next_before = b + len(page) if has_older else 0
        return page, off4, has_older, next_before, latest_diag

    def _ensure_chat_index(
        self, session_id: str, *, min_events: int, before: int
    ) -> tuple[list[dict[str, Any]], int, bool, int, dict[str, Any] | None]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return [], 0, False, 0, None
            lp = s.log_path
            scan_bytes = (
                int(s.chat_index_scan_bytes)
                if s.chat_index_scan_bytes > 0
                else CHAT_INIT_SEED_SCAN_BYTES
            )
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
            ready = bool(s2 and (s2.chat_index_events is not None))
            cached_count = len(s2.chat_index_events) if s2 else 0
            scan_complete = bool(s2.chat_index_scan_complete) if s2 else False

        target_events = max(0, int(min_events) + max(0, int(before)))
        if (not ready) or ((target_events > cached_count) and (not scan_complete)):
            events, token_update, used_scan, complete, log_size = (
                _read_chat_tail_snapshot(
                    lp,
                    min_events=max(20, target_events),
                    initial_scan_bytes=max(CHAT_INIT_SEED_SCAN_BYTES, scan_bytes),
                    max_scan_bytes=CHAT_INIT_MAX_SCAN_BYTES,
                )
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
                events, token_update, used_scan, complete, log_size = (
                    _read_chat_tail_snapshot(
                        lp3,
                        min_events=max(20, target_events),
                        initial_scan_bytes=max(CHAT_INIT_SEED_SCAN_BYTES, scan_bytes),
                        max_scan_bytes=CHAT_INIT_MAX_SCAN_BYTES,
                    )
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
                    objs, new_off = _read_jsonl_from_offset(
                        lp3, cur, max_bytes=CHAT_INDEX_INCREMENT_BYTES
                    )
                    if new_off <= cur:
                        break
                    _th, _tools, _sys, _last_ts, token_update, new_events = (
                        _analyze_log_chunk(objs)
                    )
                    if token_update is not None:
                        latest_token = token_update
                    if new_events:
                        aggregated_events.extend(new_events)
                    cur = new_off
                    loops += 1
                self._append_chat_events(
                    session_id,
                    aggregated_events,
                    new_off=cur,
                    latest_token=latest_token,
                )

        with self._lock:
            s4 = self._sessions.get(session_id)
            if not s4:
                return prev_events, off3, False, 0, None
            events2 = list(s4.chat_index_events)
            log_off2 = int(s4.chat_index_log_off)
            scan_complete2 = bool(s4.chat_index_scan_complete)
            token2 = (
                s4.token if isinstance(s4.token, dict) or s4.token is None else None
            )

        n = len(events2)
        b = max(0, int(before))
        end = max(0, n - b)
        start = max(0, end - max(20, int(min_events)))
        page = self._attach_notification_texts(events2[start:end])
        has_older = (start > 0) or ((not scan_complete2) and bool(page))
        next_before = b + len(page) if has_older else 0
        return page, log_off2, has_older, next_before, token2

    def mark_log_delta(
        self, session_id: str, *, objs: list[dict[str, Any]], new_off: int
    ) -> None:
        _th, _tools, _sys, last_ts, token_update, new_events = _analyze_log_chunk(objs)
        model = None
        reasoning_effort = None
        for obj in reversed(objs):
            if not isinstance(obj, dict) or obj.get("type") != "turn_context":
                continue
            model, reasoning_effort = _turn_context_run_settings(obj.get("payload"))
            break
        self._append_chat_events(
            session_id, new_events, new_off=new_off, latest_token=token_update
        )
        durable_session_id: str | None = None
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                durable_session_id = self._durable_session_id_for_session(s)
                if isinstance(last_ts, (int, float)):
                    tsf = float(last_ts)
                    s.last_chat_ts = (
                        tsf if s.last_chat_ts is None else max(s.last_chat_ts, tsf)
                    )
                if model is not None:
                    s.model = model
                if reasoning_effort is not None:
                    s.reasoning_effort = reasoning_effort
                s.idle_cache_log_off = -1
        if durable_session_id is not None and (new_events or token_update is not None or model is not None or reasoning_effort is not None):
            _publish_session_live_invalidate(
                durable_session_id,
                runtime_id=session_id,
                reason="log_delta",
            )
            _publish_session_workspace_invalidate(
                durable_session_id,
                runtime_id=session_id,
                reason="log_delta",
            )
            if any(_is_attention_worthy_session_event(event) for event in new_events):
                _publish_sessions_invalidate(reason="conversation_changed")

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

    def get_messages_page(
        self,
        session_id: str,
        *,
        offset: int,
        init: bool,
        limit: int,
        before: int,
        view: str = "conversation",
    ) -> dict[str, Any]:
        historical_row = _historical_session_row(session_id)
        if historical_row is not None:
            historical_backend = normalize_agent_backend(
                historical_row.get("agent_backend", historical_row.get("backend")),
                default="codex",
            )
            if historical_backend != "pi":
                raise KeyError("unknown session")
            session_path_raw = historical_row.get("session_path")
            session_path = (
                Path(session_path_raw)
                if isinstance(session_path_raw, str) and session_path_raw
                else None
            )
            if session_path is None or (not session_path.exists()):
                return {
                    "thread_id": historical_row.get("resume_session_id"),
                    "log_path": str(session_path) if session_path is not None else None,
                    "offset": 0,
                    "events": [],
                    "meta_delta": {"thinking": 0, "tool": 0, "system": 0},
                    "turn_start": False,
                    "turn_end": False,
                    "turn_aborted": False,
                    "diag": {"pending_log": True},
                    "busy": False,
                    "queue_len": 0,
                    "token": None,
                    "has_older": False,
                    "next_before": 0,
                }
            if init and offset == 0:
                historical_events, new_off, has_older, next_before, diag = (
                    _pi_messages.read_pi_message_page(
                        session_path,
                        limit=limit,
                        before=before,
                    )
                )
                meta_delta = {"thinking": 0, "tool": 0, "system": 0}
                flags = {"turn_start": False, "turn_end": False, "turn_aborted": False}
            else:
                historical_events, new_off, meta_delta, flags, diag = (
                    _pi_messages.read_pi_message_delta(
                        session_path,
                        offset=offset,
                    )
                )
                has_older = False
                next_before = 0
            return {
                "thread_id": historical_row.get("resume_session_id"),
                "log_path": str(session_path),
                "offset": int(new_off),
                "events": historical_events,
                "meta_delta": meta_delta,
                "turn_start": bool(flags.get("turn_start")),
                "turn_end": bool(flags.get("turn_end")),
                "turn_aborted": bool(flags.get("turn_aborted")),
                "diag": diag,
                "busy": False,
                "queue_len": 0,
                "token": None,
                "has_older": bool(has_older),
                "next_before": int(next_before),
            }

        runtime_session_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_session_id is None:
            raise KeyError("unknown session")
        session_id = runtime_session_id
        self.refresh_session_meta(session_id, strict=False)
        s = self.get_session(session_id)
        if not s:
            raise KeyError("unknown session")

        state = self.get_state(session_id)
        if not isinstance(state, dict):
            raise ValueError("invalid broker state response")
        if "busy" not in state:
            raise ValueError("missing busy from broker state response")
        if "queue_len" not in state:
            raise ValueError("missing queue_len from broker state response")
        state_token = state.get("token")
        if not (isinstance(state_token, dict) or (state_token is None)):
            raise ValueError("invalid token from broker state response")

        if s.backend == "pi":
            diag = {"tool_names": [], "last_tool": None}
            flags = {"turn_start": False, "turn_end": False, "turn_aborted": False}
            meta_delta = {"thinking": 0, "tool": 0, "system": 0}
            new_off = 0
            has_older = False
            next_before = 0
            # Lazy discover session_path for PTY-wrapped pi (piox).
            # Only discover when session_path is truly unknown (None).
            # When session_path is set but the file doesn't exist yet
            # (Pi hasn't written to it), we must NOT fall back to discovery
            # because _discover_pi_session_for_cwd picks the most-recently-
            # modified file and would grab another session's file when two
            # piox sessions share the same CWD.
            if s.session_path is None and s.cwd:
                claimed = self._claimed_pi_session_paths(exclude_sid=session_id)
                discovered, discovered_source = _resolve_pi_session_path(
                    thread_id=s.thread_id,
                    cwd=s.cwd,
                    start_ts=s.start_ts,
                    preferred=None,
                    exclude=claimed,
                )
                if discovered is not None:
                    s.session_path = discovered
                    if discovered_source in {"exact", "discovered"}:
                        s.pi_session_path_discovered = True
                    _patch_metadata_session_path(s.sock_path, discovered)
            if s.session_path is not None and s.session_path.exists():
                if init and offset == 0:
                    events, new_off, has_older, next_before, diag = self._ensure_pi_chat_index(
                        session_id,
                        min_events=limit,
                        before=before,
                    )
                else:
                    events, new_off, meta_delta, flags, diag = (
                        _pi_messages.read_pi_message_delta(
                            s.session_path,
                            offset=offset,
                        )
                    )
                    if new_off > offset:
                        self._append_chat_events(
                            session_id,
                            events,
                            new_off=new_off,
                            latest_token=None,
                        )
                self._update_pi_last_chat_ts(
                    session_id, events, session_path=s.session_path
                )
            # Detect session switch via /resume: when the current session file
            # has gone stale (no writes) but a different, more recently modified
            # file exists in the same session directory, the pi CLI likely
            # switched sessions.  Re-discover and swap.
            if (
                s.pi_session_path_discovered
                and s.session_path is not None
                and s.cwd
                and not events
            ):
                sp_mtime = _safe_path_mtime(s.session_path)
                if sp_mtime is not None and (time.time() - sp_mtime) > 2.0:
                    old_sp = s.session_path
                    claimed = self._claimed_pi_session_paths(exclude_sid=session_id)
                    claimed.add(old_sp)
                    newer_sp, newer_sp_source = _resolve_pi_session_path(
                        thread_id=s.thread_id,
                        cwd=s.cwd,
                        start_ts=s.start_ts,
                        preferred=None,
                        exclude=claimed,
                    )
                    newer_sp_mtime = (
                        _safe_path_mtime(newer_sp) if newer_sp is not None else None
                    )
                    if (
                        newer_sp is not None
                        and newer_sp != old_sp
                        and (
                            newer_sp_source == "exact"
                            or (
                                newer_sp_mtime is not None and newer_sp_mtime > sp_mtime
                            )
                        )
                    ):
                        s.session_path = newer_sp
                        _patch_metadata_session_path(s.sock_path, newer_sp, force=True)
                        self._reset_log_caches(s, meta_log_off=0)
                        # Re-read from the newly discovered session file.
                        events, new_off, has_older, next_before, diag = self._ensure_pi_chat_index(
                            session_id,
                            min_events=limit,
                            before=0,
                        )
                        self._update_pi_last_chat_ts(
                            session_id, events, session_path=s.session_path
                        )
            pi_busy = _display_pi_busy(s, broker_busy=_state_busy_value(state))
            return {
                "thread_id": s.thread_id,
                "log_path": str(s.session_path) if s.session_path is not None else None,
                "offset": int(new_off),
                "events": events,
                "meta_delta": meta_delta,
                "turn_start": bool(flags.get("turn_start")),
                "turn_end": bool(flags.get("turn_end")),
                "turn_aborted": bool(flags.get("turn_aborted")),
                "diag": diag,
                "busy": pi_busy,
                "queue_len": int(self._queue_len(session_id)),
                "token": state_token,
                "has_older": bool(has_older),
                "next_before": int(next_before),
            }

        if s.log_path is None or (not s.log_path.exists()):
            return {
                "thread_id": s.thread_id,
                "log_path": None,
                "offset": 0,
                "events": [],
                "meta_delta": {"thinking": 0, "tool": 0, "system": 0},
                "turn_start": False,
                "turn_end": False,
                "turn_aborted": False,
                "diag": {"pending_log": True},
                "busy": False,
                "queue_len": int(self._queue_len(session_id)),
                "token": state_token,
                "has_older": False,
                "next_before": 0,
            }

        if init and offset == 0:
            events, new_off, has_older, next_before, token_update = (
                self._ensure_chat_index(
                    session_id,
                    min_events=limit,
                    before=before,
                )
            )
            meta_delta = {"thinking": 0, "tool": 0, "system": 0}
            flags = {"turn_start": False, "turn_end": False, "turn_aborted": False}
            diag = {"tool_names": [], "last_tool": None}
        else:
            has_older = False
            next_before = 0
            objs, new_off = _read_jsonl_from_offset(s.log_path, offset)
            events, meta_delta, flags, diag = _extract_chat_events(objs)
            token_update = _extract_token_update(objs)
            self.mark_log_delta(session_id, objs=objs, new_off=new_off)

        s2 = self.get_session(session_id)
        if token_update is not None and s2 is not None:
            s2.token = token_update
        idle_val = self.idle_from_log(session_id)
        busy_val = _state_busy_value(state) or (not bool(idle_val))
        token_val = (
            state_token
            if state_token is not None
            else token_update
            if isinstance(token_update, dict)
            else None
        )
        return {
            "thread_id": s.thread_id,
            "log_path": str(s.log_path),
            "offset": int(new_off),
            "events": events,
            "meta_delta": meta_delta,
            "turn_start": bool(flags.get("turn_start")),
            "turn_end": bool(flags.get("turn_end")),
            "turn_aborted": bool(flags.get("turn_aborted")),
            "diag": diag,
            "busy": bool(busy_val),
            "queue_len": int(self._queue_len(session_id)),
            "token": token_val,
            "has_older": bool(has_older),
            "next_before": int(next_before),
        }

    def _sock_call(
        self, sock_path: Path, req: dict[str, Any], timeout_s: float = 2.0
    ) -> dict[str, Any]:
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
            payload = json.loads(line.decode("utf-8"))
            return payload if isinstance(payload, dict) else {"error": "invalid response"}
        finally:
            s.close()

    def _kill_session_via_pids(self, s: Session) -> bool:
        group_alive = _process_group_alive(int(s.codex_pid))
        broker_alive = _pid_alive(int(s.broker_pid))
        if not group_alive and not broker_alive:
            _unlink_quiet(s.sock_path)
            _unlink_quiet(s.sock_path.with_suffix(".json"))
            return True
        if group_alive and (
            not _terminate_process_group(int(s.codex_pid), wait_seconds=1.0)
        ):
            return False
        if _pid_alive(int(s.broker_pid)) and (
            not _terminate_process(int(s.broker_pid), wait_seconds=1.0)
        ):
            return False
        group_dead = not _process_group_alive(int(s.codex_pid))
        broker_dead = not _pid_alive(int(s.broker_pid))
        if group_dead and broker_dead:
            _unlink_quiet(s.sock_path)
            _unlink_quiet(s.sock_path.with_suffix(".json"))
            return True
        return False

    def kill_session(self, session_id: str) -> bool:
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is None:
            return False
        with self._lock:
            s = self._sessions.get(runtime_id)
        if not s:
            return False
        try:
            resp = self._sock_call(s.sock_path, {"cmd": "shutdown"}, timeout_s=1.0)
        except Exception:
            return self._kill_session_via_pids(s)
        if resp.get("ok") is True:
            return True
        return self._kill_session_via_pids(s)

    def spawn_web_session(
        self,
        *,
        cwd: str,
        args: list[str] | None = None,
        agent_backend: str = "codex",
        resume_session_id: str | None = None,
        worktree_branch: str | None = None,
        model_provider: str | None = None,
        preferred_auth_method: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        create_in_tmux: bool = False,
        backend: str | None = None,
    ) -> dict[str, Any]:
        backend_name = normalize_agent_backend(
            backend, default=normalize_agent_backend(agent_backend, default="codex")
        )
        cwd_path = _resolve_dir_target(cwd, field_name="cwd")
        if not cwd_path.exists():
            try:
                cwd_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                detail = e.strerror or str(e)
                raise ValueError(
                    f"cwd could not be created: {cwd_path}: {detail}"
                ) from e
        if not cwd_path.is_dir():
            raise ValueError(f"cwd is not a directory: {cwd_path}")
        cwd3 = str(cwd_path)
        if backend_name == "pi":
            spawn_nonce = secrets.token_hex(8)
            created_pending_session_id: str | None = None
            if resume_session_id is not None:
                resume_id = _clean_optional_resume_session_id(resume_session_id)
                if not resume_id:
                    raise ValueError("resume_session_id must be a non-empty string")
                session_path: Path | None = None
                for row in _list_resume_candidates_for_cwd(
                    cwd3, limit=1000, backend="pi"
                ):
                    if row.get("session_id") != resume_id:
                        continue
                    raw_session_path = row.get("session_path")
                    if isinstance(raw_session_path, str) and raw_session_path:
                        session_path = Path(raw_session_path)
                        break
                if session_path is None:
                    raise ValueError(f"resume session not found for cwd: {resume_id}")
            else:
                created_pending_session_id = str(uuid.uuid4())
                session_path = _pi_new_session_file_for_cwd(cwd_path)
                _write_pi_session_header(
                    session_path,
                    session_id=created_pending_session_id,
                    cwd=cwd3,
                    provider=model_provider,
                    model_id=model,
                    thinking_level=reasoning_effort,
                )
                self._persist_durable_session_record(
                    DurableSessionRecord(
                        backend="pi",
                        session_id=created_pending_session_id,
                        cwd=cwd3,
                        source_path=str(session_path),
                        created_at=_safe_path_mtime(session_path),
                        updated_at=_safe_path_mtime(session_path),
                        pending_startup=True,
                    )
                )
            session_path.parent.mkdir(parents=True, exist_ok=True)
            argv = [
                sys.executable,
                "-m",
                "codoxear.pi_broker",
                "--cwd",
                str(cwd_path),
                "--session-file",
                str(session_path),
                "--",
                "-e",
                str(
                    Path(__file__).resolve().parent
                    / "pi_extensions"
                    / "ask_user_bridge.ts"
                ),
            ]
            env = dict(os.environ)
            if _DOTENV.exists():
                for k, v in _load_env_file(_DOTENV).items():
                    env.setdefault(k, v)
            env["CODEX_WEB_OWNER"] = "web"
            env["CODEX_WEB_SPAWN_NONCE"] = spawn_nonce
            env.setdefault("PI_HOME", str(PI_HOME))
            if create_in_tmux:
                tmux_bin = shutil.which("tmux")
                if tmux_bin is None:
                    if created_pending_session_id is not None:
                        self._delete_durable_session_record(("pi", created_pending_session_id))
                    raise ValueError("tmux is unavailable on this host")
                tmux_window = _safe_filename(
                    f"{Path(cwd3).name or 'session'}-{spawn_nonce[:6]}",
                    default="session",
                )
                env["CODEX_WEB_TRANSPORT"] = "tmux"
                env["CODEX_WEB_TMUX_SESSION"] = TMUX_SESSION_NAME
                env["CODEX_WEB_TMUX_WINDOW"] = tmux_window
                short_app_dir = _ensure_tmux_short_app_dir()
                inline_env = {
                    "CODEX_WEB_OWNER": "web",
                    "CODEX_WEB_AGENT_BACKEND": "pi",
                    "CODEX_WEB_TRANSPORT": "tmux",
                    "CODEX_WEB_TMUX_SESSION": TMUX_SESSION_NAME,
                    "CODEX_WEB_TMUX_WINDOW": tmux_window,
                    "CODEX_WEB_SPAWN_NONCE": spawn_nonce,
                    "CODOXEAR_APP_DIR": short_app_dir,
                    "PI_HOME": str(env["PI_HOME"]),
                }
                repo_root = Path(__file__).resolve().parent.parent
                inline_argv = [
                    "env",
                    *[f"{key}={value}" for key, value in inline_env.items()],
                    *argv,
                ]
                shell_cmd = f"cd {shlex.quote(str(repo_root))} && exec {shlex.join(inline_argv)}"
                has_session = subprocess.run(
                    [tmux_bin, "has-session", "-t", TMUX_SESSION_NAME],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                )
                if has_session.returncode == 0:
                    tmux_argv = [
                        tmux_bin,
                        "new-window",
                        "-d",
                        "-P",
                        "-F",
                        "#{pane_id}",
                        "-t",
                        f"{TMUX_SESSION_NAME}:",
                        "-n",
                        tmux_window,
                        shell_cmd,
                    ]
                else:
                    tmux_argv = [
                        tmux_bin,
                        "new-session",
                        "-d",
                        "-P",
                        "-F",
                        "#{pane_id}",
                        "-s",
                        TMUX_SESSION_NAME,
                        "-n",
                        tmux_window,
                        shell_cmd,
                    ]
                tmux_proc = subprocess.run(
                    tmux_argv, capture_output=True, text=True, env=env, check=False
                )
                if tmux_proc.returncode != 0:
                    if created_pending_session_id is not None:
                        self._delete_durable_session_record(("pi", created_pending_session_id))
                    detail = (
                        tmux_proc.stderr
                        or tmux_proc.stdout
                        or f"exit status {tmux_proc.returncode}"
                    ).strip()
                    raise RuntimeError(f"tmux launch failed: {detail}")
                if created_pending_session_id is not None:
                    threading.Thread(
                        target=self._finalize_pending_pi_spawn,
                        kwargs={
                            "spawn_nonce": spawn_nonce,
                            "durable_session_id": created_pending_session_id,
                            "cwd": cwd3,
                            "session_path": session_path,
                            "proc": None,
                        },
                        daemon=True,
                    ).start()
                    return {
                        "session_id": created_pending_session_id,
                        "runtime_id": None,
                        "backend": "pi",
                        "pending_startup": True,
                        "tmux_session": TMUX_SESSION_NAME,
                        "tmux_window": tmux_window,
                    }
                meta = _wait_for_spawned_broker_meta(spawn_nonce)
                payload = _spawn_result_from_meta(meta)
                return {
                    **payload,
                    "tmux_session": TMUX_SESSION_NAME,
                    "tmux_window": tmux_window,
                }
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
                if created_pending_session_id is not None:
                    self._delete_durable_session_record(("pi", created_pending_session_id))
                raise RuntimeError(f"spawn failed: {e}") from e
            if proc.stderr is not None:
                threading.Thread(
                    target=_drain_stream, args=(proc.stderr,), daemon=True
                ).start()
            threading.Thread(target=proc.wait, daemon=True).start()
            if created_pending_session_id is not None:
                threading.Thread(
                    target=self._finalize_pending_pi_spawn,
                    kwargs={
                        "spawn_nonce": spawn_nonce,
                        "durable_session_id": created_pending_session_id,
                        "cwd": cwd3,
                        "session_path": session_path,
                        "proc": proc,
                    },
                    daemon=True,
                ).start()
                payload = {
                    "session_id": created_pending_session_id,
                    "runtime_id": None,
                    "backend": "pi",
                    "pending_startup": True,
                }
                _publish_sessions_invalidate(reason="session_created")
                return payload
            _wait_or_raise(proc, label="pi broker", timeout_s=1.5)
            meta = _wait_for_spawned_broker_meta(spawn_nonce)
            payload = _spawn_result_from_meta(meta)
            _publish_sessions_invalidate(reason="session_created")
            return payload
        if resume_session_id is not None and worktree_branch is not None:
            raise ValueError("worktree_branch cannot be used when resuming a session")
        spawn_cwd = cwd_path
        if worktree_branch is not None:
            spawn_cwd = _create_git_worktree(cwd_path, worktree_branch)

        argv = [sys.executable, "-m", "codoxear.broker", "--cwd", str(spawn_cwd), "--"]
        codex_args: list[str] = []
        resume_row: dict[str, Any] | None = None
        if backend_name == "codex":
            # Web-owned Codex sessions need a remote-safe mode that does not block on TUI confirmations.
            codex_args = [
                "-c",
                _codex_trust_override_for_path(spawn_cwd),
                "--dangerously-bypass-approvals-and-sandbox",
            ]
            if model is not None:
                codex_args.extend(["--model", model])
            if reasoning_effort is not None:
                codex_args.extend(
                    ["-c", f'model_reasoning_effort="{reasoning_effort}"']
                )
            if model_provider is not None:
                codex_args.extend(["-c", f'model_provider="{model_provider}"'])
            if preferred_auth_method is not None:
                codex_args.extend(
                    ["-c", f'preferred_auth_method="{preferred_auth_method}"']
                )
            if service_tier is not None:
                codex_args.extend(["-c", f'service_tier="{service_tier}"'])
        else:
            if preferred_auth_method is not None:
                raise ValueError("preferred_auth_method is not supported for pi")
            if service_tier is not None:
                raise ValueError("service_tier is not supported for pi")
            if model_provider is not None:
                codex_args.extend(["--provider", model_provider])
            if model is not None:
                codex_args.extend(["--model", model])
            if reasoning_effort is not None:
                codex_args.extend(["--thinking", reasoning_effort])
        if resume_session_id is not None:
            resume_id = _clean_optional_resume_session_id(resume_session_id)
            if not resume_id:
                raise ValueError("resume_session_id must be a non-empty string")
            found = False
            for row in _list_resume_candidates_for_cwd(
                cwd3, agent_backend=backend_name, limit=1000
            ):
                if row.get("session_id") == resume_id:
                    found = True
                    resume_row = row
                    break
            if not found:
                raise ValueError(f"resume session not found for cwd: {resume_id}")
            if backend_name == "codex":
                codex_args.extend(["resume", resume_id])
            else:
                resume_target = (
                    str(resume_row.get("log_path") or "").strip()
                    if isinstance(resume_row, dict)
                    else ""
                )
                codex_args.extend(["--session", resume_target or resume_id])
        codex_args.extend(args or [])
        argv.extend(codex_args)

        env = dict(os.environ)
        if _DOTENV.exists():
            for k, v in _load_env_file(_DOTENV).items():
                env.setdefault(k, v)
        env["CODEX_WEB_OWNER"] = "web"
        env["CODEX_WEB_AGENT_BACKEND"] = backend_name
        if backend_name == "codex":
            env.setdefault("CODEX_HOME", str(CODEX_HOME))
            env.pop("PI_HOME", None)
        else:
            env.setdefault("PI_HOME", str(PI_HOME))
            env.pop("CODEX_HOME", None)
        env.pop("CODEX_WEB_MODEL_PROVIDER", None)
        env.pop("CODEX_WEB_PREFERRED_AUTH_METHOD", None)
        env.pop("CODEX_WEB_MODEL", None)
        env.pop("CODEX_WEB_REASONING_EFFORT", None)
        env.pop("CODEX_WEB_SERVICE_TIER", None)
        env.pop("CODEX_WEB_TRANSPORT", None)
        env.pop("CODEX_WEB_TMUX_SESSION", None)
        env.pop("CODEX_WEB_TMUX_WINDOW", None)
        env.pop("CODEX_WEB_SPAWN_NONCE", None)
        env.pop("CODEX_WEB_RESUME_SESSION_ID", None)
        env.pop("CODEX_WEB_RESUME_LOG_PATH", None)
        spawn_nonce = secrets.token_hex(8)
        env["CODEX_WEB_SPAWN_NONCE"] = spawn_nonce
        if model_provider is not None:
            env["CODEX_WEB_MODEL_PROVIDER"] = model_provider
        if preferred_auth_method is not None:
            env["CODEX_WEB_PREFERRED_AUTH_METHOD"] = preferred_auth_method
        if model is not None:
            env["CODEX_WEB_MODEL"] = model
        if reasoning_effort is not None:
            env["CODEX_WEB_REASONING_EFFORT"] = reasoning_effort
        if service_tier is not None:
            env["CODEX_WEB_SERVICE_TIER"] = service_tier
        if resume_session_id is not None:
            env["CODEX_WEB_RESUME_SESSION_ID"] = resume_session_id
        if create_in_tmux:
            tmux_bin = shutil.which("tmux")
            if tmux_bin is None:
                raise ValueError("tmux is unavailable on this host")
            tmux_window = _safe_filename(
                f"{Path(spawn_cwd).name or 'session'}-{spawn_nonce[:6]}",
                default="session",
            )
            env["CODEX_WEB_TRANSPORT"] = "tmux"
            env["CODEX_WEB_TMUX_SESSION"] = TMUX_SESSION_NAME
            env["CODEX_WEB_TMUX_WINDOW"] = tmux_window
            env["CODEX_WEB_SPAWN_NONCE"] = spawn_nonce
            short_app_dir = _ensure_tmux_short_app_dir()
            inline_env = {
                "CODEX_WEB_OWNER": "web",
                "CODEX_WEB_AGENT_BACKEND": backend_name,
                "CODEX_WEB_TRANSPORT": "tmux",
                "CODEX_WEB_TMUX_SESSION": TMUX_SESSION_NAME,
                "CODEX_WEB_TMUX_WINDOW": tmux_window,
                "CODEX_WEB_SPAWN_NONCE": spawn_nonce,
                "CODOXEAR_APP_DIR": short_app_dir,
            }
            if backend_name == "codex":
                inline_env["CODEX_HOME"] = str(env["CODEX_HOME"])
            else:
                inline_env["PI_HOME"] = str(env["PI_HOME"])
            if resume_session_id is not None:
                inline_env["CODEX_WEB_RESUME_SESSION_ID"] = resume_session_id
            if model_provider is not None:
                inline_env["CODEX_WEB_MODEL_PROVIDER"] = model_provider
            if preferred_auth_method is not None:
                inline_env["CODEX_WEB_PREFERRED_AUTH_METHOD"] = preferred_auth_method
            if model is not None:
                inline_env["CODEX_WEB_MODEL"] = model
            if reasoning_effort is not None:
                inline_env["CODEX_WEB_REASONING_EFFORT"] = reasoning_effort
            if service_tier is not None:
                inline_env["CODEX_WEB_SERVICE_TIER"] = service_tier
            codex_bin = _clean_optional_text(os.environ.get("CODEX_BIN"))
            if codex_bin is not None:
                inline_env["CODEX_BIN"] = codex_bin
            repo_root = Path(__file__).resolve().parent.parent
            inline_argv = [
                "env",
                *[f"{key}={value}" for key, value in inline_env.items()],
                *argv,
            ]
            shell_cmd = (
                f"cd {shlex.quote(str(repo_root))} && exec {shlex.join(inline_argv)}"
            )
            has_session = subprocess.run(
                [tmux_bin, "has-session", "-t", TMUX_SESSION_NAME],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            if has_session.returncode == 0:
                tmux_argv = [
                    tmux_bin,
                    "new-window",
                    "-d",
                    "-P",
                    "-F",
                    "#{pane_id}",
                    "-t",
                    f"{TMUX_SESSION_NAME}:",
                    "-n",
                    tmux_window,
                    shell_cmd,
                ]
            else:
                tmux_argv = [
                    tmux_bin,
                    "new-session",
                    "-d",
                    "-P",
                    "-F",
                    "#{pane_id}",
                    "-s",
                    TMUX_SESSION_NAME,
                    "-n",
                    tmux_window,
                    shell_cmd,
                ]
            tmux_proc = subprocess.run(
                tmux_argv, capture_output=True, text=True, env=env, check=False
            )
            if tmux_proc.returncode != 0:
                detail = (
                    tmux_proc.stderr
                    or tmux_proc.stdout
                    or f"exit status {tmux_proc.returncode}"
                ).strip()
                raise RuntimeError(f"tmux launch failed: {detail}")
            meta = _wait_for_spawned_broker_meta(spawn_nonce)
            payload = _spawn_result_from_meta(meta)
            return {
                **payload,
                "tmux_session": TMUX_SESSION_NAME,
                "tmux_window": tmux_window,
            }

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
            threading.Thread(
                target=_drain_stream, args=(proc.stderr,), daemon=True
            ).start()

        # Prevent zombies when the broker exits.
        threading.Thread(target=proc.wait, daemon=True).start()
        meta = _wait_for_spawned_broker_meta(spawn_nonce)
        payload = _spawn_result_from_meta(meta)
        _publish_sessions_invalidate(reason="session_created")
        return payload

    def delete_session(self, session_id: str) -> bool:
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is None:
            ref = self._page_state_ref_for_session_id(session_id)
            if ref is None:
                return False
            backend, durable_id = ref
            self._hide_session_identity_values(
                session_id,
                durable_id,
                durable_id if backend == "pi" else None,
                backend,
            )
            self._delete_durable_session_record(ref)
            self._clear_deleted_session_state(session_id)
            return True
        with self._lock:
            s = self._sessions.get(runtime_id)
        if not s:
            return False
        ok = self.kill_session(runtime_id)
        if ok:
            # Hide the session immediately so stale sidecars do not repopulate the
            # sidebar before the broker and child process have fully exited.
            self._hide_session_identity(s)
            self.files_clear(runtime_id)
            self._clear_deleted_session_state(runtime_id)
            with self._lock:
                self._sessions.pop(runtime_id, None)
        return ok

    def send(self, session_id: str, text: str) -> dict[str, Any]:
        historical_row = _historical_session_row(session_id)
        if historical_row is None and self._runtime_session_id_for_identifier(session_id) is None:
            listed_row = _listed_session_row(self, session_id)
            if isinstance(listed_row, dict) and listed_row.get("historical"):
                historical_row = listed_row
        if historical_row is not None:
            backend = normalize_agent_backend(
                historical_row.get("agent_backend", historical_row.get("backend")),
                default="codex",
            )
            if backend != "pi":
                raise KeyError("unknown session")
            cwd = _clean_optional_text(historical_row.get("cwd"))
            resume_session_id = _clean_optional_text(
                historical_row.get("resume_session_id")
            )
            if cwd is None or resume_session_id is None:
                raise ValueError("historical session is missing resume metadata")
            spawn_res = self.spawn_web_session(
                cwd=cwd,
                backend="pi",
                resume_session_id=resume_session_id,
            )
            self._discover_existing(force=True, skip_invalid_sidecars=True)
            live_runtime_id = _clean_optional_text(spawn_res.get("runtime_id"))
            live_session_id = _clean_optional_text(spawn_res.get("session_id"))
            if live_runtime_id is None or live_session_id is None:
                raise RuntimeError("spawned session did not return session identities")
            if self._runtime_session_id_for_identifier(live_runtime_id) is None:
                raise RuntimeError("spawned session is not yet discoverable")
            resp = self.send(live_runtime_id, text)
            out = dict(resp)
            out["session_id"] = live_session_id
            out["runtime_id"] = live_runtime_id
            out["backend"] = "pi"
            return out

        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is None:
            raise KeyError("unknown session")
        with self._lock:
            s = self._sessions.get(runtime_id)
            if not s:
                raise KeyError("unknown session")
            durable_session_id = self._durable_session_id_for_session(s)
        transport_state, transport_error = self._probe_bridge_transport(runtime_id)
        if transport_state == "dead":
            with self._lock:
                self._sessions.pop(runtime_id, None)
            self._clear_deleted_session_state(runtime_id)
            _unlink_quiet(s.sock_path)
            _unlink_quiet(s.sock_path.with_suffix(".json"))
            raise KeyError("unknown session")
        request = self._enqueue_outbound_request(runtime_id, text)
        return {
            "ok": True,
            "accepted": True,
            "request_id": request.request_id,
            "delivery_state": request.state,
            "session_id": durable_session_id,
            "runtime_id": runtime_id,
            "backend": s.backend,
            "transport_state": transport_state,
            "transport_error": transport_error,
        }

    def enqueue(self, session_id: str, text: str) -> dict[str, Any]:
        historical_row = _historical_session_row(session_id)
        if historical_row is None and self._runtime_session_id_for_identifier(session_id) is None:
            listed_row = _listed_session_row(self, session_id)
            if isinstance(listed_row, dict) and listed_row.get("historical"):
                historical_row = listed_row
        if historical_row is not None:
            backend = normalize_agent_backend(
                historical_row.get("agent_backend", historical_row.get("backend")),
                default="codex",
            )
            if backend != "pi":
                raise KeyError("unknown session")
            cwd = _clean_optional_text(historical_row.get("cwd"))
            resume_session_id = _clean_optional_text(
                historical_row.get("resume_session_id")
            )
            if cwd is None or resume_session_id is None:
                raise ValueError("historical session is missing resume metadata")
            spawn_res = self.spawn_web_session(
                cwd=cwd,
                backend="pi",
                resume_session_id=resume_session_id,
            )
            self._discover_existing(force=True, skip_invalid_sidecars=True)
            live_runtime_id = _clean_optional_text(spawn_res.get("runtime_id"))
            live_session_id = _clean_optional_text(spawn_res.get("session_id"))
            if live_runtime_id is None or live_session_id is None:
                raise RuntimeError("spawned session did not return session identities")
            if self._runtime_session_id_for_identifier(live_runtime_id) is None:
                raise RuntimeError("spawned session is not yet discoverable")
            resp = self.enqueue(live_runtime_id, text)
            out = dict(resp)
            out["session_id"] = live_session_id
            out["runtime_id"] = live_runtime_id
            out["backend"] = "pi"
            return out

        # Persist queued messages on the server so they survive broker restarts.
        return self._queue_enqueue_local(session_id, text)

    def queue_list(self, session_id: str) -> list[str]:
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is None:
            raise KeyError("unknown session")
        return self._queue_list_local(runtime_id)

    def queue_delete(self, session_id: str, index: int) -> dict[str, Any]:
        return self._queue_delete_local(session_id, int(index))

    def queue_update(self, session_id: str, index: int, text: str) -> dict[str, Any]:
        return self._queue_update_local(session_id, int(index), text)

    def get_state(self, session_id: str) -> dict[str, Any]:
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is None:
            raise KeyError("unknown session")
        with self._lock:
            s = self._sessions.get(runtime_id)
            if not s:
                raise KeyError("unknown session")
            sock = s.sock_path
        cached_state = {
            "busy": bool(s.busy),
            "queue_len": int(s.queue_len),
            "token": s.token,
        }
        try:
            resp = self._sock_call(sock, {"cmd": "state"}, timeout_s=1.5)
            _validated_session_state(resp)
        except Exception:
            if not _pid_alive(s.broker_pid) and not _pid_alive(s.codex_pid):
                with self._lock:
                    self._sessions.pop(runtime_id, None)
                self._clear_deleted_session_state(runtime_id)
                _unlink_quiet(sock)
                _unlink_quiet(sock.with_suffix(".json"))
                raise KeyError("unknown session")
            # Broker is alive but socket temporarily unavailable (e.g. during
            # session switch).  Return a degraded state instead of crashing.
            return cached_state
        with self._lock:
            s2 = self._sessions.get(runtime_id)
            if s2:
                s2.busy = _state_busy_value(resp)
                s2.queue_len = _state_queue_len_value(resp)
                if "token" in resp:
                    tok = resp.get("token")
                    if isinstance(tok, dict):
                        s2.token = tok
                return resp
        return cached_state

    def get_ui_state(self, session_id: str) -> dict[str, Any]:
        return _pi_ui_bridge.get_ui_state(self, session_id)

    def get_session_commands(self, session_id: str) -> dict[str, Any]:
        return _pi_ui_bridge.get_session_commands(self, session_id)

    def submit_ui_response(
        self, session_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return _pi_ui_bridge.submit_ui_response(self, session_id, payload)

    def get_tail(self, session_id: str) -> str:
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is None:
            raise KeyError("unknown session")
        with self._lock:
            s = self._sessions.get(runtime_id)
            if not s:
                raise KeyError("unknown session")
            sock = s.sock_path
        try:
            resp = self._sock_call(sock, {"cmd": "tail"}, timeout_s=1.5)
        except Exception:
            if not _pid_alive(s.broker_pid) and not _pid_alive(s.codex_pid):
                with self._lock:
                    self._sessions.pop(runtime_id, None)
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
        runtime_id = self._runtime_session_id_for_identifier(session_id)
        if runtime_id is None:
            raise KeyError("unknown session")
        with self._lock:
            s = self._sessions.get(runtime_id)
            if not s:
                raise KeyError("unknown session")
            sock = s.sock_path
        try:
            resp = self._sock_call(sock, {"cmd": "keys", "seq": seq}, timeout_s=2.0)
        except Exception:
            if not _pid_alive(s.broker_pid) and not _pid_alive(s.codex_pid):
                with self._lock:
                    self._sessions.pop(runtime_id, None)
                _unlink_quiet(sock)
                _unlink_quiet(sock.with_suffix(".json"))
                raise KeyError("unknown session")
            raise
        err = resp.get("error")
        if isinstance(err, str) and err:
            raise ValueError(err)
        return resp

    def mark_turn_complete(self, session_id: str, payload: dict[str, Any]) -> None:
        return


MANAGER = SessionManager()


def _static_asset_version(static_dir: Path = STATIC_DIR) -> str:
    base = static_dir.resolve()
    digest = hashlib.sha256()
    for rel in STATIC_ASSET_VERSION_FILES:
        path = (base / rel).resolve()
        if not str(path).startswith(str(base)):
            raise ValueError(f"static asset escaped static dir: {path}")
        if not path.is_file():
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def _read_static_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix != ".html":
        return data
    replacements = {
        STATIC_ASSET_VERSION_PLACEHOLDER.encode("ascii"): _static_asset_version(
            path.parent
        ).encode("ascii"),
        STATIC_ATTACH_MAX_BYTES_PLACEHOLDER.encode("ascii"): str(
            ATTACH_UPLOAD_MAX_BYTES
        ).encode("ascii"),
    }
    for placeholder, value in replacements.items():
        if placeholder in data:
            data = data.replace(placeholder, value)
    return data


def _is_path_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _candidate_web_dist_dirs() -> list[Path]:
    out: list[Path] = []
    for candidate in (WEB_DIST_DIR, PACKAGED_WEB_DIST_DIR):
        if candidate not in out:
            out.append(candidate)
    return out


def _served_web_dist_dir() -> Path | None:
    for candidate in _candidate_web_dist_dirs():
        if (candidate / "index.html").is_file():
            return candidate
    return None


def _vite_manifest_path(dist_dir: Path | None = None) -> Path:
    if dist_dir is None:
        for candidate_dir in _candidate_web_dist_dirs():
            vite_manifest = candidate_dir / ".vite" / "manifest.json"
            if vite_manifest.is_file():
                return vite_manifest
            manifest = candidate_dir / "manifest.json"
            if manifest.is_file():
                return manifest
        dist_dir = WEB_DIST_DIR
    vite_manifest = dist_dir / ".vite" / "manifest.json"
    if vite_manifest.is_file():
        return vite_manifest
    return dist_dir / "manifest.json"


def _hashed_asset_suffix(asset_path: str) -> str | None:
    stem = Path(asset_path).stem
    if "-" not in stem:
        return None
    suffix = stem.rsplit("-", 1)[-1].strip()
    return suffix or None


def _manifest_asset_token(asset_path: str) -> str | None:
    asset_path = asset_path.strip()
    if not asset_path:
        return None
    hashed = _hashed_asset_suffix(asset_path)
    if hashed:
        return hashed
    return hashlib.sha256(asset_path.encode("utf-8")).hexdigest()[:12]


def _asset_version_from_manifest(manifest: dict[str, object]) -> str:
    if not isinstance(manifest, dict):
        return "dev"
    entry = manifest.get("src/main.tsx")
    if not isinstance(entry, dict):
        entry = manifest.get("index.html")
    if not isinstance(entry, dict):
        for value in manifest.values():
            if isinstance(value, dict) and value.get("file"):
                entry = value
                break
    if not isinstance(entry, dict):
        return "dev"
    parts: list[str] = []
    js_token = _manifest_asset_token(str(entry.get("file") or ""))
    if js_token:
        parts.append(js_token)
    css_files = entry.get("css")
    if isinstance(css_files, list):
        for css_path in css_files:
            css_token = _manifest_asset_token(str(css_path or ""))
            if css_token:
                parts.append(css_token)
    return "-".join(parts) or "dev"


def _pi_model_context_window(provider: str | None, model: str | None) -> int | None:
    return _pi_model_context_window_impl(provider, model)


def _current_app_version() -> str:
    served_dist_dir = _served_web_dist_dir()
    if served_dist_dir is not None:
        manifest_path = _vite_manifest_path(served_dist_dir)
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            manifest_data = {}
        version = _asset_version_from_manifest(
            manifest_data if isinstance(manifest_data, dict) else {}
        )
        if version != "dev":
            return version
    return _static_asset_version()


def _rewrite_web_index_html(data: str) -> str:
    if not URL_PREFIX:
        return data
    prefix_body = re.escape(URL_PREFIX.lstrip("/"))
    pattern = rf'((?:href|src|content)=["\'])/(?!/|{prefix_body}/)'
    return re.sub(pattern, rf"\1{URL_PREFIX}/", data)


def _read_web_index() -> tuple[str, str]:
    dist_dir = _served_web_dist_dir()
    if dist_dir is not None:
        dist_index = dist_dir / "index.html"
        return _rewrite_web_index_html(
            dist_index.read_text(encoding="utf-8")
        ), "text/html; charset=utf-8"
    legacy_index = LEGACY_STATIC_DIR / "index.html"
    return _read_static_bytes(legacy_index).decode("utf-8"), "text/html; charset=utf-8"


def _resolve_public_web_asset(rel: str) -> Path | None:
    rel_path = Path(rel.lstrip("/"))
    served_dist_dir = _served_web_dist_dir()
    if served_dist_dir is not None:
        dist_candidate = (served_dist_dir / rel_path).resolve()
        if (
            _is_path_within(served_dist_dir.resolve(), dist_candidate)
            and dist_candidate.is_file()
        ):
            return dist_candidate
    legacy_candidate = (LEGACY_STATIC_DIR / rel_path).resolve()
    if (
        _is_path_within(LEGACY_STATIC_DIR.resolve(), legacy_candidate)
        and legacy_candidate.is_file()
    ):
        return legacy_candidate
    return None


def _content_type_for_path(path: Path) -> str:
    if path.suffix == ".html":
        return "text/html; charset=utf-8"
    if path.suffix == ".js":
        return "text/javascript; charset=utf-8"
    if path.suffix == ".css":
        return "text/css; charset=utf-8"
    if path.suffix == ".webmanifest":
        return "application/manifest+json; charset=utf-8"
    if path.suffix == ".svg":
        return "image/svg+xml; charset=utf-8"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _cache_control_for_path(path: Path) -> str:
    if "/assets/" in path.as_posix():
        return "public, max-age=31536000, immutable"
    return "no-store"


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "codoxear/0.1"

    def _send_bytes(
        self, data: bytes, ctype: str, *, cache_control: str = "no-store"
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache_control)
        if cache_control == "no-store":
            # UI is used for interactive debugging; serve HTML and legacy assets without caching
            # so changes show up immediately on refresh.
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(data)

    def _send_path(self, path: Path) -> None:
        data = _read_static_bytes(path)
        self._send_bytes(
            data,
            _content_type_for_path(path),
            cache_control=_cache_control_for_path(path),
        )

    def _send_static(self, rel: str) -> None:
        path = (STATIC_DIR / rel.lstrip("/")).resolve()
        if not _is_path_within(STATIC_DIR.resolve(), path):
            self.send_error(404)
            return
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        self._send_path(path)

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
            for route_module in (
                _http_assets_routes,
                _http_auth_routes,
                _http_events_routes,
                _http_notification_routes,
                _http_session_read_routes,
                _http_file_routes,
            ):
                if route_module.handle_get(self, path, u):
                    return
            self.send_error(404)
        except Exception as e:
            traceback.print_exc()
            _json_response(
                self, 500, {"error": str(e), "trace": traceback.format_exc()}
            )

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
            for route_module in (
                _http_auth_routes,
                _http_notification_routes,
                _http_session_write_routes,
                _http_file_routes,
            ):
                if route_module.handle_post(self, path, u):
                    return
            self.send_error(404)
        except KeyError:
            _json_response(self, 404, {"error": "unknown session"})
        except Exception as e:
            traceback.print_exc()
            _json_response(
                self, 500, {"error": str(e), "trace": traceback.format_exc()}
            )

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
