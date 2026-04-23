#!/usr/bin/env python3
from __future__ import annotations

import base64
import copy
import errno
import fnmatch
import gzip
import hashlib
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
from .runtime import ServerRuntime, build_server_runtime
from .sessions import background as _session_background
from .sessions import lifecycle as _session_lifecycle
from .sessions import listing as _session_listing
from .sessions import live_payloads as _session_live_payloads
from .sessions import message_history as _message_history
from .sessions import page_state as _page_state
from .sessions import payloads as _session_payloads
from .sessions import pi_session_files as _pi_session_files
from .sessions import resume_candidates as _resume_candidates
from .sessions import session_catalog as _session_catalog
from .sessions import session_control as _session_control
from .sessions import sidebar_state as _sidebar_state_module
from .sessions import transport as _session_transport
from .sessions.sidebar_state import SidebarStateFacade
from .util import default_app_dir as _default_app_dir
from .workspace import file_access as _workspace_file_access
from .workspace import file_search as _workspace_file_search
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
PI_MODELS_CACHE_NAMESPACE = "pi_models"

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


def _start_proc_stderr_drain(proc: subprocess.Popen[Any]) -> None:
    stderr = getattr(proc, "stderr", None)
    if stderr is None:
        return
    threading.Thread(target=_drain_stream, args=(stderr,), daemon=True).start()


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
    if isinstance(exc, (FileNotFoundError, ConnectionRefusedError, TimeoutError)):
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
    return _workspace_file_access.resolve_unique_bare_filename(search_root, raw_path)


def _resolve_tracked_file_by_basename(session_id: str, raw_path: str) -> Path | None:
    return _workspace_file_access.resolve_tracked_file_by_basename(
        RUNTIME,
        session_id,
        raw_path,
    )


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
    return _workspace_file_search.file_search_score(candidate, query)


def _push_file_search_match(
    heap: list[tuple[int, str]], *, path: str, score: int, limit: int
) -> None:
    return _workspace_file_search._push_file_search_match(
        heap,
        path=path,
        score=score,
        limit=limit,
    )


def _finish_file_search(
    heap: list[tuple[int, str]], *, mode: str, query: str, scanned: int, truncated: bool
) -> dict[str, Any]:
    return _workspace_file_search._finish_file_search(
        heap,
        mode=mode,
        query=query,
        scanned=scanned,
        truncated=truncated,
    )


def _search_walk_relative_files(
    root: Path, *, query: str, limit: int
) -> dict[str, Any]:
    return _workspace_file_search.search_walk_relative_files(
        RUNTIME,
        root,
        query=query,
        limit=limit,
    )


def _search_git_relative_files(cwd: Path, *, query: str, limit: int) -> dict[str, Any]:
    return _workspace_file_search.search_git_relative_files(
        RUNTIME,
        cwd,
        query=query,
        limit=limit,
    )


def _search_session_relative_files(
    base: Path, *, query: str, limit: int = FILE_SEARCH_LIMIT
) -> dict[str, Any]:
    return _workspace_file_search.search_session_relative_files(
        RUNTIME,
        base,
        query=query,
        limit=limit,
    )


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
SESSION_HISTORY_PAGE_SIZE = 300
SESSION_LIST_FALLBACK_GROUP_KEY = "__no_working_directory__"


def _clean_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _normalize_session_cwd_row(row: dict[str, Any]) -> dict[str, Any]:
    return _session_listing.normalize_session_cwd_row(RUNTIME, row)


def _frontend_session_list_row(row: dict[str, Any]) -> dict[str, Any]:
    return _session_listing.frontend_session_list_row(RUNTIME, row)


def _session_list_payload(
    rows: list[dict[str, Any]],
    *,
    group_key: str | None = None,
    offset: int = 0,
    limit: int = SESSION_LIST_PAGE_SIZE,
    group_offset: int = 0,
    group_limit: int = SESSION_LIST_RECENT_GROUP_LIMIT,
) -> dict[str, Any]:
    return _session_listing.session_list_payload(
        RUNTIME,
        rows,
        group_key=group_key,
        offset=offset,
        limit=limit,
        group_offset=group_offset,
        group_limit=group_limit,
    )


def _listed_session_row(manager: "SessionManager", session_id: str) -> dict[str, Any] | None:
    return _session_catalog.service(manager).listed_session_row(session_id)


def _session_details_payload(
    manager: "SessionManager", session_id: str
) -> dict[str, Any]:
    return _session_payloads.session_details_payload(RUNTIME, manager, session_id)


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
    return _workspace_file_access.resolve_client_file_path(
        RUNTIME,
        session_id=session_id,
        raw_path=raw_path,
    )


def _inspect_openable_file(path_obj: Path) -> tuple[bytes, int, str, str | None]:
    return _workspace_file_access.inspect_openable_file(RUNTIME, path_obj)


def _inspect_path_metadata(path_obj: Path) -> tuple[int, str, str | None]:
    return _workspace_file_access.inspect_path_metadata(RUNTIME, path_obj)


def _read_client_file_view(path_obj: Path) -> ClientFileView:
    return _workspace_file_access.read_client_file_view(RUNTIME, path_obj)


def _read_text_or_image(path_obj: Path) -> tuple[str, int, str | None, bytes | None]:
    return _workspace_file_access.read_text_or_image(RUNTIME, path_obj)


def _read_downloadable_file(path_obj: Path) -> tuple[bytes, int]:
    return _workspace_file_access.read_downloadable_file(path_obj)


def _inspect_client_path(path_obj: Path) -> tuple[int, str, str | None]:
    return _workspace_file_access.inspect_client_path(RUNTIME, path_obj)


def _download_disposition(path_obj: Path) -> str:
    return _workspace_file_access.download_disposition(path_obj)


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


def _fallback_path_mtime(path: Path) -> float | None:
    return _resume_candidates.fallback_path_mtime(path)


def _last_pi_conversation_ts(path: Path) -> float | None:
    return _resume_candidates.last_pi_conversation_ts(RUNTIME, path)


def _resume_candidate_updated_ts(path: Path, *, agent_backend: str) -> float | None:
    return _resume_candidates.resume_candidate_updated_ts(RUNTIME, path, agent_backend=agent_backend)


def _resume_candidate_from_log(
    log_path: Path, *, agent_backend: str = "codex"
) -> dict[str, Any] | None:
    return _resume_candidates.resume_candidate_from_log(RUNTIME, log_path, agent_backend=agent_backend)


def _pi_native_session_dir_for_cwd(cwd: str | Path) -> Path:
    return _pi_session_files.pi_native_session_dir_for_cwd(RUNTIME, cwd)


def _pi_new_session_file_for_cwd(cwd: str | Path) -> Path:
    return _pi_session_files.pi_new_session_file_for_cwd(RUNTIME, cwd)


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
    return _pi_session_files.write_pi_session_header(RUNTIME, session_path, session_id=session_id, cwd=cwd, parent_session=parent_session, provider=provider, model_id=model_id, thinking_level=thinking_level)


def _pi_session_history_glob(session_path: Path) -> str:
    return _pi_session_files.pi_session_history_glob(session_path)


def _pi_session_has_handoff_history(session_path: Path) -> bool:
    return _pi_session_files.pi_session_has_handoff_history(session_path)


def _next_pi_handoff_history_path(session_path: Path) -> Path:
    return _pi_session_files.next_pi_handoff_history_path(session_path)


def _copy_file_atomic(source_path: Path, target_path: Path) -> None:
    return _pi_session_files.copy_file_atomic(RUNTIME, source_path, target_path)


def _append_pi_user_message(session_path: Path, *, text: str) -> None:
    return _pi_session_files.append_pi_user_message(RUNTIME, session_path, text=text)


def _pi_handoff_message_text(
    *, source_session_id: str, history_path: Path, cwd: str
) -> str:
    return _pi_session_files.pi_handoff_message_text(
        source_session_id=source_session_id,
        history_path=history_path,
        cwd=cwd,
    )


def _write_pi_handoff_session(
    session_path: Path,
    *,
    session_id: str,
    cwd: str,
    source_session_id: str,
    history_path: Path,
    provider: str | None = None,
    model_id: str | None = None,
    thinking_level: str | None = None,
) -> None:
    return _pi_session_files.write_pi_handoff_session(RUNTIME, session_path, session_id=session_id, cwd=cwd, source_session_id=source_session_id, history_path=history_path, provider=provider, model_id=model_id, thinking_level=thinking_level)


def _pi_session_name_from_session_file(
    session_path: Path, *, max_scan_bytes: int = 512 * 1024
) -> str:
    return _pi_session_files.pi_session_name_from_session_file(
        RUNTIME,
        session_path,
        max_scan_bytes=max_scan_bytes,
    )



def _pi_resume_candidate_from_session_file(session_path: Path) -> dict[str, Any] | None:
    return _resume_candidates.pi_resume_candidate_from_session_file(RUNTIME, session_path)


def _discover_pi_session_for_cwd(
    cwd: str, start_ts: float, *, exclude: set[Path] | None = None
) -> Path | None:
    return _resume_candidates.discover_pi_session_for_cwd(RUNTIME, cwd, start_ts, exclude=exclude)


def _resolve_pi_session_path(
    *,
    thread_id: str | None,
    cwd: str,
    start_ts: float,
    preferred: Path | None = None,
    exclude: set[Path] | None = None,
) -> tuple[Path | None, str | None]:
    return _resume_candidates.resolve_pi_session_path(RUNTIME, thread_id=thread_id, cwd=cwd, start_ts=start_ts, preferred=preferred, exclude=exclude)


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
    return _resume_candidates.list_resume_candidates_for_cwd(RUNTIME, cwd, limit=limit, offset=offset, backend=backend, agent_backend=agent_backend)


def _iter_all_resume_candidates(*, limit: int = 200) -> list[dict[str, Any]]:
    return _resume_candidates.iter_all_resume_candidates(RUNTIME, limit=limit)


def _historical_session_id(backend: str, resume_session_id: str) -> str:
    return _session_listing.historical_session_id(RUNTIME, backend, resume_session_id)


def _parse_historical_session_id(session_id: str) -> tuple[str, str] | None:
    return _session_listing.parse_historical_session_id(RUNTIME, session_id)


def _historical_session_row(session_id: str) -> dict[str, Any] | None:
    return _session_listing.historical_session_row(RUNTIME, session_id)


def _historical_sidebar_items(
    *, live_resume_keys: set[tuple[str, str]], now_ts: float
) -> list[dict[str, Any]]:
    return _session_listing.historical_sidebar_items(
        RUNTIME,
        live_resume_keys=live_resume_keys,
        now_ts=now_ts,
    )


def _first_user_message_preview_from_log(
    log_path: Path, *, max_scan_bytes: int = 256 * 1024
) -> str:
    return _session_listing.first_user_message_preview_from_log(
        RUNTIME,
        log_path,
        max_scan_bytes=max_scan_bytes,
    )


def _first_user_message_preview_from_pi_session(
    session_path: Path, *, max_scan_bytes: int = 256 * 1024
) -> str:
    return _session_listing.first_user_message_preview_from_pi_session(
        RUNTIME,
        session_path,
        max_scan_bytes=max_scan_bytes,
    )


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
    return _session_payloads.session_context_usage_payload(RUNTIME, s, token_val)



def _session_turn_timing_payload(
    s: Session,
    events: list[dict[str, Any]],
    *,
    busy: bool,
) -> dict[str, Any] | None:
    return _session_payloads.session_turn_timing_payload(RUNTIME, s, events, busy=busy)


def _session_diagnostics_payload(
    manager: "SessionManager", session_id: str, s: Session, state: dict[str, Any]
) -> dict[str, Any]:
    return _session_payloads.session_diagnostics_payload(RUNTIME, manager, session_id, s, state)


def _session_workspace_payload(
    manager: "SessionManager", session_id: str
) -> dict[str, Any]:
    return _session_payloads.session_workspace_payload(RUNTIME, manager, session_id)


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
        RUNTIME,
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
    return _session_live_payloads.pi_live_messages_payload(RUNTIME, manager, session, offset=offset)


def _merge_pi_live_message_events(
    durable_events: list[dict[str, Any]], streamed_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return _session_live_payloads.merge_pi_live_message_events(RUNTIME, durable_events, streamed_events)


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
        self._runtime = build_server_runtime(
            sys.modules[__name__],
            manager=self,
            event_hub=EVENT_HUB,
        )
        globals()["MANAGER"] = self
        globals()["RUNTIME"] = self._runtime
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
        if getattr(self, "_runtime", None) is None:
            self._runtime = RUNTIME
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
        return _session_catalog.service(self).runtime_session_id_for_identifier(session_id)

    def _durable_session_id_for_identifier(self, session_id: str) -> str | None:
        return _session_catalog.service(self).durable_session_id_for_identifier(session_id)

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
        _session_background.set_bridge_transport_state(
            self,
            runtime_id,
            state=state,
            error=error,
            checked_ts=checked_ts,
        )

    def _probe_bridge_transport(
        self, session_id: str, *, force_rpc: bool = False
    ) -> tuple[str, str | None]:
        return _session_background.probe_bridge_transport(
            self,
            session_id,
            force_rpc=force_rpc,
        )

    def _enqueue_outbound_request(self, runtime_id: str, text: str) -> BridgeOutboundRequest:
        return _session_background.enqueue_outbound_request(self, runtime_id, text)

    def _fail_outbound_request(self, request: BridgeOutboundRequest, error: str) -> None:
        _session_background.fail_outbound_request(self, request, error)

    def _mark_outbound_request_buffered_for_compaction(
        self, request: BridgeOutboundRequest
    ) -> None:
        _session_background.mark_outbound_request_buffered_for_compaction(self, request)

    def _maybe_drain_outbound_request(self, runtime_id: str) -> bool:
        return _session_background.maybe_drain_outbound_request(self, runtime_id)

    def _catalog_record_for_ref(self, ref: SessionRef) -> DurableSessionRecord | None:
        return _session_lifecycle.service(self).catalog_record_for_ref(ref)

    def _refresh_durable_session_catalog(self, *, force: bool = False) -> None:
        _session_lifecycle.service(self).refresh_durable_session_catalog(force=force)

    def _page_state_ref_for_session_id(self, session_id: str) -> SessionRef | None:
        return _session_catalog.service(self).page_state_ref_for_session_id(session_id)

    def _persist_durable_session_record(self, row: DurableSessionRecord) -> None:
        db = getattr(self, "_page_state_db", None)
        if isinstance(db, PageStateDB):
            db.upsert_session(row)

    def _delete_durable_session_record(self, ref: SessionRef | None) -> None:
        db = getattr(self, "_page_state_db", None)
        if ref is not None and isinstance(db, PageStateDB):
            db.delete_session(ref)

    def _wait_for_live_session(
        self,
        durable_session_id: str,
        *,
        timeout_s: float = 8.0,
    ) -> Session:
        return _session_lifecycle.service(self).wait_for_live_session(
            durable_session_id,
            timeout_s=timeout_s,
        )

    def _copy_session_ui_identity(
        self,
        *,
        source_session_id: str,
        target_session_id: str,
    ) -> str | None:
        return _session_lifecycle.service(self).copy_session_ui_identity(
            source_session_id=source_session_id,
            target_session_id=target_session_id,
        )

    def _capture_runtime_bound_restart_state(
        self, runtime_id: str, ref: SessionRef
    ) -> dict[str, Any]:
        return _session_lifecycle.service(self).capture_runtime_bound_restart_state(
            runtime_id,
            ref,
        )

    def _stage_runtime_bound_restart_state(
        self, runtime_id: str, ref: SessionRef, state: dict[str, Any]
    ) -> None:
        _session_lifecycle.service(self).stage_runtime_bound_restart_state(
            runtime_id,
            ref,
            state,
        )

    def _restore_runtime_bound_restart_state(
        self, runtime_id: str, ref: SessionRef, state: dict[str, Any]
    ) -> None:
        _session_lifecycle.service(self).restore_runtime_bound_restart_state(
            runtime_id,
            ref,
            state,
        )

    def _finalize_pending_pi_restart_state(
        self,
        *,
        durable_session_id: str,
        ref: SessionRef,
        state: dict[str, Any],
        timeout_s: float = 8.0,
    ) -> None:
        try:
            session = self._wait_for_live_session(
                durable_session_id,
                timeout_s=timeout_s,
            )
        except Exception:
            return
        self._restore_runtime_bound_restart_state(session.session_id, ref, state)

    def restart_session(self, session_id: str) -> dict[str, Any]:
        return _session_control.service(self).restart_session(session_id)

    def handoff_session(self, session_id: str) -> dict[str, Any]:
        return _session_control.service(self).handoff_session(session_id)

    def _finalize_pending_pi_spawn(
        self,
        *,
        spawn_nonce: str,
        durable_session_id: str,
        cwd: str,
        session_path: Path,
        proc: subprocess.Popen[bytes] | None = None,
        delete_on_failure: bool = True,
        restore_record_on_failure: DurableSessionRecord | None = None,
    ) -> None:
        _session_lifecycle.service(self).finalize_pending_pi_spawn(
            spawn_nonce=spawn_nonce,
            durable_session_id=durable_session_id,
            cwd=cwd,
            session_path=session_path,
            proc=proc,
            delete_on_failure=delete_on_failure,
            restore_record_on_failure=restore_record_on_failure,
        )

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
        _session_lifecycle.service(self).reset_log_caches(s, meta_log_off=meta_log_off)

    def _session_source_changed(
        self, s: Session, *, log_path: Path | None, session_path: Path | None
    ) -> bool:
        return _session_lifecycle.service(self).session_source_changed(
            s,
            log_path=log_path,
            session_path=session_path,
        )

    def _claimed_pi_session_paths(self, *, exclude_sid: str = "") -> set[Path]:
        return _session_lifecycle.service(self).claimed_pi_session_paths(
            exclude_sid=exclude_sid,
        )

    def _apply_session_source(
        self, s: Session, *, log_path: Path | None, session_path: Path | None
    ) -> None:
        _session_lifecycle.service(self).apply_session_source(
            s,
            log_path=log_path,
            session_path=session_path,
        )

    def _session_run_settings(
        self,
        *,
        meta: dict[str, Any],
        log_path: Path | None,
        backend: str | None = None,
        agent_backend: str | None = None,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        return _session_lifecycle.service(self).session_run_settings(
            meta=meta,
            log_path=log_path,
            backend=backend,
            agent_backend=agent_backend,
        )

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
        _page_state.service(self).load_harness()

    def _save_harness(self) -> None:
        _page_state.service(self).save_harness()

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
        _page_state.service(self).clear_deleted_session_state(session_id)

    def _load_files(self) -> None:
        _page_state.service(self).load_files()

    def _save_files(self) -> None:
        _page_state.service(self).save_files()

    def _load_queues(self) -> None:
        _page_state.service(self).load_queues()

    def _save_queues(self) -> None:
        _page_state.service(self).save_queues()

    def _load_recent_cwds(self) -> None:
        _page_state.service(self).load_recent_cwds()

    def _save_recent_cwds(self) -> None:
        _page_state.service(self).save_recent_cwds()

    def _load_cwd_groups(self) -> None:
        _page_state.service(self).load_cwd_groups()

    def _save_cwd_groups(self) -> None:
        _page_state.service(self).save_cwd_groups()

    def cwd_groups_get(self) -> dict[str, dict[str, Any]]:
        return _page_state.service(self).cwd_groups_get()

    def _prune_stale_workspace_dirs(self) -> None:
        _page_state.service(self).prune_stale_workspace_dirs()

    def _known_cwd_group_keys(self) -> set[str]:
        return _page_state.service(self).known_cwd_group_keys()

    def cwd_group_set(
        self, cwd: str, label: str | None = None, collapsed: bool | None = None
    ) -> tuple[str, dict[str, Any]]:
        return _page_state.service(self).cwd_group_set(
            cwd,
            label=label,
            collapsed=collapsed,
        )

    def _remember_recent_cwd(self, cwd: Any, *, ts: Any = None) -> bool:
        return _page_state.service(self).remember_recent_cwd(cwd, ts=ts)

    def _backfill_recent_cwds_from_logs(self) -> None:
        _page_state.service(self).backfill_recent_cwds_from_logs()

    def recent_cwds(self, *, limit: int = RECENT_CWD_MAX) -> list[str]:
        return _page_state.service(self).recent_cwds(limit=limit)

    def _queue_len(self, session_id: str) -> int:
        return _page_state.service(self).queue_len(session_id)

    def _queue_list_local(self, session_id: str) -> list[str]:
        return _page_state.service(self).queue_list_local(session_id)

    def _queue_enqueue_local(self, session_id: str, text: str) -> dict[str, Any]:
        return _page_state.service(self).queue_enqueue_local(session_id, text)

    def _queue_delete_local(self, session_id: str, index: int) -> dict[str, Any]:
        return _page_state.service(self).queue_delete_local(session_id, index)

    def _queue_update_local(
        self, session_id: str, index: int, text: str
    ) -> dict[str, Any]:
        return _page_state.service(self).queue_update_local(session_id, index, text)

    def _files_key_for_session(self, session_id: str) -> tuple[str, SessionRef, "Session"]:
        return _page_state.service(self).files_key_for_session(session_id)

    def files_get(self, session_id: str) -> list[str]:
        return _page_state.service(self).files_get(session_id)

    def files_add(self, session_id: str, path: str) -> list[str]:
        return _page_state.service(self).files_add(session_id, path)

    def files_clear(self, session_id: str) -> None:
        _page_state.service(self).files_clear(session_id)

    def harness_get(self, session_id: str) -> dict[str, Any]:
        return _page_state.service(self).harness_get(session_id)

    def harness_set(
        self,
        session_id: str,
        *,
        enabled: bool | None = None,
        request: str | None = None,
        cooldown_minutes: int | None = None,
        remaining_injections: int | None = None,
    ) -> dict[str, Any]:
        return _page_state.service(self).harness_set(
            session_id,
            enabled=enabled,
            request=request,
            cooldown_minutes=cooldown_minutes,
            remaining_injections=remaining_injections,
        )

    def _session_display_name(self, session_id: str) -> str:
        return _session_background.session_display_name(self, session_id)

    def _observe_rollout_delta(
        self, session_id: str, *, objs: list[dict[str, Any]], new_off: int
    ) -> None:
        _session_background.observe_rollout_delta(
            self,
            session_id,
            objs=objs,
            new_off=new_off,
        )

    def _voice_push_scan_loop(self) -> None:
        _session_background.voice_push_scan_loop(self)

    def _voice_push_scan_sweep(self) -> None:
        _session_background.voice_push_scan_sweep(self)

    def _harness_loop(self) -> None:
        _session_background.harness_loop(self)

    def _harness_sweep(self) -> None:
        _session_background.harness_sweep(self)

    def _queue_loop(self) -> None:
        _session_background.queue_loop(self)

    def _maybe_drain_session_queue(
        self, session_id: str, *, now_ts: float | None = None
    ) -> bool:
        return _session_background.maybe_drain_session_queue(
            self,
            session_id,
            now_ts=now_ts,
        )

    def _queue_sweep(self) -> None:
        _session_background.queue_sweep(self)

    def _discover_existing(
        self, *, force: bool = False, skip_invalid_sidecars: bool = False
    ) -> None:
        if getattr(self, "_runtime", None) is None:
            self._runtime = RUNTIME
        _session_catalog.service(self).discover_existing(
            force=force,
            skip_invalid_sidecars=skip_invalid_sidecars,
        )

    def _refresh_session_state(
        self, session_id: str, sock_path: Path, timeout_s: float = 0.4
    ) -> tuple[bool, BaseException | None]:
        if getattr(self, "_runtime", None) is None:
            self._runtime = RUNTIME
        return _session_catalog.service(self).refresh_session_state(
            session_id,
            sock_path,
            timeout_s=timeout_s,
        )

    def _prune_dead_sessions(self) -> None:
        if getattr(self, "_runtime", None) is None:
            self._runtime = RUNTIME
        _session_catalog.service(self).prune_dead_sessions()

    def _update_meta_counters(self) -> None:
        _session_background.update_meta_counters(self)

    def list_sessions(self) -> list[dict[str, Any]]:
        if getattr(self, "_runtime", None) is None:
            self._runtime = RUNTIME
        return _session_catalog.service(self).list_sessions()

    def get_session(self, session_id: str) -> Session | None:
        return _session_catalog.service(self).get_session(session_id)

    def refresh_session_meta(self, session_id: str, *, strict: bool = True) -> None:
        if getattr(self, "_runtime", None) is None:
            self._runtime = RUNTIME
        _session_catalog.service(self).refresh_session_meta(session_id, strict=strict)

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
        _message_history.set_chat_index_snapshot(
            self,
            session_id=session_id,
            events=events,
            token_update=token_update,
            scan_bytes=scan_bytes,
            scan_complete=scan_complete,
            log_off=log_off,
        )

    def _append_chat_events(
        self,
        session_id: str,
        new_events: list[dict[str, Any]],
        *,
        new_off: int,
        latest_token: dict[str, Any] | None,
    ) -> None:
        _message_history.append_chat_events(
            self,
            session_id,
            new_events,
            new_off=new_off,
            latest_token=latest_token,
        )

    def _attach_notification_texts(
        self, events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return _message_history.attach_notification_texts(self, events)

    def _update_pi_last_chat_ts(
        self,
        session_id: str,
        events: list[dict[str, Any]],
        *,
        session_path: Path | None,
    ) -> None:
        _message_history.update_pi_last_chat_ts(
            self,
            session_id,
            events,
            session_path=session_path,
        )

    def _ensure_pi_chat_index(
        self, session_id: str, *, min_events: int, before: int
    ) -> tuple[list[dict[str, Any]], int, bool, int, dict[str, Any]]:
        return _message_history.ensure_pi_chat_index(
            self,
            session_id,
            min_events=min_events,
            before=before,
        )

    def _ensure_chat_index(
        self, session_id: str, *, min_events: int, before: int
    ) -> tuple[list[dict[str, Any]], int, bool, int, dict[str, Any] | None]:
        return _message_history.ensure_chat_index(
            self,
            session_id,
            min_events=min_events,
            before=before,
        )

    def mark_log_delta(
        self, session_id: str, *, objs: list[dict[str, Any]], new_off: int
    ) -> None:
        _message_history.mark_log_delta(
            self,
            session_id,
            objs=objs,
            new_off=new_off,
        )

    def idle_from_log(self, session_id: str) -> bool:
        return _message_history.idle_from_log(self, session_id)

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
        return _message_history.get_messages_page(
            self,
            session_id,
            offset=offset,
            init=init,
            limit=limit,
            before=before,
            view=view,
        )

    def _sock_call(
        self, sock_path: Path, req: dict[str, Any], timeout_s: float = 2.0
    ) -> dict[str, Any]:
        return _session_transport.sock_call(self, sock_path, req, timeout_s=timeout_s)

    def _kill_session_via_pids(self, s: Session) -> bool:
        return _session_transport.kill_session_via_pids(self, s)

    def kill_session(self, session_id: str) -> bool:
        return _session_transport.kill_session(self, session_id)

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
        return _session_control.service(self).spawn_web_session(
            cwd=cwd,
            args=args,
            agent_backend=agent_backend,
            resume_session_id=resume_session_id,
            worktree_branch=worktree_branch,
            model_provider=model_provider,
            preferred_auth_method=preferred_auth_method,
            model=model,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            create_in_tmux=create_in_tmux,
            backend=backend,
        )

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
        return _session_control.service(self).send(session_id, text)

    def enqueue(self, session_id: str, text: str) -> dict[str, Any]:
        return _session_control.service(self).enqueue(session_id, text)

    def queue_list(self, session_id: str) -> list[str]:
        return _session_control.service(self).queue_list(session_id)

    def queue_delete(self, session_id: str, index: int) -> dict[str, Any]:
        return _session_control.service(self).queue_delete(session_id, int(index))

    def queue_update(self, session_id: str, index: int, text: str) -> dict[str, Any]:
        return _session_control.service(self).queue_update(session_id, int(index), text)

    def get_state(self, session_id: str) -> dict[str, Any]:
        return _session_transport.get_state(self, session_id)

    def get_ui_state(self, session_id: str) -> dict[str, Any]:
        return _pi_ui_bridge.get_ui_state(RUNTIME, self, session_id)

    def get_session_commands(self, session_id: str) -> dict[str, Any]:
        return _pi_ui_bridge.get_session_commands(RUNTIME, self, session_id)

    def submit_ui_response(
        self, session_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return _pi_ui_bridge.submit_ui_response(RUNTIME, self, session_id, payload)

    def get_tail(self, session_id: str) -> str:
        return _session_transport.get_tail(self, session_id)

    def inject_keys(self, session_id: str, seq: str) -> dict[str, Any]:
        return _session_transport.inject_keys(self, session_id, seq)

    def mark_turn_complete(self, session_id: str, payload: dict[str, Any]) -> None:
        return


MANAGER = SessionManager()
RUNTIME = MANAGER._runtime


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
                if route_module.handle_get(RUNTIME, self, path, u):
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
                if route_module.handle_post(RUNTIME, self, path, u):
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
