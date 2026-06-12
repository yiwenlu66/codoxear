#!/usr/bin/env python3
from __future__ import annotations

import errno
import hashlib
import hmac
import http.server
import json
import math
import os
import re
import secrets
import signal
import shlex
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .agent_backend import get_agent_backend
from .agent_backend import normalize_agent_backend
from .backend_launch import apply_backend_environment as _apply_backend_environment
from .backend_launch import build_backend_args as _build_backend_args
from .backend_launch import build_backend_resume_args as _build_backend_resume_args
from .backend_launch import build_tmux_inline_env as _build_tmux_inline_env
from .backend_launch import tmux_unset_vars as _tmux_unset_vars
from .auth import CookieAuthSettings
from .auth import load_or_create_hmac_secret as _load_or_create_hmac_secret_impl
from .auth import parse_cookies as _parse_cookies_impl
from .auth import require_auth as _require_auth_impl
from .auth import set_auth_cookie as _set_auth_cookie_impl
from .auth import sign_cookie as _sign_cookie_impl
from .auth import verify_cookie as _verify_cookie_impl
from . import rollout_log as _rollout_log
from .file_response import send_attachment_file_response as _send_attachment_file_response
from .file_response import send_inline_file_response as _send_inline_file_response
from .file_response import single_byte_range as _single_byte_range
from .file_search import FILE_LIST_IGNORED_DIRS
from .file_search import FILE_SEARCH_LIMIT
from .file_search import file_search_score as _file_search_score
from .file_search import search_session_relative_files as _search_session_relative_files_impl
from .file_text import FILE_READ_MAX_BYTES
from .file_text import read_text_file_for_client as _read_text_file_for_client
from .file_text import read_text_file_for_write as _read_text_file_for_write
from .file_text import read_text_file_strict as _read_text_file_strict
from .file_text import write_new_text_file_atomic as _write_new_text_file_atomic
from .file_text import write_text_file_atomic as _write_text_file_atomic
from .file_types import file_kind as _file_kind
from .file_upload import attachment_inject_text as _attachment_inject_text
from .file_upload import safe_filename as _safe_filename
from .file_upload import stage_uploaded_file as _stage_uploaded_file_impl
from .launch_config import LaunchConfigPaths
from .launch_config import LaunchRequestValidationError
from .launch_config import NewSessionLaunchRequest
from .launch_config import SUPPORTED_CC_REASONING_EFFORTS
from .launch_config import SUPPORTED_PI_REASONING_EFFORTS
from .launch_config import SUPPORTED_REASONING_EFFORTS
from .launch_config import clean_reasoning_effort_list as _launch_clean_reasoning_effort_list
from .launch_config import configured_model_providers as _launch_configured_model_providers
from .launch_config import display_pi_reasoning_effort as _launch_display_pi_reasoning_effort
from .launch_config import display_reasoning_effort as _launch_display_reasoning_effort
from .launch_config import fallback_cc_launch_defaults as _launch_fallback_cc_launch_defaults
from .launch_config import fallback_codex_launch_defaults as _launch_fallback_codex_launch_defaults
from .launch_config import fallback_pi_launch_defaults as _launch_fallback_pi_launch_defaults
from .launch_config import launch_defaults_warning as _launch_defaults_warning_impl
from .launch_config import normalize_requested_cc_reasoning_effort as _launch_normalize_requested_cc_reasoning_effort
from .launch_config import normalize_requested_model as _launch_normalize_requested_model
from .launch_config import normalize_requested_model_provider as _launch_normalize_requested_model_provider
from .launch_config import normalize_requested_pi_reasoning_effort as _launch_normalize_requested_pi_reasoning_effort
from .launch_config import normalize_requested_preferred_auth_method as _launch_normalize_requested_preferred_auth_method
from .launch_config import normalize_requested_reasoning_effort as _launch_normalize_requested_reasoning_effort
from .launch_config import normalize_requested_service_tier as _launch_normalize_requested_service_tier
from .launch_config import parse_new_session_launch_request as _launch_parse_new_session_launch_request
from .launch_config import pi_allowed_reasoning_efforts_for_model as _launch_pi_allowed_reasoning_efforts_for_model
from .launch_config import pi_reasoning_effort_key as _launch_pi_reasoning_effort_key
from .launch_config import pi_reasoning_efforts_for_model_row as _launch_pi_reasoning_efforts_for_model_row
from .launch_config import provider_choice_for_settings as _launch_provider_choice_for_settings
from .launch_config import read_cc_launch_defaults as _launch_read_cc_launch_defaults
from .launch_config import read_codex_launch_defaults as _launch_read_codex_launch_defaults
from .launch_config import read_new_session_defaults as _launch_read_new_session_defaults
from .launch_config import read_pi_launch_defaults as _launch_read_pi_launch_defaults
from .launch_config import read_pi_reasoning_efforts_by_model as _launch_read_pi_reasoning_efforts_by_model
from .file_view import download_disposition as _download_disposition
from .file_view import inspect_client_path as _inspect_client_path
from .file_view import inspect_downloadable_file as _inspect_downloadable_file
from .file_view import inspect_openable_file as _inspect_openable_file
from .file_view import inspect_path_metadata as _inspect_path_metadata
from .file_view import read_client_file_view as _read_client_file_view
from .file_view import read_text_or_image as _read_text_or_image
from .video_preview import ensure_video_preview as _ensure_video_preview_impl
from .video_preview import video_preview_path as _video_preview_path_impl
from .video_preview import video_response_payload as _video_response_payload
from .cc_log import cc_user_text as _cc_user_text
from .cc_log import read_cc_run_settings as _read_cc_run_settings
from .message_cursor import MessageCursorError
from .message_cursor import attach_history_cursors as _attach_history_cursors_impl
from .message_cursor import decode_message_cursor as _decode_message_cursor_impl
from .message_cursor import encode_message_cursor as _encode_message_cursor_impl
from .message_cursor import sign_message_cursor as _sign_message_cursor_impl
from .message_cursor import verify_message_cursor as _verify_message_cursor_impl
from .pi_log import pi_user_text as _pi_user_text
from .pi_log import read_pi_run_settings as _read_pi_run_settings
from .queue_store import QueueStore
from .queue_store import coerce_queue_item as _queue_store_coerce_item
from .queue_store import copy_queue_item as _queue_store_copy_item
from .queue_store import new_queue_item as _queue_store_new_item
from .queue_store import new_queue_item_id as _queue_store_new_item_id
from .util import append_launch_attempt as _append_launch_attempt
from .util import atomic_write_json as _atomic_write_json
from .util import default_app_dir as _default_app_dir
from .util import classify_session_log as _classify_session_log
from .util import find_new_session_log as _find_new_session_log_impl
from .util import find_session_log_for_session_id as _find_session_log_for_session_id_impl
from .util import is_subagent_session_meta as _is_subagent_session_meta
from .util import iter_session_logs as _iter_session_logs_impl
from .util import launch_attempts_path as _launch_attempts_path
from .util import now as _now
from .util import pid_alive as _pid_alive
from .util import process_group_alive as _process_group_alive
from .util import proc_find_open_rollout_log as _proc_find_open_rollout_log
from .util import read_launch_attempts as _read_launch_attempts
from .util import load_json_file as _load_json_file
from .util import read_jsonl_from_offset as _read_jsonl_from_offset_impl
from .util import read_session_meta_payload as _read_session_meta_payload_impl
from .util import session_id_from_rollout_path as _session_id_from_rollout_path
from .util import subagent_parent_thread_id as _subagent_parent_thread_id
from .unattended import UnattendedStore
from .unattended import clean_unattended_cooldown_minutes as _clean_unattended_cooldown_minutes_impl
from .unattended import clean_unattended_remaining_injections as _clean_unattended_remaining_injections_impl
from .unattended import render_unattended_prompt as _render_unattended_prompt_impl
from .voice_push import VoicePushCoordinator


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
    session_id = parts[3]
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


APP_DIR = _default_app_dir()
PROC_ROOT = Path("/proc")
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_ASSET_VERSION_PLACEHOLDER = "__CODOXEAR_ASSET_VERSION__"
STATIC_ATTACH_MAX_BYTES_PLACEHOLDER = "__CODOXEAR_ATTACH_MAX_BYTES__"
STATIC_ASSET_VERSION_FILES = ("app.js", "app.css")
CONTENT_SECURITY_POLICY = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; worker-src 'self' blob:; font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
SOCK_DIR = APP_DIR / "socks"
STATE_PATH = APP_DIR / "state.json"
HMAC_SECRET_PATH = APP_DIR / "hmac_secret"
LAUNCH_ATTEMPTS_PATH = _launch_attempts_path(APP_DIR)
UPLOAD_DIR = APP_DIR / "uploads"
UNATTENDED_PATH = APP_DIR / "unattended.json"
ALIAS_PATH = APP_DIR / "session_aliases.json"
SIDEBAR_META_PATH = APP_DIR / "session_sidebar.json"
HIDDEN_SESSIONS_PATH = APP_DIR / "hidden_sessions.json"
FILE_HISTORY_PATH = APP_DIR / "session_files.json"
VIDEO_PREVIEW_DIR = APP_DIR / "video_previews"
QUEUE_PATH = APP_DIR / "session_queues.json"
PENDING_ATTACHMENTS_PATH = APP_DIR / "pending_attachments.json"
COMMIT_UNKNOWN_SENDS_PATH = APP_DIR / "commit_unknown_sends.json"
RECENT_CWD_PATH = APP_DIR / "recent_cwds.json"
VOICE_SETTINGS_PATH = APP_DIR / "voice_settings.json"
PUSH_SUBSCRIPTIONS_PATH = APP_DIR / "push_subscriptions.json"
DELIVERY_LEDGER_PATH = APP_DIR / "voice_delivery_ledger.json"
VAPID_PRIVATE_KEY_PATH = APP_DIR / "webpush_vapid_private.pem"

_DOTENV = (Path.cwd() / ".env").resolve()
if _DOTENV.exists():
    for _k, _v in _load_env_file(_DOTENV).items():
        os.environ.setdefault(_k, _v)

COOKIE_NAME = "codoxear_auth"
COOKIE_EXPIRES = "Fri, 31 Dec 9999 23:59:59 GMT"
COOKIE_SECURE = os.environ.get("CODEX_WEB_COOKIE_SECURE", "0") == "1"
URL_PREFIX = _normalize_url_prefix(os.environ.get("CODEX_WEB_URL_PREFIX"))
COOKIE_PATH = (URL_PREFIX + "/") if URL_PREFIX else "/"
TMUX_SESSION_NAME = (os.environ.get("CODEX_WEB_TMUX_SESSION") or "codoxear").strip() or "codoxear"
TMUX_META_WAIT_SECONDS = 3.0

_CODEX_HOME_ENV = os.environ.get("CODEX_HOME")
if _CODEX_HOME_ENV is None or (not _CODEX_HOME_ENV.strip()):
    CODEX_HOME = Path.home() / ".codex"
else:
    CODEX_HOME = Path(_CODEX_HOME_ENV)
CODEX_SESSIONS_DIR = CODEX_HOME / "sessions"
CODEX_CONFIG_PATH = CODEX_HOME / "config.toml"
MODELS_CACHE_PATH = CODEX_HOME / "models_cache.json"
PI_HOME = get_agent_backend("pi").home()
PI_SESSIONS_DIR = get_agent_backend("pi").sessions_dir()
PI_SETTINGS_PATH = PI_HOME / "agent" / "settings.json"
PI_MODELS_PATH = PI_HOME / "agent" / "models.json"
PI_AUTH_PATH = PI_HOME / "agent" / "auth.json"
CC_HOME = get_agent_backend("cc").home()
CC_SESSIONS_DIR = get_agent_backend("cc").sessions_dir()
CC_SETTINGS_PATH = CC_HOME / "settings.json"
DEFAULT_AGENT_BACKEND = normalize_agent_backend(os.environ.get("CODEX_WEB_DEFAULT_AGENT_BACKEND"), default="codex")
DEFAULT_HOST = os.environ.get("CODEX_WEB_HOST", "::")
DEFAULT_PORT = int(os.environ.get("CODEX_WEB_PORT", "8743"))
UNATTENDED_DEFAULT_IDLE_MINUTES = 5
UNATTENDED_DEFAULT_MAX_INJECTIONS = 10
UNATTENDED_SWEEP_SECONDS = float(os.environ.get("CODEX_WEB_UNATTENDED_SWEEP_SECONDS", "2.5"))
QUEUE_SWEEP_SECONDS = float(os.environ.get("CODEX_WEB_QUEUE_SWEEP_SECONDS", "1.0"))
VOICE_PUSH_SWEEP_SECONDS = float(os.environ.get("CODEX_WEB_VOICE_PUSH_SWEEP_SECONDS", "1.0"))
QUEUE_IDLE_GRACE_SECONDS = float(os.environ.get("CODEX_WEB_QUEUE_IDLE_GRACE_SECONDS", "10.0"))
UNATTENDED_MAX_SCAN_BYTES = int(os.environ.get("CODEX_WEB_UNATTENDED_MAX_SCAN_BYTES", str(8 * 1024 * 1024)))
DISCOVER_MIN_INTERVAL_SECONDS = float(os.environ.get("CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS", "1.0"))
METRICS_WINDOW = int(os.environ.get("CODEX_WEB_METRICS_WINDOW", "256"))
FILE_HISTORY_MAX = int(os.environ.get("CODEX_WEB_FILE_HISTORY_MAX", "20"))
GIT_DIFF_MAX_BYTES = int(os.environ.get("CODEX_WEB_GIT_DIFF_MAX_BYTES", str(800 * 1024)))
GIT_DIFF_TIMEOUT_SECONDS = float(os.environ.get("CODEX_WEB_GIT_DIFF_TIMEOUT_SECONDS", "4.0"))
GIT_WORKTREE_TIMEOUT_SECONDS = float(os.environ.get("CODEX_WEB_GIT_WORKTREE_TIMEOUT_SECONDS", "10.0"))
GIT_CHANGED_FILES_MAX = int(os.environ.get("CODEX_WEB_GIT_CHANGED_FILES_MAX", "400"))
ATTACH_UPLOAD_MAX_BYTES = int(os.environ.get("CODEX_WEB_ATTACH_MAX_BYTES", str(16 * 1024 * 1024)))
ATTACH_UPLOAD_BODY_MAX_BYTES = int(
    os.environ.get(
        "CODEX_WEB_ATTACH_BODY_MAX_BYTES",
        str((4 * ((ATTACH_UPLOAD_MAX_BYTES + 2) // 3)) + (64 * 1024)),
    )
)
SEND_COMMIT_TIMEOUT_SECONDS = float(os.environ.get("CODEX_WEB_SEND_COMMIT_TIMEOUT_SECONDS", "30"))
SIDEBAR_PRIORITY_HALF_LIFE_SECONDS = 8.0 * 3600.0
SIDEBAR_PRIORITY_LAMBDA = math.log(2.0) / SIDEBAR_PRIORITY_HALF_LIFE_SECONDS
RECENT_CWD_MAX = int(os.environ.get("CODEX_WEB_RECENT_CWD_MAX", "256"))
STATIC_CACHE_ENABLED = str(os.environ.get("CODEX_WEB_STATIC_CACHE") or "").strip() == "1"
TRANSCRIPT_EXPORT_MAX_BYTES = int(os.environ.get("CODEX_WEB_TRANSCRIPT_EXPORT_MAX_BYTES", str(50 * 1024 * 1024)))


def _static_cache_control_headers(*, enabled: bool = STATIC_CACHE_ENABLED) -> dict[str, str]:
    if enabled:
        return {"Cache-Control": "public, max-age=31536000, immutable"}
    return {
        "Cache-Control": "no-store",
        "Pragma": "no-cache",
        "Expires": "0",
    }


UNATTENDED_PROMPT_PREFIX = """Unattended-mode instructions (optimize for 8+ hours, minimal turns, minimal repetition, maximal progress)

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


def _render_unattended_prompt(request: str | None) -> str:
    return _render_unattended_prompt_impl(request, prompt_prefix=UNATTENDED_PROMPT_PREFIX)


def _clean_unattended_cooldown_minutes(raw: Any) -> int:
    return _clean_unattended_cooldown_minutes_impl(raw, default_idle_minutes=UNATTENDED_DEFAULT_IDLE_MINUTES)


def _clean_unattended_remaining_injections(raw: Any, *, allow_zero: bool) -> int:
    return _clean_unattended_remaining_injections_impl(
        raw,
        default_max_injections=UNATTENDED_DEFAULT_MAX_INJECTIONS,
        allow_zero=allow_zero,
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


def _tmux_available() -> bool:
    return shutil.which("tmux") is not None


def _wait_for_spawned_broker_meta(spawn_nonce: str, *, timeout_s: float = TMUX_META_WAIT_SECONDS) -> dict[str, Any]:
    deadline = time.time() + max(timeout_s, 0.0)
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
            return meta
        time.sleep(0.05)
    raise RuntimeError(f"tmux launch did not publish broker metadata within {timeout_s:.1f}s")


def _tmux_pane_snapshot(tmux_bin: str, *, pane_id: str | None = None, window: str | None = None) -> dict[str, Any]:
    target = _clean_optional_text(pane_id)
    if target is None and _clean_optional_text(window) is not None:
        target = f"{TMUX_SESSION_NAME}:{window}"
    if target is None:
        return {}
    fmt = "#{pane_id}\t#{pane_pid}\t#{pane_dead}\t#{pane_dead_status}\t#{pane_current_command}\t#{window_name}"
    proc = subprocess.run(
        [tmux_bin, "display-message", "-p", "-t", target, fmt],
        capture_output=True,
        text=True,
        check=False,
    )
    out: dict[str, Any] = {"tmux_target": target}
    if proc.returncode != 0:
        out["tmux_inspect_error"] = (proc.stderr or proc.stdout or f"exit status {proc.returncode}").strip()
        return out
    parts = (proc.stdout or "").strip().split("\t")
    keys = ("tmux_pane_id", "tmux_pane_pid", "tmux_pane_dead", "tmux_pane_dead_status", "tmux_pane_command", "tmux_window")
    for key, value in zip(keys, parts):
        out[key] = value
    cap = subprocess.run(
        [tmux_bin, "capture-pane", "-p", "-t", target, "-S", "-80"],
        capture_output=True,
        text=True,
        check=False,
    )
    if cap.returncode == 0:
        out["tmux_pane_tail"] = (cap.stdout or "")[-4000:]
    else:
        out["tmux_capture_error"] = (cap.stderr or cap.stdout or f"exit status {cap.returncode}").strip()
    return out


def _record_launch_attempt(record: dict[str, Any]) -> dict[str, Any]:
    rec = _append_launch_attempt(record, path=LAUNCH_ATTEMPTS_PATH)
    if rec.get("state") == "failed":
        sys.stderr.write(
            "error: session launch failed: "
            f"{rec.get('launch_id')}: {rec.get('stage')}: {rec.get('error')}\n"
        )
        sys.stderr.flush()
    return rec


def _launch_attempt_id(record: dict[str, Any]) -> str:
    raw = _clean_optional_text(record.get("launch_id"))
    if raw is None:
        updated_ts = record.get("updated_ts", record.get("created_ts", 0))
        try:
            millis = int(float(updated_ts) * 1000)
        except (TypeError, ValueError):
            millis = 0
        raw = f"launch-{millis}"
    return raw


def _latest_launch_attempt(launch_id: str) -> dict[str, Any] | None:
    needle = str(launch_id or "").strip()
    if not needle:
        return None
    for rec in _read_launch_attempts(path=LAUNCH_ATTEMPTS_PATH, max_records=100, max_age_s=24 * 3600):
        if _launch_attempt_id(rec) == needle:
            return rec
    return None


def _submitted_user_messages(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(record, dict):
        return []
    raw = record.get("submitted_user_messages")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        ts = item.get("ts")
        ts_out = float(ts) if isinstance(ts, (int, float)) else float(record.get("updated_ts", time.time()) or time.time())
        source = _clean_optional_text(item.get("source")) or "send"
        out.append({"text": text, "ts": ts_out, "source": source})
    return out


def _launch_failure_tail(record: dict[str, Any]) -> str:
    for key in ("pty_tail", "tmux_pane_tail"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val[-4000:]
    return ""


def _launch_attempt_transcript_payload(record: dict[str, Any]) -> dict[str, Any]:
    launch_id = _launch_attempt_id(record)
    ts = record.get("updated_ts", record.get("created_ts", time.time()))
    ts_f = float(ts) if isinstance(ts, (int, float)) else time.time()
    events: list[dict[str, Any]] = []
    for msg in _submitted_user_messages(record):
        events.append({"role": "user", "text": msg["text"], "ts": msg["ts"]})
    if record.get("state") == "failed":
        stage = _clean_optional_text(record.get("stage"))
        err = _clean_optional_text(record.get("error")) or "session launch failed"
        lines = ["Session launch failed before a transcript log was created."]
        if stage:
            lines.append(f"Stage: {stage}")
        lines.append(f"Error: {err}")
        agent_status = record.get("agent_exit_status", record.get("exit_code"))
        broker_status = record.get("broker_exit_status")
        if isinstance(agent_status, int):
            lines.append(f"Agent exit status: {agent_status}")
        if isinstance(broker_status, int):
            lines.append(f"Broker exit status: {broker_status}")
        tail = _launch_failure_tail(record)
        if tail:
            lines.extend(["", "Pre-log terminal tail:", tail])
        events.append({"role": "assistant", "text": "\n".join(lines), "ts": ts_f, "message_class": "error"})
    return {
        "transcript_state": "failed",
        "thread_id": launch_id or None,
        "log_path": None,
        "live_cursor": None,
        "history_cursor": None,
        "events": events,
        "has_older": False,
        "busy": False,
        "queue_len": 0,
        "token": None,
    }


def _launch_attempt_transcript_for_session_id(session_id: str) -> dict[str, Any] | None:
    rec = _latest_launch_attempt(session_id)
    if rec is None or rec.get("state") != "failed":
        return None
    row = _launch_attempt_row(rec)
    if row is None or row.get("session_id") != session_id:
        return None
    return _launch_attempt_transcript_payload(rec)


def _launch_attempt_row(record: dict[str, Any]) -> dict[str, Any] | None:
    launch_id = _launch_attempt_id(record)
    state = _clean_optional_text(record.get("state")) or "starting"
    if state in {"live", "log_bound", "broker_spawned", "broker_meta_bound"}:
        return None
    cwd = _clean_optional_text(record.get("cwd")) or "?"
    start_ts_raw = record.get("created_ts", record.get("start_ts", record.get("updated_ts", time.time())))
    updated_ts_raw = record.get("updated_ts", start_ts_raw)
    try:
        start_ts = float(start_ts_raw)
    except (TypeError, ValueError):
        start_ts = time.time()
    try:
        updated_ts = float(updated_ts_raw)
    except (TypeError, ValueError):
        updated_ts = start_ts
    backend = normalize_agent_backend(record.get("agent_backend"), default=DEFAULT_AGENT_BACKEND)
    provider = _clean_optional_text(record.get("model_provider"))
    preferred_auth = _clean_optional_text(record.get("preferred_auth_method"))
    failed = state == "failed"
    return {
        "session_id": launch_id,
        "thread_id": launch_id,
        "pid": None,
        "broker_pid": record.get("broker_pid") if isinstance(record.get("broker_pid"), int) else None,
        "agent_backend": backend,
        "owned": True,
        "transport": _clean_optional_text(record.get("transport")),
        "cwd": cwd,
        "start_ts": start_ts,
        "updated_ts": updated_ts,
        "log_path": None,
        "state_busy": False,
        "queue_len": 0,
        "token": None,
        "thinking": 0,
        "tools": 0,
        "system": 0,
        "unattended_enabled": False,
        "unattended_cooldown_minutes": UNATTENDED_DEFAULT_IDLE_MINUTES,
        "unattended_remaining_injections": UNATTENDED_DEFAULT_MAX_INJECTIONS,
        "alias": "",
        "files": [],
        "git_branch": "",
        "model_provider": provider,
        "preferred_auth_method": preferred_auth,
        "provider_choice": _provider_choice_for_settings(model_provider=provider, preferred_auth_method=preferred_auth),
        "model": _clean_optional_text(record.get("model")),
        "reasoning_effort": _clean_optional_text(record.get("reasoning_effort")),
        "service_tier": _clean_optional_text(record.get("service_tier")),
        "tmux_session": _clean_optional_text(record.get("tmux_session")),
        "tmux_window": _clean_optional_text(record.get("tmux_window")),
        "priority_offset": 0.0,
        "snooze_until": None,
        "dependency_session_id": None,
        "time_priority": 1.0,
        "base_priority": 1.0,
        "final_priority": 1.0,
        "blocked": False,
        "snoozed": False,
        "busy": False,
        "spawn_nonce": _clean_optional_text(record.get("spawn_nonce")),
        "launch_id": launch_id,
        "launch_state": state,
        "launch_error": _clean_optional_text(record.get("error")) or ("session launch failed" if failed else ""),
        "launch_stage": _clean_optional_text(record.get("stage")),
        "submitted_user_message_count": len(_submitted_user_messages(record)),
    }


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


def _extract_token_update(objs: list[dict[str, Any]]) -> dict[str, Any] | None:
    return _rollout_log._extract_token_update(objs)


def _video_preview_path(path: Path) -> Path:
    return _video_preview_path_impl(path, preview_dir=VIDEO_PREVIEW_DIR)


def _ensure_video_preview(path: Path) -> Path:
    return _ensure_video_preview_impl(path, preview_dir=VIDEO_PREVIEW_DIR)


_CLIENT_DISCONNECT_ERRNOS = {errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED}
_CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class BadRequestError(Exception):
    """Client request body or shape was invalid."""


class RequestPayloadTooLargeError(Exception):
    """Client request body exceeded the configured size limit."""


def _is_client_disconnect(exc: BaseException) -> bool:
    if isinstance(exc, _CLIENT_DISCONNECT_ERRORS):
        return True
    return isinstance(exc, OSError) and getattr(exc, "errno", None) in _CLIENT_DISCONNECT_ERRNOS


def _handle_route_exception(handler: http.server.BaseHTTPRequestHandler, exc: BaseException) -> None:
    if _is_client_disconnect(exc):
        return
    if isinstance(exc, BadRequestError):
        _json_response(handler, 400, {"error": str(exc)})
        return
    if isinstance(exc, RequestPayloadTooLargeError):
        _json_response(handler, 413, {"error": str(exc)})
        return
    traceback.print_exc()
    payload = {"error": str(exc)}
    if os.environ.get("CODEX_WEB_DEBUG_ERRORS") == "1":
        payload["trace"] = traceback.format_exc()
    _json_response(handler, 500, payload)


def _json_response(handler: http.server.BaseHTTPRequestHandler, status: int, obj: Any) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    if getattr(handler, "_codoxear_refresh_auth_cookie", False):
        _set_auth_cookie(handler)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _if_none_match_contains(header_value: str | None, etag: str) -> bool:
    if header_value is None:
        return False
    values = [part.strip() for part in str(header_value).split(",")]
    return "*" in values or etag in values


def _json_response_with_etag(handler: http.server.BaseHTTPRequestHandler, obj: Any) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    etag = '"' + _sha256_hex(body) + '"'
    if _if_none_match_contains(handler.headers.get("If-None-Match"), etag):
        handler.send_response(304)
        if getattr(handler, "_codoxear_refresh_auth_cookie", False):
            _set_auth_cookie(handler)
        handler.send_header("ETag", etag)
        handler.send_header("Cache-Control", "private, no-cache")
        handler.send_header("Content-Length", "0")
        handler.end_headers()
        return
    handler.send_response(200)
    if getattr(handler, "_codoxear_refresh_auth_cookie", False):
        _set_auth_cookie(handler)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("ETag", etag)
    handler.send_header("Cache-Control", "private, no-cache")
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: http.server.BaseHTTPRequestHandler, limit: int = 2 * 1024 * 1024) -> bytes:
    cl = handler.headers.get("Content-Length")
    if cl is None:
        cl = "0"
    cl2 = str(cl).strip()
    if not cl2:
        cl2 = "0"
    try:
        n = int(cl2)
    except (TypeError, ValueError) as e:
        raise BadRequestError("invalid content-length") from e
    if n < 0:
        raise BadRequestError(f"invalid content-length: {n}")
    if n > limit:
        raise RequestPayloadTooLargeError(f"request body too large (max {limit} bytes)")
    return handler.rfile.read(n)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_or_create_hmac_secret() -> bytes:
    return _load_or_create_hmac_secret_impl(app_dir=APP_DIR, secret_path=HMAC_SECRET_PATH)


HMAC_SECRET = _load_or_create_hmac_secret()


def _cookie_auth_settings() -> CookieAuthSettings:
    return CookieAuthSettings(
        cookie_name=COOKIE_NAME,
        cookie_path=COOKIE_PATH,
        cookie_expires=COOKIE_EXPIRES,
        cookie_secure=COOKIE_SECURE,
        secret=HMAC_SECRET,
    )


def _sign_cookie(payload: dict[str, Any]) -> str:
    return _sign_cookie_impl(payload, secret=HMAC_SECRET)


def _verify_cookie(value: str) -> dict[str, Any] | None:
    return _verify_cookie_impl(value, secret=HMAC_SECRET)


class SessionLaunchError(RuntimeError):
    def __init__(self, record: dict[str, Any]):
        msg = str(record.get("error") or record.get("message") or "session launch failed")
        super().__init__(msg)
        self.record = record


class SessionNotReadyError(RuntimeError):
    pass


class SessionInjectionError(RuntimeError):
    pass


class SessionCommitUnknownError(RuntimeError):
    pass


class ControlSocketCallError(RuntimeError):
    def __init__(self, message: str, *, request_sent: bool):
        super().__init__(message)
        self.request_sent = bool(request_sent)


def _sign_message_cursor(payload: dict[str, Any]) -> str:
    return _sign_message_cursor_impl(payload, secret=HMAC_SECRET)


def _verify_message_cursor(token: str) -> dict[str, Any]:
    return _verify_message_cursor_impl(token, secret=HMAC_SECRET)


def _encode_message_cursor(*, kind: str, session: "Session", pos: int) -> str:
    return _encode_message_cursor_impl(kind=kind, session=session, pos=pos, secret=HMAC_SECRET)


def _decode_message_cursor(token: str, *, kind: str, session: "Session") -> int:
    return _decode_message_cursor_impl(token, kind=kind, session=session, secret=HMAC_SECRET)


def _attach_history_cursors(events: list[dict[str, Any]], *, session: "Session") -> list[dict[str, Any]]:
    return _attach_history_cursors_impl(events, session=session, encode_cursor=_encode_message_cursor)


def _parse_cookies(header: str | None) -> dict[str, str]:
    return _parse_cookies_impl(header)


def _require_auth(handler: http.server.BaseHTTPRequestHandler) -> bool:
    return _require_auth_impl(handler, settings=_cookie_auth_settings(), verify=_verify_cookie)


def _set_auth_cookie(handler: http.server.BaseHTTPRequestHandler) -> None:
    _set_auth_cookie_impl(handler, settings=_cookie_auth_settings())

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


def _require_existing_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError("file not found")
    if not path.is_file():
        raise ValueError("path is not a file")
    return path


def _resolve_existing_session_file(base: Path, raw_path: str) -> Path:
    return _require_existing_file(_resolve_session_path(base, raw_path))


def _resolve_existing_absolute_file(raw_path: str) -> Path:
    return _require_existing_file(Path(raw_path).expanduser().resolve())


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


def _list_session_relative_files(base: Path) -> list[str]:
    root = base.expanduser()
    if not root.is_absolute():
        root = root.resolve()
    if not root.exists():
        raise FileNotFoundError("session cwd not found")
    if not root.is_dir():
        raise ValueError("session cwd is not a directory")
    out: list[str] = []

    def _onerror(err: OSError) -> None:
        raise err

    for current_root, dirnames, filenames in os.walk(root, topdown=True, onerror=_onerror, followlinks=False):
        dirnames[:] = [name for name in sorted(dirnames) if name not in FILE_LIST_IGNORED_DIRS]
        current_path = Path(current_root)
        for name in sorted(filenames):
            rel = (current_path / name).relative_to(root)
            out.append(rel.as_posix())
    out.sort()
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
    _run_git(cwd, ["rev-parse", "--is-inside-work-tree"], timeout_s=GIT_DIFF_TIMEOUT_SECONDS, max_bytes=4096)


def _git_repo_root(cwd: Path) -> Path | None:
    try:
        root = _run_git(cwd, ["rev-parse", "--show-toplevel"], timeout_s=GIT_DIFF_TIMEOUT_SECONDS, max_bytes=64 * 1024).strip()
    except (RuntimeError, FileNotFoundError):
        return None
    if not root:
        return None
    return Path(root).resolve()


def _search_session_relative_files(base: Path, *, query: str, limit: int = FILE_SEARCH_LIMIT) -> dict[str, Any]:
    return _search_session_relative_files_impl(base, query=query, limit=limit, git_root_func=_git_repo_root)


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


def _stage_uploaded_file(session_id: str, filename: str, raw: bytes, *, max_bytes: int = ATTACH_UPLOAD_MAX_BYTES) -> Path:
    return _stage_uploaded_file_impl(
        session_id,
        filename,
        raw,
        upload_dir=UPLOAD_DIR,
        now_fn=_now,
        max_bytes=max_bytes,
    )


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


def _launch_config_paths() -> LaunchConfigPaths:
    return LaunchConfigPaths(
        codex_config_path=CODEX_CONFIG_PATH,
        models_cache_path=MODELS_CACHE_PATH,
        pi_settings_path=PI_SETTINGS_PATH,
        pi_models_path=PI_MODELS_PATH,
        pi_auth_path=PI_AUTH_PATH,
        cc_settings_path=CC_SETTINGS_PATH,
    )


def _normalize_requested_model(value: Any) -> str | None:
    return _launch_normalize_requested_model(value)


def _display_reasoning_effort(value: Any) -> str | None:
    return _launch_display_reasoning_effort(value)


def _display_pi_reasoning_effort(value: Any) -> str | None:
    return _launch_display_pi_reasoning_effort(value)


def _normalize_requested_reasoning_effort(value: Any) -> str | None:
    return _launch_normalize_requested_reasoning_effort(value)


def _clean_reasoning_effort_list(raw: Any, *, supported: tuple[str, ...]) -> list[str] | None:
    return _launch_clean_reasoning_effort_list(raw, supported=supported)


def _pi_reasoning_efforts_for_model_row(row: dict[str, Any]) -> list[str] | None:
    return _launch_pi_reasoning_efforts_for_model_row(row)


def _pi_reasoning_effort_key(provider: str | None, model: str | None) -> str | None:
    return _launch_pi_reasoning_effort_key(provider, model)


def _read_pi_reasoning_efforts_by_model() -> dict[str, list[str]]:
    return _launch_read_pi_reasoning_efforts_by_model(_launch_config_paths())


def _pi_allowed_reasoning_efforts_for_model(
    *,
    model_provider: str | None,
    model: str | None,
    reasoning_efforts_by_model: Mapping[str, list[str]] | None = None,
) -> list[str] | None:
    return _launch_pi_allowed_reasoning_efforts_for_model(
        model_provider=model_provider,
        model=model,
        reasoning_efforts_by_model=reasoning_efforts_by_model,
        paths=_launch_config_paths(),
    )


def _normalize_requested_pi_reasoning_effort(
    value: Any,
    *,
    model_provider: str | None = None,
    model: str | None = None,
    reasoning_efforts_by_model: Mapping[str, list[str]] | None = None,
) -> str | None:
    return _launch_normalize_requested_pi_reasoning_effort(
        value,
        model_provider=model_provider,
        model=model,
        reasoning_efforts_by_model=reasoning_efforts_by_model,
        paths=_launch_config_paths(),
    )


def _normalize_requested_cc_reasoning_effort(value: Any) -> str | None:
    return _launch_normalize_requested_cc_reasoning_effort(value)


def _normalize_requested_model_provider(value: Any, *, allowed: set[str] | None = None) -> str | None:
    return _launch_normalize_requested_model_provider(value, allowed=allowed)


def _normalize_requested_service_tier(value: Any) -> str | None:
    return _launch_normalize_requested_service_tier(value)


def _normalize_requested_preferred_auth_method(value: Any) -> str | None:
    return _launch_normalize_requested_preferred_auth_method(value)


def _configured_model_providers(data: dict[str, Any]) -> list[str]:
    return _launch_configured_model_providers(data)


def _provider_choice_for_settings(*, model_provider: str | None, preferred_auth_method: str | None) -> str:
    return _launch_provider_choice_for_settings(model_provider=model_provider, preferred_auth_method=preferred_auth_method)

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


def _sessions_dir_for_backend(agent_backend: str) -> Path:
    backend_name = normalize_agent_backend(agent_backend)
    if backend_name == "codex":
        return CODEX_SESSIONS_DIR
    if backend_name == "pi":
        return PI_SESSIONS_DIR
    if backend_name == "cc":
        return CC_SESSIONS_DIR
    raise ValueError(f"unsupported agent_backend: {backend_name}")


def _iter_session_logs(*, agent_backend: str = "codex") -> list[Path]:
    backend_name = normalize_agent_backend(agent_backend)
    return _iter_session_logs_impl(_sessions_dir_for_backend(backend_name), agent_backend=backend_name)


def _find_session_log_for_session_id(session_id: str, *, agent_backend: str = "codex") -> Path | None:
    backend_name = normalize_agent_backend(agent_backend)
    return _find_session_log_for_session_id_impl(_sessions_dir_for_backend(backend_name), session_id, agent_backend=backend_name)


def _find_new_session_log(
    *,
    agent_backend: str = "codex",
    after_ts: float,
    preexisting: set[Path],
    timeout_s: float = 15.0,
) -> tuple[str, Path] | None:
    backend_name = normalize_agent_backend(agent_backend)
    sessions_dir = _sessions_dir_for_backend(backend_name)
    return _find_new_session_log_impl(
        sessions_dir=sessions_dir,
        agent_backend=backend_name,
        after_ts=after_ts,
        preexisting=preexisting,
        timeout_s=timeout_s,
    )


def _read_jsonl_from_offset(path: Path, offset: int, max_bytes: int = 2 * 1024 * 1024) -> tuple[list[dict[str, Any]], int]:
    return _read_jsonl_from_offset_impl(path, offset, max_bytes=max_bytes)


def _discover_log_for_session_id(session_id: str, *, agent_backend: str = "codex") -> Path | None:
    return _find_session_log_for_session_id(session_id, agent_backend=agent_backend)


def _read_session_meta(log_path: Path, *, agent_backend: str | None = None) -> dict[str, Any]:
    if agent_backend is None:
        try:
            log_path.resolve().relative_to(PI_SESSIONS_DIR.resolve())
            inferred = "pi"
        except Exception:
            try:
                log_path.resolve().relative_to(CC_SESSIONS_DIR.resolve())
                inferred = "cc"
            except Exception:
                inferred = "codex"
        backend_name = inferred
    else:
        backend_name = normalize_agent_backend(agent_backend)
    payload = _read_session_meta_payload_impl(log_path, agent_backend=backend_name, timeout_s=0.0)
    if payload is None:
        raise ValueError(f"missing session metadata in {log_path}")
    return payload


_INVALID_SESSION_META_WARNINGS: set[tuple[str, str]] = set()


def _read_session_meta_or_none(log_path: Path, *, agent_backend: str | None = None, context: str) -> dict[str, Any] | None:
    try:
        return _read_session_meta(log_path, agent_backend=agent_backend)
    except (FileNotFoundError, ValueError) as e:
        warning_key = (context, str(log_path))
        if warning_key not in _INVALID_SESSION_META_WARNINGS:
            _INVALID_SESSION_META_WARNINGS.add(warning_key)
            sys.stderr.write(f"warning: {context}: ignoring invalid session metadata in {log_path}: {type(e).__name__}: {e}\n")
            sys.stderr.flush()
        return None


def _turn_context_run_settings(payload: Any) -> tuple[str | None, str | None]:
    if not isinstance(payload, dict):
        return None, None
    return (
        _clean_optional_text(payload.get("model")),
        _display_reasoning_effort(payload.get("reasoning_effort") or payload.get("effort")),
    )


def _read_run_settings_from_log(log_path: Path, *, agent_backend: str = "codex") -> tuple[str | None, str | None, str | None]:
    backend_name = normalize_agent_backend(agent_backend)
    if backend_name == "pi":
        return _read_pi_run_settings(log_path)
    if backend_name == "cc":
        return _read_cc_run_settings(log_path)
    meta = _read_session_meta_or_none(log_path, agent_backend="codex", context="run settings")
    model_provider = _clean_optional_text(meta.get("model_provider")) if meta is not None else None
    model = _clean_optional_text(meta.get("model")) if meta is not None else None
    reasoning_effort = _display_reasoning_effort(meta.get("reasoning_effort")) if meta is not None else None
    if model is None or reasoning_effort is None:
        ctx_model, ctx_effort = _turn_context_run_settings(_rollout_log._find_latest_turn_context(log_path, max_scan_bytes=8 * 1024 * 1024))
        if model is None:
            model = ctx_model
        if reasoning_effort is None:
            reasoning_effort = ctx_effort
    return model_provider, model, reasoning_effort


def _new_queue_item_id() -> str:
    return _queue_store_new_item_id()


def _new_queue_item(text: str, *, created_ts: float | None = None) -> dict[str, Any]:
    return _queue_store_new_item(text, created_ts=created_ts)


def _copy_queue_item(item: dict[str, Any]) -> dict[str, Any]:
    return _queue_store_copy_item(item)


def _coerce_queue_item(raw: Any) -> dict[str, Any] | None:
    return _queue_store_coerce_item(raw)

def _fallback_codex_launch_defaults() -> dict[str, Any]:
    return _launch_fallback_codex_launch_defaults()


def _fallback_pi_launch_defaults() -> dict[str, Any]:
    return _launch_fallback_pi_launch_defaults()


def _fallback_cc_launch_defaults() -> dict[str, Any]:
    return _launch_fallback_cc_launch_defaults()


def _read_codex_launch_defaults() -> dict[str, Any]:
    return _launch_read_codex_launch_defaults(_launch_config_paths())


def _read_pi_launch_defaults() -> dict[str, Any]:
    return _launch_read_pi_launch_defaults(_launch_config_paths())


def _read_cc_launch_defaults() -> dict[str, Any]:
    return _launch_read_cc_launch_defaults(_launch_config_paths())


def _launch_defaults_warning(exc: BaseException) -> str:
    return _launch_defaults_warning_impl(exc)


def _read_new_session_defaults() -> dict[str, Any]:
    return _launch_read_new_session_defaults(_launch_config_paths(), default_agent_backend=DEFAULT_AGENT_BACKEND)


def _codex_launch_defaults_for_request() -> dict[str, Any]:
    try:
        return _read_codex_launch_defaults()
    except Exception:
        return _fallback_codex_launch_defaults()


def _pi_launch_defaults_for_request() -> dict[str, Any]:
    try:
        return _read_pi_launch_defaults()
    except Exception:
        return _fallback_pi_launch_defaults()


def _parse_new_session_launch_request(obj: dict[str, Any]) -> NewSessionLaunchRequest:
    return _launch_parse_new_session_launch_request(
        obj,
        default_agent_backend=DEFAULT_AGENT_BACKEND,
        codex_launch_defaults_provider=_codex_launch_defaults_for_request,
        pi_launch_defaults_provider=_pi_launch_defaults_for_request,
    )

def _resume_candidate_from_log(log_path: Path, *, agent_backend: str = "codex") -> dict[str, Any] | None:
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
    try:
        stat = log_path.stat()
        updated_ts = float(stat.st_mtime)
    except FileNotFoundError:
        return None
    except Exception:
        updated_ts = 0.0
    git_branch = ""
    if backend_name in {"codex", "cc"}:
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


def _list_resume_candidates_for_cwd(cwd: str, *, agent_backend: str = "codex", limit: int = 12) -> list[dict[str, Any]]:
    backend_name = normalize_agent_backend(agent_backend)
    cwd2 = str(Path(cwd).expanduser().resolve())
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for log_path in _iter_session_logs(agent_backend=backend_name):
        try:
            row = _resume_candidate_from_log(log_path, agent_backend=backend_name)
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
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") == "message":
                    text = _pi_user_text(obj) or ""
                elif obj.get("type") == "user":
                    text = _cc_user_text(obj) or ""
                elif obj.get("type") == "response_item":
                    payload = obj.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    if payload.get("type") != "message" or payload.get("role") != "user":
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


def _coerce_main_thread_log(*, thread_id: str, log_path: Path) -> tuple[str, Path]:
    sm = _read_session_meta_or_none(log_path, agent_backend="codex", context="main-thread coercion")
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


def _extract_positioned_chat_events(records: list[Any]) -> list[dict[str, Any]]:
    return _rollout_log._extract_positioned_chat_events(records)


def _extract_delivery_messages(objs: list[dict[str, Any]]) -> list[Any]:
    return _rollout_log._extract_delivery_messages(objs)


def _read_jsonl_records_from_offset(
    path: Path,
    offset: int,
    *,
    max_bytes: int = 2 * 1024 * 1024,
) -> tuple[list[Any], int]:
    return _rollout_log._read_jsonl_records_from_offset(path, offset, max_bytes=max_bytes)


def _read_chat_tail_page(log_path: Path, *, limit: int) -> tuple[list[dict[str, Any]], int, int, bool]:
    return _rollout_log._read_chat_tail_page(log_path, limit=limit)


def _read_chat_history_page(log_path: Path, *, before_byte: int, limit: int) -> tuple[list[dict[str, Any]], int, bool]:
    return _rollout_log._read_chat_history_page(log_path, before_byte=before_byte, limit=limit)


def _read_chat_export_events(log_path: Path, *, max_bytes: int = TRANSCRIPT_EXPORT_MAX_BYTES) -> list[dict[str, Any]]:
    size = int(log_path.stat().st_size)
    limit = max(1, int(max_bytes))
    if size > limit:
        raise ValueError(f"transcript log is too large to export ({size} bytes > {limit} bytes)")
    records, _next_after = _read_jsonl_records_from_offset(log_path, 0, max_bytes=max(size, 1))
    return _extract_positioned_chat_events(records)


def _parse_bounded_query_int(
    qs: Mapping[str, list[str]],
    name: str,
    *,
    default: int,
    min_value: int,
    max_value: int,
) -> tuple[int, str | None]:
    values = qs.get(name)
    if not values:
        return default, None
    try:
        value = int(values[0])
    except (TypeError, ValueError):
        return default, f"{name} must be an integer"
    return max(min_value, min(max_value, value)), None


def _search_chat_events(events: list[dict[str, Any]], query: str, *, limit: int = 20) -> tuple[int, list[dict[str, Any]]]:
    needle = query.strip().casefold()
    if not needle:
        return 0, []
    max_matches = max(0, int(limit))
    count = 0
    matches: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        role = event.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = event.get("text")
        if not isinstance(text, str) or needle not in text.casefold():
            continue
        count += 1
        if len(matches) < max_matches:
            matches.append(event)
    return count, matches


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
    agent_backend: str
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
    last_chat_history_scanned: bool = False
    meta_thinking: int = 0
    meta_tools: int = 0
    meta_system: int = 0
    meta_log_off: int = 0
    delivery_log_off: int = 0
    idle_cache_log_off: int = -1
    idle_cache_value: bool | None = None
    queue_idle_since: float | None = None
    queue_sending_item_id: str | None = None
    model_provider: str | None = None
    preferred_auth_method: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None
    transport: str | None = None
    tmux_session: str | None = None
    tmux_window: str | None = None
    launch_id: str | None = None
    spawn_nonce: str | None = None
    resume_session_id: str | None = None
    pending_attachment: bool = False
    commit_unknown_send: dict[str, Any] | None = None
    sync_send_supported: bool = False
    key_write_errors_supported: bool = False


def _message_transcript_identity(session: Session) -> dict[str, Any]:
    log_path = session.log_path
    if log_path is None or (not log_path.exists()):
        return {
            "transcript_state": "pending_bind",
            "thread_id": None,
            "log_path": None,
        }
    return {
        "transcript_state": "bound",
        "thread_id": session.thread_id,
        "log_path": str(log_path),
    }


def _metadata_ignored_rollout_paths(meta: dict[str, Any], *, sock: Path) -> set[Path]:
    raw = meta.get("ignored_rollout_paths")
    if raw is None:
        return set()
    if not isinstance(raw, list):
        raise ValueError(f"invalid ignored_rollout_paths in metadata for socket {sock}")
    out: set[Path] = set()
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"invalid ignored_rollout_paths entry in metadata for socket {sock}")
        out.add(Path(item))
    return out


def _metadata_sync_send_supported(meta: dict[str, Any]) -> bool:
    caps = meta.get("control_capabilities")
    return meta.get("control_protocol_version") == 2 and isinstance(caps, dict) and caps.get("sync_send") is True


def _metadata_key_write_errors_supported(meta: dict[str, Any]) -> bool:
    caps = meta.get("control_capabilities")
    return meta.get("control_protocol_version") == 2 and isinstance(caps, dict) and caps.get("key_write_errors") is True


def _metadata_detaches_current_log(meta: dict[str, Any], current_log_path: Path | None) -> bool:
    if current_log_path is None:
        return False
    return _clean_optional_text(meta.get("session_id")) is None and meta.get("log_path") is None


def _broker_tail_has_session_detach_marker(agent_backend: str, tail: Any) -> bool:
    if agent_backend != "codex" or not isinstance(tail, str):
        return False
    return "To continue this session, run " in tail


class SessionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._stop = threading.Event()
        self._last_discover_ts = 0.0
        self._unattended: dict[str, dict[str, Any]] = {}
        self._unattended_store = UnattendedStore(
            path=UNATTENDED_PATH,
            default_idle_minutes=UNATTENDED_DEFAULT_IDLE_MINUTES,
            default_max_injections=UNATTENDED_DEFAULT_MAX_INJECTIONS,
        )
        self._aliases: dict[str, str] = {}
        self._sidebar_meta: dict[str, dict[str, Any]] = {}
        self._hidden_sessions: set[str] = set()
        self._files: dict[str, list[str]] = {}
        self._queues: dict[str, list[dict[str, Any]]] = {}
        self._queue_store = QueueStore(QUEUE_PATH)
        self._pending_attachment_ids: set[str] = set()
        self._commit_unknown_sends: dict[str, dict[str, Any]] = {}
        self._input_locks: dict[str, threading.Lock] = {}
        self._recent_cwds: dict[str, float] = {}
        self._include_launch_attempts = True
        self._unattended_last_injected: dict[str, float] = {}
        self._unattended_last_injected_scope: dict[str, float] = {}
        self._load_unattended()
        self._load_aliases()
        self._load_sidebar_meta()
        self._load_hidden_sessions()
        self._load_files()
        self._load_queues()
        self._load_pending_attachments()
        self._load_commit_unknown_sends()
        self._load_recent_cwds()
        self._backfill_recent_cwds_from_logs()
        self._voice_push = VoicePushCoordinator(
            app_dir=APP_DIR,
            stop_event=self._stop,
            settings_path=VOICE_SETTINGS_PATH,
            subscriptions_path=PUSH_SUBSCRIPTIONS_PATH,
            delivery_ledger_path=DELIVERY_LEDGER_PATH,
            vapid_private_key_path=VAPID_PRIVATE_KEY_PATH,
        )
        self._discover_existing(force=True)
        self._prune_missing_commit_unknown_sends()
        self._unattended_thr = threading.Thread(target=self._unattended_loop, name="unattended", daemon=True)
        self._unattended_thr.start()
        self._queue_thr = threading.Thread(target=self._queue_loop, name="queue", daemon=True)
        self._queue_thr.start()
        self._voice_push_scan_thr = threading.Thread(target=self._voice_push_scan_loop, name="voice-push-scan", daemon=True)
        self._voice_push_scan_thr.start()

    def stop(self) -> None:
        self._stop.set()

    def _reset_log_caches(self, s: Session, *, meta_log_off: int) -> None:
        s.meta_thinking = 0
        s.meta_tools = 0
        s.meta_system = 0
        s.last_chat_ts = None
        s.last_chat_history_scanned = False
        s.meta_log_off = int(meta_log_off)
        s.delivery_log_off = int(meta_log_off)
        s.idle_cache_log_off = -1
        s.idle_cache_value = None
        s.queue_idle_since = None
        s.queue_sending_item_id = None
        s.model_provider = None
        s.preferred_auth_method = None
        s.model = None
        s.reasoning_effort = None
        s.service_tier = None

    def _session_run_settings(self, *, meta: dict[str, Any], log_path: Path | None, agent_backend: str) -> tuple[str | None, str | None, str | None, str | None]:
        backend_name = normalize_agent_backend(agent_backend)
        model_provider = _clean_optional_text(meta.get("model_provider"))
        preferred_auth_method = _normalize_requested_preferred_auth_method(meta.get("preferred_auth_method"))
        model = _clean_optional_text(meta.get("model"))
        if backend_name == "codex":
            reasoning_effort = _display_reasoning_effort(meta.get("reasoning_effort"))
        elif backend_name == "pi":
            reasoning_effort = _display_pi_reasoning_effort(meta.get("reasoning_effort"))
        else:
            reasoning_effort = _normalize_requested_cc_reasoning_effort(meta.get("reasoning_effort"))
        if log_path is not None and log_path.exists():
            log_provider, log_model, log_effort = _read_run_settings_from_log(log_path, agent_backend=backend_name)
            if log_provider is not None:
                model_provider = log_provider
            if log_model is not None:
                model = log_model
            if log_effort is not None:
                reasoning_effort = log_effort
        return model_provider, preferred_auth_method, model, reasoning_effort

    def _session_transport(self, *, meta: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
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
            self._discover_existing(force=force)
        except TypeError:
            self._discover_existing()

    def _load_unattended(self) -> None:
        cleaned = self._unattended_store.load()
        with self._lock:
            self._unattended = cleaned

    def _save_unattended(self) -> None:
        with self._lock:
            obj = dict(self._unattended)
        self._unattended_store.save(obj)

    def _load_aliases(self) -> None:
        obj = _load_json_file(ALIAS_PATH, default=None)
        if obj is None:
            return
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
        _atomic_write_json(ALIAS_PATH, obj)

    def _load_sidebar_meta(self) -> None:
        obj = _load_json_file(SIDEBAR_META_PATH, default=None)
        if obj is None:
            return
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
        _atomic_write_json(SIDEBAR_META_PATH, obj)

    def _load_hidden_sessions(self) -> None:
        obj = _load_json_file(HIDDEN_SESSIONS_PATH, default=None)
        if obj is None:
            return
        if not isinstance(obj, list):
            raise ValueError("invalid hidden_sessions.json (expected list)")
        cleaned = {sid.strip() for sid in obj if isinstance(sid, str) and sid.strip()}
        with self._lock:
            self._hidden_sessions = cleaned

    def _save_hidden_sessions(self) -> None:
        with self._lock:
            obj = sorted(getattr(self, "_hidden_sessions", set()))
        _atomic_write_json(HIDDEN_SESSIONS_PATH, obj, sort_keys=True)

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

    def _prune_stale_socket_without_metadata(self, session_id: str, sock: Path) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
        self._unhide_session(session_id)
        self._clear_deleted_session_state(session_id)
        _unlink_quiet(sock)
        _unlink_quiet(sock.with_suffix(".json"))

    def _clear_deleted_session_state(self, session_id: str) -> None:
        changed_sidebar = False
        changed_unattended = False
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
            unattended = getattr(self, "_unattended", None)
            if isinstance(unattended, dict) and session_id in unattended:
                unattended.pop(session_id, None)
                changed_unattended = True
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
            input_locks = getattr(self, "_input_locks", None)
            if isinstance(input_locks, dict):
                input_locks.pop(session_id, None)
            pending_attachment_ids = getattr(self, "_pending_attachment_ids", None)
            if isinstance(pending_attachment_ids, set):
                pending_attachment_ids.discard(session_id)
            unknown_sends = getattr(self, "_commit_unknown_sends", None)
            if isinstance(unknown_sends, dict):
                unknown_sends.pop(session_id, None)
        self._save_pending_attachments()
        self._save_commit_unknown_sends()
        self._save_aliases()
        if changed_sidebar:
            self._save_sidebar_meta()
        if changed_unattended:
            self._save_unattended()
        if changed_files:
            self._save_files()
        if changed_queues:
            self._save_queues()

    def _load_files(self) -> None:
        obj = _load_json_file(FILE_HISTORY_PATH, default=None)
        if obj is None:
            return
        if not isinstance(obj, dict):
            raise ValueError("invalid session_files.json (expected object)")
        cleaned: dict[str, list[str]] = {}
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

    def _save_files(self) -> None:
        with self._lock:
            obj = dict(self._files)
        _atomic_write_json(FILE_HISTORY_PATH, obj)

    def _queue_store_for_manager(self) -> QueueStore:
        store = getattr(self, "_queue_store", None)
        if not isinstance(store, QueueStore):
            store = QueueStore(QUEUE_PATH)
            self._queue_store = store
        return store

    def _input_lock_for_session(self, session_id: str) -> threading.Lock:
        with self._lock:
            locks = getattr(self, "_input_locks", None)
            if not isinstance(locks, dict):
                self._input_locks = {}
                locks = self._input_locks
            lock = locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                locks[session_id] = lock
            return lock

    def _load_queues(self) -> None:
        cleaned = self._queue_store_for_manager().load()
        with self._lock:
            self._queues = cleaned

    def _save_queues(self) -> None:
        with self._lock:
            obj = dict(self._queues)
        self._queue_store_for_manager().save(obj)

    def _load_pending_attachments(self) -> None:
        obj = _load_json_file(PENDING_ATTACHMENTS_PATH, default=None)
        if obj is None:
            return
        if not isinstance(obj, list):
            raise ValueError("invalid pending_attachments.json (expected array)")
        cleaned = {str(item).strip() for item in obj if isinstance(item, str) and str(item).strip()}
        with self._lock:
            self._pending_attachment_ids = cleaned

    def _save_pending_attachments(self) -> None:
        with self._lock:
            ids = sorted(str(item) for item in getattr(self, "_pending_attachment_ids", set()) if str(item).strip())
        _atomic_write_json(PENDING_ATTACHMENTS_PATH, ids)

    def _set_pending_attachment(self, session_id: str, value: bool) -> None:
        with self._lock:
            ids = getattr(self, "_pending_attachment_ids", None)
            if not isinstance(ids, set):
                self._pending_attachment_ids = set()
                ids = self._pending_attachment_ids
            s = self._sessions.get(session_id)
            if s:
                s.pending_attachment = bool(value)
            if value:
                ids.add(session_id)
            else:
                ids.discard(session_id)
        self._save_pending_attachments()

    def clear_pending_attachment(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("unknown session")
        self._set_pending_attachment(session_id, False)
        return {"ok": True, "pending_attachment": False}

    def _clean_commit_unknown_send_record(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        ts_raw = raw.get("created_ts")
        try:
            created_ts = float(ts_raw) if ts_raw is not None else time.time()
        except (TypeError, ValueError):
            created_ts = time.time()
        if not math.isfinite(created_ts) or created_ts <= 0:
            created_ts = time.time()
        error = raw.get("error")
        record: dict[str, Any] = {"text": text, "created_ts": created_ts}
        if isinstance(error, str) and error.strip():
            record["error"] = error.strip()
        return record

    def _load_commit_unknown_sends(self) -> None:
        obj = _load_json_file(COMMIT_UNKNOWN_SENDS_PATH, default=None)
        if obj is None:
            return
        if not isinstance(obj, dict):
            raise ValueError("invalid commit_unknown_sends.json (expected object)")
        cleaned: dict[str, dict[str, Any]] = {}
        for sid, raw in obj.items():
            if not isinstance(sid, str) or not sid.strip():
                continue
            rec = self._clean_commit_unknown_send_record(raw)
            if rec is not None:
                cleaned[sid.strip()] = rec
        with self._lock:
            self._commit_unknown_sends = cleaned

    def _save_commit_unknown_sends(self) -> None:
        with self._lock:
            source = getattr(self, "_commit_unknown_sends", {})
            obj = {str(sid): dict(rec) for sid, rec in source.items() if str(sid).strip() and isinstance(rec, dict)}
        _atomic_write_json(COMMIT_UNKNOWN_SENDS_PATH, obj)

    def _set_commit_unknown_send(self, session_id: str, record: dict[str, Any] | None) -> None:
        cleaned = self._clean_commit_unknown_send_record(record) if record is not None else None
        with self._lock:
            unknown_sends = getattr(self, "_commit_unknown_sends", None)
            if not isinstance(unknown_sends, dict):
                self._commit_unknown_sends = {}
                unknown_sends = self._commit_unknown_sends
            s = self._sessions.get(session_id)
            if cleaned is None:
                unknown_sends.pop(session_id, None)
                if s:
                    s.commit_unknown_send = None
            else:
                unknown_sends[session_id] = dict(cleaned)
                if s:
                    s.commit_unknown_send = dict(cleaned)
        self._save_commit_unknown_sends()

    def clear_commit_unknown_send(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("unknown session")
        self._set_commit_unknown_send(session_id, None)
        return {"ok": True, "commit_unknown_send": False}

    def _prune_missing_commit_unknown_sends(self) -> bool:
        changed = False
        with self._lock:
            unknown_sends = getattr(self, "_commit_unknown_sends", None)
            if not isinstance(unknown_sends, dict):
                self._commit_unknown_sends = {}
                return False
            active_ids = set(getattr(self, "_sessions", {}).keys())
            for sid in list(unknown_sends.keys()):
                if sid not in active_ids:
                    unknown_sends.pop(sid, None)
                    changed = True
        if changed:
            self._save_commit_unknown_sends()
        return changed

    def _load_recent_cwds(self) -> None:
        obj = _load_json_file(RECENT_CWD_PATH, default=None)
        if obj is None:
            return
        if not isinstance(obj, dict):
            raise ValueError("invalid recent_cwds.json (expected object)")
        cleaned: dict[str, float] = {}
        for raw_cwd, raw_ts in obj.items():
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
        with self._lock:
            items = sorted(getattr(self, "_recent_cwds", {}).items(), key=lambda item: (-float(item[1]), item[0]))[:RECENT_CWD_MAX]
        obj = {cwd: ts for cwd, ts in items}
        _atomic_write_json(RECENT_CWD_PATH, obj)

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
                keep = dict(sorted(recent.items(), key=lambda item: (-float(item[1]), item[0]))[:RECENT_CWD_MAX])
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
        with self._lock:
            items = sorted(getattr(self, "_recent_cwds", {}).items(), key=lambda item: (-float(item[1]), item[0]))
        return [cwd for cwd, _ts in items[: max(0, int(limit))]]

    def _queue_len(self, session_id: str) -> int:
        with self._lock:
            qmap = getattr(self, "_queues", None)
            if not isinstance(qmap, dict):
                return 0
            return self._queue_store_for_manager().queue_len(qmap, session_id)

    def _queue_list_local(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            qmap = getattr(self, "_queues", None)
            if not isinstance(qmap, dict):
                return []
            s = self._sessions.get(session_id)
            sending_id = s.queue_sending_item_id if s else None
            return self._queue_store_for_manager().list_items(qmap, session_id, sending_item_id=sending_id)

    def _queue_append_item_local(self, session_id: str, text: str) -> tuple[dict[str, Any], int]:
        t = str(text)
        if not t.strip():
            raise ValueError("text required")
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("unknown session")
            item, ql = self._queue_store_for_manager().append(self._queues, session_id, text)
        self._save_queues()
        return item, int(ql)

    def _queue_enqueue_local(self, session_id: str, text: str) -> dict[str, Any]:
        item, ql = self._queue_append_item_local(session_id, text)
        return {"queued": True, "queue_len": int(ql), "item": item}

    def _queue_delete_local(self, session_id: str, item_id: str, *, allow_commit_unknown: bool = False) -> dict[str, Any]:
        item_id_clean = str(item_id).strip()
        if not item_id_clean:
            raise ValueError("id required")
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("unknown session")
            s = self._sessions.get(session_id)
            sending_id = s.queue_sending_item_id if s else None
            ql = self._queue_store_for_manager().delete(
                self._queues,
                session_id,
                item_id_clean,
                sending_item_id=sending_id,
                allow_commit_unknown=allow_commit_unknown,
            )
        self._save_queues()
        return {"ok": True, "queue_len": int(ql)}

    def _queue_update_local(self, session_id: str, item_id: str, text: str) -> dict[str, Any]:
        item_id_clean = str(item_id).strip()
        t = str(text)
        if not item_id_clean:
            raise ValueError("id required")
        if not t.strip():
            raise ValueError("text required")
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("unknown session")
            s = self._sessions.get(session_id)
            sending_id = s.queue_sending_item_id if s else None
            item, ql = self._queue_store_for_manager().update(self._queues, session_id, item_id_clean, t, sending_item_id=sending_id)
        self._save_queues()
        return {"ok": True, "queue_len": int(ql), "item": item}

    def _queue_move_local(self, session_id: str, item_id: str, to_index: int) -> dict[str, Any]:
        item_id_clean = str(item_id).strip()
        if not item_id_clean:
            raise ValueError("id required")
        if isinstance(to_index, bool):
            raise ValueError("to_index must be an integer")
        target = int(to_index)
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("unknown session")
            s = self._sessions.get(session_id)
            sending_id = s.queue_sending_item_id if s else None
            ql = self._queue_store_for_manager().move(self._queues, session_id, item_id_clean, target, sending_item_id=sending_id)
        self._save_queues()
        return {"ok": True, "queue_len": int(ql)}

    def _queue_session_state(self, session_id: str) -> tuple[Session, Path | None]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            return s, s.log_path

    def _send_remote_ready(self, session_id: str, *, allow_pending_attachment: bool = False) -> bool:
        self._refresh_session_meta_if_sidecar_exists(session_id, drain_queue=False)
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            if s.commit_unknown_send:
                return False
            if s.pending_attachment and not allow_pending_attachment:
                return False
            log_path = s.log_path
        st = self.get_state(session_id)
        if not isinstance(st, dict) or "busy" not in st or "queue_len" not in st:
            raise ValueError("invalid broker state response")
        if bool(st.get("busy")) or int(st.get("queue_len")) > 0:
            return False
        self._refresh_session_meta_if_sidecar_exists(session_id, drain_queue=False)
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            log_path = s.log_path
        if isinstance(log_path, Path) and log_path.exists() and (not self.idle_from_log(session_id)):
            return False
        return True

    def _queue_remote_ready(self, session_id: str, *, log_path: Path | None) -> bool:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            if s.commit_unknown_send:
                return False
            if s.pending_attachment:
                return False
        st = self.get_state(session_id)
        if not isinstance(st, dict) or "busy" not in st or "queue_len" not in st:
            raise ValueError("invalid broker state response")
        if bool(st.get("busy")) or int(st.get("queue_len")) > 0:
            return False
        if isinstance(log_path, Path) and log_path.exists() and (not self.idle_from_log(session_id)):
            return False
        return True

    def _promote_queue_head_if_sendable(
        self,
        session_id: str,
        *,
        require_idle_grace: bool,
        now_ts: float | None = None,
        expected_item_id: str | None = None,
    ) -> dict[str, Any] | None:
        if now_ts is None:
            now_ts = time.time()
        _session, log_path = self._queue_session_state(session_id)
        try:
            ready = self._queue_remote_ready(session_id, log_path=log_path)
        except Exception:
            with self._lock:
                s0 = self._sessions.get(session_id)
                if s0:
                    s0.queue_idle_since = None
            return None
        with self._lock:
            s0 = self._sessions.get(session_id)
            if not s0:
                return None
            q = self._queues.get(session_id)
            if not isinstance(q, list) or not q:
                s0.queue_idle_since = None
                return None
            if s0.queue_sending_item_id is not None:
                return None
            head = q[0]
            if bool(head.get("commit_unknown")):
                s0.queue_idle_since = None
                return None
            head_id = str(head.get("id") or "")
            if expected_item_id is not None and head_id != expected_item_id:
                return None
            if not ready:
                s0.queue_idle_since = None
                return None
            if require_idle_grace:
                idle_since = s0.queue_idle_since
                if idle_since is None:
                    s0.queue_idle_since = float(now_ts)
                    return None
                if (float(now_ts) - idle_since) < QUEUE_IDLE_GRACE_SECONDS:
                    return None
            s0.queue_idle_since = None
            s0.queue_sending_item_id = head_id
            head["commit_unknown"] = True
            head["commit_unknown_ts"] = time.time()
            text = str(head.get("text") or "")
        self._save_queues()
        try:
            resp = self.send(session_id, text, queue_item_id=head_id)
        except (SessionNotReadyError, SessionInjectionError):
            with self._lock:
                s0 = self._sessions.get(session_id)
                if s0 and s0.queue_sending_item_id == head_id:
                    s0.queue_sending_item_id = None
                    s0.queue_idle_since = None
                q = self._queues.get(session_id)
                if isinstance(q, list):
                    for item in q:
                        if str(item.get("id") or "") == head_id:
                            item.pop("commit_unknown", None)
                            item.pop("commit_unknown_ts", None)
                            break
            self._save_queues()
            return None
        except SessionCommitUnknownError:
            unknown_item: dict[str, Any] | None = None
            queue_len = 0
            with self._lock:
                s0 = self._sessions.get(session_id)
                if s0 and s0.queue_sending_item_id == head_id:
                    s0.queue_sending_item_id = None
                    s0.queue_idle_since = None
                q = self._queues.get(session_id)
                if isinstance(q, list):
                    queue_len = len(q)
                    for item in q:
                        if str(item.get("id") or "") == head_id:
                            item["commit_unknown"] = True
                            item["commit_unknown_ts"] = time.time()
                            unknown_item = dict(item)
                            break
            self._save_queues()
            return {"queued": True, "queue_len": int(queue_len), "item": unknown_item, "commit_unknown": True}
        except Exception:
            with self._lock:
                s0 = self._sessions.get(session_id)
                if s0 and s0.queue_sending_item_id == head_id:
                    s0.queue_sending_item_id = None
                    s0.queue_idle_since = None
                q = self._queues.get(session_id)
                if isinstance(q, list):
                    for item in q:
                        if str(item.get("id") or "") == head_id:
                            item.pop("commit_unknown", None)
                            item.pop("commit_unknown_ts", None)
                            break
            self._save_queues()
            return None
        with self._lock:
            s0 = self._sessions.get(session_id)
            if s0 and s0.queue_sending_item_id == head_id:
                s0.queue_sending_item_id = None
                s0.queue_idle_since = None
            self._queue_store_for_manager().pop_sent(self._queues, session_id, head_id)
        self._save_queues()
        return resp

    def _files_key_for_session(self, session_id: str) -> tuple[str, list[str], "Session"]:
        s = self._sessions.get(session_id)
        if not s:
            raise KeyError("unknown session")
        sid_key = f"sid:{session_id}"
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
            keys_to_clear = list(legacy_keys)
            cwd = str(getattr(s, "cwd", "") or "").strip()
            if cwd:
                # `cwd:` buckets are legacy pre-session-scoping state. Do not
                # migrate them into active sessions because they leak history
                # across sessions with the same cwd, but do discard the matching
                # legacy bucket when the owning session/cwd is deleted.
                keys_to_clear.append(f"cwd:{cwd}")
            for lk in keys_to_clear:
                if lk in self._files:
                    self._files.pop(lk, None)
                    dirty = True
            if key in self._files:
                self._files.pop(key, None)
                dirty = True
        if dirty:
            self._save_files()

    def unattended_get(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            cfg0 = self._unattended.get(session_id)
            cfg = dict(cfg0) if isinstance(cfg0, dict) else {}
        enabled = bool(cfg.get("enabled"))
        request = cfg.get("request")
        if not isinstance(request, str):
            request = ""
        cooldown_minutes = _clean_unattended_cooldown_minutes(cfg.get("cooldown_minutes"))
        remaining_injections = _clean_unattended_remaining_injections(cfg.get("remaining_injections"), allow_zero=True)
        return {
            "enabled": enabled,
            "request": request,
            "cooldown_minutes": cooldown_minutes,
            "remaining_injections": remaining_injections,
        }

    def unattended_set(
        self,
        session_id: str,
        *,
        enabled: bool | None = None,
        request: str | None = None,
        cooldown_minutes: int | None = None,
        remaining_injections: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            cur0 = self._unattended.get(session_id)
            cur = dict(cur0) if isinstance(cur0, dict) else {}
            if enabled is not None:
                cur["enabled"] = bool(enabled)
            if request is not None:
                cur["request"] = str(request)
            if cooldown_minutes is not None:
                cur["cooldown_minutes"] = _clean_unattended_cooldown_minutes(cooldown_minutes)
            if remaining_injections is not None:
                cur["remaining_injections"] = _clean_unattended_remaining_injections(remaining_injections, allow_zero=True)
            cur["cooldown_minutes"] = _clean_unattended_cooldown_minutes(cur.get("cooldown_minutes"))
            cur["remaining_injections"] = _clean_unattended_remaining_injections(cur.get("remaining_injections"), allow_zero=True)
            self._unattended[session_id] = cur
            if enabled is not None and bool(enabled) is False:
                self._unattended_last_injected.pop(session_id, None)
        self._save_unattended()
        return self.unattended_get(session_id)

    def _session_display_name(self, session_id: str) -> str:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return "Session"
            alias = self._aliases.get(session_id)
            if isinstance(alias, str) and alias.strip():
                return alias.strip()
            cwd_name = Path(s.cwd).expanduser().name.strip()
            return cwd_name or "Session"

    def _observe_rollout_delta(self, session_id: str, *, objs: list[dict[str, Any]], new_off: int) -> None:
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
        voice_push.observe_messages(session_id=session_id, session_display_name=session_name, messages=messages)
        with self._lock:
            s = self._sessions.get(session_id)
            if s is not None:
                s.delivery_log_off = max(int(s.delivery_log_off), int(new_off))

    def _voice_push_scan_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._voice_push_scan_sweep()
            except Exception as e:
                sys.stderr.write(f"error: voice-push scan failed: {type(e).__name__}: {e}\n")
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
                objs, new_off = _read_jsonl_from_offset(log_path, off, max_bytes=256 * 1024)
                if new_off <= off:
                    break
                self._observe_rollout_delta(sid, objs=objs, new_off=new_off)
                off = new_off
                loops += 1

    def _unattended_loop(self) -> None:
        # Persist across browser disconnects: server is the scheduler.
        while not self._stop.is_set():
            try:
                self._unattended_sweep()
            except Exception as e:
                sys.stderr.write(f"error: unattended sweep failed: {type(e).__name__}: {e}\n")
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
            self._stop.wait(UNATTENDED_SWEEP_SECONDS)

    def _unattended_sweep(self) -> None:
        now = time.time()
        # Keep discovery fresh; sessions can appear/disappear without UI polling.
        self._discover_existing_if_stale()
        self._prune_dead_sessions()
        with self._lock:
            items: list[tuple[str, Session, dict[str, Any], float]] = []
            for sid, s in self._sessions.items():
                cfg0 = self._unattended.get(sid)
                cfg = dict(cfg0) if isinstance(cfg0, dict) else {}
                last_inj = float(self._unattended_last_injected.get(sid, 0.0))
                items.append((sid, s, cfg, last_inj))

        for sid, s, cfg, last_inj in items:
            if not bool(cfg.get("enabled")):
                continue
            try:
                cooldown_minutes = _clean_unattended_cooldown_minutes(cfg.get("cooldown_minutes"))
                cooldown_seconds = float(cooldown_minutes * 60)
                remaining_injections = _clean_unattended_remaining_injections(cfg.get("remaining_injections"), allow_zero=True)
                if remaining_injections <= 0:
                    with self._lock:
                        cur0 = self._unattended.get(sid)
                        cur = dict(cur0) if isinstance(cur0, dict) else {}
                        cur["enabled"] = False
                        cur["remaining_injections"] = 0
                        self._unattended[sid] = cur
                        self._unattended_last_injected.pop(sid, None)
                    self._save_unattended()
                    continue
                request = cfg.get("request")
                if not isinstance(request, str):
                    request = ""
                prompt = _render_unattended_prompt(request)
                lp = s.log_path
                if lp is None or (not lp.exists()):
                    continue
                scope_key = f"thread:{s.thread_id}" if s.thread_id else f"log:{str(lp)}"
                with self._lock:
                    scope_last = float(self._unattended_last_injected_scope.get(scope_key, 0.0))
                if (last_inj and (now - last_inj) < cooldown_seconds) or (scope_last and (now - scope_last) < cooldown_seconds):
                    continue
                st = self.get_state(sid)
                if not isinstance(st, dict):
                    raise ValueError("invalid broker state response")
                if "busy" not in st or "queue_len" not in st:
                    raise ValueError("invalid broker state response")
                busy = bool(st.get("busy"))
                ql = int(st.get("queue_len"))
                if busy or ql > 0 or self._queue_len(sid) > 0:
                    continue
                last = _last_chat_role_ts_from_tail(lp, max_scan_bytes=UNATTENDED_MAX_SCAN_BYTES)
                if not last:
                    continue
                role, ts = last
                if role != "assistant":
                    continue
                if (now - float(ts)) < cooldown_seconds:
                    continue
                with self._lock:
                    scope_last = float(self._unattended_last_injected_scope.get(scope_key, 0.0))
                if scope_last and (now - scope_last) < cooldown_seconds:
                    continue
                self.send(sid, prompt)
                with self._lock:
                    self._unattended_last_injected[sid] = now
                    self._unattended_last_injected_scope[scope_key] = now
                    cur0 = self._unattended.get(sid)
                    cur = dict(cur0) if isinstance(cur0, dict) else {}
                    next_remaining = max(0, remaining_injections - 1)
                    cur["remaining_injections"] = next_remaining
                    if next_remaining <= 0:
                        cur["enabled"] = False
                        self._unattended_last_injected.pop(sid, None)
                    self._unattended[sid] = cur
                self._save_unattended()
            except Exception as e:
                sys.stderr.write(f"error: unattended session {sid} skipped: {type(e).__name__}: {e}\n")
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()

    def _queue_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._queue_sweep()
            except Exception:
                sys.stderr.write("error: queue sweep crashed; continuing\n")
                sys.stderr.flush()
            self._stop.wait(QUEUE_SWEEP_SECONDS)

    def _maybe_drain_session_queue(self, session_id: str, *, now_ts: float | None = None) -> bool:
        resp = self._promote_queue_head_if_sendable(session_id, require_idle_grace=True, now_ts=now_ts)
        return isinstance(resp, dict)

    def _queue_sweep(self) -> None:
        self._discover_existing_if_stale()
        self._prune_dead_sessions()
        with self._lock:
            # Drop queues for sessions that no longer exist.
            dropped = self._queue_store_for_manager().drop_missing_sessions(self._queues, self._sessions.keys())
            session_ids = self._queue_store_for_manager().nonempty_session_ids(self._queues)
        if dropped:
            self._save_queues()
        for sid in session_ids:
            if self._maybe_drain_session_queue(sid):
                break

    def _discover_existing(self, *, force: bool = False) -> None:
        if not force:
            now = time.time()
            with self._lock:
                last = float(self._last_discover_ts)
            if (now - last) < DISCOVER_MIN_INTERVAL_SECONDS:
                return
        SOCK_DIR.mkdir(parents=True, exist_ok=True)
        recent_cwd_dirty = False
        for sock in sorted(SOCK_DIR.glob("*.sock")):
            session_id = sock.stem
            # Prefer metadata file written by sessiond.
            meta_path = sock.with_suffix(".json")
            if not meta_path.exists():
                self._prune_stale_socket_without_metadata(session_id, sock)
                continue
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
            agent_backend = normalize_agent_backend(meta.get("agent_backend"), default="codex")
            owned = (meta.get("owner") == "web") if isinstance(meta.get("owner"), str) else False
            transport, tmux_session, tmux_window = self._session_transport(meta=meta)
            sync_send_supported = _metadata_sync_send_supported(meta)
            key_write_errors_supported = _metadata_key_write_errors_supported(meta)
            launch_id = _clean_optional_text(meta.get("launch_id"))
            spawn_nonce = _clean_optional_text(meta.get("spawn_nonce"))
            cwd_raw = meta.get("cwd")
            if not isinstance(cwd_raw, str) or (not cwd_raw.strip()):
                raise ValueError(f"invalid cwd in metadata for socket {sock}")
            cwd = cwd_raw

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
            if log_path is not None and not log_path.exists():
                log_path = None
            ignored_rollout_paths = _metadata_ignored_rollout_paths(meta, sock=sock)
            if log_path is None and agent_backend in {"codex", "cc"} and _pid_alive(codex_pid):
                discovered_log_path = _proc_find_open_rollout_log(
                    proc_root=PROC_ROOT,
                    root_pid=codex_pid,
                    agent_backend=agent_backend,
                    cwd=cwd,
                    ignored_paths=ignored_rollout_paths,
                )
                if discovered_log_path is not None and discovered_log_path.exists():
                    log_path = discovered_log_path
            if log_path is not None and agent_backend == "codex":
                session_meta = _read_session_meta_or_none(log_path, agent_backend="codex", context="session discovery")
                meta_session_id = session_meta.get("id") if session_meta else None
                if isinstance(meta_session_id, str) and meta_session_id:
                    thread_id = meta_session_id
                thread_id, log_path = _coerce_main_thread_log(thread_id=thread_id, log_path=log_path)

            if (log_path is None) and (not _pid_alive(codex_pid)) and (not _pid_alive(broker_pid)):
                if owned:
                    try:
                        _record_launch_attempt(
                            {
                                "launch_id": launch_id,
                                "state": "failed",
                                "stage": "broker_exit_before_log_bind",
                                "error": "broker exited before publishing a session log",
                                "agent_backend": agent_backend,
                                "cwd": meta.get("cwd"),
                                "created_ts": meta.get("start_ts"),
                                "broker_pid": broker_pid,
                                "agent_pid": codex_pid,
                                "transport": transport,
                                "tmux_session": tmux_session,
                                "tmux_window": tmux_window,
                                "spawn_nonce": spawn_nonce,
                                "model_provider": meta.get("model_provider"),
                                "preferred_auth_method": meta.get("preferred_auth_method"),
                                "model": meta.get("model"),
                                "reasoning_effort": meta.get("reasoning_effort"),
                                "service_tier": meta.get("service_tier"),
                            }
                        )
                    except Exception as e:
                        sys.stderr.write(f"error: failed to record launch failure for {sock}: {type(e).__name__}: {e}\n")
                        sys.stderr.flush()
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

            if self._remember_recent_cwd(cwd, ts=meta.get("updated_ts", meta.get("start_ts"))):
                recent_cwd_dirty = True

            start_ts_raw = meta.get("start_ts")
            if not isinstance(start_ts_raw, (int, float)):
                raise ValueError(f"invalid start_ts in metadata for socket {sock}")
            start_ts = float(start_ts_raw)
            resume_session_id = _clean_optional_text(meta.get("resume_session_id"))
            model_provider, preferred_auth_method, model, reasoning_effort = self._session_run_settings(
                meta=meta,
                log_path=log_path,
                agent_backend=agent_backend,
            )
            service_tier = _normalize_requested_service_tier(meta.get("service_tier")) if agent_backend == "codex" else None

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
                token = _rollout_log._find_latest_token_update(log_path)
            else:
                meta_log_off = 0
                token = None
            if token is None and log_path is None:
                token = resp.get("token") if isinstance(resp.get("token"), (dict, type(None))) else None

            s = Session(
                session_id=session_id,
                thread_id=thread_id,
                broker_pid=broker_pid,
                codex_pid=codex_pid,
                agent_backend=agent_backend,
                owned=owned,
                transport=transport,
                start_ts=float(start_ts),
                cwd=str(cwd),
                log_path=log_path,
                sock_path=sock,
                busy=bool(resp.get("busy")),
                queue_len=int(resp.get("queue_len")),
                token=token,
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
                launch_id=launch_id,
                spawn_nonce=spawn_nonce,
                resume_session_id=resume_session_id,
                pending_attachment=session_id in getattr(self, "_pending_attachment_ids", set()),
                commit_unknown_send=dict(getattr(self, "_commit_unknown_sends", {}).get(session_id) or {}) or None,
                sync_send_supported=sync_send_supported,
                key_write_errors_supported=key_write_errors_supported,
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
                    prev.broker_pid = s.broker_pid
                    prev.codex_pid = s.codex_pid
                    prev.agent_backend = s.agent_backend
                    prev.owned = s.owned
                    prev.transport = s.transport
                    prev.start_ts = s.start_ts
                    prev.cwd = s.cwd
                    prev.busy = s.busy
                    prev.queue_len = s.queue_len
                    prev.token = s.token
                    if prev.log_path != s.log_path:
                        prev.log_path = s.log_path
                        self._reset_log_caches(prev, meta_log_off=meta_log_off)
                    prev.model_provider = model_provider
                    prev.preferred_auth_method = preferred_auth_method
                    prev.model = model
                    prev.reasoning_effort = reasoning_effort
                    prev.service_tier = service_tier
                    prev.tmux_session = tmux_session
                    prev.tmux_window = tmux_window
                    prev.launch_id = launch_id
                    prev.spawn_nonce = spawn_nonce
                    prev.resume_session_id = resume_session_id
                    prev.pending_attachment = bool(prev.pending_attachment or session_id in getattr(self, "_pending_attachment_ids", set()))
                    prev.commit_unknown_send = dict(getattr(self, "_commit_unknown_sends", {}).get(session_id) or {}) or None
                    prev.sync_send_supported = sync_send_supported
                    prev.key_write_errors_supported = key_write_errors_supported
        if recent_cwd_dirty:
            self._save_recent_cwds()
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
                        log_available = s2.log_path is not None and s2.log_path.exists()
                        if not log_available:
                            s2.token = tok
        return True, None

    def _prune_dead_sessions(self) -> None:
        with self._lock:
            items = list(self._sessions.items())
        dead: list[tuple[str, Path, Session]] = []
        for sid, s in items:
            if not s.sock_path.exists():
                dead.append((sid, s.sock_path, s))
                continue
            ok, err = self._refresh_session_state(sid, s.sock_path, timeout_s=0.4)
            if ok:
                continue
            if err is not None and _sock_error_definitely_stale(err):
                dead.append((sid, s.sock_path, s))
                continue
            if _pid_alive(s.broker_pid) or _pid_alive(s.codex_pid):
                continue
            dead.append((sid, s.sock_path, s))
        if not dead:
            return
        with self._lock:
            for sid, _sock, _s in dead:
                self._sessions.pop(sid, None)
        for sid, sock, s in dead:
            existing_launch_failed = False
            latest_launch_record: dict[str, Any] | None = None
            if s.launch_id:
                latest_launch_record = _latest_launch_attempt(s.launch_id)
                existing_launch_failed = bool(latest_launch_record and latest_launch_record.get("state") == "failed")
            if s.owned and s.log_path is None and not existing_launch_failed:
                try:
                    tmux_snapshot: dict[str, Any] = {}
                    if s.transport == "tmux":
                        tmux_bin = shutil.which("tmux")
                        if tmux_bin is not None:
                            pane_id = (
                                _clean_optional_text(latest_launch_record.get("tmux_pane_id"))
                                if isinstance(latest_launch_record, dict)
                                else None
                            )
                            tmux_snapshot = _tmux_pane_snapshot(tmux_bin, pane_id=pane_id, window=s.tmux_window)
                    submitted_messages = _submitted_user_messages(latest_launch_record)
                    prior_tail = _launch_failure_tail(latest_launch_record) if isinstance(latest_launch_record, dict) else ""
                    snapshot_tail = _launch_failure_tail(tmux_snapshot)
                    agent_status = None
                    broker_status = None
                    if isinstance(latest_launch_record, dict):
                        prev_agent_status = latest_launch_record.get("agent_exit_status", latest_launch_record.get("exit_code"))
                        prev_broker_status = latest_launch_record.get("broker_exit_status")
                        if isinstance(prev_agent_status, int):
                            agent_status = prev_agent_status
                        if isinstance(prev_broker_status, int):
                            broker_status = prev_broker_status
                    failure_record: dict[str, Any] = {
                        "launch_id": s.launch_id,
                        "state": "failed",
                        "stage": "session_pruned_before_log_bind",
                        "error": "web-owned session process disappeared before a session log was bound",
                        "agent_backend": s.agent_backend,
                        "cwd": s.cwd,
                        "created_ts": s.start_ts,
                        "broker_pid": s.broker_pid,
                        "agent_pid": s.codex_pid,
                        "transport": s.transport,
                        "tmux_session": s.tmux_session,
                        "tmux_window": s.tmux_window,
                        "spawn_nonce": s.spawn_nonce,
                        "model_provider": s.model_provider,
                        "preferred_auth_method": s.preferred_auth_method,
                        "model": s.model,
                        "reasoning_effort": s.reasoning_effort,
                        "service_tier": s.service_tier,
                    }
                    if submitted_messages:
                        failure_record["submitted_user_messages"] = submitted_messages
                    if prior_tail:
                        failure_record["pty_tail"] = prior_tail
                    if snapshot_tail:
                        failure_record["tmux_pane_tail"] = snapshot_tail
                    if agent_status is not None:
                        failure_record["agent_exit_status"] = agent_status
                    if broker_status is not None:
                        failure_record["broker_exit_status"] = broker_status
                    failure_record.update(tmux_snapshot)
                    _record_launch_attempt(
                        failure_record
                    )
                except Exception as e:
                    sys.stderr.write(f"error: failed to record pruned launch failure for {sid}: {type(e).__name__}: {e}\n")
                    sys.stderr.flush()
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
        recent_cwd_dirty = False
        now_ts = time.time()
        with self._lock:
            items: list[dict[str, Any]] = []
            qmap = getattr(self, "_queues", None)
            meta_map = getattr(self, "_sidebar_meta", None)
            active_ids = set(self._sessions.keys())
            for s in self._sessions.values():
                cfg0 = self._unattended.get(s.session_id)
                unattended_enabled = bool(cfg0.get("enabled")) if isinstance(cfg0, dict) else False
                unattended_cooldown_minutes = _clean_unattended_cooldown_minutes(cfg0.get("cooldown_minutes")) if isinstance(cfg0, dict) else UNATTENDED_DEFAULT_IDLE_MINUTES
                unattended_remaining_injections = (
                    _clean_unattended_remaining_injections(cfg0.get("remaining_injections"), allow_zero=True)
                    if isinstance(cfg0, dict)
                    else UNATTENDED_DEFAULT_MAX_INJECTIONS
                )
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
                needs_run_settings = bool(log_exists and s.log_path is not None and (s.model_provider is None or s.model is None or s.reasoning_effort is None))
                needs_history_scan = bool(s.last_chat_ts is None and log_exists and s.log_path is not None and (not s.last_chat_history_scanned))
                updated_ts = float(s.last_chat_ts) if isinstance(s.last_chat_ts, (int, float)) else float(s.start_ts)
                cwd_recent = _clean_recent_cwd(s.cwd)
                recent_map = getattr(self, "_recent_cwds", None)
                if cwd_recent is not None:
                    if not isinstance(recent_map, dict):
                        self._recent_cwds = {}
                        recent_map = self._recent_cwds
                    prev_recent_ts = recent_map.get(cwd_recent)
                    if prev_recent_ts is None or prev_recent_ts < updated_ts:
                        recent_map[cwd_recent] = updated_ts
                        recent_cwd_dirty = True
                queue_len = 0
                if isinstance(qmap, dict):
                    q0 = qmap.get(s.session_id)
                    if isinstance(q0, list):
                        queue_len = len(q0)
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
                items.append(
                    {
                        "session_id": s.session_id,
                        "thread_id": s.thread_id,
                        "pid": s.codex_pid,
                        "broker_pid": s.broker_pid,
                        "agent_backend": s.agent_backend,
                        "owned": s.owned,
                        "transport": s.transport,
                        "cwd": s.cwd,
                        "start_ts": s.start_ts,
                        "updated_ts": updated_ts,
                        "log_path": (str(s.log_path) if s.log_path is not None else None),
                        "_log_path_obj": s.log_path,
                        "log_exists": log_exists,
                        "needs_run_settings": needs_run_settings,
                        "needs_history_scan": needs_history_scan,
                        "state_busy": bool(s.busy),
                        "queue_len": int(queue_len),
                        "pending_attachment": bool(s.pending_attachment),
                        "commit_unknown_send": bool(s.commit_unknown_send),
                        "commit_unknown_send_text": (str(s.commit_unknown_send.get("text")) if isinstance(s.commit_unknown_send, dict) and isinstance(s.commit_unknown_send.get("text"), str) else None),
                        "commit_unknown_send_ts": (float(s.commit_unknown_send.get("created_ts")) if isinstance(s.commit_unknown_send, dict) and isinstance(s.commit_unknown_send.get("created_ts"), (int, float)) else None),
                        "token": s.token,
                        "thinking": int(s.meta_thinking),
                        "tools": int(s.meta_tools),
                        "system": int(s.meta_system),
                        "unattended_enabled": unattended_enabled,
                        "unattended_cooldown_minutes": unattended_cooldown_minutes,
                        "unattended_remaining_injections": unattended_remaining_injections,
                        "alias": alias,
                        "files": list(files),
                        "_cwd_path_obj": cwd_path,
                        "model_provider": s.model_provider,
                        "preferred_auth_method": s.preferred_auth_method,
                        "provider_choice": _provider_choice_for_settings(
                            model_provider=s.model_provider,
                            preferred_auth_method=s.preferred_auth_method,
                        ),
                        "model": s.model,
                        "reasoning_effort": s.reasoning_effort,
                        "service_tier": s.service_tier,
                        "tmux_session": s.tmux_session,
                        "tmux_window": s.tmux_window,
                        "launch_id": s.launch_id,
                        "spawn_nonce": s.spawn_nonce,
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
            log_path_obj = it.get("_log_path_obj")
            if bool(it.get("needs_history_scan")) and isinstance(log_path_obj, Path):
                # Discovery seeds offsets at EOF, so recover preexisting chat history once.
                conv_ts: float | None
                try:
                    conv_ts = _last_conversation_ts_from_tail(log_path_obj)
                except FileNotFoundError:
                    conv_ts = None
                with self._lock:
                    s_cur = self._sessions.get(sid)
                    if s_cur is not None and s_cur.log_path == log_path_obj and not s_cur.last_chat_history_scanned:
                        s_cur.last_chat_history_scanned = True
                        if isinstance(conv_ts, (int, float)):
                            s_cur.last_chat_ts = float(conv_ts)
                        updated_ts = float(s_cur.last_chat_ts) if isinstance(s_cur.last_chat_ts, (int, float)) else float(s_cur.start_ts)
                        it["updated_ts"] = updated_ts
                        cwd_recent = _clean_recent_cwd(s_cur.cwd)
                        recent_map = getattr(self, "_recent_cwds", None)
                        if cwd_recent is not None:
                            if not isinstance(recent_map, dict):
                                self._recent_cwds = {}
                                recent_map = self._recent_cwds
                            prev_recent_ts = recent_map.get(cwd_recent)
                            if prev_recent_ts is None or prev_recent_ts < updated_ts:
                                recent_map[cwd_recent] = updated_ts
                                recent_cwd_dirty = True
                        elapsed_s = max(0.0, now_ts - updated_ts)
                        time_priority = _priority_from_elapsed_seconds(elapsed_s)
                        base_priority = _clip01(time_priority + float(it.get("priority_offset", 0.0)))
                        final_priority = 0.0 if (it.get("snoozed") or it.get("blocked")) else base_priority
                        it["time_priority"] = time_priority
                        it["base_priority"] = base_priority
                        it["final_priority"] = final_priority
            if bool(it.get("needs_run_settings")) and isinstance(log_path_obj, Path):
                try:
                    log_provider, log_model, log_effort = _read_run_settings_from_log(log_path_obj, agent_backend=str(it.get("agent_backend") or "codex"))
                except (FileNotFoundError, ValueError):
                    log_provider = log_model = log_effort = None
                with self._lock:
                    s_cur = self._sessions.get(sid)
                    if s_cur is not None and s_cur.log_path == log_path_obj:
                        if s_cur.model_provider is None:
                            s_cur.model_provider = log_provider
                        if s_cur.model is None:
                            s_cur.model = log_model
                        if s_cur.reasoning_effort is None:
                            s_cur.reasoning_effort = log_effort
                        it["model_provider"] = s_cur.model_provider
                        it["preferred_auth_method"] = s_cur.preferred_auth_method
                        it["model"] = s_cur.model
                        it["reasoning_effort"] = s_cur.reasoning_effort
                        it["provider_choice"] = _provider_choice_for_settings(
                            model_provider=s_cur.model_provider,
                            preferred_auth_method=s_cur.preferred_auth_method,
                        )
            if (not log_exists) or not isinstance(log_path_obj, Path):
                busy_out = False
            else:
                try:
                    idle_val = bool(self.idle_from_log_path(sid, log_path_obj))
                    busy_out = not idle_val
                except FileNotFoundError:
                    busy_out = False
            cwd_path_obj = it.get("_cwd_path_obj")
            git_branch = _current_git_branch(cwd_path_obj) if isinstance(cwd_path_obj, Path) else None
            it2 = dict(it)
            it2.pop("_log_path_obj", None)
            it2.pop("_cwd_path_obj", None)
            it2.pop("log_exists", None)
            it2.pop("needs_run_settings", None)
            it2.pop("needs_history_scan", None)
            it2.pop("state_busy", None)
            it2["git_branch"] = git_branch
            it2["busy"] = bool(busy_out)
            out.append(it2)
        if bool(getattr(self, "_include_launch_attempts", False)):
            with self._lock:
                hidden_failure_ids = set(getattr(self, "_hidden_sessions", set()))
                active_launch_ids = {
                    s.launch_id
                    for s in self._sessions.values()
                    if isinstance(s.launch_id, str) and s.launch_id
                }
                active_spawn_nonces = {
                    s.spawn_nonce
                    for s in self._sessions.values()
                    if isinstance(s.spawn_nonce, str) and s.spawn_nonce
                }
            for rec in _read_launch_attempts(path=LAUNCH_ATTEMPTS_PATH, max_records=100, max_age_s=24 * 3600):
                row = _launch_attempt_row(rec)
                if row is None:
                    continue
                if row["session_id"] in hidden_failure_ids:
                    continue
                launch_id = row.get("launch_id")
                if isinstance(launch_id, str) and launch_id and launch_id in active_launch_ids:
                    continue
                nonce = row.get("spawn_nonce")
                if isinstance(nonce, str) and nonce and nonce in active_spawn_nonces:
                    continue
                out.append(row)
        if files_dirty:
            self._save_files()
        if sidebar_dirty:
            self._save_sidebar_meta()
        if recent_cwd_dirty:
            self._save_recent_cwds()
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

    def refresh_session_meta(self, session_id: str, *, drain_queue: bool = False) -> None:
        # The broker may rewrite the sock .json when Codex switches threads (/new, /resume).
        # Refresh the log path and thread id without requiring the UI to poll /api/sessions.
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return
            sock = s.sock_path
            current_log_path = s.log_path
        meta_path = sock.with_suffix(".json")
        if not meta_path.exists():
            self._prune_stale_socket_without_metadata(session_id, sock)
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError(f"invalid metadata json for socket {sock}")

        thread_id = meta.get("session_id") if isinstance(meta.get("session_id"), str) and meta.get("session_id") else s.thread_id
        owned = (meta.get("owner") == "web") if isinstance(meta.get("owner"), str) else s.owned
        agent_backend = normalize_agent_backend(meta.get("agent_backend"), default=s.agent_backend)
        transport, tmux_session, tmux_window = self._session_transport(meta=meta)
        sync_send_supported = _metadata_sync_send_supported(meta)
        key_write_errors_supported = _metadata_key_write_errors_supported(meta)
        cwd_raw = meta.get("cwd")
        if not isinstance(cwd_raw, str) or (not cwd_raw.strip()):
            raise ValueError(f"invalid cwd in metadata for socket {sock}")
        cwd = cwd_raw
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
        if log_path is not None and not log_path.exists():
            log_path = None
        ignored_rollout_paths = _metadata_ignored_rollout_paths(meta, sock=sock)
        if _metadata_detaches_current_log(meta, current_log_path):
            try:
                tail_state = self._sock_call(sock, {"cmd": "tail"}, timeout_s=0.4)
            except Exception:
                tail_state = {}
            if _broker_tail_has_session_detach_marker(agent_backend, tail_state.get("tail") if isinstance(tail_state, dict) else None):
                ignored_rollout_paths.add(current_log_path)
        if log_path is None and agent_backend in {"codex", "cc"} and _pid_alive(s.codex_pid):
            discovered_log_path = _proc_find_open_rollout_log(
                proc_root=PROC_ROOT,
                root_pid=s.codex_pid,
                agent_backend=agent_backend,
                cwd=cwd,
                ignored_paths=ignored_rollout_paths,
            )
            if discovered_log_path is not None and discovered_log_path.exists():
                log_path = discovered_log_path
        if log_path is not None and agent_backend == "codex":
            session_meta = _read_session_meta_or_none(log_path, agent_backend="codex", context="session refresh")
            meta_session_id = session_meta.get("id") if session_meta else None
            if isinstance(meta_session_id, str) and meta_session_id:
                thread_id = meta_session_id
            thread_id, log_path = _coerce_main_thread_log(thread_id=thread_id, log_path=log_path)

        resume_session_id = _clean_optional_text(meta.get("resume_session_id"))
        model_provider, preferred_auth_method, model, reasoning_effort = self._session_run_settings(
            meta=meta,
            log_path=log_path,
            agent_backend=agent_backend,
        )
        service_tier = _normalize_requested_service_tier(meta.get("service_tier")) if agent_backend == "codex" else None

        with self._lock:
            s2 = self._sessions.get(session_id)
            if not s2:
                return
            s2.thread_id = thread_id
            s2.agent_backend = agent_backend
            s2.cwd = str(cwd)
            s2.owned = bool(owned)
            s2.transport = transport
            if s2.log_path != log_path:
                s2.log_path = log_path
                if log_path is not None:
                    log_off = int(log_path.stat().st_size)
                else:
                    log_off = 0
                self._reset_log_caches(s2, meta_log_off=log_off)
            s2.model_provider = model_provider
            s2.preferred_auth_method = preferred_auth_method
            s2.model = model
            s2.reasoning_effort = reasoning_effort
            s2.service_tier = service_tier
            s2.tmux_session = tmux_session
            s2.tmux_window = tmux_window
            s2.resume_session_id = resume_session_id
            s2.sync_send_supported = sync_send_supported
            s2.key_write_errors_supported = key_write_errors_supported
        if drain_queue and self._queue_len(session_id) > 0:
            self._maybe_drain_session_queue(session_id)

    def _attach_notification_texts(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        voice_push = getattr(self, "_voice_push", None)
        if voice_push is None:
            return list(events)
        out: list[dict[str, Any]] = []
        for ev in events:
            if not isinstance(ev, dict):
                out.append(ev)
                continue
            if ev.get("role") != "assistant" or ev.get("message_class") != "final_response":
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

    def mark_log_delta(self, session_id: str, *, objs: list[dict[str, Any]], new_off: int) -> None:
        _th, _tools, _sys, last_ts, token_update, _chat_events = _analyze_log_chunk(objs)
        model = None
        reasoning_effort = None
        for obj in reversed(objs):
            if not isinstance(obj, dict) or obj.get("type") != "turn_context":
                continue
            model, reasoning_effort = _turn_context_run_settings(obj.get("payload"))
            break
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                if isinstance(last_ts, (int, float)):
                    tsf = float(last_ts)
                    s.last_chat_ts = tsf if s.last_chat_ts is None else max(s.last_chat_ts, tsf)
                if model is not None:
                    s.model = model
                if reasoning_effort is not None:
                    s.reasoning_effort = reasoning_effort
                s.idle_cache_log_off = -1

    def idle_from_log(self, session_id: str) -> bool:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            lp = s.log_path
        if lp is None:
            raise FileNotFoundError(f"missing rollout log for session {session_id}")
        return self.idle_from_log_path(session_id, lp)

    def idle_from_log_path(self, session_id: str, log_path: Path) -> bool:
        lp = log_path
        with self._lock:
            s = self._sessions.get(session_id)
            cache_matches_path = bool(s and s.log_path == lp)
            cached_off = int(s.idle_cache_log_off) if cache_matches_path and s else -1
            cached_idle = s.idle_cache_value if cache_matches_path and s else None
        if not lp.exists():
            raise FileNotFoundError(f"missing rollout log for session {session_id}")
        sz = int(lp.stat().st_size)
        if cache_matches_path and (sz >= 0) and (cached_off == sz) and isinstance(cached_idle, bool):
            return bool(cached_idle)
        idle = _compute_idle_from_log(lp)
        with self._lock:
            s2 = self._sessions.get(session_id)
            if s2 and s2.log_path == lp:
                s2.idle_cache_log_off = sz
                s2.idle_cache_value = idle
        if idle is None:
            raise RuntimeError("unable to compute idle state from log")
        return bool(idle)

    def _sock_call(self, sock_path: Path, req: dict[str, Any], timeout_s: float | None = 2.0, *, track_request_sent: bool = False) -> dict[str, Any]:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout_s)
        request_sent = False
        try:
            s.connect(str(sock_path))
            s.sendall((json.dumps(req) + "\n").encode("utf-8"))
            request_sent = True
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
        except Exception as e:
            if track_request_sent:
                raise ControlSocketCallError(str(e), request_sent=request_sent) from e
            raise
        finally:
            s.close()

    def _kill_session_via_pids(self, s: Session) -> bool:
        group_alive = _process_group_alive(int(s.codex_pid))
        broker_alive = _pid_alive(int(s.broker_pid))
        if not group_alive and not broker_alive:
            _unlink_quiet(s.sock_path)
            _unlink_quiet(s.sock_path.with_suffix(".json"))
            return True
        if group_alive and (not _terminate_process_group(int(s.codex_pid), wait_seconds=1.0)):
            return False
        if _pid_alive(int(s.broker_pid)) and (not _terminate_process(int(s.broker_pid), wait_seconds=1.0)):
            return False
        group_dead = not _process_group_alive(int(s.codex_pid))
        broker_dead = not _pid_alive(int(s.broker_pid))
        if group_dead and broker_dead:
            _unlink_quiet(s.sock_path)
            _unlink_quiet(s.sock_path.with_suffix(".json"))
            return True
        return False

    def kill_session(self, session_id: str) -> bool:
        with self._lock:
            s = self._sessions.get(session_id)
        if not s:
            return False
        try:
            resp = self._sock_call(s.sock_path, {"cmd": "shutdown"}, timeout_s=1.0)
        except Exception:
            return self._kill_session_via_pids(s)
        if resp.get("ok") is True:
            return True
        return self._kill_session_via_pids(s)

    def _live_session_for_resume_target(self, resume_id: str, resume_row: dict[str, Any] | None) -> Session | None:
        sessions = getattr(self, "_sessions", None)
        if not isinstance(sessions, dict):
            return None
        target_log_raw = _clean_optional_text(resume_row.get("log_path") if isinstance(resume_row, dict) else None)
        try:
            target_log = str(Path(target_log_raw).expanduser().resolve(strict=False)) if target_log_raw is not None else None
        except OSError:
            target_log = target_log_raw

        lock = getattr(self, "_lock", None)
        if lock is not None and hasattr(lock, "acquire") and hasattr(lock, "release"):
            with lock:
                values = list(sessions.values())
        else:
            values = list(sessions.values())
        for session in values:
            if not isinstance(session, Session):
                continue
            same_thread = session.session_id == resume_id or session.thread_id == resume_id
            same_log = False
            if target_log is not None and session.log_path is not None:
                try:
                    session_log = str(session.log_path.expanduser().resolve(strict=False))
                except OSError:
                    session_log = str(session.log_path)
                same_log = session_log == target_log
            if (same_thread or same_log) and (_pid_alive(session.broker_pid) or _pid_alive(session.codex_pid)):
                return session
        return None

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
    ) -> dict[str, Any]:
        backend_name = normalize_agent_backend(agent_backend)
        cwd_path = _resolve_dir_target(cwd, field_name="cwd")
        if not cwd_path.exists():
            try:
                cwd_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                detail = e.strerror or str(e)
                raise ValueError(f"cwd could not be created: {cwd_path}: {detail}") from e
        if not cwd_path.is_dir():
            raise ValueError(f"cwd is not a directory: {cwd_path}")
        cwd3 = str(cwd_path)
        if resume_session_id is not None and worktree_branch is not None:
            raise ValueError("worktree_branch cannot be used when resuming a session")
        spawn_cwd = cwd_path
        if worktree_branch is not None:
            spawn_cwd = _create_git_worktree(cwd_path, worktree_branch)

        argv = [sys.executable, "-m", "codoxear.broker", "--cwd", str(spawn_cwd), "--"]
        backend_args = _build_backend_args(
            agent_backend=backend_name,
            spawn_cwd=spawn_cwd,
            codex_trust_override=_codex_trust_override_for_path(spawn_cwd),
            model_provider=model_provider,
            preferred_auth_method=preferred_auth_method,
            model=model,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
        )
        resume_row: dict[str, Any] | None = None
        if resume_session_id is not None:
            resume_id = str(resume_session_id).strip()
            if not resume_id:
                raise ValueError("resume_session_id must be a non-empty string")
            found = False
            for row in _list_resume_candidates_for_cwd(cwd3, agent_backend=backend_name, limit=1000):
                if row.get("session_id") == resume_id:
                    found = True
                    resume_row = row
                    break
            if not found:
                raise ValueError(f"resume session not found for cwd: {resume_id}")
            live_target = self._live_session_for_resume_target(resume_id, resume_row)
            if live_target is not None:
                raise ValueError(
                    "resume target is already live as "
                    f"{live_target.session_id}; select that session instead of creating another session bound to the same transcript"
                )
            backend_args.extend(_build_backend_resume_args(agent_backend=backend_name, resume_id=resume_id, resume_row=resume_row))
        backend_args.extend(args or [])
        argv.extend(backend_args)

        env = dict(os.environ)
        if _DOTENV.exists():
            for k, v in _load_env_file(_DOTENV).items():
                env.setdefault(k, v)
        _apply_backend_environment(
            env,
            agent_backend=backend_name,
            homes={"codex": CODEX_HOME, "pi": PI_HOME, "cc": CC_HOME},
            model_provider=model_provider,
            preferred_auth_method=preferred_auth_method,
            model=model,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            resume_session_id=resume_session_id,
        )

        launch_started_ts = time.time()
        launch_id = f"launch-{int(launch_started_ts * 1000)}-{secrets.token_hex(4)}"
        spawn_nonce = secrets.token_hex(8)
        env["CODEX_WEB_LAUNCH_ID"] = launch_id
        env["CODEX_WEB_SPAWN_NONCE"] = spawn_nonce

        base_launch_record: dict[str, Any] = {
            "launch_id": launch_id,
            "state": "starting",
            "agent_backend": backend_name,
            "cwd": str(spawn_cwd),
            "requested_cwd": cwd3,
            "created_ts": launch_started_ts,
            "updated_ts": launch_started_ts,
            "spawn_nonce": spawn_nonce,
            "model_provider": model_provider,
            "preferred_auth_method": preferred_auth_method,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "service_tier": service_tier,
            "resume_session_id": resume_session_id,
            "worktree_branch": worktree_branch,
        }

        def record_launch(state: str, **extra: Any) -> dict[str, Any]:
            rec = dict(base_launch_record)
            rec["state"] = state
            rec["updated_ts"] = time.time()
            rec.update(extra)
            return _record_launch_attempt(rec)

        record_launch("starting", transport="tmux" if create_in_tmux else "direct")

        def fail_launch(stage: str, error: BaseException | str, **extra: Any) -> None:
            msg = str(error)
            rec: dict[str, Any] = dict(base_launch_record)
            rec.update(
                {
                    "state": "failed",
                    "stage": stage,
                    "error": msg,
                    "updated_ts": time.time(),
                }
            )
            rec.update(extra)
            try:
                rec = _record_launch_attempt(rec)
            except Exception as log_exc:
                sys.stderr.write(f"error: failed to write launch attempt record: {type(log_exc).__name__}: {log_exc}\n")
                sys.stderr.flush()
            raise SessionLaunchError(rec)

        def tmux_launch_fields(snapshot: dict[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
            out = dict(snapshot or {})
            out.update(fields)
            return out

        if create_in_tmux:
            tmux_bin = shutil.which("tmux")
            if tmux_bin is None:
                raise ValueError("tmux is unavailable on this host")
            tmux_window = _safe_filename(f"{Path(spawn_cwd).name or 'session'}-{spawn_nonce[:6]}", default="session")
            env["CODEX_WEB_TRANSPORT"] = "tmux"
            env["CODEX_WEB_TMUX_SESSION"] = TMUX_SESSION_NAME
            env["CODEX_WEB_TMUX_WINDOW"] = tmux_window
            backend_bin_env_var = get_agent_backend(backend_name).bin_env_var
            inline_env = _build_tmux_inline_env(
                env,
                agent_backend=backend_name,
                tmux_session=TMUX_SESSION_NAME,
                tmux_window=tmux_window,
                launch_id=launch_id,
                spawn_nonce=spawn_nonce,
                resume_session_id=resume_session_id,
                model_provider=model_provider,
                preferred_auth_method=preferred_auth_method,
                model=model,
                reasoning_effort=reasoning_effort,
                service_tier=service_tier,
                inherited_backend_bin=_clean_optional_text(os.environ.get(backend_bin_env_var)),
            )
            repo_root = Path(__file__).resolve().parent.parent
            tmux_unset_vars = _tmux_unset_vars()
            inline_argv = ["env", *[f"{key}={value}" for key, value in inline_env.items()], *argv]
            shell_cmd = f"cd {shlex.quote(str(repo_root))} && unset {shlex.join(tmux_unset_vars)} && exec {shlex.join(inline_argv)}"
            new_window_argv = [tmux_bin, "new-window", "-d", "-P", "-F", "#{pane_id}", "-t", f"{TMUX_SESSION_NAME}:", "-n", tmux_window, shell_cmd]
            new_session_argv = [tmux_bin, "new-session", "-d", "-P", "-F", "#{pane_id}", "-s", TMUX_SESSION_NAME, "-n", tmux_window, shell_cmd]

            def tmux_run(argv2: list[str]) -> subprocess.CompletedProcess[str]:
                return subprocess.run(argv2, capture_output=True, text=True, env=env, check=False)

            def tmux_detail(proc2: subprocess.CompletedProcess[str]) -> str:
                return (proc2.stderr or proc2.stdout or f"exit status {proc2.returncode}").strip()

            def tmux_missing_session(detail2: str) -> bool:
                low = detail2.lower()
                return "can't find session" in low or "no server running" in low or "error connecting to" in low

            def tmux_duplicate_session(detail2: str) -> bool:
                return "duplicate session" in detail2.lower()

            attempts: list[dict[str, Any]] = []
            tmux_proc = tmux_run(new_window_argv)
            attempts.append({"cmd": "new-window", "returncode": tmux_proc.returncode, "stderr": (tmux_proc.stderr or "").strip(), "stdout": (tmux_proc.stdout or "").strip()})
            if tmux_proc.returncode != 0 and tmux_missing_session(tmux_detail(tmux_proc)):
                tmux_proc = tmux_run(new_session_argv)
                attempts.append({"cmd": "new-session", "returncode": tmux_proc.returncode, "stderr": (tmux_proc.stderr or "").strip(), "stdout": (tmux_proc.stdout or "").strip()})
                if tmux_proc.returncode != 0 and tmux_duplicate_session(tmux_detail(tmux_proc)):
                    tmux_proc = tmux_run(new_window_argv)
                    attempts.append({"cmd": "new-window-after-duplicate", "returncode": tmux_proc.returncode, "stderr": (tmux_proc.stderr or "").strip(), "stdout": (tmux_proc.stdout or "").strip()})
            tmux_pane_id = _clean_optional_text(tmux_proc.stdout)
            if tmux_proc.returncode != 0:
                detail = tmux_detail(tmux_proc)
                fail_launch(
                    "tmux_launch",
                    f"tmux launch failed: {detail}",
                    transport="tmux",
                    tmux_session=TMUX_SESSION_NAME,
                    tmux_window=tmux_window,
                    spawn_nonce=spawn_nonce,
                    tmux_exit_status=tmux_proc.returncode,
                    tmux_stdout=(tmux_proc.stdout or "").strip(),
                    tmux_stderr=(tmux_proc.stderr or "").strip(),
                    tmux_attempts=attempts,
                )
            snapshot = _tmux_pane_snapshot(tmux_bin, pane_id=tmux_pane_id, window=tmux_window)
            record_launch(
                "tmux_pane_created",
                **tmux_launch_fields(
                    snapshot,
                    transport="tmux",
                    tmux_session=TMUX_SESSION_NAME,
                    tmux_window=tmux_window,
                    tmux_attempts=attempts,
                ),
            )
            try:
                meta = _wait_for_spawned_broker_meta(spawn_nonce)
            except Exception as e:
                if tmux_pane_id is not None and not snapshot.get("tmux_inspect_error") and str(snapshot.get("tmux_pane_dead") or "0") != "1":
                    record_launch(
                        "tmux_pane_created",
                        **tmux_launch_fields(
                            snapshot,
                            stage="broker_metadata_pending",
                            error=str(e),
                            transport="tmux",
                            tmux_session=TMUX_SESSION_NAME,
                            tmux_window=tmux_window,
                        ),
                    )
                    return {"launch_id": launch_id, "pending": True, "tmux_session": TMUX_SESSION_NAME, "tmux_window": tmux_window}
                fail_launch(
                    "broker_metadata",
                    e,
                    **tmux_launch_fields(
                        _tmux_pane_snapshot(tmux_bin, pane_id=tmux_pane_id, window=tmux_window),
                        transport="tmux",
                        tmux_session=TMUX_SESSION_NAME,
                        tmux_window=tmux_window,
                        tmux_pane_id=tmux_pane_id,
                        spawn_nonce=spawn_nonce,
                    ),
                )
            broker_pid = meta.get("broker_pid")
            if not isinstance(broker_pid, int):
                fail_launch(
                    "broker_metadata",
                    "tmux launch metadata is missing broker_pid",
                    transport="tmux",
                    tmux_session=TMUX_SESSION_NAME,
                    tmux_window=tmux_window,
                    tmux_pane_id=tmux_pane_id,
                    spawn_nonce=spawn_nonce,
                    metadata=meta,
                )
            record_launch(
                "broker_meta_bound",
                transport="tmux",
                tmux_session=TMUX_SESSION_NAME,
                tmux_window=tmux_window,
                tmux_pane_id=tmux_pane_id,
                broker_pid=int(broker_pid),
            )
            return {"broker_pid": int(broker_pid), "tmux_session": TMUX_SESSION_NAME, "tmux_window": tmux_window}

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
            fail_launch("broker_spawn", f"spawn failed: {e}", transport="direct")

        try:
            _wait_or_raise(proc, label="broker", timeout_s=1.5)
        except Exception as e:
            fail_launch("broker_early_exit", e, transport="direct", broker_pid=int(proc.pid))
        record_launch("broker_spawned", transport="direct", broker_pid=int(proc.pid))
        if proc.stderr is not None:
            threading.Thread(target=_drain_stream, args=(proc.stderr,), daemon=True).start()

        # Prevent zombies when the broker exits.
        threading.Thread(target=proc.wait, daemon=True).start()
        return {"broker_pid": int(proc.pid)}

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            s = self._sessions.get(session_id)
        if not s:
            for rec in _read_launch_attempts(path=LAUNCH_ATTEMPTS_PATH, max_records=100, max_age_s=24 * 3600):
                row = _launch_attempt_row(rec)
                if row is not None and row.get("session_id") == session_id:
                    self._hide_session(session_id)
                    return True
            return False
        ok = self.kill_session(session_id)
        if ok:
            launch_id = s.launch_id
            self.files_clear(session_id)
            with self._lock:
                self._sessions.pop(session_id, None)
            if launch_id:
                self._hide_session(launch_id)
            self._clear_deleted_session_state(session_id)
        return ok

    def _record_prelog_user_message(self, session: Session, text: str, *, source: str) -> None:
        if not session.owned or session.log_path is not None or not session.launch_id:
            return
        previous = _latest_launch_attempt(session.launch_id)
        messages = _submitted_user_messages(previous)
        messages.append({"text": text, "ts": time.time(), "source": source})
        if len(messages) > 20:
            messages = messages[-20:]
        base: dict[str, Any] = dict(previous) if isinstance(previous, dict) else {}
        base.update(
            {
                "launch_id": session.launch_id,
                "state": _clean_optional_text(base.get("state")) or "broker_meta_bound",
                "agent_backend": session.agent_backend,
                "cwd": session.cwd,
                "created_ts": base.get("created_ts", session.start_ts),
                "updated_ts": time.time(),
                "broker_pid": session.broker_pid,
                "agent_pid": session.codex_pid,
                "transport": session.transport,
                "tmux_session": session.tmux_session,
                "tmux_window": session.tmux_window,
                "spawn_nonce": session.spawn_nonce,
                "model_provider": session.model_provider,
                "preferred_auth_method": session.preferred_auth_method,
                "model": session.model,
                "reasoning_effort": session.reasoning_effort,
                "service_tier": session.service_tier,
                "submitted_user_messages": messages,
            }
        )
        _record_launch_attempt(base)

    def send(self, session_id: str, text: str, *, allow_pending_attachment: bool = False, queue_item_id: str | None = None) -> dict[str, Any]:
        input_lock = self._input_lock_for_session(session_id)
        with input_lock:
            with self._lock:
                s = self._sessions.get(session_id)
                if not s:
                    raise KeyError("unknown session")
                if s.commit_unknown_send:
                    raise SessionNotReadyError("resolve the unknown send before submitting more text")
                if s.pending_attachment and not allow_pending_attachment:
                    raise SessionNotReadyError("send the pending attachment explicitly before submitting other text")
                local_queue_len = self._queue_store_for_manager().queue_len(self._queues, session_id)
                if queue_item_id is None and (local_queue_len > 0 or s.queue_sending_item_id is not None):
                    raise SessionNotReadyError("send queued prompts before submitting new text")
                if queue_item_id is not None and s.queue_sending_item_id != queue_item_id:
                    raise SessionNotReadyError("queued prompt is no longer active")
                if not s.sync_send_supported:
                    raise SessionNotReadyError("broker must be restarted before confirmed sends are available")
                sock = s.sock_path
            if not self._send_remote_ready(session_id, allow_pending_attachment=allow_pending_attachment):
                raise SessionNotReadyError("session is busy; wait before sending")

            def raise_commit_unknown(message: str, cause: BaseException | None = None) -> None:
                if queue_item_id is None:
                    self._set_commit_unknown_send(
                        session_id,
                        {"text": text, "created_ts": time.time(), "error": message},
                    )
                if cause is None:
                    raise SessionCommitUnknownError(message)
                raise SessionCommitUnknownError(message) from cause

            try:
                timeout_s = SEND_COMMIT_TIMEOUT_SECONDS if SEND_COMMIT_TIMEOUT_SECONDS > 0 else None
                resp = self._sock_call(sock, {"cmd": "send", "text": text, "sync": True}, timeout_s=timeout_s, track_request_sent=True)
            except ControlSocketCallError as e:
                if e.request_sent:
                    raise_commit_unknown("send commit status unknown; broker response failed", e)
                if not _pid_alive(s.broker_pid) and not _pid_alive(s.codex_pid):
                    with self._lock:
                        self._sessions.pop(session_id, None)
                    self._clear_deleted_session_state(session_id)
                    _unlink_quiet(sock)
                    _unlink_quiet(sock.with_suffix(".json"))
                    raise KeyError("unknown session")
                raise SessionNotReadyError("session control socket unavailable") from e
            except (TimeoutError, socket.timeout) as e:
                raise_commit_unknown("send commit status unknown; broker did not reply before timeout", e)
            if not isinstance(resp, dict):
                raise_commit_unknown("send commit status unknown; broker response was malformed")
            if bool(resp.get("commit_unknown")):
                raise_commit_unknown("send commit status unknown; broker marked commit unknown")
            if resp.get("error"):
                err = str(resp.get("error"))
                if err == "empty response":
                    raise_commit_unknown("send commit status unknown; broker response was empty")
                if bool(resp.get("commit_unknown")):
                    raise_commit_unknown(f"send commit status unknown; {err}")
                raise SessionInjectionError(err)
            if "queue_len" not in resp:
                raise_commit_unknown("send commit status unknown; broker response was incomplete")
            queue_len_raw = resp.get("queue_len")
            if isinstance(queue_len_raw, bool) or not isinstance(queue_len_raw, int) or queue_len_raw < 0:
                raise_commit_unknown("send commit status unknown; broker response was invalid")
            queue_len = int(queue_len_raw)
            with self._lock:
                self._record_prelog_user_message(s, text, source="send")
                s2 = self._sessions.get(session_id)
                if s2:
                    if "busy" in resp:
                        s2.busy = bool(resp.get("busy"))
                    s2.queue_len = queue_len
            self._set_pending_attachment(session_id, False)
            if queue_item_id is None:
                self._set_commit_unknown_send(session_id, None)
        return resp

    def enqueue(self, session_id: str, text: str) -> dict[str, Any]:
        input_lock = self._input_lock_for_session(session_id)
        with input_lock:
            with self._lock:
                s = self._sessions.get(session_id)
                if not s:
                    raise KeyError("unknown session")
                if s.commit_unknown_send:
                    raise SessionNotReadyError("resolve the unknown send before queueing another prompt")
                if s.pending_attachment:
                    raise SessionNotReadyError("send the pending attachment before queueing another prompt")
                if not s.sync_send_supported:
                    raise SessionNotReadyError("broker must be restarted before queueing prompts")
            item, ql = self._queue_append_item_local(session_id, text)
        if ql != 1:
            return {"queued": True, "queue_len": int(ql), "item": item}
        resp = self._promote_queue_head_if_sendable(session_id, require_idle_grace=False, expected_item_id=str(item["id"]))
        if isinstance(resp, dict):
            return resp
        return {"queued": True, "queue_len": 1, "item": item}

    def queue_list(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("unknown session")
        return self._queue_list_local(session_id)

    def queue_delete(self, session_id: str, item_id: str, *, allow_commit_unknown: bool = False) -> dict[str, Any]:
        return self._queue_delete_local(session_id, item_id, allow_commit_unknown=allow_commit_unknown)

    def queue_update(self, session_id: str, item_id: str, text: str) -> dict[str, Any]:
        return self._queue_update_local(session_id, item_id, text)

    def queue_move(self, session_id: str, item_id: str, to_index: int) -> dict[str, Any]:
        return self._queue_move_local(session_id, item_id, to_index)

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
                        log_available = s2.log_path is not None and s2.log_path.exists()
                        if not log_available:
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

    def _refresh_session_meta_if_sidecar_exists(self, session_id: str, *, drain_queue: bool = False) -> None:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            meta_path = s.sock_path.with_suffix(".json")
        if meta_path.exists():
            self.refresh_session_meta(session_id, drain_queue=drain_queue)

    def attachment_injection_ready(self, session_id: str) -> bool:
        self._refresh_session_meta_if_sidecar_exists(session_id, drain_queue=False)
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            if s.commit_unknown_send:
                raise SessionNotReadyError("resolve the unknown send before attaching a file")
            if not (s.sync_send_supported and s.key_write_errors_supported):
                raise SessionNotReadyError("broker must be restarted before file attachments are available")
            if s.queue_sending_item_id is not None:
                return False
            if self._queue_store_for_manager().queue_len(self._queues, session_id) > 0:
                return False
            log_path = s.log_path
        st = self.get_state(session_id)
        if not isinstance(st, dict) or "busy" not in st or "queue_len" not in st:
            raise ValueError("invalid broker state response")
        if bool(st.get("busy")) or int(st.get("queue_len")) > 0:
            return False
        self._refresh_session_meta_if_sidecar_exists(session_id, drain_queue=False)
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            if s.commit_unknown_send:
                raise SessionNotReadyError("resolve the unknown send before attaching a file")
            if s.queue_sending_item_id is not None:
                return False
            if self._queue_store_for_manager().queue_len(self._queues, session_id) > 0:
                return False
            log_path = s.log_path
        if isinstance(log_path, Path) and log_path.exists() and (not self.idle_from_log(session_id)):
            return False
        return True

    def inject_attachment_keys(self, session_id: str, seq: str) -> dict[str, Any]:
        input_lock = self._input_lock_for_session(session_id)
        with input_lock:
            if not self.attachment_injection_ready(session_id):
                raise SessionNotReadyError("session is busy; wait before attaching a file")
            try:
                resp = self.inject_keys(session_id, seq, track_request_sent=True)
            except SessionCommitUnknownError:
                self._set_pending_attachment(session_id, True)
                raise
            if not isinstance(resp, dict):
                self._set_pending_attachment(session_id, True)
                raise SessionCommitUnknownError("attachment commit status unknown; broker response was malformed")
            if bool(resp.get("commit_unknown")):
                self._set_pending_attachment(session_id, True)
                raise SessionCommitUnknownError("attachment commit status unknown; broker marked commit unknown")
            if resp.get("error"):
                err = str(resp.get("error"))
                if bool(resp.get("commit_unknown")) or err == "empty response":
                    self._set_pending_attachment(session_id, True)
                    raise SessionCommitUnknownError(f"attachment commit status unknown; {err}")
                raise SessionInjectionError(err)
            if resp.get("ok") is not True:
                self._set_pending_attachment(session_id, True)
                raise SessionCommitUnknownError("attachment commit status unknown; broker response was incomplete")
            self._set_pending_attachment(session_id, True)
            return resp

    def inject_keys(self, session_id: str, seq: str, *, track_request_sent: bool = False) -> dict[str, Any]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            sock = s.sock_path
        try:
            resp = self._sock_call(sock, {"cmd": "keys", "seq": seq}, timeout_s=2.0, track_request_sent=track_request_sent)
        except ControlSocketCallError as e:
            if track_request_sent and e.request_sent:
                raise SessionCommitUnknownError("attachment commit status unknown; broker response failed") from e
            if not _pid_alive(s.broker_pid) and not _pid_alive(s.codex_pid):
                with self._lock:
                    self._sessions.pop(session_id, None)
                _unlink_quiet(sock)
                _unlink_quiet(sock.with_suffix(".json"))
                raise KeyError("unknown session")
            raise
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
        STATIC_ASSET_VERSION_PLACEHOLDER.encode("ascii"): _static_asset_version(path.parent).encode("ascii"),
        STATIC_ATTACH_MAX_BYTES_PLACEHOLDER.encode("ascii"): str(ATTACH_UPLOAD_MAX_BYTES).encode("ascii"),
    }
    for placeholder, value in replacements.items():
        if placeholder in data:
            data = data.replace(placeholder, value)
    return data


def _message_runtime_snapshot(
    session_id: str,
    s: Session,
    *,
    token_update: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool, int, dict[str, Any] | None]:
    state = MANAGER.get_state(session_id)
    if not isinstance(state, dict):
        raise ValueError("invalid broker state response")
    if "busy" not in state:
        raise ValueError("missing busy from broker state response")
    if "queue_len" not in state:
        raise ValueError("missing queue_len from broker state response")
    if s.log_path is not None and s.log_path.exists():
        idle_val = MANAGER.idle_from_log(session_id)
        busy_val = not bool(idle_val)
    else:
        busy_val = False
    queue_val = MANAGER._queue_len(session_id)
    token_val: dict[str, Any] | None = None
    if "token" in state:
        state_token = state.get("token")
        if not (isinstance(state_token, dict) or state_token is None):
            raise ValueError("invalid token from broker state response")
    log_available = s.log_path is not None and s.log_path.exists()
    if isinstance(token_update, dict):
        token_val = token_update
    elif isinstance(s.token, dict):
        token_val = s.token
    elif (not log_available) and "token" in state and isinstance(state.get("token"), dict):
        token_val = state.get("token")
    return state, bool(busy_val), int(queue_val), token_val


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "codoxear/0.1"

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except Exception as e:
            if _is_client_disconnect(e):
                return
            raise

    def finish(self) -> None:
        try:
            super().finish()
        except Exception as e:
            if _is_client_disconnect(e):
                return
            raise

    def _send_static(self, rel: str) -> None:
        static_root = STATIC_DIR.resolve()
        path = (static_root / rel.lstrip("/")).resolve()
        try:
            path.relative_to(static_root)
        except ValueError:
            self.send_error(404)
            return
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        data = _read_static_bytes(path)
        if path.suffix == ".html":
            ctype = "text/html; charset=utf-8"
        elif path.suffix == ".js":
            ctype = "text/javascript; charset=utf-8"
        elif path.suffix == ".css":
            ctype = "text/css; charset=utf-8"
        elif path.suffix == ".webmanifest":
            ctype = "application/manifest+json; charset=utf-8"
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
        if path.suffix == ".html":
            self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
            self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Length", str(len(data)))
        # UI is used for interactive debugging; serve assets without caching by
        # default so changes (including inline JS) show up immediately on
        # refresh. Packaged deployments may opt into immutable static caching
        # with CODEX_WEB_STATIC_CACHE=1.
        for name, value in _static_cache_control_headers().items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def _unauthorized(self) -> None:
        _json_response(self, 401, {"error": "unauthorized"})

    def _parse_prefixed_request_path(self) -> tuple[urllib.parse.ParseResult, str] | None:
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
                return None
            stripped = _strip_url_prefix(URL_PREFIX, path)
            if stripped is None:
                self.send_error(404)
                return None
            path = stripped
        return u, path

    def _handle_static_get(self, path: str) -> bool:
        if path == "/favicon.ico":
            self._send_static("favicon.png")
            return True
        if path == "/manifest.webmanifest":
            self._send_static("manifest.webmanifest")
            return True
        if path == "/service-worker.js":
            self._send_static("service-worker.js")
            return True
        if path == "/app.js":
            self._send_static("app.js")
            return True
        if path == "/app.css":
            self._send_static("app.css")
            return True
        if path == "/favicon.png":
            self._send_static("favicon.png")
            return True
        if path == "/":
            self._send_static("index.html")
            return True
        if path.startswith("/static/"):
            self._send_static(path[len("/static/") :])
            return True
        return False

    def _handle_voice_get(self, path: str, query: str) -> bool:
        if path == "/api/settings/voice":
            if not _require_auth(self):
                self._unauthorized()
                return True
            _json_response(self, 200, {"ok": True, **MANAGER._voice_push.settings_snapshot(redact_secrets=True)})
            return True

        if path == "/api/notifications/subscription":
            if not _require_auth(self):
                self._unauthorized()
                return True
            _json_response(self, 200, {"ok": True, **MANAGER._voice_push.subscriptions_snapshot()})
            return True

        if path == "/api/notifications/message":
            if not _require_auth(self):
                self._unauthorized()
                return True
            qs = urllib.parse.parse_qs(query)
            message_id = (qs.get("message_id") or [""])[0].strip()
            if not message_id:
                _json_response(self, 400, {"error": "message_id required"})
                return True
            state = MANAGER._voice_push.notification_state_for_message(message_id)
            if state is None:
                _json_response(self, 404, {"error": "unknown message"})
                return True
            _json_response(self, 200, {"ok": True, **state})
            return True

        if path == "/api/notifications/feed":
            if not _require_auth(self):
                self._unauthorized()
                return True
            qs = urllib.parse.parse_qs(query)
            since_raw = (qs.get("since") or ["0"])[0].strip()
            try:
                since_ts = float(since_raw or "0")
            except ValueError:
                _json_response(self, 400, {"error": "invalid since"})
                return True
            items = MANAGER._voice_push.notification_feed_since(since_ts)
            _json_response(self, 200, {"ok": True, "items": items})
            return True

        if path == "/api/audio/live.m3u8":
            if not _require_auth(self):
                self._unauthorized()
                return True
            body = MANAGER._voice_push.playlist_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.apple.mpegurl")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(body)
            return True

        if path.startswith("/api/audio/segments/"):
            if not _require_auth(self):
                self._unauthorized()
                return True
            segment_name = path.split("/api/audio/segments/", 1)[1]
            try:
                segment_path = MANAGER._voice_push.segment_path(segment_name)
            except FileNotFoundError:
                self.send_error(404)
                return True
            raw = segment_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp2t")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(raw)
            return True

        return False

    def _read_json_body(self, *, limit: int = 2 * 1024 * 1024, too_large_error: str | None = None) -> dict[str, Any]:
        try:
            body = _read_body(self, limit=limit)
        except RequestPayloadTooLargeError as e:
            if too_large_error:
                raise RequestPayloadTooLargeError(too_large_error) from e
            raise
        try:
            body_text = body.decode("utf-8")
        except UnicodeDecodeError as e:
            raise BadRequestError("request body must be utf-8") from e
        if not body_text.strip():
            raise BadRequestError("empty request body")
        try:
            obj = json.loads(body_text)
        except json.JSONDecodeError as e:
            raise BadRequestError("invalid json body") from e
        if not isinstance(obj, dict):
            raise BadRequestError("invalid json body (expected object)")
        return obj

    def _handle_voice_post(self, path: str) -> bool:
        if path == "/api/settings/voice":
            if not _require_auth(self):
                self._unauthorized()
                return True
            obj = self._read_json_body()
            try:
                payload = MANAGER._voice_push.set_settings(obj, preserve_blank_api_key=True, redact_response=True)
            except ValueError as e:
                _json_response(self, 400, {"error": str(e)})
                return True
            _json_response(self, 200, {"ok": True, **payload})
            return True

        if path == "/api/notifications/subscription":
            if not _require_auth(self):
                self._unauthorized()
                return True
            obj = self._read_json_body()
            try:
                payload = MANAGER._voice_push.upsert_subscription(
                    subscription=obj.get("subscription"),
                    user_agent=str(obj.get("user_agent") or ""),
                    device_label=str(obj.get("device_label") or ""),
                    device_class=str(obj.get("device_class") or ""),
                )
            except ValueError as e:
                _json_response(self, 400, {"error": str(e)})
                return True
            _json_response(self, 200, {"ok": True, **payload})
            return True

        if path == "/api/notifications/subscription/toggle":
            if not _require_auth(self):
                self._unauthorized()
                return True
            obj = self._read_json_body()
            endpoint = obj.get("endpoint")
            enabled = obj.get("enabled")
            if not isinstance(endpoint, str) or not endpoint.strip():
                _json_response(self, 400, {"error": "endpoint required"})
                return True
            if not isinstance(enabled, bool):
                _json_response(self, 400, {"error": "enabled must be a boolean"})
                return True
            try:
                payload = MANAGER._voice_push.toggle_subscription(endpoint=endpoint, enabled=enabled)
            except KeyError:
                _json_response(self, 404, {"error": "unknown subscription"})
                return True
            except ValueError as e:
                _json_response(self, 400, {"error": str(e)})
                return True
            _json_response(self, 200, {"ok": True, **payload})
            return True

        if path == "/api/audio/listener":
            if not _require_auth(self):
                self._unauthorized()
                return True
            obj = self._read_json_body()
            client_id = obj.get("client_id")
            enabled = obj.get("enabled")
            if not isinstance(client_id, str) or not client_id.strip():
                _json_response(self, 400, {"error": "client_id required"})
                return True
            if not isinstance(enabled, bool):
                _json_response(self, 400, {"error": "enabled must be a boolean"})
                return True
            payload = MANAGER._voice_push.listener_heartbeat(client_id=client_id, enabled=enabled)
            _json_response(self, 200, {"ok": True, **payload})
            return True

        return False

    def do_GET(self) -> None:
        try:
            parsed = self._parse_prefixed_request_path()
            if parsed is None:
                return
            u, path = parsed
            if self._handle_static_get(path):
                return

            if path == "/api/me":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                _json_response(self, 200, {"ok": True})
                return

            if self._handle_voice_get(path, u.query):
                return

            if path == "/api/sessions":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                t0 = time.perf_counter()
                sessions = MANAGER.list_sessions()
                recent_cwds = MANAGER.recent_cwds()
                new_session_defaults = _read_new_session_defaults()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                _record_metric("api_sessions_ms", dt_ms)
                _json_response_with_etag(
                    self,
                    {
                        "app_version": _static_asset_version(),
                        "sessions": sessions,
                        "recent_cwds": recent_cwds,
                        "new_session_defaults": new_session_defaults,
                        "tmux_available": _tmux_available(),
                        "tmux_session_name": TMUX_SESSION_NAME,
                    },
                )
                return

            if path == "/api/session_resume_candidates":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                qs = urllib.parse.parse_qs(u.query)
                cwd_raw = qs.get("cwd", [""])[0]
                try:
                    agent_backend = normalize_agent_backend(qs.get("agent_backend", [""])[0], default=DEFAULT_AGENT_BACKEND)
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                try:
                    cwd_path = _resolve_dir_target(str(cwd_raw), field_name="cwd")
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e), "field": "cwd"})
                    return
                info = _describe_session_cwd(cwd_path)
                rows = _list_resume_candidates_for_cwd(info["cwd"], agent_backend=agent_backend) if info["exists"] else []
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

            session_id = _match_session_route(path, "diagnostics")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
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
                    token_val = st_token if isinstance(st_token, dict) else (s.token if isinstance(s.token, dict) else None)
                model_provider = s.model_provider
                preferred_auth_method = s.preferred_auth_method
                model = s.model
                reasoning_effort = s.reasoning_effort
                service_tier = s.service_tier
                if (model_provider is None or model is None or reasoning_effort is None) and s.log_path is not None and s.log_path.exists():
                    log_provider, log_model, log_effort = _read_run_settings_from_log(s.log_path, agent_backend=s.agent_backend)
                    if model_provider is None:
                        model_provider = log_provider
                    if model is None:
                        model = log_model
                    if reasoning_effort is None:
                        reasoning_effort = log_effort
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
                broker_busy = bool(state.get("busy"))
                busy_val = broker_busy
                if s.log_path is not None and s.log_path.exists():
                    idle_val = MANAGER.idle_from_log(session_id)
                    busy_val = not bool(idle_val)
                _json_response(
                    self,
                    200,
                    {
                        "session_id": s.session_id,
                        "thread_id": s.thread_id,
                        "agent_backend": s.agent_backend,
                        "owned": bool(s.owned),
                        "transport": s.transport,
                        "cwd": s.cwd,
                        "start_ts": float(s.start_ts),
                        "updated_ts": float(s.last_chat_ts) if isinstance(s.last_chat_ts, (int, float)) else float(s.start_ts),
                        "log_path": (str(s.log_path) if s.log_path is not None else None),
                        "broker_pid": int(s.broker_pid),
                        "codex_pid": int(s.codex_pid),
                        "busy": bool(busy_val),
                        "broker_busy": broker_busy,
                        "queue_len": MANAGER._queue_len(session_id),
                        "token": token_val,
                        "model_provider": model_provider,
                        "preferred_auth_method": preferred_auth_method,
                        "provider_choice": _provider_choice_for_settings(
                            model_provider=model_provider,
                            preferred_auth_method=preferred_auth_method,
                        ),
                        "model": model,
                        "reasoning_effort": reasoning_effort,
                        "service_tier": service_tier,
                        "tmux_session": s.tmux_session,
                        "tmux_window": s.tmux_window,
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

            session_id = _match_session_route(path, "queue")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                try:
                    q = MANAGER.queue_list(session_id)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                except ValueError as e:
                    _json_response(self, 502, {"error": str(e)})
                    return
                _json_response(self, 200, {"ok": True, "items": q, "queue": [str(item.get("text") or "") for item in q]})
                return

            session_id = _match_session_route(path, "file", "read")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
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
                try:
                    p = _resolve_existing_session_file(base, rel)
                except FileNotFoundError as e:
                    _json_response(self, 404, {"error": str(e)})
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                try:
                    view = _read_client_file_view(p)
                except PermissionError as e:
                    _json_response(self, 403, {"error": str(e)})
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                try:
                    MANAGER.files_add(session_id, str(p))
                except KeyError:
                    pass
                if view.kind == "image":
                    _json_response(
                        self,
                        200,
                        {
                            "ok": True,
                            "kind": "image",
                            "content_type": view.content_type,
                            "path": str(p),
                            "rel": str(rel),
                            "size": int(view.size),
                            "image_url": f"/api/sessions/{session_id}/file/blob?path={urllib.parse.quote(rel)}",
                        },
                    )
                    return
                if view.kind == "pdf":
                    _json_response(
                        self,
                        200,
                        {
                            "ok": True,
                            "kind": "pdf",
                            "content_type": view.content_type,
                            "path": str(p),
                            "rel": str(rel),
                            "size": int(view.size),
                            "pdf_url": f"/api/sessions/{session_id}/file/blob?path={urllib.parse.quote(rel)}",
                        },
                    )
                    return
                if view.kind == "video":
                    _json_response(
                        self,
                        200,
                        _video_response_payload(
                            path_obj=p,
                            rel=str(rel),
                            size=int(view.size),
                            content_type=view.content_type,
                            video_url=f"/api/sessions/{session_id}/file/blob?path={urllib.parse.quote(rel)}",
                            preview_url=f"/api/sessions/{session_id}/file/video_preview?path={urllib.parse.quote(rel)}",
                        ),
                    )
                    return
                if view.kind == "download_only":
                    _json_response(
                        self,
                        200,
                        {
                            "ok": True,
                            "kind": "download_only",
                            "path": str(p),
                            "rel": str(rel),
                            "size": int(view.size),
                            "reason": view.blocked_reason,
                            "viewer_max_bytes": view.viewer_max_bytes,
                        },
                    )
                    return
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "kind": view.kind,
                        "path": str(p),
                        "rel": str(rel),
                        "size": int(view.size),
                        "text": view.text,
                        "editable": bool(view.editable),
                        "version": view.version,
                    },
                )
                return

            session_id = _match_session_route(path, "file", "search")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                MANAGER.refresh_session_meta(session_id)
                s = MANAGER.get_session(session_id)
                if not s:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                qs = urllib.parse.parse_qs(u.query)
                query_raw = qs.get("q")
                if not query_raw or not query_raw[0].strip():
                    _json_response(self, 400, {"error": "q required"})
                    return
                limit_raw = qs.get("limit", [str(FILE_SEARCH_LIMIT)])[0]
                try:
                    limit = int(str(limit_raw).strip() or str(FILE_SEARCH_LIMIT))
                except ValueError:
                    _json_response(self, 400, {"error": "limit must be an integer"})
                    return
                if limit < 1:
                    _json_response(self, 400, {"error": "limit must be >= 1"})
                    return
                base = Path(s.cwd).expanduser()
                if not base.is_absolute():
                    base = base.resolve()
                try:
                    result = _search_session_relative_files(base, query=query_raw[0], limit=limit)
                except FileNotFoundError as e:
                    _json_response(self, 404, {"error": str(e)})
                    return
                except PermissionError as e:
                    _json_response(self, 403, {"error": str(e)})
                    return
                except (RuntimeError, ValueError) as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "cwd": str(base),
                        "query": result["query"],
                        "mode": result["mode"],
                        "matches": result["matches"],
                        "scanned": result["scanned"],
                        "truncated": result["truncated"],
                    },
                )
                return

            session_id = _match_session_route(path, "file", "list")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                MANAGER.refresh_session_meta(session_id)
                s = MANAGER.get_session(session_id)
                if not s:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                base = Path(s.cwd).expanduser()
                if not base.is_absolute():
                    base = base.resolve()
                try:
                    files = _list_session_relative_files(base)
                except FileNotFoundError as e:
                    _json_response(self, 404, {"error": str(e)})
                    return
                except PermissionError as e:
                    _json_response(self, 403, {"error": str(e)})
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                _json_response(self, 200, {"ok": True, "cwd": str(base), "files": files})
                return

            session_id = _match_session_route(path, "file", "blob")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
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
                try:
                    p = _resolve_existing_session_file(base, rel)
                except FileNotFoundError as e:
                    _json_response(self, 404, {"error": str(e)})
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                with p.open("rb") as f:
                    prefix = f.read(4096)
                _kind, ctype = _file_kind(p, prefix)
                if _kind == "video":
                    _send_inline_file_response(self, p, ctype or "application/octet-stream")
                    return
                if _kind not in {"image", "pdf"} or ctype is None:
                    _json_response(self, 400, {"error": "file is not previewable inline"})
                    return
                _send_inline_file_response(self, p, ctype)
                return

            session_id = _match_session_route(path, "file", "video_preview")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
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
                try:
                    p = _resolve_existing_session_file(base, rel)
                except FileNotFoundError as e:
                    _json_response(self, 404, {"error": str(e)})
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                with p.open("rb") as f:
                    prefix = f.read(4096)
                _kind, _ctype = _file_kind(p, prefix)
                if _kind != "video":
                    _json_response(self, 400, {"error": "file is not a video"})
                    return
                try:
                    preview = _ensure_video_preview(p)
                except RuntimeError as e:
                    _json_response(self, 500, {"error": f"video preview failed: {e}"})
                    return
                _send_inline_file_response(self, preview, "video/mp4")
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
                try:
                    path_obj = _resolve_existing_absolute_file(path_q[0])
                except FileNotFoundError as e:
                    _json_response(self, 404, {"error": str(e)})
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                with path_obj.open("rb") as f:
                    prefix = f.read(4096)
                _kind, ctype = _file_kind(path_obj, prefix)
                if _kind == "video":
                    _send_inline_file_response(self, path_obj, ctype or "application/octet-stream")
                    return
                if _kind not in {"image", "pdf"} or ctype is None:
                    _json_response(self, 400, {"error": "file is not previewable inline"})
                    return
                _send_inline_file_response(self, path_obj, ctype)
                return

            if path == "/api/files/video_preview":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                qs = urllib.parse.parse_qs(u.query)
                path_q = qs.get("path")
                if not path_q or not path_q[0]:
                    _json_response(self, 400, {"error": "path required"})
                    return
                try:
                    path_obj = _resolve_existing_absolute_file(path_q[0])
                except FileNotFoundError as e:
                    _json_response(self, 404, {"error": str(e)})
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                with path_obj.open("rb") as f:
                    prefix = f.read(4096)
                _kind, _ctype = _file_kind(path_obj, prefix)
                if _kind != "video":
                    _json_response(self, 400, {"error": "file is not a video"})
                    return
                try:
                    preview = _ensure_video_preview(path_obj)
                except RuntimeError as e:
                    _json_response(self, 500, {"error": f"video preview failed: {e}"})
                    return
                _send_inline_file_response(self, preview, "video/mp4")
                return

            session_id = _match_session_route(path, "file", "download")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
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
                    size = _inspect_downloadable_file(p)
                except FileNotFoundError as e:
                    _json_response(self, 404, {"error": str(e)})
                    return
                except PermissionError as e:
                    _json_response(self, 403, {"error": str(e)})
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                _send_attachment_file_response(self, p, size=size, content_disposition=_download_disposition(p))
                return

            session_id = _match_session_route(path, "git", "changed_files")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
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

            session_id = _match_session_route(path, "git", "diff")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
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

            session_id = _match_session_route(path, "git", "file_versions")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
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

            session_id = _match_session_route(path, "messages", "export")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                MANAGER.refresh_session_meta(session_id)
                s = MANAGER.get_session(session_id)
                if not s:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                transcript = _message_transcript_identity(s)
                if s.log_path is None or (not s.log_path.exists()):
                    _json_response(self, 200, {**transcript, "events": [], "event_count": 0})
                    return
                try:
                    events = _read_chat_export_events(s.log_path)
                except ValueError as e:
                    _json_response(self, 413, {"error": str(e), "max_bytes": int(TRANSCRIPT_EXPORT_MAX_BYTES)})
                    return
                events = MANAGER._attach_notification_texts(events)
                _json_response(self, 200, {**transcript, "events": events, "event_count": len(events)})
                return

            session_id = _match_session_route(path, "messages", "search")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                MANAGER.refresh_session_meta(session_id)
                s = MANAGER.get_session(session_id)
                if not s:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                qs = urllib.parse.parse_qs(u.query)
                query = (qs.get("q") or [""])[0]
                match_limit, limit_error = _parse_bounded_query_int(qs, "limit", default=20, min_value=0, max_value=100)
                if limit_error is not None:
                    _json_response(self, 400, {"error": limit_error})
                    return
                transcript = _message_transcript_identity(s)
                if not isinstance(query, str) or not query.strip():
                    _json_response(self, 200, {**transcript, "query": "", "match_count": 0, "matches": []})
                    return
                if s.log_path is None or (not s.log_path.exists()):
                    _json_response(self, 200, {**transcript, "query": query.strip(), "match_count": 0, "matches": []})
                    return
                try:
                    events = _read_chat_export_events(s.log_path)
                except ValueError as e:
                    _json_response(self, 413, {"error": str(e), "max_bytes": int(TRANSCRIPT_EXPORT_MAX_BYTES)})
                    return
                events = MANAGER._attach_notification_texts(events)
                match_count, matches = _search_chat_events(events, query, limit=match_limit)
                _json_response(self, 200, {**transcript, "query": query.strip(), "match_count": match_count, "matches": matches})
                return

            session_id = _match_session_route(path, "messages", "tail")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                t0_total = time.perf_counter()
                MANAGER.refresh_session_meta(session_id)
                s = MANAGER.get_session(session_id)
                if not s:
                    launch_payload = _launch_attempt_transcript_for_session_id(session_id)
                    if launch_payload is not None:
                        _json_response(self, 200, launch_payload)
                        _record_metric("api_messages_init_ms", (time.perf_counter() - t0_total) * 1000.0)
                        return
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                qs = urllib.parse.parse_qs(u.query)
                limit, limit_error = _parse_bounded_query_int(qs, "limit", default=80, min_value=20, max_value=200)
                if limit_error is not None:
                    _json_response(self, 400, {"error": limit_error})
                    return
                if s.log_path is None or (not s.log_path.exists()):
                    _state, busy_val, queue_val, token_val = _message_runtime_snapshot(session_id, s)
                    transcript = _message_transcript_identity(s)
                    _json_response(
                        self,
                        200,
                        {
                            **transcript,
                            "live_cursor": None,
                            "history_cursor": None,
                            "events": [],
                            "has_older": False,
                            "busy": bool(busy_val),
                            "queue_len": int(queue_val),
                            "token": token_val,
                        },
                    )
                    _record_metric("api_messages_init_ms", (time.perf_counter() - t0_total) * 1000.0)
                    return
                events, before_byte, after_byte, has_older = _read_chat_tail_page(s.log_path, limit=limit)
                events = MANAGER._attach_notification_texts(events)
                events = _attach_history_cursors(events, session=s)
                live_cursor = _encode_message_cursor(kind="live", session=s, pos=after_byte)
                history_cursor = _encode_message_cursor(kind="history", session=s, pos=before_byte) if has_older and before_byte > 0 else None
                _state, busy_val, queue_val, token_val = _message_runtime_snapshot(session_id, s)
                transcript = _message_transcript_identity(s)
                _json_response(
                    self,
                    200,
                    {
                        **transcript,
                        "live_cursor": live_cursor,
                        "history_cursor": history_cursor,
                        "events": events,
                        "has_older": bool(has_older),
                        "busy": bool(busy_val),
                        "queue_len": int(queue_val),
                        "token": token_val,
                    },
                )
                _record_metric("api_messages_init_ms", (time.perf_counter() - t0_total) * 1000.0)
                return

            session_id = _match_session_route(path, "messages", "history")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                MANAGER.refresh_session_meta(session_id)
                s = MANAGER.get_session(session_id)
                if not s:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                qs = urllib.parse.parse_qs(u.query)
                cursor_q = qs.get("cursor")
                if cursor_q is None or not cursor_q or not cursor_q[0].strip():
                    _json_response(self, 400, {"error": "cursor required"})
                    return
                limit, limit_error = _parse_bounded_query_int(qs, "limit", default=60, min_value=20, max_value=200)
                if limit_error is not None:
                    _json_response(self, 400, {"error": limit_error})
                    return
                if s.log_path is None or (not s.log_path.exists()):
                    _state, busy_val, queue_val, token_val = _message_runtime_snapshot(session_id, s)
                    transcript = _message_transcript_identity(s)
                    _json_response(
                        self,
                        200,
                        {
                            **transcript,
                            "history_cursor": None,
                            "events": [],
                            "has_older": False,
                            "busy": bool(busy_val),
                            "queue_len": int(queue_val),
                            "token": token_val,
                        },
                    )
                    return
                try:
                    before_byte = _decode_message_cursor(cursor_q[0], kind="history", session=s)
                except MessageCursorError as e:
                    _json_response(self, 409, {"error": str(e)})
                    return
                events, next_before, has_older = _read_chat_history_page(s.log_path, before_byte=before_byte, limit=limit)
                events = MANAGER._attach_notification_texts(events)
                events = _attach_history_cursors(events, session=s)
                history_cursor = _encode_message_cursor(kind="history", session=s, pos=next_before) if has_older and next_before > 0 else None
                _state, busy_val, queue_val, token_val = _message_runtime_snapshot(session_id, s)
                transcript = _message_transcript_identity(s)
                _json_response(
                    self,
                    200,
                    {
                        **transcript,
                        "history_cursor": history_cursor,
                        "events": events,
                        "has_older": bool(has_older),
                        "busy": bool(busy_val),
                        "queue_len": int(queue_val),
                        "token": token_val,
                    },
                )
                return

            session_id = _match_session_route(path, "messages", "live")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                t0_total = time.perf_counter()
                t0_meta = time.perf_counter()
                MANAGER.refresh_session_meta(session_id)
                dt_meta_ms = (time.perf_counter() - t0_meta) * 1000.0
                s = MANAGER.get_session(session_id)
                if not s:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                qs = urllib.parse.parse_qs(u.query)
                cursor_q = qs.get("cursor")
                if cursor_q is None or not cursor_q or not cursor_q[0].strip():
                    _json_response(self, 400, {"error": "cursor required"})
                    return
                if s.log_path is None or (not s.log_path.exists()):
                    _state, busy_val, queue_val, token_val = _message_runtime_snapshot(session_id, s)
                    transcript = _message_transcript_identity(s)
                    _json_response(
                        self,
                        200,
                        {
                            **transcript,
                            "live_cursor": None,
                            "events": [],
                            "meta_delta": {"thinking": 0, "tool": 0, "system": 0},
                            "turn_start": False,
                            "turn_end": False,
                            "turn_aborted": False,
                            "diag": {"pending_log": True, "meta_refresh_ms": round(dt_meta_ms, 3)},
                            "busy": bool(busy_val),
                            "queue_len": int(queue_val),
                            "token": token_val,
                        },
                    )
                    _record_metric("api_messages_poll_ms", (time.perf_counter() - t0_total) * 1000.0)
                    return
                try:
                    after_byte = _decode_message_cursor(cursor_q[0], kind="live", session=s)
                except MessageCursorError as e:
                    _json_response(self, 409, {"error": str(e)})
                    return
                records, next_after = _read_jsonl_records_from_offset(s.log_path, after_byte)
                objs = [record.obj for record in records]
                events, meta_delta, flags, diag = _extract_chat_events(objs)
                token_update = _extract_token_update(objs)
                events = _extract_positioned_chat_events(records)
                if objs:
                    MANAGER.mark_log_delta(session_id, objs=objs, new_off=next_after)
                s2 = MANAGER.get_session(session_id)
                if token_update is not None and s2 is not None:
                    s2.token = token_update
                events = MANAGER._attach_notification_texts(events)
                events = _attach_history_cursors(events, session=s)
                live_cursor = _encode_message_cursor(kind="live", session=s, pos=next_after)
                t0_state = time.perf_counter()
                _state, busy_val, queue_val, token_val = _message_runtime_snapshot(session_id, s, token_update=token_update)
                diag["state_ms"] = round((time.perf_counter() - t0_state) * 1000.0, 3)
                diag["meta_refresh_ms"] = round(dt_meta_ms, 3)
                transcript = _message_transcript_identity(s)
                _json_response(
                    self,
                    200,
                    {
                        **transcript,
                        "live_cursor": live_cursor,
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
                _record_metric("api_messages_poll_ms", (time.perf_counter() - t0_total) * 1000.0)
                return

            session_id = _match_session_route(path, "tail")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                try:
                    tail = MANAGER.get_tail(session_id)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, {"tail": tail})
                return

            session_id = _match_session_route(path, "unattended")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                try:
                    cfg = MANAGER.unattended_get(session_id)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, {"ok": True, **cfg})
                return

            self.send_error(404)
        except Exception as e:
            _handle_route_exception(self, e)

    def do_POST(self) -> None:
        try:
            parsed = self._parse_prefixed_request_path()
            if parsed is None:
                return
            u, path = parsed

            if path == "/api/login":
                obj = self._read_json_body()
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

            if self._handle_voice_post(path):
                return

            if path == "/api/sessions":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                obj = self._read_json_body()
                try:
                    launch_req = _parse_new_session_launch_request(obj)
                except LaunchRequestValidationError as e:
                    payload: dict[str, Any] = {"error": str(e)}
                    if e.field:
                        payload["field"] = e.field
                    _json_response(self, 400, payload)
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                try:
                    res = MANAGER.spawn_web_session(
                        cwd=launch_req.cwd,
                        args=launch_req.args,
                        agent_backend=launch_req.agent_backend,
                        resume_session_id=launch_req.resume_session_id,
                        worktree_branch=launch_req.worktree_branch,
                        model_provider=launch_req.model_provider,
                        preferred_auth_method=launch_req.preferred_auth_method,
                        model=launch_req.model,
                        reasoning_effort=launch_req.reasoning_effort,
                        service_tier=launch_req.service_tier,
                        create_in_tmux=launch_req.create_in_tmux,
                    )
                except ValueError as e:
                    payload: dict[str, Any] = {"error": str(e)}
                    if str(e).startswith("cwd "):
                        payload["field"] = "cwd"
                    _json_response(self, 400, payload)
                    return
                except SessionLaunchError as e:
                    payload = {
                        "error": str(e),
                        "launch_attempt": e.record,
                        "launch_id": e.record.get("launch_id"),
                    }
                    _json_response(self, 500, payload)
                    return
                _json_response(self, 200, {"ok": True, **res})
                return

            if path == "/api/files/read":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                obj = self._read_json_body()
                path_raw = obj.get("path")
                if not isinstance(path_raw, str) or not path_raw.strip():
                    _json_response(self, 400, {"error": "path required"})
                    return
                session_id_raw = obj.get("session_id")
                session_id = session_id_raw if isinstance(session_id_raw, str) and session_id_raw else ""
                try:
                    path_obj = _resolve_client_file_path(session_id=session_id, raw_path=path_raw)
                    view = _read_client_file_view(path_obj)
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
                if view.kind == "image":
                    _json_response(
                        self,
                        200,
                        {
                            "ok": True,
                            "kind": "image",
                            "content_type": view.content_type,
                            "path": str(path_obj),
                            "size": int(view.size),
                            "image_url": f"/api/files/blob?path={urllib.parse.quote(str(path_obj))}",
                        },
                    )
                    return
                if view.kind == "pdf":
                    _json_response(
                        self,
                        200,
                        {
                            "ok": True,
                            "kind": "pdf",
                            "content_type": view.content_type,
                            "path": str(path_obj),
                            "size": int(view.size),
                            "pdf_url": f"/api/files/blob?path={urllib.parse.quote(str(path_obj))}",
                        },
                    )
                    return
                if view.kind == "video":
                    _json_response(
                        self,
                        200,
                        _video_response_payload(
                            path_obj=path_obj,
                            size=int(view.size),
                            content_type=view.content_type,
                            video_url=f"/api/files/blob?path={urllib.parse.quote(str(path_obj))}",
                            preview_url=f"/api/files/video_preview?path={urllib.parse.quote(str(path_obj))}",
                        ),
                    )
                    return
                if view.kind == "download_only":
                    _json_response(
                        self,
                        200,
                        {
                            "ok": True,
                            "kind": "download_only",
                            "path": str(path_obj),
                            "size": int(view.size),
                            "reason": view.blocked_reason,
                            "viewer_max_bytes": view.viewer_max_bytes,
                        },
                    )
                    return
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "kind": view.kind,
                        "path": str(path_obj),
                        "size": int(view.size),
                        "text": view.text,
                        "editable": bool(view.editable),
                        "version": view.version,
                    },
                )
                return

            if path == "/api/files/inspect":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                obj = self._read_json_body()
                path_raw = obj.get("path")
                if not isinstance(path_raw, str) or not path_raw.strip():
                    _json_response(self, 400, {"error": "path required"})
                    return
                session_id_raw = obj.get("session_id")
                session_id = session_id_raw if isinstance(session_id_raw, str) and session_id_raw else ""
                try:
                    path_obj = _resolve_client_file_path(session_id=session_id, raw_path=path_raw)
                    view = _read_client_file_view(path_obj)
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
                        "kind": view.kind,
                        "content_type": view.content_type,
                        "size": int(view.size),
                        "reason": view.blocked_reason,
                        "viewer_max_bytes": view.viewer_max_bytes,
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
                try:
                    path_obj = _resolve_existing_absolute_file(path_q[0])
                except FileNotFoundError as e:
                    _json_response(self, 404, {"error": str(e)})
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                with path_obj.open("rb") as f:
                    prefix = f.read(4096)
                _kind, ctype = _file_kind(path_obj, prefix)
                if _kind == "video":
                    _send_inline_file_response(self, path_obj, ctype or "application/octet-stream")
                    return
                if _kind not in {"image", "pdf"} or ctype is None:
                    _json_response(self, 400, {"error": "file is not previewable inline"})
                    return
                _send_inline_file_response(self, path_obj, ctype)
                return

            if path == "/api/files/video_preview":
                if not _require_auth(self):
                    self._unauthorized()
                    return
                qs = urllib.parse.parse_qs(u.query)
                path_q = qs.get("path")
                if not path_q or not path_q[0]:
                    _json_response(self, 400, {"error": "path required"})
                    return
                try:
                    path_obj = _resolve_existing_absolute_file(path_q[0])
                except FileNotFoundError as e:
                    _json_response(self, 404, {"error": str(e)})
                    return
                except ValueError as e:
                    _json_response(self, 400, {"error": str(e)})
                    return
                with path_obj.open("rb") as f:
                    prefix = f.read(4096)
                _kind, _ctype = _file_kind(path_obj, prefix)
                if _kind != "video":
                    _json_response(self, 400, {"error": "file is not a video"})
                    return
                try:
                    preview = _ensure_video_preview(path_obj)
                except RuntimeError as e:
                    _json_response(self, 500, {"error": f"video preview failed: {e}"})
                    return
                _send_inline_file_response(self, preview, "video/mp4")
                return

            session_id = _match_session_route(path, "file", "write")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                obj = self._read_json_body()
                path_raw = obj.get("path")
                if not isinstance(path_raw, str) or not path_raw.strip():
                    _json_response(self, 400, {"error": "path required"})
                    return
                text_raw = obj.get("text")
                if not isinstance(text_raw, str):
                    _json_response(self, 400, {"error": "text must be a string"})
                    return
                create_raw = obj.get("create")
                create = create_raw if isinstance(create_raw, bool) else False
                version_raw = obj.get("version")
                if not create and (not isinstance(version_raw, str) or not version_raw.strip()):
                    _json_response(self, 400, {"error": "version required"})
                    return
                MANAGER.refresh_session_meta(session_id)
                s = MANAGER.get_session(session_id)
                if not s:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                base = Path(s.cwd).expanduser()
                if not base.is_absolute():
                    base = base.resolve()
                if create:
                    try:
                        p = _resolve_under(base, path_raw)
                    except ValueError as e:
                        _json_response(self, 400, {"error": str(e)})
                        return
                    try:
                        size, next_version = _write_new_text_file_atomic(p, text=text_raw)
                    except FileExistsError:
                        payload: dict[str, Any] = {"error": "file already exists", "conflict": True, "path": str(p)}
                        if p.is_file():
                            try:
                                _current_text, _current_size, current_version = _read_text_file_for_write(p, max_bytes=FILE_READ_MAX_BYTES)
                                payload["version"] = current_version
                            except (FileNotFoundError, PermissionError, ValueError):
                                pass
                        _json_response(self, 409, payload)
                        return
                    except FileNotFoundError as e:
                        _json_response(self, 404, {"error": str(e)})
                        return
                    except PermissionError as e:
                        _json_response(self, 403, {"error": str(e)})
                        return
                    except ValueError as e:
                        _json_response(self, 400, {"error": str(e)})
                        return
                else:
                    p = _resolve_session_path(base, path_raw)
                    try:
                        _current_text, _current_size, current_version = _read_text_file_for_write(p, max_bytes=FILE_READ_MAX_BYTES)
                    except FileNotFoundError as e:
                        _json_response(self, 404, {"error": str(e)})
                        return
                    except PermissionError as e:
                        _json_response(self, 403, {"error": str(e)})
                        return
                    except ValueError as e:
                        _json_response(self, 400, {"error": str(e)})
                        return
                    if current_version != version_raw:
                        _json_response(
                            self,
                            409,
                            {"error": "file changed on disk", "conflict": True, "path": str(p), "version": current_version},
                        )
                        return
                    try:
                        size, next_version = _write_text_file_atomic(p, text=text_raw)
                    except FileNotFoundError as e:
                        _json_response(self, 404, {"error": str(e)})
                        return
                    except PermissionError as e:
                        _json_response(self, 403, {"error": str(e)})
                        return
                    except ValueError as e:
                        _json_response(self, 400, {"error": str(e)})
                        return
                try:
                    MANAGER.files_add(session_id, str(p))
                except KeyError:
                    pass
                _json_response(
                    self,
                    200,
                    {"ok": True, "path": str(p), "rel": str(path_raw), "size": int(size), "version": next_version, "editable": True},
                )
                return

            session_id = _match_session_route(path, "delete")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                _read_body(self)
                ok = MANAGER.delete_session(session_id)
                if not ok:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, {"ok": True})
                return

            session_id = _match_session_route(path, "edit")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                obj = self._read_json_body()
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

            session_id = _match_session_route(path, "rename")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                obj = self._read_json_body()
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

            session_id = _match_session_route(path, "pending_attachment", "clear")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                try:
                    res = MANAGER.clear_pending_attachment(session_id)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, res)
                return

            session_id = _match_session_route(path, "commit_unknown_send", "clear")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                try:
                    res = MANAGER.clear_commit_unknown_send(session_id)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                _json_response(self, 200, res)
                return

            session_id = _match_session_route(path, "send")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                obj = self._read_json_body()
                text = obj.get("text")
                if not isinstance(text, str) or not text.strip():
                    _json_response(self, 400, {"error": "text required"})
                    return
                allow_pending_attachment = bool(obj.get("allow_pending_attachment"))
                try:
                    res = MANAGER.send(session_id, text, allow_pending_attachment=allow_pending_attachment)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                except SessionNotReadyError as e:
                    _json_response(self, 409, {"error": str(e)})
                    return
                except SessionInjectionError as e:
                    _json_response(self, 502, {"error": str(e)})
                    return
                except SessionCommitUnknownError as e:
                    _json_response(self, 504, {"error": str(e), "commit_unknown": True})
                    return
                _json_response(self, 200, res)
                return

            session_id = _match_session_route(path, "enqueue")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                obj = self._read_json_body()
                text = obj.get("text")
                if not isinstance(text, str) or not text.strip():
                    _json_response(self, 400, {"error": "text required"})
                    return
                try:
                    res = MANAGER.enqueue(session_id, text)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                except SessionNotReadyError as e:
                    _json_response(self, 409, {"error": str(e)})
                    return
                except ValueError as e:
                    _json_response(self, 502, {"error": str(e)})
                    return
                _json_response(self, 200, res)
                return

            session_id = _match_session_route(path, "queue", "delete")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                obj = self._read_json_body()
                item_id = obj.get("id")
                if not isinstance(item_id, str) or not item_id.strip():
                    _json_response(self, 400, {"error": "id required"})
                    return
                allow_commit_unknown = bool(obj.get("allow_commit_unknown"))
                try:
                    res = MANAGER.queue_delete(session_id, item_id, allow_commit_unknown=allow_commit_unknown)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                except ValueError as e:
                    status = 409 if "commit" in str(e).lower() else 502
                    _json_response(self, status, {"error": str(e)})
                    return
                _json_response(self, 200, res)
                return

            session_id = _match_session_route(path, "queue", "update")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                obj = self._read_json_body()
                item_id = obj.get("id")
                text = obj.get("text")
                if not isinstance(item_id, str) or not item_id.strip():
                    _json_response(self, 400, {"error": "id required"})
                    return
                if not isinstance(text, str) or not text.strip():
                    _json_response(self, 400, {"error": "text required"})
                    return
                try:
                    res = MANAGER.queue_update(session_id, item_id, text)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                except ValueError as e:
                    status = 409 if "commit" in str(e).lower() else 502
                    _json_response(self, status, {"error": str(e)})
                    return
                _json_response(self, 200, res)
                return

            session_id = _match_session_route(path, "queue", "move")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                obj = self._read_json_body()
                item_id = obj.get("id")
                to_index = obj.get("to_index")
                if not isinstance(item_id, str) or not item_id.strip():
                    _json_response(self, 400, {"error": "id required"})
                    return
                if not isinstance(to_index, int):
                    _json_response(self, 400, {"error": "to_index required"})
                    return
                try:
                    res = MANAGER.queue_move(session_id, item_id, to_index)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                except ValueError as e:
                    status = 409 if "commit" in str(e).lower() else 502
                    _json_response(self, status, {"error": str(e)})
                    return
                _json_response(self, 200, res)
                return

            session_id = _match_session_route(path, "unattended")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                obj = self._read_json_body()
                enabled_raw = obj.get("enabled", None)
                request_raw = obj.get("request", None)
                cooldown_minutes_raw = obj.get("cooldown_minutes", None)
                remaining_injections_raw = obj.get("remaining_injections", None)
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
                cooldown_minutes: int | None
                if cooldown_minutes_raw is not None:
                    try:
                        cooldown_minutes = _clean_unattended_cooldown_minutes(cooldown_minutes_raw)
                    except ValueError as e:
                        _json_response(self, 400, {"error": str(e)})
                        return
                else:
                    cooldown_minutes = None
                remaining_injections: int | None
                if remaining_injections_raw is not None:
                    try:
                        remaining_injections = _clean_unattended_remaining_injections(remaining_injections_raw, allow_zero=True)
                    except ValueError as e:
                        _json_response(self, 400, {"error": str(e)})
                        return
                else:
                    remaining_injections = None

                cfg = MANAGER.unattended_set(
                    session_id,
                    enabled=enabled,
                    request=request,
                    cooldown_minutes=cooldown_minutes,
                    remaining_injections=remaining_injections,
                )
                _json_response(self, 200, {"ok": True, **cfg})
                return

            session_id = _match_session_route(path, "interrupt")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
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

            session_id = _match_session_route(path, "inject_file")
            if session_id is None:
                session_id = _match_session_route(path, "inject_image")
            if session_id is not None:
                if not _require_auth(self):
                    self._unauthorized()
                    return
                obj = self._read_json_body(
                    limit=ATTACH_UPLOAD_BODY_MAX_BYTES,
                    too_large_error=f"file too large (max {ATTACH_UPLOAD_MAX_BYTES} bytes)",
                )
                data_b64 = obj.get("data_b64")
                filename = obj.get("filename")
                attachment_index = obj.get("attachment_index")
                if not isinstance(filename, str) or (not filename.strip()):
                    _json_response(self, 400, {"error": "filename required"})
                    return
                if isinstance(attachment_index, bool) or not isinstance(attachment_index, int):
                    _json_response(self, 400, {"error": "attachment_index must be an integer"})
                    return
                if not isinstance(data_b64, str) or not data_b64:
                    _json_response(self, 400, {"error": "data_b64 required"})
                    return
                try:
                    ready_for_attachment = MANAGER.attachment_injection_ready(session_id)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                except SessionNotReadyError as e:
                    _json_response(self, 409, {"error": str(e)})
                    return
                except Exception:
                    _json_response(self, 409, {"error": "session state unavailable; wait before attaching a file"})
                    return
                if not ready_for_attachment:
                    _json_response(self, 409, {"error": "session is busy; wait before attaching a file"})
                    return
                try:
                    raw = base64.b64decode(data_b64.encode("ascii"), validate=True)
                except Exception:
                    _json_response(self, 400, {"error": "invalid base64"})
                    return
                try:
                    out_path = _stage_uploaded_file(session_id, filename, raw)
                except ValueError as e:
                    status = 413 if str(e).startswith("file too large") else 400
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
                    resp = MANAGER.inject_attachment_keys(session_id, seq)
                except KeyError:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                except SessionNotReadyError as e:
                    _json_response(self, 409, {"error": str(e)})
                    return
                except SessionInjectionError as e:
                    _json_response(self, 502, {"error": str(e)})
                    return
                except SessionCommitUnknownError as e:
                    _json_response(self, 504, {"error": str(e), "commit_unknown": True})
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
            _handle_route_exception(self, e)

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
