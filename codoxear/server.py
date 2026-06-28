#!/usr/bin/env python3
from __future__ import annotations

import base64
import copy
import errno
from contextlib import contextmanager
import hashlib
import hmac
import http.server
import json
import math
import os
import posixpath
import re
import signal
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Iterator, Mapping

from .agent_backend import get_agent_backend
from .agent_backend import normalize_agent_backend
from .auth import CookieAuthSettings
from .auth import load_or_create_hmac_secret as _load_or_create_hmac_secret_impl
from .auth import parse_cookies as _parse_cookies_impl
from .auth import require_auth as _require_auth_impl
from .auth import set_auth_cookie as _set_auth_cookie_impl
from .auth import sign_cookie as _sign_cookie_impl
from .auth import verify_cookie as _verify_cookie_impl
from . import rollout_log as _rollout_log
from .control_socket import ControlSocketCallError
from .control_socket import call_control_socket as _call_control_socket_impl
from .file_response import send_attachment_file_response as _send_attachment_file_response
from .file_response import send_inline_file_response as _send_inline_file_response
from .file_response import single_byte_range as _single_byte_range
from .file_search import FILE_LIST_IGNORED_DIRS
from .file_search import FILE_SEARCH_LIMIT
from .file_search import file_search_score as _file_search_score
from .file_search import search_session_relative_files as _search_session_relative_files_impl
from .file_text import FILE_READ_MAX_BYTES
from .file_text import read_text_file_strict as _read_text_file_strict
from .file_types import file_kind as _file_kind
from .file_upload import attachment_inject_text as _attachment_inject_text
from .file_upload import stage_uploaded_file as _stage_uploaded_file_impl
from . import git_ops as _git_ops
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
from .launch_ledger import launch_attempt_id as _launch_attempt_id_impl
from .launch_ledger import launch_attempt_row as _launch_attempt_row_impl
from .launch_ledger import launch_attempt_transcript_for_session_id as _launch_attempt_transcript_for_session_id_impl
from .launch_ledger import launch_attempt_transcript_payload as _launch_attempt_transcript_payload_impl
from .launch_ledger import launch_failure_tail as _launch_failure_tail_impl
from .launch_ledger import latest_launch_attempt as _latest_launch_attempt_impl
from .launch_ledger import record_launch_attempt as _record_launch_attempt_impl
from .launch_ledger import submitted_user_messages as _submitted_user_messages_impl
from .file_view import ClientFileView
from .file_view import download_disposition as _download_disposition
from .file_view import inspect_client_path as _inspect_client_path
from .file_view import inspect_downloadable_file as _inspect_downloadable_file
from .file_view import inspect_openable_file as _inspect_openable_file
from .file_view import inspect_path_metadata as _inspect_path_metadata
from .file_view import read_client_file_view as _read_client_file_view
from .file_view import read_text_or_image as _read_text_or_image
from .video_preview import ensure_video_preview as _ensure_video_preview_impl
from .video_preview import video_preview_path as _video_preview_path_impl
from .cc_log import cc_user_text as _cc_user_text
from .cc_log import read_cc_run_settings as _read_cc_run_settings
from .message_cursor import MessageCursorError
from .message_cursor import attach_history_cursors as _attach_history_cursors_impl
from .message_cursor import decode_message_cursor as _decode_message_cursor_impl
from .message_cursor import encode_message_cursor as _encode_message_cursor_impl
from .message_cursor import sign_message_cursor as _sign_message_cursor_impl
from .message_cursor import verify_message_cursor as _verify_message_cursor_impl
from .transcript_search import TRANSCRIPT_SEARCH_MAX_LINE_BYTES
from .transcript_search import casefold_match_span as _casefold_match_span
from .transcript_search import chat_event_matches_query as _chat_event_matches_query
from .transcript_search import clip_search_match_text as _clip_search_match_text
from .transcript_search import clip_search_text_around_query as _clip_search_text_around_query
from .transcript_search import iter_jsonl_records_forward_bounded as _iter_jsonl_records_forward_bounded
from .transcript_search import iter_positioned_chat_events_forward as _iter_positioned_chat_events_forward
from .transcript_search import search_chat_events as _search_chat_events
from .transcript_search import search_chat_log_bounded as _search_chat_log_bounded
from .pi_log import pi_user_text as _pi_user_text
from .pi_log import read_pi_run_settings as _read_pi_run_settings
from .queue_store import QueueStore
from .queue_store import coerce_queue_item as _queue_store_coerce_item
from .queue_sweep import QueueSweepCoordinator
from .session_queue import SessionQueueCoordinator
from .session_attachment import SessionAttachmentCoordinator
from .session_cleanup import SessionCleanupCoordinator
from .session_discovery import DiscoveryDeps
from .session_discovery import DiscoveryRegistration
from .session_discovery import DiscoveryResult
from .session_discovery import discover_sessions as _discover_sessions
from .session_discovery_registry import SessionDiscoveryRegistryCoordinator
from .session_files import SessionFilesCoordinator
from .session_launcher import drain_stream as _drain_stream_impl
from .session_launcher import wait_for_spawned_broker_meta as _wait_for_spawned_broker_meta_impl
from .session_launcher import wait_or_raise as _wait_or_raise_impl
from .session_control import SessionControlCoordinator
from .session_lifecycle import SessionLifecycleCoordinator
from .session_log_runtime import SessionLogRuntimeCoordinator
from .session_list import SessionListCoordinator
from .session_refresh import SessionRefreshCoordinator
from .session_readiness import SessionReadinessCoordinator
from .session_recent_cwd import SessionRecentCwdCoordinator
from .session_listing import clip01 as _listing_clip01
from .session_listing import priority_from_elapsed_seconds as _listing_priority_from_elapsed_seconds
from .session_listing import sidebar_priority_elapsed_seconds as _listing_sidebar_priority_elapsed_seconds
from .session_listing import sidebar_time_priority_from_elapsed_seconds as _listing_sidebar_time_priority_from_elapsed_seconds
from .session_model import Session
from .session_pending_state import SessionPendingStateCoordinator
from .session_prune import SessionPruneCoordinator
from .session_send import PrelogUserMessageRecorder
from .session_send import SessionSendCoordinator
from .session_runtime import ListingRuntimeProbes
from .session_runtime import clear_session_confirmed_send_boundary as _clear_session_confirmed_send_boundary
from .session_runtime import consume_session_confirmed_send_boundary as _consume_session_confirmed_send_boundary
from .session_runtime import log_path_size_or_none as _log_path_size_or_none
from .session_runtime import broker_allows_interrupted_idle_override as _runtime_broker_allows_interrupted_idle_override
from .session_runtime import broker_busy_queue as _runtime_broker_busy_queue
from .session_runtime import broker_interrupted_idle as _runtime_broker_interrupted_idle
from .session_runtime import broker_runtime_state as _runtime_broker_state
from .session_runtime import resolve_runtime_status as _resolve_runtime_status
from .session_runtime import select_runtime_token as _select_runtime_token
from .session_store import SessionStore
from .session_store import SessionStorePaths
from .session_web_launch import SessionWebLaunchCoordinator
from .server_handler import make_server_handler
from .server_http import BadRequestError
from .server_http import RequestPayloadTooLargeError
from .server_http import handle_route_exception as _handle_route_exception_impl
from .server_http import if_none_match_contains as _if_none_match_contains_impl
from .server_http import is_client_disconnect as _is_client_disconnect_impl
from .server_http import json_response as _json_response_impl
from .server_http import json_response_with_etag as _json_response_with_etag_impl
from .server_http import read_body as _read_body_impl
from .server_main import ThreadingHTTPServer
from .server_main import ThreadingHTTPServerV6
from .server_main import run_main as _run_server_main
from .server_route_deps import ServerRouteDepsFactory
from .session_ui_state import SessionUiStateCoordinator
from .session_unattended_config import SessionUnattendedConfigCoordinator
from .static_routes import CONTENT_SECURITY_POLICY
from .static_routes import FRONTEND_ASSET_FILES
from .static_routes import STATIC_ASSET_VERSION_FILES
from .static_routes import STATIC_ASSET_VERSION_PLACEHOLDER
from .static_routes import STATIC_ATTACH_MAX_BYTES_PLACEHOLDER
from .static_routes import STATIC_DIR
from .static_routes import TOP_LEVEL_STATIC_ASSETS
from .static_routes import read_static_bytes as _read_static_bytes_impl
from .static_routes import static_asset_version as _static_asset_version
from .static_routes import static_cache_control_headers as _static_cache_control_headers_impl
from .tmux_runtime import tmux_pane_snapshot as _tmux_pane_snapshot_impl
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
from .sidecar_metadata import log_invalid as _log_invalid_sidecar_metadata
from .util import pid_alive as _pid_alive
from .util import process_group_alive as _process_group_alive
from .util import proc_find_open_rollout_log as _proc_find_open_rollout_log
from .util import read_launch_attempts as _read_launch_attempts
from .util import load_json_file as _load_json_file
from .util import redact_launch_failure_text as _redact_launch_failure_text
from .util import redacted_launch_attempt_persist_record as _redacted_launch_attempt_persist_record
from .util import redacted_launch_attempt_response_record as _redacted_launch_attempt_record
from .util import read_jsonl_from_offset as _read_jsonl_from_offset_impl
from .util import read_session_meta_payload as _read_session_meta_payload_impl
from .util import session_id_from_rollout_path as _session_id_from_rollout_path
from .util import subagent_parent_thread_id as _subagent_parent_thread_id
from .unattended import UnattendedStore
from .unattended import clean_unattended_cooldown_minutes as _clean_unattended_cooldown_minutes_impl
from .unattended import clean_unattended_remaining_injections as _clean_unattended_remaining_injections_impl
from .unattended import render_unattended_prompt as _render_unattended_prompt_impl
from .unattended_sweep import UnattendedSweepCoordinator
from .voice_push import VoicePushCoordinator
from .voice_runtime import VoiceRuntimeCoordinator


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
TMUX_AVAILABLE_TTL_SECONDS = float(os.environ.get("CODEX_WEB_TMUX_AVAILABLE_TTL_SECONDS", "30.0"))

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
COMMIT_UNKNOWN_ORPHAN_PRUNE_SECONDS = float(os.environ.get("CODEX_WEB_COMMIT_UNKNOWN_ORPHAN_PRUNE_SECONDS", str(7 * 24 * 3600)))
SIDEBAR_PRIORITY_HALF_LIFE_SECONDS = 8.0 * 3600.0
SIDEBAR_PRIORITY_BUCKET_SECONDS = float(os.environ.get("CODEX_WEB_SIDEBAR_PRIORITY_BUCKET_SECONDS", "10.0"))
RECENT_CWD_MAX = int(os.environ.get("CODEX_WEB_RECENT_CWD_MAX", "256"))
STATIC_CACHE_ENABLED = str(os.environ.get("CODEX_WEB_STATIC_CACHE") or "").strip() == "1"
TRANSCRIPT_EXPORT_MAX_BYTES = int(os.environ.get("CODEX_WEB_TRANSCRIPT_EXPORT_MAX_BYTES", str(50 * 1024 * 1024)))


def _static_cache_control_headers(*, enabled: bool = STATIC_CACHE_ENABLED) -> dict[str, str]:
    return _static_cache_control_headers_impl(enabled=enabled)


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
_FILE_WRITE_LOCKS_LOCK = threading.Lock()
_FILE_WRITE_LOCKS: dict[str, tuple[threading.Lock, int]] = {}
_TMUX_AVAILABLE_CACHE: tuple[float, bool] | None = None
_TMUX_AVAILABLE_CACHE_LOCK = threading.Lock()
_LAUNCH_DEFAULTS_CACHE: tuple[tuple[tuple[str, bool, int | None, int | None], ...], dict[str, Any]] | None = None
_LAUNCH_DEFAULTS_CACHE_LOCK = threading.Lock()


def _path_signature(path: Path) -> tuple[str, bool, int | None, int | None]:
    try:
        st = path.stat()
    except FileNotFoundError:
        return (str(path), False, None, None)
    except OSError:
        return (str(path), False, None, None)
    return (str(path), True, int(st.st_mtime_ns), int(st.st_size))


@contextmanager
def _file_write_lock(path: Path) -> Iterator[None]:
    key = str(path)
    with _FILE_WRITE_LOCKS_LOCK:
        entry = _FILE_WRITE_LOCKS.get(key)
        if entry is None:
            lock = threading.Lock()
            refcount = 0
        else:
            lock, refcount = entry
        _FILE_WRITE_LOCKS[key] = (lock, refcount + 1)
    try:
        with lock:
            yield
    finally:
        with _FILE_WRITE_LOCKS_LOCK:
            entry = _FILE_WRITE_LOCKS.get(key)
            if entry is not None and entry[0] is lock:
                refcount = entry[1] - 1
                if refcount <= 0:
                    _FILE_WRITE_LOCKS.pop(key, None)
                else:
                    _FILE_WRITE_LOCKS[key] = (lock, refcount)


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
    return _wait_or_raise_impl(proc, label=label, timeout_s=timeout_s)


def _drain_stream(f: Any) -> None:
    return _drain_stream_impl(f)


def _tmux_available() -> bool:
    global _TMUX_AVAILABLE_CACHE
    now = time.time()
    ttl = max(0.0, float(TMUX_AVAILABLE_TTL_SECONDS))
    with _TMUX_AVAILABLE_CACHE_LOCK:
        if _TMUX_AVAILABLE_CACHE is not None:
            cached_at, cached = _TMUX_AVAILABLE_CACHE
            if ttl > 0 and (now - cached_at) < ttl:
                return bool(cached)
    available = shutil.which("tmux") is not None
    with _TMUX_AVAILABLE_CACHE_LOCK:
        _TMUX_AVAILABLE_CACHE = (now, bool(available))
    return bool(available)


def _wait_for_spawned_broker_meta(spawn_nonce: str, *, timeout_s: float = TMUX_META_WAIT_SECONDS) -> dict[str, Any]:
    return _wait_for_spawned_broker_meta_impl(spawn_nonce, sock_dir=SOCK_DIR, timeout_s=timeout_s)


def _tmux_pane_snapshot(tmux_bin: str, *, pane_id: str | None = None, window: str | None = None) -> dict[str, Any]:
    return _tmux_pane_snapshot_impl(
        tmux_bin,
        tmux_session_name=TMUX_SESSION_NAME,
        pane_id=pane_id,
        window=window,
        run=subprocess.run,
    )


def _record_launch_attempt(record: dict[str, Any]) -> dict[str, Any]:
    return _record_launch_attempt_impl(record, path=LAUNCH_ATTEMPTS_PATH, stderr=sys.stderr)


def _launch_attempt_id(record: dict[str, Any]) -> str:
    return _launch_attempt_id_impl(record)


def _latest_launch_attempt(launch_id: str) -> dict[str, Any] | None:
    return _latest_launch_attempt_impl(launch_id, path=LAUNCH_ATTEMPTS_PATH)


def _submitted_user_messages(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    return _submitted_user_messages_impl(record)


def _launch_failure_tail(record: dict[str, Any]) -> str:
    return _launch_failure_tail_impl(record)


def _launch_attempt_transcript_payload(record: dict[str, Any]) -> dict[str, Any]:
    return _launch_attempt_transcript_payload_impl(record)


def _launch_attempt_transcript_for_session_id(session_id: str) -> dict[str, Any] | None:
    return _launch_attempt_transcript_for_session_id_impl(
        session_id,
        path=LAUNCH_ATTEMPTS_PATH,
        default_agent_backend=DEFAULT_AGENT_BACKEND,
        unattended_default_idle_minutes=UNATTENDED_DEFAULT_IDLE_MINUTES,
        unattended_default_max_injections=UNATTENDED_DEFAULT_MAX_INJECTIONS,
    )


def _launch_attempt_row(record: dict[str, Any]) -> dict[str, Any] | None:
    return _launch_attempt_row_impl(
        record,
        default_agent_backend=DEFAULT_AGENT_BACKEND,
        unattended_default_idle_minutes=UNATTENDED_DEFAULT_IDLE_MINUTES,
        unattended_default_max_injections=UNATTENDED_DEFAULT_MAX_INJECTIONS,
    )


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


def _video_preview_path(path: Path) -> Path:
    return _video_preview_path_impl(path, preview_dir=VIDEO_PREVIEW_DIR)


def _ensure_video_preview(path: Path) -> Path:
    return _ensure_video_preview_impl(path, preview_dir=VIDEO_PREVIEW_DIR)



def _is_client_disconnect(exc: BaseException) -> bool:
    return _is_client_disconnect_impl(exc)


def _handle_route_exception(handler: http.server.BaseHTTPRequestHandler, exc: BaseException) -> None:
    return _handle_route_exception_impl(handler, exc, json_response=_json_response)


def _json_response(handler: http.server.BaseHTTPRequestHandler, status: int, obj: Any) -> None:
    return _json_response_impl(handler, status, obj, set_auth_cookie=_set_auth_cookie)


def _if_none_match_contains(header_value: str | None, etag: str) -> bool:
    return _if_none_match_contains_impl(header_value, etag)


def _json_response_with_etag(handler: http.server.BaseHTTPRequestHandler, obj: Any) -> None:
    return _json_response_with_etag_impl(handler, obj, sha256_hex=_sha256_hex, set_auth_cookie=_set_auth_cookie)


def _read_body(handler: http.server.BaseHTTPRequestHandler, limit: int = 2 * 1024 * 1024) -> bytes:
    return _read_body_impl(handler, limit=limit)


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
        safe = _redacted_launch_attempt_record(record)
        msg = str(safe.get("error") or safe.get("message") or "session launch failed")
        super().__init__(msg)
        self.record = safe


class SessionNotReadyError(RuntimeError):
    pass


class SessionInjectionError(RuntimeError):
    pass


class SessionCommitUnknownError(RuntimeError):
    pass


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
    try:
        resolved.relative_to(resolved_base)
    except ValueError as e:
        raise ValueError("path escapes session cwd") from e
    return resolved


def _expanduser_path(path: Path) -> Path:
    try:
        return path.expanduser()
    except RuntimeError as e:
        raise ValueError(str(e)) from e


def _resolve_session_cwd(raw_cwd: str) -> Path:
    if not isinstance(raw_cwd, str) or not raw_cwd.strip() or "\x00" in raw_cwd:
        raise ValueError("invalid session cwd")
    try:
        cwd = _expanduser_path(Path(raw_cwd))
        if not cwd.is_absolute():
            cwd = cwd.resolve()
    except (OSError, ValueError) as e:
        raise ValueError(str(e)) from e
    return cwd


def _resolve_session_path(base: Path, raw_path: str) -> Path:
    if not isinstance(raw_path, str) or raw_path == "":
        raise ValueError("path required")
    if "\x00" in raw_path:
        raise ValueError("invalid path")
    p = Path(raw_path)
    if p.is_absolute():
        return _expanduser_path(p).resolve()
    resolved_base = _expanduser_path(base)
    if not resolved_base.is_absolute():
        resolved_base = resolved_base.resolve()
    return (resolved_base / p).resolve()


def _require_existing_file(path: Path) -> Path:
    try:
        st = path.stat()
    except FileNotFoundError:
        raise FileNotFoundError("file not found")
    except PermissionError:
        raise
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("path is not a file")
    return path


def _resolve_existing_session_file(base: Path, raw_path: str) -> Path:
    return _require_existing_file(_resolve_session_path(base, raw_path))


def _resolve_existing_absolute_file(raw_path: str) -> Path:
    return _require_existing_file(_expanduser_path(Path(raw_path)).resolve())


def _resolve_git_path(cwd: Path, raw_path: str) -> tuple[Path, Path, str]:
    return _git_ops.resolve_git_path(cwd, raw_path, run_git_func=_run_git, timeout_s=GIT_DIFF_TIMEOUT_SECONDS)

def _git_error_is_missing_head(message: str) -> bool:
    return _git_ops.git_error_is_missing_head(message)

def _git_head_blob_oid(cwd: Path, rel: str) -> str | None:
    return _git_ops.git_head_blob_oid(cwd, rel, run_git_func=_run_git, timeout_s=GIT_DIFF_TIMEOUT_SECONDS)

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
        candidate = _expanduser_path(Path(raw)).resolve()
        if candidate.name != name:
            continue
        if match is None:
            match = candidate
            continue
        if candidate != match:
            return None
    return match


def _list_session_relative_files(base: Path) -> list[str]:
    root = _expanduser_path(base)
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


def _run_git(cwd: Path, args: list[str], *, timeout_s: float, max_bytes: int, literal_pathspecs: bool = False) -> str:
    return _git_ops.run_git(cwd, args, timeout_s=timeout_s, max_bytes=max_bytes, literal_pathspecs=literal_pathspecs)

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
    return _git_ops.clean_worktree_branch(raw)

def _require_git_repo(cwd: Path) -> None:
    _git_ops.require_git_repo(cwd, run_git_func=_run_git, timeout_s=GIT_DIFF_TIMEOUT_SECONDS)

def _git_repo_root(cwd: Path) -> Path | None:
    return _git_ops.git_repo_root(cwd, run_git_func=_run_git, timeout_s=GIT_DIFF_TIMEOUT_SECONDS)

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
    return _git_ops.worktree_path_slug(branch)

def _default_worktree_path(source_cwd: Path, branch: str) -> Path:
    return _git_ops.default_worktree_path(source_cwd, branch)

def _create_git_worktree(source_cwd: Path, worktree_branch: str) -> Path:
    return _git_ops.create_git_worktree(source_cwd, worktree_branch, git_repo_root_func=_git_repo_root, timeout_s=GIT_WORKTREE_TIMEOUT_SECONDS)

def _split_git_nul_paths(text: str) -> list[str]:
    return _git_ops.split_git_nul_paths(text)

def _parse_git_numstat(text: str) -> dict[str, dict[str, int | None]]:
    return _git_ops.parse_git_numstat(text)

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
    return _listing_clip01(v)


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
    return _listing_priority_from_elapsed_seconds(elapsed_s, half_life_seconds=SIDEBAR_PRIORITY_HALF_LIFE_SECONDS)


def _sidebar_priority_elapsed_seconds(elapsed_s: float) -> float:
    return _listing_sidebar_priority_elapsed_seconds(elapsed_s, bucket_seconds=SIDEBAR_PRIORITY_BUCKET_SECONDS)


def _sidebar_time_priority_from_elapsed_seconds(elapsed_s: float) -> float:
    return _listing_sidebar_time_priority_from_elapsed_seconds(
        elapsed_s,
        half_life_seconds=SIDEBAR_PRIORITY_HALF_LIFE_SECONDS,
        bucket_seconds=SIDEBAR_PRIORITY_BUCKET_SECONDS,
    )


def _current_git_branch(cwd: Path) -> str | None:
    return _git_ops.current_git_branch(cwd, run_git_func=_run_git, timeout_s=GIT_DIFF_TIMEOUT_SECONDS)

def _path_resolves_inside(path_obj: Path, root: Path) -> bool:
    try:
        path_obj.resolve().relative_to(root)
        return True
    except (OSError, ValueError):
        return False


def _symlink_payload_view(path_obj: Path) -> ClientFileView:
    raw = os.readlink(path_obj).encode("utf-8", errors="surrogateescape")
    text = raw.decode("utf-8", errors="replace")
    return ClientFileView(
        kind="text",
        size=len(raw),
        text=text,
        editable=False,
        version=hashlib.sha256(raw).hexdigest(),
    )


def _resolve_git_client_file_view(*, session_id: str, raw_path: str) -> tuple[Path, str, ClientFileView]:
    if not session_id:
        raise ValueError("session_id required for git path")
    MANAGER.refresh_session_meta(session_id)
    s = MANAGER.get_session(session_id)
    if s is None:
        raise FileNotFoundError("unknown session")
    cwd = _resolve_session_cwd(s.cwd)
    path_obj, repo_root, rel = _resolve_git_path(cwd, raw_path)
    if _path_resolves_inside(path_obj.parent, repo_root) and path_obj.is_symlink():
        return path_obj, rel, _symlink_payload_view(path_obj)
    try:
        real = path_obj.resolve()
        real.relative_to(repo_root)
    except (OSError, ValueError) as e:
        raise FileNotFoundError("file not found") from e
    return real, rel, _read_client_file_view(real)


def _resolve_git_existing_regular_file(*, session_id: str, raw_path: str) -> tuple[Path, str]:
    if not session_id:
        raise ValueError("session_id required for git path")
    MANAGER.refresh_session_meta(session_id)
    s = MANAGER.get_session(session_id)
    if s is None:
        raise FileNotFoundError("unknown session")
    cwd = _resolve_session_cwd(s.cwd)
    path_obj, repo_root, rel = _resolve_git_path(cwd, raw_path)
    try:
        real = path_obj.resolve()
        real.relative_to(repo_root)
    except (OSError, ValueError) as e:
        raise FileNotFoundError("file not found") from e
    return _require_existing_file(real), rel


def _resolve_client_file_path(*, session_id: str, raw_path: str) -> Path:
    s: Session | None = None
    if session_id:
        MANAGER.refresh_session_meta(session_id)
        s = MANAGER.get_session(session_id)
        if s is None:
            raise FileNotFoundError("unknown session")
    path_obj = _expanduser_path(Path(raw_path))
    if not path_obj.is_absolute():
        if s is not None:
            base = _resolve_session_cwd(s.cwd)
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


def _launch_defaults_signature(paths: LaunchConfigPaths) -> tuple[tuple[str, bool, int | None, int | None], ...]:
    return tuple(_path_signature(path) for path in vars(paths).values() if isinstance(path, Path))


def _read_new_session_defaults() -> dict[str, Any]:
    global _LAUNCH_DEFAULTS_CACHE
    paths = _launch_config_paths()
    signature = _launch_defaults_signature(paths)
    with _LAUNCH_DEFAULTS_CACHE_LOCK:
        if _LAUNCH_DEFAULTS_CACHE is not None and _LAUNCH_DEFAULTS_CACHE[0] == signature:
            return copy.deepcopy(_LAUNCH_DEFAULTS_CACHE[1])
    defaults = _launch_read_new_session_defaults(paths, default_agent_backend=DEFAULT_AGENT_BACKEND)
    with _LAUNCH_DEFAULTS_CACHE_LOCK:
        _LAUNCH_DEFAULTS_CACHE = (signature, copy.deepcopy(defaults))
    return defaults


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
    *,
    initial_cc_pending_tool_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, bool], dict[str, Any]]:
    return _rollout_log._extract_chat_events(objs, initial_cc_pending_tool_ids=initial_cc_pending_tool_ids)


def _extract_delivery_messages(objs: list[dict[str, Any]], *, initial_cc_pending_tool_ids: set[str] | None = None) -> list[Any]:
    return _rollout_log._extract_delivery_messages(objs, initial_cc_pending_tool_ids=initial_cc_pending_tool_ids)


def _read_jsonl_records_from_offset(
    path: Path,
    offset: int,
    *,
    max_bytes: int = 2 * 1024 * 1024,
) -> tuple[list[Any], int]:
    return _rollout_log._read_jsonl_records_from_offset(path, offset, max_bytes=max_bytes)


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
    final_assistant_only: bool = False,
) -> tuple[str, float] | None:
    return _rollout_log._last_chat_role_ts_from_tail(path, max_scan_bytes=max_scan_bytes, final_assistant_only=final_assistant_only)


def _broker_busy_queue_from_state(state: dict[str, Any]) -> tuple[bool, int]:
    return _runtime_broker_busy_queue(state)


def _broker_interrupted_idle_from_state(state: dict[str, Any]) -> bool:
    return _runtime_broker_interrupted_idle(state)


def _broker_allows_interrupted_idle_override(state: dict[str, Any]) -> bool:
    return _runtime_broker_allows_interrupted_idle_override(state)


def _broker_tail_has_session_detach_marker(agent_backend: str, tail: Any) -> bool:
    if agent_backend != "codex" or not isinstance(tail, str):
        return False
    return "To continue this session, run " in tail


class SessionManager:
    @property
    def _unattended(self) -> dict[str, dict[str, Any]]:
        return self._session_store_for_manager().unattended

    @_unattended.setter
    def _unattended(self, value: dict[str, dict[str, Any]]) -> None:
        self._session_store_for_manager().unattended = value

    @property
    def _aliases(self) -> dict[str, str]:
        return self._session_store_for_manager().aliases

    @_aliases.setter
    def _aliases(self, value: dict[str, str]) -> None:
        self._session_store_for_manager().aliases = value

    @property
    def _sidebar_meta(self) -> dict[str, dict[str, Any]]:
        return self._session_store_for_manager().sidebar_meta

    @_sidebar_meta.setter
    def _sidebar_meta(self, value: dict[str, dict[str, Any]]) -> None:
        self._session_store_for_manager().sidebar_meta = value

    @property
    def _hidden_sessions(self) -> set[str]:
        return self._session_store_for_manager().hidden_sessions

    @_hidden_sessions.setter
    def _hidden_sessions(self, value: set[str]) -> None:
        self._session_store_for_manager().hidden_sessions = value

    @property
    def _files(self) -> dict[str, list[str]]:
        return self._session_store_for_manager().files

    @_files.setter
    def _files(self, value: dict[str, list[str]]) -> None:
        self._session_store_for_manager().files = value

    @property
    def _queues(self) -> dict[str, list[dict[str, Any]]]:
        return self._session_store_for_manager().queues

    @_queues.setter
    def _queues(self, value: dict[str, list[dict[str, Any]]]) -> None:
        self._session_store_for_manager().queues = value

    @property
    def _pending_attachment_ids(self) -> set[str]:
        return self._session_store_for_manager().pending_attachment_ids

    @_pending_attachment_ids.setter
    def _pending_attachment_ids(self, value: set[str]) -> None:
        self._session_store_for_manager().pending_attachment_ids = value

    @property
    def _commit_unknown_sends(self) -> dict[str, dict[str, Any]]:
        return self._session_store_for_manager().commit_unknown_sends

    @_commit_unknown_sends.setter
    def _commit_unknown_sends(self, value: dict[str, dict[str, Any]]) -> None:
        self._session_store_for_manager().commit_unknown_sends = value

    @property
    def _recent_cwds(self) -> dict[str, float]:
        return self._session_store_for_manager().recent_cwds

    @_recent_cwds.setter
    def _recent_cwds(self, value: dict[str, float]) -> None:
        self._session_store_for_manager().recent_cwds = value

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._stop = threading.Event()
        self._last_discover_ts = 0.0
        self._store = SessionStore(
            paths=SessionStorePaths(
                aliases=ALIAS_PATH,
                sidebar_meta=SIDEBAR_META_PATH,
                hidden_sessions=HIDDEN_SESSIONS_PATH,
                files=FILE_HISTORY_PATH,
                queues=QUEUE_PATH,
                pending_attachments=PENDING_ATTACHMENTS_PATH,
                commit_unknown_sends=COMMIT_UNKNOWN_SENDS_PATH,
                recent_cwds=RECENT_CWD_PATH,
                unattended=UNATTENDED_PATH,
            ),
            file_history_max=FILE_HISTORY_MAX,
            recent_cwd_max=RECENT_CWD_MAX,
            unattended_default_idle_minutes=UNATTENDED_DEFAULT_IDLE_MINUTES,
            unattended_default_max_injections=UNATTENDED_DEFAULT_MAX_INJECTIONS,
            clean_alias=_clean_alias,
            clean_priority_offset=_clean_priority_offset,
            clean_snooze_until=_clean_snooze_until,
            clean_dependency_session_id=_clean_dependency_session_id,
            clean_recent_cwd=_clean_recent_cwd,
            clean_commit_unknown_send_record=self._clean_commit_unknown_send_record,
        )
        self._unattended: dict[str, dict[str, Any]] = {}
        self._aliases: dict[str, str] = {}
        self._sidebar_meta: dict[str, dict[str, Any]] = {}
        self._hidden_sessions: set[str] = set()
        self._files: dict[str, list[str]] = {}
        self._queues: dict[str, list[dict[str, Any]]] = {}
        self._pending_attachment_ids: set[str] = set()
        self._commit_unknown_sends: dict[str, dict[str, Any]] = {}
        self._input_locks: dict[str, threading.RLock] = {}
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

    def _session_store_for_manager(self) -> SessionStore:
        paths = SessionStorePaths(
            aliases=ALIAS_PATH,
            sidebar_meta=SIDEBAR_META_PATH,
            hidden_sessions=HIDDEN_SESSIONS_PATH,
            files=FILE_HISTORY_PATH,
            queues=QUEUE_PATH,
            pending_attachments=PENDING_ATTACHMENTS_PATH,
            commit_unknown_sends=COMMIT_UNKNOWN_SENDS_PATH,
            recent_cwds=RECENT_CWD_PATH,
            unattended=UNATTENDED_PATH,
        )
        existing = getattr(self, "_store", None)
        if isinstance(existing, SessionStore) and existing.paths == paths:
            return existing
        store = SessionStore(
            paths=paths,
            file_history_max=FILE_HISTORY_MAX,
            recent_cwd_max=RECENT_CWD_MAX,
            unattended_default_idle_minutes=UNATTENDED_DEFAULT_IDLE_MINUTES,
            unattended_default_max_injections=UNATTENDED_DEFAULT_MAX_INJECTIONS,
            clean_alias=_clean_alias,
            clean_priority_offset=_clean_priority_offset,
            clean_snooze_until=_clean_snooze_until,
            clean_dependency_session_id=_clean_dependency_session_id,
            clean_recent_cwd=_clean_recent_cwd,
            clean_commit_unknown_send_record=self._clean_commit_unknown_send_record,
        )
        if isinstance(existing, SessionStore):
            store.unattended = existing.unattended
            store.aliases = existing.aliases
            store.sidebar_meta = existing.sidebar_meta
            store.hidden_sessions = existing.hidden_sessions
            store.files = existing.files
            store.queues = existing.queues
            store.pending_attachment_ids = existing.pending_attachment_ids
            store.commit_unknown_sends = existing.commit_unknown_sends
            store.recent_cwds = existing.recent_cwds
        self._store = store
        return store

    def _load_unattended(self) -> None:
        cleaned = self._session_store_for_manager().load_unattended()
        with self._lock:
            self._unattended = cleaned

    def _save_unattended(self) -> None:
        with self._lock:
            obj = dict(self._unattended)
        self._session_store_for_manager().save_unattended(obj)

    def _load_aliases(self) -> None:
        cleaned = self._session_store_for_manager().load_aliases()
        with self._lock:
            self._aliases = cleaned

    def _save_aliases(self) -> None:
        with self._lock:
            obj = dict(self._aliases)
        self._session_store_for_manager().save_aliases(obj)

    def _load_sidebar_meta(self) -> None:
        cleaned = self._session_store_for_manager().load_sidebar_meta()
        with self._lock:
            self._sidebar_meta = cleaned

    def _save_sidebar_meta(self) -> None:
        with self._lock:
            obj = dict(self._sidebar_meta)
        self._session_store_for_manager().save_sidebar_meta(obj)

    def _load_hidden_sessions(self) -> None:
        cleaned = self._session_store_for_manager().load_hidden_sessions()
        with self._lock:
            self._hidden_sessions = cleaned

    def _save_hidden_sessions(self) -> None:
        with self._lock:
            obj = set(getattr(self, "_hidden_sessions", set()))
        self._session_store_for_manager().save_hidden_sessions(obj)

    def _hide_session(self, session_id: str) -> None:
        return self._ui_state_coordinator_for_manager().hide_session(session_id)

    def _unhide_session(self, session_id: str) -> None:
        return self._ui_state_coordinator_for_manager().unhide_session(session_id)

    def alias_set(self, session_id: str, name: str) -> str:
        return self._ui_state_coordinator_for_manager().alias_set(session_id, name)

    def alias_get(self, session_id: str) -> str:
        return self._ui_state_coordinator_for_manager().alias_get(session_id)

    def alias_clear(self, session_id: str) -> None:
        return self._ui_state_coordinator_for_manager().alias_clear(session_id)

    def sidebar_meta_get(self, session_id: str) -> dict[str, Any]:
        return self._ui_state_coordinator_for_manager().sidebar_meta_get(session_id)

    def sidebar_meta_set(
        self,
        session_id: str,
        *,
        priority_offset: Any,
        snooze_until: Any,
        dependency_session_id: Any,
    ) -> dict[str, Any]:
        return self._ui_state_coordinator_for_manager().sidebar_meta_set(
            session_id,
            priority_offset=priority_offset,
            snooze_until=snooze_until,
            dependency_session_id=dependency_session_id,
        )

    def edit_session(
        self,
        session_id: str,
        *,
        name: str,
        priority_offset: Any,
        snooze_until: Any,
        dependency_session_id: Any,
    ) -> tuple[str, dict[str, Any]]:
        return self._ui_state_coordinator_for_manager().edit_session(
            session_id,
            name=name,
            priority_offset=priority_offset,
            snooze_until=snooze_until,
            dependency_session_id=dependency_session_id,
        )

    def _prune_stale_socket_without_metadata(self, session_id: str, sock: Path) -> None:
        return self._cleanup_coordinator_for_manager().prune_stale_socket_without_metadata(session_id, sock)

    def _clear_deleted_session_state(self, session_id: str, *, clear_recovery: bool = False) -> None:
        return self._cleanup_coordinator_for_manager().clear_deleted_session_state(session_id, clear_recovery=clear_recovery)

    def _load_files(self) -> None:
        cleaned = self._session_store_for_manager().load_files()
        with self._lock:
            self._files = cleaned

    def _save_files(self) -> None:
        with self._lock:
            obj = dict(self._files)
        self._session_store_for_manager().save_files(obj)

    def _queue_store_for_manager(self) -> QueueStore:
        return self._session_store_for_manager().queue_store

    def _queue_coordinator_for_manager(self) -> SessionQueueCoordinator:
        return SessionQueueCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            queues=lambda: self._queues,
            queue_store=self._queue_store_for_manager,
            commit_unknown_sends=lambda: self._commit_unknown_sends,
            save_queues=self._save_queues,
            input_lock_for_session=self._input_lock_for_session,
            remote_ready=lambda session_id, log_path: self._queue_remote_ready(session_id, log_path=log_path),
            send=self.send,
            not_ready_error=SessionNotReadyError,
            retryable_send_errors=(SessionNotReadyError, SessionInjectionError),
            commit_unknown_error=SessionCommitUnknownError,
            queue_idle_grace_seconds=QUEUE_IDLE_GRACE_SECONDS,
            now=time.time,
            recovery_items_locked=lambda session_id: self._queue_has_recovery_items_locked(session_id),
        )

    def _input_lock_for_session(self, session_id: str) -> threading.RLock:
        with self._lock:
            locks = getattr(self, "_input_locks", None)
            if not isinstance(locks, dict):
                self._input_locks = {}
                locks = self._input_locks
            lock = locks.get(session_id)
            if lock is None:
                lock = threading.RLock()
                locks[session_id] = lock
            return lock

    def _load_queues(self) -> None:
        cleaned = self._session_store_for_manager().load_queues()
        with self._lock:
            self._queues = cleaned

    def _save_queues(self) -> None:
        with self._lock:
            obj = dict(self._queues)
        self._session_store_for_manager().save_queues(obj)

    def _load_pending_attachments(self) -> None:
        cleaned = self._session_store_for_manager().load_pending_attachments()
        with self._lock:
            self._pending_attachment_ids = cleaned

    def _save_pending_attachments(self) -> None:
        with self._lock:
            ids = set(str(item) for item in getattr(self, "_pending_attachment_ids", set()) if str(item).strip())
        self._session_store_for_manager().save_pending_attachments(ids)

    def _set_pending_attachment(self, session_id: str, value: bool) -> None:
        return self._pending_state_coordinator_for_manager().set_pending_attachment(session_id, value)

    def clear_pending_attachment(self, session_id: str) -> dict[str, Any]:
        return self._pending_state_coordinator_for_manager().clear_pending_attachment(session_id)

    def _clean_commit_unknown_send_record(self, raw: Any) -> dict[str, Any] | None:
        return self._pending_state_coordinator_for_manager().clean_commit_unknown_send_record(raw)

    def _load_commit_unknown_sends(self) -> None:
        cleaned = self._session_store_for_manager().load_commit_unknown_sends()
        with self._lock:
            self._commit_unknown_sends = cleaned

    def _save_commit_unknown_sends(self) -> None:
        with self._lock:
            source = dict(getattr(self, "_commit_unknown_sends", {}))
        self._session_store_for_manager().save_commit_unknown_sends(source)

    def _set_commit_unknown_send(self, session_id: str, record: dict[str, Any] | None) -> None:
        return self._pending_state_coordinator_for_manager().set_commit_unknown_send(session_id, record)

    def clear_commit_unknown_send(self, session_id: str) -> dict[str, Any]:
        return self._pending_state_coordinator_for_manager().clear_commit_unknown_send(session_id)

    def _prune_missing_commit_unknown_sends(self, *, max_age_seconds: float = COMMIT_UNKNOWN_ORPHAN_PRUNE_SECONDS) -> bool:
        return self._pending_state_coordinator_for_manager().prune_missing_commit_unknown_sends(max_age_seconds=max_age_seconds)

    def _load_recent_cwds(self) -> None:
        cleaned = self._session_store_for_manager().load_recent_cwds()
        with self._lock:
            self._recent_cwds = cleaned

    def _save_recent_cwds(self) -> None:
        with self._lock:
            obj = dict(getattr(self, "_recent_cwds", {}))
        self._session_store_for_manager().save_recent_cwds(obj)

    def _remember_recent_cwd(self, cwd: Any, *, ts: Any = None) -> bool:
        return self._recent_cwd_coordinator_for_manager().remember(cwd, ts=ts)

    def _backfill_recent_cwds_from_logs(self) -> None:
        return self._recent_cwd_coordinator_for_manager().backfill_from_logs()

    def recent_cwds(self, *, limit: int = RECENT_CWD_MAX) -> list[str]:
        return self._recent_cwd_coordinator_for_manager().list_recent(limit=limit)

    def _queue_len(self, session_id: str) -> int:
        return self._queue_coordinator_for_manager().queue_len(session_id)

    def _mark_queue_orphan_recovery_locked(self, session_id: str) -> bool:
        return self._queue_coordinator_for_manager().mark_orphan_recovery_locked(session_id)

    def _queue_has_recovery_items_locked(self, session_id: str) -> bool:
        return self._queue_coordinator_for_manager().has_recovery_items_locked(session_id)

    def _queue_list_local(self, session_id: str) -> list[dict[str, Any]]:
        return self._queue_coordinator_for_manager().list_local(session_id)

    def _queue_append_item_local(self, session_id: str, text: str, *, reject_recovery_barrier: bool = False) -> tuple[dict[str, Any], int]:
        return self._queue_coordinator_for_manager().append_item_local(
            session_id,
            text,
            reject_recovery_barrier=reject_recovery_barrier,
        )

    def _queue_enqueue_local(self, session_id: str, text: str) -> dict[str, Any]:
        return self._queue_coordinator_for_manager().enqueue_local(session_id, text)

    def _queue_delete_local(self, session_id: str, item_id: str, *, allow_commit_unknown: bool = False, allow_orphan_recovery: bool = False) -> dict[str, Any]:
        return self._queue_coordinator_for_manager().delete_local(
            session_id,
            item_id,
            allow_commit_unknown=allow_commit_unknown,
            allow_orphan_recovery=allow_orphan_recovery,
        )

    def _queue_update_local(self, session_id: str, item_id: str, text: str) -> dict[str, Any]:
        return self._queue_coordinator_for_manager().update_local(session_id, item_id, text)

    def _queue_move_local(self, session_id: str, item_id: str, to_index: int) -> dict[str, Any]:
        return self._queue_coordinator_for_manager().move_local(session_id, item_id, to_index)

    def _queue_session_state(self, session_id: str) -> tuple[Session, Path | None]:
        return self._queue_coordinator_for_manager().session_state(session_id)

    def _promote_queue_head_if_sendable(
        self,
        session_id: str,
        *,
        require_idle_grace: bool,
        now_ts: float | None = None,
        expected_item_id: str | None = None,
    ) -> dict[str, Any] | None:
        return self._queue_coordinator_for_manager().promote_head_if_sendable(
            session_id,
            require_idle_grace=require_idle_grace,
            now_ts=now_ts,
            expected_item_id=expected_item_id,
        )

    def _broker_busy_queue_from_state(self, state: dict[str, Any]) -> tuple[bool, int]:
        return _broker_busy_queue_from_state(state)

    def _log_size_or_none(self, log_path: Path | None) -> int | None:
        return _log_path_size_or_none(log_path)

    def _clear_confirmed_send_boundary_locked(self, s: Session) -> None:
        _clear_session_confirmed_send_boundary(s)

    def _confirmed_send_boundary_unresolved_for_session(self, session_id: str, log_path: Path | None, log_size: int | None) -> bool:
        with self._lock:
            s = self._sessions.get(session_id)
            return _consume_session_confirmed_send_boundary(s, log_path, log_size)

    def _remote_ready_from_state_and_log(self, session_id: str, state: dict[str, Any], log_path: Path | None) -> bool:
        return self._readiness_coordinator_for_manager().remote_ready_from_state_and_log(session_id, state, log_path)

    def _remote_state_after_metadata_probe(self, session_id: str, *, log_path_before_state: Path | None) -> tuple[dict[str, Any], Path | None]:
        return self._readiness_coordinator_for_manager().remote_state_after_metadata_probe(
            session_id,
            log_path_before_state=log_path_before_state,
        )

    def _send_remote_ready(self, session_id: str, *, allow_pending_attachment: bool = False) -> bool:
        return self._readiness_coordinator_for_manager().send_remote_ready(
            session_id,
            allow_pending_attachment=allow_pending_attachment,
        )

    def _queue_remote_ready(self, session_id: str, *, log_path: Path | None) -> bool:
        return self._readiness_coordinator_for_manager().queue_remote_ready(session_id, log_path=log_path)

    def _files_key_for_session(self, session_id: str) -> tuple[str, list[str], "Session"]:
        return self._files_coordinator_for_manager().files_key_for_session(session_id)

    def files_get(self, session_id: str) -> list[str]:
        return self._files_coordinator_for_manager().get(session_id)

    def files_add(self, session_id: str, path: str) -> list[str]:
        return self._files_coordinator_for_manager().add(session_id, path)

    def files_clear(self, session_id: str) -> None:
        return self._files_coordinator_for_manager().clear(session_id)

    def unattended_get(self, session_id: str) -> dict[str, Any]:
        return self._unattended_config_coordinator_for_manager().get(session_id)

    def unattended_set(
        self,
        session_id: str,
        *,
        enabled: bool | None = None,
        request: str | None = None,
        cooldown_minutes: int | None = None,
        remaining_injections: int | None = None,
    ) -> dict[str, Any]:
        return self._unattended_config_coordinator_for_manager().set(
            session_id,
            enabled=enabled,
            request=request,
            cooldown_minutes=cooldown_minutes,
            remaining_injections=remaining_injections,
        )

    def _session_display_name(self, session_id: str) -> str:
        return self._voice_runtime_for_manager().session_display_name(session_id)

    def _observe_rollout_delta(self, session_id: str, *, log_path: Path | None = None, old_off: int = 0, objs: list[dict[str, Any]], new_off: int) -> None:
        return self._voice_runtime_for_manager().observe_rollout_delta(
            session_id,
            log_path=log_path,
            old_off=old_off,
            objs=objs,
            new_off=new_off,
        )

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
        return self._voice_runtime_for_manager().scan_sweep()

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
        return self._unattended_sweep_coordinator_for_manager().sweep()

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
        return self._queue_sweep_coordinator_for_manager().sweep()

    def _discovery_deps(self) -> DiscoveryDeps:
        return DiscoveryDeps(
            pid_alive=_pid_alive,
            proc_find_open_rollout_log=lambda proc_root, root_pid, agent_backend, cwd, ignored_paths: _proc_find_open_rollout_log(
                proc_root=proc_root,
                root_pid=root_pid,
                agent_backend=agent_backend,
                cwd=cwd,
                ignored_paths=ignored_paths,
            ),
            read_session_meta_or_none=lambda log_path, agent_backend, context: _read_session_meta_or_none(
                log_path,
                agent_backend=agent_backend,
                context=context,
            ),
            coerce_main_thread_log=lambda thread_id, log_path: _coerce_main_thread_log(thread_id=thread_id, log_path=log_path),
            session_transport=lambda meta: self._session_transport(meta=meta),
            session_run_settings=lambda meta, log_path, agent_backend: self._session_run_settings(
                meta=meta,
                log_path=log_path,
                agent_backend=agent_backend,
            ),
            sock_call=lambda sock, req, timeout_s: self._sock_call(sock, req, timeout_s=timeout_s),
            broker_busy_queue_from_state=self._broker_busy_queue_from_state,
            broker_interrupted_idle_from_state=_broker_interrupted_idle_from_state,
            sock_error_definitely_stale=_sock_error_definitely_stale,
            token_update_finder=_rollout_log._find_latest_token_update,
        )

    def _apply_discovery_result(self, result: DiscoveryResult) -> None:
        return self._discovery_registry_for_manager().apply_result(result)

    def _upsert_discovery_registration(self, registration: DiscoveryRegistration) -> None:
        return self._discovery_registry_for_manager().upsert_registration(registration)

    def _discover_existing(self, *, force: bool = False) -> None:
        if not force:
            now = time.time()
            with self._lock:
                last = float(self._last_discover_ts)
            if (now - last) < DISCOVER_MIN_INTERVAL_SECONDS:
                return
        with self._lock:
            hidden_sessions = set(getattr(self, "_hidden_sessions", set()))
        result = _discover_sessions(
            SOCK_DIR,
            proc_root=PROC_ROOT,
            hidden_sessions=hidden_sessions,
            deps=self._discovery_deps(),
        )
        self._apply_discovery_result(result)
        with self._lock:
            self._last_discover_ts = time.time()

    def _refresh_session_state(self, session_id: str, sock_path: Path, timeout_s: float = 0.4) -> tuple[bool, BaseException | None]:
        return self._prune_coordinator_for_manager().refresh_session_state(session_id, sock_path, timeout_s=timeout_s)

    def _prune_dead_sessions(self) -> None:
        return self._prune_coordinator_for_manager().prune_dead_sessions()

    def _update_meta_counters(self) -> None:
        return self._log_runtime_for_manager().update_meta_counters()

    def list_sessions(self) -> list[dict[str, Any]]:
        return self._list_coordinator_for_manager().list_sessions()

    def get_session(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def refresh_session_meta(self, session_id: str, *, drain_queue: bool = False) -> None:
        return self._refresh_coordinator_for_manager().refresh_session_meta(session_id, drain_queue=drain_queue)

    def _attach_notification_texts(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._voice_runtime_for_manager().attach_notification_texts(events)

    def mark_log_delta(self, session_id: str, *, objs: list[dict[str, Any]], new_off: int) -> None:
        return self._log_runtime_for_manager().mark_log_delta(session_id, objs=objs, new_off=new_off)

    def idle_from_log(self, session_id: str) -> bool:
        return self._log_runtime_for_manager().idle_from_log(session_id)

    def idle_from_log_path(self, session_id: str, log_path: Path) -> bool:
        return self._log_runtime_for_manager().idle_from_log_path(session_id, log_path)

    def _sock_call(self, sock_path: Path, req: dict[str, Any], timeout_s: float | None = 2.0, *, track_request_sent: bool = False) -> dict[str, Any]:
        return _call_control_socket_impl(sock_path, req, timeout_s=timeout_s, track_request_sent=track_request_sent)

    def _control_coordinator_for_manager(self) -> SessionControlCoordinator:
        return SessionControlCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            sock_call=lambda sock, req, **kwargs: self._sock_call(sock, req, **kwargs),
            pid_alive=_pid_alive,
            unlink_quiet=_unlink_quiet,
            clear_deleted_session_state=self._clear_deleted_session_state,
            broker_busy_queue=self._broker_busy_queue_from_state,
            broker_interrupted_idle=_broker_interrupted_idle_from_state,
            control_socket_call_error=ControlSocketCallError,
            commit_unknown_error=SessionCommitUnknownError,
        )

    def _attachment_coordinator_for_manager(self) -> SessionAttachmentCoordinator:
        return SessionAttachmentCoordinator(
            input_lock_for_session=self._input_lock_for_session,
            attachment_injection_ready=self.attachment_injection_ready,
            inject_keys=self.inject_keys,
            set_pending_attachment=self._set_pending_attachment,
            not_ready_error=SessionNotReadyError,
            injection_error=SessionInjectionError,
            commit_unknown_error=SessionCommitUnknownError,
        )

    def _list_coordinator_for_manager(self) -> SessionListCoordinator:
        return SessionListCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            queues=lambda: self._queues,
            unattended=lambda: self._unattended,
            aliases=lambda: self._aliases,
            hidden_sessions=lambda: self._hidden_sessions,
            commit_unknown_sends=lambda: self._commit_unknown_sends,
            store=self._session_store_for_manager(),
            discover_existing_if_stale=self._discover_existing_if_stale,
            prune_dead_sessions=self._prune_dead_sessions,
            update_meta_counters=self._update_meta_counters,
            save_files=self._save_files,
            save_sidebar_meta=self._save_sidebar_meta,
            save_recent_cwds=self._save_recent_cwds,
            now=time.time,
            runtime_probes=ListingRuntimeProbes(
                last_conversation_ts_from_tail=lambda path: _last_conversation_ts_from_tail(path),
                read_run_settings_from_log=lambda path, agent_backend: _read_run_settings_from_log(path, agent_backend=agent_backend),
                log_size_or_none=self._log_size_or_none,
                send_boundary_unresolved=self._confirmed_send_boundary_unresolved_for_session,
                idle_from_log_path=self.idle_from_log_path,
                current_git_branch=_current_git_branch,
            ),
            include_launch_attempts=lambda: bool(getattr(self, "_include_launch_attempts", False)),
            read_launch_attempts=lambda: _read_launch_attempts(path=LAUNCH_ATTEMPTS_PATH, max_records=100, max_age_s=24 * 3600),
            launch_attempt_row=_launch_attempt_row,
            clean_unattended_cooldown_minutes=_clean_unattended_cooldown_minutes,
            clean_unattended_remaining_injections=_clean_unattended_remaining_injections,
            provider_choice_for_settings=_provider_choice_for_settings,
            resolve_session_cwd=_resolve_session_cwd,
            unattended_default_idle_minutes=UNATTENDED_DEFAULT_IDLE_MINUTES,
            unattended_default_max_injections=UNATTENDED_DEFAULT_MAX_INJECTIONS,
            priority_half_life_seconds=SIDEBAR_PRIORITY_HALF_LIFE_SECONDS,
            priority_bucket_seconds=SIDEBAR_PRIORITY_BUCKET_SECONDS,
        )

    def _refresh_coordinator_for_manager(self) -> SessionRefreshCoordinator:
        return SessionRefreshCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            prune_stale_socket_without_metadata=self._prune_stale_socket_without_metadata,
            log_invalid_sidecar_metadata=_log_invalid_sidecar_metadata,
            session_transport=self._session_transport,
            sock_call=lambda sock, req, **kwargs: self._sock_call(sock, req, **kwargs),
            broker_tail_has_session_detach_marker=_broker_tail_has_session_detach_marker,
            pid_alive=_pid_alive,
            proc_find_open_rollout_log=_proc_find_open_rollout_log,
            proc_root=PROC_ROOT,
            read_session_meta_or_none=_read_session_meta_or_none,
            coerce_main_thread_log=_coerce_main_thread_log,
            clean_optional_text=_clean_optional_text,
            session_run_settings=self._session_run_settings,
            normalize_requested_service_tier=_normalize_requested_service_tier,
            reset_log_caches=lambda session, log_off: self._reset_log_caches(session, meta_log_off=log_off),
            queue_len=self._queue_len,
            maybe_drain_session_queue=self._maybe_drain_session_queue,
        )

    def _readiness_coordinator_for_manager(self) -> SessionReadinessCoordinator:
        return SessionReadinessCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            refresh_session_meta_if_sidecar_exists=self._refresh_session_meta_if_sidecar_exists,
            get_state=self.get_state,
            log_size_or_none=self._log_size_or_none,
            confirmed_send_boundary_unresolved_for_session=self._confirmed_send_boundary_unresolved_for_session,
            idle_from_log=self.idle_from_log,
            queue_len=lambda session_id: self._queue_store_for_manager().queue_len(self._queues, session_id),
            not_ready_error=SessionNotReadyError,
        )

    def _unattended_sweep_coordinator_for_manager(self) -> UnattendedSweepCoordinator:
        return UnattendedSweepCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            unattended=lambda: self._unattended,
            unattended_last_injected=lambda: self._unattended_last_injected,
            unattended_last_injected_scope=lambda: self._unattended_last_injected_scope,
            discover_existing_if_stale=self._discover_existing_if_stale,
            prune_dead_sessions=self._prune_dead_sessions,
            input_lock_for_session=self._input_lock_for_session,
            save_unattended=self._save_unattended,
            get_state=self.get_state,
            broker_busy_queue_from_state=self._broker_busy_queue_from_state,
            queue_len=self._queue_len,
            last_chat_role_ts_from_tail=_last_chat_role_ts_from_tail,
            send=self.send,
            now=time.time,
            prompt_prefix=UNATTENDED_PROMPT_PREFIX,
            default_idle_minutes=UNATTENDED_DEFAULT_IDLE_MINUTES,
            default_max_injections=UNATTENDED_DEFAULT_MAX_INJECTIONS,
            max_scan_bytes=UNATTENDED_MAX_SCAN_BYTES,
        )

    def _queue_sweep_coordinator_for_manager(self) -> QueueSweepCoordinator:
        return QueueSweepCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            queues=lambda: self._queues,
            commit_unknown_sends=lambda: self._commit_unknown_sends,
            queue_store=self._queue_store_for_manager(),
            discover_existing_if_stale=self._discover_existing_if_stale,
            prune_dead_sessions=self._prune_dead_sessions,
            mark_queue_orphan_recovery_locked=self._mark_queue_orphan_recovery_locked,
            save_queues=self._save_queues,
            maybe_drain_session_queue=self._maybe_drain_session_queue,
        )

    def _voice_runtime_for_manager(self) -> VoiceRuntimeCoordinator:
        return VoiceRuntimeCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            aliases=lambda: self._aliases,
            voice_push=lambda: getattr(self, "_voice_push", None),
            discover_existing_if_stale=self._discover_existing_if_stale,
            prune_dead_sessions=self._prune_dead_sessions,
            refresh_session_meta=lambda session_id: self.refresh_session_meta(session_id),
            read_jsonl_from_offset=_read_jsonl_from_offset,
            extract_delivery_messages=lambda objs, **kwargs: _extract_delivery_messages(objs, **kwargs),
            cc_pending_tool_ids_before=_rollout_log._cc_pending_tool_ids_before,
        )

    def _log_runtime_for_manager(self) -> SessionLogRuntimeCoordinator:
        return SessionLogRuntimeCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            analyze_log_chunk=_analyze_log_chunk,
            turn_context_run_settings=_turn_context_run_settings,
            compute_idle_from_log=_compute_idle_from_log,
            read_jsonl_from_offset=_read_jsonl_from_offset,
            find_latest_token_update=_rollout_log._find_latest_token_update,
        )

    def _files_coordinator_for_manager(self) -> SessionFilesCoordinator:
        return SessionFilesCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            store=self._session_store_for_manager(),
            save_files=self._save_files,
        )

    def _ui_state_coordinator_for_manager(self) -> SessionUiStateCoordinator:
        return SessionUiStateCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            aliases=lambda: self._aliases,
            set_aliases=lambda value: setattr(self, "_aliases", value),
            sidebar_meta=lambda: self._sidebar_meta,
            set_sidebar_meta=lambda value: setattr(self, "_sidebar_meta", value),
            hidden_sessions=lambda: self._hidden_sessions,
            set_hidden_sessions=lambda value: setattr(self, "_hidden_sessions", value),
            save_aliases=self._save_aliases,
            save_sidebar_meta=self._save_sidebar_meta,
            save_hidden_sessions=self._save_hidden_sessions,
            clean_alias=_clean_alias,
            clean_priority_offset=_clean_priority_offset,
            clean_snooze_until=_clean_snooze_until,
            clean_dependency_session_id=_clean_dependency_session_id,
        )

    def _unattended_config_coordinator_for_manager(self) -> SessionUnattendedConfigCoordinator:
        return SessionUnattendedConfigCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            unattended=lambda: self._unattended,
            unattended_last_injected=lambda: self._unattended_last_injected,
            input_lock_for_session=self._input_lock_for_session,
            save_unattended=self._save_unattended,
            clean_unattended_cooldown_minutes=_clean_unattended_cooldown_minutes,
            clean_unattended_remaining_injections=_clean_unattended_remaining_injections,
        )

    def _cleanup_coordinator_for_manager(self) -> SessionCleanupCoordinator:
        return SessionCleanupCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            aliases=lambda: self._aliases,
            sidebar_meta=lambda: self._sidebar_meta,
            unattended=lambda: self._unattended,
            files=lambda: self._files,
            queues=lambda: self._queues,
            commit_unknown_sends=lambda: self._commit_unknown_sends,
            input_locks=lambda: getattr(self, "_input_locks", {}),
            pending_attachment_ids=lambda: getattr(self, "_pending_attachment_ids", set()),
            unhide_session=self._unhide_session,
            mark_queue_orphan_recovery_locked=self._mark_queue_orphan_recovery_locked,
            unlink_quiet=_unlink_quiet,
            save_pending_attachments=self._save_pending_attachments,
            save_commit_unknown_sends=self._save_commit_unknown_sends,
            save_aliases=self._save_aliases,
            save_sidebar_meta=self._save_sidebar_meta,
            save_unattended=self._save_unattended,
            save_files=self._save_files,
            save_queues=self._save_queues,
        )

    def _pending_state_coordinator_for_manager(self) -> SessionPendingStateCoordinator:
        return SessionPendingStateCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            pending_attachment_ids=lambda: getattr(self, "_pending_attachment_ids", None),
            set_pending_attachment_ids=lambda value: setattr(self, "_pending_attachment_ids", value),
            commit_unknown_sends=lambda: getattr(self, "_commit_unknown_sends", None),
            set_commit_unknown_sends=lambda value: setattr(self, "_commit_unknown_sends", value),
            mark_queue_orphan_recovery_locked=self._mark_queue_orphan_recovery_locked,
            save_pending_attachments=self._save_pending_attachments,
            save_commit_unknown_sends=self._save_commit_unknown_sends,
            save_queues=self._save_queues,
            now=time.time,
            commit_unknown_orphan_prune_seconds=COMMIT_UNKNOWN_ORPHAN_PRUNE_SECONDS,
        )

    def _recent_cwd_coordinator_for_manager(self) -> SessionRecentCwdCoordinator:
        return SessionRecentCwdCoordinator(
            lock=self._lock,
            recent_cwds=lambda: getattr(self, "_recent_cwds", None),
            set_recent_cwds=lambda value: setattr(self, "_recent_cwds", value),
            clean_recent_cwd=_clean_recent_cwd,
            iter_session_logs=_iter_session_logs,
            resume_candidate_from_log=_resume_candidate_from_log,
            save_recent_cwds=self._save_recent_cwds,
            now=time.time,
            max_recent_cwds=RECENT_CWD_MAX,
        )

    def _lifecycle_coordinator_for_manager(self) -> SessionLifecycleCoordinator:
        return SessionLifecycleCoordinator(
            lock=getattr(self, "_lock", threading.RLock()),
            sessions=lambda: getattr(self, "_sessions", {}),
            sock_call=lambda sock, req, **kwargs: self._sock_call(sock, req, **kwargs),
            process_group_alive=_process_group_alive,
            pid_alive=_pid_alive,
            terminate_process_group=_terminate_process_group,
            terminate_process=_terminate_process,
            unlink_quiet=_unlink_quiet,
            commit_unknown_sends=lambda: getattr(self, "_commit_unknown_sends", {}),
            queue_has_recovery_items_locked=self._queue_has_recovery_items_locked,
            clear_deleted_session_state=self._clear_deleted_session_state,
            read_launch_attempts=lambda: _read_launch_attempts(path=LAUNCH_ATTEMPTS_PATH, max_records=100, max_age_s=24 * 3600),
            launch_attempt_row=_launch_attempt_row,
            hide_session=self._hide_session,
            files_clear=self.files_clear,
            clean_optional_text=_clean_optional_text,
            kill_session_via_pids_fallback=self._kill_session_via_pids,
        )

    def _discovery_registry_for_manager(self) -> SessionDiscoveryRegistryCoordinator:
        return SessionDiscoveryRegistryCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            pending_attachment_ids=lambda: getattr(self, "_pending_attachment_ids", set()),
            commit_unknown_sends=lambda: getattr(self, "_commit_unknown_sends", {}),
            reset_log_caches=lambda session, log_off: self._reset_log_caches(session, meta_log_off=log_off),
            record_launch_attempt=_record_launch_attempt,
            prune_stale_socket_without_metadata=self._prune_stale_socket_without_metadata,
            unhide_session=self._unhide_session,
            unlink_quiet=_unlink_quiet,
            remember_recent_cwd=self._remember_recent_cwd,
            save_recent_cwds=self._save_recent_cwds,
            stderr=sys.stderr,
        )

    def _prune_coordinator_for_manager(self) -> SessionPruneCoordinator:
        return SessionPruneCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            sock_call=lambda sock, req, **kwargs: self._sock_call(sock, req, **kwargs),
            broker_busy_queue_from_state=self._broker_busy_queue_from_state,
            broker_interrupted_idle_from_state=_broker_interrupted_idle_from_state,
            sock_error_definitely_stale=_sock_error_definitely_stale,
            pid_alive=_pid_alive,
            latest_launch_attempt=_latest_launch_attempt,
            submitted_user_messages=_submitted_user_messages,
            launch_failure_tail=lambda record: _launch_failure_tail(record or {}),
            which_tmux=shutil.which,
            tmux_pane_snapshot=_tmux_pane_snapshot,
            clean_optional_text=_clean_optional_text,
            record_launch_attempt=_record_launch_attempt,
            clear_deleted_session_state=self._clear_deleted_session_state,
            unlink_quiet=_unlink_quiet,
            stderr=sys.stderr,
        )

    def _send_coordinator_for_manager(self) -> SessionSendCoordinator:
        return SessionSendCoordinator(
            lock=self._lock,
            sessions=lambda: self._sessions,
            input_lock_for_session=self._input_lock_for_session,
            queue_len=lambda session_id: self._queue_store_for_manager().queue_len(getattr(self, "_queues", {}), session_id),
            send_remote_ready=self._send_remote_ready,
            log_size_or_none=self._log_size_or_none,
            call_confirmed_send=lambda session_id, **kwargs: self._control_coordinator_for_manager().call_confirmed_send(session_id, **kwargs),
            set_pending_attachment=self._set_pending_attachment,
            set_commit_unknown_send=self._set_commit_unknown_send,
            record_prelog_user_message=lambda session, text: self._record_prelog_user_message(session, text, source="send"),
            now=time.time,
            send_commit_timeout_seconds=SEND_COMMIT_TIMEOUT_SECONDS,
            not_ready_error=SessionNotReadyError,
            commit_unknown_error=SessionCommitUnknownError,
            injection_error=SessionInjectionError,
            timeout_errors=(TimeoutError, socket.timeout),
        )

    def _prelog_user_message_recorder_for_manager(self) -> PrelogUserMessageRecorder:
        return PrelogUserMessageRecorder(
            latest_launch_attempt=_latest_launch_attempt,
            submitted_user_messages=_submitted_user_messages,
            clean_optional_text=_clean_optional_text,
            record_launch_attempt=_record_launch_attempt,
            now=time.time,
        )

    def _kill_session_via_pids(self, s: Session) -> bool:
        return self._lifecycle_coordinator_for_manager().kill_session_via_pids(s)

    def kill_session(self, session_id: str) -> bool:
        return self._lifecycle_coordinator_for_manager().kill_session(session_id)

    def _live_session_for_resume_target(self, resume_id: str, resume_row: dict[str, Any] | None) -> Session | None:
        return self._lifecycle_coordinator_for_manager().live_session_for_resume_target(resume_id, resume_row)

    def _web_launch_coordinator_for_manager(self) -> SessionWebLaunchCoordinator:
        return SessionWebLaunchCoordinator(
            resolve_dir_target=_resolve_dir_target,
            create_git_worktree=_create_git_worktree,
            codex_trust_override_for_path=_codex_trust_override_for_path,
            list_resume_candidates_for_cwd=_list_resume_candidates_for_cwd,
            live_session_for_resume_target=self._live_session_for_resume_target,
            load_env_file=_load_env_file,
            environ=os.environ,
            dotenv_path=_DOTENV,
            homes={"codex": CODEX_HOME, "pi": PI_HOME, "cc": CC_HOME},
            python_executable=sys.executable,
            tmux_session_name=TMUX_SESSION_NAME,
            repo_root=Path(__file__).resolve().parent.parent,
            record_launch_attempt=_record_launch_attempt,
            now=time.time,
            stderr=sys.stderr,
            which_tmux=shutil.which,
            run=subprocess.run,
            popen=subprocess.Popen,
            wait_or_raise=_wait_or_raise,
            wait_for_spawned_broker_meta=_wait_for_spawned_broker_meta,
            tmux_pane_snapshot=_tmux_pane_snapshot,
            drain_stream=_drain_stream,
            launch_error=SessionLaunchError,
        )

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
        return self._web_launch_coordinator_for_manager().spawn_web_session(
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
        )

    def delete_session(self, session_id: str) -> bool:
        return self._lifecycle_coordinator_for_manager().delete_session(session_id)

    def _record_prelog_user_message(self, session: Session, text: str, *, source: str) -> None:
        return self._prelog_user_message_recorder_for_manager().record(session, text, source=source)

    def send(self, session_id: str, text: str, *, allow_pending_attachment: bool = False, queue_item_id: str | None = None) -> dict[str, Any]:
        return self._send_coordinator_for_manager().send(
            session_id,
            text,
            allow_pending_attachment=allow_pending_attachment,
            queue_item_id=queue_item_id,
        )

    def enqueue(self, session_id: str, text: str) -> dict[str, Any]:
        return self._queue_coordinator_for_manager().enqueue(session_id, text)

    def queue_list(self, session_id: str) -> list[dict[str, Any]]:
        return self._queue_list_local(session_id)

    def queue_delete(self, session_id: str, item_id: str, *, allow_commit_unknown: bool = False, allow_orphan_recovery: bool = False) -> dict[str, Any]:
        return self._queue_delete_local(session_id, item_id, allow_commit_unknown=allow_commit_unknown, allow_orphan_recovery=allow_orphan_recovery)

    def queue_update(self, session_id: str, item_id: str, text: str) -> dict[str, Any]:
        return self._queue_update_local(session_id, item_id, text)

    def queue_move(self, session_id: str, item_id: str, to_index: int) -> dict[str, Any]:
        return self._queue_move_local(session_id, item_id, to_index)

    def get_state(self, session_id: str) -> dict[str, Any]:
        return self._control_coordinator_for_manager().get_state(session_id)

    def get_tail(self, session_id: str) -> str:
        return self._control_coordinator_for_manager().get_tail(session_id)

    def _refresh_session_meta_if_sidecar_exists(self, session_id: str, *, drain_queue: bool = False) -> None:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            meta_path = s.sock_path.with_suffix(".json")
        if meta_path.exists():
            self.refresh_session_meta(session_id, drain_queue=drain_queue)

    def attachment_injection_ready(self, session_id: str) -> bool:
        return self._readiness_coordinator_for_manager().attachment_injection_ready(session_id)

    def inject_attachment_keys(self, session_id: str, seq: str) -> dict[str, Any]:
        return self._attachment_coordinator_for_manager().inject_attachment_keys(session_id, seq)

    def inject_keys(self, session_id: str, seq: str, *, track_request_sent: bool = False, interrupt: bool = False) -> dict[str, Any]:
        return self._control_coordinator_for_manager().inject_keys(
            session_id,
            seq,
            track_request_sent=track_request_sent,
            interrupt=interrupt,
        )

    def mark_turn_complete(self, session_id: str, payload: dict[str, Any]) -> None:
        return


MANAGER = SessionManager()


def _read_static_bytes(path: Path) -> bytes:
    return _read_static_bytes_impl(path, attach_upload_max_bytes=ATTACH_UPLOAD_MAX_BYTES)



def _route_deps_factory() -> ServerRouteDepsFactory:
    return ServerRouteDepsFactory(sys.modules[__name__])


def _message_runtime_snapshot(
    session_id: str,
    s: Session,
    *,
    token_update: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool, int, dict[str, Any] | None]:
    return _route_deps_factory().message_runtime_snapshot(session_id, s, token_update=token_update)



Handler = make_server_handler(sys.modules[__name__])



def main() -> None:
    return _run_server_main(sys.modules[__name__])


if __name__ == "__main__":
    main()
