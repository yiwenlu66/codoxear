#!/usr/bin/env python3
from __future__ import annotations

import base64
import errno
import hashlib
import hmac
import http.server
import math
import os
import posixpath
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from .agent_backend import get_agent_backend
from .agent_backend import normalize_agent_backend
from .auth import CookieAuthSettings
from .auth import load_or_create_hmac_secret as _load_or_create_hmac_secret_impl
from .auth import parse_cookies as _parse_cookies_impl
from .auth import require_auth as _require_auth_impl
from .auth import set_auth_cookie as _set_auth_cookie_impl
from .auth import sign_cookie as _sign_cookie_impl
from .auth import verify_cookie as _verify_cookie_impl
from .client_file_paths import describe_session_cwd as _describe_session_cwd_impl
from .client_file_paths import list_session_relative_files as _list_session_relative_files_impl
from .client_file_paths import path_resolves_inside as _path_resolves_inside_impl
from .client_file_paths import resolve_client_file_path as _resolve_client_file_path_impl
from .client_file_paths import resolve_git_client_file_view as _resolve_git_client_file_view_impl
from .client_file_paths import resolve_git_existing_regular_file as _resolve_git_existing_regular_file_impl
from .client_file_paths import resolve_tracked_file_by_basename as _resolve_tracked_file_by_basename_impl
from .client_file_paths import resolve_unique_bare_filename as _resolve_unique_bare_filename_impl
from .client_file_paths import symlink_payload_view as _symlink_payload_view_impl
from . import rollout_log as _rollout_log
from .control_socket import ControlSocketCallError
from .control_socket import call_control_socket as _call_control_socket_impl
from .file_response import send_attachment_file_response as _send_attachment_file_response
from .file_lock_runtime import file_write_lock as _file_write_lock_impl
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
from .launch_path_runtime import codex_trust_override_for_path as _codex_trust_override_for_path_impl
from .launch_path_runtime import expand_user_path as _expand_user_path_impl
from .launch_path_runtime import load_env_file as _load_env_file_impl
from .launch_path_runtime import resolve_dir_target as _resolve_dir_target_impl
from .launch_path_runtime import resolve_existing_dir as _resolve_existing_dir_impl
from .launch_path_runtime import resolve_new_path as _resolve_new_path_impl
from .launch_defaults_runtime import launch_defaults_for_request as _launch_defaults_for_request_impl
from .launch_defaults_runtime import launch_defaults_signature as _launch_defaults_signature_impl
from .launch_defaults_runtime import path_signature as _path_signature_impl
from .launch_defaults_runtime import read_new_session_defaults_cached as _read_new_session_defaults_cached_impl
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
from .path_runtime import expanduser_path as _expanduser_path_impl
from .path_runtime import require_existing_file as _require_existing_file_impl
from .path_runtime import resolve_existing_absolute_file as _resolve_existing_absolute_file_impl
from .path_runtime import resolve_existing_session_file as _resolve_existing_session_file_impl
from .path_runtime import resolve_session_cwd as _resolve_session_cwd_impl
from .path_runtime import resolve_session_path as _resolve_session_path_impl
from .path_runtime import resolve_under as _resolve_under_impl
from .pi_log import pi_user_text as _pi_user_text
from .pi_log import read_pi_run_settings as _read_pi_run_settings
from .process_runtime import terminate_process as _terminate_process_impl
from .process_runtime import terminate_process_group as _terminate_process_group_impl
from .queue_store import QueueStore
from .queue_store import coerce_queue_item as _queue_store_coerce_item
from .session_discovery import DiscoveryDeps
from .session_discovery import DiscoveryRegistration
from .session_discovery import DiscoveryResult
from .session_discovery import discover_sessions as _discover_sessions
from .session_launcher import drain_stream as _drain_stream_impl
from .session_launcher import wait_for_spawned_broker_meta as _wait_for_spawned_broker_meta_impl
from .session_launcher import wait_or_raise as _wait_or_raise_impl
from .session_log_metadata import find_new_session_log as _find_new_session_log_metadata_impl
from .session_log_metadata import find_session_log_for_session_id as _find_session_log_for_session_id_metadata_impl
from .session_log_metadata import iter_session_logs_for_backend as _iter_session_logs_for_backend_impl
from .session_log_metadata import read_run_settings_from_log as _read_run_settings_from_log_impl
from .session_log_metadata import read_session_meta as _read_session_meta_impl
from .session_log_metadata import read_session_meta_or_none as _read_session_meta_or_none_impl
from .session_log_metadata import sessions_dir_for_backend as _sessions_dir_for_backend_impl
from .session_log_metadata import turn_context_run_settings as _turn_context_run_settings_impl
from .session_resume import coerce_main_thread_log as _coerce_main_thread_log_impl
from .session_resume import first_user_message_preview_from_log as _first_user_message_preview_from_log_impl
from .session_resume import is_scaffold_user_text as _is_scaffold_user_text_impl
from .session_resume import list_resume_candidates_for_cwd as _list_resume_candidates_for_cwd_impl
from .session_resume import resume_candidate_from_log as _resume_candidate_from_log_impl
from .session_resume import resume_preview_from_text as _resume_preview_from_text_impl
from .session_resume import user_message_text as _user_message_text_impl
from .session_listing import clip01 as _listing_clip01
from .session_listing import priority_from_elapsed_seconds as _listing_priority_from_elapsed_seconds
from .session_listing import sidebar_priority_elapsed_seconds as _listing_sidebar_priority_elapsed_seconds
from .session_listing import sidebar_time_priority_from_elapsed_seconds as _listing_sidebar_time_priority_from_elapsed_seconds
from .session_manager_bootstrap import create_voice_push_coordinator as _create_voice_push_coordinator_impl
from .session_manager_bootstrap import load_manager_persistent_state as _load_manager_persistent_state_impl
from .session_manager_bootstrap import seed_manager_in_memory_state as _seed_manager_in_memory_state_impl
from .session_manager_bootstrap import start_manager_worker_threads as _start_manager_worker_threads_impl
from .session_manager_discovery import discover_existing_for_manager as _discover_existing_for_manager_impl
from .session_manager_factories import attachment_coordinator_for_manager as _attachment_coordinator_for_manager_impl
from .session_manager_factories import cleanup_coordinator_for_manager as _cleanup_coordinator_for_manager_impl
from .session_manager_factories import control_coordinator_for_manager as _control_coordinator_for_manager_impl
from .session_manager_factories import discovery_deps_for_manager as _discovery_deps_for_manager_impl
from .session_manager_factories import discovery_registry_for_manager as _discovery_registry_for_manager_impl
from .session_manager_factories import files_coordinator_for_manager as _files_coordinator_for_manager_impl
from .session_manager_factories import lifecycle_coordinator_for_manager as _lifecycle_coordinator_for_manager_impl
from .session_manager_factories import list_coordinator_for_manager as _list_coordinator_for_manager_impl
from .session_manager_factories import log_runtime_for_manager as _log_runtime_for_manager_impl
from .session_manager_factories import pending_state_coordinator_for_manager as _pending_state_coordinator_for_manager_impl
from .session_manager_factories import prelog_user_message_recorder_for_manager as _prelog_user_message_recorder_for_manager_impl
from .session_manager_factories import prune_coordinator_for_manager as _prune_coordinator_for_manager_impl
from .session_manager_factories import queue_coordinator_for_manager as _queue_coordinator_for_manager_impl
from .session_manager_factories import queue_sweep_coordinator_for_manager as _queue_sweep_coordinator_for_manager_impl
from .session_manager_factories import readiness_coordinator_for_manager as _readiness_coordinator_for_manager_impl
from .session_manager_factories import recent_cwd_coordinator_for_manager as _recent_cwd_coordinator_for_manager_impl
from .session_manager_factories import refresh_coordinator_for_manager as _refresh_coordinator_for_manager_impl
from .session_manager_factories import send_coordinator_for_manager as _send_coordinator_for_manager_impl
from .session_manager_factories import unattended_config_coordinator_for_manager as _unattended_config_coordinator_for_manager_impl
from .session_manager_factories import unattended_sweep_coordinator_for_manager as _unattended_sweep_coordinator_for_manager_impl
from .session_manager_factories import ui_state_coordinator_for_manager as _ui_state_coordinator_for_manager_impl
from .session_manager_factories import voice_runtime_for_manager as _voice_runtime_for_manager_impl
from .session_manager_factories import web_launch_coordinator_for_manager as _web_launch_coordinator_for_manager_impl
from .session_manager_store import create_session_store as _create_session_store_impl
from .session_manager_store_attrs import store_backed_attr as _store_backed_attr
from .session_manager_store import session_store_for_manager as _session_store_for_manager_impl
from .session_manager_store import session_store_paths as _session_store_paths_impl
from .session_model import Session
from .session_runtime import clear_session_confirmed_send_boundary as _clear_session_confirmed_send_boundary
from .session_runtime import consume_session_confirmed_send_boundary as _consume_session_confirmed_send_boundary
from .session_runtime import log_path_size_or_none as _log_path_size_or_none
from .session_runtime import reset_session_log_caches as _reset_session_log_caches_impl
from .session_runtime import session_run_settings_from_meta as _session_run_settings_from_meta_impl
from .session_runtime import session_transport_from_meta as _session_transport_from_meta_impl
from .session_runtime import broker_allows_interrupted_idle_override as _runtime_broker_allows_interrupted_idle_override
from .session_runtime import broker_busy_queue as _runtime_broker_busy_queue
from .session_runtime import broker_interrupted_idle as _runtime_broker_interrupted_idle
from .session_runtime import broker_runtime_state as _runtime_broker_state
from .session_runtime import resolve_runtime_status as _resolve_runtime_status
from .session_runtime import select_runtime_token as _select_runtime_token
from .session_store import SessionStore
from .session_store import SessionStorePaths
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
from .server_metrics import metric_percentile as _metric_percentile_impl
from .server_metrics import metrics_snapshot as _metrics_snapshot_impl
from .server_metrics import record_metric as _record_metric_impl
from .server_route_deps import ServerRouteDepsFactory
from .server_routing import match_session_route as _match_session_route_impl
from .server_routing import normalize_url_prefix as _normalize_url_prefix_impl
from .server_routing import strip_url_prefix as _strip_url_prefix_impl
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
from .tmux_runtime import tmux_available as _tmux_available_impl
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
from .voice_push import VoicePushCoordinator


def _load_env_file(path: Path) -> dict[str, str]:
    return _load_env_file_impl(path)


def _normalize_url_prefix(raw: str | None) -> str:
    return _normalize_url_prefix_impl(raw)


def _match_session_route(path: str, *suffix: str) -> str | None:
    return _match_session_route_impl(path, *suffix)


def _strip_url_prefix(prefix: str, path: str) -> str | None:
    return _strip_url_prefix_impl(prefix, path)


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
    return _path_signature_impl(path)


def _file_write_lock(path: Path) -> Any:
    return _file_write_lock_impl(path, locks_lock=_FILE_WRITE_LOCKS_LOCK, locks=_FILE_WRITE_LOCKS)


def _record_metric(name: str, value_ms: float) -> None:
    return _record_metric_impl(
        name,
        value_ms,
        metrics_lock=_METRICS_LOCK,
        metrics=_METRICS,
        metrics_window=METRICS_WINDOW,
    )


def _metric_percentile(sorted_values: list[float], p: float) -> float:
    return _metric_percentile_impl(sorted_values, p)


def _metrics_snapshot() -> dict[str, dict[str, float | int]]:
    return _metrics_snapshot_impl(metrics_lock=_METRICS_LOCK, metrics=_METRICS)


def _wait_or_raise(proc: subprocess.Popen[bytes], *, label: str, timeout_s: float = 1.5) -> None:
    return _wait_or_raise_impl(proc, label=label, timeout_s=timeout_s)


def _drain_stream(f: Any) -> None:
    return _drain_stream_impl(f)


def _tmux_available() -> bool:
    global _TMUX_AVAILABLE_CACHE

    def _set_cache(value: tuple[float, bool]) -> None:
        global _TMUX_AVAILABLE_CACHE
        _TMUX_AVAILABLE_CACHE = value

    return _tmux_available_impl(
        ttl_seconds=TMUX_AVAILABLE_TTL_SECONDS,
        cache_lock=_TMUX_AVAILABLE_CACHE_LOCK,
        get_cache=lambda: _TMUX_AVAILABLE_CACHE,
        set_cache=_set_cache,
        which=shutil.which,
        now=time.time,
    )


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
    return _terminate_process_group_impl(
        root_pid,
        process_group_alive=_process_group_alive,
        now=_now,
        sleep=time.sleep,
        wait_seconds=wait_seconds,
    )


def _terminate_process(pid: int, *, wait_seconds: float = 1.0) -> bool:
    return _terminate_process_impl(
        pid,
        pid_alive=_pid_alive,
        now=_now,
        sleep=time.sleep,
        wait_seconds=wait_seconds,
    )


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
    return _resolve_under_impl(base, rel)


def _expanduser_path(path: Path) -> Path:
    return _expanduser_path_impl(path)


def _resolve_session_cwd(raw_cwd: str) -> Path:
    return _resolve_session_cwd_impl(raw_cwd)


def _resolve_session_path(base: Path, raw_path: str) -> Path:
    return _resolve_session_path_impl(base, raw_path)


def _require_existing_file(path: Path) -> Path:
    return _require_existing_file_impl(path)


def _resolve_existing_session_file(base: Path, raw_path: str) -> Path:
    return _resolve_existing_session_file_impl(base, raw_path)


def _resolve_existing_absolute_file(raw_path: str) -> Path:
    return _resolve_existing_absolute_file_impl(raw_path)



def _resolve_git_path(cwd: Path, raw_path: str) -> tuple[Path, Path, str]:
    return _git_ops.resolve_git_path(cwd, raw_path, run_git_func=_run_git, timeout_s=GIT_DIFF_TIMEOUT_SECONDS)

def _git_error_is_missing_head(message: str) -> bool:
    return _git_ops.git_error_is_missing_head(message)

def _git_head_blob_oid(cwd: Path, rel: str) -> str | None:
    return _git_ops.git_head_blob_oid(cwd, rel, run_git_func=_run_git, timeout_s=GIT_DIFF_TIMEOUT_SECONDS)

def _resolve_unique_bare_filename(search_root: Path, raw_path: str) -> Path | None:
    return _resolve_unique_bare_filename_impl(search_root, raw_path)


def _resolve_tracked_file_by_basename(session_id: str, raw_path: str) -> Path | None:
    return _resolve_tracked_file_by_basename_impl(
        session_id,
        raw_path,
        files_get=lambda tracked_session_id: MANAGER.files_get(tracked_session_id),
        expanduser_path=_expanduser_path,
    )


def _list_session_relative_files(base: Path) -> list[str]:
    return _list_session_relative_files_impl(base, expanduser_path=_expanduser_path)



def _run_git(cwd: Path, args: list[str], *, timeout_s: float, max_bytes: int, literal_pathspecs: bool = False) -> str:
    return _git_ops.run_git(cwd, args, timeout_s=timeout_s, max_bytes=max_bytes, literal_pathspecs=literal_pathspecs)

def _expand_user_path(raw: str) -> Path:
    return _expand_user_path_impl(raw)


def _resolve_existing_dir(raw: str, *, field_name: str) -> Path:
    return _resolve_existing_dir_impl(raw, field_name=field_name)


def _resolve_dir_target(raw: str, *, field_name: str) -> Path:
    return _resolve_dir_target_impl(raw, field_name=field_name)


def _codex_trust_override_for_path(path: Path) -> str:
    return _codex_trust_override_for_path_impl(path)


def _resolve_new_path(raw: str, *, field_name: str) -> Path:
    return _resolve_new_path_impl(raw, field_name=field_name)


def _clean_worktree_branch(raw: str) -> str:
    return _git_ops.clean_worktree_branch(raw)

def _require_git_repo(cwd: Path) -> None:
    _git_ops.require_git_repo(cwd, run_git_func=_run_git, timeout_s=GIT_DIFF_TIMEOUT_SECONDS)

def _git_repo_root(cwd: Path) -> Path | None:
    return _git_ops.git_repo_root(cwd, run_git_func=_run_git, timeout_s=GIT_DIFF_TIMEOUT_SECONDS)

def _search_session_relative_files(base: Path, *, query: str, limit: int = FILE_SEARCH_LIMIT) -> dict[str, Any]:
    return _search_session_relative_files_impl(base, query=query, limit=limit, git_root_func=_git_repo_root)


def _describe_session_cwd(cwd: Path) -> dict[str, Any]:
    return _describe_session_cwd_impl(cwd, git_repo_root=_git_repo_root, current_git_branch=_current_git_branch)


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
    return _path_resolves_inside_impl(path_obj, root)


def _symlink_payload_view(path_obj: Path) -> ClientFileView:
    return _symlink_payload_view_impl(path_obj)


def _resolve_git_client_file_view(*, session_id: str, raw_path: str) -> tuple[Path, str, ClientFileView]:
    return _resolve_git_client_file_view_impl(
        session_id=session_id,
        raw_path=raw_path,
        refresh_session_meta=MANAGER.refresh_session_meta,
        get_session=MANAGER.get_session,
        resolve_session_cwd=_resolve_session_cwd,
        resolve_git_path=_resolve_git_path,
        read_client_file_view=_read_client_file_view,
    )


def _resolve_git_existing_regular_file(*, session_id: str, raw_path: str) -> tuple[Path, str]:
    return _resolve_git_existing_regular_file_impl(
        session_id=session_id,
        raw_path=raw_path,
        refresh_session_meta=MANAGER.refresh_session_meta,
        get_session=MANAGER.get_session,
        resolve_session_cwd=_resolve_session_cwd,
        resolve_git_path=_resolve_git_path,
        require_existing_file=_require_existing_file,
    )


def _resolve_client_file_path(*, session_id: str, raw_path: str) -> Path:
    return _resolve_client_file_path_impl(
        session_id=session_id,
        raw_path=raw_path,
        refresh_session_meta=MANAGER.refresh_session_meta,
        get_session=MANAGER.get_session,
        files_get=lambda tracked_session_id: MANAGER.files_get(tracked_session_id),
        expanduser_path=_expanduser_path,
        resolve_session_cwd=_resolve_session_cwd,
        run_git=_run_git,
        git_timeout_s=GIT_DIFF_TIMEOUT_SECONDS,
    )



def _sessions_dir_for_backend(agent_backend: str) -> Path:
    return _sessions_dir_for_backend_impl(
        agent_backend,
        codex_sessions_dir=CODEX_SESSIONS_DIR,
        pi_sessions_dir=PI_SESSIONS_DIR,
        cc_sessions_dir=CC_SESSIONS_DIR,
    )


def _iter_session_logs(*, agent_backend: str = "codex") -> list[Path]:
    return _iter_session_logs_for_backend_impl(
        agent_backend=agent_backend,
        sessions_dir_for_backend_func=_sessions_dir_for_backend,
        iter_session_logs=_iter_session_logs_impl,
    )


def _find_session_log_for_session_id(session_id: str, *, agent_backend: str = "codex") -> Path | None:
    return _find_session_log_for_session_id_metadata_impl(
        session_id,
        agent_backend=agent_backend,
        sessions_dir_for_backend_func=_sessions_dir_for_backend,
        find_session_log_for_session_id_func=_find_session_log_for_session_id_impl,
    )


def _find_new_session_log(
    *,
    agent_backend: str = "codex",
    after_ts: float,
    preexisting: set[Path],
    timeout_s: float = 15.0,
) -> tuple[str, Path] | None:
    return _find_new_session_log_metadata_impl(
        agent_backend=agent_backend,
        after_ts=after_ts,
        preexisting=preexisting,
        timeout_s=timeout_s,
        sessions_dir_for_backend_func=_sessions_dir_for_backend,
        find_new_session_log_func=_find_new_session_log_impl,
    )



def _read_jsonl_from_offset(path: Path, offset: int, max_bytes: int = 2 * 1024 * 1024) -> tuple[list[dict[str, Any]], int]:
    return _read_jsonl_from_offset_impl(path, offset, max_bytes=max_bytes)


def _discover_log_for_session_id(session_id: str, *, agent_backend: str = "codex") -> Path | None:
    return _find_session_log_for_session_id(session_id, agent_backend=agent_backend)


def _read_session_meta(log_path: Path, *, agent_backend: str | None = None) -> dict[str, Any]:
    return _read_session_meta_impl(
        log_path,
        agent_backend=agent_backend,
        pi_sessions_dir=PI_SESSIONS_DIR,
        cc_sessions_dir=CC_SESSIONS_DIR,
        read_session_meta_payload=_read_session_meta_payload_impl,
    )


_INVALID_SESSION_META_WARNINGS: set[tuple[str, str]] = set()


def _read_session_meta_or_none(log_path: Path, *, agent_backend: str | None = None, context: str) -> dict[str, Any] | None:
    return _read_session_meta_or_none_impl(
        log_path,
        agent_backend=agent_backend,
        context=context,
        read_session_meta_func=_read_session_meta,
        invalid_warnings=_INVALID_SESSION_META_WARNINGS,
        stderr=sys.stderr,
    )


def _turn_context_run_settings(payload: Any) -> tuple[str | None, str | None]:
    return _turn_context_run_settings_impl(
        payload,
        clean_optional_text=_clean_optional_text,
        display_reasoning_effort=_display_reasoning_effort,
    )


def _read_run_settings_from_log(log_path: Path, *, agent_backend: str = "codex") -> tuple[str | None, str | None, str | None]:
    return _read_run_settings_from_log_impl(
        log_path,
        agent_backend=agent_backend,
        read_pi_run_settings=_read_pi_run_settings,
        read_cc_run_settings=_read_cc_run_settings,
        read_session_meta_or_none_func=_read_session_meta_or_none,
        clean_optional_text=_clean_optional_text,
        display_reasoning_effort=_display_reasoning_effort,
        find_latest_turn_context=_rollout_log._find_latest_turn_context,
    )



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
    return _launch_defaults_signature_impl(paths)


def _read_new_session_defaults() -> dict[str, Any]:
    global _LAUNCH_DEFAULTS_CACHE

    def _set_cache(value: tuple[tuple[tuple[str, bool, int | None, int | None], ...], dict[str, Any]] | None) -> None:
        global _LAUNCH_DEFAULTS_CACHE
        _LAUNCH_DEFAULTS_CACHE = value

    return _read_new_session_defaults_cached_impl(
        paths_provider=_launch_config_paths,
        defaults_reader=_launch_read_new_session_defaults,
        default_agent_backend=DEFAULT_AGENT_BACKEND,
        cache_lock=_LAUNCH_DEFAULTS_CACHE_LOCK,
        get_cache=lambda: _LAUNCH_DEFAULTS_CACHE,
        set_cache=_set_cache,
    )


def _codex_launch_defaults_for_request() -> dict[str, Any]:
    return _launch_defaults_for_request_impl(read_defaults=_read_codex_launch_defaults, fallback_defaults=_fallback_codex_launch_defaults)


def _pi_launch_defaults_for_request() -> dict[str, Any]:
    return _launch_defaults_for_request_impl(read_defaults=_read_pi_launch_defaults, fallback_defaults=_fallback_pi_launch_defaults)


def _parse_new_session_launch_request(obj: dict[str, Any]) -> NewSessionLaunchRequest:
    return _launch_parse_new_session_launch_request(
        obj,
        default_agent_backend=DEFAULT_AGENT_BACKEND,
        codex_launch_defaults_provider=_codex_launch_defaults_for_request,
        pi_launch_defaults_provider=_pi_launch_defaults_for_request,
    )

def _resume_candidate_from_log(log_path: Path, *, agent_backend: str = "codex") -> dict[str, Any] | None:
    return _resume_candidate_from_log_impl(
        log_path,
        agent_backend=agent_backend,
        read_session_meta=_read_session_meta,
        is_subagent_session_meta=_is_subagent_session_meta,
    )


def _list_resume_candidates_for_cwd(cwd: str, *, agent_backend: str = "codex", limit: int = 12) -> list[dict[str, Any]]:
    return _list_resume_candidates_for_cwd_impl(
        cwd,
        agent_backend=agent_backend,
        limit=limit,
        iter_session_logs=_iter_session_logs,
        resume_candidate_from_log_func=_resume_candidate_from_log,
    )


def _resume_preview_from_text(text: str, *, max_chars: int = 120) -> str:
    return _resume_preview_from_text_impl(text, max_chars=max_chars)


def _user_message_text(payload: dict[str, Any]) -> str:
    return _user_message_text_impl(payload)


def _is_scaffold_user_text(text: str) -> bool:
    return _is_scaffold_user_text_impl(text)


def _first_user_message_preview_from_log(log_path: Path, *, max_scan_bytes: int = 256 * 1024) -> str:
    return _first_user_message_preview_from_log_impl(
        log_path,
        pi_user_text=_pi_user_text,
        cc_user_text=_cc_user_text,
        max_scan_bytes=max_scan_bytes,
    )


def _coerce_main_thread_log(*, thread_id: str, log_path: Path) -> tuple[str, Path]:
    return _coerce_main_thread_log_impl(
        thread_id=thread_id,
        log_path=log_path,
        read_session_meta_or_none=_read_session_meta_or_none,
        is_subagent_session_meta=_is_subagent_session_meta,
        subagent_parent_thread_id=_subagent_parent_thread_id,
        find_session_log_for_session_id=lambda parent: _find_session_log_for_session_id_impl(CODEX_SESSIONS_DIR, parent),
    )



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


def _session_store_paths_for_manager() -> SessionStorePaths:
    return _session_store_paths_impl(
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


class SessionManager:
    _unattended = _store_backed_attr("unattended")
    _aliases = _store_backed_attr("aliases")
    _sidebar_meta = _store_backed_attr("sidebar_meta")
    _hidden_sessions = _store_backed_attr("hidden_sessions")
    _files = _store_backed_attr("files")
    _queues = _store_backed_attr("queues")
    _pending_attachment_ids = _store_backed_attr("pending_attachment_ids")
    _commit_unknown_sends = _store_backed_attr("commit_unknown_sends")
    _recent_cwds = _store_backed_attr("recent_cwds")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._stop = threading.Event()
        self._last_discover_ts = 0.0
        self._store = self._new_session_store_for_manager(_session_store_paths_for_manager())
        _seed_manager_in_memory_state_impl(self)
        _load_manager_persistent_state_impl(self)
        self._voice_push = _create_voice_push_coordinator_impl(
            voice_push_factory=VoicePushCoordinator,
            app_dir=APP_DIR,
            stop_event=self._stop,
            settings_path=VOICE_SETTINGS_PATH,
            subscriptions_path=PUSH_SUBSCRIPTIONS_PATH,
            delivery_ledger_path=DELIVERY_LEDGER_PATH,
            vapid_private_key_path=VAPID_PRIVATE_KEY_PATH,
        )
        self._discover_existing(force=True)
        self._prune_missing_commit_unknown_sends()
        _start_manager_worker_threads_impl(manager=self, thread_factory=threading.Thread)

    def stop(self) -> None:
        self._stop.set()

    def _reset_log_caches(self, s: Session, *, meta_log_off: int) -> None:
        return _reset_session_log_caches_impl(s, meta_log_off=meta_log_off)

    def _session_run_settings(self, *, meta: dict[str, Any], log_path: Path | None, agent_backend: str) -> tuple[str | None, str | None, str | None, str | None]:
        return _session_run_settings_from_meta_impl(
            meta=meta,
            log_path=log_path,
            agent_backend=agent_backend,
            clean_optional_text=_clean_optional_text,
            normalize_requested_preferred_auth_method=_normalize_requested_preferred_auth_method,
            display_reasoning_effort=_display_reasoning_effort,
            display_pi_reasoning_effort=_display_pi_reasoning_effort,
            normalize_requested_cc_reasoning_effort=_normalize_requested_cc_reasoning_effort,
            read_run_settings_from_log=_read_run_settings_from_log,
        )

    def _session_transport(self, *, meta: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
        return _session_transport_from_meta_impl(meta=meta, clean_optional_text=_clean_optional_text)

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

    def _new_session_store_for_manager(self, paths: SessionStorePaths) -> SessionStore:
        return _create_session_store_impl(
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

    def _session_store_for_manager(self) -> SessionStore:
        store = _session_store_for_manager_impl(
            existing=getattr(self, "_store", None),
            paths=_session_store_paths_for_manager(),
            create_store=self._new_session_store_for_manager,
        )
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
        return _queue_coordinator_for_manager_impl(self, sys.modules[__name__])

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
        return _discovery_deps_for_manager_impl(self, sys.modules[__name__])

    def _apply_discovery_result(self, result: DiscoveryResult) -> None:
        return self._discovery_registry_for_manager().apply_result(result)

    def _upsert_discovery_registration(self, registration: DiscoveryRegistration) -> None:
        return self._discovery_registry_for_manager().upsert_registration(registration)

    def _discover_existing(self, *, force: bool = False) -> None:
        return _discover_existing_for_manager_impl(
            self,
            force=force,
            discover_min_interval_seconds=DISCOVER_MIN_INTERVAL_SECONDS,
            sock_dir=SOCK_DIR,
            proc_root=PROC_ROOT,
            discover_sessions=_discover_sessions,
            now=time.time,
        )

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
        return _control_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _attachment_coordinator_for_manager(self) -> SessionAttachmentCoordinator:
        return _attachment_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _list_coordinator_for_manager(self) -> SessionListCoordinator:
        return _list_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _refresh_coordinator_for_manager(self) -> SessionRefreshCoordinator:
        return _refresh_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _readiness_coordinator_for_manager(self) -> SessionReadinessCoordinator:
        return _readiness_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _unattended_sweep_coordinator_for_manager(self) -> UnattendedSweepCoordinator:
        return _unattended_sweep_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _queue_sweep_coordinator_for_manager(self) -> QueueSweepCoordinator:
        return _queue_sweep_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _voice_runtime_for_manager(self) -> VoiceRuntimeCoordinator:
        return _voice_runtime_for_manager_impl(self, sys.modules[__name__])

    def _log_runtime_for_manager(self) -> SessionLogRuntimeCoordinator:
        return _log_runtime_for_manager_impl(self, sys.modules[__name__])

    def _files_coordinator_for_manager(self) -> SessionFilesCoordinator:
        return _files_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _ui_state_coordinator_for_manager(self) -> SessionUiStateCoordinator:
        return _ui_state_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _unattended_config_coordinator_for_manager(self) -> SessionUnattendedConfigCoordinator:
        return _unattended_config_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _cleanup_coordinator_for_manager(self) -> SessionCleanupCoordinator:
        return _cleanup_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _pending_state_coordinator_for_manager(self) -> SessionPendingStateCoordinator:
        return _pending_state_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _recent_cwd_coordinator_for_manager(self) -> SessionRecentCwdCoordinator:
        return _recent_cwd_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _lifecycle_coordinator_for_manager(self) -> SessionLifecycleCoordinator:
        return _lifecycle_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _discovery_registry_for_manager(self) -> SessionDiscoveryRegistryCoordinator:
        return _discovery_registry_for_manager_impl(self, sys.modules[__name__])

    def _prune_coordinator_for_manager(self) -> SessionPruneCoordinator:
        return _prune_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _send_coordinator_for_manager(self) -> SessionSendCoordinator:
        return _send_coordinator_for_manager_impl(self, sys.modules[__name__])

    def _prelog_user_message_recorder_for_manager(self) -> PrelogUserMessageRecorder:
        return _prelog_user_message_recorder_for_manager_impl(self, sys.modules[__name__])

    def _kill_session_via_pids(self, s: Session) -> bool:
        return self._lifecycle_coordinator_for_manager().kill_session_via_pids(s)

    def kill_session(self, session_id: str) -> bool:
        return self._lifecycle_coordinator_for_manager().kill_session(session_id)

    def _live_session_for_resume_target(self, resume_id: str, resume_row: dict[str, Any] | None) -> Session | None:
        return self._lifecycle_coordinator_for_manager().live_session_for_resume_target(resume_id, resume_row)

    def _web_launch_coordinator_for_manager(self) -> SessionWebLaunchCoordinator:
        return _web_launch_coordinator_for_manager_impl(self, sys.modules[__name__])

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
