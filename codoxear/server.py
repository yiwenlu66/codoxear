#!/usr/bin/env python3
from __future__ import annotations

import base64
import errno
import hashlib
import hmac
import http.server
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

from .agent_backend import normalize_agent_backend
from .auth import CookieAuthSettings
from .auth import load_or_create_hmac_secret as _load_or_create_hmac_secret_impl
from .auth import require_auth as _require_auth_impl
from .auth import set_auth_cookie as _set_auth_cookie_impl
from .auth import sign_cookie as _sign_cookie_impl
from .auth import verify_cookie as _verify_cookie_impl
from .client_file_paths import describe_session_cwd as _describe_session_cwd_impl
from .client_file_paths import list_session_relative_files as _list_session_relative_files_impl
from .client_file_paths import list_session_relative_file_entries as _list_session_relative_file_entries_impl
from .client_file_paths import resolve_client_file_path as _resolve_client_file_path_impl
from .client_file_paths import resolve_git_client_file_view as _resolve_git_client_file_view_impl
from .client_file_paths import resolve_git_existing_regular_file as _resolve_git_existing_regular_file_impl
from .client_file_paths import resolve_unique_bare_filename as _resolve_unique_bare_filename_impl
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
from .file_text import read_regular_file_prefix_no_symlink as _read_regular_file_prefix_no_symlink
from .file_text import read_text_file_strict as _read_text_file_strict
from .file_types import file_kind as _file_kind
from .file_upload import stage_uploaded_file as _stage_uploaded_file_impl
from . import git_ops as _git_ops
from .launch_config import LaunchConfigPaths
from .launch_config import LaunchRequestValidationError
from .launch_config import NewSessionLaunchRequest
from .launch_config import SUPPORTED_CC_REASONING_EFFORTS
from .launch_config import SUPPORTED_PI_REASONING_EFFORTS
from .launch_config import SUPPORTED_REASONING_EFFORTS
from .launch_config import display_pi_reasoning_effort as _launch_display_pi_reasoning_effort
from .launch_config import display_reasoning_effort as _launch_display_reasoning_effort
from .launch_config import fallback_codex_launch_defaults as _launch_fallback_codex_launch_defaults
from .launch_config import fallback_pi_launch_defaults as _launch_fallback_pi_launch_defaults
from .launch_config import normalize_requested_cc_reasoning_effort as _launch_normalize_requested_cc_reasoning_effort
from .launch_config import normalize_requested_model_provider as _launch_normalize_requested_model_provider
from .launch_config import normalize_requested_pi_reasoning_effort as _launch_normalize_requested_pi_reasoning_effort
from .launch_config import normalize_requested_preferred_auth_method as _launch_normalize_requested_preferred_auth_method
from .launch_config import normalize_requested_service_tier as _launch_normalize_requested_service_tier
from .launch_config import parse_new_session_launch_request as _launch_parse_new_session_launch_request
from .launch_config import provider_choice_for_settings as _launch_provider_choice_for_settings
from .launch_config import read_cc_launch_defaults as _launch_read_cc_launch_defaults
from .launch_config import read_codex_launch_defaults as _launch_read_codex_launch_defaults
from .launch_config import read_new_session_defaults as _launch_read_new_session_defaults
from .launch_config import read_pi_launch_defaults as _launch_read_pi_launch_defaults
from .launch_config import read_pi_reasoning_efforts_by_model as _launch_read_pi_reasoning_efforts_by_model
from .launch_ledger import launch_attempt_row as _launch_attempt_row_impl
from .launch_ledger import launch_attempt_transcript_for_session_id as _launch_attempt_transcript_for_session_id_impl
from .launch_ledger import launch_attempt_transcript_payload as _launch_attempt_transcript_payload_impl
from .launch_ledger import launch_failure_tail as _launch_failure_tail_impl
from .launch_ledger import latest_launch_attempt as _latest_launch_attempt_impl
from .launch_ledger import record_launch_attempt as _record_launch_attempt_impl
from .launch_ledger import submitted_user_messages as _submitted_user_messages_impl
from .launch_path_runtime import codex_trust_override_for_path as _codex_trust_override_for_path_impl
from .launch_path_runtime import load_env_file as _load_env_file_impl
from .launch_path_runtime import resolve_dir_target as _resolve_dir_target_impl
from .launch_defaults_runtime import launch_defaults_for_request as _launch_defaults_for_request_impl
from .launch_defaults_runtime import read_new_session_defaults_cached as _read_new_session_defaults_cached_impl
from .file_view import ClientFileView
from .file_view import download_disposition as _download_disposition
from .file_view import inspect_client_path as _inspect_client_path
from .file_view import inspect_downloadable_file as _inspect_downloadable_file
from .file_view import inspect_openable_file as _inspect_openable_file
from .file_view import read_client_file_view as _read_client_file_view
from .file_view import read_text_or_image as _read_text_or_image
from .video_preview import ensure_video_preview as _ensure_video_preview_impl
from .cc_log import cc_user_text as _cc_user_text
from .cc_log import read_cc_run_settings as _read_cc_run_settings
from .message_cursor import MessageCursorError
from .message_cursor import attach_history_cursors as _attach_history_cursors_impl
from .message_cursor import decode_message_cursor as _decode_message_cursor_impl
from .message_cursor import encode_message_cursor as _encode_message_cursor_impl
from .transcript_search import TRANSCRIPT_SEARCH_MAX_LINE_BYTES
from .transcript_search import clip_search_match_text as _clip_search_match_text
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
from .session_cleaners import clean_alias as _clean_alias_impl
from .session_cleaners import clean_dependency_session_id as _clean_dependency_session_id_impl
from .session_cleaners import clean_optional_text as _clean_optional_text_impl
from .session_cleaners import clean_priority_offset as _clean_priority_offset_impl
from .session_cleaners import clean_recent_cwd as _clean_recent_cwd_impl
from .session_cleaners import clean_snooze_until as _clean_snooze_until_impl
from .session_discovery import discover_sessions as _discover_sessions
from .session_errors import SessionCommitUnknownError
from .session_errors import SessionInjectionError
from .session_errors import SessionLaunchError
from .session_errors import SessionNotReadyError
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
from .session_resume import list_resume_candidates_for_cwd as _list_resume_candidates_for_cwd_impl
from .session_refresh import broker_tail_has_session_detach_marker as _broker_tail_has_session_detach_marker
from .session_resume import resume_candidate_from_log as _resume_candidate_from_log_impl
from .session_listing import clip01 as _listing_clip01
from .session_listing import sidebar_time_priority_from_elapsed_seconds as _listing_sidebar_time_priority_from_elapsed_seconds
from .session_manager_bootstrap import create_voice_push_coordinator as _create_voice_push_coordinator_impl
from .session_manager_bootstrap import input_lock_for_session as _input_lock_for_session_impl
from .session_manager_bootstrap import load_manager_persistent_state as _load_manager_persistent_state_impl
from .session_manager_bootstrap import queue_loop as _queue_loop_impl
from .session_manager_bootstrap import seed_manager_in_memory_state as _seed_manager_in_memory_state_impl
from .session_manager_bootstrap import start_manager_worker_threads as _start_manager_worker_threads_impl
from .session_manager_bootstrap import unattended_loop as _unattended_loop_impl
from .session_manager_bootstrap import voice_push_scan_loop as _voice_push_scan_loop_impl
from .session_manager_discovery import discover_existing_for_manager as _discover_existing_for_manager_impl
from .session_manager_discovery import discover_existing_if_stale_for_manager as _discover_existing_if_stale_for_manager_impl
from .session_manager_factories import session_manager_factory_caps as _session_manager_factory_caps_impl
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
from .session_manager_method_bindings import bind_session_manager_methods as _bind_session_manager_methods
from .session_manager_store import create_session_store as _create_session_store_impl
from .session_manager_store_attrs import load_store_attr as _load_store_attr
from .session_manager_store_attrs import save_dict_store_attr as _save_dict_store_attr
from .session_manager_store_attrs import save_pending_attachment_ids_attr as _save_pending_attachment_ids_attr
from .session_manager_store_attrs import save_set_store_attr as _save_set_store_attr
from .session_manager_store_attrs import store_backed_attr as _store_backed_attr
from .session_manager_store import session_store_for_manager as _session_store_for_manager_impl
from .session_manager_store import session_store_paths as _session_store_paths_impl
from .session_model import Session
from .session_registry import registry_backed_attr as _registry_backed_attr
from .session_registry import session_registry_for_manager as _session_registry_for_manager
from .session_runtime import clear_session_confirmed_send_boundary as _clear_session_confirmed_send_boundary
from .session_runtime import consume_session_confirmed_send_boundary as _consume_session_confirmed_send_boundary
from .session_runtime import log_path_size_or_none as _log_path_size_or_none
from .session_runtime import reset_session_log_caches as _reset_session_log_caches_impl
from .session_runtime import session_run_settings_from_meta as _session_run_settings_from_meta_impl
from .session_runtime import session_transport_from_meta as _session_transport_from_meta_impl
from .session_runtime import broker_busy_queue as _runtime_broker_busy_queue
from .session_runtime import broker_interrupted_idle as _runtime_broker_interrupted_idle
from .session_runtime import broker_runtime_state as _runtime_broker_state
from .session_runtime import resolve_runtime_status as _resolve_runtime_status
from .session_runtime import select_runtime_token as _select_runtime_token
from .server_handler import make_server_handler
from .server_http import BadRequestError
from .server_http import RequestPayloadTooLargeError
from .server_http import handle_route_exception as _handle_route_exception_impl
from .server_http import is_client_disconnect as _is_client_disconnect_impl
from .server_http import json_response as _json_response_impl
from .server_http import json_response_with_etag as _json_response_with_etag_impl
from .server_http import read_body as _read_body_impl
from .server_main import ThreadingHTTPServer
from .server_main import ThreadingHTTPServerV6
from .server_main import run_main as _run_server_main
from .server_metrics import metrics_snapshot as _metrics_snapshot_impl
from .server_metrics import record_metric as _record_metric_impl
from .server_route_deps import ServerRouteDepsFactory
from .server_route_deps import server_route_caps as _server_route_caps_impl
from .server_config import build_server_config as _build_server_config
from .server_config import export_server_config as _export_server_config
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
from .util import append_launch_attempt as _append_launch_attempt
from .util import find_new_session_log as _find_new_session_log_impl
from .util import find_session_log_for_session_id as _find_session_log_for_session_id_impl
from .util import is_subagent_session_meta as _is_subagent_session_meta
from .util import iter_session_logs as _iter_session_logs_impl
from .util import now as _now
from .sidecar_metadata import log_invalid as _log_invalid_sidecar_metadata
from .util import pid_alive as _pid_alive
from .util import process_group_alive as _process_group_alive
from .util import proc_find_open_rollout_log as _proc_find_open_rollout_log
from .util import read_launch_attempts as _read_launch_attempts
from .util import redact_launch_failure_text as _redact_launch_failure_text
from .util import redacted_launch_attempt_persist_record as _redacted_launch_attempt_persist_record
from .util import read_jsonl_from_offset as _read_jsonl_from_offset_impl
from .util import read_session_meta_payload as _read_session_meta_payload_impl
from .util import session_id_from_rollout_path as _session_id_from_rollout_path
from .util import subagent_parent_thread_id as _subagent_parent_thread_id
from .unattended import UNATTENDED_PROMPT_PREFIX as _UNATTENDED_PROMPT_PREFIX
from .unattended import UnattendedStore
from .unattended import clean_unattended_cooldown_minutes as _clean_unattended_cooldown_minutes_impl
from .unattended import clean_unattended_remaining_injections as _clean_unattended_remaining_injections_impl
from .voice_push import VoicePushCoordinator


_load_env_file = _load_env_file_impl
_normalize_url_prefix = _normalize_url_prefix_impl
_match_session_route = _match_session_route_impl
_strip_url_prefix = _strip_url_prefix_impl


_SERVER_CONFIG = _build_server_config()
_export_server_config(globals(), _SERVER_CONFIG)

def _static_cache_control_headers(*, enabled: bool = STATIC_CACHE_ENABLED) -> dict[str, str]:
    return _static_cache_control_headers_impl(enabled=enabled)


UNATTENDED_PROMPT_PREFIX = _UNATTENDED_PROMPT_PREFIX


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


def _metrics_snapshot() -> dict[str, dict[str, float | int]]:
    return _metrics_snapshot_impl(metrics_lock=_METRICS_LOCK, metrics=_METRICS)


def _wait_or_raise(proc: subprocess.Popen[bytes], *, label: str, timeout_s: float = 1.5) -> None:
    return _wait_or_raise_impl(proc, label=label, timeout_s=timeout_s)


_drain_stream = _drain_stream_impl


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


def _latest_launch_attempt(launch_id: str) -> dict[str, Any] | None:
    return _latest_launch_attempt_impl(launch_id, path=LAUNCH_ATTEMPTS_PATH)


_submitted_user_messages = _submitted_user_messages_impl
_launch_failure_tail = _launch_failure_tail_impl
_launch_attempt_transcript_payload = _launch_attempt_transcript_payload_impl


def _launch_attempt_transcript_for_session_id(session_id: str, *, max_bytes: int | None = None) -> dict[str, Any] | None:
    return _launch_attempt_transcript_for_session_id_impl(
        session_id,
        path=LAUNCH_ATTEMPTS_PATH,
        default_agent_backend=DEFAULT_AGENT_BACKEND,
        unattended_default_idle_minutes=UNATTENDED_DEFAULT_IDLE_MINUTES,
        unattended_default_max_injections=UNATTENDED_DEFAULT_MAX_INJECTIONS,
        max_bytes=max_bytes,
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


def _ensure_video_preview(path: Path) -> Path:
    return _ensure_video_preview_impl(path, preview_dir=VIDEO_PREVIEW_DIR)


_is_client_disconnect = _is_client_disconnect_impl


def _handle_route_exception(handler: http.server.BaseHTTPRequestHandler, exc: BaseException) -> None:
    return _handle_route_exception_impl(handler, exc, json_response=_json_response)


def _json_response(handler: http.server.BaseHTTPRequestHandler, status: int, obj: Any) -> None:
    return _json_response_impl(handler, status, obj, set_auth_cookie=_set_auth_cookie)


def _json_response_with_etag(handler: http.server.BaseHTTPRequestHandler, obj: Any) -> None:
    return _json_response_with_etag_impl(handler, obj, sha256_hex=_sha256_hex, set_auth_cookie=_set_auth_cookie)


_read_body = _read_body_impl


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




def _encode_message_cursor(*, kind: str, session: "Session", pos: int) -> str:
    return _encode_message_cursor_impl(kind=kind, session=session, pos=pos, secret=HMAC_SECRET)


def _decode_message_cursor(token: str, *, kind: str, session: "Session") -> int:
    return _decode_message_cursor_impl(token, kind=kind, session=session, secret=HMAC_SECRET)


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


_resolve_under = _resolve_under_impl
_expanduser_path = _expanduser_path_impl
_resolve_session_cwd = _resolve_session_cwd_impl
_resolve_session_path = _resolve_session_path_impl
_require_existing_file = _require_existing_file_impl
_resolve_existing_session_file = _resolve_existing_session_file_impl
_resolve_existing_absolute_file = _resolve_existing_absolute_file_impl


def _resolve_git_path(cwd: Path, raw_path: str) -> tuple[Path, Path, str]:
    return _git_ops.resolve_git_path(cwd, raw_path, run_git_func=_run_git, timeout_s=GIT_DIFF_TIMEOUT_SECONDS)


def _git_head_blob_oid(cwd: Path, rel: str) -> str | None:
    return _git_ops.git_head_blob_oid(cwd, rel, run_git_func=_run_git, timeout_s=GIT_DIFF_TIMEOUT_SECONDS)

_resolve_unique_bare_filename = _resolve_unique_bare_filename_impl


def _list_session_relative_files(base: Path) -> list[str]:
    return _list_session_relative_files_impl(base, expanduser_path=_expanduser_path)


def _list_session_relative_file_entries(base: Path) -> list[dict[str, Any]]:
    return _list_session_relative_file_entries_impl(base, expanduser_path=_expanduser_path)


def _run_git(
    cwd: Path,
    args: list[str],
    *,
    timeout_s: float,
    max_bytes: int,
    literal_pathspecs: bool = False,
    decode_errors: str = "replace",
) -> str:
    return _git_ops.run_git(
        cwd,
        args,
        timeout_s=timeout_s,
        max_bytes=max_bytes,
        literal_pathspecs=literal_pathspecs,
        decode_errors=decode_errors,
    )


_resolve_dir_target = _resolve_dir_target_impl
_codex_trust_override_for_path = _codex_trust_override_for_path_impl


def _require_git_repo(cwd: Path) -> None:
    _git_ops.require_git_repo(cwd, run_git_func=_run_git, timeout_s=GIT_DIFF_TIMEOUT_SECONDS)

def _git_repo_root(cwd: Path) -> Path | None:
    return _git_ops.git_repo_root(cwd, run_git_func=_run_git, timeout_s=GIT_DIFF_TIMEOUT_SECONDS)

def _search_session_relative_files(base: Path, *, query: str, limit: int = FILE_SEARCH_LIMIT) -> dict[str, Any]:
    return _search_session_relative_files_impl(base, query=query, limit=limit, git_root_func=_git_repo_root)


def _describe_session_cwd(cwd: Path) -> dict[str, Any]:
    return _describe_session_cwd_impl(cwd, git_repo_root=_git_repo_root, current_git_branch=_current_git_branch)


_default_worktree_path = _git_ops.default_worktree_path

def _create_git_worktree(source_cwd: Path, worktree_branch: str) -> Path:
    return _git_ops.create_git_worktree(source_cwd, worktree_branch, git_repo_root_func=_git_repo_root, timeout_s=GIT_WORKTREE_TIMEOUT_SECONDS)

_split_git_nul_paths = _git_ops.split_git_nul_paths
_parse_git_numstat = _git_ops.parse_git_numstat

def _stage_uploaded_file(session_id: str, filename: str, raw: bytes, *, max_bytes: int = ATTACH_UPLOAD_MAX_BYTES) -> Path:
    return _stage_uploaded_file_impl(
        session_id,
        filename,
        raw,
        upload_dir=UPLOAD_DIR,
        now_fn=_now,
        max_bytes=max_bytes,
    )


_clean_alias = _clean_alias_impl
_clean_recent_cwd = _clean_recent_cwd_impl
_clip01 = _listing_clip01
_clean_priority_offset = _clean_priority_offset_impl
_clean_snooze_until = _clean_snooze_until_impl
_clean_dependency_session_id = _clean_dependency_session_id_impl
_clean_optional_text = _clean_optional_text_impl


def _launch_config_paths() -> LaunchConfigPaths:
    return LaunchConfigPaths(
        codex_config_path=CODEX_CONFIG_PATH,
        models_cache_path=MODELS_CACHE_PATH,
        pi_settings_path=PI_SETTINGS_PATH,
        pi_models_path=PI_MODELS_PATH,
        pi_auth_path=PI_AUTH_PATH,
        cc_settings_path=CC_SETTINGS_PATH,
    )


_display_reasoning_effort = _launch_display_reasoning_effort
_display_pi_reasoning_effort = _launch_display_pi_reasoning_effort


def _read_pi_reasoning_efforts_by_model() -> dict[str, list[str]]:
    return _launch_read_pi_reasoning_efforts_by_model(_launch_config_paths())


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


_normalize_requested_cc_reasoning_effort = _launch_normalize_requested_cc_reasoning_effort
_normalize_requested_model_provider = _launch_normalize_requested_model_provider
_normalize_requested_service_tier = _launch_normalize_requested_service_tier
_normalize_requested_preferred_auth_method = _launch_normalize_requested_preferred_auth_method
_provider_choice_for_settings = _launch_provider_choice_for_settings


def _sidebar_time_priority_from_elapsed_seconds(elapsed_s: float) -> float:
    return _listing_sidebar_time_priority_from_elapsed_seconds(
        elapsed_s,
        half_life_seconds=SIDEBAR_PRIORITY_HALF_LIFE_SECONDS,
        bucket_seconds=SIDEBAR_PRIORITY_BUCKET_SECONDS,
    )


def _current_git_branch(cwd: Path) -> str | None:
    return _git_ops.current_git_branch(cwd, run_git_func=_run_git, timeout_s=GIT_DIFF_TIMEOUT_SECONDS)


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


_fallback_codex_launch_defaults = _launch_fallback_codex_launch_defaults
_fallback_pi_launch_defaults = _launch_fallback_pi_launch_defaults


def _read_codex_launch_defaults() -> dict[str, Any]:
    return _launch_read_codex_launch_defaults(_launch_config_paths())


def _read_pi_launch_defaults() -> dict[str, Any]:
    return _launch_read_pi_launch_defaults(_launch_config_paths())


def _read_cc_launch_defaults() -> dict[str, Any]:
    return _launch_read_cc_launch_defaults(_launch_config_paths())


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


_extract_chat_events = _rollout_log._extract_chat_events
_extract_delivery_messages = _rollout_log._extract_delivery_messages
_event_ts = _rollout_log._event_ts
_has_assistant_output_text = _rollout_log._has_assistant_output_text
_analyze_log_chunk = _rollout_log._analyze_log_chunk
_last_conversation_ts_from_tail = _rollout_log._last_conversation_ts_from_tail
_compute_idle_from_log = _rollout_log._compute_idle_from_log
_last_chat_role_ts_from_tail = _rollout_log._last_chat_role_ts_from_tail


def _read_jsonl_records_from_offset(
    path: Path,
    offset: int,
    *,
    max_bytes: int = 2 * 1024 * 1024,
) -> tuple[list[Any], int]:
    return _rollout_log._read_jsonl_records_from_offset(path, offset, max_bytes=max_bytes)


_broker_busy_queue_from_state = _runtime_broker_busy_queue
_broker_interrupted_idle_from_state = _runtime_broker_interrupted_idle




@_bind_session_manager_methods(__name__)
class SessionManager:
    _lock = _registry_backed_attr("lock")
    _sessions = _registry_backed_attr("sessions")
    _stop = _registry_backed_attr("stop_event")
    _last_discover_ts = _registry_backed_attr("last_discover_ts")
    _input_locks = _registry_backed_attr("input_locks")
    _store = _registry_backed_attr("store")
    _unattended = _store_backed_attr("unattended")
    _aliases = _store_backed_attr("aliases")
    _sidebar_meta = _store_backed_attr("sidebar_meta")
    _hidden_sessions = _store_backed_attr("hidden_sessions")
    _files = _store_backed_attr("files")
    _queues = _store_backed_attr("queues")
    _pending_attachment_ids = _store_backed_attr("pending_attachment_ids")
    _staged_attachments = _store_backed_attr("staged_attachments")
    _commit_unknown_sends = _store_backed_attr("commit_unknown_sends")
    _recent_cwds = _store_backed_attr("recent_cwds")
    _load_unattended = _load_store_attr("_unattended", "load_unattended")
    _save_unattended = _save_dict_store_attr("_unattended", "save_unattended")
    _load_aliases = _load_store_attr("_aliases", "load_aliases")
    _save_aliases = _save_dict_store_attr("_aliases", "save_aliases")
    _load_sidebar_meta = _load_store_attr("_sidebar_meta", "load_sidebar_meta")
    _save_sidebar_meta = _save_dict_store_attr("_sidebar_meta", "save_sidebar_meta")
    _load_hidden_sessions = _load_store_attr("_hidden_sessions", "load_hidden_sessions")
    _save_hidden_sessions = _save_set_store_attr("_hidden_sessions", "save_hidden_sessions")
    _load_files = _load_store_attr("_files", "load_files")
    _save_files = _save_dict_store_attr("_files", "save_files")
    _load_queues = _load_store_attr("_queues", "load_queues")
    _save_queues = _save_dict_store_attr("_queues", "save_queues")
    _load_pending_attachments = _load_store_attr("_pending_attachment_ids", "load_pending_attachments")
    _save_pending_attachments = _save_pending_attachment_ids_attr("_pending_attachment_ids", "save_pending_attachments")
    _load_staged_attachments = _load_store_attr("_staged_attachments", "load_staged_attachments")
    _save_staged_attachments = _save_dict_store_attr("_staged_attachments", "save_staged_attachments")
    _load_commit_unknown_sends = _load_store_attr("_commit_unknown_sends", "load_commit_unknown_sends")
    _save_commit_unknown_sends = _save_dict_store_attr("_commit_unknown_sends", "save_commit_unknown_sends")
    _load_recent_cwds = _load_store_attr("_recent_cwds", "load_recent_cwds")
    _save_recent_cwds = _save_dict_store_attr("_recent_cwds", "save_recent_cwds")

    def refresh_session_meta(self, session_id: str, *, drain_queue: bool = False) -> None:
        return self._refresh_coordinator_for_manager().refresh_session_meta(session_id, drain_queue=drain_queue)

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

    def send(self, session_id: str, text: str, *, allow_pending_attachment: bool = False, queue_item_id: str | None = None) -> dict[str, Any]:
        return self._send_coordinator_for_manager().send(
            session_id,
            text,
            allow_pending_attachment=allow_pending_attachment,
            queue_item_id=queue_item_id,
        )

    def _refresh_session_meta_if_sidecar_exists(self, session_id: str, *, drain_queue: bool = False) -> None:
        registry = _session_registry_for_manager(self)
        with registry.lock:
            s = registry.sessions.get(session_id)
            if not s:
                raise KeyError("unknown session")
            meta_path = s.sock_path.with_suffix(".json")
        if meta_path.exists():
            self.refresh_session_meta(session_id, drain_queue=drain_queue)

    def inject_keys(self, session_id: str, seq: str, *, track_request_sent: bool = False, interrupt: bool = False) -> dict[str, Any]:
        return self._control_coordinator_for_manager().inject_keys(
            session_id,
            seq,
            track_request_sent=track_request_sent,
            interrupt=interrupt,
        )


MANAGER = SessionManager()

def _read_static_bytes(path: Path) -> bytes:
    return _read_static_bytes_impl(path, attach_upload_max_bytes=ATTACH_UPLOAD_MAX_BYTES)

def _route_deps_factory() -> ServerRouteDepsFactory:
    server_module = sys.modules[__name__]
    return ServerRouteDepsFactory(_server_route_caps_impl(server_module))

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
