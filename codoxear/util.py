from __future__ import annotations

import datetime
import errno
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .agent_backend import get_agent_backend
from .agent_backend import infer_agent_backend_from_log_path
from .agent_backend import normalize_agent_backend
from .cc_log import read_cc_session_header
from .cc_log import read_cc_session_id
from .json_state import atomic_write_json
from .json_state import load_json_file
from .jsonl_offset import read_jsonl_from_offset as _read_jsonl_from_offset_impl
from .launch_attempt_store import _LAUNCH_ERROR_RESPONSE_FIELDS
from .launch_attempt_store import _SENSITIVE_LAUNCH_FIELD_RE
from .launch_attempt_store import _jsonable
from .launch_attempt_store import append_launch_attempt as _append_launch_attempt_impl
from .launch_attempt_store import read_launch_attempts as _read_launch_attempts_impl
from .launch_attempt_store import redact_launch_failure_text
from .launch_attempt_store import redact_launch_failure_value
from .launch_attempt_store import redacted_launch_attempt_persist_record
from .launch_attempt_store import redacted_launch_attempt_response_record
from .process_log_paths import _fd_has_write_intent
from .process_log_paths import _macos_children
from .process_log_paths import _macos_descendants
from .process_log_paths import _macos_open_rollout_logs
from .process_log_paths import _proc_children
from .process_log_paths import _proc_descendants
from .process_log_paths import _proc_fd_flags
from .process_log_paths import _proc_pid_uid
from .process_log_paths import proc_open_rollout_logs
from .process_log_paths import proc_open_rollout_logs_for_backend
from .process_log_paths import proc_open_writable_rollout_logs
from .process_log_paths import proc_open_writable_rollout_logs_for_backend
from .session_log_paths import _is_cc_session_log_path
from .session_log_paths import _is_codex_rollout_log_path
from .session_log_paths import _is_pi_session_log_path
from .session_log_paths import _path_in_set
from .session_log_paths import _paths_match
from .session_log_paths import _payload_cwd_matches
from .session_log_paths import session_id_from_rollout_path
from .session_log_discovery import _read_session_meta_payload_once as _read_session_meta_payload_once_impl
from .session_log_discovery import classify_session_log as _classify_session_log_impl
from .session_log_discovery import find_new_session_log as _find_new_session_log_impl
from .session_log_discovery import find_session_log_for_session_id as _find_session_log_for_session_id_impl
from .session_log_discovery import is_subagent_session_meta as _is_subagent_session_meta_impl
from .session_log_discovery import iter_session_logs as _iter_session_logs_impl
from .session_log_discovery import read_session_meta_payload as _read_session_meta_payload_impl
from .session_log_discovery import subagent_parent_thread_id as _subagent_parent_thread_id_impl
from .pi_log import read_pi_log_cwd


_LEGACY_WARNED = False
LAUNCH_ATTEMPTS_FILENAME = "session_launches.jsonl"


def _log_error(msg: str) -> None:
    sys.stderr.write(msg.rstrip("\n") + "\n")
    sys.stderr.flush()


def _log_exception(context: str, exc: BaseException) -> None:
    ts = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    _log_error(f"error: {context}: {type(exc).__name__}: {exc}")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip("\n")
    if tb:
        _log_error(f"traceback ({ts}):\n{tb}")


def _socket_peer_disconnected(exc: BaseException) -> bool:
    if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
        return True
    return isinstance(exc, OSError) and exc.errno in (
        errno.EPIPE,
        errno.ECONNRESET,
        errno.ECONNABORTED,
        errno.ENOTCONN,
        errno.ESHUTDOWN,
    )


def _send_socket_json_line(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The PID exists but is owned by another user.
        return True
    except Exception:
        return False
    return True


def process_group_alive(root_pid: int) -> bool:
    if root_pid <= 0:
        return False
    try:
        os.killpg(root_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def default_app_dir() -> Path:
    base = Path.home() / ".local" / "share"
    new = base / "codoxear"
    old = base / "codex-web"
    if old.exists():
        global _LEGACY_WARNED
        if not _LEGACY_WARNED:
            _LEGACY_WARNED = True
            _log_error(
                f"error: legacy runtime dir detected at {old}; it is no longer used. "
                f"migrate runtime state to {new}."
            )
    return new


def launch_attempts_path(app_dir: Path | None = None) -> Path:
    return (app_dir or default_app_dir()) / LAUNCH_ATTEMPTS_FILENAME


def append_launch_attempt(record: dict[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    return _append_launch_attempt_impl(record, path=path or launch_attempts_path(), now_ts=now())


def read_launch_attempts(
    *,
    path: Path | None = None,
    max_records: int = 200,
    max_age_s: float = 24 * 3600,
    now_ts: float | None = None,
) -> list[dict[str, Any]]:
    return _read_launch_attempts_impl(
        path=path or launch_attempts_path(),
        max_records=max_records,
        max_age_s=max_age_s,
        now_ts=now() if now_ts is None else float(now_ts),
    )


def now() -> float:
    return time.time()


def _read_session_meta_payload_once(log_path: Path, *, max_bytes: int) -> dict[str, Any] | None:
    return _read_session_meta_payload_once_impl(log_path, max_bytes=max_bytes, log_exception=_log_exception)


def read_session_meta_payload(
    log_path: Path,
    *,
    agent_backend: str | None = None,
    timeout_s: float = 0.0,
    poll_s: float = 0.05,
    max_bytes: int = 64 * 1024,
) -> dict[str, Any] | None:
    return _read_session_meta_payload_impl(
        log_path,
        agent_backend=agent_backend,
        timeout_s=timeout_s,
        poll_s=poll_s,
        max_bytes=max_bytes,
        now_func=now,
        sleep_func=time.sleep,
        log_exception=_log_exception,
    )


def is_subagent_session_meta(payload: dict[str, Any]) -> bool:
    return _is_subagent_session_meta_impl(payload)


def subagent_parent_thread_id(payload: dict[str, Any]) -> str | None:
    return _subagent_parent_thread_id_impl(payload)


def classify_session_log(log_path: Path, *, agent_backend: str | None = None, timeout_s: float = 0.0) -> str | None:
    return _classify_session_log_impl(
        log_path,
        agent_backend=agent_backend,
        timeout_s=timeout_s,
        read_session_meta_payload_func=read_session_meta_payload,
        is_subagent_session_meta_func=is_subagent_session_meta,
    )


def iter_session_logs(sessions_dir: Path, *, agent_backend: str = "codex") -> list[Path]:
    return _iter_session_logs_impl(sessions_dir, agent_backend=agent_backend, log_exception=_log_exception)


def find_session_log_for_session_id(sessions_dir: Path, session_id: str, *, agent_backend: str = "codex") -> Path | None:
    return _find_session_log_for_session_id_impl(
        sessions_dir,
        session_id,
        agent_backend=agent_backend,
        iter_session_logs_func=iter_session_logs,
    )


def find_new_session_log(
    *,
    sessions_dir: Path,
    agent_backend: str = "codex",
    cwd: str | None = None,
    after_ts: float,
    preexisting: set[Path],
    exclude_paths: set[Path] | None = None,
    timeout_s: float,
) -> tuple[str, Path] | None:
    return _find_new_session_log_impl(
        sessions_dir=sessions_dir,
        agent_backend=agent_backend,
        cwd=cwd,
        after_ts=after_ts,
        preexisting=preexisting,
        exclude_paths=exclude_paths,
        timeout_s=timeout_s,
        now_func=now,
        sleep_func=time.sleep,
        iter_session_logs_func=iter_session_logs,
        read_session_meta_payload_func=read_session_meta_payload,
        is_subagent_session_meta_func=is_subagent_session_meta,
    )


def proc_find_open_rollout_log(
    *,
    proc_root: Path,
    root_pid: int,
    agent_backend: str = "codex",
    cwd: str | None = None,
    ignored_paths: set[Path] | None = None,
) -> Path | None:
    backend_name = normalize_agent_backend(agent_backend)
    cands = list(proc_open_writable_rollout_logs_for_backend(proc_root, root_pid, agent_backend=backend_name))
    if not cands:
        return None
    ignored_resolved: set[Path] = set()
    for p in ignored_paths or set():
        try:
            ignored_resolved.add(p.resolve())
        except Exception:
            ignored_resolved.add(p)
    try:
        cands.sort(key=lambda p: float(p.stat().st_mtime), reverse=True)
    except Exception:
        pass
    matches: list[Path] = []
    for p in cands:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in ignored_resolved:
            continue
        payload = read_session_meta_payload(p, agent_backend=backend_name, timeout_s=0.0)
        if not payload:
            continue
        if backend_name == "codex" and is_subagent_session_meta(payload):
            continue
        if cwd is not None:
            if not _payload_cwd_matches(payload.get("cwd"), cwd):
                continue
        matches.append(p)
    if len(matches) != 1:
        return None
    return matches[0]


def read_jsonl_from_offset(path: Path, offset: int, *, max_bytes: int, advance_on_oversized_unterminated: bool = True) -> tuple[list[dict[str, Any]], int]:
    return _read_jsonl_from_offset_impl(
        path,
        offset,
        max_bytes=max_bytes,
        advance_on_oversized_unterminated=advance_on_oversized_unterminated,
        log_exception=_log_exception,
    )
