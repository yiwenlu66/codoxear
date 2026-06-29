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
from .pi_log import read_pi_log_cwd
from .pi_log import read_pi_session_header
from .pi_log import read_pi_session_id


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
    try:
        with log_path.open("rb") as f:
            data = f.read(int(max_bytes))
    except FileNotFoundError:
        return None
    except Exception as e:
        _log_exception(f"read session log {log_path}", e)
        raise

    for raw in data.splitlines():
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        except Exception as e:
            _log_exception(f"decode session log line from {log_path}", e)
            raise
        if obj.get("type") != "session_meta":
            continue
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"invalid session_meta payload in {log_path}")
        return payload
    return None


def read_session_meta_payload(
    log_path: Path,
    *,
    agent_backend: str | None = None,
    timeout_s: float = 0.0,
    poll_s: float = 0.05,
    max_bytes: int = 64 * 1024,
) -> dict[str, Any] | None:
    backend_name = normalize_agent_backend(
        agent_backend if agent_backend is not None else infer_agent_backend_from_log_path(log_path) or "codex"
    )
    if backend_name == "pi":
        return read_pi_session_header(log_path)
    if backend_name == "cc":
        return read_cc_session_header(log_path)
    deadline = now() + float(timeout_s)
    while True:
        payload = _read_session_meta_payload_once(log_path, max_bytes=max_bytes)
        if payload is not None:
            return payload
        if timeout_s <= 0:
            return None
        if now() >= deadline:
            return None
        time.sleep(float(poll_s))


def is_subagent_session_meta(payload: dict[str, Any]) -> bool:
    src = payload.get("source")
    return isinstance(src, dict) and ("subagent" in src)


def subagent_parent_thread_id(payload: dict[str, Any]) -> str | None:
    src = payload.get("source")
    if not isinstance(src, dict):
        return None
    sub = src.get("subagent")
    if not isinstance(sub, dict):
        return None
    spawn = sub.get("thread_spawn")
    if not isinstance(spawn, dict):
        return None
    parent = spawn.get("parent_thread_id")
    return parent if isinstance(parent, str) and parent else None


def classify_session_log(log_path: Path, *, agent_backend: str | None = None, timeout_s: float = 0.0) -> str | None:
    payload = read_session_meta_payload(log_path, agent_backend=agent_backend, timeout_s=timeout_s)
    if payload is None:
        return None
    return "subagent" if is_subagent_session_meta(payload) else "main"


def iter_session_logs(sessions_dir: Path, *, agent_backend: str = "codex") -> list[Path]:
    backend_name = normalize_agent_backend(agent_backend)
    if not sessions_dir.exists():
        return []
    out: list[tuple[float, Path]] = []
    pattern = "rollout-*.jsonl" if backend_name == "codex" else "*.jsonl"
    for p in sessions_dir.rglob(pattern):
        if backend_name == "codex" and not _is_codex_rollout_log_path(p):
            continue
        if backend_name == "pi" and not _is_pi_session_log_path(p, sessions_dir=sessions_dir):
            continue
        if backend_name == "cc" and not _is_cc_session_log_path(p, sessions_dir=sessions_dir):
            continue
        try:
            mt = float(p.stat().st_mtime)
        except FileNotFoundError:
            continue
        except Exception as e:
            _log_exception(f"stat {p}", e)
            raise
        out.append((mt, p))
    out.sort(key=lambda t: t[0], reverse=True)
    return [p for _mt, p in out]


def find_session_log_for_session_id(sessions_dir: Path, session_id: str, *, agent_backend: str = "codex") -> Path | None:
    backend_name = normalize_agent_backend(agent_backend)
    if not session_id:
        return None
    for p in iter_session_logs(sessions_dir, agent_backend=backend_name):
        if backend_name == "codex":
            if session_id in p.name:
                return p
            continue
        if backend_name == "pi":
            if read_pi_session_id(p) == session_id:
                return p
            continue
        if backend_name == "cc" and read_cc_session_id(p) == session_id:
            return p
    return None


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
    backend_name = normalize_agent_backend(agent_backend)
    if cwd is not None:
        if not isinstance(cwd, str) or (not cwd.strip()):
            raise ValueError("cwd must be a non-empty string when provided")
    deadline = now() + float(timeout_s)
    while True:
        matches: list[tuple[str, Path]] = []
        for p in iter_session_logs(sessions_dir, agent_backend=backend_name):
            if _path_in_set(p, preexisting):
                continue
            if exclude_paths and _path_in_set(p, exclude_paths):
                continue
            try:
                if p.stat().st_mtime < after_ts - 2:
                    continue
            except FileNotFoundError:
                continue
            payload = read_session_meta_payload(p, agent_backend=backend_name, timeout_s=0.0)
            if not payload:
                continue
            if backend_name == "codex" and is_subagent_session_meta(payload):
                continue
            if cwd is not None:
                if not _payload_cwd_matches(payload.get("cwd"), cwd):
                    continue
            if backend_name == "pi":
                sid = read_pi_session_id(p)
            elif backend_name == "cc":
                sid = read_cc_session_id(p)
            else:
                sid = payload.get("id")
            if isinstance(sid, str) and sid:
                matches.append((sid, p))
        if len(matches) == 1:
            return matches[0]
        if now() >= deadline:
            return None
        time.sleep(0.2)


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
    try:
        with path.open("rb") as f:
            f.seek(offset)
            target = max(1, int(max_bytes))
            chunk_size = max(64 * 1024, min(target, 1024 * 1024))
            data = f.read(target)
            if b"\n" not in data:
                if advance_on_oversized_unterminated:
                    # Read at most one overflow chunk so a live, unterminated JSONL
                    # record cannot make every poll read the rest of a huge file.
                    # Complete oversized records with a nearby newline still advance;
                    # fragments with no newline in this bounded window are skipped.
                    data += f.read(chunk_size)
                else:
                    # Live broker tailing must not advance over an incomplete row,
                    # but it also must not get stuck once an oversized row is
                    # completed beyond the bounded poll window. In no-advance mode,
                    # keep scanning until a newline proves at least one full record
                    # is available, or EOF proves the row is still incomplete.
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        data += chunk
                        if b"\n" in chunk:
                            break
    except Exception as e:
        _log_exception(f"read jsonl {path} from offset {offset}", e)
        raise

    if not data:
        return [], int(offset)

    # When tailing a live JSONL file, we can read a chunk that ends in the middle
    # of the last record, including the middle of a multibyte UTF-8 sequence.
    # Only parse newline-terminated records, and do not advance the offset past
    # the last newline we observed.
    last_nl = data.rfind(b"\n")
    if last_nl < 0:
        read_cap = max(1, int(max_bytes)) + max(64 * 1024, min(max(1, int(max_bytes)), 1024 * 1024))
        if advance_on_oversized_unterminated and len(data) >= read_cap:
            return [], int(offset) + len(data)
        return [], int(offset)
    data = data[: last_nl + 1]
    new_off = int(offset) + int(last_nl) + 1

    lines = data.splitlines()
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        except Exception as e:
            _log_exception(f"decode jsonl line from {path}", e)
            raise
        if isinstance(obj, dict):
            out.append(obj)
    return out, new_off
