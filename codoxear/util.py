from __future__ import annotations

import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping

from .app_dir_runtime import resolve_default_app_dir as _resolve_default_app_dir
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
from .process_runtime import pid_alive
from .process_runtime import process_group_alive
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
from .session_log_discovery import proc_find_open_rollout_log as _proc_find_open_rollout_log_impl
from .session_log_discovery import is_subagent_session_meta as _is_subagent_session_meta_impl
from .session_log_discovery import iter_session_logs as _iter_session_logs_impl
from .session_log_discovery import read_session_meta_payload as _read_session_meta_payload_impl
from .session_log_discovery import subagent_parent_thread_id as _subagent_parent_thread_id_impl
from .socket_json import send_socket_json_line as _send_socket_json_line
from .socket_json import socket_peer_disconnected as _socket_peer_disconnected
from .pi_log import read_pi_log_cwd


_LEGACY_WARNED = False
LAUNCH_ATTEMPTS_FILENAME = "session_launches.jsonl"

# pi-subagents leaves completed runs on disk, so callers must not rescan the
# entire tree for every /api/sessions request. The cache is deliberately small:
# the session list poll is slower than this TTL, while an active run disappears
# from the UI within one additional poll after it completes.
_SUBAGENT_RUNS_CACHE_TTL_S = 2.0
_SUBAGENT_RUNS_CACHE_ROOT: str | None = None
_SUBAGENT_RUNS_CACHE_AT = 0.0
_SUBAGENT_RUNS_CACHE: dict[str, list[dict[str, Any]]] = {}
_ACTIVE_SUBAGENT_STATES = frozenset({"running", "pending"})
# Codex retains completed child rollout files indefinitely. Child liveness is a
# process signal when a writer still has the file open, with a short mtime grace
# period for a just-started writer that has not yet exposed its FD through
# /proc. The cache keeps session-list polling from walking a rollout tree and
# /proc on every request.
_CODEX_SUBAGENT_ACTIVITY_GRACE_S = 8.0
_CODEX_SUBAGENT_CACHE_TTL_S = 2.0
_CODEX_SUBAGENT_CACHE_ROOTS: tuple[str, ...] | None = None
_CODEX_SUBAGENT_CACHE_AT = 0.0
_CODEX_SUBAGENT_CACHE: dict[str, list[dict[str, Any]]] = {}


def _codex_sessions_dir_for_log(log_path: Path) -> Path | None:
    for parent in (log_path.parent, *log_path.parents):
        if parent.name == "sessions":
            return parent
    return None


def _writable_codex_rollout_paths(proc_root: Path = Path("/proc")) -> set[Path]:
    """Find Codex rollout files held writable by a process owned by this user.

    Codex subagents may be owned by an app-server rather than the parent TUI,
    so walking only a main session's process tree misses real child activity.
    """
    if sys.platform == "darwin":
        # The portable fallback below is the recent-mtime grace; lsof's
        # PID-scoped helper cannot enumerate every user process safely here.
        return set()
    out: set[Path] = set()
    try:
        process_entries = list(proc_root.iterdir())
    except OSError:
        return out
    uid = os.getuid()
    backend = get_agent_backend("codex")
    for process_entry in process_entries:
        try:
            pid = int(process_entry.name)
        except ValueError:
            continue
        if _proc_pid_uid(proc_root, pid) != uid:
            continue
        fd_dir = process_entry / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd in fds:
            flags = _proc_fd_flags(proc_root, pid, fd.name)
            if flags is None or not _fd_has_write_intent(flags):
                continue
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.endswith(" (deleted)"):
                continue
            path = Path(target)
            if target.startswith("/") and backend.is_session_log_path(path):
                out.add(path)
    return out


def _codex_child_rollout_is_terminal(log_path: Path) -> bool:
    """Whether the last decisive child event closes its work episode."""
    try:
        with log_path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 64 * 1024))
            rows = stream.read().splitlines()
    except OSError:
        return False
    terminal_events = frozenset({"turn_aborted", "thread_rolled_back", "task_complete", "turn_complete"})
    activity_events = frozenset({"user_message", "agent_reasoning", "agent_message", "function_call", "function_call_output"})
    for raw in reversed(rows):
        try:
            obj = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(obj, dict) or obj.get("type") != "event_msg":
            continue
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        event_type = payload.get("type")
        if event_type in terminal_events:
            return True
        if event_type in activity_events:
            return False
    return False


def scan_active_codex_subagents(
    *,
    sessions_dirs: Iterable[Path] | None = None,
    now_monotonic: float | None = None,
    now_wall: float | None = None,
    writable_paths: set[Path] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return live Codex child rollouts grouped by parent thread ID.

    A child header establishes lineage. A retained header alone is never
    activity: the child must have a writable owner or have changed inside the
    short writer-discovery grace period, and its latest decisive event must not
    close the child work episode.
    """
    global _CODEX_SUBAGENT_CACHE_AT, _CODEX_SUBAGENT_CACHE_ROOTS, _CODEX_SUBAGENT_CACHE
    roots_source = (get_agent_backend("codex").sessions_dir(),) if sessions_dirs is None else sessions_dirs
    roots = tuple(sorted({os.fspath(path) for path in roots_source}))
    monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
    if writable_paths is None and _CODEX_SUBAGENT_CACHE_ROOTS == roots and monotonic - _CODEX_SUBAGENT_CACHE_AT < _CODEX_SUBAGENT_CACHE_TTL_S:
        return {parent: [dict(run) for run in runs] for parent, runs in _CODEX_SUBAGENT_CACHE.items()}

    active_paths = _writable_codex_rollout_paths() if writable_paths is None else set(writable_paths)
    active_resolved: set[Path] = set()
    for path in active_paths:
        try:
            active_resolved.add(path.resolve())
        except OSError:
            active_resolved.add(path)
    wall_time = time.time() if now_wall is None else float(now_wall)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for root_raw in roots:
        root = Path(root_raw)
        try:
            child_paths = list(root.rglob("rollout-*.jsonl"))
        except OSError:
            continue
        for child_path in child_paths:
            try:
                stat = child_path.stat()
                resolved = child_path.resolve()
            except OSError:
                continue
            writable = resolved in active_resolved
            mtime_age = wall_time - float(stat.st_mtime)
            recent = 0.0 <= mtime_age <= _CODEX_SUBAGENT_ACTIVITY_GRACE_S
            if not writable and not recent:
                continue
            if _codex_child_rollout_is_terminal(child_path):
                continue
            payload = read_session_meta_payload(child_path, agent_backend="codex", timeout_s=0.0)
            if not payload or not is_subagent_session_meta(payload):
                continue
            parent_thread_id = subagent_parent_thread_id(payload)
            if parent_thread_id is None:
                continue
            child_thread_id = payload.get("id")
            grouped.setdefault(parent_thread_id, []).append({
                "thread_id": child_thread_id if isinstance(child_thread_id, str) and child_thread_id else child_path.stem,
                "log_path": str(child_path),
                "updated_at": float(stat.st_mtime),
            })

    for runs in grouped.values():
        runs.sort(key=lambda run: str(run["thread_id"]))
    if writable_paths is None:
        _CODEX_SUBAGENT_CACHE_ROOTS = roots
        _CODEX_SUBAGENT_CACHE_AT = monotonic
        _CODEX_SUBAGENT_CACHE = grouped
    return {parent: [dict(run) for run in runs] for parent, runs in grouped.items()}


def _cc_subagent_runs_root() -> Path:
    configured = os.environ.get("CODEX_WEB_CC_SUBAGENT_RUNS_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return _resolve_default_app_dir(legacy_warned=False).app_dir / "cc-subagent-runs"


def scan_active_cc_subagents(*, parent_broker_pids: Mapping[str, int]) -> dict[str, list[dict[str, Any]]]:
    """Return hook-authoritative Claude Code subagents for live web sessions.

    Claude child transcripts remain on disk after completion and do not encode
    live state. The only accepted records are app-owned Start-hook records that
    still belong to the exact live broker PID for their parent session. The
    Stop hook removes the record; missing, malformed, terminal-owned, or stale
    records project no activity.
    """
    root = _cc_subagent_runs_root()
    grouped: dict[str, list[dict[str, Any]]] = {}
    try:
        parent_dirs = list(root.iterdir())
    except OSError:
        return grouped
    for parent_dir in parent_dirs:
        if not parent_dir.is_dir():
            continue
        parent_session_id = parent_dir.name
        expected_broker_pid = parent_broker_pids.get(parent_session_id)
        if not isinstance(expected_broker_pid, int) or expected_broker_pid <= 0 or not pid_alive(expected_broker_pid):
            continue
        try:
            status_paths = list(parent_dir.glob("*.json"))
        except OSError:
            continue
        for status_path in status_paths:
            try:
                payload = load_json_file(status_path)
            except (OSError, ValueError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            agent_id = payload.get("agent_id")
            if (
                payload.get("version") != 1
                or payload.get("state") != "running"
                or payload.get("parent_session_id") != parent_session_id
                or not isinstance(agent_id, str)
                or status_path.stem != agent_id
                or payload.get("broker_pid") != expected_broker_pid
            ):
                continue
            grouped.setdefault(parent_session_id, []).append({"agent_id": agent_id})
    for runs in grouped.values():
        runs.sort(key=lambda run: str(run["agent_id"]))
    return grouped


def _subagent_runs_root() -> Path:
    configured = os.environ.get("CODEX_WEB_SUBAGENT_RUNS_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(f"/tmp/pi-subagents-uid-{os.getuid()}/async-subagent-runs")


def scan_active_pi_subagents(*, now_monotonic: float | None = None) -> dict[str, list[dict[str, Any]]]:
    """Return active pi-subagents grouped by their parent session log path.

    pi-subagents status files are an auxiliary, best-effort signal. A missing
    or concurrently changing run directory must therefore project as no runs,
    rather than making the session-list route fail.
    """
    global _SUBAGENT_RUNS_CACHE_AT, _SUBAGENT_RUNS_CACHE_ROOT, _SUBAGENT_RUNS_CACHE
    root = _subagent_runs_root()
    root_key = os.fspath(root)
    now_value = time.monotonic() if now_monotonic is None else float(now_monotonic)
    if _SUBAGENT_RUNS_CACHE_ROOT == root_key and now_value - _SUBAGENT_RUNS_CACHE_AT < _SUBAGENT_RUNS_CACHE_TTL_S:
        return {parent: [dict(run) for run in runs] for parent, runs in _SUBAGENT_RUNS_CACHE.items()}

    grouped: dict[str, list[dict[str, Any]]] = {}
    try:
        status_paths = list(root.glob("*/status.json"))
    except OSError:
        status_paths = []
    for status_path in sorted(status_paths):
        try:
            with status_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, UnicodeDecodeError, ValueError, TypeError):
            continue
        if not isinstance(payload, dict) or str(payload.get("state", "")).strip().lower() not in _ACTIVE_SUBAGENT_STATES:
            continue
        parent = payload.get("sessionId")
        run_id = payload.get("runId") or status_path.parent.name
        started_at = payload.get("startedAt")
        if not isinstance(parent, str) or not parent or not isinstance(run_id, str) or not run_id:
            continue
        if not isinstance(started_at, (int, float)) or isinstance(started_at, bool):
            continue
        agent = payload.get("agent")
        if not isinstance(agent, str) or not agent.strip():
            steps = payload.get("steps")
            if isinstance(steps, list):
                current_step = payload.get("currentStep")
                ordered_steps = (
                    [steps[current_step], *steps[:current_step], *steps[current_step + 1 :]]
                    if isinstance(current_step, int) and not isinstance(current_step, bool) and 0 <= current_step < len(steps)
                    else steps
                )
                for step in ordered_steps:
                    if isinstance(step, dict) and isinstance(step.get("agent"), str) and step["agent"].strip():
                        agent = step["agent"]
                        break
        grouped.setdefault(parent, []).append({
            "run_id": run_id,
            "agent": agent.strip() if isinstance(agent, str) else None,
            "started_at": started_at,
        })

    _SUBAGENT_RUNS_CACHE_ROOT = root_key
    _SUBAGENT_RUNS_CACHE_AT = now_value
    _SUBAGENT_RUNS_CACHE = grouped
    return {parent: [dict(run) for run in runs] for parent, runs in grouped.items()}


def _log_error(msg: str) -> None:
    sys.stderr.write(msg.rstrip("\n") + "\n")
    sys.stderr.flush()


def _log_exception(context: str, exc: BaseException) -> None:
    ts = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    _log_error(f"error: {context}: {type(exc).__name__}: {exc}")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip("\n")
    if tb:
        _log_error(f"traceback ({ts}):\n{tb}")


def default_app_dir() -> Path:
    global _LEGACY_WARNED
    resolution = _resolve_default_app_dir(legacy_warned=_LEGACY_WARNED)
    _LEGACY_WARNED = resolution.legacy_warned
    if resolution.warning:
        _log_error(resolution.warning)
    return resolution.app_dir


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
    return _proc_find_open_rollout_log_impl(
        proc_root=proc_root,
        root_pid=root_pid,
        agent_backend=agent_backend,
        cwd=cwd,
        ignored_paths=ignored_paths,
        normalize_agent_backend_func=normalize_agent_backend,
        proc_open_writable_rollout_logs_for_backend_func=proc_open_writable_rollout_logs_for_backend,
        read_session_meta_payload_func=read_session_meta_payload,
        is_subagent_session_meta_func=is_subagent_session_meta,
        payload_cwd_matches_func=_payload_cwd_matches,
    )


def read_jsonl_from_offset(path: Path, offset: int, *, max_bytes: int, advance_on_oversized_unterminated: bool = True) -> tuple[list[dict[str, Any]], int]:
    return _read_jsonl_from_offset_impl(
        path,
        offset,
        max_bytes=max_bytes,
        advance_on_oversized_unterminated=advance_on_oversized_unterminated,
        log_exception=_log_exception,
    )
