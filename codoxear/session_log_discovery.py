from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .agent_backend import get_agent_backend
from .agent_backend import infer_agent_backend_from_log_path
from .agent_backend import normalize_agent_backend
from .cc_log import read_cc_session_header
from .pi_log import read_pi_session_header
from .session_log_paths import _path_in_set
from .session_log_paths import _payload_cwd_matches


LogException = Callable[[str, BaseException], None]
NowFunc = Callable[[], float]
SleepFunc = Callable[[float], None]
ReadMetaFunc = Callable[..., dict[str, Any] | None]
IsSubagentFunc = Callable[[dict[str, Any]], bool]
IterLogsFunc = Callable[..., Iterable[Path]]
NormalizeBackendFunc = Callable[[object], str]
ProcOpenWritableLogsFunc = Callable[..., Iterable[Path]]
PayloadCwdMatchesFunc = Callable[[object, str], bool]


def _read_session_meta_payload_once(log_path: Path, *, max_bytes: int, log_exception: LogException) -> dict[str, Any] | None:
    try:
        with log_path.open("rb") as f:
            data = f.read(int(max_bytes))
    except FileNotFoundError:
        return None
    except Exception as e:
        log_exception(f"read session log {log_path}", e)
        raise

    for raw in data.splitlines():
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        except Exception as e:
            log_exception(f"decode session log line from {log_path}", e)
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
    now_func: NowFunc,
    sleep_func: SleepFunc = time.sleep,
    log_exception: LogException,
) -> dict[str, Any] | None:
    backend_name = normalize_agent_backend(
        agent_backend if agent_backend is not None else infer_agent_backend_from_log_path(log_path) or "codex"
    )
    if backend_name == "pi":
        return read_pi_session_header(log_path)
    if backend_name == "cc":
        return read_cc_session_header(log_path)
    deadline = now_func() + float(timeout_s)
    while True:
        payload = _read_session_meta_payload_once(log_path, max_bytes=max_bytes, log_exception=log_exception)
        if payload is not None:
            return payload
        if timeout_s <= 0:
            return None
        if now_func() >= deadline:
            return None
        sleep_func(float(poll_s))


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


def classify_session_log(
    log_path: Path,
    *,
    agent_backend: str | None = None,
    timeout_s: float = 0.0,
    read_session_meta_payload_func: ReadMetaFunc,
    is_subagent_session_meta_func: IsSubagentFunc,
) -> str | None:
    payload = read_session_meta_payload_func(log_path, agent_backend=agent_backend, timeout_s=timeout_s)
    if payload is None:
        return None
    return "subagent" if is_subagent_session_meta_func(payload) else "main"


def iter_session_logs(sessions_dir: Path, *, agent_backend: str = "codex", log_exception: LogException) -> list[Path]:
    backend = get_agent_backend(agent_backend)
    if not sessions_dir.exists():
        return []
    out: list[tuple[float, Path]] = []
    for p in sessions_dir.rglob(backend.log_glob_pattern()):
        if not backend.is_session_log_path(p, sessions_dir=sessions_dir):
            continue
        try:
            mt = float(p.stat().st_mtime)
        except FileNotFoundError:
            continue
        except Exception as e:
            log_exception(f"stat {p}", e)
            raise
        out.append((mt, p))
    out.sort(key=lambda t: t[0], reverse=True)
    return [p for _mt, p in out]


def find_session_log_for_session_id(
    sessions_dir: Path,
    session_id: str,
    *,
    agent_backend: str = "codex",
    iter_session_logs_func: IterLogsFunc,
) -> Path | None:
    backend = get_agent_backend(agent_backend)
    if not session_id:
        return None
    for p in iter_session_logs_func(sessions_dir, agent_backend=backend.name):
        if backend.log_matches_session_id(p, session_id):
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
    now_func: NowFunc,
    sleep_func: SleepFunc = time.sleep,
    iter_session_logs_func: IterLogsFunc,
    read_session_meta_payload_func: ReadMetaFunc,
    is_subagent_session_meta_func: IsSubagentFunc,
) -> tuple[str, Path] | None:
    backend = get_agent_backend(agent_backend)
    backend_name = backend.name
    if cwd is not None:
        if not isinstance(cwd, str) or (not cwd.strip()):
            raise ValueError("cwd must be a non-empty string when provided")
    deadline = now_func() + float(timeout_s)
    while True:
        matches: list[tuple[str, Path]] = []
        for p in iter_session_logs_func(sessions_dir, agent_backend=backend_name):
            if _path_in_set(p, preexisting):
                continue
            if exclude_paths and _path_in_set(p, exclude_paths):
                continue
            try:
                if p.stat().st_mtime < after_ts - 2:
                    continue
            except FileNotFoundError:
                continue
            payload = read_session_meta_payload_func(p, agent_backend=backend_name, timeout_s=0.0)
            if not payload:
                continue
            if backend_name == "codex" and is_subagent_session_meta_func(payload):
                continue
            if cwd is not None:
                if not _payload_cwd_matches(payload.get("cwd"), cwd):
                    continue
            sid = backend.session_id_from_payload_or_log(p, payload)
            if isinstance(sid, str) and sid:
                matches.append((sid, p))
        if len(matches) == 1:
            return matches[0]
        if now_func() >= deadline:
            return None
        sleep_func(0.2)


def proc_find_open_rollout_log(
    *,
    proc_root: Path,
    root_pid: int,
    agent_backend: str = "codex",
    cwd: str | None = None,
    ignored_paths: set[Path] | None = None,
    normalize_agent_backend_func: NormalizeBackendFunc,
    proc_open_writable_rollout_logs_for_backend_func: ProcOpenWritableLogsFunc,
    read_session_meta_payload_func: ReadMetaFunc,
    is_subagent_session_meta_func: IsSubagentFunc,
    payload_cwd_matches_func: PayloadCwdMatchesFunc = _payload_cwd_matches,
) -> Path | None:
    backend_name = normalize_agent_backend_func(agent_backend)
    cands = list(proc_open_writable_rollout_logs_for_backend_func(proc_root, root_pid, agent_backend=backend_name))
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
        payload = read_session_meta_payload_func(p, agent_backend=backend_name, timeout_s=0.0)
        if not payload:
            continue
        if backend_name == "codex" and is_subagent_session_meta_func(payload):
            continue
        if cwd is not None:
            if not payload_cwd_matches_func(payload.get("cwd"), cwd):
                continue
        matches.append(p)
    if len(matches) != 1:
        return None
    return matches[0]
