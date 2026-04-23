from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..runtime import ServerRuntime


def fallback_path_mtime(path: Path) -> float | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    except Exception:
        return 0.0
    return float(stat.st_mtime)


def last_pi_conversation_ts(runtime: ServerRuntime, path: Path) -> float | None:
    sv = runtime
    try:
        for entry in sv._rollout_log._iter_jsonl_objects_reverse(path):
            if entry.get("type") != "message":
                continue
            message = entry.get("message")
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role not in {"user", "assistant", "toolResult"}:
                continue
            ts = sv._pi_messages._entry_ts(message)
            if ts is None:
                ts = sv._pi_messages._entry_ts(entry)
            if isinstance(ts, (int, float)) and math.isfinite(float(ts)) and float(ts) > 0:
                return float(ts)
    except FileNotFoundError:
        return None
    except Exception:
        return 0.0
    return None


def resume_candidate_updated_ts(
    runtime: ServerRuntime,
    path: Path,
    *,
    agent_backend: str,
) -> float | None:
    sv = runtime
    backend_name = sv.normalize_agent_backend(agent_backend)
    if backend_name == "pi":
        ts = last_pi_conversation_ts(runtime, path)
    else:
        ts = sv._last_conversation_ts_from_tail(path)
    if isinstance(ts, (int, float)) and math.isfinite(float(ts)) and float(ts) > 0:
        return float(ts)
    return fallback_path_mtime(path)


def resume_candidate_from_log(
    runtime: ServerRuntime,
    log_path: Path,
    *,
    agent_backend: str = "codex",
) -> dict[str, Any] | None:
    sv = runtime
    backend_name = sv.normalize_agent_backend(agent_backend)
    meta = sv._read_session_meta(log_path, agent_backend=backend_name)
    if backend_name == "codex" and sv._is_subagent_session_meta(meta):
        return None
    session_id = meta.get("id")
    cwd = meta.get("cwd")
    if not isinstance(session_id, str) or not session_id:
        return None
    if not isinstance(cwd, str) or not cwd:
        return None
    updated_ts = resume_candidate_updated_ts(runtime, log_path, agent_backend=backend_name)
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


def pi_resume_candidate_from_session_file(
    runtime: ServerRuntime,
    session_path: Path,
) -> dict[str, Any] | None:
    sv = runtime
    if sv._pi_session_has_handoff_history(session_path):
        return None
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
                updated_ts = resume_candidate_updated_ts(
                    runtime,
                    session_path,
                    agent_backend="pi",
                )
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
                    "title": sv._pi_session_name_from_session_file(session_path),
                }
    except OSError:
        return None
    return None


def discover_pi_session_for_cwd(
    runtime: ServerRuntime,
    cwd: str,
    start_ts: float,
    *,
    exclude: set[Path] | None = None,
) -> Path | None:
    sv = runtime
    session_dir = sv._pi_native_session_dir_for_cwd(cwd)
    if not session_dir.is_dir():
        return None
    best: Path | None = None
    best_mtime: float = 0
    for f in session_dir.glob("*.jsonl"):
        if exclude and f in exclude:
            continue
        if sv._pi_session_has_handoff_history(f):
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


def resolve_pi_session_path(
    runtime: ServerRuntime,
    *,
    thread_id: str | None,
    cwd: str,
    start_ts: float,
    preferred: Path | None = None,
    exclude: set[Path] | None = None,
) -> tuple[Path | None, str | None]:
    sv = runtime
    clean_thread_id = str(thread_id or "").strip()
    if preferred is not None:
        try:
            preferred_exists = preferred.exists()
        except OSError:
            preferred_exists = False
        if preferred_exists:
            if (not clean_thread_id) or (
                sv._read_pi_session_id(preferred) == clean_thread_id
            ):
                return preferred, "preferred"
    if clean_thread_id:
        exact = sv._find_session_log_for_session_id(clean_thread_id, agent_backend="pi")
        if exact is not None:
            return exact, "exact"
    if preferred is not None:
        return preferred, "preferred"
    discovered = sv._discover_pi_session_for_cwd(cwd, start_ts, exclude=exclude)
    if discovered is not None:
        return discovered, "discovered"
    return None, None


def list_resume_candidates_for_cwd(
    runtime: ServerRuntime,
    cwd: str,
    *,
    limit: int = 12,
    offset: int = 0,
    backend: str | None = None,
    agent_backend: str | None = None,
) -> list[dict[str, Any]]:
    sv = runtime
    cwd2 = str(sv._safe_expanduser(Path(cwd)).resolve())
    backend_raw = backend if backend is not None else agent_backend
    backend2 = sv.normalize_agent_backend(backend_raw, default="codex")
    limit2 = max(1, int(limit))
    offset2 = max(0, int(offset))
    if backend2 == "pi":
        rows: list[dict[str, Any]] = []
        session_dir = sv._pi_native_session_dir_for_cwd(cwd2)
        if not session_dir.exists():
            return rows
        for session_path in session_dir.glob("*.jsonl"):
            row = sv._pi_resume_candidate_from_session_file(session_path)
            if not isinstance(row, dict):
                continue
            if row.get("cwd") != cwd2:
                continue
            rows.append(row)
        rows.sort(key=lambda row: -float(row.get("updated_ts") or 0.0))
        return rows[offset2 : offset2 + limit2]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for log_path in sv._iter_session_logs(agent_backend=backend2):
        try:
            row = sv._resume_candidate_from_log(log_path, agent_backend=backend2)
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


def iter_all_resume_candidates(
    runtime: ServerRuntime,
    *,
    limit: int = 200,
) -> list[dict[str, Any]]:
    sv = runtime
    seen: set[tuple[str, str]] = set()
    ranked_rows: list[tuple[float, dict[str, Any]]] = []

    if sv.PI_NATIVE_SESSIONS_DIR.exists():
        for session_path in sv.PI_NATIVE_SESSIONS_DIR.glob("--*--/*.jsonl"):
            row = sv._pi_resume_candidate_from_session_file(session_path)
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

    for log_path in sv._iter_session_logs(agent_backend="codex"):
        try:
            row = sv._resume_candidate_from_log(log_path, agent_backend="codex")
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
