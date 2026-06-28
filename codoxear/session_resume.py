from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any, Callable, Iterable

from .agent_backend import normalize_agent_backend


def resume_candidate_from_log(
    log_path: Path,
    *,
    agent_backend: str = "codex",
    read_session_meta: Callable[..., dict[str, Any]],
    is_subagent_session_meta: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    backend_name = normalize_agent_backend(agent_backend)
    meta = read_session_meta(log_path, agent_backend=backend_name)
    if backend_name == "codex" and is_subagent_session_meta(meta):
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


def list_resume_candidates_for_cwd(
    cwd: str,
    *,
    agent_backend: str = "codex",
    limit: int = 12,
    iter_session_logs: Callable[..., Iterable[Path]],
    resume_candidate_from_log_func: Callable[..., dict[str, Any] | None],
) -> list[dict[str, Any]]:
    backend_name = normalize_agent_backend(agent_backend)
    cwd_resolved = str(Path(cwd).expanduser().resolve())
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for log_path in iter_session_logs(agent_backend=backend_name):
        try:
            row = resume_candidate_from_log_func(log_path, agent_backend=backend_name)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        session_id = row.get("session_id")
        row_cwd = row.get("cwd")
        if not (isinstance(session_id, str) and session_id):
            continue
        if not (isinstance(row_cwd, str) and row_cwd == cwd_resolved):
            continue
        if session_id in seen:
            continue
        out.append(row)
        seen.add(session_id)
        if len(out) >= limit:
            break
    return out


def resume_preview_from_text(text: str, *, max_chars: int = 120) -> str:
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


def user_message_text(payload: dict[str, Any]) -> str:
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


def is_scaffold_user_text(text: str) -> bool:
    cleaned = text.strip()
    return cleaned.startswith("# AGENTS.md instructions") or cleaned.startswith("<environment_context>")


def first_user_message_preview_from_log(
    log_path: Path,
    *,
    pi_user_text: Callable[[dict[str, Any]], str | None],
    cc_user_text: Callable[[dict[str, Any]], str | None],
    max_scan_bytes: int = 256 * 1024,
) -> str:
    try:
        with log_path.open("rb") as handle:
            total = 0
            for raw in handle:
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
                    text = pi_user_text(obj) or ""
                elif obj.get("type") == "user":
                    text = cc_user_text(obj) or ""
                elif obj.get("type") == "response_item":
                    payload = obj.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    if payload.get("type") != "message" or payload.get("role") != "user":
                        continue
                    text = user_message_text(payload)
                else:
                    continue
                if not text or is_scaffold_user_text(text):
                    continue
                return resume_preview_from_text(text)
    except FileNotFoundError:
        return ""
    return ""


def coerce_main_thread_log(
    *,
    thread_id: str,
    log_path: Path,
    read_session_meta_or_none: Callable[..., dict[str, Any] | None],
    is_subagent_session_meta: Callable[[dict[str, Any]], bool],
    subagent_parent_thread_id: Callable[[dict[str, Any]], str | None],
    find_session_log_for_session_id: Callable[[str], Path | None],
) -> tuple[str, Path]:
    session_meta = read_session_meta_or_none(log_path, agent_backend="codex", context="main-thread coercion")
    if not session_meta:
        return thread_id, log_path
    if not is_subagent_session_meta(session_meta):
        return thread_id, log_path
    parent = subagent_parent_thread_id(session_meta)
    if not parent:
        return thread_id, log_path
    parent_log = find_session_log_for_session_id(parent)
    if parent_log is None or not parent_log.exists():
        return thread_id, log_path
    return parent, parent_log
