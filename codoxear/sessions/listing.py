from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..runtime import ServerRuntime


def session_row_display_name(
    row: dict[str, Any], *, fallback: str = "Session"
) -> str:
    if not isinstance(row, dict):
        return fallback
    for key in ("alias", "title", "first_user_message", "session_id"):
        value = row.get(key)
        if not isinstance(value, str):
            continue
        out = value.strip()
        if out:
            return out
    return fallback


def normalize_session_cwd_row(
    runtime: ServerRuntime, row: dict[str, Any]
) -> dict[str, Any]:
    sv = runtime
    if not isinstance(row, dict):
        return row
    normalized = dict(row)
    if "cwd" in normalized:
        canonical_cwd = sv._canonical_session_cwd(normalized.get("cwd"))
        if canonical_cwd is not None:
            normalized["cwd"] = canonical_cwd
    normalized["display_name"] = session_row_display_name(normalized)
    return normalized


def frontend_session_list_row(
    runtime: ServerRuntime, row: dict[str, Any]
) -> dict[str, Any]:
    sv = runtime
    normalized = normalize_session_cwd_row(runtime, row)
    if not isinstance(normalized, dict):
        return normalized
    return {
        key: normalized[key] for key in sv.SESSION_LIST_ROW_KEYS if key in normalized
    }


def session_list_group_key(runtime: ServerRuntime, row: dict[str, Any]) -> str:
    sv = runtime
    cwd = sv._canonical_session_cwd(row.get("cwd"))
    return cwd or sv.SESSION_LIST_FALLBACK_GROUP_KEY


def session_list_payload(
    runtime: ServerRuntime,
    rows: list[dict[str, Any]],
    *,
    group_key: str | None = None,
    offset: int = 0,
    limit: int = 100,
    group_offset: int = 0,
    group_limit: int = 12,
) -> dict[str, Any]:
    sv = runtime
    start = max(0, int(offset))
    stop = start + max(1, int(limit))

    if (
        group_key is None
        and group_offset <= 0
        and group_limit == sv.SESSION_LIST_RECENT_GROUP_LIMIT
    ):
        page_rows = [frontend_session_list_row(runtime, row) for row in rows[start:stop]]
        remaining = max(0, len(rows) - stop)
        payload: dict[str, Any] = {"sessions": page_rows}
        if remaining > 0:
            payload["remaining_count"] = remaining
        return payload

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = session_list_group_key(runtime, row)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)

    def _page_rows_for_group(
        group_rows: list[dict[str, Any]], page_size: int
    ) -> list[dict[str, Any]]:
        if len(group_rows) <= page_size:
            return list(group_rows)
        page_rows = list(group_rows[:page_size])
        focused_rows = [row for row in group_rows if bool(row.get("focused"))]
        if not focused_rows:
            return page_rows
        existing_ids = {id(row) for row in page_rows}
        extras = [row for row in focused_rows if id(row) not in existing_ids]
        if not extras:
            return page_rows
        keep = [row for row in page_rows if bool(row.get("focused"))]
        keep.extend(extras)
        keep = keep[:page_size]
        keep_ids = {id(item) for item in keep}
        for row in group_rows:
            if len(keep) >= page_size:
                break
            if id(row) in keep_ids:
                continue
            keep.append(row)
            keep_ids.add(id(row))
        return keep

    def _group_sort_key(key: str) -> tuple[int, float]:
        group_rows = grouped[key]
        busy = any(bool(row.get("busy")) for row in group_rows)
        latest_updated = max(float(row.get("updated_ts") or 0.0) for row in group_rows)
        return (0 if busy else 1, -latest_updated)

    group_order = sorted(grouped.keys(), key=_group_sort_key)

    if group_key is not None:
        group_rows = grouped.get(group_key, [])
        page_rows = [
            frontend_session_list_row(runtime, row) for row in group_rows[start:stop]
        ]
        remaining = max(0, len(group_rows) - stop)
        return {
            "sessions": page_rows,
            "remaining_by_group": {group_key: remaining} if remaining > 0 else {},
        }

    selected_group_keys = set(group_order[: sv.SESSION_LIST_RECENT_GROUP_LIMIT])
    omitted_group_count = 0
    for key, group_rows in grouped.items():
        if any(bool(row.get("busy")) for row in group_rows) or any(
            bool(row.get("focused")) for row in group_rows
        ):
            selected_group_keys.add(key)

    if group_offset > 0 or group_limit != sv.SESSION_LIST_RECENT_GROUP_LIMIT:
        group_stop = max(group_offset, 0) + max(1, int(group_limit))
        extra_group_order = group_order[group_offset:group_stop]
        selected_group_keys = set(extra_group_order)
        omitted_group_count = max(0, len(group_order) - group_stop)

    sessions: list[dict[str, Any]] = []
    remaining_by_group: dict[str, int] = {}
    for key in group_order:
        if key not in selected_group_keys:
            continue
        group_rows = grouped[key]
        page_rows = _page_rows_for_group(group_rows, sv.SESSION_LIST_GROUP_PAGE_SIZE)
        sessions.extend(frontend_session_list_row(runtime, row) for row in page_rows)
        remaining = len(group_rows) - len(page_rows)
        if remaining > 0:
            remaining_by_group[key] = remaining
    result: dict[str, Any] = {
        "sessions": sessions,
        "remaining_by_group": remaining_by_group,
    }
    if group_offset <= 0 and group_limit == sv.SESSION_LIST_RECENT_GROUP_LIMIT:
        omitted_group_count = max(0, len(group_order) - len(selected_group_keys))
    result["omitted_group_count"] = omitted_group_count
    return result


def historical_session_id(
    runtime: ServerRuntime, backend: str, resume_session_id: str
) -> str:
    _ = runtime
    return f"history:{backend}:{resume_session_id}"


def parse_historical_session_id(
    runtime: ServerRuntime, session_id: str
) -> tuple[str, str] | None:
    sv = runtime
    raw = str(session_id or "").strip()
    if not raw.startswith("history:"):
        return None
    _prefix, backend, resume_session_id = (
        raw.split(":", 2) if raw.count(":") >= 2 else ("", "", "")
    )
    backend_clean = sv.normalize_agent_backend(backend, default="codex")
    resume_clean = sv._clean_optional_text(resume_session_id)
    if not resume_clean:
        return None
    return backend_clean, resume_clean


def historical_session_row(
    runtime: ServerRuntime, session_id: str
) -> dict[str, Any] | None:
    sv = runtime
    parsed = parse_historical_session_id(runtime, session_id)
    if parsed is None:
        return None
    backend, resume_session_id = parsed
    for row in sv._iter_all_resume_candidates():
        if (
            sv.normalize_agent_backend(
                row.get("agent_backend", row.get("backend")), default="codex"
            )
            != backend
        ):
            continue
        if sv._clean_optional_text(row.get("session_id")) != resume_session_id:
            continue
        out = dict(row)
        out["session_id"] = historical_session_id(runtime, backend, resume_session_id)
        out["resume_session_id"] = resume_session_id
        out["historical"] = True
        return out
    return None


def historical_sidebar_items(
    runtime: ServerRuntime,
    *,
    live_resume_keys: set[tuple[str, str]],
    now_ts: float,
) -> list[dict[str, Any]]:
    sv = runtime
    out: list[dict[str, Any]] = []
    for row in sv._iter_all_resume_candidates():
        resume_session_id = row.get("session_id")
        cwd = row.get("cwd")
        if not isinstance(resume_session_id, str) or not resume_session_id:
            continue
        if not isinstance(cwd, str) or not cwd:
            continue
        backend = sv.normalize_agent_backend(
            row.get("agent_backend", row.get("backend")), default="codex"
        )
        live_key = (backend, resume_session_id)
        if live_key in live_resume_keys:
            continue
        updated_ts_raw = row.get("updated_ts")
        updated_ts = (
            float(updated_ts_raw)
            if isinstance(updated_ts_raw, (int, float))
            else float(now_ts)
        )
        elapsed_s = max(0.0, now_ts - updated_ts)
        time_priority = sv._priority_from_elapsed_seconds(elapsed_s)
        first_user_message = ""
        try:
            if backend == "pi":
                session_path_raw = row.get("session_path")
                if isinstance(session_path_raw, str) and session_path_raw:
                    first_user_message = first_user_message_preview_from_pi_session(
                        runtime,
                        Path(session_path_raw),
                    )
            else:
                log_path_raw = row.get("log_path")
                if isinstance(log_path_raw, str) and log_path_raw:
                    first_user_message = first_user_message_preview_from_log(
                        runtime,
                        Path(log_path_raw),
                    )
        except Exception:
            first_user_message = ""
        out.append(
            {
                "session_id": historical_session_id(runtime, backend, resume_session_id),
                "runtime_id": None,
                "thread_id": resume_session_id,
                "backend": backend,
                "pid": None,
                "broker_pid": None,
                "agent_backend": backend,
                "owned": False,
                "transport": None,
                "cwd": cwd,
                "start_ts": updated_ts,
                "updated_ts": updated_ts,
                "log_path": row.get("log_path")
                if isinstance(row.get("log_path"), str)
                else row.get("session_path")
                if isinstance(row.get("session_path"), str)
                else None,
                "queue_len": 0,
                "token": None,
                "thinking": 0,
                "tools": 0,
                "system": 0,
                "harness_enabled": False,
                "harness_cooldown_minutes": sv.HARNESS_DEFAULT_IDLE_MINUTES,
                "harness_remaining_injections": sv.HARNESS_DEFAULT_MAX_INJECTIONS,
                "alias": "",
                "first_user_message": first_user_message,
                "files": [],
                "git_branch": row.get("git_branch"),
                "model_provider": None,
                "preferred_auth_method": None,
                "provider_choice": None,
                "model": None,
                "reasoning_effort": None,
                "service_tier": None,
                "tmux_session": None,
                "tmux_window": None,
                "priority_offset": 0.0,
                "snooze_until": None,
                "dependency_session_id": None,
                "time_priority": time_priority,
                "base_priority": time_priority,
                "final_priority": time_priority,
                "blocked": False,
                "snoozed": False,
                "busy": False,
                "historical": True,
                "resume_session_id": resume_session_id,
            }
        )

    out.sort(
        key=lambda row: (
            -float(row.get("updated_ts") or 0.0),
            str(row.get("backend") or ""),
            str(row.get("session_id") or ""),
        )
    )
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
    s = text.strip()
    return s.startswith("# AGENTS.md instructions") or s.startswith(
        "<environment_context>"
    )


def first_user_message_preview_from_log(
    runtime: ServerRuntime,
    log_path: Path,
    *,
    max_scan_bytes: int = 256 * 1024,
) -> str:
    sv = runtime
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
                    text = sv._pi_user_text(obj) or ""
                elif obj.get("type") == "response_item":
                    payload = obj.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    if (
                        payload.get("type") != "message"
                        or payload.get("role") != "user"
                    ):
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


def first_user_message_preview_from_pi_session(
    runtime: ServerRuntime,
    session_path: Path,
    *,
    max_scan_bytes: int = 256 * 1024,
) -> str:
    try:
        with session_path.open("rb") as f:
            total = 0
            for raw in f:
                total += len(raw)
                if total > max_scan_bytes:
                    break
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(obj, dict) or obj.get("type") != "message":
                    continue
                payload = obj.get("message")
                if not isinstance(payload, dict):
                    payload = obj.get("payload")
                if not isinstance(payload, dict):
                    payload = obj
                if payload.get("role") != "user":
                    continue
                text = user_message_text(payload)
                if not text:
                    content = payload.get("content")
                    if isinstance(content, str):
                        text = content.strip()
                if not text or is_scaffold_user_text(text):
                    continue
                return resume_preview_from_text(text)
    except FileNotFoundError:
        return ""
    return ""
