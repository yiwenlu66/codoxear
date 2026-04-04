from __future__ import annotations

import datetime
import json as _json
from pathlib import Path
from typing import Any

from .rollout_log import _read_jsonl_tail as _read_jsonl_tail
from .util import read_jsonl_from_offset as _read_jsonl_from_offset

_MILLIS_THRESHOLD = 9_999_999_999  # ~2286 in epoch seconds; anything above is milliseconds

_PI_READ_MAX_BYTES = 2 * 1024 * 1024
_PI_TODO_SCAN_START_BYTES = 64 * 1024
_PI_TODO_SCAN_MAX_BYTES = 8 * 1024 * 1024
_PI_DIAG_TOOL_NAMES_LIMIT = 32
_PI_ABORT_EVENT_TYPES = {"turn.aborted", "turn.failed"}
_PI_END_EVENT_TYPES = {"turn.completed"}
_PI_TEXT_BLOCK_TYPES = {"output_text", "input_text", "text"}
_PI_TOOL_EVENT_TYPES = {
    "tool.started",
    "tool.finished",
    "toolResult",
    "bashExecution",
    "bashExecution.started",
    "bashExecution.finished",
}
_PI_TODO_STATUS_KEYS = {
    "completed": "completed",
    "in-progress": "in_progress",
    "not-started": "not_started",
}


def _payload_for_entry(entry: dict[str, Any]) -> dict[str, Any]:
    message = entry.get("message")
    if entry.get("type") == "message" and isinstance(message, dict):
        return message
    payload = entry.get("payload")
    if entry.get("type") == "message" and isinstance(payload, dict):
        return payload
    return entry


def _entry_type(entry: dict[str, Any]) -> str:
    raw = entry.get("type")
    return raw if isinstance(raw, str) else ""


def _parse_iso8601_to_epoch(ts: str) -> float | None:
    t = ts.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(t).timestamp()
    except ValueError:
        return None


def _entry_ts(entry: dict[str, Any]) -> float | None:
    for key in ("ts", "timestamp", "created_at", "updated_at"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            ts = float(value)
            if ts > _MILLIS_THRESHOLD:
                ts /= 1000.0
            return ts
        if key == "timestamp" and isinstance(value, str):
            v = _parse_iso8601_to_epoch(value)
            if v is not None:
                return v
    return None


def _entry_text(payload: dict[str, Any]) -> str | None:
    content = payload.get("content")
    parts: list[str] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") not in _PI_TEXT_BLOCK_TYPES:
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    if parts:
        return "".join(parts)
    text = payload.get("text")
    if isinstance(text, str) and text:
        return text
    return None


def _normalize_pi_todo_snapshot(todos: list[dict[str, Any]]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    counts = {"total": 0, "completed": 0, "in_progress": 0, "not_started": 0}

    for todo in todos:
        if not isinstance(todo, dict):
            continue
        title = todo.get("title")
        if not isinstance(title, str):
            continue
        title = title.strip()
        if not title:
            continue

        description = todo.get("description")
        if not isinstance(description, str):
            description = ""

        status = todo.get("status")
        if isinstance(status, str):
            status = status.strip()
        if not isinstance(status, str) or not status:
            status = "not-started"

        items.append({
            "id": todo.get("id"),
            "title": title,
            "description": description,
            "status": status,
        })
        counts["total"] += 1
        count_key = _PI_TODO_STATUS_KEYS.get(status)
        if count_key is not None:
            counts[count_key] += 1

    return {
        "items": items,
        "counts": counts,
        "progress_text": f'{counts["completed"]}/{counts["total"]} completed',
    }


def read_latest_pi_todo_snapshot(session_path: Path, *, max_scan_bytes: int | None = None) -> dict[str, Any] | None:
    if max_scan_bytes is None:
        max_scan_bytes = _PI_TODO_SCAN_MAX_BYTES
    max_scan_bytes = max(1, int(max_scan_bytes))
    scan = min(_PI_TODO_SCAN_START_BYTES, max_scan_bytes)

    while scan <= max_scan_bytes:
        for entry in reversed(_read_jsonl_tail(session_path, scan)):
            if not isinstance(entry, dict):
                continue
            payload = _payload_for_entry(entry)
            if not isinstance(payload, dict):
                continue
            if payload.get("role") != "toolResult":
                continue
            if payload.get("toolName") != "manage_todo_list":
                continue
            if payload.get("isError") is True:
                continue
            details = payload.get("details")
            if not isinstance(details, dict):
                continue
            todos = details.get("todos")
            if isinstance(todos, list):
                snapshot = _normalize_pi_todo_snapshot([todo for todo in todos if isinstance(todo, dict)])
                if todos and not snapshot["items"]:
                    continue
                return snapshot
        if scan == max_scan_bytes:
            break
        scan = min(scan * 2, max_scan_bytes)

    return None


def _message_event_info(entry: dict[str, Any]) -> tuple[str | None, str | None, bool]:
    payload = _payload_for_entry(entry)
    if not isinstance(payload, dict):
        return None, None, False
    payload_type = _entry_type(payload)
    if payload_type not in {"", "message"}:
        return None, None, False
    role = payload.get("role")
    if role not in {"user", "assistant"}:
        return None, None, False
    text = _entry_text(payload)
    if not isinstance(text, str) or not text:
        return None, None, False
    has_ts = _entry_ts(payload) is not None or _entry_ts(entry) is not None
    return role, text, has_ts


def _synthetic_ts_start_from_offset(*, offset: int) -> float:
    # Offset-based synthetic timestamps stay monotonic across delta polls
    # without rescanning the entire session file.
    return float(max(0, int(offset)))


def _tool_name(entry: dict[str, Any], payload: dict[str, Any]) -> str | None:
    for source in (payload, entry):
        for key in ("tool_name", "tool", "name", "type"):
            value = source.get(key)
            if isinstance(value, str) and value and value != "message":
                return value
    return None


def _coerce_tool_arguments(args: Any) -> dict[str, Any]:
    if isinstance(args, str):
        try:
            args = _json.loads(args)
        except (ValueError, TypeError):
            args = {}
    return args if isinstance(args, dict) else {}


def _normalized_bool_arg(args: dict[str, Any], *keys: str, default: bool = False) -> bool:
    for key in keys:
        if key in args:
            return bool(args.get(key))
    return default


def _cap_tool_names(tool_names: list[str]) -> list[str]:
    if len(tool_names) <= _PI_DIAG_TOOL_NAMES_LIMIT:
        return tool_names
    return tool_names[-_PI_DIAG_TOOL_NAMES_LIMIT :]


def _is_subagent_call(payload: dict[str, Any]) -> bool:
    if payload.get("role") != "assistant":
        return False
    content = payload.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("type") == "toolCall"
        and item.get("name") == "subagent"
        for item in content
    )


def _extract_all_subagent_calls(payload: dict[str, Any]) -> list[tuple[str | None, str, str | None]]:
    results: list[tuple[str | None, str, str | None]] = []
    content = payload.get("content", [])
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "toolCall" or item.get("name") != "subagent":
            continue
        call_id = item.get("id")
        args = _coerce_tool_arguments(item.get("arguments", {}))
        agent = args.get("agent", "subagent") if isinstance(args, dict) else "subagent"
        task = args.get("task") if isinstance(args, dict) else None
        results.append((
            call_id if isinstance(call_id, str) else None,
            agent if isinstance(agent, str) else "subagent",
            task if isinstance(task, str) else None,
        ))
    return results


def _is_subagent_result(payload: dict[str, Any]) -> bool:
    return payload.get("role") == "toolResult" and payload.get("toolName") == "subagent"


def _extract_subagent_output(payload: dict[str, Any]) -> str | None:
    content = payload.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "".join(parts)
    text = payload.get("text")
    return text if isinstance(text, str) and text else None


def _thinking_summary_text(item: dict[str, Any]) -> str | None:
    signature = item.get("thinkingSignature")
    data: dict[str, Any] | None = None
    if isinstance(signature, str) and signature:
        try:
            loaded = _json.loads(signature)
        except (ValueError, TypeError):
            loaded = None
        if isinstance(loaded, dict):
            data = loaded
    elif isinstance(signature, dict):
        data = signature
    if not isinstance(data, dict):
        return None
    summary = data.get("summary")
    if not isinstance(summary, list):
        return None
    parts: list[str] = []
    for entry in summary:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    if not parts:
        return None
    return "\n\n".join(parts)


def _event_ts_value(*, preferred_ts: float | None, fallback_ts: float) -> float:
    ts = float(fallback_ts)
    if isinstance(preferred_ts, (int, float)) and float(preferred_ts) >= ts:
        ts = float(preferred_ts)
    return ts


def _tool_result_details(payload: dict[str, Any]) -> dict[str, Any] | None:
    details = payload.get("details")
    return details if isinstance(details, dict) and details else None


def _normalize_ask_user_args(args: Any) -> dict[str, Any]:
    normalized = _coerce_tool_arguments(args)
    options = normalized.get("options")
    return {
        "question": normalized.get("question") if isinstance(normalized.get("question"), str) else "",
        "context": normalized.get("context") if isinstance(normalized.get("context"), str) else "",
        "options": list(options) if isinstance(options, list) else [],
        "allow_freeform": _normalized_bool_arg(normalized, "allow_freeform", "allowFreeform", default=True),
        "allow_multiple": _normalized_bool_arg(normalized, "allow_multiple", "allowMultiple"),
        "timeout_ms": next(
            (
                normalized.get(key)
                for key in ("timeout_ms", "timeoutMs", "timeout")
                if isinstance(normalized.get(key), int)
            ),
            None,
        ),
    }


def _ask_user_event(args: Any, *, call_id: str | None, ts: float, resolved: bool = False) -> dict[str, Any]:
    return {
        "type": "ask_user",
        "tool_call_id": call_id,
        **_normalize_ask_user_args(args),
        "resolved": resolved,
        "ts": float(ts),
    }


def _normalize_ask_user_answer(answer: Any, *, allow_multiple: bool) -> str | list[str] | None:
    if isinstance(answer, str):
        return answer
    if allow_multiple and isinstance(answer, list):
        normalized = [item for item in answer if isinstance(item, str)]
        return normalized
    return None


def _todo_snapshot_event(payload: dict[str, Any], *, ts: float) -> dict[str, Any] | None:
    details = _tool_result_details(payload)
    if not isinstance(details, dict):
        return None
    todos = details.get("todos")
    if not isinstance(todos, list):
        return None
    snapshot = _normalize_pi_todo_snapshot([todo for todo in todos if isinstance(todo, dict)])
    event: dict[str, Any] = {"type": "todo_snapshot", "ts": float(ts), **snapshot}
    operation = details.get("operation")
    if isinstance(operation, str) and operation.strip():
        event["operation"] = operation.strip()
    text = _extract_subagent_output(payload)
    if isinstance(text, str) and text.strip():
        event["text"] = text
    return event
def _flush_assistant_text_buffer(
    *,
    text_parts: list[str],
    events: list[dict[str, Any]],
    fallback_ts: float,
    preferred_ts: float | None = None,
) -> float:
    if not text_parts:
        return fallback_ts
    text = "".join(text_parts)
    text_parts.clear()
    if text:
        ts = float(fallback_ts)
        if isinstance(preferred_ts, (int, float)) and float(preferred_ts) >= ts:
            ts = float(preferred_ts)
        events.append({"role": "assistant", "text": text, "ts": ts})
        return ts + 1.0
    return fallback_ts


def normalize_pi_entries(
    entries: list[dict[str, Any]], *, ts_start: float = 0.0, include_system: bool = False
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, bool], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    meta_delta = {"thinking": 0, "tool": 0, "system": 0}
    flags = {"turn_start": False, "turn_end": False, "turn_aborted": False}
    tool_names: list[str] = []
    active_turn_last_tool: str | None = None
    fallback_ts = float(ts_start)
    pending_ask_user: dict[str, dict[str, Any]] = {}
    pending_subagents: dict[str, dict[str, Any]] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        payload = _payload_for_entry(entry)
        if not isinstance(payload, dict):
            continue
        entry_type = _entry_type(entry)
        payload_type = _entry_type(payload)
        source_event = payload.get("source_event") if isinstance(payload.get("source_event"), str) else None

        if source_event == "turn.started" or entry_type == "turn.started":
            flags["turn_start"] = True
            active_turn_last_tool = None
        if source_event in _PI_END_EVENT_TYPES or entry_type in _PI_END_EVENT_TYPES:
            flags["turn_end"] = True
        if source_event in _PI_ABORT_EVENT_TYPES or entry_type in _PI_ABORT_EVENT_TYPES:
            flags["turn_aborted"] = True

        entry_ts = _entry_ts(entry)
        payload_ts = _entry_ts(payload)

        # --- toolResult ---
        if payload.get("role") == "toolResult" and payload.get("toolName") == "ask_user":
            meta_delta["tool"] += 1
            tool_names.append("ask_user")
            active_turn_last_tool = "ask_user"
            call_id = payload.get("toolCallId")
            details = _tool_result_details(payload) or {}
            info = pending_ask_user.pop(call_id, None) if isinstance(call_id, str) else None
            ts = info["ts"] if info else _event_ts_value(preferred_ts=payload_ts or entry_ts, fallback_ts=fallback_ts)
            event: dict[str, Any] = info.copy() if info else _ask_user_event({}, call_id=call_id if isinstance(call_id, str) else None, ts=ts)
            answer = details.get("answer")
            event["answer"] = _normalize_ask_user_answer(answer, allow_multiple=bool(event.get("allow_multiple")))
            event["cancelled"] = bool(details.get("cancelled"))
            event["was_custom"] = bool(details.get("wasCustom"))
            event["resolved"] = True
            event["ts"] = float(ts)
            events.append(event)
            if not info:
                fallback_ts = ts + 0.1
            continue

        if _is_subagent_result(payload):
            call_id = payload.get("toolCallId")
            output = _extract_subagent_output(payload)
            info = pending_subagents.pop(call_id, None) if isinstance(call_id, str) else None
            ts = info["ts"] if info else _event_ts_value(preferred_ts=payload_ts or entry_ts, fallback_ts=fallback_ts)
            events.append({
                "type": "subagent",
                "agent": info["agent"] if info else "subagent",
                "task": info.get("task") if info else None,
                "output": output,
                "ts": ts,
            })
            if not info:
                fallback_ts = ts + 0.1
            continue

        if payload.get("role") == "toolResult":
            meta_delta["tool"] += 1
            tn = payload.get("toolName")
            if isinstance(tn, str) and tn:
                tool_names.append(tn)
                active_turn_last_tool = tn
            ts = _event_ts_value(preferred_ts=payload_ts or entry_ts, fallback_ts=fallback_ts)
            if isinstance(tn, str) and tn == "manage_todo_list":
                todo_event = _todo_snapshot_event(payload, ts=ts)
                if todo_event is not None:
                    events.append(todo_event)
                    fallback_ts = ts + 0.1
                    continue
            output = _extract_subagent_output(payload)
            details = _tool_result_details(payload)
            if isinstance(tn, str) and tn and ((isinstance(output, str) and output) or details):
                event: dict[str, Any] = {"type": "tool_result", "name": tn, "ts": ts}
                if isinstance(output, str):
                    event["text"] = output
                if payload.get("isError") is True:
                    event["is_error"] = True
                if details is not None:
                    event["details"] = details
                events.append(event)
            fallback_ts = ts + 0.1
            continue

        # --- assistant content blocks in original order ---
        if payload.get("role") == "assistant":
            content = payload.get("content")
            if isinstance(content, list):
                text_parts: list[str] = []
                assistant_ts = payload_ts if payload_ts is not None else entry_ts
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type in _PI_TEXT_BLOCK_TYPES:
                        text = item.get("text")
                        if isinstance(text, str) and text:
                            text_parts.append(text)
                        continue
                    if item_type == "thinking":
                        fallback_ts = _flush_assistant_text_buffer(
                            text_parts=text_parts,
                            events=events,
                            fallback_ts=fallback_ts,
                            preferred_ts=assistant_ts,
                        )
                        thinking_text = item.get("thinking")
                        summary = _thinking_summary_text(item)
                        body_text = thinking_text if isinstance(thinking_text, str) and thinking_text else summary
                        if isinstance(body_text, str) and body_text:
                            ts = _event_ts_value(preferred_ts=assistant_ts, fallback_ts=fallback_ts)
                            reasoning_event: dict[str, Any] = {"type": "reasoning", "text": body_text, "ts": ts}
                            if isinstance(summary, str) and summary and summary != body_text:
                                reasoning_event["summary"] = summary
                            events.append(reasoning_event)
                            meta_delta["thinking"] += 1
                            fallback_ts = ts + 0.1
                            assistant_ts = None
                        continue
                    if item_type != "toolCall":
                        continue
                    fallback_ts = _flush_assistant_text_buffer(
                        text_parts=text_parts,
                        events=events,
                        fallback_ts=fallback_ts,
                        preferred_ts=assistant_ts,
                    )
                    assistant_ts = None
                    tc_name = item.get("name")
                    if not isinstance(tc_name, str):
                        continue
                    call_id = item.get("id")
                    if tc_name == "ask_user":
                        ts = float(_event_ts_value(preferred_ts=assistant_ts, fallback_ts=fallback_ts))
                        if isinstance(call_id, str) and call_id:
                            pending_ask_user[call_id] = _ask_user_event(item.get("arguments"), call_id=call_id, ts=ts)
                            meta_delta["tool"] += 1
                            tool_names.append(tc_name)
                            active_turn_last_tool = tc_name
                            fallback_ts = ts + 0.1
                            continue
                        events.append(_ask_user_event(item.get("arguments"), call_id=None, ts=ts))
                        meta_delta["tool"] += 1
                        tool_names.append(tc_name)
                        active_turn_last_tool = tc_name
                        fallback_ts = ts + 0.1
                        continue
                    if tc_name == "subagent":
                        args = _coerce_tool_arguments(item.get("arguments", {}))
                        agent = args.get("agent", "subagent") if isinstance(args, dict) else "subagent"
                        task = args.get("task") if isinstance(args, dict) else None
                        if isinstance(call_id, str) and call_id:
                            pending_subagents[call_id] = {
                                "agent": agent if isinstance(agent, str) else "subagent",
                                "task": task if isinstance(task, str) else None,
                                "ts": float(_event_ts_value(preferred_ts=assistant_ts, fallback_ts=fallback_ts)),
                            }
                            fallback_ts += 0.1
                        continue
                    tool_event: dict[str, Any] = {
                        "type": "tool",
                        "name": tc_name,
                        "ts": float(_event_ts_value(preferred_ts=assistant_ts, fallback_ts=fallback_ts)),
                    }
                    events.append(tool_event)
                    meta_delta["tool"] += 1
                    tool_names.append(tc_name)
                    active_turn_last_tool = tc_name
                    fallback_ts += 0.1
                fallback_ts = _flush_assistant_text_buffer(
                    text_parts=text_parts,
                    events=events,
                    fallback_ts=fallback_ts,
                    preferred_ts=assistant_ts,
                )
                continue

        role, text, has_ts = _message_event_info(entry)
        if role is not None and text is not None:
            if role == "user":
                active_turn_last_tool = None
            if has_ts:
                ts = _entry_ts(payload)
                if ts is None:
                    ts = _entry_ts(entry)
                if ts is None:
                    ts = fallback_ts
                elif ts < fallback_ts:
                    ts = fallback_ts
            else:
                ts = fallback_ts
            fallback_ts = float(ts) + 1.0
            events.append({"role": role, "text": text, "ts": float(ts)})
            continue

        if entry_type in _PI_TOOL_EVENT_TYPES or payload_type in _PI_TOOL_EVENT_TYPES:
            meta_delta["tool"] += 1
            tool_name = _tool_name(entry, payload)
            if tool_name is not None:
                tool_names.append(tool_name)
                active_turn_last_tool = tool_name
            ev_name = tool_name or entry_type or payload_type
            if ev_name:
                ts = _event_ts_value(preferred_ts=entry_ts or payload_ts, fallback_ts=fallback_ts)
                events.append({"type": "tool", "name": ev_name, "ts": ts})
                fallback_ts = ts + 0.1
            continue

    pending_events: list[dict[str, Any]] = [*pending_ask_user.values()]
    for info in pending_subagents.values():
        pending_events.append({"type": "subagent", "agent": info["agent"], "task": info.get("task"), "output": None, "ts": info["ts"]})
    pending_events.sort(key=lambda event: float(event.get("ts", 0.0)))
    events.extend(pending_events)

    return events, meta_delta, flags, {"tool_names": _cap_tool_names(tool_names), "last_tool": active_turn_last_tool}


def _read_all_entries(session_path: Path) -> tuple[list[dict[str, Any]], int]:
    if not session_path.exists():
        return [], 0
    size = int(session_path.stat().st_size)
    offset = 0
    entries: list[dict[str, Any]] = []
    while offset < size:
        chunk, new_off = _read_jsonl_from_offset(session_path, offset, max_bytes=_PI_READ_MAX_BYTES)
        if new_off <= offset:
            break
        entries.extend(obj for obj in chunk if isinstance(obj, dict))
        offset = new_off
    return entries, offset


def read_pi_message_page(
    session_path: Path, *, limit: int, before: int, include_system: bool = False
) -> tuple[list[dict[str, Any]], int, bool, int, dict[str, Any]]:
    entries, new_off = _read_all_entries(session_path)
    events, _meta, _flags, diag = normalize_pi_entries(entries, include_system=include_system)
    total = len(events)
    end = max(0, total - max(0, int(before)))
    start = max(0, end - max(20, int(limit)))
    page = events[start:end]
    has_older = start > 0
    next_before = max(0, int(before)) + len(page) if has_older else 0
    return page, new_off, has_older, next_before, diag


def read_pi_message_delta(
    session_path: Path, *, offset: int, include_system: bool = False
) -> tuple[list[dict[str, Any]], int, dict[str, int], dict[str, bool], dict[str, Any]]:
    if not session_path.exists():
        return [], 0, {"thinking": 0, "tool": 0, "system": 0}, {"turn_start": False, "turn_end": False, "turn_aborted": False}, {"tool_names": [], "last_tool": None}
    entries, new_off = _read_jsonl_from_offset(session_path, int(offset), max_bytes=_PI_READ_MAX_BYTES)
    if new_off == int(offset):
        return [], new_off, {"thinking": 0, "tool": 0, "system": 0}, {"turn_start": False, "turn_end": False, "turn_aborted": False}, {"tool_names": [], "last_tool": None}
    ts_start = _synthetic_ts_start_from_offset(offset=int(offset))
    events, meta_delta, flags, diag = normalize_pi_entries(
        [obj for obj in entries if isinstance(obj, dict)],
        ts_start=ts_start,
        include_system=include_system,
    )
    return events, new_off, meta_delta, flags, diag


def is_pi_session_idle(session_path: Path) -> bool | None:
    """Check if the Pi session's last turn has ended (idle).

    Pi does not emit turn lifecycle events (turn.started / turn.completed).
    Instead we inspect the tail of the session file and check:

    * If the last turn-lifecycle event is a completion/abort → True (idle).
    * If the last turn-lifecycle event is turn.started → False (busy).
    * Otherwise fall back to message roles: if the last message with text
      content has role "assistant" the turn is done; if the last role is
      "user" (or no messages at all) Pi is still processing.
    * Returns None when no useful signal is found (caller should fall back
      to the broker's busy flag).
    """
    if not session_path.exists():
        return None
    try:
        size = session_path.stat().st_size
    except OSError:
        return None
    scan_bytes = min(size, 64 * 1024)
    offset = max(0, size - scan_bytes)
    try:
        entries, _ = _read_jsonl_from_offset(session_path, offset, max_bytes=scan_bytes)
    except Exception:
        return None

    _TERMINAL = _PI_END_EVENT_TYPES | _PI_ABORT_EVENT_TYPES
    last_role: str | None = None
    last_raw_role: str | None = None

    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        payload = _payload_for_entry(entry)
        entry_type = _entry_type(entry)
        source_event = payload.get("source_event") if isinstance(payload, dict) and isinstance(payload.get("source_event"), str) else None

        # Prefer explicit turn lifecycle events when available.
        if source_event in _TERMINAL or entry_type in _TERMINAL:
            return True
        if source_event == "turn.started" or entry_type == "turn.started":
            return False

        # Track the last message role that carried visible text.
        if last_role is None:
            role, text, _has_ts = _message_event_info(entry)
            if role is not None:
                last_role = role

        # Track any message role (including toolResult and text-less assistant).
        if last_raw_role is None and isinstance(payload, dict):
            raw_role = payload.get("role")
            if isinstance(raw_role, str) and raw_role in {"user", "assistant", "toolResult"}:
                last_raw_role = raw_role

    if last_role == "assistant":
        return True   # last text message is from assistant → idle
    if last_role == "user":
        return False  # user spoke last → Pi should be processing
    # No text-bearing message in scan range; fall back to raw roles.
    if last_raw_role == "toolResult":
        return False  # tool result without subsequent text → still processing
    if last_raw_role == "assistant":
        return False  # assistant with only toolCall blocks → still invoking tools
    return None       # no useful signal
