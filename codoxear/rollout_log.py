from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

from .constants import CONTEXT_WINDOW_BASELINE_TOKENS


def _parse_iso8601_to_epoch(ts: str) -> float | None:
    t = ts.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    return datetime.datetime.fromisoformat(t).timestamp()


def _event_ts(obj: dict[str, Any]) -> float | None:
    ts = obj.get("ts")
    if isinstance(ts, (int, float)):
        return float(ts)
    ts2 = obj.get("timestamp")
    if isinstance(ts2, (int, float)):
        return float(ts2)
    if isinstance(ts2, str):
        v = _parse_iso8601_to_epoch(ts2)
        if v is not None:
            return float(v)
    return None


def _context_percent_remaining(*, tokens_in_context: int, context_window: int) -> int:
    if context_window <= CONTEXT_WINDOW_BASELINE_TOKENS:
        return 0
    effective = context_window - CONTEXT_WINDOW_BASELINE_TOKENS
    used = max(tokens_in_context - CONTEXT_WINDOW_BASELINE_TOKENS, 0)
    remaining = max(effective - used, 0)
    return int(round((remaining / effective) * 100.0))


def _extract_token_update(objs: list[dict[str, Any]]) -> dict[str, Any] | None:
    # Prefer the newest token_count in this batch.
    for obj in reversed(objs):
        if obj.get("type") != "event_msg":
            continue
        p = obj.get("payload")
        if not isinstance(p, dict):
            raise ValueError("invalid token_count payload")
        if p.get("type") != "token_count":
            continue
        info = p.get("info")
        if not isinstance(info, dict) or not isinstance(info.get("total_token_usage"), dict):
            continue
        ctx = info.get("model_context_window")
        last = info.get("last_token_usage")
        if not isinstance(ctx, int) or not isinstance(last, dict):
            continue
        tt = last.get("total_tokens")
        if not isinstance(tt, int):
            continue
        return {
            "context_window": ctx,
            "tokens_in_context": tt,
            "tokens_remaining": max(ctx - tt, 0),
            "percent_remaining": _context_percent_remaining(tokens_in_context=tt, context_window=ctx),
            "baseline_tokens": CONTEXT_WINDOW_BASELINE_TOKENS,
            "as_of": obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None,
        }
    return None


def _read_jsonl_tail(path: Path, max_bytes: int) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, size - max_bytes)
        f.seek(start)
        data = f.read()

    if not data:
        return []
    if start > 0:
        nl = data.find(b"\n")
        if nl >= 0:
            data = data[nl + 1 :]

    out: list[dict[str, Any]] = []
    for line in data.splitlines():
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


def _find_latest_token_update(log_path: Path, max_scan_bytes: int = 32 * 1024 * 1024) -> dict[str, Any] | None:
    scan = min(256 * 1024, max_scan_bytes)
    if scan <= 0:
        return None
    while scan <= max_scan_bytes:
        token = _extract_token_update(_read_jsonl_tail(log_path, scan))
        if token is not None:
            return token
        scan *= 2
    return None


def _extract_chat_events(
    objs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, bool], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    total_thinking = 0
    total_tools = 0
    total_system = 0
    turn_start = False
    turn_end = False
    turn_aborted = False
    tool_names: set[str] = set()
    last_tool: str | None = None
    skip_next_assistant = 0

    def event_ts(o: dict[str, Any]) -> float | None:
        ts = o.get("ts")
        if isinstance(ts, (int, float)):
            return float(ts)
        ts2 = o.get("timestamp")
        if isinstance(ts2, (int, float)):
            return float(ts2)
        if isinstance(ts2, str):
            v = _parse_iso8601_to_epoch(ts2)
            if v is not None:
                return float(v)
        return None

    for obj in objs:
        typ = obj.get("type")
        if typ == "event_msg":
            p = obj.get("payload")
            if not isinstance(p, dict):
                raise ValueError("invalid event_msg payload")
            pt = p.get("type")
            if pt == "user_message":
                msg = p.get("message")
                if isinstance(msg, str):
                    turn_start = True
                    ets = event_ts(obj)
                    ev: dict[str, Any] = {"role": "user", "text": msg}
                    if ets is not None:
                        ev["ts"] = ets
                    events.append(ev)
                continue
            if pt == "agent_reasoning":
                total_thinking += 1
                continue
            if pt == "turn_aborted":
                turn_aborted = True
                continue
            if pt == "token_count":
                continue

        if typ == "response_item":
            p = obj.get("payload")
            if not isinstance(p, dict):
                raise ValueError("invalid response_item payload")
            pt = p.get("type")
            if pt == "message":
                role = p.get("role")
                if role in ("developer", "system"):
                    total_system += 1
                    continue
                if role == "assistant":
                    content = p.get("content")
                    if not isinstance(content, list):
                        raise ValueError("invalid assistant message content")
                    out_text_parts: list[str] = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                            out_text_parts.append(part["text"])
                    if out_text_parts:
                        text = "".join(out_text_parts)
                        if skip_next_assistant > 0:
                            skip_next_assistant -= 1
                            continue
                        ets = event_ts(obj)
                        ev2: dict[str, Any] = {"role": "assistant", "text": text}
                        if ets is not None:
                            ev2["ts"] = ets
                        events.append(ev2)
                    continue

            if pt == "reasoning":
                total_thinking += 1
                continue
            if pt == "function_call":
                nm = p.get("name")
                if isinstance(nm, str) and nm:
                    tool_names.add(nm)
                    last_tool = nm
                total_tools += 1
                continue
            if pt in (
                "function_call_output",
                "custom_tool_call",
                "custom_tool_call_output",
                "web_search_call",
                "local_shell_call",
            ):
                total_tools += 1
                continue

    return (
        events,
        {"thinking": total_thinking, "tool": total_tools, "system": total_system},
        {"turn_start": turn_start, "turn_end": turn_end, "turn_aborted": turn_aborted},
        {"tool_names": sorted(tool_names), "last_tool": last_tool},
    )


def _read_chat_tail_snapshot(
    log_path: Path,
    *,
    min_events: int,
    initial_scan_bytes: int,
    max_scan_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int, bool, int]:
    size = int(log_path.stat().st_size)
    scan = min(max(256 * 1024, int(initial_scan_bytes)), int(max_scan_bytes))
    if scan <= 0:
        return [], None, 0, True, size

    best_events: list[dict[str, Any]] = []
    best_token: dict[str, Any] | None = None
    while True:
        objs = _read_jsonl_tail(log_path, scan)
        events, _meta, _flags, _diag = _extract_chat_events(objs)
        best_events = events
        tok = _extract_token_update(objs)
        if tok is not None:
            best_token = tok
        if len(events) >= min_events or scan >= max_scan_bytes:
            break
        next_scan = min(scan * 2, max_scan_bytes)
        if next_scan <= scan:
            break
        scan = next_scan

    scan_complete = (size <= scan)
    return best_events, best_token, scan, scan_complete, size


def _read_chat_events_from_tail(
    log_path: Path,
    min_events: int = 120,
    max_scan_bytes: int = 128 * 1024 * 1024,
) -> list[dict[str, Any]]:
    events, _token, _scan_bytes, _scan_complete, _size = _read_chat_tail_snapshot(
        log_path,
        min_events=min_events,
        initial_scan_bytes=min(256 * 1024, max_scan_bytes),
        max_scan_bytes=max_scan_bytes,
    )
    return events


def _has_assistant_output_text(obj: dict[str, Any]) -> bool:
    p = obj.get("payload")
    if not isinstance(p, dict):
        raise ValueError("invalid response_item payload")
    if p.get("type") != "message" or p.get("role") != "assistant":
        return False
    content = p.get("content")
    if not isinstance(content, list):
        raise ValueError("invalid assistant message content")
    for part in content:
        if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str) and part.get("text"):
            return True
    return False


def _analyze_log_chunk(
    objs: list[dict[str, Any]],
) -> tuple[int, int, int, float | None, dict[str, Any] | None, list[dict[str, Any]]]:
    d_th = 0
    d_tools = 0
    d_sys = 0
    last_chat_ts: float | None = None
    token_update = _extract_token_update(objs)
    chat_events, _meta, _flags, _diag = _extract_chat_events(objs)

    for obj in objs:
        typ = obj.get("type")
        if typ == "event_msg":
            p = obj.get("payload")
            if not isinstance(p, dict):
                raise ValueError("invalid event_msg payload")
            pt = p.get("type")
            if pt == "agent_reasoning":
                d_th += 1
            if pt == "user_message":
                d_th = 0
                d_tools = 0
                d_sys = 0
                if isinstance(p.get("message"), str):
                    last_chat_ts = _event_ts(obj)
            if pt == "agent_message":
                msg = p.get("message")
                if isinstance(msg, str) and msg.strip():
                    last_chat_ts = _event_ts(obj)
        if typ == "response_item":
            p = obj.get("payload")
            if not isinstance(p, dict):
                raise ValueError("invalid response_item payload")
            pt = p.get("type")
            if pt == "reasoning":
                d_th += 1
            if pt in (
                "function_call",
                "function_call_output",
                "custom_tool_call",
                "custom_tool_call_output",
                "web_search_call",
                "local_shell_call",
            ):
                d_tools += 1
            if pt == "message" and p.get("role") in ("developer", "system"):
                d_sys += 1
            if _has_assistant_output_text(obj):
                last_chat_ts = _event_ts(obj)

    return d_th, d_tools, d_sys, last_chat_ts, token_update, chat_events


def _last_conversation_ts_from_tail(
    log_path: Path,
    *,
    max_scan_bytes: int,
) -> float | None:
    def event_ts(o: dict[str, Any]) -> float | None:
        ts = o.get("ts")
        if isinstance(ts, (int, float)):
            return float(ts)
        ts2 = o.get("timestamp")
        if isinstance(ts2, (int, float)):
            return float(ts2)
        if isinstance(ts2, str):
            v = _parse_iso8601_to_epoch(ts2)
            if v is not None:
                return float(v)
        return None

    def has_assistant_text(obj: dict[str, Any]) -> bool:
        p = obj.get("payload")
        if not isinstance(p, dict):
            raise ValueError("invalid response_item payload")
        if p.get("type") != "message" or p.get("role") != "assistant":
            return False
        content = p.get("content")
        if not isinstance(content, list):
            raise ValueError("invalid assistant message content")
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str) and part.get("text"):
                return True
        return False

    scan = 256 * 1024
    while True:
        objs = _read_jsonl_tail(log_path, scan)
        last_idx: int | None = None
        last_ts: float | None = None
        for i, obj in enumerate(objs):
            typ = obj.get("type")
            if typ == "event_msg":
                p = obj.get("payload")
                if not isinstance(p, dict):
                    raise ValueError("invalid event_msg payload")
                pt = p.get("type")
                if pt == "user_message" and isinstance(p.get("message"), str):
                    last_idx = i
                    last_ts = event_ts(obj)
                    continue
                if pt == "agent_message":
                    msg = p.get("message")
                    if isinstance(msg, str) and msg.strip():
                        last_idx = i
                        last_ts = event_ts(obj)
                        continue
            if typ == "response_item" and has_assistant_text(obj):
                last_idx = i
                last_ts = event_ts(obj)
                continue

        if last_idx is not None:
            return last_ts
        if scan >= max_scan_bytes:
            return None
        scan *= 2


def _compute_idle_from_log(path: Path, max_scan_bytes: int = 8 * 1024 * 1024) -> bool | None:
    sz = int(path.stat().st_size)

    scan = min(256 * 1024, max_scan_bytes)
    if scan <= 0:
        return None
    objs: list[dict[str, Any]] = []
    saw_user = False
    turn_open = False
    turn_has_completion_candidate = False
    last_terminal_event: str | None = None

    def has_assistant_text(obj: dict[str, Any]) -> bool:
        p = obj.get("payload")
        if not isinstance(p, dict):
            raise ValueError("invalid response_item payload")
        if p.get("type") != "message" or p.get("role") != "assistant":
            return False
        content = p.get("content")
        if not isinstance(content, list):
            raise ValueError("invalid assistant message content")
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str) and part.get("text"):
                return True
        return False

    while True:
        objs = _read_jsonl_tail(path, scan)
        for obj in objs:
            typ = obj.get("type")
            if typ == "event_msg":
                p = obj.get("payload")
                if not isinstance(p, dict):
                    raise ValueError("invalid event_msg payload")
                pt = p.get("type")
                if pt == "user_message" and isinstance(p.get("message"), str):
                    saw_user = True
                    turn_open = True
                    turn_has_completion_candidate = False
                    last_terminal_event = "user"
                    continue
                if pt == "agent_message":
                    msg = p.get("message")
                    if isinstance(msg, str) and msg.strip():
                        last_terminal_event = "assistant"
                        continue
                if pt in ("turn_aborted", "thread_rolled_back"):
                    turn_open = False
                    turn_has_completion_candidate = False
                    last_terminal_event = "aborted"
                    continue
                if pt == "agent_reasoning" and turn_open:
                    turn_has_completion_candidate = False
                    continue
            if typ == "response_item":
                p = obj.get("payload")
                if not isinstance(p, dict):
                    raise ValueError("invalid response_item payload")
                pt = p.get("type")
                if has_assistant_text(obj):
                    if turn_open:
                        turn_has_completion_candidate = True
                    last_terminal_event = "assistant"
                    continue
                if pt in (
                    "reasoning",
                    "function_call",
                    "function_call_output",
                    "custom_tool_call",
                    "custom_tool_call_output",
                    "web_search_call",
                    "local_shell_call",
                ) and turn_open:
                    turn_has_completion_candidate = False
                    continue

        if saw_user or (last_terminal_event is not None) or scan >= max_scan_bytes:
            break
        scan *= 2

    if not objs:
        return None

    if (not saw_user) and (last_terminal_event is None):
        return True if sz <= 128 * 1024 else False

    if turn_open:
        return bool(turn_has_completion_candidate)

    if last_terminal_event in ("assistant", "aborted"):
        return True
    if last_terminal_event == "user":
        return False
    return False


def _last_chat_role_ts_from_tail(
    path: Path,
    *,
    max_scan_bytes: int,
) -> tuple[str, float] | None:
    def event_ts(o: dict[str, Any]) -> float | None:
        ts = o.get("ts")
        if isinstance(ts, (int, float)):
            return float(ts)
        ts2 = o.get("timestamp")
        if isinstance(ts2, (int, float)):
            return float(ts2)
        if isinstance(ts2, str):
            v = _parse_iso8601_to_epoch(ts2)
            if v is not None:
                return float(v)
        return None

    def has_assistant_text(obj: dict[str, Any]) -> bool:
        p = obj.get("payload")
        if not isinstance(p, dict):
            raise ValueError("invalid response_item payload")
        if p.get("type") != "message" or p.get("role") != "assistant":
            return False
        content = p.get("content")
        if not isinstance(content, list):
            raise ValueError("invalid assistant message content")
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str) and part.get("text"):
                return True
        return False

    scan = 256 * 1024
    while scan <= max_scan_bytes:
        objs = _read_jsonl_tail(path, scan)
        last_user: tuple[int, float | None] | None = None
        last_assistant: tuple[int, float | None] | None = None
        for i, obj in enumerate(objs):
            typ = obj.get("type")
            if typ == "event_msg":
                p = obj.get("payload")
                if not isinstance(p, dict):
                    raise ValueError("invalid event_msg payload")
                pt = p.get("type")
                if pt == "user_message" and isinstance(p.get("message"), str):
                    last_user = (i, event_ts(obj))
                    continue
                if pt == "agent_message":
                    msg = p.get("message")
                    if isinstance(msg, str) and msg.strip():
                        last_assistant = (i, event_ts(obj))
                        continue
            if typ == "response_item" and has_assistant_text(obj):
                last_assistant = (i, event_ts(obj))

        best: tuple[str, tuple[int, float | None]] | None = None
        if last_user is not None:
            best = ("user", last_user)
        if last_assistant is not None:
            if best is None or last_assistant[0] > best[1][0]:
                best = ("assistant", last_assistant)
        if best is not None:
            role, (_i, ts) = best
            if ts is None:
                return None
            return (role, float(ts))
        scan *= 2
    return None

