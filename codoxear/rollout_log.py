from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from typing import Iterator

from .constants import CONTEXT_WINDOW_BASELINE_TOKENS
from .pi_log import pi_assistant_thinking_count
from .pi_log import pi_assistant_tool_use_count
from .pi_log import pi_assistant_text
from .pi_log import pi_assistant_is_final_turn_end
from .pi_log import pi_message_role
from .pi_log import pi_token_update
from .pi_log import pi_user_text
from .voice_push import ClassifiedAssistantMessage


_OAI_MEM_CITATION_TAIL_RE = re.compile(r"\s*<oai-mem-citation>\s*.*?</oai-mem-citation>\s*\Z", re.DOTALL)


def _parse_iso8601_to_epoch(ts: str) -> float | None:
    t = ts.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(t).timestamp()
    except ValueError:
        return None


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


def _strip_oai_mem_citation_tail(text: str) -> str:
    # Delivery notifications should follow the assistant reply itself, not the appended memory-citation envelope.
    return _OAI_MEM_CITATION_TAIL_RE.sub("", text)


def _sidebar_conversation_ts(obj: dict[str, Any]) -> float | None:
    typ = obj.get("type")
    if typ == "event_msg":
        p = obj.get("payload")
        if not isinstance(p, dict):
            raise ValueError("invalid event_msg payload")
        pt = p.get("type")
        if pt == "user_message" and isinstance(p.get("message"), str):
            return _event_ts(obj)
        if pt in ("task_complete", "turn_complete"):
            last_msg = p.get("last_agent_message")
            if isinstance(last_msg, str) and last_msg.strip():
                return _event_ts(obj)
        if pt == "agent_message":
            msg = p.get("message")
            phase = p.get("phase")
            if isinstance(msg, str) and msg.strip() and phase == "final_answer":
                return _event_ts(obj)
        return None

    if typ == "message":
        if pi_user_text(obj):
            return _event_ts(obj)
        if pi_assistant_text(obj):
            return _event_ts(obj)
        return None

    if typ == "response_item":
        p = obj.get("payload")
        if not isinstance(p, dict):
            raise ValueError("invalid response_item payload")
        if p.get("type") != "message" or p.get("role") != "assistant":
            return None
        phase = p.get("phase")
        end_turn = p.get("end_turn")
        if phase != "final_answer" and end_turn is not True:
            return None
        content = p.get("content")
        if not isinstance(content, list):
            raise ValueError("invalid assistant message content")
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str) and part.get("text"):
                return _event_ts(obj)
        return None

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
        pi_token = pi_token_update(obj)
        if pi_token is not None:
            return pi_token
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


def _pi_message_keeps_turn_busy(obj: dict[str, Any]) -> bool:
    role = pi_message_role(obj)
    if role == "toolResult":
        return True
    return (pi_assistant_thinking_count(obj) > 0) or (pi_assistant_tool_use_count(obj) > 0)


def _parse_jsonl_line(raw_line: bytes | str) -> dict[str, Any] | None:
    if isinstance(raw_line, bytes):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            return None
    else:
        line = raw_line
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


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
        obj = _parse_jsonl_line(line)
        if obj is not None:
            out.append(obj)
    return out


def _iter_jsonl_objects_reverse(path: Path, *, block_bytes: int = 64 * 1024) -> Iterator[dict[str, Any]]:
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        offset = f.tell()
        carry = b""
        while offset > 0:
            read_size = min(block_bytes, offset)
            offset -= read_size
            f.seek(offset)
            chunk = f.read(read_size)
            data = chunk + carry
            parts = data.split(b"\n")
            if offset > 0:
                carry = parts[0]
                parts = parts[1:]
            else:
                carry = b""
            for raw_line in reversed(parts):
                line = raw_line.rstrip(b"\r")
                if not line:
                    continue
                obj = _parse_jsonl_line(line)
                if obj is not None:
                    yield obj
        if carry:
            line = carry.rstrip(b"\r")
            if line:
                obj = _parse_jsonl_line(line)
                if obj is not None:
                    yield obj


def _find_latest_token_update(log_path: Path, max_scan_bytes: int = 32 * 1024 * 1024) -> dict[str, Any] | None:
    scan = min(256 * 1024, max_scan_bytes)
    if scan <= 0:
        return None
    while scan <= max_scan_bytes:
        token = _extract_token_update(_read_jsonl_tail(log_path, scan))
        if token is not None:
            return token
        scan *= 2


def _find_latest_turn_context(log_path: Path, max_scan_bytes: int = 8 * 1024 * 1024) -> dict[str, Any] | None:
    scan = min(256 * 1024, max_scan_bytes)
    if scan <= 0:
        return None
    while scan <= max_scan_bytes:
        objs = _read_jsonl_tail(log_path, scan)
        for obj in reversed(objs):
            if not isinstance(obj, dict):
                continue
            if obj.get("type") != "turn_context":
                continue
            payload = obj.get("payload")
            if isinstance(payload, dict):
                return payload
        scan *= 2
    return None
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

    def text_message_id(*, message_class: str, text: str, ts: float | None) -> str:
        ts_ms = int(round(ts * 1000.0)) if isinstance(ts, (int, float)) else None
        payload = json.dumps({"class": message_class, "text": " ".join(text.split()), "ts_ms": ts_ms}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    for obj in objs:
        typ = obj.get("type")
        if typ == "message":
            user_text = pi_user_text(obj)
            if isinstance(user_text, str) and user_text:
                turn_start = True
                ets = event_ts(obj)
                evp: dict[str, Any] = {"role": "user", "text": user_text}
                if ets is not None:
                    evp["ts"] = ets
                events.append(evp)
                continue

            assistant_text = pi_assistant_text(obj)
            tool_count = pi_assistant_tool_use_count(obj)
            thinking_count = pi_assistant_thinking_count(obj)
            if thinking_count > 0:
                total_thinking += thinking_count
            if tool_count > 0:
                total_tools += tool_count
                tool_names.add("pi_tool")
                last_tool = "pi_tool"
            if isinstance(assistant_text, str) and assistant_text:
                ets = event_ts(obj)
                message_class = "final_response" if pi_assistant_is_final_turn_end(obj) else "narration"
                if message_class == "final_response":
                    turn_end = True
                eva: dict[str, Any] = {
                    "role": "assistant",
                    "text": assistant_text,
                    "message_class": message_class,
                    "message_id": text_message_id(message_class=message_class, text=assistant_text, ts=ets),
                }
                if ets is not None:
                    eva["ts"] = ets
                events.append(eva)
            continue

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
            if pt in ("task_complete", "turn_complete"):
                turn_end = True
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
                        ets = event_ts(obj)
                        message_class = "final_response" if (p.get("phase") == "final_answer" or p.get("end_turn") is True) else "narration"
                        ev2: dict[str, Any] = {
                            "role": "assistant",
                            "text": text,
                            "message_class": message_class,
                            "message_id": text_message_id(message_class=message_class, text=text, ts=ets),
                        }
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
                tool_ev: dict[str, Any] = {"type": "tool", "name": nm or "tool"}
                ets_t = event_ts(obj)
                if ets_t is not None:
                    tool_ev["ts"] = ets_t
                events.append(tool_ev)
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


def _extract_delivery_messages(objs: list[dict[str, Any]]) -> list[ClassifiedAssistantMessage]:
    out: list[ClassifiedAssistantMessage] = []
    seen: set[str] = set()
    last_text_key: tuple[str, str] | None = None

    def _text_message_id(*, message_class: str, text: str, ts: float | None) -> str:
        ts_ms = int(round(ts * 1000.0)) if isinstance(ts, (int, float)) else None
        payload = json.dumps({"class": message_class, "text": " ".join(text.split()), "ts_ms": ts_ms}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    for obj in objs:
        if not isinstance(obj, dict):
            continue
        typ = obj.get("type")
        message_class: str | None = None
        text = ""
        if typ == "message":
            text = pi_assistant_text(obj) or ""
            if not text.strip():
                continue
            message_class = "final_response" if pi_assistant_is_final_turn_end(obj) else "narration"
        elif typ == "event_msg":
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("invalid event_msg payload")
            if payload.get("type") != "agent_message":
                continue
            message = payload.get("message")
            if not isinstance(message, str) or not message.strip():
                continue
            text = message
            message_class = "final_response" if payload.get("phase") == "final_answer" else "narration"
        elif typ == "response_item":
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("invalid response_item payload")
            if payload.get("type") != "message" or payload.get("role") != "assistant":
                continue
            content = payload.get("content")
            if not isinstance(content, list):
                raise ValueError("invalid assistant message content")
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            text = "".join(text_parts)
            if not text.strip():
                continue
            message_class = "final_response" if (payload.get("phase") == "final_answer" or payload.get("end_turn") is True) else "narration"
        else:
            continue
        text = _strip_oai_mem_citation_tail(text)
        if not text.strip():
            continue
        ts = _event_ts(obj)
        normalized_text = " ".join(text.split())
        text_key = (str(message_class), normalized_text)
        if last_text_key == text_key:
            continue
        message_id = _text_message_id(message_class=message_class, text=text, ts=ts)
        if message_id in seen:
            continue
        seen.add(message_id)
        last_text_key = text_key
        out.append(ClassifiedAssistantMessage(message_id=message_id, message_class=message_class, text=text, ts=ts))
    return out


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
    if obj.get("type") == "message":
        return bool(pi_assistant_text(obj))
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
        sidebar_ts = _sidebar_conversation_ts(obj)
        if sidebar_ts is not None:
            last_chat_ts = sidebar_ts
        if typ == "message":
            if pi_user_text(obj):
                d_th = 0
                d_tools = 0
                d_sys = 0
                continue
            d_th += pi_assistant_thinking_count(obj)
            d_tools += pi_assistant_tool_use_count(obj)
            continue
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

    return d_th, d_tools, d_sys, last_chat_ts, token_update, chat_events


def _last_conversation_ts_from_tail(
    log_path: Path,
    *,
    max_scan_bytes: int | None = None,
) -> float | None:
    # Keep the argument for compatibility with older callers, but recover the
    # last conversation timestamp exactly by scanning JSONL records backward.
    _ = max_scan_bytes
    for obj in _iter_jsonl_objects_reverse(log_path):
        ts = _sidebar_conversation_ts(obj)
        if ts is not None:
            return ts
    return None


def _compute_idle_from_log(path: Path, max_scan_bytes: int = 8 * 1024 * 1024) -> bool | None:
    sz = int(path.stat().st_size)

    scan = min(256 * 1024, max_scan_bytes)
    if scan <= 0:
        return None
    objs: list[dict[str, Any]] = []
    saw_terminal_signal = False
    idle = True

    while True:
        objs = _read_jsonl_tail(path, scan)
        saw_terminal_signal = False
        idle = True
        for obj in objs:
            typ = obj.get("type")
            if typ == "message":
                if pi_user_text(obj):
                    saw_terminal_signal = True
                    idle = False
                    continue
                if pi_assistant_text(obj):
                    saw_terminal_signal = True
                    idle = pi_assistant_is_final_turn_end(obj)
                    continue
                if _pi_message_keeps_turn_busy(obj):
                    saw_terminal_signal = True
                    idle = False
                    continue
            if typ == "event_msg":
                p = obj.get("payload")
                if not isinstance(p, dict):
                    raise ValueError("invalid event_msg payload")
                pt = p.get("type")
                if pt == "user_message" and isinstance(p.get("message"), str):
                    saw_terminal_signal = True
                    idle = False
                    continue
                if pt == "agent_message":
                    msg = p.get("message")
                    if isinstance(msg, str) and msg.strip():
                        saw_terminal_signal = True
                        idle = False
                    continue
                if pt == "agent_reasoning":
                    saw_terminal_signal = True
                    idle = False
                    continue
                if pt in ("turn_aborted", "thread_rolled_back", "task_complete", "turn_complete"):
                    saw_terminal_signal = True
                    idle = True
                    continue
            if typ == "response_item":
                p = obj.get("payload")
                if not isinstance(p, dict):
                    raise ValueError("invalid response_item payload")
                pt = p.get("type")
                if _has_assistant_output_text(obj):
                    saw_terminal_signal = True
                    idle = (p.get("end_turn") is True)
                    continue
                if pt == "reasoning":
                    saw_terminal_signal = True
                    idle = False
                    continue
                if pt in (
                    "function_call",
                    "function_call_output",
                    "custom_tool_call",
                    "custom_tool_call_output",
                    "web_search_call",
                    "local_shell_call",
                ):
                    saw_terminal_signal = True
                    idle = False
                    continue

        if saw_terminal_signal or scan >= max_scan_bytes:
            break
        scan *= 2

    if not objs:
        return None

    if not saw_terminal_signal:
        return True if sz <= 128 * 1024 else False

    return idle


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

    scan = 256 * 1024
    while scan <= max_scan_bytes:
        objs = _read_jsonl_tail(path, scan)
        last_user: tuple[int, float | None] | None = None
        last_assistant: tuple[int, float | None] | None = None
        for i, obj in enumerate(objs):
            typ = obj.get("type")
            if typ == "message":
                if pi_user_text(obj):
                    last_user = (i, event_ts(obj))
                    continue
                if pi_assistant_text(obj) or _pi_message_keeps_turn_busy(obj):
                    last_assistant = (i, event_ts(obj))
                    continue
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
            if typ == "response_item" and _has_assistant_output_text(obj):
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
