from __future__ import annotations

from pathlib import Path
from typing import Any

from .cc_log import cc_assistant_is_final_turn_end
from .cc_log import cc_assistant_pending_tool_use_ids
from .cc_log import cc_assistant_text
from .cc_log import cc_assistant_thinking_count
from .cc_log import cc_apply_tool_result_to_pending
from .cc_log import cc_assistant_tool_use_count
from .cc_log import cc_current_turn_state_before
from .cc_log import cc_is_turn_end
from .cc_log import cc_message_role
from .cc_log import cc_user_text
from .pi_log import pi_assistant_thinking_count
from .pi_log import pi_assistant_tool_use_count
from .pi_log import pi_assistant_error_text
from .pi_log import pi_assistant_is_aborted_turn
from .pi_log import pi_assistant_text
from .pi_log import pi_assistant_is_final_turn_end
from .pi_log import pi_message_role
from .pi_log import pi_user_text
from .voice_push_state import ClassifiedAssistantMessage
from .rollout_jsonl import JsonlRecord
from .rollout_jsonl import _iter_jsonl_objects_reverse
from .rollout_jsonl import _iter_jsonl_records_reverse
from .rollout_jsonl import _parse_jsonl_line
from .rollout_jsonl import _read_jsonl_records_from_offset
from .rollout_jsonl import _read_jsonl_tail
from .rollout_events import _codex_error_affects_turn_status
from .rollout_events import _codex_event_text
from .rollout_events import _event_ts
from .rollout_events import _parse_iso8601_to_epoch
from .rollout_events import _strip_oai_mem_citation_tail
from .rollout_events import _text_message_id
from .rollout_chat_events import _cc_message_keeps_turn_busy
from .rollout_chat_events import _chat_assistant_dedupe_key
from .rollout_chat_events import _dedupe_assistant_chat_events
from .rollout_chat_events import _pi_message_keeps_turn_busy
from .rollout_chat_events import _sidebar_conversation_ts
from .rollout_chat_events import _single_chat_event
from .rollout_chat_events import _update_cc_pending_tool_ids
from .rollout_tokens import _extract_token_update
from .rollout_tokens import _find_latest_token_update
from .rollout_tokens import _find_latest_turn_context
from .rollout_delivery import _extract_delivery_messages
from .rollout_chat_batch import _extract_chat_events


def _with_chat_position(event: dict[str, Any], *, before_byte: int | None = None) -> dict[str, Any]:
    if before_byte is None:
        return event
    out = dict(event)
    out["_before_byte"] = int(before_byte)
    return out


















def _cc_pending_tool_ids_before(log_path: Path, before: int, *, max_scan_bytes: int | None = None) -> set[str]:
    before = max(0, int(before))
    if before <= 0 or (max_scan_bytes is not None and max_scan_bytes <= 0):
        return set()
    lower_bound = max(0, before - int(max_scan_bytes)) if max_scan_bytes is not None else 0
    newest_first: list[JsonlRecord] = []
    for record in _iter_jsonl_records_reverse(log_path, before=before):
        if max_scan_bytes is not None and record.end <= lower_bound:
            break
        newest_first.append(record)
        if record.obj.get("type") == "user" and cc_user_text(record.obj):
            break
    pending: set[str] = set()
    for record in reversed(newest_first):
        _update_cc_pending_tool_ids(record.obj, pending)
    return pending


def _read_chat_page_reverse(
    log_path: Path,
    *,
    limit: int,
    before_byte: int | None = None,
    skip_events: int = 0,
) -> tuple[list[dict[str, Any]], int, bool, int]:
    size = int(log_path.stat().st_size)
    end = size if before_byte is None else max(0, min(int(before_byte), size))
    page_limit = max(0, int(limit))
    skip = max(0, int(skip_events))
    if page_limit <= 0 or end <= 0:
        return [], 0, False, size

    newest_first_records: list[JsonlRecord] = []
    skipped = 0
    kept_events = 0
    has_older = False
    for record in _iter_jsonl_records_reverse(log_path, before=end):
        event = _single_chat_event(record.obj)
        if event is not None:
            if skipped < skip:
                skipped += 1
                continue
            if kept_events >= page_limit:
                has_older = True
                break
            kept_events += 1
        if skipped >= skip:
            newest_first_records.append(record)

    records = list(reversed(newest_first_records))
    initial_pending = _cc_pending_tool_ids_before(log_path, records[0].start) if records else set()
    events = _extract_positioned_chat_events(records, initial_cc_pending_tool_ids=initial_pending)
    next_before = records[0].start if records else 0
    return events, next_before, has_older, size


def _extract_positioned_chat_events(
    records: list[JsonlRecord],
    *,
    initial_cc_pending_tool_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cc_pending_tool_ids = set(initial_cc_pending_tool_ids or set())
    for record in records:
        event = _single_chat_event(record.obj, cc_pending_tool_ids=cc_pending_tool_ids)
        if event is not None:
            events.append(_with_chat_position(event, before_byte=record.start))
    return _dedupe_assistant_chat_events(events)


def _read_chat_tail_page(log_path: Path, *, limit: int) -> tuple[list[dict[str, Any]], int, int, bool]:
    events, before_byte, has_older, after_byte = _read_chat_page_reverse(log_path, limit=limit, before_byte=None, skip_events=0)
    return events, before_byte, after_byte, has_older


def _read_chat_history_page(log_path: Path, *, before_byte: int, limit: int) -> tuple[list[dict[str, Any]], int, bool]:
    events, next_before, has_older, _after_byte = _read_chat_page_reverse(
        log_path,
        limit=limit,
        before_byte=before_byte,
        skip_events=0,
    )
    return events, next_before, has_older


def _read_chat_live_delta(
    log_path: Path,
    *,
    after_byte: int,
    max_bytes: int = 2 * 1024 * 1024,
) -> tuple[list[dict[str, Any]], int, dict[str, int], dict[str, bool], dict[str, Any], dict[str, Any] | None]:
    records, next_after = _read_jsonl_records_from_offset(log_path, after_byte, max_bytes=max_bytes)
    initial_pending = _cc_pending_tool_ids_before(log_path, after_byte) if records and after_byte > 0 else set()
    events = _extract_positioned_chat_events(records, initial_cc_pending_tool_ids=initial_pending)
    objs = [record.obj for record in records]
    _events, meta, flags, diag = _extract_chat_events(objs, initial_cc_pending_tool_ids=initial_pending)
    token_update = _extract_token_update(objs)
    return events, next_after, meta, flags, diag, token_update










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
    if obj.get("type") == "assistant":
        return bool(cc_assistant_text(obj))
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
        if typ == "user":
            if cc_user_text(obj):
                d_th = 0
                d_tools = 0
                d_sys = 0
                continue
            if cc_message_role(obj) == "toolResult":
                d_tools += 1
                continue
        if typ == "assistant":
            d_th += cc_assistant_thinking_count(obj)
            d_tools += cc_assistant_tool_use_count(obj)
            continue
        if typ == "system" and cc_is_turn_end(obj):
            d_sys += 1
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


def _compute_cc_idle_from_current_turn(path: Path) -> bool | None:
    try:
        _pending, idle = cc_current_turn_state_before(path, int(path.stat().st_size))
    except FileNotFoundError:
        return None
    return idle


def _compute_idle_from_log(path: Path, max_scan_bytes: int = 8 * 1024 * 1024) -> bool | None:
    cc_idle = _compute_cc_idle_from_current_turn(path)
    if cc_idle is not None:
        return cc_idle

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
        cc_pending_tool_ids: set[str] = set()
        cc_seen_turn_context = False
        cc_seen_user_turn_start = False
        cc_terminal_without_context = False
        for obj in objs:
            typ = obj.get("type")
            if typ == "message":
                if pi_user_text(obj):
                    saw_terminal_signal = True
                    idle = False
                    continue
                if pi_assistant_is_aborted_turn(obj):
                    saw_terminal_signal = True
                    idle = True
                    continue
                if pi_assistant_text(obj):
                    saw_terminal_signal = True
                    idle = pi_assistant_is_final_turn_end(obj)
                    continue
                if pi_assistant_error_text(obj):
                    saw_terminal_signal = True
                    idle = True
                    continue
                if _pi_message_keeps_turn_busy(obj):
                    saw_terminal_signal = True
                    idle = False
                    continue
            if typ == "user":
                if cc_user_text(obj):
                    cc_seen_turn_context = True
                    cc_seen_user_turn_start = True
                    cc_pending_tool_ids.clear()
                    saw_terminal_signal = True
                    idle = False
                    continue
                if cc_message_role(obj) == "toolResult":
                    cc_seen_turn_context = True
                    cc_apply_tool_result_to_pending(obj, cc_pending_tool_ids)
                    saw_terminal_signal = True
                    idle = False
                    continue
                if _cc_message_keeps_turn_busy(obj):
                    saw_terminal_signal = True
                    idle = False
                    continue
            if typ == "assistant":
                tool_use_count = cc_assistant_tool_use_count(obj)
                if tool_use_count > 0:
                    cc_seen_turn_context = True
                    cc_pending_tool_ids.update(cc_assistant_pending_tool_use_ids(obj))
                if cc_assistant_text(obj):
                    cc_seen_turn_context = True
                    saw_terminal_signal = True
                    if cc_assistant_is_final_turn_end(obj) and not cc_pending_tool_ids:
                        if not cc_seen_user_turn_start:
                            cc_terminal_without_context = True
                        idle = True
                    else:
                        idle = False
                    continue
                if tool_use_count > 0 or cc_assistant_thinking_count(obj) > 0:
                    cc_seen_turn_context = True
                    saw_terminal_signal = True
                    idle = False
                    continue
                if _cc_message_keeps_turn_busy(obj):
                    saw_terminal_signal = True
                    idle = False
                    continue
            if typ == "system" and cc_is_turn_end(obj):
                saw_terminal_signal = True
                if cc_pending_tool_ids:
                    idle = False
                else:
                    if not cc_seen_user_turn_start:
                        cc_terminal_without_context = True
                    idle = True
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
                if pt == "error":
                    saw_terminal_signal = True
                    idle = _codex_error_affects_turn_status(p)
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

        if saw_terminal_signal and not (cc_terminal_without_context and scan < min(sz, max_scan_bytes)):
            if cc_terminal_without_context and scan >= min(sz, max_scan_bytes):
                idle = False
            break
        if scan >= max_scan_bytes:
            if cc_terminal_without_context:
                idle = False
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
    final_assistant_only: bool = False,
) -> tuple[str, float] | None:
    scan = 256 * 1024
    while scan <= max_scan_bytes:
        objs = _read_jsonl_tail(path, scan)
        last_user: tuple[int, float | None] | None = None
        last_assistant: tuple[int, float | None] | None = None
        last_nonfinal_assistant: tuple[int, float | None] | None = None

        def remember_assistant(idx: int, ts: float | None, *, final: bool) -> None:
            nonlocal last_assistant, last_nonfinal_assistant
            if final:
                last_assistant = (idx, ts)
            elif final_assistant_only:
                last_nonfinal_assistant = (idx, ts)
            else:
                last_assistant = (idx, ts)

        for i, obj in enumerate(objs):
            typ = obj.get("type")
            if typ == "message":
                if pi_user_text(obj):
                    last_user = (i, _event_ts(obj))
                    continue
                if pi_assistant_is_aborted_turn(obj):
                    continue
                if pi_assistant_text(obj) or pi_assistant_error_text(obj) or _pi_message_keeps_turn_busy(obj):
                    remember_assistant(i, _event_ts(obj), final=pi_assistant_is_final_turn_end(obj))
                    continue
            if typ == "user":
                if cc_user_text(obj):
                    last_user = (i, _event_ts(obj))
                    continue
                if _cc_message_keeps_turn_busy(obj):
                    remember_assistant(i, _event_ts(obj), final=False)
                    continue
            if typ == "assistant":
                if cc_assistant_text(obj) or _cc_message_keeps_turn_busy(obj):
                    remember_assistant(i, _event_ts(obj), final=cc_assistant_is_final_turn_end(obj))
                    continue
            if typ == "event_msg":
                p = obj.get("payload")
                if not isinstance(p, dict):
                    raise ValueError("invalid event_msg payload")
                pt = p.get("type")
                if pt == "user_message" and isinstance(p.get("message"), str):
                    last_user = (i, _event_ts(obj))
                    continue
                if pt in ("task_complete", "turn_complete"):
                    last_msg = p.get("last_agent_message")
                    if isinstance(last_msg, str) and last_msg.strip():
                        remember_assistant(i, _event_ts(obj), final=True)
                    continue
                if pt == "agent_message":
                    msg = p.get("message")
                    if isinstance(msg, str) and msg.strip():
                        remember_assistant(i, _event_ts(obj), final=p.get("phase") == "final_answer")
                        continue
            if typ == "response_item" and _has_assistant_output_text(obj):
                payload = obj.get("payload")
                final = isinstance(payload, dict) and (payload.get("phase") == "final_answer" or payload.get("end_turn") is True)
                remember_assistant(i, _event_ts(obj), final=final)

        best: tuple[str, tuple[int, float | None]] | None = None
        if last_user is not None:
            best = ("user", last_user)
        if last_assistant is not None:
            if best is None or last_assistant[0] > best[1][0]:
                best = ("assistant", last_assistant)
        if last_nonfinal_assistant is not None:
            if best is None or last_nonfinal_assistant[0] > best[1][0]:
                best = ("assistant_nonfinal", last_nonfinal_assistant)
        if best is not None:
            role, (_i, ts) = best
            if role == "assistant_nonfinal" or ts is None:
                return None
            return (role, float(ts))
        scan *= 2
    return None
