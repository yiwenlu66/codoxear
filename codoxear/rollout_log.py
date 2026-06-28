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
from .rollout_idle import _analyze_log_chunk
from .rollout_idle import _compute_cc_idle_from_current_turn
from .rollout_idle import _compute_idle_from_log
from .rollout_idle import _has_assistant_output_text
from .rollout_idle import _last_chat_role_ts_from_tail
from .rollout_idle import _last_conversation_ts_from_tail
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
