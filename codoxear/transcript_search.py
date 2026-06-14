from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

from . import rollout_log as _rollout_log


TRANSCRIPT_SEARCH_MAX_LINE_BYTES = int(os.environ.get("CODEX_WEB_TRANSCRIPT_SEARCH_MAX_LINE_BYTES", str(4 * 1024 * 1024)))


def chat_event_matches_query(event: dict[str, Any], needle: str) -> bool:
    role = event.get("role")
    if role not in {"user", "assistant"}:
        return False
    text = event.get("text")
    return isinstance(text, str) and needle in text.casefold()


def search_chat_events(events: list[dict[str, Any]], query: str, *, limit: int = 20) -> tuple[int, list[dict[str, Any]]]:
    needle = query.strip().casefold()
    if not needle:
        return 0, []
    max_matches = max(0, int(limit))
    count = 0
    matches: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if not chat_event_matches_query(event, needle):
            continue
        count += 1
        if len(matches) < max_matches:
            matches.append(event)
    return count, matches


def casefold_match_span(text: str, query: str) -> tuple[int, int] | None:
    needle = query.strip().casefold()
    if not needle:
        return None
    folded_parts: list[str] = []
    folded_to_original: list[int] = []
    for idx, ch in enumerate(text):
        folded = ch.casefold()
        if not folded:
            continue
        folded_parts.append(folded)
        folded_to_original.extend([idx] * len(folded))
    folded_text = "".join(folded_parts)
    folded_idx = folded_text.find(needle)
    if folded_idx < 0 or not folded_to_original:
        return None
    folded_end = min(len(folded_to_original) - 1, folded_idx + len(needle) - 1)
    return folded_to_original[folded_idx], folded_to_original[folded_end] + 1


def clip_search_text_around_query(text: str, query: str, limit: int) -> tuple[str, bool]:
    max_chars = max(0, int(limit))
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    span = casefold_match_span(text, query)
    if span is not None:
        idx, match_end = span
        match_len = max(1, match_end - idx)
        context = max(0, (max_chars - min(match_len, max_chars)) // 2)
        start = max(0, idx - context)
    else:
        start = 0
    end = min(len(text), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)
    return text[start:end], True


def clip_search_match_text(matches: list[dict[str, Any]], text_max: int, *, query: str = "") -> list[dict[str, Any]]:
    limit = max(0, int(text_max))
    if limit <= 0:
        return matches
    clipped: list[dict[str, Any]] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        item = dict(match)
        text = item.get("text")
        if isinstance(text, str):
            clipped_text, was_clipped = clip_search_text_around_query(text, query, limit)
            if was_clipped:
                item["text"] = clipped_text
                item["text_truncated"] = True
        clipped.append(item)
    return clipped


def iter_jsonl_records_forward_bounded(
    log_path: Path,
    *,
    max_line_bytes: int = TRANSCRIPT_SEARCH_MAX_LINE_BYTES,
) -> Iterator[_rollout_log.JsonlRecord]:
    limit = max(1, int(max_line_bytes))
    with log_path.open("rb") as f:
        offset = 0
        while True:
            start = offset
            raw = f.readline(limit + 1)
            if not raw:
                break
            offset += len(raw)
            if len(raw) > limit and not raw.endswith(b"\n"):
                while True:
                    skipped = f.readline(64 * 1024)
                    if not skipped:
                        break
                    offset += len(skipped)
                    if skipped.endswith(b"\n"):
                        break
                continue
            if not raw.endswith(b"\n"):
                break
            line = raw.rstrip(b"\r\n")
            try:
                obj = _rollout_log._parse_jsonl_line(line)
            except Exception:
                continue
            if obj is None:
                continue
            yield _rollout_log.JsonlRecord(start=start, end=offset, obj=obj)


def iter_positioned_chat_events_forward(
    log_path: Path,
    *,
    max_line_bytes: int = TRANSCRIPT_SEARCH_MAX_LINE_BYTES,
) -> Iterator[dict[str, Any]]:
    cc_pending_tool_ids: set[str] = set()
    last_assistant_key: tuple[str, str] | None = None
    for record in iter_jsonl_records_forward_bounded(log_path, max_line_bytes=max_line_bytes):
        try:
            event = _rollout_log._single_chat_event(record.obj, cc_pending_tool_ids=cc_pending_tool_ids)
        except Exception:
            continue
        if event is None:
            continue
        event = _rollout_log._with_chat_position(event, before_byte=record.start)
        event["_after_byte"] = int(record.end)
        role = event.get("role")
        if role == "user":
            last_assistant_key = None
            yield event
            continue
        if role == "assistant":
            key = _rollout_log._chat_assistant_dedupe_key(event)
            if key is not None and key == last_assistant_key:
                continue
            last_assistant_key = key
            yield event
            continue
        yield event


def search_chat_log(
    log_path: Path,
    query: str,
    *,
    limit: int = 20,
    max_line_bytes: int = TRANSCRIPT_SEARCH_MAX_LINE_BYTES,
    before_byte: int | None = None,
    order: str = "first",
) -> tuple[int, list[dict[str, Any]]]:
    needle = query.strip().casefold()
    if not needle:
        return 0, []
    max_matches = max(0, int(limit))
    stop_before = None if before_byte is None else max(0, int(before_byte))
    keep_latest = order == "latest"
    count = 0
    matches: list[dict[str, Any]] = []
    for event in iter_positioned_chat_events_forward(log_path, max_line_bytes=max_line_bytes):
        event_before = event.get("_before_byte")
        if stop_before is not None and isinstance(event_before, int) and event_before >= stop_before:
            break
        if not chat_event_matches_query(event, needle):
            continue
        count += 1
        if max_matches <= 0:
            continue
        if keep_latest:
            matches.append(event)
            if len(matches) > max_matches:
                matches.pop(0)
        elif len(matches) < max_matches:
            matches.append(event)
    return count, matches
