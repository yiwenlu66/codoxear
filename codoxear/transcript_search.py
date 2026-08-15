from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Iterator

from .agent_backend import get_agent_backend
from .agent_backend import infer_agent_backend_from_log_path
from . import rollout_log as _rollout_log
from .rollout_chat_events import _build_no_response_event


TRANSCRIPT_SEARCH_MAX_LINE_BYTES = int(os.environ.get("CODEX_WEB_TRANSCRIPT_SEARCH_MAX_LINE_BYTES", str(4 * 1024 * 1024)))


def session_log_paths_for_search(log_path: Path, *, agent_backend: str | None = None, session_id: str | None = None) -> list[Path]:
    """Return every backend log belonging to the active session.

    A broker sidecar stores only the currently bound file. Resume and rotation
    can leave earlier files in the backend's session tree, so searching that
    one path silently loses history. The backend-specific log identity check
    prevents unrelated sessions in the same tree from entering the projection.
    """
    current = Path(log_path)
    backend = get_agent_backend(agent_backend or infer_agent_backend_from_log_path(current) or "codex")
    sid = session_id or backend.session_id_from_log_path(current)
    if not sid:
        return [current]
    directory_names = {"codex": "sessions", "pi": "sessions", "cc": "projects"}
    root: Path | None = None
    for parent in (current.parent, *current.parents):
        if parent.name == directory_names[backend.name]:
            root = parent
            break
    if root is None or not root.exists():
        return [current]
    paths: list[Path] = []
    try:
        candidates = root.rglob("*.jsonl")
    except OSError:
        candidates = ()
    for candidate in candidates:
        if not backend.is_session_log_path(candidate, sessions_dir=root):
            continue
        try:
            if backend.log_matches_session_id(candidate, sid):
                paths.append(candidate)
        except (FileNotFoundError, OSError, ValueError):
            continue
    if not any(_same_path(candidate, current) for candidate in paths):
        paths.append(current)
    def sort_key(path: Path) -> tuple[float, int, str]:
        try:
            mtime = float(path.stat().st_mtime)
        except OSError:
            mtime = 0.0
        return (mtime, 1 if _same_path(path, current) else 0, str(path))
    paths.sort(key=sort_key)
    return paths


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a.absolute() == b.absolute()


def search_chat_logs_bounded(
    log_paths: list[Path],
    query: str,
    *,
    limit: int = 20,
    max_line_bytes: int = TRANSCRIPT_SEARCH_MAX_LINE_BYTES,
    before_byte: int | None = None,
    after_byte: int = 0,
    order: str = "first",
    count_limit: int | None = None,
    role: str | None = None,
    match_all: bool = False,
) -> tuple[int, list[dict[str, Any]], bool]:
    """Search an ordered set of session logs as one transcript projection."""
    if not log_paths:
        return 0, [], False
    max_matches = max(0, int(limit))
    max_count = None if count_limit is None else max(0, int(count_limit))
    if order == "latest" and max_count is not None:
        raise ValueError("count_limit is only supported with order=first")
    total = 0
    truncated = False
    matches: list[dict[str, Any]] = []
    for index, path in enumerate(log_paths):
        remaining = None if max_count is None else max(0, max_count - total)
        if remaining == 0:
            # A bounded count is only marked truncated when another match is
            # observed; avoid claiming truncation merely because the boundary
            # landed exactly at EOF.
            break
        per_limit = max_matches if order == "latest" else max(0, max_matches - len(matches))
        count, found, file_truncated = search_chat_log_bounded(
            path,
            query,
            limit=max_matches if order == "latest" else per_limit,
            max_line_bytes=max_line_bytes,
            before_byte=before_byte if index == len(log_paths) - 1 else None,
            after_byte=after_byte if index == len(log_paths) - 1 else 0,
            order=order,
            count_limit=remaining,
            role=role,
            match_all=match_all,
        )
        total += count
        for event in found:
            tagged = dict(event)
            tagged["_log_path"] = str(path)
            matches.append(tagged)
        if order == "latest" and len(matches) > max_matches:
            del matches[:-max_matches]
        if file_truncated:
            truncated = True
            break
    if order == "latest":
        # Files are oldest-first; each file's matches are oldest-first too.
        matches = matches[-max_matches:] if max_matches else []
    return total, matches, truncated


def chat_event_matches_query(
    event: dict[str, Any],
    needle: str,
    *,
    role: str | None = None,
    match_all: bool = False,
) -> bool:
    event_role = event.get("role")
    if event_role not in {"user", "assistant"}:
        return False
    if role is not None and event_role != role:
        return False
    if match_all:
        return True
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
    on_oversized_skip: Callable[[int, int], None] | None = None,
    before_byte: int | None = None,
    start_byte: int = 0,
) -> Iterator[_rollout_log.JsonlRecord]:
    limit = max(1, int(max_line_bytes))
    stop_before = None if before_byte is None else max(0, int(before_byte))
    start_at = max(0, int(start_byte))
    with log_path.open("rb") as f:
        f.seek(start_at)
        offset = start_at
        while True:
            start = offset
            if stop_before is not None and start >= stop_before:
                break
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
                if on_oversized_skip is not None:
                    on_oversized_skip(start, offset)
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


def _search_record_user_byte(obj: dict[str, Any], record_start: int) -> int | None:
    typ = obj.get("type")
    if typ == "event_msg":
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            return None
        if payload.get("type") == "user_message" and isinstance(payload.get("message"), str) and payload["message"].strip():
            return int(record_start)
        return None
    if typ == "user":
        user_text = _rollout_log.cc_user_text(obj)
        if isinstance(user_text, str) and user_text:
            return int(record_start)
    return None


def _search_record_closes_turn(obj: dict[str, Any]) -> bool:
    typ = obj.get("type")
    if typ == "event_msg":
        payload = obj.get("payload")
        return isinstance(payload, dict) and payload.get("type") in {"task_complete", "turn_complete"}
    if typ == "system":
        return _rollout_log.cc_is_turn_end(obj) or _rollout_log.cc_system_api_error_is_terminal(obj)
    return False


def _position_search_event(event: dict[str, Any], *, record_start: int, record_end: int) -> dict[str, Any]:
    positioned = _rollout_log._with_chat_position(event, before_byte=record_start)
    positioned["_after_byte"] = int(record_end)
    return positioned


def iter_positioned_chat_events_forward(
    log_path: Path,
    *,
    max_line_bytes: int = TRANSCRIPT_SEARCH_MAX_LINE_BYTES,
    on_oversized_skip: Callable[[int, int], None] | None = None,
    before_byte: int | None = None,
    start_byte: int = 0,
) -> Iterator[dict[str, Any]]:
    # Search must project the same visible rows as tail/history/live without
    # first building whole-log record and event lists. The two batch transforms
    # used historically are stream-local: adjacent assistant dedupe is one
    # previous assistant key (reset on user), and no-response injection is the
    # currently open user turn plus whether a deduped assistant row has been
    # emitted since that user. Close records are processed after their own
    # visible event is emitted so terminal error rows suppress generic
    # no-response injection exactly as the batch injector did.
    stop_before = None if before_byte is None else max(0, int(before_byte))
    start_at = max(0, int(start_byte))
    cc_pending_tool_ids: set[str] = _rollout_log._cc_pending_tool_ids_before(log_path, start_at) if start_at > 0 else set()
    last_assistant_key: tuple[str, str] | None = None
    if start_at > 0:
        open_user_byte, open_turn_has_assistant = _rollout_log._prior_open_turn_context(log_path, start_at)
    else:
        open_user_byte, open_turn_has_assistant = None, False

    for record in iter_jsonl_records_forward_bounded(
        log_path,
        max_line_bytes=max_line_bytes,
        on_oversized_skip=on_oversized_skip,
        before_byte=stop_before,
        start_byte=start_at,
    ):
        if stop_before is not None and record.start >= stop_before:
            break
        emitted_assistant = False
        try:
            raw_event = _rollout_log._single_chat_event(record.obj, cc_pending_tool_ids=cc_pending_tool_ids)
        except Exception:
            raw_event = None
        if raw_event is not None:
            role = raw_event.get("role")
            if role == "user":
                last_assistant_key = None
                yield _position_search_event(raw_event, record_start=record.start, record_end=record.end)
            elif role == "assistant":
                key = _rollout_log._chat_assistant_dedupe_key(raw_event)
                if key is None or key != last_assistant_key:
                    last_assistant_key = key
                    emitted_assistant = True
                    yield _position_search_event(raw_event, record_start=record.start, record_end=record.end)
            else:
                yield _position_search_event(raw_event, record_start=record.start, record_end=record.end)

        user_byte = _search_record_user_byte(record.obj, record.start)
        if user_byte is not None:
            open_user_byte = user_byte
            open_turn_has_assistant = False
        elif emitted_assistant and open_user_byte is not None:
            open_turn_has_assistant = True

        if _search_record_closes_turn(record.obj):
            if open_user_byte is not None and not open_turn_has_assistant:
                no_response = _position_search_event(
                    _build_no_response_event(record.obj),
                    record_start=record.start,
                    record_end=record.end,
                )
                yield no_response
            open_user_byte = None
            open_turn_has_assistant = False


def search_chat_log_bounded(
    log_path: Path,
    query: str,
    *,
    limit: int = 20,
    max_line_bytes: int = TRANSCRIPT_SEARCH_MAX_LINE_BYTES,
    before_byte: int | None = None,
    after_byte: int = 0,
    order: str = "first",
    count_limit: int | None = None,
    role: str | None = None,
    match_all: bool = False,
) -> tuple[int, list[dict[str, Any]], bool]:
    needle = query.strip().casefold()
    if not needle and not match_all:
        return 0, [], False
    max_matches = max(0, int(limit))
    stop_before = None if before_byte is None else max(0, int(before_byte))
    keep_latest = order == "latest"
    max_count = None if count_limit is None else max(0, int(count_limit))
    if keep_latest and max_count is not None:
        raise ValueError("count_limit is only supported with order=first")
    count = 0
    truncated = False
    skipped_oversized = False
    matches: list[dict[str, Any]] = []

    def mark_oversized_skip(start: int, _end: int) -> None:
        nonlocal skipped_oversized
        if stop_before is None or start < stop_before:
            skipped_oversized = True

    for event in iter_positioned_chat_events_forward(
        log_path,
        max_line_bytes=max_line_bytes,
        on_oversized_skip=mark_oversized_skip,
        before_byte=stop_before,
        start_byte=max(0, int(after_byte)),
    ):
        event_before = event.get("_before_byte")
        if stop_before is not None and isinstance(event_before, int) and event_before >= stop_before:
            break
        if not chat_event_matches_query(event, needle, role=role, match_all=match_all):
            continue
        if max_count is not None and count >= max_count:
            truncated = True
            break
        count += 1
        if max_matches <= 0:
            continue
        if keep_latest:
            matches.append(event)
            if len(matches) > max_matches:
                matches.pop(0)
        elif len(matches) < max_matches:
            matches.append(event)
    return count, matches, truncated or skipped_oversized


def read_chat_window_around(
    log_path: Path,
    *,
    position_byte: int,
    before_limit: int = 30,
    after_limit: int = 30,
    max_line_bytes: int = TRANSCRIPT_SEARCH_MAX_LINE_BYTES,
) -> tuple[list[dict[str, Any]], int, bool, bool]:
    """Return a bounded normalized transcript window straddling a record cursor."""
    size = int(log_path.stat().st_size)
    position = max(0, min(int(position_byte), size))
    left, next_before, has_older = _rollout_log._read_chat_history_page(
        log_path,
        before_byte=position,
        limit=max(0, int(before_limit)),
    )
    right_limit = max(0, int(after_limit)) + 1  # target plus following rows
    right: list[dict[str, Any]] = []
    has_newer = False
    for event in iter_positioned_chat_events_forward(
        log_path,
        max_line_bytes=max_line_bytes,
        start_byte=position,
    ):
        if len(right) >= right_limit:
            has_newer = True
            break
        right.append(event)
    return left + right, next_before, bool(has_older), has_newer


def search_chat_log(
    log_path: Path,
    query: str,
    *,
    limit: int = 20,
    max_line_bytes: int = TRANSCRIPT_SEARCH_MAX_LINE_BYTES,
    before_byte: int | None = None,
    order: str = "first",
) -> tuple[int, list[dict[str, Any]]]:
    count, matches, _truncated = search_chat_log_bounded(
        log_path,
        query,
        limit=limit,
        max_line_bytes=max_line_bytes,
        before_byte=before_byte,
        order=order,
        count_limit=None,
    )
    return count, matches
