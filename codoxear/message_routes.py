from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping
import time
import urllib.parse

from . import rollout_log as _rollout_log
from .message_cursor import MessageCursorError
from .message_cursor import attach_history_cursors as _attach_history_cursors_impl
from .transcript_search import clip_search_match_text as _clip_search_match_text
from .transcript_search import search_chat_log_bounded as _search_chat_log_bounded


JsonResponse = Callable[[Any, int, dict[str, Any]], None]
RouteMatcher = Callable[[str, str, str], str | None]

# Per-poll read bound for the live message stream. Restores the 2 MiB default
# that the pre-extraction server wrapper applied to this exact call path.
LIVE_POLL_READ_MAX_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class MessageRouteDeps:
    require_auth: Callable[[Any], bool]
    json_response: JsonResponse
    launch_attempt_transcript_for_session_id: Callable[..., dict[str, Any] | None]
    transcript_export_max_bytes: int
    transcript_search_max_line_bytes: int
    decode_message_cursor: Callable[..., int]
    encode_message_cursor: Callable[..., str]
    record_metric: Callable[[str, float], None]
    message_runtime_snapshot: Callable[..., tuple[dict[str, Any], bool, int, Any]]


def _parse_bounded_query_int(
    qs: Mapping[str, list[str]],
    name: str,
    *,
    default: int,
    min_value: int,
    max_value: int,
) -> tuple[int, str | None]:
    values = qs.get(name)
    if not values:
        return default, None
    try:
        value = int(values[0])
    except (TypeError, ValueError):
        return default, f"{name} must be an integer"
    return max(min_value, min(max_value, value)), None


def _message_transcript_identity(session: Any) -> dict[str, Any]:
    log_path = session.log_path
    if log_path is None or (not log_path.exists()):
        return {
            "transcript_state": "pending_bind",
            "thread_id": None,
            "log_path": None,
        }
    return {
        "transcript_state": "bound",
        "thread_id": session.thread_id,
        "log_path": str(log_path),
    }


def _read_chat_export_events(log_path: Path, *, max_bytes: int) -> list[dict[str, Any]]:
    size = int(log_path.stat().st_size)
    limit = max(1, int(max_bytes))
    if size > limit:
        raise ValueError(f"transcript log is too large to export ({size} bytes > {limit} bytes)")
    records, _next_after = _rollout_log._read_jsonl_records_from_offset(log_path, 0, max_bytes=max(size, 1))
    return _rollout_log._extract_positioned_chat_events(records)


def _search_launch_payload_events(payload: dict[str, Any], query: str, *, limit: int, text_max: int) -> tuple[int, list[dict[str, Any]]]:
    needle = query.strip().casefold()
    if not needle:
        return 0, []
    matches: list[dict[str, Any]] = []
    count = 0
    for event in payload.get("events") if isinstance(payload.get("events"), list) else []:
        if not isinstance(event, dict):
            continue
        text = event.get("text")
        if not isinstance(text, str) or needle not in text.casefold():
            continue
        count += 1
        if len(matches) >= max(0, int(limit)):
            continue
        item = dict(event)
        clipped = _clip_search_match_text([{"text": text}], text_max, query=query)[0]
        item["text"] = clipped.get("text", text)
        item["text_truncated"] = bool(clipped.get("text_truncated"))
        matches.append(item)
    return count, matches


def _launch_payload_for_missing_session(
    deps: MessageRouteDeps,
    session_id: str,
    *,
    max_bytes: int | None = None,
) -> dict[str, Any] | None:
    if max_bytes is None:
        return deps.launch_attempt_transcript_for_session_id(session_id)
    try:
        return deps.launch_attempt_transcript_for_session_id(session_id, max_bytes=max_bytes)
    except TypeError:
        return deps.launch_attempt_transcript_for_session_id(session_id)


def _launch_payload_cursor_session(payload: dict[str, Any]) -> Any:
    log_path_raw = payload.get("log_path")
    log_path = Path(log_path_raw) if isinstance(log_path_raw, str) and log_path_raw else None
    thread_id = payload.get("thread_id") if isinstance(payload.get("thread_id"), str) else None
    if not thread_id:
        thread_id = payload.get("session_id") if isinstance(payload.get("session_id"), str) else ""
    return SimpleNamespace(thread_id=thread_id, log_path=log_path)


def _launch_payload_events_with_cursors(payload: dict[str, Any], *, encode_cursor: Callable[..., str]) -> list[dict[str, Any]]:
    events = payload.get("events") if isinstance(payload.get("events"), list) else []
    return _attach_history_cursors_impl(list(events), session=_launch_payload_cursor_session(payload), encode_cursor=encode_cursor)


def _attach_search_load_cursors(matches: list[dict[str, Any]], *, session: Any, encode_cursor: Callable[..., str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        item = dict(match)
        before_byte = item.get("_before_byte")
        if isinstance(before_byte, bool):
            before_byte = None
        if isinstance(before_byte, int) and before_byte >= 0:
            item["history_cursor"] = encode_cursor(kind="history", session=session, pos=before_byte)
        after_byte = item.pop("_after_byte", None)
        if isinstance(after_byte, bool):
            after_byte = None
        if isinstance(after_byte, int) and after_byte > 0:
            item["load_cursor"] = encode_cursor(kind="history", session=session, pos=after_byte)
        out.append(item)
    return out


def handle_messages_get_route(
    handler: Any,
    *,
    path: str,
    query: str,
    manager: Any,
    deps: MessageRouteDeps,
    match_session_route: RouteMatcher,
) -> bool:
    for route_name, route_handler in MESSAGE_GET_ROUTES:
        session_id = match_session_route(path, "messages", route_name)
        if session_id is None:
            continue
        route_handler(handler, session_id=session_id, query=query, manager=manager, deps=deps)
        return True
    return False


def handle_messages_export(handler: Any, *, session_id: str, query: str = "", manager: Any, deps: MessageRouteDeps) -> None:
    if not deps.require_auth(handler):
        handler._unauthorized()
        return
    manager.refresh_session_meta(session_id)
    s = manager.get_session(session_id)
    if not s:
        launch_payload = _launch_payload_for_missing_session(
            deps,
            session_id,
            max_bytes=deps.transcript_export_max_bytes,
        )
        if launch_payload is not None:
            log_path_raw = launch_payload.get("log_path")
            if isinstance(log_path_raw, str) and log_path_raw:
                log_path = Path(log_path_raw)
                if log_path.exists():
                    size = int(log_path.stat().st_size)
                    limit = max(1, int(deps.transcript_export_max_bytes))
                    if size > limit:
                        deps.json_response(
                            handler,
                            413,
                            {
                                "error": f"transcript log is too large to export ({size} bytes > {limit} bytes)",
                                "max_bytes": limit,
                            },
                        )
                        return
            events = launch_payload.get("events") if isinstance(launch_payload.get("events"), list) else []
            deps.json_response(handler, 200, {**launch_payload, "events": events, "event_count": len(events)})
            return
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    transcript = _message_transcript_identity(s)
    if s.log_path is None or (not s.log_path.exists()):
        deps.json_response(handler, 200, {**transcript, "events": [], "event_count": 0})
        return
    try:
        events = _read_chat_export_events(s.log_path, max_bytes=deps.transcript_export_max_bytes)
    except ValueError as e:
        deps.json_response(handler, 413, {"error": str(e), "max_bytes": int(deps.transcript_export_max_bytes)})
        return
    events = manager._attach_notification_texts(events)
    deps.json_response(handler, 200, {**transcript, "events": events, "event_count": len(events)})


def handle_messages_search(handler: Any, *, session_id: str, query: str, manager: Any, deps: MessageRouteDeps) -> None:
    if not deps.require_auth(handler):
        handler._unauthorized()
        return
    manager.refresh_session_meta(session_id)
    s = manager.get_session(session_id)
    qs = urllib.parse.parse_qs(query)
    search_query = (qs.get("q") or [""])[0]
    match_limit, limit_error = _parse_bounded_query_int(qs, "limit", default=20, min_value=0, max_value=100)
    if limit_error is not None:
        deps.json_response(handler, 400, {"error": limit_error})
        return
    text_max, text_max_error = _parse_bounded_query_int(qs, "text_max", default=0, min_value=0, max_value=4096)
    if text_max_error is not None:
        deps.json_response(handler, 400, {"error": text_max_error})
        return
    count_max, count_max_error = _parse_bounded_query_int(qs, "count_max", default=0, min_value=0, max_value=100000)
    if count_max_error is not None:
        deps.json_response(handler, 400, {"error": count_max_error})
        return
    order = (qs.get("order") or ["first"])[0]
    if order not in {"first", "latest"}:
        deps.json_response(handler, 400, {"error": "order must be first or latest"})
        return
    if count_max > 0 and order == "latest":
        deps.json_response(handler, 400, {"error": "count_max is only supported with order=first"})
        return
    before_byte: int | None = None
    before_q = qs.get("before")

    if not s:
        launch_payload = _launch_payload_for_missing_session(deps, session_id)
        if launch_payload is not None:
            if not isinstance(search_query, str) or not search_query.strip():
                deps.json_response(handler, 200, {**launch_payload, "query": "", "match_count": 0, "match_count_truncated": False, "matches": []})
                return
            payload_with_cursors = {
                **launch_payload,
                "events": _launch_payload_events_with_cursors(launch_payload, encode_cursor=deps.encode_message_cursor),
            }
            match_count, matches = _search_launch_payload_events(
                payload_with_cursors,
                search_query,
                limit=match_limit,
                text_max=text_max,
            )
            if order == "latest":
                matches = list(reversed(matches))
            deps.json_response(
                handler,
                200,
                {
                    **launch_payload,
                    "query": search_query.strip(),
                    "match_count": match_count,
                    "match_count_truncated": False,
                    "matches": matches,
                },
            )
            return
        deps.json_response(handler, 404, {"error": "unknown session"})
        return

    if before_q is not None and before_q and before_q[0].strip():
        try:
            before_byte = deps.decode_message_cursor(before_q[0], kind="history", session=s)
        except MessageCursorError as e:
            deps.json_response(handler, 409, {"error": str(e)})
            return
    transcript = _message_transcript_identity(s)
    if not isinstance(search_query, str) or not search_query.strip():
        deps.json_response(handler, 200, {**transcript, "query": "", "match_count": 0, "match_count_truncated": False, "matches": []})
        return
    if s.log_path is None or (not s.log_path.exists()):
        deps.json_response(handler, 200, {**transcript, "query": search_query.strip(), "match_count": 0, "match_count_truncated": False, "matches": []})
        return
    match_count, matches, match_count_truncated = _search_chat_log_bounded(
        s.log_path,
        search_query,
        limit=match_limit,
        max_line_bytes=deps.transcript_search_max_line_bytes,
        before_byte=before_byte,
        order=order,
        count_limit=count_max if count_max > 0 else None,
    )
    matches = _attach_search_load_cursors(matches, session=s, encode_cursor=deps.encode_message_cursor)
    matches = manager._attach_notification_texts(matches)
    matches = _clip_search_match_text(matches, text_max, query=search_query)
    deps.json_response(handler, 200, {**transcript, "query": search_query.strip(), "match_count": match_count, "match_count_truncated": bool(match_count_truncated), "matches": matches})


def handle_messages_tail(handler: Any, *, session_id: str, query: str, manager: Any, deps: MessageRouteDeps) -> None:
    if not deps.require_auth(handler):
        handler._unauthorized()
        return
    t0_total = time.perf_counter()
    manager.refresh_session_meta(session_id)
    s = manager.get_session(session_id)
    if not s:
        launch_payload = _launch_payload_for_missing_session(deps, session_id)
        if launch_payload is not None:
            events = _launch_payload_events_with_cursors(launch_payload, encode_cursor=deps.encode_message_cursor)
            deps.json_response(handler, 200, {**launch_payload, "events": events})
            deps.record_metric("api_messages_init_ms", (time.perf_counter() - t0_total) * 1000.0)
            return
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    qs = urllib.parse.parse_qs(query)
    limit, limit_error = _parse_bounded_query_int(qs, "limit", default=80, min_value=20, max_value=200)
    if limit_error is not None:
        deps.json_response(handler, 400, {"error": limit_error})
        return
    if s.log_path is None or (not s.log_path.exists()):
        _state, busy_val, queue_val, token_val = deps.message_runtime_snapshot(session_id, s)
        transcript = _message_transcript_identity(s)
        deps.json_response(
            handler,
            200,
            {
                **transcript,
                "live_cursor": None,
                "history_cursor": None,
                "events": [],
                "has_older": False,
                "busy": bool(busy_val),
                "queue_len": int(queue_val),
                "token": token_val,
            },
        )
        deps.record_metric("api_messages_init_ms", (time.perf_counter() - t0_total) * 1000.0)
        return
    events, before_byte, after_byte, has_older = _rollout_log._read_chat_tail_page(s.log_path, limit=limit)
    events = manager._attach_notification_texts(events)
    events = _attach_history_cursors_impl(events, session=s, encode_cursor=deps.encode_message_cursor)
    live_cursor = deps.encode_message_cursor(kind="live", session=s, pos=after_byte)
    history_cursor = deps.encode_message_cursor(kind="history", session=s, pos=before_byte) if has_older and before_byte > 0 else None
    _state, busy_val, queue_val, token_val = deps.message_runtime_snapshot(session_id, s)
    transcript = _message_transcript_identity(s)
    deps.json_response(
        handler,
        200,
        {
            **transcript,
            "live_cursor": live_cursor,
            "history_cursor": history_cursor,
            "events": events,
            "has_older": bool(has_older),
            "busy": bool(busy_val),
            "queue_len": int(queue_val),
            "token": token_val,
        },
    )
    deps.record_metric("api_messages_init_ms", (time.perf_counter() - t0_total) * 1000.0)


def handle_messages_history(handler: Any, *, session_id: str, query: str, manager: Any, deps: MessageRouteDeps) -> None:
    if not deps.require_auth(handler):
        handler._unauthorized()
        return
    manager.refresh_session_meta(session_id)
    s = manager.get_session(session_id)
    if not s:
        launch_payload = _launch_payload_for_missing_session(deps, session_id)
        if launch_payload is not None:
            events = _launch_payload_events_with_cursors(launch_payload, encode_cursor=deps.encode_message_cursor)
            deps.json_response(
                handler,
                200,
                {
                    **launch_payload,
                    "history_cursor": None,
                    "events": events,
                    "has_older": False,
                    "busy": False,
                    "queue_len": 0,
                    "token": None,
                },
            )
            return
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    qs = urllib.parse.parse_qs(query)
    cursor_q = qs.get("cursor")
    if cursor_q is None or not cursor_q or not cursor_q[0].strip():
        deps.json_response(handler, 400, {"error": "cursor required"})
        return
    limit, limit_error = _parse_bounded_query_int(qs, "limit", default=60, min_value=20, max_value=200)
    if limit_error is not None:
        deps.json_response(handler, 400, {"error": limit_error})
        return
    if s.log_path is None or (not s.log_path.exists()):
        _state, busy_val, queue_val, token_val = deps.message_runtime_snapshot(session_id, s)
        transcript = _message_transcript_identity(s)
        deps.json_response(
            handler,
            200,
            {
                **transcript,
                "history_cursor": None,
                "events": [],
                "has_older": False,
                "busy": bool(busy_val),
                "queue_len": int(queue_val),
                "token": token_val,
            },
        )
        return
    try:
        before_byte = deps.decode_message_cursor(cursor_q[0], kind="history", session=s)
    except MessageCursorError as e:
        deps.json_response(handler, 409, {"error": str(e)})
        return
    events, next_before, has_older = _rollout_log._read_chat_history_page(s.log_path, before_byte=before_byte, limit=limit)
    events = manager._attach_notification_texts(events)
    events = _attach_history_cursors_impl(events, session=s, encode_cursor=deps.encode_message_cursor)
    history_cursor = deps.encode_message_cursor(kind="history", session=s, pos=next_before) if has_older and next_before > 0 else None
    _state, busy_val, queue_val, token_val = deps.message_runtime_snapshot(session_id, s)
    transcript = _message_transcript_identity(s)
    deps.json_response(
        handler,
        200,
        {
            **transcript,
            "history_cursor": history_cursor,
            "events": events,
            "has_older": bool(has_older),
            "busy": bool(busy_val),
            "queue_len": int(queue_val),
            "token": token_val,
        },
    )


def handle_messages_live(handler: Any, *, session_id: str, query: str, manager: Any, deps: MessageRouteDeps) -> None:
    if not deps.require_auth(handler):
        handler._unauthorized()
        return
    t0_total = time.perf_counter()
    t0_meta = time.perf_counter()
    manager.refresh_session_meta(session_id)
    dt_meta_ms = (time.perf_counter() - t0_meta) * 1000.0
    s = manager.get_session(session_id)
    if not s:
        launch_payload = _launch_payload_for_missing_session(deps, session_id)
        if launch_payload is not None:
            events = _launch_payload_events_with_cursors(launch_payload, encode_cursor=deps.encode_message_cursor)
            deps.json_response(
                handler,
                200,
                {
                    **launch_payload,
                    "live_cursor": None,
                    "events": events,
                    "meta_delta": {"thinking": 0, "tool": 0, "system": 0},
                    "turn_start": False,
                    "turn_end": True,
                    "turn_aborted": False,
                    "diag": {"post_log_recovery": True, "meta_refresh_ms": round(dt_meta_ms, 3)},
                    "busy": False,
                    "queue_len": 0,
                    "token": None,
                },
            )
            deps.record_metric("api_messages_poll_ms", (time.perf_counter() - t0_total) * 1000.0)
            return
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    qs = urllib.parse.parse_qs(query)
    cursor_q = qs.get("cursor")
    if cursor_q is None or not cursor_q or not cursor_q[0].strip():
        deps.json_response(handler, 400, {"error": "cursor required"})
        return
    if s.log_path is None or (not s.log_path.exists()):
        _state, busy_val, queue_val, token_val = deps.message_runtime_snapshot(session_id, s)
        transcript = _message_transcript_identity(s)
        deps.json_response(
            handler,
            200,
            {
                **transcript,
                "live_cursor": None,
                "events": [],
                "meta_delta": {"thinking": 0, "tool": 0, "system": 0},
                "turn_start": False,
                "turn_end": False,
                "turn_aborted": False,
                "diag": {"pending_log": True, "meta_refresh_ms": round(dt_meta_ms, 3)},
                "busy": bool(busy_val),
                "queue_len": int(queue_val),
                "token": token_val,
            },
        )
        deps.record_metric("api_messages_poll_ms", (time.perf_counter() - t0_total) * 1000.0)
        return
    try:
        after_byte = deps.decode_message_cursor(cursor_q[0], kind="live", session=s)
    except MessageCursorError as e:
        deps.json_response(handler, 409, {"error": str(e)})
        return
    records, next_after = _rollout_log._read_jsonl_records_from_offset(s.log_path, after_byte, max_bytes=LIVE_POLL_READ_MAX_BYTES)
    objs = [record.obj for record in records]
    initial_cc_pending = _rollout_log._cc_pending_tool_ids_before(s.log_path, after_byte) if records and after_byte > 0 else set()
    events, meta_delta, flags, diag = _rollout_log._extract_chat_events(objs, initial_cc_pending_tool_ids=initial_cc_pending)
    token_update = _rollout_log._extract_token_update(objs)
    prior_user_byte, prior_turn_has_assistant = (
        _rollout_log._prior_open_turn_context(s.log_path, after_byte) if after_byte > 0 else (None, False)
    )
    events = _rollout_log._extract_positioned_chat_events(
        records,
        initial_cc_pending_tool_ids=initial_cc_pending,
        prior_user_byte=prior_user_byte,
        prior_turn_has_assistant=prior_turn_has_assistant,
    )
    if objs:
        manager.mark_log_delta(session_id, objs=objs, new_off=next_after)
    s2 = manager.get_session(session_id)
    if token_update is not None and s2 is not None:
        s2.token = token_update
    events = manager._attach_notification_texts(events)
    events = _attach_history_cursors_impl(events, session=s, encode_cursor=deps.encode_message_cursor)
    live_cursor = deps.encode_message_cursor(kind="live", session=s, pos=next_after)
    t0_state = time.perf_counter()
    _state, busy_val, queue_val, token_val = deps.message_runtime_snapshot(session_id, s, token_update=token_update)
    diag["state_ms"] = round((time.perf_counter() - t0_state) * 1000.0, 3)
    diag["meta_refresh_ms"] = round(dt_meta_ms, 3)
    transcript = _message_transcript_identity(s)
    deps.json_response(
        handler,
        200,
        {
            **transcript,
            "live_cursor": live_cursor,
            "events": events,
            "meta_delta": meta_delta,
            "turn_start": bool(flags.get("turn_start")),
            "turn_end": bool(flags.get("turn_end")),
            "turn_aborted": bool(flags.get("turn_aborted")),
            "diag": diag,
            "busy": bool(busy_val),
            "queue_len": int(queue_val),
            "token": token_val,
        },
    )
    deps.record_metric("api_messages_poll_ms", (time.perf_counter() - t0_total) * 1000.0)


MESSAGE_GET_ROUTES = (
    ("export", handle_messages_export),
    ("search", handle_messages_search),
    ("tail", handle_messages_tail),
    ("history", handle_messages_history),
    ("live", handle_messages_live),
)
