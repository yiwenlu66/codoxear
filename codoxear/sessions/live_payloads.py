from __future__ import annotations

from typing import Any

_SERVER = None


def bind_server_runtime(runtime: Any) -> None:
    global _SERVER
    _SERVER = runtime



def _sv() -> Any:
    if _SERVER is None:
        raise RuntimeError("server runtime not bound")
    return _SERVER



def session_live_payload(
    manager: Any,
    session_id: str,
    *,
    offset: int = 0,
    live_offset: int = 0,
    bridge_offset: int = 0,
    requests_version: str | None = None,
) -> dict[str, Any]:
    sv = _sv()
    manager.refresh_session_meta(session_id, strict=False)
    s = manager.get_session(session_id)
    if not s:
        historical_row = sv._historical_session_row(session_id)
        if historical_row is None:
            historical_row = sv._listed_session_row(manager, session_id)
        if historical_row is None:
            raise KeyError("unknown session")
        try:
            page = manager.get_messages_page(
                session_id,
                offset=max(0, int(offset)),
                init=(offset <= 0),
                limit=sv.SESSION_HISTORY_PAGE_SIZE,
                before=0,
            )
        except KeyError:
            page = {
                "events": [],
                "offset": max(0, int(offset)),
                "has_older": False,
                "next_before": 0,
            }
        durable_session_id = str(historical_row.get("session_id") or session_id)
        bridge_session_key = sv._clean_optional_text(historical_row.get("resume_session_id")) or durable_session_id
        bridge_events, next_bridge_offset = manager._bridge_events_since(
            bridge_session_key,
            offset=bridge_offset,
        )
        base_events = page.get("events") if isinstance(page.get("events"), list) else []
        merged_events = [*base_events, *bridge_events]
        return {
            "ok": True,
            "session_id": durable_session_id,
            "runtime_id": None,
            "offset": int(page.get("offset", max(0, int(offset))) or 0),
            "live_offset": 0,
            "bridge_offset": next_bridge_offset,
            "has_older": bool(page.get("has_older")),
            "next_before": int(page.get("next_before", 0) or 0),
            "busy": False,
            "events": merged_events,
            "requests_version": "",
            "requests": [],
            "token": None,
            "context_usage": None,
            "turn_timing": None,
        }
    page = manager.get_messages_page(
        session_id,
        offset=max(0, int(offset)),
        init=(offset <= 0),
        limit=sv.SESSION_HISTORY_PAGE_SIZE,
        before=0,
    )
    state = manager.get_state(session_id)
    busy, _broker_busy = sv._display_session_busy(manager, session_id, s, state)
    state_token = state.get("token") if isinstance(state, dict) else None
    token_val = sv._resolved_session_token(
        s,
        state_token if isinstance(state_token, dict) else None,
    )
    requests: list[dict[str, Any]] = []
    if s.backend == "pi":
        requests_payload = manager.get_ui_state(session_id)
        live_requests = requests_payload.get("requests")
        if isinstance(live_requests, list):
            requests = [item for item in live_requests if isinstance(item, dict)]
    current_requests_version = sv._ui_requests_version(requests)
    events = page.get("events")
    merged_events = events if isinstance(events, list) else []
    next_live_offset = max(0, int(live_offset))
    next_bridge_offset = max(0, int(bridge_offset))
    if sv._session_supports_live_pi_ui(s):
        streamed_payload = pi_live_messages_payload(manager, s, offset=live_offset)
        next_live_offset = int(streamed_payload.get("offset", max(0, int(live_offset))) or max(0, int(live_offset)))
        merged_events = merge_pi_live_message_events(
            merged_events,
            [item for item in (streamed_payload.get("events") or []) if isinstance(item, dict)],
        )
    bridge_events, next_bridge_offset = manager._bridge_events_since(
        sv._durable_session_id_for_live_session(s),
        offset=bridge_offset,
    )
    if bridge_events:
        merged_events = [*merged_events, *bridge_events]
    payload: dict[str, Any] = {
        "ok": True,
        "session_id": sv._durable_session_id_for_live_session(s),
        "runtime_id": s.session_id,
        "offset": int(page.get("offset", max(0, int(offset))) or 0),
        "live_offset": next_live_offset,
        "bridge_offset": next_bridge_offset,
        "has_older": bool(page.get("has_older")),
        "next_before": int(page.get("next_before", 0) or 0),
        "busy": bool(busy),
        "events": merged_events,
        "requests_version": current_requests_version,
        "token": token_val,
        "context_usage": sv._session_context_usage_payload(s, token_val),
        "turn_timing": sv._session_turn_timing_payload(s, merged_events, busy=bool(busy)),
        "transport_state": s.bridge_transport_state,
        "transport_error": s.bridge_transport_error,
    }
    if requests_version != current_requests_version:
        payload["requests"] = requests
    return payload



def pi_live_messages_payload(manager: Any, session: Any, *, offset: int = 0) -> dict[str, Any]:
    sv = _sv()
    if not sv._session_supports_live_pi_ui(session):
        return {"offset": max(0, int(offset)), "events": []}
    try:
        payload = manager._sock_call(
            session.sock_path,
            {"cmd": "live_messages", "offset": max(0, int(offset))},
            timeout_s=1.5,
        )
    except Exception:
        return {"offset": max(0, int(offset)), "events": []}
    events = payload.get("events")
    return {
        "offset": int(payload.get("offset", max(0, int(offset))) or 0),
        "events": [item for item in events if isinstance(item, dict)] if isinstance(events, list) else [],
    }



def _event_ts(event: dict[str, Any]) -> float | None:
    ts = event.get("ts")
    if isinstance(ts, (int, float)):
        return float(ts)
    return None



def _insert_event_by_ts(
    merged: list[dict[str, Any]], event: dict[str, Any]
) -> None:
    ts = _event_ts(event)
    if ts is None:
        merged.append(event)
        return
    insert_at = len(merged)
    while insert_at > 0:
        prev_ts = _event_ts(merged[insert_at - 1])
        if prev_ts is None or prev_ts <= ts:
            break
        insert_at -= 1
    merged.insert(insert_at, event)



def merge_pi_live_message_events(
    durable_events: list[dict[str, Any]], streamed_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = list(durable_events)
    durable_turn_ids = {
        str(event.get("turn_id"))
        for event in durable_events
        if event.get("role") == "assistant"
        and isinstance(event.get("turn_id"), str)
        and str(event.get("turn_id") or "")
    }
    tail_durable_text = ""
    for event in reversed(durable_events):
        if event.get("role") != "assistant":
            if event.get("role") == "user":
                break
            continue
        text = event.get("text")
        if isinstance(text, str) and text.strip():
            tail_durable_text = text.strip()
            break
    for event in streamed_events:
        if event.get("role") != "assistant":
            _insert_event_by_ts(merged, event)
            continue
        turn_id = event.get("turn_id") if isinstance(event.get("turn_id"), str) else None
        text = str(event.get("text") or "").strip()
        if turn_id and turn_id in durable_turn_ids:
            continue
        if bool(event.get("completed")) and text and text == tail_durable_text:
            continue
        _insert_event_by_ts(merged, event)
    return merged
