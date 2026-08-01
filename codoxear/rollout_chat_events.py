from __future__ import annotations

from typing import Any

from .cc_log import cc_assistant_is_final_turn_end
from .cc_log import cc_assistant_pending_tool_use_ids
from .cc_log import cc_assistant_text
from .agent_backend import get_agent_backend
from .cc_log import cc_apply_tool_result_to_pending
from .cc_log import cc_assistant_tool_use_count
from .cc_log import cc_is_turn_end
from .cc_log import cc_message_role
from .cc_log import cc_system_api_error_is_terminal
from .cc_log import cc_user_text
from .pi_log import pi_assistant_error_text
from .pi_log import pi_assistant_is_aborted_turn
from .pi_log import pi_assistant_is_terminal_no_visible_response
from .pi_log import pi_assistant_text
from .pi_log import pi_assistant_is_final_turn_end
from .pi_log import pi_user_text
from .rollout_events import _codex_error_affects_turn_status
from .rollout_events import _codex_event_text
from .rollout_events import _event_ts
from .rollout_events import _strip_oai_mem_citation_tail
from .rollout_events import _text_message_id


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
        if pi_assistant_is_aborted_turn(obj):
            return None
        if pi_assistant_text(obj) or pi_assistant_error_text(obj) or pi_assistant_is_terminal_no_visible_response(obj):
            return _event_ts(obj)
        return None

    if typ == "user":
        if cc_user_text(obj):
            return _event_ts(obj)
        return None

    if typ == "assistant":
        if cc_assistant_text(obj):
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


def _update_cc_pending_tool_ids(obj: dict[str, Any], pending: set[str]) -> None:
    typ = obj.get("type")
    if typ == "user":
        user_text = cc_user_text(obj)
        if isinstance(user_text, str) and user_text:
            pending.clear()
            return
        if cc_message_role(obj) == "toolResult":
            cc_apply_tool_result_to_pending(obj, pending)
            return
    if typ == "assistant" and cc_assistant_tool_use_count(obj) > 0:
        pending.update(cc_assistant_pending_tool_use_ids(obj))


def _single_chat_event(obj: dict[str, Any], *, cc_pending_tool_ids: set[str] | None = None) -> dict[str, Any] | None:
    typ = obj.get("type")
    if typ in ("user", "assistant", "system"):
        return get_agent_backend("cc").chat_event_from_log_row(obj, cc_pending_tool_ids=cc_pending_tool_ids)
    if typ in ("message", "custom_message", "active_long_running"):
        return get_agent_backend("pi").chat_event_from_log_row(obj)
    if typ in ("event_msg", "response_item"):
        return get_agent_backend("codex").chat_event_from_log_row(obj)
    return None


def _pi_message_keeps_turn_busy(obj: dict[str, Any]) -> bool:
    return get_agent_backend("pi").message_keeps_turn_busy(obj)


def _cc_message_keeps_turn_busy(obj: dict[str, Any]) -> bool:
    return get_agent_backend("cc").message_keeps_turn_busy(obj)


def _chat_assistant_dedupe_key(event: dict[str, Any]) -> tuple[str, str] | None:
    if event.get("role") != "assistant":
        return None
    text = event.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    message_class = event.get("message_class")
    normalized_text = " ".join(_strip_oai_mem_citation_tail(text).split())
    if not normalized_text:
        return None
    return (str(message_class or ""), normalized_text)


def _dedupe_assistant_chat_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    last_assistant_key: tuple[str, str] | None = None
    seen_pi_subagent_ids: set[str] = set()
    for event in events:
        role = event.get("role")
        if role == "user":
            last_assistant_key = None
            out.append(event)
            continue
        if role == "assistant":
            message_id = event.get("message_id")
            if isinstance(message_id, str) and message_id.startswith("pi-subagent:"):
                if message_id in seen_pi_subagent_ids:
                    continue
                seen_pi_subagent_ids.add(message_id)
            key = _chat_assistant_dedupe_key(event)
            if key is not None and key == last_assistant_key:
                continue
            last_assistant_key = key
            out.append(event)
            continue
        out.append(event)
    return out


# User-facing, backend-truthful text for a Codex turn that closed (task_complete
# / turn_complete) after a user message but produced no assistant output and no
# explicit error. The text states the observable fact — the turn ended with no
# answer — without inventing assistant content or leaking internal diagnostics
# (no stack traces, API details, or credential hints). It uses message_class
# "error" so the existing transcript renderer surfaces it with error styling.
_NO_RESPONSE_TEXT = "The backend completed this turn without producing a response."


def _build_no_response_event(obj: dict[str, Any]) -> dict[str, Any]:
    ts = _event_ts(obj)
    event: dict[str, Any] = {
        "role": "assistant",
        "text": _NO_RESPONSE_TEXT,
        "message_class": "error",
        "message_id": _text_message_id(message_class="error", text=_NO_RESPONSE_TEXT, ts=ts),
    }
    if ts is not None:
        event["ts"] = ts
    return event


def _detect_codex_no_response_closes(records: list[Any]) -> list[tuple[int, int, dict[str, Any]]]:
    """Locate Codex turn-close boundaries for no-response detection.

    Returns ``(user_byte, close_byte, close_obj)`` triples for each Codex
    ``event_msg`` ``task_complete`` / ``turn_complete`` record that closes a
    turn begun by a non-empty ``user_message``. ``user_byte`` is the byte
    offset of the most recent ``user_message`` record before the close;
    ``close_byte`` is ``record.start`` of the closing record.

    This function only identifies turn boundaries. It deliberately does **not**
    decide whether the turn produced a response: that judgement belongs to
    :func:`_inject_no_response_events`, which consults the already-extracted
    chat events (the source of truth for what is visibly rendered). Only Codex
    ``event_msg`` turn-close types trigger detection, so valid Pi and Claude
    Code turns — which use different close mechanisms — are unaffected.
    """
    closes: list[tuple[int, int, dict[str, Any]]] = []
    user_byte: int | None = None
    for record in records:
        obj = record.obj
        if obj.get("type") != "event_msg":
            continue
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        pt = payload.get("type")
        if pt == "user_message":
            if isinstance(payload.get("message"), str) and payload["message"].strip():
                user_byte = record.start
            continue
        if pt in ("task_complete", "turn_complete"):
            if user_byte is not None:
                closes.append((user_byte, record.start, obj))
            user_byte = None
            continue
    return closes


def _visible_assistant_event_bytes(
    events: list[dict[str, Any]],
) -> tuple[set[int], bool]:
    """Return ``(bytes, has_unpositioned)`` for positioned assistant events.

    ``bytes`` holds the ``_before_byte`` of every positioned assistant event
    (the source of truth for what the transcript renders). ``has_unpositioned``
    is True if any assistant event lacked a byte offset — such events cannot
    be ordered against closes and are treated as globally visible to avoid
    false no-response injection (matches the historical behaviour for
    synthetic test inputs).
    """
    bytes_set: set[int] = set()
    has_unpositioned = False
    for event in events:
        if event.get("role") != "assistant":
            continue
        ev_byte = event.get("_before_byte")
        if isinstance(ev_byte, int):
            bytes_set.add(ev_byte)
        else:
            has_unpositioned = True
    return bytes_set, has_unpositioned


def _inject_no_response_events(
    records: list[Any],
    events: list[dict[str, Any]],
    *,
    prior_user_byte: int | None = None,
    prior_turn_has_assistant: bool = False,
) -> list[dict[str, Any]]:
    """Inject explicit no-response events when a user turn closes with no
    visible assistant output.

    Visible assistant transcript messages remain the source of truth for
    suppressing a no-response row. In-window visibility is decided from the
    positioned ``events`` (the normalizer's output); pre-window visibility —
    relevant when a live delta / history page splits the user row from the
    close row — is supplied via ``prior_user_byte`` / ``prior_turn_has_assistant``
    (computed by the caller from the same ``_single_chat_event`` extractor, so
    the detector cannot diverge from the extractor).

    ``prior_user_byte`` seeds the open-turn state: if the window begins with an
    open user turn (delivered in an earlier poll, no close yet), a close
    arriving in this window still triggers detection. ``prior_turn_has_assistant``
    suppresses injection when that prior open turn already produced a visible
    assistant event before the window. An in-window user row overrides the prior
    context (a newer turn supersedes it).

    Codex closes are ``event_msg`` ``task_complete`` / ``turn_complete`` rows.
    Claude Code closes are ``system/turn_duration`` rows. Terminal
    ``system/api_error`` rows project their own error event and also close the
    turn so a later close cannot add a generic no-response duplicate.
    """
    asst_bytes, has_unpositioned_asst = _visible_assistant_event_bytes(events)
    if has_unpositioned_asst:
        # An assistant event that cannot be byte-ordered cannot be reliably
        # placed relative to a close; treat the window as answered and do not
        # inject (preserves historical behaviour for synthetic test inputs).
        return events

    result: list[dict[str, Any]] = []
    i = 0
    n = len(events)
    user_byte: int | None = prior_user_byte
    user_from_prior = prior_user_byte is not None

    for record in records:
        obj = record.obj
        typ = obj.get("type")
        if typ == "event_msg":
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                continue
            pt = payload.get("type")
            if pt == "user_message":
                if isinstance(payload.get("message"), str) and payload["message"].strip():
                    user_byte = record.start
                    user_from_prior = False
                continue
            is_close = pt in ("task_complete", "turn_complete")
        elif typ == "user":
            user_text = cc_user_text(obj)
            if isinstance(user_text, str) and user_text:
                user_byte = record.start
                user_from_prior = False
            continue
        elif typ == "system":
            is_close = cc_is_turn_end(obj) or cc_system_api_error_is_terminal(obj)
        else:
            continue

        if is_close:
            close_byte = record.start
            # Drain every positioned event up to and including this close so
            # the merge cursor stays byte-ordered regardless of injection.
            while i < n:
                ev_byte = events[i].get("_before_byte")
                if isinstance(ev_byte, int) and ev_byte > close_byte:
                    break
                result.append(events[i])
                i += 1
            if user_byte is not None:
                answered = (user_from_prior and prior_turn_has_assistant) or any(
                    user_byte < b <= close_byte for b in asst_bytes
                )
                if not answered:
                    positioned = dict(_build_no_response_event(obj))
                    positioned["_before_byte"] = close_byte
                    result.append(positioned)
            user_byte = None
            user_from_prior = False
            continue

    while i < n:
        result.append(events[i])
        i += 1
    return result
