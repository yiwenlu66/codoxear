from __future__ import annotations

from typing import Any

from .cc_log import cc_assistant_is_final_turn_end
from .cc_log import cc_assistant_pending_tool_use_ids
from .cc_log import cc_assistant_text
from .agent_backend import get_agent_backend
from .cc_log import cc_apply_tool_result_to_pending
from .cc_log import cc_assistant_tool_use_count
from .cc_log import cc_message_role
from .cc_log import cc_user_text
from .pi_log import pi_assistant_error_text
from .pi_log import pi_assistant_is_aborted_turn
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
        if pi_assistant_text(obj) or pi_assistant_error_text(obj):
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
    if typ in ("user", "assistant"):
        return get_agent_backend("cc").chat_event_from_log_row(obj, cc_pending_tool_ids=cc_pending_tool_ids)
    if typ == "message":
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
    for event in events:
        role = event.get("role")
        if role == "user":
            last_assistant_key = None
            out.append(event)
            continue
        if role == "assistant":
            key = _chat_assistant_dedupe_key(event)
            if key is not None and key == last_assistant_key:
                continue
            last_assistant_key = key
            out.append(event)
            continue
        out.append(event)
    return out
