from __future__ import annotations

from typing import Any

from .cc_log import cc_apply_tool_result_to_pending
from .cc_log import cc_assistant_is_final_turn_end
from .cc_log import cc_assistant_pending_tool_use_ids
from .cc_log import cc_assistant_text
from .cc_log import cc_assistant_tool_use_count
from .cc_log import cc_message_role
from .cc_log import cc_user_text
from .pi_log import pi_assistant_is_aborted_turn
from .pi_log import pi_assistant_is_final_turn_end
from .pi_log import pi_assistant_text
from .rollout_events import _event_ts
from .rollout_events import _strip_oai_mem_citation_tail
from .rollout_events import _text_message_id
from .voice_push_state import ClassifiedAssistantMessage


def _extract_delivery_messages(
    objs: list[dict[str, Any]],
    *,
    initial_cc_pending_tool_ids: set[str] | None = None,
) -> list[ClassifiedAssistantMessage]:
    out: list[ClassifiedAssistantMessage] = []
    seen: set[str] = set()
    last_text_key: tuple[str, str] | None = None
    cc_pending_tool_ids: set[str] = set(initial_cc_pending_tool_ids or set())

    for obj in objs:
        if not isinstance(obj, dict):
            continue
        typ = obj.get("type")
        message_class: str | None = None
        text = ""
        if typ == "user":
            user_text = cc_user_text(obj)
            if isinstance(user_text, str) and user_text:
                cc_pending_tool_ids.clear()
                continue
            if cc_message_role(obj) == "toolResult":
                cc_apply_tool_result_to_pending(obj, cc_pending_tool_ids)
                continue
        if typ == "message":
            if pi_assistant_is_aborted_turn(obj):
                continue
            text = pi_assistant_text(obj) or ""
            if not text.strip():
                continue
            message_class = "final_response" if pi_assistant_is_final_turn_end(obj) else "narration"
        elif typ == "assistant":
            if cc_assistant_tool_use_count(obj) > 0:
                cc_pending_tool_ids.update(cc_assistant_pending_tool_use_ids(obj))
            text = cc_assistant_text(obj) or ""
            if not text.strip():
                continue
            message_class = "final_response" if cc_assistant_is_final_turn_end(obj) and not cc_pending_tool_ids else "narration"
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

