from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PiUnknownToolCallId:
    nonce: str
    index: int


@dataclass(frozen=True)
class PiDuplicateToolCallId:
    tool_id: str
    nonce: str
    index: int


PiPendingToolCallId = str | PiUnknownToolCallId | PiDuplicateToolCallId


def _text_parts(content: Any) -> list[str]:
    if isinstance(content, str):
        text = content.strip()
        return [text] if text else []
    if not isinstance(content, list):
        return []
    out: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            out.append(text)
    return out


def pi_user_text(obj: dict[str, Any]) -> str | None:
    if obj.get("type") != "message":
        return None
    message = obj.get("message")
    if not isinstance(message, dict) or message.get("role") != "user":
        return None
    parts = _text_parts(message.get("content"))
    if not parts:
        return None
    return "".join(parts)


def pi_assistant_content_parts(obj: dict[str, Any]) -> list[dict[str, Any]]:
    if obj.get("type") != "message":
        return []
    message = obj.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [part for part in content if isinstance(part, dict)]


def pi_assistant_text(obj: dict[str, Any]) -> str | None:
    out: list[str] = []
    for part in pi_assistant_content_parts(obj):
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            out.append(text)
    if not out:
        return None
    return "".join(out)


def pi_assistant_error_text(obj: dict[str, Any]) -> str | None:
    if obj.get("type") != "message":
        return None
    message = obj.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None
    if message.get("stopReason") != "error":
        return None
    error_message = message.get("errorMessage")
    if isinstance(error_message, str) and error_message.strip():
        return error_message.strip()
    return "Pi turn failed without errorMessage in log"


def pi_assistant_is_aborted_turn(obj: dict[str, Any]) -> bool:
    if obj.get("type") != "message":
        return False
    message = obj.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return False
    return message.get("stopReason") == "aborted"


def pi_assistant_is_terminal_no_visible_response(obj: dict[str, Any]) -> bool:
    if obj.get("type") != "message":
        return False
    message = obj.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return False
    if message.get("stopReason") not in {"stop", "end_turn"}:
        return False
    error_message = message.get("errorMessage")
    if isinstance(error_message, str) and error_message.strip():
        return False
    if message.get("isError") is True:
        return False
    if pi_assistant_text(obj):
        return False
    if pi_assistant_tool_use_count(obj) > 0:
        return False
    for part in pi_assistant_content_parts(obj):
        if part.get("type") == "toolResult":
            return False
    return True


def pi_assistant_is_final_turn_end(obj: dict[str, Any]) -> bool:
    if obj.get("type") != "message":
        return False
    message = obj.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return False
    if message.get("stopReason") == "length":
        return False
    if not pi_assistant_text(obj):
        return False
    if pi_assistant_tool_use_count(obj) > 0:
        return False

    if pi_assistant_thinking_count(obj) <= 0:
        stop_reason = message.get("stopReason")
        if not isinstance(stop_reason, str) or stop_reason != "toolUse":
            return True

    stop_reason = message.get("stopReason")
    if isinstance(stop_reason, str) and stop_reason and stop_reason != "toolUse":
        return True

    for part in pi_assistant_content_parts(obj):
        if part.get("type") != "text":
            continue
        raw_sig = part.get("textSignature")
        if not isinstance(raw_sig, str) or not raw_sig.strip():
            continue
        try:
            sig = json.loads(raw_sig)
        except Exception:
            continue
        if isinstance(sig, dict) and sig.get("phase") == "final_answer":
            return True
    return False


def pi_assistant_tool_use_count(obj: dict[str, Any]) -> int:
    count = 0
    for part in pi_assistant_content_parts(obj):
        if part.get("type") == "toolCall":
            count += 1
    return count


def pi_unknown_tool_call_id(index: int = 0) -> PiUnknownToolCallId:
    return PiUnknownToolCallId(nonce=uuid.uuid4().hex, index=index)


def pi_duplicate_tool_call_id(tool_id: str, index: int = 0) -> PiDuplicateToolCallId:
    return PiDuplicateToolCallId(tool_id=tool_id, nonce=uuid.uuid4().hex, index=index)


def pi_pending_tool_call_is_unknown(pending_id: object) -> bool:
    return isinstance(pending_id, PiUnknownToolCallId)


def pi_pending_tool_call_is_duplicate(pending_id: object) -> bool:
    return isinstance(pending_id, PiDuplicateToolCallId)


def pi_assistant_pending_tool_call_ids(obj: dict[str, Any]) -> list[PiPendingToolCallId]:
    ids: list[PiPendingToolCallId] = []
    seen: set[str] = set()
    unknown_index = 0
    duplicate_index = 0
    for part in pi_assistant_content_parts(obj):
        if part.get("type") != "toolCall":
            continue
        tool_id = part.get("id")
        if isinstance(tool_id, str):
            if tool_id not in seen:
                ids.append(tool_id)
                seen.add(tool_id)
            else:
                ids.append(pi_duplicate_tool_call_id(tool_id, duplicate_index))
                duplicate_index += 1
        else:
            ids.append(pi_unknown_tool_call_id(unknown_index))
            unknown_index += 1
    return ids


def pi_tool_result_id(obj: dict[str, Any]) -> str | None:
    if obj.get("type") != "message":
        return None
    message = obj.get("message")
    if not isinstance(message, dict) or message.get("role") != "toolResult":
        return None
    tool_id = message.get("toolCallId")
    return tool_id if isinstance(tool_id, str) else None


def pi_apply_tool_result_to_pending(obj: dict[str, Any], pending: set[PiPendingToolCallId]) -> None:
    tool_id = pi_tool_result_id(obj)
    if tool_id is None:
        return
    if tool_id in pending:
        pending.discard(tool_id)
        return
    for pending_id in list(pending):
        if isinstance(pending_id, PiDuplicateToolCallId) and pending_id.tool_id == tool_id:
            pending.discard(pending_id)
            return


def pi_apply_assistant_tool_calls_to_pending(obj: dict[str, Any], pending: set[PiPendingToolCallId]) -> None:
    for pending_id in pi_assistant_pending_tool_call_ids(obj):
        if isinstance(pending_id, str) and pending_id in pending:
            pending.add(pi_duplicate_tool_call_id(pending_id, len(pending)))
        else:
            pending.add(pending_id)


def pi_assistant_thinking_count(obj: dict[str, Any]) -> int:
    count = 0
    for part in pi_assistant_content_parts(obj):
        if part.get("type") == "thinking":
            count += 1
    return count


def pi_message_role(obj: dict[str, Any]) -> str | None:
    if obj.get("type") != "message":
        return None
    message = obj.get("message")
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    return role if isinstance(role, str) and role else None
