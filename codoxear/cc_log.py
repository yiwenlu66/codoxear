from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any
from typing import Iterator


CC_SUPPORTED_REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")
CC_UNKNOWN_TOOL_USE_ID_PREFIX = "__codoxear_cc_unknown_tool_use__:"
CC_UNKNOWN_TOOL_USE_ID = f"{CC_UNKNOWN_TOOL_USE_ID_PREFIX}0"


def _read_jsonl_first_object_with_session_id(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                if isinstance(obj, dict) and isinstance(obj.get("sessionId"), str) and obj.get("sessionId"):
                    return obj
    except FileNotFoundError:
        return None
    return None


def _message(obj: dict[str, Any], *, role: str | None = None) -> dict[str, Any] | None:
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return None
    if role is not None and msg.get("role") != role:
        return None
    return msg


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


def cc_user_text(obj: dict[str, Any]) -> str | None:
    if obj.get("type") != "user" or obj.get("isMeta") is True:
        return None
    msg = _message(obj, role="user")
    if msg is None:
        return None
    if obj.get("toolUseResult") is not None:
        return None
    content = msg.get("content")
    if isinstance(content, list):
        # Claude routes tool results through user-role records. Those are transport,
        # not human chat messages, even if schema drift adds sibling text parts.
        if any(isinstance(part, dict) and part.get("type") == "tool_result" for part in content):
            return None
    parts = _text_parts(content)
    if not parts:
        return None
    text = "".join(parts).strip()
    if not text:
        return None
    return text


def cc_assistant_content_parts(obj: dict[str, Any]) -> list[dict[str, Any]]:
    if obj.get("type") != "assistant":
        return []
    msg = _message(obj, role="assistant")
    if msg is None:
        return []
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    return [part for part in content if isinstance(part, dict)]


def cc_assistant_text(obj: dict[str, Any]) -> str | None:
    out: list[str] = []
    for part in cc_assistant_content_parts(obj):
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            out.append(text)
    if not out:
        return None
    return "".join(out)


def cc_assistant_tool_use_ids(obj: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for part in cc_assistant_content_parts(obj):
        if part.get("type") != "tool_use":
            continue
        tool_id = part.get("id")
        if isinstance(tool_id, str) and tool_id.strip():
            ids.append(tool_id)
    return ids


def cc_user_tool_result_ids(obj: dict[str, Any]) -> list[str]:
    if obj.get("type") != "user":
        return []
    ids: list[str] = []
    top_level_result = obj.get("toolUseResult")
    if isinstance(top_level_result, dict):
        for key in ("tool_use_id", "toolUseId", "toolUseID", "tool_useID", "id"):
            tool_id = top_level_result.get(key)
            if isinstance(tool_id, str) and tool_id.strip():
                ids.append(tool_id)
                break
    msg = _message(obj, role="user")
    if msg is None:
        return ids
    content = msg.get("content")
    if not isinstance(content, list):
        return ids
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "tool_result":
            continue
        tool_id = part.get("tool_use_id")
        if isinstance(tool_id, str) and tool_id.strip():
            ids.append(tool_id)
    return ids


def cc_assistant_pending_tool_use_ids(obj: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    unknown_index = 0
    for part in cc_assistant_content_parts(obj):
        if part.get("type") != "tool_use":
            continue
        tool_id = part.get("id")
        if isinstance(tool_id, str) and tool_id.strip():
            ids.add(tool_id)
        else:
            ids.add(f"{CC_UNKNOWN_TOOL_USE_ID_PREFIX}{uuid.uuid4().hex}:{unknown_index}")
            unknown_index += 1
    return ids


def cc_discard_one_unknown_tool_use_id(pending: set[str]) -> None:
    for tool_id in sorted(pending):
        if tool_id.startswith(CC_UNKNOWN_TOOL_USE_ID_PREFIX):
            pending.discard(tool_id)
            return


def cc_apply_tool_result_to_pending(obj: dict[str, Any], pending: set[str]) -> None:
    result_ids = cc_user_tool_result_ids(obj)
    if result_ids:
        for tool_id in result_ids:
            pending.discard(tool_id)
        return
    if obj.get("toolUseResult") is not None and len(pending) == 1:
        pending.clear()
        return
    cc_discard_one_unknown_tool_use_id(pending)


def cc_update_pending_tool_ids(obj: dict[str, Any], pending: set[str]) -> None:
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


def _iter_jsonl_objects_reverse(path: Path, *, before: int | None = None, block_bytes: int = 64 * 1024) -> Iterator[dict[str, Any]]:
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        end = size if before is None else max(0, min(int(before), size))
        offset = end
        carry = b""
        drop_trailing_partial = False
        if end > 0:
            f.seek(end - 1)
            drop_trailing_partial = f.read(1) != b"\n"
        while offset > 0:
            read_size = min(block_bytes, offset)
            offset -= read_size
            f.seek(offset)
            chunk = f.read(read_size)
            data = chunk + carry
            parts = data.split(b"\n")
            if drop_trailing_partial and parts:
                parts = parts[:-1]
                drop_trailing_partial = False
            if offset > 0:
                carry = parts[0] if parts else b""
                parts = parts[1:] if parts else []
            else:
                carry = b""
            for raw_line in reversed(parts):
                line = raw_line.rstrip(b"\r")
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj


def cc_current_turn_state_before(log_path: Path, before: int) -> tuple[set[str], bool | None]:
    before = max(0, int(before))
    if before <= 0:
        return set(), None
    newest_first: list[dict[str, Any]] = []
    for obj in _iter_jsonl_objects_reverse(log_path, before=before):
        newest_first.append(obj)
        if obj.get("type") == "user" and cc_user_text(obj):
            break

    pending: set[str] = set()
    saw_signal = False
    idle = True
    for obj in reversed(newest_first):
        typ = obj.get("type")
        if typ == "user":
            user_text = cc_user_text(obj)
            if isinstance(user_text, str) and user_text:
                pending.clear()
                saw_signal = True
                idle = False
                continue
            if cc_message_role(obj) == "toolResult":
                cc_update_pending_tool_ids(obj, pending)
                saw_signal = True
                idle = False
                continue
        if typ == "assistant":
            tool_count = cc_assistant_tool_use_count(obj)
            if tool_count > 0:
                pending.update(cc_assistant_pending_tool_use_ids(obj))
            assistant_text = cc_assistant_text(obj)
            if isinstance(assistant_text, str) and assistant_text:
                saw_signal = True
                if cc_assistant_is_api_error(obj):
                    # A backend API error closes the turn regardless of
                    # stop_reason (which is "stop_sequence" on these rows).
                    pending.clear()
                    idle = True
                else:
                    idle = bool(cc_assistant_is_final_turn_end(obj) and not pending)
                continue
            if tool_count > 0 or cc_assistant_thinking_count(obj) > 0:
                saw_signal = True
                idle = False
                continue
        if typ == "system" and cc_is_turn_end(obj):
            saw_signal = True
            idle = not pending
            continue
    return pending, (idle if saw_signal else None)


def cc_pending_tool_ids_before(log_path: Path, before: int) -> set[str]:
    pending, _idle = cc_current_turn_state_before(log_path, before)
    return pending


def cc_assistant_tool_use_count(obj: dict[str, Any]) -> int:
    return sum(1 for part in cc_assistant_content_parts(obj) if part.get("type") == "tool_use")


def cc_assistant_thinking_count(obj: dict[str, Any]) -> int:
    return sum(1 for part in cc_assistant_content_parts(obj) if part.get("type") == "thinking")


def cc_assistant_is_final_turn_end(obj: dict[str, Any]) -> bool:
    if obj.get("type") != "assistant":
        return False
    msg = _message(obj, role="assistant")
    if msg is None:
        return False
    if not cc_assistant_text(obj):
        return False
    if cc_assistant_tool_use_count(obj) > 0:
        return False
    return msg.get("stop_reason") == "end_turn"


def cc_assistant_is_api_error(obj: dict[str, Any]) -> bool:
    """Detect a Claude Code API/gateway error assistant row.

    Claude Code logs transport/provider failures as synthetic assistant rows
    with ``isApiErrorMessage: true`` (and ``message.model == "<synthetic>"``).
    These rows carry the backend error text in a normal ``text`` content block
    but are NOT real assistant output — they must be classified as errors and
    close the turn.
    """
    if obj.get("type") != "assistant":
        return False
    return obj.get("isApiErrorMessage") is True


def cc_message_role(obj: dict[str, Any]) -> str | None:
    typ = obj.get("type")
    if typ == "assistant":
        return "assistant"
    if typ == "system":
        return "system"
    if typ != "user":
        return None
    if obj.get("toolUseResult") is not None:
        return "toolResult"
    msg = _message(obj)
    if msg is None:
        return None
    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_result":
                return "toolResult"
    role = msg.get("role")
    return role if isinstance(role, str) and role else None


def cc_is_turn_end(obj: dict[str, Any]) -> bool:
    return obj.get("type") == "system" and obj.get("subtype") == "turn_duration"


def read_cc_session_header(path: Path, *, max_scan_bytes: int = 512 * 1024) -> dict[str, Any] | None:
    session_id: str | None = None
    cwd: str | None = None
    timestamp: str | None = None
    git_branch: str | None = None
    version: str | None = None
    offset = 0
    try:
        with path.open("rb") as f:
            for raw in f:
                line_start = offset
                offset += len(raw)
                if line_start >= max_scan_bytes:
                    break
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                raw_session_id = obj.get("sessionId")
                if session_id is None and isinstance(raw_session_id, str) and raw_session_id.strip():
                    session_id = raw_session_id
                raw_cwd = obj.get("cwd")
                if cwd is None and isinstance(raw_cwd, str) and raw_cwd.strip():
                    cwd = raw_cwd
                raw_timestamp = obj.get("timestamp")
                if timestamp is None and isinstance(raw_timestamp, str) and raw_timestamp.strip():
                    timestamp = raw_timestamp
                raw_git_branch = obj.get("gitBranch")
                if git_branch is None and isinstance(raw_git_branch, str) and raw_git_branch.strip():
                    git_branch = raw_git_branch
                raw_version = obj.get("version")
                if version is None and isinstance(raw_version, str) and raw_version.strip():
                    version = raw_version
                if session_id is not None and cwd is not None:
                    break
    except FileNotFoundError:
        return None
    if session_id is None:
        return None
    payload: dict[str, Any] = {
        "id": session_id,
        "sessionId": session_id,
    }
    if cwd is not None:
        payload["cwd"] = cwd
    if timestamp is not None:
        payload["timestamp"] = timestamp
    if git_branch is not None:
        payload["git"] = {"branch": git_branch}
    if version is not None:
        payload["version"] = version
    return payload


def read_cc_session_id(path: Path) -> str | None:
    header = read_cc_session_header(path)
    if not isinstance(header, dict):
        return None
    session_id = header.get("id")
    return session_id if isinstance(session_id, str) and session_id.strip() else None


def read_cc_log_cwd(path: Path) -> str | None:
    header = read_cc_session_header(path)
    if not isinstance(header, dict):
        return None
    cwd = header.get("cwd")
    return cwd if isinstance(cwd, str) and cwd.strip() else None


def read_cc_run_settings(path: Path, *, max_scan_bytes: int = 8 * 1024 * 1024) -> tuple[str | None, str | None, str | None]:
    model: str | None = None
    try:
        size = int(path.stat().st_size)
    except FileNotFoundError:
        return None, None, None
    except Exception:
        return None, None, None
    start = max(0, size - int(max_scan_bytes))
    try:
        with path.open("rb") as f:
            if start > 0:
                f.seek(start)
                _ = f.readline()
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(obj, dict) or obj.get("type") != "assistant":
                    continue
                msg = _message(obj, role="assistant")
                if msg is None:
                    continue
                raw_model = msg.get("model")
                if isinstance(raw_model, str) and raw_model.strip():
                    model = raw_model
    except FileNotFoundError:
        return None, model, None
    return None, model, None
