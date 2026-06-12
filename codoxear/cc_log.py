from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CC_SUPPORTED_REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")
CC_UNKNOWN_TOOL_USE_ID = "__codoxear_cc_unknown_tool_use__"


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
    msg = _message(obj, role="user")
    if msg is None:
        return []
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    ids: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "tool_result":
            continue
        tool_id = part.get("tool_use_id")
        if isinstance(tool_id, str) and tool_id.strip():
            ids.append(tool_id)
    return ids


def cc_assistant_pending_tool_use_ids(obj: dict[str, Any]) -> set[str]:
    ids = set(cc_assistant_tool_use_ids(obj))
    if cc_assistant_tool_use_count(obj) > len(ids):
        ids.add(CC_UNKNOWN_TOOL_USE_ID)
    return ids


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


def cc_message_role(obj: dict[str, Any]) -> str | None:
    typ = obj.get("type")
    if typ == "assistant":
        return "assistant"
    if typ == "system":
        return "system"
    if typ != "user":
        return None
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


def read_cc_session_header(path: Path) -> dict[str, Any] | None:
    obj = _read_jsonl_first_object_with_session_id(path)
    if not isinstance(obj, dict):
        return None
    session_id = obj.get("sessionId")
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    payload: dict[str, Any] = {
        "id": session_id,
        "sessionId": session_id,
    }
    cwd = obj.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        payload["cwd"] = cwd
    timestamp = obj.get("timestamp")
    if isinstance(timestamp, str) and timestamp.strip():
        payload["timestamp"] = timestamp
    git_branch = obj.get("gitBranch")
    if isinstance(git_branch, str) and git_branch.strip():
        payload["git"] = {"branch": git_branch}
    version = obj.get("version")
    if isinstance(version, str) and version.strip():
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
