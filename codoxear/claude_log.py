from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import CONTEXT_WINDOW_BASELINE_TOKENS


def _read_jsonl_first_object(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                return obj if isinstance(obj, dict) else None
    except FileNotFoundError:
        return None
    return None


def _context_percent_remaining(*, tokens_in_context: int, context_window: int) -> int:
    if context_window <= CONTEXT_WINDOW_BASELINE_TOKENS:
        return 0
    effective = context_window - CONTEXT_WINDOW_BASELINE_TOKENS
    used = max(tokens_in_context - CONTEXT_WINDOW_BASELINE_TOKENS, 0)
    remaining = max(effective - used, 0)
    return int(round((remaining / effective) * 100.0))


def _claude_context_window(model: str | None) -> int | None:
    if not isinstance(model, str) or not model.strip():
        return None
    model = model.strip()
    if model.startswith("claude-opus-4"):
        return 1024 * 1024
    if model.startswith("claude-sonnet-4"):
        return 200 * 1024
    if model.startswith("claude-haiku-4"):
        return 200 * 1024
    if model.startswith("claude-3-5-"):
        return 200 * 1024
    return None


def read_claude_session_id(path: Path) -> str | None:
    try:
        with path.open("rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                session_id = obj.get("sessionId")
                if isinstance(session_id, str) and session_id.strip():
                    return session_id
    except FileNotFoundError:
        return None
    return None


def read_claude_log_cwd(path: Path) -> str | None:
    try:
        with path.open("rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                cwd = obj.get("cwd")
                if isinstance(cwd, str) and cwd.strip():
                    return cwd
    except FileNotFoundError:
        return None
    return None


def read_claude_run_settings(path: Path, *, max_scan_bytes: int = 8 * 1024 * 1024) -> tuple[str | None, str | None, str | None]:
    provider: str | None = None
    model: str | None = None
    reasoning_level: str | None = None

    try:
        size = int(path.stat().st_size)
    except FileNotFoundError:
        return provider, model, reasoning_level
    except Exception:
        return provider, model, reasoning_level

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
                if not isinstance(obj, dict):
                    continue
                message = obj.get("message")
                if isinstance(message, dict):
                    if model is None:
                        raw_model = message.get("model")
                        if isinstance(raw_model, str) and raw_model.strip():
                            model = raw_model
                            if provider is None:
                                provider = "anthropic"
    except FileNotFoundError:
        return provider, model, reasoning_level

    if provider is None:
        provider = "anthropic"
    return provider, model, reasoning_level


def claude_user_text(obj: dict[str, Any]) -> str | None:
    if obj.get("type") != "user":
        return None
    message = obj.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
        return text if text else None
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    out.append(text)
            elif part_type == "tool_result":
                return None
        if out:
            return "".join(out)
    return None


def claude_assistant_text(obj: dict[str, Any]) -> str | None:
    if obj.get("type") != "assistant":
        return None
    message = obj.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None
    out: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            out.append(text)
    if not out:
        return None
    return "".join(out)


def claude_assistant_tool_use_count(obj: dict[str, Any]) -> int:
    if obj.get("type") != "assistant":
        return 0
    message = obj.get("message")
    if not isinstance(message, dict):
        return 0
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    count = 0
    for part in content:
        if isinstance(part, dict) and part.get("type") == "tool_use":
            count += 1
    return count


def claude_assistant_thinking_count(obj: dict[str, Any]) -> int:
    if obj.get("type") != "assistant":
        return 0
    message = obj.get("message")
    if not isinstance(message, dict):
        return 0
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    count = 0
    for part in content:
        if isinstance(part, dict) and part.get("type") == "thinking":
            count += 1
    return count


def claude_is_turn_end(obj: dict[str, Any]) -> bool:
    if obj.get("type") != "system":
        return False
    subtype = obj.get("subtype")
    return subtype == "turn_duration"


def claude_is_final_answer(obj: dict[str, Any]) -> bool:
    if obj.get("type") != "assistant":
        return False
    if not claude_assistant_text(obj):
        return False
    message = obj.get("message")
    if not isinstance(message, dict):
        return False
    stop_reason = message.get("stop_reason")
    if stop_reason != "end_turn":
        return False
    if claude_assistant_tool_use_count(obj) > 0:
        return False
    return True


def claude_token_update(obj: dict[str, Any]) -> dict[str, Any] | None:
    if obj.get("type") != "assistant":
        return None
    message = obj.get("message")
    if not isinstance(message, dict):
        return None
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    cache_read_input_tokens = usage.get("cache_read_input_tokens", 0)
    cache_creation_input_tokens = usage.get("cache_creation_input_tokens", 0)

    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None

    model = message.get("model")
    context_window = _claude_context_window(model if isinstance(model, str) else None)
    if not isinstance(context_window, int) or context_window <= 0:
        return None

    tokens_in_context = int(input_tokens) + int(output_tokens) + int(cache_read_input_tokens or 0) + int(cache_creation_input_tokens or 0)
    as_of = obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None

    return {
        "context_window": context_window,
        "tokens_in_context": tokens_in_context,
        "tokens_remaining": max(context_window - tokens_in_context, 0),
        "percent_remaining": _context_percent_remaining(tokens_in_context=tokens_in_context, context_window=context_window),
        "baseline_tokens": CONTEXT_WINDOW_BASELINE_TOKENS,
        "as_of": as_of,
    }
