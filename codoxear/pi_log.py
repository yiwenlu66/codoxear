from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from typing import Iterator

from .pi_context import PI_DEFAULT_RESERVED_TOKENS
from .pi_context import PI_MODEL_QUERY_ID
from .pi_context import PI_MODEL_QUERY_TIMEOUT_SECONDS
from .pi_context import _context_percent_remaining as _context_percent_remaining_impl
from .pi_context import _context_token_update as _context_token_update_impl
from .pi_context import _context_windows_from_model_rows
from .pi_context import _context_windows_from_models_file
from .pi_context import _default_pi_models_path as _default_pi_models_path_impl
from .pi_context import _default_pi_settings_path as _default_pi_settings_path_impl
from .pi_context import _file_mtime_ns
from .pi_context import _pi_context_windows
from .pi_context import _pi_reserved_tokens
from .pi_context import _pi_rpc_context_windows
from .pi_context import _query_pi_context_windows as _query_pi_context_windows_impl
from .pi_message import PiDuplicateToolCallId
from .pi_message import PiPendingToolCallId
from .pi_message import PiUnknownToolCallId
from .pi_message import pi_apply_assistant_tool_calls_to_pending
from .pi_message import pi_apply_tool_result_to_pending
from .pi_message import pi_assistant_content_parts
from .pi_message import pi_assistant_error_text
from .pi_message import pi_assistant_is_aborted_turn
from .pi_message import pi_assistant_is_final_turn_end
from .pi_message import pi_assistant_is_terminal_no_visible_response
from .pi_message import pi_assistant_pending_tool_call_ids
from .pi_message import pi_assistant_text
from .pi_message import pi_assistant_thinking_count
from .pi_message import pi_assistant_tool_use_count
from .pi_message import pi_duplicate_tool_call_id
from .pi_message import pi_message_role
from .pi_message import pi_pending_tool_call_is_duplicate
from .pi_message import pi_pending_tool_call_is_unknown
from .pi_message import pi_tool_result_id
from .pi_message import pi_unknown_tool_call_id
from .pi_message import pi_user_text

PI_UNKNOWN_TOOL_CALL_ID_PREFIX = "__pi_unknown_tool_call__:"
PI_DUPLICATE_TOOL_CALL_ID_PREFIX = "__pi_duplicate_tool_call__:"


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


def _context_percent_remaining(*, used_input_context_tokens: int, max_input_tokens: int) -> int:
    return _context_percent_remaining_impl(
        used_input_context_tokens=used_input_context_tokens,
        max_input_tokens=max_input_tokens,
    )


def _context_token_update(*, context_window: int, tokens_in_context: int, reserved_tokens: int, as_of: str | None = None) -> dict[str, Any]:
    return _context_token_update_impl(
        context_window=context_window,
        tokens_in_context=tokens_in_context,
        reserved_tokens=reserved_tokens,
        as_of=as_of,
    )


def _default_pi_models_path() -> Path:
    return _default_pi_models_path_impl()


def _default_pi_settings_path() -> Path:
    return _default_pi_settings_path_impl()


def _query_pi_context_windows(models_path: Path) -> dict[tuple[str, str], int]:
    return _query_pi_context_windows_impl(models_path)


def pi_model_context_window(provider: str | None, model: str | None, *, models_path: Path | None = None) -> int | None:
    if not isinstance(provider, str) or not provider.strip():
        return None
    if not isinstance(model, str) or not model.strip():
        return None
    key = (provider.strip(), model.strip())
    path = _default_pi_models_path() if models_path is None else models_path
    local_index = _context_windows_from_models_file(path)
    local_context_window = local_index.get(key)
    if isinstance(local_context_window, int) and local_context_window > 0:
        return local_context_window
    if models_path is None:
        queried_context_window = _query_pi_context_windows(path).get(key)
        if isinstance(queried_context_window, int) and queried_context_window > 0:
            return queried_context_window
    return None


def pi_reserved_tokens(*, settings_path: Path | None = None) -> int:
    path = _default_pi_settings_path() if settings_path is None else settings_path
    try:
        stat = path.stat()
    except FileNotFoundError:
        return PI_DEFAULT_RESERVED_TOKENS
    except Exception:
        return PI_DEFAULT_RESERVED_TOKENS
    try:
        return _pi_reserved_tokens(str(path.resolve()), int(stat.st_mtime_ns))
    except Exception:
        return PI_DEFAULT_RESERVED_TOKENS


def pi_context_token_update(*, context_window: int, tokens_in_context: int, as_of: str | None = None, reserved_tokens: int | None = None, settings_path: Path | None = None) -> dict[str, Any]:
    reserve = pi_reserved_tokens(settings_path=settings_path) if reserved_tokens is None else reserved_tokens
    return _context_token_update(context_window=context_window, tokens_in_context=tokens_in_context, reserved_tokens=reserve, as_of=as_of)

def read_pi_session_header(path: Path) -> dict[str, Any] | None:
    obj = _read_jsonl_first_object(path)
    if not isinstance(obj, dict) or obj.get("type") != "session":
        return None
    return obj


def read_pi_session_id(path: Path) -> str | None:
    obj = read_pi_session_header(path)
    if not isinstance(obj, dict):
        return None
    session_id = obj.get("id")
    return session_id if isinstance(session_id, str) and session_id.strip() else None


def read_pi_log_cwd(path: Path) -> str | None:
    obj = read_pi_session_header(path)
    if not isinstance(obj, dict):
        return None
    cwd = obj.get("cwd")
    return cwd if isinstance(cwd, str) and cwd.strip() else None



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


def pi_complete_jsonl_offset_before(log_path: Path, before: int) -> int:
    before = max(0, int(before))
    if before <= 0:
        return 0
    with log_path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        end = max(0, min(before, size))
        if end <= 0:
            return 0
        f.seek(end - 1)
        if f.read(1) == b"\n":
            return end
        offset = end
        while offset > 0:
            read_size = min(64 * 1024, offset)
            offset -= read_size
            f.seek(offset)
            chunk = f.read(read_size)
            found = chunk.rfind(b"\n")
            if found >= 0:
                return offset + found + 1
    return 0


def pi_current_turn_state_before(log_path: Path, before: int) -> tuple[set[PiPendingToolCallId], bool | None]:
    before = pi_complete_jsonl_offset_before(log_path, before)
    if before <= 0:
        return set(), None
    newest_first: list[dict[str, Any]] = []
    for obj in _iter_jsonl_objects_reverse(log_path, before=before):
        newest_first.append(obj)
        if obj.get("type") == "message" and pi_user_text(obj):
            break

    pending: set[PiPendingToolCallId] = set()
    saw_signal = False
    idle = True
    for obj in reversed(newest_first):
        if obj.get("type") != "message":
            continue
        if pi_user_text(obj):
            pending.clear()
            saw_signal = True
            idle = False
            continue
        if pi_assistant_is_aborted_turn(obj):
            pending.clear()
            saw_signal = True
            idle = True
            continue
        if pi_assistant_error_text(obj):
            pending.clear()
            saw_signal = True
            idle = True
            continue
        if pi_assistant_is_terminal_no_visible_response(obj):
            pending.clear()
            saw_signal = True
            idle = True
            continue
        if pi_message_role(obj) == "toolResult":
            pi_apply_tool_result_to_pending(obj, pending)
            saw_signal = True
            idle = False
            continue
        tool_count = pi_assistant_tool_use_count(obj)
        if tool_count > 0:
            pi_apply_assistant_tool_calls_to_pending(obj, pending)
            saw_signal = True
            idle = False
            continue
        if pi_assistant_text(obj):
            saw_signal = True
            if pi_assistant_is_final_turn_end(obj):
                pending.clear()
                idle = True
            else:
                idle = False
            continue
        if pi_assistant_thinking_count(obj) > 0:
            saw_signal = True
            idle = False
            continue
    return pending, (idle if saw_signal else None)


def pi_pending_tool_ids_before(log_path: Path, before: int) -> set[PiPendingToolCallId]:
    pending, _idle = pi_current_turn_state_before(log_path, before)
    return pending



def pi_token_update(obj: dict[str, Any], *, models_path: Path | None = None, settings_path: Path | None = None) -> dict[str, Any] | None:
    if obj.get("type") != "message":
        return None
    message = obj.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None
    stop_reason = message.get("stopReason")
    if stop_reason == "aborted" or stop_reason == "error":
        return None
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None
    total_tokens = usage.get("totalTokens")
    if not isinstance(total_tokens, int):
        return None
    provider = message.get("provider")
    model = message.get("model")
    context_window = pi_model_context_window(provider if isinstance(provider, str) else None, model if isinstance(model, str) else None, models_path=models_path)
    if not isinstance(context_window, int) or context_window <= 0:
        return None
    as_of = obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None
    return pi_context_token_update(context_window=context_window, tokens_in_context=total_tokens, as_of=as_of, settings_path=settings_path)


def read_pi_run_settings(path: Path, *, max_scan_bytes: int = 8 * 1024 * 1024) -> tuple[str | None, str | None, str | None]:
    provider: str | None = None
    model: str | None = None
    thinking_level: str | None = None

    header = read_pi_session_header(path)
    if isinstance(header, dict):
        raw_provider = header.get("provider")
        raw_model = header.get("modelId")
        raw_thinking = header.get("thinkingLevel")
        if isinstance(raw_provider, str) and raw_provider.strip():
            provider = raw_provider
        if isinstance(raw_model, str) and raw_model.strip():
            model = raw_model
        if isinstance(raw_thinking, str) and raw_thinking.strip():
            thinking_level = raw_thinking

    try:
        size = int(path.stat().st_size)
    except FileNotFoundError:
        return provider, model, thinking_level
    except Exception:
        return provider, model, thinking_level

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
                typ = obj.get("type")
                if typ == "model_change":
                    raw_provider = obj.get("provider")
                    raw_model = obj.get("modelId")
                    if isinstance(raw_provider, str) and raw_provider.strip():
                        provider = raw_provider
                    if isinstance(raw_model, str) and raw_model.strip():
                        model = raw_model
                    continue
                if typ == "thinking_level_change":
                    raw_thinking = obj.get("thinkingLevel")
                    if isinstance(raw_thinking, str) and raw_thinking.strip():
                        thinking_level = raw_thinking
    except FileNotFoundError:
        return provider, model, thinking_level
    return provider, model, thinking_level
