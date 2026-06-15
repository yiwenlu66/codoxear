from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterator

from .agent_backend import get_agent_backend

PI_DEFAULT_RESERVED_TOKENS = 16384
PI_MODEL_QUERY_TIMEOUT_SECONDS = 10.0
PI_MODEL_QUERY_ID = "codoxear-models"
PI_UNKNOWN_TOOL_CALL_ID_PREFIX = "__pi_unknown_tool_call__:"
PI_DUPLICATE_TOOL_CALL_ID_PREFIX = "__pi_duplicate_tool_call__:"


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
    if max_input_tokens <= 0:
        return 0
    remaining = max(max_input_tokens - used_input_context_tokens, 0)
    return int(round((remaining / max_input_tokens) * 100.0))


def _context_token_update(*, context_window: int, tokens_in_context: int, reserved_tokens: int, as_of: str | None = None) -> dict[str, Any]:
    normalized_reserved_tokens = min(max(int(reserved_tokens), 0), int(context_window))
    max_input_tokens = max(int(context_window) - normalized_reserved_tokens, 0)
    return {
        "context_window": int(context_window),
        "tokens_in_context": int(tokens_in_context),
        "tokens_remaining": max(max_input_tokens - int(tokens_in_context), 0),
        "percent_remaining": _context_percent_remaining(used_input_context_tokens=int(tokens_in_context), max_input_tokens=max_input_tokens),
        "reserved_tokens": normalized_reserved_tokens,
        "max_input_tokens": max_input_tokens,
        "as_of": as_of,
    }


def _default_pi_models_path() -> Path:
    return get_agent_backend("pi").home().joinpath("agent", "models.json")


def _default_pi_settings_path() -> Path:
    return get_agent_backend("pi").home().joinpath("agent", "settings.json")


def _context_windows_from_model_rows(rows: Any) -> dict[tuple[str, str], int]:
    if not isinstance(rows, list):
        return {}
    out: dict[tuple[str, str], int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        provider_name = row.get("provider")
        model_id = row.get("id")
        context_window = row.get("contextWindow")
        if not isinstance(provider_name, str) or not provider_name.strip():
            continue
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        if not isinstance(context_window, int) or context_window <= 0:
            continue
        out[(provider_name.strip(), model_id.strip())] = int(context_window)
    return out


@functools.lru_cache(maxsize=8)
def _pi_context_windows(models_path_str: str, mtime_ns: int) -> dict[tuple[str, str], int]:
    path = Path(models_path_str)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return {}
    rows: list[dict[str, Any]] = []
    for provider_name, provider_cfg in providers.items():
        if not isinstance(provider_name, str) or not isinstance(provider_cfg, dict):
            continue
        models = provider_cfg.get("models")
        if not isinstance(models, list):
            continue
        for row in models:
            if not isinstance(row, dict):
                continue
            row2 = dict(row)
            row2["provider"] = provider_name
            rows.append(row2)
    return _context_windows_from_model_rows(rows)


def _context_windows_from_models_file(path: Path) -> dict[tuple[str, str], int]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    try:
        return _pi_context_windows(str(path.resolve()), int(stat.st_mtime_ns))
    except Exception:
        return {}


def _file_mtime_ns(path: Path) -> int:
    try:
        return int(path.stat().st_mtime_ns)
    except FileNotFoundError:
        return -1
    except Exception:
        return -1


@functools.lru_cache(maxsize=8)
def _pi_rpc_context_windows(pi_executable: str, pi_mtime_ns: int, models_path_str: str, models_mtime_ns: int) -> dict[tuple[str, str], int]:
    request = json.dumps({"id": PI_MODEL_QUERY_ID, "type": "get_available_models"}) + "\n"
    env = dict(os.environ)
    env["PI_OFFLINE"] = "1"
    env.setdefault("PI_HOME", str(get_agent_backend("pi").home()))
    cmd = [
        pi_executable,
        "--mode",
        "rpc",
        "--no-session",
        "--no-tools",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "--offline",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=request,
            text=True,
            capture_output=True,
            timeout=PI_MODEL_QUERY_TIMEOUT_SECONDS,
            env=env,
            check=False,
        )
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}
    for line in proc.stdout.splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("id") != PI_MODEL_QUERY_ID or obj.get("command") != "get_available_models" or obj.get("success") is not True:
            continue
        data = obj.get("data")
        if not isinstance(data, dict):
            return {}
        return _context_windows_from_model_rows(data.get("models"))
    return {}


def _query_pi_context_windows(models_path: Path) -> dict[tuple[str, str], int]:
    configured_pi = get_agent_backend("pi").cli_bin()
    pi_executable = shutil.which(configured_pi)
    if not pi_executable:
        return {}
    pi_path = Path(pi_executable)
    return _pi_rpc_context_windows(
        str(pi_path),
        _file_mtime_ns(pi_path),
        str(models_path),
        _file_mtime_ns(models_path),
    )


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


@functools.lru_cache(maxsize=8)
def _pi_reserved_tokens(settings_path_str: str, mtime_ns: int) -> int:
    path = Path(settings_path_str)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return PI_DEFAULT_RESERVED_TOKENS
    if not isinstance(data, dict):
        return PI_DEFAULT_RESERVED_TOKENS
    compaction = data.get("compaction")
    if not isinstance(compaction, dict):
        return PI_DEFAULT_RESERVED_TOKENS
    reserve_tokens = compaction.get("reserveTokens")
    if not isinstance(reserve_tokens, int) or reserve_tokens < 0:
        return PI_DEFAULT_RESERVED_TOKENS
    return int(reserve_tokens)


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


def pi_assistant_is_final_turn_end(obj: dict[str, Any]) -> bool:
    if obj.get("type") != "message":
        return False
    message = obj.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
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
