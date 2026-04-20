from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any

from .agent_backend import get_agent_backend
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


def _default_pi_models_path() -> Path:
    return get_agent_backend("pi").home().joinpath("agent", "models.json")


def _default_pi_builtin_models_path() -> Path | None:
    workspace_root = Path(__file__).resolve().parents[2]
    candidate = workspace_root / "pi-mono" / "packages" / "ai" / "src" / "models.generated.ts"
    return candidate if candidate.is_file() else None


@functools.lru_cache(maxsize=8)
def _pi_builtin_context_windows(models_path_str: str, mtime_ns: int) -> dict[tuple[str, str], int]:
    text = Path(models_path_str).read_text(encoding="utf-8")
    blocks = re.finditer(r'"(?P<model_id>[^"]+)":\s*\{(?P<body>.*?)\}\s*satisfies\s+Model<', text, re.DOTALL)
    out: dict[tuple[str, str], int] = {}
    for match in blocks:
        body = match.group("body")
        provider_match = re.search(r'provider:\s*"(?P<provider>[^"]+)"', body)
        context_match = re.search(r'contextWindow:\s*(?P<context_window>\d+)', body)
        if provider_match is None or context_match is None:
            continue
        provider_name = provider_match.group("provider").strip()
        model_id = match.group("model_id").strip()
        context_window = int(context_match.group("context_window"))
        if not provider_name or not model_id or context_window <= 0:
            continue
        out[(provider_name, model_id)] = context_window
    return out


@functools.lru_cache(maxsize=8)
def _pi_context_windows(
    models_path_str: str | None,
    mtime_ns: int,
    builtin_models_path_str: str | None,
    builtin_mtime_ns: int,
) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    if builtin_models_path_str:
        out.update(_pi_builtin_context_windows(builtin_models_path_str, builtin_mtime_ns))
    if not models_path_str:
        return out
    path = Path(models_path_str)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return out
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return out
    for provider_name, provider_cfg in providers.items():
        if not isinstance(provider_name, str) or not isinstance(provider_cfg, dict):
            continue
        provider_key = provider_name.strip()
        if not provider_key:
            continue
        models = provider_cfg.get("models")
        if isinstance(models, list):
            for row in models:
                if not isinstance(row, dict):
                    continue
                model_id = row.get("id")
                context_window = row.get("contextWindow")
                if not isinstance(model_id, str) or not model_id.strip():
                    continue
                if not isinstance(context_window, int) or context_window <= 0:
                    continue
                out[(provider_key, model_id.strip())] = int(context_window)
        model_overrides = provider_cfg.get("modelOverrides")
        if not isinstance(model_overrides, dict):
            continue
        for model_id, override in model_overrides.items():
            if not isinstance(model_id, str) or not model_id.strip() or not isinstance(override, dict):
                continue
            context_window = override.get("contextWindow")
            if not isinstance(context_window, int) or context_window <= 0:
                continue
            out[(provider_key, model_id.strip())] = int(context_window)
    return out


def _fallback_context_window_for_model(index: dict[tuple[str, str], int], model: str) -> int | None:
    wanted = model.strip()
    if not wanted:
        return None
    matches = {ctx for (_provider, model_id), ctx in index.items() if model_id == wanted}
    if len(matches) == 1:
        return next(iter(matches))
    return None


def pi_model_context_window(
    provider: str | None,
    model: str | None,
    *,
    models_path: Path | None = None,
    builtin_models_path: Path | None = None,
) -> int | None:
    if not isinstance(model, str) or not model.strip():
        return None
    path: Path | None = _default_pi_models_path() if models_path is None else models_path
    builtin_path: Path | None = _default_pi_builtin_models_path() if builtin_models_path is None else builtin_models_path
    models_path_str: str | None = None
    models_mtime_ns = 0
    if path is not None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            path = None
        except Exception:
            path = None
        else:
            models_path_str = str(path.resolve())
            models_mtime_ns = int(stat.st_mtime_ns)
    builtin_path_str: str | None = None
    builtin_mtime_ns = 0
    if builtin_path is not None:
        try:
            builtin_stat = builtin_path.stat()
        except FileNotFoundError:
            builtin_path = None
        except Exception:
            builtin_path = None
        else:
            builtin_path_str = str(builtin_path.resolve())
            builtin_mtime_ns = int(builtin_stat.st_mtime_ns)
    try:
        index = _pi_context_windows(models_path_str, models_mtime_ns, builtin_path_str, builtin_mtime_ns)
    except Exception:
        return None
    if isinstance(provider, str) and provider.strip():
        direct = index.get((provider.strip(), model.strip()))
        if direct is not None:
            return direct
        return None
    return _fallback_context_window_for_model(index, model)


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
        raw_text = part.get("text")
        if isinstance(raw_text, str) and raw_text.strip():
            out.append(raw_text)
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


def pi_assistant_is_final_turn_end(obj: dict[str, Any]) -> bool:
    if obj.get("type") != "message":
        return False
    message = obj.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return False
    if not pi_assistant_text(obj):
        return False

    if pi_assistant_tool_use_count(obj) <= 0 and pi_assistant_thinking_count(obj) <= 0:
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


def pi_token_update(obj: dict[str, Any], *, models_path: Path | None = None) -> dict[str, Any] | None:
    if obj.get("type") != "message":
        return None
    message = obj.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
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
    return {
        "context_window": context_window,
        "tokens_in_context": total_tokens,
        "tokens_remaining": max(context_window - total_tokens, 0),
        "percent_remaining": _context_percent_remaining(tokens_in_context=total_tokens, context_window=context_window),
        "baseline_tokens": CONTEXT_WINDOW_BASELINE_TOKENS,
        "as_of": as_of,
    }


def _merge_pi_run_settings(
    provider: str | None,
    model: str | None,
    thinking_level: str | None,
    obj: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    typ = obj.get("type")
    if typ == "model_change":
        raw_provider = obj.get("provider")
        raw_model = obj.get("modelId")
        if isinstance(raw_provider, str) and raw_provider.strip():
            provider = raw_provider
        if isinstance(raw_model, str) and raw_model.strip():
            model = raw_model
    elif typ == "thinking_level_change":
        raw_thinking = obj.get("thinkingLevel")
        if isinstance(raw_thinking, str) and raw_thinking.strip():
            thinking_level = raw_thinking
    return provider, model, thinking_level


def _scan_pi_run_settings_range(
    path: Path,
    *,
    start: int,
    limit_bytes: int,
    provider: str | None,
    model: str | None,
    thinking_level: str | None,
) -> tuple[str | None, str | None, str | None]:
    if limit_bytes <= 0:
        return provider, model, thinking_level
    with path.open("rb") as f:
        if start > 0:
            f.seek(start)
            _ = f.readline()
        consumed = 0
        for raw in f:
            consumed += len(raw)
            if consumed > limit_bytes:
                break
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            provider, model, thinking_level = _merge_pi_run_settings(
                provider, model, thinking_level, obj
            )
    return provider, model, thinking_level


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

    head_scan_bytes = min(size, 256 * 1024)
    tail_scan_bytes = min(size, int(max_scan_bytes))
    try:
        provider, model, thinking_level = _scan_pi_run_settings_range(
            path,
            start=0,
            limit_bytes=head_scan_bytes,
            provider=provider,
            model=model,
            thinking_level=thinking_level,
        )
        if size > tail_scan_bytes:
            provider, model, thinking_level = _scan_pi_run_settings_range(
                path,
                start=max(0, size - tail_scan_bytes),
                limit_bytes=tail_scan_bytes,
                provider=provider,
                model=model,
                thinking_level=thinking_level,
            )
    except FileNotFoundError:
        return provider, model, thinking_level
    return provider, model, thinking_level
