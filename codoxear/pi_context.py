from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .agent_backend import get_agent_backend

PI_DEFAULT_RESERVED_TOKENS = 16384
PI_MODEL_QUERY_TIMEOUT_SECONDS = 10.0
PI_MODEL_QUERY_ID = "codoxear-models"


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
