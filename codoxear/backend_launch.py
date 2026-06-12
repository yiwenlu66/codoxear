from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .agent_backend import get_agent_backend
from .agent_backend import normalize_agent_backend

_BACKEND_HOME_ENV = {
    "codex": "CODEX_HOME",
    "pi": "PI_HOME",
    "cc": "CLAUDE_CONFIG_DIR",
}

_BACKEND_HOME_UNSET = {
    "codex": ("PI_HOME", "CLAUDE_CONFIG_DIR"),
    "pi": ("CODEX_HOME", "CLAUDE_CONFIG_DIR"),
    "cc": ("CODEX_HOME", "PI_HOME"),
}

_REQUEST_ENV_VARS = (
    "CODEX_WEB_MODEL_PROVIDER",
    "CODEX_WEB_PREFERRED_AUTH_METHOD",
    "CODEX_WEB_MODEL",
    "CODEX_WEB_REASONING_EFFORT",
    "CODEX_WEB_SERVICE_TIER",
    "CODEX_WEB_TRANSPORT",
    "CODEX_WEB_TMUX_SESSION",
    "CODEX_WEB_TMUX_WINDOW",
    "CODEX_WEB_LAUNCH_ID",
    "CODEX_WEB_SPAWN_NONCE",
    "CODEX_WEB_RESUME_SESSION_ID",
    "CODEX_WEB_RESUME_LOG_PATH",
)


def build_backend_args(
    *,
    agent_backend: str,
    spawn_cwd: Path,
    codex_trust_override: str,
    model_provider: str | None = None,
    preferred_auth_method: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
) -> list[str]:
    backend_name = normalize_agent_backend(agent_backend)
    if backend_name == "codex":
        args = [
            "-c",
            codex_trust_override,
            "-c",
            "check_for_update_on_startup=false",
            "--disable",
            "goals",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        if model is not None:
            args.extend(["--model", model])
        if reasoning_effort is not None:
            args.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
        if model_provider is not None:
            args.extend(["-c", f'model_provider="{model_provider}"'])
        if preferred_auth_method is not None:
            args.extend(["-c", f'preferred_auth_method="{preferred_auth_method}"'])
        if service_tier is not None:
            args.extend(["-c", f'service_tier="{service_tier}"'])
        return args
    if backend_name == "pi":
        if preferred_auth_method is not None:
            raise ValueError("preferred_auth_method is not supported for pi")
        if service_tier is not None:
            raise ValueError("service_tier is not supported for pi")
        args: list[str] = []
        if model_provider is not None:
            args.extend(["--provider", model_provider])
        if model is not None:
            args.extend(["--model", model])
        if reasoning_effort is not None:
            args.extend(["--thinking", reasoning_effort])
        return args
    if backend_name == "cc":
        if model_provider is not None:
            raise ValueError("model_provider is not supported for cc")
        if preferred_auth_method is not None:
            raise ValueError("preferred_auth_method is not supported for cc")
        if service_tier is not None:
            raise ValueError("service_tier is not supported for cc")
        args = ["--dangerously-skip-permissions"]
        if model is not None:
            args.extend(["--model", model])
        if reasoning_effort is not None:
            args.extend(["--effort", reasoning_effort])
        return args
    raise ValueError(f"unsupported agent_backend: {backend_name}")


def build_backend_resume_args(*, agent_backend: str, resume_id: str, resume_row: Mapping[str, Any] | None = None) -> list[str]:
    backend_name = normalize_agent_backend(agent_backend)
    if backend_name == "codex":
        return ["resume", resume_id]
    if backend_name == "pi":
        resume_target = str((resume_row or {}).get("log_path") or "").strip()
        return ["--session", resume_target or resume_id]
    if backend_name == "cc":
        return ["--resume", resume_id]
    raise ValueError(f"unsupported agent_backend: {backend_name}")


def apply_backend_environment(
    env: dict[str, str],
    *,
    agent_backend: str,
    homes: Mapping[str, str | Path],
    model_provider: str | None = None,
    preferred_auth_method: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
    resume_session_id: str | None = None,
) -> dict[str, str]:
    backend_name = normalize_agent_backend(agent_backend)
    home_env = _BACKEND_HOME_ENV.get(backend_name)
    if home_env is None:
        raise ValueError(f"unsupported agent_backend: {backend_name}")
    if backend_name not in homes:
        raise ValueError(f"missing home path for {backend_name}")

    env["CODEX_WEB_OWNER"] = "web"
    env["CODEX_WEB_AGENT_BACKEND"] = backend_name
    env.setdefault(home_env, str(homes[backend_name]))
    for key in _BACKEND_HOME_UNSET[backend_name]:
        env.pop(key, None)
    for key in _REQUEST_ENV_VARS:
        env.pop(key, None)

    if model_provider is not None:
        env["CODEX_WEB_MODEL_PROVIDER"] = model_provider
    if preferred_auth_method is not None:
        env["CODEX_WEB_PREFERRED_AUTH_METHOD"] = preferred_auth_method
    if model is not None:
        env["CODEX_WEB_MODEL"] = model
    if reasoning_effort is not None:
        env["CODEX_WEB_REASONING_EFFORT"] = reasoning_effort
    if service_tier is not None:
        env["CODEX_WEB_SERVICE_TIER"] = service_tier
    if resume_session_id is not None:
        env["CODEX_WEB_RESUME_SESSION_ID"] = resume_session_id
    return env


def build_tmux_inline_env(
    env: Mapping[str, str],
    *,
    agent_backend: str,
    tmux_session: str,
    tmux_window: str,
    launch_id: str,
    spawn_nonce: str,
    resume_session_id: str | None = None,
    model_provider: str | None = None,
    preferred_auth_method: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
    inherited_backend_bin: str | None = None,
) -> dict[str, str]:
    backend_name = normalize_agent_backend(agent_backend)
    home_env = _BACKEND_HOME_ENV.get(backend_name)
    if home_env is None:
        raise ValueError(f"unsupported agent_backend: {backend_name}")
    inline_env = {
        "CODEX_WEB_OWNER": "web",
        "CODEX_WEB_AGENT_BACKEND": backend_name,
        "CODEX_WEB_TRANSPORT": "tmux",
        "CODEX_WEB_TMUX_SESSION": tmux_session,
        "CODEX_WEB_TMUX_WINDOW": tmux_window,
        "CODEX_WEB_LAUNCH_ID": launch_id,
        "CODEX_WEB_SPAWN_NONCE": spawn_nonce,
        home_env: str(env[home_env]),
    }
    if resume_session_id is not None:
        inline_env["CODEX_WEB_RESUME_SESSION_ID"] = resume_session_id
    if model_provider is not None:
        inline_env["CODEX_WEB_MODEL_PROVIDER"] = model_provider
    if preferred_auth_method is not None:
        inline_env["CODEX_WEB_PREFERRED_AUTH_METHOD"] = preferred_auth_method
    if model is not None:
        inline_env["CODEX_WEB_MODEL"] = model
    if reasoning_effort is not None:
        inline_env["CODEX_WEB_REASONING_EFFORT"] = reasoning_effort
    if service_tier is not None:
        inline_env["CODEX_WEB_SERVICE_TIER"] = service_tier
    backend_bin_env_var = get_agent_backend(backend_name).bin_env_var
    if inherited_backend_bin is not None:
        inline_env[backend_bin_env_var] = inherited_backend_bin
    return inline_env


def tmux_unset_vars() -> list[str]:
    return [
        "CODEX_HOME",
        "PI_HOME",
        "CLAUDE_CONFIG_DIR",
        "CODEX_BIN",
        "PI_BIN",
        "CLAUDE_BIN",
        "CODEX_WEB_OWNER",
        "CODEX_WEB_AGENT_BACKEND",
        "CODEX_WEB_MODEL_PROVIDER",
        "CODEX_WEB_PREFERRED_AUTH_METHOD",
        "CODEX_WEB_MODEL",
        "CODEX_WEB_REASONING_EFFORT",
        "CODEX_WEB_SERVICE_TIER",
        "CODEX_WEB_TRANSPORT",
        "CODEX_WEB_TMUX_SESSION",
        "CODEX_WEB_TMUX_WINDOW",
        "CODEX_WEB_LAUNCH_ID",
        "CODEX_WEB_SPAWN_NONCE",
        "CODEX_WEB_RESUME_SESSION_ID",
        "CODEX_WEB_RESUME_LOG_PATH",
    ]
