from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .agent_backend import AgentBackend
from .agent_backend import get_agent_backend


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
    return get_agent_backend(agent_backend).build_launch_args(
        spawn_cwd=spawn_cwd,
        codex_trust_override=codex_trust_override,
        model_provider=model_provider,
        preferred_auth_method=preferred_auth_method,
        model=model,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
    )


def build_backend_resume_args(*, agent_backend: str, resume_id: str, resume_row: Mapping[str, Any] | None = None) -> list[str]:
    return get_agent_backend(agent_backend).build_resume_args(resume_id=resume_id, resume_row=resume_row)


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
    return get_agent_backend(agent_backend).apply_launch_environment(
        env,
        homes=homes,
        model_provider=model_provider,
        preferred_auth_method=preferred_auth_method,
        model=model,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
        resume_session_id=resume_session_id,
    )


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
    return get_agent_backend(agent_backend).build_tmux_inline_env(
        env,
        tmux_session=tmux_session,
        tmux_window=tmux_window,
        launch_id=launch_id,
        spawn_nonce=spawn_nonce,
        resume_session_id=resume_session_id,
        model_provider=model_provider,
        preferred_auth_method=preferred_auth_method,
        model=model,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
        inherited_backend_bin=inherited_backend_bin,
    )


def tmux_unset_vars() -> list[str]:
    return AgentBackend.tmux_unset_vars()
