from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .agent_backend import normalize_agent_backend
from .backend_launch import apply_backend_environment
from .backend_launch import build_backend_args
from .backend_launch import build_backend_resume_args


@dataclass(frozen=True)
class LaunchPlanRequest:
    cwd: str
    args: list[str] | None = None
    agent_backend: str = "codex"
    resume_session_id: str | None = None
    worktree_branch: str | None = None
    model_provider: str | None = None
    preferred_auth_method: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None
    create_in_tmux: bool = False


@dataclass(frozen=True)
class LaunchPlanDeps:
    resolve_dir_target: Callable[..., Path]
    create_git_worktree: Callable[[Path, str], Path]
    codex_trust_override_for_path: Callable[[Path], str]
    list_resume_candidates_for_cwd: Callable[..., list[dict[str, Any]]]
    live_session_for_resume_target: Callable[[str, dict[str, Any]], Any | None]
    load_env_file: Callable[[Path], Mapping[str, str]]
    environ: Mapping[str, str]
    dotenv_path: Path
    homes: Mapping[str, str | Path]
    python_executable: str


@dataclass(frozen=True)
class LaunchPlan:
    backend_name: str
    requested_cwd: str
    spawn_cwd: Path
    argv: list[str]
    env: dict[str, str]
    resume_session_id: str | None = None
    worktree_branch: str | None = None
    model_provider: str | None = None
    preferred_auth_method: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None
    create_in_tmux: bool = False


def prepare_launch_plan(request: LaunchPlanRequest, *, deps: LaunchPlanDeps) -> LaunchPlan:
    backend_name = normalize_agent_backend(request.agent_backend)
    cwd_path = deps.resolve_dir_target(request.cwd, field_name="cwd")
    if not cwd_path.exists():
        try:
            cwd_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            detail = e.strerror or str(e)
            raise ValueError(f"cwd could not be created: {cwd_path}: {detail}") from e
    if not cwd_path.is_dir():
        raise ValueError(f"cwd is not a directory: {cwd_path}")
    requested_cwd = str(cwd_path)
    if request.resume_session_id is not None and request.worktree_branch is not None:
        raise ValueError("worktree_branch cannot be used when resuming a session")
    spawn_cwd = cwd_path
    if request.worktree_branch is not None:
        spawn_cwd = deps.create_git_worktree(cwd_path, request.worktree_branch)

    argv = [deps.python_executable, "-m", "codoxear.broker", "--cwd", str(spawn_cwd), "--"]
    backend_args = build_backend_args(
        agent_backend=backend_name,
        spawn_cwd=spawn_cwd,
        codex_trust_override=deps.codex_trust_override_for_path(spawn_cwd),
        model_provider=request.model_provider,
        preferred_auth_method=request.preferred_auth_method,
        model=request.model,
        reasoning_effort=request.reasoning_effort,
        service_tier=request.service_tier,
    )
    if request.resume_session_id is not None:
        resume_id = str(request.resume_session_id).strip()
        if not resume_id:
            raise ValueError("resume_session_id must be a non-empty string")
        resume_row: dict[str, Any] | None = None
        for row in deps.list_resume_candidates_for_cwd(requested_cwd, agent_backend=backend_name, limit=1000):
            if row.get("session_id") == resume_id:
                resume_row = row
                break
        if resume_row is None:
            raise ValueError(f"resume session not found for cwd: {resume_id}")
        live_target = deps.live_session_for_resume_target(resume_id, resume_row)
        if live_target is not None:
            raise ValueError(
                "resume target is already live as "
                f"{live_target.session_id}; select that session instead of creating another session bound to the same transcript"
            )
        backend_args.extend(build_backend_resume_args(agent_backend=backend_name, resume_id=resume_id, resume_row=resume_row))
    backend_args.extend(request.args or [])
    argv.extend(backend_args)

    env = dict(deps.environ)
    if deps.dotenv_path.exists():
        for key, value in deps.load_env_file(deps.dotenv_path).items():
            env.setdefault(key, value)
    apply_backend_environment(
        env,
        agent_backend=backend_name,
        homes=deps.homes,
        model_provider=request.model_provider,
        preferred_auth_method=request.preferred_auth_method,
        model=request.model,
        reasoning_effort=request.reasoning_effort,
        service_tier=request.service_tier,
        resume_session_id=request.resume_session_id,
    )

    return LaunchPlan(
        backend_name=backend_name,
        requested_cwd=requested_cwd,
        spawn_cwd=spawn_cwd,
        argv=argv,
        env=env,
        resume_session_id=request.resume_session_id,
        worktree_branch=request.worktree_branch,
        model_provider=request.model_provider,
        preferred_auth_method=request.preferred_auth_method,
        model=request.model,
        reasoning_effort=request.reasoning_effort,
        service_tier=request.service_tier,
        create_in_tmux=request.create_in_tmux,
    )
