from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

from .session_launch_plan import LaunchPlanDeps
from .session_launch_plan import LaunchPlanRequest
from .session_launch_plan import prepare_launch_plan
from .session_launcher import LaunchContextRequest
from .session_launcher import LaunchProcessDeps
from .session_launcher import LaunchProcessFailure
from .session_launcher import launch_broker_process
from .session_launcher import prepare_launch_process_context
from .session_model import Session


@dataclass(frozen=True)
class SessionWebLaunchCoordinator:
    resolve_dir_target: Callable[..., Path]
    create_git_worktree: Callable[[Path, str], Path]
    codex_trust_override_for_path: Callable[[Path], str]
    list_resume_candidates_for_cwd: Callable[[Path], list[dict[str, Any]]]
    live_session_for_resume_target: Callable[[str, dict[str, Any] | None], Session | None]
    load_env_file: Callable[[Path], dict[str, str]]
    environ: MutableMapping[str, str]
    dotenv_path: Path
    homes: Mapping[str, Path]
    python_executable: str
    tmux_session_name: str
    repo_root: Path
    record_launch_attempt: Callable[[dict[str, Any]], None]
    now: Callable[[], float]
    stderr: Any
    which_tmux: Callable[[str], str | None]
    run: Callable[..., Any]
    popen: Callable[..., Any]
    wait_or_raise: Callable[..., None]
    wait_for_spawned_broker_meta: Callable[..., dict[str, Any]]
    tmux_pane_snapshot: Callable[..., dict[str, Any]]
    drain_stream: Callable[..., str]
    launch_error: Callable[[dict[str, Any]], BaseException]

    def spawn_web_session(
        self,
        *,
        cwd: str,
        args: list[str] | None = None,
        agent_backend: str = "codex",
        resume_session_id: str | None = None,
        worktree_branch: str | None = None,
        model_provider: str | None = None,
        preferred_auth_method: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        create_in_tmux: bool = False,
    ) -> dict[str, Any]:
        launch_plan = prepare_launch_plan(
            LaunchPlanRequest(
                cwd=cwd,
                args=args,
                agent_backend=agent_backend,
                resume_session_id=resume_session_id,
                worktree_branch=worktree_branch,
                model_provider=model_provider,
                preferred_auth_method=preferred_auth_method,
                model=model,
                reasoning_effort=reasoning_effort,
                service_tier=service_tier,
                create_in_tmux=create_in_tmux,
            ),
            deps=LaunchPlanDeps(
                resolve_dir_target=self.resolve_dir_target,
                create_git_worktree=self.create_git_worktree,
                codex_trust_override_for_path=self.codex_trust_override_for_path,
                list_resume_candidates_for_cwd=self.list_resume_candidates_for_cwd,
                live_session_for_resume_target=self.live_session_for_resume_target,
                load_env_file=self.load_env_file,
                environ=self.environ,
                dotenv_path=self.dotenv_path,
                homes=self.homes,
                python_executable=self.python_executable,
            ),
        )

        launch_context = prepare_launch_process_context(
            LaunchContextRequest(
                argv=launch_plan.argv,
                env=launch_plan.env,
                agent_backend=launch_plan.backend_name,
                spawn_cwd=launch_plan.spawn_cwd,
                requested_cwd=launch_plan.requested_cwd,
                create_in_tmux=launch_plan.create_in_tmux,
                tmux_session_name=self.tmux_session_name,
                repo_root=self.repo_root,
                resume_session_id=launch_plan.resume_session_id,
                worktree_branch=launch_plan.worktree_branch,
                model_provider=launch_plan.model_provider,
                preferred_auth_method=launch_plan.preferred_auth_method,
                model=launch_plan.model,
                reasoning_effort=launch_plan.reasoning_effort,
                service_tier=launch_plan.service_tier,
            ),
            record_launch_attempt=self.record_launch_attempt,
            now=self.now,
            stderr=self.stderr,
        )

        process_deps = LaunchProcessDeps(
            which_tmux=self.which_tmux,
            run=self.run,
            popen=self.popen,
            wait_or_raise=self.wait_or_raise,
            wait_for_spawned_broker_meta=self.wait_for_spawned_broker_meta,
            tmux_pane_snapshot=self.tmux_pane_snapshot,
            drain_stream=self.drain_stream,
        )
        try:
            return launch_broker_process(launch_context.request, recorder=launch_context.recorder, deps=process_deps)
        except LaunchProcessFailure as exc:
            raise self.launch_error(exc.record) from exc
