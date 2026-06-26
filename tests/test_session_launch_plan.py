from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from codoxear.session_launch_plan import LaunchPlanDeps
from codoxear.session_launch_plan import LaunchPlanRequest
from codoxear.session_launch_plan import prepare_launch_plan


def _deps(**overrides) -> LaunchPlanDeps:
    values = {
        "resolve_dir_target": lambda raw, *, field_name: Path(raw).expanduser().resolve(),
        "create_git_worktree": lambda source, branch: source / branch.replace("/", "-"),
        "codex_trust_override_for_path": lambda path: f"projects={{ {json.dumps(str(path.resolve()))} = {{ trust_level = \"trusted\" }} }}",
        "list_resume_candidates_for_cwd": lambda *args, **kwargs: [],
        "live_session_for_resume_target": lambda _resume_id, _resume_row: None,
        "load_env_file": lambda _path: {},
        "environ": {},
        "dotenv_path": Path("/tmp/codoxear-missing-dotenv"),
        "homes": {"codex": "/codex-home", "pi": "/pi-home", "cc": "/claude-home"},
        "python_executable": "/python",
    }
    values.update(overrides)
    return LaunchPlanDeps(**values)


def test_session_launch_plan_import_does_not_load_server() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import sys; import codoxear.session_launch_plan; raise SystemExit('codoxear.server' in sys.modules)"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_prepare_launch_plan_creates_missing_cwd_and_builds_codex_contract() -> None:
    with TemporaryDirectory() as td:
        target = Path(td) / "new" / "session"
        plan = prepare_launch_plan(
            LaunchPlanRequest(
                cwd=str(target),
                args=["--search"],
                model_provider="bytecat",
                preferred_auth_method="apikey",
                model="gpt-5.4",
                reasoning_effort="xhigh",
                service_tier="fast",
            ),
            deps=_deps(environ={"CODEX_HOME": "/existing-codex"}),
        )
        assert target.is_dir()

    trust_override = f'projects={{ {json.dumps(str(target.resolve()))} = {{ trust_level = "trusted" }} }}'
    assert plan.backend_name == "codex"
    assert plan.requested_cwd == str(target.resolve())
    assert plan.spawn_cwd == target.resolve()
    assert plan.argv == [
        "/python",
        "-m",
        "codoxear.broker",
        "--cwd",
        str(target.resolve()),
        "--",
        "-c",
        trust_override,
        "-c",
        "check_for_update_on_startup=false",
        "--disable",
        "goals",
        "--dangerously-bypass-approvals-and-sandbox",
        "--model",
        "gpt-5.4",
        "-c",
        'model_reasoning_effort="xhigh"',
        "-c",
        'model_provider="bytecat"',
        "-c",
        'preferred_auth_method="apikey"',
        "-c",
        'service_tier="fast"',
        "--search",
    ]
    assert plan.env["CODEX_WEB_OWNER"] == "web"
    assert plan.env["CODEX_WEB_AGENT_BACKEND"] == "codex"
    assert plan.env["CODEX_HOME"] == "/existing-codex"
    assert plan.env["CODEX_WEB_MODEL_PROVIDER"] == "bytecat"
    assert plan.env["CODEX_WEB_PREFERRED_AUTH_METHOD"] == "apikey"
    assert plan.env["CODEX_WEB_MODEL"] == "gpt-5.4"
    assert plan.env["CODEX_WEB_REASONING_EFFORT"] == "xhigh"
    assert plan.env["CODEX_WEB_SERVICE_TIER"] == "fast"


def test_prepare_launch_plan_dotenv_only_fills_missing_environment_values() -> None:
    with TemporaryDirectory() as td:
        dotenv = Path(td) / ".env"
        dotenv.write_text("ignored", encoding="utf-8")
        plan = prepare_launch_plan(
            LaunchPlanRequest(cwd=td, agent_backend="pi", model_provider="macaron", model="gpt-5.4", reasoning_effort="medium"),
            deps=_deps(
                dotenv_path=dotenv,
                environ={"PI_HOME": "/existing-pi", "CODEX_WEB_TRANSPORT": "tmux", "STALE": "kept"},
                load_env_file=lambda _path: {"PI_HOME": "/dotenv-pi", "NEW_FROM_DOTENV": "yes"},
            ),
        )

    assert plan.argv == ["/python", "-m", "codoxear.broker", "--cwd", str(Path(td).resolve()), "--", "--provider", "macaron", "--model", "gpt-5.4", "--thinking", "medium"]
    assert plan.env["PI_HOME"] == "/existing-pi"
    assert plan.env["NEW_FROM_DOTENV"] == "yes"
    assert plan.env["STALE"] == "kept"
    assert "CODEX_WEB_TRANSPORT" not in plan.env
    assert "CODEX_HOME" not in plan.env


def test_prepare_launch_plan_resume_uses_stripped_id_for_args_but_raw_id_for_env() -> None:
    seen_candidates_args: list[tuple] = []

    def candidates(*args, **kwargs):
        seen_candidates_args.append((args, kwargs))
        return [{"session_id": "resume-a", "log_path": "/tmp/pi-resume.jsonl"}]

    with TemporaryDirectory() as td:
        plan = prepare_launch_plan(
            LaunchPlanRequest(cwd=td, agent_backend="pi", resume_session_id=" resume-a "),
            deps=_deps(list_resume_candidates_for_cwd=candidates),
        )

    assert seen_candidates_args == [((str(Path(td).resolve()),), {"agent_backend": "pi", "limit": 1000})]
    assert plan.argv == ["/python", "-m", "codoxear.broker", "--cwd", str(Path(td).resolve()), "--", "--session", "/tmp/pi-resume.jsonl"]
    assert plan.env["CODEX_WEB_RESUME_SESSION_ID"] == " resume-a "
    assert plan.resume_session_id == " resume-a "


def test_prepare_launch_plan_rejects_live_resume_target() -> None:
    with TemporaryDirectory() as td:
        with pytest.raises(ValueError, match="resume target is already live as live-row"):
            prepare_launch_plan(
                LaunchPlanRequest(cwd=td, resume_session_id="resume-a"),
                deps=_deps(
                    list_resume_candidates_for_cwd=lambda *args, **kwargs: [{"session_id": "resume-a"}],
                    live_session_for_resume_target=lambda _resume_id, _resume_row: SimpleNamespace(session_id="live-row"),
                ),
            )


def test_prepare_launch_plan_worktree_is_spawn_cwd_and_incompatible_with_resume() -> None:
    with TemporaryDirectory() as td:
        worktree = Path(td) / "repo-worktree"
        plan = prepare_launch_plan(
            LaunchPlanRequest(cwd=td, worktree_branch="feature/test"),
            deps=_deps(create_git_worktree=lambda _source, _branch: worktree),
        )
        with pytest.raises(ValueError, match="worktree_branch cannot be used when resuming a session"):
            prepare_launch_plan(
                LaunchPlanRequest(cwd=td, resume_session_id="resume-a", worktree_branch="feature/test"),
                deps=_deps(),
            )

    trust_override = f'projects={{ {json.dumps(str(worktree.resolve()))} = {{ trust_level = "trusted" }} }}'
    assert plan.spawn_cwd == worktree
    assert plan.worktree_branch == "feature/test"
    assert plan.argv[:6] == ["/python", "-m", "codoxear.broker", "--cwd", str(worktree), "--"]
    assert plan.argv[6:8] == ["-c", trust_override]
