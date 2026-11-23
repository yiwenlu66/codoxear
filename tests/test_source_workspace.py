from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

import pytest

from codoxear import source_workspace


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _add_contribution_skills(path: Path) -> None:
    for name in ("codoxear-report-issue", "codoxear-contribute-fix"):
        skill = path / ".agents" / "skills" / name / "SKILL.md"
        skill.parent.mkdir(parents=True, exist_ok=True)
        skill.write_text(f"---\nname: {name}\n---\nUse the source checkout.\n", encoding="utf-8")
        projection = path / ".claude" / "skills" / name
        projection.parent.mkdir(parents=True, exist_ok=True)
        projection.symlink_to(f"../../.agents/skills/{name}", target_is_directory=True)


def _make_repo(path: Path, *, agents: bool = True, skills: bool = True) -> Path:
    path.mkdir(parents=True)
    if agents:
        (path / "AGENTS.md").write_text("# Codoxear source instructions\n", encoding="utf-8")
    (path / "README.md").write_text("fixture source\n", encoding="utf-8")
    (path / "pyproject.toml").write_text(
        "[project]\nname = \"codoxear\"\n\n"
        "[project.scripts]\n"
        "codoxear-server = \"codoxear.server:main\"\n"
        "codoxear-broker = \"codoxear.broker:main\"\n"
        "codoxear-sessiond = \"codoxear.sessiond:main\"\n",
        encoding="utf-8",
    )
    for module in ("server", "broker", "sessiond"):
        module_path = path / "codoxear" / f"{module}.py"
        module_path.parent.mkdir(exist_ok=True)
        module_path.write_text("def main():\n    pass\n", encoding="utf-8")
    if skills:
        _add_contribution_skills(path)
    _git("init", "-b", "main", cwd=path)
    _git("config", "user.email", "source-workspace@example.invalid", cwd=path)
    _git("config", "user.name", "Source Workspace Test", cwd=path)
    tracked = [
        "README.md", "pyproject.toml", "codoxear/server.py", "codoxear/broker.py", "codoxear/sessiond.py",
    ]
    if agents:
        tracked.append("AGENTS.md")
    if skills:
        tracked.extend([".agents/skills", ".claude/skills"])
    _git("add", "--", *tracked, cwd=path)
    _git("commit", "-m", "fixture source", cwd=path)
    return path


def _direct_url(path: Path, *, editable: bool) -> dict[str, Any]:
    return {"url": path.as_uri(), "dir_info": {"editable": editable}}


def _discover(app_dir: Path, source: Path, **kwargs: Any) -> dict[str, Any]:
    return source_workspace.discover_source_workspace(
        app_dir=app_dir,
        module_root=source,
        direct_url={},
        **kwargs,
    )


def test_editable_direct_url_accepts_existing_checkout_with_fork_origin(tmp_path: Path) -> None:
    source = _make_repo(tmp_path / "fork-checkout")
    _git("remote", "add", "origin", "git@github.com:someone/a-fork.git", cwd=source)
    app_dir = tmp_path / "runtime"

    result = source_workspace.discover_source_workspace(
        app_dir=app_dir,
        module_root=tmp_path / "not-source",
        direct_url=_direct_url(source, editable=True),
    )

    assert result == {
        "available": True,
        "path": str(source.resolve()),
        "can_setup": False,
        "reason": "source workspace is ready",
    }
    assert not app_dir.exists(), "discovery must not create runtime state"


def test_non_editable_direct_url_can_point_to_primary_source_checkout(tmp_path: Path) -> None:
    source = _make_repo(tmp_path / "original")
    result = source_workspace.discover_source_workspace(
        app_dir=tmp_path / "runtime",
        module_root=tmp_path / "not-source",
        direct_url=_direct_url(source, editable=False),
    )
    assert result["available"] is True
    assert result["path"] == str(source.resolve())


def test_non_editable_deploy_worktree_resolves_git_common_primary(tmp_path: Path) -> None:
    source = _make_repo(tmp_path / "original")
    deploy = tmp_path / "runtime" / "deploy"
    deploy.parent.mkdir()
    _git("worktree", "add", "--detach", str(deploy), cwd=source)

    result = source_workspace.discover_source_workspace(
        app_dir=deploy.parent,
        working_dir=deploy,
        module_root=tmp_path / "installed-package",
        direct_url=_direct_url(deploy, editable=False),
    )

    assert result["available"] is True
    assert result["path"] == str(source.resolve())
    assert result["path"] != str(deploy.resolve())


def test_detached_linked_module_root_is_not_mistaken_for_source(tmp_path: Path) -> None:
    source = _make_repo(tmp_path / "original")
    deploy = tmp_path / "custom-deploy"
    _git("worktree", "add", "--detach", str(deploy), cwd=source)

    result = source_workspace.discover_source_workspace(
        app_dir=tmp_path / "runtime",
        working_dir=deploy,
        module_root=deploy,
        direct_url={},
    )

    assert result["available"] is True
    assert result["path"] == str(source.resolve())
    assert result["path"] != str(deploy.resolve())


def test_managed_symlink_to_serving_snapshot_is_excluded(tmp_path: Path) -> None:
    source = _make_repo(tmp_path / "original")
    app_dir = tmp_path / "runtime"
    deploy = app_dir / "deploy"
    app_dir.mkdir()
    _git("worktree", "add", "--detach", str(deploy), cwd=source)
    (app_dir / "source-workspace").symlink_to(deploy, target_is_directory=True)

    result = source_workspace.discover_source_workspace(
        app_dir=app_dir,
        working_dir=deploy,
        module_root=tmp_path / "installed-package",
        direct_url={},
    )

    assert result["available"] is False
    assert result["path"] is None
    assert result["can_setup"] is False
    assert "refusing to replace" in result["reason"]


def test_installed_package_and_non_file_metadata_are_not_source(tmp_path: Path) -> None:
    package = tmp_path / "venv" / "site-packages"
    package.mkdir(parents=True)
    app_dir = tmp_path / "runtime"

    result = source_workspace.discover_source_workspace(
        app_dir=app_dir,
        module_root=package,
        direct_url={"url": "https://example.invalid/codoxear.git", "dir_info": {"editable": True}},
    )

    assert result == {
        "available": False,
        "path": None,
        "can_setup": True,
        "reason": "no editable source checkout found; set up a managed workspace",
    }
    assert not app_dir.exists()


def test_checkout_requires_nonempty_agents_and_repo_skill(tmp_path: Path) -> None:
    no_skills = _make_repo(tmp_path / "no-skills", skills=False)
    assert _discover(tmp_path / "runtime-a", no_skills)["available"] is False

    no_agents = _make_repo(tmp_path / "no-agents", agents=False)
    assert _discover(tmp_path / "runtime-b", no_agents)["available"] is False

    empty_skill = _make_repo(tmp_path / "empty-skill")
    (empty_skill / ".agents" / "skills" / "codoxear-report-issue" / "SKILL.md").write_text("", encoding="utf-8")
    assert _discover(tmp_path / "runtime-c", empty_skill)["available"] is False


def test_historical_or_unrelated_git_repositories_are_not_source(tmp_path: Path) -> None:
    historical = _make_repo(tmp_path / "historical", skills=False)
    old_skill = historical / ".codex" / "skills" / "codoxear-docker-test" / "SKILL.md"
    old_skill.parent.mkdir(parents=True)
    old_skill.write_text("Historical testing skill\n", encoding="utf-8")
    assert _discover(tmp_path / "runtime-old", historical)["available"] is False

    unrelated = _make_repo(tmp_path / "unrelated")
    pyproject = unrelated / "pyproject.toml"
    pyproject.write_text(pyproject.read_text(encoding="utf-8").replace('name = "codoxear"', 'name = "unrelated"'), encoding="utf-8")
    assert _discover(tmp_path / "runtime-unrelated", unrelated)["available"] is False

    managed_app = tmp_path / "managed-unrelated-runtime"
    managed = _make_repo(managed_app / "source-workspace")
    managed_pyproject = managed / "pyproject.toml"
    managed_pyproject.write_text(
        managed_pyproject.read_text(encoding="utf-8").replace('name = "codoxear"', 'name = "unrelated"'),
        encoding="utf-8",
    )
    managed_result = _discover(managed_app, tmp_path / "not-source")
    assert managed_result["available"] is False
    assert managed_result["can_setup"] is False

    wrong_projection = _make_repo(tmp_path / "wrong-projection")
    projection = wrong_projection / ".claude" / "skills" / "codoxear-report-issue"
    projection.unlink()
    projection.mkdir()
    (projection / "SKILL.md").write_text("A second, drifting body\n", encoding="utf-8")
    assert _discover(tmp_path / "runtime-projection", wrong_projection)["available"] is False


def test_managed_checkout_is_discovered_as_fallback(tmp_path: Path) -> None:
    app_dir = tmp_path / "runtime"
    managed = _make_repo(app_dir / "source-workspace")
    result = _discover(app_dir, tmp_path / "not-source")
    assert result["available"] is True
    assert result["path"] == str(managed.resolve())


def test_setup_clones_patched_canonical_remote_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _make_repo(tmp_path / "canonical-remote")
    app_dir = tmp_path / "runtime"
    monkeypatch.setattr(source_workspace, "CANONICAL_CLONE_URL", str(remote))
    real_run_git = source_workspace._run_git
    clone_calls: list[list[str]] = []

    def recording_run_git(args: list[str], **kwargs: Any):
        if args[0] == "clone":
            clone_calls.append(args)
        return real_run_git(args, **kwargs)

    monkeypatch.setattr(source_workspace, "_run_git", recording_run_git)
    kwargs = {"app_dir": app_dir, "module_root": tmp_path / "not-source", "direct_url": {}}

    before = source_workspace.discover_source_workspace(**kwargs)
    assert before["can_setup"] is True
    assert not app_dir.exists()

    first = source_workspace.setup_source_workspace(**kwargs)
    second = source_workspace.setup_source_workspace(**kwargs)

    assert first == second
    assert first["available"] is True
    assert first["path"] == str((app_dir / "source-workspace").resolve())
    assert len(clone_calls) == 1
    assert clone_calls[0][2] == str(remote)
    assert (app_dir / "source-workspace" / "AGENTS.md").is_file()
    assert (app_dir / "source-workspace" / ".agents" / "skills" / "codoxear-report-issue" / "SKILL.md").is_file()


def test_setup_returns_existing_source_without_cloning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_repo(tmp_path / "existing")

    real_run_git = source_workspace._run_git

    def forbidden_clone(args: list[str], **kwargs: Any):
        if args[0] == "clone":
            raise AssertionError("setup must not clone when source already exists")
        return real_run_git(args, **kwargs)

    monkeypatch.setattr(source_workspace, "_run_git", forbidden_clone)
    result = source_workspace.setup_source_workspace(
        app_dir=tmp_path / "runtime",
        module_root=tmp_path / "not-source",
        direct_url=_direct_url(source, editable=True),
        clone_url=str(tmp_path / "must-not-be-used"),
    )
    assert result["available"] is True
    assert result["path"] == str(source.resolve())
    assert not (tmp_path / "must-not-be-used").exists()


def test_setup_refuses_to_replace_occupied_managed_path(tmp_path: Path) -> None:
    app_dir = tmp_path / "runtime"
    managed = app_dir / "source-workspace"
    managed.mkdir(parents=True)
    sentinel = managed / "keep-me.txt"
    sentinel.write_text("existing user data\n", encoding="utf-8")

    with pytest.raises(source_workspace.SourceWorkspaceSetupError) as caught:
        source_workspace.setup_source_workspace(
            app_dir=app_dir,
            module_root=tmp_path / "not-source",
            direct_url={},
            clone_url=str(tmp_path / "unused"),
        )

    assert caught.value.status == 409
    assert "refusing to replace" in str(caught.value)
    assert sentinel.read_text(encoding="utf-8") == "existing user data\n"


def test_invalid_clone_is_cleaned_up_and_can_be_retried(tmp_path: Path) -> None:
    remote = _make_repo(tmp_path / "initially-invalid-remote", skills=False)
    app_dir = tmp_path / "runtime"
    kwargs = {
        "app_dir": app_dir,
        "module_root": tmp_path / "not-source",
        "direct_url": {},
        "clone_url": str(remote),
    }

    with pytest.raises(source_workspace.SourceWorkspaceSetupError) as caught:
        source_workspace.setup_source_workspace(**kwargs)
    assert caught.value.status == 502
    assert "not a usable" in str(caught.value)
    assert not os.path.lexists(app_dir / "source-workspace")

    _add_contribution_skills(remote)
    _git("add", "--", ".agents/skills", ".claude/skills", cwd=remote)
    _git("commit", "-m", "add required contribution skills", cwd=remote)
    retry = source_workspace.setup_source_workspace(**kwargs)
    assert retry["available"] is True
    assert retry["path"] == str((app_dir / "source-workspace").resolve())


def test_clone_timeout_has_explicit_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_run_git = source_workspace._run_git

    def timeout_on_clone(args: list[str], **kwargs: Any):
        if args[0] == "clone":
            raise subprocess.TimeoutExpired(cmd="git clone", timeout=1)
        return real_run_git(args, **kwargs)

    monkeypatch.setattr(source_workspace, "_run_git", timeout_on_clone)
    with pytest.raises(source_workspace.SourceWorkspaceSetupError) as caught:
        source_workspace.setup_source_workspace(
            app_dir=tmp_path / "runtime",
            module_root=tmp_path / "not-source",
            direct_url={},
            clone_url=str(tmp_path / "unused"),
        )
    assert caught.value.status == 504


def test_concurrent_setup_uses_one_clone_and_all_callers_get_ready_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _make_repo(tmp_path / "canonical-remote")
    app_dir = tmp_path / "runtime"
    real_run_git = source_workspace._run_git
    count_lock = threading.Lock()
    clone_calls = 0

    def slow_recording_clone(args: list[str], **kwargs: Any):
        nonlocal clone_calls
        if args[0] == "clone":
            with count_lock:
                clone_calls += 1
            time.sleep(0.15)
        return real_run_git(args, **kwargs)

    monkeypatch.setattr(source_workspace, "_run_git", slow_recording_clone)

    def setup_one(_index: int) -> dict[str, Any]:
        return source_workspace.setup_source_workspace(
            app_dir=app_dir,
            module_root=tmp_path / "not-source",
            direct_url={},
            clone_url=str(remote),
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(setup_one, range(6)))

    assert clone_calls == 1
    assert all(result["available"] for result in results)
    assert {result["path"] for result in results} == {str((app_dir / "source-workspace").resolve())}


def test_missing_git_disables_setup_without_creating_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(source_workspace.shutil, "which", lambda _name: None)
    app_dir = tmp_path / "runtime"
    result = _discover(app_dir, tmp_path / "not-source")
    assert result["available"] is False
    assert result["can_setup"] is False
    assert result["reason"] == "git is required to discover or set up a source workspace"
    assert not app_dir.exists()
