"""Locate or explicitly provision an editable Codoxear source checkout.

The running Python package is not itself a source workspace.  In particular,
the normal deployment installs a detached, non-editable worktree into pipx.  Its
``direct_url.json`` nevertheless identifies the deploy worktree, whose Git
common directory leads back to the user's primary checkout.

Discovery is read-only.  Only :func:`setup_source_workspace` creates anything,
and it clones the canonical repository into one managed path without replacing
an existing filesystem entry.
"""

from __future__ import annotations

import fcntl
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping
import urllib.parse
import urllib.request


CANONICAL_CLONE_URL = "https://github.com/yiwenlu66/codoxear"
MANAGED_DIR_NAME = "source-workspace"
SETUP_LOCK_NAME = ".source-workspace.setup.lock"
CLONE_TIMEOUT_SECONDS = 120.0
GIT_TIMEOUT_SECONDS = 10.0

_READY_REASON = "source workspace is ready"
_NOT_FOUND_REASON = "no editable source checkout found; set up a managed workspace"
_OCCUPIED_REASON = "source-workspace exists but is not a usable Codoxear checkout; refusing to replace it"
_GIT_REQUIRED_REASON = "git is required to discover or set up a source workspace"

_CONTRIBUTION_SKILLS = ("codoxear-report-issue", "codoxear-contribute-fix")
_EXPECTED_ENTRYPOINTS = {
    "codoxear-server": ("codoxear.server:main", "codoxear/server.py"),
    "codoxear-broker": ("codoxear.broker:main", "codoxear/broker.py"),
    "codoxear-sessiond": ("codoxear.sessiond:main", "codoxear/sessiond.py"),
}


class SourceWorkspaceSetupError(Exception):
    """A setup failure with an HTTP status safe to expose to the user."""

    def __init__(self, message: str, *, status: int = 500) -> None:
        super().__init__(message)
        self.status = status


def _ready(path: Path) -> dict[str, Any]:
    return {"available": True, "path": str(path), "can_setup": False, "reason": _READY_REASON}


def _unavailable(*, can_setup: bool, reason: str) -> dict[str, Any]:
    return {"available": False, "path": None, "can_setup": can_setup, "reason": reason}


def managed_workspace_path(app_dir: Path) -> Path:
    return Path(app_dir) / MANAGED_DIR_NAME


def _installed_direct_url() -> Mapping[str, Any] | None:
    """Return this installation's PEP 610 direct-URL metadata, if present."""
    try:
        distribution = metadata.distribution("codoxear")
        raw = distribution.read_text("direct_url.json")
        value = json.loads(raw) if raw else None
    except (metadata.PackageNotFoundError, OSError, UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _direct_url_checkout(value: Mapping[str, Any] | None) -> tuple[Path, bool] | None:
    if not isinstance(value, Mapping):
        return None
    url = value.get("url")
    dir_info = value.get("dir_info")
    if not isinstance(url, str) or not isinstance(dir_info, Mapping):
        return None
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return None
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        return None
    if parsed.params or parsed.query or parsed.fragment:
        return None
    try:
        path = Path(urllib.request.url2pathname(parsed.path))
    except (TypeError, ValueError):
        return None
    if not path.is_absolute():
        return None
    return path, dir_info.get("editable") is True


def _run_git(args: list[str], *, timeout: float = GIT_TIMEOUT_SECONDS) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        ["git", *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def _git_checkout(path: Path) -> tuple[Path, Path, Path, bool] | None:
    """Return (worktree root, Git dir, Git common dir, detached HEAD)."""
    try:
        root = path.resolve(strict=True)
    except OSError:
        return None
    if not root.is_dir():
        return None
    try:
        result = _run_git([
            "-C", str(root), "rev-parse", "--path-format=absolute",
            "--show-toplevel", "--git-dir", "--git-common-dir", "--abbrev-ref", "HEAD",
        ])
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    if len(lines) != 4:
        return None
    try:
        top, git_dir, common_dir = (Path(line).resolve(strict=True) for line in lines[:3])
    except OSError:
        return None
    if top != root:
        return None
    return top, git_dir, common_dir, lines[3] == "HEAD"


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _toml_section(document: str, name: str) -> str | None:
    match = re.search(
        rf"(?ms)^\[{re.escape(name)}\][ \t]*\r?\n(.*?)(?=^\[|\Z)",
        document,
    )
    return match.group(1) if match else None


def _toml_string(section: str, key: str) -> str | None:
    match = re.search(
        rf"(?m)^[ \t]*{re.escape(key)}[ \t]*=[ \t]*(['\"])(.*?)\1[ \t]*(?:#.*)?$",
        section,
    )
    return match.group(2) if match else None


def _has_codoxear_structure(root: Path) -> bool:
    try:
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    project = _toml_section(pyproject, "project")
    scripts = _toml_section(pyproject, "project.scripts")
    if project is None or scripts is None or _toml_string(project, "name") != "codoxear":
        return False
    for script, (entrypoint, module_path) in _EXPECTED_ENTRYPOINTS.items():
        if _toml_string(scripts, script) != entrypoint or not _nonempty_file(root / module_path):
            return False
    return True


def _has_contribution_skills(root: Path) -> bool:
    for name in _CONTRIBUTION_SKILLS:
        body = root / ".agents" / "skills" / name / "SKILL.md"
        projection = root / ".claude" / "skills" / name
        # The .agents copy is the one real body. Claude's projection must be a
        # symlink to that same body, not a second copy that can drift.
        if body.is_symlink() or not _nonempty_file(body) or not projection.is_symlink():
            return False
        try:
            if not os.path.samefile(body, projection / "SKILL.md"):
                return False
        except OSError:
            return False
    return True


def _usable_checkout(path: Path) -> Path | None:
    checkout = _git_checkout(path)
    if checkout is None:
        return None
    root = checkout[0]
    if not _nonempty_file(root / "AGENTS.md"):
        return None
    if not _has_codoxear_structure(root) or not _has_contribution_skills(root):
        return None
    return root


def _primary_checkout(common_dir: Path) -> Path | None:
    # A normal repository's common dir is <primary-checkout>/.git.  A bare or
    # custom Git dir has no inferred editable root and is simply not a source.
    if common_dir.name != ".git":
        return None
    return common_dir.parent


def _is_serving_snapshot(
    checkout: tuple[Path, Path, Path, bool],
    *,
    app_dir: Path,
    working_dir: Path | None,
) -> bool:
    root, git_dir, common_dir, detached = checkout
    if not detached or git_dir == common_dir:
        return False
    if root == (app_dir / "deploy").resolve():
        return True
    return working_dir is not None and root == Path(working_dir).resolve()


def discover_source_workspace(
    *,
    app_dir: Path,
    working_dir: Path | None = None,
    module_root: Path | None = None,
    direct_url: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Discover an existing source checkout without changing the filesystem.

    ``direct_url=None`` reads the installed distribution's metadata at call
    time.  Tests and embedders can pass a decoded mapping explicitly.
    """
    if shutil.which("git") is None:
        return _unavailable(can_setup=False, reason=_GIT_REQUIRED_REASON)

    app_dir = Path(app_dir).resolve()
    managed = managed_workspace_path(app_dir)
    candidates: list[Path] = []
    installed = _direct_url_checkout(_installed_direct_url() if direct_url is None else direct_url)
    if installed is not None:
        installed_path, editable = installed
        checkout = _git_checkout(installed_path)
        if checkout is not None:
            root, git_dir, common_dir, _detached = checkout
            if _is_serving_snapshot(checkout, app_dir=app_dir, working_dir=working_dir):
                primary = _primary_checkout(common_dir)
                if primary is not None and primary != root:
                    candidates.append(primary)
            elif editable or git_dir == common_dir:
                candidates.append(root)
            else:
                # A non-editable install from any linked worktree is a snapshot
                # install, even when it is not the currently serving worktree.
                primary = _primary_checkout(common_dir)
                if primary is not None and primary != root:
                    candidates.append(primary)

    module_candidate = module_root or Path(__file__).resolve().parent.parent
    module_checkout = _git_checkout(module_candidate)
    if module_checkout is not None:
        root, _git_dir, common_dir, _detached = module_checkout
        if _is_serving_snapshot(module_checkout, app_dir=app_dir, working_dir=working_dir):
            primary = _primary_checkout(common_dir)
            if primary is not None:
                candidates.append(primary)
        else:
            candidates.append(root)
    else:
        candidates.append(module_candidate)
    candidates.append(managed)

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        checkout = _git_checkout(candidate)
        if checkout is not None and _is_serving_snapshot(
            checkout,
            app_dir=app_dir,
            working_dir=working_dir,
        ):
            continue
        usable = _usable_checkout(candidate)
        if usable is not None:
            return _ready(usable)

    if os.path.lexists(managed):
        return _unavailable(can_setup=False, reason=_OCCUPIED_REASON)
    return _unavailable(can_setup=True, reason=_NOT_FOUND_REASON)


def _discard_failed_new_checkout(path: Path) -> str:
    """Best-effort cleanup for a path created by this failed clone only.

    The caller proves the managed path did not exist before invoking Git.  An
    unexpected non-directory/symlink is left alone.  A non-empty return value is
    the actionable recovery note appended to the setup error.
    """
    if not os.path.lexists(path):
        return ""
    recovery = f"partial checkout remains at {path}; move it aside before retrying"
    if path.is_symlink() or not path.is_dir():
        return recovery
    try:
        shutil.rmtree(path)
    except OSError:
        return recovery
    return ""


def setup_source_workspace(
    *,
    app_dir: Path,
    working_dir: Path | None = None,
    module_root: Path | None = None,
    direct_url: Mapping[str, Any] | None = None,
    clone_url: str | None = None,
) -> dict[str, Any]:
    """Explicitly provision the managed checkout, or return an existing one."""
    clone_url = clone_url or CANONICAL_CLONE_URL
    initial = discover_source_workspace(
        app_dir=app_dir,
        working_dir=working_dir,
        module_root=module_root,
        direct_url=direct_url,
    )
    if initial["available"]:
        return initial
    if not initial["can_setup"]:
        raise SourceWorkspaceSetupError(initial["reason"], status=409)

    app_dir = Path(app_dir)
    managed = managed_workspace_path(app_dir)
    try:
        app_dir.mkdir(parents=True, exist_ok=True)
        lock_path = app_dir / SETUP_LOCK_NAME
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            # Another request/process may have completed setup while we waited.
            current = discover_source_workspace(
                app_dir=app_dir,
                working_dir=working_dir,
                module_root=module_root,
                direct_url=direct_url,
            )
            if current["available"]:
                return current
            if os.path.lexists(managed):
                raise SourceWorkspaceSetupError(_OCCUPIED_REASON, status=409)
            try:
                try:
                    result = _run_git(
                        ["clone", "--", clone_url, str(managed)],
                        timeout=CLONE_TIMEOUT_SECONDS,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise SourceWorkspaceSetupError(
                        "timed out while cloning the Codoxear source workspace",
                        status=504,
                    ) from exc
                except OSError as exc:
                    raise SourceWorkspaceSetupError(
                        f"could not run git to set up the source workspace: {exc}",
                    ) from exc
                if result.returncode != 0:
                    detail = (result.stderr or result.stdout).strip()
                    message = "could not clone the canonical Codoxear source workspace"
                    if detail:
                        message = f"{message}: {detail[-2000:]}"
                    raise SourceWorkspaceSetupError(message, status=502)

                usable = _usable_checkout(managed)
                if usable is None:
                    raise SourceWorkspaceSetupError(
                        "cloned repository is not a usable Codoxear source checkout",
                        status=502,
                    )
                return _ready(usable)
            except SourceWorkspaceSetupError as exc:
                recovery = _discard_failed_new_checkout(managed)
                if recovery:
                    raise SourceWorkspaceSetupError(f"{exc}; {recovery}", status=exc.status) from exc
                raise
    except SourceWorkspaceSetupError:
        raise
    except OSError as exc:
        raise SourceWorkspaceSetupError(f"could not prepare the source workspace: {exc}") from exc
