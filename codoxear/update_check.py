"""Self-update check and settings for Codoxear.

The server runs from the deployed snapshot worktree (see AGENTS.md
"Deployment"), so its own working directory *is* the git checkout the running
code came from — the pipx-installed package lives elsewhere, which is why the
check anchors on the working directory and never on ``__file__``. This module
answers "is that checkout behind its remote?" and persists the two small files
the update UI needs: the auto-update preference and the outcome of the last
update attempt.

Both are read-only views of the checkout: no query here moves HEAD, writes the
working tree, or changes refs other than the remote-tracking refs refreshed by
the debounced ``git fetch``.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .json_state import atomic_write_json
from .system_dependencies import resolve_executable

_FETCH_DEBOUNCE_SECONDS = 60.0
_FETCH_TIMEOUT_SECONDS = 30.0
_QUERY_TIMEOUT_SECONDS = 5.0
_UPDATE_SETTINGS_NAME = "update_settings.json"
_UPDATE_STATUS_NAME = "update_status.json"
_UPDATE_LOG_NAME = "update.log"

_last_fetch_ts: float = 0.0


def update_settings_path(app_dir: Path) -> Path:
    return app_dir / _UPDATE_SETTINGS_NAME


def update_status_path(app_dir: Path) -> Path:
    return app_dir / _UPDATE_STATUS_NAME


def update_log_path(app_dir: Path) -> Path:
    return app_dir / _UPDATE_LOG_NAME


def _git(git_root: Path, args: list[str], *, timeout: float) -> str | None:
    """Run git in ``git_root``; return its stripped stdout, or None on any failure.

    Every query in this module is a single short line of stdout, so the failure
    channel (non-zero exit, unreadable cwd, missing git) collapses to None and
    each caller decides what the absence means.
    """
    binary = resolve_executable("git")
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary, *args],
            cwd=str(git_root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_root_from_working_dir(working_dir: Path) -> Path | None:
    """Find the git root of the server's WorkingDirectory."""
    top = _git(working_dir, ["rev-parse", "--show-toplevel"], timeout=_QUERY_TIMEOUT_SECONDS)
    return Path(top) if top else None


def _git_fetch(git_root: Path) -> bool:
    """Run ``git fetch origin``, debounced to at most one attempt per window.

    Returns True when the fetch ran and succeeded; False while the debounce
    holds or when it failed. The *attempt* stamps the window rather than the
    success: a fetch that hangs to its timeout must not be retried by the next
    poll, or every check would pay the same 30s stall again.
    """
    global _last_fetch_ts
    now = time.time()
    if now - _last_fetch_ts < _FETCH_DEBOUNCE_SECONDS:
        return False
    _last_fetch_ts = now
    binary = resolve_executable("git")
    if binary is None:
        return False
    try:
        result = subprocess.run(
            [binary, "fetch", "origin"],
            cwd=str(git_root),
            capture_output=True,
            timeout=_FETCH_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _resolve_remote_ref(git_root: Path) -> str | None:
    """Find the remote-tracking ref the local HEAD is compared against.

    ``@{upstream}`` is the configured answer, but the deployed snapshot is a
    *detached* worktree (see ``scripts/deploy.sh``) and so has no upstream at
    all; the remote's own default branch is the next best answer, with the two
    conventional branch names as the last resort.
    """
    upstream = _git(
        git_root,
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        timeout=_QUERY_TIMEOUT_SECONDS,
    )
    if upstream and upstream != "HEAD" and "/" in upstream:
        return upstream
    remote_head = _git(
        git_root,
        ["rev-parse", "--abbrev-ref", "refs/remotes/origin/HEAD"],
        timeout=_QUERY_TIMEOUT_SECONDS,
    )
    if remote_head and remote_head.startswith("origin/"):
        return remote_head
    for branch in ("origin/main", "origin/master"):
        if _git(git_root, ["rev-parse", "--verify", branch], timeout=_QUERY_TIMEOUT_SECONDS):
            return branch
    return None


def check_for_updates(working_dir: Path) -> dict[str, Any]:
    """Compare the checkout's HEAD against its remote-tracking branch.

    Returns ``available``, the two short commit ids, and the commit count
    behind; ``error`` is present only when the comparison cannot be made at
    all. A failed fetch is not an error — the comparison then runs against the
    remote-tracking ref as of the last successful fetch, so an offline server
    reports the last thing it knew rather than nothing.
    """
    git_root = _git_root_from_working_dir(working_dir)
    if git_root is None:
        error = "git is not available" if resolve_executable("git") is None else "not a git repository"
        return {"available": False, "error": error}

    _git_fetch(git_root)

    current_commit = _git(git_root, ["rev-parse", "HEAD"], timeout=_QUERY_TIMEOUT_SECONDS)
    remote_ref = _resolve_remote_ref(git_root)
    if remote_ref is None:
        result: dict[str, Any] = {"available": False, "error": "no remote branch"}
        if current_commit:
            result["current_commit"] = current_commit[:7]
        return result

    latest_commit = _git(git_root, ["rev-parse", remote_ref], timeout=_QUERY_TIMEOUT_SECONDS)
    if not current_commit or not latest_commit:
        return {"available": False, "error": "cannot read commits"}

    if current_commit == latest_commit:
        return {
            "available": False,
            "current_commit": current_commit[:7],
            "latest_commit": latest_commit[:7],
            "behind_count": 0,
        }

    behind = _git(git_root, ["rev-list", "--count", f"HEAD..{remote_ref}"], timeout=_QUERY_TIMEOUT_SECONDS)
    if behind is None or not behind.isdigit():
        return {
            "available": False,
            "error": "cannot count commits",
            "current_commit": current_commit[:7],
            "latest_commit": latest_commit[:7],
        }
    behind_count = int(behind)

    return {
        "available": behind_count > 0,
        "current_commit": current_commit[:7],
        "latest_commit": latest_commit[:7],
        "behind_count": behind_count,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object, treating missing, unreadable, corrupt, or non-object as empty."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # ValueError covers JSONDecodeError and UnicodeDecodeError
        return {}
    return data if isinstance(data, dict) else {}


def read_update_settings(app_dir: Path) -> dict[str, Any]:
    return _read_json_object(update_settings_path(app_dir))


def write_update_settings(app_dir: Path, settings: dict[str, Any]) -> None:
    atomic_write_json(update_settings_path(app_dir), settings)


def read_update_status(app_dir: Path) -> dict[str, Any]:
    return _read_json_object(update_status_path(app_dir))


def write_update_status(app_dir: Path, status: dict[str, Any]) -> None:
    atomic_write_json(update_status_path(app_dir), status)
