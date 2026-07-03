from __future__ import annotations

import os
from pathlib import Path

from .agent_backend import CC_BACKEND
from .agent_backend import CODEX_BACKEND
from .agent_backend import PI_BACKEND


def session_id_from_rollout_path(log_path: Path) -> str | None:
    return CODEX_BACKEND.session_id_from_log_path(log_path)


def _is_codex_rollout_log_path(path: Path) -> bool:
    return CODEX_BACKEND.is_session_log_path(path)


def _is_pi_session_log_path(path: Path, *, sessions_dir: Path | None = None) -> bool:
    return PI_BACKEND.is_session_log_path(path, sessions_dir=sessions_dir)


def _is_cc_session_log_path(path: Path, *, sessions_dir: Path | None = None) -> bool:
    return CC_BACKEND.is_session_log_path(path, sessions_dir=sessions_dir)


def _paths_match(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:
        try:
            return a.absolute() == b.absolute()
        except Exception:
            return str(a) == str(b)


def _path_in_set(path: Path, paths: set[Path]) -> bool:
    for candidate in paths:
        if _paths_match(path, candidate):
            return True
    return False


def _payload_cwd_matches(payload_cwd: object, cwd: str) -> bool:
    if not isinstance(payload_cwd, str):
        return False
    if payload_cwd == cwd:
        return True
    payload_path = Path(payload_cwd)
    cwd_path = Path(cwd)
    if not (payload_path.is_absolute() and cwd_path.is_absolute()):
        return False
    try:
        return bool(os.path.samefile(payload_path, cwd_path))
    except Exception:
        return False
