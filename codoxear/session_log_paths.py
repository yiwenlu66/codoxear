from __future__ import annotations

import os
import re
from pathlib import Path


_SESSION_ID_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)


def session_id_from_rollout_path(log_path: Path) -> str | None:
    matches = _SESSION_ID_RE.findall(log_path.name)
    return matches[-1] if matches else None


def _is_codex_rollout_log_path(path: Path) -> bool:
    return path.name.startswith("rollout-") and path.suffix == ".jsonl"


def _is_pi_session_log_path(path: Path, *, sessions_dir: Path | None = None) -> bool:
    if path.suffix != ".jsonl":
        return False
    if sessions_dir is None:
        return "/.pi/agent/sessions/" in str(path).replace("\\", "/")
    try:
        path.resolve().relative_to(sessions_dir.resolve())
    except Exception:
        return False
    return True


def _is_cc_session_log_path(path: Path, *, sessions_dir: Path | None = None) -> bool:
    if path.suffix != ".jsonl":
        return False
    path_text = str(path).replace("\\", "/")
    if "/subagents/" in path_text:
        return False
    if path.name == "history.jsonl":
        return False
    if sessions_dir is None:
        return "/.claude/projects/" in path_text
    try:
        path.resolve().relative_to(sessions_dir.resolve())
    except Exception:
        return False
    return True


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
