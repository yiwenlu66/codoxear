from __future__ import annotations

from pathlib import Path
import stat


def expanduser_path(path: Path) -> Path:
    try:
        return path.expanduser()
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc


def resolve_under(base: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("path required")
    if "\x00" in rel:
        raise ValueError("invalid path")
    path = Path(rel)
    if path.is_absolute():
        raise ValueError("path must be relative")
    resolved_base = base.resolve()
    resolved = (resolved_base / path).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError("path escapes session cwd") from exc
    return resolved


def resolve_session_cwd(raw_cwd: str) -> Path:
    if not isinstance(raw_cwd, str) or not raw_cwd.strip() or "\x00" in raw_cwd:
        raise ValueError("invalid session cwd")
    try:
        cwd = expanduser_path(Path(raw_cwd))
        if not cwd.is_absolute():
            cwd = cwd.resolve()
    except (OSError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    return cwd


def resolve_session_path(base: Path, raw_path: str) -> Path:
    if not isinstance(raw_path, str) or raw_path == "":
        raise ValueError("path required")
    if "\x00" in raw_path:
        raise ValueError("invalid path")
    path = Path(raw_path)
    if path.is_absolute():
        return expanduser_path(path).resolve()
    resolved_base = expanduser_path(base)
    if not resolved_base.is_absolute():
        resolved_base = resolved_base.resolve()
    return (resolved_base / path).resolve()


def require_existing_file(path: Path) -> Path:
    try:
        stat_result = path.stat()
    except FileNotFoundError:
        raise FileNotFoundError("file not found")
    except PermissionError:
        raise
    if not stat.S_ISREG(stat_result.st_mode):
        raise ValueError("path is not a file")
    return path


def resolve_existing_session_file(base: Path, raw_path: str) -> Path:
    return require_existing_file(resolve_session_path(base, raw_path))


def resolve_existing_absolute_file(raw_path: str) -> Path:
    return require_existing_file(expanduser_path(Path(raw_path)).resolve())
