from __future__ import annotations

from pathlib import Path
import hashlib
import os
from typing import Any, Callable, Iterable

from .file_search import FILE_LIST_IGNORED_DIRS
from .file_view import ClientFileView
from .git_ops import path_json_text


def path_resolves_inside(path_obj: Path, root: Path) -> bool:
    try:
        path_obj.resolve().relative_to(root)
        return True
    except (OSError, ValueError):
        return False


def symlink_payload_view(path_obj: Path) -> ClientFileView:
    raw = os.readlink(path_obj).encode("utf-8", errors="surrogateescape")
    text = raw.decode("utf-8", errors="replace")
    return ClientFileView(
        kind="text",
        size=len(raw),
        text=text,
        editable=False,
        version=hashlib.sha256(raw).hexdigest(),
    )


def resolve_unique_bare_filename(search_root: Path, raw_path: str) -> Path | None:
    name = str(raw_path).strip()
    if not name or "/" in name or "\\" in name or "\x00" in name:
        return None
    if "." not in Path(name).name:
        return None
    root = search_root.resolve()
    match: Path | None = None
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", ".hg", ".svn", "__pycache__", "node_modules", "build", "dist"}]
        if name not in filenames:
            continue
        candidate = (Path(current_root) / name).resolve()
        if match is None:
            match = candidate
            continue
        if candidate != match:
            return None
    return match


def resolve_tracked_file_by_basename(
    session_id: str,
    raw_path: str,
    *,
    files_get: Callable[[str], Iterable[str]],
    expanduser_path: Callable[[Path], Path],
) -> Path | None:
    name = str(raw_path).strip()
    if not name or "/" in name or "\\" in name or "\x00" in name:
        return None
    try:
        tracked = files_get(session_id)
    except KeyError:
        return None
    match: Path | None = None
    for raw in tracked:
        candidate = expanduser_path(Path(raw)).resolve()
        if candidate.name != name:
            continue
        if match is None:
            match = candidate
            continue
        if candidate != match:
            return None
    return match


def list_session_relative_files(base: Path, *, expanduser_path: Callable[[Path], Path]) -> list[str]:
    root = expanduser_path(base)
    if not root.is_absolute():
        root = root.resolve()
    if not root.exists():
        raise FileNotFoundError("session cwd not found")
    if not root.is_dir():
        raise ValueError("session cwd is not a directory")
    out: list[str] = []

    def _onerror(err: OSError) -> None:
        raise err

    for current_root, dirnames, filenames in os.walk(root, topdown=True, onerror=_onerror, followlinks=False):
        dirnames[:] = [name for name in sorted(dirnames) if name not in FILE_LIST_IGNORED_DIRS]
        current_path = Path(current_root)
        for name in sorted(filenames):
            rel = (current_path / name).relative_to(root)
            # os.walk yields filenames decoded with surrogateescape; a non-UTF-8
            # name (raw byte 0xff -> lone surrogate) would make the response
            # body fail UTF-8 encoding. Serialize through the same surrogate-safe
            # display path codec the git path layer uses.
            out.append(path_json_text(rel.as_posix()))
    out.sort()
    return out


def resolve_git_client_file_view(
    *,
    session_id: str,
    raw_path: str,
    refresh_session_meta: Callable[[str], None],
    get_session: Callable[[str], Any | None],
    resolve_session_cwd: Callable[[str], Path],
    resolve_git_path: Callable[[Path, str], tuple[Path, Path, str]],
    read_client_file_view: Callable[[Path], ClientFileView],
) -> tuple[Path, str, ClientFileView]:
    if not session_id:
        raise ValueError("session_id required for git path")
    refresh_session_meta(session_id)
    session = get_session(session_id)
    if session is None:
        raise FileNotFoundError("unknown session")
    cwd = resolve_session_cwd(session.cwd)
    path_obj, repo_root, rel = resolve_git_path(cwd, raw_path)
    if path_resolves_inside(path_obj.parent, repo_root) and path_obj.is_symlink():
        return path_obj, rel, symlink_payload_view(path_obj)
    try:
        real = path_obj.resolve()
        real.relative_to(repo_root)
    except (OSError, ValueError) as exc:
        raise FileNotFoundError("file not found") from exc
    return real, rel, read_client_file_view(real)


def resolve_git_existing_regular_file(
    *,
    session_id: str,
    raw_path: str,
    refresh_session_meta: Callable[[str], None],
    get_session: Callable[[str], Any | None],
    resolve_session_cwd: Callable[[str], Path],
    resolve_git_path: Callable[[Path, str], tuple[Path, Path, str]],
    require_existing_file: Callable[[Path], Path],
) -> tuple[Path, str]:
    if not session_id:
        raise ValueError("session_id required for git path")
    refresh_session_meta(session_id)
    session = get_session(session_id)
    if session is None:
        raise FileNotFoundError("unknown session")
    cwd = resolve_session_cwd(session.cwd)
    path_obj, repo_root, rel = resolve_git_path(cwd, raw_path)
    try:
        real = path_obj.resolve()
        real.relative_to(repo_root)
    except (OSError, ValueError) as exc:
        raise FileNotFoundError("file not found") from exc
    return require_existing_file(real), rel


def resolve_client_file_path(
    *,
    session_id: str,
    raw_path: str,
    refresh_session_meta: Callable[[str], None],
    get_session: Callable[[str], Any | None],
    files_get: Callable[[str], Iterable[str]],
    expanduser_path: Callable[[Path], Path],
    resolve_session_cwd: Callable[[str], Path],
    run_git: Callable[..., str],
    git_timeout_s: float,
) -> Path:
    session: Any | None = None
    if session_id:
        refresh_session_meta(session_id)
        session = get_session(session_id)
        if session is None:
            raise FileNotFoundError("unknown session")
    path_obj = expanduser_path(Path(raw_path))
    if not path_obj.is_absolute():
        if session is not None:
            base = resolve_session_cwd(session.cwd)
            direct = (base / path_obj).resolve()
            if direct.exists():
                path_obj = direct
            else:
                tracked = resolve_tracked_file_by_basename(
                    session_id,
                    raw_path,
                    files_get=files_get,
                    expanduser_path=expanduser_path,
                )
                if tracked is not None:
                    path_obj = tracked
                    return path_obj
                try:
                    repo_root = Path(
                        run_git(base, ["rev-parse", "--show-toplevel"], timeout_s=git_timeout_s, max_bytes=64 * 1024).strip()
                    ).resolve()
                except RuntimeError:
                    repo_root = base.resolve()
                path_obj = resolve_unique_bare_filename(repo_root, raw_path) or direct
        else:
            path_obj = (Path.cwd() / path_obj).resolve()
    else:
        path_obj = path_obj.resolve()
    return path_obj


def describe_session_cwd(
    cwd: Path,
    *,
    git_repo_root: Callable[[Path], Path | None],
    current_git_branch: Callable[[Path], str | None],
) -> dict[str, Any]:
    exists = cwd.exists()
    if exists and not cwd.is_dir():
        raise ValueError(f"cwd is not a directory: {cwd}")
    repo_root = git_repo_root(cwd) if exists else None
    git_branch = (current_git_branch(cwd) or "") if exists else ""
    return {
        "cwd": str(cwd),
        "exists": exists,
        "will_create": not exists,
        "git_repo": repo_root is not None,
        "git_root": str(repo_root) if repo_root is not None else "",
        "git_branch": git_branch,
    }
