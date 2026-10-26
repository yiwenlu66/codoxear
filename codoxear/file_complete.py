from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import os
from pathlib import Path

from .file_search import file_search_score
from .git_ops import path_has_surrogate_bytes


FILE_COMPLETE_DEFAULT_LIMIT = 50
FILE_COMPLETE_MAX_LIMIT = 120
FILE_COMPLETE_TIMEOUT_SECONDS = 1.0
_FILE_COMPLETE_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="file-complete")


@dataclass(frozen=True)
class FileCompletionResult:
    directory: Path | None
    entries: list[dict[str, object]]
    truncated: bool


def _resolve_directory(session_cwd: Path, dir_part: str) -> Path:
    if dir_part.startswith("~/") or dir_part == "~":
        return Path(dir_part).expanduser().resolve()
    if dir_part.startswith("/"):
        return Path(dir_part).resolve()
    return (session_cwd / dir_part).resolve()


def _complete_path_token(session_cwd: Path, token: str, limit: int) -> FileCompletionResult:
    # A token without a slash is a bare fragment in the session cwd. The
    # composer's `@` mention is sent in exactly that form (`@comp` -> `comp`,
    # `@` -> ``), and it is also the shape a user gets from completing the cwd
    # before any directory has been chosen.
    slash_index = token.rfind("/")
    dir_part = token[: slash_index + 1] if slash_index >= 0 else ""
    fragment = token[slash_index + 1 :] if slash_index >= 0 else token
    directory = _resolve_directory(session_cwd, dir_part)
    include_hidden = fragment.startswith(".")
    matches: list[tuple[int, str, bool]] = []
    try:
        with os.scandir(directory) as scan:
            for entry in scan:
                name = entry.name
                if name.startswith(".") and not include_hidden:
                    continue
                if path_has_surrogate_bytes(name):
                    continue
                try:
                    is_dir = entry.is_dir()
                except OSError:
                    is_dir = False
                score = 0 if not fragment else file_search_score(name, fragment)
                if score >= 0:
                    matches.append((score, name, is_dir))
    except (OSError, ValueError):
        return FileCompletionResult(directory=directory, entries=[], truncated=False)

    if fragment:
        matches.sort(key=lambda item: (-item[0], not item[2], item[1].casefold()))
    else:
        matches.sort(key=lambda item: (not item[2], item[1].casefold()))
    truncated = len(matches) > limit
    entries = [
        {
            "name": name,
            "completion": f"{dir_part}{name}{'/' if is_dir else ''}",
            "is_dir": is_dir,
        }
        for _score, name, is_dir in matches[:limit]
    ]
    return FileCompletionResult(directory=directory, entries=entries, truncated=truncated)


def complete_session_path(session_cwd: Path, *, token: str, limit: int = FILE_COMPLETE_DEFAULT_LIMIT) -> FileCompletionResult:
    """Complete one composer path token without allowing a slow mount to stall HTTP.

    A token may carry a directory part (`./src/al`, `../be`, `~/ga`, `/etc/ho`)
    or be a bare cwd fragment (`alpha`, or empty for the cwd itself).

    Resolution and directory enumeration both run in a bounded worker because
    either operation can block while a network mount is unavailable.  A path
    completion failure is intentionally an empty result: the composer can keep
    accepting ordinary text even when a directory is unreachable.
    """
    clamped_limit = max(1, min(int(limit), FILE_COMPLETE_MAX_LIMIT))
    future = _FILE_COMPLETE_EXECUTOR.submit(_complete_path_token, Path(session_cwd), str(token), clamped_limit)
    try:
        return future.result(timeout=FILE_COMPLETE_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        future.cancel()
    except (OSError, ValueError):
        pass
    return FileCompletionResult(directory=None, entries=[], truncated=False)
