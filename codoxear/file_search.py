from __future__ import annotations

import heapq
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .git_ops import path_json_text

FILE_SEARCH_LIMIT = int(os.environ.get("CODEX_WEB_FILE_SEARCH_LIMIT", "120"))
FILE_SEARCH_TIMEOUT_SECONDS = float(os.environ.get("CODEX_WEB_FILE_SEARCH_TIMEOUT_SECONDS", "0.75"))
FILE_SEARCH_MAX_CANDIDATES = int(os.environ.get("CODEX_WEB_FILE_SEARCH_MAX_CANDIDATES", "200000"))
FILE_SEARCH_GIT_ROOT_TIMEOUT_SECONDS = float(os.environ.get("CODEX_WEB_GIT_DIFF_TIMEOUT_SECONDS", "4.0"))
FILE_LIST_IGNORED_DIRS = frozenset({".git", ".hg", ".mypy_cache", ".pytest_cache", ".svn", "__pycache__", "build", "dist", "node_modules", "venv", ".venv"})

GitRootFunc = Callable[[Path], Path | None]


def file_search_score(candidate: str, query: str) -> int:
    text = str(candidate or "")
    raw = str(query or "").strip().lower()
    if not raw:
        return 0
    lower = text.lower()
    if lower == raw:
        return 12000
    base = Path(text).name.lower()
    if base == raw:
        return 10000
    total = 0
    for token in [part for part in raw.split() if part]:
        exact_idx = lower.find(token)
        if exact_idx >= 0:
            prev = lower[exact_idx - 1] if exact_idx > 0 else ""
            boundary_bonus = 24 if (not prev or prev in "/._-") else 0
            base_idx = base.find(token)
            total += 240 - exact_idx * 2 + boundary_bonus + (44 - base_idx if base_idx >= 0 else 0)
            continue
        pos = -1
        first = -1
        last = -1
        consecutive = 0
        boundaries = 0
        for ch in token:
            pos = lower.find(ch, pos + 1)
            if pos < 0:
                return -1
            if first < 0:
                first = pos
            if last >= 0 and pos == last + 1:
                consecutive += 1
            if pos == 0 or lower[pos - 1] in "/._-":
                boundaries += 1
            last = pos
        span = last - first + 1
        total += 120 - first - max(0, span - len(token)) * 4 + consecutive * 10 + boundaries * 8
    return total


def _push_file_search_match(heap: list[tuple[int, str]], *, path: str, score: int, limit: int) -> None:
    item = (score, path)
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    if item > heap[0]:
        heapq.heapreplace(heap, item)


def _finish_file_search(heap: list[tuple[int, str]], *, mode: str, query: str, scanned: int, truncated: bool) -> dict[str, Any]:
    matches = [{"path": path, "score": score} for score, path in sorted(heap, key=lambda item: (-item[0], item[1]))]
    return {
        "mode": mode,
        "query": query,
        "matches": matches,
        "scanned": scanned,
        "truncated": truncated,
    }


def search_walk_relative_files(root: Path, *, query: str, limit: int) -> dict[str, Any]:
    deadline = time.monotonic() + FILE_SEARCH_TIMEOUT_SECONDS
    heap: list[tuple[int, str]] = []
    scanned = 0
    truncated = False

    def _onerror(err: OSError) -> None:
        raise err

    for current_root, dirnames, filenames in os.walk(root, topdown=True, onerror=_onerror, followlinks=False):
        dirnames[:] = [name for name in sorted(dirnames) if name not in FILE_LIST_IGNORED_DIRS]
        current_path = Path(current_root)
        for name in sorted(filenames):
            scanned += 1
            if scanned > FILE_SEARCH_MAX_CANDIDATES or time.monotonic() > deadline:
                truncated = True
                return _finish_file_search(heap, mode="walk", query=query, scanned=scanned - 1, truncated=truncated)
            rel = (current_path / name).relative_to(root).as_posix()
            score = file_search_score(rel, query)
            if score < 0:
                continue
            # os.walk yields filenames decoded with surrogateescape; serialize
            # through the surrogate-safe display path codec used by the git path
            # layer so the JSON response body can be UTF-8 encoded.
            _push_file_search_match(heap, path=path_json_text(rel), score=score, limit=limit)
    return _finish_file_search(heap, mode="walk", query=query, scanned=scanned, truncated=truncated)


def search_git_relative_files(cwd: Path, *, query: str, limit: int) -> dict[str, Any]:
    deadline = time.monotonic() + FILE_SEARCH_TIMEOUT_SECONDS
    heap: list[tuple[int, str]] = []
    scanned = 0
    truncated = False
    proc = subprocess.Popen(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            path = raw_line.rstrip("\n")
            if not path:
                continue
            scanned += 1
            if scanned > FILE_SEARCH_MAX_CANDIDATES or time.monotonic() > deadline:
                truncated = True
                proc.kill()
                break
            score = file_search_score(path, query)
            if score < 0:
                continue
            _push_file_search_match(heap, path=path, score=score, limit=limit)
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        return_code = proc.wait()
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
    if truncated:
        return _finish_file_search(heap, mode="git", query=query, scanned=scanned - 1, truncated=True)
    if return_code != 0:
        err = stderr.strip()
        raise RuntimeError(err or f"git ls-files failed with code {return_code}")
    return _finish_file_search(heap, mode="git", query=query, scanned=scanned, truncated=False)


def _default_git_repo_root(cwd: Path) -> Path | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=FILE_SEARCH_GIT_ROOT_TIMEOUT_SECONDS,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None
    root = proc.stdout.strip()
    return Path(root).resolve() if root else None


def search_session_relative_files(
    base: Path,
    *,
    query: str,
    limit: int = FILE_SEARCH_LIMIT,
    git_root_func: GitRootFunc | None = None,
) -> dict[str, Any]:
    root = base.expanduser()
    if not root.is_absolute():
        root = root.resolve()
    if not root.exists():
        raise FileNotFoundError("session cwd not found")
    if not root.is_dir():
        raise ValueError("session cwd is not a directory")
    raw_query = str(query).strip()
    if not raw_query:
        raise ValueError("query required")
    clamped_limit = max(1, min(int(limit), FILE_SEARCH_LIMIT))
    root_func = git_root_func or _default_git_repo_root
    repo_root = root_func(root)
    if repo_root is not None:
        return search_git_relative_files(root, query=raw_query, limit=clamped_limit)
    return search_walk_relative_files(root, query=raw_query, limit=clamped_limit)
