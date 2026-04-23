from __future__ import annotations

import heapq
import subprocess
import time
from pathlib import Path
from typing import Any

from ..runtime import ServerRuntime


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
            total += (
                240 - exact_idx * 2 + boundary_bonus + (44 - base_idx if base_idx >= 0 else 0)
            )
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
        total += (
            120 - first - max(0, span - len(token)) * 4 + consecutive * 10 + boundaries * 8
        )
    return total


def _push_file_search_match(
    heap: list[tuple[int, str]], *, path: str, score: int, limit: int
) -> None:
    item = (score, path)
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    if item > heap[0]:
        heapq.heapreplace(heap, item)


def _finish_file_search(
    heap: list[tuple[int, str]], *, mode: str, query: str, scanned: int, truncated: bool
) -> dict[str, Any]:
    matches = [
        {"path": path, "score": score}
        for score, path in sorted(heap, key=lambda item: (-item[0], item[1]))
    ]
    return {
        "mode": mode,
        "query": query,
        "matches": matches,
        "scanned": scanned,
        "truncated": truncated,
    }


def search_walk_relative_files(
    runtime: ServerRuntime,
    root: Path,
    *,
    query: str,
    limit: int,
) -> dict[str, Any]:
    sv = runtime
    deadline = time.monotonic() + sv.FILE_SEARCH_TIMEOUT_SECONDS
    heap: list[tuple[int, str]] = []
    scanned = 0
    truncated = False

    def _onerror(err: OSError) -> None:
        raise err

    for current_root, dirnames, filenames in sv.os.walk(
        root, topdown=True, onerror=_onerror, followlinks=False
    ):
        dirnames[:] = [
            name for name in sorted(dirnames) if name not in sv.FILE_LIST_IGNORED_DIRS
        ]
        current_path = Path(current_root)
        for name in sorted(filenames):
            scanned += 1
            if scanned > sv.FILE_SEARCH_MAX_CANDIDATES or time.monotonic() > deadline:
                truncated = True
                return _finish_file_search(
                    heap,
                    mode="walk",
                    query=query,
                    scanned=scanned - 1,
                    truncated=truncated,
                )
            rel = (current_path / name).relative_to(root).as_posix()
            score = file_search_score(rel, query)
            if score < 0:
                continue
            _push_file_search_match(heap, path=rel, score=score, limit=limit)
    return _finish_file_search(
        heap,
        mode="walk",
        query=query,
        scanned=scanned,
        truncated=truncated,
    )


def search_git_relative_files(
    runtime: ServerRuntime,
    cwd: Path,
    *,
    query: str,
    limit: int,
) -> dict[str, Any]:
    sv = runtime
    deadline = time.monotonic() + sv.FILE_SEARCH_TIMEOUT_SECONDS
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
            if scanned > sv.FILE_SEARCH_MAX_CANDIDATES or time.monotonic() > deadline:
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
        return _finish_file_search(
            heap, mode="git", query=query, scanned=scanned - 1, truncated=True
        )
    if return_code != 0:
        err = stderr.strip()
        raise RuntimeError(err or f"git ls-files failed with code {return_code}")
    return _finish_file_search(
        heap, mode="git", query=query, scanned=scanned, truncated=False
    )


def search_session_relative_files(
    runtime: ServerRuntime,
    base: Path,
    *,
    query: str,
    limit: int,
) -> dict[str, Any]:
    sv = runtime
    root = sv._safe_expanduser(base)
    if not root.is_absolute():
        root = root.resolve()
    if not root.exists():
        raise FileNotFoundError("session cwd not found")
    if not root.is_dir():
        raise ValueError("session cwd is not a directory")
    raw_query = str(query).strip()
    if not raw_query:
        raise ValueError("query required")
    clamped_limit = max(1, min(int(limit), sv.FILE_SEARCH_LIMIT))
    repo_root = sv._git_repo_root(root)
    if repo_root is not None:
        return search_git_relative_files(runtime, root, query=raw_query, limit=clamped_limit)
    return search_walk_relative_files(runtime, root, query=raw_query, limit=clamped_limit)
