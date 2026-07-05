from __future__ import annotations

import heapq
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .git_ops import path_response_fields

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


def _push_file_search_match(heap: list[tuple[int, str, dict[str, Any]]], *, entry: dict[str, Any], score: int, limit: int) -> None:
    item = (score, str(entry.get("path", "")), entry)
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    if item > heap[0]:
        heapq.heapreplace(heap, item)


def _finish_file_search(heap: list[tuple[int, str, dict[str, Any]]], *, mode: str, query: str, scanned: int, truncated: bool) -> dict[str, Any]:
    ordered = sorted(heap, key=lambda item: (-item[0], item[1]))
    matches: list[dict[str, Any]] = []
    for score, _display, entry in ordered:
        match = dict(entry)
        match["score"] = score
        matches.append(match)
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
            # through path_response_fields so the JSON response body is UTF-8
            # safe and, when the raw name has undecodable bytes, attach a
            # reversible api_path token so the picker can re-open the file.
            entry = path_response_fields(rel)
            _push_file_search_match(heap, entry=entry, score=score, limit=limit)
    return _finish_file_search(heap, mode="walk", query=query, scanned=scanned, truncated=truncated)


def search_git_relative_files(cwd: Path, *, query: str, limit: int) -> dict[str, Any]:
    deadline = time.monotonic() + FILE_SEARCH_TIMEOUT_SECONDS
    heap: list[tuple[int, str]] = []
    scanned = 0
    truncated = False
    # ``-z`` makes git emit raw-byte paths NUL-delimited (no per-line newline).
    # Decoding with ``surrogateescape`` preserves undecodable bytes (e.g. a
    # raw 0xff in a filename) instead of mangling them into U+FFFD, so the
    # path can be serialized through ``path_response_fields`` into a JSON-safe
    # display string plus a reversible ``api_path`` token -- matching the
    # walk-mode contract. Reading binary chunks lets us stop early once the
    # candidate cap or deadline is hit by killing the process.
    proc = subprocess.Popen(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdout is not None
        buf = b""
        partial_at_eof = False
        while True:
            # Deadline guard between chunk reads, when no candidate increment
            # is pending. The per-candidate cap/deadline check inside the NUL
            # loop handles truncation mid-path.
            if time.monotonic() > deadline:
                truncated = True
                proc.kill()
                break
            chunk = proc.stdout.read(65536)
            if not chunk:
                if buf:
                    partial_at_eof = True
                break
            buf += chunk
            while b"\x00" in buf:
                raw_path, buf = buf.split(b"\x00", 1)
                if not raw_path:
                    continue
                scanned += 1
                if scanned > FILE_SEARCH_MAX_CANDIDATES or time.monotonic() > deadline:
                    # This candidate was counted by ``scanned += 1`` above but is
                    # NOT processed (we break before decode/score), mirroring walk
                    # mode where the cap-exceeding file is counted then dropped.
                    # Roll the increment back so ``scanned`` always equals the
                    # number of candidates actually admitted/considered under the
                    # cap, giving git and walk modes identical truncation
                    # metadata (e.g. cap=2 with 3 files -> scanned=2, not 3).
                    scanned -= 1
                    truncated = True
                    proc.kill()
                    buf = b""
                    break
                path = raw_path.decode("utf-8", errors="surrogateescape")
                score = file_search_score(path, query)
                if score < 0:
                    continue
                entry = path_response_fields(path)
                _push_file_search_match(heap, entry=entry, score=score, limit=limit)
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        return_code = proc.wait()
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
    # A trailing non-NUL-terminated fragment means git was killed mid-output;
    # treat the already-collected matches as a truncated result rather than
    # parsing a partial path. ``scanned`` already reflects only complete,
    # admitted candidates, so no rollback is needed here.
    if partial_at_eof:
        truncated = True
    if truncated:
        return _finish_file_search(heap, mode="git", query=query, scanned=scanned, truncated=True)
    if return_code != 0:
        err = stderr.decode("utf-8", errors="replace").strip() if isinstance(stderr, bytes) else str(stderr).strip()
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
