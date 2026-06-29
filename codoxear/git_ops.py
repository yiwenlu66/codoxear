from __future__ import annotations

import base64
import binascii
import os
import posixpath
import re
import subprocess
from pathlib import Path
from typing import Callable

RunGit = Callable[..., str]
GitRepoRoot = Callable[[Path], Path | None]

GIT_PATH_TOKEN_PREFIX = "codoxear-git-path-bytes-v1:"


def _surrogate_path_bytes(value: str) -> bytes:
    return str(value).encode("utf-8", errors="surrogateescape")


def path_has_surrogate_bytes(value: str) -> bool:
    return any(0xDC80 <= ord(ch) <= 0xDCFF for ch in str(value))


def path_json_text(value: str | Path) -> str:
    text = str(value)
    if not path_has_surrogate_bytes(text):
        return text
    return _surrogate_path_bytes(text).decode("utf-8", errors="backslashreplace")


def git_path_token(value: str) -> str:
    raw = _surrogate_path_bytes(value)
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"{GIT_PATH_TOKEN_PREFIX}{encoded}"


def git_path_from_token(token: str) -> str:
    raw_token = str(token or "")
    if not raw_token.startswith(GIT_PATH_TOKEN_PREFIX):
        raise ValueError("invalid path token")
    payload = raw_token[len(GIT_PATH_TOKEN_PREFIX) :]
    if not payload or not re.fullmatch(r"[A-Za-z0-9_-]+", payload):
        raise ValueError("invalid path token")
    padding = "=" * (-len(payload) % 4)
    try:
        raw = base64.urlsafe_b64decode((payload + padding).encode("ascii"))
    except (binascii.Error, ValueError) as e:
        raise ValueError("invalid path token") from e
    if b"\x00" in raw:
        raise ValueError("invalid path")
    return raw.decode("utf-8", errors="surrogateescape")


def git_path_response_fields(path: str) -> dict[str, object]:
    fields: dict[str, object] = {"path": path_json_text(path)}
    if path_has_surrogate_bytes(path):
        fields["api_path"] = git_path_token(path)
        fields["non_utf8_path"] = True
    return fields


def git_path_query(path: str) -> str:
    display = path_json_text(path)
    query = f"path={posixpath_quote(display)}"
    if path_has_surrogate_bytes(path):
        query += f"&path_token={posixpath_quote(git_path_token(path))}"
    return query


def posixpath_quote(value: str) -> str:
    # Local import keeps urllib out of the git subprocess path unless URL assembly is needed.
    import urllib.parse

    return urllib.parse.quote(str(value))


def run_git(
    cwd: Path,
    args: list[str],
    *,
    timeout_s: float,
    max_bytes: int,
    literal_pathspecs: bool = False,
    decode_errors: str = "replace",
) -> str:
    cmd = ["git", *args]
    env = None
    if literal_pathspecs:
        env = os.environ.copy()
        for key in ("GIT_GLOB_PATHSPECS", "GIT_NOGLOB_PATHSPECS", "GIT_ICASE_PATHSPECS"):
            env.pop(key, None)
        env["GIT_LITERAL_PATHSPECS"] = "1"
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
            env=env,
        )
    except (OSError, ValueError) as e:
        raise RuntimeError(str(e)) from e
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(err or f"git failed with code {proc.returncode}")
    if len(proc.stdout) > max_bytes:
        raise ValueError(f"git output too large (max {max_bytes} bytes)")
    return proc.stdout.decode("utf-8", errors=decode_errors)


def resolve_git_path(cwd: Path, raw_path: str, *, run_git_func: RunGit, timeout_s: float) -> tuple[Path, Path, str]:
    repo_root = Path(
        run_git_func(
            cwd,
            ["rev-parse", "--show-toplevel"],
            timeout_s=timeout_s,
            max_bytes=64 * 1024,
            decode_errors="surrogateescape",
        ).strip()
    ).resolve()
    if not isinstance(raw_path, str) or raw_path == "":
        raise ValueError("path required")
    if "\x00" in raw_path:
        raise ValueError("invalid path")
    raw = Path(raw_path)
    if raw.is_absolute():
        target = Path(os.path.normpath(str(raw)))
        try:
            rel = target.relative_to(repo_root).as_posix()
        except ValueError as e:
            raise ValueError("path is outside git repo") from e
        if rel in {"", "."}:
            raise ValueError("path required")
    else:
        rel = posixpath.normpath(raw_path)
        if rel in {"", "."}:
            raise ValueError("path required")
        if rel == ".." or rel.startswith("../"):
            raise ValueError("path is outside git repo")
        target = repo_root / Path(rel)
    return target, repo_root, rel


def git_error_is_missing_head(message: str) -> bool:
    return "Not a valid object name HEAD" in message or "ambiguous argument 'HEAD'" in message or "bad revision 'HEAD'" in message


def git_head_blob_oid(cwd: Path, rel: str, *, run_git_func: RunGit, timeout_s: float) -> str | None:
    try:
        tree_match = run_git_func(
            cwd,
            ["ls-tree", "-z", "HEAD", "--", rel],
            timeout_s=timeout_s,
            max_bytes=64 * 1024,
            literal_pathspecs=True,
            decode_errors="surrogateescape",
        )
    except RuntimeError as e:
        if git_error_is_missing_head(str(e)):
            return None
        raise
    for entry in tree_match.split("\0"):
        if not entry:
            continue
        meta, sep, path = entry.partition("\t")
        if not sep or path != rel:
            continue
        parts = meta.split()
        if len(parts) >= 3:
            if parts[1] == "blob":
                return parts[2]
            raise RuntimeError(f"HEAD path is not a file: {rel}")
    return None


def require_git_repo(cwd: Path, *, run_git_func: RunGit, timeout_s: float) -> None:
    run_git_func(cwd, ["rev-parse", "--is-inside-work-tree"], timeout_s=timeout_s, max_bytes=4096)


def git_repo_root(cwd: Path, *, run_git_func: RunGit, timeout_s: float) -> Path | None:
    try:
        root = run_git_func(
            cwd,
            ["rev-parse", "--show-toplevel"],
            timeout_s=timeout_s,
            max_bytes=64 * 1024,
            decode_errors="surrogateescape",
        ).strip()
    except (RuntimeError, FileNotFoundError):
        return None
    if not root:
        return None
    return Path(root).resolve()


def current_git_branch(cwd: Path, *, run_git_func: RunGit, timeout_s: float) -> str | None:
    try:
        branch = run_git_func(cwd, ["rev-parse", "--abbrev-ref", "HEAD"], timeout_s=timeout_s, max_bytes=64 * 1024).strip()
    except (RuntimeError, FileNotFoundError, NotADirectoryError):
        return None
    return branch or None


def clean_worktree_branch(raw: str) -> str:
    if not isinstance(raw, str):
        raise ValueError("worktree_branch must be a string")
    branch = raw.strip()
    if not branch:
        raise ValueError("worktree_branch required")
    return branch


def worktree_path_slug(branch: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", branch).strip(".-")
    return slug or "worktree"


def default_worktree_path(source_cwd: Path, branch: str) -> Path:
    slug = worktree_path_slug(branch)
    return (source_cwd.parent / f"{source_cwd.name}-{slug}").resolve()


def create_git_worktree(source_cwd: Path, worktree_branch: str, *, git_repo_root_func: GitRepoRoot, timeout_s: float) -> Path:
    repo_root = git_repo_root_func(source_cwd)
    if repo_root is None:
        raise ValueError("cwd is not inside a git worktree")
    branch = clean_worktree_branch(worktree_branch)
    target = default_worktree_path(source_cwd, branch)
    if target.exists():
        raise ValueError(f"derived worktree path already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            ["git", "worktree", "add", "-b", branch, str(target)],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise ValueError("git worktree add timed out") from e
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        out = proc.stdout.decode("utf-8", errors="replace").strip()
        raise ValueError(err or out or f"git worktree add failed with code {proc.returncode}")
    return target.resolve()


def split_git_nul_paths(text: str) -> list[str]:
    return [part for part in text.split("\0") if part]


def parse_git_numstat(text: str) -> dict[str, dict[str, int | None]]:
    out: dict[str, dict[str, int | None]] = {}

    def add_entry(add_raw: str, del_raw: str, path_s: str) -> None:
        if path_s == "":
            return
        add_v = None if add_raw == "-" else int(add_raw)
        del_v = None if del_raw == "-" else int(del_raw)
        prev = out.get(path_s)
        if prev is None:
            out[path_s] = {"additions": add_v, "deletions": del_v}
            return
        if add_v is None or prev["additions"] is None:
            prev["additions"] = None
        else:
            prev["additions"] = int(prev["additions"]) + add_v
        if del_v is None or prev["deletions"] is None:
            prev["deletions"] = None
        else:
            prev["deletions"] = int(prev["deletions"]) + del_v

    if "\0" in text:
        records = text.split("\0")
        idx = 0
        while idx < len(records):
            raw = records[idx]
            idx += 1
            if raw == "":
                continue
            parts = raw.split("\t", 2)
            if len(parts) != 3:
                continue
            add_raw, del_raw, path = parts
            if path == "":
                # With --numstat -z, rename/copy records are encoded as:
                # add<TAB>del<TAB><NUL>old-path<NUL>new-path<NUL>.
                if idx + 1 >= len(records):
                    continue
                idx += 1
                path = records[idx]
                idx += 1
            add_entry(add_raw, del_raw, path)
    else:
        for raw in [raw for raw in text.splitlines() if raw != ""]:
            parts = raw.split("\t", 2)
            if len(parts) != 3:
                continue
            add_entry(parts[0], parts[1], parts[2])
    return out
