from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import os
import urllib.parse

from .git_ops import git_path_from_token
from .git_ops import git_path_response_fields
from .git_ops import path_json_text


JsonResponse = Callable[[Any, int, dict[str, Any]], None]
RouteMatcher = Callable[..., str | None]


@dataclass(frozen=True)
class GitRouteDeps:
    require_auth: Callable[[Any], bool]
    json_response: JsonResponse
    resolve_session_cwd: Callable[[str], Path]
    require_git_repo: Callable[[Path], None]
    split_git_nul_paths: Callable[[str], list[str]]
    run_git: Callable[..., str]
    parse_git_numstat: Callable[[str], dict[str, dict[str, int | None]]]
    resolve_git_path: Callable[[Path, str], tuple[Path, Path, str]]
    read_text_file_strict: Callable[..., tuple[str, int]]
    git_head_blob_oid: Callable[[Path, str], str | None]
    git_changed_files_max: int
    git_diff_timeout_seconds: float
    git_diff_max_bytes: int
    file_read_max_bytes: int


def handle_git_get_route(
    handler: Any,
    *,
    path: str,
    query: str,
    manager: Any,
    deps: GitRouteDeps,
    match_session_route: RouteMatcher,
) -> bool:
    session_id = match_session_route(path, "git", "changed_files")
    if session_id is not None:
        _handle_changed_files(handler, session_id=session_id, manager=manager, deps=deps)
        return True
    session_id = match_session_route(path, "git", "diff")
    if session_id is not None:
        _handle_diff(handler, session_id=session_id, query=query, manager=manager, deps=deps)
        return True
    session_id = match_session_route(path, "git", "file_versions")
    if session_id is not None:
        _handle_file_versions(handler, session_id=session_id, query=query, manager=manager, deps=deps)
        return True
    return False


def _authorized(handler: Any, deps: GitRouteDeps) -> bool:
    if deps.require_auth(handler):
        return True
    handler._unauthorized()
    return False


def _session_git_cwd(handler: Any, *, session_id: str, manager: Any, deps: GitRouteDeps) -> tuple[Any, Path] | None:
    manager.refresh_session_meta(session_id)
    session = manager.get_session(session_id)
    if not session:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return None
    try:
        cwd = deps.resolve_session_cwd(session.cwd)
        deps.require_git_repo(cwd)
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return None
    except RuntimeError as e:
        deps.json_response(handler, 409, {"error": str(e)})
        return None
    return session, cwd


def _norm_changed_list(paths: list[str], *, limit: int) -> list[str]:
    out: list[str] = []
    for path in paths:
        if path == "":
            continue
        out.append(path)
        if len(out) >= limit:
            break
    return out


def _path_from_query(qs: dict[str, list[str]]) -> str:
    token_q = qs.get("path_token")
    if token_q and token_q[0]:
        return git_path_from_token(token_q[0])
    path_q = qs.get("path")
    if not path_q or not path_q[0]:
        raise ValueError("path required")
    return path_q[0]


def _safe_path_list(paths: list[str]) -> list[str]:
    return [path_json_text(path) for path in paths]


def _handle_changed_files(handler: Any, *, session_id: str, manager: Any, deps: GitRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    session_cwd = _session_git_cwd(handler, session_id=session_id, manager=manager, deps=deps)
    if session_cwd is None:
        return
    _session, cwd = session_cwd
    try:
        unstaged = deps.split_git_nul_paths(
            deps.run_git(
                cwd,
                ["diff", "--name-only", "-z"],
                timeout_s=deps.git_diff_timeout_seconds,
                max_bytes=64 * 1024,
                decode_errors="surrogateescape",
            )
        )
        staged = deps.split_git_nul_paths(
            deps.run_git(
                cwd,
                ["diff", "--name-only", "--cached", "-z"],
                timeout_s=deps.git_diff_timeout_seconds,
                max_bytes=64 * 1024,
                decode_errors="surrogateescape",
            )
        )
        unstaged_numstat = deps.run_git(
            cwd,
            ["diff", "--numstat", "-z"],
            timeout_s=deps.git_diff_timeout_seconds,
            max_bytes=128 * 1024,
            decode_errors="surrogateescape",
        )
        staged_numstat = deps.run_git(
            cwd,
            ["diff", "--numstat", "--cached", "-z"],
            timeout_s=deps.git_diff_timeout_seconds,
            max_bytes=128 * 1024,
            decode_errors="surrogateescape",
        )
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return
    except RuntimeError as e:
        deps.json_response(handler, 409, {"error": str(e)})
        return
    unstaged2 = _norm_changed_list(unstaged, limit=deps.git_changed_files_max)
    staged2 = _norm_changed_list(staged, limit=deps.git_changed_files_max)
    seen: set[str] = set()
    merged: list[str] = []
    for path_key in [*unstaged2, *staged2]:
        if path_key in seen:
            continue
        seen.add(path_key)
        merged.append(path_key)
    stats = deps.parse_git_numstat(unstaged_numstat)
    for path_key, vals in deps.parse_git_numstat(staged_numstat).items():
        prev = stats.get(path_key)
        if prev is None:
            stats[path_key] = vals
            continue
        add_prev = prev.get("additions")
        del_prev = prev.get("deletions")
        add_new = vals.get("additions")
        del_new = vals.get("deletions")
        prev["additions"] = None if add_prev is None or add_new is None else int(add_prev) + int(add_new)
        prev["deletions"] = None if del_prev is None or del_new is None else int(del_prev) + int(del_new)
    entries: list[dict[str, Any]] = []
    for path_key in merged:
        vals = stats.get(path_key, {})
        entries.append(
            {
                **git_path_response_fields(path_key),
                "additions": vals.get("additions"),
                "deletions": vals.get("deletions"),
                "changed": True,
            }
        )
    deps.json_response(
        handler,
        200,
        {
            "ok": True,
            "cwd": path_json_text(cwd),
            "files": _safe_path_list(merged),
            "entries": entries,
            "unstaged": _safe_path_list(unstaged2),
            "staged": _safe_path_list(staged2),
        },
    )


def _handle_diff(handler: Any, *, session_id: str, query: str, manager: Any, deps: GitRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    session_cwd = _session_git_cwd(handler, session_id=session_id, manager=manager, deps=deps)
    if session_cwd is None:
        return
    _session, cwd = session_cwd
    qs = urllib.parse.parse_qs(query)
    try:
        rel = _path_from_query(qs)
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return
    staged_q = qs.get("staged")
    staged = bool(staged_q and staged_q[0] == "1")
    try:
        _target, repo_root, rel = deps.resolve_git_path(cwd, rel)
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return
    except RuntimeError as e:
        deps.json_response(handler, 409, {"error": str(e)})
        return
    args = ["diff", "-U3"]
    if staged:
        args.append("--cached")
    args.extend(["--", rel])
    try:
        diff = deps.run_git(
            repo_root,
            args,
            timeout_s=deps.git_diff_timeout_seconds,
            max_bytes=deps.git_diff_max_bytes,
            literal_pathspecs=True,
        )
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return
    except RuntimeError as e:
        deps.json_response(handler, 409, {"error": str(e)})
        return
    deps.json_response(
        handler,
        200,
        {"ok": True, "cwd": path_json_text(cwd), **git_path_response_fields(rel), "staged": staged, "diff": diff},
    )


def _handle_file_versions(handler: Any, *, session_id: str, query: str, manager: Any, deps: GitRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    session_cwd = _session_git_cwd(handler, session_id=session_id, manager=manager, deps=deps)
    if session_cwd is None:
        return
    _session, cwd = session_cwd
    qs = urllib.parse.parse_qs(query)
    try:
        rel = _path_from_query(qs)
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return
    try:
        p, repo_root, rel = deps.resolve_git_path(cwd, rel)
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return
    except RuntimeError as e:
        deps.json_response(handler, 409, {"error": str(e)})
        return
    current_text = ""
    current_size = 0
    current_exists = False
    try:
        p.parent.resolve().relative_to(repo_root)
        parent_inside_repo = True
    except (OSError, ValueError):
        parent_inside_repo = False
    if parent_inside_repo and p.is_symlink():
        try:
            current_raw = os.readlink(p).encode("utf-8", errors="surrogateescape")
            current_text = current_raw.decode("utf-8", errors="replace")
            current_size = len(current_raw)
            current_exists = True
        except FileNotFoundError:
            current_exists = False
            current_text = ""
            current_size = 0
        except PermissionError as e:
            deps.json_response(handler, 403, {"error": str(e)})
            return
        except OSError as e:
            deps.json_response(handler, 400, {"error": str(e)})
            return
    else:
        try:
            current_real = p.resolve()
            current_real.relative_to(repo_root)
            current_exists = bool(current_real.is_file())
        except (OSError, ValueError):
            current_exists = False
        if current_exists:
            try:
                current_text, current_size = deps.read_text_file_strict(current_real, max_bytes=deps.file_read_max_bytes)
            except FileNotFoundError:
                current_exists = False
                current_text = ""
                current_size = 0
            except PermissionError as e:
                deps.json_response(handler, 403, {"error": str(e)})
                return
            except ValueError as e:
                deps.json_response(handler, 400, {"error": str(e)})
                return
    try:
        manager.files_add(session_id, path_json_text(p))
    except KeyError:
        pass
    base_exists = False
    base_text = ""
    try:
        base_oid = deps.git_head_blob_oid(repo_root, rel)
        if base_oid:
            base_text = deps.run_git(
                repo_root,
                ["cat-file", "-p", base_oid],
                timeout_s=deps.git_diff_timeout_seconds,
                max_bytes=deps.file_read_max_bytes,
            )
            base_exists = True
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return
    except RuntimeError as e:
        deps.json_response(handler, 409, {"error": str(e)})
        return
    deps.json_response(
        handler,
        200,
        {
            "ok": True,
            "cwd": path_json_text(cwd),
            **git_path_response_fields(rel),
            "abs_path": path_json_text(p),
            "current_exists": current_exists,
            "current_size": int(current_size),
            "current_text": current_text,
            "base_exists": base_exists,
            "base_text": base_text,
        },
    )
