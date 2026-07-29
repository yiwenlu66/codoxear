from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import urllib.parse

from .cwd_suggest import cwd_suggestions


JsonResponse = Callable[[Any, int, dict[str, Any]], None]
JsonResponseWithEtag = Callable[[Any, dict[str, Any]], None]
ReadJsonBody = Callable[..., dict[str, Any]]
RouteMatcher = Callable[..., str | None]


@dataclass(frozen=True)
class SessionRouteDeps:
    require_auth: Callable[[Any], bool]
    json_response: JsonResponse
    json_response_with_etag: JsonResponseWithEtag
    read_json_body: ReadJsonBody
    read_new_session_defaults: Callable[[], dict[str, Any]]
    static_asset_version: Callable[[], str]
    tmux_available: Callable[[], bool]
    tmux_session_name: str
    metrics_snapshot: Callable[[], dict[str, Any]]
    record_metric: Callable[[str, float], None]
    perf_counter: Callable[[], float]
    normalize_agent_backend: Callable[..., str]
    default_agent_backend: str
    resolve_dir_target: Callable[..., Path]
    describe_session_cwd: Callable[[Path], dict[str, Any]]
    list_resume_candidates_for_cwd: Callable[..., list[dict[str, Any]]]
    first_user_message_preview_from_log: Callable[[Path], str]
    parse_new_session_launch_request: Callable[[dict[str, Any]], Any]
    launch_request_validation_error: type[BaseException]
    session_launch_error: type[BaseException]


def handle_session_get_route(
    handler: Any,
    *,
    path: str,
    query: str,
    manager: Any,
    deps: SessionRouteDeps,
    match_session_route: RouteMatcher,
) -> bool:
    if path == "/api/sessions":
        _handle_sessions_list(handler, manager=manager, deps=deps)
        return True
    if path == "/api/session_resume_candidates":
        _handle_session_resume_candidates(handler, query=query, manager=manager, deps=deps)
        return True
    if path == "/api/cwd-suggest":
        _handle_cwd_suggest(handler, query=query, deps=deps)
        return True
    if path == "/api/metrics":
        _handle_metrics(handler, deps=deps)
        return True

    session_id = match_session_route(path, "tail")
    if session_id is not None:
        _handle_tail(handler, session_id=session_id, manager=manager, deps=deps)
        return True

    session_id = match_session_route(path, "unattended")
    if session_id is not None:
        _handle_unattended_get(handler, session_id=session_id, manager=manager, deps=deps)
        return True

    return False


def handle_session_post_route(
    handler: Any,
    *,
    path: str,
    manager: Any,
    deps: SessionRouteDeps,
) -> bool:
    if path != "/api/sessions":
        return False
    _handle_session_create(handler, manager=manager, deps=deps)
    return True


def _authorized(handler: Any, deps: SessionRouteDeps) -> bool:
    if deps.require_auth(handler):
        return True
    handler._unauthorized()
    return False


def _handle_sessions_list(handler: Any, *, manager: Any, deps: SessionRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    t0 = deps.perf_counter()
    sessions = manager.list_sessions()
    recent_cwds = manager.recent_cwds()
    new_session_defaults = deps.read_new_session_defaults()
    dt_ms = (deps.perf_counter() - t0) * 1000.0
    deps.record_metric("api_sessions_ms", dt_ms)
    deps.json_response_with_etag(
        handler,
        {
            "app_version": deps.static_asset_version(),
            "sessions": sessions,
            "recent_cwds": recent_cwds,
            "new_session_defaults": new_session_defaults,
            "tmux_available": deps.tmux_available(),
            "tmux_session_name": deps.tmux_session_name,
        },
    )


def _handle_session_resume_candidates(handler: Any, *, query: str, manager: Any, deps: SessionRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    qs = urllib.parse.parse_qs(query)
    cwd_raw = qs.get("cwd", [""])[0]
    try:
        agent_backend = deps.normalize_agent_backend(qs.get("agent_backend", [""])[0], default=deps.default_agent_backend)
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return
    try:
        cwd_path = deps.resolve_dir_target(str(cwd_raw), field_name="cwd")
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e), "field": "cwd"})
        return
    info = deps.describe_session_cwd(cwd_path)
    rows = deps.list_resume_candidates_for_cwd(info["cwd"], agent_backend=agent_backend) if info["exists"] else []
    for row in rows:
        sid = row.get("session_id")
        log_path_raw = row.get("log_path")
        alias = manager.alias_get(sid) if isinstance(sid, str) and sid else ""
        preview = ""
        if isinstance(log_path_raw, str) and log_path_raw:
            preview = deps.first_user_message_preview_from_log(Path(log_path_raw))
        row["alias"] = alias
        row["first_user_message"] = preview
    deps.json_response(handler, 200, {"ok": True, **info, "sessions": rows})


def _handle_cwd_suggest(handler: Any, *, query: str, deps: SessionRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    qs = urllib.parse.parse_qs(query)
    path = qs.get("path", [""])[0]
    prefix = qs.get("prefix", [""])[0]
    deps.json_response(handler, 200, {"directories": cwd_suggestions(path, prefix=prefix)})


def _handle_metrics(handler: Any, *, deps: SessionRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    deps.json_response(handler, 200, {"metrics": deps.metrics_snapshot()})


def _handle_tail(handler: Any, *, session_id: str, manager: Any, deps: SessionRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    try:
        tail = manager.get_tail(session_id)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    deps.json_response(handler, 200, {"tail": tail})


def _handle_unattended_get(handler: Any, *, session_id: str, manager: Any, deps: SessionRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    try:
        cfg = manager.unattended_get(session_id)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    deps.json_response(handler, 200, {"ok": True, **cfg})


def _handle_session_create(handler: Any, *, manager: Any, deps: SessionRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    obj = deps.read_json_body(handler)
    try:
        launch_req = deps.parse_new_session_launch_request(obj)
    except deps.launch_request_validation_error as e:
        payload: dict[str, Any] = {"error": str(e)}
        field = getattr(e, "field", None)
        if field:
            payload["field"] = field
        deps.json_response(handler, 400, payload)
        return
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return
    try:
        res = manager.spawn_web_session(
            cwd=launch_req.cwd,
            args=launch_req.args,
            agent_backend=launch_req.agent_backend,
            resume_session_id=launch_req.resume_session_id,
            worktree_branch=launch_req.worktree_branch,
            model_provider=launch_req.model_provider,
            preferred_auth_method=launch_req.preferred_auth_method,
            model=launch_req.model,
            reasoning_effort=launch_req.reasoning_effort,
            service_tier=launch_req.service_tier,
            create_in_tmux=launch_req.create_in_tmux,
        )
    except ValueError as e:
        payload = {"error": str(e)}
        if str(e).startswith("cwd "):
            payload["field"] = "cwd"
        deps.json_response(handler, 400, payload)
        return
    except deps.session_launch_error as e:
        record = getattr(e, "record")
        deps.json_response(
            handler,
            500,
            {
                "error": str(e),
                "launch_attempt": record,
                "launch_id": record.get("launch_id"),
            },
        )
        return
    deps.json_response(handler, 200, {"ok": True, **res})
