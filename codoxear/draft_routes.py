from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


JsonResponse = Callable[[Any, int, dict[str, Any]], None]
RouteMatcher = Callable[..., str | None]
ReadJsonBody = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True)
class DraftRouteDeps:
    require_auth: Callable[[Any], bool]
    json_response: JsonResponse
    read_json_body: ReadJsonBody


def _authorized(handler: Any, deps: DraftRouteDeps) -> bool:
    if deps.require_auth(handler):
        return True
    handler._unauthorized()
    return False


def handle_draft_get_route(
    handler: Any,
    *,
    path: str,
    manager: Any,
    deps: DraftRouteDeps,
    match_session_route: RouteMatcher,
) -> bool:
    session_id = match_session_route(path, "draft")
    if session_id is None:
        return False
    if not _authorized(handler, deps):
        return True
    try:
        res = manager.draft_get(session_id)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return True
    deps.json_response(handler, 200, {"ok": True, "text": str(res.get("text") or ""), "updated_ts": float(res.get("updated_ts") or 0.0)})
    return True


def handle_draft_post_route(
    handler: Any,
    *,
    path: str,
    manager: Any,
    deps: DraftRouteDeps,
    match_session_route: RouteMatcher,
) -> bool:
    session_id = match_session_route(path, "draft")
    if session_id is None:
        return False
    if not _authorized(handler, deps):
        return True
    obj = deps.read_json_body(handler)
    text = obj.get("text")
    if not isinstance(text, str):
        deps.json_response(handler, 400, {"error": "text required"})
        return True
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        deps.json_response(handler, 400, {"error": "text must be valid UTF-8"})
        return True
    try:
        updated_ts = manager.draft_set(session_id, text)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return True
    except ValueError as e:
        deps.json_response(handler, 413, {"error": str(e)})
        return True
    deps.json_response(handler, 200, {"ok": True, "updated_ts": float(updated_ts)})
    return True
