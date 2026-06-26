from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


JsonResponse = Callable[[Any, int, dict[str, Any]], None]
RouteMatcher = Callable[..., str | None]
ReadJsonBody = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True)
class QueueRouteDeps:
    require_auth: Callable[[Any], bool]
    json_response: JsonResponse
    read_json_body: ReadJsonBody
    session_not_ready_error: type[BaseException]


def _queue_conflict_status(exc: ValueError) -> int:
    err_text = str(exc).lower()
    return 409 if ("commit" in err_text or "recovery" in err_text) else 502


def _authorized(handler: Any, deps: QueueRouteDeps) -> bool:
    if deps.require_auth(handler):
        return True
    handler._unauthorized()
    return False


def handle_queue_get_route(
    handler: Any,
    *,
    path: str,
    manager: Any,
    deps: QueueRouteDeps,
    match_session_route: RouteMatcher,
) -> bool:
    session_id = match_session_route(path, "queue")
    if session_id is None:
        return False
    if not _authorized(handler, deps):
        return True
    try:
        items = manager.queue_list(session_id)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return True
    except ValueError as e:
        deps.json_response(handler, 502, {"error": str(e)})
        return True
    deps.json_response(handler, 200, {"ok": True, "items": items, "queue": [str(item.get("text") or "") for item in items]})
    return True


def handle_queue_post_route(
    handler: Any,
    *,
    path: str,
    manager: Any,
    deps: QueueRouteDeps,
    match_session_route: RouteMatcher,
) -> bool:
    session_id = match_session_route(path, "enqueue")
    if session_id is not None:
        _handle_enqueue(handler, session_id=session_id, manager=manager, deps=deps)
        return True
    session_id = match_session_route(path, "queue", "delete")
    if session_id is not None:
        _handle_queue_delete(handler, session_id=session_id, manager=manager, deps=deps)
        return True
    session_id = match_session_route(path, "queue", "update")
    if session_id is not None:
        _handle_queue_update(handler, session_id=session_id, manager=manager, deps=deps)
        return True
    session_id = match_session_route(path, "queue", "move")
    if session_id is not None:
        _handle_queue_move(handler, session_id=session_id, manager=manager, deps=deps)
        return True
    return False


def _handle_enqueue(handler: Any, *, session_id: str, manager: Any, deps: QueueRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    obj = deps.read_json_body(handler)
    text = obj.get("text")
    if not isinstance(text, str) or not text.strip():
        deps.json_response(handler, 400, {"error": "text required"})
        return
    try:
        res = manager.enqueue(session_id, text)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    except deps.session_not_ready_error as e:
        deps.json_response(handler, 409, {"error": str(e)})
        return
    except ValueError as e:
        deps.json_response(handler, 502, {"error": str(e)})
        return
    deps.json_response(handler, 200, res)


def _handle_queue_delete(handler: Any, *, session_id: str, manager: Any, deps: QueueRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    obj = deps.read_json_body(handler)
    item_id = obj.get("id")
    if not isinstance(item_id, str) or not item_id.strip():
        deps.json_response(handler, 400, {"error": "id required"})
        return
    allow_commit_unknown_raw = obj.get("allow_commit_unknown", False)
    if not isinstance(allow_commit_unknown_raw, bool):
        deps.json_response(handler, 400, {"error": "allow_commit_unknown must be a boolean"})
        return
    allow_orphan_recovery_raw = obj.get("allow_orphan_recovery", False)
    if not isinstance(allow_orphan_recovery_raw, bool):
        deps.json_response(handler, 400, {"error": "allow_orphan_recovery must be a boolean"})
        return
    try:
        res = manager.queue_delete(
            session_id,
            item_id,
            allow_commit_unknown=allow_commit_unknown_raw is True,
            allow_orphan_recovery=allow_orphan_recovery_raw is True,
        )
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    except ValueError as e:
        deps.json_response(handler, _queue_conflict_status(e), {"error": str(e)})
        return
    deps.json_response(handler, 200, res)


def _handle_queue_update(handler: Any, *, session_id: str, manager: Any, deps: QueueRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    obj = deps.read_json_body(handler)
    item_id = obj.get("id")
    text = obj.get("text")
    if not isinstance(item_id, str) or not item_id.strip():
        deps.json_response(handler, 400, {"error": "id required"})
        return
    if not isinstance(text, str) or not text.strip():
        deps.json_response(handler, 400, {"error": "text required"})
        return
    try:
        res = manager.queue_update(session_id, item_id, text)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    except ValueError as e:
        deps.json_response(handler, _queue_conflict_status(e), {"error": str(e)})
        return
    deps.json_response(handler, 200, res)


def _handle_queue_move(handler: Any, *, session_id: str, manager: Any, deps: QueueRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    obj = deps.read_json_body(handler)
    item_id = obj.get("id")
    to_index = obj.get("to_index")
    if not isinstance(item_id, str) or not item_id.strip():
        deps.json_response(handler, 400, {"error": "id required"})
        return
    if isinstance(to_index, bool) or not isinstance(to_index, int):
        deps.json_response(handler, 400, {"error": "to_index required"})
        return
    try:
        res = manager.queue_move(session_id, item_id, to_index)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    except ValueError as e:
        deps.json_response(handler, _queue_conflict_status(e), {"error": str(e)})
        return
    deps.json_response(handler, 200, res)
