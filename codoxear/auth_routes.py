from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


JsonResponse = Callable[[Any, int, dict[str, Any]], None]
ReadJsonBody = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class AuthRouteDeps:
    require_auth: Callable[[Any], bool]
    json_response: JsonResponse
    read_json_body: ReadJsonBody
    is_same_password: Callable[[str], bool]
    set_auth_cookie: Callable[[Any], None]
    cookie_name: str
    cookie_path: str


def handle_auth_get_route(handler: Any, *, path: str, deps: AuthRouteDeps) -> bool:
    if path != "/api/me":
        return False
    if not _authorized(handler, deps):
        return True
    deps.json_response(handler, 200, {"ok": True})
    return True


def handle_auth_post_route(handler: Any, *, path: str, deps: AuthRouteDeps) -> bool:
    if path == "/api/login":
        obj = deps.read_json_body(handler)
        pw = obj.get("password")
        if not isinstance(pw, str) or not deps.is_same_password(pw):
            deps.json_response(handler, 403, {"error": "bad password"})
            return True
        body = b'{"ok":true}'
        handler.send_response(200)
        deps.set_auth_cookie(handler)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
        return True

    if path == "/api/logout":
        if not _authorized(handler, deps):
            return True
        body = b'{"ok":true}'
        handler.send_response(200)
        handler.send_header(
            "Set-Cookie",
            f"{deps.cookie_name}=deleted; Path={deps.cookie_path}; Max-Age=0; HttpOnly; SameSite=Strict",
        )
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
        return True

    return False


def _authorized(handler: Any, deps: AuthRouteDeps) -> bool:
    if deps.require_auth(handler):
        return True
    handler._unauthorized()
    return False
