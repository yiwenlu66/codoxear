from __future__ import annotations

from dataclasses import dataclass
import http.server
import json
from typing import Any, Callable
import urllib.parse

from .auth_routes import handle_auth_get_route
from .auth_routes import handle_auth_post_route
from .control_routes import handle_control_post_route
from .diagnostics_routes import handle_diagnostics_get_route
from .file_routes import handle_absolute_file_preview_route
from .file_routes import handle_file_get_route
from .file_routes import handle_file_write_post_route
from .file_routes import handle_global_file_post_route
from .git_routes import handle_git_get_route
from .hook_routes import handle_hook_post_route
from .message_routes import handle_messages_get_route
from .queue_routes import handle_queue_get_route
from .queue_routes import handle_queue_post_route
from .session_routes import handle_session_get_route
from .session_routes import handle_session_post_route
from .static_routes import handle_static_get_route
from .voice_routes import handle_voice_get_route
from .voice_routes import handle_voice_post_route


@dataclass(frozen=True)
class ServerHandlerDeps:
    url_prefix: str
    strip_url_prefix: Callable[[str, str], str | None]
    is_client_disconnect: Callable[[BaseException], bool]
    json_response: Callable[[http.server.BaseHTTPRequestHandler, int, dict[str, Any]], None]
    handle_route_exception: Callable[[http.server.BaseHTTPRequestHandler, Exception], None]
    read_body: Callable[..., bytes]
    bad_request_error: type[BaseException]
    request_payload_too_large_error: type[BaseException]
    manager: Callable[[], Any]
    match_session_route: Callable[..., Any]
    static_route_deps: Callable[[], Any]
    auth_route_deps: Callable[[], Any]
    voice_route_deps: Callable[[], Any]
    session_route_deps: Callable[[], Any]
    diagnostics_route_deps: Callable[[], Any]
    queue_route_deps: Callable[[], Any]
    file_get_route_deps: Callable[[], Any]
    file_write_route_deps: Callable[[], Any]
    global_file_route_deps: Callable[[], Any]
    git_route_deps: Callable[[], Any]
    message_route_deps: Callable[[], Any]
    control_route_deps: Callable[[], Any]
    hook_route_deps: Callable[[], Any]


class CodoxearHandler(http.server.BaseHTTPRequestHandler):
    server_version = "codoxear/0.1"
    deps: ServerHandlerDeps

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except Exception as exc:
            if self.deps.is_client_disconnect(exc):
                return
            raise

    def finish(self) -> None:
        try:
            super().finish()
        except Exception as exc:
            if self.deps.is_client_disconnect(exc):
                return
            raise

    def _unauthorized(self) -> None:
        self.deps.json_response(self, 401, {"error": "unauthorized"})

    def _parse_prefixed_request_path(self) -> tuple[urllib.parse.ParseResult, str] | None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if self.deps.url_prefix:
            if path == self.deps.url_prefix:
                location = self.deps.url_prefix + "/"
                if parsed.query:
                    location = location + "?" + parsed.query
                self.send_response(308)
                self.send_header("Location", location)
                self.end_headers()
                return None
            stripped = self.deps.strip_url_prefix(self.deps.url_prefix, path)
            if stripped is None:
                self.send_error(404)
                return None
            path = stripped
        return parsed, path

    def _handle_static_get(self, path: str) -> bool:
        return handle_static_get_route(
            self,
            path=path,
            deps=self.deps.static_route_deps(),
        )

    def _handle_voice_get(self, path: str, query: str) -> bool:
        manager = self.deps.manager()
        return handle_voice_get_route(
            self,
            path=path,
            query=query,
            voice_push=getattr(manager, "_voice_push", None),
            deps=self.deps.voice_route_deps(),
        )

    def _read_json_body(self, *, limit: int = 2 * 1024 * 1024, too_large_error: str | None = None) -> dict[str, Any]:
        try:
            body = self.deps.read_body(self, limit=limit)
        except self.deps.request_payload_too_large_error as exc:
            if too_large_error:
                raise self.deps.request_payload_too_large_error(too_large_error) from exc
            raise
        try:
            body_text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise self.deps.bad_request_error("request body must be utf-8") from exc
        if not body_text.strip():
            raise self.deps.bad_request_error("empty request body")
        try:
            obj = json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise self.deps.bad_request_error("invalid json body") from exc
        if not isinstance(obj, dict):
            raise self.deps.bad_request_error("invalid json body (expected object)")
        return obj

    def _handle_voice_post(self, path: str) -> bool:
        manager = self.deps.manager()
        return handle_voice_post_route(
            self,
            path=path,
            voice_push=getattr(manager, "_voice_push", None),
            deps=self.deps.voice_route_deps(),
        )

    def do_GET(self) -> None:
        try:
            parsed = self._parse_prefixed_request_path()
            if parsed is None:
                return
            url, path = parsed
            manager = self.deps.manager()
            if self._handle_static_get(path):
                return
            if handle_auth_get_route(self, path=path, deps=self.deps.auth_route_deps()):
                return
            if self._handle_voice_get(path, url.query):
                return
            if handle_session_get_route(
                self,
                path=path,
                query=url.query,
                manager=manager,
                deps=self.deps.session_route_deps(),
                match_session_route=self.deps.match_session_route,
            ):
                return
            if handle_diagnostics_get_route(
                self,
                path=path,
                manager=manager,
                deps=self.deps.diagnostics_route_deps(),
                match_session_route=self.deps.match_session_route,
            ):
                return
            if handle_queue_get_route(
                self,
                path=path,
                manager=manager,
                deps=self.deps.queue_route_deps(),
                match_session_route=self.deps.match_session_route,
            ):
                return
            if handle_file_get_route(
                self,
                path=path,
                query=url.query,
                manager=manager,
                deps=self.deps.file_get_route_deps(),
                match_session_route=self.deps.match_session_route,
            ):
                return
            if handle_git_get_route(
                self,
                path=path,
                query=url.query,
                manager=manager,
                deps=self.deps.git_route_deps(),
                match_session_route=self.deps.match_session_route,
            ):
                return
            if handle_messages_get_route(
                self,
                path=path,
                query=url.query,
                manager=manager,
                deps=self.deps.message_route_deps(),
                match_session_route=self.deps.match_session_route,
            ):
                return
            self.send_error(404)
        except KeyError:
            self.deps.json_response(self, 404, {"error": "unknown session"})
        except Exception as exc:
            self.deps.handle_route_exception(self, exc)

    def do_POST(self) -> None:
        try:
            parsed = self._parse_prefixed_request_path()
            if parsed is None:
                return
            url, path = parsed
            manager = self.deps.manager()
            if handle_auth_post_route(self, path=path, deps=self.deps.auth_route_deps()):
                return
            if self._handle_voice_post(path):
                return
            if handle_session_post_route(
                self,
                path=path,
                manager=manager,
                deps=self.deps.session_route_deps(),
            ):
                return
            if handle_global_file_post_route(
                self,
                path=path,
                manager=manager,
                deps=self.deps.global_file_route_deps(),
            ):
                return
            if handle_absolute_file_preview_route(
                self,
                path=path,
                query=url.query,
                deps=self.deps.file_get_route_deps(),
            ):
                return
            if handle_file_write_post_route(
                self,
                path=path,
                manager=manager,
                deps=self.deps.file_write_route_deps(),
                match_session_route=self.deps.match_session_route,
            ):
                return
            if handle_control_post_route(
                self,
                path=path,
                manager=manager,
                deps=self.deps.control_route_deps(),
                match_session_route=self.deps.match_session_route,
            ):
                return
            if handle_queue_post_route(
                self,
                path=path,
                manager=manager,
                deps=self.deps.queue_route_deps(),
                match_session_route=self.deps.match_session_route,
            ):
                return
            if handle_hook_post_route(self, path=path, deps=self.deps.hook_route_deps()):
                return
            self.send_error(404)
        except KeyError:
            self.deps.json_response(self, 404, {"error": "unknown session"})
        except Exception as exc:
            self.deps.handle_route_exception(self, exc)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def make_handler_class(deps: ServerHandlerDeps) -> type[CodoxearHandler]:
    class Handler(CodoxearHandler):
        pass

    Handler.deps = deps
    Handler.__name__ = "Handler"
    Handler.__qualname__ = "Handler"
    return Handler
