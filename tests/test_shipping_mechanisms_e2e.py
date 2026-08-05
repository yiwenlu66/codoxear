"""End-to-end public API contract for a newly created web-owned session.

The loopback server uses the production route handlers, response encoders, and
HTTP/1.1 request path.  Its manager is deliberately in-memory so creating a
session cannot launch a real broker during the test.
"""
from __future__ import annotations

import hashlib
import http.client
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace
import threading
from typing import Any
from urllib.parse import urlsplit

import pytest

from codoxear.auth_routes import AuthRouteDeps
from codoxear.auth_routes import handle_auth_get_route
from codoxear.diagnostics_routes import DiagnosticsRouteDeps
from codoxear.diagnostics_routes import handle_diagnostics_get_route
from codoxear.queue_routes import QueueRouteDeps
from codoxear.queue_routes import handle_queue_get_route
from codoxear.server import SessionNotReadyError
from codoxear.server import _match_session_route
from codoxear.server_http import BadRequestError
from codoxear.server_http import json_response
from codoxear.server_http import json_response_with_etag
from codoxear.server_http import read_body
from codoxear.server_main import ThreadingHTTPServer
from codoxear.session_listing import build_active_session_rows_snapshot
from codoxear.session_listing import build_public_session_row
from codoxear.session_model import Session
from codoxear.session_store import SessionStore
from codoxear.session_store import SessionStorePaths
from codoxear.session_routes import SessionRouteDeps
from codoxear.session_routes import handle_session_get_route
from codoxear.session_routes import handle_session_post_route
from codoxear.session_runtime import broker_runtime_state
from codoxear.session_runtime import resolve_runtime_status
from codoxear.voice_routes import VoiceRouteDeps
from codoxear.voice_routes import handle_voice_get_route


_AUTH_COOKIE = "codoxear_auth=shipping-e2e"


class _NotificationFeed:
    def notification_feed_since(self, since_ts: float) -> list[dict[str, object]]:
        assert since_ts == 0.0
        return []


class _ShippingManager:
    """In-memory stand-in for the process-owning session manager boundary."""

    def __init__(self, cwd: Path) -> None:
        self._cwd = cwd
        self._sessions: dict[str, Session] = {}
        self._queues: dict[str, list[dict[str, object]]] = {}
        self._store = SessionStore(
            paths=SessionStorePaths(
                aliases=cwd / "aliases.json",
                sidebar_meta=cwd / "sidebar.json",
                hidden_sessions=cwd / "hidden.json",
                files=cwd / "files.json",
                queues=cwd / "queues.json",
                pending_attachments=cwd / "pending.json",
                commit_unknown_sends=cwd / "commit_unknown.json",
                recent_cwds=cwd / "recent_cwds.json",
                unattended=cwd / "unattended.json",
            ),
            file_history_max=5,
            recent_cwd_max=5,
            unattended_default_idle_minutes=5,
            unattended_default_max_injections=10,
            clean_alias=lambda value: value if isinstance(value, str) else "",
            clean_priority_offset=lambda value: float(value or 0.0),
            clean_snooze_until=lambda value: float(value) if value not in (None, "", 0) else None,
            clean_dependency_session_id=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
            clean_recent_cwd=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
            clean_commit_unknown_send_record=lambda value: value if isinstance(value, dict) else None,
        )

    def spawn_web_session(self, **kwargs: object) -> dict[str, str]:
        session_id = "shipping-session-1"
        self._sessions[session_id] = Session(
            session_id=session_id,
            thread_id="shipping-thread-1",
            broker_pid=1,
            codex_pid=1,
            agent_backend=str(kwargs["agent_backend"]),
            owned=True,
            start_ts=1.0,
            cwd=str(kwargs["cwd"]),
            log_path=None,
            sock_path=self._cwd / "shipping.sock",
        )
        self._queues[session_id] = []
        return {"id": session_id}

    def list_sessions(self) -> list[dict[str, object]]:
        snapshot = build_active_session_rows_snapshot(
            sessions=self._sessions.values(),
            queues=self._queues,
            unattended={},
            aliases={},
            store=self._store,
            now_ts=1.0,
            unattended_default_idle_minutes=5,
            unattended_default_max_injections=10,
            clean_unattended_cooldown_minutes=lambda value: int(value),
            clean_unattended_remaining_injections=lambda value, *, allow_zero: int(value),
            provider_choice_for_settings=lambda **_kwargs: "default",
            resolve_session_cwd=Path,
            priority_half_life_seconds=60.0,
            priority_bucket_seconds=1.0,
            subagent_runs={},
        )
        return [build_public_session_row(row, git_branch=None, busy=False) for row in snapshot.rows]

    def recent_cwds(self) -> list[str]:
        return []

    def queue_list(self, session_id: str) -> list[dict[str, object]]:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        return list(self._queues[session_id])

    def refresh_session_meta(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise KeyError(session_id)

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_state(self, session_id: str) -> dict[str, object]:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        return {"busy": False, "queue_len": len(self._queues[session_id]), "token": None}

    def _runtime_status_from_state_and_log(self, session_id: str, state: dict[str, object], log_path: Path | None):
        assert session_id in self._sessions
        assert log_path is None
        return resolve_runtime_status(
            broker=broker_runtime_state(state),
            log_exists=False,
            log_idle=None,
            send_boundary_unresolved=False,
        )

    def _queue_len(self, session_id: str) -> int:
        return len(self._queues[session_id])

    def sidebar_meta_get(self, session_id: str) -> dict[str, object]:
        assert session_id in self._sessions
        return {"priority_offset": 0.0, "snooze_until": None, "dependency_session_id": None}


def _is_authenticated(handler: BaseHTTPRequestHandler) -> bool:
    return handler.headers.get("Cookie") == _AUTH_COOKIE


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    json_response(handler, status, payload, set_auth_cookie=lambda _handler: None)


def _session_route_deps() -> SessionRouteDeps:
    def parse_launch_request(obj: dict[str, object]) -> SimpleNamespace:
        cwd = obj.get("cwd")
        if not isinstance(cwd, str) or not cwd:
            raise BadRequestError("cwd required")
        return SimpleNamespace(
            cwd=cwd,
            args=[],
            agent_backend="pi",
            resume_session_id=None,
            worktree_branch=None,
            model_provider=None,
            preferred_auth_method=None,
            model=None,
            reasoning_effort=None,
            service_tier=None,
            create_in_tmux=False,
        )

    return SessionRouteDeps(
        require_auth=_is_authenticated,
        json_response=_json_response,
        json_response_with_etag=lambda handler, payload: json_response_with_etag(
            handler,
            payload,
            sha256_hex=lambda body: hashlib.sha256(body).hexdigest(),
            set_auth_cookie=lambda _handler: None,
        ),
        read_json_body=lambda handler, **_kwargs: json.loads(read_body(handler).decode("utf-8")),
        read_new_session_defaults=lambda: {"backend": "pi"},
        tmux_available=lambda: False,
        tmux_session_name="codoxear-test",
        metrics_snapshot=lambda: {},
        record_metric=lambda _name, _value: None,
        perf_counter=lambda: 1.0,
        normalize_agent_backend=lambda value, *, default="pi": value or default,
        default_agent_backend="pi",
        resolve_dir_target=lambda value, *, field_name: Path(value),
        describe_session_cwd=lambda cwd: {"cwd": str(cwd), "exists": cwd.is_dir()},
        list_resume_candidates_for_cwd=lambda _cwd, *, agent_backend: [],
        first_user_message_preview_from_log=lambda _path: "",
        parse_new_session_launch_request=parse_launch_request,
        launch_request_validation_error=BadRequestError,
        session_launch_error=RuntimeError,
    )


def _diagnostics_route_deps() -> DiagnosticsRouteDeps:
    return DiagnosticsRouteDeps(
        require_auth=_is_authenticated,
        json_response=_json_response,
        provider_choice_for_settings=lambda **_kwargs: "default",
        read_run_settings_from_log=lambda _path, **_kwargs: (None, None, None),
        resolve_session_cwd=Path,
        current_git_branch=lambda _cwd: None,
        sidebar_time_priority_from_elapsed_seconds=lambda _elapsed: 1.0,
        clip01=lambda value: value,
        time_fn=lambda: 1.0,
    )


class _ShippingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _unauthorized(self) -> None:
        _json_response(self, 401, {"error": "unauthorized"})

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if handle_auth_get_route(self, path=parsed.path, deps=self.server.auth_route_deps):
            return
        if handle_voice_get_route(
            self,
            path=parsed.path,
            query=parsed.query,
            voice_push=self.server.notification_feed,
            deps=self.server.voice_route_deps,
        ):
            return
        if handle_session_get_route(
            self,
            path=parsed.path,
            query=parsed.query,
            manager=self.server.manager,
            deps=self.server.session_route_deps,
            match_session_route=_match_session_route,
        ):
            return
        if handle_queue_get_route(
            self,
            path=parsed.path,
            manager=self.server.manager,
            deps=self.server.queue_route_deps,
            match_session_route=_match_session_route,
        ):
            return
        if handle_diagnostics_get_route(
            self,
            path=parsed.path,
            manager=self.server.manager,
            deps=self.server.diagnostics_route_deps,
            match_session_route=_match_session_route,
        ):
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if handle_session_post_route(
            self,
            path=urlsplit(self.path).path,
            manager=self.server.manager,
            deps=self.server.session_route_deps,
            match_session_route=_match_session_route,
        ):
            return
        self.send_error(404)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _ShippingApiServer:
    def __init__(self, cwd: Path) -> None:
        self.manager = _ShippingManager(cwd)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _ShippingHandler)
        self.httpd.manager = self.manager
        self.httpd.notification_feed = _NotificationFeed()
        self.httpd.auth_route_deps = AuthRouteDeps(
            require_auth=_is_authenticated,
            json_response=_json_response,
            read_json_body=lambda _handler, **_kwargs: {},
            is_same_password=lambda _password: False,
            set_auth_cookie=lambda _handler: None,
            cookie_name="codoxear_auth",
            cookie_path="/",
        )
        self.httpd.voice_route_deps = VoiceRouteDeps(
            require_auth=_is_authenticated,
            json_response=_json_response,
            read_json_body=lambda _handler, **_kwargs: {},
            load_unattended_prompt=lambda: "",
            save_unattended_prompt=lambda value: value,
            default_unattended_prompt="",
        )
        self.httpd.session_route_deps = _session_route_deps()
        self.httpd.queue_route_deps = QueueRouteDeps(
            require_auth=_is_authenticated,
            json_response=_json_response,
            read_json_body=lambda _handler, **_kwargs: {},
            session_not_ready_error=SessionNotReadyError,
        )
        self.httpd.diagnostics_route_deps = _diagnostics_route_deps()
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


@pytest.fixture
def shipping_api(tmp_path: Path):
    server = _ShippingApiServer(tmp_path)
    server.thread.start()
    try:
        yield server
    finally:
        server.close()


def _request(
    connection: http.client.HTTPConnection,
    method: str,
    path: str,
    *,
    body: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    raw_body = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Cookie": _AUTH_COOKIE}
    if raw_body is not None:
        headers.update({"Content-Type": "application/json", "Content-Length": str(len(raw_body))})
    connection.request(method, path, body=raw_body, headers=headers)
    response = connection.getresponse()
    raw_response = response.read()
    return response.status, json.loads(raw_response)


def test_web_owned_session_shipping_mechanisms_share_the_public_api_contract(
    shipping_api: _ShippingApiServer,
    tmp_path: Path,
) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", shipping_api.port, timeout=5)
    try:
        status, created = _request(connection, "POST", "/api/sessions", body={"cwd": str(tmp_path)})
        assert status == 200
        session_id = created["id"]
        assert isinstance(session_id, str)

        status, listing = _request(connection, "GET", "/api/sessions")
        assert status == 200
        session_rows = listing["sessions"]
        assert isinstance(session_rows, list)
        session = next(row for row in session_rows if row["session_id"] == session_id)
        assert session["subagents_running"] == 0

        status, notifications = _request(connection, "GET", "/api/notifications/feed")
        assert status == 200
        assert isinstance(notifications["items"], list)

        status, queue = _request(connection, "GET", f"/api/sessions/{session_id}/queue")
        assert status == 200
        assert isinstance(queue["items"], list)
        assert queue["queue_len"] == len(queue["items"])

        status, diagnostics = _request(connection, "GET", f"/api/sessions/{session_id}/diagnostics")
        assert status == 200
        assert isinstance(diagnostics, dict)

        status, authenticated_user = _request(connection, "GET", "/api/me")
        assert status == 200
        assert authenticated_user == {"ok": True}
    finally:
        connection.close()
