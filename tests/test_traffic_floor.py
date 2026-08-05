"""CI traffic floor for a quiet, selected web-owned session.

The fixture uses the production session create/list route handlers behind a
real loopback HTTP/1.1 server.  The client advances a 30-second *virtual* quiet
window rather than sleeping, so the wire protocol and conditional list
responses are real while the test remains suitable for CI.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import http.client
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

from codoxear.server_http import BadRequestError
from codoxear.server_http import json_response
from codoxear.server_http import json_response_with_etag
from codoxear.server_http import read_body
from codoxear.server_main import ThreadingHTTPServer
from codoxear.session_routes import SessionRouteDeps
from codoxear.session_routes import handle_session_get_route
from codoxear.session_routes import handle_session_post_route


WINDOW_SECONDS = 30
MAX_REQUESTS = 25
MAX_WIRE_BYTES = 50 * 1024
MAX_SESSIONS_RESPONSE_BYTES = 10 * 1024
MIN_ENDPOINT_AVERAGE_INTERVAL_SECONDS = 5
ROOT = Path(__file__).resolve().parents[1]
POLLING_MODULE = ROOT / "codoxear" / "static" / "app_polling.js"


class _TrafficLaunchError(RuntimeError):
    pass


class _TrafficManager:
    """The smallest web-launch manager that the production session routes need."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, object]] = {}
        self._next_session_number = 1

    def spawn_web_session(self, **kwargs: object) -> dict[str, str]:
        session_id = f"traffic-session-{self._next_session_number}"
        self._next_session_number += 1
        self.sessions[session_id] = {
            "id": session_id,
            "agent_backend": kwargs["agent_backend"],
            "cwd": kwargs["cwd"],
            "owned": True,
            "busy": False,
            "queue_len": 0,
        }
        return {"id": session_id}

    def list_sessions(self) -> list[dict[str, object]]:
        return list(self.sessions.values())

    def recent_cwds(self) -> list[str]:
        return []


def _launch_request(obj: dict[str, object]) -> SimpleNamespace:
    cwd = obj.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        raise ValueError("cwd is required")
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


def _route_deps() -> SessionRouteDeps:
    def no_cookie(_handler: BaseHTTPRequestHandler) -> None:
        return None

    return SessionRouteDeps(
        require_auth=lambda _handler: True,
        json_response=lambda handler, status, payload: json_response(handler, status, payload, set_auth_cookie=no_cookie),
        json_response_with_etag=lambda handler, payload: json_response_with_etag(
            handler,
            payload,
            sha256_hex=lambda body: hashlib.sha256(body).hexdigest(),
            set_auth_cookie=no_cookie,
        ),
        read_json_body=lambda handler, **_kwargs: json.loads(read_body(handler).decode("utf-8")),
        read_new_session_defaults=lambda: {"backend": "pi"},
        tmux_available=lambda: False,
        tmux_session_name="codoxear-test",
        metrics_snapshot=lambda: {},
        record_metric=lambda _name, _value: None,
        perf_counter=time.perf_counter,
        normalize_agent_backend=lambda value, *, default="pi": value or default,
        default_agent_backend="pi",
        resolve_dir_target=lambda value, *, field_name: Path(value),
        describe_session_cwd=lambda cwd: {"cwd": str(cwd), "exists": cwd.is_dir()},
        list_resume_candidates_for_cwd=lambda _cwd, *, agent_backend: [],
        first_user_message_preview_from_log=lambda _path: "",
        parse_new_session_launch_request=_launch_request,
        launch_request_validation_error=BadRequestError,
        session_launch_error=_TrafficLaunchError,
    )


def _match_session_route(path: str, *suffix: str) -> str | None:
    prefix = "/api/sessions/"
    ending = "/" + "/".join(suffix)
    if not path.startswith(prefix) or not path.endswith(ending):
        return None
    session_id = path[len(prefix):-len(ending)]
    return session_id or None


class _TrafficHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _unauthorized(self) -> None:
        json_response(self, 401, {"error": "unauthorized"}, set_auth_cookie=lambda _handler: None)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if handle_session_get_route(
            self,
            path=parsed.path,
            query=parsed.query,
            manager=self.server.manager,
            deps=self.server.route_deps,
            match_session_route=_match_session_route,
        ):
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if handle_session_post_route(
            self,
            path=urlsplit(self.path).path,
            manager=self.server.manager,
            deps=self.server.route_deps,
            match_session_route=_match_session_route,
        ):
            return
        self.send_error(404)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _TrafficApiServer:
    def __init__(self) -> None:
        self.manager = _TrafficManager()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _TrafficHandler)
        self.httpd.manager = self.manager
        self.httpd.route_deps = _route_deps()
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    def __enter__(self) -> _TrafficApiServer:
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


@dataclass(frozen=True)
class _TrafficRecord:
    endpoint: str
    virtual_second: int
    status: int
    wire_bytes: int


def _response_wire_bytes(response: http.client.HTTPResponse, body: bytes) -> int:
    """Count response status line, headers, separator, and received body bytes."""
    version = "1.1" if response.version == 11 else "1.0"
    status_line = f"HTTP/{version} {response.status} {response.reason}\r\n".encode("iso-8859-1")
    headers = b"".join(
        f"{name}: {value}\r\n".encode("iso-8859-1") for name, value in response.getheaders()
    )
    return len(status_line) + len(headers) + 2 + len(body)


def _fetch(
    connection: http.client.HTTPConnection,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes, int]:
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    return response.status, {name.lower(): value for name, value in response.getheaders()}, payload, _response_wire_bytes(response, payload)


def _visible_session_poll_seconds() -> float:
    """Execute the browser polling policy without needing a browser runtime."""
    script = """
        const fs = require("fs");
        const vm = require("vm");
        const ctx = { window: {} };
        vm.createContext(ctx);
        vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), ctx);
        process.stdout.write(String(ctx.window.CodoxearPolling.sessionsPollDelayMs("visible") / 1000));
    """
    result = subprocess.run(
        ["node", "-e", script, str(POLLING_MODULE)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout)


def _capture_quiet_window(
    connection: http.client.HTTPConnection,
    *,
    session_id: str,
    poll_seconds: float,
) -> list[_TrafficRecord]:
    """Run the visible-session polling profile across a virtual 30-second window."""
    records: list[_TrafficRecord] = []
    etag: str | None = None
    virtual_second = 0.0
    while virtual_second < WINDOW_SECONDS:
        headers = {"Accept-Encoding": "identity"}
        if etag:
            headers["If-None-Match"] = etag
        status, response_headers, _payload, wire_bytes = _fetch(connection, "GET", "/api/sessions", headers=headers)
        if status not in (200, 304):
            raise AssertionError(f"unexpected session-list status {status}")
        if not records:
            sessions = json.loads(_payload)["sessions"]
            assert any(row["id"] == session_id and row["owned"] is True for row in sessions)
        etag = response_headers.get("etag", etag)
        records.append(_TrafficRecord("/api/sessions", int(virtual_second), status, wire_bytes))
        virtual_second += poll_seconds
    return records


def test_quiet_web_session_traffic_stays_within_the_continuous_floor() -> None:
    started = time.monotonic()
    with TemporaryDirectory() as temp_dir, _TrafficApiServer() as server:
        connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
        try:
            create_body = json.dumps({"cwd": temp_dir}).encode("utf-8")
            status, _headers, payload, _wire_bytes = _fetch(
                connection,
                "POST",
                "/api/sessions",
                body=create_body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(create_body))},
            )
            assert status == 200
            session_id = json.loads(payload)["id"]

            poll_seconds = _visible_session_poll_seconds()
            traffic = _capture_quiet_window(connection, session_id=session_id, poll_seconds=poll_seconds)
        finally:
            connection.close()

    assert server.manager.sessions[session_id]["owned"] is True
    assert len(traffic) < MAX_REQUESTS
    assert sum(record.wire_bytes for record in traffic) < MAX_WIRE_BYTES
    assert all(record.wire_bytes < MAX_SESSIONS_RESPONSE_BYTES for record in traffic if record.endpoint == "/api/sessions")

    endpoint_counts: dict[str, int] = {}
    for record in traffic:
        endpoint_counts[record.endpoint] = endpoint_counts.get(record.endpoint, 0) + 1
    assert all(
        count * MIN_ENDPOINT_AVERAGE_INTERVAL_SECONDS <= WINDOW_SECONDS
        for count in endpoint_counts.values()
    )
    assert {record.status for record in traffic} == {200, 304}
    assert time.monotonic() - started < WINDOW_SECONDS
