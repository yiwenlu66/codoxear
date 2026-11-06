"""End-to-end federation tests: in-process hub and node on loopback.

The hub side is a real codoxear HTTP server (real handler, real discovery,
real SessionManager) whose config paths are redirected to a tmp dir. The node
side is a FederationNode wired to a FAKE broker (a tiny unix-socket server
answering the control op table) plus a minimal stub HTTP server that stands
in for the node's content routes and checks the federation auth header.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from codoxear import federation_hub
from codoxear import federation_node
from codoxear import federation_proto
from codoxear import server
from codoxear.auth import sign_cookie
from codoxear.control_socket import call_control_socket
from codoxear.server_main import ThreadingHTTPServer
from codoxear.server_handler import make_server_handler
from codoxear.socket_json import send_socket_json_line

SECRET = "federation-e2e-secret"
NODE_NAME = "node-b"
SESSION_ID = "fed-e2e-session-0001"
_TEST_SECRET = b"federation-e2e-hmac-key"


# --------------------------------------------------------------------------- #
# Fake broker: one unix-socket server answering the control op table.

_FAKE_TAIL = "fake broker tail output"


class _FakeBroker:
    def __init__(self, sock_dir: Path, session_id: str, cwd: Path) -> None:
        self.sock_dir = sock_dir
        self.session_id = session_id
        self.sock_path = sock_dir / f"{session_id}.sock"
        self.received: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._write_sidecar(cwd)

    def _write_sidecar(self, cwd: Path) -> None:
        meta = {
            "session_id": self.session_id,
            "broker_pid": os.getpid(),
            "codex_pid": os.getpid(),
            "agent_backend": "codex",
            "owner": "web",
            "cwd": str(cwd),
            "start_ts": time.time(),
            "log_path": None,
            "control_protocol_version": 1,
        }
        self.sock_dir.mkdir(parents=True, exist_ok=True)
        (self.sock_dir / f"{self.session_id}.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")

    def _answer(self, request: dict[str, Any]) -> dict[str, Any]:
        self.received.append(request)
        cmd = request.get("cmd")
        if cmd == "state":
            return {"busy": False, "queue_len": 0, "interrupted_idle": False}
        if cmd == "tail":
            return {"tail": _FAKE_TAIL}
        if cmd == "send":
            return {"ok": True, "queued": False, "queue_len": 0}
        if cmd == "keys":
            return {"ok": True, "queued": False, "n": 1, "key_queue_len": 0}
        if cmd == "shutdown":
            return {"ok": True}
        return {"error": f"unknown cmd {cmd!r}"}

    def _serve(self, conn: socket.socket) -> None:
        try:
            buf = bytearray()
            while b"\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf.extend(chunk)
            request = json.loads(bytes(buf).split(b"\n", 1)[0].decode("utf-8"))
            send_socket_json_line(conn, self._answer(request))
        except OSError:
            pass
        finally:
            conn.close()

    def _accept_loop(self) -> None:
        listener = self._listener
        assert listener is not None
        while not self._stop.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def start(self) -> None:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock_path.unlink()
        except FileNotFoundError:
            pass
        listener.bind(str(self.sock_path))
        listener.listen(8)
        listener.settimeout(0.2)
        self._listener = listener
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        for path in (self.sock_path, self.sock_path.with_suffix(".json")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


# --------------------------------------------------------------------------- #
# Stub node HTTP server: stands in for the node's content routes.

class _NodeHttpState:
    requests: list[dict[str, Any]] = []

    @classmethod
    def reset(cls) -> None:
        cls.requests = []


def _make_node_http_handler() -> type:
    class NodeHandler(server.http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _handle(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            token = self.headers.get(federation_proto.FEDERATION_AUTH_HEADER)
            _NodeHttpState.requests.append({"method": self.command, "path": self.path, "token": token})
            if token != federation_proto.federation_token(SECRET, NODE_NAME):
                body = json.dumps({"error": "unauthorized"}).encode()
                self.send_response(401)
            else:
                body = json.dumps({"served_by": NODE_NAME, "path": self.path}).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = _handle
        do_POST = _handle

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return NodeHandler


# --------------------------------------------------------------------------- #
# Hub server fixture: real handler + real fresh SessionManager on tmp paths.

def _hub_path_patches(app: Path) -> dict[str, Any]:
    return {
        "APP_DIR": app,
        "SOCK_DIR": app / "socks",
        "STATE_PATH": app / "state.json",
        "HMAC_SECRET_PATH": app / "hmac_secret",
        "LAUNCH_ATTEMPTS_PATH": app / "launch_attempts.jsonl",
        "UPLOAD_DIR": app / "uploads",
        "SEND_JOURNAL_DIR": app / "send_journal",
        "UNATTENDED_PATH": app / "unattended.json",
        "ALIAS_PATH": app / "session_aliases.json",
        "SIDEBAR_META_PATH": app / "session_sidebar.json",
        "HIDDEN_SESSIONS_PATH": app / "hidden_sessions.json",
        "FILE_HISTORY_PATH": app / "session_files.json",
        "VIDEO_PREVIEW_DIR": app / "video_previews",
        "OFFICE_PREVIEW_DIR": app / "office_previews",
        "QUEUE_PATH": app / "session_queues.json",
        "DRAFT_PATH": app / "session_drafts.json",
        "PENDING_ATTACHMENTS_PATH": app / "pending_attachments.json",
        "STAGED_ATTACHMENTS_PATH": app / "staged_attachments.json",
        "SESSION_UNREAD_PATH": app / "session_unread.json",
        "COMMIT_UNKNOWN_SENDS_PATH": app / "commit_unknown_sends.json",
        "RECENT_CWD_PATH": app / "recent_cwds.json",
        "QUICK_DIR": app / "quick",
        "QUICK_STATE_PATH": app / "quick_state.json",
        "VOICE_SETTINGS_PATH": app / "voice_settings.json",
        "UNATTENDED_PROMPT_PATH": app / "unattended_prompt.txt",
        "PUSH_SUBSCRIPTIONS_PATH": app / "push_subscriptions.json",
        "DELIVERY_LEDGER_PATH": app / "voice_delivery_ledger.json",
        "VAPID_PRIVATE_KEY_PATH": app / "webpush_vapid_private.pem",
        "CODEX_HOME": app / "codex_home",
        "CODEX_SESSIONS_DIR": app / "codex_home" / "sessions",
        "CODEX_CONFIG_PATH": app / "codex_home" / "config.toml",
        "MODELS_CACHE_PATH": app / "codex_home" / "models_cache.json",
        "PI_HOME": app / "pi_home",
        "PI_SESSIONS_DIR": app / "pi_home" / "sessions",
        "PI_SETTINGS_PATH": app / "pi_home" / "agent" / "settings.json",
        "PI_MODELS_PATH": app / "pi_home" / "agent" / "models.json",
        "PI_AUTH_PATH": app / "pi_home" / "agent" / "auth.json",
        "CC_HOME": app / "cc_home",
        "CC_SESSIONS_DIR": app / "cc_home" / "sessions",
        "CC_SETTINGS_PATH": app / "cc_home" / "settings.json",
        "SESSION_SNAPSHOT_PATH": app / "session_snapshot.json",
        "HMAC_SECRET": _TEST_SECRET,
    }


def _http_request(address: tuple[str, int], method: str, path: str, *, body: dict[str, Any] | None = None, auth: bool = True) -> tuple[int, dict[str, Any] | bytes]:
    conn = http.client.HTTPConnection(address[0], address[1], timeout=10)
    headers = {"Host": "test"}
    if auth:
        headers["Cookie"] = f"{server.COOKIE_NAME}={sign_cookie({'v': 1}, secret=_TEST_SECRET)}"
    raw: bytes | None = None
    if body is not None:
        raw = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=raw, headers=headers)
    response = conn.getresponse()
    payload = response.read()
    conn.close()
    text = payload.decode("utf-8")
    try:
        parsed: dict[str, Any] | bytes = json.loads(text)
    except json.JSONDecodeError:
        parsed = payload
    return response.status, parsed


def _wait_for(predicate, *, timeout_s: float = 10.0, interval_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(interval_s)
    return False


class _HubWorld:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.app_dir = tmp_path / "hub_app"
        self.sock_dir = self.app_dir / "socks"
        self.sock_dir.mkdir(parents=True, exist_ok=True)
        for name, value in _hub_path_patches(self.app_dir).items():
            monkeypatch.setattr(server, name, value)
        self.manager = server.SessionManager()
        monkeypatch.setattr(server, "MANAGER", self.manager)
        # Start the real server glue: hub listener from env, active-hub wiring,
        # drop_sessions callback into this manager.
        monkeypatch.setenv(federation_hub.ENV_FEDERATION_LISTEN, "127.0.0.1:0")
        monkeypatch.setenv(federation_hub.ENV_FEDERATION_TOKEN, SECRET)
        handler = make_server_handler(server)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.http_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.http_thread.start()
        server._start_federation(self.httpd.server_address[1])
        hub = federation_hub.active_hub()
        assert hub is not None
        self.hub = hub
        self.address = ("127.0.0.1", self.httpd.server_address[1])

    def list_sessions(self) -> list[dict[str, Any]]:
        self.manager._discover_existing(force=True)
        status, payload = _http_request(self.address, "GET", "/api/sessions")
        assert status == 200, payload
        assert isinstance(payload, dict)
        return list(payload["sessions"])

    def stop(self) -> None:
        server._stop_federation()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.http_thread.join(timeout=3)
        self.manager.stop()


class _NodeWorld:
    def __init__(self, tmp_path: Path) -> None:
        self.sock_dir = tmp_path / "node_app" / "socks"
        self.sock_dir.mkdir(parents=True, exist_ok=True)
        _NodeHttpState.reset()
        self.httpd = server.http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_node_http_handler())
        self.http_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.http_thread.start()
        self.launches: list[dict[str, Any]] = []
        self.node: federation_node.FederationNode | None = None

    @property
    def http_port(self) -> int:
        return int(self.httpd.server_address[1])

    def _launch_stub(self, params: dict[str, Any]) -> dict[str, Any]:
        self.launches.append(dict(params))
        return {"session_id": "launched-on-node", "cwd": params.get("cwd")}

    def start_client(self, hub_port: int) -> None:
        config = federation_node.FederationNodeConfig(
            hub_host="127.0.0.1",
            hub_port=hub_port,
            secret=SECRET,
            node_name=NODE_NAME,
            http_port=self.http_port,
            sock_dir=self.sock_dir,
            watch_poll_s=0.1,
        )
        # Hub and node share this test process: the mirrored-session guard is
        # for a single server that is both hub and node, so it is disabled
        # here (in production the two roles are different processes/dirs).
        self.node = federation_node.FederationNode(
            config,
            launch_session=self._launch_stub,
            is_mirrored_session=lambda _sid: False,
        )
        self.node.start()

    def stop_client(self) -> None:
        if self.node is not None:
            self.node.stop()
            self.node = None

    def stop(self) -> None:
        self.stop_client()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.http_thread.join(timeout=3)


@pytest.fixture
def worlds(tmp_path, monkeypatch):
    hub = _HubWorld(tmp_path, monkeypatch)
    node = _NodeWorld(tmp_path)
    broker = _FakeBroker(node.sock_dir, SESSION_ID, tmp_path)
    broker.start()
    try:
        yield hub, node, broker
    finally:
        broker.stop()
        node.stop()
        hub.stop()


def test_federation_end_to_end(worlds, tmp_path):
    hub_world, node_world, broker = worlds
    node_world.start_client(hub_world.hub.port)

    # -- The node registers its session; the hub mirrors sock + sidecar. ---- #
    assert _wait_for(lambda: (hub_world.sock_dir / f"{SESSION_ID}.json").exists()
                     and (hub_world.sock_dir / f"{SESSION_ID}.sock").exists())
    mirrored = json.loads((hub_world.sock_dir / f"{SESSION_ID}.json").read_text())
    assert mirrored["broker_pid"] == os.getpid()
    assert mirrored["codex_pid"] == os.getpid()
    assert mirrored["log_path"] is None
    assert mirrored["session_id"] == SESSION_ID
    assert mirrored["cwd"] == str(tmp_path)

    # -- Session appears in the hub session list payload with a node badge. - #
    rows: list[dict[str, Any]] = []

    def listed() -> bool:
        nonlocal rows
        rows = hub_world.list_sessions()
        return any(row.get("session_id") == SESSION_ID for row in rows)

    assert _wait_for(listed)
    row = next(row for row in rows if row["session_id"] == SESSION_ID)
    assert row["node"] == NODE_NAME

    # -- Hub sock_call path: state/send/keys/tail through the relay socket. - #
    relay_sock = hub_world.sock_dir / f"{SESSION_ID}.sock"
    state = call_control_socket(relay_sock, {"cmd": "state"}, timeout_s=5.0)
    assert state == {"busy": False, "queue_len": 0, "interrupted_idle": False}
    tail = call_control_socket(relay_sock, {"cmd": "tail"}, timeout_s=5.0)
    assert tail == {"tail": _FAKE_TAIL}
    send = call_control_socket(relay_sock, {"cmd": "send", "text": "hello"}, timeout_s=5.0)
    assert send["ok"] is True
    keys = call_control_socket(relay_sock, {"cmd": "keys", "seq": ""}, timeout_s=5.0)
    assert keys["ok"] is True
    assert {req["cmd"] for req in broker.received} >= {"state", "tail", "send", "keys"}

    # -- Hub HTTP content route is reverse-proxied to the node server. ------ #
    status, payload = _http_request(hub_world.address, "GET", f"/api/sessions/{SESSION_ID}/messages?limit=1")
    assert status == 200
    assert payload == {"served_by": NODE_NAME, "path": f"/api/sessions/{SESSION_ID}/messages?limit=1"}
    proxied = [req for req in _NodeHttpState.requests if req["path"].startswith(f"/api/sessions/{SESSION_ID}/")]
    assert proxied, "node server saw no proxied request"
    assert all(req["token"] == federation_proto.federation_token(SECRET, NODE_NAME) for req in proxied)

    # -- Create with a node field launches on the node, not locally. -------- #
    status, payload = _http_request(
        hub_world.address,
        "POST",
        "/api/sessions",
        body={"cwd": str(tmp_path), "agent_backend": "codex", "node": NODE_NAME},
    )
    assert status == 200, payload
    assert payload["ok"] is True
    assert payload["session_id"] == "launched-on-node"
    assert _wait_for(lambda: len(node_world.launches) == 1)
    assert node_world.launches[0]["cwd"] == str(tmp_path)
    assert node_world.launches[0]["agent_backend"] == "codex"

    # -- Unknown node names are rejected. ----------------------------------- #
    status, payload = _http_request(
        hub_world.address,
        "POST",
        "/api/sessions",
        body={"cwd": str(tmp_path), "node": "no-such-node"},
    )
    assert status == 404, payload

    # -- Node disconnect: mirror files and the hub session row disappear. --- #
    node_world.stop_client()
    assert _wait_for(lambda: not (hub_world.sock_dir / f"{SESSION_ID}.json").exists()
                     and not (hub_world.sock_dir / f"{SESSION_ID}.sock").exists())
    assert _wait_for(lambda: all(row.get("session_id") != SESSION_ID for row in hub_world.list_sessions()))

    # -- Reconnect: the session is re-registered and reappears. ------------- #
    node_world.start_client(hub_world.hub.port)
    assert _wait_for(lambda: (hub_world.sock_dir / f"{SESSION_ID}.json").exists())

    def listed_again() -> bool:
        return any(row.get("session_id") == SESSION_ID and row.get("node") == NODE_NAME
                   for row in hub_world.list_sessions())

    assert _wait_for(listed_again)
    state = call_control_socket(hub_world.sock_dir / f"{SESSION_ID}.sock", {"cmd": "state"}, timeout_s=5.0)
    assert state["busy"] is False


def test_federation_token_auth_acceptance_on_node(monkeypatch):
    """Node-side auth: X-Federation-Token unlocks API routes when node-configured."""
    monkeypatch.setenv(federation_node.ENV_FEDERATION_HUB, "127.0.0.1:8744")
    monkeypatch.setenv(federation_node.ENV_FEDERATION_TOKEN, SECRET)
    monkeypatch.setenv(federation_node.ENV_NODE_NAME, NODE_NAME)

    class _FakeHandler:
        def __init__(self, headers: dict[str, str]) -> None:
            self.headers = headers

    good = _FakeHandler({federation_proto.FEDERATION_AUTH_HEADER: federation_proto.federation_token(SECRET, NODE_NAME)})
    assert server._federation_header_auth_ok(good) is True
    bad = _FakeHandler({federation_proto.FEDERATION_AUTH_HEADER: "wrong"})
    assert server._federation_header_auth_ok(bad) is False
    missing = _FakeHandler({})
    assert server._federation_header_auth_ok(missing) is False

    # Without federation env vars the header is never accepted.
    monkeypatch.delenv(federation_node.ENV_FEDERATION_HUB)
    assert server._federation_header_auth_ok(good) is False
