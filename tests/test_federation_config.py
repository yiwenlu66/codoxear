"""Federation settings: persistence, precedence, runtime apply, token rotation."""

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

from codoxear import federation_config
from codoxear import federation_hub
from codoxear import federation_node
from codoxear import federation_proto
from codoxear import server
from codoxear.auth import sign_cookie
from codoxear.server_handler import make_server_handler
from codoxear.server_main import ThreadingHTTPServer

_TEST_SECRET = b"federation-config-test-key"
_ENV_VARS = (
    "CODEX_WEB_FEDERATION_LISTEN",
    "CODEX_WEB_FEDERATION_HUB",
    "CODEX_WEB_FEDERATION_TOKEN",
    "CODEX_WEB_NODE_NAME",
)


# --------------------------------------------------------------------------- #
# Pure config-layer tests

def test_settings_file_roundtrip(tmp_path):
    path = tmp_path / "app" / "federation.json"
    settings = {
        "hub": {"enabled": True, "listen": "0.0.0.0:8744", "token": "abc123"},
        "node": {"enabled": False, "hub": "hub:8744", "token": "def456", "name": "laptop"},
    }
    federation_config.save_federation_settings(path, settings)
    loaded = federation_config.load_federation_settings(path)
    assert loaded == settings
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_load_missing_file_returns_none(tmp_path):
    assert federation_config.load_federation_settings(tmp_path / "nope.json") is None


def test_env_only_mode_without_file(monkeypatch):
    monkeypatch.setenv("CODEX_WEB_FEDERATION_LISTEN", "0.0.0.0:9744")
    monkeypatch.setenv("CODEX_WEB_FEDERATION_HUB", "hub:9744")
    monkeypatch.setenv("CODEX_WEB_FEDERATION_TOKEN", "envtoken")
    monkeypatch.setenv("CODEX_WEB_NODE_NAME", "env-node")
    config = federation_config.resolve_effective_config(None)
    assert config.source == "env"
    assert config.hub_enabled and config.hub_listen == "0.0.0.0:9744" and config.hub_token == "envtoken"
    assert config.node_enabled and config.node_hub == "hub:9744" and config.node_token == "envtoken"
    assert config.node_name == "env-node"


def test_saved_file_wins_over_env(monkeypatch):
    monkeypatch.setenv("CODEX_WEB_FEDERATION_LISTEN", "0.0.0.0:9744")
    monkeypatch.setenv("CODEX_WEB_FEDERATION_TOKEN", "envtoken")
    monkeypatch.setenv("CODEX_WEB_FEDERATION_HUB", "env-hub:8744")
    saved = {
        "hub": {"enabled": False, "listen": "127.0.0.1:8000"},
        "node": {"enabled": True, "hub": "file-hub:8744", "token": "filetoken", "name": "file-node"},
    }
    config = federation_config.resolve_effective_config(saved)
    assert config.source == "file"
    # File section present: file decides enablement; missing fields fall back
    # to env per field.
    assert config.hub_enabled is False
    assert config.hub_listen == "127.0.0.1:8000"
    assert config.hub_token == "envtoken"
    assert config.node_enabled is True
    assert config.node_hub == "file-hub:8744"
    assert config.node_token == "filetoken"
    assert config.node_name == "file-node"


def test_generate_token_format():
    token = federation_config.generate_token()
    assert len(token) == 24
    assert all(char in "0123456789abcdef" for char in token)
    assert federation_config.generate_token() != token


# --------------------------------------------------------------------------- #
# Runtime apply via the real HTTP routes

def _request(address, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    conn = http.client.HTTPConnection(address[0], address[1], timeout=10)
    headers = {"Cookie": f"{server.COOKIE_NAME}={sign_cookie({'v': 1}, secret=_TEST_SECRET)}"}
    raw = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=raw, headers=headers)
    response = conn.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    conn.close()
    return response.status, payload


def _wait_for(predicate, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(0.05)
    return False


@pytest.fixture
def fed_server(tmp_path, monkeypatch):
    """Real HTTP server with federation config isolated to tmp paths."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    app = tmp_path / "app"
    (app / "socks").mkdir(parents=True)
    monkeypatch.setattr(server, "FEDERATION_CONFIG_PATH", app / "federation.json")
    monkeypatch.setattr(server, "SOCK_DIR", app / "socks")
    monkeypatch.setattr(server, "HMAC_SECRET", _TEST_SECRET)
    handler = make_server_handler(server)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield ("127.0.0.1", httpd.server_address[1])
    finally:
        server._stop_federation()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def test_runtime_apply_and_disable(fed_server, tmp_path):
    address = fed_server
    # Enable the hub from the UI route: listener starts without a restart.
    status, payload = _request(address, "POST", "/api/federation", {
        "hub": {"enabled": True, "listen": "127.0.0.1:0", "token": "hub-token-1"},
        "node": {"enabled": False},
    })
    assert status == 200, payload
    assert payload["ok"] is True
    listening = payload["status"]["hub"]["listening"]
    assert listening and listening.startswith("127.0.0.1:")
    hub_port = int(listening.rsplit(":", 1)[1])
    # The config file was persisted.
    saved = federation_config.load_federation_settings(server.FEDERATION_CONFIG_PATH)
    assert saved["hub"] == {"enabled": True, "listen": "127.0.0.1:0", "token": "hub-token-1"}

    # Connect this same server as a node of its own hub via saved config.
    status, payload = _request(address, "POST", "/api/federation", {
        "hub": {"enabled": True, "listen": "127.0.0.1:0"},
        "node": {"enabled": True, "hub": f"127.0.0.1:{hub_port}", "token": "hub-token-1", "name": "cfg-node"},
    })
    assert status == 200, payload

    def node_connected() -> bool:
        _, current = _request(address, "GET", "/api/federation")
        return current["status"]["node"]["state"] == "connected"

    assert _wait_for(node_connected)
    assert _wait_for(lambda: [row["name"] for row in server._federation_node_rows()] == ["cfg-node"])

    # Disable the node: disconnect and stay disconnected.
    status, payload = _request(address, "POST", "/api/federation", {
        "hub": {"enabled": True, "listen": "127.0.0.1:0"},
        "node": {"enabled": False},
    })
    assert status == 200, payload
    assert payload["status"]["node"]["state"] == "off"
    assert _wait_for(lambda: server._federation_node_rows() == [])

    # Disable the hub: the listener closes.
    status, payload = _request(address, "POST", "/api/federation", {
        "hub": {"enabled": False},
        "node": {"enabled": False},
    })
    assert status == 200, payload
    assert payload["status"]["hub"]["listening"] is None

    def port_closed() -> bool:
        try:
            sock = socket.create_connection(("127.0.0.1", hub_port), timeout=0.5)
        except OSError:
            return True
        sock.close()
        return False

    assert _wait_for(port_closed)


def test_token_rotation_rotates_auth(fed_server):
    address = fed_server
    status, payload = _request(address, "POST", "/api/federation", {
        "hub": {"enabled": True, "listen": "127.0.0.1:0", "token": "token-old"},
        "node": {"enabled": False},
    })
    assert status == 200, payload
    hub_addr = payload["status"]["hub"]["listening"]
    host, _, port = hub_addr.rpartition(":")

    # A raw client with the current token is welcomed.
    conn = socket.create_connection((host, int(port)), timeout=5)
    federation_proto.send_frame(conn, federation_proto.hello_message(node_name="n1", secret="token-old", http_port=9999))
    assert federation_proto.recv_frame(conn)["type"] == federation_proto.WELCOME
    conn.close()

    # Rotate: fresh token returned and persisted, hub restarted under it.
    status, payload = _request(address, "POST", "/api/federation/token", {})
    assert status == 200, payload
    new_token = payload["token"]
    assert new_token != "token-old" and len(new_token) == 24
    saved = federation_config.load_federation_settings(server.FEDERATION_CONFIG_PATH)
    assert saved["hub"]["token"] == new_token
    # The saved listen was 127.0.0.1:0, so the restarted hub is on a fresh port.
    hub_addr = payload["status"]["hub"]["listening"]
    host, _, port = hub_addr.rpartition(":")

    # Old token rejected (connection closed without welcome), new accepted.
    conn = socket.create_connection((host, int(port)), timeout=5)
    federation_proto.send_frame(conn, federation_proto.hello_message(node_name="n1", secret="token-old", http_port=9999))
    with pytest.raises((ConnectionError, OSError)):
        federation_proto.recv_frame(conn)
    conn.close()
    conn = socket.create_connection((host, int(port)), timeout=5)
    federation_proto.send_frame(conn, federation_proto.hello_message(node_name="n1", secret=new_token, http_port=9999))
    assert federation_proto.recv_frame(conn)["type"] == federation_proto.WELCOME
    conn.close()


def test_hub_bind_conflict_returns_409(fed_server):
    address = fed_server
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    taken_port = blocker.getsockname()[1]
    try:
        status, payload = _request(address, "POST", "/api/federation", {
            "hub": {"enabled": True, "listen": f"127.0.0.1:{taken_port}", "token": "tok"},
            "node": {"enabled": False},
        })
        assert status == 409, payload
        assert payload["ok"] is False
        assert "bind" in payload["error"] or "address" in payload["error"].lower()
        assert payload["status"]["hub"]["listening"] is None
        # The config was still persisted so the saved state survives restart.
        saved = federation_config.load_federation_settings(server.FEDERATION_CONFIG_PATH)
        assert saved["hub"]["enabled"] is True
        assert saved["hub"]["listen"] == f"127.0.0.1:{taken_port}"
    finally:
        blocker.close()


def test_enabling_hub_without_token_mints_one(fed_server):
    address = fed_server
    status, payload = _request(address, "POST", "/api/federation", {
        "hub": {"enabled": True, "listen": "127.0.0.1:0"},
        "node": {"enabled": False},
    })
    assert status == 200, payload
    token = payload["hub"]["token"]
    assert len(token) == 24


def test_get_federation_requires_auth(fed_server):
    conn = http.client.HTTPConnection(fed_server[0], fed_server[1], timeout=10)
    conn.request("GET", "/api/federation")
    response = conn.getresponse()
    response.read()
    conn.close()
    assert response.status == 401
