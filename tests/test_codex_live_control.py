from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest

from codoxear import broker as broker_module
from codoxear.broker_control import _handle_broker_control_connection
from codoxear.broker_turn_state import State
from codoxear.codex_live_control import CodexLiveControlError
from codoxear.codex_live_control import codex_app_server_config_args
from codoxear.codex_live_control import codex_live_control_compatible_args
from codoxear.codex_live_control import codex_live_settings_supported
from codoxear.codex_live_control import start_codex_app_server
from codoxear.codex_live_control import stop_codex_app_server
from codoxear.codex_live_control import update_codex_thread_settings


class FakeWebSocket:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []
        self.closed = False

    def send(self, raw):
        self.sent.append(json.loads(raw))

    def recv(self, *, timeout):
        assert timeout > 0
        return json.dumps(self.responses.pop(0))

    def close(self):
        self.closed = True


def _connect_with(websocket):
    def connect(path, *, timeout_s):
        assert isinstance(path, Path)
        assert timeout_s > 0
        return websocket

    return connect


def test_app_server_launch_copies_only_shared_config_flags() -> None:
    assert codex_live_control_compatible_args(["-c", 'model_provider="custom"', "--model", "gpt-5.4"])
    assert not codex_live_control_compatible_args(["--profile", "work"])
    assert not codex_live_control_compatible_args(["--oss"])
    assert not codex_live_control_compatible_args(["--remote", "unix:///tmp/other.sock"])
    assert codex_app_server_config_args(
        [
            "-c", "model_provider=\"custom\"",
            "--disable", "goals",
            "--model", "gpt-5.4",
            "--dangerously-bypass-approvals-and-sandbox",
            "resume", "thread-1",
        ]
    ) == ["-c", "model_provider=\"custom\"", "--disable", "goals"]


def test_broker_keeps_codex_app_server_socket_outside_discovery_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sock_dir = tmp_path / "socks"
    captured: dict[str, Path] = {}

    monkeypatch.setattr(broker_module, "AGENT_BACKEND", "codex")
    monkeypatch.setattr(broker_module, "SOCK_DIR", sock_dir)
    monkeypatch.setattr(
        broker_module,
        "start_codex_app_server",
        lambda **kwargs: (captured.setdefault("socket_path", kwargs["socket_path"]) and object(), None),
    )

    live_broker = broker_module.Broker(cwd=str(tmp_path), codex_args=[])
    live_broker._prepare_codex_live_control()

    assert captured["socket_path"].parent == sock_dir / "private"
    assert captured["socket_path"].name.startswith("codex-app-server-")
    assert live_broker.codex_args[:2] == ["--remote", f"unix://{captured['socket_path']}"]


def test_broker_settings_command_targets_bound_thread_without_pty_keys(tmp_path: Path) -> None:
    left, right = socket.socketpair()
    state = State(
        codex_pid=1,
        pty_master_fd=2,
        cwd=str(tmp_path),
        start_ts=0.0,
        codex_home=tmp_path,
        sessions_dir=tmp_path,
        session_id="thread-live",
    )
    updates = []

    worker = threading.Thread(
        target=_handle_broker_control_connection,
        args=(left,),
        kwargs={
            "lock": threading.Lock(),
            "get_state": lambda: state,
            "seq_bytes": lambda _raw: b"",
            "encode_enter": lambda: b"\r",
            "write_all": lambda _fd, _data: None,
            "inject": lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("settings must not touch PTY")),
            "now": lambda: 1.0,
            "teardown_managed_process_group": lambda: None,
            "update_codex_settings": lambda **kwargs: updates.append(kwargs) or {"ok": True, **kwargs},
        },
    )
    worker.start()
    right.sendall(b'{"cmd":"settings","model":"gpt-5.4"}\n')
    response = json.loads(right.makefile("rb").readline())
    worker.join(timeout=2)
    right.close()

    assert updates == [{"thread_id": "thread-live", "model": "gpt-5.4", "effort": None}]
    assert response == {"ok": True, "thread_id": "thread-live", "model": "gpt-5.4", "effort": None}


def test_update_codex_thread_settings_negotiates_experimental_api_and_waits_for_ack() -> None:
    websocket = FakeWebSocket(
        [
            {"id": 1, "result": {"userAgent": "probe"}},
            {"method": "configWarning", "params": {"message": "ignored interleaved notice"}},
            {"id": 2, "result": {}},
        ]
    )

    result = update_codex_thread_settings(
        Path("/tmp/codex.sock"),
        thread_id="thread-1",
        model="gpt-5.4",
        connect=_connect_with(websocket),
    )

    assert result == {"ok": True, "model": "gpt-5.4", "effort": None}
    assert websocket.sent == [
        {
            "method": "initialize",
            "id": 1,
            "params": {
                "clientInfo": {"name": "codoxear", "title": "Codoxear", "version": "0.1.0"},
                "capabilities": {"experimentalApi": True},
            },
        },
        {"method": "initialized", "params": {}},
        {
            "method": "thread/settings/update",
            "id": 2,
            "params": {"threadId": "thread-1", "model": "gpt-5.4"},
        },
    ]
    assert websocket.closed is True


def test_update_codex_thread_settings_surfaces_protocol_rejection() -> None:
    websocket = FakeWebSocket(
        [
            {"id": 1, "result": {}},
            {"id": 2, "error": {"code": -32600, "message": "unsupported effort"}},
        ]
    )
    with pytest.raises(CodexLiveControlError, match="unsupported effort"):
        update_codex_thread_settings(
            Path("/tmp/codex.sock"),
            thread_id="thread-1",
            effort="high",
            connect=_connect_with(websocket),
        )


def test_capability_probe_distinguishes_supported_method_from_method_not_found() -> None:
    supported = FakeWebSocket(
        [
            {"id": 1, "result": {}},
            {"id": 2, "error": {"code": -32600, "message": "no rollout found for thread id"}},
        ]
    )
    unsupported = FakeWebSocket(
        [
            {"id": 1, "result": {}},
            {"id": 2, "error": {"code": -32601, "message": "Method not found"}},
        ]
    )

    assert codex_live_settings_supported(Path("/tmp/supported.sock"), connect=_connect_with(supported)) is True
    assert codex_live_settings_supported(Path("/tmp/unsupported.sock"), connect=_connect_with(unsupported)) is False


def test_start_codex_app_server_uses_private_unix_socket_and_remote_capability_probe(tmp_path: Path) -> None:
    socket_path = tmp_path / "codex-control.sock"
    calls = []

    class FakeProcess:
        pid = 12345

        def poll(self):
            return 0

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        socket_path.touch()
        process = FakeProcess()
        process.poll = lambda: None
        return process

    server, error = start_codex_app_server(
        agent_bin="codex-custom",
        cwd=str(tmp_path),
        codex_home=tmp_path / "home",
        socket_path=socket_path,
        shell_argv_for_command=lambda command: ["shell", "-c", command],
        config_args=["-c", 'model_provider="custom"'],
        environ={"PATH": "/bin"},
        popen=popen,
        probe=lambda path, **_kwargs: path == socket_path,
    )

    assert error is None
    assert server is not None
    argv, kwargs = calls[0]
    assert argv[:2] == ["shell", "-c"]
    assert "codex-custom -c" in argv[2]
    assert 'model_provider=\"custom\"' in argv[2]
    assert "app-server --listen" in argv[2]
    assert f"unix://{socket_path}" in argv[2]
    assert kwargs["env"]["CODEX_HOME"] == str(tmp_path / "home")
    server.process.poll = lambda: 0
    stop_codex_app_server(server)
    assert not socket_path.exists()
