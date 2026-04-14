import json
import signal
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from codoxear.agent_backend import get_agent_backend
from codoxear.broker import Broker
from codoxear.pi_broker import PiBroker
from codoxear.pi_broker import State as PiBrokerState
from codoxear.pi_broker import _tail_delta


PI_ASK_USER_BRIDGE_PATH = (
    Path(__file__).resolve().parents[1]
    / "codoxear"
    / "pi_extensions"
    / "ask_user_bridge.ts"
).resolve()


def _recv_line(sock: socket.socket) -> bytes:
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
    return buf.split(b"\n", 1)[0]


def _roundtrip_json(broker: PiBroker, payload: dict[str, object]) -> dict[str, object]:
    server_sock, client_sock = socket.socketpair()
    try:
        thread = threading.Thread(
            target=broker._handle_conn, args=(server_sock,), daemon=True
        )
        thread.start()
        client_sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        resp = json.loads(_recv_line(client_sock).decode("utf-8"))
        thread.join(1.0)
        return resp
    finally:
        client_sock.close()


class TestPiLaunchDelegation(unittest.TestCase):
    def test_broker_run_delegates_pi_launches_to_pi_broker_with_session_path(
        self,
    ) -> None:
        fake_stdin = SimpleNamespace(isatty=lambda: False, fileno=lambda: 9)
        with (
            tempfile.TemporaryDirectory() as td,
            patch("codoxear.broker.sys.stdin", fake_stdin),
            patch.dict("os.environ", {"PI_HOME": td}, clear=False),
            patch("codoxear.broker.AGENT_BACKEND", "pi"),
            patch("codoxear.broker.BACKEND", get_agent_backend("pi")),
        ):
            sessions_dir = get_agent_backend("pi").sessions_dir()
            sessions_dir.mkdir(parents=True, exist_ok=True)
            session_path = sessions_dir / "resume.jsonl"
            session_path.write_text(
                '{"type":"session","id":"resume-a","cwd":"/tmp/pi-work"}\n',
                encoding="utf-8",
            )
            broker = Broker(
                cwd="/tmp/pi-work", codex_args=["--session", str(session_path)]
            )

            with (
                patch("codoxear.broker.PiBroker") as pi_broker_cls,
                patch(
                    "codoxear.broker._term_size",
                    side_effect=AssertionError(
                        "PTY launch path should be skipped for Pi"
                    ),
                ),
                patch(
                    "codoxear.broker._require_proc",
                    side_effect=AssertionError(
                        "PTY launch path should be skipped for Pi"
                    ),
                ),
                patch(
                    "codoxear.broker.pty.fork",
                    side_effect=AssertionError(
                        "PTY launch path should be skipped for Pi"
                    ),
                ),
            ):
                pi_broker_cls.return_value.run.return_value = 23

                exit_code = broker.run()

        pi_broker_cls.assert_called_once_with(
            cwd="/tmp/pi-work",
            session_path=session_path.resolve(),
            agent_args=["-e", str(PI_ASK_USER_BRIDGE_PATH)],
            resume_session_id="resume-a",
        )
        pi_broker_cls.return_value.run.assert_called_once_with(foreground=True)
        self.assertEqual(exit_code, 23)

    def test_broker_run_forwards_non_session_pi_args_to_pi_broker(self) -> None:
        fake_stdin = SimpleNamespace(isatty=lambda: False, fileno=lambda: 9)
        with (
            tempfile.TemporaryDirectory() as td,
            patch("codoxear.broker.sys.stdin", fake_stdin),
            patch.dict("os.environ", {"PI_HOME": td}, clear=False),
            patch("codoxear.broker.AGENT_BACKEND", "pi"),
            patch("codoxear.broker.BACKEND", get_agent_backend("pi")),
        ):
            sessions_dir = get_agent_backend("pi").sessions_dir()
            sessions_dir.mkdir(parents=True, exist_ok=True)
            session_path = sessions_dir / "resume.jsonl"
            session_path.write_text(
                '{"type":"session","id":"resume-a","cwd":"/tmp/pi-work"}\n',
                encoding="utf-8",
            )
            broker = Broker(
                cwd="/tmp/pi-work",
                codex_args=[
                    "--session",
                    str(session_path),
                    "--model",
                    "sonnet",
                    "--fast",
                ],
            )

            with patch("codoxear.broker.PiBroker") as pi_broker_cls:
                pi_broker_cls.return_value.run.return_value = 0

                exit_code = broker.run()

        pi_broker_cls.assert_called_once_with(
            cwd="/tmp/pi-work",
            session_path=session_path.resolve(),
            agent_args=[
                "-e",
                str(PI_ASK_USER_BRIDGE_PATH),
                "--model",
                "sonnet",
                "--fast",
            ],
            resume_session_id="resume-a",
        )
        pi_broker_cls.return_value.run.assert_called_once_with(foreground=True)
        self.assertEqual(exit_code, 0)

    def test_broker_run_preserves_resume_id_session_arg_for_pi_broker(self) -> None:
        fake_stdin = SimpleNamespace(isatty=lambda: False, fileno=lambda: 9)
        with (
            tempfile.TemporaryDirectory() as td,
            patch("codoxear.broker.sys.stdin", fake_stdin),
            patch.dict("os.environ", {"PI_HOME": td}, clear=False),
            patch("codoxear.broker.AGENT_BACKEND", "pi"),
            patch("codoxear.broker.BACKEND", get_agent_backend("pi")),
        ):
            sessions_dir = get_agent_backend("pi").sessions_dir()
            sessions_dir.mkdir(parents=True, exist_ok=True)
            session_dir = sessions_dir / "--tmp-pi-work--"
            session_dir.mkdir(parents=True, exist_ok=True)
            session_path = session_dir / "resume.jsonl"
            session_path.write_text(
                '{"type":"session","id":"resume-a","cwd":"/tmp/pi-work"}\n',
                encoding="utf-8",
            )
            broker = Broker(
                cwd="/tmp/pi-work", codex_args=["--session", "resume-a", "--fast"]
            )

            with patch("codoxear.broker.PiBroker") as pi_broker_cls:
                pi_broker_cls.return_value.run.return_value = 0

                exit_code = broker.run()

        pi_broker_cls.assert_called_once_with(
            cwd="/tmp/pi-work",
            session_path=session_path,
            agent_args=["-e", str(PI_ASK_USER_BRIDGE_PATH), "--fast"],
            resume_session_id="resume-a",
        )
        pi_broker_cls.return_value.run.assert_called_once_with(foreground=True)
        self.assertEqual(exit_code, 0)

    def test_broker_run_resolves_resume_id_from_custom_session_dir(self) -> None:
        fake_stdin = SimpleNamespace(isatty=lambda: False, fileno=lambda: 9)
        with (
            tempfile.TemporaryDirectory() as td,
            patch("codoxear.broker.sys.stdin", fake_stdin),
            patch.dict("os.environ", {"PI_HOME": td}, clear=False),
            patch("codoxear.broker.AGENT_BACKEND", "pi"),
            patch("codoxear.broker.BACKEND", get_agent_backend("pi")),
        ):
            custom_dir = Path(td) / "custom-sessions"
            custom_dir.mkdir(parents=True, exist_ok=True)
            session_path = custom_dir / "resume.jsonl"
            session_path.write_text(
                '{"type":"session","id":"resume-a","cwd":"/tmp/pi-work"}\n',
                encoding="utf-8",
            )
            broker = Broker(
                cwd="/tmp/pi-work",
                codex_args=[
                    "--session",
                    "resume-a",
                    "--session-dir",
                    str(custom_dir),
                    "--fast",
                ],
            )

            with patch("codoxear.broker.PiBroker") as pi_broker_cls:
                pi_broker_cls.return_value.run.return_value = 0

                exit_code = broker.run()

        pi_broker_cls.assert_called_once_with(
            cwd="/tmp/pi-work",
            session_path=session_path,
            agent_args=["-e", str(PI_ASK_USER_BRIDGE_PATH), "--fast"],
            resume_session_id="resume-a",
        )
        pi_broker_cls.return_value.run.assert_called_once_with(foreground=True)
        self.assertEqual(exit_code, 0)

    def test_broker_run_preserves_no_session_flag_for_pi_broker(self) -> None:
        fake_stdin = SimpleNamespace(isatty=lambda: False, fileno=lambda: 9)
        with (
            patch("codoxear.broker.sys.stdin", fake_stdin),
            patch("codoxear.broker.AGENT_BACKEND", "pi"),
            patch("codoxear.broker.BACKEND", get_agent_backend("pi")),
        ):
            broker = Broker(cwd="/tmp/pi-work", codex_args=["--no-session", "--fast"])

            with patch("codoxear.broker.PiBroker") as pi_broker_cls:
                pi_broker_cls.return_value.run.return_value = 0

                exit_code = broker.run()

        pi_broker_cls.assert_called_once_with(
            cwd="/tmp/pi-work",
            session_path=None,
            agent_args=["-e", str(PI_ASK_USER_BRIDGE_PATH), "--no-session", "--fast"],
            resume_session_id=None,
        )
        pi_broker_cls.return_value.run.assert_called_once_with(foreground=True)
        self.assertEqual(exit_code, 0)

    def test_broker_run_forwards_external_session_log_path_to_pi_broker(self) -> None:
        fake_stdin = SimpleNamespace(isatty=lambda: False, fileno=lambda: 9)
        with (
            tempfile.TemporaryDirectory() as td,
            patch("codoxear.broker.sys.stdin", fake_stdin),
            patch("codoxear.broker.AGENT_BACKEND", "pi"),
            patch("codoxear.broker.BACKEND", get_agent_backend("pi")),
        ):
            session_path = Path(td) / "custom-session.jsonl"
            session_path.write_text(
                '{"type":"session","id":"resume-a","cwd":"/tmp/pi-work"}\n',
                encoding="utf-8",
            )
            broker = Broker(
                cwd="/tmp/pi-work",
                codex_args=["--session", str(session_path), "--fast"],
            )

            with patch("codoxear.broker.PiBroker") as pi_broker_cls:
                pi_broker_cls.return_value.run.return_value = 0

                exit_code = broker.run()

        pi_broker_cls.assert_called_once_with(
            cwd="/tmp/pi-work",
            session_path=session_path.resolve(),
            agent_args=["-e", str(PI_ASK_USER_BRIDGE_PATH), "--fast"],
            resume_session_id="resume-a",
        )
        pi_broker_cls.return_value.run.assert_called_once_with(foreground=True)
        self.assertEqual(exit_code, 0)

    def test_broker_run_resolves_relative_session_log_path_against_broker_cwd(
        self,
    ) -> None:
        fake_stdin = SimpleNamespace(isatty=lambda: False, fileno=lambda: 9)
        with (
            tempfile.TemporaryDirectory() as td,
            patch("codoxear.broker.sys.stdin", fake_stdin),
            patch.dict("os.environ", {"PI_HOME": td}, clear=False),
            patch("codoxear.broker.AGENT_BACKEND", "pi"),
            patch("codoxear.broker.BACKEND", get_agent_backend("pi")),
        ):
            cwd = Path(td) / "project"
            cwd.mkdir(parents=True, exist_ok=True)
            session_path = cwd / "relative-session.jsonl"
            session_path.write_text(
                '{"type":"session","id":"resume-a","cwd":"/tmp/pi-work"}\n',
                encoding="utf-8",
            )
            broker = Broker(
                cwd=str(cwd), codex_args=["--session", "relative-session.jsonl"]
            )

            with patch("codoxear.broker.PiBroker") as pi_broker_cls:
                pi_broker_cls.return_value.run.return_value = 0

                exit_code = broker.run()

        pi_broker_cls.assert_called_once_with(
            cwd=str(cwd),
            session_path=session_path.resolve(),
            agent_args=["-e", str(PI_ASK_USER_BRIDGE_PATH)],
            resume_session_id="resume-a",
        )
        pi_broker_cls.return_value.run.assert_called_once_with(foreground=True)
        self.assertEqual(exit_code, 0)


class _FakeRpc:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.state = {"busy": False, "history_len": 2, "session_id": "pi-session-001"}
        self.state_error: Exception | None = None
        self.events: list[dict[str, object]] = []
        self.stderr_lines: list[str] = []
        self.ui_responses: list[dict[str, object]] = []
        self.closed = False

    def get_state(self) -> dict[str, object]:
        self.calls.append(("get_state", None))
        if self.state_error is not None:
            raise self.state_error
        return dict(self.state)

    def prompt(
        self, text: str, *, streaming_behavior: str | None = None
    ) -> dict[str, object]:
        self.calls.append(
            ("prompt", {"text": text, "streaming_behavior": streaming_behavior})
        )
        self.state["busy"] = True
        return {"turn_id": "turn-001", "queued": False}

    def abort(self, turn_id: str | None = None) -> dict[str, object]:
        self.calls.append(("abort", turn_id))
        self.state["busy"] = False
        return {"status": "aborted", "turn_id": turn_id}

    def send_ui_response(self, request_id: str, **payload: object) -> None:
        self.ui_responses.append({"id": request_id, **payload})

    def close(self) -> None:
        self.calls.append(("close", None))
        self.closed = True

    def drain_events(self) -> list[dict[str, object]]:
        events = list(self.events)
        self.events.clear()
        return events

    def drain_stderr_lines(self) -> list[str]:
        lines = list(self.stderr_lines)
        self.stderr_lines.clear()
        return lines


class _BlockingPromptRpc(_FakeRpc):
    def __init__(self) -> None:
        super().__init__()
        self.prompt_started = threading.Event()
        self.release_prompt = threading.Event()

    def prompt(
        self, text: str, *, streaming_behavior: str | None = None
    ) -> dict[str, object]:
        self.calls.append(
            ("prompt", {"text": text, "streaming_behavior": streaming_behavior})
        )
        self.prompt_started.set()
        self.release_prompt.wait(1.0)
        self.state["busy"] = True
        return {"turn_id": "turn-001", "queued": False}


class _BlockingAbortRpc(_FakeRpc):
    def __init__(self) -> None:
        super().__init__()
        self.abort_started = threading.Event()
        self.release_abort = threading.Event()

    def abort(self, turn_id: str | None = None) -> dict[str, object]:
        self.calls.append(("abort", turn_id))
        self.abort_started.set()
        self.release_abort.wait(1.0)
        self.state["busy"] = False
        return {"status": "aborted", "turn_id": turn_id}


class _FlakyUiResponseRpc(_FakeRpc):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_ui_response = True

    def send_ui_response(self, request_id: str, **payload: object) -> None:
        if self.fail_next_ui_response:
            self.fail_next_ui_response = False
            raise RuntimeError("send failed")
        super().send_ui_response(request_id, **payload)


class _FailingAbortRpc(_FakeRpc):
    def abort(self, turn_id: str | None = None) -> dict[str, object]:
        self.calls.append(("abort", turn_id))
        raise RuntimeError("abort failed")


class _ExitedProcRpc(_FakeRpc):
    def __init__(self, exit_code: int) -> None:
        super().__init__()
        self.pid = 4321
        self._proc = SimpleNamespace(poll=lambda: exit_code)


class TestPiBroker(unittest.TestCase):
    def test_write_meta_marks_rpc_transport_and_live_ui_support(self) -> None:
        rpc = _FakeRpc()
        with tempfile.TemporaryDirectory() as td:
            sock_path = Path(td) / "pi.sock"
            broker = PiBroker(cwd="/tmp")
            broker.state = PiBrokerState(
                session_id="pi-session-001",
                codex_pid=123,
                sock_path=sock_path,
                session_path=Path(td) / "pi-session.jsonl",
                start_ts=0.0,
                rpc=rpc,
            )

            broker._write_meta()

            meta = json.loads(
                sock_path.with_suffix(".json").read_text(encoding="utf-8")
            )

        self.assertEqual(meta["transport"], "pi-rpc")
        self.assertTrue(meta["supports_live_ui"])
        self.assertEqual(meta["ui_protocol_version"], 1)

    def test_write_meta_preserves_resume_session_id_for_agent_managed_sessions(
        self,
    ) -> None:
        rpc = _FakeRpc()
        with tempfile.TemporaryDirectory() as td:
            sock_path = Path(td) / "pi.sock"
            broker = PiBroker(
                cwd="/tmp", agent_args=["--session", "resume-a", "--fast"]
            )
            broker.state = PiBrokerState(
                session_id="pi-session-001",
                codex_pid=123,
                sock_path=sock_path,
                session_path=None,
                start_ts=0.0,
                rpc=rpc,
            )

            broker._write_meta()

            meta = json.loads(
                sock_path.with_suffix(".json").read_text(encoding="utf-8")
            )

        self.assertEqual(meta["resume_session_id"], "resume-a")
        self.assertNotIn("session_path", meta)

    def test_sync_state_clears_resume_session_id_once_broker_is_idle(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp", rpc=rpc, resume_session_id="resume-a")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            busy=False,
            last_turn_id=None,
        )

        with patch.object(broker, "_write_meta") as write_meta:
            broker._sync_state_from_rpc()

        self.assertIsNone(broker.resume_session_id)
        write_meta.assert_called_once()

    def test_write_meta_uses_pi_session_path_sidecar_field(self) -> None:
        rpc = _FakeRpc()
        with tempfile.TemporaryDirectory() as td:
            sock_path = Path(td) / "pi.sock"
            session_path = Path(td) / "pi-session.jsonl"
            broker = PiBroker(cwd="/tmp")
            broker.state = PiBrokerState(
                session_id="pi-session-001",
                codex_pid=123,
                sock_path=sock_path,
                session_path=session_path,
                start_ts=0.0,
                rpc=rpc,
            )

            broker._write_meta()

            meta = json.loads(
                sock_path.with_suffix(".json").read_text(encoding="utf-8")
            )

        self.assertIsNone(meta["log_path"])
        self.assertEqual(meta["session_path"], str(session_path))
        self.assertTrue(meta["supports_web_control"])

    def test_run_foreground_mode_writes_meta_without_pty_wrapper(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp", rpc=rpc)

        with (
            tempfile.TemporaryDirectory() as td,
            patch("codoxear.pi_broker.SOCK_DIR", Path(td)),
            patch("codoxear.pi_broker.PI_SESSION_DIR", Path(td)),
            patch.object(PiBroker, "_bg_sync_loop", side_effect=lambda: None),
            patch.object(PiBroker, "_sock_server", side_effect=lambda: None),
        ):
            exit_code = broker.run(foreground=False)
            assert broker.state is not None
            meta = json.loads(
                broker.state.sock_path.with_suffix(".json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 0)
        self.assertIsNotNone(broker.state)
        assert broker.state is not None
        self.assertEqual(broker.state.backend, "pi")
        self.assertEqual(meta["transport"], "pi-rpc")
        self.assertTrue(meta["supports_live_ui"])

    def test_run_returns_rpc_exit_code_when_rpc_process_exits(self) -> None:
        rpc = _ExitedProcRpc(exit_code=17)
        broker = PiBroker(cwd="/tmp", rpc=rpc)

        with (
            tempfile.TemporaryDirectory() as td,
            patch("codoxear.pi_broker.SOCK_DIR", Path(td)),
            patch("codoxear.pi_broker.PI_SESSION_DIR", Path(td)),
            patch.object(PiBroker, "_bg_sync_loop", side_effect=lambda: None),
            patch.object(
                PiBroker, "_sock_server", side_effect=lambda: broker._stop.wait(0.1)
            ),
        ):
            exit_code = broker.run(foreground=False)

        self.assertEqual(exit_code, 17)

    def test_run_without_generated_session_path_when_agent_args_manage_session(
        self,
    ) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp", rpc=rpc, agent_args=["--no-session"])

        with (
            tempfile.TemporaryDirectory() as td,
            patch("codoxear.pi_broker.SOCK_DIR", Path(td)),
            patch("codoxear.pi_broker.PI_SESSION_DIR", Path(td)),
            patch.object(PiBroker, "_bg_sync_loop", side_effect=lambda: None),
            patch.object(PiBroker, "_sock_server", side_effect=lambda: None),
        ):
            exit_code = broker.run(foreground=False)

        self.assertEqual(exit_code, 0)
        self.assertIsNone(broker.state.session_path)

    def test_write_meta_disables_web_control_for_no_session_launches(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp", rpc=rpc, agent_args=["--no-session"])

        with (
            tempfile.TemporaryDirectory() as td,
            patch("codoxear.pi_broker.SOCK_DIR", Path(td)),
            patch("codoxear.pi_broker.PI_SESSION_DIR", Path(td)),
            patch.object(PiBroker, "_bg_sync_loop", side_effect=lambda: None),
            patch.object(PiBroker, "_sock_server", side_effect=lambda: None),
        ):
            broker.run(foreground=False)
            assert broker.state is not None
            meta = json.loads(
                broker.state.sock_path.with_suffix(".json").read_text(encoding="utf-8")
            )

        self.assertFalse(meta["supports_web_control"])
        self.assertNotIn("session_path", meta)

    def test_state_returns_busy_queue_and_token(self) -> None:
        rpc = _FakeRpc()
        rpc.state["busy"] = True
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            busy=True,
            token={"completion_tokens": 12},
        )
        server_sock, client_sock = socket.socketpair()
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "state"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(
                resp, {"busy": True, "queue_len": 0, "token": {"completion_tokens": 12}}
            )
        finally:
            client_sock.close()

    def test_ui_state_returns_pending_requests_after_event_drain(self) -> None:
        rpc = _FakeRpc()
        rpc.events = [
            {
                "type": "extension_ui_request",
                "id": "ui-req-1",
                "method": "select",
                "title": "Pick a location",
                "message": "Where should this go?",
                "question": "Where should this go?",
                "context": "Choose one or type your own.",
                "options": ["Details", "Sidebar"],
                "allowMultiple": True,
                "timeout": 10000,
            }
        ]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )

        broker._sync_output_from_rpc()

        resp = _roundtrip_json(broker, {"cmd": "ui_state"})

        self.assertEqual(
            resp,
            {
                "requests": [
                    {
                        "id": "ui-req-1",
                        "method": "select",
                        "title": "Pick a location",
                        "message": "Where should this go?",
                        "question": "Where should this go?",
                        "context": "Choose one or type your own.",
                        "options": ["Details", "Sidebar"],
                        "allow_freeform": True,
                        "allow_multiple": True,
                        "timeout_ms": 10000,
                        "status": "pending",
                    }
                ]
            },
        )

    def test_ui_state_accepts_snake_case_ui_request_flags(self) -> None:
        rpc = _FakeRpc()
        rpc.events = [
            {
                "type": "extension_ui_request",
                "id": "ui-req-2",
                "method": "select",
                "question": "Pick destinations",
                "context": "Multiple answers are allowed.",
                "options": ["Details", "Sidebar"],
                "allow_freeform": False,
                "allow_multiple": True,
            }
        ]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )

        broker._sync_output_from_rpc()

        resp = _roundtrip_json(broker, {"cmd": "ui_state"})

        self.assertEqual(
            resp,
            {
                "requests": [
                    {
                        "id": "ui-req-2",
                        "method": "select",
                        "title": None,
                        "message": None,
                        "question": "Pick destinations",
                        "context": "Multiple answers are allowed.",
                        "options": ["Details", "Sidebar"],
                        "allow_freeform": False,
                        "allow_multiple": True,
                        "timeout_ms": None,
                        "status": "pending",
                    }
                ]
            },
        )

    def test_ui_response_forwards_value_once_and_rejects_replay(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            pending_ui_requests={
                "ui-req-1": {
                    "id": "ui-req-1",
                    "method": "select",
                    "title": "Pick a location",
                    "message": "Where should this go?",
                    "options": ["Details", "Sidebar"],
                    "timeout_ms": 10000,
                    "status": "pending",
                }
            },
        )

        first = _roundtrip_json(
            broker, {"cmd": "ui_response", "id": "ui-req-1", "value": "Details"}
        )
        second = _roundtrip_json(
            broker, {"cmd": "ui_response", "id": "ui-req-1", "value": "Sidebar"}
        )

        self.assertEqual(first, {"ok": True})
        self.assertEqual(rpc.ui_responses, [{"id": "ui-req-1", "value": "Details"}])
        self.assertEqual(second, {"error": "request already resolved"})
        self.assertEqual(
            broker.state.pending_ui_requests["ui-req-1"]["status"], "resolved"
        )

    def test_ui_state_ignores_fire_and_forget_ui_requests(self) -> None:
        rpc = _FakeRpc()
        rpc.events = [
            {
                "type": "extension_ui_request",
                "id": "ui-notify-1",
                "method": "notify",
                "message": "Indexed 88 searchable tools and skills",
            },
            {
                "type": "extension_ui_request",
                "id": "ui-status-1",
                "method": "setStatus",
                "statusKey": "my-ext",
                "statusText": "Running",
            },
            {
                "type": "extension_ui_request",
                "id": "ui-select-1",
                "method": "select",
                "question": "Pick destinations",
                "options": ["Details", "Sidebar"],
            },
        ]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )

        broker._sync_output_from_rpc()

        resp = _roundtrip_json(broker, {"cmd": "ui_state"})

        self.assertEqual(
            resp,
            {
                "requests": [
                    {
                        "id": "ui-select-1",
                        "method": "select",
                        "title": None,
                        "message": None,
                        "question": "Pick destinations",
                        "context": None,
                        "options": ["Details", "Sidebar"],
                        "allow_freeform": True,
                        "allow_multiple": False,
                        "timeout_ms": None,
                        "status": "pending",
                    }
                ]
            },
        )

    def test_ui_response_rolls_back_resolution_when_send_fails(self) -> None:
        rpc = _FlakyUiResponseRpc()
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            pending_ui_requests={
                "ui-req-1": {
                    "id": "ui-req-1",
                    "method": "select",
                    "title": "Pick a location",
                    "message": "Where should this go?",
                    "options": ["Details", "Sidebar"],
                    "timeout_ms": 10000,
                    "status": "pending",
                }
            },
        )

        first = _roundtrip_json(
            broker, {"cmd": "ui_response", "id": "ui-req-1", "value": "Details"}
        )

        self.assertEqual(first, {"error": "send failed"})
        self.assertEqual(
            broker.state.pending_ui_requests["ui-req-1"]["status"], "pending"
        )

        retry = _roundtrip_json(
            broker, {"cmd": "ui_response", "id": "ui-req-1", "value": "Details"}
        )

        self.assertEqual(retry, {"ok": True})
        self.assertEqual(rpc.ui_responses, [{"id": "ui-req-1", "value": "Details"}])

    def test_ui_response_forwards_confirmed_and_ui_state_hides_resolved_requests(
        self,
    ) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            pending_ui_requests={
                "ui-req-1": {
                    "id": "ui-req-1",
                    "method": "confirm",
                    "title": "Proceed?",
                    "message": "Continue with the action?",
                    "options": [],
                    "timeout_ms": 10000,
                    "status": "pending",
                }
            },
        )

        resp = _roundtrip_json(
            broker, {"cmd": "ui_response", "id": "ui-req-1", "confirmed": True}
        )
        ui_state = _roundtrip_json(broker, {"cmd": "ui_state"})

        self.assertEqual(resp, {"ok": True})
        self.assertEqual(rpc.ui_responses, [{"id": "ui-req-1", "confirmed": True}])
        self.assertEqual(ui_state, {"requests": []})

    def test_confirm_requests_do_not_default_to_freeform_input(self) -> None:
        rpc = _FakeRpc()
        rpc.events = [
            {
                "type": "extension_ui_request",
                "id": "ui-confirm-1",
                "method": "confirm",
                "title": "Proceed?",
                "message": "Continue with the action?",
            }
        ]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )

        broker._sync_output_from_rpc()

        self.assertEqual(
            _roundtrip_json(broker, {"cmd": "ui_state"}),
            {
                "requests": [
                    {
                        "id": "ui-confirm-1",
                        "method": "confirm",
                        "title": "Proceed?",
                        "message": "Continue with the action?",
                        "question": None,
                        "context": None,
                        "options": [],
                        "allow_freeform": False,
                        "allow_multiple": False,
                        "timeout_ms": None,
                        "status": "pending",
                    }
                ]
            },
        )

    def test_message_end_retires_pending_request_when_pi_resolves_elsewhere(
        self,
    ) -> None:
        rpc = _FakeRpc()
        rpc.events = [
            {
                "type": "extension_ui_request",
                "id": "ui-req-1",
                "method": "select",
                "question": "Pick destinations",
                "options": ["Details", "Sidebar"],
            }
        ]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )

        broker._sync_output_from_rpc()
        self.assertEqual(
            _roundtrip_json(broker, {"cmd": "ui_state"}),
            {
                "requests": [
                    {
                        "id": "ui-req-1",
                        "method": "select",
                        "title": None,
                        "message": None,
                        "question": "Pick destinations",
                        "context": None,
                        "options": ["Details", "Sidebar"],
                        "allow_freeform": True,
                        "allow_multiple": False,
                        "timeout_ms": None,
                        "status": "pending",
                    }
                ]
            },
        )

        rpc.events = [
            {
                "type": "message_end",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "ui-req-1",
                    "toolName": "ask_user",
                    "details": {"answer": "Sidebar", "cancelled": False},
                },
            }
        ]

        broker._sync_output_from_rpc()

        self.assertEqual(_roundtrip_json(broker, {"cmd": "ui_state"}), {"requests": []})

    def test_turn_end_clears_stale_pending_requests_after_timeout_style_resolution(
        self,
    ) -> None:
        rpc = _FakeRpc()
        rpc.events = [
            {
                "type": "extension_ui_request",
                "id": "ui-req-timeout",
                "method": "select",
                "question": "Pick destinations",
                "options": ["Details", "Sidebar"],
                "turn_id": "turn-001",
            }
        ]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )

        broker._sync_output_from_rpc()
        self.assertEqual(len(broker.state.pending_ui_requests), 1)

        rpc.events = [
            {
                "type": "turn_end",
                "turn_id": "turn-001",
                "toolResults": [],
            }
        ]

        broker._sync_output_from_rpc()

        self.assertEqual(_roundtrip_json(broker, {"cmd": "ui_state"}), {"requests": []})
        self.assertEqual(broker.state.pending_ui_requests, {})

    def test_send_maps_to_prompt_command(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )
        server_sock, client_sock = socket.socketpair()
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall(
                (json.dumps({"cmd": "send", "text": "hello pi"}) + "\n").encode("utf-8")
            )
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp, {"queued": False, "queue_len": 0})
            self.assertIn(
                ("prompt", {"text": "hello pi", "streaming_behavior": None}), rpc.calls
            )
            self.assertEqual(broker.state.last_turn_id, "turn-001")
            self.assertTrue(broker.state.busy)
        finally:
            client_sock.close()

    def test_handle_terminal_line_submits_prompt_via_rpc(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp", rpc=rpc)
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )

        broker._submit_terminal_prompt("hello from tty")

        self.assertIn(
            ("prompt", {"text": "hello from tty", "streaming_behavior": None}),
            rpc.calls,
        )
        self.assertTrue(broker.state.busy)
        self.assertEqual(broker.state.last_turn_id, "turn-001")

    def test_handle_terminal_interrupt_maps_to_rpc_abort(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp", rpc=rpc)
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            last_turn_id="turn-001",
            busy=True,
        )

        broker._interrupt_terminal_turn()

        self.assertEqual(rpc.calls, [("abort", None)])
        self.assertFalse(broker.state.busy)
        self.assertIsNone(broker.state.last_turn_id)

    def test_tail_delta_only_returns_new_suffix_after_tail_rollover(self) -> None:
        previous = "alpha\nbeta\ngamma\n"
        current = "gamma\ndelta\n"

        self.assertEqual(_tail_delta(previous, current), "delta\n")

    def test_foreground_run_registers_sigint_handler_for_terminal_interrupts(
        self,
    ) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp", rpc=rpc)

        with (
            tempfile.TemporaryDirectory() as td,
            patch("codoxear.pi_broker.SOCK_DIR", Path(td)),
            patch("codoxear.pi_broker.PI_SESSION_DIR", Path(td)),
            patch.object(PiBroker, "_bg_sync_loop", side_effect=lambda: None),
            patch.object(
                PiBroker, "_sock_server", side_effect=lambda: broker._stop.set()
            ),
            patch("codoxear.pi_broker.sys.stdin.isatty", return_value=True),
            patch("codoxear.pi_broker.sys.stdout.isatty", return_value=True),
            patch("codoxear.pi_broker.signal.getsignal", return_value="old-handler"),
            patch("codoxear.pi_broker.signal.signal") as signal_mock,
        ):
            broker.run(foreground=True)

        self.assertEqual(signal_mock.call_args_list[0].args[0], signal.SIGINT)
        self.assertEqual(
            signal_mock.call_args_list[1].args, (signal.SIGINT, "old-handler")
        )

    def test_handle_sigint_delegates_to_previous_handler_when_idle(self) -> None:
        broker = PiBroker(cwd="/tmp")
        previous_calls: list[tuple[int, object | None]] = []

        broker._previous_sigint_handler = lambda signum, frame: previous_calls.append(
            (signum, frame)
        )
        broker._handle_sigint(signal.SIGINT, None)

        self.assertEqual(previous_calls, [(signal.SIGINT, None)])

    def test_handle_sigint_delegates_to_previous_handler_when_abort_fails(self) -> None:
        rpc = _FailingAbortRpc()
        broker = PiBroker(cwd="/tmp", rpc=rpc)
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            busy=True,
            last_turn_id="turn-001",
        )
        previous_calls: list[tuple[int, object | None]] = []
        broker._previous_sigint_handler = lambda signum, frame: previous_calls.append(
            (signum, frame)
        )

        broker._handle_sigint(signal.SIGINT, None)

        self.assertEqual(rpc.calls, [("abort", None)])
        self.assertEqual(previous_calls, [(signal.SIGINT, None)])

    def test_handle_sigint_uses_previous_handler_when_no_turn_is_active(self) -> None:
        broker = PiBroker(cwd="/tmp", rpc=_FakeRpc())
        previous_handler = unittest.mock.Mock()
        broker._previous_sigint_handler = previous_handler

        broker._handle_sigint(signal.SIGINT, None)

        previous_handler.assert_called_once_with(signal.SIGINT, None)

    def test_handle_sigint_uses_previous_handler_when_only_stale_turn_id_remains(
        self,
    ) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp", rpc=rpc)
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            busy=False,
            last_turn_id="turn-001",
        )
        previous_handler = unittest.mock.Mock()
        broker._previous_sigint_handler = previous_handler

        broker._handle_sigint(signal.SIGINT, None)

        previous_handler.assert_called_once_with(signal.SIGINT, None)
        self.assertNotIn(("abort", "turn-001"), rpc.calls)

    def test_send_does_not_block_state_reads_while_prompt_rpc_waits(self) -> None:
        rpc = _BlockingPromptRpc()
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            busy=False,
        )
        send_server, send_client = socket.socketpair()
        state_server, state_client = socket.socketpair()
        state_client.settimeout(0.2)
        try:
            send_thread = threading.Thread(
                target=broker._handle_conn, args=(send_server,), daemon=True
            )
            send_thread.start()
            send_client.sendall(
                (json.dumps({"cmd": "send", "text": "hello pi"}) + "\n").encode("utf-8")
            )
            self.assertTrue(rpc.prompt_started.wait(0.2))

            state_thread = threading.Thread(
                target=broker._handle_conn, args=(state_server,), daemon=True
            )
            state_thread.start()
            state_client.sendall((json.dumps({"cmd": "state"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(state_client).decode("utf-8"))

            self.assertEqual(resp, {"busy": True, "queue_len": 0, "token": None})

            rpc.release_prompt.set()
            send_resp = json.loads(_recv_line(send_client).decode("utf-8"))
            send_thread.join(1.0)
            state_thread.join(1.0)

            self.assertEqual(send_resp, {"queued": False, "queue_len": 0})
            self.assertTrue(broker.state.busy)
        finally:
            rpc.release_prompt.set()
            send_client.close()
            state_client.close()

    def test_send_surfaces_rpc_error_payload(self) -> None:
        class _ErrorRpc(_FakeRpc):
            def prompt(
                self, text: str, *, streaming_behavior: str | None = None
            ) -> dict[str, object]:
                self.calls.append(
                    (
                        "prompt",
                        {
                            "text": text,
                            "streaming_behavior": streaming_behavior,
                        },
                    )
                )
                return {"error": "prompt rejected"}

        rpc = _ErrorRpc()
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )
        server_sock, client_sock = socket.socketpair()
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall(
                (json.dumps({"cmd": "send", "text": "hello pi"}) + "\n").encode("utf-8")
            )
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp, {"error": "prompt rejected"})
            self.assertIn(
                ("prompt", {"text": "hello pi", "streaming_behavior": None}), rpc.calls
            )
            self.assertIsNone(broker.state.last_turn_id)
            self.assertFalse(broker.state.busy)
        finally:
            client_sock.close()

    def test_send_uses_steer_when_broker_is_already_busy(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            busy=True,
            last_turn_id="turn-active",
        )
        server_sock, client_sock = socket.socketpair()
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall(
                (
                    json.dumps({"cmd": "send", "text": "interrupt with steer"}) + "\n"
                ).encode("utf-8")
            )
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp, {"queued": False, "queue_len": 0})
            self.assertIn(
                (
                    "prompt",
                    {"text": "interrupt with steer", "streaming_behavior": "steer"},
                ),
                rpc.calls,
            )
            self.assertTrue(broker.state.busy)
        finally:
            client_sock.close()

    def test_send_surfaces_rpc_exception_message(self) -> None:
        class _ErrorRpc(_FakeRpc):
            def prompt(
                self, text: str, *, streaming_behavior: str | None = None
            ) -> dict[str, object]:
                self.calls.append(
                    (
                        "prompt",
                        {
                            "text": text,
                            "streaming_behavior": streaming_behavior,
                        },
                    )
                )
                raise RuntimeError("prompt rejected")

        rpc = _ErrorRpc()
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )
        server_sock, client_sock = socket.socketpair()
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall(
                (json.dumps({"cmd": "send", "text": "hello pi"}) + "\n").encode("utf-8")
            )
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp.get("error"), "prompt rejected")
            self.assertNotEqual(resp.get("error"), "exception")
            self.assertIn(
                ("prompt", {"text": "hello pi", "streaming_behavior": None}),
                rpc.calls,
            )
            self.assertIsNone(broker.state.last_turn_id)
            self.assertFalse(broker.state.busy)
        finally:
            client_sock.close()

    def test_keys_interrupt_maps_escape_to_abort(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            last_turn_id="turn-001",
            busy=True,
        )
        server_sock, client_sock = socket.socketpair()
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall(
                (json.dumps({"cmd": "keys", "seq": "\\x1b"}) + "\n").encode("utf-8")
            )
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp, {"ok": True, "queued": False, "n": 1})
            self.assertEqual(rpc.calls, [("abort", None)])
            self.assertFalse(broker.state.busy)
        finally:
            client_sock.close()

    def test_abort_does_not_block_tail_reads_while_abort_rpc_waits(self) -> None:
        rpc = _BlockingAbortRpc()
        rpc.events = [
            {"type": "message.delta", "turn_id": "turn-001", "delta": "working"}
        ]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            last_turn_id="turn-001",
            busy=True,
        )
        key_server, key_client = socket.socketpair()
        tail_server, tail_client = socket.socketpair()
        tail_client.settimeout(0.2)
        try:
            key_thread = threading.Thread(
                target=broker._handle_conn, args=(key_server,), daemon=True
            )
            key_thread.start()
            key_client.sendall(
                (json.dumps({"cmd": "keys", "seq": "\\x1b"}) + "\n").encode("utf-8")
            )
            self.assertTrue(rpc.abort_started.wait(0.2))

            tail_thread = threading.Thread(
                target=broker._handle_conn, args=(tail_server,), daemon=True
            )
            tail_thread.start()
            tail_client.sendall((json.dumps({"cmd": "tail"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(tail_client).decode("utf-8"))

            self.assertIn("working", resp["tail"])

            rpc.release_abort.set()
            key_resp = json.loads(_recv_line(key_client).decode("utf-8"))
            key_thread.join(1.0)
            tail_thread.join(1.0)

            self.assertEqual(key_resp, {"ok": True, "queued": False, "n": 1})
            self.assertFalse(broker.state.busy)
        finally:
            rpc.release_abort.set()
            key_client.close()
            tail_client.close()

    def test_state_refreshes_last_turn_id_from_stream_events(self) -> None:
        rpc = _FakeRpc()
        rpc.state["busy"] = True
        rpc.events = [
            {"type": "message.delta", "turn_id": "turn-002", "delta": "working"}
        ]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            last_turn_id="turn-001",
            busy=True,
        )
        # Background sync drains events and updates turn_id
        broker._sync_state_from_rpc()
        self.assertEqual(broker.state.last_turn_id, "turn-002")

        # Socket state command returns cached values without blocking
        server_sock, client_sock = socket.socketpair()
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "state"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp, {"busy": True, "queue_len": 0, "token": None})
        finally:
            client_sock.close()

    def test_tail_returns_streamed_pi_output_and_events(self) -> None:
        rpc = _FakeRpc()
        rpc.events = [
            {"type": "message.delta", "turn_id": "turn-002", "delta": "working"},
            {"type": "tool.started", "turn_id": "turn-002", "tool_name": "read"},
        ]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )

        server_sock, client_sock = socket.socketpair()
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "tail"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertIn("working", resp["tail"])
            self.assertIn("read", resp["tail"])
            self.assertEqual(resp["tail"], broker.state.output_tail)
            self.assertEqual(rpc.calls, [])
        finally:
            client_sock.close()

    def test_tail_includes_pi_stderr_diagnostics(self) -> None:
        rpc = _FakeRpc()
        rpc.stderr_lines = ["startup failed\n"]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )

        server_sock, client_sock = socket.socketpair()
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "tail"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertIn("startup failed", resp["tail"])
            self.assertEqual(rpc.calls, [])
        finally:
            client_sock.close()

    def test_tail_still_includes_pi_stderr_when_state_refresh_fails(self) -> None:
        rpc = _FakeRpc()
        rpc.state_error = RuntimeError("state probe failed")
        rpc.stderr_lines = ["startup failed\n"]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )
        # Simulate background sync — get_state fails but stderr still drains
        broker._sync_state_from_rpc()

        server_sock, client_sock = socket.socketpair()
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "tail"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertIn("startup failed", resp["tail"])
            self.assertEqual(resp["tail"], broker.state.output_tail)
        finally:
            client_sock.close()

    def test_tail_returns_cached_output_without_blocking(self) -> None:
        rpc = _FakeRpc()
        rpc.stderr_lines = ["startup failed\n"]
        rpc.events = [
            {"type": "message.delta", "turn_id": "turn-002", "delta": "working"}
        ]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )
        # Simulate background sync draining output
        broker._sync_state_from_rpc()

        server_sock, client_sock = socket.socketpair()
        client_sock.settimeout(0.2)
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "tail"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))

            self.assertIn("startup failed", resp["tail"])
            self.assertIn("working", resp["tail"])
            self.assertEqual(resp["tail"], broker.state.output_tail)
        finally:
            thread.join(1.0)
            client_sock.close()

    def test_keys_interrupt_aborts_without_relying_on_cached_turn_id(self) -> None:
        rpc = _FakeRpc()
        rpc.state["busy"] = True
        rpc.state["turn_id"] = "turn-002"
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            last_turn_id="turn-001",
            busy=True,
        )
        server_sock, client_sock = socket.socketpair()
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall(
                (json.dumps({"cmd": "keys", "seq": "\\x1b"}) + "\n").encode("utf-8")
            )
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp, {"ok": True, "queued": False, "n": 1})
            self.assertEqual(rpc.calls, [("abort", None)])
            self.assertFalse(broker.state.busy)
        finally:
            client_sock.close()

    def test_turn_end_marks_broker_idle(self) -> None:
        rpc = _FakeRpc()
        rpc.events = [{"type": "turn_end", "turn_id": "turn-001", "toolResults": []}]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            last_turn_id="turn-001",
            busy=True,
            prompt_sent_at=1.0,
        )

        broker._sync_output_from_rpc()

        self.assertFalse(broker.state.busy)
        self.assertEqual(broker.state.prompt_sent_at, 0.0)
        self.assertIsNone(broker.state.last_turn_id)

    def test_stale_turn_end_does_not_clear_newer_active_turn_state(self) -> None:
        rpc = _FakeRpc()
        rpc.events = [{"type": "turn_end", "turn_id": "turn-001", "toolResults": []}]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            last_turn_id="turn-002",
            busy=True,
            prompt_sent_at=1.0,
        )

        broker._sync_output_from_rpc()

        self.assertTrue(broker.state.busy)
        self.assertEqual(broker.state.prompt_sent_at, 1.0)
        self.assertEqual(broker.state.last_turn_id, "turn-002")

    def test_stale_turn_end_does_not_clear_newer_pending_ui_requests(self) -> None:
        rpc = _FakeRpc()
        rpc.events = [{"type": "turn_end", "turn_id": "turn-001", "toolResults": []}]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            last_turn_id="turn-002",
            busy=True,
            pending_ui_requests={
                "ui-req-1": {"id": "ui-req-1", "status": "pending", "method": "select"}
            },
        )

        broker._sync_output_from_rpc()

        self.assertIn("ui-req-1", broker.state.pending_ui_requests)

    def test_stale_turn_end_does_not_clear_inflight_prompt_without_turn_id_yet(
        self,
    ) -> None:
        rpc = _FakeRpc()
        rpc.events = [{"type": "turn_end", "turn_id": "turn-001", "toolResults": []}]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            last_turn_id=None,
            busy=True,
            prompt_sent_at=1.0,
            pending_ui_requests={
                "ui-req-1": {"id": "ui-req-1", "status": "pending", "method": "select"}
            },
        )

        broker._sync_output_from_rpc()

        self.assertTrue(broker.state.busy)
        self.assertEqual(broker.state.prompt_sent_at, 1.0)
        self.assertIn("ui-req-1", broker.state.pending_ui_requests)

    def test_idless_turn_end_clears_current_turn_only_when_no_turn_id_is_known(
        self,
    ) -> None:
        rpc = _FakeRpc()
        rpc.events = [{"type": "turn_end", "toolResults": []}]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            last_turn_id=None,
            busy=True,
            prompt_sent_at=0.0,
            pending_ui_requests={
                "ui-req-1": {"id": "ui-req-1", "status": "pending", "method": "select"}
            },
        )

        broker._sync_output_from_rpc()

        self.assertFalse(broker.state.busy)
        self.assertIsNone(broker.state.last_turn_id)
        self.assertEqual(broker.state.pending_ui_requests, {})

    def test_idless_turn_end_does_not_clear_known_active_turn_state(self) -> None:
        rpc = _FakeRpc()
        rpc.events = [{"type": "turn_end", "toolResults": []}]
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            last_turn_id="turn-002",
            busy=True,
            prompt_sent_at=0.0,
            pending_ui_requests={
                "ui-req-1": {"id": "ui-req-1", "status": "pending", "method": "select"}
            },
        )

        broker._sync_output_from_rpc()

        self.assertTrue(broker.state.busy)
        self.assertEqual(broker.state.last_turn_id, "turn-002")
        self.assertIn("ui-req-1", broker.state.pending_ui_requests)

    def test_submit_terminal_prompt_marks_broker_busy_before_prompt_returns(
        self,
    ) -> None:
        rpc = _BlockingPromptRpc()
        broker = PiBroker(cwd="/tmp", rpc=rpc)
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            busy=False,
        )

        thread = threading.Thread(
            target=broker._submit_terminal_prompt, args=("hello from tty",), daemon=True
        )
        thread.start()
        self.assertTrue(rpc.prompt_started.wait(0.2))
        self.assertTrue(broker.state.busy)
        rpc.release_prompt.set()
        thread.join(1.0)

    def test_run_closes_rpc_on_exit(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp", rpc=rpc)

        with (
            tempfile.TemporaryDirectory() as td,
            patch("codoxear.pi_broker.SOCK_DIR", Path(td)),
            patch("codoxear.pi_broker.PI_SESSION_DIR", Path(td)),
            patch.object(PiBroker, "_bg_sync_loop", side_effect=lambda: None),
            patch.object(
                PiBroker, "_sock_server", side_effect=lambda: broker._stop.set()
            ),
        ):
            broker.run(foreground=False)

        self.assertTrue(rpc.closed)

    def test_keys_interrupt_aborts_without_local_turn_id(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            busy=False,
        )
        server_sock, client_sock = socket.socketpair()
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall(
                (json.dumps({"cmd": "keys", "seq": "\\x1b"}) + "\n").encode("utf-8")
            )
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp, {"ok": True, "queued": False, "n": 1})
            self.assertEqual(rpc.calls, [("abort", None)])
            self.assertFalse(broker.state.busy)
        finally:
            client_sock.close()

    def test_keys_rejects_unsupported_sequences(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            last_turn_id="turn-001",
            busy=True,
        )
        server_sock, client_sock = socket.socketpair()
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall(
                (json.dumps({"cmd": "keys", "seq": "\\x03"}) + "\n").encode("utf-8")
            )
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp, {"error": "unsupported key sequence: \\x03"})
            self.assertEqual(rpc.calls, [])
        finally:
            client_sock.close()

    def test_shutdown_stops_subprocess(self) -> None:
        rpc = _FakeRpc()
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
        )
        server_sock, client_sock = socket.socketpair()
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall(
                (json.dumps({"cmd": "shutdown"}) + "\n").encode("utf-8")
            )
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp, {"ok": True})
            self.assertTrue(rpc.closed)
        finally:
            client_sock.close()


class _SlowBusyRpc(_FakeRpc):
    """Simulates Pi that accepts the prompt but doesn't report busy immediately."""

    def __init__(self) -> None:
        super().__init__()
        self._report_busy = False

    def prompt(
        self, text: str, *, streaming_behavior: str | None = None
    ) -> dict[str, object]:
        self.calls.append(
            (
                "prompt",
                {"text": text, "streaming_behavior": streaming_behavior},
            )
        )
        # Pi accepts the prompt but doesn't set busy yet in its internal state
        return {"turn_id": "turn-001", "queued": False}

    def get_state(self) -> dict[str, object]:
        self.calls.append(("get_state", None))
        return {"busy": self._report_busy, "session_id": "pi-session-001"}


class TestPiBrokerPromptGrace(unittest.TestCase):
    def test_sync_preserves_busy_during_prompt_grace_period(self) -> None:
        """After a prompt is accepted, bg-sync should not clear busy even if
        Pi's get_state returns busy=false (Pi hasn't started the turn yet)."""
        rpc = _SlowBusyRpc()
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            busy=False,
        )

        # Send a prompt via socket
        server_sock, client_sock = socket.socketpair()
        try:
            thread = threading.Thread(
                target=broker._handle_conn, args=(server_sock,), daemon=True
            )
            thread.start()
            client_sock.sendall(
                (json.dumps({"cmd": "send", "text": "hello"}) + "\n").encode("utf-8")
            )
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)
            self.assertEqual(resp, {"queued": False, "queue_len": 0})
            self.assertTrue(broker.state.busy)
            self.assertGreater(broker.state.prompt_sent_at, 0)
        finally:
            client_sock.close()

        # Pi reports not-busy (hasn't started the turn yet) — sync should preserve busy
        rpc._report_busy = False
        broker._sync_state_from_rpc()
        self.assertTrue(
            broker.state.busy, "busy should be preserved during prompt grace period"
        )

    def test_sync_clears_busy_when_pi_confirms_idle_after_grace(self) -> None:
        """Once Pi truly becomes idle (after grace period), sync should clear busy."""
        rpc = _SlowBusyRpc()
        broker = PiBroker(cwd="/tmp")
        broker.state = PiBrokerState(
            session_id="pi-session-001",
            codex_pid=123,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
            start_ts=0.0,
            rpc=rpc,
            busy=True,
        )
        # Simulate prompt sent long ago (grace period expired)
        import time

        broker.state.prompt_sent_at = time.monotonic() - 10.0

        rpc._report_busy = False
        broker._sync_state_from_rpc()
        self.assertFalse(
            broker.state.busy, "busy should be cleared after grace period expires"
        )


if __name__ == "__main__":
    unittest.main()
