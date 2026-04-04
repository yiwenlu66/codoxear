import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from codoxear.pi_broker import PiBroker
from codoxear.pi_broker import State as PiBrokerState


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
        thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
        thread.start()
        client_sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        resp = json.loads(_recv_line(client_sock).decode("utf-8"))
        thread.join(1.0)
        return resp
    finally:
        client_sock.close()


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

    def prompt(self, text: str) -> dict[str, object]:
        self.calls.append(("prompt", text))
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

    def prompt(self, text: str) -> dict[str, object]:
        self.calls.append(("prompt", text))
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


class TestPiBroker(unittest.TestCase):
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

            meta = json.loads(sock_path.with_suffix(".json").read_text(encoding="utf-8"))

        self.assertIsNone(meta["log_path"])
        self.assertEqual(meta["session_path"], str(session_path))

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
            thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "state"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp, {"busy": True, "queue_len": 0, "token": {"completion_tokens": 12}})
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

        first = _roundtrip_json(broker, {"cmd": "ui_response", "id": "ui-req-1", "value": "Details"})
        second = _roundtrip_json(broker, {"cmd": "ui_response", "id": "ui-req-1", "value": "Sidebar"})

        self.assertEqual(first, {"ok": True})
        self.assertEqual(rpc.ui_responses, [{"id": "ui-req-1", "value": "Details"}])
        self.assertEqual(second, {"error": "request already resolved"})
        self.assertEqual(broker.state.pending_ui_requests["ui-req-1"]["status"], "resolved")

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

        first = _roundtrip_json(broker, {"cmd": "ui_response", "id": "ui-req-1", "value": "Details"})

        self.assertEqual(first, {"error": "send failed"})
        self.assertEqual(broker.state.pending_ui_requests["ui-req-1"]["status"], "pending")

        retry = _roundtrip_json(broker, {"cmd": "ui_response", "id": "ui-req-1", "value": "Details"})

        self.assertEqual(retry, {"ok": True})
        self.assertEqual(rpc.ui_responses, [{"id": "ui-req-1", "value": "Details"}])

    def test_ui_response_forwards_confirmed_and_ui_state_hides_resolved_requests(self) -> None:
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

        resp = _roundtrip_json(broker, {"cmd": "ui_response", "id": "ui-req-1", "confirmed": True})
        ui_state = _roundtrip_json(broker, {"cmd": "ui_state"})

        self.assertEqual(resp, {"ok": True})
        self.assertEqual(rpc.ui_responses, [{"id": "ui-req-1", "confirmed": True}])
        self.assertEqual(ui_state, {"requests": []})

    def test_message_end_retires_pending_request_when_pi_resolves_elsewhere(self) -> None:
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
        self.assertEqual(_roundtrip_json(broker, {"cmd": "ui_state"}), {"requests": [
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
        ]})

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

    def test_turn_end_clears_stale_pending_requests_after_timeout_style_resolution(self) -> None:
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
            thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "send", "text": "hello pi"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp, {"queued": False, "queue_len": 0})
            self.assertIn(("prompt", "hello pi"), rpc.calls)
            self.assertEqual(broker.state.last_turn_id, "turn-001")
            self.assertTrue(broker.state.busy)
        finally:
            client_sock.close()

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
            send_thread = threading.Thread(target=broker._handle_conn, args=(send_server,), daemon=True)
            send_thread.start()
            send_client.sendall((json.dumps({"cmd": "send", "text": "hello pi"}) + "\n").encode("utf-8"))
            self.assertTrue(rpc.prompt_started.wait(0.2))

            state_thread = threading.Thread(target=broker._handle_conn, args=(state_server,), daemon=True)
            state_thread.start()
            state_client.sendall((json.dumps({"cmd": "state"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(state_client).decode("utf-8"))

            self.assertEqual(resp, {"busy": False, "queue_len": 0, "token": None})

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
            def prompt(self, text: str) -> dict[str, object]:
                self.calls.append(("prompt", text))
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
            thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "send", "text": "hello pi"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp, {"error": "prompt rejected"})
            self.assertIn(("prompt", "hello pi"), rpc.calls)
            self.assertIsNone(broker.state.last_turn_id)
            self.assertFalse(broker.state.busy)
        finally:
            client_sock.close()

    def test_send_surfaces_rpc_exception_message(self) -> None:
        class _ErrorRpc(_FakeRpc):
            def prompt(self, text: str) -> dict[str, object]:
                self.calls.append(("prompt", text))
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
            thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "send", "text": "hello pi"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp.get("error"), "prompt rejected")
            self.assertNotEqual(resp.get("error"), "exception")
            self.assertIn(("prompt", "hello pi"), rpc.calls)
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
            thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "keys", "seq": "\\x1b"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp, {"ok": True, "queued": False, "n": 1})
            self.assertIn(("abort", "turn-001"), rpc.calls)
            self.assertFalse(broker.state.busy)
        finally:
            client_sock.close()

    def test_abort_does_not_block_tail_reads_while_abort_rpc_waits(self) -> None:
        rpc = _BlockingAbortRpc()
        rpc.events = [{"type": "message.delta", "turn_id": "turn-001", "delta": "working"}]
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
            key_thread = threading.Thread(target=broker._handle_conn, args=(key_server,), daemon=True)
            key_thread.start()
            key_client.sendall((json.dumps({"cmd": "keys", "seq": "\\x1b"}) + "\n").encode("utf-8"))
            self.assertTrue(rpc.abort_started.wait(0.2))

            tail_thread = threading.Thread(target=broker._handle_conn, args=(tail_server,), daemon=True)
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
        rpc.events = [{"type": "message.delta", "turn_id": "turn-002", "delta": "working"}]
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
            thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
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
            thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
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
            thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
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
            thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
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
        rpc.events = [{"type": "message.delta", "turn_id": "turn-002", "delta": "working"}]
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
            thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "tail"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))

            self.assertIn("startup failed", resp["tail"])
            self.assertIn("working", resp["tail"])
            self.assertEqual(resp["tail"], broker.state.output_tail)
        finally:
            thread.join(1.0)
            client_sock.close()

    def test_keys_interrupt_refreshes_turn_id_before_abort(self) -> None:
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
            thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "keys", "seq": "\\x1b"}) + "\n").encode("utf-8"))
            resp = json.loads(_recv_line(client_sock).decode("utf-8"))
            thread.join(1.0)

            self.assertEqual(resp, {"ok": True, "queued": False, "n": 1})
            self.assertEqual(rpc.calls, [("abort", "turn-001")])
            self.assertFalse(broker.state.busy)
        finally:
            client_sock.close()

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
            thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "keys", "seq": "\\x1b"}) + "\n").encode("utf-8"))
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
            thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "keys", "seq": "\\x03"}) + "\n").encode("utf-8"))
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
            thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "shutdown"}) + "\n").encode("utf-8"))
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

    def prompt(self, text: str) -> dict[str, object]:
        self.calls.append(("prompt", text))
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
            thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
            thread.start()
            client_sock.sendall((json.dumps({"cmd": "send", "text": "hello"}) + "\n").encode("utf-8"))
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
        self.assertTrue(broker.state.busy, "busy should be preserved during prompt grace period")

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
        self.assertFalse(broker.state.busy, "busy should be cleared after grace period expires")


if __name__ == "__main__":
    unittest.main()
