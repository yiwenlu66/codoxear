import json
import socket
import threading
import time
import unittest
import errno
from pathlib import Path
from unittest.mock import patch

from codoxear.broker import Broker, State as BrokerState, _inject as broker_inject
from codoxear.sessiond import Sessiond, State as SessiondState, _inject as sessiond_inject


def _recv_line(sock: socket.socket) -> bytes:
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
    return buf.split(b"\n", 1)[0]


class _FakeReadFile:
    def __init__(self, line: bytes) -> None:
        self._line = line
        self.closed = False

    def readline(self) -> bytes:
        return self._line

    def close(self) -> None:
        self.closed = True


class _BrokenPipeConn:
    def __init__(self, line: bytes) -> None:
        self._line = line
        self.file = _FakeReadFile(line)
        self.sendall_calls = 0
        self.closed = False

    def makefile(self, _mode: str) -> _FakeReadFile:
        return self.file

    def sendall(self, _data: bytes) -> None:
        self.sendall_calls += 1
        raise BrokenPipeError(errno.EPIPE, "Broken pipe")

    def close(self) -> None:
        self.closed = True


class TestSendAck(unittest.TestCase):
    def test_broker_inject_uses_bracketed_paste_and_handles_partial_writes(self) -> None:
        writes: list[bytes] = []

        def fake_write(_fd: int, data: bytes | memoryview) -> int:
            chunk = bytes(data)
            n = min(7, len(chunk))
            writes.append(chunk[:n])
            return n

        with patch("codoxear.broker.os.write", side_effect=fake_write), patch("codoxear.broker.time.sleep"):
            broker_inject(1, text="hello world", suffix=b"\r", delay_s=0.0)

        payload = b"".join(writes)
        self.assertEqual(payload, b"\x1b[200~hello world\x1b[201~\r")

    def test_sessiond_inject_uses_bracketed_paste_and_handles_partial_writes(self) -> None:
        writes: list[bytes] = []

        def fake_write(_fd: int, data: bytes | memoryview) -> int:
            chunk = bytes(data)
            n = min(5, len(chunk))
            writes.append(chunk[:n])
            return n

        with patch("codoxear.sessiond.os.write", side_effect=fake_write), patch("codoxear.sessiond.time.sleep"):
            sessiond_inject(1, text="hello world", suffix=b"\r", delay_s=0.0)

        payload = b"".join(writes)
        self.assertEqual(payload, b"\x1b[200~hello world\x1b[201~\r")

    def test_broker_send_ack_does_not_wait_for_full_inject(self) -> None:
        broker = Broker(cwd="/tmp", codex_args=[])
        broker.state = BrokerState(
            codex_pid=1,
            pty_master_fd=1,
            cwd="/tmp",
            start_ts=0.0,
            codex_home=Path("/tmp"),
            sessions_dir=Path("/tmp"),
        )
        server_sock, client_sock = socket.socketpair()
        try:
            with patch("codoxear.broker._inject", side_effect=lambda *_a, **_k: time.sleep(0.5)):
                thread = threading.Thread(target=broker._handle_conn, args=(server_sock,), daemon=True)
                thread.start()
                client_sock.settimeout(0.2)
                client_sock.sendall((json.dumps({"cmd": "send", "text": "x" * 20000}) + "\n").encode("utf-8"))
                t0 = time.monotonic()
                line = _recv_line(client_sock)
                dt = time.monotonic() - t0
                self.assertLess(dt, 0.2)
                self.assertEqual(json.loads(line.decode("utf-8")), {"queued": False, "queue_len": 0})
                self.assertTrue(thread.is_alive())
                thread.join(1.0)
                self.assertFalse(thread.is_alive())
        finally:
            client_sock.close()

    def test_sessiond_send_ack_does_not_wait_for_full_inject(self) -> None:
        sessiond = Sessiond("/tmp", [])
        sessiond.state = SessiondState(
            session_id="sid",
            codex_pid=1,
            log_path=Path("/tmp/log.jsonl"),
            sock_path=Path("/tmp/test.sock"),
            pty_master_fd=1,
            start_ts=0.0,
        )
        server_sock, client_sock = socket.socketpair()
        try:
            with patch("codoxear.sessiond._inject", side_effect=lambda *_a, **_k: time.sleep(0.5)):
                thread = threading.Thread(target=sessiond._handle_conn, args=(server_sock,), daemon=True)
                thread.start()
                client_sock.settimeout(0.2)
                client_sock.sendall((json.dumps({"cmd": "send", "text": "x" * 20000}) + "\n").encode("utf-8"))
                t0 = time.monotonic()
                line = _recv_line(client_sock)
                dt = time.monotonic() - t0
                self.assertLess(dt, 0.2)
                self.assertEqual(json.loads(line.decode("utf-8")), {"queued": False, "queue_len": 0})
                self.assertTrue(thread.is_alive())
                thread.join(1.0)
                self.assertFalse(thread.is_alive())
        finally:
            client_sock.close()

    def test_sessiond_ui_response_ack_does_not_wait_for_full_inject(self) -> None:
        sessiond = Sessiond("/tmp", [])
        sessiond.state = SessiondState(
            session_id="sid",
            codex_pid=1,
            log_path=Path("/tmp/log.jsonl"),
            sock_path=Path("/tmp/test.sock"),
            pty_master_fd=1,
            start_ts=0.0,
        )
        server_sock, client_sock = socket.socketpair()
        try:
            with patch("codoxear.sessiond._inject", side_effect=lambda *_a, **_k: time.sleep(0.5)) as inject:
                thread = threading.Thread(target=sessiond._handle_conn, args=(server_sock,), daemon=True)
                thread.start()
                client_sock.settimeout(0.2)
                client_sock.sendall((json.dumps({"cmd": "ui_response", "id": "ask-1", "value": "Details"}) + "\n").encode("utf-8"))
                t0 = time.monotonic()
                line = _recv_line(client_sock)
                dt = time.monotonic() - t0
                self.assertLess(dt, 0.2)
                self.assertEqual(json.loads(line.decode("utf-8")), {"ok": True})
                self.assertTrue(thread.is_alive())
                thread.join(1.0)
                self.assertFalse(thread.is_alive())
                inject.assert_called_once_with(1, text="Details", suffix=b"\r")
        finally:
            client_sock.close()

    def test_broker_ignores_broken_pipe_while_replying(self) -> None:
        broker = Broker(cwd="/tmp", codex_args=[])
        broker.state = BrokerState(
            codex_pid=1,
            pty_master_fd=1,
            cwd="/tmp",
            start_ts=0.0,
            codex_home=Path("/tmp"),
            sessions_dir=Path("/tmp"),
        )
        conn = _BrokenPipeConn((json.dumps({"cmd": "state"}) + "\n").encode("utf-8"))

        with patch("codoxear.broker.traceback.print_exc") as print_exc:
            broker._handle_conn(conn)

        self.assertEqual(conn.sendall_calls, 1)
        self.assertTrue(conn.file.closed)
        self.assertTrue(conn.closed)
        print_exc.assert_not_called()

    def test_sessiond_ignores_broken_pipe_while_replying(self) -> None:
        sessiond = Sessiond("/tmp", [])
        sessiond.state = SessiondState(
            session_id="sid",
            codex_pid=1,
            log_path=Path("/tmp/log.jsonl"),
            sock_path=Path("/tmp/test.sock"),
            pty_master_fd=1,
            start_ts=0.0,
        )
        conn = _BrokenPipeConn((json.dumps({"cmd": "state"}) + "\n").encode("utf-8"))

        with patch("codoxear.sessiond.traceback.print_exc") as print_exc:
            sessiond._handle_conn(conn)

        self.assertEqual(conn.sendall_calls, 1)
        self.assertTrue(conn.file.closed)
        self.assertTrue(conn.closed)
        print_exc.assert_not_called()
