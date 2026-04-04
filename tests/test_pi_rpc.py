import io
import json
import os
import queue
import subprocess
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from codoxear.pi_rpc import PiRpcClient
from tests.pi_fixtures import pi_rpc_request_payloads
from tests.pi_fixtures import pi_rpc_response_lines
from tests.pi_fixtures import pi_stream_events
from tests.pi_fixtures import pi_ui_request_event
from tests.pi_fixtures import pi_ui_response_payload


class _QueueReader:
    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._closed = False

    def put_line(self, line: str) -> None:
        self._queue.put(line)

    def readline(self) -> str:
        item = self._queue.get(timeout=1.0)
        if item == "__EOF__":
            self._closed = True
            return ""
        return item

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put("__EOF__")


class _FakeProc:
    def __init__(self) -> None:
        self.stdin: Any = io.StringIO()
        self.stdout = _QueueReader()
        self.stderr = _QueueReader()
        self.terminated = False

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0


class _BrokenWriter:
    def write(self, _data: str) -> int:
        raise BrokenPipeError("stdin unavailable")

    def flush(self) -> None:
        raise BrokenPipeError("stdin unavailable")

    def close(self) -> None:
        return


class _BlockingWriter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.first_write_started = threading.Event()
        self.allow_first_write = threading.Event()
        self.concurrent_write_detected = threading.Event()
        self.payloads: list[str] = []
        self._write_calls = 0
        self._write_in_progress = False

    def write(self, data: str) -> int:
        with self._lock:
            self._write_calls += 1
            first_write = self._write_calls == 1
            if self._write_in_progress:
                self.concurrent_write_detected.set()
            self._write_in_progress = True
        if first_write:
            self.first_write_started.set()
            if not self.allow_first_write.wait(timeout=1.0):
                raise TimeoutError("timed out waiting to release first stdin write")

        with self._lock:
            self.payloads.append(data)
            self._write_in_progress = False
        return len(data)

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


class _ObservableLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.contended_acquire_attempted = threading.Event()

    def __enter__(self) -> "_ObservableLock":
        if not self._lock.acquire(blocking=False):
            self.contended_acquire_attempted.set()
            self._lock.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._lock.release()


class TestPiRpc(unittest.TestCase):
    def test_stderr_reader_preserves_child_diagnostics(self) -> None:
        proc = _FakeProc()
        client = PiRpcClient(proc=proc)
        try:
            proc.stderr.put_line("startup failed\n")
            deadline = time.time() + 1.0
            while "startup failed" not in client.stderr_tail():
                if time.time() >= deadline:
                    self.fail("pi rpc stderr diagnostics were not captured")
                time.sleep(0.01)
        finally:
            client.close()

    def test_send_command_cleans_pending_on_stdin_failure(self) -> None:
        proc = _FakeProc()
        proc.stdin = _BrokenWriter()
        client = PiRpcClient(proc=proc)
        try:
            with self.assertRaises(BrokenPipeError):
                client.send_command("prompt", payload={"message": "hello"}, request_id="cmd-broken")

            self.assertNotIn("cmd-broken", client._pending)
        finally:
            client.close()

    def test_send_command_allows_success_without_result_payload(self) -> None:
        proc = _FakeProc()
        client = PiRpcClient(proc=proc)
        try:
            result_box: dict[str, object] = {}

            def _call() -> None:
                result_box["result"] = client.send_command("prompt", request_id="cmd-prompt-empty")

            thread = threading.Thread(target=_call)
            thread.start()
            proc.stdout.put_line(json.dumps({"type": "response", "id": "cmd-prompt-empty", "command": "prompt", "success": True}) + "\n")
            thread.join(1.0)

            self.assertFalse(thread.is_alive())
            self.assertEqual(result_box["result"], {})
        finally:
            client.close()

    def test_send_command_raises_timeout_error_when_response_never_arrives(self) -> None:
        proc = _FakeProc()
        client = PiRpcClient(proc=proc)
        try:
            with self.assertRaises(TimeoutError) as cm:
                client.send_command("prompt", request_id="cmd-timeout", timeout_s=0.01)

            self.assertEqual(str(cm.exception), "pi rpc prompt timed out")
            self.assertNotIn("cmd-timeout", client._pending)
        finally:
            client.close()

    def test_send_command_writes_jsonl_and_correlates_response(self) -> None:
        proc = _FakeProc()
        client = PiRpcClient(proc=proc)
        try:
            prompt = pi_rpc_request_payloads()["prompt"]
            response_line = pi_rpc_response_lines()["prompt"]

            result_box: dict[str, object] = {}

            def _call() -> None:
                result_box["result"] = client.send_command(
                    prompt["type"],
                    payload={"message": prompt["message"]},
                    request_id=prompt["id"],
                )

            thread = threading.Thread(target=_call)
            thread.start()
            proc.stdout.put_line(response_line)
            thread.join(1.0)

            self.assertFalse(thread.is_alive())
            self.assertEqual(json.loads(proc.stdin.getvalue()), prompt)
            self.assertEqual(result_box["result"], {"queued": False})
        finally:
            client.close()

    def test_event_reader_collects_async_events_without_blocking_responses(self) -> None:
        proc = _FakeProc()
        client = PiRpcClient(proc=proc)
        try:
            state = pi_rpc_request_payloads()["get_state"]
            response_line = pi_rpc_response_lines()["get_state"]
            events = pi_stream_events()

            result_box: dict[str, object] = {}

            def _call() -> None:
                result_box["result"] = client.send_command(
                    state["type"],
                    request_id=state["id"],
                )

            thread = threading.Thread(target=_call)
            thread.start()
            for event in events:
                proc.stdout.put_line(json.dumps(event) + "\n")
            proc.stdout.put_line(response_line)
            thread.join(1.0)

            self.assertFalse(thread.is_alive())
            self.assertEqual(
                result_box["result"],
                {
                    "isStreaming": False,
                    "pendingMessageCount": 0,
                    "sessionId": "pi-session-001",
                    "sessionFile": "/tmp/pi-session-001.jsonl",
                    "messageCount": 2,
                },
            )
            self.assertEqual(client.drain_events(), events)
            self.assertEqual(client.drain_events(), [])
        finally:
            client.close()

    def test_event_reader_retains_extension_ui_request_events(self) -> None:
        proc = _FakeProc()
        client = PiRpcClient(proc=proc)
        try:
            event = pi_ui_request_event()

            proc.stdout.put_line(json.dumps(event) + "\n")
            deadline = time.time() + 1.0
            drained: list[dict[str, Any]] = []
            while not drained:
                drained = client.drain_events()
                if drained:
                    break
                if time.time() >= deadline:
                    self.fail("pi rpc extension_ui_request event was not captured")
                time.sleep(0.01)

            self.assertEqual(drained, [event])
        finally:
            client.close()

    def test_send_ui_response_writes_raw_extension_ui_response_payload(self) -> None:
        proc = _FakeProc()
        client = PiRpcClient(proc=proc)
        try:
            client.send_ui_response("ui-req-1", value="Details")

            self.assertEqual(json.loads(proc.stdin.getvalue()), pi_ui_response_payload())
        finally:
            client.close()

    def test_send_ui_response_writes_confirmed_payload(self) -> None:
        proc = _FakeProc()
        client = PiRpcClient(proc=proc)
        try:
            client.send_ui_response("ui-req-1", confirmed=True)

            self.assertEqual(
                json.loads(proc.stdin.getvalue()),
                {
                    "type": "extension_ui_response",
                    "id": "ui-req-1",
                    "confirmed": True,
                },
            )
        finally:
            client.close()

    def test_send_ui_response_writes_explicit_false_confirmed_payload(self) -> None:
        proc = _FakeProc()
        client = PiRpcClient(proc=proc)
        try:
            client.send_ui_response("ui-req-1", confirmed=False)

            self.assertEqual(
                json.loads(proc.stdin.getvalue()),
                {
                    "type": "extension_ui_response",
                    "id": "ui-req-1",
                    "confirmed": False,
                },
            )
        finally:
            client.close()

    def test_send_ui_response_writes_cancelled_payload(self) -> None:
        proc = _FakeProc()
        client = PiRpcClient(proc=proc)
        try:
            client.send_ui_response("ui-req-1", cancelled=True)

            self.assertEqual(
                json.loads(proc.stdin.getvalue()),
                {
                    "type": "extension_ui_response",
                    "id": "ui-req-1",
                    "cancelled": True,
                },
            )
        finally:
            client.close()

    def test_send_ui_response_shares_stdin_lock_with_send_command(self) -> None:
        proc = _FakeProc()
        proc.stdin = _BlockingWriter()
        client = PiRpcClient(proc=proc)
        client._stdin_lock = _ObservableLock()
        try:
            result_box: dict[str, object] = {}

            def _send_command() -> None:
                result_box["result"] = client.send_command("prompt", request_id="cmd-lock")

            command_thread = threading.Thread(target=_send_command)
            command_thread.start()
            self.assertTrue(proc.stdin.first_write_started.wait(timeout=1.0))

            ui_thread = threading.Thread(
                target=lambda: client.send_ui_response("ui-req-1", confirmed=True)
            )
            ui_thread.start()
            self.assertTrue(client._stdin_lock.contended_acquire_attempted.wait(timeout=1.0))

            proc.stdin.allow_first_write.set()
            proc.stdout.put_line(
                json.dumps(
                    {
                        "type": "response",
                        "id": "cmd-lock",
                        "command": "prompt",
                        "success": True,
                        "data": {},
                    }
                )
                + "\n"
            )

            command_thread.join(1.0)
            ui_thread.join(1.0)

            self.assertFalse(command_thread.is_alive())
            self.assertFalse(ui_thread.is_alive())
            self.assertFalse(proc.stdin.concurrent_write_detected.is_set())
            self.assertEqual(
                [json.loads(payload) for payload in proc.stdin.payloads],
                [
                    {"id": "cmd-lock", "type": "prompt"},
                    {
                        "type": "extension_ui_response",
                        "id": "ui-req-1",
                        "confirmed": True,
                    },
                ],
            )
            self.assertEqual(result_box["result"], {})
        finally:
            client.close()

    def test_spawn_uses_pi_session_flag_not_unsupported_session_file(self) -> None:
        client = PiRpcClient.__new__(PiRpcClient)

        with patch("codoxear.pi_rpc.subprocess.Popen") as popen:
            popen.return_value = _FakeProc()

            PiRpcClient._spawn(client, cwd="/tmp", session_path=Path("/tmp/test-pi-session.jsonl"))

        argv = popen.call_args.args[0]
        self.assertIn("--session", argv)
        self.assertNotIn("--session-file", argv)

    def test_spawn_starts_new_session_for_server_kill_contract(self) -> None:
        client = PiRpcClient.__new__(PiRpcClient)

        with patch("codoxear.pi_rpc.subprocess.Popen") as popen:
            popen.return_value = _FakeProc()

            PiRpcClient._spawn(client, cwd="/tmp", session_path=None)

        self.assertTrue(popen.called)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_spawn_pipes_stderr_for_noninteractive_diagnostics(self) -> None:
        client = PiRpcClient.__new__(PiRpcClient)

        with patch("codoxear.pi_rpc.subprocess.Popen") as popen:
            popen.return_value = _FakeProc()

            PiRpcClient._spawn(client, cwd="/tmp", session_path=None)

        self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.PIPE)

    def test_close_terminates_managed_process_group(self) -> None:
        proc = subprocess.Popen(
            ["sh", "-c", "sleep 100 & child=$!; printf '%s\\n' \"$child\"; wait"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        try:
            assert proc.stdout is not None
            child_pid = int(proc.stdout.readline().strip())
            client = PiRpcClient.__new__(PiRpcClient)
            client._closed = False
            client._proc = proc

            PiRpcClient.close(client)

            proc.wait(timeout=2.0)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, 9)
                proc.wait(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
