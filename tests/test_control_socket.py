import json
import socket
import threading
import unittest

from codoxear.control_socket import handle_control_socket_connection
from codoxear.util import _send_socket_json_line
from codoxear.util import _socket_peer_disconnected


def roundtrip(req, handlers):
    left, right = socket.socketpair()
    try:
        thread = threading.Thread(
            target=handle_control_socket_connection,
            kwargs={
                "conn": right,
                "handlers": handlers,
                "send_json_line": _send_socket_json_line,
                "socket_peer_disconnected": _socket_peer_disconnected,
            },
            daemon=True,
        )
        thread.start()
        left.sendall(json.dumps(req).encode("utf-8") + b"\n")
        data = b""
        while not data.endswith(b"\n"):
            chunk = left.recv(4096)
            if not chunk:
                break
            data += chunk
        thread.join(timeout=2)
        if thread.is_alive():
            raise AssertionError("control socket handler did not finish")
        return json.loads(data.decode("utf-8"))
    finally:
        left.close()


class TestControlSocket(unittest.TestCase):
    def test_dispatches_known_command_and_after_reply_action(self) -> None:
        actions = []

        def state_handler(req):
            self.assertEqual(req["cmd"], "state")
            return {"busy": False, "queue_len": 0}, lambda: actions.append("after")

        resp = roundtrip({"cmd": "state"}, {"state": state_handler})
        self.assertEqual(resp, {"busy": False, "queue_len": 0})
        self.assertEqual(actions, ["after"])

    def test_unknown_and_invalid_request_errors_are_protocol_owned(self) -> None:
        self.assertEqual(roundtrip({"cmd": "missing"}, {}), {"error": "unknown cmd"})
        self.assertEqual(roundtrip([], {}), {"error": "invalid request"})

    def test_handler_exception_returns_trace_payload(self) -> None:
        def boom(_req):
            raise RuntimeError("boom")

        resp = roundtrip({"cmd": "state"}, {"state": boom})
        self.assertEqual(resp["error"], "exception")
        self.assertIn("RuntimeError: boom", resp["trace"])


if __name__ == "__main__":
    unittest.main()
