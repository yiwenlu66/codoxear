import errno
import json
import socket
import unittest
from pathlib import Path

from codoxear import socket_json
from codoxear import util


ROOT = Path(__file__).resolve().parents[1]


class TestSocketJsonSource(unittest.TestCase):
    def test_socket_helpers_are_defined_once_with_util_facade(self) -> None:
        sources = {path: path.read_text(encoding="utf-8") for path in (ROOT / "codoxear").glob("*.py")}
        self.assertEqual(sum(src.count("def socket_peer_disconnected(") for src in sources.values()), 1)
        self.assertEqual(sum(src.count("def send_socket_json_line(") for src in sources.values()), 1)
        self.assertIn("def socket_peer_disconnected(exc: BaseException) -> bool:", sources[ROOT / "codoxear" / "socket_json.py"])
        self.assertIn("def send_socket_json_line(conn: socket.socket, payload: dict[str, Any]) -> None:", sources[ROOT / "codoxear" / "socket_json.py"])
        self.assertIn("from .socket_json import send_socket_json_line as _send_socket_json_line", sources[ROOT / "codoxear" / "util.py"])
        self.assertIn("from .socket_json import socket_peer_disconnected as _socket_peer_disconnected", sources[ROOT / "codoxear" / "util.py"])
        self.assertIn("from codoxear.util import _send_socket_json_line", sources[ROOT / "codoxear" / "broker_control.py"])
        self.assertIn("from .util import _send_socket_json_line as _send_socket_json_line", sources[ROOT / "codoxear" / "sessiond.py"])

    def test_socket_facade_exports_runtime_functions(self) -> None:
        self.assertIs(util._socket_peer_disconnected, socket_json.socket_peer_disconnected)
        self.assertIs(util._send_socket_json_line, socket_json.send_socket_json_line)

    def test_socket_peer_disconnected_semantics(self) -> None:
        self.assertTrue(socket_json.socket_peer_disconnected(BrokenPipeError()))
        self.assertTrue(socket_json.socket_peer_disconnected(ConnectionResetError()))
        self.assertTrue(socket_json.socket_peer_disconnected(ConnectionAbortedError()))
        for err_no in (errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED, errno.ENOTCONN, errno.ESHUTDOWN):
            self.assertTrue(socket_json.socket_peer_disconnected(OSError(err_no, "disconnect")))
        self.assertFalse(socket_json.socket_peer_disconnected(OSError(errno.EINVAL, "other")))
        self.assertFalse(socket_json.socket_peer_disconnected(RuntimeError("other")))

    def test_send_socket_json_line_writes_single_json_line(self) -> None:
        left, right = socket.socketpair()
        try:
            socket_json.send_socket_json_line(left, {"ok": True, "n": 2})
            self.assertEqual(json.loads(right.recv(4096).decode("utf-8")), {"ok": True, "n": 2})
        finally:
            left.close()
            right.close()


if __name__ == "__main__":
    unittest.main()
