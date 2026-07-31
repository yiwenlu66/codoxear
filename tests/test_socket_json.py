import errno
import json
import socket
import unittest

from codoxear import socket_json
from codoxear import util


class TestSocketJson(unittest.TestCase):
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
