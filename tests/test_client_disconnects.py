import errno
import unittest

import codoxear.server as server
from codoxear.server import _handle_route_exception
from codoxear.server import _is_client_disconnect


class TestClientDisconnects(unittest.TestCase):
    def test_disconnect_classifier_matches_transport_disconnects_only(self) -> None:
        self.assertTrue(_is_client_disconnect(BrokenPipeError()))
        self.assertTrue(_is_client_disconnect(ConnectionResetError()))
        self.assertTrue(_is_client_disconnect(ConnectionAbortedError()))
        self.assertTrue(_is_client_disconnect(OSError(errno.EPIPE, "pipe closed")))
        self.assertTrue(_is_client_disconnect(OSError(errno.ECONNRESET, "reset")))
        self.assertTrue(_is_client_disconnect(OSError(errno.ECONNABORTED, "aborted")))
        self.assertFalse(_is_client_disconnect(OSError(errno.ENOENT, "missing")))
        self.assertFalse(_is_client_disconnect(RuntimeError("boom")))

    def test_route_exception_handler_quiets_disconnects(self) -> None:
        calls: list[tuple[str, object]] = []
        original_print_exc = server.traceback.print_exc
        original_json_response = server._json_response
        try:
            server.traceback.print_exc = lambda: calls.append(("print_exc", None))  # type: ignore[assignment]
            server._json_response = lambda handler, status, obj: calls.append(("json", status, obj))  # type: ignore[assignment]

            _handle_route_exception(object(), BrokenPipeError("client went away"))

            self.assertEqual(calls, [])
        finally:
            server.traceback.print_exc = original_print_exc  # type: ignore[assignment]
            server._json_response = original_json_response  # type: ignore[assignment]

    def test_route_exception_handler_preserves_real_500s(self) -> None:
        calls: list[tuple[str, object]] = []
        original_print_exc = server.traceback.print_exc
        original_json_response = server._json_response
        try:
            server.traceback.print_exc = lambda: calls.append(("print_exc", None))  # type: ignore[assignment]
            server._json_response = lambda handler, status, obj: calls.append(("json", status, obj))  # type: ignore[assignment]

            try:
                raise RuntimeError("real bug")
            except RuntimeError as exc:
                _handle_route_exception(object(), exc)

            self.assertEqual(calls[0], ("print_exc", None))
            self.assertEqual(calls[1][0], "json")
            self.assertEqual(calls[1][1], 500)
            self.assertIn("real bug", calls[1][2]["error"])
        finally:
            server.traceback.print_exc = original_print_exc  # type: ignore[assignment]
            server._json_response = original_json_response  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
