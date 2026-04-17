import errno
import unittest
from typing import Any
from typing import cast
from unittest.mock import patch

from codoxear.server import _handle_handler_exception


class _WriteBuffer:
    def __init__(self, exc: BaseException | None = None) -> None:
        self.exc = exc
        self.calls = 0
        self.payloads: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        self.payloads.append(data)


class _HandlerStub:
    def __init__(self, *, write_exc: BaseException | None = None, end_headers_exc: BaseException | None = None) -> None:
        self.wfile = _WriteBuffer(write_exc)
        self._end_headers_exc = end_headers_exc
        self.statuses: list[int] = []
        self.headers: list[tuple[str, str]] = []
        self.end_headers_calls = 0

    def send_response(self, status: int) -> None:
        self.statuses.append(status)

    def send_header(self, name: str, value: str) -> None:
        self.headers.append((name, value))

    def end_headers(self) -> None:
        self.end_headers_calls += 1
        if self._end_headers_exc is not None:
            raise self._end_headers_exc


class TestServerDisconnectHandling(unittest.TestCase):
    def test_client_disconnect_is_ignored_without_traceback(self) -> None:
        handler = _HandlerStub()

        with patch("codoxear.server.traceback.print_exc") as print_exc:
            _handle_handler_exception(cast(Any, handler), BrokenPipeError(errno.EPIPE, "Broken pipe"))

        self.assertEqual(handler.statuses, [])
        self.assertEqual(handler.end_headers_calls, 0)
        self.assertEqual(handler.wfile.calls, 0)
        print_exc.assert_not_called()

    def test_disconnect_while_sending_error_response_stays_quiet(self) -> None:
        handler = _HandlerStub(write_exc=BrokenPipeError(errno.EPIPE, "Broken pipe"))

        with patch("codoxear.server.traceback.print_exc") as print_exc:
            try:
                raise RuntimeError("boom")
            except Exception as exc:
                _handle_handler_exception(cast(Any, handler), exc)

        self.assertEqual(handler.statuses, [500])
        self.assertEqual(handler.end_headers_calls, 1)
        self.assertEqual(handler.wfile.calls, 1)
        print_exc.assert_called_once()


if __name__ == "__main__":
    unittest.main()
