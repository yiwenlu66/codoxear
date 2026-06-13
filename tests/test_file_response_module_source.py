import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from codoxear.file_response import _stream_open_file_bytes
from codoxear.file_response import send_attachment_file_response
from codoxear.file_response import send_inline_file_response


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
FILE_RESPONSE_PY = ROOT / "codoxear" / "file_response.py"


class FakeHandler:
    def __init__(self) -> None:
        self.headers = {}
        self.status = None
        self.sent_headers = []
        self.wfile = io.BytesIO()
        self.error = None
        self.logged_errors = []

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        self.sent_headers.append((name, value))

    def end_headers(self) -> None:
        self.ended = True

    def send_error(self, status: int, message: str = "") -> None:
        self.status = status
        self.error = message

    def log_error(self, fmt: str, *args: object) -> None:
        self.logged_errors.append(fmt % args if args else fmt)


class BrokenWrite(io.BytesIO):
    def write(self, b: bytes) -> int:  # type: ignore[override]
        raise BrokenPipeError("client disconnected")


class FailingRead(io.BytesIO):
    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        raise OSError("disk read failed")


class TestFileResponseModuleSource(unittest.TestCase):
    def test_file_response_helpers_live_outside_server(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        module_source = FILE_RESPONSE_PY.read_text(encoding="utf-8")

        self.assertIn("from .file_response import send_attachment_file_response as _send_attachment_file_response", server_source)
        self.assertIn("from .file_response import send_inline_file_response as _send_inline_file_response", server_source)
        self.assertIn("from .file_response import single_byte_range as _single_byte_range", server_source)
        self.assertNotIn("def _single_byte_range(", server_source)
        self.assertNotIn("def _send_inline_file_response(", server_source)
        self.assertNotIn("def _send_attachment_file_response(", server_source)

        self.assertIn("def _open_file_for_response(", module_source)
        self.assertIn("def _stream_open_file_bytes(", module_source)
        self.assertIn("def _log_late_stream_error(", module_source)
        self.assertIn("def _stream_file_bytes(", module_source)
        self.assertIn("def single_byte_range(", module_source)
        self.assertIn("def send_inline_file_response(", module_source)
        self.assertIn("def send_attachment_file_response(", module_source)
        self.assertIn('handler.send_response(416)', module_source)
        self.assertIn('handler.send_header("Accept-Ranges", "bytes")', module_source)
        self.assertIn('handler.send_header("Cache-Control", "no-store")', module_source)
        self.assertIn('handler.send_header("Content-Disposition", content_disposition)', module_source)
        self.assertIn("def _open_file_size(", module_source)
        self.assertIn("os.fstat(f.fileno()).st_size", module_source)
        self.assertIn("_stream_open_file_bytes(handler, stream, length=length)", module_source)
        self.assertIn("file response stream failed after headers", module_source)

    def test_inline_response_maps_missing_file_before_headers(self) -> None:
        path = Path("/tmp/codoxear-missing-inline-response.bin")
        handler = FakeHandler()

        send_inline_file_response(handler, path, "application/octet-stream")

        self.assertEqual(handler.status, 404)
        self.assertEqual(handler.wfile.getvalue(), b"")

    def test_inline_response_maps_open_permission_before_headers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blob.bin"
            path.write_bytes(b"hello")
            handler = FakeHandler()
            original_open = Path.open
            try:
                Path.open = lambda self, *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied"))  # type: ignore[assignment]
                send_inline_file_response(handler, path, "application/octet-stream")
            finally:
                Path.open = original_open  # type: ignore[assignment]

        self.assertEqual(handler.status, 403)
        self.assertEqual(handler.error, "denied")
        self.assertEqual(handler.wfile.getvalue(), b"")

    def test_inline_response_uses_open_file_size_for_range_headers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blob.bin"
            raw = b"0123456789"
            path.write_bytes(raw)
            handler = FakeHandler()
            handler.headers = {"Range": "bytes=2-4"}
            original_stat = Path.stat
            try:
                Path.stat = lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("inline response must size the opened file"))  # type: ignore[assignment]
                send_inline_file_response(handler, path, "application/octet-stream")
            finally:
                Path.stat = original_stat  # type: ignore[assignment]
            self.assertEqual(handler.status, 206)
            self.assertIn(("Content-Length", "3"), handler.sent_headers)
            self.assertIn(("Content-Range", f"bytes 2-4/{len(raw)}"), handler.sent_headers)
            self.assertEqual(handler.wfile.getvalue(), raw[2:5])

    def test_attachment_response_caps_stale_declared_size_to_open_file_size(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blob.bin"
            raw = b"short"
            path.write_bytes(raw)
            handler = FakeHandler()
            send_attachment_file_response(
                handler,
                path,
                size=len(raw) + 10,
                content_disposition="attachment; filename*=UTF-8''blob.bin",
            )
            self.assertEqual(handler.status, 200)
            self.assertIn(("Content-Length", str(len(raw))), handler.sent_headers)
            self.assertEqual(handler.wfile.getvalue(), raw)

    def test_inline_response_logs_late_write_error_after_headers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blob.bin"
            path.write_bytes(b"hello")
            handler = FakeHandler()
            handler.wfile = BrokenWrite()

            send_inline_file_response(handler, path, "application/octet-stream")

            self.assertEqual(handler.status, 200)
            self.assertTrue(handler.ended)
            self.assertEqual(handler.error, None)
            self.assertTrue(any("client disconnected" in msg for msg in handler.logged_errors))

    def test_stream_open_file_bytes_logs_late_read_error(self) -> None:
        handler = FakeHandler()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            _stream_open_file_bytes(handler, FailingRead())

        self.assertTrue(any("disk read failed" in msg for msg in handler.logged_errors))
        self.assertIn("error: file response stream failed after headers: OSError: disk read failed", stderr.getvalue())
        self.assertEqual(handler.wfile.getvalue(), b"")

    def test_attachment_response_streams_without_read_bytes_and_caps_to_declared_size(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blob.bin"
            raw = (b"0123456789abcdef" * 1024) + b"tail"
            path.write_bytes(raw)
            declared_size = len(raw) - 4
            handler = FakeHandler()
            original_read_bytes = Path.read_bytes
            try:
                Path.read_bytes = lambda self: (_ for _ in ()).throw(AssertionError("attachment response must stream"))  # type: ignore[assignment]
                send_attachment_file_response(
                    handler,
                    path,
                    size=declared_size,
                    content_disposition="attachment; filename*=UTF-8''blob.bin",
                )
            finally:
                Path.read_bytes = original_read_bytes  # type: ignore[assignment]
            self.assertEqual(handler.status, 200)
            self.assertIn(("Content-Length", str(declared_size)), handler.sent_headers)
            self.assertIn(("Content-Disposition", "attachment; filename*=UTF-8''blob.bin"), handler.sent_headers)
            self.assertEqual(handler.wfile.getvalue(), raw[:declared_size])


if __name__ == "__main__":
    unittest.main()
