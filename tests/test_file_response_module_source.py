import io
import tempfile
import unittest
from pathlib import Path

from codoxear.file_response import send_attachment_file_response


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
FILE_RESPONSE_PY = ROOT / "codoxear" / "file_response.py"


class FakeHandler:
    def __init__(self) -> None:
        self.headers = {}
        self.status = None
        self.sent_headers = []
        self.wfile = io.BytesIO()

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        self.sent_headers.append((name, value))

    def end_headers(self) -> None:
        self.ended = True


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

        self.assertIn("def _stream_file_bytes(", module_source)
        self.assertIn("def single_byte_range(", module_source)
        self.assertIn("def send_inline_file_response(", module_source)
        self.assertIn("def send_attachment_file_response(", module_source)
        self.assertIn('handler.send_response(416)', module_source)
        self.assertIn('handler.send_header("Accept-Ranges", "bytes")', module_source)
        self.assertIn('handler.send_header("Cache-Control", "no-store")', module_source)
        self.assertIn('handler.send_header("Content-Disposition", content_disposition)', module_source)
        self.assertIn("_stream_file_bytes(handler, path, length=size)", module_source)

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
