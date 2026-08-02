import io
import tempfile
from pathlib import Path
from unittest.mock import patch

from codoxear import server


class FakeHandler:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.sent_headers: list[tuple[str, str]] = []
        self.wfile = io.BytesIO()

    def send_response(self, _status: int) -> None:
        pass

    def send_header(self, name: str, value: str) -> None:
        self.sent_headers.append((name, value))

    def end_headers(self) -> None:
        pass

    def send_error(self, status: int, message: str = "") -> None:
        raise AssertionError(f"unexpected {status}: {message}")


def test_uploaded_inline_image_route_uses_immutable_private_cache_headers_only_for_uploads() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        upload = root / "uploads" / "session-1" / "chart.png"
        upload.parent.mkdir(parents=True)
        upload.write_bytes(b"png")
        mutable = root / "workspace" / "chart.png"
        mutable.parent.mkdir()
        mutable.write_bytes(b"png")

        with patch.object(server, "UPLOAD_DIR", root / "uploads"):
            upload_handler = FakeHandler()
            server._send_inline_file_response(upload_handler, upload, "image/png")
            mutable_handler = FakeHandler()
            server._send_inline_file_response(mutable_handler, mutable, "image/png")

    assert ("Cache-Control", "private, max-age=31536000, immutable") in upload_handler.sent_headers
    assert ("Pragma", "no-cache") not in upload_handler.sent_headers
    assert ("Expires", "0") not in upload_handler.sent_headers
    assert ("Cache-Control", "no-store") in mutable_handler.sent_headers
    assert ("Pragma", "no-cache") in mutable_handler.sent_headers
    assert ("Expires", "0") in mutable_handler.sent_headers
