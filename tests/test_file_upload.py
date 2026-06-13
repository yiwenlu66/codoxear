import base64
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import patch

from codoxear import pty_util
from codoxear import server
from codoxear.server import _attachment_inject_text
from codoxear.server import _stage_uploaded_file


class TestStageUploadedFile(unittest.TestCase):
    def test_stage_uploaded_file_preserves_binary_bytes_and_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            upload_root = Path(td)
            with patch("codoxear.server.UPLOAD_DIR", upload_root), patch("codoxear.server._now", return_value=1234.567):
                path = _stage_uploaded_file("sess-1", "../../payload.tar.gz", b"\x00\x01payload")

            self.assertEqual(path, upload_root / "sess-1" / "1234567_payload.tar.gz")
            self.assertEqual(path.read_bytes(), b"\x00\x01payload")

    def test_stage_uploaded_file_falls_back_to_generic_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            upload_root = Path(td)
            with patch("codoxear.server.UPLOAD_DIR", upload_root), patch("codoxear.server._now", return_value=2.0):
                path = _stage_uploaded_file("sess-2", "///", b"abc")

            self.assertEqual(path.name, "2000_file")
            self.assertEqual(path.parent, upload_root / "sess-2")

    def test_stage_uploaded_file_rejects_oversize_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            upload_root = Path(td)
            with patch("codoxear.server.UPLOAD_DIR", upload_root):
                with self.assertRaisesRegex(ValueError, "file too large"):
                    _stage_uploaded_file("sess-3", "big.bin", b"abcd", max_bytes=3)

    def test_stage_uploaded_file_preserves_unicode_filename(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            upload_root = Path(td)
            with patch("codoxear.server.UPLOAD_DIR", upload_root), patch("codoxear.server._now", return_value=3.0):
                path = _stage_uploaded_file("sess-4", "南京大学_程元_简历.pdf", b"pdf")

            self.assertEqual(path.name, "3000_南京大学_程元_简历.pdf")
            self.assertEqual(path.read_bytes(), b"pdf")

    def test_attachment_inject_text_uses_readable_label_and_newline(self) -> None:
        text = _attachment_inject_text(2, Path("/tmp/example.txt"))
        self.assertEqual(text, "Attachment 2: /tmp/example.txt\n")

    def test_attachment_inject_text_rejects_non_positive_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "attachment_index must be >= 1"):
            _attachment_inject_text(0, Path("/tmp/example.txt"))


class TestInjectFileRoute(unittest.TestCase):
    def test_valid_base64_payload_is_staged_and_injected(self) -> None:
        class FakeManager:
            def __init__(self) -> None:
                self.ready_calls = []
                self.inject_calls = []

            def attachment_injection_ready(self, session_id: str) -> bool:
                self.ready_calls.append(session_id)
                return True

            def inject_attachment_keys(self, session_id: str, seq: str) -> dict:
                self.inject_calls.append((session_id, seq))
                return {"ok": True}

        fake_manager = FakeManager()
        handler = server.Handler.__new__(server.Handler)
        parsed = urllib.parse.urlparse("/api/sessions/sess-1/inject_file")
        handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
        handler._handle_voice_post = lambda _path: False  # type: ignore[attr-defined]
        handler._read_json_body = lambda **_kwargs: {  # type: ignore[attr-defined]
            "filename": "note.txt",
            "attachment_index": 1,
            "data_b64": base64.b64encode(b"hello attachment").decode("ascii"),
        }
        responses = []
        with tempfile.TemporaryDirectory() as td:
            upload_root = Path(td) / "uploads"
            with patch.object(server, "MANAGER", fake_manager), patch.object(server, "UPLOAD_DIR", upload_root), patch.object(
                server, "_require_auth", return_value=True
            ), patch.object(server, "_now", return_value=42.0), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_POST(handler)

            staged = upload_root / "sess-1" / "42000_note.txt"
            self.assertEqual(staged.read_bytes(), b"hello attachment")

        self.assertEqual(fake_manager.ready_calls, ["sess-1"])
        self.assertEqual(len(fake_manager.inject_calls), 1)
        session_id, seq = fake_manager.inject_calls[0]
        self.assertEqual(session_id, "sess-1")
        self.assertIn("\x1b[200~Attachment 1: ", seq)
        self.assertIn("42000_note.txt\n\x1b[201~", seq)
        self.assertEqual(responses, [(200, {"ok": True, "path": str(staged), "inject_text": f"Attachment 1: {staged}\n", "broker": {"ok": True}})])


class TestSeqBytes(unittest.TestCase):
    def test_seq_bytes_preserves_unicode_text(self) -> None:
        raw = "Attachment 1: /tmp/南京大学_程元_简历.pdf\n"
        self.assertEqual(pty_util.seq_bytes(raw), raw.encode("utf-8"))

    def test_seq_bytes_decodes_escape_only_sequences(self) -> None:
        self.assertEqual(pty_util.seq_bytes("\\x1b"), b"\x1b")


if __name__ == "__main__":
    unittest.main()
