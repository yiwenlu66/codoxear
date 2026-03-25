import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codoxear import pty_util
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


class TestSeqBytes(unittest.TestCase):
    def test_seq_bytes_preserves_unicode_text(self) -> None:
        raw = "Attachment 1: /tmp/南京大学_程元_简历.pdf\n"
        self.assertEqual(pty_util.seq_bytes(raw), raw.encode("utf-8"))

    def test_seq_bytes_decodes_escape_only_sequences(self) -> None:
        self.assertEqual(pty_util.seq_bytes("\\x1b"), b"\x1b")


if __name__ == "__main__":
    unittest.main()
