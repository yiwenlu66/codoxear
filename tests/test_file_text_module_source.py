import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
FILE_TEXT_PY = ROOT / "codoxear" / "file_text.py"


class TestFileTextModuleSource(unittest.TestCase):
    def test_file_text_helpers_live_outside_server(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        module_source = FILE_TEXT_PY.read_text(encoding="utf-8")

        self.assertIn("from .file_text import read_text_file_for_client as _read_text_file_for_client", server_source)
        self.assertIn("from .file_text import write_text_file_atomic as _write_text_file_atomic", server_source)
        self.assertNotIn("def _read_text_file_for_client(", server_source)
        self.assertNotIn("def _read_text_file_for_write(", server_source)
        self.assertNotIn("def _write_text_file_atomic(", server_source)
        self.assertNotIn("def _write_new_text_file_atomic(", server_source)
        self.assertNotIn("def _decode_text_view_for_client(", server_source)

        self.assertIn("def read_text_file_for_client(", module_source)
        self.assertIn("def read_text_file_for_write(", module_source)
        self.assertIn("def write_text_file_atomic(", module_source)
        self.assertIn("def write_new_text_file_atomic(", module_source)
        self.assertIn("def decode_text_view_for_client(", module_source)


if __name__ == "__main__":
    unittest.main()
