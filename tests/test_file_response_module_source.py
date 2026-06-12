import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
FILE_RESPONSE_PY = ROOT / "codoxear" / "file_response.py"


class TestFileResponseModuleSource(unittest.TestCase):
    def test_file_response_helpers_live_outside_server(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        module_source = FILE_RESPONSE_PY.read_text(encoding="utf-8")

        self.assertIn("from .file_response import send_inline_file_response as _send_inline_file_response", server_source)
        self.assertIn("from .file_response import single_byte_range as _single_byte_range", server_source)
        self.assertNotIn("def _single_byte_range(", server_source)
        self.assertNotIn("def _send_inline_file_response(", server_source)

        self.assertIn("def single_byte_range(", module_source)
        self.assertIn("def send_inline_file_response(", module_source)
        self.assertIn('handler.send_response(416)', module_source)
        self.assertIn('handler.send_header("Accept-Ranges", "bytes")', module_source)
        self.assertIn('handler.send_header("Cache-Control", "no-store")', module_source)


if __name__ == "__main__":
    unittest.main()
