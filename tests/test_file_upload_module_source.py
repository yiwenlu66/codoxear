import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
FILE_UPLOAD_PY = ROOT / "codoxear" / "file_upload.py"


class TestFileUploadModuleSource(unittest.TestCase):
    def test_upload_helpers_live_outside_server_with_server_state_injection(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        module_source = FILE_UPLOAD_PY.read_text(encoding="utf-8")

        self.assertIn("from .file_upload import safe_filename as _safe_filename", server_source)
        self.assertIn("from .file_upload import stage_uploaded_file as _stage_uploaded_file_impl", server_source)
        self.assertIn("from .file_upload import attachment_inject_text as _attachment_inject_text", server_source)
        self.assertIn("upload_dir=UPLOAD_DIR", server_source)
        self.assertIn("now_fn=_now", server_source)
        self.assertNotIn("def _safe_filename(", server_source)
        self.assertNotIn("def _attachment_inject_text(", server_source)

        self.assertIn("def safe_filename(", module_source)
        self.assertIn("def stage_uploaded_file(", module_source)
        self.assertIn("def attachment_inject_text(", module_source)
        self.assertIn("upload_dir: Path", module_source)
        self.assertIn("now_fn: Callable[[], float]", module_source)


if __name__ == "__main__":
    unittest.main()
