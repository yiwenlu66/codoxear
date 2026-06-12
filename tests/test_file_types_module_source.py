import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
FILE_TYPES_PY = ROOT / "codoxear" / "file_types.py"


class TestFileTypesModuleSource(unittest.TestCase):
    def test_file_type_classification_lives_outside_server(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        module_source = FILE_TYPES_PY.read_text(encoding="utf-8")

        self.assertIn("from .file_types import file_kind as _file_kind", server_source)
        self.assertIn("from .file_types import sniff_image_ext as _sniff_image_ext", server_source)
        self.assertNotIn("def _image_content_type(", server_source)
        self.assertNotIn("def _pdf_content_type(", server_source)
        self.assertNotIn("def _video_content_type(", server_source)
        self.assertNotIn("def _file_kind(", server_source)

        self.assertIn("def image_content_type(", module_source)
        self.assertIn("def pdf_content_type(", module_source)
        self.assertIn("def video_content_type(", module_source)
        self.assertIn("def file_kind(", module_source)


if __name__ == "__main__":
    unittest.main()
