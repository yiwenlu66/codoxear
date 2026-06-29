import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
FILE_VIEW_PY = ROOT / "codoxear" / "file_view.py"


class TestFileViewModuleSource(unittest.TestCase):
    def test_file_view_helpers_live_outside_server(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        module_source = FILE_VIEW_PY.read_text(encoding="utf-8")

        self.assertIn("from .file_view import inspect_downloadable_file as _inspect_downloadable_file", server_source)
        self.assertIn("from .file_view import read_client_file_view as _read_client_file_view", server_source)
        self.assertIn("from .file_view import download_disposition as _download_disposition", server_source)
        self.assertNotIn("class ClientFileView:", server_source)
        self.assertNotIn("def _read_client_file_view(", server_source)
        self.assertNotIn("def _inspect_openable_file(", server_source)
        self.assertNotIn("def _read_text_or_image(", server_source)
        self.assertNotIn("def _inspect_downloadable_file(", server_source)
        self.assertNotIn("def _download_disposition(", server_source)

        self.assertIn("class ClientFileView:", module_source)
        self.assertIn("def read_client_file_view(", module_source)
        self.assertIn("def inspect_openable_file(", module_source)
        self.assertIn("def read_text_or_image(", module_source)
        self.assertIn("def inspect_downloadable_file(", module_source)
        self.assertIn("def download_disposition(", module_source)
        self.assertIn("stat_path_no_symlink", module_source)
        self.assertIn("read_regular_file_prefix_no_symlink", module_source)
        self.assertIn("read_regular_file_bytes_no_symlink", module_source)
        self.assertNotIn("path_obj.read_bytes()", module_source)
        self.assertNotIn("path_obj.open(\"rb\")", module_source)
        download_start = module_source.index("def inspect_downloadable_file(")
        download_end = module_source.index("def inspect_client_path(", download_start)
        self.assertNotIn("read_bytes()", module_source[download_start:download_end])


if __name__ == "__main__":
    unittest.main()
