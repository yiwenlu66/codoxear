import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILE_ROUTES = ROOT / "codoxear" / "file_routes.py"
FILE_GLOBAL_ROUTES = ROOT / "codoxear" / "file_global_routes.py"


class TestFileGlobalRoutesSource(unittest.TestCase):
    def test_global_file_routes_have_dedicated_owner_with_file_routes_facade(self) -> None:
        facade = FILE_ROUTES.read_text(encoding="utf-8")
        global_source = FILE_GLOBAL_ROUTES.read_text(encoding="utf-8")

        self.assertIn("from .file_global_routes import GlobalFileRequest", facade)
        self.assertIn("from .file_global_routes import GlobalFileRouteDeps", facade)
        self.assertIn("from .file_global_routes import global_file_read_payload", facade)
        self.assertIn("from .file_global_routes import handle_global_file_post_route", facade)
        self.assertNotIn("class GlobalFileRequest", facade)
        self.assertNotIn("class GlobalFileRouteDeps", facade)
        self.assertNotIn("def handle_global_file_post_route(", facade)
        self.assertNotIn("def global_file_read_payload(", facade)
        self.assertNotIn("def _global_file_request(", facade)

        self.assertIn("class GlobalFileRequest:", global_source)
        self.assertIn("class GlobalFileRouteDeps:", global_source)
        self.assertIn("def handle_global_file_post_route(", global_source)
        self.assertIn("def _global_file_request(", global_source)
        self.assertIn("def _global_file_view(", global_source)
        self.assertIn("def _handle_global_file_read(", global_source)
        self.assertIn("def _handle_global_file_inspect(", global_source)
        self.assertIn("def global_file_read_payload(", global_source)
        self.assertIn('deps.json_response(handler, 400, {"error": "path required"})', global_source)
        self.assertIn('deps.json_response(handler, 400, {"error": "session_id must be a string"})', global_source)
        self.assertIn('manager.files_add(request.session_id, path_json_text(path_obj))', global_source)
        self.assertIn('except KeyError:', global_source)
        self.assertIn("preview_url=media_preview_url", global_source)


if __name__ == "__main__":
    unittest.main()
