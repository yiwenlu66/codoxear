import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILE_ROUTES = ROOT / "codoxear" / "file_routes.py"
FILE_GET_ROUTES = ROOT / "codoxear" / "file_get_routes.py"


class TestFileGetRoutesSource(unittest.TestCase):
    def test_get_routes_have_dedicated_owner_with_file_routes_facade(self) -> None:
        facade = FILE_ROUTES.read_text(encoding="utf-8")
        get_source = FILE_GET_ROUTES.read_text(encoding="utf-8")

        for name in (
            "FileGetRouteDeps",
            "JsonResponse",
            "RouteMatcher",
            "handle_absolute_file_preview_route",
            "handle_file_get_route",
            "session_file_read_payload",
        ):
            self.assertIn(f"from .file_get_routes import {name}", facade)
            self.assertIn(name, get_source)

        self.assertNotIn("class FileGetRouteDeps", facade)
        self.assertNotIn("def handle_file_get_route(", facade)
        self.assertNotIn("def handle_absolute_file_preview_route(", facade)
        self.assertNotIn("def session_file_read_payload(", facade)
        self.assertNotIn("def _handle_session_file_read(", facade)
        self.assertNotIn("def _send_video_preview(", facade)

        self.assertIn("class FileGetRouteDeps:", get_source)
        self.assertIn("def handle_file_get_route(", get_source)
        self.assertIn('match_session_route(path, "file", "read")', get_source)
        self.assertIn('match_session_route(path, "file", "search")', get_source)
        self.assertIn('match_session_route(path, "file", "list")', get_source)
        self.assertIn('match_session_route(path, "file", "blob")', get_source)
        self.assertIn('match_session_route(path, "file", "video_preview")', get_source)
        self.assertIn('match_session_route(path, "file", "download")', get_source)
        self.assertIn('path == "/api/files/blob"', get_source)
        self.assertIn('path == "/api/files/video_preview"', get_source)
        self.assertIn('manager.refresh_session_meta(session_id)', get_source)
        self.assertIn('manager.files_add(session_id, path_json_text(path_obj), api_path=rel_token)', get_source)
        self.assertIn('except KeyError:', get_source)
        self.assertIn('deps.json_response(handler, 400, {"error": f"{name} required"})', get_source)
        self.assertIn('deps.json_response(handler, 500, {"error": f"video preview failed: {e}"})', get_source)
        self.assertIn('read_regular_file_prefix: Callable[[Path, int], tuple[bytes, int]]', get_source)
        self.assertIn('prefix, _size = deps.read_regular_file_prefix(path_obj, 4096)', get_source)
        self.assertNotIn('path_obj.open("rb")', get_source)
        self.assertIn('deps.send_attachment_file_response(handler, path_obj, size=size, content_disposition=deps.download_disposition(path_obj))', get_source)


if __name__ == "__main__":
    unittest.main()
