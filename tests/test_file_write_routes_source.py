import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILE_ROUTES = ROOT / "codoxear" / "file_routes.py"
FILE_WRITE_ROUTES = ROOT / "codoxear" / "file_write_routes.py"
FILE_ROUTE_COMMON = ROOT / "codoxear" / "file_route_common.py"


class TestFileWriteRoutesSource(unittest.TestCase):
    def test_write_route_logic_lives_outside_file_routes_facade(self) -> None:
        facade = FILE_ROUTES.read_text(encoding="utf-8")
        write_source = FILE_WRITE_ROUTES.read_text(encoding="utf-8")
        common_source = FILE_ROUTE_COMMON.read_text(encoding="utf-8")

        self.assertIn("from .file_write_routes import handle_file_write_post_route", facade)
        self.assertIn("from .file_write_routes import session_file_write_response", facade)
        self.assertIn("from .file_write_routes import write_session_file", facade)
        self.assertIn("from .file_route_common import FileRouteResponse", facade)
        self.assertIn("from .file_route_common import resolve_session_write_update_path", facade)

        self.assertNotIn("def handle_file_write_post_route(", facade)
        self.assertNotIn("def parse_session_file_write_request(", facade)
        self.assertNotIn("def write_session_file(", facade)
        self.assertNotIn("def _create_session_file(", facade)
        self.assertNotIn("def _update_session_file(", facade)

        self.assertIn("class FileWriteRouteDeps:", write_source)
        self.assertIn("def handle_file_write_post_route(", write_source)
        self.assertIn("def parse_session_file_write_request(", write_source)
        self.assertIn("def session_file_write_response(", write_source)
        self.assertIn("def write_session_file(", write_source)
        self.assertIn("def _create_session_file(", write_source)
        self.assertIn("def _update_session_file(", write_source)
        self.assertIn('raise FileRouteError(400, {"error": "git_path is only supported for existing files"})', write_source)
        self.assertIn('raise FileRouteError(409, payload) from e', write_source)
        self.assertIn('raise FileRouteError(409, {"error": str(e)}) from e', write_source)
        self.assertIn("record_file(str(path_obj))", write_source)

        self.assertIn("class FileRouteResponse:", common_source)
        self.assertIn("class FileRouteError(Exception):", common_source)
        self.assertIn("class SessionFileWriteRequest:", common_source)
        self.assertIn("def body_flag(obj: Mapping[str, Any], name: str) -> bool:", common_source)
        self.assertIn("def resolve_session_write_update_path(base: Path, raw_path: str) -> Path:", common_source)
        self.assertIn('raise ValueError("path escapes session cwd") from e', common_source)


if __name__ == "__main__":
    unittest.main()
