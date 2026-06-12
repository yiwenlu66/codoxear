import unittest
from pathlib import Path


SERVER_PY = Path(__file__).resolve().parents[1] / "codoxear" / "server.py"


class TestRouteDecompositionSource(unittest.TestCase):
    def test_handler_shares_url_prefix_parsing_between_get_and_post(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        self.assertIn("def _parse_prefixed_request_path(self)", source)
        self.assertEqual(source.count("self._parse_prefixed_request_path()"), 2)
        self.assertIn("def _handle_static_get(self, path: str) -> bool:", source)
        self.assertIn("if self._handle_static_get(path):", source)
        self.assertEqual(source.count("if path == URL_PREFIX:"), 1)


if __name__ == "__main__":
    unittest.main()
