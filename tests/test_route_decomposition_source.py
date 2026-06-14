import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
TRANSCRIPT_SEARCH_PY = ROOT / "codoxear" / "transcript_search.py"


class TestRouteDecompositionSource(unittest.TestCase):
    def test_handler_shares_url_prefix_parsing_between_get_and_post(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        self.assertIn("def _parse_prefixed_request_path(self)", source)
        self.assertEqual(source.count("self._parse_prefixed_request_path()"), 2)
        self.assertIn("def _handle_static_get(self, path: str) -> bool:", source)
        self.assertIn("if self._handle_static_get(path):", source)
        self.assertIn("def _handle_voice_get(self, path: str, query: str) -> bool:", source)
        self.assertIn("if self._handle_voice_get(path, u.query):", source)
        self.assertIn("def _handle_voice_post(self, path: str) -> bool:", source)
        self.assertIn("if self._handle_voice_post(path):", source)
        self.assertEqual(source.count("if path == URL_PREFIX:"), 1)

    def test_transcript_search_helpers_live_outside_server(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        module_source = TRANSCRIPT_SEARCH_PY.read_text(encoding="utf-8")
        self.assertIn("from .transcript_search import search_chat_log_bounded as _search_chat_log_bounded", server_source)
        self.assertIn("from .transcript_search import clip_search_match_text as _clip_search_match_text", server_source)
        self.assertIn("def search_chat_log_bounded(", module_source)
        self.assertIn("def search_chat_log(", module_source)
        self.assertIn("def clip_search_match_text(", module_source)
        self.assertNotIn("def _search_chat_log(", server_source)
        self.assertNotIn("def _clip_search_match_text(", server_source)


if __name__ == "__main__":
    unittest.main()
