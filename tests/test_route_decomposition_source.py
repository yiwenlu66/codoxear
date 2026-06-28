import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
SERVER_HANDLER_PY = ROOT / "codoxear" / "server_handler.py"
TRANSCRIPT_SEARCH_PY = ROOT / "codoxear" / "transcript_search.py"
STATIC_ROUTES_PY = ROOT / "codoxear" / "static_routes.py"
HOOK_ROUTES_PY = ROOT / "codoxear" / "hook_routes.py"
VOICE_ROUTES_PY = ROOT / "codoxear" / "voice_routes.py"


class TestRouteDecompositionSource(unittest.TestCase):
    def test_handler_shares_url_prefix_parsing_between_get_and_post(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        handler_source = SERVER_HANDLER_PY.read_text(encoding="utf-8")
        static_route_source = STATIC_ROUTES_PY.read_text(encoding="utf-8")
        hook_route_source = HOOK_ROUTES_PY.read_text(encoding="utf-8")
        voice_route_source = VOICE_ROUTES_PY.read_text(encoding="utf-8")
        self.assertIn("def _parse_prefixed_request_path(self)", handler_source)
        self.assertEqual(handler_source.count("self._parse_prefixed_request_path()"), 2)
        self.assertIn("def _handle_static_get(self, path: str) -> bool:", handler_source)
        self.assertIn("return handle_static_get_route(", handler_source)
        self.assertIn("if self._handle_static_get(path):", handler_source)
        self.assertIn("def handle_static_get_route(", static_route_source)
        self.assertIn("def send_static_file(", static_route_source)
        self.assertNotIn("def _send_static(self, rel: str) -> None:", source)
        self.assertIn("def handle_hook_post_route(", hook_route_source)
        self.assertIn('if path != "/api/hooks/notify":', hook_route_source)
        self.assertIn("if handle_hook_post_route(", handler_source)
        self.assertNotIn('if path == "/api/hooks/notify":', source)
        self.assertIn("def _handle_voice_get(self, path: str, query: str) -> bool:", handler_source)
        self.assertIn("return handle_voice_get_route(", handler_source)
        self.assertIn("if self._handle_voice_get(path, url.query):", handler_source)
        self.assertIn("def _handle_voice_post(self, path: str) -> bool:", handler_source)
        self.assertIn("return handle_voice_post_route(", handler_source)
        self.assertIn("if self._handle_voice_post(path):", handler_source)
        self.assertIn("def handle_voice_get_route(", voice_route_source)
        self.assertIn("def handle_voice_post_route(", voice_route_source)
        self.assertIn('if path == "/api/settings/voice":', voice_route_source)
        self.assertIn('if path == "/api/audio/listener":', voice_route_source)
        self.assertNotIn('if path == "/api/settings/voice":', source)
        self.assertEqual(handler_source.count("if path == self.deps.url_prefix:"), 1)

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
