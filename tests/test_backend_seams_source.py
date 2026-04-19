import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "codoxear"
SERVER = ROOT / "server.py"
VOICE_PUSH = ROOT / "voice_push.py"


class TestBackendSeamsSource(unittest.TestCase):
    def test_server_dispatches_http_routes_through_modules(self) -> None:
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn("from .http.routes import assets as _http_assets_routes", source)
        self.assertIn("from .http.routes import sessions_read as _http_session_read_routes", source)
        self.assertIn("route_module.handle_get(self, path, u)", source)
        self.assertIn("route_module.handle_post(self, path, u)", source)

    def test_server_uses_payload_sidebar_and_pi_bridge_seams(self) -> None:
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn("from .sessions import payloads as _session_payloads", source)
        self.assertIn("from .sessions import live_payloads as _session_live_payloads", source)
        self.assertIn("from .sessions.sidebar_state import SidebarStateFacade", source)
        self.assertIn("from .pi import ui_bridge as _pi_ui_bridge", source)
        self.assertIn("self._sidebar_state_facade().persist_session_ui_state()", source)
        self.assertIn("return _pi_ui_bridge.submit_ui_response(self, session_id, payload)", source)

    def test_voice_push_uses_attention_namespace(self) -> None:
        source = VOICE_PUSH.read_text(encoding="utf-8")
        self.assertIn("from .attention.derive import compact_notification_state", source)
        self.assertIn("from .attention.derive import final_response_attention_feed", source)
        self.assertIn("return final_response_attention_feed(", source)


if __name__ == "__main__":
    unittest.main()
