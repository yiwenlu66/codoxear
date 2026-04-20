import unittest
from pathlib import Path

from codoxear import server as server_module
from codoxear.http.routes import assets as assets_routes
from codoxear.http.routes import auth as auth_routes
from codoxear.http.routes import files as file_routes
from codoxear.http.routes import notifications as notification_routes
from codoxear.http.routes import sessions_read as session_read_routes
from codoxear.http.routes import sessions_write as session_write_routes
from codoxear.pi import ui_bridge as pi_ui_bridge
from codoxear.sessions import live_payloads as live_payloads
from codoxear.sessions import payloads as payloads
from codoxear.sessions import sidebar_state as sidebar_state


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
        self.assertIn("from .sessions import sidebar_state as _sidebar_state_module", source)
        self.assertIn("from .pi import ui_bridge as _pi_ui_bridge", source)
        self.assertIn("self._sidebar_state_facade().persist_session_ui_state()", source)
        self.assertIn("return _pi_ui_bridge.submit_ui_response(self, session_id, payload)", source)

    def test_seam_modules_bind_to_loaded_server_runtime(self) -> None:
        modules = [
            assets_routes,
            auth_routes,
            file_routes,
            notification_routes,
            session_read_routes,
            session_write_routes,
            payloads,
            live_payloads,
            sidebar_state,
            pi_ui_bridge,
        ]
        for module in modules:
            self.assertIs(module._sv(), server_module)

    def test_seam_modules_support_explicit_runtime_rebind(self) -> None:
        modules = [
            assets_routes,
            auth_routes,
            file_routes,
            notification_routes,
            session_read_routes,
            session_write_routes,
            payloads,
            live_payloads,
            sidebar_state,
            pi_ui_bridge,
        ]
        sentinel = object()
        for module in modules:
            module.bind_server_runtime(sentinel)
            self.assertIs(module._sv(), sentinel)
            module.bind_server_runtime(server_module)
            self.assertIs(module._sv(), server_module)

    def test_server_runtime_exposes_pi_context_window_helper(self) -> None:
        self.assertTrue(callable(getattr(server_module, "_pi_model_context_window", None)))

    def test_voice_push_uses_attention_namespace(self) -> None:
        source = VOICE_PUSH.read_text(encoding="utf-8")
        self.assertIn("from .attention.derive import", source)
        self.assertIn("compact_notification_state", source)
        self.assertIn("final_response_attention_feed", source)
        self.assertIn("return final_response_attention_feed(", source)
        self.assertNotIn("from codoxear import server as sv", source)


if __name__ == "__main__":
    unittest.main()
