import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestSessionPollingSource(unittest.TestCase):
    def test_session_polling_is_visibility_aware_timeout_loop(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("const SESSION_POLL_VISIBLE_MS = 2500;", source)
        self.assertIn("const SESSION_POLL_HIDDEN_MS = 15000;", source)
        self.assertIn('document.visibilityState === "hidden" ? SESSION_POLL_HIDDEN_MS : SESSION_POLL_VISIBLE_MS', source)
        self.assertIn("function scheduleSessionsPoll(delayMs = sessionsPollDelayMs())", source)
        self.assertIn("sessionsTimer = setTimeout", source)
        self.assertNotIn("sessionsTimer = setInterval", source)

    def test_visibility_change_refreshes_immediately_when_visible(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('if (document.visibilityState === "visible") {', source)
        self.assertIn("scheduleSessionsPoll(0);", source)
        self.assertIn("scheduleSessionsPoll(sessionsPollDelayMs());", source)

    def test_session_polling_stops_on_auth_loss_and_unload(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("sessionsPollingEnabled = false;", source)
        self.assertIn("stopSessionsPolling();", source)
        self.assertIn("renderLogin(renderApp);", source)


if __name__ == "__main__":
    unittest.main()
