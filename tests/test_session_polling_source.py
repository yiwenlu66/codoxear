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

    def test_sessions_api_uses_etag_cache_for_unchanged_polls(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("const apiEtags = new Map();", source)
        self.assertIn('const cacheableSessionsRequest = method === "GET" && rawPath === "/api/sessions";', source)
        self.assertIn('opts.headers["If-None-Match"] = apiEtags.get(rawPath).etag;', source)
        self.assertIn("if (res.status === 304 && cacheableSessionsRequest && apiEtags.has(rawPath)) {", source)
        self.assertIn("return JSON.parse(apiEtags.get(rawPath).text);", source)
        self.assertIn('const etag = cacheableSessionsRequest ? res.headers.get("ETag") : null;', source)
        self.assertIn("if (etag) apiEtags.set(rawPath, { etag, text: txt });", source)

    def test_secondary_polling_is_decoupled_from_session_polling(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("const SECONDARY_POLL_VISIBLE_MS = 10000;", source)
        self.assertIn("const SECONDARY_POLL_HIDDEN_MS = 60000;", source)
        self.assertIn('document.visibilityState === "hidden" ? SECONDARY_POLL_HIDDEN_MS : SECONDARY_POLL_VISIBLE_MS', source)
        self.assertIn("function scheduleSecondaryPoll(delayMs = secondaryPollDelayMs())", source)
        self.assertIn("secondaryPollTimer = setTimeout", source)
        session_tick = source[source.index("async function runSessionsPollTick()") : source.index("async function runSecondaryPollTick()")]
        self.assertIn("await refreshSessions();", session_tick)
        self.assertNotIn("loadVoiceSettings", session_tick)
        self.assertNotIn("syncNotificationState", session_tick)
        self.assertNotIn("pollNotificationFeed", session_tick)
        secondary_tick = source[source.index("async function runSecondaryPollTick()") : source.index("function scheduleSessionsPoll")]
        self.assertIn("await loadVoiceSettings();", secondary_tick)
        self.assertIn("await syncNotificationState();", secondary_tick)
        self.assertIn("await pollNotificationFeed();", secondary_tick)

    def test_visibility_change_refreshes_immediately_when_visible(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('if (document.visibilityState === "visible") {', source)
        self.assertIn("scheduleSessionsPoll(0);", source)
        self.assertIn("scheduleSecondaryPoll(0);", source)
        self.assertIn("scheduleSessionsPoll(sessionsPollDelayMs());", source)
        self.assertIn("scheduleSecondaryPoll(secondaryPollDelayMs());", source)

    def test_session_polling_stops_on_auth_loss_and_unload(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function handlePollingAuthLoss()", source)
        self.assertIn("sessionsPollingEnabled = false;", source)
        self.assertIn("secondaryPollingEnabled = false;", source)
        self.assertIn("stopAllPolling();", source)
        self.assertIn("stopSessionsPolling();", source)
        self.assertIn("stopSecondaryPolling();", source)
        self.assertIn("renderLogin(renderApp);", source)


if __name__ == "__main__":
    unittest.main()
