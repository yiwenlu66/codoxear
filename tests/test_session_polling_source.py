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
        self.assertIn("const API_NOT_MODIFIED = Symbol(\"api.notModified\");", source)
        self.assertIn("function apiResponseNotModified(obj) {", source)
        self.assertIn("if (res.status === 304 && cacheableSessionsRequest && apiEtags.has(rawPath)) {", source)
        self.assertIn("Object.defineProperty(cached, API_NOT_MODIFIED, { value: true });", source)
        self.assertIn("if (apiResponseNotModified(data)) return latestSessions;", source)
        self.assertIn('const etag = cacheableSessionsRequest ? res.headers.get("ETag") : null;', source)
        self.assertIn("if (etag) apiEtags.set(rawPath, { etag, text: txt });", source)

    def test_sessions_304_fast_path_precedes_sidebar_rebuild(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        block = source[source.index("async function refreshSessions()") : source.index("function appendEvent")]
        self.assertLess(block.index("if (apiResponseNotModified(data)) return latestSessions;"), block.index('sessionsWrap.innerHTML = "";'))
        self.assertLess(block.index("if (apiResponseNotModified(data)) return latestSessions;"), block.index("newSessionDefaults ="))

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
        self.assertIn("function handleAppAuthLoss()", source)
        self.assertIn("function cleanupApp()", source)
        self.assertIn("sessionsPollingEnabled = false;", source)
        self.assertIn("secondaryPollingEnabled = false;", source)
        self.assertIn("stopAllPolling();", source)
        self.assertIn("stopSessionsPolling();", source)
        self.assertIn("stopSecondaryPolling();", source)
        self.assertIn("cleanupApp();\n          renderLogin(renderApp);", source)
        self.assertIn('addAppEvent(window, "beforeunload", () => {\n                cleanupApp();\n              });', source)


if __name__ == "__main__":
    unittest.main()
