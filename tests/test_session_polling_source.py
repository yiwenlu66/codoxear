import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_API_JS = ROOT / "codoxear" / "static" / "app_api.js"


def eval_message_poll_delay_policy() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function browserOffline()")
    end = source.index("function stopSessionsPolling()", start)
    helper_source = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          document: {{ visibilityState: "visible" }},
          navigator: {{ onLine: true }},
          MESSAGE_POLL_FAST_MS: 200,
          MESSAGE_POLL_RUNNING_MS: 250,
          MESSAGE_POLL_IDLE_MS: 900,
          MESSAGE_POLL_HIDDEN_MS: 5000,
          MESSAGE_POLL_OFFLINE_MS: 15000,
          MESSAGE_POLL_ERROR_MIN_MS: 2000,
          MESSAGE_POLL_ERROR_MAX_MS: 30000,
          messagePollErrorStreak: 0,
          pollFastUntilMs: 0,
          turnOpen: false,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(helper_source + "\nglobalThis.__test = { messagePollDelayMs, normalizeMessagePollKickDelay, markMessagePollFailure, markMessagePollSuccess };\n")}, ctx);
        const idle = ctx.__test.messagePollDelayMs(1000);
        ctx.turnOpen = true;
        const running = ctx.__test.messagePollDelayMs(1000);
        ctx.turnOpen = false;
        ctx.pollFastUntilMs = 2000;
        const fast = ctx.__test.messagePollDelayMs(1000);
        ctx.pollFastUntilMs = 0;
        ctx.document.visibilityState = "hidden";
        const hidden = ctx.__test.messagePollDelayMs(1000);
        ctx.document.visibilityState = "visible";
        ctx.navigator.onLine = false;
        const offline = ctx.__test.messagePollDelayMs(1000);
        ctx.navigator.onLine = true;
        ctx.__test.markMessagePollFailure();
        const error1 = ctx.__test.messagePollDelayMs(1000);
        const errorKick0 = ctx.__test.normalizeMessagePollKickDelay(0);
        ctx.__test.markMessagePollFailure();
        const error2 = ctx.__test.messagePollDelayMs(1000);
        for (let i = 0; i < 5; i += 1) ctx.__test.markMessagePollFailure();
        ctx.navigator.onLine = false;
        const offlineHighError = ctx.__test.messagePollDelayMs(1000);
        const offlineHighErrorKick0 = ctx.__test.normalizeMessagePollKickDelay(0);
        ctx.navigator.onLine = true;
        ctx.__test.markMessagePollSuccess();
        const recovered = ctx.__test.messagePollDelayMs(1000);
        process.stdout.write(JSON.stringify({{ idle, running, fast, hidden, offline, error1, errorKick0, error2, offlineHighError, offlineHighErrorKick0, recovered }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


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
        api_source = APP_API_JS.read_text(encoding="utf-8")
        self.assertIn("const codoxearApi = window.CodoxearApi;", source)
        self.assertIn('throw new Error("Codoxear API helpers failed to load")', source)
        self.assertIn("typeof codoxearApi.clearApiCache !== \"function\"", source)
        self.assertIn("function apiResponseNotModified(obj) {", source)
        self.assertIn("return codoxearApi.apiResponseNotModified(obj);", source)
        self.assertIn("function clearApiCache() {", source)
        self.assertIn("async function api(path, options = {}) {", source)
        self.assertIn("const apiEtags = new Map();", api_source)
        self.assertIn("function clearApiCache() {", api_source)
        self.assertIn("apiEtags.clear();", api_source)
        self.assertIn('const cacheableSessionsRequest = method === "GET" && rawPath === "/api/sessions";', api_source)
        self.assertIn('opts.headers["If-None-Match"] = apiEtags.get(rawPath).etag;', api_source)
        self.assertIn("const API_NOT_MODIFIED = Symbol(\"api.notModified\");", api_source)
        self.assertIn("if (res.status === 304 && cacheableSessionsRequest && apiEtags.has(rawPath)) {", api_source)
        self.assertIn("Object.defineProperty(cached, API_NOT_MODIFIED, { value: true });", api_source)
        self.assertIn("const notModified = apiResponseNotModified(data);", source)
        self.assertIn("if (notModified && !swipeRefreshDeferred) return latestSessions;", source)
        self.assertIn('const etag = cacheableSessionsRequest ? res.headers.get("ETag") : null;', api_source)
        self.assertIn("if (etag) apiEtags.set(rawPath, { etag, text: txt });", api_source)

    def test_session_refreshes_are_serialized(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("let sessionsRefreshInFlight = null;", source)
        self.assertIn("let sessionsRefreshQueued = false;", source)
        self.assertIn("if (sessionsRefreshInFlight) {", source)
        self.assertIn("sessionsRefreshQueued = true;", source)
        self.assertIn("result = await refreshSessionsOnce();", source)
        self.assertIn("while (sessionsRefreshQueued && !appDisposed);", source)
        self.assertIn("sessionsRefreshInFlight = null;", source)

    def test_sessions_304_fast_path_precedes_sidebar_rebuild(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        block = source[source.index("async function refreshSessionsOnce()") : source.index("function appendEvent")]
        self.assertLess(block.index("if (notModified && !swipeRefreshDeferred) return latestSessions;"), block.index('sessionsWrap.innerHTML = "";'))
        self.assertLess(block.index("if (notModified && !swipeRefreshDeferred) return latestSessions;"), block.index("newSessionDefaults ="))

    def test_sessions_identical_sidebar_signature_skips_dom_rebuild(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('let lastSidebarRenderSignature = "";', source)
        self.assertIn("function sidebarRenderSignature(entries, { selectedId = \"\", swipeActions = false } = {})", source)
        block = source[source.index("async function refreshSessionsOnce()") : source.index("function appendEvent")]
        self.assertIn("const sidebarSignature = sidebarRenderSignature(sidebarEntries, { selectedId: selected, swipeActions });", block)
        self.assertIn("const sidebarUnchanged = !applyingDeferredSwipeRefresh && sessionsWrap.childElementCount > 0 && sidebarSignature === lastSidebarRenderSignature;", block)
        self.assertIn("if (!sidebarUnchanged) {", block)
        self.assertIn("lastSidebarRenderSignature = sidebarSignature;", block)
        self.assertLess(block.index("const sidebarUnchanged"), block.index('sessionsWrap.innerHTML = "";'))

    def test_sessions_304_preserves_deferred_swipe_refresh(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        block = source[source.index("async function refreshSessionsOnce()") : source.index("function appendEvent")]
        self.assertIn("if (notModified && !swipeRefreshDeferred) return latestSessions;", block)
        self.assertIn("if (!notModified) {", block)
        self.assertIn("const applyingDeferredSwipeRefresh = swipeRefreshDeferred && !openSwipeSessionId;", block)
        self.assertIn("const sessions = latestSessions", block)
        self.assertIn("if (applyingDeferredSwipeRefresh) swipeRefreshDeferred = false;", block)
        self.assertLess(block.index("const sessions = latestSessions"), block.index("if (swipeActions && openSwipeSessionId"))
        self.assertLess(block.index("const sidebarUnchanged"), block.index("if (applyingDeferredSwipeRefresh) swipeRefreshDeferred = false;"))
        self.assertLess(block.index("if (applyingDeferredSwipeRefresh) swipeRefreshDeferred = false;"), block.index('sessionsWrap.innerHTML = "";'))

    def test_closing_swipe_keeps_deferred_refresh_flag_until_render(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        close_start = source.index("function closeOpenSwipe()")
        close_end = source.index("async function doDelete", close_start)
        block = source[close_start:close_end]
        self.assertIn("if (swipeRefreshDeferred) {", block)
        self.assertIn("void refreshSessions().catch", block)
        self.assertNotIn("swipeRefreshDeferred = false;", block)

    def test_active_message_poll_delay_policy(self) -> None:
        result = eval_message_poll_delay_policy()
        self.assertEqual(result["idle"], 900)
        self.assertEqual(result["running"], 250)
        self.assertEqual(result["fast"], 200)
        self.assertEqual(result["hidden"], 5000)
        self.assertEqual(result["offline"], 15000)
        self.assertEqual(result["error1"], 2000)
        self.assertEqual(result["errorKick0"], 2000)
        self.assertEqual(result["error2"], 4000)
        self.assertEqual(result["offlineHighError"], 30000)
        self.assertEqual(result["offlineHighErrorKick0"], 30000)
        self.assertEqual(result["recovered"], 900)

    def test_active_message_polling_is_visibility_offline_error_aware(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("const MESSAGE_POLL_FAST_MS = 200;", source)
        self.assertIn("const MESSAGE_POLL_RUNNING_MS = 250;", source)
        self.assertIn("const MESSAGE_POLL_IDLE_MS = 900;", source)
        self.assertIn("const MESSAGE_POLL_HIDDEN_MS = 5000;", source)
        self.assertIn("const MESSAGE_POLL_OFFLINE_MS = 15000;", source)
        self.assertIn("const MESSAGE_POLL_ERROR_MIN_MS = 2000;", source)
        self.assertIn("let pollKickDelayMs = null;", source)
        self.assertIn("let messagePollErrorStreak = 0;", source)
        self.assertIn("function messagePollDelayMs(now = Date.now())", source)
        self.assertIn("function normalizeMessagePollKickDelay(ms = 0)", source)
        self.assertIn("const errorDelay = messagePollErrorDelayMs();", source)
        self.assertIn("if (browserOffline()) return Math.max(MESSAGE_POLL_OFFLINE_MS, errorDelay);", source)
        self.assertIn('if (document.visibilityState === "hidden") return Math.max(MESSAGE_POLL_HIDDEN_MS, errorDelay);', source)
        self.assertIn("return Math.max(delay, errorDelay);", source)
        self.assertIn("return Math.max(requested, errorDelay);", source)
        self.assertIn("markMessagePollSuccess();", source)
        self.assertIn("markMessagePollFailure();", source)
        open_start = source.index("async function openSession(sessionId")
        open_end = source.index("async function pollMessages", open_start)
        open_block = source[open_start:open_end]
        self.assertIn("markMessagePollFailure();", open_block)
        self.assertIn("renderTranscriptLoadError(sessionId, e, { preserveTranscript: displayedCachedTail });", open_block)
        self.assertIn("if (!appDisposed && selected === sessionId && pollGen === myGen) kickPoll(messagePollDelayMs());", open_block)
        self.assertIn("markMessagePollSuccess();", open_block)
        self.assertIn("pollTimer = setTimeout(pollLoop, messagePollDelayMs());", source)
        self.assertIn("const delay = pollKickDelayMs == null ? 0 : pollKickDelayMs;", source)
        self.assertIn("pollKickDelayMs = null;", source)
        kick_start = source.index("function kickPoll(ms = 0)")
        kick_end = source.index("async function jumpToLatest", kick_start)
        kick_block = source[kick_start:kick_end]
        self.assertIn("const delay = normalizeMessagePollKickDelay(ms);", kick_block)
        self.assertIn("pollKickDelayMs = delay;", kick_block)
        self.assertIn("pollTimer = setTimeout(pollLoop, delay);", kick_block)

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
        self.assertIn("if (selected) kickPoll(0);", source)
        self.assertIn("scheduleSessionsPoll(0);", source)
        self.assertIn("scheduleSecondaryPoll(0);", source)
        self.assertIn("if (selected) kickPoll(messagePollDelayMs());", source)
        self.assertIn("scheduleSessionsPoll(sessionsPollDelayMs());", source)
        self.assertIn("scheduleSecondaryPoll(secondaryPollDelayMs());", source)
        self.assertIn('addAppEvent(window, "online", () => {', source)
        self.assertIn("messagePollErrorStreak = 0;", source)
        self.assertIn('addAppEvent(window, "offline", () => {', source)

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
