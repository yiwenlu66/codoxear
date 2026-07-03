import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_API_JS = ROOT / "codoxear" / "static" / "app_api.js"
APP_POLLING_JS = ROOT / "codoxear" / "static" / "app_polling.js"


def eval_message_poll_request_abort() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    helper_start = source.index("function stopMessagePolling()")
    helper_end = source.index("function cleanupApp", helper_start)
    poll_start = source.index("async function pollMessages(")
    poll_end = source.index("async function pollLoop()", poll_start)
    snippet = (
        "let openSessionTailAbortController = null;\n"
        "let messagePollAbortController = null;\n"
        + source[helper_start:helper_end]
        + "\n"
        + source[poll_start:poll_end]
    )
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const calls = [];
        const pending = [];
        class AbortController {{
          constructor() {{
            const listeners = [];
            this.signal = {{
              aborted: false,
              addEventListener(type, cb) {{ if (type === "abort") listeners.push(cb); }},
              _listeners: listeners,
            }};
          }}
          abort() {{
            if (this.signal.aborted) return;
            this.signal.aborted = true;
            for (const cb of this.signal._listeners.slice()) cb();
          }}
        }}
        const ctx = {{
          AbortController,
          console,
          encodeURIComponent,
          appDisposed: false,
          selected: "sid-a",
          pollGen: 7,
          pollTimer: null,
          pollKickPending: false,
          pollKickDelayMs: null,
          messagePollErrorStreak: 0,
          pollFastUntilMs: 0,
          turnOpen: false,
          transcriptActive: {{ state: "bound", logPath: "/log-a.jsonl", threadId: "thread-a", liveCursor: "cursor-a" }},
          transcriptSlotRuntime: {{
            activeSnapshot: () => ({{ ...ctx.transcriptActive }}),
            setLiveCursor: (value) => {{ ctx.transcriptActive.liveCursor = value || null; calls.push(["transcriptSlotRuntime.setLiveCursor", value || null]); }},
          }},
          activeTranscriptSnapshot: () => ({{ ...ctx.transcriptActive }}),
          sessionIndex: new Map(),
          titleLabel: {{ textContent: "" }},
          toast: {{ textContent: "" }},
          clearTimeout: (...args) => calls.push(["clearTimeout", ...args]),
          initPageLimit: () => 60,
          api: (url, options = {{}}) => {{
            calls.push(["api", url, Boolean(options.signal)]);
            return new Promise((resolve, reject) => {{
              const req = {{ url, signal: options.signal || null, resolve, reject }};
              pending.push(req);
              if (options.signal && typeof options.signal.addEventListener === "function") {{
                options.signal.addEventListener("abort", () => {{
                  calls.push(["abort", url]);
                  const err = new Error("aborted");
                  err.name = "AbortError";
                  reject(err);
                }});
              }}
            }});
          }},
          handleAppAuthLoss: () => calls.push(["handleAppAuthLoss"]),
          openSession: async (...args) => calls.push(["openSession", ...args]),
          clearSelectedSessionAfterRemoval: (...args) => calls.push(["clearSelectedSessionAfterRemoval", ...args]),
          refreshSessions: async () => calls.push(["refreshSessions"]),
          markMessagePollFailure: () => calls.push(["markMessagePollFailure"]),
          markMessagePollSuccess: () => calls.push(["markMessagePollSuccess"]),
          updateSessionTranscriptSlot: () => {{ calls.push(["updateSessionTranscriptSlot"]); return {{ ignoredStaleBound: false, current: {{ state: "bound" }} }}; }},
          renderPendingTranscriptSlot: () => calls.push(["renderPendingTranscriptSlot"]),
          applySessionRuntimeFromTail: () => calls.push(["applySessionRuntimeFromTail"]),
          renderSessionTail: () => calls.push(["renderSessionTail"]),
          transcriptSnapshotFromData: () => {{ calls.push(["transcriptSnapshotFromData"]); return {{ state: "bound", logPath: "/log-a.jsonl" }}; }},
          resetChatRenderState: () => calls.push(["resetChatRenderState"]),
          setAttachCount: (...args) => calls.push(["setAttachCount", ...args]),
          appendEvent: (...args) => calls.push(["appendEvent", ...args]),
          setStatus: (...args) => calls.push(["setStatus", ...args]),
          setContext: (...args) => calls.push(["setContext", ...args]),
          setTyping: (...args) => calls.push(["setTyping", ...args]),
          appendTailSnapshotEvents: (...args) => calls.push(["appendTailSnapshotEvents", ...args]),
          sessionTitleWithId: (s) => `title:${{s.session_id}}`,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test = { pollMessages, stopMessagePolling, controllerActive: () => Boolean(messagePollAbortController) };\n")}, ctx);
        (async () => {{
          const first = ctx.__test.pollMessages("sid-a", 7);
          await Promise.resolve();
          const firstReq = pending[0];
          ctx.selected = "sid-b";
          ctx.pollGen = 8;
          ctx.transcriptActive = {{ state: "bound", logPath: "/log-a.jsonl", threadId: "thread-a", liveCursor: "cursor-b" }};
          const second = ctx.__test.pollMessages("sid-b", 8);
          await Promise.resolve();
          const secondReq = pending[1];
          await first;
          const activeAfterFirstFinally = ctx.__test.controllerActive();
          ctx.__test.stopMessagePolling();
          await second;
          const activeAfterSecondAbort = ctx.__test.controllerActive();

          ctx.selected = "sid-p";
          ctx.pollGen = 20;
          ctx.transcriptActive = {{ state: "pending_bind", logPath: null, threadId: null, liveCursor: null }};
          const third = ctx.__test.pollMessages("sid-p", 20);
          await Promise.resolve();
          const thirdReq = pending[2];
          ctx.__test.stopMessagePolling();
          await third;

          process.stdout.write(JSON.stringify({{
            apiCalls: calls.filter((call) => call[0] === "api"),
            abortCalls: calls.filter((call) => call[0] === "abort"),
            failureCalls: calls.filter((call) => call[0] === "markMessagePollFailure"),
            successCalls: calls.filter((call) => call[0] === "markMessagePollSuccess"),
            firstSignalAborted: Boolean(firstReq && firstReq.signal && firstReq.signal.aborted),
            secondSignalAborted: Boolean(secondReq && secondReq.signal && secondReq.signal.aborted),
            thirdSignalAborted: Boolean(thirdReq && thirdReq.signal && thirdReq.signal.aborted),
            activeAfterFirstFinally,
            activeAfterSecondAbort,
            activeAfterThirdAbort: ctx.__test.controllerActive(),
            toastText: ctx.toast.textContent,
          }}));
        }})().catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)



def eval_message_poll_delay_policy() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    polling_source = APP_POLLING_JS.read_text(encoding="utf-8")
    start = source.index("function browserOffline()")
    end = source.index("function stopSessionsPolling()", start)
    helper_source = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const moduleCtx = {{ window: {{}} }};
        vm.createContext(moduleCtx);
        vm.runInContext({json.dumps(polling_source)}, moduleCtx);
        const ctx = {{
          document: {{ visibilityState: "visible" }},
          navigator: {{ onLine: true }},
          codoxearPolling: moduleCtx.window.CodoxearPolling,
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
        polling_source = APP_POLLING_JS.read_text(encoding="utf-8")
        self.assertIn("SESSION_POLL_VISIBLE_MS: 2500", polling_source)
        self.assertIn("SESSION_POLL_HIDDEN_MS: 15000", polling_source)
        self.assertIn('return visibilityState === "hidden" ? hiddenMs : visibleMs;', polling_source)
        self.assertIn("return codoxearPolling.sessionsPollDelayMs(document.visibilityState);", source)
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

    def test_message_poll_requests_abort_on_supersede_and_stop(self) -> None:
        result = eval_message_poll_request_abort()
        self.assertEqual(
            result["apiCalls"],
            [
                ["api", "/api/sessions/sid-a/messages/live?cursor=cursor-a", True],
                ["api", "/api/sessions/sid-b/messages/live?cursor=cursor-b", True],
                ["api", "/api/sessions/sid-p/messages/tail?limit=60", True],
            ],
        )
        self.assertEqual(
            result["abortCalls"],
            [
                ["abort", "/api/sessions/sid-a/messages/live?cursor=cursor-a"],
                ["abort", "/api/sessions/sid-b/messages/live?cursor=cursor-b"],
                ["abort", "/api/sessions/sid-p/messages/tail?limit=60"],
            ],
        )
        self.assertTrue(result["firstSignalAborted"])
        self.assertTrue(result["secondSignalAborted"])
        self.assertTrue(result["thirdSignalAborted"])
        self.assertTrue(result["activeAfterFirstFinally"])
        self.assertFalse(result["activeAfterSecondAbort"])
        self.assertFalse(result["activeAfterThirdAbort"])
        self.assertEqual(result["failureCalls"], [])
        self.assertEqual(result["successCalls"], [])
        self.assertEqual(result["toastText"], "")

    def test_active_message_polling_is_visibility_offline_error_aware(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        polling_source = APP_POLLING_JS.read_text(encoding="utf-8")
        self.assertIn("MESSAGE_POLL_FAST_MS: 200", polling_source)
        self.assertIn("MESSAGE_POLL_RUNNING_MS: 250", polling_source)
        self.assertIn("MESSAGE_POLL_IDLE_MS: 900", polling_source)
        self.assertIn("MESSAGE_POLL_HIDDEN_MS: 5000", polling_source)
        self.assertIn("MESSAGE_POLL_OFFLINE_MS: 15000", polling_source)
        self.assertIn("MESSAGE_POLL_ERROR_MIN_MS: 2000", polling_source)
        self.assertIn("let pollKickDelayMs = null;", source)
        self.assertIn("let messagePollAbortController = null;", source)
        self.assertIn("let messagePollErrorStreak = 0;", source)
        self.assertIn("function messagePollDelayMs(now = Date.now())", source)
        self.assertIn("function normalizeMessagePollKickDelay(ms = 0)", source)
        self.assertIn("return codoxearPolling.messagePollErrorDelayMs(messagePollErrorStreak);", source)
        self.assertIn("return codoxearPolling.messagePollDelayMs({", source)
        self.assertIn("visibilityState: document.visibilityState", source)
        self.assertIn("offline: browserOffline()", source)
        self.assertIn("errorStreak: messagePollErrorStreak", source)
        self.assertIn("pollFastUntilMs,", source)
        self.assertIn("turnOpen,", source)
        self.assertIn("return codoxearPolling.normalizeMessagePollKickDelay({", source)
        self.assertIn("markMessagePollSuccess();", source)
        self.assertIn("markMessagePollFailure();", source)
        open_start = source.index("async function openSession(sessionId")
        open_end = source.index("async function pollMessages", open_start)
        open_block = source[open_start:open_end]
        self.assertIn("markMessagePollFailure();", open_block)
        self.assertIn("const tailRequest = beginOpenSessionTailRequest(sessionId, myGen);", open_block)
        self.assertIn("abortMessagePollRequest();", open_block)
        self.assertIn("signal: tailRequest.signal,", open_block)
        self.assertIn("if (isOpenSessionTailAbortError(tailRequest, e)) return null;", open_block)
        self.assertIn("renderTranscriptLoadError(sessionId, e, { preserveTranscript: displayedCachedTail });", open_block)
        self.assertIn("if (!appDisposed && selected === sessionId && pollGen === myGen) kickPoll(messagePollDelayMs());", open_block)
        self.assertIn("markMessagePollSuccess();", open_block)
        poll_start = source.index("async function pollMessages(")
        poll_end = source.index("async function pollLoop()", poll_start)
        poll_block = source[poll_start:poll_end]
        self.assertIn("function beginMessagePollRequest(sessionId, gen)", source)
        self.assertIn("pollRequest = beginMessagePollRequest(sid, gen);", poll_block)
        self.assertIn("{ signal: pollRequest.signal }", poll_block)
        self.assertIn("if (isMessagePollAbortError(pollRequest, e)) return;", poll_block)
        self.assertIn("finishMessagePollRequest(pollRequest);", poll_block)
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
        polling_source = APP_POLLING_JS.read_text(encoding="utf-8")
        self.assertIn("SECONDARY_POLL_VISIBLE_MS: 10000", polling_source)
        self.assertIn("SECONDARY_POLL_HIDDEN_MS: 60000", polling_source)
        self.assertIn("return codoxearPolling.secondaryPollDelayMs(document.visibilityState);", source)
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
        stop_message_start = source.index("function stopMessagePolling()")
        stop_message_end = source.index("function abortController(controller)", stop_message_start)
        stop_message_block = source[stop_message_start:stop_message_end]
        self.assertIn("abortOpenSessionTailRequest();", stop_message_block)
        self.assertIn("abortMessagePollRequest();", stop_message_block)
        self.assertIn("sessionsPollingEnabled = false;", source)
        self.assertIn("secondaryPollingEnabled = false;", source)
        self.assertIn("stopAllPolling();", source)
        self.assertIn("stopSessionsPolling();", source)
        self.assertIn("stopSecondaryPolling();", source)
        self.assertIn("cleanupApp();\n          renderLogin(renderApp);", source)
        self.assertIn('addAppEvent(window, "beforeunload", () => {\n                cleanupApp();\n              });', source)


if __name__ == "__main__":
    unittest.main()
