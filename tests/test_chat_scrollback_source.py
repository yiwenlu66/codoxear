import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_DISPLAY_JS = ROOT / "codoxear" / "static" / "app_display.js"
APP_LAUNCH_JS = ROOT / "codoxear" / "static" / "app_launch.js"
APP_TRANSCRIPT_JS = ROOT / "codoxear" / "static" / "app_transcript.js"
APP_MESSAGE_ROWS_JS = ROOT / "codoxear" / "static" / "app_message_rows.js"
APP_CSS = ROOT / "codoxear" / "static" / "app.css"


def eval_launch_recovery_helpers() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    launch_source = APP_LAUNCH_JS.read_text(encoding="utf-8")
    redactor_start = source.index("function redactedLaunchErrorText(value) {")
    redactor_end = source.index("function sessionLaunchLabel(s)", redactor_start)
    start = source.index("function recoverySessionInfo(sessionId) {")
    end = source.index("function focusedRecoveryActionDescriptor(sessionId)", start)
    snippet = source[redactor_start:redactor_end] + "\n" + source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const launchRow = {{
          session_id: "launch-dead",
          agent_backend: "pi",
          cwd: "/tmp/work",
          launch_state: "failed",
          launch_stage: "pty_fork",
          launch_error: "pty fork failed before agent start API_TOKEN: secret-token password: hunter2 \\\"api_key\\\":\\\"json-secret\\\" Authorization: Bearer abcdefghijklmnop",
          model_provider: "macaron",
          provider_choice: "macaron",
          preferred_auth_method: null,
          model: "gpt-5.4",
          reasoning_effort: "medium",
          service_tier: "fast",
          transport: "tmux",
          tmux_session: "codoxear",
          tmux_window: "work-abc123",
          submitted_user_message_count: 2,
        }};
        const moduleCtx = {{
          URL,
          window: {{
            CodoxearUrls: {{ resolveAppUrl: (path) => String(path) }},
            CodoxearStorage: {{ getItem: () => null, setItem: () => true, removeItem: () => true }},
          }},
        }};
        vm.createContext(moduleCtx);
        vm.runInContext({json.dumps(display_source)}, moduleCtx);
        vm.runInContext({json.dumps(launch_source)}, moduleCtx);
        const ctx = {{
          codoxearDisplay: moduleCtx.window.CodoxearDisplay,
          codoxearLaunch: moduleCtx.window.CodoxearLaunch,
          sessionIndex: new Map([["launch-dead", launchRow]]),
          selected: "launch-dead",
          sessionLaunchFailed: (s) => Boolean(s && String(s.launch_state || "").toLowerCase() === "failed"),
          setToast: (value) => {{ ctx.toast = value; }},
          confirm: () => false,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test = { recoverySessionInfo, launchPresetFromSessionInfo, recoveryDetailsText };\n")}, ctx);
        const info = ctx.__test.recoverySessionInfo("launch-dead");
        process.stdout.write(JSON.stringify({{
          hasInfo: info === launchRow,
          details: ctx.__test.recoveryDetailsText("launch-dead", launchRow),
          preset: ctx.__test.launchPresetFromSessionInfo(launchRow),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)



def eval_open_session_tail_request_abort() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    helper_start = source.index("function abortController(controller)")
    helper_end = source.index("function cleanupApp", helper_start)
    open_start = source.index("async function openSession(")
    open_end = source.index("async function pollMessages(", open_start)
    snippet = "let openSessionTailAbortController = null;\nlet messagePollAbortController = null;\n" + source[helper_start:helper_end] + "\n" + source[open_start:open_end]
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
          performance: {{ now: () => 123 }},
          clearTimeout: () => calls.push(["clearTimeout"]),
          pollGen: 0,
          pollTimer: null,
          pollKickPending: false,
          pollKickDelayMs: null,
          selected: null,
          unattendedMenuOpen: false,
          unattendedMenuSessionId: "",
          activeTranscriptState: "pending_bind",
          activeLogPath: null,
          activeThreadId: null,
          liveCursor: null,
          turnOpen: false,
          clickLoadT0: 0,
          clickMetricPending: false,
          fileDirty: false,
          currentFileDirty: () => false,
          appDisposed: false,
          sessionIndex: new Map([
            ["sid-a", {{ session_id: "sid-a", busy: false, queue_len: 0, token: null }}],
            ["sid-b", {{ session_id: "sid-b", busy: false, queue_len: 0, token: null }}],
          ]),
          sessionTailCache: new Map(),
          transcriptSlotRuntime: {{
            setActivePending: () => calls.push(["transcriptSlotRuntime.setActivePending"]),
            setActiveFailed: () => calls.push(["transcriptSlotRuntime.setActiveFailed"]),
            getTailCache: () => null,
          }},
          titleLabel: {{ textContent: "" }},
          hideUnattendedMenu: () => calls.push(["hideUnattendedMenu"]),
          storageSetItem: (...args) => calls.push(["storageSetItem", ...args]),
          storageRemoveItem: (...args) => calls.push(["storageRemoveItem", ...args]),
          setSessionHash: (...args) => calls.push(["setSessionHash", ...args]),
          clearRenderedTranscriptRange: () => calls.push(["clearRenderedTranscriptRange"]),
          setAttachCount: (...args) => calls.push(["setAttachCount", ...args]),
          updateQueueBadge: () => calls.push(["updateQueueBadge"]),
          setStatus: (...args) => calls.push(["setStatus", ...args]),
          setContext: (...args) => calls.push(["setContext", ...args]),
          setTyping: (...args) => calls.push(["setTyping", ...args]),
          resetChatRenderState: () => calls.push(["resetChatRenderState"]),
          sessionTitleWithId: (s) => `title:${{s.session_id}}`,
          isFileViewerOpen: () => false,
          ensureCurrentFileViewerSession: async () => calls.push(["ensureCurrentFileViewerSession"]),
          renderPendingTranscriptSlot: (...args) => calls.push(["renderPendingTranscriptSlot", ...args]),
          syncAttachButtonState: () => calls.push(["syncAttachButtonState"]),
          syncQueueSubmitState: () => calls.push(["syncQueueSubmitState"]),
          syncSendButtonState: () => calls.push(["syncSendButtonState"]),
          updateUnattendedBtnState: () => calls.push(["updateUnattendedBtnState"]),
          isMobile: () => false,
          setSidebarOpen: (...args) => calls.push(["setSidebarOpen", ...args]),
          tailCacheMatchesSession: () => false,
          applyCachedTail: (...args) => calls.push(["applyCachedTail", ...args]),
          renderTranscriptLoading: (...args) => calls.push(["renderTranscriptLoading", ...args]),
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
          markMessagePollFailure: () => calls.push(["markMessagePollFailure"]),
          handleFileViewerSessionUnavailable: (...args) => calls.push(["handleFileViewerSessionUnavailable", ...args]),
          refreshSessions: async () => calls.push(["refreshSessions"]),
          renderTranscriptLoadError: (...args) => calls.push(["renderTranscriptLoadError", ...args]),
          kickPoll: (...args) => calls.push(["kickPoll", ...args]),
          messagePollDelayMs: () => 900,
          markMessagePollSuccess: () => calls.push(["markMessagePollSuccess"]),
          updateSessionTranscriptSlot: (sid, data) => {{ calls.push(["updateSessionTranscriptSlot", sid]); return {{ ignoredStaleBound: false, current: {{ state: "bound" }} }}; }},
          applySessionRuntimeFromTail: (...args) => calls.push(["applySessionRuntimeFromTail", ...args]),
          renderSessionTail: (...args) => calls.push(["renderSessionTail", ...args]),
          refreshFileCandidates: async (...args) => calls.push(["refreshFileCandidates", ...args]),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test = { openSession, abortOpenSessionTailRequest };\n")}, ctx);
        (async () => {{
          const firstPromise = ctx.__test.openSession("sid-a", {{ useCache: false }});
          await Promise.resolve();
          const firstSignal = pending[0] && pending[0].signal;
          const secondPromise = ctx.__test.openSession("sid-b", {{ useCache: false }});
          await Promise.resolve();
          const secondReq = pending[1];
          secondReq.resolve({{ transcript_state: "bound", events: [{{ role: "assistant", text: "ok" }}], busy: false, queue_len: 0, token: null }});
          const secondResult = await secondPromise;
          const firstResult = await firstPromise;
          process.stdout.write(JSON.stringify({{
            firstResult,
            secondResult,
            firstSignalAborted: Boolean(firstSignal && firstSignal.aborted),
            secondSignalAborted: Boolean(secondReq.signal && secondReq.signal.aborted),
            pollGen: ctx.pollGen,
            selected: ctx.selected,
            title: ctx.titleLabel.textContent,
            apiCalls: calls.filter((call) => call[0] === "api"),
            abortCalls: calls.filter((call) => call[0] === "abort"),
            failureCalls: calls.filter((call) => call[0] === "markMessagePollFailure"),
            loadErrorCalls: calls.filter((call) => call[0] === "renderTranscriptLoadError"),
            successCalls: calls.filter((call) => call[0] === "markMessagePollSuccess"),
            renderTailCalls: calls.filter((call) => call[0] === "renderSessionTail"),
          }}));
        }})().catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)



def eval_clear_deleted_session_client_state() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function clearDeletedSessionClientState(")
    end = source.index("async function dismissFailedLaunchRecord", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const calls = [];
        const ctx = {{
          clearSelectedSessionAfterRemoval: (sid) => {{ calls.push(["clearSelectedSessionAfterRemoval", sid]); return sid === "selected"; }},
          transcriptSlotRuntime: {{ deleteSession: (sid) => calls.push(["transcriptSlotRuntime.deleteSession", sid]) }},
          dropPendingUserRows: (sid, predicate) => calls.push(["dropPendingUserRows", sid, predicate({{}})]),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_clear_deleted = clearDeletedSessionClientState;\n")}, ctx);
        const selectedResult = ctx.__test_clear_deleted("selected");
        const selectedCalls = calls.slice();
        calls.length = 0;
        const otherResult = ctx.__test_clear_deleted("other");
        const otherCalls = calls.slice();
        process.stdout.write(JSON.stringify({{ selectedResult, selectedCalls, otherResult, otherCalls }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)



def eval_clear_selected_session_after_removal() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function clearSelectedSessionAfterRemoval(")
    end = source.index("async function dismissFailedLaunchRecord", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const calls = [];
        const ctx = {{
          selected: "sid-1",
          pollGen: 7,
          pollTimer: 123,
          pollKickPending: true,
          pollKickDelayMs: 50,
          unattendedMenuOpen: true,
          activeTranscriptState: "bound",
          activeLogPath: "/old.jsonl",
          activeThreadId: "old-thread",
          liveCursor: "cursor",
          transcriptSlotRuntime: {{
            setActivePending: () => {{
              calls.push(["transcriptSlotRuntime.setActivePending"]);
              ctx.activeTranscriptState = "pending_bind";
              ctx.activeLogPath = null;
              ctx.activeThreadId = null;
              ctx.liveCursor = null;
            }},
            deleteSession: (sid) => calls.push(["transcriptSlotRuntime.deleteSession", sid]),
          }},
          turnOpen: true,
          titleLabel: {{ textContent: "old title" }},
          handleFileViewerSessionUnavailable: (sid) => calls.push(["handleFileViewerSessionUnavailable", sid, ctx.selected]),
          abortMessagePollRequest: () => calls.push(["abortMessagePollRequest"]),
          clearTimeout: (...args) => calls.push(["clearTimeout", ...args]),
          clearRenderedTranscriptRange: () => calls.push(["clearRenderedTranscriptRange"]),
          storageRemoveItem: (...args) => calls.push(["storageRemoveItem", ...args]),
          setSessionHash: (...args) => calls.push(["setSessionHash", ...args]),
          setStatus: (...args) => calls.push(["setStatus", ...args]),
          setContext: (...args) => calls.push(["setContext", ...args]),
          setTyping: (...args) => calls.push(["setTyping", ...args]),
          setAttachCount: (...args) => calls.push(["setAttachCount", ...args]),
          resetChatRenderState: () => calls.push(["resetChatRenderState"]),
          updateQueueBadge: () => calls.push(["updateQueueBadge"]),
          hideUnattendedMenu: () => calls.push(["hideUnattendedMenu"]),
          updateUnattendedBtnState: () => calls.push(["updateUnattendedBtnState"]),
          syncSendButtonState: () => calls.push(["syncSendButtonState"]),
          syncQueueSubmitState: () => calls.push(["syncQueueSubmitState"]),
          syncAttachButtonState: () => calls.push(["syncAttachButtonState"]),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_clear = clearSelectedSessionAfterRemoval;\n")}, ctx);
        const noop = ctx.__test_clear("other", {{ incrementPollGen: true, clearPollState: true }});
        const noopState = {{ selected: ctx.selected, pollGen: ctx.pollGen, calls: calls.slice() }};
        calls.length = 0;
        const applied = ctx.__test_clear("sid-1", {{ incrementPollGen: true, clearPollState: true }});
        const appliedState = {{
          selected: ctx.selected,
          pollGen: ctx.pollGen,
          pollTimer: ctx.pollTimer,
          pollKickPending: ctx.pollKickPending,
          pollKickDelayMs: ctx.pollKickDelayMs,
          activeTranscriptState: ctx.activeTranscriptState,
          activeLogPath: ctx.activeLogPath,
          activeThreadId: ctx.activeThreadId,
          liveCursor: ctx.liveCursor,
          turnOpen: ctx.turnOpen,
          title: ctx.titleLabel.textContent,
          calls: calls.slice(),
        }};
        process.stdout.write(JSON.stringify({{ noop, noopState, applied, appliedState }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestChatScrollbackSource(unittest.TestCase):
    def test_clear_deleted_session_client_state_clears_explicit_delete_state(self) -> None:
        result = eval_clear_deleted_session_client_state()
        self.assertTrue(result["selectedResult"])
        self.assertEqual(
            result["selectedCalls"],
            [
                ["clearSelectedSessionAfterRemoval", "selected"],
                ["transcriptSlotRuntime.deleteSession", "selected"],
                ["dropPendingUserRows", "selected", True],
            ],
        )
        self.assertFalse(result["otherResult"])
        self.assertEqual(
            result["otherCalls"],
            [
                ["clearSelectedSessionAfterRemoval", "other"],
                ["transcriptSlotRuntime.deleteSession", "other"],
                ["dropPendingUserRows", "other", True],
            ],
        )

    def test_clear_selected_session_after_removal_resets_missing_session_state(self) -> None:
        result = eval_clear_selected_session_after_removal()
        self.assertFalse(result["noop"])
        self.assertEqual(result["noopState"], {"selected": "sid-1", "pollGen": 7, "calls": []})
        self.assertTrue(result["applied"])
        state = result["appliedState"]
        self.assertIsNone(state["selected"])
        self.assertEqual(state["pollGen"], 8)
        self.assertIsNone(state["pollTimer"])
        self.assertFalse(state["pollKickPending"])
        self.assertIsNone(state["pollKickDelayMs"])
        self.assertEqual(state["activeTranscriptState"], "pending_bind")
        self.assertIsNone(state["activeLogPath"])
        self.assertIsNone(state["activeThreadId"])
        self.assertIsNone(state["liveCursor"])
        self.assertFalse(state["turnOpen"])
        self.assertEqual(state["title"], "No session selected")
        self.assertEqual(state["calls"][0], ["handleFileViewerSessionUnavailable", "sid-1", "sid-1"])
        self.assertIn(["abortMessagePollRequest"], state["calls"])
        for expected in [
            ["clearTimeout", 123],
            ["storageRemoveItem", "codexweb.selected"],
            ["setSessionHash", ""],
            ["setStatus", {"running": False, "queueLen": 0}],
            ["setContext", None],
            ["setTyping", False],
            ["setAttachCount", 0],
            ["resetChatRenderState"],
            ["updateQueueBadge"],
            ["hideUnattendedMenu"],
            ["updateUnattendedBtnState"],
            ["syncSendButtonState"],
            ["syncQueueSubmitState"],
            ["syncAttachButtonState"],
        ]:
            self.assertIn(expected, state["calls"])

    def test_jump_button_reloads_selected_tail(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function jumpToLatest() {")
        end = source.index("async function selectSession(id) {", start)
        block = source[start:end]
        self.assertIn("invalidateOlderLoad();", block)
        self.assertIn("await openSession(sid, { useCache: false, fallbackToCacheOnFailure: true });", block)
        self.assertIn("kickPoll(0);", block)

    def test_open_session_tail_request_aborts_superseded_open(self) -> None:
        result = eval_open_session_tail_request_abort()
        self.assertIsNone(result["firstResult"])
        self.assertEqual(result["secondResult"]["events"], [{"role": "assistant", "text": "ok"}])
        self.assertTrue(result["firstSignalAborted"])
        self.assertFalse(result["secondSignalAborted"])
        self.assertEqual(result["pollGen"], 2)
        self.assertEqual(result["selected"], "sid-b")
        self.assertEqual(result["title"], "title:sid-b")
        self.assertEqual(len(result["apiCalls"]), 2)
        self.assertTrue(all(call[2] for call in result["apiCalls"]))
        self.assertEqual(len(result["abortCalls"]), 1)
        self.assertEqual(result["failureCalls"], [])
        self.assertEqual(result["loadErrorCalls"], [])
        self.assertEqual(result["successCalls"], [["markMessagePollSuccess"]])
        self.assertEqual(len(result["renderTailCalls"]), 1)

    def test_open_session_is_single_render_path(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function openSession(")
        end = source.index("async function pollMessages(", start)
        block = source[start:end]
        self.assertIn("transcriptSlotRuntime.setActivePending();", block)
        self.assertIn("const optimisticBusy = Boolean(s && s.busy);", block)
        self.assertIn("setStatus({ running: optimisticBusy, queueLen: optimisticQueueLen });", block)
        self.assertIn("setTyping(optimisticBusy);", block)
        self.assertIn("const cachedTail = s ? transcriptSlotRuntime.getTailCache(sessionId) : null;", block)
        self.assertIn("let displayedCachedTail = false;", block)
        self.assertIn("tailCacheMatchesSession(cachedTail, s)", block)
        self.assertIn("applyCachedTail(sessionId, cachedTail, s);", block)
        self.assertIn("displayedCachedTail = true;", block)
        self.assertIn("if (!displayedCachedTail) renderTranscriptLoading(sessionId);", block)
        self.assertIn("const tailRequest = beginOpenSessionTailRequest(sessionId, myGen);", block)
        self.assertIn("data = await api(`/api/sessions/${sessionId}/messages/tail?limit=${initPageLimit()}`, {", block)
        self.assertIn("signal: tailRequest.signal,", block)
        self.assertIn("async function openSession(sessionId, { useCache = true, fallbackToCacheOnFailure = false } = {})", block)
        self.assertIn("if (fallbackToCacheOnFailure && !displayedCachedTail && !useCache && s && cachedTail && tailCacheMatchesSession(cachedTail, s) && Array.isArray(cachedTail.events) && cachedTail.events.length) {", block)
        self.assertIn("applyCachedTail(sessionId, cachedTail, s);", block)
        self.assertIn("displayedCachedTail = true;", block)
        self.assertIn("renderTranscriptLoadError(sessionId, e, { preserveTranscript: displayedCachedTail });", block)
        self.assertIn("if (e && e.status === 401) {", block)
        self.assertIn("handleAppAuthLoss();", block)
        self.assertIn("if (isOpenSessionTailAbortError(tailRequest, e)) return null;", block)
        self.assertIn("if (!isCurrentOpenSessionTailRequest(tailRequest)) return null;", block)
        self.assertIn("finally {\n            finishOpenSessionTailRequest(tailRequest);\n          }", block)
        self.assertIn("const slotChange = updateSessionTranscriptSlot(sessionId, data);", block)
        self.assertIn('if (slotChange.current.state === "bound" || slotChange.current.state === "failed") renderSessionTail(Array.isArray(data.events) ? data.events : []);', block)
        self.assertIn("else renderPendingTranscriptSlot(sessionId);", block)
        self.assertIn("applySessionRuntimeFromTail(sessionId, data);", block)
        self.assertIn('if (slotChange.current.state !== "failed") kickPoll(900);', block)
        self.assertNotIn("refreshInitPageState", block)

    def test_transcript_loading_row_is_non_transcript_feedback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css = APP_CSS.read_text(encoding="utf-8")
        start = source.index("function renderTranscriptLoading(sessionId)")
        end = source.index("function renderTranscriptLoadError", start)
        block = source[start:end]
        self.assertIn('class: "msg-row assistant typing-row transcript-loading-row"', block)
        self.assertIn('role: "status", "aria-live": "polite", text: "Loading transcript…"', block)
        self.assertIn("chatInner.insertBefore(row, bottomSentinel);", block)
        self.assertIn(".msg.loading", css)
        self.assertIn("color: var(--muted);", css)

    def test_tail_cache_identity_uses_authoritative_tail_payload(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        self.assertIn("function transcriptIdentityFromData(data, fallback = null)", transcript_source)
        self.assertIn("const dataThreadId = typeof data?.thread_id", transcript_source)
        self.assertIn("const dataLogPath = typeof data?.log_path", transcript_source)
        self.assertIn("const identity = transcriptIdentityFromData(data, session);", transcript_source)
        self.assertIn("const identity = transcriptIdentityFromData(identityData, meta || current || null);", transcript_source)
        self.assertIn("threadId: identity.threadId", transcript_source)
        self.assertIn("logPath: identity.logPath", transcript_source)
        remember_start = source.index("function rememberTailSnapshot(sessionId, session, data)")
        remember_end = source.index("function appendTailSnapshotEvents", remember_start)
        remember_block = source[remember_start:remember_end]
        self.assertIn("return transcriptSlotRuntime.rememberTail(sessionId, session, data);", remember_block)
        append_start = source.index("function appendTailSnapshotEvents(sessionId, events")
        append_end = source.index("function restorePendingUserRowsForSession", append_start)
        append_block = source[append_start:append_end]
        self.assertIn("identityData = null", append_block)
        self.assertIn("return transcriptSlotRuntime.appendTailEvents(sessionId, events", append_block)
        poll_start = source.index("appendTailSnapshotEvents(sid, evs")
        poll_end = source.index("});", poll_start)
        poll_block = source[poll_start:poll_end]
        self.assertIn("identityData: data", poll_block)

    def test_transcript_load_error_row_is_non_transcript_feedback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css = APP_CSS.read_text(encoding="utf-8")
        start = source.index("function renderTranscriptLoadError(sessionId, err, { preserveTranscript = false } = {})")
        end = source.index("function applyCachedTail", start)
        block = source[start:end]
        self.assertIn('chatInner.querySelectorAll(".transcript-error-row")', block)
        self.assertIn("if (!preserveTranscript) {", block)
        self.assertIn("clearTranscriptDom();", block)
        self.assertIn("setOlderState({ hasMore: false, isLoading: false });", block)
        self.assertIn('class: "msg-row assistant typing-row transcript-error-row"', block)
        self.assertIn('role: "alert"', block)
        self.assertIn("Could not load transcript.", block)
        self.assertIn('class: "icon-btn text-btn transcriptRetryBtn"', block)
        self.assertIn('text: "Retry"', block)
        self.assertIn("if (selected !== sessionId) return;", block)
        self.assertIn("void openSession(sessionId, { useCache: true });", block)
        self.assertIn("setTyping(false);", block)
        self.assertIn("markClickFirstPaint();", block)
        self.assertIn(".msg.transcript-error", css)
        self.assertIn(".transcriptRetryBtn.icon-btn.text-btn", css)

    def test_tail_poll_auth_loss_precedes_stale_generation_guards(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        open_start = source.index("async function openSession(")
        open_end = source.index("async function pollMessages(", open_start)
        open_block = source[open_start:open_end]
        open_catch = open_block[open_block.index("} catch (e) {") : open_block.index("renderTranscriptLoadError(sessionId, e", open_block.index("} catch (e) {"))]
        self.assertLess(open_catch.index("if (e && e.status === 401)"), open_catch.index("if (!isCurrentOpenSessionTailRequest(tailRequest)) return null;"))
        poll_start = source.index("async function pollMessages(")
        poll_end = source.index("async function pollLoop()", poll_start)
        poll_block = source[poll_start:poll_end]
        poll_catch = poll_block[poll_block.rindex("} catch (e) {") :]
        self.assertLess(poll_catch.index("if (e && e.status === 401)"), poll_catch.index("if (isMessagePollAbortError(pollRequest, e)) return;"))
        self.assertLess(poll_catch.index("if (isMessagePollAbortError(pollRequest, e)) return;"), poll_catch.index("if (gen !== pollGen || sid !== selected) return;"))
        self.assertIn("clearSelectedSessionAfterRemoval(sid, { incrementPollGen: true, clearPollState: true });", poll_catch)

    def test_refresh_sessions_does_not_fetch_messages(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function refreshSessions() {")
        end = source.index("function appendEvent(ev) {", start)
        block = source[start:end]
        self.assertNotIn("/messages/tail", block)
        self.assertNotIn("/messages/live", block)
        self.assertNotIn("/messages/history", block)
        self.assertNotIn("await openSession(", block)
        self.assertIn("if (selected && !sessionIndex.has(selected)) clearSelectedSessionAfterRemoval(selected);", block)
        self.assertIn("function clearSelectedSessionAfterRemoval", source)
        self.assertIn("storageRemoveItem(\"codexweb.selected\");", source)
        self.assertIn("titleLabel.textContent = \"No session selected\";", source)
        self.assertIn("applySessionListTranscriptIdentity(selected, sessionIndex.get(selected));", block)

    def test_session_list_pending_bind_clears_active_transcript_slot(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function applySessionListTranscriptIdentity(")
        end = source.index("function updateQueueBadge()", start)
        block = source[start:end]
        self.assertIn("const slotChange = updateSessionTranscriptSlot(sessionId, sessionMeta);", block)
        self.assertIn("if (!slotChange.resetPending) return;", block)
        self.assertIn("transcriptSlotRuntime.deleteTailCache(sessionId);", block)
        self.assertIn("transcriptSlotRuntime.clearLiveCursor();", block)
        self.assertIn("clearRenderedTranscriptRange();", block)
        self.assertIn('if (slotChange.current.state === "pending_bind") {', block)
        self.assertIn("renderPendingTranscriptSlot(sessionId);", block)
        self.assertIn("kickPoll(0);", block)

    def test_load_older_messages_uses_oldest_rendered_row_cursor(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function loadOlderMessages({ auto = false, cancelOnScroll = true } = {}) {")
        end = source.index("function maybeAutoLoadOlder()", start)
        block = source[start:end]
        self.assertIn("const reqCursor = oldestRenderedHistoryCursor();", block)
        self.assertIn("if (!reqCursor) throw new Error(\"history cursor missing\");", block)
        self.assertIn("`/api/sessions/${sid}/messages/history?cursor=${encodeURIComponent(reqCursor)}&limit=${olderPageLimit()}`", block)
        self.assertNotIn("historyCursor", block)
        self.assertIn("await openSession(sid, { useCache: false });", block)

    def test_load_older_failure_has_inline_retry_without_resetting_history(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn('const olderErrorText = el("span", { class: "olderErrorText", text: "" });', source)
        self.assertIn('const olderRetryBtn = el("button", { class: "olderRetryBtn", type: "button", text: "Retry" });', source)
        self.assertIn('const olderError = el("div", { class: "olderError", id: "olderError", role: "status" }', source)
        self.assertIn("olderWrap.appendChild(olderError);", source)
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        self.assertIn("function clearOlderLoadError() {", source)
        self.assertIn("function showOlderLoadError() {", source)
        self.assertIn("olderLoadRuntime.clearError();", source)
        self.assertIn("olderLoadRuntime.showError();", source)
        self.assertIn('errorText.textContent = String(message || "Couldn’t load older messages.");', transcript_source)
        state_start = transcript_source.index("function setState({ hasMore: nextHasMore, isLoading: nextLoading } = {})")
        state_end = transcript_source.index("function resetAutoTrigger()", state_start)
        state_block = transcript_source[state_start:state_end]
        self.assertIn("if (isLoading || !hasMore) clearError();", state_block)
        load_start = source.index("async function loadOlderMessages({ auto = false, cancelOnScroll = true } = {}) {")
        load_end = source.index("function maybeAutoLoadOlder()", load_start)
        load_block = source[load_start:load_end]
        self.assertIn("clearOlderLoadError();\n            setOlderState({ hasMore: nextHasOlder, isLoading: false });", load_block)
        self.assertIn("handleAppAuthLoss();", load_block)
        catch_block = load_block[load_block.index("} catch (e) {") :]
        self.assertLess(catch_block.index("if (e && e.status === 401)"), catch_block.index("if (selected !== sid || pollGen !== gen || !olderLoadRuntime.isCurrent(load)) return false;"))
        self.assertLess(load_block.index("if (e && e.status === 401)"), load_block.index("if (e && e.status === 409)"))
        self.assertLess(load_block.index("if (e && e.status === 409)"), load_block.index("showOlderLoadError();"))
        self.assertIn("await openSession(sid, { useCache: false });", load_block)
        self.assertIn("setOlderState({ hasMore: hasOlderMessages(), isLoading: false });\n            showOlderLoadError();", load_block)
        self.assertNotIn("clearTranscriptDom();", load_block)
        self.assertIn("olderRetryBtn.onclick = () => {", source)
        self.assertIn("clearOlderLoadError();\n          void loadOlderMessages({ auto: false });", source)
        self.assertIn(".olderError", css)
        self.assertIn(".olderRetryBtn", css)

    def test_live_append_does_not_splice_into_history_window(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function appendEvent(ev) {")
        end = source.index("function renderTranscript(", start)
        block = source[start:end]
        self.assertIn("const stick = pending || transcriptScrollRuntime.shouldStickToBottom();", block)
        self.assertIn("if (!pending && !transcriptScrollRuntime.snapshot().renderedAtLiveTail) {", block)
        self.assertIn("markEventSeen(ev);", block)
        self.assertIn("return;", block)
        self.assertIn("trimRenderedRows({ fromTop: stick });", block)
        self.assertNotIn("trimRenderedRows({ fromTop: true });", block)

    def test_history_request_cursor_is_derived_from_rendered_rows(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        row_source = APP_MESSAGE_ROWS_JS.read_text(encoding="utf-8")
        start = source.index("function oldestRenderedHistoryCursor() {")
        end = source.index("function clearRenderedTranscriptRange()", start)
        block = source[start:end]
        self.assertIn("return codoxearMessageRows.oldestRenderedHistoryCursor(renderedMessageRows());", block)
        self.assertIn("function oldestRenderedHistoryCursor(rows)", row_source)
        self.assertIn("row.dataset.historyCursor", row_source)
        self.assertIn("return cursor;", row_source)

    def test_history_prepend_does_not_trim_newly_fetched_older_rows(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function prependOlderEvents(allEvents")
        end = source.index("async function loadOlderMessages", start)
        block = source[start:end]
        self.assertIn("chatInner.insertBefore(frag, anchor);", block)
        self.assertIn("trimRenderedRows({ fromTop: false, maxRows: CHAT_DOM_WINDOW_WITH_HISTORY_SLACK });", block)
        self.assertNotIn("trimRenderedRowsBeforeViewport({ maxRows: CHAT_DOM_WINDOW_WITH_HISTORY_SLACK });", block)

    def test_rendered_rows_keep_server_history_cursor(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        row_source = APP_MESSAGE_ROWS_JS.read_text(encoding="utf-8")
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        self.assertIn('if (typeof ev.history_cursor === "string" && ev.history_cursor) out.history_cursor = ev.history_cursor;', transcript_source)
        self.assertIn('row.dataset.historyCursor = ev.history_cursor;', row_source)
        self.assertIn('return codoxearMessageRows.makeRow(ev, { ts, pending }, messageRowDeps());', source)
        self.assertNotIn("let historyCursor", source)

    def test_poll_messages_uses_live_cursor_only(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function pollMessages(")
        end = source.index("async function pollLoop()", start)
        block = source[start:end]
        self.assertIn("if (!transcriptSlotRuntime.activeSnapshot().liveCursor) {", block)
        self.assertIn('if (activeTranscriptSnapshot().state === "pending_bind") {', block)
        self.assertIn("const slotChange = updateSessionTranscriptSlot(sid, data);", block)
        self.assertIn('if (slotChange.current.state === "bound" || slotChange.current.state === "failed") renderSessionTail(Array.isArray(data.events) ? data.events : []);', block)
        self.assertIn('if (activeTranscriptSnapshot().state === "failed") return;', block)
        self.assertIn("await openSession(sid, { useCache: false });", block)
        self.assertIn("await api(`/api/sessions/${sid}/messages/live?cursor=${encodeURIComponent(reqCursor)}`, { signal: pollRequest.signal });", block)
        self.assertIn("const slotInfo = transcriptSnapshotFromData(data);", block)
        self.assertIn("transcriptSlotRuntime.setLiveCursor(typeof data.live_cursor === \"string\" && data.live_cursor ? data.live_cursor : null);", block)
        self.assertNotIn("after_byte", block)
        self.assertNotIn("before_byte", block)

    def test_no_transcript_localstorage_cache(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertNotIn("codexweb.cache.v7", source)
        self.assertNotIn("cacheStorageKey(", source)
        self.assertNotIn("setCacheMeta(", source)
        self.assertNotIn("replaceCacheEvents(", source)
        self.assertNotIn("appendCacheEvents(", source)

    def test_send_text_scopes_optimistic_echo_to_transcript_epoch(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function sendText(")
        end = source.index("form.onsubmit = async", start)
        block = source[start:end]
        self.assertIn("const slot = getSessionTranscriptSlot(sessionId);", block)
        self.assertIn("transcriptEventRuntime.addPendingUser({ id: localId, sessionId, epoch: slot.epoch, text: raw, t0 });", block)
        self.assertNotIn("pendingUser.push", block)
        self.assertIn("appendEvent({ role: \"user\", text: raw, pending: true, localId, ts: t0 });", block)
        self.assertIn("void refreshSessions().catch((e) => {", block)
        self.assertIn("if (e && e.status === 401) handleAppAuthLoss();", block)
        self.assertIn("else console.error(\"refreshSessions failed\", e);", block)
        self.assertIn("return true;", block)
        self.assertIn("return false;", block)

    def test_submit_clears_composer_only_after_send_success(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        send_start = source.index("async function sendText(")
        start = source.index("form.onsubmit = async", send_start)
        end = source.index("(async () =>", start)
        block = source[start:end]
        self.assertNotIn("clearComposer();\n          await sendText(raw);", block)
        self.assertIn("const ok = await sendText(raw);", block)
        self.assertIn('if (ok && $("#msg").value === raw) clearComposer();', block)

    def test_restore_pending_rows_is_bound_to_current_transcript_slot(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function restorePendingUserRowsForSession(sessionId) {")
        end = source.index("function updateQueueBadge()", start)
        block = source[start:end]
        self.assertIn("const slot = getSessionTranscriptSlot(sessionId);", block)
        self.assertIn("transcriptEventRuntime.pendingUsersForSession(sessionId, Number(slot.epoch || 0));", block)
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        self.assertIn("Number(item.epoch || 0) === slotEpoch", transcript_source)

    def test_render_transcript_rebuilds_authoritative_events_after_pending_match(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function normalizedTranscriptEvents(events, { consumePending = false } = {}) {")
        end = source.index("function renderTranscript(events, { preserveScroll = false } = {}) {", start)
        helper_block = source[start:end]
        render_start = source.index("function renderTranscript(events, { preserveScroll = false } = {}) {")
        render_end = source.index("function prependOlderEvents(", render_start)
        render_block = source[render_start:render_end]
        self.assertIn("if (consumePending) takePendingUserMatch(ev, selected, { allowUntimedCommit: false });", helper_block)
        self.assertIn("msgs.push(ev);", helper_block)
        self.assertIn("const msgs = normalizedTranscriptEvents(events, { consumePending: true });", render_block)
        self.assertIn("const msgs = normalizedTranscriptEvents(events, { consumePending: false });", render_block)
        self.assertNotIn("if (consumePendingUserIfMatches(ev)) continue;", helper_block)

    def test_pending_commit_reconciliation_does_not_require_text_equality(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        start = source.index("function takePendingUserMatch(")
        end = source.index("function consumePendingUserIfMatches(", start)
        block = source[start:end]
        self.assertIn("return transcriptEventRuntime.takePendingUserMatch(ev, sessionId, Number(slot.epoch || 0), { allowUntimedCommit });", block)
        self.assertIn("const sameSlot = [];", transcript_source)
        self.assertIn("const exactCandidates = [];", transcript_source)
        self.assertIn("sameSlot.push(candidate);", transcript_source)
        self.assertIn("exactCandidates.length", transcript_source)
        self.assertIn("evTs >= Number(x.t0 || 0) - 5", transcript_source)
        self.assertIn("allowUntimedCommit", transcript_source)

    def test_error_and_warning_message_classes_are_rendered(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        row_source = APP_MESSAGE_ROWS_JS.read_text(encoding="utf-8")
        self.assertIn('messageClass === "error" || messageClass === "warning"', row_source)
        self.assertIn("bubble.classList.add(messageClass);", row_source)
        self.assertIn('return codoxearMessageRows.makeRow(ev, { ts, pending }, messageRowDeps());', source)

    def test_recovery_state_renders_in_chat_pane(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        launch_source = APP_LAUNCH_JS.read_text(encoding="utf-8")
        row_source = APP_MESSAGE_ROWS_JS.read_text(encoding="utf-8")
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn("function renderRecoveryPanelIfNeeded(sessionId)", source)
        self.assertIn(".recovery-panel-row", source)
        self.assertIn('const panelLabel = launchFailed ? "Launch failed" : "Recovery needed";', source)
        self.assertIn('text: panelLabel', source)
        self.assertIn('role: "group", "aria-label": panelLabel', source)
        self.assertIn('text: "Review queue"', source)
        self.assertIn('showQueueViewer({ opener: e.currentTarget });', source)
        self.assertIn('text: "Clear unknown marker"', source)
        self.assertIn('await clearCommitUnknownSend(sessionId, s.commit_unknown_send_text || "");', source)
        self.assertIn('text: "Copy details"', source)
        self.assertIn('await copyToClipboard(recoveryDetailsText(sessionId, s));', source)
        self.assertIn('text: "New like this"', source)
        self.assertIn('openNewSessionDialog({ likeSession: preset, statusText: "Review copied launch settings before starting.", returnFocusEl: e.currentTarget });', source)
        self.assertIn('text: "Dismiss launch"', source)
        self.assertIn('await dismissFailedLaunchRecord(sessionId);', source)
        self.assertIn('function launchPresetFromSessionInfo(s)', source)
        self.assertIn('function dismissFailedLaunchRecord(sessionId)', source)
        self.assertIn('await api(`/api/sessions/${sessionId}/delete`, { method: "POST", body: {} });', source)
        self.assertIn('function clearSelectedSessionAfterRemoval(sessionId, { incrementPollGen = false, clearPollState = false } = {})', source)
        self.assertIn('function clearDeletedSessionClientState(sessionId)', source)
        self.assertIn('clearDeletedSessionClientState(s.session_id);', source)
        self.assertIn('clearDeletedSessionClientState(sessionId);', source)
        self.assertIn('if (!s || (!sessionLaunchFailed(s) && !s.orphan_recovery && !s.queue_recovery && !s.commit_unknown_send)) return null;', source)
        self.assertIn('if (launchFailed) {', source)
        self.assertIn('text: "This web-owned session failed before a usable session log was bound."', source)
        self.assertIn('function redactedLaunchErrorText(value)', source)
        self.assertIn('return codoxearLaunch.redactedLaunchErrorText(value);', source)
        self.assertIn('typeof codoxearLaunch.redactedLaunchErrorText !== "function"', source)
        self.assertIn('function redactedLaunchErrorText(value)', launch_source)
        self.assertIn('const sensitiveKey = "[A-Z0-9_.-]*(?:TOKEN|SECRET|KEY|PASSWORD|CREDENTIAL|AUTH)[A-Z0-9_.-]*";', launch_source)
        self.assertIn('const secretValue = "', launch_source)
        self.assertIn('(?:Bearer|Basic)', launch_source)
        self.assertIn('[^\\\\s\\\\\\"\',;}\\\\[\\\\]]+', launch_source)
        self.assertIn('$1=[redacted]', launch_source)
        self.assertIn('$1$2[redacted]', launch_source)
        self.assertIn('[A-Za-z0-9._~+/=-]{12,}', launch_source)
        self.assertNotIn('const sensitiveKey = "[A-Z0-9_.-]*(?:TOKEN|SECRET|KEY|PASSWORD|CREDENTIAL|AUTH)[A-Z0-9_.-]*";', source)
        self.assertIn('return redactedLaunchErrorText(s && s.launch_error) || "session launch failed";', source)
        self.assertIn('title: redactedLaunchErrorText(s.launch_error) || "Session launch failed"', source)
        self.assertIn('const safeLaunchError = redactedLaunchErrorText(s.launch_error);', source)
        self.assertIn('const launchError = launchFailed ? recoveryPromptPreview(redactedLaunchErrorText(s.launch_error), 1200) : "";', source)
        self.assertIn('const sessionEditActions = launchRow ? [] : [renameBtn, dupBtn];', source)
        self.assertIn('const rightActions = el("div", { class: "sessionActions right" }, sessionEditActions);', source)
        self.assertIn('const actions = el("div", { class: "sessionActionsInline" }, [...sessionEditActions, delBtn]);', source)
        self.assertIn('if (launchRow) {\n                 if (launchFailed) void selectSession(s.session_id);', source)
        self.assertNotIn('class: "msg assistant recovery-panel", role: "status"', source)
        self.assertIn('const anchor = typingRow && typingRow.isConnected ? typingRow : bottomSentinel;', source)
        self.assertIn('let pendingRecoveryFocusDescriptor = null;', source)
        self.assertIn('function focusedRecoveryActionDescriptor(sessionId)', source)
        self.assertIn('pendingRecoveryFocusDescriptor.sessionId === sessionId', source)
        self.assertIn('sessionId,\n            text:', source)
        self.assertIn('function focusRecoveryAction(row, descriptor)', source)
        self.assertIn('pendingRecoveryFocusDescriptor = descriptor;', source)
        self.assertIn('pendingRecoveryFocusDescriptor = null;', source)
        self.assertIn('function focusRecoveryFallback(descriptor)', source)
        self.assertIn('if (focusDescriptor && !focusRecoveryAction(row, focusDescriptor)) focusRecoveryFallback(focusDescriptor);', source)
        self.assertIn('focusRecoveryFallback(focusDescriptor);', source)
        self.assertIn('pendingRecoveryFocusDescriptor = null;\n            return;', source)
        self.assertIn('function syncRecoveryUiForSession(sessionId)', source)
        self.assertIn('setStatus({ running: currentRunning, queueLen });', source)
        self.assertIn('if (selected === sessionId) syncRecoveryUiForSession(sessionId);', source)
        self.assertIn('if (commitUnknown) syncRecoveryUiForSession(sessionId);', source)
        self.assertIn('syncRecoveryUiForSession(sid);', source)
        self.assertIn('syncRecoveryUiForSession(selected);', source)
        self.assertIn('if (typeof renderRecoveryPanelIfNeeded === "function") renderRecoveryPanelIfNeeded(typeof selected === "undefined" ? null : selected);', source)
        self.assertGreaterEqual(source.count('!row.classList.contains("typing-row") && !row.classList.contains("recovery-panel-row")'), 1)
        self.assertGreaterEqual(row_source.count('!row.classList.contains("typing-row") && !row.classList.contains("recovery-panel-row")'), 1)
        self.assertEqual(source.count('!x.classList.contains("typing-row") && !x.classList.contains("recovery-panel-row")'), 0)
        self.assertIn('return codoxearMessageRows.firstVisibleMessageRow(renderedMessageRows(), chat.scrollTop + 1);', source)
        self.assertIn('const targets = trimRenderedRowTargets(renderedMessageRows(), fromTop, maxRows);', source)
        self.assertIn('const targets = trimRowsBeforeViewportTargets(renderedMessageRows(), maxRows, chat.scrollTop + 1);', source)
        self.assertIn('document.querySelector(".recovery-panel .icon-btn") || queueBtn || null', source)
        self.assertIn('function selectedSessionLaunchFailed()', source)
        self.assertIn('queueControl.disabled = !!queueSubmitBusy || !selected || launchFailed || (unknownSend && !orphanQueueRecovery);', source)
        self.assertIn('sendControl.disabled = !!sending || !selected || launchFailed || unknownSend || orphanRecovery || recoveryQueue;', source)
        self.assertIn('setToast("failed launch cannot receive queued messages");', source)
        self.assertIn('setToast("failed launch cannot receive messages");', source)
        load_error_start = source.index('function renderTranscriptLoadError(sessionId, err')
        load_error_end = source.index('function applyCachedTail', load_error_start)
        load_error_block = source[load_error_start:load_error_end]
        self.assertLess(load_error_block.index('chatInner.insertBefore(row, bottomSentinel);'), load_error_block.index('setTyping(false);'))
        self.assertLess(load_error_block.index('setTyping(false);'), load_error_block.index('renderRecoveryPanelIfNeeded(sessionId);'))
        self.assertIn('renderRecoveryPanelIfNeeded(sessionId);\n          markClickFirstPaint();', source)
        self.assertIn('renderRecoveryPanelIfNeeded(selected);\n          markClickFirstPaint();', source)
        self.assertIn(".msg.recovery-panel", css)
        self.assertIn(".recoveryPanelActions", css)

    def test_launch_recovery_helpers_are_allowlisted(self) -> None:
        result = eval_launch_recovery_helpers()
        self.assertTrue(result["hasInfo"])
        details = result["details"]
        self.assertIn("state: launch failed", details)
        self.assertIn("launch stage: pty_fork", details)
        self.assertIn('launch error: pty fork failed before agent start API_TOKEN: [redacted] password: [redacted] "api_key":[redacted] Authorization: [redacted]', details)
        self.assertNotIn("secret-token", details)
        self.assertNotIn("hunter2", details)
        self.assertNotIn("json-secret", details)
        self.assertNotIn("abcdefghijklmnop", details)
        self.assertIn("model provider: macaron", details)
        self.assertIn("model: gpt-5.4", details)
        self.assertIn("reasoning: medium", details)
        self.assertIn("tmux: codoxear:work-abc123", details)
        self.assertIn("submitted prompts: 2", details)
        self.assertEqual(
            result["preset"],
            {
                "session_id": "launch-dead",
                "cwd": "/tmp/work",
                "agent_backend": "pi",
                "provider_choice": "macaron",
                "model_provider": "macaron",
                "preferred_auth_method": None,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
                "service_tier": "fast",
                "transport": "tmux",
                "tmux_session": "codoxear",
                "tmux_window": "work-abc123",
            },
        )
        self.assertNotIn("launch_state", result["preset"])
        self.assertNotIn("launch_error", result["preset"])

    def test_orphan_recovery_session_does_not_fetch_transcript_tail(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function openSession(")
        end = source.index("const cachedTail =", start)
        block = source[start:end]
        self.assertIn("if (s && s.orphan_recovery) {", block)
        self.assertIn("transcriptSlotRuntime.setActiveFailed();", block)
        self.assertIn("syncAttachButtonState();", block)
        self.assertIn("syncQueueSubmitState();", block)
        self.assertIn("syncSendButtonState();", block)
        self.assertIn("return { events: [], busy: false, queue_len: optimisticQueueLen, token: null, transcript_state: \"failed\" };", block)

    def test_new_command_begins_transcript_renewal_after_send_ack(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function sendText(")
        end = source.index("form.onsubmit = async", start)
        block = source[start:end]
        self.assertIn("const renewsTranscript = isTranscriptRenewalCommand(raw, sessionId);", block)
        self.assertIn("const sessionInfo = sessionIndex.get(sessionId) || null;", block)
        self.assertIn("if (sessionInfo && sessionInfo.commit_unknown_send) {", block)
        self.assertIn("void clearCommitUnknownSend(sessionId, sessionInfo.commit_unknown_send_text || \"\");", block)
        self.assertIn("const localAttachmentCount = typeof attachedFiles === \"number\" ? attachedFiles : 0;", block)
        self.assertIn("let allowPendingAttachment = Boolean(renderHere && localAttachmentCount > 0);", block)
        self.assertIn("sessionInfo && sessionInfo.pending_attachment", block)
        self.assertIn("window.confirm(\"This session has a pending file attachment. Send it with this message?\")", block)
        self.assertLess(block.index("window.confirm"), block.index("transcriptEventRuntime.addPendingUser"))
        self.assertIn("const res = await api(`/api/sessions/${sessionId}/send`, { method: \"POST\", body: { text: raw, allow_pending_attachment: allowPendingAttachment } });", block)
        self.assertIn("if (renderHere && renewsTranscript) {", block)
        self.assertIn("beginTranscriptRenewal(sessionId);", block)
        self.assertIn("renderPendingTranscriptSlot(sessionId);", block)
        self.assertLess(block.index("const res = await api"), block.index("beginTranscriptRenewal(sessionId);"))
        self.assertIn("if (renderHere && !renewsTranscript) {", block)
        self.assertIn("if (!transcriptScrollRuntime.snapshot().renderedAtLiveTail)", block)
        self.assertIn("const commitUnknown = Boolean(e2 && e2.obj && e2.obj.commit_unknown);", block)
        self.assertIn("setToast(\"send status unknown; check transcript before retrying\");", block)
        self.assertIn("currentSessionInfo.commit_unknown_send = true;", block)
        self.assertIn("currentSessionInfo.commit_unknown_send_text = raw;", block)
        self.assertIn("syncAttachButtonState();", block)
        self.assertIn("void refreshSessions().catch((e) => {", block)
        self.assertIn("if (e && e.status === 401) handleAppAuthLoss();", block)
        self.assertIn("else console.error(\"refreshSessions failed\", e);", block)
        self.assertIn("/broker must be restarted/i.test", block)
        self.assertIn("pending_attachment/clear", block)
        self.assertIn("attachment status unknown; check before retrying", source)
        self.assertIn("transcriptEventRuntime.dropPendingUsers(sessionId, (pending) => pending && pending.id === localId);", block)
        self.assertNotIn("pendingUser.splice", block)
        self.assertIn("const pendingRow = pendingEl.closest(\".msg-row\");", block)
        self.assertIn("if (pendingRow) pendingRow.remove();", block)
        self.assertIn("currentRunning = false;", block)


if __name__ == "__main__":
    unittest.main()
