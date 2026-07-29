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
    # recoverySessionInfo moved into the CodoxearRecovery controller; the pure
    # helpers (recoveryPromptPreview / launchPresetFromSessionInfo /
    # recoveryDetailsText) stay in app.js and are exercised here so diagnostics
    # and recovery share a single source of truth.
    start = source.index("function recoveryPromptPreview(text, maxLen = 320)")
    end = source.index("function clearSelectedSessionAfterRemoval(sessionId, {", start)
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
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test = { launchPresetFromSessionInfo, recoveryDetailsText };\n")}, ctx);
        process.stdout.write(JSON.stringify({{
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
          sessionsWrap: {{ querySelectorAll: () => [], querySelector: () => null }},
          unattendedController: {{ isOpen: () => false, menuSessionId: () => null }},
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
          storageGetItem: () => null,
          saveSessionDraft: () => {{}},
          loadSessionDraft: () => {{}},
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
          unattendedController: {{ isOpen: () => true, menuSessionId: () => null }},
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
        self.assertContains(["abortMessagePollRequest"], state["calls"])
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
            self.assertContains(expected, state["calls"])

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

    def test_launch_recovery_helpers_are_allowlisted(self) -> None:
        result = eval_launch_recovery_helpers()
        details = result["details"]
        self.assertContains("state: launch failed", details)
        self.assertContains("launch stage: pty_fork", details)
        self.assertContains('launch error: pty fork failed before agent start API_TOKEN: [redacted] password: [redacted] "api_key":[redacted] Authorization: [redacted]', details)
        self.assertNotContains("secret-token", details)
        self.assertNotContains("hunter2", details)
        self.assertNotContains("json-secret", details)
        self.assertNotContains("abcdefghijklmnop", details)
        self.assertContains("model provider: macaron", details)
        self.assertContains("model: gpt-5.4", details)
        self.assertContains("reasoning: medium", details)
        self.assertContains("tmux: codoxear:work-abc123", details)
        self.assertContains("submitted prompts: 2", details)
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
        self.assertNotContains("launch_state", result["preset"])
        self.assertNotContains("launch_error", result["preset"])

        # orphan_recovery no longer short-circuits openSession
        # orphan_recovery early return was removed; openSession now fetches tail normally

if __name__ == "__main__":
    unittest.main()
