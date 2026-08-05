import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app_application.js"
APP_SESSION_OPEN_JS = ROOT / "codoxear" / "static" / "app_session_open.js"
APP_DISPLAY_JS = ROOT / "codoxear" / "static" / "app_display.js"
APP_LAUNCH_JS = ROOT / "codoxear" / "static" / "app_launch.js"
APP_TRANSCRIPT_JS = ROOT / "codoxear" / "static" / "app_transcript.js"
APP_MESSAGE_ROWS_JS = ROOT / "codoxear" / "static" / "app_message_rows.js"
APP_CSS = ROOT / "codoxear" / "static" / "app.css"


def eval_launch_recovery_details() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    launch_source = APP_LAUNCH_JS.read_text(encoding="utf-8")
    redactor_start = source.index("function redactedLaunchErrorText(value) {")
    redactor_end = source.index("function sessionLaunchLabel(s)", redactor_start)
    # The pure helpers (recoveryPromptPreview / recoveryDetailsText) stay in
    # app.js so diagnostics and recovery share a single source of truth.
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
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test = { recoveryDetailsText };\n")}, ctx);
        process.stdout.write(JSON.stringify({{
          details: ctx.__test.recoveryDetailsText("launch-dead", launchRow),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)



def eval_open_session_tail_request_abort() -> dict:
    source = APP_SESSION_OPEN_JS.read_text(encoding="utf-8")
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
              addEventListener(type, callback) {{ if (type === "abort") listeners.push(callback); }},
              listeners,
            }};
          }}
          abort() {{
            if (this.signal.aborted) return;
            this.signal.aborted = true;
            this.signal.listeners.slice().forEach((callback) => callback());
          }}
        }}
        const ctx = {{ window: {{}}, console, AbortController }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const sessions = new Map([
          ["sid-a", {{ session_id: "sid-a", busy: false, queue_len: 0, token: null }}],
          ["sid-b", {{ session_id: "sid-b", busy: false, queue_len: 0, token: null }}],
        ]);
        const state = {{ selected: null, pollGen: 0, title: "" }};
        let activeTailController = null;
        const abortOpen = () => {{
          const controller = activeTailController;
          activeTailController = null;
          if (controller) controller.abort();
        }};
        const messageFlow = {{
          prepareSessionOpen: abortOpen,
          beginOpenSessionTailRequest(sessionId, generation) {{
            abortOpen();
            const controller = new AbortController();
            activeTailController = controller;
            return {{ sessionId, generation, controller, signal: controller.signal }};
          }},
          isOpenSessionTailAbortError: (request, error) => Boolean(error && error.name === "AbortError" && request.signal.aborted),
          isCurrentOpenSessionTailRequest: (request) => state.selected === request.sessionId && state.pollGen === request.generation,
          finishOpenSessionTailRequest(request) {{ if (activeTailController === request.controller) activeTailController = null; }},
          markMessagePollFailure: () => calls.push(["markMessagePollFailure"]),
          markMessagePollSuccess: () => calls.push(["markMessagePollSuccess"]),
        }};
        const api = (url, options = {{}}) => new Promise((resolve, reject) => {{
          calls.push(["api", url, Boolean(options.signal)]);
          const request = {{ url, signal: options.signal, resolve, reject }};
          pending.push(request);
          options.signal.addEventListener("abort", () => {{
            calls.push(["abort", url]);
            const error = new Error("aborted");
            error.name = "AbortError";
            reject(error);
          }});
        }});
        const controller = ctx.window.CodoxearSessionOpen.createSessionOpenController({{
          nextPollGeneration: () => ++state.pollGen,
          prepareSessionOpen: () => messageFlow.prepareSessionOpen(),
          getSelected: () => state.selected,
          setSelected: (sid) => {{ state.selected = sid; }},
          setActiveSession: () => {{}}, saveComposerDraft: () => {{}}, loadComposerDraft: () => {{}},
          closeUnattendedForOtherSession: () => {{}}, persistSelected: () => {{}}, setSessionHash: () => {{}},
          resetTranscriptForSession: () => {{}}, syncAttachments: () => {{}}, updateQueueBadge: () => {{}},
          setStatus: () => {{}}, setContext: () => {{}}, setTyping: () => {{}}, resetChatRenderState: () => {{}},
          getSession: (sid) => sessions.get(sid),
          isCurrent: (sid, generation) => state.selected === sid && state.pollGen === generation,
          setTitle: (session) => {{ state.title = `title:${{session.session_id}}`; }}, markClickLoad: () => {{}},
          setTurnOpen: () => {{}}, updateTypingStats: () => {{}}, beginFileViewerSync: () => false,
          finishFileViewerSync: () => {{}}, getTailCache: () => null, tailCacheMatchesSession: () => false,
          applyCachedTail: () => {{}}, renderTranscriptLoading: () => {{}}, messageFlow: () => messageFlow, api,
          initPageLimit: () => 60, handleAuthLoss: () => calls.push(["handleAuthLoss"]),
          clearRemovedSession: () => {{}}, refreshSessions: async () => {{}},
          renderTranscriptLoadError: () => calls.push(["renderTranscriptLoadError"]), isDisposed: () => false,
          kickPoll: () => {{}}, messagePollDelayMs: () => 900,
          updateTranscriptSlot: () => ({{ ignoredStaleBound: false, current: {{ state: "bound" }} }}),
          renderPendingTranscriptSlot: () => {{}}, applySessionRuntimeFromTail: () => {{}},
          renderSessionTail: () => calls.push(["renderSessionTail"]), openMessageEventSource: () => {{}},
          isMobile: () => false, closeSidebar: () => {{}}, updateUnattendedButton: () => {{}},
          refreshFileCandidates: async () => {{}}, consoleError: () => {{}},
        }});
        (async () => {{
          const firstPromise = controller.openSession("sid-a", {{ useCache: false }});
          const firstSignal = pending[0].signal;
          const secondPromise = controller.openSession("sid-b", {{ useCache: false }});
          const secondRequest = pending[1];
          secondRequest.resolve({{ transcript_state: "bound", events: [{{ role: "assistant", text: "ok" }}], busy: false, queue_len: 0, token: null }});
          const secondResult = await secondPromise;
          const firstResult = await firstPromise;
          process.stdout.write(JSON.stringify({{
            firstResult, secondResult, firstSignalAborted: firstSignal.aborted,
            secondSignalAborted: secondRequest.signal.aborted, pollGen: state.pollGen,
            selected: state.selected, title: state.title,
            apiCalls: calls.filter((call) => call[0] === "api"),
            abortCalls: calls.filter((call) => call[0] === "abort"),
            failureCalls: calls.filter((call) => call[0] === "markMessagePollFailure"),
            loadErrorCalls: calls.filter((call) => call[0] === "renderTranscriptLoadError"),
            successCalls: calls.filter((call) => call[0] === "markMessagePollSuccess"),
            renderTailCalls: calls.filter((call) => call[0] === "renderSessionTail"),
          }}));
        }})().catch((error) => {{ console.error(error && error.stack || error); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)

def eval_clear_deleted_session_client_state() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function clearDeletedSessionClientState(")
    end = source.index("function syncRecoveryUiForSession", start)
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
    end = source.index("function syncRecoveryUiForSession", start)
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
          clearTimeout: (timer) => calls.push(["clearTimeout", timer]),
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
          syncComposerSendButton: () => calls.push(["syncSendButtonState"]),
          syncQueueSubmitState: () => calls.push(["syncQueueSubmitState"]),
          syncAttachButtonState: () => calls.push(["syncAttachButtonState"]),
          attachmentsController: {{
            setStagedAttachments: (...args) => calls.push(["attachmentsController.setStagedAttachments", ...args]),
            syncAttachButtonState: () => calls.push(["attachmentsController.syncAttachButtonState"]),
          }},
          messageFlowController: {{
            abortMessagePollRequest: () => calls.push(["abortMessagePollRequest"]),
            clearPollSchedule: () => {{
              if (ctx.pollTimer) ctx.clearTimeout(ctx.pollTimer);
              ctx.pollTimer = null;
              ctx.pollKickPending = false;
              ctx.pollKickDelayMs = null;
            }},
          }},
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
            ["attachmentsController.setStagedAttachments", []],
            ["resetChatRenderState"],
            ["updateQueueBadge"],
            ["hideUnattendedMenu"],
            ["updateUnattendedBtnState"],
            ["syncSendButtonState"],
            ["syncQueueSubmitState"],
            ["attachmentsController.syncAttachButtonState"],
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
        self.assertTrue(all(call[1].endswith("/messages/tail?limit=60") for call in result["apiCalls"]))
        self.assertTrue(all(call[2] for call in result["apiCalls"]))
        self.assertEqual(len(result["abortCalls"]), 1)
        self.assertEqual(result["failureCalls"], [])
        self.assertEqual(result["loadErrorCalls"], [])
        self.assertEqual(result["successCalls"], [["markMessagePollSuccess"]])
        self.assertEqual(len(result["renderTailCalls"]), 1)

    def test_launch_recovery_details_are_allowlisted(self) -> None:
        result = eval_launch_recovery_details()
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

        # orphan_recovery no longer short-circuits openSession
        # orphan_recovery early return was removed; openSession now fetches tail normally

if __name__ == "__main__":
    unittest.main()
