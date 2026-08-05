import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app_application.js"
APP_COMPOSITION_JS = ROOT / "codoxear" / "static" / "app_application_composition.js"
APP_CHAT_INTERACTION_JS = ROOT / "codoxear" / "static" / "app_chat_interaction.js"
APP_SESSION_LIFECYCLE_JS = ROOT / "codoxear" / "static" / "app_session_lifecycle.js"
APP_DISPLAY_JS = ROOT / "codoxear" / "static" / "app_display.js"
APP_LAUNCH_JS = ROOT / "codoxear" / "static" / "app_launch.js"
APP_TRANSCRIPT_JS = ROOT / "codoxear" / "static" / "app_transcript.js"
APP_MESSAGE_ROWS_JS = ROOT / "codoxear" / "static" / "app_message_rows.js"
APP_CSS = ROOT / "codoxear" / "static" / "app.css"


def eval_launch_recovery_details() -> dict:
    source = "\n".join((
        APP_JS.read_text(encoding="utf-8"),
        APP_COMPOSITION_JS.read_text(encoding="utf-8"),
        APP_CHAT_INTERACTION_JS.read_text(encoding="utf-8"),
    ))
    display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    launch_source = APP_LAUNCH_JS.read_text(encoding="utf-8")
    redactor_start = source.index("function redactedLaunchErrorText(value) {")
    redactor_end = source.index("function sessionLaunchLabel(s)", redactor_start)
    # The pure helpers (recoveryPromptPreview / recoveryDetailsText) stay in
    # app.js so diagnostics and recovery share a single source of truth.
    start = source.index("function recoveryPromptPreview(text, maxLen = 320)")
    end = source.index("async function dismissFailedLaunchRecord(sessionId)", start)
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
    source = APP_SESSION_LIFECYCLE_JS.read_text(encoding="utf-8")
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
        const controller = ctx.window.CodoxearSessionLifecycle.createSessionLifecycleController({{
          nextPollGeneration: () => ++state.pollGen,
          incrementPollGeneration: () => ++state.pollGen,
          prepareSessionOpen: () => messageFlow.prepareSessionOpen(),
          getSelected: () => state.selected,
          setSelected: (sid) => {{ state.selected = sid; }},
          setActiveSession: () => {{}}, saveComposerDraft: () => {{}}, loadComposerDraft: () => {{}},
          closeUnattendedForOtherSession: () => {{}}, persistSelected: () => {{}}, removePersistedSelected: () => {{}}, setSessionHash: () => {{}},
          resetTranscriptForSession: () => {{}}, clearTranscriptForRemovedSession: () => {{}}, syncAttachments: () => {{}}, clearAttachments: () => {{}}, syncAttachmentButton: () => {{}}, updateQueueBadge: () => {{}},
          setStatus: () => {{}}, setContext: () => {{}}, setTyping: () => {{}}, resetChatRenderState: () => {{}},
          getSession: (sid) => sessions.get(sid),
          isCurrent: (sid, generation) => state.selected === sid && state.pollGen === generation,
          setTitle: (session) => {{ state.title = `title:${{session.session_id}}`; }}, setNoSessionTitle: () => {{}}, markClickLoad: () => {{}},
          setTurnOpen: () => {{}}, updateTypingStats: () => {{}}, beginFileViewerSync: () => false, handleFileViewerSessionUnavailable: () => {{}},
          finishFileViewerSync: () => {{}}, getTailCache: () => null, tailCacheMatchesSession: () => false,
          applyCachedTail: () => {{}}, renderTranscriptLoading: () => {{}}, messageFlow: () => messageFlow, api,
          initPageLimit: () => 60, handleAuthLoss: () => calls.push(["handleAuthLoss"]),
          refreshSessions: async () => {{}},
          renderTranscriptLoadError: () => calls.push(["renderTranscriptLoadError"]), isDisposed: () => false,
          kickPoll: () => {{}}, messagePollDelayMs: () => 900,
          updateTranscriptSlot: () => ({{ ignoredStaleBound: false, current: {{ state: "bound" }} }}),
          renderPendingTranscriptSlot: () => {{}}, applySessionRuntimeFromTail: () => {{}},
          renderSessionTail: () => calls.push(["renderSessionTail"]), openMessageEventSource: () => {{}},
          isMobile: () => false, closeSidebar: () => {{}}, updateUnattendedButton: () => {{}},
          refreshFileCandidates: async () => {{}}, isUnattendedOpen: () => false, hideUnattendedMenu: () => {{}},
          syncComposerSendButton: () => {{}}, syncQueueSubmitState: () => {{}}, setActiveTranscriptPending: () => {{}},
          deleteTranscriptSession: () => {{}}, dropPendingUserRows: () => {{}}, sessionIdFromHash: () => "", rememberPendingHashSession: () => {{}},
          sessionSelectable: () => false, normalizeAgentBackendName: (value) => value, providerChoiceToSettings: () => ({{}}),
          backendSupportsFast: () => false, setToast: () => {{}}, confirmAction: async () => false, syncRecoveryUiForSession: () => {{}}, sleep: async () => {{}}, consoleError: () => {{}},
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

def _run_lifecycle(body: str) -> dict:
    source = APP_SESSION_LIFECYCLE_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const calls = [];
        const noop = () => {{}};
        const selected = {{ value: "sid-1" }};
        const options = new Proxy({{
          getSelected: () => selected.value,
          setSelected: (value) => {{ selected.value = value; }},
          incrementPollGeneration: () => {{ calls.push(["incrementPollGeneration"]); }},
          messageFlow: () => ({{ abortMessagePollRequest: () => calls.push(["abortMessagePollRequest"]), clearPollSchedule: () => calls.push(["clearPollSchedule"]) }}),
          handleFileViewerSessionUnavailable: (sid) => calls.push(["handleFileViewerSessionUnavailable", sid]),
          setActiveTranscriptPending: () => calls.push(["setActiveTranscriptPending"]),
          clearTranscriptForRemovedSession: () => calls.push(["clearTranscriptForRemovedSession"]),
          removePersistedSelected: () => calls.push(["removePersistedSelected"]),
          setSessionHash: (value) => calls.push(["setSessionHash", value]),
          setNoSessionTitle: () => calls.push(["setNoSessionTitle"]),
          setStatus: (value) => calls.push(["setStatus", value]), setContext: (value) => calls.push(["setContext", value]), setTyping: (value) => calls.push(["setTyping", value]),
          clearAttachments: () => calls.push(["clearAttachments"]), syncAttachmentButton: () => calls.push(["syncAttachmentButton"]), resetChatRenderState: () => calls.push(["resetChatRenderState"]), updateQueueBadge: () => calls.push(["updateQueueBadge"]),
          isUnattendedOpen: () => true, hideUnattendedMenu: () => calls.push(["hideUnattendedMenu"]), updateUnattendedButton: () => calls.push(["updateUnattendedButton"]),
          syncComposerSendButton: () => calls.push(["syncComposerSendButton"]), syncQueueSubmitState: () => calls.push(["syncQueueSubmitState"]),
          deleteTranscriptSession: (sid) => calls.push(["deleteTranscriptSession", sid]), dropPendingUserRows: (sid) => calls.push(["dropPendingUserRows", sid]),
        }}, {{ get: (target, name) => name in target ? target[name] : noop }});
        const controller = ctx.window.CodoxearSessionLifecycle.createSessionLifecycleController(options);
        {body}
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_clear_deleted_session_client_state() -> dict:
    return _run_lifecycle(
        """
        const selectedResult = controller.clearDeletedSessionClientState("sid-1");
        const selectedCalls = calls.slice();
        calls.length = 0;
        const otherResult = controller.clearDeletedSessionClientState("other");
        process.stdout.write(JSON.stringify({ selectedResult, selectedCalls, otherResult, otherCalls: calls }));
        """
    )


def eval_clear_selected_session_after_removal() -> dict:
    return _run_lifecycle(
        """
        const noopResult = controller.clearSelectedSessionAfterRemoval("other", { incrementPollGen: true, clearPollState: true });
        const noopCalls = calls.slice();
        calls.length = 0;
        const applied = controller.clearSelectedSessionAfterRemoval("sid-1", { incrementPollGen: true, clearPollState: true });
        process.stdout.write(JSON.stringify({ noopResult, noopCalls, applied, calls }));
        """
    )


class TestChatScrollbackSource(unittest.TestCase):
    def test_clear_deleted_session_client_state_clears_explicit_delete_state(self) -> None:
        result = eval_clear_deleted_session_client_state()
        self.assertTrue(result["selectedResult"])
        self.assertContains(["deleteTranscriptSession", "sid-1"], result["selectedCalls"])
        self.assertContains(["dropPendingUserRows", "sid-1"], result["selectedCalls"])
        self.assertFalse(result["otherResult"])
        self.assertEqual(result["otherCalls"], [["deleteTranscriptSession", "other"], ["dropPendingUserRows", "other"]])

    def test_clear_selected_session_after_removal_resets_missing_session_state(self) -> None:
        result = eval_clear_selected_session_after_removal()
        self.assertFalse(result["noopResult"])
        self.assertEqual(result["noopCalls"], [])
        self.assertTrue(result["applied"])
        self.assertEqual(result["calls"][0], ["handleFileViewerSessionUnavailable", "sid-1"])
        for expected in [
            ["abortMessagePollRequest"], ["clearPollSchedule"], ["incrementPollGeneration"], ["setActiveTranscriptPending"],
            ["clearTranscriptForRemovedSession"], ["removePersistedSelected"], ["setSessionHash", ""], ["setNoSessionTitle"],
            ["setStatus", {"running": False, "queueLen": 0}], ["setContext", None], ["setTyping", False], ["clearAttachments"],
            ["syncAttachmentButton"], ["resetChatRenderState"], ["updateQueueBadge"], ["hideUnattendedMenu"], ["updateUnattendedButton"],
            ["syncComposerSendButton"], ["syncQueueSubmitState"],
        ]:
            self.assertContains(expected, result["calls"])

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
