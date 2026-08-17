from frontend_module_loader import module_path
import json
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_POLLING_JS = module_path("app_polling.js")
APP_TRANSCRIPT_JS = module_path("app_transcript.js")
APP_MESSAGE_FLOW_JS = module_path("app_message_flow.js")
APP_SESSION_CATALOG_JS = module_path("app_session_catalog.js")
APP_SESSION_STATE_JS = module_path("app_session_state.js")


def run_flow(body: str) -> dict:
    sources = [
        APP_POLLING_JS.read_text(encoding="utf-8"),
        APP_TRANSCRIPT_JS.read_text(encoding="utf-8"),
        APP_MESSAGE_FLOW_JS.read_text(encoding="utf-8"),
        APP_SESSION_CATALOG_JS.read_text(encoding="utf-8"),
        APP_SESSION_STATE_JS.read_text(encoding="utf-8"),
    ]
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, console, Date, URL, encodeURIComponent }};
        vm.createContext(ctx);
        {''.join(f'vm.runInContext({json.dumps(source)}, ctx);' for source in sources)}
        const noop = () => {{}};
        function harness(overrides = {{}}) {{
          const state = {{
            selected: "sid", generation: 1, disposed: false, turnOpen: false,
            active: {{ state: "bound", liveCursor: "c1", logPath: "/tmp/log" }},
            session: {{ session_id: "sid", agent_backend: "pi" }}, events: [],
            errors: [], stats: {{ thinking: 0, thinkingTokens: 0, tools: 0 }},
          }};
          const typingRowRuntime = {{
            snapshot: () => ({{ stats: state.stats }}), updateTypingStats: noop,
            updateSubagentGauge: noop, resetTypingStats: noop,
          }};
          const options = {{
            sessionState: (() => {{ const store = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }}); store.applyRuntime({{ selected: state.selected, turnOpen: state.turnOpen, sending: state.sending }}); return store; }})(),
            sessionCatalog: (() => {{ const catalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: () => {{}} }}); catalog.set("latestSessions", [state.session]); return catalog; }})(),
            currentGeneration: () => state.generation,
            isAppDisposed: () => state.disposed, sessionLaunchFailed: () => false,
            api: async () => ({{ events: [], busy: false, queue_len: 0, token: null }}),
            resolveAppUrl: (path) => `http://example.test${{path}}`, handleAppAuthLoss: noop,
            refreshSessions: async () => [], openSession: async () => null, clearSelectedSessionAfterRemoval: noop,
            activeTranscriptSnapshot: () => ({{ ...state.active }}),
            updateSessionTranscriptSlot: () => ({{ ignoredStaleBound: false, current: {{ state: "bound" }} }}),
            renderPendingTranscriptSlot: noop, renderSessionTail: noop, applySessionRuntimeFromTail: noop,
            resetChatRenderState: noop, setAttachCount: noop,
            setLiveCursor: (cursor) => {{ state.active.liveCursor = cursor; }},
            appendEvent: (event) => state.events.push(event), appendTailSnapshotEvents: noop,
            setStatus: noop, setContext: noop, setTyping: noop, setSubagentsRunning: noop,
            updateSessionTitle: noop, initPageLimit: () => 60, typingRowRuntime,
            getCurrentRunning: () => false, setCurrentRunning: noop,
            getStagedAttachments: () => [], normalizedStagedAttachments: () => [],
            setSelectedSessionPendingAttachment: noop, syncSendButtonState: noop, syncAttachButtonState: noop,
            syncQueueSubmitState: noop, syncRecoveryUiForSession: noop, confirmAction: async () => false,
            setToast: (text) => state.errors.push(text), isTranscriptRenewalCommand: () => false,
            nextLocalEchoId: () => 1, renderedAtLiveTail: () => true, clearTranscriptDom: noop,
            clearRenderedTranscriptRange: noop, setOlderState: noop, getSessionTranscriptSlot: () => ({{ epoch: 1 }}),
            addPendingUser: noop, deleteTailCache: noop, beginTranscriptRenewal: noop, clearLiveCursor: noop,
            invalidateOlderLoad: noop, dropPendingUser: noop, removePendingUserRow: noop, hasPendingForSession: () => false,
            visibilityState: () => "visible", navigatorValue: () => ({{ onLine: true }}),
            EventSource: null, AbortController: null, setTimeout: () => 0, clearTimeout: noop,
            now: () => 1000, consoleWarn: noop, consoleError: noop, ...overrides,
          }};
          return {{ state, options, controller: ctx.window.CodoxearMessageFlow.createMessageFlowController(options) }};
        }}
        {body}
        """
    )
    completed = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def test_commit_unknown_patch_updates_catalog_projection_without_imperative_render_relays() -> None:
    result = run_flow(
        """
        const projectionCalls = [];
        let test;
        test = harness({
          api: async () => { const error = new Error("unknown"); error.obj = { commit_unknown: true }; throw error; },
          refreshSessions: async () => [],
          setTimeout: () => 0,
        });
        test.options.sessionCatalog.subscribe("sessionIndex", () => {
          const session = test.options.sessionCatalog.get("sessionIndex").get("sid");
          projectionCalls.push({ unknown: session.commit_unknown_send, text: session.commit_unknown_send_text });
        });
        test.controller.sendText("possibly sent").then((sent) => process.stdout.write(JSON.stringify({
          sent,
          projectionCalls,
          session: test.options.sessionCatalog.get("sessionIndex").get("sid"),
          toast: test.state.errors[0],
        })));
        """
    )
    assert result["sent"] is False
    assert result["projectionCalls"] == [{"unknown": True, "text": "possibly sent"}]
    assert result["session"]["commit_unknown_send"] is True
    assert result["session"]["commit_unknown_send_text"] == "possibly sent"
    assert result["toast"] == "sending..."


def test_poll_generation_prevents_stale_tick_from_scheduling_another_poll() -> None:
    result = run_flow(
        """
        const timers = [];
        const test = harness({
          setTimeout: (callback, delay) => { const timer = { callback, delay, cleared: false }; timers.push(timer); return timer; },
          clearTimeout: (timer) => { timer.cleared = true; },
          api: async () => { test.state.generation = 2; return { live_cursor: "c2", events: [], busy: false, queue_len: 0, token: null }; },
        });
        test.controller.kickPoll(0);
        Promise.resolve(timers[0].callback()).then(() => process.stdout.write(JSON.stringify({
          timerCount: timers.length, firstDelay: timers[0].delay, staleTimerCleared: timers[0].cleared,
        })));
        """
    )
    assert result == {"timerCount": 1, "firstDelay": 0, "staleTimerCleared": False}


def test_aborted_tail_request_is_distinguished_from_poll_failure() -> None:
    result = run_flow(
        """
        class AbortController { constructor() { this.signal = { aborted: false }; } abort() { this.signal.aborted = true; } }
        const test = harness({ AbortController });
        const request = test.controller.beginOpenSessionTailRequest("sid", 1);
        test.controller.abortOpenSessionTailRequest();
        process.stdout.write(JSON.stringify({
          aborted: request.signal.aborted,
          recognized: test.controller.isOpenSessionTailAbortError(request, { name: "AbortError" }),
          errorStreak: test.controller.snapshot().messagePollErrorStreak,
        }));
        """
    )
    assert result == {"aborted": True, "recognized": True, "errorStreak": 0}



def test_visibility_resume_reuses_inflight_sse_without_a_fallback_poll() -> None:
    result = run_flow(
        """
        const sources = [];
        const timers = [];
        class FakeEventSource {
          constructor(url) { this.url = url; this.listeners = {}; sources.push(this); }
          addEventListener(type, listener) { this.listeners[type] = listener; }
          close() { this.closed = true; }
        }
        const test = harness({
          EventSource: FakeEventSource,
          setTimeout: (callback, delay) => { const timer = { callback, delay }; timers.push(timer); return timer; },
          clearTimeout: noop,
        });
        const opened = test.controller.openMessageEventSource("sid", 1);
        test.controller.resumeLiveDelivery();
        process.stdout.write(JSON.stringify({ opened, streams: sources.length, fallbackTimers: timers.length, snapshot: test.controller.snapshot() }));
        """
    )
    assert result["opened"] is True
    assert result["streams"] == 1
    assert result["fallbackTimers"] == 0
    assert result["snapshot"]["hasEventSource"] is True
    assert result["snapshot"]["messageSseOpen"] is False


def test_sse_reconnect_waits_until_document_is_visible() -> None:
    result = run_flow(
        """
        const sources = [];
        const timers = [];
        let visibility = "visible";
        class FakeEventSource {
          constructor(url) { this.url = url; this.listeners = {}; sources.push(this); }
          addEventListener(type, listener) { this.listeners[type] = listener; }
          close() { this.closed = true; }
        }
        const test = harness({
          visibilityState: () => visibility, EventSource: FakeEventSource,
          setTimeout: (callback, delay) => { const timer = { callback, delay }; timers.push(timer); return timer; },
          clearTimeout: noop,
        });
        test.controller.openMessageEventSource("sid", 1);
        sources[0].onopen();
        sources[0].listeners.error();
        const retry = timers.find((timer) => timer.delay === 6000);
        visibility = "hidden";
        retry.callback();
        const afterHiddenRetry = sources.length;
        visibility = "visible";
        test.controller.resumeLiveDelivery();
        process.stdout.write(JSON.stringify({
          retryDelay: retry.delay, afterHiddenRetry, afterVisibleResume: sources.length,
        }));
        """
    )
    assert result == {"retryDelay": 6000, "afterHiddenRetry": 1, "afterVisibleResume": 2}
