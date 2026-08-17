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


def _run_node(body: str) -> dict:
    sources = [
        APP_POLLING_JS.read_text(encoding="utf-8"),
        APP_TRANSCRIPT_JS.read_text(encoding="utf-8"),
        APP_MESSAGE_FLOW_JS.read_text(encoding="utf-8"),
        APP_SESSION_CATALOG_JS.read_text(encoding="utf-8"),
        APP_SESSION_STATE_JS.read_text(encoding="utf-8"),
    ]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, console, Date, URL, encodeURIComponent }};
        vm.createContext(ctx);
        {''.join(f'vm.runInContext({json.dumps(source)}, ctx);' for source in sources)}
        const noop = () => {{}};
        function createHarness(overrides = {{}}) {{
          const state = {{
            selected: "sid",
            generation: 1,
            disposed: false,
            turnOpen: false,
            running: false,
            sending: false,
            active: {{ state: "bound", liveCursor: "c1", logPath: "/tmp/log.jsonl" }},
            session: {{ session_id: "sid", agent_backend: "pi" }},
            stats: {{ thinking: 0, thinkingTokens: 0, thinkingMode: "blocks", tools: 0 }},
            resets: 0,
            events: [],
            toasts: [],
            statuses: [],
            apiCalls: [],
          }};
          const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
          sessionState.applyRuntime({{ selected: state.selected, turnOpen: state.turnOpen, sending: state.sending }});
          const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: () => {{}} }});
          sessionCatalog.set("latestSessions", [state.session]);
          Object.defineProperty(state, "running", {{
            get: () => sessionState.get("running"),
            set: (value) => sessionState.applyRuntime({{ running: Boolean(value) }}),
          }});
          const typingRowRuntime = {{
            snapshot: () => ({{ stats: {{ ...state.stats }} }}),
            updateTypingStats: (next, options = {{}}) => {{
              if (options.delta) {{
                state.stats.thinking += Number(next.thinking) || 0;
                state.stats.thinkingTokens += Number(next.thinkingTokens) || 0;
                state.stats.tools += Number(next.tools) || 0;
                state.stats.thinkingMode = next.thinkingMode || state.stats.thinkingMode;
              }} else {{
                state.stats = {{
                  thinking: Number(next.thinking) || 0,
                  thinkingTokens: Number(next.thinkingTokens) || 0,
                  thinkingMode: next.thinkingMode || "blocks",
                  tools: Number(next.tools) || 0,
                }};
              }}
            }},
            updateSubagentGauge: noop,
            resetTypingStats: () => {{
              state.resets += 1;
              state.stats = {{ thinking: 0, thinkingTokens: 0, thinkingMode: "blocks", tools: 0 }};
            }},
          }};
          const options = {{
            sessionState, sessionCatalog,
                        currentGeneration: () => state.generation,
            isAppDisposed: () => state.disposed,
            sessionLaunchFailed: () => false,
            api: async (path, options) => {{ state.apiCalls.push([path, options]); return {{}}; }},
            resolveAppUrl: (path) => `http://example.test${{path}}`,
            handleAppAuthLoss: noop,
            refreshSessions: async () => [],
            openSession: async () => null,
            clearSelectedSessionAfterRemoval: noop,
            activeTranscriptSnapshot: () => ({{ ...state.active }}),
            updateSessionTranscriptSlot: () => ({{ ignoredStaleBound: false, current: {{ state: "bound" }} }}),
            renderPendingTranscriptSlot: noop,
            renderSessionTail: noop,
            applySessionRuntimeFromTail: noop,
            resetChatRenderState: noop,
            setAttachCount: noop,
            setLiveCursor: (cursor) => {{ state.active.liveCursor = cursor; }},
            appendEvent: (event) => state.events.push(event),
            appendTailSnapshotEvents: noop,
            setStatus: (status) => state.statuses.push(status),
            setContext: noop,
            setTyping: noop,
            setSubagentsRunning: noop,
            updateSessionTitle: noop,
            initPageLimit: () => 60,
            typingRowRuntime,
                                    getCurrentRunning: () => state.running,
            setCurrentRunning: (value) => {{ state.running = Boolean(value); }},
            getStagedAttachments: () => [],
            normalizedStagedAttachments: (items) => Array.isArray(items) ? items : [],
            setSelectedSessionPendingAttachment: noop,
            syncSendButtonState: noop,
            syncAttachButtonState: noop,
            syncQueueSubmitState: noop,
            syncRecoveryUiForSession: noop,
            confirmAction: async () => false,
            setToast: (text) => state.toasts.push(text),
            isTranscriptRenewalCommand: () => false,
            nextLocalEchoId: () => 7,
            renderedAtLiveTail: () => true,
            clearTranscriptDom: noop,
            clearRenderedTranscriptRange: noop,
            setOlderState: noop,
            getSessionTranscriptSlot: () => ({{ epoch: 0 }}),
            addPendingUser: noop,
            deleteTailCache: noop,
            beginTranscriptRenewal: noop,
            clearLiveCursor: () => {{ state.active.liveCursor = null; }},
            invalidateOlderLoad: noop,
            dropPendingUser: noop,
            removePendingUserRow: noop,
            hasPendingForSession: () => false,
            visibilityState: () => "visible",
            navigatorValue: () => ({{ onLine: true }}),
            EventSource: null,
            AbortController: null,
            setTimeout: () => 0,
            clearTimeout: noop,
            now: () => 1000,
            consoleWarn: noop,
            consoleError: noop,
            ...overrides,
          }};
          return {{ state, options, create: () => ctx.window.CodoxearMessageFlow.createMessageFlowController(options) }};
        }}
        {body}
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def test_confirmed_send_resets_idle_typing_window_but_preserves_steer_counts() -> None:
    out = _run_node(
        """
        async function run(running) {
          const harness = createHarness({
            api: async () => ({ queued: false, queue_len: 0 }),
          });
          harness.state.running = running;
          harness.state.turnOpen = running;
          harness.options.sessionState.applyRuntime({ running, turnOpen: running });
          harness.state.stats.tools = 4;
          const controller = harness.create();
          const ok = await controller.sendText("steer or start");
          return {
            ok,
            resets: harness.state.resets,
            tools: harness.state.stats.tools,
            running: harness.state.running,
            toast: harness.state.toasts.at(-1),
          };
        }
        Promise.all([run(false), run(true)]).then(([idle, steer]) => {
          process.stdout.write(JSON.stringify({ idle, steer }));
        });
        """
    )
    assert out == {
        "idle": {"ok": True, "resets": 1, "tools": 0, "running": True, "toast": "sent"},
        "steer": {"ok": True, "resets": 0, "tools": 4, "running": True, "toast": "sent"},
    }



def test_control_slash_commands_do_not_open_a_conversation_turn_and_model_rechecks_sessions() -> None:
    out = _run_node(
        """
        const timers = [];
        let refreshes = 0;
        async function send(raw) {
          const harness = createHarness({
            api: async () => ({ queued: false, queue_len: 0 }),
            refreshSessions: async () => { refreshes += 1; },
            setTimeout: (callback, delay) => { timers.push({ callback, delay }); return timers.length; },
          });
          const ok = await harness.create().sendText(raw);
          return {
            ok,
            turnOpen: harness.options.sessionState.get("turnOpen"),
            running: harness.state.running,
            pendingEvents: harness.state.events.length,
            resets: harness.state.resets,
          };
        }
        Promise.all(["/model", "/effort", "/thinking", "/new"].map(send)).then(async (commands) => {
          const modelRefresh = timers.find((timer) => timer.delay === 1500);
          await modelRefresh.callback();
          process.stdout.write(JSON.stringify({ commands, refreshes, delayedRefreshes: timers.filter((timer) => timer.delay === 1500).length }));
        });
        """
    )
    assert out["commands"] == [
        {"ok": True, "turnOpen": False, "running": False, "pendingEvents": 0, "resets": 0},
        {"ok": True, "turnOpen": False, "running": False, "pendingEvents": 0, "resets": 0},
        {"ok": True, "turnOpen": False, "running": False, "pendingEvents": 0, "resets": 0},
        {"ok": True, "turnOpen": False, "running": False, "pendingEvents": 0, "resets": 0},
    ]
    assert out["delayedRefreshes"] == 1
    assert out["refreshes"] == 5

    out = _run_node(
        """
        const stateChanges = { renewals: 0, cacheDeletes: 0, pendingRenders: 0 };
        const harness = createHarness({
          api: async () => { throw new Error("broker down"); },
          isTranscriptRenewalCommand: () => true,
          deleteTailCache: () => { stateChanges.cacheDeletes += 1; },
          beginTranscriptRenewal: () => { stateChanges.renewals += 1; },
          renderPendingTranscriptSlot: () => { stateChanges.pendingRenders += 1; },
        });
        harness.create().sendText("/new").then((ok) => {
          process.stdout.write(JSON.stringify({
            ok,
            ...stateChanges,
            cursor: harness.state.active.liveCursor,
            toast: harness.state.toasts.at(-1),
          }));
        });
        """
    )
    assert out == {
        "ok": False,
        "renewals": 0,
        "cacheDeletes": 0,
        "pendingRenders": 0,
        "cursor": "c1",
        "toast": "send error: broker down",
    }


def test_sse_drop_polls_from_latest_cursor_then_reconnects_from_resumed_cursor() -> None:
    out = _run_node(
        """
        const sources = [];
        class FakeEventSource {
          constructor(url) { this.url = url; this.listeners = {}; this.closed = false; sources.push(this); }
          addEventListener(type, callback) { this.listeners[type] = callback; }
          close() { this.closed = true; }
          emit(type, payload = {}) { this.listeners[type](payload); }
        }
        const timers = [];
        let timerId = 0;
        const setTimer = (callback, delay) => {
          const timer = { id: ++timerId, callback, delay, canceled: false };
          timers.push(timer);
          return timer;
        };
        const clearTimer = (timer) => { if (timer) timer.canceled = true; };
        const liveRequests = [];
        const harness = createHarness({
          EventSource: FakeEventSource,
          setTimeout: setTimer,
          clearTimeout: clearTimer,
          api: async (path) => {
            liveRequests.push(path);
            return {
              transcript_state: "bound",
              thread_id: "thread-1",
              log_path: "/tmp/log.jsonl",
              live_cursor: "c3",
              events: [{ role: "assistant", text: "poll resumed" }],
              busy: true,
              queue_len: 0,
              token: null,
            };
          },
        });
        const controller = harness.create();
        controller.openMessageEventSource("sid", 1);
        sources[0].onopen();
        sources[0].emit("message", { data: JSON.stringify({
          transcript_state: "bound",
          thread_id: "thread-1",
          log_path: "/tmp/log.jsonl",
          live_cursor: "c2",
          events: [{ role: "assistant", text: "sse before drop" }],
          busy: true,
          queue_len: 0,
          token: null,
        }) });
        Promise.resolve().then(async () => {
          sources[0].emit("error");
          await controller.pollMessages("sid", 1);
          const retry = timers.filter((timer) => !timer.canceled).sort((a, b) => b.delay - a.delay)[0];
          retry.callback();
          process.stdout.write(JSON.stringify({
            firstUrl: sources[0].url,
            firstClosed: sources[0].closed,
            liveRequests,
            reconnectUrl: sources[1].url,
            cursor: harness.state.active.liveCursor,
            eventTexts: harness.state.events.map((event) => event.text),
            snapshot: controller.snapshot(),
          }));
        });
        """
    )
    assert out["firstUrl"].endswith("/api/sessions/sid/live?cursor=c1")
    assert out["firstClosed"] is True
    assert out["liveRequests"] == ["/api/sessions/sid/messages/live?cursor=c2"]
    assert out["reconnectUrl"].endswith("/api/sessions/sid/live?cursor=c3")
    assert out["cursor"] == "c3"
    assert out["eventTexts"] == ["sse before drop", "poll resumed"]
    assert out["snapshot"]["hasEventSource"] is True
