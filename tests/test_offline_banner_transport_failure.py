from frontend_module_loader import NETWORK_STATUS_STUB_JS, module_path
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_POLLING_JS = module_path("app_polling.js")
APP_TRANSCRIPT_JS = module_path("app_transcript.js")
APP_NETWORK_JS = module_path("app_network.js")
APP_MESSAGE_FLOW_JS = module_path("app_message_flow.js")
APP_SESSION_CATALOG_JS = module_path("app_session_catalog.js")
APP_SESSION_STATE_JS = module_path("app_session_state.js")


def run_transport_failure_flow() -> dict:
    sources = [
        APP_POLLING_JS.read_text(encoding="utf-8"),
        APP_TRANSCRIPT_JS.read_text(encoding="utf-8"),
        APP_NETWORK_JS.read_text(encoding="utf-8"),
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

{NETWORK_STATUS_STUB_JS}
        const noop = () => {{}};
        const navigatorLike = {{ onLine: true }};
        const net = ctx.window.CodoxearNetwork.createNetworkStatusController({{ status, statusText, navigatorLike }});
        const timers = [];
        const eventSources = [];
        let fetchFails = true;
        let fetchAttempts = 0;
        async function fetch() {{
          fetchAttempts += 1;
          if (fetchFails) throw new TypeError("network connection lost");
          return {{ ok: true }};
        }}
        class FakeEventSource {{
          constructor(url) {{ this.url = url; this.listeners = {{}}; eventSources.push(this); }}
          addEventListener(type, listener) {{ this.listeners[type] = listener; }}
          close() {{ this.closed = true; }}
        }}
        const state = {{
          selected: "sid", generation: 1, disposed: false, turnOpen: false,
          active: {{ state: "bound", liveCursor: "c1", logPath: "/tmp/log" }},
          session: {{ session_id: "sid", agent_backend: "pi" }}, stats: {{ thinking: 0, thinkingTokens: 0, tools: 0 }},
        }};
        const typingRowRuntime = {{
          snapshot: () => ({{ stats: state.stats }}), updateTypingStats: noop,
          updateSubagentGauge: noop, resetTypingStats: noop, updateActivityPeek: noop,
        }};
        const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
        sessionState.set("selected", "sid");
        const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: () => {{}} }});
        sessionCatalog.set("latestSessions", [state.session]);
        const controller = ctx.window.CodoxearMessageFlow.createMessageFlowController({{
            sessionState, sessionCatalog,
          currentGeneration: () => state.generation,
          isAppDisposed: () => state.disposed,
          sessionLaunchFailed: () => false,
          api: async () => {{ await fetch(); return {{ events: [], busy: false, queue_len: 0, token: null }}; }},
          resolveAppUrl: (path) => `http://example.test${{path}}`, handleAppAuthLoss: noop,
          refreshSessions: async () => [], openSession: async () => null, clearSelectedSessionAfterRemoval: noop,
          activeTranscriptSnapshot: () => ({{ ...state.active }}),
          updateSessionTranscriptSlot: () => ({{ ignoredStaleBound: false, current: {{ state: "bound" }} }}),
          renderPendingTranscriptSlot: noop, renderSessionTail: noop, applySessionRuntimeFromTail: noop,
          resetChatRenderState: noop, setAttachCount: noop, setLiveCursor: (cursor) => {{ state.active.liveCursor = cursor; }},
          appendEvent: noop, appendTailSnapshotEvents: noop, setStatus: noop, setContext: noop, setTyping: noop,
          setSubagentsRunning: noop, updateSessionTitle: noop, initPageLimit: () => 60, typingRowRuntime,
           getCurrentRunning: () => false, setCurrentRunning: noop,
          getStagedAttachments: () => [], normalizedStagedAttachments: () => [],
          setSelectedSessionPendingAttachment: noop, syncSendButtonState: noop, syncAttachButtonState: noop,
          syncQueueSubmitState: noop, syncRecoveryUiForSession: noop, confirmAction: async () => false,
          setToast: noop, isTranscriptRenewalCommand: () => false, nextLocalEchoId: () => 1,
          renderedAtLiveTail: () => true, clearTranscriptDom: noop, clearRenderedTranscriptRange: noop,
          setOlderState: noop, getSessionTranscriptSlot: () => ({{ epoch: 1 }}), addPendingUser: noop,
          deleteTailCache: noop, beginTranscriptRenewal: noop, clearLiveCursor: noop, invalidateOlderLoad: noop,
          dropPendingUser: noop, removePendingUserRow: noop, hasPendingForSession: () => false, restoreUndeliveredDraft: () => false,
          visibilityState: () => "visible", navigatorValue: () => navigatorLike,
          reportTransportSuccess: () => net.reportSuccess(), reportTransportFailure: () => net.reportFailure(),
          EventSource: FakeEventSource, AbortController: null,
          setTimeout: (callback, delay) => {{ const timer = {{ callback, delay, cleared: false }}; timers.push(timer); return timer; }},
          clearTimeout: (timer) => {{ timer.cleared = true; }}, now: () => 1000, consoleWarn: noop, consoleError: noop,
        }});

        controller.openMessageEventSource("sid", 1);
        eventSources[0].onopen();
        eventSources[0].listeners.error();
        const firstPoll = timers.find((timer) => timer.delay === 2000 && !timer.cleared);
        Promise.resolve(firstPoll.callback()).then(() => {{
          const afterFailure = {{
            ...netSnap(net.sync()),
            fetchAttempts,
            sseClosed: Boolean(eventSources[0].closed),
            sourceCount: eventSources.length,
            hasRetryTimer: controller.snapshot().hasRetryTimer,
            transportUnavailable: controller.snapshot().messageTransportUnavailable,
          }};
          fetchFails = false;
          const recoveryPoll = timers.find((timer) => timer.delay === 4000 && !timer.cleared);
          return Promise.resolve(recoveryPoll.callback()).then(() => {{
            const afterRecovery = {{
              ...netSnap(net.sync()),
              fetchAttempts,
              sourceCount: eventSources.length,
              transportUnavailable: controller.snapshot().messageTransportUnavailable,
            }};
            eventSources[1].onopen();
            navigatorLike.onLine = false;
            const offlineState = net.sync();
            controller.closeMessageEventSource();
            const afterOffline = {{
              ...netSnap(offlineState),
              sseClosed: Boolean(eventSources[1].closed),
              hasEventSource: controller.snapshot().hasEventSource,
            }};
            navigatorLike.onLine = true;
            const onlineState = net.reportSuccess();
            controller.resetMessagePollBackoff();
            controller.resumeLiveDelivery();
            process.stdout.write(JSON.stringify({{
              afterFailure,
              afterRecovery,
              afterOffline,
              afterOnline: {{
                ...netSnap(onlineState),
                sourceCount: eventSources.length,
              }},
            }}));
          }});
        }}).catch((error) => {{ console.error(error); process.exit(1); }});
        """
    )
    # The embedded sources approach the kernel's MAX_ARG_STRLEN (128 KiB) for a
    # single argv entry, so run the program from a file instead of `node -e`.
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(script)
        program_path = handle.name
    try:
        completed = subprocess.run(["node", program_path], check=True, capture_output=True, text=True)
    finally:
        Path(program_path).unlink(missing_ok=True)
    return json.loads(completed.stdout)


def test_online_transport_failure_shows_connection_icon_and_recovers() -> None:
    result = run_transport_failure_flow()

    failure = result["afterFailure"]
    assert failure["fetchAttempts"] == 1
    # Two consecutive failures are required; one failure must not flash the icon.
    assert failure["state"] == ""
    assert failure["hidden"] is True
    assert failure["sseClosed"] is True
    assert failure["sourceCount"] == 1
    assert failure["hasRetryTimer"] is False
    assert failure["transportUnavailable"] is True

    recovery = result["afterRecovery"]
    assert recovery["fetchAttempts"] == 2
    assert recovery["state"] == ""
    assert recovery["hidden"] is True
    assert recovery["title"] is None
    assert recovery["spoken"] == ""
    assert recovery["sourceCount"] == 2
    assert recovery["transportUnavailable"] is False

    offline = result["afterOffline"]
    assert offline["state"] == "offline"
    assert offline["hidden"] is False
    assert offline["spoken"] == "Offline — waiting for a network connection. Updates retry automatically."
    assert offline["sseClosed"] is True
    assert offline["hasEventSource"] is False

    online = result["afterOnline"]
    assert online["state"] == ""
    assert online["hidden"] is True
    assert online["sourceCount"] == 3
