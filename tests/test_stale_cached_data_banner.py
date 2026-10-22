"""Behavior tests for stale-cache visibility: when the service worker answers
API polls from its offline cache, the network banner must say so instead of
rendering stale content as if it were fresh.

Chain under test:
  service-worker marks fallback responses (X-Codoxear-Stale)
  -> app_api surfaces the mark (apiResponseStale, non-enumerable)
  -> poll success paths route it to networkStatus.reportStale()
  -> the banner shows the "showing cached data" variant.
"""
from frontend_module_loader import module_path
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path


APP_APPLICATION_JS = module_path("app_application.js")
APP_API_JS = module_path("app_api.js")
APP_NETWORK_JS = module_path("app_network.js")


def run_node_json(js: str) -> dict:
    # Embedded modules can approach the kernel's MAX_ARG_STRLEN (128 KiB) for a
    # single argv entry, so run the program from a file instead of `node -e`.
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(js)
        program_path = handle.name
    try:
        proc = subprocess.run(["node", program_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    finally:
        Path(program_path).unlink(missing_ok=True)
    return json.loads(proc.stdout)


def test_api_surfaces_stale_mark_without_polluting_payload() -> None:
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        let staleHeader = null;
        const ctx = {{
          URL, console, JSON,
          performance: {{ now: () => 0 }},
          fetch: async () => ({{
            status: 200,
            ok: true,
            headers: {{ get(name) {{
              if (String(name).toLowerCase() === "x-codoxear-stale") return staleHeader;
              if (String(name).toLowerCase() === "etag") return "etag-1";
              return null;
            }} }},
            text: async () => '{{"sessions":[{{"session_id":"s1"}}]}}',
          }}),
          window: {{
            location: {{ href: "http://localhost/", origin: "http://localhost", pathname: "/" }},
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(APP_APPLICATION_JS.read_text(encoding="utf-8"))}, ctx);
        vm.runInContext({json.dumps(APP_API_JS.read_text(encoding="utf-8"))}, ctx);
        (async () => {{
          const api = ctx.window.CodoxearApi.api;
          const apiResponseStale = ctx.window.CodoxearApi.apiResponseStale;

          const live = await api("/api/sessions");
          const liveResult = {{ stale: apiResponseStale(live) }};

          staleHeader = "1";
          const stale = await api("/api/sessions");
          const staleResult = {{
            stale: apiResponseStale(stale),
            payloadIntact: Array.isArray(stale.sessions) && stale.sessions[0].session_id === "s1",
            enumerableKeys: Object.keys(stale),
            serialized: JSON.stringify(stale),
          }};

          staleHeader = null;
          const recovered = await api("/api/sessions");
          const recoveredResult = {{ stale: apiResponseStale(recovered) }};

          process.stdout.write(JSON.stringify({{ liveResult, staleResult, recoveredResult }}));
        }})().catch((error) => {{ console.error(error); process.exit(1); }});
        """
    )
    out = run_node_json(script)
    assert out["liveResult"]["stale"] is False
    assert out["staleResult"]["stale"] is True
    assert out["staleResult"]["payloadIntact"] is True
    # The flag is non-enumerable: payload shape and serialization are unchanged,
    # so reducers and caches cannot tell a marked response apart from a live one.
    assert out["staleResult"]["enumerableKeys"] == ["sessions"]
    assert "Stale" not in out["staleResult"]["serialized"]
    assert out["recoveredResult"]["stale"] is False


def test_network_banner_stale_state() -> None:
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(APP_NETWORK_JS.read_text(encoding="utf-8"))}, ctx);
        const banner = {{
          textContent: "", hidden: true, attributes: {{}},
          setAttribute(name, value) {{ this.attributes[name] = String(value); }},
        }};
        const navigatorLike = {{ onLine: true }};
        const status = ctx.window.CodoxearNetwork.createNetworkStatusController({{ banner, navigatorLike }});
        const snap = () => ({{ text: banner.textContent, hidden: banner.hidden }});

        const initial = snap();
        status.reportStale();
        const afterStale = snap();
        // One stale answer is proof of unreachability: no failure threshold.
        status.reportSuccess();
        const afterSuccess = snap();

        status.reportStale();
        navigatorLike.onLine = false;
        status.sync();
        const whileOffline = snap();
        navigatorLike.onLine = true;
        status.sync();
        const backOnline = snap();
        status.reportSuccess();
        const afterRecovery = snap();

        process.stdout.write(JSON.stringify({{ initial, afterStale, afterSuccess, whileOffline, backOnline, afterRecovery }}));
        """
    )
    out = run_node_json(script)
    assert out["initial"]["hidden"] is True
    assert out["afterStale"]["hidden"] is False
    assert "cached data" in out["afterStale"]["text"]
    assert "Connection unavailable" in out["afterStale"]["text"]
    assert out["afterSuccess"]["hidden"] is True
    # The browser-offline signal is the stronger statement and wins while it holds.
    assert "Offline" in out["whileOffline"]["text"]
    # Connectivity returning does not erase the fact that the server is still
    # unreachable: the stale state persists until a live response arrives.
    assert "cached data" in out["backOnline"]["text"]
    assert out["afterRecovery"]["hidden"] is True


def test_message_poll_stale_success_reports_degraded_transport() -> None:
    sources = [
        module_path("app_polling.js").read_text(encoding="utf-8"),
        module_path("app_transcript.js").read_text(encoding="utf-8"),
        APP_NETWORK_JS.read_text(encoding="utf-8"),
        module_path("app_message_flow.js").read_text(encoding="utf-8"),
        module_path("app_session_catalog.js").read_text(encoding="utf-8"),
        module_path("app_session_state.js").read_text(encoding="utf-8"),
    ]
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, console, Date, URL, encodeURIComponent }};
        vm.createContext(ctx);
        {''.join(f'vm.runInContext({json.dumps(source)}, ctx);' for source in sources)}

        const noop = () => {{}};
        const banner = {{
          textContent: "", hidden: true, attributes: {{}},
          setAttribute(name, value) {{ this.attributes[name] = String(value); }},
        }};
        const navigatorLike = {{ onLine: true }};
        const networkStatus = ctx.window.CodoxearNetwork.createNetworkStatusController({{ banner, navigatorLike }});
        const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: noop }});
        sessionState.set("selected", "sid");
        const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: noop }});
        sessionCatalog.set("latestSessions", [{{ session_id: "sid", agent_backend: "pi" }}]);
        const active = {{ state: "bound", liveCursor: "c1", logPath: "/tmp/log" }};
        const typingRowRuntime = {{
          snapshot: () => ({{ stats: {{ thinking: 0, thinkingTokens: 0, tools: 0 }} }}),
          updateTypingStats: noop, updateSubagentGauge: noop, resetTypingStats: noop, updateActivityPeek: noop,
        }};
        // The api stub returns a payload marked as served from the offline
        // cache, and the injected apiResponseStale recognizes that mark.
        const staleData = {{ events: [], busy: false, queue_len: 0, token: null, __stale: true }};
        const controller = ctx.window.CodoxearMessageFlow.createMessageFlowController({{
          sessionState, sessionCatalog,
          currentGeneration: () => 1,
          isAppDisposed: () => false,
          sessionLaunchFailed: () => false,
          api: async () => staleData,
          apiResponseStale: (data) => Boolean(data && data.__stale),
          resolveAppUrl: (path) => `http://example.test${{path}}`, handleAppAuthLoss: noop,
          refreshSessions: async () => [], openSession: async () => null, clearSelectedSessionAfterRemoval: noop,
          activeTranscriptSnapshot: () => ({{ ...active }}),
          updateSessionTranscriptSlot: () => ({{ ignoredStaleBound: false, current: {{ state: "bound" }} }}),
          renderPendingTranscriptSlot: noop, renderSessionTail: noop, applySessionRuntimeFromTail: noop,
          resetChatRenderState: noop, setAttachCount: noop, setLiveCursor: noop,
          appendEvents: noop, appendTailSnapshotEvents: noop, initPageLimit: () => 60, typingRowRuntime,
          getStagedAttachments: () => [], normalizedStagedAttachments: () => [],
          setSelectedSessionPendingAttachment: noop, confirmAction: async () => false,
          setToast: noop, isTranscriptRenewalCommand: () => false, nextLocalEchoId: () => 1,
          renderedAtLiveTail: () => true, getSessionTranscriptSlot: () => ({{ epoch: 1 }}), addPendingUser: noop,
          deleteTailCache: noop, beginTranscriptRenewal: noop, clearLiveCursor: noop, invalidateOlderLoad: noop,
          dropPendingUser: noop, removePendingUserRow: noop, hasPendingForSession: () => false, restoreUndeliveredDraft: () => false,
          visibilityState: () => "visible", navigatorValue: () => navigatorLike,
          reportTransportSuccess: () => networkStatus.reportSuccess(),
          reportTransportFailure: () => networkStatus.reportFailure(),
          reportTransportStale: () => networkStatus.reportStale(),
          EventSource: null, AbortController: null,
          setTimeout: () => ({{ cleared: false }}), clearTimeout: noop, now: () => 1000,
          consoleWarn: noop, consoleError: noop,
        }});

        Promise.resolve(controller.pollMessages("sid", 1)).then(() => {{
          const afterStalePoll = {{ text: banner.textContent, hidden: banner.hidden }};
          process.stdout.write(JSON.stringify({{ afterStalePoll }}));
        }}).catch((error) => {{ console.error(error); process.exit(1); }});
        """
    )
    out = run_node_json(script)
    assert out["afterStalePoll"]["hidden"] is False
    assert "cached data" in out["afterStalePoll"]["text"]
