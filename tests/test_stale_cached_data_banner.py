"""Behavior tests for stale-cache visibility: when the service worker answers
API polls from its offline cache, the connection indicator must stay up
instead of rendering stale content as if it were fresh.

Chain under test:
  service-worker marks fallback responses (X-Codoxear-Stale)
  -> app_api surfaces the mark (apiResponseStale, non-enumerable) AND reports it
     to the connection indicator as transport evidence
  -> the icon stays visible and its spoken text names cached data.
"""
from frontend_module_loader import NETWORK_STATUS_STUB_JS, module_path
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


def test_network_status_stale_state() -> None:
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(APP_NETWORK_JS.read_text(encoding="utf-8"))}, ctx);
{NETWORK_STATUS_STUB_JS}
        const navigatorLike = {{ onLine: true }};
        const net = ctx.window.CodoxearNetwork.createNetworkStatusController({{ status, statusText, navigatorLike }});

        const initial = netSnap(net.sync());
        const afterStale = netSnap(net.reportStale());
        // One stale answer is proof of unreachability: no failure threshold.
        const afterSuccess = netSnap(net.reportSuccess());

        net.reportStale();
        navigatorLike.onLine = false;
        const whileOffline = netSnap(net.sync());
        navigatorLike.onLine = true;
        const backOnline = netSnap(net.sync());
        const afterRecovery = netSnap(net.reportSuccess());

        process.stdout.write(JSON.stringify({{ initial, afterStale, afterSuccess, whileOffline, backOnline, afterRecovery }}));
        """
    )
    out = run_node_json(script)
    assert out["initial"]["hidden"] is True
    assert out["afterStale"]["state"] == "cached"
    assert out["afterStale"]["hidden"] is False
    assert "showing cached data" in out["afterStale"]["spoken"]
    assert out["afterSuccess"]["hidden"] is True
    assert out["afterSuccess"]["state"] == ""
    # The browser-offline signal is the stronger statement and wins while it holds.
    assert out["whileOffline"]["state"] == "offline"
    # Connectivity returning does not erase the fact that the server is still
    # unreachable: the cached reason persists until a live response arrives.
    assert out["backOnline"]["state"] == "cached"
    assert out["afterRecovery"]["hidden"] is True


def test_api_responses_report_client_transport_state() -> None:
    # The indicator's evidence must come from the transport layer, not from
    # whichever session's poll happened to run: a session switch, a file read,
    # or a send has to update it as promptly as a poll tick does.
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        let responses = [];
        const ctx = {{
          URL, console, JSON,
          performance: {{ now: () => 0 }},
          fetch: async () => {{
            const next = responses.shift();
            if (!next) throw new TypeError("no response queued");
            if (next.kind === "throw") throw new TypeError("network down");
            if (next.kind === "abort") throw Object.assign(new Error("aborted"), {{ name: "AbortError" }});
            const status = next.status || 200;
            return {{
              status,
              ok: status < 400,
              headers: {{ get(name) {{
                return String(name).toLowerCase() === "x-codoxear-stale" && next.stale ? "1" : null;
              }} }},
              text: async () => next.body || "{{}}",
            }};
          }},
          window: {{
            location: {{ href: "http://localhost/", origin: "http://localhost", pathname: "/" }},
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(APP_APPLICATION_JS.read_text(encoding="utf-8"))}, ctx);
        {NETWORK_STATUS_STUB_JS}
        vm.runInContext({json.dumps(APP_NETWORK_JS.read_text(encoding="utf-8"))}, ctx);
        vm.runInContext({json.dumps(APP_API_JS.read_text(encoding="utf-8"))}, ctx);

        const navigatorLike = {{ onLine: true }};
        const net = ctx.window.CodoxearNetwork.createNetworkStatusController({{ status, statusText, navigatorLike }});
        const api = ctx.window.CodoxearApi.api;
        ctx.window.CodoxearApi.setTransportReporter((kind) => {{
          if (kind === "stale") net.reportStale();
          else if (kind === "failed") net.reportFailure();
          else net.reportSuccess();
        }});

        const call = (response) => {{
          responses = [response];
          return api("/api/sessions").catch(() => null);
        }};

        (async () => {{
          const out = {{ initial: netSnap(net.sync()) }};
          await call({{ body: '{{"sessions":[]}}' }});
          out.afterLive = netSnap(net.sync());
          await call({{ body: '{{"sessions":[]}}', stale: true }});
          out.afterStale = netSnap(net.sync());
          await call({{ body: '{{"sessions":[]}}' }});
          out.afterLiveAgain = netSnap(net.sync());
          await call({{ kind: "throw" }});
          out.afterOneFailure = netSnap(net.sync());
          await call({{ kind: "throw" }});
          out.afterTwoFailures = netSnap(net.sync());
          await call({{ kind: "abort" }});
          out.afterAbort = netSnap(net.sync());
          await call({{ status: 500, body: '{{"error":"boom"}}' }});
          out.afterServerError = netSnap(net.sync());
          process.stdout.write(JSON.stringify(out));
        }})().catch((error) => {{ console.error(error); process.exit(1); }});
        """
    )
    out = run_node_json(script)

    assert out["initial"]["state"] == ""
    assert out["afterLive"]["hidden"] is True
    # A cache-served answer is not transport health, whatever its status line says.
    assert out["afterStale"]["state"] == "cached"
    assert out["afterStale"]["hidden"] is False
    assert "showing cached data" in out["afterStale"]["spoken"]
    # One live response clears it, with no dependence on a poll cycle.
    assert out["afterLiveAgain"]["state"] == ""
    assert out["afterLiveAgain"]["hidden"] is True
    # One failed request is below the threshold; two in a row trips it.
    assert out["afterOneFailure"]["state"] == ""
    assert out["afterTwoFailures"]["state"] == "failing"
    # A caller's own cancellation is not evidence about the connection.
    assert out["afterAbort"]["state"] == "failing"
    # An error response still proves the server answered.
    assert out["afterServerError"]["state"] == ""
