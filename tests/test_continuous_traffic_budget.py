import json
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLLING_SOURCE = (ROOT / "codoxear" / "static" / "app_polling.js").read_text(encoding="utf-8")
API_SOURCE = (ROOT / "codoxear" / "static" / "app_api.js").read_text(encoding="utf-8")


def run_idle_traffic_window() -> dict:
    """Execute the production poll/API helpers against a deterministic idle server.

    The window begins after HTML/static assets are warm. It includes bootstrap
    API calls, the selected session tail, its SSE connection, and every visible
    poll due before 60 seconds. The fake server returns one stable session-list
    ETag, so later sidebar polls exercise the real 304 cache path.
    """
    script = textwrap.dedent(
        f"""
        const vm = require("node:vm");
        const encoder = new TextEncoder();
        const records = [];
        let sessionVersion = 0;
        const bodies = {{
          "/api/me": {{ ok: true }},
          "/api/sessions": {{ sessions: [{{ session_id: "s1", thread_id: "t1", busy: false, queue_len: 0, token: null }}] }},
          "/api/sessions/s1/messages/tail?limit=60": {{ events: [], live_cursor: "cursor", history_cursor: null, has_older: false, busy: false, queue_len: 0, token: null }},
          "/api/settings/voice": {{ tts_enabled_for_narration: false, audio: {{}}, notifications: {{}}, vapid_public_key: "key", subscriptions: [] }},
          "/api/notifications/feed?since=0": {{ items: [] }},
        }};
        const ctx = {{
          window: {{
            CodoxearUrls: {{ resolveAppUrl: (path) => "http://traffic.test" + path }},
            CodoxearPerf: {{ pushSample() {{}} }},
          }},
          performance: {{ now: () => 1 }},
          fetch: async (url, options = {{}}) => {{
            const path = String(url).replace("http://traffic.test", "");
            const isSessions = path === "/api/sessions";
            const cached = options.headers && options.headers["If-None-Match"];
            const status = isSessions && sessionVersion++ > 0 && cached ? 304 : 200;
            const text = status === 304 ? "" : JSON.stringify(bodies[path] || {{}});
            // Request/response headers, cookies, and framing are budgeted at
            // 512 bytes per exchange, deliberately above this API's normal
            // localhost HTTP/1.1 framing. Response data is measured exactly.
            records.push({{ path, status, wireBytes: 512 + encoder.encode(text).length }});
            return {{
              status,
              ok: status >= 200 && status < 300,
              headers: {{ get: (name) => isSessions && name === "ETag" && status === 200 ? '"sessions-v1"' : null }},
              text: async () => text,
            }};
          }},
          TextEncoder,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(POLLING_SOURCE)}, ctx);
        vm.runInContext({json.dumps(API_SOURCE)}, ctx);
        (async () => {{
          const api = ctx.window.CodoxearApi.api;
          const polling = ctx.window.CodoxearPolling;
          await api("/api/me");
          for (let at = 0; at < 60000; at += polling.sessionsPollDelayMs("visible")) await api("/api/sessions");
          await api("/api/sessions/s1/messages/tail?limit=60");
          // EventSource is not fetch-backed; account for its one HTTP open.
          records.push({{ path: "/api/sessions/s1/live?cursor=cursor", status: 200, wireBytes: 512 }});
          for (let at = 0; at < 60000; at += polling.secondaryPollDelayMs("visible")) {{
            // The aggregate voice response carries subscriptions, so no
            // /api/notifications/subscription request is emitted.
            await api("/api/settings/voice");
            await api("/api/notifications/feed?since=0");
          }}
          process.stdout.write(JSON.stringify({{ records }}));
        }})().catch((error) => {{ console.error(error.stack || error); process.exit(1); }});
        """
    )
    completed = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def test_idle_open_session_stays_below_the_60_second_api_traffic_floor() -> None:
    result = run_idle_traffic_window()
    records = result["records"]
    total_requests = len(records)
    total_wire_bytes = sum(record["wireBytes"] for record in records)
    paths = [record["path"] for record in records]

    assert total_requests == 19
    assert total_requests < 25
    assert total_wire_bytes < 50 * 1024
    assert paths.count("/api/sessions") == 12
    assert sum(record["status"] == 304 for record in records if record["path"] == "/api/sessions") == 11
    assert paths.count("/api/settings/voice") == 2
    assert paths.count("/api/notifications/feed?since=0") == 2
    assert "/api/notifications/subscription" not in paths
    assert paths.count("/api/sessions/s1/messages/tail?limit=60") == 1
    assert paths.count("/api/sessions/s1/live?cursor=cursor") == 1
