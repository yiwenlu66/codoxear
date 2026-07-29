import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_URL_JS = ROOT / "codoxear" / "static" / "app_url.js"
APP_PERF_JS = ROOT / "codoxear" / "static" / "app_perf.js"
APP_API_JS = ROOT / "codoxear" / "static" / "app_api.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def run_node_json(js: str) -> dict:
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_api_module_real_order() -> dict:
    scripts = [APP_URL_JS, APP_PERF_JS, APP_API_JS]
    js = textwrap.dedent(
        f"""
        const fs = require("fs");
        const vm = require("vm");
        let now = 1000;
        const requests = [];
        const responses = [
          {{ status: 200, ok: true, etag: "etag-1", text: '{{"sessions":[{{"session_id":"s1"}}]}}' }},
          {{ status: 304, ok: true, etag: null, text: '' }},
          {{ status: 200, ok: true, etag: "etag-2", text: '{{"sessions":[{{"session_id":"s2"}}]}}' }},
          {{ status: 200, ok: true, etag: null, text: '{{"messages":[]}}' }},
          {{ status: 200, ok: true, etag: null, text: '{{"messages":[{{"kind":"init"}}]}}' }},
        ];
        const ctx = {{
          URL,
          console,
          performance: {{ now() {{ now += 17; return now; }} }},
          fetch: async (url, opts) => {{
            requests.push({{ url, opts }});
            const item = responses.shift();
            return {{
              status: item.status,
              ok: item.ok,
              headers: {{ get(name) {{ return String(name).toLowerCase() === "etag" ? item.etag : null; }} }},
              text: async () => item.text,
            }};
          }},
          window: {{
            location: {{ href: "http://localhost/codoxear/", origin: "http://localhost", pathname: "/codoxear/" }},
          }},
        }};
        vm.createContext(ctx);
        for (const file of {json.dumps([str(path) for path in scripts])}) {{
          vm.runInContext(fs.readFileSync(file, "utf8"), ctx, {{ filename: file }});
        }}
        const api = ctx.window.CodoxearApi;
        (async () => {{
          const first = await api.api("/api/sessions");
          const second = await api.api("/api/sessions");
          api.clearApiCache();
          const afterClear = await api.api("/api/sessions");
          const pollMessages = await api.api("/api/sessions/s1/messages/live?cursor=abc");
          const initMessages = await api.api("/api/sessions/s1/messages/tail?init=1");
          process.stdout.write(JSON.stringify({{
            first,
            second,
            afterClear,
            pollMessages,
            initMessages,
            secondNotModified: api.apiResponseNotModified(second),
            firstNotModified: api.apiResponseNotModified(first),
            frozen: Object.isFrozen(api),
            requests,
            perf: ctx.window.CodoxearPerf.summarize(),
          }}));
        }})().catch((err) => {{ console.error(err && err.stack ? err.stack : err); process.exit(1); }});
        """
    )
    return run_node_json(js)


def eval_api_error_contract() -> dict:
    scripts = [APP_URL_JS, APP_PERF_JS, APP_API_JS]
    js = textwrap.dedent(
        f"""
        const fs = require("fs");
        const vm = require("vm");
        let now = 2000;
        const consoleErrors = [];
        const responses = [
          {{ status: 418, ok: false, etag: null, text: '{{"error":"teapot","detail":"short"}}' }},
          {{ status: 200, ok: true, etag: null, text: 'not json' }},
        ];
        const ctx = {{
          URL,
          console: {{
            error: (...args) => consoleErrors.push(args.map((arg) => typeof arg === "string" ? arg : JSON.stringify(arg)).join(" ")),
          }},
          performance: {{ now() {{ now += 11; return now; }} }},
          fetch: async (_url, _opts) => {{
            const item = responses.shift();
            return {{
              status: item.status,
              ok: item.ok,
              headers: {{ get(name) {{ return String(name).toLowerCase() === "etag" ? item.etag : null; }} }},
              text: async () => item.text,
            }};
          }},
          window: {{
            location: {{ href: "http://localhost/codoxear/", origin: "http://localhost", pathname: "/codoxear/" }},
          }},
        }};
        vm.createContext(ctx);
        for (const file of {json.dumps([str(path) for path in scripts])}) {{
          vm.runInContext(fs.readFileSync(file, "utf8"), ctx, {{ filename: file }});
        }}
        const api = ctx.window.CodoxearApi;
        (async () => {{
          let nonOk;
          try {{
            await api.api("/api/fail");
          }} catch (err) {{
            nonOk = {{ message: err.message, status: err.status, obj: err.obj }};
          }}
          let invalid;
          try {{
            await api.api("/api/bad-json");
          }} catch (err) {{
            invalid = {{ name: err.name, message: String(err.message || "") }};
          }}
          process.stdout.write(JSON.stringify({{ nonOk, invalid, consoleErrors }}));
        }})().catch((err) => {{ console.error(err && err.stack ? err.stack : err); process.exit(1); }});
        """
    )
    return run_node_json(js)


class TestFrontendApiModuleSource(unittest.TestCase):
    def test_api_module_preserves_etag_perf_and_url_prefix_behavior(self) -> None:
        result = eval_api_module_real_order()
        self.assertEqual(result["first"], {"sessions": [{"session_id": "s1"}]})
        self.assertEqual(result["second"], {"sessions": [{"session_id": "s1"}]})
        self.assertEqual(result["afterClear"], {"sessions": [{"session_id": "s2"}]})
        self.assertEqual(result["pollMessages"], {"messages": []})
        self.assertEqual(result["initMessages"], {"messages": [{"kind": "init"}]})
        self.assertFalse(result["firstNotModified"])
        self.assertTrue(result["secondNotModified"])
        self.assertTrue(result["frozen"])
        self.assertEqual(result["requests"][0]["url"], "http://localhost/codoxear/api/sessions")
        self.assertEqual(result["requests"][1]["opts"]["headers"]["If-None-Match"], "etag-1")
        self.assertNotIn("If-None-Match", result["requests"][2]["opts"]["headers"])
        self.assertEqual(result["requests"][3]["url"], "http://localhost/codoxear/api/sessions/s1/messages/live?cursor=abc")
        self.assertEqual(result["requests"][4]["url"], "http://localhost/codoxear/api/sessions/s1/messages/tail?init=1")
        self.assertEqual(result["perf"]["api_sessions_ms"]["count"], 3)
        self.assertEqual(result["perf"]["api_messages_poll_ms"]["count"], 1)
        self.assertEqual(result["perf"]["api_messages_init_ms"]["count"], 1)

    def test_api_module_preserves_error_contract(self) -> None:
        result = eval_api_error_contract()
        self.assertEqual(result["nonOk"], {"message": "teapot", "status": 418, "obj": {"error": "teapot", "detail": "short"}})
        self.assertEqual(result["invalid"]["name"], "SyntaxError")
        self.assertIn("api: invalid json response", result["consoleErrors"][0])
        self.assertIn("/api/bad-json", result["consoleErrors"][0])


if __name__ == "__main__":
    unittest.main()
