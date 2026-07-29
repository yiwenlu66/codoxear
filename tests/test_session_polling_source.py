import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLLING = ROOT / "codoxear" / "static" / "app_polling.js"
API = ROOT / "codoxear" / "static" / "app_api.js"


def eval_polling_and_cache() -> dict:
    sources = [POLLING.read_text(encoding="utf-8"), API.read_text(encoding="utf-8")]
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const requests = []; let responseNo = 0;
        const ctx = {{ window: {{ CodoxearUrls: {{resolveAppUrl:p=>"http://host"+p}}, CodoxearPerf: {{pushSample:()=>{{}}}} }}, performance: {{now:()=>1}},
          fetch: async (url, options) => {{ requests.push([url, options.headers["If-None-Match"] || null]); responseNo += 1;
            const notModified=responseNo===2; return {{status:notModified?304:200, ok:true, headers:{{get:()=>notModified?null:'"sessions-v1"'}}, text:async()=>JSON.stringify({{sessions:["one"]}})}}; }} }};
        vm.createContext(ctx); for (const source of {json.dumps(sources)}) vm.runInContext(source,ctx);
        (async()=>{{ const polling=ctx.window.CodoxearPolling, api=ctx.window.CodoxearApi;
          const first=await api.api("/api/sessions"); const second=await api.api("/api/sessions");
          process.stdout.write(JSON.stringify({{sessionsVisible:polling.sessionsPollDelayMs("visible"), sessionsHidden:polling.sessionsPollDelayMs("hidden"), idle:polling.messagePollDelayMs({{now:1}}), running:polling.messagePollDelayMs({{now:1,turnOpen:true}}), fast:polling.messagePollDelayMs({{now:1,pollFastUntilMs:2}}), hidden:polling.messagePollDelayMs({{now:1,visibilityState:"hidden"}}), offline:polling.messagePollDelayMs({{now:1,offline:true,errorStreak:7}}), errorKick:polling.normalizeMessagePollKickDelay({{requested:0,errorStreak:2}}), requests, first, second, cached:api.apiResponseNotModified(second)}})); }})().catch(e=>{{console.error(e);process.exit(1)}});
        """
    )
    proc = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    return json.loads(proc.stdout)


class TestSessionPollingBehavior(unittest.TestCase):
    def test_visibility_turn_and_errors_choose_the_correct_poll_delay(self) -> None:
        result = eval_polling_and_cache()
        self.assertEqual(result["sessionsVisible"], 2500)
        self.assertEqual(result["sessionsHidden"], 15000)
        self.assertEqual(result["idle"], 900)
        self.assertEqual(result["running"], 250)
        self.assertEqual(result["fast"], 200)
        self.assertEqual(result["hidden"], 5000)
        self.assertEqual(result["offline"], 30000)
        self.assertEqual(result["errorKick"], 4000)

    def test_sessions_poll_reuses_etag_payload_on_not_modified_response(self) -> None:
        result = eval_polling_and_cache()
        self.assertEqual(result["requests"], [["http://host/api/sessions", None], ["http://host/api/sessions", '"sessions-v1"']])
        self.assertEqual(result["first"], {"sessions": ["one"]})
        self.assertEqual(result["second"], {"sessions": ["one"]})
        self.assertTrue(result["cached"])


if __name__ == "__main__":
    unittest.main()
