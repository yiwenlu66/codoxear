import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_POLLING_JS = ROOT / "codoxear" / "static" / "app_polling.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def eval_polling_policy() -> dict:
    source = APP_POLLING_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const helpers = ctx.window.CodoxearPolling;
        const idle = helpers.messagePollDelayMs({{ now: 1000 }});
        const running = helpers.messagePollDelayMs({{ now: 1000, turnOpen: true }});
        const fast = helpers.messagePollDelayMs({{ now: 1000, pollFastUntilMs: 2000 }});
        const hidden = helpers.messagePollDelayMs({{ now: 1000, visibilityState: "hidden" }});
        const offline = helpers.messagePollDelayMs({{ now: 1000, offline: true }});
        const error1 = helpers.messagePollDelayMs({{ now: 1000, errorStreak: 1 }});
        const errorKick0 = helpers.normalizeMessagePollKickDelay({{ requested: 0, errorStreak: 1 }});
        const error2 = helpers.messagePollDelayMs({{ now: 1000, errorStreak: 2 }});
        const offlineHighError = helpers.messagePollDelayMs({{ now: 1000, offline: true, errorStreak: 7 }});
        const offlineHighErrorKick0 = helpers.normalizeMessagePollKickDelay({{ requested: 0, offline: true, errorStreak: 7 }});
        const recovered = helpers.messagePollDelayMs({{ now: 1000, errorStreak: 0 }});
        process.stdout.write(JSON.stringify({{
          intervals: helpers.POLLING_INTERVALS,
          sessionsVisible: helpers.sessionsPollDelayMs("visible"),
          sessionsHidden: helpers.sessionsPollDelayMs("hidden"),
          secondaryVisible: helpers.secondaryPollDelayMs("visible"),
          secondaryHidden: helpers.secondaryPollDelayMs("hidden"),
          offlineUndefined: helpers.browserOffline(undefined),
          offlineTrue: helpers.browserOffline({{ onLine: false }}),
          offlineFalse: helpers.browserOffline({{ onLine: true }}),
          idle,
          running,
          fast,
          hidden,
          offline,
          error1,
          errorKick0,
          error2,
          offlineHighError,
          offlineHighErrorKick0,
          recovered,
          negativeKick: helpers.normalizeMessagePollKickDelay({{ requested: -5 }}),
          stringKick: helpers.normalizeMessagePollKickDelay({{ requested: "42" }}),
          positiveKick: helpers.normalizeMessagePollKickDelay({{ requested: 5000 }}),
          frozen: Object.isFrozen(helpers),
          intervalsFrozen: Object.isFrozen(helpers.POLLING_INTERVALS),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def run_app_polling_guard(setup_js: str = "") -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("const codoxearPolling = window.CodoxearPolling;")
    end = source.index("const codoxearConversationCopy = window.CodoxearConversationCopy;", start)
    guard_source = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        try {{
          vm.runInContext({json.dumps(setup_js + "\n" + guard_source)}, ctx);
          process.stdout.write(JSON.stringify({{ ok: true, message: "" }}));
        }} catch (err) {{
          process.stdout.write(JSON.stringify({{ ok: false, message: String(err && err.message || err) }}));
        }}
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendPollingModuleSource(unittest.TestCase):
    def test_app_polling_guard_throws_for_missing_or_partial_helper(self) -> None:
        missing = run_app_polling_guard()
        self.assertEqual(missing, {"ok": False, "message": "Codoxear polling helpers failed to load"})
        partial = run_app_polling_guard(
            "window.CodoxearPolling = { POLLING_INTERVALS: {}, sessionsPollDelayMs() {}, secondaryPollDelayMs() {}, browserOffline() {}, messagePollErrorDelayMs() {}, messagePollDelayMs() {} };"
        )
        self.assertEqual(partial, {"ok": False, "message": "Codoxear polling helpers failed to load"})
        complete = run_app_polling_guard(
            "window.CodoxearPolling = { POLLING_INTERVALS: {}, sessionsPollDelayMs() {}, secondaryPollDelayMs() {}, browserOffline() {}, messagePollErrorDelayMs() {}, messagePollDelayMs() {}, normalizeMessagePollKickDelay() {} };"
        )
        self.assertEqual(complete, {"ok": True, "message": ""})

    def test_polling_policy_preserves_delay_contracts(self) -> None:
        result = eval_polling_policy()
        self.assertEqual(result["intervals"], {
            "SESSION_POLL_VISIBLE_MS": 5000,
            "SESSION_POLL_HIDDEN_MS": 15000,
            "SECONDARY_POLL_VISIBLE_MS": 30000,
            "SECONDARY_POLL_HIDDEN_MS": 60000,
            "MESSAGE_POLL_FAST_MS": 500,
            "MESSAGE_POLL_RUNNING_MS": 800,
            "MESSAGE_POLL_IDLE_MS": 2500,
            "MESSAGE_POLL_HIDDEN_MS": 5000,
            "MESSAGE_POLL_OFFLINE_MS": 15000,
            "MESSAGE_POLL_ERROR_MIN_MS": 2000,
            "MESSAGE_POLL_ERROR_MAX_MS": 30000,
        })
        self.assertEqual(result["sessionsVisible"], 5000)
        self.assertEqual(result["sessionsHidden"], 15000)
        self.assertEqual(result["secondaryVisible"], 30000)
        self.assertEqual(result["secondaryHidden"], 60000)
        self.assertFalse(result["offlineUndefined"])
        self.assertTrue(result["offlineTrue"])
        self.assertFalse(result["offlineFalse"])
        self.assertEqual(result["idle"], 2500)
        self.assertEqual(result["running"], 800)
        self.assertEqual(result["fast"], 500)
        self.assertEqual(result["hidden"], 5000)
        self.assertEqual(result["offline"], 15000)
        self.assertEqual(result["error1"], 2500)
        self.assertEqual(result["errorKick0"], 2000)
        self.assertEqual(result["error2"], 4000)
        self.assertEqual(result["offlineHighError"], 30000)
        self.assertEqual(result["offlineHighErrorKick0"], 30000)
        self.assertEqual(result["recovered"], 2500)
        self.assertEqual(result["negativeKick"], 0)
        self.assertEqual(result["stringKick"], 42)
        self.assertEqual(result["positiveKick"], 5000)
        self.assertTrue(result["frozen"])
        self.assertTrue(result["intervalsFrozen"])


if __name__ == "__main__":
    unittest.main()
