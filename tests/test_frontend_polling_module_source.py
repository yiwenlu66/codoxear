from frontend_module_loader import module_path
import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = module_path("app_application.js")
APP_POLLING_JS = module_path("app_polling.js")
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
        const timers = [];
        const canceled = [];
        const runtime = helpers.createPollingRuntime({{
          setTimeout(callback, delay) {{ const timer = {{ callback, delay }}; timers.push(timer); return timer; }},
          clearTimeout(timer) {{ canceled.push(timer); }},
        }});
        const sessionsTick = () => "sessions";
        const secondaryTick = () => "secondary";
        runtime.scheduleSessions(12, sessionsTick);
        runtime.scheduleSessions(24, sessionsTick);
        runtime.scheduleSecondary(36, secondaryTick);
        runtime.markSessionsPollFailure();
        runtime.markSessionsPollFailure();
        runtime.markSecondaryPollFailure();
        const beforeReset = {{ sessions: runtime.sessionsPollErrorStreak(), secondary: runtime.secondaryPollErrorStreak() }};
        runtime.resetStreaks();
        const initialGeneration = runtime.currentGeneration();
        const nextGeneration = runtime.nextGeneration();
        runtime.incrementGeneration();
        runtime.disable();
        runtime.scheduleSessions(48, sessionsTick);
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
        const sessionsOffline = helpers.networkRetryDelayMs({{ normalDelayMs: helpers.sessionsPollDelayMs("visible"), offline: true }});
        const secondaryOffline = helpers.networkRetryDelayMs({{ normalDelayMs: helpers.secondaryPollDelayMs("hidden"), offline: true }});
        const sessionsFailed = helpers.networkRetryDelayMs({{ normalDelayMs: helpers.sessionsPollDelayMs("visible"), errorStreak: 2 }});
        const recovered = helpers.messagePollDelayMs({{ now: 1000, errorStreak: 0 }});
        process.stdout.write(JSON.stringify({{
          intervals: helpers.POLLING_INTERVALS,
          runtime: {{
            initialGeneration,
            nextGeneration,
            currentGeneration: runtime.currentGeneration(),
            beforeReset,
            afterReset: {{ sessions: runtime.sessionsPollErrorStreak(), secondary: runtime.secondaryPollErrorStreak() }},
            canceled: canceled.length,
            scheduled: timers.map((timer) => timer.delay),
            frozen: Object.isFrozen(runtime),
          }},
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
          sessionsOffline,
          secondaryOffline,
          sessionsFailed,
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


class TestFrontendPollingModuleSource(unittest.TestCase):
    def test_polling_policy_preserves_delay_contracts(self) -> None:
        result = eval_polling_policy()
        self.assertEqual(result["intervals"], {
            "SESSION_POLL_VISIBLE_MS": 5000,
            "SESSION_POLL_HIDDEN_MS": 15000,
            "SECONDARY_POLL_VISIBLE_MS": 30000,
            "SECONDARY_POLL_HIDDEN_MS": 60000,
            "MESSAGE_POLL_FAST_MS": 300,
            "MESSAGE_POLL_RUNNING_MS": 500,
            "MESSAGE_POLL_IDLE_MS": 1500,
            "MESSAGE_POLL_HIDDEN_MS": 5000,
            "MESSAGE_POLL_OFFLINE_MS": 15000,
            "MESSAGE_POLL_ERROR_MIN_MS": 2000,
            "MESSAGE_POLL_ERROR_MAX_MS": 30000,
        })
        self.assertEqual(result["runtime"], {
            "initialGeneration": 0,
            "nextGeneration": 1,
            "currentGeneration": 2,
            "beforeReset": {"sessions": 2, "secondary": 1},
            "afterReset": {"sessions": 0, "secondary": 0},
            "canceled": 3,
            "scheduled": [12, 24, 36],
            "frozen": True,
        })
        self.assertEqual(result["sessionsVisible"], 5000)
        self.assertEqual(result["sessionsHidden"], 15000)
        self.assertEqual(result["secondaryVisible"], 30000)
        self.assertEqual(result["secondaryHidden"], 60000)
        self.assertFalse(result["offlineUndefined"])
        self.assertTrue(result["offlineTrue"])
        self.assertFalse(result["offlineFalse"])
        self.assertEqual(result["idle"], 1500)
        self.assertEqual(result["running"], 500)
        self.assertEqual(result["fast"], 300)
        self.assertEqual(result["hidden"], 5000)
        self.assertEqual(result["offline"], 15000)
        self.assertEqual(result["error1"], 2000)
        self.assertEqual(result["errorKick0"], 2000)
        self.assertEqual(result["error2"], 4000)
        self.assertEqual(result["offlineHighError"], 30000)
        self.assertEqual(result["offlineHighErrorKick0"], 30000)
        self.assertEqual(result["sessionsOffline"], 15000)
        self.assertEqual(result["secondaryOffline"], 60000)
        self.assertEqual(result["sessionsFailed"], 5000)
        self.assertEqual(result["recovered"], 1500)
        self.assertEqual(result["negativeKick"], 0)
        self.assertEqual(result["stringKick"], 42)
        self.assertEqual(result["positiveKick"], 5000)
        self.assertTrue(result["frozen"])
        self.assertTrue(result["intervalsFrozen"])


if __name__ == "__main__":
    unittest.main()
