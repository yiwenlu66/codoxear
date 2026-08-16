import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "codoxear" / "pi_active_session_bridge.ts"


class TestPiThinkingBridgeLifecycle(unittest.TestCase):
    def test_lifecycle_registration_and_caps_snapshot_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            marker = temp_dir / "active.json"
            bundle = temp_dir / "pi_active_session_bridge.cjs"
            build = subprocess.run(
                [
                    "npx",
                    "esbuild",
                    str(BRIDGE),
                    "--bundle",
                    "--platform=node",
                    "--format=cjs",
                    f"--outfile={bundle}",
                    "--log-level=error",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            if build.returncode:
                raise AssertionError(build.stderr or build.stdout)

            script = temp_dir / "exercise_bridge.cjs"
            script.write_text(
                textwrap.dedent(
                    """
                    const fs = require("node:fs");
                    const bridgeModule = require(process.argv[2]);
                    const bridge = bridgeModule.default || bridgeModule;
                    const handlers = new Map();
                    const registered = new Map();
                    let failFirstRegistration = true;
                    let registrationCalls = 0;
                    let commandCalls = 0;
                    let model = { provider: "openai", id: "gpt-5" };
                    let thinking = "low";
                    const pi = {
                      on(event, handler) { handlers.set(event, handler); },
                      registerCommand(name, options) {
                        registrationCalls += 1;
                        if (failFirstRegistration) {
                          failFirstRegistration = false;
                          throw new Error("runtime still binding");
                        }
                        registered.set(name, options);
                      },
                      getCommands() {
                        commandCalls += 1;
                        return [
                          { name: "prompt-template", description: "Template" },
                          ...Array.from(registered, ([name, options]) => ({ name, description: options.description })),
                        ];
                      },
                      getModel() { return model; },
                      getThinkingLevel() { return thinking; },
                      setThinkingLevel(level) { thinking = level; },
                    };
                    const ctx = {
                      sessionManager: {
                        getSessionFile() { return "/tmp/session.jsonl"; },
                        getSessionId() { return "session-id"; },
                        getCwd() { return "/tmp"; },
                      },
                      ui: { notify() {} },
                    };
                    const capsPath = `${process.env.CODEX_WEB_PI_ACTIVE_SESSION_FILE}.caps`;
                    const readCaps = () => JSON.parse(fs.readFileSync(capsPath, "utf8"));

                    bridge(pi);
                    const commandCallsAtLoad = commandCalls;
                    handlers.get("session_start")({ type: "session_start" }, ctx);
                    handlers.get("turn_end")({ type: "turn_end", turnIndex: 1 });
                    const afterRegistration = readCaps();
                    thinking = "high";
                    handlers.get("turn_end")({ type: "turn_end", turnIndex: 2 });
                    const afterSettingsChange = readCaps();
                    const mode = fs.statSync(capsPath).mode & 0o777;
                    process.stdout.write(JSON.stringify({
                      commandCallsAtLoad,
                      registrationCalls,
                      afterRegistration,
                      afterSettingsChange,
                      mode,
                    }));
                    """
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                ["node", str(script), str(bundle)],
                check=False,
                capture_output=True,
                text=True,
                env={"CODEX_WEB_PI_ACTIVE_SESSION_FILE": str(marker)},
            )
            if result.returncode:
                raise AssertionError(result.stderr or result.stdout)
            observed = json.loads(result.stdout)

        registered_names = [command["name"] for command in observed["afterRegistration"]["commands"]]
        self.assertEqual(observed["commandCallsAtLoad"], 0)
        self.assertEqual(observed["registrationCalls"], 3)
        self.assertIn("effort", registered_names)
        self.assertIn("thinking", registered_names)
        self.assertEqual(
            observed["afterSettingsChange"]["commands"],
            observed["afterRegistration"]["commands"],
        )
        self.assertEqual(observed["afterSettingsChange"]["reasoning_effort"], "high")
        self.assertEqual(observed["afterSettingsChange"]["bridgeVersion"], 2)
        self.assertEqual(observed["afterSettingsChange"]["features"], ["effort", "thinking"])
        self.assertEqual(observed["mode"], 0o600)


if __name__ == "__main__":
    unittest.main()
