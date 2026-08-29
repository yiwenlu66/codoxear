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
                      getThinkingLevel() { return thinking; },
                      setThinkingLevel(level) { thinking = level; },
                    };
                    const header = { type: "session", id: "session-id", cwd: "/tmp", timestamp: "2026-01-01T00:00:00.000Z" };
                    const newSessionPath = process.env.NEW_SESSION_PATH;
                    // Mirror Pi's real runtime (verified against 0.82.1): the
                    // event ctx exposes the live model as a `model` property;
                    // neither pi nor ctx has a getModel function.
                    const makeCtx = (sessionFile, fileEntries = [header]) => ({
                      sessionManager: {
                        getSessionFile() { return sessionFile; },
                        getSessionId() { return "session-id"; },
                        getCwd() { return "/tmp"; },
                        fileEntries,
                        flushed: false,
                      },
                      ui: { notify() {} },
                      get model() { return model; },
                    });
                    const ctx = makeCtx("/tmp/session.jsonl");
                    const newCtx = makeCtx(newSessionPath);
                    const resumePath = `${newSessionPath}.resume`;
                    const resumeCtx = makeCtx(resumePath);
                    const forkPath = `${newSessionPath}.fork`;
                    const forkCtx = makeCtx(forkPath);
                    const invalidPath = `${newSessionPath}.invalid`;
                    const invalidCtx = makeCtx(invalidPath, [{ type: "message" }]);
                    const capsPath = `${process.env.CODEX_WEB_PI_ACTIVE_SESSION_FILE}.caps`;
                    const readCaps = () => JSON.parse(fs.readFileSync(capsPath, "utf8"));

                    bridge(pi);
                    const commandCallsAtLoad = commandCalls;
                    handlers.get("session_start")({ type: "session_start", reason: "startup" }, ctx);
                    const initialMaterialized = fs.existsSync(newSessionPath);
                    handlers.get("session_start")({ type: "session_start", reason: "new" }, newCtx);
                    const newSessionRows = fs.readFileSync(newSessionPath, "utf8").trim().split("\\n").map(JSON.parse);
                    handlers.get("session_start")({ type: "session_start", reason: "new" }, newCtx);
                    handlers.get("session_start")({ type: "session_start", reason: "resume" }, resumeCtx);
                    handlers.get("session_start")({ type: "session_start", reason: "fork" }, forkCtx);
                    handlers.get("session_start")({ type: "session_start", reason: "new" }, invalidCtx);
                    handlers.get("turn_end")({ type: "turn_end", turnIndex: 1 }, ctx);
                    const afterRegistration = readCaps();
                    thinking = "high";
                    handlers.get("turn_end")({ type: "turn_end", turnIndex: 2 }, ctx);
                    const afterSettingsChange = readCaps();
                    model = { provider: "anthropic", id: "claude-opus-4-6" };
                    handlers.get("model_select")({ type: "model_select" }, ctx);
                    const afterModelSelect = readCaps();
                    const mode = fs.statSync(capsPath).mode & 0o777;
                    process.stdout.write(JSON.stringify({
                      commandCallsAtLoad,
                      registrationCalls,
                      afterRegistration,
                      afterSettingsChange,
                      afterModelSelect,
                      mode,
                      initialMaterialized,
                      newSessionRows,
                      newSessionFlushed: newCtx.sessionManager.flushed,
                      resumeMaterialized: fs.existsSync(resumePath),
                      forkMaterialized: fs.existsSync(forkPath),
                      invalidMaterialized: fs.existsSync(invalidPath),
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
                env={
                    "CODEX_WEB_PI_ACTIVE_SESSION_FILE": str(marker),
                    "NEW_SESSION_PATH": str(temp_dir / "new-session.jsonl"),
                },
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
        self.assertEqual(observed["afterRegistration"]["model"], "gpt-5")
        self.assertEqual(observed["afterRegistration"]["model_provider"], "openai")
        self.assertEqual(observed["afterSettingsChange"]["reasoning_effort"], "high")
        self.assertEqual(observed["afterSettingsChange"]["model"], "gpt-5")
        self.assertEqual(observed["afterModelSelect"]["model"], "claude-opus-4-6")
        self.assertEqual(observed["afterModelSelect"]["model_provider"], "anthropic")
        self.assertEqual(observed["afterModelSelect"]["reasoning_effort"], "high")
        self.assertEqual(observed["afterSettingsChange"]["bridgeVersion"], 2)
        self.assertEqual(observed["afterSettingsChange"]["features"], ["effort", "thinking"])
        self.assertEqual(observed["mode"], 0o600)
        self.assertFalse(observed["initialMaterialized"])
        self.assertEqual(
            observed["newSessionRows"],
            [{"type": "session", "id": "session-id", "cwd": "/tmp", "timestamp": "2026-01-01T00:00:00.000Z"}],
        )
        self.assertTrue(observed["newSessionFlushed"])
        self.assertFalse(observed["resumeMaterialized"])
        self.assertFalse(observed["forkMaterialized"])
        self.assertFalse(observed["invalidMaterialized"])


if __name__ == "__main__":
    unittest.main()
