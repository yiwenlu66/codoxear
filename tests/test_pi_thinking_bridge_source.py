import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "codoxear" / "pi_active_session_bridge.ts"


class TestPiThinkingBridgeSource(unittest.TestCase):
    def test_thinking_command_uses_pi_setter_and_reports_effective_level(self) -> None:
        source = BRIDGE.read_text(encoding="utf-8")

        self.assertIn('pi.registerCommand("effort"', source)
        self.assertIn('pi.registerCommand("thinking"', source)
        self.assertIn("setThinkingLevel(level: ThinkingLevel): void", source)
        self.assertIn("getThinkingLevel(): ThinkingLevel", source)
        self.assertIn("pi.setThinkingLevel(requested as ThinkingLevel)", source)
        self.assertIn("const effective = pi.getThinkingLevel()", source)
        self.assertIn("adjusted for the current model", source)
        self.assertIn("Choose one of: ${THINKING_LEVELS.join", source)
        self.assertIn('ctx.ui.notify(message, "info")', source)

    def test_marker_writes_remain_guarded_and_registered(self) -> None:
        source = BRIDGE.read_text(encoding="utf-8")

        self.assertIn("if (!markerPath) return", source)
        self.assertIn("bridgeVersion: 2", source)
        self.assertIn("sessionFile,", source)
        self.assertIn("fs.renameSync(tmp, markerPath)", source)
        self.assertIn('pi.on("session_start"', source)
        self.assertIn('pi.on("session_switch"', source)
        self.assertIn('pi.on("session_fork"', source)

    def test_load_writes_thinking_capability_file_for_current_process(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / "active.json"
            script = textwrap.dedent(
                f"""
                const fs = require("node:fs");
                const vm = require("node:vm");
                let source = fs.readFileSync({json.dumps(str(BRIDGE))}, "utf8");
                source = source
                  .replace(/type SessionManager[\\s\\S]*?const THINKING_LEVELS/, "const THINKING_LEVELS")
                  .replace(/: readonly ThinkingLevel\\[\\]/, "")
                  .replace(/function writeActiveSession\\(ctx: ExtensionContext, reason: string\\): void/, "function writeActiveSession(ctx, reason)")
                  .replace(/function writeThinkingCapabilities\\(\\): void/, "function writeThinkingCapabilities()")
                  .replace(/export default function \\(pi: ExtensionAPI\\): void/, "function bridge(pi)")
                  .replace(/ as ThinkingLevel/g, "");
                const context = {{ require, process, console }};
                vm.createContext(context);
                vm.runInContext(`${{source}}; globalThis.bridge = bridge;`, context);
                context.bridge({{ on() {{}}, registerCommand() {{}} }});
                const capsPath = `${{process.env.CODEX_WEB_PI_ACTIVE_SESSION_FILE}}.caps`;
                const stat = fs.statSync(capsPath);
                process.stdout.write(JSON.stringify({{ payload: JSON.parse(fs.readFileSync(capsPath, "utf8")), mode: stat.mode & 0o777, pid: process.pid }}));
                """
            )
            result = subprocess.run(
                ["node", "-e", script],
                check=False,
                capture_output=True,
                text=True,
                env={"CODEX_WEB_PI_ACTIVE_SESSION_FILE": str(marker)},
            )
            if result.returncode:
                raise AssertionError(result.stderr or result.stdout)
            observed = json.loads(result.stdout)

        self.assertEqual(observed["payload"]["bridgeVersion"], 2)
        self.assertEqual(observed["payload"]["features"], ["effort", "thinking"])
        self.assertEqual(observed["payload"]["pid"], observed["pid"])
        self.assertIn("updatedAt", observed["payload"])
        self.assertEqual(observed["mode"], 0o600)


if __name__ == "__main__":
    unittest.main()
