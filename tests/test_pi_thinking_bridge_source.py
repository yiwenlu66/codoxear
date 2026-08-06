import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "codoxear" / "pi_active_session_bridge.ts"


class TestPiThinkingBridgeSource(unittest.TestCase):
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
                  .replace(/function writeThinkingCapabilities\\(commands\\)/, "function writeThinkingCapabilities(commands)")
                  .replace(/function refreshCaps\\(commands\\)/, "function refreshCaps(commands)")
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
