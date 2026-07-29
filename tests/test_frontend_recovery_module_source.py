import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_RECOVERY_JS = ROOT / "codoxear" / "static" / "app_recovery.js"


def run_node_json(js: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    return json.loads(proc.stdout)


class TestFrontendRecoveryModuleBehavior(unittest.TestCase):
    """Lifecycle recovery is a transcript projection, so this asset is inert compatibility state."""

    def test_recovery_asset_loads_as_an_immutable_inert_compatibility_module(self) -> None:
        source = APP_RECOVERY_JS.read_text(encoding="utf-8")
        result = run_node_json(
            textwrap.dedent(
                f"""
                const vm = require("vm");
                const domCalls = [];
                const ctx = {{
                  window: {{}},
                  document: {{
                    createElement() {{ domCalls.push("createElement"); }},
                    querySelector() {{ domCalls.push("querySelector"); }},
                    addEventListener() {{ domCalls.push("addEventListener"); }},
                  }},
                }};
                vm.createContext(ctx);
                vm.runInContext({json.dumps(source)}, ctx);
                const recovery = ctx.window.CodoxearRecovery;
                const mutation = Reflect.set(recovery, "render", () => {{}});
                process.stdout.write(JSON.stringify({{
                  frozen: Object.isFrozen(recovery),
                  keys: Object.keys(recovery),
                  mutation,
                  render: recovery.render || null,
                  domCalls,
                }}));
                """
            )
        )
        self.assertTrue(result["frozen"])
        self.assertEqual(result["keys"], [])
        self.assertFalse(result["mutation"])
        self.assertIsNone(result["render"])
        self.assertEqual(result["domCalls"], [])

if __name__ == "__main__":
    unittest.main()
