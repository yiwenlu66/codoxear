import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_PERF_JS = ROOT / "codoxear" / "static" / "app_perf.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def eval_perf_module() -> dict:
    source = APP_PERF_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const perf = ctx.window.CodoxearPerf;
        perf.pushSample("api", -1);
        perf.pushSample("api", Number.NaN);
        for (let i = 1; i <= 205; i += 1) perf.pushSample("api", i);
        perf.pushSample("other", 10.125);
        process.stdout.write(JSON.stringify({{ summary: perf.summarize() }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendPerfModuleSource(unittest.TestCase):
    def test_perf_module_preserves_summary_policy(self) -> None:
        result = eval_perf_module()["summary"]
        api = result["api"]
        self.assertEqual(api["count"], 200)
        self.assertEqual(api["max_ms"], 205)
        self.assertEqual(api["last_ms"], 205)
        self.assertEqual(api["p50_ms"], 105.5)
        self.assertEqual(api["p95_ms"], 195.05)
        self.assertEqual(result["other"], {"count": 1, "p50_ms": 10.13, "p95_ms": 10.13, "max_ms": 10.13, "last_ms": 10.13})


if __name__ == "__main__":
    unittest.main()
