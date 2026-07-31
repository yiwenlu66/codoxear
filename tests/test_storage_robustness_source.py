import json
import re
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_STORAGE_JS = ROOT / "codoxear" / "static" / "app_storage.js"


def eval_storage_helpers(storage_expression: str) -> dict:
    source = APP_STORAGE_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        {storage_expression}
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const storage = ctx.window.CodoxearStorage;
        const out = {{}};
        try {{ out.getMissing = storage.getItem("missing"); }} catch (err) {{ out.getMissingError = err && err.name || String(err); }}
        try {{ out.setValue = storage.setItem("k", "v"); }} catch (err) {{ out.setValueError = err && err.name || String(err); }}
        try {{ out.removeValue = storage.removeItem("k"); }} catch (err) {{ out.removeValueError = err && err.name || String(err); }}
        process.stdout.write(JSON.stringify(out));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestStorageRobustnessSource(unittest.TestCase):
    def test_throwing_local_storage_getter_degrades_to_defaults(self) -> None:
        result = eval_storage_helpers(
            """
            Object.defineProperty(ctx.window, "localStorage", {
              get() { throw new DOMException("blocked", "SecurityError"); }
            });
            """
        )
        self.assertEqual(result, {"getMissing": None, "setValue": False, "removeValue": False})

    def test_throwing_storage_methods_do_not_escape(self) -> None:
        result = eval_storage_helpers(
            """
            ctx.window.localStorage = {
              getItem(key) { throw new DOMException("blocked", "SecurityError"); },
              setItem(key, value) { throw new DOMException("full", "QuotaExceededError"); },
              removeItem(key) { throw new DOMException("blocked", "SecurityError"); }
            };
            """
        )
        self.assertEqual(result, {"getMissing": None, "setValue": False, "removeValue": False})


if __name__ == "__main__":
    unittest.main()
