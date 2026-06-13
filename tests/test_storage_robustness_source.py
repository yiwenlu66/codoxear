import json
import re
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def eval_storage_helpers(storage_expression: str) -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function optionalLocalStorage() {")
    end = source.index("\n\n      let newSessionBackend", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        {storage_expression}
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        const out = {{}};
        try {{ out.getMissing = ctx.storageGetItem("missing"); }} catch (err) {{ out.getMissingError = err && err.name || String(err); }}
        try {{ out.setValue = ctx.storageSetItem("k", "v"); }} catch (err) {{ out.setValueError = err && err.name || String(err); }}
        try {{ out.removeValue = ctx.storageRemoveItem("k"); }} catch (err) {{ out.removeValueError = err && err.name || String(err); }}
        process.stdout.write(JSON.stringify(out));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestStorageRobustnessSource(unittest.TestCase):
    def test_storage_access_is_wrapped(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function optionalLocalStorage() {", source)
        self.assertIn("function storageGetItem(key) {", source)
        self.assertIn("function storageSetItem(key, value) {", source)
        self.assertIn("function storageRemoveItem(key) {", source)
        direct_calls = re.findall(r"(?<!window\.)\blocalStorage\.(?:getItem|setItem|removeItem)\(", source)
        self.assertEqual(direct_calls, [])
        self.assertIn('storageGetItem("codexweb.selected")', source)
        self.assertIn('storageSetItem("codexweb.selected", sessionId)', source)
        self.assertIn('storageRemoveItem("codexweb.selected")', source)

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
