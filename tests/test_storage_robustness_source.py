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
    def test_storage_access_is_wrapped(self) -> None:
        app_source = APP_JS.read_text(encoding="utf-8")
        storage_source = APP_STORAGE_JS.read_text(encoding="utf-8")
        self.assertIn("function optionalLocalStorage() {", storage_source)
        self.assertIn("function getItem(key) {", storage_source)
        self.assertIn("function setItem(key, value) {", storage_source)
        self.assertIn("function removeItem(key) {", storage_source)
        self.assertIn("window.CodoxearStorage = Object.freeze", storage_source)
        self.assertIn("const codoxearStorage = window.CodoxearStorage;", app_source)
        self.assertIn('throw new Error("Codoxear storage helpers failed to load")', app_source)
        self.assertIn("function storageGetItem(key) {", app_source)
        self.assertIn("function storageSetItem(key, value) {", app_source)
        self.assertIn("function storageRemoveItem(key) {", app_source)
        direct_calls = re.findall(r"(?<!window\.)\blocalStorage\.(?:getItem|setItem|removeItem)\(", app_source)
        self.assertEqual(direct_calls, [])
        self.assertIn('storageGetItem("codexweb.selected")', app_source)
        self.assertIn('storageSetItem("codexweb.selected", sessionId)', app_source)
        self.assertIn('storageRemoveItem("codexweb.selected")', app_source)

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
