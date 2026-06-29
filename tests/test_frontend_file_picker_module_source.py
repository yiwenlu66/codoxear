import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DISPLAY_JS = ROOT / "codoxear" / "static" / "app_display.js"
APP_FILE_HELPERS_JS = ROOT / "codoxear" / "static" / "app_file_helpers.js"
APP_FILE_PICKER_JS = ROOT / "codoxear" / "static" / "app_file_picker.js"
APP_JS = ROOT / "codoxear" / "static" / "app.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"
STATIC_ROUTES = ROOT / "codoxear" / "static_routes.py"


def run_picker_module_probe() -> dict[str, object]:
    display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    helper_source = APP_FILE_HELPERS_JS.read_text(encoding="utf-8")
    picker_source = APP_FILE_PICKER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(display_source)}, ctx);
        vm.runInContext({json.dumps(helper_source)}, ctx);
        vm.runInContext({json.dumps(picker_source)}, ctx);
        const picker = ctx.window.CodoxearFilePicker;
        let missingError = "";
        const missingCtx = {{ window: {{}} }};
        vm.createContext(missingCtx);
        try {{
          vm.runInContext({json.dumps(picker_source)}, missingCtx);
        }} catch (err) {{
          missingError = err && err.message ? err.message : String(err);
        }}
        let hostError = "";
        try {{
          picker.createSearchState({{}});
        }} catch (err) {{
          hostError = err && err.message ? err.message : String(err);
        }}
        process.stdout.write(JSON.stringify({{
          frozen: Object.isFrozen(picker),
          exports: Object.keys(picker).sort(),
          missingError,
          hostError,
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendFilePickerModuleSource(unittest.TestCase):
    def test_file_picker_module_exports_and_fails_closed(self) -> None:
        result = run_picker_module_probe()
        self.assertTrue(result["frozen"])
        self.assertEqual(
            result["exports"],
            [
                "createSearchState",
                "localFilePickerSearchEntries",
                "normalizeSamePathFilePickerScores",
                "pendingSessionPathEntry",
                "prependPendingSessionPathEntry",
                "visibleFilePickerEntries",
            ],
        )
        self.assertIn("Codoxear file picker helpers failed to load", result["missingError"])
        self.assertIn("Codoxear file picker host missing blocked", result["hostError"])

    def test_file_picker_module_registered_before_app_js(self) -> None:
        index_source = INDEX_HTML.read_text(encoding="utf-8")
        routes_source = STATIC_ROUTES.read_text(encoding="utf-8")
        app_source = APP_JS.read_text(encoding="utf-8")

        self.assertLess(index_source.index("app_file_helpers.js"), index_source.index("app_file_picker.js"))
        self.assertLess(index_source.index("app_file_picker.js"), index_source.index("app.js"))
        self.assertIn('"app_file_picker.js"', routes_source)
        self.assertIn("const codoxearFilePicker = window.CodoxearFilePicker;", app_source)
        self.assertIn('throw new Error("Codoxear file picker helpers failed to load")', app_source)


if __name__ == "__main__":
    unittest.main()
