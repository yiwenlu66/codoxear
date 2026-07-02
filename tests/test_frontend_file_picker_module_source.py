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
        let menuHostError = "";
        try {{
          picker.createMenuState({{}});
        }} catch (err) {{
          menuHostError = err && err.message ? err.message : String(err);
        }}
        const menu = picker.createMenuState({{
          normalizeLineNumber: (value) => {{
            const n = Number(value);
            return Number.isFinite(n) && n >= 1 ? Math.floor(n) : null;
          }},
        }});
        const opened = menu.openSearchQuery("src/app.js", {{ line: "42", suppressDraft: true }});
        const selectionBeforeInput = menu.selectionLine("src/app.js");
        const draftSuppressedBeforeInput = menu.draftSuppressed("src/app.js");
        menu.setPreserveSearchOnFocus(true);
        const preservedFirst = menu.takePreservedSearchOnFocus();
        const preservedSecond = menu.takePreservedSearchOnFocus();
        menu.handleInput("src/other.js");
        const selectionAfterInput = menu.selectionLine("src/other.js");
        const draftSuppressedAfterInput = menu.draftSuppressed("src/other.js");
        const visibleQueryAfterInput = menu.visibleQuery(" src/other.js ");
        menu.setFocus(10);
        const clampedFocus = menu.clampFocus(2);
        const movedFocus = menu.moveFocus(2, 1);
        const enterIndex = menu.enterIndex();
        const closed = menu.close();
        const domEvents = [];
        function classNode(name) {{
          return {{
            name,
            classes: {{}},
            classList: {{ toggle(className, enabled) {{ domEvents.push(["toggle", name, className, Boolean(enabled)]); if (enabled) this.owner.classes[className] = true; else delete this.owner.classes[className]; }} }},
          }};
        }}
        const field = classNode("field");
        const menuNode = classNode("menu");
        field.classList.owner = field;
        menuNode.classList.owner = menuNode;
        const input = {{
          value: "typed",
          attrs: {{}},
          setAttribute(name, value) {{ this.attrs[name] = String(value); domEvents.push(["setAttr", name, String(value)]); }},
          removeAttribute(name) {{ delete this.attrs[name]; domEvents.push(["removeAttr", name]); }},
        }};
        const domMenu = picker.createMenuState({{ normalizeLineNumber: (value) => Number(value) || null }});
        const domRuntime = picker.createMenuDomRuntime({{ field, menu: menuNode, input, menuState: domMenu }});
        domMenu.openSearchQuery("src/app.js", {{ line: 12 }});
        const openDomState = domRuntime.apply();
        const openDomSnapshot = {{ fieldActive: Boolean(field.classes.active), menuOpen: Boolean(menuNode.classes.open), expanded: input.attrs["aria-expanded"] }};
        const resetDomState = domRuntime.resetInput("src/current.py");
        const resetDomSnapshot = {{ value: input.value, activeDescendant: input.attrs["aria-activedescendant"] || "", focus: resetDomState.focus }};
        domMenu.openSearchQuery("next.js", {{ line: 9 }});
        input.setAttribute("aria-activedescendant", "filePickerOption-0");
        const closeDomState = domRuntime.close({{ restoreInput: true, inputValue: "restored.py" }});
        const closeDomSnapshot = {{ fieldActive: Boolean(field.classes.active), menuOpen: Boolean(menuNode.classes.open), expanded: input.attrs["aria-expanded"], activeDescendant: input.attrs["aria-activedescendant"] || "", value: input.value, focus: closeDomState.focus }};
        let domHostError = "";
        try {{ picker.createMenuDomRuntime({{ field, menu: menuNode, input }}); }} catch (err) {{ domHostError = err && err.message ? err.message : String(err); }}
        process.stdout.write(JSON.stringify({{
          frozen: Object.isFrozen(picker),
          exports: Object.keys(picker).sort(),
          missingError,
          hostError,
          menuHostError,
          domHostError,
          menuState: {{
            opened,
            selectionBeforeInput,
            draftSuppressedBeforeInput,
            preservedFirst,
            preservedSecond,
            selectionAfterInput,
            draftSuppressedAfterInput,
            visibleQueryAfterInput,
            clampedFocus,
            movedFocus,
            enterIndex,
            closed,
          }},
          domState: {{
            frozen: Object.isFrozen(domRuntime),
            openDomState,
            openDomSnapshot,
            resetDomSnapshot,
            closeDomSnapshot,
            domEvents,
          }},
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
                "createMenuDomRuntime",
                "createMenuState",
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
        self.assertIn("Codoxear file picker host missing normalizeLineNumber", result["menuHostError"])
        self.assertIn("Codoxear file picker host missing snapshot", result["domHostError"])

    def test_file_picker_menu_state_behavior(self) -> None:
        result = run_picker_module_probe()["menuState"]
        self.assertTrue(result["opened"])
        self.assertEqual(result["selectionBeforeInput"], 42)
        self.assertTrue(result["draftSuppressedBeforeInput"])
        self.assertTrue(result["preservedFirst"])
        self.assertFalse(result["preservedSecond"])
        self.assertIsNone(result["selectionAfterInput"])
        self.assertFalse(result["draftSuppressedAfterInput"])
        self.assertEqual(result["visibleQueryAfterInput"], "src/other.js")
        self.assertEqual(result["clampedFocus"], 1)
        self.assertEqual(result["movedFocus"], 0)
        self.assertEqual(result["enterIndex"], 0)
        self.assertFalse(result["closed"]["open"])
        self.assertEqual(result["closed"]["focus"], -1)

    def test_file_picker_dom_runtime_behavior(self) -> None:
        result = run_picker_module_probe()["domState"]
        self.assertTrue(result["frozen"])
        self.assertTrue(result["openDomState"]["open"])
        self.assertEqual(result["openDomSnapshot"], {"fieldActive": True, "menuOpen": True, "expanded": "true"})
        self.assertEqual(result["resetDomSnapshot"], {"value": "src/current.py", "activeDescendant": "", "focus": -1})
        self.assertEqual(result["closeDomSnapshot"], {"fieldActive": False, "menuOpen": False, "expanded": "false", "activeDescendant": "", "value": "restored.py", "focus": -1})
        self.assertEqual(result["domEvents"], [
            ["toggle", "field", "active", True],
            ["toggle", "menu", "open", True],
            ["setAttr", "aria-expanded", "true"],
            ["removeAttr", "aria-activedescendant"],
            ["setAttr", "aria-activedescendant", "filePickerOption-0"],
            ["removeAttr", "aria-activedescendant"],
            ["toggle", "field", "active", False],
            ["toggle", "menu", "open", False],
            ["setAttr", "aria-expanded", "false"],
            ["removeAttr", "aria-activedescendant"],
        ])

    def test_file_picker_module_registered_before_app_js(self) -> None:
        index_source = INDEX_HTML.read_text(encoding="utf-8")
        routes_source = STATIC_ROUTES.read_text(encoding="utf-8")
        app_source = APP_JS.read_text(encoding="utf-8")

        self.assertLess(index_source.index("app_file_helpers.js"), index_source.index("app_file_picker.js"))
        self.assertLess(index_source.index("app_file_picker.js"), index_source.index("app.js"))
        self.assertIn('"app_file_picker.js"', routes_source)
        self.assertIn("const codoxearFilePicker = window.CodoxearFilePicker;", app_source)
        self.assertIn('typeof codoxearFilePicker.createMenuDomRuntime !== "function"', app_source)
        self.assertIn("const filePickerDomRuntime = codoxearFilePicker.createMenuDomRuntime", app_source)
        self.assertIn("return filePickerDomRuntime.apply();", app_source)
        self.assertIn("return filePickerDomRuntime.resetInput(activeFilePathValue() || \"\");", app_source)
        self.assertIn("return filePickerDomRuntime.close({ restoreInput, inputValue: activeFilePathValue() || \"\" });", app_source)
        self.assertNotIn("filePickerField.classList.toggle(\"active\", state.open);", app_source)
        self.assertNotIn("filePickerMenuState.resetInputState();\n          filePickerInput.value = activeFilePathValue() || \"\";", app_source)
        self.assertIn('throw new Error("Codoxear file picker helpers failed to load")', app_source)


if __name__ == "__main__":
    unittest.main()
