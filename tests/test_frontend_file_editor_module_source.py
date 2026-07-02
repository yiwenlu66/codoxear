import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_FILE_EDITOR_JS = ROOT / "codoxear" / "static" / "app_file_editor.js"
APP_JS = ROOT / "codoxear" / "static" / "app.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"
STATIC_ROUTES = ROOT / "codoxear" / "static_routes.py"


def run_file_editor_runtime_probe() -> dict[str, object]:
    editor_source = APP_FILE_EDITOR_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(editor_source)}, ctx);
        const mod = ctx.window.CodoxearFileEditor;
        const events = [];
        function disposable(name, throws = false) {{
          return {{ dispose() {{ events.push(["dispose", name]); if (throws) throw new Error(`${{name}} failed`); }} }};
        }}
        const runtime = mod.createFileEditorRuntime();
        const modifiedEditor = {{ tag: "modified" }};
        const fileEditor = {{ tag: "file", dispose: () => events.push(["dispose", "fileEditor"]) }};
        const diffEditor = {{
          tag: "diff",
          getModifiedEditor: () => modifiedEditor,
          updateOptions: (options) => events.push(["updateOptions", options]),
          dispose: () => events.push(["dispose", "diffEditor"]),
        }};
        const emptyActive = runtime.activeCodeEditor("file");
        runtime.setEditor(fileEditor);
        const activeFile = runtime.activeCodeEditor("file") === fileEditor;
        const noDiffEditor = runtime.activeCodeEditor("diff");
        runtime.setEditor(diffEditor);
        const activeDiff = runtime.activeCodeEditor("diff") === modifiedEditor;
        const updateWrongKind = runtime.updateEditorOptions("file", {{ readOnly: true }});
        const updateDiff = runtime.updateEditorOptions("diff", {{ hideUnchangedRegions: {{ enabled: false }} }});
        const modelA = disposable("modelA");
        const modelB = disposable("modelB", true);
        runtime.setModels([modelA, null, modelB]);
        const modelCountBeforeDispose = runtime.currentModels().length;
        runtime.setChangeDisposable(disposable("change"));
        const withCurrent = runtime.withCurrentEditor((editor) => editor && editor.tag);
        const disposeResult = runtime.dispose({{
          clearHost: () => events.push(["clearHost"]),
          afterDispose: () => events.push(["afterDispose"]),
        }});
        const currentAfterDispose = runtime.currentEditor();
        const modelCountAfterDispose = runtime.currentModels().length;
        let missingCallbackError = "";
        try {{ runtime.withCurrentEditor(null); }} catch (err) {{ missingCallbackError = err && err.message ? err.message : String(err); }}
        process.stdout.write(JSON.stringify({{
          frozen: Object.isFrozen(mod),
          runtimeFrozen: Object.isFrozen(runtime),
          exports: Object.keys(mod).sort(),
          emptyActive,
          activeFile,
          noDiffEditor,
          activeDiff,
          updateWrongKind,
          updateDiff,
          modelCountBeforeDispose,
          withCurrent,
          disposeResult,
          currentAfterDispose,
          modelCountAfterDispose,
          events,
          missingCallbackError,
        }}));
        """
    )
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendFileEditorModuleSource(unittest.TestCase):
    def test_file_editor_runtime_lifecycle_behavior(self) -> None:
        result = run_file_editor_runtime_probe()
        self.assertTrue(result["frozen"])
        self.assertTrue(result["runtimeFrozen"])
        self.assertEqual(result["exports"], ["createFileEditorRuntime"])
        self.assertIsNone(result["emptyActive"])
        self.assertTrue(result["activeFile"])
        self.assertIsNone(result["noDiffEditor"])
        self.assertTrue(result["activeDiff"])
        self.assertFalse(result["updateWrongKind"])
        self.assertTrue(result["updateDiff"])
        self.assertEqual(result["modelCountBeforeDispose"], 2)
        self.assertEqual(result["withCurrent"], "diff")
        self.assertTrue(result["disposeResult"])
        self.assertIsNone(result["currentAfterDispose"])
        self.assertEqual(result["modelCountAfterDispose"], 0)
        self.assertEqual(
            result["events"],
            [
                ["updateOptions", {"hideUnchangedRegions": {"enabled": False}}],
                ["dispose", "change"],
                ["clearHost"],
                ["dispose", "modelA"],
                ["dispose", "modelB"],
                ["dispose", "diffEditor"],
                ["afterDispose"],
            ],
        )
        self.assertIn("file editor dependency missing: withCurrentEditor", result["missingCallbackError"])

    def test_file_editor_module_registered_before_app_js(self) -> None:
        index_source = INDEX_HTML.read_text(encoding="utf-8")
        routes_source = STATIC_ROUTES.read_text(encoding="utf-8")
        app_source = APP_JS.read_text(encoding="utf-8")
        self.assertLess(index_source.index("app_file_viewer.js"), index_source.index("app_file_editor.js"))
        self.assertLess(index_source.index("app_file_editor.js"), index_source.index("app_session_helpers.js"))
        self.assertLess(index_source.index("app_file_editor.js"), index_source.index("app.js"))
        self.assertIn('"app_file_editor.js"', routes_source)
        self.assertIn("const codoxearFileEditor = window.CodoxearFileEditor;", app_source)
        self.assertIn('throw new Error("Codoxear file editor runtime failed to load")', app_source)

    def test_app_js_no_longer_owns_raw_editor_lifecycle_lists(self) -> None:
        app_source = APP_JS.read_text(encoding="utf-8")
        editor_source = APP_FILE_EDITOR_JS.read_text(encoding="utf-8")
        self.assertNotIn("let fileEditor = null;", app_source)
        self.assertNotIn("let fileEditorModels = [];", app_source)
        self.assertNotIn("let fileEditorChangeDisposable = null;", app_source)
        self.assertIn("let editor = null;", editor_source)
        self.assertIn("let models = [];", editor_source)
        self.assertIn("let changeDisposable = null;", editor_source)
        self.assertIn("fileEditorRuntime.dispose({", app_source)
        self.assertIn("fileEditorRuntime.setEditor(editor);", app_source)
        self.assertIn("fileEditorRuntime.setModels([editor.getModel()].filter(Boolean));", app_source)
        self.assertIn("fileEditorRuntime.setChangeDisposable(editor.onDidChangeModelContent", app_source)


if __name__ == "__main__":
    unittest.main()
