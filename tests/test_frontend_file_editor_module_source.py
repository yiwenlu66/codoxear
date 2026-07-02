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
        function Selection(startLineNumber, startColumn, endLineNumber, endColumn) {{
          this.startLineNumber = startLineNumber;
          this.startColumn = startColumn;
          this.endLineNumber = endLineNumber;
          this.endColumn = endColumn;
        }}
        let currentSelection = {{ startLineNumber: 1, startColumn: 1, endLineNumber: 1, endColumn: 3 }};
        const fileModel = {{
          getLineCount: () => 4,
          getLineMaxColumn: (lineNumber) => lineNumber === 2 ? 6 : 4,
          getValueInRange: (range) => `text:${{range.startLineNumber}}:${{range.startColumn}}-${{range.endLineNumber}}:${{range.endColumn}}`,
        }};
        const modifiedEditor = {{
          tag: "modified",
          setPosition: (position) => events.push(["setPosition", "modified", position]),
          revealLineInCenter: (line) => events.push(["revealLineInCenter", "modified", line]),
          focus: () => events.push(["focus", "modified"]),
        }};
        const fileEditor = {{
          tag: "file",
          layout: () => events.push(["layout", "file"]),
          setPosition: (position) => events.push(["setPosition", "file", position]),
          revealLineInCenter: (line) => events.push(["revealLineInCenter", "file", line]),
          focus: () => events.push(["focus", "file"]),
          getModel: () => fileModel,
          getSelection: () => currentSelection,
          setSelection: (selection) => {{ currentSelection = selection; events.push(["setSelection", selection]); }},
          revealPositionInCenterIfOutsideViewport: (position) => events.push(["revealPosition", position]),
          dispose: () => events.push(["dispose", "fileEditor"]),
        }};
        const diffEditor = {{
          tag: "diff",
          getModifiedEditor: () => modifiedEditor,
          layout: () => events.push(["layout", "diff"]),
          updateOptions: (options) => events.push(["updateOptions", options]),
          dispose: () => events.push(["dispose", "diffEditor"]),
        }};
        const emptyActive = runtime.activeCodeEditor("file");
        runtime.setEditor(fileEditor);
        const activeFile = runtime.activeCodeEditor("file") === fileEditor;
        const noDiffEditor = runtime.activeCodeEditor("diff");
        const focusFile = runtime.focusActiveCodeEditor("file") === fileEditor;
        const normalizedPosition = runtime.normalizePosition(fileEditor, {{ lineNumber: 20, column: 20 }});
        const selectionTextBefore = runtime.activeSelectionText("file");
        const applyAnchoredSelection = runtime.applySelection(fileEditor, {{ lineNumber: 2, column: 20 }}, {{ lineNumber: 0, column: 0 }}, Selection);
        const selectionTextAfter = runtime.selectionText(fileEditor);
        const collapsedSelection = runtime.isCollapsedSelection({{ startLineNumber: 2, startColumn: 1, endLineNumber: 2, endColumn: 1 }});
        runtime.setEditor(diffEditor);
        const activeDiff = runtime.activeCodeEditor("diff") === modifiedEditor;
        const updateWrongKind = runtime.updateEditorOptions("file", {{ readOnly: true }});
        const updateDiff = runtime.updateEditorOptions("diff", {{ hideUnchangedRegions: {{ enabled: false }} }});
        const layoutDiff = runtime.layoutCurrent();
        const focusDiff = runtime.focusLine("diff", "9", (value) => Number(value) || null);
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
          focusFile,
          normalizedPosition,
          selectionTextBefore,
          applyAnchoredSelection,
          selectionTextAfter,
          collapsedSelection,
          updateWrongKind,
          updateDiff,
          layoutDiff,
          focusDiff,
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


def run_monaco_loader_probe() -> dict[str, object]:
    editor_source = APP_FILE_EDITOR_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(editor_source)}, ctx);
        const mod = ctx.window.CodoxearFileEditor;
        const events = [];
        class Selection {{
          constructor(startLineNumber, startColumn, endLineNumber, endColumn) {{
            this.startLineNumber = startLineNumber;
            this.startColumn = startColumn;
            this.endLineNumber = endLineNumber;
            this.endColumn = endColumn;
          }}
        }}
        const monaco = {{
          Selection,
          editor: {{
            defineTheme: (name, theme) => events.push(["defineTheme", name, theme.colors["editor.background"]]),
          }},
        }};
        const fakeGlobal = {{
          setTimeout: (fn, _ms) => {{ events.push(["setTimeout"]); fn(); }},
        }};
        fakeGlobal.require = (deps, success, failure) => {{
          events.push(["require", deps]);
          fakeGlobal.monaco = monaco;
          success();
        }};
        fakeGlobal.require.config = (options) => events.push(["config", options]);
        const loader = mod.createMonacoLoader({{
          resolveAppUrl: (path) => `app:${{path}}`,
          globalObject: fakeGlobal,
          timeoutMs: 10,
          pollMs: 1,
        }});
        (async () => {{
          const beforeSupport = loader.editSupportAvailable();
          const first = await loader.ensure();
          const afterSupport = loader.editSupportAvailable();
          const second = await loader.ensure();
          const selection = new (loader.selectionCtor())(1, 2, 3, 4);
          const workerUrl = fakeGlobal.MonacoEnvironment.getWorkerUrl("", "");
          const workerScript = decodeURIComponent(workerUrl.split(",", 2)[1] || "");
          let missingResolveError = "";
          try {{ mod.createMonacoLoader({{}}); }} catch (err) {{ missingResolveError = err && err.message ? err.message : String(err); }}
          process.stdout.write(JSON.stringify({{
            beforeSupport,
            afterSupport,
            samePromiseValue: first === second,
            currentIsMonaco: loader.currentMonaco() === monaco,
            selection,
            workerUrl,
            workerScript,
            events,
            missingResolveError,
          }}));
        }})().catch((err) => {{ console.error(err && err.stack ? err.stack : err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendFileEditorModuleSource(unittest.TestCase):
    def test_file_editor_runtime_lifecycle_behavior(self) -> None:
        result = run_file_editor_runtime_probe()
        self.assertTrue(result["frozen"])
        self.assertTrue(result["runtimeFrozen"])
        self.assertEqual(result["exports"], ["createFileEditorRuntime", "createMonacoLoader"])
        self.assertIsNone(result["emptyActive"])
        self.assertTrue(result["activeFile"])
        self.assertIsNone(result["noDiffEditor"])
        self.assertTrue(result["activeDiff"])
        self.assertTrue(result["focusFile"])
        self.assertEqual(result["normalizedPosition"], {"lineNumber": 4, "column": 4})
        self.assertEqual(result["selectionTextBefore"], "text:1:1-1:3")
        self.assertTrue(result["applyAnchoredSelection"])
        self.assertEqual(result["selectionTextAfter"], "text:1:1-2:6")
        self.assertTrue(result["collapsedSelection"])
        self.assertFalse(result["updateWrongKind"])
        self.assertTrue(result["updateDiff"])
        self.assertTrue(result["layoutDiff"])
        self.assertTrue(result["focusDiff"])
        self.assertEqual(result["modelCountBeforeDispose"], 2)
        self.assertEqual(result["withCurrent"], "diff")
        self.assertTrue(result["disposeResult"])
        self.assertIsNone(result["currentAfterDispose"])
        self.assertEqual(result["modelCountAfterDispose"], 0)
        self.assertEqual(
            result["events"],
            [
                ["focus", "file"],
                ["setSelection", {"startLineNumber": 1, "startColumn": 1, "endLineNumber": 2, "endColumn": 6}],
                ["revealPosition", {"lineNumber": 2, "column": 6}],
                ["updateOptions", {"hideUnchangedRegions": {"enabled": False}}],
                ["layout", "diff"],
                ["setPosition", "modified", {"lineNumber": 9, "column": 1}],
                ["revealLineInCenter", "modified", 9],
                ["focus", "modified"],
                ["dispose", "change"],
                ["clearHost"],
                ["dispose", "modelA"],
                ["dispose", "modelB"],
                ["dispose", "diffEditor"],
                ["afterDispose"],
            ],
        )
        self.assertIn("file editor dependency missing: withCurrentEditor", result["missingCallbackError"])

    def test_monaco_loader_behavior(self) -> None:
        result = run_monaco_loader_probe()
        self.assertFalse(result["beforeSupport"])
        self.assertTrue(result["afterSupport"])
        self.assertTrue(result["samePromiseValue"])
        self.assertTrue(result["currentIsMonaco"])
        self.assertEqual(result["selection"], {"startLineNumber": 1, "startColumn": 2, "endLineNumber": 3, "endColumn": 4})
        self.assertIn("app:monaco/vs/base/worker/workerMain.js", result["workerScript"])
        self.assertEqual(
            result["events"],
            [
                ["config", {"paths": {"vs": "app:monaco/vs"}}],
                ["require", ["vs/editor/editor.main"]],
                ["defineTheme", "codoxear-github-light", "#ffffff"],
            ],
        )
        self.assertIn("file editor dependency missing: resolveAppUrl", result["missingResolveError"])

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
        self.assertIn("fileEditorRuntime.focusLine(currentFileEditorKind(), lineNumber, normalizeLineNumber);", app_source)
        self.assertIn("fileEditorRuntime.focusActiveCodeEditor(currentFileEditorKind())", app_source)
        self.assertIn("fileEditorRuntime.normalizePosition(editor, position)", app_source)
        self.assertIn("fileEditorRuntime.applySelection(editor, cursor, anchor, fileEditorMonacoLoader.selectionCtor())", app_source)
        self.assertIn("fileEditorRuntime.isCollapsedSelection(selection)", app_source)
        self.assertIn("fileEditorRuntime.activeSelectionText(currentFileEditorKind())", app_source)
        self.assertIn("fileEditorRuntime.layoutCurrent()", app_source)
        self.assertIn("fileEditorRuntime.setEditor(editor);", app_source)
        self.assertIn("fileEditorRuntime.setModels([editor.getModel()].filter(Boolean));", app_source)
        self.assertIn("fileEditorRuntime.setChangeDisposable(editor.onDidChangeModelContent", app_source)
        self.assertNotIn("let monacoReadyPromise = null;", app_source)
        self.assertNotIn("let monacoNs = null;", app_source)
        self.assertNotIn("let monacoThemeReady = false;", app_source)
        self.assertIn("function createMonacoLoader(options = {})", editor_source)
        self.assertIn("fileEditorMonacoLoader.ensure();", app_source)
        self.assertIn("fileEditorMonacoLoader.selectionCtor())", app_source)
        self.assertIn("fileEditorMonacoLoader.editSupportAvailable()", app_source)


if __name__ == "__main__":
    unittest.main()
