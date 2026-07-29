import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_FILE_EDITOR_JS = ROOT / "codoxear" / "static" / "app_file_editor.js"


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
          getValue: () => "current editor text",
          getValueInRange: (range) => `text:${{range.startLineNumber}}:${{range.startColumn}}-${{range.endLineNumber}}:${{range.endColumn}}`,
        }};
        const modifiedEditor = {{
          tag: "modified",
          setPosition: (position) => events.push(["setPosition", "modified", position]),
          revealLineInCenter: (line) => events.push(["revealLineInCenter", "modified", line]),
          focus: () => events.push(["focus", "modified"]),
        }};
        const editorInput = {{ classList: {{ contains: (name) => name === "inputarea" }} }};
        const otherInput = {{ classList: {{ contains: (name) => name === "inputarea" }} }};
        const nonInput = {{ classList: {{ contains: () => false }} }};
        const domNode = {{ contains: (target) => target === editorInput }};
        const fileEditor = {{
          tag: "file",
          layout: () => events.push(["layout", "file"]),
          setPosition: (position) => events.push(["setPosition", "file", position]),
          revealLineInCenter: (line) => events.push(["revealLineInCenter", "file", line]),
          focus: () => events.push(["focus", "file"]),
          getDomNode: () => domNode,
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
        const activeInput = runtime.isActiveInput("file", editorInput);
        const outsideInput = runtime.isActiveInput("file", otherInput);
        const wrongClassInput = runtime.isActiveInput("file", nonInput);
        const wrongCtorInput = runtime.isActiveInput("file", editorInput, function DifferentElement() {{}});
        const normalizedPosition = runtime.normalizePosition(fileEditor, {{ lineNumber: 20, column: 20 }});
        const currentFileText = runtime.currentFileText("file", "fallback text");
        const selectionTextBefore = runtime.activeSelectionText("file");
        const applyAnchoredSelection = runtime.applySelection(fileEditor, {{ lineNumber: 2, column: 20 }}, {{ lineNumber: 0, column: 0 }}, Selection);
        const selectionTextAfter = runtime.selectionText(fileEditor);
        const collapsedSelection = runtime.isCollapsedSelection({{ startLineNumber: 2, startColumn: 1, endLineNumber: 2, endColumn: 1 }});
        runtime.setEditor(diffEditor);
        const activeDiff = runtime.activeCodeEditor("diff") === modifiedEditor;
        const currentDiffTextFallback = runtime.currentFileText("diff", "fallback text");
        const updateWrongKind = runtime.updateEditorOptions("file", {{ readOnly: true }});
        const updateDiff = runtime.updateEditorOptions("diff", {{ hideUnchangedRegions: {{ enabled: false }} }});
        const layoutDiff = runtime.layoutCurrent();
        const focusDiff = runtime.focusLine("diff", "9", (value) => Number(value) || null);
        const scheduledLineFocus = runtime.scheduleLineFocus("diff", "5", {{
          requestAnimationFrame: (callback) => {{ events.push(["scheduleFrame"]); callback(); }},
          setTimeout: (callback, delay) => {{ events.push(["scheduleTimer", delay]); callback(); }},
          isCurrent: () => true,
          delayMs: 12,
        }});
        const staleScheduledLineFocus = runtime.scheduleLineFocus("diff", "6", {{
          requestAnimationFrame: (callback) => {{ events.push(["staleFrame"]); callback(); }},
          setTimeout: (callback, delay) => {{ events.push(["staleTimer", delay]); callback(); }},
          isCurrent: () => false,
        }});
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

        const disposeLifecycleEvents = [];
        const disposeLifecycleRuntime = mod.createFileEditorRuntime();
        disposeLifecycleRuntime.setEditor({{ dispose: () => disposeLifecycleEvents.push(["disposeEditor"]) }});
        disposeLifecycleRuntime.setModels([{{ dispose: () => disposeLifecycleEvents.push(["disposeModel"]) }}]);
        disposeLifecycleRuntime.setChangeDisposable({{ dispose: () => disposeLifecycleEvents.push(["disposeChange"]) }});
        const disposeCurrentFileResult = disposeLifecycleRuntime.disposeCurrentFile({{
          finishProgrammaticChange: () => disposeLifecycleEvents.push(["finishProgrammaticChange"]),
          clearHost: () => disposeLifecycleEvents.push(["clearHost"]),
          setFileEditorKind: (kind) => disposeLifecycleEvents.push(["setKind", kind]),
          clearFileTouchSelectionState: () => disposeLifecycleEvents.push(["clearTouchSelection"]),
        }});
        let disposeCurrentMissingError = "";
        try {{ disposeLifecycleRuntime.disposeCurrentFile({{}}); }} catch (err) {{ disposeCurrentMissingError = err && err.message ? err.message : String(err); }}

        const restoreLifecycleEvents = [];
        const restoreLifecycleRuntime = mod.createFileEditorRuntime();
        restoreLifecycleRuntime.setEditor({{ getModel: () => ({{ setValue: (value) => restoreLifecycleEvents.push(["setValue", value]) }}) }});
        const restoreCurrentFileResult = restoreLifecycleRuntime.restoreCurrentFileText("input", {{
          prepareFileEditorTextRestore: (value) => {{ restoreLifecycleEvents.push(["prepare", value]); return {{ kind: "restore", text: "restored text" }}; }},
          currentFileEditorKind: () => "file",
          runFileEditorProgrammaticChange: (callback) => {{ restoreLifecycleEvents.push(["programmaticStart"]); callback(); restoreLifecycleEvents.push(["programmaticEnd"]); }},
          finishFileEditorTextRestore: () => restoreLifecycleEvents.push(["finish"]),
        }});
        const restoreCurrentNoop = restoreLifecycleRuntime.restoreCurrentFileText("skip", {{
          prepareFileEditorTextRestore: () => null,
          currentFileEditorKind: () => "file",
          runFileEditorProgrammaticChange: (callback) => callback(),
          finishFileEditorTextRestore: () => restoreLifecycleEvents.push(["unexpectedFinish"]),
        }});
        let restoreCurrentMissingError = "";
        try {{ restoreLifecycleRuntime.restoreCurrentFileText("x", {{}}); }} catch (err) {{ restoreCurrentMissingError = err && err.message ? err.message : String(err); }}

        const creationEvents = [];
        const fileRuntime = mod.createFileEditorRuntime();
        const createdFileModel = {{
          setValue: (value) => creationEvents.push(["setValue", value]),
        }};
        const createdFileEditor = {{
          tag: "createdFile",
          getModel: () => createdFileModel,
          onDidChangeModelContent: (callback) => {{ creationEvents.push(["bindChange"]); return {{ dispose: () => creationEvents.push(["disposeChange"]) }}; }},
          setScrollPosition: (position) => creationEvents.push(["fileScroll", position]),
          setPosition: (position) => creationEvents.push(["filePosition", position]),
          revealPositionInCenter: (position) => creationEvents.push(["fileReveal", position]),
          layout: () => creationEvents.push(["fileLayout"]),
        }};
        const monacoCreate = {{
          editor: {{
            create: (host, options) => {{ creationEvents.push(["createFile", host.name, options]); return createdFileEditor; }},
            setModelLanguage: (model, language) => creationEvents.push(["setLanguage", model === createdFileModel, language]),
          }},
        }};
        const createFileEditorResult = fileRuntime.createFileEditor(monacoCreate, {{ name: "fileHost" }}, {{
          path: "src/main.py",
          text: "print(1)",
          readOnly: true,
          onDidChangeModelContent: () => creationEvents.push(["changed"]),
        }}).tag;
        const filePositionState = fileRuntime.positionCurrentEditorAtLine("file", "3", (value) => Number(value) || null);
        const updateFileTextResult = fileRuntime.updateFileEditorText(monacoCreate, {{
          path: "README.md",
          text: "# title",
          runProgrammaticChange: (callback) => {{ creationEvents.push(["programmaticStart"]); callback(); creationEvents.push(["programmaticEnd"]); }},
        }});
        const restoreFileTextResult = fileRuntime.restoreFileText("file", "restored", (callback) => {{ creationEvents.push(["restoreProgrammaticStart"]); callback(); creationEvents.push(["restoreProgrammaticEnd"]); }});
        const restoreWrongKind = fileRuntime.restoreFileText("diff", "ignored", () => creationEvents.push(["unexpectedRestore"]));

        const diffRuntime = mod.createFileEditorRuntime();
        const originalEditor = {{
          tag: "original",
          updateOptions: (options) => creationEvents.push(["originalOptions", options]),
          setScrollPosition: (position) => creationEvents.push(["originalScroll", position]),
          setPosition: (position) => creationEvents.push(["originalPosition", position]),
        }};
        const modifiedEditorForCreate = {{
          tag: "modifiedCreated",
          updateOptions: (options) => creationEvents.push(["modifiedOptions", options]),
          setScrollPosition: (position) => creationEvents.push(["modifiedScroll", position]),
          setPosition: (position) => creationEvents.push(["modifiedPosition", position]),
          revealPositionInCenter: (position) => creationEvents.push(["modifiedReveal", position]),
        }};
        const createdDiffEditor = {{
          tag: "createdDiff",
          setModel: (model) => creationEvents.push(["setDiffModel", model.original.text, model.modified.text]),
          getOriginalEditor: () => originalEditor,
          getModifiedEditor: () => modifiedEditorForCreate,
          layout: () => creationEvents.push(["diffLayout"]),
        }};
        const monacoDiff = {{
          editor: {{
            createModel: (text, language) => ({{ text, language }}),
            createDiffEditor: (host, options) => {{ creationEvents.push(["createDiff", host.name, options]); return createdDiffEditor; }},
          }},
        }};
        const diffCreateResult = diffRuntime.createDiffEditor(monacoDiff, {{ name: "diffHost" }}, {{ path: "src/app.ts", originalText: "old", modifiedText: "new" }}).diffEditor.tag;
        const diffPositionState = diffRuntime.positionCurrentEditorAtLine("diff", null, (value) => Number(value) || null);
        const creation = {{
          createFileEditorResult,
          filePositionState,
          updateFileTextResult,
          restoreFileTextResult,
          restoreWrongKind,
          diffCreateResult,
          diffPositionState,
          fileModelCount: fileRuntime.currentModels().length,
          diffModelCount: diffRuntime.currentModels().length,
          creationEvents,
        }};
        process.stdout.write(JSON.stringify({{
          frozen: Object.isFrozen(mod),
          runtimeFrozen: Object.isFrozen(runtime),
          exports: Object.keys(mod).sort(),
          emptyActive,
          activeFile,
          noDiffEditor,
          activeDiff,
          focusFile,
          activeInput,
          outsideInput,
          wrongClassInput,
          wrongCtorInput,
          normalizedPosition,
          currentFileText,
          currentDiffTextFallback,
          selectionTextBefore,
          applyAnchoredSelection,
          selectionTextAfter,
          collapsedSelection,
          updateWrongKind,
          updateDiff,
          layoutDiff,
          focusDiff,
          scheduledLineFocus,
          staleScheduledLineFocus,
          modelCountBeforeDispose,
          withCurrent,
          disposeResult,
          currentAfterDispose,
          modelCountAfterDispose,
          disposeCurrentFileResult,
          disposeLifecycleEvents,
          disposeCurrentMissingError,
          restoreCurrentFileResult,
          restoreCurrentNoop,
          restoreLifecycleEvents,
          restoreCurrentMissingError,
          creation,
          events,
          missingCallbackError,
        }}));
        """
    )
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def run_file_editor_renderer_probe() -> dict[str, object]:
    editor_source = APP_FILE_EDITOR_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(editor_source)}, ctx);
        const mod = ctx.window.CodoxearFileEditor;
        const events = [];
        let currentKind = "";
        let activeText = "old text";
        let currentText = "changed text";
        let programmatic = false;
        let touchMode = true;
        let current = true;
        let ensureMode = "success";
        let changeCallback = null;
        const host = {{ name: "fileHost" }};
        const runtime = {{
          createFileEditor(monaco, editorHost, options) {{
            events.push(["createFile", monaco.name, editorHost.name, options.path, options.text, options.languageOverride, options.readOnly]);
            changeCallback = options.onDidChangeModelContent;
            return {{ tag: "createdFile" }};
          }},
          updateFileEditorText(monaco, options) {{
            events.push(["updateFile", monaco.name, options.path, options.text, options.languageOverride]);
            options.runProgrammaticChange(() => events.push(["updateProgrammaticBody"]));
            return true;
          }},
          createDiffEditor(monaco, editorHost, options) {{
            events.push(["createDiff", monaco.name, editorHost.name, options.path, options.originalText, options.modifiedText]);
            return {{ diffEditor: {{ tag: "createdDiff" }} }};
          }},
          positionCurrentEditorAtLine(kind, lineNumber, normalizeLineNumber) {{
            const requestedLine = normalizeLineNumber(lineNumber);
            events.push(["position", kind, requestedLine]);
            return {{ requestedLine, targetLine: requestedLine || 1 }};
          }},
          scheduleLineFocus(kind, requestedLine, options) {{
            events.push(["schedule", kind, requestedLine, options.isCurrent()]);
            options.requestAnimationFrame(() => events.push(["frameCallback", kind]));
            options.setTimeout(() => events.push(["timerCallback", kind]), 60);
            return true;
          }},
          currentFileText(kind, fallbackText) {{
            events.push(["currentFileText", kind, fallbackText]);
            return currentText;
          }},
        }};
        const loader = {{
          ensure: async () => {{
            events.push(["ensure", ensureMode]);
            if (ensureMode === "fail") throw new Error("loader boom");
            return {{ name: "monaco" }};
          }},
        }};
        const renderer = mod.createFileEditorRenderer({{
          runtime,
          monacoLoader: loader,
          host,
          normalizeLineNumber: (value) => Number(value) || null,
          requestAnimationFrame: (callback) => {{ events.push(["requestFrame"]); callback(); }},
          setTimeout: (callback, delay) => {{ events.push(["setTimeout", delay]); callback(); }},
          isCurrentFileOpenRequest: () => current,
          renderPlainTextFallback: (rel, text, lineNumber, reason) => events.push(["fallback", rel, text, lineNumber, reason]),
          disposeFileEditor: () => {{ events.push(["dispose"]); currentKind = ""; }},
          currentEditorKind: () => currentKind,
          setEditorKind: (kind) => {{ currentKind = kind; events.push(["setKind", kind]); }},
          currentFileEditMode: () => true,
          currentActiveFileEditable: () => true,
          isUnavailable: () => false,
          isProgrammaticChange: () => programmatic,
          currentTouchSelectMode: () => touchMode,
          resetTouchSelectionState: () => {{ events.push(["resetTouch"]); touchMode = false; }},
          currentActiveFileText: () => activeText,
          setDirty: (dirty) => events.push(["setDirty", dirty]),
          runProgrammaticChange: (callback) => {{ events.push(["programmaticStart"]); callback(); events.push(["programmaticEnd"]); }},
          syncReadOnly: () => events.push(["syncReadOnly"]),
          updateTouchToolbar: () => events.push(["updateTouchToolbar"]),
        }});

        (async () => {{
          const fileCreated = await renderer.renderFile("src/a.js", "old text", 3, "javascript", {{ id: "req" }});
          changeCallback();
          programmatic = true;
          currentText = "programmatic text";
          changeCallback();
          programmatic = false;
          touchMode = false;
          activeText = "fresh";
          currentText = "fresh";
          const fileUpdated = await renderer.renderFile("src/a.ts", "fresh", null, "typescript", {{ id: "req" }});
          current = false;
          const staleFile = await renderer.renderFile("src/stale.js", "stale", 8, "", {{ id: "req" }});
          current = true;
          ensureMode = "fail";
          const fileFallback = await renderer.renderFile("src/fallback.txt", "plain", 4, "", {{ id: "req" }});
          const diffFallback = await renderer.renderDiff("src/fallback.diff", "old", "new", 5, {{ id: "req" }});
          ensureMode = "success";
          currentKind = "plain-fallback";
          const diffRendered = await renderer.renderDiff("src/a.js", "old", "new", 2, {{ id: "req" }});
          const ensured = await renderer.ensureMonaco();
          let missingHostError = "";
          try {{
            mod.createFileEditorRenderer({{
              runtime,
              monacoLoader: loader,
              host: null,
              normalizeLineNumber: () => null,
              requestAnimationFrame: () => null,
              setTimeout: () => null,
              isCurrentFileOpenRequest: () => true,
              renderPlainTextFallback: () => null,
              disposeFileEditor: () => null,
              currentEditorKind: () => "",
              setEditorKind: () => null,
              currentFileEditMode: () => false,
              currentActiveFileEditable: () => false,
              isUnavailable: () => false,
              isProgrammaticChange: () => false,
              currentTouchSelectMode: () => false,
              resetTouchSelectionState: () => null,
              currentActiveFileText: () => "",
              setDirty: () => null,
              runProgrammaticChange: (callback) => callback(),
              syncReadOnly: () => null,
              updateTouchToolbar: () => null,
            }});
          }} catch (err) {{ missingHostError = err && err.message ? err.message : String(err); }}
          process.stdout.write(JSON.stringify({{
            rendererFrozen: Object.isFrozen(renderer),
            fileCreated,
            fileUpdated,
            staleFile,
            fileFallback,
            diffFallback,
            diffRendered,
            ensuredName: ensured.name,
            currentKind,
            missingHostError,
            events,
          }}));
        }})().catch((err) => {{ console.error(err && err.stack ? err.stack : err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def run_diff_fallback_probe() -> dict[str, object]:
    editor_source = APP_FILE_EDITOR_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(editor_source)}, ctx);
        const mod = ctx.window.CodoxearFileEditor;
        const events = [];
        let current = true;
        const loader = {{
          ensure() {{ events.push(["ensure"]); return Promise.reject(new Error("loader unavailable")); }},
        }};
        const runtime = {{
          createFileEditor: () => ({{}}),
          updateFileEditorText: () => true,
          createDiffEditor: () => ({{}}),
          positionCurrentEditorAtLine: () => null,
          scheduleLineFocus: () => null,
          currentFileText: () => "",
        }};
        const host = {{ name: "diffHost" }};
        const renderer = mod.createFileEditorRenderer({{
          runtime,
          monacoLoader: loader,
          host,
          normalizeLineNumber: (value) => Number(value) || null,
          requestAnimationFrame: () => null,
          setTimeout: (callback) => callback(),
          isCurrentFileOpenRequest: () => current,
          renderPlainTextFallback: (rel, text, lineNumber, reason) => {{ events.push(["fallback", rel, text, lineNumber, reason]); return {{}}; }},
          disposeFileEditor: () => events.push(["dispose"]),
          currentEditorKind: () => "",
          setEditorKind: (kind) => events.push(["setKind", kind]),
          currentFileEditMode: () => false,
          currentActiveFileEditable: () => false,
          isUnavailable: () => false,
          isProgrammaticChange: () => false,
          currentTouchSelectMode: () => false,
          resetTouchSelectionState: () => null,
          currentActiveFileText: () => "",
          setDirty: () => null,
          runProgrammaticChange: (callback) => callback(),
          syncReadOnly: () => null,
          updateTouchToolbar: () => null,
        }});
        (async () => {{
          const withDiff = await renderer.renderDiff("src/note.md", "base", "working", 1, {{ id: "req" }}, "@@ -1 +1 @@ -base +working");
          const withoutDiff = await renderer.renderDiff("src/note2.md", "base", "working", 1, {{ id: "req" }}, "");
          process.stdout.write(JSON.stringify({{ withDiff, withoutDiff, events }}));
        }})().catch((err) => {{ console.error(err && err.stack ? err.stack : err); process.exit(1); }});
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
          const monacoEnvironment = fakeGlobal.MonacoEnvironment || {{}};
          const hasWorkerOverride = typeof monacoEnvironment.getWorker === "function" || typeof monacoEnvironment.getWorkerUrl === "function";
          let missingResolveError = "";
          try {{ mod.createMonacoLoader({{}}); }} catch (err) {{ missingResolveError = err && err.message ? err.message : String(err); }}
          process.stdout.write(JSON.stringify({{
            beforeSupport,
            afterSupport,
            samePromiseValue: first === second,
            currentIsMonaco: loader.currentMonaco() === monaco,
            selection,
            hasWorkerOverride,
            monacoEnvironmentKeys: Object.keys(monacoEnvironment).sort(),
            events,
            missingResolveError,
          }}));
        }})().catch((err) => {{ console.error(err && err.stack ? err.stack : err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendFileEditorModuleBehavior(unittest.TestCase):
    def test_file_editor_runtime_lifecycle_behavior(self) -> None:
        result = run_file_editor_runtime_probe()
        self.assertTrue(result["frozen"])
        self.assertTrue(result["runtimeFrozen"])
        self.assertEqual(result["exports"], ["createFileEditorRenderer", "createFileEditorRuntime", "createMonacoLoader"])
        self.assertIsNone(result["emptyActive"])
        self.assertTrue(result["activeFile"])
        self.assertIsNone(result["noDiffEditor"])
        self.assertTrue(result["activeDiff"])
        self.assertTrue(result["focusFile"])
        self.assertTrue(result["activeInput"])
        self.assertFalse(result["outsideInput"])
        self.assertFalse(result["wrongClassInput"])
        self.assertFalse(result["wrongCtorInput"])
        self.assertEqual(result["normalizedPosition"], {"lineNumber": 4, "column": 4})
        self.assertEqual(result["currentFileText"], "current editor text")
        self.assertEqual(result["currentDiffTextFallback"], "fallback text")
        self.assertEqual(result["selectionTextBefore"], "text:1:1-1:3")
        self.assertTrue(result["applyAnchoredSelection"])
        self.assertEqual(result["selectionTextAfter"], "text:1:1-2:6")
        self.assertTrue(result["collapsedSelection"])
        self.assertFalse(result["updateWrongKind"])
        self.assertTrue(result["updateDiff"])
        self.assertTrue(result["layoutDiff"])
        self.assertTrue(result["focusDiff"])
        self.assertTrue(result["scheduledLineFocus"])
        self.assertTrue(result["staleScheduledLineFocus"])
        self.assertEqual(result["modelCountBeforeDispose"], 2)
        self.assertEqual(result["withCurrent"], "diff")
        self.assertTrue(result["disposeResult"])
        self.assertIsNone(result["currentAfterDispose"])
        self.assertEqual(result["modelCountAfterDispose"], 0)
        self.assertTrue(result["disposeCurrentFileResult"])
        self.assertEqual(result["disposeLifecycleEvents"], [
            ["finishProgrammaticChange"],
            ["disposeChange"],
            ["clearHost"],
            ["disposeModel"],
            ["disposeEditor"],
            ["setKind", ""],
            ["clearTouchSelection"],
        ])
        self.assertIn("file editor dependency missing: finishProgrammaticChange", result["disposeCurrentMissingError"])
        self.assertTrue(result["restoreCurrentFileResult"])
        self.assertFalse(result["restoreCurrentNoop"])
        self.assertEqual(result["restoreLifecycleEvents"], [
            ["prepare", "input"],
            ["programmaticStart"],
            ["setValue", "restored text"],
            ["programmaticEnd"],
            ["finish"],
        ])
        self.assertIn("file editor dependency missing: prepareFileEditorTextRestore", result["restoreCurrentMissingError"])
        creation = result["creation"]
        self.assertEqual(creation["createFileEditorResult"], "createdFile")
        self.assertTrue(creation["updateFileTextResult"])
        self.assertTrue(creation["restoreFileTextResult"])
        self.assertFalse(creation["restoreWrongKind"])
        self.assertEqual(creation["diffCreateResult"], "createdDiff")
        self.assertEqual(creation["fileModelCount"], 1)
        self.assertEqual(creation["diffModelCount"], 2)
        self.assertEqual(creation["filePositionState"], {"requestedLine": 3, "targetLine": 3})
        self.assertEqual(creation["diffPositionState"], {"requestedLine": None, "targetLine": 1})
        creation_events = creation["creationEvents"]
        self.assertEqual(creation_events[0][0:2], ["createFile", "fileHost"])
        self.assertEqual(creation_events[0][2]["language"], "python")
        self.assertEqual(creation_events[0][2]["value"], "print(1)")
        self.assertTrue(creation_events[0][2]["readOnly"])
        self.assertEqual(creation_events[0][2]["theme"], "codoxear-github-light")
        self.assertIn(["bindChange"], creation_events)
        self.assertIn(["filePosition", {"lineNumber": 3, "column": 1}], creation_events)
        self.assertIn(["setLanguage", True, "markdown"], creation_events)
        self.assertIn(["setValue", "# title"], creation_events)
        self.assertIn(["setValue", "restored"], creation_events)
        diff_create = next(event for event in creation_events if event[0] == "createDiff")
        self.assertEqual(diff_create[1], "diffHost")
        self.assertTrue(diff_create[2]["readOnly"])
        self.assertEqual(diff_create[2]["hideUnchangedRegions"], {"enabled": True, "contextLineCount": 4, "minimumLineCount": 1, "revealLineCount": 2})
        self.assertIn(["originalOptions", {"wordWrap": "on", "lineNumbers": "off", "glyphMargin": False, "lineDecorationsWidth": 0, "lineNumbersMinChars": 0}], creation_events)
        self.assertIn(["modifiedOptions", {"wordWrap": "on", "lineNumbers": "on", "glyphMargin": False, "lineDecorationsWidth": 0, "lineNumbersMinChars": 3}], creation_events)
        self.assertIn(["modifiedReveal", {"lineNumber": 1, "column": 1}], creation_events)
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
                ["scheduleFrame"],
                ["layout", "diff"],
                ["setPosition", "modified", {"lineNumber": 5, "column": 1}],
                ["revealLineInCenter", "modified", 5],
                ["focus", "modified"],
                ["scheduleTimer", 12],
                ["layout", "diff"],
                ["setPosition", "modified", {"lineNumber": 5, "column": 1}],
                ["revealLineInCenter", "modified", 5],
                ["focus", "modified"],
                ["staleFrame"],
                ["staleTimer", 60],
                ["dispose", "change"],
                ["clearHost"],
                ["dispose", "modelA"],
                ["dispose", "modelB"],
                ["dispose", "diffEditor"],
                ["afterDispose"],
            ],
        )
        self.assertIn("file editor dependency missing: withCurrentEditor", result["missingCallbackError"])

    def test_file_editor_renderer_behavior(self) -> None:
        result = run_file_editor_renderer_probe()
        self.assertTrue(result["rendererFrozen"])
        self.assertTrue(result["fileCreated"])
        self.assertTrue(result["fileUpdated"])
        self.assertFalse(result["staleFile"])
        self.assertTrue(result["fileFallback"])
        self.assertTrue(result["diffFallback"])
        self.assertTrue(result["diffRendered"])
        self.assertEqual(result["ensuredName"], "monaco")
        self.assertEqual(result["currentKind"], "diff")
        self.assertIn("file editor dependency missing: host", result["missingHostError"])
        self.assertEqual(
            result["events"],
            [
                ["ensure", "success"],
                ["dispose"],
                ["createFile", "monaco", "fileHost", "src/a.js", "old text", "javascript", False],
                ["setKind", "file"],
                ["syncReadOnly"],
                ["position", "file", 3],
                ["schedule", "file", 3, True],
                ["requestFrame"],
                ["frameCallback", "file"],
                ["setTimeout", 60],
                ["timerCallback", "file"],
                ["updateTouchToolbar"],
                ["resetTouch"],
                ["currentFileText", "file", "old text"],
                ["setDirty", True],
                ["ensure", "success"],
                ["updateFile", "monaco", "src/a.ts", "fresh", "typescript"],
                ["programmaticStart"],
                ["updateProgrammaticBody"],
                ["programmaticEnd"],
                ["syncReadOnly"],
                ["position", "file", None],
                ["updateTouchToolbar"],
                ["ensure", "success"],
                ["ensure", "fail"],
                ["fallback", "src/fallback.txt", "plain", 4, "Code editor unavailable. Editing disabled: loader boom"],
                ["updateTouchToolbar"],
                ["ensure", "fail"],
                ["fallback", "src/fallback.diff", "", 5, "Diff editor unavailable because Monaco failed to load: loader boom"],
                ["updateTouchToolbar"],
                ["ensure", "success"],
                ["dispose"],
                ["createDiff", "monaco", "fileHost", "src/a.js", "old", "new"],
                ["setKind", "diff"],
                ["position", "diff", 2],
                ["schedule", "diff", 2, True],
                ["requestFrame"],
                ["frameCallback", "diff"],
                ["setTimeout", 60],
                ["timerCallback", "diff"],
                ["updateTouchToolbar"],
                ["ensure", "success"],
            ],
        )

    def test_file_editor_diff_requires_monaco_when_monaco_unavailable(self) -> None:
        result = run_diff_fallback_probe()
        self.assertTrue(result["withDiff"])
        self.assertTrue(result["withoutDiff"])
        events = result["events"]
        with_diff_fallback = next(e for e in events if e[0] == "fallback" and e[1] == "src/note.md")
        self.assertEqual(with_diff_fallback[2], "")
        self.assertIn("Diff editor unavailable because Monaco failed to load", with_diff_fallback[4])
        self.assertNotIn("unified diff", with_diff_fallback[4])
        self.assertNotIn("@@ -1 +1 @@ -base +working", str(events))
        without_diff_fallback = next(e for e in events if e[0] == "fallback" and e[1] == "src/note2.md")
        self.assertEqual(without_diff_fallback[2], "")
        self.assertIn("Diff editor unavailable because Monaco failed to load", without_diff_fallback[4])

    def test_monaco_loader_behavior(self) -> None:
        result = run_monaco_loader_probe()
        self.assertFalse(result["beforeSupport"])
        self.assertTrue(result["afterSupport"])
        self.assertTrue(result["samePromiseValue"])
        self.assertTrue(result["currentIsMonaco"])
        self.assertEqual(result["selection"], {"startLineNumber": 1, "startColumn": 2, "endLineNumber": 3, "endColumn": 4})
        self.assertFalse(result["hasWorkerOverride"])
        self.assertEqual(result["monacoEnvironmentKeys"], [])
        self.assertEqual(
            result["events"],
            [
                ["config", {"paths": {"vs": "app:monaco/vs"}}],
                ["require", ["vs/editor/editor.main"]],
                ["defineTheme", "codoxear-github-light", "#ffffff"],
            ],
        )
        self.assertIn("file editor dependency missing: resolveAppUrl", result["missingResolveError"])

if __name__ == "__main__":
    unittest.main()
