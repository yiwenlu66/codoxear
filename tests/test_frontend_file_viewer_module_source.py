import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_FILE_VIEWER_JS = ROOT / "codoxear" / "static" / "app_file_viewer.js"
APP_JS = ROOT / "codoxear" / "static" / "app.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"
STATIC_ROUTES = ROOT / "codoxear" / "static_routes.py"


def run_file_viewer_controller_probe() -> dict[str, object]:
    viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(viewer_source)}, ctx);
        const fileViewer = ctx.window.CodoxearFileViewer;
        let missingDependencyError = "";
        try {{
          fileViewer.createFileViewerController({{}});
        }} catch (err) {{
          missingDependencyError = err && err.message ? err.message : String(err);
        }}
        const state = {{
          sessionId: "sid-1",
          path: "src/app.py",
          line: 7,
          gitPath: true,
          apiPath: "api-token",
          unavailable: false,
          viewMode: "file",
          activeEntry: null,
          gitFresh: false,
          markdownPreviewable: true,
        }};
        const events = [];
        const fileStatus = {{
          textContent: "",
          children: [],
          replaceChildren(...nodes) {{ this.children = nodes; this.textContent = ""; events.push(["replaceChildren", nodes.map((node) => node.tag)]); }},
        }};
        function el(tag, attrs = {{}}, children = []) {{
          const node = {{ tag, attrs, children: Array.isArray(children) ? children : [], onclick: null }};
          if (attrs && Object.prototype.hasOwnProperty.call(attrs, "text")) node.text = attrs.text;
          return node;
        }}
        function makeController() {{
          return fileViewer.createFileViewerController({{
            el,
            fileStatus,
            currentSessionId: () => state.sessionId,
            normalizeLineNumber: (value) => value == null || value === "" ? null : Number(value),
            normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
            fileApiPathForPath: (_path, existing) => existing || "derived-token",
            isUnavailable: () => state.unavailable,
            confirmReload: (message) => {{ events.push(["confirm", message]); return state.confirmResult !== false; }},
            openFilePath: async (path, opts) => {{
              events.push(["open", path, opts]);
              if (state.openEffect === "unavailable") {{
                state.unavailable = true;
                fileStatus.textContent = "Session is no longer available; copy unsaved edits before closing.";
              }} else if (state.openEffect === "switch-session") {{
                state.sessionId = "sid-2";
                state.path = "src/app.py";
              }}
              return state.openResult === true;
            }},
            focusEditor: () => ({{ focus: () => events.push(["focus"]) }}),
            disposeOpenRender: () => events.push(["disposeOpenRender"]),
            currentFileViewMode: () => state.viewMode,
            activeFileEntry: () => state.activeEntry,
            fileCandidateGitStateFresh: () => state.gitFresh,
            isMarkdownPreviewable: () => state.markdownPreviewable,
          }});
        }}
        function event() {{
          return {{
            preventDefault() {{ events.push(["preventDefault"]); }},
            stopPropagation() {{ events.push(["stopPropagation"]); }},
          }};
        }}
        async function runReloadCase(openEffect, confirmResult = true) {{
          state.sessionId = "sid-1";
          state.path = "src/app.py";
          state.line = 7;
          state.gitPath = true;
          state.apiPath = "api-token";
          state.unavailable = false;
          state.confirmResult = confirmResult;
          state.openResult = false;
          state.openEffect = openEffect || "";
          events.length = 0;
          fileStatus.textContent = "";
          fileStatus.children = [];
          const controller = makeController();
          controller.setActiveFileIdentity("src/app.py", {{ line: state.line, gitPath: state.gitPath, apiPath: state.apiPath }});
          controller.renderSaveConflict("sid-1", "src/app.py", "version mismatch");
          const actions = fileStatus.children[1];
          const reloadBtn = actions.children[0];
          await reloadBtn.onclick(event());
          return {{
            status: fileStatus.textContent,
            events: events.slice(),
            current: controller.isSaveConflictCurrent(controller.currentSaveConflict()),
            conflict: controller.currentSaveConflict(),
          }};
        }}
        async function runKeepCase(stale, unavailable = false) {{
          state.sessionId = "sid-1";
          state.path = "src/app.py";
          state.unavailable = unavailable;
          events.length = 0;
          fileStatus.textContent = "";
          fileStatus.children = [];
          const controller = makeController();
          controller.setActiveFileIdentity("src/app.py", {{ line: state.line, gitPath: state.gitPath, apiPath: state.apiPath }});
          controller.renderSaveConflict("sid-1", "src/app.py", "version mismatch");
          if (stale) state.sessionId = "sid-2";
          const actions = fileStatus.children[1];
          const keepBtn = actions.children[1];
          await keepBtn.onclick(event());
          return {{ status: fileStatus.textContent, events: events.slice() }};
        }}
        (async () => {{
          const renderController = makeController();
          renderController.setActiveFileIdentity("src/app.py", {{ line: 7, gitPath: true, apiPath: "api-token" }});
          const conflict = renderController.renderSaveConflict("sid-1", "src/app.py", "version mismatch");
          state.viewMode = "diff";
          state.gitFresh = true;
          state.activeEntry = {{ changed: false }};
          const diffFallback = renderController.resolveFileOpenViewMode({{ gitPath: true }}, "src/app.py");
          state.activeEntry = {{ changed: true }};
          const diffAllowed = renderController.resolveFileOpenViewMode({{ gitPath: true }}, "src/app.py");
          state.viewMode = "preview";
          state.markdownPreviewable = false;
          const previewFallback = renderController.resolveFileOpenViewMode({{ gitPath: false }}, "src/app.py");
          const explicitDiff = renderController.resolveFileOpenViewMode({{ gitPath: false }}, "src/app.py", "diff");
          let invalidModeMessage = "";
          try {{ renderController.normalizeExplicitFileOpenMode("bogus"); }} catch (err) {{ invalidModeMessage = err && err.message ? err.message : String(err); }}
          const render = {{
            exportFrozen: Object.isFrozen(fileViewer),
            exports: Object.keys(fileViewer).sort(),
            conflict,
            currentConflict: renderController.currentSaveConflict(),
            labelText: fileStatus.children[0].text,
            actionTexts: fileStatus.children[1].children.map((node) => node.text),
            modeResolution: {{ diffFallback, diffAllowed, previewFallback, explicitDiff, invalidModeMessage }},
          }};
          const availableReloadFailure = await runReloadCase("");
          const canceledReload = await runReloadCase("", false);
          const unavailableReloadFailure = await runReloadCase("unavailable");
          const switchedSessionReloadFailure = await runReloadCase("switch-session");
          const keepCurrent = await runKeepCase(false);
          const keepStale = await runKeepCase(true);
          const keepUnavailable = await runKeepCase(false, true);
          process.stdout.write(JSON.stringify({{
            render,
            missingDependencyError,
            availableReloadFailure,
            canceledReload,
            unavailableReloadFailure,
            switchedSessionReloadFailure,
            keepCurrent,
            keepStale,
            keepUnavailable,
          }}));
        }})().catch((err) => {{ console.error(err && err.stack ? err.stack : err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendFileViewerModuleSource(unittest.TestCase):
    def test_file_viewer_controller_owns_save_conflict_behavior(self) -> None:
        result = run_file_viewer_controller_probe()
        self.assertTrue(result["render"]["exportFrozen"])
        self.assertEqual(result["render"]["exports"], ["createFileViewerController"])
        self.assertEqual(result["render"]["conflict"], {"sessionId": "sid-1", "path": "src/app.py"})
        self.assertEqual(result["render"]["currentConflict"], {"sessionId": "sid-1", "path": "src/app.py"})
        self.assertEqual(result["render"]["labelText"], "src/app.py - save conflict: version mismatch")
        self.assertEqual(result["render"]["actionTexts"], ["Reload from disk", "Keep editing"])
        self.assertEqual(result["render"]["modeResolution"], {
            "diffFallback": "file",
            "diffAllowed": "diff",
            "previewFallback": "file",
            "explicitDiff": "diff",
            "invalidModeMessage": "invalid file open mode",
        })
        self.assertIn("file viewer dependency missing: el", result["missingDependencyError"])

        available = result["availableReloadFailure"]
        self.assertEqual(available["status"], "src/app.py - reload failed")
        self.assertIn(["confirm", "Reload src/app.py from disk and discard your unsaved editor draft?"], available["events"])
        self.assertIn(["open", "src/app.py", {"line": 7, "gitPath": True, "apiPath": "api-token"}], available["events"])
        self.assertIn(["preventDefault"], available["events"])
        self.assertIn(["stopPropagation"], available["events"])
        self.assertTrue(available["current"])

        canceled = result["canceledReload"]
        self.assertEqual(canceled["status"], "")
        self.assertFalse(any(event[0] == "open" for event in canceled["events"]))

        unavailable = result["unavailableReloadFailure"]
        self.assertEqual(unavailable["status"], "Session is no longer available; copy unsaved edits before closing.")
        self.assertFalse(unavailable["current"])

        switched = result["switchedSessionReloadFailure"]
        self.assertEqual(switched["status"], "Reloading src/app.py...")
        self.assertFalse(switched["current"])

        keep_current = result["keepCurrent"]
        self.assertEqual(keep_current["status"], "src/app.py - editing unsaved conflict")
        self.assertIn(["focus"], keep_current["events"])
        keep_stale = result["keepStale"]
        self.assertEqual(keep_stale["status"], "")
        self.assertNotIn(["focus"], keep_stale["events"])
        keep_unavailable = result["keepUnavailable"]
        self.assertEqual(keep_unavailable["status"], "")
        self.assertNotIn(["focus"], keep_unavailable["events"])

    def test_active_file_identity_state_lives_in_file_viewer_module(self) -> None:
        app_source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        for declaration in [
            "let activeFilePath =",
            "let activeFileApiPath =",
            "let activeFileGitPath =",
            "let activeFileLine =",
        ]:
            self.assertNotIn(declaration, app_source)
            self.assertIn(declaration, viewer_source)
        for api_name in [
            "nextActiveFileIdentity",
            "currentActiveFileIdentity",
            "currentActiveFileLine",
            "clearActiveFileIdentity",
            "setActiveFileIdentity",
            "beginActiveFileIdentity",
            "abortPendingFileOpenTransport",
            "cancelPendingFileOpen",
            "beginFileOpenRequest",
            "isCurrentFileOpenRequest",
            "finalizeFileOpenRequest",
            "startFileOpenRequest",
        ]:
            self.assertIn(api_name, viewer_source)
        self.assertIn("normalizeLineNumber", app_source)
        self.assertIn("fileViewerController.currentActiveFileIdentity()", app_source)
        self.assertNotIn("activeFilePath: () => activeFilePath", app_source)

    def test_file_viewer_module_registered_before_app_js(self) -> None:
        index_source = INDEX_HTML.read_text(encoding="utf-8")
        routes_source = STATIC_ROUTES.read_text(encoding="utf-8")
        app_source = APP_JS.read_text(encoding="utf-8")
        self.assertLess(index_source.index("app_file_picker.js"), index_source.index("app_file_viewer.js"))
        self.assertLess(index_source.index("app_file_viewer.js"), index_source.index("app_session_helpers.js"))
        self.assertLess(index_source.index("app_file_viewer.js"), index_source.index("app.js"))
        self.assertIn('"app_file_viewer.js"', routes_source)
        self.assertIn("const codoxearFileViewer = window.CodoxearFileViewer;", app_source)
        self.assertIn('throw new Error("Codoxear file viewer controller failed to load")', app_source)
        self.assertIn("fileViewerController.renderSaveConflict", app_source)
        self.assertNotIn("function renderFileSaveConflict", app_source)


if __name__ == "__main__":
    unittest.main()
