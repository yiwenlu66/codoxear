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
          resetCount: 0,
          touchCount: 0,
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
            api: async (url, options = {{}}) => {{
              events.push(["api", url, Boolean(options.signal)]);
              if (url.includes("/git/file_versions")) return {{ base_text: "old", current_text: "new", base_exists: true, current_exists: false, abs_path: "/abs/diff" }};
              return {{ kind: "text", text: "body", path: "/abs/read" }};
            }},
            focusEditor: () => ({{ focus: () => events.push(["focus"]) }}),
            disposeOpenRender: () => events.push(["disposeOpenRender"]),
            currentFileViewMode: () => state.viewMode,
            activeFileEntry: () => state.activeEntry,
            fileCandidateGitStateFresh: () => state.gitFresh,
            isMarkdownPreviewable: () => state.markdownPreviewable,
            resetActiveFileBufferState: () => {{ state.resetCount += 1; events.push(["resetBuffer"]); }},
            updateFileTouchToolbar: () => {{ state.touchCount += 1; events.push(["touchToolbar"]); }},
            setFileViewMode: (mode) => {{ state.viewMode = mode; events.push(["setFileViewMode", mode]); }},
            applyActiveFileTextState: (nextState) => events.push(["applyActiveFileTextState", nextState]),
            renderMonacoFile: async (rel, text, line, lang) => {{ events.push(["renderMonacoFile", rel, text, line, lang]); return state.renderOk !== false; }},
            setFileEditMode: (enabled) => events.push(["setFileEditMode", enabled]),
            currentActiveFileKind: () => "text",
            currentActiveFileDraft: () => false,
            currentActiveFileVersion: () => "",
            currentActiveFileEditable: () => true,
            getFileEditorText: () => "",
            setFileDirty: () => events.push(["setFileDirty"]),
            syncFileEditorReadOnly: () => events.push(["syncFileEditorReadOnly"]),
            fmtBytes: (value) => `${{value}}B`,
            applyFileMode: () => events.push(["applyFileMode"]),
            rememberOpenedFile: (rel, absPath) => events.push(["rememberOpenedFile", rel, absPath]),
            rememberActiveFileSelection: () => events.push(["rememberActiveFileSelection"]),
            updateFileEditButton: () => events.push(["updateFileEditButton"]),
            renderFilePickerMenu: () => events.push(["renderFilePickerMenu"]),
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
          events.length = 0;
          const diffFetch = await renderController.fetchFileOpenResult({{ sessionId: "sid-1", apiPath: "tok", gitPath: true, signal: {{}} }}, "src/app.py", "diff");
          const diffFetchEvents = events.slice();
          events.length = 0;
          const readFetch = await renderController.fetchFileOpenResult({{ sessionId: "sid-1", apiPath: "tok", gitPath: true, signal: {{}} }}, "src/app.py", "file");
          const readFetchEvents = events.slice();
          state.sessionId = "sid-1";
          renderController.setActiveFileIdentity("src/app.py", {{ line: 7, gitPath: true, apiPath: "tok" }});
          const currentRequest = {{ requestId: 0, sessionId: "sid-1", path: "src/app.py", apiPath: "tok" }};
          events.length = 0;
          const currentErrorResult = renderController.renderFileOpenError(currentRequest, new Error("boom"));
          const currentError = {{ result: currentErrorResult, status: fileStatus.textContent, resetCount: state.resetCount, touchCount: state.touchCount, events: events.slice() }};
          events.length = 0;
          const abortError = new Error("aborted");
          abortError.name = "AbortError";
          const abortResult = renderController.renderFileOpenError(currentRequest, abortError);
          const abortErrorResult = {{ result: abortResult, status: fileStatus.textContent, resetCount: state.resetCount, touchCount: state.touchCount, events: events.slice(), abortCheck: renderController.isFileOpenAbortError(abortError) }};
          events.length = 0;
          state.sessionId = "sid-2";
          const staleResult = renderController.renderFileOpenError(currentRequest, new Error("stale"));
          const staleError = {{ result: staleResult, status: fileStatus.textContent, resetCount: state.resetCount, touchCount: state.touchCount, events: events.slice() }};
          state.sessionId = "sid-1";
          events.length = 0;
          const unknownResult = renderController.renderFileOpenError(currentRequest, {{}});
          const unknownError = {{ result: unknownResult, status: fileStatus.textContent, resetCount: state.resetCount, touchCount: state.touchCount, events: events.slice() }};
          events.length = 0;
          const finalizeResult = renderController.finalizeFileOpenSuccess("src/app.py", "/abs/src/app.py");
          const finalize = {{ result: finalizeResult, events: events.slice() }};
          state.viewMode = "preview";
          renderController.setActiveFileIdentity("draft/new.txt", {{ line: 5, gitPath: false, apiPath: "" }});
          events.length = 0;
          const draftRequest = {{ requestId: 0, sessionId: "sid-1", path: "draft/new.txt", apiPath: "", line: 5 }};
          const draftResult = await renderController.applyDraftFileLoad("draft/new.txt", draftRequest);
          const draft = {{ result: draftResult, status: fileStatus.textContent, viewMode: state.viewMode, events: events.slice() }};
          state.resetCount = 0;
          state.touchCount = 0;
          fileStatus.textContent = "";
          events.length = 0;
          const draftErrorResult = renderController.renderDraftFileOpenError(draftRequest, new Error("draft boom"));
          const currentDraftError = {{ result: draftErrorResult, status: fileStatus.textContent, resetCount: state.resetCount, touchCount: state.touchCount, events: events.slice() }};
          events.length = 0;
          const draftAbortError = new Error("aborted");
          draftAbortError.name = "AbortError";
          const draftAbortResult = renderController.renderDraftFileOpenError(draftRequest, draftAbortError);
          const abortDraftError = {{ result: draftAbortResult, status: fileStatus.textContent, resetCount: state.resetCount, touchCount: state.touchCount, events: events.slice() }};
          events.length = 0;
          state.sessionId = "sid-2";
          const draftStaleResult = renderController.renderDraftFileOpenError(draftRequest, new Error("stale"));
          const staleDraftError = {{ result: draftStaleResult, status: fileStatus.textContent, resetCount: state.resetCount, touchCount: state.touchCount, events: events.slice() }};
          state.sessionId = "sid-1";
          const saveBodies = {{
            draft: renderController.buildActiveFileSaveBody({{ path: "new.py", text: "NEW", draft: true, gitPath: true, version: "v1", apiPath: "tok" }}),
            gitToken: renderController.buildActiveFileSaveBody({{ path: "existing.py", text: "BODY", draft: false, gitPath: true, version: "v2", apiPath: "tok" }}),
            gitNoToken: renderController.buildActiveFileSaveBody({{ path: "existing.py", text: "BODY", draft: false, gitPath: true, version: "v2", apiPath: "" }}),
            plainToken: renderController.buildActiveFileSaveBody({{ path: "plain.py", text: "TEXT", draft: false, gitPath: false, version: "v3", apiPath: "tok" }}),
          }};
          renderController.renderActiveFileSaveError({{ sessionId: "sid-1", path: "src/app.py" }}, {{ status: 409, message: "version mismatch" }});
          const saveConflict = {{ label: fileStatus.children[0].text, actions: fileStatus.children[1].children.map((node) => node.text) }};
          renderController.renderActiveFileSaveError({{ sessionId: "sid-1", path: "src/app.py" }}, {{ status: 500, message: "disk full" }});
          const genericSaveError = fileStatus.textContent;
          renderController.renderActiveFileSaveError({{ sessionId: "sid-1", path: "src/app.py" }}, {{}});
          const unknownSaveError = fileStatus.textContent;
          state.unavailable = false;
          fileStatus.textContent = "old status";
          const availableBlocked = renderController.blockUnavailableFileAction();
          const availableBlockStatus = fileStatus.textContent;
          state.unavailable = true;
          const unavailableBlocked = renderController.blockUnavailableFileAction();
          const unavailableBlockStatus = fileStatus.textContent;
          state.unavailable = false;
          const render = {{
            exportFrozen: Object.isFrozen(fileViewer),
            exports: Object.keys(fileViewer).sort(),
            conflict,
            currentConflict: renderController.currentSaveConflict(),
            labelText: fileStatus.children[0].text,
            actionTexts: fileStatus.children[1].children.map((node) => node.text),
            modeResolution: {{ diffFallback, diffAllowed, previewFallback, explicitDiff, invalidModeMessage }},
            fetchResults: {{ diffFetch, diffFetchEvents, readFetch, readFetchEvents }},
            openErrors: {{ currentError, abortErrorResult, staleError, unknownError }},
            finalize,
            draft,
            draftErrors: {{ currentDraftError, abortDraftError, staleDraftError }},
            saveBodies,
            saveErrors: {{ saveConflict, genericSaveError, unknownSaveError }},
            unavailableAction: {{ availableBlocked, availableBlockStatus, unavailableBlocked, unavailableBlockStatus }},
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
        self.assertEqual(result["render"]["fetchResults"], {
            "diffFetch": {
                "result": {"kind": "diff", "baseText": "old", "currentText": "new", "baseExists": True, "currentExists": False},
                "absPath": "/abs/diff",
            },
            "diffFetchEvents": [["api", "/api/sessions/sid-1/git/file_versions?path=src%2Fapp.py&path_token=tok", True]],
            "readFetch": {"result": {"kind": "text", "text": "body", "path": "/abs/read"}, "absPath": "/abs/read"},
            "readFetchEvents": [["api", "/api/sessions/sid-1/file/read?path=src%2Fapp.py&path_token=tok&git_path=1", True]],
        })
        self.assertEqual(result["render"]["openErrors"], {
            "currentError": {"result": False, "status": "error: boom", "resetCount": 1, "touchCount": 1, "events": [["resetBuffer"], ["touchToolbar"]]},
            "abortErrorResult": {"result": False, "status": "error: boom", "resetCount": 1, "touchCount": 1, "events": [], "abortCheck": True},
            "staleError": {"result": False, "status": "error: boom", "resetCount": 1, "touchCount": 1, "events": []},
            "unknownError": {"result": False, "status": "error: unknown error", "resetCount": 2, "touchCount": 2, "events": [["resetBuffer"], ["touchToolbar"]]},
        })
        self.assertEqual(result["render"]["finalize"], {
            "result": True,
            "events": [
                ["applyFileMode"],
                ["rememberOpenedFile", "src/app.py", "/abs/src/app.py"],
                ["rememberActiveFileSelection"],
                ["updateFileEditButton"],
                ["renderFilePickerMenu"],
            ],
        })
        self.assertEqual(result["render"]["draft"], {
            "result": True,
            "status": "draft/new.txt - new file",
            "viewMode": "file",
            "events": [
                ["setFileViewMode", "file"],
                ["applyActiveFileTextState", {"text": "", "editable": True, "version": "", "draft": True}],
                ["applyFileMode"],
                ["renderMonacoFile", "draft/new.txt", "", 5, ""],
                ["setFileEditMode", True],
                ["rememberActiveFileSelection"],
                ["renderFilePickerMenu"],
            ],
        })
        self.assertEqual(result["render"]["draftErrors"], {
            "currentDraftError": {"result": False, "status": "error: draft boom", "resetCount": 1, "touchCount": 0, "events": [["resetBuffer"]]},
            "abortDraftError": {"result": False, "status": "error: draft boom", "resetCount": 1, "touchCount": 0, "events": []},
            "staleDraftError": {"result": False, "status": "error: draft boom", "resetCount": 1, "touchCount": 0, "events": []},
        })
        self.assertEqual(result["render"]["saveBodies"], {
            "draft": {"path": "new.py", "text": "NEW", "create": True},
            "gitToken": {"path": "existing.py", "text": "BODY", "version": "v2", "git_path": True, "path_token": "tok"},
            "gitNoToken": {"path": "existing.py", "text": "BODY", "version": "v2", "git_path": True},
            "plainToken": {"path": "plain.py", "text": "TEXT", "version": "v3", "git_path": False},
        })
        self.assertEqual(result["render"]["saveErrors"], {
            "saveConflict": {"label": "src/app.py - save conflict: version mismatch", "actions": ["Reload from disk", "Keep editing"]},
            "genericSaveError": "save error: disk full",
            "unknownSaveError": "save error: unknown error",
        })
        self.assertEqual(result["render"]["unavailableAction"], {
            "availableBlocked": False,
            "availableBlockStatus": "old status",
            "unavailableBlocked": True,
            "unavailableBlockStatus": "Session is no longer available; copy unsaved edits before closing.",
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
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertLess(index_source.index("app_file_picker.js"), index_source.index("app_file_viewer.js"))
        self.assertLess(index_source.index("app_file_viewer.js"), index_source.index("app_session_helpers.js"))
        self.assertLess(index_source.index("app_file_viewer.js"), index_source.index("app.js"))
        self.assertIn('"app_file_viewer.js"', routes_source)
        self.assertIn("const codoxearFileViewer = window.CodoxearFileViewer;", app_source)
        self.assertIn('throw new Error("Codoxear file viewer controller failed to load")', app_source)
        self.assertIn("renderSaveConflict", viewer_source)
        self.assertNotIn("function renderFileSaveConflict", app_source)


if __name__ == "__main__":
    unittest.main()
