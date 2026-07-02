import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_FILE_VIEWER_JS = ROOT / "codoxear" / "static" / "app_file_viewer.js"
APP_FILE_EDITOR_JS = ROOT / "codoxear" / "static" / "app_file_editor.js"
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
          markdownPreviewable: true,
          kind: "text",
          draft: false,
          version: "",
          editable: true,
          dirty: false,
          editorKind: "file",
          editMode: true,
          resetCount: 0,
          touchCount: 0,
          selectionText: "",
          copyError: "",
          recordFocus: false,
        }};
        const events = [];
        const fileStatus = {{
          textContent: "",
          children: [],
          replaceChildren(...nodes) {{ this.children = nodes; this.textContent = ""; events.push(["replaceChildren", nodes.map((node) => node.tag)]); }},
        }};
        const fileEditButton = {{
          disabled: false,
          innerHTML: "",
          title: "",
          attrs: {{}},
          classes: {{}},
          classList: {{ toggle(name, enabled) {{ if (enabled) fileEditButton.classes[name] = true; else delete fileEditButton.classes[name]; events.push(["buttonClass", name, Boolean(enabled)]); }} }},
          setAttribute(name, value) {{ this.attrs[name] = String(value); events.push(["buttonAttr", name, String(value)]); }},
        }};
        function resetFileEditButton() {{
          fileEditButton.disabled = false;
          fileEditButton.innerHTML = "";
          fileEditButton.title = "";
          fileEditButton.attrs = {{}};
          fileEditButton.classes = {{}};
        }}
        function fileEditButtonSnapshot() {{
          return {{
            disabled: Boolean(fileEditButton.disabled),
            innerHTML: fileEditButton.innerHTML,
            title: fileEditButton.title,
            ariaLabel: fileEditButton.attrs["aria-label"] || "",
            classes: Object.keys(fileEditButton.classes).sort(),
          }};
        }}
        function el(tag, attrs = {{}}, children = []) {{
          const node = {{ tag, attrs, children: Array.isArray(children) ? children : [], onclick: null }};
          if (attrs && Object.prototype.hasOwnProperty.call(attrs, "text")) node.text = attrs.text;
          return node;
        }}
        function makeController() {{
          let controller = null;
          controller = fileViewer.createFileViewerController({{
            el,
            fileStatus,
            fileEditButton,
            iconSvg: (name) => `icon:${{name}}`,
            currentSessionId: () => state.sessionId,
            currentFileSessionId: () => state.sessionId,
            normalizeLineNumber: (value) => value == null || value === "" ? null : Number(value),
            normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
            fileApiPathForPath: (_path, existing) => existing || "derived-token",
            isFileViewerOpen: () => state.viewerOpen !== false,
            hideFileUnsavedDialog: (choice) => events.push(["hideFileUnsavedDialog", choice]),
            resetFileSearchState: () => events.push(["resetFileSearchState"]),
            closeFilePickerMenu: (options) => events.push(["closeFilePickerMenu", options]),
            isTextFileKind: (kind) => kind === "text" || kind === "markdown",
            isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
            confirmReload: (message) => {{ events.push(["confirm", message]); return state.confirmResult !== false; }},
            promptUnsavedFileChoice: async () => {{ events.push(["promptUnsaved", state.unsavedChoice || "cancel"]); return state.unsavedChoice || "cancel"; }},
            restoreFileEditorText: (text) => events.push(["restoreFileEditorText", text]),
            hideFileViewer: () => events.push(["hideFileViewer"]),
            applyFileLoadResult: async (rel, result, request, opts) => {{
              events.push(["applyFileLoadResult", rel, result && result.kind, opts && opts.viewMode]);
              if (state.openEffect === "unavailable") {{
                if (controller) controller.disableFileViewerForUnavailableSession(state.sessionId);
              }} else if (state.openEffect === "switch-session") {{
                state.sessionId = "sid-2";
                state.path = "src/app.py";
              }}
              return state.openResult !== false;
            }},
            setFilePath: (path, opts) => events.push(["setFilePath", path, opts]),
            resetFileViewerPanel: () => events.push(["resetFileViewerPanel"]),
            normalizeDraftFilePath: (value) => String(value || "").trim().replace(/^[/]+/, ""),
            inspectSessionFilePath: async (path) => {{ events.push(["inspect", path]); if (state.inspectError) throw new Error(state.inspectError); return state.inspectResult || {{ exists: false }}; }},
            api: async (url, options = {{}}) => {{
              events.push(["api", url, Boolean(options.signal)]);
              if (url.includes("/git/file_versions")) return {{ base_text: "old", current_text: "new", base_exists: true, current_exists: false, abs_path: "/abs/diff" }};
              return {{ kind: "text", text: "body", path: "/abs/read" }};
            }},
            focusEditor: () => ({{ focus: () => events.push(["focus"]), updateOptions: (opts) => events.push(["editorOptions", opts]) }}),
            disposeOpenRender: () => events.push(["disposeOpenRender"]),
            initialFileViewMode: state.viewMode,
            initialFileNonDiffMode: state.viewMode === "preview" ? "preview" : "file",
            persistFileViewMode: (mode) => {{ state.viewMode = mode; events.push(["persistFileViewMode", mode]); }},
            persistFileNonDiffMode: (mode) => events.push(["persistFileNonDiffMode", mode]),
            currentFileEditorKind: () => state.editorKind || "file",
            currentFileEditMode: () => state.editMode !== false,
            isMarkdownPreviewable: () => state.markdownPreviewable,
            resetActiveFileBufferState: () => {{ state.resetCount += 1; events.push(["resetBuffer"]); }},
            updateFileTouchToolbar: () => {{ state.touchCount += 1; events.push(["touchToolbar"]); }},
            currentFileTouchSelectMode: () => state.touchSelectMode !== false,
            isFileTouchToolbarActive: () => state.touchToolbarActive !== false,
            fileEditorShortcutBlocked: (target) => Boolean(target && target.shortcutBlocked),
            eventTargetElement: (value) => value || null,
            normalizeFileEditorPosition: (_editor, position) => position ? {{ lineNumber: Number(position.lineNumber) || 1, column: Number(position.column) || 1 }} : null,
            applyFileEditorSelection: () => {{}},
            isCollapsedFileSelection: (selection) => !selection || (selection.startLineNumber === selection.endLineNumber && selection.startColumn === selection.endColumn),
            positionAfterInsertedText: (start, text) => ({{ lineNumber: Number(start && start.lineNumber) || 1, column: (Number(start && start.column) || 1) + String(text || "").length }}),
            fileEditorEditSupportAvailable: () => true,
            syncFileDiffSelectionMode: () => {{}},
            showFilePasteDialog: () => false,
            hideFilePasteDialog: () => {{}},
            clipboardReadAvailable: () => false,
            readClipboardText: async () => "",
            resetFileTouchSelectionState: (options) => events.push(["resetFileTouchSelectionState", options || {{}}]),
            moveFileTouchSelection: (direction) => events.push(["moveFileTouchSelection", direction]),
            fileEditorDeleteCommandForKey: () => "",
            isActiveFileEditorInput: () => false,
            getActiveFileSelectionText: () => state.selectionText,
            copyToClipboard: async (text) => {{ events.push(["copyToClipboard", text]); if (state.copyError) throw new Error(state.copyError); }},
            focusActiveFileCodeEditor: () => {{ if (state.recordFocus) events.push(["focusActiveFileCodeEditor"]); return null; }},
            setFileTouchDeleteNativeSuppressUntil: () => {{}},
            nowMs: () => 0,
            setToast: (message) => events.push(["toast", message]),
            setFileViewMode: (mode) => {{ state.viewMode = mode; events.push(["setFileViewMode", mode]); }},
            applyActiveFileTextState: (nextState) => events.push(["applyActiveFileTextState", nextState]),
            renderMonacoFile: async (rel, text, line, lang) => {{ events.push(["renderMonacoFile", rel, text, line, lang]); return state.renderOk !== false; }},
            setFileEditMode: (enabled) => {{ state.editMode = Boolean(enabled); events.push(["setFileEditMode", enabled]); }},
            currentActiveFileKind: () => state.kind,
            currentActiveFileDraft: () => state.draft,
            currentActiveFileVersion: () => state.version,
            currentActiveFileEditable: () => state.editable,
            currentFileDirty: () => state.dirty,
            currentActiveFileText: () => "",
            getFileEditorText: () => "",
            setFileDirty: () => events.push(["setFileDirty"]),
            fmtBytes: (value) => `${{value}}B`,
            applyFileMode: () => events.push(["applyFileMode"]),
            rememberOpenedFile: (rel, absPath) => events.push(["rememberOpenedFile", rel, absPath]),
            historyFileSelectionForSession: () => state.historySelection || ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
            rememberActiveFileSelection: () => events.push(["rememberActiveFileSelection"]),
            renderFilePickerMenu: () => events.push(["renderFilePickerMenu"]),
          }});
          return controller;
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
          if (unavailable) controller.disableFileViewerForUnavailableSession("sid-1");
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
          renderController.setFileViewMode("diff");
          renderController.applyFileCandidateEntries([{{ path: "src/app.py", gitPath: true, changed: false, apiPath: "api-token" }}]);
          renderController.setFileCandidateGitStateFresh(true);
          const diffFallback = renderController.resolveFileOpenViewMode({{ gitPath: true }}, "src/app.py");
          renderController.applyFileCandidateEntries([{{ path: "src/app.py", gitPath: true, changed: true, apiPath: "api-token" }}]);
          const diffAllowed = renderController.resolveFileOpenViewMode({{ gitPath: true }}, "src/app.py");
          renderController.setFileViewMode("preview");
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
          const downloadController = makeController();
          downloadController.clearFileViewerUnavailableSession();
          downloadController.setActiveFileIdentity("src/app.py", {{ line: 7, gitPath: true, apiPath: "tok space" }});
          const downloadGitToken = downloadController.activeFileDownloadApiPath();
          downloadController.setActiveFileIdentity("plain.txt", {{ line: 1, gitPath: false, apiPath: "plain-token" }});
          const downloadPlain = downloadController.activeFileDownloadApiPath();
          state.sessionId = "";
          const downloadMissingSession = downloadController.activeFileDownloadApiPath();
          state.sessionId = "sid-1";
          downloadController.disableFileViewerForUnavailableSession("sid-1");
          fileStatus.textContent = "old download status";
          const downloadUnavailablePath = downloadController.activeFileDownloadApiPath();
          const downloadUnavailableStatus = fileStatus.textContent;
          downloadController.clearFileViewerUnavailableSession();
          const downloadPaths = {{ downloadGitToken, downloadPlain, downloadMissingSession, downloadUnavailablePath, downloadUnavailableStatus }};
          const kindController = makeController();
          const editorKindTransitions = {{ initial: kindController.currentFileEditorKind() }};
          editorKindTransitions.file = kindController.setFileEditorKind("file");
          editorKindTransitions.currentFile = kindController.currentFileEditorKind();
          editorKindTransitions.diff = kindController.setFileEditorKind("diff");
          editorKindTransitions.plain = kindController.setFileEditorKind("plain-fallback");
          editorKindTransitions.clear = kindController.setFileEditorKind("");
          try {{ kindController.setFileEditorKind("bogus"); }} catch (err) {{ editorKindTransitions.invalidMessage = err && err.message ? err.message : String(err); }}
          const restoreController = makeController();
          restoreController.setFileDirty(true);
          const skipRestorePlan = restoreController.prepareFileEditorTextRestore("skip-text");
          const skipRestoreDirty = restoreController.currentFileDirty();
          restoreController.setFileDirty(true);
          restoreController.setFileEditorKind("file");
          const restorePlan = restoreController.prepareFileEditorTextRestore("body text");
          const restoreDirtyBeforeFinish = restoreController.currentFileDirty();
          restoreController.finishFileEditorTextRestore();
          const restoreDirtyAfterFinish = restoreController.currentFileDirty();
          const restorePlanning = {{ skip: skipRestorePlan, skipFrozen: Object.isFrozen(skipRestorePlan), skipDirty: skipRestoreDirty, restore: restorePlan, restoreFrozen: Object.isFrozen(restorePlan), dirtyBeforeFinish: restoreDirtyBeforeFinish, dirtyAfterFinish: restoreDirtyAfterFinish }};
          const selectionController = makeController();
          state.historySelection = {{ path: "history.txt", line: 3, gitPath: false, apiPath: "" }};
          const selectionNoSession = selectionController.preferredFileSelectionForSession("");
          const selectionHistory = selectionController.preferredFileSelectionForSession("sid-history");
          selectionController.setActiveFileIdentity("src/app.py", {{ line: 12, gitPath: true, apiPath: "tok-1" }});
          selectionController.rememberActiveFileSelection("sid-remembered");
          const selectionRemembered = selectionController.preferredFileSelectionForSession("sid-remembered");
          state.historySelection = null;
          const selectionMemory = {{ noSession: selectionNoSession, history: selectionHistory, remembered: selectionRemembered }};
          const candidateSeqController = makeController();
          const firstCandidateSeq = candidateSeqController.beginFileCandidateRefresh();
          const firstCandidateCurrent = candidateSeqController.isCurrentFileCandidateRefresh(firstCandidateSeq);
          const secondCandidateSeq = candidateSeqController.beginFileCandidateRefresh();
          const candidateRefreshSeq = {{ first: firstCandidateSeq, second: secondCandidateSeq, firstCurrent: firstCandidateCurrent, firstAfterSecond: candidateSeqController.isCurrentFileCandidateRefresh(firstCandidateSeq), secondCurrent: candidateSeqController.isCurrentFileCandidateRefresh(secondCandidateSeq) }};
          const sessionSyncController = makeController();
          const firstSessionSync = sessionSyncController.beginFileViewerSessionSync();
          const firstSessionCurrent = sessionSyncController.isCurrentFileViewerSessionSync(firstSessionSync);
          const invalidatedSessionSync = sessionSyncController.invalidateFileViewerSessionSync();
          const sessionSyncSeq = {{ first: firstSessionSync, invalidated: invalidatedSessionSync, firstCurrent: firstSessionCurrent, firstAfterInvalidate: sessionSyncController.isCurrentFileViewerSessionSync(firstSessionSync), invalidatedCurrent: sessionSyncController.isCurrentFileViewerSessionSync(invalidatedSessionSync) }};
          state.editMode = true;
          state.dirty = false;
          renderController.setFileDirty(false);
          state.resetCount = 0;
          state.touchCount = 0;
          fileStatus.textContent = "";
          events.length = 0;
          renderController.setActiveFileIdentity("src/app.py", {{ line: 7, gitPath: true, apiPath: "tok" }});
          renderController.setFileEditorKind("file");
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
          state.editMode = true;
          state.dirty = false;
          renderController.setFileDirty(false);
          events.length = 0;
          const finalizeResult = renderController.finalizeFileOpenSuccess("src/app.py", "/abs/src/app.py");
          const finalize = {{ result: finalizeResult, events: events.slice() }};
          renderController.setFileViewMode("preview");
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
          renderController.clearFileViewerUnavailableSession();
          fileStatus.textContent = "old status";
          const availableBlocked = renderController.blockUnavailableFileAction();
          const availableBlockStatus = fileStatus.textContent;
          events.length = 0;
          renderController.disableFileViewerForUnavailableSession("sid-1");
          const unavailableTransitionEvents = events.slice();
          fileStatus.textContent = "old unavailable status";
          const unavailableBlocked = renderController.blockUnavailableFileAction();
          const unavailableBlockStatus = fileStatus.textContent;
          const unsavedChoiceController = makeController();
          unsavedChoiceController.clearFileViewerUnavailableSession();
          state.sessionId = "sid-1";
          events.length = 0;
          fileStatus.textContent = "old unsaved status";
          const unsavedSaveAvailable = {{ result: unsavedChoiceController.handleFileUnsavedSaveChoice(), status: fileStatus.textContent, events: events.slice() }};
          unsavedChoiceController.disableFileViewerForUnavailableSession("sid-1");
          events.length = 0;
          fileStatus.textContent = "old unsaved status";
          const unsavedSaveUnavailable = {{ result: unsavedChoiceController.handleFileUnsavedSaveChoice(), status: fileStatus.textContent, events: events.slice() }};
          unsavedChoiceController.clearFileViewerUnavailableSession();
          events.length = 0;
          const unsavedDiscard = {{ result: unsavedChoiceController.handleFileUnsavedDiscardChoice(), events: events.slice() }};
          events.length = 0;
          const unsavedCancel = {{ result: unsavedChoiceController.handleFileUnsavedCancelChoice(), events: events.slice() }};
          const unsavedChoices = {{ unsavedSaveAvailable, unsavedSaveUnavailable, unsavedDiscard, unsavedCancel }};
          renderController.clearFileViewerUnavailableSession();
          state.editMode = true;
          state.dirty = false;
          renderController.applyActiveFileTextState({{ kind: "markdown", text: "body", editable: true, version: "v7", draft: false }});
          renderController.setFileEditMode(true);
          renderController.setFileDirty(false);
          async function runCopySelectionCase({{ text = "", error = "" }} = {{}}) {{
            state.selectionText = text;
            state.copyError = error;
            state.recordFocus = true;
            events.length = 0;
            const result = await renderController.copyActiveFileSelection();
            const output = {{ result, events: events.slice() }};
            state.selectionText = "";
            state.copyError = "";
            state.recordFocus = false;
            return output;
          }}
          const copyNoSelection = await runCopySelectionCase();
          const copySelectionSuccess = await runCopySelectionCase({{ text: "selected text" }});
          const copySelectionError = await runCopySelectionCase({{ text: "selected text", error: "denied" }});
          state.resetCount = 0;
          state.touchCount = 0;
          events.length = 0;
          renderController.clearFileViewerUnavailableSession();
          state.kind = "markdown";
          state.version = "v7";
          state.editable = true;
          state.draft = false;
          renderController.setFileViewMode("file");
          state.editorKind = "file";
          state.editMode = true;
          state.dirty = true;
          renderController.setFileDirty(true);
          renderController.setActiveFileIdentity("state.md", {{ line: 11, gitPath: true, apiPath: "state-token" }});
          renderController.setFileEditMode(true);
          const editorState = renderController.currentFileEditorState();
          const editorStateFrozen = Object.isFrozen(editorState);
          const derivedCapabilities = {{
            capabilities: renderController.activeFileEditorCapabilities(),
            canEnter: renderController.activeFileCanEnterEditMode(),
            writable: renderController.activeFileEditorWritable(),
            idleWritable: renderController.activeFileEditorIdleWritable(),
            idleTextWritable: renderController.activeFileEditorIdleTextWritable(),
            editModeAllowed: renderController.activeFileEditModeAllowedInCurrentView(),
          }};
          events.length = 0;
          resetFileEditButton();
          renderController.updateFileEditButton();
          const editButtonEditMode = {{ button: fileEditButtonSnapshot(), events: events.slice() }};
          events.length = 0;
          renderController.syncFileEditorReadOnly();
          const readOnlyWritable = events.slice();
          state.editMode = false;
          renderController.setFileEditMode(false);
          state.dirty = false;
          renderController.setFileDirty(false);
          events.length = 0;
          resetFileEditButton();
          renderController.updateFileEditButton();
          renderController.syncFileEditorReadOnly();
          const editButtonViewMode = {{ button: fileEditButtonSnapshot(), events: events.slice() }};
          state.editMode = true;
          state.dirty = true;
          renderController.setFileDirty(true);
          renderController.disableFileViewerForUnavailableSession("sid-1");
          events.length = 0;
          resetFileEditButton();
          renderController.updateFileEditButton();
          const editButtonUnavailable = {{ button: fileEditButtonSnapshot(), events: events.slice() }};
          renderController.clearFileViewerUnavailableSession();
          state.editMode = true;
          renderController.setFileEditMode(true);
          events.length = 0;
          resetFileEditButton();
          renderController.markActiveFileSavePending({{ path: "state.md" }});
          const editButtonSavePending = {{ button: fileEditButtonSnapshot(), events: events.slice(), status: fileStatus.textContent }};
          renderController.clearActiveFileSaveState();
          state.dirty = false;
          renderController.setFileDirty(false);
          events.length = 0;
          const cleanUnsaved = {{ result: await renderController.maybeHandleUnsavedFileChanges(), events: events.slice() }};
          state.dirty = true;
          renderController.setFileDirty(true);
          state.unsavedChoice = "discard";
          events.length = 0;
          const discardUnsaved = {{ result: await renderController.maybeHandleUnsavedFileChanges(), events: events.slice() }};
          state.unsavedChoice = "cancel";
          events.length = 0;
          const cancelUnsaved = {{ result: await renderController.maybeHandleUnsavedFileChanges(), events: events.slice() }};
          state.unsavedChoice = "";
          async function runViewModeCase({{ startMode = "file", target = "preview", draft = false, dirty = false, choice = "", unavailable = false }} = {{}}) {{
            renderController.setFileViewMode(startMode);
            state.draft = draft;
            state.dirty = dirty;
            state.unsavedChoice = choice;
            renderController.clearFileViewerUnavailableSession();
            if (unavailable) renderController.disableFileViewerForUnavailableSession("sid-1");
            events.length = 0;
            fileStatus.textContent = "";
            renderController.setActiveFileIdentity("state.md", {{ line: 11, gitPath: true, apiPath: "state-token" }});
            renderController.applyActiveFileTextState({{ kind: "markdown", text: "body", editable: true, version: "v7", draft }});
            if (dirty) renderController.setFileEditMode(true);
            else renderController.setFileEditMode(false);
            renderController.setFileDirty(dirty);
            events.length = 0;
            const result = await renderController.setFileViewModeWithGuard(target);
            const output = {{ result, viewMode: state.viewMode, status: fileStatus.textContent, events: events.slice() }};
            renderController.clearFileViewerUnavailableSession();
            state.unsavedChoice = "";
            state.draft = false;
            return output;
          }}
          const viewModeSame = await runViewModeCase({{ startMode: "file", target: "file", dirty: true, choice: "discard" }});
          const viewModeDraftBlocked = await runViewModeCase({{ startMode: "file", target: "diff", draft: true }});
          const viewModeDiscardOpen = await runViewModeCase({{ startMode: "file", target: "preview", dirty: true, choice: "discard" }});
          const viewModeCancel = await runViewModeCase({{ startMode: "file", target: "preview", dirty: true, choice: "cancel" }});
          const viewModeUnavailable = await runViewModeCase({{ startMode: "file", target: "preview", unavailable: true }});
          renderController.setFileViewMode("file");
          state.dirty = false;
          renderController.setFileEditMode(false);
          renderController.setFileDirty(false);
          events.length = 0;
          const hideClean = {{ result: await renderController.requestHideFileViewer(), events: events.slice() }};
          state.dirty = true;
          renderController.setFileEditMode(true);
          renderController.setFileDirty(true);
          state.unsavedChoice = "cancel";
          events.length = 0;
          const hideCancel = {{ result: await renderController.requestHideFileViewer(), events: events.slice() }};
          state.unsavedChoice = "discard";
          events.length = 0;
          const hideDiscard = {{ result: await renderController.requestHideFileViewer(), events: events.slice() }};
          state.unsavedChoice = "";
          function runUnavailableHandlerCase({{ sid = "sid-1", viewerOpen = true, dirty = false, currentSession = "sid-1" }} = {{}}) {{
            renderController.clearFileViewerUnavailableSession();
            state.sessionId = currentSession;
            state.viewerOpen = viewerOpen;
            state.dirty = dirty;
            renderController.setFileDirty(dirty);
            state.editMode = true;
            events.length = 0;
            fileStatus.textContent = "";
            const result = renderController.handleFileViewerSessionUnavailable(sid);
            const output = {{ result, unavailable: renderController.isFileViewerSessionUnavailable(), status: fileStatus.textContent, events: events.slice() }};
            state.viewerOpen = true;
            state.sessionId = "sid-1";
            renderController.clearFileViewerUnavailableSession();
            return output;
          }}
          const unavailableHandleClean = runUnavailableHandlerCase({{ dirty: false }});
          const unavailableHandleDirty = runUnavailableHandlerCase({{ dirty: true }});
          const unavailableHandleMismatch = runUnavailableHandlerCase({{ sid: "sid-2", currentSession: "sid-1", dirty: true }});
          const unavailableHandleClosed = runUnavailableHandlerCase({{ viewerOpen: false, dirty: true }});
          async function runDraftGuardCase({{ path = "draft/new.txt", inspectResult = {{ exists: false }}, inspectError = "", dirty = false, choice = "" }} = {{}}) {{
            renderController.clearFileViewerUnavailableSession();
            state.inspectResult = inspectResult;
            state.inspectError = inspectError;
            state.dirty = dirty;
            renderController.setFileDirty(dirty);
            state.unsavedChoice = choice;
            renderController.setFileViewMode("diff");
            events.length = 0;
            fileStatus.textContent = "";
            const result = await renderController.openDraftFilePathWithGuard(path);
            const output = {{ result, status: fileStatus.textContent, viewMode: state.viewMode, events: events.slice() }};
            state.inspectResult = null;
            state.inspectError = "";
            state.unsavedChoice = "";
            return output;
          }}
          const draftInvalidPath = await runDraftGuardCase({{ path: "/" }});
          const draftDirectory = await runDraftGuardCase({{ inspectResult: {{ exists: true, kind: "directory" }} }});
          const draftExisting = await runDraftGuardCase({{ inspectResult: {{ exists: true, kind: "text" }} }});
          const draftInspectError = await runDraftGuardCase({{ inspectError: "inspect boom" }});
          const draftNew = await runDraftGuardCase({{ inspectResult: {{ exists: false }} }});
          state.dirty = true;
          const editableCapabilities = renderController.fileEditorCapabilities({{ path: "src/app.py", kind: "markdown", editable: true, unavailable: false, viewMode: "file", editorKind: "file", editMode: true, savePending: false }});
          const pendingCapabilities = renderController.fileEditorCapabilities({{ path: "src/app.py", kind: "markdown", editable: true, unavailable: false, viewMode: "file", editorKind: "file", editMode: true, savePending: true }});
          const binaryCapabilities = renderController.fileEditorCapabilities({{ path: "img.png", kind: "image", editable: true, unavailable: false, viewMode: "file", editorKind: "file", editMode: true, savePending: false }});
          const missingPathCapabilities = renderController.fileEditorCapabilities({{ path: "", kind: "markdown", editable: true, unavailable: false, viewMode: "file", editorKind: "file", editMode: true, savePending: false }});
          function runModeControlState({{ path = "state.md", viewMode = "file", kind = "markdown", draft = false, gitPath = true, gitFresh = false, changed = false, editMode = false, videoPreviewAvailable = false, videoPreviewPreparing = false }} = {{}}) {{
            renderController.setFileViewMode(editMode ? "file" : viewMode);
            renderController.applyFileCandidateEntries(changed ? [{{ path, gitPath, changed: true, apiPath: "tok" }}] : []);
            renderController.setFileCandidateGitStateFresh(gitFresh);
            renderController.setActiveFileIdentity(path, {{ gitPath, apiPath: "tok" }});
            renderController.applyActiveFileTextState({{ kind, text: "body", editable: true, version: "v1", draft }});
            renderController.setFileEditMode(editMode);
            renderController.setFileViewMode(viewMode);
            if (videoPreviewAvailable) renderController.setActiveVideoFallback({{ token: "video-token", previewUrl: "/preview.mp4", rel: "clip.mkv", size: 12, preparing: videoPreviewPreparing }});
            else renderController.clearActiveVideoFallback();
            const value = renderController.currentFileModeControlState();
            return {{ value, frozen: Object.isFrozen(value) }};
          }}
          const modeControlDiffable = runModeControlState({{ viewMode: "diff", gitFresh: true, changed: true }});
          const modeControlNoPath = runModeControlState({{ path: "", viewMode: "file", kind: "text", gitFresh: true, changed: true }});
          const modeControlPreviewExitEdit = runModeControlState({{ viewMode: "preview", editMode: true, videoPreviewAvailable: true, videoPreviewPreparing: true }});
          const modeControlDraft = runModeControlState({{ path: "new.txt", viewMode: "file", kind: "text", draft: true, gitPath: false }});
          state.editorKind = "file";
          renderController.setActiveFileIdentity("fallback.txt", {{ gitPath: false, apiPath: "" }});
          renderController.applyActiveFileTextState({{ kind: "text", text: "fallback body", editable: true, version: "v1", draft: false }});
          renderController.setFileEditMode(true);
          renderController.setFileDirty(true);
          events.length = 0;
          state.editorKind = "plain-fallback";
          renderController.applyPlainTextFallbackState();
          const plainFallbackState = {{ editMode: renderController.currentFileEditMode(), dirty: renderController.currentFileDirty(), events: events.slice() }};
          state.editorKind = "file";
          events.length = 0;
          fileStatus.textContent = "";
          const videoPlan = renderController.prepareActiveVideoLoadResult("clip.mkv", {{ kind: "video", video_url: "/video.mov", video_preview_url: "/preview.mp4", content_type: "video/quicktime; charset=utf-8", size: 123 }}, {{ requestId: 9 }});
          const videoPlanState = {{ plan: videoPlan, frozen: Object.isFrozen(videoPlan), fallback: renderController.currentActiveVideoFallback(), events: events.slice(), status: fileStatus.textContent }};
          events.length = 0;
          const videoRetryResult = renderController.handleActiveVideoLoadError(videoPlan.token, {{ rel: videoPlan.rel, previewUrl: videoPlan.previewUrl, clearVideoHandlers: () => events.push(["clearVideoHandlers"]), loadPreview: (token, options) => events.push(["loadPreview", token, options]) }});
          const videoRetry = {{ result: videoRetryResult, status: fileStatus.textContent, events: events.slice(), fallback: renderController.currentActiveVideoFallback() }};
          renderController.completeCompatibleVideoPreview(videoPlan);
          events.length = 0;
          const videoMetadataResult = renderController.handleActiveVideoLoadedMetadata(videoPlan.token);
          const videoMetadata = {{ result: videoMetadataResult, status: fileStatus.textContent, events: events.slice() }};
          events.length = 0;
          const videoUsedErrorResult = renderController.handleActiveVideoLoadError(videoPlan.token, {{ rel: videoPlan.rel, previewUrl: videoPlan.previewUrl, clearVideoHandlers: () => events.push(["clearVideoHandlers"]), loadPreview: (token, options) => events.push(["loadPreview", token, options]) }});
          const videoUsedError = {{ result: videoUsedErrorResult, status: fileStatus.textContent, events: events.slice(), fallback: renderController.currentActiveVideoFallback() }};
          events.length = 0;
          const unsupportedPlan = renderController.prepareActiveVideoLoadResult("raw.avi", {{ kind: "video", video_url: "/raw.avi", content_type: "video/x-msvideo", size: 5 }}, {{ requestId: 10 }});
          const unsupportedErrorResult = renderController.handleActiveVideoLoadError(unsupportedPlan.token, {{ rel: unsupportedPlan.rel, previewUrl: unsupportedPlan.previewUrl, clearVideoHandlers: () => events.push(["clearVideoHandlers"]), loadPreview: (token, options) => events.push(["loadPreview", token, options]) }});
          const unsupportedVideo = {{ plan: unsupportedPlan, result: unsupportedErrorResult, status: fileStatus.textContent, events: events.slice(), fallback: renderController.currentActiveVideoFallback() }};
          let invalidVideoMessage = "";
          try {{ renderController.prepareActiveVideoLoadResult("bad.mkv", {{ kind: "video", video_preview_url: "/preview.mp4" }}, {{ requestId: 11 }}); }} catch (err) {{ invalidVideoMessage = err && err.message ? err.message : String(err); }}
          const videoPolicy = {{ videoPlanState, videoRetry, videoMetadata, videoUsedError, unsupportedVideo, invalidVideoMessage }};
          const pdfStateA = {{ id: "a" }};
          const pdfStateB = {{ id: "b" }};
          const pdfSetA = renderController.setActivePdfRenderState(pdfStateA) === pdfStateA;
          const pdfAActive = renderController.isActivePdfRenderState(pdfStateA);
          const pdfBActive = renderController.isActivePdfRenderState(pdfStateB);
          const pdfTaken = renderController.takeActivePdfRenderState() === pdfStateA;
          const pdfAAfterTake = renderController.isActivePdfRenderState(pdfStateA);
          renderController.setActivePdfRenderState(pdfStateB);
          const pdfClearResult = renderController.clearActivePdfRenderState();
          const pdfBAfterClear = renderController.isActivePdfRenderState(pdfStateB);
          const pdfRenderState = {{ pdfSetA, pdfAActive, pdfBActive, pdfTaken, pdfAAfterTake, pdfClearResult, pdfBAfterClear }};
          events.length = 0;
          const loadOpen = renderController.startFileOpenRequest("plan.md", {{ line: 4, gitPath: false, apiPath: "" }});
          const loadRequest = loadOpen.request;
          const loadPlanDiff = renderController.prepareFileLoadResult("plan.md", {{ kind: "diff", baseText: "old", currentText: "new", baseExists: true, currentExists: false }}, loadRequest, {{ viewMode: "diff" }});
          const loadPlanNoDiff = renderController.prepareFileLoadResult("plan.md", {{ kind: "diff", baseText: "", currentText: "", baseExists: false, currentExists: false }}, loadRequest, {{ viewMode: "diff" }});
          const loadPlanImage = renderController.prepareFileLoadResult("plan.md", {{ kind: "image", image_url: "/img.png", size: 4 }}, loadRequest, {{ viewMode: "file" }});
          const loadPlanPdf = renderController.prepareFileLoadResult("plan.md", {{ kind: "pdf", pdf_url: "/doc.pdf", size: 8 }}, loadRequest, {{ viewMode: "file" }});
          const loadPlanDownload = renderController.prepareFileLoadResult("plan.md", {{ kind: "download_only", reason: "too large", viewer_max_bytes: 1024, size: 2048 }}, loadRequest, {{ viewMode: "file" }});
          const loadPlanMarkdown = renderController.prepareFileLoadResult("plan.md", {{ kind: "markdown", text: "# plan", editable: false, version: "v9", size: 6 }}, loadRequest, {{ viewMode: "preview" }});
          const loadPlanUnknownText = renderController.prepareFileLoadResult("plan.md", {{ kind: "custom", text: "x", editable: true }}, loadRequest, {{ viewMode: "file" }});
          const loadPlanVideo = renderController.prepareFileLoadResult("plan.md", {{ kind: "video", video_url: "/plan.mov", video_preview_url: "/plan-preview.mp4", content_type: "video/quicktime", size: 77 }}, loadRequest, {{ viewMode: "file" }});
          const loadPlanStale = renderController.prepareFileLoadResult("plan.md", {{ kind: "image", image_url: "/img.png" }}, {{ ...loadRequest, path: "other.md" }}, {{ viewMode: "file" }});
          let invalidLoadMessage = "";
          try {{ renderController.prepareFileLoadResult("plan.md", {{ kind: "image" }}, loadRequest, {{ viewMode: "file" }}); }} catch (err) {{ invalidLoadMessage = err && err.message ? err.message : String(err); }}
          const loadPlans = {{
            diff: {{ plan: loadPlanDiff, frozen: Object.isFrozen(loadPlanDiff) }},
            noDiff: {{ plan: loadPlanNoDiff, frozen: Object.isFrozen(loadPlanNoDiff) }},
            image: {{ plan: loadPlanImage, frozen: Object.isFrozen(loadPlanImage) }},
            pdf: {{ plan: loadPlanPdf, frozen: Object.isFrozen(loadPlanPdf) }},
            download: {{ plan: loadPlanDownload, frozen: Object.isFrozen(loadPlanDownload) }},
            markdown: {{ plan: loadPlanMarkdown, frozen: Object.isFrozen(loadPlanMarkdown) }},
            unknownText: {{ plan: loadPlanUnknownText, frozen: Object.isFrozen(loadPlanUnknownText) }},
            video: {{ plan: loadPlanVideo, frozen: Object.isFrozen(loadPlanVideo) }},
            stale: loadPlanStale,
            invalidLoadMessage,
            events: events.slice(),
          }};
          loadOpen.done();
          const programmaticBefore = renderController.isFileEditorProgrammaticChange();
          let programmaticInside = false;
          const programmaticReturn = renderController.runFileEditorProgrammaticChange(() => {{
            programmaticInside = renderController.isFileEditorProgrammaticChange();
            return "ok";
          }});
          const programmaticAfter = renderController.isFileEditorProgrammaticChange();
          let programmaticErrorMessage = "";
          let programmaticErrorInside = false;
          try {{
            renderController.runFileEditorProgrammaticChange(() => {{
              programmaticErrorInside = renderController.isFileEditorProgrammaticChange();
              throw new Error("boom");
            }});
          }} catch (err) {{
            programmaticErrorMessage = err && err.message ? err.message : String(err);
          }}
          const programmaticAfterError = renderController.isFileEditorProgrammaticChange();
          const programmaticChange = {{
            before: programmaticBefore,
            inside: programmaticInside,
            returnValue: programmaticReturn,
            after: programmaticAfter,
            errorInside: programmaticErrorInside,
            errorMessage: programmaticErrorMessage,
            afterError: programmaticAfterError,
          }};
          renderController.setFileDirty(false);
          const unsavedCleanPlan = renderController.fileUnsavedPromptPlan();
          const unsavedCleanChoice = await renderController.beginFileUnsavedPrompt();
          renderController.setFileDirty(true);
          const unsavedPromptPlan = renderController.fileUnsavedPromptPlan();
          const unsavedPendingBefore = renderController.isFileUnsavedPromptPending();
          const unsavedPendingPromise = renderController.beginFileUnsavedPrompt();
          const unsavedPendingAfterBegin = renderController.isFileUnsavedPromptPending();
          const unsavedDuplicatePlan = renderController.fileUnsavedPromptPlan();
          const unsavedResolved = renderController.resolveFileUnsavedPrompt("save");
          const unsavedChoice = await unsavedPendingPromise;
          const unsavedPendingAfterResolve = renderController.isFileUnsavedPromptPending();
          const unsavedResolveMissing = renderController.resolveFileUnsavedPrompt("discard");
          const unsavedPromptState = {{
            cleanPlan: unsavedCleanPlan,
            cleanChoice: unsavedCleanChoice,
            promptPlan: unsavedPromptPlan,
            pendingBefore: unsavedPendingBefore,
            pendingAfterBegin: unsavedPendingAfterBegin,
            duplicatePlan: unsavedDuplicatePlan,
            resolved: unsavedResolved,
            choice: unsavedChoice,
            pendingAfterResolve: unsavedPendingAfterResolve,
            resolveMissing: unsavedResolveMissing,
          }};
          const render = {{
            exportFrozen: Object.isFrozen(fileViewer),
            exports: Object.keys(fileViewer).sort(),
            conflict,
            currentConflict: renderController.currentSaveConflict(),
            labelText: fileStatus.children[0].text,
            actionTexts: fileStatus.children[1].children.map((node) => node.text),
            modeResolution: {{ diffFallback, diffAllowed, previewFallback, explicitDiff, invalidModeMessage }},
            fetchResults: {{ diffFetch, diffFetchEvents, readFetch, readFetchEvents }},
            downloadPaths,
            editorKindTransitions,
            programmaticChange,
            unsavedPromptState,
            restorePlanning,
            selectionMemory,
            candidateRefreshSeq,
            sessionSyncSeq,
            openErrors: {{ currentError, abortErrorResult, staleError, unknownError }},
            finalize,
            draft,
            draftErrors: {{ currentDraftError, abortDraftError, staleDraftError }},
            saveBodies,
            saveErrors: {{ saveConflict, genericSaveError, unknownSaveError }},
            unavailableAction: {{ availableBlocked, availableBlockStatus, unavailableBlocked, unavailableBlockStatus, unavailableTransitionEvents }},
            unsavedChoices,
            copySelection: {{ copyNoSelection, copySelectionSuccess, copySelectionError }},
            editorState: {{ editorState, editorStateFrozen }},
            editabilityUi: {{ editButtonEditMode, readOnlyWritable, editButtonViewMode, editButtonUnavailable, editButtonSavePending }},
            unsavedDecision: {{ cleanUnsaved, discardUnsaved, cancelUnsaved }},
            viewModeGuard: {{ viewModeSame, viewModeDraftBlocked, viewModeDiscardOpen, viewModeCancel, viewModeUnavailable }},
            hideRequest: {{ hideClean, hideCancel, hideDiscard }},
            unavailableHandler: {{ unavailableHandleClean, unavailableHandleDirty, unavailableHandleMismatch, unavailableHandleClosed }},
            draftGuard: {{ draftInvalidPath, draftDirectory, draftExisting, draftInspectError, draftNew }},
            capabilities: {{ derivedCapabilities, editableCapabilities, pendingCapabilities, binaryCapabilities, missingPathCapabilities, editableFrozen: Object.isFrozen(editableCapabilities) }},
            modeControl: {{ modeControlDiffable, modeControlNoPath, modeControlPreviewExitEdit, modeControlDraft }},
            plainFallbackState,
            videoPolicy,
            pdfRenderState,
            loadPlans,
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
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def run_file_viewer_touch_binding_probe() -> dict[str, object]:
    viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, setTimeout, clearTimeout, Date }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(viewer_source)}, ctx);
        const fileViewer = ctx.window.CodoxearFileViewer;
        const events = [];
        function button() {{
          return {{
            listeners: {{}},
            options: {{}},
            addEventListener(type, handler, options) {{
              this.listeners[type] = handler;
              this.options[type] = options || null;
            }},
          }};
        }}
        function event(type, pointerType = "") {{
          return {{
            type,
            pointerType,
            prevented: 0,
            stopped: 0,
            preventDefault() {{ this.prevented += 1; }},
            stopPropagation() {{ this.stopped += 1; }},
          }};
        }}
        let now = 1000;
        const pressButton = button();
        const clickButton = button();
        const pressBound = fileViewer.bindFileTouchPress(pressButton, () => events.push(["press"]), {{ nowMs: () => now }});
        const clickBound = fileViewer.bindFileTouchClick(clickButton, () => events.push(["click"]));
        const pointer = event("pointerdown", "touch");
        pressButton.listeners.pointerdown(pointer);
        now = 1100;
        const touch = event("touchstart");
        pressButton.listeners.touchstart(touch);
        now = 1200;
        const suppressedClick = event("click");
        pressButton.listeners.click(suppressedClick);
        now = 1800;
        const laterClick = event("click");
        pressButton.listeners.click(laterClick);
        const plainClick = event("click");
        clickButton.listeners.click(plainClick);
        process.stdout.write(JSON.stringify({{
          pressBound,
          clickBound,
          invalidPress: fileViewer.bindFileTouchPress(null, () => {{}}),
          invalidClick: fileViewer.bindFileTouchClick(clickButton, null),
          touchPassiveFalse: pressButton.options.touchstart && pressButton.options.touchstart.passive === false,
          events,
          pointerPrevented: pointer.prevented,
          touchSuppressed: [touch.prevented, touch.stopped],
          suppressedClick: [suppressedClick.prevented, suppressedClick.stopped],
          laterClick: [laterClick.prevented, laterClick.stopped],
          plainClick: [plainClick.prevented, plainClick.stopped],
        }}));
        """
    )
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def run_file_viewer_pdf_loader_probe() -> dict[str, object]:
    viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, setTimeout, clearTimeout }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(viewer_source)}, ctx);
        const fileViewer = ctx.window.CodoxearFileViewer;
        const events = [];
        const importedPdf = {{ GlobalWorkerOptions: {{}} }};
        const loader = fileViewer.createPdfLoader({{
          resolveAppUrl: (path) => `app:${{path}}`,
          importModule: async (url) => {{ events.push(["import", url]); return importedPdf; }},
          timeoutMs: 50,
        }});
        (async () => {{
          const first = await loader.ensure();
          const second = await loader.ensure();
          const globalPdf = {{ getDocument() {{}}, GlobalWorkerOptions: {{}} }};
          const globalLoader = fileViewer.createPdfLoader({{
            resolveAppUrl: (path) => `app:${{path}}`,
            globalObject: {{ pdfjsLib: globalPdf }},
            importModule: async (url) => {{ events.push(["unexpected-import", url]); return {{}}; }},
          }});
          const globalResult = await globalLoader.ensure();
          let missingResolveError = "";
          try {{ fileViewer.createPdfLoader({{}}); }} catch (err) {{ missingResolveError = err && err.message ? err.message : String(err); }}
          process.stdout.write(JSON.stringify({{
            sameImported: first === importedPdf && second === importedPdf,
            importedWorker: importedPdf.GlobalWorkerOptions.workerSrc,
            globalResult: globalResult === globalPdf,
            globalWorker: globalPdf.GlobalWorkerOptions.workerSrc,
            events,
            missingResolveError,
          }}));
        }})().catch((err) => {{ console.error(err && err.stack ? err.stack : err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendFileViewerModuleSource(unittest.TestCase):
    def test_file_viewer_controller_owns_save_conflict_behavior(self) -> None:
        result = run_file_viewer_controller_probe()
        self.assertTrue(result["render"]["exportFrozen"])
        self.assertEqual(result["render"]["exports"], ["bindFileTouchClick", "bindFileTouchPress", "createFileViewerController", "createPdfLoader"])
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
        self.assertEqual(result["render"]["downloadPaths"], {
            "downloadGitToken": "/api/sessions/sid-1/file/download?path=src%2Fapp.py&path_token=tok%20space&git_path=1",
            "downloadPlain": "/api/sessions/sid-1/file/download?path=plain.txt",
            "downloadMissingSession": "",
            "downloadUnavailablePath": "",
            "downloadUnavailableStatus": "Session is no longer available; copy unsaved edits before closing.",
        })
        self.assertEqual(result["render"]["editorKindTransitions"], {"initial": "", "file": "file", "currentFile": "file", "diff": "diff", "plain": "plain-fallback", "clear": "", "invalidMessage": "invalid file editor kind"})
        self.assertEqual(result["render"]["programmaticChange"], {"before": False, "inside": True, "returnValue": "ok", "after": False, "errorInside": True, "errorMessage": "boom", "afterError": False})
        self.assertEqual(result["render"]["unsavedPromptState"], {"cleanPlan": {"kind": "choice", "choice": "discard"}, "cleanChoice": "discard", "promptPlan": {"kind": "prompt"}, "pendingBefore": False, "pendingAfterBegin": True, "duplicatePlan": {"kind": "choice", "choice": "cancel"}, "resolved": True, "choice": "save", "pendingAfterResolve": False, "resolveMissing": False})
        self.assertEqual(result["render"]["restorePlanning"], {"skip": {"kind": "skip"}, "skipFrozen": True, "skipDirty": False, "restore": {"kind": "restore", "text": "body text"}, "restoreFrozen": True, "dirtyBeforeFinish": True, "dirtyAfterFinish": False})
        self.assertEqual(result["render"]["selectionMemory"], {"noSession": {"path": "", "line": None, "gitPath": False}, "history": {"path": "history.txt", "line": 3, "gitPath": False, "apiPath": ""}, "remembered": {"path": "src/app.py", "apiPath": "tok-1", "line": 12, "gitPath": True}})
        self.assertEqual(result["render"]["candidateRefreshSeq"], {"first": 1, "second": 2, "firstCurrent": True, "firstAfterSecond": False, "secondCurrent": True})
        self.assertEqual(result["render"]["sessionSyncSeq"], {"first": 1, "invalidated": 2, "firstCurrent": True, "firstAfterInvalidate": False, "invalidatedCurrent": True})
        reset_events = [
            ["editorOptions", {"readOnly": True}],
            ["touchToolbar"],
            ["buttonClass", "active", False],
            ["buttonClass", "primary", False],
            ["buttonClass", "dirty", False],
            ["buttonAttr", "aria-label", "Edit file"],
            ["touchToolbar"],
            ["touchToolbar"],
        ]
        self.assertEqual(result["render"]["openErrors"], {
            "currentError": {"result": False, "status": "error: boom", "resetCount": 0, "touchCount": 3, "events": reset_events},
            "abortErrorResult": {"result": False, "status": "error: boom", "resetCount": 0, "touchCount": 3, "events": [], "abortCheck": True},
            "staleError": {"result": False, "status": "error: boom", "resetCount": 0, "touchCount": 3, "events": []},
            "unknownError": {"result": False, "status": "error: unknown error", "resetCount": 0, "touchCount": 6, "events": reset_events},
        })
        self.assertEqual(result["render"]["finalize"], {
            "result": True,
            "events": [
                ["applyFileMode"],
                ["rememberOpenedFile", "src/app.py", "/abs/src/app.py"],
                ["buttonClass", "active", False],
                ["buttonClass", "primary", False],
                ["buttonClass", "dirty", False],
                ["buttonAttr", "aria-label", "Edit file"],
                ["touchToolbar"],
                ["renderFilePickerMenu"],
            ],
        })
        self.assertEqual(result["render"]["draft"], {
            "result": True,
            "status": "draft/new.txt - new file",
            "viewMode": "file",
            "events": [
                ["persistFileViewMode", "file"],
                ["persistFileNonDiffMode", "file"],
                ["applyFileMode"],
                ["applyFileMode"],
                ["renderMonacoFile", "draft/new.txt", "", 5, ""],
                ["editorOptions", {"readOnly": False}],
                ["buttonClass", "active", True],
                ["buttonClass", "primary", True],
                ["buttonClass", "dirty", False],
                ["buttonAttr", "aria-label", "Save file"],
                ["touchToolbar"],
                ["renderFilePickerMenu"],
            ],
        })
        draft_reset_events = [
            ["editorOptions", {"readOnly": True}],
            ["touchToolbar"],
            ["buttonClass", "active", False],
            ["buttonClass", "primary", False],
            ["buttonClass", "dirty", False],
            ["buttonAttr", "aria-label", "Edit file"],
            ["touchToolbar"],
        ]
        self.assertEqual(result["render"]["draftErrors"], {
            "currentDraftError": {"result": False, "status": "error: draft boom", "resetCount": 0, "touchCount": 2, "events": draft_reset_events},
            "abortDraftError": {"result": False, "status": "error: draft boom", "resetCount": 0, "touchCount": 2, "events": []},
            "staleDraftError": {"result": False, "status": "error: draft boom", "resetCount": 0, "touchCount": 2, "events": []},
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
            "unavailableTransitionEvents": [
                ["editorOptions", {"readOnly": True}],
                ["buttonClass", "active", False],
                ["buttonClass", "primary", False],
                ["buttonClass", "dirty", False],
                ["buttonAttr", "aria-label", "Session unavailable; copy edits before closing"],
                ["touchToolbar"],
                ["hideFileUnsavedDialog", "cancel"],
                ["disposeOpenRender"],
                ["resetFileSearchState"],
                ["closeFilePickerMenu", {"restoreInput": True}],
                ["editorOptions", {"readOnly": True}],
                ["buttonClass", "active", False],
                ["buttonClass", "primary", False],
                ["buttonClass", "dirty", False],
                ["buttonAttr", "aria-label", "Session unavailable; copy edits before closing"],
                ["touchToolbar"],
                ["touchToolbar"],
            ],
        })
        self.assertEqual(result["render"]["unsavedChoices"], {
            "unsavedSaveAvailable": {"result": True, "status": "old unsaved status", "events": [["hideFileUnsavedDialog", "save"]]},
            "unsavedSaveUnavailable": {"result": False, "status": "Session is no longer available; copy unsaved edits before closing.", "events": []},
            "unsavedDiscard": {"result": True, "events": [["hideFileUnsavedDialog", "discard"]]},
            "unsavedCancel": {"result": True, "events": [["hideFileUnsavedDialog", "cancel"]]},
        })
        self.assertEqual(result["render"]["copySelection"], {
            "copyNoSelection": {"result": False, "events": [["toast", "nothing selected"]]},
            "copySelectionSuccess": {"result": True, "events": [["copyToClipboard", "selected text"], ["editorOptions", {"readOnly": False}], ["touchToolbar"], ["toast", "selection copied"], ["focusActiveFileCodeEditor"]]},
            "copySelectionError": {"result": False, "events": [["copyToClipboard", "selected text"], ["toast", "copy error: denied"], ["focusActiveFileCodeEditor"]]},
        })
        self.assertEqual(result["render"]["editorState"], {
            "editorState": {
                "path": "state.md",
                "apiPath": "state-token",
                "gitPath": True,
                "kind": "markdown",
                "editable": True,
                "version": "v7",
                "draft": False,
                "viewMode": "file",
                "editorKind": "file",
                "editMode": True,
                "dirty": True,
                "savePending": False,
                "sessionId": "sid-1",
                "unavailable": False,
            },
            "editorStateFrozen": True,
        })
        self.assertEqual(result["render"]["editabilityUi"], {
            "editButtonEditMode": {
                "button": {"disabled": False, "innerHTML": "icon:save", "title": "Save file", "ariaLabel": "Save file", "classes": ["active", "dirty", "primary"]},
                "events": [["buttonClass", "active", True], ["buttonClass", "primary", True], ["buttonClass", "dirty", True], ["buttonAttr", "aria-label", "Save file"], ["touchToolbar"]],
            },
            "readOnlyWritable": [["editorOptions", {"readOnly": False}]],
            "editButtonViewMode": {
                "button": {"disabled": False, "innerHTML": "icon:edit", "title": "Edit file", "ariaLabel": "Edit file", "classes": []},
                "events": [["buttonClass", "active", False], ["buttonClass", "primary", False], ["buttonClass", "dirty", False], ["buttonAttr", "aria-label", "Edit file"], ["touchToolbar"], ["editorOptions", {"readOnly": True}]],
            },
            "editButtonUnavailable": {
                "button": {"disabled": True, "innerHTML": "icon:edit", "title": "Session unavailable; copy edits before closing", "ariaLabel": "Session unavailable; copy edits before closing", "classes": ["dirty"]},
                "events": [["buttonClass", "active", False], ["buttonClass", "primary", False], ["buttonClass", "dirty", True], ["buttonAttr", "aria-label", "Session unavailable; copy edits before closing"], ["touchToolbar"]],
            },
            "editButtonSavePending": {
                "button": {"disabled": True, "innerHTML": "icon:save", "title": "Saving file", "ariaLabel": "Saving file", "classes": ["active", "dirty", "primary"]},
                "events": [["buttonClass", "active", True], ["buttonClass", "primary", True], ["buttonClass", "dirty", True], ["buttonAttr", "aria-label", "Saving file"], ["touchToolbar"], ["editorOptions", {"readOnly": False}]],
                "status": "Saving state.md...",
            },
        })
        self.assertEqual(result["render"]["unsavedDecision"], {
            "cleanUnsaved": {"result": True, "events": []},
            "discardUnsaved": {"result": True, "events": [["promptUnsaved", "discard"], ["restoreFileEditorText", "body"], ["editorOptions", {"readOnly": True}], ["buttonClass", "active", False], ["buttonClass", "primary", False], ["buttonClass", "dirty", True], ["buttonAttr", "aria-label", "Edit file"], ["touchToolbar"]]},
            "cancelUnsaved": {"result": False, "events": [["promptUnsaved", "cancel"]]},
        })
        self.assertEqual(result["render"]["viewModeGuard"], {
            "viewModeSame": {"result": True, "viewMode": "file", "status": "", "events": []},
            "viewModeDraftBlocked": {"result": False, "viewMode": "file", "status": "", "events": []},
            "viewModeDiscardOpen": {
                "result": True,
                "viewMode": "file",
                "status": "Loading...",
                "events": [["promptUnsaved", "discard"], ["restoreFileEditorText", "body"], ["editorOptions", {"readOnly": True}], ["buttonClass", "active", False], ["buttonClass", "primary", False], ["buttonClass", "dirty", True], ["buttonAttr", "aria-label", "Edit file"], ["touchToolbar"], ["persistFileViewMode", "preview"], ["persistFileNonDiffMode", "preview"], ["applyFileMode"], ["renderFilePickerMenu"], ["disposeOpenRender"], ["resetFileViewerPanel"], ["persistFileViewMode", "file"], ["persistFileNonDiffMode", "file"], ["applyFileMode"], ["api", "/api/sessions/sid-1/file/read?path=state.md&path_token=state-token&git_path=1", False], ["applyFileLoadResult", "state.md", "text", "file"], ["applyFileMode"], ["rememberOpenedFile", "state.md", "/abs/read"], ["buttonClass", "active", False], ["buttonClass", "primary", False], ["buttonClass", "dirty", True], ["buttonAttr", "aria-label", "Edit file"], ["touchToolbar"], ["renderFilePickerMenu"]],
            },
            "viewModeCancel": {"result": False, "viewMode": "file", "status": "", "events": [["promptUnsaved", "cancel"]]},
            "viewModeUnavailable": {"result": False, "viewMode": "file", "status": "Session is no longer available; copy unsaved edits before closing.", "events": []},
        })
        self.assertEqual(result["render"]["hideRequest"], {
            "hideClean": {"result": True, "events": [["hideFileViewer"]]},
            "hideCancel": {"result": False, "events": [["promptUnsaved", "cancel"]]},
            "hideDiscard": {"result": True, "events": [["promptUnsaved", "discard"], ["restoreFileEditorText", "body"], ["editorOptions", {"readOnly": True}], ["buttonClass", "active", False], ["buttonClass", "primary", False], ["buttonClass", "dirty", True], ["buttonAttr", "aria-label", "Edit file"], ["touchToolbar"], ["hideFileViewer"]]},
        })
        self.assertEqual(result["render"]["unavailableHandler"], {
            "unavailableHandleClean": {"result": True, "unavailable": False, "status": "", "events": [["hideFileViewer"]]},
            "unavailableHandleDirty": {
                "result": True,
                "unavailable": True,
                "status": "Session is no longer available; copy unsaved edits before closing.",
                "events": [["editorOptions", {"readOnly": True}], ["buttonClass", "active", False], ["buttonClass", "primary", False], ["buttonClass", "dirty", True], ["buttonAttr", "aria-label", "Session unavailable; copy edits before closing"], ["touchToolbar"], ["hideFileUnsavedDialog", "cancel"], ["disposeOpenRender"], ["resetFileSearchState"], ["closeFilePickerMenu", {"restoreInput": True}], ["editorOptions", {"readOnly": True}], ["buttonClass", "active", False], ["buttonClass", "primary", False], ["buttonClass", "dirty", True], ["buttonAttr", "aria-label", "Session unavailable; copy edits before closing"], ["touchToolbar"], ["touchToolbar"]],
            },
            "unavailableHandleMismatch": {"result": False, "unavailable": False, "status": "", "events": []},
            "unavailableHandleClosed": {"result": False, "unavailable": False, "status": "", "events": []},
        })
        self.assertEqual(result["render"]["draftGuard"], {
            "draftInvalidPath": {"result": False, "status": "Choose a valid relative file path.", "viewMode": "diff", "events": []},
            "draftDirectory": {"result": False, "status": "draft/new.txt - path is a directory", "viewMode": "diff", "events": [["inspect", "draft/new.txt"]]},
            "draftExisting": {"result": True, "status": "Loading...", "viewMode": "file", "events": [["inspect", "draft/new.txt"], ["setFilePath", "draft/new.txt", {"line": None, "gitPath": False, "apiPath": ""}], ["persistFileViewMode", "file"], ["persistFileNonDiffMode", "file"], ["applyFileMode"], ["renderFilePickerMenu"], ["disposeOpenRender"], ["resetFileViewerPanel"], ["api", "/api/sessions/sid-1/file/read?path=draft%2Fnew.txt", False], ["applyFileLoadResult", "draft/new.txt", "text", "file"], ["applyFileMode"], ["rememberOpenedFile", "draft/new.txt", "/abs/read"], ["buttonClass", "active", False], ["buttonClass", "primary", False], ["buttonClass", "dirty", False], ["buttonAttr", "aria-label", "Edit file"], ["touchToolbar"], ["renderFilePickerMenu"]]},
            "draftInspectError": {"result": False, "status": "error: inspect boom", "viewMode": "diff", "events": [["inspect", "draft/new.txt"]]},
            "draftNew": {"result": True, "status": "draft/new.txt - new file", "viewMode": "file", "events": [["inspect", "draft/new.txt"], ["persistFileViewMode", "file"], ["persistFileNonDiffMode", "file"], ["applyFileMode"], ["setFilePath", "draft/new.txt", {"line": None, "gitPath": False}], ["renderFilePickerMenu"], ["disposeOpenRender"], ["resetFileViewerPanel"], ["applyFileMode"], ["renderMonacoFile", "draft/new.txt", "", None, ""], ["editorOptions", {"readOnly": False}], ["buttonClass", "active", True], ["buttonClass", "primary", True], ["buttonClass", "dirty", False], ["buttonAttr", "aria-label", "Save file"], ["touchToolbar"], ["renderFilePickerMenu"]]},
        })
        self.assertEqual(result["render"]["capabilities"], {
            "derivedCapabilities": {
                "capabilities": {"canEnterEditMode": True, "writable": True, "idleWritable": True, "idleTextWritable": True, "editModeAllowedInCurrentView": True},
                "canEnter": True,
                "writable": True,
                "idleWritable": True,
                "idleTextWritable": True,
                "editModeAllowed": True,
            },
            "editableCapabilities": {"canEnterEditMode": True, "writable": True, "idleWritable": True, "idleTextWritable": True, "editModeAllowedInCurrentView": True},
            "pendingCapabilities": {"canEnterEditMode": False, "writable": True, "idleWritable": False, "idleTextWritable": False, "editModeAllowedInCurrentView": True},
            "binaryCapabilities": {"canEnterEditMode": False, "writable": True, "idleWritable": True, "idleTextWritable": False, "editModeAllowedInCurrentView": False},
            "missingPathCapabilities": {"canEnterEditMode": False, "writable": True, "idleWritable": True, "idleTextWritable": True, "editModeAllowedInCurrentView": True},
            "editableFrozen": True,
        })
        self.assertEqual(result["render"]["modeControl"], {
            "modeControlDiffable": {"value": {"diffActive": True, "previewActive": False, "diffDisabled": False, "previewDisabled": False, "downloadDisabled": False, "videoPreviewVisible": False, "videoPreviewDisabled": True, "videoPreviewTitle": "Use compatible MP4 preview", "markdownPreviewVisible": True, "shouldHidePasteDialog": True, "shouldExitEditMode": False}, "frozen": True},
            "modeControlNoPath": {"value": {"diffActive": False, "previewActive": False, "diffDisabled": True, "previewDisabled": True, "downloadDisabled": True, "videoPreviewVisible": False, "videoPreviewDisabled": True, "videoPreviewTitle": "Use compatible MP4 preview", "markdownPreviewVisible": False, "shouldHidePasteDialog": False, "shouldExitEditMode": False}, "frozen": True},
            "modeControlPreviewExitEdit": {"value": {"diffActive": False, "previewActive": True, "diffDisabled": True, "previewDisabled": False, "downloadDisabled": False, "videoPreviewVisible": True, "videoPreviewDisabled": True, "videoPreviewTitle": "Building compatible MP4 preview", "markdownPreviewVisible": True, "shouldHidePasteDialog": True, "shouldExitEditMode": True}, "frozen": True},
            "modeControlDraft": {"value": {"diffActive": False, "previewActive": False, "diffDisabled": True, "previewDisabled": True, "downloadDisabled": True, "videoPreviewVisible": False, "videoPreviewDisabled": True, "videoPreviewTitle": "Use compatible MP4 preview", "markdownPreviewVisible": False, "shouldHidePasteDialog": False, "shouldExitEditMode": False}, "frozen": True},
        })
        self.assertFalse(result["render"]["plainFallbackState"]["editMode"])
        self.assertFalse(result["render"]["plainFallbackState"]["dirty"])
        self.assertIn(["touchToolbar"], result["render"]["plainFallbackState"]["events"])
        self.assertIn(["buttonClass", "dirty", False], result["render"]["plainFallbackState"]["events"])
        self.assertIn(["buttonAttr", "aria-label", "Edit file"], result["render"]["plainFallbackState"]["events"])
        self.assertEqual(result["render"]["videoPolicy"], {
            "videoPlanState": {
                "plan": {"token": "9:clip.mkv:0", "rel": "clip.mkv", "videoUrl": "/video.mov", "previewUrl": "/preview.mp4", "size": 123, "contentType": "video/quicktime", "shouldPreviewFirst": True, "initialStatus": "clip.mkv - video - 123B"},
                "frozen": True,
                "fallback": {"token": "9:clip.mkv:0", "previewUrl": "/preview.mp4", "used": False, "preparing": False, "rel": "clip.mkv", "size": 123},
                "events": [["applyFileMode"]],
                "status": "",
            },
            "videoRetry": {
                "result": True,
                "status": "",
                "events": [["loadPreview", "9:clip.mkv:0", {"explicit": False}]],
                "fallback": {"token": "9:clip.mkv:0", "previewUrl": "/preview.mp4", "used": False, "preparing": False, "rel": "clip.mkv", "size": 123},
            },
            "videoMetadata": {"result": True, "status": "clip.mkv - compatible video preview - 123B", "events": []},
            "videoUsedError": {"result": True, "status": "clip.mkv - video preview unavailable after conversion", "events": [["applyFileMode"], ["clearVideoHandlers"]], "fallback": None},
            "unsupportedVideo": {
                "plan": {"token": "10:raw.avi:0", "rel": "raw.avi", "videoUrl": "/raw.avi", "previewUrl": "", "size": 5, "contentType": "video/x-msvideo", "shouldPreviewFirst": False, "initialStatus": "raw.avi - video - 5B"},
                "result": False,
                "status": "raw.avi - video unsupported",
                "events": [["applyFileMode"]],
                "fallback": None,
            },
            "invalidVideoMessage": "invalid video response",
        })
        self.assertEqual(result["render"]["pdfRenderState"], {"pdfSetA": True, "pdfAActive": True, "pdfBActive": False, "pdfTaken": True, "pdfAAfterTake": False, "pdfClearResult": True, "pdfBAfterClear": False})
        load_plans = result["render"]["loadPlans"]
        self.assertEqual(load_plans["diff"], {"plan": {"kind": "diff", "noDiff": False, "baseText": "old", "currentText": "new", "status": "plan.md - diff"}, "frozen": True})
        self.assertEqual(load_plans["noDiff"], {"plan": {"kind": "diff", "noDiff": True, "status": "plan.md - no diff"}, "frozen": True})
        self.assertEqual(load_plans["image"], {"plan": {"kind": "image", "imageUrl": "/img.png", "alt": "plan.md", "status": "plan.md - 4B"}, "frozen": True})
        self.assertEqual(load_plans["pdf"], {"plan": {"kind": "pdf", "pdfUrl": "/doc.pdf", "status": "plan.md - PDF - 8B"}, "frozen": True})
        self.assertEqual(load_plans["download"], {"plan": {"kind": "download_only", "reason": "too large", "viewerMaxBytes": 1024, "size": 2048, "status": "plan.md - download only - 2048B"}, "frozen": True})
        self.assertEqual(load_plans["markdown"], {"plan": {"kind": "text", "text": "# plan", "renderPreview": True, "status": "plan.md - preview - read-only - 6B"}, "frozen": True})
        self.assertEqual(load_plans["unknownText"], {"plan": {"kind": "text", "text": "x", "renderPreview": False, "status": "plan.md - 1B"}, "frozen": True})
        video_load_plan = load_plans["video"]["plan"]
        self.assertRegex(video_load_plan["token"], r"^\d+:plan\.md:0$")
        self.assertEqual({key: value for key, value in video_load_plan.items() if key != "token"}, {"kind": "video", "rel": "plan.md", "videoUrl": "/plan.mov", "previewUrl": "/plan-preview.mp4", "size": 77, "contentType": "video/quicktime", "shouldPreviewFirst": True, "initialStatus": "plan.md - video - 77B"})
        self.assertTrue(load_plans["video"]["frozen"])
        self.assertIsNone(load_plans["stale"])
        self.assertEqual(load_plans["invalidLoadMessage"], "invalid image response")
        self.assertEqual(load_plans["events"][0], ["disposeOpenRender"])
        self.assertIn(["applyFileMode"], load_plans["events"])
        self.assertIn("file viewer dependency missing: el", result["missingDependencyError"])

        available = result["availableReloadFailure"]
        self.assertEqual(available["status"], "src/app.py - reload failed")
        self.assertIn(["confirm", "Reload src/app.py from disk and discard your unsaved editor draft?"], available["events"])
        self.assertIn(["api", "/api/sessions/sid-1/file/read?path=src%2Fapp.py&path_token=api-token&git_path=1", False], available["events"])
        self.assertIn(["applyFileLoadResult", "src/app.py", "text", "file"], available["events"])
        self.assertIn(["preventDefault"], available["events"])
        self.assertIn(["stopPropagation"], available["events"])
        self.assertTrue(available["current"])

        canceled = result["canceledReload"]
        self.assertEqual(canceled["status"], "")
        self.assertFalse(any(event[0] == "api" for event in canceled["events"]))

        unavailable = result["unavailableReloadFailure"]
        self.assertEqual(unavailable["status"], "Session is no longer available; copy unsaved edits before closing.")
        self.assertFalse(unavailable["current"])

        switched = result["switchedSessionReloadFailure"]
        self.assertEqual(switched["status"], "Loading...")
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

    def test_file_viewer_touch_binding_behavior(self) -> None:
        result = run_file_viewer_touch_binding_probe()
        self.assertTrue(result["pressBound"])
        self.assertTrue(result["clickBound"])
        self.assertFalse(result["invalidPress"])
        self.assertFalse(result["invalidClick"])
        self.assertTrue(result["touchPassiveFalse"])
        self.assertEqual(result["events"], [["press"], ["press"], ["click"]])
        self.assertEqual(result["pointerPrevented"], 1)
        self.assertEqual(result["touchSuppressed"], [1, 1])
        self.assertEqual(result["suppressedClick"], [1, 1])
        self.assertEqual(result["laterClick"], [1, 1])
        self.assertEqual(result["plainClick"], [1, 1])

    def test_file_viewer_pdf_loader_behavior(self) -> None:
        result = run_file_viewer_pdf_loader_probe()
        self.assertTrue(result["sameImported"])
        self.assertEqual(result["importedWorker"], "app:pdf.worker.mjs")
        self.assertTrue(result["globalResult"])
        self.assertEqual(result["globalWorker"], "app:pdf.worker.mjs")
        self.assertEqual(result["events"], [["import", "app:pdf.mjs"]])
        self.assertIn("file viewer dependency missing: resolveAppUrl", result["missingResolveError"])

    def test_file_viewer_module_registered_before_app_js(self) -> None:
        index_source = INDEX_HTML.read_text(encoding="utf-8")
        routes_source = STATIC_ROUTES.read_text(encoding="utf-8")
        app_source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        editor_source = APP_FILE_EDITOR_JS.read_text(encoding="utf-8")
        self.assertLess(index_source.index("app_file_picker.js"), index_source.index("app_file_viewer.js"))
        self.assertLess(index_source.index("app_file_viewer.js"), index_source.index("app_file_editor.js"))
        self.assertLess(index_source.index("app_file_editor.js"), index_source.index("app_session_helpers.js"))
        self.assertLess(index_source.index("app_file_editor.js"), index_source.index("app.js"))
        self.assertIn('"app_file_viewer.js"', routes_source)
        self.assertIn('"app_file_editor.js"', routes_source)
        self.assertIn("const codoxearFileViewer = window.CodoxearFileViewer;", app_source)
        self.assertIn('throw new Error("Codoxear file viewer controller failed to load")', app_source)
        self.assertIn("const codoxearFileEditor = window.CodoxearFileEditor;", app_source)
        self.assertIn('throw new Error("Codoxear file editor runtime failed to load")', app_source)
        self.assertIn("createFileEditorRuntime", editor_source)
        self.assertIn("createMonacoLoader", editor_source)
        self.assertIn("renderSaveConflict", viewer_source)
        self.assertIn("bindFileTouchPress", viewer_source)
        self.assertIn("bindFileTouchClick", viewer_source)
        self.assertIn("createPdfLoader", viewer_source)
        self.assertNotIn("function renderFileSaveConflict", app_source)


if __name__ == "__main__":
    unittest.main()
