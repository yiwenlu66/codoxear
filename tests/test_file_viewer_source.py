import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_MARKDOWN_JS = ROOT / "codoxear" / "static" / "app_markdown.js"
APP_FILE_HELPERS_JS = ROOT / "codoxear" / "static" / "app_file_helpers.js"
APP_FILE_PICKER_JS = ROOT / "codoxear" / "static" / "app_file_picker.js"
APP_FILE_EDITOR_JS = ROOT / "codoxear" / "static" / "app_file_editor.js"
APP_FILE_VIEWER_JS = ROOT / "codoxear" / "static" / "app_file_viewer.js"
APP_VIEWPORT_JS = ROOT / "codoxear" / "static" / "app_viewport.js"
APP_CSS = ROOT / "codoxear" / "static" / "app.css"
SERVER_PY = ROOT / "codoxear" / "server.py"


def controller_identity_ctx_js(
    path: str = "src/app.py",
    api_path: str = "token-1",
    git_path: bool = True,
    line: int | None = 42,
) -> str:
    identity = {
        "path": path,
        "apiPath": api_path,
        "gitPath": git_path,
        "line": line,
    }
    return f"""
          identity: {json.dumps(identity)},
          fileViewerController: {{
            nextActiveFileIdentity(current, nextPath, opts = {{}}) {{
              if (!current || typeof current !== "object") throw new Error("current file identity required");
              const rel = String(nextPath ?? "");
              const useGitPath = Object.prototype.hasOwnProperty.call(opts, "gitPath") && opts.gitPath !== undefined ? Boolean(opts.gitPath) : Boolean(current.gitPath);
              const previousPath = String(current.path ?? "");
              const previousApiPath = String(current.apiPath || "");
              const reusableApiPath = rel === previousPath ? previousApiPath : "";
              const apiPath = Object.prototype.hasOwnProperty.call(opts, "apiPath") && opts.apiPath !== undefined
                ? (typeof opts.apiPath === "string" && opts.apiPath !== "" ? opts.apiPath : "")
                : (useGitPath ? ctx.fileApiPathForPath(rel, reusableApiPath) : "");
              return Object.freeze({{ path: rel, gitPath: useGitPath, apiPath }});
            }},
            currentActiveFileIdentity() {{
              return Object.freeze({{ path: String(ctx.identity.path ?? ""), gitPath: Boolean(ctx.identity.gitPath), apiPath: String(ctx.identity.apiPath || "") }});
            }},
            currentActiveFileLine() {{ return ctx.identity.line; }},
            currentFileViewerSessionId() {{ return String(ctx.fileViewerSessionId || "").trim(); }},
            setFileViewerSessionId(sid) {{ ctx.fileViewerSessionId = String(sid || "").trim(); return ctx.fileViewerSessionId; }},
            clearFileViewerSessionId() {{ ctx.fileViewerSessionId = ""; }},
            isFileViewerSessionUnavailable() {{ return Boolean(ctx.fileViewerUnavailableSessionId && ctx.fileViewerSessionId && ctx.fileViewerUnavailableSessionId === ctx.fileViewerSessionId); }},
            clearFileViewerUnavailableSession() {{ ctx.fileViewerUnavailableSessionId = ""; }},
            beginFileViewerSessionSync() {{ ctx.fileViewerSessionSyncToken += 1; return ctx.fileViewerSessionSyncToken; }},
            invalidateFileViewerSessionSync() {{ ctx.fileViewerSessionSyncToken += 1; return ctx.fileViewerSessionSyncToken; }},
            isCurrentFileViewerSessionSync(token) {{ return token === ctx.fileViewerSessionSyncToken; }},
            rememberActiveFileSelection(sid = ctx.currentFileSessionId ? ctx.currentFileSessionId() : "") {{
              const saved = {{
                key: sid,
                path: ctx.activeFilePathValue(),
                apiPath: ctx.activeFileApiPathValue(),
                gitPath: ctx.activeFileGitPathValue(),
                line: ctx.activeFileLineValue(),
                syncToken: ctx.fileViewerSessionSyncToken,
                editMode: ctx.fileEditMode,
                savePending: ctx.fileSavePending,
                saveToken: ctx.activeFileSaveToken,
              }};
              if (Array.isArray(ctx.savedSelections)) ctx.savedSelections.push(saved);
              if (typeof calls !== "undefined" && Array.isArray(calls)) calls.push(["rememberActiveFileSelection", saved]);
              return saved;
            }},
            disableFileViewerForUnavailableSession(sid) {{
              ctx.rememberActiveFileSelection(sid);
              this.invalidateFileViewerSessionSync();
              ctx.fileViewerUnavailableSessionId = String(sid || "").trim();
              if (typeof ctx.clearActiveFileSaveState === "function") ctx.clearActiveFileSaveState();
              ctx.fileEditMode = false;
              ctx.hideFileUnsavedDialog("cancel");
              ctx.cancelPendingFileOpen();
              ctx.resetFileSearchState();
              ctx.closeFilePickerMenu({{ restoreInput: true }});
              ctx.syncFileEditorReadOnly();
              ctx.fileStatus.textContent = "Session is no longer available; copy unsaved edits before closing.";
              ctx.updateFileEditButton();
              ctx.updateFileTouchToolbar();
              return true;
            }},
            handleFileViewerSessionUnavailable(sessionId) {{
              const sid = String(sessionId || "").trim();
              if (!sid || !ctx.isFileViewerOpen()) return false;
              if (ctx.fileViewerSessionId && ctx.fileViewerSessionId !== sid) return false;
              if (!ctx.fileDirty) {{ ctx.hideFileViewer(); return true; }}
              return this.disableFileViewerForUnavailableSession(sid);
            }},
            clearActiveFileIdentity({{ line = null }} = {{}}) {{
              ctx.identity.path = "";
              ctx.identity.apiPath = "";
              ctx.identity.gitPath = false;
              ctx.identity.line = ctx.normalizeLineNumber(line);
            }},
            setActiveFileIdentity(nextPath, {{ line = null, gitPath = undefined, apiPath = undefined }} = {{}}) {{
              const identity = this.nextActiveFileIdentity(this.currentActiveFileIdentity(), nextPath, {{ gitPath, apiPath }});
              ctx.identity.path = identity.path;
              ctx.identity.gitPath = identity.gitPath;
              ctx.identity.apiPath = identity.apiPath;
              ctx.identity.line = ctx.normalizeLineNumber(line);
              return Object.freeze({{ ...identity, line: ctx.identity.line }});
            }},
            beginActiveFileIdentity(nextPath = null, {{ line = undefined, gitPath = undefined, apiPath = undefined }} = {{}}) {{
              const current = this.currentActiveFileIdentity();
              const identity = this.nextActiveFileIdentity(current, nextPath == null ? current.path : nextPath, {{ gitPath, apiPath }});
              ctx.identity.path = identity.path;
              ctx.identity.gitPath = identity.gitPath;
              ctx.identity.apiPath = identity.apiPath;
              ctx.identity.line = line === undefined ? ctx.identity.line : ctx.normalizeLineNumber(line);
              return Object.freeze({{ ...identity, line: ctx.identity.line }});
            }},
            abortPendingFileOpenTransport() {{
              if (!ctx.fileOpenAbortController) return;
              try {{ ctx.fileOpenAbortController.abort(); }} catch (_) {{}}
              ctx.fileOpenAbortController = null;
            }},
            cancelPendingFileOpen() {{
              ctx.fileOpenRequestId = (ctx.fileOpenRequestId || 0) + 1;
              if (typeof ctx.disposePdfRender === "function") ctx.disposePdfRender();
              this.abortPendingFileOpenTransport();
            }},
            beginFileOpenRequest(nextPath = null, {{ line = undefined, gitPath = undefined, apiPath = undefined }} = {{}}) {{
              this.cancelPendingFileOpen();
              const identity = this.beginActiveFileIdentity(nextPath, {{ line, gitPath, apiPath }});
              const controller = typeof AbortController === "function" ? new AbortController() : null;
              if (controller) ctx.fileOpenAbortController = controller;
              return Object.freeze({{ requestId: ctx.fileOpenRequestId || 0, sessionId: ctx.currentFileSessionId ? ctx.currentFileSessionId() : String(ctx.fileViewerSessionId || ctx.selected || "").trim(), path: identity.path, apiPath: identity.apiPath, gitPath: identity.gitPath, line: identity.line, signal: controller ? controller.signal : null }});
            }},
            isCurrentFileOpenRequest(request) {{
              if (!request) return false;
              const identity = this.currentActiveFileIdentity();
              return Boolean(request.requestId === (ctx.fileOpenRequestId || 0) && request.sessionId === (ctx.currentFileSessionId ? ctx.currentFileSessionId() : String(ctx.fileViewerSessionId || ctx.selected || "").trim()) && request.path === String(identity.path ?? "") && String(request.apiPath || "") === String(identity.apiPath || ""));
            }},
            finalizeFileOpenRequest(request) {{
              if (!request || !ctx.fileOpenAbortController) return;
              if (ctx.fileOpenAbortController.signal !== request.signal) return;
              if (!this.isCurrentFileOpenRequest(request)) return;
              ctx.fileOpenAbortController = null;
            }},
            startFileOpenRequest(nextPath = null, opts = {{}}) {{
              const request = this.beginFileOpenRequest(nextPath, opts);
              return Object.freeze({{ request, path: request.path, done: () => this.finalizeFileOpenRequest(request) }});
            }},
            normalizeExplicitFileOpenMode(requestedMode) {{
              if (requestedMode === null || requestedMode === undefined || requestedMode === "") return null;
              if (requestedMode === "preview" || requestedMode === "file" || requestedMode === "diff") return requestedMode;
              throw new Error("invalid file open mode");
            }},
            resolveFileOpenViewMode(request, rel, requestedMode = null) {{
              const openMode = this.normalizeExplicitFileOpenMode(requestedMode);
              if (openMode) return openMode;
              const entry = typeof ctx.activeFileEntry === "function" ? ctx.activeFileEntry() : null;
              const canUseDiffView = request && request.gitPath && Boolean(ctx.fileCandidateGitStateFresh) && Boolean(entry && entry.changed);
              const viewMode = String(ctx.fileViewMode || "");
              return viewMode === "preview" && !(typeof ctx.isMarkdownPreviewable === "function" && ctx.isMarkdownPreviewable(rel)) ? "file" : viewMode === "diff" && !canUseDiffView ? "file" : viewMode;
            }},
            async fetchFileOpenResult(request, rel, viewMode) {{
              if (viewMode === "diff") {{
                const pathTokenQuery = request.apiPath ? `&path_token=${{encodeURIComponent(request.apiPath)}}` : "";
                const res = await ctx.api(`/api/sessions/${{request.sessionId}}/git/file_versions?path=${{encodeURIComponent(rel)}}${{pathTokenQuery}}`, {{ signal: request.signal }});
                return {{ result: {{ kind: "diff", baseText: res && typeof res.base_text === "string" ? res.base_text : "", currentText: res && typeof res.current_text === "string" ? res.current_text : "", baseExists: res && res.base_exists, currentExists: res && res.current_exists }}, absPath: res && typeof res.abs_path === "string" ? res.abs_path : null }};
              }}
              const gitPathQuery = request.gitPath ? "&git_path=1" : "";
              const pathTokenQuery = request.gitPath && request.apiPath ? `&path_token=${{encodeURIComponent(request.apiPath)}}` : "";
              const res = await ctx.api(`/api/sessions/${{request.sessionId}}/file/read?path=${{encodeURIComponent(rel)}}${{pathTokenQuery}}${{gitPathQuery}}`, {{ signal: request.signal }});
              return {{ result: res, absPath: res && typeof res.path === "string" ? res.path : null }};
            }},
            isFileOpenAbortError(error) {{ return Boolean(error && error.name === "AbortError"); }},
            currentFileEditorState() {{
              const identity = this.currentActiveFileIdentity();
              return Object.freeze({{
                path: String(identity.path || ""),
                apiPath: String(identity.apiPath || ""),
                gitPath: Boolean(identity.gitPath),
                kind: String(ctx.activeFileKind || ""),
                editable: Boolean(ctx.activeFileEditable),
                version: String(ctx.activeFileVersion || ""),
                draft: Boolean(ctx.activeFileDraft),
                viewMode: String(ctx.fileViewMode || ""),
                editorKind: String(ctx.fileEditorKind || ""),
                editMode: Boolean(ctx.fileEditMode),
                dirty: Boolean(ctx.fileDirty),
                savePending: Boolean(ctx.fileSavePendingValue()),
                sessionId: String(ctx.fileViewerSessionId || ""),
                unavailable: ctx.isFileViewerSessionUnavailable(),
              }});
            }},
            fileEditorCapabilities(state) {{
              if (!state || typeof state !== "object") throw new Error("file editor state required");
              const kind = String(state.kind || "");
              const textKind = ctx.isTextFileKind(kind);
              const editable = Boolean(state.editable);
              const unavailable = Boolean(state.unavailable);
              const viewMode = String(state.viewMode || "");
              const editorKind = String(state.editorKind || "");
              const editMode = Boolean(state.editMode);
              const savePending = Boolean(state.savePending);
              const canEnterEditMode = Boolean(!unavailable && String(state.path || "") && !savePending && (!kind || textKind) && editorKind !== "plain-fallback" && editable);
              const writable = Boolean(editMode && editable && viewMode === "file" && !unavailable);
              const idleWritable = Boolean(writable && !savePending);
              const idleTextWritable = Boolean(idleWritable && textKind);
              const editModeAllowedInCurrentView = Boolean(viewMode === "file" && textKind && editable && !unavailable);
              return Object.freeze({{ canEnterEditMode, writable, idleWritable, idleTextWritable, editModeAllowedInCurrentView }});
            }},
            activeFileEditorCapabilities() {{ return this.fileEditorCapabilities(this.currentFileEditorState()); }},
            activeFileCanEnterEditMode() {{ return this.activeFileEditorCapabilities().canEnterEditMode; }},
            activeFileEditorWritable() {{ return this.activeFileEditorCapabilities().writable; }},
            activeFileEditorIdleWritable() {{ return this.activeFileEditorCapabilities().idleWritable; }},
            activeFileEditorIdleTextWritable() {{ return this.activeFileEditorCapabilities().idleTextWritable; }},
            activeFileEditModeAllowedInCurrentView() {{ return this.activeFileEditorCapabilities().editModeAllowedInCurrentView; }},
            syncFileEditorReadOnly() {{ if (typeof ctx.syncFileEditorReadOnly === "function") return ctx.syncFileEditorReadOnly(); }},
            updateFileEditButton() {{ if (typeof ctx.updateFileEditButton === "function") return ctx.updateFileEditButton(); }},
            maybeHandleUnsavedFileChanges() {{ return typeof ctx.maybeHandleUnsavedFileChanges === "function" ? ctx.maybeHandleUnsavedFileChanges() : !ctx.fileDirty; }},
            setFileViewModeWithGuard(mode) {{ return typeof ctx.setFileViewModeWithGuard === "function" ? ctx.setFileViewModeWithGuard(mode) : Promise.resolve(true); }},
            requestHideFileViewer() {{ return typeof ctx.requestHideFileViewer === "function" ? ctx.requestHideFileViewer() : Promise.resolve(true); }},
            openDraftFilePathWithGuard(path) {{ return typeof ctx.openDraftFilePathWithGuard === "function" ? ctx.openDraftFilePathWithGuard(path) : Promise.resolve(true); }},
            renderFileOpenError(request, error) {{
              if (this.isFileOpenAbortError(error)) return false;
              if (!this.isCurrentFileOpenRequest(request)) return false;
              ctx.resetActiveFileBufferState();
              ctx.fileStatus.textContent = `error: ${{error && error.message ? error.message : "unknown error"}}`;
              ctx.updateFileTouchToolbar();
              return false;
            }},
            renderDraftFileOpenError(request, error) {{
              if (this.isFileOpenAbortError(error)) return false;
              if (!this.isCurrentFileOpenRequest(request)) return false;
              ctx.resetActiveFileBufferState();
              ctx.fileStatus.textContent = `error: ${{error && error.message ? error.message : "unknown error"}}`;
              return false;
            }},
            applyDraftFileLoad: async (rel, request) => {{
              if (ctx.fileViewMode !== "file") ctx.setFileViewMode("file");
              ctx.applyActiveFileTextState({{ text: "", editable: true, version: "", draft: true }});
              ctx.applyFileMode();
              const rendered = await ctx.renderMonacoFile(rel, "", request.line, "", request);
              if (!rendered || !ctx.fileViewerController.isCurrentFileOpenRequest(request)) return false;
              ctx.setFileEditMode(true);
              ctx.fileStatus.textContent = `${{rel}} - new file`;
              ctx.rememberActiveFileSelection();
              ctx.renderFilePickerMenu();
              return true;
            }},
          }},
          fileOpenRequestId: 0,
          fileOpenAbortController: null,
          currentActiveFileIdentity: () => ctx.fileViewerController.currentActiveFileIdentity(),
          activeFilePathValue: () => ctx.fileViewerController.currentActiveFileIdentity().path,
          activeFileApiPathValue: () => ctx.fileViewerController.currentActiveFileIdentity().apiPath,
          activeFileGitPathValue: () => ctx.fileViewerController.currentActiveFileIdentity().gitPath,
          activeFileLineValue: () => ctx.fileViewerController.currentActiveFileLine(),
          fileApiPathForPath: (path, apiPath = "") => apiPath ? `kept:${{apiPath}}` : `derived:${{path}}`,
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          normalizeLineNumber: (value) => value == null || value === "" ? null : Number(value),
    """


def eval_video_preview_failure_path() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    file_helpers_source = APP_FILE_HELPERS_JS.read_text(encoding="utf-8")
    start = source.index("function fileVideoPreviewErrorText(err) {")
    end = source.index("function clearFileVideo() {", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const moduleCtx = {{ window: {{ CodoxearDisplay: {{ fmtBytes(value) {{ return String(value); }}, baseName(path) {{ return String(path || "").split("/").filter(Boolean).pop() || String(path || ""); }} }} }} }};
        vm.createContext(moduleCtx);
        vm.runInContext({json.dumps(file_helpers_source)}, moduleCtx);
        const events = [];
        const ctx = {{
          window: {{}},
          codoxearFileHelpers: moduleCtx.window.CodoxearFileHelpers,
          applyCount: 0,
          authLost: false,
          fileStatus: {{ textContent: "", replaceChildren(...nodes) {{ this.children = nodes; this.textContent = ""; }} }},
          fileEditButton: {{ classList: {{ toggle() {{}} }}, setAttribute() {{}}, disabled: false }},
          fileVideo: {{ src: "", loadCount: 0, load() {{ this.loadCount += 1; }} }},
          resolveAppUrl: (url) => `resolved:${{url}}`,
          handleAppAuthLoss: () => {{ ctx.authLost = true; }},
          fetch: async (_url, _opts) => ({{
            ok: false,
            status: 500,
            clone: () => ({{ json: async () => ({{ error: "video preview failed: bad codec" }}) }}),
            text: async () => "fallback text",
          }}),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(viewer_source)}, ctx);
        ctx.fileViewerController = ctx.window.CodoxearFileViewer.createFileViewerController({{
          el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children }}),
          fileStatus: ctx.fileStatus,
          fileEditButton: ctx.fileEditButton,
          iconSvg: (name) => `icon:${{name}}`,
          currentSessionId: () => "sid-1",
          currentFileSessionId: () => "sid-1",
          normalizeLineNumber: (value) => value == null || value === "" ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          isFileViewerOpen: () => true,
          hideFileUnsavedDialog: () => {{}},
          resetFileSearchState: () => {{}},
          closeFilePickerMenu: () => {{}},
          isTextFileKind: (kind) => kind === "text" || kind === "markdown",
          isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
          confirmReload: () => true,
          promptUnsavedFileChoice: async () => "cancel",
          restoreFileEditorText: () => {{}},
          hideFileViewer: () => {{}},
          setFilePath: () => {{}},
          resetFileViewerPanel: () => {{}},
          applyFileLoadResult: async () => true,
          normalizeDraftFilePath: (value) => String(value || "").trim(),
          inspectSessionFilePath: async () => ({{ exists: false }}),
          api: async () => ({{}}),
          focusEditor: () => null,
          disposeOpenRender: () => {{}},
          persistFileViewMode: () => {{}},
          persistFileNonDiffMode: () => {{}},
          isMarkdownPreviewable: () => false,
          updateFileTouchToolbar: () => {{}},
          isFileTouchToolbarActive: () => false,
          fileEditorShortcutBlocked: () => false,
          eventTargetElement: (value) => value || null,
          normalizeFileEditorPosition: (_editor, position) => position || null,
          applyFileEditorSelection: () => {{}},
          isCollapsedFileSelection: () => true,
          positionAfterInsertedText: (start, text) => ({{ lineNumber: start.lineNumber, column: start.column + String(text || "").length }}),
          fileEditorEditSupportAvailable: () => false,
          syncFileDiffSelectionMode: () => {{}},
          showFilePasteDialog: () => false,
          hideFilePasteDialog: () => {{}},
          clipboardReadAvailable: () => false,
          readClipboardText: async () => "",
          fileEditorDeleteCommandForKey: () => "",
          isActiveFileEditorInput: () => false,
          getActiveFileSelectionText: () => "",
          copyToClipboard: async () => {{}},
          focusActiveFileCodeEditor: () => null,
          nowMs: () => 0,
          setToast: (message) => events.push(["toast", message]),
          renderMonacoFile: async () => true,
          getFileEditorText: () => "",
          fmtBytes: (value) => String(value),
          applyFileMode: () => {{ ctx.applyCount += 1; }},
          rememberOpenedFile: () => {{}},
          historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
          renderFilePickerMenu: () => {{}},
        }});
        ctx.fileViewerController.setActiveFileIdentity("clip.mkv", {{ gitPath: false, apiPath: "" }});
        ctx.fileViewerController.setActiveVideoFallback({{ token: "video-1", previewUrl: "/preview.mp4", used: false, preparing: false, rel: "clip.mkv" }});
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_loadCompatibleVideoPreview = loadCompatibleVideoPreview;\n")}, ctx);
        (async () => {{
          const ok = await ctx.__test_loadCompatibleVideoPreview("video-1", {{ explicit: true }});
          const fallback = ctx.fileViewerController.currentActiveVideoFallback();
          process.stdout.write(JSON.stringify({{
            ok,
            status: ctx.fileStatus.textContent,
            used: fallback.used,
            preparing: fallback.preparing,
            applyCount: ctx.applyCount,
            videoSrc: ctx.fileVideo.src,
            loadCount: ctx.fileVideo.loadCount,
            authLost: ctx.authLost,
          }}));
        }})().catch((err) => {{ console.error(err && err.stack ? err.stack : err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_empty_file_viewer_target() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function renderEmptyFileViewerTarget(")
    end = source.index("async function ensureCurrentFileViewerSession", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const calls = [];
        const ctx = {{
          fileStatus: {{ textContent: "old" }},
          resetFileViewerPanel: () => calls.push(["resetFileViewerPanel"]),
          clearActiveFileIdentity: () => calls.push(["clearActiveFileIdentity"]),
          resetFilePickerInput: () => calls.push(["resetFilePickerInput"]),
          renderFilePickerMenu: () => calls.push(["renderFilePickerMenu"]),
          updateFileTouchToolbar: () => calls.push(["updateFileTouchToolbar"]),
          currentFileTouchSelectMode: () => false,
          isFileTouchToolbarActive: () => false,
          fileEditorShortcutBlocked: () => false,
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
          resetFileTouchSelectionState: (options) => calls.push(["resetFileTouchSelectionState", options || {{}}]),
          moveFileTouchSelection: (direction) => calls.push(["moveFileTouchSelection", direction]),
          fileEditorDeleteCommandForKey: () => "",
          isActiveFileEditorInput: () => false,
          getActiveFileSelectionText: () => "",
          copyToClipboard: async () => {{}},
          focusActiveFileCodeEditor: () => null,
          nowMs: () => 0,
          setToast: (message) => calls.push(["toast", message]),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_empty = renderEmptyFileViewerTarget;\n")}, ctx);
        ctx.__test_empty();
        const defaultCalls = calls.slice();
        const defaultStatus = ctx.fileStatus.textContent;
        calls.length = 0;
        ctx.fileStatus.textContent = "old again";
        ctx.__test_empty({{ updateTouchToolbar: true }});
        process.stdout.write(JSON.stringify({{
          defaultCalls,
          defaultStatus,
          touchCalls: calls.slice(),
          touchStatus: ctx.fileStatus.textContent,
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)



def eval_hide_file_viewer_identity_cleanup() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    clear_start = source.index("function clearActiveFileIdentity(")
    clear_end = source.index("function isCurrentFileOpenRequest", clear_start)
    hide_start = source.index("function hideFileViewer()")
    hide_end = source.index("function handleFileViewerSessionUnavailable", hide_start)
    snippet = source[clear_start:clear_end] + "\n" + source[hide_start:hide_end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const calls = [];
        const ctx = {{
{controller_identity_ctx_js("src/app.py", "token-1", True, 42)}
          fileViewerSessionId: "sid-1",
          fileViewerUnavailableSessionId: "",
          fileViewerSessionSyncToken: 10,
          fileViewerReturnFocusEl: {{ id: "return" }},
          fileBackdrop: {{ style: {{ display: "block" }} }},
          fileViewer: {{ style: {{ display: "block" }} }},
          filePickerSearchState: {{ setSessionId: (sid) => calls.push(["filePickerSearchState.setSessionId", sid]) }},
          modalOpen: true,
          normalizeLineNumber: (value) => value == null ? null : Number(value),
          isModalTargetOpen: () => ctx.modalOpen,
          cancelPendingFileOpen: () => calls.push(["cancelPendingFileOpen", ctx.activeFilePathValue()]),
          hideFileUnsavedDialog: () => calls.push(["hideFileUnsavedDialog", ctx.activeFilePathValue()]),
          hideFilePasteDialog: () => calls.push(["hideFilePasteDialog", ctx.activeFilePathValue()]),
          rememberActiveFileSelection: () => calls.push(["rememberActiveFileSelection", ctx.activeFilePathValue(), ctx.activeFileApiPathValue(), ctx.activeFileGitPathValue(), ctx.activeFileLineValue()]),
          resetFileViewerPanel: () => calls.push(["resetFileViewerPanel", ctx.activeFilePathValue()]),
          closeFilePickerMenu: (opts) => calls.push(["closeFilePickerMenu", opts, ctx.activeFilePathValue()]),
          resetFileSearchState: () => calls.push(["resetFileSearchState", ctx.activeFilePathValue()]),
          updateFileTouchToolbar: () => calls.push(["updateFileTouchToolbar", ctx.activeFilePathValue(), ctx.activeFileLineValue()]),
          afterModalVisibilityChanged: () => calls.push(["afterModalVisibilityChanged", ctx.fileViewer.style.display]),
          restoreModalFocus: (target, predicate) => calls.push(["restoreModalFocus", target && target.id, predicate()]),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_hide = hideFileViewer;\n")}, ctx);
        ctx.__test_hide();
        process.stdout.write(JSON.stringify({{
          calls,
          identity: {{ path: ctx.activeFilePathValue(), apiPath: ctx.activeFileApiPathValue(), gitPath: ctx.activeFileGitPathValue(), line: ctx.activeFileLineValue() }},
          session: {{ id: ctx.fileViewerSessionId, unavailable: ctx.fileViewerUnavailableSessionId, syncToken: ctx.fileViewerSessionSyncToken }},
          displays: {{ backdrop: ctx.fileBackdrop.style.display, viewer: ctx.fileViewer.style.display }},
          returnFocus: ctx.fileViewerReturnFocusEl,
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)



def eval_disable_file_viewer_for_unavailable_session() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    remember_start = source.index("function rememberActiveFileSelection(")
    remember_end = source.index("function historyFileSelectionForSession", remember_start)
    disable_start = source.index("function disableFileViewerForUnavailableSession(")
    disable_end = source.index("function handleFileViewerSessionUnavailable", disable_start)
    snippet = source[remember_start:remember_end] + "\n" + source[disable_start:disable_end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const calls = [];
        const ctx = {{
{controller_identity_ctx_js("src/app.py", "token-1", True, 42)}
          fileViewerSessionId: "dead-sid",
          selected: "dead-sid",
          fileViewerUnavailableSessionId: "",
          fileViewerSessionSyncToken: 5,
          activeFileSaveToken: 99,
          fileSavePending: true,
          fileEditMode: true,
          fileStatus: {{ textContent: "old" }},
          savedSelections: [],
          currentFileSessionId: () => String(ctx.fileViewerSessionId || ctx.selected || "").trim(),
          clearActiveFileSaveState: () => {{ ctx.activeFileSaveToken = 0; ctx.fileSavePending = false; calls.push(["clearActiveFileSaveState", ctx.fileViewerSessionSyncToken]); }},
          hideFileUnsavedDialog: (choice) => calls.push(["hideFileUnsavedDialog", choice, ctx.fileViewerSessionSyncToken]),
          cancelPendingFileOpen: () => calls.push(["cancelPendingFileOpen", ctx.fileViewerSessionSyncToken]),
          resetFileSearchState: () => calls.push(["resetFileSearchState", ctx.fileViewerSessionSyncToken]),
          closeFilePickerMenu: (opts) => calls.push(["closeFilePickerMenu", opts, ctx.fileViewerSessionSyncToken]),
          syncFileEditorReadOnly: () => calls.push(["syncFileEditorReadOnly", ctx.fileEditMode]),
          updateFileEditButton: () => calls.push(["updateFileEditButton", ctx.fileEditMode]),
          updateFileTouchToolbar: () => calls.push(["updateFileTouchToolbar", ctx.fileViewerUnavailableSessionId]),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_disable_unavailable = disableFileViewerForUnavailableSession;\n")}, ctx);
        ctx.__test_disable_unavailable("dead-sid");
        process.stdout.write(JSON.stringify({{
          calls,
          savedSelections: ctx.savedSelections,
          state: {{
            unavailable: ctx.fileViewerUnavailableSessionId,
            syncToken: ctx.fileViewerSessionSyncToken,
            saveToken: ctx.activeFileSaveToken,
            savePending: ctx.fileSavePending,
            editMode: ctx.fileEditMode,
            status: ctx.fileStatus.textContent,
            path: ctx.activeFilePathValue(),
            apiPath: ctx.activeFileApiPathValue(),
            gitPath: ctx.activeFileGitPathValue(),
            line: ctx.activeFileLineValue(),
          }},
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)



def eval_file_viewer_open_target() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function preferredFileSelectionForSession(")
    end = source.index("function fileVideoPreviewErrorText", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          selected: "sid-1",
          fileViewerSessionId: "viewer-sid",
          rememberedSelections: new Map(),
          fileCandidateList: [],
          fileEntryMap: new Map(),
          normalizeLineNumber: (value) => value == null || value === "" ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
          fileViewerController: {{
            preferredFileSelectionForSession: (sid) => {{
              const remembered = ctx.rememberedSelections.get(String(sid || "").trim());
              if (!remembered) return ctx.historyFileSelectionForSession(sid);
              return {{ path: remembered.path, apiPath: ctx.normalizeFileApiPath(remembered.apiPath), line: ctx.normalizeLineNumber(remembered.line), gitPath: Boolean(remembered.gitPath) }};
            }},
            resolveFileViewerOpenTarget: ({{ sessionId = "", explicitPath = "", explicitLine = null }} = {{}}) => {{
              const sid = String(sessionId || "").trim();
              if (!sid) return {{ kind: "none" }};
              const requestedPath = String(explicitPath ?? "");
              if (requestedPath) return {{ kind: "path", source: "explicit", path: requestedPath, line: ctx.normalizeLineNumber(explicitLine), changed: null, gitPath: false, apiPath: "" }};
              const preferred = ctx.fileViewerController.preferredFileSelectionForSession(sid);
              if (preferred.path) return {{ kind: "path", source: "preferred", path: preferred.path, line: preferred.line, changed: null, gitPath: Boolean(preferred.gitPath), apiPath: ctx.normalizeFileApiPath(preferred.apiPath) }};
              const firstKey = ctx.fileCandidateList.length ? ctx.fileCandidateList[0] : "";
              const first = firstKey ? ctx.fileEntryMap.get(firstKey) : null;
              if (first) return {{ kind: "path", source: "first", path: first.path, line: null, changed: Boolean(first.changed), gitPath: Boolean(first.gitPath), apiPath: ctx.normalizeFileApiPath(first.apiPath) }};
              return {{ kind: "none" }};
            }},
          }},
          sessionIndex: new Map(),
          listFromFilesField: () => [],
          sessionRelativePath: () => "",
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_target = resolveFileViewerOpenTarget;\n")}, ctx);
        ctx.rememberedSelections.set("sid-1", {{ path: "remembered.txt", line: "9", gitPath: true, apiPath: "api-remembered" }});
        ctx.fileCandidateList = ["first-key"];
        ctx.fileEntryMap.set("first-key", {{ path: "first.txt", changed: true, gitPath: true, apiPath: "api-first" }});
        const explicit = ctx.__test_target({{ sessionId: "sid-1", explicitPath: "explicit.md", explicitLine: "42" }});
        const preferred = ctx.__test_target({{ sessionId: "sid-1" }});
        ctx.rememberedSelections.clear();
        const first = ctx.__test_target({{ sessionId: "sid-1" }});
        ctx.fileCandidateList = [];
        ctx.fileEntryMap.clear();
        const none = ctx.__test_target({{ sessionId: "sid-1" }});
        const noSession = ctx.__test_target({{ sessionId: "" }});
        process.stdout.write(JSON.stringify({{ explicit, preferred, first, none, noSession }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)



def eval_use_touch_file_editor_controls(query_matches: dict[str, bool]) -> bool:
    viewport_source = APP_VIEWPORT_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const queryMatches = {json.dumps(query_matches)};
        const ctx = {{
          window: {{
            matchMedia: (query) => ({{ matches: Boolean(queryMatches[query]) }}),
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(viewport_source)}, ctx);
        process.stdout.write(JSON.stringify(ctx.window.CodoxearViewport.useTouchFileEditorControls()));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_open_file_reference_nonliteral() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    markdown_source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
    prelude_start = source.index("const codoxearMarkdown = window.CodoxearMarkdown;")
    prelude_end = source.index("function iconSvg", prelude_start)
    prelude = source[prelude_start:prelude_end]
    open_start = source.index("async function openFileReference(ref) {")
    open_end = source.index("async function confirmDirectorySession", open_start)
    snippet = prelude + "\n" + source[open_start:open_end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          URL,
          console,
          location: {{ origin: "http://localhost", href: "http://localhost/" }},
          window: {{
            CodoxearUrls: {{
              resolveAppUrl: (path) => new URL(String(path ?? "").replace(/^[/]/, ""), "http://localhost/").toString(),
            }},
          }},
          selected: "sid-1",
          sessionIndex: new Map([["sid-1", {{ session_id: "sid-1", cwd: "/repo" }}]]),
          toastMessages: [],
          showCalls: [],
          selectCalls: [],
          fmtBytes: (n) => `${{n}} B`,
        }};
        ctx.setToast = (message) => ctx.toastMessages.push(String(message));
        ctx.showFileViewer = (options) => {{ ctx.showCalls.push(options); return Promise.resolve(); }};
        ctx.sessionRelativePath = () => null;
        ctx.selectSession = async (sessionId) => {{ ctx.selectCalls.push(sessionId); ctx.selected = sessionId; }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(markdown_source)}, ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_openFileReference = openFileReference;\n")}, ctx);
        (async () => {{
          await ctx.__test_openFileReference({{ path: "src/app.py", line: 7 }});
          await ctx.__test_openFileReference({{ path: "not a local ref" }});
          process.stdout.write(JSON.stringify({{
            showCalls: ctx.showCalls,
            toastMessages: ctx.toastMessages,
            selectCalls: ctx.selectCalls,
          }}));
        }})().catch((err) => {{ console.error(err && err.stack ? err.stack : err); process.exit(1); }});
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_file_paste_dialog_fallback() -> dict:
    app_source = APP_JS.read_text(encoding="utf-8")
    viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    start = app_source.index("function hideFilePasteDialog({ restoreFocus = false } = {}) {")
    end = app_source.index("async function pasteFromClipboardIntoActiveFile", start)
    dialog_snippet = app_source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const dialogSnippet = {json.dumps(dialog_snippet + "\nglobalThis.__test_showPaste = showFilePasteDialog;\nglobalThis.__test_hidePaste = hideFilePasteDialog;\n")};
        const viewerSource = {json.dumps(viewer_source)};
        async function runCase(opts) {{
          const ctx = {{ window: {{}} }};
          ctx.filePasteBackdrop = {{ style: {{ display: "none" }} }};
          ctx.filePasteDialog = {{ style: {{ display: "none" }} }};
          ctx.filePasteInput = {{
            value: "stale",
            focusCount: 0,
            selectCount: 0,
            focus() {{ this.focusCount += 1; }},
            select() {{ this.selectCount += 1; }},
          }};
          ctx.toastMessages = [];
          ctx.inserted = [];
          ctx.focusEditorCount = 0;
          ctx.prepareCount = 0;
          ctx.modalSyncCount = 0;
          ctx.rafCount = 0;
          ctx.dirtyValues = [];
          ctx.activeFileEditorIdleTextWritable = () => true;
          ctx.prepareModalOpen = () => {{ ctx.prepareCount += 1; }};
          ctx.afterModalVisibilityChanged = () => {{ ctx.modalSyncCount += 1; }};
          ctx.requestAnimationFrame = (cb) => {{ ctx.rafCount += 1; cb(); }};
          ctx.focusActiveFileCodeEditor = () => {{ ctx.focusEditorCount += 1; return editor; }};
          ctx.currentFileEditorKind = () => "file";
          ctx.fileEditorRuntime = {{ focusActiveCodeEditor: () => ctx.focusActiveFileCodeEditor() }};
          vm.createContext(ctx);
          vm.runInContext(dialogSnippet, ctx);
          vm.runInContext(viewerSource, ctx);
          const pos = {{ lineNumber: 1, column: 1 }};
          const model = {{ value: "", getValue() {{ return this.value; }} }};
          const editor = {{
            getPosition: () => ({{ ...pos }}),
            getSelection: () => null,
            getModel: () => model,
            updateOptions() {{}},
            pushUndoStop() {{}},
            executeEdits(_source, edits) {{
              const text = String(edits && edits[0] && edits[0].text || "");
              ctx.inserted.push(text);
              model.value += text;
            }},
          }};
          const fileStatus = {{ textContent: "", replaceChildren() {{}} }};
          const fileEditButton = {{ classList: {{ toggle() {{}} }}, setAttribute() {{}} }};
          let controller = null;
          controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
            el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children: Array.isArray(children) ? children : [] }}),
            fileStatus,
            fileEditButton,
            iconSvg: (name) => `icon:${{name}}`,
            currentSessionId: () => "sid-1",
            currentFileSessionId: () => "sid-1",
            normalizeLineNumber: (value) => value == null || value === "" ? null : Number(value),
            normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
            fileApiPathForPath: (_path, existing) => existing || "",
            isFileViewerOpen: () => true,
            invalidateFileViewerSessionSync: () => {{}},
            hideFileUnsavedDialog: () => {{}},
            resetFileSearchState: () => {{}},
            closeFilePickerMenu: () => {{}},
            isTextFileKind: (kind) => kind === "text" || kind === "markdown",
            isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
            confirmReload: () => true,
            promptUnsavedFileChoice: async () => "cancel",
            restoreFileEditorText: () => {{}},
            hideFileViewer: () => {{}},
            applyFileLoadResult: async () => true,
            setFilePath: () => {{}},
            resetFileViewerPanel: () => {{}},
            normalizeDraftFilePath: (value) => String(value || "").trim(),
            inspectSessionFilePath: async () => ({{ exists: false }}),
            api: async () => ({{}}),
            focusEditor: () => editor,
            disposeOpenRender: () => {{}},
            initialFileViewMode: "file",
            initialFileNonDiffMode: "file",
            persistFileViewMode: (mode) => {{ if (ctx && ctx.calls) ctx.calls.push(["persistFileViewMode", mode]); }},
            persistFileNonDiffMode: (mode) => {{ if (ctx && ctx.calls) ctx.calls.push(["persistFileNonDiffMode", mode]); }},
            currentFileEditorKind: () => "file",
            currentFileEditMode: () => true,
            activeFileEntry: () => null,
            fileCandidateGitStateFresh: () => false,
            isMarkdownPreviewable: () => false,
            resetActiveFileBufferState: () => {{}},
            updateFileTouchToolbar: () => {{}},
            isFileTouchToolbarActive: () => false,
            fileEditorShortcutBlocked: () => false,
            eventTargetElement: (value) => value || null,
            normalizeFileEditorPosition: (_editor, position) => position ? {{ lineNumber: Number(position.lineNumber) || 1, column: Number(position.column) || 1 }} : null,
            applyFileEditorSelection: (_editor, cursor) => {{ pos.lineNumber = cursor.lineNumber; pos.column = cursor.column; }},
            isCollapsedFileSelection: (selection) => !selection || (selection.startLineNumber === selection.endLineNumber && selection.startColumn === selection.endColumn),
            positionAfterInsertedText: (start, text) => ({{ lineNumber: Number(start && start.lineNumber) || 1, column: (Number(start && start.column) || 1) + String(text || "").length }}),
            fileEditorEditSupportAvailable: () => opts.insertOk !== false,
            syncFileDiffSelectionMode: () => {{}},
            showFilePasteDialog: () => ctx.__test_showPaste(),
            hideFilePasteDialog: (options) => ctx.__test_hidePaste(options),
            clipboardReadAvailable: () => Boolean(opts.secure && opts.clipboard !== "missing"),
            readClipboardText: async () => {{ if (opts.clipboard === "deniedAfterReadonly") {{ controller.setFileEditMode(false); throw new Error("denied"); }} if (opts.clipboard === "denied") throw new Error("denied"); return opts.clipboardText || ""; }},
            fileEditorDeleteCommandForKey: () => "",
            isActiveFileEditorInput: () => false,
            getActiveFileSelectionText: () => "",
            copyToClipboard: async () => {{}},
            focusActiveFileCodeEditor: () => ctx.focusActiveFileCodeEditor(),
            nowMs: () => 0,
            setToast: (message) => ctx.toastMessages.push(String(message)),
            setFileViewMode: () => {{}},
            applyActiveFileTextState: () => {{}},
            renderMonacoFile: async () => true,
            setFileEditMode: () => {{}},
            currentActiveFileKind: () => "text",
            currentActiveFileDraft: () => false,
            currentActiveFileVersion: () => "v1",
            currentActiveFileEditable: () => true,
            currentFileDirty: () => false,
            currentActiveFileText: () => "",
            getFileEditorText: () => model.getValue(),
            setFileDirty: (dirty) => ctx.dirtyValues.push(Boolean(dirty)),
            fmtBytes: (value) => `${{value}}B`,
            applyFileMode: () => {{}},
            rememberOpenedFile: () => {{}},
            historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
            rememberActiveFileSelection: () => {{}},
            renderFilePickerMenu: () => {{}},
          }});
          controller.setActiveFileIdentity("note.txt", {{}});
          controller.applyActiveFileTextState({{ kind: "text", text: model.value, editable: true, version: "v1", draft: false }});
          controller.setFileEditMode(true);
          await controller.pasteFromClipboardIntoActiveFile();
          if (opts.hideAfter) ctx.__test_hidePaste({{ restoreFocus: true }});
          return {{
            backdrop: ctx.filePasteBackdrop.style.display,
            dialog: ctx.filePasteDialog.style.display,
            inputValue: ctx.filePasteInput.value,
            focusCount: ctx.filePasteInput.focusCount,
            selectCount: ctx.filePasteInput.selectCount,
            toasts: ctx.toastMessages,
            inserted: ctx.inserted,
            focusEditorCount: ctx.focusEditorCount,
            prepareCount: ctx.prepareCount,
            modalSyncCount: ctx.modalSyncCount,
            rafCount: ctx.rafCount,
            dirty: controller.currentFileDirty(),
          }};
        }}
        (async () => {{
          const missing = await runCase({{ secure: false, clipboard: "missing" }});
          const denied = await runCase({{ secure: true, clipboard: "denied" }});
          const deniedAfterReadonly = await runCase({{ secure: true, clipboard: "deniedAfterReadonly" }});
          const direct = await runCase({{ secure: true, clipboard: "ok", clipboardText: "hello" }});
          const empty = await runCase({{ secure: true, clipboard: "ok", clipboardText: "" }});
          const dismissed = await runCase({{ secure: false, clipboard: "missing", hideAfter: true }});
          process.stdout.write(JSON.stringify({{ missing, denied, deniedAfterReadonly, direct, empty, dismissed }}));
        }})().catch((err) => {{ console.error(err && err.stack ? err.stack : err); process.exit(1); }});
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)

def eval_file_paste_insert_button_guard() -> dict:
    source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        function runCase(unavailable, text) {{
          const state = {{ unavailable: Boolean(unavailable), hidden: 0, toasts: [], status: [], inserted: [], dirtyValues: [], inputValue: String(text || "") }};
          const pos = {{ lineNumber: 1, column: 1 }};
          const model = {{ value: "", getValue() {{ return this.value; }} }};
          const editor = {{
            getPosition: () => ({{ ...pos }}),
            getSelection: () => null,
            getModel: () => model,
            updateOptions() {{}},
            pushUndoStop() {{}},
            executeEdits(_source, edits) {{ const value = String(edits && edits[0] && edits[0].text || ""); state.inserted.push(value); model.value += value; }},
          }};
          const fileStatus = {{ textContent: "", replaceChildren() {{}} }};
          const fileEditButton = {{ classList: {{ toggle() {{}} }}, setAttribute() {{}} }};
          const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
            el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children: Array.isArray(children) ? children : [] }}),
            fileStatus,
            fileEditButton,
            iconSvg: (name) => `icon:${{name}}`,
            currentSessionId: () => "sid-1",
            currentFileSessionId: () => "sid-1",
            normalizeLineNumber: (value) => value == null || value === "" ? null : Number(value),
            normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
            fileApiPathForPath: (_path, existing) => existing || "",
            isFileViewerOpen: () => true,
            invalidateFileViewerSessionSync: () => {{}},
            hideFileUnsavedDialog: () => {{}},
            resetFileSearchState: () => {{}},
            closeFilePickerMenu: () => {{}},
            isTextFileKind: (kind) => kind === "text" || kind === "markdown",
            isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
            confirmReload: () => true,
            promptUnsavedFileChoice: async () => "cancel",
            restoreFileEditorText: () => {{}},
            hideFileViewer: () => {{}},
            applyFileLoadResult: async () => true,
            setFilePath: () => {{}},
            resetFileViewerPanel: () => {{}},
            normalizeDraftFilePath: (value) => String(value || "").trim(),
            inspectSessionFilePath: async () => ({{ exists: false }}),
            api: async () => ({{}}),
            focusEditor: () => editor,
            disposeOpenRender: () => {{}},
            initialFileViewMode: "file",
            initialFileNonDiffMode: "file",
            persistFileViewMode: (mode) => {{ if (ctx && ctx.calls) ctx.calls.push(["persistFileViewMode", mode]); }},
            persistFileNonDiffMode: (mode) => {{ if (ctx && ctx.calls) ctx.calls.push(["persistFileNonDiffMode", mode]); }},
            currentFileEditorKind: () => "file",
            currentFileEditMode: () => true,
            activeFileEntry: () => null,
            fileCandidateGitStateFresh: () => false,
            isMarkdownPreviewable: () => false,
            resetActiveFileBufferState: () => {{}},
            updateFileTouchToolbar: () => {{}},
            isFileTouchToolbarActive: () => false,
            fileEditorShortcutBlocked: () => false,
            eventTargetElement: (value) => value || null,
            normalizeFileEditorPosition: (_editor, position) => position ? {{ lineNumber: Number(position.lineNumber) || 1, column: Number(position.column) || 1 }} : null,
            applyFileEditorSelection: (_editor, cursor) => {{ pos.lineNumber = cursor.lineNumber; pos.column = cursor.column; }},
            isCollapsedFileSelection: (selection) => !selection || (selection.startLineNumber === selection.endLineNumber && selection.startColumn === selection.endColumn),
            positionAfterInsertedText: (start, value) => ({{ lineNumber: Number(start && start.lineNumber) || 1, column: (Number(start && start.column) || 1) + String(value || "").length }}),
            fileEditorEditSupportAvailable: () => true,
            syncFileDiffSelectionMode: () => {{}},
            showFilePasteDialog: () => false,
            hideFilePasteDialog: () => {{ state.hidden += 1; state.inputValue = ""; }},
            clipboardReadAvailable: () => false,
            readClipboardText: async () => "",
            fileEditorDeleteCommandForKey: () => "",
            isActiveFileEditorInput: () => false,
            getActiveFileSelectionText: () => "",
            copyToClipboard: async () => {{}},
            focusActiveFileCodeEditor: () => editor,
            nowMs: () => 0,
            setToast: (message) => state.toasts.push(String(message)),
            setFileViewMode: () => {{}},
            applyActiveFileTextState: () => {{}},
            renderMonacoFile: async () => true,
            setFileEditMode: () => {{}},
            currentActiveFileKind: () => "text",
            currentActiveFileDraft: () => false,
            currentActiveFileVersion: () => "v1",
            currentActiveFileEditable: () => true,
            currentFileDirty: () => false,
            currentActiveFileText: () => "",
            getFileEditorText: () => model.getValue(),
            setFileDirty: (dirty) => state.dirtyValues.push(Boolean(dirty)),
            fmtBytes: (value) => `${{value}}B`,
            applyFileMode: () => {{}},
            rememberOpenedFile: () => {{}},
            historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
            rememberActiveFileSelection: () => {{}},
            renderFilePickerMenu: () => {{}},
          }});
          controller.setActiveFileIdentity("note.txt", {{}});
          controller.applyActiveFileTextState({{ kind: "text", text: model.value, editable: true, version: "v1", draft: false }});
          controller.setFileEditMode(true);
          if (state.unavailable) controller.disableFileViewerForUnavailableSession("sid-1");
          const result = controller.handleFilePasteInsert(state.inputValue);
          return {{ result, inputValue: state.inputValue, inserted: state.inserted, hidden: state.hidden, toasts: state.toasts, status: fileStatus.textContent, dirty: controller.currentFileDirty() }};
        }}
        const unavailable = runCase(true, "typed text");
        const available = runCase(false, "allowed text");
        process.stdout.write(JSON.stringify({{ unavailable, available }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_file_editor_capability_predicates() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function currentFileEditorState() {")
    end = source.index("function syncFileEditorReadOnly()", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const snippet = {json.dumps(snippet + "\nglobalThis.__test_eval = () => { const state = currentFileEditorState(); return { state, capabilities: fileEditorCapabilities(state), wrappers: { canEnter: activeFileCanEnterEditMode(), writable: activeFileEditorWritable(), idleWritable: activeFileEditorIdleWritable(), idleTextWritable: activeFileEditorIdleTextWritable(), editModeAllowed: activeFileEditModeAllowedInCurrentView() } }; };\n")};
        function runCase(overrides = {{}}) {{
          const ctx = {{
{controller_identity_ctx_js("", "", False, None)}
            activeFileKind: overrides.kind || "markdown",
            activeFileEditable: overrides.editable !== false,
            activeFileVersion: overrides.version || "v1",
            activeFileDraft: Boolean(overrides.draft),
            fileViewMode: overrides.viewMode || "file",
            fileEditorKind: overrides.editorKind || "file",
            fileEditMode: overrides.editMode !== false,
            fileDirty: Boolean(overrides.dirty),
            fileSavePending: Boolean(overrides.pending),
            fileSavePendingValue: () => ctx.fileSavePending,
            fileViewerSessionId: overrides.sessionId === false ? "" : "sid-1",
            unavailable: Boolean(overrides.unavailable),
            isTextFileKind: (kind) => kind === "text" || kind === "markdown",
            isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
            isFileViewerSessionUnavailable: () => ctx.unavailable,
          }};
          ctx.fileViewerController.setActiveFileIdentity(overrides.path === false ? "" : "note.md", {{ gitPath: Boolean(overrides.gitPath), apiPath: overrides.apiPath || "" }});
          vm.createContext(ctx);
          vm.runInContext(snippet, ctx);
          return ctx.__test_eval();
        }}
        const cases = {{
          editableText: runCase(),
          savePending: runCase({{ pending: true }}),
          previewMode: runCase({{ viewMode: "preview" }}),
          binaryKind: runCase({{ kind: "image" }}),
          unavailable: runCase({{ unavailable: true }}),
          plainFallback: runCase({{ editorKind: "plain-fallback" }}),
          notEditing: runCase({{ editMode: false }}),
          missingPath: runCase({{ path: false }}),
        }};
        process.stdout.write(JSON.stringify(cases));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_file_editor_save_shortcut() -> dict:
    viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        class FakeElement {{
          constructor(opts = {{}}) {{
            this.textEntry = Boolean(opts.textEntry);
            this.editorInput = Boolean(opts.editorInput);
            this._inputarea = Boolean(opts.inputarea);
            this.classList = {{ contains: (name) => name === "inputarea" && this._inputarea }};
          }}
        }}
        const ctx = {{ window: {{}}, HTMLElement: FakeElement, AbortController }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(viewer_source)}, ctx);
        const fileViewer = ctx.window.CodoxearFileViewer;
        function el(tag, attrs = {{}}, children = []) {{ return {{ tag, attrs, children: Array.isArray(children) ? children : [], text: attrs && attrs.text }}; }}
        async function runCase(overrides = {{}}) {{
          const events = [];
          const state = {{
            sessionId: overrides.sessionId === false ? "" : "sid-1",
            editorKind: overrides.editorKind || "file",
            viewerOpen: overrides.viewerOpen !== false,
            nestedDialog: Boolean(overrides.nestedDialog),
            dirty: overrides.dirty !== false,
          }};
          const editorNode = {{ contains: (node) => Boolean(node && node.editorInput) }};
          const fileStatus = {{ textContent: "", replaceChildren() {{}} }};
          const fileEditButton = {{
            disabled: false,
            innerHTML: "",
            title: "",
            classList: {{ toggle() {{}} }},
            setAttribute() {{}},
          }};
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
            fileApiPathForPath: (_path, existing) => existing || "tok",
            isFileViewerOpen: () => state.viewerOpen,
            invalidateFileViewerSessionSync: () => events.push(["invalidate"]),
            hideFileUnsavedDialog: (choice) => events.push(["hideUnsaved", choice]),
            resetFileSearchState: () => events.push(["resetSearch"]),
            closeFilePickerMenu: (options) => events.push(["closePicker", options]),
            isTextFileKind: (kind) => kind === "text" || kind === "markdown",
            isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
            confirmReload: () => true,
            promptUnsavedFileChoice: async () => "cancel",
            restoreFileEditorText: () => {{}},
            hideFileViewer: () => events.push(["hideViewer"]),
            setFilePath: () => {{}},
            resetFileViewerPanel: () => events.push(["resetPanel"]),
            applyFileLoadResult: async () => true,
            normalizeDraftFilePath: (value) => String(value || "").trim().replace(/^[/]+/, ""),
            inspectSessionFilePath: async () => ({{ exists: false }}),
            api: async (url, options = {{}}) => {{ events.push(["api", url, options.method || "GET", options.body || null]); return {{ version: "v2", editable: true, size: 4, path: "/abs/note.txt" }}; }},
            focusEditor: () => ({{ updateOptions: (opts) => events.push(["editorOptions", opts]) }}),
            disposeOpenRender: () => events.push(["disposeOpenRender"]),
            initialFileViewMode: "file",
            initialFileNonDiffMode: "file",
            persistFileViewMode: () => {{}},
            persistFileNonDiffMode: () => {{}},
            currentFileEditorKind: () => state.editorKind,
            activeFileEntry: () => null,
            fileCandidateGitStateFresh: () => false,
            isMarkdownPreviewable: () => true,
            updateFileTouchToolbar: () => events.push(["touchToolbar"]),
            isFileTouchToolbarActive: () => true,
            fileEditorShortcutBlocked: (target) => Boolean(!state.viewerOpen || state.nestedDialog || (target && target.textEntry && !target.editorInput)),
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
            fileEditorDeleteCommandForKey: () => "",
            isActiveFileEditorInput: (target) => Boolean(target && target.inputarea && editorNode.contains(target)),
            getActiveFileSelectionText: () => "",
            copyToClipboard: async () => {{}},
            focusActiveFileCodeEditor: () => null,
            nowMs: () => 0,
            setToast: (message) => events.push(["toast", message]),
            renderMonacoFile: async () => true,
            getFileEditorText: () => "body",
            fmtBytes: (value) => `${{value}}B`,
            applyFileMode: () => events.push(["applyFileMode"]),
            rememberOpenedFile: (rel, absPath) => events.push(["rememberOpenedFile", rel, absPath]),
            historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
            rememberActiveFileSelection: () => events.push(["rememberSelection"]),
            renderFilePickerMenu: () => events.push(["renderPicker"]),
          }});
          controller.setActiveFileIdentity(overrides.path === false ? "" : "note.txt", {{ gitPath: Boolean(overrides.gitPath), apiPath: overrides.apiPath || "" }});
          if (overrides.kind === "image") controller.applyActiveFileNonTextState("image");
          else controller.applyActiveFileTextState({{ kind: overrides.kind || "text", text: "body", editable: overrides.editable !== false, version: overrides.version || "v1", draft: Boolean(overrides.draft) }});
          controller.setFileEditMode(overrides.editMode !== false);
          controller.setFileDirty(state.dirty);
          if (overrides.pending) controller.markActiveFileSavePending({{ path: "note.txt" }});
          if (overrides.unavailable) controller.disableFileViewerForUnavailableSession("sid-1");
          events.length = 0;
          const event = {{
            key: overrides.key || "s",
            ctrlKey: overrides.ctrl !== false,
            metaKey: Boolean(overrides.meta),
            altKey: Boolean(overrides.alt),
            shiftKey: Boolean(overrides.shift),
            isComposing: Boolean(overrides.composing),
            defaultPrevented: Boolean(overrides.defaultPrevented),
            target: overrides.target || null,
            prevented: 0,
            stopped: 0,
            preventDefault() {{ this.prevented += 1; }},
            stopPropagation() {{ this.stopped += 1; }},
          }};
          const handled = controller.handleFileEditorSaveShortcut(event);
          await Promise.resolve();
          await Promise.resolve();
          return {{ handled, prevented: event.prevented, stopped: event.stopped, events: events.slice(), apiEvents: events.filter((entry) => entry[0] === "api") }};
        }}
        (async () => {{
          const editorInput = new FakeElement({{ textEntry: true, inputarea: true, editorInput: true }});
          const otherInput = new FakeElement({{ textEntry: true, inputarea: false, editorInput: false }});
          const validCtrl = await runCase();
          const validMeta = await runCase({{ ctrl: false, meta: true, target: editorInput }});
          const noModifier = await runCase({{ ctrl: false }});
          const wrongKey = await runCase({{ key: "p" }});
          const notEdit = await runCase({{ editMode: false }});
          const pending = await runCase({{ pending: true }});
          const unavailable = await runCase({{ unavailable: true }});
          const nestedDialog = await runCase({{ nestedDialog: true }});
          const otherTextEntry = await runCase({{ target: otherInput }});
          const noPath = await runCase({{ path: false }});
          const viewerClosed = await runCase({{ viewerOpen: false }});
          process.stdout.write(JSON.stringify({{ validCtrl, validMeta, noModifier, wrongKey, notEdit, pending, unavailable, nestedDialog, otherTextEntry, noPath, viewerClosed }}));
        }})().catch((err) => {{ console.error(err && err.stack ? err.stack : err); process.exit(1); }});
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)

def eval_file_touch_selection_keydown() -> dict:
    source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        class FakeElement {{
          constructor(opts = {{}}) {{
            this.textEntry = Boolean(opts.textEntry);
            this.editorInput = Boolean(opts.editorInput);
            this.inViewer = opts.inViewer !== false;
            this.shortcutBlocked = Boolean(opts.shortcutBlocked);
          }}
          closest(selector) {{
            return selector === "#fileViewer" && this.inViewer ? this : null;
          }}
        }}
        function runCase(overrides = {{}}) {{
          const events = {{ moves: [], selections: [], syncReadOnly: 0, syncDiff: 0, toolbar: 0, focus: 0, toasts: [] }};
          const pos = {{ lineNumber: 1, column: 1 }};
          const editor = {{
            getPosition: () => ({{ ...pos }}),
            getModel: () => ({{ getLineCount: () => 20, getLineMaxColumn: () => 80 }}),
            trigger(source, command, args) {{
              if (command === "cursorMove") {{
                events.moves.push(args.to);
                if (args.to === "right") pos.column += 1;
                if (args.to === "left") pos.column = Math.max(1, pos.column - 1);
                if (args.to === "down") pos.lineNumber += 1;
                if (args.to === "up") pos.lineNumber = Math.max(1, pos.lineNumber - 1);
              }}
            }},
          }};
          const fileStatus = {{ textContent: "", replaceChildren() {{}} }};
          const fileEditButton = {{ classList: {{ toggle() {{}} }}, setAttribute() {{}} }};
          const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
            el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children: Array.isArray(children) ? children : [] }}),
            fileStatus,
            fileEditButton,
            iconSvg: (name) => `icon:${{name}}`,
            currentSessionId: () => "sid-1",
            currentFileSessionId: () => "sid-1",
            normalizeLineNumber: (value) => value == null || value === "" ? null : Number(value),
            normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
            fileApiPathForPath: (_path, existing) => existing || "",
            isFileViewerOpen: () => overrides.viewerOpen !== false,
            invalidateFileViewerSessionSync: () => {{}},
            hideFileUnsavedDialog: () => {{}},
            resetFileSearchState: () => {{}},
            closeFilePickerMenu: () => {{}},
            isTextFileKind: (kind) => kind === "text" || kind === "markdown",
            isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
            confirmReload: () => true,
            promptUnsavedFileChoice: async () => "cancel",
            restoreFileEditorText: () => {{}},
            hideFileViewer: () => {{}},
            applyFileLoadResult: async () => true,
            setFilePath: () => {{}},
            resetFileViewerPanel: () => {{}},
            normalizeDraftFilePath: (value) => String(value || "").trim(),
            inspectSessionFilePath: async () => ({{ exists: false }}),
            api: async () => ({{}}),
            focusEditor: () => editor,
            disposeOpenRender: () => {{}},
            initialFileViewMode: "file",
            initialFileNonDiffMode: "file",
            persistFileViewMode: (mode) => {{ if (ctx && ctx.calls) ctx.calls.push(["persistFileViewMode", mode]); }},
            persistFileNonDiffMode: (mode) => {{ if (ctx && ctx.calls) ctx.calls.push(["persistFileNonDiffMode", mode]); }},
            currentFileEditorKind: () => "file",
            currentFileEditMode: () => false,
            activeFileEntry: () => null,
            fileCandidateGitStateFresh: () => false,
            isMarkdownPreviewable: () => false,
            resetActiveFileBufferState: () => {{}},
            updateFileTouchToolbar: () => {{ events.toolbar += 1; }},
            isFileTouchToolbarActive: () => overrides.toolbarActive !== false && overrides.viewerOpen !== false,
            fileEditorShortcutBlocked: (target) => Boolean(overrides.nestedDialog || (target && target.shortcutBlocked) || (target && target.textEntry && !target.editorInput)),
            eventTargetElement: (value) => value instanceof FakeElement ? value : null,
            normalizeFileEditorPosition: (_editor, position) => position ? {{ lineNumber: Number(position.lineNumber) || 1, column: Number(position.column) || 1 }} : null,
            applyFileEditorSelection: (_editor, cursor, anchor) => {{ events.selections.push({{ cursor, anchor: anchor || null }}); }},
            isCollapsedFileSelection: (selection) => !selection || (selection.startLineNumber === selection.endLineNumber && selection.startColumn === selection.endColumn),
            positionAfterInsertedText: (start, text) => ({{ lineNumber: Number(start && start.lineNumber) || 1, column: (Number(start && start.column) || 1) + String(text || "").length }}),
            fileEditorEditSupportAvailable: () => true,
            syncFileDiffSelectionMode: () => {{ events.syncDiff += 1; }},
            showFilePasteDialog: () => false,
            hideFilePasteDialog: () => {{}},
            clipboardReadAvailable: () => false,
            readClipboardText: async () => "",
            fileEditorDeleteCommandForKey: () => "",
            isActiveFileEditorInput: () => false,
            getActiveFileSelectionText: () => "",
            copyToClipboard: async () => {{}},
            focusActiveFileCodeEditor: () => {{ events.focus += 1; return editor; }},
            nowMs: () => 0,
            setToast: (message) => events.toasts.push(message),
            setFileViewMode: () => {{}},
            applyActiveFileTextState: () => {{}},
            renderMonacoFile: async () => true,
            setFileEditMode: () => {{}},
            currentActiveFileKind: () => "text",
            currentActiveFileDraft: () => false,
            currentActiveFileVersion: () => "",
            currentActiveFileEditable: () => true,
            currentFileDirty: () => false,
            currentActiveFileText: () => "",
            getFileEditorText: () => "",
            setFileDirty: () => {{}},
            fmtBytes: (value) => `${{value}}B`,
            applyFileMode: () => {{}},
            rememberOpenedFile: () => {{}},
            historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
            rememberActiveFileSelection: () => {{}},
            renderFilePickerMenu: () => {{}},
          }});
          controller.setActiveFileIdentity("note.txt", {{}});
          if (overrides.selectMode !== false) controller.toggleFileTouchSelectionMode();
          const target = Object.prototype.hasOwnProperty.call(overrides, "target") ? overrides.target : new FakeElement({{ inViewer: true }});
          const event = {{
            key: overrides.key || "h",
            ctrlKey: Boolean(overrides.ctrl),
            metaKey: Boolean(overrides.meta),
            altKey: Boolean(overrides.alt),
            defaultPrevented: Boolean(overrides.defaultPrevented),
            target,
            prevented: 0,
            stopped: 0,
            preventDefault() {{ this.prevented += 1; }},
            stopPropagation() {{ this.stopped += 1; }},
          }};
          controller.handleFileTouchSelectionKeydown(event);
          return {{ prevented: event.prevented, stopped: event.stopped, moves: events.moves, mode: controller.currentFileTouchSelectMode(), selections: events.selections }};
        }}
        (() => {{
          const editorInput = new FakeElement({{ textEntry: true, editorInput: true, inViewer: true }});
          const otherInput = new FakeElement({{ textEntry: true, editorInput: false, inViewer: false }});
          const validMove = runCase({{ target: editorInput, key: "l" }});
          const validEscape = runCase({{ target: editorInput, key: "Escape" }});
          const printableBlocked = runCase({{ target: editorInput, key: "x" }});
          const nestedDialog = runCase({{ target: editorInput, nestedDialog: true, key: "l" }});
          const viewerClosed = runCase({{ target: editorInput, viewerOpen: false, key: "l" }});
          const otherTextEntry = runCase({{ target: otherInput, key: "l" }});
          const outsideViewerButton = runCase({{ target: new FakeElement({{ textEntry: false, inViewer: false }}), key: "l" }});
          const toolbarInactive = runCase({{ target: editorInput, toolbarActive: false, key: "l" }});
          process.stdout.write(JSON.stringify({{ validMove, validEscape, printableBlocked, nestedDialog, viewerClosed, otherTextEntry, outsideViewerButton, toolbarInactive }}));
        }})();
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)

def eval_file_editor_delete_shortcut() -> dict:
    source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        class FakeElement {{
          constructor(opts = {{}}) {{
            this.textEntry = Boolean(opts.textEntry);
            this.editorInput = Boolean(opts.editorInput);
          }}
        }}
        function runCase(overrides = {{}}) {{
          const state = {{
            sessionId: overrides.sessionId === false ? "" : "sid-1",
            editMode: overrides.editMode !== false,
            editable: overrides.editable !== false,
            kind: overrides.kind || "text",
            version: overrides.version || "v1",
            draft: Boolean(overrides.draft),
            editorKind: overrides.editorKind || "file",
            dirty: Boolean(overrides.dirty),
            pending: Boolean(overrides.pending),
            viewMode: overrides.fileViewMode || "file",
            unavailable: Boolean(overrides.unavailable),
            selectMode: overrides.selectMode !== false,
          }};
          const calls = {{ triggers: [], toasts: [], focusCount: 0 }};
          const editorNode = {{ contains: (node) => Boolean(node && node.editorInput) }};
          const editor = {{
            getDomNode: () => editorNode,
            getPosition: () => ({{ lineNumber: 1, column: 1 }}),
            getModel: () => ({{ getLineCount: () => 20, getLineMaxColumn: () => 80 }}),
            trigger: (source, command, payload) => calls.triggers.push({{ source, command, payload }}),
          }};
          const fileStatus = {{ textContent: "", replaceChildren() {{}} }};
          const fileEditButton = {{ classList: {{ toggle() {{}} }}, setAttribute() {{}} }};
          const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
            el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children: Array.isArray(children) ? children : [] }}),
            fileStatus,
            fileEditButton,
            iconSvg: (name) => `icon:${{name}}`,
            currentSessionId: () => state.sessionId,
            currentFileSessionId: () => state.sessionId,
            normalizeLineNumber: (value) => value == null || value === "" ? null : Number(value),
            normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
            fileApiPathForPath: (_path, existing) => existing || "",
            isFileViewerOpen: () => overrides.viewerOpen !== false,
            invalidateFileViewerSessionSync: () => {{}},
            hideFileUnsavedDialog: () => {{}},
            resetFileSearchState: () => {{}},
            closeFilePickerMenu: () => {{}},
            isTextFileKind: (kind) => kind === "text" || kind === "markdown",
            isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
            confirmReload: () => true,
            promptUnsavedFileChoice: async () => "cancel",
            restoreFileEditorText: () => {{}},
            hideFileViewer: () => {{}},
            applyFileLoadResult: async () => true,
            setFilePath: () => {{}},
            resetFileViewerPanel: () => {{}},
            normalizeDraftFilePath: (value) => String(value || "").trim(),
            inspectSessionFilePath: async () => ({{ exists: false }}),
            api: async () => ({{}}),
            focusEditor: () => editor,
            disposeOpenRender: () => {{}},
            initialFileViewMode: state.viewMode,
            initialFileNonDiffMode: state.viewMode === "preview" ? "preview" : "file",
            persistFileViewMode: (mode) => {{ state.viewMode = mode; state.calls && state.calls.push(["persistFileViewMode", mode]); }},
            persistFileNonDiffMode: (mode) => {{ state.calls && state.calls.push(["persistFileNonDiffMode", mode]); }},
            currentFileEditorKind: () => state.editorKind,
            currentFileEditMode: () => state.editMode,
            activeFileEntry: () => null,
            fileCandidateGitStateFresh: () => false,
            isMarkdownPreviewable: () => false,
            resetActiveFileBufferState: () => {{}},
            updateFileTouchToolbar: () => {{}},
            isFileTouchToolbarActive: () => true,
            fileEditorShortcutBlocked: (target) => Boolean(overrides.nestedDialog || overrides.viewerOpen === false || (target && target.textEntry && !target.editorInput)),
            eventTargetElement: (value) => value instanceof FakeElement ? value : null,
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
            fileEditorDeleteCommandForKey: (key) => key === "backspace" ? "deleteLeft" : key === "delete" ? "deleteRight" : "",
            isActiveFileEditorInput: (target) => Boolean(target && target.editorInput),
            getActiveFileSelectionText: () => "",
            copyToClipboard: async () => {{}},
            focusActiveFileCodeEditor: () => {{ calls.focusCount += 1; return editor; }},
            nowMs: () => 123456,
            setToast: (message) => calls.toasts.push(message),
            setFileViewMode: () => {{}},
            applyActiveFileTextState: () => {{}},
            renderMonacoFile: async () => true,
            setFileEditMode: () => {{}},
            currentActiveFileKind: () => state.kind,
            currentActiveFileDraft: () => state.draft,
            currentActiveFileVersion: () => state.version,
            currentActiveFileEditable: () => state.editable,
            currentFileDirty: () => state.dirty,
            currentActiveFileText: () => "",
            getFileEditorText: () => "",
            setFileDirty: () => {{}},
            fmtBytes: (value) => `${{value}}B`,
            applyFileMode: () => {{}},
            rememberOpenedFile: () => {{}},
            historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
            rememberActiveFileSelection: () => {{}},
            renderFilePickerMenu: () => {{}},
          }});
          controller.setActiveFileIdentity(overrides.path === false ? "" : "note.txt", {{ gitPath: Boolean(overrides.gitPath), apiPath: overrides.apiPath || "" }});
          controller.applyActiveFileTextState({{ kind: state.kind, text: "", editable: state.editable, version: state.version, draft: state.draft }});
          if (state.editMode) controller.setFileEditMode(true);
          if (state.selectMode) controller.toggleFileTouchSelectionMode();
          calls.focusCount = 0;
          if (state.unavailable) controller.disableFileViewerForUnavailableSession("sid-1");
          const event = {{
            key: overrides.key || "Backspace",
            ctrlKey: Boolean(overrides.ctrl),
            metaKey: Boolean(overrides.meta),
            altKey: Boolean(overrides.alt),
            isComposing: Boolean(overrides.composing),
            defaultPrevented: Boolean(overrides.defaultPrevented),
            target: overrides.target || new FakeElement({{ textEntry: true, editorInput: true }}),
            prevented: 0,
            stopped: 0,
            preventDefault() {{ this.prevented += 1; }},
            stopPropagation() {{ this.stopped += 1; }},
          }};
          const handled = controller.handleFileEditorDeleteKeydown(event);
          const nativeEvent = {{
            inputType: "deleteContentBackward",
            target: event.target,
            cancelable: true,
            prevented: 0,
            stopped: 0,
            preventDefault() {{ this.prevented += 1; }},
            stopPropagation() {{ this.stopped += 1; }},
          }};
          const nativeResult = controller.suppressFileEditorNativeDelete(nativeEvent);
          const nativeSecond = {{
            inputType: "deleteContentBackward",
            target: event.target,
            cancelable: true,
            prevented: 0,
            stopped: 0,
            preventDefault() {{ this.prevented += 1; }},
            stopPropagation() {{ this.stopped += 1; }},
          }};
          const nativeSecondResult = controller.suppressFileEditorNativeDelete(nativeSecond);
          return {{ handled, prevented: event.prevented, stopped: event.stopped, triggers: calls.triggers, focusCount: calls.focusCount, mode: controller.currentFileTouchSelectMode(), native: {{ result: nativeResult, prevented: nativeEvent.prevented, stopped: nativeEvent.stopped, secondResult: nativeSecondResult, secondPrevented: nativeSecond.prevented, secondStopped: nativeSecond.stopped }}, toasts: calls.toasts }};
        }}
        (() => {{
          const editorInput = new FakeElement({{ textEntry: true, editorInput: true }});
          const otherInput = new FakeElement({{ textEntry: true, editorInput: false }});
          const validBackspace = runCase({{ target: editorInput }});
          const validDelete = runCase({{ target: editorInput, key: "Delete" }});
          const nestedDialog = runCase({{ target: editorInput, nestedDialog: true }});
          const viewerClosed = runCase({{ target: editorInput, viewerOpen: false }});
          const otherTextEntry = runCase({{ target: otherInput }});
          const notEdit = runCase({{ target: editorInput, editMode: false }});
          const unavailable = runCase({{ target: editorInput, unavailable: true }});
          process.stdout.write(JSON.stringify({{ validBackspace, validDelete, nestedDialog, viewerClosed, otherTextEntry, notEdit, unavailable }}));
        }})();
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)

def eval_file_open_request_sequence() -> dict:
    viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        class AbortController {{
          constructor() {{
            this.signal = {{ aborted: false }};
          }}
          abort() {{
            this.signal.aborted = true;
          }}
        }}
        let disposeCalls = 0;
        const state = {{ sessionId: "sid-1" }};
        const ctx = {{ window: {{}}, AbortController }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(viewer_source)}, ctx);
        const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
          el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children }}),
          fileStatus: {{ replaceChildren() {{}} }},
          fileEditButton: {{ classList: {{ toggle() {{}} }}, setAttribute() {{}} }},
          iconSvg: (name) => name,
          currentSessionId: () => state.sessionId,
          currentFileSessionId: () => state.sessionId,
          normalizeLineNumber: (value) => value == null ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          isFileViewerOpen: () => state.viewerOpen !== false,
          invalidateFileViewerSessionSync: () => calls.push(["invalidateFileViewerSessionSync"]),
          hideFileUnsavedDialog: (choice) => calls.push(["hideFileUnsavedDialog", choice]),
          resetFileSearchState: () => calls.push(["resetFileSearchState"]),
          closeFilePickerMenu: (options) => calls.push(["closeFilePickerMenu", options]),
          isUnavailable: () => false,
          isTextFileKind: (kind) => kind === "text" || kind === "markdown",
          isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
          confirmReload: () => true,
          promptUnsavedFileChoice: async () => "cancel",
          restoreFileEditorText: (text) => calls.push(["restoreFileEditorText", text]),
          hideFileViewer: () => calls.push(["hideFileViewer"]),
          applyFileLoadResult: async () => true,
          setFilePath: (...args) => calls.push(["setFilePath", ...args]),
          resetFileViewerPanel: () => calls.push(["resetFileViewerPanel"]),
          normalizeDraftFilePath: (value) => String(value || "").trim(),
          inspectSessionFilePath: async () => ({{ exists: false }}),
          api: async () => ({{}}),
          focusEditor: () => null,
          disposeOpenRender: () => {{ disposeCalls += 1; }},
          initialFileViewMode: "file",
          initialFileNonDiffMode: "file",
          persistFileViewMode: (mode) => calls.push(["persistFileViewMode", mode]),
          persistFileNonDiffMode: (mode) => calls.push(["persistFileNonDiffMode", mode]),
          currentFileEditorKind: () => "file",
          currentFileEditMode: () => true,
          activeFileEntry: () => null,
          fileCandidateGitStateFresh: () => false,
          isMarkdownPreviewable: () => true,
          resetActiveFileBufferState: () => {{}},
          updateFileTouchToolbar: () => {{}},
          currentFileTouchSelectMode: () => false,
          isFileTouchToolbarActive: () => false,
          fileEditorShortcutBlocked: () => false,
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
          resetFileTouchSelectionState: (options) => calls.push(["resetFileTouchSelectionState", options || {{}}]),
          moveFileTouchSelection: (direction) => calls.push(["moveFileTouchSelection", direction]),
          fileEditorDeleteCommandForKey: () => "",
          isActiveFileEditorInput: () => false,
          getActiveFileSelectionText: () => "",
          copyToClipboard: async () => {{}},
          focusActiveFileCodeEditor: () => null,
          nowMs: () => 0,
          setToast: (message) => calls.push(["toast", message]),
          setFileViewMode: () => {{}},
          applyActiveFileTextState: () => {{}},
          renderMonacoFile: async () => true,
          setFileEditMode: () => {{}},
          currentActiveFileKind: () => "text",
          currentActiveFileDraft: () => false,
          currentActiveFileVersion: () => "",
          currentActiveFileEditable: () => true,
          currentFileDirty: () => true,
          currentActiveFileText: () => "",
          getFileEditorText: () => "",
          setFileDirty: () => {{}},
          syncFileEditorReadOnly: () => {{}},
          fmtBytes: (value) => `${{value}}B`,
          applyFileMode: () => {{}},
          rememberOpenedFile: () => {{}},
          historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
          rememberActiveFileSelection: () => {{}},
          updateFileEditButton: () => {{}},
          renderFilePickerMenu: () => {{}},
        }});
        controller.setActiveFileIdentity("old.txt", {{ line: 1, gitPath: false, apiPath: "" }});
        const first = controller.beginFileOpenRequest("first.txt", {{ line: 3 }});
        const firstCurrent = controller.isCurrentFileOpenRequest(first);
        const second = controller.beginFileOpenRequest(" trail.md ", {{ line: 8, gitPath: true }});
        const result = {{
          currentSessionId: state.sessionId,
          firstCurrent,
          firstSignalAborted: Boolean(first.signal && first.signal.aborted),
          firstAfterSecond: controller.isCurrentFileOpenRequest(first),
          secondCurrent: controller.isCurrentFileOpenRequest(second),
          secondGitPath: second.gitPath,
          secondApiPath: second.apiPath,
          activeIdentity: controller.currentActiveFileIdentity(),
          activeLine: controller.currentActiveFileLine(),
        }};
        controller.setActiveFileIdentity("same.py", {{ line: 8, gitPath: true, apiPath: "tok-same" }});
        const same = controller.beginFileOpenRequest(null, {{}});
        const explicit = controller.beginFileOpenRequest("explicit.py", {{ gitPath: true, apiPath: "explicit-token" }});
        const nongit = controller.beginFileOpenRequest("plain.py", {{ gitPath: false }});
        result.sameApiPath = same.apiPath;
        result.sameGitPath = same.gitPath;
        result.explicitApiPath = explicit.apiPath;
        result.nongitApiPath = nongit.apiPath;
        result.nongitGitPath = nongit.gitPath;
        result.helperRejectsMissingCurrent = false;
        try {{
          controller.nextActiveFileIdentity(null, "x.py");
        }} catch (_) {{
          result.helperRejectsMissingCurrent = true;
        }}
        controller.setActiveFileIdentity("clear.py", {{ line: 99, gitPath: true, apiPath: "tok-clear" }});
        controller.clearActiveFileIdentity({{ line: "12" }});
        result.clearWithLine = {{ ...controller.currentActiveFileIdentity(), line: controller.currentActiveFileLine() }};
        controller.setActiveFileIdentity("clear-again.py", {{ line: 12, gitPath: true, apiPath: "tok-again" }});
        controller.clearActiveFileIdentity();
        result.clearDefault = {{ ...controller.currentActiveFileIdentity(), line: controller.currentActiveFileLine() }};
        const handle = controller.startFileOpenRequest("handled.txt", {{ line: 4, gitPath: false }});
        result.handlePath = handle.path;
        result.handleCurrentBeforeDone = controller.isCurrentFileOpenRequest(handle.request);
        handle.done();
        result.handleSignalAbortedAfterDone = Boolean(handle.request.signal && handle.request.signal.aborted);
        const afterHandle = controller.startFileOpenRequest("after-handle.txt", {{ gitPath: false }});
        result.handleSignalAbortedAfterNext = Boolean(handle.request.signal && handle.request.signal.aborted);
        result.afterHandleCurrent = controller.isCurrentFileOpenRequest(afterHandle.request);
        afterHandle.done();
        controller.cancelPendingFileOpen();
        result.secondAfterCancel = controller.isCurrentFileOpenRequest(second);
        result.disposeCalls = disposeCalls;
        process.stdout.write(JSON.stringify(result));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_file_viewer_session_sync_race() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function currentFileViewerSessionId() {")
    end = source.index("function extToEditorLang", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        let resolveUnsaved;
        const calls = [];
        const ctx = {{
          selected: "sid-b",
          fileViewerSessionId: "sid-a",
{controller_identity_ctx_js("old.txt", "", False, 1)}
          fileSearchSessionId: "sid-a",
          fileCandidateList: ["candidate.txt"],
          fileViewerSessionSyncToken: 0,
          fileStatus: {{ textContent: "" }},
          fileOpenAbortController: null,
          isFileViewerOpen: () => true,
          maybeHandleUnsavedFileChanges: () => new Promise((resolve) => {{ resolveUnsaved = resolve; }}),
          disposePdfRender: () => calls.push("disposePdfRender"),
          resetFileSearchState: () => calls.push("resetFileSearchState"),
          refreshFileCandidates: async () => calls.push("refreshFileCandidates"),
          preferredFileSelectionForSession: () => ({{ path: "preferred.txt", line: 9 }}),
          setFilePath: (...args) => calls.push(["setFilePath", ...args]),
          openFilePathWithResolvedMode: async (...args) => calls.push(["openFilePathWithResolvedMode", ...args]),
          resetFileViewerPanel: () => calls.push("resetFileViewerPanel"),
          resetFilePickerInput: () => calls.push("resetFilePickerInput"),
          renderFilePickerMenu: () => calls.push("renderFilePickerMenu"),
          updateFileTouchToolbar: () => calls.push("updateFileTouchToolbar"),
          currentFileTouchSelectMode: () => false,
          isFileTouchToolbarActive: () => false,
          fileEditorShortcutBlocked: () => false,
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
          resetFileTouchSelectionState: (options) => calls.push(["resetFileTouchSelectionState", options || {{}}]),
          moveFileTouchSelection: (direction) => calls.push(["moveFileTouchSelection", direction]),
          fileEditorDeleteCommandForKey: () => "",
          isActiveFileEditorInput: () => false,
          getActiveFileSelectionText: () => "",
          copyToClipboard: async () => {{}},
          focusActiveFileCodeEditor: () => null,
          nowMs: () => 0,
          setToast: (message) => calls.push(["toast", message]),
          normalizeLineNumber: (value) => value == null ? null : Number(value),
          AbortController: class {{
            constructor() {{ this.signal = {{ aborted: false }}; }}
            abort() {{ this.signal.aborted = true; }}
          }},
          console,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test = { ensureCurrentFileViewerSession };\n")}, ctx);
        const promise = ctx.__test.ensureCurrentFileViewerSession();
        ctx.selected = "sid-c";
        resolveUnsaved(true);
        promise.then((result) => {{
          process.stdout.write(JSON.stringify({{
            result,
            selected: ctx.selected,
            fileViewerSessionId: ctx.fileViewerSessionId,
            calls,
            status: ctx.fileStatus.textContent,
          }}));
        }}).catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_resolved_open_current_guard() -> dict:
    viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        let resolveInspect;
        let current = true;
        const calls = [];
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(viewer_source)}, ctx);
        const fileStatus = {{ textContent: "", replaceChildren() {{ this.textContent = ""; }} }};
        const fileEditButton = {{ classList: {{ toggle() {{}} }}, setAttribute() {{}}, disabled: false }};
        const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
          el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children }}),
          fileStatus,
          fileEditButton,
          iconSvg: (name) => `icon:${{name}}`,
          currentSessionId: () => "sid-b",
          currentFileSessionId: () => "sid-b",
          normalizeLineNumber: (value) => value == null || value === "" ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          isFileViewerOpen: () => true,
          hideFileUnsavedDialog: () => {{}},
          resetFileSearchState: () => {{}},
          closeFilePickerMenu: () => {{}},
          isTextFileKind: (kind) => kind === "text" || kind === "markdown",
          isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
          confirmReload: () => true,
          promptUnsavedFileChoice: async () => "cancel",
          restoreFileEditorText: () => {{}},
          hideFileViewer: () => {{}},
          setFilePath: (...args) => calls.push(["setFilePath", ...args]),
          resetFileViewerPanel: () => calls.push(["resetFileViewerPanel"]),
          applyFileLoadResult: async (...args) => {{ calls.push(["applyFileLoadResult", ...args]); return true; }},
          normalizeDraftFilePath: (value) => String(value || "").trim(),
          inspectSessionFilePath: async (...args) => {{ calls.push(["inspect", ...args]); return await new Promise((resolve) => {{ resolveInspect = resolve; }}); }},
          api: async (...args) => {{ calls.push(["api", ...args]); return {{ kind: "text", text: "body", path: "/abs" }}; }},
          focusEditor: () => null,
          disposeOpenRender: () => calls.push(["disposeOpenRender"]),
          persistFileViewMode: (mode) => calls.push(["persistFileViewMode", mode]),
          persistFileNonDiffMode: (mode) => calls.push(["persistFileNonDiffMode", mode]),
          isMarkdownPreviewable: () => false,
          updateFileTouchToolbar: () => calls.push(["touchToolbar"]),
          isFileTouchToolbarActive: () => false,
          fileEditorShortcutBlocked: () => false,
          eventTargetElement: (value) => value || null,
          normalizeFileEditorPosition: (_editor, position) => position || null,
          applyFileEditorSelection: () => {{}},
          isCollapsedFileSelection: () => true,
          positionAfterInsertedText: (start, text) => ({{ lineNumber: start.lineNumber, column: start.column + String(text || "").length }}),
          fileEditorEditSupportAvailable: () => false,
          syncFileDiffSelectionMode: () => {{}},
          showFilePasteDialog: () => false,
          hideFilePasteDialog: () => {{}},
          clipboardReadAvailable: () => false,
          readClipboardText: async () => "",
          fileEditorDeleteCommandForKey: () => "",
          isActiveFileEditorInput: () => false,
          getActiveFileSelectionText: () => "",
          copyToClipboard: async () => {{}},
          focusActiveFileCodeEditor: () => null,
          nowMs: () => 0,
          setToast: (message) => calls.push(["toast", message]),
          renderMonacoFile: async () => true,
          getFileEditorText: () => "",
          fmtBytes: (value) => String(value),
          applyFileMode: () => calls.push(["applyFileMode"]),
          rememberOpenedFile: (...args) => calls.push(["rememberOpenedFile", ...args]),
          historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
          renderFilePickerMenu: () => calls.push(["renderFilePickerMenu"]),
        }});
        const promise = controller.openFilePathWithResolvedMode("b-file.txt", {{ isCurrent: () => current }});
        current = false;
        resolveInspect({{ exists: true, kind: "text" }});
        promise.then((result) => {{
          process.stdout.write(JSON.stringify({{ result, calls }}));
        }}).catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)




def eval_open_file_guard_mode_validation() -> dict:
    source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const calls = [];
        const state = {{ sessionId: "sid-1", dirty: false, unavailable: false }};
        const fileStatus = {{ textContent: "", replaceChildren() {{}} }};
        const fileEditButton = {{
          disabled: false,
          innerHTML: "",
          title: "",
          attrs: {{}},
          classList: {{ toggle(name, enabled) {{ calls.push(["buttonToggle", name, Boolean(enabled)]); }} }},
          setAttribute(name, value) {{ this.attrs[name] = String(value); calls.push(["buttonAttr", name, String(value)]); }},
        }};
        const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
          el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children: Array.isArray(children) ? children : [] }}),
          fileStatus,
          fileEditButton,
          iconSvg: (name) => `icon:${{name}}`,
          currentSessionId: () => state.sessionId,
          currentFileSessionId: () => state.sessionId,
          normalizeLineNumber: (value) => value == null || value === "" ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          fileApiPathForPath: (_path, existing) => existing || "",
          isFileViewerOpen: () => true,
          invalidateFileViewerSessionSync: () => calls.push(["invalidateFileViewerSessionSync"]),
          hideFileUnsavedDialog: (choice) => calls.push(["hideFileUnsavedDialog", choice]),
          resetFileSearchState: () => calls.push(["resetFileSearchState"]),
          closeFilePickerMenu: (options) => calls.push(["closeFilePickerMenu", options]),
          isTextFileKind: (kind) => kind === "text" || kind === "markdown",
          isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
          confirmReload: () => true,
          promptUnsavedFileChoice: async () => "discard",
          restoreFileEditorText: (text) => {{ state.dirty = false; calls.push(["restoreFileEditorText", text]); }},
          hideFileViewer: () => calls.push(["hideFileViewer"]),
          applyFileLoadResult: async (...args) => {{ calls.push(["applyFileLoadResult", args[0], args[1] && args[1].kind, args[3] && args[3].viewMode]); return true; }},
          setFilePath: (...args) => calls.push(["setFilePath", ...args]),
          resetFileViewerPanel: () => calls.push(["resetFileViewerPanel"]),
          normalizeDraftFilePath: (value) => String(value || "").trim().replace(/^[/]+/, ""),
          inspectSessionFilePath: async () => ({{ exists: false }}),
          api: async () => ({{}}),
          focusEditor: () => ({{ focus() {{}}, updateOptions(opts) {{ calls.push(["editorOptions", opts]); }} }}),
          disposeOpenRender: () => calls.push(["disposeOpenRender"]),
          initialFileViewMode: "file",
          initialFileNonDiffMode: "file",
          persistFileViewMode: (mode) => calls.push(["persistFileViewMode", mode]),
          persistFileNonDiffMode: (mode) => calls.push(["persistFileNonDiffMode", mode]),
          currentFileEditorKind: () => "file",
          currentFileEditMode: () => false,
          activeFileEntry: () => null,
          fileCandidateGitStateFresh: () => false,
          isMarkdownPreviewable: () => false,
          resetActiveFileBufferState: () => calls.push(["resetActiveFileBufferState"]),
          updateFileTouchToolbar: () => calls.push(["touchToolbar"]),
          currentFileTouchSelectMode: () => false,
          isFileTouchToolbarActive: () => false,
          fileEditorShortcutBlocked: () => false,
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
          resetFileTouchSelectionState: (options) => calls.push(["resetFileTouchSelectionState", options || {{}}]),
          moveFileTouchSelection: (direction) => calls.push(["moveFileTouchSelection", direction]),
          fileEditorDeleteCommandForKey: () => "",
          isActiveFileEditorInput: () => false,
          getActiveFileSelectionText: () => "",
          copyToClipboard: async () => {{}},
          focusActiveFileCodeEditor: () => null,
          nowMs: () => 0,
          setToast: (message) => calls.push(["toast", message]),
          setFileViewMode: (...args) => calls.push(["setFileViewMode", ...args]),
          applyActiveFileTextState: (next) => calls.push(["applyActiveFileTextState", next]),
          renderMonacoFile: async () => true,
          setFileEditMode: (...args) => calls.push(["setFileEditMode", ...args]),
          currentActiveFileKind: () => "text",
          currentActiveFileDraft: () => false,
          currentActiveFileVersion: () => "",
          currentActiveFileEditable: () => true,
          currentFileDirty: () => state.dirty,
          currentActiveFileText: () => "",
          getFileEditorText: () => "",
          setFileDirty: (...args) => calls.push(["setFileDirty", ...args]),
          fmtBytes: (value) => `${{value}}B`,
          applyFileMode: () => calls.push(["applyFileMode"]),
          rememberOpenedFile: (...args) => calls.push(["rememberOpenedFile", ...args]),
          historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
          rememberActiveFileSelection: () => calls.push(["rememberActiveFileSelection"]),
          renderFilePickerMenu: () => calls.push(["renderFilePickerMenu"]),
        }});
        (async () => {{
          let invalidMessage = "";
          try {{ await controller.openFilePathWithGuard("x.txt", {{ mode: "bogus" }}); }} catch (err) {{ invalidMessage = err && err.message || ""; }}
          const invalidCalls = calls.slice();
          calls.length = 0;
          const validResult = await controller.openFilePathWithGuard("x.txt", {{ line: 4, mode: "diff", gitPath: true, apiPath: "tok" }});
          const validCalls = calls.slice();
          calls.length = 0;
          const staleResult = await controller.openFilePathWithGuard("x.txt", {{ mode: "file", isCurrent: () => false }});
          const staleCalls = calls.slice();
          process.stdout.write(JSON.stringify({{ invalidMessage, invalidCalls, validResult, validCalls, staleResult, staleCalls }}));
        }})().catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)

def eval_open_file_path_mode_ownership() -> dict:
    source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const calls = [];
        const state = {{
          sessionId: "sid-1",
          viewMode: "file",
          gitFresh: false,
          activeEntryValue: null,
          activeEntryCalls: 0,
          markdownPreviewable: false,
          dirty: false,
        }};
        const fileStatus = {{ textContent: "", replaceChildren() {{}} }};
        const fileEditButton = {{
          disabled: false,
          innerHTML: "",
          title: "",
          attrs: {{}},
          classList: {{ toggle(name, enabled) {{ calls.push(["buttonToggle", name, Boolean(enabled)]); }} }},
          setAttribute(name, value) {{ this.attrs[name] = String(value); calls.push(["buttonAttr", name, String(value)]); }},
        }};
        const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
          el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children: Array.isArray(children) ? children : [] }}),
          fileStatus,
          fileEditButton,
          iconSvg: (name) => `icon:${{name}}`,
          currentSessionId: () => state.sessionId,
          currentFileSessionId: () => state.sessionId,
          normalizeLineNumber: (value) => value == null || value === "" ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          fileApiPathForPath: (_path, existing) => existing || "",
          isFileViewerOpen: () => true,
          invalidateFileViewerSessionSync: () => calls.push(["invalidateFileViewerSessionSync"]),
          hideFileUnsavedDialog: (choice) => calls.push(["hideFileUnsavedDialog", choice]),
          resetFileSearchState: () => calls.push(["resetFileSearchState"]),
          closeFilePickerMenu: (options) => calls.push(["closeFilePickerMenu", options]),
          isTextFileKind: (kind) => kind === "text" || kind === "markdown",
          isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
          confirmReload: () => true,
          promptUnsavedFileChoice: async () => "discard",
          restoreFileEditorText: (text) => {{ state.dirty = false; calls.push(["restoreFileEditorText", text]); }},
          hideFileViewer: () => calls.push(["hideFileViewer"]),
          applyFileLoadResult: async (...args) => {{ calls.push(["applyFileLoadResult", args[0], args[1] && args[1].kind, args[3] && args[3].viewMode]); return true; }},
          setFilePath: (...args) => calls.push(["setFilePath", ...args]),
          resetFileViewerPanel: () => calls.push(["resetFileViewerPanel"]),
          normalizeDraftFilePath: (value) => String(value || "").trim(),
          inspectSessionFilePath: async () => ({{ exists: false }}),
          api: async (url, options = {{}}) => {{
            calls.push(["api", url, Boolean(options.signal)]);
            if (url.includes("/git/file_versions")) return {{ base_text: "old", current_text: "new", base_exists: true, current_exists: true, abs_path: "/abs/diff" }};
            return {{ kind: "text", text: "body", path: "/abs/read" }};
          }},
          focusEditor: () => ({{ focus() {{}}, updateOptions(opts) {{ calls.push(["editorOptions", opts]); }} }}),
          disposeOpenRender: () => calls.push(["disposeOpenRender"]),
          initialFileViewMode: state.viewMode,
          initialFileNonDiffMode: state.viewMode === "preview" ? "preview" : "file",
          persistFileViewMode: (mode) => {{ state.viewMode = mode; calls.push(["persistFileViewMode", mode]); }},
          persistFileNonDiffMode: (mode) => calls.push(["persistFileNonDiffMode", mode]),
          currentFileEditorKind: () => "file",
          currentFileEditMode: () => false,
          activeFileEntry: () => {{ state.activeEntryCalls += 1; return state.activeEntryValue; }},
          fileCandidateGitStateFresh: () => state.gitFresh,
          isMarkdownPreviewable: (rel) => {{ calls.push(["isMarkdownPreviewable", rel]); return state.markdownPreviewable; }},
          resetActiveFileBufferState: () => calls.push(["resetActiveFileBufferState"]),
          updateFileTouchToolbar: () => calls.push(["touchToolbar"]),
          currentFileTouchSelectMode: () => false,
          isFileTouchToolbarActive: () => false,
          fileEditorShortcutBlocked: () => false,
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
          resetFileTouchSelectionState: (options) => calls.push(["resetFileTouchSelectionState", options || {{}}]),
          moveFileTouchSelection: (direction) => calls.push(["moveFileTouchSelection", direction]),
          fileEditorDeleteCommandForKey: () => "",
          isActiveFileEditorInput: () => false,
          getActiveFileSelectionText: () => "",
          copyToClipboard: async () => {{}},
          focusActiveFileCodeEditor: () => null,
          nowMs: () => 0,
          setToast: (message) => calls.push(["toast", message]),
          setFileViewMode: (mode) => {{ state.viewMode = mode; calls.push(["setFileViewMode", mode]); }},
          applyActiveFileTextState: (next) => calls.push(["applyActiveFileTextState", next]),
          renderMonacoFile: async () => true,
          setFileEditMode: (...args) => calls.push(["setFileEditMode", ...args]),
          currentActiveFileKind: () => "text",
          currentActiveFileDraft: () => false,
          currentActiveFileVersion: () => "",
          currentActiveFileEditable: () => true,
          currentFileDirty: () => state.dirty,
          currentActiveFileText: () => "",
          getFileEditorText: () => "",
          setFileDirty: (...args) => calls.push(["setFileDirty", ...args]),
          fmtBytes: (value) => `${{value}}B`,
          applyFileMode: () => calls.push(["applyFileMode"]),
          rememberOpenedFile: (...args) => calls.push(["rememberOpenedFile", ...args]),
          historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
          rememberActiveFileSelection: () => calls.push(["rememberActiveFileSelection"]),
          renderFilePickerMenu: () => calls.push(["renderFilePickerMenu"]),
        }});
        (async () => {{
          const explicitResult = await controller.openFilePath("stale.txt", {{ line: 5, gitPath: true, apiPath: "tok", mode: "diff" }});
          const explicit = {{ result: explicitResult, calls: calls.slice(), activeEntryCalls: state.activeEntryCalls, fileViewMode: state.viewMode }};
          calls.length = 0;
          fileStatus.textContent = "";
          state.viewMode = "diff";
          state.gitFresh = true;
          state.activeEntryValue = {{ changed: false }};
          state.activeEntryCalls = 0;
          const fallbackResult = await controller.openFilePath("stale.txt", {{ line: 5, gitPath: true, apiPath: "tok" }});
          const fallback = {{ result: fallbackResult, calls: calls.slice(), activeEntryCalls: state.activeEntryCalls, fileViewMode: state.viewMode }};
          let invalidMessage = "";
          try {{ controller.resolveFileOpenViewMode({{ gitPath: false }}, "x.txt", "bogus"); }} catch (err) {{ invalidMessage = err && err.message || ""; }}
          process.stdout.write(JSON.stringify({{ explicit, fallback, invalidMessage }}));
        }})().catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)

def eval_active_file_load_state_writers() -> dict:
    source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    start = source.index("let activeFileKind = \"\"")
    end = source.index("function clearActiveFileIdentity", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          clearActiveFileSaveState() {{}},
          resetFileTouchSelectionState() {{}},
          syncFileEditorReadOnly() {{}},
          updateFileEditButton() {{}},
          setFileDirty() {{}},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_load_state = { applyActiveFileTextState, applyActiveFileDiffState, applyActiveFileNonTextState, resetActiveFileBufferState, currentActiveFileKind, currentActiveFileText, currentActiveFileEditable, currentActiveFileVersion, currentActiveFileDraft };\n")}, ctx);
        function stale() {{
          ctx.__test_load_state.applyActiveFileTextState({{ kind: "markdown", text: "stale text", editable: true, version: "stale-version", draft: true }});
        }}
        function state() {{
          return {{
            kind: ctx.__test_load_state.currentActiveFileKind(),
            text: ctx.__test_load_state.currentActiveFileText(),
            editable: ctx.__test_load_state.currentActiveFileEditable(),
            version: ctx.__test_load_state.currentActiveFileVersion(),
            draft: ctx.__test_load_state.currentActiveFileDraft(),
          }};
        }}
        const result = {{}};
        stale();
        ctx.__test_load_state.applyActiveFileTextState({{ kind: "markdown", text: "# hi", editable: false, version: "v2", draft: false }});
        result.markdown = state();
        stale();
        ctx.__test_load_state.applyActiveFileTextState({{ text: "", editable: true, version: "", draft: true }});
        result.draft = state();
        stale();
        ctx.__test_load_state.applyActiveFileDiffState({{ currentText: "current", currentExists: true }});
        result.diff = state();
        stale();
        ctx.__test_load_state.applyActiveFileNonTextState("image");
        result.image = state();
        ctx.__test_load_state.resetActiveFileBufferState();
        result.reset = state();
        result.invalidTextThrows = false;
        try {{ ctx.__test_load_state.applyActiveFileTextState({{ kind: "image" }}); }} catch (err) {{ result.invalidTextThrows = err && err.message === "invalid active file text kind"; }}
        result.invalidNonTextThrows = false;
        try {{ ctx.__test_load_state.applyActiveFileNonTextState("text"); }} catch (err) {{ result.invalidNonTextThrows = err && err.message === "invalid active file non-text kind"; }}
        process.stdout.write(JSON.stringify(result));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_active_file_save_request_helpers() -> dict:
    source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const state = {{
          sessionId: "sid-1",
          draft: true,
          version: "v1",
          text: "body text",
          unavailable: false,
        }};
        const calls = [];
        const fileStatus = {{ textContent: "", replaceChildren() {{}} }};
        const fileEditButton = {{
          disabled: false,
          innerHTML: "",
          title: "",
          attrs: {{}},
          classes: {{}},
          classList: {{ toggle(name, enabled) {{ fileEditButton.classes[name] = Boolean(enabled); calls.push(["buttonToggle", name, Boolean(enabled)]); }} }},
          setAttribute(name, value) {{ this.attrs[name] = String(value); calls.push(["buttonAttr", name, String(value)]); }},
        }};
        const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
          el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children }}),
          fileStatus,
          fileEditButton,
          iconSvg: (name) => `icon:${{name}}`,
          currentSessionId: () => state.sessionId,
          currentFileSessionId: () => state.sessionId,
          normalizeLineNumber: (value) => value == null ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          fileApiPathForPath: (_path, existing = "") => existing || "derived-token",
          isFileViewerOpen: () => state.viewerOpen !== false,
          invalidateFileViewerSessionSync: () => calls.push(["invalidateFileViewerSessionSync"]),
          hideFileUnsavedDialog: (choice) => calls.push(["hideFileUnsavedDialog", choice]),
          resetFileSearchState: () => calls.push(["resetFileSearchState"]),
          closeFilePickerMenu: (options) => calls.push(["closeFilePickerMenu", options]),
          isUnavailable: () => state.unavailable,
          isTextFileKind: (kind) => kind === "text" || kind === "markdown",
          isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
          confirmReload: () => true,
          promptUnsavedFileChoice: async () => "cancel",
          restoreFileEditorText: (text) => calls.push(["restoreFileEditorText", text]),
          hideFileViewer: () => calls.push(["hideFileViewer"]),
          applyFileLoadResult: async () => true,
          setFilePath: (...args) => calls.push(["setFilePath", ...args]),
          resetFileViewerPanel: () => calls.push(["resetFileViewerPanel"]),
          normalizeDraftFilePath: (value) => String(value || "").trim(),
          inspectSessionFilePath: async () => ({{ exists: false }}),
          api: async () => ({{ kind: "text", text: "body", path: "/abs/read" }}),
          focusEditor: () => ({{ updateOptions: (opts) => calls.push(["updateOptions", opts]) }}),
          disposeOpenRender: () => calls.push(["disposeOpenRender"]),
          initialFileViewMode: "file",
          initialFileNonDiffMode: "file",
          persistFileViewMode: (mode) => calls.push(["persistFileViewMode", mode]),
          persistFileNonDiffMode: (mode) => calls.push(["persistFileNonDiffMode", mode]),
          currentFileEditorKind: () => "file",
          currentFileEditMode: () => true,
          activeFileEntry: () => null,
          fileCandidateGitStateFresh: () => false,
          isMarkdownPreviewable: () => true,
          resetActiveFileBufferState: () => calls.push(["resetActiveFileBufferState"]),
          updateFileTouchToolbar: () => calls.push(["updateFileTouchToolbar"]),
          currentFileTouchSelectMode: () => false,
          isFileTouchToolbarActive: () => false,
          fileEditorShortcutBlocked: () => false,
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
          resetFileTouchSelectionState: (options) => calls.push(["resetFileTouchSelectionState", options || {{}}]),
          moveFileTouchSelection: (direction) => calls.push(["moveFileTouchSelection", direction]),
          fileEditorDeleteCommandForKey: () => "",
          isActiveFileEditorInput: () => false,
          getActiveFileSelectionText: () => "",
          copyToClipboard: async () => {{}},
          focusActiveFileCodeEditor: () => null,
          nowMs: () => 0,
          setToast: (message) => calls.push(["toast", message]),
          setFileViewMode: () => {{}},
          applyActiveFileTextState: () => {{}},
          renderMonacoFile: async () => true,
          setFileEditMode: () => {{}},
          currentActiveFileKind: () => "text",
          currentActiveFileDraft: () => state.draft,
          currentActiveFileVersion: () => state.version,
          currentActiveFileEditable: () => true,
          currentFileDirty: () => true,
          currentActiveFileText: () => "",
          getFileEditorText: () => {{ calls.push(["getFileEditorText"]); return state.text; }},
          setFileDirty: () => calls.push(["setFileDirty"]),
          fmtBytes: (value) => `${{value}}B`,
          applyFileMode: () => {{}},
          rememberOpenedFile: () => {{}},
          historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
          rememberActiveFileSelection: () => {{}},
          renderFilePickerMenu: () => {{}},
        }});
        controller.setActiveFileIdentity("src/app.py", {{ line: 42, gitPath: true, apiPath: "token-1" }});
        controller.setFileEditorKind("file");
        controller.applyActiveFileTextState({{ kind: "text", text: "old text", editable: true, version: state.version, draft: state.draft }});
        controller.setFileEditMode(true);
        controller.setFileDirty(true);
        calls.length = 0;
        const save = controller.beginActiveFileSaveRequest();
        const result = {{
          save,
          frozen: Object.isFrozen(save),
          pendingAfterBegin: controller.isFileSavePending(),
          callsAfterBegin: calls.slice(),
          currentInitial: controller.isCurrentActiveFileSaveRequest(save),
        }};
        controller.markActiveFileSavePending(save);
        result.pendingAfterMark = controller.isFileSavePending();
        result.statusAfterMark = fileStatus.textContent;
        result.callsAfterMark = calls.slice();
        controller.setActiveFileIdentity("src/app.py", {{ line: 42, gitPath: true, apiPath: "other-token" }});
        result.currentWrongApiPath = controller.isCurrentActiveFileSaveRequest(save);
        controller.setActiveFileIdentity("src/app.py", {{ line: 42, gitPath: false, apiPath: "token-1" }});
        result.currentWrongGitPath = controller.isCurrentActiveFileSaveRequest(save);
        controller.setActiveFileIdentity("src/app.py", {{ line: 42, gitPath: true, apiPath: "token-1" }});
        controller.disableFileViewerForUnavailableSession("sid-1");
        result.currentUnavailable = controller.isCurrentActiveFileSaveRequest(save);
        controller.clearFileViewerUnavailableSession();
        state.editMode = true;
        controller.setFileEditMode(true);
        const save2 = controller.beginActiveFileSaveRequest();
        controller.markActiveFileSavePending(save2);
        controller.finishActiveFileSaveRequest(save);
        result.afterMismatchedFinish = {{ pending: controller.isFileSavePending(), calls: calls.slice() }};
        controller.finishActiveFileSaveRequest(save2);
        result.afterMatchedFinish = {{ pending: controller.isFileSavePending(), calls: calls.slice() }};
        process.stdout.write(JSON.stringify(result));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_active_file_save_body_builder() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function buildActiveFileSaveBody(")
    end = source.index("function applyActiveFileSaveSuccess", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          fileViewerController: {{
            buildActiveFileSaveBody: (save) => {{
              const body = save.draft
                ? {{ path: save.path, text: save.text, create: true }}
                : {{ path: save.path, text: save.text, version: save.version, git_path: save.gitPath }};
              if (!save.draft && save.gitPath && save.apiPath) body.path_token = save.apiPath;
              return body;
            }},
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_save_body = buildActiveFileSaveBody;\n")}, ctx);
        const draft = ctx.__test_save_body({{ path: "new.py", text: "NEW", draft: true, gitPath: true, version: "v1", apiPath: "tok" }});
        const gitToken = ctx.__test_save_body({{ path: "existing.py", text: "BODY", draft: false, gitPath: true, version: "v2", apiPath: "tok" }});
        const gitNoToken = ctx.__test_save_body({{ path: "existing.py", text: "BODY", draft: false, gitPath: true, version: "v2", apiPath: "" }});
        const plainToken = ctx.__test_save_body({{ path: "plain.py", text: "TEXT", draft: false, gitPath: false, version: "v3", apiPath: "tok" }});
        process.stdout.write(JSON.stringify({{ draft, gitToken, gitNoToken, plainToken }}));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)



def eval_active_file_save_error_renderer() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function renderActiveFileSaveError(")
    end = source.index("function applyActiveFileSaveSuccess", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          fileStatus: {{ textContent: "" }},
          calls: [],
          fileViewerController: {{
            renderActiveFileSaveError: (save, error) => {{
              if (error && error.status === 409) {{
                ctx.calls.push(["renderSaveConflict", save.sessionId, save.path, error && error.message ? error.message : "conflict"]);
              }} else {{
                ctx.fileStatus.textContent = `save error: ${{error && error.message ? error.message : "unknown error"}}`;
              }}
            }},
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_save_error = renderActiveFileSaveError;\n")}, ctx);
        const save = {{ sessionId: "sid-1", path: "src/app.py" }};
        ctx.__test_save_error(save, {{ status: 409, message: "version mismatch" }});
        const conflict = {{ calls: ctx.calls.slice(), status: ctx.fileStatus.textContent }};
        ctx.calls = [];
        ctx.fileStatus.textContent = "";
        ctx.__test_save_error(save, {{ status: 500, message: "disk full" }});
        const generic = {{ calls: ctx.calls.slice(), status: ctx.fileStatus.textContent }};
        ctx.calls = [];
        ctx.fileStatus.textContent = "";
        ctx.__test_save_error(save, {{}});
        const unknown = {{ calls: ctx.calls.slice(), status: ctx.fileStatus.textContent }};
        process.stdout.write(JSON.stringify({{ conflict, generic, unknown }}));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_active_file_save_success() -> dict:
    source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const state = {{
          sessionId: "sid-1",
          kind: "text",
          text: "old text",
          version: "v0",
          editable: false,
          draft: true,
          dirty: true,
          editMode: true,
        }};
        const calls = [];
        const fileStatus = {{ textContent: "", replaceChildren() {{}} }};
        const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
          el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children }}),
          fileStatus,
          fileEditButton: {{ classList: {{ toggle() {{}} }}, setAttribute() {{}} }},
          iconSvg: (name) => name,
          currentSessionId: () => state.sessionId,
          currentFileSessionId: () => state.sessionId,
          normalizeLineNumber: (value) => value == null ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          fileApiPathForPath: (_path, existing = "") => existing || "derived-token",
          isFileViewerOpen: () => state.viewerOpen !== false,
          invalidateFileViewerSessionSync: () => calls.push(["invalidateFileViewerSessionSync"]),
          hideFileUnsavedDialog: (choice) => calls.push(["hideFileUnsavedDialog", choice]),
          resetFileSearchState: () => calls.push(["resetFileSearchState"]),
          closeFilePickerMenu: (options) => calls.push(["closeFilePickerMenu", options]),
          isUnavailable: () => false,
          isTextFileKind: (kind) => kind === "text" || kind === "markdown",
          isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
          confirmReload: () => true,
          promptUnsavedFileChoice: async () => "cancel",
          restoreFileEditorText: (text) => calls.push(["restoreFileEditorText", text]),
          hideFileViewer: () => calls.push(["hideFileViewer"]),
          applyFileLoadResult: async () => true,
          setFilePath: (...args) => calls.push(["setFilePath", ...args]),
          resetFileViewerPanel: () => calls.push(["resetFileViewerPanel"]),
          normalizeDraftFilePath: (value) => String(value || "").trim(),
          inspectSessionFilePath: async () => ({{ exists: false }}),
          api: async () => ({{ kind: "text", text: "body", path: "/abs/read" }}),
          focusEditor: () => null,
          disposeOpenRender: () => calls.push(["disposeOpenRender"]),
          initialFileViewMode: "file",
          initialFileNonDiffMode: "file",
          persistFileViewMode: (mode) => calls.push(["persistFileViewMode", mode]),
          persistFileNonDiffMode: (mode) => calls.push(["persistFileNonDiffMode", mode]),
          currentFileEditorKind: () => "file",
          currentFileEditMode: () => true,
          activeFileEntry: () => null,
          fileCandidateGitStateFresh: () => false,
          isMarkdownPreviewable: () => true,
          resetActiveFileBufferState: () => calls.push(["resetActiveFileBufferState"]),
          updateFileTouchToolbar: () => calls.push(["updateFileTouchToolbar"]),
          currentFileTouchSelectMode: () => false,
          isFileTouchToolbarActive: () => false,
          fileEditorShortcutBlocked: () => false,
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
          resetFileTouchSelectionState: (options) => calls.push(["resetFileTouchSelectionState", options || {{}}]),
          moveFileTouchSelection: (direction) => calls.push(["moveFileTouchSelection", direction]),
          fileEditorDeleteCommandForKey: () => "",
          isActiveFileEditorInput: () => false,
          getActiveFileSelectionText: () => "",
          copyToClipboard: async () => {{}},
          focusActiveFileCodeEditor: () => null,
          nowMs: () => 0,
          setToast: (message) => calls.push(["toast", message]),
          setFileViewMode: () => {{}},
          applyActiveFileTextState: (nextState) => {{
            state.kind = nextState.kind;
            state.text = nextState.text;
            state.editable = nextState.editable;
            state.version = nextState.version;
            state.draft = nextState.draft;
            calls.push(["applyActiveFileTextState", nextState]);
          }},
          renderMonacoFile: async () => true,
          setFileEditMode: (value) => {{ state.editMode = Boolean(value); calls.push(["setFileEditMode", Boolean(value)]); }},
          currentActiveFileKind: () => state.kind,
          currentActiveFileDraft: () => state.draft,
          currentActiveFileVersion: () => state.version,
          currentActiveFileEditable: () => state.editable,
          currentFileDirty: () => state.dirty,
          currentActiveFileText: () => "",
          getFileEditorText: () => state.text,
          setFileDirty: (value) => {{ state.dirty = Boolean(value); calls.push(["setFileDirty", Boolean(value)]); }},
          syncFileEditorReadOnly: () => calls.push(["syncFileEditorReadOnly"]),
          fmtBytes: (value) => `${{value}}B`,
          applyFileMode: () => calls.push(["applyFileMode"]),
          rememberOpenedFile: (...args) => calls.push(["rememberOpenedFile", ...args]),
          historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
          rememberActiveFileSelection: () => calls.push(["rememberActiveFileSelection"]),
          updateFileEditButton: () => calls.push(["updateFileEditButton"]),
          renderFilePickerMenu: () => calls.push(["renderFilePickerMenu"]),
        }});
        controller.setActiveFileIdentity("new.py", {{ line: 42, gitPath: true, apiPath: "old-token" }});
        controller.applyActiveFileTextState({{ kind: state.kind, text: state.text, editable: state.editable, version: state.version, draft: state.draft }});
        controller.setFileEditMode(true);
        calls.length = 0;
        function snapshot() {{
          const identity = controller.currentActiveFileIdentity();
          return {{
            kind: controller.currentActiveFileKind(),
            text: controller.currentActiveFileText(),
            version: controller.currentActiveFileVersion(),
            editable: controller.currentActiveFileEditable(),
            draft: controller.currentActiveFileDraft(),
            path: identity.path,
            gitPath: identity.gitPath,
            apiPath: identity.apiPath,
            line: controller.currentActiveFileLine(),
            dirty: controller.currentFileDirty(),
            editMode: controller.currentFileEditMode(),
            status: fileStatus.textContent,
            calls: calls.slice(),
          }};
        }}
        const draftSave = {{ path: "new.py", text: "NEW", draft: true }};
        const draftOk = controller.applyActiveFileSaveSuccess(draftSave, {{ version: "v2", editable: true, size: 3, path: "/abs/new.py" }}, {{ exitEditMode: true }});
        const draft = {{ ok: draftOk, state: snapshot() }};
        state.kind = "markdown";
        state.text = "old again";
        state.version = "v0";
        state.editable = true;
        state.draft = false;
        state.dirty = true;
        state.editMode = true;
        controller.applyActiveFileTextState({{ kind: state.kind, text: state.text, editable: state.editable, version: state.version, draft: state.draft }});
        controller.setFileEditMode(true);
        fileStatus.textContent = "";
        calls.length = 0;
        controller.setFileDirty(true);
        calls.length = 0;
        controller.setActiveFileIdentity("existing.md", {{ line: 42, gitPath: true, apiPath: "keep-token" }});
        const nondraftSave = {{ path: "existing.md", text: "BODY", draft: false }};
        const nondraftOk = controller.applyActiveFileSaveSuccess(nondraftSave, {{}}, {{ exitEditMode: false }});
        const nondraft = {{ ok: nondraftOk, state: snapshot() }};
        process.stdout.write(JSON.stringify({{ draft, nondraft }}));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_active_file_save_transport() -> dict:
    source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const state = {{
          sessionId: "sid-1",
          kind: "text",
          text: "old",
          editorText: "NEW",
          version: "v1",
          editable: true,
          draft: false,
          dirty: true,
          editMode: true,
          unavailable: false,
          behavior: "success",
        }};
        const calls = [];
        const fileStatus = {{ textContent: "", replaceChildren() {{}} }};
        const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
          el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children }}),
          fileStatus,
          fileEditButton: {{ classList: {{ toggle() {{}} }}, setAttribute() {{}} }},
          iconSvg: (name) => name,
          currentSessionId: () => state.sessionId,
          currentFileSessionId: () => state.sessionId,
          normalizeLineNumber: (value) => value == null ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          fileApiPathForPath: (_path, existing = "") => existing || "derived-token",
          isFileViewerOpen: () => state.viewerOpen !== false,
          invalidateFileViewerSessionSync: () => calls.push(["invalidateFileViewerSessionSync"]),
          hideFileUnsavedDialog: (choice) => calls.push(["hideFileUnsavedDialog", choice]),
          resetFileSearchState: () => calls.push(["resetFileSearchState"]),
          closeFilePickerMenu: (options) => calls.push(["closeFilePickerMenu", options]),
          isUnavailable: () => Boolean(state.unavailable),
          isTextFileKind: (kind) => kind === "text" || kind === "markdown",
          isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
          confirmReload: () => true,
          promptUnsavedFileChoice: async () => "cancel",
          restoreFileEditorText: (text) => calls.push(["restoreFileEditorText", text]),
          hideFileViewer: () => calls.push(["hideFileViewer"]),
          applyFileLoadResult: async () => true,
          setFilePath: (...args) => calls.push(["setFilePath", ...args]),
          resetFileViewerPanel: () => calls.push(["resetFileViewerPanel"]),
          normalizeDraftFilePath: (value) => String(value || "").trim(),
          inspectSessionFilePath: async () => ({{ exists: false }}),
          api: async (url, options = {{}}) => {{
            calls.push(["api", url, options.method, options.body]);
            if (state.behavior === "success-stale") {{
              state.sessionId = "sid-2";
              return {{ version: "late", editable: false, size: 4, path: "/abs/late.py" }};
            }}
            if (state.behavior === "error-stale") {{
              state.sessionId = "sid-2";
              throw {{ status: 500, message: "late failure" }};
            }}
            if (state.behavior === "error-current") throw {{ status: 500, message: "disk full" }};
            return {{ version: "v2", editable: false, size: 4, path: "/abs/src.py" }};
          }},
          focusEditor: () => null,
          disposeOpenRender: () => calls.push(["disposeOpenRender"]),
          initialFileViewMode: "file",
          initialFileNonDiffMode: "file",
          persistFileViewMode: (mode) => calls.push(["persistFileViewMode", mode]),
          persistFileNonDiffMode: (mode) => calls.push(["persistFileNonDiffMode", mode]),
          currentFileEditorKind: () => "file",
          currentFileEditMode: () => true,
          activeFileEntry: () => null,
          fileCandidateGitStateFresh: () => false,
          isMarkdownPreviewable: () => true,
          resetActiveFileBufferState: () => calls.push(["resetActiveFileBufferState"]),
          updateFileTouchToolbar: () => calls.push(["updateFileTouchToolbar"]),
          currentFileTouchSelectMode: () => false,
          isFileTouchToolbarActive: () => false,
          fileEditorShortcutBlocked: () => false,
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
          resetFileTouchSelectionState: (options) => calls.push(["resetFileTouchSelectionState", options || {{}}]),
          moveFileTouchSelection: (direction) => calls.push(["moveFileTouchSelection", direction]),
          fileEditorDeleteCommandForKey: () => "",
          isActiveFileEditorInput: () => false,
          getActiveFileSelectionText: () => "",
          copyToClipboard: async () => {{}},
          focusActiveFileCodeEditor: () => null,
          nowMs: () => 0,
          setToast: (message) => calls.push(["toast", message]),
          setFileViewMode: () => {{}},
          applyActiveFileTextState: (nextState) => {{
            state.kind = nextState.kind;
            state.text = nextState.text;
            state.editable = nextState.editable;
            state.version = nextState.version;
            state.draft = nextState.draft;
            calls.push(["applyActiveFileTextState", nextState]);
          }},
          renderMonacoFile: async () => true,
          setFileEditMode: (value) => {{ state.editMode = Boolean(value); calls.push(["setFileEditMode", Boolean(value)]); }},
          currentActiveFileKind: () => state.kind,
          currentActiveFileDraft: () => state.draft,
          currentActiveFileVersion: () => state.version,
          currentActiveFileEditable: () => state.editable,
          currentFileDirty: () => state.dirty,
          currentActiveFileText: () => "",
          getFileEditorText: () => state.editorText,
          setFileDirty: (value) => {{ state.dirty = Boolean(value); calls.push(["setFileDirty", Boolean(value)]); }},
          syncFileEditorReadOnly: () => calls.push(["syncFileEditorReadOnly"]),
          fmtBytes: (value) => `${{value}}B`,
          applyFileMode: () => calls.push(["applyFileMode"]),
          rememberOpenedFile: (...args) => calls.push(["rememberOpenedFile", ...args]),
          historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
          rememberActiveFileSelection: () => calls.push(["rememberActiveFileSelection"]),
          updateFileEditButton: () => calls.push(["updateFileEditButton"]),
          renderFilePickerMenu: () => calls.push(["renderFilePickerMenu"]),
        }});
        async function runCase(behavior) {{
          state.sessionId = "sid-1";
          state.kind = "text";
          state.text = "old";
          state.editorText = "NEW";
          state.version = "v1";
          state.editable = true;
          state.draft = false;
          state.dirty = true;
          state.editMode = true;
          state.unavailable = false;
          state.behavior = behavior;
          controller.applyActiveFileTextState({{ kind: state.kind, text: state.text, editable: state.editable, version: state.version, draft: state.draft }});
          controller.setFileEditMode(true);
          calls.length = 0;
          fileStatus.textContent = "";
          controller.setFileDirty(state.dirty);
          calls.length = 0;
          controller.setActiveFileIdentity("src.py", {{ line: 9, gitPath: true, apiPath: "tok" }});
          const save = controller.beginActiveFileSaveRequest();
          const ok = await controller.submitActiveFileSave(save, {{ exitEditMode: true }});
          const identity = controller.currentActiveFileIdentity();
          return {{
            ok,
            pending: controller.isFileSavePending(),
            sessionId: state.sessionId,
            text: controller.currentActiveFileText(),
            version: controller.currentActiveFileVersion(),
            editable: controller.currentActiveFileEditable(),
            dirty: controller.currentFileDirty(),
            editMode: controller.currentFileEditMode(),
            path: identity.path,
            gitPath: identity.gitPath,
            apiPath: identity.apiPath,
            status: fileStatus.textContent,
            calls: calls.slice(),
          }};
        }}
        async function runPrecondition(overrides = {{}}) {{
          state.sessionId = overrides.sessionId === false ? "" : "sid-1";
          state.kind = overrides.kind || "text";
          state.text = "old";
          state.editorText = "NEW";
          state.version = "v1";
          state.editable = overrides.editable !== false;
          state.draft = Boolean(overrides.draft);
          state.dirty = overrides.dirty !== false;
          state.editMode = true;
          state.behavior = "success";
          if (state.kind === "text" || state.kind === "markdown") controller.applyActiveFileTextState({{ kind: state.kind, text: state.text, editable: state.editable, version: state.version, draft: state.draft }});
          else controller.applyActiveFileNonTextState(state.kind);
          if (state.editMode) controller.setFileEditMode(true);
          controller.setFileDirty(state.dirty);
          calls.length = 0;
          fileStatus.textContent = "";
          controller.clearFileViewerUnavailableSession();
          if (overrides.path === false) controller.clearActiveFileIdentity();
          else controller.setActiveFileIdentity("src.py", {{ line: 9, gitPath: true, apiPath: "tok" }});
          if (overrides.unavailable) {{
            controller.disableFileViewerForUnavailableSession("sid-1");
            calls.length = 0;
            fileStatus.textContent = "Session is no longer available; copy unsaved edits before closing.";
          }}
          const ok = await controller.saveActiveFileEdits({{ exitEditMode: overrides.exitEditMode !== false }});
          return {{ ok, status: fileStatus.textContent, calls: calls.slice(), dirty: controller.currentFileDirty(), editMode: controller.currentFileEditMode(), text: controller.currentActiveFileText() }};
        }}
        runCase("success")
          .then(async (success) => {{
            const staleSuccess = await runCase("success-stale");
            const currentError = await runCase("error-current");
            const staleError = await runCase("error-stale");
            const preconditions = {{
              unavailable: await runPrecondition({{ unavailable: true }}),
              noSession: await runPrecondition({{ sessionId: false }}),
              noPath: await runPrecondition({{ path: false }}),
              nonText: await runPrecondition({{ kind: "image" }}),
              notEditable: await runPrecondition({{ editable: false }}),
              cleanExit: await runPrecondition({{ dirty: false, draft: false }}),
              dirtySubmit: await runPrecondition({{ dirty: true }}),
            }};
            process.stdout.write(JSON.stringify({{ success, staleSuccess, currentError, staleError, preconditions }}));
          }})
          .catch((err) => {{ console.error(err && err.stack ? err.stack : err); process.exit(1); }});
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_draft_file_load_choreography() -> dict:
    source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const state = {{ sessionId: "sid-1", viewMode: "preview", renderOk: true, staleAfterRender: false }};
        const calls = [];
        const fileStatus = {{ textContent: "", replaceChildren() {{}} }};
        const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
          el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children }}),
          fileStatus,
          fileEditButton: {{ classList: {{ toggle() {{}} }}, setAttribute() {{}} }},
          iconSvg: (name) => name,
          currentSessionId: () => state.sessionId,
          currentFileSessionId: () => state.sessionId,
          normalizeLineNumber: (value) => value == null ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          fileApiPathForPath: (_path, existing = "") => existing || "derived-token",
          isFileViewerOpen: () => state.viewerOpen !== false,
          invalidateFileViewerSessionSync: () => calls.push(["invalidateFileViewerSessionSync"]),
          hideFileUnsavedDialog: (choice) => calls.push(["hideFileUnsavedDialog", choice]),
          resetFileSearchState: () => calls.push(["resetFileSearchState"]),
          closeFilePickerMenu: (options) => calls.push(["closeFilePickerMenu", options]),
          isUnavailable: () => false,
          isTextFileKind: (kind) => kind === "text" || kind === "markdown",
          isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
          confirmReload: () => true,
          promptUnsavedFileChoice: async () => "cancel",
          restoreFileEditorText: (text) => calls.push(["restoreFileEditorText", text]),
          hideFileViewer: () => calls.push(["hideFileViewer"]),
          applyFileLoadResult: async () => true,
          setFilePath: (...args) => calls.push(["setFilePath", ...args]),
          resetFileViewerPanel: () => calls.push(["resetFileViewerPanel"]),
          normalizeDraftFilePath: (value) => String(value || "").trim(),
          inspectSessionFilePath: async () => ({{ exists: false }}),
          api: async () => ({{ kind: "text", text: "body", path: "/abs/read" }}),
          focusEditor: () => null,
          disposeOpenRender: () => calls.push(["disposeOpenRender"]),
          initialFileViewMode: state.viewMode,
          initialFileNonDiffMode: state.viewMode === "preview" ? "preview" : "file",
          persistFileViewMode: (mode) => {{ state.viewMode = mode; calls.push(["persistFileViewMode", mode]); }},
          persistFileNonDiffMode: (mode) => calls.push(["persistFileNonDiffMode", mode]),
          currentFileEditorKind: () => state.editorKind || "file",
          currentFileEditMode: () => state.editMode !== false,
          activeFileEntry: () => null,
          fileCandidateGitStateFresh: () => false,
          isMarkdownPreviewable: () => true,
          resetActiveFileBufferState: () => calls.push(["resetActiveFileBufferState"]),
          updateFileTouchToolbar: () => calls.push(["updateFileTouchToolbar"]),
          currentFileTouchSelectMode: () => false,
          isFileTouchToolbarActive: () => false,
          fileEditorShortcutBlocked: () => false,
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
          resetFileTouchSelectionState: (options) => calls.push(["resetFileTouchSelectionState", options || {{}}]),
          moveFileTouchSelection: (direction) => calls.push(["moveFileTouchSelection", direction]),
          fileEditorDeleteCommandForKey: () => "",
          isActiveFileEditorInput: () => false,
          getActiveFileSelectionText: () => "",
          copyToClipboard: async () => {{}},
          focusActiveFileCodeEditor: () => null,
          nowMs: () => 0,
          setToast: (message) => calls.push(["toast", message]),
          setFileViewMode: (mode) => {{ calls.push(["setFileViewMode", mode]); state.viewMode = mode; }},
          applyActiveFileTextState: (nextState) => calls.push(["applyActiveFileTextState", nextState]),
          renderMonacoFile: async (...args) => {{ calls.push(["renderMonacoFile", ...args.slice(0, 4)]); if (state.staleAfterRender) state.sessionId = "sid-2"; return state.renderOk; }},
          setFileEditMode: (mode) => calls.push(["setFileEditMode", mode]),
          currentActiveFileKind: () => "text",
          currentActiveFileDraft: () => false,
          currentActiveFileVersion: () => "",
          currentActiveFileEditable: () => true,
          currentFileDirty: () => true,
          currentActiveFileText: () => "",
          getFileEditorText: () => "",
          setFileDirty: () => calls.push(["setFileDirty"]),
          syncFileEditorReadOnly: () => calls.push(["syncFileEditorReadOnly"]),
          fmtBytes: (value) => `${{value}}B`,
          applyFileMode: () => calls.push(["applyFileMode"]),
          rememberOpenedFile: (...args) => calls.push(["rememberOpenedFile", ...args]),
          historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
          rememberActiveFileSelection: () => calls.push(["rememberActiveFileSelection"]),
          updateFileEditButton: () => calls.push(["updateFileEditButton"]),
          renderFilePickerMenu: () => calls.push(["renderFilePickerMenu"]),
        }});
        function request() {{ return {{ requestId: 0, sessionId: "sid-1", path: "new/file.txt", apiPath: "", line: 7 }}; }}
        async function run() {{
          controller.setActiveFileIdentity("new/file.txt", {{ line: 7, gitPath: false, apiPath: "" }});
          const ok = await controller.applyDraftFileLoad("new/file.txt", request());
          const success = {{ ok, calls: calls.slice(), status: fileStatus.textContent }};
          calls.length = 0;
          fileStatus.textContent = "";
          state.sessionId = "sid-1";
          state.viewMode = "file";
          state.renderOk = false;
          state.staleAfterRender = false;
          controller.setActiveFileIdentity("new/file.txt", {{ line: 7, gitPath: false, apiPath: "" }});
          const renderFalse = await controller.applyDraftFileLoad("new/file.txt", request());
          const failedRender = {{ ok: renderFalse, calls: calls.slice(), status: fileStatus.textContent }};
          calls.length = 0;
          fileStatus.textContent = "";
          state.sessionId = "sid-1";
          state.viewMode = "file";
          state.renderOk = true;
          state.staleAfterRender = true;
          controller.setActiveFileIdentity("new/file.txt", {{ line: 7, gitPath: false, apiPath: "" }});
          const stale = await controller.applyDraftFileLoad("new/file.txt", request());
          const staleResult = {{ ok: stale, calls: calls.slice(), status: fileStatus.textContent }};
          calls.length = 0;
          fileStatus.textContent = "";
          state.sessionId = "sid-1";
          state.viewMode = "file";
          state.renderOk = true;
          state.staleAfterRender = false;
          const primitiveReturn = await controller.openDraftFilePath("new/file.txt", {{ line: 7 }});
          const primitiveSuccess = {{ returnedUndefined: primitiveReturn === undefined, calls: calls.slice(), status: fileStatus.textContent }};
          calls.length = 0;
          fileStatus.textContent = "";
          const primitiveInvalidReturn = await controller.openDraftFilePath("   ", {{ line: 3 }});
          const primitiveInvalid = {{ returnedUndefined: primitiveInvalidReturn === undefined, calls: calls.slice(), status: fileStatus.textContent }};
          calls.length = 0;
          fileStatus.textContent = "";
          state.sessionId = "";
          const primitiveNoSessionReturn = await controller.openDraftFilePath("new/file.txt", {{ line: 7 }});
          const primitiveNoSession = {{ returnedUndefined: primitiveNoSessionReturn === undefined, calls: calls.slice(), status: fileStatus.textContent }};
          process.stdout.write(JSON.stringify({{ success, failedRender, staleResult, primitiveSuccess, primitiveInvalid, primitiveNoSession }}));
        }}
        run().catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_file_open_success_finalizer() -> dict:
    source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const calls = [];
        const fileStatus = {{ textContent: "", replaceChildren() {{}} }};
        const fileEditButton = {{
          disabled: false,
          innerHTML: "",
          title: "",
          attrs: {{}},
          classList: {{ toggle(name, enabled) {{ calls.push(["buttonToggle", name, Boolean(enabled)]); }} }},
          setAttribute(name, value) {{ this.attrs[name] = String(value); calls.push(["buttonAttr", name, String(value)]); }},
        }};
        const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
          el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children: Array.isArray(children) ? children : [] }}),
          fileStatus,
          fileEditButton,
          iconSvg: (name) => `icon:${{name}}`,
          currentSessionId: () => "sid-1",
          currentFileSessionId: () => "sid-1",
          normalizeLineNumber: (value) => value == null || value === "" ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          fileApiPathForPath: (_path, existing) => existing || "",
          isFileViewerOpen: () => true,
          invalidateFileViewerSessionSync: () => calls.push(["invalidateFileViewerSessionSync"]),
          hideFileUnsavedDialog: (choice) => calls.push(["hideFileUnsavedDialog", choice]),
          resetFileSearchState: () => calls.push(["resetFileSearchState"]),
          closeFilePickerMenu: (options) => calls.push(["closeFilePickerMenu", options]),
          isTextFileKind: (kind) => kind === "text" || kind === "markdown",
          isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
          confirmReload: () => true,
          promptUnsavedFileChoice: async () => "cancel",
          restoreFileEditorText: (text) => calls.push(["restoreFileEditorText", text]),
          hideFileViewer: () => calls.push(["hideFileViewer"]),
          applyFileLoadResult: async () => true,
          setFilePath: (...args) => calls.push(["setFilePath", ...args]),
          resetFileViewerPanel: () => calls.push(["resetFileViewerPanel"]),
          normalizeDraftFilePath: (value) => String(value || "").trim(),
          inspectSessionFilePath: async () => ({{ exists: false }}),
          api: async () => ({{}}),
          focusEditor: () => ({{ focus() {{}}, updateOptions(opts) {{ calls.push(["editorOptions", opts]); }} }}),
          disposeOpenRender: () => calls.push(["disposeOpenRender"]),
          initialFileViewMode: "file",
          initialFileNonDiffMode: "file",
          persistFileViewMode: (mode) => calls.push(["persistFileViewMode", mode]),
          persistFileNonDiffMode: (mode) => calls.push(["persistFileNonDiffMode", mode]),
          currentFileEditorKind: () => "file",
          currentFileEditMode: () => false,
          activeFileEntry: () => null,
          fileCandidateGitStateFresh: () => false,
          isMarkdownPreviewable: () => false,
          resetActiveFileBufferState: () => calls.push(["resetActiveFileBufferState"]),
          updateFileTouchToolbar: () => calls.push(["touchToolbar"]),
          currentFileTouchSelectMode: () => false,
          isFileTouchToolbarActive: () => false,
          fileEditorShortcutBlocked: () => false,
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
          resetFileTouchSelectionState: (options) => calls.push(["resetFileTouchSelectionState", options || {{}}]),
          moveFileTouchSelection: (direction) => calls.push(["moveFileTouchSelection", direction]),
          fileEditorDeleteCommandForKey: () => "",
          isActiveFileEditorInput: () => false,
          getActiveFileSelectionText: () => "",
          copyToClipboard: async () => {{}},
          focusActiveFileCodeEditor: () => null,
          nowMs: () => 0,
          setToast: (message) => calls.push(["toast", message]),
          setFileViewMode: (...args) => calls.push(["setFileViewMode", ...args]),
          applyActiveFileTextState: (next) => calls.push(["applyActiveFileTextState", next]),
          renderMonacoFile: async () => true,
          setFileEditMode: (...args) => calls.push(["setFileEditMode", ...args]),
          currentActiveFileKind: () => "text",
          currentActiveFileDraft: () => false,
          currentActiveFileVersion: () => "",
          currentActiveFileEditable: () => true,
          currentFileDirty: () => false,
          currentActiveFileText: () => "",
          getFileEditorText: () => "",
          setFileDirty: (...args) => calls.push(["setFileDirty", ...args]),
          fmtBytes: (value) => `${{value}}B`,
          applyFileMode: () => calls.push(["applyFileMode"]),
          rememberOpenedFile: (...args) => calls.push(["rememberOpenedFile", ...args]),
          historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
          rememberActiveFileSelection: () => calls.push(["rememberActiveFileSelection"]),
          renderFilePickerMenu: () => calls.push(["renderFilePickerMenu"]),
        }});
        const ok = controller.finalizeFileOpenSuccess("src/app.py", "/abs/src/app.py");
        process.stdout.write(JSON.stringify({{ ok, calls }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)

def eval_file_load_result_dispatcher() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    surface_start = source.index("function setFileRenderSurface(surface)")
    surface_end = source.index("function resetFileViewerPanel()", surface_start)
    load_start = source.index("async function applyFileLoadResult(")
    load_end = source.index("fileBtn.onclick", load_start)
    state_helpers = """
function currentActiveFileKind() { return activeFileKind; }
function currentActiveFileEditable() { return activeFileEditable; }
function applyActiveFileTextState({ kind = \"text\", text = \"\", editable = false, version = \"\", draft = false } = {}) {
  const nextKind = String(kind || \"text\");
  if (nextKind !== \"text\" && nextKind !== \"markdown\") throw new Error(\"invalid active file text kind\");
  activeFileKind = nextKind;
  activeFileText = String(text ?? \"\");
  activeFileEditable = Boolean(editable);
  activeFileVersion = typeof version === \"string\" ? version : \"\";
  activeFileDraft = Boolean(draft);
}
function applyActiveFileDiffState({ currentText = \"\", currentExists = false } = {}) {
  applyActiveFileTextState({ kind: \"text\", text: currentText, editable: Boolean(currentExists), version: \"\", draft: false });
}
function applyActiveFileNonTextState(kind) {
  const nextKind = String(kind || \"\");
  if (nextKind !== \"image\" && nextKind !== \"pdf\" && nextKind !== \"video\" && nextKind !== \"download_only\") throw new Error(\"invalid active file non-text kind\");
  activeFileKind = nextKind;
  activeFileText = \"\";
  activeFileEditable = false;
  activeFileVersion = \"\";
  activeFileDraft = false;
}
"""
    snippet = state_helpers + "\n" + source[surface_start:surface_end] + "\n" + source[load_start:load_end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          activeFileKind: "",
          activeFileText: "",
          activeFileEditable: false,
          activeFileVersion: "",
          activeFileDraft: false,
          activeVideoFallback: null,
          current: true,
          calls: [],
          fileDiff: {{ style: {{ display: "" }}, innerHTML: "" }},
          fileImage: {{ style: {{ display: "" }}, src: "", alt: "" }},
          fileVideo: {{ style: {{ display: "" }}, src: "", onerror: null, onloadedmetadata: null }},
          fileStatus: {{ textContent: "" }},
          Date: {{ now: () => 4242 }},
          isCurrentFileOpenRequest: () => ctx.current,
          applyFileMode: () => ctx.calls.push(["applyFileMode"]),
          fileViewerController: {{
            setActiveVideoFallback(nextState) {{ ctx.activeVideoFallback = nextState ? {{ token: String(nextState.token || ""), previewUrl: String(nextState.previewUrl || ""), used: Boolean(nextState.used), preparing: Boolean(nextState.preparing), rel: String(nextState.rel || ""), size: Number(nextState.size || 0) }} : null; return ctx.activeVideoFallback; }},
            currentActiveVideoFallback() {{ return ctx.activeVideoFallback ? {{ ...ctx.activeVideoFallback }} : null; }},
            clearUsedCompatibleVideoPreview(token) {{ if (!ctx.activeVideoFallback || ctx.activeVideoFallback.token !== String(token || "") || !ctx.activeVideoFallback.used) return false; ctx.activeVideoFallback = null; ctx.calls.push(["applyFileMode"]); return true; }},
            prepareFileLoadResult(rel, result, request, {{ viewMode = "file" }} = {{}}) {{
              if (!ctx.isCurrentFileOpenRequest(request)) return null;
              if (!result || typeof result.kind !== "string") throw new Error("invalid response");
              if (result.kind === "diff") {{
                const baseText = typeof result.baseText === "string" ? result.baseText : "";
                const currentText = typeof result.currentText === "string" ? result.currentText : "";
                ctx.applyActiveFileDiffState({{ currentText, currentExists: result.currentExists }});
                if (!result.baseExists && !result.currentExists) return {{ kind: "diff", noDiff: true, status: `${{rel}} - no diff` }};
                return {{ kind: "diff", noDiff: false, baseText, currentText, status: `${{rel}} - diff` }};
              }}
              if (result.kind === "image") {{
                ctx.applyActiveFileNonTextState("image");
                if (typeof result.image_url !== "string" || !result.image_url) throw new Error("invalid image response");
                const size = typeof result.size === "number" ? result.size : 0;
                return {{ kind: "image", imageUrl: result.image_url, alt: rel, status: `${{rel}} - ${{ctx.fmtBytes(size)}}` }};
              }}
              if (result.kind === "pdf") {{
                ctx.applyActiveFileNonTextState("pdf");
                if (typeof result.pdf_url !== "string" || !result.pdf_url) throw new Error("invalid pdf response");
                const size = typeof result.size === "number" ? result.size : 0;
                return {{ kind: "pdf", pdfUrl: result.pdf_url, status: `${{rel}} - PDF - ${{ctx.fmtBytes(size)}}` }};
              }}
              if (result.kind === "video") return {{ kind: "video", ...ctx.fileViewerController.prepareActiveVideoLoadResult(rel, result, request) }};
              if (result.kind === "download_only") {{
                ctx.applyActiveFileNonTextState("download_only");
                const size = typeof result.size === "number" ? result.size : 0;
                return {{ kind: "download_only", reason: String(result.reason || ""), viewerMaxBytes: Number(result.viewer_max_bytes || 0), size, status: `${{rel}} - download only - ${{ctx.fmtBytes(size)}}` }};
              }}
              if (typeof result.text !== "string") throw new Error("invalid response");
              ctx.applyActiveFileTextState({{ kind: result.kind === "markdown" ? "markdown" : "text", text: result.text, editable: Boolean(result.editable), version: typeof result.version === "string" ? result.version : "" }});
              const renderPreview = viewMode === "preview" && ctx.currentActiveFileKind() === "markdown";
              const size = typeof result.size === "number" ? result.size : result.text.length;
              const statusParts = [rel];
              if (renderPreview) statusParts.push("preview");
              if (!ctx.currentActiveFileEditable()) statusParts.push("read-only");
              statusParts.push(ctx.fmtBytes(size));
              return {{ kind: "text", text: result.text, renderPreview, status: statusParts.join(" - ") }};
            }},
            prepareActiveVideoLoadResult(rel, result, request) {{
              ctx.applyActiveFileNonTextState("video");
              if (!result || typeof result.video_url !== "string" || !result.video_url) throw new Error("invalid video response");
              const previewUrl = typeof result.video_preview_url === "string" ? result.video_preview_url : "";
              const size = typeof result.size === "number" ? result.size : 0;
              const contentType = typeof result.content_type === "string" ? result.content_type.split(";", 1)[0].trim().toLowerCase() : "";
              const token = `${{request.requestId}}:${{rel}}:${{ctx.Date.now()}}`;
              const safe = new Set(["video/mp4", "video/webm", "video/ogg"]);
              ctx.fileViewerController.setActiveVideoFallback(previewUrl ? {{ token, previewUrl, rel, size }} : null);
              ctx.applyFileMode();
              return {{ token, rel, videoUrl: result.video_url, previewUrl, size, contentType, shouldPreviewFirst: Boolean(previewUrl && contentType && !safe.has(contentType)), initialStatus: `${{rel}} - video - ${{ctx.fmtBytes(size)}}` }};
            }},
            handleActiveVideoLoadError(token, options = {{}}) {{
              const expected = String(token || "");
              const state = ctx.activeVideoFallback;
              const rel = String((state && state.rel) || options.rel || "video");
              if (!state || state.token !== expected) {{
                if (!options.previewUrl) ctx.fileStatus.textContent = `${{rel}} - video unsupported`;
                return false;
              }}
              if (ctx.fileViewerController.clearUsedCompatibleVideoPreview(expected)) {{
                options.clearVideoHandlers();
                ctx.fileStatus.textContent = `${{rel}} - video preview unavailable after conversion`;
                return true;
              }}
              options.loadPreview(expected, {{ explicit: false }});
              return true;
            }},
            handleActiveVideoLoadedMetadata(token) {{
              const state = ctx.activeVideoFallback;
              if (!state || state.token !== String(token || "") || !state.used) return false;
              ctx.fileStatus.textContent = `${{state.rel || "video"}} - compatible video preview - ${{ctx.fmtBytes(state.size)}}`;
              return true;
            }},
          }},
          disposeFileEditor: () => ctx.calls.push(["disposeFileEditor"]),
          clearFileVideo: () => {{ ctx.calls.push(["clearFileVideo"]); ctx.fileVideo.style.display = "none"; ctx.fileVideo.src = ""; }},
          renderMonacoDiff: async (...args) => {{ ctx.calls.push(["renderMonacoDiff", ...args.slice(0, 4)]); return ctx.renderOk !== false; }},
          renderPdfFile: async (...args) => {{ ctx.calls.push(["renderPdfFile", ...args]); if (ctx.staleAfterRender) ctx.current = false; return ctx.renderOk !== false; }},
          renderMonacoFile: async (...args) => {{ ctx.calls.push(["renderMonacoFile", ...args.slice(0, 4)]); return ctx.renderOk !== false; }},
          renderMarkdownPreview: (...args) => ctx.calls.push(["renderMarkdownPreview", ...args]),
          renderBlockedFileNotice: (...args) => ctx.calls.push(["renderBlockedFileNotice", ...args]),
          loadCompatibleVideoPreview: (token, opts) => {{ ctx.calls.push(["loadCompatibleVideoPreview", token, opts]); return Promise.resolve(true); }},
          resolveAppUrl: (path) => `app:${{path}}`,
          fmtBytes: (n) => `${{n}}B`,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_load_result = applyFileLoadResult;\n")}, ctx);
        const request = {{ requestId: 7, line: 5 }};
        function reset() {{
          ctx.activeFileKind = "stale";
          ctx.activeFileText = "stale text";
          ctx.activeFileEditable = true;
          ctx.activeFileVersion = "stale-version";
          ctx.activeFileDraft = true;
          ctx.activeVideoFallback = null;
          ctx.current = true;
          ctx.renderOk = true;
          ctx.staleAfterRender = false;
          ctx.calls = [];
          ctx.fileDiff.style.display = "";
          ctx.fileImage.style.display = "";
          ctx.fileImage.src = "";
          ctx.fileImage.alt = "";
          ctx.fileVideo.style.display = "";
          ctx.fileVideo.src = "";
          ctx.fileVideo.onerror = null;
          ctx.fileVideo.onloadedmetadata = null;
          ctx.fileStatus.textContent = "";
        }}
        function snapshot(ok) {{
          return {{
            ok,
            state: {{ kind: ctx.activeFileKind, text: ctx.activeFileText, editable: ctx.activeFileEditable, version: ctx.activeFileVersion, draft: ctx.activeFileDraft }},
            surface: {{ diff: ctx.fileDiff.style.display, image: ctx.fileImage.style.display, video: ctx.fileVideo.style.display }},
            calls: ctx.calls,
            status: ctx.fileStatus.textContent,
            image: {{ src: ctx.fileImage.src, alt: ctx.fileImage.alt }},
            video: {{ src: ctx.fileVideo.src, fallback: ctx.activeVideoFallback }},
          }};
        }}
        async function run(result, opts = {{}}) {{
          reset();
          Object.assign(ctx, opts);
          const ok = await ctx.__test_load_result("doc.md", result, request, opts.helperOptions || {{}});
          return snapshot(ok);
        }}
        (async () => {{
          const result = {{}};
          result.diff = await run({{ kind: "diff", baseText: "base", currentText: "current", baseExists: true, currentExists: true }});
          result.noDiff = await run({{ kind: "diff", baseText: "", currentText: "", baseExists: false, currentExists: false }});
          result.image = await run({{ kind: "image", image_url: "/img.png", size: 4 }});
          result.videoPreview = await run({{ kind: "video", video_url: "/video.mov", video_preview_url: "/preview.mp4", content_type: "video/quicktime", size: 9 }});
          result.markdownPreview = await run({{ kind: "markdown", text: "# h", editable: false, version: "v3", size: 3 }}, {{ helperOptions: {{ viewMode: "preview" }} }});
          result.pdfStale = await run({{ kind: "pdf", pdf_url: "/doc.pdf", size: 8 }}, {{ staleAfterRender: true }});
          process.stdout.write(JSON.stringify(result));
        }})().catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_file_render_surface_visibility() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function setFileRenderSurface(surface)")
    end = source.index("function resetFileViewerPanel()", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          fileDiff: {{ style: {{ display: "" }} }},
          fileImage: {{ style: {{ display: "" }} }},
          fileVideo: {{ style: {{ display: "" }} }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_surface = setFileRenderSurface;\n")}, ctx);
        function snapshot() {{
          return {{ diff: ctx.fileDiff.style.display, image: ctx.fileImage.style.display, video: ctx.fileVideo.style.display }};
        }}
        const result = {{}};
        ctx.__test_surface("diff");
        result.diff = snapshot();
        ctx.__test_surface("image");
        result.image = snapshot();
        ctx.__test_surface("video");
        result.video = snapshot();
        result.invalidThrows = false;
        try {{
          ctx.__test_surface("audio");
        }} catch (err) {{
          result.invalidThrows = err && err.message === "invalid file render surface";
        }}
        process.stdout.write(JSON.stringify(result));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


class TestFileViewerSource(unittest.TestCase):
    def test_render_empty_file_viewer_target_resets_empty_state(self) -> None:
        result = eval_empty_file_viewer_target()
        base_calls = [
            ["resetFileViewerPanel"],
            ["clearActiveFileIdentity"],
            ["resetFilePickerInput"],
            ["renderFilePickerMenu"],
        ]
        self.assertEqual(result["defaultCalls"], base_calls)
        self.assertEqual(result["defaultStatus"], "Type to search files.")
        self.assertEqual(result["touchCalls"], base_calls + [["updateFileTouchToolbar"]])
        self.assertEqual(result["touchStatus"], "Type to search files.")

    def test_hide_file_viewer_clears_active_file_identity_after_saving_selection(self) -> None:
        result = eval_hide_file_viewer_identity_cleanup()
        self.assertEqual(
            result["identity"],
            {"path": "", "apiPath": "", "gitPath": False, "line": None},
        )
        self.assertEqual(result["session"], {"id": "", "unavailable": "", "syncToken": 11})
        self.assertEqual(result["displays"], {"backdrop": "none", "viewer": "none"})
        self.assertIsNone(result["returnFocus"])
        self.assertIn(["rememberActiveFileSelection", "src/app.py", "token-1", True, 42], result["calls"])
        self.assertIn(["closeFilePickerMenu", {"restoreInput": True}, "src/app.py"], result["calls"])
        self.assertIn(["updateFileTouchToolbar", "", None], result["calls"])
        self.assertLess(
            result["calls"].index(["rememberActiveFileSelection", "src/app.py", "token-1", True, 42]),
            result["calls"].index(["updateFileTouchToolbar", "", None]),
        )
        self.assertLess(
            result["calls"].index(["closeFilePickerMenu", {"restoreInput": True}, "src/app.py"]),
            result["calls"].index(["updateFileTouchToolbar", "", None]),
        )

    def test_disable_file_viewer_for_unavailable_session_saves_then_disables(self) -> None:
        result = eval_disable_file_viewer_for_unavailable_session()
        self.assertEqual(result["savedSelections"], [{
            "key": "dead-sid",
            "path": "src/app.py",
            "apiPath": "token-1",
            "gitPath": True,
            "line": 42,
            "syncToken": 5,
            "editMode": True,
            "savePending": True,
            "saveToken": 99,
        }])
        self.assertEqual(result["state"], {
            "unavailable": "dead-sid",
            "syncToken": 6,
            "saveToken": 0,
            "savePending": False,
            "editMode": False,
            "status": "Session is no longer available; copy unsaved edits before closing.",
            "path": "src/app.py",
            "apiPath": "token-1",
            "gitPath": True,
            "line": 42,
        })
        self.assertEqual(result["calls"][0][0], "rememberActiveFileSelection")
        self.assertIn(["hideFileUnsavedDialog", "cancel", 6], result["calls"])
        self.assertIn(["cancelPendingFileOpen", 6], result["calls"])
        self.assertIn(["resetFileSearchState", 6], result["calls"])
        self.assertIn(["closeFilePickerMenu", {"restoreInput": True}, 6], result["calls"])
        self.assertIn(["syncFileEditorReadOnly", False], result["calls"])
        self.assertIn(["updateFileEditButton", False], result["calls"])
        self.assertIn(["updateFileTouchToolbar", "dead-sid"], result["calls"])

    def test_file_paste_insert_button_blocks_unavailable_session(self) -> None:
        result = eval_file_paste_insert_button_guard()
        self.assertEqual(result["unavailable"], {
            "result": False,
            "inputValue": "typed text",
            "inserted": [],
            "hidden": 0,
            "toasts": [],
            "status": "Session is no longer available; copy unsaved edits before closing.",
            "dirty": False,
        })
        self.assertEqual(result["available"], {
            "result": True,
            "inputValue": "",
            "inserted": ["allowed text"],
            "hidden": 1,
            "toasts": ["text inserted"],
            "status": "",
            "dirty": True,
        })

    def test_file_editor_disables_monaco_suggestions(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('accessibilitySupport: "off"', source)
        self.assertIn("quickSuggestions: false", source)
        self.assertIn("suggestOnTriggerCharacters: false", source)
        self.assertIn('acceptSuggestionOnEnter: "off"', source)
        self.assertIn('tabCompletion: "off"', source)
        self.assertIn('wordBasedSuggestions: "off"', source)

    def test_file_viewer_helpers_remain_app_owned(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        markdown_source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
        for helper in [
            "isTextFileKind",
            "isDiffableFileKind",
            "blockedFileMessage",
            "formatPriorityOffset",
        ]:
            self.assertIn(f"function {helper}", source)
            self.assertNotIn(f"function {helper}", markdown_source)

    def test_open_file_reference_nonliteral_uses_exported_parser(self) -> None:
        result = eval_open_file_reference_nonliteral()
        self.assertEqual(result["showCalls"], [{"path": "src/app.py", "mode": "file", "manual": False, "line": 7}])
        self.assertEqual(result["toastMessages"], ["unsupported file reference"])
        self.assertEqual(result["selectCalls"], [])

    def test_file_viewer_session_sync_aborts_after_selected_changes(self) -> None:
        result = eval_file_viewer_session_sync_race()
        self.assertFalse(result["result"])
        self.assertEqual(result["selected"], "sid-c")
        self.assertEqual(result["fileViewerSessionId"], "sid-a")
        self.assertNotIn("refreshFileCandidates", result["calls"])
        self.assertFalse(any(isinstance(call, list) and call[0] == "openFilePathWithResolvedMode" for call in result["calls"]))
        self.assertEqual(result["status"], "")

    def test_resolved_file_open_aborts_when_session_guard_turns_stale(self) -> None:
        result = eval_resolved_open_current_guard()
        self.assertFalse(result["result"])
        self.assertEqual(result["calls"], [["inspect", "b-file.txt", {"gitPath": False, "apiPath": ""}]])
        self.assertFalse(any(call[0] in {"setFilePath", "disposeOpenRender", "resetFileViewerPanel", "api", "applyFileLoadResult", "rememberOpenedFile"} for call in result["calls"]))

    def test_open_file_path_guard_validates_mode_before_mutating_state(self) -> None:
        result = eval_open_file_guard_mode_validation()
        self.assertEqual(result["invalidMessage"], "invalid file open mode")
        self.assertEqual(result["invalidCalls"], [])
        self.assertTrue(result["validResult"])
        self.assertEqual(result["validCalls"], [
            ["setFilePath", "x.txt", {"line": 4, "gitPath": True, "apiPath": "tok"}],
            ["persistFileViewMode", "diff"],
            ["applyFileMode"],
            ["renderFilePickerMenu"],
            ["disposeOpenRender"],
            ["resetFileViewerPanel"],
            ["applyFileLoadResult", "x.txt", "diff", "diff"],
            ["applyFileMode"],
            ["rememberOpenedFile", "x.txt", None],
            ["buttonToggle", "active", False],
            ["buttonToggle", "primary", False],
            ["buttonToggle", "dirty", False],
            ["buttonAttr", "aria-label", "Edit file"],
            ["touchToolbar"],
            ["renderFilePickerMenu"],
        ])
        self.assertFalse(result["staleResult"])
        self.assertEqual(result["staleCalls"], [])

    def test_open_file_path_mode_ownership_respects_resolved_mode(self) -> None:
        result = eval_open_file_path_mode_ownership()
        explicit = result["explicit"]
        self.assertTrue(explicit["result"])
        self.assertEqual(explicit["activeEntryCalls"], 0)
        self.assertEqual(explicit["fileViewMode"], "diff")
        self.assertIn(["persistFileViewMode", "diff"], explicit["calls"])
        self.assertTrue(any(call[0] == "api" and "/git/file_versions?path=stale.txt&path_token=tok" in call[1] for call in explicit["calls"]))
        self.assertFalse(any(call[0] == "api" and "/file/read" in call[1] for call in explicit["calls"]))
        self.assertIn(["applyFileLoadResult", "stale.txt", "diff", "diff"], explicit["calls"])
        self.assertIn(["rememberOpenedFile", "stale.txt", "/abs/diff"], explicit["calls"])

        fallback = result["fallback"]
        self.assertTrue(fallback["result"])
        self.assertEqual(fallback["activeEntryCalls"], 0)
        self.assertEqual(fallback["fileViewMode"], "file")
        self.assertIn(["persistFileViewMode", "file"], fallback["calls"])
        self.assertIn(["persistFileNonDiffMode", "file"], fallback["calls"])
        self.assertTrue(any(call[0] == "api" and "/file/read?path=stale.txt&path_token=tok&git_path=1" in call[1] for call in fallback["calls"]))
        self.assertFalse(any(call[0] == "api" and "/git/file_versions" in call[1] for call in fallback["calls"]))
        self.assertIn(["applyFileLoadResult", "stale.txt", "text", "file"], fallback["calls"])
        self.assertIn(["rememberOpenedFile", "stale.txt", "/abs/read"], fallback["calls"])
        self.assertEqual(result["invalidMessage"], "invalid file open mode")

    def test_resolve_file_viewer_open_target_prioritizes_sources(self) -> None:
        result = eval_file_viewer_open_target()
        self.assertEqual(result["explicit"], {
            "kind": "path",
            "source": "explicit",
            "path": "explicit.md",
            "line": 42,
            "changed": None,
            "gitPath": False,
            "apiPath": "",
        })
        self.assertEqual(result["preferred"], {
            "kind": "path",
            "source": "preferred",
            "path": "remembered.txt",
            "line": 9,
            "changed": None,
            "gitPath": True,
            "apiPath": "api-remembered",
        })
        self.assertEqual(result["first"], {
            "kind": "path",
            "source": "first",
            "path": "first.txt",
            "line": None,
            "changed": True,
            "gitPath": True,
            "apiPath": "api-first",
        })
        self.assertEqual(result["none"], {"kind": "none"})
        self.assertEqual(result["noSession"], {"kind": "none"})

    def test_file_viewer_session_sync_has_commit_guards(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("let fileViewerSessionSyncToken = 0;", viewer_source)
        self.assertNotIn("let fileViewerSessionSyncToken = 0;", source)
        self.assertIn("function beginFileViewerSessionSync()", viewer_source)
        self.assertIn("function invalidateFileViewerSessionSync()", viewer_source)
        self.assertIn("function isCurrentFileViewerSessionSync(token)", viewer_source)
        self.assertIn("let fileCandidateRequestSeq = 0;", viewer_source)
        self.assertNotIn("let fileCandidateRequestSeq = 0;", source)
        self.assertIn("function beginFileCandidateRefresh()", viewer_source)
        self.assertIn("function isCurrentFileCandidateRefresh(requestSeq)", viewer_source)
        self.assertIn("let fileSessionSelections = new Map();", viewer_source)
        self.assertNotIn("let fileSessionSelections = new Map();", source)
        self.assertIn("function rememberActiveFileSelection(sessionId = currentFileSessionId())", viewer_source)
        self.assertIn("function preferredFileSelectionForSession(sessionId)", viewer_source)
        self.assertIn("function rememberActiveFileSelection(sessionId = currentFileSessionId())", source)
        self.assertIn("return fileViewerController.rememberActiveFileSelection(sessionId);", source)
        self.assertIn("function preferredFileSelectionForSession(sessionId)", source)
        self.assertIn("return fileViewerController.preferredFileSelectionForSession(sessionId);", source)
        self.assertIn("historyFileSelectionForSession: (sessionId) => historyFileSelectionForSession(sessionId)", source)
        self.assertIn("function isFileViewerSelectionCurrent(sessionId, token = null)", source)
        self.assertIn("function isFileViewerSessionCurrent(sessionId, token = null)", source)
        ensure_start = source.index("async function ensureCurrentFileViewerSession()")
        ensure_end = source.index("function extToEditorLang", ensure_start)
        ensure_block = source[ensure_start:ensure_end]
        self.assertIn("const syncToken = fileViewerController.beginFileViewerSessionSync();", ensure_block)
        self.assertIn("if (!isFileViewerSelectionCurrent(sid, syncToken)) return false;", ensure_block)
        self.assertIn("await refreshFileCandidates({ sessionId: sid, syncToken });", ensure_block)
        self.assertIn("if (!isFileViewerSessionCurrent(sid, syncToken)) return false;", ensure_block)
        self.assertIn("const target = resolveFileViewerOpenTarget({ sessionId: sid });", ensure_block)
        self.assertIn("setFilePath(target.path, { line: target.line, gitPath: target.gitPath, apiPath: target.apiPath });", ensure_block)
        self.assertIn("await openFilePathWithResolvedMode(target.path, { line: target.line, changed: target.changed, gitPath: target.gitPath, apiPath: target.apiPath, isCurrent: () => isFileViewerSessionCurrent(sid, syncToken) });", ensure_block)
        self.assertIn("renderEmptyFileViewerTarget({ updateTouchToolbar: true });", ensure_block)
        self.assertNotIn("const first = firstKey ? fileEntryMap.get(firstKey) : null;", ensure_block)
        refresh_start = source.index("async function refreshFileCandidates(")
        refresh_end = source.index("async function showFileViewer", refresh_start)
        refresh_block = source[refresh_start:refresh_end]
        self.assertIn("const requestSeq = fileViewerController.beginFileCandidateRefresh();", refresh_block)
        self.assertIn("const current = () => fileViewerController.isCurrentFileCandidateRefresh(requestSeq)", refresh_block)
        self.assertIn("if (!current()) return;", refresh_block)
        show_start = source.index("async function showFileViewer")
        show_end = source.index("function hideFileViewer", show_start)
        show_block = source[show_start:show_end]
        self.assertIn("const syncToken = fileViewerController.beginFileViewerSessionSync();", show_block)
        self.assertIn("await refreshFileCandidates({ sessionId: sid, syncToken });", show_block)
        self.assertIn("if (!isFileViewerSessionCurrent(sid, syncToken)) return;", show_block)
        self.assertIn("const target = resolveFileViewerOpenTarget({ sessionId: sid, explicitPath, explicitLine: line });", show_block)
        self.assertIn("void openFilePathWithResolvedMode(target.path, { line: target.line, changed: target.changed, gitPath: target.gitPath, apiPath: target.apiPath, isCurrent: () => isFileViewerSessionCurrent(sid, syncToken) })", show_block)
        self.assertIn("renderEmptyFileViewerTarget();", show_block)
        self.assertNotIn("const preferredGitPath = explicitPath ? false : Boolean(preferredSelection.gitPath);", show_block)
        self.assertNotIn("const first = firstKey ? fileEntryMap.get(firstKey) : null;", show_block)
        self.assertIn("fileViewerController.invalidateFileViewerSessionSync();\n          cancelPendingFileOpen();", source)
        open_start = source.index("async function openSession(sessionId")
        open_end = source.index("async function pollMessages", open_start)
        open_block = source[open_start:open_end]
        self.assertIn("const fileViewerSyncStarted = Boolean(isFileViewerOpen() && !currentFileDirty());", open_block)
        self.assertIn("void ensureCurrentFileViewerSession().catch((e) => console.error(\"file viewer session sync failed after selection\", e));", open_block)
        self.assertLess(open_block.index("const fileViewerSyncStarted"), open_block.index("messages/tail"))
        self.assertIn("if (isFileViewerOpen() && !currentFileDirty() && !fileViewerSyncStarted) {", open_block)
        self.assertIn("void ensureCurrentFileViewerSession();", open_block)
        self.assertIn("void refreshFileCandidates({ sessionId }).catch((e) => console.error(\"file candidates refresh failed after transcript load\", e));", open_block)
        resolved_start = source.index("async function openFilePathWithResolvedMode")
        resolved_end = source.index("function cloneFileCandidateEntry", resolved_start)
        resolved_block = source[resolved_start:resolved_end]
        self.assertIn("isCurrent = null", resolved_block)
        self.assertIn("return await fileViewerController.openFilePathWithResolvedMode(path, { line, changed, isCurrent, gitPath, apiPath });", resolved_block)
        self.assertNotIn("const sessionAtStart = currentFileSessionId();", resolved_block)
        self.assertNotIn("mode = await resolveFileOpenMode(path", resolved_block)
        guard_start = source.index("async function openFilePathWithGuard")
        guard_end = source.index("async function openDraftFilePathWithGuard", guard_start)
        guard_block = source[guard_start:guard_end]
        self.assertIn("return await fileViewerController.openFilePathWithGuard(path, { line, mode, isCurrent, gitPath, apiPath });", guard_block)
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        resolved_start = viewer_source.index("async function openFilePathWithResolvedMode")
        resolved_end = viewer_source.index("async function openDraftFilePathWithGuard", resolved_start)
        resolved_block = viewer_source[resolved_start:resolved_end]
        self.assertIn("isCurrent = null", resolved_block)
        self.assertIn("const sessionAtStart = currentFileSessionId();", resolved_block)
        self.assertIn("const currentGuard = typeof isCurrent === \"function\" ? isCurrent : () => currentFileSessionId() === sessionAtStart && !isFileViewerSessionUnavailable();", resolved_block)
        self.assertIn("if (!currentGuard()) return false;", resolved_block)
        self.assertIn("try {", resolved_block)
        self.assertIn("const useGitPath = gitPath === null || gitPath === undefined ? isGitFileCandidatePath(path, changed, null, token) : Boolean(gitPath);", resolved_block)
        self.assertIn("mode = await resolveFileOpenMode(path, { changed, gitPath: useGitPath, apiPath: requestApiPath });", resolved_block)
        self.assertIn("if (blockUnavailableFileAction()) return false;", resolved_block)
        self.assertIn("return await openFilePathWithGuard(path, { line, mode, isCurrent: currentGuard, gitPath: useGitPath, apiPath: requestApiPath });", resolved_block)
        guard_start = viewer_source.index("async function openFilePathWithGuard")
        guard_end = viewer_source.index("async function openFilePathWithResolvedMode", guard_start)
        guard_block = viewer_source[guard_start:guard_end]
        self.assertIn("gitPath = false", guard_block)
        self.assertIn("isCurrent = null", guard_block)
        self.assertIn("const sessionAtStart = currentFileSessionId();", guard_block)
        self.assertIn("const currentGuard = typeof isCurrent === \"function\" ? isCurrent : () => currentFileSessionId() === sessionAtStart && !isFileViewerSessionUnavailable();", guard_block)
        self.assertIn("if (!currentGuard()) return false;", guard_block)
        self.assertIn("const openMode = normalizeExplicitFileOpenMode(mode);", guard_block)
        self.assertIn("if (openMode) setFileViewMode(openMode);", guard_block)
        self.assertIn("await openFilePath(path, { line, gitPath, apiPath, mode: openMode });", guard_block)
        self.assertIn("return Boolean(currentGuard());", guard_block)
        self.assertNotIn("const diffable = canToggleMode && activeFileGitPathValue() && fileCandidateGitStateFresh", source)
        self.assertIn("function currentFileModeControlState", viewer_source)
        self.assertIn("const diffable = Boolean(canToggleMode && identity.gitPath && currentFileCandidateGitStateFresh() && entry && entry.changed && isDiffableFileKind(currentActiveFileKind()));", viewer_source)
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertNotIn("function normalizeExplicitFileOpenMode(requestedMode)", source)
        self.assertNotIn("function resolveFileOpenViewMode(request, rel, requestedMode = null)", source)
        self.assertIn("function normalizeExplicitFileOpenMode(requestedMode)", viewer_source)
        self.assertIn("function resolveFileOpenViewMode(request, rel, requestedMode = null)", viewer_source)
        self.assertIn("async function fetchFileOpenResult(request, rel, viewMode)", viewer_source)
        self.assertIn("const canUseDiffView = request && request.gitPath && currentFileCandidateGitStateFresh() && Boolean(entry && entry.changed);", viewer_source)
        self.assertIn('viewMode === "diff" && !canUseDiffView ? "file"', viewer_source)
        self.assertIn("/git/file_versions?path=${encodeURIComponent(rel)}${pathTokenQuery}", viewer_source)
        self.assertIn("/file/read?path=${encodeURIComponent(rel)}${pathTokenQuery}${gitPathQuery}", viewer_source)
        self.assertIn("git_path: save.gitPath", viewer_source)
        self.assertNotIn("async function openFilePath(nextPath = null, { line = undefined, gitPath = undefined, apiPath = undefined, mode = null } = {})", source)
        self.assertIn("async function openFilePath(nextPath = null, { line = undefined, gitPath = undefined, apiPath = undefined, mode = null } = {})", viewer_source)
        self.assertIn("const viewMode = resolveFileOpenViewMode(request, rel, mode);", viewer_source)
        self.assertIn("const openResult = await fetchFileOpenResult(request, rel, viewMode);", viewer_source)
        self.assertNotIn('const gitPathQuery = request.gitPath ? "&git_path=1" : "";', source)
        self.assertIn("if (gitPath) {\n              body.git_path = true;", source)
        self.assertNotIn("startFileOpenRequest(path, { line, gitPath: false })", source)
        self.assertIn("startFileOpenRequest(path, { line, gitPath: false })", viewer_source)
        self.assertIn("setFilePath(rel, { line: null, gitPath: false })", viewer_source)
        self.assertIn("if (save.draft) {\n        setActiveFileIdentity(save.path", viewer_source)

    def test_file_viewer_handles_selected_session_removal(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertNotIn("let fileViewerUnavailableSessionId = \"\";", source)
        self.assertIn("function isFileViewerSessionUnavailable()", source)
        self.assertIn("return fileViewerController.isFileViewerSessionUnavailable();", source)
        self.assertIn("function blockUnavailableFileAction()", source)
        self.assertIn("function disableFileViewerForUnavailableSession(sid)", source)
        self.assertIn("return fileViewerController.disableFileViewerForUnavailableSession(sid);", source)
        self.assertIn("function handleFileViewerSessionUnavailable(sessionId)", source)
        self.assertIn("return fileViewerController.handleFileViewerSessionUnavailable(sessionId);", source)
        self.assertIn("let unavailableSessionId = \"\";", viewer_source)
        self.assertIn("function isFileViewerSessionUnavailable()", viewer_source)
        transition_start = viewer_source.index("function disableFileViewerForUnavailableSession(sessionId)")
        helper_start = viewer_source.index("function handleFileViewerSessionUnavailable(sessionId)", transition_start)
        helper_end = viewer_source.index("function nextActiveFileIdentity", helper_start)
        transition_block = viewer_source[transition_start:helper_start]
        helper_block = viewer_source[helper_start:helper_end]
        self.assertIn("const viewerSessionId = normalizeSessionId(currentSessionId());", helper_block)
        self.assertIn("if (viewerSessionId && viewerSessionId !== sid) return false;", helper_block)
        self.assertIn("if (!currentFileDirty()) {", helper_block)
        self.assertIn("hideFileViewer();", helper_block)
        self.assertIn("return disableFileViewerForUnavailableSession(sid);", helper_block)
        self.assertNotIn("invalidateFileViewerSessionSync();", helper_block)
        self.assertIn("rememberActiveFileSelection(sid);", transition_block)
        self.assertIn("invalidateFileViewerSessionSync();", transition_block)
        self.assertIn("unavailableSessionId = sid;", transition_block)
        self.assertIn("clearActiveFileSaveState();", transition_block)
        self.assertIn("setFileEditMode(false);", transition_block)
        self.assertIn('hideFileUnsavedDialog("cancel");', transition_block)
        self.assertIn("cancelPendingFileOpen();", transition_block)
        self.assertIn("resetFileSearchState();", transition_block)
        self.assertIn("Session is no longer available; copy unsaved edits before closing.", transition_block)
        self.assertIn("syncFileEditorReadOnly();", transition_block)
        self.assertIn("updateFileEditButton();", transition_block)
        self.assertLess(transition_block.index("rememberActiveFileSelection(sid);"), transition_block.index("invalidateFileViewerSessionSync();"))
        self.assertIn("fileViewerController.clearFileViewerUnavailableSession();", source)
        self.assertIn("unavailable: isUnavailable(),", viewer_source)
        self.assertIn("const unavailable = Boolean(state.unavailable);", viewer_source)
        self.assertIn("return await fileViewerController.openFilePathWithResolvedMode(path, { line, changed, isCurrent, gitPath, apiPath });", source)
        resolved_start = viewer_source.index("async function openFilePathWithResolvedMode")
        resolved_end = viewer_source.index("async function openDraftFilePathWithGuard", resolved_start)
        self.assertIn("if (blockUnavailableFileAction()) return false;", viewer_source[resolved_start:resolved_end])
        self.assertIn("function renderEmptyFileViewerTarget({ updateTouchToolbar = false } = {})", source)
        self.assertEqual(source.count("resetFileViewerPanel();"), 3)
        self.assertEqual(viewer_source.count("resetFileViewerPanel();"), 2)
        hide_start = source.index("function hideFileViewer()")
        hide_end = source.index("function handleFileViewerSessionUnavailable", hide_start)
        hide_block = source[hide_start:hide_end]
        self.assertIn("rememberActiveFileSelection();", hide_block)
        self.assertIn("clearActiveFileIdentity();", hide_block)
        self.assertNotIn("activeFileLine = null;", hide_block)
        self.assertLess(hide_block.index("rememberActiveFileSelection();"), hide_block.index("clearActiveFileIdentity();"))
        self.assertLess(hide_block.index("closeFilePickerMenu({ restoreInput: true });"), hide_block.index("clearActiveFileIdentity();"))
        open_primitive_start = viewer_source.index("async function openFilePath(nextPath")
        open_primitive_end = viewer_source.index("async function applyDraftFileLoad", open_primitive_start)
        open_primitive_block = viewer_source[open_primitive_start:open_primitive_end]
        self.assertIn("if (blockUnavailableFileAction()) return false;", open_primitive_block)
        self.assertIn("fileStatus.textContent = \"Loading...\";\n      resetFileViewerPanel();\n      try {", open_primitive_block)
        self.assertIn("const viewMode = resolveFileOpenViewMode(request, rel, mode);", open_primitive_block)
        self.assertNotIn("const activeEntry = activeFileEntry();", open_primitive_block)
        self.assertNotIn("disposeFileEditor();\n          resetActiveFileBufferState();\n          fileImage.removeAttribute", open_primitive_block)
        file_picker_source = APP_FILE_PICKER_JS.read_text(encoding="utf-8")
        self.assertIn("if (blocked()) return [];", file_picker_source)
        draft_guard_start = source.index("async function openDraftFilePathWithGuard")
        draft_guard_end = source.index("async function setFileViewModeWithGuard", draft_guard_start)
        draft_guard_block = source[draft_guard_start:draft_guard_end]
        self.assertIn("return await fileViewerController.openDraftFilePathWithGuard(path);", draft_guard_block)
        self.assertIn("async function openDraftFilePathWithGuard(path)", viewer_source)
        self.assertIn("const rel = normalizeDraftFilePath(path);", viewer_source)
        self.assertIn("fileStatus.textContent = \"Choose a valid relative file path.\";", viewer_source)
        self.assertIn("const inspect = await inspectSessionFilePath(rel);", viewer_source)
        self.assertIn("fileStatus.textContent = `${rel} - path is a directory`;", viewer_source)
        self.assertIn("return await openFilePathWithGuard(rel, { line: null, mode: \"file\" });", viewer_source)
        self.assertIn("await openDraftFilePath(rel, { line: null });", viewer_source)
        draft_start = viewer_source.index("async function openDraftFilePath(path")
        draft_end = viewer_source.index("function finalizeFileOpenSuccess", draft_start)
        draft_block = viewer_source[draft_start:draft_end]
        self.assertIn("if (blockUnavailableFileAction()) return;", draft_block)
        self.assertIn("fileStatus.textContent = \"Preparing new file...\";\n      resetFileViewerPanel();\n      try {", draft_block)
        self.assertNotIn("disposeFileEditor();\n          resetActiveFileBufferState();\n          fileImage.removeAttribute", draft_block)
        self.assertIn("if (blockUnavailableFileAction()) return false;", viewer_source)
        self.assertIn("if (blockUnavailableFileAction()) return false;\n        if (!text)", viewer_source)
        self.assertIn("function insertIntoActiveFileEditor(text)", viewer_source)
        self.assertIn("return fileViewerController.insertIntoActiveFileEditor(text);", source)
        self.assertNotIn("function positionAfterInsertedText(start, text)", source)
        self.assertNotIn("return codoxearFileHelpers.positionAfterInsertedText(start, text);", source)
        self.assertIn("CodoxearFileHelpers.positionAfterInsertedText", viewer_source)
        self.assertIn("const nextCursor = positionAfterInsertedText({ lineNumber: range.startLineNumber, column: range.startColumn }, text);", viewer_source)
        self.assertIn("function activeFileEditorIdleWritable()", viewer_source)
        self.assertIn("if (!activeFileEditorIdleWritable()) return false;", viewer_source)
        self.assertIn("!isFileViewerSessionUnavailable()", source)
        self.assertIn("activeFileSaveToken === save.token", viewer_source)
        self.assertIn("fileEditMode = Boolean(nextMode) && activeFileEditModeAllowedInCurrentView();", viewer_source)
        self.assertNotIn("let fileEditMode = false;", source)
        self.assertIn("function syncFileUnsavedDialogMode()", source)
        self.assertIn('title.textContent = unavailable ? "Session unavailable" : "Unsaved changes"', source)
        self.assertIn('message.textContent = unavailable ? "This session is no longer available. Copy your edits before closing; they cannot be saved here." : "Save this file before leaving the editor?"', source)
        self.assertIn("saveBtn.hidden = unavailable;", source)
        self.assertIn("saveBtn.disabled = unavailable;", source)
        self.assertIn('discardBtn.textContent = unavailable ? "Close without saving" : "Discard"', source)
        self.assertIn("syncFileUnsavedDialogMode();", source)
        self.assertIn('$("#fileUnsavedSaveBtn").onclick = () => handleFileUnsavedSaveChoice();', source)
        self.assertIn("function handleFileUnsavedSaveChoice()", viewer_source)
        self.assertIn("if (blockUnavailableFileAction()) return false;", viewer_source)
        sessions_start = source.index("async function refreshSessionsOnce()")
        sessions_end = source.index("function appendEvent", sessions_start)
        sessions_block = source[sessions_start:sessions_end]
        self.assertIn("if (selected && !sessionIndex.has(selected)) clearSelectedSessionAfterRemoval(selected);", sessions_block)
        self.assertIn("clearDeletedSessionClientState(s.session_id);", source)
        self.assertIn("function clearSelectedSessionAfterRemoval(sessionId, { incrementPollGen = false, clearPollState = false } = {})", source)
        self.assertIn("function clearDeletedSessionClientState(sessionId)", source)
        self.assertIn("handleFileViewerSessionUnavailable(sessionId);", source)
        self.assertIn("syncAttachButtonState();", source)
        open_start = source.index("async function openSession(sessionId")
        open_end = source.index("async function pollMessages", open_start)
        open_block = source[open_start:open_end]
        self.assertIn("if (e && e.status === 404) {", open_block)
        self.assertIn("clearSelectedSessionAfterRemoval(sessionId, { clearPollState: true });", open_block)
        self.assertIn('console.error("refreshSessions failed after session disappeared", e2);', open_block)

    def test_file_open_requests_are_single_owner(self) -> None:
        result = eval_file_open_request_sequence()
        self.assertEqual(result["currentSessionId"], "sid-1")
        self.assertTrue(result["firstCurrent"])
        self.assertTrue(result["firstSignalAborted"])
        self.assertFalse(result["firstAfterSecond"])
        self.assertTrue(result["secondCurrent"])
        self.assertEqual(result["activeIdentity"]["path"], " trail.md ")
        self.assertTrue(result["secondGitPath"])
        self.assertEqual(result["secondApiPath"], "")
        self.assertEqual(result["activeLine"], 8)
        self.assertEqual(result["sameApiPath"], "tok-same")
        self.assertTrue(result["sameGitPath"])
        self.assertEqual(result["explicitApiPath"], "explicit-token")
        self.assertEqual(result["nongitApiPath"], "")
        self.assertFalse(result["nongitGitPath"])
        self.assertTrue(result["helperRejectsMissingCurrent"])
        self.assertEqual(result["clearWithLine"], {"path": "", "gitPath": False, "apiPath": "", "line": 12})
        self.assertEqual(result["clearDefault"], {"path": "", "gitPath": False, "apiPath": "", "line": None})
        self.assertEqual(result["handlePath"], "handled.txt")
        self.assertTrue(result["handleCurrentBeforeDone"])
        self.assertFalse(result["handleSignalAbortedAfterDone"])
        self.assertFalse(result["handleSignalAbortedAfterNext"])
        self.assertTrue(result["afterHandleCurrent"])
        self.assertFalse(result["secondAfterCancel"])

    def test_draft_file_load_choreography_is_single_owned(self) -> None:
        result = eval_draft_file_load_choreography()
        self.assertTrue(result["success"]["ok"])
        self.assertEqual(result["success"]["calls"], [
            ["persistFileViewMode", "file"],
            ["persistFileNonDiffMode", "file"],
            ["applyFileMode"],
            ["applyFileMode"],
            ["renderMonacoFile", "new/file.txt", "", 7, ""],
            ["updateFileTouchToolbar"],
            ["renderFilePickerMenu"],
        ])
        self.assertEqual(result["success"]["status"], "new/file.txt - new file")
        self.assertFalse(result["failedRender"]["ok"])
        self.assertEqual(result["failedRender"]["calls"], [
            ["applyFileMode"],
            ["renderMonacoFile", "new/file.txt", "", 7, ""],
        ])
        self.assertEqual(result["failedRender"]["status"], "")
        self.assertFalse(result["staleResult"]["ok"])
        self.assertEqual(result["staleResult"]["calls"], [
            ["applyFileMode"],
            ["renderMonacoFile", "new/file.txt", "", 7, ""],
        ])
        self.assertEqual(result["staleResult"]["status"], "")
        self.assertTrue(result["primitiveSuccess"]["returnedUndefined"])
        self.assertEqual(result["primitiveSuccess"]["calls"], [
            ["disposeOpenRender"],
            ["resetFileViewerPanel"],
            ["applyFileMode"],
            ["renderMonacoFile", "new/file.txt", "", 7, ""],
            ["updateFileTouchToolbar"],
            ["renderFilePickerMenu"],
        ])
        self.assertEqual(result["primitiveSuccess"]["status"], "new/file.txt - new file")
        self.assertTrue(result["primitiveInvalid"]["returnedUndefined"])
        self.assertEqual(result["primitiveInvalid"], {"returnedUndefined": True, "calls": [["disposeOpenRender"]], "status": "Choose a valid relative file path."})
        self.assertTrue(result["primitiveNoSession"]["returnedUndefined"])
        self.assertEqual(result["primitiveNoSession"], {"returnedUndefined": True, "calls": [], "status": ""})
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertNotIn("async function applyDraftFileLoad(rel, request)", source)
        self.assertNotIn("async function openDraftFilePath(path", source)
        self.assertIn("async function openDraftFilePath(path, { line = null } = {})", viewer_source)
        self.assertIn("const loaded = await applyDraftFileLoad(rel, request);\n        if (!loaded) return;", viewer_source)
        self.assertIn("async function applyDraftFileLoad(rel, request)", viewer_source)
        self.assertIn("function renderDraftFileOpenError(request, error)", viewer_source)
        self.assertIn("renderDraftFileOpenError(request, error);\n        return;", viewer_source)
        self.assertIn("applyActiveFileTextState({ text: \"\", editable: true, version: \"\", draft: true });", viewer_source)
        draft_block = viewer_source[viewer_source.index("async function applyDraftFileLoad("):viewer_source.index("function renderFileOpenError", viewer_source.index("async function applyDraftFileLoad("))]
        self.assertNotIn("rememberOpenedFile(rel", draft_block)
        draft_error_block = viewer_source[viewer_source.index("function renderDraftFileOpenError("):viewer_source.index("async function fetchFileOpenResult", viewer_source.index("function renderDraftFileOpenError("))]
        self.assertNotIn("updateFileTouchToolbar();", draft_error_block)

    def test_active_file_load_state_writers_are_single_owned(self) -> None:
        result = eval_active_file_load_state_writers()
        self.assertEqual(result["markdown"], {"kind": "markdown", "text": "# hi", "editable": False, "version": "v2", "draft": False})
        self.assertEqual(result["draft"], {"kind": "text", "text": "", "editable": True, "version": "", "draft": True})
        self.assertEqual(result["diff"], {"kind": "text", "text": "current", "editable": True, "version": "", "draft": False})
        self.assertEqual(result["image"], {"kind": "image", "text": "", "editable": False, "version": "", "draft": False})
        self.assertEqual(result["reset"], {"kind": "", "text": "", "editable": False, "version": "", "draft": False})
        self.assertTrue(result["invalidTextThrows"])
        self.assertTrue(result["invalidNonTextThrows"])
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("let activeFileKind = \"\";", viewer_source)
        self.assertNotIn("let activeFileKind = \"\";", source)
        self.assertIn("function applyActiveFileTextState({ kind = \"text\", text = \"\", editable = false, version = \"\", draft = false } = {})", viewer_source)
        self.assertIn("function applyActiveFileDiffState({ currentText = \"\", currentExists = false } = {})", viewer_source)
        self.assertIn("function applyActiveFileNonTextState(kind)", viewer_source)
        self.assertIn('throw new Error("invalid active file text kind")', viewer_source)
        self.assertIn('throw new Error("invalid active file non-text kind")', viewer_source)
        self.assertIn("function currentActiveFileKind()", source)
        self.assertIn("return fileViewerController.currentActiveFileKind();", source)
        self.assertIn("function resetActiveFileBufferState()", viewer_source)
        self.assertIn("fileViewerController.resetActiveFileBufferState();", source)
        self.assertIn('applyActiveFileTextState({ text: "", editable: true, version: "", draft: true });', viewer_source)
        self.assertIn("function prepareFileLoadResult(rel, result, request, { viewMode = \"file\" } = {})", viewer_source)
        self.assertIn("applyActiveFileDiffState({ currentText, currentExists: result.currentExists });", viewer_source)
        self.assertIn('applyActiveFileNonTextState("image");', viewer_source)
        self.assertIn('applyActiveFileNonTextState("pdf");', viewer_source)
        self.assertIn('applyActiveFileNonTextState("video");', viewer_source)
        self.assertIn('applyActiveFileNonTextState("download_only");', viewer_source)
        self.assertIn('applyActiveFileTextState({ kind: result.kind === "markdown" ? "markdown" : "text", text: result.text, editable: Boolean(result.editable), version: typeof result.version === "string" ? result.version : "" });', viewer_source)
        self.assertNotIn("applyActiveFileDiffState({ currentText, currentExists: result.currentExists });", source)
        self.assertNotIn('applyActiveFileNonTextState("image");', source)
        self.assertNotIn('applyActiveFileNonTextState("pdf");', source)
        self.assertNotIn('applyActiveFileNonTextState("video");', source)
        self.assertNotIn('applyActiveFileNonTextState("download_only");', source)
        self.assertNotIn('applyActiveFileTextState({ kind: result.kind === "markdown" ? "markdown" : "text", text: result.text, editable: Boolean(result.editable), version: typeof result.version === "string" ? result.version : "" });', source)

    def test_file_open_success_finalizer_is_single_owned(self) -> None:
        result = eval_file_open_success_finalizer()
        self.assertTrue(result["ok"])
        self.assertEqual(result["calls"], [
            ["applyFileMode"],
            ["rememberOpenedFile", "src/app.py", "/abs/src/app.py"],
            ["buttonToggle", "active", False],
            ["buttonToggle", "primary", False],
            ["buttonToggle", "dirty", False],
            ["buttonAttr", "aria-label", "Edit file"],
            ["touchToolbar"],
            ["renderFilePickerMenu"],
        ])
        source = APP_JS.read_text(encoding="utf-8")
        self.assertNotIn("function finalizeFileOpenSuccess(rel, absPath = null)", source)
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("return finalizeFileOpenSuccess(rel, openResult.absPath);", viewer_source)
        self.assertIn("function finalizeFileOpenSuccess(rel, absPath = null)", viewer_source)
        self.assertIn("rememberOpenedFile(rel, absPath);", viewer_source)
        self.assertIn('absPath: res && typeof res.abs_path === "string" ? res.abs_path : null', viewer_source)
        self.assertIn('absPath: res && typeof res.path === "string" ? res.path : null', viewer_source)

    def test_file_load_result_dispatcher_preserves_branch_state_and_rendering(self) -> None:
        result = eval_file_load_result_dispatcher()
        self.assertTrue(result["diff"]["ok"])
        self.assertEqual(result["diff"]["state"], {"kind": "text", "text": "current", "editable": True, "version": "", "draft": False})
        self.assertEqual(result["diff"]["calls"], [["renderMonacoDiff", "doc.md", "base", "current", 5]])
        self.assertEqual(result["diff"]["status"], "doc.md - diff")
        self.assertTrue(result["noDiff"]["ok"])
        self.assertEqual(result["noDiff"]["calls"], [["disposeFileEditor"]])
        self.assertEqual(result["noDiff"]["status"], "doc.md - no diff")
        self.assertTrue(result["image"]["ok"])
        self.assertEqual(result["image"]["state"], {"kind": "image", "text": "", "editable": False, "version": "", "draft": False})
        self.assertEqual(result["image"]["surface"], {"diff": "none", "image": "block", "video": "none"})
        self.assertEqual(result["image"]["image"], {"src": "app:/img.png", "alt": "doc.md"})
        self.assertTrue(result["videoPreview"]["ok"])
        self.assertEqual(result["videoPreview"]["state"]["kind"], "video")
        self.assertEqual(result["videoPreview"]["surface"], {"diff": "none", "image": "none", "video": "block"})
        self.assertEqual(result["videoPreview"]["calls"], [["applyFileMode"], ["loadCompatibleVideoPreview", "7:doc.md:4242", {"explicit": False}]])
        self.assertEqual(result["videoPreview"]["video"]["fallback"], {"token": "7:doc.md:4242", "previewUrl": "/preview.mp4", "used": False, "preparing": False, "rel": "doc.md", "size": 9})
        self.assertTrue(result["markdownPreview"]["ok"])
        self.assertEqual(result["markdownPreview"]["state"], {"kind": "markdown", "text": "# h", "editable": False, "version": "v3", "draft": False})
        self.assertEqual(result["markdownPreview"]["calls"], [["renderMarkdownPreview", "doc.md", "# h"]])
        self.assertEqual(result["markdownPreview"]["status"], "doc.md - preview - read-only - 3B")
        self.assertFalse(result["pdfStale"]["ok"])
        self.assertEqual(result["pdfStale"]["calls"], [["renderPdfFile", "doc.md", "app:/doc.pdf", {"requestId": 7, "line": 5}]])

    def test_file_render_surface_visibility_is_single_owned(self) -> None:
        result = eval_file_render_surface_visibility()
        self.assertEqual(result["diff"], {"diff": "block", "image": "none", "video": "none"})
        self.assertEqual(result["image"], {"diff": "none", "image": "block", "video": "none"})
        self.assertEqual(result["video"], {"diff": "none", "image": "none", "video": "block"})
        self.assertTrue(result["invalidThrows"])

    def test_touch_file_editor_controls_target_touch_capabilities(self) -> None:
        self.assertTrue(eval_use_touch_file_editor_controls({"(pointer: coarse)": True, "(hover: none)": False}))
        self.assertTrue(eval_use_touch_file_editor_controls({"(pointer: coarse)": False, "(hover: none)": True}))
        self.assertFalse(eval_use_touch_file_editor_controls({"(pointer: coarse)": False, "(hover: none)": False}))

    def test_touch_toolbar_supports_select_copy_paste_and_arrow_selection(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn('id: "fileTouchSelectBtn"', source)
        self.assertIn('id: "fileTouchCopyBtn"', source)
        self.assertIn('id: "fileTouchPasteBtn"', source)
        self.assertIn('id: "fileTouchUpBtn"', source)
        self.assertIn('id: "fileTouchLeftBtn"', source)
        self.assertIn('id: "fileTouchDownBtn"', source)
        self.assertIn('id: "fileTouchRightBtn"', source)
        self.assertIn('html: iconSvg("select")', source)
        self.assertIn('html: iconSvg("copy")', source)
        self.assertIn('html: iconSvg("paste")', source)
        self.assertIn('html: iconSvg("up")', source)
        self.assertIn('html: iconSvg("left")', source)
        self.assertIn('html: iconSvg("down")', source)
        self.assertIn('html: iconSvg("right")', source)
        self.assertIn("function handleFileTouchMoveButtonPress(direction)", viewer_source)
        self.assertIn("focusActiveFileCodeEditor();", viewer_source)
        self.assertIn("moveFileTouchSelection(direction);", viewer_source)
        self.assertIn("return fileViewerController.handleFileTouchSelectionKeydown(e);", source)
        self.assertIn("return fileViewerController.handleFileTouchMoveButtonPress(direction);", source)
        self.assertIn('handleFileTouchMoveButtonPress("up")', source)
        self.assertIn('handleFileTouchMoveButtonPress("left")', source)
        self.assertIn('handleFileTouchMoveButtonPress("down")', source)
        self.assertIn('handleFileTouchMoveButtonPress("right")', source)
        self.assertNotIn('focusActiveFileCodeEditor();\n          moveFileTouchSelection("up")', source)
        self.assertIn('editor.trigger("file-touch-select", "cursorMove", args);', viewer_source)
        self.assertIn('{ to: "left", by: "character", value: 1, select: true }', viewer_source)
        self.assertIn('{ to: "right", by: "character", value: 1, select: true }', viewer_source)
        self.assertIn('{ to: "up", by: "wrappedLine", value: 1, select: true }', viewer_source)
        self.assertIn('{ to: "down", by: "wrappedLine", value: 1, select: true }', viewer_source)
        self.assertIn("fileTouchSelectHead", viewer_source)
        self.assertNotIn("fileTouchSelectHead", source)
        self.assertIn('addAppEvent(document, "keydown", handleFileTouchSelectionKeydown, true);', source)
        self.assertIn('fileTouchSelectMode', viewer_source)
        self.assertNotIn('currentFileTouchSelectMode: () => fileTouchSelectMode', source)
        self.assertIn('function currentFileTouchSelectMode()', viewer_source)
        self.assertIn('function handleFileTouchSelectionKeydown(event)', viewer_source)
        self.assertIn('useTouchFileEditorControls()', source)
        self.assertNotIn('if (current.column > 1) {', source)

    def test_touch_toolbar_hides_unusable_controls(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css_source = APP_CSS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("function currentFileTouchToolbarState()", viewer_source)
        self.assertIn("copyVisible: Boolean(getActiveFileSelectionText())", viewer_source)
        self.assertIn("pasteVisible: activeFileEditorIdleTextWritable()", viewer_source)
        self.assertIn('fileTouchDpad.style.display = toolbarState.dpadVisible ? "grid" : "none";', source)
        self.assertIn('fileTouchCopyBtn.style.display = toolbarState.copyVisible ? "" : "none";', source)
        self.assertIn('fileTouchPasteBtn.style.display = toolbarState.pasteVisible ? "" : "none";', source)
        self.assertIn("justify-content: space-between;", css_source)
        self.assertIn("pointer-events: none;", css_source)
        self.assertIn("margin-left: auto;", css_source)

    def test_file_editor_capability_predicates_preserve_distinctions(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("function currentFileEditorState()", source)
        self.assertIn("return fileViewerController.currentFileEditorState();", source)
        self.assertIn("function currentFileEditorState()", viewer_source)
        self.assertIn("function fileEditorCapabilities(state)", source)
        self.assertIn("return fileViewerController.fileEditorCapabilities(state);", source)
        self.assertIn("let fileEditorKind = \"\";", viewer_source)
        self.assertNotIn("let fileEditorKind = \"\";", source)
        self.assertIn("function currentFileEditorKind()", viewer_source)
        self.assertIn("let fileEditorProgrammaticChange = false;", viewer_source)
        self.assertNotIn("let fileEditorProgrammaticChange = false;", source)
        self.assertIn("function isFileEditorProgrammaticChange()", viewer_source)
        self.assertIn("function runFileEditorProgrammaticChange(callback)", viewer_source)
        self.assertIn("fileViewerController.isFileEditorProgrammaticChange()", source)
        self.assertIn("fileViewerController.runFileEditorProgrammaticChange(() =>", source)
        self.assertIn("function setFileEditorKind(kind)", viewer_source)
        self.assertIn('throw new Error("invalid file editor kind")', viewer_source)
        self.assertIn("function currentFileEditorKind()", source)
        self.assertIn("return fileViewerController.currentFileEditorKind();", source)
        self.assertIn("function setFileEditorKind(kind)", source)
        self.assertIn("return fileViewerController.setFileEditorKind(kind);", source)
        self.assertNotIn("currentFileEditorKind: () => fileEditorKind", source)
        self.assertIn("function fileEditorCapabilities(state)", viewer_source)
        self.assertIn("return Object.freeze({ canEnterEditMode, writable, idleWritable, idleTextWritable, editModeAllowedInCurrentView });", viewer_source)
        self.assertIn("function activeFileEditorCapabilities()", source)
        self.assertIn("return fileViewerController.activeFileEditorCapabilities();", source)
        self.assertIn("return fileEditorCapabilities(currentFileEditorState());", viewer_source)
        self.assertIn("function activeFileEditorIdleTextWritable()", viewer_source)
        self.assertIn("return fileViewerController.syncFileEditorReadOnly();", source)
        self.assertIn("function syncFileEditorReadOnly()", viewer_source)
        self.assertIn("editor.updateOptions({ readOnly: !activeFileEditorWritable() });", viewer_source)
        self.assertIn("return fileViewerController.updateFileEditButton();", source)
        self.assertIn("function updateFileEditButton()", viewer_source)
        self.assertIn("fileEditButton.disabled = unavailable || !canEdit;", viewer_source)
        self.assertIn("async function handleFileEditButtonPress()", viewer_source)
        self.assertIn("return await fileViewerController.handleFileEditButtonPress();", source)
        self.assertIn("await handleFileEditButtonPress();", source)
        self.assertIn("if (isFileSavePending()) return false;", viewer_source)
        self.assertIn("await saveActiveFileEdits({ exitEditMode: true });", viewer_source)
        self.assertIn("const changed = await setFileViewModeWithGuard(\"file\");", viewer_source)
        self.assertIn("setFileEditMode(true);", viewer_source)
        self.assertIn("pasteVisible: activeFileEditorIdleTextWritable()", viewer_source)
        self.assertIn("const toolbarState = fileViewerController.currentFileTouchToolbarState();", source)
        self.assertIn("fileEditMode = Boolean(nextMode) && activeFileEditModeAllowedInCurrentView();", viewer_source)
        self.assertNotIn("let fileEditMode = false;", source)
        result = eval_file_editor_capability_predicates()

        def assert_capability_case(name: str, expected: dict[str, bool]) -> None:
            case = result[name]
            self.assertEqual(
                case["wrappers"],
                {
                    "canEnter": expected["canEnter"],
                    "writable": expected["writable"],
                    "idleWritable": expected["idleWritable"],
                    "idleTextWritable": expected["idleTextWritable"],
                    "editModeAllowed": expected["editModeAllowed"],
                },
            )
            self.assertEqual(
                case["capabilities"],
                {
                    "canEnterEditMode": expected["canEnter"],
                    "writable": expected["writable"],
                    "idleWritable": expected["idleWritable"],
                    "idleTextWritable": expected["idleTextWritable"],
                    "editModeAllowedInCurrentView": expected["editModeAllowed"],
                },
            )

        assert_capability_case("editableText", {"canEnter": True, "writable": True, "idleWritable": True, "idleTextWritable": True, "editModeAllowed": True})
        assert_capability_case("savePending", {"canEnter": False, "writable": True, "idleWritable": False, "idleTextWritable": False, "editModeAllowed": True})
        assert_capability_case("previewMode", {"canEnter": True, "writable": False, "idleWritable": False, "idleTextWritable": False, "editModeAllowed": False})
        assert_capability_case("binaryKind", {"canEnter": False, "writable": True, "idleWritable": True, "idleTextWritable": False, "editModeAllowed": False})
        assert_capability_case("unavailable", {"canEnter": False, "writable": False, "idleWritable": False, "idleTextWritable": False, "editModeAllowed": False})
        assert_capability_case("plainFallback", {"canEnter": False, "writable": True, "idleWritable": True, "idleTextWritable": True, "editModeAllowed": True})
        assert_capability_case("notEditing", {"canEnter": True, "writable": False, "idleWritable": False, "idleTextWritable": False, "editModeAllowed": True})
        assert_capability_case("missingPath", {"canEnter": False, "writable": True, "idleWritable": True, "idleTextWritable": True, "editModeAllowed": True})
        self.assertEqual(result["editableText"]["state"]["path"], "note.md")
        self.assertEqual(result["editableText"]["state"]["kind"], "markdown")
        self.assertEqual(result["editableText"]["state"]["sessionId"], "sid-1")
        self.assertFalse(result["editableText"]["state"]["unavailable"])
        self.assertTrue(result["savePending"]["state"]["savePending"])
        self.assertEqual(result["missingPath"]["state"]["path"], "")

    def test_file_editor_save_shortcut_is_scoped_to_active_edit_mode(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("function handleFileEditorSaveShortcut(e)", source)
        self.assertIn("return fileViewerController.handleFileEditorSaveShortcut(e);", source)
        self.assertIn('key !== "s" || !(event.ctrlKey || event.metaKey) || event.altKey || event.shiftKey', viewer_source)
        self.assertIn("fileEditorShortcutBlocked(target)", viewer_source)
        self.assertIn("void saveActiveFileEdits({ exitEditMode: false });", viewer_source)
        self.assertNotIn('key !== "s" || !(e.ctrlKey || e.metaKey) || e.altKey || e.shiftKey', source)
        self.assertIn('addAppEvent(document, "keydown", handleFileEditorSaveShortcut, true);', source)
        result = eval_file_editor_save_shortcut()
        for key in ("validCtrl", "validMeta"):
            with self.subTest(key=key):
                case = result[key]
                self.assertTrue(case["handled"])
                self.assertEqual(case["prevented"], 1)
                self.assertEqual(case["stopped"], 1)
                self.assertEqual(case["apiEvents"], [["api", "/api/sessions/sid-1/file/write", "POST", {"path": "note.txt", "text": "body", "version": "v1", "git_path": False}]])
        for key in ("noModifier", "wrongKey", "notEdit", "pending", "unavailable", "nestedDialog", "otherTextEntry", "noPath", "viewerClosed"):
            with self.subTest(key=key):
                case = result[key]
                self.assertFalse(case["handled"])
                self.assertEqual(case["prevented"], 0)
                self.assertEqual(case["stopped"], 0)
                self.assertEqual(case["apiEvents"], [])

    def test_touch_select_mode_refocuses_editor_and_blocks_printable_edits(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertNotIn("function focusActiveFileCodeEditor()", source)
        self.assertIn("focusActiveFileCodeEditor()", viewer_source)
        self.assertIn("syncFileDiffSelectionMode()", source)
        self.assertIn("? { enabled: false }", source)
        self.assertNotIn('function bindFileTouchPress(button, handler)', source)
        self.assertNotIn('function bindFileTouchClick(button, handler)', source)
        self.assertIn('function bindFileTouchPress(button, handler, options = {})', viewer_source)
        self.assertIn('function bindFileTouchClick(button, handler)', viewer_source)
        self.assertNotIn('button.addEventListener("pointerdown"', source)
        self.assertIn('button.addEventListener("pointerdown"', viewer_source)
        self.assertIn('"touchstart"', viewer_source)
        self.assertIn("let sawPointerTouchAt = 0;", viewer_source)
        self.assertIn("if (event && event.pointerType === \"touch\") sawPointerTouchAt = nowMs();", viewer_source)
        self.assertIn("if (nowMs() - sawPointerTouchAt < 700)", viewer_source)
        self.assertIn('touch-action: none;', APP_CSS.read_text(encoding="utf-8"))
        self.assertIn('const blocksEdit =', viewer_source)
        self.assertIn('key === "backspace"', viewer_source)
        self.assertIn('key.length === 1', viewer_source)
        self.assertIn('resetFileTouchSelectionState({ collapse: true });', viewer_source)
        self.assertIn('if (fileEditorShortcutBlocked(target)) return;', viewer_source)
        self.assertNotIn('isTextEntryElement(target) && !target.classList.contains("inputarea")', source)
        result = eval_file_touch_selection_keydown()
        self.assertEqual(result["validMove"]["prevented"], 1)
        self.assertEqual(result["validMove"]["stopped"], 1)
        self.assertEqual(result["validMove"]["moves"], ["right"])
        self.assertTrue(result["validMove"]["mode"])
        self.assertEqual(result["validEscape"]["prevented"], 1)
        self.assertEqual(result["validEscape"]["stopped"], 1)
        self.assertEqual(result["validEscape"]["moves"], [])
        self.assertFalse(result["validEscape"]["mode"])
        self.assertEqual(result["validEscape"]["selections"][-1], {"cursor": {"lineNumber": 1, "column": 1}, "anchor": None})
        self.assertEqual(result["printableBlocked"]["prevented"], 1)
        self.assertEqual(result["printableBlocked"]["stopped"], 1)
        self.assertEqual(result["printableBlocked"]["moves"], [])
        self.assertTrue(result["printableBlocked"]["mode"])
        for key in ("nestedDialog", "viewerClosed", "otherTextEntry", "outsideViewerButton", "toolbarInactive"):
            with self.subTest(key=key):
                self.assertEqual(result[key]["prevented"], 0)
                self.assertEqual(result[key]["stopped"], 0)
                self.assertEqual(result[key]["moves"], [])

    def test_delete_backspace_is_single_owned_in_touch_select_edit_mode(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        editor_source = APP_FILE_EDITOR_JS.read_text(encoding="utf-8")
        helper_source = (APP_JS.parent / "app_file_helpers.js").read_text(encoding="utf-8")
        self.assertIn("function handleFileEditorDeleteKeydown(e)", source)
        self.assertIn("return fileViewerController.handleFileEditorDeleteKeydown(e);", source)
        self.assertIn("function handleFileEditorDeleteKeydown(event)", viewer_source)
        self.assertNotIn("function isActiveFileEditorInput(target)", source)
        self.assertIn("function isActiveInput(kind, target, ElementCtor = null)", editor_source)
        self.assertIn("fileEditorRuntime.isActiveInput(currentFileEditorKind(), target, HTMLElement)", source)
        self.assertNotIn("function fileEditorDeleteCommandForKey(key)", source)
        self.assertNotIn("return codoxearFileHelpers.fileEditorDeleteCommandForKey(key);", source)
        self.assertIn("CodoxearFileHelpers.fileEditorDeleteCommandForKey", viewer_source)
        self.assertNotIn("let fileTouchDeleteNativeSuppressUntil = 0;", source)
        self.assertIn("let fileTouchDeleteNativeSuppressUntil = 0;", viewer_source)
        self.assertIn('const key = String(e.key || "").toLowerCase();', viewer_source)
        self.assertIn('if (key === "backspace") return "deleteLeft";', helper_source)
        self.assertIn('if (key === "delete") return "deleteRight";', helper_source)
        self.assertNotIn('if (key === "backspace") return "deleteLeft";', source)
        self.assertIn("fileTouchDeleteNativeSuppressUntil = nowMs() + 250;", viewer_source)
        self.assertNotIn("setFileTouchDeleteNativeSuppressUntil: (value) => { fileTouchDeleteNativeSuppressUntil = value; }", source)
        self.assertIn('editor.trigger("file-editor-delete-key", command, null);', viewer_source)
        self.assertIn("if (fileEditorShortcutBlocked(target)) return false;", viewer_source)
        self.assertNotIn("function isFileEditorNativeDeleteEvent(e)", source)
        self.assertIn("function isFileEditorNativeDeleteEvent(event)", viewer_source)
        self.assertIn('inputType !== "deleteContentBackward" && inputType !== "deleteContentForward"', viewer_source)
        self.assertIn('addAppEvent(document, "keydown", handleFileEditorDeleteKeydown, true);', source)
        self.assertIn('addAppEvent(\n          document,\n          "beforeinput",', source)
        self.assertIn('addAppEvent(\n          document,\n          "input",', source)
        self.assertIn("e.preventDefault();\n      e.stopPropagation();", viewer_source)
        self.assertNotIn("const allowEditorDelete =", source)
        self.assertIn("if (currentFileTouchSelectMode()) resetFileTouchSelectionState();", viewer_source)
        result = eval_file_editor_delete_shortcut()
        expected = {
            "validBackspace": "deleteLeft",
            "validDelete": "deleteRight",
        }
        for key, command in expected.items():
            with self.subTest(key=key):
                case = result[key]
                self.assertTrue(case["handled"])
                self.assertEqual(case["prevented"], 1)
                self.assertEqual(case["stopped"], 1)
                self.assertEqual(case["triggers"], [{"source": "file-editor-delete-key", "command": command, "payload": None}])
                self.assertEqual(case["focusCount"], 1)
                self.assertFalse(case["mode"])
                self.assertEqual(case["native"], {"result": True, "prevented": 1, "stopped": 1, "secondResult": False, "secondPrevented": 0, "secondStopped": 0})
        for key in ("nestedDialog", "viewerClosed", "otherTextEntry", "notEdit", "unavailable"):
            with self.subTest(key=key):
                case = result[key]
                self.assertFalse(case["handled"])
                self.assertEqual(case["prevented"], 0)
                self.assertEqual(case["stopped"], 0)
                self.assertEqual(case["triggers"], [])
                self.assertEqual(case["focusCount"], 0)
                self.assertTrue(case["mode"])

    def test_range_selection_does_not_collapse_back_to_cursor(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        editor_source = APP_FILE_EDITOR_JS.read_text(encoding="utf-8")
        self.assertIn('if (!nextAnchor && typeof targetEditor.setPosition === "function") targetEditor.setPosition(nextCursor);', editor_source)

    def test_file_open_race_guard_is_wired_through_fetch_and_render(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        open_file_start = viewer_source.index("async function openFilePath(nextPath")
        open_file_end = viewer_source.index("async function applyDraftFileLoad", open_file_start)
        open_file_block = viewer_source[open_file_start:open_file_end]
        self.assertNotIn("let fileOpenRequestId = 0;", source)
        self.assertNotIn("let fileOpenAbortController = null;", source)
        self.assertIn("let fileOpenRequestId = 0;", viewer_source)
        self.assertIn("let fileOpenAbortController = null;", viewer_source)
        self.assertIn("function cancelPendingFileOpen()", source)
        self.assertIn("fileViewerController.cancelPendingFileOpen();", source)
        self.assertIn("function cancelPendingFileOpen()", viewer_source)
        self.assertIn("disposeOpenRender();", viewer_source)
        self.assertIn("function nextActiveFileIdentity(current, nextPath", viewer_source)
        self.assertIn("function currentActiveFileIdentity()", viewer_source)
        self.assertIn("function clearActiveFileIdentity({ line = null } = {})", viewer_source)
        self.assertIn("clearActiveFileIdentity({ line });", source)
        self.assertNotIn("function startFileOpenRequest(nextPath = null, { line = undefined, gitPath = undefined, apiPath = undefined } = {})", source)
        self.assertIn("function startFileOpenRequest(nextPath = null, { line = undefined, gitPath = undefined, apiPath = undefined } = {})", viewer_source)
        self.assertIn("function setFileRenderSurface(surface)", source)
        self.assertIn('throw new Error("invalid file render surface")', source)
        self.assertIn("async function applyFileLoadResult(rel, result, request, { viewMode = \"file\" } = {})", source)
        self.assertNotIn("function finalizeFileOpenSuccess(rel, absPath = null)", source)
        self.assertIn("function finalizeFileOpenSuccess(rel, absPath = null)", viewer_source)
        self.assertIn("const openResult = await fetchFileOpenResult(request, rel, viewMode);", viewer_source)
        self.assertIn("const loaded = await applyFileLoadResult(rel, openResult.result, request, { viewMode });", viewer_source)
        self.assertIn("return renderFileOpenError(request, error);", open_file_block)
        self.assertIn("function renderFileOpenError(request, error)", viewer_source)
        self.assertIn("fileStatus.textContent = `error: ${error && error.message ? error.message : \"unknown error\"}`;", viewer_source)
        self.assertNotIn("fileStatus.textContent = `error: ${e && e.message ? e.message : \"unknown error\"}`;", source)
        self.assertIn('result: Object.freeze({', viewer_source)
        self.assertIn('kind: "diff"', viewer_source)
        self.assertIn('baseText: res && typeof res.base_text === "string" ? res.base_text : ""', viewer_source)
        self.assertIn('currentText: res && typeof res.current_text === "string" ? res.current_text : ""', viewer_source)
        self.assertEqual(source.count('setFileRenderSurface("diff");'), 6)
        self.assertIn('setFileRenderSurface("image");', source)
        self.assertIn('setFileRenderSurface("video");', source)
        self.assertEqual(source.count("fileDiff.style.display ="), 1)
        self.assertEqual(source.count("fileImage.style.display ="), 1)
        self.assertEqual(source.count("fileVideo.style.display ="), 2)
        self.assertNotIn("return fileViewerController.beginFileOpenRequest(nextPath, { line, gitPath, apiPath });", source)
        self.assertIn("const request = beginFileOpenRequest(nextPath, { line, gitPath, apiPath });", viewer_source)
        self.assertIn("const openRequest = startFileOpenRequest(nextPath, { line, gitPath, apiPath });", viewer_source)
        self.assertIn("const request = openRequest.request;", viewer_source)
        self.assertIn("signal: request.signal", viewer_source)
        self.assertIn("if (!isCurrentFileOpenRequest(request)) return false;", source)
        self.assertIn("if (!isCurrentFileOpenRequest(request)) return false;", viewer_source)
        self.assertIn("async function openFilePathWithResolvedMode(path, { line = null, changed = null, isCurrent = null, gitPath = null, apiPath = \"\" } = {})", source)
        self.assertIn("async function renderMonacoFile(rel, text, lineNumber = null, langOverride = \"\", request = null)", source)
        self.assertIn("async function renderMonacoDiff(rel, originalText, modifiedText, lineNumber = null, request = null)", source)
        self.assertIn("if (request && !isCurrentFileOpenRequest(request)) return false;", source)
        self.assertIn("const requestedLine = normalizeLineNumber(lineNumber);", source)
        self.assertIn("if (requestedLine) {", source)
        self.assertIn("applyEditorLineFocus(requestedLine);", source)
        self.assertNotIn("applyEditorLineFocus(targetLine);", source)
        self.assertIn("renderPlainTextFallback(rel, text, lineNumber", source)
        self.assertIn("renderPlainTextFallback(rel, modifiedText, lineNumber", source)
        fallback_block = source[source.index("function renderPlainTextFallback("):source.index("function renderDownloadFallback", source.index("function renderPlainTextFallback("))]
        self.assertIn('setFileEditorKind("plain-fallback");', fallback_block)
        self.assertIn("fileViewerController.applyPlainTextFallbackState();", fallback_block)
        self.assertNotIn("setFileEditMode(false);", fallback_block)
        self.assertNotIn("setFileDirty(false);", fallback_block)
        self.assertIn("function applyPlainTextFallbackState()", viewer_source)
        self.assertIn("setFileEditMode(false);\n      setFileDirty(false);\n      updateFileEditButton();\n      updateFileTouchToolbar();", viewer_source)
        self.assertIn("cancelPendingFileOpen();\n          if (!wasOpen) fileViewerReturnFocusEl = document.activeElement instanceof HTMLElement ? document.activeElement : null;\n          prepareModalOpen();\n          const explicitPath = String(path ?? \"\");\n          const query = String(pickerQuery ?? \"\");\n          const queryOpen = !explicitPath && query !== \"\";\n          fileBackdrop.style.display = \"block\";", source)
        self.assertIn("cancelPendingFileOpen();\n          hideFileUnsavedDialog();", source)

    def test_active_file_save_request_helpers_are_single_owned(self) -> None:
        result = eval_active_file_save_request_helpers()
        self.assertEqual(result["save"], {
            "sessionId": "sid-1",
            "path": "src/app.py",
            "apiPath": "token-1",
            "draft": True,
            "gitPath": True,
            "version": "v1",
            "text": "body text",
            "token": 1,
        })
        self.assertTrue(result["frozen"])
        self.assertFalse(result["pendingAfterBegin"])
        self.assertEqual(result["callsAfterBegin"], [["getFileEditorText"]])
        self.assertTrue(result["currentInitial"])
        self.assertTrue(result["pendingAfterMark"])
        self.assertEqual(result["statusAfterMark"], "Saving src/app.py...")
        self.assertEqual(result["callsAfterMark"], [
            ["getFileEditorText"],
            ["buttonToggle", "active", True],
            ["buttonToggle", "primary", True],
            ["buttonToggle", "dirty", True],
            ["buttonAttr", "aria-label", "Saving file"],
            ["updateFileTouchToolbar"],
            ["updateOptions", {"readOnly": False}],
        ])
        self.assertFalse(result["currentWrongApiPath"])
        self.assertFalse(result["currentWrongGitPath"])
        self.assertFalse(result["currentUnavailable"])
        self.assertTrue(result["afterMismatchedFinish"]["pending"])
        self.assertFalse(result["afterMatchedFinish"]["pending"])
        self.assertEqual(result["afterMatchedFinish"]["calls"][-6:], [
            ["updateOptions", {"readOnly": False}],
            ["buttonToggle", "active", True],
            ["buttonToggle", "primary", True],
            ["buttonToggle", "dirty", True],
            ["buttonAttr", "aria-label", "Save file"],
            ["updateFileTouchToolbar"],
        ])
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("function fileSavePendingValue()", source)
        self.assertIn("return fileViewerController.isFileSavePending();", source)
        self.assertIn("function beginActiveFileSaveRequest()", source)
        self.assertIn("return fileViewerController.beginActiveFileSaveRequest();", source)
        self.assertIn("function beginActiveFileSaveRequest()", viewer_source)
        self.assertIn("return Object.freeze({ sessionId, path, apiPath, draft, gitPath, version, text, token });", viewer_source)
        self.assertIn("function isCurrentActiveFileSaveRequest(save)", viewer_source)
        self.assertIn("currentSessionId() === save.sessionId", viewer_source)
        self.assertIn("identity.path === save.path", viewer_source)
        self.assertIn("identity.apiPath === save.apiPath", viewer_source)
        self.assertIn("identity.gitPath === save.gitPath", viewer_source)
        self.assertIn("activeFileSaveToken === save.token", viewer_source)
        self.assertIn("!isUnavailable()", viewer_source)
        self.assertIn("function markActiveFileSavePending(save)", viewer_source)
        self.assertIn("function finishActiveFileSaveRequest(save)", viewer_source)
        self.assertNotIn("let fileSavePending", source)
        self.assertNotIn("let fileSaveSeq", source)
        self.assertNotIn("let activeFileSaveToken", source)

    def test_active_file_save_body_builder_preserves_api_contract(self) -> None:
        result = eval_active_file_save_body_builder()
        self.assertEqual(result["draft"], {"path": "new.py", "text": "NEW", "create": True})
        self.assertEqual(result["gitToken"], {"path": "existing.py", "text": "BODY", "version": "v2", "git_path": True, "path_token": "tok"})
        self.assertEqual(result["gitNoToken"], {"path": "existing.py", "text": "BODY", "version": "v2", "git_path": True})
        self.assertEqual(result["plainToken"], {"path": "plain.py", "text": "TEXT", "version": "v3", "git_path": False})
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("function buildActiveFileSaveBody(save)", source)
        self.assertIn("return fileViewerController.buildActiveFileSaveBody(save);", source)
        self.assertNotIn("const saveBody = buildActiveFileSaveBody(save);", source)
        self.assertIn("const saveBody = buildActiveFileSaveBody(save);", viewer_source)
        self.assertIn("function buildActiveFileSaveBody(save)", viewer_source)
        self.assertIn("if (!save.draft && save.gitPath && save.apiPath) body.path_token = save.apiPath;", viewer_source)

    def test_active_file_save_error_renderer_preserves_conflict_and_generic_status(self) -> None:
        result = eval_active_file_save_error_renderer()
        self.assertEqual(result["conflict"], {
            "calls": [["renderSaveConflict", "sid-1", "src/app.py", "version mismatch"]],
            "status": "",
        })
        self.assertEqual(result["generic"], {"calls": [], "status": "save error: disk full"})
        self.assertEqual(result["unknown"], {"calls": [], "status": "save error: unknown error"})
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("function renderActiveFileSaveError(save, error)", source)
        self.assertIn("return fileViewerController.renderActiveFileSaveError(save, error);", source)
        self.assertIn("renderActiveFileSaveError(save, error);\n        return false;", viewer_source)
        self.assertIn("function renderActiveFileSaveError(save, error)", viewer_source)
        self.assertIn("renderSaveConflict(save.sessionId, save.path", viewer_source)
        self.assertIn("fileStatus.textContent = `save error: ${error && error.message ? error.message : \"unknown error\"}`;", viewer_source)

    def test_active_file_save_success_applies_response_state(self) -> None:
        result = eval_active_file_save_success()
        self.assertTrue(result["draft"]["ok"])
        self.assertEqual(result["draft"]["state"], {
            "kind": "text",
            "text": "NEW",
            "version": "v2",
            "editable": True,
            "draft": False,
            "path": "new.py",
            "gitPath": False,
            "apiPath": "",
            "line": 42,
            "dirty": False,
            "editMode": False,
            "status": "new.py - 3B",
            "calls": [
                ["applyFileMode"],
                ["updateFileTouchToolbar"],
                ["updateFileTouchToolbar"],
                ["updateFileTouchToolbar"],
                ["rememberOpenedFile", "new.py", "/abs/new.py"],
                ["renderFilePickerMenu"],
            ],
        })
        self.assertTrue(result["nondraft"]["ok"])
        self.assertEqual(result["nondraft"]["state"], {
            "kind": "markdown",
            "text": "BODY",
            "version": "v0",
            "editable": True,
            "draft": False,
            "path": "existing.md",
            "gitPath": True,
            "apiPath": "keep-token",
            "line": 42,
            "dirty": False,
            "editMode": True,
            "status": "existing.md - 4B",
            "calls": [
                ["applyFileMode"],
                ["updateFileTouchToolbar"],
                ["updateFileTouchToolbar"],
                ["rememberOpenedFile", "existing.md", None],
                ["renderFilePickerMenu"],
            ],
        })
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("function applyActiveFileSaveSuccess(save, res, { exitEditMode = true } = {})", source)
        self.assertIn("return fileViewerController.applyActiveFileSaveSuccess(save, res, { exitEditMode });", source)
        self.assertIn("function applyActiveFileSaveSuccess(save, res, { exitEditMode = true } = {})", viewer_source)
        self.assertIn("const nextKind = String(currentActiveFileKind() || \"text\");", viewer_source)
        self.assertIn("const nextVersion = res && typeof res.version === \"string\" ? res.version : currentActiveFileVersion();", viewer_source)
        self.assertIn("const nextEditable = res && typeof res.editable === \"boolean\" ? res.editable : currentActiveFileEditable();", viewer_source)

    def test_active_file_save_transport_preserves_currentness_returns(self) -> None:
        result = eval_active_file_save_transport()
        self.assertTrue(result["success"]["ok"])
        self.assertFalse(result["success"]["pending"])
        self.assertEqual(result["success"]["text"], "NEW")
        self.assertEqual(result["success"]["version"], "v2")
        self.assertFalse(result["success"]["editable"])
        self.assertFalse(result["success"]["dirty"])
        self.assertFalse(result["success"]["editMode"])
        self.assertEqual(result["success"]["status"], "src.py - 4B")
        self.assertIn(["api", "/api/sessions/sid-1/file/write", "POST", {"path": "src.py", "text": "NEW", "version": "v1", "git_path": True, "path_token": "tok"}], result["success"]["calls"])
        self.assertNotIn(["applyActiveFileTextState", {"kind": "text", "text": "NEW", "editable": False, "version": "v2", "draft": False}], result["success"]["calls"])
        self.assertEqual(result["success"]["calls"][-2:], [["renderFilePickerMenu"], ["updateFileTouchToolbar"]])

        self.assertTrue(result["staleSuccess"]["ok"])
        self.assertFalse(result["staleSuccess"]["pending"])
        self.assertEqual(result["staleSuccess"]["sessionId"], "sid-2")
        self.assertEqual(result["staleSuccess"]["text"], "old")
        self.assertEqual(result["staleSuccess"]["status"], "Saving src.py...")
        self.assertFalse(any(call and call[0] == "applyActiveFileTextState" for call in result["staleSuccess"]["calls"]))

        self.assertFalse(result["currentError"]["ok"])
        self.assertFalse(result["currentError"]["pending"])
        self.assertEqual(result["currentError"]["status"], "save error: disk full")
        self.assertFalse(any(call and call[0] == "applyActiveFileTextState" for call in result["currentError"]["calls"]))

        self.assertFalse(result["staleError"]["ok"])
        self.assertFalse(result["staleError"]["pending"])
        self.assertEqual(result["staleError"]["sessionId"], "sid-2")
        self.assertEqual(result["staleError"]["text"], "old")
        self.assertEqual(result["staleError"]["status"], "Saving src.py...")
        self.assertFalse(any(call and call[0] == "applyActiveFileTextState" for call in result["staleError"]["calls"]))

        preconditions = result["preconditions"]
        self.assertEqual(preconditions["unavailable"], {
            "ok": False,
            "status": "Session is no longer available; copy unsaved edits before closing.",
            "calls": [],
            "dirty": True,
            "editMode": False,
            "text": "old",
        })
        for name in ["noSession", "noPath"]:
            self.assertEqual(preconditions[name], {"ok": False, "status": "", "calls": [], "dirty": True, "editMode": True, "text": "old"})
        self.assertEqual(preconditions["notEditable"], {"ok": False, "status": "", "calls": [], "dirty": True, "editMode": False, "text": "old"})
        self.assertEqual(preconditions["nonText"], {"ok": False, "status": "", "calls": [], "dirty": True, "editMode": False, "text": ""})
        self.assertEqual(preconditions["cleanExit"], {"ok": True, "status": "", "calls": [["updateFileTouchToolbar"]], "dirty": False, "editMode": False, "text": "old"})
        self.assertTrue(preconditions["dirtySubmit"]["ok"])
        self.assertEqual(preconditions["dirtySubmit"]["status"], "src.py - 4B")
        self.assertFalse(preconditions["dirtySubmit"]["dirty"])
        self.assertFalse(preconditions["dirtySubmit"]["editMode"])
        self.assertEqual(preconditions["dirtySubmit"]["text"], "NEW")
        self.assertIn(["api", "/api/sessions/sid-1/file/write", "POST", {"path": "src.py", "text": "NEW", "version": "v1", "git_path": True, "path_token": "tok"}], preconditions["dirtySubmit"]["calls"])

    def test_file_save_response_is_bound_to_original_session_and_path(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function saveActiveFileEdits")
        end = source.index("async function maybeHandleUnsavedFileChanges", start)
        block = source[start:end]
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("return fileViewerController.beginActiveFileSaveRequest();", source)
        self.assertIn("const sessionId = currentSessionId();", viewer_source)
        self.assertIn("const identity = currentActiveFileIdentity();", viewer_source)
        self.assertIn("const path = identity.path;", viewer_source)
        self.assertIn("const apiPath = identity.apiPath || \"\";", viewer_source)
        self.assertIn("const draft = Boolean(currentActiveFileDraft());", viewer_source)
        self.assertIn("const version = currentActiveFileVersion();", viewer_source)
        self.assertIn("const text = getFileEditorText();", viewer_source)
        self.assertIn("const token = ++fileSaveSeq;", viewer_source)
        self.assertIn("activeFileSaveToken = token;", viewer_source)
        self.assertIn("return await fileViewerController.saveActiveFileEdits({ exitEditMode });", block)
        maybe_start = source.index("async function maybeHandleUnsavedFileChanges")
        maybe_end = source.index("async function openFilePathWithGuard", maybe_start)
        maybe_block = source[maybe_start:maybe_end]
        self.assertIn("return await fileViewerController.maybeHandleUnsavedFileChanges();", maybe_block)
        view_mode_start = source.index("async function setFileViewModeWithGuard")
        view_mode_end = source.index("async function requestHideFileViewer", view_mode_start)
        view_mode_block = source[view_mode_start:view_mode_end]
        self.assertIn("return await fileViewerController.setFileViewModeWithGuard(mode);", view_mode_block)
        self.assertIn("async function saveActiveFileEdits({ exitEditMode = true } = {})", viewer_source)
        self.assertIn("if (blockUnavailableFileAction()) return false;", viewer_source)
        self.assertIn("const identity = currentActiveFileIdentity();", viewer_source)
        self.assertIn("if (!currentSessionId() || !identity.path || !isTextFileKind(currentActiveFileKind()) || !currentActiveFileEditable()) return false;", viewer_source)
        self.assertIn("if (!currentFileDirty() && !currentActiveFileDraft()) {", viewer_source)
        self.assertIn("if (exitEditMode) setFileEditMode(false);", viewer_source)
        self.assertIn("const save = beginActiveFileSaveRequest();", viewer_source)
        self.assertIn("return await submitActiveFileSave(save, { exitEditMode });", viewer_source)
        self.assertIn("async function maybeHandleUnsavedFileChanges()", viewer_source)
        self.assertIn("if (!currentFileDirty()) return true;", viewer_source)
        self.assertIn("const choice = await promptUnsavedFileChoice();", viewer_source)
        self.assertIn("let fileUnsavedPromptResolver = null;", viewer_source)
        self.assertNotIn("let fileUnsavedResolver = null;", source)
        self.assertIn("function fileUnsavedPromptPlan()", viewer_source)
        self.assertIn("function beginFileUnsavedPrompt()", viewer_source)
        self.assertIn("function resolveFileUnsavedPrompt(choice = \"cancel\")", viewer_source)
        self.assertIn("const plan = fileViewerController.fileUnsavedPromptPlan();", source)
        self.assertIn("return fileViewerController.beginFileUnsavedPrompt();", source)
        self.assertIn("fileViewerController.resolveFileUnsavedPrompt(choice);", source)
        self.assertIn("function handleFileUnsavedSaveChoice()", viewer_source)
        self.assertIn("if (blockUnavailableFileAction()) return false;", viewer_source)
        self.assertIn('hideFileUnsavedDialog("save");', viewer_source)
        self.assertIn('hideFileUnsavedDialog("discard");', viewer_source)
        self.assertIn('hideFileUnsavedDialog("cancel");', viewer_source)
        self.assertIn("return fileViewerController.handleFileUnsavedSaveChoice();", source)
        self.assertIn("return fileViewerController.handleFileUnsavedDiscardChoice();", source)
        self.assertIn("return fileViewerController.handleFileUnsavedCancelChoice();", source)
        self.assertNotIn('$("#fileUnsavedSaveBtn").onclick = () => {\n          if (blockUnavailableFileAction()) return;', source)
        self.assertIn("function prepareFileEditorTextRestore(text)", viewer_source)
        self.assertIn("function finishFileEditorTextRestore()", viewer_source)
        self.assertIn("return Object.freeze({ kind: \"skip\" });", viewer_source)
        self.assertIn("return Object.freeze({ kind: \"restore\", text: restoredText });", viewer_source)
        restore_block = source[source.index("function restoreFileEditorText(text)"):source.index("function renderPlainTextFallback", source.index("function restoreFileEditorText(text)"))]
        self.assertIn("const restorePlan = fileViewerController.prepareFileEditorTextRestore(text);", restore_block)
        self.assertIn("fileViewerController.finishFileEditorTextRestore();", restore_block)
        self.assertIn("model.setValue(restorePlan.text);", restore_block)
        self.assertNotIn("setFileDirty(false);", restore_block)
        self.assertIn("function discardActiveFileEdits()", viewer_source)
        self.assertIn("restoreFileEditorText(currentActiveFileText());", viewer_source)
        self.assertIn("setFileEditMode(false);", viewer_source)
        self.assertIn("discardActiveFileEdits();", viewer_source)
        self.assertIn("if (choice === \"save\") return await saveActiveFileEdits({ exitEditMode: true });", viewer_source)
        self.assertIn("async function setFileViewModeWithGuard(mode)", viewer_source)
        self.assertIn("async function handleFileDiffModeButtonPress()", viewer_source)
        self.assertIn("return await fileViewerController.handleFileDiffModeButtonPress();", source)
        self.assertIn("void handleFileDiffModeButtonPress();", source)
        self.assertIn("const nextMode = currentFileViewMode() === \"diff\" ? currentFileNonDiffMode() : \"diff\";", viewer_source)
        self.assertNotIn("let fileNonDiffMode", source)
        self.assertNotIn("handleFileDiffModeButtonPress(fileNonDiffMode)", source)
        self.assertIn("async function handleFilePreviewModeButtonPress()", viewer_source)
        self.assertIn("return await fileViewerController.handleFilePreviewModeButtonPress();", source)
        self.assertIn("void handleFilePreviewModeButtonPress();", source)
        self.assertIn("if (!isMarkdownPreviewable(identity.path)) return false;", viewer_source)
        self.assertNotIn("isMarkdownPreviewable(activeFilePathValue())", source)
        self.assertIn("async function copyActiveFileSelection()", viewer_source)
        self.assertIn("const text = getActiveFileSelectionText();", viewer_source)
        self.assertIn("await copyToClipboard(text);", viewer_source)
        self.assertIn("resetFileTouchSelectionState({ collapse: true });", viewer_source)
        copy_wrapper_start = source.index("async function copyActiveFileSelection()")
        copy_wrapper_end = source.index("function hideFilePasteDialog", copy_wrapper_start)
        copy_wrapper_block = source[copy_wrapper_start:copy_wrapper_end]
        self.assertIn("return await fileViewerController.copyActiveFileSelection();", copy_wrapper_block)
        self.assertNotIn("await copyToClipboard(text);", copy_wrapper_block)
        self.assertIn("function activeFileDownloadApiPath()", viewer_source)
        self.assertIn("return fileViewerController.activeFileDownloadApiPath();", source)
        self.assertIn("const apiPath = activeFileDownloadApiPath();", source)
        self.assertIn("const tokenQuery = identity.gitPath && identity.apiPath ? `&path_token=${encodeURIComponent(identity.apiPath)}` : \"\";", viewer_source)
        self.assertIn("return `/api/sessions/${sessionId}/file/download?path=${encodeURIComponent(identity.path)}${tokenQuery}${identity.gitPath ? \"&git_path=1\" : \"\"}`;", viewer_source)
        self.assertNotIn("/file/download?path=${encodeURIComponent(identity.path)}", source)
        self.assertIn("if (currentActiveFileDraft() && next !== \"file\") return false;", viewer_source)
        self.assertIn("if (!(await maybeHandleUnsavedFileChanges())) return false;", viewer_source)
        self.assertIn("await openFilePath(identity.path, { line: activeFileLine, gitPath: identity.gitPath, apiPath: identity.apiPath });", viewer_source)
        hide_request_start = source.index("async function requestHideFileViewer")
        hide_request_end = source.index("function setFileViewMode", hide_request_start)
        hide_request_block = source[hide_request_start:hide_request_end]
        self.assertIn("return await fileViewerController.requestHideFileViewer();", hide_request_block)
        self.assertIn("async function requestHideFileViewer()", viewer_source)
        self.assertIn("if (!(await maybeHandleUnsavedFileChanges())) return false;", viewer_source)
        self.assertIn("hideFileViewer();", viewer_source)
        self.assertIn("return true;", viewer_source)
        self.assertIn("? { path: save.path, text: save.text, create: true }", viewer_source)
        self.assertIn(": { path: save.path, text: save.text, version: save.version, git_path: save.gitPath };", viewer_source)
        self.assertIn("if (!save.draft && save.gitPath && save.apiPath) body.path_token = save.apiPath;", viewer_source)
        self.assertIn("async function submitActiveFileSave(save, { exitEditMode = true } = {})", viewer_source)
        self.assertIn("const saveStillCurrent = () => isCurrentActiveFileSaveRequest(save);", viewer_source)
        self.assertIn("const saveBody = buildActiveFileSaveBody(save);", viewer_source)
        self.assertIn("await api(`/api/sessions/${save.sessionId}/file/write`", viewer_source)
        self.assertIn("if (!saveStillCurrent()) return true;", viewer_source)
        self.assertIn("return applyActiveFileSaveSuccess(save, res, { exitEditMode });", viewer_source)
        self.assertIn("if (!saveStillCurrent()) return false;", viewer_source)
        self.assertIn("finishActiveFileSaveRequest(save);", viewer_source)
        self.assertIn("return fileViewerController.submitActiveFileSave(save, { exitEditMode });", source)
        self.assertIn("fileStatus.textContent = `${save.path} - ${fmtBytes(size)}`;", viewer_source)
        self.assertIn("rememberOpenedFile(save.path,", viewer_source)
        self.assertNotIn("activeFileText = save.text;", source)

    def test_file_save_conflict_delegates_to_file_viewer_controller(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        css_source = APP_CSS.read_text(encoding="utf-8")
        self.assertIn("const codoxearFileViewer = window.CodoxearFileViewer;", source)
        self.assertIn("createFileViewerController", source)
        controller_start = source.index("const fileViewerController = codoxearFileViewer.createFileViewerController")
        controller_end = source.index("function beginActiveFileSaveRequest", controller_start)
        controller_block = source[controller_start:controller_end]
        self.assertNotIn("let fileViewerSessionId = \"\";", source)
        self.assertIn("let fileViewerSessionId = \"\";", viewer_source)
        self.assertIn("function currentFileViewerSessionId()", source)
        self.assertIn("return fileViewerController.currentFileViewerSessionId();", source)
        self.assertIn("currentSessionId: () => currentFileViewerSessionId()", controller_block)
        self.assertIn("currentFileSessionId: () => currentFileSessionId()", controller_block)
        self.assertIn("normalizeLineNumber", controller_block)
        self.assertIn("normalizeFileApiPath", controller_block)
        self.assertNotIn("fileApiPathForPath", controller_block)
        self.assertNotIn("isUnavailable: () => isFileViewerSessionUnavailable()", controller_block)
        self.assertIn("isFileViewerOpen: () => isFileViewerOpen()", controller_block)
        self.assertNotIn("invalidateFileViewerSessionSync: () =>", controller_block)
        self.assertIn("hideFileUnsavedDialog: (choice) => hideFileUnsavedDialog(choice)", controller_block)
        self.assertIn("resetFileSearchState: () => resetFileSearchState()", controller_block)
        self.assertIn("closeFilePickerMenu: (options) => closeFilePickerMenu(options)", controller_block)
        self.assertIn("isTextFileKind: (kind) => isTextFileKind(kind)", controller_block)
        self.assertIn("isDiffableFileKind: (kind) => isDiffableFileKind(kind)", controller_block)
        self.assertNotIn("currentFileDirty: () => fileDirty", controller_block)
        self.assertNotIn("setFileDirty: (dirty) => setFileDirty(dirty)", controller_block)
        self.assertNotIn("let fileDirty = false;", source)
        self.assertIn("let fileDirty = false;", viewer_source)
        self.assertIn("function currentFileDirty()", viewer_source)
        self.assertIn("function setFileDirty(nextDirty)", viewer_source)
        self.assertIn("return fileViewerController.currentFileDirty();", source)
        self.assertIn("return fileViewerController.setFileDirty(nextDirty);", source)
        self.assertIn("confirmReload: (message) => window.confirm(message)", controller_block)
        self.assertIn("promptUnsavedFileChoice: () => promptFileUnsavedChoice()", controller_block)
        self.assertIn("restoreFileEditorText: (text) => restoreFileEditorText(text)", controller_block)
        self.assertNotIn("discardActiveFileEdits: () => discardActiveFileEdits()", controller_block)
        self.assertIn("hideFileViewer: () => hideFileViewer()", controller_block)
        self.assertIn("applyFileLoadResult: (rel, result, request, options) => applyFileLoadResult(rel, result, request, options)", controller_block)
        self.assertNotIn("openFilePath: (path, options) => openFilePath(path, options)", controller_block)
        self.assertNotIn("openFilePathWithGuard: (path, options) => openFilePathWithGuard(path, options)", controller_block)
        self.assertIn("setFilePath: (path, options) => setFilePath(path, options)", controller_block)
        self.assertIn("resetFileViewerPanel: () => resetFileViewerPanel()", controller_block)
        self.assertNotIn("openDraftFilePath: (path, options) => openDraftFilePath(path, options)", controller_block)
        self.assertIn("normalizeDraftFilePath: (path) => normalizeDraftFilePath(path)", controller_block)
        self.assertIn("inspectSessionFilePath: (path, options) => inspectSessionFilePath(path, options)", controller_block)
        self.assertIn("focusEditor: () => fileEditorRuntime.focusActiveCodeEditor(currentFileEditorKind())", controller_block)
        self.assertIn("normalizeFileEditorPosition: (editor, position) => fileEditorRuntime.normalizePosition(editor, position)", controller_block)
        self.assertIn("applyFileEditorSelection: (editor, cursor, anchor) => fileEditorRuntime.applySelection(editor, cursor, anchor, fileEditorMonacoLoader.selectionCtor())", controller_block)
        self.assertIn("isCollapsedFileSelection: (selection) => fileEditorRuntime.isCollapsedSelection(selection)", controller_block)
        self.assertIn("getActiveFileSelectionText: () => fileEditorRuntime.activeSelectionText(currentFileEditorKind())", controller_block)
        save_start = source.index("async function saveActiveFileEdits")
        save_end = source.index("async function maybeHandleUnsavedFileChanges", save_start)
        save_block = source[save_start:save_end]
        self.assertIn("return await fileViewerController.saveActiveFileEdits({ exitEditMode });", save_block)
        self.assertIn("async function openFilePathWithGuard(path, { line = null, mode = null, isCurrent = null, gitPath = false, apiPath = \"\" } = {})", viewer_source)
        self.assertIn("const sessionAtStart = currentFileSessionId();", viewer_source)
        self.assertIn("return Boolean(currentGuard());", viewer_source)
        self.assertIn("return await submitActiveFileSave(save, { exitEditMode });", viewer_source)
        self.assertIn("renderActiveFileSaveError(save, error);", viewer_source)
        self.assertIn("renderSaveConflict(save.sessionId, save.path, error && error.message ? error.message : \"conflict\");", viewer_source)
        self.assertNotIn("function renderFileSaveConflict", source)
        self.assertNotIn("let fileSaveSeq = 0;", source)
        self.assertNotIn("let activeFileSaveToken = 0;", source)
        self.assertIn("let fileSaveSeq = 0;", viewer_source)
        self.assertIn("let activeFileSaveToken = 0;", viewer_source)
        self.assertIn("if (!save || activeFileSaveToken !== save.token) return;", viewer_source)
        self.assertIn("finishActiveFileSaveRequest(save);", viewer_source)
        self.assertIn(".fileConflictActions", css_source)


    def test_file_viewer_handles_pdf_video_and_download_only_kinds(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        css_source = APP_CSS.read_text(encoding="utf-8")
        self.assertNotIn('el("iframe"', source)
        self.assertIn('importModule(resolveAppUrl("pdf.mjs"))', viewer_source)
        self.assertIn('resolveAppUrl("pdf.worker.mjs")', viewer_source)
        self.assertNotIn('import(resolveAppUrl("pdf.mjs"))', source)
        self.assertIn('result.kind === "pdf"', viewer_source)
        self.assertIn('loadPlan.kind === "pdf"', source)
        self.assertIn("const MONACO_LOADER_TIMEOUT_MS = 4000;", source)
        self.assertIn("const PDFJS_LOADER_TIMEOUT_MS = 6000;", source)
        self.assertIn("function renderPlainTextFallback(rel, text, lineNumber = null", source)
        self.assertIn("function renderDownloadFallback(rel, url, reason = \"Preview unavailable\")", source)
        self.assertIn("async function renderPdfFile(rel, url, request)", source)
        self.assertIn('renderDownloadFallback(rel, url, "PDF lazy renderer unavailable");', source)
        self.assertIn("pdfjs.getDocument({ url, withCredentials: true })", source)
        self.assertIn("new IntersectionObserver", source)
        self.assertIn('root: fileDiff, rootMargin: "900px 0px"', source)
        self.assertIn('container.querySelectorAll(".filePdfPage").forEach((slot) => state.observer.observe(slot));', source)
        self.assertIn("state.renderTasks.add(task);", source)
        self.assertIn("disposePdfRender();", source)
        self.assertNotIn("let activePdfRender = null;", source)
        self.assertIn("let activePdfRender = null;", viewer_source)
        self.assertIn("function setActivePdfRenderState(state)", viewer_source)
        self.assertIn("function takeActivePdfRenderState()", viewer_source)
        self.assertIn("function isActivePdfRenderState(state)", viewer_source)
        self.assertIn("fileViewerController.setActivePdfRenderState(state);", source)
        self.assertIn("fileViewerController.takeActivePdfRenderState();", source)
        self.assertIn("fileViewerController.isActivePdfRenderState(state)", source)
        self.assertIn("PDF renderer timed out", viewer_source)
        self.assertIn("readyPromise.catch(() => {", viewer_source)
        self.assertNotIn("let pdfjsReadyPromise = null;", source)
        self.assertIn("function createPdfLoader(options = {})", viewer_source)
        self.assertIn("const filePdfLoader = codoxearFileViewer.createPdfLoader", source)
        self.assertIn("return await filePdfLoader.ensure();", source)
        self.assertIn('const fileVideoPreviewBtn = el("button", {', source)
        self.assertIn('id: "fileVideoPreviewBtn"', source)
        self.assertIn('title: "Use compatible MP4 preview"', source)
        self.assertIn('const fileVideo = el("video", { id: "fileVideo", class: "fileVideo", controls: true, preload: "metadata" });', source)
        self.assertIn('result.kind === "video"', viewer_source)
        self.assertIn('loadPlan.kind === "video"', source)
        self.assertIn("function clearFileVideo()", source)
        self.assertIn("fileVideo.pause();", source)
        self.assertIn('fileVideo.src = resolveAppUrl(loadPlan.videoUrl);', source)
        self.assertIn('const BROWSER_SAFE_VIDEO_TYPES = new Set(["video/mp4", "video/webm", "video/ogg"]);', viewer_source)
        self.assertIn('function prepareActiveVideoLoadResult(rel, result, request)', viewer_source)
        self.assertIn('const shouldPreviewFirst = Boolean(previewUrl && contentType && !BROWSER_SAFE_VIDEO_TYPES.has(contentType));', viewer_source)
        self.assertNotIn("const previewUrl = typeof result.video_preview_url === \"string\" ? result.video_preview_url : \"\";", source)
        self.assertNotIn('const browserSafeVideoTypes = new Set(["video/mp4", "video/webm", "video/ogg"]);', source)
        self.assertNotIn('const shouldPreviewFirst = Boolean(previewUrl && contentType && !browserSafeVideoTypes.has(contentType));', source)
        self.assertIn('async function prepareCompatibleVideoPreview(previewUrl) {', source)
        self.assertIn('headers: { Range: "bytes=0-0" }', source)
        self.assertIn('if (res.status === 401) {', source)
        self.assertIn('handleAppAuthLoss();', source)
        self.assertIn('const obj = await res.clone().json();', source)
        self.assertIn('if (obj && typeof obj.error === "string") detail = obj.error;', source)
        self.assertIn('throw new Error(detail || `video preview failed (${res.status})`);', source)
        self.assertIn('async function loadCompatibleVideoPreview(expectedToken = "", { explicit = false } = {})', source)
        self.assertIn('return await fileViewerController.loadCompatibleVideoPreview(expectedToken, {', source)
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn('async function loadCompatibleVideoPreview(expectedToken = "", options = {})', viewer_source)
        self.assertIn('async function handleFileVideoPreviewButtonPress(token, loadPreview)', viewer_source)
        self.assertIn('return await loadCompatiblePreview(token || "", { explicit: true });', viewer_source)
        self.assertIn('return await fileViewerController.handleFileVideoPreviewButtonPress(token, (nextToken, options) => loadCompatibleVideoPreview(nextToken, options));', source)
        self.assertIn('fileVideoPreviewBtn.onclick = (e) => {', source)
        self.assertIn('void handleFileVideoPreviewButtonPress();', source)
        self.assertNotIn('void loadCompatibleVideoPreview(token, { explicit: true });', source)
        self.assertIn('if (loadPlan.shouldPreviewFirst) {', source)
        self.assertIn('void loadCompatibleVideoPreview(loadPlan.token, { explicit: false });', source)
        self.assertIn("fileVideo.onerror = () => {", source)
        self.assertIn("fileViewerController.handleActiveVideoLoadError(loadPlan.token", source)
        self.assertIn("fileViewerController.handleActiveVideoLoadedMetadata(loadPlan.token);", source)
        self.assertIn("fileStatus.textContent = explicit ? `${rel} - building compatible video preview...` : `${rel} - trying compatible video preview...`;", viewer_source)
        self.assertNotIn("fileStatus.textContent = explicit ? `${rel} - building compatible video preview...` : `${rel} - trying compatible video preview...`;", source)
        self.assertIn("fileVideo.src = resolveAppUrl(previewUrl);", source)
        self.assertIn("fileStatus.textContent = `${fallback.rel || \"video\"} - compatible video preview - ${fmtBytes(fallback.size)}`;", viewer_source)
        self.assertIn("fileStatus.textContent = `${rel} - video preview unavailable after conversion`;", viewer_source)
        self.assertNotIn("fileStatus.textContent = `${rel} - compatible video preview - ${fmtBytes(size)}`;", source)
        self.assertNotIn("fileStatus.textContent = `${rel} - video preview unavailable after conversion`;", source)
        self.assertIn('setFileRenderSurface("video");', source)
        self.assertIn('fileStatus.textContent = loadPlan.initialStatus;', source)
        self.assertIn('initialStatus: `${path} - video - ${fmtBytes(size)}`', viewer_source)
        self.assertIn('result.kind === "download_only"', viewer_source)
        self.assertIn('loadPlan.kind === "download_only"', source)
        self.assertIn("renderBlockedFileNotice(rel, loadPlan.reason, loadPlan.viewerMaxBytes, loadPlan.size);", source)
        self.assertIn("reason: String(result.reason || \"\")", viewer_source)
        self.assertIn('fileStatus.textContent = loadPlan.status;', source)
        self.assertIn('status: `${path} - PDF - ${fmtBytes(size)}`', viewer_source)
        self.assertIn(".filePdfPages {", css_source)
        self.assertIn(".filePlainFallback {", css_source)
        self.assertIn(".filePlainFallbackText {", css_source)
        self.assertIn(".filePdfPage {", css_source)
        self.assertIn(".fileVideo {", css_source)
        self.assertIn(".fileBlockedNotice {", css_source)

    def test_video_preview_failure_path_surfaces_route_error(self) -> None:
        result = eval_video_preview_failure_path()

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "clip.mkv - video preview failed: bad codec")
        self.assertFalse(result["used"])
        self.assertFalse(result["preparing"])
        self.assertGreaterEqual(result["applyCount"], 2)
        self.assertEqual(result["videoSrc"], "")
        self.assertEqual(result["loadCount"], 0)
        self.assertFalse(result["authLost"])

    def test_video_preview_uses_browser_safe_server_transcode(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        config_source = (SERVER_PY.parent / "server_config.py").read_text(encoding="utf-8")
        module_source = (SERVER_PY.parent / "video_preview.py").read_text(encoding="utf-8")
        file_routes_source = (SERVER_PY.parent / "file_routes.py").read_text(encoding="utf-8")
        file_get_routes_source = (SERVER_PY.parent / "file_get_routes.py").read_text(encoding="utf-8")
        file_global_routes_source = (SERVER_PY.parent / "file_global_routes.py").read_text(encoding="utf-8")
        route_deps_source = (SERVER_PY.parent / "server_route_deps.py").read_text(encoding="utf-8")
        self.assertIn('VIDEO_PREVIEW_DIR=app_dir / "video_previews"', config_source)
        self.assertIn("_export_server_config(globals(), _SERVER_CONFIG)", server_source)
        self.assertIn("def _ensure_video_preview(path: Path) -> Path:", server_source)
        self.assertIn("return _ensure_video_preview_impl(path, preview_dir=VIDEO_PREVIEW_DIR)", server_source)
        self.assertIn("ensure_video_preview=server._ensure_video_preview", route_deps_source)
        self.assertIn('"libx264"', module_source)
        self.assertIn('"scale=ceil(iw/2)*2:ceil(ih/2)*2"', module_source)
        self.assertIn('"-pix_fmt"', module_source)
        self.assertIn('"yuv420p"', module_source)
        self.assertIn('"aac"', module_source)
        self.assertIn("from .file_get_routes import handle_absolute_file_preview_route", file_routes_source)
        self.assertIn('"video preview failed:', file_get_routes_source)
        self.assertIn('"preview_content_type": "video/mp4"', module_source)
        self.assertIn('"video_preview_url": preview_url', module_source)
        self.assertIn('VIDEO_PREVIEW_CACHE_MAX_FILES = _positive_int_env("CODEX_WEB_VIDEO_PREVIEW_MAX_FILES", 256)', module_source)
        self.assertIn('VIDEO_PREVIEW_CACHE_MAX_BYTES = _positive_int_env("CODEX_WEB_VIDEO_PREVIEW_MAX_BYTES", 10 * 1024 * 1024 * 1024)', module_source)
        self.assertIn('def prune_video_preview_cache(', module_source)
        self.assertIn('prune_video_preview_cache(preview_dir, keep=out)', module_source)
        self.assertIn("preview_url=media_preview_url", file_global_routes_source)
        self.assertIn('deps.send_inline_file_response(handler, path_obj, content_type or "application/octet-stream")', file_get_routes_source)
        self.assertIn('deps.send_inline_file_response(handler, preview, "video/mp4")', file_get_routes_source)

    def test_attach_limit_comes_from_server_constant(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("window.CODOXEAR_ATTACH_MAX_BYTES", source)
        self.assertIn("const ATTACH_UPLOAD_MAX_BYTES = (() => {", source)
        self.assertIn("Attach file (max", source)
        self.assertIn("/inject_file", source)
        self.assertIn('throw new Error(`file too large (max ${fmtBytes(maxBytes)})`);', source)

    def test_clickable_file_extensions_include_pdf_and_archives(self) -> None:
        source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
        self.assertIn('"pdf"', source)
        self.assertIn('"mp4"', source)
        self.assertIn('"mkv"', source)
        self.assertIn('"avi"', source)
        self.assertIn('"webm"', source)
        self.assertIn('"zip"', source)
        self.assertIn('"tar"', source)

    def test_touch_paste_tries_clipboard_then_manual_dialog_fallback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("navigator.clipboard", source)
        self.assertIn("readText", source)
        self.assertIn('setToast("pasted")', viewer_source)
        self.assertIn('function showFilePasteDialog()', source)
        self.assertIn('function hideFilePasteDialog({ restoreFocus = false } = {})', source)
        self.assertIn('hideFilePasteDialog({ restoreFocus: true });', source)
        self.assertIn('function requestManualFilePasteDialog()', viewer_source)
        show_start = source.index('function showFilePasteDialog()')
        show_end = source.index('async function pasteFromClipboardIntoActiveFile', show_start)
        self.assertNotIn('if (!activeFileEditorIdleTextWritable()) return false;', source[show_start:show_end])
        self.assertIn('$("#filePasteInsertBtn").onclick = () => {\n          handleFilePasteInsert(filePasteInput.value);\n        };', source)
        self.assertIn('function handleFilePasteInsert(text)', viewer_source)
        self.assertIn('filePasteDialog.style.display = "flex";', source)
        self.assertIn('filePasteInput.focus({ preventScroll: true });', source)
        self.assertIn('setToast("paste manually")', viewer_source)
        self.assertIn('function pasteFromClipboardIntoActiveFile()', viewer_source)
        self.assertIn('return await fileViewerController.pasteFromClipboardIntoActiveFile();', source)
        self.assertIn('function insertIntoActiveFileEditor(text)', viewer_source)
        self.assertIn('return fileViewerController.insertIntoActiveFileEditor(text);', source)
        self.assertIn('setToast("paste unavailable")', viewer_source)
        self.assertIn('setToast("clipboard empty")', viewer_source)
        self.assertIn('codoxearFileViewer.bindFileTouchClick(fileTouchPasteBtn, () => {', source)
        self.assertNotIn('codoxearFileViewer.bindFileTouchPress(fileTouchPasteBtn, () => {', source)

    def test_touch_paste_manual_dialog_fallback_behavior(self) -> None:
        result = eval_file_paste_dialog_fallback()
        for key in ("missing", "denied"):
            with self.subTest(key=key):
                case = result[key]
                self.assertEqual(case["backdrop"], "block")
                self.assertEqual(case["dialog"], "flex")
                self.assertEqual(case["inputValue"], "")
                self.assertEqual(case["focusCount"], 1)
                self.assertEqual(case["selectCount"], 1)
                self.assertEqual(case["toasts"], ["paste manually"])
                self.assertEqual(case["inserted"], [])
                self.assertEqual(case["focusEditorCount"], 0)
                self.assertEqual(case["prepareCount"], 1)
                self.assertEqual(case["modalSyncCount"], 1)
                self.assertEqual(case["rafCount"], 1)
        self.assertEqual(result["deniedAfterReadonly"]["dialog"], "none")
        self.assertEqual(result["deniedAfterReadonly"]["toasts"], ["paste error: denied"])
        self.assertEqual(result["deniedAfterReadonly"]["focusEditorCount"], 1)
        self.assertEqual(result["deniedAfterReadonly"]["prepareCount"], 0)
        self.assertEqual(result["deniedAfterReadonly"]["modalSyncCount"], 0)
        self.assertEqual(result["direct"]["dialog"], "none")
        self.assertEqual(result["direct"]["inserted"], ["hello"])
        self.assertEqual(result["direct"]["toasts"], ["pasted"])
        self.assertEqual(result["direct"]["focusEditorCount"], 2)
        self.assertTrue(result["direct"]["dirty"])
        self.assertEqual(result["empty"]["dialog"], "none")
        self.assertEqual(result["empty"]["toasts"], ["clipboard empty"])
        self.assertEqual(result["empty"]["focusEditorCount"], 1)
        self.assertEqual(result["dismissed"]["backdrop"], "none")
        self.assertEqual(result["dismissed"]["dialog"], "none")
        self.assertEqual(result["dismissed"]["focusEditorCount"], 1)
        self.assertEqual(result["dismissed"]["modalSyncCount"], 2)

    def test_touch_copy_uses_click_activation_not_press_wrapper(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('codoxearFileViewer.bindFileTouchClick(fileTouchCopyBtn, () => {', source)
        self.assertNotIn('codoxearFileViewer.bindFileTouchPress(fileTouchCopyBtn, () => {', source)


if __name__ == "__main__":
    unittest.main()
