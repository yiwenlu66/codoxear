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
        const ctx = {{
          codoxearFileHelpers: moduleCtx.window.CodoxearFileHelpers,
          activeVideoFallback: {{ token: "video-1", previewUrl: "/preview.mp4", used: false, preparing: false, rel: "clip.mkv" }},
          activeFilePath: "clip.mkv",
          applyCount: 0,
          authLost: false,
          fileStatus: {{ textContent: "" }},
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
        ctx.applyFileMode = () => {{ ctx.applyCount += 1; }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_loadCompatibleVideoPreview = loadCompatibleVideoPreview;\n")}, ctx);
        (async () => {{
          const ok = await ctx.__test_loadCompatibleVideoPreview("video-1", {{ explicit: true }});
          process.stdout.write(JSON.stringify({{
            ok,
            status: ctx.fileStatus.textContent,
            used: ctx.activeVideoFallback.used,
            preparing: ctx.activeVideoFallback.preparing,
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
    clear_end = source.index("function beginFileOpenRequest", clear_start)
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
        ctx.fileSessionSelections = {{
          set(key, value) {{
            const saved = {{
              key,
              path: value.path,
              apiPath: value.apiPath,
              gitPath: value.gitPath,
              line: value.line,
              syncToken: ctx.fileViewerSessionSyncToken,
              editMode: ctx.fileEditMode,
              savePending: ctx.fileSavePending,
              saveToken: ctx.activeFileSaveToken,
            }};
            ctx.savedSelections.push(saved);
            calls.push(["rememberActiveFileSelection", saved]);
          }},
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
          fileSessionSelections: new Map(),
          fileCandidateList: [],
          fileEntryMap: new Map(),
          normalizeLineNumber: (value) => value == null || value === "" ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
          sessionIndex: new Map(),
          listFromFilesField: () => [],
          sessionRelativePath: () => "",
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_target = resolveFileViewerOpenTarget;\n")}, ctx);
        ctx.fileSessionSelections.set("sid-1", {{ path: "remembered.txt", line: "9", gitPath: true, apiPath: "api-remembered" }});
        ctx.fileCandidateList = ["first-key"];
        ctx.fileEntryMap.set("first-key", {{ path: "first.txt", changed: true, gitPath: true, apiPath: "api-first" }});
        const explicit = ctx.__test_target({{ sessionId: "sid-1", explicitPath: "explicit.md", explicitLine: "42" }});
        const preferred = ctx.__test_target({{ sessionId: "sid-1" }});
        ctx.fileSessionSelections.clear();
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
    source = APP_JS.read_text(encoding="utf-8")
    predicate_start = source.index("function currentFileEditorState() {")
    predicate_end = source.index("function syncFileEditorReadOnly()", predicate_start)
    start = source.index("function hideFilePasteDialog({ restoreFocus = false } = {}) {")
    end = source.index("function insertIntoActiveFileEditor(text)", start)
    snippet = source[predicate_start:predicate_end] + "\n" + source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const snippet = {json.dumps(snippet + "\nglobalThis.__test_paste = pasteFromClipboardIntoActiveFile;\nglobalThis.__test_showPaste = showFilePasteDialog;\nglobalThis.__test_hidePaste = hideFilePasteDialog;\n")};
        async function runCase(opts) {{
          const ctx = {{
            window: {{ isSecureContext: Boolean(opts.secure) }},
            navigator: opts.clipboard === "missing" ? {{}} : {{
              clipboard: {{
                readText: async () => {{
                  if (opts.clipboard === "denied") throw new Error("denied");
                  return opts.clipboardText || "";
                }},
              }},
            }},
            fileEditMode: true,
            activeFileEditable: true,
            fileViewMode: "file",
            activeFileKind: "text",
{controller_identity_ctx_js("note.txt", "", False, None)}
            activeFileVersion: "v1",
            activeFileDraft: false,
            fileEditorKind: "file",
            fileDirty: false,
            fileSavePending: false,
            fileSavePendingValue: () => ctx.fileSavePending,
            fileViewerSessionId: "sid-1",
            filePasteBackdrop: {{ style: {{ display: "none" }} }},
            filePasteDialog: {{ style: {{ display: "none" }} }},
            filePasteInput: {{
              value: "stale",
              focusCount: 0,
              selectCount: 0,
              focus() {{ this.focusCount += 1; }},
              select() {{ this.selectCount += 1; }},
            }},
            toastMessages: [],
            inserted: [],
            focusEditorCount: 0,
            prepareCount: 0,
            modalSyncCount: 0,
            rafCount: 0,
            isTextFileKind: (kind) => kind === "text",
            isFileViewerSessionUnavailable: () => false,
            blockUnavailableFileAction: () => false,
            prepareModalOpen: () => {{ ctx.prepareCount += 1; }},
            afterModalVisibilityChanged: () => {{ ctx.modalSyncCount += 1; }},
            requestAnimationFrame: (cb) => {{ ctx.rafCount += 1; cb(); }},
            setToast: (message) => ctx.toastMessages.push(String(message)),
            focusActiveFileCodeEditor: () => {{ ctx.focusEditorCount += 1; }},
            insertIntoActiveFileEditor: (text) => {{ ctx.inserted.push(String(text)); return opts.insertOk !== false; }},
          }};
          vm.createContext(ctx);
          vm.runInContext(snippet, ctx);
          await ctx.__test_paste();
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
          }};
        }}
        (async () => {{
          const missing = await runCase({{ secure: false, clipboard: "missing" }});
          const denied = await runCase({{ secure: true, clipboard: "denied" }});
          const direct = await runCase({{ secure: true, clipboard: "ok", clipboardText: "hello" }});
          const empty = await runCase({{ secure: true, clipboard: "ok", clipboardText: "" }});
          const dismissed = await runCase({{ secure: false, clipboard: "missing", hideAfter: true }});
          process.stdout.write(JSON.stringify({{ missing, denied, direct, empty, dismissed }}));
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
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index('$("#filePasteInsertBtn").onclick = () => {')
    end = source.index('$("#filePasteCancelBtn").onclick', start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const button = {{ onclick: null }};
        const ctx = {{
          button,
          unavailable: true,
          filePasteInput: {{ value: "typed text" }},
          inserted: [],
          hidden: 0,
          toasts: [],
          status: [],
          $: (selector) => {{
            if (selector !== '#filePasteInsertBtn') throw new Error(`unexpected selector ${{selector}}`);
            return button;
          }},
          blockUnavailableFileAction: () => {{
            if (!ctx.unavailable) return false;
            ctx.status.push("Session is no longer available; copy unsaved edits before closing.");
            return true;
          }},
          insertIntoActiveFileEditor: (text) => {{ ctx.inserted.push(String(text)); return true; }},
          hideFilePasteDialog: () => {{ ctx.hidden += 1; ctx.filePasteInput.value = ""; }},
          setToast: (message) => ctx.toasts.push(String(message)),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        ctx.button.onclick();
        const unavailable = {{
          inputValue: ctx.filePasteInput.value,
          inserted: ctx.inserted.slice(),
          hidden: ctx.hidden,
          toasts: ctx.toasts.slice(),
          status: ctx.status.slice(),
        }};
        ctx.unavailable = false;
        ctx.filePasteInput.value = "allowed text";
        ctx.button.onclick();
        const available = {{
          inputValue: ctx.filePasteInput.value,
          inserted: ctx.inserted.slice(),
          hidden: ctx.hidden,
          toasts: ctx.toasts.slice(),
          status: ctx.status.slice(),
        }};
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
    source = APP_JS.read_text(encoding="utf-8")
    predicate_start = source.index("function currentFileEditorState() {")
    predicate_end = source.index("function syncFileEditorReadOnly()", predicate_start)
    handler_start = source.index("function isActiveFileEditorInput(target) {")
    handler_end = source.index("function isFileEditorNativeDeleteEvent(e)", handler_start)
    snippet = source[predicate_start:predicate_end] + "\n" + source[handler_start:handler_end]
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
        const snippet = {json.dumps(snippet + "\nglobalThis.__test_saveShortcut = handleFileEditorSaveShortcut;\n")};
        async function runCase(overrides = {{}}) {{
          const editorNode = {{ contains: (node) => Boolean(node && node.editorInput) }};
          const fileViewer = {{ style: {{ display: overrides.viewerOpen === false ? "none" : "flex" }} }};
          const filePasteDialog = {{ style: {{ display: overrides.nestedDialog ? "flex" : "none" }} }};
          const fileUnsavedDialog = {{ style: {{ display: "none" }} }};
          const ctx = {{
            HTMLElement: FakeElement,
            fileViewer,
            filePasteDialog,
            fileUnsavedDialog,
            modalIsolationTargets: [fileViewer, filePasteDialog, fileUnsavedDialog],
            fileEditMode: overrides.editMode !== false,
            activeFileEditable: overrides.editable !== false,
            fileViewMode: overrides.fileViewMode || "file",
            activeFileKind: overrides.kind || "text",
{controller_identity_ctx_js("", "", False, None)}
            activeFileVersion: overrides.version || "v1",
            activeFileDraft: Boolean(overrides.draft),
            fileEditorKind: overrides.editorKind || "file",
            fileDirty: Boolean(overrides.dirty),
            fileSavePending: Boolean(overrides.pending),
            fileSavePendingValue: () => ctx.fileSavePending,
            fileViewerSessionId: overrides.sessionId === false ? "" : "sid-1",
            unavailable: Boolean(overrides.unavailable),
            saves: [],
            isFileViewerOpen: () => fileViewer.style.display === "flex",
            isModalTargetOpen: (node) => node && node.style && node.style.display === "flex",
            isTextEntryElement: (target) => Boolean(target && target.textEntry),
            getActiveFileCodeEditor: () => ({{ getDomNode: () => editorNode }}),
            isTextFileKind: (kind) => kind === "text" || kind === "markdown",
            isFileViewerSessionUnavailable: () => ctx.unavailable,
            saveActiveFileEdits: async (opts) => {{ ctx.saves.push(opts); return true; }},
          }};
          ctx.fileViewerController.setActiveFileIdentity(overrides.path === false ? "" : "note.txt", {{ gitPath: Boolean(overrides.gitPath), apiPath: overrides.apiPath || "" }});
          vm.createContext(ctx);
          vm.runInContext(snippet, ctx);
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
          const handled = ctx.__test_saveShortcut(event);
          await Promise.resolve();
          return {{ handled, prevented: event.prevented, stopped: event.stopped, saves: ctx.saves }};
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
    source = APP_JS.read_text(encoding="utf-8")
    helper_start = source.index("function isActiveFileEditorInput(target) {")
    helper_end = source.index("function handleFileEditorSaveShortcut(e)", helper_start)
    handler_start = source.index("function handleFileTouchSelectionKeydown(e) {")
    handler_end = source.index('addAppEvent(document, "keydown", handleFileTouchSelectionKeydown, true);', handler_start)
    snippet = source[helper_start:helper_end] + "\n" + source[handler_start:handler_end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        class FakeElement {{
          constructor(opts = {{}}) {{
            this.textEntry = Boolean(opts.textEntry);
            this.editorInput = Boolean(opts.editorInput);
            this.inViewer = opts.inViewer !== false;
            this._inputarea = Boolean(opts.inputarea);
            this.classList = {{ contains: (name) => name === "inputarea" && this._inputarea }};
          }}
          closest(selector) {{
            return selector === "#fileViewer" && this.inViewer ? this : null;
          }}
        }}
        const snippet = {json.dumps(snippet + "\nglobalThis.__test_touchSelectionKeydown = handleFileTouchSelectionKeydown;\n")};
        function runCase(overrides = {{}}) {{
          const editorNode = {{ contains: (node) => Boolean(node && node.editorInput) }};
          const fileViewer = {{ style: {{ display: overrides.viewerOpen === false ? "none" : "flex" }} }};
          const filePasteDialog = {{ style: {{ display: overrides.nestedDialog ? "flex" : "none" }} }};
          const fileUnsavedDialog = {{ style: {{ display: "none" }} }};
          const ctx = {{
            HTMLElement: FakeElement,
            fileViewer,
            filePasteDialog,
            fileUnsavedDialog,
            modalIsolationTargets: [fileViewer, filePasteDialog, fileUnsavedDialog],
            fileTouchSelectMode: overrides.selectMode !== false,
            toolbarActive: overrides.toolbarActive !== false,
            moves: [],
            resetArgs: [],
            isFileViewerOpen: () => fileViewer.style.display === "flex",
            isModalTargetOpen: (node) => node && node.style && node.style.display === "flex",
            isTextEntryElement: (target) => Boolean(target && target.textEntry),
            getActiveFileCodeEditor: () => ({{ getDomNode: () => editorNode }}),
            isFileTouchToolbarActive: () => ctx.toolbarActive,
            resetFileTouchSelectionState: (opts) => ctx.resetArgs.push(opts || {{}}),
            moveFileTouchSelection: (direction) => ctx.moves.push(direction),
          }};
          vm.createContext(ctx);
          vm.runInContext(snippet, ctx);
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
          ctx.__test_touchSelectionKeydown(event);
          return {{ prevented: event.prevented, stopped: event.stopped, moves: ctx.moves, resetArgs: ctx.resetArgs }};
        }}
        (() => {{
          const editorInput = new FakeElement({{ textEntry: true, inputarea: true, editorInput: true, inViewer: true }});
          const otherInput = new FakeElement({{ textEntry: true, inputarea: false, editorInput: false, inViewer: false }});
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
    source = APP_JS.read_text(encoding="utf-8")
    predicate_start = source.index("function currentFileEditorState() {")
    predicate_end = source.index("function syncFileEditorReadOnly()", predicate_start)
    handler_start = source.index("function isActiveFileEditorInput(target) {")
    handler_end = source.index("function isFileEditorNativeDeleteEvent(e)", handler_start)
    snippet = source[predicate_start:predicate_end] + "\n" + source[handler_start:handler_end]
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
        const snippet = {json.dumps(snippet + "\nglobalThis.__test_deleteShortcut = handleFileEditorDeleteKeydown;\n")};
        function runCase(overrides = {{}}) {{
          const editorNode = {{ contains: (node) => Boolean(node && node.editorInput) }};
          const fileViewer = {{ style: {{ display: overrides.viewerOpen === false ? "none" : "flex" }} }};
          const filePasteDialog = {{ style: {{ display: overrides.nestedDialog ? "flex" : "none" }} }};
          const fileUnsavedDialog = {{ style: {{ display: "none" }} }};
          const ctx = {{
            HTMLElement: FakeElement,
            Date: {{ now: () => 123456 }},
            fileViewer,
            filePasteDialog,
            fileUnsavedDialog,
            modalIsolationTargets: [fileViewer, filePasteDialog, fileUnsavedDialog],
            fileEditMode: overrides.editMode !== false,
            activeFileEditable: overrides.editable !== false,
            fileViewMode: overrides.fileViewMode || "file",
            activeFileKind: overrides.kind || "text",
{controller_identity_ctx_js("", "", False, None)}
            activeFileVersion: overrides.version || "v1",
            activeFileDraft: Boolean(overrides.draft),
            fileEditorKind: overrides.editorKind || "file",
            fileDirty: Boolean(overrides.dirty),
            fileSavePending: Boolean(overrides.pending),
            fileSavePendingValue: () => ctx.fileSavePending,
            fileViewerSessionId: overrides.sessionId === false ? "" : "sid-1",
            unavailable: Boolean(overrides.unavailable),
            fileTouchSelectMode: overrides.selectMode !== false,
            fileTouchDeleteNativeSuppressUntil: 0,
            triggers: [],
            focusCount: 0,
            resetCount: 0,
            toasts: [],
            isFileViewerOpen: () => fileViewer.style.display === "flex",
            isModalTargetOpen: (node) => node && node.style && node.style.display === "flex",
            isTextEntryElement: (target) => Boolean(target && target.textEntry),
            getActiveFileCodeEditor: () => ({{
              getDomNode: () => editorNode,
              trigger: (source, command, payload) => ctx.triggers.push({{ source, command, payload }}),
            }}),
            isTextFileKind: (kind) => kind === "text" || kind === "markdown",
            isFileViewerSessionUnavailable: () => ctx.unavailable,
            fileEditorDeleteCommandForKey: (key) => key === "backspace" ? "deleteLeft" : key === "delete" ? "deleteRight" : "",
            focusActiveFileCodeEditor: () => {{ ctx.focusCount += 1; }},
            resetFileTouchSelectionState: () => {{ ctx.resetCount += 1; }},
            setToast: (message) => ctx.toasts.push(message),
          }};
          ctx.fileViewerController.setActiveFileIdentity(overrides.path === false ? "" : "note.txt", {{ gitPath: Boolean(overrides.gitPath), apiPath: overrides.apiPath || "" }});
          vm.createContext(ctx);
          vm.runInContext(snippet, ctx);
          const event = {{
            key: overrides.key || "Backspace",
            ctrlKey: Boolean(overrides.ctrl),
            metaKey: Boolean(overrides.meta),
            altKey: Boolean(overrides.alt),
            isComposing: Boolean(overrides.composing),
            defaultPrevented: Boolean(overrides.defaultPrevented),
            target: overrides.target || new FakeElement({{ textEntry: true, inputarea: true, editorInput: true }}),
            prevented: 0,
            stopped: 0,
            preventDefault() {{ this.prevented += 1; }},
            stopPropagation() {{ this.stopped += 1; }},
          }};
          const handled = ctx.__test_deleteShortcut(event);
          return {{ handled, prevented: event.prevented, stopped: event.stopped, triggers: ctx.triggers, focusCount: ctx.focusCount, resetCount: ctx.resetCount, suppressUntil: ctx.fileTouchDeleteNativeSuppressUntil, toasts: ctx.toasts }};
        }}
        (() => {{
          const editorInput = new FakeElement({{ textEntry: true, inputarea: true, editorInput: true }});
          const otherInput = new FakeElement({{ textEntry: true, inputarea: false, editorInput: false }});
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
        const fileApiPathCalls = [];
        let disposeCalls = 0;
        const state = {{ sessionId: "sid-1" }};
        const ctx = {{ window: {{}}, AbortController }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(viewer_source)}, ctx);
        const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
          el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children }}),
          fileStatus: {{ replaceChildren() {{}} }},
          currentSessionId: () => state.sessionId,
          normalizeLineNumber: (value) => value == null ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          fileApiPathForPath: (path, apiPath = "") => {{
            fileApiPathCalls.push([String(path), String(apiPath || "")]);
            return apiPath ? `kept:${{apiPath}}` : `derived:${{path}}`;
          }},
          isUnavailable: () => false,
          confirmReload: () => true,
          openFilePath: async () => true,
          api: async () => ({{}}),
          focusEditor: () => null,
          disposeOpenRender: () => {{ disposeCalls += 1; }},
          currentFileViewMode: () => "file",
          activeFileEntry: () => null,
          fileCandidateGitStateFresh: () => false,
          isMarkdownPreviewable: () => true,
          resetActiveFileBufferState: () => {{}},
          updateFileTouchToolbar: () => {{}},
          setFileViewMode: () => {{}},
          applyActiveFileTextState: () => {{}},
          renderMonacoFile: async () => true,
          setFileEditMode: () => {{}},
          currentActiveFileDraft: () => false,
          currentActiveFileVersion: () => "",
          getFileEditorText: () => "",
          syncFileEditorReadOnly: () => {{}},
          applyFileMode: () => {{}},
          rememberOpenedFile: () => {{}},
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
        result.fileApiPathCalls = fileApiPathCalls;
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
    start = source.index("function currentFileSessionId() {")
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
    source = APP_JS.read_text(encoding="utf-8")
    guard_start = source.index("async function openFilePathWithGuard")
    guard_end = source.index("async function openDraftFilePathWithGuard", guard_start)
    resolved_start = source.index("async function openFilePathWithResolvedMode")
    resolved_end = source.index("async function openDraftFilePath", resolved_start)
    snippet = source[guard_start:guard_end] + "\n" + source[resolved_start:resolved_end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        let resolveMode;
        let current = true;
        const calls = [];
        const ctx = {{
          maybeHandleUnsavedFileChanges: async () => true,
          setFilePath: (...args) => calls.push(["setFilePath", ...args]),
          setFileViewMode: (...args) => calls.push(["setFileViewMode", ...args]),
          renderFilePickerMenu: () => calls.push("renderFilePickerMenu"),
          openFilePath: async (...args) => calls.push(["openFilePath", ...args]),
          currentFileSessionId: () => "sid-b",
          blockUnavailableFileAction: () => false,
          isGitFileCandidatePath: () => true,
          fileEntryForPath: () => null,
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          resolveFileOpenMode: () => new Promise((resolve) => {{ resolveMode = resolve; }}),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test = { openFilePathWithResolvedMode };\n")}, ctx);
        const promise = ctx.__test.openFilePathWithResolvedMode("b-file.txt", {{ isCurrent: () => current }});
        current = false;
        resolveMode("file");
        promise.then((result) => {{
          process.stdout.write(JSON.stringify({{ result, calls }}));
        }}).catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)




def eval_open_file_guard_mode_validation() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    helper_start = source.index("function normalizeExplicitFileOpenMode(")
    helper_end = source.index("function resolveFileOpenViewMode", helper_start)
    guard_start = source.index("async function openFilePathWithGuard")
    guard_end = source.index("async function openDraftFilePathWithGuard", guard_start)
    snippet = source[helper_start:helper_end] + "\n" + source[guard_start:guard_end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const calls = [];
        const ctx = {{
          blockUnavailableFileAction: () => false,
          fileViewerController: {{
            normalizeExplicitFileOpenMode: (requestedMode) => {{
              if (requestedMode === null || requestedMode === undefined || requestedMode === "") return null;
              if (requestedMode === "preview" || requestedMode === "file" || requestedMode === "diff") return requestedMode;
              throw new Error("invalid file open mode");
            }},
          }},
          currentFileSessionId: () => "sid-1",
          isFileViewerSessionUnavailable: () => false,
          maybeHandleUnsavedFileChanges: async () => true,
          setFilePath: (...args) => calls.push(["setFilePath", ...args]),
          setFileViewMode: (...args) => calls.push(["setFileViewMode", ...args]),
          renderFilePickerMenu: () => calls.push(["renderFilePickerMenu"]),
          openFilePath: async (...args) => calls.push(["openFilePath", ...args]),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_guard = openFilePathWithGuard;\n")}, ctx);
        (async () => {{
          let invalidMessage = "";
          try {{ await ctx.__test_guard("x.txt", {{ mode: "bogus" }}); }} catch (err) {{ invalidMessage = err && err.message || ""; }}
          const invalidCalls = calls.slice();
          calls.length = 0;
          const validResult = await ctx.__test_guard("x.txt", {{ line: 4, mode: "diff", gitPath: true, apiPath: "tok" }});
          const validCalls = calls.slice();
          process.stdout.write(JSON.stringify({{ invalidMessage, invalidCalls, validResult, validCalls }}));
        }})().catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_open_file_path_mode_ownership() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function normalizeExplicitFileOpenMode(")
    end = source.index("fileBtn.onclick", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const calls = [];
        const ctx = {{
          fileViewerSessionId: "sid-1",
          fileStatus: {{ textContent: "" }},
          fileViewMode: "file",
          fileCandidateGitStateFresh: false,
          activeEntryValue: null,
          activeEntryCalls: 0,
          blockUnavailableFileAction: () => false,
          fileViewerController: {{
            normalizeExplicitFileOpenMode: (requestedMode) => {{
              if (requestedMode === null || requestedMode === undefined || requestedMode === "") return null;
              if (requestedMode === "preview" || requestedMode === "file" || requestedMode === "diff") return requestedMode;
              throw new Error("invalid file open mode");
            }},
            resolveFileOpenViewMode: (request, rel, requestedMode = null) => {{
              const openMode = ctx.fileViewerController.normalizeExplicitFileOpenMode(requestedMode);
              if (openMode) return openMode;
              const activeEntry = ctx.activeFileEntry();
              const canUseDiffView = request.gitPath && ctx.fileCandidateGitStateFresh && Boolean(activeEntry && activeEntry.changed);
              return ctx.fileViewMode === "preview" && !ctx.isMarkdownPreviewable(rel) ? "file" : ctx.fileViewMode === "diff" && !canUseDiffView ? "file" : ctx.fileViewMode;
            }},
            fetchFileOpenResult: async (request, rel, viewMode) => {{
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
            renderFileOpenError: (request, error) => {{
              if (error && error.name === "AbortError") return false;
              if (!ctx.isCurrentFileOpenRequest(request)) return false;
              ctx.resetActiveFileBufferState();
              ctx.fileStatus.textContent = `error: ${{error && error.message ? error.message : "unknown error"}}`;
              ctx.updateFileTouchToolbar();
              return false;
            }},
          }},
          startFileOpenRequest: (path, opts = {{}}) => {{
            const request = {{
              sessionId: "sid-1",
              signal: {{ aborted: false }},
              gitPath: Boolean(opts.gitPath),
              apiPath: typeof opts.apiPath === "string" ? opts.apiPath : "",
              line: opts.line ?? null,
            }};
            return {{ request, path: String(path || ""), done: () => calls.push(["done"]) }};
          }},
          resetFileViewerPanel: () => calls.push(["resetFileViewerPanel"]),
          activeFileEntry: () => {{ ctx.activeEntryCalls += 1; return ctx.activeEntryValue; }},
          isMarkdownPreviewable: (rel) => {{ calls.push(["isMarkdownPreviewable", rel]); return false; }},
          setFileViewMode: (mode) => {{ ctx.fileViewMode = mode; calls.push(["setFileViewMode", mode]); }},
          api: async (url, options = {{}}) => {{
            calls.push(["api", url, Boolean(options.signal)]);
            if (url.includes("/git/file_versions")) return {{ base_text: "old", current_text: "new", base_exists: true, current_exists: true, abs_path: "/abs/diff" }};
            return {{ kind: "text", text: "body", path: "/abs/read" }};
          }},
          isCurrentFileOpenRequest: () => true,
          applyFileLoadResult: async (...args) => {{ calls.push(["applyFileLoadResult", args[0], args[1].kind, args[3].viewMode]); return true; }},
          finalizeFileOpenSuccess: (...args) => {{ calls.push(["finalizeFileOpenSuccess", ...args]); return true; }},
          resetActiveFileBufferState: () => calls.push(["resetActiveFileBufferState"]),
          updateFileTouchToolbar: () => calls.push(["updateFileTouchToolbar"]),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_open_mode = { openFilePath, resolveFileOpenViewMode };\n")}, ctx);
        (async () => {{
          const explicitResult = await ctx.__test_open_mode.openFilePath("stale.txt", {{ line: 5, gitPath: true, apiPath: "tok", mode: "diff" }});
          const explicit = {{ result: explicitResult, calls: calls.slice(), activeEntryCalls: ctx.activeEntryCalls, fileViewMode: ctx.fileViewMode }};
          calls.length = 0;
          ctx.fileStatus.textContent = "";
          ctx.fileViewMode = "diff";
          ctx.fileCandidateGitStateFresh = true;
          ctx.activeEntryValue = {{ changed: false }};
          ctx.activeEntryCalls = 0;
          const fallbackResult = await ctx.__test_open_mode.openFilePath("stale.txt", {{ line: 5, gitPath: true, apiPath: "tok" }});
          const fallback = {{ result: fallbackResult, calls: calls.slice(), activeEntryCalls: ctx.activeEntryCalls, fileViewMode: ctx.fileViewMode }};
          let invalidMessage = "";
          try {{ ctx.__test_open_mode.resolveFileOpenViewMode({{ gitPath: false }}, "x.txt", "bogus"); }} catch (err) {{ invalidMessage = err && err.message || ""; }}
          process.stdout.write(JSON.stringify({{ explicit, fallback, invalidMessage }}));
        }})().catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_active_file_load_state_writers() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function applyActiveFileTextState(")
    end = source.index("function getFileEditorText()", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          activeFileKind: "",
          activeFileText: "",
          activeFileEditable: false,
          activeFileVersion: "",
          activeFileDraft: false,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_load_state = { applyActiveFileTextState, applyActiveFileDiffState, applyActiveFileNonTextState };\n")}, ctx);
        function stale() {{
          ctx.activeFileKind = "stale";
          ctx.activeFileText = "stale text";
          ctx.activeFileEditable = true;
          ctx.activeFileVersion = "stale-version";
          ctx.activeFileDraft = true;
        }}
        function state() {{
          return {{
            kind: ctx.activeFileKind,
            text: ctx.activeFileText,
            editable: ctx.activeFileEditable,
            version: ctx.activeFileVersion,
            draft: ctx.activeFileDraft,
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
        const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
          el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children }}),
          fileStatus,
          currentSessionId: () => state.sessionId,
          normalizeLineNumber: (value) => value == null ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          fileApiPathForPath: (_path, existing = "") => existing || "derived-token",
          isUnavailable: () => state.unavailable,
          confirmReload: () => true,
          openFilePath: async () => true,
          api: async () => ({{ kind: "text", text: "body", path: "/abs/read" }}),
          focusEditor: () => null,
          disposeOpenRender: () => calls.push(["disposeOpenRender"]),
          currentFileViewMode: () => "file",
          activeFileEntry: () => null,
          fileCandidateGitStateFresh: () => false,
          isMarkdownPreviewable: () => true,
          resetActiveFileBufferState: () => calls.push(["resetActiveFileBufferState"]),
          updateFileTouchToolbar: () => calls.push(["updateFileTouchToolbar"]),
          setFileViewMode: () => {{}},
          applyActiveFileTextState: () => {{}},
          renderMonacoFile: async () => true,
          setFileEditMode: () => {{}},
          currentActiveFileDraft: () => state.draft,
          currentActiveFileVersion: () => state.version,
          getFileEditorText: () => {{ calls.push(["getFileEditorText"]); return state.text; }},
          syncFileEditorReadOnly: () => calls.push(["syncFileEditorReadOnly"]),
          applyFileMode: () => {{}},
          rememberOpenedFile: () => {{}},
          rememberActiveFileSelection: () => {{}},
          updateFileEditButton: () => calls.push(["updateFileEditButton"]),
          renderFilePickerMenu: () => {{}},
        }});
        controller.setActiveFileIdentity("src/app.py", {{ line: 42, gitPath: true, apiPath: "token-1" }});
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
        state.unavailable = true;
        result.currentUnavailable = controller.isCurrentActiveFileSaveRequest(save);
        state.unavailable = false;
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
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function applyActiveFileSaveSuccess(")
    end = source.index("async function saveActiveFileEdits", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          activeFileText: "old text",
          activeFileVersion: "v0",
          activeFileEditable: false,
          activeFileDraft: true,
{controller_identity_ctx_js("new.py", "old-token", True, 42)}
          fileDirty: true,
          fileEditMode: true,
          fileStatus: {{ textContent: "" }},
          calls: [],
          applyFileMode: () => ctx.calls.push(["applyFileMode"]),
          setFileDirty: (value) => {{ ctx.fileDirty = Boolean(value); ctx.calls.push(["setFileDirty", Boolean(value)]); }},
          setFileEditMode: (value) => {{ ctx.fileEditMode = Boolean(value); ctx.calls.push(["setFileEditMode", Boolean(value)]); }},
          fmtBytes: (value) => `${{value}}B`,
          rememberOpenedFile: (...args) => ctx.calls.push(["rememberOpenedFile", ...args]),
          renderFilePickerMenu: () => ctx.calls.push(["renderFilePickerMenu"]),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_save_success = applyActiveFileSaveSuccess;\n")}, ctx);
        function state() {{
          return {{
            text: ctx.activeFileText,
            version: ctx.activeFileVersion,
            editable: ctx.activeFileEditable,
            draft: ctx.activeFileDraft,
            gitPath: ctx.activeFileGitPathValue(),
            apiPath: ctx.activeFileApiPathValue(),
            dirty: ctx.fileDirty,
            editMode: ctx.fileEditMode,
            status: ctx.fileStatus.textContent,
            calls: ctx.calls.slice(),
          }};
        }}
        const draftSave = {{ path: "new.py", text: "NEW", draft: true }};
        const draftOk = ctx.__test_save_success(draftSave, {{ version: "v2", editable: true, size: 3, path: "/abs/new.py" }}, {{ exitEditMode: true }});
        const draft = {{ ok: draftOk, state: state() }};
        ctx.activeFileText = "old again";
        ctx.activeFileVersion = "v0";
        ctx.activeFileEditable = true;
        ctx.activeFileDraft = false;
        ctx.fileViewerController.setActiveFileIdentity("existing.py", {{ line: 42, gitPath: true, apiPath: "keep-token" }});
        ctx.fileDirty = true;
        ctx.fileEditMode = true;
        ctx.fileStatus.textContent = "";
        ctx.calls = [];
        const nondraftSave = {{ path: "existing.py", text: "BODY", draft: false }};
        const nondraftOk = ctx.__test_save_success(nondraftSave, {{}}, {{ exitEditMode: false }});
        const nondraft = {{ ok: nondraftOk, state: state() }};
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
          currentSessionId: () => state.sessionId,
          normalizeLineNumber: (value) => value == null ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          fileApiPathForPath: (_path, existing = "") => existing || "derived-token",
          isUnavailable: () => false,
          confirmReload: () => true,
          openFilePath: async () => true,
          api: async () => ({{ kind: "text", text: "body", path: "/abs/read" }}),
          focusEditor: () => null,
          disposeOpenRender: () => calls.push(["disposeOpenRender"]),
          currentFileViewMode: () => state.viewMode,
          activeFileEntry: () => null,
          fileCandidateGitStateFresh: () => false,
          isMarkdownPreviewable: () => true,
          resetActiveFileBufferState: () => calls.push(["resetActiveFileBufferState"]),
          updateFileTouchToolbar: () => calls.push(["updateFileTouchToolbar"]),
          setFileViewMode: (mode) => {{ calls.push(["setFileViewMode", mode]); state.viewMode = mode; }},
          applyActiveFileTextState: (nextState) => calls.push(["applyActiveFileTextState", nextState]),
          renderMonacoFile: async (...args) => {{ calls.push(["renderMonacoFile", ...args.slice(0, 4)]); if (state.staleAfterRender) state.sessionId = "sid-2"; return state.renderOk; }},
          setFileEditMode: (mode) => calls.push(["setFileEditMode", mode]),
          currentActiveFileDraft: () => false,
          currentActiveFileVersion: () => "",
          getFileEditorText: () => "",
          syncFileEditorReadOnly: () => calls.push(["syncFileEditorReadOnly"]),
          applyFileMode: () => calls.push(["applyFileMode"]),
          rememberOpenedFile: (...args) => calls.push(["rememberOpenedFile", ...args]),
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
          process.stdout.write(JSON.stringify({{ success, failedRender, staleResult }}));
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
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function finalizeFileOpenSuccess(")
    end = source.index("async function openFilePath", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          calls: [],
          fileViewerController: {{
            finalizeFileOpenSuccess: (rel, absPath = null) => {{
              ctx.calls.push(["applyFileMode"]);
              ctx.calls.push(["rememberOpenedFile", rel, absPath]);
              ctx.calls.push(["rememberActiveFileSelection"]);
              ctx.calls.push(["updateFileEditButton"]);
              ctx.calls.push(["renderFilePickerMenu"]);
              return true;
            }},
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_finalize_open = finalizeFileOpenSuccess;\n")}, ctx);
        const ok = ctx.__test_finalize_open("src/app.py", "/abs/src/app.py");
        process.stdout.write(JSON.stringify({{ ok, calls: ctx.calls }}));
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


def eval_file_load_result_dispatcher() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    state_start = source.index("function applyActiveFileTextState(")
    state_end = source.index("function getFileEditorText()", state_start)
    surface_start = source.index("function setFileRenderSurface(surface)")
    surface_end = source.index("function resetFileViewerPanel()", surface_start)
    load_start = source.index("async function applyFileLoadResult(")
    load_end = source.index("async function openFilePath", load_start)
    snippet = source[state_start:state_end] + "\n" + source[surface_start:surface_end] + "\n" + source[load_start:load_end]
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
            "inputValue": "typed text",
            "inserted": [],
            "hidden": 0,
            "toasts": [],
            "status": ["Session is no longer available; copy unsaved edits before closing."],
        })
        self.assertEqual(result["available"], {
            "inputValue": "",
            "inserted": ["allowed text"],
            "hidden": 1,
            "toasts": ["text inserted"],
            "status": ["Session is no longer available; copy unsaved edits before closing."],
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
        self.assertEqual(result["calls"], [])

    def test_open_file_path_guard_validates_mode_before_mutating_state(self) -> None:
        result = eval_open_file_guard_mode_validation()
        self.assertEqual(result["invalidMessage"], "invalid file open mode")
        self.assertEqual(result["invalidCalls"], [])
        self.assertTrue(result["validResult"])
        self.assertEqual(result["validCalls"], [
            ["setFilePath", "x.txt", {"line": 4, "gitPath": True, "apiPath": "tok"}],
            ["setFileViewMode", "diff"],
            ["renderFilePickerMenu"],
            ["openFilePath", "x.txt", {"line": 4, "gitPath": True, "apiPath": "tok", "mode": "diff"}],
        ])

    def test_open_file_path_mode_ownership_respects_resolved_mode(self) -> None:
        result = eval_open_file_path_mode_ownership()
        explicit = result["explicit"]
        self.assertTrue(explicit["result"])
        self.assertEqual(explicit["activeEntryCalls"], 0)
        self.assertEqual(explicit["fileViewMode"], "diff")
        self.assertIn(["setFileViewMode", "diff"], explicit["calls"])
        self.assertTrue(any(call[0] == "api" and "/git/file_versions?path=stale.txt&path_token=tok" in call[1] for call in explicit["calls"]))
        self.assertFalse(any(call[0] == "api" and "/file/read" in call[1] for call in explicit["calls"]))
        self.assertIn(["applyFileLoadResult", "stale.txt", "diff", "diff"], explicit["calls"])
        self.assertIn(["finalizeFileOpenSuccess", "stale.txt", "/abs/diff"], explicit["calls"])

        fallback = result["fallback"]
        self.assertTrue(fallback["result"])
        self.assertEqual(fallback["activeEntryCalls"], 1)
        self.assertEqual(fallback["fileViewMode"], "file")
        self.assertIn(["setFileViewMode", "file"], fallback["calls"])
        self.assertTrue(any(call[0] == "api" and "/file/read?path=stale.txt&path_token=tok&git_path=1" in call[1] for call in fallback["calls"]))
        self.assertFalse(any(call[0] == "api" and "/git/file_versions" in call[1] for call in fallback["calls"]))
        self.assertIn(["applyFileLoadResult", "stale.txt", "text", "file"], fallback["calls"])
        self.assertIn(["finalizeFileOpenSuccess", "stale.txt", "/abs/read"], fallback["calls"])
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
        self.assertIn("let fileViewerSessionSyncToken = 0;", source)
        self.assertIn("let fileCandidateRequestSeq = 0;", source)
        self.assertIn("function isFileViewerSelectionCurrent(sessionId, token = null)", source)
        self.assertIn("function isFileViewerSessionCurrent(sessionId, token = null)", source)
        ensure_start = source.index("async function ensureCurrentFileViewerSession()")
        ensure_end = source.index("function extToEditorLang", ensure_start)
        ensure_block = source[ensure_start:ensure_end]
        self.assertIn("const syncToken = ++fileViewerSessionSyncToken;", ensure_block)
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
        self.assertIn("const requestSeq = ++fileCandidateRequestSeq;", refresh_block)
        self.assertIn("const current = () => requestSeq === fileCandidateRequestSeq", refresh_block)
        self.assertIn("if (!current()) return;", refresh_block)
        show_start = source.index("async function showFileViewer")
        show_end = source.index("function hideFileViewer", show_start)
        show_block = source[show_start:show_end]
        self.assertIn("const syncToken = ++fileViewerSessionSyncToken;", show_block)
        self.assertIn("await refreshFileCandidates({ sessionId: sid, syncToken });", show_block)
        self.assertIn("if (!isFileViewerSessionCurrent(sid, syncToken)) return;", show_block)
        self.assertIn("const target = resolveFileViewerOpenTarget({ sessionId: sid, explicitPath, explicitLine: line });", show_block)
        self.assertIn("void openFilePathWithResolvedMode(target.path, { line: target.line, changed: target.changed, gitPath: target.gitPath, apiPath: target.apiPath, isCurrent: () => isFileViewerSessionCurrent(sid, syncToken) })", show_block)
        self.assertIn("renderEmptyFileViewerTarget();", show_block)
        self.assertNotIn("const preferredGitPath = explicitPath ? false : Boolean(preferredSelection.gitPath);", show_block)
        self.assertNotIn("const first = firstKey ? fileEntryMap.get(firstKey) : null;", show_block)
        self.assertIn("fileViewerSessionSyncToken += 1;\n          cancelPendingFileOpen();", source)
        open_start = source.index("async function openSession(sessionId")
        open_end = source.index("async function pollMessages", open_start)
        open_block = source[open_start:open_end]
        self.assertIn("const fileViewerSyncStarted = Boolean(isFileViewerOpen() && !fileDirty);", open_block)
        self.assertIn("void ensureCurrentFileViewerSession().catch((e) => console.error(\"file viewer session sync failed after selection\", e));", open_block)
        self.assertLess(open_block.index("const fileViewerSyncStarted"), open_block.index("messages/tail"))
        self.assertIn("if (isFileViewerOpen() && !fileDirty && !fileViewerSyncStarted) {", open_block)
        self.assertIn("void ensureCurrentFileViewerSession();", open_block)
        self.assertIn("void refreshFileCandidates({ sessionId }).catch((e) => console.error(\"file candidates refresh failed after transcript load\", e));", open_block)
        resolved_start = source.index("async function openFilePathWithResolvedMode")
        resolved_end = source.index("async function openDraftFilePath", resolved_start)
        resolved_block = source[resolved_start:resolved_end]
        self.assertIn("isCurrent = null", resolved_block)
        self.assertIn("const sessionAtStart = currentFileSessionId();", resolved_block)
        self.assertIn("const currentGuard = typeof isCurrent === \"function\" ? isCurrent : () => currentFileSessionId() === sessionAtStart && !isFileViewerSessionUnavailable();", resolved_block)
        self.assertIn("if (!currentGuard()) return false;", resolved_block)
        self.assertIn("try {", resolved_block)
        self.assertIn("const useGitPath = gitPath === null || gitPath === undefined ? isGitFileCandidatePath(path, changed, null, token) : Boolean(gitPath);", resolved_block)
        self.assertIn("mode = await resolveFileOpenMode(path, { changed, gitPath: useGitPath, apiPath: requestApiPath });", resolved_block)
        self.assertIn("if (blockUnavailableFileAction()) return false;", resolved_block)
        self.assertIn("return await openFilePathWithGuard(path, { line, mode, isCurrent: currentGuard, gitPath: useGitPath, apiPath: requestApiPath });", resolved_block)
        guard_start = source.index("async function openFilePathWithGuard")
        guard_end = source.index("async function openFilePathWithResolvedMode", guard_start)
        guard_block = source[guard_start:guard_end]
        self.assertIn("gitPath = false", guard_block)
        self.assertIn("isCurrent = null", guard_block)
        self.assertIn("const sessionAtStart = currentFileSessionId();", guard_block)
        self.assertIn("const currentGuard = typeof isCurrent === \"function\" ? isCurrent : () => currentFileSessionId() === sessionAtStart && !isFileViewerSessionUnavailable();", guard_block)
        self.assertIn("if (!currentGuard()) return false;", guard_block)
        self.assertIn("const openMode = normalizeExplicitFileOpenMode(mode);", guard_block)
        self.assertIn("if (openMode) setFileViewMode(openMode);", guard_block)
        self.assertIn("await openFilePath(path, { line, gitPath, apiPath, mode: openMode });", guard_block)
        self.assertIn("return Boolean(currentGuard());", guard_block)
        self.assertIn("const diffable = canToggleMode && activeFileGitPathValue() && fileCandidateGitStateFresh && Boolean(entry && entry.changed) && isDiffableFileKind(activeFileKind);", source)
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("function normalizeExplicitFileOpenMode(requestedMode)", source)
        self.assertIn("function resolveFileOpenViewMode(request, rel, requestedMode = null)", source)
        self.assertIn("function normalizeExplicitFileOpenMode(requestedMode)", viewer_source)
        self.assertIn("function resolveFileOpenViewMode(request, rel, requestedMode = null)", viewer_source)
        self.assertIn("async function fetchFileOpenResult(request, rel, viewMode)", viewer_source)
        self.assertIn("const canUseDiffView = request && request.gitPath && fileCandidateGitStateFresh() && Boolean(entry && entry.changed);", viewer_source)
        self.assertIn('viewMode === "diff" && !canUseDiffView ? "file"', viewer_source)
        self.assertIn("/git/file_versions?path=${encodeURIComponent(rel)}${pathTokenQuery}", viewer_source)
        self.assertIn("/file/read?path=${encodeURIComponent(rel)}${pathTokenQuery}${gitPathQuery}", viewer_source)
        self.assertIn("git_path: save.gitPath", viewer_source)
        self.assertIn("async function openFilePath(nextPath = null, { line = undefined, gitPath = undefined, apiPath = undefined, mode = null } = {})", source)
        self.assertIn("const viewMode = resolveFileOpenViewMode(request, rel, mode);", source)
        self.assertIn("const openResult = await fileViewerController.fetchFileOpenResult(request, rel, viewMode);", source)
        self.assertNotIn('const gitPathQuery = request.gitPath ? "&git_path=1" : "";', source)
        self.assertIn("if (gitPath) {\n              body.git_path = true;", source)
        self.assertIn("startFileOpenRequest(path, { line, gitPath: false })", source)
        self.assertIn("setFilePath(rel, { line: null, gitPath: false })", source)
        self.assertIn("if (save.draft) {\n            fileViewerController.setActiveFileIdentity(save.path", source)

    def test_file_viewer_handles_selected_session_removal(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("let fileViewerUnavailableSessionId = \"\";", source)
        self.assertIn("function isFileViewerSessionUnavailable()", source)
        self.assertIn("function blockUnavailableFileAction()", source)
        self.assertIn("function disableFileViewerForUnavailableSession(sid)", source)
        self.assertIn("function handleFileViewerSessionUnavailable(sessionId)", source)
        self.assertEqual(source.count("disableFileViewerForUnavailableSession("), 2)
        transition_start = source.index("function disableFileViewerForUnavailableSession(sid)")
        helper_start = source.index("function handleFileViewerSessionUnavailable(sessionId)", transition_start)
        helper_end = source.index("async function openFilePath", helper_start)
        transition_block = source[transition_start:helper_start]
        helper_block = source[helper_start:helper_end]
        self.assertIn("if (fileViewerSessionId && fileViewerSessionId !== sid) return;", helper_block)
        self.assertIn("if (!fileDirty) {\n            hideFileViewer();\n            return;\n          }", helper_block)
        self.assertIn("disableFileViewerForUnavailableSession(sid);", helper_block)
        self.assertNotIn("fileViewerSessionSyncToken += 1;", helper_block)
        self.assertIn("rememberActiveFileSelection(sid);", transition_block)
        self.assertIn("fileViewerSessionSyncToken += 1;", transition_block)
        self.assertIn("fileViewerUnavailableSessionId = sid;", transition_block)
        self.assertIn("clearActiveFileSaveState();", transition_block)
        self.assertIn("fileEditMode = false;", transition_block)
        self.assertIn('hideFileUnsavedDialog("cancel");', transition_block)
        self.assertIn("cancelPendingFileOpen();", transition_block)
        self.assertIn("resetFileSearchState();", transition_block)
        self.assertIn("Session is no longer available; copy unsaved edits before closing.", transition_block)
        self.assertIn("syncFileEditorReadOnly();", transition_block)
        self.assertIn("updateFileEditButton();", transition_block)
        self.assertLess(transition_block.index("rememberActiveFileSelection(sid);"), transition_block.index("fileViewerSessionSyncToken += 1;"))
        self.assertIn("fileViewerUnavailableSessionId = \"\";", source)
        self.assertIn("unavailable: isFileViewerSessionUnavailable(),", source)
        self.assertIn("const unavailable = Boolean(state.unavailable);", source)
        self.assertIn("if (blockUnavailableFileAction()) return false;", source)
        self.assertIn("function renderEmptyFileViewerTarget({ updateTouchToolbar = false } = {})", source)
        self.assertEqual(source.count("resetFileViewerPanel();"), 5)
        hide_start = source.index("function hideFileViewer()")
        hide_end = source.index("function handleFileViewerSessionUnavailable", hide_start)
        hide_block = source[hide_start:hide_end]
        self.assertIn("rememberActiveFileSelection();", hide_block)
        self.assertIn("clearActiveFileIdentity();", hide_block)
        self.assertNotIn("activeFileLine = null;", hide_block)
        self.assertLess(hide_block.index("rememberActiveFileSelection();"), hide_block.index("clearActiveFileIdentity();"))
        self.assertLess(hide_block.index("closeFilePickerMenu({ restoreInput: true });"), hide_block.index("clearActiveFileIdentity();"))
        open_primitive_start = source.index("async function openFilePath(nextPath")
        open_primitive_end = source.index("fileBtn.onclick", open_primitive_start)
        open_primitive_block = source[open_primitive_start:open_primitive_end]
        self.assertIn("if (blockUnavailableFileAction()) return false;", open_primitive_block)
        self.assertIn("fileStatus.textContent = \"Loading...\";\n          resetFileViewerPanel();\n          try {", open_primitive_block)
        self.assertIn("const viewMode = resolveFileOpenViewMode(request, rel, mode);", open_primitive_block)
        self.assertNotIn("const activeEntry = activeFileEntry();", open_primitive_block)
        self.assertNotIn("disposeFileEditor();\n          resetActiveFileBufferState();\n          fileImage.removeAttribute", open_primitive_block)
        file_picker_source = APP_FILE_PICKER_JS.read_text(encoding="utf-8")
        self.assertIn("if (blocked()) return [];", file_picker_source)
        draft_guard_start = source.index("async function openDraftFilePathWithGuard")
        draft_guard_end = source.index("async function setFileViewModeWithGuard", draft_guard_start)
        draft_guard_block = source[draft_guard_start:draft_guard_end]
        self.assertGreaterEqual(draft_guard_block.count("if (blockUnavailableFileAction()) return false;"), 4)
        draft_start = source.index("async function openDraftFilePath(path")
        draft_end = source.index("function cloneFileCandidateEntry", draft_start)
        draft_block = source[draft_start:draft_end]
        self.assertIn("if (blockUnavailableFileAction()) return;", draft_block)
        self.assertIn("fileStatus.textContent = \"Preparing new file...\";\n          resetFileViewerPanel();\n          try {", draft_block)
        self.assertNotIn("disposeFileEditor();\n          resetActiveFileBufferState();\n          fileImage.removeAttribute", draft_block)
        self.assertIn("if (blockUnavailableFileAction()) return;", source)
        self.assertIn("if (blockUnavailableFileAction()) return;\n            if (!text)", source)
        self.assertIn("function insertIntoActiveFileEditor(text)", source)
        self.assertIn("function positionAfterInsertedText(start, text)", source)
        self.assertIn("return codoxearFileHelpers.positionAfterInsertedText(start, text);", source)
        self.assertIn("const nextCursor = positionAfterInsertedText({ lineNumber: range.startLineNumber, column: range.startColumn }, text);", source)
        self.assertIn("function activeFileEditorIdleWritable()", source)
        self.assertIn("if (!activeFileEditorIdleWritable()) return false;", source)
        self.assertIn("!isFileViewerSessionUnavailable()", source)
        self.assertIn("activeFileSaveToken === save.token", viewer_source)
        self.assertIn("fileEditMode = Boolean(nextMode) && activeFileEditModeAllowedInCurrentView();", source)
        self.assertIn("function syncFileUnsavedDialogMode()", source)
        self.assertIn('title.textContent = unavailable ? "Session unavailable" : "Unsaved changes"', source)
        self.assertIn('message.textContent = unavailable ? "This session is no longer available. Copy your edits before closing; they cannot be saved here." : "Save this file before leaving the editor?"', source)
        self.assertIn("saveBtn.hidden = unavailable;", source)
        self.assertIn("saveBtn.disabled = unavailable;", source)
        self.assertIn('discardBtn.textContent = unavailable ? "Close without saving" : "Discard"', source)
        self.assertIn("syncFileUnsavedDialogMode();", source)
        self.assertIn('$("#fileUnsavedSaveBtn").onclick = () => {\n          if (blockUnavailableFileAction()) return;', source)
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
        self.assertEqual(result["secondApiPath"], "derived: trail.md ")
        self.assertEqual(result["activeLine"], 8)
        self.assertEqual(result["sameApiPath"], "kept:tok-same")
        self.assertTrue(result["sameGitPath"])
        self.assertEqual(result["explicitApiPath"], "explicit-token")
        self.assertEqual(result["nongitApiPath"], "")
        self.assertFalse(result["nongitGitPath"])
        self.assertTrue(result["helperRejectsMissingCurrent"])
        self.assertEqual(result["fileApiPathCalls"], [[" trail.md ", ""], ["same.py", "tok-same"]])
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
            ["setFileViewMode", "file"],
            ["applyActiveFileTextState", {"text": "", "editable": True, "version": "", "draft": True}],
            ["applyFileMode"],
            ["renderMonacoFile", "new/file.txt", "", 7, ""],
            ["setFileEditMode", True],
            ["rememberActiveFileSelection"],
            ["renderFilePickerMenu"],
        ])
        self.assertEqual(result["success"]["status"], "new/file.txt - new file")
        self.assertFalse(result["failedRender"]["ok"])
        self.assertEqual(result["failedRender"]["calls"], [
            ["applyActiveFileTextState", {"text": "", "editable": True, "version": "", "draft": True}],
            ["applyFileMode"],
            ["renderMonacoFile", "new/file.txt", "", 7, ""],
        ])
        self.assertEqual(result["failedRender"]["status"], "")
        self.assertFalse(result["staleResult"]["ok"])
        self.assertEqual(result["staleResult"]["calls"], [
            ["applyActiveFileTextState", {"text": "", "editable": True, "version": "", "draft": True}],
            ["applyFileMode"],
            ["renderMonacoFile", "new/file.txt", "", 7, ""],
        ])
        self.assertEqual(result["staleResult"]["status"], "")
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("async function applyDraftFileLoad(rel, request)", source)
        self.assertIn("return fileViewerController.applyDraftFileLoad(rel, request);", source)
        self.assertIn("const loaded = await applyDraftFileLoad(rel, request);\n            if (!loaded) return;", source)
        self.assertIn("async function applyDraftFileLoad(rel, request)", viewer_source)
        self.assertIn("function renderDraftFileOpenError(request, error)", viewer_source)
        self.assertIn("fileViewerController.renderDraftFileOpenError(request, e);\n            return;", source)
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
        self.assertTrue(result["invalidTextThrows"])
        self.assertTrue(result["invalidNonTextThrows"])
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn("function applyActiveFileTextState({ kind = \"text\", text = \"\", editable = false, version = \"\", draft = false } = {})", source)
        self.assertIn("function applyActiveFileDiffState({ currentText = \"\", currentExists = false } = {})", source)
        self.assertIn("function applyActiveFileNonTextState(kind)", source)
        self.assertIn('throw new Error("invalid active file text kind")', source)
        self.assertIn('throw new Error("invalid active file non-text kind")', source)
        self.assertIn('applyActiveFileTextState({ text: "", editable: true, version: "", draft: true });', viewer_source)
        self.assertIn("applyActiveFileDiffState({ currentText, currentExists: result.currentExists });", source)
        self.assertIn('applyActiveFileNonTextState("image");', source)
        self.assertIn('applyActiveFileNonTextState("pdf");', source)
        self.assertIn('applyActiveFileNonTextState("video");', source)
        self.assertIn('applyActiveFileNonTextState("download_only");', source)
        self.assertIn('applyActiveFileTextState({ kind: result.kind === "markdown" ? "markdown" : "text", text: result.text, editable: Boolean(result.editable), version: typeof result.version === "string" ? result.version : "" });', source)

    def test_file_open_success_finalizer_is_single_owned(self) -> None:
        result = eval_file_open_success_finalizer()
        self.assertTrue(result["ok"])
        self.assertEqual(result["calls"], [
            ["applyFileMode"],
            ["rememberOpenedFile", "src/app.py", "/abs/src/app.py"],
            ["rememberActiveFileSelection"],
            ["updateFileEditButton"],
            ["renderFilePickerMenu"],
        ])
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function finalizeFileOpenSuccess(rel, absPath = null)", source)
        self.assertIn("return fileViewerController.finalizeFileOpenSuccess(rel, absPath);", source)
        self.assertIn("return finalizeFileOpenSuccess(rel, openResult.absPath);", source)
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
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
        self.assertIn("moveFileTouchSelection(direction);", source)
        self.assertIn('moveFileTouchSelection("up")', source)
        self.assertIn('moveFileTouchSelection("left")', source)
        self.assertIn('moveFileTouchSelection("down")', source)
        self.assertIn('moveFileTouchSelection("right")', source)
        self.assertIn('editor.trigger("file-touch-select", "cursorMove", args);', source)
        self.assertIn('{ to: "left", by: "character", value: 1, select: true }', source)
        self.assertIn('{ to: "right", by: "character", value: 1, select: true }', source)
        self.assertIn('{ to: "up", by: "wrappedLine", value: 1, select: true }', source)
        self.assertIn('{ to: "down", by: "wrappedLine", value: 1, select: true }', source)
        self.assertIn("fileTouchSelectHead", source)
        self.assertIn('addAppEvent(document, "keydown", handleFileTouchSelectionKeydown, true);', source)
        self.assertIn('fileTouchSelectMode', source)
        self.assertIn('useTouchFileEditorControls()', source)
        self.assertNotIn('if (current.column > 1) {', source)

    def test_touch_toolbar_hides_unusable_controls(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css_source = APP_CSS.read_text(encoding="utf-8")
        self.assertIn('fileTouchDpad.style.display = fileTouchSelectMode ? "grid" : "none";', source)
        self.assertIn('fileTouchCopyBtn.style.display = hasSelection ? "" : "none";', source)
        self.assertIn('fileTouchPasteBtn.style.display = canPaste ? "" : "none";', source)
        self.assertIn("justify-content: space-between;", css_source)
        self.assertIn("pointer-events: none;", css_source)
        self.assertIn("margin-left: auto;", css_source)

    def test_file_editor_capability_predicates_preserve_distinctions(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function currentFileEditorState()", source)
        self.assertIn("function fileEditorCapabilities(state)", source)
        self.assertIn("function activeFileEditorCapabilities()", source)
        self.assertIn("return fileEditorCapabilities(currentFileEditorState());", source)
        self.assertIn("Object.freeze({", source)
        self.assertIn("fileEditor.updateOptions({ readOnly: !activeFileEditorWritable() });", source)
        self.assertIn("const canPaste = activeFileEditorIdleTextWritable();", source)
        self.assertIn("fileEditMode = Boolean(nextMode) && activeFileEditModeAllowedInCurrentView();", source)
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
        self.assertIn("function handleFileEditorSaveShortcut(e)", source)
        self.assertIn('key !== "s" || !(e.ctrlKey || e.metaKey) || e.altKey || e.shiftKey', source)
        self.assertIn("fileEditorShortcutBlocked(target)", source)
        self.assertIn("void saveActiveFileEdits({ exitEditMode: false });", source)
        self.assertIn('addAppEvent(document, "keydown", handleFileEditorSaveShortcut, true);', source)
        result = eval_file_editor_save_shortcut()
        for key in ("validCtrl", "validMeta"):
            with self.subTest(key=key):
                case = result[key]
                self.assertTrue(case["handled"])
                self.assertEqual(case["prevented"], 1)
                self.assertEqual(case["stopped"], 1)
                self.assertEqual(case["saves"], [{"exitEditMode": False}])
        for key in ("noModifier", "wrongKey", "notEdit", "pending", "unavailable", "nestedDialog", "otherTextEntry", "noPath", "viewerClosed"):
            with self.subTest(key=key):
                case = result[key]
                self.assertFalse(case["handled"])
                self.assertEqual(case["prevented"], 0)
                self.assertEqual(case["stopped"], 0)
                self.assertEqual(case["saves"], [])

    def test_touch_select_mode_refocuses_editor_and_blocks_printable_edits(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("focusActiveFileCodeEditor()", source)
        self.assertIn("syncFileDiffSelectionMode()", source)
        self.assertIn("? { enabled: false }", source)
        self.assertIn('function bindFileTouchPress(button, handler)', source)
        self.assertIn('function bindFileTouchClick(button, handler)', source)
        self.assertIn('button.addEventListener("pointerdown"', source)
        self.assertIn('"touchstart"', source)
        self.assertIn("let sawPointerTouchAt = 0;", source)
        self.assertIn("if (e && e.pointerType === \"touch\") sawPointerTouchAt = Date.now();", source)
        self.assertIn("if (Date.now() - sawPointerTouchAt < 700)", source)
        self.assertIn('touch-action: none;', APP_CSS.read_text(encoding="utf-8"))
        self.assertIn('const blocksEdit =', source)
        self.assertIn('key === "backspace"', source)
        self.assertIn('key.length === 1', source)
        self.assertIn('resetFileTouchSelectionState({ collapse: true });', source)
        self.assertIn('if (fileEditorShortcutBlocked(target)) return;', source)
        self.assertNotIn('isTextEntryElement(target) && !target.classList.contains("inputarea")', source)
        result = eval_file_touch_selection_keydown()
        self.assertEqual(result["validMove"], {"prevented": 1, "stopped": 1, "moves": ["right"], "resetArgs": []})
        self.assertEqual(result["validEscape"], {"prevented": 1, "stopped": 1, "moves": [], "resetArgs": [{"collapse": True}]})
        self.assertEqual(result["printableBlocked"], {"prevented": 1, "stopped": 1, "moves": [], "resetArgs": []})
        for key in ("nestedDialog", "viewerClosed", "otherTextEntry", "outsideViewerButton", "toolbarInactive"):
            with self.subTest(key=key):
                self.assertEqual(result[key], {"prevented": 0, "stopped": 0, "moves": [], "resetArgs": []})

    def test_delete_backspace_is_single_owned_in_touch_select_edit_mode(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        helper_source = (APP_JS.parent / "app_file_helpers.js").read_text(encoding="utf-8")
        self.assertIn("function handleFileEditorDeleteKeydown(e)", source)
        self.assertIn("function isActiveFileEditorInput(target)", source)
        self.assertIn("function fileEditorDeleteCommandForKey(key)", source)
        self.assertIn("return codoxearFileHelpers.fileEditorDeleteCommandForKey(key);", source)
        self.assertIn("let fileTouchDeleteNativeSuppressUntil = 0;", source)
        self.assertIn('const key = String(e.key || "").toLowerCase();', source)
        self.assertIn('if (key === "backspace") return "deleteLeft";', helper_source)
        self.assertIn('if (key === "delete") return "deleteRight";', helper_source)
        self.assertNotIn('if (key === "backspace") return "deleteLeft";', source)
        self.assertIn("fileTouchDeleteNativeSuppressUntil = Date.now() + 250;", source)
        self.assertIn('editor.trigger("file-editor-delete-key", command, null);', source)
        self.assertIn("if (fileEditorShortcutBlocked(target)) return false;", source)
        self.assertIn("function isFileEditorNativeDeleteEvent(e)", source)
        self.assertIn('inputType !== "deleteContentBackward" && inputType !== "deleteContentForward"', source)
        self.assertIn('addAppEvent(document, "keydown", handleFileEditorDeleteKeydown, true);', source)
        self.assertIn('addAppEvent(\n          document,\n          "beforeinput",', source)
        self.assertIn('addAppEvent(\n          document,\n          "input",', source)
        self.assertIn("e.preventDefault();\n          e.stopPropagation();", source)
        self.assertNotIn("const allowEditorDelete =", source)
        self.assertIn("if (fileTouchSelectMode) resetFileTouchSelectionState();", source)
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
                self.assertEqual(case["resetCount"], 1)
                self.assertEqual(case["suppressUntil"], 123706)
        for key in ("nestedDialog", "viewerClosed", "otherTextEntry", "notEdit", "unavailable"):
            with self.subTest(key=key):
                case = result[key]
                self.assertFalse(case["handled"])
                self.assertEqual(case["prevented"], 0)
                self.assertEqual(case["stopped"], 0)
                self.assertEqual(case["triggers"], [])
                self.assertEqual(case["focusCount"], 0)
                self.assertEqual(case["resetCount"], 0)

    def test_range_selection_does_not_collapse_back_to_cursor(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('if (!nextAnchor && typeof editor.setPosition === "function") editor.setPosition(nextCursor);', source)

    def test_file_open_race_guard_is_wired_through_fetch_and_render(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        open_file_start = source.index("async function openFilePath(nextPath")
        open_file_end = source.index("fileBtn.onclick", open_file_start)
        open_file_block = source[open_file_start:open_file_end]
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
        self.assertIn("function startFileOpenRequest(nextPath = null, { line = undefined, gitPath = undefined, apiPath = undefined } = {})", source)
        self.assertIn("function startFileOpenRequest(nextPath = null, { line = undefined, gitPath = undefined, apiPath = undefined } = {})", viewer_source)
        self.assertIn("function setFileRenderSurface(surface)", source)
        self.assertIn('throw new Error("invalid file render surface")', source)
        self.assertIn("async function applyFileLoadResult(rel, result, request, { viewMode = \"file\" } = {})", source)
        self.assertIn("function finalizeFileOpenSuccess(rel, absPath = null)", source)
        self.assertIn("function finalizeFileOpenSuccess(rel, absPath = null)", viewer_source)
        self.assertIn("return fileViewerController.finalizeFileOpenSuccess(rel, absPath);", source)
        self.assertIn("const openResult = await fileViewerController.fetchFileOpenResult(request, rel, viewMode);", source)
        self.assertIn("const loaded = await applyFileLoadResult(rel, openResult.result, request, { viewMode });", source)
        self.assertIn("return fileViewerController.renderFileOpenError(request, e);", open_file_block)
        self.assertIn("function renderFileOpenError(request, error)", viewer_source)
        self.assertIn("fileStatus.textContent = `error: ${error && error.message ? error.message : \"unknown error\"}`;", viewer_source)
        self.assertNotIn("fileStatus.textContent = `error: ${e && e.message ? e.message : \"unknown error\"}`;", open_file_block)
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
        self.assertIn("return fileViewerController.beginFileOpenRequest(nextPath, { line, gitPath, apiPath });", source)
        self.assertIn("const request = beginFileOpenRequest(nextPath, { line, gitPath, apiPath });", viewer_source)
        self.assertIn("const openRequest = startFileOpenRequest(nextPath, { line, gitPath, apiPath });", source)
        self.assertIn("const request = openRequest.request;", source)
        self.assertIn("signal: request.signal", viewer_source)
        self.assertIn("if (!isCurrentFileOpenRequest(request)) return false;", source)
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
        self.assertEqual(result["callsAfterMark"], [["getFileEditorText"], ["updateFileEditButton"], ["syncFileEditorReadOnly"]])
        self.assertFalse(result["currentWrongApiPath"])
        self.assertFalse(result["currentWrongGitPath"])
        self.assertFalse(result["currentUnavailable"])
        self.assertTrue(result["afterMismatchedFinish"]["pending"])
        self.assertFalse(result["afterMatchedFinish"]["pending"])
        self.assertEqual(result["afterMatchedFinish"]["calls"][-2:], [["syncFileEditorReadOnly"], ["updateFileEditButton"]])
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
        self.assertIn("const saveBody = buildActiveFileSaveBody(save);", source)
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
        self.assertIn("renderActiveFileSaveError(save, e);\n            return false;", source)
        self.assertIn("function renderActiveFileSaveError(save, error)", viewer_source)
        self.assertIn("renderSaveConflict(save.sessionId, save.path", viewer_source)
        self.assertIn("fileStatus.textContent = `save error: ${error && error.message ? error.message : \"unknown error\"}`;", viewer_source)

    def test_active_file_save_success_applies_response_state(self) -> None:
        result = eval_active_file_save_success()
        self.assertTrue(result["draft"]["ok"])
        self.assertEqual(result["draft"]["state"], {
            "text": "NEW",
            "version": "v2",
            "editable": True,
            "draft": False,
            "gitPath": False,
            "apiPath": "",
            "dirty": False,
            "editMode": False,
            "status": "new.py - 3B",
            "calls": [
                ["applyFileMode"],
                ["setFileDirty", False],
                ["setFileEditMode", False],
                ["rememberOpenedFile", "new.py", "/abs/new.py"],
                ["renderFilePickerMenu"],
            ],
        })
        self.assertTrue(result["nondraft"]["ok"])
        self.assertEqual(result["nondraft"]["state"], {
            "text": "BODY",
            "version": "v0",
            "editable": True,
            "draft": False,
            "gitPath": True,
            "apiPath": "keep-token",
            "dirty": False,
            "editMode": True,
            "status": "existing.py - 4B",
            "calls": [
                ["applyFileMode"],
                ["setFileDirty", False],
                ["rememberOpenedFile", "existing.py", None],
                ["renderFilePickerMenu"],
            ],
        })
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function applyActiveFileSaveSuccess(save, res, { exitEditMode = true } = {})", source)
        self.assertIn("return applyActiveFileSaveSuccess(save, res, { exitEditMode });", source)

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
        self.assertIn("const saveStillCurrent = () => isCurrentActiveFileSaveRequest(save);", block)
        self.assertIn("? { path: save.path, text: save.text, create: true }", viewer_source)
        self.assertIn(": { path: save.path, text: save.text, version: save.version, git_path: save.gitPath };", viewer_source)
        self.assertIn("if (!save.draft && save.gitPath && save.apiPath) body.path_token = save.apiPath;", viewer_source)
        self.assertIn("const saveBody = buildActiveFileSaveBody(save);", block)
        self.assertIn("await api(`/api/sessions/${save.sessionId}/file/write`", block)
        self.assertIn("if (!saveStillCurrent()) return true;", block)
        self.assertIn("return applyActiveFileSaveSuccess(save, res, { exitEditMode });", block)
        self.assertIn("if (!saveStillCurrent()) return false;", block)
        self.assertIn("fileStatus.textContent = `${save.path} - ${fmtBytes(size)}`;", source)
        self.assertIn("rememberOpenedFile(save.path,", source)

    def test_file_save_conflict_delegates_to_file_viewer_controller(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        css_source = APP_CSS.read_text(encoding="utf-8")
        self.assertIn("const codoxearFileViewer = window.CodoxearFileViewer;", source)
        self.assertIn("createFileViewerController", source)
        controller_start = source.index("const fileViewerController = codoxearFileViewer.createFileViewerController")
        controller_end = source.index("function beginActiveFileSaveRequest", controller_start)
        controller_block = source[controller_start:controller_end]
        self.assertIn("currentSessionId: () => fileViewerSessionId", controller_block)
        self.assertIn("normalizeLineNumber", controller_block)
        self.assertIn("normalizeFileApiPath", controller_block)
        self.assertIn("fileApiPathForPath", controller_block)
        self.assertIn("isUnavailable: () => isFileViewerSessionUnavailable()", controller_block)
        self.assertIn("confirmReload: (message) => window.confirm(message)", controller_block)
        self.assertIn("openFilePath: (path, options) => openFilePath(path, options)", controller_block)
        self.assertIn("focusEditor: () => getActiveFileCodeEditor()", controller_block)
        save_start = source.index("async function saveActiveFileEdits")
        save_end = source.index("async function maybeHandleUnsavedFileChanges", save_start)
        save_block = source[save_start:save_end]
        self.assertIn("renderActiveFileSaveError(save, e);", save_block)
        self.assertIn("renderSaveConflict(save.sessionId, save.path, error && error.message ? error.message : \"conflict\");", viewer_source)
        self.assertNotIn("function renderFileSaveConflict", source)
        self.assertNotIn("let fileSaveSeq = 0;", source)
        self.assertNotIn("let activeFileSaveToken = 0;", source)
        self.assertIn("let fileSaveSeq = 0;", viewer_source)
        self.assertIn("let activeFileSaveToken = 0;", viewer_source)
        self.assertIn("if (!save || activeFileSaveToken !== save.token) return;", viewer_source)
        self.assertIn("finishActiveFileSaveRequest(save);", save_block)
        self.assertIn(".fileConflictActions", css_source)


    def test_file_viewer_handles_pdf_video_and_download_only_kinds(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css_source = APP_CSS.read_text(encoding="utf-8")
        self.assertNotIn('el("iframe"', source)
        self.assertIn('import(resolveAppUrl("pdf.mjs"))', source)
        self.assertIn('resolveAppUrl("pdf.worker.mjs")', source)
        self.assertIn('result.kind === "pdf"', source)
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
        self.assertIn("PDF renderer timed out", source)
        self.assertIn("pdfjsReadyPromise.catch(() => {", source)
        self.assertIn('const fileVideoPreviewBtn = el("button", {', source)
        self.assertIn('id: "fileVideoPreviewBtn"', source)
        self.assertIn('title: "Use compatible MP4 preview"', source)
        self.assertIn('const fileVideo = el("video", { id: "fileVideo", class: "fileVideo", controls: true, preload: "metadata" });', source)
        self.assertIn('result.kind === "video"', source)
        self.assertIn("function clearFileVideo()", source)
        self.assertIn("fileVideo.pause();", source)
        self.assertIn('fileVideo.src = resolveAppUrl(result.video_url);', source)
        self.assertIn("const previewUrl = typeof result.video_preview_url === \"string\" ? result.video_preview_url : \"\";", source)
        self.assertIn('const browserSafeVideoTypes = new Set(["video/mp4", "video/webm", "video/ogg"]);', source)
        self.assertIn('const shouldPreviewFirst = Boolean(previewUrl && contentType && !browserSafeVideoTypes.has(contentType));', source)
        self.assertIn('async function prepareCompatibleVideoPreview(previewUrl) {', source)
        self.assertIn('headers: { Range: "bytes=0-0" }', source)
        self.assertIn('if (res.status === 401) {', source)
        self.assertIn('handleAppAuthLoss();', source)
        self.assertIn('const obj = await res.clone().json();', source)
        self.assertIn('if (obj && typeof obj.error === "string") detail = obj.error;', source)
        self.assertIn('throw new Error(detail || `video preview failed (${res.status})`);', source)
        self.assertIn('async function loadCompatibleVideoPreview(expectedToken = "", { explicit = false } = {})', source)
        self.assertIn('fileVideoPreviewBtn.onclick = (e) => {', source)
        self.assertIn('void loadCompatibleVideoPreview(token, { explicit: true });', source)
        self.assertIn('if (shouldPreviewFirst) {', source)
        self.assertIn('void loadCompatibleVideoPreview(videoToken, { explicit: false });', source)
        self.assertIn("fileVideo.onerror = () => {", source)
        self.assertIn("fileStatus.textContent = explicit ? `${rel} - building compatible video preview...` : `${rel} - trying compatible video preview...`;", source)
        self.assertIn("fileVideo.src = resolveAppUrl(state.previewUrl);", source)
        self.assertIn("fileStatus.textContent = `${rel} - compatible video preview - ${fmtBytes(size)}`;", source)
        self.assertIn("fileStatus.textContent = `${rel} - video preview unavailable after conversion`;", source)
        self.assertIn('setFileRenderSurface("video");', source)
        self.assertIn('fileStatus.textContent = `${rel} - video - ${fmtBytes(size)}`;', source)
        self.assertIn('result.kind === "download_only"', source)
        self.assertIn("renderBlockedFileNotice(rel, String(result.reason || \"\"), Number(result.viewer_max_bytes || 0), size);", source)
        self.assertIn('fileStatus.textContent = `${rel} - PDF - ${fmtBytes(size)}`;', source)
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
        self.assertIn("navigator.clipboard", source)
        self.assertIn("readText", source)
        self.assertIn('setToast("pasted")', source)
        self.assertIn('function showFilePasteDialog()', source)
        self.assertIn('function hideFilePasteDialog({ restoreFocus = false } = {})', source)
        self.assertIn('hideFilePasteDialog({ restoreFocus: true });', source)
        self.assertIn('$("#filePasteInsertBtn").onclick = () => {\n          if (blockUnavailableFileAction()) return;\n          if (insertIntoActiveFileEditor(filePasteInput.value)) {', source)
        self.assertIn('filePasteDialog.style.display = "flex";', source)
        self.assertIn('filePasteInput.focus({ preventScroll: true });', source)
        self.assertIn('setToast("paste manually")', source)
        self.assertIn('function pasteFromClipboardIntoActiveFile()', source)
        self.assertIn('setToast("paste unavailable")', source)
        self.assertIn('setToast("clipboard empty")', source)
        self.assertIn('bindFileTouchClick(fileTouchPasteBtn, () => {', source)
        self.assertNotIn('bindFileTouchPress(fileTouchPasteBtn, () => {', source)

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
        self.assertEqual(result["direct"]["dialog"], "none")
        self.assertEqual(result["direct"]["inserted"], ["hello"])
        self.assertEqual(result["direct"]["toasts"], ["pasted"])
        self.assertEqual(result["direct"]["focusEditorCount"], 1)
        self.assertEqual(result["empty"]["dialog"], "none")
        self.assertEqual(result["empty"]["toasts"], ["clipboard empty"])
        self.assertEqual(result["empty"]["focusEditorCount"], 1)
        self.assertEqual(result["dismissed"]["backdrop"], "none")
        self.assertEqual(result["dismissed"]["dialog"], "none")
        self.assertEqual(result["dismissed"]["focusEditorCount"], 1)
        self.assertEqual(result["dismissed"]["modalSyncCount"], 2)

    def test_touch_copy_uses_click_activation_not_press_wrapper(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('bindFileTouchClick(fileTouchCopyBtn, () => {', source)
        self.assertNotIn('bindFileTouchPress(fileTouchCopyBtn, () => {', source)


if __name__ == "__main__":
    unittest.main()
