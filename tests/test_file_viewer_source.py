import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"
SERVER_PY = Path(__file__).resolve().parents[1] / "codoxear" / "server.py"


def eval_use_touch_file_editor_controls(query_matches: dict[str, bool]) -> bool:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function useTouchFileEditorControls() {")
    end = source.index("function setSidebarOpen(open) {", start)
    snippet = source[start:end]
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
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_useTouchFileEditorControls = useTouchFileEditorControls;\n")}, ctx);
        process.stdout.write(JSON.stringify(ctx.__test_useTouchFileEditorControls()));
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
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("let fileOpenRequestId = 0;")
    end = source.index("function rememberActiveFileSelection(", start)
    snippet = source[start:end]
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
        const ctx = {{
          activeFilePath: "old.txt",
          activeFileGitPath: false,
          activeFileLine: 1,
          fileViewerSessionId: "sid-1",
          selected: "",
          AbortController,
          disposePdfRender: () => {{}},
          normalizeLineNumber: (value) => value == null ? null : Number(value),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_file_open = { beginFileOpenRequest, isCurrentFileOpenRequest, cancelPendingFileOpen, currentFileSessionId };\n")}, ctx);
        const first = ctx.__test_file_open.beginFileOpenRequest("first.txt", {{ line: 3 }});
        const firstCurrent = ctx.__test_file_open.isCurrentFileOpenRequest(first);
        const second = ctx.__test_file_open.beginFileOpenRequest(" trail.md ", {{ line: 8, gitPath: true }});
        const result = {{
          currentSessionId: ctx.__test_file_open.currentFileSessionId(),
          firstCurrent,
          firstSignalAborted: Boolean(first.signal && first.signal.aborted),
          firstAfterSecond: ctx.__test_file_open.isCurrentFileOpenRequest(first),
          secondCurrent: ctx.__test_file_open.isCurrentFileOpenRequest(second),
          secondGitPath: second.gitPath,
          activeFilePath: ctx.activeFilePath,
          activeFileLine: ctx.activeFileLine,
        }};
        ctx.__test_file_open.cancelPendingFileOpen();
        result.secondAfterCancel = ctx.__test_file_open.isCurrentFileOpenRequest(second);
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
    start = source.index("let fileOpenRequestId = 0;")
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
          activeFilePath: "old.txt",
          activeFileGitPath: false,
          activeFileLine: 1,
          fileSearchSessionId: "sid-a",
          fileCandidateList: ["candidate.txt"],
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


class TestFileViewerSource(unittest.TestCase):
    def test_file_editor_disables_monaco_suggestions(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('accessibilitySupport: "off"', source)
        self.assertIn("quickSuggestions: false", source)
        self.assertIn("suggestOnTriggerCharacters: false", source)
        self.assertIn('acceptSuggestionOnEnter: "off"', source)
        self.assertIn('tabCompletion: "off"', source)
        self.assertIn('wordBasedSuggestions: "off"', source)

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
        self.assertIn("openFilePathWithResolvedMode(preferred.path, { line: preferred.line, isCurrent: () => isFileViewerSessionCurrent(sid, syncToken) })", ensure_block)
        self.assertIn("openFilePathWithResolvedMode(first, { line: null, isCurrent: () => isFileViewerSessionCurrent(sid, syncToken) })", ensure_block)
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
        self.assertIn("openFilePathWithResolvedMode(preferred, { line: preferredLine, isCurrent: () => isFileViewerSessionCurrent(sid, syncToken) })", show_block)
        self.assertIn("openFilePathWithResolvedMode(first, { line: null, isCurrent: () => isFileViewerSessionCurrent(sid, syncToken) })", show_block)
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
        self.assertIn("const gitPath = isGitFileCandidatePath(path, changed);", resolved_block)
        self.assertIn("mode = await resolveFileOpenMode(path, { changed });", resolved_block)
        self.assertIn("if (blockUnavailableFileAction()) return false;", resolved_block)
        self.assertIn("return await openFilePathWithGuard(path, { line, mode, isCurrent: currentGuard, gitPath });", resolved_block)
        guard_start = source.index("async function openFilePathWithGuard")
        guard_end = source.index("async function openFilePathWithResolvedMode", guard_start)
        guard_block = source[guard_start:guard_end]
        self.assertIn("gitPath = false", guard_block)
        self.assertIn("isCurrent = null", guard_block)
        self.assertIn("const sessionAtStart = currentFileSessionId();", guard_block)
        self.assertIn("const currentGuard = typeof isCurrent === \"function\" ? isCurrent : () => currentFileSessionId() === sessionAtStart && !isFileViewerSessionUnavailable();", guard_block)
        self.assertIn("if (!currentGuard()) return false;", guard_block)
        self.assertIn("await openFilePath(path, { line, gitPath });", guard_block)
        self.assertIn("return Boolean(currentGuard());", guard_block)
        self.assertIn("if (gitPath) body.git_path = true;", source)
        self.assertIn('const gitPathQuery = request.gitPath ? "&git_path=1" : "";', source)
        self.assertIn("git_path: Boolean(activeFileGitPath)", source)
        self.assertIn("beginFileOpenRequest(path, { line, gitPath: false })", source)
        self.assertIn("setFilePath(rel, { line: null, gitPath: false })", source)
        self.assertIn("if (saveDraft) activeFileGitPath = false;", source)

    def test_file_viewer_handles_selected_session_removal(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("let fileViewerUnavailableSessionId = \"\";", source)
        self.assertIn("function isFileViewerSessionUnavailable()", source)
        self.assertIn("function blockUnavailableFileAction()", source)
        self.assertIn("function handleFileViewerSessionUnavailable(sessionId)", source)
        helper_start = source.index("function handleFileViewerSessionUnavailable(sessionId)")
        helper_end = source.index("async function openFilePath", helper_start)
        helper_block = source[helper_start:helper_end]
        self.assertIn("if (fileViewerSessionId && fileViewerSessionId !== sid) return;", helper_block)
        self.assertIn("if (!fileDirty) {\n            hideFileViewer();\n            return;\n          }", helper_block)
        self.assertIn("fileViewerSessionSyncToken += 1;", helper_block)
        self.assertIn("fileViewerUnavailableSessionId = sid;", helper_block)
        self.assertIn("activeFileSaveToken = 0;", helper_block)
        self.assertIn("fileSavePending = false;", helper_block)
        self.assertIn("fileEditMode = false;", helper_block)
        self.assertIn('hideFileUnsavedDialog("cancel");', helper_block)
        self.assertIn("cancelPendingFileOpen();", helper_block)
        self.assertIn("resetFileSearchState();", helper_block)
        self.assertIn("Session is no longer available; copy unsaved edits before closing.", helper_block)
        self.assertIn("syncFileEditorReadOnly();", helper_block)
        self.assertIn("updateFileEditButton();", helper_block)
        self.assertIn("fileViewerUnavailableSessionId = \"\";", source)
        self.assertIn("if (isFileViewerSessionUnavailable()) return false;", source)
        self.assertIn("if (blockUnavailableFileAction()) return false;", source)
        open_primitive_start = source.index("async function openFilePath(nextPath")
        open_primitive_end = source.index("fileBtn.onclick", open_primitive_start)
        open_primitive_block = source[open_primitive_start:open_primitive_end]
        self.assertIn("if (blockUnavailableFileAction()) return false;", open_primitive_block)
        self.assertIn("if (blockUnavailableFileAction()) return [];", source)
        draft_guard_start = source.index("async function openDraftFilePathWithGuard")
        draft_guard_end = source.index("async function setFileViewModeWithGuard", draft_guard_start)
        draft_guard_block = source[draft_guard_start:draft_guard_end]
        self.assertGreaterEqual(draft_guard_block.count("if (blockUnavailableFileAction()) return false;"), 4)
        draft_start = source.index("async function openDraftFilePath(path")
        draft_end = source.index("function normalizeFileCandidateSource", draft_start)
        draft_block = source[draft_start:draft_end]
        self.assertIn("if (blockUnavailableFileAction()) return;", draft_block)
        self.assertIn("if (blockUnavailableFileAction()) return;", source)
        self.assertIn("if (blockUnavailableFileAction()) return;\n            if (!text)", source)
        self.assertIn("function insertIntoActiveFileEditor(text)", source)
        self.assertIn('fileEditMode && activeFileEditable && fileViewMode === "file" && !fileSavePending && !isFileViewerSessionUnavailable()', source)
        self.assertIn("!isFileViewerSessionUnavailable()", source)
        self.assertIn("activeFileSaveToken === saveToken && !isFileViewerSessionUnavailable()", source)
        self.assertIn('fileEditMode = Boolean(nextMode) && fileViewMode === "file" && isTextFileKind(activeFileKind) && activeFileEditable && !isFileViewerSessionUnavailable();', source)
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
        self.assertIn("const removedSelected = selected;", sessions_block)
        self.assertIn("handleFileViewerSessionUnavailable(removedSelected);", sessions_block)
        self.assertIn("handleFileViewerSessionUnavailable(s.session_id);", source)
        self.assertIn("handleFileViewerSessionUnavailable(sid);", source)
        open_start = source.index("async function openSession(sessionId")
        open_end = source.index("async function pollMessages", open_start)
        open_block = source[open_start:open_end]
        self.assertIn("if (e && e.status === 404) {", open_block)
        self.assertIn("handleFileViewerSessionUnavailable(sessionId);", open_block)
        self.assertIn("selected = null;", open_block)
        self.assertIn('storageRemoveItem("codexweb.selected");', open_block)
        self.assertIn('setSessionHash("");', open_block)
        self.assertIn('titleLabel.textContent = "No session selected";', open_block)
        self.assertIn("syncAttachButtonState();", open_block)
        self.assertIn('console.error("refreshSessions failed after session disappeared", e2);', open_block)

    def test_file_open_requests_are_single_owner(self) -> None:
        result = eval_file_open_request_sequence()
        self.assertEqual(result["currentSessionId"], "sid-1")
        self.assertTrue(result["firstCurrent"])
        self.assertTrue(result["firstSignalAborted"])
        self.assertFalse(result["firstAfterSecond"])
        self.assertTrue(result["secondCurrent"])
        self.assertEqual(result["activeFilePath"], " trail.md ")
        self.assertTrue(result["secondGitPath"])
        self.assertEqual(result["activeFileLine"], 8)
        self.assertFalse(result["secondAfterCancel"])

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

    def test_delete_backspace_is_single_owned_in_touch_select_edit_mode(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function handleFileEditorDeleteKeydown(e)", source)
        self.assertIn("function isActiveFileEditorInput(target)", source)
        self.assertIn("function fileEditorDeleteCommandForKey(key)", source)
        self.assertIn("let fileTouchDeleteNativeSuppressUntil = 0;", source)
        self.assertIn('if (key === "backspace") return "deleteLeft";', source)
        self.assertIn('if (key === "delete") return "deleteRight";', source)
        self.assertIn("fileTouchDeleteNativeSuppressUntil = Date.now() + 250;", source)
        self.assertIn('editor.trigger("file-editor-delete-key", command, null);', source)
        self.assertIn("function isFileEditorNativeDeleteEvent(e)", source)
        self.assertIn('inputType !== "deleteContentBackward" && inputType !== "deleteContentForward"', source)
        self.assertIn('addAppEvent(document, "keydown", handleFileEditorDeleteKeydown, true);', source)
        self.assertIn('addAppEvent(\n          document,\n          "beforeinput",', source)
        self.assertIn('addAppEvent(\n          document,\n          "input",', source)
        self.assertIn("e.preventDefault();\n          e.stopPropagation();", source)
        self.assertNotIn("const allowEditorDelete =", source)
        self.assertIn("if (fileTouchSelectMode) resetFileTouchSelectionState();", source)

    def test_range_selection_does_not_collapse_back_to_cursor(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('if (!nextAnchor && typeof editor.setPosition === "function") editor.setPosition(nextCursor);', source)

    def test_file_open_race_guard_is_wired_through_fetch_and_render(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("let fileOpenRequestId = 0;", source)
        self.assertIn("let fileOpenAbortController = null;", source)
        self.assertIn("function cancelPendingFileOpen()", source)
        self.assertIn("const request = beginFileOpenRequest(nextPath, { line, gitPath });", source)
        self.assertIn("signal: request.signal", source)
        self.assertIn("if (!isCurrentFileOpenRequest(request)) return false;", source)
        self.assertIn("async function openFilePathWithResolvedMode(path, { line = null, changed = null, isCurrent = null } = {})", source)
        self.assertIn("async function renderMonacoFile(rel, text, lineNumber = null, langOverride = \"\", request = null)", source)
        self.assertIn("async function renderMonacoDiff(rel, originalText, modifiedText, lineNumber = null, request = null)", source)
        self.assertIn("if (request && !isCurrentFileOpenRequest(request)) return false;", source)
        self.assertIn("renderPlainTextFallback(rel, text, lineNumber", source)
        self.assertIn("renderPlainTextFallback(rel, modifiedText, lineNumber", source)
        self.assertIn("cancelPendingFileOpen();\n          prepareModalOpen();\n          fileBackdrop.style.display = \"block\";", source)
        self.assertIn("cancelPendingFileOpen();\n          hideFileUnsavedDialog();", source)

    def test_file_save_response_is_bound_to_original_session_and_path(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function saveActiveFileEdits")
        end = source.index("async function maybeHandleUnsavedFileChanges", start)
        block = source[start:end]
        self.assertIn("const saveSessionId = fileViewerSessionId;", block)
        self.assertIn("const savePath = activeFilePath;", block)
        self.assertIn("const saveDraft = Boolean(activeFileDraft);", block)
        self.assertIn("const saveVersion = activeFileVersion;", block)
        self.assertIn("const saveToken = ++fileSaveSeq;", block)
        self.assertIn("activeFileSaveToken = saveToken;", block)
        self.assertIn("const saveStillCurrent = () => fileViewerSessionId === saveSessionId && activeFilePath === savePath && activeFileSaveToken === saveToken && !isFileViewerSessionUnavailable();", block)
        self.assertIn("? { path: savePath, text, create: true }", block)
        self.assertIn(": { path: savePath, text, version: saveVersion, git_path: Boolean(activeFileGitPath) };", block)
        self.assertIn("await api(`/api/sessions/${saveSessionId}/file/write`", block)
        self.assertIn("if (!saveStillCurrent()) return true;", block)
        self.assertIn("if (!saveStillCurrent()) return false;", block)
        self.assertIn("fileStatus.textContent = `${savePath} - ${fmtBytes(size)}`;", block)
        self.assertIn("rememberOpenedFile(savePath,", block)

    def test_file_save_conflict_offers_reload_or_keep_editing(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css_source = APP_CSS.read_text(encoding="utf-8")
        start = source.index("function renderFileSaveConflict(saveSessionId, savePath")
        end = source.index("async function saveActiveFileEdits", start)
        block = source[start:end]
        self.assertIn('text: "Reload from disk"', block)
        self.assertIn('text: "Keep editing"', block)
        self.assertIn("window.confirm(`Reload ${savePath} from disk and discard your unsaved editor draft?`);", block)
        self.assertIn("if (fileViewerSessionId !== saveSessionId || activeFilePath !== savePath) return;", block)
        self.assertIn("await openFilePath(savePath, { line: activeFileLine, gitPath: activeFileGitPath });", block)
        self.assertIn("getActiveFileCodeEditor();", block)
        self.assertIn("fileStatus.replaceChildren(label, actions);", block)
        save_start = source.index("async function saveActiveFileEdits")
        save_end = source.index("async function maybeHandleUnsavedFileChanges", save_start)
        save_block = source[save_start:save_end]
        self.assertIn("renderFileSaveConflict(saveSessionId, savePath, e && e.message ? e.message : \"conflict\");", save_block)
        self.assertIn("let fileSaveSeq = 0;", source)
        self.assertIn("let activeFileSaveToken = 0;", source)
        self.assertIn("if (activeFileSaveToken === saveToken) {", save_block)
        self.assertIn("activeFileSaveToken = 0;", save_block)
        self.assertIn(".fileConflictActions", css_source)
        self.assertNotIn("overwrite", block.lower())

    def test_file_write_conflict_check_is_locked_with_write(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        self.assertIn("_FILE_WRITE_LOCKS_LOCK = threading.Lock()", source)
        self.assertIn("_FILE_WRITE_LOCKS: dict[str, tuple[threading.Lock, int]] = {}", source)
        self.assertIn("@contextmanager\ndef _file_write_lock(path: Path) -> Iterator[None]:", source)
        self.assertIn("_FILE_WRITE_LOCKS[key] = (lock, refcount + 1)", source)
        self.assertIn("_FILE_WRITE_LOCKS.pop(key, None)", source)
        route_start = source.index('session_id = _match_session_route(path, "file", "write")')
        route_end = source.index('session_id = _match_session_route(path, "delete")', route_start)
        block = source[route_start:route_end]
        update_block = block[block.index("else:\n                    try:\n                        if git_path:") :]
        self.assertIn('except ValueError as e:\n                        _json_response(self, 400, {"error": str(e)})', update_block)
        self.assertIn("with _file_write_lock(p):", update_block)
        self.assertLess(update_block.index("with _file_write_lock(p):"), update_block.index("_read_text_file_for_write(p"))
        self.assertLess(update_block.index("_read_text_file_for_write(p"), update_block.index("_write_text_file_atomic(p"))

    def test_file_viewer_handles_pdf_video_and_download_only_kinds(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css_source = APP_CSS.read_text(encoding="utf-8")
        self.assertNotIn('el("iframe"', source)
        self.assertIn('import(resolveAppUrl("pdf.mjs"))', source)
        self.assertIn('resolveAppUrl("pdf.worker.mjs")', source)
        self.assertIn('res.kind === "pdf"', source)
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
        self.assertIn('const fileVideo = el("video", { id: "fileVideo", class: "fileVideo", controls: true, preload: "metadata" });', source)
        self.assertIn('res.kind === "video"', source)
        self.assertIn("function clearFileVideo()", source)
        self.assertIn("fileVideo.pause();", source)
        self.assertIn('fileVideo.src = resolveAppUrl(res.video_url);', source)
        self.assertIn("const previewUrl = typeof res.video_preview_url === \"string\" ? res.video_preview_url : \"\";", source)
        self.assertIn('const browserSafeVideoTypes = new Set(["video/mp4", "video/webm", "video/ogg"]);', source)
        self.assertIn('const shouldPreviewFirst = Boolean(previewUrl && contentType && !browserSafeVideoTypes.has(contentType));', source)
        self.assertIn('const usePreview = () => {', source)
        self.assertIn('if (shouldPreviewFirst) {', source)
        self.assertIn("fileVideo.onerror = () => {", source)
        self.assertIn("fileStatus.textContent = `${rel} - building compatible video preview...`;", source)
        self.assertIn("fileVideo.src = resolveAppUrl(state.previewUrl);", source)
        self.assertIn("fileStatus.textContent = `${rel} - compatible video preview - ${fmtBytes(size)}`;", source)
        self.assertIn("fileStatus.textContent = `${rel} - video preview unavailable (ffmpeg missing or conversion failed)`;", source)
        self.assertIn('fileVideo.style.display = "block";', source)
        self.assertIn('fileStatus.textContent = `${rel} - video - ${fmtBytes(size)}`;', source)
        self.assertIn('res.kind === "download_only"', source)
        self.assertIn("renderBlockedFileNotice(rel, String(res.reason || \"\"), Number(res.viewer_max_bytes || 0), size);", source)
        self.assertIn('fileStatus.textContent = `${rel} - PDF - ${fmtBytes(size)}`;', source)
        self.assertIn(".filePdfPages {", css_source)
        self.assertIn(".filePlainFallback {", css_source)
        self.assertIn(".filePlainFallbackText {", css_source)
        self.assertIn(".filePdfPage {", css_source)
        self.assertIn(".fileVideo {", css_source)
        self.assertIn(".fileBlockedNotice {", css_source)

    def test_video_preview_uses_browser_safe_server_transcode(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        module_source = (SERVER_PY.parent / "video_preview.py").read_text(encoding="utf-8")
        self.assertIn("VIDEO_PREVIEW_DIR = APP_DIR / \"video_previews\"", server_source)
        self.assertIn("def _ensure_video_preview(path: Path) -> Path:", server_source)
        self.assertIn("return _ensure_video_preview_impl(path, preview_dir=VIDEO_PREVIEW_DIR)", server_source)
        self.assertIn('"libx264"', module_source)
        self.assertIn('"scale=ceil(iw/2)*2:ceil(ih/2)*2"', module_source)
        self.assertIn('"-pix_fmt"', module_source)
        self.assertIn('"yuv420p"', module_source)
        self.assertIn('"aac"', module_source)
        self.assertIn('"video preview failed:', server_source)
        self.assertIn('"preview_content_type": "video/mp4"', module_source)
        self.assertIn('"video_preview_url": preview_url', module_source)
        self.assertIn('VIDEO_PREVIEW_CACHE_MAX_FILES = _positive_int_env("CODEX_WEB_VIDEO_PREVIEW_MAX_FILES", 256)', module_source)
        self.assertIn('VIDEO_PREVIEW_CACHE_MAX_BYTES = _positive_int_env("CODEX_WEB_VIDEO_PREVIEW_MAX_BYTES", 10 * 1024 * 1024 * 1024)', module_source)
        self.assertIn('def prune_video_preview_cache(', module_source)
        self.assertIn('prune_video_preview_cache(preview_dir, keep=out)', module_source)
        self.assertIn('preview_url=f"/api/sessions/{session_id}/file/video_preview?path={urllib.parse.quote(rel)}{git_suffix}"', server_source)
        self.assertIn("preview_url=media_preview_url", server_source)
        self.assertIn('_send_inline_file_response(self, p, ctype or "application/octet-stream")', server_source)
        self.assertIn('_send_inline_file_response(self, path_obj, ctype or "application/octet-stream")', server_source)
        self.assertIn('_send_inline_file_response(self, preview, "video/mp4")', server_source)

    def test_attach_limit_comes_from_server_constant(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("window.CODOXEAR_ATTACH_MAX_BYTES", source)
        self.assertIn("const ATTACH_UPLOAD_MAX_BYTES = (() => {", source)
        self.assertIn("Attach file (max", source)
        self.assertIn("/inject_file", source)
        self.assertIn('throw new Error(`file too large (max ${fmtBytes(maxBytes)})`);', source)

    def test_clickable_file_extensions_include_pdf_and_archives(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('"pdf"', source)
        self.assertIn('"mp4"', source)
        self.assertIn('"mkv"', source)
        self.assertIn('"avi"', source)
        self.assertIn('"webm"', source)
        self.assertIn('"zip"', source)
        self.assertIn('"tar"', source)

    def test_touch_paste_tries_direct_clipboard_before_bridge(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("navigator.clipboard", source)
        self.assertIn("readText", source)
        self.assertIn('setToast("pasted")', source)
        self.assertIn('function pasteFromClipboardIntoActiveFile()', source)
        self.assertIn('setToast("paste unavailable")', source)
        self.assertIn('setToast("clipboard empty")', source)
        self.assertIn('bindFileTouchClick(fileTouchPasteBtn, () => {', source)
        self.assertNotIn('bindFileTouchPress(fileTouchPasteBtn, () => {', source)

    def test_touch_copy_uses_click_activation_not_press_wrapper(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('bindFileTouchClick(fileTouchCopyBtn, () => {', source)
        self.assertNotIn('bindFileTouchPress(fileTouchCopyBtn, () => {', source)


if __name__ == "__main__":
    unittest.main()
