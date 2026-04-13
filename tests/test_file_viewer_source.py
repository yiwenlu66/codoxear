import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


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
          activeFileLine: 1,
          fileViewerSessionId: "sid-1",
          selected: "",
          AbortController,
          normalizeLineNumber: (value) => value == null ? null : Number(value),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_file_open = { beginFileOpenRequest, isCurrentFileOpenRequest, cancelPendingFileOpen, currentFileSessionId };\n")}, ctx);
        const first = ctx.__test_file_open.beginFileOpenRequest("first.txt", {{ line: 3 }});
        const firstCurrent = ctx.__test_file_open.isCurrentFileOpenRequest(first);
        const second = ctx.__test_file_open.beginFileOpenRequest("second.txt", {{ line: 8 }});
        const result = {{
          currentSessionId: ctx.__test_file_open.currentFileSessionId(),
          firstCurrent,
          firstSignalAborted: Boolean(first.signal && first.signal.aborted),
          firstAfterSecond: ctx.__test_file_open.isCurrentFileOpenRequest(first),
          secondCurrent: ctx.__test_file_open.isCurrentFileOpenRequest(second),
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


class TestFileViewerSource(unittest.TestCase):
    def test_file_editor_disables_monaco_suggestions(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("quickSuggestions: false", source)
        self.assertIn("suggestOnTriggerCharacters: false", source)
        self.assertIn('acceptSuggestionOnEnter: "off"', source)
        self.assertIn('tabCompletion: "off"', source)
        self.assertIn('wordBasedSuggestions: "off"', source)

    def test_client_reload_checks_server_ui_version(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("window.CODOXEAR_ASSET_VERSION", source)
        self.assertIn("function maybeReloadForUpdatedUi(nextVersion)", source)
        self.assertIn("maybeReloadForUpdatedUi(data && data.app_version)", source)
        self.assertIn("window.location.reload();", source)

    def test_file_open_requests_are_single_owner(self) -> None:
        result = eval_file_open_request_sequence()
        self.assertEqual(result["currentSessionId"], "sid-1")
        self.assertTrue(result["firstCurrent"])
        self.assertTrue(result["firstSignalAborted"])
        self.assertFalse(result["firstAfterSecond"])
        self.assertTrue(result["secondCurrent"])
        self.assertEqual(result["activeFilePath"], "second.txt")
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
        self.assertIn('document.addEventListener("keydown", handleFileTouchSelectionKeydown, true);', source)
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

    def test_delete_backspace_can_remove_selection_in_edit_mode(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('const allowEditorDelete = (key === "backspace" || key === "delete") && fileEditMode && activeFileEditable && fileViewMode === "file";', source)
        self.assertIn("if (allowEditorDelete) {", source)
        self.assertNotIn('editor.executeEdits("file-touch-delete"', source)
        self.assertIn("if (fileTouchSelectMode) resetFileTouchSelectionState();", source)

    def test_range_selection_does_not_collapse_back_to_cursor(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('if (!nextAnchor && typeof editor.setPosition === "function") editor.setPosition(nextCursor);', source)

    def test_file_open_race_guard_is_wired_through_fetch_and_render(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("let fileOpenRequestId = 0;", source)
        self.assertIn("let fileOpenAbortController = null;", source)
        self.assertIn("function cancelPendingFileOpen()", source)
        self.assertIn("const request = beginFileOpenRequest(nextPath, { line });", source)
        self.assertIn("signal: request.signal", source)
        self.assertIn("if (!isCurrentFileOpenRequest(request)) return false;", source)
        self.assertIn("async function openFilePathWithResolvedMode(path, { line = null, changed = null } = {})", source)
        self.assertIn("async function renderMonacoFile(rel, text, lineNumber = null, langOverride = \"\", request = null)", source)
        self.assertIn("async function renderMonacoDiff(rel, originalText, modifiedText, lineNumber = null, request = null)", source)
        self.assertIn("if (request && !isCurrentFileOpenRequest(request)) return false;", source)
        self.assertIn("cancelPendingFileOpen();\n          fileBackdrop.style.display = \"block\";", source)
        self.assertIn("cancelPendingFileOpen();\n          hideFileUnsavedDialog();", source)

    def test_file_viewer_handles_pdf_and_download_only_kinds(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css_source = APP_CSS.read_text(encoding="utf-8")
        self.assertIn('const filePdf = el("iframe", { id: "filePdf", class: "filePdf", title: "PDF preview" });', source)
        self.assertIn('res.kind === "pdf"', source)
        self.assertIn('res.kind === "download_only"', source)
        self.assertIn("renderBlockedFileNotice(rel, String(res.reason || \"\"), Number(res.viewer_max_bytes || 0), size);", source)
        self.assertIn('fileStatus.textContent = `${rel} - PDF - ${fmtBytes(size)}`;', source)
        self.assertIn(".filePdf {", css_source)
        self.assertIn(".fileBlockedNotice {", css_source)

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
