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
        self.assertIn('function deleteActiveFileSelection()', source)
        self.assertIn('editor.executeEdits("file-touch-delete"', source)
        self.assertIn('(key === "backspace" || key === "delete") && fileEditMode && activeFileEditable && fileViewMode === "file"', source)

    def test_range_selection_does_not_collapse_back_to_cursor(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('if (!nextAnchor && typeof editor.setPosition === "function") editor.setPosition(nextCursor);', source)

    def test_touch_paste_tries_direct_clipboard_before_bridge(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("navigator.clipboard", source)
        self.assertIn("readText", source)
        self.assertIn('setToast("pasted")', source)
        self.assertIn('filePasteDialog.style.display = "flex";', source)


if __name__ == "__main__":
    unittest.main()
