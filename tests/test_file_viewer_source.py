import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


def render_js(template: str, **replacements: str) -> str:
    rendered = textwrap.dedent(template)
    for key, value in replacements.items():
        rendered = rendered.replace(f"__{key}__", value)
    return rendered


def run_node_json(script: str):
    proc = subprocess.run(
        ["node", "-e", script],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_use_touch_file_editor_controls(query_matches: dict[str, bool]) -> bool:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function useTouchFileEditorControls() {")
    end = source.index("function setSidebarOpen(open) {", start)
    snippet = source[start:end]
    js = render_js(
        """
        const vm = require("vm");
        const queryMatches = __QUERY_MATCHES__;
        const ctx = {
          window: {
            matchMedia: (query) => ({ matches: Boolean(queryMatches[query]) }),
          },
        };
        vm.createContext(ctx);
        vm.runInContext(__SNIPPET__, ctx);
        process.stdout.write(JSON.stringify(ctx.__test_useTouchFileEditorControls()));
        """,
        QUERY_MATCHES=json.dumps(query_matches),
        SNIPPET=json.dumps(snippet + "\nglobalThis.__test_useTouchFileEditorControls = useTouchFileEditorControls;\n"),
    )
    return run_node_json(js)


def eval_file_open_request_sequence() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("let fileOpenRequestId = 0;")
    end = source.index("function rememberActiveFileSelection(", start)
    snippet = source[start:end]
    js = render_js(
        """
        const vm = require("vm");
        class AbortController {
          constructor() {
            this.signal = { aborted: false };
          }
          abort() {
            this.signal.aborted = true;
          }
        }
        const ctx = {
          activeFilePath: "old.txt",
          activeFileLine: 1,
          fileViewerSessionId: "sid-1",
          selected: "",
          AbortController,
          normalizeLineNumber: (value) => value == null ? null : Number(value),
        };
        vm.createContext(ctx);
        vm.runInContext(__SNIPPET__, ctx);
        const first = ctx.__test_file_open.beginFileOpenRequest("first.txt", { line: 3 });
        const firstCurrent = ctx.__test_file_open.isCurrentFileOpenRequest(first);
        const second = ctx.__test_file_open.beginFileOpenRequest("second.txt", { line: 8 });
        const result = {
          currentSessionId: ctx.__test_file_open.currentFileSessionId(),
          firstCurrent,
          firstSignalAborted: Boolean(first.signal && first.signal.aborted),
          firstAfterSecond: ctx.__test_file_open.isCurrentFileOpenRequest(first),
          secondCurrent: ctx.__test_file_open.isCurrentFileOpenRequest(second),
          activeFilePath: ctx.activeFilePath,
          activeFileLine: ctx.activeFileLine,
        };
        ctx.__test_file_open.cancelPendingFileOpen();
        result.secondAfterCancel = ctx.__test_file_open.isCurrentFileOpenRequest(second);
        process.stdout.write(JSON.stringify(result));
        """,
        SNIPPET=json.dumps(
            snippet
            + "\nglobalThis.__test_file_open = { beginFileOpenRequest, isCurrentFileOpenRequest, cancelPendingFileOpen, currentFileSessionId };\n"
        ),
    )
    return run_node_json(js)


def eval_refresh_file_candidates_race() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("async function refreshFileCandidates() {")
    end = source.index("async function showFileViewer(", start)
    snippet = source[start:end]
    js = render_js(
        """
        const vm = require("vm");
        const pending = new Map();
        const ctx = {
          Map,
          Set,
          fileCandidateList: [],
          fileEntryMap: new Map(),
          fileViewerSessionId: "sid-a",
          selected: "",
          sessionIndex: new Map([
            ["sid-a", { files: [] }],
            ["sid-b", { files: [] }],
          ]),
          listFromFilesField: (value) => Array.isArray(value) ? value : [],
          sessionRelativePath: () => null,
          collectMessageFileRefs: () => [],
          renderFilePickerMenu: () => {},
          currentFileSessionId: () => String(ctx.fileViewerSessionId || ctx.selected || "").trim(),
          api: (url) => {
            const parts = String(url).split("/");
            const sid = parts.length > 3 ? parts[3] : "";
            return new Promise((resolve) => pending.set(sid, resolve));
          },
        };
        vm.createContext(ctx);
        vm.runInContext(__SNIPPET__, ctx);
        (async () => {
          const first = ctx.__test_refresh.refreshFileCandidates();
          ctx.fileViewerSessionId = "sid-b";
          const second = ctx.__test_refresh.refreshFileCandidates();
          pending.get("sid-b")({ entries: [{ path: "b.txt" }] });
          await second;
          pending.get("sid-a")({ entries: [{ path: "a.txt" }] });
          await first;
          process.stdout.write(JSON.stringify({
            paths: ctx.fileCandidateList,
            hasA: ctx.fileEntryMap.has("a.txt"),
            hasB: ctx.fileEntryMap.has("b.txt"),
          }));
        })().catch((error) => {
          console.error(error);
          process.exit(1);
        });
        """,
        SNIPPET=json.dumps(snippet + "\nglobalThis.__test_refresh = { refreshFileCandidates };\n"),
    )
    return run_node_json(js)


def eval_open_draft_file_session_race() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("async function openDraftFilePathWithGuard(path) {")
    end = source.index("async function setFileViewModeWithGuard(", start)
    snippet = source[start:end]
    js = render_js(
        """
        const vm = require("vm");
        let resolveInspect;
        const setViewCalls = [];
        const setPathCalls = [];
        const openDraftCalls = [];
        const ctx = {
          fileViewerSessionId: "sid-a",
          selected: "sid-a",
          fileStatus: { textContent: "" },
          normalizeDraftFilePath: (value) => String(value || "").trim(),
          maybeHandleUnsavedFileChanges: async () => true,
          inspectSessionFilePath: () => new Promise((resolve) => {
            resolveInspect = resolve;
          }),
          openFilePathWithGuard: async () => false,
          setFileViewMode: (mode) => setViewCalls.push(mode),
          setFilePath: (path, options) => setPathCalls.push({ path, options }),
          renderFilePickerMenu: () => {},
          openDraftFilePath: async (path, options) => {
            openDraftCalls.push({ path, options });
          },
          currentFileSessionId: () => String(ctx.fileViewerSessionId || ctx.selected || "").trim(),
        };
        vm.createContext(ctx);
        vm.runInContext(__SNIPPET__, ctx);
        (async () => {
          const pending = ctx.__test_draft.openDraftFilePathWithGuard("draft.txt");
          await new Promise((resolve) => setImmediate(resolve));
          ctx.fileViewerSessionId = "sid-b";
          ctx.selected = "sid-b";
          resolveInspect({ exists: false });
          const returned = await pending;
          process.stdout.write(JSON.stringify({
            returned,
            setViewCalls,
            setPathCalls,
            openDraftCalls,
            statusText: ctx.fileStatus.textContent,
          }));
        })().catch((error) => {
          console.error(error);
          process.exit(1);
        });
        """,
        SNIPPET=json.dumps(snippet + "\nglobalThis.__test_draft = { openDraftFilePathWithGuard };\n"),
    )
    return run_node_json(js)


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

    def test_refresh_file_candidates_ignores_stale_session_results(self) -> None:
        result = eval_refresh_file_candidates_race()
        self.assertEqual(result["paths"], ["b.txt"])
        self.assertFalse(result["hasA"])
        self.assertTrue(result["hasB"])

    def test_open_draft_file_guard_ignores_stale_inspect_results(self) -> None:
        result = eval_open_draft_file_session_race()
        self.assertFalse(result["returned"])
        self.assertEqual(result["setViewCalls"], [])
        self.assertEqual(result["setPathCalls"], [])
        self.assertEqual(result["openDraftCalls"], [])

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
        self.assertIn('if (e && e.pointerType === "touch") sawPointerTouchAt = Date.now();', source)
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

    def test_file_open_race_guard_is_wired_through_fetch_and_render(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("let fileOpenRequestId = 0;", source)
        self.assertIn("let fileOpenAbortController = null;", source)
        self.assertIn("function cancelPendingFileOpen()", source)
        self.assertIn("const request = beginFileOpenRequest(nextPath, { line });", source)
        self.assertIn("signal: request.signal", source)
        self.assertIn("if (!isCurrentFileOpenRequest(request)) return false;", source)
        self.assertIn('async function renderMonacoFile(rel, text, lineNumber = null, langOverride = "", request = null)', source)
        self.assertIn('async function renderMonacoDiff(rel, originalText, modifiedText, lineNumber = null, request = null)', source)
        self.assertIn("if (request && !isCurrentFileOpenRequest(request)) return false;", source)
        self.assertIn('cancelPendingFileOpen();\n          fileBackdrop.style.display = "block";', source)
        self.assertIn('cancelPendingFileOpen();\n          hideFileUnsavedDialog();', source)

    def test_touch_paste_tries_direct_clipboard_before_bridge(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("navigator.clipboard", source)
        self.assertIn("readText", source)
        self.assertIn('setToast("pasted")', source)
        self.assertIn('filePasteDialog.style.display = "flex";', source)


if __name__ == "__main__":
    unittest.main()
