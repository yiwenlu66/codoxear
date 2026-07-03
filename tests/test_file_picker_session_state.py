import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_FILE_VIEWER_JS = ROOT / "codoxear" / "static" / "app_file_viewer.js"


def eval_file_picker_session_helpers() -> dict[str, object]:
    source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, AbortController }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const sessionIndex = new Map([
          ["session-a", {{ cwd: "/project-A", files: ["/project-A/file-a.py"] }}],
          ["session-b", {{ cwd: "/project-B", files: ["/project-B/file-b.py"] }}],
        ]);
        function historyFileSelectionForSession(sessionId) {{
          const session = sessionIndex.get(String(sessionId || ""));
          if (!session || !session.cwd || !Array.isArray(session.files) || !session.files.length) return {{ path: "", line: null, gitPath: false }};
          const cwd = String(session.cwd || "").replace(/\\/+$/, "");
          const abs = String(session.files[0] || "").trim();
          const path = abs.startsWith(cwd + "/") ? abs.slice(cwd.length + 1) : "";
          return {{ path, line: null, gitPath: false }};
        }}
        const fileStatus = {{ textContent: "", replaceChildren() {{}} }};
        const fileEditButton = {{ classList: {{ toggle() {{}} }}, setAttribute() {{}}, disabled: false }};
        const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
          el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children }}),
          fileStatus,
          fileEditButton,
          iconSvg: (name) => name,
          currentSessionId: () => "session-a",
          currentFileSessionId: () => controller.currentFileViewerSessionId() || "session-a",
          normalizeLineNumber: (value) => {{
            if (value == null || value === "") return null;
            const n = Number(value);
            return Number.isFinite(n) && n >= 1 ? Math.floor(n) : null;
          }},
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
          isMarkdownPreviewable: () => true,
          updateFileTouchToolbar: () => {{}},
          useTouchFileEditorControls: () => false,
          hasActiveFileCodeEditor: () => false,
          hasBlockingFileEditorModal: () => false,
          isTextEntryTarget: () => false,
          eventTargetElement: (value) => value || null,
          normalizeFileEditorPosition: (_editor, position) => position || null,
          applyFileEditorSelection: () => {{}},
          isCollapsedFileSelection: () => true,
          positionAfterInsertedText: (start, text) => ({{ lineNumber: start.lineNumber, column: start.column + String(text || "").length }}),
          fileEditorEditSupportAvailable: () => true,
          updateFileDiffEditorOptions: () => {{}},
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
          setToast: () => {{}},
          renderMonacoFile: async () => true,
          getFileEditorText: () => "",
          fmtBytes: (value) => `${{value}}B`,
          applyFileMode: () => {{}},
          rememberOpenedFile: () => {{}},
          historyFileSelectionForSession,
          renderFilePickerMenu: () => {{}},
        }});
        controller.setFileViewerSessionId("session-a");
        controller.setActiveFileIdentity("file-a.py", {{ line: 7, gitPath: true, apiPath: "token-a" }});
        controller.rememberActiveFileSelection();
        process.stdout.write(JSON.stringify({{
          sessionA: controller.preferredFileSelectionForSession("session-a"),
          sessionB: controller.preferredFileSelectionForSession("session-b"),
        }}));
        """
    )
    proc = subprocess.run(
        ["node"],
        input=js,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


class TestFilePickerSessionState(unittest.TestCase):
    def test_preferred_file_selection_is_session_scoped(self) -> None:
        result = eval_file_picker_session_helpers()
        self.assertEqual(result["sessionA"], {"path": "file-a.py", "apiPath": "token-a", "line": 7, "gitPath": True})
        self.assertEqual(result["sessionB"], {"path": "file-b.py", "line": None, "gitPath": False})

    def test_global_file_path_local_storage_is_not_used(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function ensureCurrentFileViewerSession()", source)
        self.assertNotIn('localStorage.getItem("codexweb.filePath")', source)
        self.assertNotIn('localStorage.setItem("codexweb.filePath"', source)


if __name__ == "__main__":
    unittest.main()
