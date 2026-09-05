from frontend_module_loader import module_path
import json
import subprocess
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER_SCRIPTS = [
    module_path(name)
    for name in (
        "app_file_candidates.js", "app_file_candidate_state.js", "app_file_viewer_operations.js", "app_file_download.js", "app_file_viewer_lifecycle.js", "app_file_viewer_panel.js",
        "app_file_unsaved_dialog.js", "app_file_paste_dialog.js", "app_file_pdf.js", "app_file_video.js",
        "app_file_mode.js", "app_file_render_surface.js", "app_file_viewer_controller.js", "app_file_viewer.js",
    )
]


def run_vm(body: str) -> dict:
    sources = json.dumps([path.read_text(encoding="utf-8") for path in VIEWER_SCRIPTS])
    script = f"""
const vm = require('vm');
const ctx = {{ window: {{}}, console }};
vm.createContext(ctx);
{sources}.forEach((source) => vm.runInContext(source, ctx));
{body}
"""
    proc = subprocess.run(["node"], input=textwrap.dedent(script), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(proc.stdout)


class TestFrontendFileViewerModuleBehavior(unittest.TestCase):
    def test_controller_fails_loudly_for_missing_dependencies(self) -> None:
        result = run_vm(
            """
const module = ctx.window.CodoxearFileViewer;
let error = ''; try { module.createFileViewerController({}); } catch (e) { error = e.message; }
process.stdout.write(JSON.stringify({ frozen: Object.isFrozen(module), error }));
"""
        )
        self.assertTrue(result["frozen"])
        self.assertEqual(result["error"], "file viewer dependency missing: el")

    def test_controller_requires_operations_wiring_after_base_dependencies(self) -> None:
        result = run_vm(
            """
const module = ctx.window.CodoxearFileViewer;
const noop = () => {};
const required = new Proxy({
  el: noop,
  fileStatus: { replaceChildren() {} },
  fileEditButton: { classList: { toggle() {} }, setAttribute() {} },
  iconSvg: noop,
}, { get: (target, key) => key in target ? target[key] : noop });
let error = ''; try { module.createFileViewerController(required); } catch (e) { error = e.message; }
process.stdout.write(JSON.stringify({ error }));
"""
        )
        self.assertEqual(result["error"], "file viewer dependency missing: wiring")

    def test_file_inspection_routes_tokens_and_treats_404_as_missing(self) -> None:
        result = run_vm(
            r'''
const module = ctx.window.CodoxearFileViewer; const calls = []; let sid = 's1'; let missing = false;
const runtime = module.createFileInspectRuntime({ currentSessionId: () => sid, sessionState: { get: () => '' }, normalizeFileApiPath: (value) => String(value || ''), api: async (url, options) => { calls.push([url, options.body]); if (missing) { const e = new Error('missing'); e.status = 404; throw e; } return { kind: 'text', path: '/repo/' + options.body.path }; } });
(async () => { const plain = await runtime.inspectSessionFilePath('a.py'); const git = await runtime.inspectSessionFilePath('b.py', { gitPath: true, apiPath: 'raw-token' }); missing = true; const absent = await runtime.inspectSessionFilePath('gone.py'); process.stdout.write(JSON.stringify({ plain, git, absent, calls })); })().catch((e) => { console.error(e); process.exit(1); });
'''
        )
        self.assertEqual(result["plain"], {"exists": True, "kind": "text", "path": "/repo/a.py"})
        self.assertEqual(result["git"], {"exists": True, "kind": "text", "path": "/repo/b.py"})
        self.assertEqual(result["absent"], {"exists": False})
        self.assertEqual(result["calls"][1], ["/api/files/inspect", {"session_id": "s1", "path": "b.py", "git_path": True, "path_token": "raw-token"}])

    def test_candidate_reference_upgrade_batches_direct_paths_and_skips_missing_files(self) -> None:
        result = run_vm(
            r'''
const module = ctx.window.CodoxearFileViewer; const calls = [];
function candidate(path) { return { textContent: path, getAttribute(name) { return name === 'data-candidate-file-path' ? path : null; }, replaceWith(next) { this.replacement = next; } }; }
const nodes = [candidate('src/present.py'), candidate('src/gone.py')];
const runtime = module.createFileReferenceRuntime({ sessionState: { get: () => 's1' }, sessionById: () => null, chatRoot: null, sessionRelativePath: (path) => path, listFromFilesField: () => [], listFromFileRecords: () => [], normalizeFileApiPath: () => '', normalizeLineNumber: () => null, parseLocalFileRef: () => null, showFileViewer: async () => true, selectSession: async () => true, openDirectorySession: () => true, setToast: () => {}, api: async (url, options) => { calls.push([url, options.body]); return { results: [{ path: 'src/present.py', exists: true, resolved_path: '/repo/src/present.py', kind: 'text' }, { path: 'src/gone.py', exists: false }] }; }, el: (tag, attrs) => ({ tag, attrs, setAttribute(name, value) { this.attrs[name] = value; }, appendChild() {} }) });
(async () => { await runtime.upgradeCandidateRefs({ querySelectorAll: () => nodes }); process.stdout.write(JSON.stringify({ calls, linked: nodes[0].replacement && nodes[0].replacement.attrs['data-file-path'], missingLinked: Boolean(nodes[1].replacement) })); })().catch((e) => { console.error(e); process.exit(1); });
'''
        )
        self.assertEqual(result["calls"], [["/api/files/inspect-batch", {"session_id": "s1", "paths": ["src/present.py", "src/gone.py"]}]])
        self.assertEqual(result["linked"], "/repo/src/present.py")
        self.assertFalse(result["missingLinked"])

    def test_touch_editor_runtime_owns_selection_and_edit_projection(self) -> None:
        source = json.dumps(module_path("app_file_ops.js").read_text(encoding="utf-8"))
        script = f'''
const vm = require('vm');
const ctx = {{ window: {{}}, console }};
vm.createContext(ctx);
vm.runInContext({source}, ctx);
let cursor = {{ lineNumber: 2, column: 3 }};
let selection = {{ startLineNumber: 2, startColumn: 3, endLineNumber: 2, endColumn: 3 }};
let editorText = 'before';
const calls = [];
const editor = {{
  getPosition: () => cursor,
  getSelection: () => selection,
  trigger(source, command, args) {{ calls.push(['trigger', source, command, args]); }},
  executeEdits(source, edits) {{ calls.push(['edit', source, edits]); editorText = edits[0].text; }},
  pushUndoStop() {{ calls.push(['undo']); }},
}};
const runtime = ctx.window.CodoxearFileTouch.createFileTouchController({{
  isFileViewerOpen: () => true,
  isTextFileKind: () => true,
  focusEditor: () => editor,
  updateFileTouchToolbar: () => calls.push(['toolbar']),
  useTouchFileEditorControls: () => true,
  hasActiveFileCodeEditor: () => true,
  fileEditorShortcutBlocked: () => false,
  normalizeFileEditorPosition: (_editor, value) => value,
  applyFileEditorSelection: (_editor, nextCursor, anchor) => {{ cursor = nextCursor; calls.push(['selection', nextCursor, anchor]); }},
  isCollapsedFileSelection: (value) => value.startLineNumber === value.endLineNumber && value.startColumn === value.endColumn,
  positionAfterInsertedText: (start, text) => ({{ lineNumber: start.lineNumber, column: start.column + String(text).length }}),
  fileEditorEditSupportAvailable: () => true,
  updateFileDiffEditorOptions: (value) => calls.push(['diff-options', value]),
  showFilePasteDialog: () => true,
  hideFilePasteDialog: () => calls.push(['hide-paste']),
  clipboardReadAvailable: () => true,
  readClipboardText: async () => 'clip',
  fileEditorDeleteCommandForKey: (key) => key === 'backspace' ? 'deleteLeft' : '',
  isActiveFileEditorInput: () => true,
  getActiveFileSelectionText: () => 'selected',
  copyToClipboard: async (text) => calls.push(['copy', text]),
  focusActiveFileCodeEditor: () => calls.push(['focus']),
  nowMs: () => 1000,
  setToast: (text) => calls.push(['toast', text]),
  getFileEditorText: () => editorText,
  currentFileViewMode: () => 'file',
  currentActiveFileKind: () => 'text',
  currentActiveFileText: () => 'before',
  activeFileEditorWritable: () => true,
  activeFileEditorIdleWritable: () => true,
  activeFileEditorIdleTextWritable: () => true,
  blockUnavailableFileAction: () => false,
  eventTargetElement: (value) => value,
  syncFileEditorReadOnly: () => calls.push(['sync-readonly']),
  setFileDirty: (dirty) => calls.push(['dirty', dirty]),
}});
runtime.toggleFileTouchSelectionMode();
const selectedState = runtime.currentFileTouchToolbarState();
const inserted = runtime.insertIntoActiveFileEditor('new');
const resetState = runtime.currentFileTouchToolbarState();
process.stdout.write(JSON.stringify({{
  frozen: Object.isFrozen(runtime),
  api: Object.keys(runtime).sort(),
  selectedState,
  inserted,
  resetState,
  calls,
}}));
'''
        proc = subprocess.run(["node"], input=textwrap.dedent(script), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = json.loads(proc.stdout)
        self.assertTrue(result["frozen"])
        self.assertEqual(
            result["api"],
            sorted([
                "clearFileTouchSelectionState", "currentFileTouchSelectMode", "currentFileTouchToolbarState",
                "resetFileTouchSelectionState", "toggleFileTouchSelectionMode", "handleFileTouchMoveButtonPress",
                "moveFileTouchSelection", "handleFileTouchSelectionKeydown", "handleFileEditorDeleteKeydown",
                "suppressFileEditorNativeDelete", "insertIntoActiveFileEditor", "pasteFromClipboardIntoActiveFile",
                "handleFilePasteInsert", "copyActiveFileSelection",
            ]),
        )
        self.assertEqual(result["selectedState"], {"visible": True, "selectActive": True, "dpadVisible": True, "copyVisible": True, "pasteVisible": True})
        self.assertTrue(result["inserted"])
        self.assertEqual(result["resetState"]["selectActive"], False)
        self.assertIn(["sync-readonly"], result["calls"])
        self.assertIn(["dirty", True], result["calls"])

    def test_save_conflict_rendering_tracks_current_file_identity(self) -> None:
        result = run_vm(
            r'''
const module = ctx.window.CodoxearFileViewerOperations;
let sessionId = 'session-1';
let identity = { path: 'notes.txt', gitPath: false, apiPath: '' };
let unavailable = false;
const created = [];
const el = (tag, attrs = {}, children = []) => {
  const node = { tag, attrs, children, onclick: null };
  created.push(node);
  return node;
};
const fileStatus = {
  children: [],
  replaceChildren(...children) { this.children = children; },
};
const runtime = module.createFileViewerOperationsRuntime({
  el,
  fileStatus,
  currentSessionId: () => sessionId,
  currentActiveFileIdentity: () => identity,
  isUnavailable: () => unavailable,
});
const conflictIdentity = { path: 'notes.txt', gitPath: false, apiPath: 'raw-token' };
identity = conflictIdentity;
const conflict = runtime.renderSaveConflict('session-1', conflictIdentity, 'disk changed');
const current = runtime.isSaveConflictCurrent(conflict);
identity = { path: 'notes.txt', gitPath: false, apiPath: 'other-token' };
const afterApiPathChange = runtime.isSaveConflictCurrent(conflict);
identity = { path: 'notes.txt', gitPath: true, apiPath: 'raw-token' };
const afterGitPathChange = runtime.isSaveConflictCurrent(conflict);
identity = { path: 'other.txt', gitPath: false, apiPath: 'raw-token' };
const afterPathChange = runtime.isSaveConflictCurrent(conflict);
identity = conflictIdentity;
sessionId = 'session-2';
const afterSessionChange = runtime.isSaveConflictCurrent(conflict);
sessionId = 'session-1';
unavailable = true;
const whileUnavailable = runtime.isSaveConflictCurrent(conflict);
process.stdout.write(JSON.stringify({
  conflict,
  frozen: Object.isFrozen(conflict),
  current,
  afterApiPathChange,
  afterGitPathChange,
  afterPathChange,
  afterSessionChange,
  whileUnavailable,
  statusText: fileStatus.children[0].attrs.text,
  actions: fileStatus.children[1].children.map((button) => button.attrs.text),
  handlers: fileStatus.children[1].children.map((button) => typeof button.onclick),
}));
'''
        )
        self.assertEqual(result["conflict"], {
            "sessionId": "session-1",
            "path": "notes.txt",
            "gitPath": False,
            "apiPath": "raw-token",
        })
        self.assertTrue(result["frozen"])
        self.assertTrue(result["current"])
        self.assertFalse(result["afterApiPathChange"])
        self.assertFalse(result["afterGitPathChange"])
        self.assertFalse(result["afterPathChange"])
        self.assertFalse(result["afterSessionChange"])
        self.assertFalse(result["whileUnavailable"])
        self.assertEqual(result["statusText"], "notes.txt - save conflict: disk changed")
        self.assertEqual(result["actions"], ["Reload from disk", "Keep editing"])
        self.assertEqual(result["handlers"], ["function", "function"])

    def test_mode_controls_and_touch_toolbar_follow_active_file_capabilities(self) -> None:
        result = run_vm(
            r'''
const module = ctx.window.CodoxearFileViewer; const node = () => ({ style: {}, disabled: false, attrs: {}, classList: { toggle(name, on) { this[name] = on; } }, setAttribute(k, v) { this.attrs[k] = v; } });
const diff = node(), preview = node(), download = node(), video = node();
const mode = { diffActive: true, previewActive: false, diffDisabled: false, previewDisabled: true, downloadDisabled: false, videoPreviewVisible: true, videoPreviewDisabled: false, videoPreviewTitle: 'Convert preview', markdownPreviewVisible: true, shouldHidePasteDialog: false, shouldExitEditMode: false };
const controls = module.createFileModeControlsRuntime({ diffButton: diff, previewButton: preview, downloadButton: download, videoPreviewButton: video, hideFilePasteDialog() {}, setFileEditMode() {}, syncFileEditorReadOnly() {}, updateFileEditButton() {} });
controls.apply(mode);
const toolbar = node(), dpad = node(), select = node(), copy = node(), paste = node(), actions = node();
const touch = module.createFileTouchToolbarRuntime({ toolbar, dpad, selectButton: select, copyButton: copy, pasteButton: paste, actions });
touch.update({ visible: true, selectActive: true, dpadVisible: true, copyVisible: true, pasteVisible: false });
process.stdout.write(JSON.stringify({ controls: { diffDisabled: diff.disabled, previewDisabled: preview.disabled, video: video.style.display, title: video.attrs['aria-label'] }, touch: { toolbar: toolbar.style.display, dpad: dpad.style.display, copy: copy.style.display, active: select.classList.active } }));
'''
        )
        self.assertEqual(result["controls"], {"diffDisabled": False, "previewDisabled": True, "video": "", "title": "Convert preview"})
        self.assertEqual(result["touch"], {"toolbar": "flex", "dpad": "grid", "copy": "", "active": True})

    def test_ctrl_s_saves_from_vim_normal_mode(self) -> None:
        # Full chain through the real viewer controller with the vim normal
        # gate active: capabilities must expose the gate-ignoring idle-text
        # writable variant, and Ctrl-S must save while the vim-gated variant
        # stays false (the vim layer does not consume Ctrl-S).
        result = run_vm(
            r'''
const module = ctx.window.CodoxearFileViewer;
const noop = () => {};
const apiCalls = [];
const toasts = [];
const overrides = {
  fileStatus: { textContent: "", replaceChildren() {} },
  fileEditButton: { disabled: false, innerHTML: "", title: "", classList: { toggle() {} }, setAttribute() {}, removeAttribute() {} },
  iconSvg: () => "",
  currentSessionId: () => "s1",
  isFileViewerOpen: () => true,
  hasBlockingFileEditorModal: () => false,
  isTextFileKind: (kind) => kind === "text",
  isTextEntryTarget: () => false,
  isActiveFileEditorInput: () => false,
  eventTargetElement: (value) => value,
  initialFileViewMode: "file",
  getFileEditorText: () => "edited",
  api: async (url, options) => { apiCalls.push([url, { method: options.method, body: options.body }]); return { version: "v2", editable: true, size: 6 }; },
  applyFileMode: noop,
  fmtBytes: (value) => `${value} B`,
  rememberOpenedFile: noop,
  renderFilePickerMenu: noop,
  updateFileTouchToolbar: noop,
  focusEditor: () => null,
  setToast: (message) => toasts.push(message),
  vimNormalActive: () => true,
  wiring: { createFileViewerOperationsOptions: (options) => options },
};
const required = new Proxy(overrides, { get: (target, key) => (key in target ? target[key] : noop) });
const controller = module.createFileViewerController(required);
controller.setActiveFileIdentity("notes.txt", {});
controller.applyActiveFileTextState({ kind: "text", text: "edited", editable: true, version: "v1" });
controller.setFileEditMode(true);
controller.setFileDirty(true);
const gates = {
  idleTextWritable: controller.activeFileEditorIdleTextWritable(),
  insertIdleTextWritable: controller.activeFileEditorInsertIdleTextWritable(),
  editMode: controller.currentFileEditMode(),
};
const ctrlS = { key: "s", ctrlKey: true, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, stopPropagation() {} };
const accepted = controller.handleFileEditorSaveShortcut(ctrlS);
const saveShortcutResult = { accepted, consumed: ctrlS.defaultPrevented };
const nonSave = { key: "x", ctrlKey: true, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, stopPropagation() {} };
const nonSaveAccepted = controller.handleFileEditorSaveShortcut(nonSave);
(async () => {
  await new Promise((resolve) => setTimeout(resolve, 0));
  process.stdout.write(JSON.stringify({ gates, saveShortcutResult, nonSaveAccepted, apiCalls, toasts }));
})();
'''
        )
        # With vim normal mode owning the keys the gated variant is false,
        # the gate-ignoring variant is true, and the save runs anyway.
        self.assertFalse(result["gates"]["idleTextWritable"])
        self.assertTrue(result["gates"]["insertIdleTextWritable"])
        self.assertTrue(result["gates"]["editMode"])
        self.assertTrue(result["saveShortcutResult"]["accepted"])
        self.assertTrue(result["saveShortcutResult"]["consumed"])
        self.assertFalse(result["nonSaveAccepted"])
        self.assertEqual(result["apiCalls"], [
            ["/api/sessions/s1/file/write", {"method": "POST", "body": {"path": "notes.txt", "text": "edited", "version": "v1", "git_path": False}}],
        ])
        self.assertEqual(result["toasts"], [])


if __name__ == "__main__":
    unittest.main()
