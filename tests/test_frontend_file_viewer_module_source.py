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

    def test_file_inspection_routes_tokens_and_treats_404_as_missing(self) -> None:
        result = run_vm(
            r'''
const module = ctx.window.CodoxearFileViewer; const calls = []; let sid = 's1'; let missing = false;
const runtime = module.createFileInspectRuntime({ currentSessionId: () => sid, selectedSessionId: () => '', normalizeFileApiPath: (value) => String(value || ''), api: async (url, options) => { calls.push([url, options.body]); if (missing) { const e = new Error('missing'); e.status = 404; throw e; } return { kind: 'text', path: '/repo/' + options.body.path }; } });
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
const runtime = module.createFileReferenceRuntime({ selectedSessionId: () => 's1', sessionById: () => null, chatRoot: null, sessionRelativePath: (path) => path, listFromFilesField: () => [], listFromFileRecords: () => [], normalizeFileApiPath: () => '', normalizeLineNumber: () => null, parseLocalFileRef: () => null, showFileViewer: async () => true, selectSession: async () => true, openDirectorySession: () => true, setToast: () => {}, api: async (url, options) => { calls.push([url, options.body]); return { results: [{ path: 'src/present.py', exists: true, resolved_path: '/repo/src/present.py', kind: 'text' }, { path: 'src/gone.py', exists: false }] }; }, el: (tag, attrs) => ({ tag, attrs, setAttribute(name, value) { this.attrs[name] = value; }, appendChild() {} }) });
(async () => { await runtime.upgradeCandidateRefs({ querySelectorAll: () => nodes }); process.stdout.write(JSON.stringify({ calls, linked: nodes[0].replacement && nodes[0].replacement.attrs['data-file-path'], missingLinked: Boolean(nodes[1].replacement) })); })().catch((e) => { console.error(e); process.exit(1); });
'''
        )
        self.assertEqual(result["calls"], [["/api/files/inspect-batch", {"session_id": "s1", "paths": ["src/present.py", "src/gone.py"]}]])
        self.assertEqual(result["linked"], "/repo/src/present.py")
        self.assertFalse(result["missingLinked"])

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


if __name__ == "__main__":
    unittest.main()
