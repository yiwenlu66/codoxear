import json
import subprocess
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "codoxear" / "static" / "app_file_viewer.js"


def run_vm(body: str) -> dict:
    source = json.dumps(VIEWER.read_text(encoding="utf-8"))
    script = f"""
const vm = require('vm');
const ctx = {{ window: {{}}, console }};
vm.createContext(ctx);
vm.runInContext({source}, ctx);
{body}
"""
    proc = subprocess.run(["node", "-e", textwrap.dedent(script)], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
const runtime = module.createFileInspectRuntime({ currentSessionId: () => sid, api: async (url, options) => { calls.push([url, options.body]); if (missing) { const e = new Error('missing'); e.status = 404; throw e; } return { kind: 'text', path: '/repo/' + options.body.path }; } });
(async () => { const plain = await runtime.inspectSessionFilePath('a.py'); const git = await runtime.inspectSessionFilePath('b.py', { gitPath: true, apiPath: 'raw-token' }); missing = true; const absent = await runtime.inspectSessionFilePath('gone.py'); process.stdout.write(JSON.stringify({ plain, git, absent, calls })); })().catch((e) => { console.error(e); process.exit(1); });
'''
        )
        self.assertEqual(result["plain"], {"kind": "text", "path": "/repo/a.py"})
        self.assertEqual(result["git"], {"kind": "text", "path": "/repo/b.py"})
        self.assertEqual(result["absent"], {"exists": False})
        self.assertEqual(result["calls"][1], ["/api/files/inspect", {"session_id": "s1", "path": "b.py", "git_path": True, "path_token": "raw-token"}])

    def test_mode_controls_and_touch_toolbar_follow_active_file_capabilities(self) -> None:
        result = run_vm(
            r'''
const module = ctx.window.CodoxearFileViewer; const node = () => ({ style: {}, disabled: false, attrs: {}, classList: { toggle(name, on) { this[name] = on; } }, setAttribute(k, v) { this.attrs[k] = v; } });
const diff = node(), preview = node(), download = node(), video = node();
const controls = module.createFileModeControlsRuntime({ fileDiffBtn: diff, filePreviewBtn: preview, fileDownloadBtn: download, fileVideoPreviewButton: video, fileModeState: () => ({ diffActive: true, previewActive: false, diffDisabled: false, previewDisabled: true, downloadDisabled: false, videoPreviewVisible: true, videoPreviewDisabled: false, videoPreviewTitle: 'Convert preview', shouldHidePasteDialog: false, shouldExitEditMode: false }), hideFilePasteDialog() {}, setFileEditMode() {}, syncFileEditorReadOnly() {}, updateFileEditButton() {} });
controls.apply();
const toolbar = node(), dpad = node(), select = node(), copy = node(), paste = node(), actions = node();
const touch = module.createFileTouchToolbarRuntime({ fileTouchToolbar: toolbar, fileTouchDpad: dpad, fileTouchSelectButton: select, fileTouchCopyBtn: copy, fileTouchPasteBtn: paste, fileTouchActions: actions, state: () => ({ visible: true, selectActive: true, dpadVisible: true, copyVisible: true, pasteVisible: false }) });
touch.update();
process.stdout.write(JSON.stringify({ controls: { diffDisabled: diff.disabled, previewDisabled: preview.disabled, video: video.style.display, title: video.attrs['aria-label'] }, touch: { toolbar: toolbar.style.display, dpad: dpad.style.display, copy: copy.style.display, active: select.classList.active } }));
'''
        )
        self.assertEqual(result["controls"], {"diffDisabled": False, "previewDisabled": True, "video": "", "title": "Convert preview"})
        self.assertEqual(result["touch"], {"toolbar": "flex", "dpad": "grid", "copy": "", "active": True})


if __name__ == "__main__":
    unittest.main()
