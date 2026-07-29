import json
import subprocess
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "codoxear" / "static" / "app_file_viewer.js"


def run_viewer(js: str) -> dict:
    source = json.dumps(VIEWER.read_text(encoding="utf-8"))
    program = f"""
const vm = require('vm');
const ctx = {{ window: {{}}, console }};
vm.createContext(ctx);
vm.runInContext({source}, ctx);
{js}
"""
    proc = subprocess.run(["node", "-e", textwrap.dedent(program)], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(proc.stdout)


class TestFileViewerBehavior(unittest.TestCase):
    def test_module_routes_preview_surfaces_downloads_and_binary_fallbacks(self) -> None:
        result = run_viewer(
            r'''
const viewer = ctx.window.CodoxearFileViewer;
const makeSurface = () => ({ style: {} });
const surfaces = { diff: makeSurface(), image: makeSurface(), video: makeSurface() };
const image = { style: {}, removeAttribute() {}, src: '', alt: '' };
const video = { style: {}, removeAttribute() {}, load() {}, pause() {}, src: '' };
const surface = viewer.createFileRenderSurfaceRuntime({ fileDiff: surfaces.diff, fileImage: image, fileVideo: video });
surface.setSurface('diff'); const diff = { ...surfaces.diff.style, image: surfaces.image.style.display, video: surfaces.video.style.display };
surface.setSurface('image'); const imageState = { ...surfaces.image.style, diff: surfaces.diff.style.display, video: surfaces.video.style.display };
surface.setSurface('video'); const videoState = { ...surfaces.video.style, diff: surfaces.diff.style.display, image: surfaces.image.style.display };
let download = [];
const document = { createElement() { return { style: {}, set href(v) { this._href = v; }, get href() { return this._href; }, click() { download.push(this._href); }, remove() {} }; }, body: { appendChild() {} } };
const downloadRuntime = viewer.createFileDownloadRuntime({ document });
const host = { innerHTML: 'old', scrollTop: 22, appendChild(node) { this.node = node; } };
const fallback = viewer.createFileFallbackRuntime({ host, document: { createElement(tag) { return { tagName: tag, className: '', textContent: '', appendChild(child) { this.child = child; } }; }, createTextNode(text) { return { textContent: text }; } }, requestAnimationFrame(fn) { fn(); }, disposeFileEditor() {}, disposePdfRender() {}, clearFileVideo() {}, setFileRenderSurface() {}, setFileEditorKind() {}, applyActiveFileTextState() {}, updateFileTouchToolbar() {}, markdownPreviewHtml() { return '<p>x</p>'; }, renderMarkdownPreview() {} });
const plain = fallback.applyPlainText('notes.txt', 'hello', 3);
fallback.applyBlocked('blob.bin', 'binary', 0, 3);
const blocked = host.node.textContent;
downloadRuntime.download('app:/download');
process.stdout.write(JSON.stringify({ exports: Object.keys(viewer).sort(), diff, imageState, videoState, download, plain, blocked }));
'''
        )
        self.assertIn("createFileViewerController", result["exports"])
        self.assertEqual(result["diff"], {"display": "block", "image": "none", "video": "none"})
        self.assertEqual(result["imageState"], {"display": "block", "diff": "none", "video": "none"})
        self.assertEqual(result["videoState"], {"display": "block", "diff": "none", "image": "none"})
        self.assertEqual(result["download"], ["app:/download"])
        self.assertEqual(result["plain"], {"targetLine": 3})
        self.assertIn("not renderable", result["blocked"])

    def test_modal_and_touch_dpad_behave_as_interaction_boundaries(self) -> None:
        result = run_viewer(
            r'''
const viewer = ctx.window.CodoxearFileViewer;
const events = []; const backdrop = { style: {} }; const panel = { style: {} };
const picker = { focus() { events.push('picker-focus'); } };
const modal = viewer.createFileViewerModalRuntime({ backdrop, viewer: panel, pickerInput: picker, closeButton: {}, prepareModalOpen() { events.push('prepare'); }, afterModalVisibilityChanged() { events.push('sync'); }, focusModalCloseButton() { events.push('close-focus'); }, restoreModalFocus() { events.push('restore'); }, isModalTargetOpen(node) { return node.style.display === 'flex'; }, setReturnFocusElement() { events.push('remember'); }, takeReturnFocusElement() { return { id: 'origin' }; } });
modal.show({ wasOpen: false, queryOpen: true, activeElement: {} }); const beforeHide = modal.beginHide(); modal.hideDisplay(); modal.finishHide(beforeHide);
const listeners = {}; const button = { addEventListener(type, fn) { listeners[type] = fn; } };
let presses = 0; let now = 0; viewer.bindFileTouchPress(button, () => { presses += 1; }, { nowMs: () => now });
const event = () => ({ pointerType: 'touch', preventDefault() {}, stopPropagation() {} });
listeners.pointerdown(event()); listeners.click(event()); now = 800; listeners.click(event());
process.stdout.write(JSON.stringify({ display: { backdrop: backdrop.style.display, panel: panel.style.display }, events, presses }));
'''
        )
        self.assertEqual(result["display"], {"backdrop": "none", "panel": "none"})
        self.assertEqual(result["events"], ["remember", "prepare", "sync", "picker-focus", "sync", "restore"])
        self.assertEqual(result["presses"], 2)


if __name__ == "__main__":
    unittest.main()
