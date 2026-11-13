"""Behavioral checks for the file viewer's preview-by-default contract.

Contract under test:
  * a previewable document (markdown, html) opens in its rendered preview;
  * a changed diffable candidate still opens as a diff;
  * an explicit caller mode still wins (line-targeted transcript references
    keep source view because only source can show a line);
  * a non-document kind still opens as source;
  * an html document previews only through a sandboxed frame fed by the
    server-provided document URL, and falls back to source (with an explicit
    status) when the server could not produce one.

Everything here executes the real modules in a VM; no source-string assertions.
"""
from frontend_module_loader import module_path
import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIEWER_SCRIPTS = [
    module_path(name)
    for name in (
        "app_file_candidates.js", "app_file_candidate_state.js", "app_file_viewer_operations.js", "app_file_download.js", "app_file_viewer_lifecycle.js", "app_file_viewer_panel.js",
        "app_file_unsaved_dialog.js", "app_file_paste_dialog.js", "app_file_pdf.js", "app_file_media.js",
        "app_file_mode.js", "app_file_render_surface.js", "app_file_viewer_controller.js", "app_file_viewer.js",
    )
]


def run_vm(body: str, extra_scripts=()) -> dict:
    paths = list(VIEWER_SCRIPTS) + [Path(extra) for extra in extra_scripts]
    sources = json.dumps([path.read_text(encoding="utf-8") for path in paths])
    script = f"""
const vm = require('vm');
const ctx = {{ window: {{}}, console }};
vm.createContext(ctx);
{sources}.forEach((source) => vm.runInContext(source, ctx));
{body}
"""
    proc = subprocess.run(["node"], input=textwrap.dedent(script), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise AssertionError(f"VM harness failed ({proc.returncode}):\n{proc.stderr}\n{proc.stdout}")
    return json.loads(proc.stdout)


CONTROLLER_HARNESS = r"""
const module = ctx.window.CodoxearFileViewer;
const noop = () => {};
const applied = [];
const toasts = [];
function kindFor(path) {
  const ext = String(path).split('.').pop().toLowerCase();
  if (['md', 'markdown', 'mdown', 'mkd'].includes(ext)) return 'markdown';
  if (['html', 'htm'].includes(ext)) return 'html';
  return 'text';
}
const overrides = {
  wiring: { createFileViewerOperationsOptions: (options) => options },
  el: (tag, attrs = {}, children = []) => ({ tag, attrs, children }),
  fileStatus: { textContent: '', replaceChildren() {} },
  fileEditButton: { disabled: false, innerHTML: '', title: '', classList: { toggle() {} }, setAttribute() {}, removeAttribute() {} },
  iconSvg: () => '',
  currentSessionId: () => 's1',
  currentFileSessionId: () => 's1',
  normalizeLineNumber: (value) => {
    const n = Number(value);
    return Number.isFinite(n) && n >= 1 ? Math.floor(n) : null;
  },
  normalizeFileApiPath: (value) => (typeof value === 'string' ? value : ''),
  isFileViewerOpen: () => true,
  isTextFileKind: (kind) => kind === 'text' || kind === 'markdown' || kind === 'html',
  isDiffableFileKind: (kind) => kind === 'text' || kind === 'markdown' || kind === 'html',
  isPreviewableFilePath: (path) => /\.(md|markdown|mdown|mkd|html|htm)$/i.test(String(path || '')),
  isPreviewableFileKind: (kind) => kind === 'markdown' || kind === 'html',
  confirmReload: () => true,
  promptUnsavedFileChoice: async () => 'cancel',
  restoreFileEditorText: noop,
  hideFileViewer: noop,
  setFilePath: noop,
  resetFileViewerPanel: noop,
  applyFileLoadResult: (rel, result, request, options) => {
    applied.push({ rel, kind: result.kind, viewMode: options.viewMode });
    return true;
  },
  normalizeDraftFilePath: (value) => String(value || '').trim(),
  inspectSessionFilePath: async (path) => ({ exists: true, kind: kindFor(path) }),
  api: async (url) => {
    if (String(url).includes('/git/file_versions')) {
      return { base_text: 'old\n', current_text: 'new\n', base_exists: true, current_exists: true };
    }
    const rel = decodeURIComponent(String(url).split('path=')[1].split('&')[0]);
    const kind = kindFor(rel);
    return kind === 'html'
      ? { kind, text: '<h1>report</h1>', editable: true, version: 'v1', size: 15, preview_url: '/api/sessions/s1/file/raw/' + rel }
      : { kind, text: '# title', editable: true, version: 'v1', size: 7 };
  },
  focusEditor: () => null,
  disposeOpenRender: noop,
  persistFileViewMode: noop,
  persistFileNonDiffMode: noop,
  updateFileTouchToolbar: noop,
  hasBlockingFileEditorModal: () => false,
  isTextEntryTarget: () => false,
  eventTargetElement: (value) => value,
  isActiveFileEditorInput: () => false,
  focusActiveFileCodeEditor: () => null,
  nowMs: () => 0,
  setToast: (message) => { toasts.push(message); },
  renderMonacoFile: async () => ({ status: '' }),
  getFileEditorText: () => '',
  fmtBytes: (value) => `${value} B`,
  applyFileMode: noop,
  rememberOpenedFile: noop,
  renderFilePickerMenu: noop,
  historyFileSelectionForSession: () => ({ path: '', line: null, gitPath: false, apiPath: '' }),
  initialFileViewMode: 'file',
  initialFileNonDiffMode: 'file',
};
function buildController(extra = {}) {
  const merged = { ...overrides, ...extra };
  const required = new Proxy(merged, { get: (target, key) => (key in target ? target[key] : noop) });
  return module.createFileViewerController(required);
}
async function openWith(controller, path, options = {}) {
  await controller.openFilePathWithResolvedMode(path, options);
  return { path, mode: controller.currentFileViewMode(), last: applied[applied.length - 1] };
}
"""


class TestFilePreviewDefault(unittest.TestCase):
    def test_open_mode_default_follows_document_kind(self) -> None:
        result = run_vm(
            CONTROLLER_HARNESS
            + r"""
(async () => {
  const controller = buildController();
  // The changed-candidate branch reads the viewer's live git-candidate state;
  // mark it fresh so a changed diffable file resolves to diff.
  controller.setFileCandidateGitStateFresh(true);
  const out = {};
  out.markdown = await openWith(controller, 'docs/notes.md');
  out.markdownUppercase = await openWith(controller, 'docs/NOTES.MD');
  out.html = await openWith(controller, 'docs/report.html');
  out.source = await openWith(controller, 'src/main.py');
  out.diff = await openWith(controller, 'src/main.py', { changed: true, gitPath: true });
  out.explicitSource = await openWith(controller, 'docs/notes.md', { mode: 'file' });
  out.explicitDiff = await openWith(controller, 'docs/notes.md', { mode: 'diff' });
  out.changedMarkdownFromPicker = await openWith(controller, 'docs/changed.md', { changed: true, gitPath: true });
  process.stdout.write(JSON.stringify(out));
})();
""",
            extra_scripts=(),
        )
        self.assertEqual(result["markdown"]["mode"], "preview")
        self.assertEqual(result["markdown"]["last"]["viewMode"], "preview")
        self.assertEqual(result["markdownUppercase"]["mode"], "preview")
        self.assertEqual(result["html"]["mode"], "preview")
        self.assertEqual(result["html"]["last"]["viewMode"], "preview")
        self.assertEqual(result["source"]["mode"], "file")
        self.assertEqual(result["source"]["last"]["viewMode"], "file")
        self.assertEqual(result["diff"]["mode"], "diff")
        self.assertEqual(result["diff"]["last"]["viewMode"], "diff")
        self.assertEqual(result["explicitSource"]["mode"], "file")
        self.assertEqual(result["explicitDiff"]["mode"], "diff")
        # A changed candidate opened from the picker still prefers the diff,
        # even for a previewable document: the point of that entry is "what
        # changed".
        self.assertEqual(result["changedMarkdownFromPicker"]["mode"], "diff")

    def test_html_preview_requires_server_document_url(self) -> None:
        result = run_vm(
            """
const module = ctx.window.CodoxearFileViewerOperations;
let activeKind = '';
let viewMode = 'preview';
const calls = [];
const runtime = module.createFileViewerOperationsRuntime({
  el: (tag, attrs = {}, children = []) => ({ tag, attrs, children }),
  fileStatus: { textContent: '', replaceChildren() {} },
  fileEditButton: { disabled: false, innerHTML: '', classList: { toggle() {} }, setAttribute() {}, removeAttribute() {} },
  isCurrentFileOpenRequest: () => true,
  isPreviewableFileKind: (kind) => kind === 'markdown' || kind === 'html',
  applyActiveFileTextState: ({ kind }) => { activeKind = kind; },
  currentActiveFileKind: () => activeKind,
  currentActiveFileEditable: () => true,
  fmtBytes: (value) => `${value} B`,
  currentFileViewMode: () => viewMode,
  setFileViewMode: (mode) => { calls.push(mode); viewMode = mode; },
});
const request = { requestId: 1 };
const withUrl = runtime.prepareFileLoadResult('docs/report.html', { kind: 'html', text: '<h1>x</h1>', editable: false, version: 'v', size: 10, preview_url: '/api/sessions/s1/file/raw/docs/report.html' }, request, { viewMode: 'preview' });
const markdown = runtime.prepareFileLoadResult('docs/notes.md', { kind: 'markdown', text: '# t', editable: true, version: 'v', size: 3 }, request, { viewMode: 'preview' });
const sourceView = runtime.prepareFileLoadResult('docs/notes.md', { kind: 'markdown', text: '# t', editable: true, version: 'v', size: 3 }, request, { viewMode: 'file' });
const plainText = runtime.prepareFileLoadResult('notes.txt', { kind: 'text', text: 'x', editable: true, version: 'v', size: 1 }, request, { viewMode: 'preview' });
viewMode = 'preview';
const withoutUrl = runtime.prepareFileLoadResult('docs/other.html', { kind: 'html', text: '<h1>y</h1>', editable: true, version: 'v', size: 10 }, request, { viewMode: 'preview' });
process.stdout.write(JSON.stringify({
  withUrl: { renderPreview: withUrl.renderPreview, previewKind: withUrl.previewKind, previewUrl: withUrl.previewUrl, status: withUrl.status },
  markdown: { renderPreview: markdown.renderPreview, previewKind: markdown.previewKind, previewUrl: markdown.previewUrl, status: markdown.status },
  sourceView: { renderPreview: sourceView.renderPreview, status: sourceView.status },
  plainText: { renderPreview: plainText.renderPreview, status: plainText.status },
  withoutUrl: { renderPreview: withoutUrl.renderPreview, status: withoutUrl.status },
  modeCalls: calls,
  viewModeAfterWithoutUrl: viewMode,
  activeKind: activeKind,
}));
"""
        )
        self.assertEqual(result["withUrl"]["renderPreview"], True)
        self.assertEqual(result["withUrl"]["previewKind"], "html")
        self.assertEqual(result["withUrl"]["previewUrl"], "/api/sessions/s1/file/raw/docs/report.html")
        # The sandboxed frame never runs scripts, so the status says so instead
        # of letting an inert JavaScript-dependent page read as a broken preview.
        self.assertIn("preview (scripts blocked)", result["withUrl"]["status"])
        self.assertEqual(result["markdown"]["renderPreview"], True)
        self.assertEqual(result["markdown"]["previewKind"], "markdown")
        self.assertEqual(result["markdown"]["previewUrl"], "")
        self.assertIn("preview", result["markdown"]["status"])
        self.assertNotIn("scripts blocked", result["markdown"]["status"])
        self.assertEqual(result["sourceView"]["renderPreview"], False)
        self.assertEqual(result["plainText"]["renderPreview"], False)
        # No server document URL: source view, and the mode is corrected so the
        # preview control never claims a preview that is not on screen.
        self.assertEqual(result["withoutUrl"]["renderPreview"], False)
        self.assertIn("preview unavailable", result["withoutUrl"]["status"])
        self.assertEqual(result["modeCalls"], ["file"])
        self.assertEqual(result["viewModeAfterWithoutUrl"], "file")

    def test_html_preview_frame_is_sandboxed_and_document_is_dispatched(self) -> None:
        result = run_vm(
            r"""
const viewer = ctx.window.CodoxearFileViewer;
const created = [];
const el = (tag, attrs = {}, children = []) => {
  const node = { tag, attrs, children };
  created.push(node);
  return node;
};
const host = { innerHTML: 'stale', appended: [], appendChild(node) { this.appended.push(node); } };
const noop = () => {};
const fallback = viewer.createFileFallbackRuntime({
  host, el,
  normalizeLineNumber: (value) => Number(value) || null,
  requestAnimationFrame: (fn) => fn(),
  disposeFileEditor: noop, disposePdfRender: noop, clearFileMedia: noop,
  setFileRenderSurface: noop, setFileEditorKind: noop, applyPlainTextFallbackState: noop,
  updateFileTouchToolbar: noop, currentSessionId: () => 's1',
  markdownPreviewHtml: () => '<p>md</p>', upgradeCandidateFileRefs: noop,
  blockedFileMessage: () => '',
});
const htmlNode = fallback.applyFilePreview('docs/report.html', '<h1>x</h1>', { kind: 'html', url: 'app:/api/sessions/s1/file/raw/docs/report.html' });
const htmlAttrs = { ...htmlNode.attrs };
const afterHtmlHost = host.innerHTML;
const mdNode = fallback.applyFilePreview('docs/notes.md', '# t', { kind: 'markdown', url: '' });
const events = [];
const controller = {
  prepareFileLoadResult: () => ({ kind: 'text', text: '<h1>x</h1>', renderPreview: true, previewKind: 'html', previewUrl: '/api/sessions/s1/file/raw/docs/report.html', status: 'docs/report.html - preview' }),
  isCurrentFileOpenRequest: () => true,
  handleActiveMediaLoadError: () => true,
  handleActiveMediaLoadedMetadata: () => true,
};
const loadRuntime = viewer.createFileLoadResultRuntime({
  controller,
  resolveAppUrl: (path) => `app:${path}`,
  setStatus: (status) => events.push(['status', status]),
  disposeFileEditor: noop,
  renderMonacoDiff: async () => ({ monacoUnavailable: false }),
  renderMonacoFile: async () => ({ monacoUnavailable: false }),
  renderFilePreview: (rel, text, preview) => events.push(['preview', rel, preview]),
  renderBlockedFileNotice: noop,
  renderPdfFile: async () => true,
  showImage: noop,
  showMedia: noop,
  loadCompatibleMediaPreview: async () => true,
});
(async () => {
  await loadRuntime.apply('docs/report.html', { kind: 'html', text: '<h1>x</h1>' }, { requestId: 1 }, { viewMode: 'preview' });
  process.stdout.write(JSON.stringify({
    htmlTag: htmlNode.tag,
    htmlAttrs,
    htmlClearedHost: afterHtmlHost,
    mdTag: mdNode.tag,
    mdHtml: mdNode.attrs.html,
    events,
  }));
})();
"""
        )
        self.assertEqual(result["htmlTag"], "iframe")
        self.assertEqual(result["htmlAttrs"]["class"], "fileHtmlPreview")
        self.assertEqual(result["htmlAttrs"]["src"], "app:/api/sessions/s1/file/raw/docs/report.html")
        self.assertEqual(result["htmlAttrs"]["sandbox"], "allow-same-origin")
        self.assertNotIn("allow-scripts", result["htmlAttrs"]["sandbox"])
        self.assertEqual(result["htmlClearedHost"], "")
        self.assertEqual(result["mdTag"], "div")
        self.assertEqual(result["mdHtml"], "<p>md</p>")
        self.assertEqual(result["events"][0], ["preview", "docs/report.html", {"kind": "html", "url": "app:/api/sessions/s1/file/raw/docs/report.html"}])
        self.assertEqual(result["events"][1], ["status", "docs/report.html - preview"])

    def test_line_targeted_reference_keeps_source_view(self) -> None:
        result = run_vm(
            r"""
const viewer = ctx.window.CodoxearFileViewer;
const opens = [];
const runtime = viewer.createFileReferenceRuntime({
  sessionState: { get: () => 's1' },
  sessionById: () => ({ session_id: 's1', cwd: '/repo' }),
  sessions: () => [],
  chatRoot: null,
  ElementCtor: null,
  sessionRelativePath: (path) => String(path || ''),
  listFromFilesField: () => [],
  listFromFileRecords: () => [],
  normalizeFileApiPath: () => '',
  normalizeLineNumber: (value) => {
    const n = Number(value);
    return Number.isFinite(n) && n >= 1 ? Math.floor(n) : null;
  },
  api: async () => ({ kind: 'markdown', exists: true }),
  el: (tag, attrs = {}, children = []) => ({ tag, attrs, children }),
  showFileViewer: async (options) => { opens.push(options); return true; },
  parseLocalFileRef: (raw) => {
    const m = String(raw).match(/^(.*):(\d+)$/);
    return m ? { path: m[1], line: Number(m[2]) } : { path: String(raw), line: null };
  },
  selectSession: async () => true,
  openDirectorySession: () => true,
  setToast: () => {},
});
(async () => {
  await runtime.openReference({ path: 'docs/notes.md:120', line: 120 });
  await runtime.openReference({ path: 'docs/notes.md', line: null });
  process.stdout.write(JSON.stringify({ opens }));
})();
"""
        )
        self.assertEqual(result["opens"][0]["mode"], "file")
        self.assertEqual(result["opens"][0]["line"], 120)
        self.assertEqual(result["opens"][1]["mode"], "")
        self.assertEqual(result["opens"][1]["line"], None)

    def test_lifecycle_forwards_caller_mode_into_open_resolution(self) -> None:
        result = run_vm(
            r"""
const viewer = ctx.window.CodoxearFileViewer;
const calls = [];
const target = { kind: 'path', path: 'docs/notes.md', line: null, gitPath: false, apiPath: '', changed: false };
const controller = new Proxy({
  currentFileViewerSessionId: () => 's1',
  isCurrentFileViewerSessionSync: () => true,
  beginFileViewerSessionSync: () => 1,
  setFileViewerSessionId: () => true,
  clearFileViewerUnavailableSession: () => true,
  resolveFileViewerOpenTarget: () => target,
}, { get: (target, key) => (key in target ? target[key] : () => true) });
const noop = () => true;
const lifecycleOptions = {
  controller,
  sessionState: { get: () => 's1' },
  isFileViewerOpen: () => true,
  maybeHandleUnsavedFileChanges: async () => true,
  filePickerSearchSessionId: () => 's1',
  refreshFileCandidates: async () => true,
  setFilePath: noop,
  openFilePathWithResolvedMode: async (path, options) => { calls.push({ path, mode: options.mode }); return true; },
  renderEmptyFileViewerTarget: noop,
  setStatus: noop,
  showModal: noop,
  updateFileTouchToolbar: noop,
  setFileViewMode: () => true,
  applyFileMode: noop,
  resetFileViewerPanel: noop,
  closeFilePickerMenu: noop,
  resetFileSearchState: noop,
  setFileSearchSessionId: noop,
  hideFileUnsavedDialog: noop,
  hideFilePasteDialog: noop,
  openFilePickerSearchQuery: noop,
  setPreserveSearchOnFocus: noop,
  focusFilePickerInput: noop,
};
const runtime = viewer.createFileViewerLifecycleRuntime(new Proxy(lifecycleOptions, { get: (target, key) => (key in target ? target[key] : () => true) }));
(async () => {
  await runtime.show({ path: 'docs/notes.md', mode: 'file', line: 120 });
  await runtime.show({ path: 'docs/notes.md', mode: '', line: null });
  process.stdout.write(JSON.stringify({ calls }));
})();
"""
        )
        self.assertEqual(result["calls"][0]["mode"], "file")
        self.assertEqual(result["calls"][1]["mode"], None)

    def test_previewable_extension_sets_match_the_server(self) -> None:
        from codoxear.file_text import document_kind
        from codoxear.file_types import HTML_EXTENSIONS, MARKDOWN_EXTENSIONS

        # The client decides previewability from the path before a load; the
        # server decides it from the loaded kind. Compare both directions of
        # the same mapping, including the cases where "looks like a document"
        # and "is the document kind" could drift (uppercase, dotfile, no
        # extension, a document extension used as another extension's suffix).
        samples = [
            "docs/notes.md",
            "a.MARKDOWN",
            "x.mkd",
            "page.html",
            "page.HTM",
            "src/main.py",
            "notes.txt",
            "docs/README",
            ".md",
            "notes.md.txt",
            "img/pic.png",
            "data.jsonl",
        ]
        result = run_vm(
            f"""
const helpers = ctx.window.CodoxearFileHelpers;
const samples = {json.dumps(samples)};
process.stdout.write(JSON.stringify({{
  markdown: helpers.MARKDOWN_EXTENSIONS,
  html: helpers.HTML_EXTENSIONS,
  textKindHtml: helpers.isTextFileKind('html'),
  textKindMarkdown: helpers.isTextFileKind('markdown'),
  textKindImage: helpers.isTextFileKind('image'),
  diffableHtml: helpers.isDiffableFileKind('html'),
  previewKindHtml: helpers.isPreviewableFileKind('html'),
  previewKindMarkdown: helpers.isPreviewableFileKind('markdown'),
  previewKindText: helpers.isPreviewableFileKind('text'),
  previewPathHtml: helpers.isPreviewableFilePath('docs/report.htm'),
  previewPathMarkdown: helpers.isPreviewableFilePath('/repo/x.MARKDOWN'),
  previewPathText: helpers.isPreviewableFilePath('src/main.py'),
  previewPathDotfile: helpers.isPreviewableFilePath('.md'),
  previewPathNoExt: helpers.isPreviewableFilePath('docs/README'),
  sampleFlags: samples.map((path) => helpers.isPreviewableFilePath(path)),
}}));
""",
            extra_scripts=(module_path("app_file_helpers.js"),),
        )
        self.assertEqual(sorted(result["markdown"]), sorted(MARKDOWN_EXTENSIONS))
        self.assertEqual(sorted(result["html"]), sorted(HTML_EXTENSIONS))
        self.assertTrue(result["textKindHtml"])
        self.assertTrue(result["textKindMarkdown"])
        self.assertFalse(result["textKindImage"])
        self.assertTrue(result["diffableHtml"])
        self.assertTrue(result["previewKindHtml"])
        self.assertTrue(result["previewKindMarkdown"])
        self.assertFalse(result["previewKindText"])
        self.assertTrue(result["previewPathHtml"])
        self.assertTrue(result["previewPathMarkdown"])
        self.assertFalse(result["previewPathText"])
        self.assertFalse(result["previewPathDotfile"])
        self.assertFalse(result["previewPathNoExt"])
        # The client path predicate must agree with the server's document kind,
        # which is the authoritative classification the viewer renders from.
        self.assertEqual(len(result["sampleFlags"]), len(samples))
        for sample, client_flag in zip(samples, result["sampleFlags"]):
            server_flag = document_kind(Path(sample)) in {"markdown", "html"}
            self.assertEqual(client_flag, server_flag, f"previewability drift for {sample}")


if __name__ == "__main__":
    unittest.main()
