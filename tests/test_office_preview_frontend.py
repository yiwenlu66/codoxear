"""Execute the Office/PDF viewer seams; no conversion is demanded by inspection."""
import json
import unittest

from frontend_module_loader import module_path
from test_file_preview_default import CONTROLLER_HARNESS, run_vm


PDF_HARNESS = r"""
const viewer = ctx.window.CodoxearFileViewer;
const events = [], fallbacks = [];
let current = true, active = null, observerCallback = null;
function node(tag, attrs = {}, children = []) {
  return { tag, attrs, children, clientWidth: 624, dataset: { pageNumber: attrs['data-page-number'] },
    textContent: attrs.text || '', classList: { add() {}, remove() {} },
    appendChild(child) { this.children.push(child); }, replaceChildren(...items) { this.children = items; },
    querySelectorAll() { return this.children.filter(child => child.attrs.class === 'filePdfPage'); } };
}
const host = node('div'); host.innerHTML = '';
const request = { signal: { id: 'original-signal' } };
const page = { getViewport: ({ scale }) => ({ width: 600 * scale, height: 800 * scale }),
  render() { events.push('page-render'); return { promise: Promise.resolve(), cancel() {} }; } };
let fetchImpl = async (url, options) => { events.push(['fetch', url, options.headers.Range, options.signal === request.signal, options.credentials]);
  return { ok: true, body: { cancel: async () => events.push('cancel-range-body') } }; };
let ensureImpl = async () => ({ getDocument(options) { events.push(['getDocument', options]);
  return { promise: Promise.resolve({ numPages: 2, getPage: async () => page }), destroy() {} }; } });
const runtime = viewer.createFilePdfRenderRuntime({ host, el: node,
  ensurePdfJs: () => ensureImpl(), fetchPreview: (url, options) => fetchImpl(url, options),
  handleAuthLoss: () => events.push('auth-loss'), createCanvas: () => ({ style: {}, getContext: () => ({}) }),
  devicePixelRatio: () => 1, disposeFileEditor() {}, disposePdfRender() { active = null; events.push('dispose'); },
  clearFileVideo() {}, setFileRenderSurface() {}, renderDownloadFallback: (...args) => fallbacks.push(args),
  isCurrentFileOpenRequest: () => current, setActivePdfRenderState: state => { active = state; },
  isActivePdfRenderState: state => active === state, updateFileTouchToolbar() {},
  IntersectionObserverCtor: class { constructor(callback) { observerCallback = callback; } observe() {} disconnect() {} },
});
const preview = { downloadUrl: '/source/report.docx', prepareDocument: true };
"""


class TestOfficePreviewFrontend(unittest.TestCase):
    def test_office_plan_is_nontext_and_keeps_original_download_identity(self):
        result = run_vm(CONTROLLER_HARNESS + r"""
const controller = buildController();
const request = controller.startFileOpenRequest('report.docx', { apiPath: 'raw-token', gitPath: true }).request;
const plan = controller.prepareFileLoadResult('report.docx', { kind: 'office', preview_url: '/preview.pdf', download_url: '/original.docx', size: 42 }, request);
const fallbackPlan = controller.prepareFileLoadResult('report.docx', { kind: 'office', preview_url: '/preview.pdf' }, request);
process.stdout.write(JSON.stringify({ plan, fallbackPlan, kind: controller.currentActiveFileKind(), editable: controller.currentActiveFileEditable(), controls: controller.currentFileModeControlState() }));
""")
        self.assertEqual(result['plan']['kind'], 'office')
        self.assertEqual(result['plan']['pdfUrl'], '/preview.pdf')
        self.assertEqual(result['plan']['downloadUrl'], '/original.docx')
        self.assertIn('path_token=raw-token', result['fallbackPlan']['downloadUrl'])
        self.assertIn('git_path=1', result['fallbackPlan']['downloadUrl'])
        self.assertEqual(result['kind'], 'office')
        self.assertFalse(result['editable'])
        self.assertFalse(result['controls']['downloadDisabled'])
        self.assertFalse(result['controls']['markdownPreviewVisible'])

    def test_load_result_handoff_preserves_preparation_and_failure_status(self):
        result = run_vm(r"""
const viewer = ctx.window.CodoxearFileViewer;
const events = [], request = { id: 1 };
const runtime = viewer.createFileLoadResultRuntime({
 controller: { prepareFileLoadResult: () => ({ kind: 'office', pdfUrl: '/preview.pdf', downloadUrl: '/original.docx', status: 'done' }), isCurrentFileOpenRequest: () => true, handleActiveVideoLoadError() {}, handleActiveVideoLoadedMetadata() {} },
 resolveAppUrl: url => '/prefix' + url, setStatus: status => events.push(['status', status]),
 disposeFileEditor() {}, renderMonacoDiff() {}, renderMonacoFile() {}, renderFilePreview() {}, renderBlockedFileNotice() {}, showImage() {}, showVideo() {}, loadCompatibleVideoPreview() {},
 renderPdfFile: async (rel, url, req, preview) => { events.push(['pdf', rel, url, req === request, preview]); return { error: 'LibreOffice is not installed' }; },
});
(async () => { const result = await runtime.apply('report.docx', {}, request); process.stdout.write(JSON.stringify({ result, events })); })();
""")
        self.assertTrue(result['result'])
        self.assertIn('Preparing document preview…', result['events'][0][1])
        self.assertEqual(result['events'][1][2], '/prefix/preview.pdf')
        self.assertEqual(result['events'][1][4], {'downloadUrl': '/prefix/original.docx', 'prepareDocument': True})
        self.assertIn('LibreOffice is not installed', result['events'][2][1])

    def test_office_preflight_then_lazy_pdf_pages(self):
        result = run_vm(PDF_HARNESS + r"""
(async () => {
 const promise = runtime.render('report.docx', '/preview.pdf', request, preview);
 const preparing = host.children[0].textContent;
 const result = await promise;
 const container = host.children[0];
 const before = events.filter(x => x === 'page-render').length;
 observerCallback([{ isIntersecting: true, target: container.children[0] }]);
 await Promise.resolve();
 await Promise.resolve();
 process.stdout.write(JSON.stringify({ result, preparing, events, before, pages: container.children.length, fallbacks }));
})();
""")
        self.assertTrue(result['result'])
        self.assertEqual(result['preparing'], 'Preparing document preview…')
        self.assertEqual(result['pages'], 2)
        self.assertEqual(result['before'], 0)
        self.assertIn('page-render', result['events'])
        self.assertIn(['fetch', '/preview.pdf', 'bytes=0-0', True, 'same-origin'], result['events'])
        self.assertEqual(result['fallbacks'], [])

    def test_every_failure_uses_original_download(self):
        for stage, expected in [('conversion', 'Encrypted document'), ('module', 'module failed'), ('loading', 'Invalid PDF'), ('page', 'page unavailable'), ('auth', 'authentication required')]:
            with self.subTest(stage=stage):
                result = run_vm(PDF_HARNESS + '\nconst stage = ' + json.dumps(stage) + r""";
if (stage === 'conversion') fetchImpl = async () => ({ ok: false, status: 422, json: async () => ({ error: 'Encrypted document' }) });
if (stage === 'auth') fetchImpl = async () => ({ ok: false, status: 401 });
if (stage === 'module') ensureImpl = async () => { throw new Error('module failed'); };
if (stage === 'loading') ensureImpl = async () => ({ getDocument: () => ({ promise: Promise.reject(new Error('Invalid PDF')), destroy() {} }) });
if (stage === 'page') ensureImpl = async () => ({ getDocument: () => ({ promise: Promise.resolve({ getPage: async () => { throw new Error('page unavailable'); } }), destroy() {} }) });
(async () => { const result = await runtime.render('report.docx', '/preview.pdf', request, preview); process.stdout.write(JSON.stringify({ result, events, fallbacks })); })();
""")
                self.assertEqual(result['result']['error'], expected)
                self.assertEqual(result['fallbacks'], [['report.docx', '/source/report.docx', expected]])
                if stage == 'auth':
                    self.assertIn('auth-loss', result['events'])

    def test_aborted_or_replaced_preview_never_renders_fallback(self):
        for cancelled in (False, True):
            result = run_vm(PDF_HARNESS + '\nconst cancelled = ' + json.dumps(cancelled) + r""";
fetchImpl = async () => { if (!cancelled) current = false; const error = new Error('aborted'); error.name = 'AbortError'; throw error; };
(async () => { const result = await runtime.render('report.docx', '/preview.pdf', request, preview); process.stdout.write(JSON.stringify({ result, fallbacks, events })); })();
""")
            self.assertFalse(result['result'])
            self.assertEqual(result['fallbacks'], [])
            self.assertFalse(any(isinstance(event, list) and event[0] == 'getDocument' for event in result['events']))

    def test_native_pdf_does_not_demand_office_preflight(self):
        result = run_vm(PDF_HARNESS + r"""
(async () => { const result = await runtime.render('file.pdf', '/original.pdf', request); process.stdout.write(JSON.stringify({ result, events })); })();
""")
        self.assertTrue(result['result'])
        self.assertFalse(any(isinstance(event, list) and event[0] == 'fetch' for event in result['events']))

    def test_bare_office_transcript_refs_are_candidates(self):
        result = run_vm(r"""
const parse = ctx.window.CodoxearMarkdown.parseLocalFileRef;
process.stdout.write(JSON.stringify(['report.doc', 'report.docx', 'data.xls', 'data.xlsx', 'deck.ppt', 'deck.PPTX', '/tmp/report.docx', 'notes.unsupported'].map(path => parse(path))));
""", extra_scripts=(module_path('app_markdown.js'),))
        self.assertEqual([item['path'] for item in result[:7]], ['report.doc', 'report.docx', 'data.xls', 'data.xlsx', 'deck.ppt', 'deck.PPTX', '/tmp/report.docx'])
        self.assertIsNone(result[7])


if __name__ == '__main__':
    unittest.main()
