"""Behavior of the file download runtime: who names the file, and which transfer runs.

The runtime is the single place that decides (a) the filename handed to the
browser and (b) whether the bytes stream or are buffered into a page-owned blob.
Both are contract-visible on iOS, where a streamed response's name is derived by
Cocoa rather than from the app's own value.
"""
from frontend_module_loader import module_path
import json
import subprocess
import textwrap
import unittest
from pathlib import Path

DOWNLOAD_SCRIPTS = [module_path("app_file_download.js")]


def run_download(js: str) -> dict:
    sources = json.dumps([path.read_text(encoding="utf-8") for path in DOWNLOAD_SCRIPTS])
    program = f"""
const vm = require('vm');
const ctx = {{ window: {{}}, console }};
vm.createContext(ctx);
{sources}.forEach((source) => vm.runInContext(source, ctx));
(async () => {{
{js}
}})().then((out) => process.stdout.write(JSON.stringify(out)), (error) => {{
  process.stdout.write(JSON.stringify({{ error: String(error && error.stack || error) }}));
}});
"""
    proc = subprocess.run(["node"], input=textwrap.dedent(program), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = json.loads(proc.stdout)
    if "error" in result:
        raise AssertionError(result["error"])
    return result


DOCUMENT_STUB = r"""
const makeDocument = () => {
  const clicks = [];
  return {
    clicks,
    createElement() {
      return {
        style: {},
        download: null,
        set href(value) { this._href = value; },
        get href() { return this._href; },
        click() { clicks.push({ href: this._href, download: this.download }); },
        remove() {},
      };
    },
    body: { appendChild() {} },
  };
};
const makeReader = (chunks) => {
  let index = 0;
  const reader = {
    cancels: 0,
    async read() {
      if (index < chunks.length) return { value: chunks[index++], done: false };
      return { done: true };
    },
    async cancel() { reader.cancels += 1; },
    releaseLock() {},
  };
  return reader;
};
"""


class TestDownloadFileName(unittest.TestCase):
    def test_name_is_the_basename_and_control_characters_are_dropped(self) -> None:
        result = run_download(
            r'''
const mod = ctx.window.CodoxearFileDownload;
return {
  nested: mod.downloadFileName('src/dir/notes.md'),
  windows: mod.downloadFileName('C:\\dir\\file.zip'),
  control: mod.downloadFileName('src/a\u0000b\u001fc.py'),
  flat: mod.downloadFileName('photo.jpg'),
  empty: mod.downloadFileName(''),
  missing: mod.downloadFileName(null),
  dot: mod.downloadFileName('src/.'),
  traversal: mod.downloadFileName('../../etc/passwd'),
  spaced: mod.downloadFileName('my report 1.pdf'),
};
'''
        )
        self.assertEqual(result["nested"], "notes.md")
        self.assertEqual(result["windows"], "file.zip")
        self.assertEqual(result["control"], "abc.py")
        self.assertEqual(result["flat"], "photo.jpg")
        self.assertEqual(result["empty"], "")
        self.assertEqual(result["missing"], "")
        self.assertEqual(result["dot"], "")
        # A basename keeps no path meaning: the separators are gone, not resolved.
        self.assertEqual(result["traversal"], "passwd")
        self.assertEqual(result["spaced"], "my report 1.pdf")


class TestStreamingDownload(unittest.TestCase):
    def test_streaming_click_carries_the_app_filename(self) -> None:
        result = run_download(
            DOCUMENT_STUB
            + r'''
const doc = makeDocument();
const runtime = ctx.window.CodoxearFileDownload.createFileDownloadRuntime({
  document: doc,
  resolveAppUrl: (path) => 'app:' + path,
});
const ok = runtime.download('api/sessions/s/file/download?path=src%2Fnotes.md', 'src/notes.md');
return { ok, clicks: doc.clicks };
'''
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            result["clicks"],
            [{"href": "app:api/sessions/s/file/download?path=src%2Fnotes.md", "download": "notes.md"}],
        )

    def test_empty_name_leaves_the_browser_its_own_derivation(self) -> None:
        result = run_download(
            DOCUMENT_STUB
            + r'''
const doc = makeDocument();
const runtime = ctx.window.CodoxearFileDownload.createFileDownloadRuntime({
  document: doc,
  resolveAppUrl: (path) => path,
});
runtime.download('api/x');
return { clicks: doc.clicks };
'''
        )
        self.assertEqual(result["clicks"], [{"href": "api/x", "download": ""}])

    def test_missing_api_path_never_clicks(self) -> None:
        result = run_download(
            DOCUMENT_STUB
            + r'''
const doc = makeDocument();
const runtime = ctx.window.CodoxearFileDownload.createFileDownloadRuntime({
  document: doc,
  resolveAppUrl: (path) => path,
});
const ok = runtime.download('');
return { ok, clicks: doc.clicks };
'''
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["clicks"], [])


class TestStrategySelection(unittest.TestCase):
    def test_no_file_returns_without_touching_the_network(self) -> None:
        result = run_download(
            DOCUMENT_STUB
            + r'''
const doc = makeDocument();
let fetches = 0;
const runtime = ctx.window.CodoxearFileDownload.createFileDownloadRuntime({
  document: doc,
  resolveAppUrl: (path) => path,
  fetchFn: async () => { fetches += 1; return { ok: true, status: 200 }; },
  createObjectUrl: () => 'blob:1',
  revokeObjectUrl: () => {},
  BlobCtor: (parts, options) => ({ parts, type: options.type }),
  shouldBufferDownload: () => true,
  resolveDownloadFileName: () => 'notes.md',
});
const out = await runtime.downloadActive('');
return { out, fetches, clicks: doc.clicks };
'''
        )
        self.assertEqual(result["out"], {"ok": False, "reason": "no-file"})
        self.assertEqual(result["fetches"], 0)
        self.assertEqual(result["clicks"], [])

    def test_platform_without_buffering_streams_without_fetching(self) -> None:
        result = run_download(
            DOCUMENT_STUB
            + r'''
const doc = makeDocument();
let fetches = 0;
const runtime = ctx.window.CodoxearFileDownload.createFileDownloadRuntime({
  document: doc,
  resolveAppUrl: (path) => 'app:' + path,
  resolveDownloadFileName: () => 'src/notes.md',
  fetchFn: async () => { fetches += 1; return { ok: true, status: 200 }; },
  createObjectUrl: () => 'blob:1',
  revokeObjectUrl: () => {},
  BlobCtor: (parts, options) => ({ parts, type: options.type }),
  shouldBufferDownload: () => false,
});
const out = await runtime.downloadActive('api/x');
return { out, fetches, clicks: doc.clicks };
'''
        )
        self.assertEqual(result["out"], {"ok": True, "mode": "stream"})
        self.assertEqual(result["fetches"], 0)
        self.assertEqual(result["clicks"], [{"href": "app:api/x", "download": "notes.md"}])

    def test_buffering_falls_back_to_streaming_when_the_runtime_cannot_buffer(self) -> None:
        result = run_download(
            DOCUMENT_STUB
            + r'''
const doc = makeDocument();
const runtime = ctx.window.CodoxearFileDownload.createFileDownloadRuntime({
  document: doc,
  resolveAppUrl: (path) => 'app:' + path,
  resolveDownloadFileName: () => 'notes.md',
  shouldBufferDownload: () => true,
});
const out = await runtime.downloadActive('api/x');
return { out, clicks: doc.clicks };
'''
        )
        self.assertEqual(result["out"], {"ok": True, "mode": "stream"})
        self.assertEqual(result["clicks"], [{"href": "app:api/x", "download": "notes.md"}])


class TestBufferedDownload(unittest.TestCase):
    def test_blob_handoff_names_the_file_and_revokes_later(self) -> None:
        result = run_download(
            DOCUMENT_STUB
            + r'''
const doc = makeDocument();
const pending = [];
const fetches = [];
const blobs = [];
const revoked = [];
const runtime = ctx.window.CodoxearFileDownload.createFileDownloadRuntime({
  document: doc,
  resolveAppUrl: (path) => 'app:' + path,
  resolveDownloadFileName: () => 'reports/q3.zip',
  fetchFn: async (url, init) => {
    fetches.push({ url, init });
    return { ok: true, status: 200, body: { getReader: () => makeReader([new Uint8Array([1, 2, 3]), new Uint8Array([4, 5])]) } };
  },
  createObjectUrl: (blob) => { blobs.push({ parts: blob.parts, size: blob.size, type: blob.type }); return 'blob:q3'; },
  revokeObjectUrl: (url) => revoked.push(url),
  BlobCtor: (parts, options) => ({ parts, size: parts.reduce((sum, part) => sum + part.byteLength, 0), type: options.type }),
  shouldBufferDownload: () => true,
  setTimeoutFn: (callback, delay) => pending.push({ callback, delay }),
});
const out = await runtime.downloadActive('api/sessions/s/file/download?path=reports%2Fq3.zip');
const beforeRevoke = revoked.slice();
pending.forEach((entry) => entry.callback());
return { out, fetches, blobs, revokedBefore: beforeRevoke, revoked, clicks: doc.clicks, delays: pending.map((entry) => entry.delay) };
'''
        )
        self.assertEqual(result["out"], {"ok": True, "mode": "buffer", "size": 5})
        self.assertEqual(result["fetches"][0]["url"], "app:api/sessions/s/file/download?path=reports%2Fq3.zip")
        # `no-store` is what keeps the body out of the service worker's caches.
        self.assertEqual(result["fetches"][0]["init"], {"credentials": "same-origin", "cache": "no-store"})
        self.assertEqual(result["clicks"], [{"href": "blob:q3", "download": "q3.zip"}])
        self.assertEqual(len(result["blobs"]), 1)
        # An inert type cannot be reconciled against the name's extension.
        self.assertEqual(result["blobs"][0]["type"], "application/octet-stream")
        self.assertEqual(result["blobs"][0]["size"], 5)
        self.assertEqual(result["revokedBefore"], [])
        self.assertEqual(result["revoked"], ["blob:q3"])
        self.assertEqual(result["delays"], [60000])

    def test_oversized_body_streams_instead_and_cancels_the_read(self) -> None:
        result = run_download(
            DOCUMENT_STUB
            + r'''
const doc = makeDocument();
const reader = makeReader([new Uint8Array([1, 2, 3]), new Uint8Array([4, 5, 6, 7])]);
let objectUrls = 0;
const runtime = ctx.window.CodoxearFileDownload.createFileDownloadRuntime({
  document: doc,
  resolveAppUrl: (path) => 'app:' + path,
  resolveDownloadFileName: () => 'clip.mp4',
  fetchFn: async () => ({ ok: true, status: 200, body: { getReader: () => reader } }),
  createObjectUrl: () => { objectUrls += 1; return 'blob:never'; },
  revokeObjectUrl: () => {},
  BlobCtor: (parts, options) => ({ parts, type: options.type }),
  shouldBufferDownload: () => true,
  bufferLimitBytes: 4,
});
const out = await runtime.downloadActive('api/x');
return { out, objectUrls, cancels: reader.cancels, clicks: doc.clicks };
'''
        )
        self.assertEqual(result["out"], {"ok": True, "mode": "stream"})
        self.assertEqual(result["objectUrls"], 0)
        self.assertEqual(result["cancels"], 1)
        self.assertEqual(result["clicks"], [{"href": "app:api/x", "download": "clip.mp4"}])

    def test_authentication_failure_is_reported_not_downloaded(self) -> None:
        result = run_download(
            DOCUMENT_STUB
            + r'''
const doc = makeDocument();
const errors = [];
const runtime = ctx.window.CodoxearFileDownload.createFileDownloadRuntime({
  document: doc,
  resolveAppUrl: (path) => path,
  resolveDownloadFileName: () => 'notes.md',
  fetchFn: async () => ({ ok: false, status: 401 }),
  createObjectUrl: () => 'blob:never',
  revokeObjectUrl: () => {},
  BlobCtor: (parts, options) => ({ parts, type: options.type }),
  shouldBufferDownload: () => true,
  onError: (message) => errors.push(message),
});
const out = await runtime.downloadActive('api/x');
return { out, errors, clicks: doc.clicks };
'''
        )
        self.assertEqual(result["out"], {"ok": False, "reason": "auth"})
        self.assertEqual(result["errors"], ["download failed: sign in again"])
        self.assertEqual(result["clicks"], [])

    def test_transport_failure_is_reported_not_downloaded(self) -> None:
        result = run_download(
            DOCUMENT_STUB
            + r'''
const doc = makeDocument();
const errors = [];
const runtime = ctx.window.CodoxearFileDownload.createFileDownloadRuntime({
  document: doc,
  resolveAppUrl: (path) => path,
  resolveDownloadFileName: () => 'notes.md',
  fetchFn: async () => { throw new Error('connection lost'); },
  createObjectUrl: () => 'blob:never',
  revokeObjectUrl: () => {},
  BlobCtor: (parts, options) => ({ parts, type: options.type }),
  shouldBufferDownload: () => true,
  onError: (message) => errors.push(message),
});
const out = await runtime.downloadActive('api/x');
return { out, errors, clicks: doc.clicks };
'''
        )
        self.assertEqual(result["out"], {"ok": False, "reason": "transport"})
        self.assertEqual(result["errors"], ["download failed: connection lost"])
        self.assertEqual(result["clicks"], [])

    def test_a_second_click_while_fetching_does_not_start_another_transfer(self) -> None:
        result = run_download(
            DOCUMENT_STUB
            + r'''
const doc = makeDocument();
let fetches = 0;
let release = null;
const runtime = ctx.window.CodoxearFileDownload.createFileDownloadRuntime({
  document: doc,
  resolveAppUrl: (path) => path,
  resolveDownloadFileName: () => 'notes.md',
  fetchFn: () => {
    fetches += 1;
    return new Promise((resolve) => { release = resolve; });
  },
  createObjectUrl: () => 'blob:1',
  revokeObjectUrl: () => {},
  BlobCtor: (parts, options) => ({ parts, size: parts.reduce((sum, part) => sum + part.byteLength, 0), type: options.type }),
  shouldBufferDownload: () => true,
  setTimeoutFn: () => {},
});
const first = runtime.downloadActive('api/x');
const second = await runtime.downloadActive('api/x');
release({ ok: true, status: 200, body: { getReader: () => makeReader([new Uint8Array([1])]) } });
const firstOut = await first;
return { firstOut, second, fetches, clicks: doc.clicks };
'''
        )
        self.assertEqual(result["second"], {"ok": False, "reason": "busy"})
        self.assertEqual(result["fetches"], 1)
        self.assertEqual(result["firstOut"], {"ok": True, "mode": "buffer", "size": 1})
        self.assertEqual(result["clicks"], [{"href": "blob:1", "download": "notes.md"}])


if __name__ == "__main__":
    unittest.main()
