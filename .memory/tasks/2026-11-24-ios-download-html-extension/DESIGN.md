# Design — make the viewer's download name survive iOS

Discussion artifact for `PROMPT.md`. Nothing here is implemented; it is the
change I would make and why, ordered by evidence strength per unit of risk.

## Principle

The saved filename currently has exactly one author: the browser's inference over
(a) an empty `download` attribute, (b) a Content-Disposition carrying only the
RFC 5987 form, (c) a URL whose last path segment is `download` with no
extension, and (d) a blanket `application/octet-stream` type. Four signals, none
of which names the file in a form every engine is known to honour. The fix is to
stop relying on inference: the app knows the filename, so it should say it — in
the channels that are actually authoritative — and the server should stop
declaring a type that cannot agree with the name.

## Lever 1 (highest value, lowest risk) — the anchor names the file

`codoxear/static/app_file_download.js`

```js
function download(apiPath, filename) {          // filename is new, optional
  …
  link.download = safeDownloadName(filename);   // was: ""
  …
}
```

- `safeDownloadName`: basename only (strip `/` and `\`), drop NUL/CR/LF, keep the
  extension, allow an empty result (a non-string or empty name keeps today's
  behaviour rather than inventing one).
- Callers already hold the name: `app_file_picker_ops.js` has `baseName` +
  `activeFileDownloadIdentity().path`; pass `baseName(identity.path)`.
- Same treatment for the second download surface — the "Download original file"
  link rendered by `app_file_viewer.js::renderDownload` (`target="_blank"` +
  `download=""`). `target` is ignored once the attribute is present
  (`FrameLoader.cpp:1746`), so giving that anchor a real name is the whole fix
  there.

Why this is the primary lever: WebKit carries the attribute value into the
network load as the download's suggested name
(`PendingDownload.cpp:76` → `NetworkLoad::setSuggestedFilename` →
`m_task->setSuggestedFilename`), with a dedicated `FromDownloadAttribute` flag
in `DownloadManager::startDownload`. That path exists precisely so the page's
name outranks header/URL derivation. It costs nothing, keeps streaming, and also
improves Chrome/Firefox (they already honour CD, but an explicit name removes the
dependency).

## Lever 2 — REVISED after sourced research: neither change is safe as written

Superseded reasoning preserved below for the record. Two reversals:

- Adding a plain `filename=` **transfers naming authority away from the client**:
  WebKit's `DownloadProxy` prefers Content-Disposition only when
  `isAttachmentWithFilename()` is true, and that helper reads only the literal
  `filename` parameter. Today our `filename*=`-only disposition leaves the
  attachment branch untaken, so the `download` attribute wins — which is the
  behaviour we want. Adding `filename=` would hand the name to Cocoa (plus its
  MIME-extension step). Keep the extended form only.
- Replacing `application/octet-stream` with the real MIME type is **not** safe by
  default: WebKit's `appendFileExtensionIfNecessary` returns early for the default
  MIME type but the WHATWG suggested-filename rule *adds* an extension for a
  *known* claimed type that disagrees with the name's extension. A blanket
  `text/plain` for textual files would make `notes.md` → `notes.md.txt`. Only a
  type that matches the extension exactly is neutral.

### Original (superseded) text

`codoxear/file_view.py::download_disposition` → emit both parameters (RFC 6266):

```
attachment; filename="data.zip"; filename*=UTF-8''data.zip
```

ASCII names go in the quoted form; non-ASCII get an ASCII-replaced fallback plus
the `filename*` form. Rationale: codoxear currently sends the extended form
*only*, even for `notes.md` — the simple form is what virtually every other
server sends, and WebKit's own `filenameFromHTTPContentDisposition`
(`HTTPParsers.cpp:345`) accepts only the literal key `filename`.

`codoxear/file_response.py::send_attachment_file_response` → take a content type
and stop hardcoding `application/octet-stream`:

- pass `raw_file_content_type(path)` (already exists in `file_types.py`: maps
  ext → MIME, textual → `text/plain; charset=utf-8`, unknown → octet-stream);
- add `X-Content-Type-Options: nosniff`;
- for the executable-document types (`text/html`, `application/xhtml+xml`,
  `image/svg+xml`) add `Content-Security-Policy: sandbox allow-same-origin`,
  mirroring what `/file/raw` already does (`file_get_routes.py:339-343`).

Why: the observed Safari failure class is "the declared type disagrees with the
name, so the browser repairs the name". `application/octet-stream` cannot agree
with any extension, so it maximises the chance of repair. With `nosniff` plus
`attachment` plus the sandbox header for documents, typing the response is not
less safe than today: an attachment response is never rendered, and a direct
navigation to an HTML-typed download URL lands in the same sandbox posture as
`/file/raw`.

## Lever 3 (retracted) — service-worker exclusion of `/file/download`

An earlier draft proposed excluding `/file/download` from SW interception on the
belief that download bodies were being cached and that the navigate branch's
`{ignoreSearch: true}` could hand back a different file. **Measurement killed
this**: attribute-triggered downloads never reach the SW fetch handler in
WebKitGTK/WPE 26.5 or Chromium 151, so no cache entry is written and no
collision is reachable from any in-app download link (every one of them carries
the attribute). The `ignoreSearch` collision remains a theoretical hazard only
for a genuine navigation to that URL, e.g. pasting it. Not part of this fix.

## Lever 4 (only if a device still mangles the name) — fetch → Blob → object URL

AGENTS.md already sanctions "a fetch/blob handoff". `Blob([bytes], {type})` +
`URL.createObjectURL` + `a[download="<name>.<ext>"]` removes Content-Disposition,
URL derivation and type inference from the decision entirely — the name is a JS
string and the type is a property the app sets.

The cost is real and must be priced, not waved away: the whole file is buffered
in page memory, which is unacceptable for the large-media case the viewer
supports (a video the size of the free RAM). So if this lever is needed it has to
be size-bounded (e.g. ≤ 64 MiB → blob handoff, larger → today's streaming
anchor), and `revokeObjectURL` must run on a timer, not immediately after
`click()`.

`navigator.share({files:[new File(...)]})` is not a fix here: it needs the same
buffered File and adds a picker in the middle of a download.

## Verification plan (device-truth is the acceptance boundary)

This repo cannot run iOS Safari, so acceptance is split:

1. **Wire contract (Docker):** a test that hits
   `/api/sessions/<id>/file/download?path=…` and asserts the header block —
   `Content-Disposition` containing both `filename=` and `filename*=`,
   `Content-Type` equal to the file's real type, `X-Content-Type-Options:
   nosniff`, `Content-Length` equal to the file size, and body bytes hashing
   equal to the on-disk file. Today nothing pins the naming contract: the only
   existing check (`scripts/docker_office_preview_verify.py:168`) asserts that a
   download *href* exists.
2. **Client contract (VM harness):** `app_file_download.js` with a stub document
   asserting the anchor's `download` value equals the basename and that a
   traversal-shaped input is reduced to a basename. The existing viewer harness
   already asserts `download === ""`, so it flips with the change.
3. **Device:** on the phone, download from the viewer and read the name in Files.
   That observation — not any local test — is what closes the task. The
   distinguishing question is in `EPISTEMIC.md`: does the real stem survive
   (`foo.zip.html` → Lever 1/2 target) or not (`download.html` → Lever 2's
   `filename=` parameter is the missing piece)?

## What I would not do

- Not switch the endpoint to `Content-Disposition: inline` for previewable types:
  that trades a naming bug for browsers deciding to render content in place.
- Not remove the `download` attribute in favour of Content-Disposition alone:
  `c8330b6d` exists precisely because a navigational download blanks the app on
  `beforeunload`.
- Not add a per-extension MIME table of our own: `raw_file_content_type` already
  owns that mapping and is used by `/file/raw`.
