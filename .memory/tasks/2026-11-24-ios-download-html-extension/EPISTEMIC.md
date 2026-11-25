# EPISTEMIC — iOS saves the download as `<name>.<ext>.html`

## Phenomenon

On iOS, the file viewer's Download action yields a file that iOS will not open
because it carries an `.html` suffix.

## Established by measurement (ours)

- **Wire**: `200`, `Content-Type: application/octet-stream`,
  `Content-Length`, `Content-Disposition: attachment; filename*=UTF-8''<name>`,
  `no-store`. No `filename=`, no `nosniff`, no `Accept-Ranges`. Long-standing
  shape, not a regression from the streaming rewrite.
- **Client sends no name**: `<a href="…/file/download?path=…" download="">`; the
  URL's last path segment is `download`, with no extension.
- **Real app in Docker (Chromium 151)**: clicking `#fileDownloadBtn` saves
  `payload.bin`, `notes.txt`, `notes.md` — correct extensions, byte-identical,
  with and without the service worker; URL-prefix handling correct.
- **Engine matrix (WebKitGTK/WPE 26.5 + Chromium, 72 runs)**: `filename*` alone is
  honoured; no engine appends `.html` even for `text/html` + `filename="foo.zip"`;
  `download="explicit.zip"` beats a `filename*`-only disposition on WebKit while
  the header wins on Chromium.
- **Attribute downloads never reach a service worker's fetch handler** in either
  engine, confirmed on the real app with a positive control. (Retracts the earlier
  cache-collision theory.)

## Established from source (WebKit branches + Apple/WHATWG docs)

- Two naming layers. Cocoa's `NSURLResponse.suggestedFilename` (CD → URL last
  path component → host → `unknown`, and "in most cases appends the proper file
  extension based on the MIME type"), then WebKit's `DownloadProxy`:
  **CD-with-plain-filename wins; else a non-empty `download` attribute wins; else
  the response name.**
  `isAttachmentWithFilename()` reads only the literal `filename` parameter, so a
  `filename*=`-only attachment does *not* take that branch — which is why the
  attribute won in the GTK run, and why **adding a plain `filename=` would hand
  naming back to Cocoa.**
- WebKit's own `appendFileExtensionIfNecessary` appends *nothing* when the name
  contains a dot or the type is `application/octet-stream` (the default). It
  cannot produce `File.pdf.html`.
- The double extension comes from the WHATWG "suggested filename" rule: a
  *known* claimed type that disagrees with the name's extension causes an
  extension to be **added** for the claimed type.
- `filename*=` alone has been supported since Safari 6 (greenbytes tc2231), so
  the name is not being lost to the header form.

## Live mechanisms

Leading, if the user is in an **installed Home-Screen PWA**:
M-PWA — iOS 16/17 does not run a Safari-style download there. Documented WebKit
bugs: same-scope `<a download>` + `attachment` → an "Open in …" splash with no way
back (290847, reproduced on iOS 16.7.10); the download popup becomes a *preview*
the user must save from (275288, iOS 17); stranded navigation (236943). Apple
closed them as Safari/Web.app UI issues outside open WebKit. If the saved artifact
is the previewed **document**, it is an HTML page carrying the file's name — which
predicts that renaming it does **not** make it open, and that its size is not the
source file's.

If instead the user is in a **Safari tab**, the naming layers above are the whole
story: the name Cocoa derives plus a claimed-vs-named type step. Our claimed type
is `application/octet-stream`, which maps to no extension, so Cocoa should append
nothing and the name should survive — which is consistent with this being
reported from a PWA context rather than from tabs. The unverified possibility is
that Safari's closed save layer substitutes `.html` when it considers the artifact
a document.

Dead: name lost to the RFC 5987 form (Safari 6+); the service-worker cache channel
(not reached); a WebKit-engine append of `.html` (measured absent).

## The two observations that decide it

1. Rename the downloaded `…html` to its real extension: does it open? Yes ⇒ bytes
   are the file, it is a naming-layer problem (M1). No ⇒ the artifact is a preview
   page (M-PWA).
2. Is the app an installed Home-Screen icon or a Safari tab? Plus the file's size
   versus the source.

## Fix, revised

- **Keep** `link.download = exact basename`. Mechanism: our disposition is
  attachment-with-filename only in the *extended* form, which WebKit's
  `isAttachmentWithFilename()` does not recognise, so the non-empty attribute wins
  (shared UIProcess code, not a port quirk). Also harmless in Chromium, where CD
  already wins and is already right.
- **Do not add a plain `filename=`** — that flips precedence to Cocoa's derived
  name plus MIME-extension processing.
- **Do not blanket-replace `application/octet-stream` with the real MIME type.**
  octet-stream is the safe declaration (no extension mapping; early return). A
  *known but mismatched* type is exactly the append trigger (`notes.md` +
  `text/plain` → `notes.md.txt`). Type it only if typed *exactly*.
- **Promote the Blob handoff** for the PWA case: fetch → `Blob` with the accurate
  type → object URL → `a[download="name.ext"]`, revoked on a delay, inside the
  gesture. It is the documented workaround class for exactly these failures. Cost:
  the whole file in page memory, so size-bounded. Caveat: an iOS 17 installed PWA
  may still show a preview instead of an automatic download — but of the **real
  file**, so the subsequent Save-to-Files produces the right artifact.
- **Optionally** `navigator.share({files:[File]})` in the PWA as an explicit
  Save-to-Files path; same buffering cost, bypasses the download UI.
- Honest limit: in an installed PWA part of this is a closed Apple bug. The app
  can guarantee that the artifact offered to iOS is the real file with its real
  name and type; it cannot make iOS's PWA download UI behave like Safari's.
