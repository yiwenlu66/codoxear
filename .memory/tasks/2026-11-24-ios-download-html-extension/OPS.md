# OPS — append-only evidence trail

## 2026-11-24

### Source facts established (read from the checkout at /home/yiwen/codoxear)

- `codoxear/static/app_file_download.js` (`createFileDownloadRuntime.download`):
  `<a href>` + `link.download = ""` + `rel="noopener"`, appended to body,
  `click()`, removed. Returns bool.
- `codoxear/static/app_file_picker_ops.js:116`:
  `fileDownloadRuntime.download(activeFileDownloadApiPath())` — the filename is
  never given to the runtime; the browser is the only namer.
- `codoxear/static/app_file_viewer_operations.js:751` (`activeFileDownloadApiPath`):
  builds `/api/sessions/<sid>/file/download?path=…`; the URL's last path segment
  is the literal `download` and carries no extension.
- `codoxear/file_get_routes.py:691` `_handle_session_file_download` →
  `deps.send_attachment_file_response(...)`; error paths map to JSON
  (404/403/400/409) via `_map_resolve_error`, never HTML.
- `codoxear/file_response.py:178` `send_attachment_file_response`: `200`,
  `Content-Type: application/octet-stream`, `Content-Length`,
  `Content-Disposition: <given>`, `Cache-Control: no-store`, `no-cache`,
  `Expires: 0`. No `Accept-Ranges`, no `X-Content-Type-Options`.
- `codoxear/file_view.py:110` `download_disposition`:
  `attachment; filename*=UTF-8''<urllib.parse.quote(safe name)>` — RFC 5987
  form only.
- `git log -S'application/octet-stream'` on file_response.py → `b8c58e38`
  ("fix: stream downloaded files instead of buffering"); `git show b8c58e38^:codoxear/server.py`
  (line ~5918) shows the same octet-stream + `filename*` header pair before the
  streaming rewrite. Long-standing, not a regression from streaming.
- `c8330b6d` ("Prevent download click from blanking app via beforeunload
  teardown") added `link.download = ""` precisely because a plain anchor click
  navigated and `cleanupApp` emptied the root.
- `codoxear/static/service-worker.js` fetch handler: intercepts same-origin GET
  in scope; exclusions are `cache === "no-store"`, `attachments/upload-status`,
  `file/complete`, `(office|media)_preview`, any request with a `range` header,
  and `/live`. A download request is therefore intercepted: `mode === "navigate"`
  → `networkFirst(HTML_CACHE, {ignoreSearch:true})`, else
  `networkFirst(API_CACHE)`; `storeResponse` writes any 200 body into the cache.
- `.memory/project/ARCHITECTURE.md:350` confirms intent: "non-Range
  blob/raw/download caching is unchanged".
- `resolveAppUrl` (`app_application.js:72`) = `new URL(path.slice(1), appBaseUrl)`
  — preserves query, same origin.
- `tests/test_file_inspect.py:2286` pins the disposition string to the
  `filename*`-only form (a fix that adds a `filename=` fallback must update it).
- No `download_url` producer exists in Python: `app_file_viewer_operations.js:182`
  always falls back to `activeFileDownloadApiPath()`, so the viewer's download
  always hits `/file/download`.

### WebKit source read (raw.githubusercontent.com/WebKit/WebKit/main)

- `Source/WebCore/platform/network/ResourceResponseBase.cpp:361`
  `suggestedFilename()` → `downcast<ResourceResponse>(*this).platformSuggestedFilename()`.
- `Source/WebCore/platform/network/cocoa/ResourceResponseCocoa.mm:195`
  `platformSuggestedFilename()` → `[nsURLResponse suggestedFilename]`
  (i.e. CFNetwork decides, using Content-Disposition then the URL).
- `Source/WebKit/NetworkProcess/Downloads/PendingDownload.cpp:254`
  `DownloadProxy::DecideDestinationWithSuggestedFilename(response, response.suggestedFilename())`
  — the suggested name handed to the UI is the *response's*.
  Line 76: `m_networkLoad->setSuggestedFilename(suggestedName)` — the value that
  came from the `download` attribute is pushed into the load, and
  `DownloadManager::startDownload(... const String& suggestedName, FromDownloadAttribute ...)`
  carries a `FromDownloadAttribute` flag. So a non-empty `download` attribute is
  a first-class name source that outranks nothing else being consulted.
- `Source/WebKit/NetworkProcess/NetworkLoad.cpp:373` `setSuggestedFilename`
  → `m_task->setSuggestedFilename(...)`.
- `Source/WebCore/html/HTMLAnchorElement.cpp:552,580` — the anchor parses the
  attribute via `parseDownloadAttribute(...)` and passes it to
  `navigateHyperlink(...)`; `FrameLoader.cpp:1746` comment: "Anchor target is
  ignored when the download attribute is set since it will download the
  hyperlink rather than follow it" (so `target="_blank"` is irrelevant once the
  attribute is present); `FrameLoader.cpp:1747` distinguishes
  `downloadAttribute().isNull()` (absent) from empty.
- `Source/WebCore/platform/network/HTTPParsers.cpp:345`
  `filenameFromHTTPContentDisposition` accepts ONLY the exact key `filename` —
  it skips `filename*`. (This is WebCore's own helper; the Cocoa path defers to
  CFNetwork, which is closed source, so this is suggestive, not decisive.)

### Community evidence (mechanism class)

- stackoverflow.com/questions/2734423 (via r.jina.ai): response with
  `Content-Disposition: attachment; filename="File #1.pdf"` **and**
  `Content-Type: text/html` is saved by Safari as `File #1.pdf.html`; every
  answer attributes it to the MIME type. Confirms Safari appends a
  type-derived extension to the Content-Disposition name.
- apple.stackexchange.com/questions/189419 (via r.jina.ai): `file.docx` from one
  specific site becomes `file.docx.html` in Safari only; the answer: the site's
  `Content-Type`/`Content-Disposition` headers decide what Safari appends.
- code2care.org: `.zip` downloaded as `.zip.html` caused by a misspelled/
  duplicated `Content-Type` header resolving to `text/html`.
- Consistent picture: the suffix is Safari substituting the extension it
  believes belongs to the response's declared type, on top of the name it
  obtained elsewhere.

### Commands run (host, read-only)

```
grep -rn "download" codoxear/static/*.js codoxear/*.py
git log --oneline -- codoxear/static/app_file_download.js
git log -1 --format=%B c8330b6d
git show b8c58e38 [-- codoxear/file_response.py codoxear/server.py]
git show b8c58e38^:codoxear/server.py | grep -n _read_downloadable_file -A 25
curl raw.githubusercontent.com/WebKit/WebKit/main/<file>   (see list above)
```

### Delegated measurements (pending at time of writing)

- `4227fa9f` — Docker + Chromium: raw response headers for
  `/file/download`, service-worker involvement, saved filename, body hash.
- `dd13c403` — sourced answers on Safari/WebKit download-naming rules,
  `filename*` support, empty `download` attribute, SW effects, iOS specifics.
- `47c9e3de` — Playwright WebKit (GTK) + Chromium control: saved filename for
  six header/attribute variants (A–F) against a local server.

### 2026-11-24 — delegated experiment 47c9e3de (Playwright WebKit 26.5 GTK+WPE / Chromium 151, Docker)

72 downloads, 6 header/attribute variants × 2 bodies × SW/no-SW × 3 engines.
Variants: A octet-stream+`filename*` only; B octet-stream+both params; C
`application/zip`+both; D `text/html`+`filename="foo.zip"`; E no download
attribute; F `download="explicit.zip"` + A's headers.

Findings that changed the model:

1. **A `.html` suffix was not reproduced by any engine, including variant D
   (`Content-Type: text/html` + CD `filename="foo.zip"`).** WebKit and Chromium
   both suggested `foo.zip`. So the append is NOT in WebKit's download layer.
2. **`filename*` alone is honoured** by WebKitGTK 26.5 and WPE 26.5 (`foo.zip`),
   and adding the ASCII `filename=` changed nothing. So M2's premise ("name lost
   because only RFC 5987 is sent") is not reproduced in WebKit either.
3. **Attribute downloads never reach the service worker's fetch handler** in
   either engine (SW was demonstrably controlling: a `/probe` fetch arrived with
   mode `cors`; the download produced no fetch event). Chromium still marks the
   network request `Sec-Fetch-Mode: navigate`, `Sec-Fetch-Dest: empty`, so
   `navigate` does not imply an ordinary document navigation.
   → **Retracts the earlier claim that `/file/download` bodies are cached into
   `codoxear-api-v1`/`codoxear-html-v1` and that `ignoreSearch` could hand back
   another file's response on the button path.** That channel is not reachable
   from an anchor download in these engines; it would need a genuine navigation
   to the URL.
4. **Name precedence differs by engine**: variant F gave `explicit.zip` on WebKit
   and `foo.zip` (the header's name) on Chromium — i.e. **WebKit prefers the
   `download` attribute; Chromium prefers Content-Disposition**. Setting a real
   attribute name is therefore the authoritative lever on WebKit/iOS and a no-op
   where the header is already right.
5. Body bytes matched the served bytes in all 72 runs; suggested == on-disk name
   in every case (native artifact basenames are GUIDs; named copies were made by
   `save_as`, so this is engine-level, not Safari-UI-level, evidence).

Linux engines cannot speak for iOS: WebKitGTK/WPE do not use CFNetwork and have
no Safari download UI, so the `.html` must be produced in Apple's Cocoa layer
(CFNetwork `NSURLResponse.suggestedFilename` or the Safari/Files save path) —
the one layer no local engine reproduces. That is exactly why the suffix is
iOS-only. Local experiments can bound what the engines do; they cannot prove the
iOS rule.

### 2026-11-24 — delegated experiment 4227fa9f (real app in Docker: server + SPA + Chromium 151)

Setup: checkout `ecfc04fa`, `scripts/codoxear-docker-sandbox start` (container
`codoxear-download-wire-4227`), browser in
`codoxear-download-browser-4227`, login via `POST /api/login` inside Docker,
playwright `Browser.setDownloadBehavior` so the browser chose the saved name.

Wire (verbatim `curl -D -`, three files):

```
HTTP/1.1 200 OK
Content-Type: application/octet-stream
Content-Length: 1057
Content-Disposition: attachment; filename*=UTF-8''payload.bin
Cache-Control: no-store
Pragma: no-cache
Expires: 0
```

(and `notes.txt` 43 B, `notes.md` 58 B with the same shape — no `filename=`, no
nosniff, no Accept-Ranges.)

Body integrity: sha256 identical across source file, curl body, SW-enabled
browser download and SW-bypassed browser download for all three files
(`c0d5545b…`, `344390ba…`, `0bda1583…`).

Real UI: logged in, selected the session, opened each file, clicked
`#fileDownloadBtn` on the actual SPA. **Saved names: `payload.bin`, `notes.txt`,
`notes.md`** — original extensions, byte-identical — both with the SW controlling
and with `Network.setBypassServiceWorker(true)`. Observed anchors: `download=""`,
`rel="noopener"`, same-origin URL.

Service worker: an observation-only wrapper around the active worker's
`networkFirst` recorded **zero calls for the six real downloads**; a plain
`fetch()` of the same endpoint immediately produced a worker call with
`cacheName: "codoxear-api-v1"`, `fromServiceWorker: true`. NetLog captured the
actual download requests: `Sec-Fetch-Mode: navigate`, `Sec-Fetch-Dest: empty`,
response `application/octet-stream` + the correct disposition — i.e. Chromium
issues them outside the page's SW-controlled fetch even though the mode text says
navigate. Confirms 47c9e3de independently, now on the real app.

URL resolution: `resolveAppUrl` executed in-browser returns the same-origin URL
unchanged, and with `CODEX_WEB_URL_PREFIX=/wire-prefix` returns
`/wire-prefix/api/sessions/…/file/download?path=…` with the query preserved
(`path`, `path_token`, `git_path`); the prefixed viewer download saved
`notes.txt` correctly.

Consequence for the model: the server response, the URL construction, the prefix
handling and the cache path are all *correct* for a Chromium-class browser, and
the bytes always match. Nothing left to blame below the iOS layer.

### 2026-11-24 — delegated research dd13c403 (sourced WebKit/Safari download-naming rules)

Sources with code quoted per Safari 15/16/17 branch. Findings that changed the plan:

1. **Two naming layers.** Cocoa `NSURLResponse.suggestedFilename` (Apple docs: order
   is Content-Disposition → URL last path component → host → "unknown", and "in most
   cases, this property appends the proper file extension based on the MIME type"),
   reached via `ResourceResponseCocoa.mm:195`. Then WebKit's
   `DownloadProxy.cpp` precedence, and a *narrow* helper
   `MIMETypeRegistry::appendFileExtensionIfNecessary` which returns the name
   unchanged when it `contains('.')` **or** when the MIME type equals
   `defaultMIMEType()` = `application/octet-stream`.
2. **What produces `File.pdf.html` is the standards algorithm, not that helper.**
   WHATWG "getting the suggested filename": when the claimed type (Content-Type) is
   known and differs from the type implied by the filename's extension, *add* an
   extension for the claimed type.
   → The trigger is claimed-vs-named type disagreement, generic per MIME mapping.
3. **`DownloadProxy` precedence (shared UIProcess code, not per-port):**
   ```cpp
   if (response.isAttachmentWithFilename()) { suggestedFilename = response.suggestedFilename(); }   // CD wins
   else if (!m_suggestedFilename.isEmpty()) { suggestedFilename = m_suggestedFilename; }            // attribute wins
   else if (suggestedFilename.isEmpty()) suggestedFilename = response.suggestedFilename();
   ```
   and `PolicyChecker.cpp`: `suggestedFilename = action.downloadAttribute().isEmpty() ? nullAtom() : action.downloadAttribute();`
   `isAttachmentWithFilename()` uses WebKit's limited
   `filenameFromHTTPContentDisposition()` (literal `filename` key only).
   → **With our `filename*=`-only disposition the attachment branch is NOT taken, so
   a non-empty `download` attribute wins.** This also explains the GTK variant-F
   result (attribute beat the header) without appealing to a port difference, and
   it means adding a plain `filename=` would *transfer* naming authority to Cocoa
   (and to the MIME-extension algorithm).
4. **`filename*=` alone is supported** (greenbytes tc2231: Safari 6 passes
   `attwithfn2231utf8`; `attfnboth` → Safari picks the RFC 2231 value). Kills M2.
5. **octet-stream is the safe declaration at the WebKit layer** (explicit early
   return). A *known but mismatched* type is the trigger for extension appending —
   so serving `text/plain` for `notes.md` risks `notes.md.txt`. Retracts the
   blanket "serve the real MIME type" advice.
6. **Installed Home-Screen PWA is a documented-broken download surface**:
   WebKit bug 290847 (iOS 16.7.10: same-scope `<a href download>` + attachment →
   "Open in …" splash, no way back); 275288 (iOS 17 installed PWA turns the blob
   download popup into a *preview* the user must save from); 236943 (stranded with
   no back navigation). WebKit engineers closed these as Safari/Web.app UI issues
   outside open-source WebKit. iOS 13 added `download` attribute support.
   → New leading hypothesis: in an installed PWA the artifact the user saves is the
   *previewed document*, carrying the file's name with `.html` appended — which
   also predicts that renaming it does NOT make it open.
7. Documented workaround class: fetch → `Blob` with the accurate type → object URL
   → `a[download="name.ext"]`, revoke on a delay, inside a user gesture
   (WebKit engineer's suggestion for SW-generated content, bug 202142; practitioner
   reports for iOS 13 PWAs). Caveats: Safari 17 PWA still previewed the blob; iOS 13
   stopped sniffing `application/octet-stream` responses; buffering cost.

### 2026-11-24 — implementation (commit 9c875932, pushed to origin/main)

Decision from `EPISTEMIC.md`: ship the two levers with mechanism, skip the two
the research retired (plain `filename=`, real MIME type), and make the iOS
transfer page-owned because the platform's own save path is the broken layer.

Client:
- `app_file_download.js`: `downloadFileName(path)` (basename, control chars
  stripped, lone surrogates made well-formed, `.`/`..`/empty → ""),
  `download(apiPath, filename)` sets the anchor's `download` attribute,
  `downloadActive(apiPath)` chooses streaming vs buffered. Buffered:
  `fetch(cache:"no-store", credentials:"same-origin")` → cap-enforcing reader
  (32 MiB, cancels past the cap) → `Blob(application/octet-stream)` →
  `createObjectURL` → click → revoke after 60 s. 401 → `reason:"auth"` toast;
  other failures → transport toast; never clicks on failure; in-flight guard.
- `app_file_picker_ops.js`: the button dispatches `downloadActive(activeFileDownloadApiPath())`.
- `app_file_ops.js`: wires `resolveDownloadFileName` (viewer identity, guarded on
  a null controller), fetch/URL/Blob/setTimeout seams, `onError: setToast`.
- `app_application_composition.js`: `shouldBufferDownload: () => iosViewportController.isIOS()`.
- `app_wiring.js`: select lists extended (download options + file ops options).
- `app_file_viewer.js`: the "Download original file" fallback link now carries
  `downloadFileName(rel)`; `downloadFileName` re-exported.

Tests: new `tests/test_file_download_runtime.py` (12 cases: naming/sanitization,
streaming, no-file, non-buffering platform, missing buffering deps, blob handoff
with fetch init + object-URL lifetime, oversize fallback with reader cancel, auth
failure, transport failure, in-flight guard). Updated
`tests/test_file_viewer_source.py` (attribute is now the real name) and
`tests/test_frontend_file_picker_ops.py` (button → `downloadActive` with the
viewer's path; harness rewritten around a shared option-bag template).

Docs: AGENTS.md gains "The app owns the download filename" and "iOS buffers the
transfer into a page-owned blob", including why the disposition stays
`filename*`-only and the response stays octet-stream.

Local gates: `scripts/check_wiring.py` green; `node --check` on the edited
modules; `esbuild --bundle` to /tmp builds and the bundle contains
`downloadActive`/`shouldBufferDownload`. Full suite: 3056 passed with two
pre-existing failures that also fail on HEAD (`test_hint_mode_coverage::
test_rendered_shell_interactives_are_declared_or_explicitly_excluded` —
`draftUndoBtn`; `test_pi_prompt_receipt::
test_bridge_writes_user_ingestion_receipt_without_materializing_rollout`).

### 2026-11-24 — deployed

`scripts/deploy.sh fc8b6acd` succeeded: detached worktree at
`~/.local/share/codoxear/deploy` moved to the commit, esbuild rebuilt
`dist/app.bundle.js` (contains `downloadActive`), `check_wiring.py` green in the
snapshot, pipx reinstalled, unit repointed, `codoxear-server.service` restarted,
and the script's own checks passed ("smoke test passed: initial load and reload
rendered a session card").

Live health boundary (the sanctioned check): `GET /` → 200, unauthenticated
`GET /api/sessions` → 401, service active. Deployed bundle contains the new
runtime.

Follow-up commit `fc8b6acd` guards the name resolver when the viewer controller
is not yet assembled (`fileViewerController` null → empty name → the browser's
own derivation), because the resolver runs per click.

Not verifiable on this machine: whether iOS Safari now saves the intended name.
That requires the user's device; the local engines cannot speak for Cocoa or for
the installed-PWA save path (see EPISTEMIC.md).

### 2026-11-24 — delegated verification a6ce9539 (real SPA, Docker, both UA branches)

Setup: `codoxear-sandbox:latest` (server, port 19942) + `codoxear-safari-video-playwright:1.62.0`
(Chromium 151); the verifier copied the source to /tmp and rebuilt that copy's
bundle, seeded three files in the session cwd, logged in, opened each file, and
clicked the real `#fileDownloadBtn` in two contexts (desktop UA, iPhone 17 UA).

Results (all six downloads):
- Saved names: `payload.bin`, `notes.txt`, `notes.md` in **both** contexts; sha256
  equals the seeded bytes in all six (`c0d5545b…`, `344390ba…`, `0bda1583…`).
- Desktop: streaming. Anchor was the HTTP `/file/download` URL with
  `download="<name>"`; Chromium NetLog shows the three browser GETs with 200 and
  matching body lengths; no page fetch.
- iPhone UA: buffered. `resource_type: "fetch"` for every `/file/download`
  request, init observed `cache: "no-store"` + `credentials: "same-origin"`,
  response `from_service_worker: false`, clicked anchor href was `blob:http://…`
  and matched the download URL.
- Service worker: page was controlling; after every download the caches were
  `codoxear-html-v1`, `codoxear-api-v1`, `codoxear-static-v1` with **zero**
  `/file/download` keys (13 ordinary API keys present). Confirms `no-store` keeps
  the body out of Cache Storage.
- `bufferLimitBytes` default 33554432 confirmed by inspection (no oversized
  fixture run).
- Noise observed and not explained: Playwright logged `net::ERR_ABORTED` for the
  iPhone-UA fetches despite HTTP 200, complete NetLog byte counts and matching
  blob downloads; also CSP/font messages, 409s, clipboard permission errors and
  unrelated "Canceled" page errors. Bounded: the downloads themselves are
  byte-correct, and the same noise appears in the desktop run's console.

Still unverified (and unverifiable here): the native iOS save step — Cocoa naming
and the installed-PWA download UI.
