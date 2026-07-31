I have a complete mechanism-level picture. Working tree is clean, nothing staged, I changed no files. Here is the scout deliverable.

---

# File Workbench: Upload/Attachment Next-Tranche Scout

**Committed recommendation (one line):** After the editor browser proof lands, run **browser/Docker certification of the upload→attach→send→agent-reads-the-file end-to-end claim (desktop + 390×844 mobile, real Pi in Docker)**; bundle retirement of `tests/test_file_upload.py`'s server-global monkeypatch seam as a **subordinate, low-risk rider gated on clean certification**, migrating its coverage into the *already-present* `tests/test_control_routes.py` injected-deps seam plus a new direct `file_upload` module test. Product evidence leads; the seam is debt, not an emergency.

---

## Q1 — Current user promise vs. what is under-evidenced/false

**Implemented promise (narrow):** Tap the paperclip (`#attachBtn`, title "Attach file (max 16MB)") → OS file picker for **exactly one** file (`#imgInput` is `type=file` with **no `accept`, no `multiple`, no `capture`** — `app.js:1230`). The file is read client-side to bytes → base64 → `POST /api/sessions/<sid>/inject_file`. Server stages the bytes to `~/.local/share/codoxear/uploads/<session_id>/<ms>_<safe_name>` (chmod `0600`) and injects a bracketed-paste **text reference** `Attachment N: <absolute_path>\n` into the backend PTY. A badge counts attachments; the session enters `pending_attachment=True` until the next send.

So the real semantic is: **"stage a copy server-side and hand the agent an absolute path,"** not "upload into the session cwd."

**Under-evidenced or likely-false from the browser:**
1. **The target claim is unproven: can the agent actually read the staged path?** The file lands in the **app dir** (`UPLOAD_DIR = app_dir/uploads`, `server_config.py:232`), *outside* the session cwd, and the agent receives an absolute path to it. If a backend's file access is cwd-scoped/sandboxed, the attachment is *delivered-but-unreadable*: HTTP 200 + badge "1", yet the agent cannot open it. Every existing test measures the proxy (endpoint 200 / staged bytes), never the target (agent consumes the file).
2. **Zero browser evidence for the entire surface** (EPISTEMIC confirms: "browser proof is missing"). All current claims are source/unit-level, which the project's own validation norm rejects as acceptance.
3. **HEIC transcode is a desktop/mobile divergence.** Transcode fires only for images that are HEIC or >16MB (`app.js:6690`). iOS Safari decodes HEIC natively; desktop Chrome/Firefox generally do **not**, so `img.decode()`/canvas throws and the user sees `attach error: decode failed`. Fail-loud is preserved, but the desktop-HEIC path is effectively unsupported and unverified.
4. **Failure visibility is terse and unproven in-browser:** oversize→client "file too large" toast (pre-upload, `app.js:6712`); 409 busy; 413/400/502/504 → single-line toasts (`attach error:`, `attachment status unknown; check before retrying`). Never seen rendered.
5. **`pending_attachment` persistence** survives restart (`pending_attachments.json`, `session_store.py:328-337`) and **blocks queueing** until sent/cleared — unproven from the browser, including the `window.confirm()` recovery dialogs (`app.js:6777`, `6849`) that browser automation cannot auto-accept.

## Q2 — Control/data flow (browser → PTY)

```
paperclip #attachBtn.onclick (app.js:6619)   guards: !selected / launchFailed / currentRunning / sending → toast+return
  → #imgInput.click()  (single file, no type filter)
#imgInput "change" (app.js:6633)             re-guards launchFailed/running/sending against captured sid
  → if image & (>max or HEIC): canvas → JPEG (toJpegBlob, tries 5 dim/quality pairs)
  → uploadBlob.arrayBuffer(); if byteLength>max → throw "file too large" (client fail-fast)
  → b64 = bytesToBase64(...)  (app_file_helpers.js:345)
  → api POST /api/sessions/<sid>/inject_file { filename, data_b64, attachment_index }
server_handler.do_POST (server_handler.py:216)
  → handle_control_post_route(... deps=control_route_deps(), manager=server.MANAGER)  (control_routes.py:38)
  → _handle_inject_attachment (control_routes.py:231):
       auth → read_json_body(limit=ATTACH_UPLOAD_BODY_MAX_BYTES)
       validate filename/attachment_index(int, not bool)/data_b64
       manager.attachment_injection_ready(sid)  → KeyError=404 / NotReady=409 / Exception=409 / False=409
       base64.b64decode(validate=True) → 400 on failure
       out_path = deps.stage_uploaded_file(sid, filename, raw)  → ValueError: 413 if "file too large" else 400
       inject_text = deps.attachment_inject_text(idx, out_path) → 400 on ValueError
       seq = "\x1b[200~{inject_text}\x1b[201~"
       resp = manager.inject_attachment_keys(sid, seq)  → KeyError=404 / NotReady=409 / InjectionError=502 / CommitUnknown=504(+commit_unknown)
       → 200 {ok, path, inject_text, broker: resp}
file_upload.stage_uploaded_file (file_upload.py:22): safe_filename → subdir=UPLOAD_DIR/sid → write bytes → chmod 600; boundary via safe_filename basename-strip + startswith(subdir+sep)
session_attachment.SessionAttachmentCoordinator.inject_attachment_keys: input_lock → readiness recheck → inject_keys(track_request_sent=True) → set_pending_attachment(True); malformed/empty/commit_unknown/incomplete → commit_unknown_error
```
Send afterwards: `sendText` computes `allow_pending_attachment` (`app.js:6775`), and the queue path (`sendChoice`) **disables "Later"** when attachments are pending (`app.js:5979`, "attachments can only be sent now").

## Q3 — Source of truth & invariants

- **Path/write boundary (source of truth):** `file_upload.stage_uploaded_file`. `safe_filename` reduces to `Path(name).name` and strips to `[alnum,-_. ]`, so `../../payload.tar.gz`→`payload.tar.gz`; the `startswith(subdir+os.sep)` check is redundant belt-and-suspenders. **Files never enter the cwd** — the "no leakage outside cwd" invariant is satisfied by construction (staging is app-dir-scoped; injection is a text path reference only).
- **Size/type:** server max `ATTACH_UPLOAD_MAX_BYTES` (16MB default), body cap `ATTACH_UPLOAD_BODY_MAX_BYTES` (base64-inflated, `server_config.py:217`); client mirrors it via `__CODOXEAR_ATTACH_MAX_BYTES__` placeholder (`static_routes.py:12`). **No server-side type validation** (any file type accepted).
- **Non-UTF name handling:** *different class than the file-viewer D2/D4 bug.* The filename arrives as a **JSON string** (already valid Unicode), so no `os.walk` surrogate risk. Unicode alnum (e.g. `南京大学_程元_简历.pdf`) survives `str.isalnum()`. No `path_token` machinery is needed here.
- **Failure visibility:** server maps every error to a distinct HTTP status; client maps each to a toast. Contract is present; browser rendering unproven.
- **Readiness authority:** `manager.attachment_injection_ready` derives from the same binary busy/idle authority (`session_readiness.py`), sharing the pre-log relaxation and post-send confirmed-send boundary (EPISTEMIC "Attachment injection shares the pre-log readiness relaxation").
- **Queue/send interaction:** `pending_attachment` blocks queueing (`session_queue.py`: "send the pending attachment before queueing another prompt") and forbids queuing attachments through `sendChoice`.

## Q4 — Exact browser evidence that closes the product claim

The certification must prove the **target**, not the proxy. Minimum decisive set (Docker, real Pi):

**Desktop:**
1. New Session → Pi (healthy) → idle; paperclip enabled, tooltip "Attach file (max 16MB)".
2. Attach a small **text file with a known sentinel string** (automation must drive the hidden `#imgInput` via `setInputFiles`, not click — clicking opens a native dialog the tool can't use). Expect `uploading file...`→`file attached`, badge "1".
3. **Decisive step:** send "Reply with the exact contents of the attached file." Assistant transcript must return the sentinel → proves the agent read the staged absolute path (**closes the real claim**). If it cannot, that is the headline defect (see Q6 stop rule).
4. Failure visibility: (a) >16MB non-image → client "file too large" toast, no request; (b) attach during a live turn → button disabled + tooltip; (c) force a 409/502/504 and confirm the specific toast renders.
5. Queue interaction: attach, start a turn, attempt queue → "Later" disabled, "attachments can only be sent now" toast.
6. `pending_attachment` persistence: attach, don't send, second-server rediscovery → session still `pending_attachment`, queue blocked; resolve via send or `/pending_attachment/clear`.

**Mobile (390×844):**
7. Paperclip touch target ≥44×44 (D3 pattern) and reachable; picker opens.
8. End-to-end read-attachment proof on mobile viewport (use a JPEG in automation since headless can't emit HEIC); badge/toasts readable; soft keyboard doesn't strand the composer.
9. **Record the desktop-HEIC boundary explicitly** (decode-failed toast) or state it as an unreproducible-headless boundary. Do not claim HEIC works cross-platform without evidence.

## Q5 — Is `tests/test_file_upload.py` dangerous now, or debt?

**Debt, not an acute danger — and its clean replacement already exists.** Mechanism:

- **The route logic is already covered cleanly.** `tests/test_control_routes.py` drives the real `handle_control_post_route` → `_handle_inject_attachment` with an explicit injected `ControlRouteDeps` (fake handler/manager, injected `stage_uploaded_file`/`attachment_inject_text`/`json_response`), asserting the readiness-gate 409 and the stage+bracketed-paste 200. This is exactly the converted-seam pattern used for queue/diagnostics/etc.
- **What `test_file_upload.py` adds is a *liability*, not unique safety.** `TestInjectFileRoute` reaches `server.Handler.do_POST` and monkeypatches five module globals (`MANAGER`, `UPLOAD_DIR`, `_now`, `_require_auth`, **`_json_response`**) plus stubs three private handler methods (`_parse_prefixed_request_path`, `_handle_voice_post`, `_read_json_body`). `patch.object(server,"MANAGER",...)` is the exact anti-pattern named in `ARCHITECTURE.md` known-failure-modes. Replacing `_json_response` **blinds it to real serialization**, and coupling to `do_POST`'s internal call order forces lockstep edits on any dispatch refactor. `TestStageUploadedFile` patches `server.UPLOAD_DIR`/`server._now` purely out of habit — the impl `file_upload.stage_uploaded_file(...)` already takes `upload_dir`/`now_fn`/`max_bytes` explicitly and can be tested with **zero patching**.
- **Why not acute:** because the route is independently covered by the clean seam, `test_file_upload.py` is not currently masking a live break; its risk is latent fragility + false confidence. That makes it *cheap-to-retire debt with a ready replacement* — act on it as a rider, not an emergency.

**Direct seam that should replace it:**
1. **Pure module test** (new `tests/test_file_upload_module.py` or a rewrite of `test_file_upload.py`): call `file_upload.safe_filename` / `stage_uploaded_file(..., upload_dir=tmp, now_fn=lambda:.., max_bytes=..)` / `attachment_inject_text` directly. Covers binary-bytes+suffix, `..`-traversal reduction, generic-name fallback, oversize `ValueError`, Unicode name, inject-text label/newline, non-positive index. No server globals.
2. **Expand `tests/test_control_routes.py`** to the full status matrix through the existing injected seam: filename-missing 400, attachment_index bool/non-int 400, data_b64-missing 400, invalid-base64 400, stage oversize 413 vs other 400, inject-text ValueError 400, ready→KeyError 404 / NotReady 409 / generic-Exception 409, inject→KeyError 404 / InjectionError 502 / CommitUnknown 504(+`commit_unknown`), and unauthorized. Optionally assert the 200 body is `json.dumps`-able (recovers the one real coverage `_json_response`-patching loses).
3. Optional: retire the grep-style `test_file_upload_module_source.py` once (1)+(2) are executable — but that is separable and should be an explicit decision, not silent.

## Q6 — Implementation contract to dispatch next (if evidence finds a real gap)

Two ordered work items. **Product evidence is main-agent/browser-verifier work; code/test work is executor work.**

**Item A — Browser/Docker upload/attachment certification (main agent / browser verifier).** Deliver the Q4 evidence set. This is the spine of the tranche and gates everything else.

**Item B — Executor contract: retire the monkeypatch seam (dispatch only after Item A passes clean).**
- *Goal:* remove `tests/test_file_upload.py`'s server-global/do_POST monkeypatch coverage while preserving+expanding behavior via the direct module test + `test_control_routes.py`.
- *Read first:* this scout, `tests/test_control_routes.py`, `codoxear/control_routes.py`, `codoxear/file_upload.py`, EPISTEMIC.md.
- *Files in scope:* `tests/test_file_upload.py` (rewrite→direct module test or delete), `tests/test_control_routes.py` (expand matrix); optionally `tests/test_file_upload_module_source.py` (retire, explicit). **No production code** in Item B.
- *Hard constraints:* no `git add -A`; functional/test commit separate from memory; no out-of-scope edits; **do not run the full local suite while any other contract/verifier is in flight**.
- *Validation:* `python3 -m pytest -q tests/test_control_routes.py tests/test_file_upload_module.py` (targeted) → then full `python3 -m pytest -q` **only when no verifier/contract is live** → Docker `test` + `smoke` at a free `190NN` port.
- *Stop rules:* if any migrated case can't reproduce a prior assertion through the clean seam, STOP and report the missing coverage (don't delete blindly).

**Item C — Contingent product fix (executor, dispatched only if Item A names a mechanism).** Most likely mechanism: agent cannot read the app-dir staged path.
- *Stop-for-decision:* moving staging from app-dir into the session cwd, or changing the injected reference format, **changes a product invariant** ("no cwd writes") and requires explicit approval — do not do it unilaterally.
- *Also stop-for-decision before adding drag-drop / multi-file / paste-to-attach / `accept` / `capture`:* these are **features**, not defects. Only pursue if certification proves the single-file paperclip flow fails a real target workflow AND a decision approves.
- *Likely files if approved:* `codoxear/file_upload.py`, `codoxear/control_routes.py`, `codoxear/static/app.js`, `codoxear/static/app_file_helpers.js`, `codoxear/session_attachment.py`.

---

## Ranked findings (by product risk)

| # | Finding | Evidence | Severity |
|---|---------|----------|----------|
| **F1** | **Attachment usefulness unproven — agent may not be able to read the app-dir staged path.** Staging is outside cwd; agent gets an absolute path only. Every test measures proxy (200/bytes), not target (consumed). | `server_config.py:232`, `file_upload.py:38`, `control_routes.py:277` | **Blocking for the product claim** |
| **F2** | **Zero browser evidence for the whole upload/attach surface, desktop+mobile.** Source/unit only; violates project acceptance norm. | EPISTEMIC "browser proof is missing" | **Blocking** |
| **F3** | **HEIC transcode is a desktop/mobile divergence** (iOS decodes; desktop generally fails → "decode failed" toast). Fail-loud OK; desktop-HEIC unsupported+unverified. | `app.js:6653-6690`, `app_file_helpers.js:332` | Impairing (bounded) |
| **F4** | **`test_file_upload.py` = named monkeypatch debt with a ready clean replacement** (`test_control_routes.py`). Latent fragility (blinds serialization, couples do_POST), not an acute break. | `test_file_upload.py:63-110`, `test_control_routes.py:171-211` | Debt (act as rider) |
| **F5** | **Staged uploads are never pruned.** Session delete unlinks only sock+sidecar; `uploads/<sid>/` (0600) grows unbounded in the app dir. | `session_cleanup.py:31-32` (no `uploads` handling) | Low (hygiene) |
| **F6** | **No drag-drop / multi-file / paste-to-attach / `accept` / `capture`.** Task's "drop" model isn't implemented; product is single-file paperclip only. | `app.js:1230`; no `drop`/`dataTransfer` handlers anywhere | Feature gap (do not add speculatively) |
| **F7** | **Millisecond filename collision** — two same-name files staged in the same ms overwrite (`{int(now*1000)}_{name}`). Not human-reachable via picker. | `file_upload.py:41` | Negligible |

**Note:** `send_attachment_file_response` (`file_response.py:165`) is the misleadingly-named **file-viewer download** of *cwd* files (already certified in D4/Git Workbench) — it is unrelated to upload staging. "Download" in Workbench item 6 is already covered; the genuinely new target is upload/attach only.

---

## Committed recommendation (detailed)

Sequence the next tranche as **product-evidence-first, seam-retirement-as-rider**:

1. **Certify upload/attachment in the browser** (Item A) once the editor proof lands — desktop + 390×844, real Pi in Docker, proving **attach→send→agent returns the sentinel content** plus failure/queue/persistence visibility. This closes F1+F2, the only blocking findings.
2. **If certification passes clean,** dispatch the executor to **retire `test_file_upload.py`'s monkeypatch seam** (Item B) into the existing `test_control_routes.py` seam + a direct `file_upload` module test. This satisfies PROMPT items 6/12 without release risk.
3. **If certification exposes F1 (unreadable staged path)** or any real defect, fix the mechanism first (Item C) — but **stop for a decision** before changing the staging location/reference format or adding drop/multi/paste, because those alter product invariants or add surface.

This ordering honors the roadmap explicitly: product-critical evidence precedes bounded non-product debt; the test seam is never the headline and never a release blocker by default; features aren't added without proven user need.

---