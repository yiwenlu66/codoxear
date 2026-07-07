# Clean-room adversarial review — Capture attachment producer

**Repo:** `/home/yiwen/codex-web-product-recovery` (branch `recovery/product-gaps`)
**Commits reviewed:** `98880bc` (functional) and `c7fd396` (browser proof)
**Mode:** read-only. No edits, no staging, no commits made by this review.
**Note on tree state:** During review a concurrent session advanced `HEAD` to `394f9c0 Record capture producer proof memory` (touches only `.memory/.../EPISTEMIC.md` + `OPS.md`, no code/proof) and left two untracked files (`docker-gate-failure-note.txt`, `docker-test-19358.txt`). Both reviewed commits remain ancestors of HEAD; the code/proof under review is unchanged by `394f9c0`. The untracked files are not mine.

## Verdict: **ACCEPTED** (2 nonblocker follow-ups)

The change implements capture as a client-only `File` producer that feeds the shared `stageFiles()` path into the single existing `/api/sessions/<sid>/inject_file` route. It adds no backend route/state/PTY/key/send write; the send boundary is untouched and remains the only commit. All required attach blockers apply at both the pre-open gate and inside `stageFiles()`. Badge/chips derive from server staged-list truth, and confirmed send clears the staged list. The browser proof discriminates zero pre-send backend commit from a single send-boundary commit.

Two nonblockers: a stale mobile-layout CSS comment (now under-counts composer controls) and a product-semantics observation about the camera button on desktop. Neither breaks an accepted invariant.

---

## Findings by review question (mechanism + file/line evidence)

### Q1 — Visible capture control + hidden capture-capable input, paperclip unchanged → **PASS**
- `captureBtn` added as a visible `<button type="button">` with camera icon, `title`/`aria-label` "Capture photo": `app.js:1242-1249`. Camera glyph added to the shared icon map: `app_display.js:203-204` (`iconSvg("camera")`; `app.js:656` delegates to `codoxearDisplay.iconSvg`).
- Hidden capture-capable input: `el("input", { id: "captureInput", type: "file", accept: "image/*", capture: "environment", style: "display:none" })` at `app.js:1256`. `createElement` maps unknown keys via `setAttribute` (`app_dom.js:4-11`), so `accept`/`capture` land as real DOM attributes — confirmed live in the proof (`captureAccept:"image/*"`, `captureAttr:"environment"`).
- Paperclip untouched: `attachBtn` markup, `imgInput` (with `multiple`), and the picker `change` listener (`source:"picker"`, `app.js:6939-6945`) are unchanged in the diff. `captureInput` intentionally omits `multiple` (single-photo capture).

### Q2 — Routes through `stageFiles(...,{source:"capture"})` only; exactly one `/inject_file` inside `stageFiles`; no backend write → **PASS**
- `captureInput` `change` → `await stageFiles(files, { sid, source: "capture" })` only (`app.js:6959-6965`). `captureBtn.onclick` only opens the input; it issues no route call (`app.js:6948-6958`).
- Exactly one `/inject_file` occurrence in `app.js`, at line `6899`, inside `stageFiles()` (`stageFiles` spans `6853-6944`). Verified by `grep -c` (=1) and by the new test assertion `source.count('/inject_file') == 1`.
- No backend files changed. `git show --name-status 98880bc` = `codoxear/static/app.js`, `codoxear/static/app_display.js`, `tests/test_attach_button_source.py` only. No `.py` server/broker/sessiond route/state/PTY/key/send added. The only `/send` call (`app.js:7072`) and `keys` machinery are pre-existing and untouched by the diff; the capture block contains neither.

### Q3 — Shares `attachmentBlockerForSession()` before opening input and inside `stageFiles()` before route calls → **PASS**
- Pre-open gate: `captureBtn.onclick` calls `attachmentBlockerForSession(sid, sessionInfo)` and returns on a nonempty blocker before `captureInput.click()` (`app.js:6951-6957`) — same shape as `attachBtn.onclick` (`6928-6937`).
- Inside `stageFiles`: `attachmentBlockerForSession(sessionId, sessionInfo)` is checked and returns `false` before any `/inject_file` call (`app.js:6857-6862`).
- Full blocker set is centralized (`app.js:6689-6698`): no session, failed launch, unknown send, orphan recovery, orphan queue recovery, `currentRunning`, `sending`. Capture inherits all of them via the shared function — no capture-specific bypass.

### Q4 — Meaningful no-name image fallback names; coherent compression → **PASS**
- `capturedFileName(file, index, seed)` (`app.js:6840-6851`) → `captured-${seed}${suffix}.${ext}` with `ext` from MIME (`jpg` default; `png/webp/gif/heic/heif/avif`). Applied only when `f.name` is falsy: `uploadName = f.name || (... producer === "capture" ? capturedFileName(...) : "file")` (`app.js:6876`).
- Compression coherence: image/HEIC detection runs on the **raw `File`** (`attachmentLooksLikeImage`/`attachmentIsLikelyHeic` read `file.type`/`file.name`, `app_file_helpers.js:400-411`), not the computed fallback name, so the fallback name never distorts detection. When compression triggers, the name is rebuilt as `${safeAttachmentStem(uploadName)}.jpg` (`app.js:6879-6880`), keeping stem and extension consistent with the actual JPEG bytes. Verified coherent across: no-name JPEG (→`captured-<seed>.jpg`), no-name HEIC (→`captured-<seed>.heic` then →`.jpg` on recompress), named HEIC, and oversized JPEG. Typeless-file edge case behaves identically to existing paste/drop/picker producers (no capture-specific regression); `accept="image/*"` makes it unlikely.

### Q5 — Truthful state/title/disabled for both paperclip and capture → **PASS**
- `syncAttachButtonState()` (`app.js:6699-6717`) computes one `attachBlocker`, then sets `disabled`/`title`/`aria-label` on **both** `attachControl` and `captureControl`, with distinct truthful labels: `Attach file (max …)` vs `Capture photo (max …)`. Both are disabled together whenever a blocker is present.
- Initial `!selected` path also disables both controls (`app.js:6721-6729`).
- Proof confirms live truth: `captureTitle:"Capture photo (max 16.0 MB)"`, `captureDisabled:false` when unblocked.
- Minor stylistic inconsistency (not a defect): `captureBtn`/`captureControl` are guarded with `if (...)` in two spots while `attachBtn` is used unguarded; both elements are always rendered together, and `captureBtn.onclick` is assigned unguarded like `attachBtn.onclick`, so behavior is consistent.

### Q6 — Proof exercises the real `change` listener and distinguishes zero pre-send backend commit from send-boundary commit → **PASS**
- Real-listener path is proven indirectly but dispositively: the server staged entry is `captured-1783381957888.jpg` (`browser-after-capture.json`, `combined-evidence.json`). `capturedFileName` is reachable **only** via `stageFiles(source:"capture")`, and `source:"capture"` is passed **only** from the `captureInput` `change` listener (`app.js:6964`). `stageFiles` is a closure inside composer init, not attached to `window` (verified: no `window.stageFiles`/`.stageFiles =`), so it cannot be invoked from a browser `evaluate` except through the wired handler. Therefore a `captured-` staged name implies the real change listener fired against the unmodified app.
- Zero pre-send backend commit: `docker-calls-after-capture-compact.json` → `send_count:0`, `keys_count:0` (only 891 `state` calls). Browser shows one chip, badge `1`, `pending_attachment:true`, textarea empty.
- Single send-boundary commit: `docker-calls-after-send-compact.json` → `send_count:1`, `keys_count:0`, single payload begins `Attachment 1: /home/tester/.local/share/codoxear/uploads/capture-proof/…jpg\nplease process captured photo`.
- Confirmed-send clears staged: `browser-after-send.json` → `chips:[]`, `badge:""`, `textarea:""`, api `attachments:[]`, `pending_attachment:false`.
- Auth gate active pre-login (`api-me-before-login.status:401`). Fake broker advertised `sync_send:true`/`key_write_errors:false` (`fake_capture_session.py`).
- Evidentiary gap (non-fatal): the verbatim browser driver JS that set `#captureInput.files` and dispatched `change` is **not** saved among artifacts (only the broker fixture `fake_capture_session.py` is). The conclusion holds regardless because of the `captured-` filename + closure scoping above; recommend saving the driver snippet in future proofs for full independent reconstructability.

### Q7 — Regression risk (tests/source shape, mobile space, browser compat, semantics) → **PASS with 2 nonblockers**
- **Tests/source shape:** `node --check` passes for `app.js` and `app_display.js`. Independently re-ran `tests/test_attach_button_source.py` (8 passed) and `tests/test_overlay_accessibility_source.py tests/test_composer_sendability_source.py tests/test_send_button_source.py tests/test_mobile_toast_source.py tests/test_queue_button_source.py tests/test_frontend_display_module_source.py` (26 passed). New assertions lock capture wiring, single `/inject_file`, blocker sharing, and the picker/capture block boundaries. No regressions.
- **NONBLOCKER 1 — stale mobile CSS comment / reduced margin:** the composer gains a 5th `.icon-btn` (attach, **capture**, queue, stop, send). The comment at `app.css:2766-2772` still reasons about "the four 44px controls cannot force horizontal overflow at 390px." Under `@media (max-width:520px)` (opened `app.css:2668`) each control has `min-width:44px`. Horizontal overflow is still prevented because `.composer .inputWrap` is `flex:1 1 auto; min-width:0` (`app.css:1353-1356`), so the textarea absorbs the extra width: at 390px, 5×44 + gaps + padding ≈ 276px chrome → ~114px for the textarea; safe down to ~280px containers. The invariant holds, but the comment now under-counts and the textarea's free width shrank by ~50px. Recommend updating the comment to "five controls" and re-stating the width math.
- **Browser compat:** `capture="environment"` + `accept="image/*"` is modern spec; unsupported/desktop browsers ignore `capture` and fall back to an image file picker. No breakage. Attribute wiring confirmed live in proof.
- **NONBLOCKER 2 — desktop semantics (observation):** on a device without a camera, the camera button opens a single-image file picker (no `multiple`), partially overlapping the paperclip but image-only. This is acceptable graceful degradation for a client-only File producer; flag for a product decision only if distinct desktop behavior is undesired.

---

## Accepted-invariant scorecard
- Client-only File source → shared `stageFiles()` → existing `/inject_file`; no backend route/state/commit boundary: **holds** (Q2).
- Full attach blockers (no session, failed launch, unknown send, orphan recovery, orphan queue recovery, current response running, send in flight): **holds** at both gates (Q3).
- Badge/chips derive from server staged-list truth: **holds** (`setSelectedSessionStagedAttachments(res.attachments)` at `app.js:6905`; proof badge/chips mirror server `attachments`).
- Confirmed send clears staged; commit-unknown/failure preserves: **holds** — capture never touches the send path (unchanged); proof shows staged cleared only after confirmed send success.

## Required fixes
None blocking. Recommended (nonblocking):
1. Update `app.css:2766-2772` comment to reflect five composer controls and the current overflow math.
2. Optional product note on desktop camera-button behavior (degrades to image picker).
3. Optional: persist the verbatim browser driver JS in future capture proofs for independent reconstructability.

## Evidence checked
- `git show 98880bc` / `git show c7fd396` (full diffs); `git show --name-status 98880bc` (only app.js, app_display.js, test).
- `git merge-base --is-ancestor c7fd396 HEAD` → ancestor; `git diff --stat c7fd396 HEAD` → memory docs only.
- Source read in context: `app.js:1242-1256, 6469, 6536, 6689-6729, 6833-6944, 6948-6965, 7072`; `app_display.js:180,203-204,277`; `app_dom.js:4-19`; `app_file_helpers.js:387-411`; `app.css:1322-1356, 2668, 2754-2775, 2920-2930`.
- `grep`: `/inject_file` count = 1 (line 6899); no `window.stageFiles`/`.stageFiles =`; capture block contains no `send`/`keys`/route calls.
- `node --check codoxear/static/app.js` and `app_display.js` → OK.
- `python3 -m pytest -q tests/test_attach_button_source.py` → 8 passed; plus 6-file composer/frontend source suite → 26 passed.
- Proof artifacts: `VERIFICATION-REPORT.md`, `browser-ready.json`, `browser-after-capture.json`, `browser-after-send.json`, `docker-calls-after-capture-compact.json`, `docker-calls-after-send-compact.json`, `combined-evidence.json`, `fake_capture_session.py`, `api-me-before-login.{json,status}`, `docker-final-state.txt`.
- `git diff --cached --name-only` (empty) and `git diff --name-only` (empty) → nothing staged, no tracked-file edits by this review.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Review scoped exactly to commits 98880bc and c7fd396; capture implemented as client-only File producer via stageFiles(source:'capture') -> single existing /inject_file (app.js:6899, count=1). No backend .py changes (git name-status). No edits/staging/commits by the review (git diff --cached and git diff both empty)."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "All 7 review questions answered with file/line mechanism evidence and independently re-run validation (node --check x2, 8+26 tests passed) plus proof-artifact analysis distinguishing send_count=0/keys_count=0 pre-send from send_count=1/keys_count=0 at the send boundary."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {"command": "git show 98880bc / c7fd396 (+ --name-status, --stat, merge-base)", "result": "passed", "summary": "Only app.js, app_display.js, test changed in 98880bc; c7fd396 is proof-only; both ancestors of HEAD"},
    {"command": "node --check codoxear/static/app.js && node --check codoxear/static/app_display.js", "result": "passed", "summary": "both parse OK"},
    {"command": "python3 -m pytest -q tests/test_attach_button_source.py", "result": "passed", "summary": "8 passed"},
    {"command": "python3 -m pytest -q tests/test_overlay_accessibility_source.py tests/test_composer_sendability_source.py tests/test_send_button_source.py tests/test_mobile_toast_source.py tests/test_queue_button_source.py tests/test_frontend_display_module_source.py", "result": "passed", "summary": "26 passed"},
    {"command": "grep -c '/inject_file' + window.stageFiles scope check", "result": "passed", "summary": "exactly 1 /inject_file inside stageFiles; stageFiles is closure-only (not on window)"},
    {"command": "git diff --cached --name-only; git diff --name-only", "result": "passed", "summary": "both empty — nothing staged, no tracked-file edits"}
  ],
  "validationOutput": [
    "node --check app.js: OK; app_display.js: OK",
    "tests/test_attach_button_source.py: 8 passed",
    "6-file composer/frontend source suite: 26 passed",
    "/inject_file occurrences in app.js: 1 (line 6899, inside stageFiles)",
    "proof after-capture: send_count=0 keys_count=0; staged chip 'captured-1783381957888.jpg', badge 1, pending_attachment true",
    "proof after-send: send_count=1 keys_count=0, payload 'Attachment 1: <path>\\nplease process captured photo'; chips [], badge '', attachments []"
  ],
  "residualRisks": [
    "nonblocker: app.css:2766-2772 comment says 'four 44px controls' but composer now has five; horizontal overflow still prevented by inputWrap min-width:0, but textarea free width shrank ~50px at <=520px widths (safe to ~280px).",
    "nonblocker: on desktop/no-camera devices the camera button degrades to a single-image file picker (product-semantics observation).",
    "minor evidentiary gap: verbatim browser driver JS that dispatched the change event is not saved in artifacts; conclusion still holds via 'captured-' filename + closure scoping."
  ],
  "noStagedFiles": true,
  "diffSummary": "Review only; no diff produced. Under review: 98880bc adds captureBtn/captureInput + capturedFileName + dual-control state sync in app.js, a camera icon in app_display.js, and test assertions; c7fd396 adds browser-proof artifacts under .memory/. HEAD later advanced to 394f9c0 (memory docs only) by a concurrent session.",
  "reviewFindings": [
    "no blockers",
    "accepted: capture is a client-only File producer -> stageFiles(source:'capture') -> single existing /inject_file (app.js:6899); no backend route/state/PTY/key/send added; send remains the only commit boundary",
    "accepted: full attach blockers applied at captureBtn.onclick (app.js:6951) and inside stageFiles (app.js:6858) via shared attachmentBlockerForSession",
    "accepted: no-name captured images get captured-<seed>.<ext> (app.js:6840-6851); compression detection on raw File, rewrite to .jpg with preserved stem — coherent",
    "accepted: syncAttachButtonState keeps both controls truthful (app.js:6699-6717); proof confirms",
    "accepted: proof distinguishes zero pre-send backend commit (send_count=0,keys_count=0) from single send-boundary commit (send_count=1,keys_count=0) and staged-list clear on confirmed send",
    "nonblocker: app.css:2766-2772 stale comment (5th composer control) — overflow invariant still holds via inputWrap min-width:0",
    "nonblocker: desktop camera button degrades to image file picker (product-semantics observation)"
  ],
  "manualNotes": "Read-only clean-room review; made no edits/staging/commits. Working tree shows two untracked files (docker-gate-failure-note.txt, docker-test-19358.txt) and HEAD moved to 394f9c0 during review — both from a concurrent session, not this review. Both reviewed SHAs remain ancestors of HEAD and are unaffected by 394f9c0 (memory-only). Findings written to /tmp/capture-producer-cleanroom-review.md."
}
```
