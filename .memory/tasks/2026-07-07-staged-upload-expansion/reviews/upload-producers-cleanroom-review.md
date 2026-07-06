# Clean-room adversarial review — paste/drop staged-upload producers

Repo: `/home/yiwen/codex-web-product-recovery` @ branch `recovery/product-gaps`
Commits reviewed: `27ca144` (producer impl) and `35b13dc` (browser proof), against accepted context `e1c8315`/`b1e6bc2`/`b2da8a8`/`9fefa4f`/`75986c1`/`b7148bb`.
Mode: review-only. No edits, no staging, no commits. Working tree left clean.

## Verdict: ACCEPTED (no blockers; 4 nonblocker nits)

The producer slice is client-only, all three producers (picker/paste/drop) funnel
through one `stageFiles()` path that enforces the full attach-blocker set before the
single `/inject_file` staging call, text paste is not hijacked, off-composer file drop
is a navigation fail-safe that does not stage, no producer writes to the backend PTY/
control socket before explicit send, and the browser proof genuinely exercises the real
listeners + real network route + real broker command channel. Artifact hygiene is clean.

---

## Deliverable-by-deliverable verification

### 1. `27ca144` is client-only, shared `stageFiles()` path — CONFIRMED
`git show --stat 27ca144` touches only `codoxear/static/{app.js,app_file_helpers.js,app.css}`
plus three test files. No `.py` route/manager/server change. Mechanism:
- Exactly one `/inject_file` occurrence in `app.js` (`grep -c` = 1), at `app.js:6861`, inside
  `stageFiles` (`app.js:6815`).
- `stageFiles` has exactly three callers (`grep "stageFiles("`): picker `imgInput` change
  (`app.js:6907`), `paste` (`app.js:6914`), composer `drop` (`app.js:6945`). Picker, paste,
  drop are producers feeding the same staging route.
- Server `_handle_inject_attachment` (`control_routes.py:294`) is unchanged by this commit and
  only stages to disk (`deps.stage_uploaded_file`) + server-owned list (`manager.add_staged_attachment`),
  gated by `manager.attachment_staging_ready` (409 when busy). No new backend route/state/commit boundary.

### 2. Full attach-blocker set applies to all producers — CONFIRMED
`attachmentBlockerForSession(sessionId, sessionInfo)` (`app.js:6678`) returns a non-empty
reason string for every blocker, else `""`:
- no session → `!sessionId` (6679)
- failed launch → `sessionLaunchFailed(info)` (6681)
- unknown send → `sessionHasUnknownSend(info)` (6682)
- orphan recovery → `sessionIsOrphanRecovery(info)` (6683)
- orphan queue recovery → `sessionHasOrphanQueueRecovery(info)` (6684)
- current response running → `currentRunning` (6685)
- sending in flight → `sending` (6686)

`stageFiles` calls this blocker BEFORE the upload loop (`app.js:6819-6824`) and returns early
(with a toast, only when the captured session is still selected) — so the blocker fires before
any `/inject_file` call for **all three producers**, not just the paperclip. `syncAttachButtonState`
(6689) and `attachBtn.onclick` (6891) use the same helper, so button-disable and producer-gating
share one source of truth. Server readiness remains final authority via `attachment_staging_ready`
(409, `control_routes.py:315-327`). The proof's fake broker returns readiness through the real
server, and pre-send producer actions produced zero `send`/`keys`.

### 3. File paste stages, text paste not hijacked — CONFIRMED
`textarea` paste listener (`app.js:6910`): `extractFilesFromClipboardData(e.clipboardData)`;
if `!files.length` it returns WITHOUT `preventDefault` (text path untouched); only when files are
present does it `preventDefault()` + `stageFiles(..., "paste")`. Proof: `browser-after-paste.json`
`defaultPrevented:true`, `value:""`; `browser-after-text-paste.json` `defaultPrevented:false`,
`value:"TEXT-ONLY-PASTE"`, chip count unchanged.

### 4. Composer drop stages + highlight; off-composer drop is nav fail-safe only — CONFIRMED
Composer `dragenter/dragover/dragleave/drop` (`app.js:6921-6946`), each gated by
`dataTransferHasFiles`. `drop` (6938) `preventDefault` → clears highlight → `extractFilesFromDropData`
→ `stageFiles(..., "drop")`. Depth counter (`composerDragDepth`) balances enter/leave across child
elements; `setComposerDropActive` toggles `.drop-active` (CSS `app.css:1340` scoped to
`.composer.drop-active form`). Window `dragover`/`drop` (6947-6956) `preventDefault` for file drags
only and DO NOT call `stageFiles` (test asserts `stageFiles(` absent in the window-drop block).
Proof: `browser-after-drop.json` `activeAfterDragover:true`, `activeAfterDrop:false`, 3 chips;
`browser-after-offzone-drop.json` `defaultPrevented:true`, href unchanged, chip count stays 3.

### 5. No PTY/control-socket write before explicit send — CONFIRMED
Client staging path calls only `/inject_file` (staging), never `/send` or `/keys`. Server
`/inject_file` writes to disk + in-memory staged list, no PTY. The only broker writer is
`_handle_send` (`control_routes.py:208`). Proof broker call summaries: `docker-calls-after-paste-summary.json`
= 171 `state`, 0 `send`, 0 `keys`; same for text-paste/drop/offzone; `docker-calls-after-send-summary.json`
= 1 `send` (carrying three server-prepended `Attachment N:` lines + user text, `sync:true`), 0 `keys`.
Send remains the sole commit boundary; server clears the staged list afterward
(`api-after-send.json` `attachments:[]`, `pending_attachment:false`).

### 6. Browser proof honesty — CONFIRMED
Fake broker (`fake_upload_session.py`) advertises `control_capabilities:{sync_send:true,key_write_errors:false}`
(matches claim) and records every command to a JSONL. Staging goes to the REAL codoxear server in
Docker; the broker only ever sees `state` until the explicit send. Events are dispatched to the real
page via `element.dispatchEvent` — the `dispatchReturned`/`defaultPrevented` fields are the actual
return/flag of that dispatch, proving the real listeners ran. The real network route is proven by
server-generated staged records: `api-after-paste.json` returns a real `id`, timestamped `filename`,
absolute `path`, and `size:11` — values only producible by the real `/inject_file` handler consuming a
real File's bytes. `api_count` transitions 0→1 (paste) →3 (drop) →3 (offzone) →0 (send) come from the
real attachment API. All five scenarios (paste stage, text paste, drop stage, off-zone drop, send
boundary) are covered.

---

## Audit questions

- **Any path where paste/drop bypass disabled attach state or call `/inject_file` despite a blocker?**
  No. One `/inject_file` call site, inside `stageFiles`, downstream of `attachmentBlockerForSession`.
  All three producers are the only `stageFiles` callers. The button's `disabled` state is cosmetic;
  the authoritative gate is the blocker call inside `stageFiles`, so paste/drop are gated even though
  they never consult the button.

- **Does the proof exercise real listeners + network, not just helpers?**
  Yes. `dispatchEvent`-driven events with observed `defaultPrevented`, real server-generated staged
  paths/ids/sizes, and real broker command counts. A helper-only test could not produce a real
  server upload path with a correct byte count.

- **Does the window drop fail-safe accidentally stage or suppress unrelated non-file drops?**
  No. Every window/composer handler early-returns when `dataTransferHasFiles` is false, so text
  drags and in-page image-element drags are never `preventDefault`ed and never staged. Independently
  verified in node: text drag → false, in-page image drag (`text/html`+`text/uri-list`) → false,
  OS-file dragover (`types` includes `"Files"`) → true. The window `drop` handler never calls `stageFiles`.

- **Does text paste remain normal?**
  Yes. No `preventDefault` when the clipboard carries no file items (`extractFilesFromClipboardData`
  returns `[]` for text-only). Confirmed by node run and `browser-after-text-paste.json`.

- **Are the helper extractors correct for real ClipboardEvent/DataTransfer shapes and not over-broad?**
  Yes. `arrayFromDataTransferItems` (`app_file_helpers.js:332`) filters `kind==="file"` and calls
  `getAsFile()`; `extractFilesFromClipboardData` (373) prefers items then falls back to `files`;
  `extractFilesFromDropData` (380) prefers `files` (reliable on drop) then items;
  `dataTransferHasFiles` (357) checks `files`→item kind→`types` includes `"Files"` (handles both
  DOMStringList `.contains` and array). All defensively reject string/undefined inputs. Independent
  node run of 19 realistic shapes all matched expectations.

- **Hidden regression to picker multi-file / auth / badge / send boundary?**
  None found. `imgInput` keeps `multiple:"multiple"` (`app.js:1247`) and `stageFiles` loops over all
  files. 401 handling preserved: `stageFiles` catch does `handleAppAuthLoss(); return false;` before
  `failures.push`, and the `refreshSessions` catch re-checks 401 (test_auth_cleanup updated to match).
  Badge stays a projection of the server-owned staged list via `setSelectedSessionStagedAttachments`
  / `projectSelectedAttachmentIndicator`. Send path (`sendText`, `app.js:~6989`) is unchanged;
  staged commit still happens server-side at `/send` with `allow_pending_attachment`.

- **Artifact hygiene?**
  Clean. Proof dir = 128K. No cookie (`codoxear_auth`), password, or secret *contents*: the only
  matches for secret-ish tokens are (a) `docker-final-state.txt` showing `hmac_secret`/`webpush_vapid_private.pem`
  as `ls -l` entries (names/perms only, no contents), (b) `"token": null` telemetry field, (c) fake
  broker's `'token':None` state field. `api-login.json` = `{"ok":true}` only.

---

## Findings (nonblocker nits — no fix required)

1. **Blocker is evaluated once per batch, not per file.** `stageFiles` checks the blocker before the
   loop; if `currentRunning`/`sending` flips true mid multi-file upload, remaining `/inject_file`
   calls in that batch still fire (loop only guards `selected !== sessionId`, `app.js:6832`). This is
   identical to the pre-refactor picker behavior and the server's `attachment_staging_ready` backstops
   each call (409). Mechanism: UX pre-filter, not a security gate; server is final authority. No change needed.

2. **Combined text+file paste drops the text.** A paste carrying both a File and text hits the
   file branch, `preventDefault`s, and stages the file — the text is not inserted (`app.js:6910-6915`).
   Rare; acceptable given the composer can't hold a file inline. Nit only.

3. **Small non-PNG clipboard image can get an extension-less staged name.** `pastedFileName`
   (`app.js:6808`) only appends `.png` for `image/png`; other image types that are small enough to skip
   compression keep a bare `pasted-<seed>` name. Clipboard images are near-universally PNG, and any
   image exceeding `ATTACH_UPLOAD_MAX_BYTES`/HEIC is renamed to `.jpg` during compression, so this is a
   cosmetic edge. Nit only.

4. **`.drop-active` highlight can stick if a file drag leaves the window without dropping.** No
   `dragend`/window-exit reset exists; it self-heals on the next drag/drop (composer + window drop both
   reset `composerDragDepth=0` and clear the class). Purely cosmetic. Nit only.

Transparency note on the proof: for the text-paste case a synthetic (untrusted) event does not
trigger the browser's native text insertion, so `value:"TEXT-ONLY-PASTE"` is harness-simulated; the
load-bearing claim (listener does not `preventDefault` and does not stage) is still genuinely
exercised via `defaultPrevented:false` and unchanged chip/api counts. The event-dispatch driver
script itself is not committed (only its outputs), a minor reproducibility gap; the committed outputs
are internally consistent and hard to fabricate.

---

## Evidence checked (independently re-run)

- `node --check codoxear/static/app.js` → OK
- `node --check codoxear/static/app_file_helpers.js` → OK
- `git diff --check` → clean; `git status --porcelain` → empty (review made no edits)
- `pytest tests/test_attach_button_source.py tests/test_frontend_file_helpers_source.py tests/test_auth_cleanup_source.py` → 17 passed
- `pytest -k "attach or upload or file_helper or staged or clipboard or drop or paste or compos or inject or send"` → 231 passed, 15 subtests passed, 0 failed
- Independent node execution of the six extraction helpers against 19 realistic ClipboardData/DataTransfer/DOMStringList shapes → all expected
- Read: `27ca144`/`35b13dc` full diffs; `app.js` (blocker/stageFiles/producers), `app_file_helpers.js`
  (extractors), `app.css` (drop-active), `control_routes.py` (`_handle_inject_attachment`, `_handle_send`)
- Proof artifacts: `VERIFICATION-REPORT.md`, `combined-evidence.json`, `fake_upload_session.py`, all
  `browser-after-*.json`, `api-after-*.json`, `docker-calls-*-summary.json`, `docker-final-state.txt`

## Required fixes
None. The slice is acceptable as committed.
