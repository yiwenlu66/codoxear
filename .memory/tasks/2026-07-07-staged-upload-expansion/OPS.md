# Operational ledger

## 2026-07-06T18:50:00Z Task initialized
- Objective: first upload expansion slice with server-owned staged attachment list and multi-file picker.
- Success boundary: attachments are staged/removable before send; generated backend-readable path references are committed only at send boundary.

## 2026-07-06T18:54:00Z Mechanism decision before implementation
- Observed current control flow: `/inject_file` validates readiness, stages bytes, builds `Attachment N: <path>`, wraps it in bracketed paste, and calls `manager.inject_attachment_keys` immediately.
- Observed direct send flow: `SessionSendCoordinator.send()` already owns the confirmed-send boundary and clears `pending_attachment` on success.
- Mechanism decision: upload routes must become stage-only; send must compose attachment references into the confirmed send text and clear staged entries only on confirmed success.
- Rationale: separate pre-send PTY writes would recreate the wrong boundary and cannot truthfully support remove/clear before send.

## 2026-07-06T20:33:37Z Functional implementation committed
- Functional commit: `e1c8315 Stage attachments until send boundary`.
- Dirty executor work was reviewed before commit; two mechanism defects were corrected before committing:
  - same-display-name uploads in the same millisecond could collide/overwrite under the staged upload directory;
  - stage-only upload readiness still inherited the old immediate-key-write requirement, which incorrectly disabled staging for brokers that can confirmed-send but cannot report key write errors.
- Validation before functional commit:
  - `node --check codoxear/static/app.js`
  - focused pytest slice for upload/control/send/store/frontend source suites: `233 passed, 22 subtests passed`
  - `git diff --check`
  - full local `python3 -m pytest -q`: `1782 passed, 132 subtests passed`
- Source guard observation: `/inject_file` no longer contains `inject_attachment_keys`, `inject_keys`, or bracketed paste writes; attachment references are composed in the send path.

## 2026-07-06T20:33:37Z Docker/browser proof committed
- Proof commit: `b1e6bc2 Record staged upload browser proof`.
- Artifact root: `.memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/staged-upload-19331/`.
- Docker sandbox: named container `codoxear-upload-19331`, port `19331`, app dir `/home/tester/.local/share/codoxear`; stopped by exact container name after proof.
- Proof mechanism: fake broker sidecars advertised `sync_send:true` and `key_write_errors:false`, separating the new stage-only upload path from the old immediate PTY/key-paste path.
- Observations:
  - Browser multi-file picker staged `alpha.txt` and `beta.txt`; UI rendered two chips with server-derived paths and badge `2`.
  - Server attachment API returned two staged entries and `pending_attachment:true`.
  - Broker call summary after upload had only `state` calls; zero `send` and zero `keys`.
  - Removing one chip left one staged chip; clear-all left zero chips and server `pending_attachment:false`; broker call summary still had zero `send`/`keys`.
  - Re-staging two files and sending generated exactly one broker `send` with `Attachment 1: <path>`, `Attachment 2: <path>`, then user text; zero `keys`; confirmed success cleared staged entries.
  - A second fake session forced `commit_unknown`; route returned HTTP 504 with `commit_unknown:true`, send payload included `Attachment 1: <path>`, and the staged entry remained with `pending_attachment:true`.
- Raw bulky broker polling logs were reduced to compact command-count/send-payload summaries before commit; no cookie jar was committed.

## 2026-07-06T20:33:37Z Clean-room review dispatched
- Async critic id: `94e20e49-0ee1-4d60-86dd-d153c2ae8985`.
- Scope: adversarial product-level audit of `38e2120..b1e6bc2`, especially hidden pre-send backend writes, staged identity, commit boundary, failure preservation, cleanup, proof credibility, and artifact hygiene.
- Result later committed as `b2da8a8 Record staged upload cleanroom review`; verdict accepted with no blockers and six nonblocking observations.

## 2026-07-06T20:40:00Z Clean-room review accepted
- Review artifact committed: `b2da8a8 Record staged upload cleanroom review`.
- Verdict: accepted, no blockers.
- Independent checks reproduced stage-only/send-boundary invariants and focused validation (`240 passed, 22 subtests passed`).
- Nonblocking observations:
  - `attachments/clear` should harden `ValueError` handling for tampered/symlink app-dir states; current behavior fails loud as HTTP 500 and requires app-dir write access.
  - Browser sees absolute staged paths; current UI uses them as tooltip and this follows prior attachment behavior, but basename/path-redaction cleanup can reduce exposure later.
  - Dead immediate-PTY `inject_attachment_keys` coordinator remains bound but no HTTP route reaches it; remove in later cleanup if no internal user remains.
  - `attachment_index` is now vestigial after send-boundary numbering.
  - A narrow stage-during-active-send race is self-correcting and practically guarded by readiness.
  - Staging while the agent is busy remains intentionally blocked.
- Decision: first staged-upload slice is accepted. N1 is useful defensive hardening but not a blocker for acceptance.

## 2026-07-06T20:58:00Z Attachment clear guard hardening committed
- Functional follow-up commit: `75986c1 Harden staged attachment clear failures`.
- Mechanism: `attachments/clear` now maps staged cleanup guard `ValueError` to HTTP 400 JSON instead of uncaught 500; `SessionStore.clear_staged_attachments` preflights all staged file targets before unlinking so deterministic tamper/path guard failures leave both staged list and valid files intact.
- Validation:
  - `python3 -m pytest -q tests/test_control_routes.py tests/test_file_upload.py tests/test_session_store.py` → `72 passed in 1.80s`.
  - `git diff --check` → clean.
- Scope: no send/queue/frontend paths changed; staged-upload commit boundary remains unchanged.

## 2026-07-06T21:19:00Z Legacy clear guard fix and review accepted
- Review artifact committed: `731da2c Record attachment clear hardening review`.
- Blocker found: legacy `pending_attachment/clear` invoked the same staged cleanup guard but still mapped `ValueError` to uncaught HTTP 500.
- Functional fix committed: `b7148bb Map legacy attachment clear guard failures`.
- Validation for fix:
  - `python3 -m pytest -q tests/test_control_routes.py::test_attachment_clear_maps_tamper_guard_failure_to_400 tests/test_control_routes.py::test_pending_attachment_clear_maps_tamper_guard_failure_to_400 tests/test_control_routes.py tests/test_file_upload.py tests/test_session_store.py` → `73 passed in 1.86s`.
  - `git diff --check` → clean.
- Fix review committed: `1950474 Record attachment clear fix review`.
- Fix review verdict: accepted. End-to-end repro through real route/coordinator/store proved tamper → 400 with list/symlink/target preserved, unknown session → 404, happy path → 200. Residual send-path post-commit cleanup failure remains pre-existing/out of scope.

## 2026-07-06T21:19:00Z Paste/drop producer implementation and proof committed
- Functional commit: `27ca144 Add paste and drop attachment producers`.
- Mechanism: picker, paste, and drop now feed a shared client `stageFiles()` path that posts only to existing `/inject_file`; `attachmentBlockerForSession()` centralizes full attach blockers across paperclip and producers. Text-only paste returns before `preventDefault`; file paste/drop use real event listeners and server-staged list truth.
- Local validation before proof:
  - `node --check codoxear/static/app.js`
  - `node --check codoxear/static/app_file_helpers.js`
  - focused frontend/upload suite → `183 passed, 22 subtests passed`
  - full local suite → `1787 passed, 132 subtests passed`
  - `git diff --check` → clean.
- Browser proof commit: `35b13dc Record paste drop attachment browser proof`.
- Artifact root: `.memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/upload-producers-19343/`.
- Docker proof observations:
  - file-bearing paste on `#msg` prevented default, staged one file, and broker command summary had zero `send`/`keys`;
  - text-only paste was not prevented and did not stage;
  - composer drag/drop prevented default, toggled/cleared drop highlight, staged two files, and broker command summary had zero `send`/`keys`;
  - off-composer file drop prevented navigation without staging;
  - explicit send produced exactly one broker `send`, zero `keys`, three generated `Attachment N:` lines, and cleared staged entries.
- Clean-room review dispatched: async id `663ce592-4800-4b56-a861-19befe6f7e8d`.
- Result later committed as `245dcb2 Record upload producer cleanroom review`; verdict accepted with no blockers and four nonblocking nits.

## 2026-07-06T21:25:00Z Paste/drop producer review accepted
- Review artifact committed: `245dcb2 Record upload producer cleanroom review`.
- Verdict: accepted, no blockers.
- Review independently confirmed:
  - `27ca144` is client-only and has exactly one `/inject_file` occurrence inside shared `stageFiles()`; picker/paste/drop are the only `stageFiles()` callers.
  - full attach blockers apply inside `stageFiles()` before upload route calls.
  - text-only paste is not prevented; file paste/drop use the real browser listeners and real server staging route.
  - off-composer file drop prevents navigation without staging.
  - broker proof showed zero `send`/`keys` before explicit send and exactly one `send` at the send boundary.
- Nonblocking nits retained for later UX polish: blocker is checked once per batch, combined text+file paste drops the text, small non-PNG clipboard images may get extensionless pasted names, and `.drop-active` may stick if a file drag leaves the window without dropping.

## 2026-07-06T21:31:00Z Post-send cleanup failure policy committed
- Functional commit: `4963ba6 Report post-send attachment cleanup failures`.
- Mechanism: after confirmed send success and prelog/state application, a staged cleanup guard `ValueError` no longer propagates as send failure. `SessionSendCoordinator.send()` returns the normal send response augmented with `attachment_cleanup_error`; it clears `commit_unknown_send` because backend delivery was confirmed, and leaves staged/pending truth intact because cleanup failed.
- Frontend behavior: send success with `attachment_cleanup_error` shows `sent; attachment cleanup failed: ...` (or queued equivalent) and does not clear staged browser/server projection, so the delivered turn is not treated as retryable failure.
- Validation:
  - targeted cleanup policy tests → `6 passed in 1.75s`;
  - focused send/control/frontend suite → `125 passed, 22 subtests passed in 1.85s`;
  - full local suite → `1789 passed, 132 subtests passed in 24.49s`;
  - `node --check codoxear/static/app.js` and `git diff --check` → clean.
- Clean-room review dispatched: async id `2056e07e-a15e-4afd-89b0-b65cda060475`.

## 2026-07-06T21:46:00Z Post-send cleanup policy review accepted with residual
- Review artifact committed: `09a9aa9 Record post-send cleanup policy review`.
- Verdict: accepted for deterministic `ValueError` guard path; no blockers.
- Review confirmed: cleanup guard failure after confirmed delivery returns 200 with `attachment_cleanup_error`, preserves staged/pending truth, clears commit-unknown state, and frontend surfaces cleanup failure without implying resend.
- Nonblocking residual: post-confirmation `OSError` from unlink/persistence and rare post-confirmation `KeyError` can still become false send failures after delivery. Follow-up executor dispatched: `7898908e-b067-4656-8cc4-27b50d94e254`.

## 2026-07-06T22:01:00Z Post-confirmation tail isolation committed
- Functional commit: `785b3d2 Isolate post-send cleanup tail failures`.
- Mechanism: after confirmed send parse succeeds, post-confirmation tail failures in prelog projection, staged cleanup, pending projection clearing, and commit_unknown clearing are converted into explicit response warning fields (`attachment_cleanup_error` or `send_state_cleanup_error`) rather than route errors. Pre-delivery not-ready/injection/commit_unknown paths remain before this boundary.
- Validation:
  - targeted tail tests → `7 passed in 1.75s`;
  - focused send/control/frontend suite → `130 passed, 22 subtests passed in 1.87s`;
  - full local suite → `1794 passed, 132 subtests passed in 24.69s`;
  - `node --check codoxear/static/app.js` and `git diff --check` → clean.
- Clean-room review dispatched: async id `c48a075c-24aa-43e7-8ac3-7ebd53ec5671`.

## 2026-07-06T22:18:00Z Post-confirmation tail isolation review and fix
- Review artifact: `.memory/tasks/2026-07-07-staged-upload-expansion/reviews/post-send-tail-isolation-review.md`.
- Review verdict: accepted with two nonblocking findings.
- Finding NB1: post-confirmation prelog projection caught `OSError`/`KeyError` but not `ValueError`; a corrupted launch ledger could still produce a false post-delivery 500 and skip cleanup/state application. Although low-realism, this contradicted the tail-failure invariant.
- Finding NB2: prelog catch branch lacked direct regression coverage.
- Functional fix committed separately as `b0a6a09 Isolate prelog tail value errors`: shared post-confirmation tail error tuple now includes `ValueError` for prelog projection, and regression coverage proves delivered send + prelog `ValueError` returns `send_state_cleanup_error`, applies send-boundary busy state, and continues through staged cleanup.
- Validation after fix:
  - targeted regression/focused tails → `4 passed in 1.87s`;
  - focused send/control/frontend suite → `131 passed, 22 subtests passed in 2.00s`;
  - full local suite → `1795 passed, 132 subtests passed in 28.79s`;
  - `node --check codoxear/static/app.js` and `git diff --check` → clean.
- Follow-up review dispatched for `b0a6a09`: async id `42f4215a-c8a9-4db7-874c-4a6a5a5e873a`.

## 2026-07-06T22:35:00Z Post-confirmation tail isolation fix review accepted
- Review artifact: `.memory/tasks/2026-07-07-staged-upload-expansion/reviews/post-send-tail-isolation-fix-review.md`.
- Review verdict: accepted, no blockers.
- Mechanism confirmed: `b0a6a09` closes the prelog `ValueError` escape by using one post-confirmation tail error tuple for prelog projection, staged cleanup, pending clear, and commit_unknown clear. The catch sites remain after `parse_confirmed_send_response`, so pre-delivery not-ready/injection/commit_unknown behavior remains unchanged.
- Review reproduced validation: after-confirmed-send tail matrix `6 passed`; prelog regression `1 passed`; broader focused suite `155 passed, 22 subtests passed`; `node --check codoxear/static/app.js`; `git diff --check`; clean tree.

## 2026-07-06T23:56:00Z Capture producer implemented and browser-proved pending review
- Delegation note: executor `cd6271bb-3b47-41fb-8789-a0a2eb9c7a2f` failed before work due provider 429; executor `fdbfb7ba-52d3-459d-b0bf-fc4dd95eb6d0` completed without edits. The main agent implemented the small frontend seam directly after confirming no active subagent owned the task.
- Functional commit: `98880bc Add capture attachment producer`.
- Mechanism: added visible `#captureBtn`, hidden `#captureInput` with `accept="image/*"` and `capture="environment"`, and routed captured/selected files through existing `stageFiles(files, {source:"capture"})`. Capture shares `attachmentBlockerForSession()` before opening the input and inside `stageFiles()` before `/inject_file`; no backend route or new commit boundary was added.
- Local validation:
  - `node --check codoxear/static/app.js`;
  - `node --check codoxear/static/app_display.js`;
  - `python3 -m pytest -q tests/test_attach_button_source.py tests/test_frontend_display_module_source.py` → `13 passed`;
  - focused upload/control/frontend suite → `184 passed, 22 subtests passed`;
  - full local suite → `1796 passed, 132 subtests passed`;
  - `git diff --check` → clean.
- Browser proof commit: `c7fd396 Record capture producer browser proof`.
- Artifact root: `.memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/capture-producer-19357/`.
- Docker/browser observations: fake broker advertised `sync_send:true` and `key_write_errors:false`; browser DOM exposed enabled capture button and capture input; synthetic no-name JPEG was installed on `#captureInput.files` and the real `change` listener staged one server entry named `captured-...jpg`; broker calls before send had zero `send` and zero `keys`; visible send button produced exactly one `send`, zero `keys`, payload began with generated `Attachment 1: <path>`, and staged list cleared.
- Clean-room review dispatched: async id `768d89ba-ff5e-4087-be32-d60b75c654ae`.

## 2026-07-07T00:12:00Z Capture producer review accepted and nonblockers closed
- Review artifact: `.memory/tasks/2026-07-07-staged-upload-expansion/reviews/capture-producer-cleanroom-review.md`.
- Review verdict: accepted, no blockers.
- Review confirmed: capture is a client-only `File` producer feeding `stageFiles(source:"capture")` and the single existing `/inject_file` route; no backend route/state/PTY/key/send write was added; full attach blockers apply before opening the capture input and inside `stageFiles`; no-name captured images get `captured-<seed>.<ext>` fallback names; proof distinguishes zero pre-send backend `send`/`keys` from one send-boundary `send`.
- Nonblocker 1 closed in `ef42bb8 Clarify capture composer mobile comment`: stale mobile CSS comment now accounts for five composer icon controls.
- Nonblocker 3 closed in `0c61bb6 Record capture proof driver`: browser proof now includes `capture-input-change-driver.js`, the exact script used to set `#captureInput.files` and dispatch the real `change` listener.
- Retained nonblocking product observation: on desktop/no-camera devices the capture button degrades to a single-image picker, which is acceptable graceful degradation for a client-only File producer.
- Docker gates after proof: `93a7a45 Record capture producer docker gates` records `DOCKER_BUILDKIT=0` sandbox test (`1795 passed, 1 skipped, 132 subtests passed`) and smoke (`/api/me` 401 before login, `/api/sessions` 200 after login, app dir under `/home/tester/.local/share/codoxear`).

## 2026-07-07T00:20:00Z Upload attachment_index contract removed pending review
- Functional commit: `fa74c6a Remove upload attachment index contract`.
- Mechanism: `/inject_file` and `/inject_image` no longer require or validate `attachment_index`; the frontend no longer sends it; legacy clients that still send any `attachment_index` value are ignored and remain compatible. The route-layer `attachment_inject_text` dependency was removed; send-boundary numbering still uses `file_upload.attachment_inject_text` from `session_send.py`.
- Validation:
  - `node --check codoxear/static/app.js`;
  - focused route/upload/source suite → `68 passed in 1.87s`;
  - full local suite → `1795 passed, 132 subtests passed in 24.77s`;
  - `git diff --check` → clean.
- Clean-room review dispatched: async id `91409321-c108-47de-9175-7dc9f3e20f46`.


## 2026-07-07T00:34:00Z Upload attachment_index cleanup review accepted
- Review artifact: `.memory/tasks/2026-07-07-staged-upload-expansion/reviews/attachment-index-cleanup-review.md`.
- Review verdict: accepted, no blockers or required fixes.
- Review confirmed: frontend sends only `filename` and `data_b64`; `/inject_file` and `/inject_image` ignore any legacy `attachment_index`; all other validation/order and readiness-before-decode/stage remain intact; route-layer `attachment_inject_text` dependency is gone while `file_upload.attachment_inject_text` remains available to `session_send.py` for send-boundary numbering; no immediate backend key/PTY write or new commit boundary was introduced.
- Review evidence: focused checks `38 passed` and broadened upload/send checks `84 passed`; full suite reproduced `1795 passed, 132 subtests passed`; `node --check codoxear/static/app.js`; `git diff --check`.
- Nonblocker: older `.memory/tasks/2026-07-03-usable-product-ui-architecture/upload-attachment-scout.md` still describes historical `attachment_index` scouting; active task/project memory now supersede it.

## 2026-07-07T00:45:00Z Immediate attachment key injection removed pending review
- Functional commit: `18cd64c Remove immediate attachment key injection`.
- Mechanism: deleted `codoxear/session_attachment.py` and removed the `SessionAttachmentCoordinator` factory/binding, `SessionManager.inject_attachment_keys`, and `attachment_injection_ready`. The remaining active upload path is staged upload through `/inject_file`/`/inject_image`, `attachment_staging_ready`, staged-list state, and confirmed-send composition in `session_send.py` using `file_upload.attachment_inject_text`.
- Readiness effect: staging still requires an active known session, no commit-unknown send, sync-send capability, no queue-sending item, no local queue, and idle broker/log runtime. It no longer has any route to require `key_write_errors_supported`, matching accepted stage-only browser proofs with `key_write_errors:false`.
- Validation:
  - targeted retirement suite → `163 passed, 18 subtests passed`;
  - full local suite → `1788 passed, 128 subtests passed`;
  - `git diff --check` → clean.
- Clean-room review dispatched: async id `a8f88f05-999c-4922-b917-7d823b69c4b6`. A needs-attention nudge could not be delivered because the child was live but its intercom target was not registered; review remains pending.

## 2026-07-07T00:50:00Z Immediate key injection review relaunched after empty paused run
- Prior clean-room review async id `a8f88f05-999c-4922-b917-7d823b69c4b6` paused after interrupt and produced `/tmp/immediate-attachment-key-removal-review.md` as a 0-byte artifact; status reported acceptance rejected and resume unavailable.
- Replacement clean-room review dispatched from product checkout `/home/yiwen/codex-web-product-recovery`: async id `62eb6953-a691-425e-91d3-f14055ffa102`, output target `/tmp/immediate-attachment-key-removal-review.md`.

## 2026-07-07T00:58:00Z Immediate key injection review rerouted after second empty paused run
- Replacement review async id `62eb6953-a691-425e-91d3-f14055ffa102` also paused unrecoverably and produced `/tmp/immediate-attachment-key-removal-review.md` as a 0-byte artifact; status reported acceptance rejected and resume unavailable.
- Third clean-room review dispatched from product checkout `/home/yiwen/codex-web-product-recovery` using `zai/glm-5.2` with no injected acceptance wrapper: async id `f7c960a3-0848-4329-a573-f89fe57dc35b`, output target `/tmp/immediate-attachment-key-removal-review.md`.

## 2026-07-07T01:05:00Z Immediate key injection review rerouted after GLM quota failure
- Third review async id `f7c960a3-0848-4329-a573-f89fe57dc35b` failed before review with provider quota: `429 You have reached the 7-day usage limit... after 2026-07-10 16:11:14`.
- Fourth clean-room review dispatched from product checkout `/home/yiwen/codex-web-product-recovery` using `openai-codex/gpt-5.5` with no injected acceptance wrapper: async id `d203bdc3-9780-4c68-a46d-cea2520bafd3`, output target `/tmp/immediate-attachment-key-removal-review.md`.


## 2026-07-07T01:12:00Z Immediate attachment key removal review accepted
- Review artifact: `.memory/tasks/2026-07-07-staged-upload-expansion/reviews/immediate-attachment-key-removal-review.md`.
- Review verdict: accepted, no blockers or required fixes.
- Review confirmed: `SessionAttachmentCoordinator`, its factory/binding, `attachment_injection_ready`, and `SessionManager.inject_attachment_keys` were removed; `/inject_file` and `/inject_image` still stage bytes through `attachment_staging_ready`, `stage_uploaded_file`, and `add_staged_attachment`; frontend producers still post to the shared staged route; send-boundary numbering remains in `session_send.py` via `file_upload.attachment_inject_text`.
- Review confirmed staging readiness still blocks unknown session, commit-unknown send, missing sync-send capability, queue sending item, local queue, and broker/log busy runtime; dropping key-write-error support as a staging precondition matches accepted stage-only proof with `key_write_errors:false`.
- Residual risk characterized as deliberate: old non-HTTP/in-process callers of `SessionManager.inject_attachment_keys` would break, but no tracked caller remains, it was not a declared package API, and preserving it would keep the forbidden pre-send key-write mechanism.
- Review evidence reproduced: targeted suite `163 passed, 18 subtests passed`; full suite `1788 passed, 128 subtests passed`; `git diff --check`; removed-symbol grep over tracked `codoxear`/`tests` found only negative/source assertions; no staged files.

## 2026-07-07T01:35:00Z Staged attachment public path redaction implemented and proved pending review
- Functional commit: `f4f38dc Redact staged attachment paths from public UI`.
- Mechanism: server/public staged attachment projection now omits backend-readable `path` from `/inject_file`, `/inject_image`, `/attachments`, attachment delete/clear, and `/api/sessions` staged entries; frontend normalization no longer stores `item.path` or uses it for chip titles/fallback labels. Internal `SessionStore.staged_attachments[*].path` remains intact for cleanup and `SessionSendCoordinator` confirmed-send `Attachment N: <path>` composition.
- Local validation:
  - `node --check codoxear/static/app.js`;
  - focused redaction/upload suite → `200 passed, 18 subtests passed`;
  - full local suite → `1791 passed, 128 subtests passed`;
  - `git diff --check` → clean.
- Proof commit: `e7a02cb Record staged attachment path redaction proof`.
- Artifact root: `.memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/path-redaction-19371/`.
- Docker/browser observations: browser staged two files through the real `#imgInput` change listener; captured `/inject_file`, `/attachments`, and `/api/sessions` staged entries had no `path` key and did not contain `/home/tester/.local/share/codoxear/uploads`; chip titles/text contained no slash; after explicit send the fake broker recorded exactly one `send`, zero `keys`, and the send payload contained absolute internal `Attachment 1:`/`Attachment 2:` upload paths plus user text; confirmed send cleared browser chips and server attachments.
- Docker gates: separate sandbox on port `19372` passed unit gate (`1790 passed, 1 skipped, 128 subtests passed`) and smoke (`/api/me` 401 before login, `/api/sessions` 200 after login, app dir `/home/tester/.local/share/codoxear`).
- Clean-room review pending.

## 2026-07-07T01:38:00Z Staged attachment path review found commit-unknown preview boundary
- GLM critic `1e0e84e3-e648-4256-98bb-b032a9fa2427` failed before review because of provider quota/connection and produced a 0-byte artifact; it was ignored as evidence.
- Codex critic `977008c0-78e2-4fab-909b-6010d8c1a05d` produced a usable clean-room report at `/tmp/staged-attachment-path-redaction-review.md`; wrapper acceptance rejected it only because the report did not satisfy the wrapper's changed-files bookkeeping field.
- Review artifact committed as `39d3efa Record staged attachment path redaction review`.
- Verdict: accepted scoped staged-list/browser redaction with no blockers. Nonblocking boundary: `commit_unknown_send_text` projected already-composed `Attachment N: <absolute path>` text after an explicit send attempt whose receipt was unknown. This was outside pre-send staged-list projection but still browser/API-visible.

## 2026-07-07T01:55:00Z Commit-unknown preview path redaction implemented and proved
- Functional commit: `9074d4e Redact commit unknown attachment previews`.
- Mechanism: direct unknown-send records now store private committed `text` plus public `display_text`. Session listing prefers `display_text` for `commit_unknown_send_text`; legacy records without `display_text` redact only leading generated `Attachment N: /...` prefix lines. Private `text` remains full for audit/recovery and broker send semantics.
- Validation before commit:
  - focused regression set → `4 passed`;
  - broader focused suite → `160 passed, 18 subtests passed`;
  - full local suite → `1793 passed, 128 subtests passed`;
  - `node --check codoxear/static/app.js`, `node --check codoxear/static/app_display.js`, and `git diff --check` → clean.
- Proof commit: `447750f Record commit unknown preview redaction proof`.
- Artifact root: `.memory/tasks/2026-07-07-staged-upload-expansion/browser-artifacts/commit-unknown-redaction-19373/`.
- Docker/browser observations: fake broker returned `commit_unknown:true`; browser staged one file through real `#imgInput`, sent via visible `#sendBtn`, and `/api/sessions` projected `commit_unknown_send_text` equal to the user prompt with no `Attachment 1:` line and no upload root in row/body/public payload. `/attachments` preserved the staged entry without `path`. Fake broker recorded exactly one `send`, zero `keys`, and absolute `Attachment 1: /home/tester/.local/share/codoxear/uploads/...` in the send payload. Container `commit_unknown_sends.json` kept private `text` with the upload root and `display_text` without it.
- Docker gates: first full gate on port `19374` had one transient packaging test failure; isolated rerun of that test on port `19375` passed. Fresh full Docker gate on port `19376` passed (`1792 passed, 1 skipped, 128 subtests passed`) and smoke passed (`/api/me` 401 before login, `/api/sessions` 200 after login, app dir `/home/tester/.local/share/codoxear`).

## 2026-07-07T02:05:00Z Final staged attachment path redaction review accepted
- Final critic `41be91fd-25e9-413c-a223-181bc1a212d1` produced a usable report at `/tmp/staged-attachment-path-redaction-final-review.md`; wrapper acceptance rejected it only because of changed-files bookkeeping.
- Review artifact committed as `7d76aca Record final staged attachment path redaction review`.
- Verdict: accepted current staged attachment path redaction with no blockers.
- Review confirmed: public staged attachment payloads omit upload paths; confirmed send and private commit-unknown state preserve backend-readable paths; commit-unknown public recovery preview uses `display_text` or legacy generated-prefix redaction; immediate key injection remains retired; proof/test coverage is adequate.
- Nonblocking boundaries: legacy commit-unknown fallback redacts only generated leading `Attachment N: /...` lines, not arbitrary user-authored absolute paths; public `display_name` remains the client-provided filename while generated internal paths are omitted.
