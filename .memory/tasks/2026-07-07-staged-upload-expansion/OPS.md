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
