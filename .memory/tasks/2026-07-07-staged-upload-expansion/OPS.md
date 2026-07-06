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
