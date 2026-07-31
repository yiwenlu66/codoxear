# Epistemic model

## Phenomenon
Upload expansion moved attachment truth from the backend PTY stream into server-owned pre-send state. The core product requirement was truthful staging: the browser can show, remove, and clear attachments only if the backend has not already received their path references.

## Accepted mechanism
Stage first, commit later.
- `SessionStore` owns staged attachment lifecycle as per-session entries with stable ids, display names, backend-readable staged paths, byte sizes, and creation timestamps.
- Upload routes stage bytes under app-dir `uploads/<session>/` with unique destination paths and update the staged list; they accept `filename` + `data_b64`, ignore any legacy `attachment_index`, and do not call `inject_keys` or bracketed paste writes. Commit `18cd64c` removes the old `inject_attachment_keys` API entirely; clean-room review `d203bdc3-9780-4c68-a46d-cea2520bafd3` accepted the removal.
- The selected-session attachment API exposes a public server list plus compatibility `pending_attachment` projection derived from list non-emptiness; public entries omit internal backend-readable `path`.
- Browser composer renders chips from server state, displays identity beyond filename through public name/size/id detail without exposing backend upload paths, supports multi-file picker selection, per-entry remove, and clear-all.
- Send is the commit boundary: when staged attachments exist and the user explicitly sends, the server prepends generated `Attachment N: <path>` lines to the confirmed-send text. Confirmed send success clears staged entries; commit-unknown or send failure preserves them.
- Stage-only attach readiness requires confirmed-send capability but not the old key-write-error capability, because pre-send staging no longer writes keys.

## Evidence supporting acceptance
- Functional commit `e1c8315` implements the mechanism and includes tests for staging state, uniqueness/collision avoidance, readiness split, send-boundary composition, success clearing, commit-unknown preservation, session-list projection, and frontend source behavior.
- Local validation before proof: JS syntax check, focused pytest (`233 passed, 22 subtests passed`), `git diff --check`, and full pytest (`1782 passed, 132 subtests passed`). See OPS.md 2026-07-06T20:33:37Z.
- Docker/browser proof commit `b1e6bc2` exercised the user-visible path against a fake broker with `sync_send:true` and `key_write_errors:false`, which would expose any dependence on old immediate key writes. Multi-file staging, remove-one, clear-all, confirmed-send clearing, and commit-unknown preservation matched the target mechanism. See OPS.md 2026-07-06T20:33:37Z and `browser-artifacts/staged-upload-19331/VERIFICATION-REPORT.md`.
- Clean-room review commit `b2da8a8` independently accepted the slice with no blockers and reproduced focused validation (`240 passed, 22 subtests passed`). See OPS.md 2026-07-06T20:40:00Z and `reviews/cleanroom-review.md`.
- Follow-up commit `75986c1` hardens deterministic `attachments/clear` guard failures; commit `b7148bb` applies the same visible 400 mapping to the legacy `pending_attachment/clear` route. Fix review `1950474` accepted the route/coordinator/store behavior. See OPS.md 2026-07-06T20:58:00Z and 2026-07-06T21:19:00Z.

## Ruled out
- Immediate PTY paste as a hidden compatibility layer: source inspection and Docker broker call summaries showed no `send`/`keys` before explicit send.
- Filename-only identity as sufficient: same-name collision risk was fixed with unique staged destination naming and list entries carry stable ids/paths.
- Key-write-error support as a staging precondition: proof used a broker without key-write-error capability and still staged/removed/cleared truthfully.
- Clearing on uncertain backend receipt: forced commit-unknown preserved the staged entry.

## Accepted legacy path retirement
- Commit `18cd64c` deleted the old immediate-key attachment path (`SessionAttachmentCoordinator`, `attachment_injection_ready`, and `SessionManager.inject_attachment_keys`). Clean-room review `d203bdc3-9780-4c68-a46d-cea2520bafd3` accepted the scope: staged upload routes, staged-list state, and send-boundary `Attachment N:` composition remain intact; staging readiness keeps active-session/busy/queue/commit-unknown/sync-send blockers and no longer depends on key-write-error support. See OPS.md 2026-07-07T00:45:00Z through 2026-07-07T01:12:00Z.

## Accepted route contract cleanup
- Commit `fa74c6a` retired vestigial pre-send `attachment_index` from the upload request contract. Clean-room review `91409321-c108-47de-9175-7dc9f3e20f46` accepted the change: the frontend no longer sends the field; the route ignores malformed legacy values rather than rejecting them; `/inject_image` shares the same relaxed handler; route-layer `attachment_inject_text` dependency was removed; send-boundary numbering remains in `session_send.py`. See OPS.md 2026-07-07T00:20:00Z and 2026-07-07T00:34:00Z.

## Accepted producer extension
- Paste-to-attach, drag/drop, and capture/camera are implemented as client-only producers feeding the existing staged upload route through shared `stageFiles()`. Docker/browser proofs `35b13dc` and `c7fd396` show producer events preserve zero pre-send backend writes and send-boundary attachment commits. Clean-room reviews `245dcb2` and `fb0334a` accepted these producer slices with no blockers.
- Producer polish commits `b4b018b`/`54a66d1` close the main UX nits without changing the commit boundary: mixed text+file paste preserves text while staging files, no-name pasted images get MIME-derived extensions, and file-drag highlight clears on window leave/off-composer drop. Browser/Docker proof shows the polished producers still use staged entries with no public path, one send-boundary backend `send`, and zero `keys`; review `447a124` accepted with no blockers.
- Batch-blocker commits `ec37237`/`1a986a9` close the long-upload race at the producer boundary: `stageFiles()` re-confirms the target session and re-reads the latest attachment blocker before each file, treats public session-row `busy` as blocking, stops remaining files before compression/`arrayBuffer()`/`/inject_file` when a blocker appears, and leaves already-staged files visible. Browser/Docker proof inserted a public commit-unknown blocker between two picker files and observed only the first `/inject_file`, one visible staged entry, no public `path`, and zero upload-producer `keys`; review `c94bca0` accepted with no blockers.
- Photo-affordance commits `537fd18`/`107af43` close the remaining capture/camera wording gap: the visible producer now says `Add photo`, no-name image fallback uses `photo-*`, and progress says `staging photo...`, while the hidden input still carries `accept="image/*" capture="environment"` for mobile camera support and routes through `stageFiles(..., { source: "capture" })`. Browser/Docker proof exercised the real `#captureInput` listener, observed truthful Add-photo title/aria and `photo-*.jpg` staged identity, zero pre-send broker writes, and one explicit send-boundary commit; review `a6498bb` accepted with no blockers.

## Accepted post-send cleanup policy
- Commit `4963ba6` makes deterministic post-confirmed-send staged cleanup guard failures explicit cleanup warnings (`attachment_cleanup_error`) instead of send failures. Delivery remains success, commit-unknown state is cleared, and staged/pending truth is preserved if cleanup fails. Clean-room review `09a9aa9` accepted this path with no blockers.
- Commits `785b3d2` and `b0a6a09` fully isolate covered post-confirmation tail failures after backend delivery: prelog projection, staged cleanup, pending projection clearing, and commit_unknown clearing convert `ValueError`/`OSError`/`KeyError` into explicit warning fields rather than route errors. `attachment_cleanup_error` preserves staged/pending UI truth when staged cleanup failed; `send_state_cleanup_error` reports bookkeeping/projection failures without implying resend. Clean-room reviews `c48a075c-24aa-43e7-8ac3-7ebd53ec5671` and `42f4215a-c8a9-4db7-874c-4a6a5a5e873a` accepted the mechanism with no remaining blockers.

## Accepted public path redaction
- Commits `f4f38dc` and `e7a02cb` split private staged attachment state from public browser/API projection: internal entries keep backend-readable `path`, while `/inject_file`, `/inject_image`, `/attachments`, attachment delete/clear responses, `/api/sessions` rows, and browser chips omit `path` and avoid absolute upload roots. Docker/browser proof staged files through the real picker listener, observed no public `path`/upload-root exposure, then confirmed the fake broker still received absolute `Attachment N: <path>` lines at send.
- Review `39d3efa` accepted the staged-list redaction and found one residual: commit-unknown recovery previews exposed already-composed `Attachment N: <absolute path>` text after an explicit unknown send. Commit `9074d4e` closes that public preview leak by storing private committed `text` plus public `display_text`, and by redacting legacy leading generated attachment lines when `display_text` is absent. Proof `447750f` shows `/api/sessions`/browser recovery preview contain only the user prompt while private `commit_unknown_sends.json` and broker send payload retain the absolute attachment path. Final review `7d76aca` accepted the full current invariant with no blockers.

## Remaining nonblocking follow-ups
- None known for the staged-upload workbench. The next product target should move off upload unless new evidence exposes a concrete upload defect.

## Current justified claim
The staged upload expansion slice is accepted through multi-file picker, paste/drop, truthful Add-photo/mobile-capture affordance, producer polish, batch-level blocker rechecks, post-confirmed-send tail isolation, and public path redaction: attachments are server-staged before send, visibly manageable in the browser without exposing backend upload paths, committed to the backend only at confirmed send, and never turned into retry-inviting send failures by covered post-delivery cleanup/projection errors.
