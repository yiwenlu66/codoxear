# Epistemic model

## Phenomenon
Upload expansion moved attachment truth from the backend PTY stream into server-owned pre-send state. The core product requirement was truthful staging: the browser can show, remove, and clear attachments only if the backend has not already received their path references.

## Accepted mechanism
Stage first, commit later.
- `SessionStore` owns staged attachment lifecycle as per-session entries with stable ids, display names, backend-readable staged paths, byte sizes, and creation timestamps.
- Upload routes stage bytes under app-dir `uploads/<session>/` with unique destination paths and update the staged list; they do not call `inject_attachment_keys`, `inject_keys`, or bracketed paste writes.
- The selected-session attachment API exposes the server list plus compatibility `pending_attachment` projection derived from list non-emptiness.
- Browser composer renders chips from server state, displays identity beyond filename through server path/id-derived detail, supports multi-file picker selection, per-entry remove, and clear-all.
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

## Accepted producer extension
- Paste-to-attach and drag/drop are implemented in `27ca144` as client-only producers feeding the existing staged upload route through shared `stageFiles()`. Docker/browser proof `35b13dc` shows paste/drop/off-zone-drop/text-paste semantics preserve zero pre-send backend writes and send-boundary attachment commits. Clean-room review `245dcb2` accepted the slice with no blockers.

## Accepted post-send cleanup policy
- Commit `4963ba6` makes deterministic post-confirmed-send staged cleanup guard failures explicit cleanup warnings (`attachment_cleanup_error`) instead of send failures. Delivery remains success, commit-unknown state is cleared, and staged/pending truth is preserved if cleanup fails. Clean-room review `09a9aa9` accepted this path with no blockers.
- Commit `785b3d2` extends post-confirmation tail isolation to filesystem/persistence failures (`OSError`) and rare post-confirmation `KeyError`, returning explicit warning fields instead of route errors after confirmed delivery. This remains pending clean-room review `c48a075c-24aa-43e7-8ac3-7ebd53ec5671`.

## Remaining nonblocking follow-ups
- Consider reducing absolute-path exposure in browser tooltips once a safe backend-readable identity/display split exists.
- Remove dead immediate-PTY attachment injection methods if no non-HTTP consumer remains.
- Remove vestigial `attachment_index` after compatibility impact is checked.
- Later upload producer remaining: capture/camera.
- Producer UX polish: re-check blockers per file in long batches if needed, preserve text in mixed text+file paste if a clear UI rule exists, add extensions for non-PNG pasted image names, and clear `.drop-active` when a file drag leaves the window without dropping.

## Current justified claim
The first upload expansion slice is accepted: multi-file picker attachments are server-staged before send, visibly manageable in the browser, and committed to the backend only at confirmed send.
