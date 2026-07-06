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
- Follow-up commit `75986c1` hardens the review's only cleanup-failure concern: deterministic `attachments/clear` guard failures now return visible 400 JSON and preflight all staged paths before unlink, avoiding a false clear success or partial deterministic cleanup. See OPS.md 2026-07-06T20:58:00Z.

## Ruled out
- Immediate PTY paste as a hidden compatibility layer: source inspection and Docker broker call summaries showed no `send`/`keys` before explicit send.
- Filename-only identity as sufficient: same-name collision risk was fixed with unique staged destination naming and list entries carry stable ids/paths.
- Key-write-error support as a staging precondition: proof used a broker without key-write-error capability and still staged/removed/cleared truthfully.
- Clearing on uncertain backend receipt: forced commit-unknown preserved the staged entry.

## Remaining nonblocking follow-ups
- Consider reducing absolute-path exposure in browser tooltips once a safe backend-readable identity/display split exists.
- Remove dead immediate-PTY attachment injection methods if no non-HTTP consumer remains.
- Remove vestigial `attachment_index` after compatibility impact is checked.
- Later upload producers remain: drag/drop, paste-to-attach, and capture/camera.

## Current justified claim
The first upload expansion slice is accepted: multi-file picker attachments are server-staged before send, visibly manageable in the browser, and committed to the backend only at confirmed send.
