# Epistemic model

## Phenomenon
Codoxear is a mobile companion UI. Several always-used shell controls had remained below the 44x44 touch-target floor on phone-sized viewports even though composer and file-viewer controls had already been corrected.

## Accepted mechanism
- The affected controls were a CSS cascade gap, not a runtime or backend issue.
- Base shell/backend selectors kept selected-session utility rails, chat navigation rails, topbar actions, sidebar header actions, and New Session backend tabs at compact 34px sizing.
- The fix is mobile-scoped: inside `@media (max-width: 520px)`, the requested shell/backend selectors set `width`, `height`, `min-width`, and `min-height` to `44px`.
- Desktop/base compact sizing remains intact; generic mobile `.icon-btn` is not globally raised.

## Evidence
- Functional implementation and validation are recorded in OPS `2026-07-07T06:35:00Z`: CSS/source-test patch committed, focused source/static suite passed, full local suite passed, and `git diff --check` was clean.
- Docker/browser proof is recorded in OPS `2026-07-07T06:45:00Z`: the real served UI at `390x844` and `320x844` measured the target shell controls and backend tabs at `44x44`, with no body horizontal overflow.
- Clean-room review is recorded in OPS `2026-07-07T06:48:00Z`: no blockers; reviewer independently confirmed cascade scope, proof credibility, unchanged semantic files, and no send/key broker calls.

## Boundaries
- This slice changes mobile geometry only. It does not change send, queue, attachment, busy/idle, backend launch, transcript parsing, Monaco, upload, or event handling.
- The proof driver’s `selectedSession` field is non-informative because it read a nonexistent `dataset.sid`; the geometry proof remains valid because the URL/API selected the synthetic session and all visible selected-session controls were measured.

## Current justified claim
Mobile shell touch-target floor is accepted for the named shell command surfaces and New Session backend tabs. There are no known blockers in this slice.
