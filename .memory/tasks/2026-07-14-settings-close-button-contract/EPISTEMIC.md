# Epistemic model

## Phenomenon

Settings dialog close button renders as a narrow vertical pill with an accent
focus ring in clay dark; New session's close button renders as a proper square.

## Mechanism (fully traced)

Two independent layers compose the visual:

1. **Geometry (the defect).** `app.css` `.formViewer .icon-btn` sets
   `width: auto; min-width: 0` on every icon button inside dialog surfaces.
   That content-sizing is correct for text-bearing dialog controls
   (`.choiceChip`) but degenerate for icon-only buttons: the button collapses
   to its 18px svg glyph while keeping 32px height → the vertical pill.
   Commit 7a058d9b ("Redesign new session dialog layout") introduced the rule
   and immediately patched the symptom for New session only via an ID-scoped
   rescue (`#newSessionCloseBtn`, `#newSessionViewer .agentBackendTab` — both
   icon-only). The rescue was never applied to the other dialogs sharing the
   `.formViewer` class: Settings (`#settingsCloseBtn`) and Edit conversation
   (`#editCloseBtn`) still render the degenerate pill.

2. **Focus ring (platform behavior, present in both dialogs).** Both dialogs
   auto-focus their close button on open (`focusModalCloseButton`; New session
   does it on mobile only). iOS Safari's `:focus-visible` heuristic matches
   after programmatic focus with no previously-focused element, so the accent
   ring (clay `--focus-ring: var(--accent)`) shows on open. It shows in New
   session too; on a square button it reads as deliberate, on the degenerate
   pill it reads as broken. Pixel evidence: ring RGB ≈ accent #d97757 blended
   over paper, not --border #4a4236.

## The architectural inconsistency

The implicit contract — **icon-only dialog chrome is square at
`--dialog-control-h`; content-sizing belongs to text-bearing controls** — is
stated nowhere and is locally re-derived five times:

- `.formViewer .icon-btn { width: auto }` — the single rule that breaks it
- `#newSessionCloseBtn` + `#newSessionViewer .agentBackendTab` ID rescue
- `.diagViewer .queueHeader .icon-btn` surface rescue (square + border:0,
  using `--ctl-chrome`, a token aliased to the same value)
- `.queueIconBtn` local square re-statement (queue rows)
- `.queueViewer`/`.helpViewer` closes escape breakage only because they are
  not `.formViewer` surfaces (token coincidence via base `.icon-btn`)

Compounding factor: four modal surface vocabularies (`.formViewer`,
`.queueViewer`, `.diagViewer`, `.helpViewer`) over one identical header
pattern (`.queueHeader` + icon-btn close), so header rules scoped to
`.formViewer` spawned duplicates elsewhere. `border: 0` for header closes is
stated twice for this reason.

## Fix direction

State the contract once in app.css: icon-only dialog chrome (`.icon-btn`,
`.agentBackendTab` logo tab) square at `--dialog-control-h` in all four modal
surfaces; text-bearing `.choiceChip` keeps content-sizing. Generalize header
`border: 0` to `.queueHeader .icon-btn`. Delete all rescue blocks
(`#newSessionCloseBtn` group, `.diagViewer` group). No focus-behavior change
(ring is equal across dialogs today; changing it is a separate a11y decision).

## Verified collisions (why the square rule is safe)

- Zero text-bearing `.icon-btn` inside any dialog surface
  (`fileConflictActions .icon-btn.text-btn` lives in the file viewer status
  bar, not a dialog).
- `.agentBackendTab` contains only a 20×20 logo img → icon-only → square is
  its current rendered shape (via the ID rescue).
- All six `.queueHeader` instances are modal dialog headers (new session,
  queue, settings, diag, edit, help) → un-scoping border:0 touches only
  dialogs, harmonizing queue/diag/help to the borderless header every
  formViewer dialog already has.
