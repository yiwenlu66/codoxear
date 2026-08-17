# EPISTEMIC — frontend architecture refactor

## What is being explained
Why a one-concept UI change (remove Busy/Idle + ▸N from the topbar chip)
required edits in 6 JS files, and what target architecture makes that class of
change single-module by construction.

## Verified diagnosis (evidence in OPS.md 2026-08-16 entries)
- `app_wiring.js` enforces per-controller `select()` contracts for ~65
  controllers, but exactly three factories are pass-throughs
  (`createChatInteractionOptions`, `createSessionDisplayOptions`,
  `createFileOpsOptions`) — holes in an otherwise disciplined fence.
- The chat subtree spreads: composition `{...deps}` → chat_interaction
  `{...options}` → transcript_render / send_lifecycle. Any name in the bag can
  reach any leaf; in practice names do (e.g. `renderStatusChip` appeared in
  three modules that never called it).
- Composition owns ~35 closure `let`s with hand-written getter/setter pairs
  injected into controllers — a hand-rolled state store with worst-case
  ergonomics, and the mechanism behind dual-channel rendering (two modules
  re-invoking one render through different paths).
- Widget fragmentation: the status chip's element, renderer, state, and render
  triggers lived in four different modules (extraction-wave boundaries, not
  cohesion boundaries).
- Pre-ESM vestiges persist after the ESM conversion: `const global = window`,
  mid-file `"use strict"`, runtime "failed to load" checks for statically
  imported modules.
- `check_wiring.py` exists but only runs at deploy time — nothing ratchets the
  architecture during development.

## Live commitments (the plan's bets)
- Constructor-injected DI is preserved: it is what the VM-harness behavioral
  tests inject through. Discipline, not replacement.
- A minimal per-field get/set/subscribe session-state store is justified over
  continued closure getter/setter plumbing: it makes state authority structural
  and eliminates cross-module renderer calls. Deliberately small — no events
  bus, no middleware. VALIDATED through Phase 3: 70 accessor incidences
  eliminated, nine dual-channel surfaces reduced to store subscriptions (after
  two revision waves caught by adversarial review + live behavior checks).
- Guards with a shrink-only allowlist ratchet are the mechanism that keeps
  garbage from regrowing; cleanup without the ratchet re-fills. VALIDATED:
  the ratchet caught zero regressions itself, but its expansions
  (undercoverage, unbound-value, callsite-coverage) systematically surfaced
  the latent defect classes the behavior gate kept finding.

## Phase 3 lessons absorbed
- The dominant defect class of explicit wiring is option-literal mismatch:
  unbound values, case typos, missing call-site keys, stale selector keys.
  Now guarded statically in all four variants.
- Subscriptions must own BOTH the data and the visibility/materialization of
  their widget — splitting them across subscriptions recreates dual-channel
  bugs in subtler form (Bug B).
- A store write gated behind an unrelated reconciliation condition silently
  traps liveness (Bug A); runtime projection must be unconditional per fresh
  snapshot.
- The behavior gate (live Docker browser) is the only check that has caught
  every integration-level regression class at least once; deterministic gates
  each have blind spots the others cover.

## Ruled out
- Replacing DI with imports/globals/framework (breaks the test injection seams).
- Numeric size targets (user: "pure theatre") — criteria are judgment-based,
  adjudicated via adversarial review.
- Backend refactor without evidence — Phase 7 is an audit with go/no-go, not a
  commitment.

## Unaudited / open
- Whether `app_transcript.js`, `app_voice.js`, `app_file_viewer_operations.js`,
  `app_new_session.js` contain genuine second responsibilities (Phase 6 is
  evidence-gated; some bulk may be wiring plumbing that Phases 2–3 remove).
- Whether the closure-state mesh hides cross-field invariants a store must
  preserve (Phase 3 discovery work).
- Backend Python (server.py, message_routes.py, broker.py are size outliers;
  no evidence of rot yet).
- Pre-existing anomaly noted during Phase 1: esbuild warns of a duplicate
  `transcriptView` object key in app_transcript_render.js (~lines 844/858) —
  one definition silently overwrites the other; investigate in Phase 2/6.

## Known accepted limitations
- VM-harness projection: namespace globals resolve dynamically at call time
  in tests, where production ESM bindings are fixed at import (critic Phase 1
  objection 2, rejected as too costly to fix). Consequence is limited to
  test ergonomics: partial namespace mocks fail at first call, not at load.
- The wiring guard is a tripwire for idiomatic drift, not a proof system:
  deliberate obfuscation (helper indirection, bag renaming, defineProperty
  globals) is out of its scope and covered by per-phase adversarial review.

## Question that would most change the model
ANSWERED (Phase 2a): the chat subtree's true dependency surface is ~140
keys — large, so Phase 3's store is the critical path, and Phase 2's
explicit contracts are the map Phase 3 will shrink. Confirmed further by
pruning: ~40-50% of mechanically derived contract keys were dead.

## Phase 2 lessons absorbed
- Under-selection and unbound values in giant option literals are the
  silent-failure classes of explicit wiring; both are now guarded
  statically (select-undercoverage, unbound-option-value) and the
  post-login boot path surfaces failures in the live UI.
- Behavior verification gates EVERY product-code commit, including
  review revisions — the one time it didn't (4d50c9c4), a boot crash
  shipped forward.
- Token-blind text scanning is unsafe for identifier analysis (the '$'
  misprune); use token-aware scans.
