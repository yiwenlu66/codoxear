# EPISTEMIC — frontend architecture refactor

## STATUS: COMPLETE (2026-08-17)

All plan phases done and gated: 0 guards, 1 ESM hygiene, 2 wiring
discipline, 3 session-state store, 4 widget cohesion, 5 composition
residual, 6 evidence-gated splits (3 splits, 1 respected COHESIVE
verdict), 7 backend (audit → GO on demonstrated live defect → B1
ordered projection, B2 unified live-delta, B3 manager wiring, plus
5 corrections from the final review). Suite 1728 green; guards clean;
Docker behavior parity judged "identical" per phase; all adversarial
reviews adjudicated to zero open objections.

## What the architecture is now
- Every mutable state field has one owner: app_session_state (7-field
  runtime), app_session_catalog (session list/defaults, derived index,
  observable patchSession, atomic applySnapshot), app_polling
  (scheduling/streaks + separate createAsyncEpoch).
- Every rendered value has one trigger: widget-internal subscriptions.
  No cross-module renderer calls; imperative render relays are gone.
- Controller wiring is explicit per-controller select() contracts; the
  wiring guard (pass-through/bag-spread/direct-bag/global-registration/
  undercoverage/unbound-value/callsite-coverage) with a monotonic
  allowlist ratchet prevents regrowth.
- Backend: log-derived session projection commits through one ordered,
  log-identity-aware boundary; poll and SSE share one live-delta
  projection; manager coordinators are retained with focused contracts.

## Documented residuals (not defects of the refactor)
- Truncate-regrow projection poisoning: same-inode truncate+regrow past
  the old boundary can be accepted as forward progress. Full fix needs
  read-coherent prefix fingerprints; design recorded in
  phase7-backend-audit.md boundary section. Low benign likelihood.
- Issues #30/#31 (topbar geometry, pre-existing) in ISSUES_TRACKER.md.
- VM projection limitation: namespace globals resolve dynamically in
  tests (Phase 1, accepted).
- Backend audit was bounded: not every route/auth/git path reviewed.
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
