# OPS — frontend architecture refactor

Append-only evidence trail. Cross-reference EPISTEMIC.md.

## 2026-08-16

- 21:10–22:00 — Trigger investigation. User asked why the topbar showed
  "Idle/Busy" redundant with existing indicators. Git archaeology: chip hid
  while busy in initial commit `d958e6bf`; `6fd57321` ("Show Pi thinking tokens
  in typing stats") made it always-visible to carry `▸N`, introducing "Busy" as
  filler text; `6a91c7f5` added "Busy · Queue N". No user instruction for the
  label exists in history or memory.
- 22:0x — User ordered removal. Implemented chip = queue-only payload across 6
  JS files + 2 tests + bundle. All 1676 tests green; `check_js_refs` clean.
- 22:55 — INCIDENT: uncommitted edits wiped by a working-tree restore at
  22:55:18. Sibling session `subagent-chat-01a009a8` (slash-command task) active
  in same repo; its dirty files untouched, mine reverted — near-certain broad
  `git restore/checkout` by that session or its parent. Intercom ask failed:
  two sessions report the colliding id `01a009a8`. Work redone from context and
  committed immediately as `9e2e902e`. Lesson logged in PROMPT.md constraints.
- 23:0x — Docker behavioral verification of commit range HEAD=`24682c44`
  (includes `9e2e902e`): PASS, all browser checks green; accessibility snapshot
  contains no "Idle"/"Busy"/"▸" in the topbar.
- 23:1x — User: "why does such a simple change need changing wiring in so many
  js files? things are architecturally wrong." Analysis delivered: 2 of 6 files
  behaviorally necessary, 4 were hygiene for bag-spread scatter. Root causes:
  pass-through wiring holes, spread-inherited options bag, dual-channel updates
  for one rendered value, widget fragmentation.
- 23:2x — User ordered full architectural refactor plan. Audit before planning:
  `app_wiring.js` has `select()` discipline for ~65 controllers; exactly three
  pass-throughs (`createChatInteractionOptions`, `createSessionDisplayOptions`,
  `createFileOpsOptions`); composition spreads `...deps` into chatInteraction
  (line ~682); chat_interaction spreads `...options` into transcript_render
  (x2) and send_lifecycle; composition holds ~35 `let` state vars with
  hand-written getter/setter injection; `check_wiring.py` runs only at deploy
  time (deploy.sh:126), not in the test suite. Modules still carry pre-ESM
  vestiges (`const global = window`, dead "failed to load" checks for static
  imports).
- 23:3x — Plan delivered. User approved with amendments: (1) no numeric targets
  in criteria — "pure theatre"; (2) adversarial subagent reviews, harshness
  adjudicated by main agent's subjective judgment; (3) verification must
  include subagent browser-interaction with the Docker service and subjective
  behavior-parity judgment, not only deterministic gates; (4) main agent owns
  process, delegates detail work.
- 23:4x — Task persisted: PROMPT.md, PLAN.md (amendments integrated), OPS.md,
  EPISTEMIC.md. Phase 0 delegated to implementation subagent (async).
- 23:5x — Phase 0 implementation complete (run f739e8ba): three new checks,
  allowlist with 3 pass-through factories + 5 spread sites (one previously
  uncatalogued: file_viewer_controller → createFileViewerOperationsRuntime),
  0 global registrations. Guard + new tests pass; full suite 1679 green.
- 2026-08-17 00:1x — Adversarial review (critic, run 3a622b30, sol): 11
  objections, core conclusion that the guard matches exact textual spellings
  and is evadable. ADJUDICATION (main agent):
  ACCEPT + fix: (1) ratchet not monotonic → pytest subset-check vs
  `git show HEAD:` allowlist; (3/4/10) spread check inverted to "any
  `...options`/`...deps` spread element in an object literal" with correct
  bracket depth — surfaced real-tree hits in app_transcript_view.js the old
  check missed; (6) global check broadened to any window/globalThis property
  assignment incl. bracket-string; (7) strip comments/strings lexically before
  scanning; (9) guard emits scan coverage so empty-allowlist real-tree test is
  not a vacuous oracle; (8-micro) loader rejects duplicate entries; (11)
  deploy.sh must run the guard from the deploy snapshot, not the editable
  checkout (pre-existing release-integrity hole).
  ACCEPT AS PHASE 2 SCOPE: (5) raw `hooks` bags in app_file_ops.js /
  file-viewer subtree bypass wiring contracts — Phase 2 expands there; too
  heavy for a static guard rule.
  REJECT (too harsh — tripwire polices idiomatic drift, not intent; deliberate
  obfuscation is what per-phase adversarial review is for): helper-identity
  indirection, bag-variable renaming, Object.assign/defineProperty globals,
  same-name violation substitution.
- 00:1x — Phase 0 revision dispatched to same executor (resume, run 6a18a111)
  with adjudicated fix list.
- 00:3x — Phase 0 revision complete: monotonic ratchet (subset vs HEAD),
  inverted bag-spread rule (9 real-tree hits incl. previously invisible
  app_transcript_view.js and file-ops/video/transcript sites), broadened
  global rule (8 existing registrations allowlisted: codoxearPerf,
  __codoxearAppBootstrapped/__codoxearLoadError et al. in index.html),
  lexical masking, scan-coverage reporting, duplicate-entry rejection,
  deploy.sh runs both static guards from the deploy snapshot. Main agent
  extended the same self-containment fix to check_js_refs.py (identical
  hole class). Full suite 1681 green; guard clean (files=69,
  option_factories=71). COMMITTED 3174e721. Phase 0 done; newly catalogued
  bag spreads feed Phase 2 scope.
- 00:5x — Phase 1 (ESM hygiene) implemented (executor, run 459abb82): 65
  strict directives, 14 dead global bindings, 69 dead load guards removed;
  bundle rebuilt 582.5→559.6kb; suite 1677 green (obsolete guard tests
  removed per approved protocol; latent Counter import fixed). COMMITTED
  bad72b4d.
- 01:1x — Phase 1 verification: behavior-parity subagent (docker_verify +
  docker_ui_flows vs HEAD): BOTH PASS, zero browser errors/console output,
  queue badge and /model picker flows correct, topbar shows no Idle/Busy;
  verdict "identical". Caveat: harnesses tear down containers, so no manual
  extra flows; offline Pi bootstrap logs show a provider OAuth 403
  (region) that does not affect the app.
- 01:1x — Phase 1 adversarial review (critic, sol): 5 objections.
  ADJUDICATION:
  ACCEPT: (1) REMOVING "use strict" BROKE VM-HARNESS STRICTNESS —
  frontend_module_loader.py wraps modules in non-strict IIFEs, so tests now
  execute sloppy-mode while production is strict; fix the loader projection
  (prepend "use strict" in the wrapper) + behavioral strictness test.
  (3) SWEEP INCOMPLETE — main agent verified the critic's facts against
  source: executor's skip rationales misread in-file objects
  (codoxearDom/codoxearUrls/codoxearEventBindings) and static imports
  (codoxearSessionHelpers/codoxearPolling in app_application.js,
  app_file_picker.js top guard, app_transcript_render.js dual guard) as
  injected; all are the dead class and must go. (5) stale comment at
  app_application.js:124-128.
  REJECT (too harsh to fix now; recorded as known projection limitation):
  (2) VM fixtures resolve namespace globals dynamically where aliases used
  to capture once — test-ergonomics divergence only, no production impact.
  PROCESS NOTE (valid, no code action): (4) the Phase 0 Counter fix rode in
  the Phase 1 commit — future phase commits stay scope-pure.
- 01:2x — Phase 1 revision dispatched (resume, run 20aacf75).
- 01:4x — Phase 1 revision complete: loader wrapper now emits "use strict"
  (tests/frontend_module_loader.py:139) + behavioral ReferenceError test;
  all five misdiagnosed dead-guard clusters removed; stale comment fixed;
  suite 1678 green. COMMITS: 4d50c9c4 (revision), ad69e491 (strictness
  test), e89dbdc0 (task memory; PROMPT.md stays local per .gitignore:4).
  PHASE 1 COMPLETE — gates: suite green, guards clean, Docker behavior
  parity judged identical, adversarial review adjudicated.

## 2026-08-17

- Phase 2 sub-phases committed: 4a44af21 (2a chat subtree, select lists
  139/141/137/34 keys — the honest dependency surface is LARGE, answering
  EPISTEMIC's open question: Phase 3 is the critical path), 0d3c963c (2b
  session display 11 + file ops 127), 982d3dc4 (2c file subtree incl.
  hooks-bag dissolution, 94/35/24/15/48), aafbfbb0 (2d transcript
  leftovers). Allowlist reached end state: 0 bag spreads, 0 pass-throughs,
  8 reviewed-benign boot globals.
- Phase 2 adversarial review (critic sol, run 1de1c0bd; fable attempt
  986e6569 timed out at 30m — model override applied): 3 objections, all
  ACCEPTED: (1) guard accepts direct bag argument createX(options);
  (2) select() under-selection is silent — later proven in production;
  (3) contracts ~40-50% dead keys (57/141, 74/137, 1/127 measured).
- REGRESSION HUNT (the big one): Docker gate failed at HEAD with cards:0,
  ZERO browser errors, bootstrapped=true. Deterministic. Instrumented
  debugging executor (sol, run 36dbf0b2) localized: 4a44af21 introduced
  two case-typo'd shorthand values (codoxearCodeCopy/codoxearModal vs
  CodoxearCodeCopy/CodoxearModal) in composition's deps map; ReferenceError
  aborted chat-interaction construction; the login form's catch wrote the
  error into a DOM node renderApp had ALREADY DETACHED — total boot
  failure invisible to every deterministic gate. Fixed in 8f46c325 (after
  an earlier same-class boot crash, codoxearDom, fixed in 8f8b1e04).
  Docker verify PASS at 8f46c325.
- PROCESS CORRECTION (main agent error): the Phase 1 revision commit
  4d50c9c4 was gated suite-only; the codoxearDom boot crash it introduced
  reached Phase 2's gate. Behavior verification now gates EVERY
  product-code commit including review revisions.
- Phase 2 revision (parallel executors, run 97d4273f): guard hardening —
  direct-bag-argument, select-undercoverage, unbound-option-value checks
  (the last catches the boot-crash class statically); dead-key pruning
  57+75+1 keys; boot-error surfacing fix (app.js: post-login bootstrap
  failure now console.errors AND replaces #root with a role=alert panel).
  Commits ce760533, 4988b72d. Parallel executors raced on the bundle —
  caught by main agent, rebuilt from combined sources before commit.
- Second regression: pruning mis-scanned '$' (token-blind regex) in
  send_lifecycle; getTray $("#stagedAttachments") threw at boot — and the
  NEW failure surfacing made it loud in the Docker gate (console showed
  the exact ReferenceError). Token-aware audit of all 133 pruned keys:
  exactly one misprune. Fixed 70a12140.
- PHASE 2 COMPLETE: suite 1681 green, guards clean, docker_verify PASS
  (70a12140), docker_ui_flows OBSERVED (queue + /model picker flows
  correct).
