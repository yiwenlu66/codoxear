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
- Phase 3 inventory (executor c126549c): 35 composition `let`s mapped;
  60 direct accessor-key incidences (70 with selected aliases); nine
  dual-channel rendered surfaces with exact trigger paths; five dead/
  shadowed bindings found; risk-ranked migration order. Report at
  phase3-inventory.md. DESIGN PINS (main agent): seven-field store
  (selected, running, queueLen, subagentsRunning, turnOpen, sending,
  token); widgets subscribe internally; reducers write via atomic
  applyRuntime; slices = store/render-fields/selected+sending+turnOpen/
  dead-binding cleanup.
- Slice 1 (baf62edc): app_session_state.js + 7 behavioral tests. Suite
  1688 green.
- Slice 2 (5cb906db): running/queueLen/subagentsRunning/token migrated.
  Session display, typing row, queue badge subscribe internally;
  setStatus/setContext/setTyping/setSubagentsRunning contract keys dead;
  reducers write applyRuntime. Four dual-channel surfaces collapsed to
  single-trigger subscriptions. Docker verify PASS at 5cb906db.
- Slice 3 (run b50d4500, terra): selected/sending/turnOpen + alias
  collapse. Production migration implemented; executor TIMED OUT at 30m
  with 17/~101 harness migrations remaining (1671 passed). Resumed
  (6f1336f3) to finish the tail.
- Slice 3 complete (0c397b60): all 70 legacy accessor/alias incidences
  eliminated (measured against baf62edc baseline). 62 files.
- Slice 4 (a727e3d9): 3 dead bindings deleted; click_to_first_message_ms
  metric repaired (shadowed pair since the extraction era). Behavioral
  test for the repaired wire (2e436f1a).
- REGRESSION CHAIN (all caught by the Docker behavior gate, all made
  loud by the 4988b72d failure surfacing): (1) attachments boot crash —
  call-site literal missing sessionState (2168aec5 fix); (2) stale
  setSending shorthand in message-flow literal (93a88fe8) — invisible
  because unbound-option-value was scoped to 4 Phase-2 selectors;
  expanded guard to ALL selectors, surfacing 7 mismatches; (3) my own
  slice-4 error — a failed edit call silently dropped the composition
  markClickLoad rebinding while its deletions landed (a727e3d9),
  clickLoadT0 ReferenceError; completed in eb9fa9a8. Deploy-script tests
  caught a consistency issue: snapshot guard needs the allowlist commit
  (0a284514). Triage of the 7: iconSvg + dead unsaved-file duplication
  pruned from session edit, defaulted destructures excluded, appendEvent
  allowlisted with mandatory reason (0a284514).
- PHASE 3 GATE: suite 1690 green, guard clean, docker_verify PASS
  (eb9fa9a8). Adversarial review + behavior-interaction subagent next.
- Phase 3 adversarial review (critic sol, cceb6609): 7 objections.
  ADJUDICATION — ACCEPT: (1) idle subagent row never materializes
  (subscription split visibility from gauge); (2) guard lacks call-site
  coverage — the attachments-crash class; (3) attachment button stale on
  session switch mid-send (reads store, no subscription); (4) third
  channels: imperative render calls duplicating subscriptions;
  (5) sidebar active class still two DOM writers; (6) applyRuntime
  notifies per field (2 renders/patch vs old 1). REJECT (too harsh for
  this phase): (7) sessionState pass-through in ChatInteraction —
  construction-tree pattern, Phase 5 scope.
- INDEPENDENT behavior finding (executor, live Docker browser): typing
  row stuck "working" 12s after interrupt while sidebar went idle —
  pre-Phase-3 parity broken, invisible to deterministic gates.
- Revision wave 1 (sol executor): Bug A root cause = selected-session
  store write trapped behind transcript-identity reconciliation
  (`if (!slotChange.resetPending) return`); session refresh now applies
  selected-session runtime on EVERY fresh list response; Bug B fixed via
  createTypingRowStoreProjection (subagent transitions drive visibility
  per app_transcript.js semantics). Guard wave: select-callsite-coverage
  check added. Main agent overruled an executor's allowlist entry for
  stale selector key updateQueueBadge — deleted the key instead
  (allowlist is not for contract drift). Commit 780debde.
- Behavior re-verification (same executor, same flow): interrupt →
  API busy:false → sidebar idle → typing row absent at +12s. PASS.
- Revision wave 2: attachments subscribe sending/selected; redundant
  imperative renders removed (message_history queue badge, message_flow
  composer/attach syncs); queue mutations write sessionState.queueLen
  (store is queueLen's single authority for the badge); sidebar owns
  active class via selected subscription (imperative setActiveSession
  dead); applyRuntime notifies each shared subscriber once per patch.
  Commit d95da129.
- PHASE 3 COMPLETE: suite 1697 green, guard clean (incl. callsite
  coverage), docker_verify PASS + ui_flows OBSERVED (d95da129), live
  behavior re-verified, adversarial review adjudicated to zero.
- PHASE 4 (921c5676): app_topbar.js owns status chip, ctx chip,
  interrupt button end-to-end; app_session_display.js dissolved; shell
  keeps layout slots. Adversarial review (82eb27bd): 5 objections.
  ACCEPTED: docs authority (AGENTS.md/README named the deleted module),
  conversation-copy source-slicing test (removed per absolute policy;
  contract documented in AGENTS.md; full *_module_source.py family
  audited — all others behavioral), dead .status-chip.running CSS, stale
  test filename. PARTIALLY ACCEPTED: maximal one-module litmus —
  ownership boundary documented instead (topbar=logic, shell=slots,
  css=presentation, help overlay=shortcut table). Revision 9f61ec7c.
- Phase 4 accessibility verification (live Docker browser, VALIDATION.md
  boundary): all activation paths PASS (hint z/y, Enter, Space, pointer,
  hidden-state exclusion, no backend side effects, overflow). Two
  PRE-EXISTING geometry gaps filed as issues #30/#31 (ctx chip 32px
  mobile hit area; interrupt ::after slop possibly inert).
  docker_verify PASS at 9f61ec7c. PHASE 4 COMPLETE.
- PHASE 5 part 1 (eb9097ba): polling state extracted to
  app_polling.js createPollingRuntime (timers, enabled flags, streaks,
  generation). Docker PASS.
- Phase 5 adversarial review (critic sol, 663ef5f6; infra misclassified
  it failed, report complete): 12 objections, ALL ACCEPTED — none
  overruled. Highlights: (2) session-catalog/launch-defaults closure
  mesh survives (inventory rank-3 cluster); (3) LIVE dual-authority
  defect — backendSupportsFast reads the stale outer defaults copy,
  service_tier fast may be sent from fallback defaults; (4) my own
  polling-cluster spec put the async epoch inside polling — capability
  boundary regression; (1/6/7/8) composition still owns workflows
  (unattended-aggregate, confirm, copy, help, modal policy,
  title/placeholder multi-writer); (9-12) listener registry bypass,
  dead inputs/functions, stale comments. Revision decomposed: A
  (catalog + dual-authority, sol), B (epoch split + polling test gap),
  C (workflow extraction), D (hygiene).
- Revision A (7c0645e8, sol): app_session_catalog.js + dual-authority
  fix implemented; executor timed out at 30m mid-verification with
  suite green (1699); resumed for close-out (e637d727).
- Revision A committed (05eedb99): catalog owns latestSessions/
  sessionIndex(derived)/recentCwds/newSessionDefaults/tmuxAvailable;
  dual-authority fast-tier defect fixed with launch-authority
  regression test. Docker PASS.
- Revision B (392e6563): createAsyncEpoch split from createPollingRuntime;
  polling tests fire callbacks. Deploy tripwire modernized to the new
  ownership form (363942b2, 3a075b94).
- Revision C1 (171e213c): confirmation/conversation-copy/help workflows
  + modal policy extracted. TDZ boot crash from a const replacing a
  hoisted function (fixed f4da61a9 — hoisting-semantics lesson recorded:
  VM suite does not catch TDZ, only Docker boot does).
- Revision C2+D (586b41bb): updateUnattendedBtnState decomposed to named
  subscribing owners; title single-writer (audited: one
  titleLabel.textContent writer); transcript listeners cleanup-
  registered; dead inputs/functions/inits/comments swept. Docker PASS.
- Phase 5 closing review (critic sol, 05cbc4ec): 11/12 fixed; 3 new
  objections, all ACCEPTED, one root cause — catalog exposes shared
  mutable records so patches are invisible to subscriptions, which is
  WHY imperative relays survived. Closing revision (81940ea1):
  patchSession (observable in-place mutation) + applySnapshot (atomic
  multi-field publish) added; ALL imperative render relays removed —
  every examined transition covered by store/catalog notification.
  Commit d1907983. Behavior parity judged "identical" by the live
  Docker browser subagent (title rename, send lifecycle, search, nav,
  diagnostics, unattended all verified).
- PHASE 5 COMPLETE: suite 1708, guard clean, docker_verify + ui_flows
  PASS (d1907983), behavior parity verified, closing objections
  mechanically verified dead (patchSession at all 3 mutation sites,
  applySnapshot in refresh, zero relay calls).
- PHASE 6 (evidence-gated splits): cohesion audit (theorist d593bc28)
  verdicts — transcript/voice/file-viewer SPLITTABLE, new_session
  COHESIVE (respected; not touched). Splits: e498f5bb (search runtimes
  →chat_search, older-load →message_history, dead duplicate key),
  09c21567 (notifications →app_notifications.js, double-writer
  consolidated), f897b609 (touch-editor consolidation; save-conflict
  ReferenceErrors fixed — fileSaveConflictTarget recovered from git
  history c5078736). Adversarial review: older-load objection REJECTED
  (consumers calling an owner's public API is normal ownership);
  accepted: notification DOM seam completed, conflict token strengthened
  to full (sessionId,path,gitPath,apiPath) identity, dead duplicate.
  Behavior gate caught picker ReferenceError (menuState binding) —
  fixed with regression test (0ee3b292), live re-verified PASS.
  Audit also found: duplicate transcriptView key (dead syntax, removed).
  PHASE 6 COMPLETE: suite 1712, guard clean, docker_verify PASS,
  behavior re-verified.
- PHASE 7 (backend audit, theorist bc1565f2): GO, narrowly scoped.
  DEMONSTRATED LIVE DEFECT: cursor-relative poll/SSE deltas mutate
  shared Session model/provider/effort/token with no monotonic guard —
  mark_log_delta ignores new_off; probe regressed new-model→old-model
  through the production poll route on a >2MiB log. Token has the same
  class via lockless direct writes. Rejected: broad server.py/broker
  rewrites, SessionStore/voice/launch-config work, broker/sessiond
  merger (all LATENT-STRUCTURAL at worst). Scoped phases: B1 monotonic
  log-derived projection owner, B2 poll/SSE unified live-delta
  projection, B3 manager wiring contracts (lowest priority).
- B1 (73d20c50 + amendment): LogDerivedSessionObservation (log path,
  device/inode/size/mtime revision, byte range) committed atomically
  via commit_log_observation under the registry lock; stale/prior-
  generation observations rejected; rebind/truncation accepted as new
  generations. Regression-verified against pre-fix archive.
- B2 (8faf4a31): _project_live_record_window unifies poll and SSE;
  parity tests through both production handlers incl. mutation-spy
  verification that bypassing the shared seam fails.
- B3 (e51bb6a3): retained coordinator graph (96 vararg methods → 0,
  mega-caps → 22 focused dependency records, queue-sweep cursor homed,
  13 dead relays removed, 63 relay signatures mechanically checked).
- FINAL ADVERSARIAL REVIEW (critic sol, 3df58821): B2 accepted; B1
  rejected pending correction — 5 objections, all ACCEPTED and fixed
  (c2daefd4): same-log live bridge settings dropped (priority
  violation), append-between-stat-and-read ValueError killing polls/
  SSE during normal generation, stale listing rows on rejected
  backfills, B3 lazy-init race; truncate-regrow DOCUMENTED not fixed
  (partial fingerprint would be worse — full design recorded in audit
  boundary). Each correction regression-verified against pre-correction
  archive (5 failures there). Commit 912f58db.
- REFACTOR COMPLETE 2026-08-17: suite 1728, guards clean, docker_verify
  + ui_flows PASS (912f58db), behavior judged "identical" across all
  phases by live Docker browser subagents, all adversarial reviews
  adjudicated to zero open objections.
- FINAL ACCEPTANCE AUDIT (critic sol, 48e1f2f9): REJECTED the completion
  declaration with 7 numbered objections. ADJUDICATION: accepted 6 —
  ratchet not monotonic across commits (subset-vs-HEAD only guards
  uncommitted changes; hardened to reason-required entries + pinned
  counts + tripwire test + documented boundary), sidebar active class
  and title initial-writer splits (unified into single writer functions
  invoked at construction and on change), dead queue render relays
  (removed with their export), dead selector key prefersReducedMotion
  (full 83-selector overcoverage sweep found only that one),
  isSidebarOpen undocumented input (made explicit), app.js impossible
  load guard (removed — app.js itself was missed in Phase 1). REJECTED:
  const-global retention (verified live fallbacks, recorded). Docs
  promoted to canonical AGENTS.md/README/ARCHITECTURE.md (opus
  executor) including the initial-render rule. Revision 6fae1114.
  Docker PASS. Re-audit for final verdict dispatched (359a81bc).
- FINAL ACCEPTANCE RE-AUDIT (359a81bc): ACCEPT — all seven criteria
  pass (guards+ratchet, single owners/writers, topbar litmus, honest
  wiring map, backend ordered projection, plan phases/invariants,
  behavior parity). No blocking findings. Declared boundaries stand:
  truncate-regrow, geometry #30/#31, VM projection, bounded backend
  audit. TASK CLOSED at 6fae1114.
