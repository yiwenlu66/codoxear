# Frontend architecture refactor plan

Approved by user 2026-08-16 with three amendments, integrated below:
1. No numeric targets (line counts etc.) in criteria — that is theatre. Criteria are
   judgment-based and adjudicated through adversarial review.
2. Adversarial subagent-based reviews at every phase boundary. Whether a review
   objection is real, too harsh, or out of scope is the main agent's subjective
   judgment, recorded in OPS.md.
3. Verification is never only deterministic scripts/tests. Every phase that touches
   product code also requires a subagent interacting with the Docker-run service
   through a real browser, exercising user flows and judging behavior parity
   subjectively.

Ownership: the main agent owns the process, the epistemic state, phase gating, and
review adjudication. Detail work (mechanical edits, audits, verification runs) is
delegated to subagents.

## Target architecture

Every piece of mutable state and every rendered value has exactly one owner module.
Controllers receive minimal per-controller option contracts built by `select()`
lists in `app_wiring.js` — never a shared bag, never a spread. Session-runtime
state lives in one small observable store; renderers subscribe to it instead of
being re-invoked from distant modules. A widget (element + render + state
subscription) lives in one file. Composition only boots and assembles. Structural
guards make the rules self-enforcing so the garbage cannot regrow.

## Invariants (non-negotiable)

- Constructor-injected DI stays. It is load-bearing: the testing policy forbids
  source-string tests, so behavioral JS tests load modules in node:vm and inject
  mocks through these seams. Discipline the injection; do not replace it.
- Locked DOM branches stay locked (session-card touch/desktop split and the other
  sanctioned viewport branches in AGENTS.md).
- No behavior change. Any behavioral difference discovered is either a bug (fixed
  in its own commit with justification) or a refactor mistake (reverted).
- Backend Python is out of scope until the Phase 7 audit produces evidence.

## Phases

### Phase 0 — Architecture guards (stop the bleeding first)
Extend `scripts/check_wiring.py` to fail on:
- pass-through option factories in `app_wiring.js` (`return deps` verbatim);
- `...options` / `...deps` spread into controller factory calls;
- new `window.Codoxear*` / `global.Codoxear*` registrations.

Current violations ship in an explicit checked-in allowlist that can only shrink
(the ratchet); each later phase burns it down. Wire the guard into the pytest
suite (precedent: `tests/test_check_js_refs.py`) — today it only runs at deploy
time. The guard itself must be behaviorally verified: plant a violation, watch it
fail.

### Phase 1 — ESM hygiene sweep
Mechanical, repo-wide deletion of pre-ESM scaffolding: `const global = window`,
mid-file `"use strict"` (ESM is strict by definition), and composition's runtime
"failed to load" checks for statically imported modules (cannot fail; vestiges of
the script-tag era). One commit, zero logic change. Wide-touching: announce in
intercom before starting, commit immediately (a sibling session already destroyed
uncommitted work once).

### Phase 2 — Wiring discipline (fixes the demonstrated bug class)
- Give the three pass-through factories (`createChatInteractionOptions`,
  `createSessionDisplayOptions`, `createFileOpsOptions`) real `select()` lists,
  derived mechanically from each controller's header destructure.
- Eliminate the spread chain composition → chat_interaction → transcript_render /
  send_lifecycle; wiring provides per-controller lists instead.
- Burn the Phase 0 allowlist to empty for these checks.
Effect: a name like `renderStatusChip` can never again appear in four modules when
one needs it.

### Phase 3 — Session-state store (the deep change)
Replace composition's ~35 closure `let`s and hand-written getter/setter pairs with
`app_session_state.js`: a minimal per-field get/set/subscribe store owning the
selected session's runtime projection (selection, running, queue length, subagent
count, turn-open, sending, token snapshot). Polling timers/error streaks get their
own `app_polling.js` owner. Renderers subscribe; updaters write through the store.
No renderer function crosses a module boundary again — this kills the
dual-channel bug class (two modules re-invoking one render).

### Phase 4 — Widget cohesion
Codify the status-chip lesson: a widget is element + render + subscription in one
module. `app_topbar.js` owns status chip, ctx chip, and interrupt button
end-to-end (DOM creation out of `app_shell.js`, render out of
`app_session_display.js`, state from the Phase 3 store). `app_session_display.js`
dissolves. Litmus test: "remove X from the topbar" touches exactly one module.

### Phase 5 — Composition residual
After Phases 2–4, composition retains only boot sequencing, controller assembly
via wiring, and cleanup-registered listeners. Anything else remaining must be
justified by name or extracted. Judged by adversarial review, not by line count.

### Phase 6 — Oversized module splits, evidence-gated
`app_transcript.js`, `app_voice.js`, `app_file_viewer_operations.js`,
`app_new_session.js` — split only where a second responsibility is demonstrably
present, along existing section boundaries, one module per commit. Size alone is
not grounds; cohesion failure is. Some shrink on their own after Phases 2–3.

### Phase 7 — Backend audit (bounded, go/no-go)
Timeboxed audit of the Python side (size outliers: `server.py`,
`message_routes.py`, `broker.py`) for the same failure signatures: pass-through
plumbing, split ownership of single values, god objects. Output is a go/no-go
decision with a scoped phase list — not a preemptive commitment.

## Verification protocol (every phase boundary)

Deterministic gates:
1. Full test suite green (expect VM-harness mock updates in Phases 2–4 — mocks
   inject through the seams being narrowed; that is the tests doing their job).
2. `check_js_refs.py` + extended `check_wiring.py` clean.
3. Bundle rebuilt in the same commit as its source.

Behavioral gates (for every phase touching product code):
4. `scripts/docker_verify.sh HEAD` green.
5. A behavior-interaction subagent drives the Docker-run service through a real
   browser (agent-browser): login, select session, send a message, observe
   streaming, queue behavior, interrupt, file viewer, slash commands, sidebar
   states. The subagent judges — subjectively, from screenshots and interaction —
   whether the product behaves identically to before the phase. "Tests pass" is
   not evidence; observed behavior is.

Adversarial review (every phase):
6. A fresh-context adversarial reviewer subagent attacks the phase's premise:
   is the architecture actually more correct, or did the garbage just move? The
   review is framed at the level of real uncertainty, not mechanics. The main
   agent adjudicates each objection (accept / reject as too harsh / reject as out
   of scope) and records the adjudication in OPS.md.

## Rollback and coordination

Each phase is one or a few atomic commits; rollback is `git revert` of the phase.
Wide-touching phases (1, 2) must be sequenced against other active sessions in
this tree: announce in intercom before starting, commit immediately.

## Success criteria (judgment-based, review-adjudicated)

- Guard-enforced: zero pass-through factories, zero spread-into-factory, zero new
  globals — and the allowlist ratchet proves nothing regrows.
- Every mutable state field has exactly one owner module; every rendered value
  has exactly one writer, documented in code.
- The status-chip litmus test: a change like "remove ▸N from the topbar" touches
  exactly one module.
- `app_wiring.js` reads as the complete, honest map of who needs what.
- A browser-driving subagent confirms, through real interaction, that the product
  behaves identically after each phase.

## Explicitly refused

Rewriting the backend on suspicion. Replacing DI with imports/events/a framework.
Touching locked DOM branches. "While we're in there" behavior changes. Numeric
size targets of any kind.
