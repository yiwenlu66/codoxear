Confirmed: `3f08d37` is a memory-only commit (the OPS entry dispatching me and executor `83f22848`). Functional HEAD is still `f278474`, and the parallel executor's scope (stale-interrupted-idle → `session_runtime`/`session_listing`/broker turn state) is disjoint from my top recommendation's scope (transcript projection). My analysis is complete.

---

# Next Codoxear product target — ranked shortlist

## Summary of the decisive finding

The "selectable backend promise" invariant (ARCHITECTURE.md: *a turn that records user input then terminates without assistant output or explicit error must emit a truthful visible no-response/failure event; ordinary idle silence violates the contract*) is enforced **only for Codex**. The single implementation, `_inject_no_response_events` in `codoxear/rollout_chat_events.py`, is hard-gated on Codex `event_msg` rows and its own docstring states "valid Pi and Claude Code turns are unaffected." Claude Code — an advertised backend tab — closes turns through a **`system`-row family** that Codoxear's transcript projection is blind to. This is a single-authority-doesn't-cover-all-cases mismatch, evidence-backed by both a synthetic projection probe and real log structure, and it is fully agent-actionable (no credentials/gateway needed).

---

## 1. TOP — Claude Code turn-outcome transcript truth (`system`-row blind spot) · agent-actionable

**Mechanism.** Three code facts combine into a silent no-answer on the CC tab:
- `_single_chat_event` (`codoxear/rollout_chat_events.py`) returns `None` for every `system` row — so no `system`-row content ever becomes a transcript message.
- `cc_is_turn_end` (`codoxear/cc_log.py:348`) recognizes only `system`/`subtype:turn_duration`; `cc_current_turn_state_before` (`cc_log.py:~279`) sets `idle = not pending` on that close. So a turn can go **idle** with no assistant text.
- `_inject_no_response_events` is Codex-only, so no synthetic no-response row is added for CC.
- CC error detection (`cc_assistant_is_api_error`, `agent_backend.py:~783`) only matches `assistant` rows with `isApiErrorMessage`. The real `system`/`subtype:api_error` rows are not seen by projection or the reducer.

**Evidence status — confirmed (read-only probe + real logs):**
- Synthetic probe (in-memory, no runtime): a CC `user` → `system/turn_duration` log projects **only the user message** (both live-delta and tail). Same for `user` → `tool_use` → `tool_result` → `turn_duration` (tools ran, no final text). The answered control turn projects correctly.
- Real logs (`~/.claude/projects`, structural scan of 384 files): `system/api_error` appears **64×**, `system/turn_duration` **5×**. `system/api_error` rows carry an `error` text field + `retryAttempt`/`maxRetries` (retry notifications). Several logs contain `system/api_error` rows with **zero** `assistant/isApiErrorMessage` rows.
- Negative-coverage proof: the existing CC guard `test_cc_live_delta_does_not_emit_no_response` (`tests/test_codex_no_response_projection.py:568`) only asserts an *answered* CC turn gets no false no-response. The silent-close case is untested — uncovered, not proven safe.

**Why it matters to a real user.** Claude Code is one of three backend tabs. When a CC turn ends without a final answer — a tools-only turn that stops, retries that exhaust, or a bare `turn_duration` — the user sees their prompt, the session goes idle (or spins), and there is no answer, no error, and no "no response" message. This is exactly the failure the Codex no-answer fix eliminated, reproduced on the CC tab. The most common terminal CC error (`assistant/isApiErrorMessage`, e.g. gateway 503) *is* handled; the gap is the `system`-row close family.

**Cheapest decisive discriminator (agent, pure Python, no browser/credentials):** construct CC logs — (a) `user` + `system/turn_duration`, no assistant text; (b) `user` + tool_use + tool_result + `turn_duration`; (c) `user` + terminal `system/api_error` (`retryAttempt >= maxRetries`), no following assistant — read each through `_read_chat_live_delta`/`_read_chat_tail_page` and assert a visible no-response/error event exists. (a) and (b) already reproduce silent output; (c) is the one still to classify. Then confirm against the `~/.claude` structural scan whether terminal `system/api_error` ever occurs without a trailing `assistant/isApiErrorMessage`.

### Concrete intervention contract (for the main agent to hand an executor)

- **Goal.** Make Claude Code turn-closes obey the same visible-outcome guarantee as Codex: a CC turn that closes with no assistant text and no explicit error must project a truthful no-response event; a terminal `system/api_error` (retries exhausted) with no trailing assistant error row must project an error-styled event. Do not change Codex or Pi answered-turn behavior. Do not invent assistant content.
- **Discriminator first.** Before any fix, add failing tests reproducing shapes (a)/(b)/(c) above and record current (silent) output. If any already projects a visible event, stop and report.
- **Files in scope.** `codoxear/rollout_chat_events.py` (generalize/extend the no-response detector to CC turn-close semantics), `codoxear/cc_log.py` (expose a CC "turn closed with no assistant output since last user" detector and a terminal-vs-transient `system/api_error` classifier via `retryAttempt`/`maxRetries`), `codoxear/agent_backend.py` (CC `chat_event_from_log_row` if terminal `system/api_error` should project as `message_class:"error"`), `codoxear/rollout_log.py` (caller wiring for split live-delta/history windows, mirroring `_codex_prior_open_turn_context`), new `tests/test_cc_no_response_projection.py` + extend `tests/test_cc_backend_error_projection.py`.
- **Hard constraints.** No live runtime / sockets / server; no commit; no staging; no broad refactor. Reuse `_NO_RESPONSE_TEXT` + `message_class:"error"` (already browser-certified renderer). Transient retries (`retryAttempt < maxRetries`) must NOT each spawn a message — only a terminal outcome. Keep scope disjoint from executor `83f22848` (stale-interrupted-idle touches `session_runtime`/`session_listing`/broker turn state, not transcript projection — confirm before dispatch).
- **Validation.** `pytest -q tests/test_cc_no_response_projection.py tests/test_cc_backend_error_projection.py tests/test_codex_no_response_projection.py tests/test_cc_log.py` (Codex/Pi must stay green) → full local pytest. Certification (main agent, not executor): Docker/browser proof injecting CC silent-close and terminal-error logs into a discovered session, verifying the browser transcript shows the message — mirroring the Codex no-answer browser proof (claim4, sandbox 19200).
- **Output shape.** Unified diff summary + focused test output + a short table of which CC close shapes now project a visible event vs. remain intentionally silent (transient retries).
- **Stop rules.** If real CC never emits `turn_duration`/terminal `system/api_error` without a trailing assistant row (verify via `~/.claude` scan), downgrade to defensive tests + a memory note instead of a behavior change — do not invent events for a shape that cannot occur.

---

## Alternatives (with why not first)

**2. Degraded-backend failure-state BROWSER certification for Codex & CC · agent-actionable.** The projection logic is unit-tested and the `message_class:"error"` renderer is browser-certified (via Codex no-answer), but no browser run has exercised a *CC* error/no-answer end-to-end. *Not first* because it is the verification follow-on to #1 — certifying before fixing #1 would certify a known-silent path. Do it immediately after #1's projection fix.

**3. `.fileTouchBtn` mobile dpad touch targets (34px → 44px) · agent-actionable.** Confirmed open: the file-viewer image/video pan dpad is laid out on a fixed 34px grid (`app.css:1966–1980`), and the D3 mobile fix explicitly excluded it (`app.css:2763–2764`; OPS 2026-07-05T18:40 residual). *Not first* because it is a secondary pan control (the primary file-viewer toolbar — Edit/Download/Close — is already 44×44 certified), and it is a layout/design question, not a truthfulness defect. Cheap discriminator: measure at 390×844 with the existing browser harness.

**4. Pi aborted / user-interrupted turn visible recovery message · agent-actionable, related to #1.** `test_pi_aborted_turn_does_not_emit_no_response` confirms an aborted Pi turn projects only the user message (Pi renders nothing for aborts). *Not first* because interruption is user-initiated (the user knows they interrupted) and the interrupt affordance already gives feedback; lower severity than an unannounced backend failure. Belongs to the same "non-Codex silent-close" family as #1 and can be folded in.

**5. Mobile modal ergonomics for New Session / Diagnostics at 390×844 · verification, weakest.** Voice, unattended, and queue (via recovery) have mobile evidence; New Session and Diagnostics full-modal mobile ergonomics evidence is thin. *Not first* because there is no named mechanism/defect — recommending it risks an "audit everything" survey rather than a mechanism-backed fix, which the roadmap explicitly warns against.

---

## Parked user decisions (not agent-actionable now)

- **Real Codex / Claude inference parity** — blocked by credentials/gateway (Codex MCP auth; CC gateway 503). Distinct from #1/#2, which are UI-truth projections testable synthetically. Keep as an explicit release boundary.
- **Upload scope expansion** — multi-file / drag-drop / paste-to-attach / capture. Single-file paperclip is the certified semantic; expansion needs a product decision.
- **Monaco rich editor** — provision the assets and certify rich edit/diff, or retire the affordance. Plain-textarea editing is the certified baseline; the fail-loud Monaco path is truthful but not a real capability.

---

## What NOT to do next, and why

- **Do not touch the log-only stale-`interrupted_idle` variant** — executor `83f22848` owns it; duplicating collides and wastes the run. (#1's scope is disjoint: transcript projection vs. runtime busy/idle.)
- **Do not re-open certified tranches** — attachment badge (`72d6eef` reconcile makes it server-authoritative; `projectSelectedAttachmentIndicator` recomputes from server `pending_attachment`, so no stale-after-reload), mobile composer 44×44 (`fb42cfd`), Codex no-answer (claim4), idle/resume, git workbench (`4cf7e3c`), plain editor (`ef83bda`), upload (`e1bacb3`). No new evidence contradicts them.
- **Do not chase app.js extraction/refactor** — negative evidence: the sidebar/swipe extraction after `c3693df` was discarded for following "next extractable cluster" instead of a proven defect. Extraction is an intervention, not the roadmap.
- **Do not try to prove real Codex/CC inference in the cert container** — credential/gateway-blocked; a projection fix (#1) does not need it.
- **Do not add sandbox/verification-convenience UI** — precedent: the Pi `-ne` checkbox was reverted for polluting the product surface to satisfy a broken test container.