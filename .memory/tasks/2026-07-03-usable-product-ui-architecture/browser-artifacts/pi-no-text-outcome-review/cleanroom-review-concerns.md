Clean-room adversarial review complete. Verdict: **CONCERNS**.

## What the fix gets right (survived falsification)

The core mechanism is sound and its integration is disciplined:

- **Primary target is genuinely terminal.** Across 1372 real Pi logs, every no-text `stop` row (7: 6 thinking-only + 1 empty) is terminal — 6 followed by a user message, 1 at EOF, zero auto-continue. `end_turn` never occurs in practice. So classifying `stop`/`end_turn` no-text rows as no-response + idle is correct, and it fixes the real busy-forever defect.
- **Projection preserves cursors (Q2).** The Pi no-response event is a first-class event from `chat_event_from_log_row`, positioned at `record.start` by `_with_chat_position`, so it flows unchanged through tail/history (`_extract_positioned_chat_events`), live batch, search (`transcript_search` reuses the same extractor+dedupe+inject), and export. `_inject_no_response_events` correctly needs no Pi change — Pi has no separate close event (the assistant row *is* the boundary), so projecting from the row is the right mechanism and avoids double injection.
- **Authority coverage complete, no third state (Q3).** The predicate reached all six authorities — broker+sessiond shared reducer (`_apply_rollout_obj_to_state`, both routes via `broker_log_watcher._apply_log_objects_to_state`), `_compute_idle_from_log`, `_last_chat_role_ts_from_tail`, `pi_current_turn_state_before`, `_sidebar_conversation_ts`, `message_keeps_turn_busy`. It maps to existing `turn_open/busy=False` + existing `message_class:"error"`.
- **Imports clean (Q5).** `rollout_chat_events` imports `agent_backend` at module top (line 8); `agent_backend` breaks the cycle with a function-local lazy `from .rollout_chat_events import _build_no_response_event` (agent_backend.py:655). All modules import; no new cycle. Reusing `_build_no_response_event` avoids duplicating the event shape.
- **Commit hygiene clean (Q6).** Functional commit `32d914b` contains only `codoxear/*.py` + `tests/*.py` (no memory/evidence leak); defect-proof and fixed-proof are separate. Working tree clean, nothing staged. The Codex-named test file received Pi cases for the shared mechanism — in-scope.

## Primary finding — the predicate over-reaches on `length` (Q1)

The predicate is a **denylist**: any non-empty `stopReason` not in `{toolUse, error, aborted}` with no visible text/tool is "terminal no-response." This sweeps in `length`, which is a token-truncation, not a clean stop — **Pi auto-continues after a length truncation via context compaction.**

Real-log evidence (read-only scan of `~/.pi/agent/sessions`, 1372 logs): of 16 `length`+thinking-only/empty rows, **2 are followed (after `compaction`/`custom_message`, with no user message) by a continuing assistant `toolUse`+text row** — mid-turn auto-continue. Both real cases have the identical shape `toolResult → length+thinking → compaction → assistant(toolUse+text+toolCall)`.

I demonstrated the resulting regressions on the **shipped code** with that exact row shape:
- `pi_assistant_is_terminal_no_visible_response` → `True`
- `message_keeps_turn_busy` → `False` (pre-fix it returned `True`, because `thinking_count>0` — so this is a **regression**, not the fixed defect)
- `chat_event_from_log_row` → projects a **fabricated** `{message_class:"error", text:"The backend completed this turn without producing a response."}` immediately before the turn's real continuation
- shared reducer: `busy=True → False` at the length row, stays idle through compaction, recovers to busy only at the continuation → **transient false idle**
- `_compute_idle_from_log` (log ending at the length row) → `True` → a poll landing in the compaction window reports **idle while the agent is still working**

This contradicts the truthfulness invariant *in the fix's own domain*: it renders a "no answer" for a turn that did not terminate and did produce an answer. The busy/idle close is immediate (log-driven `_close_turn_state`, not debounced), so the false idle is observable during the compaction window (self-heals at the next activity row).

Secondary: the predicate's `pi_assistant_error_text(obj) or pi_assistant_is_aborted_turn(obj)` check is dead (those stopReasons are already excluded above) — cosmetic. And the denylist is forward-fragile: any future nonterminal `stopReason` defaults to "terminal no-response."

## Evidence gap that bears on acceptance (Q4)

The fixed proof is strong for `stop`/`end_turn` (Docker synthetic sessions, real server API, real Chrome DOM, reload persistence, screenshots; I independently re-ran the focused suite → `170 passed`). But **no `length` scenario appears anywhere** — not in synthetic sessions, the discriminator table, browser/API, or the defect scout (which never mentions `length`/`compaction`). The only `length` coverage, `tests/test_pi_message_source.py:76`, *asserts* `length`→terminal=`True`, enshrining the regression rather than catching it. So the evidence supports "stop/end_turn no-text render truthfully," not the broader "all Pi terminal no-text outcomes render truthfully."

## Why CONCERNS, not BLOCKERS

Net-positive and no hard-constraint violation: the defect it fixes (busy-forever for `stop`/`end_turn` no-text) is permanent and common; the regression it adds is rare (2/1372 sessions; `length` is 0.08% of terminal rows), transient/self-healing for busy/idle, and still shows the real answer right after the spurious row. Core mechanism, cursor semantics, authority coverage, imports, and commit hygiene all hold. Boundary of what I verified: I confirmed the false idle exists at the reducer and log-scan level; I did **not** trace whether the unattended/queue system acts on a transient idle — that is the path by which this could escalate from cosmetic to harmful (a queued prompt dispatched mid-compaction).

## Required follow-up

1. Treat `length` as non-terminal. Prefer switching the predicate to an **allowlist** (`stopReason in {"stop","end_turn"}`) over the current denylist — this fixes `length` and is forward-safe. (Alternatively add `length` to the exclusion set.)
2. Add a `length`+thinking-only-then-`compaction`-then-continuation test asserting the turn **stays busy** and projects **no** no-response row; update `tests/test_pi_message_source.py:76`.
3. Cosmetic: drop the redundant error/aborted checks inside the predicate.