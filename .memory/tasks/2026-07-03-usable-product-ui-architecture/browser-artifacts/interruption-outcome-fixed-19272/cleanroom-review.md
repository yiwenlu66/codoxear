I have completed a thorough mechanistic audit. Let me compile the final report.

## Audit Result: PASS (no blockers) — 2 non-blocking concerns

The fix is architecturally correct and well-isolated. It changes exactly one thing — the shared transcript projector `_single_chat_event` → `chat_event_from_log_row` — and every other subsystem (idle, delivery, sidebar, notifications) reads raw log rows independently and is provably untouched.

### Q1 — Fixes across tail/history/live/search/export, no UI-only state, no duplicate parser: CONFIRMED
All five read paths funnel through the single projector `_single_chat_event` (`rollout_chat_events.py:100`):
- tail/history/export → `_extract_positioned_chat_events` (`rollout_log.py:150`)
- live → `_extract_positioned_chat_events` (`message_routes.py:389`)
- search → `iter_positioned_chat_events_forward` → `_single_chat_event` (`transcript_search.py:159`)
- flags/live-delta → `_extract_chat_events` → `_single_chat_event` (`rollout_chat_batch.py:36`)

The commit touches only `agent_backend.py` + `rollout_events.py` (no `static/app.js` change), so the row reuses existing `message_class:"error"` rendering — no UI-only state, no second parser. I verified tail reverse-pagination parity directly (the abort row now correctly counts as one page event).

### Q2 — Preserves partial Pi text, avoids `_NO_RESPONSE_TEXT`: CONFIRMED
`agent_backend.py:626` passes `partial_text=pi_assistant_text(row)`; `_build_interrupted_event` (`rollout_events.py:73`) appends it under a label. Pi aborts never reach no-response injection (that path only fires on Codex `task_complete`/`turn_complete` and CC `system` closes). For Codex, the abort event's `_before_byte` is a visible assistant byte, so the `answered` check (`rollout_chat_events.py:317`) suppresses no-response.

### Q3 — `message_class:"error"` does NOT harmfully alter turn_end / busy-idle / delivery / sidebar / no-response: CONFIRMED
- **turn_end**: Pi branch uses `elif` so abort takes precedence (`rollout_chat_batch.py:102-105`, turn_end untouched); Codex `event_msg` branch never consults event class (`:121-122`). Pinned by `test_server_chat_flags.py:33` (Codex) and the modified Pi flags test.
- **busy/idle**: computed from raw rows in `rollout_idle.py:178-181,270-273` — abort→idle=True, independent of the projection. Pinned by `test_turn_aborted_is_idle` / `test_pi_aborted_message_is_idle`.
- **delivery**: `rollout_delivery.py:46` skips aborts. Pinned by `test_pi_aborted_text_is_not_delivered`.
- **sidebar ts**: `_sidebar_conversation_ts` returns None for aborts (`rollout_chat_events.py:48`), unchanged.
- **notifications**: `voice_runtime.py:31` only attaches to `final_response`; interruption is `error`, so untouched.

### Q4 — Cursor/search semantics correct: CONFIRMED
Direct projection yields `_before_byte`=abort-row start, `_after_byte`=abort-row end (`transcript_search.py:172-176`); `load_cursor` rehydrates the row from history. Split windows work because a direct projection needs no prior context, and `_record_produces_visible_assistant` (`rollout_log.py:185`) now recognizes the abort row, correctly suppressing phantom no-response in the split case. A later `task_complete` produces no double no-response (tested). Edge case: `task_complete` carrying `last_agent_message` after `turn_aborted` yields both an interruption and a final_response row — contradictory backend state, unlikely, non-harmful.

### Q5 — Missing tests
Core invariants are well-covered. Two genuine residual gaps below.

### CONCERN 1 (low, latent coupling — currently harmless)
`session_log_runtime.py:78-85` — the stale-`interrupted_idle` guard clears the override when a post-baseline chunk contains a user/assistant chat event. After this fix, an abort row now *produces* such a chat event (`_analyze_log_chunk` → `_extract_chat_events` → interruption event), so an abort landing after the interrupt baseline can clear the override earlier than before. **This is harmless in current code**: the same abort row forces `_compute_idle_from_log`→True, so `busy = not(log_idle or override)` stays False regardless; and `interrupted_idle_suppressed` resets on user send (`session_input.py:78`). It becomes a live defect only if the abort→idle mapping in `rollout_idle.py` is ever changed without revisiting this guard. Suggested discriminator: a `test_stale_interrupted_idle` case appending an explicit `turn_aborted`/`stopReason:aborted` row post-baseline and asserting the listing still projects `busy=False`.

### CONCERN 2 (low, test gap)
No test asserts (a) the CONCERN-1 guard interaction, or (b) that interruption events do not emit a voice-push notification for Codex (Pi delivery-suppression is tested; Codex `turn_aborted` is trivially non-delivered). Both are defensive; the mechanisms are correct.

### Validation
- `pytest tests/` → 1736 passed, 132 subtests passed.
- Targeted files → 97 passed.
- Scratch harness (deleted) confirmed: single/split-window no-response suppression, turn_end=False for lone abort, idle=True, sidebar ts None, search cursor rehydration, tail reverse parity, dedupe of consecutive identical aborts.
- Working tree clean; nothing staged. (One untracked dir `browser-artifacts/…` belongs to the browser-evidence worker, not this audit.)
