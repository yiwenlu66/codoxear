# Session: 2026-03-02 Gemini Busy Stop Detection

## Focus
Fix Gemini sessions that stayed in `busy=true` after the assistant had already finished replying.

## Requests
- Gemini replies should clear busy state after completion.

## Actions Taken
- Reproduced and traced the stuck path to Gemini log offset handling in the broker:
  - broker initialized watch offsets from physical file `st_size`,
  - Gemini polling uses synthesized JSONL offsets, so physical offsets could skip Gemini updates.
- Added Gemini-aware rollout tail offset selection in broker registration/switch paths:
  - use synthesized Gemini tail offset for `session-*.json`,
  - keep physical `st_size` for Codex/Claude JSONL logs.
- Hardened Gemini turn-end handling:
  - broker now closes turns on `_gemini_turn_end` even when assistant rows have no text (thinking/tool-only rows),
  - rollout parser marks `turn_end` on `_gemini_turn_end` independent of text extraction,
  - idle heuristic closes open turns on `_gemini_turn_end` to avoid `busy`/`idle` disagreement.
- Normalized Gemini offset reads in `util`:
  - oversized or mismatched offsets are now normalized to the synthesized Gemini tail offset.
- Added regression tests for all three paths above.

## Outcomes
- Gemini sessions no longer remain stuck in busy state after reply completion.
- Busy/idle convergence now works for Gemini rows that only contain thinking/tool metadata.
- Offset handling is resilient when callers provide physical offsets for synthesized Gemini logs.

## Tests
- `python3 -m unittest tests.test_broker_busy_state`
- `python3 -m unittest tests.test_idle_heuristics`
- `python3 -m unittest tests.test_server_chat_flags`
- `python3 -m unittest tests.test_util_gemini_offset`
- `python3 -m unittest discover -s tests`
