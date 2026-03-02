# Session: 2026-03-02 Gemini Thinking Idle False Positive

## Focus
Fix Gemini sessions incorrectly showing `idle` while the model is still in a long thinking phase.

## Requests
- Gemini long-thinking periods should stay `busy`.

## Actions Taken
- Updated Gemini rollout mapping so `_gemini_turn_end` is only emitted for completion rows (for example rows with assistant text), not thinking-only/tool-only rows.
- Kept existing broker/log idle logic, but changed input semantics so thinking updates no longer look like turn-end markers.
- Added regression coverage:
  - Gemini thinking-only row does not emit turn-end in CLI mapping.
  - Broker does not close busy state on thinking-only assistant updates.
  - Server chat flags do not mark `turn_end` for thinking-only assistant rows.
  - Idle heuristic returns busy for thinking-only Gemini updates.
- Updated feature docs to reflect the corrected Gemini turn-end behavior.

## Outcomes
- Gemini sessions remain `busy` during long reasoning/tool phases instead of flipping to `idle` early.
- Turn-end and queue-drain semantics remain intact for actual completion rows.

## Tests
- `python3 -m unittest tests.test_cli_support tests.test_broker_busy_state tests.test_idle_heuristics tests.test_server_chat_flags`
- `python3 -m unittest discover -s tests`
