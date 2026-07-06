# Transcript search synthetic-outcome defect

Verdict: DEFECT.

The transcript invariant says search, older-history pagination, live polling, and rehydration must preserve the same transcript messages. Tail/history/live/re-read preserve synthetic no-response rows; search does not.

Observed route-level behavior:
- Codex no-response turn (`user_message` + `task_complete` with no assistant): history pagination emits the synthetic no-response row, but `/messages/search?q=backend completed this turn...` returns `match_count=0`.
- Claude Code no-response turn (`user` + `system/subtype:turn_duration` with no assistant): tail emits the synthetic no-response row, but search returns `match_count=0`.
- Claude Code terminal `system/api_error` is searchable because it is projected directly from a log row by `_single_chat_event`, not synthesized later.
- Ordinary user prompt search still works.

Root mechanism:
`codoxear/transcript_search.py::iter_positioned_chat_events_forward()` calls `_single_chat_event()` per row and never calls the no-response injector. Tail/history/live call `_extract_positioned_chat_events()`, which runs `_inject_no_response_events()`. The synthetic row therefore exists in rendered transcript surfaces but not in the search event stream, so it has no search match or signed cursor.

Minimal fix target:
Make transcript search build its event stream through the same positioned no-response-injecting semantics as the other transcript readers while preserving `_before_byte`/`_after_byte` for signed `history_cursor` and `load_cursor`. Add route-level tests for Codex and Claude Code no-response text search and an answered-turn no-false-positive guard.

Artifacts:
- `scout_transcript_outcomes.py` and `.out`: route-level probe and output.
- `scout-report.md`: scout report with mechanism and acceptance JSON.
- `scout_debug_history*.py`: history boundary probes.
