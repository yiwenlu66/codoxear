## PASS

The transcript search synthetic-outcome defect is fixed. Search now uses the same positioned no-response semantics as tail/history/live.

### Changed files & exact mechanism

**`codoxear/transcript_search.py`** — `iter_positioned_chat_events_forward()` previously called `_single_chat_event()` per record and never invoked the no-response injector, so synthetic rows existed in tail/history/live but not in search.

The rewrite:
1. Collects bounded forward records (preserving `max_line_bytes` oversized-line skipping via `on_oversized_skip`, and an optional `before_byte` cutoff so bounded scanning is preserved when a window is requested).
2. Builds positioned events with the **existing try/except robustness** against malformed payloads — this matters because `_extract_positioned_chat_events` raises `ValueError` on malformed payloads (`{"payload": "notdict"}`), which the search path must tolerate (`test_streaming_search_skips_malformed_dict_records_without_stopping`).
3. Reuses the shared **`_dedupe_assistant_chat_events`** + **`_inject_no_response_events`** from `rollout_log` directly — no no-response logic is duplicated.
4. Attaches `_after_byte` via a `record.start → record.end` map. Regular events map to their own record end; injected no-response rows carry `_before_byte = close_byte` so they resolve to the **closing record's end**, which is exactly the byte window that `_read_chat_history_page` re-injects the synthetic row into.

`search_chat_log_bounded` now forwards `before_byte` to the iterator for bounded reading.

**`tests/test_transcript_export.py`** — added 6 tests + a `_session_cc` helper.

### How the defect failed before
- Codex no-answer turn (`event_msg/user_message` → `event_msg/task_complete`, no assistant): `/messages/search?q=backend completed this turn...` returned `match_count=0`.
- CC no-answer turn (`user` → `system/turn_duration`, no assistant): same, `match_count=0`.
Root cause: search's event stream never ran `_inject_no_response_events`, so the synthetic `_NO_RESPONSE_TEXT` row had no event to match and no cursor.

### Validation
- `python3 -m pytest -q tests/test_transcript_export.py tests/test_codex_no_response_projection.py tests/test_cc_no_response_projection.py tests/test_message_routes.py` → **84 passed** (78 original + 6 new; the narrow bounded test makes the file total 30).
- `python3 -m pytest -q tests/` → **1724 passed, 132 subtests passed**.
- `python3 /tmp/scout_transcript_outcomes.py` → **0 DEFECT** (both search rows flipped DEFECT→PASS; all 8 probes PASS).

Docker/API proof was not run: the direct route-handler tests exercise the identical code path (`handle_messages_search` → `search_chat_log_bounded` → `iter_positioned_chat_events_forward` → `_inject_no_response_events`, plus `_attach_search_load_cursors` + `decode_message_cursor` + `_read_chat_history_page`), so end-to-end HTTP would only re-test the socket layer, not the fixed logic. The scout probe provides independent route-level confirmation.

### Residual risks
- `iter_positioned_chat_events_forward` now buffers forward records (within the `before_byte` window) instead of streaming lazily. For a `before_byte`-bounded search this still stops at the window; for an unbounded full-file search it materializes record dicts for the whole log. This matches how tail/history already buffer their windows and is a correctness-for-parity tradeoff, not a behavior change to any flagged search semantic.

`git status --short` shows only the two modified files; nothing staged.
