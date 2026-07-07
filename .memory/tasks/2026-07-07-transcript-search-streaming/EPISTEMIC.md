# Transcript search streaming epistemic model

## Phenomenon
Transcript search is the user-facing recovery/orientation surface for long sessions and the recommended alternative when full conversation copy exceeds the export cap. The previous implementation preserved transcript truth but required full-log record/event materialization before `count_max` could stop first-order search work.

## Accepted mechanism
`iter_positioned_chat_events_forward()` now streams bounded JSONL records and emits positioned chat events incrementally. The former batch transforms are represented as constant state:

- Adjacent assistant dedupe tracks the previous assistant dedupe key and resets on user rows, matching the old `_dedupe_assistant_chat_events()` skip predicate.
- Synthetic no-response injection tracks the current open user byte plus whether a deduped visible assistant event has been emitted since that user. Close rows are evaluated after their own visible event extraction, so terminal error/answer rows suppress generic no-response duplicates.
- Regular events carry `_before_byte=record.start` and `_after_byte=record.end`; injected no-response rows use the close record's start/end, preserving `history_cursor`/`load_cursor` behavior.
- `iter_jsonl_records_forward_bounded()` accepts `before_byte` and stops before reading records starting at or beyond the boundary.

`search_chat_log_bounded()` keeps the public API shape. For `order=first` with `count_limit`, the loop breaks once the bounded count is reached, so the lazy record iterator is not consumed to the end. For `order=latest`, the search still scans the bounded region but no longer stores whole-log record/event lists.

## Evidence
- Local focused validation passed: `python3 -m pytest -q tests/test_transcript_export.py tests/test_message_routes.py` (`48 passed`). Full local validation passed: `1827 passed, 134 subtests`.
- `test_count_limited_first_order_search_stops_record_stream` monkeypatches the record iterator and proves `count_limit=5` consumes at most six matching records from a 1000-record stream, which would fail under the previous batch materialization mechanism.
- Existing route tests for Codex/Claude synthetic no-response search, cursors, oversized-line truncation, recovered-session search merging, and limit parsing passed in the focused set.
- Docker focused validation on port `19470` passed after fixing a test-portability import; Docker smoke on port `19471` proved isolated server/login/app-dir boundaries.
- Desktop and mobile browser proof on container `codoxear-search-streaming-19472` used a runtime-only 3002-row/539KiB transcript log. The real chat search UI reported `needle` as a truncated all-history count (`1000+`) and loaded an older `EARLY_ONLY_TARGET` row through the real Next/cursor path with no horizontal overflow.
- Clean-room review `8b3d14d8` accepted the slice. It verified logical equivalence for dedupe and no-response injection, `_after_byte` preservation, early-stop generator mechanics, before-boundary stopping, oversized-line signaling, CC pending-tool state, recovered-session search merging, and artifact hygiene.

## Current claim
The transcript-search streaming slice is accepted: first-order count-limited search can stop consuming records before the end of a large log, latest-order search avoids whole-log materialization, and the normalized transcript/search projection contract is preserved.

## Boundaries
Latest-order search still scans the requested bounded region because returning the latest match requires seeing later records. Pi no-response injection remains absent as before; this slice did not add new Pi no-response semantics. Browser proof demonstrates user-visible search behavior, while the iterator-consumption test is the discriminating evidence for internal early-stop work reduction.
