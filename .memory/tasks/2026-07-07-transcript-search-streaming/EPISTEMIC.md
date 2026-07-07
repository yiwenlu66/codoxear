# Transcript search streaming epistemic model

## Phenomenon
Transcript search is a user-facing recovery/orientation surface for long sessions, and it is now the recommended alternative when full conversation copy exceeds the export cap. The current search implementation preserves transcript truth but pays for it by materializing the whole log before route-level limits can take effect.

## Current mechanism
`iter_positioned_chat_events_forward()` reads all bounded JSONL records, extracts all chat events, then runs assistant dedupe and synthetic no-response injection over the full batch. `search_chat_log_bounded()` only sees events after this batch phase, so `count_max=1000` in the browser's all-history count request bounds the reported match count but not the parsing work or memory footprint.

## Working hypothesis
The batch phase is not intrinsically required. Adjacent assistant dedupe is stream-local state. Synthetic no-response injection can be decided at close rows with open-turn state plus whether a visible assistant event occurred since the user row. Therefore a streaming positioned event iterator should preserve projection semantics while allowing count-limited first-order searches to stop early and latest-order searches to keep only bounded match memory.

## Live risks
- Same-row close/error rows must mark the turn answered before injecting no-response.
- `_after_byte` must remain usable for `load_cursor` and history windows.
- `before` history boundaries must not leak matches at or beyond the cursor.
- Oversized skipped lines still need `match_count_truncated` signaling when reached.
- CC pending-tool state must remain threaded through extraction.

## Current claim
This is a bounded product-performance/reliability slice, not a UI redesign: keep the same search API and user surface, but make the underlying mechanism honor route bounds earlier.
