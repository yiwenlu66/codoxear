# Pi RPC Streaming Message Rendering Design

**Date:** 2026-04-14

## Goal

Make live `pi-rpc` assistant replies render incrementally in the web conversation pane instead of appearing only after the session file catches up.

## Problem

Today `Pi` sessions have two different data paths:

- durable history comes from the Pi session file through `codoxear/pi_messages.py`
- live browser refreshes come from `/api/sessions/<id>/live`

However, the current live path still depends on `get_messages_page()`, which for Pi reads from `session_path`. That means the browser only sees assistant output once it is persisted into the session file. The broker already receives `message.delta` RPC events, but it only folds them into `output_tail` text and does not expose them as structured conversation events.

So the current behavior is semantically delayed:

- `pi-rpc` is already streaming
- the broker already sees the deltas
- the browser conversation view still renders as if replies are batch-complete

## Approved Direction

Implement streaming rendering for active `pi-rpc` sessions by extending the existing polling-based `/live` path.

The design keeps three invariants:

1. durable chat history still comes from the Pi session file
2. live in-progress assistant text comes from broker-held RPC events
3. the frontend renders at most one assistant bubble for the active streamed turn and replaces it cleanly when the durable assistant message lands

This is intentionally smaller than switching the whole app to SSE or WebSocket push.

## Scope

### In Scope

- active `Pi` sessions with `transport: "pi-rpc"`
- structured broker-side capture of `message.delta`, `turn.started`, and terminal turn completion events
- `/api/sessions/<id>/live` returning temporary streaming assistant events in addition to durable events
- frontend message-store merging for one in-progress assistant message per turn
- replacement of the temporary streamed assistant message by the durable persisted assistant message
- regression tests for broker capture, live payload shape, merge semantics, and deduplication

### Out of Scope

- replacing `/live` polling with SSE or WebSocket transport
- changing Codex backend message behavior
- rewriting Pi durable history normalization away from `codoxear/pi_messages.py`
- making historical Pi sessions stream after reconnect unless they are currently backed by a live `pi-rpc` broker
- redesigning non-message live UI interactions beyond keeping current `ui_state` behavior intact

## Design Principles

1. **Durable and live state stay separate**
   - The session file remains the source of truth for replayable history.
   - Broker memory is only the source of current in-progress rendering.

2. **One semantic assistant reply, not two**
   - During streaming there may be one temporary assistant event.
   - Once the durable assistant message appears, the temporary event is removed or replaced.

3. **Do not infer from text tails when structure is available**
   - `output_tail` is for diagnostics.
   - Streaming conversation rendering must use structured broker events.

4. **Keep the existing polling envelope**
   - Extend `/live` instead of introducing a new transport layer.
   - Minimize UI churn while fixing the semantics.

## Architecture

### 1. Broker keeps structured live message state

`codoxear/pi_broker.py` already drains RPC events. Today it:

- updates busy state
- tracks pending UI requests
- appends human-readable output into `output_tail`

It must additionally track structured streaming state for the active turn.

For the active assistant turn, the broker should maintain enough information to describe a temporary message event:

- `turn_id`
- `role` (`assistant`)
- accumulated streamed text
- whether the turn has started
- whether the turn has reached a terminal event
- a monotonic live-event sequence or offset suitable for polling deltas

This state should be updated directly from RPC events such as:

- `turn.started`
- `message.delta`
- `turn.completed`
- `turn.failed`
- `turn.aborted`

### 2. Broker exposes structured live events over the control socket

The broker socket contract needs a read-only command for live message events. The command should return only broker-held transient streaming state, not replayed file history.

Minimal response shape:

- current live offset
- zero or more live events since the caller's offset
- enough fields to correlate updates for the same streamed assistant turn

The important behavior is polling safety:

- repeated polls with no new broker events should be cheap
- repeated polls must not duplicate already-consumed live events
- the server must be able to reconstruct the current in-progress assistant message deterministically from the broker stream

### 3. Server merges durable page data with broker live data

`codoxear/server.py` currently builds `/live` with:

- `get_messages_page()` for conversation events
- `get_ui_state()` for pending Pi requests

For `transport == "pi-rpc"`, `_session_live_payload()` should additionally query the broker's live message stream and append a temporary assistant event when appropriate.

The merge contract is:

- durable events from the session file remain unchanged
- broker live events only describe the active not-yet-durable assistant reply
- if the durable history already contains the matching completed assistant message, do not also return the temporary streamed event

This keeps `/messages` historical semantics intact while making `/live` actually live.

### 4. Frontend message store merges streamed assistant updates

`web/src/domains/messages/store.ts` currently only supports:

- replace full event list
- append new events

It needs one more capability: upsert a temporary streamed assistant event keyed by turn identity rather than always appending a fresh row.

The frontend should treat streamed Pi assistant events as:

- one logical assistant bubble per active turn
- text that grows across polls
- replaced by the persisted assistant message when the durable event arrives

That means the store needs merge rules roughly like:

- if a streamed event with the same stream key already exists, update its text in place
- if a durable assistant event arrives for the same turn or same finalized text boundary, drop the temporary event
- keep ordinary historical events append-only

### 5. Conversation rendering stays mostly unchanged

`web/src/components/conversation/ConversationPane.tsx` already renders assistant rows from `MessageEvent` records.

The goal is to preserve that rendering path by making streamed assistant updates look like normal assistant events plus a small amount of metadata, for example:

- `role: "assistant"`
- `text: "current accumulated text"`
- `streaming: true`
- `stream_id` or `turn_id`
- optional `message_class: "narration" | "final_response"` once known

This keeps the UI change focused on store semantics rather than a second rendering pipeline.

## Event Model

### Durable events

Durable events continue to come from `codoxear/pi_messages.py` and the session file. They remain replayable and offset-based.

### Temporary streamed events

Temporary streamed events exist only while the broker knows about an in-progress assistant reply that has not yet been committed in durable history.

Recommended event shape:

- `role: "assistant"`
- `text: <accumulated text>`
- `streaming: true`
- `turn_id: <pi turn id>` when available
- `stream_id: <stable synthetic id for frontend merge>`
- `ts: <broker event timestamp or synthetic monotonic timestamp>`

This event is presentation state, not historical truth.

## Replacement Boundary

The replacement boundary is the critical semantic rule.

### While streaming

- `message.delta` extends the temporary assistant event text
- the conversation shows exactly one temporary assistant bubble for that turn

### When the turn reaches a terminal RPC event

- the temporary assistant event may remain visible with its completed text
- it is still temporary until durable history confirms the assistant message

### When the session file contains the assistant message

- the durable assistant event becomes authoritative
- the temporary streamed event is removed or replaced in-place
- the user sees one assistant reply, not a streamed copy plus a persisted copy

## Failure and Edge Cases

### No `turn_id`

If some RPC events lack `turn_id`, the broker may synthesize a stream key from the active turn slot. This should be limited to one active assistant stream at a time.

### Empty delta stream

If a turn starts and finishes without visible assistant text, no temporary assistant event needs to be rendered.

### Aborted turn

If the turn is aborted before any durable assistant message appears:

- drop the temporary streamed assistant event, or
- mark it as interrupted only if the existing UI already has a clear interrupted-message style

The preferred minimal behavior is to remove the temporary event and rely on normal interruption indicators elsewhere.

### File wins over broker memory

If broker state and durable file history disagree, the file wins for historical display. Broker state should only fill the gap before durable persistence.

## Testing Strategy

### Broker tests

Add tests around `codoxear/pi_broker.py` to verify:

- `message.delta` builds structured live assistant state
- repeated deltas for one turn update the same stream
- terminal turn events mark the stream complete
- read-only broker command returns stable incremental live payloads

### Server tests

Add tests around `codoxear/server.py` to verify:

- `/api/sessions/<id>/live` includes streamed assistant events for active `pi-rpc` sessions
- polling with offsets does not duplicate temporary streamed events
- once the durable assistant message is present, the streamed temporary event is not returned as a duplicate

### Frontend tests

Add tests around:

- `web/src/domains/messages/store.ts`
- `web/src/components/conversation/ConversationPane.test.tsx`

Verify:

- one assistant row grows across multiple streamed polls
- streamed rows do not append duplicates on every poll
- durable persisted assistant events replace temporary streamed rows cleanly

## Implementation Plan Preview

1. add structured broker-side live message capture for Pi RPC events
2. expose a broker socket command for incremental live message events
3. merge broker live events into `_session_live_payload()` for `pi-rpc` sessions
4. extend the frontend message store with streamed assistant upsert and replacement rules
5. add regression tests across broker, server, and frontend

## Open Questions Resolved

### Why not use `output_tail`?

Because `output_tail` is unstructured diagnostic text. It cannot safely express turn identity, replacement boundaries, or deduplicated assistant message semantics.

### Why not switch to SSE now?

Because the immediate user-visible defect is semantic, not transport-level. Polling can carry correct streaming state with far less disruption.

### Why keep file-backed history?

Because the session file is already the durable replay source used by the rest of the app. Replacing it would expand scope far beyond message streaming.
