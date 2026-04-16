# Web Polling Optimization Design

**Date:** 2026-04-10

## Goal

Reduce unnecessary frontend polling and request fan-out in the web UI while preserving near-real-time chat behavior.

The design should aggressively cut request volume now, and it should make a later move to SSE or WebSocket straightforward instead of forcing another semantic rewrite.

## User Intent

The user wants the current page's polling behavior reviewed and optimized as far as it reasonably can be optimized.

Approved preferences:

- optimize both frontend and backend, not frontend-only throttling
- maximize request reduction rather than preserving every current polling pattern
- keep only chat messages as strongly real-time by default
- preserve a future path to SSE or WebSocket
- plan this work assuming the `file/list` redesign documented in `.pi/claude-plan-mode/plans/6bced13c-9da7-45f7-93ae-5d110f26cad7.md` has already landed

## Approved Direction

Use an aggressive polling consolidation design instead of a push-transport rewrite.

The approved high-level shape is:

- keep HTTP polling for now
- define a stable live-session data contract that later transports can reuse
- add a high-frequency `live` session endpoint for chat-critical data only
- add a low-frequency `workspace` endpoint for details panels only
- remove files from the global session refresh path entirely
- make polling state-driven based on active/busy/hidden visibility state

## Current State

Current polling behavior is distributed across several independent paths.

### Sessions list polling

`web/src/app/app-shell/useAppShellSessionEffects.ts` refreshes the sessions list:

- every 3 seconds while any session is busy
- every 15 seconds while no sessions are busy
- only while the page is visible

This part is already visibility-aware, but it is separate from the rest of the polling model.

### Active session polling

The same hook also treats the active session as a fixed 3-second loop.

On active-session selection it immediately performs:

- `messagesStoreApi.loadInitial(activeSessionId)`
- `sessionUiStoreApi.refresh(activeSessionId, { agentBackend })`

Then every 3 seconds it performs:

- `messagesStoreApi.poll(activeSessionId)`
- `sessionUiStoreApi.refresh(activeSessionId, { agentBackend })`

At the store level, `sessionUiStoreApi.refresh()` currently fetches multiple resources together:

- `ui_state` for Pi sessions
- `diagnostics`
- `queue`

This means the active session's high-frequency path still fans out into several requests that do not all need real-time freshness.

### Background busy session polling

For reply-sound behavior, the shell also primes and polls non-active busy sessions.

This currently uses message-store polling for every background busy session on a 3-second interval after initial priming.

### Notifications feed polling

`web/src/app/app-shell/useAppShellNotifications.ts` polls `/api/notifications/feed` every 5 seconds whenever desktop notifications or reply sounds are enabled.

This path is not currently tied to page visibility.

### Files and workspace detail data

Per the separate file-list design, the system is moving away from eager file-list loading. The intended end state is:

- no file-list fetch during global session UI refresh
- file browsing owned locally by `FileViewerDialog`
- `/api/sessions/:id/file/list` used as a lazy directory listing endpoint

That future state is assumed by this design.

## Problem Statement

The current page performs too many independent polls for data with very different freshness requirements.

Observed structural issues:

- active-session polling mixes truly real-time data with low-value detail data
- multiple polling loops operate independently instead of sharing one state model
- active sessions remain on a fixed 3-second cadence even when idle
- hidden-page behavior is inconsistent across sessions, active conversation polling, and notifications polling
- background busy sessions use message polling only for minimal notification/reply-beep semantics, but that intent is not expressed as a separate contract
- current frontend store boundaries reflect endpoint history more than current UI semantics

The result is unnecessary request volume, avoidable request bursts, and a code shape that will be harder to migrate to SSE or WebSocket later.

## Design Principles

1. **Real-time belongs only to the chat-critical path**
   - Messages must stay responsive.
   - UI requests may travel with the same live contract.
   - Diagnostics, queue details, and files do not belong in the same high-frequency loop.

2. **Data semantics must be separated from transport**
   - The frontend should consume a stable live delta model.
   - HTTP polling is only the first transport implementation.

3. **Visibility should control polling**
   - If the page is hidden, default polling should pause.
   - Visibility restoration should trigger immediate catch-up refreshes.

4. **Background sessions should use minimal observation**
   - Background busy sessions only need enough data to detect completion and trigger notification/reply-sound flows.
   - They do not need the full active-session refresh path.

5. **File browsing is entirely on-demand**
   - File trees should not re-enter global polling through a side door.
   - The file browser owns its own lazy loading and caching lifecycle.

## Recommended Architecture

Split session-related data into three distinct domains.

### 1. Session list domain

Purpose:
- populate the sidebar and high-level session summaries

Route:
- continue using `GET /api/sessions`

Data examples:
- session title
- busy state
- last activity metadata
- queue length summaries
- backend metadata

Freshness:
- medium

### 2. Live session domain

Purpose:
- provide only the data needed to keep the current conversation responsive
- provide the minimum extra state needed for live UI request handling

New route:
- `GET /api/sessions/:id/live`

Data examples:
- message delta events
- offset or cursor
- busy state
- pending UI requests

Freshness:
- high

### 3. Workspace details domain

Purpose:
- populate the workspace/details surfaces when explicitly opened

New route:
- `GET /api/sessions/:id/workspace`

Data examples:
- diagnostics
- queue

Freshness:
- low

### Explicitly excluded from polling domains

Files are excluded from both live and workspace polling.

File data is owned by:
- `FileViewerDialog`
- lazy `/api/sessions/:id/file/list?path=...`
- file read and diff endpoints on demand

## API Design

## `GET /api/sessions/:id/live`

### Purpose

Replace the active session's current high-frequency fan-out across `/messages` and `/ui_state` with one live-response contract.

### Request

Query parameters:
- `offset` optional; the last consumed live message offset

The exact cursor name may later evolve to `cursor`, but `offset` is sufficient for the initial HTTP version.

### Response shape

```json
{
  "ok": true,
  "session_id": "sess-1",
  "offset": 128,
  "busy": true,
  "events": [
    { "role": "assistant", "message_class": "delta", "text": "hello" }
  ],
  "requests": [
    { "id": "ask-1", "method": "select", "question": "Pick one" }
  ]
}
```

### Semantics

- `events` follows the same normalized message-event model the frontend already consumes
- `offset` advances the live stream position for polling clients
- `busy` gives the frontend a single authoritative live-session busy bit
- `requests` contains the current pending UI requests for the session
- no diagnostics, queue payload, or file payload appears here

### Backend implementation direction

The server should reuse existing normalized readers where possible:

- message extraction from current `/messages` logic
- request extraction from current `/ui_state` logic
- busy determination from the same session-state logic already used for `/api/sessions`

The new route is a semantic aggregator, not a new source of truth.

## `GET /api/sessions/:id/workspace`

### Purpose

Provide low-frequency workspace detail data without coupling it to live chat polling.

### Response shape

```json
{
  "ok": true,
  "session_id": "sess-1",
  "diagnostics": {
    "status": "ok"
  },
  "queue": {
    "items": []
  }
}
```

### Semantics

- only workspace/detail data belongs here
- no live message deltas
- no UI request list unless implementation later decides the details surface needs a redundant copy
- no file listings

## Existing routes during migration

Keep these routes during the migration period:

- `GET /api/sessions/:id/messages`
- `GET /api/sessions/:id/ui_state`
- `GET /api/sessions/:id/diagnostics`
- `GET /api/sessions/:id/queue`

Reason:
- minimize rollout risk
- preserve compatibility for existing code paths and targeted regression checks
- allow phased frontend migration and easier debugging

The new frontend should prefer `live` and `workspace`.

## Polling State Machine

Polling behavior should be controlled by session role, busy state, workspace visibility, and page visibility.

### State: `active-busy`

Meaning:
- the currently selected session is busy

Allowed polling:
- `GET /api/sessions/:id/live`
- `GET /api/sessions`
- `GET /api/sessions/:id/workspace` only if workspace is open

Target intervals:
- `live`: 2000 ms
- `sessions`: 5000 ms
- `workspace` if open: 15000 ms

### State: `active-idle`

Meaning:
- the currently selected session is not busy

Allowed polling:
- `GET /api/sessions/:id/live`
- `GET /api/sessions`
- `GET /api/sessions/:id/workspace` only if workspace is open

Target intervals:
- `live`: 12000 ms
- `sessions`: 15000 ms
- `workspace` if open: 15000 ms

### State: `background-busy`

Meaning:
- a non-active session is busy

Allowed polling:
- minimal background live observation only for sessions that matter to reply-sound / completion detection
- `GET /api/sessions` still covers summary state for the sidebar

Target intervals:
- background completion detection: 5000 ms

Important restriction:
- do not run workspace or file-related refreshes for background sessions

### State: `background-idle`

Meaning:
- a non-active session is not busy

Allowed polling:
- no session-specific polling
- rely on `GET /api/sessions` summary refresh

### State: `hidden`

Meaning:
- document visibility is hidden

Allowed polling:
- default behavior is pause `live`, pause `workspace`, and pause background busy polling
- optionally keep `sessions` paused as well and refresh immediately when visible again
- notification-feed polling should also pause and catch up on resume

This is the recommended default because the user prioritized request reduction over background freshness.

## Immediate Refresh Triggers

Do not wait for the next timer tick when a user action or lifecycle event already tells the app that data is stale.

Immediate refresh events:

- switching the active session
- sending a message
- interrupting a session
- submitting a UI request response
- page visibility changing from hidden to visible
- opening the workspace panel
- opening the file viewer, for file-viewer-local data only

Required refreshes by event:

- switch active session -> immediate `live` load
- send message -> immediate `live` refresh
- interrupt -> immediate `sessions` + active `live` + `workspace` if open
- submit UI request -> immediate `live`, plus `workspace` if open
- resume visibility -> immediate `sessions` + active `live` + `workspace` if open
- open workspace -> immediate `workspace`

## Frontend State Design

## Live store

Introduce a dedicated live-session store instead of overloading the existing messages store with unrelated responsibilities.

Recommended location:
- `web/src/domains/live-session/store.ts`

Responsibilities:
- load and poll `live` payloads
- merge event deltas into conversation state
- track per-session live offsets or cursors
- expose current pending UI requests from the live contract
- expose per-session busy state if needed by polling logic

Reasoning:
- message history pagination and live delta polling are related but not identical responsibilities
- a dedicated store makes the later switch from polling to SSE/WebSocket cleaner

## Messages store

`web/src/domains/messages/store.ts` should keep ownership of conversation history concerns such as:
- initial history loads
- offset bookkeeping used by the rendered message timeline
- older-history pagination

A thin integration layer may bridge live deltas into the message timeline, but the system should not force the history store to own all transport concerns.

## Workspace store

`web/src/domains/session-ui/store.ts` should narrow to workspace-style detail data.

Primary responsibilities after this change:
- diagnostics
- queue
- loading state for explicit workspace refreshes

It should no longer define the active session's real-time contract.

## File viewer state

File viewer state remains local to `web/src/components/workspace/FileViewerDialog.tsx`.

Expected long-term behavior, aligned with the separate file-list design:
- fetch root directory listing on open
- fetch child directory listings lazily on expand
- cache directory children locally while the dialog is open
- read file contents and diffs on explicit file selection

## Frontend Control Flow

`web/src/app/app-shell/useAppShellSessionEffects.ts` should become the single place that decides:
- which polling lanes are active
- which intervals apply
- which immediate refreshes fire on UI events and visibility changes

The implementation should replace multiple unrelated `setInterval` loops with a state-driven scheduler model.

The exact internal abstraction is flexible, but the behavior should follow the polling state machine in this document.

## Notification Feed Behavior

`web/src/app/app-shell/useAppShellNotifications.ts` should become visibility-aware.

Recommended behavior:
- pause `/api/notifications/feed` polling while hidden
- immediately catch up once the page becomes visible again
- keep reply-sound and desktop-notification dedupe semantics unchanged

Longer term, if live transport later subsumes enough of the current notification needs, the feed poll can be reduced further.

## Migration Path to SSE or WebSocket

This design intentionally separates live-session semantics from transport so the future migration is mechanical instead of architectural.

### Transport-independent live contract

The frontend should consume one conceptual payload type, for example:
- `LiveSessionDelta`

It should not care whether this arrives through:
- HTTP polling
- SSE events
- WebSocket frames

### HTTP now

Initial transport:
- `pollLive(sessionId, offset)`

### SSE later

Future transport:
- `subscribeLive(sessionId, cursor, onDelta)`

The payload shape should remain the same or nearly the same.

### WebSocket later

Future transport:
- a socket message carrying the same live delta structure

This means the store merge logic should survive transport migration with minimal or no semantic change.

## Interaction with File-List Redesign

This design assumes the already-approved end state of the hierarchical file list plan.

Consequences:
- `files` is not part of session live state
- `files` is not part of workspace polling state
- `sessionUiStore.refresh()` should not trigger `/file/list`
- the file browser remains an on-demand local subsystem

This is required to avoid reintroducing a heavy file enumeration path into the polling model.

## Testing Strategy

### Backend tests

Add coverage for:
- `GET /api/sessions/:id/live` response shape
- `live` offset forwarding and delta behavior
- `live` inclusion of pending UI requests
- `live` exclusion of diagnostics, queue, and files
- `GET /api/sessions/:id/workspace` response shape
- `workspace` inclusion of diagnostics and queue only

### Frontend unit tests

Update or add coverage for:
- active busy polling intervals
- active idle polling intervals
- background busy polling intervals
- no background idle per-session polling
- hidden-page pause behavior
- visible-again immediate catch-up behavior
- workspace polling only while workspace is open
- notification-feed pause/resume on visibility changes
- file viewer no longer depending on global session refresh state

### Integration-oriented behavior checks

Verify:
- active chat remains responsive while generating
- idle sessions stop generating high request volume
- opening workspace still shows diagnostics and queue when needed
- file viewing still works through its own local fetches
- background busy completion still triggers reply sound / notification behavior

## Constraints

- keep chat messages near real-time
- do not reattach file listing to the global polling path
- do not require an SSE or WebSocket rollout in this change
- preserve existing route compatibility during migration where practical
- keep the design incremental enough that frontend and backend can land in stages

## Out of Scope

- immediate replacement of polling with SSE or WebSocket
- redesigning notification product semantics
- redesigning workspace UI visuals
- changing file read authorization policy
- changing the approved hierarchical `/file/list` plan
- broad store architecture cleanup unrelated to live/workspace polling semantics

## Risks and Mitigations

### Risk: route duplication during migration

Mitigation:
- keep old routes temporarily
- migrate one frontend path at a time
- add targeted tests for both new aggregate routes

### Risk: live store and message history store can drift semantically

Mitigation:
- keep one normalized message event shape
- treat live data as another source of the same event model rather than inventing a second message model

### Risk: hidden-page pausing can delay background awareness

Mitigation:
- accept this trade-off intentionally because request reduction is the explicit priority
- trigger immediate catch-up when visibility resumes

### Risk: workspace details may feel stale

Mitigation:
- refresh immediately when opening the workspace
- allow a low-frequency refresh while open
- keep chat responsiveness decoupled from workspace freshness

## Acceptance Criteria

This change is successful if:

- the default active-session high-frequency path uses one `live` request instead of several separate requests
- diagnostics and queue are no longer part of the active session's high-frequency loop
- files are fully removed from global polling behavior
- idle sessions poll substantially less often than busy sessions
- hidden pages pause default polling and recover immediately on visibility restore
- background busy sessions use only minimal completion-detection polling
- the design leaves a clear path to swapping polling transport for SSE or WebSocket later
