# Pi Web Ask-User Interaction Design

**Date:** 2026-04-04

## Goal

Extend the Pi-backed web session UI so `ask_user` interactions render as full cards in the browser and can be answered directly from the web UI, while establishing a reusable bridge for other Pi RPC dialog-style interactions.

## Problem

Pi already records `ask_user` activity in session logs:

- the `toolCall` contains structured request data such as `context`, `question`, `options`, `allowFreeform`, and related flags
- the matching `toolResult` contains the outcome such as `answer`, `cancelled`, and `wasCustom`

The current web stack does not preserve that interaction model end to end:

- `codoxear/pi_messages.py` only exposes generic tool / tool-result events, not an `ask_user`-aware event shape
- `codoxear/static/app.js` can render generic tool cards, but not a full interaction card with request/response semantics
- `codoxear/pi_broker.py` only bridges normal prompt sending and interrupt keys, so the web UI cannot answer a pending Pi interaction without faking it as a new chat prompt

That leads to two distinct gaps:

1. historical `ask_user` interactions are visible only as low-level tool plumbing
2. live `ask_user` requests cannot be answered correctly from the browser

## Approved Direction

Build a general Pi interaction bridge for RPC dialog-style UI requests, and use `ask_user` as the first fully rendered interaction type.

The design intentionally separates two data sources:

- **Historical interaction history** comes from Pi session logs and is normalized into durable UI events.
- **Live pending interaction state** comes from Pi RPC `extension_ui_request` messages and is exposed through a dedicated broker/server bridge.

This keeps log playback and live interactivity semantically correct instead of conflating them into normal message sending.

## Scope

### In Scope

- full `ask_user` history rendering in web sessions
- answering a pending `ask_user` request from the browser
- reusable server/broker protocol support for Pi RPC dialog methods:
  - `select`
  - `confirm`
  - `input`
  - `editor`
- matching live pending requests to historical tool cards when possible
- explicit failure and stale-state handling for pending interactions
- regression coverage for normalization, bridge behavior, and UI rendering

### Out of Scope

- changing Codex session behavior
- implementing arbitrary Pi extension UI widgets beyond dialog-style request/response methods
- building a global interaction inbox across sessions
- rewriting Pi log storage or Pi RPC protocol semantics

## User Experience

### 1. Historical Ask-User Cards

When a Pi session contains a completed `ask_user` interaction in the log, the chat view should render a dedicated card instead of a generic tool/result pair.

The card should display:

- `Ask user` badge
- optional `context`
- main `question`
- selectable options as read-only chips or rows
- final answer
- whether the answer was cancelled
- whether the answer was a custom freeform answer
- timestamp metadata

Historical cards are always read-only.

### 2. Live Pending Interaction Cards

When Pi is currently waiting for user input through the RPC UI protocol, the browser should show the corresponding card as interactive.

Supported live states:

- `pending-select`: user must choose one of the provided options
- `pending-input`: user may type a freeform answer
- `pending-confirm`: yes/no style confirmation
- `pending-editor`: multi-line text entry
- `submitting`: response is being sent; controls disabled
- `resolved-awaiting-log`: submission succeeded locally and is waiting for log confirmation
- `expired` / `stale`: request is no longer answerable
- `error`: submission failed and can be retried

### 3. Ask-User Interaction Rules

For `ask_user`, the web card should support the complete data model:

- render `context` when present
- render `question` prominently
- render all options when present
- allow selecting an option directly
- allow freeform input when the request allows it
- show final resolved answer after completion

If both options and freeform input are available, both are shown in the same card.

## Architecture

### Two-Lane Model

The feature is built around two coordinated but separate lanes.

#### Lane A: Historical Log Normalization

`codoxear/pi_messages.py` parses Pi session log entries and emits browser-friendly interaction events derived from `toolCall` + `toolResult` pairs.

This lane is authoritative for:

- completed interaction history
- scrollback
- replay after refresh or reconnect

#### Lane B: Live RPC Interaction Bridge

`codoxear/pi_rpc.py`, `codoxear/pi_broker.py`, and `codoxear/server.py` expose currently pending Pi RPC dialog requests to the browser and accept responses back from the browser.

This lane is authoritative for:

- active requests that are waiting on the user now
- submission of answers without using normal chat send
- stale/expired/conflict handling before the result lands in the session log

The browser merges both lanes into one card model keyed by stable IDs when available.

## Data Model

### Historical Interaction Event Shape

Add a normalized event family for interaction tools. For `ask_user`, the first concrete type is:

- `ask_user`

Suggested payload:

- `type: "ask_user"`
- `tool_call_id`
- `tool_name`
- `question`
- `context`
- `options`
- `allow_freeform`
- `allow_multiple`
- `timeout_ms`
- `answer`
- `cancelled`
- `was_custom`
- `resolved`
- `ts`

The normalization layer should prefer rendering-ready fields so the browser does not need to understand raw Pi tool payloads.

### Live Interaction Request Shape

Expose a generic pending UI request model from the broker/server bridge.

Suggested payload:

- `id`
- `method`
- `title`
- `message`
- `options`
- `placeholder`
- `prefill`
- `timeout_ms`
- `created_at`
- `tool_call_id` when available
- `tool_name` when derivable
- `interaction_kind` such as `ask_user`
- `status` (`pending`, `submitting`, `resolved`, `expired`, `error`)

This structure is intentionally general so future Pi dialog requests do not require a second protocol redesign.

## Frontend Design

### Card Composition

`codoxear/static/app.js` should add a dedicated interaction card renderer used by both historical and live data.

Card sections:

1. header badge (`Ask user` for the first implementation)
2. optional context block
3. question block
4. options block
5. input/editor block when supported
6. answer/result block
7. metadata/footer block

### State Rendering

#### Historical Resolved

- read-only options
- final answer shown inline
- status chips for `cancelled` and `custom`

#### Pending Select

- one button per option
- optional custom-answer input below if allowed
- controls disabled while submitting

#### Pending Confirm

- two explicit buttons such as `Yes` / `No`
- mapped back to the underlying generic `confirm` payload

#### Pending Input / Editor

- text area or input field depending on request type
- submit button enabled only when input is valid

#### Resolved Awaiting Log Sync

- local optimistic resolved state
- still visually marked as syncing until the durable log result arrives
- final log payload replaces the optimistic state when available

#### Expired / Error

- read-only error or stale label
- retry only when the request is still valid and the last submission failed locally

### Merge Rules

The browser should merge live and historical interaction records when they refer to the same interaction.

Preferred key order:

1. `tool_call_id`
2. pending request `id` mapped to known tool call metadata
3. no merge; render as separate live-only card if neither side has a stable shared key

Historical log data wins once present, because it is the durable source of truth.

## Broker and RPC Bridge

### `codoxear/pi_rpc.py`

Extend the RPC client so it can:

- retain `extension_ui_request` events from stdout
- send `extension_ui_response` records over stdin with the exact request `id`

Suggested method:

- `send_ui_response(request_id, payload)`

This should reuse the same RPC process and framing already used for prompt and abort commands.

### `codoxear/pi_broker.py`

Add broker-managed pending interaction state.

Responsibilities:

- collect live `extension_ui_request` events from the RPC client
- track active pending requests for the session
- expose them via broker socket commands
- forward validated browser responses back to Pi RPC
- prevent double submission for the same request
- mark requests stale when the session state makes them no longer actionable

Suggested new socket commands:

- `{"cmd":"ui_state"}`
- `{"cmd":"ui_response","id":"...",...}`

### Single-Response Guarantee

Broker state should enforce that a given pending request can only be accepted once.

If two browser tabs race to answer the same request:

- the first valid response succeeds
- later responses receive a conflict or already-resolved error

## Server API Design

Add Pi-only interaction routes in `codoxear/server.py`.

### Read Pending UI State

- `GET /api/sessions/<id>/ui_state`

Response includes the current live pending request set for that Pi session.

### Submit UI Response

- `POST /api/sessions/<id>/ui_response`

Request body contains:

- request `id`
- response payload appropriate for the method:
  - `value`
  - `confirmed`
  - `cancelled`

The server remains a thin validation and forwarding layer:

- verify the session exists
- verify backend is `pi`
- verify request shape is minimally valid
- forward to the broker

The server should not invent interaction semantics beyond those provided by the broker and Pi RPC protocol.

## Normalization Strategy for Ask-User History

`codoxear/pi_messages.py` should convert `ask_user` tool usage into a first-class normalized event instead of leaving it as a generic `tool` plus generic `tool_result` pair.

Recommended behavior:

- detect `toolCall.name == "ask_user"`
- extract rendering fields from the call arguments
- detect matching `toolResult.toolName == "ask_user"`
- extract `answer`, `cancelled`, and `wasCustom` from `details`
- emit one browser-facing `ask_user` event rather than two generic cards

For incomplete or malformed data:

- degrade gracefully to generic tool rendering if required fields cannot be recovered
- never drop the interaction silently

## Failure Handling

### Submission Failure

If browser submission fails before Pi accepts the response:

- keep the card interactive
- surface the error inline
- allow retry when still valid

### Expired or Missing Request

If the browser tries to answer a request that no longer exists:

- server/broker return a specific stale or not-found error
- UI marks the card stale and read-only

### Timeout / Cancellation

If Pi auto-resolves the request due to timeout or cancellation:

- pending state disappears or is marked resolved in the live bridge
- historical log result, when available, updates the card to a durable cancelled state

### Reconnect / Refresh

After a reload:

- completed interactions are restored from logs
- currently pending requests are restored from `ui_state`
- if a previously pending request no longer exists, the optimistic pending UI is discarded

## Testing

Add coverage in the existing Python and source-level UI tests.

### Backend Tests

- `codoxear/pi_rpc.py`
  - receives `extension_ui_request`
  - sends `extension_ui_response`
- `codoxear/pi_broker.py`
  - tracks pending requests
  - accepts only one successful response per request
  - handles stale/conflict/error paths
- `codoxear/server.py`
  - exposes `ui_state`
  - forwards `ui_response`
  - rejects non-Pi sessions and malformed payloads
- `codoxear/pi_messages.py`
  - normalizes historical `ask_user` events correctly
  - falls back safely when data is incomplete

### Frontend / Source Tests

- renderer existence for `ask_user` cards
- state branches for pending, resolved, syncing, stale, and error modes
- option click and freeform input submission paths
- disabled state while submitting
- merge behavior between live pending state and historical log data

### Integration Verification

Use fixture-based Pi session logs plus simulated RPC events to verify:

1. historical `ask_user` interactions render as full cards
2. pending requests appear without page reload
3. browser responses travel through `ui_response` instead of normal `/send`
4. successful responses settle into durable log-backed resolved cards

## Acceptance Criteria

The work is successful when:

1. historical Pi `ask_user` interactions render as full cards in the web UI
2. a currently pending `ask_user` request can be answered directly in the browser
3. answers are sent through a dedicated Pi UI-response bridge, not as ordinary chat prompts
4. the bridge protocol is generic enough to support `select`, `confirm`, `input`, and `editor` interactions later
5. stale, timeout, and multi-tab conflict cases fail predictably and visibly
6. automated tests cover normalization, bridge behavior, and UI rendering regressions
