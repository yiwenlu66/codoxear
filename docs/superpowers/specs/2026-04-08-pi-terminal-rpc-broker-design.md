# Pi Terminal RPC Broker Design

**Date:** 2026-04-08

## Goal

Make terminal-owned Pi sessions use the same live RPC control plane as web-owned Pi sessions so browser answers to `ask_user` close the active terminal prompt through a real `extension_ui_response`, not a text-send fallback.

## Problem

Today the Pi implementation is split across two different control models.

### Web-owned Pi sessions

Web-created Pi sessions already run through `codoxear/pi_broker.py`, which:

- owns the Pi RPC subprocess
- tracks pending `extension_ui_request` state
- exposes `ui_state` and `ui_response`
- forwards browser replies as real `extension_ui_response` records

This path preserves Pi's interaction semantics.

### Terminal-owned Pi sessions

Terminal-created Pi sessions currently run through `codoxear/broker.py`, which treats Pi like a foreground PTY process.

That path:

- does not implement `ui_state`
- does not implement `ui_response`
- cannot own pending Pi RPC interaction state
- forces `server.py` to fall back from browser `ui_response` to ordinary `send`

That fallback is semantically wrong for dialog-style Pi UI requests. It can make the web UI look interactive while the terminal still waits on the original prompt.

## Key Observation From OpenClaw

OpenClaw avoids this class of bug by making the gateway the single control plane for sessions, chat history, and live interaction state.

The important lesson is not "add a giant gateway". The lesson is:

- one runtime component owns the session state machine
- all clients talk to that component through structured methods
- clients render state; they do not invent interaction semantics by injecting terminal text

For Codoxear's Pi backend, the smallest correct adaptation is to make `codoxear/pi_broker.py` the single live control plane for Pi sessions, including terminal-owned sessions.

## Approved Direction

Unify all new Pi sessions behind `codoxear/pi_broker.py` and add a foreground terminal mode to that broker.

After this change:

- `pi_broker` owns the Pi RPC subprocess for both web-owned and terminal-owned Pi sessions
- the browser talks to `pi_broker` through the existing broker socket contract
- the local terminal also talks to the same `pi_broker`, but as an attached foreground UI bridge
- `ask_user` replies from the browser go through real `ui_response` handling for terminal-owned Pi sessions

This keeps Pi live interaction semantics in one place instead of splitting them across `broker.py` and `pi_broker.py`.

## Scope

### In Scope

- terminal-owned Pi sessions launched after this change
- foreground terminal support inside `codoxear/pi_broker.py`
- real `ui_state` / `ui_response` for terminal-owned Pi sessions
- capability metadata so the server and UI can tell whether live Pi interaction is truly supported
- removal of silent `ui_response -> send` fallback for new Pi RPC broker sessions
- targeted regression coverage for browser reply -> terminal close behavior

### Out of Scope

- retrofitting already-running legacy Pi PTY sessions
- changing Codex backend architecture
- replacing polling with WebSocket push in the Codoxear web UI
- introducing an OpenClaw-style global gateway for all backends
- supporting arbitrary non-dialog Pi UI widgets beyond the current dialog request/response model

## Design Principles

1. **Single source of truth**
   - `pi_broker` owns live Pi interaction state.
   - Web UI and terminal UI must consume the same broker state.

2. **Structured control beats text injection**
   - `prompt`, `abort`, `ui_state`, and `ui_response` are primary semantics.
   - PTY text injection is not an acceptable substitute for `ui_response`.

3. **Live state and durable history stay separate**
   - Pi RPC events remain the source for pending requests and immediate runtime state.
   - Pi session logs remain the source for durable conversation history.

4. **Clients render capability-aware state**
   - The browser must know whether a session supports real live interaction.
   - Unsupported sessions must be visibly non-interactive rather than degraded silently.

## Architecture

### 1. `codoxear/pi_broker.py` becomes the Pi runtime owner

`pi_broker` already contains the correct Pi RPC state machine for web-owned sessions:

- RPC polling and event draining
- pending request tracking
- `ui_state`
- `ui_response`
- prompt forwarding
- abort handling

This file becomes the canonical Pi broker implementation for all newly launched Pi sessions.

### 2. Add a foreground terminal mode to `pi_broker`

`pi_broker` gains an attached terminal mode that:

- reads local stdin when running in a real terminal
- writes Pi output to stdout/stderr in real time
- maps local message submission to Pi RPC `prompt`
- maps interrupt keys to Pi RPC `abort`
- leaves `ui_state` / `ui_response` available for the web side at the same time

This turns the terminal into a client-facing presentation layer over the broker-owned Pi RPC session instead of making the terminal process itself the session owner.

### 3. `codoxear/broker.py` stops owning Pi semantics

For `AGENT_BACKEND=pi`, `broker.py` should no longer maintain Pi as a PTY-owned session model.

Instead it should:

- delegate to foreground `pi_broker`, or
- be bypassed entirely by the Pi-specific launch path

The exact shell-level entrypoint can stay compatible with existing CLI usage, but the Pi state machine must live only in `pi_broker`.

### 4. `server.py` becomes capability-aware

`server.py` should distinguish between:

- Pi sessions backed by the real Pi RPC broker with live UI support
- legacy Pi sessions that do not support true live UI responses

For real Pi RPC broker sessions:

- `GET /api/sessions/<id>/ui_state` must come from the broker's pending request set
- `POST /api/sessions/<id>/ui_response` must require real broker support
- silent downgrade to `send` is no longer allowed

For legacy sessions:

- keep read-only historical rendering
- keep explicit non-live behavior
- do not claim live reply support

## Runtime Roles

### `pi_broker`

Owns:

- Pi RPC subprocess lifecycle
- broker socket
- pending UI request set
- busy state
- last turn bookkeeping
- output tail buffer
- live capability metadata

### Terminal attachment layer

Owns:

- local raw terminal input handling
- local stdout/stderr rendering
- local prompt composition UX
- local interrupt key mapping

Does not own:

- pending request truth
- browser interaction semantics
- durable session history

### Web UI

Owns:

- rendering historical log-backed `ask_user` cards
- rendering live pending cards only when broker capability + pending state are present
- submitting structured `ui_response` payloads

Does not own:

- request matching semantics beyond broker-provided IDs and known merge rules
- fallback conversion from UI response to plain chat text

## Broker Metadata Changes

Pi sidecar metadata should expose capability hints so the server and UI can reason about the session correctly.

Add or standardize fields such as:

- `backend: "pi"`
- `transport: "pi-rpc"`
- `supports_web_control: true`
- `supports_live_ui: true`
- `ui_protocol_version: 1`
- `session_path`
- `sock_path`
- `broker_pid`
- `agent_pid` / compatibility `codex_pid`

These fields do not need to create a new cross-backend abstraction. They only need to let the server avoid lying about live Pi UI support.

## Control-Plane Contract

The Pi broker socket contract remains method-oriented.

Required commands:

- `state`
- `tail`
- `send`
- `keys`
- `ui_state`
- `ui_response`
- `shutdown`

### `send`

For Pi RPC broker sessions, `send` maps to Pi RPC `prompt`.

### `keys`

For Pi RPC broker sessions, `keys` must keep using semantic abort behavior for supported interrupt sequences instead of raw PTY injection.

### `ui_state`

Returns only broker-owned pending dialog requests.

### `ui_response`

Accepts broker-owned request IDs and forwards real Pi RPC `extension_ui_response` payloads.

It must reject:

- unknown request IDs
- already-resolved requests
- broker instances that do not actually support live Pi UI

## Terminal Behavior

### Prompt submission

The foreground terminal mode should preserve the current user expectation that typing a prompt in the terminal starts a Pi turn.

Implementation can be line-oriented; it does not need to recreate every existing PTY nuance if correctness would suffer.

### Output rendering

Broker-drained output should be written to the terminal as Pi events arrive.

The output path should remain simple:

- assistant/user/tool text visible in the local terminal
- stderr visible distinctly
- no second state machine in the terminal renderer

### Ask-user handling

When Pi emits a dialog request:

- the terminal continues to show Pi's native textual prompt
- `pi_broker` records the pending request
- the web UI may answer it through `ui_response`
- once Pi resolves it, the terminal naturally advances because the underlying RPC request has been satisfied

This is the key semantic improvement over the current fallback model.

## Server Behavior Changes

`submit_ui_response()` currently falls back to plain `send` when the broker replies `unknown cmd`.

After this redesign:

- real Pi RPC broker sessions must not use that fallback
- fallback may remain only for explicitly legacy Pi session types
- errors should be explicit so the web UI can present "live response not supported for this session"

This preserves correctness over convenience.

## Web UI Behavior Changes

The current browser guard that only enables reply controls when a live request exists remains correct and should stay.

Additional improvements:

- only show interactive controls for Pi sessions with live-ui capability
- show a clear read-only explanation for historical-only unresolved prompts
- treat a missing live request on a live-capable session as stale/expired, not as something to coerce into chat send

## Migration Strategy

### New sessions

All new terminal-owned Pi sessions should use the Pi RPC broker path.

### Existing sessions

Existing legacy Pi PTY sessions remain readable but are not upgraded in place.

Expected behavior:

- historical chat remains visible
- historical `ask_user` cards remain read-only
- no promise of browser reply synchronization for those sessions

This avoids in-place mutation of semantics for already-running sessions.

## Testing

### Backend tests

#### `tests/test_pi_broker.py`

Add coverage for:

- foreground Pi broker mode initialization
- prompt submission from terminal-owned Pi sessions still using RPC `prompt`
- pending UI request tracking in foreground mode
- browser `ui_response` resolving a request owned by a foreground Pi broker session
- replay protection and stale request handling still working in foreground mode

#### `tests/test_pi_server_backend.py`

Add coverage for:

- terminal-owned Pi sessions advertising live Pi UI capability
- `submit_ui_response()` not falling back to `send` for live-capable Pi RPC sessions
- legacy Pi sessions still being treated as non-live when appropriate

#### `tests/test_pi_rpc.py`

Keep or extend coverage for:

- `extension_ui_request`
- `extension_ui_response`
- prompt/abort behavior used by the foreground broker mode

### Frontend tests

#### `web/src/components/conversation/AskUserCard.test.tsx`

Keep coverage for:

- interactive reply only when a real live request exists
- no submission for historical-only unresolved cards

Add capability-aware cases if the UI starts surfacing explicit live-support indicators.

## Acceptance Criteria

The redesign is successful when:

1. a newly launched terminal-owned Pi session is backed by `pi_broker`
2. the same session exposes real `ui_state` and `ui_response` to the web UI
3. answering `ask_user` from the browser resolves the active Pi dialog without converting it into a normal chat send
4. the terminal advances because the real Pi RPC dialog was satisfied
5. legacy Pi sessions remain safely read-only for browser-side interaction
6. tests prove that live-capable Pi sessions no longer rely on the `unknown cmd` fallback path

## Risks and Mitigations

### Risk: terminal UX regresses relative to the old PTY path

Mitigation:

- keep the foreground terminal layer minimal
- verify prompt, output, and interrupt flows before adding extra polish

### Risk: server capability detection is ambiguous

Mitigation:

- write explicit sidecar metadata fields for live Pi support
- gate fallback behavior on those fields instead of inference alone

### Risk: two Pi launch paths survive accidentally

Mitigation:

- make one code path canonical for new Pi sessions
- keep legacy behavior only as a compatibility read path, not as an equal launch mode

## Non-Goals

This design does not attempt to:

- centralize all backends behind an OpenClaw-sized gateway
- eliminate log-based history loading
- redesign the Codex backend around Pi RPC ideas
- introduce full push-based realtime browser transport in this change

## Summary

OpenClaw's useful lesson is that clients should not guess or fake session control semantics. A single runtime component should own them.

For Codoxear Pi sessions, that owner should be `codoxear/pi_broker.py`.

The redesign therefore moves terminal-owned Pi sessions onto the same broker-owned RPC control plane already used by web-owned Pi sessions, so browser answers to `ask_user` become real Pi UI responses rather than text injection hacks.
