# Pi Web Message Rendering Design

**Date:** 2026-04-03

> Superseded on 2026-04-04: the Pi web UI now uses a single `Conversation` view. The `Event Stream` tab and its backend plumbing were removed after proving redundant in practice.

## Goal

Upgrade the Pi-backed web session view so it can faithfully present Pi session history as a production-ready browser experience, while keeping the default chat view readable for day-to-day use.

## Problem

`codoxear` currently treats Pi session files as a narrow chat stream. It normalizes a few visible shapes (`user`, `assistant`, `tool`, `tool_result`, `subagent`), but real Pi session files also include:

- session bootstrap entries
- model / thinking-level changes
- assistant reasoning blocks (`thinking`)
- tool-call-only assistant steps
- structured tool results with `details`, errors, and todo snapshots
- heterogeneous provider payloads (OpenAI responses, Gemini CLI, etc.)

This creates two failures:

1. important Pi semantics are lost or hidden
2. adding more raw events directly into the main chat timeline would make the UI noisy

## Approved Direction

Use a dual-view model:

- `Conversation` view: optimized for everyday reading
- `Event Stream` view: complete, faithful Pi event timeline for debugging and auditability

This keeps the default experience calm while still exposing the full Pi runtime history.

## User Experience

### 1. Conversation View

Conversation view remains the default when opening a Pi session.

It should show:

- user messages
- assistant visible text/final output
- assistant reasoning blocks as collapsible reasoning cards
- tool calls as lightweight execution markers
- tool results as expandable result cards with clear error styling when `isError=true`
- subagent calls/results as dedicated cards
- `manage_todo_list` snapshots as progress cards summarizing current task state

It should not inline low-value system noise such as raw session bootstrap entries or configuration changes unless they materially affect understanding of the turn.

### 2. Event Stream View

Event Stream is a separate toggle for Pi sessions only.

It should show the full normalized event sequence, including:

- session start
- model changes
- thinking-level changes
- user / assistant / reasoning / tool / tool results / subagents
- todo snapshots
- generic fallback events for unsupported Pi entry types

Each row should show:

- event type badge
- timestamp
- concise summary
- expandable body for markdown / structured details / raw JSON excerpts when useful

This view must be complete enough that an engineer can reconstruct what Pi actually did from the browser without opening the raw session file.

## Data Model

### Normalized Pi Event Types

Extend Pi normalization to emit a richer event vocabulary.

Core visible events:

- `user`
- `assistant`
- `reasoning`
- `tool`
- `tool_result`
- `subagent`
- `todo_snapshot`

System / event-stream events:

- `pi_session`
- `pi_model_change`
- `pi_thinking_level_change`
- `pi_event` (generic fallback)

### Event Payload Principles

Each normalized event should carry enough data for rendering without reparsing the original entry in the browser.

Examples:

- `reasoning`: text, summary if available, timestamp
- `tool_result`: tool name, text, `is_error`, details summary, timestamp
- `todo_snapshot`: progress counts, items, operation, timestamp
- `pi_model_change`: provider/model label, timestamp
- `pi_event`: kind, summary, compact details/raw snippet, timestamp

## Rendering Rules

### Conversation View Filtering

Include:

- `user`
- `assistant`
- `reasoning`
- `tool`
- `tool_result`
- `subagent`
- `todo_snapshot`

Exclude by default:

- `pi_session`
- `pi_model_change`
- `pi_thinking_level_change`
- generic `pi_event`

### Event Stream Filtering

Include every normalized Pi event type.

## UI Structure

### Top Bar Toggle

When the selected session backend is Pi, show a compact segmented control in the chat header:

- `Conversation`
- `Event Stream`

Hide it for Codex sessions.

### Row Components

Add dedicated renderers for:

- reasoning card
- todo snapshot card
- system/config event card
- generic event-stream card

Reuse existing markdown rendering and clickable file-path upgrade logic where possible.

## Error and Detail Handling

- tool results with `isError=true` should use distinct error accents
- tool results with structured `details` should surface a short summary plus expandable detail body
- generic fallback rows should never silently drop unsupported Pi entry types
- unknown content should degrade to readable summaries, not empty rows

## Caching and Refresh

- keep the existing fast cached experience for conversation view
- allow raw event-stream mode to force a fresh init render when switching views so complete Pi history is available
- maintain live polling in both views using the same normalized event stream, with the browser deciding what to render per active view

## Testing

Add regression coverage for:

- reasoning block normalization
- todo snapshot extraction from `manage_todo_list`
- model/thinking-level/system event normalization
- fallback `pi_event` handling for unsupported entries
- UI source tests ensuring the Pi view toggle exists and the render pipeline recognizes new event types

## Non-Goals

- changing Codex backend message rendering semantics
- building a generalized multi-backend event inspector
- exposing full raw JSON for every row by default in conversation view

## Acceptance Criteria

The work is successful when:

1. Pi sessions expose both `Conversation` and `Event Stream` modes in the web UI.
2. Conversation mode cleanly renders reasoning, tool activity, todo progress, subagents, and normal messages.
3. Event Stream mode surfaces configuration/system events and unsupported Pi entries instead of dropping them.
4. Existing Codex session rendering continues to work unchanged.
5. Automated tests cover the new normalization and UI plumbing.
