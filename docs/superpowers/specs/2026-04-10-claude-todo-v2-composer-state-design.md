# Claude Todo V2 Composer State Design

**Date:** 2026-04-10

## Goal

Make the existing composer todo bar show the current Claude Todo V2 task list state when Pi session logs contain the new `claude-todo-v2-state` and `TodoWrite` records.

The user-facing result is simple:

- the todo surface stays in the existing position above the composer
- it behaves like the current todo bar and panel
- it shows the current todo list state instead of ignoring the newer Claude Todo V2 log format

## Approved Direction

Use backend normalization to fold the new Claude Todo V2 state records into the existing `todo_snapshot` contract.

- keep `diagnostics.todo_snapshot` as the only composer data source
- treat `claude-todo-v2-state` as enablement and recency metadata, not a standalone UI card for this task
- treat `TodoWrite.details.newTodos` as the authoritative current Claude Todo V2 list snapshot
- keep the composer UI shape unchanged so the new format renders through the existing todo bar and panel

## User-Approved Scope

### In scope

- Support Pi session logs that contain `type: "custom"` with `customType: "claude-todo-v2-state"`
- Support Pi tool results with `toolName: "TodoWrite"`
- Map the latest `TodoWrite.details.newTodos` into the existing normalized todo snapshot structure
- Respect `panelEnabled` as a visibility gate for Claude Todo V2 composer todo display
- Render the resulting current todo list through the existing composer todo bar above the input
- Add focused backend and frontend tests for the new normalization path

### Out of scope

- New dedicated API fields for Claude Todo V2 state
- Rendering raw `claude-todo-v2-state` metadata directly to end users
- Reworking the composer todo component hierarchy
- Editing todo items from the browser
- Inventing extra Claude Todo V2 state transitions that are not present in the observed logs

## Why This Direction

The requested behavior is “show the current todo list state above the input like the other todo UI,” not “add a second todo system.”

A backend-normalized path best preserves that invariant:

- the composer already consumes `diagnostics.todo_snapshot`
- the workspace and other todo surfaces can reuse the same compatibility work later
- the parsing logic stays next to the session log semantics instead of being duplicated in web components
- old Pi todo snapshots and new Claude Todo V2 snapshots can coexist behind one stable web-facing shape

A frontend-only reconstruction would force the web app to understand raw session log semantics that the backend already owns.

## Current Facts From The Code And Logs

### Existing code behavior

- `codoxear/server.py` exposes one todo payload for diagnostics via `_todo_snapshot_payload_for_session`
- `codoxear/pi_messages.py` currently builds todo snapshots from:
  - legacy Pi `manage_todo_list`
  - Claude Todo V2 task-assignment aggregation
  - `TaskCreate` and `TaskUpdate`
- `web/src/components/composer/Composer.tsx` only renders todo above the composer when `diagnostics.todo_snapshot` is current for the active Pi session
- `web/src/components/composer/TodoComposerPanel.tsx` already renders a compact summary bar plus expandable list when a normalized snapshot is available

### Observed new log format

The provided session log includes:

```json
{"type":"custom","customType":"claude-todo-v2-state","data":{"panelEnabled":true,"lastActivationKey":"4448f1b1-f9f1-4ed1-9ac1-cd26850f0db2:4448f1b1-f9f1-4ed1-9ac1-cd26850f0db2"},"id":"920462a4","parentId":"76eea1e3","timestamp":"2026-04-10T09:58:42.627Z"}
```

The same log family also contains `TodoWrite` tool results whose `details.newTodos` array carries the current task list state.

Observed `TodoWrite` semantics from the log sample:

- `newTodos` is a full replacement snapshot, not a partial patch
- each item includes:
  - `content`
  - `status`
  - `activeForm`
- later `TodoWrite` entries supersede earlier ones

This is the missing source for the current Claude Todo V2 composer todo state.

## Architecture

### 1. Extend backend Claude Todo V2 snapshot reading in `codoxear/pi_messages.py`

Add support for the new Pi/Claude Todo V2 records inside the latest-snapshot code path.

Responsibilities:

- scan the log for the latest relevant Claude Todo V2 state records
- recognize `type == "custom"` with `customType == "claude-todo-v2-state"`
- read `data.panelEnabled` and the event timestamp defensively
- recognize `toolResult` entries where `toolName == "TodoWrite"`
- read `details.newTodos` as the candidate current snapshot payload
- continue to support the existing fallback paths for legacy Pi todo snapshots and prior Claude Todo V2 compatibility work

The output remains the same normalized snapshot shape already consumed by the web app.

### 2. Claude Todo V2 state interpretation

Interpret the new event pair with minimal semantics.

#### `claude-todo-v2-state`

Use this record for:

- whether the Claude Todo V2 panel is enabled for the session
- timestamp ordering so newer Claude Todo V2 state can supersede older state metadata

Do not use it to fabricate task rows, because it does not contain the actual list.

#### `TodoWrite`

Use this record for:

- the current todo list contents via `details.newTodos`
- replacement semantics: the latest valid `newTodos` wins

This matches the observed payload and avoids inventing a speculative event merge model.

### 3. Normalization rules for `TodoWrite.details.newTodos`

Map each valid `newTodos` item into the existing normalized todo item shape.

Suggested mapping:

- `content -> title`
- `status -> status`
- `activeForm -> description` when present
- `source -> "claude-todo-v2"`

Status normalization:

- `pending` -> `not-started`
- `in_progress` -> `in-progress`
- `completed` -> `completed`
- any other missing or unknown value -> `not-started`

Item filtering:

- ignore malformed items
- ignore items with no non-empty `content`
- preserve incoming list order

Snapshot counts and summary:

- total count from normalized items
- completed count from normalized statuses
- in-progress count from normalized statuses
- not-started count from normalized statuses
- `progress_text` formatted with the existing summary style such as `1/3 completed`

### 4. Visibility rules for composer display

Keep the current composer display contract, with one Claude Todo V2-specific gate.

The composer todo bar should render only when all of these are true:

- active session exists
- active session backend is `pi`
- session UI diagnostics belong to the active session
- normalized `todo_snapshot.available === true`
- normalized `todo_snapshot.items.length > 0`
- for Claude Todo V2-derived snapshots, the latest known `panelEnabled` is `true`

Important semantic boundary:

- `panelEnabled` means the Claude Todo V2 surface is allowed to appear
- it does not force an empty placeholder when there are no valid todo items yet

This preserves the current UI behavior of hiding the composer todo bar when there is nothing meaningful to show.

### 5. Frontend impact in `web/`

The frontend should keep using the existing todo bar pipeline.

Expected changes are minimal:

- `web/src/components/composer/Composer.tsx` continues reading `diagnostics.todo_snapshot`
- `web/src/components/composer/TodoComposerPanel.tsx` continues rendering the normalized snapshot
- no separate Claude Todo V2 composer widget is introduced

The compatibility work should therefore land primarily in backend normalization and tests, not in a new frontend rendering path.

## Data Flow

1. Pi session log records a `claude-todo-v2-state` custom event.
2. Pi session log records one or more `TodoWrite` tool results.
3. `codoxear/pi_messages.py` reads the latest enabled Claude Todo V2 state and latest valid `newTodos` snapshot.
4. The backend normalizes those records into the existing `todo_snapshot` structure.
5. `codoxear/server.py` returns that normalized snapshot through session diagnostics.
6. `Composer` receives the updated diagnostics and renders the existing todo bar above the input.
7. Expanding the bar shows the current Claude Todo V2 todo list through the existing panel UI.

## Error Handling And Edge Cases

- If `claude-todo-v2-state` exists but `panelEnabled` is false, do not show a Claude Todo V2-derived composer snapshot
- If `TodoWrite` exists but `newTodos` is missing or malformed, ignore that entry and continue scanning older valid snapshots
- If `newTodos` exists but all items normalize away, treat the snapshot as non-displayable
- If only `panelEnabled` exists and no valid `newTodos` snapshot is found, keep the composer todo bar hidden
- If older legacy Pi todo snapshot data exists for the same session, keep the existing legacy-first behavior unchanged unless a valid Claude Todo V2 current-state snapshot is explicitly selected by the parser path for that session
- If future Claude Todo V2 payloads add fields, preserve the smallest invariant-preserving model instead of guessing new semantics

## Testing Strategy

### Backend tests in `tests/test_pi_todo_snapshot.py`

Add coverage for:

- `claude-todo-v2-state` alone does not produce a displayable snapshot
- `panelEnabled: true` plus `TodoWrite.details.newTodos` produces the expected normalized snapshot
- latest `TodoWrite` wins over earlier snapshots
- `pending`, `in_progress`, and `completed` map to the expected normalized statuses
- `activeForm` becomes item description
- `panelEnabled: false` suppresses Claude Todo V2 composer snapshot display

### Frontend tests in `web/src/components/composer/Composer.test.tsx`

Add or adjust coverage for:

- a Claude Todo V2-derived normalized snapshot still renders through the existing composer todo bar
- expanded composer panel shows the normalized task content and `activeForm`-backed description
- the existing Pi-only and stale-session guards continue to hold

The frontend tests should stay focused on the normalized snapshot contract, not on raw log parsing.

## Acceptance Criteria

The work is complete when all of the following are true:

- a Pi session with the new Claude Todo V2 log format produces a normalized current todo snapshot
- the composer shows that current todo list above the input without introducing a separate UI path
- the composer remains hidden when Claude Todo V2 is disabled or when there is no valid current todo list to show
- existing legacy todo behavior is not regressed
- focused backend and frontend tests cover the new normalization behavior

## Implementation Notes For The Next Step

Implementation should stay narrow:

- prefer extending existing helpers over creating a second todo data pipeline
- keep the public diagnostics shape unchanged
- add tests first for the new backend normalization path before changing parser logic
- use the composer UI as a verification surface, not as the place where raw Claude Todo V2 semantics are reconstructed
