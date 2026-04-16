# Pi Todo Popover Restoration Design

**Date:** 2026-04-07

## Goal

Restore the old web-style todo floating window in the new `web/` frontend by adding a dedicated `Todo` toolbar button for Pi sessions that opens a lightweight anchored popover.

## Approved Direction

Use a toolbar-triggered popover, not a timeline expansion and not the existing `Details` dialog.

- Show a `Todo` button in the conversation toolbar only when the active session backend is `pi`
- Open a compact floating panel anchored near that button
- Reuse the existing `diagnostics.todo_snapshot` payload as the data source
- Keep the existing `Details` dialog and main timeline behavior unchanged

## User-Approved Scope

### In scope

- Restore a dedicated todo floating surface in the new web UI
- Add a toolbar button in the active session header for Pi sessions only
- Show progress summary, todo items, status tags, and optional descriptions
- Support empty and unavailable states
- Close the popover on outside click, button toggle, and active-session change
- Keep the interaction mobile-safe with a bounded panel layout

### Out of scope

- Editing todo items from the browser
- Showing historical todo snapshots
- Moving todo content out of `Details`
- Adding new backend endpoints just for the popover
- Showing the button for non-Pi sessions

## Why This Direction

This is the closest match to the requested legacy behavior with the smallest change to the new architecture.

- A toolbar-triggered panel feels like a real floating window instead of another full dialog
- Using `diagnostics.todo_snapshot` avoids duplicating server logic and keeps data semantics consistent
- Limiting the feature to Pi sessions matches the current backend reality and avoids misleading empty affordances for other backends

## Existing Data Source

The new frontend already refreshes diagnostics through `sessionUiStore.refresh()`.

- `web/src/domains/session-ui/store.ts` fetches diagnostics for the active session
- `web/src/lib/api.ts` already exposes `/api/sessions/<id>/diagnostics`
- The diagnostics payload already includes `todo_snapshot` for Pi-backed sessions

The popover should read from this existing store state rather than issuing its own fetch.

## Component Design

### 1. Toolbar integration in `AppShell`

`web/src/app/AppShell.tsx` should own the popover open/close state because the toolbar button already lives there.

Responsibilities:

- Compute whether the active session is Pi-backed
- Render a `Todo` toolbar button only when `activeSession?.agent_backend === "pi"`
- Toggle the popover open state from that button
- Close the popover when:
  - the user clicks outside the popover/button area
  - the user clicks the button again
  - the active session changes
  - the Pi toolbar button disappears because the backend is no longer Pi

Suggested local state:

```ts
const [todoOpen, setTodoOpen] = useState(false);
```

`AppShell` should also hold refs for the button and popover root so outside-click handling stays localized.

### 2. New presentational component: `TodoPopover`

Add a focused component, for example:

- `web/src/components/workspace/TodoPopover.tsx`

Responsibilities:

- Accept a normalized `todo_snapshot`-shaped object from diagnostics
- Render the popover shell, title, summary, item list, and empty states
- Stay display-only with no network or store side effects

Suggested props:

```ts
interface TodoPopoverProps {
  snapshot: Record<string, unknown> | null;
}
```

The component should not know about sessions, polling, or toolbar logic.

### 3. Session workspace remains unchanged

`web/src/components/workspace/SessionWorkspace.tsx` should keep its current role.

The todo popover is a distinct floating affordance, not another workspace section. Reusing `SessionWorkspace` for the popover would blur two different UI semantics:

- `Details` is a broader diagnostics view
- `Todo` is a fast, focused glance surface

## Rendering Rules

The popover should render these states:

### Snapshot available

- Header title: `Todo`
- Summary line from `progress_text` when present
- One row per todo item in original order
- Each row shows:
  - title when present, otherwise `Untitled todo`
  - status chip using the raw Pi status string
  - description only when present and non-empty

### Snapshot unavailable but not erroneous

- Show an empty-state message such as `No todo list yet`

### Snapshot unavailable due to diagnostics/read failure

- Show an unavailable-state message such as `Todo list unavailable`

### Malformed or missing snapshot object

- Fail soft and render the same empty-state treatment as no snapshot

## Interaction Rules

- Button click toggles the popover
- Clicking outside closes the popover
- Clicking inside the popover does not close it
- Opening `Details` does not implicitly open `Todo`
- Opening `Todo` does not block use of the conversation pane
- Selecting another session closes the popover before the next session content is shown

## Layout and Styling

Add styles in `web/src/styles/global.css` for a legacy-adjacent floating card.

Suggested selectors:

- `.todoToolbarAnchor`
- `.toolbarButton.todoToggle`
- `.toolbarButton.todoToggle.isActive`
- `.todoPopover`
- `.todoPopoverHeader`
- `.todoPopoverSummary`
- `.todoPopoverList`
- `.todoPopoverItem`
- `.todoPopoverStatus`
- `.todoPopoverEmpty`

Styling guidance:

- Position the popover absolutely relative to a toolbar anchor wrapper
- Use a compact width on desktop, roughly in the 320px to 380px range
- Use stronger shadow and elevation than inline message cards so it reads as a floating surface
- Keep rounded corners and bordered card treatment aligned with the rest of the new UI
- Reuse the existing todo status color language where possible, but tighten density for popover rows
- On narrow screens, clamp width to viewport and align toward the visible screen edge to prevent overflow

This should feel lighter than `Details` and more focused than the right-side workspace rail.

## Data Flow

1. User selects a Pi session.
2. `AppShell` refreshes session UI state through the existing polling flow.
3. `sessionUiStore` stores `diagnostics`, including `todo_snapshot` when available.
4. `AppShell` shows the `Todo` toolbar button because the active backend is Pi.
5. User clicks `Todo`.
6. `AppShell` renders `TodoPopover`, passing `diagnostics?.todo_snapshot`.
7. `TodoPopover` renders either the summary/items view or a graceful empty/unavailable state.

## Testing Strategy

### `AppShell` behavior tests

Extend `web/src/app/AppShell.test.tsx` to cover:

- Pi session shows the `Todo` button
- Non-Pi session hides the `Todo` button
- Clicking the button opens the popover
- Clicking again closes the popover
- Changing the active session closes the popover

### `TodoPopover` rendering tests

Add focused tests, likely in:

- `web/src/components/workspace/TodoPopover.test.tsx`

Cover:

- available snapshot with summary and multiple items
- empty snapshot state
- unavailable/error state
- missing titles/descriptions fall back safely

### Style regression checks

At minimum, lock the presence of the key popover selectors in `web/src/styles/global.css` so future UI cleanup does not silently remove the floating surface.

## Acceptance Criteria

1. The toolbar shows a `Todo` button only for Pi sessions.
2. Clicking the button opens a distinct floating todo panel.
3. The panel content reflects `diagnostics.todo_snapshot` without extra backend calls.
4. Empty and unavailable states are clearly distinguishable.
5. Existing `Details`, workspace, and timeline todo rendering continue to work.

## Risks and Guardrails

- The current diagnostics type is loose, so the popover must validate snapshot fields defensively before rendering them.
- Toolbar layout is already compact; the new button should not push existing controls into awkward wrapping on narrow widths.
- Outside-click handling must be scoped carefully so the popover does not close immediately when the toggle button is used.

## Implementation Notes

- Prefer keeping the state local to `AppShell` instead of adding another store slice
- Prefer a new focused display component instead of expanding `SessionWorkspace`
- Do not change backend polling cadence or diagnostics endpoint behavior for v1
