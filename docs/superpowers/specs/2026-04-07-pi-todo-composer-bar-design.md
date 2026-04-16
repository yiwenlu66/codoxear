# Pi Todo Composer Bar Design

**Date:** 2026-04-07

## Goal

Restore the old web-style Pi todo visibility in the new `web/` frontend by showing a compact todo summary bar directly above the composer, with click-to-expand behavior for the full todo list.

## Approved Direction

Use a composer-adjacent inline surface, not a toolbar popover.

- Place the todo UI immediately above the input composer
- Default to a compact summary bar when todo data is available
- Expand inline to show the full todo list when the summary bar is clicked
- Show nothing when there is no current todo snapshot for the active session

## User-Approved Scope

### In scope

- Show todo state above the composer for Pi sessions only
- Render a compact summary row when the active Pi session has a non-empty todo snapshot
- Expand and collapse the full todo list inline in the same area
- Use the existing `diagnostics.todo_snapshot` payload as the only data source
- Hide the surface entirely for non-Pi sessions, stale session UI state, missing snapshots, or empty todo lists
- Keep the layout mobile-safe and visually integrated with the composer area

### Out of scope

- Dedicated toolbar todo button
- Floating popover or independent modal behavior
- Editing todo items from the browser
- New backend endpoints or polling behavior
- Showing empty-state placeholder rows above the composer when no todo exists

## Why This Direction

This matches the requested legacy semantics more closely than the toolbar popover:

- The todo lives where the user is actively typing, which makes it feel like part of the current task context
- A compact summary bar preserves space better than a permanently expanded list
- Inline expansion avoids the positioning and clipping issues that come with popovers

## Data Source

Continue using the existing session UI diagnostics payload.

- `web/src/domains/session-ui/store.ts` already refreshes `diagnostics`
- `diagnostics.todo_snapshot` remains the source of truth
- The composer-facing UI should render only when `sessionUiStore.sessionId === activeSessionId`

This avoids stale cross-session rendering and keeps the todo UI aligned with current session state.

## Component Design

### 1. Composer-top todo bar

`web/src/components/composer/Composer.tsx` should own the inline expansion state because the bar belongs to the composer area.

Responsibilities:

- Read the active session id from the sessions store
- Read the current diagnostics and session-ui session id from the session UI store
- Determine whether the active session is Pi-backed
- Render the todo summary bar only when all of these are true:
  - active session exists
  - active session backend is `pi`
  - `sessionUiStore.sessionId === activeSessionId`
  - `diagnostics.todo_snapshot.available === true`
  - `diagnostics.todo_snapshot.items.length > 0`
- Toggle the expanded state when the summary bar is clicked
- Reset the expanded state when the active session changes or when the todo bar disappears because the snapshot is no longer displayable

Suggested local state:

```ts
const [todoExpanded, setTodoExpanded] = useState(false);
```

### 2. Reusable inline todo panel content

Refactor the current todo display component into an inline-friendly surface.

Suggested direction:

- Replace the popover-specific `TodoPopover` presentation with a composer-surface component such as `TodoComposerPanel`
- Keep the rendering logic focused on:
  - summary text
  - todo item list
  - title / status / description rows
- The component stays display-only and receives pre-filtered snapshot data from its parent

Responsibilities:

- Render a compact header/summary row
- Render the expanded todo list only when asked
- Preserve Pi item order and raw status strings
- Keep defensive normalization against malformed snapshot payloads

### 3. Remove toolbar todo behavior

`web/src/app/AppShell.tsx` should no longer own todo open state or render a toolbar `Todo` button.

Responsibilities after the change:

- Keep current session selection and diagnostics refresh behavior
- Stop rendering the toolbar todo entry
- Stop managing todo-specific open/close state
- Leave `Details` and other toolbar controls unchanged

This keeps todo ownership close to the composer instead of splitting it between shell and input UI.

## Rendering Rules

### Hidden state

Render nothing above the composer when any of these are true:

- no active session
- active session backend is not `pi`
- `sessionUiStore.sessionId !== activeSessionId`
- `todo_snapshot.available !== true`
- `todo_snapshot.items` is missing, malformed, or empty after normalization

This is deliberate. The user explicitly chose “only show it when there is todo content.”

### Collapsed summary state

When the todo bar is visible but not expanded:

- Render a compact clickable row above the composer
- Show `progress_text` when present, otherwise a compact fallback such as `Todo`
- Show a subtle affordance that the row is expandable
- Keep the row visually lighter than full cards in the conversation timeline

### Expanded state

When expanded:

- Keep the summary row visible as the header
- Show the full todo list directly below it in the same inline container
- Each item shows:
  - title, or `Untitled todo` fallback
  - raw Pi status string without rewriting the text
  - description only when present
- Cap height and enable scrolling so the composer remains visible

## Interaction Rules

- Clicking the summary row toggles expand/collapse
- Clicking inside the expanded list does not collapse it unless the click is on the summary toggle
- Switching sessions collapses the bar
- Losing display eligibility collapses the bar
- Sending a message does not automatically collapse it

## Layout and Styling

Add composer-adjacent styles in `web/src/styles/global.css`.

Suggested selectors:

- `.composerTodoBar`
- `.composerTodoBarButton`
- `.composerTodoBarButton.isExpanded`
- `.composerTodoSummary`
- `.composerTodoPanel`
- `.composerTodoList`
- `.composerTodoItem`
- `.composerTodoItemHead`
- `.composerTodoStatus`

Styling guidance:

- Visually attach the todo bar to the composer area rather than the conversation timeline
- Use a compact row height for the collapsed state
- Use a rounded, lightly elevated panel for the expanded content
- Keep max height bounded with internal scrolling
- Ensure long titles, descriptions, and statuses wrap without horizontal overflow
- On mobile, keep the bar full-width within the composer stack and avoid overlay positioning

## Data Flow

1. User selects a session.
2. `AppShell` continues to refresh session UI diagnostics through the existing store flow.
3. `Composer` reads the active session and current diagnostics.
4. If the active session is a Pi session with a non-empty, current todo snapshot, `Composer` renders the summary bar above the input.
5. User clicks the summary bar.
6. The composer-local expanded state toggles, and the inline panel reveals or hides the full todo list.

## Testing Strategy

### Composer tests

Update `web/src/components/composer/Composer.test.tsx` to cover:

- Pi session with non-empty todo snapshot shows the summary bar above the composer
- Clicking the summary bar expands the full todo list
- Clicking again collapses it
- No bar is shown for non-Pi sessions
- No bar is shown when `sessionUiStore.sessionId !== activeSessionId`
- No bar is shown when the snapshot exists but has no valid items

### AppShell tests

Update `web/src/app/AppShell.test.tsx` to remove toolbar todo expectations and ensure the shell still works without a `Todo` toolbar button.

### Style regression checks

Replace the popover-oriented style contract checks with composer-bar checks in `web/src/styles/layout-scroll.test.ts`.

Lock at least:

- composer todo summary/button selectors
- expanded panel selectors
- bounded height/overflow behavior
- mobile-safe composer-adjacent layout hooks

## Acceptance Criteria

1. The toolbar no longer contains a dedicated `Todo` button.
2. A Pi session with current non-empty todo data shows a compact summary bar directly above the composer.
3. Clicking the summary bar expands and collapses the inline todo list.
4. Non-Pi sessions and sessions without current todo data show no todo surface above the composer.
5. The composer remains usable on desktop and mobile while the expanded todo list is open.

## Risks and Guardrails

- Do not let stale diagnostics from a previous session render above the new composer
- Do not let an expanded todo list grow so tall that the input area disappears
- Do not reintroduce toolbar-width pressure or popover clipping issues through leftover shell logic
- Keep the summary and expanded list in the same semantic component family so behavior stays coherent

## Implementation Notes

- Prefer moving todo ownership from `AppShell` into `Composer`
- Reuse the existing normalization logic from the current todo display component rather than duplicating it
- Keep the current timeline todo card behavior unchanged unless a later request says otherwise
- No commit is required for the spec or implementation unless the user explicitly asks for one
