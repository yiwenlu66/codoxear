# New Session Backend Selector Simplification

## Goal

Remove the duplicate backend selector from the bottom of the `New session` dialog and keep the header backend tabs (`Codex` / `Pi`) as the only backend switching control.

## Current State

The dialog currently exposes backend selection in two places:

1. Header icon tabs (`newSessionBackendTabs`)
2. A separate `Backend` picker row in the form body (`newSessionBackendRow`)

Both controls update the same `newSessionBackend` state, so the lower picker duplicates the header behavior without adding any extra capability.

## Desired Behavior

- The header backend tabs remain visible and continue to control `newSessionBackend`.
- The bottom `Backend` field is removed from the dialog entirely.
- All backend-dependent UI continues to update from the selected header tab, including:
  - provider / tmux visibility for Codex
  - reasoning options per backend
  - Pi session mode and continue-session UI
  - resume candidate loading
  - worktree availability
- Remembered backend behavior remains unchanged.

## Implementation Shape

### Remove duplicate form UI

Delete the bottom backend form row and the picker elements that only exist to render that row:

- `newSessionBackendRow`
- `newSessionBackendField`
- `newSessionBackendBtn`
- `newSessionBackendMenu`

### Keep a single backend state source

Retain:

- `newSessionBackendTabs`
- `renderNewSessionBackendTabs()`
- `setNewSessionBackend(...)`

The header tabs become the only visible control that changes backend state.

### Clean up menu wiring

Remove backend-picker-specific menu state handling that only supported the deleted bottom picker, including:

- dialog menu open/close toggling for the backend picker
- backend picker positioning logic
- backend picker click handler
- backend picker reset code during dialog open/close

## Acceptance Criteria

- `New session` no longer shows a `Backend` field in the form body.
- Clicking the header `Codex` / `Pi` icons still switches backend state.
- Switching backend still updates the rest of the form exactly as before.
- Opening and closing the dialog does not produce JavaScript errors.
- No empty spacing or layout regression remains where the removed row used to be.

## Risks

- If any dialog menu bookkeeping still references removed backend picker nodes, the modal may throw at runtime.
- If any code path still expects the deleted picker button for label syncing, backend changes may stop reflecting correctly.

## Out of Scope

- Changing backend defaults or launch behavior
- Redesigning the header tabs
- Adjusting provider/reasoning semantics
