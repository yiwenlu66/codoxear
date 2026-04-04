# Enter-to-Send Composer Preference Design

**Date:** 2026-04-04

## Goal

Add an optional browser-local composer preference so users can choose whether pressing `Enter` sends a message directly.

## Problem

The current composer behavior in `codoxear/static/app.js` only sends on `Ctrl+Enter` / `Cmd+Enter`.

That works well for multiline prompts, but it is slower on mobile and laptop use when the user wants chat-style behavior. At the same time, changing the default behavior globally would be risky because existing users may already rely on plain `Enter` for newlines.

## Approved Direction

Add a new setting in the existing `Settings` dialog:

- label: `Press Enter to send`
- default: off
- storage: browser-local `localStorage`

Behavior:

- when off: keep the current behavior (`Ctrl+Enter` / `Cmd+Enter` sends; plain `Enter` inserts a newline)
- when on: `Enter` sends and `Shift+Enter` inserts a newline

## User Experience

### 1. Settings Surface

The preference lives in the existing Settings dialog rather than the composer itself.

Reasons:

- it matches the current pattern for per-browser preferences
- it avoids adding more controls to the already dense composer row
- it keeps mobile layout unchanged

The new checkbox should be presented as a simple local preference, alongside the existing local device/browser behaviors.

### 2. Composer Behavior

When the setting is disabled:

- `Enter` keeps its normal textarea behavior
- `Ctrl+Enter` sends
- `Cmd+Enter` sends

When the setting is enabled:

- `Enter` sends the current draft
- `Shift+Enter` creates a newline
- `Ctrl+Enter` and `Cmd+Enter` may continue to send as compatible shortcuts

### 3. Draft Safety

Sending should continue to use the existing form submission path so the feature does not introduce a second send implementation.

The preference only changes keyboard interpretation. It does not change:

- queueing behavior
- send button behavior
- attachment behavior
- textarea auto-grow behavior
- toast / pending-send / polling logic

## State Model

Store the preference in `localStorage` under a dedicated key such as `codoxear.enterToSend`.

Properties:

- browser-local, not server-synced
- loaded during app initialization together with other local toggles
- immediately reflected in the Settings checkbox and composer keyboard handler

This keeps the feature consistent with existing local-only controls like announcement and notification enablement.

## Implementation Outline

### 1. Settings Dialog

In `codoxear/static/app.js`:

- add a new checkbox control to the existing Settings dialog
- initialize it from `localStorage`
- persist changes immediately on toggle

No backend API changes are needed.

### 2. Composer Key Handling

Update the existing `textarea` `keydown` handler in `codoxear/static/app.js`.

Rules:

- ignore non-`Enter` keys
- ignore IME composition (`e.isComposing`)
- if `Press Enter to send` is off, preserve the current `Ctrl/Cmd+Enter` behavior
- if the setting is on:
  - `Shift+Enter` falls through to normal textarea newline behavior
  - plain `Enter` prevents default and submits the form
  - `Ctrl+Enter` / `Cmd+Enter` can also submit for compatibility

### 3. Styling

The existing Settings dialog structure should be reused.

Only minimal styling adjustments are expected, if any, to keep checkbox rows visually aligned with the current settings layout.

## Error Handling and Edge Cases

- do not send while IME composition is active
- do not break multiline input when the option is disabled
- do not require server availability to toggle the preference
- do not introduce duplicate submits by bypassing the existing `form.requestSubmit()` flow
- preserve current behavior for users who never open Settings

## Testing

Regression coverage should verify:

- the Settings source includes the new `Press Enter to send` control
- the composer keyboard handler branches on the stored preference
- enabled mode uses `Enter` to submit and `Shift+Enter` to preserve newline behavior
- disabled mode preserves the current `Ctrl/Cmd+Enter` behavior
- IME composition still blocks keyboard-triggered submission

If the project already has UI source tests for composer shortcuts, extend those rather than creating a disconnected test path.

## Non-Goals

- changing default composer behavior for all users
- syncing the preference across browsers or devices
- adding a composer toolbar toggle
- changing backend send semantics

## Acceptance Criteria

The work is successful when:

1. Users can enable a `Press Enter to send` setting from the existing Settings dialog.
2. The setting is off by default.
3. With the setting enabled, `Enter` sends and `Shift+Enter` inserts a newline.
4. With the setting disabled, the current `Ctrl+Enter` / `Cmd+Enter` behavior remains unchanged.
5. The setting persists locally in the browser without any backend changes.
