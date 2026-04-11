# Session List Compact Design

Date: 2026-04-09

## Goal

Reduce the visual size of the session list items so the sidebar reads like a compact desktop file/session list instead of a stack of large cards, while keeping the most useful information visible at a glance.

## User Intent

The current session items feel too large and card-like.

The user wants a denser list treatment closer to Finder or VS Code:
- preserve backend and busy status visibility
- preserve a small amount of preview/context text
- replace text action buttons with hover-only icon actions

## Current State

The current implementation in `web/src/components/sessions/SessionCard.tsx` and `web/src/styles/global.css` creates a card-heavy presentation:
- each item is wrapped in a full `Card`
- spacing, radii, and shadows make each session visually tall
- title can span two lines
- preview can span two lines
- actions appear as full text buttons below the content
- grouped session shells are also rounded and padded, which amplifies the stacked-card feel

This makes the session pane feel heavier and longer than necessary, especially when many sessions are present.

## Recommended Direction

Adopt a compact two-line list item.

This keeps the UI readable while significantly reducing height and visual mass. It also preserves the information the user explicitly wants to keep, without collapsing everything into a single cramped row.

## Design Overview

### 1. Item Structure

Each session item becomes a compact list row with two text lines:

- first line: title, status/backend cluster, hover actions
- second line: short preview/context, with optional short session id at the end if space allows

The overall item should read as a list entry rather than a standalone card.

### 2. Visual Treatment

Session items should move from elevated cards to light rows:
- reduce outer padding substantially
- reduce corner radius
- remove or greatly soften box shadows
- reduce border prominence
- use subtle background changes for hover

The goal is to make the session list feel continuous and scannable.

### 3. Active State

The active session should remain easy to identify without restoring the heavy card style.

Recommended active treatment:
- a subtle tinted background
- a thin left accent bar
- slightly stronger text contrast
- optional modest border tint

Avoid thick rings, strong elevation, or large glowing highlights.

### 4. Information Density

To reduce height while keeping meaning:
- title becomes single-line with ellipsis
- preview becomes single-line with ellipsis
- backend badge becomes smaller and quieter
- busy indicator dot becomes smaller
- owner badge becomes lower emphasis than backend and should only remain if it still fits cleanly
- queue badge remains visible when present, but should be smaller than today

This preserves the important signals while cutting down visual clutter.

### 5. Actions

Replace the current text action row (`Edit`, `Duplicate`, `Delete`) with icon-only actions.

Behavior:
- hidden by default on desktop-sized pointer devices
- revealed on hover for inactive rows
- visible on active rows as needed for discoverability
- aligned to the right side of the first row

Styling:
- small ghost buttons
- compact hit targets, but still clickable
- low emphasis until hover/focus
- no second full action row beneath the item

This is the biggest layout change for reducing item height.

### 6. Group Container Tone-Down

The user asked specifically about session list items, but the group shell styling contributes to the oversized feeling. To support the compact rows:
- slightly reduce group padding
- soften the group shell radius
- reduce gap between items
- keep group hierarchy visible, but less decorative

This should be a supporting change, not a redesign of group behavior.

## Component-Level Changes

### `web/src/components/sessions/SessionCard.tsx`

Expected changes:
- remove the visual dependence on a large card presentation
- restructure the markup so actions live inline in the top row
- swap text buttons for icon buttons
- ensure active and hover states map cleanly to row-style presentation
- keep existing actions and selection semantics intact

No behavioral changes are intended for selecting, editing, duplicating, or deleting.

### `web/src/styles/global.css`

Expected changes:
- reduce `sessionCard`, `sessionCardSurface`, and `sessionCardContent` visual weight
- compress `compactSessionButton` vertical spacing
- make `sessionTitle` and `sessionPreview` single-line
- scale down badges and metadata spacing
- add inline icon action styles for the new compact action cluster
- reduce `sessionGroupList` gaps and slightly tone down `sessionGroupShell`

## Interaction Notes

- Hover should make the row feel interactive without making it jumpy.
- Focus-visible styles must remain clear for keyboard users.
- Hover-only actions should still be reachable by keyboard focus.
- Active row actions should not require precise pointer hover to discover.

## Constraints

- keep the existing grouping model
- keep the existing session metadata semantics
- avoid changing backend or store logic
- keep mobile behavior functional even if hover interactions are desktop-first

## Out of Scope

- changing session grouping rules
- changing session sorting rules
- changing new-session flows
- redesigning the entire sidebar
- removing metadata entirely

## Testing / Verification Plan

After implementation:
- verify session items are visibly shorter in the sidebar
- verify title, backend, busy state, and preview remain readable
- verify hover reveals icon actions on desktop
- verify active item styling remains obvious
- verify keyboard focus still exposes actionable controls
- verify the layout still works on narrow/mobile widths

## Acceptance Criteria

The redesign is successful if:
- the session list feels meaningfully denser than the current card stack
- each item reads as a compact list row rather than a mini card
- backend/status and one line of preview remain visible
- action controls no longer consume a dedicated text-button row
- active and hover states remain clear and usable
