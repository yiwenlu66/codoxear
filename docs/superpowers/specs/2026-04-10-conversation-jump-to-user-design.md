# Conversation Navigation Buttons Design

Date: 2026-04-10

## Goal

When the conversation timeline is scrolled away from the latest content, show floating navigation buttons in the bottom-right corner:
- one button jumps to the nearest earlier rendered `user` message above the current viewport
- one button scrolls back to the bottom of the currently loaded conversation

## User Intent

The user wants faster navigation inside long conversations.

Specifically:
- the controls should live as floating buttons in the bottom-right corner of the conversation area
- the previous-user control should appear after scrolling about one screen upward, not only after a large fixed pixel distance
- both controls may appear at the same time
- both controls should be icon-only, without visible text labels
- clicking the previous-user control should jump to the nearest previous `user` message relative to the current viewport
- clicking the bottom control should return the viewport to the latest currently loaded content

## Current State

`web/src/components/conversation/ConversationPane.tsx` already owns the relevant behavior:
- it renders the full timeline rows
- it manages the scroll container through `.conversationPane`
- it auto-loads older history when the user reaches the top
- it already exposes timeline-level controls such as `Load older` and `Jump to latest`

The current floating affordance only covers previous-user navigation, and it still presents visible text. There is no dedicated floating control for returning to the bottom of the message list.

## Recommended Direction

Use a shared floating navigation cluster inside `ConversationPane` with two independent icon buttons.

This keeps the behavior local to the scrollable conversation surface instead of pushing it into stores or server state. Both buttons depend on the live viewport position, so the simplest and most robust implementation remains DOM-aware logic inside the existing conversation pane.

## Design Overview

### 1. Button Set

Render two separate icon buttons in a shared bottom-right floating stack:
- `previous-user` button with an upward navigation icon
- `scroll-to-bottom` button with a downward navigation icon

They should be visually matched in size, shape, and tone, but each should have its own visibility rule and click behavior.

## 2. Visibility Rules

### Previous-user button

The previous-user button should only be visible when all of the following are true:
- there is an active session with rendered conversation rows
- the pane has been scrolled upward by at least roughly one viewport height
- there is at least one rendered `.messageRow.user` above the current viewport top that can serve as a jump target

This threshold should be based on the pane's visible height when available, with a small fallback fixed value only if the layout height cannot be determined.

### Scroll-to-bottom button

The bottom button should be visible when the user is meaningfully away from the latest content.

Recommended rule:
- show it when the remaining distance from the viewport bottom to the content bottom exceeds a threshold such as half a screen or one compact fixed fallback

This makes the control appear when it is actually useful without constantly lingering while the user is already near the bottom.

## 3. Previous-User Target Selection

The target is the nearest earlier rendered `user` message above the current viewport.

Selection rule:
- inspect rendered `.messageRow.user` elements inside the conversation pane
- compare each row's vertical position against the pane's current scroll position
- choose the last `user` row whose top sits above the current viewport top, with a small tolerance to avoid selecting a row that is already effectively in view

This preserves the requested behavior: one click returns to the closest earlier prompt the user previously sent.

## 4. Click Behavior

### Previous-user button

When clicked:
- scroll the pane so the target `user` row lands near the top of the viewport with a small offset margin
- use smooth scrolling when available
- let the normal scroll listener recompute whether another earlier `user` row is still available after the move

### Scroll-to-bottom button

When clicked:
- smoothly scroll the current pane to its bottom-most rendered position
- do not call `loadInitial`
- do not change the semantics of the existing history control labeled `Jump to latest`

This button is a viewport shortcut, not a history reset action.

## 5. Rendering and Placement

Render the controls inside `ConversationPane` as a floating stack layered over the scroll area.

Visual direction:
- compact rounded icon buttons
- no visible text labels
- clear `aria-label` values for accessibility
- subtle surface treatment that matches the current conversation UI
- enough spacing between the two buttons to keep them easy to tap on mobile
- high enough z-index to remain visible above message cards without covering too much content

The stack should feel like a timeline navigation utility, not a primary action area.

## 6. Scroll/Event Model

The conversation pane already listens for scroll events to trigger older-history loading. The new controls should piggyback on that same scroll container rather than introducing a second scrolling abstraction.

Expected local state:
- whether the previous-user button should be visible
- whether the scroll-to-bottom button should be visible
- the current previous-user jump target, derived from rendered rows rather than persisted externally

State updates should happen when:
- the active session changes
- the message list changes
- the pane scroll position changes

## 7. History Loading Interaction

These controls should operate only against currently rendered rows.

That means:
- the previous-user button should not auto-load older history just to find an older `user` message
- the bottom button should not call `loadInitial`
- existing `Load older` and `Jump to latest` controls remain the explicit history-navigation controls for wider state transitions

This keeps the floating buttons predictable and avoids surprising background data changes.

## Component-Level Changes

### `web/src/components/conversation/ConversationPane.tsx`

Expected changes:
- keep previous-user target detection local to the rendered DOM
- switch the previous-user threshold from a fixed late distance to approximately one viewport height
- add visibility tracking and click handling for a second scroll-to-bottom icon button
- render both controls as an icon-only bottom-right floating stack
- keep explicit `aria-label` text for both buttons

### `web/src/styles/global.css`

Expected changes:
- update the floating control container from a single-button wrapper to a stacked navigation cluster
- keep matching button sizing and spacing for both controls
- preserve desktop and mobile positioning without blocking the composer or history controls

### `web/src/components/conversation/ConversationPane.test.tsx`

Expected additions/updates:
- previous-user button appears after about one screen of upward scrolling
- previous-user button remains hidden when there is no valid earlier `user` target
- scroll-to-bottom button appears when the user is meaningfully away from the bottom
- both buttons can appear at the same time
- clicking the previous-user button still targets the nearest earlier `user` row
- clicking the bottom button scrolls to the bottom of the currently loaded pane

## Interaction Notes

- The controls should not flash on tiny scroll movements.
- Smooth scrolling is preferred, but a direct `scrollTop` fallback is acceptable.
- Keyboard users should be able to tab to both buttons and activate them normally.
- Focus-visible treatment should remain obvious against the conversation background.
- Because the buttons are icon-only, accessibility labels are mandatory.

## Constraints

- preserve the existing conversation layout and row ordering
- do not change message grouping semantics
- do not introduce store-level persistence for this purely local navigation state
- do not trigger implicit older-history loads from the previous-user jump action
- do not reuse the bottom button to change `Jump to latest` semantics

## Out of Scope

- jumping between assistant messages
- multi-step previous/next message navigation UI beyond the existing previous-user behavior
- automatically highlighting destination rows
- session-wide search or message index panels
- redesigning existing top-of-pane history controls

## Testing / Verification Plan

After implementation:
- verify the previous-user button appears after roughly one screen of upward scrolling
- verify the scroll-to-bottom button appears when the viewport is meaningfully away from the bottom
- verify both icon buttons can be shown at the same time
- verify clicking the previous-user button lands on the nearest earlier `user` row
- verify clicking the bottom button scrolls to the bottom of the current pane without calling `loadInitial`
- verify no unexpected history loading occurs
- verify the buttons remain usable on mobile-sized layouts

## Acceptance Criteria

The feature is successful if:
- the bottom-right controls are presented as two independent icon-only buttons
- the previous-user button appears after about one viewport of upward scrolling, not much later
- the bottom button appears when returning to the latest content would be useful
- both controls can coexist cleanly in the same floating area
- the previous-user button still targets the nearest earlier rendered `user` message
- the bottom button smoothly returns the pane to the latest currently rendered content
- the controls coexist cleanly with existing auto-scroll and history-loading behavior
