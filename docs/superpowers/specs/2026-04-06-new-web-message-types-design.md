# New Web Message Types Design

**Date:** 2026-04-06

## Goal

Restore legacy-style message-type visibility in the new `web/` frontend so the main conversation timeline can show more than plain user/assistant bubbles.

## Approved Direction

Use a single main timeline that renders all legacy-visible event types in-order.

- Keep `user` and `assistant` as chat bubbles.
- Keep `ask_user` visible as an interaction card.
- Render `reasoning`, `tool`, `tool_result`, `subagent`, `todo_snapshot`, and system-style events as dedicated activity cards.
- Do not move these types into a separate tab or debug view.

## Timeline Rules

- Preserve event order from the backend.
- Only group consecutive messages when they are the same chat-style type.
- Break grouping whenever an activity card appears between messages.
- Show all supported message types directly in the main timeline.

## Card Mapping

- `user`: right-aligned bubble
- `assistant`: left-aligned primary bubble
- `ask_user`: left-aligned interaction card with prompt and answer/cancel state
- `reasoning`: visible reasoning card
- `tool`: compact tool call card
- `tool_result`: result card with error state support and expandable-looking layout for long text
- `subagent`: card showing agent, task, output, and pending state
- `todo_snapshot`: progress card with summary plus compact item list
- `pi_session`, `pi_model_change`, `pi_thinking_level_change`, `pi_event`: low-emphasis system cards

## Implementation Notes

- Extend `web/src/lib/types.ts` so renderers can access known event fields without loose `unknown` access.
- Refactor `web/src/components/conversation/ConversationPane.tsx` from a single bubble renderer into a typed event dispatcher.
- Add CSS variants in `web/src/styles/global.css` for activity-card classes while keeping the current chat bubble styling.
- Add regression coverage in `web/src/components/conversation/ConversationPane.test.tsx` for all visible message types.

## Acceptance Criteria

1. The new web timeline no longer filters out `reasoning`, `tool`, `tool_result`, `subagent`, `todo_snapshot`, or supported system events.
2. Each supported message type has a distinct, readable visual treatment.
3. Existing `user`, `assistant`, and `ask_user` rendering keeps working.
4. Focused frontend tests cover the expanded message-type rendering.
