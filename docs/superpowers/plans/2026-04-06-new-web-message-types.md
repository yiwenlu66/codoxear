# New Web Message Types Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the new web conversation timeline render all legacy-visible message types directly in the main flow.

**Architecture:** Keep the existing single conversation pane, but replace the narrow chat-only filter with a typed event renderer. Chat messages remain bubbles while activity events become dedicated cards backed by richer `MessageEvent` typings.

**Tech Stack:** Preact, TypeScript, Vitest, CSS

---

## File Map

- Modify: `web/src/lib/types.ts` - richer message event fields
- Modify: `web/src/components/conversation/ConversationPane.tsx` - typed event rendering
- Modify: `web/src/components/conversation/ConversationPane.test.tsx` - regression coverage for all supported message types
- Modify: `web/src/styles/global.css` - activity-card styling

### Task 1: Expand the component tests first

**Files:**
- Modify: `web/src/components/conversation/ConversationPane.test.tsx`
- Test: `web/src/components/conversation/ConversationPane.test.tsx`

- [ ] Add failing expectations for `reasoning`, `tool`, `tool_result`, `subagent`, `todo_snapshot`, and system-style events appearing in the main timeline.
- [ ] Run `npm test -- ConversationPane` from `web/` to verify the new assertions fail.

### Task 2: Implement typed event renderers

**Files:**
- Modify: `web/src/lib/types.ts`
- Modify: `web/src/components/conversation/ConversationPane.tsx`

- [ ] Extend `MessageEvent` with the known fields used by Pi activity events.
- [ ] Replace the main-conversation filter with an event-kind helper and per-type renderer functions.
- [ ] Keep chat grouping for consecutive `user`/`assistant`/`ask_user` rows only.
- [ ] Add readable fallback text for partially populated events.

### Task 3: Style the new cards

**Files:**
- Modify: `web/src/styles/global.css`

- [ ] Add a shared activity-card base plus variants for reasoning, tool, tool-result, subagent, todo, and system rows.
- [ ] Keep assistant/user bubbles visually dominant while preserving a coherent timeline rhythm.

### Task 4: Verify focused behavior

**Files:**
- Modify: none unless tests fail
- Test: `web/src/components/conversation/ConversationPane.test.tsx`

- [ ] Run `npm test -- ConversationPane` from `web/` until it passes.
- [ ] Inspect the diff for only the intended message-type rendering changes.

## Self-Review

- Spec coverage: message visibility, renderer structure, styling, and tests all map to explicit tasks.
- Placeholder scan: all files and commands are concrete.
- Type consistency: event names in the plan match the approved design and current backend output.
