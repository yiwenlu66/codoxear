# Mobile Conversation-First UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the web UI mobile experience so phone users can quickly open a session, send a prompt, and read replies without desktop-style controls getting in the way.

**Architecture:** Keep the existing desktop shell intact, but add a mobile-first toolbar and mobile tools sheet in `AppShell`, tighten the session list into a denser directory-style layout, and rebalance conversation/composer styling for small screens. Keep existing dialogs and sheets, but route mobile entry points through simpler controls.

**Tech Stack:** Preact, Vitest, Vite, Tailwind utility classes plus `web/src/styles/global.css`

---

### Task 1: Cover the new mobile shell behavior with tests

**Files:**
- Modify: `web/src/app/AppShell.test.tsx`
- Modify: `web/src/components/sessions/SessionsPane.test.tsx`

- [ ] Add a test that renders the shell and checks for a mobile-friendly tools trigger alongside the sessions trigger.
- [ ] Add a test that opening the tools trigger reveals grouped mobile actions for files, workspace, and harness.
- [ ] Add a test that session cards expose the compact metadata structure needed for the denser mobile list.
- [ ] Run `npm test -- --runInBand web/src/app/AppShell.test.tsx web/src/components/sessions/SessionsPane.test.tsx` from `web/` and confirm the new assertions fail before implementation.

### Task 2: Implement the mobile-first shell controls

**Files:**
- Modify: `web/src/app/AppShell.tsx`

- [ ] Add mobile-only state and handlers for a tools sheet without regressing desktop toolbar actions.
- [ ] Keep the existing sessions sheet, but reduce toolbar clutter by introducing a single mobile tools entry point.
- [ ] Route mobile actions from the tools sheet to the existing file viewer, workspace sheet/dialog, and harness dialog.
- [ ] Keep existing desktop buttons available above the mobile breakpoint.

### Task 3: Compact the session list and improve message/composer density

**Files:**
- Modify: `web/src/components/sessions/SessionCard.tsx`
- Modify: `web/src/components/composer/Composer.tsx` (if markup hooks are needed)
- Modify: `web/src/styles/global.css`

- [ ] Tighten the session card layout into a denser title + preview + compact metadata format.
- [ ] Make the active session visually clear without relying on heavy card chrome.
- [ ] Reduce mobile conversation padding, weaken non-chat system/tool surfaces, and keep the composer anchored and single-thumb friendly.
- [ ] Hide or de-emphasize secondary composer controls on narrow screens while preserving desktop behavior.

### Task 4: Verify the change end-to-end

**Files:**
- Verify only

- [ ] Run `npm test -- --runInBand web/src/app/AppShell.test.tsx web/src/components/sessions/SessionsPane.test.tsx web/src/components/composer/Composer.test.tsx web/src/components/conversation/ConversationPane.test.tsx` from `web/`.
- [ ] Run `npm run test` from `web/` if the focused suite passes.
- [ ] If needed, run `npm run build` from `web/` to catch responsive build regressions before reporting completion.
