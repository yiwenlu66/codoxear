# Conversation Navigation Buttons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an icon-only dual-button navigation cluster in the conversation pane so users can jump to the nearest earlier `user` message and scroll back to the bottom of the currently loaded conversation.

**Architecture:** Extend the existing `ConversationPane` scroll-derived UI state so both floating buttons are computed from the same live `.conversationPane` container. Keep all behavior local to the component: one path derives the nearest earlier `user` row, the other computes distance-from-bottom, and both render inside a shared bottom-right stack with focused tests and CSS hooks.

**Tech Stack:** Preact, TypeScript, Vitest, shared `Button` primitive, global CSS

---

## File Structure

### Files to modify
- `web/src/components/conversation/ConversationPane.tsx` — replace the labeled previous-user button with an icon-only version, add bottom-distance detection and a scroll-to-bottom handler, and render both controls in a shared floating stack.
- `web/src/components/conversation/ConversationPane.test.tsx` — update existing previous-user tests for icon-only behavior, add scroll-to-bottom visibility/click coverage, and verify both buttons can coexist.
- `web/src/styles/global.css` — convert the single-button wrapper into a stacked floating navigation cluster and give both icon buttons stable tap targets on desktop/mobile.
- `web/src/styles/layout-scroll.test.ts` — update the CSS contract checks from a single-button wrapper to the stacked navigation cluster.
- `docs/superpowers/plans/2026-04-10-conversation-navigation-buttons.md` — durable plan copy saved after approval.

### Files to leave untouched
- `web/src/components/ui/scroll-area.tsx` — existing wrapper already exposes the correct scroll container.
- `web/src/components/ui/button.tsx` — existing variants are enough for icon-only controls.
- message/session stores — this remains viewport-local UI state.

### Dirty worktree note
- The repository already contains unrelated uncommitted changes, including files touched by current web work. Do not revert or stage unrelated changes. If a commit is requested later, stage only the files listed above unless the user explicitly says otherwise.

## Task 1: Update `ConversationPane` tests first

**Files:**
- Modify: `web/src/components/conversation/ConversationPane.test.tsx`
- Test: `web/src/components/conversation/ConversationPane.test.tsx`

- [ ] **Step 1: Update the existing previous-user visibility test to expect an icon-only button with an accessibility label**

In the current test named `shows a floating previous-user button only after scrolling above an earlier user message`, replace the final assertions with:

```tsx
    const jumpButton = root.querySelector("[data-testid='jump-to-previous-user']") as HTMLButtonElement | null;
    expect(jumpButton).not.toBeNull();
    expect(jumpButton?.textContent).toBe("");
    expect(jumpButton?.getAttribute("aria-label")).toBe("Jump to previous user message");
```

- [ ] **Step 2: Run the focused file and verify the updated expectation fails**

Run:

```bash
cd web && pnpm exec vitest run src/components/conversation/ConversationPane.test.tsx
```

Expected: FAIL because the button still renders visible text (`上一条提问`).

- [ ] **Step 3: Add a failing test that shows the bottom button when the pane is away from the latest content**

Append this test after the previous-user tests:

```tsx
  it("shows a scroll-to-bottom button when the pane is away from the latest content", async () => {
    const sessionsStore = createStaticStore(
      { items: [{ session_id: "sess-bottom", agent_backend: "pi" }], activeSessionId: "sess-bottom", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-bottom": [
            { role: "user", text: "Question 1" },
            { role: "assistant", text: "Answer 1" },
            { role: "assistant", text: "Answer 2" },
            { role: "assistant", text: "Answer 3" },
          ],
        },
        offsetsBySessionId: { "sess-bottom": 4 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const pane = root.querySelector(".conversationPane") as HTMLDivElement | null;
    Object.defineProperty(pane!, "clientHeight", { configurable: true, value: 360 });
    Object.defineProperty(pane!, "scrollHeight", { configurable: true, value: 1500 });
    Object.defineProperty(pane!, "scrollTop", { configurable: true, writable: true, value: 620 });

    await act(async () => {
      pane?.dispatchEvent(new Event("scroll"));
      await Promise.resolve();
    });

    const bottomButton = root.querySelector("[data-testid='scroll-to-bottom']") as HTMLButtonElement | null;
    expect(bottomButton).not.toBeNull();
    expect(bottomButton?.textContent).toBe("");
    expect(bottomButton?.getAttribute("aria-label")).toBe("Scroll to conversation bottom");
  });
```

- [ ] **Step 4: Add a failing test that both buttons can appear together and the bottom button scrolls to the pane bottom**

Append this test after the bottom-button visibility test:

```tsx
  it("shows both floating buttons together and scrolls to the bottom when requested", async () => {
    const sessionsStore = createStaticStore(
      { items: [{ session_id: "sess-both", agent_backend: "pi" }], activeSessionId: "sess-both", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-both": [
            { role: "user", text: "Question 1" },
            { role: "assistant", text: "Answer 1" },
            { role: "user", text: "Question 2" },
            { role: "assistant", text: "Answer 2" },
            { role: "assistant", text: "Answer 3" },
            { role: "assistant", text: "Newest answer" },
          ],
        },
        offsetsBySessionId: { "sess-both": 6 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );

    const scrollTo = vi.fn();

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const pane = root.querySelector(".conversationPane") as HTMLDivElement | null;
    Object.defineProperty(pane!, "scrollTo", { configurable: true, value: scrollTo });
    Object.defineProperty(pane!, "clientHeight", { configurable: true, value: 360 });
    Object.defineProperty(pane!, "scrollHeight", { configurable: true, value: 1800 });
    Object.defineProperty(pane!, "scrollTop", { configurable: true, writable: true, value: 700 });
    const rows = Array.from(root.querySelectorAll<HTMLElement>(".messageRow"));
    rows.forEach((row, index) => {
      Object.defineProperty(row, "offsetTop", { configurable: true, value: index * 220 });
    });

    await act(async () => {
      pane?.dispatchEvent(new Event("scroll"));
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(root.querySelector("[data-testid='jump-to-previous-user']")).not.toBeNull();
    const bottomButton = root.querySelector("[data-testid='scroll-to-bottom']") as HTMLButtonElement | null;
    expect(bottomButton).not.toBeNull();

    await act(async () => {
      bottomButton?.click();
      await Promise.resolve();
    });

    expect(scrollTo).toHaveBeenCalledWith({ top: 1800, behavior: "smooth" });
  });
```

- [ ] **Step 5: Run the focused file again to confirm the new tests are red for the correct reasons**

Run:

```bash
cd web && pnpm exec vitest run src/components/conversation/ConversationPane.test.tsx
```

Expected: FAIL because the previous-user button still renders text, there is no bottom button yet, and no combined floating cluster exists.

## Task 2: Extend `ConversationPane` for dual icon-only buttons

**Files:**
- Modify: `web/src/components/conversation/ConversationPane.tsx`
- Test: `web/src/components/conversation/ConversationPane.test.tsx`

- [ ] **Step 1: Add icon helpers and bottom-distance helpers near the existing scroll helpers**

After the current `ArrowUpTurnIcon`, add a downward icon and a helper to decide when the bottom button should show:

```tsx
function ArrowDownIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M8 3v10" />
      <path d="m5 10 3 3 3-3" />
    </svg>
  );
}

function shouldShowScrollToBottom(pane: HTMLElement): boolean {
  const distanceFromBottom = pane.scrollHeight - (pane.scrollTop + pane.clientHeight);
  const threshold = pane.clientHeight > 0 ? Math.max(160, Math.round(pane.clientHeight * 0.5)) : 180;
  return distanceFromBottom > threshold;
}
```

- [ ] **Step 2: Track both visibility states together**

Change the local state section from a single flag to two flags:

```tsx
  const [showPreviousUserJump, setShowPreviousUserJump] = useState(false);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
```

Then replace `recomputePreviousUserJump` with:

```tsx
  const recomputeFloatingNavigation = () => {
    const pane = sectionRef.current?.querySelector(".conversationPane") as HTMLElement | null;
    if (!pane || !activeSessionId) {
      setShowPreviousUserJump(false);
      setShowScrollToBottom(false);
      return;
    }
    setShowPreviousUserJump(Boolean(findPreviousUserRow(pane)));
    setShowScrollToBottom(shouldShowScrollToBottom(pane));
  };
```

- [ ] **Step 3: Update all existing recompute/reset call sites**

In `useLayoutEffect`, replace `setShowPreviousUserJump(false)` with resetting both flags and replace `recomputePreviousUserJump()` with `recomputeFloatingNavigation()`.

In the scroll listener, replace the single previous-user update with:

```tsx
      setShowPreviousUserJump(Boolean(findPreviousUserRow(pane)));
      setShowScrollToBottom(shouldShowScrollToBottom(pane));
```

- [ ] **Step 4: Add the scroll-to-bottom handler**

Insert this next to `handleJumpToPreviousUser`:

```tsx
  const handleScrollToBottom = () => {
    const pane = sectionRef.current?.querySelector(".conversationPane") as HTMLElement | null;
    if (!pane) return;
    scrollPaneToPosition(pane, pane.scrollHeight);
  };
```

- [ ] **Step 5: Make the previous-user button icon-only and render both buttons in one cluster**

Replace the current floating button block with:

```tsx
      {showPreviousUserJump || showScrollToBottom ? (
        <div className="conversationNavButtons">
          {showPreviousUserJump ? (
            <Button
              data-testid="jump-to-previous-user"
              type="button"
              variant="secondary"
              size="icon"
              className="conversationJumpButton shadow-lg"
              onClick={handleJumpToPreviousUser}
              aria-label="Jump to previous user message"
            >
              <ArrowUpTurnIcon />
            </Button>
          ) : null}
          {showScrollToBottom ? (
            <Button
              data-testid="scroll-to-bottom"
              type="button"
              variant="secondary"
              size="icon"
              className="conversationJumpButton shadow-lg"
              onClick={handleScrollToBottom}
              aria-label="Scroll to conversation bottom"
            >
              <ArrowDownIcon />
            </Button>
          ) : null}
        </div>
      ) : null}
```

This intentionally reuses the same button visual class and removes the visible `span` label from the previous-user control.

- [ ] **Step 6: Run the focused conversation tests and make sure they pass**

Run:

```bash
cd web && pnpm exec vitest run src/components/conversation/ConversationPane.test.tsx
```

Expected: PASS for the updated icon-only previous-user tests, the new bottom-button tests, and all existing `ConversationPane` tests.

## Task 3: Convert the floating wrapper into a stacked navigation cluster

**Files:**
- Modify: `web/src/styles/global.css`
- Modify: `web/src/styles/layout-scroll.test.ts`
- Test: `web/src/styles/layout-scroll.test.ts`

- [ ] **Step 1: Replace the single-button wrapper CSS with a stacked cluster**

In `web/src/styles/global.css`, replace:
- `.conversationJumpButtonWrap`
- `.conversationJumpButton svg`

with the following block while keeping the existing hover/focus tone rules:

```css
.conversationNavButtons {
  position: absolute;
  right: 18px;
  bottom: 18px;
  z-index: 20;
  display: flex;
  flex-direction: column;
  gap: 10px;
  pointer-events: none;
}

.conversationJumpButton {
  width: 42px;
  height: 42px;
  padding: 0;
  pointer-events: auto;
  border: 1px solid hsl(var(--border) / 0.72);
  background: hsl(var(--background) / 0.92);
  color: hsl(var(--foreground));
  backdrop-filter: blur(14px);
}

.conversationJumpButton svg {
  width: 16px;
  height: 16px;
}
```

Keep the existing `.conversationJumpButton:hover` and `.conversationJumpButton:focus-visible` styles.

- [ ] **Step 2: Update the mobile spacing rules**

Inside `@media (max-width: 880px)`, replace the current `.conversationJumpButtonWrap` rules with:

```css
  .conversationNavButtons {
    right: 12px;
    bottom: 12px;
    gap: 8px;
  }

  .conversationJumpButton {
    width: 40px;
    height: 40px;
  }
```

- [ ] **Step 3: Update the CSS contract test to match the new cluster**

Replace the current floating-button CSS test in `web/src/styles/layout-scroll.test.ts` with:

```ts
  it("positions the conversation navigation buttons as a bottom-right stacked overlay", () => {
    const timelineRule = ruleBody(css, ".conversationTimeline");
    const navRule = ruleBody(css, ".conversationNavButtons");
    const jumpButtonRule = ruleBody(css, ".conversationJumpButton");
    const mobileRules = mediaBody(css, "(max-width: 880px)");
    const mobileNavRule = ruleBody(mobileRules, ".conversationNavButtons");

    expect(timelineRule).toContain("position: relative;");
    expect(navRule).toContain("position: absolute;");
    expect(navRule).toContain("display: flex;");
    expect(navRule).toContain("flex-direction: column;");
    expect(navRule).toContain("right: 18px;");
    expect(navRule).toContain("bottom: 18px;");
    expect(navRule).toContain("pointer-events: none;");
    expect(jumpButtonRule).toContain("width: 42px;");
    expect(jumpButtonRule).toContain("height: 42px;");
    expect(jumpButtonRule).toContain("pointer-events: auto;");
    expect(mobileNavRule).toContain("right: 12px;");
    expect(mobileNavRule).toContain("bottom: 12px;");
  });
```

- [ ] **Step 4: Run the CSS contract test together with the conversation tests**

Run:

```bash
cd web && pnpm exec vitest run src/styles/layout-scroll.test.ts src/components/conversation/ConversationPane.test.tsx
```

Expected: PASS for both files.

## Task 4: Save the durable plan copy and run full verification

**Files:**
- Write: `docs/superpowers/plans/2026-04-10-conversation-navigation-buttons.md`
- Read/verify: `web/src/components/conversation/ConversationPane.tsx`
- Read/verify: `web/src/components/conversation/ConversationPane.test.tsx`
- Read/verify: `web/src/styles/global.css`
- Read/verify: `web/src/styles/layout-scroll.test.ts`

- [ ] **Step 1: Save the approved plan copy**

Copy the approved plan into:

```text
docs/superpowers/plans/2026-04-10-conversation-navigation-buttons.md
```

- [ ] **Step 2: Run the full web verification suite**

Run:

```bash
cd web && pnpm test
```

Expected: TypeScript test config passes, then Vitest reports all tests passing with exit code 0.

- [ ] **Step 3: Inspect the scoped diff for this feature only**

Run:

```bash
git diff -- web/src/components/conversation/ConversationPane.tsx web/src/components/conversation/ConversationPane.test.tsx web/src/styles/global.css web/src/styles/layout-scroll.test.ts docs/superpowers/plans/2026-04-10-conversation-navigation-buttons.md
```

Expected: Diff contains only the dual icon-only navigation button changes and the durable plan file.

- [ ] **Step 4: Re-check dirty worktree status before any later commit**

Run:

```bash
git status --short
```

Expected: There will likely still be unrelated dirty files. Do not stage or revert them.

- [ ] **Step 5: Report verification evidence**

In the final implementation response, include:
- the files changed for this navigation update
- the focused and full verification commands run
- whether `cd web && pnpm test` passed or what failed
- the reminder that unrelated dirty worktree changes remain present
