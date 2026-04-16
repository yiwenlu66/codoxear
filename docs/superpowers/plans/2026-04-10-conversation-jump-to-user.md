# Conversation Jump-To-User Button Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bottom-right floating button in the conversation pane that appears after scrolling upward and jumps to the nearest earlier rendered `user` message above the current viewport.

**Architecture:** Keep the feature local to `ConversationPane` by deriving the jump target from rendered `.messageRow.user` DOM nodes inside the existing `.conversationPane` scroll container. Extend the current scroll listener to recompute button visibility, render a floating control layered over the conversation shell, and cover the behavior with focused component tests plus a small CSS contract check.

**Tech Stack:** Preact, TypeScript, Vitest, shared `Button` primitive, global CSS

---

## File Structure

### Files to modify
- `web/src/components/conversation/ConversationPane.tsx` — add previous-user target detection, scroll-driven button visibility, jump handler, and floating button markup.
- `web/src/components/conversation/ConversationPane.test.tsx` — add tests for visibility, target selection, repeated stepping, and session reset.
- `web/src/styles/global.css` — add bottom-right floating control styles and relative positioning for the conversation shell.
- `web/src/styles/layout-scroll.test.ts` — lock in the new CSS positioning hooks for the floating conversation control.

### Files to leave untouched
- `web/src/components/ui/scroll-area.tsx` — existing wrapper already exposes the correct scroll container.
- `web/src/components/ui/button.tsx` — existing variants are enough for the floating affordance.
- message/session stores — no persistence or API changes are needed because this is purely viewport-local UI state.

### Durable plan copy
- After plan approval and before implementation edits, save this plan to `docs/superpowers/plans/2026-04-10-conversation-jump-to-user.md` so the repository keeps a reviewable implementation record.

## Task 1: Lock the jump-button behavior in tests first

**Files:**
- Modify: `web/src/components/conversation/ConversationPane.test.tsx`
- Test: `web/src/components/conversation/ConversationPane.test.tsx`

- [ ] **Step 1: Add a failing test that the floating button appears only when an earlier `user` row exists above the viewport**

Append this test near the existing scroll/history tests:

```tsx
  it("shows a floating previous-user button only after scrolling above an earlier user message", async () => {
    const sessionsStore = createStaticStore(
      { items: [{ session_id: "sess-jump", agent_backend: "pi" }], activeSessionId: "sess-jump", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-jump": [
            { role: "user", text: "First question" },
            { role: "assistant", text: "First answer" },
            { role: "user", text: "Second question" },
            { role: "assistant", text: "Second answer" },
            { role: "assistant", text: "Newest answer" },
          ],
        },
        offsetsBySessionId: { "sess-jump": 5 },
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
    const rows = Array.from(root.querySelectorAll<HTMLElement>(".messageRow"));
    expect(pane).not.toBeNull();
    expect(root.querySelector("[data-testid='jump-to-previous-user']")).toBeNull();

    rows.forEach((row, index) => {
      Object.defineProperty(row, "offsetTop", { configurable: true, value: index * 180 });
    });
    Object.defineProperty(pane!, "scrollTop", { configurable: true, writable: true, value: 410 });

    await act(async () => {
      pane?.dispatchEvent(new Event("scroll"));
      await Promise.resolve();
    });

    const jumpButton = root.querySelector("[data-testid='jump-to-previous-user']") as HTMLButtonElement | null;
    expect(jumpButton).not.toBeNull();
    expect(jumpButton?.textContent).toContain("上一条提问");
  });
```

- [ ] **Step 2: Run the focused test file and verify the new assertion fails**

Run:

```bash
cd web && pnpm exec vitest run src/components/conversation/ConversationPane.test.tsx
```

Expected: FAIL because `ConversationPane` does not yet render a floating jump button or recompute one on scroll.

- [ ] **Step 3: Add a second failing test that clicking the button targets the nearest earlier `user` row and can step backward again after the pane scroll position changes**

Append this test after the visibility test:

```tsx
  it("jumps to the nearest earlier rendered user message and can step backward again", async () => {
    const sessionsStore = createStaticStore(
      { items: [{ session_id: "sess-jump-target", agent_backend: "pi" }], activeSessionId: "sess-jump-target", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const loadOlder = vi.fn().mockResolvedValue(undefined);
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-jump-target": [
            { role: "user", text: "Question 1" },
            { role: "assistant", text: "Answer 1" },
            { role: "user", text: "Question 2" },
            { role: "assistant", text: "Answer 2" },
            { role: "assistant", text: "Answer 3" },
          ],
        },
        offsetsBySessionId: { "sess-jump-target": 5 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder },
    );

    const scrollTo = vi.fn();
    const originalScrollTo = HTMLElement.prototype.scrollTo;
    HTMLElement.prototype.scrollTo = scrollTo;

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const pane = root.querySelector(".conversationPane") as HTMLDivElement | null;
    const rows = Array.from(root.querySelectorAll<HTMLElement>(".messageRow"));
    rows.forEach((row, index) => {
      Object.defineProperty(row, "offsetTop", { configurable: true, value: index * 160 });
    });
    Object.defineProperty(pane!, "scrollTop", { configurable: true, writable: true, value: 520 });

    await act(async () => {
      pane?.dispatchEvent(new Event("scroll"));
      await Promise.resolve();
    });

    const jumpButton = root.querySelector("[data-testid='jump-to-previous-user']") as HTMLButtonElement | null;
    await act(async () => {
      jumpButton?.click();
      await Promise.resolve();
    });

    expect(scrollTo).toHaveBeenCalledWith({ top: 304, behavior: "smooth" });
    expect(loadOlder).not.toHaveBeenCalled();

    pane!.scrollTop = 304;
    await act(async () => {
      pane?.dispatchEvent(new Event("scroll"));
      await Promise.resolve();
    });

    await act(async () => {
      jumpButton?.click();
      await Promise.resolve();
    });

    expect(scrollTo).toHaveBeenCalledWith({ top: 0, behavior: "smooth" });
    HTMLElement.prototype.scrollTo = originalScrollTo;
  });
```

This expects the second `user` row (`offsetTop === 320`) with `16px` top padding on the first click, then the first user row on the next click after the pane position changes.

- [ ] **Step 4: Add a third failing test that switching sessions clears the floating control state**

Append this test after the target-selection test:

```tsx
  it("clears the previous-user jump button when switching to a session without earlier user rows", async () => {
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-a": [
            { role: "user", text: "Question A" },
            { role: "assistant", text: "Answer A" },
            { role: "assistant", text: "Answer A2" },
          ],
          "sess-b": [{ role: "assistant", text: "Only answer" }],
        },
        offsetsBySessionId: { "sess-a": 3, "sess-b": 1 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders
        sessionsStore={createStaticStore(
          { items: [], activeSessionId: "sess-a", loading: false, newSessionDefaults: null },
          { refresh: () => Promise.resolve(), select: () => undefined },
        ) as any}
        messagesStore={messagesStore as any}
      >
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const pane = root.querySelector(".conversationPane") as HTMLDivElement | null;
    const rows = Array.from(root.querySelectorAll<HTMLElement>(".messageRow"));
    rows.forEach((row, index) => {
      Object.defineProperty(row, "offsetTop", { configurable: true, value: index * 180 });
    });
    Object.defineProperty(pane!, "scrollTop", { configurable: true, writable: true, value: 360 });

    await act(async () => {
      pane?.dispatchEvent(new Event("scroll"));
      await Promise.resolve();
    });
    expect(root.querySelector("[data-testid='jump-to-previous-user']")).not.toBeNull();

    await act(async () => {
      render(
        <AppProviders
          sessionsStore={createStaticStore(
            { items: [], activeSessionId: "sess-b", loading: false, newSessionDefaults: null },
            { refresh: () => Promise.resolve(), select: () => undefined },
          ) as any}
          messagesStore={messagesStore as any}
        >
          <ConversationPane />
        </AppProviders>,
        root!,
      );
      await Promise.resolve();
    });

    expect(root.querySelector("[data-testid='jump-to-previous-user']")).toBeNull();
  });
```

- [ ] **Step 5: Run the focused test file again to confirm the suite is red for the right reasons**

Run:

```bash
cd web && pnpm exec vitest run src/components/conversation/ConversationPane.test.tsx
```

Expected: FAIL only in the new tests because the floating button, nearest-target selection, repeated stepping, and session-reset behavior do not exist yet.

- [ ] **Step 6: Commit the red test stage in the isolated worktree**

```bash
git add web/src/components/conversation/ConversationPane.test.tsx
git commit -m "test: define previous-user jump behavior"
```

If not working in an isolated worktree because the existing checkout contains unrelated user changes, skip this commit and keep the staged file list limited to `web/src/components/conversation/ConversationPane.test.tsx`.

## Task 2: Implement the viewport-aware jump logic in `ConversationPane`

**Files:**
- Modify: `web/src/components/conversation/ConversationPane.tsx`
- Test: `web/src/components/conversation/ConversationPane.test.tsx`

- [ ] **Step 1: Import the shared button primitive**

Add this import near the existing UI imports:

```tsx
import { Button } from "@/components/ui/button";
```

- [ ] **Step 2: Add constants, helper functions, and the icon near the existing scroll helpers**

Insert this block after `scrollPaneToBottom`:

```tsx
const PREVIOUS_USER_BUTTON_SCROLL_THRESHOLD = 320;
const PREVIOUS_USER_TARGET_TOLERANCE = 24;
const PREVIOUS_USER_SCROLL_TOP_PADDING = 16;

function ArrowUpTurnIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M5 6 8 3l3 3" />
      <path d="M8 3v7" />
      <path d="M8 10h3.5A2.5 2.5 0 0 1 14 12.5" />
    </svg>
  );
}

function findPreviousUserRow(pane: HTMLElement): HTMLElement | null {
  const threshold = pane.scrollTop - PREVIOUS_USER_TARGET_TOLERANCE;
  if (threshold < PREVIOUS_USER_BUTTON_SCROLL_THRESHOLD) {
    return null;
  }

  const rows = Array.from(pane.querySelectorAll<HTMLElement>(".messageRow.user"));
  let candidate: HTMLElement | null = null;
  for (const row of rows) {
    if (row.offsetTop <= threshold) {
      candidate = row;
      continue;
    }
    break;
  }
  return candidate;
}

function scrollPaneToPosition(element: HTMLElement, top: number) {
  const nextTop = Math.max(0, top);
  if (typeof element.scrollTo === "function") {
    element.scrollTo({ top: nextTop, behavior: "smooth" });
    return;
  }
  element.scrollTop = nextTop;
}
```

- [ ] **Step 3: Add local state and a recompute helper inside `ConversationPane`**

Add this after the existing refs in the component:

```tsx
  const [showPreviousUserJump, setShowPreviousUserJump] = useState(false);

  const recomputePreviousUserJump = () => {
    const pane = sectionRef.current?.querySelector(".conversationPane") as HTMLElement | null;
    if (!pane || !activeSessionId) {
      setShowPreviousUserJump(false);
      return;
    }
    setShowPreviousUserJump(Boolean(findPreviousUserRow(pane)));
  };
```

- [ ] **Step 4: Extend the layout effect so render changes reset or recompute the button state**

In the existing `useLayoutEffect`, keep the scroll-preservation and auto-bottom behavior, but add `setShowPreviousUserJump(false)` for missing/empty panes and call the helper after scroll changes:

```tsx
  useLayoutEffect(() => {
    const pane = sectionRef.current?.querySelector(".conversationPane") as HTMLElement | null;
    if (!pane || (!messages.length && !isBusy)) {
      setShowPreviousUserJump(false);
      return;
    }

    if (scrollModeRef.current === "preserve") {
      const anchor = historyAnchorRef.current;
      if (anchor) {
        const anchorRow = pane.querySelector(`[data-row-key="${anchor.key}"]`) as HTMLElement | null;
        if (anchorRow) {
          pane.scrollTop = Math.max(0, anchorRow.offsetTop - anchor.top);
        }
      }
      historyAnchorRef.current = null;
      scrollModeRef.current = null;
      recomputePreviousUserJump();
      return;
    }

    scrollPaneToBottom(pane);
    scrollModeRef.current = null;
    recomputePreviousUserJump();
  }, [messages.length, activeSessionId, isBusy]);
```

- [ ] **Step 5: Extend the existing scroll listener to update the button state without changing history loading**

Inside the existing scroll `useEffect`, update `onScroll` to:

```tsx
    const onScroll = () => {
      if (pane.scrollTop <= 12 && !olderLoading && (hasOlder || olderCursor > 0)) {
        void handleLoadOlder();
      }
      setShowPreviousUserJump(Boolean(findPreviousUserRow(pane)));
    };
```

- [ ] **Step 6: Add the click handler inside `ConversationPane`**

Add this near `handleJumpToLatest`:

```tsx
  const handleJumpToPreviousUser = () => {
    const pane = sectionRef.current?.querySelector(".conversationPane") as HTMLElement | null;
    if (!pane) return;
    const target = findPreviousUserRow(pane);
    if (!target) {
      setShowPreviousUserJump(false);
      return;
    }
    scrollPaneToPosition(pane, target.offsetTop - PREVIOUS_USER_SCROLL_TOP_PADDING);
  };
```

- [ ] **Step 7: Render the bottom-right floating button as a sibling of the scroll area**

Change the root `section` class from `"conversationTimeline flex min-h-0 flex-1"` to `"conversationTimeline relative flex min-h-0 flex-1"`, keep the existing `ScrollArea` body unchanged, and append this block immediately after `</ScrollArea>`:

```tsx
      {showPreviousUserJump ? (
        <div className="conversationJumpButtonWrap">
          <Button
            data-testid="jump-to-previous-user"
            type="button"
            variant="secondary"
            size="sm"
            className="conversationJumpButton rounded-full px-3 shadow-lg"
            onClick={handleJumpToPreviousUser}
            aria-label="Jump to previous user message"
          >
            <ArrowUpTurnIcon />
            <span>上一条提问</span>
          </Button>
        </div>
      ) : null}
```

- [ ] **Step 8: Run the focused conversation test file and make sure the new tests pass**

Run:

```bash
cd web && pnpm exec vitest run src/components/conversation/ConversationPane.test.tsx
```

Expected: PASS for the new floating-button tests and all pre-existing `ConversationPane` tests.

- [ ] **Step 9: Commit the green component stage in the isolated worktree**

```bash
git add web/src/components/conversation/ConversationPane.tsx web/src/components/conversation/ConversationPane.test.tsx
git commit -m "feat: add previous-user jump control"
```

If the checkout contains unrelated user changes, skip this commit and record the exact modified files instead.

## Task 3: Add the floating control styles and CSS contract coverage

**Files:**
- Modify: `web/src/styles/global.css`
- Modify: `web/src/styles/layout-scroll.test.ts`
- Test: `web/src/styles/layout-scroll.test.ts`

- [ ] **Step 1: Add the floating control styles next to the conversation layout rules**

Insert these rules after `.conversationPane.emptyState .messageList` and before `.messageList`:

```css
.conversationTimeline {
  position: relative;
}

.conversationJumpButtonWrap {
  position: absolute;
  right: 18px;
  bottom: 18px;
  z-index: 20;
  pointer-events: none;
}

.conversationJumpButton {
  pointer-events: auto;
  border: 1px solid hsl(var(--border) / 0.72);
  background: hsl(var(--background) / 0.92);
  color: hsl(var(--foreground));
  backdrop-filter: blur(14px);
}

.conversationJumpButton:hover,
.conversationJumpButton:focus-visible {
  background: hsl(var(--accent) / 0.96);
  color: hsl(var(--accent-foreground));
}

.conversationJumpButton svg {
  width: 16px;
  height: 16px;
}
```

- [ ] **Step 2: Add mobile-specific spacing so the button does not crowd narrow layouts**

Inside the existing `@media (max-width: 880px)` block, add:

```css
  .conversationJumpButtonWrap {
    right: 12px;
    bottom: 12px;
  }

  .conversationJumpButton {
    padding-inline: 10px;
  }
```

- [ ] **Step 3: Add a CSS contract test for the floating control hooks**

Append this test in `web/src/styles/layout-scroll.test.ts` inside `describe("conversation layout scroll guards", () => { ... })`:

```ts
  it("positions the previous-user jump button as a bottom-right conversation overlay", () => {
    const timelineRule = ruleBody(css, ".conversationTimeline");
    const jumpWrapRule = ruleBody(css, ".conversationJumpButtonWrap");
    const jumpButtonRule = ruleBody(css, ".conversationJumpButton");
    const mobileRules = mediaBody(css, "(max-width: 880px)");
    const mobileJumpWrapRule = ruleBody(mobileRules, ".conversationJumpButtonWrap");

    expect(timelineRule).toContain("position: relative;");
    expect(jumpWrapRule).toContain("position: absolute;");
    expect(jumpWrapRule).toContain("right: 18px;");
    expect(jumpWrapRule).toContain("bottom: 18px;");
    expect(jumpWrapRule).toContain("pointer-events: none;");
    expect(jumpButtonRule).toContain("pointer-events: auto;");
    expect(jumpButtonRule).toContain("backdrop-filter: blur(14px);");
    expect(mobileJumpWrapRule).toContain("right: 12px;");
    expect(mobileJumpWrapRule).toContain("bottom: 12px;");
  });
```

- [ ] **Step 4: Run the CSS contract test and the focused conversation test**

Run:

```bash
cd web && pnpm exec vitest run src/styles/layout-scroll.test.ts src/components/conversation/ConversationPane.test.tsx
```

Expected: PASS for both files.

- [ ] **Step 5: Commit the styling and contract-test stage in the isolated worktree**

```bash
git add web/src/styles/global.css web/src/styles/layout-scroll.test.ts
git commit -m "style: position previous-user jump button"
```

If the checkout contains unrelated user changes, skip this commit and record the exact modified files instead.

## Task 4: Final verification and cleanup

**Files:**
- Read/verify: `web/src/components/conversation/ConversationPane.tsx`
- Read/verify: `web/src/components/conversation/ConversationPane.test.tsx`
- Read/verify: `web/src/styles/global.css`
- Read/verify: `web/src/styles/layout-scroll.test.ts`
- Write once after approval: `docs/superpowers/plans/2026-04-10-conversation-jump-to-user.md`

- [ ] **Step 1: Save the durable implementation plan copy**

Create `docs/superpowers/plans/2026-04-10-conversation-jump-to-user.md` with the approved plan content.

- [ ] **Step 2: Run the complete web verification command**

Run:

```bash
cd web && pnpm test
```

Expected: TypeScript test config passes, then Vitest reports all tests passing with exit code 0.

- [ ] **Step 3: Inspect the final diff for accidental unrelated edits**

Run:

```bash
git diff -- web/src/components/conversation/ConversationPane.tsx web/src/components/conversation/ConversationPane.test.tsx web/src/styles/global.css web/src/styles/layout-scroll.test.ts docs/superpowers/plans/2026-04-10-conversation-jump-to-user.md
```

Expected: Diff contains only the previous-user jump button logic, related tests, related CSS, and the plan file.

- [ ] **Step 4: Check for repository-wide unrelated work before any commit**

Run:

```bash
git status --short
```

Expected: The repository may already contain unrelated dirty files. Do not stage or revert unrelated user changes. If committing is requested later, stage only the five files listed in Task 4 Step 3 unless the user explicitly says otherwise.

- [ ] **Step 5: Report verification evidence**

In the final response, include:
- the files changed for this feature
- the focused and full verification commands that were run
- whether `cd web && pnpm test` passed or what failed
- any dirty-worktree caveat if unrelated changes remain present
