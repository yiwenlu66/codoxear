# UI Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the web client into a calmer two-part shell with a persistent sessions rail, a dominant conversation canvas, and a toolbar-triggered workspace dialog.

**Architecture:** Keep the existing Preact stores and backend contracts intact, but replace the current three-column shell in `AppShell` with a left rail + right canvas composition. Reuse `SessionWorkspace` inside a dismissible dialog, then consolidate the legacy shell CSS into one coherent editorial layout system that controls surface hierarchy, button sizing, text overflow, and responsive behavior. Execute this plan in a dedicated worktree so the current dirty workspace is not mixed into implementation commits.

**Tech Stack:** Preact, TypeScript, Vitest, project-owned dialog/sheet primitives, Tailwind utility classes, `web/src/styles/global.css`

---

## File Map

- Modify: `web/src/components/ui/dialog.tsx`
  - Add dismiss support for backdrop click and Escape while preserving focus trapping and focus restore.
- Modify: `web/src/components/ui/dialog.test.tsx`
  - Lock the new dismissible dialog behavior.
- Modify: `web/src/app/AppShell.tsx`
  - Remove the permanent desktop workspace rail, add `workspaceOpen` state, and open workspace content from a toolbar button.
- Modify: `web/src/app/AppShell.test.tsx`
  - Verify the shell is now two-part and the workspace appears in a dialog.
- Modify: `web/src/components/sessions/SessionCard.tsx`
  - Lighten the active-state surface and expose stable wrappers for title/preview truncation.
- Modify: `web/src/components/composer/Composer.tsx`
  - Keep the todo bar behavior, but stabilize the composer control rhythm around fixed icon buttons and a flexible input.
- Modify: `web/src/styles/global.css`
  - Replace the three-column shell rules with a two-column editorial shell, dialog sizing rules, and calmer surface hierarchy.
- Modify: `web/src/styles/layout-scroll.test.ts`
  - Lock the new shell columns, dialog bounds, text clamping, and composer sizing behavior.
- Verify unchanged behavior with: `web/src/components/sessions/SessionsPane.test.tsx`, `web/src/components/composer/Composer.test.tsx`, `web/src/components/workspace/SessionWorkspace.test.tsx`

## Task 0: Create an isolated worktree

**Files:**
- No code changes yet

- [ ] **Step 1: Create a dedicated worktree from the current HEAD**

Run: `git worktree add ../codoxear-ui-cleanup -b feat/web-ui-cleanup-dialog-shell HEAD`
Expected: a new sibling checkout at `../codoxear-ui-cleanup` with a fresh branch named `feat/web-ui-cleanup-dialog-shell`.

- [ ] **Step 2: Install frontend dependencies in the worktree**

Run: `cd ../codoxear-ui-cleanup/web && npm install`
Expected: `up to date` or a normal `added/changed` package report with no install errors.

- [ ] **Step 3: Capture a clean baseline test run before edits**

Run: `cd ../codoxear-ui-cleanup/web && npm test`
Expected: PASS on the current baseline so regressions during the shell rewrite are attributable to new edits.

## Task 1: Make the dialog primitive dismissible

**Files:**
- Modify: `web/src/components/ui/dialog.test.tsx`
- Modify: `web/src/components/ui/dialog.tsx`
- Test: `web/src/components/ui/dialog.test.tsx`

- [ ] **Step 1: Write the failing dialog dismissal tests**

```tsx
// web/src/components/ui/dialog.test.tsx
import { useState } from "preact/hooks";

it("dismisses the dialog when the backdrop is clicked", async () => {
  root = document.createElement("div");
  document.body.appendChild(root);

  function Wrapper() {
    const [open, setOpen] = useState(true);
    return (
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent titleId="dialog-title">
          <DialogHeader>
            <DialogTitle id="dialog-title">Dialog title</DialogTitle>
          </DialogHeader>
          <button type="button">Inner action</button>
        </DialogContent>
      </Dialog>
    );
  }

  render(<Wrapper />, root);
  const backdrop = root.querySelector('[data-testid="dialog-backdrop"]') as HTMLButtonElement | null;
  backdrop?.click();
  await flush();

  expect(root.querySelector('[role="dialog"]')).toBeNull();
});

it("dismisses the dialog when Escape is pressed", async () => {
  root = document.createElement("div");
  document.body.appendChild(root);

  function Wrapper() {
    const [open, setOpen] = useState(true);
    return (
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent titleId="dialog-title">
          <DialogHeader>
            <DialogTitle id="dialog-title">Dialog title</DialogTitle>
          </DialogHeader>
          <button type="button">Inner action</button>
        </DialogContent>
      </Dialog>
    );
  }

  render(<Wrapper />, root);
  const dialog = root.querySelector('[role="dialog"]');
  await pressKey(dialog, "Escape");
  await flush();

  expect(root.querySelector('[role="dialog"]')).toBeNull();
});
```

- [ ] **Step 2: Run the dialog tests to verify they fail**

Run: `cd ../codoxear-ui-cleanup/web && npx vitest run src/components/ui/dialog.test.tsx`
Expected: FAIL because `Dialog` currently has no backdrop button and `DialogContent` ignores Escape.

- [ ] **Step 3: Implement the dismissible dialog primitive**

```tsx
// web/src/components/ui/dialog.tsx
import { createContext, type ComponentChildren } from "preact";
import { useContext, useLayoutEffect, useRef } from "preact/hooks";

import { cn } from "@/lib/utils";

const DialogOpenChangeContext = createContext<((open: boolean) => void) | null>(null);

export interface DialogProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  children?: ComponentChildren;
}

export function Dialog({ open, onOpenChange, children }: DialogProps) {
  if (!open) {
    return null;
  }

  return (
    <DialogOpenChangeContext.Provider value={onOpenChange ?? null}>
      <div className="fixed inset-0 z-50 grid place-items-center p-4">
        <button
          type="button"
          data-testid="dialog-backdrop"
          aria-label="Close dialog"
          className="absolute inset-0 bg-slate-950/45"
          onClick={() => onOpenChange?.(false)}
        />
        {children}
      </div>
    </DialogOpenChangeContext.Provider>
  );
}

export function DialogContent({ className, children, titleId, ariaLabel }: DialogContentProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const onOpenChange = useContext(DialogOpenChangeContext);

  useLayoutEffect(() => {
    const content = contentRef.current;
    if (!content) {
      return undefined;
    }

    previousFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const [firstFocusable] = getFocusableElements(content);
    (firstFocusable ?? content).focus();

    return () => {
      const previousFocus = previousFocusRef.current;
      const activeElement = document.activeElement instanceof HTMLElement ? document.activeElement : null;

      if (
        previousFocus &&
        previousFocus.isConnected &&
        (!activeElement || activeElement === document.body || content.contains(activeElement))
      ) {
        queueMicrotask(() => {
          if (previousFocus.isConnected) {
            previousFocus.focus();
          }
        });
      }
    };
  }, []);

  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === "Escape") {
      event.preventDefault();
      onOpenChange?.(false);
      return;
    }

    if (event.key !== "Tab") {
      return;
    }

    const content = contentRef.current;
    if (!content) {
      return;
    }

    const focusableElements = getFocusableElements(content);
    if (focusableElements.length === 0) {
      event.preventDefault();
      content.focus();
      return;
    }

    const firstFocusable = focusableElements[0];
    const lastFocusable = focusableElements[focusableElements.length - 1];
    const activeElement = document.activeElement instanceof HTMLElement ? document.activeElement : null;

    if (event.shiftKey) {
      if (!activeElement || activeElement === firstFocusable || !content.contains(activeElement)) {
        event.preventDefault();
        lastFocusable.focus();
      }
      return;
    }

    if (!activeElement || activeElement === lastFocusable || !content.contains(activeElement)) {
      event.preventDefault();
      firstFocusable.focus();
    }
  }

  return (
    <div
      ref={contentRef}
      role="dialog"
      tabIndex={-1}
      aria-modal="true"
      aria-labelledby={titleId}
      aria-label={ariaLabel}
      onKeyDown={handleKeyDown}
      className={cn("relative z-10 w-full max-w-3xl rounded-3xl border bg-card text-card-foreground shadow-2xl", className)}
    >
      {children}
    </div>
  );
}
```

- [ ] **Step 4: Run the dialog tests to verify they pass**

Run: `cd ../codoxear-ui-cleanup/web && npx vitest run src/components/ui/dialog.test.tsx`
Expected: PASS with the two new dismissal tests and the existing focus-trap tests all green.

- [ ] **Step 5: Commit the dialog primitive update**

```bash
cd ../codoxear-ui-cleanup
git add web/src/components/ui/dialog.tsx web/src/components/ui/dialog.test.tsx
git commit -m "feat(web): make dialog overlays dismissible"
```

## Task 2: Convert the shell to a two-part layout with a workspace dialog

**Files:**
- Modify: `web/src/app/AppShell.test.tsx`
- Modify: `web/src/app/AppShell.tsx`
- Test: `web/src/app/AppShell.test.tsx`

- [ ] **Step 1: Write the failing AppShell tests for the new layout**

```tsx
// web/src/app/AppShell.test.tsx
it("renders a two-part shell without a persistent workspace rail", () => {
  renderAppShell({ activeSessionId: "sess-1", diagnostics: { status: "ok" } });

  expect(getRoot().querySelector("[data-testid='app-shell']")).not.toBeNull();
  expect(getRoot().querySelector("[data-testid='workspace-rail']")).toBeNull();
  expect(getRoot().querySelector(".desktopSessionsRail")).not.toBeNull();
  expect(getRoot().querySelector(".conversationColumn")).not.toBeNull();
  expect(getRoot().querySelector("[data-testid='workspace-dialog']")).toBeNull();
});

it("opens workspace content from a toolbar button", async () => {
  renderAppShell({
    activeSessionId: "sess-1",
    sessionUiSessionId: "sess-1",
    diagnostics: { status: "ok", queue_len: 1 },
    queue: { items: [{ text: "next task" }] },
    files: ["src/main.tsx"],
  });

  const workspaceButton = requireButtonByText("Workspace");
  workspaceButton.click();
  await flush();

  const dialog = getRoot().querySelector("[data-testid='workspace-dialog']");
  expect(dialog).not.toBeNull();
  expect(dialog?.textContent).toContain("Workspace");
  expect(dialog?.textContent).toContain("Diagnostics");
  expect(dialog?.textContent).toContain("Queue");
});
```

- [ ] **Step 2: Run the AppShell tests to verify they fail**

Run: `cd ../codoxear-ui-cleanup/web && npx vitest run src/app/AppShell.test.tsx -t "two-part shell|opens workspace content from a toolbar button"`
Expected: FAIL because `AppShell` still renders `workspace-rail` and does not expose a `Workspace` toolbar button or dialog container.

- [ ] **Step 3: Implement the new shell composition**

```tsx
// web/src/app/AppShell.tsx
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const [workspaceOpen, setWorkspaceOpen] = useState(false);

useEffect(() => {
  if (activeSessionId) {
    setSidebarOpen(false);
  }
}, [activeSessionId]);

useEffect(() => {
  setFileViewerOpen(false);
  setHarnessOpen(false);
  setWorkspaceOpen(false);
}, [activeSessionId]);

const shellClassName = "appShell editorialShell";

return (
  <>
    <div className={shellClassName} data-testid="app-shell">
      <audio ref={liveAudioRef} className="liveAudioElement" preload="none" />
      <aside className="sidebarColumn desktopSessionsRail">{renderSessionsRail()}</aside>
      <section className="conversationColumn">
        <div className="conversationToolbar">
          <div className="conversationToolbarGroup conversationToolbarGroupPrimary">
            <Button type="button" variant="outline" size="sm" className="toolbarButton mobileSheetTrigger" onClick={() => setSidebarOpen(true)}>
              Sessions
            </Button>
            <div className="conversationTitle">{activeSessionId ? activeTitle : "No session selected"}</div>
          </div>
          <div className="conversationToolbarGroup conversationToolbarGroupActions">
            <Button type="button" variant="outline" size="sm" className="toolbarTextButton" disabled={!activeSessionId} onClick={() => openFileViewer()}>
              Files
            </Button>
            <Button type="button" variant="outline" size="sm" className="toolbarTextButton workspaceToolbarButton" disabled={!activeSessionId} onClick={() => setWorkspaceOpen(true)}>
              Workspace
            </Button>
            <Button type="button" variant="outline" size="sm" className="toolbarTextButton" disabled={!activeSessionId} onClick={() => setHarnessOpen(true)}>
              Harness
            </Button>
          </div>
        </div>
        <ConversationPane onOpenFilePath={(path, line) => openFileViewer(path, line ?? null, "file")} />
        <Composer />
      </section>

      <div data-testid="mobile-sessions-sheet">
        <Sheet open={sidebarOpen}>
          <button type="button" className="sheetBackdropButton" aria-label="Close sessions panel" onClick={() => setSidebarOpen(false)} />
          <SheetContent side="left" className="mobileSheetContent" titleId="mobile-sessions-title">
            <div className="mobileSheetRail">
              <header className="mobileSheetHeader">
                <h2 id="mobile-sessions-title">Sessions</h2>
                <Button type="button" variant="ghost" size="sm" onClick={() => setSidebarOpen(false)}>Close</Button>
              </header>
              {renderSessionsRail()}
            </div>
          </SheetContent>
        </Sheet>
      </div>
    </div>

    <Dialog open={workspaceOpen} onOpenChange={setWorkspaceOpen}>
      <DialogContent titleId="workspace-dialog-title" className="workspaceDialog">
        <div data-testid="workspace-dialog" className="workspaceDialogBody">
          <DialogHeader className="workspaceDialogHeader">
            <DialogTitle id="workspace-dialog-title">Workspace</DialogTitle>
          </DialogHeader>
          {sessionUiMatchesActiveSession ? <SessionWorkspace mode="details" /> : <EmptyDetailsWorkspace />}
        </div>
      </DialogContent>
    </Dialog>

    <FileViewerDialog
      open={fileViewerOpen}
      sessionId={activeSessionId}
      files={sessionUiMatchesActiveSession ? files : []}
      initialPath={fileViewerPath}
      initialLine={fileViewerLine}
      initialMode={fileViewerMode}
      openRequestKey={fileViewerRequestKey}
      onClose={closeFileViewer}
    />
    <HarnessDialog open={harnessOpen} sessionId={activeSessionId} onClose={() => setHarnessOpen(false)} />
    <NewSessionDialog open={newSessionOpen} onClose={() => setNewSessionOpen(false)} />
  </>
);
```

- [ ] **Step 4: Run the AppShell tests to verify they pass**

Run: `cd ../codoxear-ui-cleanup/web && npx vitest run src/app/AppShell.test.tsx`
Expected: PASS, including the new two-part shell tests and the existing file-viewer, harness, and notification coverage.

- [ ] **Step 5: Commit the shell conversion**

```bash
cd ../codoxear-ui-cleanup
git add web/src/app/AppShell.tsx web/src/app/AppShell.test.tsx
git commit -m "feat(web): move workspace into a shell dialog"
```

## Task 3: Rebuild the shell styling around a calm editorial rhythm

**Files:**
- Modify: `web/src/styles/layout-scroll.test.ts`
- Modify: `web/src/styles/global.css`
- Modify: `web/src/components/sessions/SessionCard.tsx`
- Modify: `web/src/components/composer/Composer.tsx`
- Test: `web/src/styles/layout-scroll.test.ts`
- Verify still green: `web/src/components/sessions/SessionsPane.test.tsx`, `web/src/components/composer/Composer.test.tsx`, `web/src/components/workspace/SessionWorkspace.test.tsx`

- [ ] **Step 1: Add failing CSS contract coverage for the new shell and overflow rules**

```ts
// web/src/styles/layout-scroll.test.ts
it("uses a two-column editorial shell and a bounded workspace dialog", () => {
  expect(css).toMatch(/\.appShell\.editorialShell\s*\{[^}]*grid-template-columns:\s*minmax\(16rem,\s*var\(--sidebar-w\)\)\s+minmax\(0,\s*1fr\);/);
  expect(css).not.toMatch(/\.workspaceRail\s*\{/);

  const dialogRule = ruleBody(css, ".workspaceDialog");
  const dialogBodyRule = ruleBody(css, ".workspaceDialogBody");
  expect(dialogRule).toContain("max-height: min(84dvh, 52rem);");
  expect(dialogRule).toContain("overflow: hidden;");
  expect(dialogBodyRule).toContain("min-height: 0;");
  expect(dialogBodyRule).toContain("overflow: hidden;");
});

it("clamps session copy and keeps toolbar/composer controls from collapsing", () => {
  const sessionTitleRule = ruleBody(css, ".sessionTitle");
  const sessionPreviewRule = ruleBody(css, ".sessionPreview");
  const toolbarTextButtonRule = ruleBody(css, ".toolbarTextButton");
  const composerInputRule = ruleBody(css, ".composerInputWrap");
  const sendButtonRule = ruleBody(css, ".sendButton");

  expect(sessionTitleRule).toContain("display: -webkit-box;");
  expect(sessionTitleRule).toContain("-webkit-line-clamp: 2;");
  expect(sessionPreviewRule).toContain("display: -webkit-box;");
  expect(sessionPreviewRule).toContain("-webkit-line-clamp: 2;");
  expect(toolbarTextButtonRule).toContain("min-width: fit-content;");
  expect(composerInputRule).toContain("min-width: 0;");
  expect(sendButtonRule).toContain("width: 44px;");
});
```

- [ ] **Step 2: Run the layout and supporting component tests to verify they fail**

Run: `cd ../codoxear-ui-cleanup/web && npx vitest run src/styles/layout-scroll.test.ts src/components/sessions/SessionsPane.test.tsx src/components/composer/Composer.test.tsx src/components/workspace/SessionWorkspace.test.tsx`
Expected: FAIL because the CSS still defines the old three-column shell and the current session/composer rules do not satisfy the new clamp and sizing expectations.

- [ ] **Step 3: Update the session card markup and composer controls to match the new shell rhythm**

```tsx
// web/src/components/sessions/SessionCard.tsx
return (
  <div
    data-testid="session-card"
    className="sessionCard"
    aria-current={active ? "true" : undefined}
  >
    <Card className={cn("sessionCardSurface h-full border-border/60 bg-card/90 shadow-sm", active && "ring-1 ring-primary/30 shadow-md") }>
      <CardContent className="p-4">
        <button
          type="button"
          className="sessionCardButton w-full text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50"
          aria-current={active ? "true" : undefined}
          onClick={onSelect}
        >
          <div className="sessionMetaLine">
            <span className={cn("stateDot", session.busy && "busy")} />
            <Badge variant="secondary" className="backendBadge">{session.agent_backend || "codex"}</Badge>
            {session.owned ? <Badge variant="outline" className="ownerBadge">web</Badge> : null}
            {session.queue_len ? <Badge className="queueBadge">{session.queue_len} queued</Badge> : null}
          </div>
          <div className="sessionTitle">{title}</div>
          <div className="sessionPreview">{preview}</div>
        </button>
        {showActions ? (
          <div className="sessionActionRow mt-3 flex items-center justify-end gap-2">
            <Button type="button" variant="outline" size="sm" className="sessionDeleteButton" onClick={() => onDelete?.()}>
              Delete
            </Button>
          </div>
        ) : null}
      </CardContent>
    </Card>
  </div>
);

// web/src/components/composer/Composer.tsx
<Button
  type="button"
  variant="outline"
  size="icon"
  className="composerAttachButton"
  aria-label="Attach file"
>
  <span className="buttonGlyph">📎</span>
  <span className="visuallyHidden">Attach file</span>
</Button>
<div className="composerInputWrap flex-1">
  <Textarea
    value={draft}
    placeholder="Enter your instructions here"
    className="composerTextarea"
    onInput={(event) => composerStoreApi.setDraft(event.currentTarget.value)}
    onKeyDown={(event) => {
      if (event.key !== "Enter" || event.isComposing) {
        return;
      }
      if (event.shiftKey) {
        return;
      }
      if (!enterToSendEnabled() && !event.ctrlKey && !event.metaKey) {
        return;
      }
      if (!activeSessionId) {
        return;
      }
      event.preventDefault();
      composerStoreApi.submit(activeSessionId).catch(() => undefined);
    }}
    disabled={sending}
  />
</div>
<Button
  type="button"
  variant="outline"
  size="sm"
  className="composerQueueButton"
  aria-label="Queued messages"
>
  Queue
</Button>
<Button
  type="submit"
  className="sendButton"
  aria-label={sending ? "Sending" : "Send"}
  disabled={sending || !draft.trim()}
>
  <span className="buttonGlyph">➤</span>
  <span className="visuallyHidden">{sending ? "Sending..." : "Send"}</span>
</Button>
```

- [ ] **Step 4: Replace the legacy shell blocks in `global.css` with the new two-part rules**

```css
/* web/src/styles/global.css */
.appShell.editorialShell {
  display: grid;
  grid-template-columns: minmax(16rem, var(--sidebar-w)) minmax(0, 1fr);
  grid-template-rows: 1fr;
  position: fixed;
  inset: 0;
  gap: 1rem;
  padding: 1rem;
  background:
    radial-gradient(circle at top left, hsl(var(--primary) / 0.08), transparent 28%),
    linear-gradient(180deg, color-mix(in srgb, var(--panel) 82%, white), color-mix(in srgb, var(--bg) 94%, #eef2f7));
  color: var(--text);
  overflow: hidden;
}

.sidebarColumn {
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--legacy-border) 72%, white);
  border-radius: 30px;
  background: color-mix(in srgb, var(--panel) 92%, white);
  box-shadow: 0 20px 48px rgba(15, 23, 42, 0.07);
}

.conversationColumn {
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--legacy-border) 70%, white);
  border-radius: 34px;
  background: color-mix(in srgb, var(--panel) 97%, white);
  box-shadow: 0 28px 64px rgba(15, 23, 42, 0.08);
}

.conversationToolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.9rem;
  flex-wrap: wrap;
  padding: 0.95rem 1.1rem;
  border-bottom: 1px solid color-mix(in srgb, var(--legacy-border) 78%, white);
  background: color-mix(in srgb, var(--panel) 95%, white);
}

.conversationToolbarGroup {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  min-width: 0;
}

.conversationToolbarGroupPrimary {
  flex: 1 1 18rem;
}

.conversationToolbarGroupActions {
  flex: 0 1 auto;
  justify-content: flex-end;
  flex-wrap: wrap;
}

.conversationTitle {
  min-width: 0;
  font-weight: 650;
  color: var(--text);
  text-wrap: balance;
}

.toolbarTextButton {
  min-width: fit-content;
  height: 2.375rem;
  padding: 0 0.95rem;
  border-radius: 999px;
}

.sessionCardSurface {
  border-radius: 1.45rem;
}

.sessionTitle,
.sessionPreview {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.sessionTitle {
  margin-top: 0.85rem;
  font-weight: 600;
  font-size: 1rem;
  line-height: 1.3;
  -webkit-line-clamp: 2;
}

.sessionPreview {
  margin-top: 0.45rem;
  color: var(--legacy-muted);
  font-size: 0.9rem;
  line-height: 1.45;
  -webkit-line-clamp: 2;
}

.messageBubble,
.messageCard {
  border-color: color-mix(in srgb, var(--legacy-border) 62%, white);
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
}

.composerShell {
  display: flex;
  align-items: flex-end;
  gap: 0.75rem;
  padding: 0.85rem 1rem calc(0.85rem + env(safe-area-inset-bottom));
  border-top: 1px solid color-mix(in srgb, var(--legacy-border) 78%, white);
  background: color-mix(in srgb, var(--panel) 96%, white);
}

.composerInputWrap {
  flex: 1 1 auto;
  min-width: 0;
  border: 1px solid color-mix(in srgb, var(--legacy-border) 76%, white);
  border-radius: 999px;
  background: #fff;
}

.composerTextarea {
  min-height: 44px;
  border: 0;
  border-radius: 999px;
  background: transparent;
  padding: 12px 16px;
}

.composerAttachButton,
.sendButton {
  width: 44px;
  height: 44px;
  flex: 0 0 44px;
  border-radius: 999px;
}

.composerQueueButton {
  min-width: fit-content;
  height: 44px;
  padding: 0 14px;
  border-radius: 999px;
}

.workspaceDialog {
  width: min(64rem, calc(100vw - 2rem));
  max-height: min(84dvh, 52rem);
  padding: 0;
  overflow: hidden;
}

.workspaceDialogBody {
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.workspaceDialogHeader {
  border-bottom: 1px solid color-mix(in srgb, var(--legacy-border) 78%, white);
}

@media (max-width: 880px) {
  .appShell.editorialShell {
    grid-template-columns: 1fr;
    padding: 0.75rem;
    gap: 0.75rem;
  }

  .desktopSessionsRail {
    display: none;
  }

  .conversationToolbar {
    padding: 0.85rem 0.9rem;
  }

  .workspaceDialog {
    width: min(calc(100vw - 1rem), 36rem);
    max-height: min(90dvh, 40rem);
  }

  .composerQueueButton {
    padding: 0 12px;
  }
}
```

- [ ] **Step 5: Run the layout and supporting component tests to verify they pass**

Run: `cd ../codoxear-ui-cleanup/web && npx vitest run src/styles/layout-scroll.test.ts src/components/sessions/SessionsPane.test.tsx src/components/composer/Composer.test.tsx src/components/workspace/SessionWorkspace.test.tsx`
Expected: PASS with the new shell/style assertions and the unchanged behavior tests still green.

- [ ] **Step 6: Commit the shell styling update**

```bash
cd ../codoxear-ui-cleanup
git add web/src/styles/global.css web/src/styles/layout-scroll.test.ts web/src/components/sessions/SessionCard.tsx web/src/components/composer/Composer.tsx
git commit -m "feat(web): calm the shell surface hierarchy"
```

## Task 4: Run the full verification sweep

**Files:**
- No new files unless a failing test forces a direct regression fix

- [ ] **Step 1: Run the full frontend test suite**

Run: `cd ../codoxear-ui-cleanup/web && npm test`
Expected: PASS with `tsc --noEmit -p tsconfig.test.json` and `vitest run` both succeeding.

- [ ] **Step 2: Build the frontend bundle**

Run: `cd ../codoxear-ui-cleanup/web && npm run build`
Expected: PASS with `vite build` succeeding and assets copied into `../codoxear/static/dist/`.

- [ ] **Step 3: Manually smoke-check the shell in the browser**

Run: `cd ../codoxear-ui-cleanup/web && npm run dev`
Expected: local Vite dev server starts; verify these manual checks before stopping it:

- Desktop shows only the sessions rail and conversation canvas
- `Workspace` opens and closes as a dialog from the toolbar
- Long session titles clamp instead of stretching cards
- Toolbar and composer controls keep readable sizes in English and Chinese
- Mobile-width responsive view keeps session access in a sheet and conversation as the default surface

- [ ] **Step 4: Commit the final verified state**

```bash
cd ../codoxear-ui-cleanup
git add web/src/app/AppShell.tsx web/src/app/AppShell.test.tsx web/src/components/ui/dialog.tsx web/src/components/ui/dialog.test.tsx web/src/components/sessions/SessionCard.tsx web/src/components/composer/Composer.tsx web/src/styles/global.css web/src/styles/layout-scroll.test.ts
git commit -m "feat(web): simplify the app shell layout"
```

## Self-Review

### Spec coverage

- Two-part desktop shell: covered by Task 2 and Task 3
- Workspace as toolbar-triggered dialog: covered by Task 1 and Task 2
- Reduced borders and quieter hierarchy: covered by Task 3
- Button rhythm and overflow handling: covered by Task 3
- Mobile consistency: covered by Task 2 and Task 4 manual checks
- Verification before completion: covered by Task 4

### Placeholder scan

- No `TODO`, `TBD`, or `...` placeholders remain in the plan steps.
- Every task names exact files and exact verification commands.
- Each implementation step includes the concrete code block or CSS block to write.

### Type and naming consistency

- Workspace open state is consistently named `workspaceOpen`.
- The new shell class is consistently named `editorialShell`.
- The new workspace overlay hooks are consistently named `workspaceDialog`, `workspaceDialogBody`, and `workspaceDialogHeader`.
- The new toolbar text-button class is consistently named `toolbarTextButton`.
