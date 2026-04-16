# Pi Todo Composer Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the toolbar todo popover with a compact Pi todo summary bar directly above the composer that expands inline to show the full todo list.

**Architecture:** Move todo ownership out of `AppShell` and into the composer area. Reuse the existing todo snapshot normalization logic in a composer-oriented display component, gate rendering on a current non-empty Pi `diagnostics.todo_snapshot`, and remove the now-wrong toolbar popover behavior and styles.

**Tech Stack:** Preact, TypeScript, Vitest, Vite CSS

---

## File Structure

### Files to create

- `web/src/components/composer/TodoComposerPanel.tsx` - display-only composer-adjacent todo summary bar and expandable inline panel
- `web/src/components/composer/TodoComposerPanel.test.tsx` - focused rendering coverage for collapsed, expanded, and normalization-safe todo display above the composer

### Files to modify

- `web/src/components/composer/Composer.tsx` - render the todo summary bar above the input and own expanded/collapsed state
- `web/src/components/composer/Composer.test.tsx` - cover visibility gating, expand/collapse, and stale-session suppression
- `web/src/app/AppShell.tsx` - remove toolbar todo button/state and keep the rest of the shell unchanged
- `web/src/app/AppShell.test.tsx` - remove toolbar todo expectations and keep shell expectations aligned with the new design
- `web/src/lib/types.ts` - retain shared todo snapshot types and ensure composer-facing code can consume them cleanly
- `web/src/styles/global.css` - replace toolbar-popover styling with composer-top summary/panel styling
- `web/src/styles/layout-scroll.test.ts` - replace popover-oriented selector assertions with composer bar style-contract checks

### Files to delete

- `web/src/components/workspace/TodoPopover.tsx` - obsolete after moving todo UI ownership into the composer area
- `web/src/components/workspace/TodoPopover.test.tsx` - obsolete after replacing popover behavior with the composer-top bar

### Notes

- This plan supersedes `docs/superpowers/plans/2026-04-07-pi-todo-popover-restoration.md`.
- Do **not** add commit steps; the user explicitly asked to skip commits.
- Keep timeline `todo_snapshot` rendering in `ConversationPane` unchanged.

## Task 1: Build the display-only composer todo panel component

**Files:**
- Create: `web/src/components/composer/TodoComposerPanel.tsx`
- Create: `web/src/components/composer/TodoComposerPanel.test.tsx`
- Modify: `web/src/lib/types.ts`
- Delete: `web/src/components/workspace/TodoPopover.tsx`
- Delete: `web/src/components/workspace/TodoPopover.test.tsx`

- [ ] **Step 1: Write the failing tests for the new composer-oriented todo component**

```tsx
import { render } from "preact";
import { afterEach, describe, expect, it } from "vitest";
import { TodoComposerPanel } from "./TodoComposerPanel";

describe("TodoComposerPanel", () => {
  let root: HTMLDivElement | null = null;

  afterEach(() => {
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("renders a compact summary row when collapsed", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(
      <TodoComposerPanel
        snapshot={{
          available: true,
          error: false,
          progress_text: "2/4 completed",
          items: [
            { title: "Explore project context", status: "completed" },
            { title: "Move todo above composer", status: "in-progress" },
          ],
        }}
        expanded={false}
        onToggle={() => undefined}
      />,
      root,
    );

    expect(root.querySelector(".composerTodoBar")).not.toBeNull();
    expect(root.querySelector(".composerTodoBarButton")).not.toBeNull();
    expect(root.querySelector(".composerTodoPanel")).toBeNull();
    expect(root.textContent).toContain("2/4 completed");
  });

  it("renders the full list when expanded", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(
      <TodoComposerPanel
        snapshot={{
          available: true,
          error: false,
          progress_text: "1/2 completed",
          items: [
            { title: "Restore legacy placement", status: "completed", description: "Move it above the input" },
            { title: "Polish summary bar", status: "in-progress" },
          ],
        }}
        expanded={true}
        onToggle={() => undefined}
      />,
      root,
    );

    expect(root.querySelector(".composerTodoPanel")).not.toBeNull();
    expect(root.querySelectorAll(".composerTodoItem")).toHaveLength(2);
    expect(root.textContent).toContain("Restore legacy placement");
    expect(root.textContent).toContain("Move it above the input");
    expect(root.textContent).toContain("in-progress");
  });

  it("keeps raw Pi status strings and normalization fallbacks", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(
      <TodoComposerPanel
        snapshot={{
          available: true,
          error: false,
          progress_text: "1/1 completed",
          items: [{ title: "  ", status: "  custom-status  ", description: "  Needs trimming  " }],
        }}
        expanded={true}
        onToggle={() => undefined}
      />,
      root,
    );

    expect(root.textContent).toContain("Untitled todo");
    expect(root.textContent).toContain("custom-status");
    expect(root.textContent).toContain("Needs trimming");
  });
});
```

- [ ] **Step 2: Run the focused component test to verify it fails**

Run: `cd web && npx vitest run src/components/composer/TodoComposerPanel.test.tsx`
Expected: FAIL because `TodoComposerPanel.tsx` does not exist yet.

- [ ] **Step 3: Create the new component and move the normalization logic into it**

```tsx
// web/src/components/composer/TodoComposerPanel.tsx
import type { TodoSnapshot, TodoSnapshotItem } from "../../lib/types";

interface TodoComposerPanelProps {
  snapshot: unknown;
  expanded: boolean;
  onToggle: () => void;
}

function normalizeText(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function normalizeTodoItem(value: unknown): TodoSnapshotItem | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const item = value as Record<string, unknown>;
  const normalized: TodoSnapshotItem = {
    id: typeof item.id === "number" || typeof item.id === "string" ? item.id : undefined,
    title: normalizeText(item.title),
    status: normalizeText(item.status),
    description: normalizeText(item.description),
  };
  return [normalized.title, normalized.status, normalized.description].some(Boolean) ? normalized : null;
}

function normalizeSnapshot(snapshot: unknown): TodoSnapshot {
  if (!snapshot || typeof snapshot !== "object") {
    return { available: false, error: false, items: [] };
  }
  const raw = snapshot as Record<string, unknown>;
  return {
    available: raw.available === true,
    error: raw.error === true,
    progress_text: normalizeText(raw.progress_text),
    items: Array.isArray(raw.items)
      ? raw.items.map(normalizeTodoItem).filter((item): item is TodoSnapshotItem => Boolean(item))
      : [],
  };
}

function statusClassName(status: string | undefined) {
  return status ? status.replace(/[^a-z0-9_-]+/gi, "-") : "unknown";
}

export function TodoComposerPanel({ snapshot, expanded, onToggle }: TodoComposerPanelProps) {
  const todo = normalizeSnapshot(snapshot);
  const summary = todo.progress_text || "Todo";

  return (
    <div className="composerTodoBar">
      <button
        type="button"
        className={`composerTodoBarButton${expanded ? " isExpanded" : ""}`}
        aria-expanded={expanded ? "true" : "false"}
        onClick={onToggle}
      >
        <span className="composerTodoSummary">{summary}</span>
        <span className="composerTodoToggleHint">{expanded ? "Hide" : "Show"}</span>
      </button>
      {expanded ? (
        <div className="composerTodoPanel">
          <div className="composerTodoList">
            {todo.items.map((item, index) => (
              <article key={`${item.title || "todo"}-${index}`} className="composerTodoItem">
                <div className="composerTodoItemHead">
                  <strong>{item.title || "Untitled todo"}</strong>
                  <span className={`composerTodoStatus ${statusClassName(item.status)}`}>{item.status || "unknown"}</span>
                </div>
                {item.description ? <p>{item.description}</p> : null}
              </article>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
```

```ts
// web/src/lib/types.ts
export interface TodoSnapshotItem {
  id?: number | string;
  title?: string;
  status?: string;
  description?: string;
}

export interface TodoSnapshot {
  available?: boolean;
  error?: boolean;
  progress_text?: string;
  items: TodoSnapshotItem[];
  counts?: Record<string, number>;
}
```

- [ ] **Step 4: Delete the old popover component and rerun the focused test**

Run: `rm web/src/components/workspace/TodoPopover.tsx web/src/components/workspace/TodoPopover.test.tsx`

Run: `cd web && npx vitest run src/components/composer/TodoComposerPanel.test.tsx`
Expected: PASS with 3 tests passing.

## Task 2: Move todo ownership into `Composer` and gate it on current Pi diagnostics

**Files:**
- Modify: `web/src/components/composer/Composer.tsx`
- Modify: `web/src/components/composer/Composer.test.tsx`
- Modify: `web/src/app/providers.tsx` only if needed for existing session-ui hook usage in Composer

- [ ] **Step 1: Write the failing composer integration tests**

Add tests like these to `web/src/components/composer/Composer.test.tsx`:

```tsx
it("shows a todo summary bar above the composer for a current pi session with todo items", async () => {
  const sessionsStore = createStore(
    {
      items: [{ session_id: "sess-1", agent_backend: "pi", busy: false }],
      activeSessionId: "sess-1",
      loading: false,
      newSessionDefaults: null,
    },
    () => ({ refresh: vi.fn(), select: vi.fn() }),
  );
  const composerStore = createStore(
    { draft: "Hello", sending: false },
    (setState) => ({
      setDraft(value: string) {
        setState({ draft: value });
      },
      submit: vi.fn().mockResolvedValue(undefined),
    }),
  );
  const sessionUiStore = createStore(
    {
      sessionId: "sess-1",
      diagnostics: {
        todo_snapshot: {
          available: true,
          error: false,
          progress_text: "2/3 completed",
          items: [{ title: "Move todo above composer", status: "in-progress" }],
        },
      },
      queue: null,
      files: [],
      requests: [],
      loading: false,
    },
    () => ({ refresh: vi.fn() }),
  );

  root = document.createElement("div");
  document.body.appendChild(root);
  render(
    <AppProviders sessionsStore={sessionsStore as any} composerStore={composerStore as any} sessionUiStore={sessionUiStore as any}>
      <Composer />
    </AppProviders>,
    root,
  );

  expect(root.querySelector(".composerTodoBar")).not.toBeNull();
  expect(root.textContent).toContain("2/3 completed");
  expect(root.querySelector(".composerShell")).not.toBeNull();
});

it("expands and collapses the todo panel when the summary bar is clicked", async () => {
  // same store setup as above
  const toggle = root.querySelector(".composerTodoBarButton") as HTMLButtonElement;
  toggle.click();
  await Promise.resolve();
  expect(root.querySelector(".composerTodoPanel")).not.toBeNull();
  toggle.click();
  await Promise.resolve();
  expect(root.querySelector(".composerTodoPanel")).toBeNull();
});

it("hides the todo bar when session ui state is stale or the session is not pi", () => {
  // stale sessionUiStore.sessionId case
  // non-pi active session case
});
```

- [ ] **Step 2: Run the focused composer test file to verify it fails**

Run: `cd web && npx vitest run src/components/composer/Composer.test.tsx`
Expected: FAIL because the composer does not yet render any todo bar.

- [ ] **Step 3: Integrate the todo bar into `Composer.tsx`**

```tsx
// web/src/components/composer/Composer.tsx
import { useEffect, useMemo, useState } from "preact/hooks";
import { useSessionUiStore, useSessionsStore, useComposerStore, useComposerStoreApi } from "../../app/providers";
import { TodoComposerPanel } from "./TodoComposerPanel";
import type { TodoSnapshot } from "../../lib/types";

export function Composer() {
  const { activeSessionId, items } = useSessionsStore();
  const { draft, sending } = useComposerStore();
  const { sessionId: sessionUiSessionId, diagnostics } = useSessionUiStore();
  const composerStoreApi = useComposerStoreApi();
  const [todoExpanded, setTodoExpanded] = useState(false);

  const activeSession = items.find((session) => session.session_id === activeSessionId) ?? null;
  const todoSnapshot = useMemo(() => {
    if (!activeSessionId || activeSession?.agent_backend !== "pi") {
      return null;
    }
    if (sessionUiSessionId !== activeSessionId) {
      return null;
    }
    if (!diagnostics || typeof diagnostics !== "object") {
      return null;
    }
    const snapshot = (diagnostics as { todo_snapshot?: TodoSnapshot }).todo_snapshot;
    if (!snapshot?.available || !Array.isArray(snapshot.items) || snapshot.items.length === 0) {
      return null;
    }
    return snapshot;
  }, [activeSession?.agent_backend, activeSessionId, diagnostics, sessionUiSessionId]);

  useEffect(() => {
    if (!todoSnapshot) {
      setTodoExpanded(false);
      return;
    }
  }, [todoSnapshot, activeSessionId]);

  return (
    <div className="composerStack">
      {todoSnapshot ? (
        <TodoComposerPanel
          snapshot={todoSnapshot}
          expanded={todoExpanded}
          onToggle={() => setTodoExpanded((value) => !value)}
        />
      ) : null}
      <form className={`composer composerShell${draft.includes("\n") ? " multiline" : ""}`}>
        {/* existing composer controls unchanged */}
      </form>
    </div>
  );
}
```

- [ ] **Step 4: Run the focused composer tests to verify they pass**

Run: `cd web && npx vitest run src/components/composer/Composer.test.tsx src/components/composer/TodoComposerPanel.test.tsx`
Expected: PASS with all composer todo tests green.

## Task 3: Remove toolbar todo behavior from `AppShell` and align shell tests

**Files:**
- Modify: `web/src/app/AppShell.tsx`
- Modify: `web/src/app/AppShell.test.tsx`

- [ ] **Step 1: Write the failing shell regression updates**

Update `web/src/app/AppShell.test.tsx` so it expects no toolbar todo control:

```tsx
it("does not render a Todo button in the toolbar", () => {
  renderAppShell({
    agentBackend: "pi",
    diagnostics: {
      todo_snapshot: {
        available: true,
        error: false,
        progress_text: "1/2 completed",
        items: [{ title: "Shown in composer instead", status: "in-progress" }],
      },
    },
  });

  expect(findButtonByText("Todo")).toBeUndefined();
  expect(getRoot().querySelector(".todoToolbarAnchor")).toBeNull();
});
```

Remove or replace the popover-oriented tests that currently assert:

- toolbar todo visibility
- toolbar toggle behavior
- outside-pointer close behavior
- toolbar-specific stale-state behavior

Keep the rest of the shell tests for sidebar, details, title, and workspace semantics.

- [ ] **Step 2: Run the focused AppShell tests to verify they fail**

Run: `cd web && npx vitest run src/app/AppShell.test.tsx`
Expected: FAIL because `AppShell` still renders the toolbar todo entry.

- [ ] **Step 3: Remove the obsolete toolbar todo logic from `AppShell.tsx`**

```tsx
// web/src/app/AppShell.tsx
import { useEffect, useMemo, useState } from "preact/hooks";
// remove TodoComposer import, todo open state, anchor refs, and todo-specific effects

export function AppShell() {
  const { activeSessionId, items } = useSessionsStore();
  const { sessionId: sessionUiSessionId, diagnostics, requests } = useSessionUiStore();
  // keep stale session-ui gating for diagnostics/requests if still needed for workspace/details

  return (
    <>
      <div className={shellClassName}>
        {/* ... */}
        <section className="conversationColumn">
          <div className="conversationToolbar">
            <div className="conversationToolbarGroup">{/* existing left controls */}</div>
            <div className="conversationToolbarGroup">
              <button type="button" className="toolbarButton" disabled={!activeSessionId}>
                <span className="buttonGlyph">📄</span>
                <span className="visuallyHidden">View file</span>
              </button>
              <button type="button" className="toolbarButton" disabled={!activeSessionId} onClick={() => setDetailsOpen(true)}>
                <span className="buttonGlyph">ⓘ</span>
                <span className="visuallyHidden">Details</span>
              </button>
              <button type="button" className="toolbarButton" disabled={!activeSessionId}>
                <span className="buttonGlyph">⟳</span>
                <span className="visuallyHidden">Harness mode</span>
              </button>
            </div>
          </div>
          <ConversationPane />
          <Composer />
        </section>
      </div>
    </>
  );
}
```

- [ ] **Step 4: Run the AppShell tests to verify they pass**

Run: `cd web && npx vitest run src/app/AppShell.test.tsx`
Expected: PASS with the shell aligned to the composer-owned todo design.

## Task 4: Replace popover styles with composer-top summary/panel styles and final verification

**Files:**
- Modify: `web/src/styles/global.css`
- Modify: `web/src/styles/layout-scroll.test.ts`
- Modify: `web/src/components/composer/Composer.test.tsx`

- [ ] **Step 1: Add the failing style-contract assertions for the composer bar**

Replace the popover-specific assertions in `web/src/styles/layout-scroll.test.ts` with composer-bar assertions like:

```ts
it("keeps composer todo selectors in the global shell stylesheet", () => {
  expect(css).toMatch(/\.composerTodoBar\s*\{/);
  expect(css).toMatch(/\.composerTodoBarButton\.isExpanded\s*\{/);
  expect(css).toMatch(/\.composerTodoPanel\s*\{/);
  expect(css).toMatch(/\.composerTodoStatus\.completed\s*\{/);
});

it("bounds the expanded composer todo panel without hiding the composer", () => {
  expect(css).toMatch(/\.composerTodoPanel\s*\{[^}]*max-height:\s*min\(32dvh,\s*260px\);/);
  expect(css).toMatch(/\.composerTodoPanel\s*\{[^}]*overflow:\s*auto;/);
});
```

- [ ] **Step 2: Run the style regression test to verify it fails**

Run: `cd web && npx vitest run src/styles/layout-scroll.test.ts`
Expected: FAIL because the stylesheet still contains popover-specific todo rules instead of composer-bar rules.

- [ ] **Step 3: Add the composer-adjacent styles and align composer tests**

```css
/* web/src/styles/global.css */
.composerStack {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px 12px 12px;
}

.composerTodoBar {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.composerTodoBarButton {
  width: 100%;
  min-width: 0;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: color-mix(in srgb, var(--panel) 94%, #eef6e8);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.composerTodoBarButton.isExpanded {
  border-color: color-mix(in srgb, var(--accent) 32%, var(--border));
}

.composerTodoSummary,
.composerTodoItem p {
  overflow-wrap: anywhere;
}

.composerTodoPanel {
  max-height: min(32dvh, 260px);
  overflow: auto;
  padding: 12px;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.88);
}

.composerTodoList {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.composerTodoItemHead {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
  min-width: 0;
}

.composerTodoItemHead strong {
  flex: 1 1 auto;
  min-width: 0;
  overflow-wrap: anywhere;
}

.composerTodoStatus {
  flex: 0 1 auto;
  min-width: 0;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.08);
  color: var(--muted);
  font-size: 10px;
  line-height: 1.5;
  overflow-wrap: anywhere;
}

.composerTodoStatus.completed {
  background: rgba(22, 163, 74, 0.12);
  color: #166534;
}

.composerTodoStatus.in-progress {
  background: rgba(37, 99, 235, 0.12);
  color: #1d4ed8;
}

.composerTodoStatus.not-started {
  background: rgba(245, 158, 11, 0.14);
  color: #92400e;
}

@media (max-width: 880px) {
  .composerStack {
    padding: 8px 10px calc(8px + env(safe-area-inset-bottom));
  }

  .composerTodoPanel {
    max-height: min(28dvh, 220px);
  }
}
```

Also update `Composer.test.tsx` to assert the composer todo bar sits above the form:

```tsx
expect(root.querySelector(".composerTodoBar")?.nextElementSibling?.classList.contains("composerShell")).toBe(true);
```

- [ ] **Step 4: Run the targeted and final verification commands**

Run: `cd web && npx vitest run src/components/composer/Composer.test.tsx src/components/composer/TodoComposerPanel.test.tsx src/app/AppShell.test.tsx src/styles/layout-scroll.test.ts`
Expected: PASS with the new composer-top todo bar behavior covered.

Run: `cd web && npx tsc --noEmit -p tsconfig.test.json && npx vitest run src/components/composer/Composer.test.tsx src/components/composer/TodoComposerPanel.test.tsx src/app/AppShell.test.tsx src/styles/layout-scroll.test.ts`
Expected: PASS with no TypeScript errors and all selected tests green.

## Self-Review

### Spec coverage

- Composer-top placement: covered in Task 2 and Task 4
- Collapsed summary + inline expansion: covered in Tasks 1 and 2
- Hide when no todo content: covered in Task 2 tests and gating logic
- Remove toolbar todo behavior: covered in Task 3
- Mobile-safe composer-adjacent layout: covered in Task 4 CSS and tests

### Placeholder scan

- No `TODO`, `TBD`, or deferred implementation markers remain
- All file paths are explicit
- All test and verification commands are explicit and runnable
- Each code-changing step includes concrete code

### Type consistency

- `TodoSnapshot` / `TodoSnapshotItem` stay the shared todo types
- `TodoComposerPanel` owns display normalization and is referenced consistently by `Composer`
- Style selectors use the `composerTodo*` prefix consistently across CSS and tests
- `AppShell` no longer carries any `todoToolbar*` or popover-specific contract after Task 3
