# Pi Todo Popover Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the old web-style Pi todo floating window in the new `web/` frontend with a toolbar `Todo` button that opens a lightweight popover.

**Architecture:** Keep todo data in the existing diagnostics flow and add a new display-only `TodoPopover` component plus local `AppShell` state for toggling and outside-click handling. Do not change backend APIs or the meaning of the existing `Details` dialog; the popover is a separate, faster glance surface that reads `diagnostics.todo_snapshot` defensively.

**Tech Stack:** Preact, TypeScript, Vitest, Vite CSS

---

## File Structure

### Files to create

- `web/src/components/workspace/TodoPopover.tsx` - focused display-only popover component for the Pi todo snapshot
- `web/src/components/workspace/TodoPopover.test.tsx` - focused rendering coverage for available, empty, and unavailable snapshot states

### Files to modify

- `web/src/app/AppShell.tsx` - add Pi-only toolbar button, local popover state, refs, and outside-click/session-change closing behavior
- `web/src/app/AppShell.test.tsx` - lock Pi-only visibility and popover toggle behavior
- `web/src/lib/types.ts` - add reusable `TodoSnapshot` and `TodoSnapshotItem` types for frontend rendering
- `web/src/styles/global.css` - add anchored floating popover styles and mobile-safe layout rules
- `web/src/styles/layout-scroll.test.ts` - lock key popover selector presence so later cleanup does not silently remove the floating window

### Notes

- Do **not** modify `web/src/components/workspace/SessionWorkspace.tsx`; `Details` keeps its existing semantics.
- Do **not** add commit steps; the user explicitly asked to skip commits for this work.

## Task 1: Build the display-only `TodoPopover` component with defensive snapshot handling

**Files:**
- Create: `web/src/components/workspace/TodoPopover.tsx`
- Create: `web/src/components/workspace/TodoPopover.test.tsx`
- Modify: `web/src/lib/types.ts`

- [ ] **Step 1: Write the failing rendering tests for the new component**

```tsx
import { render } from "preact";
import { afterEach, describe, expect, it } from "vitest";
import { TodoPopover } from "./TodoPopover";

describe("TodoPopover", () => {
  let root: HTMLDivElement | null = null;

  afterEach(() => {
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("renders progress text and todo items when a snapshot is available", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(
      <TodoPopover
        snapshot={{
          available: true,
          error: false,
          progress_text: "2/3 completed",
          items: [
            { title: "Explore project context", status: "completed", description: "Read the relevant UI files" },
            { title: "Restore todo popover", status: "in-progress" },
          ],
        }}
      />,
      root,
    );

    expect(root.textContent).toContain("Todo");
    expect(root.textContent).toContain("2/3 completed");
    expect(root.textContent).toContain("Explore project context");
    expect(root.textContent).toContain("completed");
    expect(root.textContent).toContain("Read the relevant UI files");
    expect(root.textContent).toContain("Restore todo popover");
    expect(root.querySelectorAll(".todoPopoverItem")).toHaveLength(2);
  });

  it("renders the empty state when no todo snapshot is available yet", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(<TodoPopover snapshot={{ available: false, error: false, items: [] }} />, root);

    expect(root.textContent).toContain("No todo list yet");
  });

  it("renders the unavailable state when snapshot loading failed", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(<TodoPopover snapshot={{ available: false, error: true, items: [] }} />, root);

    expect(root.textContent).toContain("Todo list unavailable");
  });

  it("falls back safely when items are malformed or missing titles", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(
      <TodoPopover
        snapshot={{
          available: true,
          error: false,
          items: [{ status: "not-started", description: "Needs a title fallback" }],
        }}
      />,
      root,
    );

    expect(root.textContent).toContain("Untitled todo");
    expect(root.textContent).toContain("Needs a title fallback");
  });
});
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `cd web && npx vitest run src/components/workspace/TodoPopover.test.tsx`
Expected: FAIL with a module resolution error because `./TodoPopover` does not exist yet.

- [ ] **Step 3: Add the shared todo types and minimal popover implementation**

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

```tsx
// web/src/components/workspace/TodoPopover.tsx
import type { TodoSnapshot, TodoSnapshotItem } from "../../lib/types";

interface TodoPopoverProps {
  snapshot: unknown;
}

function normalizeTodoItem(value: unknown): TodoSnapshotItem | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const item = value as Record<string, unknown>;
  return {
    id: typeof item.id === "number" || typeof item.id === "string" ? item.id : undefined,
    title: typeof item.title === "string" ? item.title : undefined,
    status: typeof item.status === "string" ? item.status : undefined,
    description: typeof item.description === "string" ? item.description : undefined,
  };
}

function normalizeSnapshot(snapshot: unknown): TodoSnapshot {
  if (!snapshot || typeof snapshot !== "object") {
    return { available: false, error: false, items: [] };
  }
  const raw = snapshot as Record<string, unknown>;
  const items = Array.isArray(raw.items)
    ? raw.items.map(normalizeTodoItem).filter((item): item is TodoSnapshotItem => Boolean(item))
    : [];
  return {
    available: raw.available === true,
    error: raw.error === true,
    progress_text: typeof raw.progress_text === "string" ? raw.progress_text : undefined,
    items,
  };
}

function statusClassName(status: string | undefined) {
  return status ? status.replace(/[^a-z0-9_-]+/gi, "-") : "unknown";
}

export function TodoPopover({ snapshot }: TodoPopoverProps) {
  const todo = normalizeSnapshot(snapshot);

  return (
    <section className="todoPopover" role="dialog" aria-label="Todo">
      <header className="todoPopoverHeader">
        <h3>Todo</h3>
        {todo.available && todo.progress_text ? <p className="todoPopoverSummary">{todo.progress_text}</p> : null}
      </header>
      {!todo.available ? (
        <p className="todoPopoverEmpty">{todo.error ? "Todo list unavailable" : "No todo list yet"}</p>
      ) : (
        <div className="todoPopoverList">
          {todo.items.map((item, index) => (
            <article key={`${item.title || "todo"}-${index}`} className="todoPopoverItem">
              <div className="todoPopoverItemHead">
                <strong>{item.title || "Untitled todo"}</strong>
                <span className={`todoPopoverStatus ${statusClassName(item.status)}`}>{item.status || "unknown"}</span>
              </div>
              {item.description ? <p>{item.description}</p> : null}
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
```

- [ ] **Step 4: Run the focused component test to verify it passes**

Run: `cd web && npx vitest run src/components/workspace/TodoPopover.test.tsx`
Expected: PASS with 4 tests passing.

## Task 2: Integrate the popover into `AppShell` with Pi-only visibility and close behavior

**Files:**
- Modify: `web/src/app/AppShell.tsx`
- Modify: `web/src/app/AppShell.test.tsx`

- [ ] **Step 1: Extend `AppShell` tests with the failing toolbar/popover behavior**

Add these two test cases and tighten the existing `Details` assertion so it does not depend on hard-coded button indexes:

```tsx
it("shows the Todo button only for Pi sessions", () => {
  const sessionsStore = createStaticStore(
    {
      items: [
        { session_id: "sess-pi", alias: "Pi", agent_backend: "pi" },
        { session_id: "sess-codex", alias: "Codex", agent_backend: "codex" },
      ],
      activeSessionId: "sess-pi",
      loading: false,
      newSessionDefaults: null,
    },
    { refresh: vi.fn(), select: vi.fn() },
  );
  const messagesStore = createStaticStore(
    { bySessionId: { "sess-pi": [] }, offsetsBySessionId: { "sess-pi": 0 }, loading: false },
    { loadInitial: vi.fn(), poll: vi.fn() },
  );
  const composerStore = createStaticStore(
    { draft: "", sending: false },
    { setDraft: vi.fn(), submit: vi.fn() },
  );
  const sessionUiStore = createStaticStore(
    {
      sessionId: "sess-pi",
      diagnostics: {
        todo_snapshot: {
          available: true,
          error: false,
          progress_text: "1/2 completed",
          items: [{ title: "Restore popover", status: "in-progress" }],
        },
      },
      queue: null,
      files: [],
      requests: [],
      loading: false,
    },
    { refresh: vi.fn() },
  );

  root = document.createElement("div");
  document.body.appendChild(root);
  render(
    <AppProviders
      sessionsStore={sessionsStore as any}
      messagesStore={messagesStore as any}
      composerStore={composerStore as any}
      sessionUiStore={sessionUiStore as any}
    >
      <AppShell />
    </AppProviders>,
    root,
  );

  expect(root.textContent).toContain("Todo");
  render(
    <AppProviders
      sessionsStore={createStaticStore(
        {
          items: [{ session_id: "sess-codex", alias: "Codex", agent_backend: "codex" }],
          activeSessionId: "sess-codex",
          loading: false,
          newSessionDefaults: null,
        },
        { refresh: vi.fn(), select: vi.fn() },
      ) as any}
      messagesStore={messagesStore as any}
      composerStore={composerStore as any}
      sessionUiStore={createStaticStore(
        { sessionId: "sess-codex", diagnostics: {}, queue: null, files: [], requests: [], loading: false },
        { refresh: vi.fn() },
      ) as any}
    >
      <AppShell />
    </AppProviders>,
    root,
  );

  expect(root.textContent).not.toContain("Todo");
});

it("toggles the Todo popover for Pi sessions and closes it on outside click", () => {
  const sessionsStore = createStaticStore(
    {
      items: [{ session_id: "sess-1", alias: "Pi", agent_backend: "pi" }],
      activeSessionId: "sess-1",
      loading: false,
      newSessionDefaults: null,
    },
    { refresh: vi.fn(), select: vi.fn() },
  );
  const messagesStore = createStaticStore(
    { bySessionId: { "sess-1": [] }, offsetsBySessionId: { "sess-1": 0 }, loading: false },
    { loadInitial: vi.fn(), poll: vi.fn() },
  );
  const composerStore = createStaticStore(
    { draft: "", sending: false },
    { setDraft: vi.fn(), submit: vi.fn() },
  );
  const sessionUiStore = createStaticStore(
    {
      sessionId: "sess-1",
      diagnostics: {
        todo_snapshot: {
          available: true,
          error: false,
          progress_text: "1/1 completed",
          items: [{ title: "Restore popover", status: "completed" }],
        },
      },
      queue: null,
      files: [],
      requests: [],
      loading: false,
    },
    { refresh: vi.fn() },
  );

  root = document.createElement("div");
  document.body.appendChild(root);
  render(
    <AppProviders
      sessionsStore={sessionsStore as any}
      messagesStore={messagesStore as any}
      composerStore={composerStore as any}
      sessionUiStore={sessionUiStore as any}
    >
      <AppShell />
    </AppProviders>,
    root,
  );

  const todoButton = Array.from(root.querySelectorAll("button")).find((button) => button.textContent?.includes("Todo")) as HTMLButtonElement;
  todoButton.click();
  expect(root.querySelector(".todoPopover")).not.toBeNull();
  expect(root.textContent).toContain("1/1 completed");

  document.body.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
  expect(root.querySelector(".todoPopover")).toBeNull();
});
```

Also replace the brittle details-button lookup with:

```tsx
const detailsButton = Array.from(root.querySelectorAll("button")).find((button) => button.textContent?.includes("Details")) as HTMLButtonElement;
```

- [ ] **Step 2: Run the focused shell test to verify it fails**

Run: `cd web && npx vitest run src/app/AppShell.test.tsx`
Expected: FAIL because no `Todo` button or `.todoPopover` exists yet.

- [ ] **Step 3: Add Pi-only toolbar integration and close behavior in `AppShell`**

```tsx
// web/src/app/AppShell.tsx
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { TodoPopover } from "../components/workspace/TodoPopover";
import type { TodoSnapshot } from "../lib/types";

export function AppShell() {
  const { activeSessionId, items } = useSessionsStore();
  const { requests, diagnostics } = useSessionUiStore();
  const [todoOpen, setTodoOpen] = useState(false);
  const todoAnchorRef = useRef<HTMLDivElement | null>(null);

  const activeSession = items.find((session) => session.session_id === activeSessionId) ?? null;
  const isPiSession = activeSession?.agent_backend === "pi";
  const todoSnapshot = ((diagnostics as { todo_snapshot?: TodoSnapshot } | null)?.todo_snapshot ?? null) as TodoSnapshot | null;

  useEffect(() => {
    setTodoOpen(false);
  }, [activeSessionId, isPiSession]);

  useEffect(() => {
    if (!todoOpen) {
      return;
    }

    const onPointerDown = (event: MouseEvent) => {
      if (todoAnchorRef.current?.contains(event.target as Node)) {
        return;
      }
      setTodoOpen(false);
    };

    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [todoOpen]);

  return (
    <>
      <div className={shellClassName}>
        {/* existing shell */}
        <section className="conversationColumn">
          <div className="conversationToolbar">
            <div className="conversationToolbarGroup">
              {/* existing title controls */}
            </div>
            <div className="conversationToolbarGroup">
              <button type="button" className="toolbarButton" disabled={!activeSessionId}>
                <span className="buttonGlyph">📄</span>
                <span className="visuallyHidden">View file</span>
              </button>
              {isPiSession ? (
                <div className="todoToolbarAnchor" ref={todoAnchorRef}>
                  <button
                    type="button"
                    className={`toolbarButton todoToggle${todoOpen ? " isActive" : ""}`}
                    aria-expanded={todoOpen ? "true" : "false"}
                    onClick={() => setTodoOpen((current) => !current)}
                  >
                    <span className="buttonGlyph">☑</span>
                    <span>Todo</span>
                  </button>
                  {todoOpen ? <TodoPopover snapshot={todoSnapshot} /> : null}
                </div>
              ) : null}
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

- [ ] **Step 4: Run the focused shell test to verify it passes**

Run: `cd web && npx vitest run src/app/AppShell.test.tsx`
Expected: PASS with the new Pi-only `Todo` behavior covered.

## Task 3: Add floating-window styles and lock the selectors with regression coverage

**Files:**
- Modify: `web/src/styles/global.css`
- Modify: `web/src/styles/layout-scroll.test.ts`

- [ ] **Step 1: Add the failing CSS regression assertions**

Append a focused assertion block:

```ts
it("keeps todo popover selectors in the global shell stylesheet", () => {
  expect(css).toMatch(/\.todoToolbarAnchor\s*\{/);
  expect(css).toMatch(/\.toolbarButton\.todoToggle\.isActive\s*\{/);
  expect(css).toMatch(/\.todoPopover\s*\{/);
  expect(css).toMatch(/\.todoPopoverStatus\.completed\s*\{/);
  expect(css).toMatch(/@media \(max-width:\s*880px\)[\s\S]*\.todoPopover\s*\{/);
});
```

- [ ] **Step 2: Run the style regression test to verify it fails**

Run: `cd web && npx vitest run src/styles/layout-scroll.test.ts`
Expected: FAIL because the todo popover selectors are not in `src/styles/global.css` yet.

- [ ] **Step 3: Add the anchored floating-window styles**

```css
/* web/src/styles/global.css */
.todoToolbarAnchor {
  position: relative;
  display: inline-flex;
}

.toolbarButton.todoToggle.isActive {
  background: color-mix(in srgb, var(--accent) 12%, white);
  border-color: color-mix(in srgb, var(--accent) 40%, var(--border));
}

.todoPopover {
  position: absolute;
  top: calc(100% + 10px);
  right: 0;
  width: min(360px, 92vw);
  max-height: min(70dvh, 520px);
  overflow: auto;
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: 18px;
  background: color-mix(in srgb, var(--panel) 96%, #f7fafc);
  box-shadow: 0 18px 48px rgba(15, 23, 42, 0.18);
  z-index: 30;
}

.todoPopoverHeader h3,
.todoPopoverHeader p,
.todoPopoverItem p {
  margin: 0;
}

.todoPopoverHeader {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 10px;
}

.todoPopoverSummary,
.todoPopoverEmpty,
.todoPopoverItem p {
  color: var(--muted);
  font-size: 0.92rem;
}

.todoPopoverList {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.todoPopoverItem {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.78);
  border: 1px solid rgba(15, 23, 42, 0.08);
}

.todoPopoverItemHead {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
}

.todoPopoverStatus {
  flex: 0 0 auto;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.08);
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-size: 10px;
  line-height: 1.5;
}

.todoPopoverStatus.completed {
  background: rgba(22, 163, 74, 0.12);
  color: #166534;
}

.todoPopoverStatus.in-progress {
  background: rgba(37, 99, 235, 0.12);
  color: #1d4ed8;
}

.todoPopoverStatus.not-started {
  background: rgba(245, 158, 11, 0.14);
  color: #92400e;
}

@media (max-width: 880px) {
  .todoToolbarAnchor {
    position: static;
  }

  .todoPopover {
    position: fixed;
    top: 72px;
    right: 12px;
    left: 12px;
    width: auto;
    max-height: min(68dvh, 520px);
  }
}
```

- [ ] **Step 4: Run the targeted and broader verification commands**

Run: `cd web && npx vitest run src/styles/layout-scroll.test.ts src/app/AppShell.test.tsx src/components/workspace/TodoPopover.test.tsx`
Expected: PASS with all targeted popover tests green.

Run: `cd web && npx tsc --noEmit -p tsconfig.test.json && npx vitest run src/app/AppShell.test.tsx src/components/workspace/TodoPopover.test.tsx src/styles/layout-scroll.test.ts`
Expected: PASS with no TypeScript errors and all selected Vitest files green.

## Self-Review

### Spec coverage

- Pi-only toolbar button: covered in Task 2
- Floating popover surface: covered in Tasks 1 and 3
- Reuse of `diagnostics.todo_snapshot`: covered in Task 2 implementation
- Empty and unavailable states: covered in Task 1 tests and implementation
- No backend/API changes: preserved by file list and architecture
- `Details` semantics unchanged: preserved by leaving `SessionWorkspace` untouched

### Placeholder scan

- No `TODO`, `TBD`, or deferred implementation notes remain
- All tasks reference exact file paths
- Every code-changing step includes concrete code snippets
- Commands are explicit and runnable from the repo root

### Type consistency

- `TodoSnapshot` / `TodoSnapshotItem` names are consistent across `types.ts`, `TodoPopover.tsx`, and `AppShell.tsx`
- CSS selectors match the class names used by `TodoPopover.tsx` and `AppShell.tsx`
- Test expectations match the rendered class names and copy
