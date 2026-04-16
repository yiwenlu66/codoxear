# Legacy Web Parity Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the new `web/` frontend so it feels much closer to the legacy Codoxear UI from commit `c1e1c24` without regressing back to a monolithic imperative frontend.

**Architecture:** Keep the current `preact` component tree and store boundaries, but rebuild the shell, sidebar, conversation surface, composer, dialog, and workspace around the legacy UI semantics. Treat `c1e1c24:codoxear/static/app.css` and `c1e1c24:codoxear/static/app.js` as the visual and interaction baseline, then translate those behaviors into focused components plus shared CSS tokens in `web/src/styles/*`.

**Tech Stack:** `preact`, `typescript`, `vite`, `vitest`, existing `web/src` stores, Python-hosted `/api/*` backend, legacy CSS/JS as parity reference

---

## File Map

### Existing files to modify

- `web/src/lib/types.ts` - extend session/message UI types so the new components can render legacy-like metadata without unsafe casts
- `web/src/app/AppShell.tsx` - restore the legacy shell, sidebar open state, backdrop, and workspace placement
- `web/src/components/sessions/SessionsPane.tsx` - replace simple buttons with session-card rendering
- `web/src/components/conversation/ConversationPane.tsx` - render legacy-like message surfaces instead of plain `pre` blocks
- `web/src/components/composer/Composer.tsx` - restore legacy-like composer shell and input behavior
- `web/src/components/new-session/NewSessionDialog.tsx` - restore legacy modal grouping and structure
- `web/src/components/workspace/SessionWorkspace.tsx` - replace raw JSON-heavy output with structured workspace sections
- `web/src/styles/theme.css` - restore legacy token set
- `web/src/styles/global.css` - restore legacy shell, card, modal, and responsive rules
- `web/src/components/conversation/ConversationPane.test.tsx` - update expectations for structured message rendering
- `web/src/components/composer/Composer.test.tsx` - keep keyboard behavior coverage while adapting to the new composer shell
- `web/src/components/new-session/NewSessionDialog.test.tsx` - update modal structure expectations while preserving create flow coverage
- `web/src/components/workspace/SessionWorkspace.test.tsx` - verify structured workspace sections and request controls

### New files to create

- `web/src/app/AppShell.test.tsx` - lock shell layout and mobile sidebar behavior
- `web/src/components/sessions/SessionsPane.test.tsx` - lock session-card rendering and metadata hierarchy
- `web/src/components/sessions/SessionCard.tsx` - small presentational component for the sidebar cards

### Verification commands used repeatedly

- `cd web && npx vitest run src/app/AppShell.test.tsx`
- `cd web && npx vitest run src/components/sessions/SessionsPane.test.tsx`
- `cd web && npx vitest run src/components/conversation/ConversationPane.test.tsx src/components/composer/Composer.test.tsx`
- `cd web && npx vitest run src/components/new-session/NewSessionDialog.test.tsx src/components/workspace/SessionWorkspace.test.tsx`
- `cd web && npm test`
- `cd web && npm run build`
- `python3 -m pytest tests/test_vite_dist_serving.py tests/test_vite_asset_versioning.py tests/test_frontend_contract_source.py -q`

## Task 1: Rebuild the shell contract around legacy layout semantics

**Files:**
- Create: `web/src/app/AppShell.test.tsx`
- Modify: `web/src/lib/types.ts`
- Modify: `web/src/app/AppShell.tsx`
- Modify: `web/src/styles/theme.css`
- Modify: `web/src/styles/global.css`
- Test: `web/src/app/App.test.tsx`

- [ ] **Step 1: Write the failing shell test for the restored sidebar/backdrop/workspace layout**

```tsx
import { render } from "preact";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppProviders } from "./providers";
import { AppShell } from "./AppShell";

function createStaticStore<TState extends object, TActions extends Record<string, (...args: any[]) => any>>(
  state: TState,
  actions: TActions,
) {
  return {
    getState: () => state,
    subscribe: () => () => undefined,
    ...actions,
  };
}

describe("AppShell", () => {
  let root: HTMLDivElement | null = null;

  afterEach(() => {
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("renders the restored app shell with sidebar toggle, backdrop, and workspace rail", () => {
    const sessionsStore = createStaticStore(
      {
        items: [{ session_id: "sess-1", alias: "Legacy shell", agent_backend: "pi", busy: true }],
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
      { sessionId: "sess-1", diagnostics: null, queue: null, files: [], requests: [], loading: false },
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

    expect(root.querySelector(".appShell.legacyShell")).not.toBeNull();
    expect(root.querySelector(".sidebarToggle")).not.toBeNull();
    expect(root.querySelector(".sidebarBackdrop")).not.toBeNull();
    expect(root.querySelector(".workspacePane")).not.toBeNull();
    expect(root.textContent).toContain("New session");
  });
});
```

- [ ] **Step 2: Run the shell test and confirm it fails against the current simplified layout**

Run: `cd web && npx vitest run src/app/AppShell.test.tsx`
Expected: FAIL because `.legacyShell`, `.sidebarToggle`, and `.sidebarBackdrop` do not exist yet.

- [ ] **Step 3: Extend the frontend types so shell and sidebar code can render legacy metadata safely**

```ts
// web/src/lib/types.ts
export interface SessionSummary {
  session_id: string;
  title?: string;
  alias?: string;
  first_user_message?: string;
  cwd?: string;
  files?: string[];
  owned?: boolean;
  busy?: boolean;
  queue_len?: number;
  updated_ts?: number;
  agent_backend?: "codex" | "pi" | string;
  broker_pid?: number;
  git_branch?: string | null;
  model?: string | null;
  provider_choice?: string | null;
  reasoning_effort?: string | null;
}
```

- [ ] **Step 4: Replace the shell markup with a legacy-like layout controller and responsive sidebar state**

```tsx
// web/src/app/AppShell.tsx
import { useEffect, useMemo, useState } from "preact/hooks";

export function AppShell() {
  const { activeSessionId, items } = useSessionsStore();
  const [newSessionOpen, setNewSessionOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const activeSession = items.find((session) => session.session_id === activeSessionId) ?? null;

  useEffect(() => {
    if (!activeSessionId) {
      setSidebarOpen(items.length > 0);
      return;
    }
    setSidebarOpen(false);
  }, [activeSessionId, items.length]);

  const shellClassName = useMemo(
    () => `appShell legacyShell${sidebarOpen ? " sidebarOpen" : ""}`,
    [sidebarOpen],
  );

  return (
    <>
      <div className={shellClassName}>
        <button type="button" className="sidebarToggle" onClick={() => setSidebarOpen((value) => !value)}>
          Sessions
        </button>
        <div className="sidebarBackdrop" onClick={() => setSidebarOpen(false)} />
        <aside className="sidebarColumn">
          <div className="sidebarActions">
            <button type="button" className="primaryAction" onClick={() => setNewSessionOpen(true)}>
              New session
            </button>
          </div>
          <SessionsPane />
        </aside>
        <section className="conversationColumn">
          <ConversationPane />
          <Composer />
        </section>
        <SessionWorkspace />
      </div>
      <NewSessionDialog open={newSessionOpen} onClose={() => setNewSessionOpen(false)} />
    </>
  );
}
```

- [ ] **Step 5: Restore the shared shell tokens and responsive layout rules that every subsequent task depends on**

```css
/* web/src/styles/theme.css */
:root {
  color-scheme: light;
  --bg: #e9eef5;
  --panel: #ffffff;
  --border: rgba(15, 23, 42, 0.12);
  --text: #111827;
  --muted: rgba(17, 24, 39, 0.6);
  --accent: #1d4ed8;
  --accent-weak: rgba(29, 78, 216, 0.1);
  --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.08);
  --sidebar-w: 320px;
  --workspace-w: 320px;
  --bubble-user: #cfe7ff;
  --bubble-assistant: #ffffff;
}
```

```css
/* web/src/styles/global.css */
.appShell.legacyShell {
  display: grid;
  grid-template-columns: var(--sidebar-w) minmax(0, 1fr) var(--workspace-w);
  height: 100dvh;
  background: var(--bg);
  color: var(--text);
  overflow: hidden;
}

.sidebarToggle,
.sidebarBackdrop {
  display: none;
}

@media (max-width: 880px) {
  .appShell.legacyShell {
    grid-template-columns: 1fr;
  }

  .sidebarToggle {
    display: inline-flex;
    position: fixed;
    top: 12px;
    left: 12px;
    z-index: 40;
  }

  .sidebarColumn {
    position: fixed;
    inset: 0 auto 0 0;
    width: min(92vw, 360px);
    transform: translateX(-100%);
    transition: transform 180ms ease;
    z-index: 50;
  }

  .appShell.legacyShell.sidebarOpen .sidebarColumn {
    transform: translateX(0);
  }

  .sidebarBackdrop {
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 42, 0.28);
    z-index: 45;
  }

  .appShell.legacyShell.sidebarOpen .sidebarBackdrop {
    display: block;
  }
}
```

- [ ] **Step 6: Run the app-level tests that touch the shell and confirm they pass**

Run: `cd web && npx vitest run src/app/AppShell.test.tsx src/app/App.test.tsx`
Expected: PASS with the restored shell classes and no regressions in the app bootstrap test.

- [ ] **Step 7: Commit the shell baseline before moving deeper into the UI**

```bash
git add web/src/lib/types.ts web/src/app/AppShell.tsx web/src/app/AppShell.test.tsx web/src/styles/theme.css web/src/styles/global.css web/src/app/App.test.tsx
 git commit -m "feat(web): restore legacy shell layout baseline"
```

## Task 2: Restore the sidebar as legacy-style session cards

**Files:**
- Create: `web/src/components/sessions/SessionCard.tsx`
- Create: `web/src/components/sessions/SessionsPane.test.tsx`
- Modify: `web/src/components/sessions/SessionsPane.tsx`
- Modify: `web/src/styles/global.css`

- [ ] **Step 1: Write the failing sidebar test for metadata-rich session cards**

```tsx
import { render } from "preact";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppProviders } from "../../app/providers";
import { SessionsPane } from "./SessionsPane";

function createStaticStore(state: any) {
  return {
    getState: () => state,
    subscribe: () => () => undefined,
    refresh: vi.fn(),
    select: vi.fn(),
  };
}

describe("SessionsPane", () => {
  let root: HTMLDivElement | null = null;

  afterEach(() => {
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("renders legacy-like session cards with alias, backend, busy state, and preview text", () => {
    const sessionsStore = createStaticStore({
      items: [
        {
          session_id: "sess-1",
          alias: "Inbox cleanup",
          first_user_message: "整理一下今天的会话",
          agent_backend: "pi",
          busy: true,
          owned: true,
          queue_len: 2,
        },
      ],
      activeSessionId: "sess-1",
      loading: false,
      newSessionDefaults: null,
    });

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any}>
        <SessionsPane />
      </AppProviders>,
      root,
    );

    expect(root.querySelector(".sessionsPane .sessionCard.active")).not.toBeNull();
    expect(root.textContent).toContain("Inbox cleanup");
    expect(root.textContent).toContain("pi");
    expect(root.textContent).toContain("整理一下今天的会话");
    expect(root.querySelector(".stateDot.busy")).not.toBeNull();
    expect(root.querySelector(".queueBadge")).not.toBeNull();
  });
});
```

- [ ] **Step 2: Run the sidebar test and confirm the current one-line buttons are insufficient**

Run: `cd web && npx vitest run src/components/sessions/SessionsPane.test.tsx`
Expected: FAIL because `.sessionCard`, `.stateDot`, and `.queueBadge` do not exist.

- [ ] **Step 3: Introduce a focused `SessionCard` component so the sidebar does not become another oversized file**

```tsx
// web/src/components/sessions/SessionCard.tsx
import type { SessionSummary } from "../../lib/types";

interface SessionCardProps {
  session: SessionSummary;
  active: boolean;
  onSelect: () => void;
}

export function SessionCard({ session, active, onSelect }: SessionCardProps) {
  const title = session.alias || session.title || session.session_id;
  const preview = session.first_user_message || session.cwd || "No messages yet";

  return (
    <button type="button" className={`sessionCard${active ? " active" : ""}`} onClick={onSelect}>
      <div className="sessionMetaLine">
        <span className={`stateDot${session.busy ? " busy" : ""}`} />
        <span className="backendBadge">{session.agent_backend || "codex"}</span>
        {session.owned ? <span className="ownerBadge">web</span> : null}
        {session.queue_len ? <span className="queueBadge">{session.queue_len}</span> : null}
      </div>
      <div className="sessionTitle">{title}</div>
      <div className="sessionPreview">{preview}</div>
    </button>
  );
}
```

- [ ] **Step 4: Rebuild `SessionsPane` around the new card component and a legacy-like sidebar header**

```tsx
// web/src/components/sessions/SessionsPane.tsx
import { useSessionsStore, useSessionsStoreApi } from "../../app/providers";
import { SessionCard } from "./SessionCard";

export function SessionsPane() {
  const { items, activeSessionId } = useSessionsStore();
  const sessionsStoreApi = useSessionsStoreApi();

  return (
    <aside className="sessionsPane">
      <header className="sessionsHeader">
        <h1>Codoxear</h1>
        <p>Active sessions</p>
      </header>
      <div className="sessionsList">
        {items.map((session) => (
          <SessionCard
            key={session.session_id}
            session={session}
            active={session.session_id === activeSessionId}
            onSelect={() => sessionsStoreApi.select(session.session_id)}
          />
        ))}
      </div>
    </aside>
  );
}
```

- [ ] **Step 5: Add the session-card CSS rules that match the legacy sidebar density**

```css
.sessionsPane {
  border-right: 1px solid var(--border);
  background: var(--panel);
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.sessionsHeader {
  padding: 12px 14px;
  border-bottom: 1px solid var(--border);
}

.sessionsList {
  padding: 8px;
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.sessionCard {
  width: 100%;
  text-align: left;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: #fff;
  box-shadow: var(--shadow-sm);
}

.sessionCard.active {
  background: #dbeafe;
  border-color: rgba(37, 99, 235, 0.55);
}

.sessionMetaLine,
.sessionTitle,
.sessionPreview {
  display: block;
}
```

- [ ] **Step 6: Run the sidebar tests and confirm card rendering works before moving on**

Run: `cd web && npx vitest run src/components/sessions/SessionsPane.test.tsx`
Expected: PASS with alias, backend, busy state, preview text, and queue badge rendered.

- [ ] **Step 7: Commit the sidebar parity work**

```bash
git add web/src/components/sessions/SessionCard.tsx web/src/components/sessions/SessionsPane.tsx web/src/components/sessions/SessionsPane.test.tsx web/src/styles/global.css
 git commit -m "feat(web): restore legacy session sidebar cards"
```

## Task 3: Restore the conversation surface and composer shell

**Files:**
- Modify: `web/src/components/conversation/ConversationPane.tsx`
- Modify: `web/src/components/conversation/ConversationPane.test.tsx`
- Modify: `web/src/components/composer/Composer.tsx`
- Modify: `web/src/components/composer/Composer.test.tsx`
- Modify: `web/src/styles/global.css`

- [ ] **Step 1: Add a failing conversation test for role-aware message cards instead of raw `pre` output**

```tsx
it("renders user, assistant, and ask_user events with legacy-like message classes", () => {
  const sessionsStore = createStaticStore(
    { items: [], activeSessionId: "sess-1", loading: false, newSessionDefaults: null },
    { refresh: () => Promise.resolve(), select: () => undefined },
  );
  const messagesStore = createStaticStore(
    {
      bySessionId: {
        "sess-1": [
          { role: "user", text: "Hello" },
          { type: "ask_user", question: "Choose a provider", answer: "openai", resolved: true },
          { role: "assistant", text: "Done." },
        ],
      },
      offsetsBySessionId: { "sess-1": 3 },
      loading: false,
    },
    { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
  );

  render(
    <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
      <ConversationPane />
    </AppProviders>,
    root!,
  );

  expect(root!.querySelectorAll(".messageBubble")).toHaveLength(3);
  expect(root!.querySelector(".messageBubble.user")).not.toBeNull();
  expect(root!.querySelector(".messageBubble.assistant")).not.toBeNull();
  expect(root!.querySelector(".messageBubble.ask_user")).not.toBeNull();
});
```

- [ ] **Step 2: Add a failing composer test for the restored shell classes while preserving keyboard behavior**

```tsx
it("renders the legacy-style composer shell", () => {
  const submit = vi.fn().mockResolvedValue(undefined);
  const sessionsStore = createStore(
    { items: [], activeSessionId: "sess-1", loading: false, newSessionDefaults: null },
    () => ({ refresh: vi.fn(), select: vi.fn() }),
  );
  const composerStore = createStore(
    { draft: "Hello", sending: false },
    (setState) => ({
      setDraft(value: string) {
        setState({ draft: value });
      },
      submit,
    }),
  );

  render(
    <AppProviders sessionsStore={sessionsStore as any} composerStore={composerStore as any}>
      <Composer />
    </AppProviders>,
    root!,
  );

  expect(root!.querySelector(".composerShell")).not.toBeNull();
  expect(root!.querySelector(".composerInputWrap")).not.toBeNull();
  expect(root!.querySelector("button.sendButton")).not.toBeNull();
});
```

- [ ] **Step 3: Run the conversation and composer tests to see the current structure fail the new expectations**

Run: `cd web && npx vitest run src/components/conversation/ConversationPane.test.tsx src/components/composer/Composer.test.tsx`
Expected: FAIL because the current components do not emit the new semantic classes and shell wrappers.

- [ ] **Step 4: Replace the raw message rendering with a role-aware message card structure**

```tsx
// web/src/components/conversation/ConversationPane.tsx
function eventRole(event: MessageEvent): string {
  if (typeof event.role === "string" && event.role) return event.role;
  if (typeof event.message?.role === "string" && event.message.role) return event.message.role;
  if (event.toolName === "ask_user" || event.type === "ask_user") return "ask_user";
  return event.type || "event";
}

export function ConversationPane() {
  const { activeSessionId } = useSessionsStore();
  const { bySessionId, loading } = useMessagesStore();
  const messages = activeSessionId ? bySessionId[activeSessionId] ?? [] : [];

  return (
    <section className="conversationPane">
      {loading ? <p className="conversationStatus">Loading messages...</p> : null}
      <div className="messageList">
        {messages.map((message, index) => {
          const role = eventRole(message);
          return (
            <article key={index} className={`messageBubble ${role}`}>
              <span className="messageRole">{role}</span>
              <div className="messageBody">{contentTextFromMessage(message)}</div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
```

- [ ] **Step 5: Rewrap the composer in the legacy shell while preserving the current submit flow and keybindings**

```tsx
// web/src/components/composer/Composer.tsx
export function Composer() {
  const { activeSessionId } = useSessionsStore();
  const { draft, sending } = useComposerStore();
  const composerStoreApi = useComposerStoreApi();

  return (
    <form className={`composer composerShell${draft.includes("\n") ? " multiline" : ""}`} onSubmit={(event) => {
      event.preventDefault();
      if (activeSessionId) composerStoreApi.submit(activeSessionId);
    }}>
      <div className="composerInputWrap">
        <textarea
          value={draft}
          className="composerTextarea"
          onInput={(event) => composerStoreApi.setDraft(event.currentTarget.value)}
          onKeyDown={(event) => {
            if (event.key !== "Enter" || event.isComposing || event.shiftKey) return;
            if (!enterToSendEnabled() && !event.ctrlKey && !event.metaKey) return;
            if (!activeSessionId) return;
            event.preventDefault();
            composerStoreApi.submit(activeSessionId).catch(() => undefined);
          }}
          disabled={sending}
        />
      </div>
      <button type="submit" className="sendButton" disabled={sending || !draft.trim()}>
        {sending ? "Sending..." : "Send"}
      </button>
    </form>
  );
}
```

- [ ] **Step 6: Add the corresponding message/composer styles before rerunning tests**

```css
.conversationPane {
  overflow: auto;
  padding: 20px 20px 12px;
}

.messageList {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.messageBubble {
  max-width: min(820px, 100%);
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid var(--border);
  white-space: pre-wrap;
}

.messageBubble.user {
  align-self: flex-end;
  background: var(--bubble-user);
}

.messageBubble.assistant,
.messageBubble.ask_user {
  align-self: flex-start;
  background: var(--bubble-assistant);
}

.composerShell {
  display: flex;
  gap: 10px;
  padding: 10px 14px calc(10px + env(safe-area-inset-bottom));
  border-top: 1px solid var(--border);
  background: var(--panel);
}

.composerInputWrap {
  flex: 1 1 auto;
  border: 1px solid var(--border);
  border-radius: 18px;
  background: #fff;
}
```

- [ ] **Step 7: Run the focused tests and confirm keyboard behavior still passes along with the new shell assertions**

Run: `cd web && npx vitest run src/components/conversation/ConversationPane.test.tsx src/components/composer/Composer.test.tsx`
Expected: PASS for message classes, readable ask-user text, Enter/ctrl+Enter behavior, and composer shell structure.

- [ ] **Step 8: Commit the conversation/composer parity work**

```bash
git add web/src/components/conversation/ConversationPane.tsx web/src/components/conversation/ConversationPane.test.tsx web/src/components/composer/Composer.tsx web/src/components/composer/Composer.test.tsx web/src/styles/global.css
 git commit -m "feat(web): restore legacy conversation and composer surfaces"
```

## Task 4: Rebuild the new-session dialog and workspace panel as structured surfaces

**Files:**
- Modify: `web/src/components/new-session/NewSessionDialog.tsx`
- Modify: `web/src/components/new-session/NewSessionDialog.test.tsx`
- Modify: `web/src/components/workspace/SessionWorkspace.tsx`
- Modify: `web/src/components/workspace/SessionWorkspace.test.tsx`
- Modify: `web/src/styles/global.css`

- [ ] **Step 1: Add a failing dialog test that locks the legacy-like grouped modal structure**

```tsx
it("renders grouped launch controls inside a legacy-style modal shell", async () => {
  const sessionsStore = createSessionsStore({
    items: [],
    activeSessionId: null,
    loading: false,
    newSessionDefaults: {
      default_backend: "pi",
      backends: { pi: { provider_choice: "macaron" }, codex: {} },
    },
  });

  render(
    <AppProviders sessionsStore={sessionsStore as any}>
      <NewSessionDialog open onClose={() => undefined} />
    </AppProviders>,
    root!,
  );

  expect(root!.querySelector(".dialogBackdrop")).not.toBeNull();
  expect(root!.querySelector(".dialogCard.legacyDialog")).not.toBeNull();
  expect(root!.querySelector(".dialogSection")).not.toBeNull();
  expect(root!.textContent).toContain("Backend");
  expect(root!.textContent).toContain("Working directory");
});
```

- [ ] **Step 2: Add a failing workspace test that forbids the JSON-dump presentation**

```tsx
it("renders diagnostics, queue, files, and requests as structured workspace sections", () => {
  const sessionUiStore = createStaticStore(
    {
      sessionId: "sess-1",
      diagnostics: { status: "ok" },
      queue: { items: [{ text: "next task" }] },
      files: ["src/main.tsx"],
      loading: false,
      requests: [],
    },
    { refresh: vi.fn() },
  );

  render(
    <AppProviders sessionUiStore={sessionUiStore as any}>
      <SessionWorkspace />
    </AppProviders>,
    root!,
  );

  expect(root!.querySelectorAll(".workspaceSection")).toHaveLength(4);
  expect(root!.querySelector("pre")).toBeNull();
  expect(root!.textContent).toContain("Diagnostics");
  expect(root!.textContent).toContain("next task");
  expect(root!.textContent).toContain("src/main.tsx");
});
```

- [ ] **Step 3: Run the dialog/workspace tests and confirm they fail against the current plain structure**

Run: `cd web && npx vitest run src/components/new-session/NewSessionDialog.test.tsx src/components/workspace/SessionWorkspace.test.tsx`
Expected: FAIL because the current dialog lacks grouped shell classes and the workspace still renders `pre` blocks.

- [ ] **Step 4: Reorganize the new-session dialog into grouped modal sections while preserving the create flow**

```tsx
// web/src/components/new-session/NewSessionDialog.tsx
return (
  <div className="dialogBackdrop" onClick={onClose}>
    <section className="dialogCard legacyDialog" onClick={(event) => event.stopPropagation()}>
      <header className="dialogHeader">
        <h2>New session</h2>
        <p>Launch a backend in a project directory.</p>
      </header>
      <form
        onSubmit={async (event) => {
          event.preventDefault();
          setSubmitting(true);
          setError("");
          try {
            const response = await api.createSession({
              cwd,
              backend,
              model: backendDefaults.model ?? undefined,
              reasoning_effort: backendDefaults.reasoning_effort ?? undefined,
              model_provider: backendDefaults.model_provider ?? backendDefaults.provider_choice ?? undefined,
            });
            await sessionsStoreApi.refresh();
            const createdSession = sessionsStoreApi
              .getState()
              .items.find((session) => session.broker_pid === response.broker_pid);
            if (createdSession) {
              sessionsStoreApi.select(createdSession.session_id);
            } else {
              await sessionsStoreApi.refresh({ preferNewest: true });
            }
            onClose();
          } catch (submitError) {
            setError(submitError instanceof Error ? submitError.message : "Failed to create session");
          } finally {
            setSubmitting(false);
          }
        }}
      >
        <div className="dialogSection">
          <label className="fieldBlock">
            <span>Backend</span>
            <select value={backend} onInput={(event) => setBackend(event.currentTarget.value)}>
              {Object.keys(newSessionDefaults?.backends || { codex: {}, pi: {} }).map((backendName) => (
                <option key={backendName} value={backendName}>{backendName}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="dialogSection">
          <label className="fieldBlock">
            <span>Working directory</span>
            <input value={cwd} onInput={(event) => setCwd(event.currentTarget.value)} placeholder="/path/to/project" />
          </label>
        </div>
        {error ? <p className="errorText">{error}</p> : null}
        <div className="formActions">
          <button type="button" onClick={onClose} disabled={submitting}>Cancel</button>
          <button type="submit" disabled={submitting || !cwd.trim()}>{submitting ? "Launching..." : "Launch"}</button>
        </div>
      </form>
    </section>
  </div>
);
```

- [ ] **Step 5: Replace the workspace `pre` blocks with structured sections while preserving request submission behavior**

```tsx
// web/src/components/workspace/SessionWorkspace.tsx
function keyValueRows(value: Record<string, unknown> | null) {
  return value ? Object.entries(value) : [];
}

export function SessionWorkspace() {
  const { sessionId, diagnostics, queue, files, requests, loading } = useSessionUiStore();

  return (
    <aside className="workspacePane">
      {loading ? <p className="workspaceStatus">Loading workspace...</p> : null}
      <section className="workspaceSection">
        <h3>Diagnostics</h3>
        <dl>{keyValueRows(diagnostics).map(([key, value]) => <><dt>{key}</dt><dd>{String(value)}</dd></>)}</dl>
      </section>
      <section className="workspaceSection">
        <h3>Queue</h3>
        <ul>{Array.isArray(queue?.items) ? queue.items.map((item, index) => <li key={index}>{String((item as any).text || item)}</li>) : <li>No queued items</li>}</ul>
      </section>
      <section className="workspaceSection">
        <h3>Files</h3>
        <ul>{files.length ? files.map((file) => <li key={file}>{file}</li>) : <li>No tracked files</li>}</ul>
      </section>
      <section className="workspaceSection">
        <h3>UI Requests</h3>
        {requests.length ? (
          requests.map((request, index) => {
            const requestId = String(request.id ?? index);
            const draftValue = drafts[requestId] ?? getInitialDraftValue(request);
            return (
              <div key={requestId} className="uiRequestCard">
                <strong>{getRequestHeading(request)}</strong>
                {getRequestBody(request) ? <p>{getRequestBody(request)}</p> : null}
                <button
                  type="button"
                  onClick={async () => {
                    const normalizedValue = normalizeRequestValue(request, draftValue);
                    await api.submitUiResponse(sessionId!, request.method === "confirm"
                      ? { id: request.id, confirmed: true }
                      : { id: request.id, value: normalizedValue });
                    await sessionUiStoreApi.refresh(sessionId!, { agentBackend: "pi" });
                  }}
                >
                  Confirm
                </button>
              </div>
            );
          })
        ) : (
          <p>No pending requests</p>
        )}
      </section>
    </aside>
  );
}
```

- [ ] **Step 6: Add the dialog and workspace CSS rules that make these surfaces read as part of the same legacy system**

```css
.dialogBackdrop {
  position: fixed;
  inset: 0;
  display: grid;
  place-items: center;
  background: rgba(15, 23, 42, 0.28);
  z-index: 80;
}

.dialogCard.legacyDialog,
.workspaceSection {
  border: 1px solid var(--border);
  border-radius: 16px;
  background: var(--panel);
  box-shadow: 0 12px 36px rgba(15, 23, 42, 0.12);
}

.dialogCard.legacyDialog {
  width: min(640px, calc(100vw - 24px));
  padding: 18px;
}

.dialogSection + .dialogSection {
  margin-top: 14px;
}

.workspacePane {
  padding: 12px;
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
  border-left: 1px solid var(--border);
  background: color-mix(in srgb, var(--panel) 92%, var(--bg));
}
```

- [ ] **Step 7: Run the focused tests and verify the modal flow and workspace request submission both still pass**

Run: `cd web && npx vitest run src/components/new-session/NewSessionDialog.test.tsx src/components/workspace/SessionWorkspace.test.tsx`
Expected: PASS for the modal create flow, structured workspace sections, and existing request submission behavior.

- [ ] **Step 8: Commit the dialog/workspace parity work**

```bash
git add web/src/components/new-session/NewSessionDialog.tsx web/src/components/new-session/NewSessionDialog.test.tsx web/src/components/workspace/SessionWorkspace.tsx web/src/components/workspace/SessionWorkspace.test.tsx web/src/styles/global.css
 git commit -m "feat(web): restore legacy dialog and workspace surfaces"
```

## Task 5: Finish secondary parity details and run the full regression pass

**Files:**
- Modify: `web/src/components/sessions/SessionCard.tsx`
- Modify: `web/src/styles/global.css`
- Verify: `web/src/**/*.test.tsx`
- Verify: `tests/test_vite_dist_serving.py`
- Verify: `tests/test_vite_asset_versioning.py`
- Verify: `tests/test_frontend_contract_source.py`

- [ ] **Step 1: Add a final failing sidebar assertion for the secondary metadata details that make the UI feel complete**

```tsx
expect(root!.querySelector(".sessionCard .sessionMetaLine")).not.toBeNull();
expect(root!.querySelector(".sessionCard .backendBadge")).not.toBeNull();
expect(root!.querySelector(".sessionCard .ownerBadge")).not.toBeNull();
```

- [ ] **Step 2: Run the sidebar test once more and confirm the new details are either missing or visually incomplete**

Run: `cd web && npx vitest run src/components/sessions/SessionsPane.test.tsx`
Expected: FAIL until the sidebar exposes the final metadata selectors and styling hooks.

- [ ] **Step 3: Tighten the final session-card polish and mobile-safe shell details**

```css
.sessionMetaLine {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.backendBadge,
.ownerBadge,
.queueBadge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 18px;
  padding: 0 6px;
  border-radius: 999px;
  font-size: 11px;
  line-height: 1;
}

.backendBadge {
  background: rgba(15, 23, 42, 0.06);
}

.ownerBadge {
  border: 1px solid var(--border);
  background: rgba(255, 255, 255, 0.92);
}

@media (max-width: 520px) {
  .composerShell {
    padding: 8px 10px calc(8px + env(safe-area-inset-bottom));
  }

  .workspacePane {
    display: none;
  }
}
```

- [ ] **Step 4: Run the complete frontend test suite, production build, and Python integration checks**

Run: `cd web && npm test && npm run build && cd .. && python3 -m pytest tests/test_vite_dist_serving.py tests/test_vite_asset_versioning.py tests/test_frontend_contract_source.py -q`
Expected: PASS for all Vitest coverage, successful Vite build into `codoxear/static/dist`, and passing Python integration checks.

- [ ] **Step 5: Manually compare the new UI against the legacy baseline in the browser before finalizing**

Run: `git show c1e1c24:codoxear/static/app.css >/tmp/legacy-app.css && git show c1e1c24:codoxear/static/app.js >/tmp/legacy-app.js && cd web && npm run dev`
Expected: Open the current app and compare its shell, sidebar, conversation area, composer, new-session dialog, and workspace against the legacy references; the new UI should no longer read as a stripped-down placeholder.

- [ ] **Step 6: Commit the finished parity restoration**

```bash
git add web/src/components/sessions/SessionCard.tsx web/src/styles/global.css web/src/app web/src/components web/src/lib/types.ts
 git commit -m "feat(web): restore legacy web parity"
```
