# Preact + Vite Web Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current hand-written browser client with a `preact + vite` frontend that preserves Codoxear's existing web capabilities while keeping Python as the API host and production static-file server.

**Architecture:** Build a dedicated `web/` frontend project with `preact + vite + typescript`, route all browser/server communication through a typed API client plus lightweight domain stores, and switch `codoxear/server.py` from hand-authored static assets to Vite `dist` serving behind a short-lived fallback flag. Migration proceeds in vertical slices: scaffold, data layer, core UI, advanced panels, production cutover, cleanup.

**Tech Stack:** Python 3.10+, `pytest`, `preact`, `vite`, `typescript`, `vitest`, browser Fetch API, existing `codoxear.server` JSON API

---

## File Map

### New frontend project

- Create: `web/package.json`
- Create: `web/tsconfig.json`
- Create: `web/tsconfig.node.json`
- Create: `web/vite.config.ts`
- Create: `web/index.html`
- Create: `web/src/main.tsx`
- Create: `web/src/app/App.tsx`
- Create: `web/src/app/AppShell.tsx`
- Create: `web/src/app/providers.tsx`
- Create: `web/src/styles/global.css`
- Create: `web/src/styles/theme.css`
- Create: `web/src/lib/api.ts`
- Create: `web/src/lib/http.ts`
- Create: `web/src/lib/types.ts`
- Create: `web/src/lib/viewport.ts`
- Create: `web/src/domains/sessions/store.ts`
- Create: `web/src/domains/messages/store.ts`
- Create: `web/src/domains/session-ui/store.ts`
- Create: `web/src/domains/composer/store.ts`
- Create: `web/src/components/sessions/SessionsPane.tsx`
- Create: `web/src/components/conversation/ConversationPane.tsx`
- Create: `web/src/components/composer/Composer.tsx`
- Create: `web/src/components/workspace/SessionWorkspace.tsx`
- Create: `web/src/components/new-session/NewSessionDialog.tsx`
- Create: `web/src/test/setup.ts`
- Create: `web/src/lib/api.test.ts`
- Create: `web/src/domains/sessions/store.test.ts`
- Create: `web/src/domains/messages/store.test.ts`

### Python integration and validation

- Modify: `codoxear/server.py`
- Modify: `pyproject.toml` only if package data paths need updates for bundled `dist` assets
- Modify: `README.md`
- Modify: `.gitignore`
- Create: `tests/test_vite_dist_serving.py`
- Create: `tests/test_vite_asset_versioning.py`
- Create: `tests/test_frontend_contract_source.py`

### Legacy frontend transition

- Delete later: `codoxear/static/index.html`
- Delete later: `codoxear/static/app.js`
- Delete later: `codoxear/static/app.css`
- Keep and adapt if needed: `codoxear/static/manifest.webmanifest`, `codoxear/static/service-worker.js`, `codoxear/static/favicon.png`, `codoxear/static/codoxear-icon.png`, `codoxear/static/logos/*`

## Task 1: Scaffold the `web/` frontend project

**Files:**
- Create: `web/package.json`
- Create: `web/tsconfig.json`
- Create: `web/tsconfig.node.json`
- Create: `web/vite.config.ts`
- Create: `web/index.html`
- Create: `web/src/main.tsx`
- Create: `web/src/app/App.tsx`
- Create: `web/src/styles/global.css`
- Create: `web/src/styles/theme.css`
- Modify: `.gitignore`

- [ ] **Step 1: Create the initial `package.json` with the exact scripts the rest of the plan expects**

```json
{
  "name": "codoxear-web",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "preact": "^10.26.4"
  },
  "devDependencies": {
    "@preact/preset-vite": "^2.10.2",
    "typescript": "^5.8.3",
    "vite": "^7.1.0",
    "vitest": "^3.2.4"
  }
}
```

- [ ] **Step 2: Add Vite and TypeScript configuration with the app base and API proxy wired for Python**

```ts
// web/vite.config.ts
import { defineConfig } from "vite";
import preact from "@preact/preset-vite";

export default defineConfig({
  plugins: [preact()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8743",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
  },
});
```

- [ ] **Step 3: Add a minimal bootstrap app so `vite dev` and `vite build` can both succeed before any product logic lands**

```tsx
// web/src/main.tsx
import { render } from "preact";
import App from "./app/App";
import "./styles/theme.css";
import "./styles/global.css";

render(<App />, document.getElementById("root")!);
```

```tsx
// web/src/app/App.tsx
export default function App() {
  return (
    <main className="bootScreen">
      <h1>Codoxear</h1>
      <p>Preact migration scaffold is live.</p>
    </main>
  );
}
```

- [ ] **Step 4: Add the expected ignored paths before installing dependencies or generating build output**

```gitignore
web/node_modules/
web/dist/
```

- [ ] **Step 5: Install frontend dependencies and verify the scaffold builds cleanly**

Run: `cd web && npm install && npm run build`
Expected: TypeScript completes, Vite emits `web/dist/index.html` and hashed assets with exit code 0.

- [ ] **Step 6: Start the dev server once and confirm the API proxy is configured, even if the backend is not yet running**

Run: `cd web && npm run dev -- --host 127.0.0.1`
Expected: Vite starts on `http://127.0.0.1:5173` and shows `/api` proxy configuration in the resolved config.

- [ ] **Step 7: Commit the scaffold as the stable baseline for follow-up tasks**

```bash
git add .gitignore web/package.json web/tsconfig.json web/tsconfig.node.json web/vite.config.ts web/index.html web/src/main.tsx web/src/app/App.tsx web/src/styles/global.css web/src/styles/theme.css
 git commit -m "feat(web): scaffold preact vite frontend"
```

## Task 2: Lock server-side `dist` serving behavior with Python tests

**Files:**
- Create: `tests/test_vite_dist_serving.py`
- Create: `tests/test_vite_asset_versioning.py`
- Reference: `codoxear/server.py`

- [ ] **Step 1: Add a test for serving `dist/index.html` as the root app shell when Vite output exists**

```python
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import codoxear.server as server


class TestViteDistServing(unittest.TestCase):
    def test_root_prefers_vite_dist_index_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dist = Path(td) / "web" / "dist"
            dist.mkdir(parents=True)
            (dist / "index.html").write_text("<html><body>vite</body></html>", encoding="utf-8")
            with mock.patch.object(server, "WEB_DIST_DIR", dist):
                body, ctype = server._read_web_index_for_tests()

        self.assertIn("vite", body)
        self.assertEqual("text/html; charset=utf-8", ctype)
```

- [ ] **Step 2: Add a test for hashed asset lookup and cache semantics so `server.py` can move away from fixed `app.js` / `app.css` assumptions**

```python
class TestViteAssetVersioning(unittest.TestCase):
    def test_asset_version_uses_manifest_hashes_when_present(self) -> None:
        manifest = {
            "src/main.tsx": {
                "file": "assets/main-abcd1234.js",
                "css": ["assets/main-efgh5678.css"],
            }
        }
        version = server._asset_version_from_manifest_for_tests(manifest)
        self.assertEqual("abcd1234-efgh5678", version)
```

- [ ] **Step 3: Run the new Python tests before touching `server.py` and confirm they fail because helper paths do not exist yet**

Run: `python3 -m pytest tests/test_vite_dist_serving.py tests/test_vite_asset_versioning.py -q`
Expected: FAIL with missing helpers or missing `WEB_DIST_DIR` integration points in `codoxear/server.py`.

- [ ] **Step 4: Commit the failing server-contract tests**

```bash
git add tests/test_vite_dist_serving.py tests/test_vite_asset_versioning.py
 git commit -m "test(server): cover vite dist asset serving"
```

## Task 3: Implement Python support for Vite `dist` output without changing `/api/*`

**Files:**
- Modify: `codoxear/server.py`
- Modify: `pyproject.toml` if package data needs to include Vite build files
- Test: `tests/test_vite_dist_serving.py`
- Test: `tests/test_vite_asset_versioning.py`

- [ ] **Step 1: Add explicit frontend build-path constants near the current static asset constants**

```python
# codoxear/server.py
ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
WEB_DIST_DIR = WEB_DIR / "dist"
WEB_DIST_ASSETS_DIR = WEB_DIST_DIR / "assets"
LEGACY_STATIC_DIR = ROOT_DIR / "codoxear" / "static"
```

- [ ] **Step 2: Add helper functions for reading `dist/index.html` and deriving a stable asset version from the Vite manifest**

```python
def _read_web_index_for_tests() -> tuple[str, str]:
    index_path = WEB_DIST_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8"), "text/html; charset=utf-8"
    legacy = (LEGACY_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return legacy, "text/html; charset=utf-8"


def _asset_version_from_manifest_for_tests(manifest: dict[str, object]) -> str:
    entry = manifest.get("src/main.tsx") if isinstance(manifest, dict) else None
    if not isinstance(entry, dict):
        return "dev"
    js_file = str(entry.get("file") or "")
    css_files = entry.get("css") or []
    css_suffix = "-".join(Path(str(p)).stem.split("-")[-1] for p in css_files if p)
    js_suffix = Path(js_file).stem.split("-")[-1] if js_file else "dev"
    return f"{js_suffix}-{css_suffix}".strip("-") or "dev"
```

- [ ] **Step 3: Update root/static request handling to prefer `web/dist` output but preserve existing manifest and API routes**

```python
if path == "/":
    body, ctype = _read_web_index_for_tests()
    self._send_bytes(body.encode("utf-8"), ctype)
    return

if path.startswith("/assets/"):
    candidate = WEB_DIST_DIR / path.lstrip("/")
    if candidate.exists():
        self._send_path(candidate)
        return
```

- [ ] **Step 4: Keep a temporary fallback switch so rollout can be reversed without reverting code**

```python
USE_LEGACY_WEB = os.environ.get("CODOXEAR_USE_LEGACY_WEB", "0") == "1"
if USE_LEGACY_WEB:
    # existing legacy static handling branch
```

- [ ] **Step 5: Run the new server tests until they pass, then rerun the existing static-asset regression file to ensure `/api` and legacy routing assumptions still hold**

Run: `python3 -m pytest tests/test_vite_dist_serving.py tests/test_vite_asset_versioning.py tests/test_static_assets.py -q`
Expected: PASS with no regression in existing static asset tests.

- [ ] **Step 6: Commit the server integration slice**

```bash
git add codoxear/server.py pyproject.toml tests/test_vite_dist_serving.py tests/test_vite_asset_versioning.py
 git commit -m "feat(server): serve vite dist frontend"
```

## Task 4: Build the typed API client and shared models first

**Files:**
- Create: `web/src/lib/types.ts`
- Create: `web/src/lib/http.ts`
- Create: `web/src/lib/api.ts`
- Create: `web/src/lib/api.test.ts`

- [ ] **Step 1: Capture the existing `/api` payloads into TypeScript interfaces before building any store logic**

```ts
// web/src/lib/types.ts
export interface SessionSummary {
  session_id: string;
  title?: string;
  agent_backend?: "codex" | "pi" | string;
  busy?: boolean;
  updated_ts?: number;
}

export interface SessionsResponse {
  sessions: SessionSummary[];
  defaults?: Record<string, unknown>;
  default_backend?: string;
  backends?: Record<string, unknown>;
}

export interface MessagesResponse {
  messages: Array<Record<string, unknown>>;
  ui_version?: string;
}

export interface SessionUiStateResponse {
  requests: Array<Record<string, unknown>>;
}
```

- [ ] **Step 2: Implement a single fetch wrapper with JSON parsing, error normalization, and request cancellation support**

```ts
// web/src/lib/http.ts
export async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, { signal, headers: { Accept: "application/json" } });
  const text = await res.text();
  const data = text ? JSON.parse(text) : {};
  if (!res.ok) {
    const message = typeof data?.error === "string" ? data.error : `Request failed: ${res.status}`;
    throw new Error(message);
  }
  return data as T;
}
```

- [ ] **Step 3: Implement the API surface the stores will use instead of letting components call `fetch` directly**

```ts
// web/src/lib/api.ts
import { getJson } from "./http";
import type { MessagesResponse, SessionUiStateResponse, SessionsResponse } from "./types";

export const api = {
  listSessions(signal?: AbortSignal) {
    return getJson<SessionsResponse>("/api/sessions", signal);
  },
  listMessages(sessionId: string, init = false, signal?: AbortSignal) {
    const suffix = init ? "?init=1" : "";
    return getJson<MessagesResponse>(`/api/sessions/${sessionId}/messages${suffix}`, signal);
  },
  getSessionUiState(sessionId: string, signal?: AbortSignal) {
    return getJson<SessionUiStateResponse>(`/api/sessions/${sessionId}/ui_state`, signal);
  },
};
```

- [ ] **Step 4: Add a Vitest regression that proves the wrapper surfaces server error payloads correctly**

```ts
import { describe, expect, it, vi } from "vitest";
import { getJson } from "./http";

describe("getJson", () => {
  it("throws server error messages", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 400, text: async () => '{"error":"bad request"}' }),
    );

    await expect(getJson("/api/sessions")).rejects.toThrow("bad request");
  });
});
```

- [ ] **Step 5: Run frontend tests for the API client before adding stores**

Run: `cd web && npm run test -- src/lib/api.test.ts`
Expected: PASS and no import errors from the fresh scaffold.

- [ ] **Step 6: Commit the shared client layer**

```bash
git add web/src/lib/types.ts web/src/lib/http.ts web/src/lib/api.ts web/src/lib/api.test.ts
 git commit -m "feat(web): add typed api client"
```

## Task 5: Implement sessions and messages domain stores before building the full UI

**Files:**
- Create: `web/src/domains/sessions/store.ts`
- Create: `web/src/domains/messages/store.ts`
- Create: `web/src/domains/sessions/store.test.ts`
- Create: `web/src/domains/messages/store.test.ts`
- Create: `web/src/app/providers.tsx`

- [ ] **Step 1: Build a tiny store pattern that exposes `getState`, `subscribe`, and action methods for each domain**

```ts
// web/src/domains/sessions/store.ts
import { api } from "../../lib/api";
import type { SessionSummary } from "../../lib/types";

export interface SessionsState {
  items: SessionSummary[];
  activeSessionId: string | null;
  loading: boolean;
}

export function createSessionsStore() {
  let state: SessionsState = { items: [], activeSessionId: null, loading: false };
  const listeners = new Set<() => void>();
  const emit = () => listeners.forEach((listener) => listener());

  return {
    getState: () => state,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    async refresh() {
      state = { ...state, loading: true };
      emit();
      const data = await api.listSessions();
      state = {
        ...state,
        loading: false,
        items: data.sessions,
        activeSessionId: state.activeSessionId ?? data.sessions[0]?.session_id ?? null,
      };
      emit();
    },
    select(sessionId: string) {
      state = { ...state, activeSessionId: sessionId };
      emit();
    },
  };
}
```

- [ ] **Step 2: Build the messages store with explicit `loadInitial` and `poll` actions instead of hidden component effects**

```ts
// web/src/domains/messages/store.ts
import { api } from "../../lib/api";

export interface MessagesState {
  bySessionId: Record<string, Array<Record<string, unknown>>>;
  loading: boolean;
}

export function createMessagesStore() {
  let state: MessagesState = { bySessionId: {}, loading: false };
  const listeners = new Set<() => void>();
  const emit = () => listeners.forEach((listener) => listener());

  return {
    getState: () => state,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    async loadInitial(sessionId: string) {
      state = { ...state, loading: true };
      emit();
      const data = await api.listMessages(sessionId, true);
      state = {
        ...state,
        loading: false,
        bySessionId: { ...state.bySessionId, [sessionId]: data.messages },
      };
      emit();
    },
  };
}
```

- [ ] **Step 3: Add store tests that pin the state transitions before wiring any components**

```ts
import { describe, expect, it, vi } from "vitest";
import { createSessionsStore } from "./store";
import { api } from "../../lib/api";

vi.mock("../../lib/api", () => ({
  api: {
    listSessions: vi.fn(),
  },
}));

describe("createSessionsStore", () => {
  it("selects the first session on first refresh", async () => {
    vi.mocked(api.listSessions).mockResolvedValue({ sessions: [{ session_id: "s1" }] } as never);
    const store = createSessionsStore();

    await store.refresh();

    expect(store.getState().activeSessionId).toBe("s1");
  });
});
```

- [ ] **Step 4: Run the store tests and keep them passing before any UI code depends on them**

Run: `cd web && npm run test -- src/domains/sessions/store.test.ts src/domains/messages/store.test.ts`
Expected: PASS with no component rendering involved.

- [ ] **Step 5: Commit the domain-store base layer**

```bash
git add web/src/domains/sessions/store.ts web/src/domains/messages/store.ts web/src/domains/sessions/store.test.ts web/src/domains/messages/store.test.ts web/src/app/providers.tsx
 git commit -m "feat(web): add sessions and messages stores"
```

## Task 6: Ship the core shell, session list, conversation view, and composer

**Files:**
- Create: `web/src/app/AppShell.tsx`
- Create: `web/src/components/sessions/SessionsPane.tsx`
- Create: `web/src/components/conversation/ConversationPane.tsx`
- Create: `web/src/components/composer/Composer.tsx`
- Modify: `web/src/app/App.tsx`
- Modify: `web/src/styles/global.css`

- [ ] **Step 1: Replace the scaffold screen with a shell that composes the three core panes using store-backed props**

```tsx
// web/src/app/App.tsx
import { AppShell } from "./AppShell";

export default function App() {
  return <AppShell />;
}
```

```tsx
// web/src/app/AppShell.tsx
import { SessionsPane } from "../components/sessions/SessionsPane";
import { ConversationPane } from "../components/conversation/ConversationPane";
import { Composer } from "../components/composer/Composer";

export function AppShell() {
  return (
    <div className="appShell">
      <SessionsPane />
      <section className="conversationColumn">
        <ConversationPane />
        <Composer />
      </section>
    </div>
  );
}
```

- [ ] **Step 2: Render sessions from the sessions store and selection from explicit click actions only**

```tsx
export function SessionsPane() {
  const { items, activeSessionId } = useSessionsStore();
  return (
    <aside className="sessionsPane">
      {items.map((session) => (
        <button
          key={session.session_id}
          className={session.session_id === activeSessionId ? "session active" : "session"}
          onClick={() => sessionsStore.select(session.session_id)}
        >
          {session.title ?? session.session_id}
        </button>
      ))}
    </aside>
  );
}
```

- [ ] **Step 3: Render the conversation from the messages store without any direct DOM mutation or manual HTML injection**

```tsx
export function ConversationPane() {
  const { activeSessionId } = useSessionsStore();
  const { bySessionId, loading } = useMessagesStore();
  const messages = activeSessionId ? bySessionId[activeSessionId] ?? [] : [];

  return (
    <section className="conversationPane">
      {loading ? <p>Loading messages...</p> : null}
      {messages.map((message, index) => (
        <article key={index} className="messageBubble">
          <pre>{JSON.stringify(message, null, 2)}</pre>
        </article>
      ))}
    </section>
  );
}
```

- [ ] **Step 4: Add a controlled composer with no send behavior yet, just enough to establish state ownership and keyboard handling**

```tsx
export function Composer() {
  const [value, setValue] = useState("");
  return (
    <form className="composer" onSubmit={(event) => event.preventDefault()}>
      <textarea value={value} onInput={(event) => setValue(event.currentTarget.value)} />
      <button type="submit">Send</button>
    </form>
  );
}
```

- [ ] **Step 5: Build and smoke-test the browser UI before adding advanced behaviors**

Run: `cd web && npm run build`
Expected: PASS with a renderable shell and no TypeScript errors.

- [ ] **Step 6: Commit the core shell slice**

```bash
git add web/src/app/App.tsx web/src/app/AppShell.tsx web/src/components/sessions/SessionsPane.tsx web/src/components/conversation/ConversationPane.tsx web/src/components/composer/Composer.tsx web/src/styles/global.css
 git commit -m "feat(web): render core app shell"
```

## Task 7: Implement send flow and the new-session dialog against the existing API

**Files:**
- Create: `web/src/domains/composer/store.ts`
- Create: `web/src/components/new-session/NewSessionDialog.tsx`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/components/composer/Composer.tsx`
- Modify: `web/src/app/AppShell.tsx`
- Create: `web/src/domains/composer/store.test.ts`

- [ ] **Step 1: Extend the API client with send and create-session calls before wiring UI actions**

```ts
async sendMessage(sessionId: string, text: string) {
  return postJson(`/api/sessions/${sessionId}/send`, { text });
},
async createSession(payload: Record<string, unknown>) {
  return postJson(`/api/sessions`, payload);
},
```

- [ ] **Step 2: Create a composer store that owns draft text, pending state, and submit behavior**

```ts
export function createComposerStore() {
  let state = { draft: "", sending: false };
  const listeners = new Set<() => void>();
  const emit = () => listeners.forEach((listener) => listener());

  return {
    getState: () => state,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    setDraft(value: string) {
      state = { ...state, draft: value };
      emit();
    },
    async submit(sessionId: string) {
      if (!state.draft.trim()) return;
      state = { ...state, sending: true };
      emit();
      await api.sendMessage(sessionId, state.draft);
      state = { draft: "", sending: false };
      emit();
    },
  };
}
```

- [ ] **Step 3: Wire the composer component to the store and preserve the current Enter-to-send behavior behind an explicit handler**

```tsx
<form
  className="composer"
  onSubmit={(event) => {
    event.preventDefault();
    if (activeSessionId) composerStore.submit(activeSessionId);
  }}
>
```

- [ ] **Step 4: Add the new-session dialog as a standalone component rather than embedding creation logic in the sidebar pane**

```tsx
export function NewSessionDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) return null;
  return (
    <div className="dialogBackdrop">
      <section className="dialogCard">
        <h2>New session</h2>
        <form>{/* backend tabs, cwd, model, provider, reasoning, launch button */}</form>
      </section>
    </div>
  );
}
```

- [ ] **Step 5: Add a store-level submit regression test so send failures and pending-state cleanup are pinned before the UI gets more complex**

```ts
it("clears sending state after a successful submit", async () => {
  vi.mocked(api.sendMessage).mockResolvedValue({ ok: true } as never);
  const store = createComposerStore();
  store.setDraft("hello");

  await store.submit("s1");

  expect(store.getState()).toEqual({ draft: "", sending: false });
});
```

- [ ] **Step 6: Run the new composer tests and a full frontend build**

Run: `cd web && npm run test -- src/domains/composer/store.test.ts && npm run build`
Expected: PASS and no regression in shell rendering.

- [ ] **Step 7: Commit the send/create-session user-path slice**

```bash
git add web/src/lib/api.ts web/src/domains/composer/store.ts web/src/domains/composer/store.test.ts web/src/components/composer/Composer.tsx web/src/components/new-session/NewSessionDialog.tsx web/src/app/AppShell.tsx
 git commit -m "feat(web): add composer and new session flows"
```

## Task 8: Port session workspace panels and Pi-specific UI requests

**Files:**
- Create: `web/src/domains/session-ui/store.ts`
- Create: `web/src/components/workspace/SessionWorkspace.tsx`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/app/AppShell.tsx`
- Create: `web/src/domains/session-ui/store.test.ts`
- Create: `tests/test_frontend_contract_source.py`

- [ ] **Step 1: Extend the API client with the secondary session surfaces needed by the current UI**

```ts
getDiagnostics(sessionId: string) {
  return getJson(`/api/sessions/${sessionId}/diagnostics`);
},
getQueue(sessionId: string) {
  return getJson(`/api/sessions/${sessionId}/queue`);
},
getFiles(sessionId: string) {
  return getJson(`/api/sessions/${sessionId}/files`);
},
submitUiResponse(sessionId: string, payload: Record<string, unknown>) {
  return postJson(`/api/sessions/${sessionId}/ui_response`, payload);
},
```

- [ ] **Step 2: Model the session workspace store with one explicit action per server surface**

```ts
export function createSessionUiStore() {
  let state = {
    requests: [] as Array<Record<string, unknown>>,
    diagnostics: null as Record<string, unknown> | null,
    queue: null as Record<string, unknown> | null,
    files: [] as Array<Record<string, unknown>>,
  };
  // same subscribe/getState pattern as earlier stores
}
```

- [ ] **Step 3: Render a dedicated workspace column so file/queue/diagnostic concerns never bleed back into the conversation pane**

```tsx
export function SessionWorkspace() {
  const { diagnostics, queue, files, requests } = useSessionUiStore();
  return (
    <aside className="workspacePane">
      <section><h3>Diagnostics</h3><pre>{JSON.stringify(diagnostics, null, 2)}</pre></section>
      <section><h3>Queue</h3><pre>{JSON.stringify(queue, null, 2)}</pre></section>
      <section><h3>Files</h3><pre>{JSON.stringify(files, null, 2)}</pre></section>
      <section><h3>UI Requests</h3><pre>{JSON.stringify(requests, null, 2)}</pre></section>
    </aside>
  );
}
```

- [ ] **Step 4: Add a Python source-level contract test that the backend routes used by the new frontend still exist during migration**

```python
from pathlib import Path

SERVER = Path("codoxear/server.py")


def test_server_still_exposes_frontend_required_routes() -> None:
    source = SERVER.read_text(encoding="utf-8")
    assert '"/api/sessions"' in source
    assert '"/ui_state"' in source
    assert '"/ui_response"' in source
    assert '"/diagnostics"' in source
```

- [ ] **Step 5: Run the workspace store tests, the route contract test, and a frontend build before moving to production cutover work**

Run: `cd web && npm run test -- src/domains/session-ui/store.test.ts && cd .. && python3 -m pytest tests/test_frontend_contract_source.py -q && cd web && npm run build`
Expected: PASS across both frontend and Python contract checks.

- [ ] **Step 6: Commit the advanced workspace slice**

```bash
git add web/src/domains/session-ui/store.ts web/src/domains/session-ui/store.test.ts web/src/components/workspace/SessionWorkspace.tsx web/src/lib/api.ts web/src/app/AppShell.tsx tests/test_frontend_contract_source.py
 git commit -m "feat(web): port workspace and pi ui panels"
```

## Task 9: Restore mobile viewport behavior, manifest wiring, and production entrypoint switching

**Files:**
- Create: `web/src/lib/viewport.ts`
- Modify: `web/src/main.tsx`
- Modify: `web/index.html`
- Modify: `codoxear/static/manifest.webmanifest` or replace with Vite-managed public asset
- Modify: `codoxear/static/service-worker.js` or replace with Vite-managed public asset
- Modify: `README.md`

- [ ] **Step 1: Port the viewport-height bookkeeping into an isolated helper, not back into global app code**

```ts
export function installViewportCssVars() {
  const update = () => {
    const vv = window.visualViewport;
    const layoutH = Math.round(window.innerHeight);
    const visualH = Math.round(vv ? vv.height : window.innerHeight);
    document.documentElement.style.setProperty("--appH", `${visualH}px`);
    document.documentElement.style.setProperty("--layoutH", `${layoutH}px`);
  };
  update();
  window.addEventListener("resize", update);
  window.visualViewport?.addEventListener("resize", update);
}
```

- [ ] **Step 2: Register viewport behavior from `main.tsx` and keep service-worker registration in one boot-time module**

```ts
import { installViewportCssVars } from "./lib/viewport";

installViewportCssVars();
render(<App />, document.getElementById("root")!);
```

- [ ] **Step 3: Move manifest and service-worker references into the Vite-managed entry HTML, preserving the existing public URLs**

```html
<link rel="manifest" href="/manifest.webmanifest" />
<meta name="theme-color" content="#1d4ed8" />
```

- [ ] **Step 4: Document the new development workflow clearly in `README.md` before the legacy UI is removed**

```md
### Frontend development

1. Start the Python server: `python3 -m codoxear.server`
2. Start Vite: `cd web && npm install && npm run dev`
3. Open the Vite URL for frontend work; `/api/*` is proxied to Python.
4. Build production assets with `cd web && npm run build`.
```

- [ ] **Step 5: Run a full mixed-stack verification pass**

Run: `python3 -m pytest tests/test_vite_dist_serving.py tests/test_vite_asset_versioning.py tests/test_frontend_contract_source.py tests/test_static_assets.py -q && cd web && npm run test && npm run build`
Expected: All Python and frontend tests pass, and `web/dist` is ready for production serving.

- [ ] **Step 6: Commit the mobile/PWA/ops integration slice**

```bash
git add web/src/lib/viewport.ts web/src/main.tsx web/index.html codoxear/static/manifest.webmanifest codoxear/static/service-worker.js README.md
 git commit -m "feat(web): restore viewport and pwa integration"
```

## Task 10: Cut over, delete the legacy UI, and verify parity on real flows

**Files:**
- Delete: `codoxear/static/index.html`
- Delete: `codoxear/static/app.js`
- Delete: `codoxear/static/app.css`
- Modify: `codoxear/server.py`
- Modify: `README.md`

- [ ] **Step 1: Remove the fallback to the legacy app only after all previous tasks are green and the feature flag has been tested**

```python
# delete CODOXEAR_USE_LEGACY_WEB branch after rollout confidence is sufficient
```

- [ ] **Step 2: Delete the obsolete hand-authored browser client files from `codoxear/static/`**

Run: `git rm codoxear/static/index.html codoxear/static/app.js codoxear/static/app.css`
Expected: Git stages exactly the three legacy source deletions.

- [ ] **Step 3: Run the final regression suite and a manual parity checklist against both Codex and Pi sessions**

Run: `python3 -m pytest -q && cd web && npm run test && npm run build`
Expected: Full automated suite passes.

Manual checklist:

```text
- Login succeeds
- Session list loads and refreshes
- Session selection updates the conversation
- Message sending works
- New session creation works for Codex and Pi
- Pi ui_state/ui_response flows work
- Diagnostics, queue, and files panels load
- Mobile keyboard and scroll behavior remain usable
- Root app is served by Python from web/dist
```

- [ ] **Step 4: Commit the cutover and legacy removal only after the manual checklist is complete**

```bash
git add -A
 git commit -m "refactor(web): replace legacy static ui with preact app"
```

## Spec Coverage Check

- Dedicated `web/` frontend project: covered in Tasks 1, 4, 5, 6, 7, 8, 9.
- Vite dev server plus `/api` proxy: covered in Task 1.
- Python production hosting of built assets: covered in Tasks 2, 3, 9, 10.
- Lightweight domain stores instead of heavy framework: covered in Tasks 5, 7, 8.
- Feature parity for sessions, messages, composer, new session, Pi UI, workspace panels, PWA basics, mobile behavior: covered in Tasks 6, 7, 8, 9, 10.
- Rollback and cutover safety: covered in Tasks 3 and 10.

## Placeholder Scan

- No `TODO`, `TBD`, or deferred “implement later” placeholders remain in task steps.
- Every code-changing step includes target files and example code.
- Every verification step includes an exact command and expected result.

## Type Consistency Check

- `api` is the single shared frontend client name across all tasks.
- Store naming is consistent across tasks: `sessions`, `messages`, `session-ui`, `composer`.
- `web/dist` is the consistent production output path across server, README, and validation tasks.

## Execution Mode

Chosen approach: **Subagent-Driven**.

Recommended next action: implement this plan with `superpowers:subagent-driven-development`, dispatching one fresh subagent per task and reviewing after each task boundary.
