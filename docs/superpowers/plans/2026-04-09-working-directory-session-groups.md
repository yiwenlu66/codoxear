# Working Directory Session Groups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group the web sidebar sessions by working directory, add server-persisted cwd labels and collapse state, and keep existing session-card behavior intact inside each group.

**Architecture:** Keep `/api/sessions` as a flat session list, add a parallel `cwd_groups` metadata map from the backend, and let the Preact sidebar derive grouped buckets client-side. Persist cwd-level metadata in a new `cwd_groups.json` file inside the app state directory, expose a narrow `POST /api/cwd_groups/edit` endpoint for partial updates, and render a new `SessionGroup` wrapper that owns the group header, path subtitle, inline rename field, and expand/collapse affordance.

**Tech Stack:** Python stdlib HTTP server, Preact, TypeScript, Vitest, pytest/unittest-style backend tests, `web/src/styles/global.css`

---

## File Map

- Modify: `codoxear/server.py`
  - Add cwd-group persistence, normalization helpers, manager accessors/mutators, `/api/sessions` payload extension, and `POST /api/cwd_groups/edit`.
- Modify: `tests/test_session_sidebar_priority.py`
  - Add manager-level tests for cwd-group normalization, persistence semantics, and empty-entry cleanup.
- Modify: `tests/test_pi_server_backend.py`
  - Add route tests for `/api/sessions` returning `cwd_groups` and `/api/cwd_groups/edit` request validation.
- Modify: `web/src/lib/types.ts`
  - Add `CwdGroupMeta`, `EditCwdGroupResponse`, and `cwd_groups` on `SessionsResponse`.
- Modify: `web/src/lib/api.ts`
  - Add `editCwdGroup()`.
- Modify: `web/src/lib/api.test.ts`
  - Lock the new API route shape.
- Modify: `web/src/domains/sessions/store.ts`
  - Persist `cwdGroups` in store state.
- Modify: `web/src/domains/sessions/store.test.ts`
  - Update store expectations to include `cwdGroups` and response propagation.
- Create: `web/src/components/sessions/SessionGroup.tsx`
  - Render cwd group heading, subtitle, rename input, and children.
- Modify: `web/src/components/sessions/SessionsPane.tsx`
  - Group sessions by cwd, sort groups by freshest activity, and wire rename/collapse actions.
- Modify: `web/src/components/sessions/SessionsPane.test.tsx`
  - Cover grouped rendering, fallback group, rename, collapse, and existing session-card actions inside groups.
- Modify: `web/src/styles/global.css`
  - Add group-shell, header, chevron, subtitle, and inline-edit styles without regressing existing card layout.

## Task 0: Create an isolated worktree and baseline

**Files:**
- No code changes yet

- [ ] **Step 1: Create a dedicated worktree from the current HEAD**

Run: `git worktree add ../codoxear-cwd-groups -b feat/web-cwd-session-groups HEAD`
Expected: a new sibling checkout at `../codoxear-cwd-groups` on a fresh branch named `feat/web-cwd-session-groups`.

- [ ] **Step 2: Install frontend dependencies in the worktree**

Run: `cd ../codoxear-cwd-groups/web && npm install`
Expected: `up to date` or a normal install summary with no dependency resolution errors.

- [ ] **Step 3: Capture backend and frontend baselines before edits**

Run: `cd ../codoxear-cwd-groups && python3 -m pytest tests/test_session_sidebar_priority.py tests/test_pi_server_backend.py -q && cd web && npx vitest run src/lib/api.test.ts src/domains/sessions/store.test.ts src/components/sessions/SessionsPane.test.tsx`
Expected: PASS on the current baseline so the later failures come only from the new feature tests.

## Task 1: Add backend cwd-group persistence behind manager tests

**Files:**
- Modify: `tests/test_session_sidebar_priority.py`
- Modify: `codoxear/server.py`
- Test: `tests/test_session_sidebar_priority.py`

- [ ] **Step 1: Write the failing manager tests for cwd-group normalization and empty-entry cleanup**

```python
# tests/test_session_sidebar_priority.py
class TestSessionSidebarPriority(unittest.TestCase):
    def test_cwd_group_set_normalizes_path_and_round_trips_metadata(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            raw = str(Path(td) / "project" / ".." / "project")
            cwd, meta = SessionManager.cwd_group_set(mgr, cwd=raw, label="Project Atlas", collapsed=True)

        self.assertTrue(Path(cwd).is_absolute())
        self.assertEqual(meta, {"label": "Project Atlas", "collapsed": True})
        self.assertEqual(SessionManager.cwd_groups_get(mgr)[cwd], meta)

    def test_cwd_group_set_drops_empty_default_entries(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            cwd = str(Path(td).resolve())
            SessionManager.cwd_group_set(mgr, cwd=cwd, label="Atlas", collapsed=True)
            normalized, meta = SessionManager.cwd_group_set(mgr, cwd=cwd, label="", collapsed=False)

        self.assertEqual(meta, {"label": "", "collapsed": False})
        self.assertNotIn(normalized, SessionManager.cwd_groups_get(mgr))

    def test_cwd_group_set_rejects_non_boolean_collapsed(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            cwd = str(Path(td).resolve())
            with self.assertRaisesRegex(ValueError, "collapsed must be a boolean"):
                SessionManager.cwd_group_set(mgr, cwd=cwd, label="Atlas", collapsed="yes")
```

- [ ] **Step 2: Run the manager tests to verify they fail**

Run: `cd ../codoxear-cwd-groups && python3 -m pytest tests/test_session_sidebar_priority.py -q`
Expected: FAIL because `SessionManager` has no cwd-group store or methods yet.

- [ ] **Step 3: Implement cwd-group persistence and manager helpers**

```python
# codoxear/server.py
CWD_GROUPS_PATH = APP_DIR / "cwd_groups.json"


def _normalize_cwd_group_key(raw: Any) -> str:
    if not isinstance(raw, str):
        raise ValueError("cwd required")
    text = raw.strip()
    if not text:
        raise ValueError("cwd required")
    return str(Path(text).expanduser().resolve(strict=False))


class SessionManager:
    def __init__(self) -> None:
        self._cwd_groups: dict[str, dict[str, Any]] = {}
        self._load_cwd_groups()
        # existing loads continue here

    def _load_cwd_groups(self) -> None:
        try:
            raw = CWD_GROUPS_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("invalid cwd_groups.json (expected object)")
        cleaned: dict[str, dict[str, Any]] = {}
        for raw_cwd, value in obj.items():
            if not isinstance(value, dict):
                continue
            try:
                cwd = _normalize_cwd_group_key(raw_cwd)
            except ValueError:
                continue
            label = _clean_alias(value.get("label") if isinstance(value.get("label"), str) else "")
            collapsed_raw = value.get("collapsed")
            collapsed = collapsed_raw if type(collapsed_raw) is bool else False
            if label or collapsed:
                cleaned[cwd] = {"label": label, "collapsed": collapsed}
        with self._lock:
            self._cwd_groups = cleaned

    def _save_cwd_groups(self) -> None:
        with self._lock:
            obj = dict(self._cwd_groups)
        os.makedirs(APP_DIR, exist_ok=True)
        tmp = CWD_GROUPS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, CWD_GROUPS_PATH)

    def cwd_groups_get(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {cwd: {"label": str(v.get("label") or ""), "collapsed": bool(v.get("collapsed"))} for cwd, v in self._cwd_groups.items()}

    def cwd_group_set(self, *, cwd: Any, label: Any = None, collapsed: Any = None) -> tuple[str, dict[str, Any]]:
        key = _normalize_cwd_group_key(cwd)
        if collapsed is not None and type(collapsed) is not bool:
            raise ValueError("collapsed must be a boolean")
        label_clean = _clean_alias(label) if isinstance(label, str) else None
        with self._lock:
            current = dict(self._cwd_groups.get(key) or {})
            next_label = label_clean if label is not None else str(current.get("label") or "")
            next_collapsed = bool(collapsed) if collapsed is not None else bool(current.get("collapsed"))
            meta = {"label": next_label, "collapsed": next_collapsed}
            if next_label or next_collapsed:
                self._cwd_groups[key] = meta
            else:
                self._cwd_groups.pop(key, None)
        self._save_cwd_groups()
        return key, meta
```

- [ ] **Step 4: Update the shared manager test harness to include the new store**

```python
# tests/test_session_sidebar_priority.py

def _make_manager() -> SessionManager:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    mgr._sessions = {}
    mgr._harness = {}
    mgr._aliases = {}
    mgr._sidebar_meta = {}
    mgr._cwd_groups = {}
    mgr._hidden_sessions = set()
    mgr._files = {}
    mgr._queues = {}
    mgr._recent_cwds = {}
    mgr._save_cwd_groups = lambda *args, **kwargs: None  # type: ignore[method-assign]
    # existing stubs continue here
    return mgr
```

- [ ] **Step 5: Run the manager tests to verify they pass**

Run: `cd ../codoxear-cwd-groups && python3 -m pytest tests/test_session_sidebar_priority.py -q`
Expected: PASS with the new cwd-group tests green and the existing sidebar-priority tests still green.

- [ ] **Step 6: Commit the backend persistence layer**

```bash
git add codoxear/server.py tests/test_session_sidebar_priority.py
git commit -m "feat(server): persist working directory group metadata"
```

## Task 2: Expose cwd-group metadata through HTTP and lock the contract

**Files:**
- Modify: `tests/test_pi_server_backend.py`
- Modify: `codoxear/server.py`
- Modify: `web/src/lib/types.ts`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/lib/api.test.ts`
- Modify: `web/src/domains/sessions/store.ts`
- Modify: `web/src/domains/sessions/store.test.ts`
- Test: `tests/test_pi_server_backend.py`, `web/src/lib/api.test.ts`, `web/src/domains/sessions/store.test.ts`

- [ ] **Step 1: Write the failing backend route tests for `/api/sessions` and `/api/cwd_groups/edit`**

```python
# tests/test_pi_server_backend.py
class TestPiBackendRouting(unittest.TestCase):
    def test_sessions_route_includes_cwd_groups(self) -> None:
        handler = _HandlerHarness("/api/sessions")
        mgr = _make_manager()
        mgr.list_sessions = lambda: [{"session_id": "sess-1", "cwd": "/tmp/project"}]  # type: ignore[method-assign]
        mgr.recent_cwds = lambda limit=None: ["/tmp/project"]  # type: ignore[method-assign]
        mgr.cwd_groups_get = lambda: {"/tmp/project": {"label": "Project Atlas", "collapsed": True}}  # type: ignore[method-assign]

        with patch("codoxear.server._require_auth", return_value=True), patch("codoxear.server.MANAGER", mgr), patch(
            "codoxear.server._read_new_session_defaults", return_value={"default_backend": "codex"}
        ):
            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload["cwd_groups"], {"/tmp/project": {"label": "Project Atlas", "collapsed": True}})

    def test_cwd_groups_edit_route_updates_label_and_collapsed(self) -> None:
        body = json.dumps({"cwd": "/tmp/project", "label": "Project Atlas", "collapsed": True}).encode("utf-8")
        handler = _HandlerHarness("/api/cwd_groups/edit", body=body)
        mgr = _make_manager()
        mgr.cwd_group_set = lambda **kwargs: ("/tmp/project", {"label": "Project Atlas", "collapsed": True})  # type: ignore[method-assign]

        with patch("codoxear.server._require_auth", return_value=True), patch("codoxear.server.MANAGER", mgr):
            Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload, {"ok": True, "cwd": "/tmp/project", "label": "Project Atlas", "collapsed": True})
```

- [ ] **Step 2: Write the failing frontend contract tests for the new payload**

```ts
// web/src/lib/api.test.ts
it("posts cwd group edit payloads", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    text: async () => '{"ok":true,"cwd":"/tmp/project","label":"Atlas","collapsed":true}',
  });
  vi.stubGlobal("fetch", fetchMock);

  await expect(api.editCwdGroup({ cwd: "/tmp/project", label: "Atlas", collapsed: true })).resolves.toEqual({
    ok: true,
    cwd: "/tmp/project",
    label: "Atlas",
    collapsed: true,
  });

  expect(fetchMock).toHaveBeenCalledWith("api/cwd_groups/edit", expect.objectContaining({
    method: "POST",
    body: JSON.stringify({ cwd: "/tmp/project", label: "Atlas", collapsed: true }),
  }));
});

// web/src/domains/sessions/store.test.ts
it("stores cwd groups returned from listSessions", async () => {
  vi.mocked(api.listSessions).mockResolvedValue({
    sessions: [{ session_id: "s1", cwd: "/tmp/project" }],
    cwd_groups: { "/tmp/project": { label: "Atlas", collapsed: true } },
  } as never);
  const store = createSessionsStore();

  await store.refresh();

  expect(store.getState().cwdGroups).toEqual({ "/tmp/project": { label: "Atlas", collapsed: true } });
});
```

- [ ] **Step 3: Run the route and contract tests to verify they fail**

Run: `cd ../codoxear-cwd-groups && python3 -m pytest tests/test_pi_server_backend.py -q && cd web && npx vitest run src/lib/api.test.ts src/domains/sessions/store.test.ts`
Expected: FAIL because the backend does not yet emit `cwd_groups`, the edit route does not exist, and the store does not track cwd-group metadata.

- [ ] **Step 4: Extend the backend routes and frontend contracts**

```python
# codoxear/server.py
if path == "/api/sessions":
    _json_response(
        self,
        200,
        {
            "app_version": _current_app_version(),
            "sessions": sessions,
            "cwd_groups": MANAGER.cwd_groups_get(),
            "recent_cwds": recent_cwds,
            "new_session_defaults": new_session_defaults,
            "tmux_available": _tmux_available(),
            "tmux_session_name": TMUX_SESSION_NAME,
        },
    )
    return

if path == "/api/cwd_groups/edit":
    if not _require_auth(self):
        self._unauthorized()
        return
    body = _read_body(self)
    obj = json.loads(body.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("invalid json body (expected object)")
    try:
        cwd, meta = MANAGER.cwd_group_set(cwd=obj.get("cwd"), label=obj.get("label"), collapsed=obj.get("collapsed"))
    except ValueError as e:
        _json_response(self, 400, {"error": str(e)})
        return
    _json_response(self, 200, {"ok": True, "cwd": cwd, **meta})
    return
```

```ts
// web/src/lib/types.ts
export interface CwdGroupMeta {
  label?: string;
  collapsed?: boolean;
}

export interface SessionsResponse {
  app_version?: string;
  sessions: SessionSummary[];
  cwd_groups?: Record<string, CwdGroupMeta>;
  recent_cwds?: string[];
  new_session_defaults?: NewSessionDefaults;
  tmux_available?: boolean;
  tmux_session_name?: string;
}

export interface EditCwdGroupResponse {
  ok?: boolean;
  cwd?: string;
  label?: string;
  collapsed?: boolean;
}

// web/src/lib/api.ts
editCwdGroup(payload: Record<string, unknown>) {
  return postJson<EditCwdGroupResponse>(`/api/cwd_groups/edit`, payload);
}

// web/src/domains/sessions/store.ts
export interface SessionsState {
  items: SessionSummary[];
  activeSessionId: string | null;
  loading: boolean;
  newSessionDefaults: NewSessionDefaults | null;
  recentCwds: string[];
  cwdGroups: Record<string, { label?: string; collapsed?: boolean }>;
  tmuxAvailable: boolean;
}

state = {
  items: data.sessions,
  activeSessionId: nextActiveSessionId,
  loading: false,
  newSessionDefaults: data.new_session_defaults ?? state.newSessionDefaults,
  recentCwds: Array.isArray(data.recent_cwds) ? data.recent_cwds.filter((cwd): cwd is string => typeof cwd === "string") : state.recentCwds,
  cwdGroups: data.cwd_groups && typeof data.cwd_groups === "object" ? data.cwd_groups : state.cwdGroups,
  tmuxAvailable: typeof data.tmux_available === "boolean" ? data.tmux_available : state.tmuxAvailable,
};
```

- [ ] **Step 5: Run the route and contract tests to verify they pass**

Run: `cd ../codoxear-cwd-groups && python3 -m pytest tests/test_pi_server_backend.py -q && cd web && npx vitest run src/lib/api.test.ts src/domains/sessions/store.test.ts`
Expected: PASS with the new backend route tests, API helper test, and store-state propagation tests all green.

- [ ] **Step 6: Commit the HTTP/store contract update**

```bash
git add codoxear/server.py tests/test_pi_server_backend.py web/src/lib/types.ts web/src/lib/api.ts web/src/lib/api.test.ts web/src/domains/sessions/store.ts web/src/domains/sessions/store.test.ts
git commit -m "feat(web): expose working directory group metadata"
```

## Task 3: Render grouped session buckets in the sidebar

**Files:**
- Modify: `web/src/components/sessions/SessionsPane.test.tsx`
- Create: `web/src/components/sessions/SessionGroup.tsx`
- Modify: `web/src/components/sessions/SessionsPane.tsx`
- Modify: `web/src/styles/global.css`
- Test: `web/src/components/sessions/SessionsPane.test.tsx`

- [ ] **Step 1: Write the failing integration tests for grouped rendering and the fallback group**

```tsx
// web/src/components/sessions/SessionsPane.test.tsx
it("groups sessions by working directory and orders groups by freshest activity", () => {
  const sessionsStore = createStaticStore({
    items: [
      { session_id: "sess-1", alias: "Backend bug", cwd: "/repo/api", updated_ts: 10, agent_backend: "pi" },
      { session_id: "sess-2", alias: "Sidebar polish", cwd: "/repo/web", updated_ts: 30, agent_backend: "pi" },
      { session_id: "sess-3", alias: "Queue audit", cwd: "/repo/api", updated_ts: 20, agent_backend: "pi" },
    ],
    activeSessionId: null,
    loading: false,
    cwdGroups: { "/repo/api": { label: "API", collapsed: false } },
    newSessionDefaults: null,
    recentCwds: [],
    tmuxAvailable: false,
  });

  root = document.createElement("div");
  document.body.appendChild(root);
  render(
    <AppProviders sessionsStore={sessionsStore as any}>
      <SessionsPane />
    </AppProviders>,
    root,
  );

  const headings = Array.from(root.querySelectorAll("[data-testid='session-group-title']")).map((node) => node.textContent?.trim());
  expect(headings).toEqual(["web", "API"]);
  expect(root.textContent).toContain("Sidebar polish");
  expect(root.textContent).toContain("Backend bug");
  expect(root.textContent).toContain("Queue audit");
});

it("renders a fallback group for sessions without a working directory", () => {
  const sessionsStore = createStaticStore({
    items: [{ session_id: "sess-1", alias: "Detached", agent_backend: "codex" }],
    activeSessionId: null,
    loading: false,
    cwdGroups: {},
    newSessionDefaults: null,
    recentCwds: [],
    tmuxAvailable: false,
  });

  root = document.createElement("div");
  document.body.appendChild(root);
  render(
    <AppProviders sessionsStore={sessionsStore as any}>
      <SessionsPane />
    </AppProviders>,
    root,
  );

  expect(root.textContent).toContain("No working directory");
  expect(root.querySelectorAll("[data-testid='session-card']")).toHaveLength(1);
});
```

- [ ] **Step 2: Run the sidebar test to verify it fails**

Run: `cd ../codoxear-cwd-groups/web && npx vitest run src/components/sessions/SessionsPane.test.tsx`
Expected: FAIL because `SessionsPane` still renders one flat `sessionsList` with no group wrappers or fallback heading.

- [ ] **Step 3: Implement grouped rendering with a dedicated `SessionGroup` component**

```tsx
// web/src/components/sessions/SessionGroup.tsx
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface SessionGroupProps {
  title: string;
  subtitle?: string;
  expanded: boolean;
  fallback?: boolean;
  editing: boolean;
  renameValue: string;
  saving?: boolean;
  error?: string;
  onToggle: () => void;
  onStartRename?: () => void;
  onRenameValueChange?: (value: string) => void;
  onRenameCommit?: () => void;
  onRenameCancel?: () => void;
  children: preact.ComponentChildren;
}

export function SessionGroup(props: SessionGroupProps) {
  return (
    <section className="sessionGroup" data-testid="session-group">
      <div className="sessionGroupHeader">
        <button type="button" className="sessionGroupToggle" onClick={props.onToggle} aria-expanded={props.expanded}>
          <span className={cn("sessionGroupChevron", props.expanded && "isExpanded")} aria-hidden="true">▸</span>
          <span className="sessionGroupHeading">
            {props.editing ? (
              <Input
                autoFocus
                value={props.renameValue}
                onInput={(event) => props.onRenameValueChange?.((event.currentTarget as HTMLInputElement).value)}
                onBlur={() => props.onRenameCommit?.()}
                onKeyDown={(event) => {
                  if (event.key === "Enter") props.onRenameCommit?.();
                  if (event.key === "Escape") props.onRenameCancel?.();
                }}
              />
            ) : (
              <>
                <span data-testid="session-group-title" className="sessionGroupTitle">{props.title}</span>
                {props.subtitle ? <span className="sessionGroupSubtitle">{props.subtitle}</span> : null}
              </>
            )}
          </span>
        </button>
        {!props.fallback && !props.editing ? (
          <Button type="button" variant="ghost" size="sm" onClick={props.onStartRename} disabled={props.saving}>Rename</Button>
        ) : null}
      </div>
      {props.error ? <p className="sessionGroupError">{props.error}</p> : null}
      {props.expanded ? <div className="sessionGroupList">{props.children}</div> : null}
    </section>
  );
}
```

```tsx
// web/src/components/sessions/SessionsPane.tsx
const FALLBACK_CWD_GROUP_KEY = "__missing_cwd__";

function basenameFromCwd(cwd: string) {
  const parts = cwd.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || cwd;
}

const groupedItems = useMemo(() => {
  const buckets = new Map<string, { key: string; cwd: string | null; sessions: SessionSummary[]; newestTs: number; label?: string; collapsed?: boolean }>();
  for (const session of items) {
    const cwd = String(session.cwd || "").trim();
    const key = cwd || FALLBACK_CWD_GROUP_KEY;
    const bucket = buckets.get(key) ?? {
      key,
      cwd: cwd || null,
      sessions: [],
      newestTs: Number(session.updated_ts || session.start_ts || 0),
      label: cwd ? cwdGroups[cwd]?.label : undefined,
      collapsed: cwd ? Boolean(cwdGroups[cwd]?.collapsed) : false,
    };
    bucket.sessions.push(session);
    bucket.newestTs = Math.max(bucket.newestTs, Number(session.updated_ts || session.start_ts || 0));
    buckets.set(key, bucket);
  }
  return Array.from(buckets.values()).sort((a, b) => b.newestTs - a.newestTs || a.key.localeCompare(b.key));
}, [items, cwdGroups]);
```

- [ ] **Step 4: Add the new group-shell styles**

```css
/* web/src/styles/global.css */
.sessionGroup {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
  padding: 0.35rem 0;
}

.sessionGroupHeader {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.65rem;
}

.sessionGroupToggle {
  width: 100%;
  display: flex;
  align-items: flex-start;
  gap: 0.65rem;
  padding: 0.25rem 0;
  border: 0;
  background: transparent;
  text-align: left;
}

.sessionGroupChevron {
  margin-top: 0.1rem;
  transition: transform 160ms ease;
}

.sessionGroupChevron.isExpanded {
  transform: rotate(90deg);
}

.sessionGroupHeading {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}

.sessionGroupTitle {
  font-size: 0.92rem;
  font-weight: 650;
}

.sessionGroupSubtitle {
  color: var(--legacy-muted);
  font-size: 0.76rem;
  overflow-wrap: anywhere;
}

.sessionGroupList {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding-left: 1.2rem;
}
```

- [ ] **Step 5: Run the sidebar integration test to verify it passes**

Run: `cd ../codoxear-cwd-groups/web && npx vitest run src/components/sessions/SessionsPane.test.tsx`
Expected: PASS with the new cwd-group ordering and fallback-group tests green alongside the existing session-card tests.

- [ ] **Step 6: Commit the grouped sidebar rendering**

```bash
git add web/src/components/sessions/SessionGroup.tsx web/src/components/sessions/SessionsPane.tsx web/src/components/sessions/SessionsPane.test.tsx web/src/styles/global.css
git commit -m "feat(web): group sidebar sessions by working directory"
```

## Task 4: Add rename and collapse interactions with persistence

**Files:**
- Modify: `web/src/components/sessions/SessionsPane.test.tsx`
- Modify: `web/src/components/sessions/SessionsPane.tsx`
- Modify: `web/src/components/sessions/SessionGroup.tsx`
- Modify: `web/src/styles/global.css`
- Test: `web/src/components/sessions/SessionsPane.test.tsx`

- [ ] **Step 1: Write the failing interaction tests for rename and collapse persistence calls**

```tsx
// web/src/components/sessions/SessionsPane.test.tsx
it("renames a working directory group and refreshes sessions", async () => {
  const { api } = await import("../../lib/api");
  const refresh = vi.fn().mockResolvedValue(undefined);
  const sessionsStore = createStaticStore({
    items: [{ session_id: "sess-1", alias: "Backend bug", cwd: "/repo/api", updated_ts: 10, agent_backend: "pi" }],
    activeSessionId: null,
    loading: false,
    cwdGroups: {},
    newSessionDefaults: null,
    recentCwds: [],
    tmuxAvailable: false,
  });
  sessionsStore.refresh = refresh;

  root = document.createElement("div");
  document.body.appendChild(root);
  render(
    <AppProviders sessionsStore={sessionsStore as any}>
      <SessionsPane />
    </AppProviders>,
    root,
  );

  const renameButton = Array.from(root.querySelectorAll("button")).find((button) => button.textContent?.includes("Rename"));
  renameButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  await flush();

  const input = root.querySelector(".sessionGroup input") as HTMLInputElement;
  await act(async () => {
    input.value = "Platform API";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  });
  await flush();

  expect(api.editCwdGroup).toHaveBeenCalledWith({ cwd: "/repo/api", label: "Platform API" });
  expect(refresh).toHaveBeenCalled();
});

it("collapses a working directory group and hides its cards", async () => {
  const { api } = await import("../../lib/api");
  const sessionsStore = createStaticStore({
    items: [{ session_id: "sess-1", alias: "Backend bug", cwd: "/repo/api", updated_ts: 10, agent_backend: "pi" }],
    activeSessionId: null,
    loading: false,
    cwdGroups: { "/repo/api": { label: "API", collapsed: false } },
    newSessionDefaults: null,
    recentCwds: [],
    tmuxAvailable: false,
  });

  root = document.createElement("div");
  document.body.appendChild(root);
  render(
    <AppProviders sessionsStore={sessionsStore as any}>
      <SessionsPane />
    </AppProviders>,
    root,
  );

  const toggle = root.querySelector(".sessionGroupToggle") as HTMLButtonElement;
  toggle.click();
  await flush();

  expect(api.editCwdGroup).toHaveBeenCalledWith({ cwd: "/repo/api", collapsed: true });
  expect(root.querySelectorAll("[data-testid='session-card']")).toHaveLength(0);
});
```

- [ ] **Step 2: Run the sidebar test to verify the interaction cases fail**

Run: `cd ../codoxear-cwd-groups/web && npx vitest run src/components/sessions/SessionsPane.test.tsx`
Expected: FAIL because the group header does not yet open rename mode, post rename updates, or persist collapse state through the API.

- [ ] **Step 3: Implement rename/collapse behavior in `SessionsPane`**

```tsx
// web/src/components/sessions/SessionsPane.tsx
const [editingCwd, setEditingCwd] = useState<string | null>(null);
const [renameValue, setRenameValue] = useState("");
const [pendingGroups, setPendingGroups] = useState<Record<string, { label?: string; collapsed?: boolean }>>({});

async function saveCwdGroup(cwd: string, patch: { label?: string; collapsed?: boolean }) {
  setActionError("");
  setPendingGroups((current) => ({ ...current, [cwd]: { ...current[cwd], ...patch } }));
  try {
    await api.editCwdGroup({ cwd, ...patch });
    await sessionsStoreApi.refresh();
  } catch (error) {
    setActionError(error instanceof Error ? error.message : "Failed to update working directory");
  } finally {
    setPendingGroups((current) => {
      const next = { ...current };
      delete next[cwd];
      return next;
    });
  }
}

async function commitRename(cwd: string) {
  const nextLabel = renameValue.trim();
  setEditingCwd(null);
  await saveCwdGroup(cwd, { label: nextLabel });
}
```

```tsx
// inside the SessionsPane render
{groupedItems.map((group) => {
  const cwd = group.cwd;
  const pending = cwd ? pendingGroups[cwd] : undefined;
  const collapsed = cwd ? (pending?.collapsed ?? group.collapsed ?? false) : false;
  const title = group.cwd ? ((pending?.label ?? group.label)?.trim() || basenameFromCwd(group.cwd)) : "No working directory";
  return (
    <SessionGroup
      key={group.key}
      title={title}
      subtitle={group.cwd || undefined}
      expanded={!collapsed}
      fallback={!group.cwd}
      editing={editingCwd === cwd}
      renameValue={editingCwd === cwd ? renameValue : title}
      onToggle={() => {
        if (!cwd) {
          return;
        }
        void saveCwdGroup(cwd, { collapsed: !collapsed });
      }}
      onStartRename={cwd ? () => {
        setRenameValue((group.label || "").trim());
        setEditingCwd(cwd);
      } : undefined}
      onRenameValueChange={setRenameValue}
      onRenameCommit={cwd ? () => { void commitRename(cwd); } : undefined}
      onRenameCancel={() => {
        setEditingCwd(null);
        setRenameValue("");
      }}
    >
      {group.sessions.map((session) => (
        <SessionCard key={session.session_id} session={session} active={session.session_id === activeSessionId} onSelect={() => sessionsStoreApi.select(session.session_id)} onEdit={() => { setActionError(""); setEditingSessionId(session.session_id); }} onDuplicate={() => { void duplicateSession(session); }} onDelete={() => { void deleteSession(session); }} />
      ))}
    </SessionGroup>
  );
})}
```

- [ ] **Step 4: Polish the inline-edit and collapsed visuals**

```css
/* web/src/styles/global.css */
.sessionGroupHeader .sessionGroupToggle:hover .sessionGroupTitle {
  color: var(--legacy-accent);
}

.sessionGroup input {
  max-width: 18rem;
}

.sessionGroupError {
  margin: 0 0 0 1.85rem;
  color: #b91c1c;
  font-size: 0.78rem;
}
```

- [ ] **Step 5: Run the sidebar interaction tests to verify they pass**

Run: `cd ../codoxear-cwd-groups/web && npx vitest run src/components/sessions/SessionsPane.test.tsx`
Expected: PASS with rename, collapse, and the existing duplicate/delete/edit session-card behaviors all green inside the grouped layout.

- [ ] **Step 6: Commit the interactive cwd-group controls**

```bash
git add web/src/components/sessions/SessionsPane.tsx web/src/components/sessions/SessionsPane.test.tsx web/src/components/sessions/SessionGroup.tsx web/src/styles/global.css
git commit -m "feat(web): add working directory group controls"
```

## Task 5: Run focused verification and capture final behavior

**Files:**
- No new code files; verification only

- [ ] **Step 1: Run the focused backend and frontend test suites together**

Run: `cd ../codoxear-cwd-groups && python3 -m pytest tests/test_session_sidebar_priority.py tests/test_pi_server_backend.py -q && cd web && npx vitest run src/lib/api.test.ts src/domains/sessions/store.test.ts src/components/sessions/SessionsPane.test.tsx`
Expected: PASS across all new and touched tests.

- [ ] **Step 2: Run one broader frontend smoke suite around the sidebar shell**

Run: `cd ../codoxear-cwd-groups/web && npx vitest run src/app/AppShell.test.tsx src/components/workspace/SessionWorkspace.test.tsx`
Expected: PASS, confirming the grouped sidebar still composes correctly with the rest of the shell.

- [ ] **Step 3: Manually sanity-check the browser UI if a dev server is already available**

Run: `cd ../codoxear-cwd-groups/web && npm run dev`
Expected: a local Vite URL opens in the terminal; verify that cwd groups render, rename, collapse, and session actions behave correctly on desktop and mobile-width responsive layouts.

- [ ] **Step 4: Record the final implementation state**

```bash
git status --short
git log --oneline -5
```
Expected: only the planned implementation files remain changed, and the recent history shows the cwd-group feature commits in order.
