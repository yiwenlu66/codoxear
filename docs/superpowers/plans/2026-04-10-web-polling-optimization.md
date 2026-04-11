# Web Polling Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the live/workspace polling split, then split `GET /api/sessions` into a lightweight polled session-list contract plus event-driven bootstrap metadata so the sidebar stops polling large, mostly static payloads.

**Architecture:** Treat the current dirty branch as the starting point: the backend already exposes `live` and `workspace`, and the frontend already has an in-progress live-session store. The remaining work is to stabilize that split with tests, project `GET /api/sessions` down to only the fields the sidebar actually consumes, move `recent_cwds` / `cwd_groups` / `new_session_defaults` / `tmux_available` off the polling path into a separate bootstrap route, and update the sessions store so only the lightweight session rows are polled.

**Tech Stack:** Python stdlib HTTP server, existing backend normalization helpers, Preact, TypeScript, Vitest, pytest.

---

## Context

Approved spec: `docs/superpowers/specs/2026-04-10-web-polling-optimization-design.md`

Additional user direction gathered after the spec:
- continue splitting `GET /api/sessions`
- reduce response size because the current route returns a large payload
- remove invalid polling of mostly static session-shell data

Confirmed from the current codebase and dirty branch state:
- `codoxear/server.py` already has `_session_live_payload()`, `_session_workspace_payload()`, and `GET /api/sessions/:id/live|workspace` routes.
- `web/src/domains/live-session/store.ts` already exists and updates `messagesStore.applyLive()` from `api.getLiveSession()`.
- `web/src/domains/session-ui/store.ts` already moved to `api.getWorkspace()` only.
- `web/src/app/app-shell/useAppShellSessionEffects.ts` already moved active/background polling to the live store and gates workspace polling on `workspaceOpen`.
- `GET /api/sessions` still returns both frequently changing `sessions` rows and mostly static bootstrap data (`recent_cwds`, `cwd_groups`, `new_session_defaults`, `tmux_available`, unused `app_version`, `tmux_session_name`) on every poll.
- `SessionManager.list_sessions()` still builds large row objects containing many fields the web UI does not consume directly, including `files`, `log_path`, `token`, `thinking`, `tools`, `system`, harness metadata, provider internals, tmux internals, and priority score breakdowns.
- Frontend consumers of the polled session rows mainly need: `session_id`, `thread_id`, `resume_session_id`, `title`, `alias`, `first_user_message`, `cwd`, `agent_backend`, `broker_pid`, `owned`, `busy`, `queue_len`, `git_branch`, `model`, `provider_choice`, `reasoning_effort`, `service_tier`, `transport`, `priority_offset`, `snooze_until`, `dependency_session_id`, `blocked`, `snoozed`, and `historical`.
- `web/src/domains/sessions/store.ts` still polls `api.listSessions()` for both session rows and bootstrap metadata on every refresh.
- `web/src/components/new-session/NewSessionDialog.tsx` is the main consumer of `newSessionDefaults`, `recentCwds`, and `tmuxAvailable`.
- `web/src/components/sessions/SessionsPane.tsx` is the main consumer of `cwdGroups`.

## Recommended Route Split

- Keep `GET /api/sessions` as the lightweight, frequently polled sidebar route.
- Add `GET /api/sessions/bootstrap` for low-frequency shell metadata:
  - `recent_cwds`
  - `cwd_groups`
  - `new_session_defaults`
  - `tmux_available`
  - optionally `app_version` if future UI needs it, otherwise omit it from the frontend contract
- Keep `POST /api/sessions` unchanged for create-session flows.
- Do not change the approved `live`, `workspace`, or hierarchical `file/list` directions.

## Existing Code To Reuse

### Backend
- `codoxear/server.py:3207` `_session_diagnostics_payload()`
- `codoxear/server.py:3280` `_session_workspace_payload()`
- `codoxear/server.py:3294` `_session_live_payload()`
- `codoxear/server.py:5122` `SessionManager.list_sessions()` as the source of internal session ordering and metadata, while projecting a smaller frontend row at the HTTP layer instead of rewriting internal manager semantics all at once
- `codoxear/server.py:7252` current `GET /api/sessions` route

### Frontend
- `web/src/domains/live-session/store.ts`
- `web/src/domains/session-ui/store.ts`
- `web/src/app/app-shell/useAppShellSessionEffects.ts`
- `web/src/domains/sessions/store.ts`
- `web/src/components/new-session/NewSessionDialog.tsx`
- `web/src/components/sessions/SessionsPane.tsx`

## Files To Change

### Backend
- `codoxear/server.py`
- `tests/test_pi_server_backend.py`
- `tests/test_frontend_contract_source.py`

### Frontend API + Types
- `web/src/lib/types.ts`
- `web/src/lib/api.ts`
- `web/src/lib/api.test.ts`

### Frontend Stores / Providers
- `web/src/domains/messages/store.ts`
- `web/src/domains/messages/store.test.ts`
- `web/src/domains/live-session/store.ts`
- `web/src/domains/live-session/store.test.ts`
- `web/src/domains/session-ui/store.ts`
- `web/src/domains/session-ui/store.test.ts`
- `web/src/domains/sessions/store.ts`
- `web/src/domains/sessions/store.test.ts`
- `web/src/app/providers.tsx`

### Frontend UI / Scheduling
- `web/src/app/app-shell/useAppShellSessionEffects.ts`
- `web/src/app/app-shell/useAppShellSessionEffects.test.tsx`
- `web/src/app/app-shell/useAppShellNotifications.ts`
- `web/src/app/AppShell.tsx`
- `web/src/app/AppShell.test.tsx`
- `web/src/components/conversation/AskUserCard.tsx`
- `web/src/components/conversation/AskUserCard.test.tsx`
- `web/src/components/conversation/ConversationPane.tsx`
- `web/src/components/conversation/ConversationPane.test.tsx`
- `web/src/components/workspace/SessionWorkspace.tsx`
- `web/src/components/workspace/SessionWorkspace.test.tsx`
- `web/src/components/new-session/NewSessionDialog.tsx`
- `web/src/components/new-session/NewSessionDialog.test.tsx`
- `web/src/components/sessions/SessionsPane.tsx`
- `web/src/components/sessions/SessionsPane.test.tsx`
- `web/src/app/app-shell/AppShellWorkspaceOverlays.tsx` only if stale `files` wiring still remains in this branch

## Implementation Tasks

### Task 1: Stabilize the current live/workspace split that is already in the branch

**Files:**
- `codoxear/server.py`
- `tests/test_pi_server_backend.py`
- `tests/test_frontend_contract_source.py`
- `web/src/lib/types.ts`
- `web/src/lib/api.ts`
- `web/src/lib/api.test.ts`
- `web/src/domains/messages/store.ts`
- `web/src/domains/messages/store.test.ts`
- `web/src/domains/live-session/store.ts`
- `web/src/domains/live-session/store.test.ts`
- `web/src/domains/session-ui/store.ts`
- `web/src/domains/session-ui/store.test.ts`
- `web/src/app/providers.tsx`
- `web/src/components/conversation/AskUserCard.tsx`
- `web/src/components/workspace/SessionWorkspace.tsx`
- `web/src/app/app-shell/useAppShellSessionEffects.ts`
- related frontend tests already added on the branch

- [ ] Run the targeted live/workspace backend tests first to see what the dirty branch still lacks.
- [ ] Run the targeted frontend API/store/scheduler tests next to identify missing glue between the new live store, workspace store, and UI consumers.
- [ ] Fix the live/workspace path only where tests prove the current branch is incomplete; do not re-expand scope.
- [ ] Keep `/messages`, `/ui_state`, `/diagnostics`, and `/queue` available for compatibility while the web UI prefers `live` and `workspace`.
- [ ] Save the working route/store shape with a focused commit before touching `GET /api/sessions`.

### Task 2: Split `GET /api/sessions` into polled rows and bootstrap metadata

**Files:**
- `codoxear/server.py`
- `tests/test_pi_server_backend.py`
- `tests/test_frontend_contract_source.py`
- `web/src/lib/types.ts`
- `web/src/lib/api.ts`
- `web/src/lib/api.test.ts`

- [ ] Add backend tests for a new `GET /api/sessions/bootstrap` route returning `recent_cwds`, `cwd_groups`, `new_session_defaults`, and `tmux_available`.
- [ ] Add or update route tests so `GET /api/sessions` now returns only `sessions` rows plus only the minimal fields that truly belong to the polling contract.
- [ ] Add a backend projection helper that takes the rich internal `SessionManager.list_sessions()` row and emits a frontend-sized row instead of changing the internal manager return shape directly.
- [ ] Remove clearly unused row fields from the polled HTTP response, especially `files`, `log_path`, `log_exists`, `state_busy`, `token`, `thinking`, `tools`, `system`, harness metadata, provider internals, tmux internals, and time-priority breakdowns.
- [ ] Add `SessionBootstrapResponse` in `web/src/lib/types.ts` and `api.getSessionsBootstrap()` in `web/src/lib/api.ts`.
- [ ] Update the frontend route-contract smoke test to include `"/bootstrap"`.
- [ ] Commit this backend/API split before changing the sessions store.

### Task 3: Refactor the sessions store so polling only refreshes session rows

**Files:**
- `web/src/domains/sessions/store.ts`
- `web/src/domains/sessions/store.test.ts`
- `web/src/app/providers.tsx` only if helper shape changes
- `web/src/lib/api.ts`
- `web/src/lib/types.ts`

- [ ] Change the sessions store tests to model two independent fetch paths: `refresh()` for polled rows and `refreshBootstrap()` for low-frequency metadata.
- [ ] Keep the active-session selection semantics and dedupe logic exactly as they work today.
- [ ] Update `createSessionsStore()` so `refresh()` only reads `api.listSessions()` session rows, while `refreshBootstrap()` reads `api.getSessionsBootstrap()` and updates `newSessionDefaults`, `recentCwds`, `cwdGroups`, and `tmuxAvailable` without touching the selected session.
- [ ] Add a small invalidation mechanism or explicit refresh calls so bootstrap data can be refreshed after create-session and cwd-group mutations without reintroducing polling.
- [ ] Commit the store split separately so later UI rewiring is easy to review.

### Task 4: Rewire the UI so bootstrap data is event-driven instead of polled

**Files:**
- `web/src/app/AppShell.tsx`
- `web/src/app/AppShell.test.tsx`
- `web/src/app/app-shell/useAppShellSessionEffects.ts`
- `web/src/app/app-shell/useAppShellSessionEffects.test.tsx`
- `web/src/components/new-session/NewSessionDialog.tsx`
- `web/src/components/new-session/NewSessionDialog.test.tsx`
- `web/src/components/sessions/SessionsPane.tsx`
- `web/src/components/sessions/SessionsPane.test.tsx`

- [ ] Load bootstrap metadata once during initial shell hydration instead of every polling tick.
- [ ] Keep the polling loop in `useAppShellSessionEffects.ts` pointed only at `sessionsStoreApi.refresh()` for the lightweight session rows.
- [ ] Refresh bootstrap metadata explicitly after:
  - successful cwd-group edits
  - opening the new-session dialog when bootstrap data is missing or stale
  - successful create/resume/duplicate flows if recent cwd/default-dependent data needs to be refreshed
- [ ] Keep delete-session and normal sidebar polling on the lightweight session-row path only.
- [ ] Preserve current visible/hidden polling semantics while removing bootstrap metadata from the interval path.
- [ ] Commit the UI/store integration separately.

### Task 5: Final cleanup and regression pass

**Files:**
- any touched files above
- `docs/superpowers/plans/2026-04-10-web-polling-optimization.md` to save the approved durable copy before execution starts

- [ ] Remove stale fallback wiring left over from the pre-split session-ui shape only if tests prove it is no longer needed.
- [ ] Rerun the focused backend and frontend suites for live/workspace/session-list/bootstrap behavior.
- [ ] Manually verify the UI with the browser open: sidebar polling, workspace open/close, ask-user refresh, new-session defaults, and cwd-group edits.
- [ ] Land the final commit for the `/api/sessions` split after verification passes.

## Verification

### Automated

```bash
python3 -m pytest \
  tests/test_pi_server_backend.py \
  tests/test_frontend_contract_source.py -q

cd web && npm test -- \
  src/lib/api.test.ts \
  src/domains/messages/store.test.ts \
  src/domains/live-session/store.test.ts \
  src/domains/session-ui/store.test.ts \
  src/domains/sessions/store.test.ts \
  src/components/conversation/AskUserCard.test.tsx \
  src/components/conversation/ConversationPane.test.tsx \
  src/components/workspace/SessionWorkspace.test.tsx \
  src/components/new-session/NewSessionDialog.test.tsx \
  src/components/sessions/SessionsPane.test.tsx \
  src/app/app-shell/useAppShellSessionEffects.test.tsx \
  src/app/AppShell.test.tsx
```

### Manual

- active busy session polls `live`, not legacy detail endpoints
- idle active session uses the slower live cadence
- workspace refreshes only while the workspace dialog/sheet is open
- hidden tab pauses polling and visible resume catches up immediately
- sidebar polling hits the lightweight `GET /api/sessions` response only
- opening the new-session dialog still shows provider/model defaults and recent working directories
- editing a cwd group still updates labels/collapsed state without putting `cwd_groups` back on the polling path
- duplicating/resuming sessions still works with the slimmer session-row payload
