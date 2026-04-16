# Preact + Vite Web Refactor

## Goal

Replace the current hand-written browser client in `codoxear/static/index.html`, `codoxear/static/app.js`, and `codoxear/static/app.css` with a single new `preact + vite` frontend while preserving existing product behavior and allowing small UX improvements.

## Current State

The current web UI is served directly by `codoxear.server` as static files and is implemented as one large JavaScript file plus one large stylesheet. The frontend mixes DOM rendering, polling, API calls, state mutation, mobile viewport handling, modal logic, and backend-specific behavior in the same layer.

This makes the UI hard to evolve in three ways:

1. State ownership is implicit and spread across global variables.
2. Feature additions have to thread through a very large `app.js` file.
3. The Python server is coupled to the current static asset layout (`/app.js`, `/app.css`, `/manifest.webmanifest`, `/service-worker.js`).

## Chosen Direction

Use a one-shot rewrite of the browser client with these constraints:

- Migrate directly to `preact + vite` instead of incremental coexistence.
- Keep feature coverage equivalent to the current UI.
- Allow small improvements to layout and interaction clarity, but not a broad product redesign.
- Use Vite dev server during development.
- Keep Python as the production host for built frontend assets and all `/api/*` routes.
- Use lightweight domain stores rather than a heavy state-management framework.

## Recommended Architecture

### Frontend project boundary

Create a dedicated frontend project under `web/`.

Responsibilities:

- `web/` owns source code, build config, TypeScript config, and generated frontend bundles.
- `codoxear.server` remains the runtime server for the app and continues to own all API endpoints.
- Production serving changes from hand-authored static files to Vite build output.

### Runtime integration

Development:

- Run Vite dev server for the browser client.
- Proxy `/api/*` requests from Vite to the Python server.
- Keep mobile/browser testing available without changing backend semantics.

Production:

- Build the frontend with Vite.
- Serve the generated `dist` output from Python.
- Preserve existing app URLs and API semantics.
- Keep manifest and service worker behavior aligned with the current PWA entrypoints.

### State and data flow

Use four layers:

1. `app shell` for top-level layout, global overlays, and app lifecycle wiring.
2. `domain stores` for session list, message stream, session workspace, composer state, and polling.
3. `api client` for all fetch logic against `/api/*`.
4. `ui components` for rendering only, with no direct polling or cross-feature orchestration.

This keeps service interactions centralized and prevents a JSX rewrite from reintroducing the same hidden coupling as the current DOM-driven implementation.

## Proposed Module Breakdown

## Directory shape

Recommended frontend structure:

- `web/src/main.tsx` - application bootstrap
- `web/src/app/` - app shell, providers, layout composition
- `web/src/domains/sessions/` - session list state, selection, refresh, metadata
- `web/src/domains/messages/` - message loading, incremental polling, busy state, normalization
- `web/src/domains/session-ui/` - file viewer, queue, diagnostics, Pi `ui_state` requests and responses
- `web/src/domains/composer/` - prompt draft, send flow, disabled state, submission handling
- `web/src/components/` - reusable presentational components
- `web/src/lib/` - fetch wrapper, formatting helpers, viewport helpers, shared utilities
- `web/src/styles/` - theme tokens, global CSS, layout primitives

### Primary UI surfaces

- `AppShell` handles mobile and desktop layout, sidebar collapse, global modals, and top-level notifications.
- `SessionsPane` handles session discovery, selection, owner/backend labels, status badges, and lightweight list interactions.
- `ConversationPane` handles message rendering, auto-scroll rules, loading states, and incremental updates.
- `Composer` handles input, send actions, keyboard behavior, and UI-response submission.
- `SessionWorkspace` handles secondary tools such as file reads, queue state, diagnostics, and related inspectors.

### Store boundaries

Recommended store split:

- `sessions store` - list polling, current session, launch defaults, remembered choices
- `messages store` - initial fetch, poll cursoring, append/replace rules, busy state
- `session-ui store` - Pi UI requests, file/queue/diagnostic panels, panel loading state
- `composer store` - draft text, submit lifecycle, disabled state, request-specific response payloads

Each store should expose derived selectors and actions instead of raw mutable objects.

## Functional Scope

The rewrite should preserve these existing capabilities before the old frontend is retired:

- authentication gate and session cookie flow
- session list polling and session selection
- session detail and message rendering
- prompt sending and backend-aware submission flows
- Pi `ui_state` fetch and `ui_response` submission
- new-session dialog and backend-specific defaults
- harness-related UI
- file viewer, queue, diagnostics, and related session workspace panels
- PWA shell basics including manifest and service worker integration
- mobile viewport and soft-keyboard behavior currently handled in the browser client

## Small UX Improvements Allowed

Small improvements are in scope when they do not redefine product behavior:

- clearer visual hierarchy between session list, conversation, and workspace panels
- more consistent modal and drawer structure
- better isolation of mobile-specific layout behavior
- cleaner grouping of backend/session metadata
- reduced UI jitter from broad rerenders during polling

Out of scope for this rewrite:

- new product concepts
- route-level information architecture changes
- backend API redesign
- broad visual rebranding

## Migration Milestones

### Milestone 1: Frontend scaffold

- Create `web/` with `preact + vite + typescript`.
- Configure dev server proxying to Python `/api`.
- Establish app shell, theme variables, and a minimal smoke-test page.

### Milestone 2: Data layer and contracts

- Rebuild the current `api()` helper as a typed client boundary.
- Recreate session polling, message polling, and Pi UI request flows in domain stores.
- Define normalized frontend models for sessions, messages, and workspace data.

### Milestone 3: Core user path

- Ship login shell, session list, conversation view, composer, and new-session dialog.
- Reach feature-equivalent behavior for the most common daily flow.

### Milestone 4: Advanced panels and backend-specific behavior

- Reattach file viewer, queue, diagnostics, harness UI, and Pi-specific interaction surfaces.
- Verify Codex and Pi backends both behave correctly.

### Milestone 5: Production switch-over

- Change Python static serving to the Vite build output.
- Keep a short-lived rollback switch for old vs new frontend entrypoint.
- Remove old hand-written frontend sources after validation.

## Server-Side Changes Required

`codoxear/server.py` will need a focused integration update:

- replace hard-coded static file assumptions with Vite build artifact serving
- preserve `/api/*` routes untouched
- preserve public entrypoints for manifest and service worker or map them cleanly to Vite-managed outputs
- keep cache/version behavior compatible with hashed assets or generated build metadata

The server should not absorb new frontend semantics; it should only host the new build correctly.

## Risks

### State-coupling regression

The main implementation risk is accidentally reproducing the current implicit coupling in a new component tree. This is avoided by forcing all server interaction through the client/store layer and preventing components from owning polling orchestration.

### Feature gap during one-shot cutover

A one-shot rewrite can miss lesser-used interactions. The mitigation is to build a migration checklist directly from the current frontend's visible UI regions and `/api/*` usage before implementation begins.

### Asset-path and PWA integration issues

Moving from fixed static filenames to Vite build output can break base paths, icons, manifest loading, and service worker registration if handled late. These paths should be decided at the start of the integration.

### Mobile behavior regressions

The current UI has explicit viewport and gesture handling. The rewrite must validate soft keyboard, viewport height variables, scroll locking, and touch interactions on real mobile browsers before cutover.

## Validation Strategy

Validation should follow capability parity, not visual polish first.

Required checks:

- login and authenticated app boot
- session list refresh and session switching
- initial message load and incremental message polling
- prompt send flow
- Pi UI request rendering and response submission
- new-session creation for both backends
- file/queue/diagnostic panel interactions
- harness UI behavior
- mobile sidebar, composer, and viewport behavior
- production asset serving from Python

Testing emphasis:

- unit tests for API client and domain store behavior
- targeted interaction tests for critical UI flows
- real-browser verification for mobile/PWA behaviors

## Acceptance Criteria

The rewrite is complete when:

- the new `preact + vite` frontend covers all current core web functionality
- Codex and Pi backends both work through the new UI
- development uses Vite with `/api` proxying to Python
- production is still served by Python from built frontend assets
- the old static `index.html` / `app.js` / `app.css` path is no longer required for normal operation
- mobile behavior is verified not to regress in key flows

## Open Follow-On Work

After the rewrite lands, future cleanup can consider:

- deeper component extraction for especially dense panels
- stricter typing for backend payloads
- more robust interaction or end-to-end test coverage
- optional visual refresh beyond the small improvements allowed in this rewrite
