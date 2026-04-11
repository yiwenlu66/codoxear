# AppShell Responsibility Split Design

**Date:** 2026-04-10

## Goal

Refactor `web/src/app/AppShell.tsx` into smaller, responsibility-focused modules so the shell becomes easier to read, test, and extend without changing current user-visible behavior.

The immediate target is maintainability, not a product redesign.

## Approved Direction

Use a responsibility-based split instead of a purely visual split.

User-approved decisions:

- Split `AppShell` by responsibility rather than only by screen region
- Allow light responsibility correction during the refactor
- Preserve current user-facing behavior unless a boundary is obviously wrong
- Keep the existing shell layout and current features working while the internals are reorganized

## Problem Statement

`web/src/app/AppShell.tsx` has grown into a mixed-control file that currently owns too many unrelated concerns.

Observed issues:

- The file mixes shell composition, modal open state, session polling, push notification logic, audio playback, reply sound dedupe, and toolbar rendering
- Several helpers and icon components live in the same file as long-running side effects, which makes the main control flow hard to scan
- The file contains multiple effect clusters with different lifecycles, making regressions harder to isolate
- Notification behavior is split across more than one local helper path, which increases the risk of semantic drift
- The current shape makes both review and future feature work slower because the file is too large to hold in working memory comfortably

## In Scope

- Split `AppShell` into focused UI subcomponents and side-effect hooks
- Keep top-level shell composition in `AppShell.tsx`
- Move session polling and background session refresh behavior into a dedicated hook
- Move announcement playback and related audio lifecycle logic into a dedicated hook
- Move desktop/push notification coordination and reply-sound dedupe logic into a dedicated hook
- Move shell-specific dialogs, sheets, and overlay composition into focused components
- Consolidate duplicate or overlapping notification helper logic into one clear responsibility boundary
- Add or update targeted tests so the refactor is locked down behaviorally

## Out of Scope

- Redesigning the shell UI
- Changing API contracts or backend endpoints
- Replacing the current stores or introducing a new global state library
- Reworking conversation, session, or workspace semantics
- Broad cleanup of unrelated large test files outside what this refactor needs to keep behavior safe
- New features unrelated to AppShell maintainability

## Why This Direction

A responsibility split addresses the actual maintenance problem more directly than a visual-only split.

- The hardest part of the current file is not the JSX alone; it is the interleaving of UI composition and long-lived effects
- Polling, notifications, and audio each have distinct lifecycle rules and should be understandable independently
- Extracting the shell overlays and toolbar composition makes the remaining `AppShell` read like an orchestrator instead of a monolith
- This structure gives future changes obvious homes, which reduces the chance that new logic gets bolted back onto the root file

## Design Principles

1. **`AppShell` should orchestrate, not implement everything**
   - The root component should assemble shell pieces and own only state that is truly page-local.

2. **Effects should live with the lifecycle they manage**
   - Session polling, audio playback, and notification delivery should be split by side-effect domain, not left interwoven in one file.

3. **Preserve semantics before improving internals**
   - This refactor is successful only if the current shell behavior stays intact.

4. **Prefer clear module seams over clever abstraction**
   - Extract modules that correspond to recognizable responsibilities.
   - Do not build a generic framework around one component.

5. **Keep state ownership close to the UI that actually controls it**
   - Dialog open state can remain in the shell when it is page-local.
   - Derived side-effect state should move into hooks when it belongs to a longer lifecycle.

## Proposed Module Structure

### Root orchestrator

`web/src/app/AppShell.tsx` becomes the composition layer.

Responsibilities that remain here:

- read top-level stores needed for shell composition
- own page-local open/close state for shell overlays when that state is directly tied to visible shell controls
- wire callbacks between toolbar, overlays, and existing panes
- compute small view-model values such as the active title and current enablement state for toolbar actions

The root file should no longer directly contain large audio, notification, or polling implementations.

### Session effects hook

Create `web/src/app/app-shell/useAppShellSessionEffects.ts`.

Responsibilities:

- refresh the sessions list on its existing interval
- load active-session messages and session UI state when the active session changes
- poll the active session on its existing interval
- prime background busy sessions for reply-sound behavior
- poll background busy sessions that are eligible for reply-sound tracking
- keep the current missing-session recovery behavior

Inputs should be explicit and close to current semantics:

- active session identity and backend
- current sessions list
- store APIs
- reply-sound enablement where needed for existing priming behavior

This hook should not render UI and should not own unrelated modal state.

### Audio hook

Create `web/src/app/app-shell/useAppShellAudio.ts`.

Responsibilities:

- read and merge voice settings
- manage announcement listener keepalive registration
- manage native HLS vs `hls.js` playback selection
- retry audio playback when the browser blocks or the stream errors
- expose imperative actions needed by the shell, such as toggling announcements and optionally triggering playback from a user gesture
- expose test-sound and test-announcement helpers used by settings UI

This hook should own:

- the live audio element ref
- HLS instance lifecycle
- audio retry timer lifecycle
- voice settings state and drafts that are semantically part of the voice settings surface

### Notification hook

Create `web/src/app/app-shell/useAppShellNotifications.ts`.

Responsibilities:

- manage desktop notification enablement and permission state
- keep push-subscription status in sync for mobile devices
- own delivered-notification dedupe, message lookup retry state, and reply-sound dedupe
- trigger local reply beep behavior on notification feed or final response events according to current semantics
- expose notification labels and toggle handlers used by the sidebar actions and settings surface

This hook should also consolidate the current duplicate desktop-notification helper into one implementation.

The notification hook may depend on small utility functions shared with the audio hook, but the ownership boundary should remain: notifications decide *when* to notify, audio decides *how* announcement playback works.

### UI subcomponents

Create focused render components under `web/src/app/app-shell/`.

#### `AppShellSidebar.tsx`

Responsibilities:

- render the sessions rail wrapper
- render the brand/action header
- render footer actions such as settings and logout
- delegate the actual session list body to `SessionsPane`

Expected inputs:

- notification/announcement labels and toggle handlers
- settings/logout callbacks
- new-session callback
- any current enablement flags used by the existing controls

#### `AppShellToolbar.tsx`

Responsibilities:

- render the conversation title
- render desktop toolbar actions
- render the mobile tools trigger and tools menu
- keep the mobile-tools outside-click handling local if that remains the cleanest ownership boundary

Expected inputs:

- active title
- active session availability/busy state
- file/workspace/harness/interrupt callbacks
- whether mobile sheets are in use

#### `AppShellWorkspaceOverlays.tsx`

Responsibilities:

- render the mobile sessions sheet
- render the mobile workspace sheet
- render the desktop workspace dialog
- render `FileViewerDialog`, `HarnessDialog`, and `NewSessionDialog`
- render the voice settings dialog if it remains shell-owned

Expected inputs:

- the current open/close state and handlers for each overlay
- active session identity and session-ui derived data
- settings draft state and action callbacks

This component groups overlay composition so the root shell no longer interleaves the main page structure with every dialog and sheet.

### Utilities and local types

Move pure helpers into one or more small modules, for example:

- `web/src/app/app-shell/utils.ts`
- `web/src/app/app-shell/icons.tsx`
- `web/src/app/app-shell/types.ts`

Good candidates:

- `shortSessionId`
- local-storage toggle helpers
- device detection helpers
- `base64UrlToUint8Array`
- `mergeVoiceSettings`
- reply-sound key helpers
- icon components now embedded in the root file

These should stay pure and independently testable.

## State Ownership Plan

### State that stays in `AppShell`

Keep shell-local UI control state in the root when it is only used to open and close visible shell surfaces.

Examples:

- `newSessionOpen`
- `detailsOpen`
- `workspaceOpen`
- `fileViewerOpen`
- `harnessOpen`
- `sidebarOpen`
- `mobileToolsOpen` unless it is cleaner to localize it inside `AppShellToolbar`

### State that moves into hooks

Move state into hooks when it is part of a longer-running side-effect lifecycle or is tightly coupled to one subsystem.

Examples:

- voice settings data and announcement playback internals move into the audio hook
- notification permission, push-subscription tracking, delivered notification caches, lookup retry state, and reply-sound dedupe move into the notification hook
- polling timers and background-session priming state move into the session-effects hook

### State that should not be duplicated

The refactor should avoid copying store-derived values into local state unless the value is explicitly a UI draft.

## Behavior Preservation Requirements

The refactor must preserve these current behaviors:

1. Session list refresh still occurs on the existing interval.
2. Active session messages and session UI still refresh on the existing interval.
3. Missing-session `404` recovery still triggers a sessions refresh.
4. Background busy sessions still prime message history for reply-sound behavior.
5. Announcement playback still supports native HLS where available and `hls.js` otherwise.
6. Mobile push notification enrollment still uses the existing service worker path logic.
7. Desktop notifications still dedupe by message id and use the current fallback lookup path when notification text is missing.
8. Reply beep behavior still respects existing dedupe rules.
9. Toolbar buttons, mobile sheets, workspace dialog, file viewer, harness dialog, and new-session dialog still open and close as they do today.
10. Existing no-session disabled states still hold.

## Data Flow

### Session lifecycle flow

1. `AppShell` reads the active session and sessions list from the existing stores.
2. `useAppShellSessionEffects` receives the active-session context plus store APIs.
3. The hook refreshes session lists and polls active/background sessions.
4. Store updates continue to drive `ConversationPane`, `Composer`, and workspace surfaces.

### Audio and notification flow

1. `AppShell` mounts a live audio element and passes its ref into the audio hook.
2. `useAppShellAudio` manages voice settings, listener registration, and announcement playback.
3. `useAppShellNotifications` consumes current message data, permission state, and push settings to decide when to emit desktop/push/local notification cues.
4. Sidebar actions and settings UI call hook-exposed handlers rather than reaching directly into low-level browser APIs.

### Overlay flow

1. `AppShell` owns shell-local open state and passes it to the toolbar/sidebar/overlay components.
2. `AppShellToolbar` and `AppShellSidebar` raise user actions through callbacks.
3. `AppShellWorkspaceOverlays` renders the current overlay surfaces from a single grouped composition point.

## Testing Strategy

Keep tests behavior-first.

### Existing integration coverage

Continue using `web/src/app/AppShell.test.tsx` as the primary shell behavior test file for:

- toolbar actions and disabled states
- mobile sheets and workspace dialog behavior
- voice settings surface behavior
- push registration and service worker wiring already covered there

Update tests only where file paths or render boundaries require it.

### New focused tests

Add targeted tests where extraction creates a cleaner seam.

Recommended coverage:

- `useAppShellSessionEffects` refresh/poll lifecycle behavior
- notification helper behavior that is easier to verify outside the full shell render
- pure utility helpers moved out of the root file

The goal is not to explode the test count, but to avoid keeping every side-effect assertion trapped in one giant integration file.

## File-Level Plan

Likely new files:

- `web/src/app/app-shell/useAppShellSessionEffects.ts`
- `web/src/app/app-shell/useAppShellAudio.ts`
- `web/src/app/app-shell/useAppShellNotifications.ts`
- `web/src/app/app-shell/AppShellSidebar.tsx`
- `web/src/app/app-shell/AppShellToolbar.tsx`
- `web/src/app/app-shell/AppShellWorkspaceOverlays.tsx`
- `web/src/app/app-shell/utils.ts`
- `web/src/app/app-shell/icons.tsx`
- optional focused test files under `web/src/app/app-shell/`

Likely modified files:

- `web/src/app/AppShell.tsx`
- `web/src/app/AppShell.test.tsx`

## Risks and Mitigations

### Risk: hidden behavior regressions during extraction

Mitigation:

- preserve current intervals and condition checks exactly before attempting cleanup
- move logic first, then simplify only where semantics are clearly duplicated
- rely on focused behavior tests plus the existing shell integration suite

### Risk: hooks become "mega-hooks"

Mitigation:

- split by subsystem, not by convenience
- keep hook APIs explicit and narrow
- leave page-local dialog state in the root instead of shoving everything into one extracted controller hook

### Risk: settings dialog ownership becomes ambiguous

Mitigation:

- keep the settings dialog under the overlay composition component for now
- only move it further if a later feature creates a clearer dedicated owner

### Risk: current dirty workspace makes the refactor harder to isolate

Mitigation:

- limit file touch surface to `AppShell` and the new `web/src/app/app-shell/` subtree where possible
- avoid opportunistic cleanup outside the approved scope

## Acceptance Criteria

This design is complete when:

- `web/src/app/AppShell.tsx` becomes substantially smaller and reads primarily as a shell orchestrator
- session polling, notifications, and announcement playback no longer live inline in the root file
- dialogs/sheets/toolbar/sidebar rendering are grouped into focused components
- current shell behavior remains intact under automated tests
- future changes to a single shell subsystem can be made without reloading the entire monolith into context
