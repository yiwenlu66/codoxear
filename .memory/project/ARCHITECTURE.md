# Codoxear Frontend Architecture

## Problem

The frontend has 66 IIFE modules connected by implicit dependency injection.
Functions from the original monolithic `app.js` were extracted into separate
closures without verifying that every destructured reference is actually
provided. This creates silent runtime failures that only surface when a user
action triggers the broken path.

There is no single source of truth for any displayed state. The transcript's
rendered range, scroll position, and history cursor are managed by three
uncoordinated runtimes. The session list's model/effort display is written
by two functions with different priority rules. The sidebar's active-session
highlight is applied by one CSS rule but overridden by another.

## Principles

1. **One state authority per domain.** Every displayed value has exactly one
   declared writer. Other modules read through the authority's interface.

2. **Every dependency is explicit.** A module declares its required inputs
   as a typed options object. The wiring checker verifies at deploy time that
   every required name is provided. No implicit closure references.

3. **State transitions are explicit.** The transcript, session list, and
   modal stack each have a state machine that controls what operations are
   legal in each state.

4. **Design tokens encode decisions, not just values.** `--ctl` means
   "primary action touch target." `--ctl-chrome` means "compact secondary
   chrome control." The rule for which to use is in the token's comment.

## Architecture

### State Stores

Three domain stores, each owned by a single controller:

**SessionStore** — owns the session list, active session ID, per-session
metadata (model, effort, busy, queue). All writes go through this store.
Modules like the sidebar, message flow, and new-session dialog read from it.

**TranscriptViewController** — owns the rendered DOM range, scroll position,
and history cursor. State machine: LIVE / BROWSING / LOADING_OLDER / REPLACING.
Only REPLACING may clear the DOM. All other operations are append/prepend.

**UIStore** — owns modal visibility, sidebar drawer state, and viewport
classification (desktop/touch). One writer per modal.

### Dependency Injection

Every module factory declares its options:

```javascript
function createTranscriptViewController(options) {
  // options: {
  //   domRuntime: TranscriptDomRuntime,
  //   scrollRuntime: TranscriptScrollRuntime,
  //   olderLoadRuntime: OlderLoadRuntime,
  //   onStateChange: (state) => void,
  // }
}
```

The wiring checker (`scripts/check_wiring.py`) verifies statically that
every required option is provided by the creation call. No implicit
references through closure or `window` globals.

### Transcript View State Machine

```
LIVE ←→ BROWSING ←→ LOADING_OLDER
  ↑                       ↓
  └──── REPLACING ←───────┘
       (session switch)
```

| State | Entry | Allowed operations | Exit |
|-------|-------|-------------------|------|
| LIVE | After openSession, after send | appendEvents, scrollToBottom | User scrolls up → BROWSING |
| BROWSING | User scrolled up | appendEvents (no scroll), loadOlderMessages | User scrolls to bottom → LIVE, older load → LOADING_OLDER |
| LOADING_OLDER | Top edge hit or click | prependEvents on success, stay on failure | Load completes → BROWSING |
| REPLACING | Session switch, jump-to-latest force | replaceWith(new events) | Render complete → LIVE |

**Forbidden:** DOM clearing outside REPLACING. Poll/SSE triggering REPLACING.
Send triggering REPLACING.

### File Structure

Target: ~20 focused domain modules. Each module has one clear responsibility.

```
app.js                          — boot (34 lines)
app_application.js              — factory + deps assembly (854)
app_application_composition.js  — renderApp + wiring (~1400)
app_state.js                    — SessionStore, UIStore
app_transcript_view.js          — TranscriptViewController (state machine)
app_transcript.js               — transcript data model, message normalization
app_transcript_render.js        — DOM rendering, row construction
app_transcript_scroll.js        — scroll position, jump button, auto-load trigger
app_message_flow.js             — poll/SSE transport, send lifecycle
app_message_history.js          — older message paging, cursor management
app_sessions.js                 — sidebar rendering, session actions
app_session_lifecycle.js        — open/select/spawn/delete sessions
app_session_refresh.js          — session list refresh, sidebar reconciliation
app_file_viewer.js              — file viewer orchestration (737 lines after decomposition)
app_file_picker.js              — file candidate picker, search, navigation
app_file_editor.js              — Monaco editor integration, save/dirty
app_composer.js                 — message input, draft persistence, attachments
app_modal.js                    — dialog framework, keyboard handling
app_wiring.js                   — options selectors for all modules
app_shell.js                    — DOM construction, element references
app_display.js                  — formatting, truncation, display helpers
app_api.js                      — HTTP client, ETag caching
```

### CSS Design System

Two control sizes, defined by role:

| Token | Value | Use |
|-------|-------|-----|
| `--ctl` | 44px | Primary actions (Send, Start session, dialog confirm) |
| `--ctl-chrome` | 32px | Secondary chrome (topbar nav, sidebar hover, composer secondary) |

Rules:
- All `.icon-btn` default to `--ctl-chrome` (32px)
- `.icon-btn.primary` overrides to `--ctl` (44px)
- Touch hit area is handled by `::after` pseudo-element, not visual size
- No hardcoded control dimensions anywhere in the stylesheet

## What This Fixes

1. **"Messages disappear on send"** — TranscriptViewController's REPLACING
   state is the only path that can clear. Poll, send, and SSE cannot trigger it.

2. **"Load older messages broken"** — The cursor is owned by one controller.
   Poll responses cannot overwrite it. The history endpoint accepts stale
   cursors via `decode_message_cursor_target`.

3. **"Hover buttons misplaced"** — The two-branch DOM split is locked
   (AGENTS.md). Desktop uses `sessionActionsInline`, touch uses `sessionSwipe`.
   `.session` has `position: relative` so absolutely-positioned children
   are contained within the card.

4. **"Sidebar model name wrong"** — `apply_run_settings_backfill` respects
   the bridge-live priority. The bridge reads `pi.getModel()` on every
   `turn_end` and writes it to the caps file. No log scan needed for
   active sessions.

5. **"Button sizes inconsistent"** — Two tokens, one rule. Base button
   is `--ctl-chrome`. Primary is `--ctl`. No exceptions.

## Verification

Each architectural claim must be verified behaviorally, not by assertion:

- TranscriptViewController: select session, send message, verify count
  doesn't decrease. Click load-older, verify count increases. Scroll up,
  click jump-to-bottom, verify count returns to tail.
- Wiring checker: deploy gate must pass with 0 definite bugs.
- CSS: computed styles of active session vs inactive must differ visibly.
- Button sizes: all `.icon-btn` must be 32px or 44px, no other sizes.
