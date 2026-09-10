# Codoxear Architecture

## Frontend

### State ownership

Every mutable state field has one owner module:

- `app_session_state.js` — 8-field observable runtime store (`selected`,
  `running`, `queueLen`, `subagentsRunning`, `subagentDetails`, `turnOpen`,
  `sending`, `token`). `subagentsRunning` and `subagentDetails` commit as one
  selected-session liveness snapshot so same-count telemetry updates remain
  observable.
  API: `get`/`set`/`subscribe`/`applyRuntime`.
- `app_session_catalog.js` — session list (`latestSessions`), derived
  `sessionIndex`, `recentCwds`, `newSessionDefaults`, `tmuxAvailable`.
  API: `patchSession`/`applySnapshot` (atomic writes).
- `app_polling.js` — poll scheduling (`createPollingRuntime`) and
  async-epoch cancellation (`createAsyncEpoch`).

Every rendered value has one trigger: widget-internal store subscriptions.
No cross-module renderer calls. Reducers write stores atomically
(`applyRuntime`/`applySnapshot`/`patchSession`); imperative render relays
are an anti-pattern the wiring guard and reviews reject.

### Widget rule

Element + render + subscription in one module:

- `app_topbar.js` — status chip, context chip, interrupt button.
- `app_notifications.js` — notification runtime (`enabledLocally`/
  `syncState`/`pollFeed`/`dispose`).
- `app_help.js` — help overlay.
- `app_conversation_copy.js` — conversation-copy formatter and
  message count.

### Wiring

All controllers receive per-controller `select()` contracts from
`app_wiring.js`. No option bags, no spreads. The architecture guard
`scripts/check_wiring.py` enforces 7 check types:

1. `pass-through-factory` — factories that return deps verbatim
2. `bag-spread` — `...options` / `...deps` into controller calls
3. `direct-bag-argument` — passing a bag object directly
4. `global-registration` — `window.Codoxear*` / `global.Codoxear*`
5. `select-undercoverage` — selector missing keys a controller uses
6. `unbound-option-value` — option literal with no matching dep
7. `select-callsite-coverage` — selector keys not covered by callers

A checked-in allowlist (`scripts/wiring_guard_allowlist.json`) tracks existing
violations. Every entry requires a reason, and tests pin the exact entry count
per check type. Cross-commit monotonicity is enforced through those visible
reasons and counts plus review; comparing against the current commit cannot
prove that a committed violation and exception did not grow together.

### Composition

- `app.js` — 34-line boot entrypoint.
- `app_application.js` — dependency facade / factory.
- `app_application_composition.js` — lifecycle assembly, controller
  creation via `app_wiring.js`, cleanup-registered listeners.
- `app_shell.js` — layout slots (`topMeta`, `titleRow`, `topActions`).
- `app.css` — presentation.
- `app_modal.js` — modal policy (`createModalPolicyController`,
  `createConfirmationController`, `createModalKeyboardHandler`).

### Domain and peripheral controllers

Domain: `app_chat_interaction.js`, `app_file_ops.js`,
`app_session_lifecycle.js`, `app_session_refresh.js`.

Peripheral: `app_message_flow.js` (confirmed-send/SSE/polling),
`app_composer.js`, `app_unattended.js`, `app_attachments.js`, plus
focused transcript, queue, file-viewer, launch, search, and voice
modules.

`app_session_display.js` no longer exists.

### Dependency injection

Constructor-injected DI is preserved — the VM-harness behavioral tests
inject mocks through these seams. The wiring guard disciplines injection;
it does not replace it.

### Transcript view

`app_transcript_view.js` owns the rendered DOM range and history cursor.
State machine: LIVE / BROWSING / LOADING_OLDER / REPLACING. Only
REPLACING may clear the DOM.

`createTypingRowStoreProjection` owns transcript child-activity projection from
the selected-session store. It reads count and detail records atomically and
writes the same automatically visible per-child lines into busy and idle
activity bubbles. The sidebar marker remains summary-only; message flow never
calls a child-detail renderer directly.

### CSS design system

Two control sizes by role: `--ctl` (44px, primary actions) and
`--ctl-chrome` (32px, secondary chrome). Touch hit area is `::after`,
not visual size. Full design-language rules are in AGENTS.md.

## Backend

### Log-derived session projection

All log-derived session state (busy, tokens, model, effort) commits
through one ordered, log-identity-aware boundary:

- `LogDerivedSessionObservation` (dataclass in
  `codoxear/session_log_projection.py`) — the observation record.
- `commit_log_observation` (in `codoxear/session_log_runtime.py`) —
  the commit function that updates session state atomically with log
  identity tracking.

### Live-delta projection

Poll and SSE share one live-delta projection:
`_project_live_record_window` in `codoxear/message_routes.py`. Both
transport paths produce identical transcript state.

### SessionManager wiring

`SessionManager` uses a retained coordinator graph. Coordinator
dependencies are focused dataclass records
(`SessionManagerCoordinatorDeps` and per-coordinator `*FactoryDeps` in
`codoxear/session_manager_factories.py`). No mega-caps objects, no
`*args`/`**kwargs` forwarding.

### Log normalization

- `codoxear/rollout_log.py` — chat-event extraction, delivery messages,
  idle detection, token snapshots (all backends).
- `codoxear/pi_log.py` — Pi-specific session headers, text extraction,
  final-turn detection, run settings, context usage.
- `codoxear/cc_log.py` — Claude Code log parsing.

## Pi integration strategy

Codoxear wraps all backends in a PTY so web and terminal share the same
session. Pi `--mode rpc` is a separate headless mode that cannot coexist
with the TUI. The extension bridge supplies Pi effort control from within
the live TUI process, while Pi `/model` remains its shared native
command. Extension load may write its passive capability marker and
register lifecycle listeners, but must not call runtime action methods at
load time. Command registration and runtime queries are deferred until
`session_start`/`session_switch`; a failed registration retries from a
later lifecycle event.
