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

Markdown code blocks always visually wrap via `.md pre` in `app.css`
(`white-space: pre-wrap; overflow-wrap: anywhere`); nested code inherits whitespace.
Code text/copy payloads remain unchanged. Parsed-CSS regression coverage is in
`tests/test_markdown_codeblock_wrap_css.py`; Docker browser/copy acceptance is in
`scripts/docker_markdown_codeblock_wrap_verify.py`.

`app_transcript_view.js` owns the rendered DOM range and history cursor.
State machine: LIVE / BROWSING / LOADING_OLDER / REPLACING. Only
REPLACING may clear the DOM.

`createTypingRowStoreProjection` owns transcript child-activity projection from
the selected-session store. It reads count and detail records atomically and
writes the same automatically visible per-child lines into busy and idle
activity bubbles. The bubbles use intrinsic inline-grid sizing up to the normal
message maximum, so short telemetry does not stretch the row and long telemetry
wraps without horizontal overflow. Qualified `provider/model` identifiers render
as `model`, and cumulative usage is labeled `tokens`. The sidebar marker remains
summary-only; message flow never calls a child-detail renderer directly.

### CSS design system

Two control sizes by role: `--ctl` (44px, primary actions) and
`--ctl-chrome` (32px, secondary chrome). Touch hit area is `::after`,
not visual size. Full design-language rules are in AGENTS.md.

## Backend

### Sidebar order and the tmux window mirror

Sidebar order has one writer: `session_listing.sort_session_rows`, key
`(group rank, -final_priority, -updated_ts, -start_ts, session_id)`. The group
rank (Now → Waiting → Later, blocked before snoozed) is part of the key because
the renderer emits blocks and both deferred groups carry `final_priority 0`
after clamping; sorting by priority alone interleaved them. The browser renders
the server's array order verbatim (`app_session_refresh.js` does not re-sort;
`sidebarSessionEntries` only partitions).

`codoxear/tmux_order.py` (pure planning) + `codoxear/tmux_order_sync.py`
(schedule) mirror that order into the window indices of
`CODEX_WEB_TMUX_SESSION`. The listing route hands each `/api/sessions`
generation — the exact array a client renders — to a coalescing single-flight
worker (`TmuxOrderSync.tick`, driven by a 1s daemon loop). The pass swaps only
codoxear windows among the indices they already occupy, so a window no session
names never moves and no window is created, killed, renamed or selected.

Load-bearing details (measured, not assumed):

- `swap-window -d` is unusable as a viewer guard: on tmux 3.5a it is ignored for
  a swapped current window, while on 3.8 it keeps the viewer's pane there but
  moves the viewer to the swap source when neither swapped window is current.
  The mirror never passes `-d`; it restores the session's current window with
  `select-window -t <id>` when the pass moved it.
- The settle window is anchored to the first *unapplied* generation, not the
  newest arrival — a client polling faster than the window otherwise starves
  the pass forever (caught by the Docker verifier).
- Work is driven by listing generations, never a timer: with nobody polling,
  `list_sessions` never runs and `updated_ts` never advances, so a timer would
  reorder against stale priorities.
- Window identity is `(row.tmux_session, row.tmux_window)`: `new-window -n`
  pins the name (`automatic-rename` off per window), so it survives the backend
  running. Ambiguous or missing names are skipped and logged, never guessed.

Behavioral contracts: `tests/test_tmux_order.py`,
`tests/test_tmux_order_sync.py`, `tests/test_tmux_runtime.py`,
`tests/test_sidebar_render_order.py` (order is rendered, not re-derived);
Docker acceptance: `scripts/docker_tmux_order_verify.sh`.

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

### Voice delivery invariants

- Voice/notification delivery reuses the transcript's structural exclusion:
  `_extract_delivery_messages` applies `pi_log_row_is_transcript_excluded`, so
  every `custom_message` envelope (intercom, subagent control, supervisor
  updates) and every tagged harness row is agent plumbing — never announced
  and never a feed item. Speakable classes are only `narration` and
  `final_response`.
- The TTS speech boundary is sanitized: `_sanitize_spoken_text`
  (`voice_push_state.py`) strips UUIDs, hex hashes, and long numeric IDs from
  anything synthesized, covering both the LLM-summary path and the verbatim
  short-message path (narration under 15 words skips summarization).
- Per-session announcement voice is deterministic:
  `DEFAULT_VOICES[sha256(sock-stem session_id)[:8] mod 13]`. The catalog is
  male-skewed (7 male-leaning of 13); the mapping itself is uniform.

### Composer-draft sync uses tombstone deletion

Server-synced drafts (`session_drafts.json`, `DraftStore`) propagate every
write — including deletion — through one last-writer-wins channel keyed on
the server wall clock. Deleting a draft (POST empty/whitespace text) writes
a timestamped empty-text tombstone entry; it never removes the map entry.
Entry removal happens only in `clear_deleted_session_state` (session
deletion). Reason: clients hold companion `server_ts > 0`; a removed entry
reads as `updated_ts 0`, looks older than their local copy, and gets
re-pushed (draft resurrection after a send from another client). Consequences:
- `load`/`save` must never prune empty-text entries (whitespace-only text
  normalizes to the canonical `{"text": "", ...}` tombstone).
- Blank clears bypass the 256 KiB cap; only non-blank text is capped.
- Session rows expose the tombstone ts as `draft_updated_ts`; `0.0` means
  the session has no entry at all.

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

Pi first-send acceptance is independent of rollout persistence. The bridge's
`message_start(user)` writes `<active-session-marker>.receipt` after preflight
and before persistence; broker/sessiond read it before quiet-window settlement,
matching Pi pid, ECMAScript-trimmed submission SHA-256, and event milliseconds.
A match marks the current idle-origin latch accepted, preventing an undelivered
verdict while the first assistant response/log is pending. Input transforms can
prevent digest matching; PTY/log evidence remains fallback. Slow preflight before
`message_start` remains ambiguous. Behavioral coverage: `test_pi_prompt_receipt.py`.

Subagent model identity comes from the child's own session, not the run mirror.
`status.json` records the launch-time model, so a child launched on a `jev-auto`
placeholder (`pi-jev-auto-model`) keeps the placeholder there until it routes
itself; only the child's session file records what it selected (the `custom`
entry `jev-auto-model:routed:v1` with provider/modelId/level, plus the
`model_change` it applies). The run's `events.jsonl` mirrors that entry only when
pi-subagents' event subscription already exists at routing time, and a
revive/resume child is created with its task already attached, so it routes
during session creation and never reaches the mirror: measured here, the mirror
named the model for 82 of 105 jev-auto children but only 1 of the 24 revive/resume
children, and the sidebar and activity bubble showed `auto-jev-subagent` for the
rest. `util._resolve_jev_routed_model`
reads the child session head first (bounded byte cap; routing sits within 2 KB of
the head) and the events mirror second, and keeps the recorded placeholder when
neither names a model. Coverage: `tests/test_pi_subagents.py` and the rendered
label in `scripts/docker_subagent_activity_layout_verify.sh` (its `jev` case).
This is display-only: the children resolve and run. A placeholder that actually
reaches the model API is a different defect in the resume path, not a projection
bug (see `.memory/tasks/2026-10-24-subagent-resume-model-routing/`).

### Offline cold-load authority
The canonical SW cache/readiness contract is in AGENTS.md. Activation must warm the uncontrolled first navigation's shell and versioned dependencies; runtime fetch caching alone requires an extra visit. `/api/me` is boot-only: when activation precedes authentication, successful login must refresh it after best-effort worker readiness, otherwise an authenticated UI can render online yet have no cached auth snapshot for offline boot. Theme links are generated dynamically, so the SW warms its explicit family list at the shared asset version; the behavioral activation test pins the list against actual theme files. Transcript caches remain populated only by opening sessions online.

Network-first response delivery and cache durability are separate lifetimes: delivery settles at headers, a synchronous clone is stored under the fetch event's pre-registered `waitUntil`, and cache-write failures cannot reject the live answer. The race no longer bounds a peer that sends headers and then stalls the body. Static/warm-up writes remain awaited. In-scope session and absolute `office_preview`/`video_preview` routes, plus every Range request, bypass interception so native streaming/206 semantics cannot be replaced by a cached full 200. Derivatives therefore have no service-worker offline fallback; non-Range blob/raw/download caching is unchanged.

## Deployment runtime resolution (verified 2026-10-10)

The pipx venv (`~/.local/share/pipx/venvs/codoxear`) supplies only the interpreter and dependencies at runtime. The service's `WorkingDirectory=~/.local/share/codoxear/deploy` plus `python -m codoxear.server` puts the deploy worktree first on `sys.path`, so all code and static assets (including `STATIC_DIR` and `dist/app.bundle.js`) resolve from the worktree, not site-packages. setuptools package-data deliberately excludes `static/dist/*`, so the installed package never contains the bundle — checking site-packages for served assets is a dead end; check the deploy worktree (`/proc/<MainPID>/cwd`) instead.

## Frontend commits must carry the rebuilt bundle

`dist/app.bundle.js` is tracked, and the Docker verify harnesses serve the
committed bundle without rebuilding it. A frontend commit that updates ESM
sources but not the bundle makes behavioral verification test the previous
UI. Rebuild with deploy.sh's exact invocation before committing:
`npx esbuild codoxear/static/app.js --bundle --minify --outfile=codoxear/static/dist/app.bundle.js --format=esm`.

## Office document preview derivatives

`office_preview.py` converts DOC/DOCX/XLS/XLSX/PPT/PPTX into cached PDFs only when
an authenticated `office_preview` URL is fetched. Read/inspect retain the source
kind/MIME/size and never invoke LibreOffice; blob/download URLs still serve the
original. Session and absolute-file routes share the converter. The frontend
uses the existing lazy PDF.js surface, with a conversion preflight to expose
readable server failures and a separate original-file download target. Failures
identify transport, conversion/authentication, or renderer stages. Retry is
available while that failed open request is still current; it re-enters the
normal open flow with a fresh request/abort identity and reruns read/preflight,
so closing or switching files makes an old Retry control inert. Conversion
failures retain the existing 15-second negative-cache cooldown; immediate Retry
can report that recent failure until it expires. The Docker Office verifier
waits out that cooldown before proving a successful transient-failure Retry.

Calc export normalizes a disposable XLSX: unhide every sheet, remove saved
Print_Area restrictions, disable fit-to-page/percentage shrinking, preserve
Print_Titles. Legacy XLS first converts to XLSX. This includes hidden-sheet and
outside-print-area content on ordinary pages without modifying the original;
SinglePageSheets is deliberately avoided because oversized pages become giant
PDF canvases. Each conversion has a private macro-disabled/link-update-disabled
LibreOffice profile, a bounded process-group deadline and concurrency, and an
atomic publish after confirming source identity is unchanged. Cache identity
includes source dev/inode/size/mtime/ctime and renderer settings. Clear only owned
PDF derivatives after font/LibreOffice changes, never source documents.

Behavioral contracts: `tests/test_office_preview.py`,
`tests/test_office_file_routes.py`, `tests/test_office_preview_frontend.py`;
Docker/browser acceptance: `scripts/docker_office_preview_verify.sh`.

## Optional native dependency provisioning

`system_dependencies.resolve_executable` is shared by Office/video conversion,
voice HLS, Git viewer operations and server tmux probe/launch/prune. It preserves
PATH priority, supports native/Homebrew/app-bundle fallbacks, and does not add
cwd lookup when PATH is empty. Broker/sessiond initialization and deploy retain
their existing raw Git/PATH contract. `codoxear-doctor` is an installed, opt-in
CLI: checks by default; explicit selected native package installation; standalone
owned-Office-PDF cache clearing. Office version/basic text probes do not certify
Calc/Impress, so explicit Office provisioning always ensures all three modules.
See README “System dependencies” for platform maps, commands and boundaries.
