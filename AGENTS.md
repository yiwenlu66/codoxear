# Codoxear architecture notes

This repo is a Linux-first companion UI for continuing local CLI agent sessions on a phone/laptop browser.

Currently supported agent backends:

- `codex`
- `pi`
- `cc` (Claude Code; launches `claude`)

## Components

### `codoxear.server`

- HTTP server (single process) that serves the UI and a small JSON API under `/api/*`.
- Auth: password gate using `CODEX_WEB_PASSWORD` (required). Cookie-based session (`codoxear_auth`).
- Session discovery: scans `~/.local/share/codoxear/socks/*.sock` for broker control sockets and reads the adjacent `*.json` metadata.
- Web-owned sessions: `/api/sessions` (POST) spawns a new broker process with `CODEX_WEB_OWNER=web` and a chosen `agent_backend`.
- Terminal-owned sessions: created by running `codoxear-broker` with the desired backend environment (for example plain Codex broker wrappers, `CODEX_WEB_AGENT_BACKEND=pi` for Pi, or `CODEX_WEB_AGENT_BACKEND=cc` for Claude Code).
- `GET /api/sessions` returns backend-aware launch defaults, including provider/model/reasoning choices per backend.
- Runtime state directory: `~/.local/share/codoxear` (legacy `~/.local/share/codex-web` is no longer used).
- Additional persisted UI state includes `session_sidebar.json`, `session_files.json`, `session_queues.json`, `unattended.json`, and `session_aliases.json` under the same app dir.

### `codoxear.broker`

- Foreground PTY wrapper intended to be run from a real terminal.
- Starts the selected backend CLI (`codex`, `pi`, or `claude`), preserves terminal UX, and creates a Unix socket control channel under `~/.local/share/codoxear/socks/`.
- Writes a `*.json` sidecar with: `agent_backend`, session/thread id, pid(s), cwd, log_path, sock_path, owner tag, and launch settings.
- Detects the active session log and keeps `log_path` updated by scanning the process tree for open backend log files (`~/.codex/sessions/rollout-*.jsonl` for Codex, `~/.pi/agent/sessions/*.jsonl` for Pi, `~/.claude/projects/**/*.jsonl` for Claude Code) plus backend-specific resume/discovery fallbacks.
- Ignores Codex sub-agent rollout logs (`session_meta.payload.source.subagent`) so the UI stays bound to the main session.
- Linux and macOS.

### `codoxear.sessiond`

- Headless session helper that can launch a backend session without an interactive terminal.
- Uses backend adapter-owned launch options for Codex, Pi, and Claude Code, including backend-specific model/provider/reasoning flags.
- Writes the same `socks/*.sock` + `socks/*.json` metadata shape the server expects and exposes the same control state schema (`busy`, `queue_len`, `token`, `interrupted_idle`) for readiness/token projection.
- Reuses shared terminal-query responses and backend log busy reducers where behavior should match the broker; it intentionally does not provide a foreground terminal UX.
- Linux and macOS.

### `codoxear.rollout_log` and `codoxear.pi_log`

- Shared normalization layer that turns backend-native logs into the UI’s common event/token/busy model.
- `rollout_log.py` handles chat-event extraction, delivery messages, idle detection, and token snapshots for both backends.
- `pi_log.py` contains Pi-specific helpers for session headers, assistant/user text extraction, final-turn detection, run settings, and context usage derived from Pi `usage.totalTokens` plus `~/.pi/agent/models.json`.

### UI (`codoxear/static/index.html`)

- UI shell served at `/` and `/static/index.html`, with assets under `codoxear/static/`.
- `app.js` is a 34-line boot entrypoint: it authenticates, then creates the application controller and chooses login or application rendering.
- `app_application.js` (854 lines) is the application factory. It validates shared dependencies, creates `app_application_composition.js`, and exposes its rendering surface.
- `app_application_composition.js` (1393 lines) owns `renderApp` composition: shell construction, application lifecycle, and controller assembly.
- Extracted domain modules own file work (`app_file_ops.js`), chat/transcript interaction (`app_chat_interaction.js`), and status/context/interrupt display (`app_session_display.js`).
- Wiring modules separate session transitions (`app_session_lifecycle.js`), session-list refresh (`app_session_refresh.js`), and controller option contracts (`app_wiring.js`); application composition owns cleanup-registered listeners.
- Peripheral controllers keep focused workflows out of composition: `app_message_flow.js` for confirmed-send/SSE/polling, `app_attachments.js` for staged uploads, `app_unattended.js` for unattended mode, `app_composer.js` for composer UI, plus focused transcript, queue, file-viewer, launch, search, modal, and voice modules.
- Supports creating web-owned sessions via the "New session" button with backend tabs for Codex/Pi/Claude Code; **Pi is the default** (overridable via `CODEX_WEB_DEFAULT_AGENT_BACKEND`).
- Remembers the last backend choice and last provider choice per backend in browser local storage.
- Shows backend status icons in the sidebar metadata line and backend logos in the new-session modal.
- Also uses queue, diagnostics, file-read, and git-viewer endpoints for the current UI.

## Data flow (high level)

1. Terminal: `codoxear-broker` runs the selected backend CLI and registers a control socket + metadata file.
2. Server: lists available sockets, reads metadata, and serves session content via `/api/*`.
3. Browser: `app.js` authenticates and delegates to the application factory/composition, which wires focused controllers. Those controllers select sessions, send prompts via `/api/sessions/<id>/send` or `/enqueue`, render normalized messages from the backend log over SSE with polling fallback, and read files/git state through `/api/sessions/<id>/*` helpers.

## Current frontend and runtime state

- **Frontend ownership is modular.** `app.js` is boot-only; the factory and `renderApp` composition are separate. Domain, lifecycle/refresh/wiring, and peripheral controllers own their focused state and actions, so composition passes explicit dependencies instead of recreating controller state.
- **Send path is unconditional confirmed-send.** The busy/queue gate was removed: `require_send_preconditions` only blocks on commit-unknown resolution, a pending attachment, a stale queue item, or missing broker `sync_send`. Direct sends submit regardless of busy state, so steering works on all backends. The queue remains an opt-in alternative, not a hard gate.
- **Live transcript delivery uses SSE.** `app_message_flow.js` owns confirmed-send and shared live transcript transport. Once a session is selected and bound to a backend log, it opens an `EventSource` on `/api/sessions/<id>/live` (`message_routes.py` `handle_messages_live_stream`) for real-time message deltas; HTTP polling is the automatic fallback. The SSE handler and the poll handler share the same live-delta/normalization path, so both produce identical transcript state.
- **Full-transcript search is server-side.** `app_chat_search.js` queries `/api/sessions/<id>/search`; `message_routes.py` searches the normalized backend log and returns cursored matches so selecting a match can load its surrounding history.
- **Pi subagent activity has two projections.** `pi-subagent:` events become inline narration rows, while `util.scan_active_pi_subagents()` supplies active-run counts for the sidebar `▸N` marker and the transcript's idle activity row.
- **Typing, thinking, and subagent indicators share one explicit reconciliation model.** `rollout_idle.py` accumulates Codex cumulative reasoning snapshots, Pi reasoning-token usage, and Claude Code thinking-token usage into `thinking_tokens`; `app_transcript.js` renders a token count only when it has positive authoritative token data. Live tool/thinking deltas are exact; while a turn remains open, resumable session snapshots may only raise those counters, never lower them. `subagents_running` is a separate current-liveness gauge and may decrease as workers finish; it must never be folded into the monotonic tool/thinking counters.
- **Slash completion and live controls are backend-aware.** Typing `/` shows browser-safe commands from `slash_commands.py`. Pi `/model` uses its shared TUI command; Pi `/effort` (with `/thinking` alias) is bridge-provided. Claude Code `/model` and `/effort` pickers inject its native commands. Capable Codex sessions run their TUI against a broker-owned app-server Unix socket; `/model` and `/effort` call the typed experimental `thread/settings/update` API, while unsupported Codex versions advertise neither command.
- **Markdown preserves line breaks and image geometry.** `app_markdown.js` calls `marked.parse(..., { breaks: true })` and caches rendered image dimensions in local storage.
- **Paper design language.** See the “Design language” section below for the full rule set: square geometry, warm charcoal `#2f2b26`, ink-on-paper primaries, square state dots, transparent undimmed backdrops, data in monospace, and no decorative shadows.
- **Performance.** Static asset responses are gzip-compressed (`static_routes.py`), served over `HTTP/1.1` (`server_handler.py`), versioned assets (`?v=...`) get immutable one-year cache headers, the asset version is memoized, and poll cadence is tuned via `CODEX_WEB_*_INTERVAL_SECONDS` env vars.
- **Session card DOM has two branches and must stay split.** Touch uses swipe actions; desktop uses hover-revealed actions (`useDesktopSessionActions()` / `swipeActions` flag in `app_session_helpers.js`). Do not attempt to unify them into one branch.
- **Keyboard.** Vimium-style hint mode: press `f`, then the letter over an activatable control in the current viewport. Hidden, off-screen, disabled, or hit-test-covered controls receive no hint. Direct shortcuts (no leader): `i` focus message, `j`/`k` scroll, `d`/`u` half-page, `G` go to bottom, `D` delete session, `/` full-transcript search. Typing `/` in the composer opens slash-command completion. The topbar interrupt button (`interruptBtn`) is the sole interrupt control on all viewports with hint `z`; the composer stop button was removed. On Pi sessions, `/model` uses Pi's native selector and the bridge registers `/effort` with `/thinking` as its alias; old live sessions need `/reload` or a restart before the capability is advertised. On Claude Code sessions, the `/model` and `/effort` pickers inject native commands; model rows refresh only from subsequent assistant `message.model` log evidence, while effort has no parsed CC log evidence and stays at its launch value. In open dialogs, the first distinctive letter of a visible button activates it, with a later distinctive letter breaking first-letter ties (`activateModalButtonForKey` in `app_modal.js`, bound by `app_chat_interaction.js`).
- **Codex control authority uses the TUI's app-server thread.** The broker probes experimental `thread/settings/update`; on capable Codex versions it starts a private app-server Unix socket and launches the same shared TUI with `--remote`. Browser model/effort choices go through broker control to that typed API, never through PTY picker navigation or config+signal reload. The sidebar remains `turn_context` log-authoritative and updates after the next turn records the effective settings. Unsupported versions continue with embedded TUI mode and do not advertise the controls.
- **Pi control authority goes through an extension bridge, not web-only RPC.** Codoxear wraps all backends in a PTY so web and terminal share the same session; Pi `--mode rpc` is a separate headless mode that cannot coexist with the TUI. The bridge supplies Pi effort control from within the live TUI process, while Pi `/model` remains its shared native command. See `.memory/project/ARCHITECTURE.md` “Pi integration strategy” for details.
- **Pi bridge lifecycle rule.** Extension load may write its passive capability marker and register lifecycle listeners, but it must not call runtime action methods at load. Defer command registration and runtime queries until `session_start`/`session_switch`, when Pi has a live session context; retry a failed registration from a later lifecycle event.

## State-authority principle

Every displayed state has one declared authoritative writer. When multiple feeds can update one displayed value, the owning module must document the reconciliation rule and make it explicit in code; do not add a second cache or derive the value independently. The typing row is the canonical dual-feed case: live SSE deltas are exact, session-list snapshots are resumable, and while a turn is open the snapshot can only raise counts in the shared `typingRowRuntime` store.

## Design language

The UI follows a single “paper” design language. These rules are invariants, not preferences.

- **Square, warm-charcoal paper palette.** Geometry stays square (`border-radius: 0`); `--ink` and `--border` are warm charcoal `#2f2b26`, with paper white, `#f6f5f1` background, and `#efeee9` wash.
- **Ink-on-paper primaries.** Primary actions invert warm charcoal and paper. Do not introduce an accent-blue primary.
- **Square state dots.** Session state dots are squares; fill and motion distinguish busy (filled + pulse), idle (hollow), suppressed/snoozed/blocked (filled, no pulse), and pending/starting (filled amber + pulse). Motion, not hue, is the primary discriminator. An active session is a full ink-on-paper inversion: its idle dot is a hollow paper ring, busy/suppressed dots are paper-filled, and pending stays amber.
- **Transparent, undimmed backdrops.** Dialog and sidebar backdrops are transparent: opening an overlay must not dim the application underneath.
- **Bounded overlays.** Pickers and dialogs have deliberate max dimensions and scroll only their content area; they never bleed beneath fixed chrome. Diagnostics rows may switch from two columns to a stacked label/value layout at narrow widths, without changing their visual treatment.
- **Monospace for data.** Use `--font-mono` for model names, token counts, paths, and other data-like text.
- **Spacing rhythm.** Use the `--space-1` through `--space-7` ladder (`4px, 6px, 8px, 10px, 12px, 14px, 16px`) for component padding and gaps. Keep a smaller literal only where it is a micro-control or geometry constraint; do not introduce new arbitrary spacing values.
- **Type rhythm.** Use the named size tokens `--font-2xs` (`10px`), `--font-xs` (`11px`), `--font-sm` (`12px`), `--font-md` (`13px`), `--font-lg` (`14px`), and `--font-xl` (`16px`) for UI text, with `--font-2xl` (`18px`) and `--font-3xl` (`20px`) for larger headings. The scale controls size only; `--font-mono` remains the family token for data.
- **No decorative shadows.** `box-shadow`/`outline` is allowed only for functional focus/state indication, never for depth. No `backdrop-filter`.
- **Design constitution for viewport behavior.** New viewport behavior may retune shared tokens, flip visibility, or switch layout mode. When a component needs genuinely different behavior, add an explicit sanctioned branch rather than a viewport-specific restyle.
- **Hit area is not visual size.** A 44px touch target does not require a 44px icon or chrome control: compact secondary controls use an `::after` hit area, while frequent primary composer/dialog actions may use the larger visible control.

### Media-query branching rule

The design constitution restricts new media-query behavior to retuning tokens, visibility, and layout changes. Do not use viewport-specific color, radius, font, or border restyling as a substitute for a component decision. When behavior really differs by viewport, branch the component.

The sanctioned component branches are:

- **Sidebar drawer** — fixed off-canvas on narrow viewports, static column on wide.
- **Session-card reveal mechanism** — touch uses swipe-revealed actions; desktop uses hover-revealed inline actions. This DOM split is **LOCKED**: do not unify the two branches (`useDesktopSessionActions()` / `swipeActions` flag in `app_session_helpers.js`).
- **Viewer fullscreen takeover** — file viewer takes over the viewport on small screens.
- **Hover → always-visible flips** — controls revealed on hover for fine pointers become persistently visible on coarse-pointer/touch.
- **Composer safe-area / anti-zoom** — iOS safe-area insets and input-focus anti-zoom handling.

## Development reminders

- Do not commit secrets: `.env`, `env`, keys, tokens, logs.
- Do not commit runtime artifacts: `codex-homes/`, `socks/`, `root-repo/`, `server.log`, `hmac_secret`, `__pycache__/`.
- Keep shared helpers in `codoxear/util.py` (avoid duplicating log-scan and app-dir logic across modules).
- When a subsystem is semantically wrong, replace it instead of layering more patches onto the broken structure.
- Prefer the smallest invariant-preserving model over incremental adaptation of an already confused implementation.
- Do not let internal pipeline stages redefine user-facing semantics. Define the semantic invariant first, then make the implementation mechanically preserve it.
- For queueing/streaming features, write down the exact replacement/commit boundary first (for example what counts as "queued", what counts as "playing", and what is still replaceable) before writing code.
- If the user provides a simpler design that preserves the invariant more directly, prefer that design over a more elaborate agent-invented state machine.
- For broker/server/session/tmux verification, Docker is the isolation boundary. A host-side throwaway `HOME` only redirects files; it does not isolate the process table, tmux socket, `/tmp`, signals, or systemd. Do not use host throwaway-HOME repros for broker/server/session work.
- Never use pattern-based process cleanup (`pkill -f`, `killall`, broad `pgrep | xargs kill`) in agent-run verification. If a host process is explicitly started for a non-session task, record its exact PID and clean up only that PID; prefer Docker container teardown for anything session-related.
- **Experience the product.** Unit tests and source checks verify structure, not behavior. Before claiming any user-facing feature works, exercise it through the real interface (browser, API, CLI) and observe the actual outcome.
- **Never claim completion without behavioral verification.** "Tests pass" and "code looks right" are not evidence. The evidence is: the user does the thing and the expected result appears.
- **Never swallow a user request.** Every request gets tracked, investigated, and either resolved or explicitly deferred with the user's knowledge. Silence is not a resolution.
- **When the user states a design rule, implement exactly that.** Do not add logic the user didn't ask for. Do not "improve" their rule with additional conditions.
- **When the user reports a bug, investigate their specific case.** "It works for me" is not a valid response. Reproduce their exact scenario. If you can't, say so and ask what they see.
- **Verify the interface before diagnosing the implementation.** Check the exact URL, API key names, parameter names, and response schema against the source before concluding anything is broken.
- **Test in Docker only.** Never probe, test, or verify against the live deployment (127.0.0.1:8743). All behavioral testing uses `scripts/docker_verify.sh` or Docker-isolated agent-browser sessions.
- Local dev:
  - Install: `python3 -m pip install -e .`
  - Run server: `codoxear-server` or `python3 -m codoxear.server`
  - Broker (Codex): `codoxear-broker -- <codex args>`
  - Broker (Pi): `CODEX_WEB_AGENT_BACKEND=pi codoxear-broker -- <pi args>`
  - Broker (Claude Code): `CODEX_WEB_AGENT_BACKEND=cc codoxear-broker -- <claude args>`

## Deployment: committed snapshot only

The deployed server must never import Python or static assets from `/home/yiwen/codoxear`, because concurrent edits in that checkout are development state, not release state. Deploy a reviewed commit with:

```sh
scripts/deploy.sh <commit-ish>
```

The script resolves the commit, creates or updates the detached worktree at `~/.local/share/codoxear/deploy`, verifies it reached that exact commit, runs `pipx install --force` against the snapshot, rewrites only the service's `WorkingDirectory` and `ExecStart`, then reloads and restarts **only** `codoxear-server.service`. It preserves the existing unit's `Environment=` and `EnvironmentFile=` settings, so the password/config remains external to the code snapshot and all runtime data remains under `~/.local/share/codoxear/` (`socks/`, uploads, queues, session stores, and so on).

The script refuses to reinstall or restart when the snapshot path is not this repository's clean worktree or the worktree cannot be updated to the requested commit. Its health boundary is `/` → `200` and unauthenticated `/api/sessions` → `401`.

To roll back, deploy the previous release commit:

```sh
scripts/deploy.sh <previous-commit>
```

That repoints the same snapshot worktree and service at the previous committed code; it does not move or delete live brokers, backend CLIs, sockets, uploads, or session state. Do not restart via process matching or manually point the service back at the editable checkout.

## Ops notes

- Restarting `codoxear.server` does **not** lose session content. Sessions live in backend log files on disk; the server only reads them.
- To avoid losing live sessions, **only** restart the server service. Do **not** kill `codoxear-broker` or the underlying backend CLI process.
- The supported service operation is `systemctl --user restart codoxear-server.service`; deploys must use `scripts/deploy.sh <commit-ish>` above.

## Testing policy (absolute)

- **No source-string tests, ever.** They are worse than nothing: they break on every refactor, produce false confidence without verifying behavior, and cost maintenance on every change. Do not assert on raw file contents (`read_text` + `assertIn`, regex over source, checking that code contains a literal). Tests verify BEHAVIOR: execute the code (VM harnesses for JS, direct calls for Python) and assert outcomes; for CSS, parse the stylesheet and assert computed rules per selector — never substring matching. If a contract has no behavioral check feasible, document it in AGENTS.md instead of a test.
