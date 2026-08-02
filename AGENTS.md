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

- UI shell served at `/` and `/static/index.html`, with assets under `codoxear/static/` (`app.css`, `app.js`).
- Polls `/api/sessions` and `/api/sessions/<id>/messages`.
- Supports creating web-owned sessions via the "New session" button with backend tabs for Codex/Pi/Claude Code; **Pi is the default** (overridable via `CODEX_WEB_DEFAULT_AGENT_BACKEND`).
- Remembers the last backend choice and last provider choice per backend in browser local storage.
- Shows backend status icons in the sidebar metadata line and backend logos in the new-session modal.
- Also uses queue, diagnostics, file-read, and git-viewer endpoints for the current UI.

## Data flow (high level)

1. Terminal: `codoxear-broker` runs the selected backend CLI and registers a control socket + metadata file.
2. Server: lists available sockets, reads metadata, and serves session content via `/api/*`.
3. Browser: selects a session, sends prompts via `/api/sessions/<id>/send` or `/enqueue`, renders normalized messages from the backend log, and reads files/git state through `/api/sessions/<id>/*` helpers.

## Current frontend and runtime state

- **Send path is unconditional confirmed-send.** The busy/queue gate was removed: `require_send_preconditions` only blocks on commit-unknown resolution, a pending attachment, a stale queue item, or missing broker `sync_send`. Direct sends submit regardless of busy state, so steering works on all backends. The queue remains an opt-in alternative, not a hard gate.
- **Live transcript delivery uses SSE.** Once a session is selected and bound to a backend log, the browser opens an `EventSource` on `/api/sessions/<id>/live` (`message_routes.py` `handle_messages_live_stream`) for real-time message deltas; HTTP polling is the automatic fallback. The SSE handler and the poll handler share the same live-delta/normalization path, so both produce identical transcript state.
- **Pi subagent activity is visible in the transcript.** Pi `pi-subagent:` custom events are normalized into inline narration rows by `agent_backend.py` (`_pi_subagent_*` helpers): background-task progress, control notices, and results render as assistant-side rows so the user sees delegation without the terminal.
- **Typing indicator shows tool/thinking counts.** `app_transcript.js` `typingRowRuntime` tracks live `{ thinking, tools }` deltas from SSE/session-list payloads and renders `tools: N · thinking: N` in the busy typing row. Counts are monotonic within a turn (the snapshot feed may only raise the count, never lower it, recovering from an SSE gap) and reset when a new turn opens.
- **Markdown rendering uses the `marked` library**, loaded from `cdn.jsdelivr.net` (`index.html`); the CSP (`script-src`/`style-src`) was updated to allow it. Codoxear post-processors run after `marked`: file-reference rewriting (`app_markdown.js`), KaTeX math, and OAI memory-citation rewriting.
- **Paper design language.** See the “Design language” section below for the full rule set. In short: zero border radius, ink `#141111`/paper/wash palette, inversion (ink-on-paper) primaries with no accent blue, square state dots, monospace for data, no translucent colors, no decorative shadows.
- **Performance.** Static asset responses are gzip-compressed (`static_routes.py`), served over `HTTP/1.1` (`server_handler.py`), versioned assets (`?v=...`) get immutable one-year cache headers, the asset version is memoized, and poll cadence is tuned via `CODEX_WEB_*_INTERVAL_SECONDS` env vars.
- **Session card DOM has two branches and must stay split.** Touch uses swipe actions; desktop uses hover-revealed actions (`useDesktopSessionActions()` / `swipeActions` flag in `app_session_helpers.js`). Do not attempt to unify them into one branch.
- **Keyboard.** Vimium-style hint mode: press `f`, then the letter over any visible control to activate it. Direct shortcuts (no leader): `i` focus message, `j`/`k` scroll, `d`/`u` half-page, `G` go to bottom, `D` delete session, `/` search. The topbar interrupt button (`interruptBtn`) is the sole interrupt control on all viewports with hint `z`; the composer stop button was removed. On Pi sessions, `/model` and `/thinking` in the composer switch models and reasoning levels live (the `/thinking` command is registered by the `pi_active_session_bridge.ts` extension; existing sessions need `/reload` or a restart to load the updated bridge). In open dialogs, the first distinctive letter of a visible button activates it, with a later distinctive letter breaking first-letter ties (`activateModalButtonForKey` in `app.js`).
- **Pi control authority goes through an extension bridge, not web-only RPC.** Codoxear wraps all backends in a PTY so web and terminal share the same session; Pi `--mode rpc` is a separate headless mode that cannot coexist with the TUI. Authoritative model/thinking control for shared sessions requires expanding the `pi_active_session_bridge.ts` extension side-channel, not a web-owned RPC adapter. See `.memory/project/ARCHITECTURE.md` “Pi integration strategy” for details.

## Design language

The UI follows a single “paper” design language. These rules are invariants, not preferences.

- **Zero border radius everywhere.** No rounded corners on any element.
- **Ink/paper/wash palette.** `--ink: #141111`, `--paper: #ffffff`, `--bg: #f6f5f1`, `--wash: #efeee9`, `--hairline: #dcdad4`. Borders are ink (`--border: #141111`), not a neutral gray.
- **Inversion primaries, no accent blue.** Primary actions are ink-on-paper inversion (`background: var(--ink); color: var(--paper)`). Do not introduce a blue/accent primary.
- **Square state dots.** Session state dots are squares; fill and motion distinguish busy (filled + pulse), idle (hollow), suppressed/snoozed/blocked (filled, no pulse), and pending/starting (filled amber + pulse). Motion, not hue, is the primary discriminator. An active session is a full ink-on-paper inversion: its idle dot is a hollow paper ring, busy/suppressed dots are paper-filled, and pending stays amber.
- **Bounded overlays.** Pickers and dialogs have deliberate max dimensions and scroll only their content area; they never bleed beneath fixed chrome. Diagnostics rows may switch from two columns to a stacked label/value layout at narrow widths, without changing their visual treatment.
- **Monospace for data.** Use `--font-mono` for model names, token counts, paths, and other data-like text.
- **No translucent colors.** No `rgba`/`hsla` with alpha for fills or borders. Backdrops are solid.
- **No decorative shadows.** `box-shadow`/`outline` is allowed only for functional focus/state indication, never for depth. No `backdrop-filter`.
- **Chrome controls are compact with touch hit-slop.** Topbar/sidebar/nav chrome buttons are 32px (`--ctl-chrome`) on all viewports, with a 44px touch hit-slop via an `::after` inset pseudo-element. Composer and dialog controls use the `--ctl` token (38px desktop, 44px on touch).

### Media-query branching rule

Media queries may only **retune tokens** (e.g. `--ctl`, `--sidebar-w`), **flip visibility** (show/hide a branch), or **switch layout mode** (grid columns, fixed/flow). They must never restyle a component (no per-viewport color, radius, font, or border changes on an existing component). When viewport-specific behavior is genuinely component-level, branch the component instead.

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
- Local dev:
  - Install: `python3 -m pip install -e .`
  - Run server: `codoxear-server` or `python3 -m codoxear.server`
  - Broker (Codex): `codoxear-broker -- <codex args>`
  - Broker (Pi): `CODEX_WEB_AGENT_BACKEND=pi codoxear-broker -- <pi args>`
  - Broker (Claude Code): `CODEX_WEB_AGENT_BACKEND=cc codoxear-broker -- <claude args>`

## Ops notes

- Restarting `codoxear.server` does **not** lose session content. Sessions live in backend log files on disk; the server only reads them.
- To avoid losing live sessions, **only** stop the server process. Do **not** kill `codoxear-broker` or the underlying backend CLI process.
- Safe restart example (server only):
  - `pgrep -f "python3 -m codoxear.server" | xargs -r kill`
  - `CODEX_WEB_PASSWORD=... CODEX_WEB_PORT=13780 CODEX_WEB_HOST=0.0.0.0 nohup python3 -m codoxear.server >/tmp/codoxear-13780.log 2>&1 &`
