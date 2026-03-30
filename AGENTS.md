# Codoxear architecture notes

This repo is a Linux-first companion UI for continuing local CLI agent sessions on a phone/laptop browser.

Currently supported agent backends:

- `codex`
- `pi`

## Components

### `codoxear.server`

- HTTP server (single process) that serves the UI and a small JSON API under `/api/*`.
- Auth: password gate using `CODEX_WEB_PASSWORD` (required). Cookie-based session (`codoxear_auth`).
- Session discovery: scans `~/.local/share/codoxear/socks/*.sock` for broker control sockets and reads the adjacent `*.json` metadata.
- Web-owned sessions: `/api/sessions` (POST) spawns a new broker process with `CODEX_WEB_OWNER=web` and a chosen `agent_backend`.
- Terminal-owned sessions: created by running `codoxear-broker` with the desired backend environment (for example plain Codex broker wrappers or `CODEX_WEB_AGENT_BACKEND=pi` for Pi).
- `GET /api/sessions` returns backend-aware launch defaults, including provider/model/reasoning choices per backend.
- Runtime state directory: `~/.local/share/codoxear` (legacy `~/.local/share/codex-web` is no longer used).
- Additional persisted UI state includes `session_sidebar.json`, `session_files.json`, `session_queues.json`, `harness.json`, and `session_aliases.json` under the same app dir.

### `codoxear.broker`

- Foreground PTY wrapper intended to be run from a real terminal.
- Starts the selected backend CLI (`codex` or `pi`), preserves terminal UX, and creates a Unix socket control channel under `~/.local/share/codoxear/socks/`.
- Writes a `*.json` sidecar with: `agent_backend`, session/thread id, pid(s), cwd, log_path, sock_path, owner tag, and launch settings.
- Detects the active session log and keeps `log_path` updated by scanning the process tree for open backend log files (`~/.codex/sessions/rollout-*.jsonl` for Codex, `~/.pi/agent/sessions/*.jsonl` for Pi) plus backend-specific resume/discovery fallbacks.
- Ignores Codex sub-agent rollout logs (`session_meta.payload.source.subagent`) so the UI stays bound to the main session.
- Linux and macOS.

### `codoxear.sessiond`

- Headless session helper that can launch a backend session without an interactive terminal.
- Writes the same `socks/*.sock` + `socks/*.json` metadata the server expects.
- Linux and macOS.

### `codoxear.rollout_log` and `codoxear.pi_log`

- Shared normalization layer that turns backend-native logs into the UI’s common event/token/busy model.
- `rollout_log.py` handles chat-event extraction, delivery messages, idle detection, and token snapshots for both backends.
- `pi_log.py` contains Pi-specific helpers for session headers, assistant/user text extraction, final-turn detection, run settings, and context usage derived from Pi `usage.totalTokens` plus `~/.pi/agent/models.json`.

### UI (`codoxear/static/index.html`)

- UI shell served at `/` and `/static/index.html`, with assets under `codoxear/static/` (`app.css`, `app.js`).
- Polls `/api/sessions` and `/api/sessions/<id>/messages`.
- Supports creating web-owned sessions via the "New session" button with backend tabs for Codex/Pi.
- Remembers the last backend choice and last provider choice per backend in browser local storage.
- Shows backend status icons in the sidebar metadata line and backend logos in the new-session modal.
- Also uses queue, diagnostics, file-read, and git-viewer endpoints for the current UI.

## Data flow (high level)

1. Terminal: `codoxear-broker` runs the selected backend CLI and registers a control socket + metadata file.
2. Server: lists available sockets, reads metadata, and serves session content via `/api/*`.
3. Browser: selects a session, sends prompts via `/api/sessions/<id>/send` or `/enqueue`, renders normalized messages from the backend log, and reads files/git state through `/api/sessions/<id>/*` helpers.

## Development reminders

- Do not commit secrets: `.env`, `env`, keys, tokens, logs.
- Do not commit runtime artifacts: `codex-homes/`, `socks/`, `root-repo/`, `server.log`, `hmac_secret`, `__pycache__/`.
- Keep shared helpers in `codoxear/util.py` (avoid duplicating log-scan and app-dir logic across modules).
- When a subsystem is semantically wrong, replace it instead of layering more patches onto the broken structure.
- Prefer the smallest invariant-preserving model over incremental adaptation of an already confused implementation.
- Do not let internal pipeline stages redefine user-facing semantics. Define the semantic invariant first, then make the implementation mechanically preserve it.
- For queueing/streaming features, write down the exact replacement/commit boundary first (for example what counts as "queued", what counts as "playing", and what is still replaceable) before writing code.
- If the user provides a simpler design that preserves the invariant more directly, prefer that design over a more elaborate agent-invented state machine.
- Local dev:
  - Install: `python3 -m pip install -e .`
  - Run server: `codoxear-server` or `python3 -m codoxear.server`
  - Broker (Codex): `codoxear-broker -- <codex args>`
  - Broker (Pi): `CODEX_WEB_AGENT_BACKEND=pi codoxear-broker -- <pi args>`

## Ops notes

- Restarting `codoxear.server` does **not** lose session content. Sessions live in backend log files on disk; the server only reads them.
- To avoid losing live sessions, **only** stop the server process. Do **not** kill `codoxear-broker` or the underlying backend CLI process.
- Safe restart example (server only):
  - `pgrep -f "python3 -m codoxear.server" | xargs -r kill`
  - `CODEX_WEB_PASSWORD=... CODEX_WEB_PORT=13780 CODEX_WEB_HOST=0.0.0.0 nohup python3 -m codoxear.server >/tmp/codoxear-13780.log 2>&1 &`
