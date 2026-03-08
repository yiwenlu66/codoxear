# Codoxear architecture notes

This repo is a Linux-first companion UI for continuing Codex CLI TUI sessions on a phone/laptop browser.

## Components

### `codoxear.server`

- HTTP server (single process) that serves the UI and a small JSON API under `/api/*`.
- Auth: password gate using `CODEX_WEB_PASSWORD` (required). Cookie-based session (`codoxear_auth`).
- Session discovery: scans `~/.local/share/codoxear/socks/*.sock` for broker control sockets and reads the adjacent `*.json` metadata.
- Web-owned sessions: `/api/sessions` (POST) spawns a new broker process with `CODEX_WEB_OWNER=web`. These sessions show a delete button in the UI and can be killed by the server.
- Terminal-owned sessions: created by running `codoxear-broker` (usually via a shell wrapper for `codex`). The current UI can also delete these sessions, which sends broker shutdown to the corresponding terminal session.
- Runtime state directory: `~/.local/share/codoxear` (legacy `~/.local/share/codex-web` is no longer used).
- Additional persisted UI state includes `session_sidebar.json`, `session_files.json`, `session_queues.json`, `harness.json`, and `session_aliases.json` under the same app dir.

### `codoxear.broker`

- Foreground PTY wrapper intended to be run from a real terminal.
- Starts Codex CLI, preserves terminal UX, and creates a Unix socket control channel under `~/.local/share/codoxear/socks/`.
- Writes a `*.json` sidecar with: session_id, pid(s), cwd, log_path, sock_path, owner tag.
- Detects the active rollout log and keeps `log_path` updated by scanning the Codex process tree for open `rollout-*.jsonl` file descriptors (`/proc` on Linux, `lsof`/`pgrep` on macOS).
- Ignores sub-agent rollout logs (`session_meta.payload.source.subagent`) so the UI stays bound to the main session.
- Linux and macOS.

### `codoxear.sessiond`

- Headless session helper that can launch a Codex session without an interactive terminal.
- Writes the same `socks/*.sock` + `socks/*.json` metadata the server expects.
- Linux and macOS.

### UI (`codoxear/static/index.html`)

- UI shell served at `/` and `/static/index.html`, with assets under `codoxear/static/` (`app.css`, `app.js`).
- Polls `/api/sessions` and `/api/sessions/<id>/messages`.
- Supports creating web-owned sessions via the "New session" button.
- Also uses queue, diagnostics, file-read, and git-viewer endpoints for the current UI.

## Data flow (high level)

1. Terminal: `codoxear-broker` runs Codex and registers a control socket + metadata file.
2. Server: lists available sockets, reads metadata, and serves session content via `/api/*`.
3. Browser: selects a session, sends prompts via `/api/sessions/<id>/send` or `/enqueue`, renders messages from the rollout log, and reads files/git state through `/api/sessions/<id>/*` helpers.

## Development reminders

- Do not commit secrets: `.env`, `env`, keys, tokens, logs.
- Do not commit runtime artifacts: `codex-homes/`, `socks/`, `root-repo/`, `server.log`, `hmac_secret`, `__pycache__/`.
- Keep shared helpers in `codoxear/util.py` (avoid duplicating log-scan and app-dir logic across modules).
- Local dev:
  - Install: `python3 -m pip install -e .`
  - Run server: `codoxear-server` or `python3 -m codoxear.server`
  - Broker: `codoxear-broker -- <codex args>`

## Ops notes

- Restarting `codoxear.server` does **not** lose session content. Sessions live in Codex rollout logs on disk; the server only reads them.
- To avoid losing live sessions, **only** stop the server process. Do **not** kill `codoxear-broker` or the Codex process.
- Safe restart example (server only):
  - `pgrep -f "python3 -m codoxear.server" | xargs -r kill`
  - `CODEX_WEB_PASSWORD=... CODEX_WEB_PORT=13780 CODEX_WEB_HOST=0.0.0.0 nohup python3 -m codoxear.server >/tmp/codoxear-13780.log 2>&1 &`
