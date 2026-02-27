# Server and API

The server is a single-process HTTP service that serves the UI and a JSON API under `/api/*`. It owns session discovery, authentication, and the server-side queue API.

## Auth and cookies
How users use it:
Set `CODEX_WEB_PASSWORD` and log in from the browser UI.

Effect:
Sets a signed cookie (`codoxear_auth`) with an expiry. All `/api/*` routes require this cookie.

Files:
- `codoxear/server.py`
- `codoxear/static/app.js`

Key flow:
1. Load `.env` from the current working directory and read `CODEX_WEB_PASSWORD`.
2. Handle `POST /api/login` and verify password hash.
3. Sign and set the auth cookie with HMAC.
4. Enforce auth on all `/api/*` requests.

Call stack:
1. `ServerHandler.do_POST`
2. `POST /api/login` block
3. `_is_same_password`
4. `_set_auth_cookie`
5. `_require_auth` for subsequent requests

Notes:
- Cookie signing uses `~/.local/share/codoxear/hmac_secret`.
- `CODEX_WEB_COOKIE_SECURE=1` or `X-Forwarded-Proto: https` adds the `Secure` cookie attribute.
- `CODEX_WEB_URL_PREFIX` scopes routing and cookie path.
- JSON API responses are served with `Cache-Control: no-store` to prevent stale session data.

## Session discovery and lifecycle
How users use it:
Start Codex/Claude via `codoxear-broker` (terminal-owned) or create a web-owned session via `POST /api/sessions`.

Effect:
The server discovers `*.sock` control sockets under `~/.local/share/codoxear/socks/`, reads their metadata (including `cli`), and exposes them in `/api/sessions`.

Files:
- `codoxear/server.py`
- `codoxear/util.py`
- `codoxear/broker.py`

Key flow:
1. Scan `socks/*.sock` and load `*.json` metadata.
2. Validate broker and Codex PIDs, prune stale sockets.
3. Call the socket to fetch `busy` and `queue_len` state.
4. Update `SessionManager` cache.

Call stack:
1. `SessionManager._discover_existing`
2. `SessionManager._sock_call`
3. `SessionManager.list_sessions`

Notes:
- Web-owned sessions are spawned with `CODEX_WEB_OWNER=web`; `POST /api/sessions` accepts optional `cli` (`codex` or `claude`).
- Spawned brokers receive `CODEX_WEB_CLI=<cli>` and matching home/bin env (`CODEX_HOME/CODEX_BIN` or `CLAUDE_HOME/CLAUDE_BIN`).
- Web-owned sessions can be started under tmux when `CODEX_WEB_TMUX=1`; session listings include `tmux_name` when available.
- When tmux is enabled, `CODEX_WEB_TMUX_INTERACTIVE=1` allows attaching to the tmux session and sending input.
- Terminal-owned sessions are attach-only; the UI hides delete for them.
- Session listings include `cli` and `last_assistant_ts` to help the UI render resume commands and unread response indicators.

## Messaging API
How users use it:
The UI polls `/api/sessions/<id>/messages` and submits prompts with `/api/sessions/<id>/send`.

Effect:
The server tails Codex rollout logs or Claude project logs, extracts chat events, and sends input to the broker socket.

Files:
- `codoxear/server.py`
- `codoxear/rollout_log.py`
- `codoxear/util.py`
- `codoxear/static/app.js`

Key flow:
1. UI calls `/api/sessions/<id>/messages` with `offset` and optional `init=1`.
2. Server reads JSONL chunks and extracts chat events.
3. Server computes `busy` using broker state and log-derived idle heuristics.
4. UI calls `/api/sessions/<id>/send` to inject text into the PTY.

Call stack:
1. `ServerHandler.do_GET` for `/messages`
2. `SessionManager.refresh_session_meta`
3. `_read_jsonl_from_offset` and `_extract_chat_events`
4. `SessionManager.get_state`
5. `ServerHandler.do_POST` for `/send`
6. `SessionManager.send`
7. `SessionManager._sock_call`

Notes:
- `/messages` supports `init=1` for chat index seeding and `before` for older history paging.
- `/tail`, `/interrupt`, and `/inject_image` are UI helpers for debug, interruption, and image attach.
- `/tail` strips ANSI/control sequences so the live tail stays readable and decodes PTY output incrementally to avoid splitting multibyte characters.

## Server-side queue
How users use it:
Use "Send after current" or the queue viewer to enqueue follow-up messages.

Effect:
Queued messages are stored in the broker and released one at a time after a turn ends (or via the idle fallback), so they continue even if the browser closes.

Files:
- `codoxear/server.py`
- `codoxear/broker.py`
- `codoxear/static/app.js`

Key flow:
1. UI calls `/api/sessions/<id>/queue` to append or replace the queue.
2. Server forwards the request to the broker socket.
3. Broker drains one queued message on turn-end markers or after the busy heuristic clears:
   - Codex: `task_complete` / `turn_aborted`
   - Claude: `system.subtype=turn_duration|api_error`

Call stack:
1. `ServerHandler.do_GET`/`do_POST` for `/queue`
2. `SessionManager.queue_get` / `queue_set` / `queue_push`
3. `SessionManager._sock_call`
4. `Broker._handle_conn` (`cmd=queue`)
5. `Broker._log_watcher`

Notes:
- Queue length is surfaced in `/api/sessions` and `/api/sessions/<id>/messages`.
- The queue list is fetched on demand via `/api/sessions/<id>/queue`.
 - Tests: `python3 -m unittest tests.test_server_queue`.

## File API
How users use it:
The UI reads and edits files from the file viewer.

Effect:
`/api/files/read` returns file contents and `/api/files/write` saves edits.

Files:
- `codoxear/server.py`
- `codoxear/static/app.js`

Notes:
- File writes only target existing files and are capped by `CODEX_WEB_FILE_WRITE_MAX_BYTES`.
- File history can be pruned via `/api/files/remove` or cleared per workspace via `/api/files/clear`; both accept an optional `cwd` to target a workspace directly.
- `/api/files/remove` accepts `scope=all` to remove a path from every stored history list as a fallback.
- `/api/files/read` accepts an optional `record_history=false` to avoid re-adding a file during background refresh.

## Static assets and URL prefix
How users use it:
Open `http://<host>:8743/` or `/<prefix>/` when `CODEX_WEB_URL_PREFIX` is set.

Effect:
Serves `index.html`, `app.js`, and `app.css` plus static assets under `/static/`.

Files:
- `codoxear/server.py`
- `codoxear/static/*`

Notes:
- Prefix logic is enforced for API and assets, including cookie path scoping.
- `CODEX_WEB_HOST` and `CODEX_WEB_PORT` control the bind address and port.
