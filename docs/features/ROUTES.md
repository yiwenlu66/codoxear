# Routes

All API routes require the `codoxear_auth` cookie unless noted.

## Auth
- `POST /api/login`  Body: `{ "password": "..." }`  Sets auth cookie.
- `POST /api/logout`  Clears auth cookie.
- `GET /api/me`  Returns `{ ok: true }` if authenticated.

## Sessions
- `GET /api/sessions`  Lists session metadata and status (includes `cli`).
- `POST /api/sessions`  Body: `{ "cwd": "...", "args": ["..."], "cli": "codex|claude|gemini" }`  Spawns a web-owned session.
- `POST /api/sessions/<id>/delete`  Deletes a web-owned session.

## Messaging
- `GET /api/sessions/<id>/messages`  Query: `offset`, `init=1`, `before`, `limit`.
- `GET /api/sessions/<id>/tail`  Returns the PTY tail buffer.
- `POST /api/sessions/<id>/send`  Body: `{ "text": "..." }`.
- `POST /api/sessions/<id>/interrupt`  Injects an ESC sequence.
- `POST /api/sessions/<id>/inject_image`  Body: `{ "filename": "...", "data_b64": "..." }`.

## Queue
- `GET /api/sessions/<id>/queue`  Returns `{ "queue": ["..."], "queue_len": n }`.
- `POST /api/sessions/<id>/queue`  Body: `{ "text": "...", "front": true|false }` to append or `{ "queue": ["..."] }` to replace.

## Files
- `POST /api/files/read`  Body: `{ "path": "...", "session_id": "..." }`  Reads a text file.
- `POST /api/files/write`  Body: `{ "path": "...", "text": "...", "session_id": "..." }`  Writes a text file (existing file only).
- `POST /api/files/remove`  Body: `{ "path": "...", "session_id": "..." }`  Removes a file from workspace history.
- `POST /api/files/clear`  Body: `{ "session_id": "..." }`  Clears workspace file history.

## Metrics and hooks
- `GET /api/metrics`  Returns timing metrics for API endpoints.
- `GET /api/update`  Returns cached GitHub update status (`update_available`, commits, remote/branch, optional compare URL).
- `POST /api/hooks/notify`  Minimal webhook endpoint for future extensions.

## Notes
- When `CODEX_WEB_URL_PREFIX` is set, the UI and API are served under that prefix.
- Cookie path is scoped to the prefix.
