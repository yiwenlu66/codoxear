# Session: 2026-02-27 Claude Log Bind and Pending Fix

## Focus
Fix two regressions after Claude support rollout:
- Claude web sessions accepted input but never produced chat events in `/messages`.
- Pending user bubbles disappeared after switching sessions.

## Requests
- Restore Claude replies in web-created Claude sessions.
- Keep unsynced local user echoes visible across session switches.

## Actions Taken
- Investigated live daemon behavior on port `13780` and confirmed Claude replied in tmux UI but server never bound `log_path`.
- Fixed Claude auth-mode conflict handling in web spawn:
  - Prefer API-key mode when both auth token and API key are present.
  - Explicitly unset `ANTHROPIC_AUTH_TOKEN` for child sessions (including tmux global environment inheritance).
  - Added `CODEX_WEB_UNSET_ANTHROPIC_AUTH_TOKEN=1` marker for broker-side enforcement.
- Hardened broker launch for headless login-shell sessions:
  - Drop `ANTHROPIC_AUTH_TOKEN` from process env when instructed.
  - Prefix login-shell command with `unset ANTHROPIC_AUTH_TOKEN;` so shell rc exports do not reintroduce token mode.
- Fixed Claude log discovery when `/proc` fd discovery misses the project JSONL:
  - Added fallback scan for recent Claude project logs by `cwd` and mtime (`_find_recent_claude_project_log`).
  - Wired fallback into `_discover_log_watcher`.
- Updated UI pending-message model to be session-scoped instead of global:
  - Keep pending user echoes per session.
  - Re-render pending entries when returning to a session.
  - Clear pending store only when the session is removed/deleted.
- Added/updated tests for spawn env handling, broker login-shell unset behavior, and Claude log fallback discovery.

## Outcomes
- Claude web sessions now bind `log_path` and stream assistant events to `/messages`.
- Claude auth conflict banner caused by dual env vars no longer blocks web sessions on this host.
- Session switching no longer drops pending local user bubbles before server reconciliation.

## Tests
- `python3 -m unittest tests.test_server_spawn_cli`
- `python3 -m unittest tests.test_broker_spawn_env`
- `python3 -m unittest tests.test_broker_proc_rollout`
- `python3 -m unittest discover -s tests`

## Manual Verification
- Restarted daemon: `supervisorctl restart codoxear`.
- Live API check on `http://127.0.0.1:13780`:
  - `POST /api/sessions` with `cli=claude`
  - `POST /api/sessions/<id>/send`
  - Verified `/api/sessions/<id>/messages` returned bound `log_path` and assistant reply (`OK`).
