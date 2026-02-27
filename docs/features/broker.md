# Broker

The broker wraps the target CLI (`codex` or `claude`) in a PTY, exposes a Unix socket for control, and watches session logs to track busy/idle state.

## Launch and PTY lifecycle
How users use it:
Run `codoxear-broker -- <cli args>` (set `CODEX_WEB_CLI=codex|claude`) or wrap your shell CLI functions to call the broker.

Effect:
Spawns the selected CLI in a PTY, keeps terminal UX intact, and starts a control socket under `~/.local/share/codoxear/socks/`. The initial socket name is `broker-<pid>.sock`, and for terminal-owned sessions the broker may also create `<session_id>-<pid>.sock` after log discovery.

Files:
- `codoxear/broker.py`
- `codoxear/pty_util.py`

Key flow:
1. Fork a PTY and exec the selected CLI binary (or a login shell for web-owned sessions).
2. Set terminal size, raw mode, and handle terminal query replies.
3. Start socket server, PTY reader, log watcher, and log discovery threads.

Call stack:
1. `Broker.run`
2. `_pty_to_stdout`
3. `_stdin_to_pty`
4. `_sock_server`
5. `_discover_log_watcher`
6. `_log_watcher`

Notes:
- Linux-only due to `/proc`, `pty`, and `termios`.
- `CODEX_WEB_EMULATE_TERMINAL=1` forces terminal query emulation when no TTY is attached.
- `CODEX_WEB_TMUX_INTERACTIVE=1` allows stdin passthrough for web-launched sessions running inside tmux.
- When `CODEX_WEB_UNSET_ANTHROPIC_AUTH_TOKEN=1`, headless login-shell launches unset `ANTHROPIC_AUTH_TOKEN` right before exec so shell rc exports do not reintroduce auth-token mode.

## Socket protocol
How users use it:
The server connects to the broker socket and issues JSON commands.

Effect:
Allows state queries and text injection into the PTY.

Files:
- `codoxear/broker.py`
- `codoxear/server.py`

Commands:
- `state`: returns `busy`, `queue_len`, and `token`.
- `send`: injects text plus the configured enter sequence.
- `queue`: get or update the server-side queue (`op=get|set|push`).
- `keys`: injects raw key sequences.
- `tail`: returns the PTY output tail.
- `shutdown`: terminates the CLI process group.

Call stack:
1. `ServerManager._sock_call`
2. `Broker._handle_conn`

Notes:
- `send` injects directly; queued messages use the `queue` command instead.
- Queue items are gate-tagged and released one at a time after their required turn-end marker.
- Idle fallback can synthesize a turn end when a queued item is pending, the turn has a completion candidate, and output remains quiet.
- Quiet idle fallback is queue-only; without queued items, busy/idle transitions rely on explicit turn-end markers.
- Enter sequence is configurable with `CODEX_WEB_ENTER_SEQ`.

## Rollout log discovery and busy/idle state
How users use it:
The broker finds the active log file to align the UI with the underlying CLI thread/session.

Effect:
Updates `log_path` in metadata and uses rollout events to track busy state and turn boundaries.

Files:
- `codoxear/broker.py`
- `codoxear/util.py`

Key flow:
1. Discover the active log after startup:
   - Codex: `~/.codex/sessions/**/rollout-*.jsonl`
   - Claude: `~/.claude/projects/**/*.jsonl` (project UUID logs, excluding `subagents`)
2. Prefer `/proc` open-fd discovery and fall back to recent Claude project logs by `cwd`/mtime when Claude does not keep the JSONL fd open.
3. Register `session_id` and write metadata JSON beside the socket.
4. Tail the log and update busy/idle heuristics.

Call stack:
1. `_discover_log_watcher`
2. `_proc_find_open_rollout_log` / `_find_recent_claude_project_log`
3. `_register_from_log`
4. `_log_watcher`
5. `_apply_rollout_obj_to_state`

Notes:
- When a new `/new` thread hint is detected, the broker clears the current log binding and re-discovers.
- `CODEX_WEB_BUSY_QUIET_SECONDS` and `CODEX_WEB_BUSY_INTERRUPT_GRACE_SECONDS` tune busy clearing heuristics.
- `CODEX_WEB_IDLE_TURN_END_QUIET_SECONDS` controls the quiet window for idle turn-end fallback used by queue release.
- Assistant phase `commentary` does not count as a completion candidate; `final_answer` (or messages without phase) does.
- Claude queue drain also triggers on `system.subtype=turn_duration|api_error`.

## Metadata sidecar
How users use it:
The server reads `*.json` files next to each socket.

Effect:
Provides `session_id`, `pid`s, `cwd`, `log_path`, `sock_path`, `owner`, and `cli`.

Files:
- `codoxear/broker.py`

Key data:
- Path: `~/.local/share/codoxear/socks/<socket>.json` (sidecar matches the socket filename).
- Socket names: `broker-<pid>.sock` on startup and optionally `<session_id>-<pid>.sock` after log discovery.
- Fields: `session_id`, `owner`, `cli`, `broker_pid`, `codex_pid`, `cwd`, `start_ts`, `log_path`, `sock_path`, `tmux_name` (when launched under tmux)
