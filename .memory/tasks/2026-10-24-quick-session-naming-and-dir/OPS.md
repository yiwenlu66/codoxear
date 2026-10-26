# OPS — quick session naming + dedicated cwd (plan only)

Append-only evidence trail. Newest at the bottom.

## 2026-10-26T15:55Z — evidence gathered by reading the code (no execution)

Current quick-launch path:

- `codoxear/static/app_quick_session.js:77` — `spawnSession("~", null, null, "", …)`;
  the label/aria copy is `Quick session — ~ · <backend> · <model> · <effort>`,
  pending copy `Starting … in ~…`; failure path `openDialog({cwd: "~", …})`.
  The comment above `start()` states all of it must stay inside the click
  gesture (iOS keyboard), so any new await must come after `textarea.focus()`.
- `codoxear/static/app_session_lifecycle.js:337` — `spawnSessionWithCwd` posts
  `{cwd, agent_backend, …}` to `/api/sessions`; when `sessionName` is non-empty
  it renames the launched session after the broker-pid row appears
  (`POST /api/sessions/<id>/rename`). Index map: 0 cwd, 1 resume, 2 worktree,
  3 sessionName, 11 selectLaunchedSession.
- `codoxear/launch_path_runtime.py:31,47` — `"~"` is expanded by the server;
  `resolve_dir_target` resolves lexically and rejects a non-dir path.
- `codoxear/session_launch_plan.py:95` — a missing cwd is created by the launch
  path (`mkdir(parents=True, exist_ok=True)`), so an allocator never needs to
  create directories.
- `codoxear/session_launch_plan.py:112` — `codex_trust_override_for_path(spawn_cwd)`
  is computed from the actual spawn cwd, so a brand-new directory is trusted.
- `codoxear/agent_backend/cc.py:231` — CC is launched with
  `--dangerously-skip-permissions`; Pi always gets `--approve`
  (`broker_launch.py:142-156`). No backend prompts for trust in a fresh dir.
- `codoxear/session_display.py:18` — display name = trimmed alias, else cwd
  basename; `session_listing.py:293-294` puts it on the row as `display_name`;
  `app_display.js:75-96` renders it and falls back locally. The tmux mirror
  titles windows with the same value (AGENTS.md).
- `codoxear/control_routes.py:122` — delete is `POST /api/sessions/<id>/delete`,
  response `{ok}`; client call sites `app_sessions.js:219`,
  `app_message_history.js:360`, `app_transcript_render.js:332`.
- `codoxear/session_routes.py:85-97` — `handle_session_post_route` only owns
  `path == "/api/sessions"`; new POST paths are added there.
- `codoxear/server_config.py:273-292` — app-dir state files are declared as
  `ServerConfig` paths; `RECENT_CWD_PATH` among them.
- `codoxear/session_store.py` — `SessionStorePaths` + `recent_cwd_max`,
  `note_recent_cwd`/`list_recent_cwds`; `session_discovery_registry.py:60`
  remembers the cwd of every discovered session (the recent-cwd pollution path).
- `codoxear/static/app_launch.js:24-56` — `loadLastLaunch`/`rememberLastLaunch`
  (`codoxear.lastLaunch`) is what the quick gesture uses; unaffected.
- `scripts/docker_quick_session_verify.sh` asserts `s['cwd']=='/home/tester'`,
  `b['cwd']=='~'`, and failure-dialog `f['cwd']=='~'` — the three assertions
  that pin today's behavior.
- `tests/test_quick_session.py:107-118` — VM harness stubs `spawnSession(...args)`
  and asserts `args[0]=='~'` plus the exact pending copy string.
- `codoxear/static/app_wiring.js:28` — `createQuickSessionOptions` select list
  (a new seam must be added there and in the composition call site
  `app_application_composition.js:1067`).

Design conclusions and the file-by-file plan live in `QUICK_SESSION_PLAN.md`.
