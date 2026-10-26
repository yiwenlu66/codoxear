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

## 2026-10-26T16:08Z — user decisions, applied to the plan

User: "no env/knob for this; just hardcode default. what if overflows 99?"

- Removed `CODEX_WEB_QUICK_DIR` from the plan. Root is now the hardcoded
  `Path.home() / "codoxear-quick"`, declared once in `server_config.py` beside
  `PROC_ROOT` (never read from the environment). Relocation mechanism becomes a
  symlink at that path, which forces `is_quick_cwd` to resolve both sides
  (resolved cwd vs `quick_root().resolve()`); a lexical root would break under a
  symlinked root.
- Added an "Overflow past 99" section: `f"{slot:02d}"` is a minimum width, so
  slot 100 renders `Quick Session 100` / `quick-100` with no truncation, no
  collision and no renaming of existing slots. Only lexicographic `ls` order
  breaks (`quick-100` sorts before `quick-99`); nothing in the app sorts by
  directory name (sidebar = priority/time per `session_listing.sort_session_rows`,
  tmux mirror follows the sidebar), and `ls -v` restores natural order. Rollover
  was rejected (breaks never-reuse); a code-wide 3-digit pad was rejected (costs
  a leading zero now for a case that has not happened).
- Parser rule pinned: `^quick-(\d+)$` → int, so slot parsing is width
  independent; a malformed `next_slot` (missing/non-int/<=0) is treated as 1 and
  the invariant-5 clamp still protects every existing directory.
- Unit-test list extended: slot 100 formatting/allocation, malformed counter
  fallback, `is_quick_cwd` through a symlinked root.

Push held: `subagent-chat-01a17685` asked for no main publication while the
upload-robustness merge lands. Committed locally only; origin/main still
`2b2f1fae` at the time of this entry.

## 2026-10-26T16:40Z — implemented and verified

Shipped as 33889b8d (feature) + 60505805 (help-text correction), pushed to main
(b7544dcb..60505805) and deployed from the committed snapshot.

Implementation shape (deviations in QUICK_SESSION_PLAN.md §Status):
- `codoxear/quick_workspace.py` owns naming, containment, the slot counter
  (`<app_dir>/quick_sessions.json`, `{"next_slot": N}`, module lock, dirs as the
  floor) and `release_workspace` (rmdir-only-if-empty).
- `POST /api/quick-sessions` → `manager.allocate_quick_session()` in
  `session_routes._handle_quick_session_allocate`; no directory, no spawn.
- Client: `app_quick_session.js` reserves inside the gesture *after*
  `textarea.focus()`, then reuses `spawnSessionWithCwd(cwd, …, sessionName)` so
  the existing rename sets the alias; new seam `reserveQuickSession` wired in
  `app_wiring.js` + `app_application_composition.js`.
- `SessionStore.quick_root` filters quick cwds out of `recent_cwds` on write and
  read; `launch_attempt_row(..., quick_root=)` names provisional rows from the
  slot; `control_routes._handle_delete` trims an empty quick workspace.

Verification evidence:
- `scripts/docker_quick_session_verify.sh 33889b8d` → PASS, EXIT=0,
  artifacts /tmp/codoxear-quick-verify.FoJ4H4. Measured: session cwd
  `/home/tester/codoxear-quick/quick-01`; alias + display_name + sidebar
  `.titleText` all `Quick Session 01`; launch body cwd = the workspace path;
  `quick_dirs == ['quick-01']` after two failed starts (reserve creates nothing);
  `recent_cwds == ['/tmp']`; draft handoff, broker ACK, mobile drawer/focus,
  failure dialog `cwd: "~"`. In-container pytest: 45 passed.
- Screenshots: starting.png shows "Starting Pi · auto-jev-main · low in Quick
  Session 01…" with a preserved draft.
- Host suite: 2613 passed, 1 failed —
  `test_hint_mode_coverage::test_rendered_shell_interactives_are_declared_or_explicitly_excluded`
  (missing `draftUndoBtn`). Reproduced at b7544dcb in a detached worktree, so it
  predates this change.

## 2026-10-26T16:46Z — deployed

`scripts/deploy.sh 60505805` → EXIT=0 ("smoke test passed: initial load and
reload rendered a session card"; "deployed 60505805e91214dbb74ea3d1bc4a973f65821508").
Snapshot worktree `~/.local/share/codoxear/deploy` at 60505805 with the bundle
rebuilt from source.

Post-deploy checks (read-only):
- service active, WorkingDirectory=/home/yiwen/.local/share/codoxear/deploy,
  ExecStart=<pipx venv>/bin/python -u -m codoxear.server
- GET / → 200, GET /api/sessions → 401, GET /api/quick-sessions → 404 (POST-only),
  POST /api/quick-sessions → 401 while POST /api/nonexistent-route → 404
  (route registered and auth-gated)
- rebuilt bundle contains reserveQuickSession, api/quick-sessions, "own workspace",
  "quick session in a fresh workspace"
- quick_workspace.py present in the snapshot; 25 sidecars / 27 broker processes,
  all predating the restart (server 00:38:15, brokers from Oct 24 / 21:20)

main tip after this is 97b80f04 (docs + task memory only): `git diff 60505805
97b80f04` touches no file under codoxear/, scripts/ or tests/, so the deployed
code is byte-identical to main's code and no second restart was performed.
