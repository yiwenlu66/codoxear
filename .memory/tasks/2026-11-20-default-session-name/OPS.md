# OPS — default session name

Append-only evidence trail. Timestamps are local (host Asia/Shanghai).

## 2026-11-20 01:15 — mechanism investigation

- Reported: unnamed session shows the cwd on the codoxear sidebar but Pi's own
  session name is `subagent_chat_xxx`.
- `grep -rn "subagent_chat|subagent-chat"` (repo): only task-memory mentions.
  Found the producer outside the repo: `pi-intercom` `index.ts:33,400`
  (`DEFAULT_UNNAMED_SESSION_ALIAS_PREFIX = "subagent-chat"`, `resolveIntercomPresenceName`
  → name or `subagent-chat-<id8>`), mirrored by
  `pi-subagents/src/intercom/intercom-bridge.js:65` for the orchestrator target.
- `grep -rh '"type":"session_info"' ~/.pi/agent/sessions` → names present are
  codoxear-set ("Quick Session 23") or pi-subagents child names
  ("subagent-executor-…"); a fresh codoxear Pi launch writes none.
- Pi bundle (`dist/bundle/chunks/chunk-VEJDBY3O.js`): `if(parsed.name!==void 0){ … sessionManager.appendSessionInfo(name) }`
  → `--name` writes the title at startup, before `session_start`; TUI footer
  renders `pwd • sessionName`; terminal title `pi - <name> - <cwd>`.
- `pi-intercom` `startNamePoll()` + `getNamePollMs()` (1 s default) republishes
  presence when `getSessionName()` changes → a name pushed later does heal the
  presence name (so the projector path is viable, no launch-argv needed).
- Reconcile triggers confirmed: `session_refresh.refresh_session_meta` end
  (sidecar refresh; reached from transcript/file/git/diagnostics routes, send
  readiness, voice) plus the rename/edit route. No timer.

## 2026-11-20 01:20 — implementation

- New `codoxear/session_default_name.py` (rule), `session_display.cwd_display_basename`
  (shared basename), `SessionUiStateCoordinator.alias_assign_default` (atomic
  assign-once store op), projection rule 4 + `origin: "default"`,
  `session_manager_factories` + `server.py` wiring.
- Tests: `tests/test_session_default_name.py`,
  `tests/test_session_ui_state_default_name.py`,
  `tests/test_session_default_name_integration.py`, extended
  `tests/test_session_name_projection.py`.
- `python3 -m pytest tests/ -q` → 2987 passed, 1 failed
  (`test_hint_mode_coverage.py::…draftUndoBtn`). Verified pre-existing: the same
  test fails with my tracked changes stashed (`git stash push` of the new files
  → 1 failed).

## 2026-11-20 01:30 — Docker behavioral verification (run 1)

- `scripts/docker_session_name_verify.sh` (port 19681) extended with a Step 0.
- Result: `PASS: Pi default + propagated session names verified (session
  broker-45, default workspace)`; `EXIT=0`.
  Artifacts `/tmp/codoxear-session-name-results.9VJa5Y`:
  - `default-name.json` → `{"count": 1, "names": ["workspace"]}` (Pi's own
    `session_info` row arrived from the default-name projection)
  - `adopt.json` → `{"count": 2, "names": ["workspace", "terminal title"]}`
    (terminal `/name` superseded the assigned default)
  - `after-rename.json` → `{"count": 3, … "web name"}` (codoxear rename, one row)
  - repeat rename → no new row; clear → empty name row, alias stays empty.

## 2026-11-20 01:35 — rule correction (numbering order)

- Integration test showed the first rule (`taken` included a still-unnamed
  session's cwd basename) hands the plain name to whichever session is
  reconciled *last*: `s1 → repo-2`, `s2 → repo`. Replaced with "taken = other
  live sessions' assigned names (alias, else backend title)"; numbering now
  follows birth order. Retests green.

## 2026-11-20 01:42 — origin survives failed/unsupported pushes

- A default whose push failed (transport) or was unsupported recorded no
  origin, so a later terminal title would have been overwritten by the retry.
  `_record_failure` and the unsupported record now carry `origin`. New tests
  `test_a_failed_default_push_still_yields_to_a_terminal_title`,
  `test_an_unsupported_default_push_still_yields_to_a_terminal_title`.

## 2026-11-20 01:45 — commit + final verification

- Commit `989a813d` pushed to `origin/main`.
- `scripts/docker_session_name_verify.sh` re-run (port 19685) against the
  committed tree: PASS, `EXIT=0`
  (`PASS: Pi default + propagated session names verified (session broker-45,
  default workspace)`). Artifacts `/tmp/codoxear-session-name-results.FVSzCx`:
  `default-name.json` `{"count":1,"names":["workspace"]}`, `adopt.json`
  `{"count":2,"names":["workspace","terminal title"]}`, `after-rename.json`
  `{"count":3,…"web name"}`, repeat rename → `reason: unchanged`, clear →
  `reason: cleared` with an empty `session_info`.
