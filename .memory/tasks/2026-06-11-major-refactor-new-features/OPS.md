# Operational ledger

## 2026-06-11 23:45
- Initialized task memory directory `.memory/tasks/2026-06-11-major-refactor-new-features/` from the user's prompt.
- Inspected repository status before creating files: `git status --short` produced no modified/untracked paths.
- Checked for existing `.memory` files with `find .memory -maxdepth 3 -type f`; no existing project/task memory files were observed.
- Read prompt template from `/home/yiwen/.pi/agent/PROMPT.template.md` and populated `PROMPT.md` with Objective, Workbench, Context, Task specifications, and Constraints.
- No live server/session operations were performed.

## 2026-06-11 23:52
- Updated `PROMPT.md` after user correction: final deliverable is now one `develop` acceptance branch, not separate final branches per workstream.
- Added explicit branch-topology guidance: temporary topic branches/worktrees are allowed as scaffolding, but accepted work integrates into `develop`; `main` remains untouched without approval.
- Searched for `harness mode` / `harness` references. Observed user-facing "Harness mode" in `codoxear/static/app.js` and server-side harness scheduling code in `codoxear/server.py`.
- Inspected the harness mechanism enough to update the task: server sweep sends a rendered prompt after an idle assistant turn, with cooldown and remaining-injection budget; prompt prefix already says "Unattended-mode instructions".
- Added a workstream requirement to rename/recast `harness mode` with mechanism-accurate terminology and validate it in the standalone Docker instance.

## 2026-06-11 23:57
- Updated `PROMPT.md` after user addition about long-conversation chat navigation ergonomics.
- Added a dedicated workstream for long-chat navigation: search, previous/next user-message jumps, latest/oldest relevant turn jumps, and time-based navigation/markers are candidate mechanisms.
- Added validation expectations for synthetic/fixture long conversations and browser-level evidence, including mobile/slow-network or slow-device conditions where feasible.
- Added a working-style section: future implementation should be creative and product-minded, with the numbered workstreams treated as a floor rather than a ceiling, bounded by hard ops constraints, validation, and product philosophy.

## 2026-06-12 00:01
- Started execution after user instruction.
- Inspected git topology: no existing `develop` branch; current `main` at `82f13ef4f1b485663c488b8c7179aeb938ae2898` and ahead of `origin/main` by 17 commits.
- Observed existing local worktrees/branches: `fix-upload-name`, `test-branch`, `feat/agent-harness`, and `fix/entering-text` are checked out in separate worktrees; main worktree was on `main` before branch creation.
- Created `develop` from the current local `main` as the single acceptance branch. No live runtime processes were touched.
- Commit `67cd430` captured `OPS.md` and `EPISTEMIC.md` but not `PROMPT.md` because repository `.gitignore` ignores files named `PROMPT.md`; this was observed from the `git add` warning.
- Force-added only the explicit task prompt file to preserve the acceptance contract in git.

## 2026-06-12 00:09
- Added Docker validation sandbox files: `.dockerignore`, `docker/sandbox.Dockerfile`, and `scripts/codoxear-docker-sandbox`.
- Ran `scripts/codoxear-docker-sandbox smoke`; result: passed. Evidence: pre-login `/api/me` returned 401, post-login `/api/sessions` returned 200, and container `APP_DIR` resolved to `/home/tester/.local/share/codoxear`. Artifacts were written under `/tmp/codoxear-docker-sandbox-18790/artifacts`.
- Ran `scripts/codoxear-docker-sandbox test` after aligning the sandbox with Python 3.13. Result: baseline before product changes was 355 passed, 2 failed, 2 skipped.
- Failing baseline tests: `tests/test_session_sidebar_priority.py::TestSessionSidebarPriority::test_delete_session_kills_terminal_owned_and_clears_dependents` (deleted session leaves `cwd:/tmp/target` file history) and `tests/test_voice_push_source.py::TestVoicePushSource::test_summary_prompts_cover_final_and_narration_targets` (voice prompt source text no longer contains expected approximate word-count phrases).

## 2026-06-12 00:11
- Fixed the two pre-existing Docker baseline failures before broader feature work.
- Changed `SessionManager.files_clear()` to discard the matching legacy `cwd:<cwd>` file-history bucket when deleting/clearing a known session, while still not migrating legacy cwd buckets into active session history.
- Updated voice summary prompts to include approximate target-range guidance (`about 15/30 words`, `roughly 12 to 18` / `24 to 36`) while preserving exact hard maximum instructions and runtime word-count validation.
- Ran targeted Docker tests for the two failures; result: passed.
- Ran full Docker test suite with `scripts/codoxear-docker-sandbox test`; result: `357 passed, 2 skipped in 4.68s`.

## 2026-06-12 00:26
- Ran six parallel read-only reconnaissance tasks via subagents. Five completed within the foreground timeout: architecture review, UI/product ergonomics, Claude Code support reconnaissance, git-history regression mining, and unattended-mode naming. The PR review child timed out, so PR inventory was completed manually with bounded `gh` and `git` commands.
- Saved reconnaissance artifacts under `recon/`: `architecture-review.md`, `ui-ergonomics.md`, `claude-code-support.md`, `git-history-bugs.md`, `unattended-mode-naming.md`, and `pr-review.md`.
- Fetched open PR heads into `refs/remotes/origin/pr/<n>` for comparison only; no source files were changed and no PR was merged/cherry-picked during triage.
- PR triage conclusion: do not merge stale PR branches wholesale; accept/reimplement selected small fixes (#12/#13/#14/#15/#17/#19) and mine #21 for minimal Claude Code support while rejecting/defering large UI-framework or auth/vendor rewrites.

## 2026-06-12 00:28
- Wrote integrated execution plan at `recon/integration-plan.md`.
- Plan prioritizes small accepted PR fixes, then network/responsiveness, long-chat navigation, Unattended mode rename, and finally minimal shared-broker Claude Code support.

## 2026-06-12 00:30
- Implemented accepted PR #13 static/package polish selectively against current code: URL-prefix-safe sidebar logo path, nested logo package data, optional immutable static cache headers, and no backend-tab rebuild during open new-session modal refresh.
- Added tests for static cache headers, sidebar icon path, modal refresh behavior, and wheel inclusion of nested logo assets.
- Targeted Docker validation: `scripts/codoxear-docker-sandbox test tests/test_static_assets.py -q` passed (`8 passed`).
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`362 passed, 2 skipped in 9.63s`).

## 2026-06-12 00:32
- Implemented accepted PR #14 tooltip polish in the shared JS element helper: buttons without explicit titles derive hover text from title, aria-label, data-tooltip, or text content.
- Added source regression test `tests/test_button_tooltips_source.py`.
- Docker validation: `scripts/codoxear-docker-sandbox test tests/test_button_tooltips_source.py tests/test_static_assets.py -q` passed (`9 passed`).

## 2026-06-12 00:35
- Implemented an opportunistic full conversation copy feature: new authenticated `messages/export` API reads all chat events for the selected transcript, rejects oversized logs instead of truncating, and the top bar copies role-marked plain text.
- Added tests in `tests/test_transcript_export.py` for all-event export order, oversized-log rejection, route wiring, and UI source wiring.
- Targeted Docker validation: `scripts/codoxear-docker-sandbox test tests/test_transcript_export.py tests/test_button_tooltips_source.py tests/test_static_assets.py -q` passed (`13 passed`).
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`367 passed, 2 skipped in 10.12s`).

## 2026-06-12 00:38
- User added scope: thinking-level/reasoning-effort support must be treated as backend/model-specific. Codex thinking-level support is incomplete, and Pi may not support all thinking efforts for all models.
- Updated `PROMPT.md` with a dedicated thinking-level capability workstream, explicit no-silent-downgrade semantics, and verification criteria for supported/unsupported combinations.

## 2026-06-12 00:41
- Implemented accepted PR #19 quiet HTTP disconnect handling: classified BrokenPipe/ConnectionReset/ConnectionAborted and equivalent errno values as client disconnects, suppressed request-boundary traceback/500 attempts for those only, and preserved normal 500 behavior for other exceptions.
- Added `tests/test_client_disconnects.py` for disconnect classification and route exception behavior.
- Targeted Docker validation: `scripts/codoxear-docker-sandbox test tests/test_client_disconnects.py tests/test_message_route_source.py -q` passed (`5 passed`).
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`370 passed, 2 skipped in 9.65s`).

## 2026-06-12 00:43
- Implemented accepted PR #17 stale sidecar handling: discovery and session metadata refresh now prune sockets/sessions whose broker metadata sidecar is missing instead of raising through the sessions/messages API path.
- Added `tests/test_stale_sidecars.py` for discovery-time lone socket pruning and selected-session refresh pruning when a sidecar disappears.
- Targeted Docker validation: `scripts/codoxear-docker-sandbox test tests/test_stale_sidecars.py -q` passed (`2 passed`).
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`372 passed, 2 skipped in 9.51s`).

## 2026-06-12 00:47
- Implemented accepted PR #12/#15 Pi log-binding hardening: broker state now records the declared Pi `--session` log path, publishes it in metadata before first write, and the discover watcher registers that declared path once it appears before consulting the active-session marker.
- Added broker regression coverage in `tests/test_broker_fail_closed.py` for new Pi session log reservation and declared-log registration after file creation.
- Targeted Docker validation: `scripts/codoxear-docker-sandbox test tests/test_broker_fail_closed.py -q` passed.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`374 passed, 2 skipped in 10.06s`).

## 2026-06-12 00:50
- Implemented Phase B adaptive session polling: replaced fixed 2.5s `setInterval` with a recursive timeout loop using 2.5s visible cadence and 15s hidden cadence; visibility returning to visible schedules an immediate refresh, while hidden reschedules at the slower interval.
- Added `tests/test_session_polling_source.py` for visibility-aware timeout source invariants and auth/unload stop behavior.
- Targeted Docker validation: `scripts/codoxear-docker-sandbox test tests/test_session_polling_source.py tests/test_static_assets.py -q` passed (`11 passed`).
- Invalid validation attempt preserved: `scripts/codoxear-docker-sandbox test sh -lc 'node --check codoxear/static/app.js'` failed because the test wrapper passed `sh` to pytest (`ERROR: file or directory not found: sh`).
- Corrected JS parse validation: direct Docker `node --check codoxear/static/app.js` passed.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`377 passed, 2 skipped in 9.45s`).

## 2026-06-12 00:52
- Implemented Phase C loaded-chat user-turn navigation: top-bar previous/next user-message buttons jump among currently loaded `.msg-row.user` rows, show explicit boundary/no-loaded-user toasts, respect reduced-motion preference, and briefly pulse the target row.
- Added `tests/test_chat_navigation_source.py` for button wiring, loaded-row-only jump semantics, boundary toasts, and pulse styling.
- Targeted Docker validation: `scripts/codoxear-docker-sandbox test tests/test_chat_navigation_source.py tests/test_button_tooltips_source.py -q` passed (`4 passed`).
- JS parse validation: direct Docker `node --check codoxear/static/app.js` passed.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`380 passed, 2 skipped in 9.48s`).
