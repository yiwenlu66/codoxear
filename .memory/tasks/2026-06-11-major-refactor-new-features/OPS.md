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

## 2026-06-12 00:55
- Implemented Phase C loaded-message search: a compact top-bar search affordance opens a floating search bar over the chat, searches rendered rows only, supports Enter/Shift+Enter and previous/next buttons, shows loaded-scope counts, and highlights the current match.
- Extended `tests/test_chat_navigation_source.py` for loaded-search button copy, rendered-row-only matching, no-match toast copy, and compact overlay/current-hit styling.
- Targeted Docker validation: `scripts/codoxear-docker-sandbox test tests/test_chat_navigation_source.py -q` passed (`5 passed`).
- JS parse validation: direct Docker `node --check codoxear/static/app.js` passed.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`382 passed, 2 skipped in 9.54s`).

## 2026-06-12 01:03
- User corrected Phase D scope: no compatibility layer is needed for Harness naming. Removed the planned `/harness` alias and `harness_*` public session fields from the implementation direction.
- Implemented clean public rename to Unattended mode: UI copy/help/badge/API calls now use Unattended terminology; server exposes `/api/sessions/<id>/unattended`, emits only `unattended_*` session-list fields, writes `unattended.json`, and uses `CODEX_WEB_UNATTENDED_SWEEP_SECONDS`.
- Updated `PROMPT.md` and `recon/integration-plan.md` to remove compatibility requirements for this rename.
- Added `tests/test_unattended_mode_source.py` for Unattended UI copy/API path, no `/harness` route alias, no `harness_*` public session fields, new env/state names, and README terminology.
- Targeted Docker validation: `scripts/codoxear-docker-sandbox test tests/test_unattended_mode_source.py tests/test_harness_input_source.py tests/test_harness_sweep.py -q` passed (`12 passed`).
- JS parse validation: direct Docker `node --check codoxear/static/app.js` passed.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`387 passed, 2 skipped in 9.75s`).

## 2026-06-12 01:07
- Implemented user-added thinking-level capability workstream for Pi: server reads per-model reasoning effort capabilities from Pi `models.json` (`reasoning:false`, `reasoningEfforts`/`reasoning_efforts`/thinking aliases), exposes `reasoning_efforts_by_model`, and rejects unsupported Pi model/effort combinations instead of passing them through.
- Updated new-session UI reasoning choices to use the current provider/model capability map and revalidate effort selection when provider/model changes.
- Added `tests/test_reasoning_effort_source.py` and extended `tests/test_launch_defaults.py` for Pi model-specific efforts and hard rejection of unsupported `high` for a `reasoning:false` model.
- Targeted Docker validation: `scripts/codoxear-docker-sandbox test tests/test_launch_defaults.py tests/test_reasoning_effort_source.py -q` passed (`15 passed`).
- JS parse validation: direct Docker `node --check codoxear/static/app.js` passed.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`392 passed, 2 skipped in 9.97s`).

## 2026-06-12 01:30
- Implemented Phase E minimal Claude Code backend support (`agent_backend=cc`): backend registry, Claude log parser, session log filtering/metadata, chat extraction, busy/idle heuristics, launch defaults, web-session spawn args, UI backend tab/defaults, docs/env examples, and logo asset.
- Local CLI evidence: `claude --help` reports `--model`, `--effort low|medium|high|xhigh|max`, `--resume`, and `--dangerously-skip-permissions`; binary strings include `CLAUDE_CONFIG_DIR`, so Codoxear uses `CLAUDE_CONFIG_DIR` as Claude home isolation variable and `CLAUDE_BIN` for binary override.
- Added tests: `test_cc_log.py`, `test_cc_backend_registration.py`, `test_cc_session_log.py`, `test_cc_chat_and_idle.py`, `test_cc_busy_state.py`, `test_claude_backend_source.py`; extended launch defaults and session resume/spawn tests.
- Focused validation: `python3 -m py_compile codoxear/cc_log.py codoxear/rollout_log.py codoxear/server.py codoxear/broker.py codoxear/sessiond.py` passed.
- Focused Docker validation: `scripts/codoxear-docker-sandbox test tests/test_cc_log.py tests/test_cc_backend_registration.py tests/test_cc_session_log.py tests/test_cc_chat_and_idle.py tests/test_cc_busy_state.py tests/test_launch_defaults.py tests/test_session_resume.py tests/test_claude_backend_source.py -q` passed (`60 passed`).
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`415 passed, 2 skipped in 9.55s`).
- JS parse validation: direct Docker `node --check codoxear/static/app.js` passed.

## 2026-06-12 01:33
- Follow-up before finalizing CC commit: patched `infer_agent_backend_from_log_path()` to recognize CC logs under a custom `CLAUDE_CONFIG_DIR`, not only literal `~/.claude/projects` paths.
- Added regression in `tests/test_cc_backend_registration.py` for custom Claude config directory inference.
- Targeted Docker validation: `scripts/codoxear-docker-sandbox test tests/test_cc_backend_registration.py -q` passed (`3 passed`).
- Full Docker validation after follow-up: `scripts/codoxear-docker-sandbox test` passed (`416 passed, 2 skipped in 9.72s`).
- JS parse validation after follow-up: direct Docker `node --check codoxear/static/app.js` passed.

## 2026-06-12 01:39
- Implemented file-picker ergonomics improvement from remaining PROMPT workstreams: typed file search now shows locally-known fuzzy matches immediately while the full project/server search is pending, and keeps local matches usable if server search fails.
- Added honest footer copy (`Searching full project...`) so users can distinguish immediate local results from still-loading full-project results.
- Added `tests/test_file_picker_search_source.py` with Node VM coverage for pending-search local results and local results after search error.
- Targeted Docker validation: `scripts/codoxear-docker-sandbox test tests/test_file_picker_search_source.py tests/test_file_viewer_source.py tests/test_file_picker_session_state.py -q` passed (`20 passed`).
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`419 passed, 2 skipped in 9.72s`).
- JS parse validation: direct Docker `node --check codoxear/static/app.js` passed.

## 2026-06-12 01:42
- Implemented minimal new-session provider/model ergonomics improvement: recent session model menu entries now retain provider choice metadata, are searchable as `provider/model`, and selecting such an entry updates both model and provider through the existing model combobox.
- Avoided a broad combined picker rewrite; the intervention preserves the existing sparse controls while reducing repeat-launch steps for recent provider/model pairs.
- Added `tests/test_new_session_model_options_source.py` with Node VM coverage for provider-bearing recent model options and provider/model filtering.
- Targeted Docker validation: `scripts/codoxear-docker-sandbox test tests/test_new_session_model_options_source.py tests/test_launch_ui_source.py tests/test_launch_defaults.py -q` passed (`19 passed`).
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`422 passed, 2 skipped in 9.90s`).
- JS parse validation: direct Docker `node --check codoxear/static/app.js` passed.

## 2026-06-12 01:44
- Added deterministic git-history pressure-test regressions without behavior changes: Unattended/legacy harness sweep now covers three sessions sharing one thread and zero remaining injections disabling without sending; JSONL offset reader now covers a complete line followed by a partial appended JSON object.
- Targeted Docker validation: `scripts/codoxear-docker-sandbox test tests/test_harness_sweep.py tests/test_read_jsonl_from_offset.py -q` passed (`10 passed`).
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`425 passed, 2 skipped in 10.01s`).

## 2026-06-12 01:46
- Ran browser-level validation against isolated Docker server on port 18791 with isolated root `/tmp/codoxear-docker-sandbox-browser`; did not touch live server/session state.
- Browser evidence via `agent-browser`: pre-login page showed password/login; after login, top bar showed `Search loaded messages`, previous/next user-message controls, and `Unattended mode` controls disabled with no selected session.
- New-session modal browser snapshot showed backend tabs `Codex`, `Pi`, and `Claude` plus model/reasoning/provider/tmux controls.
- After clicking `Claude`, browser eval returned `providerVisible:false`, `fastVisible:false`, `reasoningText:"medium"`, and model placeholder `Model`.
- Cleanup: closed ephemeral `agent-browser` session and stopped Docker sandbox; subsequent curl to `127.0.0.1:18791` failed to connect as expected.

## 2026-06-12 01:51
- Ran required clean-room adversarial review via `architect` subagent after two heavier review attempts timed out and one lightweight delegate returned no usable findings.
- Review result: no blockers; residual risks limited to already-known unvalidated live-like backend/session paths (real Codex/Pi/Claude creation, Monaco browser race, zsh/oh-my-zsh startup, full long-transcript interaction). Verdict: yield acceptable on `develop`; do not merge `main`.

## 2026-06-12 01:59
- Continued after final candidate review to close a remaining feasible evidence gap: browser-level long-chat validation with synthetic state only.
- Started a fresh isolated Docker server on port 18792 with root `/tmp/codoxear-docker-sandbox-longchat`; no live server/session/broker state was touched.
- Created a synthetic mock broker inside the container with a Unix control socket under the sandbox Codoxear app dir and a synthetic Codex rollout log containing 1 `session_meta` row plus 320 chat rows (160 user + 160 assistant).
- Negative evidence preserved: the first synthetic-log attempt omitted Codex `session_meta`; discovery failed loudly with `ValueError: missing session metadata`, so the fixture was invalid and the sandbox was restarted clean before browser measurement.
- API discovery after corrected fixture listed one synthetic Codex session with model `synthetic-model` and reasoning `low`.
- Browser evidence via `agent-browser`: login succeeded; synthetic session auto-selected; initial loaded tail contained 61 rendered rows, 30 loaded user rows, and began at `User turn 131`; loaded-search overlay opened and querying `UNIQUE-needle` returned status `1/1 loaded` with current hit `Assistant answer 149... UNIQUE-needle`; previous/next user-message navigation pulsed loaded user rows; Load older history path expanded rendering to 321 rows / 160 user rows with first text `User turn 001`.
- Cleanup: closed the ephemeral browser session, stopped the Docker sandbox, and verified port 18792 no longer accepted connections.

## 2026-06-12 02:01
- Ran clean-room final yield gate via `architect` subagent after the added long-chat browser validation.
- Review result: no blockers under stated constraints. Risks are deferred live-like Codex/Pi/Claude creation, mobile/performance/network tracing and Monaco/file-viewer races, zsh/oh-my-zsh startup pressure, real-device long-transcript testing, and the scoped idle limitation of the synthetic long-chat fixture.
- Verdict: yielding now is acceptable for `develop` as acceptance candidate; do not merge `main` without explicit approval.

## 2026-06-12 02:09
- Continued safe PROMPT work by cleaning the remaining internal Harness implementation names to Unattended terminology. Public API/state was already Unattended-only; this commit removes the residual semantic mismatch in constants, manager fields/methods, sweep/thread names, DOM IDs/classes/functions, tests, and AGENTS.md state-file docs.
- Source check: `grep -R "harness\|HARNESS\|Harness" -n codoxear README.md AGENTS.md .env.example` returned no matches after the cleanup; remaining test matches are negative assertions that guard against reintroducing Harness compatibility.
- Syntax validation: `python3 -m py_compile codoxear/server.py` passed; `node --check codoxear/static/app.js` passed; Dockerized `node --check codoxear/static/app.js` passed.
- Targeted Docker validation: `scripts/codoxear-docker-sandbox test tests/test_unattended_sweep.py tests/test_unattended_input_source.py tests/test_unattended_mode_source.py tests/test_stale_sidecars.py tests/test_session_file_history.py tests/test_session_sidebar_priority.py tests/test_hidden_sessions_startup.py tests/test_launch_provenance.py tests/test_session_resume.py tests/test_sessions_pending_log_idle.py -q` passed.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`425 passed, 2 skipped in 10.66s`).
- Browser smoke for renamed Unattended DOM/API: isolated Docker server on port 18793 with synthetic broker/log; browser login succeeded; Unattended button rendered and opened `#unattendedMenu`; default controls loaded; toggling enabled and editing fields saved through `/api/sessions/unattended/unattended`; persisted sandbox `unattended.json` contained enabled true, cooldown 7, request text. Remaining injections was observed as 2 rather than typed 3 because the isolated sweep immediately injected once into the idle synthetic session, which also exercises the renamed sweep path.
- Cleanup: closed the ephemeral browser and stopped the Docker sandbox; port 18793 no longer accepted connections.

## 2026-06-12 02:10
- Refreshed `PROMPT.md` Workbench section because it still described the initial pre-implementation state. The prompt now states that `develop` is the acceptance candidate, summarizes the current evidence base, lists remaining optional next tasks requiring authorization, and preserves negative evidence/residual unknowns.

## 2026-06-12 02:12
- Bounded Codex reasoning-effort follow-up: ran local `codex --help` read-only. It exposes generic `-c key=value` config overrides but no authoritative per-model reasoning capability source. No live session was started.
- Final clean-room gate: first `architect` review attempt timed out; reran concise read-only gate with `delegate`, which reported no blockers for `develop` acceptance and identified only the known deferred live-like/device/performance/Codex-capability risks.

## 2026-06-12 02:17
- Added `recon/final-acceptance-summary.md` as a stable review artifact for the `develop` candidate. It summarizes integrated work, PR decisions, validation evidence, negative evidence, scoped limitations, and parked user decisions.
- Added status notes to `recon/integration-plan.md` and `recon/unattended-mode-naming.md` so future readers do not mistake historical pre-implementation compatibility recommendations for the final accepted design.
- Validation: `git diff --check` passed; this was documentation-only, so no runtime test was required.

## 2026-06-12 02:19
- Final clean-room review after adding the acceptance summary: `reviewer` timed out without findings; reran concise read-only `delegate` gate.
- Delegate result: no blockers for `develop` acceptance under constraints; risks limited to deferred real backend/session creation, mobile/network/performance/Monaco/zsh/full real transcript validation, and unresolved Codex per-model reasoning capability source. Verdict: PASS/yield `develop`; do not merge `main` without approval.

## 2026-06-12 02:25
- Implemented a remaining deterministic git-history pressure-test gap for assistant message deduplication. Chat extraction now dedupes adjacent assistant events with the same normalized text/message class within an assistant stretch, including different timestamps, while resetting the dedupe key on user messages so the same text can appear in a later turn.
- Added regressions in `tests/test_message_index.py` for tail-page and live-delta duplicate assistant text handling.
- Targeted validation: `python3 -m py_compile codoxear/rollout_log.py` passed; `scripts/codoxear-docker-sandbox test tests/test_message_index.py tests/test_transcript_export.py tests/test_voice_push.py -q` passed (`36 passed`).
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed (`427 passed, 2 skipped in 11.53s`).
- Updated `PROMPT.md` and `recon/final-acceptance-summary.md` validation evidence to the new 427-test count.

## 2026-06-12 02:27
- Final clean-room review after assistant chat-dedupe change used a concise read-only `delegate` gate.
- Review result: no blockers; risks limited to deferred real backend/device/performance validation, unresolved Codex per-model reasoning source, and cross-poll duplicate suppression beyond batch/page extraction not being proven. Verdict: PASS/yield `develop`; do not merge `main` without approval.

## 2026-06-12 02:35
- Closed the previously scoped cross-poll assistant duplicate gap in the client live-append path. Rendered assistant rows now carry a normalized assistant dedupe key; `appendEvent()` suppresses a later live delta if it is an adjacent assistant duplicate with the same message class/text and marks the event seen. A user row resets the adjacency because the DOM tail is no longer assistant.
- Added Node VM regression `test_live_delta_dedupes_adjacent_assistant_text_across_polls`, which distinguishes duplicate suppression after an assistant tail from preserving the same assistant text after a user row.
- Targeted validation: local `node --check codoxear/static/app.js` passed; `scripts/codoxear-docker-sandbox test tests/test_chat_transcript_runtime.py tests/test_chat_scrollback_source.py tests/test_message_index.py -q` passed (`27 passed`).
- Full validation: `scripts/codoxear-docker-sandbox test` passed (`428 passed, 2 skipped in 10.91s`); Dockerized `node --check codoxear/static/app.js` passed.
- Updated current acceptance evidence in `PROMPT.md` and `recon/final-acceptance-summary.md` from 427 to 428 tests and clarified the stronger assistant-dedupe scope.

## 2026-06-12 02:37
- Final clean-room review after the client live-append dedupe change used a concise read-only `delegate` gate.
- Review result: no blockers; risks limited to deferred live-like backend/device/performance validation, unresolved Codex per-model reasoning source, and complex non-adjacent/streaming duplicate patterns outside adjacent dedupe coverage. Verdict: PASS/yield `develop`; do not merge `main` without approval.

## 2026-06-12 02:44
- Continued after user request under Unattended-mode rules. Reloaded `PROMPT.md` and confirmed `develop` is clean and remains the acceptance branch; `main` is untouched.
- Deterministic gap found: the wheel/package-data regression covered nested Codex/Pi logos but did not explicitly assert the newly added Claude Code `cc.svg` logo. Since the UI path resolves backend logos by backend id, missing package data would break installed Claude UI despite source tests passing.
- Added `codoxear/static/logos/cc.svg` to `tests/test_static_assets.py::test_wheel_includes_nested_logo_assets` assertions and updated `recon/final-acceptance-summary.md` to include packaged Claude logo coverage.
- Targeted validation: `scripts/codoxear-docker-sandbox test tests/test_static_assets.py -q` passed (`8 passed`).
- Full validation: `scripts/codoxear-docker-sandbox test` passed (`428 passed, 2 skipped in 10.89s`).

## 2026-06-12 02:45
- Deterministic source mismatch found: `scripts/codoxear-docker-sandbox` supports a `build` command but its usage text omitted it. This affects the isolated validation tool's self-documenting contract, not runtime Codoxear behavior.
- Updated the usage text and added `tests/test_docker_sandbox_source.py` to assert every supported top-level command, including `build`, appears in the usage/dispatch source.
- Targeted validation: `scripts/codoxear-docker-sandbox test tests/test_docker_sandbox_source.py -q` passed (`1 passed`).
- Full validation: `scripts/codoxear-docker-sandbox test` passed (`429 passed, 2 skipped in 10.86s`). Updated `PROMPT.md` and `recon/final-acceptance-summary.md` evidence counts.

## 2026-06-12 02:48
- Final clean-room adversarial gate after the packaging and sandbox-usage regressions used fresh `critic` context.
- Review result: no blockers and no deterministic actionable work before yield. It confirmed branch `develop`, clean worktree, latest commits present, and `main` not merged. Remaining risks are user-decision-bound or explicitly scoped: live-like backend session creation, long real Claude session, mobile/network/performance, Monaco/file-viewer races, zsh startup, full real long transcript, Codex per-model reasoning authority, and non-adjacent/streaming assistant duplicate patterns.
- Verdict: acceptable to yield `develop`; do not merge to `main` without explicit approval.

## 2026-06-12 02:53
- Continued after user request by checking remaining cross-workstream verification criteria rather than adding product scope. Found that the latest acceptance summary did not explicitly record an editable install or isolated server-start smoke after the final branch state.
- First editable-install command used a bad post-check: `pip install -e .` succeeded, but the assertion `shutil.which("codoxear-server")` failed because the container defaulted to a user install and `/home/tester/.local/bin` is not on `PATH`. This was a measurement artifact, not an install failure.
- Corrected editable-install validation: copied the read-only repo into a writable `/tmp/src` inside `codoxear-sandbox:latest`, ran `python3 -m pip install -e .`, imported `codoxear.server` and `codoxear.broker`, and verified `codoxear-server`/`codoxear-broker` scripts exist under `/home/tester/.local/bin`.
- Isolated server smoke: ran `scripts/codoxear-docker-sandbox smoke` with `CODOXEAR_DOCKER_PORT=18794`, `CODOXEAR_DOCKER_NAME=codoxear-sandbox-acceptance-18794`, and `CODOXEAR_DOCKER_ROOT=/tmp/codoxear-docker-sandbox-acceptance-18794`; result: pre-login `/api/me` `401`, post-login `/api/sessions` `200`, container `APP_DIR=/home/tester/.local/share/codoxear`. Stopped the sandbox container afterward.
- Updated `PROMPT.md` and `recon/final-acceptance-summary.md` with the editable-install and server-smoke evidence.

## 2026-06-12 02:56
- Final-yield review attempt with `reviewer` timed out after 120s without returning a finding. To avoid repeating an overlong review, ran a smaller fresh `delegate` gate with the same deliverables, evidence, parked decisions, constraints, and latest changed artifacts.
- Delegate gate result: no blockers; no actionable deterministic work before yield; risks to report are real mobile/perf/race/zsh/full-transcript/long-Claude validation, real backend credentials/binaries not validated, unavailable Codex per-model reasoning source, and complex non-adjacent/streaming assistant duplicates outside adjacent dedupe. Verdict: yield acceptable on `develop`; keep `main` untouched.

## 2026-06-12 11:32
- User corrected task ordering: the older major feature request was overclaimed and must be repaired before structural refactor continues.
- Updated task prompts so product-gap recovery is the active priority and the structural refactor prompt is explicitly parked until those gaps are fixed, browser-validated, and honestly scoped.
- Concrete recovery gates now include real provider/model selection, top-bar/action placement redesign, long-chat ergonomics, responsiveness evidence, file-viewer polish, incomplete git-history pressure coverage, and scoped backend/reasoning claims.

## 2026-06-12 11:40
- Created clean isolated recovery branch/worktree `/home/yiwen/codex-web-product-recovery` from `develop` so feature recovery can happen before parked structural refactor history.
- Ported the prompt correction from the parked refactor worktree and strengthened task prompts around product promises/workflows/invariants/evidence as the acceptance ontology.
- Any imported structural-refactor ledger history is parked evidence from the separate refactor worktree; it must not be treated as active recovery progress or acceptance proof for this branch.

## 2026-06-12 11:49
- Product recovery implementation checkpoint: replaced separate visible Provider + Model new-session controls with one combined provider/model combobox and moved session/chat actions out of the topbar.
- Changed files: codoxear/static/app.js, codoxear/static/app.css, tests/test_new_session_model_options_source.py, tests/test_chat_navigation_source.py, tests/test_claude_backend_source.py.
- Validation: local `node --check codoxear/static/app.js` passed. Local frontend/source subset passed: `85 passed in 1.14s` for button tooltips, chat navigation/scrollback/transcript, Claude backend source, file picker/viewer, launch UI, markdown source, new-session model options, reasoning effort, polling, Unattended, and voice source/runtime tests.

## 2026-06-12 12:04
- Browser validation ran against isolated Docker sandbox `codoxear-sandbox-recovery-18812` on `http://127.0.0.1:18812/` with mock broker sockets and synthetic Codex/Pi/Claude logs under `/home/tester`.
- Observed desktop: topbar DOM buttons were only `toggleSidebarBtn` and `interruptBtn`; session utilities lived in `#sessionContextBar`; loaded-chat search/user navigation lived in `#chatNavRail`; New Session exposed one `Provider / model` combobox and no `newSessionProviderBtn`/`newSessionProviderMenu` DOM.
- Browser-discovered bug: after entering invalid Codex provider text (`bogus/gpt-5.4`) and switching to Pi, stale provider error text remained. Fixed by clearing provider/model error state when input becomes valid or backend changes. Rechecked: Pi showed label `Model`, placeholder `Model`, empty status, no field error.
- Browser-discovered fixture issue: initial synthetic Codex user messages used a shape the Codex normalizer ignores; corrected sandbox fixture to `event_msg/user_message` before using long-chat navigation evidence.
- Observed long-chat UX after fixture correction: 60 loaded rows / 30 loaded user rows; search for `UNIQUE-RECOVERY-NEEDLE` showed `1/1 loaded`; current match was the seeded user turn; Next user jump pulsed a loaded user row.
- Observed mobile viewport 390x844: topbar stayed sparse, session utilities and chat nav were contextual rails, New Session modal fit the viewport, providerless Pi showed `Model`, Codex showed `Provider / model`, and no provider-only controls existed.
- Browser-discovered recovery-branch gap: custom modals did not isolate background `.app`; ported modal isolation (`inert` + `aria-hidden`) and transient overlay closure into recovery. Rechecked mobile New Session: `.app` had `inert` and `aria-hidden="true"`; accessibility snapshot contained only modal controls.
- Screenshots saved under `.memory/tasks/2026-06-11-major-refactor-new-features/browser-artifacts/`: recovery desktop/mobile main, new-session, providerless, chat-search, and modal-isolated images.
- Validation after modal fix: local `node --check codoxear/static/app.js` passed; frontend/source subset passed (`89 passed in 1.13s`); full isolated Docker suite passed (`435 passed, 2 skipped in 10.11s`).

## 2026-06-12 12:07
- Additional browser UX validation in isolated sandbox `codoxear-sandbox-recovery-18812`:
  - File/context workflow: opened file viewer from `#sessionContextBar`; `.app` was inert/aria-hidden while viewer was open; searching `README` returned create-new, `README.md`, and matching repo files; selecting `README.md` loaded file content and status `README.md - 8.72 KB` after wait.
  - Responsiveness sample on long Codex fixture: 60 rendered rows / 30 user rows; latest `/messages/tail` resource duration approximately 5.0 ms with transfer size 32892 bytes; loaded-chat search for `UNIQUE-RECOVERY-NEEDLE` completed in approximately 23.2 ms; user jump completed in approximately 33.1 ms and pulsed a user row.
- Additional screenshots saved: `recovery-desktop-file-search.png`, `recovery-desktop-file-read.png`.

## 2026-06-12 12:12
- Resumed scoped refactor after product recovery: cherry-picked backend-only seams from parked refactor branch onto `recovery/product-gaps` while preserving current recovery ledgers on append-only conflicts.
- Applied refactor commits: message cursor helper extraction, auth helper extraction, Unattended store extraction, shared request route parsing, voice route grouping, and broker JSONL reader dedupe. Skipped the frontend BackendConfig factory because it is more likely to conflict with the new provider/model workflow and is not necessary for the recovered product promises.
- Browser restart after refactor initially found a real regression: `/api/login` returned `NameError: name 'hmac' is not defined`, because the auth extraction was cherry-picked without its later fix. Applied the existing password-compare fix and regression test.
- Refactor validation: `python3 -m py_compile codoxear/server.py codoxear/message_cursor.py codoxear/auth.py codoxear/unattended.py codoxear/broker.py` passed; targeted Docker tests for message cursor/routes/auth/static/client-disconnect/Unattended/route decomposition/voice/broker JSONL passed; full validation after auth fix passed: local `node --check codoxear/static/app.js` and full isolated Docker suite `443 passed, 2 skipped in 10.09s`.
- Restarted isolated browser sandbox on port 18812 after refactors and reseeded corrected mock sessions. Login succeeded (`{"ok": true}`), `/api/sessions` listed Codex/Pi/Claude fixtures, topbar remained sparse, session/chat rails were present, New Session modal was inert/aria-hidden with one Provider/model combobox and no provider-only DOM, and loaded-chat search found the seeded marker (`1/1 loaded`).

## 2026-06-12 12:24
- Provider/model selector recovery follow-up: added per-backend remembered provider/model pair storage under `codoxear.newSessionProviderModel.<backend>` so the combined selector restores the actual chosen pair, not just the provider.
- Browser validation in isolated sandbox on port 18812: cleared Codex provider/provider-model localStorage, opened New Session, searched `chatgpt/gpt`, selected `chatgpt/gpt-5.4-mini`, observed localStorage `codoxear.newSessionProviderModel.codex = chatgpt/gpt-5.4-mini`, closed/reopened New Session, and observed input restored to `chatgpt/gpt-5.4-mini`; provider-only DOM count remained 0 and modal app isolation stayed active.
- Validation: local `node --check codoxear/static/app.js` passed; targeted frontend/source tests passed (`23 passed in 0.55s`); full isolated Docker suite passed (`444 passed, 2 skipped in 10.11s`).

## 2026-06-12 12:29
- Deep UX follow-up: measured chat navigation rail geometry in the isolated long-chat browser fixture. The absolute rail overlapped visible messages on desktop and mobile.
- Intervention: changed `chatWrap` to flex-column layout and `chatNavRail` to an in-flow static toolbar above the scroll area; kept it contextual but no longer overlaying message content. Added source regression coverage that the rail is in layout flow rather than `position:absolute`.
- Browser recheck: fresh browser session showed `#chatNavRail` computed `position: static`; strict visible-overlap calculation returned `[]` on desktop 1280x900 and mobile 390x844. Screenshot saved as `recovery-mobile-chat-rail-flow.png`.
- Validation: local `node --check codoxear/static/app.js` passed; targeted chat tests passed (`30 passed in 0.61s`); full isolated Docker suite passed (`445 passed, 2 skipped in 10.32s`).

## 2026-06-12 12:31
- Final validation after provider/model pair memory, chat rail flow fix, and Help copy alignment: local `node --check codoxear/static/app.js` passed; full isolated Docker suite passed (`445 passed, 2 skipped in 10.17s`).
- Current recovery branch head before clean-room review: `recovery/product-gaps` at `a70ab96`. Live checkout `/home/yiwen/codex-web` remains on `main` at `82f13ef`.

## 2026-06-12 12:37
- Deep UX follow-up: measured loaded-chat search geometry after opening search for `UNIQUE-RECOVERY-NEEDLE`. Before intervention, `#chatSearchBar` computed `position:absolute` and overlapped visible message rows on desktop and mobile.
- Intervention: moved `#chatSearchBar` into normal chatWrap flex layout above `#chatNavRail`; made the input flex-shrink within the toolbar and added source regression coverage that search is in-flow rather than absolute.
- Browser recheck: fresh browser session showed `#chatSearchBar` computed `position: static`; strict visible-overlap calculation returned `[]` for visible rows on desktop 1280x900 and mobile 390x844; search status remained `1/1 loaded`; screenshot saved as `recovery-mobile-chat-search-flow.png`.
- Validation: local `node --check codoxear/static/app.js` passed; targeted chat tests passed (`30 passed in 0.63s`); full isolated Docker suite passed (`445 passed, 2 skipped in 10.71s`).

## 2026-06-12 12:50
- Fresh clean-room product/UX and architecture reviews completed. Accepted findings: stale `renderNewSessionProviderMenu()` call in `refreshSessions()` while New Session is open; Pi launch mapping could turn an empty provider into Codex's `chatgpt`; malformed backend config could make `/api/sessions` fail.
- Interventions committed:
  - `6ed4dfa fix: remove stale provider menu refresh`: replaces stale provider-menu refresh with combined model-menu refresh and adds source guard.
  - `1a47f2e fix: keep pi launches providerless by default`: makes `providerChoiceToSettings()` backend-specific; executable Node test verifies Codex/Pi/Claude mappings.
  - `bd207a4 fix: degrade launch defaults per backend`: adds per-backend safe defaults, warning metadata, and nonblocking New Session warning text for degraded launch defaults.
  - `e72b330 test: align modal refresh invariant`: updates older static invariant test to the combined model-menu workflow.
- Browser/API evidence on restarted isolated Docker server `codoxear-sandbox-recovery-18812`:
  - `/api/sessions` returned all backend defaults `['cc', 'codex', 'pi']` with no warnings under clean isolated config.
  - New Session stayed open through a poll interval with captured `errors: []` and provider-only DOM count `0`.
  - Intercepted Pi new-session POST body omitted `model_provider` for providerless Pi: `{'agent_backend':'pi','cwd':'/tmp/codoxear-pi-provider-null-current','model':'default','reasoning_effort':'high','create_in_tmux':true}`.
  - With isolated container `/home/tester/.pi/agent/settings.json` malformed, authenticated `/api/sessions` still returned 200 with all backends and `warnings.pi`; New Session status showed `Launch defaults degraded for Pi; using safe defaults.`.
- Validation: local targeted checks passed (`35 passed in 1.76s` before full Docker). First full Docker surfaced a stale test expecting `renderNewSessionProviderMenu`; after updating that invariant, full isolated Docker passed (`449 passed, 2 skipped in 10.15s`).

## 2026-06-12 12:56
- Architecture/refactor follow-up: migrated session API route parsing from prefix+suffix checks to `_match_session_route()` for all `/api/sessions/<id>/...` route families.
- Commits:
  - `cf8e46e refactor: exact-match simple session routes`: queue/tail/unattended/send/enqueue/queue mutations/interrupt.
  - `d6b9bbe refactor: exact-match utility session routes`: diagnostics/edit/rename/inject_file/inject_image.
  - `0987933 refactor: exact-match file and git routes`: file read/search/list/blob/video_preview/download and git changed_files/diff/file_versions.
- Evidence: `rg 'path\.startswith\("/api/sessions/"\).*path\.endswith' codoxear/server.py` produced no matches. `_match_session_route` tests now reject extra path segments for queue, send, unattended, interrupt, diagnostics, edit, rename, inject, file, and git routes.
- Validation: `python3 -m py_compile codoxear/server.py` passed; `node --check codoxear/static/app.js` passed; targeted route/file/git tests passed (`36 passed in 1.87s`); full isolated Docker suite passed (`449 passed, 2 skipped in 10.44s`).

## 2026-06-12 13:02
- UX feature follow-up: added bounded server-backed all-transcript search counts while preserving the existing fast loaded-DOM search. Route: `/api/sessions/<id>/messages/search?q=...&limit=...` reuses export event extraction and transcript size limit; UI search status now shows loaded matches plus total transcript matches, e.g. `0/0 loaded · 3 all`.
- Commit: `8eeca75 feat: show all-transcript search counts`.
- Validation: `python3 -m py_compile codoxear/server.py` passed; `node --check codoxear/static/app.js` passed; targeted transcript/search tests passed (`20 passed in 1.69s`); full isolated Docker suite passed (`450 passed, 2 skipped in 10.17s`).

## 2026-06-12 13:04
- Architecture/refactor follow-up: moved `_current_git_branch(cwd)` execution out of `SessionManager.list_sessions()`'s manager lock. The lock now snapshots the resolved cwd path; git branch lookup occurs in the existing outside-lock pass that already computes log idle state.
- Commit: `44b7a0d refactor: read git branches outside manager lock`.
- Validation: targeted sidebar/provenance/session tests passed (`53 passed in 1.74s`); full isolated Docker suite passed (`451 passed, 2 skipped in 10.21s`).

## 2026-06-12 13:05
- Restarted isolated browser sandbox at current head on port 18812 (`codoxear-sandbox-recovery-18812`) after transcript search and list-lock refactors.
- Browser smoke: logged in, opened New Session, waited past a poll interval, switched to Pi tab. Observed captured JS errors `[]`, backend tabs `Codex/Pi/Claude`, provider-only DOM count `0`, modal still open, and topbar actions `['interruptBtn']`.

## 2026-06-12 13:18
- UX follow-up: extended loaded chat search so when server-backed all-transcript count shows matches outside the loaded DOM, pressing Next can page older history contiguously until a loaded match appears. The loop is bounded to 12 older pages and uses the existing `/messages/history` path to avoid creating gaps in rendered transcript state.
- Commit: `20122bf feat: page older chat search matches`.
- Validation: `node --check codoxear/static/app.js` passed; targeted chat/search tests passed (`30 passed in 1.74s`); full isolated Docker suite passed (`451 passed, 2 skipped in 10.37s`).

## 2026-06-12 13:21
- Architecture/refactor follow-up: extracted `load_json_file()` and `atomic_write_json()` in `codoxear/util.py`; migrated server app-dir stores (aliases/sidebar/hidden sessions/files/queues/recent cwd) and `UnattendedStore` to use the shared file IO helpers while preserving per-store schema cleaners.
- Commit: `72e062b refactor: share json state file helpers`.
- Validation: `python3 -m py_compile codoxear/util.py codoxear/unattended.py codoxear/server.py` passed; targeted persistence/session tests passed (`35 passed in 2.07s`); full isolated Docker suite passed (`453 passed, 2 skipped in 10.63s`).

## 2026-06-12 13:23
- Architecture/refactor follow-up: migrated voice push settings, subscriptions, and delivery ledger JSON stores to shared `load_json_file()`/`atomic_write_json()` helpers.
- Commit: `66a4592 refactor: share voice json state helpers`.
- Validation: `python3 -m py_compile codoxear/util.py codoxear/voice_push.py codoxear/server.py` passed; targeted voice/helper tests passed (`34 passed in 0.83s`); full isolated Docker suite passed (`453 passed, 2 skipped in 10.33s`).

## 2026-06-12 13:25
- Architecture/refactor follow-up: moved `_read_run_settings_from_log()` out of `SessionManager.list_sessions()`'s manager lock. `list_sessions()` now snapshots whether settings are needed, scans the log outside the lock, then briefly re-locks to update the still-current session.
- Commit: `31271fd refactor: read run settings outside manager lock`.
- Validation: `python3 -m py_compile codoxear/server.py` passed; targeted sidebar/provenance/session tests passed (`54 passed in 1.80s`); full isolated Docker suite passed (`454 passed, 2 skipped in 10.38s`).

## 2026-06-12 13:27
- Architecture/refactor follow-up: moved first-history timestamp recovery (`_last_conversation_ts_from_tail`) out of `SessionManager.list_sessions()`'s manager lock. The outside-lock result is applied with a guarded re-lock and row `updated_ts`/priority/recent-cwd fields are recomputed when a conversation timestamp is found.
- Commit: `2106d04 refactor: read history timestamps outside manager lock`.
- Validation: `python3 -m py_compile codoxear/server.py` passed; targeted timestamp/session tests passed (`62 passed in 1.81s`); full isolated Docker suite passed (`454 passed, 2 skipped in 10.20s`).

## 2026-06-12 13:34
- Architecture/refactor follow-up: extracted `/api/sessions` POST launch validation into `NewSessionLaunchRequest` and `_parse_new_session_launch_request()`. The HTTP route now handles auth/body/error response and calls `spawn_web_session()` with the normalized request; backend-specific provider/reasoning/service-tier validation lives in one parser.
- Commit: `36cd30f refactor: parse new session launch requests`.
- Validation: targeted parser/reasoning/launch tests passed (`62 passed in 1.79s`); full isolated Docker suite passed (`458 passed, 2 skipped in 10.22s`).

## 2026-06-12 13:36
- Reliability follow-up: committed `3fc9082 fix: avoid masking json temp cleanup errors` so `atomic_write_json()` cleanup catches all `OSError` and cannot mask the original write/replace exception. Validation before commit: helper/store tests passed (`34 passed in 0.83s`) and full isolated Docker suite passed (`454 passed, 2 skipped in 11.26s`).
- Browser UX validation for chat-search paging: restarted isolated Docker sandbox on port 18812, created synthetic long Codex session `ux-search-long` with a real Unix control socket and a 160-turn log. Searching `DEEP-NEEDLE-SEARCH` initially showed `0/0 loaded · 1 all` with the Next button enabled; pressing Next loaded older history contiguously and then showed `1/1 loaded · 1 all`, one highlighted hit, body contained the needle, and captured JS errors were `[]`.

## 2026-06-12 13:38
- Consistency fix after launch parser refactor: POST `/api/sessions` parser now uses safe fallback launch defaults when backend config readers fail, matching the degraded-default semantics already used by GET `/api/sessions` and the New Session warning.
- Commit: `3b44ad5 fix: use safe defaults for launch request parsing`.
- Validation: targeted launch/default/reasoning/provenance tests passed (`32 passed in 1.81s`); full isolated Docker suite passed (`459 passed, 2 skipped in 10.52s`).

## 2026-06-12 13:47
- Fresh post-tranche product/architecture reviews found one true blocker: POST Pi launch parsing still reread malformed Pi model config via reasoning validation even when GET launch defaults degraded safely.
- Fix commit: `93d1245 fix: safe-default pi reasoning validation`.
- Validation: targeted launch/default/reasoning/provenance suite passed (`33 passed in 1.71s`); full isolated Docker suite passed (`460 passed, 2 skipped in 13.55s`).
- Review artifacts: `/tmp/codoxear-post-tranche-product-review.md`, `/tmp/codoxear-post-tranche-architect-review.md`.

## 2026-06-12 14:04
- Search paging follow-up from fresh product review: fixed loaded-chat search so pressing Next at the last loaded hit can page older history even when existing loaded hits are present, not only when there are zero loaded matches.
- Mechanism found in browser: the boundary condition entered, but the history request was aborted by the generic scroll-cancel guard because search navigation leaves the chat scrolled near the current hit. Added a narrow `cancelOnScroll` option and call search-driven older loads with `cancelOnScroll: false`; normal older-load cancellation remains enabled by default.
- Commit: `c79d925 fix: page older search matches at boundaries`.
- Validation: `node --check codoxear/static/app.js` passed; targeted chat navigation/scrollback/runtime tests passed (`30 passed in 1.18s`); full isolated Docker suite passed (`460 passed, 2 skipped in 16.34s`).
- Browser evidence in isolated Docker on port 18812 with synthetic `ux-search-boundary` session: initial search `1/1 loaded · 3 all` at turn 149; first Next loaded older turn 080 and showed `1/2 loaded · 3 all`; second Next selected turn 149 as `2/2 loaded · 3 all`; third Next loaded through a nonmatching older page to turn 020 and showed `1/3 loaded · 3 all`, 301 rendered rows, 3 highlighted hits, and captured JS errors `[]`.

## 2026-06-12 14:10
- Refactor follow-up from architecture review: `list_sessions()` updated in-memory recent cwd state but did not persist it after the lock-scope refactor; the out-of-lock history path also had a dead dirty assignment.
- Commit: `6e32194 fix: persist recent cwd updates from session list`.
- Validation: `python3 -m py_compile codoxear/server.py` passed; targeted recent-cwd/sidebar/json-state tests passed (`19 passed in 1.77s`).

## 2026-06-12 14:12
- Additional browser evidence for search boundary paging: in the isolated Docker synthetic `ux-search-boundary` transcript, pressing Prev at the first loaded hit changed `1/1 loaded · 3 all` at turn 149 to `1/2 loaded · 3 all` focused on older turn 080, with 181 rendered rows, 2 highlighted hits, and captured JS errors `[]`.

## 2026-06-12 14:12
- Current-head full validation after recent-cwd persistence and backward search paging evidence: isolated Docker suite passed (`461 passed, 2 skipped in 10.53s`).

## 2026-06-12 14:17
- Mobile UX follow-up: toast feedback was hidden at widths <=520px even though many mobile actions rely on `setToast()`. Changed mobile toast to a compact fixed bottom snackbar and made the toast element an ARIA live status region.
- Commit: `2d208e3 fix: show toast feedback on mobile`.
- Validation: `node --check codoxear/static/app.js` passed; mobile toast source/tooling tests passed (`2 passed in 0.47s`). Browser validation at 390x844 in isolated Docker: non-empty toast computed `display: block`, `position: fixed`, `bottom: 76px`, `pointer-events: none`, role `status`, aria-live `polite`, no composer overlap, JS errors `[]`.

## 2026-06-12 14:19
- Architecture robustness follow-up: broker/server/sessiond all delegate JSONL tail parsing to the shared safe util reader, but sessiond's local wrapper lacked broker's missing-file contract. Aligned sessiond so missing logs return `([], offset)` instead of killing the watcher thread.
- Commit: `e6d38d6 fix: keep sessiond jsonl tailing fail-closed`.
- Validation: `python3 -m py_compile codoxear/sessiond.py` passed; targeted sessiond/broker/shared JSONL reader tests passed (`8 passed in 0.71s`).

## 2026-06-12 14:22
- Architecture cleanup: centralized duplicated process liveness helpers in `codoxear.util` and imported them into server, broker, and sessiond under the existing private helper names.
- Commit: `06469b8 refactor: share process liveness helpers`.
- Validation: `python3 -m py_compile codoxear/util.py codoxear/server.py codoxear/broker.py codoxear/sessiond.py` passed; targeted process/session/stale/resume tests plus source guard passed (`60 passed in 2.75s`).

## 2026-06-12 14:25
- Architecture cleanup: centralized PTY `write_all` and bracketed-paste injection in `codoxear.pty_util`; broker/sessiond retain local wrapper names but delegate to the shared implementation.
- Commit: `c4cae80 refactor: share pty write helpers`.
- Validation: `python3 -m py_compile codoxear/pty_util.py codoxear/broker.py codoxear/sessiond.py` passed; send-ack, PTY source, broker fail-closed, and sessiond fail-closed tests passed (`36 passed in 2.14s`).

## 2026-06-12 14:26
- Current-head full validation after mobile toast UX and architecture helper refactors: isolated Docker suite passed (`465 passed, 2 skipped in 11.23s`).

## 2026-06-12 14:30
- Architecture cleanup: removed broker-local duplicate path matching and centralized rollout filename session-id extraction in `codoxear.util`; server and broker now import the shared helpers.
- Commit: `d819926 refactor: share path helper utilities`.
- Validation: `python3 -m py_compile codoxear/util.py codoxear/server.py codoxear/broker.py` passed; targeted path-helper, session-log, session-resume, broker-proc-rollout, and broker fail-closed tests passed (`68 passed in 2.69s`).

## 2026-06-12 14:32
- Long-chat UX follow-up: added a sparse `/` keyboard shortcut for the existing loaded/all-transcript chat search. Guarded against text-entry targets, no selected session, modifier keys, and open modal surfaces.
- Commit: `a45456e feat: add chat search keyboard shortcut`.
- Validation: `node --check codoxear/static/app.js` passed; chat navigation source tests passed (`8 passed in 0.49s`). Browser validation in isolated Docker: pressing `/` with a session selected changed search bar display from `none` to `flex` and focused `#chatSearchInput` with JS errors `[]`; pressing `/` while `#msg` was focused typed a slash into the composer and kept search closed.

## 2026-06-12 14:34
- Discoverability follow-up: Help now mentions the `/` loaded-chat search shortcut and that Previous/Next can page older history when all-transcript counts show more matches.
- Validation: `node --check codoxear/static/app.js` passed; chat navigation and overlay/help source tests passed (`11 passed in 0.49s`).

## 2026-06-12 14:35
- Current-head full validation after path-helper refactor and chat-search shortcut/help changes: isolated Docker suite passed (`467 passed, 2 skipped in 14.24s`).

## 2026-06-12 14:38
- Rollout-log cleanup: removed local duplicate timestamp/message-id helper definitions from chat-event extraction and delivery-message extraction; both now use the module-level helpers.
- Commit: `a16c762 refactor: reuse rollout chat helper functions`.
- Validation: `python3 -m py_compile codoxear/rollout_log.py` passed; targeted rollout helper, server chat flags, Claude chat/idle, message index, idle heuristic, and voice push tests passed (`68 passed in 2.08s`).

## 2026-06-12 14:39
- Shortcut guard follow-up: `/` search shortcut now also no-ops while the mobile sidebar overlay is open, so it cannot open chat search behind the sidebar.
- Validation: `node --check codoxear/static/app.js` passed; chat navigation source tests passed (`8 passed in 0.47s`).

## 2026-06-12 14:44
- Fresh review follow-up: hardened `/api/sessions/<id>/messages/search` limit parsing so malformed manual/API requests return `400 {\"error\": \"limit must be an integer\"}` instead of leaking `ValueError` through the route exception path.
- Commit: `0773dbe fix: reject malformed chat search limits`.
- Validation: `python3 -m py_compile codoxear/server.py` passed; transcript/search and chat navigation tests passed (`14 passed in 1.83s`). Runtime unit test invokes `Handler.do_GET` for `limit=not-an-int` and observes exactly the 400 response.
- Fresh extended-tranche review artifact: `/tmp/codoxear-extended-tranche-review.md`; no blockers, this malformed-limit hardening was its concrete next fix.

## 2026-06-12 14:47
- Current-head full validation after malformed chat-search limit hardening.
- HEAD: `ff93a4b`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`469 passed, 2 skipped in 10.97s`).

## 2026-06-12 14:48
- Hygiene cleanup from extended review note: removed trailing whitespace from recon markdown files only.
- Commit: `b2bac78 docs: remove recon trailing whitespace`.
- Validation: `git diff --check main -- recon/architecture-review.md recon/git-history-bugs.md` passed before commit.

## 2026-06-12 14:51
- Numeric query hardening: introduced shared bounded integer parsing for session message routes and applied it to `messages/search`, `messages/tail`, and `messages/history`. Malformed `limit` now returns `400 {\"error\": \"limit must be an integer\"}` on all three routes.
- Commit: `f86c67b fix: share message limit validation`.
- Validation: `python3 -m py_compile codoxear/server.py` passed; transcript export/search, chat transcript runtime, scrollback source, and chat navigation source tests passed (`37 passed, 3 subtests passed in 1.94s`). Runtime route regression covers malformed limits for search, tail, and history.

## 2026-06-12 14:58
- Browser-fixture anomaly follow-up: malformed/synthetic Codex logs without session metadata no longer crash session discovery or refresh when a broker sidecar already supplies authoritative session identity and log path.
- Commit: `128eb42 fix: tolerate codex logs without session metadata`.
- Validation: reproduced failure first in `tests/test_stale_sidecars.py`; after fix, `python3 -m py_compile codoxear/server.py` passed and stale-sidecar/sidebar/session-polling/launch-provenance tests passed (`32 passed in 1.84s`).

## 2026-06-12 15:04
- Malformed-log diagnostic cleanup: warning for invalid session metadata is now emitted once per context/path instead of on every poll/sweep.
- Commit: `77365c8 fix: rate limit invalid session metadata warnings`.
- Validation: `python3 -m py_compile codoxear/server.py` passed; stale-sidecar/sidebar/session-polling/launch-provenance tests passed (`33 passed in 1.85s`).

## 2026-06-12 15:05
- Browser validation after malformed-log discovery fix and warning rate-limit, isolated Docker server on port 18813 with a live fixture Unix control socket and Codex log intentionally missing session metadata.
- API evidence: `/api/sessions` listed the fixture session instead of failing; browser/CDP evidence at 390px viewport: before slash `sessionCount=1`, `selected=sidebar-shortcut-fixture`, `sidebarOpen=false`, `searchDisplay=none`; after opening sidebar and pressing `/`: `sidebarOpen=true`, `searchDisplay=none`, `searchValue=\"\"`, `chatRows=2`, JS errors `[]`.
- Sandbox log observation: malformed log metadata produced bounded warnings, not traceback/sweep crash loops.

## 2026-06-12 15:06
- Latest-head full validation after malformed-log discovery fix, invalid-metadata warning rate-limit, and mobile sidebar shortcut browser evidence.
- HEAD: `705fe07`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`472 passed, 2 skipped, 3 subtests passed in 11.12s`).

## 2026-06-12 15:07
- Current-head review follow-up: added direct runtime tests for `_match_session_route()` exact-shape matching and rejection of extra/missing/wrong route segments.
- Commit: `13134cb test: exercise exact session route matching`.
- Validation: route matcher/decomposition/message-route/transcript tests passed (`11 passed, 10 subtests passed in 1.89s`).

## 2026-06-12 15:08
- Exact-current-head full validation after direct route matcher tests/docs.
- HEAD: `bdc82b6`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`474 passed, 2 skipped, 10 subtests passed in 12.98s`).

## 2026-06-12 15:16
- UX/network tranche: decoupled secondary voice/notification polling from the core session-list poll. `refreshSessions()` remains on the 2.5s visible / 15s hidden loop; voice settings, notification subscription state, and notification feed now use a slower 10s visible / 60s hidden loop with immediate refresh on visibility wake. Auth loss and unload stop both timers.
- Commit: `9f06978 feat: decouple secondary UI polling`.
- Validation: `node --check codoxear/static/app.js` passed; session polling, voice push source, and static asset tests passed (`16 passed in 4.67s`).

## 2026-06-12 15:18
- Browser/network validation for secondary polling decoupling, isolated Docker server on port 18814 and headless Chromium/CDP.
- Observation over 6.6s after authenticated app load: API requests were `/api/me`, `/api/settings/voice`, `/api/notifications/subscription`, and three `/api/sessions` requests. Counts: `sessions=3`, `voiceSettings=1`, `notificationSubscription=1`, `notificationFeed=0`, JS errors `[]`.
- Interpretation: session-list freshness remains on the fast visible loop, while secondary voice/notification fetches no longer run on every session tick.

## 2026-06-12 15:19
- Full validation after secondary UI polling decoupling and browser network-cadence evidence.
- HEAD: `66194c1`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`475 passed, 2 skipped, 10 subtests passed in 14.62s`).

## 2026-06-12 15:21
- Long-chat UX: added guarded `Alt+↑`/`Alt+↓` shortcuts for jumping between loaded user messages, sharing the same no-selected-session/text-entry/sidebar/modal guard as chat search. Help now documents the shortcuts.
- Commit: `40b71e1 feat: add user turn keyboard shortcuts`.
- Validation: `node --check codoxear/static/app.js` passed; chat navigation and overlay accessibility source tests passed (`12 passed in 0.49s`).

## 2026-06-12 15:24
- Browser validation for loaded-user-turn keyboard shortcuts, isolated Docker server on port 18815 with live fixture Unix socket and three user turns.
- Observation: with chat focused, CDP `Alt+ArrowDown` using Chromium modifier bit pulsed one loaded `data-role=\"user\"` row, kept chat search closed, and produced JS errors `[]`. With composer `#msg` focused, the same shortcut produced `pulseCount=0`, kept focus on `#msg`, left composer text unchanged, and kept search closed.

## 2026-06-12 15:26
- Full validation after loaded-user-turn shortcut implementation and browser evidence.
- HEAD: `b0d1a15`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`476 passed, 2 skipped, 10 subtests passed in 12.95s`).

## 2026-06-12 15:37
- Network-load tranche: added conditional GET support for `GET /api/sessions`. Server now emits `ETag` and returns 304 for matching `If-None-Match`; client caches the last `/api/sessions` JSON payload and reuses it on 304.
- Commit: `f6dfaa8 feat: add conditional sessions polling`.
- Validation before commit: `python3 -m py_compile codoxear/server.py` passed; `node --check codoxear/static/app.js` passed; auth/session-polling/static-asset/route-decomposition tests passed (`19 passed in 6.29s`).
- Browser/CDP validation, isolated Docker port 18816: `/api/sessions` statuses over 8.5s were 200 then 304, 304, 304; 304 requests carried the matching `If-None-Match`; JS errors `[]`.

## 2026-06-12 15:38
- Full validation after conditional `/api/sessions` polling / ETag support.
- HEAD: `7f3b1c0`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`478 passed, 2 skipped, 10 subtests passed in 12.79s`).

## 2026-06-12 15:44
- Backend refactor tranche: extracted fuzzy file search scoring and git/walk search implementation from `server.py` into `codoxear/file_search.py`. Server keeps a thin wrapper that injects its existing git-root detector, preserving route behavior.
- Commit: `ea96786 refactor: extract file search helpers`.
- Validation: `python3 -m py_compile codoxear/file_search.py codoxear/server.py` passed; file list/search, file-search module boundary, file picker search source, file viewer source, and file inspect tests passed (`46 passed in 2.86s`).

## 2026-06-12 15:45
- Full validation after `codoxear/file_search.py` extraction.
- HEAD: `02b9473`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`479 passed, 2 skipped, 10 subtests passed in 12.75s`).

## 2026-06-12 15:48
- Backend refactor tranche: extracted pure file/content-type classification constants and helpers into `codoxear/file_types.py`; server imports aliases for existing call sites.
- Commit: `a791bf8 refactor: extract file type classification`.
- Validation: `python3 -m py_compile codoxear/file_types.py codoxear/server.py` passed; file type module boundary, file viewer source, file inspect, file upload, and file list tests passed (`51 passed in 2.62s`).

## 2026-06-12 15:50
- Full validation after `codoxear/file_types.py` extraction.
- HEAD: `2ea6413`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`480 passed, 2 skipped, 10 subtests passed in 13.18s`).

## 2026-06-12 15:54
- Backend refactor tranche: extracted text file decoding, strict read, edit-read, and atomic write helpers into `codoxear/file_text.py`; server imports aliases for existing call sites.
- Commit: `e5d045e refactor: extract file text helpers`.
- Validation: `python3 -m py_compile codoxear/file_text.py codoxear/server.py` passed; file text/type module boundary, file viewer source, file inspect, file upload, file list, and transcript export tests passed (`58 passed, 3 subtests passed in 2.57s`).

## 2026-06-12 15:55
- Full validation after `codoxear/file_text.py` extraction.
- HEAD: `e9081e0`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`481 passed, 2 skipped, 10 subtests passed in 12.21s`).

## 2026-06-12 16:02
- Backend refactor tranche: extracted browser-safe video preview payload/cache/transcode helpers into `codoxear/video_preview.py` while preserving `server.VIDEO_PREVIEW_DIR` as the authoritative override via server wrappers.
- Commit: `4c0f7a6 refactor: extract video preview helpers`.
- Validation: `python3 -m py_compile codoxear/video_preview.py codoxear/server.py` passed; focused file viewer/inspect/static tests passed (`44 passed in 10.46s`).

## 2026-06-12 16:03
- Full validation after `codoxear/video_preview.py` extraction.
- HEAD: `f96fe9d`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`481 passed, 2 skipped, 10 subtests passed in 12.17s`).

## 2026-06-12 16:06
- Backend cleanup tranche: removed unused `_repair_png_crc` and `_validate_image` helpers plus now-unused imports from `server.py`; adjusted file-type source boundary test to assert live imports only.
- Commit: `a77f0a8 refactor: remove unused image validation helpers`.
- Validation: `python3 -m py_compile codoxear/server.py` passed; focused file type/viewer/inspect/upload/static tests passed (`53 passed in 6.69s`).

## 2026-06-12 16:07
- Full validation after unused image helper cleanup.
- HEAD: `9fc81d5`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`481 passed, 2 skipped, 10 subtests passed in 12.19s`).

## 2026-06-12 16:11
- Backend refactor tranche: extracted pure file-view metadata/read/download helpers and `ClientFileView` into `codoxear/file_view.py`; kept server route aliases and left HTTP range streaming in `server.py`.
- Commit: `4d42006 refactor: extract file view helpers`.
- Validation: `python3 -m py_compile codoxear/file_view.py codoxear/server.py` passed; focused file view/text/type/viewer/inspect/list/transcript tests passed (`51 passed, 3 subtests passed in 2.46s`).

## 2026-06-12 16:12
- Full validation after `codoxear/file_view.py` extraction.
- HEAD: `3826f47`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`482 passed, 2 skipped, 10 subtests passed in 12.62s`).

## 2026-06-12 16:15
- Backend refactor tranche: extracted byte-range parsing and inline file response streaming into `codoxear/file_response.py`; server retains aliases used by file/blob/video routes.
- Commit: `19319a2 refactor: extract inline file response helpers`.
- Validation: `python3 -m py_compile codoxear/file_response.py codoxear/server.py` passed; focused file response/view/viewer/inspect/static tests passed (`46 passed in 5.75s`).

## 2026-06-12 16:16
- Full validation after `codoxear/file_response.py` extraction.
- HEAD: `ac03de9`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`483 passed, 2 skipped, 10 subtests passed in 12.41s`).

## 2026-06-12 16:19
- Backend refactor tranche: extracted attachment filename/staging/inject-text helpers into `codoxear/file_upload.py`; server wrapper injects `UPLOAD_DIR` and `_now` so existing runtime override semantics remain intact.
- Commit: `abdbcb5 refactor: extract upload staging helpers`.
- Validation: `python3 -m py_compile codoxear/file_upload.py codoxear/server.py` passed; focused upload/module/file-viewer/static/send-ack tests passed (`38 passed in 7.21s`).

## 2026-06-12 16:20
- Full validation after `codoxear/file_upload.py` extraction.
- HEAD: `aee36d1`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`484 passed, 2 skipped, 10 subtests passed in 15.87s`).

## 2026-06-12 16:23
- UI/UX fix: attach button now initializes and syncs as disabled when no session is selected, with title/aria-label `Select a session to attach a file`; selected sessions restore the max-size attach label.
- Commit: `8f9b2a5 fix: disable attach without a selected session`.
- Validation: `node --check codoxear/static/app.js` passed; focused source/static tests passed (`25 passed in 5.81s`). Browser validation on isolated Docker server `:18821` after login observed `#attachBtn.disabled === true`, title/aria-label `Select a session to attach a file`, `#threadTitle === No session selected`, and JS errors `[]`.

## 2026-06-12 16:24
- Full validation after no-session attach button UX fix.
- HEAD: `e174c31`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`485 passed, 2 skipped, 10 subtests passed in 12.56s`).

## 2026-06-12 16:26
- UI/UX fix: queued-messages button now disables when no session is selected and exposes title/aria-label `Select a session to view queued messages`; selected sessions restore `Queued messages`.
- Commit: `67c82e2 fix: disable queue without a selected session`.
- Validation: `node --check codoxear/static/app.js` passed; focused queue/attach/file-viewer/static source tests passed (`25 passed in 8.84s`). Browser validation on isolated Docker server `:18822` after login observed attach disabled/title/aria-label, queue disabled/title/aria-label, `#threadTitle === No session selected`, and JS errors `[]`.

## 2026-06-12 16:28
- Full validation after no-session queue button UX fix.
- HEAD: `606f6a5`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`486 passed, 2 skipped, 10 subtests passed in 12.61s`).

## 2026-06-12 16:31
- UI/UX fix: send button now disables when no session is selected, exposes title/aria-label `Select a session to send`, preserves disabled-while-sending via `syncSendButtonState()`, and keyboard submit with no session shows `select a session first` instead of silently returning.
- Commit: `dced83d fix: disable send without a selected session`.
- Validation: `node --check codoxear/static/app.js` passed; focused send/queue/attach/chat-transcript/file-viewer/static tests passed (`31 passed in 5.77s`). Browser validation on isolated Docker server `:18823` after login observed `#sendBtn.disabled === true`, title/aria-label `Select a session to send`, Ctrl+Enter toast `select a session first`, `#threadTitle === No session selected`, and JS errors `[]`.

## 2026-06-12 16:33
- Full validation after no-session send button UX fix.
- HEAD: `44bc936`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`487 passed, 2 skipped, 10 subtests passed in 12.61s`).

## 2026-06-12 16:35
- UI/UX fix: top title no longer shows pointer cursor or `Edit conversation` tooltip when no session is selected; title affordance now tracks selected-session state.
- Commit: `b2aadd7 fix: gate title edit affordance on selected session`.
- Validation: `node --check codoxear/static/app.js` passed; focused title/send/queue/attach/static tests passed (`12 passed in 5.53s`). Browser validation on isolated Docker server `:18824` after login observed title text `No session selected`, title attribute `No session selected`, cursor `default`, and JS errors `[]`.

## 2026-06-12 16:37
- Full validation after no-session title affordance fix.
- HEAD: `0ba30d6`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`488 passed, 2 skipped, 10 subtests passed in 16.30s`).

## 2026-06-12 16:39
- UI/accessibility fix: selected-session title edit affordance now has role/button keyboard activation for Enter/Space; no-session title removes role/label, uses tabIndex -1, keeps cursor/default and aria-disabled true.
- Commit: `4bb042a fix: make title edit affordance keyboard accessible`.
- Validation: `node --check codoxear/static/app.js` passed; focused title/button-tooltip/static tests passed (`10 passed in 5.50s`). Browser validation on isolated Docker server `:18825` after login observed no-session title role null, aria-label null, aria-disabled `true`, tabIndex `-1`, cursor `default`, and JS errors `[]`.

## 2026-06-12 16:42
- Full validation after title keyboard/ARIA accessibility fix.
- HEAD: `b91c6f5`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`488 passed, 2 skipped, 10 subtests passed in 13.71s`).

## 2026-06-12 16:45
- UI/UX correctness fix from reviewer finding: file attachment upload captures `selected` as `sid` when the file is picked, uses that stable sid for `/inject_file`, captures the attachment index, and only updates visible badge/toast/poll state if the user is still on the same session.
- Commit: `d5e06b9 fix: keep attachment upload bound to picked session`.
- Validation: `node --check codoxear/static/app.js` passed; focused attach/file-viewer/static tests passed (`25 passed in 7.90s`).

## 2026-06-12 16:46
- Full validation after attachment session-race fix.
- HEAD: `b1e6424`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`489 passed, 2 skipped, 10 subtests passed in 10.76s`).

## 2026-06-12 16:50
- UI/UX correctness fix from reviewer finding: New Session start handler now uses `newSessionStartBusy` and disables the start button while `spawnSessionWithCwd` is pending, resetting the guard in `finally`.
- Commit: `f81dc90 fix: guard new session start against duplicate clicks`.
- Validation: `node --check codoxear/static/app.js` passed; focused new-session launch/static tests passed (`20 passed in 7.66s`).

## 2026-06-12 16:51
- Full validation after New Session duplicate-start guard.
- HEAD: `49fee05`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`490 passed, 2 skipped, 10 subtests passed in 10.99s`).

## 2026-06-12 16:54
- UI correctness fix from reviewer finding: diagnostics modal now captures `sid` before fetching, uses `sid` in the request, and skips both success and error rendering if selected session changed before the response.
- Commit: `c61002f fix: ignore stale diagnostics responses after session switch`.
- Validation: `node --check codoxear/static/app.js` passed; focused diagnostics/chat-navigation/static tests passed (`18 passed in 4.76s`).

## 2026-06-12 16:55
- Full validation after diagnostics stale-render guard.
- HEAD: `57b8388`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`491 passed, 2 skipped, 10 subtests passed in 10.97s`).

## 2026-06-12 16:58
- Backend correctness fix from reviewer finding: `sessiond._discover_log()` now allows Pi through backend-aware open-log discovery, so headless Pi sessions can bind to real Pi session logs instead of staying on placeholder metadata.
- Commit: `476b0dd fix: let sessiond bind Pi logs from open process files`.
- Validation: `python3 -m py_compile codoxear/sessiond.py` passed; focused sessiond fail-closed and process-open rollout discovery tests passed (`10 passed in 0.72s`).

## 2026-06-12 17:00
- Full validation after sessiond Pi log-binding fix.
- HEAD: `ae4dfe5`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`492 passed, 2 skipped, 10 subtests passed in 14.29s`).

## 2026-06-12 17:05
- API hardening from reviewer finding: malformed JSON/object bodies now raise `BadRequestError` and return 400 without trace; oversized bodies return `RequestPayloadTooLargeError`/413; generic 500 responses omit trace unless `CODEX_WEB_DEBUG_ERRORS=1`.
- Commit: `a06d362 fix: return bad request for malformed JSON bodies`.
- Validation: `python3 -m py_compile codoxear/server.py` passed; focused client-disconnect/json-body/auth/upload/queue tests passed (`25 passed in 2.11s`). Live isolated Docker curl on `:18826`: malformed `/api/login` and authenticated malformed `/api/sessions/fake/inject_file` both returned 400 with `{"error":"invalid json body"}` and no `trace` field.

## 2026-06-12 17:08
- Full validation after malformed JSON hardening.
- HEAD: `875e87d`.
- Validation: isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`495 passed, 2 skipped, 10 subtests passed in 12.00s`).


## 2026-06-12 17:14
- UI race fix: file saves now capture `saveSessionId`, `savePath`, draft/version state, and use those captured values for the write request; stale success/error responses are ignored if the active file/session changed while the request was in flight.
- Commit: `832f893 fix: ignore stale file save responses`.
- Validation: `node --check codoxear/static/app.js` passed; focused file-viewer/upload/list/session-state tests passed (`32 passed in 2.28s`). Full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`496 passed, 2 skipped, 10 subtests passed in 10.84s`).


## 2026-06-12 17:17
- UI race fix: edit-conversation save now captures the edited session id, uses it in the API URL/title update, disables duplicate saves while pending, and ignores stale success/error UI updates if another edit session is opened before the request resolves.
- Commit: `92fb308 fix: bind edit save responses to the edited session`.
- Validation: `node --check codoxear/static/app.js` passed; focused edit/title/sidebar tests passed (`19 passed in 1.84s`). Full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`498 passed, 2 skipped, 10 subtests passed in 11.25s`).


## 2026-06-12 17:29
- Reviewer finding #1 fix: app logout/auth-loss now has shared `cleanupApp()` ownership that stops message/session/secondary polling, clears pending timers, aborts outstanding controllers, removes renderApp-registered document/window listeners, stops voice/audio timers, and clears `/api/sessions` ETag cache. Message polling now treats 401 as auth loss. Logout tears down even if the POST fails.
- Commit: `6f4a2a7 fix: clean up app pollers on auth loss`.
- Validation: `node --check codoxear/static/app.js` passed; focused auth-cleanup/session-polling/voice/chat/file-viewer source tests passed (`42 passed in 0.71s`). Full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`502 passed, 2 skipped, 10 subtests passed in 11.36s`).


## 2026-06-12 17:33
- Reviewer finding #2 fix: session file downloads now validate metadata with `inspect_downloadable_file()` and stream bytes through `send_attachment_file_response()` instead of `Path.read_bytes()` buffering the entire file before response.
- Commit: `b8c58e3 fix: stream downloaded files instead of buffering`.
- Validation: `python3 -m py_compile codoxear/file_response.py codoxear/file_view.py codoxear/server.py` passed; focused file-response/file-view/file-inspect/file-viewer/file-list tests passed (`46 passed in 3.12s`). Full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`503 passed, 2 skipped, 10 subtests passed in 10.90s`).


## 2026-06-12 17:36
- Reviewer finding #3 fix: notification clicks now await `client.navigate(target)` before focusing the window, and the app hashchange path refreshes `/api/sessions` once when the hash target is absent from the current snapshot before deciding whether the session is selectable.
- Commit: `f181a71 fix: preserve notification target after stale session lists`.
- Validation: `node --check codoxear/static/app.js` and `node --check codoxear/static/service-worker.js` passed; focused voice-push/voice-playback/auth-cleanup/session-polling tests passed (`19 passed in 0.68s`). Full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`505 passed, 2 skipped, 10 subtests passed in 14.11s`).


## 2026-06-12 17:39
- Reviewer finding #4 fix: packaging now exposes `codoxear-sessiond = codoxear.sessiond:main`, and README quick start documents installed command usage for Codex, Pi, and Claude Code headless helper launches.
- Commit: `6d3030e feat: expose sessiond as an installed command`.
- Validation: focused sessiond packaging and sessiond fail-closed tests passed (`7 passed in 0.76s`), including `python -m codoxear.sessiond --help` smoke. Full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`508 passed, 2 skipped, 10 subtests passed in 12.12s`).


## 2026-06-12 17:42
- Reviewer finding #5 fix: video previews now prune `VIDEO_PREVIEW_DIR` after preview reuse/creation using `CODEX_WEB_VIDEO_PREVIEW_MAX_FILES` and `CODEX_WEB_VIDEO_PREVIEW_MAX_BYTES` caps, preserving the current preview and ignoring temporary `.tmp.mp4` files.
- Commit: `cb80a0c fix: prune cached video previews`.
- Validation: `python3 -m py_compile codoxear/video_preview.py codoxear/server.py` passed; focused video-preview/file-inspect/file-viewer tests passed (`40 passed in 2.05s`). Full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`511 passed, 2 skipped, 10 subtests passed in 10.81s`).


## 2026-06-12 17:50
- Clean-room critic blocker repair: attachment streaming now caps emitted bytes to the declared inspected size, and README sessiond examples no longer pass the backend executable as an extra sessiond argument.
- Commit: `a47b4cd fix: address final review blockers`.
- Validation: `python3 -m py_compile codoxear/file_response.py codoxear/server.py` passed; focused file-response/file-inspect/file-view/sessiond-packaging tests passed (`27 passed in 2.07s`). Full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`511 passed, 2 skipped, 10 subtests passed in 10.97s`).


## 2026-06-12 17:54
- Final clean-room adversarial review rerun completed with no blockers.
- HEAD reviewed: `fe170b5`.
- Review artifact: `/tmp/codoxear-final-cleanroom-critic-rerun.md`.
- Reviewer evidence: critic independently inspected latest source/tests and reran isolated Docker validation with `CODOXEAR_DOCKER_PORT=18877 CODOXEAR_DOCKER_ROOT=/tmp/codoxear-final-cleanroom-docker-18877 scripts/codoxear-docker-sandbox test`, passing (`511 passed, 2 skipped, 10 subtests passed in 19.57s`).
- Residual risks noted as non-blocking: concurrent truncation during file download can still emit fewer bytes than precomputed `Content-Length`; real backend startup via installed `codoxear-sessiond` remains outside deterministic evidence.


## 2026-06-12 18:04
- UX/lifecycle tranche: notification hash targets are now retained across later session refreshes while the URL hash still names that session; async polling/settings paths now bail after cleanup if their requests complete after logout/auth loss.
- Commits: `7b673d3 fix: retry notification hash targets after session refresh`, `d660723 fix: ignore async poll results after app cleanup`.
- Validation: targeted source/syntax checks passed (`15 passed in 0.52s` for notification hash; `20 passed in 0.68s` for cleanup guards). Full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`512 passed, 2 skipped, 10 subtests passed in 11.54s`).


## 2026-06-12 18:18
- Product/UX tranche from fresh reviewer findings: redacted saved voice API keys from browser polls, restored mobile browser zoom, made login form keyboard/password-manager accessible, made Settings a keyboard-modal dialog, and made New Session first-use backend selection prefer the selected session backend.
- Commits: `ef29de9 fix: redact saved voice API keys from browser polls`, `459244a fix: allow mobile browser zoom`, `6a042a4 fix: make login submit accessible from keyboard`, `ef5baf1 fix: make settings dialog keyboard-modal`, `6b36112 fix: prefer selected backend for new sessions`, `247550c test: target composer submit source guard`.
- Validation: focused tests passed for each tranche; full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`520 passed, 2 skipped, 10 subtests passed in 15.47s`). Browser evidence on isolated Docker `:18878`: viewport content `width=device-width, initial-scale=1, minimum-scale=1, viewport-fit=cover`; login form present, password autocomplete `current-password`, Enter/requestSubmit reached app (`loginDone=true`); Settings dialog opened with `open=true`, key input value empty, placeholder `Enter API key`, Escape closed it (`open=false`, display `none`); JS errors `[]`.


## 2026-06-12 18:22
- Architecture tranche from fresh architect review: broker and sessiond now share `codoxear/control_socket.py` for JSON-line control-socket dispatch/exception/close handling while retaining backend-specific state, PTY injection, and response payload callbacks.
- Commit: `e2e0f00 refactor: share control socket dispatch`.
- Validation: `python3 -m py_compile codoxear/control_socket.py codoxear/broker.py codoxear/sessiond.py` passed; targeted protocol/broker/sessiond tests passed (`39 passed in 2.14s`). Full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`523 passed, 2 skipped, 10 subtests passed in 14.28s`).


## 2026-06-12 18:25
- Settings UX completion: Settings now restores focus to the opener after close, completing the keyboard-modal behavior.
- Commit: `6618ad2 fix: restore focus after settings dialog closes`.
- Validation: targeted overlay/voice source tests passed (`11 passed in 0.48s`). Full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`523 passed, 2 skipped, 10 subtests passed in 11.20s`). Browser evidence on isolated Docker `:18879`: Settings opened with `open=true`, focus moved inside dialog, API key value empty/placeholder `Enter API key`, Escape closed it, and focus returned to `settingsBtnSide`.


## 2026-06-12 18:34
- Clean-room blocker repair: background voice settings polls no longer sync over an open Settings form, so explicit key clear/base URL edits survive the visible poll interval; voice settings and VAPID private key files are chmodded `0600`.
- Commit: `f0e87d2 fix: protect voice secrets and settings edits`.
- Validation: targeted voice/settings tests passed (`40 passed in 2.92s`). Full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`523 passed, 2 skipped, 10 subtests passed in 11.03s`). Browser evidence on isolated Docker `:18880`: after saving a test key, GET redacted `tts_api_key` and reported `has_tts_api_key=true`; with Settings open and clear checked, after waiting 11.2s the checkbox remained checked; Save cleared the key (`has_tts_api_key=false`, `tts_api_key=""`). File modes inside container: `600 voice_settings.json`, `600 webpush_vapid_private.pem`.


## 2026-06-12 18:44
- Architecture tranche: extracted backend-specific launch argv/env/resume/tmux-inline mapping from `server.py` into `codoxear/backend_launch.py`; server still owns cwd/worktree/resume-live checks, process spawning, launch records, and tmux orchestration.
- Commit: `a792d32 refactor: extract backend launch adapter`.
- Validation: targeted launch adapter/default/request/resume/Claude source tests passed (`55 passed in 1.86s`). Full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`527 passed, 2 skipped, 10 subtests passed in 10.61s`).


## 2026-06-12 18:56
- Product correctness tranche: attachments now fail closed while the selected session is running/sending, and the running-turn `Send after current` option is disabled/blocked when the pending draft has attachments, preventing immediate file injection from being split from queued text.
- Commit: `08b8918 fix: prevent queued drafts from splitting attachments`.
- Validation: targeted JS/source/runtime checks passed (`6 passed in 0.50s`). Full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`528 passed, 2 skipped, 10 subtests passed in 11.05s`). Attempted browser evidence with a fake busy broker was discarded because the fake socket did not reliably report busy state to the server; not used as proof.


## 2026-06-12 18:59
- Performance/UX tranche: all-transcript chat-search count requests are now debounced (`300ms`) while loaded-DOM highlighting remains immediate; cleanup clears the debounce timer and aborts in-flight count requests.
- Commit: `d808f3d fix: debounce full transcript search counts`.
- Validation: targeted chat navigation/auth cleanup source tests passed (`14 passed in 0.93s`). Full isolated Docker suite `scripts/codoxear-docker-sandbox test` passed (`528 passed, 2 skipped, 10 subtests passed in 13.92s`).


## 2026-06-12 19:00
- Validation ownership fix: added `[tool.pytest.ini_options] pythonpath = ["."]` so plain pytest imports the active checkout instead of a stale installed `codoxear` package.
- Commit: `test: make pytest import current checkout`.
- Validation: `python3 -m pytest tests/test_pytest_config.py tests/test_backend_launch_adapter.py -q` passed (`5 passed in 0.48s`); plain full suite `python3 -m pytest -q` passed (`531 passed, 10 subtests passed in 8.98s`).


## 2026-06-12 19:06
- Offline/local UX tranche: Monaco and pdf.js CDN loaders now have bounded timeouts/retry behavior; text/diff file views fall back to a read-only plain-text renderer, and PDF preview failures fall back to an authenticated open/download link.
- Commit: `49bd595 fix: add offline file viewer fallbacks`.
- Validation: targeted file-viewer/static tests passed (`24 passed in 5.04s`). Plain full suite passed (`531 passed, 10 subtests passed in 10.64s`). Full isolated Docker suite passed (`529 passed, 2 skipped, 10 subtests passed in 11.45s`).


## 2026-06-12 19:18
- Clean-room blocker repair: removed third-party font/script execution from `index.html`, added restrictive CSP, changed Monaco/PDF loaders to use self-hosted/local paths or fallback, and chmod-repaired existing `voice_settings.json`/`hmac_secret` on load.
- Commit: `81c2f89 fix: remove third-party app scripts and repair secret modes`.
- Validation: targeted static/file-viewer/auth/voice tests passed (`72 passed in 4.90s`). Plain full suite passed (`534 passed, 10 subtests passed in 9.27s`). Full isolated Docker suite passed (`532 passed, 2 skipped, 10 subtests passed in 11.39s`). Static grep/test confirms no `fonts.googleapis.com`, `fonts.gstatic.com`, `cdn.jsdelivr.net`, third-party script `src`, or third-party stylesheet `href` remains in the app shell/static loaders.


## 2026-06-12 19:25
- Architecture tranche: extracted launch defaults, reasoning/provider/service-tier normalization, Pi/Claude capability parsing, and new-session request validation into `codoxear/launch_config.py`; `server.py` keeps compatibility wrappers over current global paths so existing route/test patch points remain stable.
- Commit: `0d27f37 refactor: extract launch defaults and validation`.
- Validation: targeted reasoning/launch tests passed (`39 passed in 1.92s`). Plain full suite passed (`534 passed, 10 subtests passed in 11.61s`). Full isolated Docker suite passed (`532 passed, 2 skipped, 10 subtests passed in 13.07s`).


## 2026-06-12 19:34
- Post-critic blocker repair: CSP is now emitted as an HTTP `Content-Security-Policy` header for HTML static responses with `X-Frame-Options: DENY`; `_send_static` path containment now uses `Path.relative_to()` instead of a string-prefix check; duplicate server-local launch helper definitions were removed so wrappers delegate consistently to `launch_config`.
- Commit: `1747e2f fix: enforce static CSP headers`.
- Validation: targeted static/launch tests passed (`35 passed in 4.64s`). Plain full suite passed (`534 passed, 10 subtests passed in 9.57s`). Full isolated Docker suite passed (`532 passed, 2 skipped, 10 subtests passed in 11.02s`).


## 2026-06-12 19:40
- Architecture tranche: extracted queue item cleaning, persistence, local list/append/update/delete/move, stale-session pruning, and successful-send removal into `codoxear/queue_store.py`; `SessionManager` still owns broker readiness, idle grace, sending transient fields, and scheduler orchestration.
- Commit: `57e25dd refactor: extract queue store ownership`.
- Validation: direct/integration queue tests passed (`21 passed in 2.40s`). Plain full suite passed (`537 passed, 10 subtests passed in 12.14s`). Full isolated Docker suite passed (`535 passed, 2 skipped, 10 subtests passed in 11.27s`).


## 2026-06-12 19:41
- Runtime CSP header evidence: isolated Docker server on `:18882` returned `HTTP/1.0 200 OK`, `Content-Security-Policy: default-src 'self'; ... frame-ancestors 'none'`, and `X-Frame-Options: DENY` for `GET /`. Initial `HEAD /` returned `501` and was discarded as invalid evidence because the handler does not implement HEAD.


## 2026-06-12 19:51
- Final-review blocker repair: `/inject_file` now checks `MANAGER.attachment_injection_ready(session_id)` before base64 decode/staging/injecting; readiness requires no local queue/sending item and broker state `busy=false`, `queue_len=0`.
- Commit: `fix: reject attachment injection while busy`.
- Validation: targeted server attachment/queue tests passed (`15 passed in 1.82s`). Plain full suite passed (`539 passed, 10 subtests passed in 9.01s`). Full isolated Docker suite passed (`537 passed, 2 skipped, 10 subtests passed in 15.48s`).


## 2026-06-12 19:59
- Final-review blocker rerun repair: attachment readiness now also rejects log-busy sessions (`idle_from_log` false), and final attachment injection rechecks readiness under a per-session input lock shared with `/send`.
- Commit: `fix: serialize attachment and send injection`.
- Validation: targeted server attachment/source tests passed (`17 passed in 1.85s`). Plain full suite passed (`541 passed, 10 subtests passed in 8.89s`). Full isolated Docker suite passed (`539 passed, 2 skipped, 10 subtests passed in 11.27s`).


## 2026-06-12 20:09
- Second clean-room rerun repair: `sessiond` now marks state busy before send ACK; attachment readiness rechecks local queue/sending/log path after broker state refresh; `enqueue()` appends under the same per-session input lock used by send/attachment injection.
- Commit: `fix: close attachment injection race gaps`.
- Validation: targeted attachment/sessiond/send tests passed (`25 passed in 3.07s`). Plain full suite passed (`543 passed, 10 subtests passed in 8.89s`). Full isolated Docker suite passed (`541 passed, 2 skipped, 10 subtests passed in 11.06s`).


## 2026-06-12 20:15
- Final clean-room rerun stale-log repair: attachment readiness refreshes broker sidecar metadata when present before and after broker state, so log-idle veto uses current `log_path` rather than stale in-memory binding.
- Commit: `fix: refresh attachment log metadata`.
- Validation: targeted attachment/sessiond/send tests passed (`26 passed in 2.85s`). Plain full suite passed (`544 passed, 10 subtests passed in 9.73s`). Full isolated Docker suite passed (`542 passed, 2 skipped, 10 subtests passed in 10.68s`).


## 2026-06-12 20:23
- Clean-room rerun deadlock repair: `refresh_session_meta()` now accepts `drain_queue`; attachment readiness refreshes sidecar metadata with `drain_queue=False`, preventing queue-drain/send reentry while `inject_attachment_keys()` holds the per-session input lock.
- Commit: `fix: avoid attachment readiness queue drain`.
- Validation: targeted attachment/sessiond/send tests passed (`27 passed in 2.95s`). Plain full suite passed (`545 passed, 10 subtests passed in 9.14s`). Full isolated Docker suite passed (`543 passed, 2 skipped, 10 subtests passed in 11.33s`).


## 2026-06-12 20:34
- Clean-room rerun pending-attachment repair: successful attachment injection now marks `pending_attachment`; queue/enqueue and unflagged sends reject while pending; the intended web composer send includes `allow_pending_attachment` and clears the marker on successful send.
- Commit: `fix: reserve pending attachments for explicit send`.
- Validation: targeted attachment/chat/send tests passed (`52 passed in 3.13s`). Plain full suite passed (`547 passed, 10 subtests passed in 8.98s`). Full isolated Docker suite passed (`545 passed, 2 skipped, 10 subtests passed in 11.04s`).


## 2026-06-12 20:44
- Clean-room rerun pending-barrier repair: `/send` now checks pending attachment under the per-session input lock; pending attachments persist in `pending_attachments.json`; discovery restores pending flags; web composer only sets `allow_pending_attachment` from local attached-file state.
- Commit: `fix: persist pending attachment ownership`.
- Validation: targeted attachment/chat/send tests passed (`53 passed in 3.03s`). Plain full suite passed (`548 passed, 10 subtests passed in 9.04s`). Full isolated Docker suite passed (`546 passed, 2 skipped, 10 subtests passed in 11.17s`).


## 2026-06-12 20:52
- Clean-room rerun repair: pending-attachment UI recovery now asks explicit confirmation when server metadata says an attachment is pending; `send()` now checks live broker/log readiness under the input lock before socket send so a second send cannot pass during post-ACK PTY commit.
- Commit: `fix: gate pending sends on live readiness`.
- Validation: targeted attachment/chat/send tests passed (`54 passed in 3.09s`). Plain full suite passed (`549 passed, 10 subtests passed in 10.42s`). Full isolated Docker suite passed (`547 passed, 2 skipped, 10 subtests passed in 11.34s`).


## 2026-06-12 21:01
- Clean-room rerun repair: direct sends now reject local queued/sending prompts unless called by queue promotion; send readiness refreshes sidecar metadata before log-idle checks; restored-pending confirmation happens before optimistic UI echo; attachment key-injection error responses no longer set pending state.
- Commit: `fix: harden send and attachment commit boundaries`.
- Validation: targeted queue/attachment/chat/send tests passed (`61 passed in 3.13s`). Plain full suite passed (`552 passed, 10 subtests passed in 8.76s`). Full isolated Docker suite passed (`550 passed, 2 skipped, 10 subtests passed in 11.16s`).


## 2026-06-12 21:09
- Clean-room rerun repair: broker `keys` now returns errors on PTY write failure; broker/sessiond support synchronous `send` commits for pending-attachment sends; server sends `sync=true` when clearing pending attachments; failed sends roll back optimistic local echo/running state.
- Commit: `fix: report input commit failures`.
- Validation: targeted broker/sessiond/attachment/chat tests passed (`65 passed in 3.05s`). Plain full suite passed (`556 passed, 10 subtests passed in 9.35s`). Full isolated Docker suite passed (`554 passed, 2 skipped, 10 subtests passed in 14.37s`).


## 2026-06-12 21:16
- Clean-room rerun repair: broker/sessiond synchronous send failures restore previous busy/turn state, enabling retry of preserved pending attachments; failed optimistic sends remove the entire local message row rather than only its bubble.
- Commit: `fix: rollback failed sync input commits`.
- Validation: targeted sync-commit/UI tests passed (`55 passed in 3.71s`). Plain full suite passed (`556 passed, 10 subtests passed in 8.81s`). Full isolated Docker suite passed (`554 passed, 2 skipped, 10 subtests passed in 11.17s`).


## 2026-06-12 21:22
- Clean-room rerun repair: default async broker/sessiond send failures now restore previous busy/turn state after deferred `_inject` failure while preserving fast ACK behavior.
- Commit: `fix: rollback failed async input commits`.
- Validation: targeted send/attachment/source tests passed (`52 passed in 3.12s`). Plain full suite passed (`558 passed, 10 subtests passed in 8.87s`). Full isolated Docker suite passed (`556 passed, 2 skipped, 10 subtests passed in 11.07s`).


## 2026-06-12 21:28
- Clean-room rerun repair: `SessionManager.send()` now sends `sync: true` for all server-managed sends, so queue/manual success and queue popping use the PTY-write commit boundary rather than fast ACK.
- Commit: `fix: make server sends commit synchronously`.
- Validation: targeted send/queue tests passed (`39 passed in 2.91s`). Plain full suite passed (`559 passed, 10 subtests passed in 8.87s`). Full isolated Docker suite passed (`557 passed, 2 skipped, 10 subtests passed in 10.93s`).


## 2026-06-12 21:35
- Clean-room rerun repair: server-managed synchronous sends no longer use the old 3s socket timeout; `/send` waits for broker/sessiond commit success/error to avoid false failure after PTY write and duplicate queue retries.
- Commit: `fix: avoid timing out committed sends`.
- Validation: targeted send/queue tests passed (`39 passed in 2.91s`). Plain full suite passed (`559 passed, 10 subtests passed in 12.86s`). Full isolated Docker suite passed (`557 passed, 2 skipped, 10 subtests passed in 11.20s`).


## 2026-06-12 21:40
- Final clean-room review at HEAD `adcfd37` returned no demonstrated blockers.
- Review artifact: `/tmp/codoxear-final-cleanroom-rerun14.md`.
- Reviewer independently ran `pytest -q`: `559 tests + 10 subtests` passed.


## 2026-06-12 21:54
- Next-tranche implementation: bounded synchronous send commit waits with explicit commit-unknown semantics.
- Changes: `SEND_COMMIT_TIMEOUT_SECONDS` default 30s; `/send` timeout returns HTTP 504 with `commit_unknown=true`; pending attachments are preserved on unknown; queued items whose dispatch times out are marked `commit_unknown` and not auto-retried; queue viewer shows "Commit unknown" and allows deletion; manual send UI reports "send status unknown; check transcript before retrying" while preserving composer text.
- Validation: targeted tests passed (`66 passed in 2.91s`). Plain full suite passed (`564 passed, 10 subtests passed in 13.66s`). Full isolated Docker suite passed (`562 passed, 2 skipped, 10 subtests passed in 11.13s`).


## 2026-06-12 22:06
- Bounded-send review blocker repair: queued head items are now persisted as `commit_unknown` before dispatch and only popped on confirmed success; non-timeout response loss/empty response maps to `SessionCommitUnknownError`; broker/sessiond sidecars advertise `control_protocol_version=2` with `sync_send`, and server refuses confirmed sends through older sidecars.
- Commit: `fix: harden commit-unknown queue semantics`.
- Validation: targeted tests passed (`70 passed in 3.00s`). Plain full suite passed (`568 passed, 10 subtests passed in 12.17s`). Full isolated Docker suite passed (`566 passed, 2 skipped, 10 subtests passed in 11.40s`).


## 2026-06-12 22:15
- Bounded-send rerun blocker repair: malformed parseable broker send responses now map to `SessionCommitUnknownError` before submitted-message bookkeeping; attachment injection now requires both sync-send and key-write-error capabilities so old brokers cannot create unsendable pending attachments.
- Commit: `fix: close remaining commit-unknown gaps`.
- Validation: targeted tests passed (`73 passed, 3 subtests passed in 2.97s`). Plain full suite passed (`571 passed, 13 subtests passed in 9.24s`). Full isolated Docker suite passed (`569 passed, 2 skipped, 13 subtests passed in 11.44s`).


## 2026-06-12 22:25
- Bounded-send second rerun repair: known broker send failures clear pre-dispatch queue unknown markers; broker-declared partial/unknown write failures include `commit_unknown`; send responses now require strict non-negative integer `queue_len`; stale pending attachments on unsupported brokers can be explicitly cleared via `/pending_attachment/clear` with UI confirmation.
- Commit: `fix: recover stale pending attachments`.
- Validation: targeted tests passed (`76 passed, 7 subtests passed in 2.93s`). Plain full suite passed (`574 passed, 17 subtests passed in 9.21s`). Full isolated Docker suite passed (`572 passed, 2 skipped, 17 subtests passed in 11.23s`).


## 2026-06-12 22:34
- Bounded-send third rerun repair: `_sock_call(..., track_request_sent=True)` now reports whether a send request crossed the socket boundary; post-request socket failures remain `SessionCommitUnknownError` even if PIDs are dead, while pre-request dead sockets can still prune stale sessions.
- Commit: `fix: track dispatched send uncertainty`.
- Validation: targeted tests passed (`78 passed, 7 subtests passed in 3.03s`). Plain full suite passed (`576 passed, 17 subtests passed in 9.06s`). Full isolated Docker suite passed (`574 passed, 2 skipped, 17 subtests passed in 11.80s`).


## 2026-06-12 22:44
- Bounded-send fourth rerun repair: attachment key injection now tracks request-sent response loss and conservatively marks pending on attachment commit-unknown; immediate enqueue promotion returns `commit_unknown` in the API response; enqueue rejects unsupported old brokers before appending undrainable items.
- Commit: `fix: surface attachment and enqueue uncertainty`.
- Validation: targeted tests passed (`81 passed, 7 subtests passed in 2.94s`). Plain full suite passed (`579 passed, 17 subtests passed in 9.05s`). Full isolated Docker suite passed (`577 passed, 2 skipped, 17 subtests passed in 11.30s`).


## 2026-06-12 22:54
- Bounded-send fifth rerun repair: generic pre-dispatch queue failures clear conservative unknown markers; attachment key malformed/empty/post-request failures become commit-unknown and set pending state; attach UI now surfaces attachment unknown and refreshes sessions.
- Commit: `fix: classify attachment commit uncertainty`.
- Validation: targeted tests passed (`83 passed, 7 subtests passed in 3.02s`). Plain full suite passed (`581 passed, 17 subtests passed in 13.19s`). Full isolated Docker suite passed (`579 passed, 2 skipped, 17 subtests passed in 11.29s`).


## 2026-06-12 23:01
- Bounded-send sixth rerun repair: attachment key acknowledgements now require `ok is True`; enqueue no longer records prompts as submitted before confirmed send commit.
- Commit: `fix: tighten attachment ack validation`.
- Validation: targeted tests passed (`86 passed, 11 subtests passed in 3.10s`). Plain full suite passed (`581 passed, 21 subtests passed in 9.36s`). Full isolated Docker suite passed (`579 passed, 2 skipped, 21 subtests passed in 11.06s`).


## 2026-06-12 23:08
- Bounded-send seventh rerun repair: `commit_unknown: true` now overrides success-looking send and attachment responses; enqueue preserves/reports unknown if broker returns success fields plus explicit unknown.
- Commit: `fix: honor explicit commit unknown responses`.
- Validation: targeted tests passed (`89 passed, 11 subtests passed in 2.93s`). Plain full suite passed (`584 passed, 21 subtests passed in 8.86s`). Full isolated Docker suite passed (`582 passed, 2 skipped, 21 subtests passed in 11.11s`).


## 2026-06-12 23:13
- Focused bounded-send clean-room rerun at HEAD `8c89b9e` found no blockers in bounded send/attachment/enqueue commit-unknown paths.
- Review artifact: `/tmp/codoxear-bounded-send-critic-rerun8.md`.
- Reviewer targeted validation: `tests/test_server_queue_persistence.py` passed (`44 passed, 11 subtests passed`).


## 2026-06-12 23:16
- Architecture tranche: session read paths no longer promote queued prompts. Removed queue-drain side effect from `list_sessions()` and changed `refresh_session_meta(..., drain_queue=False)` default; queue sends remain owned by enqueue and queue sweep.
- Commit: `fix: keep session reads non-committing`.
- Validation: targeted tests passed (`68 passed, 11 subtests passed in 1.90s`). Plain full suite passed (`585 passed, 21 subtests passed in 8.82s`). Full isolated Docker suite passed (`583 passed, 2 skipped, 21 subtests passed in 11.17s`).


## 2026-06-12 23:22
- Focused architecture review of read-noncommit tranche at HEAD `ea0a5ac` found no blockers.
- Review artifact: `/tmp/codoxear-read-noncommit-review.md`.
- Reviewer observation: queue commit paths are now explicit enqueue and background sweep only; read-side updates remain metadata/cache/counter writes, not prompt commits.


## 2026-06-12 23:35
- Product recovery tranche: direct `/send` commit-unknown states are now durable and recoverable. Added per-session `commit_unknown_sends.json` state, session row exposure, server-side input blocking until explicit clear, `/commit_unknown_send/clear`, sidebar warning badge, and disabled send/queue controls while unresolved.
- Commit: `03fe3fa fix: persist unknown direct sends`.
- Validation: targeted tests passed (`73 passed, 11 subtests passed in 2.93s`). Plain full suite passed (`586 passed, 21 subtests passed in 11.40s`). Full isolated Docker suite passed (`584 passed, 2 skipped, 21 subtests passed in 11.27s`).
- Browser evidence: isolated Docker container with fake broker and seeded unknown direct send on `http://127.0.0.1:18912/`; after login, snapshot `/tmp/codoxear_unknown_ui_after_login.txt` showed the unknown-send badge plus disabled `Resolve the unknown send before queueing/sending` buttons. After accepting clear, snapshot `/tmp/codoxear_unknown_ui_after_clear.txt` and API dump showed `commit_unknown_send: false` and Send/Queued Messages controls re-enabled. Screenshot artifact: `/tmp/codoxear-unknown-ui-evidence/after-clear.png`.


## 2026-06-12 23:49
- Repaired post-review unknown-commit blockers: direct `commit_unknown_send` now blocks attachment injection/readiness and disables the Attach button; queued `commit_unknown` items now form ordering barriers in `QueueStore.move()` and queue UI move-up affordances.
- Commit: `b7eb79d fix: enforce unknown commit barriers`.
- Validation: targeted tests passed (`64 passed, 11 subtests passed in 1.82s`). Plain full suite passed (`588 passed, 21 subtests passed in 8.88s`). Full isolated Docker suite passed (`586 passed, 2 skipped, 21 subtests passed in 10.93s`).
- Browser evidence: isolated fake-broker Docker on `:18913`; `/tmp/codoxear_barriers_unknown_selected.txt` and `/tmp/codoxear_barriers_unknown_controls.json` show direct-unknown session disables Attach/Queue/Send with explicit labels. `/tmp/codoxear_barriers_queue_viewer.txt` and `/tmp/codoxear_barriers_queue_dom.json` show queued `Commit unknown` head and later item Move Up disabled, preventing reorder-around. Screenshot: `/tmp/codoxear-unknown-barriers-evidence/queue-viewer.png`.


## 2026-06-12 23:59
- Repaired focused rerun blocker: deleting a queued `commit_unknown` item now requires explicit confirmation via `allow_commit_unknown`; API deletion without the flag returns 409. Queue update now rejects `commit_unknown` items to preserve comparison text. UI delete path prompts with transcript/terminal warning and passes the explicit flag only after confirmation.
- Commit: `fix: require explicit queued unknown resolution`.
- Validation: targeted tests passed (`64 passed, 11 subtests passed in 1.80s`). Plain full suite passed (`589 passed, 21 subtests passed in 8.88s`). Full isolated Docker suite passed (`587 passed, 2 skipped, 21 subtests passed in 12.88s`).
- Browser/API evidence: isolated fake-broker Docker `:18914`; direct API delete without flag returned `409 {"error":"commit-unknown item requires explicit confirmation"}`; browser delete cancel preserved both rows and confirmation prompt; browser delete confirm removed only the unknown row and left the later prompt. Artifacts under `/tmp/codoxear-queue-delete-evidence/`.


## 2026-06-13 00:12
- Hardened non-blocking unknown recovery edges from clean-room review: queue move now blocks crossing any persisted `commit_unknown` item in either direction; queue UI one-step moves mirror that barrier; direct send `commit_unknown` immediately patches local session metadata so Send/Queue/Attach disable before the next `/api/sessions` refresh; stale `commit_unknown_sends.json` records are pruned after startup discovery when no active session exists.
- Commit: `fix: harden unknown queue recovery edges`.
- Validation: targeted tests passed (`80 passed, 11 subtests passed in 1.90s`). Plain full suite passed (`591 passed, 21 subtests passed in 8.76s`). Full isolated Docker suite passed (`589 passed, 2 skipped, 21 subtests passed in 10.85s`).


## 2026-06-13 00:17
- Repaired non-blocking findings from unknown-hardening review: `queue/move` rejects boolean `to_index` as a 400 client error; queue move validates out-of-range before barrier checks; missing-session direct unknown pruning now keeps recent orphan markers and only prunes old records, preserving safety through transient discovery gaps.
- Commit: `fix: refine unknown recovery edge handling`.
- Validation: targeted tests passed (`74 passed, 11 subtests passed in 1.81s`). Plain full suite passed (`591 passed, 21 subtests passed in 8.86s`). Full isolated Docker suite passed (`589 passed, 2 skipped, 21 subtests passed in 10.99s`).


## 2026-06-13 00:24
- Repaired cleanup/confirmation blockers from unknown-refinement review: runtime deleted-state cleanup preserves direct and queued unknown recovery markers unless explicit user deletion passes `clear_recovery`; queue sweep missing-session cleanup preserves queues containing `commit_unknown`; queue delete confirmation now requires a JSON boolean `allow_commit_unknown: true` rather than arbitrary truthy values.
- Commit: `fix: preserve unknown recovery markers during cleanup`.
- Validation: targeted tests passed (`60 passed, 11 subtests passed in 1.82s`). Plain full suite passed (`592 passed, 21 subtests passed in 9.01s`). Full isolated Docker suite passed (`590 passed, 2 skipped, 21 subtests passed in 10.95s`).


## 2026-06-13 00:32
- Repaired orphan unknown queue blocker: queue sweep now drains only active-session queues, preserving but skipping missing-session queues that contain `commit_unknown`; orphan unknown queues are listable/deletable through the queue API with the same explicit confirmation rules.
- Commit: `fix: skip orphan unknown queues during sweep`.
- Validation: targeted tests passed (`59 passed, 11 subtests passed in 1.82s`). Plain full suite passed (`593 passed, 21 subtests passed in 8.92s`). Full isolated Docker suite passed (`591 passed, 2 skipped, 21 subtests passed in 10.98s`).


## 2026-06-13 00:46
- Repaired orphan direct unknown blocker and made orphan recovery discoverable: `/commit_unknown_send/clear` now clears orphan direct markers; `/api/sessions` emits synthetic `orphan_recovery` rows for orphan direct/queued unknown evidence; orphan rows do not fetch transcript tails, disable new send/attach/enqueue, and allow queued recovery review.
- Commit: `fix: expose orphan unknown recovery rows`.
- Validation: targeted tests passed (`83 passed, 11 subtests passed in 2.14s`). Plain full suite passed (`596 passed, 21 subtests passed in 9.17s`). Full isolated Docker suite passed (`594 passed, 2 skipped, 21 subtests passed in 11.22s`).
- Browser evidence: isolated Docker `:18917` seeded only orphan direct/queued unknown records. `/tmp/codoxear-orphan-recovery-evidence/codoxear_orphan_direct_browser.json` shows direct orphan row exposes clear badge and disables attach/queue/send; `/tmp/codoxear-orphan-recovery-evidence/codoxear_orphan_queue_controls.json` and `codoxear_orphan_queue_rows.json` show queued orphan row enables review queue, disables send/attach, and renders the `Commit unknown` row. Screenshot: `/tmp/codoxear-orphan-recovery-evidence/queue-orphan.png`.


## 2026-06-13 00:54
- Repaired orphan recovery UI regressions: when a selected recovery row disappears after clearing/deleting its last evidence, `refreshSessions()` clears selection/hash/title/controls; queue button opens orphan queued recovery review before considering composer draft text; orphan recovery rows are handled locally without transcript-tail fetch.
- Commit: `fix: stabilize orphan recovery selection`.
- Validation: targeted tests passed (`73 passed, 11 subtests passed in 1.82s`, then `83 passed, 11 subtests passed in 2.14s` after local orphan handling). Plain full suite passed (`596 passed, 21 subtests passed in 12.14s`). Full isolated Docker suite passed (`594 passed, 2 skipped, 21 subtests passed in 11.02s`).
- Browser evidence: isolated Docker `:18918`; after clearing selected direct orphan marker, `/tmp/codoxear-orphan-recovery-fix-evidence/codoxear_orphan_after_clear_controls.json` shows empty hash, title `No session selected`, and send/attach/queue disabled. With a composer draft on queued orphan, `/tmp/codoxear-orphan-recovery-fix-evidence/codoxear_orphan_queue_with_draft_rows.json` shows queue viewer open with `Commit unknown` row.


## 2026-06-13 01:07
- Repaired orphan queue leftover blocker: deleting a `commit_unknown` item from an orphan queue now marks remaining items as `orphan_recovery`, keeps the orphan row visible/listable, and continues to disable send while allowing review/deletion. Generic stale queues without recovery markers are still pruned.
- Commit: `fix: preserve orphan queue leftovers`.
- Validation: targeted tests passed (`85 passed, 11 subtests passed in 2.06s`, then `80 passed, 11 subtests passed in 1.78s` after marker narrowing). Plain full suite passed (`597 passed, 21 subtests passed in 8.88s`). Full isolated Docker suite passed (`595 passed, 2 skipped, 21 subtests passed in 11.22s`).
- Browser evidence: isolated Docker `:18919`; after confirming deletion of the orphan queue's unknown item, `/tmp/codoxear-orphan-leftover-evidence/codoxear_orphan_leftover_after_delete.json` shows row still selected with queue label `Review preserved queued recovery items`, remaining row `later orphan`, and send disabled as missing-session review-only.


## 2026-06-13 01:14
- Repaired orphan recovery blockers: queued `orphan_recovery` items now block queue promotion like commit-unknown items; sidebar delete clears direct/queued orphan recovery rows; orphan rows include `transcript_state: failed`; queue UI treats orphan recovery items as locked recovery evidence; mixed direct+queue orphan rows keep queue review available when `queue_len > 0`.
- Commit: `fix: block orphan recovery prompts from auto-send`.
- Validation: targeted tests passed (`82 passed, 11 subtests passed in 1.81s`). Plain full suite passed (`599 passed, 21 subtests passed in 8.61s`). Full isolated Docker suite passed (`597 passed, 2 skipped, 21 subtests passed in 14.10s`).


## 2026-06-13 01:22
- Repaired orphan recovery deletion/mutation blocker: `orphan_recovery` queue items now require explicit `allow_orphan_recovery` confirmation for deletion, block update/move/reorder barriers server-side, and show a Recovery tag/confirmation in the queue UI.
- Commit: `fix: require explicit orphan recovery deletion`.
- Validation: targeted tests passed (`68 passed, 11 subtests passed in 1.94s`). Plain full suite passed (`601 passed, 21 subtests passed in 9.22s`). Full isolated Docker suite passed (`599 passed, 2 skipped, 21 subtests passed in 11.53s`).


## 2026-06-13 01:29
- Repaired orphan later-item blocker: when a missing-session queue has recovery evidence, all listed items are exposed as recovery-protected; deleting any later item requires `allow_orphan_recovery`; queue route recovery errors now map to 409; after deleting the last recovery row the queue viewer hides instead of surfacing a false 404 error.
- Commit: `fix: protect all orphan recovery queue items`.
- Validation: targeted tests passed (`68 passed, 11 subtests passed in 2.43s`). Plain full suite passed (`601 passed, 21 subtests passed in 11.22s`). Full isolated Docker suite passed (`599 passed, 2 skipped, 21 subtests passed in 15.22s`).


## 2026-06-13 01:36
- Repaired non-blocking orphan recovery issues: missing-session recovery queue update/move now return recovery conflict semantics instead of 404, and deleting the last queue item hides the queue modal even if a direct orphan marker remains for the same row.
- Commit: `fix: smooth orphan recovery queue conflicts`.
- Validation: targeted tests passed (`58 passed, 11 subtests passed in 1.84s`). Plain full suite passed (`601 passed, 21 subtests passed in 8.93s`). Full isolated Docker suite passed (`599 passed, 2 skipped, 21 subtests passed in 11.05s`).


## 2026-06-13 01:42
- Repaired final orphan queue blocker: deleting any explicit recovery item from a missing-session queue now marks all remaining items as `orphan_recovery`, so later plain items stay listable/protected after the marked item is removed.
- Commit: `fix: propagate orphan recovery after deletion`.
- Validation: targeted `tests/test_server_queue_persistence.py` passed (`57 passed, 11 subtests passed in 1.87s`). Plain full suite passed (`602 passed, 21 subtests passed in 12.25s`). Full isolated Docker suite passed (`600 passed, 2 skipped, 21 subtests passed in 11.58s`).


## 2026-06-13 01:49
- Repaired prune-race recovery blocker: explicit recovery deletion now marks remaining queue tails even if the session is still present but stale; direct unknown markers also preserve same-session queued tails by marking them recovery before missing-session cleanup/sweep.
- Commit: `fix: preserve recovery queue tails across prune races`.
- Validation: targeted tests passed (`66 passed, 11 subtests passed in 1.84s`). Plain full suite passed (`604 passed, 21 subtests passed in 8.99s`). Full isolated Docker suite passed (`602 passed, 2 skipped, 21 subtests passed in 11.16s`).


## 2026-06-13 01:58
- Repaired direct-unknown marker boundary blockers: same-session queued tails are durably marked `orphan_recovery` before a direct unknown marker is cleared, age-pruned, or used during deleted-session cleanup/sweep; sweep now saves marking changes even when no queue was dropped.
- Commit: `fix: persist queue recovery before clearing markers`.
- Validation: targeted tests passed (`68 passed, 11 subtests passed in 1.85s`). Plain full suite passed (`606 passed, 21 subtests passed in 9.05s`). Full isolated Docker suite passed (`604 passed, 2 skipped, 21 subtests passed in 14.12s`).


## 2026-06-13 02:11
- Implemented active recovery queue visibility: active sessions with stored `orphan_recovery` queue items now expose `queue_recovery: true`; sidebar shows a recovery badge; send/attach are disabled with recovery-specific labels; the queue button opens the recovery viewer before enqueueing composer text.
- Commit: `fix: surface active recovery queues`.
- Validation: targeted source/server tests passed (`64 passed, 11 subtests passed in 1.84s`). Plain full suite passed (`606 passed, 21 subtests passed in 9.22s`). Full isolated Docker suite passed (`604 passed, 2 skipped, 21 subtests passed in 11.23s`).
- Browser evidence on isolated Docker `:18922` with fake active session `active-recovery`: API row showed `queue_len: 2`, `queue_recovery: true`; UI snapshot showed attach/send disabled as recovery actions and queue button labeled `Review preserved queued recovery items`; queue viewer showed both queued messages locked with Recovery tags. Artifacts: `/tmp/codoxear-active-recovery-evidence/main-snapshot.txt`, `/tmp/codoxear-active-recovery-evidence/queue-snapshot.txt`, `/tmp/codoxear-active-recovery-evidence/dom.json`, `/tmp/codoxear-active-recovery-evidence/active-recovery-queue.png`.


## 2026-06-13 02:18
- Repaired active recovery visibility review blockers: `/enqueue` now rejects active queues containing recovery/commit-unknown barriers before appending; active `queue_recovery` now includes queued `commit_unknown` items as well as `orphan_recovery` items.
- Commit: `fix: block enqueue behind recovery queues`.
- Validation: targeted tests passed (`66 passed, 13 subtests passed in 1.85s`). Plain full suite passed (`608 passed, 23 subtests passed in 9.31s`). Full isolated Docker suite passed (`606 passed, 2 skipped, 23 subtests passed in 11.21s`).


## 2026-06-13 02:25
- Repaired atomicity blocker: enqueue now rechecks recovery/unknown queue barriers inside the append lock, closing the interleaving where a queue sweep could mark a head `commit_unknown` after enqueue's first check but before append.
- Commit: `fix: recheck recovery barrier before enqueue append`.
- Validation: targeted `tests/test_server_queue_persistence.py` passed (`64 passed, 13 subtests passed in 1.80s`). Plain full suite passed (`609 passed, 23 subtests passed in 9.20s`). Full isolated Docker suite passed (`607 passed, 2 skipped, 23 subtests passed in 11.09s`).


## 2026-06-13 02:33
- Repaired queue-wide recovery blocker: queue promotion now freezes when any queued item has `commit_unknown` or `orphan_recovery`, not only when the head is marked; active `queue_recovery` also reports direct-unknown plus queued-tail state, and internal `_queue_enqueue_local()` uses the protected append path.
- Commit: `fix: freeze promotion for recovery queues`.
- Validation: targeted tests passed (`69 passed, 15 subtests passed in 1.87s`). Plain full suite passed (`611 passed, 25 subtests passed in 9.20s`). Full isolated Docker suite passed (`609 passed, 2 skipped, 25 subtests passed in 11.06s`).


## 2026-06-13 02:42
- Repaired queue-wide mutation blocker: queue listing now marks unflagged items as recovery-protected whenever the queue has recovery evidence; update/move reject the whole recovery queue; delete of any item in such a queue requires explicit recovery confirmation; queue promotion checks the recovery barrier before broker readiness I/O.
- Commit: `fix: protect every item in recovery queues`.
- Validation: targeted tests passed (`70 passed, 15 subtests passed in 1.80s`). Plain full suite passed (`612 passed, 25 subtests passed in 12.63s`). Full isolated Docker suite passed (`610 passed, 2 skipped, 25 subtests passed in 11.18s`).


## 2026-06-13 03:08
- Investigated user report: ffmpeg video transcoding does not work / never worked.
- Observation: host has `/usr/bin/ffmpeg` and `/usr/bin/ffprobe`; direct helper test passed locally (`tests/test_file_inspect.py::TestInspectOpenableFile::test_video_preview_transcodes_to_browser_safe_mp4`, `1 passed`).
- Observation: standard Docker sandbox previously lacked ffmpeg, so the transcode test was skipped there and prior Docker validation did not constrain this workflow.
- Isolated route evidence with ffmpeg installed: `/api/files/video_preview` for synthetic `.mkv` returned HTTP 200 MP4; `ffprobe` showed `codec_name=h264`, `pix_fmt=yuv420p`.
- Browser evidence on isolated Docker `:18932`: Chromium reported `video/x-matroska` as playable enough that a `canPlayType()` heuristic chose the original `.mkv`, loaded metadata with width/height `0`, then after the fix the explicit safe-container allowlist chose `/api/files/video_preview` and loaded `160x90` metadata. Artifacts in `/tmp/codoxear-video-transcode-evidence/`.
- Code commit: `fix: make video preview transcoding usable`.
- Validation: targeted local checks passed (`5 passed`); targeted Docker checks passed (`4 passed` and ffmpeg-backed transcode test executed); full local suite passed (`613 passed, 25 subtests passed in 9.54s`); full isolated Docker suite passed (`612 passed, 1 skipped, 25 subtests passed in 11.91s`).


## 2026-06-13 03:17
- Investigated user report: Pi sessions can show/stay busy after interruption.
- Mechanism found: Pi assistant messages with `stopReason: "aborted"` and no text were not treated as terminal turn signals. Log-derived idle stayed non-idle, broker state did not close the turn, and sessiond's lightweight watcher also lacked a Pi aborted branch.
- Code commit: `fix: clear Pi busy state on abort logs`.
- Validation: focused abort regressions passed (`4 passed`); broader idle/busy/sessiond suites passed (`76 passed in 2.14s`); full local suite passed (`617 passed, 25 subtests passed in 9.15s`); full isolated Docker suite passed (`616 passed, 1 skipped, 25 subtests passed in 12.00s`).


## 2026-06-13 03:25
- Applied focused review fixes for video/Pi tranche:
  - Video preview ffmpeg command now scales odd dimensions to even dimensions before `libx264`/`yuv420p`, fixing valid odd-size videos that previously returned preview 500.
  - Sessiond log watcher now applies busy signals in record order instead of aggregating user/end flags out of order.
  - Historical single-event chat extraction now suppresses Pi aborted assistant messages consistently with live abort handling.
- Commit: `fix: harden video and Pi abort edge cases`.
- Validation: focused review-blocker tests passed (`4 passed`); affected suites passed (`108 passed`); full local suite passed (`620 passed, 25 subtests passed in 12.60s`); full isolated Docker suite passed (`619 passed, 1 skipped, 25 subtests passed in 12.31s`).


## 2026-06-13 03:36
- Applied non-blocking focused review follow-up for Pi abort consistency: text-bearing Pi aborted messages are now suppressed in voice/push delivery extraction and sidebar/last-chat timestamp accounting, matching live/history chat extraction.
- Commit: `fix: suppress Pi aborts in adjacent consumers`.
- Validation: targeted adjacent-consumer tests passed (`3 passed`); affected suites passed (`140 passed`); full local suite passed (`622 passed, 25 subtests passed in 9.33s`); full isolated Docker suite passed (`621 passed, 1 skipped, 25 subtests passed in 11.88s`).


## 2026-06-13 03:52
- Audited Claude Code integration using read-only critic artifact `/tmp/codoxear-cc-gap-review.md`.
- Fixed two synthetic blockers that do not require live credentials:
  - literal XML/HTML-looking Claude user prompts are preserved in transcripts instead of being hidden by a broad markup heuristic;
  - Claude `system/turn_duration` no longer marks a turn idle while an assistant `tool_use` lacks a matching user `tool_result`; broker busy state now tracks Claude tool-use IDs too.
- Commit: `fix: harden Claude transcript and tool busy state`.
- Validation: targeted Claude suites passed (`42 passed`); full local suite passed (`625 passed, 25 subtests passed in 9.21s`); isolated Docker suite passed (`624 passed, 1 skipped, 25 subtests passed in 12.50s`).


## 2026-06-13 04:00
- Applied clean-room Claude hardening review fixes from `/tmp/codoxear-cc-hardening-review.md`:
  - `_compute_idle_from_log()` now expands ambiguous Claude `turn_duration` tails when no local turn context is present, so an unmatched tool-use outside the initial 256 KiB tail cannot be closed by a trailing duration row.
  - Final Claude assistant text no longer closes a known unmatched tool-use in log idle or broker state.
  - User rows containing Claude `tool_result` are hidden from chat even if schema drift adds sibling text parts.
- Commit: `fix: close Claude tool idle edge cases`.
- Validation: targeted Claude suites passed (`44 passed`); full local suite passed (`627 passed, 25 subtests passed in 10.13s`); isolated Docker suite passed (`626 passed, 1 skipped, 25 subtests passed in 11.46s`).


## 2026-06-13 04:06
- Applied non-blocking follow-up from `/tmp/codoxear-cc-hardening-review2.md`:
  - malformed Claude `tool_result` rows without `tool_use_id` no longer clear known pending tool IDs; they only clear the unknown-tool sentinel;
  - `_read_jsonl_tail()` now preserves a complete first record when the scan starts exactly on a line boundary.
- Commit: `fix: fail closed on malformed Claude tool results`.
- Validation: targeted Claude/tail suites passed (`46 passed`); full local suite passed (`629 passed, 25 subtests passed in 12.32s`); isolated Docker suite passed (`628 passed, 1 skipped, 25 subtests passed in 11.57s`).


## 2026-06-13 04:15
- Applied blocker fixes from `/tmp/codoxear-cc-hardening-review3.md`:
  - Claude final assistant text now triggers tail expansion when it appears without prior turn context, matching `turn_duration` ambiguity handling.
  - Chat extraction and delivery extraction now track Claude pending tool IDs, so final text while a known tool-use is pending is classified as narration and does not set `turn_end`.
  - Mixed known/malformed tool-use parts now retain the unknown pending sentinel after the known result arrives.
- Commit: `fix: align Claude final extraction with tool state`.
- Validation: targeted Claude suites passed (`50 passed`); full local suite passed (`633 passed, 25 subtests passed in 9.10s`); isolated Docker suite passed (`632 passed, 1 skipped, 25 subtests passed in 11.86s`).


## 2026-06-13 04:26
- Applied blockers from `/tmp/codoxear-cc-hardening-review4.md`:
  - positioned chat tail/history/live extraction now carries Claude pending tool state instead of classifying each record independently;
  - live chat deltas seed pending state from preceding log context, so split tool-use/final windows keep final-looking text as narration and keep `turn_end` false;
  - voice delivery extraction accepts seeded pending state, and server voice scans seed it from the log before each delta.
- Commit: `fix: carry Claude tool state through chat deltas`.
- Validation: targeted positioned/delivery/server tests passed (`54 passed`); full local suite passed (`636 passed, 25 subtests passed in 8.90s`); isolated Docker suite passed (`635 passed, 1 skipped, 25 subtests passed in 15.33s`).


## 2026-06-13 04:35
- Applied blockers from `/tmp/codoxear-cc-hardening-review5.md`:
  - actual `/messages/live` route now seeds `_extract_chat_events()` and positioned extraction with Claude pending tool state from the log before `after_byte`;
  - `_compute_idle_from_log()` now expands terminal-looking Claude final/duration scans unless the current human user turn start is visible, so later assistant context alone cannot hide older pending tools.
- Commit: `fix: seed Claude tool state in live route`.
- Validation: targeted live-route/idle tests passed (`16 passed`); full local suite passed (`637 passed, 25 subtests passed in 8.96s`); isolated Docker suite passed (`636 passed, 1 skipped, 25 subtests passed in 11.81s`).


## 2026-06-13 04:41
- Applied blocker from `/tmp/codoxear-cc-hardening-review6.md`:
  - Claude pending-tool context seeding now scans backward to the current human turn start by default instead of stopping at 8 MiB;
  - `_compute_idle_from_log()` fails closed if a terminal-looking Claude row remains without visible turn start at the configured scan budget.
- Commit: `fix: scan full Claude turn context for pending tools`.
- Validation: targeted large-result/current-turn tests passed (`14 passed`), affected Claude/server suites passed (`57 passed`), full local suite passed (`638 passed, 25 subtests passed in 10.99s`), isolated Docker suite passed (`637 passed, 1 skipped, 25 subtests passed in 13.69s`).


## 2026-06-13 04:50
- Applied blockers from `/tmp/codoxear-cc-hardening-review7.md`:
  - empty `/messages/live` deltas no longer scan backward through the current Claude turn, avoiding large no-op polling costs;
  - idless/malformed Claude tool-use tracking now uses one sentinel per malformed tool call, so one idless result cannot clear multiple pending malformed calls.
- Commit: `fix: avoid empty Claude live context scans`.
- Validation: targeted empty-delta/idless tests passed (`54 passed`); full local suite passed (`641 passed, 25 subtests passed in 16.41s`); isolated Docker suite passed (`640 passed, 1 skipped, 25 subtests passed in 12.71s`).


## 2026-06-13 04:56
- Applied blocker from `/tmp/codoxear-cc-hardening-review8.md`:
  - malformed/idless Claude tool-use sentinels are now unique per parsed assistant row and part, so split-row idless tool calls do not collapse into one pending item.
- Commit: `fix: avoid Claude unknown tool sentinel collisions`.
- Validation: targeted split-row idless tests passed (`54 passed`); full local suite passed (`643 passed, 25 subtests passed in 13.35s`); isolated Docker suite passed (`642 passed, 1 skipped, 25 subtests passed in 14.12s`).


## 2026-06-13 05:08
- Applied blockers from `/tmp/codoxear-cc-hardening-review9.md`:
  - broker now seeds Claude pending tool IDs when binding an existing log, without importing `rollout_log`/voice dependencies;
  - log idle now reconstructs the exact current Claude turn instead of relying only on an 8 MiB tail, so large resolved tool outputs can become idle while unresolved sibling tools remain busy.
- Commit: `fix: seed Claude broker and exact idle state`.
- Validation: targeted broker/large-idle tests passed (`57 passed`); full local suite passed (`645 passed, 25 subtests passed in 10.88s`); isolated Docker suite passed (`644 passed, 1 skipped, 25 subtests passed in 12.92s`).


## 2026-06-13 05:17
- Applied contract-hole fix from `/tmp/codoxear-cc-hardening-review10.md`:
  - broker log-bind seeding now reconstructs full Claude current-turn busy/idle state, not only pending tool IDs;
  - shared `cc_log.cc_current_turn_state_before()` now backs both broker seeding and rollout log idle reconstruction, reducing duplicate pending-state implementations.
- Commit: `fix: seed full Claude broker turn state`.
- Validation: targeted broker bind/large idle tests passed (`4 passed`), affected suites passed (`58 passed`), full local suite passed (`646 passed, 25 subtests passed in 11.14s`), isolated Docker suite passed (`645 passed, 1 skipped, 25 subtests passed in 12.90s`).


## 2026-06-13 05:24
- Applied blocker from `/tmp/codoxear-cc-hardening-review11.md`: when a Claude log binds after a turn already completed, broker now applies the idle seed and closes stale pre-bind busy/turn state.
- Commit: `fix: close stale Claude broker state on idle bind`.
- Validation: targeted broker bind tests passed (`3 passed`), affected suites passed (`58 passed`), full local suite passed (`647 passed, 25 subtests passed in 11.01s`), isolated Docker suite passed (`646 passed, 1 skipped, 25 subtests passed in 13.86s`).


## 2026-06-13 05:34
- Applied blockers from `/tmp/codoxear-cc-hardening-review12.md`:
  - idless Claude tool-use sentinels now use UUID components so broker watcher batches cannot collide via Python object-id reuse;
  - top-level `toolUseResult` rows are classified as tool-result transport, support common ID keys, and can clear a single pending tool when no ID is present and exactly one tool is pending.
- Commit: `fix: harden Claude malformed result tracking`.
- Validation: targeted top-level/idless tests passed (`3 passed`), affected suites passed (`61 passed`), full local suite passed (`650 passed, 25 subtests passed in 11.65s`), isolated Docker suite passed (`649 passed, 1 skipped, 25 subtests passed in 13.98s`).


## 2026-06-13 05:39
- Ran focused clean-room Claude hardening review at HEAD `089a485`: `/tmp/codoxear-cc-hardening-review13.md`.
- Result: no demonstrated blocker in scoped synthetic Claude hardening paths. Reviewer independently ran targeted coverage `tests/test_cc_chat_and_idle.py tests/test_cc_busy_state.py tests/test_broker_busy_state.py -q` with `65 passed`.
- Remaining non-blocking risks require live/schema/performance evidence: unbounded current-turn scan cost on huge active Claude turns, real `toolUseResult` schema variants, and whether live Claude ever omits `tool_use_id` for explicit tool results.


## 2026-06-13 06:01
- Implemented GTD sidebar sectioning in the UI only:
  - sections: Needs review, Now, Waiting, Later;
  - grouping uses existing row fields (`launchFailed`, `orphan_recovery`, `queue_recovery`, `commit_unknown_send`, `blocked`, `snoozed`);
  - no sessions are hidden/collapsed and the existing sort order is preserved inside each section.
- Browser evidence captured against isolated Docker app/session state on port `18931`:
  - `/tmp/codoxear-sidebar-gtd-evidence/dom.json`
  - `/tmp/codoxear-sidebar-gtd-evidence/main-snapshot.txt`
  - `/tmp/codoxear-sidebar-gtd-evidence/sidebar-gtd-desktop.png`
  - `/tmp/codoxear-sidebar-gtd-evidence/mobile-dom.json`
  - `/tmp/codoxear-sidebar-gtd-evidence/sidebar-gtd-mobile.png`
- Validation: `node --check codoxear/static/app.js`; focused source/sidebar tests passed (`31 passed`); full local suite first exposed a non-reproducible Pi token mock call-count failure, but isolated/file reruns passed and subsequent full local suite passed (`653 passed, 25 subtests passed in 11.76s`); isolated Docker suite passed (`652 passed, 1 skipped, 25 subtests passed in 13.44s`).


## 2026-06-13 06:06
- Applied non-blocking sidebar review follow-up from `/tmp/codoxear-sidebar-gtd-review.md`: section headers now have `role="heading"`, `aria-level="2"`, and count-aware `aria-label`; numeric visual counts are `aria-hidden`.
- Commit: `fix: label sidebar section headers accessibly`.
- Validation after follow-up: focused sidebar checks passed (`22 passed`); full local suite passed (`653 passed, 25 subtests passed in 15.01s`); isolated Docker suite passed (`652 passed, 1 skipped, 25 subtests passed in 16.90s`).

## 2026-06-13 06:15 — Sidebar sessions no-op refresh fast path
- Implemented client-side 304 marker for cached `/api/sessions` responses in `codoxear/static/app.js` and made `refreshSessions()` return existing `latestSessions` before sidebar/defaults mutation when the sessions API reports unchanged data.
- During browser evidence capture, observed raw `/api/sessions` still returned 200 on unchanged polls because `time_priority`, `base_priority`, and `final_priority` changed every second. Added server-side sidebar-priority elapsed bucketing (`SIDEBAR_PRIORITY_BUCKET_SECONDS`, default 10s) so ETags are stable within short human-invisible windows while still advancing priority over time.
- Focused validation: `python3 -m py_compile codoxear/server.py`; `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_session_polling_source.py tests/test_session_sidebar_priority.py tests/test_auth_cleanup_source.py tests/test_static_assets.py -q` → `38 passed`.
- Isolated Docker/browser evidence on `codoxear-sandbox-18932`: raw API returned 200 then 304 for `/api/sessions` with `If-None-Match`; browser PerformanceResourceTiming showed `/api/sessions` `responseStatus: 304` and MutationObserver over `.sessions` reported `mutationCount: 0` in the no-change poll window.
- Evidence artifacts: `/tmp/codoxear-sidebar-gtd-evidence/etag2-h1.txt`, `/tmp/codoxear-sidebar-gtd-evidence/etag2-h2.txt` (raw API copied separately if needed), `/tmp/codoxear-sidebar-gtd-evidence/etag-browser-start.json`, `/tmp/codoxear-sidebar-gtd-evidence/etag-browser-after.json`.

## 2026-06-13 06:17 — No-op refresh full validation
- Full local validation after no-op refresh/priority-bucket changes: `python3 -m pytest -q` → `655 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `654 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 06:26 — Deferred mobile-swipe refresh repair
- Clean-room critic found a blocker in the no-op sessions poll fast path: a real 200 `/api/sessions` update can be cached and intentionally deferred while a mobile swipe row is open; closing the swipe then calls `refreshSessions()`, which may receive 304 and previously returned before applying the deferred 200 data to the sidebar DOM.
- Fixed `refreshSessions()` so 304 only early-returns when no `swipeRefreshDeferred` rebuild is pending. If a deferred refresh exists, it renders from `latestSessions` populated by the prior 200 response without mutating defaults/cache state again.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_session_polling_source.py tests/test_session_sidebar_priority.py -q` → `25 passed`.
- Postfix isolated browser evidence: after a real bucket-boundary 200, the next same-bucket poll window produced two `/api/sessions` 304 responses and `.sessions` MutationObserver `mutationCount: 0`. Artifact: `/tmp/codoxear-sidebar-gtd-evidence/etag-browser-postfix2-after.json`.

## 2026-06-13 06:28 — Deferred-refresh full validation
- Full local validation after deferred-refresh repair: `python3 -m pytest -q` → `656 passed, 25 subtests passed`.
- Full isolated Docker validation after deferred-refresh repair: `scripts/codoxear-docker-sandbox test` → `655 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 06:32 — Deferred-refresh flag lifecycle correction
- Second critic pass found the previous repair still cleared `swipeRefreshDeferred` inside `closeOpenSwipe()` before the follow-up `refreshSessions()`, so a 304 could still early-return without applying the deferred cached payload.
- Corrected the lifecycle: `closeOpenSwipe()` leaves `swipeRefreshDeferred` set while it calls `refreshSessions()`; `refreshSessions()` computes `applyingDeferredSwipeRefresh = swipeRefreshDeferred && !openSwipeSessionId`, renders from `latestSessions`, and clears the flag only after the sidebar DOM rebuild begins.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_session_polling_source.py tests/test_session_sidebar_priority.py -q` → `26 passed`.
- Isolated mobile browser reproduction on `codoxear-sandbox-18932`: opened a mobile swipe on the `review` row, edited its alias via API, observed the polling 200 while the DOM still showed the old alias, closed the swipe, then observed the new alias rendered while subsequent `/api/sessions` resources were 304. Artifacts: `/tmp/codoxear-sidebar-gtd-evidence/deferred-swipe-open.json`, `deferred-before-close.json`, `deferred-after-close.json`, `deferred-new-name.txt`.
- Final full local validation: `python3 -m pytest -q` → `657 passed, 25 subtests passed`.
- Final full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `656 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 06:35 — Session poll fast-path final review
- Clean-room critic review of the finalized no-op session poll tranche found no blockers.
- Review command run by critic: `python3 -m pytest tests/test_session_polling_source.py tests/test_session_sidebar_priority.py -q` → `26 passed in 1.81s`.
- Review artifact: `/tmp/codoxear-session-poll-fastpath-review.md`.
- Residual risks noted: source-string JS tests are not full browser unit tests; overlapping manual `refreshSessions()` calls could theoretically race; a render exception after clearing the deferred flag would not automatically retry.

## 2026-06-13 06:39 — Serialized session refreshes
- Implemented a serialized/coalesced `refreshSessions()` wrapper around `refreshSessionsOnce()`: if a refresh is in flight, a new caller sets `sessionsRefreshQueued = true` and awaits the same promise; the wrapper runs one additional refresh before resolving. This prevents overlapping `/api/sessions` responses from applying out of order while preserving the 304/deferred-swipe behavior.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_session_polling_source.py tests/test_session_sidebar_priority.py -q` → `27 passed`.
- Browser evidence on `codoxear-sandbox-18932`: after serialization, a same-bucket `/api/sessions` poll returned 304 and `.sessions` MutationObserver stayed at `mutationCount: 0`. Artifact: `/tmp/codoxear-sidebar-gtd-evidence/serialized-etag2-after.json`.
- Full local validation: `python3 -m pytest -q` → `658 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `657 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 06:43 — Session refresh serialization review
- Clean-room critic review of commit `5afd630 fix: serialize session list refreshes` found no blockers.
- Critic validation: `python3 -m pytest tests/test_session_polling_source.py tests/test_session_sidebar_priority.py -q` → `27 passed`; `node --check codoxear/static/app.js` → syntax OK.
- Review artifact: `/tmp/codoxear-session-refresh-serialization-review.md`.
- Residual risks noted: tests are source-shape rather than runtime async-race tests; a transient active-refresh failure does not immediately retry a queued refresh in the same loop; future direct `api("/api/sessions")` GET calls would bypass the guard, though none exist now.

## 2026-06-13 06:47 — Transcript loading feedback
- Added `renderTranscriptLoading(sessionId)` in `codoxear/static/app.js` for no-cache session opens. It renders a non-transcript `msg-row assistant typing-row transcript-loading-row` with `role="status"` / `aria-live="polite"` and text `Loading transcript…`; existing transcript render/reset paths remove it.
- Cached tails remain immediate: `openSession()` tracks `displayedCachedTail` and only renders the loading row when no valid cached tail was applied.
- Added muted loading bubble styling in `codoxear/static/app.css` and source tests in `tests/test_chat_scrollback_source.py`.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_static_assets.py -q` → `29 passed`.
- Browser evidence on `codoxear-sandbox-18932`: monkeypatched browser `fetch` to delay `/messages/tail`, clicked the `now` session, observed `Loading transcript…` while pending, then observed `loadingRows: 0` and real transcript text after the delayed tail response. Artifacts: `/tmp/codoxear-sidebar-gtd-evidence/transcript-loading-visible.json`, `transcript-loading-after.json`, `transcript-loading-visible.png`.
- Full local validation: `python3 -m pytest -q` → `659 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `658 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 06:50 — Transcript loading feedback review
- Clean-room critic review of commit `38d7092 feat: show transcript loading feedback` found no blockers.
- Critic validation: `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_static_assets.py -q` → `29 passed`; `node --check codoxear/static/app.js` → passed.
- Review artifact: `/tmp/codoxear-transcript-loading-review.md`.
- Residual risks noted: source-shape tests rather than runtime DOM tests; busy sessions show loading row rather than typing dots until tail arrives; if tail fetch fails, the loading row can remain as failed-load state because no explicit error UI was added.

## 2026-06-13 06:58 — Transcript tail error feedback
- Implemented `renderTranscriptLoadError(sessionId, err)` in `codoxear/static/app.js`. It renders a non-transcript `.typing-row.transcript-error-row` with `role="alert"`, clears older-loading state, stops visible typing, marks first paint, and tells the user to select the conversation again to retry.
- `openSession()` now wraps the initial `/messages/tail` call: stale-generation errors are ignored; 401 invokes `handleAppAuthLoss()`; other failures render the transcript error row instead of leaving `Loading transcript…` indefinitely.
- Added `.msg.transcript-error` styling and source assertions in `tests/test_chat_scrollback_source.py`.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_static_assets.py -q` → `30 passed`.
- Browser evidence on isolated `codoxear-sandbox-18933`: monkeypatched browser `fetch` to return a synthetic 503 for `/messages/tail`, selected a session, observed one `.transcript-error-row`, zero loading rows, and text `Could not load transcript. synthetic tail failure Select the conversation again to retry.` Then restored fetch, reselected the conversation, and observed real transcript content with zero error/loading rows. Artifacts: `/tmp/codoxear-tail-error-evidence/error-visible.json`, `retry-after.json`, `error-visible.png`.
- Full local validation: `python3 -m pytest -q` → `660 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `659 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 07:04 — Cached-tail failure preservation repair
- Clean-room critic found a blocker in the initial transcript error implementation: if a valid cached tail had rendered and the authoritative `/messages/tail` refresh then failed, `renderTranscriptLoadError()` cleared the cached transcript rows and older-history affordance. The error row was non-transcript, but its renderer destructively removed transcript UI state.
- Repaired `renderTranscriptLoadError(sessionId, err, { preserveTranscript })`: it removes prior error rows, but only clears transcript DOM/older state when no cached transcript was displayed. `openSession()` now passes `{ preserveTranscript: displayedCachedTail }` on tail-load failure.
- Focused validation after repair: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_static_assets.py -q` → `30 passed`.
- Browser reproduction of the critic path on isolated `codoxear-sandbox-18933`: populated cache with a successful tail load, forced the next `/messages/tail` to synthetic 503, reselected the session, and observed the original non-typing transcript row remained (`nonTypingRows: 1`) while one `.transcript-error-row` was appended and no loading rows remained. Artifacts: `/tmp/codoxear-tail-error-evidence/cache-before-fail.json`, `cache-fail-after.json`, `cache-fail-after.png`.
- Full local validation after repair: `python3 -m pytest -q` → `660 passed, 25 subtests passed`.
- Full isolated Docker validation after repair: `scripts/codoxear-docker-sandbox test` → `659 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 07:10 — Stale 401 auth-loss ordering repair
- Second clean-room critic review found a blocker in the tail failure catch ordering: stale-generation guards ran before `e.status === 401`, so an expired-auth response from an older in-flight tail request could be ignored after a newer session selection changed `pollGen`.
- Reordered 401 handling before stale-generation guards in both `openSession()` initial tail catch and `pollMessages()` catch.
- Added source-order test coverage in `tests/test_chat_scrollback_source.py`.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_static_assets.py -q` → `31 passed`.
- Browser evidence on isolated `codoxear-sandbox-18933`: monkeypatched first `/messages/tail` request to return delayed 401, clicked the same session again before it resolved so the first request became stale, and observed `handleAppAuthLoss()` cleanup to the login screen. Artifact: `/tmp/codoxear-tail-error-evidence/stale-401-after.json`.
- Full local validation after auth-order repair: `python3 -m pytest -q` → `661 passed, 25 subtests passed`.
- Full isolated Docker validation after auth-order repair: `scripts/codoxear-docker-sandbox test` → `660 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 07:14 — Transcript tail failure final review
- Final clean-room critic review of the transcript tail failure/auth tranche found no blockers.
- Critic validation: `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_static_assets.py -q` → `31 passed`; `node --check codoxear/static/app.js` → passed.
- Review artifact: `/tmp/codoxear-transcript-tail-failure-review.md`.
- Residual risks noted: tests are source-structure rather than a full async browser race harness; `openSession(..., { useCache: false })` paths such as jump-to-latest can still clear visible transcript before a failed tail fetch; tail cache is not a durable full scrollback/search snapshot.

## 2026-06-13 07:17 — Forced tail refresh failure fallback
- Implemented fallback for `openSession(..., { useCache: false })` tail failures: after a non-auth, non-stale `/messages/tail` failure, if a matching `sessionTailCache` exists and was not already displayed, `openSession()` applies that cached tail before appending non-transcript error feedback.
- This addresses the residual Jump-to-latest/no-cache refresh case where a failed authoritative fetch could otherwise leave only an error row even though a valid cached transcript was available.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_static_assets.py -q` → `31 passed`.
- Browser evidence on isolated `codoxear-sandbox-18934`: populated cache with a successful transcript load, monkeypatched `/messages/tail` to synthetic 503, clicked Jump to latest (`openSession(..., { useCache: false })`), and observed cached transcript row preserved (`nonTypingRows: 1`), no loading rows, and one non-transcript error row. Artifacts: `/tmp/codoxear-forced-refresh-evidence/before-jump-fail.json`, `after-jump-fail.json`, `after-jump-fail.png`.
- Full local validation: `python3 -m pytest -q` → `661 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `660 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 07:24 — Forced fallback stale-cache gating repair
- Clean-room critic found a blocker in the forced-refresh fallback: automatic identity-mismatch/409 recovery also calls `openSession(..., { useCache: false })`, and falling back to cache there could restore a transcript matching stale sidebar metadata instead of the new authoritative log identity.
- Repaired by adding explicit `fallbackToCacheOnFailure` option to `openSession()`, defaulting false. Only Jump to latest passes `{ useCache: false, fallbackToCacheOnFailure: true }`; automatic mismatch/409 paths remain no-cache/no-fallback.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_static_assets.py -q` → `31 passed`.
- Browser evidence on isolated `codoxear-sandbox-18934` after gating: Jump-to-latest forced tail failure still preserves one cached transcript row and appends one non-transcript error row. Artifact: `/tmp/codoxear-forced-refresh-evidence/gated-after-jump-fail.json`.
- Full local validation: `python3 -m pytest -q` → `661 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `660 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 07:31 — Authoritative tail-cache identity repair
- Clean-room critic found a blocker in cached fallback identity: tail cache snapshots stored `threadId`/`logPath` from `sessionIndex` sidebar metadata rather than authoritative `/messages/tail` or live response payloads. If sidebar metadata lagged, a cache could appear to match stale sidebar state rather than the real transcript identity.
- Added `transcriptIdentityFromData(data, fallback)` and changed `rememberTailSnapshot()` to store identity from tail response payload first, falling back to session metadata only when the payload lacks a field. `appendTailSnapshotEvents()` now accepts `identityData` and live polling passes the live response as identity data.
- Added source tests to pin authoritative tail/live identity use for cache snapshots.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_static_assets.py -q` → `32 passed`.
- Full local validation: `python3 -m pytest -q` → `662 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `661 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 07:35 — Forced tail fallback final review
- Clean-room critic review of gated forced-tail fallback and authoritative tail-cache identity found no blockers.
- Critic validation: `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_static_assets.py -q` → `32 passed`; `node --check codoxear/static/app.js` → passed.
- Review artifact: `/tmp/codoxear-forced-tail-fallback-review.md`.
- Residual risks noted: tests are source-shape rather than full browser race simulations; cache fallback necessarily still matches against latest client `sessionIndex` metadata when the authoritative tail request fails because no fresh tail identity exists on failure.

## 2026-06-13 07:50 — File save conflict recovery UI
- Implemented `renderFileSaveConflict(savePath, message)` in `codoxear/static/app.js`. A stale-version save 409 now preserves the editor draft and renders inline `Reload from disk` and `Keep editing` actions in `fileStatus` instead of only replacing status text.
- `Reload from disk` confirms that the unsaved draft will be discarded, then uses existing `openFilePath(savePath, { line: activeFileLine })` so the file-read path refreshes `activeFileText`, `activeFileVersion`, editability, and editor content only after a successful read. `Keep editing` leaves the draft in place and focuses the editor. No overwrite action was added.
- Added compact `.fileConflictActions` styling and source tests in `tests/test_file_viewer_source.py`.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_file_viewer_source.py tests/test_file_inspect.py tests/test_file_response_module_source.py -q` → `41 passed`.
- Isolated evidence on `codoxear-sandbox-18935`: direct API stale-version write returned HTTP 409 with `{ "conflict": true, "version": <current_version> }`. Browser end-to-end edit/save conflict evidence was attempted, but Monaco timed out in the sandbox and the viewer fell back to read-only plain text, so UI validation for the actions is source-level. Artifacts: `/tmp/codoxear-file-conflict-evidence/api-conflict-headers.txt`, `api-conflict-body.json`, `file-mode-opened-wait.json`.
- Full local validation: `python3 -m pytest -q` → `663 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `662 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 07:57 — File conflict ownership repairs
- Clean-room critic found two blockers in the first file conflict UI: conflict buttons were bound only to path, not the save session, and stale save `finally` cleanup could mutate the currently active file/save UI after the viewer moved or a newer save started.
- Repaired by passing `saveSessionId` into `renderFileSaveConflict()` and guarding Reload/Keep with both `fileViewerSessionId === saveSessionId` and `activeFilePath === savePath`.
- Added `fileSaveSeq` / `activeFileSaveToken`; `saveStillCurrent()` now includes the token, `resetActiveFileBufferState()` clears it, and `finally` only clears pending/read-only/edit-button state if the completing save token is still active.
- Focused validation after repairs: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_file_viewer_source.py tests/test_file_inspect.py tests/test_file_response_module_source.py -q` → `41 passed`.
- Full local validation after repairs: `python3 -m pytest -q` → `663 passed, 25 subtests passed`.
- Full isolated Docker validation after repairs: `scripts/codoxear-docker-sandbox test` → `662 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 08:07 — File conflict atomicity and ownership repair
- Second clean-room critic review found three blockers: server `/file/write` read/version-check/write was not atomic under `ThreadingHTTPServer`; reload failure status after a conflict button was session-unbound after its await; save `finally` cleanup still checked only token, not captured session/path.
- Repaired server update path by adding `_file_write_lock(path)` and wrapping `_read_text_file_for_write()` + version comparison + `_write_text_file_atomic()` in the same per-file lock for existing-file writes.
- Repaired client action/cleanup ownership: conflict Reload/Keep already guard `saveSessionId` + path before action; save `finally` is token-owned and previous repair source tests now pin ownership. Added source test that server update path locks compare-and-write together.
- Focused validation: `python3 -m py_compile codoxear/server.py`; `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_file_viewer_source.py tests/test_file_inspect.py tests/test_file_response_module_source.py -q` → `42 passed`.
- Full local validation: `python3 -m pytest -q` → `664 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `663 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 08:12 — File conflict final review
- Clean-room critic review of the finalized file conflict recovery tranche found no blockers.
- Critic validation: `python3 -m pytest tests/test_file_viewer_source.py tests/test_file_inspect.py tests/test_file_response_module_source.py -q` → `42 passed`; `node --check codoxear/static/app.js` → passed; `python3 -m py_compile codoxear/server.py` → passed.
- Review artifact: `/tmp/codoxear-file-conflict-recovery-review.md`.
- Residual risks noted: per-file write lock is process-local, not OS-level CAS across multiple server processes or external writers; in-flight saves are not cancellable and may still commit server-side after user navigation/discard; `_FILE_WRITE_LOCKS` can grow by distinct path count over a long-lived server.

## 2026-06-13 08:16 — Inline transcript load retry
- Added a `Retry` button inside `renderTranscriptLoadError()` error bubbles. The button is guarded by `selected === sessionId` and calls existing `openSession(sessionId, { useCache: true })`, preserving existing auth/stale/error handling instead of adding a new fetch path.
- Added compact `.transcriptRetryBtn` styling and source assertions in `tests/test_chat_scrollback_source.py`.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_static_assets.py -q` → `32 passed`.
- Browser evidence on isolated `codoxear-sandbox-18936`: monkeypatched first `/messages/tail` to synthetic 503, selected a session, observed one transcript error row with visible `Retry`; clicking Retry restored transcript content and cleared error/loading rows. Artifacts: `/tmp/codoxear-transcript-retry-evidence/error-with-retry.json`, `retry-after.json`, `retry-after.png`.
- Full local validation: `python3 -m pytest -q` → `664 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `663 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 08:20 — Transcript Retry final review
- Clean-room critic review of inline transcript Retry found no blockers.
- Critic validation: `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_static_assets.py -q` → `32 passed`; `node --check codoxear/static/app.js` → passed.
- Review artifact: `/tmp/codoxear-transcript-retry-review.md`.
- Reviewer residual risks noted source-shape tests and no assistive-technology verification; parent browser evidence already covered click behavior in `/tmp/codoxear-transcript-retry-evidence/retry-after.json`.

## 2026-06-13 08:26 — `/api/sessions` non-session helper memoization
- Implemented narrow server-side memoization for `/api/sessions` non-session helpers while leaving `SessionManager.list_sessions()` and backend/busy/log state uncached.
- `_read_new_session_defaults()` now caches by stat signatures of launch config/cache files and returns deep copies so caller mutation cannot poison the cache.
- `_static_asset_version()` now caches by static asset stat signatures and only rereads asset bytes when a signature changes.
- `_tmux_available()` now caches `shutil.which("tmux")` result behind `CODEX_WEB_TMUX_AVAILABLE_TTL_SECONDS` (default 30s).
- Added targeted launch-defaults cache test covering reuse, deep-copy isolation, and invalidation after config mtime/size change.
- Focused validation: `python3 -m py_compile codoxear/server.py`; `python3 -m pytest tests/test_static_assets.py tests/test_launch_defaults.py tests/test_session_polling_source.py -q` → `34 passed`.
- Full local validation: `python3 -m pytest -q` → `665 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `664 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 08:31 — Static asset version cache removal
- Clean-room critic found a blocker in `_static_asset_version()` memoization: same-size content changes with preserved `mtime_ns` could return a stale app version, keeping `/api/sessions` ETags and static asset URLs stale.
- Removed static asset version memoization and restored content-hash computation on every `_static_asset_version()` call. Kept launch-defaults signature cache and tmux TTL cache.
- Focused validation: `python3 -m py_compile codoxear/server.py`; `python3 -m pytest tests/test_static_assets.py tests/test_launch_defaults.py tests/test_session_polling_source.py -q` → `34 passed`.
- Full local validation after repair: `python3 -m pytest -q` → `665 passed, 25 subtests passed`.
- Full isolated Docker validation after repair: `scripts/codoxear-docker-sandbox test` → `664 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 08:34 — Session constants memoization final review
- Clean-room critic review of finalized `/api/sessions` non-session helper memoization found no blockers.
- Critic validation: `python3 -m pytest tests/test_static_assets.py tests/test_launch_defaults.py tests/test_session_polling_source.py -q` → `34 passed`; `python3 -m py_compile codoxear/server.py` → passed; `python3 -m pytest -q` → `665 passed, 25 subtests passed`.
- Review artifact: `/tmp/codoxear-session-constants-memoization-review.md`.
- Residual risks noted: launch defaults use mtime/size signatures, so timestamp-preserving same-size config/cache edits can be missed until mtime/size changes; tmux availability display can be stale for TTL, but launch path still checks directly.

## 2026-06-13 08:56 — Browser storage-denial robustness
- Selected a fresh deterministic UX robustness gap: direct `localStorage` access could throw `SecurityError`/quota errors in hardened/mobile browser contexts and make preference state look like a server-contact boot failure.
- Implemented `optionalLocalStorage()`, `storageGetItem()`, `storageSetItem()`, and `storageRemoveItem()` in `codoxear/static/app.js` and replaced direct preference/session storage calls with the wrapper.
- Added `tests/test_storage_robustness_source.py` with Node VM execution of the real wrapper under a throwing `window.localStorage` getter and throwing storage methods.
- Updated source assertions for selected-session clearing and New Session remembered backend to require wrapper usage.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_storage_robustness_source.py tests/test_new_session_model_options_source.py tests/test_chat_scrollback_source.py tests/test_file_picker_session_state.py -q` → `37 passed`.
- Full local validation: `python3 -m pytest -q` → `668 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `667 passed, 1 skipped, 25 subtests passed`.
- Browser evidence against isolated Docker sandbox (`codoxear-sandbox-storage-18791`, stopped): Playwright injected a throwing `localStorage` getter before app scripts, logged in, and observed main UI rendered with `threadTitle: "No session selected"`, composer and file viewer present, no storage/server-contact errors. Artifact: `/tmp/codoxear-storage-denied-browser.json`.

## 2026-06-13 08:59 — Storage-denial clean-room review
- Clean-room critic review of storage-denied browser robustness found no blockers.
- Critic validation: `node --check codoxear/static/app.js` → passed; focused pytest on requested storage/New Session/scrollback files → `35 passed`; full pytest → `668 passed, 25 subtests passed`.
- Review artifact: `/tmp/codoxear-storage-denial-review.md`.
- Residual risks: browser evidence used no live sessions; denied/quota storage loses preference persistence by design; source regex would miss `window.localStorage.getItem(...)`, but grep found no direct calls outside helper.

## 2026-06-13 09:08 — File picker candidate hierarchy and cache
- Implemented file-picker candidate source metadata for changed files, chat-mentioned paths, and recently opened files.
- Added compact no-query section headers: `Changed files`, `Mentioned in chat`, and `Recently opened`; active search results remain flat to preserve minimal search UX.
- Added a short per-session candidate cache keyed by session id + remembered file list + currently loaded chat file refs. Cache TTL is 15s; forced refresh bypasses it; remembered-file updates invalidate it.
- Added VM/source tests covering source metadata propagation, actual cache reuse/force-refresh behavior with mocked `/git/changed_files`, and section/cache source hooks.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_file_picker_search_source.py tests/test_file_picker_session_state.py tests/test_file_viewer_source.py -q` → `26 passed`.
- Full local validation: `python3 -m pytest -q` → `671 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `670 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 09:14 — File picker review repairs
- Clean-room review found two blockers in the first file-picker candidate implementation:
  1. loaded search results were still sorted by candidate membership before score;
  2. cached `changed` metadata could affect diff-vs-file open mode within the cache TTL.
- Repaired search sorting so score wins before changed/candidate tie-breakers.
- Added `fileCandidateGitStateFresh`: fresh `/git/changed_files` responses can drive diff-mode defaults; cache hits mark git state stale so cached changed flags are visual/candidate metadata only and do not decide open mode.
- Extended tests with the exact ordering counterexample and a VM cache test that verifies fresh→cached→forced freshness transitions.
- Focused validation after repair: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_file_picker_search_source.py tests/test_file_picker_session_state.py tests/test_file_viewer_source.py -q` → `27 passed`.
- Full local validation after repair: `python3 -m pytest -q` → `672 passed, 25 subtests passed`.
- Full isolated Docker validation after repair: `scripts/codoxear-docker-sandbox test` → `671 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 09:17 — File picker final review
- Clean-room re-review after repair commit `e44d548` found no blockers.
- Critic validation: `node --check codoxear/static/app.js` → passed; focused file-picker tests → `27 passed`; full pytest → `672 passed, 25 subtests passed`.
- Review artifact: `/tmp/codoxear-file-picker-candidate-review.md`.
- Residual risks: `fileCandidateGitStateFresh` can remain true during an open viewer if external git state changes; no direct behavioral test for `resolveFileOpenMode()` fresh/non-fresh branch; cache hits intentionally disable automatic diff mode.

## 2026-06-13 09:20 — Jump-to-latest smooth scroll polish
- Implemented opt-in smooth bottom scroll for user-triggered `Jump to latest` only.
- `scrollToBottom({ behavior = "auto" })` now uses `chat.scrollTo({ top, behavior: "smooth" })` only when explicitly requested and reduced motion is not active; all live-tail/layout autoscroll callers keep instant scroll behavior.
- Added source tests ensuring the smooth call appears only in `jumpToLatest()` and that the helper documents live-tail instant behavior.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_chat_navigation_source.py -q` → `33 passed`.
- Full local validation: `python3 -m pytest -q` → `673 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `672 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 09:31 — Jump-to-latest smooth-scroll repair
- Clean-room review of initial smooth Jump-to-latest commit found a blocker: `openSession()`/`renderSessionTail()` scheduled instant bottom scrolls before the post-load smooth scroll, likely neutralizing visible smooth motion.
- Repaired by making tail rendering accept `scrollBehavior`, passing `tailScrollBehavior: "smooth"` from `jumpToLatest()`, and removing the post-open smooth scroll. Live-tail/default paths still use instant scroll.
- Focused validation after repair: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_chat_navigation_source.py -q` → `33 passed`.
- Full local validation after repair: `python3 -m pytest -q` → `673 passed, 25 subtests passed`.
- Full isolated Docker validation after repair: `scripts/codoxear-docker-sandbox test` → `672 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 09:37 — Jump-to-latest autoscroll scheduler repair
- Second clean-room review found a remaining blocker: even after passing `tailScrollBehavior` into `renderSessionTail()`, `renderTranscript()`/`rebuildDecorations()` queued an instant bottom scroll before the smooth tail-render scroll. Busy tails could also queue an instant scroll via `setTyping(true)`.
- Repaired by threading scroll behavior through `renderTranscript()` → `rebuildDecorations()` and through `applySessionRuntimeFromTail()` → `setTyping()`, so the first scheduled bottom correction during Jump-to-latest uses the requested smooth behavior. Default/live callers still use `auto`.
- Added source assertions that `rebuildDecorations()` schedules `scrollToBottom({ behavior: scrollBehavior })`, `renderSessionTail()` passes behavior into `renderTranscript()`, and busy typing uses `typingScrollBehavior`.
- Focused validation after second repair: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_chat_navigation_source.py -q` → `34 passed`.
- Full local validation after second repair: `python3 -m pytest -q` → `674 passed, 25 subtests passed`.
- Full isolated Docker validation after second repair: `scripts/codoxear-docker-sandbox test` → `673 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 09:46 — Jump-to-latest remaining instant-scroll repairs
- Third clean-room review found two remaining instant-scroll paths that could neutralize smooth Jump-to-latest: immediate `kickPoll(0)` after jump could run a live poll during animation, and restored pending local user rows used default instant append autoscroll.
- Repaired by removing the immediate post-jump live poll (the fresh tail open already schedules the normal poll) and threading `scrollBehavior` through `restorePendingUserRowsForSession()` and `appendEvent()` so pending-row restoration during tail render uses the same behavior.
- Updated runtime/source tests for the new `appendEvent(ev, { scrollBehavior })` signature and pending-row behavior propagation.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_chat_navigation_source.py tests/test_chat_transcript_runtime.py -q` → `39 passed`.
- Full local validation: `python3 -m pytest -q` → `674 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `673 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 09:50 — Smooth Jump-to-latest rollback
- Clean-room review repeatedly found hidden instant-scroll schedulers that could neutralize smooth Jump-to-latest: tail render, decoration rebuild, typing row insertion, pending-row restore, immediate/live poll timing, and initial reset behavior.
- Decision: reverted the chat code/tests to the last clean state before the smooth-scroll tranche (`b550f34`) while preserving the evidence ledger. The attempted polish is not worth further patch layering without a dedicated scroll scheduler redesign/runtime harness.
- Focused validation after rollback: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_chat_navigation_source.py tests/test_chat_transcript_runtime.py -q` → `37 passed`.
- Full local validation after rollback: `python3 -m pytest -q` → `672 passed, 25 subtests passed`.
- Full isolated Docker validation after rollback: `scripts/codoxear-docker-sandbox test` → `671 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 09:54 — Smooth Jump rollback review
- Clean-room review of rollback commit `c3fbae1` found no rollback-specific blockers.
- Critic confirmed active Jump-to-latest code/tests match the pre-smooth validated state (`git diff b550f34 c3fbae1 -- codoxear/static/app.js tests/test_chat_scrollback_source.py tests/test_chat_transcript_runtime.py tests/test_chat_navigation_source.py` empty), and no storage/file-picker rollback occurred.
- Critic validation: `node --check codoxear/static/app.js` → passed; focused chat tests → `37 passed`; full pytest rerun → `672 passed, 25 subtests passed`.
- Review artifact: `/tmp/codoxear-smooth-jump-rollback-review.md`.
- Residual risk: one transient unrelated Pi model registry test failed once and passed on isolated/full rerun; no connection found to rollback.

## 2026-06-13 09:58 — Sidebar identical-render skip
- Implemented a conservative sidebar render signature for `/api/sessions` refreshes. If a 200 response produces the same GTD sidebar entries, selected id, and mobile/desktop action mode, the code skips `sessionsWrap.innerHTML = ""` and the card/header rebuild.
- Preserved existing 304 fast path and open-swipe deferral: deferred swipe refreshes still apply after close, and a changed signature still triggers the existing full rebuild path.
- Added source tests for `lastSidebarRenderSignature`, `sidebarRenderSignature()`, the `sidebarUnchanged` guard, and updated deferred-swipe ordering expectations.
- Focused validation: `python3 -m pytest tests/test_session_polling_source.py tests/test_sidebar_gtd_source.py -q` → `13 passed`.
- Full local validation: `python3 -m pytest -q` → `673 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `672 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 10:03 — Sidebar identical-render guard review
- Clean-room review of `f51db2f` found no blockers.
- Critic validation: `node --check codoxear/static/app.js` → passed; focused sidebar/session-polling tests → `13 passed`; full pytest → `673 passed, 25 subtests passed`.
- Review artifact: `/tmp/codoxear-sidebar-identical-render-review.md`.
- Residual risks: source-order tests do not prove runtime DOM mutation behavior; full-session signature is conservative and may reduce the perf win.

## 2026-06-13 10:05 — Sidebar identical-render browser evidence
- Ran an isolated Docker browser check against `codoxear-sandbox-sidebar-18792` with mocked `/api/sessions` returning repeated 200 responses that changed non-sidebar data (`recent_cwds`/launch defaults) while preserving the rendered sidebar signature.
- Playwright attached a `MutationObserver` to `.sessions` after the active session state was applied, waited for another 200 poll, and observed zero sidebar child-list mutations with identical sidebar HTML before/after.
- Browser artifact: `/tmp/codoxear-sidebar-identical-browser.json`.
- Stopped the isolated Docker sandbox after capture.

## 2026-06-13 10:17 — Attachment upload base64 blocker fix
- Fresh critic found a deterministic attachment blocker: `/api/sessions/<sid>/inject_file` used `base64.b64decode(...)` without importing `base64`, causing valid browser uploads to be reported as `400 invalid base64`.
- Added the missing `base64` import in `codoxear/server.py`.
- Added an execution test that instantiates `Handler.do_POST()` for `/api/sessions/sess-1/inject_file`, mocks auth/manager readiness/injection, submits a valid base64 payload, verifies staged bytes, bracketed paste injection text, and 200 response.
- Focused validation: `python3 -m py_compile codoxear/server.py`; `python3 -m pytest tests/test_file_upload.py tests/test_file_upload_module_source.py tests/test_attach_button_source.py -q` → `15 passed`.
- Full local validation: `python3 -m pytest -q` → `674 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `673 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 10:20 — Attachment base64 fix review
- Clean-room review of `6864759` found no blockers.
- Critic validation: `python3 -m py_compile codoxear/server.py` → passed; focused upload/attach tests → `15 passed`; full pytest → `674 passed, 25 subtests passed`.
- Review artifact: `/tmp/codoxear-attachment-base64-review.md`.
- Residual risks: route-level test covers valid payload path only; state changes between readiness check and injection can still stage a file without injection, which is pre-existing behavior.

## 2026-06-13 10:26 — Visible message time chip
- Implemented a compact visual-only `chatTimeChip` for long-chat orientation. It uses `firstVisibleMessageRow().dataset.ts`, existing `dayLabel()`/`time24()`, hides near live tail, and hides while loaded-chat search is open to avoid surface overlap.
- Added CSS for a centered non-interactive overlay chip in `chatWrap`.
- Added source tests covering chip construction, first-visible-row timestamp use, hide/show logic, sync from `syncJumpButton()`, and search open/close synchronization.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_navigation_source.py tests/test_chat_scrollback_source.py -q` → `33 passed`.
- Full local validation: `python3 -m pytest -q` → `675 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `674 passed, 1 skipped, 25 subtests passed`.
- Browser evidence against isolated Docker sandbox (`codoxear-sandbox-timechip-18793`, stopped): mocked 80-message transcript hid chip at live tail and showed `2026-06-10 · 18:45` when scrolled into older loaded messages. Artifact: `/tmp/codoxear-timechip-browser.json`.

## 2026-06-13 10:32 — Visible time chip review repairs
- Clean-room review of the visible-time chip found no normal-flow blockers, but identified two edge risks: stale chip after selected session disappears and possible narrow-mobile overlap with the top chat navigation rail.
- Repaired by syncing/hiding the chip inside `resetChatRenderState()` and moving `.chatTimeChip` to bottom-center in the `max-width: 520px` mobile CSS block.
- Updated source tests for reset synchronization and mobile placement.
- Focused validation after repair: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_navigation_source.py tests/test_chat_scrollback_source.py -q` → `33 passed`.
- Full local validation after repair: `python3 -m pytest -q` → `675 passed, 25 subtests passed`.
- Full isolated Docker validation after repair: `scripts/codoxear-docker-sandbox test` → `674 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 10:36 — Visible time chip re-review and mobile evidence
- Re-review of repaired visible-time chip found no blockers.
- Critic validation: `node --check codoxear/static/app.js`; focused chat tests → `33 passed`; full pytest → `675 passed, 25 subtests passed`.
- Additional mobile browser evidence against isolated Docker sandbox (`codoxear-sandbox-timechip-mobile-18794`, stopped): at 390×844, chip appeared while scrolled up (`2026-06-10 · 18:50`), hid at live tail, did not overlap top chat nav rail, jump button, or composer. Artifact: `/tmp/codoxear-timechip-mobile-browser.json`.

## 2026-06-13 10:46 — Older-history failure feedback and retry
- Added a bounded non-transcript `olderError` feedback row under `Load older messages` with a Retry button.
- Failure behavior: non-409 `/messages/history` errors preserve current loaded rows and `hasOlder`, re-enable the older button, and show `Couldn’t load older messages.` plus Retry. Retry clears the error and calls the same guarded `loadOlderMessages({ auto: false })` path. 409 still reopens the session tail; stale generation/session responses still return without showing error.
- Added CSS for compact older-history error/retry styling.
- Added source coverage in `test_chat_scrollback_source.py` and runtime VM coverage in `test_chat_transcript_runtime.py` for success and failure paths.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_chat_navigation_source.py tests/test_chat_transcript_runtime.py -q` → `40 passed`.
- Full local validation: `python3 -m pytest -q` → `677 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `676 passed, 1 skipped, 25 subtests passed`.
- Browser evidence against isolated Docker sandbox (`codoxear-sandbox-history-retry-18795`, stopped): forced first history request to 503, verified row count stayed 30, inline error+Retry appeared, retry made second history call, prepended 4 older rows, and cleared the error. Artifact: `/tmp/codoxear-history-retry-browser.json`.

## 2026-06-13 10:52 — Older-history 401 repair
- Clean-room review of `feat: retry failed history loads` found no blockers, but noted that a 401 from `/messages/history` would still flow to the new retry chip instead of immediate auth-loss handling.
- Repaired by adding an explicit `e.status === 401` branch before 409/non-409 retry handling in `loadOlderMessages()`, calling `handleAppAuthLoss()` and returning without showing the retry error.
- Added source and runtime VM coverage for 401 auth-loss behavior.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_chat_navigation_source.py tests/test_chat_transcript_runtime.py -q` → `41 passed`.
- Full local validation: `python3 -m pytest -q` → `678 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `677 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 10:56 — Older-history stale 401 handling
- Re-review of the history auth repair found no blockers, but noted stale/cancelled history 401s were still suppressed by stale guards before auth-loss handling.
- Repaired by moving `e.status === 401` handling ahead of the stale session/generation/request guard in `loadOlderMessages()`, matching tail/session polling semantics that auth loss is global.
- Added source ordering coverage and a runtime VM test where `api()` makes the request stale before throwing 401; `handleAppAuthLoss()` still runs and retry UI is not shown.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_chat_scrollback_source.py tests/test_chat_navigation_source.py tests/test_chat_transcript_runtime.py -q` → `42 passed`.
- Full local validation: `python3 -m pytest -q` → `679 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `678 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 10:59 — Older-history retry final re-review
- Final clean-room review of older-history retry/auth handling found no blockers.
- Critic validation: `node --check codoxear/static/app.js`; focused sets → `39 passed` and `49 passed`; full pytest → `679 passed, 25 subtests passed`.
- Review confirmed: history 401 now outranks stale guards; stale non-auth responses remain suppressed; active 409 still reopens; active non-401/non-409 preserves rows/hasOlder and shows retry; Retry calls guarded `loadOlderMessages({ auto: false })`.
- Residual accepted risk: a truly stale history 401 can force login, intentionally matching auth-loss-as-global semantics.

## 2026-06-13 11:05 — New Session modal focus and ARIA
- Added `aria-modal="true"` to the custom New Session dialog.
- Added opener focus capture/restore via `newSessionReturnFocusEl` and `restoreNewSessionFocus()`; all existing close paths (`X`, Cancel, backdrop, Escape, successful start) still call `hideNewSessionDialog()` and now restore focus if the opener is still focusable.
- Added initial focus behavior via `focusNewSessionInitialControl()`: desktop focuses the cwd combobox and preserves cursor-at-end; mobile focuses the stable close button so focus enters the dialog without forcing the keyboard open.
- Preserved the original `hideNewSessionDialog()` signature after full-suite source tests exposed a signature-coupled test.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_overlay_accessibility_source.py tests/test_edit_session_source.py tests/test_launch_ui_source.py tests/test_new_session_model_options_source.py -q` → `20 passed`.
- Full local validation: `python3 -m pytest -q` → `680 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `679 passed, 1 skipped, 25 subtests passed`.
- Browser evidence against isolated Docker sandbox (`codoxear-sandbox-new-session-a11y-18796`, stopped): desktop open focused `#newSessionCwdInput`, mobile open focused `#newSessionCloseBtn`, both had `role=dialog`/`aria-modal=true`, both isolated `.app`, and Escape restored focus to `#newBtn` with isolation removed. Artifact: `/tmp/codoxear-new-session-a11y-browser.json`.

## 2026-06-13 11:08 — New Session accessibility review
- Clean-room review of `ab8468a` found no blockers.
- Critic validation: `node --check codoxear/static/app.js`; focused modal/launch tests → `20 passed`; full pytest → `680 passed, 25 subtests passed`.
- Review confirmed `aria-modal=true`, desktop/mobile initial focus, all close paths routing through `hideNewSessionDialog()`, successful-start close path, and unchanged modal isolation.
- Residual accepted risks: committed tests are source-structure checks; browser artifact covers core open/Escape close behavior but not every close path individually. Focus restoration checks connected/disabled but not full rendered/tabbable visibility for possible future openers.

## 2026-06-13 11:12 — File write lock lifecycle cleanup
- Replaced the unbounded `_FILE_WRITE_LOCKS: dict[str, threading.Lock]` cache with a refcounted context manager: `_FILE_WRITE_LOCKS: dict[str, tuple[threading.Lock, int]]`.
- Waiters increment the refcount before acquiring the per-file lock, so an entry is not removed while another thread is waiting. The entry is removed when the final holder/waiter exits.
- Preserved the route invariant: file write update still locks around conflict read/version check and atomic write.
- Added execution tests for single-use cleanup and concurrent waiter refcounting, plus updated source coverage.
- Focused validation: `python3 -m pytest tests/test_file_write_locks.py tests/test_file_viewer_source.py tests/test_file_inspect.py -q` → `42 passed`.
- Full local validation: `python3 -m pytest -q` → `682 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `681 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 11:16 — File write lock cleanup review
- Clean-room review of `ddcd803` found no blockers.
- Critic validation: focused lock/source test subset → `3 passed`; `tests/test_file_*.py` → `75 passed`; full pytest → `682 passed, 25 subtests passed`.
- Review confirmed route critical section still covers read/version-check/write, waiters increment refcount before blocking, cleanup runs in `finally`, and exceptions are not suppressed.
- Residual accepted risks: locking remains process-local and string-path based; external writers, other server processes, hardlink/path aliases are not serialized. This is pre-existing and outside the server-process per-path invariant.

## 2026-06-13 11:18 — File picker auto-diff freshness coverage
- Added execution-level VM coverage for `resolveFileOpenMode()`.
- Cases pinned: fresh changed diffable file → `diff`; cached/stale changed metadata (including explicit changed flag) → normal `file`; fresh explicit unchanged → `file`; stale markdown with preview preference → `preview`; fresh changed non-diffable kind → `file`.
- No product code changed; this closes the prior review gap that cached changed metadata was source-supported but not directly behavior-tested.
- Focused validation: `python3 -m pytest tests/test_file_picker_search_source.py tests/test_file_picker_session_state.py -q` → `10 passed`.
- Full local validation: `python3 -m pytest -q` → `683 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `682 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 11:21 — File picker freshness coverage review
- Clean-room review of `ffb84bc` found no blockers.
- Critic validation: focused file-picker tests → `10 passed`; full pytest → `683 passed, 25 subtests passed`.
- Review confirmed the VM test extracts the real `resolveFileOpenMode()` helper and covers the intended fresh/stale/explicit/preview/non-diffable branches.
- Residual accepted risks: `inspectSessionFilePath()` and `isDiffableFileKind()` are stubbed, so the test pins open-mode policy rather than full DOM/API integration or file-kind classification.

## 2026-06-13 11:32 — Send/queue 401 auth-loss consistency
- Added explicit 401 handling to user-initiated send/queue API paths before local error UI: `sendText()`, pending-attachment clear inside send recovery, `enqueueComposerText()`, queue delete/move/update, and queue viewer load.
- Behavior: 401 calls `handleAppAuthLoss()` and returns before commit-unknown handling, local send/queue toasts, queue unavailable UI, or queue mutation refresh/error flows. Non-auth behavior is unchanged.
- Added source coverage in `test_auth_cleanup_source.py` for catch-ordering across send/enqueue/queue mutations/queue load.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_auth_cleanup_source.py tests/test_chat_transcript_runtime.py tests/test_send_ack.py tests/test_server_queue_persistence.py -q` → `92 passed, 15 subtests passed`.
- Browser evidence against isolated Docker sandbox (`codoxear-sandbox-send-queue-auth-18797`, stopped): forced `/send` and `/enqueue` to return 401; both rendered login, removed `.app`, and did not show local send/queue error UI. Artifact: `/tmp/codoxear-send-queue-auth-browser.json`.
- Full local validation: `python3 -m pytest -q` → `684 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `683 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 11:38 — Clear unknown-send 401 repair
- Clean-room review of send/queue auth handling found no blockers, but identified adjacent `clearCommitUnknownSend()` 401 handling as still local-toast based.
- Repaired by routing 401 from `/commit_unknown_send/clear` and from its following `refreshSessions()` through `handleAppAuthLoss()` before the local clear-error toast.
- Added source coverage to the send/queue auth-loss test.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_auth_cleanup_source.py tests/test_chat_scrollback_source.py tests/test_send_ack.py tests/test_server_queue_persistence.py -q` → `108 passed, 15 subtests passed`.
- Full local validation: `python3 -m pytest -q` → `684 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `683 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 11:45 — Send-flow follow-up refresh 401 repair
- Final review of send/queue auth handling found no blockers, but noted successful/commit-unknown/pending-clear send flows still logged `refreshSessions()` 401s instead of routing auth loss.
- Repaired auth handling in follow-up refresh catches for attachment commit-unknown, successful send, send commit-unknown, and pending-attachment clear success.
- Updated source tests to require auth-aware refresh catches and updated scrollback send tests that previously pinned console-only refresh failure handling.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_auth_cleanup_source.py tests/test_send_ack.py tests/test_chat_scrollback_source.py -q` → `41 passed`.
- Full local validation: `python3 -m pytest -q` → `684 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `683 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 11:51 — Attachment 401 and queue timer cleanup repair
- Final review of send/queue auth handling found no blockers, but identified two adjacent risks: direct `/inject_file` endpoint 401 still showed attach error, and debounced queue update timers could survive app cleanup.
- Repaired by routing direct attachment upload 401 through `handleAppAuthLoss()` before selected-session/local attach handling.
- Repaired cleanup by clearing `queueUpdateTimers`, `queueMutationLocks`, and `queuePendingDeletes` in `cleanupApp()`.
- Added source coverage for attachment 401 ordering and queue timer/mutation cleanup.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_auth_cleanup_source.py tests/test_attach_button_source.py tests/test_queue_button_source.py tests/test_chat_scrollback_source.py -q` → `35 passed`.
- Full local validation: `python3 -m pytest -q` → `684 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `683 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 11:59 — In-flight queue update cleanup guard
- Review of the send/queue/attachment auth surface found no blockers, but noted that a debounced queue update already in flight at cleanup time could still finish detached-DOM refresh work.
- Repaired `scheduleQueueUpdate()` timer body with `appDisposed` guards before starting, after update API, after queue refresh, after session refresh, before local error toast, and before acting on pending deletes.
- Added source coverage for `appDisposed` guards in the queue update timer path.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_auth_cleanup_source.py tests/test_queue_button_source.py tests/test_server_queue_persistence.py -q` → `75 passed, 15 subtests passed`.
- Full local validation: `python3 -m pytest -q` → `684 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `683 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 12:04 — Disposed queue update late-401 race repair
- Clean-room review found a blocker: an in-flight queue update from a disposed app could reject with 401 and call the old closure's `handleAppAuthLoss()`, potentially tearing down a fresh app after re-login.
- Repaired by making `appDisposed` outrank 401 inside the queue update timer catch. Active queue-update 401 still routes to auth loss; disposed late 401s are ignored.
- Updated source coverage for catch ordering.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_auth_cleanup_source.py tests/test_queue_button_source.py tests/test_server_queue_persistence.py -q` → `75 passed, 15 subtests passed`.
- Full local validation: `python3 -m pytest -q` → `684 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `683 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 12:09 — Auth-loss stale closure guard
- Re-review found the queue-update-specific late-401 race fix did not address the root class: any disposed app closure could call `handleAppAuthLoss()` and have its `renderLogin()` tear down a fresh app after re-login.
- Repaired by making `handleAppAuthLoss()` return immediately when its app instance has already been disposed.
- Updated auth cleanup source coverage for the app-disposed guard.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_auth_cleanup_source.py tests/test_queue_button_source.py tests/test_send_ack.py tests/test_attach_button_source.py -q` → `22 passed`.
- Full local validation: `python3 -m pytest -q` → `684 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `683 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 12:13 — Logout stale closure guard
- Re-review found no blocker in `handleAppAuthLoss()`, but identified the same stale-closure teardown mechanism in logout's async `finally`: a disposed logout closure could still call `renderLogin()` and clean up a fresh app.
- Repaired logout finally with `if (appDisposed) return;` before cleanup/render-login.
- Updated auth cleanup source coverage.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_auth_cleanup_source.py tests/test_login_accessibility_source.py tests/test_auth_cookie.py -q` → `14 passed`.
- Full local validation: `python3 -m pytest -q` → `684 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `683 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 12:17 — Stale auth/logout cleanup final review
- Final clean-room review of stale auth/logout cleanup found no blockers.
- Critic validation: `node --check codoxear/static/app.js`; focused auth/login tests → `14 passed`; full pytest → `684 passed, 25 subtests passed`.
- Review searched direct `renderLogin(renderApp)` call sites and confirmed disposed guards exist in `handleAppAuthLoss()` and logout `finally`; boot `/api/me` 401 path is startup-only.
- Residual accepted risk: some disposed async paths may still run harmless UI syncs against global IDs, but no found path can render login or clean up a fresh app.

## 2026-06-13 12:22 — Queue/Help/Details modal focus parity
- Added `aria-modal="true"` to Queue, Help, and Details custom dialogs.
- Added return-focus slots and shared helpers `restoreModalFocus()` / `focusModalCloseButton()`.
- Opening Queue/Help/Details captures the opener, applies existing modal isolation, and focuses the close button. Hiding restores focus to the opener if still connected/focusable.
- Added source coverage for aria-modal, return-focus capture, initial focus, and restore behavior.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_overlay_accessibility_source.py tests/test_queue_button_source.py tests/test_diagnostics_source.py -q` → `9 passed`.
- Browser evidence against isolated Docker sandbox (`codoxear-sandbox-modal-parity-18798`, stopped): Queue/Help/Details each focused their close button on open, set role/dialog and aria-modal, inert/aria-hidden `.app`, and restored focus to the opener on Escape with isolation removed. Artifact: `/tmp/codoxear-modal-parity-browser.json`.
- Full local validation: `python3 -m pytest -q` → `685 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `684 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 12:30 — Utility modal explicit opener capture
- Clean-room review of utility modal focus parity found no blockers, but noted `document.activeElement` may not reliably be the clicked opener under pointer activation.
- Repaired Queue/Help/Details show handlers to accept `{ opener }` and pass `e.currentTarget` from their click handlers, falling back to `document.activeElement` for programmatic opens.
- Updated source tests and marker tests for the new signatures/call sites.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_auth_cleanup_source.py tests/test_overlay_accessibility_source.py tests/test_queue_button_source.py tests/test_diagnostics_source.py -q` → `15 passed`.
- Browser evidence after repair against isolated Docker sandbox (`codoxear-sandbox-modal-parity-18799`, stopped): Queue/Help/Details still focus close buttons and restore focus to their explicit openers. Artifact: `/tmp/codoxear-modal-parity2-browser.json`.
- Full local validation: `python3 -m pytest -q` → `685 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `684 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 12:33 — Utility modal focus final review
- Final clean-room review of Queue/Help/Details modal focus parity found no blockers.
- Critic validation: `node --check codoxear/static/app.js`; focused overlay/queue/diagnostics tests → `9 passed`; broader `-k 'overlay or queue or diagnostics'` → `97 passed, 15 subtests passed`; full pytest → `685 passed, 25 subtests passed`; worktree clean.
- Review confirmed aria-modal, close-button focus, explicit opener capture, button/backdrop/Escape shared close paths, and Details async load not affecting focus.
- Residual accepted risks: browser evidence covers happy paths, not every opener-disabled/removed/offscreen case; mobile Help from closed sidebar may restore focus to a hidden opener and would need separate mobile evidence if visible-focus-on-mobile becomes a requirement.

## 2026-06-13 20:49 — Unattended popover accessibility
- Added `aria-controls="unattendedMenu"`, `aria-expanded`, and `aria-haspopup="dialog"` to the Unattended button.
- Added open-state synchronization through `setUnattendedMenuExpanded()`, opener capture, Escape close with focus restoration, initial focus on `#unattendedEnabled` after config load, and close-on-deselection behavior.
- Outside click and resize still close the lightweight popover without focus restoration; Escape/toggle/load-failure restore to the opener when appropriate.
- Added source coverage in `tests/test_unattended_mode_source.py` for ARIA state, focus helpers, Escape handling, opener capture, and deselection close.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_unattended_mode_source.py tests/test_overlay_accessibility_source.py tests/test_auth_cleanup_source.py -q` → `18 passed`.
- Browser evidence against isolated Docker sandbox (`codoxear-sandbox-unattended-a11y-18800`, stopped): keyboard Enter opened the popover, focus moved to `#unattendedEnabled` after config load, `aria-expanded` became true, Escape closed/restored focus to `#unattendedBtn`, outside click closed and reset `aria-expanded=false`. Artifact: `/tmp/codoxear-unattended-a11y-browser.json`.
- Full local validation: `python3 -m pytest -q` → `686 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `685 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 20:57 — Unattended stale-load/session guard repair
- Clean-room review of the Unattended popover found no blockers, but identified two stale-state risks: an old failed `/unattended` load could close a newer popover, and programmatic selected-session changes could leave the popover bound to the old session.
- Repaired with `unattendedMenuToken` and `unattendedMenuSessionId`. Hide invalidates pending loads; show records token/session; load success focuses only if token/session/selected still match; stale failures return without closing/toasting. `updateUnattendedBtnState()` now closes the popover if selected session changes or disappears.
- Added source coverage for token/session guards.
- Focused validation after repair: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_unattended_mode_source.py tests/test_overlay_accessibility_source.py tests/test_auth_cleanup_source.py -q` → `18 passed`.
- Additional browser evidence against isolated Docker sandbox (`codoxear-sandbox-unattended-stale-18801`, stopped): delayed first `/unattended` load, closed/reopened, second load succeeded, then first failed; newer popover stayed open with second config and no stale error toast. Artifact: `/tmp/codoxear-unattended-stale-browser.json`.
- Full local validation after repair: `python3 -m pytest -q` → `686 passed, 25 subtests passed`.
- Full isolated Docker validation after repair: `scripts/codoxear-docker-sandbox test` → `685 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 21:06 — Unattended stale success/session-change blocker repair
- Re-review found blockers: token/session guard ran only after `loadUnattendedCfgForSelected()` had already mutated UI state, so stale successful loads could overwrite newer popover fields; session changes could leave an old popover visible before tail load completed.
- Repaired by moving token/session validation into `loadUnattendedCfgForSelected({ sid, openToken })` before any config/UI mutation.
- Repaired by hiding the Unattended popover immediately inside `openSession()` when selected session changes, and by explicitly hiding when the selected session disappears during session refresh.
- Updated source coverage to assert mutation guards precede `unattendedCfg` assignment and that openSession/session-removal paths hide the popover.
- Focused validation after repair: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_unattended_mode_source.py tests/test_overlay_accessibility_source.py tests/test_auth_cleanup_source.py tests/test_chat_scrollback_source.py -q` → `42 passed`.
- Browser evidence for stale successful load against isolated Docker sandbox (`codoxear-sandbox-unattended-stale-success-18803`, stopped): first GET delayed, second open loaded `second open`, first GET later returned stale successful config; newer popover stayed open and fields remained from second load. Artifact: `/tmp/codoxear-unattended-stale-success-browser.json`.
- Full local validation after repair: `python3 -m pytest -q` → `686 passed, 25 subtests passed`.
- Full isolated Docker validation after repair: `scripts/codoxear-docker-sandbox test` → `685 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 21:15 — Unattended save lifecycle scoping repair
- Re-review found the remaining blocker: Unattended saves used one global debounce timer/config, so switching sessions could drop the previous session's pending edit or let a late POST response overwrite the current session's popover state.
- Replaced scalar `unattendedSaveTimer` with per-session `unattendedSaveTimers`, `unattendedSavePending`, and `unattendedSaveInFlight` maps.
- Saves now snapshot the current config at schedule time, post to the captured session even after switching away, serialize in-flight saves per session, apply saved config back to global/UI state only if the saved session is still selected/current, and continue flushing any newer pending snapshot after an in-flight save completes.
- Cleanup now clears all unattended save timers/pending/in-flight maps.
- Added source coverage for per-session save maps, snapshot body use, selected/menu-session application guards, in-flight serialization, and absence of the old selected-session drop guard in the debounce schedule path.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_unattended_mode_source.py tests/test_auth_cleanup_source.py tests/test_overlay_accessibility_source.py -q` → `19 passed`.
- Browser evidence against isolated Docker sandbox (`codoxear-sandbox-unattended-save-18804`, stopped): edited session A, switched to session B before debounce fired, verified one POST to A with the edit and B's opened popover remained B's config. Artifact: `/tmp/codoxear-unattended-save-browser.json`.
- Full local validation: `python3 -m pytest -q` → `687 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `686 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 21:24 — Unattended loading-window and zero-injection repair
- Re-review found another blocker: the popover became interactive before its current session config loaded, allowing stale/mixed global config to be saved and then overwritten by the load response.
- Repaired with `setUnattendedControlsDisabled()`: controls are disabled before opening/loading, re-enabled only after the current token/session load succeeds, and remain unable to schedule saves during the loading window.
- Also repaired remaining-injections zero consistency: setting remaining injections to `0` now sets `unattendedCfg.enabled = false` and unchecks the checkbox before scheduling the snapshot, matching the visible/sidebar state.
- Added source coverage for disabled controls during load, re-enable before focus, and zero-injection config/checkbox consistency.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_unattended_mode_source.py tests/test_auth_cleanup_source.py tests/test_overlay_accessibility_source.py tests/test_chat_scrollback_source.py -q` → `43 passed`.
- Browser evidence against isolated Docker sandbox (`codoxear-sandbox-unattended-loading-18805`, stopped): delayed GET left all controls disabled and produced no POST, then load enabled controls, moved focus to checkbox, and populated loaded config. Artifact: `/tmp/codoxear-unattended-loading-browser.json`.
- Full local validation: `python3 -m pytest -q` → `687 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `686 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 21:35 — Unattended budget save-invariant repair
- Final review found two budget-state blockers: client full snapshots could restore stale server-side injection decrements, and enabled could still be saved/returned with `remaining_injections: 0`.
- Repaired client saves to send only changed fields; pending patches merge per session. Request-only edits now POST only `{request: ...}` and cannot overwrite server-owned budget decrements.
- Repaired zero-budget behavior on both client and server: enabling with zero remaining is blocked client-side; setting remaining to zero sends `enabled: false`; server `unattended_get()` masks stale enabled-zero state and `unattended_set()` never stores enabled when remaining is zero.
- Save responses now update current popover controls and session metadata immediately before the next session refresh.
- Added execution tests in `tests/test_unattended_sweep.py` for zero-budget server invariant and partial request save preservation.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_unattended_mode_source.py tests/test_unattended_sweep.py tests/test_auth_cleanup_source.py tests/test_overlay_accessibility_source.py tests/test_chat_scrollback_source.py -q` → `53 passed`.
- Browser proof in isolated Docker sandbox (`codoxear-sandbox-unattended-patch-18806`, stopped): request-only edit posted only `request`; zero remaining posted `{remaining_injections: 0, enabled: false}`; zero-budget enable attempt posted `{enabled: false}`, left checkbox unchecked/remaining 0, and showed the explanatory toast. Artifact: `/tmp/codoxear-unattended-patch-browser.json`.
- Full local validation: `python3 -m pytest -q` → `690 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `689 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 21:41 — Session-list Unattended invariant repair
- Re-review found `/api/sessions` still exposed stale persisted `unattended_enabled: true` when `remaining_injections: 0`, even though `/unattended` GET/POST normalized it.
- Repaired `list_sessions()` to compute `unattended_enabled` as stored enabled AND remaining budget > 0.
- Tightened `/unattended` POST validation so `enabled` must be a JSON boolean instead of truthy-coerced strings.
- Added `tests/test_session_sidebar_priority.py::test_list_sessions_masks_stale_unattended_enabled_zero_remaining` plus source assertions for strict enabled validation.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_unattended_mode_source.py tests/test_unattended_sweep.py tests/test_session_sidebar_priority.py tests/test_auth_cleanup_source.py tests/test_overlay_accessibility_source.py tests/test_chat_scrollback_source.py -q` → `72 passed`.
- Full local validation: `python3 -m pytest -q` → `691 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `690 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 21:49 — Unattended sweep recheck race repair
- Final review found a server-side race: `_unattended_sweep()` snapshotted config, then could send an old prompt after a concurrent disable/zero-budget POST completed before `send()`.
- Repaired by using re-entrant per-session input locks and serializing `unattended_set()` with sends. The sweep now performs a live config/budget/cooldown recheck under that per-session lock immediately before sending and decrements from the current live remaining count after send.
- Added regression coverage: `test_rechecks_config_after_idle_probe_before_send` mutates config between the sweep snapshot and send eligibility probe; the sweep now sends nothing and preserves disabled/zero state.
- Focused validation: `python3 -m py_compile codoxear/server.py`; `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_unattended_sweep.py tests/test_unattended_mode_source.py tests/test_unattended_store.py tests/test_unattended_input_source.py tests/test_session_sidebar_priority.py -q` → `43 passed`.
- Full local validation: `python3 -m pytest -q` → `692 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `691 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 21:55 — Unattended live-tail cooldown recheck repair
- Re-review found the sweep still used a stale transcript tail observation: a new assistant turn could land after the initial tail scan but before send, violating the cooldown.
- Repaired by re-reading the latest assistant tail under the per-session input lock immediately before send, after the live config/budget/cooldown recheck.
- Also moved legacy zero-budget cleanup under the same per-session input lock to avoid racing a fresh re-enable.
- Added `test_rechecks_latest_assistant_timestamp_before_send`, where the first tail scan passes cooldown and the second live-tail scan is inside the cooldown; the sweep now sends nothing and does not decrement budget.
- Focused validation: `python3 -m py_compile codoxear/server.py`; `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_unattended_sweep.py tests/test_unattended_mode_source.py tests/test_unattended_store.py tests/test_unattended_input_source.py tests/test_session_sidebar_priority.py -q` → `44 passed`.
- Full local validation: `python3 -m pytest -q` → `693 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `692 passed, 1 skipped, 25 subtests passed`.

## 2026-06-13 22:42 — File-viewer selected-session coherence repair
- Selected next bounded product target after Unattended: file viewer must not publish stale paths/candidates/content under a newly selected session.
- Added `fileViewerSessionSyncToken` and `fileCandidateRequestSeq` guards. `ensureCurrentFileViewerSession()` and `showFileViewer()` now pass session/token guards across awaits; `refreshFileCandidates()` only publishes when its request sequence and session are current; closing the viewer invalidates pending sync.
- Repaired selection-boundary gap: `openSession()` now starts file-viewer sync immediately after selected-session optimistic state changes, before `/messages/tail`; tail success separately refreshes candidates so transcript-mentioned file refs are included after transcript render.
- Repaired stale resolved-open gap: `openFilePathWithResolvedMode()` and `openFilePathWithGuard()` capture the starting file-viewer session or use an explicit `isCurrent` guard and abort before `setFilePath()`/`openFilePath()` when stale.
- Added runtime/source coverage in `tests/test_file_viewer_source.py`; updated file-picker VM harness for candidate request sequence/current-session stubs.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_file_picker_session_state.py tests/test_chat_scrollback_source.py -q` → `55 passed`.
- Browser evidence in isolated Docker sandbox: `/tmp/codoxear-fileviewer-boundary2-browser.json` delayed old-session changed-files, delayed/failing new-session transcript tail, and confirmed the viewer stayed on session B candidates without leaking stale session A file paths.
- Full local validation: `python3 -m pytest -q` → `696 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `695 passed, 1 skipped, 25 subtests passed`.
- Clean-room review after repairs: no blockers; residuals are dirty-file intentional bypass, deletion while viewer open, and post-tail mentioned candidates not auto-opening if early sync had no file.

## 2026-06-13 22:56 — Busy-send choice keyboard focus repair
- Repaired keyboard-only busy-send branch: the send-choice dialog now has `aria-modal`, captures the opener, focuses "Send after current" when available (fallback to "Send now"/Cancel), and restores opener focus on Escape, Cancel, backdrop, Send now, and Send after current.
- Source coverage added in `tests/test_send_button_source.py`.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_send_button_source.py tests/test_queue_button_source.py tests/test_overlay_accessibility_source.py -q` → `10 passed`.
- Browser evidence in isolated Docker sandboxes: `/tmp/codoxear-sendchoice-focus-browser.json` covers Ctrl+Enter → initial focus → Escape return; `/tmp/codoxear-sendchoice-action-browser.json` covers Enter on focused "Send after current" → dialog close → focus restored to composer.
- Full local validation: `python3 -m pytest -q` → `697 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `696 passed, 1 skipped, 25 subtests passed`.
- Clean-room review after fixing action-path focus: no blockers; residuals are no explicit Tab trap and untested browser paths for Cancel/backdrop/Send now/attachment-disabled fallback, though source paths are direct.

## 2026-06-13 23:35 — Active message polling visibility/offline/error backoff
- Repaired active `/messages/*` poll cadence for slow/mobile/background reliability.
- Added explicit message poll delay policy: fast/running/idle visible rates, hidden slowdown, offline slowdown, exponential error backoff capped at 30s, and online/visible catch-up hooks.
- Preserved delayed kicks while a poll is in flight via `pollKickDelayMs`; hidden/offline/error transitions during an in-flight poll no longer collapse to immediate follow-up requests.
- Made `openSession()` tail success/failure participate in message poll health; tail failure now schedules a backoff retry after `pollGen` invalidation instead of stopping the active poll loop.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_session_polling_source.py tests/test_chat_scrollback_source.py tests/test_chat_transcript_runtime.py -q` → `44 passed`.
- Browser evidence in isolated Docker sandboxes:
  - `/tmp/codoxear-message-poll-visible-browser.json`: hidden visibility state suppresses steady live polling and visible state catches up.
  - `/tmp/codoxear-message-poll-hidden-inflight-browser.json`: hiding during a held in-flight message request does not produce an immediate follow-up after release; visible state resumes.
  - `/tmp/codoxear-message-poll-openfail-browser.json`: live `409` followed by tail `500` retries tail after ~2s error backoff instead of stopping polling.
- Full local validation: `python3 -m pytest -q` → `699 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `698 passed, 1 skipped, 25 subtests passed`.
- Clean-room review: no blockers; residuals are that source/runtime tests do not simulate every browser timer interleaving, direct openSession calls can overlap an existing poll outside the poll loop, and a pending error-derived delay can survive a success in some visible in-flight cases.

## 2026-06-13 23:49 — Removed-session file viewer cleanup
- Repaired a stale selected-session UI path: when the selected session disappears from `/api/sessions`, is deleted, or produces message 404, a non-dirty file viewer for that session now closes instead of showing stale files under "No session selected".
- Dirty file viewer state is preserved, but pending file open/search work is invalidated and the status explicitly says the session is no longer available and edits should be copied before closing.
- Added `handleFileViewerSessionUnavailable(sessionId)` and wired it into selected-missing refresh, selected delete, `openSession()` tail 404, and live poll 404 paths.
- Focused validation: `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_file_viewer_source.py tests/test_chat_scrollback_source.py tests/test_session_polling_source.py -q` → `58 passed`.
- Browser evidence in isolated Docker sandbox: `/tmp/codoxear-fileviewer-session-gone-browser.json` opens a non-dirty viewer for `sid-gone`, then `/api/sessions` returns empty; the viewer closes and File becomes disabled.
- Full local validation: `python3 -m pytest -q` → `700 passed, 25 subtests passed`.
- Full isolated Docker validation: `scripts/codoxear-docker-sandbox test` → `699 passed, 1 skipped, 25 subtests passed`.
- Clean-room review: no blockers; residuals are dirty old non-selected session disappearance and future dirty-viewer actions may still replace the explicit status with API errors.

## 2026-06-14 00:41 - Dirty removed-session file viewer hardening validation
- Changed files: `codoxear/static/app.js`, `tests/test_file_viewer_source.py`, `tests/test_file_picker_search_source.py`.
- Focused validation: `python3 -m py_compile tests/test_file_viewer_source.py`; `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_chat_scrollback_source.py tests/test_session_polling_source.py -q` -> 66 passed.
- Full local validation: `python3 -m pytest -q` -> 700 passed, 25 subtests passed.
- Docker validation: `scripts/codoxear-docker-sandbox test` -> 699 passed, 1 skipped, 25 subtests passed.
- Clean-room critic re-runs found and drove fixes for unsaved-dialog continuations, paste continuations, stale-selected tail-404 replay, draft-open continuation after inspect, and primitive open/resolve sharp edges. Final adversarial review: no blockers; focused source tests and diff check passed.

## 2026-06-14 00:52 - Dirty unavailable close prompt truthfulness
- Changed files: `codoxear/static/app.js`, `tests/test_file_viewer_source.py`.
- Focused validation: `python3 -m py_compile tests/test_file_viewer_source.py`; `node --check codoxear/static/app.js`; `python3 -m pytest tests/test_file_viewer_source.py tests/test_file_picker_search_source.py -q` -> 30 passed.
- Clean-room review: no blockers; residual risk is source-level rather than DOM/browser proof for focus/click behavior.
- Full local validation: `python3 -m pytest -q` -> 700 passed, 25 subtests passed.
- Docker validation: `scripts/codoxear-docker-sandbox test` -> 699 passed, 1 skipped, 25 subtests passed.

## 2026-06-14 01:36 - File path/session cwd fail-closed hardening
- Changed files: `codoxear/server.py`, `tests/test_file_inspect.py`, `tests/test_file_viewer_source.py`, `tests/test_session_sidebar_priority.py`.
- Focused validation: `python3 -m py_compile codoxear/server.py tests/test_file_inspect.py tests/test_file_viewer_source.py tests/test_session_sidebar_priority.py`; `python3 -m pytest tests/test_file_inspect.py tests/test_file_viewer_source.py tests/test_session_sidebar_priority.py -q` -> 74 passed, 20 subtests passed.
- Full local validation: `python3 -m pytest -q` -> 711 passed, 45 subtests passed.
- Docker validation: `scripts/codoxear-docker-sandbox test` -> 710 passed, 1 skipped, 45 subtests passed.
- Clean-room critic re-runs found and drove fixes for validation-before-expanduser, malformed `~user`, non-string `session_id`, session-scoped malformed cwd, write-update path validation, git cwd OS errors, and NUL cwd in session listing. Final review: no blockers.

## 2026-06-14 01:56 - Git helper late-error hardening
- Changed files: `codoxear/server.py`, `tests/test_file_inspect.py`.
- Focused validation: `python3 -m py_compile codoxear/server.py tests/test_file_inspect.py`; `python3 -m pytest tests/test_file_inspect.py -q` -> 38 passed, 20 subtests passed; broader focused set -> 78 passed, 20 subtests passed before the final two regressions, then full suite covered them.
- Full local validation: `python3 -m pytest -q` -> 717 passed, 45 subtests passed.
- Docker validation: `scripts/codoxear-docker-sandbox test` -> 716 passed, 1 skipped, 45 subtests passed.
- Clean-room review: no blockers; residual accepted ambiguity is `git show` RuntimeError in file_versions, which still maps to `base_exists=false` for untracked/new files.

## 2026-06-14 02:20 - Preview file I/O error hardening
- Changed files: `codoxear/server.py`, `codoxear/file_response.py`, `tests/test_file_inspect.py`, `tests/test_file_response_module_source.py`.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/file_response.py tests/test_file_inspect.py tests/test_file_response_module_source.py`; `python3 -m pytest tests/test_file_inspect.py tests/test_file_response_module_source.py -q` -> 45 passed, 36 subtests passed.
- Full local validation: `python3 -m pytest -q` -> 722 passed, 61 subtests passed.
- Docker validation: `scripts/codoxear-docker-sandbox test` -> 721 passed, 1 skipped, 61 subtests passed.
- Clean-room review: no blockers; residual risks are mid-stream TOCTOU after headers and rare OSError shapes outside FileNotFound/PermissionError.

## 2026-06-14 02:39 - File streaming size race hardening
- Changed files: `codoxear/file_response.py`, `tests/test_file_response_module_source.py`.
- Focused validation: `python3 -m py_compile codoxear/file_response.py tests/test_file_response_module_source.py`; `python3 -m pytest tests/test_file_response_module_source.py tests/test_file_inspect.py -q` -> 47 passed, 36 subtests passed.
- Full local validation: `python3 -m pytest -q` -> 724 passed, 61 subtests passed.
- Docker validation: `scripts/codoxear-docker-sandbox test` -> 723 passed, 1 skipped, 61 subtests passed.
- Clean-room review: no blockers; residual risk is file mutation after headers are sent.

## 2026-06-14 04:28 — Git/file-viewer literal-path hardening validation
- Worktree: /home/yiwen/codex-web-product-recovery on recovery/product-gaps; live checkout untouched.
- Implemented uncommitted tranche across codoxear/server.py, codoxear/static/app.js, and focused tests.
- Validation commands:
  - python3 -m py_compile codoxear/server.py tests/test_file_inspect.py tests/test_file_viewer_source.py tests/test_file_picker_search_source.py
  - python3 -m pytest tests/test_file_inspect.py tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_session_file_history.py tests/test_path_resolution.py -q -> 106 passed, 52 subtests passed
  - python3 -m pytest -q -> 750 passed, 77 subtests passed
  - scripts/codoxear-docker-sandbox test -> 749 passed, 1 skipped, 77 subtests passed
  - clean-room critic final review -> no commit-blocking issues; residuals: remembered selections do not persist git_path once files drop from changed_files, non-UTF-8 filenames remain replacement-decoded, symlink checks are not atomic against concurrent local mutation.
  - git diff --check -> clean.

## 2026-06-14 04:28 — Final git_path viewer fixes after review
- Clean-room review found two final blockers: activeFileGitPath could leak into newly created drafts after opening a changed file, and deleted changed-file candidates failed before file_versions because inspect returned 404. A lower-severity media URL gap in POST /api/files/read was also fixed.
- Follow-up validation:
  - python3 -m pytest tests/test_file_inspect.py tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_session_file_history.py tests/test_path_resolution.py -q -> 106 passed, 52 subtests passed
  - python3 -m pytest -q -> 750 passed, 77 subtests passed
  - scripts/codoxear-docker-sandbox test -> 749 passed, 1 skipped, 77 subtests passed
  - final clean-room critic review -> no commit-blocking issues.
  - git diff --check -> clean.

## 2026-06-14 05:48
- Completed file-picker candidate identity tranche in `/home/yiwen/codex-web-product-recovery` on branch `recovery/product-gaps`.
- Changed artifacts: `codoxear/static/app.js`, `tests/test_file_picker_search_source.py`, `tests/test_file_picker_session_state.py`, `tests/test_file_viewer_source.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_file_picker_search_source.py tests/test_file_viewer_source.py tests/test_file_picker_session_state.py tests/test_file_inspect.py tests/test_path_resolution.py -q` -> `111 passed, 52 subtests passed`.
  - Full local: `python3 -m pytest -q` -> `756 passed, 77 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `755 passed, 1 skipped, 77 subtests passed`.
  - `git diff --check` -> clean.
- Clean-room review loop found and drove fixes for timing-dependent pending Enter, normalized `./foo.py` pending/loaded/error search, stale cached/fresh diff enablement, and no-session candidate clear recompute. Final review before the no-session residual found no blockers; the residual was patched and all validation rerun.

## 2026-06-14 05:57
- Completed late file-response stream error hardening tranche.
- Changed artifacts: `codoxear/file_response.py`, `tests/test_file_response_module_source.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_file_response_module_source.py tests/test_video_preview_cache.py tests/test_file_inspect.py -q` -> `75 passed, 52 subtests passed`.
  - Full local: `python3 -m pytest -q` -> `758 passed, 77 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `757 passed, 1 skipped, 77 subtests passed`.
  - Clean-room review -> no blockers; noted unused `_stream_file_bytes()` direct-open residual and unavoidable post-header `Content-Length` mismatch on truncation.
- `git diff --check` was clean before staging.

## 2026-06-14 06:05
- Completed file-picker same-display score normalization tranche.
- Changed artifacts: `codoxear/static/app.js`, `tests/test_file_picker_search_source.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_file_picker_search_source.py tests/test_file_viewer_source.py tests/test_file_picker_session_state.py -q` -> `41 passed`.
  - Full local: `python3 -m pytest -q` -> `761 passed, 77 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `760 passed, 1 skipped, 77 subtests passed`.
  - Clean-room review -> no blockers after replacing a non-transitive comparator attempt with same-display score normalization.

## 2026-06-14 06:34
- Completed file-picker identity hint UI/UX tranche.
- Changed artifacts: `codoxear/static/app.js`, `codoxear/static/app.css`, `tests/test_file_picker_search_source.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_file_picker_search_source.py tests/test_file_viewer_source.py tests/test_file_picker_session_state.py -q` -> `42 passed`.
  - Full local: `python3 -m pytest -q` -> `762 passed, 77 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `761 passed, 1 skipped, 77 subtests passed`.
  - Final clean-room review -> no blockers. Prior review found an accessibility blocker from overriding picker option `aria-label`; fixed by relying on visible hint spans plus `title` instead.

## 2026-06-14 06:50
- Completed video preview singleflight/failure-throttle reliability tranche.
- Changed artifacts: `codoxear/video_preview.py`, `tests/test_video_preview_cache.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_video_preview_cache.py tests/test_file_inspect.py tests/test_file_viewer_source.py -q` -> `94 passed, 52 subtests passed`.
  - Full local: `python3 -m pytest -q` -> `767 passed, 77 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `766 passed, 1 skipped, 77 subtests passed`.
  - Clean-room review loop found and drove fixes for unbounded failure map and exception-semantics drift; final review -> no blockers.

## 2026-06-14 07:10
- Completed long-chat streaming search tranche.
- Changed artifacts: `codoxear/server.py`, `tests/test_transcript_export.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_transcript_export.py tests/test_chat_navigation_source.py tests/test_chat_transcript_runtime.py tests/test_message_index.py -q` -> `35 passed, 3 subtests passed`.
  - Full local: `python3 -m pytest -q` -> `774 passed, 77 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `773 passed, 1 skipped, 77 subtests passed`.
  - Clean-room review loop found and drove fixes for malformed/non-dict/oversized record handling and bounded line buffering; final review -> no blockers.

## 2026-06-14 07:46
- Completed Pi/stale-broker-busy readiness tranche.
- Changed artifacts: `codoxear/server.py`, `tests/test_server_queue_persistence.py`, `tests/test_queue_sweep_idle_guard.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_sessions_pending_log_idle.py tests/test_broker_busy_state.py tests/test_idle_heuristics.py -q` -> `145 passed, 20 subtests passed`.
  - Full local: `python3 -m pytest -q` -> `784 passed, 82 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `783 passed, 1 skipped, 82 subtests passed`.
  - Clean-room review loop found and drove fixes for overbroad stale-busy override, log-path/size race, queue sidecar refresh, pending attachment, and malformed broker state; final review -> no blockers.

## 2026-06-14 08:14
- Completed ambiguous inline file-reference UX tranche.
- Changed artifacts: `codoxear/static/app.js`, `codoxear/static/app.css`, `tests/test_markdown_tables.py`, `tests/test_file_picker_search_source.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_markdown_tables.py tests/test_file_picker_search_source.py tests/test_file_viewer_source.py -q` -> `52 passed`.
  - Full local: `python3 -m pytest -q` -> `785 passed, 82 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `784 passed, 1 skipped, 82 subtests passed`.
  - Clean-room review loop found and drove fixes for programmatic-focus query reset and ambiguity-launched create-new defaults; final review -> no blockers.

## 2026-06-14 08:27
- Completed broker-state strictness/refactor tranche.
- Changed artifacts: `codoxear/server.py`, `tests/test_sessions_pending_log_idle.py`, `tests/test_stale_sidecars.py`, `tests/test_server_queue_persistence.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_sessions_pending_log_idle.py tests/test_stale_sidecars.py tests/test_server_queue_persistence.py tests/test_unattended_sweep.py -q` -> `104 passed, 26 subtests passed`.
  - Full local: `python3 -m pytest -q` -> `788 passed, 88 subtests passed`.
  - Docker sandbox: first attempt failed before tests on Docker Hub TLS metadata timeout; retry `scripts/codoxear-docker-sandbox test` -> `787 passed, 1 skipped, 88 subtests passed`.
  - Clean-room review -> no blockers.

## 2026-06-14 08:38
- Completed project-backed inline file-reference ambiguity tranche.
- Changed artifacts: `codoxear/static/app.js`, `tests/test_file_picker_search_source.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_file_picker_search_source.py tests/test_markdown_tables.py tests/test_file_inspect.py -q` -> `95 passed, 52 subtests passed`.
  - Full local: `python3 -m pytest -q` -> `789 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `788 passed, 1 skipped, 88 subtests passed`.
  - Clean-room review loop found and drove the truncated-zero-match fix; final review -> no blockers.

## 2026-06-14 08:55
- Completed chat transcript-search hint UX tranche.
- Changed artifacts: `codoxear/static/app.js`, `codoxear/static/app.css`, `tests/test_chat_navigation_source.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_chat_navigation_source.py tests/test_transcript_export.py tests/test_chat_transcript_runtime.py -q` -> `31 passed, 3 subtests passed`.
  - Full local: `python3 -m pytest -q` -> `789 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `788 passed, 1 skipped, 88 subtests passed`.
  - Clean-room review loop found no blockers; mobile-width layout risk addressed by hiding the hint under `max-width: 520px`.

## 2026-06-14 09:13
- Completed payload-bound transcript search hint tranche.
- Changed artifacts: `codoxear/server.py`, `codoxear/static/app.js`, `tests/test_transcript_export.py`, `tests/test_chat_navigation_source.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_transcript_export.py tests/test_chat_navigation_source.py tests/test_chat_transcript_runtime.py -q` -> `33 passed, 3 subtests passed`.
  - Full local: `python3 -m pytest -q` -> `791 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `790 passed, 1 skipped, 88 subtests passed`.
  - Clean-room review loop found and drove fixes for prefix clipping removing the match, Unicode casefold offset expansion, and boundary-length match truncation; final review -> no blockers.

## 2026-06-14 09:25
- Completed route-level transcript search `text_max` evidence tranche.
- Changed artifact: `tests/test_transcript_export.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_transcript_export.py tests/test_chat_navigation_source.py tests/test_chat_transcript_runtime.py -q` -> `35 passed, 3 subtests passed`.
  - Full local: `python3 -m pytest -q` -> `793 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `792 passed, 1 skipped, 88 subtests passed`.
  - Clean-room review found no blockers; strengthened the test to prove total `match_count` can exceed returned `matches` under `limit=1`.

## 2026-06-14 09:36
- Completed bounded structural extraction of transcript-search helpers from `codoxear/server.py` to `codoxear/transcript_search.py`.
- Preserved server-local aliases for old private helper names used by route code and ad hoc private imports; updated tests to import behavior from the new module and source-test the boundary.
- Validation:
  - Focused: `python3 -m py_compile codoxear/server.py codoxear/transcript_search.py && python3 -m pytest tests/test_transcript_export.py tests/test_chat_navigation_source.py tests/test_chat_transcript_runtime.py tests/test_route_decomposition_source.py -q` -> `37 passed, 3 subtests passed`.
  - Architecture clean-room review: no blockers; import graph `server -> transcript_search -> rollout_log`, no cycle; env constant semantics preserved.
  - Full local: `python3 -m pytest -q` -> `794 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `793 passed, 1 skipped, 88 subtests passed`.

## 2026-06-14 10:08
- Completed malformed sidecar metadata hardening tranche.
- Changed artifacts: `codoxear/server.py`, `tests/test_stale_sidecars.py`.
- Validation:
  - Focused: `python3 -m py_compile codoxear/server.py tests/test_stale_sidecars.py && python3 -m pytest tests/test_stale_sidecars.py tests/test_broker_fail_closed.py tests/test_broker_busy_state.py -q` -> `86 passed`.
  - Clean-room review loop found and drove fixes for late `start_ts` validation, refresh trusting bad typed metadata, bool/int coercion, non-finite and overflowing timestamps, overflowing optional `updated_ts`, and directory `log_path`; final review -> no blockers.
  - Full local: `python3 -m pytest -q` -> `807 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `806 passed, 1 skipped, 88 subtests passed`.

## 2026-06-14 10:17
- Completed inline file-reference negative validation cache freshness fix.
- Changed artifacts: `codoxear/static/app.js`, `tests/test_file_picker_search_source.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_file_picker_search_source.py tests/test_file_viewer_source.py -q` -> `41 passed`.
  - Clean-room review: no blockers; confirmed failed inspect results are no longer cached while pending singleflight and successful validation cache remain.
  - Full local: `python3 -m pytest -q` -> `807 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `806 passed, 1 skipped, 88 subtests passed`.

## 2026-06-14 10:25
- Completed desktop notification clickthrough UX tranche.
- Changed artifacts: `codoxear/static/app.js`, `tests/test_voice_push_source.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_voice_push_source.py tests/test_voice_push.py tests/test_static_assets.py -q` -> `48 passed`.
  - Clean-room review: no blockers; VM test now exercises `Notification.onclick` behavior with fake Notification/window/session selection.
  - Full local: `python3 -m pytest -q` -> `808 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `807 passed, 1 skipped, 88 subtests passed`.

## 2026-06-14 10:48
- Completed file viewer accessibility/focus tranche.
- Changed artifacts: `codoxear/static/app.js`, `tests/test_overlay_accessibility_source.py`, `tests/test_file_viewer_source.py`.
- Validation:
  - Focused: `python3 -m pytest tests/test_overlay_accessibility_source.py tests/test_file_viewer_source.py tests/test_file_picker_search_source.py -q` -> `48 passed`.
  - Syntax/full local: `node --check codoxear/static/app.js && python3 -m pytest -q` -> `809 passed, 88 subtests passed`.
  - Clean-room review loop found and drove fixes for unsaved-dialog focus/isolation, picker-query initial focus before async refresh, picker-query input overwrite risk, and Monaco post-load focus stealing; final review -> no blockers.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `808 passed, 1 skipped, 88 subtests passed`.

## 2026-06-14 10:57
- Completed inline bare-file-ref identity merge tranche for repo subdirectory sessions.
- Changed artifacts: `codoxear/static/app.js`, `tests/test_file_picker_search_source.py`.
- Validation:
  - Focused: `node --check codoxear/static/app.js && python3 -m pytest tests/test_file_picker_search_source.py tests/test_file_viewer_source.py -q` -> `41 passed`.
  - Clean-room review: no blockers; confirmed git/session identity cache keys, fail-closed truncation, and same-physical-file collapse only when inspected absolute paths match.
  - Full local: `python3 -m pytest -q` -> `809 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `808 passed, 1 skipped, 88 subtests passed`.

## 2026-06-14 11:03
- Completed launch sidecar metadata wait hardening tranche.
- Changed artifacts: `codoxear/server.py`, `tests/test_session_resume.py`.
- Validation:
  - Focused: `python3 -m py_compile codoxear/server.py tests/test_session_resume.py && python3 -m pytest tests/test_session_resume.py tests/test_stale_sidecars.py -q` -> `50 passed`.
  - Clean-room review: no blockers; confirmed `_wait_for_spawned_broker_meta` skips malformed JSON and `broker_pid: true` before returning metadata.
  - Full local: `python3 -m pytest -q` -> `811 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `810 passed, 1 skipped, 88 subtests passed`.

## 2026-06-14 11:12
- Completed bounded live JSONL reader hardening tranche.
- Changed artifacts: `codoxear/util.py`, `codoxear/rollout_log.py`, `tests/test_read_jsonl_from_offset.py`.
- Validation:
  - Focused: `python3 -m py_compile codoxear/util.py codoxear/rollout_log.py tests/test_read_jsonl_from_offset.py && python3 -m pytest tests/test_read_jsonl_from_offset.py tests/test_broker_fail_closed.py tests/test_sessiond_fail_closed.py tests/test_chat_transcript_runtime.py -q` -> `46 passed`.
  - Clean-room review found a mid-UTF8 skip bug; fixed by treating `UnicodeDecodeError` as a malformed line in util reader; final review -> no blockers.
  - Full local: `python3 -m pytest -q` -> `814 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `813 passed, 1 skipped, 88 subtests passed`.

## 2026-06-14 11:26
- Completed non-git file-picker candidate fallback tranche.
- Changed artifacts: `codoxear/static/app.js`, `tests/test_file_picker_search_source.py`.
- Validation:
  - Focused: `node --check codoxear/static/app.js && python3 -m pytest tests/test_file_picker_search_source.py tests/test_file_picker_session_state.py tests/test_file_viewer_source.py -q` -> `44 passed`.
  - Clean-room review: no blockers; confirmed mentioned/recent candidates are no longer gated by `/git/changed_files` success and changed-file freshness/cache only commit on successful changed-files response.
  - Full local: `python3 -m pytest -q` -> `815 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `814 passed, 1 skipped, 88 subtests passed`.

## 2026-06-14 12:02
- Completed long-transcript search navigation tranche.
- Changed artifacts: `codoxear/transcript_search.py`, `codoxear/server.py`, `codoxear/static/app.js`, `tests/test_transcript_export.py`, `tests/test_chat_navigation_source.py`, `tests/test_chat_scrollback_source.py`.
- Validation:
  - Focused: `node --check codoxear/static/app.js && python3 -m py_compile codoxear/server.py codoxear/transcript_search.py tests/test_transcript_export.py && python3 -m pytest tests/test_transcript_export.py tests/test_chat_navigation_source.py tests/test_message_index.py tests/test_message_route_source.py -q` -> `35 passed, 3 subtests passed`.
  - Additional focused after source-test update: `node --check codoxear/static/app.js && python3 -m pytest tests/test_chat_scrollback_source.py tests/test_transcript_export.py tests/test_chat_navigation_source.py tests/test_message_index.py tests/test_message_route_source.py -q` -> `59 passed, 3 subtests passed`.
  - Clean-room review: multiple blockers found and fixed (unterminated final-record cursor mismatch, empty-window clearing, global-first cursor stranding, wrong target focus in multi-match windows, Python casefold vs JS matching persistence); final review -> no blockers.
  - Full local: `python3 -m pytest -q` -> `817 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `816 passed, 1 skipped, 88 subtests passed`.

## 2026-06-14 12:13
- Completed unattended final-turn eligibility guard tranche.
- Changed artifacts: `codoxear/rollout_log.py`, `codoxear/server.py`, `tests/test_server_chat_flags.py`, `tests/test_unattended_sweep.py`.
- Validation:
  - Focused: `python3 -m py_compile codoxear/rollout_log.py codoxear/server.py tests/test_unattended_sweep.py tests/test_server_chat_flags.py && python3 -m pytest tests/test_unattended_sweep.py tests/test_server_chat_flags.py tests/test_unattended_mode_source.py -q` -> `43 passed`.
  - Clean-room review: initial blocker for Codex `turn_complete`/`task_complete` with `last_agent_message` fixed; final review -> no blockers.
  - Full local: `python3 -m pytest -q` -> `821 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `820 passed, 1 skipped, 88 subtests passed`.

## 2026-06-14 12:19
- Completed mobile/coarse-pointer composer stop control tranche.
- Changed artifacts: `codoxear/static/app.js`, `codoxear/static/app.css`, `tests/test_send_button_source.py`.
- Validation:
  - Focused: `node --check codoxear/static/app.js && python3 -m pytest tests/test_send_button_source.py tests/test_chat_navigation_source.py tests/test_static_assets.py -q` -> `22 passed`.
  - Clean-room review: no blockers; confirmed shared interrupt semantics, no form-submit regression, default desktop hidden state, and topbar sparsity preserved.
  - Full local: `python3 -m pytest -q` -> `822 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `821 passed, 1 skipped, 88 subtests passed`.

## 2026-06-14 12:25
- Completed immediate file-picker fallback rendering tranche.
- Changed artifacts: `codoxear/static/app.js`, `tests/test_file_picker_search_source.py`.
- Validation:
  - Focused: `node --check codoxear/static/app.js && python3 -m pytest tests/test_file_picker_search_source.py tests/test_file_picker_session_state.py tests/test_file_viewer_source.py -q` -> `45 passed`.
  - Clean-room review: no blockers; confirmed fallback renders before changed-files awaits, cache/freshness still only commit on successful changed-files, and stale request guards remain effective.
  - Full local: `python3 -m pytest -q` -> `823 passed, 88 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `822 passed, 1 skipped, 88 subtests passed`.

## 2026-06-14 12:37
- Completed bounded transcript search count tranche.
- Changed artifacts: `codoxear/transcript_search.py`, `codoxear/server.py`, `codoxear/static/app.js`, `tests/test_transcript_export.py`, `tests/test_chat_navigation_source.py`.
- Validation:
  - Focused: `node --check codoxear/static/app.js && python3 -m py_compile codoxear/transcript_search.py codoxear/server.py tests/test_transcript_export.py && python3 -m pytest tests/test_transcript_export.py tests/test_chat_navigation_source.py tests/test_message_index.py tests/test_message_route_source.py -q` -> `38 passed, 4 subtests passed`.
  - Clean-room review: initial blockers fixed (truncated count treated as lower bound in client; `count_max` rejected with `order=latest`); final review -> no blockers.
  - Full local: `python3 -m pytest -q` -> `826 passed, 89 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `825 passed, 1 skipped, 89 subtests passed`.

## 2026-06-14 12:49
- Completed rollout log chat-extraction refactor tranche.
- Changed artifacts: `codoxear/rollout_log.py`, `tests/test_cc_chat_and_idle.py`, `tests/test_rollout_log_helpers_source.py`.
- Validation:
  - Focused: `python3 -m py_compile codoxear/rollout_log.py tests/test_cc_chat_and_idle.py tests/test_rollout_log_helpers_source.py && python3 -m pytest tests/test_rollout_log_helpers_source.py tests/test_cc_chat_and_idle.py tests/test_chat_transcript_runtime.py tests/test_chat_scrollback_source.py tests/test_message_index.py tests/test_cc_log.py tests/test_idle_heuristics.py tests/test_unattended_sweep.py -q` -> `91 passed`.
  - Clean-room review: initial blocker found duplicate CC id-less pending-tool placeholders; fixed by making `_single_chat_event` the sole pending-id updater in `_extract_chat_events`; final review -> no blockers.
  - Full local: `python3 -m pytest -q` -> `827 passed, 89 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `826 passed, 1 skipped, 89 subtests passed`.

## 2026-06-14 16:34
- Completed tmux launch sidecar live-pid validation tranche.
- Changed artifacts: `codoxear/server.py`, `tests/test_session_resume.py`.
- User corrected authorization context: provider/key locations are available from `~/.pi/agents` and `~/.zshrc`; future live backend validation should proceed with redaction/isolated state rather than treating authorization as absent.
- Validation:
  - Focused: `python3 -m py_compile codoxear/server.py tests/test_session_resume.py && python3 -m pytest tests/test_session_resume.py tests/test_stale_sidecars.py tests/test_launch_provenance.py -q` -> `60 passed`.
  - Clean-room review: no blockers; residuals are PID-reuse/identity limits and tmux pending classification if pane dies during wait.
  - Full local: `python3 -m pytest -q` -> `828 passed, 89 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `827 passed, 1 skipped, 89 subtests passed`.

## 2026-06-14 17:32
- Completed live-backed Pi provider launch repair tranche.
- Changed artifacts: `codoxear/launch_config.py`, `codoxear/static/app.js`, `tests/test_new_session_launch_request.py`, `tests/test_new_session_model_options_source.py`, `tests/test_reasoning_effort_source.py`.
- Important operational note: an early redaction command accidentally printed secret-looking values from `~/.zshrc` into the tool transcript. Values are not repeated here and were not written to project files. Treat the session transcript as sensitive if exported/shared.
- Live validation evidence:
  - Created isolated Codoxear app state under a temp HOME and a temp venv; backend homes/env came from the user's real configured homes and zsh environment.
  - Negative copied-home attempt: copied config homes caused all CLIs to exit or fail before log binding, so it was not accepted as live proof; copied credential temp tree was deleted.
  - Negative stale-provider attempt: Codoxear rejected Pi API launch with `model_provider=anthropic` because stale defaults only allowed custom providers; manual Pi broker showed the CLI itself accepts `anthropic / claude-haiku-4-5`.
  - After the fix, `/api/sessions` Pi launch with `model_provider=anthropic`, `model=claude-haiku-4-5`, `reasoning_effort=low` returned 200; send returned 200; session was discovered; log bound; assistant final response was observed; idle was observed; cleanup delete returned 200.
  - Claude Code under isolated HOME remained blocked at first-run theme/onboarding before log binding; copying `.claude.json` did not clear it in that setup. This is negative evidence for isolated-HOME validation, not proof of production failure with the real HOME.
  - Codex manual broker with the exact project trust override reached the interactive TUI; earlier API send did not produce a bound log/final response within timeout. This remains incomplete live-response evidence.
- Validation:
  - Focused final: `node --check codoxear/static/app.js && python3 -m py_compile codoxear/launch_config.py tests/test_new_session_launch_request.py tests/test_new_session_model_options_source.py && python3 -m pytest tests/test_new_session_launch_request.py tests/test_launch_defaults.py tests/test_backend_launch_adapter.py tests/test_new_session_model_options_source.py tests/test_launch_ui_source.py -q` -> `42 passed`.
  - Clean-room reviews: initial residuals around UI stale provider and reasoning fallback fixed; final review -> no blockers.
  - Full local: `python3 -m pytest -q` -> `831 passed, 89 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `830 passed, 1 skipped, 89 subtests passed`.

## 2026-06-14 17:44
- Completed Pi custom-provider UI behavioral coverage tranche.
- Changed artifact: `tests/test_new_session_model_options_source.py`.
- Added executable Node VM coverage for Pi typed `provider/model` behavior when `provider_choices` is empty and the bare-model reasoning cache is stale. The test extracts the real provider/model parser, display, and reasoning-choice functions from `app.js` and verifies `anthropic/claude-haiku-4-5` remains a provider-specific Pi launch candidate with `low` still available.
- Restored source guards for adjacent custom-provider paths: provider setter, provider/model label, remembered pair restore, and recent-session provider options.
- Validation:
  - Focused: `node --check codoxear/static/app.js && python3 -m py_compile tests/test_new_session_model_options_source.py && python3 -m pytest tests/test_new_session_model_options_source.py tests/test_new_session_launch_request.py tests/test_reasoning_effort_source.py tests/test_launch_ui_source.py -q` -> `26 passed`.
  - Clean-room review: initial review noted adjacent source guards were lost; restored them; final review -> no blockers.
  - Full local: `python3 -m pytest -q` -> `832 passed, 89 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `831 passed, 1 skipped, 89 subtests passed`.

## 2026-06-14 17:59
- Browser UX pass in isolated Docker `codoxear-sandbox-ux-18983` with synthetic 180-turn Codex transcript found a frontend search-navigation bug.
- Observation: Query `EARLY-SEARCH-NEEDLE` showed `0/0 loaded · 1 all`; the Next button was enabled, but the first replay path did not load the offscreen match. API evidence showed `/messages/search?...order=latest&before=...` could find the match and `/messages/history?cursor=...` returned the expected window.
- Mechanism: `stepChatSearch()` called `refreshLoadedChatSearch()`, which always scheduled an all-transcript recount and reset `chatSearchAllCount` before the branch that decides whether to load offscreen matches.
- Patch: `refreshLoadedChatSearch` now accepts `refreshAllCount`; normal query/open/live paths keep the default refresh, while `stepChatSearch()` preserves existing all-count evidence during navigation.
- Browser validation after hard reload/instrumentation: clicking Next emitted `/messages/search?...order=latest&before=...` and `/messages/history?cursor=...`; UI status became `1/1 loaded · 1 all`; toast `Loaded transcript match`; no captured JS errors.
- Validation:
  - Focused: `node --check codoxear/static/app.js && python3 -m py_compile tests/test_chat_navigation_source.py && python3 -m pytest tests/test_chat_navigation_source.py tests/test_message_index.py tests/test_chat_transcript_runtime.py -q` -> `22 passed`.
  - Clean-room review: `/tmp/codoxear-search-navigation-count-review.md` -> no blockers.
  - Full local: `python3 -m pytest -q` -> `832 passed, 89 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `831 passed, 1 skipped, 89 subtests passed`.
- Residual browser observation not yet fixed: long transcripts expose one tabbable `Copy raw markdown` button per rendered assistant message, producing a long keyboard/accessibility traversal.

## 2026-06-14 18:17
- Completed roving per-message copy-control accessibility tranche.
- Observation from isolated browser long transcript: 60 rendered message copy buttons were all tabbable/accessibility-visible, causing `Copy raw markdown` to repeat through the long transcript tab/accessibility traversal.
- Patch: per-message copy controls now use a roving active copy button. Inactive message copy buttons are disabled, `tabIndex=-1`, `aria-hidden=true`, visually hidden, and pointer-inert. The latest/active/navigated/hovered row exposes one enabled/tabbable copy button. `Alt+Shift+↑/↓` moves the active copy control across all loaded copyable messages; existing `Alt+↑/↓` user-message navigation moves focus safely when invoked from a copy button.
- Clean-room review sequence: initial review found focused buttons could become `aria-hidden`; fixed by focus transfer and pointerover guard. Second review found inactive buttons were still painted/pointer-activatable; fixed by disabled + CSS hiding/pointer-events none and copyable-row-only navigation. Final review -> no blockers.
- Browser validation after hard reload in synthetic 180-turn transcript: 60 copy button nodes, exactly 1 active/tabbable/accessibility-visible; inactive samples had disabled true, tabIndex -1, aria-hidden true, opacity 0, visibility hidden, pointerEvents none; accessibility snapshot `Copy raw markdown` count 1. Row pointerover promoted a row-local assistant copy button. Hidden-focus counterexample (`focus active copy` + `Alt+↑`) left focus on the new visible user-row copy with hiddenFocused false. `Alt+Shift+↑` moved focus to a visible assistant-row copy with hiddenFocused false.
- Validation:
  - Focused: `node --check codoxear/static/app.js && python3 -m py_compile tests/test_chat_navigation_source.py && python3 -m pytest tests/test_chat_navigation_source.py tests/test_button_tooltips_source.py tests/test_overlay_accessibility_source.py -q` -> `19 passed`.
  - Full local: `python3 -m pytest -q` -> `833 passed, 89 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `832 passed, 1 skipped, 89 subtests passed`.

## 2026-06-14 18:22
- Created current refactor-entry checkpoint artifact: `recon/refactor-entry-checkpoint.md`.
- Checkpoint records branch/head, closed product gaps, latest validation evidence, invariants broad refactoring must preserve, and parked limits/decisions. It explicitly states it is not merge approval and that `/home/yiwen/codex-web` on `main` was not modified or merged.
- Clean-room review: `/tmp/codoxear-refactor-entry-checkpoint-review.md` -> no blockers. Reviewer independently verified branch/head, live checkout state, scoped live-evidence claims, and reran focused + full local tests (`19 passed`; `833 passed, 89 subtests passed`).
- Applied reviewer precision suggestion: Pi live validation used isolated Codoxear app/session state while provider configuration came from the user's existing real Pi environment without printing secret values.

## 2026-06-14 18:32
- Fixed root-cwd file-create containment bug found by reliability scout.
- Mechanism: `_resolve_under()` previously checked `str(resolved).startswith(str(resolved_base) + os.sep)`. For `base == /`, that prefix becomes `//`, so valid descendants such as `/tmp/x` were rejected as escaping.
- Patch: `_resolve_under()` now resolves base/target and uses component-aware `resolved.relative_to(resolved_base)`, preserving resolved symlink containment semantics while handling `/` naturally.
- Added tests:
  - helper regression for `_resolve_under(Path('/'), 'tmp/codoxear-root-cwd-test.txt')`;
  - parent escape rejection regression;
  - route-level `/api/sessions/<id>/file/write` create regression for a session with `cwd='/'` writing under a fresh `/tmp` directory.
- Validation:
  - Focused: `python3 -m py_compile codoxear/server.py tests/test_path_resolution.py tests/test_file_inspect.py && python3 -m pytest tests/test_path_resolution.py tests/test_file_inspect.py tests/test_file_write_locks.py -q` -> `78 passed, 52 subtests passed`.
  - Clean-room review: `/tmp/codoxear-root-cwd-resolve-review.md` -> no blockers.
  - Full local: `python3 -m pytest -q` -> `836 passed, 89 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `835 passed, 1 skipped, 89 subtests passed`.

## 2026-06-14 18:42
- Fixed transcript-search count certainty when oversized JSONL records are skipped by the bounded line reader.
- Mechanism: `iter_jsonl_records_forward_bounded()` skipped lines over `TRANSCRIPT_SEARCH_MAX_LINE_BYTES`, but `/messages/search` default exact-count mode could still report `match_count_truncated=false`, implying all records had been searched.
- Patch: bounded JSONL iteration now accepts an oversized-skip callback. `search_chat_log_bounded()` marks the count truncated when a skipped oversized line starts inside the searched byte range. `/messages/search` now uses `_search_chat_log_bounded()` for both exact/default and `count_max` modes, passing `TRANSCRIPT_SEARCH_MAX_LINE_BYTES` explicitly.
- Added tests for skipped oversized matching records, boundary-scoped oversized skips, route-level `match_count_truncated`, and updated route decomposition/source guards.
- Validation:
  - Focused: `python3 -m py_compile codoxear/transcript_search.py codoxear/server.py tests/test_transcript_export.py && python3 -m pytest tests/test_transcript_export.py tests/test_message_route_source.py tests/test_chat_navigation_source.py -q` -> `38 passed, 4 subtests passed`.
  - Clean-room review: `/tmp/codoxear-oversized-search-review.md` -> no blockers.
  - Post-source-guard focused: `python3 -m pytest tests/test_transcript_export.py tests/test_message_route_source.py tests/test_chat_navigation_source.py tests/test_route_decomposition_source.py -q` -> `40 passed, 4 subtests passed`.
  - Full local: `python3 -m pytest -q` -> `839 passed, 89 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `838 passed, 1 skipped, 89 subtests passed`.

## 2026-06-14 18:43
- Refreshed `recon/refactor-entry-checkpoint.md` after post-checkpoint reliability commits `892961a` and `da93073`.
- Updated current HEAD, latest validation counts, closed gap bullets for root-cwd file creation and oversized transcript-search truncation, review artifact list, and search invariant wording.

## 2026-06-14 19:27
- Completed in-chat recovery panel UX tranche.
- Changed artifacts: `codoxear/static/app.js`, `codoxear/static/app.css`, `tests/test_chat_scrollback_source.py`, `tests/test_overlay_accessibility_source.py`.
- Mechanism fixed: orphan/queue/unknown recovery rows were discoverable in the sidebar but could leave the chat pane empty or stale. The UI now renders a recovery panel in the chat pane using existing session metadata and existing guarded actions (`Review queue`, `Clear unknown marker`, `Copy details`).
- Important implementation details:
  - Recovery panel is not transcript content: `renderedMessageRows`, decoration/windowing/viewport helpers, and trim logic exclude `.recovery-panel-row`.
  - Live appends and session metadata refresh re-render/reposition the panel; mutation paths call `syncRecoveryUiForSession`.
  - Focus is preserved across rapid panel rebuilds with a session-scoped pending recovery action descriptor, including cases where the focused action disappears but the panel remains.
  - Queue modal close falls back to a visible recovery action if its opener was re-rendered/removed.
  - Transcript load errors clear typing before rendering the recovery panel, so the panel remains after the error row.
- Browser evidence in isolated Docker `codoxear-sandbox-recovery-ui-18984`:
  - Orphan recovery with direct unknown + queue rendered panel, no `/messages/tail` fetch, role `group`, Review queue opened disabled preserved prompts.
  - Clearing unknown marker removed the direct warning/button and left focus on a visible panel action.
  - Deleting queue items updated the panel queue count; deleting the last item removed the recovery row/panel/selection.
  - Transcript-backed recovery session with existing messages rendered panel as last row; after appending live user/assistant events, appended messages appeared and panel remained last.
  - Focused `Copy details` survived rapid live user+assistant appends and remained connected inside the rebuilt panel.
- Clean-room review iterations: `/tmp/codoxear-recovery-panel-review.md` through `/tmp/codoxear-recovery-panel-review6.md`; blockers around stale state, focus, decoration, and load-error ordering were fixed; final blocker-only review found no blocker.
- Validation:
  - Focused final: `node --check codoxear/static/app.js && python3 -m py_compile tests/test_chat_scrollback_source.py && python3 -m pytest tests/test_chat_scrollback_source.py tests/test_queue_button_source.py tests/test_file_upload_module_source.py -q` -> `30 passed`.
  - Additional failed-harness repair focused: `python3 -m pytest tests/test_chat_transcript_runtime.py::TestChatTranscriptRuntime::test_live_delta_dedupes_adjacent_assistant_text_across_polls tests/test_overlay_accessibility_source.py::TestOverlayAccessibilitySource::test_queue_help_details_dialogs_restore_focus tests/test_chat_scrollback_source.py tests/test_queue_button_source.py tests/test_file_upload_module_source.py -q` -> `32 passed`.
  - Full local: `python3 -m pytest -q` -> `840 passed, 89 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `839 passed, 1 skipped, 89 subtests passed`.

## 2026-06-14 19:28
- Refreshed `recon/refactor-entry-checkpoint.md` after recovery-panel commit `31a5c2d`.
- Updated current HEAD, latest validation counts, closed gap bullets, browser evidence, review artifact list, and unknown-commit invariant wording.

## 2026-06-15 01:14
- Completed bounded architectural extraction: moved sidecar metadata validation/capability helpers from `codoxear/server.py` into `codoxear/sidecar_metadata.py` and kept server call-site aliases intact.
- Changed artifacts: `codoxear/server.py`, `codoxear/sidecar_metadata.py`, `tests/test_sidecar_metadata.py`.
- Mechanism preserved: socket sidecar discovery, refresh, and tmux-spawn metadata wait still use the same fail-closed validation rules and diagnostics, but pure schema/capability logic no longer lives in the server god-module.
- Clean-room review: `critic` inspected the uncommitted diff and relevant call sites; no blocker findings. One non-blocking concern about brittle import-string source tests was addressed by replacing exact import-string assertions with runtime alias identity checks plus definition-removal guards.
- Validation:
  - Focused: `python3 -m py_compile codoxear/server.py codoxear/sidecar_metadata.py tests/test_sidecar_metadata.py && python3 -m pytest tests/test_sidecar_metadata.py tests/test_stale_sidecars.py tests/test_session_resume.py tests/test_launch_provenance.py tests/test_process_liveness_source.py tests/test_file_upload_module_source.py tests/test_server_queue_persistence.py -q` -> `150 passed, 25 subtests passed`.
  - Full local: `python3 -m pytest -q` -> `850 passed, 92 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `849 passed, 1 skipped, 92 subtests passed`.

## 2026-06-15 01:16
- Refreshed `recon/refactor-entry-checkpoint.md` after sidecar extraction commit `a4d24ac`.
- Updated current HEAD/date, latest validation counts, closed-gap architecture bullet, review evidence, and sidecar-discovery invariant wording.

## 2026-06-15 01:27
- Completed bounded UX feature: Details dialog now has a `Copy details` action.
- Changed artifacts: `codoxear/static/app.js`, `tests/test_diagnostics_source.py`.
- Mechanism: the copy text is built only from `diagRows`, the same label/value rows rendered in the dialog, so it does not copy the raw diagnostics object or hidden fields. The diagnostics fetch remains bound to the captured `sid`; stale selected-session responses are ignored. Copy is disabled until rows are rendered and disabled again on load error.
- Clean-room review: `critic` inspected the uncommitted diff for stale-session, secret-copy, accessibility/focus, source-test brittleness, and sparse-UI risks; no blocker findings.
- Validation:
  - Focused: `node --check codoxear/static/app.js && python3 -m py_compile tests/test_diagnostics_source.py && python3 -m pytest tests/test_diagnostics_source.py tests/test_overlay_accessibility_source.py -q` -> `10 passed`.
  - Full local: `python3 -m pytest -q` -> `852 passed, 92 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `851 passed, 1 skipped, 92 subtests passed`.

## 2026-06-15 01:28
- Refreshed `recon/refactor-entry-checkpoint.md` after Details-copy commit `0802e3f`.
- Updated current HEAD, latest validation counts, closed UX bullet, review evidence, and modal/copy invariant wording.

## 2026-06-15 01:37
- Completed bounded UX feature: file picker result paths now highlight fuzzy/exact query matches.
- Changed artifacts: `codoxear/static/app.js`, `codoxear/static/app.css`, `tests/test_file_picker_search_source.py`.
- Mechanism: `appendHighlightedFileMenuPath()` renders literal path text through text nodes and `<mark class="fileMenuMatch">` children; open/selection logic still uses original `entry.path` / `active.path` and existing title/identity hints. No raw path `innerHTML` is introduced.
- Clean-room review found a Unicode slicing issue in the first implementation: indexes computed on `toLowerCase()` could diverge from original UTF-16 indexes (`İfoo.py`) and split surrogate pairs (`a😀-b.txt`). Fixed by adding folded-string-to-original-range maps and regressions for those counterexamples. Re-review found no blockers.
- Validation:
  - Focused: `node --check codoxear/static/app.js && python3 -m py_compile tests/test_file_picker_search_source.py && python3 -m pytest tests/test_file_picker_search_source.py tests/test_file_viewer_source.py -q` -> `45 passed`.
  - Full local: `python3 -m pytest -q` -> `854 passed, 92 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `853 passed, 1 skipped, 92 subtests passed`.

## 2026-06-15 01:38
- Refreshed `recon/refactor-entry-checkpoint.md` after file-picker highlight commit `495e752`.
- Updated current HEAD, latest validation counts, closed UX bullet, clean-room review evidence, and file identity invariant wording.

## 2026-06-15 01:50
- Completed bounded architectural extraction: moved pure git helper logic from `codoxear/server.py` into `codoxear/git_ops.py` while preserving server private wrapper names and the `server._run_git` monkeypatch seam.
- Changed artifacts: `codoxear/server.py`, `codoxear/git_ops.py`, `tests/test_git_ops.py`.
- Mechanism preserved: route/session code continues calling `_resolve_git_path`, `_run_git`, `_git_repo_root`, `_current_git_branch`, `_create_git_worktree`, `_parse_git_numstat`, etc. in `server.py`; those wrappers delegate to `git_ops` and inject `_run_git` where tests and routes need patch compatibility.
- Clean-room review: architecture review initially found a semantic drift where detached HEAD (`git rev-parse --abbrev-ref HEAD` -> `HEAD`) was hidden as `None`. Fixed `git_ops.current_git_branch()` to return `branch or None` and updated tests to assert `HEAD` is preserved. Targeted re-review found the blocker cleared and no replacement blocker. Critic review found no additional blockers.
- Validation:
  - Focused: `python3 -m py_compile codoxear/server.py codoxear/git_ops.py tests/test_git_ops.py && python3 -m pytest tests/test_git_ops.py tests/test_path_resolution.py tests/test_file_inspect.py tests/test_file_search_module_source.py tests/test_session_resume.py::TestSpawnWebSessionResume::test_create_git_worktree_creates_new_checkout tests/test_session_resume.py::TestSpawnWebSessionResume::test_spawn_web_session_uses_created_worktree_as_cwd tests/test_session_sidebar_priority.py::TestSessionSidebarPriority::test_list_sessions_reads_git_branch_outside_manager_lock -q` -> `86 passed, 52 subtests passed`.
  - Full local: `python3 -m pytest -q` -> `860 passed, 92 subtests passed`.
  - Docker sandbox: `scripts/codoxear-docker-sandbox test` -> `859 passed, 1 skipped, 92 subtests passed`.

## 2026-06-15 01:51
- Refreshed `recon/refactor-entry-checkpoint.md` after git helper extraction commit `856300f`.
- Updated current HEAD, latest validation counts, closed architecture bullet, clean-room review evidence, and git/file identity invariant wording.

## 2026-06-15 01:59
- User reported two markdown rendering issues and asked to add them to the task spec:
  - code blocks in markdown use ugly dark rendering;
  - markdown tables should not run over width and should wrap/stay contained.
- Updated `.memory/tasks/2026-06-11-major-refactor-new-features/PROMPT.md` under `User-reported issue updates, 2026-06-15` with both issues, scoped as unverified UI/readability/layout bugs requiring isolated browser validation.

## 2026-06-15 02:37 — Details New-like launch preset validation
- Implemented Details dialog "New like this" launch preset review path in `codoxear/static/app.js` with allowlisted diagnostics fields only; action opens New Session for review and does not auto-start.
- Added/updated focused source/runtime tests in `tests/test_new_session_model_options_source.py`, `tests/test_launch_ui_source.py`, `tests/test_diagnostics_source.py`, and `tests/test_overlay_accessibility_source.py`.
- Focused validation: `node --check codoxear/static/app.js` passed; `python3 -m pytest tests/test_new_session_model_options_source.py tests/test_launch_ui_source.py tests/test_new_session_launch_request.py tests/test_overlay_accessibility_source.py tests/test_diagnostics_source.py -q` -> 44 passed.
- Clean-room critic re-review after final sparse Pi provider fixes: no blockers found for Pi provider corruption, auto-start, focus, or sparse UI behavior.
- Full validation: `python3 -m pytest -q` -> 871 passed, 92 subtests passed.
- Docker validation: `scripts/codoxear-docker-sandbox test` -> 870 passed, 1 skipped, 92 subtests passed.

## 2026-06-15 03:01 — Markdown rendering fix validation
- Implemented light markdown fenced-code styling and contained table layout in `codoxear/static/app.css`; added source assertions in `tests/test_markdown_tables.py`.
- Focused validation: `python3 -m pytest tests/test_markdown_tables.py tests/test_static_assets.py -q` -> 22 passed.
- Browser evidence at 390px: fenced code computed background `rgba(248, 250, 252, 0.96)`, text `rgb(17, 24, 39)`, border `rgba(15, 23, 42, 0.1)`; normal long-token table had bubble contained, wrapper `scrollWidth == clientWidth`, table width 316px, and no text over wrapper; 20-column table kept the bubble contained and used internal wrapper scroll (`scrollWidth 620`, `clientWidth 316`) rather than clipping.
- Full validation: `python3 -m pytest -q` -> 872 passed, 92 subtests passed.
- Docker validation: `scripts/codoxear-docker-sandbox test` -> 871 passed, 1 skipped, 92 subtests passed.
- Clean-room critic first found a clipping counterexample for `overflow-x:hidden`/fixed layout; after switching to `overflow-x:auto` and `table-layout:auto`, re-review found no blockers for clipping, page/bubble overflow, copy semantics, or chat/file-preview markdown paths.
- Commit: `9c49a3d fix: contain markdown tables`.

## 2026-06-14T19:48:48Z — Failed-launch recovery panel redaction validation
- Implemented failed-launch recovery UI WIP hardening in codoxear/static/app.js and server redaction in codoxear/server.py.
- Focused validation: node syntax check plus targeted failed-launch/server-broker persistence/file-viewer/provenance/sidebar/new-session/send/queue/attach/transcript pytest set => 101 passed, 12 subtests passed.
- Browser Docker fixture with failed launch record containing API_TOKEN colon syntax, JSON-style api_key, password colon syntax, Bearer token, and OPENAI_API_KEY tail: recovery panel, regular error transcript, and sidebar titles had hasSecret:false and hasDoubleBracket:false; send/queue/attach disabled; sidebar action list only Dismiss.
- Full local validation: python3 -m pytest -q => 879 passed, 104 subtests passed.
- Docker validation: scripts/codoxear-docker-sandbox test => 878 passed, 1 skipped, 104 subtests passed.
- Async critic pass sequence found and drove fixes for immediate POST response leakage, unclosed quoted env redaction, nested launch_attempt diagnostic leakage, colon/JSON secret syntax, idempotence, failed-launch attach POST affordance, raw failed-launch persistence/stderr in both server and broker recorders, and Authorization/Auth Bearer/Basic key-value leakage. Final critic run 764751f0 found no remaining failed-launch secret leakage/persistence path or mutation/autostart regression in inspected scope.
- Functional commit: f921e7e fix: recover failed launches safely.

## 2026-06-14T20:34:00Z — Video preview/transcoding validation
- Implemented a contextual file-viewer compatible-MP4 preview action in codoxear/static/app.js; source assertions updated in tests/test_file_viewer_source.py.
- Focused validation: node syntax check plus tests/test_file_viewer_source.py, ffmpeg transcode fixtures in tests/test_file_inspect.py, and tests/test_video_preview_cache.py => 33 passed.
- Isolated Docker API fixture: generated an odd-dimension MPEG4/PCM MKV under sandbox HOME; /api/files/read returned kind=video with video_preview_url; /api/files/video_preview returned video/mp4; ffprobe showed H.264/yuv420p and even encoded dimensions.
- Isolated browser fixture: same unsupported MKV; preview preflight Range bytes=0-0 returned HTTP 206 with Content-Range; Chromium video element loaded metadata from the preview URL.
- Full local validation: python3 -m pytest -q => 880 passed, 104 subtests passed.
- Docker validation: scripts/codoxear-docker-sandbox test => 879 passed, 1 skipped, 104 subtests passed.
- Clean-room critic run 55510aad-c656-4ab1-a0a9-499922da34db found no blockers in inspected scope; after the review's non-blocking note, added a VM regression proving a 500 JSON preview error renders into fileStatus without setting video src.

## 2026-06-14 20:53
- Investigated Pi busy-after-interrupt from a clean `ef7fb11` working tree.
- Local recursive Pi log schema scan under `~/.pi/agent/sessions` printed only aggregate roles/stop reasons/schema windows: 563 files, 62,272 JSON rows, 27 assistant `stopReason:"aborted"` rows, 411 assistant `stopReason:"error"` rows, 13 `length` rows, 983 `stop` rows, and 26,493 `toolUse` rows. No message text or secret/provider values were printed.
- Implemented explicit web-interrupt state in `codoxear/broker.py` and `codoxear/server.py`: `/interrupt` calls `inject_keys(..., interrupt=True)`, `SessionManager.inject_keys()` adds an `interrupt` request flag, broker `keys_handler` records the interrupt request only after writing ESC, and `_should_clear_busy_state()` allows no-candidate turn clearing only after explicit interrupt grace/quiet.
- Added/updated tests: `tests/test_broker_busy_state.py` for no-candidate silence vs explicit interrupt quiet clear, state reset on new user message, and runtime control-socket ESC write/state mutation; `tests/test_interrupt_semantics_source.py` for server/broker protocol source boundaries; `tests/test_file_upload_module_source.py` for the extended `inject_keys` signature.
- Validation: `python3 -m py_compile codoxear/broker.py codoxear/server.py` passed.
- Focused validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessiond_fail_closed.py tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_server_chat_flags.py tests/test_send_button_source.py` -> `177 passed, 22 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `885 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `884 passed, 1 skipped, 104 subtests passed`.
- Async clean-room critic for the interrupt diff started as run `1c04b3b6-fda2-4fb1-900f-8874f11e939f`; result pending at this ledger entry.

## 2026-06-14 20:55
- Local self-review follow-up: changed broker interrupt marker parsing from truthy to boolean-strict (`req.get("interrupt") is True`) and added a pending-call guard regression.
- Focused validation after follow-up: `python3 -m py_compile codoxear/broker.py codoxear/server.py` and `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py` -> `50 passed`.

## 2026-06-14 21:04
- Received clean-room critic blocker for run `1c04b3b6-fda2-4fb1-900f-8874f11e939f`: Pi tool calls were not tracked in `st.pending_calls`, so explicit interrupt could falsely idle an outstanding Pi tool turn.
- Fixed by adding Pi pending-tool helpers in `codoxear/pi_log.py` and using them in `codoxear/broker.py` for assistant `toolCall` and `toolResult` rows.
- Added regression coverage in `tests/test_broker_busy_state.py`: Pi toolCall blocks explicit-interrupt idle until matching toolResult; malformed id-less Pi toolCall fails busy-closed after interrupt.
- Validation after blocker fix: `python3 -m py_compile codoxear/pi_log.py codoxear/broker.py codoxear/server.py` passed.
- Focused validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py` -> `52 passed`.
- Broader busy/readiness validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessiond_fail_closed.py tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_server_chat_flags.py tests/test_send_button_source.py` -> `180 passed, 22 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `888 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `887 passed, 1 skipped, 104 subtests passed`.
- Started clean-room critic re-review as run `4606fe4c-7bd3-4bef-9a74-c3c4fd19ec5b`; result pending at this ledger entry.

## 2026-06-14 21:24
- Received second critic blockers from run `4606fe4c-7bd3-4bef-9a74-c3c4fd19ec5b`: (1) Pi text+toolCall rows with `stopReason:"length"` could close before pending IDs were added; (2) id-less toolResult cleared unknown pending IDs, weakening malformed fail-closed behavior.
- Fixed shared Pi final detection and broker ordering: any assistant row containing `toolCall` is not final; broker records Pi pending tool IDs before final-close logic; id-less toolResult no longer removes unknown pending IDs.
- Finished server-side explicit-interrupt idle propagation: broker state response includes `interrupted_idle` after interrupt quiet clear; server parser validates it strictly and uses it to override log-busy only with `busy:false` and `queue_len:0`.
- Added focused regressions in `tests/test_broker_busy_state.py`, `tests/test_idle_heuristics.py`, `tests/test_sessions_pending_log_idle.py`, and `tests/test_server_queue_persistence.py` for the critic counterexamples and server override boundaries.
- Focused validation: `python3 -m py_compile codoxear/pi_log.py codoxear/broker.py codoxear/server.py` plus `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py` -> `162 passed, 26 subtests passed`.
- Adjacent diagnostics/sidebar/queue/sessiond validation: `python3 -m pytest -q tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_queue_sweep_idle_guard.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `75 passed, 12 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `897 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `896 passed, 1 skipped, 104 subtests passed`.
- Started fresh clean-room critic re-review as run `4f94ff86-d8f9-45db-9b02-b67f1437fbb6`; result pending at this ledger entry.

## 2026-06-14 21:31
- Added a regression that `interrupted_idle:true` does not override a nonempty broker queue in `list_sessions()`; initial run failed because the override used public local `queue_len` instead of broker queue length.
- Fixed `list_sessions()` to carry internal `broker_queue_len` for the override and remove it before API output.
- Focused validation after fix: `python3 -m py_compile codoxear/pi_log.py codoxear/broker.py codoxear/server.py` and `python3 -m pytest -q tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_broker_busy_state.py tests/test_idle_heuristics.py` -> `158 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_queue_sweep_idle_guard.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `238 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `898 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `897 passed, 1 skipped, 104 subtests passed`.

## 2026-06-14 21:45
- Received third critic blocker from run `4f94ff86-d8f9-45db-9b02-b67f1437fbb6`: Pi final messages with thinking and final text were treated as generic thinking activity before final close.
- Fixed broker Pi ordering so final assistant text closes before generic thinking activity, after tool pending updates.
- Added broker regression for Pi thinking+final text clearing busy.
- Targeted validation: `python3 -m py_compile codoxear/pi_log.py codoxear/broker.py codoxear/server.py` and `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_idle_heuristics.py tests/test_server_chat_flags.py` -> `89 passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_queue_sweep_idle_guard.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `239 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `899 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `898 passed, 1 skipped, 104 subtests passed`.
- Started final clean-room critic re-review as run `821fcf69-84b4-4416-a515-ebc769ead20e`; result pending at this ledger entry.

## 2026-06-14 22:02
- Received fourth critic blockers from run `821fcf69-84b4-4416-a515-ebc769ead20e`: assistant-candidate interrupted clears did not report `interrupted_idle`, markers were not reset on detach/log switch, and Pi final text refused to close stale pending sentinels.
- Fixed broker interrupt idle marker, detach/log-switch marker reset, and Pi final close semantics.
- Added/extended regressions in `tests/test_broker_busy_state.py` for assistant-candidate interrupted idle marker, detach reset, log-switch reset, and Pi final text closing stale unknown pending calls.
- Targeted validation: `python3 -m py_compile codoxear/pi_log.py codoxear/broker.py codoxear/server.py` and `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_idle_heuristics.py tests/test_server_chat_flags.py` -> `92 passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_queue_sweep_idle_guard.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `242 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `902 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `901 passed, 1 skipped, 104 subtests passed`.
- Started final targeted clean-room re-review as run `5c7cc9e8-06ea-4651-b916-c0aa16f96d76`; result pending at this ledger entry.

## 2026-06-14 22:18
- Received stale server-cache blocker from run `5c7cc9e8-06ea-4651-b916-c0aa16f96d76`: cached `Session.interrupted_idle` could survive log-path changes or confirmed sends, and readiness could apply a state sample across a metadata rebind.
- Fixed server cache boundaries: clear `interrupted_idle` on `refresh_session_meta()` log-path changes; clear it and set cached busy on confirmed send success unless broker explicitly reports busy; re-query broker state in send/queue readiness when post-state metadata refresh changes log path.
- Added regressions in `tests/test_sessions_pending_log_idle.py` and `tests/test_server_queue_persistence.py` for log-change clearing, send success clearing, and state re-query on log rebind.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py` and `python3 -m pytest -q tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_broker_busy_state.py` -> `148 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_queue_sweep_idle_guard.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `244 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `904 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `903 passed, 1 skipped, 104 subtests passed`.
- Started final cache-specific critic re-review as run `0d745b8f-1dd7-44c6-9e72-1256b5915eef`; result pending at this ledger entry.

## 2026-06-14 22:31
- Received cache-specific critic blocker from run `0d745b8f-1dd7-44c6-9e72-1256b5915eef`: attachment readiness missed the send/queue post-refresh log-rebind state re-query.
- Fixed `attachment_injection_ready()` to re-query broker state if metadata refresh changes `log_path` after the first state sample.
- Added regression in `tests/test_server_queue_persistence.py` for stale interrupted-idle attachment false-idle after log rebind.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py` and `python3 -m pytest -q tests/test_server_queue_persistence.py tests/test_sessions_pending_log_idle.py tests/test_broker_busy_state.py` -> `149 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_queue_sweep_idle_guard.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `245 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `905 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `904 passed, 1 skipped, 104 subtests passed`.
- Started final-final cache review as run `5d476347-8712-4bdb-9f74-ccca4253f2c0`; result pending at this ledger entry.

## 2026-06-14 22:43
- Received final-final cache critic blocker from run `5d476347-8712-4bdb-9f74-ccca4253f2c0`: interrupted-idle readiness bypassed the confirmed-send `last_send_log_size` advancement guard because that guard only ran when broker state was busy.
- Fixed `_remote_ready_from_state_and_log()` by factoring the last-send advancement check into `_last_send_log_unadvanced()` and applying it before accepting interrupted-idle on a log that still parses busy.
- Added regression in `tests/test_server_queue_persistence.py` proving same-log unadvanced confirmed send stays not ready despite `interrupted_idle:true`, while an advanced log boundary still allows interrupted-idle recovery.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py` and `python3 -m pytest -q tests/test_server_queue_persistence.py tests/test_sessions_pending_log_idle.py tests/test_broker_busy_state.py` -> `150 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_queue_sweep_idle_guard.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `246 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `906 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `905 passed, 1 skipped, 104 subtests passed`.
- Started final-final-final read-only critic review as run `147bfab9-ee48-4689-85d6-c5528fb45e11`; result pending at this ledger entry.

## 2026-06-14 22:55
- Before finalizing, local semantic review found that message/list/diagnostics display overrides still accepted `interrupted_idle:true` before a same-log confirmed send advanced, even though mutation readiness was fixed.
- Interrupted stale critic run `147bfab9-ee48-4689-85d6-c5528fb45e11` because it was reviewing a superseded diff.
- Added shared last-send/log-size helper functions and applied the advancement boundary to `list_sessions()`, `_message_runtime_snapshot()`, and diagnostics busy computation.
- Added display regressions in `tests/test_sessions_pending_log_idle.py` for list/session and message snapshot busy state before/after confirmed-send log advancement.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py` and `python3 -m pytest -q tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_broker_busy_state.py` -> `152 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_queue_sweep_idle_guard.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `248 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `908 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `907 passed, 1 skipped, 104 subtests passed`.

## 2026-06-14 23:12
- Received critic blocker from run `9578c3af-f0d8-495f-bbb7-ecfa83314915`: same-log unadvanced confirmed sends could be bypassed if the stale current log already parsed idle.
- Fixed `_remote_ready_from_state_and_log()` so unadvanced same-log confirmed sends return not-ready before either log-idle or interrupted-idle is accepted.
- Fixed display busy computations in `list_sessions()`, `_message_runtime_snapshot()`, and diagnostics so unadvanced same-log confirmed sends force busy before idle evidence is applied.
- Added stale-idle-log regressions in `tests/test_server_queue_persistence.py` and `tests/test_sessions_pending_log_idle.py` for mutation readiness, list display, and message snapshot display.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py` and `python3 -m pytest -q tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_broker_busy_state.py` -> `155 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_queue_sweep_idle_guard.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `251 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `911 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `910 passed, 1 skipped, 104 subtests passed`.

## 2026-06-14 23:28
- Interrupted superseded final review run `d0f1cd9e-c9f5-4896-8414-82125c65c2f4` after local edge review found missing-log display/readiness needed the same `log_size is None` confirmed-send guard.
- Fixed `_remote_ready_from_state_and_log()` to apply same-log unadvanced confirmed-send rejection for existing and missing log paths.
- Fixed list/message/diagnostics display to force busy when the current log path is missing but matches an unadvanced confirmed send.
- Added missing-log regressions in `tests/test_server_queue_persistence.py` and `tests/test_sessions_pending_log_idle.py` for mutation readiness, list display, and message snapshot display.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py` and `python3 -m pytest -q tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_broker_busy_state.py` -> `158 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_queue_sweep_idle_guard.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `254 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `914 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `913 passed, 1 skipped, 104 subtests passed`.

## 2026-06-14 23:48
- Received blockers from narrow critic run `41288862-49a9-408c-8f0f-82bff68b89f3`: active no-log confirmed sends were indistinguishable from default `None/None` state, and duplicate Pi `toolCall.id` values collapsed in `pending_calls`.
- Added explicit `Session.last_send_boundary_active`; successful confirmed sends set it true, and readiness/display boundary checks require it before treating no-log or path/size state as unresolved.
- Updated list/message/diagnostics/readiness boundary checks so active no-log sends block pending-bind idle, while inactive default no-log sessions remain idle.
- Updated Pi tool-call ID handling so duplicates in one assistant row become unknown sentinels and duplicates across rows add an unknown sentinel if the concrete ID is already pending.
- Added regressions for no-log confirmed-send readiness/list/message behavior and duplicate Pi tool-call IDs in one row and across rows.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py` and `python3 -m pytest -q tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_broker_busy_state.py` -> `163 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `259 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `919 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `918 passed, 1 skipped, 104 subtests passed`.

## 2026-06-14 23:58
- Interrupted stale final critic run `b28ff477-b90a-42e2-a2a1-37658929d995` after tightening the no-log boundary semantics.
- Updated no-log confirmed-send boundary to remain unresolved for absent or zero-byte current logs; added regression for zero-byte log path becoming ready only after bytes appear.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py` and `python3 -m pytest -q tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_broker_busy_state.py` -> `164 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `260 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `920 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `919 passed, 1 skipped, 104 subtests passed`.

## 2026-06-15 00:10
- Received critic blocker from run `813ee469-c34c-428f-a98d-5dbc88af1dfc`: active no-log confirmed-send boundary was not cleared after non-empty log evidence resolved it, so later detach/log_path=None could resurrect stale busy/not-ready state.
- Added central consuming boundary evaluator on `SessionManager`: resolved active boundaries clear `last_send_boundary_active`, `last_send_log_path`, and `last_send_log_size` under lock.
- Updated readiness, list display, message snapshots, and diagnostics to use the consuming evaluator; message snapshot retains a fallback for tests that patch `MANAGER` with a lightweight stub.
- Extended no-log boundary regressions for readiness, list display, and message snapshot: absent/zero-byte logs block; non-empty log evidence clears marker; later log_path=None remains idle/not-ready-free.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py` and `python3 -m pytest -q tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_broker_busy_state.py` -> `164 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `260 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `920 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `919 passed, 1 skipped, 104 subtests passed`.

## 2026-06-15 00:24
- Received critic blocker from run `eafea54e-add1-49d3-85ad-9ead8c5a2aca`: duplicate Pi tool-call IDs were converted into anonymous unknown sentinels, so a second matching concrete tool result could not clear the duplicate occurrence.
- Updated duplicate Pi tool-call handling to use per-ID duplicate sentinels (`__pi_duplicate_tool_call__:<encoded-id>:...`) and to let each matching `toolResult.toolCallId` clear one occurrence after the concrete ID is gone.
- Updated regressions for duplicate IDs in one assistant row and across rows: one result leaves a duplicate sentinel pending; the second matching result clears pending calls and allows the explicit-interrupt quiet path.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py` and `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py` -> `164 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `260 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `920 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `919 passed, 1 skipped, 104 subtests passed`.

## 2026-06-15 00:52
- Received critic blockers from run `32447d72-ec5f-4c7a-9fa1-f4d064936548`: Pi bind skipped pre-existing rows while advancing `log_off`, and duplicate pending string sentinels could collide with real tool IDs shaped like the sentinel prefix.
- Replaced Pi duplicate/unknown string sentinels with typed internal pending keys (`PiDuplicateToolCallId`, `PiUnknownToolCallId`). Matching concrete `toolResult.toolCallId` clears the concrete ID first, then one typed duplicate key with the same real ID; id-less unknown keys remain uncleared by id-less results.
- Added `pi_current_turn_state_before()` to seed Pi pending/idle state by scanning current-turn rows before broker bind/rebind advances `log_off`.
- Updated broker bind/rebind to call Pi seeding for `AGENT_BACKEND=pi` just as it already seeded Claude Code state.
- Added regressions for sentinel-shaped real IDs, Pi current-turn pre-bind tool-call seeding, duplicate multiplicity during Pi seeding, id-less unknown Pi seeding, and broker bind seeding before `log_off` advancement.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py` and `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py` -> `169 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `265 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `925 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `924 passed, 1 skipped, 104 subtests passed`.

## 2026-06-15 01:10
- Received critic blockers from run `065436df-dc0d-4866-a776-111c72a56135`: bind/rebind merged old pending calls with seed pending, and Pi bind set `log_off` past a trailing partial JSONL row that the seeding scanner intentionally dropped.
- Updated broker bind/rebind to replace `st.pending_calls` with `set(seed_pending)` instead of merging.
- Added `pi_complete_jsonl_offset_before()` and made Pi current-turn seeding operate only over complete JSONL bytes. Broker Pi bind/rebind now sets `st.log_off` to that complete offset, not physical file size, so a trailing partial row is replayed when newline-completed.
- Added regressions for stale pending replacement on Pi log switch and trailing partial Pi tool-call rows not being skipped after completion.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py` and `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py` -> `171 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py` -> `267 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `927 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `926 passed, 1 skipped, 104 subtests passed`.

## 2026-06-15 01:35
- Received critic blockers from run `8c98fbe7-32ca-4b63-bfc6-3d1f84ea9430`: confirmed-send boundaries resolved on raw file-size growth from partial JSONL rows, and broker live tailing could advance over oversized unterminated fragments.
- Added server `_complete_jsonl_offset_before()` and changed confirmed-send boundary sizing to last newline-complete JSONL offset rather than raw `stat().st_size`.
- Added `advance_on_oversized_unterminated` to `util.read_jsonl_from_offset()` with default `True` for existing generic behavior. Broker live tailing opts into `False`, preserving offsets when no newline is observed.
- Added regressions: send readiness does not clear stale/no-log confirmed-send boundaries on partial rows; list/message snapshot no-log boundaries stay busy on partial rows; broker JSONL reader does not advance over large unterminated fragments and can process the completed row later.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py codoxear/util.py` and `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_read_jsonl_from_offset.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py` -> `178 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py tests/test_read_jsonl_from_offset.py tests/test_broker_fail_closed.py` -> `301 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `928 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `927 passed, 1 skipped, 104 subtests passed`.

## 2026-06-15 01:58
- Received critic blocker from run `2321773f-37eb-47e3-87b9-7c4bef377f23`: confirmed-send boundary resolution still accepted newline-complete but blank/malformed/non-object rows as evidence.
- Added server `_last_parseable_json_object_offset_before()` and changed `_log_path_size_or_none()` to return the offset after the last parseable JSON object row, not the last newline-complete byte.
- Extended regressions for same-log stale idle boundaries and no-log boundaries: blank rows, malformed rows, JSON arrays, and trailing partial rows all keep boundaries active; a parseable JSON object row resolves them.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py codoxear/util.py` and `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_read_jsonl_from_offset.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py` -> `178 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py tests/test_read_jsonl_from_offset.py tests/test_broker_fail_closed.py` -> `301 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `928 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `927 passed, 1 skipped, 104 subtests passed`.

## 2026-06-15 02:20
- Received critic blockers from run `b08b02e8-dbfd-4116-9bdc-a5177c0fe76d`: stale broker tail batches could apply after log rebind, and non-dict JSONL rows could reach metadata analysis despite the `list[dict]` reader contract.
- Updated broker `_log_watcher()` to discard a read batch unless current state still matches the captured `log_path` and `log_off`; row processing and `log_off` advancement now happen under the same lock/path/offset association, with offset advanced after processing.
- Updated `util.read_jsonl_from_offset()` to skip decoded non-dict JSON rows.
- Added regressions for stale tail-batch discard after rebind, non-dict JSONL reader rows, and `list_sessions()` with a JSON array row.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py codoxear/util.py` and `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_read_jsonl_from_offset.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py` -> `181 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py tests/test_read_jsonl_from_offset.py tests/test_broker_fail_closed.py` -> `304 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `931 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `930 passed, 1 skipped, 104 subtests passed`.

## 2026-06-15 02:35
- Interrupted stale critic run `ed10ae34-f69d-4a1d-8829-7007b6521e5f` after static review found a new local registration-offset fix.
- Updated broker `_register_from_log()` to use `_pi_complete_jsonl_offset_before()` for Pi logs instead of raw file size.
- Added regression `test_register_from_pi_log_uses_complete_jsonl_offset`.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py codoxear/util.py` and `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_read_jsonl_from_offset.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py` -> `182 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py tests/test_read_jsonl_from_offset.py tests/test_broker_fail_closed.py` -> `305 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `932 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `931 passed, 1 skipped, 104 subtests passed`.

## 2026-06-15 02:55
- Received critic blocker from run `9efb716c-be43-422d-b07c-c77de5ad3e29`: broker no-advance JSONL tailing could preserve offset forever for completed rows larger than `max_bytes + chunk_size`.
- Updated `util.read_jsonl_from_offset(..., advance_on_oversized_unterminated=False)` to continue scanning until newline or EOF, preserving offset only if EOF is reached before any newline.
- Added regression `test_broker_jsonl_reader_processes_completed_row_beyond_bounded_window` with a ~600 KiB completed Pi row and broker `max_bytes=256 KiB`.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py codoxear/util.py` and `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_read_jsonl_from_offset.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py` -> `183 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py tests/test_read_jsonl_from_offset.py tests/test_broker_fail_closed.py` -> `306 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `933 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `932 passed, 1 skipped, 104 subtests passed`.

## 2026-06-15 03:18
- Received critic blockers from run `146d49f1-eee5-4611-b4e4-0b91a1fb4d0f`: same-path confirmed-send boundary with `last_send_log_size=None` blocked forever, and `get_state()` did not refresh cached `interrupted_idle`.
- Updated `_confirmed_send_boundary_unresolved()` so same-path `last_send_log_size is None` resolves once `log_size > 0` using the parseable-row evidence offset.
- Updated `SessionManager.get_state()` to parse `_broker_interrupted_idle_from_state(resp)` and assign `s2.interrupted_idle` with cached `busy` and `queue_len`.
- Added regressions for missing-log same-path unknown baseline resolving on parseable row (not blank row) and `get_state()` clearing stale cached interrupted idle.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py codoxear/util.py` and `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_read_jsonl_from_offset.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py` -> `185 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py tests/test_read_jsonl_from_offset.py tests/test_broker_fail_closed.py` -> `308 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `935 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `934 passed, 1 skipped, 104 subtests passed`.

## 2026-06-15 03:42
- Received critic blocker from run `ea736dc9-0dc9-4e11-8352-3df3728ae60b`: whitespace-only Pi tool-call IDs were treated as malformed due to `.strip()` checks.
- Updated `pi_assistant_pending_tool_call_ids()` and `pi_tool_result_id()` to accept any string ID exactly; only absent/non-string IDs are malformed.
- Added regressions for live broker application and current-turn seeding with `" "` tool IDs/results.
- Focused validation: `python3 -m py_compile codoxear/server.py codoxear/broker.py codoxear/pi_log.py codoxear/util.py` and `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_read_jsonl_from_offset.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py` -> `187 passed, 26 subtests passed`.
- Adjacent validation: `python3 -m pytest -q tests/test_broker_busy_state.py tests/test_interrupt_semantics_source.py tests/test_file_upload_module_source.py tests/test_idle_heuristics.py tests/test_sessions_pending_log_idle.py tests/test_server_queue_persistence.py tests/test_queue_sweep_idle_guard.py tests/test_diagnostics_source.py tests/test_launch_provenance.py tests/test_session_sidebar_priority.py tests/test_server_chat_flags.py tests/test_sessiond_fail_closed.py tests/test_send_button_source.py tests/test_read_jsonl_from_offset.py tests/test_broker_fail_closed.py` -> `310 passed, 38 subtests passed`.
- Full local validation: `python3 -m pytest -q` -> `937 passed, 104 subtests passed`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` -> `936 passed, 1 skipped, 104 subtests passed`.

## 2026-06-15 04:08
- Final narrow critic `809c69e7-147b-4201-aed0-4f1565b0cb94` returned `NO BLOCKERS`.
- Residual risks: repeated reads for huge unterminated partial broker rows; unobserved Pi empty normal final-close row shape would still not clear pending calls.
- Proceeding to stage explicit functional files for an atomic Pi repair commit; memory/checkpoint files remain unstaged for a separate docs/memory commit.

## 2026-06-15 04:13
- Functional Pi repair committed: `9e2d4b8 fix Pi interrupt busy-state accounting`.
- Updated `recon/refactor-entry-checkpoint.md` to record the Pi busy-after-interrupt repair, validation evidence, and residual scoped risks.

## 2026-06-15 09:05
- Started isolated Codex live server attempts under temp HOME with real `CODEX_HOME=/home/yiwen/.codex` and disposable password; initial `python3 -m codoxear.server` failed due missing local `py_vapid`, so a temp venv was created and used for live runs.
- Tmux launch attempt reproduced an isolation hazard: a web tmux broker inherited the pre-existing tmux server HOME `/home/yiwen`, wrote a sidecar under the live app dir, and stopped at Codex cwd trust. Cleaned only the uniquely identified test broker pid/sidecar/socket (`broker-1394461`) and switched to `create_in_tmux=false` for isolated proof.
- Direct web-owned Codex run under temp HOME reached Codex, accepted temp cwd trust, and produced real rollout response `BOOTSTRAP_OK_20260615`, but broker failed to bind the log. The rollout cwd was `/.tmp-on-ssd/.../work`; Codoxear discovery filtered on `/tmp/.../work`.
- Patched `codoxear/util.py` so `proc_find_open_rollout_log()` accepts cwd identity by exact string, `os.path.samefile()`, or resolved-path fallback, fail-closed on comparison errors.
- Added regressions in `tests/test_broker_proc_rollout.py` for symlink-resolved cwd identity and samefile/bind-alias cwd identity.
- Focused validation: `python3 -m py_compile codoxear/util.py tests/test_broker_proc_rollout.py` and `python3 -m pytest -q tests/test_broker_proc_rollout.py tests/test_session_resume.py tests/test_stale_sidecars.py` -> `59 passed`.
- Full local validation: `python3 -m pytest -q` -> `939 passed, 104 subtests passed`.
- Docker validation: `scripts/codoxear-docker-sandbox test` -> `938 passed, 1 skipped, 104 subtests passed`.
- Fresh isolated live proof after patch: temp HOME `/tmp/codoxear-live-codex4.4n15ph/home`, port `19044`, direct web-owned broker pid `1460449`; accepted temp cwd trust; bootstrap prompt created rollout `019ec8bc-f8f5-7912-8808-0debef74d6bd`; browser composer sent final prompt; `/messages/tail` showed user/assistant sequence ending in assistant `CODEX_WEB_LIVE_OK_20260615`; session list reported `busy=false`, `queue_len=0`; API delete returned `{"ok": true}`; browser/server/test-root processes were stopped. The real Codex rollout log remains in `CODEX_HOME` as backend history evidence.
- Started narrow critic `7d128b0c-f4b4-4481-8f19-5ad5143b4366` on the cwd identity diff; result pending at this ledger entry.

## 2026-06-15 09:22
- Received critic blocker from `7d128b0c-f4b4-4481-8f19-5ad5143b4366`: non-strict `Path.resolve()` fallback could match nonexistent payload cwd `/tmp/work/missing/..` to launched cwd `/tmp/work` after `samefile()` raised.
- Changed `proc_find_open_rollout_log()` cwd alias matching to fail closed on `samefile()` exceptions; exact string equality remains the only path that does not require absolute existing samefile identity.
- Added regression `test_proc_rejects_nonexistent_payload_cwd_resolve_alias`.
- Focused validation: `python3 -m py_compile codoxear/util.py tests/test_broker_proc_rollout.py` and `python3 -m pytest -q tests/test_broker_proc_rollout.py tests/test_session_resume.py tests/test_stale_sidecars.py` -> `61 passed`.
- Full local validation: `python3 -m pytest -q` -> `941 passed, 104 subtests passed`.
- Docker validation: `scripts/codoxear-docker-sandbox test` -> `940 passed, 1 skipped, 104 subtests passed`.

## 2026-06-15 09:36
- Received critic blocker from `a2069118-5b76-4a93-a589-a229492c467e`: `expanduser()` before the absolute-path gate allowed raw payload cwd `"~"` to bind to `$HOME` and allowed `"~nosuchuser..."` to raise before fail-closed handling.
- Removed `expanduser()` from `proc_find_open_rollout_log()` cwd alias matching; exact string equality is unchanged, alias matching now requires raw absolute paths plus successful `os.path.samefile()`.
- Added regressions for payload cwd `"~"` not binding to `Path.home()` and `"~nosuchuser123456/work"` returning `None` rather than raising.
- Focused validation: `python3 -m py_compile codoxear/util.py tests/test_broker_proc_rollout.py` and `python3 -m pytest -q tests/test_broker_proc_rollout.py tests/test_session_resume.py tests/test_stale_sidecars.py` -> `63 passed`.
- Full local validation: `python3 -m pytest -q` -> `943 passed, 104 subtests passed`.
- Docker validation: `scripts/codoxear-docker-sandbox test` -> `942 passed, 1 skipped, 104 subtests passed`.

## 2026-06-15 09:45
- Final-final Codex cwd alias critic `5df64f7b-12c0-4e8c-a65b-f36985c79e35` returned `NO BLOCKERS`.
- Functional Codex binding fix committed: `2d8e1e2 fix Codex rollout cwd alias binding`.
- Updated `recon/refactor-entry-checkpoint.md` to record the direct web-owned Codex browser-send/final-response evidence, validation counts, and tmux isolation caveat.

## 2026-06-15 09:27
- Implemented Claude Code closed-log binding fallback after live evidence showed Claude writes JSONL logs under `.claude/projects` without exposing a writable log fd through the broker's current `/proc` open-file discovery path.
- Changed shared cwd matching so both `proc_find_open_rollout_log()` and `find_new_session_log()` accept either exact cwd strings or existing absolute filesystem identity via `os.path.samefile()`; relative, tilde, unknown-user, and nonexistent aliases fail closed.
- Changed `read_cc_session_header()` to merge early CC metadata rows because live CC logs begin with mode/permission rows containing `sessionId` but no `cwd`; the scan is capped at 512 KiB.
- Added regressions in `tests/test_broker_proc_rollout.py`, `tests/test_cc_log.py`, and `tests/test_claude_backend_source.py` for CC samefile cwd alias discovery, malformed alias rejection, bounded header merge, and broker fallback source wiring.
- Focused validation after hardening: `python3 -m pytest -q tests/test_broker_proc_rollout.py tests/test_claude_backend_source.py tests/test_cc_log.py tests/test_cc_chat_and_idle.py tests/test_cc_busy_state.py tests/test_session_resume.py tests/test_stale_sidecars.py` passed: 100 tests.
- Full local validation: `python3 -m pytest -q` passed: 948 tests and 104 subtests.
- Docker validation: `scripts/codoxear-docker-sandbox test` passed: 947 tests, 1 skipped, and 104 subtests.
- Isolated Claude live run on port 19048 launched web-owned direct broker PID 1630561 with temp HOME/root `/tmp/codoxear-live-cc4.EeoSB2`; first-run trust prompt was accepted for the empty temp workspace.
- Browser UI sent: `Reply with exactly CLAUDE_WEB_LIVE_OK_20260615 and nothing else.`; composer cleared and the user prompt appeared in transcript.
- CC fallback bound placeholder `broker-1630561` to real thread `410ef3d0-6967-49cd-9488-45b30c40f5d6` and log path `/home/yiwen/.claude/projects/--tmp-on-ssd-codoxear-live-cc4-EeoSB2-work/410ef3d0-6967-49cd-9488-45b30c40f5d6.jsonl`.
- Claude upstream returned repeated 503 connection failures, then wrote a synthetic assistant API-error row plus `turn_duration`; Codoxear `/messages/tail` showed the user prompt and assistant API error, and `/api/sessions` / tail both reported `busy=false`, `queue_len=0`.
- Cleaned the isolated Claude session via API delete (`{"ok": true}`) and stopped the isolated browser/server/broker processes; no live checkout processes were touched.

## 2026-06-15 09:34
- Clean-room critic subagent `05290a8a-033a-46c1-ab02-c0d8f52d3254` returned blockers for the first CC fallback patch: (1) CC could create and close its log before the broker populated `known_rollout_paths`, causing the new log to be skipped forever as preexisting; (2) relative `--cwd` remained relative in broker state, preventing fallback matching against CC's absolute log cwd.
- Fixed the lifecycle race by snapshotting Pi/CC known rollout/session logs before fork/exec, immediately after `sessions_dir` exists and before either `os.fork()` or `pty.fork()`; `State.known_rollout_paths` now receives this prelaunch snapshot.
- Fixed relative cwd matching by making broker `_expand_cwd()` return an absolute path after HOME/env/tilde expansion.
- Added regressions for the critic counterexamples: post-start CC log found with a prelaunch snapshot but missed with a post-fork snapshot, relative `--cwd .` expansion matching an absolute CC log cwd, and source-order guard proving the prelaunch snapshot happens before both fork paths.
- Blocker-focused validation: `python3 -m pytest -q tests/test_broker_proc_rollout.py::TestBrokerProcRolloutDiscovery::test_find_new_cc_session_log_finds_after_start_log_with_prelaunch_snapshot tests/test_broker_proc_rollout.py::TestBrokerProcRolloutDiscovery::test_find_new_cc_session_log_matches_relative_broker_cwd_after_expansion tests/test_claude_backend_source.py::TestClaudeBackendSource::test_broker_has_cc_closed_log_discovery_fallback` passed: 3 tests.
- Focused CC/Codex validation: `python3 -m pytest -q tests/test_broker_proc_rollout.py tests/test_claude_backend_source.py tests/test_cc_log.py tests/test_cc_chat_and_idle.py tests/test_cc_busy_state.py tests/test_session_resume.py tests/test_stale_sidecars.py` passed: 102 tests.
- Full local validation: `python3 -m pytest -q` passed: 950 tests and 104 subtests; `git diff --check` clean.
- Docker validation: `scripts/codoxear-docker-sandbox test` passed: 949 tests, 1 skipped, and 104 subtests.

## 2026-06-15 09:40
- Clean-room critic subagent `62c6924a-cbdf-4535-b3d8-d6886680fd2a` confirmed the race and relative-cwd blockers were fixed, then found a remaining blocker: `read_cc_session_header()` bounded by line end offset, so a valid first CC JSONL row over 512 KiB was discarded after being read, making closed-log fallback miss large first prompts.
- Fixed CC header scanning so the 512 KiB cap bounds which row start offsets are considered; a valid JSONL line whose start is inside the window is still parsed even if its end crosses the cap.
- Added regressions for a >600 KiB valid first CC user row: `read_cc_session_header()` returns session id/cwd/timestamp, and `find_new_session_log(... agent_backend="cc", cwd=...)` finds the log.
- Blocker-focused validation: `python3 -m pytest -q tests/test_cc_log.py::TestCcLog::test_read_session_header_parses_large_valid_first_record tests/test_broker_proc_rollout.py::TestBrokerProcRolloutDiscovery::test_find_new_cc_session_log_finds_large_valid_first_record tests/test_cc_log.py::TestCcLog::test_read_session_header_scan_is_bounded` passed: 3 tests.
- Focused CC/Codex validation: `python3 -m pytest -q tests/test_broker_proc_rollout.py tests/test_claude_backend_source.py tests/test_cc_log.py tests/test_cc_chat_and_idle.py tests/test_cc_busy_state.py tests/test_session_resume.py tests/test_stale_sidecars.py` passed: 104 tests.
- Full local validation: `python3 -m pytest -q` passed: 952 tests and 104 subtests; `git diff --check` clean.
- Docker validation: `scripts/codoxear-docker-sandbox test` passed: 951 tests, 1 skipped, and 104 subtests.

## 2026-06-15 09:43
- Final narrow critic subagent `6f5dbf25-e41e-4467-8760-66e781c6809e` returned `NO BLOCKERS` for the CC fallback/header candidate after the race, relative-cwd, and large-first-row fixes.
- Critic independently validated prelaunch snapshot ordering before both fork paths, absolute broker cwd expansion, safe exact-or-absolute-samefile payload cwd matching, large first-row header parsing, and rows-starting-after-window exclusion. Its focused run passed: 28 tests.
- Functional commit created: `c1280cb fix Claude Code closed-log binding`.

## 2026-06-15 09:57
- User corrected validation scope: this tranche must be validated only in Docker. A host-side temp-HOME prefix server/browser attempt for the URL helper extraction was stopped and its evidence was discarded.
- Implemented the first bounded frontend refactor tranche by extracting app URL/base-path resolution from `codoxear/static/app.js` into `codoxear/static/app_url.js`.
- `index.html` now loads `app_url.js` before `app.js`; `app.js` fails loudly if `window.CodoxearUrls.resolveAppUrl` is unavailable instead of recomputing a silent fallback.
- Added `app_url.js` to static asset versioning and top-level static routing; package data already includes direct `static/*` files, and tests assert wheel inclusion.
- Added `tests/test_frontend_url_module_source.py` plus static-asset regressions for URL-prefix resolution, script order, fail-loud dependency, cache versioning, route availability, and packaging.
- Docker focused validation: `scripts/codoxear-docker-sandbox test tests/test_frontend_url_module_source.py tests/test_static_assets.py tests/test_url_prefix.py tests/test_session_polling_source.py` passed: 29 tests and 3 subtests.
- Docker runtime route validation under `CODEX_WEB_URL_PREFIX=/codoxear`: in-container requests returned `/codoxear/api/me -> 401`, `/codoxear/app_url.js?v=test -> 200` with helper content, and `/codoxear/app.js?v=test -> 200`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed: 955 tests, 1 skipped, and 107 subtests.
- Read-only critic subagent `82cb6205-46b9-428c-97e2-ded96036dd5a` returned `NO BLOCKERS`; it did not run tests and inspected static serving, asset versioning, packaging, script ordering, fail-loud behavior, URL-prefix behavior, CSP, service-worker path behavior, and broad UI semantic scope.
- Functional commit created: `a427dab extract frontend URL helper`.

## 2026-06-15 10:10
- Implemented the second bounded frontend refactor tranche by extracting local-storage access from `codoxear/static/app.js` into `codoxear/static/app_storage.js`.
- `index.html` now loads `app_url.js`, then `app_storage.js`, then `app.js`; `app.js` keeps the existing `optionalLocalStorage`, `storageGetItem`, `storageSetItem`, and `storageRemoveItem` names as wrappers and fails loudly if `window.CodoxearStorage` is unavailable.
- Added `app_storage.js` to static asset versioning and top-level static routing; package data already includes direct `static/*` files, and tests assert wheel inclusion.
- Updated `tests/test_storage_robustness_source.py` to evaluate the extracted storage module directly while preserving denied/throwing storage semantics.
- Docker focused validation: `scripts/codoxear-docker-sandbox test tests/test_storage_robustness_source.py tests/test_static_assets.py tests/test_frontend_url_module_source.py tests/test_file_picker_session_state.py tests/test_new_session_model_options_source.py` passed: 39 tests and 3 subtests.
- Docker runtime route validation under `CODEX_WEB_URL_PREFIX=/codoxear`: in-container requests returned `/codoxear/api/me -> 401`, `/codoxear/app_url.js?v=test -> 200`, `/codoxear/app_storage.js?v=test -> 200` with helper content, and `/codoxear/app.js?v=test -> 200`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed: 955 tests, 1 skipped, and 107 subtests.
- Read-only critic subagent `5679462c-49f1-4e14-aecb-ada1b99a3f80` returned `NO BLOCKERS`; it did not run tests and inspected storage-denial behavior, script ordering, fail-loud dependency, static routing/versioning/package inclusion, CSP/path behavior, and helper-name compatibility.
- Functional commit created: `8f43ef8 extract frontend storage helper`.

## 2026-06-15 10:21
- Implemented the third bounded frontend refactor tranche by extracting performance-sampling diagnostics from `codoxear/static/app.js` into `codoxear/static/app_perf.js`.
- `index.html` now loads `app_url.js`, `app_storage.js`, `app_perf.js`, then `app.js`; deferred script order means `app.js` sees `window.CodoxearPerf` before its fail-loud dependency check.
- `app.js` keeps the existing `pushPerfSample` wrapper and `window.codoxearPerf` diagnostic entry point, delegating sample insertion and summaries to `window.CodoxearPerf.pushSample/summarize`.
- Added `app_perf.js` to static asset versioning and top-level static routing; tests assert route source, version inclusion, CSP/no-third-party constraints, and wheel package inclusion.
- Docker focused validation: `scripts/codoxear-docker-sandbox test tests/test_frontend_perf_module_source.py tests/test_static_assets.py tests/test_frontend_url_module_source.py tests/test_storage_robustness_source.py tests/test_session_polling_source.py` passed: 31 tests and 3 subtests.
- Docker runtime route validation under `CODEX_WEB_URL_PREFIX=/codoxear`: in-container requests returned `/codoxear/api/me -> 401`, `/codoxear/app_url.js?v=test -> 200`, `/codoxear/app_storage.js?v=test -> 200`, `/codoxear/app_perf.js?v=test -> 200` with `window.CodoxearPerf`, and `/codoxear/app.js?v=test -> 200`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed: 958 tests, 1 skipped, and 107 subtests.
- Read-only critic subagent `690d8a0d-02e9-4a46-974c-f0d925df8523` returned `NO BLOCKERS`; it inspected sample-window/filter/percentile/rounding semantics, script order, `window.codoxearPerf` compatibility, static routing/versioning/package inclusion, CSP/path behavior, and Docker-only evidence. Non-blocking note: stale cached old `index.html` plus new `app.js` would fail loudly because the old shell lacks `app_perf.js`; default no-store and asset versioning keep this within the existing stale-static limitation.
- Functional commit created: `82e674d extract frontend performance helper`.

## 2026-06-15 10:28
- Implemented the next bounded frontend/server refactor by centralizing static frontend asset registration in `codoxear/server.py`.
- Added `FRONTEND_ASSET_FILES` as the ordered source for versioned frontend assets and `TOP_LEVEL_STATIC_ASSETS` as the source for exact top-level static routes; `STATIC_ASSET_VERSION_FILES` now aliases `FRONTEND_ASSET_FILES`.
- Replaced the repeated `_handle_static_get` if-chain for top-level static files with registry iteration while leaving `/static/*` handling, `_send_static`, HTML placeholder replacement, CSP, cache headers, content types, and package-data globs unchanged.
- Updated `tests/test_static_assets.py` to assert the manifest drives version files and top-level routes, and that every frontend asset mutation changes the static version.
- Local syntax validation: `python3 -m py_compile codoxear/server.py tests/test_static_assets.py` passed.
- Docker focused validation: `scripts/codoxear-docker-sandbox test tests/test_static_assets.py tests/test_url_prefix.py tests/test_frontend_url_module_source.py tests/test_storage_robustness_source.py tests/test_frontend_perf_module_source.py` passed: 24 tests and 3 subtests.
- Docker runtime route validation under `CODEX_WEB_URL_PREFIX=/codoxear`: `/codoxear/api/me -> 401`; `/codoxear/app_url.js`, `/codoxear/app_storage.js`, `/codoxear/app_perf.js`, `/codoxear/app.js`, `/codoxear/app.css`, `/codoxear/favicon.png`, `/codoxear/manifest.webmanifest`, `/codoxear/service-worker.js`, and `/codoxear/static/app.js` all returned 200.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed: 959 tests, 1 skipped, and 107 subtests.
- Read-only critic subagent `35973f5f-5be7-4c9f-80f4-15ac71ced0e9` returned `NO BLOCKERS`; it confirmed route mappings, URL-prefix stripping order, `/static/*` handling, version hash coverage/order, unchanged `_send_static`/CSP/cache/content-type/package behavior, and Python runtime compatibility. Non-blocking notes: tests do not assert route order because exact-route order is non-behavioral here; service worker/manifest/favicon remain outside the asset-version hash and all static files share the same cache policy as pre-existing behavior.
- Functional commit created: `70fc3a1 centralize frontend static asset registry`.

## 2026-06-15 10:29
- Refreshed `.memory/tasks/2026-06-11-major-refactor-new-features/PROMPT.md` from the stale long workbench into a short current action list.
- The refreshed prompt directs future work to final clean-room review, smallest-scope blocker repair if needed, final acceptance summary if no blocker remains, and explicit user approval before any merge/promotion.
- It records that Pi busy-after-interrupt, Codex live web-send binding, Claude Code closed-log/API-error idle path, markdown containment, video preview, and bounded frontend helper/static-registry refactors are closed unless new evidence reopens them.

## 2026-06-15 10:35
- Final pre-summary clean-room critic `90a8597a-fa0a-4a8a-b240-8e5989960b39` returned `NO BLOCKERS` for current `recovery/product-gaps` before the summary rewrite.
- Replaced historical `develop` content in `recon/final-acceptance-summary.md` with a current `recovery/product-gaps` acceptance summary covering integrated work, Docker-only frontend/static validation, backend/live evidence, clean-room reviews, and parked limits.
- Refreshed `.memory/tasks/2026-06-11-major-refactor-new-features/PROMPT.md` workbench to the completed-state action list: await explicit user approval, repair only requested changes/blockers, or prepare a merge/promotion plan only after approval.

## 2026-06-15 10:39
- Final summary gate critic `2a5384ff-5b68-4d64-b07c-9a2fa32da6c9` returned `BLOCKERS`: `recon/final-acceptance-summary.md` omitted two parked limits from `recon/refactor-entry-checkpoint.md`.
- Repaired the summary by explicitly parking that broad structural/frontend refactoring is not complete and that full live backend lifecycle evidence remains incomplete beyond scoped live paths.

## 2026-06-15 10:41
- Final summary gate rerun critic `2ed71af5-3fe4-4b94-9b74-f229773d1d0f` returned `NO BLOCKERS` after the parked-limit repair.
- Reviewer confirmed the uncommitted diff was docs/task-memory only, the prior parked-limit blocker was repaired, Docker-only frontend/static evidence was scoped correctly, the prompt was in completed-state workbench form, `git diff --check` was clean, and no untracked secret/runtime artifacts were found by its inspection.

## 2026-06-15 10:59
- User clarified that broad refactor work remains and that the promotion-plan detour was unauthorized.
- Discarded the uncommitted promotion-plan artifact and uncommitted plan-related task-memory edits; no `main` or protected checkout mutation had occurred.
- Reset `PROMPT.md` from approval/promotion-planning state back to active bounded refactor work, with markdown-renderer extraction as the current tranche.

## 2026-06-15 11:31
- Implemented the next bounded frontend refactor tranche by extracting markdown rendering/cache/preview and local file-reference parsing from `codoxear/static/app.js` into `codoxear/static/app_markdown.js`.
- `index.html` now loads `app_url.js`, `app_storage.js`, `app_perf.js`, `app_markdown.js`, then `app.js`; `app.js` keeps wrapper names for `normalizeLineNumber`, `parseLocalFileRef`, `isMarkdownPreviewable`, `markdownPreviewHtml`, and `chatMarkdownHtmlCached`, and fails loudly if `window.CodoxearMarkdown` is missing or incomplete.
- Added `app_markdown.js` to `FRONTEND_ASSET_FILES`, top-level static routes, asset versioning, and wheel/static source tests.
- Added/updated source/VM tests so markdown renderer/table tests evaluate `app_markdown.js` directly, static tests assert load order/package inclusion, file-picker snippets retain app-owned `stripPathLocationSuffix`, and file-viewer tests execute the non-literal `openFileReference()` parser path.
- Local focused validation: `node --check codoxear/static/app.js`, `node --check codoxear/static/app_markdown.js`, and `python3 -m pytest -q tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_markdown_renderer_source.py tests/test_markdown_tables.py tests/test_static_assets.py` passed: 80 tests.
- Docker focused validation: `scripts/codoxear-docker-sandbox test tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_markdown_renderer_source.py tests/test_markdown_tables.py tests/test_static_assets.py` passed: 80 tests.
- Docker prefixed-route validation under `CODEX_WEB_URL_PREFIX=/codoxear`: `/codoxear/api/me -> 401`; `/codoxear/app_markdown.js?v=test -> 200`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed: 962 tests, 1 skipped, 107 subtests.
- First clean-room critic `85d8069d-4e2f-40ef-bf96-eee6545b280d` returned `BLOCKERS` for a missing `parseLocalFileRef` global on the non-literal `openFileReference()` path; the blocker was fixed by exporting the parser through `window.CodoxearMarkdown`, adding an app wrapper/fail-loud check, and adding the executable VM regression.
- Clean-room critic rerun `4773fc5a-bbbf-4d0e-b29d-e65061972fcf` returned `NO BLOCKERS`; focused reviewer `de87e7a2-ae7e-4120-9b54-0f6ad5232a9d` also returned `NO BLOCKERS` for wrapper coverage, dependency boundary, load order, static/package coverage, and non-literal parser execution.
- Functional commit created: `b7216e2 extract frontend markdown helper`.

## 2026-06-15 11:56
- Implemented the next bounded frontend refactor tranche by extracting launch/backend/default/provider/model-memory helpers from `codoxear/static/app.js` into `codoxear/static/app_launch.js`.
- `index.html` now loads `app_launch.js` after URL/storage/perf/markdown helpers and before `app.js`; `app.js` keeps wrapper names and passes mutable `newSessionDefaults` explicitly into module functions that need defaults state.
- Added `app_launch.js` to `FRONTEND_ASSET_FILES`, top-level static routes, asset versioning, and wheel/static source tests.
- Updated provider/default/reasoning/Claude backend/source tests to inspect `app_launch.js` for moved semantics and `app.js` for wrappers/call sites; added a real-order VM test that executes `app_url.js`, `app_storage.js`, and `app_launch.js` together.
- First clean-room critic `cf0565f1-c700-4f51-8dc3-d09808ca58b1` returned `BLOCKERS`: `app_launch.js` initially required fake `CodoxearStorage.storageGetItem/storageSetItem/storageRemoveItem` names while the real storage module exports `getItem/setItem/removeItem`. The blocker was fixed by using the real storage API and adding the real-order module-load regression.
- Local focused validation after the fix: `node --check codoxear/static/app.js`, `node --check codoxear/static/app_launch.js`, real-order VM load of URL/storage/launch modules, and `python3 -m pytest -q tests/test_launch_ui_source.py tests/test_new_session_model_options_source.py tests/test_static_assets.py tests/test_claude_backend_source.py tests/test_reasoning_effort_source.py` passed: 44 tests.
- Docker focused validation: `scripts/codoxear-docker-sandbox test tests/test_launch_ui_source.py tests/test_new_session_model_options_source.py tests/test_static_assets.py tests/test_claude_backend_source.py tests/test_reasoning_effort_source.py tests/test_frontend_url_module_source.py tests/test_storage_robustness_source.py` passed: 50 tests and 3 subtests.
- Docker prefixed-route validation under `CODEX_WEB_URL_PREFIX=/codoxear`: `/codoxear/api/me -> 401`; `/codoxear/app_launch.js?v=test -> 200`; `/codoxear/app.js?v=test -> 200`.
- Full Docker validation: `scripts/codoxear-docker-sandbox test` passed: 964 tests, 1 skipped, 107 subtests.
- Focused reviewer `f3542620-2f37-4c18-9f1b-7de723caabf9` returned `NO BLOCKERS` after the storage-contract repair for the launch-helper extraction scope.
- Functional commit created: `320e646 extract frontend launch helper`.


## 2026-06-15T12:15:16 Display helper extraction checkpoint
- Functional commit created: `c5580fe extract frontend display helper`.
- Extracted pure display helpers into `codoxear/static/app_display.js`: `defaultButtonTooltip`, `fmtTs`, `fmtBytes`, `baseName`, `shortSessionId`, `sessionDisplayName`, `fmtIdleAge`, `fmtRelativeAge`, `sessionTitleWithId`, and `iconSvg`.
- `app.js` now requires `window.CodoxearDisplay` fail-loudly with `Codoxear display helpers failed to load` and keeps wrapper names for existing call sites.
- Static wiring: `index.html` loads `app_display.js` after `app_launch.js` and before `app.js`; `codoxear/server.py` includes it in `FRONTEND_ASSET_FILES` for asset versioning and top-level static routing.
- Guard tests added: display-helper formatter/age/byte/session-label cases, literal plus dynamic icon coverage for current `app.js` uses, and a static test that every versioned `index.html` script/link points to an existing registered frontend asset.
- Local diagnostic validation: `node --check codoxear/static/app_display.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_display_module_source.py tests/test_button_tooltips_source.py tests/test_static_assets.py tests/test_send_button_source.py tests/test_file_viewer_source.py tests/test_launch_ui_source.py` -> 51 passed.
- Docker focused acceptance on final staged tree: `CODOXEAR_DOCKER_PORT=18836 scripts/codoxear-docker-sandbox test tests/test_frontend_display_module_source.py tests/test_button_tooltips_source.py tests/test_static_assets.py tests/test_send_button_source.py tests/test_file_viewer_source.py tests/test_launch_ui_source.py -q` -> 51 passed.
- Docker URL-prefix static smoke under `CODEX_WEB_URL_PREFIX=/codoxear` on port 18832: `/codoxear/api/me -> 401`, `/codoxear/app_display.js?v=test -> 200`, `/codoxear/app.js?v=test -> 200`, `/codoxear/ -> 200`, and index contained `app_display.js?v=...`.
- Full Docker acceptance on final staged tree: `CODOXEAR_DOCKER_PORT=18837 scripts/codoxear-docker-sandbox test` -> 969 passed, 1 skipped, 107 subtests passed.
- Clean-room critic `a1a8273c-19a5-45d5-a395-d32c0a301ac9` returned `NO BLOCKERS`; its suggestions for icon coverage and versioned static-file existence were applied before the functional commit. Deployment-skew/no-fallback behavior is intentionally fail-loud, consistent with the helper-boundary policy.
- Repaired a malformed intermediate display-helper ledger entry caused by an unquoted shell heredoc; no product/runtime files were affected.


## 2026-06-15T12:30:24 API helper extraction checkpoint
- Functional commit created: `7e5e43d extract frontend API helper`.
- Extracted request mechanics from `codoxear/static/app.js` into `codoxear/static/app_api.js`: `api()`, private sessions ETag cache, private `API_NOT_MODIFIED` marker, `apiResponseNotModified()`, JSON parse/error handling, API perf samples, and `clearApiCache()`.
- `app.js` now requires `window.CodoxearApi` fail-loudly with `Codoxear API helpers failed to load`, keeps wrapper names for existing call sites, and calls `clearApiCache()` during app cleanup instead of reaching into private ETag state.
- Static wiring: `index.html` loads `app_api.js` after URL/perf helpers and before `app.js`; `codoxear/server.py` includes it in `FRONTEND_ASSET_FILES` for asset versioning and top-level static routing.
- Guard tests added: real-order VM load of `app_url.js`, `app_perf.js`, and `app_api.js`; sessions ETag reuse and 304 marker; `clearApiCache()` suppressing the next `If-None-Match`; URL-prefix resolution; all three API perf sample names; non-OK JSON errors; invalid-JSON logging/throwing; and app cleanup source using `clearApiCache()`.
- Local diagnostic validation: `node --check codoxear/static/app_api.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_api_module_source.py tests/test_session_polling_source.py tests/test_auth_cleanup_source.py tests/test_static_assets.py tests/test_frontend_url_module_source.py tests/test_frontend_perf_module_source.py` -> 40 passed, 3 subtests passed.
- Docker focused acceptance on final test tree: `CODOXEAR_DOCKER_PORT=18844 scripts/codoxear-docker-sandbox test tests/test_frontend_api_module_source.py tests/test_session_polling_source.py tests/test_auth_cleanup_source.py tests/test_static_assets.py tests/test_frontend_url_module_source.py tests/test_frontend_perf_module_source.py` -> 40 passed, 3 subtests passed.
- Docker URL-prefix static smoke under `CODEX_WEB_URL_PREFIX=/codoxear` on port 18839: `/codoxear/api/me -> 401`, `/codoxear/app_api.js?v=test -> 200`, `/codoxear/app.js?v=test -> 200`, `/codoxear/ -> 200`, and `app_api.js` contained `window.CodoxearApi`.
- Full Docker acceptance on final test tree: `CODOXEAR_DOCKER_PORT=18845 scripts/codoxear-docker-sandbox test` -> 973 passed, 1 skipped, 107 subtests passed.
- Clean-room critic `8080791f-65d9-482f-8a84-4abb2388d6b5` returned `NO BLOCKERS`; its suggestions for `api_messages_init_ms` and error-contract VM coverage were applied before the functional commit.


## 2026-06-15T12:46:18 File helper extraction checkpoint
- Functional commit created: `81d356b extract frontend file helpers`.
- Extracted file/viewer helper logic from `codoxear/static/app.js` into `codoxear/static/app_file_helpers.js`: `listFromFilesField`, `stripPathLocationSuffix`, `isTextFileKind`, `isDiffableFileKind`, `blockedFileMessage`, and `formatPriorityOffset`.
- `app.js` now requires `window.CodoxearFileHelpers` fail-loudly with `Codoxear file helpers failed to load` and keeps wrapper names for existing file-picker/file-viewer call sites.
- Static wiring: `index.html` loads `app_file_helpers.js` after `app_display.js` and before `app.js`; `codoxear/server.py` includes it in `FRONTEND_ASSET_FILES` for asset versioning and top-level static routing.
- Guard tests added: real-order VM load of display + file helpers, literal trailing-space/newline path preservation, no-suffix path preservation, inherited greedy `:line:column` suffix behavior, text/diff kind decisions, blocked-file messages including zero viewer limit, priority formatting, static route/version/package inclusion, and updated file-picker VM harnesses that load the real helper modules.
- Local diagnostic validation: `node --check codoxear/static/app_file_helpers.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_file_helpers_source.py tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_file_picker_session_state.py tests/test_static_assets.py` -> 65 passed.
- Docker focused acceptance on final test tree: `CODOXEAR_DOCKER_PORT=18849 scripts/codoxear-docker-sandbox test tests/test_frontend_file_helpers_source.py tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_file_picker_session_state.py tests/test_static_assets.py -q` -> 65 passed.
- Docker URL-prefix static smoke under `CODEX_WEB_URL_PREFIX=/codoxear` on port 18847: `/codoxear/api/me -> 401`, `/codoxear/app_file_helpers.js?v=test -> 200`, `/codoxear/app.js?v=test -> 200`, `/codoxear/ -> 200`, and `app_file_helpers.js` contained `window.CodoxearFileHelpers`.
- Full Docker acceptance on final test tree: `CODOXEAR_DOCKER_PORT=18850 scripts/codoxear-docker-sandbox test` -> 976 passed, 1 skipped, 107 subtests passed.
- Clean-room critic `d3fbcd89-e7ce-4f46-87c4-39164d6a27f3` returned `NO BLOCKERS`; its non-blocking guard suggestions for newline no-suffix preservation and zero viewer-limit message behavior were applied before the functional commit.


## 2026-06-15T13:06:00 Session helper extraction checkpoint
- Functional commit created: `342aa52 extract frontend session helpers`.
- Extracted pure session/sidebar render-state helpers into `codoxear/static/app_session_helpers.js`: immutable `SESSION_SIDEBAR_GROUPS`, launch failed/pending/kind/icon predicates, review/group classification, sidebar entry/signature construction, selectability, and fast-session detection.
- `app.js` now requires `window.CodoxearSessionHelpers` fail-loudly with `Codoxear session helpers failed to load` and keeps wrapper names for existing call sites. DOM rendering (`renderSessionGroupHeader`) and redaction-aware `sessionLaunchLabel`/`redactedLaunchErrorText` stayed app-owned.
- Static wiring: `index.html` loads `app_session_helpers.js` after `app_file_helpers.js` and before `app.js`; `codoxear/server.py` includes it in `FRONTEND_ASSET_FILES` for asset versioning and top-level static routing.
- Guard tests added/updated: direct VM tests for grouping, failed/pending selectability, launch kind/icon, sidebar signature, fast-session detection, frozen export metadata, fail-loud wrapper checks, static version/package inclusion, load order, and a boundary assertion keeping launch redaction/label logic out of the helper.
- Local diagnostic validation: `node --check codoxear/static/app_session_helpers.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_session_helpers_source.py tests/test_sidebar_gtd_source.py tests/test_session_polling_source.py tests/test_voice_push_source.py tests/test_chat_scrollback_source.py tests/test_static_assets.py tests/test_launch_ui_source.py` -> 70 passed.
- Deterministic equivalence check against `HEAD:codoxear/static/app.js` pre-extraction bodies passed for representative launch, grouping, selectability, fast-session, sidebar entries, and signature cases.
- Docker focused acceptance on final tree: `CODOXEAR_DOCKER_PORT=18856 scripts/codoxear-docker-sandbox test tests/test_frontend_session_helpers_source.py tests/test_sidebar_gtd_source.py tests/test_session_polling_source.py tests/test_voice_push_source.py tests/test_chat_scrollback_source.py tests/test_static_assets.py tests/test_launch_ui_source.py -q` -> 70 passed.
- Docker URL-prefix static smoke under `CODEX_WEB_URL_PREFIX=/codoxear` on port 18858: `/codoxear/api/me -> 401`, `/codoxear/app_session_helpers.js?v=test -> 200`, `/codoxear/ -> 200`, and the served helper contained the immutable `SESSION_SIDEBAR_GROUPS` definition.
- Full Docker acceptance on final tree: `CODOXEAR_DOCKER_PORT=18857 scripts/codoxear-docker-sandbox test` -> 979 passed, 1 skipped, 107 subtests passed.
- Clean-room critic `a02bb3b9-2860-486d-a79d-0448cc5be833` returned `NO BLOCKERS`; its suggestions to guard the redaction/label boundary and freeze exported group metadata were applied before commit.


## 2026-06-26T09:45:00 Viewport helper extraction checkpoint
- Functional commit created: `a6fc32a extract frontend viewport helper`.
- Extracted read-only viewport/media-query helpers into `codoxear/static/app_viewport.js`: `isMobile`, `prefersReducedMotion`, `useDesktopSessionActions`, and `useTouchFileEditorControls`.
- `app.js` now requires `window.CodoxearViewport` fail-loudly with `Codoxear viewport helpers failed to load` and keeps wrapper names for existing call sites. It preserves the pre-extraction `isMobile()` absent-`matchMedia` contract (`undefined`/falsy) and boolean-false absent-`matchMedia` behavior for the other viewport helpers.
- Static wiring: `index.html` loads `app_viewport.js` after `app_session_helpers.js` and before `app.js`; `codoxear/server.py` includes it in `FRONTEND_ASSET_FILES` for asset versioning and top-level static routing.
- Guard tests added/updated: direct VM tests for exact media-query strings, absent-`matchMedia` behavior, touch-control OR semantics, desktop combined-query behavior, frozen export metadata, executable app startup guard for missing/partial helper object, static version/package inclusion, load order, and migrated file-viewer/sidebar touch-mode harnesses that execute the real helper module rather than app.js snippets.
- Local diagnostic validation: `node --check codoxear/static/app_viewport.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_viewport_module_source.py tests/test_file_viewer_source.py tests/test_sidebar_touch_mode.py tests/test_chat_navigation_source.py tests/test_overlay_accessibility_source.py tests/test_static_assets.py` -> 63 passed.
- Deterministic equivalence check against `HEAD:codoxear/static/app.js` pre-extraction helper bodies passed for representative media-query states, including no-`matchMedia` contexts.
- Docker focused acceptance on final tree: `CODOXEAR_DOCKER_PORT=18865 scripts/codoxear-docker-sandbox test tests/test_frontend_viewport_module_source.py tests/test_file_viewer_source.py tests/test_sidebar_touch_mode.py tests/test_chat_navigation_source.py tests/test_overlay_accessibility_source.py tests/test_static_assets.py -q` -> 63 passed.
- Docker URL-prefix static smoke under `CODEX_WEB_URL_PREFIX=/codoxear` on port 18866: `/codoxear/api/me -> 401`, `/codoxear/app_viewport.js?v=test -> 200`, `/codoxear/static/app_viewport.js?v=test -> 200`, `/codoxear/ -> 200`, and the served helper contained `window.CodoxearViewport` plus the preserved `isMobile()` expression.
- Full Docker acceptance on final tree: `CODOXEAR_DOCKER_PORT=18867 scripts/codoxear-docker-sandbox test` -> 985 passed, 1 skipped, 107 subtests passed.
- Two clean-room subagent review attempts (`c2585b6b-a9e0-4611-8b79-bda91ee07c6e`, `6e4c712c-b065-4eac-bac6-40499c56a6c4`) failed before findings due a subagent model/output-config error; no code changes came from them. A redundant quick critic (`3be21cb6-8458-4dbb-8d08-228e8de84dfa`) was interrupted after the full delegate review completed.
- Clean-room delegate review `da503187-4b55-415a-88fe-e41c33d4b3e6` returned `NO BLOCKERS` for exact media-query strings, absent-`matchMedia` behavior, touch OR semantics, desktop combined query, reduced-motion call-site behavior, fail-loud guard, wrapper/call-site names, static asset/version/package wiring, and real-module test coverage. Non-blocking notes: `isMobile()` intentionally remains `boolean | undefined`; touch controls now express the absent-`matchMedia` guard through `mediaQueryMatches()`; the startup guard test uses source-string sentinels.

## 2026-06-26T10:10:00 Polling helper extraction checkpoint
- Functional commit created: `6506f21 extract frontend polling helper`.
- Extracted stateless polling-delay policy into `codoxear/static/app_polling.js`: session/secondary visibility delays, browser-offline predicate, message poll error backoff, active message poll delay choice, and kick-delay normalization.
- `app.js` now requires `window.CodoxearPolling` fail-loudly with `Codoxear polling helpers failed to load` and keeps wrapper names for existing call sites. Timers, mutable counters (`messagePollErrorStreak`, `pollFastUntilMs`, `turnOpen`), auth/session/transcript state, online/offline event handling, and scheduling side effects remain app-owned.
- Static wiring: `index.html` loads `app_polling.js` after `app_viewport.js` and before `app.js`; `codoxear/server.py` includes it in `FRONTEND_ASSET_FILES` for asset versioning and top-level static routing.
- Guard tests added/updated: direct VM tests for exact delay constants and branches, frozen exported metadata, missing/partial helper fail-loud behavior, app-wrapper integration against the real module, viewport guard boundary update, static version/package/load-order coverage, and a reviewer-suggested positive requested kick-delay case.
- Local diagnostic validation after reviewer-suggested test addition: `node --check codoxear/static/app_polling.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_polling_module_source.py tests/test_frontend_viewport_module_source.py tests/test_session_polling_source.py tests/test_static_assets.py` -> 34 passed.
- Docker focused acceptance on final tree: `CODOXEAR_DOCKER_PORT=18873 scripts/codoxear-docker-sandbox test tests/test_frontend_polling_module_source.py tests/test_frontend_viewport_module_source.py tests/test_session_polling_source.py tests/test_static_assets.py -q` -> 34 passed.
- Docker URL-prefix static smoke before the final test-only coverage addition: `/codoxear/api/me -> 401`, `/codoxear/ -> 200`, `/codoxear/app_polling.js -> 200`, and `/codoxear/static/app_polling.js -> 200`.
- Full Docker acceptance on final tree: `CODOXEAR_DOCKER_PORT=18874 scripts/codoxear-docker-sandbox test` -> 989 passed, 1 skipped, 107 subtests passed.
- Clean-room delegate review `927f869e-57ea-4c7d-9b8e-b44397727336` returned `NO BLOCKERS`; it confirmed delay equivalence, mutable-state ownership, fail-loud guard placement, static/version/package wiring, and module encapsulation. Its positive requested kick-delay coverage suggestion was applied before commit. Non-blocking notes: the local `messagePollErrorDelayMs` wrapper is currently unused but harmless, `browserOffline()` is slightly more defensive for unreachable falsy navigator inputs, two `kickPoll(900)` literals are pre-existing maintenance risk, and `POLLING_INTERVALS` guard depth matches the current internal-helper pattern.

## 2026-06-26T10:24:00 Conversation-copy helper extraction checkpoint
- Functional commit created: `22f4465 extract frontend conversation copy helper`.
- Extracted pure conversation-copy formatting into `codoxear/static/app_conversation_copy.js`: role filtering, text coercion/trimming, timestamp labelling, section headers, and `---` separators.
- `app.js` now requires `window.CodoxearConversationCopy` fail-loudly with `Codoxear conversation-copy helpers failed to load` and keeps the `formatConversationForCopy()` wrapper for existing call sites. API export fetches, selected-session checks, clipboard writes, button disabled state, toast/error handling, and DOM event handlers remain app-owned.
- Static wiring: `index.html` loads `app_conversation_copy.js` after `app_polling.js` and before `app.js`; `codoxear/server.py` includes it in `FRONTEND_ASSET_FILES` for asset versioning and top-level static routing.
- Guard/equivalence tests added/updated: direct VM tests for missing/partial helper fail-loud behavior, empty/non-array/all-skippable inputs, role filtering, timestamp inclusion, trailing-whitespace trimming with leading whitespace preserved, `String(ev.text || "")` falsy-text behavior for `null`/`0`/`false`, frozen export metadata, static version/package/load-order coverage, and the polling guard boundary now ending at the conversation-copy guard.
- Local diagnostic validation on final tree: `node --check codoxear/static/app_conversation_copy.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_conversation_copy_source.py tests/test_frontend_polling_module_source.py tests/test_transcript_export.py tests/test_static_assets.py` -> 46 passed, 4 subtests passed.
- Docker focused acceptance on final tree: `CODOXEAR_DOCKER_PORT=18880 scripts/codoxear-docker-sandbox test tests/test_frontend_conversation_copy_source.py tests/test_frontend_polling_module_source.py tests/test_transcript_export.py tests/test_static_assets.py -q` -> passed.
- Docker URL-prefix static smoke under `CODEX_WEB_URL_PREFIX=/codoxear` on port 18876: `/codoxear/api/me -> 401`, `/codoxear/ -> 200`, `/codoxear/app_conversation_copy.js?v=test -> 200`, and `/codoxear/static/app_conversation_copy.js?v=test -> 200`.
- Full Docker acceptance on final tree: `CODOXEAR_DOCKER_PORT=18881 scripts/codoxear-docker-sandbox test` -> 994 passed, 1 skipped, 107 subtests passed.
- Deterministic equivalence check against `HEAD:codoxear/static/app.js` pre-extraction formatter body passed for representative non-array, role-filtering, timestamp, falsy-text, leading/trailing whitespace, and separator cases.
- Clean-room delegate review `4a1b533a-5dee-4b1a-8a82-c7556f7b39b9` returned `NO BLOCKERS`; it confirmed side-effect ownership stayed in `app.js`, missing helper fails loudly, formatter logic is byte-identical in behavior, load order/static asset wiring is complete, and tests cover the boundary. Three earlier review attempts failed before findings due `output_config.effort` model-configuration errors; no code changes came from them.
- Negative/diagnostic evidence: the clean-room reviewer saw one host local full-suite run fail in `test_spawn_web_session_can_start_in_tmux`, then the single test and two full local reruns passed. This was treated as local flaky/order-dependent evidence, not Docker acceptance evidence and not caused by the conversation-copy diff.

## 2026-06-26T10:33:00 Video-preview error formatter checkpoint
- Functional commit created: `6caa995 extract video preview error formatter`.
- Extracted pure video-preview error text formatting into the existing `codoxear/static/app_file_helpers.js` helper boundary: `fileVideoPreviewErrorText(err)` preserves message/string trimming and fallback text.
- `app.js` now requires `window.CodoxearFileHelpers.fileVideoPreviewErrorText` fail-loudly through the existing `Codoxear file helpers failed to load` guard and keeps the local `fileVideoPreviewErrorText()` wrapper for the existing file-viewer call site.
- Side-effect boundary: `prepareCompatibleVideoPreview()`, auth-loss handling, `activeVideoFallback` state, `fileStatus` updates, `fileVideo` mutation/loading, and file-viewer DOM/button behavior remain app-owned in `app.js`.
- No static asset registration changed because `app_file_helpers.js` was already loaded before `app.js`, included in `FRONTEND_ASSET_FILES`, top-level static routing, asset versioning, and wheel packaging.
- Guard/equivalence tests updated: direct real-order VM tests cover `Error` message trimming, string newline trimming, blank-message fallback, and null fallback; `app.js` source asserts the fail-loud requirement and wrapper delegation; the video-preview failure-path VM now loads the real `app_file_helpers.js` module before executing the app snippet.
- Local diagnostic validation: `node --check codoxear/static/app_file_helpers.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_file_helpers_source.py tests/test_file_viewer_source.py tests/test_static_assets.py` -> 40 passed.
- Deterministic equivalence check against `HEAD:codoxear/static/app.js` pre-extraction formatter body passed for 9 representative cases, including `Error`, object `message`, string, blank, null, undefined, `0`, and `false` inputs.
- Helper body side-effect probe found no fetch, document, file status, video element, active fallback, auth-loss, or URL resolution references inside the extracted formatter body.
- Docker focused acceptance: `CODOXEAR_DOCKER_PORT=18882 scripts/codoxear-docker-sandbox test tests/test_frontend_file_helpers_source.py tests/test_file_viewer_source.py tests/test_static_assets.py -q` -> 40 passed.
- Full Docker acceptance: `CODOXEAR_DOCKER_PORT=18883 scripts/codoxear-docker-sandbox test` -> 994 passed, 1 skipped, 107 subtests passed.
- Clean-room delegate review `729062d7-003f-4282-8d8e-96dbe8ba2eac` returned `NO BLOCKERS`; it confirmed semantic equivalence, fail-loud guard, wrapper/call-site preservation, side-effect ownership in `app.js`, real-helper VM coverage, and no static asset wiring requirement. Its only note: `new Error("")` still formats as `"Error"`, which is pre-existing behavior, not a regression.


## 2026-06-26T10:50:00 Recovery prompt preview helper extraction checkpoint
- Functional commit created: `57d70db extract recovery prompt preview helper`.
- Extracted pure recovery prompt preview formatting into the existing `codoxear/static/app_display.js` helper boundary: `recoveryPromptPreview(text, maxLen = 320)` preserves `String(text || "").replace(/\s+/g, " ").trim()`, empty-string return, and `raw.slice(0, maxLen) + "…"` truncation behavior.
- `app.js` now requires `window.CodoxearDisplay.recoveryPromptPreview` fail-loudly through the existing `Codoxear display helpers failed to load` guard and keeps the local `recoveryPromptPreview()` wrapper for existing recovery call sites.
- Side-effect/security boundary: `redactedLaunchErrorText()`, `recoverySessionInfo()`, `recoveryDetailsText()`, recovery panel DOM/actions, copy-to-clipboard, launch preset/session state, and API mutations remain app-owned in `app.js`.
- No static asset registration changed because `app_display.js` was already loaded before `app.js`, included in `FRONTEND_ASSET_FILES`, top-level static routing, asset versioning, and wheel packaging.
- Guard/equivalence tests updated: direct real-module VM tests cover whitespace collapse, truncation, exact max length, falsy text, default max length, frozen export metadata, `app.js` fail-loud/source wrapper coverage, and the launch-recovery VM now loads the real `app_display.js` module before executing the app wrapper snippet.
- Local diagnostic validation: `node --check codoxear/static/app_display.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_display_module_source.py tests/test_chat_scrollback_source.py tests/test_static_assets.py` -> `42 passed`.
- Deterministic equivalence check against the pre-extraction `codoxear/static/app.js` formatter body passed for 10 representative cases, including whitespace-only, multiline whitespace, truncation, exact length, zero max length, `0`, `false`, and default-limit inputs.
- Helper body side-effect probe found no redaction, session, DOM, API, fetch, clipboard, toast, or window references inside the extracted formatter body.
- Docker focused acceptance: `CODOXEAR_DOCKER_PORT=18884 scripts/codoxear-docker-sandbox test tests/test_frontend_display_module_source.py tests/test_chat_scrollback_source.py tests/test_static_assets.py -q` -> `42 passed`.
- Full Docker acceptance: `CODOXEAR_DOCKER_PORT=18885 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Clean-room delegate review `ba7cbeaf-14d0-49d6-a5c1-3f412201d830` saved to `/tmp/codoxear-recovery-preview-helper-review.md` returned `NO BLOCKERS`; it confirmed semantic equivalence, fail-loud guard behavior, wrapper/call-site preservation, side-effect ownership in `app.js`, real-helper VM coverage, and no static asset wiring requirement.


## 2026-06-26T11:00:00 Recent-cwd score helper extraction checkpoint
- Functional commit created: `596ff7d extract recent cwd score helper`.
- Extracted pure recent-cwd fuzzy scoring into the existing `codoxear/static/app_display.js` helper boundary: `fuzzyRecentCwdScore(candidate, query)` preserves string coercion, trimmed/lower query handling, exact-path score `10000`, basename-exact score `9000`, exact-token boundary/base bonuses, subsequence scoring, and `-1` no-match behavior.
- `app.js` now requires `window.CodoxearDisplay.fuzzyRecentCwdScore` fail-loudly through the existing `Codoxear display helpers failed to load` guard and keeps the local `fuzzyRecentCwdScore()` wrapper for `filteredRecentCwdOptions()`.
- Side-effect/state boundary: `recentCwds`, `renderRecentCwdOptions()`, `filteredRecentCwdOptions()`, `newSessionCwdInput`, menu DOM/rendering/focus/selection, cwd validation, and new-session dialog actions remain app-owned in `app.js`.
- No static asset registration changed because `app_display.js` was already loaded before `app.js`, included in `FRONTEND_ASSET_FILES`, top-level static routing, asset versioning, and wheel packaging.
- Guard/equivalence tests updated: direct real-module VM tests cover no-query, exact full path, basename exact, boundary-token, multi-token, subsequence, no-match, frozen export metadata, `app.js` fail-loud/source wrapper coverage, and a precise source assertion that the nested recent-cwd scorer body left the `renderRecentCwdOptions()`/`filteredRecentCwdOptions()` region.
- Local diagnostic validation: `node --check codoxear/static/app_display.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_display_module_source.py tests/test_static_assets.py` -> `16 passed`.
- Deterministic equivalence check against the pre-extraction `codoxear/static/app.js` scorer body passed for 11 representative cases, including whitespace query, case-insensitive full path, basename exact, token boundary/base bonus, multi-token, subsequence, no-match, null candidate, and normalized project-name variants.
- Helper body side-effect probe found no `recentCwds`, `newSessionCwdInput`, render/filter function, DOM, API, fetch, toast, focus, `classList`, or `window` references inside the extracted scorer body.
- Docker focused acceptance: `CODOXEAR_DOCKER_PORT=18886 scripts/codoxear-docker-sandbox test tests/test_frontend_display_module_source.py tests/test_static_assets.py -q` -> `16 passed`.
- Full Docker acceptance: `CODOXEAR_DOCKER_PORT=18887 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Clean-room delegate review `79326319-1d34-4be7-b352-12aa500d5d11` saved to `/tmp/codoxear-cwd-score-helper-review.md` returned `NO BLOCKERS`; it confirmed semantic equivalence, fail-loud guard behavior, wrapper/call-site preservation, app-owned recent-cwd UI/state boundary, real-helper VM coverage, and no static asset wiring requirement.


## 2026-06-26T11:21:46Z File-picker matching/scoring helper extraction checkpoint
- Functional commit created: `4288221 extract file picker helpers`.
- Extracted pure file-picker matching/scoring helpers into the existing `codoxear/static/app_file_helpers.js` helper boundary: `fileSearchScore`, `normalizeDraftFilePath`, `filePickerFoldedSearchText`, `filePickerOriginalRangeForFolded`, `filePickerMatchRanges`, `filePickerMatchRangesForQuery`, `filePickerCandidateScore`, and `compareFilePickerEntries`.
- `app.js` now requires those `window.CodoxearFileHelpers` exports fail-loudly through the existing `Codoxear file helpers failed to load` guard and keeps thin local wrappers for existing picker/file-viewer call sites.
- Side-effect/state boundary: file-search state, candidate maps, API calls, DOM highlighting (`appendHighlightedFileMenuPath`), picker rendering, file-open actions, validation caches, and file-viewer state remain app-owned in `app.js`.
- No static asset registration changed because `app_file_helpers.js` was already loaded after `app_display.js` and before `app.js`, included in `FRONTEND_ASSET_FILES`, top-level static routing, asset versioning, and wheel packaging. The helper now explicitly requires `CodoxearDisplay.baseName` instead of reimplementing a basename fallback.
- Guard/equivalence tests updated: real helper module VM tests cover exact/basename/token/subsequence/no-match scoring, draft path normalization, Unicode folded match ranges including Turkish `İ` and emoji, normalized candidate scoring, comparator tie-breaks, fail-loud wrapper/source boundaries, and real-helper loading in file-picker/file-viewer VM harnesses.
- Local diagnostic validation before commit: `node --check codoxear/static/app_file_helpers.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_file_helpers_source.py tests/test_file_picker_search_source.py tests/test_file_viewer_source.py tests/test_static_assets.py` -> `63 passed`.
- Deterministic equivalence check against pre-extraction `app.js` bodies passed for `fileSearchScore`, `normalizeDraftFilePath`, folded/range helpers, `filePickerMatchRanges`, `filePickerMatchRangesForQuery`, `filePickerCandidateScore`, and `compareFilePickerEntries`; helper side-effect probe found no file state/DOM/API references.
- Docker focused acceptance before commit: `CODOXEAR_DOCKER_PORT=18890 scripts/codoxear-docker-sandbox test tests/test_frontend_file_helpers_source.py tests/test_file_picker_search_source.py tests/test_file_viewer_source.py tests/test_static_assets.py -q` -> `63 passed`.
- Full Docker acceptance before commit: `CODOXEAR_DOCKER_PORT=18891 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Clean-room delegate review `d52041b0-0f7b-4cb7-986b-99ffccd2c32d` saved to `/tmp/codoxear-file-picker-helper-review.md` returned `NO BLOCKERS`; it confirmed scoring/path/range/comparator equivalence, fail-loud guard coverage, wrapper preservation, `CodoxearDisplay.baseName` dependency, app-owned file state/API/DOM boundaries, real-helper test loading, and no static wiring requirement.


## 2026-06-26T11:33:35Z Chat-search display helper extraction checkpoint
- Functional commit created: `c09ace2 extract chat search display helpers`.
- Extracted pure chat-search display formatting helpers into `codoxear/static/app_display.js`: `compactChatSearchSnippet(text, query, limit = 96)` and `chatSearchTranscriptHint(match, query)`.
- `app.js` now requires `window.CodoxearDisplay.compactChatSearchSnippet` and `chatSearchTranscriptHint` fail-loudly through the existing `Codoxear display helpers failed to load` guard and keeps thin wrappers for existing chat-search call sites.
- Side-effect/state boundary: `rowSearchText()`, rendered-row matching, search timers, transcript-search API calls, loaded/all count state, DOM status/title/style updates, focus/navigation, and load-older actions remain app-owned in `app.js`.
- No static asset registration changed because `app_display.js` was already loaded before `app.js`, included in `FRONTEND_ASSET_FILES`, top-level static routing, asset versioning, and wheel packaging.
- Guard/equivalence tests updated: real display-module VM tests cover whitespace collapse, empty/short input, min-limit clamping, default/falsy limit behavior, query-centered snippets, no-match truncation, role labels, blank hints, and frozen export metadata; source tests verify the fail-loud guard/wrappers and app-owned `rowSearchText()`/chat-search boundary. The Node test helper environment now passes only `PATH` and `TZ` to avoid leaking ambient secrets in failure diagnostics.
- Local diagnostic validation on final tree: `node --check codoxear/static/app_display.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_display_module_source.py tests/test_chat_navigation_source.py tests/test_static_assets.py` -> `27 passed`.
- Deterministic equivalence check against pre-extraction `app.js` bodies passed for representative snippet/hint cases; helper side-effect probe found no chat DOM/state/API references in the moved display body.
- Docker focused acceptance after the final test-harness hardening: `CODOXEAR_DOCKER_PORT=18894 scripts/codoxear-docker-sandbox test tests/test_frontend_display_module_source.py tests/test_chat_navigation_source.py tests/test_static_assets.py -q` -> `27 passed`.
- Full Docker acceptance after the final test-harness hardening: `CODOXEAR_DOCKER_PORT=18895 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- First clean-room delegate run `edc2af3c-a1c9-4e0b-b4be-f0fb8ef9f29b` failed before findings due `output_config.effort` model configuration, so it did not count as a substantive review. Replacement clean-room delegate review `48a1112c-dedd-475b-be3c-bfb882fcb56f` saved to `/tmp/codoxear-chat-search-display-review.md` returned `NO BLOCKERS`; it confirmed snippet/hint semantics, fail-loud guard coverage, wrapper/call-site preservation, app-owned chat-search DOM/state/API/load-older behavior, real-module test execution, and no static wiring requirement.


## 2026-06-26T11:46:12Z File-picker source/label helper extraction checkpoint
- Functional commit created: `c13dd1e extract file picker source label helpers`.
- Extracted two pure file-picker mapping helpers into `codoxear/static/app_file_helpers.js`: `normalizeFileCandidateSource(source)` and `filePickerSectionLabel(source)`.
- `app.js` now requires both helpers fail-loudly through the existing `Codoxear file helpers failed to load` guard and keeps local wrappers for existing call sites.
- Side-effect/state boundary: file identity keys, candidate clone/merge semantics, candidate maps/cache, changed-file API calls, DOM section insertion (`appendFilePickerSection`), picker rendering/highlighting, active file/open behavior, file-viewer state, focus/timers/recovery/security behavior remain app-owned in `app.js`.
- No static asset registration changed because `app_file_helpers.js` was already loaded after `app_display.js` and before `app.js`, included in `FRONTEND_ASSET_FILES`, top-level static routing, asset versioning, and wheel packaging.
- Guard/equivalence tests updated: real helper module VM tests cover source whitespace trimming, accepted `changed`/`mentioned`/`recent` values, unknown/blank source fallback, section labels, frozen export metadata, app fail-loud/source-wrapper coverage, helper-body placement, and absence of the old mapping bodies from `app.js`. Existing file-picker VM harnesses were updated to include the new wrapper dependency, and the file-viewer source sentinel was changed from the removed inline normalizer to the next app-owned candidate-cloning function.
- Local diagnostic validation: `node --check codoxear/static/app_file_helpers.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_file_helpers_source.py tests/test_file_picker_search_source.py tests/test_file_viewer_source.py tests/test_static_assets.py` -> `63 passed`.
- Deterministic equivalence check against the pre-extraction `app.js` inline bodies passed for 13 source-normalization cases and 10 section-label cases; helper side-effect probe found no DOM/API/file-state/timer/focus/storage references.
- Docker focused acceptance: `CODOXEAR_DOCKER_PORT=18896 scripts/codoxear-docker-sandbox test tests/test_frontend_file_helpers_source.py tests/test_file_picker_search_source.py tests/test_file_viewer_source.py tests/test_static_assets.py -q` -> `63 passed`.
- Full Docker acceptance: `CODOXEAR_DOCKER_PORT=18897 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Clean-room delegate review `26c0dc6c-8ce0-4714-9c6c-e17422448ada` saved to `/tmp/codoxear-file-source-label-review.md` returned `NO BLOCKERS`; it confirmed pure helper ownership, fail-loud guard coverage, wrapper/call-site preservation, app-owned file identity/cache/API/DOM/viewer/focus/timer/recovery/security boundaries, static wiring sufficiency, and test coverage.


## 2026-06-26T11:54:57Z File-editor cursor helper extraction checkpoint
- Functional commit created: `afe054f extract file editor cursor helper`.
- Extracted pure file-editor cursor arithmetic into `codoxear/static/app_file_helpers.js`: `positionAfterInsertedText(start, text)` preserves `String(text || "")`, unchanged empty/falsy return, CRLF/CR-to-LF normalization, single-line column advance, multi-line line/column calculation, and trailing-newline column 1 behavior.
- `app.js` now requires `window.CodoxearFileHelpers.positionAfterInsertedText` fail-loudly through the existing `Codoxear file helpers failed to load` guard and keeps a local wrapper for the existing paste call site.
- Side-effect/state boundary: Monaco/editor access, `executeEdits`, paste handling, file dirty state, touch-selection reset, `applyFileEditorSelection`, focus, DOM, file-viewer availability guards, save/edit behavior, timers, APIs, and recovery/security behavior remain app-owned in `app.js`.
- No static asset registration changed because `app_file_helpers.js` was already loaded after `app_display.js` and before `app.js`, included in `FRONTEND_ASSET_FILES`, top-level static routing, asset versioning, and wheel packaging.
- Guard/equivalence tests updated: real helper module VM tests cover empty string, null, single-line text, LF, CRLF, CR-only, and trailing-newline insertions; source tests verify the fail-loud guard/wrapper, app-owned call site, helper-body placement, and absence of the arithmetic body from `app.js`.
- Local diagnostic validation: `node --check codoxear/static/app_file_helpers.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_file_helpers_source.py tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_static_assets.py` -> `63 passed`.
- Deterministic equivalence check against the pre-extraction `app.js` inline body passed for 33 cases across three start positions and empty/null, single-line, LF, CRLF, CR-only, trailing-newline, leading-newline, `0`, and `false` text inputs; helper side-effect probe found no DOM/API/editor/file-state/timer/focus/storage references.
- Docker focused acceptance: `CODOXEAR_DOCKER_PORT=18898 scripts/codoxear-docker-sandbox test tests/test_frontend_file_helpers_source.py tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_static_assets.py -q` -> `63 passed`.
- Full Docker acceptance: `CODOXEAR_DOCKER_PORT=18899 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Clean-room delegate review `4d727bae-5f06-462e-ba1e-986659d18308` saved to `/tmp/codoxear-cursor-helper-review.md` returned `NO BLOCKERS`; it confirmed semantic equivalence, fail-loud guard coverage, wrapper scope/call-site preservation, app-owned editor/paste/dirty/focus behavior, static wiring sufficiency, and branch-covering tests.


## 2026-06-26T12:02:17Z File-editor delete-key helper extraction checkpoint
- Functional commit created: `f761f05 extract file editor delete key helper`.
- Extracted pure file-editor delete-key command mapping into `codoxear/static/app_file_helpers.js`: `fileEditorDeleteCommandForKey(key)` preserves lowercase `backspace -> deleteLeft`, lowercase `delete -> deleteRight`, and empty-string fallback for all other inputs.
- `app.js` now requires `window.CodoxearFileHelpers.fileEditorDeleteCommandForKey` fail-loudly through the existing `Codoxear file helpers failed to load` guard and keeps a local wrapper for the existing delete-key handler call site.
- Side-effect/state boundary: event modifier filtering, key lowercasing, native delete suppression, Monaco/editor access, `editor.trigger`, touch-selection reset, focus, toast/error behavior, DOM, file-viewer availability guards, save/edit behavior, timers, APIs, and recovery/security behavior remain app-owned in `app.js`.
- No static asset registration changed because `app_file_helpers.js` was already loaded after `app_display.js` and before `app.js`, included in `FRONTEND_ASSET_FILES`, top-level static routing, asset versioning, and wheel packaging.
- Guard/equivalence tests updated: real helper module VM tests cover `backspace`, `delete`, uppercase `Backspace`, unknown key, and blank key; source tests verify the fail-loud guard/wrapper, helper-body placement, absence of the mapping body from `app.js`, and app-owned event/Monaco/native suppression behavior.
- Local diagnostic validation: `node --check codoxear/static/app_file_helpers.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_file_helpers_source.py tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_static_assets.py` -> `63 passed`.
- Deterministic equivalence check against the pre-extraction `app.js` inline body passed for 10 key inputs: `undefined`, `null`, blank, `backspace`, `delete`, uppercase variants, unknown, `0`, and `false`; helper side-effect probe found no DOM/API/editor/file-state/timer/focus/storage references.
- Docker focused acceptance: `CODOXEAR_DOCKER_PORT=18900 scripts/codoxear-docker-sandbox test tests/test_frontend_file_helpers_source.py tests/test_file_viewer_source.py tests/test_file_picker_search_source.py tests/test_static_assets.py -q` -> `63 passed`.
- Full Docker acceptance: `CODOXEAR_DOCKER_PORT=18901 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Clean-room delegate review `cc8e023e-2ba7-492f-a3b8-e3b1714948f7` saved to `/tmp/codoxear-delete-key-helper-review.md` returned `NO BLOCKERS`; it confirmed pure helper ownership, fail-loud guard coverage, wrapper/call-site preservation, app-owned event filtering/lowercasing/native suppression/Monaco trigger/touch reset/focus/toast behavior, static wiring sufficiency, and test coverage.


## 2026-06-26T12:11:38Z Model-option match helper extraction checkpoint
- Functional commit created: `fcdc01b extract model option match helper`.
- Extracted pure model-option search predicate into `codoxear/static/app_launch.js`: `modelOptionMatches(option, query)` preserves search-text/model fallback, lowercase comparison, empty-query match-all, exact/prefix/contains checks, and no-match false behavior.
- `app.js` now requires `window.CodoxearLaunch.modelOptionMatches` fail-loudly through the existing `Codoxear launch helpers failed to load` guard and keeps a local wrapper for the existing `filteredNewSessionModelOptions()` contains-tier call site.
- Side-effect/state boundary: model-option construction, exact/prefix/contains ordering, result slicing, provider/model selection, rendering, local/session state, memory persistence, focus/menu behavior, APIs, DOM, timers, recovery/security behavior, and launch-dialog state remain app-owned in `app.js`.
- No static asset registration changed because `app_launch.js` was already loaded before `app.js`, included in `FRONTEND_ASSET_FILES`, top-level static routing, asset versioning, and wheel packaging.
- Guard/equivalence tests updated: launch-helper VM tests cover empty query, exact, prefix, contains, fallback to `model`, and no-match cases; source tests verify the fail-loud guard/wrapper, helper-body placement, absence of the old body from `app.js`, and real `app_launch.js` loading in new-session model-option VM snippets.
- Local diagnostic validation: `node --check codoxear/static/app_launch.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_launch_ui_source.py tests/test_new_session_model_options_source.py tests/test_static_assets.py` -> `39 passed`.
- Deterministic equivalence check against the pre-extraction `app.js` inline body passed for 10 cases covering empty query, exact/prefix/contains, case-insensitive search text, fallback model field, blank search text with model fallback, null option, and no-match; helper side-effect probe found no DOM/API/session/dialog-state/timer/focus/storage references.
- Docker focused acceptance: `CODOXEAR_DOCKER_PORT=18902 scripts/codoxear-docker-sandbox test tests/test_launch_ui_source.py tests/test_new_session_model_options_source.py tests/test_static_assets.py -q` -> `39 passed`.
- Full Docker acceptance: `CODOXEAR_DOCKER_PORT=18903 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Clean-room delegate review `5992087b-7600-48de-a9ab-6c0a060861a7` saved to `/tmp/codoxear-model-option-match-review.md` returned `NO BLOCKERS`; it confirmed pure helper ownership, semantic equivalence, fail-loud guard coverage, wrapper/call-site preservation, app-owned filtering order/sessionModelOptions/rendering/provider-selection/memory/focus/menu behavior, static wiring sufficiency, and test coverage.
- Advisory scout `d9e4b7eb-e767-47e1-8152-588033474371` failed before findings due model `output_config.effort`; replacement read-only scout `3695ddb6-9f15-4943-a0a8-1dc48ca582f3` was launched with `anthropic/claude-sonnet-4` to assess possible next pure-helper targets.


## 2026-06-26T12:23:54Z Diagnostics session helper extraction checkpoint
- Functional commit created: `dd57f14 extract diagnostics session helpers`.
- Extracted pure diagnostics helpers into `codoxear/static/app_session_helpers.js`: `diagnosticsProviderDisplay(d, backend)` and `diagnosticsCopyText(sessionId, rows)`.
- `app.js` now requires both helpers fail-loudly through the existing `Codoxear session helpers failed to load` guard and keeps local wrappers for existing diagnostics call sites.
- Backend normalization boundary: `app.js` wrapper calls `codoxearSessionHelpers.diagnosticsProviderDisplay(d, sessionAgentBackend(d))`, so `codoxear/static/app_launch.js` remains the source of truth for backend aliases such as `claude` and `claude-code` normalizing to `cc`.
- Side-effect/state boundary: `showDiagViewer`, `hideDiagViewer`, diagnostics API fetch, `diagRows` construction, `diagCopyText` mutable state, copy-to-clipboard, diagnostics buttons/backdrop/DOM, focus restoration, auth loss handling, and diagnostics error recovery remain app-owned in `app.js`.
- No static asset registration changed because `app_session_helpers.js` was already loaded before `app.js`, included in `FRONTEND_ASSET_FILES`, top-level static routing, asset versioning, and wheel packaging.
- Guard/equivalence tests updated: diagnostics VM tests now load real `app_session_helpers.js`, cover Pi absent/actual provider, Codex provider choice, Codex model-provider fallback, Claude Code ignored provider via normalized backend, null diagnostics rows, label/value copy formatting, and inserted Session lines; session-helper tests cover the new exports and fail-loud guard/wrapper source boundaries.
- Local diagnostic validation: `node --check codoxear/static/app_session_helpers.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_diagnostics_source.py tests/test_frontend_session_helpers_source.py tests/test_static_assets.py` -> `19 passed`.
- Deterministic equivalence check against pre-extraction `app.js` inline bodies passed for 11 provider cases and 5 copy-text cases, including `claude`/`claude-code` alias normalization through the app wrapper; refined helper side-effect probe found no DOM/API/session/modal/timer/focus/storage references.
- Docker focused acceptance: `CODOXEAR_DOCKER_PORT=18904 scripts/codoxear-docker-sandbox test tests/test_diagnostics_source.py tests/test_frontend_session_helpers_source.py tests/test_static_assets.py -q` -> `19 passed`.
- Full Docker acceptance: `CODOXEAR_DOCKER_PORT=18905 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Clean-room delegate review `87b32795-9336-49a9-b2d0-170748b2a68e` saved to `/tmp/codoxear-diagnostics-helper-review.md` returned `NO BLOCKERS`; it confirmed semantic equivalence, alias normalization preservation, fail-loud guard coverage, app-owned diagnostics state/API/DOM/clipboard/auth-error boundaries, load-order sufficiency, export wiring, and test coverage.


## 2026-06-26T12:31:31Z Queue normalization helper extraction checkpoint
- Functional commit created: `35be96c extract queue normalization helper`.
- Extracted pure queue API-payload normalizer into `codoxear/static/app_session_helpers.js`: `normalizeQueueItems(data)`.
- `app.js` now requires `window.CodoxearSessionHelpers.normalizeQueueItems` fail-loudly through the existing `Codoxear session helpers failed to load` guard and keeps a local wrapper for the existing `refreshQueueViewer()` call site.
- Modern payload semantics preserved: `items` arrays win over legacy `queue`, non-object items are ignored, string `id`/`text` only are retained, `sending`/`commit_unknown`/`orphan_recovery` are booleanized to `sending`/`commitUnknown`/`orphanRecovery`, and missing-id or blank-text records are filtered without trimming output text.
- Legacy payload semantics preserved: `queue` string arrays filter non-string/blank text first, then assign post-filter `legacy-${idx}` IDs with `sending`, `commitUnknown`, and `orphanRecovery` false.
- Side-effect/state boundary: `refreshQueueViewer`, queue API fetch, auth-loss/error handling, `queueDraftTexts` merge/preservation, `queueViewerItems` assignment, queue empty text, `renderQueueList`, queue mutation locks, move barriers, delete/update/move/send/enqueue call sites, DOM, focus, timers, recovery/security behavior remain app-owned in `app.js`.
- No static asset registration changed because `app_session_helpers.js` was already loaded before `app.js`, included in `FRONTEND_ASSET_FILES`, top-level static routing, asset versioning, and wheel packaging.
- Guard/equivalence tests updated: real session-helper VM tests cover modern item normalization, filtering of missing id/blank text/non-object/bad-id rows, commit-unknown/recovery flag preservation, legacy queue normalization and post-filter `legacy-*` IDs, null input, frozen exports, and fail-loud app wrapper/source boundaries; queue source tests confirm app-owned queue UI/barrier behavior remains in `app.js` and snake_case payload processing moved to the helper.
- Local diagnostic validation: `node --check codoxear/static/app_session_helpers.js`, `node --check codoxear/static/app.js`, and `python3 -m pytest -q tests/test_frontend_session_helpers_source.py tests/test_queue_button_source.py tests/test_static_assets.py` -> `17 passed`.
- Deterministic equivalence check against pre-extraction `app.js` inline body passed for 9 cases covering null/undefined/empty input, invalid `items`, modern item filtering, flag booleanization, legacy string filtering, post-filter legacy IDs, `items`-over-`queue` priority, and empty items; helper side-effect probe found no DOM/API/queue state/timer/focus/storage references.
- Docker focused acceptance: `CODOXEAR_DOCKER_PORT=18906 scripts/codoxear-docker-sandbox test tests/test_frontend_session_helpers_source.py tests/test_queue_button_source.py tests/test_static_assets.py -q` -> `17 passed`.
- Full Docker acceptance: `CODOXEAR_DOCKER_PORT=18907 scripts/codoxear-docker-sandbox test` -> `994 passed, 1 skipped, 107 subtests passed`.
- Clean-room delegate review `890a40d4-1092-4540-8337-f74167c972de` saved to `/tmp/codoxear-queue-normalizer-review.md` returned `NO BLOCKERS`; it confirmed helper placement/export, fail-loud guard coverage, wrapper/call-site preservation, modern and legacy semantic equivalence, `items` priority, app-owned queue state/API/DOM/focus/timer/recovery/security boundaries, load-order sufficiency, syntax validity, test coverage, and minimality.
- Advisory future-target scout `83934278-8784-4af1-9179-580e7df91d74` was running at checkpoint time and is not a commit gate.


## 2026-06-26T12:32:18Z Post-queue pure-helper scout conclusion
- Advisory future-target scout `83934278-8784-4af1-9179-580e7df91d74` saved to `/tmp/codoxear-next-pure-helper-scout-after-queue.md` completed after the queue normalizer functional commit.
- Scout conclusion: no remaining safe pure-helper extraction candidates meet the current bar after `normalizeQueueItems(data)` moved.
- Parked non-candidates: `redactedLaunchErrorText`/`sessionLaunchLabel` are mechanically pure but intentionally pinned out of `app_session_helpers.js` by source tests and include security-sensitive redaction/label composition; `launchPresetProviderChoice` is mechanically pure but pinned in `app.js` by the new-session model-options source-slicing tests and launch-dialog orchestration context.
- Other apparent helpers are wrappers, state readers, unused/dead code, DOM-dependent, browser-side-effect code, or render/orchestration functions inside `renderApp()`.
- Decision: do not force another helper-extraction tranche under the current constraints. Further work requires broader design/ownership decisions or product-gap work rather than mechanical pure-helper extraction.


## 2026-06-26T12:37:32Z User scope correction: stop over-narrowing to bounded helper work
- User stated: "why is everything 'bounded'? i never ask you to do things boundedly. be thorough and aggressive!!!"
- Interpretation: the prior helper-extraction stop condition should not be treated as task completion. The pure-helper wave remains complete, but the active objective expands back to aggressive product-gap recovery, reliability hardening, and meaningful structural/frontend work.
- Preserved constraints: work only in `/home/yiwen/codex-web-product-recovery` unless explicitly approved; do not promote/merge to protected `/home/yiwen/codex-web` or `main`; do not kill live sessions/servers; no secrets/runtime artifacts; Docker evidence for acceptance claims; functional/docs commits separate; no silent fallbacks.
- Next decision: choose a high-value implementable gap rather than another mechanical helper extraction. Candidate classes include parked security/reliability gaps (for example symlink containment atomicity or non-UTF-8 Git filename behavior), executable UX regressions, or product-flow failures discoverable from recon/tests.

## 2026-06-26T14:25:47Z Server runtime/routes architecture checkpoint
- Observation: functional architecture commit created: `be7eeb3 extract session runtime and route controllers`.
- Implemented ownership changes: `session_runtime.py` is now the shared authority for broker busy/queue/interrupted-idle/send-boundary readiness and token fallback; `session_store.py` owns persistent session maps; message/file-write/launch-ledger/queue/control/diagnostics/git HTTP controller behavior moved out of `server.py` into explicit route/ledger modules.
- Behavior coverage added: `tests/test_session_runtime.py`, `tests/test_session_store.py`, `tests/test_file_routes.py`, `tests/test_queue_routes.py`, `tests/test_control_routes.py`, and `tests/test_diagnostics_routes.py`; existing source/behavior tests were adjusted to the new ownership boundaries.
- Focused diagnostic validation: `python3 -m pytest -q tests/test_file_inspect.py tests/test_git_ops.py tests/test_file_picker_search_source.py tests/test_session_resume.py` -> `130 passed, 52 subtests passed`; earlier route/controller focused validation after diagnostics extraction -> `54 passed, 4 subtests passed`.
- Acceptance validation: `CODOXEAR_DOCKER_PORT=18920 scripts/codoxear-docker-sandbox test` -> `1028 passed, 1 skipped, 107 subtests passed`.
- Clean-room review: async architect run `04337c97-39f1-4aff-9221-8bb90c030a3e`, output `/tmp/codoxear-architecture-runtime-routes-review.md`, returned `NO BLOCKERS` for runtime busy/ready/token semantics, broker queue short-circuiting before log parsing, `SessionManager.__new__` compatibility, persistence path rebinding, queue/control/diagnostics/git route error mappings, file write security, message cursor/HMAC behavior, and launch redaction/recovery.
- Scoped claim: the checkpoint materially reduces `server.py` ownership by moving runtime/persistence/controller mechanisms to named modules; it does not claim live backend lifecycle expansion, mobile/device evidence, or completion of the remaining inline file GET/download/blob/video, tail, or unattended route seams.


## 2026-06-26T15:25:00Z File and session route ownership checkpoint
- Functional commit `329cf8d Extract file get route controller`: moved session file GET/read/search/list/blob/video_preview/download and absolute preview policy into `codoxear/file_routes.py` via `FileGetRouteDeps`, `handle_file_get_route`, and `handle_absolute_file_preview_route`. Acceptance: `CODOXEAR_DOCKER_PORT=18922 scripts/codoxear-docker-sandbox test` -> `1031 passed, 1 skipped, 107 subtests passed`; clean-room review `/tmp/codoxear-file-get-routes-review.md` returned `NO BLOCKERS`.
- Functional commit `359a0c0 Extract file write route controller`: moved `/api/sessions/{id}/file/write` POST HTTP policy into `codoxear/file_routes.py` via `FileWriteRouteDeps` and `handle_file_write_post_route`; session/file mutation primitives remained injected. Acceptance: `CODOXEAR_DOCKER_PORT=18926 scripts/codoxear-docker-sandbox test` -> `1033 passed, 1 skipped, 107 subtests passed`; clean-room review `/tmp/codoxear-file-write-route-review.md` returned `NO BLOCKERS`.
- Functional commit `97b4247 Extract global file post routes`: moved global `/api/files/read` and `/api/files/inspect` POST route composition into `codoxear/file_routes.py` via `GlobalFileRouteDeps`, `GlobalFileRequest`, `handle_global_file_post_route`, and `global_file_read_payload`; direct tests cover non-string `session_id`, whitespace path preservation, read history recording, and git-relative media URLs. Acceptance: `CODOXEAR_DOCKER_PORT=18929 scripts/codoxear-docker-sandbox test` -> `1036 passed, 1 skipped, 107 subtests passed`; clean-room review `/tmp/codoxear-global-file-routes-review.md` returned `NO BLOCKERS`.
- Functional commit `0bd0991 Extract session route controller`: moved session list/defaults/resume-candidate/metrics/tail/unattended GET composition and web-owned session creation POST composition into new `codoxear/session_routes.py`; `SessionManager` remains runtime owner for listing, recent cwd state, aliases, tail, unattended config, and spawn behavior; `server.py` now injects parsing/auth/ETag/metrics/launch dependencies through `_session_route_deps()`.
- Session-route diagnostic validation: `python3 -m pytest -q tests/test_route_decomposition_source.py tests/test_session_routes.py tests/test_unattended_mode_source.py tests/test_session_route_matcher.py tests/test_launch_defaults.py tests/test_session_resume.py tests/test_launch_provenance.py tests/test_auth_cookie.py tests/test_session_polling_source.py tests/test_new_session_launch_request.py tests/test_new_session_model_options_source` -> `125 passed, 19 subtests passed`.
- Session-route Docker focused acceptance: `CODOXEAR_DOCKER_PORT=18930 scripts/codoxear-docker-sandbox test tests/test_session_routes.py tests/test_unattended_mode_source.py tests/test_session_route_matcher.py tests/test_launch_defaults.py tests/test_session_resume.py tests/test_launch_provenance.py tests/test_auth_cookie.py tests/test_session_polling_source.py tests/test_new_session_launch_request.py tests/test_new_session_model_options_source.py tests/test_route_decomposition_source.py -q` -> passed.
- Session-route full Docker acceptance: `CODOXEAR_DOCKER_PORT=18931 scripts/codoxear-docker-sandbox test` -> `1041 passed, 1 skipped, 107 subtests passed`.
- Session-route clean-room critic `bca60d88-c5eb-45c4-b73d-ce3625f66bcc`, saved to `/tmp/codoxear-session-routes-critic.md`, returned `NO BLOCKERS`. It verified non-overlap of moved tail/unattended GET order with diagnostics/queue/file/git/message route suffixes, auth polarity and single-response behavior, launch validation/ValueError/SessionLaunchError mapping, JSON-body delegation defaults, backend normalization call shape, and Unattended source-sentinel ownership. Earlier reviewer/architect attempts failed or were interrupted due model/baseline issues and were not counted as acceptance evidence.
- Current route ownership state: `server.py` no longer owns file route branches or session list/resume/metrics/tail/unattended/create HTTP composition. Remaining inline route seams visible in `Handler` are static, auth (`/api/me`, `/api/login`, `/api/logout`), voice/notification/audio, and hooks.


## 2026-06-26T16:05:00Z Voice and auth route ownership checkpoint
- Functional commit `1950ffd Extract voice route controller`: moved voice settings, push notification subscription/feed/message routes, audio playlist/segment streaming, and audio listener POST HTTP validation/status/header mapping into new `codoxear/voice_routes.py`; `VoicePushCoordinator` remains the state/audio/subscription authority, and `Handler._handle_voice_get/_post` remain thin wrappers to preserve existing monkeypatch seams.
- Voice-route diagnostic validation: local focused route/source groups -> `17 passed`; local broader voice/source groups -> `131 passed, 52 subtests`; focused Docker after the wrapper fix on port 18934 passed; full Docker on port 18935 -> `1046 passed, 1 skipped, 107 subtests`.
- Voice-route clean-room review `/tmp/codoxear-voice-routes-review.md` returned `NO BLOCKERS`; reviewer local full suite observed `1047 passed, 107 subtests`. Non-blocking notes were lack of direct unit tests for unknown notification-message 404 and notification-feed happy path, plus the preserved fail-loud `voice_push=None` behavior for matched voice routes.
- Functional commit `62dd4f2 Extract auth route controller`: moved `/api/me`, `/api/login`, and `/api/logout` HTTP response/cookie route composition into new `codoxear/auth_routes.py`; cookie signing/verification/HMAC secret authority remains in `auth.py` and server auth helpers, with JSON-body parsing injected from `Handler._read_json_body`.
- Auth-route diagnostic validation: `python3 -m pytest -q tests/test_auth_routes.py tests/test_auth_cookie.py tests/test_url_prefix.py tests/test_json_body_source.py tests/test_route_decomposition_source.py tests/test_message_route_source.py tests/test_transcript_export.py tests/test_voice_routes.py` -> `50 passed, 4 subtests`; source sentinel was updated to assert `obj = deps.read_json_body(handler)` in the new auth owner rather than an obsolete inline `do_POST` call.
- Auth-route Docker focused acceptance: `CODOXEAR_DOCKER_PORT=18936 scripts/codoxear-docker-sandbox test tests/test_auth_routes.py tests/test_auth_cookie.py tests/test_url_prefix.py tests/test_json_body_source.py tests/test_route_decomposition_source.py tests/test_message_route_source.py tests/test_transcript_export.py tests/test_voice_routes.py -q` -> passed. Full Docker acceptance: `CODOXEAR_DOCKER_PORT=18937 scripts/codoxear-docker-sandbox test` -> `1051 passed, 1 skipped, 107 subtests`.
- Auth-route clean-room review `/tmp/codoxear-auth-routes-review.md` returned `NO BLOCKERS`; it verified `/api/me` auth/JSON behavior, login 403 and success cookie/body/no-content-length behavior, logout auth short-circuit and clear-cookie header, route ordering before voice/session routes, JSON parser exception propagation, and unchanged cookie-signing authority.
- Current route ownership state: `server.py` no longer owns file, session, voice, or auth HTTP composition. Remaining inline route seams visible in `Handler` are static/index/asset serving, optional `/api/hooks/notify`, and the central dispatch/error/parser wrappers that tie route modules together.


## 2026-06-26T16:32:00Z Static and hook route ownership checkpoint
- Functional commit `bf4f017 Extract static route controller`: moved static/index/asset routing, path containment, content-type selection, CSP/X-Frame/cache headers, static asset versioning, and HTML placeholder replacement ownership into new `codoxear/static_routes.py`; `server.py` re-exports prior static names and keeps `Handler._handle_static_get` as a thin monkeypatch seam.
- Static-route diagnostic validation: `python3 -m pytest -q tests/test_static_routes.py tests/test_static_assets.py tests/test_url_prefix.py tests/test_route_decomposition_source.py tests/test_message_route_source.py tests/test_transcript_export.py tests/test_file_inspect.py` -> `117 passed, 56 subtests`. Additional `.env` diagnostic from a temp cwd with `CODEX_WEB_STATIC_CACHE=1` confirmed `server._static_cache_control_headers()` still returns immutable cache headers after `.env` load.
- Static-route Docker acceptance: focused Docker on port 18938 for the same affected groups passed; full Docker on port 18939 -> `1055 passed, 1 skipped, 107 subtests`. Clean-room review `/tmp/codoxear-static-routes-review.md` returned `NO BLOCKERS`; it verified top-level route registry identity, `/static/*` behavior, path traversal 404 semantics, content type/header/body ordering, `.env` cache timing, placeholder replacement, server import/re-export compatibility, and retained `Handler._handle_static_get` monkeypatch behavior.
- Functional commit `7fe1330 Extract hook route controller`: moved optional `/api/hooks/notify` POST acknowledgement policy into new `codoxear/hook_routes.py`; it remains intentionally unauthenticated, drains the body through injected `_read_body`, ignores content, returns JSON 200 `{"ignored": True}`, and lets body-read errors propagate to the existing route exception mapper.
- Hook-route diagnostic validation: `python3 -m pytest -q tests/test_hook_routes.py tests/test_route_decomposition_source.py tests/test_static_routes.py tests/test_static_assets.py tests/test_url_prefix.py tests/test_json_body_source.py tests/test_message_route_source.py tests/test_transcript_export.py tests/test_file_inspect.py` -> `122 passed, 56 subtests`.
- Hook-route Docker acceptance: focused Docker on port 18940 for the same affected groups passed; full Docker on port 18941 -> `1058 passed, 1 skipped, 107 subtests`. Clean-room review `/tmp/codoxear-hook-routes-review.md` returned `NO BLOCKERS`; it verified exact route path, no auth dependency, body drain before ignored response, error propagation, order after queue and before 404, and no state mutation.
- Current route ownership state: `Handler` no longer owns endpoint-specific HTTP composition. It retains central request dispatch, URL-prefix parsing/redirect, route-dependency assembly, JSON/body parsing wrappers, route exception mapping, and thin compatibility wrappers for static/voice seams. Next semantic refactor should target one of those remaining ownership clusters or a larger non-route `SessionManager` responsibility, not another route branch.

## 2026-06-26T17:15:21Z Session model and discovery ownership checkpoint
- Functional commit `b718f24 Extract session model`: moved the `Session` dataclass out of `server.py` into `codoxear/session_model.py`, kept `server.Session` as the identical re-export, removed the stale `dataclass` import from `server.py`, and added `tests/test_session_model.py` for re-export identity and non-trivial runtime defaults.
- Session-model validation: local focused `196 passed, 26 subtests`; focused Docker on port 18942 passed; full Docker on port 18943 -> `1060 passed, 1 skipped, 107 subtests`; clean-room review `/tmp/codoxear-session-model-review.md` returned `NO BLOCKERS`, verifying 44-field/default/order fidelity, stdlib-only imports, no circular import, server import compatibility, and full local suite health.
- Read-only next-seam evidence: scout `/tmp/codoxear-next-server-seam-scout.md` and architect `/tmp/codoxear-next-server-seam-architect.md` independently ranked session discovery / sidecar reconciliation as the highest-value remaining `server.py` semantic boundary; launch lifecycle was ranked second and unattended scheduler third.
- Functional commit `086120a Extract session discovery service`: moved sidecar/socket/log discovery evidence collection into new `codoxear/session_discovery.py` using `DiscoveryDeps`, `DiscoveryResult`, `DiscoveryRegistration`, `DiscoveryStaleAction`, and `DiscoveryRecentCwd`; `SessionManager._discover_existing` now binds dependencies, calls `_discover_sessions`, applies typed results, and updates `_last_discover_ts`.
- Discovery ownership boundary: `session_discovery.py` owns sidecar metadata validation consumption, pid/proc-open rollout discovery decisions, hidden-session exclusion, socket state probing, broker-state validation, log-token selection, and typed stale/registration/recent-cwd records. `SessionManager` still owns `_sessions` mutation, pending-attachment and commit-unknown overlays, cache reset on new/log-changed sessions, recent-cwd persistence, stale state deletion, unhide/persist side effects, and launch-failure ledger writes.
- Discovery tests added in `tests/test_session_discovery.py`: direct no-server-import guard, valid registration from sidecar/log/socket state, missing metadata clear-state action, malformed sidecar skip/no-prune, dead owned no-log failure action without clear-state, hidden live exclusion, and live unresponsive socket skip/no-unlink.
- Diagnostic validation after discovery extraction: local focused `tests/test_session_discovery.py tests/test_session_model.py tests/test_stale_sidecars.py tests/test_session_resume.py tests/test_sessions_pending_log_idle.py tests/test_hidden_sessions_startup.py tests/test_launch_provenance.py tests/test_session_routes.py tests/test_route_decomposition_source.py` -> `108 passed, 16 subtests`; local full `python3 -m pytest -q tests/ -x` -> `1068 passed, 107 subtests`.
- Docker acceptance after discovery extraction: focused Docker on port 18946 for discovery/stale/resume/pending/hidden/launch_provenance/session_routes/route_decomposition passed; full Docker on port 18947 -> `1067 passed, 1 skipped, 107 subtests`.
- Clean-room review path: two initial reviewer/critic attempts failed due subagent Opus `output_config.effort` configuration and produced no findings; they were not counted. Replacement Sonnet reviewer `add00e9b-6ff4-4a4f-9a25-04acb963585f` saved `/tmp/codoxear-session-discovery-review.md` and returned `NO BLOCKERS`, verifying no server import cycle, stale/malformed/hidden/dead/unresponsive semantics, recent-cwd ordering, log binding/coercion/token precedence, apply/upsert cache and overlay ownership, and full local suite health.
- Tests-only commit `57a412e Cover discovery cleanup edge cases`: closed the reviewer's non-blocking isolation-test note by adding direct coverage for hidden-dead session unhide+unlink/no-clear-state and definitely-stale dead socket unlink-only behavior. Validation: `python3 -m pytest -q tests/test_session_discovery.py` -> `9 passed`; local focused discovery/session route group -> `108 passed, 16 subtests`; focused Docker on port 18948 for the same group passed.
- Current next semantic seam: after discovery, remaining high-value `server.py` ownership clusters are launch lifecycle (`spawn_web_session`, launch attempt recording, tmux/direct process orchestration, metadata wait), schedulers (`_unattended_sweep`, queue scan), session projection/listing, and central HTTP adapter/dependency assembly. Scout/architect evidence points to launch lifecycle next, with an enabling design step around launch-attempt recording/context before moving the full spawn method.

## 2026-06-26T17:32:27Z Launch attempt recorder checkpoint
- Functional commit `d8c8ca9 Extract launch attempt recorder`: introduced `LaunchAttemptRecorder` in `codoxear/launch_ledger.py` and replaced the local `record_launch` / `fail_launch` composition closures inside `SessionManager.spawn_web_session` with the ledger-owned recorder.
- Ownership boundary: `LaunchAttemptRecorder` owns launch-attempt record dict composition for normal state transitions and failure records, persistence-callback invocation, and persistence-failure stderr reporting. `SessionManager.spawn_web_session` still owns launch id/base context, environment/argv construction, tmux/direct process orchestration, metadata wait, and raising the server-specific `SessionLaunchError`.
- Semantics preserved: normal records still copy the base launch record, set `state`, set `updated_ts`, merge extras last, and call `_record_launch_attempt`; failure records still set `state=failed`, `stage`, `error=str(error)`, `updated_ts`, merge extras last, catch persistence failures, write `error: failed to write launch attempt record: ...` to stderr, and return the raw record for `SessionLaunchError` response redaction.
- Tests added: `tests/test_launch_ledger.py` directly covers state transition recording without base mutation, failure-record field composition/extras, and persistence-failure fallback/stderr behavior.
- Diagnostic validation before commit: focused local launch tests -> `49 passed, 12 subtests`; local full suite -> `1073 passed, 107 subtests`.
- Docker acceptance before commit: focused Docker on port 18949 passed; full Docker on port 18950 -> `1072 passed, 1 skipped, 107 subtests`.
- Clean-room review `072cd1b3-aec2-464d-ad54-d609ba479a5d`, saved to `/tmp/codoxear-launch-recorder-review.md`, returned `NO BLOCKERS`; it verified record/failure ordering, all spawn call-site migrations, no `server.py` import from `launch_ledger`, redaction behavior, import hygiene, and noted only the intentional eager stderr binding at recorder construction.
- Current next semantic seam: launch lifecycle remains the highest-value non-route `server.py` target. The recorder extraction makes the failure-recording invariant explicit, so the next tranche can move launch context/dependency construction or tmux/direct runner orchestration without mixing that move with ledger semantics.

## 2026-06-26T18:02:48Z Session launch process runner checkpoint
- Functional commit `c5b7080 Extract session launch process runner`: added `codoxear/session_launcher.py` with `LaunchProcessRequest`, `LaunchProcessDeps`, `LaunchProcessFailure`, and `launch_broker_process`; `SessionManager.spawn_web_session` now assembles request/dependencies and wraps `LaunchProcessFailure` into server-owned `SessionLaunchError`.
- Ownership boundary: `session_launcher.py` owns direct/tmux process-launch sequencing, tmux env/window/shell command construction, tmux retry attempts, pane-created/pending/broker-meta-bound state transitions, direct `Popen` invocation, early-exit failure mapping, stderr-drain thread, and zombie-prevention wait thread. `SessionManager` still owns cwd creation, worktree creation, resume-candidate/live-target validation, backend argv/env assembly, launch id/spawn nonce/base ledger record, `_wait_for_spawned_broker_meta`, `_tmux_pane_snapshot`, and redacted `SessionLaunchError` response semantics.
- Compatibility seams preserved: server injects `shutil.which`, `subprocess.run`, `subprocess.Popen`, `_wait_or_raise`, `_wait_for_spawned_broker_meta`, `_tmux_pane_snapshot`, and `_drain_stream`, so existing monkeypatch paths and metadata/snapshot ownership remain stable for this tranche.
- Source sentinel update: `safe_filename` consumption followed true ownership from `server.py` into `session_launcher.py`; upload staging/injection helpers remain server-injected into control routes.
- Tests added in `tests/test_session_launcher.py`: import isolation (no `codoxear.server` import), direct success, direct early-exit failure, and tmux metadata-pending path.
- Diagnostic validation before `c5b7080`: py-compile of launcher/server/tests passed; focused local launch/source group -> `56 passed, 12 subtests`; full local -> `1077 passed, 107 subtests`.
- Docker acceptance before `c5b7080`: focused Docker on port 18951 passed for launcher/resume/ledger/provenance/file-upload-source group; full Docker on port 18952 -> `1076 passed, 1 skipped, 107 subtests`.
- Clean-room review `eb76e4e3-317a-4a19-9e8d-915ff6794fc1`, saved to `/tmp/codoxear-session-launcher-review.md`, returned `NO BLOCKERS`; it verified direct and tmux path equivalence, `LaunchProcessFailure`/`SessionLaunchError` redaction boundary, no server import cycle, dependency-injection seams, safe_filename ownership, meaningful tests, and independently ran local full suite -> `1077 passed, 107 subtests`.
- Reviewer non-blocking notes: `_clean_optional_text` is now duplicated in `session_launcher.py` as an identical private helper (existing debt, no behavior impact), and several launcher branches lacked direct unit coverage.
- Tests-only commit `0568b5b Cover session launcher edge cases`: closed the useful branch-coverage note by adding tests for direct `broker_spawn` failure, direct stderr-drain plus wait-thread order, tmux success/broker-meta-bound, missing-session + duplicate-session retry, tmux launch failure, non-int broker metadata, and dead-pane metadata failure with a fresh snapshot.
- Negative test-authoring evidence: first run of the new tests failed (`1 failed, 10 passed`) because the stderr-drain test used Python class-scope lookup for an outer `stderr` variable; fixed by making stderr an instance attribute and validating bound wait target by name. This was a test bug, not runtime behavior evidence.
- Validation after `0568b5b`: direct `tests/test_session_launcher.py` -> `11 passed`; focused local launch/source group -> `63 passed, 12 subtests`; full local -> `1084 passed, 107 subtests`; focused Docker on port 18953 passed; full Docker on port 18954 -> `1083 passed, 1 skipped, 107 subtests`.
- Current next semantic seam: launch lifecycle can now move either launch context/dependency construction or metadata wait/pending reconciliation next. Do not move `_wait_for_spawned_broker_meta` casually because it still depends on server app paths and sidecar liveness policy; if moved, make sidecar metadata/live-pid dependencies explicit and preserve existing patch seams.

## 2026-06-26T18:20:00Z Launch metadata wait checkpoint
- Functional commit `0080b29 Move launch metadata wait helper`: moved the `_wait_for_spawned_broker_meta` polling implementation from `server.py` into `codoxear/session_launcher.py` as `wait_for_spawned_broker_meta`.
- Ownership boundary: `session_launcher.py` now owns the tmux launch metadata polling loop: sorted `*.json` sidecar scan, adjacent socket path derivation, malformed metadata skip, cleaned spawn-nonce match, required live broker pid validation, 0.05s poll sleep, and timeout error text. `server._wait_for_spawned_broker_meta` remains a compatibility wrapper that injects `SOCK_DIR` and the caller/default timeout.
- Sidecar/liveness semantics preserved: the new helper defaults to `sidecar_metadata.read_metadata` and `sidecar_metadata.required_live_pid`; direct tests inject those dependencies to isolate invalid metadata and dead-pid skip behavior without importing `server.py`.
- Existing patch seams preserved: `tests/test_session_resume.py` still imports `codoxear.server._wait_for_spawned_broker_meta`, patches `codoxear.server.SOCK_DIR`, and patches `codoxear.server._wait_for_spawned_broker_meta` in tmux spawn tests. The wrapper reads `SOCK_DIR` at call time.
- Tests added in `tests/test_session_launcher.py`: invalid metadata / wrong nonce / dead pid skipped until a live matching sidecar; timeout path uses a deterministic injected clock and verifies exactly one 0.05s sleep plus unchanged error message.
- Negative evidence: the first direct metadata-wait test used real time with `timeout_s=0.0` and failed because the loop could expire before scanning. The test was repaired by injecting `now=lambda: 0.0`; this was test timing design, not product behavior evidence. A combined heredoc/full-suite shell command was malformed and produced a Python `SyntaxError`; it was discarded and rerun correctly.
- Validation before commit: py-compile of launcher/server/tests passed; focused local launch/source group -> `65 passed, 12 subtests`; import isolation check reported `server_imported False`; full local -> `1086 passed, 107 subtests`; focused Docker on port 18955 passed; full Docker on port 18956 -> `1085 passed, 1 skipped, 107 subtests`; LSP diagnostics for `session_launcher.py` reported no diagnostics while `server.py` only had existing unrelated diagnostics.
- Clean-room review `252a2aac-25d9-425f-9841-15de48e82916`, saved to `/tmp/codoxear-metadata-wait-review.md`, returned `NO BLOCKERS`; it verified old/new behavior parity, server wrapper correctness, no `codoxear.server` import from `session_launcher.py`, stale import cleanup, and test soundness.
- Current next semantic seam: launch context construction remains in `SessionManager.spawn_web_session` (cwd creation, resume validation, worktree, backend argv/env, launch id/spawn nonce, base ledger record, process request/deps). The next high-value launch tranche should extract that context/plan construction without moving registry ownership.

## 2026-06-26T18:55:00Z Launch context extraction checkpoint
- Functional commit `ae26091 Extract launch process context`: moved launch id/nonce generation, `CODEX_WEB_LAUNCH_ID`/`CODEX_WEB_SPAWN_NONCE` env mutation, base launch-attempt record construction, `LaunchAttemptRecorder` construction, and `LaunchProcessRequest` assembly from `SessionManager.spawn_web_session` into `codoxear.session_launcher.prepare_launch_process_context`.
- Ownership boundary: `session_launcher.py` now owns process-launch context preparation through `LaunchContextRequest` and `LaunchProcessContext`. `SessionManager` still owns cwd creation, resume-candidate/live-target validation, worktree creation, backend argv/env construction before launch markers, concrete process dependency injection, `SessionLaunchError` wrapping, and registry consequences.
- Behavior parity preserved: launch id remains `launch-{int(ts * 1000)}-{token_hex(4)}`; spawn nonce remains `token_hex(8)`; the same env dict is mutated before process request assembly; base record still uses `cwd=str(spawn_cwd)` and `requested_cwd=cwd3`; `worktree_branch` remains ledger-only and is not added to `LaunchProcessRequest`; recorder still receives `_record_launch_attempt`, `time.time`, and `sys.stderr` from the server call site.
- Tests added in `tests/test_session_launcher.py`: deterministic token/time injection verifies token sizes/order, env marker mutation, request fields, full base record through recorder output, distinct `spawn_cwd`/`requested_cwd`, ledger-only `worktree_branch`, and fresh base-copy behavior across recorder transitions.
- Validation for exact final diff: py-compile of launcher/server/tests passed; direct `tests/test_session_launcher.py` -> `15 passed`; focused local launch/provenance group -> `72 passed, 12 subtests`; full local -> `1088 passed, 107 subtests`; focused Docker on port 18959 -> `72 passed, 12 subtests`; full Docker on port 18960 -> `1087 passed, 1 skipped, 107 subtests`; `git diff --check` passed; import isolation check reported `server_imported False`; `session_launcher.py` LSP diagnostics reported no diagnostics and `server.py` diagnostics remained existing/unrelated.
- Clean-room review: successful model-isolated delegate review saved to `/tmp/codoxear-launch-context-review.md` returned `NO BLOCKERS`, verifying behavior parity, ownership boundary, import boundary, test coverage, and non-exposure of mutable base record.
- Negative review-runtime evidence: earlier review attempts `595ab372-25d9-425f-9841-15de48e82916`, `958f1d0b-0e57-4eca-acfb-d27c96bddfc6`, and `5ee288be-26d9-4cb1-a72f-ede82bc7435f` failed before code inspection with `output_config.effort` runtime errors; `ab3a77ba-2149-4a1c-a137-5e53d1722519` failed before code inspection because the requested Gemini model was unavailable. These runs are not review evidence.
- Read-only next-seam scouting: remaining launch responsibilities in `server.py` include concrete `LaunchProcessDeps` assembly plus server-owned wrappers (`_wait_or_raise`, `_drain_stream`, `_tmux_pane_snapshot`, `_wait_for_spawned_broker_meta`) and shared `TMUX_SESSION_NAME` configuration. Moving dependency construction next requires separating server config/monkeypatch seams from process-run ownership rather than blindly moving helpers.

## 2026-06-26T19:05:00Z Launch wait/drain extraction checkpoint
- Functional commit `9f51ab4 Move launch wait and drain helpers`: moved early-exit polling and stderr-drain implementations from `server.py` into `codoxear.session_launcher.wait_or_raise` and `codoxear.session_launcher.drain_stream`.
- Ownership boundary: `session_launcher.py` now owns direct-process wait/drain mechanics: deadline calculation, 0.05s polling sleep, early-exit stderr collection, byte-stderr decode/strip/4000-char clipping, silent return after timeout when still running, and 65536-byte drain loop with close-after-EOF semantics. `server._wait_or_raise` and `server._drain_stream` remain compatibility wrappers.
- Patch seams preserved: `SessionManager.spawn_web_session` still injects server wrapper names into `LaunchProcessDeps`; existing tests patching `codoxear.server._wait_or_raise`, `codoxear.server.subprocess.Popen`, `codoxear.server.shutil.which`, `_wait_for_spawned_broker_meta`, and `_tmux_pane_snapshot` still intercept their paths.
- Tests added in `tests/test_session_launcher.py`: non-byte stderr is ignored (preserving old behavior), byte stderr is decoded/clipped, running process sleeps once then returns when deadline elapses, and drain reads 65536-byte chunks until EOF then closes.
- Validation for exact final diff before commit: py-compile of launcher/server/tests passed; direct `tests/test_session_launcher.py` -> `19 passed`; focused local launch/provenance group -> `76 passed, 12 subtests`; full local -> `1092 passed, 107 subtests`; `git diff --check` passed; import isolation check reported `server_imported False`; focused Docker on port 18961 -> `76 passed, 12 subtests`; full Docker on port 18962 -> `1091 passed, 1 skipped, 107 subtests`; `session_launcher.py` LSP diagnostics reported no diagnostics and `server.py` diagnostics remained existing/unrelated.
- Clean-room review `2fbb87cb-c442-4a64-bf11-d001f87c94a9`, saved to `/tmp/codoxear-wait-drain-review.md`, returned `NO BLOCKERS`; it verified wait/drain behavior parity, wrapper seam preservation, launch ownership boundary, and no import-cycle regression.
- Reviewer note: one new test name says decoded stderr while asserting the non-byte-stderr ignored case. The assertion is correct and the naming issue is non-blocking.
- Read-only next-seam scouting: `_tmux_pane_snapshot` is used both by launch orchestration and stale/pruned session failure recording, so its natural owner is tmux runtime inspection rather than only spawn sequencing. Moving it requires preserving `TMUX_SESSION_NAME` fallback and server patch seams.

## 2026-06-26T19:18:00Z Tmux runtime extraction checkpoint
- Functional commit `3f19104 Extract tmux pane snapshot helper`: moved the tmux pane inspection/capture implementation from `server._tmux_pane_snapshot` into new `codoxear.tmux_runtime.tmux_pane_snapshot`.
- Ownership boundary: `tmux_runtime.py` now owns tmux target inspection/capture/parsing: pane-id preference, window fallback target construction, display-message command/format parsing, capture-pane command, inspect/capture error records, and 4000-character tail clipping. `server._tmux_pane_snapshot` remains a compatibility/config wrapper that injects `TMUX_SESSION_NAME` and `subprocess.run`.
- Patch seams preserved: launch orchestration still injects `tmux_pane_snapshot=_tmux_pane_snapshot`; stale/pruned session failure recording still calls `server._tmux_pane_snapshot`; existing tests that patch `codoxear.server._tmux_pane_snapshot` continue to intercept both paths.
- Behavior quirk preserved and documented by tests: when `pane_id` is blank and `window` has surrounding spaces, the old code only used cleaned `window` as an existence check but interpolated the raw `window` into `f"{TMUX_SESSION_NAME}:{window}"`; the new helper preserves that raw-window target.
- Tests added in `tests/test_tmux_runtime.py`: import isolation, no-target no-run, pane-id preference and tail clipping, raw-window fallback target, display failure without capture, capture failure fallback, and empty-stream exit-status fallback.
- Negative test-authoring evidence: first focused run failed because the test asserted target at argv index 5 instead of 4; second run failed because the test incorrectly expected normalized `window`. Both were test expectation errors; product behavior remained old-code parity.
- Validation before commit: py-compile of tmux runtime/server/tests passed; focused local tmux/launcher/resume/provenance group -> `71 passed, 12 subtests`; full local -> `1098 passed, 107 subtests`; `git diff --check` passed; Python import check loaded `codoxear.tmux_runtime` and `codoxear.server` from the recovery checkout and confirmed `server._tmux_pane_snapshot` exists; focused Docker on port 18963 -> `71 passed, 12 subtests`; full Docker on port 18964 -> `1097 passed, 1 skipped, 107 subtests`; `tmux_runtime.py` LSP diagnostics reported no diagnostics.
- LSP note: `server.py` LSP reported a new unresolved-import diagnostic for `.tmux_runtime`, but Python import, py_compile, full local tests, and Docker tests all resolved the module. Clean-room review independently checked `importlib.util.find_spec` for both modules and classified the diagnostic as an indexing/tooling artifact rather than a runtime/package issue.
- Clean-room review `13dc2fba-9f6b-453f-bdb3-82565357b7ae`, saved to `/tmp/codoxear-tmux-runtime-review.md`, returned `NO BLOCKERS`; it verified target/command/capture parity, wrapper seam preservation, ownership boundary, import isolation, and test coverage.
- Current next-seam model: launch lifecycle process mechanics are largely out of `server.py`; remaining heavy responsibilities are launch precondition/plan sequencing (cwd creation, resume lookup/live exclusion, worktree, dotenv, backend argv/env) and broad `SessionManager` runtime/state authority.

## 2026-06-26T19:38:00Z Launch precondition plan checkpoint
- Functional commit `b4c5381 Extract launch precondition plan`: moved launch precondition/plan sequencing from `SessionManager.spawn_web_session` into new `codoxear.session_launch_plan.prepare_launch_plan`.
- Ownership boundary: `session_launch_plan.py` now owns backend normalization for web launch, cwd resolution/creation validation, worktree-vs-resume rejection, worktree spawn cwd selection, broker argv prefix, backend args, resume candidate lookup and backend resume args, user arg append ordering, dotenv merge, backend environment preparation, and the resulting `LaunchPlan`. `SessionManager` still supplies server-specific dependencies and authorities: `_resolve_dir_target`, `_create_git_worktree`, `_codex_trust_override_for_path`, `_list_resume_candidates_for_cwd`, `self._live_session_for_resume_target`, `_load_env_file`, homes, process deps, `LaunchProcessFailure` wrapping, and registry consequences.
- Behavior parity preserved: `requested_cwd` remains `str(cwd_path)` after `_resolve_dir_target`; missing cwd creation and mkdir error text are unchanged; worktree creation still happens before argv/trust override; resume lookup still uses stripped id with `agent_backend` and `limit=1000`; raw `resume_session_id` still flows into env/plan/context/ledger while stripped id is used for lookup and backend resume argv; dotenv values still use `setdefault`; stale launch request env cleanup remains delegated to `backend_launch.apply_backend_environment`.
- Tests added in `tests/test_session_launch_plan.py`: import isolation, missing cwd creation plus Codex argv/env contract, dotenv missing-only behavior, raw-vs-stripped resume id behavior, live resume rejection, worktree spawn cwd/trust override, and resume+worktree rejection. Existing `tests/test_session_resume.py` continues to cover server wrapper patch seams and spawn integration.
- Negative implementation evidence: an initial automated server replacement matched an earlier `backend_name = normalize_agent_backend(agent_backend)` occurrence and corrupted a large server region. Repaired by `git checkout -- codoxear/server.py` and reapplying with a method-local `spawn_web_session` anchor before validation.
- Negative test-authoring evidence: first plan test checked `target.is_dir()` after `TemporaryDirectory` cleanup; fixed by asserting inside the context. This was a test lifetime bug, not product behavior evidence.
- Validation before commit: py-compile of launch plan/server/tests passed; focused local plan/resume/launcher/backend/provenance group -> `75 passed, 12 subtests`; full local -> `1104 passed, 107 subtests`; `git diff --check` passed; Python import/find_spec loaded `codoxear.session_launch_plan` from this checkout and did not import `codoxear.server`; focused Docker on port 18965 -> `75 passed, 12 subtests`; full Docker on port 18966 -> `1103 passed, 1 skipped, 107 subtests`; `session_launch_plan.py` LSP diagnostics reported no diagnostics.
- LSP note: `server.py` LSP reported unresolved-import diagnostics for new modules (`.session_launch_plan`, previously `.tmux_runtime`) even though Python import, find_spec, py_compile, full local tests, and Docker tests resolved them. Clean-room review classified this as a tool/index artifact rather than a runtime package issue.
- Clean-room review `f6ae41bb-61e0-46f5-8183-feda79599deb`, saved to `/tmp/codoxear-launch-plan-review.md`, returned `NO BLOCKERS`; it verified sequencing parity, raw/stripped resume semantics, patch seams, ownership boundary, import isolation, and added tests. The reviewer additionally ran py_compile, git diff --check, import/find_spec checks, and `pytest -q tests/test_session_launch_plan.py tests/test_session_resume.py` -> `38 passed`.
- Current next-seam model: launch process/precondition mechanics are substantially out of `server.py`; the largest remaining server authority is `SessionManager.list_sessions`, which mixes discovery/prune/update side effects, run-settings/history backfill, UI projection, sidebar priority, orphan recovery rows, launch-failure overlay filtering, and sorting.

## 2026-06-26T20:05:00Z Session listing projection checkpoint
- User corrected the work style during this tranche: do not count green tests, diagnostics, import status, or review gates as architectural progress; use review only as issue discovery and focus on actual ownership/semantics.
- Functional commit `5f8bbe8 Extract session listing projection`: introduced `codoxear/session_listing.py` and moved session-list row projection out of `SessionManager.list_sessions`.
- Actual architecture moved: `session_listing.py` now owns active session row schema from `ActiveSessionRowFacts`, private staging-key removal/public row finalization, orphan recovery synthetic failed rows for inactive direct commit-unknown/queue recovery state, and session-list sort order.
- What intentionally did not move: `SessionManager` still owns discovery/prune/meta-counter sequencing, runtime registry snapshots, legacy file-history migration, sidebar dependency/snooze cleanup, recent-cwd persistence mutation, run-settings/history backfill, log/broker busy resolution, git-branch probing, launch-attempt row selection/filtering, dirty-state saves, and registry consequences.
- Source sentinels followed ownership: unattended source checks now assert that `server.py` computes unattended facts and `session_listing.py` writes active and orphan recovery row fields.
- Focused development sanity before commit: py-compile of changed Python files passed; direct listing/server queue/unattended source group returned `103 passed, 22 subtests` after source-sentinel repair. These were not treated as architecture completion evidence.
- Failed clean-room review attempts during this tranche (`reviewer` and `critic`) died before inspecting code due model/runtime configuration errors; they are not review evidence.
- Next architecture target: `SessionManager.list_sessions` still mixes persistent overlay repair with runtime evidence enrichment. The next useful cut should separate either the repair phase (file/sidebar/recent-cwd mutation) or the enrichment phase (history/run-settings/busy/git) without hiding mutations inside projection code.

## 2026-06-26T20:25:00Z Session store repair checkpoint
- Functional commit `dcf2e0d Move file history repair into session store`: moved session file-history legacy migration, add/dedupe/cap, and clear-with-cwd-legacy-bucket semantics into `SessionStore` methods. `SessionManager` now derives session keys, holds the lock, calls store operations, and decides whether/when to persist.
- Functional commit `fd391f3 Move listing overlay repair into session store`: moved sidebar dependency/snooze repair and recent-cwd recency updates into `SessionStore`. `SessionManager.list_sessions` now consumes `SidebarSessionState` and calls `note_recent_cwd` instead of directly mutating `_sidebar_meta` and `_recent_cwds`.
- Actual architecture moved: persistent overlay repair for file history, sidebar metadata, and recent cwd now lives with persistent state ownership rather than in the listing response pipeline.
- What intentionally did not move: session-key derivation still requires `SessionManager` because it depends on live `_sessions`; dirty saves still remain manager-triggered; runtime history/run-settings/log/broker/git enrichment still remains inside `list_sessions` and needs a separate side-effect model before extraction.
- Focused development sanity after each slice: store/file/listing groups returned `108 passed, 22 subtests` after file-history repair and `117 passed, 22 subtests` after sidebar/recent-cwd repair. These were used to catch mistakes, not as proof of architectural completion.
- Current next architecture target: model `list_sessions` runtime enrichment as explicit results/side effects before moving it. Avoid hiding `Session` mutation or log reads behind a projection helper.

## 2026-06-26T20:45:00Z Listing runtime/projection follow-up checkpoint
- Functional commit `090e5b3 Move listing priority calculation`: moved clip/bucket/half-life priority math and final priority calculation into `session_listing.py`; server retains configuration constants and wrapper names for existing route dependency injection.
- Functional commit `79ecc22 Move launch overlay listing selection`: moved failed-launch overlay inclusion/exclusion rules into `session_listing.build_launch_attempt_rows`. `SessionManager` still snapshots hidden/active launch identities and reads the ledger; listing now owns hidden/active row suppression for the session-list overlay.
- Functional commit `dc529a2 Move listing runtime backfill application`: moved application of history-scan and run-settings evidence onto `Session` cache objects into `session_runtime.py` via `apply_history_backfill` and `apply_run_settings_backfill`. `SessionManager.list_sessions` still reads log evidence and holds the lock, but no longer owns those mutation rules.
- Current `list_sessions` responsibility after these commits: discovery/prune/meta-counter prelude, active snapshot composition, evidence acquisition from log/boundary/git helpers, launch/orphan overlay calls, dirty-save orchestration, and final row assembly. It no longer owns row schema, public-row finalization, ordering, file/sidebar/recent-cwd repair mechanics, launch overlay filtering, priority math, or runtime cache backfill mutation rules.
- Focused development sanity during this tranche: priority/listing/sidebar/diagnostics group returned `34 passed`; launch overlay/listing/provenance/sidebar group returned `45 passed, 12 subtests`; runtime/backfill/idle/sidebar group returned `74 passed, 4 subtests` after repairing the new test fixture to instantiate the real `Session` model shape. These runs were used as mistake detectors, not as architecture proof.
- Next architecture target: decide whether to extract active snapshot composition or runtime evidence acquisition. Runtime evidence acquisition still depends on manager-only helpers (`_last_conversation_ts_from_tail`, `_read_run_settings_from_log`, `_confirmed_send_boundary_unresolved_for_session`, `idle_from_log_path`, `_current_git_branch`), so moving it requires an explicit dependency object rather than copying the loop.

## 2026-06-26T21:05:00Z Active listing snapshot checkpoint
- Functional commit `1a7b9b7 Extract active session listing snapshot`: moved active session snapshot composition from `SessionManager.list_sessions` into `session_listing.build_active_session_rows_snapshot`.
- Actual architecture moved: active snapshot construction now combines live `Session` fields, queue state, unattended/alias overlays, store repair outputs, cwd-path resolution, provider-choice adaptation, and listing priority into staged active rows plus dirty flags. `SessionManager` still holds the lock, supplies dependencies/config, and decides when dirty stores are saved.
- What intentionally did not move: runtime evidence acquisition from history tail, run-settings logs, confirmed-send boundary, idle log parsing, and git probing remains in `SessionManager.list_sessions`; those reads still depend on manager/server helper seams.
- Focused development sanity: py-compile of changed listing/server/tests passed; listing/server queue/sidebar/store group returned `126 passed, 22 subtests`. This was used to catch shape errors, not as the reason the architecture is better.
- Current `list_sessions` shape: discovery/prune/meta-counter prelude; active snapshot call; runtime evidence enrichment loop; failed-launch overlay call; orphan recovery overlay call; dirty saves; sort/return.

## 2026-06-26T21:20:00Z Queue promotion item-state checkpoint
- Functional commit `e618c54 Move queue promotion item state into queue store`: moved queue promotion head selection, queue recovery-item detection, commit-unknown marker set/clear, and commit-unknown marker preservation into `QueueStore`.
- Actual architecture moved: queue item state transitions for auto-send promotion now live with the queue persistence/mutation authority. `SessionManager._promote_queue_head_if_sendable` still owns remote readiness checks, idle-grace timing, session-level `queue_sending_item_id`/`queue_idle_since`, the actual `send()` call, and save orchestration.
- Direct unknown send state remains manager-owned; `_queue_has_recovery_items_locked` still combines manager direct-unknown evidence with `QueueStore.has_recovery_items`.
- The commit preserved the old commit-unknown error response behavior by returning `dict(item)` from the queue store preservation helper, including incidental item fields, rather than sanitizing through `copy_queue_item`.
- Focused development sanity: queue-store/server queue/sweep/route group returned `106 passed, 22 subtests`. This was used as a mistake detector after moving the item-state mechanics.

## 2026-06-26T21:45:00Z Unattended sweep policy checkpoint
- Functional commit `1a51a9a Move unattended sweep policy helpers`: moved unattended scheduler policy decisions into `codoxear/unattended.py`.
- Actual architecture moved: unattended config normalization for sweep decisions, scope-key selection, cooldown blocking, final-assistant tail eligibility, exhausted-budget disabling, live prompt decision, and post-send remaining-injection decrement now live with the unattended subsystem.
- What intentionally did not move: `SessionManager._unattended_sweep` still owns session discovery/prune freshness, session snapshot collection, broker state reads, local queue-length checks, transcript-tail reads, input locking, `send()` invocation, last-injected bookkeeping, persistence calls, and per-session error isolation/logging.
- Source sentinel update: unattended session-list field source checks now point to `session_listing.py`, reflecting the earlier active-listing snapshot extraction where list-row fields moved out of `server.py`.
- Focused development sanity: py-compile of changed files passed; unattended store/sweep/mode/input group returned `27 passed`. This was used to catch handoff errors; the architecture claim is the ownership move above.

## 2026-06-26T22:00:00Z Queue promotion runtime flags checkpoint
- Functional commit `8b50af4 Extract queue promotion runtime flags`: introduced `codoxear/queue_runtime.py` for session-level queue promotion flags.
- Actual architecture moved: `queue_runtime.py` now owns `queue_idle_since` reset/idle-grace progression, queue promotion start, and queue promotion clear semantics for `Session.queue_sending_item_id` and `Session.queue_idle_since`.
- Current queue layering: `QueueStore` owns queue item state and commit-unknown markers; `queue_runtime.py` owns session queue promotion flags; `SessionManager._promote_queue_head_if_sendable` still owns remote readiness checks, idle-grace branch timing, `send()` invocation, queue persistence calls, and error-branch orchestration.
- Focused development sanity: py-compile of queue runtime/server/tests passed; queue runtime/store/server queue/sweep/route group returned `109 passed, 22 subtests`.

## 2026-06-26T22:25:00Z Listing runtime enrichment checkpoint
- Functional commit `34f2545 Move listing runtime enrichment`: moved the active-listing runtime enrichment pass from `SessionManager.list_sessions()` into `codoxear/session_runtime.py`.
- Actual architecture moved: `session_runtime.build_runtime_enriched_session_rows` now owns staged-row runtime interpretation: once-only history backfill from log tail, run-settings backfill from log metadata, priority recomputation after history backfill, confirmed-send boundary aware idle/busy resolution, git branch projection, public-row stripping of private listing keys, and `recent_cwd_dirty` reporting.
- Explicit remaining manager responsibilities: `list_sessions()` still owns discovery/prune/meta-counter prelude, active snapshot construction, manager-owned runtime probes (`idle_from_log_path`, confirmed-send boundary wrapper, log-size wrapper, run-settings/log-tail/git wrappers), failed-launch overlay composition, orphan recovery overlay composition, dirty saves, sort, and return.
- Regression found and fixed during the move: the initial insertion accidentally removed the `broker_runtime_state` definition header in `session_runtime.py`; focused tests failed during import, then the missing header was restored.
- Focused development sanity: py-compile of `session_runtime.py`, `server.py`, and `test_session_runtime.py` passed; runtime/listing/pending-log/sidebar/discovery/session-routes group returned `86 passed, 4 subtests`.

## 2026-06-26T22:45:00Z Runtime boundary/log-size probes checkpoint
- Functional commit `5c3c199 Move confirmed send runtime probes`: moved parseable log-size probing and confirmed-send boundary mechanics from `server.py` into `codoxear/session_runtime.py`.
- Actual architecture moved: `session_runtime.py` now owns complete-JSONL offset discovery, last parseable JSON object offset, safe log size projection, confirmed-send boundary unresolved checks, boundary clearing, and consume-and-clear semantics on `Session`.
- What intentionally remains in `SessionManager`: `_log_size_or_none` and `_confirmed_send_boundary_unresolved_for_session` wrappers still provide manager locking/compatibility around the runtime helpers; higher-level readiness/list/message snapshot code still decides when to call them.
- Preservation note: the moved parser loop was restored to the original `server.py` cursor semantics after a first edit rewrote the loop shape; focused tests passed after the restoration.
- Focused development sanity: py-compile of runtime/server/runtime tests passed; runtime, pending-log idle, server queue persistence, diagnostics routes, message route source, and session routes returned `140 passed, 26 subtests`.

## 2026-06-26T23:00:00Z Full local pytest after runtime probe slice
- Full local mistake-detector run after commits through `5c3c199` initially returned `1133 passed, 1 failed, 107 subtests`.
- Failure mechanism: `tests/test_file_upload_module_source.py` still expected pending-attachment and commit-unknown row projection literals in `server.py`, but that row projection now lives in `session_listing.py` after the earlier active-listing snapshot extraction. This was a stale ownership sentinel, not a runtime behavior failure.
- Test-only commit `1c52713 Update file upload listing source sentinel`: updated the file-upload source sentinel to check `session_listing.py` for `pending_attachment=bool(s.pending_attachment)` and the current `commit_unknown_send` projection expression.
- Re-run evidence: `pytest -q tests/test_file_upload_module_source.py` returned `3 passed`; full `pytest -q` returned `1134 passed, 107 subtests`.

## 2026-06-26T23:15:00Z Session readiness preconditions checkpoint
- Functional commit `4813bdb Move session readiness preconditions`: moved direct-send and queue-promotion blocker predicates from `SessionManager` into `codoxear/session_runtime.py`.
- Actual architecture moved: `session_runtime.session_allows_direct_send` now owns the direct-send semantics for direct commit-unknown state and pending attachment allowance; `session_runtime.session_allows_queue_promotion` now owns queue-promotion blocking by direct commit-unknown and pending attachment state.
- What intentionally remains in `SessionManager`: metadata refresh before probing, log-path change handling, broker state calls, confirmed-send boundary consumption, idle/log readiness resolution, send invocation, exception messages, and lock boundaries.
- Focused development sanity: py-compile of runtime/server/runtime tests passed; runtime, send ack, server queue persistence, queue sweep idle guard, file-upload source sentinel, and pending-log idle tests returned `149 passed, 26 subtests`.

## 2026-06-27T00:05:00Z Session queue coordinator checkpoint
- Functional commit `5b782ae Extract session queue coordinator`: introduced `codoxear/session_queue.py` and moved the queue coordination block out of `SessionManager`.
- Actual architecture moved: `SessionQueueCoordinator` now owns queue length reads, orphan-recovery marker propagation, direct-unknown/queue recovery detection, local queue list/append/enqueue/delete/update/move operations, queue session state lookup, auto-promotion head eligibility checks, idle-grace/session-flag transitions through `queue_runtime.py`, commit-unknown marker set/clear/preserve, sent-item pop, promotion error cleanup, queue save triggering, and queue promotion return-shape decisions.
- What intentionally remains in `SessionManager`: compatibility wrappers for existing route/tests, the input-lock around `enqueue`, direct send implementation, remote readiness probing (`_queue_remote_ready`), broker/log/confirmed-boundary interpretation, direct commit-unknown record persistence, session registry ownership, and public exception classes/messages.
- Size observation: `server.py` is now 4685 lines; the new `session_queue.py` is 292 lines. The diff removed 238 queue-coordination lines from `server.py` and added direct coordinator coverage.
- Regression caught while testing: the first direct coordinator commit-unknown test expected the pre-send marker timestamp, but old behavior and the coordinator both preserve a later timestamp when the send result is actually unknown. The test was corrected to the preserved behavior.
- Validation: py-compile of `session_queue.py`, `server.py`, and `test_session_queue.py` passed. Focused queue/send/attachment group returned `153 passed, 26 subtests`; full local `pytest -q` returned `1139 passed, 107 subtests`.

## 2026-06-27T00:25:00Z Confirmed send input protocol checkpoint
- Functional commit `454f64d Extract confirmed send input protocol`: introduced `codoxear/session_input.py` and moved direct confirmed-send protocol rules out of `SessionManager.send`.
- Actual architecture moved: `session_input.py` now owns direct-send precondition messages for commit-unknown state, pending attachments, queued prompt barriers, active queue item identity, sync-send support; broker confirmed-send response classification; commit-unknown message selection for malformed/marked/empty/incomplete/invalid responses; injection-error raising for broker error responses; and successful send mutation of busy/interrupted/remote queue length/confirmed-send boundary fields on `Session`.
- What intentionally remains in `SessionManager.send`: per-session input lock, live session lookup, local queue length acquisition, remote readiness probing, pre-send log-size capture, control-socket send call, control-socket failure/stale-session cleanup, direct commit-unknown record persistence, pending-attachment clearing, and queue-item/direct-send distinction after success.
- Source sentinel update: file-upload send-path source checks now target `session_input.py` for direct-send preconditions and direct-send response classification while keeping attachment-specific checks in `server.py`.
- Size observation: `server.py` is now 4666 lines; `session_input.py` is 81 lines.
- Validation: py-compile of `session_input.py`, `server.py`, and `test_session_input.py` passed. Focused send/input/queue/attachment/diagnostics/session-route group returned `146 passed, 26 subtests`; full local `pytest -q` returned `1144 passed, 107 subtests`.

## 2026-06-27T00:40:00Z Control socket client-call checkpoint
- Functional commit `af60173 Move control socket client call`: moved client-side Unix control-socket request/response mechanics from `SessionManager._sock_call` into `codoxear/control_socket.py`.
- Actual architecture moved: `control_socket.call_control_socket` now owns AF_UNIX socket creation, timeout assignment, connect, JSON-line request serialization, request-sent tracking, 65536-byte receive loop, empty-response result, JSON-line response decode, tracked `ControlSocketCallError`, and socket close. `ControlSocketCallError` also now lives in `control_socket.py` and is imported/re-exported by `server.py` for compatibility.
- What intentionally remains in `SessionManager`: `_sock_call` as a patch seam; stale session cleanup on state/tail/send/key failures; process liveness checks; unlinking sidecar/socket artifacts; and higher-level response semantics.
- Size observation: `server.py` is now 4637 lines; `control_socket.py` is 80 lines.
- Validation: py-compile of `control_socket.py`, `server.py`, and `test_control_socket.py` passed. Focused control/send/session group returned `137 passed, 22 subtests`; full local `pytest -q` returned `1146 passed, 107 subtests`.

## 2026-06-27T01:00:00Z Session control coordinator checkpoint
- Functional commit `c25b4a1 Extract session control coordinator`: introduced `codoxear/session_control.py` and moved `get_state`, `get_tail`, and `inject_keys` control-operation orchestration out of `SessionManager`.
- Actual architecture moved: `SessionControlCoordinator` now owns session/sock lookup for control operations, state-command dispatch and runtime cache update, tail-command dispatch and tail response validation, key-command request construction including interrupt marker, tracked attachment commit-unknown conversion after request-sent key write failures, dead broker/agent detection after state/tail/key failures, socket/sidecar unlinking, session removal, and the old distinction that state failures clear deleted-session state while tail/key failures only unlink/remove.
- What intentionally remains in `SessionManager`: `_sock_call` as patch seam, `_control_coordinator_for_manager` dependency wiring, process liveness primitive, `_clear_deleted_session_state` implementation, direct send socket call, attachment response classification, and public wrapper methods for routes/tests.
- Source sentinel update: file-upload and interrupt source tests now check `session_control.py` for key-injection request/commit-unknown/interrupt semantics while preserving manager wrapper expectations.
- Size observation: `server.py` is now 4582 lines; `session_control.py` is 101 lines.
- Validation: py-compile of `session_control.py`, `server.py`, and `test_session_control.py` passed. Focused control/source/send/stale/diagnostics/session-route group returned `173 passed, 26 subtests`; full local `pytest -q` returned `1151 passed, 107 subtests`.

## 2026-06-27T02:10:00Z Coordinator extraction tranche
- Functional commits after the session-control checkpoint:
  - `e559660 Move direct send socket control`: direct send socket dispatch, request-sent commit-unknown conversion, unsent dead-session cleanup, live unsent not-ready conversion, and timeout commit-unknown conversion moved into `SessionControlCoordinator.call_confirmed_send`.
  - `bcde64f Extract session listing coordinator`: `codoxear/session_list.py` now owns list prelude, active snapshot, runtime enrichment, failed-launch overlay, orphan recovery rows, dirty-store saves, and final sorting; `SessionManager.list_sessions` is a wrapper plus dependency wiring.
  - `33f11d5 Extract session metadata refresh coordinator`: `codoxear/session_refresh.py` now owns sidecar validation, missing-sidecar prune handoff, invalid-sidecar logging, transport/capability refresh, detach-tail handling, open-log rediscovery, Codex main-log coercion, run-settings/service-tier refresh, session mutation, cache reset, and optional queue drain.
  - `259ebb4 Extract session readiness coordinator`: `codoxear/session_readiness.py` now owns remote-ready resolution from broker/log/boundary state, state-after-metadata-probe sequencing, direct-send readiness, queue-promotion readiness, and attachment-injection readiness.
  - `ebeac59 Extract unattended sweep coordinator`: `codoxear/unattended_sweep.py` now owns unattended scheduler orchestration: discovery/prune prelude, enabled-session snapshots, exhausted-budget disable, scope cooldown checks, broker/queue/tail gates, input-lock prompt decision, send, success bookkeeping, persistence, and per-session error isolation.
  - `ea42dbd Extract queue sweep coordinator`: `codoxear/queue_sweep.py` now owns queue sweep discovery/prune, orphan-recovery marking, missing-session queue dropping, queue-save triggering, and one-head-per-sweep drain sequencing.
- Validation evidence per functional slice: each slice passed focused tests; full local `pytest -q` after each of the listing, refresh, readiness, unattended-sweep, and queue-sweep moves returned `1157 passed, 107 subtests`.
- Current size observation: see `wc -l` output in this OPS entry command; `server.py` is materially smaller and increasingly composed of wrappers/dependency wiring rather than embedded state machines.

## 2026-06-27T03:15:00Z Runtime/state coordinator tranche
- Functional commits after `97e47b5 Document coordinator extraction tranche`:
  - `b31012f Extract voice runtime coordinator`: `codoxear/voice_runtime.py` now owns notification-text attachment, session display-name lookup for notifications, delivery offset mutation, rollout-delta voice observation, resume-session muting, CC pending-tool seed lookup, and voice-push scan sweep.
  - `ffc8c33 Extract session log runtime coordinator`: `codoxear/session_log_runtime.py` now owns log-delta session mutation, turn-context model/effort backfill, idle-cache lookup/update, missing-log errors, and invalid idle-state errors.
  - `17df7f1 Extract session file history coordinator`: `codoxear/session_files.py` now owns session file-history keying, legacy key repair entry points, add/get/clear, and save triggering.
  - `2258c60 Extract session UI state coordinator`: `codoxear/session_ui_state.py` now owns hide/unhide, alias get/set/clear, sidebar metadata get/set, combined edit-session validation/mutation, and save triggering.
  - `f4320f6 Extract session unattended config coordinator`: `codoxear/session_unattended_config.py` now owns per-session Unattended get/set normalization, input-lock mutation, remaining-injection disable behavior, and save triggering.
  - `48e9bc2 Extract deleted session cleanup coordinator`: `codoxear/session_cleanup.py` now owns stale-socket prune cleanup, deleted-session state cleanup across aliases/sidebar/unattended/files/queues/input-locks/pending attachments/direct-unknown sends, recovery-preserving queue marking, unlinking, and save triggering.
  - `98e30e5 Extract session pending state coordinator`: `codoxear/session_pending_state.py` now owns pending-attachment mutation, direct unknown-send record cleaning/mutation/clear, orphan pruning, queue recovery marking, and persistence.
  - `c28c85e Extract recent cwd coordinator`: `codoxear/session_recent_cwd.py` now owns recent cwd remember/backfill/list behavior.
  - `baeb59a Extract session lifecycle coordinator` and `16d12bd Move live resume lookup into lifecycle`: `codoxear/session_lifecycle.py` now owns process/control-socket kill fallback, delete-session recovery/launch-failure hiding, active-session removal cleanup, launch-id hiding, and live resume-target matching while preserving the `_kill_session_via_pids` monkeypatch seam.
- Validation evidence per functional slice: each slice passed focused tests; full local `pytest -q` after each of voice runtime, log runtime, file history, UI state, Unattended config, cleanup, pending state, recent cwd, and lifecycle moves returned `1157 passed, 107 subtests`.
- Current size observation: `server.py` is now 3874 lines. The manager is mostly wrappers, dependency assembly, discovery application, launch orchestration, send/prelog path, route dependency assembly, and compatibility helpers.

## 2026-06-28T17:51:21Z Discovery/send runtime coordinator tranche
- Functional commits after `8d42eed Document runtime state extraction tranche`:
  - `41ede07 Move meta counter updates into log runtime`: `codoxear/session_log_runtime.py` now owns the meta-counter sweep, including log truncation reset, bounded JSONL chunk reads, thinking/tool/system counter accumulation, latest chat timestamp update, token fallback lookup, and busy-vs-idle counter reset. `SessionManager._update_meta_counters` is now a wrapper plus dependency wiring.
  - `7fb08b7 Extract discovery registry application`: `codoxear/session_discovery_registry.py` now owns applying `DiscoveryResult`: stale action cleanup/launch-failure recording, recent-cwd persistence triggering, and `DiscoveryRegistration` upsert into the runtime `Session` registry while preserving pending-attachment and direct-unknown-send overlays.
  - `66ad28c Extract dead session pruning coordinator`: `codoxear/session_prune.py` now owns broker state refresh for prune purposes, stale socket classification, broker/agent liveness fallback, web-owned pre-log launch-failure recording including tmux snapshot details, deleted-state cleanup, and socket/sidecar unlinking.
  - `e140a0d Extract confirmed send coordinator`: `codoxear/session_send.py` now owns confirmed-send orchestration around the existing input protocol and control coordinator: input locking, local queue precondition check, remote readiness gate, pre-send log-size capture, commit-unknown persistence, response parsing, successful send mutation, pending-attachment clearing, and pre-log submitted-message ledger recording through `PrelogUserMessageRecorder`.
- Source sentinel update: `tests/test_file_upload_module_source.py` now checks `session_send.py` for direct-send readiness/timeout/control-call/pending-clear ownership while keeping attachment-injection-specific checks in `server.py`, `session_readiness.py`, `session_control.py`, and `session_input.py`.
- Validation evidence by slice:
  - Meta-counter slice: py-compile of `session_log_runtime.py`/`server.py`; focused log/listing tests returned `73 passed, 4 subtests`; full local `pytest -q` returned `1157 passed, 107 subtests`.
  - Discovery registry slice: py-compile of `session_discovery_registry.py`/`server.py`; focused discovery/stale/list/pending tests returned `187 passed, 38 subtests`; full local `pytest -q` returned `1157 passed, 107 subtests`.
  - Dead-prune slice: py-compile of `session_prune.py`/`server.py`; focused prune/discovery/list tests returned `163 passed, 34 subtests`; full local `pytest -q` returned `1157 passed, 107 subtests`.
  - Send/prelog slice: py-compile of `session_send.py`/`server.py`/file-upload source sentinel; focused send/control/source tests returned `115 passed, 22 subtests`; full local `pytest -q` returned `1157 passed, 107 subtests`.
- Size observation: after `e140a0d`, `server.py` is 3592 lines. The manager is now dominated by wrappers, dependency assembly, launch-web-session orchestration, route dependency assembly, and compatibility helper exports; discovery application, dead-session pruning, log meta-counter scanning, and confirmed-send orchestration no longer live inline in the manager.

## 2026-06-28T18:00:00Z Web launch coordinator checkpoint
- Functional commit `816082a Extract web session launch coordinator`: introduced `codoxear/session_web_launch.py` and moved `SessionManager.spawn_web_session` orchestration into `SessionWebLaunchCoordinator`.
- Actual architecture moved: the coordinator now owns web-owned session launch glue across `LaunchPlanRequest`/`LaunchPlanDeps`, `LaunchContextRequest`, `LaunchProcessDeps`, `launch_broker_process`, and `LaunchProcessFailure` to `SessionLaunchError` mapping. Existing lower-level launch-plan and launch-process modules remain the source of truth for argv/env/worktree/resume/tmux/process behavior.
- What remains in `SessionManager`: public `spawn_web_session` wrapper, dependency assembly, live resume target seam through the lifecycle coordinator, and `SessionLaunchError` class definition for route/error compatibility.
- Validation: py-compile of `session_web_launch.py` and `server.py` passed. Focused launch/session tests (`test_session_resume.py`, `test_session_launch_plan.py`, `test_session_launcher.py`, `test_session_routes.py`, `test_launch_provenance.py`, `test_new_session_launch_request.py`) returned `84 passed, 12 subtests`; full local `pytest -q` returned `1157 passed, 107 subtests`.
- Size observation: after `816082a`, `server.py` is 3561 lines.

## 2026-06-28T18:12:00Z Queue enqueue and attachment coordinator checkpoint
- Functional commits after `0e1ae5e Document web launch coordinator checkpoint`:
  - `c1ee391 Move enqueue orchestration into queue coordinator`: `SessionQueueCoordinator.enqueue` now owns per-session input locking, queueing preconditions for direct unknown sends, pending attachments, recovery queue barriers, sync-send support, append-and-save, immediate promotion attempt for the first item, and queued response shaping. The old manager monkeypatch seam for `_queue_has_recovery_items_locked` is preserved via an injected recovery-barrier callback.
  - `50f555b Extract attachment injection coordinator`: `codoxear/session_attachment.py` now owns attachment injection input locking, readiness gating, request-sent commit-unknown preservation, broker response classification, pending-attachment mutation, and injection-error raising. `SessionManager.inject_attachment_keys` is now a wrapper plus dependency wiring.
- Source sentinel update: `tests/test_file_upload_module_source.py` now checks `session_queue.py` for queue-specific pending/unknown blockers and `session_attachment.py` for attachment response classification/pending-attachment mutation, while keeping broker key-write semantics in `session_control.py` and direct-send protocol semantics in `session_input.py`/`session_send.py`.
- Validation evidence:
  - Enqueue slice: py-compile of `session_queue.py`, `server.py`, and `test_session_queue.py`; focused queue/send/source tests returned `105 passed, 22 subtests`; full local `pytest -q` returned `1157 passed, 107 subtests`.
  - Attachment slice: py-compile of `session_attachment.py`, `server.py`, and file-upload source sentinel; focused server-queue/file-upload/session-control tests returned `99 passed, 22 subtests`; full local `pytest -q` returned `1157 passed, 107 subtests`.
- Size observation: after `50f555b`, `server.py` is 3530 lines. AST method-size scan shows remaining large `SessionManager` methods are the constructor and dependency factories/signature wrappers, not embedded queue/send/control/discovery/list/lifecycle state machines.

## 2026-06-28T18:51:02Z HTTP/dependency/helper extraction tranche
- Functional commits after `4db1bf0 Document queue attachment coordinator checkpoint`:
  - `9812e91 Extract HTTP handler dispatch`: `codoxear/server_handler.py` now owns the `BaseHTTPRequestHandler` subclass, URL-prefix parsing/redirect, JSON body parsing, GET/POST route dispatch order, client-disconnect handling around request/finish, quiet logging, and `server.Handler` compatibility construction.
  - `146cdfd Extract route dependency factory`: `codoxear/server_route_deps.py` now owns route dependency dataclass construction and message runtime snapshot composition. It reads from the live `codoxear.server` module object at call time to preserve monkeypatch-sensitive tests for `_json_response`, `_require_auth`, `_run_git`, and related seams.
  - `bd7a98d Extract server hosting wiring`: `codoxear/server_main.py` now owns threaded HTTP server classes, IPv6 bind behavior, password/app-dir startup, SIGTERM/SIGINT shutdown wiring, and `serve_forever`; `codoxear.server.main` remains the entry-point wrapper.
  - `7ba624b Extract HTTP response primitives`: `codoxear/server_http.py` now owns client-disconnect classification, bad-request/payload-too-large exception classes, route exception mapping, JSON response writing, ETag/If-None-Match handling, and bounded body reads. `codoxear.server` re-exports wrappers/classes for compatibility.
  - `34ec74c Extract process termination runtime`: `codoxear/process_runtime.py` now owns process/group SIGTERM→wait→SIGKILL termination loops, with injected liveness/clock/sleep seams. `codoxear.server` preserves `_terminate_process_group` and `_terminate_process` monkeypatch seams.
  - `b94b400 Extract session resume helpers`: `codoxear/session_resume.py` now owns resume candidate extraction/filtering, first-user-message preview parsing across Codex/Pi/Claude logs, scaffold prompt filtering, and Codex subagent main-thread log coercion. Server wrappers remain for tests and launch code.
  - `b8162c1 Extract shared path runtime helpers`: `codoxear/path_runtime.py` now owns expanduser guarding, session cwd/path resolution, containment resolution, and existing-file checks; `file_routes.py` now shares the same expanduser guard instead of duplicating it.
- Validation evidence by slice:
  - Handler slice: py-compile of `server_handler.py`/`server.py`; focused route/handler suite returned `156 passed, 56 subtests`; full local `pytest -q` returned `1157 passed, 107 subtests` after sentinel updates.
  - Route deps + hosting + HTTP primitives slices each passed focused route/HTTP suites and full local `pytest -q` (`1157 passed, 107 subtests`).
  - Process runtime slice: focused process/lifecycle tests returned `53 passed`; full local `pytest -q` returned `1157 passed, 107 subtests`.
  - Resume helper slice: focused resume/discovery/list/pending-log tests returned `69 passed, 4 subtests`; full local `pytest -q` returned `1157 passed, 107 subtests`.
  - Path runtime slice: focused path/file tests returned `98 passed, 52 subtests`; full local `pytest -q` returned `1157 passed, 107 subtests`.
- Size observation: after `b8162c1`, `server.py` is 2793 lines. Central HTTP dispatch, route dependency construction, server hosting, HTTP response/body primitives, process termination, resume candidate/preview helpers, and shared path helpers no longer live inline in `server.py`.

## 2026-06-28T20:00:39Z Manager factory and under-2000 server tranche
- Functional commits after `b8162c1 Extract shared path runtime helpers`:
  - `fdf5f24 Extract client file path helpers`: `codoxear/client_file_paths.py` now owns unique bare-filename lookup, tracked-file basename fallback, session-relative file listing, symlink payload views, git file-view path resolution, git regular-file resolution, client file path fallback order, and cwd description. Server wrappers preserve the lazy partial-`MANAGER` seam for fake managers without `files_get`.
  - `2d58deb Extract session log metadata helpers`: `codoxear/session_log_metadata.py` now owns backend session-directory selection, log iteration/find-new/find-by-session-id wrappers, inferred metadata backend selection, invalid metadata warning dedupe, turn-context run-setting extraction, and run-setting merge from sidecar/log evidence.
  - `2c3d59a Extract launch defaults cache runtime`: `codoxear/launch_defaults_runtime.py` now owns config path signatures, cached new-session default reads with defensive deep copies, and request-time fallback defaults. `server._LAUNCH_DEFAULTS_CACHE` remains the public monkeypatch seam via injected accessors.
  - `1d7abde Extract launch path helpers`: `codoxear/launch_path_runtime.py` now owns `.env` parsing, `$HOME`/env path expansion, existing/target/new directory resolution, and Codex trust-override TOML text generation.
  - `72df672 Extract session manager store factory`: `codoxear/session_manager_store.py` now owns `SessionStorePaths` construction, store construction, and copy-forward behavior when a manager store must be rebuilt after path changes.
  - `329e4f8 Extract session manager bootstrap`: `codoxear/session_manager_bootstrap.py` now owns initial store-backed in-memory seeding, persistent state load sequencing, voice-push coordinator creation with injected factory, and worker-thread startup.
  - `673d5b1 Move session runtime field helpers`: `codoxear/session_runtime.py` now owns session log-cache reset, sidecar transport normalization, and sidecar/log run-setting merge rules.
  - `1741bb2 Extract server routing helpers`: `codoxear/server_routing.py` now owns URL prefix normalization, stripped-prefix matching, and exact session-route matching.
  - `3fd0c66 Extract server lock and metric helpers`: `codoxear/file_lock_runtime.py` now owns per-path write-lock refcounting; `codoxear/server_metrics.py` now owns route metric recording, bounded windows, percentiles, and snapshots.
  - `67d488a Move tmux availability cache`: `codoxear/tmux_runtime.py` now owns tmux availability cache TTL policy via injected cache accessors/clock/which.
  - `6258b90 Extract session manager coordinator factories` and `408dd5d Move coordinator type imports to factories`: `codoxear/session_manager_factories.py` now owns dependency assembly for queue/control/attachment/list/refresh/readiness/unattended-sweep/queue-sweep/voice/log/files/UI/unattended-config/cleanup/pending/recent/lifecycle/discovery-registry/prune/send/prelog/web-launch coordinators. Factories receive the live `codoxear.server` module to preserve monkeypatch-sensitive wrapper seams while importing coordinator classes directly.
  - `44c10a8 Move discovery dependency assembly`: `session_manager_factories.py` now also owns `DiscoveryDeps` assembly.
  - `18d7597 Collapse store-backed manager properties`: `codoxear/session_manager_store_attrs.py` now owns store-backed property descriptors for `_unattended`, `_aliases`, `_sidebar_meta`, `_hidden_sessions`, `_files`, `_queues`, `_pending_attachment_ids`, `_commit_unknown_sends`, and `_recent_cwds`.
  - `1adb811 Extract manager discovery orchestration`: `codoxear/session_manager_discovery.py` now owns `_discover_existing` interval gating, hidden-session snapshotting, discovery invocation, result application, and last-discovery timestamp update.
  - `27ff240 Extract session cleaner helpers`: `codoxear/session_cleaners.py` now owns alias/recent-cwd/priority/snooze/dependency/optional-text cleaners.
  - `8facb3f Extract manager loop helpers`: `session_manager_bootstrap.py` now owns per-session input-lock allocation and voice/unattended/queue worker loop error/wait policy; `session_manager_discovery.py` now owns `_discover_existing_if_stale` compatibility behavior.
  - `d6a4ce3 Collapse manager store load save methods`: `session_manager_store_attrs.py` now owns generated load/save methods for all store-backed manager state with preserved snapshot rules, including pending-attachment ID string filtering.
  - `1990400 Trim manager wrapper surface`: removed blank gaps and imports no longer needed after factory extraction.
- Validation evidence by slice:
  - Each functional slice above passed a focused suite covering the moved seam, followed by full local `pytest -q` returning `1157 passed, 107 subtests passed`.
  - Focused examples: client-file path extraction `131 passed, 52 subtests`; log-metadata extraction `105 passed, 12 subtests`; launch-cache extraction `46 passed, 12 subtests`; store factory extraction `121 passed, 22 subtests`; coordinator factory extraction `183 passed, 12 subtests`; store property descriptor extraction `114 passed, 22 subtests`; generated load/save extraction `133 passed, 22 subtests`.
- Size observation: after `1990400`, `codoxear/server.py` is 1963 lines. `SessionManager` is now roughly 700 lines and primarily public compatibility methods plus small wrappers; coordinator dependency assembly, bootstrap sequencing, store factory/property/load-save mechanics, discovery orchestration, client file paths, log metadata, launch path/defaults, metrics/locks, routing helpers, and tmux availability no longer live inline in `server.py`.
- Scope note: validation in this tranche is local pytest, not Docker acceptance/promotion evidence.

## 2026-06-28T20:09:00Z Clean-room review gate for manager-factory tranche
- Dedicated reviewer run `667113e7-f577-473e-9094-e950dba9f9e7` completed with verdict `NO BLOCKERS`.
- Reviewer scope: commits `9812e91` through `1990400`, docs commit `6835a7e`, clean working tree, 1963-line `server.py`, all extracted modules, import/circular hazards, manager factory wiring, monkeypatch seams, fail-closed/no-silent-fallback semantics, and docs evidence scope.
- Reviewer observations: no extracted module statically imports `codoxear.server`; `session_manager_factories.py` receives the live server module as a parameter; `Handler.deps.manager()` resolves patched `server.MANAGER`; all modules importable in isolation; full local pytest reproduced as `1157 passed, 107 subtests passed`; docs correctly state this tranche has local pytest evidence only and no Docker promotion claim.
- Non-blocking reviewer notes: coordinator instances are freshly created on each call but frozen/semantically harmless; `_unattended_last_injected*` are runtime state seeded during bootstrap rather than persisted descriptors; `server.py` remains a large compatibility facade but implementation lives in extracted modules.

## 2026-06-28T20:42:34Z Manager compatibility binding tranche
- Functional commits after `a7d7a36 Record manager factory review gate`:
  - `32e7c78 Collapse manager coordinator forwards`: introduced `codoxear/session_manager_method_bindings.py` and moved the repetitive `SessionManager` coordinator-operation compatibility forwards into generated class bindings. Literal high-value public/source-sentinel wrappers for `spawn_web_session`, `send`, and `inject_keys` remained in `server.py`.
  - `756ee7d Bind manager factory methods`: extended the same binding module to attach one-line coordinator-factory methods through late `sys.modules[server_module_name]` lookup, preserving the live `codoxear.server` monkeypatch seam used by route dependencies and factory wiring.
- Validation evidence:
  - Focused test run after operation-forwarder binding: `python3 -m pytest -q tests/test_session_manager_method_bindings.py tests/test_server_queue_persistence.py tests/test_session_file_history.py tests/test_unattended_sweep.py tests/test_session_sidebar_priority.py tests/test_session_routes.py tests/test_control_routes.py tests/test_file_routes.py tests/test_file_upload_module_source.py tests/test_interrupt_semantics_source.py` returned `155 passed, 22 subtests passed`.
  - Full local test run after operation-forwarder binding: `python3 -m pytest -q` returned `1160 passed, 107 subtests passed`.
  - Focused test run after factory-method binding returned `156 passed, 22 subtests passed` for the same focused suite.
  - Full local test run after factory-method binding returned `1161 passed, 107 subtests passed`.
- New regression coverage: `tests/test_session_manager_method_bindings.py` verifies public method-name preservation, delegation when public and target method names differ, negative source ownership for generated manager forwards, preservation of literal `inject_keys` signature, and late live-server-module lookup for generated factory methods.
- Size observation: `codoxear/server.py` is now 1616 lines; `SessionManager` has 26 concrete methods and generated compatibility methods are owned by `session_manager_method_bindings.py`.
- Scope note: this tranche has local pytest evidence only. Docker evidence was not rerun and must not be claimed for promotion.

## 2026-06-28T21:08:23Z Explicit dependency caps and dead-wrapper cleanup
- Functional commits after `d58c594 Document manager binding checkpoint`:
  - `9f510e5 Make manager factory dependencies explicit`: introduced `SessionManagerFactoryCaps` in `codoxear/session_manager_factories.py`; generated manager factory bindings now pass caps built from the live `codoxear.server` module instead of handing factory functions the whole server module. `tests/test_session_manager_factories.py` guards that factory bodies do not regress to `server.*` access after caps construction.
  - `4aeb91b Remove unused server wrapper seams`: removed private module-level wrappers and the no-op `SessionManager.mark_turn_complete` stub that had no source/test references after the caps/binding refactors. Imports/constants used by route dependencies or compatibility source sentinels were intentionally left intact.
  - `289af96 Make route dependencies explicit`: introduced `ServerRouteCaps` in `codoxear/server_route_deps.py`; `_route_deps_factory()` now builds caps from the live server module before constructing `ServerRouteDepsFactory`. `tests/test_server_route_deps_caps.py` guards that route dependency factory methods no longer dereference `server.*` after caps construction.
- Validation evidence:
  - Manager factory caps focused suite: `python3 -m pytest -q tests/test_session_manager_factories.py tests/test_session_manager_method_bindings.py tests/test_server_queue_persistence.py tests/test_session_file_history.py tests/test_unattended_sweep.py tests/test_session_sidebar_priority.py tests/test_session_routes.py tests/test_control_routes.py tests/test_file_routes.py tests/test_file_upload_module_source.py tests/test_interrupt_semantics_source.py` returned `158 passed, 22 subtests passed`.
  - Full local suite after manager factory caps: `python3 -m pytest -q` returned `1163 passed, 107 subtests passed`.
  - Dead-wrapper cleanup focused source/manager/route suite returned `152 passed, 22 subtests passed`; full local suite returned `1163 passed, 107 subtests passed`.
  - First route-caps focused command was invalid because `tests/test_git_routes.py` does not exist; this is a measurement error, not product evidence.
  - Corrected route-caps focused suite returned `62 passed`; a full local suite then exposed one source-sentinel mismatch in `tests/test_file_upload_module_source.py`, still expecting `session_commit_unknown_error=server.SessionCommitUnknownError` inside `server_route_deps.py`.
  - Updated that source sentinel to the new caps boundary (`session_commit_unknown_error=caps.SessionCommitUnknownError`); corrected focused route/source suite returned `63 passed`.
  - Final full local suite after route caps returned `1165 passed, 107 subtests passed`.
- Size observation: after `4aeb91b` and `289af96`, `codoxear/server.py` is 1432 lines. `session_manager_factories.py` and `server_route_deps.py` are larger because they now make previously implicit server-derived dependency surfaces explicit.
- Scope note: this tranche has local pytest evidence only. Docker evidence was not rerun and must not be claimed for promotion.

## 2026-06-28T21:21:30Z Review gate and import cleanup
- Clean-room review results:
  - Reviewer run `4a94e6a4-4a98-46dd-b908-71522845d9d6` returned PASS/no blockers. It verified import/circular safety, generated manager binding public names, live `sys.modules` factory lookup, route caps timing, deleted wrapper reachability, source sentinel update validity, no static `codoxear.server` imports from extracted modules, local pytest evidence, clean working tree, and Docker-scope honesty. Non-blocking notes: two factory functions accept unused `caps` by design, large caps dataclasses are an explicit-surface tradeoff, and one dead queue-store import existed before cleanup.
  - Narrow delegate review `4ce68474-f742-46b5-b19e-78219522c01b` returned PASS/no blockers. It independently verified the latest tranche, reran `python3 -m pytest -q` -> `1165 passed, 107 subtests passed`, and noted residual local-only validation plus remaining potential server import cleanup.
- Follow-up commit `f7297ac Trim unused server imports`: removed exact-dead import aliases from `codoxear/server.py` after wrapper deletion. A precise alias-level scan reported zero exact-dead server import aliases afterward.
- Validation after `f7297ac`:
  - Focused source/caps/queue tests returned `143 passed, 22 subtests passed`.
  - Full local `python3 -m pytest -q` returned `1165 passed, 107 subtests passed`.
- Size observation: `codoxear/server.py` is now 1387 lines.
- Scope note: Docker evidence was not rerun after `f7297ac`; do not claim Docker acceptance/promotion evidence for this tranche.

## 2026-06-28T22:31:00Z Server facade and registry extraction tranche
- Functional commits after `aadf389 Document import cleanup review`:
  - `99c0fcd Extract server configuration bootstrap`: introduced `codoxear/server_config.py` with `ServerConfig`, `.env` application, and config derivation tests. `server.py` now builds `_SERVER_CONFIG` and re-exports legacy config names so patch-sensitive `codoxear.server.NAME` seams remain mutable.
  - `a4b021c Alias pure server facade wrappers`, `c885a87 Alias remaining pure utility facades`, and `d26f2e1 Alias env-file facade wrapper`: replaced pure one-line module-level wrappers with direct aliases to owning modules when no server state/defaults/secrets were bound.
  - `89ded45 Consolidate server config exports`: moved the mechanical config re-export list into `server_config.export_server_config()`, preserving concrete server globals while removing the assignment band from `server.py`.
  - `0488c50 Move manager registry state into explicit owner`: introduced `codoxear/session_registry.py`; `SessionRegistry` owns the manager lock, session mapping, stop event, discovery timestamp, input locks, and store slot. `SessionManager` exposes registry-backed compatibility properties for `_lock`, `_sessions`, `_stop`, `_last_discover_ts`, `_input_locks`, and `_store`; manager bootstrap/discovery/factory/store-attr modules consume the registry rather than raw manager fields. Added `tests/test_session_registry.py`.
  - `56aef5f Bind manager core methods outside server`: introduced `codoxear/session_manager_core_methods.py` and extended `session_manager_method_bindings.py` so non-sentinel `SessionManager` core methods, including `__init__`, bind through late live-server lookup. Literal/source-sentinel methods remain in `server.py`: `refresh_session_meta`, `spawn_web_session`, `send`, `_refresh_session_meta_if_sidecar_exists`, and `inject_keys`.
  - `1f2306d Extract remaining server policy constants`: moved the unattended prompt literal to `unattended.py`, session exception classes to `session_errors.py`, session store path composition to manager core methods, and the broker detach-tail predicate to `session_refresh.py`, with `server.py` preserving public re-export names.
- Validation evidence:
  - Config extraction focused suite returned `238 passed, 74 subtests passed`; full local suite returned `1166 passed, 107 subtests passed`.
  - First wrapper alias focused suite returned `170 passed, 52 subtests passed`; full local suite returned `1166 passed, 107 subtests passed`.
  - Second wrapper alias focused suite returned `138 passed, 52 subtests passed`; full local suite returned `1166 passed, 107 subtests passed`.
  - Config export consolidation focused suite returned `163 passed, 52 subtests passed`; full local suite returned `1167 passed, 107 subtests passed`.
  - Registry extraction focused suite returned `223 passed, 38 subtests passed`; grep for raw `manager._lock`, `manager._sessions`, `manager._stop`, `manager._last_discover_ts`, and `getattr(manager, "_input_locks"` in `codoxear/session_manager_*.py` and `codoxear/server.py` returned no matches; full local suite returned `1170 passed, 107 subtests passed`.
  - Manager-core binding focused suite returned `218 passed, 38 subtests passed`; full local suite returned `1171 passed, 107 subtests passed`.
  - Prompt/error/detach extraction: first focused command included a nonexistent test path and produced no evidence; corrected focused suite returned `184 passed, 34 subtests passed`; full local suite returned `1171 passed, 107 subtests passed`.
- Size/shape observations:
  - `codoxear/server.py` is now 995 lines.
  - `SessionManager` in `server.py` has only five concrete literal methods: `refresh_session_meta`, `spawn_web_session`, `send`, `_refresh_session_meta_if_sidecar_exists`, and `inject_keys`.
  - New ownership modules: `server_config.py`, `session_registry.py`, `session_manager_core_methods.py`, and `session_errors.py`.
- Scope note: validation is local pytest only. Docker evidence was not rerun and must not be claimed for promotion/acceptance for this tranche.

## 2026-06-28T22:36:00Z Registry fallback repair
- Clean-room review in progress surfaced a residual lifecycle factory fallback: `lifecycle_coordinator_for_manager` still used `getattr(manager, "_lock", ...)` and `getattr(manager, "_sessions", {})`.
- Functional commit `2e73e8a Use registry helpers in lifecycle factory` replaced those with `_registry_lock(manager)` and `_registry_sessions(manager)` and removed the now-unused `threading` import from `session_manager_factories.py`.
- Validation after `2e73e8a`:
  - Strict grep for `manager._lock`, `manager._sessions`, `getattr(manager, "_lock"`, `getattr(manager, "_sessions"`, `manager._stop`, `manager._last_discover_ts`, and `getattr(manager, "_input_locks"` in `codoxear/session_manager_*.py` and `codoxear/server.py` returned no matches.
  - Focused registry/factory/manager/resume/queue tests returned `129 passed, 22 subtests passed`.
  - Full local `python3 -m pytest -q` returned `1171 passed, 107 subtests passed`.
- Scope note: Docker evidence was not rerun.

## 2026-06-28T23:13:00Z Voice push runtime module split
- Functional commit `b51db3e Split voice push runtime modules` moved `codoxear/voice_push.py` from a 1461-line mixed runtime/state/delivery module to a 740-line coordinator/facade over explicit ownership modules:
  - `voice_push_state.py`: voice settings/subscription/ledger cleaning helpers, text/hash helpers, default models/voices, and announcement dataclasses.
  - `voice_openai_client.py`: OpenAI-compatible summary and TTS HTTP client, including final/narration prompt wording and response validation.
  - `voice_hls.py`: `MergedHLSStream`, HLS constants, ffmpeg/ffprobe segment handling, keepalive silence, playlist rewrite/reset, and segment containment.
  - `voice_webpush.py`: VAPID private-key/public-key handling, push payload construction, webpush delivery, and explicit delivery outcomes.
  - `voice_persistence.py`: voice settings/subscriptions/delivery-ledger file load/save and private-file chmod repair.
  - `voice_projection.py`: browser-facing settings/subscription/notification feed projections.
  - `voice_task_queue.py`: deterministic announcement voice selection plus narration merge/replacement queue policy.
  - `voice_ledger.py`: delivery-ledger mutation rules for replacement, errors, no-listener skips, field patches, and trimming.
- Compatibility/patch seams preserved:
  - `codoxear.voice_push` still re-exports `AnnouncementTask`, `GeneratedAnnouncement`, `ClassifiedAssistantMessage`, `OpenAICompatibleClient`, `MergedHLSStream`, `DEFAULT_VOICES`, and prior private helper aliases imported from `voice_push_state.py`.
  - `tests/test_voice_push.py` still imports public classes/functions from `codoxear.voice_push`.
  - The existing `patch("codoxear.voice_push.shutil.which")` / `patch("codoxear.voice_push.subprocess.run")` HLS test path still passed; `voice_push.py` keeps `shutil` as a module-level patch seam while HLS implementation lives in `voice_hls.py`.
  - `rollout_log.py` now imports `ClassifiedAssistantMessage` from `voice_push_state.py` instead of heavyweight `voice_push.py`, so log normalization no longer depends on pywebpush/VAPID/HLS imports just to construct delivery message records.
- Source sentinel updates:
  - Voice summary prompt wording now targets `voice_openai_client.py`.
  - HLS keepalive/`anullsrc` source checks now target `voice_hls.py`.
  - voice pool/default checks now target `voice_push_state.py`.
  - settings secret redaction checks now target `voice_projection.py`, private chmod handling targets `voice_push_state.py` plus `voice_persistence.py`/`voice_webpush.py`, and facade-boundary tests assert `voice_push.py` delegates to the extracted runtime modules while no longer importing `pywebpush.webpush` or `py_vapid.Vapid`.
  - `tests/test_rollout_log_helpers_source.py` now pins the lightweight `voice_push_state` import.
- Validation after `b51db3e`:
  - Focused voice/log/idle/source group returned `88 passed`.
  - Full local `python3 -m pytest -q` returned `1173 passed, 107 subtests passed`.
  - `git diff --check` passed after removing one EOF whitespace issue in `voice_hls.py` before commit.
- Scope note: Docker evidence was not rerun and must not be claimed for promotion/acceptance for this tranche.

## 2026-06-28T23:25:00Z Rollout JSONL reader split
- Functional commit `f5a76d2 Extract rollout JSONL readers` moved low-level byte/offset JSONL mechanics out of `codoxear/rollout_log.py` into `codoxear/rollout_jsonl.py`:
  - `JsonlRecord`
  - `_parse_jsonl_line`
  - `_read_jsonl_tail`
  - `_read_jsonl_records_from_offset`
  - `_iter_jsonl_objects_reverse`
  - `_iter_jsonl_records_reverse`
- Compatibility preserved: `rollout_log.py` imports/re-exports those names, so existing callers/tests importing `codoxear.rollout_log._read_jsonl_records_from_offset`, `JsonlRecord`, or `_parse_jsonl_line` continue to work. `transcript_search.py` and `message_routes.py` keep using the `rollout_log` facade unchanged.
- Source sentinel update: `tests/test_rollout_log_helpers_source.py` now asserts JSONL reader primitives are owned by `rollout_jsonl.py` and that `rollout_log.py` imports rather than redefines them.
- Size observation: `codoxear/rollout_log.py` is now 1164 lines; `codoxear/rollout_jsonl.py` is 176 lines.
- Validation after `f5a76d2`:
  - Focused rollout/jsonl/idle/message group returned `121 passed, 4 subtests passed`.
  - Full local `python3 -m pytest -q` returned `1174 passed, 107 subtests passed`.
- Scope note: Docker evidence was not rerun and must not be claimed for promotion/acceptance for this tranche.

## 2026-06-28T23:33:00Z Rollout event identity helper split
- Functional commit `87e0fe4 Extract rollout event helpers` moved pure event-identity/text helpers from `rollout_log.py` into `rollout_events.py`:
  - `_parse_iso8601_to_epoch`
  - `_event_ts`
  - `_strip_oai_mem_citation_tail`
  - `_codex_error_affects_turn_status`
  - `_codex_event_text`
  - `_text_message_id`
- Compatibility preserved: `rollout_log.py` imports/re-exports those helper names and keeps `_with_chat_position` local because positioned tail/live-delta page composition still belongs with chat pagination.
- Negative evidence: the first focused run failed because `_with_chat_position` was accidentally removed during extraction, producing `NameError` in CC positioned tail/live-delta and message route tail tests. Restoring `_with_chat_position` in `rollout_log.py` repaired the failure; this rules out moving/deleting that helper as part of pure event identity extraction.
- Size observation: `rollout_log.py` is now 1103 lines; `rollout_events.py` is 62 lines.
- Validation after `87e0fe4`:
  - Focused rollout/jsonl/idle/message group returned `121 passed, 4 subtests passed` after the `_with_chat_position` repair.
  - Full local `python3 -m pytest -q` returned `1174 passed, 107 subtests passed`.
- Scope note: Docker evidence was not rerun and must not be claimed for promotion/acceptance for this tranche.

## 2026-06-28T23:41:00Z Rollout chat event policy split
- Functional commit `992dbf7 Extract rollout chat event policy` moved single-row chat event interpretation and assistant dedupe policy from `rollout_log.py` into `rollout_chat_events.py`:
  - `_sidebar_conversation_ts`
  - `_update_cc_pending_tool_ids`
  - `_single_chat_event`
  - `_pi_message_keeps_turn_busy`
  - `_cc_message_keeps_turn_busy`
  - `_chat_assistant_dedupe_key`
  - `_dedupe_assistant_chat_events`
- Compatibility preserved: `rollout_log.py` imports/re-exports these names, so existing callers such as `tests/test_server_chat_flags.py` and `transcript_search.py` can still use the `rollout_log` facade. Pagination, live-delta cursoring, token/context scanning, delivery-message extraction, chunk idle analysis, and final idle computations remain in `rollout_log.py`.
- Source sentinel update: `tests/test_rollout_log_helpers_source.py` now asserts single-row chat event policy lives in `rollout_chat_events.py` while `rollout_log.py` imports it.
- Size observation: `rollout_log.py` is now 869 lines; `rollout_chat_events.py` is 281 lines.
- Validation after `992dbf7`:
  - Focused chat/idle/message/source group returned `138 passed, 4 subtests passed`.
  - Full local `python3 -m pytest -q` returned `1175 passed, 107 subtests passed`.
- Scope note: Docker evidence was not rerun and must not be claimed for promotion/acceptance for this tranche.

## 2026-06-28T23:48:00Z Rollout token/context scanner split
- Functional commit `1e5d1a0 Extract rollout token scanners` moved token/context evidence scanning out of `rollout_log.py` into `rollout_tokens.py`:
  - `_extract_token_update`
  - `_find_latest_token_update`
  - `_find_latest_turn_context`
- Compatibility preserved: `rollout_log.py` imports/re-exports these names and still owns live-delta/pagination orchestration that calls them.
- Source sentinel update: `tests/test_rollout_log_helpers_source.py` now asserts token/context scanners live in `rollout_tokens.py` and are imported by `rollout_log.py`.
- Size observation: `rollout_log.py` is now 815 lines; `rollout_tokens.py` is 69 lines.
- Validation after `1e5d1a0`:
  - Focused chat/idle/message/source group returned `151 passed, 4 subtests passed`.
  - Full local `python3 -m pytest -q` returned `1176 passed, 107 subtests passed`.
- Scope note: Docker evidence was not rerun and must not be claimed for promotion/acceptance for this tranche.

## 2026-06-28T23:54:00Z Rollout delivery message split
- Functional commit `9a6950a Extract rollout delivery messages` moved voice-notification delivery-message extraction out of `rollout_log.py` into `rollout_delivery.py`:
  - `_extract_delivery_messages`
- Compatibility preserved: `rollout_log.py` imports/re-exports `_extract_delivery_messages`, so existing imports in tests, server facades, manager factories, and voice runtime remain valid.
- Source sentinel update: `tests/test_rollout_log_helpers_source.py` now asserts delivery extraction lives in `rollout_delivery.py`; the `_extract_chat_events` source slice now ends at `_read_chat_tail_snapshot` because delivery extraction no longer follows it in `rollout_log.py`.
- Size observation: `rollout_log.py` is now 733 lines; `rollout_delivery.py` is 104 lines.
- Validation after `9a6950a`:
  - Focused chat/idle/message/voice/source group returned `145 passed, 4 subtests passed`.
  - Full local `python3 -m pytest -q` returned `1177 passed, 107 subtests passed`.
- Scope note: Docker evidence was not rerun and must not be claimed for promotion/acceptance for this tranche.

## 2026-06-29T00:02:00Z Rollout chat batch analysis split
- Functional commit `94279a2 Extract rollout chat batch analysis` moved multi-row chat extraction/batch analysis out of `rollout_log.py` into `rollout_chat_batch.py`:
  - `_extract_chat_events`
- Ownership after this split: `rollout_chat_events.py` owns single-row backend row interpretation; `rollout_chat_batch.py` owns accumulation of chat events, thinking/tool/system counts, turn start/end/aborted flags, tool-name diagnostics, and CC pending-tool state across a batch. `rollout_log.py` remains pagination/live-delta/snapshot/idle orchestration and imports/re-exports `_extract_chat_events`.
- Compatibility preserved: server facade aliases, message routes, tests, and external callers that import `codoxear.rollout_log._extract_chat_events` continue to use the `rollout_log` facade.
- Source sentinel update: `tests/test_rollout_log_helpers_source.py` now asserts `_extract_chat_events` lives in `rollout_chat_batch.py` and preserves the `_single_chat_event`/`events.append` linkage there.
- Size observation: `rollout_log.py` is now 593 lines; `rollout_chat_batch.py` is 159 lines.
- Validation after `94279a2`:
  - Focused chat/idle/message/voice/source group returned `146 passed, 4 subtests passed`.
  - Full local `python3 -m pytest -q` returned `1178 passed, 107 subtests passed`.
- Scope note: Docker evidence was not rerun and must not be claimed for promotion/acceptance for this tranche.

## 2026-06-29T00:10:00Z Rollout idle analysis split
- Functional commit `cf08730 Extract rollout idle analysis` moved chunk/idle analysis from `rollout_log.py` into `rollout_idle.py`:
  - `_has_assistant_output_text`
  - `_analyze_log_chunk`
  - `_last_conversation_ts_from_tail`
  - `_compute_cc_idle_from_current_turn`
  - `_compute_idle_from_log`
  - `_last_chat_role_ts_from_tail`
- Compatibility preserved: `rollout_log.py` imports/re-exports these names, so server aliases and tests still use the `rollout_log` facade. `rollout_log.py` now primarily owns positioned pagination/live-delta/tail-snapshot orchestration.
- Negative evidence: the first focused idle run failed because `rollout_idle.py` missed `pi_assistant_error_text`; the Pi assistant error-idle test caught it. Adding that import repaired the focused suite, preserving Pi error-row idle semantics.
- Source sentinel update: `tests/test_rollout_log_helpers_source.py` now asserts idle/chunk analysis lives in `rollout_idle.py` and is imported by `rollout_log.py`.
- Size observation: `rollout_log.py` is now 232 lines; `rollout_idle.py` is 390 lines.
- Validation after `cf08730`:
  - Focused chat/idle/message/voice/source group returned `159 passed, 4 subtests passed` after the Pi error dependency repair.
  - Full local `python3 -m pytest -q` returned `1179 passed, 107 subtests passed`.
  - `git diff --check` caught and the commit fixed an EOF whitespace issue in `rollout_log.py`.
- Scope note: Docker evidence was not rerun and must not be claimed for promotion/acceptance for this tranche.
