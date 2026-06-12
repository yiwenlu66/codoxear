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
