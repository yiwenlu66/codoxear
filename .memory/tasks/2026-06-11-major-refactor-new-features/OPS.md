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
