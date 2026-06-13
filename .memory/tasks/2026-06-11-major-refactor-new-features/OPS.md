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
