# Epistemic ledger

## 2026-06-11 23:45
Observations:
- User requested a major refactoring/new-features task, not immediate implementation.
- User explicitly required each workstream to live in a separate branch and forbade merging to `main` without approval.
- User explicitly constrained testing to a standalone Docker instance and forbade touching live sessions or the live server.
- Project instructions emphasize shared broker architecture, minimal UI, GTD-style sidebar without nesting, and deliberate omission of chat details.

Interpretations:
- The safest first artifact is a stable task prompt that future agents can use to create branches and execute the workstreams without violating operational constraints.
- Because the workstreams span architecture, PR review, UI performance, backend support, and regression testing, mixing them would reduce causal attribution and reviewability.

Commitments:
- Treat `PROMPT.md` as the source of task intent until the user changes scope.
- Future implementation should begin by creating isolated branches and a Docker-only validation environment.

Unresolved questions:
- Which GitHub PRs are open and compatible with the project's design philosophy.
- Which historical bugs are still reproducible.
- Whether Claude Code log/session semantics can be normalized through the existing broker/log abstraction without broad architectural changes.
- What current UI/network measurements show under slow mobile-network conditions.

## 2026-06-11 23:52
Observations:
- User corrected the branch invariant: the acceptance artifact should be one `develop` branch, not independent final branches for each workstream.
- User stated the workstreams are not orthogonal, so topology must be chosen from actual dependencies rather than imposed from the numbered list.
- User added that "harness mode" is too vague and needs a more accurate name.
- Code search found UI copy labeled "Harness mode" and server code that periodically injects a prompt only after the session is idle, assistant was last speaker, cooldown elapsed, and injection budget remains.
- The injected prompt prefix is already titled "Unattended-mode instructions".

Interpretations:
- The previous task prompt over-constrained branch structure and would have made integration harder to review; branch topology should now be treated as an implementation decision under a single `develop` acceptance target.
- "Harness mode" likely names an implementation/testing metaphor rather than the user-visible mechanism. The behavior appears closer to unattended continuation or idle follow-up prompting.

Commitments:
- Future work should present `develop` for acceptance and keep any topic branches clearly non-final.
- Future work should rename/recast the harness feature based on the confirmed mechanism, with a deliberate API/state compatibility decision rather than an accidental silent fallback.

## 2026-06-11 23:57
Observations:
- User added that long conversations need better chat-view navigation ergonomics, giving examples: search, jump to previous user message, and time-based navigation.
- User explicitly broadened the working style: be creative and do whatever makes the product better, with no limit except basic ops constraints and product philosophy.

Interpretations:
- The chat view should remain sparse in its primary representation, but may need lightweight orientation/navigation affordances that are not equivalent to showing more low-value transcript detail.
- Future agents should not treat the numbered workstreams as a closed checklist; they should make product-improving interventions when they can state the user benefit, mechanism, validation path, and tradeoff.

Commitments:
- Long-conversation navigation is now an explicit workstream and should be validated against large synthetic or fixture conversations.
- Creative latitude is bounded by standalone Docker testing, no live sessions/server, one `develop` acceptance branch, no `main` merge without approval, and the existing product philosophy.

## 2026-06-12 00:09
Observations:
- Docker smoke test passed with the server bound to host `127.0.0.1:18790`, not the live default port, and with runtime state under container home `/home/tester/.local/share/codoxear`.
- Initial full sandbox test run on `python:3.11-slim` failed at collection because several tests use PEP 701 f-string syntax valid only on Python 3.12+.
- After switching the sandbox to `python:3.13-slim`, test collection succeeded and the baseline was 355 passed, 2 failed, 2 skipped.
- One baseline failure shows delete-session cleanup does not remove cwd-keyed file history for the deleted session; another shows a source-text expectation mismatch in voice summary prompts.

Interpretations:
- The Docker sandbox now provides a valid isolation boundary for API/UI work that does not require live backend credentials.
- The 3.11 failure was a measurement artifact caused by a too-old Python image, not a product regression.
- The remaining two failures are pre-existing baseline problems and should be handled as product/test debt before relying on full-suite green as regression evidence.

Commitments:
- Use the sandbox smoke test as the minimum server isolation check before browser/UI validation.
- Treat the two baseline failures as live issues to fix or explicitly classify, rather than ignoring them during later changes.

## 2026-06-12 00:11
Observations:
- The delete-session failure was caused by legacy `cwd:<cwd>` file-history state surviving session deletion even though session-scoped keys were cleared.
- Existing file-history tests show cwd buckets are intentionally discarded on load rather than migrated, because cwd-based history leaks across sessions sharing the same project directory.
- Voice summarization code had hard maximum wording (`Use at most 15/30 words`) but had lost approximate target-range wording expected by source tests.
- After targeted fixes, the full Docker test suite passed with 357 passed and 2 skipped.

Interpretations:
- Deletion-time removal of the matching legacy cwd bucket repairs stale UI state without reintroducing cross-session file-history leakage.
- Voice prompt target-range wording and hard maximum validation are complementary: the range guides useful summaries; the maximum constrains safety and notification length.

Commitments:
- Treat the Docker suite as green baseline evidence for subsequent product changes.

## 2026-06-12 00:26
Observations:
- Architecture recon identified `server.py` as a 7.5k-line god module, duplicated PTY/process/JSONL helpers, incomplete backend abstraction, competing busy/idle authorities, and expensive synchronous `list_sessions()` work.
- UI recon identified unconditional 2.5s session polling, full sidebar rebuilds, no long-chat search/user-turn navigation, file-picker local/server search latency, and new-session provider/model separation as high-value improvement areas.
- Git-history mining showed repeated regressions in chat scrollback/cursor state, transcript binding, idle detection, queue semantics, rollout-log discovery, shell startup, and file viewer races.
- Unattended-mode analysis showed the current `harness` feature is an idle-triggered prompt injector and recommended user-facing `Unattended mode` with compatibility aliases.
- Claude Code recon outlined a minimal `cc` backend path based on `~/.claude/projects/*.jsonl`, top-level `user`/`assistant`/`system` records, and a new `cc_log.py` parser.
- PR review showed most open PRs contain large stale histories; whole-branch merge would import unrelated changes. Small top commits are more trustworthy units.

Interpretations:
- The next implementation should favor small, high-confidence interventions that reduce risk and improve user-visible behavior before broad architectural extraction.
- Whole PR branch merges would obscure causality and likely violate the one-branch acceptance goal; selective reimplementation/cherry-pick is safer.
- Long-chat navigation and network responsiveness are user-visible wins with lower risk than a large server/module extraction.

Commitments:
- Preserve recon artifacts as evidence.
- Implement accepted PR items selectively and atomically, then validate with the Docker sandbox.
- Defer large Preact/workspace rewrite and whole Claude interactive prompt UI; implement `cc` support through the existing shared broker model.

## 2026-06-12 00:38
- Observation: user explicitly expanded scope to thinking-level/reasoning-effort behavior.
- Interpretation: provider/model controls cannot assume a universal reasoning-effort enum. Codex support may be partial, and Pi capability must be discovered per model/provider.
- Commitment: future UI/API launch semantics should represent unsupported thinking efforts explicitly and avoid silent downgrades.

## 2026-06-12 00:41
- Observation: targeted tests show disconnect-like transport exceptions produce no traceback or JSON 500 attempt, while a RuntimeError still calls traceback and JSON 500 handling.
- Scoped claim: Codoxear route handlers now treat common browser/client disconnects as transport noise at route and request-boundary layers; this does not prove every OS/socket close variant is covered.

## 2026-06-12 00:43
- Observation: `_discover_existing()` previously raised on a `.sock` without adjacent `.json`, which could make session listing fail because a runtime artifact was stale.
- Interpretation: broker metadata is the source of truth for session identity/cwd/log binding; a socket without metadata cannot be safely represented as a live Codoxear session.
- Scoped claim: missing sidecars now prune stale socket/session state and clear associated UI/session-local state; invalid sidecar JSON still fails loudly as a corrupted metadata contract.

## 2026-06-12 00:47
- Observation: fetched PR #12/#15 diffs showed the intended Pi invariant: an explicit `--session` log path should be remembered even before the file exists, and discovery should register it once it appears.
- Interpretation: the declared Pi session path is stronger evidence than cwd/process fallback because Codoxear itself injects `--session`; failing to preserve it can leave the web UI pending or bound to the wrong Pi log.
- Scoped claim: Pi broker metadata now preserves the reserved log path and watcher registration favors it when it exists; this was validated with synthetic broker tests, not a live Pi CLI run.

## 2026-06-12 00:50
- Observation: previous session maintenance used a fixed 2.5s interval regardless of page visibility and bundled session refresh, voice settings, notification state, and notification feed polling.
- Intervention prediction: hidden-tab network traffic from session maintenance should drop by roughly 6x (15s vs 2.5s) while visible behavior remains at the prior cadence and returns refresh immediately.
- Scoped claim: source and parse/full-suite validation constrain the scheduling implementation; no browser network trace has yet measured the actual request-rate reduction.

## 2026-06-12 00:52
- Observation: current chat DOM already stores message role on `.msg-row`, so local navigation can target loaded user turns without changing backend message APIs.
- Intervention prediction: previous/next user-message jumps improve long-chat orientation while preserving sparse chat semantics because they add only two icon affordances and no dense transcript index.
- Scoped claim: navigation is over loaded/rendered rows only; it intentionally does not claim full-history search or jump across unloaded scrollback.

## 2026-06-12 00:55
- Observation: loaded-message search can reuse rendered `.md` text without backend indexing or transcript-detail expansion.
- Intervention prediction: a floating loaded-search bar improves orientation in long loaded chats while keeping the main transcript sparse and honest about scope.
- Scoped claim: search currently covers rendered/loaded rows only; full-history server-side search remains deferred until evidence shows loaded search is insufficient.

## 2026-06-12 01:03
- Observation: user explicitly rejected compatibility for the Harness→Unattended rename.
- Revision: previous compatibility-alias plan was over-conservative. The public contract should be clean Unattended naming rather than dual routes/fields.
- Scoped claim: public UI/API/session-list/env/state-file surfaces now use Unattended naming without `/harness` or `harness_*` aliases; internal implementation identifiers still contain harness names and can be cleaned later if it remains reviewable.

## 2026-06-12 01:07
- Observation: local Pi `models.json` model rows include a `reasoning` boolean, and the code previously exposed one Pi-wide effort list regardless of model.
- Mechanism: for Pi models that declare `reasoning:false`, only `off` should be selectable/accepted; explicit per-model effort lists should constrain both UI and API validation.
- Scoped claim: Pi thinking effort selection is now model-aware for metadata available in `models.json`; Codex remains constrained to its existing supported enum and still needs deeper current-CLI capability work before claiming model-specific Codex support.

## 2026-06-12 01:30
- Observation: local Claude Code CLI advertises the exact launch flags needed for minimal shared-broker support, and current logs use top-level `user`/`assistant`/`system` records rather than Codex `event_msg`/`response_item` or Pi `message` records.
- Mechanism: CC can share Codoxear's PTY broker and proc/lsof log discovery if log validation excludes `subagents/` logs and metadata is derived from the first record carrying `sessionId`/`cwd`.
- Prediction tested: CC user records start a turn; assistant text with `stop_reason=end_turn` closes it; assistant `tool_use`/`thinking` and user `tool_result` records keep it busy. Unit tests and idle/chat extraction tests match this prediction.
- Scoped claim: Codoxear now has test-covered minimal CC backend plumbing and UI launch support. This is not yet live-session proof against an actual long-running Claude session; residual risk remains around undocumented Claude log-format drift and interactive TUI quirks.

## 2026-06-12 01:33
- Anomaly caught before finalization: CC backend inference was path-literal for `~/.claude/projects` and did not respect the configured `CLAUDE_CONFIG_DIR` source of truth.
- Revised claim: CC log inference now works for default Claude config paths and for custom `CLAUDE_CONFIG_DIR` paths; unrelated Pi custom-home inference remains unchanged from prior behavior.

## 2026-06-12 01:39
- Observation: `visibleFilePickerEntries()` returned `null` for pending/unloaded searches, which forced the UI to show only `Searching files...` despite having local session candidates available for immediate fuzzy scoring.
- Mechanism: local candidates are enough to provide useful first results while server search broadens to the full project; a footer preserves scope honesty and avoids a silent fallback.
- Scoped claim: file picker search now reduces perceived latency for known/recent/changed files during server-search debounce/network delay; it does not replace full-project search or add fuzzy-match highlighting.

## 2026-06-12 01:42
- Observation: the new-session model menu already used recent sessions but collapsed them to model strings, losing provider information and forcing repeat launches to use both provider and model controls.
- Mechanism: carrying provider metadata on recent model options lets one existing combobox action restore a provider/model pair without adding another visible control or nested picker.
- Scoped claim: repeat launches for recent provider/model pairs now need fewer interactions; this does not implement a full provider/model selector or provider-specific model catalog.

## 2026-06-12 01:44
- Observation: git-history mining identified missing deterministic coverage for Unattended thread dedup/counter boundaries and JSONL partial-append handling.
- Interpretation: these are cheap regression tests with high evidence value because they constrain prior failure mechanisms without requiring live broker/CLI processes.
- Scoped claim: the added tests pressure the deterministic mechanisms only; live shell startup, browser/Monaco integration, and real backend lifecycle pressure tests remain outside this tests-only commit.

## 2026-06-12 01:46
- Observation: isolated browser validation loaded the real UI, authenticated successfully, exposed the new long-chat/Unattended controls, and rendered the Claude backend tab.
- Observation: in Claude mode, provider and Fast controls were hidden while reasoning effort displayed `medium`, matching the intended backend capability contract.
- Scoped claim: this validates UI wiring/rendering in a real browser for the changed controls, but not actual backend CLI session creation or long transcript interaction.

## 2026-06-12 01:51
- Clean-room review found no blocker to yielding the `develop` acceptance candidate under the stated constraints.
- Residual uncertainty is scoped to live-like backend/browser pressure tests that were not run because the task constraints forbid touching live sessions and further sandbox-realistic backend tests require user-authorized credentials/binaries/time.

## 2026-06-12 01:59
- Observation: Codex log discovery requires `session_meta`; a synthetic log without it fails loudly instead of silently binding. This is consistent with the no-silent-fallback contract and was contained to the isolated sandbox.
- Observation: in a real browser against an isolated server, the long-chat UI loaded a recent tail window from a 320-message synthetic transcript, found a loaded search marker exactly once, navigated among loaded user turns, and loaded older history back to the beginning.
- Scoped claim: long-chat search/navigation/history loading has browser-level evidence for synthetic Codex logs and rendered loaded rows. This still does not prove performance on a real mobile device, real Monaco/file-viewer races, or live backend CLI lifecycle behavior.

## 2026-06-12 02:00
- Follow-up interpretation: the synthetic long-chat rows used `phase:"final_answer"` without `end_turn:true`, so browser busy/Interrupt state from that run should not be interpreted as evidence about idle status. Existing `tests/test_idle_heuristics.py::test_response_item_end_turn_is_idle` constrains the valid Codex idle shape with `end_turn:true`.

## 2026-06-12 02:01
- Clean-room review after additional long-chat browser validation found no blocker to yielding the `develop` candidate.
- Remaining uncertainty is no longer about unrun deterministic tests; it is about live-like backend lifecycle and device/performance conditions that require user authorization or broader sandbox setup.

## 2026-06-12 02:09
- Observation: no implementation/docs source outside tests now contains Harness naming; the remaining Harness strings are deliberate negative source assertions.
- Interpretation: the public rename is now backed by internal naming consistency, reducing future maintenance risk where implementation names could reintroduce old public compatibility or confuse the mechanism.
- Browser observation: the renamed Unattended menu DOM/API path works in a real browser against isolated state, and the renamed sweep decremented the synthetic injection budget after enabling an idle session.
- Scoped claim: this is a semantic cleanup with preserved behavior under tests and browser smoke; it does not add new live-backend lifecycle evidence.

## 2026-06-12 02:10
- Prompt-memory correction: the prior Workbench status was stale and would mislead a future agent into repeating completed setup/recon work. Updating it preserves the current epistemic boundary: deterministic and browser-synthetic evidence is complete enough for `develop`; live-like backend/device evidence remains a user-authorized extension.

## 2026-06-12 02:12
- Observation: local Codex CLI help does not provide model-specific reasoning-effort capability metadata. The current Codex implementation can validate the known effort enum and pass config overrides, but cannot honestly claim Pi-style per-model capability enforcement.
- Clean-room final gate found no blocker under the user's constraints. Remaining uncertainty is scoped to user-authorized live-like backend/device/performance checks and Codex capability metadata that was not available from the inspected source.

## 2026-06-12 02:17
- Observation: recon artifacts preserved useful historical evidence but could mislead future review because some pre-implementation Harness compatibility recommendations were superseded by the user's no-compatibility correction.
- Intervention: add a final acceptance summary and explicit status notes instead of deleting historical recon. This preserves evidence while clarifying the current claim boundary for `develop`.

## 2026-06-12 02:19
- Clean-room final gate after acceptance-summary documentation found no blocker. The remaining uncertainty is user-decision-bound: authorize broader live-like sandbox/device validation or accept `develop` as-is.

## 2026-06-12 02:25
- Observation: git-history mining identified assistant-message duplication as a medium-risk area, while delivery notifications already had adjacent duplicate suppression. Chat page/live extraction lacked equivalent direct coverage.
- Mechanism: duplicate adjacent assistant rows in a single assistant stretch are more likely log/read artifacts than semantically distinct user-visible turns; resetting on user messages preserves repeated answers in separate turns.
- Scoped claim: Codoxear now constrains duplicate assistant chat events within batch/page extraction. This does not prove cross-poll duplicate suppression if a duplicate arrives in a later live delta after the previous batch has already rendered.

## 2026-06-12 02:27
- Clean-room review after closing the deterministic assistant-dedupe gap found no blocker. The remaining uncertainties are either broader live-like validation or explicitly scoped behavior beyond the implemented batch/page dedupe.

## 2026-06-12 02:35
- Observation: the client previously deduped exact event keys only, so duplicate assistant text with a different timestamp could still render if it arrived in a later live poll after the previous assistant row had already rendered.
- Mechanism: storing a normalized assistant dedupe key on rendered assistant rows lets `appendEvent()` compare the incoming assistant event to the actual rendered tail. Because the rendered tail changes to a user row after a user message, repeated assistant text in a later turn remains visible.
- Revised claim: assistant duplicate suppression now covers both server batch/page extraction and the client live-append cross-poll path. Remaining duplicate uncertainty is limited to more complex non-adjacent/streaming patterns not represented by the adjacent duplicate mechanism.

## 2026-06-12 02:37
- Clean-room review after the client cross-poll dedupe patch found no blocker. The remaining duplicate risk is scoped to patterns that are not adjacent assistant repeats, while the original cross-poll adjacent case now has a mechanism and regression test.

## 2026-06-12 02:44
- Observation: package-data coverage did not explicitly include the newly added Claude Code logo, even though the runtime UI computes backend logo paths generically as `static/logos/<backend>.svg`.
- Mechanism: asserting `codoxear/static/logos/cc.svg` inside the built wheel protects installed deployments from a source-vs-wheel asset mismatch for the Claude backend. This reduces packaging uncertainty without changing runtime behavior.

## 2026-06-12 02:45
- Observation: the Docker sandbox helper's implementation and usage text diverged for the `build` command.
- Mechanism: source regression ties the documented command list to the dispatch cases, reducing validation-tool drift. This does not change product behavior but improves the reliability of the evidence-producing toolchain.

## 2026-06-12 02:48
- Clean-room adversarial review after latest continuation found no remaining deterministic non-user-blocked gaps. The support for yielding is stronger because the review specifically checked whether the last changes introduced an acceptance blocker and found none.

## 2026-06-12 02:53
- Observation: full tests and browser checks did not explicitly prove the packaging/editable-install criterion or a current isolated server-start smoke.
- Negative observation: a first post-install script-location check failed because the sandbox installed console scripts in the user base outside `PATH`; direct inspection of `/home/tester/.local/bin` corrected the measurement.
- Scoped claim: the current `develop` branch can be installed editably from a writable source copy in the sandbox image, exposes its server/broker console scripts in the expected user-install location, and starts an isolated password-gated server without touching live app state.

## 2026-06-12 02:56
- Clean-room review evidence is mixed operationally but not substantively: a broad reviewer timed out, while a narrower fresh gate found no blocker or deterministic pre-yield action. The remaining uncertainty remains user-decision-bound or explicitly scoped.

## 2026-06-12 11:32
- Observation: prior summaries treated useful partial work as an acceptance candidate. User review exposed that several nontrivial feature requests were only touched or overclaimed, especially provider/model selection and UI cleanliness.
- Revised commitment: do not use structural refactor as a substitute for missing product behavior. Fix real gaps first; resume refactor only after the feature task is product-complete or explicitly scoped by the user.

## 2026-06-12 11:40
- Revised model: implementation mechanisms and green tests are insufficient acceptance objects. The live claims must be product promises about workflows under invariants, supported by scoped evidence.
- Prediction for recovery: if this ontology is enforced, provider/model and top-bar/action-placement work will be treated as central contract failures, not polish or optional refinements.

## 2026-06-12 11:49
- Observation: current source still had separate Provider and Model controls; topbar contained file, copy, search, user-jump, details, interrupt, and Unattended controls.
- Intervention: make provider/model a single workflow object and split action placement by user workflow: session utilities bar for files/copy/details/Unattended, chat navigation rail for loaded-chat search/user jumps, topbar only for identity/sidebar and interrupt.
- Evidence: source/runtime tests now assert no provider-only new-session button/menu remains, configured/recent provider/model pairs are offered by the combobox, typed provider/model filters work, and topbar no longer contains session utilities or chat navigation controls.
- Scoped claim: deterministic source/runtime evidence supports the new workflow contract at code level. Browser validation is still required for actual visual/mobile ergonomics and event behavior.
