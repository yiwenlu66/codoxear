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

## 2026-06-12 12:04
- Observation: browser evidence found two issues missed by source tests: stale provider/model error text across backend switches, and missing modal isolation in the clean recovery branch.
- Mechanism: provider/model validation state lived in the new-session status text and was only set on failed start, not cleared by backend changes; modal overlays were visual-only siblings and did not mark the background app inert/hidden.
- Interventions: added provider/model error clearing on valid input/backend reset; added a shared modal isolation boundary and closed transient overlays before opening custom/native modals.
- Evidence: browser rechecks showed providerless Pi had no stale error and modal-open accessibility snapshots exposed only modal controls; Docker suite passed after changes.
- Scoped claim: combined provider/model selection, sparse/contextual action placement, loaded-chat rail navigation, and modal isolation are supported for the isolated synthetic desktop/mobile workflows. This does not yet prove real-device performance or live backend startup behavior.

## 2026-06-12 12:07
- Observation: file/context workflow is reachable from the new session utilities rail, preserves modal isolation, and can search/open README.md in the isolated repo. Initial one-second read observation was still `Loading...`; after waiting, content and status appeared, so the issue was latency/async completion rather than a stuck viewer.
- Observation: bounded responsiveness measurements in the synthetic long-chat browser session were small for the tested rendered window: tail API resource about 5 ms, loaded search about 23 ms, user jump about 33 ms.
- Scoped claim: these measurements support that the redesigned contextual controls are usable in the isolated synthetic desktop browser and do not obviously regress loaded-window search/jump latency. They do not prove real mobile device performance, slow network behavior, or full unbounded transcript scalability.

## 2026-06-12 12:12
- Observation: backend refactor seams were low-risk structurally but still caused a real integration regression when the auth extraction was applied without its dependent fix. This confirms the user criticism: refactor progress is not proof unless the integrated workflow is revalidated.
- Mechanism: `_is_same_password()` remained in `server.py` and still called `hmac.compare_digest`; extracting auth helpers removed the module import until the later fix restored it.
- Revised claim: the recovered branch now includes product fixes plus selected backend refactors, but the claim is integrated only because login, source tests, targeted Docker tests, full Docker tests, and a restarted browser sandbox all ran after the refactors and auth fix.
- Scoped uncertainty: frontend modularization refactor remains parked; real backend launches, real mobile-device performance, and slow-network behavior remain outside current evidence.

## 2026-06-12 12:24
- Observation: the combined provider/model selector was real, but reopening New Session still derived its model from backend defaults plus remembered provider. That left a workflow gap: the app remembered `chatgpt`, not `chatgpt/gpt-5.4-mini`.
- Mechanism: previous persistence key stored only provider choice. Recent sessions could suggest pairs, but the user's explicit selected pair was not the remembered launch default.
- Intervention: added a separate per-backend provider/model memory key, wrote it on valid menu selection and valid start attempts, and restored it through the same provider/model parser so stale provider names are ignored loudly/safely.
- Evidence: browser showed the selected pair persisted and restored exactly; deterministic frontend tests and full Docker passed.

## 2026-06-12 12:29
- Observation: moving chat navigation out of the topbar was not sufficient; the initial rail placement was still an overlay and geometrically covered message rows. This was a product invariant failure for sparse/readable chat, not cosmetic polish.
- Mechanism: `#chatNavRail` was absolutely positioned over the chat scroll viewport. On scroll positions near the tail, visible rows could pass underneath the control cluster.
- Intervention/evidence: placing the rail in normal flex layout above the scroll area removed visible overlap while keeping navigation contextual to chat. Browser geometry on desktop and mobile found no strict visible overlap after the change.

## 2026-06-12 12:31
- Evidence update: final deterministic validation still passes after the latest UX/memory polish. No new local deterministic failure is known before clean-room review.
- Remaining uncertainty remains scoped to live-like backend launches, real mobile/slow-network performance, full real transcripts, and Codex authoritative per-model reasoning semantics.

## 2026-06-12 12:37
- Observation: after the rail was fixed, the search bar still used absolute positioning and covered message content while search was open. This preserved a weaker version of the same readability failure.
- Mechanism: `#chatSearchBar` was removed from layout flow, so chat rows continued under it. It did not overlap the rail, but it did overlap visible rows.
- Intervention/evidence: placing search in the same flex flow as the rail created explicit vertical space for both controls. Browser geometry showed no visible row overlap and retained search function (`1/1 loaded`).

## 2026-06-12 12:50
- Observation: fresh review found a stale runtime call that static selector tests had missed. Browser validation confirmed no errors after replacing it, but full Docker later found a stale invariant test. Mechanism: the provider-only menu was removed from the UI but not from every refresh/test path; tests still encoded the old internal provider-menu stage.
- Observation: `providerChoiceToSettings()` used a Codex default before branching by backend. Mechanism: empty provider choices for providerless Pi launches were converted into `chatgpt`, leaking Codex semantics into Pi. Browser POST interception after the fix showed no `model_provider` for providerless Pi.
- Observation: backend config readers could raise from malformed local files while composing `/api/sessions`. Mechanism: launch-default discovery was coupled to the main session list response. Intervention isolated each backend reader behind safe defaults plus warnings, preserving existing session visibility/control when launch defaults are degraded.
- Confidence update: core New Session provider/model workflow is stronger after runtime, HTTP, and full Docker evidence. Remaining larger risks are scoped to all-transcript search, route exactness, list_sessions lock-held IO, and broader launch config normalization; these are not falsified by the current tests.

## 2026-06-12 12:56
- Observation: session routes still had many prefix+suffix checks that could accept unintended aliases with extra path segments. This was not a user-visible bug in normal UI paths, but it weakened the API invariant that each endpoint has one documented shape.
- Mechanism: handlers extracted `parts[3]` as the session id after only checking path prefix and suffix, so paths such as `/api/sessions/s1/extra/file/read` could be interpreted as session `s1` file-read requests.
- Intervention/evidence: all session route families now use `_match_session_route()`, and the old suffix-match pattern is absent from `server.py`. Tests reject extra segments while preserving intended route shapes. Full Docker passing scopes the claim to existing unit/source/runtime coverage, not external clients using undocumented aliases.

## 2026-06-12 13:02
- Observation/interpretation: loaded-only search was honest but weak for long unattended sessions because relevant text could exist outside the rendered DOM window. A full cursor-jump search would require careful byte-boundary paging semantics; a lower-risk intervention is to surface all-transcript match counts using the export pipeline while leaving loaded-row navigation unchanged.
- Scoped claim: the UI can now tell users when matches exist beyond loaded rows (`loaded` count plus `all` count). It does not yet automatically load/jump to an older all-transcript match; that remains a possible future enhancement requiring cursor-target validation.

## 2026-06-12 13:04
- Observation: `list_sessions()` still had external IO inside the manager lock. A narrow, low-risk mechanism was identified for git branch lookup: it depends only on a resolved cwd snapshot, not mutable manager state.
- Intervention/evidence: git branch lookup now runs after releasing the lock. A regression test asserts `_current_git_branch` observes `mgr._lock.locked() == False`. This reduces one lock-held IO source; log-derived run settings and first-history scans remain known future lock-scope risks.

## 2026-06-12 13:18
- Observation: all-transcript search counts reduced uncertainty but still left the user to manually page older history when `all > loaded`. Direct byte-cursor jumping risked creating gaps in the loaded transcript because the existing UI assumes older pages are prepended contiguously.
- Intervention/evidence: search Next now uses bounded contiguous older-page loading through the existing history endpoint, refreshing loaded search after each page and stopping at the first loaded match or after 12 pages. This improves long-session search without introducing a second transcript ordering model.

## 2026-06-12 13:21
- Observation: app-dir JSON state files repeated absent-file loading, parent creation, temp write, and atomic replace mechanics. The useful invariant is shared IO semantics while keeping each store's owner-specific schema sanitizer.
- Intervention/evidence: shared helpers now own JSON file IO and atomic replace; migrated stores still perform their original validation/cleaning. Targeted persistence tests and full Docker constrain regressions for aliases/sidebar/hidden sessions/files/queues/recent cwd/unattended state.

## 2026-06-12 13:23
- Evidence update: the shared JSON state IO invariant now covers voice push settings/subscriptions/ledger in addition to server UI state and unattended state. Schema semantics remain owned by each store's cleaner; the common helper owns parent creation and atomic replacement.

## 2026-06-12 13:25
- Evidence update: another `list_sessions()` lock-held IO source was removed. Log run-settings scans now happen outside the manager lock, with a guarded re-lock to mutate the session only if it is still current. First-history timestamp recovery remains a smaller known lock-held log-read risk.

## 2026-06-12 13:27
- Evidence update: first-history timestamp recovery is no longer lock-held IO. Because the scan can affect recency, priority, and recent-cwd state, the refactor recomputes those row fields after the guarded update. `list_sessions()` still performs some filesystem existence checks under lock, but the larger log scans and git subprocess lookup have been moved out.

## 2026-06-12 13:34
- Observation: launch request semantics were spread through the HTTP handler. This increased drift risk across Codex/Pi/Claude provider/model/reasoning validation.
- Intervention/evidence: a normalized launch request parser now owns backend-specific validation while the route preserves response/spawn behavior. Tests cover Codex custom provider, providerless Pi, Claude field rejection, cwd field errors, and Pi model-specific reasoning coupling.

## 2026-06-12 13:36
- Browser evidence update: the all-transcript search paging mechanism was tested through normal session discovery, message tail/history routes, and a real broker control socket in the isolated Docker app dir. The observed transition from `0/0 loaded · 1 all` to `1/1 loaded · 1 all` supports the mechanism that bounded contiguous history paging can materialize older search matches without creating a separate jump/page model.

## 2026-06-12 13:38
- Observation: after extracting launch parsing, GET launch defaults could degrade safely while POST launch validation still depended on raw backend config readers. This could make the UI truthfully say safe defaults are in use but still fail to start a safe-default session.
- Intervention/evidence: request parsing now uses fallback defaults for provider validation when backend config readers fail. Parser tests simulate malformed Codex/Pi config readers and still parse safe launch requests.

## 2026-06-12 13:47
- Observation: fresh review showed launch-default semantics were still inconsistent for Pi when the request included `reasoning_effort`: provider validation used fallback defaults, but reasoning validation called the raw Pi model capability reader. A malformed Pi models file could therefore make GET `/api/sessions` degrade while POST `/api/sessions` failed.
- Intervention/evidence: request parsing now captures one safe Pi launch-default snapshot and passes its `reasoning_efforts_by_model` into Pi reasoning validation. Regression test patches the raw Pi reasoning capability reader to raise and confirms a fallback-supported Pi request with `reasoning_effort: high` still parses.
- Scoped claim: safe-default consistency now covers Codex provider validation, Pi provider validation, and Pi reasoning-effort validation in the launch request parser. It does not validate real credential-backed backend startup.

## 2026-06-12 14:04
- Observation: product review correctly identified that count-only all-transcript search was still not fully actionable when some matches were already loaded. Browser reproduction showed `1/1 loaded · 3 all` could page to the first older hit, but repeated boundary paging initially wrapped from `2/2 loaded · 3 all` back to the first loaded hit.
- Mechanism: search-driven history loads were aborted by the scroll-cancel invariant intended for user/auto scrollback loading. The current search hit is usually far below the top, so the scroll handler saw `loadingOlder && scrollTop > OLDER_CANCEL_PX` and aborted the request.
- Intervention/evidence: older loads now default to scroll-cancellable, but search paging passes `cancelOnScroll: false`. Browser evidence shows the third all-transcript match is loaded and focused after crossing the loaded-match boundary.
- Scoped claim: all-transcript search counts are now actionable across loaded-match boundaries in the validated synthetic long Codex transcript. This does not prove performance on very large real logs or slow networks.

## 2026-06-12 14:10
- Observation: `list_sessions()` had a dead `recent_cwd_dirty = True` assignment and active-session recent cwd updates were memory-only. This did not affect immediate sidebar rows but could lose recent cwd learning across server restarts.
- Intervention/evidence: `list_sessions()` now tracks recent-cwd dirtiness for both active-session and history-backfill updates and persists after lock-held row construction. Regression test confirms a new active cwd triggers exactly one save across repeated list calls.

## 2026-06-12 14:12
- Observation: backward all-transcript search paging follows the same mechanism as forward boundary paging under browser validation: Prev at the first loaded hit loads older history without scroll-cancel abort and focuses the older match. This strengthens the scoped claim from forward-only to both boundary directions for the synthetic long transcript.

## 2026-06-12 14:17
- Observation: prior mobile CSS explicitly hid `.toast` at <=520px. Because send/queue/copy/search/file operations report transient status through `setToast()`, mobile users could lose feedback for successful or failed actions.
- Intervention/evidence: mobile toast now uses a sparse snackbar style that remains absent when empty via existing `.toast:empty`, but visible when populated. Browser measurement at 390px width showed it does not overlap the composer and remains pointer-transparent.

## 2026-06-12 14:19
- Observation: duplicate JSONL tail wrappers mostly already shared the safe util reader, but sessiond still differed from broker on missing files. A disappearing pending/session log could terminate the sessiond watcher rather than preserving the current offset and retrying.
- Intervention/evidence: sessiond now mirrors broker's missing-file behavior. Regression test covers the contract directly.

## 2026-06-12 14:22
- Observation: `_pid_alive` and `_process_group_alive` were duplicated across server/broker/sessiond. Broker's PID helper also handled unexpected `os.kill` failures more defensively than the server copy.
- Intervention/evidence: util now owns `pid_alive` and `process_group_alive`; server/broker/sessiond import them as local private names so call-site semantics stay stable. A source guard asserts the definitions remain centralized.

## 2026-06-12 14:25
- Observation: broker and sessiond duplicated PTY full-write and bracketed-paste injection logic. This was a low-risk extraction because tests already exercise partial writes and send acknowledgements.
- Intervention/evidence: shared helpers now live in `pty_util`; module-local wrappers preserve call-site and patching structure for control-flow tests. A source guard asserts bracketed-paste constants remain centralized.

## 2026-06-12 14:30
- Observation: path comparison and rollout filename session-id extraction still had duplicate implementations after the larger helper cleanups.
- Intervention/evidence: broker now imports util's path matcher, and server/broker share util's rollout session-id extractor. Source guard asserts a single definition for each helper family.

## 2026-06-12 14:32
- Observation: long-chat search existed but required pointer discovery. A global Ctrl/Cmd+F override would conflict with browser semantics, so it was rejected.
- Intervention/evidence: added an app-specific `/` shortcut with explicit text-entry and modal guards. Browser evidence confirms it opens search from chat context and does not steal typing from the composer.

## 2026-06-12 14:38
- Observation: the broad duplicate chat-extraction refactor is still riskier than warranted, but the timestamp and message-id helper duplication was mechanical and directly testable.
- Intervention/evidence: duplicate local helper definitions were removed; a source guard now asserts only one module-level timestamp and message-id helper remains for rollout chat extraction.

## 2026-06-12 14:46
- Review observation: extended clean-room review found no blockers at `01cbad5`, and independently confirmed recent-cwd persistence, mobile toast accessibility, sessiond fail-closed JSONL reads, helper refactors, and shortcut guards.
- Nonblocking anomaly promoted to fix: malformed `messages/search?limit=...` was parsed with raw `int()`; reviewer predicted a manual/API malformed limit could become a 500 despite UI always sending numeric limits. Runtime route test confirmed the desired revised behavior after intervention: `limit=not-an-int` returns 400 with an explicit integer error.

## 2026-06-12 14:51
- Mechanism generalized: the malformed-search-limit bug was not isolated to search; tail/history had the same raw `int(limit)` parsing pattern. A shared bounded parser reduces future divergence and makes malformed numeric query behavior explicit across message routes.

## 2026-06-12 14:58
- Anomaly: isolated browser fixture setup produced repeated background sweep errors from a Codex log lacking `session_meta`. Mechanism: discovery/refresh treated log-embedded Codex session metadata as mandatory even when the broker sidecar already held session identity/log binding.
- Revised commitment: sidecar-bound sessions should fail closed on malformed log metadata by preserving sidecar identity/log path and emitting diagnostics, not by crashing discovery/listing.

## 2026-06-12 15:05
- Observation: real-browser validation now discriminates two mechanisms. The `/` shortcut guard works under the mobile sidebar overlay (search remains closed), and sidecar-bound malformed Codex logs no longer take down discovery/listing. Remaining uncertainty: this fixture simulates a broker control socket and does not validate real Codex/Pi/Claude binaries or credentials.

## 2026-06-12 15:16
- Mechanism: prior visibility-aware session polling still fetched low-priority voice/notification state every session tick. Decoupling those requests should reduce foreground/background network work while preserving immediate session-list freshness. Remaining uncertainty until browser/network validation: actual request cadence under a live page.

## 2026-06-12 15:18
- Evidence update: live browser request counts support the intended polling mechanism: core session polling continued at visible cadence, while secondary voice/notification state stayed off the fast session loop for the measured window. This constrains request cadence, not slow-network latency or real mobile-radio power use.

## 2026-06-12 15:21
- Mechanism: loaded user-turn buttons existed, but keyboard-only navigation still required focus/clicking a rail control. Guarded `Alt+↑/↓` shortcuts improve long-conversation orientation without adding visible UI density and without stealing input from composer/modals/sidebar.

## 2026-06-12 15:24
- Evidence update: browser validation supports the shortcut guard mechanism for loaded user-turn navigation. Initial CDP attempt without the Alt modifier bit produced no pulse, distinguishing test harness setup from product behavior; rerun with Chromium `modifiers:1` exercised the intended path.

## 2026-06-12 15:37
- Evidence update: conditional sessions polling works in a live browser against the isolated server. The observation constrains unchanged-payload transfer behavior for `/api/sessions`; it does not prove performance on real slow mobile networks, but it removes repeated JSON body transfer when session state is unchanged.

## 2026-06-12 15:44
- Refactor mechanism: file search was a contained server hotspot with pure scoring/search behavior and existing tests. Extracting it reduces `server.py` responsibility without changing HTTP route semantics; the injected `_git_repo_root` callback preserves current git detection behavior.

## 2026-06-12 15:48
- Refactor mechanism: content-type classification is pure and shared by file viewer/upload/video-preview paths. Moving it out of `server.py` reduces route/module load while preserving server aliases and existing call-site semantics.

## 2026-06-12 15:54
- Refactor mechanism: text-file decoding/edit/write behavior is a pure helper cluster with direct tests. Extracting it further reduces `server.py` surface while preserving imported function names at existing call sites.

## 2026-06-12 16:02
- Observation: existing video preview runtime tests mutate `server.VIDEO_PREVIEW_DIR`; a direct imported helper would have changed the override mechanism. The accepted extraction keeps server wrappers that inject the server global into the implementation module, so the observable preview cache authority remains unchanged.

## 2026-06-12 16:06
- Observation: image validation/PNG CRC helpers were not called by file preview or attachment upload paths. Treating them as active protection would overstate behavior; removing them makes the server surface better match the real invariant, where client-side image compression is best-effort and server attachment staging is byte-preserving with size limits.

## 2026-06-12 16:11
- Refactor mechanism: file-view interpretation (directory, media, too-large, binary, text/markdown) is pure path/byte classification. Route-specific byte-range streaming remains in `server.py`, avoiding a false abstraction over HTTP handler behavior.

## 2026-06-12 16:15
- Refactor mechanism: byte-range parsing and inline response streaming are route-shared HTTP mechanics. Moving them as a unit preserves 416 handling, cache headers, and range semantics while reducing repeated file-route coupling in `server.py`.

## 2026-06-12 16:19
- Observation: upload tests patch `server.UPLOAD_DIR` and `server._now`, so a direct imported staging helper would silently change test/runtime override behavior. The accepted extraction keeps server as authority over those mutable values and injects them into the pure module implementation.

## 2026-06-12 16:23
- UX observation: before this fix, the attachment action was guarded but visually enabled with no selected session, producing a silent no-op. The corrected invariant is that session-dependent composer controls should advertise unavailable state before click, not after an ignored action.

## 2026-06-12 16:26
- UX observation: queue is a per-session surface; leaving its composer button enabled without a selected session caused the same silent no-op class as attach. The session-dependent composer-control invariant now covers both attachment and queue controls.

## 2026-06-12 16:31
- UX observation: composer submit had the same enabled-looking/no-selected no-op behavior as attach and queue. The session-dependent composer-control invariant now covers attach, queue, and send, with explicit disabled state and keyboard-submit feedback.

## 2026-06-12 16:35
- UX observation: the no-session title previously retained edit affordance styling and tooltip even though click returned immediately. The visible affordance now matches the actual selected-session precondition.

## 2026-06-12 16:39
- UX/accessibility observation: a clickable div without keyboard semantics made the selected-session title edit affordance mouse-only. The updated state model distinguishes selected interactive title from no-session inert title at cursor, title, role, aria label, aria-disabled, and tab order levels.

## 2026-06-12 16:45
- Race mechanism fixed: prior attachment upload read `selected` after asynchronous image compression/arrayBuffer work, so switching sessions mid-upload could target the wrong session. The target session and attachment index are now captured before async work; UI updates are conditional on still viewing that session.

## 2026-06-12 16:50
- Race mechanism fixed: repeated taps on New Session could reach multiple awaited `spawnSessionWithCwd` calls because no in-flight guard existed. The guard starts after local validation and before the network launch request, so invalid forms remain editable while valid launches are single-flight.

## 2026-06-12 16:54
- Race mechanism fixed: diagnostics previously read `selected` at request time but rendered after await without checking whether selection changed. Rendering is now scoped to the captured session id.

## 2026-06-12 16:58
- Mechanism fixed: `sessiond` previously called `_proc_find_open_rollout_log` only for Codex/Claude, even though the util discovery function already supports Pi session paths and headers. Pi headless sessions could therefore remain bound to pending placeholder logs. Allowing Pi through the same backend-aware discovery path binds the real Pi log and removes the pending placeholder.

## 2026-06-12 17:07
- Observation: before the API hardening tranche, JSON decode/object-shape failures in POST routes could escape to `_handle_route_exception`, which returned HTTP 500 and exposed a `trace` field.
- Mechanism supported: malformed client input was being represented as ordinary exceptions rather than typed client-error evidence; duplicated inline POST parsers made this likely to recur.
- Intervention: introduced `BadRequestError` and `RequestPayloadTooLargeError`, routed `_read_json_body()` through them, replaced duplicate inline JSON parsing in POST routes, and hid generic 500 traces unless `CODEX_WEB_DEBUG_ERRORS=1`.
- Evidence: focused tests passed (`25 passed in 2.11s`); isolated Docker curl showed malformed `/api/login` and authenticated malformed `/api/sessions/fake/inject_file` return 400 with no `trace`.
- Scoped claim: representative malformed JSON and object-body errors now fail as client errors instead of internal server errors; this does not prove every semantic validation path returns an ideal status, only the centralized parse/object-shape boundary and checked attach filename path.


## 2026-06-12 17:14
- Observation: `saveActiveFileEdits()` previously built its write URL/body from mutable file-viewer globals and then applied the response to the current active file after `await`. A user opening another file/session while the save was in flight could let an old response rewrite the new viewer state.
- Mechanism supported: file-open requests already have ownership guards, but save requests lacked an equivalent session/path ownership boundary.
- Intervention: bound save request target and post-await mutation to the captured session/path snapshot. Success and error responses from stale saves no longer update the visible file state.
- Evidence: source regression asserts captured `saveSessionId`/`savePath`/draft/version, captured write URL/body, and `saveStillCurrent()` guards on success/error; targeted tests and full Docker suite passed.
- Scoped claim: stale file-save responses are prevented from mutating a different active file/session. This does not simulate Monaco in-browser interleavings; it constrains the JS ownership boundary by source and suite evidence.


## 2026-06-12 17:17
- Observation: the edit-conversation save handler previously used mutable `editSessionId` and unconditionally hid the edit modal after `await`; it also compared `selected === editSessionId` after `hideEditSession()` had cleared `editSessionId`.
- Mechanism supported: a save response for session A could close or write error text into a later edit modal for session B if the user reopened the modal while A's request was in flight. Duplicate clicks could also issue duplicate saves.
- Intervention: captured `sid` at save start, used it for the API URL and selected-title refresh, disabled the save button while pending, and ignored stale success/error/finally updates unless `editSessionId` still equals `sid`.
- Evidence: source regression constrains the captured id, disabled guard, stale success/error checks, and selected-title update; targeted tests and full Docker suite passed.
- Scoped claim: edit-save responses are now owned by the edit session that initiated them. This does not prove all edit modal UX semantics on real mobile devices, only the stale-response and duplicate-save boundaries in the current JS.


## 2026-06-12 17:29
- Observation: fresh reviewer found that logout/auth-loss could call `renderLogin(renderApp)` without stopping the active message poller or removing renderApp-registered global listeners. `pollMessages()` also lacked a 401 branch.
- Mechanism supported: replacing the root DOM is not sufficient cleanup for timers, in-flight controllers, audio heartbeat/watchdog timers, or document/window listeners captured by the old renderApp closure. A re-login could therefore create duplicate event handlers and stale pollers.
- Intervention: introduced an active-app cleanup authority, routed render boundaries/logout/auth-loss through it, moved session/secondary poll control to shared functions, cleanup-tracked renderApp global listeners, and guarded the boot `finally` so auth-loss cleanup is not followed by listener re-registration.
- Evidence: source regressions assert the cleanup boundary, stopped timers/controllers/listeners, 401 auth-loss routing, tracked global events, and the boot-finally guard; targeted tests and full Docker suite passed.
- Scoped claim: current source prevents old renderApp pollers/listeners from surviving logout/auth-loss or re-render. Runtime browser request-count evidence was not collected for this tranche, so the claim is bounded to source-level lifecycle invariants plus suite coverage.


## 2026-06-12 17:33
- Observation: fresh reviewer found the download route used `_read_downloadable_file()`, which called `Path.read_bytes()` and then wrote the full byte string to `wfile`. Existing tests only exercised a tiny binary fixture.
- Mechanism supported: large artifact downloads could allocate the whole file in the server process before sending; inline preview/blob routes already used chunked streaming, making download the outlier.
- Intervention: replaced the buffering download helper with size/permission inspection and added a shared attachment response streamer that preserves `Content-Length`, `Content-Disposition`, and no-store headers while copying chunks to `wfile`.
- Evidence: tests poison `Path.read_bytes` for both metadata inspection and attachment response streaming; targeted tests and full Docker suite passed.
- Scoped claim: the session file-download response no longer buffers the complete file in Python before sending. It still uses a precomputed `Content-Length`, so concurrent file mutation during a download remains outside this tranche.


## 2026-06-12 17:36
- Observation: fresh reviewer found that service-worker notification clicks called `client.navigate(target)` without awaiting it, then focused the old client; app hashchange ignored target sessions absent from the current `sessionIndex` snapshot.
- Mechanism supported: an existing tab with a stale session list could receive a notification hash for a newly visible session, ignore it, and never replay the target after later session polling refreshed state.
- Intervention: await navigation before focus in the service worker; route hash changes through `selectSessionFromHash({ refreshIfMissing: true })`, which refreshes sessions once and selects the target only if it becomes selectable.
- Evidence: source regressions assert awaited navigation/focus fallback and the hash refresh/select flow; targeted tests and full Docker suite passed.
- Scoped claim: notification target hashes are no longer dropped solely because the open tab had a stale session snapshot. Browser-level push-click behavior was not simulated in Docker for this tranche.


## 2026-06-12 17:39
- Observation: fresh reviewer found `sessiond.py` had a real `main()` and README/architecture docs described a headless helper, but installed scripts exposed only server and broker.
- Mechanism supported: a user installing the package would not discover or invoke `sessiond` as a first-class command, keeping the headless path less visible and less exercised.
- Intervention: added the `codoxear-sessiond` console script and README examples using the same `CODEX_WEB_AGENT_BACKEND` convention as broker wrappers.
- Evidence: source test parses `pyproject.toml`, README assertions cover the new examples, `python -m codoxear.sessiond --help` exits successfully, and full Docker suite passed.
- Scoped claim: installed package metadata and README now expose sessiond. This does not validate real backend startup under `codoxear-sessiond` with Codex/Pi/Claude credentials.


## 2026-06-12 17:42
- Observation: fresh reviewer found browser-compatible MP4 previews were content-hashed under app state and never pruned. Frequently edited or many large videos could silently consume disk.
- Mechanism supported: `ensure_video_preview()` reused/created hashed files but had no deletion path beyond temporary file cleanup.
- Intervention: added env-configured file-count and byte-size caps for the preview directory; pruning deletes oldest non-temp previews first while preserving the preview just returned.
- Evidence: direct tests cover file-count pruning, byte-cap pruning, keep preservation, and temp-file exclusion; targeted tests and full Docker suite passed.
- Scoped claim: preview cache growth is now bounded by configured caps under normal preview access. It does not proactively prune on server startup or after session deletion unless preview access occurs.


## 2026-06-12 17:50
- Observation: final clean-room critic found two blockers: the new streaming attachment response could write more bytes than `Content-Length` if a file was appended after inspection, and README `codoxear-sessiond` examples repeated backend executable names even though sessiond prepends the selected backend binary itself.
- Intervention: capped attachment streaming to the inspected size and corrected README/tests to document backend options after `--`, not backend executable names.
- Evidence: unit test now writes a larger file than the declared size and asserts only the declared prefix is emitted; sessiond packaging test rejects the duplicated-executable examples; targeted tests and full Docker suite passed.
- Scoped claim: the append-side `Content-Length` mismatch blocker is fixed. Concurrent truncation of a mutable file during download remains a general static-file streaming residual risk; the response will not emit more than the declared length.


## 2026-06-12 17:54
- Observation: final clean-room critic rerun at `fe170b5` found no blockers after the blocker repairs.
- Evidence: review artifact `/tmp/codoxear-final-cleanroom-critic-rerun.md`; independent full isolated Docker validation passed (`511 passed, 2 skipped, 10 subtests passed in 19.57s`).
- Scoped claim: the recovery branch is at a defensible yield point for the reviewed product-gap tranche. Remaining uncertainties are explicitly bounded to live backend startup/credentials, mobile-device/browser push behavior, mutable-file truncation during streaming downloads, slow networks, and very large real transcripts.


## 2026-06-12 18:04
- Observation: previous notification fallback still selected only once; if server-side session discovery lagged beyond that refresh, the target hash could be dropped. Previous cleanup also stopped future timers/listeners but could not stop already-in-flight async polling results from mutating UI after cleanup.
- Mechanisms supported: session-list state can lag notification hash targets, and async functions can cross logout/auth-loss boundaries unless they check ownership after awaits.
- Interventions: added a pending hash-session target that is retried after subsequent `refreshSessions()` while the hash still matches; added post-await `appDisposed` guards in session refresh, voice settings, notification state, notification feed, and desktop notification resolution paths.
- Evidence: source regressions assert deferred hash target ownership and post-cleanup async guards; full isolated Docker suite passed.
- Scoped claim: open tabs no longer drop notification hash targets solely because target session discovery lags one refresh, and selected async poll paths no longer apply results after app cleanup. Real service-worker click behavior and mobile push timing remain outside deterministic evidence.


## 2026-06-12 18:18
- Observation: fresh product/UX review identified five user-facing gaps: disabled pinch zoom, repeated voice API-key exposure to browser JS, click-only login, non-modal Settings dialog behavior, and New Session defaulting to Codex before selected-session backend.
- Interventions: removed viewport/gesture zoom suppression; added redacted voice settings snapshots plus blank-save preservation and explicit clear; converted login to a real form with password semantics and focus; opened Settings with native modal/cancel/Escape behavior; changed backend precedence to selected session -> remembered -> server default.
- Evidence: source/unit tests constrain each contract; full isolated Docker suite passed; browser runtime evidence confirmed viewport/login/settings behavior in an isolated server/browser session.
- Scoped claim: the reviewed UI/UX defects are fixed for the current browser-rendered app. Real mobile-device pinch gestures, real password manager behavior, and live TTS key workflows are not exhaustively proven beyond DOM/source/server evidence.


## 2026-06-12 18:22
- Observation: fresh architect review found broker and sessiond duplicated the same JSON-line control command dispatch (`state`, `tail`, `send`, `keys`, `shutdown`) plus exception/close handling.
- Mechanism supported: shared server control protocol should have one dispatch/error boundary while broker/sessiond keep their distinct state mutation and PTY-write semantics.
- Intervention: introduced `handle_control_socket_connection()` with per-command callbacks; broker/sessiond now provide local callbacks for state/tail/send/keys/shutdown. Send still replies before slow injection; keys retains each process's existing response shape.
- Evidence: new protocol-helper tests cover known/unknown/invalid/exception cases; existing send-ack and fail-closed tests pass; full isolated Docker suite passed.
- Scoped claim: duplicated dispatch mechanics were reduced without changing intended wire semantics. This is not a full control-protocol abstraction of state machines or busy/idle authority.


## 2026-06-12 18:25
- Observation: Settings modal semantics were improved earlier but focus restoration had not been proven. Browser evidence now shows focus returns to the Settings opener after Escape-close.
- Evidence: source tests constrain showModal/cancel/Escape/focus restoration; full isolated Docker suite passed; browser runtime evidence confirmed focus transition and redacted empty API-key field.
- Scoped claim: Settings behaves as a keyboard-modal dialog in the current browser runtime. This does not exhaustively prove every assistive technology path.


## 2026-06-12 18:34
- Observation: clean-room review found two blockers in the voice-key tranche: background polls could clobber open Settings edits/clear checkbox, and persisted secrets inherited umask-derived permissions.
- Interventions: split Settings form syncing from general voice UI updates and skip form sync while the dialog is open; chmod voice settings and VAPID private key files to `0600` after creation/write and on existing VAPID key load.
- Evidence: targeted tests, full isolated Docker suite, browser poll-window scenario, and container file-mode checks all passed.
- Scoped claim: explicit key clear is robust across the normal visible secondary poll interval, browser GET remains redacted, and newly/currently written voice secret files are private to the user. Existing app directories may still have broader directory permissions, but secret files are `0600`.


## 2026-06-12 18:44
- Observation: architecture review found backend launch semantics spread across server request parsing, argv assembly, environment ownership, resume args, and tmux inline environment.
- Intervention: centralized the backend-specific launch-plan mechanics in `backend_launch.py` while keeping orchestration in `SessionManager.spawn_web_session`.
- Evidence: direct adapter tests constrain Codex/Pi/Claude args, resume args, env ownership, and tmux inline env/unset contract; existing launch defaults, launch request, resume, and Claude source tests pass; full isolated Docker suite passes.
- Scoped claim: backend argv/env/resume/tmux-inline semantics now have one tested owner. UI defaults and request validation still remain in `server.py`, so this is an incremental launch-adapter extraction, not a complete backend launch abstraction.


## 2026-06-12 18:56
- Observation: fresh review identified a draft-atomicity bug: attachments are injected immediately into the broker input, while text can be queued via `Send after current`; this can split a user-visible draft across different turns.
- Mechanism: queue items currently store only text, not attachment payloads or staged file references. Until queue items support attachments, allowing attach during running turns or queueing a draft with attachments violates atomic draft semantics.
- Intervention: attach button state now depends on selected/running/sending; file-picker change handler rechecks running before upload; send-choice captures current attachment count and disables/blocks `Send after current` when attachments are present.
- Evidence: source tests assert running attach fail-closed labels and queued-attachment block; runtime sendText harness updated for attach-state sync; full isolated Docker suite passed.
- Scoped claim: the UI now prevents known text/attachment splitting paths. This does not implement attachment-aware queue items; that remains a larger feature.


## 2026-06-12 18:59
- Observation: fresh review found each chat-search input change immediately started `/messages/search?...limit=0`, which makes the server re-read/normalize the full transcript per keystroke.
- Mechanism: client aborts do not prevent server-side work already started, so superseded keystrokes can still consume disk/CPU on large logs.
- Intervention: keep loaded message marking synchronous but schedule the expensive all-transcript count through a 300ms debounce; reset/abort old requests and clear the timer on cleanup.
- Evidence: source tests constrain debounce scheduling and cleanup; full isolated Docker suite passed.
- Scoped claim: rapid typing no longer starts one full-transcript count request per input event from the client. Server-side caching for repeated final queries is not implemented.


## 2026-06-12 19:00
- Observation: fresh review found plain pytest could import stale modules from `/home/yiwen/codex-web` unless `PYTHONPATH` was forced.
- Mechanism: unattended reviewers/workers using default pytest could validate the wrong code, making green tests non-evidence for the active branch.
- Intervention: pyproject now sets pytest `pythonpath = ["."]` and a source test locks the contract.
- Evidence: targeted and full plain-pytest runs passed without manually exporting `PYTHONPATH`.
- Scoped claim: local pytest now resolves the active checkout by default. Docker sandbox still sets `PYTHONPATH=/workspace`, which remains compatible.


## 2026-06-12 19:06
- Observation: fresh review found core file viewing depended on external CDN scripts; if Monaco loader or pdf.js failed/hung, the viewer could remain stuck at loading.
- Mechanism: waiting indefinitely for `window.require` or dynamic pdf.js imports makes local/LAN use fragile under offline, captive-portal, or CDN-blocked conditions.
- Intervention: added bounded loader timeouts with retry, read-only plain-text fallback for Monaco file/diff views, and PDF open/download fallback for pdf.js/IntersectionObserver failure.
- Evidence: source tests constrain timeout constants, fallback renderers, PDF fallback path, and CSS; full local and isolated Docker suites passed.
- Scoped claim: CDN failure no longer leaves the file viewer with an unbounded loading state for text/diff/PDF paths. Monaco/pdf.js are still CDN-hosted; full vendoring is a separate larger change.


## 2026-06-12 19:18
- Observation: current-head critic found two blockers: third-party CDN scripts/fonts executed in the authenticated origin, and existing secret files were not chmod-repaired on load.
- Mechanisms: remote scripts in the app origin can call authenticated `/api/*` and exfiltrate private session/file state; permissive pre-existing `voice_settings.json`/`hmac_secret` files preserve local secret exposure after upgrade.
- Interventions: app shell now uses only self assets with CSP; Monaco/PDF loaders no longer reference jsDelivr and fall back locally; existing voice settings and HMAC secret files are chmodded `0600` during load.
- Evidence: targeted tests, full local suite, full isolated Docker suite, and static no-third-party URL assertions passed.
- Scoped claim: authenticated app execution no longer depends on third-party script/font assets in the committed shell. Rich Monaco/PDF functionality now requires future self-hosted assets or falls back; HLS.js is no longer loaded from CDN, so non-native HLS live audio support is reduced until vendored locally.


## 2026-06-12 19:25
- Observation: architecture review identified a remaining launch ownership split: UI-advertised defaults and accepted request fields lived in `server.py`, while argv/env/resume encoding lived in `backend_launch.py`.
- Intervention: introduced `launch_config.py` for pure/path-parameterized launch defaults and request parsing; server wrappers preserve existing API and tests while narrowing server ownership to HTTP/orchestration.
- Evidence: direct launch/default/request/source tests and full local/Docker suites passed.
- Scoped claim: launch defaults and request validation now have a dedicated owner, aligned with the existing backend launch adapter. Process spawning, tmux orchestration, and launch-attempt recording intentionally remain in `SessionManager.spawn_web_session`.


## 2026-06-12 19:34
- Observation: post-blocker critic found the CSP was only a meta tag, so `frame-ancestors` was not enforced; it also found duplicate server helper definitions shadowing launch_config wrappers.
- Intervention: HTML static responses now carry an enforced CSP header and `X-Frame-Options: DENY`; static path containment uses `relative_to`; shadowing launch helper definitions were removed.
- Evidence: targeted tests assert server header emission and launch wrapper behavior; full local/Docker suites passed.
- Scoped claim: anti-framing policy is now delivered as an HTTP header for served HTML. CSP still allows inline scripts/styles because current shell uses inline bootstrapping; this removes third-party execution and framing, not all possible XSS impact.


## 2026-06-12 19:40
- Observation: queue persistence and item mutation logic lived inside `SessionManager`, interleaved with live broker readiness and idle scheduling.
- Intervention: introduced `QueueStore` for durable item ownership and local mutation semantics, preserving `SessionManager._queues` as compatibility map and leaving drain/send orchestration in `SessionManager`.
- Evidence: new QueueStore tests cover legacy string migration, duplicate id repair, sending-item mutation rejection, successful-send removal by id (not duplicate text), and stale-session pruning; existing queue/unattended tests and full suites pass.
- Scoped claim: queue item storage/mutation has a dedicated owner. Live scheduling, idle gating, and send failure handling remain SessionManager responsibilities.


## 2026-06-12 19:41
- Observation: runtime GET evidence confirms CSP/X-Frame-Options are delivered as HTTP headers for `/`, satisfying the critic's enforcement concern. The failed HEAD probe was invalid because the handler does not implement HEAD.


## 2026-06-12 19:51
- Observation: final clean-room review found attachment split prevention was client-only; direct/stale `/inject_file` could still bracket-paste into a busy PTY.
- Intervention: added server-side attachment readiness guard before staging uploaded bytes or calling `inject_keys`; guard checks local queue/sending state plus broker busy/queue state.
- Evidence: targeted tests assert busy/remote-queue/local-queue/sending rejection and route guard before staging; full local/Docker suites passed.
- Scoped claim: attachment injection is now fail-closed at the server API boundary for known busy/local-queue states. It still relies on broker `state` accuracy for remote busy reporting.


## 2026-06-12 19:59
- Observation: clean-room rerun found two residual attachment injection mechanisms: broker-idle/log-busy disagreement and a TOCTOU race between readiness check and final PTY write.
- Interventions: `attachment_injection_ready()` now matches queue readiness by checking `idle_from_log`; `send()` and `inject_attachment_keys()` share a per-session input lock, and attachment injection rechecks readiness under that lock immediately before `keys`.
- Evidence: targeted tests cover broker-busy, remote queue, local queue, sending item, log-busy rejection, and final recheck; full local/Docker suites passed.
- Scoped claim: known server-side paths that could inject attachments into busy/log-active sessions are now guarded. This does not prove correctness if a backend reports/logs idle incorrectly.


## 2026-06-12 20:09
- Observation: clean-room rerun found remaining attachment races through sessiond's reply-before-busy behavior, local queue creation outside input locking, and log-path binding after readiness snapshot.
- Interventions: sessiond marks busy before acknowledging send; attachment readiness rechecks queue/sending/log path after broker state; enqueue queue append now shares the input lock.
- Evidence: tests encode the counterexamples (queue mutation during state refresh, log path binding during state refresh, sessiond busy-before-ACK) plus full local/Docker suites.
- Scoped claim: the identified attachment split/race mechanisms are now closed for server-managed send/enqueue/sessiond/broker paths; explicit interrupt remains a separate PTY mutation by design.


## 2026-06-12 20:15
- Observation: clean-room rerun found a stale metadata mechanism: broker sidecar could point to a busy new log while server memory still held an old idle `log_path`; broker `state` alone could report idle.
- Intervention: attachment readiness now refreshes sidecar metadata when available before checking local/log state and again after broker state refresh; tests simulate sidecar refresh rebinding to a busy log.
- Evidence: targeted stale-metadata regression plus full local/Docker suites passed.
- Scoped claim: attachment readiness now uses current sidecar-bound log metadata for the known stale-log bypass; it still cannot cover unreported physical-terminal keystrokes before backend state/logs observe them.


## 2026-06-12 20:23
- Observation: metadata refresh inside attachment readiness had an unintended side effect: it could drain a local queue, reenter `send()`, and self-deadlock on the same per-session input lock.
- Intervention: made queue draining explicit in `refresh_session_meta(drain_queue=True)` and disabled it for attachment-readiness metadata refresh.
- Evidence: targeted tests verify attachment metadata refresh uses `drain_queue=False` and does not drain queues; full local/Docker suites passed.
- Scoped claim: attachment readiness metadata refresh is now side-effect-free with respect to queue promotion, closing the observed self-deadlock mechanism.


## 2026-06-12 20:34
- Observation: clean-room review identified a post-injection semantic gap: a pasted attachment line was unreserved PTY input, so queue/unattended/direct sends could consume it before the user's intended prompt.
- Interventions: added session `pending_attachment` state, server-side queue/unflagged-send barriers, and an explicit UI send flag for intended attachment commit.
- Evidence: targeted tests cover pending attachment blocking enqueue/unflagged send, queue promotion stop, explicit send clearing the marker, frontend send payload, and full local/Docker suites.
- Scoped claim: web/server send and queue paths now preserve attachment ownership until an explicit attachment-commit send; physical terminal input remains outside server serialization.


## 2026-06-12 20:44
- Observation: clean-room review found pending attachment barriers were not atomic with the input lock, were lost on safe server restart, and client metadata alone could grant another tab attachment-consumption authority.
- Interventions: moved pending-send authorization inside the input lock, persisted pending session ids, restored flags during discovery, and restricted web `allow_pending_attachment` to local attached-file state.
- Evidence: targeted tests cover pending-state persistence set/clear, locked send rejection, queue blocking, and frontend no-auto-consent; full local/Docker suites passed.
- Scoped claim: server-managed send/queue paths now preserve pending attachment ownership across concurrent sends and server restarts, unless the user explicitly sends from a local attachment-owning composer.


## 2026-06-12 20:52
- Observation: clean-room review found persisted pending attachments could be unsendable after UI reload, and send ACK-before-PTY-commit let a second direct send pass after pending was cleared but before Enter was written.
- Interventions: UI now requires explicit confirmation to send restored pending attachments; server `send()` checks remote/log readiness under the input lock before calling broker/sessiond.
- Evidence: targeted tests cover remote-busy send rejection before broker call and frontend confirmation flow; full local/Docker suites passed.
- Scoped claim: recovered pending attachments have an explicit UI consent path, and server-managed sends fail closed while broker/sessiond report the previous send as busy during post-ACK PTY commit.


## 2026-06-12 21:01
- Observation: clean-room review found direct send could overtake local queues, send readiness used stale log metadata, canceling pending-attachment confirmation left a phantom local echo, and failed key injection created phantom pending state.
- Interventions: added local queue/sending guards to direct send with a queue-promotion token, refreshed send log metadata, moved UI confirmation before optimistic echo, and introduced `SessionInjectionError` before pending state is persisted.
- Evidence: targeted regressions encode all four counterexamples; full local/Docker suites passed.
- Scoped claim: server-managed direct send now respects local queue order and live log metadata, and attachment-pending state is only created after successful key injection.


## 2026-06-12 21:09
- Observation: clean-room review found control-socket success could precede or mask PTY write failure, causing phantom pending set/clear; failed sends also left phantom optimistic UI rows.
- Interventions: key writes now return error responses, pending-attachment sends use synchronous broker/sessiond commit ACKs, server preserves pending state on commit error, and frontend removes optimistic rows on send failure.
- Evidence: targeted tests cover broker/sessiond sync send failures, broker key-write failures, pending-send error preservation, and UI rollback; full local/Docker suites passed.
- Scoped claim: attachment pending state now tracks successful server-visible PTY writes for broker/sessiond paths; non-pending sends still intentionally ACK before after-reply injection for latency and do not mutate pending state.


## 2026-06-12 21:16
- Observation: clean-room review found sync send failure left broker/sessiond busy, preventing retry, and optimistic send cleanup removed only the bubble not the row.
- Interventions: restored broker/sessiond pre-send busy/turn state on synchronous `_inject` failure; frontend now removes the closest `.msg-row` for failed local echoes.
- Evidence: targeted tests assert busy rollback and row-level cleanup; full local/Docker suites passed.
- Scoped claim: failed synchronous attachment-commit sends leave the session retryable and do not leave durable optimistic transcript rows.


## 2026-06-12 21:22
- Observation: clean-room review found the non-attachment async send path shared the failed-inject busy-stuck mechanism after ACK.
- Intervention: broker/sessiond async `after_reply()` now restores pre-send busy/turn state if `_inject` fails or no PTY fd is available.
- Evidence: targeted tests assert async send failure returns fast success but leaves broker/sessiond idle/retryable; full local/Docker suites passed.
- Scoped claim: both sync and async server-visible send paths now roll back broker/sessiond busy state on local PTY injection failure; async callers still cannot be told the already-ACKed send failed.


## 2026-06-12 21:28
- Observation: clean-room review found async ACK still let queued/manual server sends be reported and popped before delivery; deferred inject failure was invisible to server.
- Intervention: server-managed sends now request synchronous broker/sessiond commit for all sends. Direct broker control clients can still choose async by omitting `sync`, but Codoxear HTTP success now means the broker/sessiond write completed or returned an error.
- Evidence: targeted tests assert normal server sends include `sync: true`; full local/Docker suites passed.
- Scoped claim: HTTP `/send` and queue promotion no longer use fast ACK as the commit boundary; this may add PTY write latency but preserves queue/send correctness.


## 2026-06-12 21:35
- Observation: clean-room review found a remaining split: sync send could write to PTY but exceed the server's 3s socket timeout, causing false failure and queue duplication.
- Intervention: removed the short socket timeout for server-managed sync sends so the HTTP commit boundary is the broker/sessiond reply after `_inject` returns.
- Evidence: targeted source test asserts `timeout_s=None`; full local/Docker suites passed.
- Scoped claim: server-managed sends no longer false-timeout at 3s after committed PTY writes; a genuinely hung broker sync commit can still hang the HTTP request until external cancellation.


## 2026-06-12 21:40
- Observation: final clean-room rerun found no demonstrated blockers at `adcfd37` after synchronous send timeout boundary fix.
- Scoped claim: the reviewed server-managed attachment/send/queue commit-boundary blockers are closed under source inspection and test evidence. Residual risks: sync send can hang if broker/PTY never replies; real credentialed Codex/Pi/Claude startup and mobile/slow-network behavior remain unproven.


## 2026-06-12 21:54
- Observation: product review identified that `timeout_s=None` for synchronous `/send` removed false 3s failures but introduced unbounded browser/server hangs if broker accepted a sync send and never replied.
- Mechanisms considered: a finite timeout after possible PTY side effects cannot be treated as ordinary failure without risking duplicate queue/manual sends; it must be an explicit unknown commit state.
- Interventions: bounded server sync send wait; `SessionCommitUnknownError` maps to 504/`commit_unknown`; queue promotion marks timed-out head items as `commit_unknown` and blocks automatic resend; frontend surfaces unknown status and keeps user text recoverable.
- Evidence: targeted regressions cover timeout preservation of pending attachments, queue unknown marking/non-promotion, queue-store persistence, frontend unknown UI, and full local/Docker suites.
- Scoped claim: server/browser no longer hang indefinitely on missing sync-send replies, and timeout does not create false success or automatic queue duplication. It remains possible that a user manually resends after an unknown commit without checking transcript/terminal.


## 2026-06-12 22:06
- Observation: adversarial review found bounded send unknown semantics still failed across server restart mid-queue-dispatch, non-timeout response loss, and old live brokers that ignore `sync`.
- Interventions: persist conservative `commit_unknown` before queue dispatch, treat empty/failed send responses as unknown after request, and require broker/sessiond sync-send capability in metadata before HTTP `/send`.
- Evidence: targeted regressions cover unsupported brokers, empty response unknown, durable pre-dispatch queue unknown marking, unknown queue non-promotion, queue-store persistence/list behavior, and full local/Docker suites.
- Scoped claim: queued prompts are no longer auto-retried after server restart or response-loss uncertainty, and HTTP send no longer silently trusts old broker control sockets for confirmed-send semantics.


## 2026-06-12 22:15
- Observation: bounded-send rerun found parseable malformed broker responses still produced ordinary failures after dispatch, and old brokers could accept attachment key injection even though later confirmed send was impossible.
- Interventions: validate send response schema before local submitted bookkeeping and classify malformed/incomplete responses as commit-unknown; require sync-send plus key-write-error capability before attachment injection.
- Evidence: targeted tests cover malformed responses, unsupported send brokers, unsupported attachment brokers, and full local/Docker suites.
- Scoped claim: post-dispatch response corruption no longer creates false ordinary failure or submitted bookkeeping, and old live brokers cannot create browser-managed pending attachment state without the confirmed-send/key-error protocol.


## 2026-06-12 22:25
- Observation: rerun found three remaining commit-unknown gaps: confirmed non-commit queue failures stayed frozen as unknown, coercible invalid `queue_len` values were accepted as success, and persisted pending attachments on unsupported old brokers had no recovery path.
- Interventions: broker/sessiond distinguish declared commit-unknown errors; queue promotion clears unknown markers for known `SessionInjectionError`; send response schema requires a non-negative integer queue length; added explicit pending-attachment clear endpoint and UI recovery confirmation.
- Evidence: targeted tests cover known vs unknown queue failure, strict queue_len schema, pending clear, UI recovery prompt, and full local/Docker suites.
- Scoped claim: commit-unknown state is now reserved for actual uncertainty, strict malformed responses are not accepted as confirmed sends, and stale pending-attachment state can be cleared explicitly after user confirmation.


## 2026-06-12 22:34
- Observation: rerun found post-request socket failure with dead PIDs was treated as stale session cleanup, deleting a pre-marked unknown queued item even though the prompt may have reached the broker.
- Intervention: control socket send now tracks whether the request was sent before failure; `send()` maps post-request failure to commit-unknown regardless of PID liveness and only prunes dead sessions for pre-request failures.
- Evidence: targeted tests cover post-request failure with dead PIDs preserving the session/unknown state and pre-request dead socket pruning; full local/Docker suites passed.
- Scoped claim: response loss after a dispatched send no longer destroys queue evidence or returns ordinary unknown-session semantics.


## 2026-06-12 22:44
- Observation: rerun found attachment key injection had the same post-request empty-response ambiguity, immediate enqueue hid commit-unknown, and old brokers could accumulate undrainable queued prompts.
- Interventions: attachment keys now use request-sent tracking and mark pending on unknown; queue promotion returns commit-unknown response; enqueue rejects brokers without sync-send support before append.
- Evidence: targeted tests cover attachment empty response pending+unknown, enqueue commit-unknown response, unsupported enqueue rejection, and full local/Docker suites.
- Scoped claim: ambiguity at attachment/enqueue boundaries is now surfaced immediately and preserved conservatively rather than hidden as ordinary failure/queued success.


## 2026-06-12 22:54
- Observation: rerun found false commit-unknown queue markers after generic pre-dispatch errors, attachment key post-request ambiguity hidden as ordinary attach error/no pending state, and malformed attachment replies treated as success.
- Interventions: clear unknown markers for generic non-unknown queue failures; classify attachment key response ambiguity as commit-unknown while conservatively setting pending state; UI shows attachment unknown and refreshes sessions.
- Evidence: targeted tests cover generic queue failure cleanup, attachment empty/malformed unknown pending state, attach UI unknown branch, and full local/Docker suites.
- Scoped claim: attachment and queue uncertainty now follows the same conservative evidence-preserving model as send uncertainty.


## 2026-06-12 23:01
- Observation: rerun found truthy non-boolean attachment `ok` values were accepted as success, and immediate enqueue commit-unknown still recorded pre-log submitted messages before commit was known.
- Interventions: attachment key responses now require boolean `ok is True`; enqueue no longer calls `_record_prelog_user_message` before promotion/commit, relying on `send()` to record only after confirmed commit.
- Evidence: targeted tests cover truthy malformed attachment replies and commit-unknown enqueue no-prelog behavior; full local/Docker suites passed.
- Scoped claim: attachment success acknowledgement is schema-strict and enqueue no longer creates false transcript fallback evidence before a prompt is committed.


## 2026-06-12 23:08
- Observation: rerun found mixed responses like `{ok: true, commit_unknown: true}` or `{queue_len: 0, commit_unknown: true}` were accepted as success because unknown was only checked in error branches.
- Intervention: send and attachment response classification now treats explicit `commit_unknown` as a semantic override before success validation.
- Evidence: targeted tests cover direct send, attachment, and enqueue mixed success+unknown responses; full local/Docker suites passed.
- Scoped claim: broker/control responses cannot downgrade an explicit unknown commit marker into confirmed success via success-looking fields.


## 2026-06-12 23:13
- Observation: focused adversarial rerun found no demonstrated blockers after explicit `commit_unknown` override handling.
- Scoped claim: bounded send/attachment/enqueue commit-unknown paths are internally consistent under reviewed source/tests. Non-blocking residual: direct-send unknown has toast-only recovery, and queue pre-dispatch unknown is conservative if a crash occurs before broker request.


## 2026-06-12 23:16
- Observation: architecture review identified `/api/sessions`/metadata refresh as hidden commit paths because reads could drain queues.
- Intervention: removed queue promotion from `list_sessions()` and made metadata refresh non-draining by default; queue promotion remains in explicit enqueue and queue-sweep mechanisms.
- Evidence: targeted test asserts `list_sessions()` does not call `_maybe_drain_session_queue`; source tests pin non-draining refresh defaults; full local/Docker suites passed.
- Scoped claim: session list/metadata reads no longer act as user-input commit paths. Queue draining may now wait for the queue sweep interval rather than piggybacking on session polling.


## 2026-06-12 23:22
- Observation: independent read-noncommit review found no remaining read endpoint that promotes queued prompts.
- Scoped claim: passive session reads no longer hide prompt commits. Residual non-blocking risk: enqueue response may be stale if the background sweep commits between enqueue append and immediate-promotion response.


## 2026-06-12 23:35
- Observation: focused review identified direct-send commit uncertainty as toast-only; a user could retry and duplicate a prompt without any durable server-side memory of the maybe-sent text.
- Intervention: direct-send unknown responses now create a persisted per-session `commit_unknown_send` record. The server blocks new sends, queues, and queue sweep promotion for that session until the user explicitly clears the marker after checking transcript/terminal evidence. Queue-item uncertainty remains owned by the queue item; attachment uncertainty remains owned by pending attachment state.
- Evidence: tests cover explicit mixed success+unknown persistence, retry/queue/sweep blocking, and clear semantics. Browser validation in isolated Docker showed the warning badge and disabled controls before clear, then re-enabled controls and `commit_unknown_send: false` after clear.
- Scoped claim: direct send commit uncertainty is no longer ephemeral UI state; it is durable server state with an explicit recovery action. It still cannot prove whether the underlying prompt committed; it preserves that uncertainty and prevents overtaking/duplication until human verification.


## 2026-06-12 23:49
- Observation: focused adversarial review found two ways uncertainty could be bypassed: attachment key injection during unresolved direct send unknown, and queue reordering around a commit-unknown head.
- Mechanism: both were alternate input mutation paths not covered by the initial direct-send barriers. Attachment paste mutates the CLI input buffer; queue move can make a later prompt the sweep head.
- Intervention: made direct unknown-send state part of attachment readiness, and made queued unknown items ordering barriers at the queue store authority plus UI affordance layer.
- Evidence: tests cover server attachment rejection and queue-store move rejection; browser validation shows disabled attach/send/queue controls and disabled move-up for later item behind an unknown queue head.
- Scoped claim: unresolved direct or queued commit uncertainty now blocks overtaking input mutations through send, enqueue, attachments, queue sweep, and queue reorder. Interrupt remains a separate control action, not a prompt commit path.


## 2026-06-12 23:59
- Observation: adversarial rerun showed ordinary queue delete removed a queued unknown barrier, allowing later queued work to sweep.
- Mechanism: move was hardened, but delete/update remained normal queue mutations; deleting the unknown head erased the only barrier and editing it could destroy the evidence text.
- Intervention: queue delete now requires explicit `allow_commit_unknown` for unknown items; queue update rejects unknown items; UI deletion of unknown items requires a transcript/terminal confirmation before passing the flag.
- Evidence: tests cover store and manager behavior; API/browser validation showed 409 without flag, cancel preserving the barrier, and confirmed deletion as explicit resolution.
- Scoped claim: queued unknown barriers cannot be bypassed by normal delete/update/move/sweep paths. Confirmed delete is now the explicit user resolution action for queued unknown items.


## 2026-06-13 00:12
- Observation: post-fix review found no blockers but identified residual uncertainty: off-head persisted queue unknowns could still be crossed, fresh direct unknown UI disablement depended on async refresh, and orphaned direct-unknown records were not self-pruned.
- Intervention: generalized queue move barrier to any crossed unknown item, locally marked direct unknown state in the browser immediately after a 504, and pruned commit-unknown direct-send records without active sessions after discovery.
- Evidence: targeted, full local, and isolated Docker suites passed.
- Scoped claim: unknown recovery barriers are now stronger against corrupted/recovered queue order, stale browser metadata during immediate retry windows, and stale disk records after restart.


## 2026-06-13 00:17
- Observation: review found no blockers but identified error-classification and safety-marker lifetime edge cases.
- Intervention: boolean queue move indices now fail at route validation; out-of-range queue moves report range before unknown barriers; startup/direct-unknown pruning uses an age threshold so recent missing-session markers survive transient sidecar absence.
- Evidence: targeted, full local, and isolated Docker suites passed.
- Scoped claim: unknown recovery behavior now preserves safety markers more conservatively and reports malformed queue move requests as client errors.


## 2026-06-13 00:24
- Observation: adversarial review showed cleanup paths could erase fresh direct or queued unknown markers without explicit user resolution, and truthy non-boolean JSON could confirm queued unknown deletion.
- Mechanism: generic deleted-session cleanup and queue missing-session sweep were not recovery-aware; route parsing used `bool(...)` on client input.
- Intervention: separated explicit user deletion from runtime cleanup via `clear_recovery`; preserved unknown queues for missing sessions; required strict boolean confirmation for unknown queue deletion.
- Evidence: tests cover state cleanup preservation, queue-store missing-session preservation, and strict route source; full local and Docker suites passed.
- Scoped claim: unknown recovery markers are no longer silently removed by stale/dead cleanup or truthy non-boolean delete confirmation.


## 2026-06-13 00:32
- Observation: preserving missing-session queued unknowns in the main queue map made `_queue_sweep()` attempt to drain an orphan session, raising `KeyError` before active queues later in iteration could drain.
- Mechanism: evidence preservation and queue scheduling shared the same map but sweep did not filter to active sessions.
- Intervention: sweep now filters drain candidates to active session ids after cleanup preservation; orphan unknown queues remain reviewable through queue list/delete APIs with explicit confirmation.
- Evidence: tests cover sweep skipping an orphan unknown queue while still considering a live queue, and full local/Docker suites passed.
- Scoped claim: missing-session queued unknown evidence no longer crashes or blocks the queue sweeper, and can be explicitly deleted via API if needed.


## 2026-06-13 00:46
- Observation: review found preserved orphan direct unknown markers were stranded: retained on disk but absent from `/api/sessions` and not clearable while the session was missing.
- Mechanism: cleanup preservation outlived the active `Session` object, while recovery endpoints and UI were keyed only to active sessions.
- Intervention: direct orphan markers can now be cleared by id; orphan direct/queue markers are surfaced as recovery rows; opening a recovery row avoids transcript fetch and only exposes review/clear affordances.
- Evidence: tests cover orphan list rows and orphan direct clear; full local/Docker suites passed; browser validation showed direct and queued orphan recovery controls.
- Scoped claim: preserved unknown evidence is no longer stranded solely in JSON files when its session disappears.


## 2026-06-13 00:54
- Observation: orphan recovery row removal left a stale selected id with controls re-enabled; queued orphan review was blocked when the composer contained draft text.
- Mechanism: session list refresh rebuilt `sessionIndex` but did not clear selected ids that disappeared; queue button prioritized enqueue behavior before recovery review.
- Intervention: refresh now clears stale selected state, and orphan recovery queue review has precedence over draft enqueue behavior. Orphan `openSession()` avoids transcript-tail fetch.
- Evidence: targeted/full/Docker tests passed; browser validation showed controls disabled after direct orphan clear and queue viewer opens despite draft text.
- Scoped claim: resolving the last orphan evidence no longer leaves a phantom selected session, and queued orphan recovery remains reachable from the enabled queue/review button.


## 2026-06-13 01:07
- Observation: deleting the unknown item from an orphan queue could make later ordinary queued prompts unlistable and sweep-prunable.
- Mechanism: orphan visibility was tied only to `commit_unknown`, so resolving that marker removed the recovery surface for other queued prompts in the same missing-session queue.
- Intervention: remaining items in an orphan queue become persisted `orphan_recovery` items when an unknown item is explicitly deleted; missing-session cleanup/listing/API gates preserve queues with either `commit_unknown` or `orphan_recovery` markers.
- Evidence: tests and browser validation show the later prompt remains visible/reviewable after deleting the unknown head, while generic stale queues without recovery markers are still pruned.
- Scoped claim: resolving one orphan queue unknown no longer strands later queued prompts for that missing session.


## 2026-06-13 01:14
- Observation: orphan recovery leftovers could auto-send if the session id became active again; orphan recovery rows had a dead sidebar delete affordance and could be reclassified to pending transcript state.
- Mechanism: queue promotion only checked `commit_unknown`, delete_session did not recognize recovery rows, and synthetic rows lacked transcript state.
- Intervention: treat `orphan_recovery` as non-promotable recovery evidence, make delete_session clear orphan recovery stores, and mark synthetic rows as failed/non-polling transcript state. Queue UI locks recovery rows while still allowing deletion.
- Evidence: targeted/full/Docker suites passed.
- Scoped claim: preserved orphan prompts remain review-only and cannot be injected automatically if a session id reappears.


## 2026-06-13 01:22
- Observation: adversarial review found `orphan_recovery` rows were visually locked but deletable without explicit confirmation, and API update/move could mutate them.
- Mechanism: delete confirmation and server mutation barriers only checked `commit_unknown`, not the new `orphan_recovery` marker.
- Intervention: added strict `allow_orphan_recovery` confirmation, server-side update/move/reorder barriers for recovery rows, and UI confirmation/tagging for recovery deletion.
- Evidence: targeted/full/Docker suites passed.
- Scoped claim: preserved orphan recovery prompts cannot be silently edited, moved, auto-sent, or deleted without explicit recovery confirmation.


## 2026-06-13 01:29
- Observation: a missing-session queue with an unknown head and later normal prompts still allowed one-click/API deletion of the later prompts before resolving the head.
- Mechanism: recovery protection was per item, but orphan recovery is a queue-level state once the session is missing and any recovery evidence exists.
- Intervention: queue listing marks all items in a missing-session recovery queue as recovery-protected; delete requires explicit orphan recovery confirmation for those later items too; recovery conflict errors map to 409.
- Evidence: tests cover later-item protection before and after unknown deletion; full local/Docker suites passed.
- Scoped claim: every prompt in a missing-session recovery queue is protected from silent deletion/mutation while the queue remains in recovery state.


## 2026-06-13 01:36
- Observation: final orphan review found no data-loss/auto-send blocker after later-item protection, but API/UI conflicts were still confusing.
- Intervention: missing-session recovery queue update/move now fail as recovery conflicts, and queue modal hides after the last queue row deletion regardless of remaining direct orphan evidence.
- Evidence: targeted/full/Docker suites passed.
- Scoped claim: orphan queue recovery now has consistent conflict semantics and does not show a false queue-unavailable error after successful final row deletion.


## 2026-06-13 01:42
- Observation: review showed a mixed orphan queue with one persisted `orphan_recovery` item and later plain items could lose the later items after deleting the marked item.
- Intervention: any explicit recovery deletion in a missing-session queue now propagates `orphan_recovery` to all remaining items.
- Evidence: regression test covers deletion of a persisted recovery item with a plain tail; full local/Docker suites passed.
- Scoped claim: missing-session recovery queues retain queue-level recovery protection until all queued prompts are explicitly resolved.


## 2026-06-13 01:49
- Observation: review found a stale-active race: deleting a recovery head before the session was pruned left a plain tail that cleanup could later drop. It also noted direct unknown markers did not preserve same-session plain queued tails.
- Intervention: recovery deletion now propagates recovery state regardless of whether the session still appears active; direct unknown markers mark same-session queues as recovery before cleanup/sweep.
- Evidence: tests cover stale-active deletion followed by cleanup and direct-unknown orphan queue preservation; full local/Docker suites passed.
- Scoped claim: queued tails associated with recovery evidence remain protected across session-prune timing races.


## 2026-06-13 01:58
- Observation: review found that direct unknown evidence could be cleared or pruned before a same-session queue tail was durably converted to recovery state, letting later cleanup drop the tail.
- Mechanism: `queue_list()` exposed recovery on returned copies, but storage still needed an `orphan_recovery` flag before marker removal.
- Intervention: introduced a shared locked marker helper and invoked it before direct unknown clear, old-marker prune, deleted-session cleanup, and queue sweep; saved queue state whenever marking changed.
- Evidence: regressions cover direct unknown clear and age prune preserving plain orphan queue tails; full local/Docker suites passed.
- Scoped claim: direct unknown markers no longer act as ephemeral-only recovery evidence for same-session queued tails at marker removal boundaries.


## 2026-06-13 02:11
- Observation: clean recovery review found no blockers, but active sessions with recovery-protected queue tails were enforced server-side without a visible row-level recovery signal.
- Interpretation: using `orphan_recovery` for active rows would falsely imply a missing transcript; a separate active `queue_recovery` state preserves the distinction between live session and protected queue.
- Intervention: expose active `queue_recovery`, render a recovery badge, and redirect send/attach/queue affordances toward queue review.
- Evidence: server/source tests, full local/Docker validation, and browser evidence on isolated Docker fake session.
- Scoped claim: active recovery queue barriers are now visible and review-oriented in the UI without marking the active transcript as missing.


## 2026-06-13 02:18
- Observation: clean-room review found that active recovery queues were only UI-blocked; direct `/enqueue` could append behind `orphan_recovery`/`commit_unknown` queue barriers, and active `commit_unknown` queue items were not surfaced as `queue_recovery`.
- Intervention: server enqueue path now rejects when queue recovery evidence is present; active `queue_recovery` reports both `commit_unknown` and `orphan_recovery` queue items.
- Evidence: regressions cover both flags and full local/Docker suites passed.
- Scoped claim: queue recovery barriers are authoritative for enqueue, not merely a browser affordance.


## 2026-06-13 02:25
- Observation: review found a check-then-append race where the queue sweeper could mark a head `commit_unknown` between enqueue's first barrier check and append.
- Mechanism: `enqueue()` held the input lock but the sweeper's promotion path marks queue items under `self._lock` before entering `send()`, so `input_lock` alone was not the commit boundary.
- Intervention: `_queue_append_item_local(..., reject_recovery_barrier=True)` now performs the authoritative recovery check under the same `self._lock` critical section as append.
- Evidence: regression simulates a barrier appearing between the first enqueue check and append; full local/Docker suites passed.
- Scoped claim: enqueue cannot append after a recovery barrier has become visible in the queue state before the append lock is acquired.


## 2026-06-13 02:33
- Observation: review showed a persisted queue ordered `[normal, recovery]` could still auto-promote the normal head, even though the session row was marked recovery-locked.
- Mechanism: promotion only inspected the head; the product invariant is queue-level once any recovery evidence exists.
- Intervention: promotion freezes on any queued recovery/unknown item; direct unknown plus queued tail is reported as `queue_recovery`; internal enqueue helper uses the same protected append path.
- Evidence: regressions cover recovery tails for both flags and direct-unknown queue visibility; full local/Docker suites passed.
- Scoped claim: active recovery queues no longer mutate backend sessions via queue promotion while any queued recovery evidence remains unresolved.


## 2026-06-13 02:42
- Observation: review found active recovery was only an enqueue/promotion barrier; unflagged items in a recovery queue could still be edited, moved, or deleted silently.
- Mechanism: mutation protection was per-item, but recovery evidence creates a queue-level evidence boundary.
- Intervention: queue listing and mutation paths now treat every item in a recovery queue as protected; promotion checks the queue-wide barrier before touching broker state.
- Evidence: regressions cover direct-unknown queues, recovery tails, unflagged item mutation attempts, and no broker polling under recovery tails; full local/Docker suites passed.
- Scoped claim: once an active queue contains recovery evidence, every queued item is preserved from silent mutation until explicit recovery deletion.


## 2026-06-13 03:08
- Claim investigated: ffmpeg video transcoding never worked.
- Mechanisms considered: missing ffmpeg dependency; server transcode failure; route failure; browser choosing the original unsupported container and never invoking preview.
- Evidence: server helper and HTTP route produce H.264/yuv420p MP4 when ffmpeg exists, so the core ffmpeg command/route is viable. Standard Docker lacked ffmpeg, so validation was previously blind. Browser evidence showed `canPlayType("video/x-matroska")` was not discriminating: it led to original MKV metadata with zero dimensions instead of preview use.
- Intervention: install ffmpeg in the isolated Docker sandbox; document ffmpeg/ffprobe as the dependency; use an explicit browser-safe container allowlist (`mp4`, `webm`, `ogg`) so `.mkv`, `.mov`, `.avi`, etc. request the compatible preview first.
- Prediction now supported: incompatible containers trigger server MP4 preview generation deterministically; browser-native types still load original first and retain error fallback.
- Scoped claim: ffmpeg-backed compatible preview generation works for the tested synthetic MKV in isolated Docker/browser. Real large videos, unusual codecs, and production hosts without ffmpeg remain separately scoped.


## 2026-06-13 03:17
- Observation: current Pi idle detection required assistant text or error text. A plausible interrupt log shape is `{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}`, which carries no text but is terminal.
- Interpretation: after interruption, Codoxear could keep Pi busy because neither log idle, broker busy state, nor sessiond's watcher recognized Pi `stopReason: "aborted"` as a turn close.
- Intervention: added `pi_assistant_is_aborted_turn()` and wired it into chat flags (`turn_aborted`), `_compute_idle_from_log()`, broker `_apply_rollout_obj_to_state()`, and sessiond log busy signals.
- Evidence: targeted regressions cover all three state paths; full local/Docker suites passed.
- Scoped claim: Pi logs containing assistant `stopReason: "aborted"` now clear busy/turn state in the tested broker, sessiond, and server log-idle paths. Real live Pi interrupt behavior still needs credentialed/runtime confirmation.


## 2026-06-13 03:25
- Review anomaly: video transcode validation used even `160x90`; `libx264` rejects odd dimensions with yuv420p, so valid odd-size videos still failed.
- Intervention: add `scale=ceil(iw/2)*2:ceil(ih/2)*2` to the ffmpeg path and regress `161x91 -> 162x92` H.264/yuv420p output.
- Review anomaly: sessiond's batch watcher could invert an abort followed by a user message because it applied aggregated end after aggregated user.
- Intervention: apply the last busy/end signal in log order; add order regression.
- Review anomaly: history extraction could classify a text-bearing Pi aborted message differently than live extraction.
- Intervention: suppress Pi aborted messages before text classification in `_single_chat_event`.
- Evidence: affected suites and full local/Docker suites passed.


## 2026-06-13 03:36
- Review residual risk: text-bearing Pi aborted records were fixed for visible chat but could still feed delivery notifications and assistant timestamp accounting.
- Intervention: suppress `pi_assistant_is_aborted_turn()` before Pi assistant text handling in delivery extraction, sidebar conversation timestamps, and last-chat role timestamps.
- Evidence: targeted regressions and full local/Docker suites passed.
- Scoped claim: text-bearing Pi aborts are no longer treated as final assistant output by the inspected chat, delivery, and timestamp consumers.


## 2026-06-13 03:52
- Observation: clean Claude review found two falsifiable synthetic bugs: `cc_user_text()` treated arbitrary XML-looking user prompts as non-chat, and `turn_duration` could close an unmatched Claude tool-use turn.
- Mechanism: the first bug came from a broad `text.startswith("<") ...` heuristic; the second came from treating Claude `system/turn_duration` as unconditional turn end without tracking tool-use/result pairing.
- Intervention: remove the arbitrary XML suppression while retaining meta/tool-result filtering; add Claude tool-use/result ID tracking in log-idle and broker busy paths.
- Evidence: targeted tests now preserve `<task>summarize</task>` in chat and keep unmatched `tool_use` + `turn_duration` busy; full local and Docker suites passed.
- Scoped claim: synthetic Claude transcript/busy invariants are stronger, but real credentialed Claude sessions and TUI injection/interrupt behavior remain unproven.


## 2026-06-13 04:00
- Review anomaly: the first Claude tool-use pairing fix was tail-local; a large gap could push the original `tool_use` outside the initial 256 KiB scan while a trailing `turn_duration` still marked idle.
- Review anomaly: final assistant text and mixed `tool_result`+text rows weakened the same pairing/transport invariants.
- Intervention: expand ambiguous `turn_duration` tails up to the configured scan budget, refuse final-answer closure while known Claude pending tool IDs remain, and classify any user row containing `tool_result` as transport rather than chat.
- Evidence: targeted regressions cover the 504 KiB tail case, final-without-tool-result, and mixed tool-result/text; full local and Docker suites passed.
- Scoped claim: synthetic Claude log normalization and busy/idle handling now preserve the inspected tool pairing invariants; live Claude credentials/TUI behavior remain unproven.


## 2026-06-13 04:06
- Review residual risks: no-id Claude `tool_result` rows cleared all pending known tool IDs, and exact scan-boundary tail reads could drop a complete first record.
- Intervention: fail closed for known pending Claude tool IDs on malformed result rows; only an unknown-id tool-use sentinel can be cleared by a no-id result. Tail reads now check whether the start offset is already a line boundary before discarding the first line.
- Evidence: targeted regressions cover no-id result after known `tool_use` and exact-boundary scan budget; full local and Docker suites passed.
- Scoped claim: Claude synthetic tool-pairing invariant is now stricter under malformed data and tail-boundary conditions.


## 2026-06-13 04:15
- Review anomaly: final Claude assistant text could be treated as idle/final when a preceding unmatched tool-use was outside the initial tail scan; chat and delivery extraction also ignored pending tool IDs.
- Intervention: treat final text without prior tail context as an ambiguous terminal requiring scan expansion; carry Claude pending tool IDs through chat and delivery extraction; classify final-looking text as narration and keep `turn_end` false while pending IDs remain.
- Evidence: regressions cover large-gap final text, chat final classification, delivery classification, and mixed malformed tool-use; full local and Docker suites passed.
- Scoped claim: inspected synthetic Claude log consumers now agree that known pending tool-use state blocks final/idle semantics.


## 2026-06-13 04:26
- Review anomaly: the stateful Claude pending-tool invariant was not actually applied to positioned chat pages/live deltas; `_single_chat_event()` and one-record extraction paths still emitted `final_response` for unresolved tool turns. Voice delivery had the same problem across offset deltas.
- Intervention: statefully classify positioned records, seed live/chat and voice delivery deltas from prior log context, and keep split-window final text as narration while pending tool IDs remain.
- Evidence: regressions cover tail page, live full and split deltas, delivery split deltas, and server voice observer behavior; full local and Docker suites passed.
- Scoped claim: inspected chat API and voice delivery paths now share the Claude pending-tool invariant under synthetic logs.


## 2026-06-13 04:35
- Review anomaly: helper-level live delta behavior was fixed, but the actual server live route still called unseeded extraction functions; idle scan also stopped too early when the tail contained later Claude context but not the turn-start/tool-use row.
- Intervention: seed actual live route extraction from prior Claude tool context; require the human user turn start, not merely assistant context, before trusting a terminal-looking Claude final/duration row without expanding.
- Evidence: route source guard, split live/delivery regressions, large later-context idle regression, and full local/Docker suites passed.
- Scoped claim: inspected server live route now participates in the Claude pending-tool invariant under synthetic split-window logs.


## 2026-06-13 04:41
- Review anomaly: a >8 MiB Claude tool-result row could hide an older unresolved sibling tool-use from context seeding and idle scanning.
- Mechanism: the seeding helper used a fixed backward byte budget; idle returned the last visible terminal state when scan budget expired without a turn start.
- Intervention: make current-turn context seeding scan back to the human user row by default; make idle fail closed if a terminal-looking row remains contextless at budget.
- Evidence: regression with 9 MiB tool-result plus unresolved sibling tool passes for seeded live delta and idle; full local/Docker suites passed.
- Scoped claim: the inspected split-delta chat/delivery paths no longer have an 8 MiB context-seeding limit for current Claude turns, though pathological full-log scans may be more expensive on enormous current turns.


## 2026-06-13 04:50
- Review anomaly: unbounded current-turn scanning fixed split-delta correctness but made no-op live polls scan large current turns; malformed idless tool-use state was also boolean rather than count-like.
- Intervention: seed prior Claude context only when a live delta contains records; represent each idless tool-use with a distinct unknown sentinel and consume one sentinel per idless result.
- Evidence: regressions verify EOF live delta does not call the context scanner, and multiple idless tool-use calls remain pending after one idless result; full local/Docker suites passed.
- Scoped claim: no-op live polling avoids the introduced unbounded scan while non-empty split deltas keep full current-turn correctness.


## 2026-06-13 04:56
- Review anomaly: idless Claude tool-use sentinels were unique only within one assistant row, so malformed calls split across rows collided and one idless result cleared both.
- Intervention: include the parsed row identity in unknown sentinel IDs and add split-row regressions across chat/delivery/positioned/idle and broker paths.
- Evidence: targeted regressions and full local/Docker suites passed.
- Scoped claim: malformed idless tool-use tracking is count-like across the inspected per-scan/per-process state, not just within a single row.


## 2026-06-13 05:08
- Review anomaly: broker pending-call state was memory-only and could miss tool-use rows written before log registration; default log idle also failed closed forever for large but fully resolved Claude turns.
- Intervention: move a lightweight Claude current-turn scanner into `cc_log`, seed broker pending state at log bind, and make `_compute_idle_from_log()` use exact current-turn reconstruction for Claude-shaped logs.
- Evidence: broker bind regression remains busy after a pre-bind unresolved tool; 9 MiB resolved output is idle, while 9 MiB output with unresolved sibling remains busy; full local/Docker suites passed.
- Scoped claim: inspected synthetic broker/log-idle Claude paths now reconstruct current-turn tool state from logs rather than depending only on in-memory observation or small tails.


## 2026-06-13 05:17
- Review residual: broker log bind seeded pending tools but not active non-tool Claude turns such as user/thinking rows, leaving broker `/state` idle while log idle was busy.
- Intervention: create shared current-turn state reconstruction in `cc_log`; broker sets busy/turn_open when reconstructed idle is false even with no pending tools; rollout idle uses the same helper.
- Evidence: broker bind regressions cover thinking-only and pending-tool active turns; large resolved/unresolved idle regressions still pass; full local/Docker suites passed.
- Scoped claim: inspected broker bind state now agrees with log-derived Claude current-turn busy state for synthetic active turns.


## 2026-06-13 05:24
- Review anomaly: exact broker log-bind reconstruction handled active turns but did not close a stale pre-bind busy state when the log was already idle/final.
- Intervention: apply idle seeds as well as active seeds during Claude log bind; a reconstructed idle current turn calls `_close_turn_state()`.
- Evidence: regression binds a final Claude log to a previously busy broker state and verifies busy/turn_open clear; full local/Docker suites passed.
- Scoped claim: inspected broker bind now reconciles stale busy, active non-tool, and pending-tool Claude current-turn states.


## 2026-06-13 05:34
- Review anomaly: unknown sentinel uniqueness depended on Python object IDs and could collide across broker watcher batches; top-level `toolUseResult` was hidden as text but not treated as a result.
- Intervention: generate UUID-backed unknown sentinels; route all result cleanup through a shared top-level-aware helper with ID extraction and safe single-pending cleanup.
- Evidence: GC/id-reuse regression keeps six idless tools distinct; top-level `toolUseResult` regressions clear pending and allow final response; full local/Docker suites passed.
- Scoped claim: inspected malformed Claude result handling is more robust, but exact real `toolUseResult` schema still needs live-log confirmation.


## 2026-06-13 05:39
- Observation: focused adversarial review of Claude synthetic hardening at `089a485` found no blockers after many prior falsifications.
- Supported claim: under inspected synthetic Claude log shapes, pending/active current-turn state now gates final/idle semantics across broker, log idle, chat helpers, positioned live/tail/history, actual live route, and delivery split deltas.
- Scope limit: this is not live credentialed battle-testing. Real Claude `toolUseResult` schema and large-turn performance remain uncertain and need live-log or profiling evidence.


## 2026-06-13 06:01
- Observation: the backend already exposed GTD-like state (`blocked`, `snoozed`, recovery/unknown), but the sidebar rendered a flat list, so users had to infer action state from small badges.
- Intervention: add client-only section headers/counts for Needs review, Now, Waiting, Later using existing session fields; do not add persistence, collapse behavior, or new priority semantics.
- Evidence: isolated browser DOM showed four sections with one synthetic session each, including unknown-send recovery, normal, blocked, and snoozed rows. Full local and Docker validation passed after one non-reproducible unrelated Pi token mock anomaly was checked.
- Scoped claim: the sidebar now visibly communicates GTD state for returned session rows without changing backend ordering semantics inside each group.


## 2026-06-13 06:06
- Review result: focused sidebar review found no blockers. It identified weak header accessibility as a non-blocking gap.
- Intervention: add semantic heading role/level and count-aware ARIA labels to group headers while keeping visual UI unchanged.
- Evidence: focused checks and full local/Docker validation passed.

## 2026-06-13 06:15 — 304 fast path required stable priority payloads
- Observation: Client `api()` already sent `If-None-Match` for `GET /api/sessions`, but `refreshSessions()` could not distinguish a cached 304 response from a fresh 200 body and therefore rebuilt the sidebar even when unchanged.
- Additional anomaly: isolated raw API evidence initially showed repeated 200 responses despite `If-None-Match`. Payload diff localized churn to continuously decaying `time_priority`, `base_priority`, and `final_priority` floats.
- Interpretation: A client-only fast path would be mostly inert while priority floats changed every poll. Stabilizing priority payloads in short buckets is necessary for server ETags to represent meaningful session-list changes.
- Intervention: Mark 304 cached responses with a private `Symbol` and early-return from `refreshSessions()` before DOM/defaults mutation; bucket sidebar priority elapsed time by default 10 seconds before emitting priority floats.
- Evidence: Focused source/runtime checks passed; isolated raw API returned 304 inside a bucket; browser evidence showed one `/api/sessions` 304 and zero `.sessions` child-list mutations during the no-change poll window.
- Scoped claim: Unchanged session polls can now avoid sidebar rebuilds within the priority bucket. Priority ordering still decays over time, but ETags may legitimately change at bucket boundaries or when any session/log/sidebar state changes.

## 2026-06-13 06:26 — 304 fast path must respect deferred renders
- Observation: A fresh-context critic identified a lost-update path: mobile swipe deferral can consume a real 200 `/api/sessions` response, update the in-memory cache/ETag, defer the DOM rebuild, and then receive 304 on the close-triggered refresh. A naive 304 early return leaves the sidebar stale until an unrelated later 200.
- Revised mechanism: 304 means the server payload matches the client cache, not that the DOM already reflects the cache. `swipeRefreshDeferred` is evidence that DOM application is still pending.
- Intervention: Restrict the 304 no-op early return to `!swipeRefreshDeferred`; when deferred rendering is pending, render from `latestSessions` instead of fetching/mutating default state again.
- Evidence: Focused source tests now pin this path, and postfix browser evidence still shows ordinary same-bucket 304 polls perform zero sidebar mutations.
- Scoped claim: The no-op fast path is safe for already-applied session payloads; deferred mobile-swipe payloads are treated as not yet applied and can still render after a 304.

## 2026-06-13 06:32 — Deferred state is an application-state invariant, not a fetch-state invariant
- Observation: The first deferred-refresh fix still cleared `swipeRefreshDeferred` before the close-triggered fetch. This confused transport freshness (304) with DOM application state and left the original stale-DOM mechanism alive.
- Revised mechanism: `swipeRefreshDeferred` must remain true until a render path has actually begun applying the cached payload. It is not merely a signal to schedule a refetch.
- Intervention: Preserve the flag through `closeOpenSwipe()`; let `refreshSessions()` bypass the 304 early return while the flag is true and clear it only after `sessionsWrap.innerHTML = ""` on the deferred render path.
- Evidence: Focused tests now pin the close-path lifecycle; isolated mobile-browser reproduction showed old alias during open swipe after a real 200, then the new alias after closing the swipe with 304 resources present.
- Scoped claim: The session-list fast path now distinguishes three states: server payload unchanged and DOM already applied (safe no-op), server payload changed (200 apply/defer), and server payload unchanged but DOM application still deferred (render cached payload).

## 2026-06-13 06:35 — No-blocker scoped claim for session poll fast path
- Observation: Final clean-room review found no blockers after the deferred-refresh flag lifecycle correction.
- Scoped claim: Under the reviewed code paths, same-bucket unchanged `/api/sessions` responses avoid sidebar/default mutation, while mobile swipe-deferred cached payloads still render after close even when the follow-up fetch returns 304.
- Remaining uncertainty: the guarantee is supported by source tests, full suites, isolated browser reproduction, and clean-room review, but not by a dedicated JS DOM unit-test harness or exhaustive refresh concurrency testing.

## 2026-06-13 06:39 — Refresh serialization preserves cache/application ordering
- Observation: Clean-room review of the fast path left a residual race risk: overlapping `refreshSessions()` calls could let `/api/sessions` responses and ETag cache updates apply in completion order unrelated to caller intent.
- Mechanism: Trying to discard by request start order is unsafe because an older-started response can observe a newer server state than a later-started 304. Serializing the request stream is the lower-risk invariant: there is only one `/api/sessions` response allowed to update ETag/cache/sidebar state at a time, and any intervening caller is represented by a queued follow-up refresh.
- Intervention: Split the body into `refreshSessionsOnce()` and make public `refreshSessions()` coalesce concurrent callers through `sessionsRefreshInFlight` / `sessionsRefreshQueued`.
- Evidence: Focused source tests pin the wrapper, browser evidence preserves same-bucket 304/no-mutation behavior, and full local/Docker suites passed.
- Scoped claim: Sidebar session refreshes no longer overlap at the client application layer; queued refreshes preserve final-state convergence without out-of-order response application.

## 2026-06-13 06:43 — No-blocker scoped claim for refresh serialization
- Observation: Clean-room review found no blockers in the serialized `refreshSessions()` wrapper and confirmed auth failures still propagate to all concurrent waiters.
- Scoped claim: Current session-list GET callers go through the serialization wrapper, so `/api/sessions` ETag/cache/sidebar updates are applied in a single client-side sequence with queued follow-up refreshes for concurrent demand.
- Remaining uncertainty: There is no dedicated JS async race harness; if a refresh fails transiently while queued demand exists, recovery depends on the next timer/manual caller rather than an immediate retry.

## 2026-06-13 06:47 — Loading feedback improves perceived transcript latency without changing transcript state
- Observation: `openSession()` cleared the chat and then awaited `/messages/tail`; without a cached tail, users could see a blank transcript during slow tail fetches.
- Mechanism: A loading indicator is safe if it is explicitly non-transcript, excluded from message-row calculations, and removed by authoritative transcript rendering.
- Intervention: Added a `typing-row` loading message only for no-cache opens; cached tails still render immediately, and `renderSessionTail()` / `renderPendingTranscriptSlot()` clear the loading row through existing DOM reset paths.
- Evidence: Focused tests pin non-transcript class and cache-aware call site; browser evidence with a delayed tail fetch showed the loading row appears while pending and disappears after real transcript content renders; full local/Docker suites passed.
- Scoped claim: No-cache session opens now provide visible loading feedback without adding transcript events or changing live/history cursor semantics.

## 2026-06-13 06:50 — No-blocker scoped claim for loading feedback
- Observation: Clean-room review found no blockers in the loading feedback tranche and confirmed the row is excluded from transcript/search/history state by `typing-row` treatment.
- Scoped claim: The loading indicator is UI feedback only; authoritative transcript, pending-bind, and failed transcript paths remain the source of truth and clear the indicator through existing DOM reset paths.
- Remaining uncertainty: There is no explicit failed-tail error state in this tranche; if the tail request fails, the loading row may remain until another action/poll changes the transcript state.

## 2026-06-13 06:58 — Tail-load failure is now explicit UI state
- Observation: The transcript loading tranche left a residual failure mode: if initial `/messages/tail` failed, `Loading transcript…` could remain without explaining the failure or retry path.
- Mechanism: A failed tail fetch should not be represented as a transcript event, but it should replace the loading indicator with visible error feedback. Auth loss is a different mechanism and should route to login cleanup instead of an in-chat error row.
- Intervention: `openSession()` now classifies tail-load failures: stale generation/session changes are ignored, 401 runs `handleAppAuthLoss()`, and other errors render a non-transcript alert row. Reselecting the conversation retries through the existing `openSession()` path.
- Evidence: Focused tests pin the non-transcript row and failure handling; browser evidence demonstrates synthetic tail failure -> explicit error -> successful retry; full local/Docker suites passed.
- Scoped claim: Initial transcript tail failures no longer leave an indefinite loading state; they are visible, non-authoritative UI feedback with a manual retry path.

## 2026-06-13 07:04 — Error feedback must not destroy valid cached evidence
- Observation: Clean-room review falsified the first tail-error implementation. The row itself was non-transcript, but rendering it by clearing the DOM destroyed valid cached transcript rows, which are the current evidence available to the user when the authoritative refresh fails.
- Revised mechanism: Tail-load failure has two distinct states: no prior transcript UI exists, where an error row may replace the loading row; and cached transcript UI exists, where the failure should be additive feedback that the refresh failed while preserving the last valid transcript evidence/search/history rows.
- Intervention: `renderTranscriptLoadError()` now accepts `preserveTranscript`; `openSession()` passes the flag based on whether `applyCachedTail()` ran. Preserve mode appends an excluded error row without clearing transcript DOM or older-state.
- Evidence: Browser reproduction of cached-success -> failed-refresh now preserves one non-typing transcript row and appends exactly one non-transcript error row; focused/full local/Docker tests passed.
- Scoped claim: Initial tail failures are explicit without hiding already-rendered cached transcript evidence.

## 2026-06-13 07:10 — Auth loss outranks stale generation
- Observation: Clean-room review identified that a stale in-flight tail request returning 401 could be ignored before auth cleanup because generation checks ran first.
- Mechanism: Authentication state is global, not scoped to the selected transcript generation. A 401 from any app-owned request is evidence that the browser session is no longer authorized and must trigger cleanup even if the UI moved on.
- Intervention: In initial tail loading and live polling catches, handle `e.status === 401` before checking `pollGen`/selected staleness.
- Evidence: Browser reproduction with delayed stale 401 reached the login screen; focused/full local/Docker validation passed.
- Scoped claim: Tail and live-message polling no longer suppress auth-loss cleanup merely because the request became stale by UI generation.

## 2026-06-13 07:14 — No-blocker scoped claim for transcript tail failure handling
- Observation: Final clean-room review found no blockers after cached-tail preservation and stale-401 ordering repairs.
- Scoped claim: Initial tail failures now have three separated outcomes: auth loss triggers global login cleanup; stale non-auth failures are ignored; active non-auth failures render explicit non-transcript error feedback, preserving cached transcript evidence when present.
- Remaining uncertainty: Jump-to-latest/no-cache refreshes intentionally do not preserve current visible transcript on failure; if that becomes a UX issue it should be treated as a separate tranche.

## 2026-06-13 07:17 — Forced refresh failure can fall back to last valid tail evidence
- Observation: The previous fix preserved cached transcript rows only when `useCache: true` had already displayed the cache. Forced refresh paths such as Jump to latest intentionally bypassed cache display before fetch, so a failed authoritative tail request could still leave only an error row.
- Mechanism: Bypassing cache on the way to an authoritative refresh does not mean the cache loses evidential value if the authoritative measurement fails. A matching cached tail is the best available transcript evidence and should be restored with explicit stale/error context.
- Intervention: On active non-auth tail failure, `openSession()` applies a matching cached tail when no cache was displayed because `useCache` was false, then appends the non-transcript error row in preserve mode.
- Evidence: Browser Jump-to-latest failure reproduction preserved the cached transcript row and added one error row; focused/full local/Docker validations passed.
- Scoped claim: Forced transcript refresh failures now degrade to the last matching cached tail when available rather than blanking the transcript.

## 2026-06-13 07:24 — Cached fallback is safe only for user-forced refresh, not identity recovery
- Observation: Clean-room review falsified the broad forced-refresh fallback. `useCache:false` includes two mechanisms: user-requested freshness (Jump to latest), where cached fallback is acceptable if the fresh measurement fails; and identity-mismatch recovery (409/log-path change), where cached fallback could resurrect the stale identity that the recovery was meant to replace.
- Revised mechanism: Cache fallback must be an explicit caller contract, not inferred from `useCache:false` alone.
- Intervention: Added `fallbackToCacheOnFailure`, default false; only Jump to latest opts in.
- Evidence: Source tests pin the option and automatic no-fallback paths; browser evidence confirms the opt-in Jump path still degrades to cached transcript plus error; full validations passed.
- Scoped claim: Forced refresh fallback now applies only where the caller explicitly accepts last-known transcript evidence after a failed freshness attempt.

## 2026-06-13 07:31 — Tail cache identity must come from transcript payloads
- Observation: The gated fallback still depended on `tailCacheMatchesSession(cache, sessionIndexEntry)`, but cache identity itself had been recorded from sidebar metadata. If sidebar metadata lagged behind the authoritative transcript response, the cache identity could encode stale UI knowledge rather than the transcript's actual source.
- Revised mechanism: The strongest identity evidence for a cached transcript is the `/messages/tail` or `/messages/live` response that produced those events. Sidebar metadata is only a fallback when the payload lacks identity fields.
- Intervention: Tail snapshot creation now stores identity from response payload fields first; live append updates pass the live response as `identityData`.
- Evidence: Source tests pin data-first identity use and full validations passed.
- Scoped claim: Tail cache identity now follows the transcript payload that produced cached events, reducing stale-sidebar false matches for cache fallback.

## 2026-06-13 07:35 — No-blocker scoped claim for forced tail fallback
- Observation: Final clean-room review found no blockers after explicit fallback gating and data-first tail cache identity.
- Scoped claim: Jump-to-latest can degrade to the last matching cached tail when the fresh tail request fails, while automatic identity recovery avoids cached fallback and tail cache identity follows transcript payload identity.
- Remaining uncertainty: The match check on failure still depends on latest client session metadata because a failed authoritative request supplies no new identity. This is acceptable for opt-in Jump fallback but should not be broadened without stronger server evidence.

## 2026-06-13 07:50 — Save conflict recovery should preserve the draft and refresh only by explicit reload
- Observation: The server already returned precise stale-version conflict evidence for `/file/write`, but the client only showed conflict text. Repeated saves would reuse the stale `activeFileVersion` and repeat the conflict without a recovery affordance.
- Mechanism: A 409 means the editor draft and disk file have diverged. The safe client actions are to keep editing the draft or explicitly discard it and reload the current disk version; blind overwrite would hide uncertainty and was not added.
- Intervention: Conflict status now renders explicit Reload/Keep actions; reload goes through the existing file-read path after confirmation, and keep-editing leaves dirty draft/version state unchanged.
- Evidence: Source tests pin action wiring and absence of overwrite; isolated API evidence confirms the server conflict shape; full local/Docker validations passed. Browser UI evidence is limited because Monaco timed out in the sandbox and forced read-only fallback.
- Scoped claim: File save conflicts are no longer a dead-end stale-version loop in the client UI; users get explicit safe recovery choices.

## 2026-06-13 07:57 — File save conflict actions and cleanup must be operation-owned
- Observation: Clean-room review falsified the first file conflict UI: buttons captured only path, and save `finally` unconditionally touched global file UI state. Both allowed an old save/conflict to act on a different active file/session or a newer save.
- Revised mechanism: A save operation is identified by session, path, and a client operation token. Conflict recovery actions are valid only while that same session/path is active, and pending-state cleanup is valid only for the active save token.
- Intervention: Bound conflict actions to `saveSessionId` + `savePath`, and added `fileSaveSeq` / `activeFileSaveToken` ownership to save completion cleanup.
- Evidence: Source tests pin session/path guards and token cleanup; focused/full local/Docker validations passed.
- Scoped claim: File conflict recovery no longer lets stale conflict buttons or stale save completions mutate a different active file/session or newer save operation.

## 2026-06-13 08:07 — Save conflict safety requires server atomicity and post-await ownership
- Observation: Clean-room review found that UI-only conflict handling was insufficient: two same-version writes could interleave on the threaded server, and post-await client paths still needed captured ownership checks.
- Revised mechanism: The authoritative no-blind-overwrite invariant lives on the server compare-and-write boundary. Client conflict actions are UX recovery, but the server must serialize check+replace per file path.
- Intervention: Added per-file write locks around existing-file read/version-check/write. Client save operation ownership remains token/session/path based, with source tests covering the update path.
- Evidence: Source tests pin lock placement and client ownership; focused/full local/Docker validations passed.
- Scoped claim: Existing-file writes now serialize the stale-version comparison and atomic replace within this server process, preventing same-version concurrent blind overwrite races in the supported single-process server model.

## 2026-06-13 08:12 — No-blocker scoped claim for file conflict recovery
- Observation: Final clean-room review found no blockers after session/path conflict action binding, save-token cleanup ownership, and server compare/write locking.
- Scoped claim: In the supported single-process server model, stale-version file saves now produce safe client recovery actions and server-side check/write serialization instead of blind overwrites or stale UI mutation.
- Remaining uncertainty: The server lock does not protect against multiple Codoxear server processes or external writers after the server read but before replace; this is acceptable for current single-process architecture but should be documented if multi-process serving is introduced.

## 2026-06-13 08:16 — Transcript retry uses the existing selection path
- Observation: Transcript load error recovery previously required reselecting the same session from the sidebar, which is awkward on mobile and when the sidebar is hidden.
- Mechanism: Retry is safe if it is scoped to the still-selected session and delegates to `openSession()`, because that path already handles auth loss, stale generations, cached tails, loading, and explicit error state.
- Intervention: Added selected-session guarded Retry button inside the non-transcript error bubble.
- Evidence: Browser evidence demonstrates synthetic failure -> Retry -> transcript recovery; source/full local/Docker validations passed.
- Scoped claim: Users can now recover from transient transcript load errors inline without introducing a new transcript fetch semantics path.

## 2026-06-13 08:20 — No-blocker scoped claim for inline transcript Retry
- Observation: Clean-room review found no blockers in the inline transcript Retry tranche.
- Scoped claim: Transcript load errors now expose a selected-session guarded, non-transcript Retry action that reuses the existing `openSession()` path and clears on successful authoritative render.
- Remaining uncertainty: Accessibility was checked structurally via button text/title/aria-label, but not with a real screen reader.

## 2026-06-13 08:26 — Cache only non-session constants on `/api/sessions`
- Observation: Client ETag/304 avoids transfer and DOM work, but the server still recomputed launch defaults, static asset version, and tmux availability on every `/api/sessions` request before it could produce a 304.
- Mechanism: These helpers are not per-session live state; they can be cached with file-signature invalidation or short TTL without changing session/busy/log semantics. `list_sessions()` remains uncached because it observes live broker/log/git state.
- Intervention: Added signature/TTL caches for launch defaults, static asset version, and tmux availability only.
- Evidence: Targeted cache test shows launch-default reuse, deep-copy isolation, and invalidation; existing static asset tests still pass; full local/Docker suites passed.
- Scoped claim: Repeated `/api/sessions` polls avoid repeated config/static/tmux helper work while preserving live session-list computation and visible invalidation on file changes represented by mtime/size signatures.

## 2026-06-13 08:31 — Static asset version must remain content-derived
- Observation: Clean-room review found a counterexample to static asset version memoization by mtime/size: timestamp-preserving same-size content replacement can keep a stale version.
- Revised mechanism: Unlike launch defaults, static asset version is itself a content identity contract for cache busting. Approximate file signatures are insufficient unless paired with content reading.
- Intervention: Removed static asset version memoization; `_static_asset_version()` again reads bytes to compute the hash. Launch defaults and tmux availability remain cached because their invalidation semantics are weaker and scoped to `/api/sessions` display/config defaults.
- Evidence: Focused/static/full local/Docker validations passed after removal.
- Scoped claim: `/api/sessions` no longer risks stale `app_version` from mtime/size-preserving static content changes.

## 2026-06-13 08:34 — No-blocker scoped claim for session constants memoization
- Observation: Final clean-room review found no blockers after removing static asset version caching.
- Scoped claim: `/api/sessions` now avoids repeated launch-default and tmux probe work without caching live session rows or weakening static asset cache-busting identity.
- Remaining uncertainty: Launch-default cache invalidation remains signature-based, not content-hash based; this is accepted because launch config changes are normal file writes with mtime/size changes in the supported workflow.

## 2026-06-13 08:56 — Preference storage is optional, not a startup dependency
- Observation: Fresh review found unguarded `localStorage` calls during startup and runtime preference handling. In storage-denied browsers this could throw before session rendering and be misreported as an inability to contact the server.
- Mechanism: selected session, sidebar state, notification/announcement toggles, New Session choices, and file-view mode are convenience preferences; none are authoritative server/session state. Failing to persist them should degrade to defaults, not abort app boot.
- Intervention: Centralized browser storage access behind catching wrappers and replaced direct calls.
- Evidence: VM tests exercise getter/method exceptions; source tests ensure selected-session/New Session paths use wrappers; full local/Docker suites pass; real Chromium with a pre-scripted throwing `localStorage` getter still reached the authenticated main UI without storage/security errors.
- Scoped claim: Browser Web Storage denial or quota errors no longer prevent Codoxear from starting and using the main UI; the affected preference persistence may be lost for that browser context.

## 2026-06-13 08:59 — No-blocker claim for optional browser storage
- Observation: Clean-room review found no blockers and confirmed no remaining direct localStorage references outside the helper.
- Scoped claim: storage-denied/quota-error browsers should still reach the main Codoxear UI; only convenience preferences are degraded.
- Remaining uncertainty: behavior with pre-existing live sessions under denied storage was not separately browser-tested, but source flow still selects from `/api/sessions` independently of storage persistence.

## 2026-06-13 09:08 — File picker orientation without expanding UI surface
- Observation: Local-first file search was already implemented, but the no-query picker still flattened changed, mentioned, and recently opened candidates into one undifferentiated list and refetched candidates on each open.
- Mechanism: These candidate sources have different user meanings: changed files support review, mentioned files support transcript follow-through, and recently opened files support continuity. Labeling them can improve orientation without adding a new panel.
- Intervention: Added source metadata, compact section dividers for no-query menus, and a short session/key-based cache. Search menus stay flat because source sections compete with score ordering during query refinement.
- Evidence: VM test shows a second same-key refresh reuses cache while forced refresh refetches; focused/full/Docker suites pass.
- Scoped claim: Reopening the file picker shortly after candidate discovery avoids redundant changed-file requests for the same loaded session state, and the no-query menu now communicates why candidates are present.
- Remaining uncertainty: This tranche is validated structurally/with VM behavior, not by a real browser screenshot with populated fake sessions.

## 2026-06-13 09:14 — File picker cache must not become an authority
- Observation: Clean-room review falsified part of the initial file-picker claim. The candidate cache did not just affect display: cached `changed` flags could choose diff-vs-file mode, and query ordering still privileged known candidates over better-scored server matches.
- Revised mechanism: candidate cache is safe only if it remains a convenience layer. Fresh git state may influence review-oriented diff defaults; cached git state must not.
- Intervention: Search ordering now follows score first. Cache hits explicitly mark git state stale, and `resolveFileOpenMode()` only treats candidate `changed` as authoritative when the candidate set came from a fresh `/git/changed_files` response.
- Evidence: Targeted VM/source tests cover the critic's counterexamples; full local and Docker suites passed after repair.
- Scoped claim: The file picker may show cached candidate labels briefly, but stale cache metadata no longer decides whether opening a file enters diff mode.

## 2026-06-13 09:17 — No-blocker scoped claim for file-picker candidate UX
- Observation: Clean-room re-review found no blockers after score-first ordering and non-authoritative cache repair.
- Scoped claim: File-picker candidate sections and short cache improve orientation/reopen behavior without making cached candidate metadata authoritative for file open content or diff-mode selection.
- Remaining uncertainty: A real populated browser screenshot was not captured; evidence is source/VM behavior plus full local/Docker suites and adversarial review.

## 2026-06-13 09:20 — Smooth scroll must not perturb live-tail tracking
- Observation: Recon noted raw bottom jumps are visually jarring, but smooth scrolling everywhere risks perturbing `isNearBottom()`/auto-scroll logic during live appends.
- Mechanism: User-triggered `Jump to latest` is a navigation action and can be smooth; live-tail autoscroll is a control-loop correction and should remain immediate.
- Intervention: Made smooth bottom scrolling opt-in and used it only in `jumpToLatest()`, with reduced-motion fallback.
- Evidence: Source tests pin a single smooth bottom caller; full local/Docker suites passed.
- Scoped claim: Jump-to-latest is less abrupt for users who allow motion, while live-tail autoscroll semantics remain instant.

## 2026-06-13 09:31 — Smooth jump must be attached to the render that scrolls
- Observation: Review found the first smooth-scroll implementation was probably ineffective because authoritative tail render had already queued instant scrolls before `jumpToLatest()` queued a smooth scroll.
- Revised mechanism: The scroll behavior belongs to the render path that owns bottom correction. Adding a later smooth scroll after an instant bottom jump is not a reliable UX change.
- Intervention: `renderSessionTail()` accepts a one-shot `scrollBehavior`; `jumpToLatest()` passes `tailScrollBehavior: "smooth"` into `openSession()`. Default live-tail/render callers remain instant.
- Evidence: Focused source tests pin single smooth option flow through Jump-to-latest; full local and Docker suites passed.
- Scoped claim: User-triggered Jump-to-latest now requests smooth behavior on the authoritative tail-render scroll, while live-tail autoscroll remains immediate.

## 2026-06-13 09:37 — Smooth jump requires scheduler-level propagation
- Observation: Re-review found the previous repair still allowed instant scroll scheduling inside `rebuildDecorations()` and `setTyping()` to neutralize smooth navigation.
- Revised mechanism: Smooth Jump-to-latest is not a property of `renderSessionTail()` alone; it must be propagated to every bottom-scroll scheduler in the synchronous tail-render path before animation frames run.
- Intervention: Propagated the scroll behavior through decoration rebuild and typing insertion; default live-tail scheduling remains instant by default argument.
- Evidence: Targeted source tests now cover the scheduler-level path; full local and Docker suites passed.
- Scoped claim: For Jump-to-latest tail renders, the first scheduled bottom correction now carries `smooth`; for live/default paths the correction remains `auto`.
