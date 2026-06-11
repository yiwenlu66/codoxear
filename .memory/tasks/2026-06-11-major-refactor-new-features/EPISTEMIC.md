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
