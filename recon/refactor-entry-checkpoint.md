# Refactor-entry checkpoint for `recovery/product-gaps`

Date: 2026-06-15
Branch: `recovery/product-gaps`
Current HEAD: `9c49a3d fix: contain markdown tables`
Protected checkout: `/home/yiwen/codex-web` on `main` was not modified or merged.

This checkpoint records the product-gap recovery state before any broad structural/frontend refactor. It is not merge approval.

## Closed product gaps in this recovery branch

Recent committed recovery checkpoints include:

- Transcript/search reliability:
  - bounded transcript search hint payloads and bounded count hints;
  - older-match search paging from unloaded transcript windows;
  - exact-by-default search-count API semantics, with bounded lower-bound UI hints;
  - search navigation now preserves all-transcript count evidence during Next/Previous navigation so an enabled `0 loaded · N all` path can load the offscreen match;
  - oversized JSONL records skipped by bounded transcript search now mark `match_count_truncated` instead of overstating exactness.
- Transcript/log robustness:
  - malformed sidecars skipped fail-closed;
  - live JSONL partial reads bounded;
  - batch chat extraction shares single-event construction;
  - Claude Code id-less tool-use placeholders generated once.
- Launch/session metadata:
  - fresh tmux launch metadata requires a live broker pid;
  - launch sidecar metadata validation hardened;
  - Pi provider/model launch path now passes explicit custom providers through instead of treating UI defaults as an API whitelist;
  - sidecar metadata schema/capability parsing now lives in `codoxear/sidecar_metadata.py`, while server discovery/refresh call sites keep fail-closed aliases.
- Pi live-backed launch path:
  - `model_provider=anthropic`, `model=claude-haiku-4-5`, `reasoning_effort=low` launched through the web path, accepted a send, bound a log, produced an assistant final response, reached idle, and cleaned up in isolated Codoxear app state.
- File/inline reference UX:
  - file candidates remain visible without git state;
  - equivalent inline refs merge only after inspected identity;
  - failed inline file inspections are not cached as durable facts;
  - file viewer modal focus restored;
  - sessions rooted at `/` can create valid relative descendant files through `/file/write` without the prior root-prefix false rejection;
  - git subprocess/path/pathspec/numstat/worktree helper logic now lives in `codoxear/git_ops.py`, with server wrappers preserving private names and `_run_git` patch seams.
- Queue/send/unattended UX:
  - unattended prompts gate on final assistant turns;
  - mobile composer stop control added;
  - read endpoints remain observation-only and do not promote queued prompts;
  - orphan, queued-recovery, and unknown-send sessions now render an in-chat recovery panel with safe review actions instead of opening to an empty disabled pane.
- Browser/desktop UX:
  - desktop notifications focus the target session;
  - Pi custom provider/model browser behavior now has executable JS/VM coverage;
  - long-transcript per-message copy controls now use a roving active button so the accessibility/tab order has one enabled copy control instead of one repeated control per rendered message;
  - Details diagnostics can be copied from the dialog using only rendered label/value rows, not the raw diagnostics object;
  - Details can open a review-only New Session dialog with copied launch settings from an allowlisted diagnostics subset; Pi provider semantics use actual `model_provider`, not synthetic `provider_choice`, including providerless and sparse-metadata cases;
  - file picker search results highlight exact/fuzzy query matches using DOM text nodes and Unicode-safe folded-index mapping, without changing path identity;
  - markdown fenced code blocks use a light Codoxear-themed surface, and markdown tables wrap/contain normal wide content with internal horizontal scroll only for impossible many-column cases.

## Latest validation evidence

Latest code-validation evidence after the markdown rendering fix:

- Focused markdown rendering validation: `python3 -m pytest tests/test_markdown_tables.py tests/test_static_assets.py -q` -> `22 passed`.
- Browser evidence at 390px: code blocks computed light background/dark text; normal long-token tables stayed within the bubble without scroll; a 20-column table kept the bubble contained and used internal wrapper scroll instead of clipping.
- Full local suite: `python3 -m pytest -q` -> `872 passed, 92 subtests passed`.
- Docker sandbox suite: `scripts/codoxear-docker-sandbox test` -> `871 passed, 1 skipped, 92 subtests passed`.

Recent clean-room reviews returned no blockers after fixes:

- `/tmp/codoxear-pi-provider-ui-behavior-review2.md`
- `/tmp/codoxear-search-navigation-count-review.md`
- `/tmp/codoxear-roving-copy-buttons-review3.md`
- `/tmp/codoxear-root-cwd-resolve-review.md`
- `/tmp/codoxear-oversized-search-review.md`
- `/tmp/codoxear-recovery-panel-review6.md`
- Clean-room critic subagent review of sidecar extraction diff and call sites -> no blocker findings; non-blocking source-test brittleness was reduced before commit.
- Clean-room critic subagent review of Details-copy diff -> no blocker findings for stale-session binding, secret-copy risk, accessibility/focus, or sparse-UI risk.
- Clean-room critic review of file-picker highlight diff found a Unicode slicing bug; folded-index mapping plus `İfoo.py`/emoji regressions fixed it. Re-review found no blockers.
- Clean-room architecture review of git helper extraction found detached-HEAD semantic drift; `git_ops.current_git_branch()` was corrected to preserve `HEAD`. Targeted re-review and critic review found no blockers.
- Clean-room critic review of Details → New like this found Pi provider corruption risks in direct presets, diagnostics provider display, duplicate/recent options, remembered providerless choices, and sparse metadata. Each counterexample was fixed with regressions; final re-review found no blockers for Pi provider corruption, auto-start, focus, or sparse UI behavior.
- Clean-room critic review of markdown rendering first found a hidden-overflow/fixed-layout clipping counterexample for many-column tables. The final implementation uses auto overflow plus auto table layout; re-review found no blockers for clipping, page/bubble overflow, copy semantics, or chat/file-preview markdown paths.

Recent isolated browser evidence:

- Synthetic 180-turn Codex transcript under Docker app state reproduced and then fixed the all-transcript-search Next no-op. After the fix, clicking Next from `0/0 loaded · 1 all` emitted `/messages/search?...order=latest&before=...` and `/messages/history?cursor=...`, then showed `1/1 loaded · 1 all` and `Loaded transcript match` with no captured JS errors.
- The same long transcript had 60 message-copy button nodes but exactly one enabled/tabbable/accessibility-visible copy button after the roving-copy fix. Inactive samples were disabled, `tabIndex=-1`, `aria-hidden=true`, `opacity:0`, `visibility:hidden`, and `pointer-events:none`. Hidden-focus counterexamples with `Alt+↑` and `Alt+Shift+↑` remained false.
- Synthetic recovery fixtures under isolated Docker app state verified the in-chat recovery panel: orphan recovery did not fetch `/messages/tail`, Review queue opened preserved prompts, clearing an unknown marker and deleting queue items kept panel/buttons/focus synchronized, transcript-backed live appends kept the panel as the latest recovery surface, and focused panel actions survived rapid panel rebuilds.
- Pi live backend evidence exists for one current configured provider/model path as described above. Codoxear app/session state was isolated; the backend provider configuration came from the user's existing real Pi environment and was handled without printing secret values.

## Invariants broad refactoring must preserve

Any broad frontend/server refactor must keep these product semantics explicit and mechanically preserved:

1. **Send commit boundary:** HTTP `/send` success means the broker/sessiond path accepted the prompt or returned explicit unknown-commit recovery state; reads must not promote queued prompts.
2. **Unknown commit state blocks unsafe actions:** unresolved direct/queued uncertainty blocks send, enqueue, attach, sweep, reorder, and silent destructive cleanup bypasses; recovery UI may explain/review/clear explicit markers but must not silently resume mutation paths.
3. **Git/file identity:** changed-file paths are repo-root-relative literals; candidate identity is `(gitPath, path)`; path text must not be normalized destructively. Visual highlighting may wrap displayed substrings but must preserve original path strings for titles, copy/open actions, and identity keys. Git helper extraction must preserve literal pathspec handling and existing server wrapper/patch seams.
4. **Inline refs:** ambiguous inline refs route through the identity-aware picker; failed/truncated project search is ambiguity, not uniqueness proof.
5. **Broker state:** `busy` is bool and `queue_len` is nonnegative non-bool int; malformed state is fail-closed, not coerced.
6. **Stale busy override:** stale broker busy can be overridden only with idle log evidence, empty queue, same-log last-send barrier cleared, and a bound log.
7. **Sidecar discovery:** malformed sidecar metadata is skipped/logged; fresh launch metadata requires a live broker pid; stale discovery still tolerates pid placeholders where explicitly allowed. Schema/type/capability parsing belongs in `codoxear.sidecar_metadata`; consumers may prune/skip only through explicit validation failure, not coercion.
8. **Transcript scale:** live JSONL readers stay bounded; impossible-sized partial records may be skipped rather than repeatedly re-read unboundedly.
9. **Search semantics:** count is exact by default only when all records in scope were parseable under the bounded line cap; skipped oversized records make `match_count_truncated` true. Bounded counts are lower bounds; `count_max` is incompatible with `order=latest`; UI hints stay sparse and server-clipped.
10. **Search navigation:** navigation refresh may recompute loaded DOM matches without discarding already-known all-transcript count evidence.
11. **Modal/accessibility focus:** active dialogs must receive focus immediately; focus must not remain in inert/`aria-hidden` content; message-copy controls must not flood tab/accessibility traversal. Dialog copy actions should copy rendered/allowlisted rows rather than hidden raw response objects.
12. **Pi launch providers:** Pi CLI/config is authority for provider names. UI defaults are hints, not an API whitelist; explicit provider/model pairs must not inherit stale bare-model reasoning constraints. Synthetic diagnostics `provider_choice` must not be treated as actual Pi provider state; providerless Pi sessions must remain providerless through copied launch presets, recent model selection, memory, parsing, and start request construction.
13. **Minimal UI philosophy:** keep the topbar sparse; utility controls belong in contextual rails/surfaces, not a generic dumping-ground menu.
14. **No silent fallbacks:** absence, malformed contracts, or unsupported combinations should fail loudly with recoverable UI when possible.
15. **Markdown containment:** code blocks should remain readable in the light UI; tables should wrap normal wide content and use internal scroll only when the column count cannot physically fit without clipping.

## Parked limits and decisions

The branch is stronger than the historical `develop` summary, but these limits remain explicit:

- Merge/promote to `main` still requires explicit user approval.
- Broad structural/frontend refactor is not complete; this checkpoint only defines its entry state.
- Real-browser/manual backend exercise of the Details → New like this button remains incomplete; source/VM tests, full pytest, Docker, and critic review cover the implemented semantics.
- Markdown rendering evidence covers CSS/source tests and headless Chromium fixtures, not real mobile-device or assistive-technology review.
- Codex live response evidence remains incomplete: current work proved the real interactive TUI can be reached with the trust override, but not a full web-send/final-response path.
- Claude Code live response evidence remains incomplete under isolated HOME because first-run theme/onboarding blocked log binding.
- Real mobile-device, assistive-tech, slow-network, huge-transcript, and full live backend lifecycle evidence remain incomplete.
- Smooth scrolling for Jump to latest remains parked until scheduler/runtime harness evidence exists.
- Non-UTF-8 Git filenames are replacement-decoded rather than byte-literal end-to-end.
- Symlink containment checks are pre-open/read/write, not atomic against concurrent local filesystem mutation.

## Recommended next step

A broad refactor may start from this branch only if it treats the invariants above as contract tests. The first refactor tranche should be bounded and evidence-preserving: extract one frontend subsystem at a time, keep behavior tests/source guards green, run the full local and Docker suites after each coherent checkpoint, and retain isolated browser evidence for high-risk UI flows.
