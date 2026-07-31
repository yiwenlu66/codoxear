# GitHub PR Review / Cherry-pick Triage

Date: 2026-06-12
Branch: `develop`
Scope: open GitHub PRs #10-#22 (excluding missing #20) plus local PR-ish branches.
Method: `gh pr list`, PR metadata, fetched PR heads under `refs/remotes/origin/pr/<n>`, and direct inspection of the topic/top commits. No merges or cherry-picks were performed during triage.

## Global observation

Most open PR branches are stale relative to local `develop`: their GitHub diffs contain tens of thousands of lines and many historical commits already represented in the current codebase under different hashes. Therefore the review unit is **not the whole PR branch**. The safe unit is the final topic commit(s), reimplemented or cherry-picked only after checking current code.

## Decisions

| PR | Title | Decision | Rationale |
|---:|---|---|---|
| #10 | feat(pi): improve session path discovery and abort handling | Defer / mine selectively | Huge stale branch; top commit introduces broad Pi RPC/UI modules and docs beyond current product philosophy. Smaller follow-up commits (`recover piox session detection`, `ignore synthetic pi timestamps`) may be useful after current Pi code is inspected. |
| #11 | Fix broker path sanitization on Python 3.11 | Already incorporated / no action | Current `broker._pi_session_dir_name()` already normalizes into a local variable before f-string/path use. Note: current tests require Python 3.12+ syntax despite pyproject claiming >=3.10; track separately if Python 3.11 support remains desired. |
| #12 | Fix Pi session registration fallback | Accept selectively | Small topic: store Pi declared log path and prefer it once it exists. Compatible with shared broker and directly addresses rollout/log binding. Should be reimplemented with tests against current broker. |
| #13 | Fix Icon URL, Flickering and Cache-Control | Accept selectively | Useful small fixes: relative sidebar icon URL for URL prefixes, include nested logo assets in wheels, avoid rebuilding backend tabs while new-session modal is open, optional static cache headers. Compatible with minimal UI and optimization goals. |
| #14 | Add default hover tooltips for buttons | Accept selectively | Small discoverability improvement for icon buttons; compatible with minimal UI because it adds no visible chrome. Add source coverage. |
| #15 | Fix Pi duplicate session log binding | Accept selectively / combine with #12 | Small topic constrains Pi cwd scanning after a log is already bound. Compatible with rollout-log binding pressure tests; combine with declared-log behavior to avoid duplicate/stale bindings. |
| #16 | Vite + Preact frontend refactor + Pi workspace layering | Reject | Large framework rewrite and hierarchical workspace/file-browser direction conflicts with project philosophy: minimal UI, no nested GTD sidebar/workspace expansion, deliberate chat detail omission. Too risky and not aligned with requested final develop branch. |
| #17 | Handle stale broker sockets without metadata | Accept selectively | Small startup/error-handling fix. Missing sidecars are stale runtime artifacts, not server errors; should remove stale session state quietly. Compatible with startup error handling pressure test. |
| #18 | Improve browser auth compatibility and localize Monaco | Defer / split | Bearer-token auth and vendored Monaco may help browser compatibility/offline use, but PR vendors a large Monaco tree and changes auth semantics. Needs separate product/security review. Do not accept whole branch. |
| #19 | Handle disconnected HTTP clients quietly | Accept selectively | Small robustness fix. Browser disconnects should not produce noisy 500 cascades. Compatible with startup/error handling and slow mobile network use. |
| #21 | Add Claude backend support with interactive prompt UI | Defer / mine for cc parser | Whole PR mixes Claude backend, interactive prompt UI, Pi ask_user, and redesign. The Claude log parser and backend support are useful references, but implementation should follow a smaller shared-broker `cc` design and avoid prompt UI/detail creep unless separately justified. |
| #22 | Fix macOS web session launches | Defer / inspect later | Top commit may be useful for macOS/headless Codex launch and service-tier cleanup, but it changes launch semantics. Need targeted inspection and tests before acceptance. Not first priority for Linux-first Docker validation. |

## Local PR-ish branches

- `pr-5`: appears to contain cwd-default/HOME expansion fix; inspect only if current new-session cwd behavior shows a gap.
- `pr/backend`, `pr/frontend`, `pr/frontend-rebased`: stale/local remnants of earlier PRs. Do not merge wholesale; mine only if a current failing test or product gap points to a specific commit.

## Initial accepted implementation queue

1. #13 packaging/static/new-session/icon fixes.
2. #14 default button tooltip fallback.
3. #19 quiet HTTP disconnect handling.
4. #17 stale sidecar handling.
5. #12/#15 Pi log-binding hardening.
6. #21-derived minimal `cc` backend design after architecture decisions.

Each accepted item should be committed atomically on `develop` and validated in the Docker sandbox. Whole PR branches should not be merged into `develop` without rebase/scope cleanup.
