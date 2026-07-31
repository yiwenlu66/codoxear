# Integrated Execution Plan for `develop`

Date: 2026-06-12
Acceptance branch: `develop`

> Status after implementation: this plan has been executed into the `develop` acceptance branch. See `recon/final-acceptance-summary.md` for the current branch contents, validation evidence, deferred risks, and parked user decisions. Historical phase text below is retained as the original execution plan.

## Working claim

The fastest safe path is not a framework rewrite or a whole-PR merge. It is a sequence of small, evidence-backed commits that:

1. tighten the validation boundary,
2. accept compatible small PR fixes,
3. improve mobile/network responsiveness,
4. add long-chat orientation affordances without adding transcript density,
5. rename/recast the idle prompt injector as Unattended mode,
6. add Claude Code support through the shared broker/log abstraction.

## Immediate implementation order

### Phase A — Low-risk accepted PR fixes

1. Packaged/static asset fixes from PR #13
   - Include nested logo SVGs in wheels.
   - Use URL-prefix-safe sidebar icon URL.
   - Avoid rebuilding backend tabs while new-session modal is open.
   - Add optional static cache headers with no-store as default.

2. Default button tooltip fallback from PR #14
   - Adds discoverability for icon-only buttons without visible UI complexity.

3. Quiet HTTP disconnect handling from PR #19
   - Treat client disconnects as transport noise, not server errors.

4. Stale broker sidecar handling from PR #17
   - Missing metadata sidecar should prune stale state, not surface a UI/server failure.

5. Pi log-binding hardening from PR #12/#15
   - Prefer declared Pi log path once it exists.
   - Avoid cwd-based Pi log switching after a session already has a current log.

### Phase B — Network and responsiveness

1. Visibility/adaptive session polling
   - Slow or pause `/api/sessions` polling when the page is hidden.
   - Decouple voice/notification polls from the session-list timer where safe.

2. Session list payload and DOM rebuild reduction
   - Avoid rebuilding stable new-session backend tabs.
   - Consider incremental sidebar rendering after measuring current costs.

3. Static asset cache control
   - Default no-store preserves current development behavior.
   - `CODEX_WEB_STATIC_CACHE=1` enables long-lived immutable cache for packaged/static assets.

### Phase C — Long-chat navigation

1. Previous/next user-message jump
   - Low-risk DOM-local navigation over already-rendered `.msg-row.user` elements.

2. In-chat search over loaded messages
   - Lightweight overlay or collapsible control.
   - Search loaded/visible rows first; server-side full-history search is deferred unless evidence shows local search is insufficient.

3. Sticky/current visible time cue
   - Use existing timestamps without adding a dense log/details panel.

### Phase D — Unattended mode rename

1. User-facing copy hard-switch from Harness mode to Unattended mode.
2. Replace the public route with `/unattended` and expose only `unattended_*` session-list fields.
3. Write state to `unattended.json`; old `harness.json` compatibility is not required per user correction.
4. Rename internals only if the diff remains reviewable; otherwise complete the public API/state migration first and leave internal cleanup for a later commit.

### Phase E — Claude Code (`cc`) support

1. Add `cc_log.py` parser and unit tests from fixture records.
2. Add backend registry entry and path inference/filtering.
3. Add rollout/log normalization and busy/idle support.
4. Add launch defaults and spawn args.
5. Add UI backend tab without prompt-detail creep.
6. Validate with synthetic CC logs first; real CC CLI test only inside Docker/sandbox if credentials and binary are explicitly available there.

## Explicit deferrals / rejects

- Reject PR #16 whole-branch Preact/workspace rewrite: conflicts with minimal UI and no-nesting philosophy.
- Do not merge PR #21 whole branch: useful Claude parser ideas, but interactive prompt UI and redesign are too broad.
- Defer PR #18 whole branch: local Monaco/offline auth changes are large and need separate security/package-size review.
- Defer large server.py module extraction until after user-visible fixes and `cc` design clarify the stable boundaries.

## Validation baseline

- Current Docker sandbox full suite: `357 passed, 2 skipped` after baseline repairs.
- Minimum per commit: relevant targeted tests plus `scripts/codoxear-docker-sandbox test` for backend/route changes.
- UI changes should receive browser evidence from the Docker sandbox when practical.
