## Objective
Continue bounded structural/frontend refactoring on `/home/yiwen/codex-web-product-recovery`, branch `recovery/product-gaps`, while preserving the product-gap fixes already validated in `recon/refactor-entry-checkpoint.md`. Do not merge or promote it to `/home/yiwen/codex-web` or `main` without explicit user approval.

## Workbench
1. Completed tranche: recovery prompt preview formatting now lives in `codoxear/static/app_display.js` at functional commit `57d70db extract recovery prompt preview helper`; checkpoint docs are being recorded separately.
2. Next bounded candidate: extract only pure recent-cwd fuzzy scoring (`fuzzyRecentCwdScore`) if the ownership boundary remains narrow. Keep `recentCwds`, `renderRecentCwdOptions()`, `filteredRecentCwdOptions()`, `newSessionCwdInput`, menu focus/selection, DOM rendering, cwd validation, and dialog actions in `app.js`.
3. Prefer extending an existing helper module only if the boundary is semantically narrow; add fail-loud helper wiring, wrapper-preserving call sites, real-module VM tests for exact/base/token/subsequence scoring, no-query behavior, no-match behavior, and source/boundary coverage.
4. Run focused Docker tests, full Docker suite, clean diff review, and exactly one clean-room review before any functional commit.
5. Commit functional changes and memory/checkpoint docs separately, then refresh this list for the next tranche.

## Context
- Active checkout: `/home/yiwen/codex-web-product-recovery`.
- Protected live checkout: `/home/yiwen/codex-web` on `main`; do not edit, merge, restart, or promote it.
- Task memory: `.memory/tasks/2026-06-11-major-refactor-new-features/`.
- Current checkpoint: `recon/refactor-entry-checkpoint.md`.
- Closed work includes Pi busy-after-interrupt repair, Codex live web-send binding, Claude Code closed-log binding/API-error idle path, markdown code/table containment, video preview/transcoding, and bounded frontend helper/static-registry refactors through recovery prompt preview formatting.
- Remaining parked limits include successful live Claude model-text response, real mobile-device/assistive-tech evidence, slow-network/huge-transcript evidence, smooth Jump to latest, non-UTF-8 Git filename byte-literal behavior, and atomic symlink containment.

## Task specifications
- Docker-only validation is acceptance evidence unless the user explicitly changes this; host validation may be diagnostic only.
- Final claims must distinguish supported behavior from parked limits.
- Do not treat stale prompt lines as active work when `recon/refactor-entry-checkpoint.md` records later evidence closing them.
- Refactors must preserve the invariants listed in `recon/refactor-entry-checkpoint.md`; no silent fallbacks.

## Constraints
- Do not edit `/home/yiwen/codex-web`.
- Do not merge or promote to `main` without explicit user approval.
- Do not touch live sessions, the live server, brokers, or backend CLI processes.
- Do not print secrets, credentials, tokens, private logs, or provider configuration values.
- Do not commit runtime artifacts, sockets, live app state, bulky scratch data, or secrets.
- Do not use `git add -A`, `git add .`, or broad staging.
