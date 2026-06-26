## Objective
Continue bounded structural/frontend refactoring on `/home/yiwen/codex-web-product-recovery`, branch `recovery/product-gaps`, while preserving the product-gap fixes already validated in `recon/refactor-entry-checkpoint.md`. Do not merge or promote it to `/home/yiwen/codex-web` or `main` without explicit user approval.

## Workbench
1. Completed tranche: queue API-payload normalization now lives in `codoxear/static/app_session_helpers.js` at functional commit `35be96c extract queue normalization helper`; checkpoint docs are recorded separately from the functional change.
2. Advisory scout `/tmp/codoxear-next-pure-helper-scout-after-queue.md` found no remaining safe pure-helper extraction candidates under the current constraints.
3. Parked non-candidates: `redactedLaunchErrorText`/`sessionLaunchLabel` are pinned/security-sensitive; `launchPresetProviderChoice` is pinned by launch-dialog source-slicing tests; other remaining inline functions are wrappers, state readers, unused/dead code, DOM/browser-side-effect code, or orchestration/render logic.
4. Do not force another tranche without explicit broader ownership/design approval or a newly identified deterministic argument-only candidate with an obvious helper-module home and validation path.
5. If work resumes, re-anchor on a clean tree, preserve the Docker-only acceptance rule, and run focused Docker, full Docker, clean diff review, and exactly one clean-room review before any functional commit.

## Context
- Active checkout: `/home/yiwen/codex-web-product-recovery`.
- Protected live checkout: `/home/yiwen/codex-web` on `main`; do not edit, merge, restart, or promote it.
- Task memory: `.memory/tasks/2026-06-11-major-refactor-new-features/`.
- Current checkpoint: `recon/refactor-entry-checkpoint.md`.
- Closed work includes Pi busy-after-interrupt repair, Codex live web-send binding, Claude Code closed-log binding/API-error idle path, markdown code/table containment, video preview/transcoding, and bounded frontend helper/static-registry refactors through recent-cwd score formatting.
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
