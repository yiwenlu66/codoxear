## Objective
Prepare `/home/yiwen/codex-web-product-recovery` on branch `recovery/product-gaps` as a validated, reviewable candidate for explicit user approval. Do not merge or promote it to `/home/yiwen/codex-web` or `main` without that approval.

## Workbench
1. Keep the branch clean and reviewable after every tranche: atomic functional commits, separate memory/checkpoint commits, and exact-file staging only.
2. Before more implementation, run a final clean-room/adversarial review against `recon/refactor-entry-checkpoint.md` and current `HEAD` to decide whether any blocker remains.
3. If review finds a blocker, fix the smallest causal scope with Docker-only validation and a follow-up critic.
4. If review finds no blocker, update `recon/final-acceptance-summary.md` with supported claims, Docker evidence, live-backend evidence already obtained, and parked limits; then stop for explicit user approval.
5. Optional only if final review identifies refactor risk as a blocker: choose the next bounded tranche from `/tmp/codoxear-next-refactor-after-perf.md` or a fresh scout, preserving existing invariants.

## Context
- Active checkout: `/home/yiwen/codex-web-product-recovery`.
- Protected live checkout: `/home/yiwen/codex-web` on `main`; do not edit, merge, restart, or promote it.
- Task memory: `.memory/tasks/2026-06-11-major-refactor-new-features/`.
- Current checkpoint: `recon/refactor-entry-checkpoint.md`.
- Closed work includes Pi busy-after-interrupt repair, Codex live web-send binding, Claude Code closed-log binding/API-error idle path, markdown code/table containment, video preview/transcoding, and bounded frontend helper/static-registry refactors.
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
