## Objective
Continue bounded structural/frontend refactoring on `/home/yiwen/codex-web-product-recovery`, branch `recovery/product-gaps`, while preserving the product-gap fixes already validated in `recon/refactor-entry-checkpoint.md`. Do not merge or promote it to `/home/yiwen/codex-web` or `main` without explicit user approval.

## Workbench
1. Completed tranche: file-picker matching/scoring helpers now live in `codoxear/static/app_file_helpers.js` at functional commit `4288221 extract file picker helpers`; checkpoint docs are recorded separately from the functional change.
2. Next bounded candidate: extract only pure chat-search display formatting helpers if the ownership boundary remains narrow. Candidate pure functions are `compactChatSearchSnippet(text, query, limit = 96)` and `chatSearchTranscriptHint(match, query)`; keep `rowSearchText()`, rendered-row matching, search timers, transcript search API calls, loaded/all count state, DOM status updates, focus/navigation, and load-older actions in `app.js`.
3. Prefer extending `app_display.js` only if the boundary stays deterministic string formatting; add fail-loud display-helper wiring, wrapper-preserving call sites, real-module VM tests for whitespace collapse, min-limit clamping, query-centered snippets, default-limit behavior, role labels, and blank-hint handling, plus source/boundary coverage that `rowSearchText()` remains app-owned.
4. Run focused Docker tests, full Docker suite, clean diff review, and exactly one clean-room review before any functional commit.
5. Commit functional changes and memory/checkpoint docs separately, then refresh this list for the next tranche.

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
