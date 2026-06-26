## Objective
Continue aggressive product-gap recovery, reliability hardening, and structural/frontend improvement work on `/home/yiwen/codex-web-product-recovery`, branch `recovery/product-gaps`, while preserving the product-gap fixes already validated in `recon/refactor-entry-checkpoint.md`. Do not merge or promote it to `/home/yiwen/codex-web` or `main` without explicit user approval.

## Workbench
1. User explicitly rejected a narrow/bounded stopping posture: proceed thoroughly and aggressively on meaningful product/reliability/frontend gaps rather than stopping at pure-helper exhaustion.
2. Completed pure-helper wave remains valid through `35be96c extract queue normalization helper`; scout `/tmp/codoxear-next-pure-helper-scout-after-queue.md` found no further safe mechanical pure-helper targets, so do not waste motion forcing helper extractions.
3. Next work should attack the highest-value implementable gap with causal evidence: security/reliability invariants, parked evidence gaps that can be converted into tests, or user-visible UX/product failures. Prefer hard gaps over easy structural bookkeeping.
4. Keep safety constraints: recovery checkout only; no protected `/home/yiwen/codex-web` mutation/promotion; no live server/session killing; no secrets; no silent fallbacks; Docker evidence for acceptance claims; functional and docs commits remain separate.
5. For each aggressive tranche: understand ownership/source of truth first, implement the mechanism (not symptoms), run focused Docker plus full Docker when scope warrants, use a clean-room review before functional commit, and checkpoint evidence separately.

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
