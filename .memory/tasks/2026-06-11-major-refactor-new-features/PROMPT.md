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

## Continuation correction
- Do not define a small "tranche" as a completion boundary for the user's broad refactor/recovery request.
- Reporting that the broad work is not done is not useful progress; continue with the next constructive intervention unless blocked by a real user decision or irreversible/high-risk action.
- Completion claims require the actual broad recovery/refactor objective to be satisfied, not merely a reviewed subset.


## Unattended-mode operating directive
- Maintain internal Deliverables, Completed, Next actions, and Parked user decisions during long-running work; surface them only when yielding is necessary.
- Default to continuing in the same turn. Yield only when all deliverables are finished, only a true user decision remains, or the next step is irreversible/high-risk and needs explicit confirmation.
- Before yielding, run a clean-room adversarial review with user intent, deliverables, completed evidence, next actions, parked decisions, constraints, and changed artifacts; apply findings before reporting unless the finding itself requires a user decision.
- Avoid trial-and-error loops: use reading, tracing, inspection, causal reasoning, and the strongest available verification. Do not repeat commands or edits without a new discriminating reason.
- User directive: push refactor/product recovery aggressively rather than treating any bounded tranche as a stopping point; continue into the next justified reliability/refactor/product target after each committed checkpoint.

## Long-run unattended directive update
- Maintain internal sections during the run: Deliverables, Completed, Next actions, and Parked user decisions. Keep them internal unless a yield is required.
- Optimize for 8+ hour unattended progress with minimal turns and minimal repetition. Default to continuing in the same turn; do not treat reviewed or committed subsets as a stopping boundary.
- Before each action, reason through the mechanism, failure modes, and verification path. Explore by reading, tracing, inspection, and causal reasoning rather than trial-and-error.
- Resolve crashes, bugs, and design mistakes directly unless the remaining issue is a true user-only decision or an irreversible/high-risk step.
- Use the strongest available verification and do not repeat commands, edits, or analysis without a concrete new discriminating reason.
- Yield only when all deliverables are finished and supported, only a parked user decision remains, or the next step is irreversible/high-risk and needs explicit confirmation.
- Before any necessary yield, run a clean-room adversarial review with user intent, deliverables, completed evidence, remaining next actions, parked decisions, constraints, and changed artifacts; apply findings before reporting unless the finding requires the user.
- Additional user emphasis: refactor aggressively and push product/reliability recovery to the limit; no bounded-action posture.
- Latest user reinforcement: keep pushing refactor/product recovery aggressively (“refactor! refactor!! refactor!!!!! push refactor to the limit!!!!! no bounded action”); do not treat committed or reviewed features as a yield boundary.
