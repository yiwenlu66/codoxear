# Issue backlog — reconciled 2026-08-04

This ledger is historical. The current disposition index is `.memory/project/PRODUCT_GAP_STATUS.md` “Reconciled disposition ledger”; it remains the live source of truth for the listed residuals and the real-hardware aesthetic review.

## Current dispositions

### Done

| Item | Disposition | Commit evidence |
| --- | --- | --- |
| ISSUE-1 — Inter font | Deliberate decision: use bare `sans-serif`, no web font. | `39c7ac28` |
| ISSUE-2 — send-now semantics | Direct confirmed-send is unconditional; busy input is steering, not silent enqueue. | `ddae3e62` |
| ISSUE-3 — queue ordering for direct sends | Direct sends are allowed with queued prompts; queue remains opt-in temporal deferral. | `ddae3e62` |
| ISSUE-4 — stale docs | Living docs were resynced to the paper language and current UX. | `1c047e9b` |
| ISSUE-5 — mobile navigation | Compact mobile navigation shipped. | `51a4c098` |
| ISSUE-6 — mobile stop control | Unified topbar interrupt shipped. | `83fe9dab` |
| ISSUE-7 — mobile toast | Toast delivery/placement reconciliation shipped. | `9ac87675` |
| ISSUE-8 — voice-button error color | Error state is gated on `voiceAnnouncementsEnabled`. | `a7569c3a` |
| PR #21 — Claude interactive-prompt UI | Close without merge; focused shared-broker Claude Code backend superseded its useful scope. | `f4a06a2e` |
| PR #22 — macOS web session launch | Rewrite shipped; close the stale PR without merging its branch. | `d86bdda6` |

### In progress

None. No current owner is assigned because no ledger item is in progress.

### Open

The existing open residuals have concrete next actions in `PRODUCT_GAP_STATUS.md`: Pi retry-status compatibility, terminal quiet-window behavior, large-log run-settings replay cost, pre-first-turn slash-command evidence, and real-hardware paper-language judgment.

## Historical closure summary (superseded 2026-08-02)

Every original issue below was closed during the 2026-08-01 → 2026-08-02 product overhaul. The earlier final-disposition summary remains as history:

- ISSUE-1 (Inter font): resolved by deliberate decision — bare `sans-serif`, no web font.
- ISSUE-2/3 (send-now semantics): closed by unconditional confirmed-send unification.
- ISSUE-4 (stale docs): closed; docs resynced again 2026-08-02 to paper language.
- ISSUE-5/6/7 (mobile nav/stop/toast): closed by paper redesign — unified topbar interrupt, 32px chrome + hit-slop, toast at top.
- ISSUE-8 (voice button red): closed (gated on voiceAnnouncementsEnabled).

## PR #21 and #22 reconciliation (2026-08-04)

- **PR #21 — “Add Claude backend support with interactive prompt UI”: close without merge; superseded.** Its useful Claude Code parser/backend scope was reimplemented as the focused shared-broker `cc` backend in [`f4a06a2e`](https://github.com/yiwenlu66/codoxear/commit/f4a06a2e876c8c4535713b7c6d25d76455f6838c), then hardened by later CC transcript, terminal-outcome, context, live-control, and subagent-liveness commits. The PR's coupled interactive-prompt UI was deliberately not adopted.
- **PR #22 — “Fix macOS web session launches”: rewrite and ship.** The old branch bypassed the login shell but did not preserve the broker PTY for direct Codex execution. [`d86bdda6`](https://github.com/yiwenlu66/codoxear/commit/d86bdda6) directly attaches web-owned Codex to that PTY before exec, preserves login-shell startup only for Pi/Claude, resolves standard user CLI locations without a shell profile, and degrades legacy `flex`/`default` service tiers to an omitted Codex flag. Close the stale PR without merging its branch.

Later same-session concerns (design language, state honesty, keyboard map, /model + /thinking, search UX, sidebar display, diag view, unattended persistence, idle-on-error, terminal-error idle, count monotonicity across steer/queue) are all done; their individual commit evidence is retained in PRODUCT_GAP_STATUS.md “Reconciled disposition ledger”.
