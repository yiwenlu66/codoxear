## Objective
Make app-owned confirmation dialogs safer for destructive keyboard paths: destructive confirmations must default focus to Cancel and keep Tab focus within the dialog, while constructive confirmations preserve confirm-focused behavior.

Done when the implementation is committed, locally validated, Docker/browser proof covers desktop and mobile keyboard behavior on real product paths, clean-room review accepts the slice, and task/project memory records the accepted invariant.

## Workbench
1. Prove the current defect with a failing discriminator: destructive confirmations currently focus Confirm and there is no Tab/Shift-Tab cycle.
2. Implement destructive-safe initial focus and a scoped Tab trap for `#appConfirm`.
3. Mark destructive confirmation call sites, while keeping constructive "Send pending attachment?" confirm-focused.
4. Preserve existing confirmation invariants: Escape/backdrop cancel, cancel-before-mutation ordering, route payloads, return-focus, no native `confirm()`.
5. Prove behavior in Docker/browser on desktop and ~390x844 mobile, then run clean-room review and record durable memory.

## Context
Active checkout: `/home/yiwen/codex-web-product-recovery` on branch `recovery/product-gaps`.
Protected checkout: `/home/yiwen/codex-web` on `main`; do not touch.
Next-slice scout: `/tmp/pi-subagents-uid-1000/artifacts/ec79d9f9-168f-4b39-a004-cf34b2f91b85_theorist_output.md` and `/tmp/codoxear-next-slice-after-code-copy.md`.
Relevant prior slice: `.memory/tasks/2026-07-07-in-app-confirm-dialogs/`.
Project memory: `.memory/project/ARCHITECTURE.md`, `.memory/project/VALIDATION.md`.
Relevant source/tests: `codoxear/static/app.js`, `codoxear/static/app.css`, `tests/test_in_app_confirm_source.py`, `codoxear/static/app_queue.js`, `codoxear/static/app_file_viewer.js`.
Docker skill: `.codex/skills/codoxear-docker-test/SKILL.md`.

## Task specifications
Current mechanism: `confirmApp()` normalizes title/message/button labels and calls `focusAppConfirmInitial()`. `focusAppConfirmInitial()` unconditionally targets `appConfirmConfirmBtn` when enabled. The document keydown handler cancels on Escape, but has no Tab/Shift-Tab cycle for the dialog. `syncModalIsolation()` inert-hides the app behind the dialog, but focus can still leave the dialog to browser chrome rather than cycling inside the dialog.

Target mechanism: confirmation options include a boolean `destructive` defaulting false. Destructive confirmations focus Cancel initially. Non-destructive confirmations keep Confirm initial focus when enabled. While `#appConfirm` is visible, Tab and Shift-Tab cycle only across enabled dialog buttons (Cancel and Confirm), with wraparound. If only one focusable control exists, Tab keeps focus there. The handler must prevent default/propagation only for active app-confirm Tab handling. Escape/backdrop/cancel semantics remain unchanged.

Destructive call sites to mark include: clear unknown-send marker, delete session, dismiss launch record, reload file from disk/discard draft, queue recovery-item delete via `confirmAction`, and clear pending attachment state. Constructive `Send pending attachment?` must remain non-destructive/confirm-focused.

The user-visible failure being corrected is accidental data-affecting keyboard confirmation: a bare Enter/Space immediately after a destructive dialog opens should cancel rather than delete/discard/clear. Touch behavior and explicit Confirm activation should remain unchanged.

Browser proof should exercise real product paths rather than a synthetic detached dialog where possible: at minimum `Delete session?` and `Reload file from disk?` destructive dialogs. Prove initial focus, Enter-on-open result, Tab/Shift-Tab cycle, and absence of unintended backend `send`/`keys` or destructive mutation on cancel. Also prove a constructive `Send pending attachment?` dialog remains confirm-focused.

## Constraints
Do not edit/promote/merge protected `/home/yiwen/codex-web` or `main`.
Do not touch live runtime dirs: `~/.local/share/codoxear`, `~/.claude`, `~/.codex`, host Pi logs/sockets, systemd/tailscale.
Docker-only for broker/server/session/tmux/browser verification; avoid port `8743`.
Cleanup must be exact-PID/container scoped; no `pkill -f`, `killall`, broad kills.
Keep functional, proof/evidence, review, and memory commits separate.
Browser + Docker evidence required for browser/product usability claims.
Delegate concrete implementation/validation work to executor subagents where possible.
Run clean-room adversarial review before yielding.
Do not copy secrets into committed artifacts; exclude cookies, auth headers, credential values, private file contents, bulky logs.
Monaco remains required; no plain textarea/diff fallback certification.
