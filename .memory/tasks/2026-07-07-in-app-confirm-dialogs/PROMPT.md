## Objective
Replace Codoxear's remaining product-surface native `window.confirm()` dialogs with an in-app accessible confirmation dialog so destructive/recovery/attachment confirmations render inside the Codoxear UI on desktop and mobile.

## Workbench
1. Implement one reusable async confirmation dialog in the app shell.
2. Route existing app confirmations through it without changing the underlying action semantics.
3. Update queue/file-viewer controller seams so confirmations can be async and testable.
4. Prove via source/unit tests plus Docker/browser that a confirmation appears as Codoxear DOM, not browser chrome, and that cancel/confirm branches still behave.

## Context
Active checkout: `/home/yiwen/codex-web-product-recovery` on branch `recovery/product-gaps`.
Protected checkout: `/home/yiwen/codex-web` on `main`; do not edit.
Current open candidate came from `/tmp/codoxear-next-slice-after-effort-markers.md`: `window.confirm()` native dialogs remain an open mobile polish gap.

## Task specifications
- Scope: product-surface confirmations in `codoxear/static/app.js` and owned frontend modules wired by it. Current known call sites are commit-unknown clear, session delete / failed-launch dismiss, file reload conflict, pending attachment send, pending attachment clear, and queue recovery-item delete.
- The new dialog must be DOM-based, accessible (`role="dialog"`, `aria-modal="true"`, labelled title/message), keyboard usable, focus-restoring, and visually consistent with existing modal/backdrop patterns.
- Confirmation API should return a Promise<boolean>; callers must await it before mutating state or calling backend routes.
- Preserve existing backend request bodies and guard semantics. Cancel must produce no backend mutation.
- Avoid using `window.confirm`/bare `confirm` in Codoxear product code after the slice, except third-party vendored Monaco assets.
- Tests may use injected fake confirm functions for modules; source tests should assert the product code no longer calls native confirm.
- Browser proof must exercise at least one cancel and one confirm path through the real UI, not only source inspection.

## Constraints
Do not touch `/home/yiwen/codex-web` or `main`.
Do not touch live runtime dirs (`~/.local/share/codoxear`, `~/.claude`, `~/.codex`, host Pi logs/sockets, systemd/tailscale).
Use Docker-only for broker/server/session/browser verification; avoid port 8743.
Cleanup must be exact container/PID scoped via sandbox commands; no `pkill -f`, `killall`, or broad kills.
Keep functional, proof/evidence, review, and memory commits separate.
Browser + Docker evidence is required for product usability claims.
Do not commit secrets, cookies, auth headers, credential values, private file contents, bulky logs, or ignored artifacts.
