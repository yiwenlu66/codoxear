# In-app confirmation dialogs epistemic model

## Phenomenon
Codoxear product-surface confirmations used to leave the app DOM and appear as native browser `confirm()` chrome. That was especially poor on mobile and made destructive/recovery decisions hard to verify as Codoxear UI.

## Accepted mechanism
A single app-owned `confirmApp()` dialog is now the confirmation mechanism for product code. It renders `#appConfirm` with `role="dialog"`, `aria-modal="true"`, labelled/described title/message nodes, an in-app backdrop, confirm/cancel buttons, Escape/backdrop cancel, modal isolation, and focus restoration. It returns `Promise<boolean>`: only the confirm button resolves `true`; cancel, Escape, backdrop, or superseding confirmation resolve `false`.

Converted call sites await the dialog before mutating state or calling backend routes:

- commit-unknown marker clear
- sidebar session delete / failed-launch dismiss
- recovery-panel failed-launch dismiss
- pending attachment send confirmation
- stale pending attachment state clear
- queue recovery-item delete through injected `confirmAction`
- file reload conflict through async `confirmReload`

The backend request bodies and action semantics did not change; the change is the user decision surface and async plumbing.

## Evidence
Source/static evidence shows no native `window.confirm(` or bare `confirm(` calls remain in `codoxear/static/*.js` outside vendored Monaco, and source/controller tests cover the async seams and cancel-before-mutation ordering. Functional commit `05a2380` passed focused local validation (`123 passed, 25 subtests`), full local pytest (`1823 passed, 134 subtests`), and `git diff --check` (OPS).

Docker/browser proof in `.memory/tasks/2026-07-07-in-app-confirm-dialogs/browser-artifacts/in-app-confirm-19443/` exercised the real sidebar failed-launch dismissal path. With `window.confirm` overridden to count/throw, desktop and mobile browser actions opened the DOM dialog with the expected accessible attributes and zero native-confirm calls. Cancel preserved the launch row and confirm removed only the confirmed row from DOM and `/api/sessions` (OPS). Docker focused tests and smoke passed (OPS).

Clean-room review `3d530118` accepted the slice with no blockers. It confirmed native-confirm removal, cancel-before-mutation, payload preservation, async conversion safety, modal proof adequacy, and sanitized evidence. Non-blocking concerns are that the confirm button receives initial focus, the dialog does not trap Tab/Shift-Tab, and direct browser proof covers failed-launch dismissal while other seams rely on source/controller tests (OPS).

## Current claim
The in-app confirmation slice is accepted for the scoped product gap: Codoxear product confirmations now render and resolve inside the app DOM while preserving destructive/recovery mutation boundaries.

## Boundary
This slice did not create a general focus trap for all modals and did not redesign destructive-action default focus. It did not browser-drive every converted confirmation seam; source/controller tests cover those equivalent await-before-mutation paths. Real provider/auth behavior is irrelevant to this UI confirmation mechanism.
