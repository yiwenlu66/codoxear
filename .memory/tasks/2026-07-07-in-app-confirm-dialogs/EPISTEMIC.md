# In-app confirmation dialogs epistemic model

## Phenomenon
Codoxear still uses native browser confirmation dialogs for several product-surface decisions. On mobile, these appear as browser chrome rather than the app's modal language, break visual continuity, and are hard to instrument as user-visible Codoxear UI.

## Current mechanism
Existing call sites synchronously call `window.confirm()` / `confirm()` immediately before guarded actions. That mechanism blocks JavaScript until the browser chrome returns a boolean. It preserves mutation boundaries but makes the confirmation surface external to Codoxear's DOM. Queue and file-viewer modules encode the same synchronous assumption through direct `window.confirm()` or a synchronous `confirmReload` injection.

## Target mechanism
A single app-owned async confirmation dialog renders a Codoxear modal/backdrop, traps the user decision in app DOM, restores focus on close, and resolves `true` only from the confirm button. Existing guarded actions await this promise before backend mutation. Cancel resolves `false` and exits before mutation.

## Live uncertainties
- Which existing focus helpers can be reused cleanly for a generic confirmation modal without interfering with open queue/file/recovery modals.
- Whether the file-viewer save-conflict path can safely await an async `confirmReload` without changing its active-conflict check semantics.
- Whether queue controller tests depend on synchronous `window.confirm()` and need a new injected async confirmation seam.

## Justified claim
The slice is bounded if it only changes frontend confirmation plumbing and preserves backend payloads/mutation points. Acceptance requires source evidence that product code no longer invokes native confirm, behavior tests for cancel/confirm branches, and Docker/browser proof that the visible decision surface is Codoxear DOM.
