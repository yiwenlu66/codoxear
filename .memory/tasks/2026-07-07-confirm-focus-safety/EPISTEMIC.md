# Destructive confirm focus safety epistemic model

## Phenomenon
Codoxear uses app-owned confirmation dialogs for destructive/data-affecting actions. Keyboard activation follows focus: Enter/Space activates the initially focused button. Before this slice, destructive dialogs made the destructive action the default keyboard outcome because Confirm received initial focus.

## Accepted mechanism
`confirmApp()` options now include `destructive` with default `false`. Destructive confirmations focus `appConfirmCancelBtn`; constructive confirmations keep `appConfirmConfirmBtn` as the initial focus target. While `#appConfirm` is visible, Tab/Shift-Tab is trapped inside enabled dialog controls and cycles Cancel ↔ Confirm. The mechanism is scoped to the app-owned confirmation dialog; other modals/dialogs are unchanged.

Destructive call sites are marked for clear unknown-send marker, delete/dismiss session, dismiss launch record, reload file from disk/discard draft, queue recovery-item delete, and clear pending attachment state. The constructive `Send pending attachment?` path remains confirm-focused.

## Evidence
- Discriminator and functional implementation: OPS entries for executor output and main validation; functional commit `15b80cf`.
- Docker/browser proof: OPS entry for `1d46919`; desktop/mobile proof shows destructive Delete/Reload dialogs focus Cancel, Enter cancels without mutation, Tab cycles inside the dialog, constructive pending-attachment send remains confirm-focused, and broker logs contain zero send/keys/shutdown calls.
- Clean-room review: OPS entry for `5cec1c0`; critic accepted the slice and found no blockers.

## Ruled out mechanisms
- Native browser confirmation: already prohibited by the product confirmation invariant and source tests; not reintroduced.
- Broad modal focus framework: unnecessary and higher-risk because the defect is confined to `#appConfirm`. A local Tab trap is sufficient.
- Destructive-by-title inference: rejected because call sites know action semantics; explicit `destructive: true` keeps the contract visible at mutation boundaries.

## Current claim
The destructive-confirm focus safety slice is accepted. Future app-confirm call sites must classify data-affecting/destructive actions with `destructive: true`; destructive dialogs must default to Cancel, constructive confirmations may default to Confirm, and Tab/Shift-Tab must remain trapped inside `#appConfirm` while preserving Escape/backdrop cancel and return-focus behavior.
