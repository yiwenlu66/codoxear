# Destructive confirm focus safety epistemic model

## Phenomenon
Codoxear uses app-owned confirmation dialogs for destructive/data-affecting actions. Keyboard users can activate the initially focused button with Enter or Space. If destructive dialogs initially focus Confirm, the dialog makes the destructive action the default keyboard outcome.

## Current mechanism
`confirmApp()` opens `#appConfirm` and calls `focusAppConfirmInitial()`. The focus helper targets `appConfirmConfirmBtn` whenever it is enabled, regardless of action type. The global keydown handler cancels on Escape, but there is no active Tab/Shift-Tab cycle for the confirmation dialog.

## Target mechanism
Confirmation options classify destructive actions. Destructive confirmations focus Cancel initially; constructive confirmations preserve Confirm initial focus. While the app confirmation dialog is visible, Tab/Shift-Tab cycles through enabled dialog controls and cannot escape the dialog. Existing cancel paths, confirm route payloads, focus restoration, and native-confirm exclusion remain unchanged.

## Live risks
- Misclassifying constructive confirmation as destructive could add friction to the "Send pending attachment?" path.
- Missing a destructive call site would leave an inconsistent keyboard hazard.
- A broad modal focus framework could accidentally interfere with file viewer, send-choice, or unsaved-file dialogs; the intervention should stay scoped to `#appConfirm`.
- Browser proof must distinguish cancel-before-mutation from merely hiding the dialog; backend call logs or state probes are needed.

## Current claim
This is a bounded accessibility/product-safety slice: a single dialog focus mechanism currently chooses the destructive action by default, and product-owned confirmation semantics make the safe default enforceable without changing backend workflows.
