# In-app confirmation dialog verification

## Claim
Codoxear product confirmations now render as an in-app DOM dialog instead of native browser `confirm()` chrome, while preserving cancel/confirm mutation boundaries.

## Evidence

- Functional commit: `05a2380 Replace native confirms with app dialog`.
- Local validation: focused source/controller tests passed (`123 passed, 25 subtests`); full local suite passed (`1823 passed, 134 subtests`).
- Static product-code search: `rg '\b(window\.)?confirm\s*\(' codoxear/static --glob '*.js' --glob '!codoxear/static/monaco/**'` returned no matches.
- Docker focused validation on port 19441: `123 passed, 25 subtests` (`docker-focused-19441.txt`).
- Docker smoke on port 19442: pre-login `/api/me=401`, post-login `/api/sessions=200`, container app dir `/home/tester/.local/share/codoxear` (`docker-smoke-19442.txt`).
- Browser proof on Docker server port 19443 used two synthetic failed-launch rows, `launch-confirm-cancel` and `launch-confirm-delete`, created inside the container launch ledger (`container/create-failed-launch-records.json`).

## Browser observations

`browser/confirm-proof-result.json` proves the desktop user-visible branch:

- Clicking the sidebar delete affordance opened `#appConfirm` with `display:"flex"`, `role:"dialog"`, `aria-modal:"true"`, `aria-labelledby:"appConfirmTitle"`, `aria-describedby:"appConfirmMessage"`.
- Dialog text was the product confirmation: title `Dismiss launch record?`, message `Dismiss this launch record?`, confirm button `Dismiss`, cancel button `Cancel`.
- The focused element inside the open dialog was `appConfirmConfirmBtn`.
- `window.confirm` was overridden to increment/throw; `nativeConfirmCount` stayed `0` across cancel and confirm branches.
- Cancel branch: after pressing Cancel, both rows remained in DOM, session count stayed `2`, and the dialog closed.
- Confirm branch: after pressing Dismiss, `launch-confirm-delete` disappeared from DOM and from `/api/sessions`; `launch-confirm-cancel` remained.

`browser/mobile-confirm-dialog.json` proves the mobile viewport branch at `390x844`:

- Clicking the remaining launch row's delete affordance opened the same in-app dialog with `display:"flex"`, `role:"dialog"`, and `aria-modal:"true"`.
- Dialog bounds were inside the viewport (`left≈15.6`, `right≈374.4`, width≈358.8), and `hasHorizontalOverflow:false`.
- Cancel closed the dialog, preserved the row, and `nativeConfirmCount` stayed `0`.

## Boundary

The browser proof exercises failed-launch dismissal because it is a deterministic container-only destructive confirmation path. Source/controller tests cover the other converted seams: pending attachment send, commit-unknown clear, queue recovery-item delete, and file reload conflict. No real provider/auth behavior is claimed.
