# Clean-room adversarial review: destructive-confirm focus safety slice

**Verdict: ACCEPT**

Repository: `/home/yiwen/codex-web-product-recovery` branch `recovery/product-gaps`
Commits: `97a8463` → `15b80cf` → `1d46919`
Review date: 2026-07-08

---

## 1. What the slice does

Adds a `destructive` boolean to `confirmApp()` options (default `false`). When `true`, `focusAppConfirmInitial()` targets `appConfirmCancelBtn` instead of `appConfirmConfirmBtn`. A document-level Tab keydown handler traps Tab/Shift-Tab inside `#appConfirm` when the dialog is visible, cycling focus between the two dialog buttons with wraparound.

Six call sites are marked `destructive: true`: clear unknown-send marker, delete/dismiss session, dismiss launch record, reload file from disk, queue recovery-item delete, and clear pending attachment state. The constructive "Send pending attachment?" call site remains unmarked (confirm-focused).

## 2. Invariant verification

| Invariant | Mechanism | Evidence |
|---|---|---|
| **Destructive → Cancel focused** | `preferred = destructive ? appConfirmCancelBtn : appConfirmConfirmBtn` in `focusAppConfirmInitial` | Browser proof: `desktop-delete-dialog-before-enter.json`, `mobile-delete-dialog-before-enter.json`, `desktop-reload-dialog-before-enter.json`, `mobile-reload-dialog-before-enter.json` all show `activeId: "appConfirmCancelBtn"` |
| **Constructive → Confirm focused** | Same ternary with `destructive: false` default | Browser proof: `desktop-constructive-dialog.json` shows `activeId: "appConfirmConfirmBtn"` for "Send pending attachment?" |
| **Tab/Shift-Tab trapped** | `e.key === "Tab" && appConfirm.style.display === "flex"` → `preventDefault()` + `stopPropagation()` + manual focus cycling on `[cancelBtn, confirmBtn]` | Browser proof: desktop cycle Cancel→Confirm→Cancel→Confirm; mobile Cancel→Confirm→Cancel |
| **Escape cancels** | Existing handler unchanged: `resolveAppConfirm(false)` | `desktop-tab-after-escape.json` shows `display: "none"`, `backdropDisplay: "none"` |
| **Backdrop cancels** | `appConfirmBackdrop.onclick = () => resolveAppConfirm(false)` — untouched | Code inspection: line unchanged |
| **Cancel-before-mutation** | Enter on Cancel-focused dialog resolves `false`; no destructive action executes | Delete: `desktop-delete-after-enter.json` → `apiHasProof: true`, `rowPresent: true`. Reload: `desktop-reload-after-enter.json` → draft preserved, conflict action visible, disk file `"disk newer text\n"` unchanged. Broker: `send_calls=0, keys_calls=0, shutdown_calls=0` |
| **Return focus** | `appConfirmReturnFocusEl` saved on open, restored via `restoreModalFocus()` — untouched | Code inspection: no changes to `resolveAppConfirm` or `restoreModalFocus` |
| **No native confirm()** | Test `test_product_static_js_has_no_native_confirm_calls` passes; browser override tracks count | `nativeConfirmCount: 0` across all browser captures |

All eight invariants hold.

## 3. Code quality assessment

**Focus logic**: The ternary `preferred/fallback` pattern with disabled-button fallback is robust. If `preferred` is disabled, `fallback` is tried. If both disabled, `target` is null and focus is skipped (no crash).

**Tab trap**: Handles the edge case where `document.activeElement` is not one of the two buttons (`currentIndex < 0`) by defaulting to the first (Tab) or last (Shift-Tab) focusable. Wraparound arithmetic `(currentIndex + offset + focusable.length) % focusable.length` is correct.

**`destructive` default**: `Boolean(raw.destructive)` converts `undefined` to `false`. String-option path explicitly returns `destructive: false`. Both preserve backward compatibility.

**Scope**: Production changes span 35 lines across `app.js` (+34) and `app_queue.js` (+1). No CSS, HTML, or server changes. No changes to other dialogs (file-unsaved, file-paste, send-choice).

## 4. Test coverage

- `test_destructive_confirm_dialog_focuses_cancel_and_traps_tab`: Verifies source-level presence of the `destructive` parameter, focus logic, Tab trap mechanism.
- `test_destructive_confirm_call_sites_are_marked`: Regex-matches all destructive titles to `destructive: true` within 360 chars; verifies "Send pending attachment?" block does NOT contain `destructive`.
- `test_async_confirm_seams_are_wired_from_app`: Updated to include `destructive: true` in `confirmReload` seam assertion.
- `test_file_viewer_source.py`: Updated `confirmReload` assertion to include `destructive: true`.
- `test_frontend_queue_module_source.py`: Added assertion for `destructive: True` on queue recovery-item confirm.
- Full test suite: **1834 passed, 134 subtests passed** (0 failures).
- Docker focused tests: **137 passed, 25 subtests passed**.

## 5. Browser proof quality

Real product paths exercised (not synthetic):
- **Delete session?** on desktop and mobile (390×844)
- **Reload file from disk?** with real file-save conflict on desktop and mobile
- **Send pending attachment?** constructive control on desktop
- Tab cycling verified on both viewports
- No horizontal overflow on either viewport
- Container cleaned up: `docker-ps-after-stop.txt` is empty

## 6. Sanitization

- `token` fields in API snapshots are `null` — no credential leak.
- No passwords, cookies, HMAC secrets, or API keys in any artifact.
- Protected checkout `/home/yiwen/codex-web` has clean `git status` — untouched.
- References to protected path exist only in `PROMPT.md` as documentation constraints, not as artifacts.

## 7. Non-blocking observations

1. **Tab trap is bubble-phase, not capture**: The keydown handler with the Tab trap is registered without `{capture: true}`. Since capture-phase file editor handlers don't process Tab, this is functionally fine. A capture-phase registration would be marginally more defensive against future bubble-phase Tab handlers, but the current implementation works correctly because `stopPropagation()` prevents further bubble-phase dispatch.

2. **Tab order is fixed Cancel→Confirm**: `appConfirmFocusableControls()` hardcodes `[appConfirmCancelBtn, appConfirmConfirmBtn]`. This matches the visual left-to-right button order in the dialog. If button order were ever reversed in CSS/HTML, the Tab order would be wrong — but this is a standard DOM-order-matches-visual-order assumption.

3. **No aria-live announcement on focus change**: Destructive dialogs don't announce "focus is on Cancel" to screen readers beyond standard button labels. The existing `role="dialog"` and `aria-modal="true"` are sufficient per ARIA dialog patterns, and the focused button's label is announced normally.

---

## Decision

**ACCEPT** — All claimed invariants verified through code inspection, local test execution (1834/1834 pass), and browser proof artifacts covering real product paths on desktop and mobile. No scope creep, no credential leaks, no protected-path modifications, no blocking concerns.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Production changes are 35 lines across app.js (+34) and app_queue.js (+1). Adds destructive boolean to confirmApp options, focus-on-cancel for destructive dialogs, Tab trap scoped to #appConfirm. No CSS/HTML/server/other-dialog changes. Six destructive call sites marked; one constructive site left unmarked. No scope widening."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Eight invariants verified (destructive→Cancel, constructive→Confirm, Tab trap, Escape, backdrop, cancel-before-mutation, return-focus, no-native-confirm) through code inspection, 1834-test full suite pass, and 48 browser proof artifacts across desktop and mobile viewports."
    }
  ],
  "changedFiles": [
    "codoxear/static/app.js",
    "codoxear/static/app_queue.js",
    "tests/test_in_app_confirm_source.py",
    "tests/test_file_viewer_source.py",
    "tests/test_frontend_queue_module_source.py"
  ],
  "testsAddedOrUpdated": [
    "tests/test_in_app_confirm_source.py::test_destructive_confirm_dialog_focuses_cancel_and_traps_tab",
    "tests/test_in_app_confirm_source.py::test_destructive_confirm_call_sites_are_marked",
    "tests/test_in_app_confirm_source.py::test_async_confirm_seams_are_wired_from_app (updated)",
    "tests/test_file_viewer_source.py::TestFileViewerSource (updated confirmReload assertion)",
    "tests/test_frontend_queue_module_source.py (added destructive=True assertion)"
  ],
  "commandsRun": [
    {
      "command": "python3 -m pytest tests/test_in_app_confirm_source.py -v",
      "result": "passed",
      "summary": "6 passed in 0.48s"
    },
    {
      "command": "python3 -m pytest tests/test_frontend_queue_module_source.py tests/test_file_viewer_source.py -v --tb=short",
      "result": "passed",
      "summary": "64 passed, 25 subtests passed in 2.03s"
    },
    {
      "command": "python3 -m pytest tests/ -v --tb=short",
      "result": "passed",
      "summary": "1834 passed, 134 subtests passed in 25.89s"
    },
    {
      "command": "git status --porcelain",
      "result": "passed",
      "summary": "clean working tree, no staged files"
    },
    {
      "command": "git diff 97a8463^..1d46919 -- codoxear/",
      "result": "passed",
      "summary": "35 lines changed in app.js and app_queue.js"
    }
  ],
  "validationOutput": [
    "All 8 invariants verified: destructive→Cancel focus, constructive→Confirm focus, Tab/Shift-Tab trap, Escape cancel, backdrop cancel, cancel-before-mutation (session preserved, draft preserved, disk unchanged, 0 broker send/keys calls), return-focus preserved, no native confirm()",
    "Browser proof covers Delete session and Reload file on desktop and 390x844 mobile; Send pending attachment constructive control on desktop",
    "Docker focused tests: 137 passed, 25 subtests passed",
    "No credential/secret values in artifacts (token fields are null)",
    "Protected checkout /home/yiwen/codex-web untouched",
    "Docker container cleaned up (docker-ps-after-stop.txt is empty)"
  ],
  "residualRisks": [
    "Tab trap uses bubble phase rather than capture phase — functionally correct but marginally less defensive against hypothetical future bubble-phase Tab handlers",
    "Tab order hardcoded to [Cancel, Confirm] matching current DOM order — would need update if button visual order is ever reversed"
  ],
  "noStagedFiles": true,
  "diffSummary": "Adds destructive boolean to confirmApp options; destructive dialogs focus Cancel, constructive focus Confirm; Tab/Shift-Tab trapped inside #appConfirm cycling Cancel↔Confirm; 6 destructive call sites marked, 1 constructive unmarked; 3 test functions added/updated",
  "reviewFindings": [
    "no blockers"
  ],
  "manualNotes": "All review was read-only. No edits, staging, or commits made to any repository. Protected /home/yiwen/codex-web verified clean."
}
```
