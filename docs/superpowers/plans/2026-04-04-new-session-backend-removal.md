# New Session Backend Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the duplicate backend selector from the bottom of the `New session` dialog so the header `Codex` / `Pi` tabs are the only backend control.

**Architecture:** Keep `newSessionBackend` as the single state source in `codoxear/static/app.js`, but drive it only from the header tabs. Delete the lower backend picker DOM, menu state, event wiring, and click-dismiss logic that only existed for the removed control, while preserving all backend-dependent form behavior.

**Tech Stack:** Python unittest + pytest runner, static browser UI in `codoxear/static/app.js`, CSS in `codoxear/static/app.css`

---

## File Map

- Modify: `tests/test_pi_continue_ui.py` — add and update source-level regression tests for the New session dialog so removal of the bottom backend picker is enforced.
- Modify: `codoxear/static/app.js` — remove duplicate backend picker DOM, menu wiring, button label syncing, and outside-click handling; keep header tab backend switching.
- Verify only: `codoxear/static/app.css` — confirm no CSS changes are needed after row removal; only touch this file if layout visibly regresses.
- Reference: `docs/superpowers/specs/2026-04-04-new-session-backend-design.md` — implementation contract.

### Task 1: Lock the UI contract with failing tests

**Files:**
- Modify: `tests/test_pi_continue_ui.py`
- Reference: `codoxear/static/app.js`

- [ ] **Step 1: Add a regression test that asserts the New session dialog keeps header tabs and drops the lower backend row**

```python
def test_new_session_dialog_uses_header_tabs_as_only_backend_control(self) -> None:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index('const newSessionViewer = el("div", { class: "formViewer newSessionViewer"')
    end = source.index("root.appendChild(newSessionBackdrop);", start)
    block = source[start:end]

    self.assertIn("newSessionBackendTabs", block)
    self.assertNotIn("newSessionBackendRow", block)
    self.assertNotIn("newSessionBackendField", block)
    self.assertNotIn("newSessionBackendBtn", block)
```

- [ ] **Step 2: Add a regression test that `applyDialogMenus()` no longer references the deleted backend picker menu/button**

```python
def test_apply_dialog_menus_no_longer_touches_backend_picker(self) -> None:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function applyDialogMenus() {")
    end = source.index("function positionDialogMenu(menu, anchorBtn) {", start)
    block = source[start:end]

    self.assertNotIn("newSessionBackendMenu.classList.toggle", block)
    self.assertNotIn("newSessionBackendBtn.setAttribute", block)
    self.assertNotIn("positionDialogMenu(newSessionBackendMenu, newSessionBackendBtn)", block)
```

- [ ] **Step 3: Update the outside-click regression to verify backend-specific dismissal logic is gone**

```python
def test_document_click_does_not_reference_removed_backend_picker(self) -> None:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index('document.addEventListener("click", (e) => {')
    end = source.index('document.addEventListener("keydown", (e) => {', start)
    block = source[start:end]

    self.assertNotIn("#newSessionBackendField", block)
    self.assertNotIn("#newSessionBackendMenu", block)
    self.assertNotIn("newSessionBackendMenuOpen", block)
```

- [ ] **Step 4: Remove the obsolete click-handler VM test that only exists for the deleted backend menu, or rewrite it into the source assertion above**

```python
# Delete these now-obsolete helpers/tests after the source assertion is added:
# - eval_new_session_document_click(...)
# - test_document_click_closes_backend_menu_on_outside_click(...)
```

- [ ] **Step 5: Run the targeted test file and verify it fails for the expected reason before touching app code**

Run: `python3 -m pytest tests/test_pi_continue_ui.py -q`
Expected: FAIL because `codoxear/static/app.js` still contains `newSessionBackendRow`, `newSessionBackendBtn`, backend-menu handling in `applyDialogMenus()`, and backend outside-click logic.

- [ ] **Step 6: Commit the red test state**

```bash
git add tests/test_pi_continue_ui.py
git commit -m "test: cover new session backend tab-only UI"
```

### Task 2: Remove the duplicate backend picker and keep header-driven behavior

**Files:**
- Modify: `codoxear/static/app.js`
- Verify only: `codoxear/static/app.css`
- Test: `tests/test_pi_continue_ui.py`

- [ ] **Step 1: Delete the lower backend picker DOM nodes from the dialog construction code**

```javascript
const newSessionBackendTabs = el("div", { class: "agentBackendTabs", id: "newSessionBackendTabs" });

const newSessionViewer = el("div", { class: "formViewer newSessionViewer", id: "newSessionViewer", role: "dialog", "aria-label": "New session" }, [
  el("div", { class: "queueHeader" }, [
    el("div", { class: "newSessionHeaderLead" }, [
      el("div", { class: "title", text: "New session" }),
      newSessionBackendTabs,
    ]),
    el("div", { class: "actions" }, [newSessionCloseBtn]),
  ]),
  newSessionStatus,
  el("div", { class: "formBody" }, [
    el("label", { class: "field" }, [
      el("span", { class: "fieldLabel", text: "Working directory" }),
      newSessionCwdField,
      newSessionCwdHint,
    ]),
    el("label", { class: "field" }, [
      el("span", { class: "fieldLabel", text: "Session name" }),
      newSessionNameInput,
    ]),
    el("div", { class: "formGrid newSessionRunConfigRow" }, [
      el("label", { class: "field" }, [
        el("span", { class: "fieldLabel", text: "Model" }),
        newSessionModelField,
      ]),
      el("label", { class: "field" }, [
        el("span", { class: "fieldLabel", text: "Reasoning effort" }),
        newSessionReasoningField,
      ]),
      newSessionFastField,
    ]),
    newSessionPiModeField,
    newSessionResumeField,
    newSessionProviderTmuxRow,
    newSessionWorktreeField,
  ]),
  el("div", { class: "formActions" }, [
    el("button", { id: "newSessionCancelBtn", type: "button", text: "Cancel" }),
    newSessionStartBtn,
  ]),
]);
```

- [ ] **Step 2: Remove backend-menu-only plumbing from the dialog root and state bookkeeping**

```javascript
// Delete these declarations entirely:
// let newSessionBackendMenuOpen = false;
// const newSessionBackendBtn = ...
// const newSessionBackendMenu = ...
// const newSessionBackendField = ...
// const newSessionBackendRow = ...

newSessionViewer.appendChild(newSessionModelMenu);
newSessionViewer.appendChild(newSessionProviderMenu);
newSessionViewer.appendChild(newSessionReasoningMenu);
newSessionViewer.appendChild(newSessionResumeMenu);
```

- [ ] **Step 3: Remove button-label syncing and menu rendering that only served the deleted picker**

```javascript
function setNewSessionBackend(value, { resetSelections = false } = {}) {
  const next = normalizeAgentBackendName(value);
  const previous = newSessionBackend;
  newSessionBackend = next;
  rememberBackendChoice(next);
  const defaults = defaultsForAgentBackend(next);
  const providerChoices = providerChoicesForBackend(next);
  const defaultProvider = typeof defaults.provider_choice === "string" ? defaults.provider_choice.trim() : "";
  const rememberedProvider = loadRememberedProviderChoice(next);
  if (resetSelections || previous !== next || !providerChoices.includes(newSessionProvider)) {
    setNewSessionProvider((rememberedProvider && providerChoices.includes(rememberedProvider) ? rememberedProvider : "") || defaultProvider || providerChoices[0] || "");
  } else {
    setNewSessionProvider(newSessionProvider);
  }
  const modelDefault = typeof defaults.model === "string" ? defaults.model.trim() : "";
  if (resetSelections || previous !== next) {
    newSessionModelInput.value = modelDefault;
  }
  const reasoningChoices = reasoningChoicesForBackend(next);
  const defaultEffort = typeof defaults.reasoning_effort === "string" ? defaults.reasoning_effort.trim().toLowerCase() : "";
  newSessionReasoningEffort = reasoningChoices.includes(defaultEffort) ? defaultEffort : (reasoningChoices[0] || "high");
  renderNewSessionBackendTabs();
  renderNewSessionProviderMenu();
  renderNewSessionReasoningMenu();
  syncNewSessionRunConfigUi();
  resetNewSessionResumeState();
  syncNewSessionBackendUi();
  scheduleNewSessionResumeLoad();
}
```

- [ ] **Step 4: Delete backend-menu references from menu application, dialog lifecycle, and global click handling**

```javascript
function applyDialogMenus() {
  editDependencyMenu.classList.toggle("open", editDependencyMenuOpen);
  newSessionCwdMenu.classList.toggle("open", newSessionCwdMenuOpen);
  newSessionModelMenu.classList.toggle("open", newSessionModelMenuOpen);
  newSessionProviderMenu.classList.toggle("open", newSessionProviderMenuOpen);
  newSessionReasoningMenu.classList.toggle("open", newSessionReasoningMenuOpen);
  newSessionResumeMenu.classList.toggle("open", newSessionResumeMenuOpen);
  editDependencyBtn.setAttribute("aria-expanded", editDependencyMenuOpen ? "true" : "false");
  newSessionCwdInput.setAttribute("aria-expanded", newSessionCwdMenuOpen ? "true" : "false");
  if (!newSessionCwdMenuOpen && newSessionCwdMenuFocus < 0) newSessionCwdInput.removeAttribute("aria-activedescendant");
  newSessionModelInput.setAttribute("aria-expanded", newSessionModelMenuOpen ? "true" : "false");
  if (!newSessionModelMenuOpen && newSessionModelMenuFocus < 0) newSessionModelInput.removeAttribute("aria-activedescendant");
  newSessionProviderBtn.setAttribute("aria-expanded", newSessionProviderMenuOpen ? "true" : "false");
  newSessionReasoningBtn.setAttribute("aria-expanded", newSessionReasoningMenuOpen ? "true" : "false");
  newSessionResumeBtn.setAttribute("aria-expanded", newSessionResumeMenuOpen ? "true" : "false");
  if (editDependencyMenuOpen) positionDialogMenu(editDependencyMenu, editDependencyBtn);
  if (newSessionCwdMenuOpen) positionDialogMenu(newSessionCwdMenu, newSessionCwdInput);
  if (newSessionModelMenuOpen) positionDialogMenu(newSessionModelMenu, newSessionModelInput);
  if (newSessionProviderMenuOpen) positionDialogMenu(newSessionProviderMenu, newSessionProviderBtn);
  if (newSessionReasoningMenuOpen) positionDialogMenu(newSessionReasoningMenu, newSessionReasoningBtn);
  if (newSessionResumeMenuOpen) positionDialogMenu(newSessionResumeMenu, newSessionResumeBtn);
}
```

```javascript
function hideNewSessionDialog() {
  newSessionStatus.textContent = "";
  newSessionCwdMenuOpen = false;
  newSessionCwdMenuFocus = -1;
  newSessionModelMenuOpen = false;
  newSessionModelMenuFocus = -1;
  newSessionProviderMenuOpen = false;
  newSessionReasoningMenuOpen = false;
  newSessionResumeMenuOpen = false;
  applyDialogMenus();
  newSessionBackdrop.style.display = "none";
  newSessionViewer.style.display = "none";
}
```

```javascript
if (newSessionProviderMenuOpen && !t.closest("#newSessionProviderField") && !t.closest("#newSessionProviderMenu")) {
  newSessionProviderMenuOpen = false;
}
// No backend picker branch remains here.
```

- [ ] **Step 5: Remove the deleted picker click handler and any helper that becomes unused after cleanup**

```javascript
// Delete this entire handler:
newSessionBackendBtn.onclick = (e) => {
  e.preventDefault();
  e.stopPropagation();
  renderNewSessionBackendMenu();
  newSessionBackendMenuOpen = !newSessionBackendMenuOpen;
  editDependencyMenuOpen = false;
  newSessionCwdMenuOpen = false;
  newSessionCwdMenuFocus = -1;
  newSessionModelMenuOpen = false;
  newSessionModelMenuFocus = -1;
  newSessionProviderMenuOpen = false;
  newSessionReasoningMenuOpen = false;
  newSessionResumeMenuOpen = false;
  applyDialogMenus();
};
```

- [ ] **Step 6: Run the targeted regression tests and verify they pass**

Run: `python3 -m pytest tests/test_pi_continue_ui.py -q`
Expected: PASS, including the new assertions that the bottom backend row and menu plumbing are gone.

- [ ] **Step 7: Run one broader UI regression slice for New session/static source safety**

Run: `python3 -m pytest tests/test_ui_regressions.py tests/test_static_assets.py -q`
Expected: PASS with no static asset or generic UI regressions.

- [ ] **Step 8: Manually inspect the modal in a browser if convenient; otherwise verify by code path review**

Run: `python3 -m pytest tests/test_pi_continue_ui.py tests/test_ui_regressions.py tests/test_static_assets.py -q`
Expected: PASS. Then confirm by reading `codoxear/static/app.js` that only `newSessionBackendTabs` remains as the visible backend control and no orphaned backend menu references remain.

- [ ] **Step 9: Commit the implementation**

```bash
git add codoxear/static/app.js tests/test_pi_continue_ui.py
git commit -m "refactor: remove duplicate new session backend picker"
```

## Self-Review

- **Spec coverage:** The plan removes the lower `Backend` field, preserves header tab switching, keeps backend-dependent UI via `setNewSessionBackend(...)`, and explicitly cleans up dialog/menu bookkeeping that would otherwise throw.
- **Placeholder scan:** No TODO/TBD placeholders remain; every test, file, and command is concrete.
- **Type consistency:** All identifiers match the current codebase: `newSessionBackendTabs`, `applyDialogMenus`, `setNewSessionBackend`, `newSessionProviderBtn`, `newSessionReasoningBtn`, and `newSessionResumeBtn`.
