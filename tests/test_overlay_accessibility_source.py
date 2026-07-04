import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_VOICE_JS = ROOT / "codoxear" / "static" / "app_voice.js"
APP_FILE_VIEWER_JS = ROOT / "codoxear" / "static" / "app_file_viewer.js"
APP_MODAL_JS = ROOT / "codoxear" / "static" / "app_modal.js"


def eval_modal_helpers() -> dict:
    source = APP_MODAL_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const calls = [];
        class FakeDialog {{ constructor(open) {{ this.open = open; }} }}
        const app = {{
          attrs: {{}},
          toggleAttribute(name, active) {{ if (active) this.attrs[name] = ""; else delete this.attrs[name]; }},
          setAttribute(name, value) {{ this.attrs[name] = value; }},
          removeAttribute(name) {{ delete this.attrs[name]; }},
        }};
        const focusTarget = {{ isConnected: true, disabled: false, focus(opts) {{ calls.push(["focus", opts && opts.preventScroll]); }} }};
        const ctx = {{
          HTMLDialogElement: FakeDialog,
          requestAnimationFrame: (fn) => {{ calls.push(["raf"]); fn(); }},
          window: {{}},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const modal = ctx.window.CodoxearModal;
        const dialogOpen = new FakeDialog(true);
        const displayOpen = {{ style: {{ display: "flex" }} }};
        const hidden = {{ style: {{ display: "none" }} }};
        const active = modal.syncModalIsolation(app, [hidden, displayOpen]);
        modal.restoreModalFocus(focusTarget, () => false, ctx.requestAnimationFrame);
        modal.restoreModalFocus(focusTarget, () => true, ctx.requestAnimationFrame);
        process.stdout.write(JSON.stringify({{
          dialogOpen: modal.isModalTargetOpen(dialogOpen),
          displayOpen: modal.isModalTargetOpen(displayOpen),
          hidden: modal.isModalTargetOpen(hidden),
          active,
          attrs: app.attrs,
          calls,
          frozen: Object.isFrozen(modal),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestOverlayAccessibilitySource(unittest.TestCase):
    def test_custom_modals_isolate_background_app(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        modal_source = APP_MODAL_JS.read_text(encoding="utf-8")
        self.assertIn('app_modal.js?v=__CODOXEAR_ASSET_VERSION__', (ROOT / "codoxear" / "static" / "index.html").read_text(encoding="utf-8"))
        self.assertIn("const codoxearModal = window.CodoxearModal;", source)
        self.assertIn('throw new Error("Codoxear modal helpers failed to load")', source)
        self.assertIn("const modalIsolationTargets = [", source)
        self.assertIn('app.toggleAttribute("inert", active);', modal_source)
        self.assertIn('app.setAttribute("aria-hidden", "true");', modal_source)
        self.assertIn("return codoxearModal.syncModalIsolation(app, modalIsolationTargets);", source)
        self.assertIn("function prepareModalOpen(options = {})", source)
        self.assertIn("closeTransientOverlays(options);", source)

    def test_modal_openers_close_transient_overlays(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("if (unattendedController.isOpen()) hideUnattendedMenu();", source)
        self.assertIn('if (document.body.classList.contains("sidebar-open")) setSidebarOpen(false);', source)
        self.assertIn('filePickerMenuState.close();', source)
        self.assertNotIn('fileMenuOpen = false;', source)
        self.assertIn('prepareModalOpen();\n          helpBackdrop.style.display = "block";', source)
        # Queue show/hide modal behavior moved into the CodoxearQueue controller;
        # app.js keeps only the delegating wrapper. The controller owns the
        # prepareModalOpen/queueViewerSid coordination.
        self.assertIn('return queueController.showQueueViewer(opts);', source)
        queue_source = (ROOT / "codoxear" / "static" / "app_queue.js").read_text(encoding="utf-8")
        self.assertIn('prepareModalOpen();\n      queueViewerSid = selected;', queue_source)
        # Voice settings show/hide modal behavior moved into the CodoxearVoice
        # controller; app.js keeps only the DOM nodes and the delegating wrapper.
        voice_source = APP_VOICE_JS.read_text(encoding="utf-8")
        self.assertIn('prepareModalOpen();\n      voiceSettingsReturnFocusEl = documentTarget.activeElement instanceof HTMLElement ? documentTarget.activeElement : null;\n      voiceSettingsBackdrop.style.display = "block";', voice_source)

    def test_settings_dialog_uses_modal_and_cancel_semantics(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        voice_source = APP_VOICE_JS.read_text(encoding="utf-8")
        show_start = voice_source.index("function showVoiceSettingsDialog()")
        show_end = voice_source.index("function hideVoiceSettingsDialog()", show_start)
        show_block = voice_source[show_start:show_end]
        hide_start = show_end
        hide_end = voice_source.index("announceBtn.onclick", hide_start)
        hide_block = voice_source[hide_start:hide_end]
        self.assertIn("voiceSettingsReturnFocusEl = documentTarget.activeElement instanceof HTMLElement ? documentTarget.activeElement : null;", show_block)
        self.assertIn('if (typeof voiceSettingsViewer.showModal === "function" && !voiceSettingsViewer.open) voiceSettingsViewer.showModal();', show_block)
        self.assertIn("const focusTarget = voiceSettingsReturnFocusEl;", hide_block)
        self.assertIn("voiceSettingsReturnFocusEl = null;", hide_block)
        self.assertIn('if (typeof voiceSettingsViewer.close === "function" && voiceSettingsViewer.open) voiceSettingsViewer.close();', hide_block)
        self.assertIn("restoreModalFocus(focusTarget, restore, requestFrame);", hide_block)
        self.assertIn('addEvent(voiceSettingsViewer, "cancel", (e) => {', voice_source)
        self.assertIn("e.preventDefault();\n      hideVoiceSettingsDialog();", voice_source)
        # app.js now uses the controller's canonical open-state query instead
        # of relying on style.display alone.
        self.assertIn("if (voiceController.isSettingsOpen()) hideVoiceSettingsDialog();", source)
        self.assertNotIn('voiceSettingsViewer.style.display === "flex") hideVoiceSettingsDialog()', source)

    def test_new_session_dialog_restores_focus_and_sets_initial_focus(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('role: "dialog", "aria-modal": "true", "aria-label": "New session"', source)
        self.assertIn("let newSessionReturnFocusEl = null;", source)
        open_start = source.index("function openNewSessionDialog(")
        open_end = source.index("editPriorityRange.oninput", open_start)
        open_block = source[open_start:open_end]
        self.assertIn("newSessionReturnFocusEl = returnFocusEl instanceof HTMLElement ? returnFocusEl : document.activeElement instanceof HTMLElement ? document.activeElement : null;", open_block)
        self.assertIn("prepareModalOpen();", open_block)
        self.assertIn("focusNewSessionInitialControl();", open_block)
        self.assertIn("function focusNewSessionInitialControl()", source)
        focus_start = source.index("function focusNewSessionInitialControl()")
        focus_end = source.index("function hideNewSessionDialog", focus_start)
        focus_block = source[focus_start:focus_end]
        self.assertIn("const target = isMobile() ? newSessionCloseBtn : newSessionCwdInput;", focus_block)
        self.assertIn("target.focus({ preventScroll: true });", focus_block)
        self.assertIn("newSessionCwdInput.setSelectionRange(end, end);", focus_block)
        hide_start = source.index("function hideNewSessionDialog")
        hide_end = source.index("function openNewSessionDialog", hide_start)
        hide_block = source[hide_start:hide_end]
        self.assertIn("const wasOpen = isModalTargetOpen(newSessionViewer);", hide_block)
        self.assertIn("if (wasOpen) restoreNewSessionFocus();", hide_block)
        restore_start = source.index("function restoreNewSessionFocus()")
        restore_end = source.index("function focusNewSessionInitialControl()", restore_start)
        restore_block = source[restore_start:restore_end]
        self.assertIn("const target = newSessionReturnFocusEl;", restore_block)
        self.assertIn("newSessionReturnFocusEl = null;", restore_block)
        self.assertIn("target.focus({ preventScroll: true });", restore_block)
        self.assertIn("if (isModalTargetOpen(newSessionViewer)) return;", restore_block)
        self.assertIn('if (newSessionViewer.style.display === "flex") hideNewSessionDialog();', source)
        self.assertIn('if (brokerPid) hideNewSessionDialog();', source)

    def test_file_viewer_dialog_restores_focus_and_announces_status(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertIn('id: "fileStatus", role: "status", "aria-live": "polite"', source)
        self.assertIn('id: "fileViewer", role: "dialog", "aria-modal": "true", "aria-label": "File viewer"', source)
        self.assertIn('id: "fileUnsavedDialog", role: "dialog", "aria-modal": "true", "aria-label": "Unsaved file changes"', source)
        self.assertIn('id: "filePasteDialog", role: "dialog", "aria-modal": "true", "aria-label": "Paste into file"', source)
        self.assertNotIn("let fileViewerReturnFocusEl = null;", source)
        self.assertIn("let fileViewerReturnFocusElement = null;", viewer_source)
        show_start = source.index("async function showFileViewer")
        hide_start = source.index("function hideFileViewer", show_start)
        show_block = source[show_start:hide_start]
        hide_end = source.index("function handleFileViewerSessionUnavailable", hide_start)
        hide_block = source[hide_start:hide_end]
        lifecycle_start = viewer_source.index("function createFileViewerLifecycleRuntime(options = {})")
        lifecycle_end = viewer_source.index("function createFileCandidateRefreshRuntime", lifecycle_start)
        lifecycle_block = viewer_source[lifecycle_start:lifecycle_end]
        self.assertIn("return await fileViewerLifecycleRuntime.show({ path, mode, line, pickerQuery });", show_block)
        self.assertIn("const wasOpen = deps.isFileViewerOpen();", lifecycle_block)
        self.assertIn("showModal: (options) => fileViewerModalRuntime.show({ ...options, activeElement: document.activeElement, ElementCtor: HTMLElement })", source)
        self.assertIn("ui.showModal({ wasOpen, queryOpen });", lifecycle_block)
        self.assertIn("setReturnFocusElement: (element, ElementCtor) => fileViewerController.setFileViewerReturnFocusElement(element, ElementCtor)", source)
        self.assertNotIn("if (!wasOpen) fileViewerController.setFileViewerReturnFocusElement(document.activeElement, HTMLElement);", show_block)
        self.assertNotIn("prepareModalOpen();", show_block)
        self.assertIn("const queryOpen = !explicitPath && query !== \"\";", lifecycle_block)
        self.assertIn("function createFileViewerModalRuntime(options = {})", viewer_source)
        self.assertIn("if (!wasOpen && queryOpen) focusPickerInput();", viewer_source)
        self.assertIn("pickerInput.focus({ preventScroll: true });", viewer_source)
        self.assertIn("focusModalCloseButton,", source)
        self.assertNotIn("} else if (!wasOpen) {\n            focusModalCloseButton(fileViewer, fileCloseBtn);\n          }", show_block)
        self.assertIn("if (queryOpen) {\n        ui.resetFileViewerPanel();", lifecycle_block)
        self.assertIn("ui.openFilePickerSearchQuery(query, { line, suppressDraft: true });", lifecycle_block)
        self.assertIn("await deps.refreshFileCandidates({ sessionId: sid, syncToken });", lifecycle_block)
        self.assertIn("if (queryOpen) {\n        ui.focusFilePickerInput();", lifecycle_block)
        self.assertIn("filePickerInput.focus({ preventScroll: true });", source)
        self.assertIn("beginHide: () => fileViewerModalRuntime.beginHide()", source)
        self.assertIn("finishHide: (state) => fileViewerModalRuntime.finishHide(state)", source)
        self.assertIn("const hideState = beginHide();", lifecycle_block)
        self.assertIn("takeReturnFocusElement: () => fileViewerController.takeFileViewerReturnFocusElement()", source)
        self.assertNotIn("fileViewerReturnFocusEl = null;", hide_block)
        self.assertIn("finishHide(hideState);", lifecycle_block)
        self.assertIn("restoreModalFocus,", source)
        self.assertNotIn("let fileUnsavedReturnFocusEl = null;", source)
        self.assertIn("let fileUnsavedReturnFocusElement = null;", viewer_source)
        self.assertNotIn("let fileUnsavedResolver = null;", source)
        self.assertIn("function focusInitialControl()", viewer_source)
        self.assertIn("focusInitialControl();", viewer_source)
        self.assertNotIn("function focusFileUnsavedInitialControl()", source)
        self.assertIn("return fileUnsavedDialogRuntime.promptChoice(document.activeElement, HTMLElement);", source)
        self.assertIn("setReturnFocusElement: (element, ElementCtor) => fileViewerController.setFileUnsavedReturnFocusElement(element, ElementCtor)", source)
        self.assertIn("takeReturnFocusElement: () => fileViewerController.takeFileUnsavedReturnFocusElement()", source)
        self.assertIn("restoreModalFocus,", source)
        self.assertIn("isModalTargetOpen,", source)
        self.assertNotIn("fileViewerController.setFileUnsavedReturnFocusElement(document.activeElement, HTMLElement);", source)
        self.assertNotIn("fileViewer.setAttribute(\"inert\", \"\");", source)
        self.assertNotIn("fileViewer.removeAttribute(\"inert\");", source)
        self.assertNotIn("const focusTarget = fileViewerController.takeFileUnsavedReturnFocusElement();", source)

    def test_queue_help_details_dialogs_restore_focus(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        queue_source = (ROOT / "codoxear" / "static" / "app_queue.js").read_text(encoding="utf-8")
        diag_source = (ROOT / "codoxear" / "static" / "app_diagnostics.js").read_text(encoding="utf-8")
        for label in [
            'id: "queueViewer", role: "dialog", "aria-modal": "true", "aria-label": "Queued messages"',
            'id: "helpViewer", role: "dialog", "aria-modal": "true", "aria-label": "Help"',
            'id: "diagViewer", role: "dialog", "aria-modal": "true", "aria-label": "Details"',
        ]:
            self.assertIn(label, source)
        # Queue + Diag return-focus state is now owned by their controllers;
        # Help keeps its app.js-owned return-focus element.
        self.assertIn("let queueReturnFocusEl = null;", queue_source)
        self.assertNotIn("let queueReturnFocusEl = null;", source)
        self.assertIn("let diagReturnFocusEl = null;", diag_source)
        self.assertNotIn("let diagReturnFocusEl = null;", source)
        self.assertIn("let helpReturnFocusEl = null;", source)
        self.assertIn("function restoreModalFocus(target, isStillOpen)", source)
        self.assertIn("return codoxearModal.restoreModalFocus(target, isStillOpen);", source)
        self.assertIn("function focusModalCloseButton(viewer, closeBtn)", source)
        self.assertIn("return codoxearModal.focusModalCloseButton(viewer, closeBtn);", source)
        # Help still owns its show/hide bodies in app.js.
        help_show_start = source.index("function showHelpViewer")
        help_hide_start = source.index("function hideHelpViewer", help_show_start)
        help_show_block = source[help_show_start:help_hide_start]
        help_next_fn = source.find("\n        function ", help_hide_start + 1)
        if help_next_fn == -1:
            help_next_fn = len(source)
        help_hide_block = source[help_hide_start:help_next_fn]
        self.assertIn("helpReturnFocusEl = opener instanceof HTMLElement ? opener : document.activeElement instanceof HTMLElement ? document.activeElement : null;", help_show_block)
        self.assertIn("prepareModalOpen();", help_show_block)
        self.assertIn("focusModalCloseButton(helpViewer, helpCloseBtn);", help_show_block)
        self.assertIn("const wasOpen = isModalTargetOpen(helpViewer);", help_hide_block)
        self.assertIn("const focusTarget = helpReturnFocusEl;", help_hide_block)
        self.assertIn("helpReturnFocusEl = null;", help_hide_block)
        self.assertIn("restoreModalFocus(focusTarget, () => isModalTargetOpen(helpViewer));", help_hide_block)
        # Diag show/hide bodies moved into the CodoxearDiagnostics controller
        # module with the same modal/focus contract: prepare modal open, focus
        # close button on show, restore focus to opener on hide (unless skipped).
        diag_show_start = diag_source.index("async function show({ opener = null } = {}) {")
        diag_hide_start = diag_source.index("function hide({ restoreFocus = true } = {}) {", diag_show_start)
        diag_show_block = diag_source[diag_show_start:diag_hide_start]
        diag_newlike_start = diag_source.index("function onNewLikeClick(e) {", diag_hide_start)
        diag_hide_block = diag_source[diag_hide_start:diag_newlike_start]
        self.assertIn("diagReturnFocusEl = opener instanceof HTMLElement ? opener : document.activeElement instanceof HTMLElement ? document.activeElement : null;", diag_show_block)
        self.assertIn("prepareModalOpen();", diag_show_block)
        self.assertIn('diagBackdrop.style.display = "block";', diag_show_block)
        self.assertIn("focusModalCloseButton(diagViewer, diagCloseBtn, requestFrame);", diag_show_block)
        self.assertIn("const wasOpen = isModalTargetOpen(diagViewer);", diag_hide_block)
        self.assertIn("const focusTarget = diagReturnFocusEl;", diag_hide_block)
        self.assertIn("diagReturnFocusEl = null;", diag_hide_block)
        self.assertIn("if (restoreFocus && wasOpen) restoreModalFocus(focusTarget, () => isModalTargetOpen(diagViewer), requestFrame);", diag_hide_block)
        # New-like click hides the diag modal without focus restore before
        # opening the New Session dialog (owned by the controller module).
        diag_newlike_block = diag_source[diag_newlike_start:diag_source.index("async function onCopyClick(e) {", diag_newlike_start)]
        self.assertIn("hide({ restoreFocus: false });", diag_newlike_block)
        self.assertIn('openNewSessionDialog({ likeSession: preset, statusText: "Review copied launch settings before starting.", returnFocusEl });', diag_newlike_block)
        # Queue show/hide live in the controller module with the same modal/focus
        # contract: prepare modal open, focus close button on show, restore focus
        # to opener or recovery-panel/queue-button fallback on hide.
        queue_show_start = queue_source.index("function showQueueViewer({ opener = null } = {})")
        queue_hide_start = queue_source.index("function hideQueueViewer()", queue_show_start)
        queue_show_block = queue_source[queue_show_start:queue_hide_start]
        queue_hide_block = queue_source[queue_hide_start:queue_source.index("function dispose()", queue_hide_start)]
        self.assertIn("queueReturnFocusEl = opener instanceof HTMLElement ? opener : document.activeElement instanceof HTMLElement ? document.activeElement : null;", queue_show_block)
        self.assertIn("prepareModalOpen();", queue_show_block)
        self.assertIn('queueBackdrop.style.display = "block";', queue_show_block)
        self.assertIn("focusModalCloseButton(queueViewer, queueCloseBtn, requestFrame);", queue_show_block)
        self.assertIn("const wasOpen = isModalTargetOpen(queueViewer);", queue_hide_block)
        self.assertIn("const focusTarget = queueReturnFocusEl;", queue_hide_block)
        self.assertIn("queueReturnFocusEl = null;", queue_hide_block)
        self.assertIn("const fallback = recoveryPanelFocusFallback() || queueBtn || null;", queue_hide_block)
        self.assertIn("restoreModalFocus(focusTarget && focusTarget.isConnected ? focusTarget : fallback, () => isModalTargetOpen(queueViewer), requestFrame);", queue_hide_block)
        # app.js keeps the diag opener/close/backdrop wiring and thin wrappers.
        self.assertIn("showQueueViewer({ opener: e.currentTarget });", source)
        self.assertIn("showHelpViewer({ opener: e.currentTarget });", source)
        self.assertIn("showDiagViewer({ opener: e.currentTarget });", source)
        self.assertIn("return diagController.show(opts);", source)
        self.assertIn("return diagController.hide(opts);", source)

    def test_modal_module_preserves_open_isolation_and_focus_contracts(self) -> None:
        result = eval_modal_helpers()
        self.assertTrue(result["dialogOpen"])
        self.assertTrue(result["displayOpen"])
        self.assertFalse(result["hidden"])
        self.assertTrue(result["active"])
        self.assertEqual(result["attrs"], {"inert": "", "aria-hidden": "true"})
        self.assertEqual(result["calls"], [["raf"], ["focus", True], ["raf"]])
        self.assertTrue(result["frozen"])

    def test_help_copy_lists_claude_backend(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("Right now that is <b>Codex</b>, <b>Pi</b>, and <b>Claude</b>.", source)
        self.assertNotIn("Right now that is <b>Codex</b> and <b>Pi</b>.", source)


if __name__ == "__main__":
    unittest.main()
