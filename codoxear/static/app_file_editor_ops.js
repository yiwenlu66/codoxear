const global = window;

/* File editor save/unsaved-dialog keyboard and modal coordination. */
  "use strict";
  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`file editor dependency missing: ${name}`);
    return value;
  }
  function createFileEditorOpsController(options = {}) {
    const addAppEvent = requireFunction(options.addAppEvent, "addAppEvent");
    const { wiring, codoxearFileEditor, resolveAppUrl, fileDiff, normalizeLineNumber, requestAnimationFrame,
      setTimeout, hooks } = options;
    if (!codoxearFileEditor || typeof codoxearFileEditor.createFileEditorRuntime !== "function")
      throw new TypeError("file editor dependency missing: codoxearFileEditor");
    if (!hooks || typeof hooks !== "object") throw new TypeError("file editor dependency missing: hooks");
    const runtime = codoxearFileEditor.createFileEditorRuntime();
    const monacoLoader = codoxearFileEditor.createMonacoLoader(wiring.createMonacoLoaderOptions({
      resolveAppUrl, timeoutMs: 4000,
    }));
    const renderer = codoxearFileEditor.createFileEditorRenderer(wiring.createFileEditorRendererOptions({
      runtime, monacoLoader, host: fileDiff, normalizeLineNumber, requestAnimationFrame, setTimeout,
      isCurrentFileOpenRequest: hooks.isCurrentFileOpenRequest, renderPlainTextFallback: hooks.renderPlainTextFallback,
      disposeFileEditor: hooks.disposeFileEditor, currentEditorKind: hooks.currentEditorKind,
      setEditorKind: hooks.setEditorKind, currentFileEditMode: hooks.currentFileEditMode,
      currentActiveFileEditable: hooks.currentActiveFileEditable, isUnavailable: hooks.isUnavailable,
      isProgrammaticChange: hooks.isProgrammaticChange, currentTouchSelectMode: hooks.currentTouchSelectMode,
      resetTouchSelectionState: hooks.resetTouchSelectionState, currentActiveFileText: hooks.currentActiveFileText,
      setDirty: hooks.setDirty, runProgrammaticChange: hooks.runProgrammaticChange,
      syncReadOnly: hooks.syncReadOnly, updateTouchToolbar: hooks.updateTouchToolbar,
    }));
    return Object.freeze({
      runtime, monacoLoader, renderer,
      bindInteractions(bindings = {}) { return bindFileEditorInteractions({ ...bindings, addAppEvent }); },
    });
  }
  function bindFileEditorInteractions(options = {}) {
    const addAppEvent = requireFunction(options.addAppEvent, "addAppEvent");
    const { document, appConfirm, appConfirmFocusableControls, resolveAppConfirm, filePasteDialogRuntime,
      hideFilePasteDialog, fileUnsavedDialog, fileUnsavedController, isFileViewerOpen, requestHideFileViewer,
      sendChoice, closeSendChoiceDialog, queueViewer, hideQueueViewer, helpViewer, hideHelpViewer, diagViewer,
      hideDiagViewer, voiceController, hideVoiceSettingsDialog, sessionEditController, newSessionDialogController,
      handleFileEditorSaveShortcut, handleFileEditorDeleteKeydown, suppressFileEditorNativeDelete,
      fileTouchController } = options;
    addAppEvent(document, "keydown", (event) => fileTouchController.handleFileTouchSelectionKeydown(event), true);
    addAppEvent(document, "keydown", handleFileEditorSaveShortcut, true);
    addAppEvent(document, "keydown", handleFileEditorDeleteKeydown, true);
    addAppEvent(document, "beforeinput", suppressFileEditorNativeDelete, true);
    addAppEvent(document, "input", suppressFileEditorNativeDelete, true);
    addAppEvent(document, "keydown", (e) => {
      if (e.key === "Tab" && appConfirm.style.display === "flex") {
        const focusable = appConfirmFocusableControls(); e.preventDefault(); e.stopPropagation();
        if (!focusable.length) return;
        const currentIndex = focusable.indexOf(document.activeElement), offset = e.shiftKey ? -1 : 1;
        const nextIndex = currentIndex < 0 ? (e.shiftKey ? focusable.length - 1 : 0) : (currentIndex + offset + focusable.length) % focusable.length;
        try { focusable[nextIndex].focus({ preventScroll: true }); } catch {} return;
      }
      if (e.key !== "Escape") return;
      if (appConfirm.style.display === "flex") { e.preventDefault(); e.stopPropagation(); resolveAppConfirm(false); return; }
      if (filePasteDialogRuntime.isOpen()) { hideFilePasteDialog({ restoreFocus: true }); return; }
      if (fileUnsavedDialog.style.display === "flex") { fileUnsavedController.hideFileUnsavedDialog("cancel"); return; }
      if (isFileViewerOpen()) { e.preventDefault(); void requestHideFileViewer(); return; }
      if (sendChoice.style.display === "flex") { e.preventDefault(); e.stopPropagation(); closeSendChoiceDialog({ restoreFocus: true }); return; }
      if (queueViewer.style.display === "flex") hideQueueViewer();
      if (helpViewer.style.display === "flex") hideHelpViewer();
      if (diagViewer.style.display === "flex") hideDiagViewer();
      if (voiceController.isSettingsOpen()) hideVoiceSettingsDialog();
      if (sessionEditController.viewer.style.display === "flex" || sessionEditController.viewer.open) sessionEditController.hideEditSession();
      if (newSessionDialogController.isOpen()) newSessionDialogController.close();
    });
  }

export { createFileEditorOpsController };
