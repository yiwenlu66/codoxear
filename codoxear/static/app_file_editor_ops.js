
/* File editor save/unsaved-dialog keyboard and modal coordination. */
  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`file editor dependency missing: ${name}`);
    return value;
  }
  function createFileEditorOpsController(options = {}) {
    const addAppEvent = requireFunction(options.addAppEvent, "addAppEvent");
    const { wiring, codoxearFileEditor, resolveAppUrl, subscribeTheme, fileDiff, normalizeLineNumber, requestAnimationFrame,
      setTimeout, isCurrentFileOpenRequest, renderPlainTextFallback, disposeFileEditor, currentEditorKind,
      setEditorKind, currentFileEditMode, currentActiveFileEditable, isUnavailable, isProgrammaticChange,
      currentTouchSelectMode, resetTouchSelectionState, currentActiveFileText, setDirty, runProgrammaticChange,
      syncReadOnly, updateTouchToolbar } = options;
    if (!codoxearFileEditor || typeof codoxearFileEditor.createFileEditorRuntime !== "function")
      throw new TypeError("file editor dependency missing: codoxearFileEditor");
    const runtime = codoxearFileEditor.createFileEditorRuntime();
    const monacoLoader = codoxearFileEditor.createMonacoLoader(wiring.createMonacoLoaderOptions({
      resolveAppUrl, timeoutMs: 4000, subscribeTheme,
    }));
    const renderer = codoxearFileEditor.createFileEditorRenderer(wiring.createFileEditorRendererOptions({
      runtime, monacoLoader, host: fileDiff, normalizeLineNumber, requestAnimationFrame, setTimeout,
      isCurrentFileOpenRequest: isCurrentFileOpenRequest, renderPlainTextFallback: renderPlainTextFallback,
      disposeFileEditor: disposeFileEditor, currentEditorKind: currentEditorKind,
      setEditorKind: setEditorKind, currentFileEditMode: currentFileEditMode,
      currentActiveFileEditable: currentActiveFileEditable, isUnavailable: isUnavailable,
      isProgrammaticChange: isProgrammaticChange, currentTouchSelectMode: currentTouchSelectMode,
      resetTouchSelectionState: resetTouchSelectionState, currentActiveFileText: currentActiveFileText,
      setDirty: setDirty, runProgrammaticChange: runProgrammaticChange,
      syncReadOnly: syncReadOnly, updateTouchToolbar: updateTouchToolbar,
    }));
    return Object.freeze({
      runtime, monacoLoader, renderer,
      bindInteractions(bindings = {}) { return bindFileEditorInteractions({ ...bindings, addAppEvent }); },
    });
  }
  function bindFileEditorInteractions(options = {}) {
    // Escape never dismisses a modal dialog anywhere in the app (global
    // policy): every dialog closes through its own buttons, backdrop, or
    // keyboard hints. Only the #appConfirm Tab focus trap remains here.
    const addAppEvent = requireFunction(options.addAppEvent, "addAppEvent");
    const { document, appConfirm, appConfirmFocusableControls,
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
    });
  }

export { createFileEditorOpsController };
