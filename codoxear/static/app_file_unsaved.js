const global = window;


  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`file unsaved controller dependency missing: ${name}`);
    return value;
  }

  function requireRuntime(value) {
    if (!value || typeof value.promptChoice !== "function" || typeof value.hide !== "function") {
      throw new TypeError("file unsaved controller dependency missing: dialogRuntime");
    }
    return value;
  }

  function createFileUnsavedController(options = {}) {
    const documentTarget = options.documentTarget || global.document;
    const ElementCtor = options.ElementCtor || global.HTMLElement;
    const dialogRuntime = requireRuntime(options.dialogRuntime);
    const getFileViewerController = requireFunction(options.getFileViewerController, "getFileViewerController");

    function fileViewerController() {
      const controller = getFileViewerController();
      if (!controller) throw new Error("file unsaved controller file viewer is unavailable");
      return controller;
    }

    function promptFileUnsavedChoice() {
      return dialogRuntime.promptChoice(documentTarget.activeElement, ElementCtor);
    }

    function hideFileUnsavedDialog(choice = "cancel") {
      return dialogRuntime.hide(choice);
    }

    function maybeHandleUnsavedFileChanges() {
      return fileViewerController().maybeHandleUnsavedFileChanges();
    }

    function handleFileUnsavedSaveChoice() {
      return fileViewerController().handleFileUnsavedSaveChoice();
    }

    function handleFileUnsavedDiscardChoice() {
      return fileViewerController().handleFileUnsavedDiscardChoice();
    }

    function handleFileUnsavedCancelChoice() {
      return fileViewerController().handleFileUnsavedCancelChoice();
    }

    return Object.freeze({
      promptFileUnsavedChoice,
      hideFileUnsavedDialog,
      maybeHandleUnsavedFileChanges,
      handleFileUnsavedSaveChoice,
      handleFileUnsavedDiscardChoice,
      handleFileUnsavedCancelChoice,
    });
  }

export { createFileUnsavedController };
