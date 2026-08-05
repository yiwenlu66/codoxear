(function (global) {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`file viewer integration dependency missing: ${name}`);
    return value;
  }

  function requireObject(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`file viewer integration dependency missing: ${name}`);
    return value;
  }

  function createFileViewerIntegration(options = {}) {
    const selectedSessionLaunchFailed = requireFunction(options.selectedSessionLaunchFailed, "selectedSessionLaunchFailed");
    const setToast = requireFunction(options.setToast, "setToast");
    const lifecycleRuntime = requireObject(options.lifecycleRuntime, "lifecycleRuntime");
    const fileViewerController = requireObject(options.fileViewerController, "fileViewerController");
    const fileLoadResultRuntime = requireObject(options.fileLoadResultRuntime, "fileLoadResultRuntime");
    const confirmAction = requireFunction(options.confirmAction, "confirmAction");
    const fileUnsavedController = requireObject(options.fileUnsavedController, "fileUnsavedController");

    function openFileViewer({ path = "", mode = "", manual = false, line = null, pickerQuery = "" } = {}) {
      void manual;
      if (selectedSessionLaunchFailed()) {
        setToast("failed launch has no file browser");
        return false;
      }
      return lifecycleRuntime.show({ path, mode, line, pickerQuery });
    }

    function closeFileViewer() {
      return lifecycleRuntime.hide();
    }

    function setFileEditMode(nextMode) {
      return fileViewerController.setFileEditMode(nextMode);
    }

    async function applyFileLoadResult(rel, result, request, { viewMode = "file" } = {}) {
      return await fileLoadResultRuntime.apply(rel, result, request, { viewMode });
    }

    function handleFileTouchSelectionKeydown(event) {
      return fileViewerController.handleFileTouchSelectionKeydown(event);
    }

    function confirmReload(message) {
      return confirmAction({
        title: "Reload file from disk?",
        message,
        confirmText: "Reload",
        cancelText: "Cancel",
        destructive: true,
      });
    }

    function promptUnsavedFileChoice() {
      return fileUnsavedController.promptFileUnsavedChoice();
    }

    return Object.freeze({
      openFileViewer,
      closeFileViewer,
      setFileEditMode,
      applyFileLoadResult,
      handleFileTouchSelectionKeydown,
      confirmReload,
      promptUnsavedFileChoice,
    });
  }

  global.CodoxearFileViewerIntegration = Object.freeze({ createFileViewerIntegration });
})(window);
