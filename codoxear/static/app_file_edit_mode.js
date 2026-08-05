(function (global) {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`file edit mode controller dependency missing: ${name}`);
    return value;
  }

  function createFileEditModeController(options = {}) {
    const fileViewerController = requireFunction(options.fileViewerController, "fileViewerController");

    function setFileEditMode(nextMode) {
      const viewer = fileViewerController();
      if (!viewer || typeof viewer.setFileEditMode !== "function") {
        throw new TypeError("file edit mode controller dependency missing: fileViewerController.setFileEditMode");
      }
      return viewer.setFileEditMode(nextMode);
    }

    return Object.freeze({ setFileEditMode });
  }

  global.CodoxearFileEditMode = Object.freeze({ createFileEditModeController });
})(window);
