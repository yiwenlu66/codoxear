(function (global) {
  "use strict";

  function createFileTouchController(options = {}) {
    if (typeof options.fileViewerController !== "function") {
      throw new TypeError("file touch controller dependency missing: fileViewerController");
    }

    function handleFileTouchSelectionKeydown(event) {
      const controller = options.fileViewerController();
      if (!controller || typeof controller.handleFileTouchSelectionKeydown !== "function") {
        throw new TypeError("file touch controller dependency missing: fileViewerController.handleFileTouchSelectionKeydown");
      }
      return controller.handleFileTouchSelectionKeydown(event);
    }

    return Object.freeze({ handleFileTouchSelectionKeydown });
  }

  global.CodoxearFileTouch = Object.freeze({ createFileTouchController });
})(window);
