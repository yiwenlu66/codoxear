(function (global) {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`dialog menus controller dependency missing: ${name}`);
    return value;
  }

  function requireController(value, name) {
    if (!value || typeof value.applyMenus !== "function") {
      throw new TypeError(`dialog menus controller dependency missing: ${name}`);
    }
    return value;
  }

  function createDialogMenusController(options = {}) {
    const sessionEditController = requireFunction(options.sessionEditController, "sessionEditController");
    const newSessionDialogController = requireFunction(options.newSessionDialogController, "newSessionDialogController");

    function applyDialogMenus() {
      const sessionEditor = sessionEditController();
      if (sessionEditor) requireController(sessionEditor, "sessionEditController").applyMenus();
      requireController(newSessionDialogController(), "newSessionDialogController").applyMenus();
    }

    return Object.freeze({ applyDialogMenus });
  }

  global.CodoxearDialogMenus = Object.freeze({ createDialogMenusController });
})(window);
