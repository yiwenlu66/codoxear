(function (global) {
  "use strict";

  function requireWindowTarget(value) {
    if (!value || typeof value.innerHeight !== "number") {
      throw new TypeError("dialog menu controller dependency missing: windowTarget");
    }
    return value;
  }

  function createDialogMenusController(options = {}) {
    function requireDialogMenusFunction(value, name) {
      if (typeof value !== "function") throw new TypeError(`dialog menus controller dependency missing: ${name}`);
      return value;
    }

    function requireDialogMenusController(value, name) {
      if (!value || typeof value.applyMenus !== "function") {
        throw new TypeError(`dialog menus controller dependency missing: ${name}`);
      }
      return value;
    }

    const sessionEditController = requireDialogMenusFunction(options.sessionEditController, "sessionEditController");
    const newSessionDialogController = requireDialogMenusFunction(options.newSessionDialogController, "newSessionDialogController");

    function applyDialogMenus() {
      const sessionEditor = sessionEditController();
      if (sessionEditor) requireDialogMenusController(sessionEditor, "sessionEditController").applyMenus();
      requireDialogMenusController(newSessionDialogController(), "newSessionDialogController").applyMenus();
    }

    return Object.freeze({ applyDialogMenus });
  }

  function createDialogMenuController(options = {}) {
    const windowTarget = requireWindowTarget(options.windowTarget || global);

    function positionDialogMenu(menu, anchorBtn) {
      if (!menu || !anchorBtn) return;
      const host = menu.parentElement;
      if (!host) return;
      const visualViewport = windowTarget.visualViewport;
      const rect = anchorBtn.getBoundingClientRect();
      const hostRect = host.getBoundingClientRect();
      const viewportWidth = hostRect.width;
      const viewportTop = visualViewport ? visualViewport.offsetTop : 0;
      const viewportBottom = viewportTop + (visualViewport ? visualViewport.height : windowTarget.innerHeight);
      const margin = 12;
      const desiredWidth = Math.min(Math.max(rect.width, 280), viewportWidth - margin * 2);
      menu.style.position = "absolute";
      const left = Math.max(margin, Math.min(viewportWidth - margin - desiredWidth, rect.left - hostRect.left));
      menu.style.left = `${left}px`;
      menu.style.width = `${desiredWidth}px`;
      menu.style.right = "auto";
      menu.style.bottom = "auto";
      menu.style.maxHeight = "";
      const menuHeight = Math.min(menu.scrollHeight || 260, Math.floor((viewportBottom - viewportTop) * 0.5));
      const spaceBelow = viewportBottom - rect.bottom - margin;
      const spaceAbove = rect.top - viewportTop - margin;
      const openAbove = spaceBelow < Math.min(220, menuHeight) && spaceAbove > spaceBelow;
      if (openAbove) {
        const maxHeight = Math.max(120, spaceAbove - 8);
        menu.style.maxHeight = `${maxHeight}px`;
        const top = Math.max(viewportTop + margin - hostRect.top, rect.top - hostRect.top - Math.min(menuHeight, maxHeight) - 8);
        menu.style.top = `${top}px`;
      } else {
        const maxHeight = Math.max(120, spaceBelow - 8);
        menu.style.maxHeight = `${maxHeight}px`;
        const top = Math.min(viewportBottom - margin - hostRect.top - Math.min(menuHeight, maxHeight), rect.bottom - hostRect.top + 8);
        menu.style.top = `${top}px`;
      }
    }

    return Object.freeze({ positionDialogMenu });
  }

  global.CodoxearDialogMenus = Object.freeze({ createDialogMenusController });
  global.CodoxearDialogMenu = { createDialogMenuController };
})(window);
