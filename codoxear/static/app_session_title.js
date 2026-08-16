

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`session title controller dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || !value.style || typeof value.setAttribute !== "function") {
      throw new TypeError(`session title controller dependency missing: ${name}`);
    }
    return value;
  }

  function createSessionTitleController(options = {}) {
    const titleLabel = requireNode(options.titleLabel, "titleLabel");
    const getSelected = requireFunction(options.getSelected, "getSelected");
    const openEditSession = requireFunction(options.openEditSession, "openEditSession");

    function openSelectedSessionEditor() {
      const sessionId = getSelected();
      if (sessionId) openEditSession(sessionId);
    }

    function syncTitleEditState() {
      const interactive = Boolean(getSelected());
      titleLabel.style.cursor = interactive ? "pointer" : "default";
      titleLabel.title = interactive ? "Edit conversation" : "No session selected";
      titleLabel.tabIndex = interactive ? 0 : -1;
      if (interactive) {
        titleLabel.setAttribute("role", "button");
        titleLabel.setAttribute("aria-label", "Edit conversation");
        titleLabel.removeAttribute("aria-disabled");
      } else {
        titleLabel.removeAttribute("role");
        titleLabel.removeAttribute("aria-label");
        titleLabel.setAttribute("aria-disabled", "true");
      }
    }

    titleLabel.onclick = openSelectedSessionEditor;
    titleLabel.onkeydown = (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      if (!getSelected()) return;
      event.preventDefault();
      openSelectedSessionEditor();
    };
    syncTitleEditState();

    return Object.freeze({ syncTitleEditState });
  }

export { createSessionTitleController };
