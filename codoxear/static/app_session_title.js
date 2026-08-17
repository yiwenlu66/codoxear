

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
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function" || typeof sessionState.subscribe !== "function") throw new TypeError("session title controller dependency missing: sessionState");
    const sessionCatalog = options.sessionCatalog;
    if (!sessionCatalog || typeof sessionCatalog.get !== "function" || typeof sessionCatalog.subscribe !== "function") throw new TypeError("session title controller dependency missing: sessionCatalog");
    const sessionTitleWithId = requireFunction(options.sessionTitleWithId, "sessionTitleWithId");
    const openEditSession = requireFunction(options.openEditSession, "openEditSession");

    function syncTitleValue() {
      const sessionId = sessionState.get("selected");
      const session = sessionId ? sessionCatalog.get("sessionIndex").get(sessionId) : null;
      titleLabel.textContent = session ? sessionTitleWithId(session) : sessionId ? String(sessionId) : "No session selected";
    }

    function openSelectedSessionEditor() {
      const sessionId = sessionState.get("selected");
      if (sessionId) openEditSession(sessionId);
    }

    function syncTitleEditState() {
      const interactive = Boolean(sessionState.get("selected"));
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
      if (!sessionState.get("selected")) return;
      event.preventDefault();
      openSelectedSessionEditor();
    };
    const unsubscribeSelected = sessionState.subscribe("selected", () => {
      syncTitleValue();
      syncTitleEditState();
    });
    const unsubscribeSessionIndex = sessionCatalog.subscribe("sessionIndex", syncTitleValue);
    syncTitleValue();
    syncTitleEditState();

    return Object.freeze({
      syncTitleValue,
      syncTitleEditState,
      dispose() {
        unsubscribeSelected();
        unsubscribeSessionIndex();
      },
    });
  }

export { createSessionTitleController };
