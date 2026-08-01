(function () {
  "use strict";

  // Loaded-chat navigation rail + document shortcut orchestration. Owns the
  // chat nav button projection (prev/next enabled/disabled state), loaded user-
  // message and loaded copy-message jump behavior (boundary toasts,
  // scrollIntoView with reduced-motion-aware behavior, row pulse), the prev/next
  // button click handlers, and the document `/` direct-to-search shortcut.
  //
  // The deeper loaded-chat search implementation, older-history search,
  // transcript rendering/runtime creation, the message-copy active-row
  // helpers, and `pulseNavigatedRow` remain in app.js (the pulse helper is
  // injected so a single row-pulse authority survives). Everything that
  // touches app-level runtime state (selected session, scroll position,
  // sidebar/modal open state, toasts, search opener, event registration) is
  // injected through createChatNavigationController(options) so the
  // controller has no hidden coupling to app.js globals and can be
  // exercised in a VM with fakes.

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`chat navigation controller dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object" || typeof value.style === "undefined") throw new TypeError(`chat navigation controller dependency missing: ${name}`);
    return value;
  }

  function requireArray(value, name) {
    if (!Array.isArray(value)) throw new TypeError(`chat navigation controller dependency missing: ${name}`);
    return value;
  }

  function createChatNavigationController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("chat navigation controller dependency missing: options");

    // DOM nodes (created and owned by app.js).
    const prevUserBtn = requireNode(options.prevUserBtn, "prevUserBtn");
    const nextUserBtn = requireNode(options.nextUserBtn, "nextUserBtn");

    // App-level runtime state accessors and effects.
    const getSelected = requireFunction(options.getSelected, "getSelected");
    const loadedUserMessageRows = requireFunction(options.loadedUserMessageRows, "loadedUserMessageRows");
    const loadedCopyMessageRows = requireFunction(options.loadedCopyMessageRows, "loadedCopyMessageRows");
    const loadedUserJumpTarget = requireFunction(options.loadedUserJumpTarget, "loadedUserJumpTarget");
    const loadedCopyJumpTarget = requireFunction(options.loadedCopyJumpTarget, "loadedCopyJumpTarget");
    const getScrollTop = requireFunction(options.getScrollTop, "getScrollTop");
    const prefersReducedMotion = requireFunction(options.prefersReducedMotion, "prefersReducedMotion");
    const pulseNavigatedRow = requireFunction(options.pulseNavigatedRow, "pulseNavigatedRow");
    const setToast = requireFunction(options.setToast, "setToast");
    const openChatSearch = requireFunction(options.openChatSearch, "openChatSearch");
    const isTextEntryElement = requireFunction(options.isTextEntryElement, "isTextEntryElement");
    const modalIsolationTargets = requireArray(options.modalIsolationTargets, "modalIsolationTargets");
    const isModalTargetOpen = requireFunction(options.isModalTargetOpen, "isModalTargetOpen");
    const addAppEvent = requireFunction(options.addAppEvent, "addAppEvent");

    // Optional injected targets / sidebar check. Default to the document/body
    // the controller is loaded into so production behavior is unchanged; tests
    // inject fakes.
    const documentTarget = options.documentTarget || document;
    const isSidebarOpen =
      typeof options.isSidebarOpen === "function"
        ? options.isSidebarOpen
        : () =>
          Boolean(documentTarget.body && typeof documentTarget.body.classList !== "undefined" && documentTarget.body.classList.contains("sidebar-open"));

    function scrollBehavior() {
      return "auto";
    }

    function syncButtons() {
      const enabled = Boolean(getSelected() && loadedUserMessageRows().length);
      prevUserBtn.disabled = !enabled;
      nextUserBtn.disabled = !enabled;
    }

    function jumpToLoadedUserMessage(direction) {
      const rows = loadedUserMessageRows();
      syncButtons();
      if (!rows.length) {
        setToast("No loaded user messages");
        return;
      }
      const result = loadedUserJumpTarget(rows, direction, getScrollTop() + 24);
      if (!result.target) {
        setToast(result.reason === "first" ? "At first loaded user message" : "At last loaded user message");
        return;
      }
      const target = result.target;
      target.scrollIntoView({ block: "start", behavior: scrollBehavior() });
      pulseNavigatedRow(target);
    }

    function jumpToLoadedMessage(direction) {
      const rows = loadedCopyMessageRows();
      if (!rows.length) {
        setToast("No loaded messages");
        return;
      }
      const result = loadedCopyJumpTarget(rows, direction, getScrollTop() + 24);
      if (!result.target) {
        setToast(result.reason === "first" ? "At first loaded message" : "At last loaded message");
        return;
      }
      const target = result.target;
      target.scrollIntoView({ block: "start", behavior: scrollBehavior() });
      pulseNavigatedRow(target);
    }

    prevUserBtn.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      jumpToLoadedUserMessage(-1);
    };
    nextUserBtn.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      jumpToLoadedUserMessage(1);
    };

    function chatNavigationShortcutBlocked(target) {
      if (!getSelected()) return true;
      if (isTextEntryElement(target)) return true;
      if (isSidebarOpen()) return true;
      return modalIsolationTargets.some(isModalTargetOpen);
    }

    function chatSearchShortcutBlocked(target) {
      return chatNavigationShortcutBlocked(target);
    }

    addAppEvent(documentTarget, "keydown", (e) => {
      if (e.defaultPrevented) return;
      if (e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey) {
        if (chatSearchShortcutBlocked(e.target)) return;
        e.preventDefault();
        openChatSearch();
      }
    });

    function dispose() {
      prevUserBtn.onclick = null;
      nextUserBtn.onclick = null;
    }

    return Object.freeze({
      syncButtons,
      jumpToLoadedUserMessage,
      jumpToLoadedMessage,
      chatNavigationShortcutBlocked,
      chatSearchShortcutBlocked,
      dispose,
    });
  }

  window.CodoxearChatNavigation = Object.freeze({ createChatNavigationController });
})();
