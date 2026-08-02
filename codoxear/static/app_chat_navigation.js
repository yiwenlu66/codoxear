(function () {
  "use strict";

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

    const prevUserBtn = requireNode(options.prevUserBtn, "prevUserBtn");
    const nextUserBtn = requireNode(options.nextUserBtn, "nextUserBtn");
    const getSelected = requireFunction(options.getSelected, "getSelected");
    const getPollGen = typeof options.getPollGen === "function" ? options.getPollGen : () => 0;
    const api = typeof options.api === "function" ? options.api : async () => ({ total: loadedUserMessageRows().length, matches: [] });
    const loadTranscriptWindowAtCursor = typeof options.loadTranscriptWindowAtCursor === "function" ? options.loadTranscriptWindowAtCursor : async () => null;
    const loadedUserMessageRows = requireFunction(options.loadedUserMessageRows, "loadedUserMessageRows");
    const loadedCopyMessageRows = requireFunction(options.loadedCopyMessageRows, "loadedCopyMessageRows");
    const loadedUserJumpTarget = requireFunction(options.loadedUserJumpTarget, "loadedUserJumpTarget");
    const loadedCopyJumpTarget = requireFunction(options.loadedCopyJumpTarget, "loadedCopyJumpTarget");
    const getScrollTop = requireFunction(options.getScrollTop, "getScrollTop");
    const pulseNavigatedRow = requireFunction(options.pulseNavigatedRow, "pulseNavigatedRow");
    const setToast = requireFunction(options.setToast, "setToast");
    const openChatSearch = requireFunction(options.openChatSearch, "openChatSearch");
    const handleAppAuthLoss = typeof options.handleAppAuthLoss === "function" ? options.handleAppAuthLoss : () => {};
    const isTextEntryElement = requireFunction(options.isTextEntryElement, "isTextEntryElement");
    const modalIsolationTargets = requireArray(options.modalIsolationTargets, "modalIsolationTargets");
    const isModalTargetOpen = requireFunction(options.isModalTargetOpen, "isModalTargetOpen");
    const addAppEvent = requireFunction(options.addAppEvent, "addAppEvent");

    const documentTarget = options.documentTarget || document;
    const isSidebarOpen = typeof options.isSidebarOpen === "function"
      ? options.isSidebarOpen
      : () => Boolean(documentTarget.body && documentTarget.body.classList && documentTarget.body.classList.contains("sidebar-open"));
    const userTotals = new Map();
    const totalRequests = new Map();

    function rowForMatch(match) {
      if (!match) return null;
      const id = typeof match.message_id === "string" ? match.message_id : "";
      const cursor = typeof match.before_byte === "string" ? match.before_byte : "";
      return loadedUserMessageRows().find((row) => (id && row.dataset && row.dataset.messageId === id) || (cursor && row.dataset && row.dataset.historyCursor === cursor)) || null;
    }

    async function refreshUserTotal(sessionId) {
      if (!sessionId || totalRequests.has(sessionId)) return;
      const gen = getPollGen();
      const request = api(`/api/sessions/${sessionId}/search?q=*&role=user&limit=1`);
      totalRequests.set(sessionId, request);
      try {
        const data = await request;
        if (getSelected() !== sessionId || getPollGen() !== gen) return;
        userTotals.set(sessionId, Math.max(0, Number(data.total) || 0));
      } catch (error) {
        if (error && error.status === 401) handleAppAuthLoss();
      } finally {
        if (totalRequests.get(sessionId) === request) totalRequests.delete(sessionId);
        syncButtons();
      }
    }

    function syncButtons() {
      const sid = getSelected();
      if (!sid) {
        prevUserBtn.disabled = true;
        nextUserBtn.disabled = true;
        return;
      }
      const knownTotal = userTotals.get(sid);
      const disabled = knownTotal === 0;
      prevUserBtn.disabled = disabled;
      nextUserBtn.disabled = disabled;
      if (knownTotal === undefined) void refreshUserTotal(sid);
    }

    function scrollToRow(row) {
      if (!row) return false;
      row.scrollIntoView({ block: "start", behavior: "auto" });
      pulseNavigatedRow(row);
      return true;
    }

    async function fetchBoundaryUserMessage(direction, rows) {
      const sid = getSelected();
      const gen = getPollGen();
      if (!sid) return null;
      const anchorRow = direction < 0 ? rows[0] : rows[rows.length - 1];
      const anchor = anchorRow && anchorRow.dataset ? String(anchorRow.dataset.historyCursor || "") : "";
      let suffix;
      if (anchor) suffix = `&direction=${direction < 0 ? "previous" : "next"}&anchor=${encodeURIComponent(anchor)}`;
      else suffix = `&order=${direction < 0 ? "latest" : "first"}`;
      try {
        const data = await api(`/api/sessions/${sid}/search?q=*&role=user&limit=1${suffix}`);
        if (getSelected() !== sid || getPollGen() !== gen) return null;
        userTotals.set(sid, Math.max(userTotals.get(sid) || 0, Number(data.total) || 0));
        const matches = Array.isArray(data.matches) ? data.matches : [];
        return matches[0] || null;
      } catch (error) {
        if (error && error.status === 401) handleAppAuthLoss();
        return null;
      }
    }

    async function jumpToLoadedUserMessage(direction) {
      const rows = loadedUserMessageRows();
      syncButtons();
      if (!getSelected()) return;
      const local = rows.length ? loadedUserJumpTarget(rows, direction, getScrollTop() + 24) : { target: null, reason: "none" };
      if (local.target) {
        scrollToRow(local.target);
        return;
      }
      const match = await fetchBoundaryUserMessage(direction, rows);
      if (!match) {
        setToast(direction < 0 ? "At first user message" : "At last user message");
        return;
      }
      const cursor = typeof match.before_byte === "string" ? match.before_byte : "";
      if (!cursor || !(await loadTranscriptWindowAtCursor(cursor))) return;
      scrollToRow(rowForMatch(match));
    }

    function jumpToLoadedMessage(direction) {
      const rows = loadedCopyMessageRows();
      if (!rows.length) {
        setToast("No messages");
        return;
      }
      const result = loadedCopyJumpTarget(rows, direction, getScrollTop() + 24);
      if (!result.target) {
        setToast(result.reason === "first" ? "At first message" : "At last message");
        return;
      }
      scrollToRow(result.target);
    }

    prevUserBtn.onclick = (event) => {
      event.preventDefault();
      event.stopPropagation();
      void jumpToLoadedUserMessage(-1);
    };
    nextUserBtn.onclick = (event) => {
      event.preventDefault();
      event.stopPropagation();
      void jumpToLoadedUserMessage(1);
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

    addAppEvent(documentTarget, "keydown", (event) => {
      if (event.defaultPrevented) return;
      if (event.key === "/" && !event.ctrlKey && !event.metaKey && !event.altKey) {
        if (chatSearchShortcutBlocked(event.target)) return;
        event.preventDefault();
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
