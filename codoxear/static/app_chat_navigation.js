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
    const loadOlderMessages = requireFunction(options.loadOlderMessages, "loadOlderMessages");
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
      const cursor = typeof match.history_cursor === "string" ? match.history_cursor : (typeof match.before_byte === "string" ? match.before_byte : "");
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

    async function fetchNeighbor(direction, rows) {
      // stale means the selected session or poll generation moved on while the
      // request was in flight: the answer no longer applies, so the caller
      // stays silent. error means this navigation genuinely failed and the
      // caller should say so.
      const sid = getSelected();
      const gen = getPollGen();
      if (!sid) return { stale: true };
      const anchorRow = direction < 0 ? rows[0] : rows[rows.length - 1];
      const cursor = anchorRow && anchorRow.dataset ? String(anchorRow.dataset.historyCursor || "") : "";
      if (!cursor) return { error: true };
      try {
        const data = await api(`/api/sessions/${sid}/messages/neighbor?role=user&direction=${direction < 0 ? "previous" : "next"}&cursor=${encodeURIComponent(cursor)}`);
        if (getSelected() !== sid || getPollGen() !== gen) return { stale: true };
        return { data, match: data && data.neighbor ? data.neighbor : null };
      } catch (error) {
        if (error && error.status === 401) handleAppAuthLoss();
        if (getSelected() !== sid || getPollGen() !== gen) return { stale: true };
        return { error: true };
      }
    }

    async function materializeNeighbor(direction, match) {
      // One owner for resolve → materialize → scroll. The transcript store is
      // anchored at the live tail and only pages backward; therefore a
      // previous/same-log target may prepend at most 20 pages while preserving
      // the live tail. Next, cross-log, and exhausted prepend targets use a
      // detached window centered on the target cursor.
      const existing = rowForMatch(match);
      if (existing) return scrollToRow(existing);
      const cursor = typeof match.history_cursor === "string" ? match.history_cursor : (typeof match.before_byte === "string" ? match.before_byte : "");
      const sameLog = match.same_log === true;
      if (direction < 0 && sameLog) {
        for (let page = 0; page < 20; page++) {
          if (!(await loadOlderMessages({ auto: false, cancelOnScroll: false, forcePreserveViewport: true }))) break;
          const row = rowForMatch(match);
          if (row) return scrollToRow(row);
        }
      }
      if (!cursor) return false;
      const loaded = await loadTranscriptWindowAtCursor(cursor);
      if (!loaded) return false;
      const target = rowForMatch(match);
      return target ? scrollToRow(target) : false;
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
      const result = await fetchNeighbor(direction, rows);
      if (result.stale) return;
      if (result.error) {
        setToast("Could not reach that message");
        return;
      }
      if (!result.match) {
        setToast(direction < 0 ? "At first user message" : "At last user message");
        return;
      }
      if (!(await materializeNeighbor(direction, result.match))) setToast("Could not reach that message");
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

export { createChatNavigationController };
