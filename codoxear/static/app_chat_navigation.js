
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
    const chatNavRail = requireNode(options.chatNavRail, "chatNavRail");
    const chatEmptyState = requireNode(options.chatEmptyState, "chatEmptyState");
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function" || typeof sessionState.subscribe !== "function") throw new TypeError("chat navigation controller dependency missing: sessionState");
    const currentGeneration = options.currentGeneration;
    if (typeof currentGeneration !== "function") {
      throw new TypeError("chat navigation controller dependency missing: currentGeneration");
    }
    const api = typeof options.api === "function" ? options.api : async () => ({ total: loadedUserMessageRows().length, matches: [] });
    const loadTranscriptWindowAtCursor = typeof options.loadTranscriptWindowAtCursor === "function" ? options.loadTranscriptWindowAtCursor : async () => null;
    const loadOlderMessages = requireFunction(options.loadOlderMessages, "loadOlderMessages");
    const renderedMessageRows = requireFunction(options.renderedMessageRows, "renderedMessageRows");
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
    // Optional live-tail/icon/timing hooks. Without them the controller
    // degrades to plain user-message navigation (the next-mode affordance
    // never leaves next-user), which keeps older test harnesses valid.
    const chatEl = options.chat && typeof options.chat === "object" ? options.chat : null;
    const renderedAtLiveTail = typeof options.renderedAtLiveTail === "function" ? options.renderedAtLiveTail : () => false;
    const iconSvg = typeof options.iconSvg === "function" ? options.iconSvg : null;
    const requestAnimationFrameFn = typeof options.requestAnimationFrame === "function" ? options.requestAnimationFrame : null;

    // Next-button modes. Past the last user message of a live-tail
    // transcript there is no "next user message"; the button then means
    // "top of the assistant message I am reading" and walks assistant
    // message tops on repeated presses.
    const NEXT_MODE_USER = "next-user";
    const NEXT_MODE_ASSISTANT_TOP = "assistant-top";
    let nextMode = null;

    function chatViewportTop() {
      // Same coordinate space as loadedUserJumpTarget: the chat scroller's
      // top edge + 1, so a row aligned to the top counts as at the top.
      if (chatEl && typeof chatEl.getBoundingClientRect === "function") return chatEl.getBoundingClientRect().top + 1;
      return getScrollTop() + 24;
    }

    function rowTop(row) {
      const rect = row && row.getBoundingClientRect ? row.getBoundingClientRect() : null;
      return rect ? rect.top : Number(row && row.offsetTop) || 0;
    }

    function computeNextMode() {
      if (!sessionState.get("selected")) return NEXT_MODE_USER;
      // Only a transcript anchored at the live tail makes "no rendered user
      // row below the viewport" proof that no later user message exists;
      // detached windows keep plain next-user navigation.
      if (!renderedAtLiveTail()) return NEXT_MODE_USER;
      const top = chatViewportTop();
      for (const row of loadedUserMessageRows()) {
        if (rowTop(row) > top + 2) return NEXT_MODE_USER;
      }
      return NEXT_MODE_ASSISTANT_TOP;
    }

    function writeNextButton(mode) {
      if (iconSvg) nextUserBtn.innerHTML = iconSvg(mode === NEXT_MODE_ASSISTANT_TOP ? "up-bar" : "down");
      const label = mode === NEXT_MODE_ASSISTANT_TOP ? "Go to top of current assistant message" : "Next user message";
      nextUserBtn.title = label;
      if (typeof nextUserBtn.setAttribute === "function") nextUserBtn.setAttribute("aria-label", label);
    }

    function syncNextMode() {
      const mode = computeNextMode();
      if (mode === nextMode) return;
      nextMode = mode;
      writeNextButton(mode);
    }

    let modeSyncScheduled = false;
    function scheduleNextModeSync() {
      if (!requestAnimationFrameFn) {
        syncNextMode();
        return;
      }
      if (modeSyncScheduled) return;
      modeSyncScheduled = true;
      requestAnimationFrameFn(() => {
        modeSyncScheduled = false;
        syncNextMode();
      });
    }

    function assistantTopTarget() {
      const top = chatViewportTop();
      const assistants = renderedMessageRows().filter((row) => row && row.dataset && row.dataset.role === "assistant");
      if (!assistants.length) return { target: null, reason: "none" };
      let current = null;
      for (const row of assistants) {
        if (rowTop(row) <= top + 2) current = row;
        else break;
      }
      // Mid-message: the current assistant row's top sits above the viewport
      // edge — jump back to it. Already at its top: walk to the previous
      // assistant message's top.
      if (current && rowTop(current) < top - 2) return { target: current, reason: "current" };
      if (current) {
        const index = assistants.indexOf(current);
        if (index > 0) return { target: assistants[index - 1], reason: "previous" };
        return { target: null, reason: "first" };
      }
      // Every assistant row is below the viewport: the user sits between the
      // last user message and its (streaming or unread) reply — take that
      // reply's top.
      return { target: assistants[0], reason: "next" };
    }

    function jumpToAssistantTop() {
      const { target, reason } = assistantTopTarget();
      if (!target) {
        if (reason === "first") {
          setToast("At first assistant message");
          return true;
        }
        return false;
      }
      scrollToRow(target);
      return true;
    }

    function rowForMatch(match) {
      if (!match) return null;
      const id = typeof match.message_id === "string" ? match.message_id : "";
      const cursor = typeof match.history_cursor === "string" ? match.history_cursor : (typeof match.before_byte === "string" ? match.before_byte : "");
      return loadedUserMessageRows().find((row) => (id && row.dataset && row.dataset.messageId === id) || (cursor && row.dataset && row.dataset.historyCursor === cursor)) || null;
    }

    function syncButtons() {
      // Enabled state is exactly "a session is selected". Boundary feedback
      // lives at click time (neighbor fetch + toasts); a cached pre-disable
      // count once permanently disabled the buttons whenever the first count
      // fetch raced session startup (e.g. a Pi session before its log bound).
      const sid = sessionState.get("selected");
      chatNavRail.style.display = sid ? "flex" : "none";
      chatEmptyState.style.display = sid ? "none" : "flex";
      prevUserBtn.disabled = !sid;
      nextUserBtn.disabled = !sid;
      syncNextMode();
    }

    function scrollToRow(row) {
      if (!row) return false;
      row.scrollIntoView({ block: "start", behavior: "auto" });
      pulseNavigatedRow(row);
      return true;
    }

    function boundaryAnchorRow(direction) {
      // The neighbor anchor is the rendered window boundary: a byte position,
      // not the nearest user row. The server resolves the first user message
      // strictly before/after that byte position, so any rendered row with a
      // history cursor is a valid anchor — including windows whose loaded
      // range contains no user rows at all (a long assistant tool-run tail).
      const rows = renderedMessageRows();
      if (direction < 0) {
        for (let index = 0; index < rows.length; index++) {
          const row = rows[index];
          if (row && row.dataset && row.dataset.historyCursor) return row;
        }
        return null;
      }
      for (let index = rows.length - 1; index >= 0; index--) {
        const row = rows[index];
        if (row && row.dataset && row.dataset.historyCursor) return row;
      }
      return null;
    }

    async function fetchNeighbor(direction) {
      // stale means the selected session or poll generation moved on while the
      // request was in flight: the answer no longer applies, so the caller
      // stays silent. error means this navigation genuinely failed and the
      // caller should say so.
      const sid = sessionState.get("selected");
      const gen = currentGeneration();
      if (!sid) return { stale: true };
      const anchorRow = boundaryAnchorRow(direction);
      const cursor = anchorRow ? String(anchorRow.dataset.historyCursor) : "";
      if (!cursor) return { error: true };
      try {
        const data = await api(`/api/sessions/${sid}/messages/neighbor?role=user&direction=${direction < 0 ? "previous" : "next"}&cursor=${encodeURIComponent(cursor)}`);
        if (sessionState.get("selected") !== sid || currentGeneration() !== gen) return { stale: true };
        return { data, match: data && data.neighbor ? data.neighbor : null };
      } catch (error) {
        if (error && error.status === 401) handleAppAuthLoss();
        if (sessionState.get("selected") !== sid || currentGeneration() !== gen) return { stale: true };
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
      if (!sessionState.get("selected")) return;
      const local = rows.length ? loadedUserJumpTarget(rows, direction, getScrollTop() + 24) : { target: null, reason: "none" };
      if (local.target) {
        scrollToRow(local.target);
        return;
      }
      const result = await fetchNeighbor(direction);
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
      if (nextMode === NEXT_MODE_ASSISTANT_TOP && jumpToAssistantTop()) return;
      void jumpToLoadedUserMessage(1);
    };

    function chatNavigationShortcutBlocked(target) {
      if (!sessionState.get("selected")) return true;
      if (isTextEntryElement(target)) return true;
      if (isSidebarOpen()) return true;
      return modalIsolationTargets.some(isModalTargetOpen);
    }

    function chatSearchShortcutBlocked(target) {
      return chatNavigationShortcutBlocked(target);
    }

    if (chatEl) addAppEvent(chatEl, "scroll", scheduleNextModeSync, { passive: true });

    addAppEvent(documentTarget, "keydown", (event) => {
      if (event.defaultPrevented) return;
      if (event.key === "/" && !event.ctrlKey && !event.metaKey && !event.altKey) {
        if (chatSearchShortcutBlocked(event.target)) return;
        event.preventDefault();
        openChatSearch();
      }
    });

    const unsubscribeSelected = sessionState.subscribe("selected", syncButtons);
    syncButtons();

    function dispose() {
      unsubscribeSelected();
      prevUserBtn.onclick = null;
      nextUserBtn.onclick = null;
    }

    return Object.freeze({
      syncButtons,
      syncNextMode,
      jumpToLoadedUserMessage,
      jumpToAssistantTop,
      jumpToLoadedMessage,
      chatNavigationShortcutBlocked,
      chatSearchShortcutBlocked,
      dispose,
    });
  }

export { createChatNavigationController };
