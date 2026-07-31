(function () {
  "use strict";

  // Loaded-chat search + older-history search orchestration. Owns the two
  // search runtimes (loaded-chat match state + all-history count scheduling),
  // the search status projection, focus/marking, open/close, the older-match
  // loading paths (page loop, nearest-older-window fetch, cursor-window fetch),
  // the step (prev/next) semantics, and the search-bar event handlers.
  //
  // The transcript runtimes themselves come from CodoxearTranscript
  // (createLoadedChatSearchRuntime / createChatSearchAllRuntime) injected as
  // factory functions. Transcript rendering/older-load authority
  // (olderLoadRuntime, loadOlderMessages, renderDetachedTranscriptWindow,
  // invalidateOlderLoad, setOlderState, showOlderLoadError,
  // oldestRenderedHistoryCursor, olderPageLimit, openSession, handleAppAuthLoss) stays in
  // app.js and is injected. Row/text helpers (renderedMessageRows, rowSearchText,
  // compareRowsInDomOrder, clearChatSearchMarks, applyChatSearchMarks,
  // pulseNavigatedRow, prefersReducedMotion) also stay in app.js and are
  // injected so a single row-pulse / mark authority survives. DOM construction
  // for the search bar and controls stays in app.js.

  const CHAT_SEARCH_ALL_DEBOUNCE_MS = 300;
  const CHAT_SEARCH_ALL_COUNT_MAX = 1000;

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`chat search controller dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object" || typeof value.style === "undefined") throw new TypeError(`chat search controller dependency missing: ${name}`);
    return value;
  }

  function requireObject(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`chat search controller dependency missing: ${name}`);
    return value;
  }

  function createChatSearchController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("chat search controller dependency missing: options");

    // --- DOM nodes (created and owned by app.js) ---
    const chatSearchBtn = requireNode(options.chatSearchBtn, "chatSearchBtn");
    const chatSearchInput = requireNode(options.chatSearchInput, "chatSearchInput");
    const chatSearchPrevBtn = requireNode(options.chatSearchPrevBtn, "chatSearchPrevBtn");
    const chatSearchNextBtn = requireNode(options.chatSearchNextBtn, "chatSearchNextBtn");
    const chatSearchCloseBtn = requireNode(options.chatSearchCloseBtn, "chatSearchCloseBtn");
    const chatSearchStatus = requireNode(options.chatSearchStatus, "chatSearchStatus");
    const chatSearchAllHintEl = requireNode(options.chatSearchAllHintEl, "chatSearchAllHintEl");
    const chatSearchBar = requireNode(options.chatSearchBar, "chatSearchBar");

    // --- Transcript runtime factories (from CodoxearTranscript) ---
    const createLoadedChatSearchRuntime = requireFunction(options.createLoadedChatSearchRuntime, "createLoadedChatSearchRuntime");
    const createChatSearchAllRuntime = requireFunction(options.createChatSearchAllRuntime, "createChatSearchAllRuntime");

    // --- App-level runtime state accessors ---
    const getSelected = requireFunction(options.getSelected, "getSelected");
    const getPollGen = requireFunction(options.getPollGen, "getPollGen");
    const api = requireFunction(options.api, "api");
    const setToast = requireFunction(options.setToast, "setToast");
    const openSession = requireFunction(options.openSession, "openSession");
    const handleAppAuthLoss = requireFunction(options.handleAppAuthLoss, "handleAppAuthLoss");
    const chatSearchTranscriptHint = requireFunction(options.chatSearchTranscriptHint, "chatSearchTranscriptHint");
    const syncVisibleTimeIndicator = requireFunction(options.syncVisibleTimeIndicator, "syncVisibleTimeIndicator");

    // --- Transcript / row helpers (owned by app.js) ---
    const renderedMessageRows = requireFunction(options.renderedMessageRows, "renderedMessageRows");
    const rowSearchText = requireFunction(options.rowSearchText, "rowSearchText");
    const compareRowsInDomOrder = requireFunction(options.compareRowsInDomOrder, "compareRowsInDomOrder");
    const clearChatSearchMarks = requireFunction(options.clearChatSearchMarks, "clearChatSearchMarks");
    const applyChatSearchMarks = requireFunction(options.applyChatSearchMarks, "applyChatSearchMarks");
    const pulseNavigatedRow = requireFunction(options.pulseNavigatedRow, "pulseNavigatedRow");
    const prefersReducedMotion = requireFunction(options.prefersReducedMotion, "prefersReducedMotion");
    const oldestRenderedHistoryCursor = requireFunction(options.oldestRenderedHistoryCursor, "oldestRenderedHistoryCursor");
    const renderDetachedTranscriptWindow = requireFunction(options.renderDetachedTranscriptWindow, "renderDetachedTranscriptWindow");
    const invalidateOlderLoad = requireFunction(options.invalidateOlderLoad, "invalidateOlderLoad");
    const setOlderState = requireFunction(options.setOlderState, "setOlderState");
    const showOlderLoadError = requireFunction(options.showOlderLoadError, "showOlderLoadError");
    const hasOlderMessages = requireFunction(options.hasOlderMessages, "hasOlderMessages");
    const isLoadingOlderMessages = requireFunction(options.isLoadingOlderMessages, "isLoadingOlderMessages");
    const olderPageLimit = requireFunction(options.olderPageLimit, "olderPageLimit");
    const loadOlderMessages = requireFunction(options.loadOlderMessages, "loadOlderMessages");

    // --- Older-load runtime object (owned by app.js) ---
    const olderLoadRuntime = requireObject(options.olderLoadRuntime, "olderLoadRuntime");

    // --- Search runtime ownership ---
    const loadedChatSearchRuntime = createLoadedChatSearchRuntime();
    const chatSearchAllRuntime = createChatSearchAllRuntime({
      setTimeout: window.setTimeout.bind(window),
      clearTimeout: window.clearTimeout.bind(window),
      AbortControllerCtor: AbortController,
      debounceMs: CHAT_SEARCH_ALL_DEBOUNCE_MS,
    });

    function snapshot() {
      return loadedChatSearchRuntime.snapshot();
    }

    function isOpen() {
      return loadedChatSearchRuntime.snapshot().open;
    }

    function currentQuery() {
      return loadedChatSearchRuntime.snapshot().query;
    }

    function currentMatches() {
      return loadedChatSearchRuntime.snapshot().matches;
    }

    function allSnapshot() {
      return chatSearchAllRuntime.snapshot();
    }

    function syncChatSearchStatus() {
      const searchState = loadedChatSearchRuntime.snapshot();
      const total = searchState.matches.length;
      const allState = chatSearchAllRuntime.snapshot();
      const allSuffix = searchState.query
        ? searchState.loadingOlder
          ? " · loading older"
          : Number.isFinite(allState.count)
            ? ` · ${allState.count}${allState.truncated ? "+" : ""} all`
            : ""
        : "";
      const canLoadOlderMatch = Boolean(
        searchState.query &&
          Number.isFinite(allState.count) &&
          (allState.truncated || allState.count > total) &&
          hasOlderMessages() &&
          !searchState.loadingOlder &&
          !isLoadingOlderMessages()
      );
      const showAllHint = Boolean(searchState.query && !searchState.loadingOlder && Number.isFinite(allState.count) && (allState.truncated || allState.count > total) && allState.hint);
      chatSearchStatus.textContent = searchState.query ? `${total ? searchState.index + 1 : 0}/${total} loaded${allSuffix}` : "Loaded";
      chatSearchAllHintEl.textContent = showAllHint ? `all: ${allState.hint}` : "";
      chatSearchAllHintEl.title = showAllHint ? allState.hint : "";
      chatSearchAllHintEl.style.display = showAllHint ? "" : "none";
      chatSearchPrevBtn.disabled = total <= 0;
      chatSearchNextBtn.disabled = total <= 0 && !canLoadOlderMatch;
    }

    function resetAllChatSearchCount() {
      chatSearchAllRuntime.reset();
    }

    function scheduleAllChatSearchCount(query) {
      const cleanQuery = String(query || "").trim();
      if (!getSelected() || !cleanQuery) {
        resetAllChatSearchCount();
        syncChatSearchStatus();
        return;
      }
      chatSearchAllRuntime.schedule(cleanQuery, (scheduledQuery) => {
        void refreshAllChatSearchCount(scheduledQuery);
      });
      syncChatSearchStatus();
    }

    async function refreshAllChatSearchCount(query) {
      const cleanQuery = String(query || "").trim();
      if (!getSelected() || !cleanQuery) {
        resetAllChatSearchCount();
        syncChatSearchStatus();
        return;
      }
      const sid = getSelected();
      const request = chatSearchAllRuntime.beginRequest();
      try {
        const data = await api(`/api/sessions/${sid}/messages/search?q=${encodeURIComponent(cleanQuery)}&limit=1&text_max=96&count_max=${CHAT_SEARCH_ALL_COUNT_MAX}`, { signal: request.signal });
        if (getSelected() !== sid || !chatSearchAllRuntime.isCurrent(request) || currentQuery() !== cleanQuery.toLowerCase()) return;
        const firstMatch = Array.isArray(data.matches) && data.matches.length ? data.matches[0] : null;
        chatSearchAllRuntime.completeRequest(request, {
          count: data.match_count,
          truncated: data.match_count_truncated,
          hint: chatSearchTranscriptHint(firstMatch, cleanQuery),
        });
        syncChatSearchStatus();
      } catch (e) {
        if (e && e.name === "AbortError") return;
        if (getSelected() !== sid || !chatSearchAllRuntime.isCurrent(request)) return;
        chatSearchAllRuntime.failRequest(request);
        syncChatSearchStatus();
      } finally {
        chatSearchAllRuntime.finishRequest(request);
      }
    }

    function focusChatSearchMatch(index, { jump = true } = {}) {
      clearChatSearchMarks();
      const result = loadedChatSearchRuntime.focusIndex(index);
      if (!result.row) {
        syncChatSearchStatus();
        return;
      }
      applyChatSearchMarks(result.matches, result.row);
      syncChatSearchStatus();
      if (jump) {
        result.row.scrollIntoView({ block: "center", behavior: prefersReducedMotion() ? "auto" : "smooth" });
        pulseNavigatedRow(result.row);
      }
    }

    function ensureChatSearchTargetRow(historyCursor) {
      const targetCursor = String(historyCursor || "").trim();
      if (!targetCursor) return -1;
      const target = renderedMessageRows().find((row) => row.dataset.historyCursor === targetCursor);
      if (!target) return -1;
      return loadedChatSearchRuntime.ensureTargetRow(target, currentQuery(), compareRowsInDomOrder);
    }

    function refreshLoaded({ jump = false, preserveCurrent = true, refreshAllCount = true } = {}) {
      const query = loadedChatSearchRuntime.setQuery(chatSearchInput.value || "");
      clearChatSearchMarks();
      if (!query) {
        loadedChatSearchRuntime.clearMatches();
        resetAllChatSearchCount();
        syncChatSearchStatus();
        return;
      }
      if (refreshAllCount) scheduleAllChatSearchCount(query);
      const matches = renderedMessageRows().filter((row) => row.dataset.searchForcedQuery === query || rowSearchText(row).toLowerCase().includes(query));
      const nextState = loadedChatSearchRuntime.setMatches(matches, { preserveCurrent });
      if (!nextState.matches.length) {
        syncChatSearchStatus();
        return;
      }
      focusChatSearchMatch(nextState.index, { jump });
    }

    function open() {
      if (!getSelected()) return;
      loadedChatSearchRuntime.setOpen(true);
      chatSearchBar.style.display = "flex";
      syncVisibleTimeIndicator();
      refreshLoaded({ jump: false, preserveCurrent: true });
      chatSearchInput.focus({ preventScroll: true });
      chatSearchInput.select();
    }

    function close() {
      loadedChatSearchRuntime.setOpen(false);
      chatSearchBar.style.display = "none";
      clearChatSearchMarks();
      resetAllChatSearchCount();
      loadedChatSearchRuntime.setLoadingOlder(false);
      syncVisibleTimeIndicator();
    }

    async function loadOlderUntilChatSearchMatch({ boundaryMatch = null, focus = "first" } = {}) {
      const startState = loadedChatSearchRuntime.snapshot();
      if (!getSelected() || !startState.query || startState.loadingOlder) return false;
      const sid = getSelected();
      const gen = getPollGen();
      const query = startState.query;
      const maxPages = 12;
      loadedChatSearchRuntime.setLoadingOlder(true);
      syncChatSearchStatus();
      try {
        for (let i = 0; i < maxPages; i += 1) {
          if (getSelected() !== sid || getPollGen() !== gen || currentQuery() !== query || !hasOlderMessages()) return false;
          const loaded = await loadOlderMessages({ auto: false, cancelOnScroll: false });
          if (getSelected() !== sid || getPollGen() !== gen || currentQuery() !== query) return false;
          refreshLoaded({ jump: false, preserveCurrent: false });
          const matches = currentMatches();
          if (boundaryMatch) {
            const boundaryIndex = matches.indexOf(boundaryMatch);
            if (boundaryIndex > 0) {
              focusChatSearchMatch(focus === "last" ? boundaryIndex - 1 : 0, { jump: true });
              return true;
            }
          } else if (matches.length) {
            focusChatSearchMatch(0, { jump: true });
            return true;
          }
          if (!loaded || !hasOlderMessages()) return false;
        }
        return false;
      } finally {
        loadedChatSearchRuntime.setLoadingOlder(false);
        syncChatSearchStatus();
      }
    }

    async function loadNearestOlderChatSearchWindow() {
      if (!getSelected() || !currentQuery()) return false;
      const boundaryCursor = oldestRenderedHistoryCursor();
      if (!boundaryCursor) return false;
      const sid = getSelected();
      const gen = getPollGen();
      const query = currentQuery();
      try {
        const data = await api(
          `/api/sessions/${sid}/messages/search?q=${encodeURIComponent(query)}&limit=1&text_max=96&order=latest&before=${encodeURIComponent(boundaryCursor)}`
        );
        if (getSelected() !== sid || getPollGen() !== gen || currentQuery() !== query) return false;
        const match = Array.isArray(data.matches) && data.matches.length ? data.matches[0] : null;
        const cursor = match && typeof match.load_cursor === "string" ? match.load_cursor : "";
        const targetHistoryCursor = match && typeof match.history_cursor === "string" ? match.history_cursor : "";
        if (!cursor) return false;
        return await loadChatSearchCursorWindow(cursor, { targetHistoryCursor });
      } catch (e) {
        if (e && e.status === 401) {
          handleAppAuthLoss();
          return false;
        }
        if (getSelected() !== sid || getPollGen() !== gen || currentQuery() !== query) return false;
        if (e && e.status === 409) {
          await openSession(sid, { useCache: false });
          return false;
        }
        return false;
      }
    }

    async function loadChatSearchCursorWindow(cursor, { targetHistoryCursor = "" } = {}) {
      const cleanCursor = String(cursor || "").trim();
      if (!getSelected() || !cleanCursor || loadedChatSearchRuntime.snapshot().loadingOlder) return false;
      const sid = getSelected();
      const gen = getPollGen();
      const query = currentQuery();
      invalidateOlderLoad();
      const load = olderLoadRuntime.beginLoad({ cancelOnScroll: false });
      loadedChatSearchRuntime.setLoadingOlder(true);
      syncChatSearchStatus();
      try {
        const data = await api(`/api/sessions/${sid}/messages/history?cursor=${encodeURIComponent(cleanCursor)}&limit=${olderPageLimit()}`, {
          signal: load.signal,
        });
        if (getSelected() !== sid || getPollGen() !== gen || !olderLoadRuntime.isCurrent(load) || currentQuery() !== query || String(currentQuery() || "") === "") return false;
        const evs = Array.isArray(data.events) ? data.events : [];
        if (!evs.length) return false;
        const rendered = renderDetachedTranscriptWindow(evs, { hasMore: Boolean(data.has_older) });
        if (!rendered) return false;
        refreshLoaded({ jump: false, preserveCurrent: false });
        const targetIndex = ensureChatSearchTargetRow(targetHistoryCursor);
        if (targetIndex >= 0) focusChatSearchMatch(targetIndex, { jump: true });
        else if (currentMatches().length) focusChatSearchMatch(currentMatches().length - 1, { jump: true });
        setToast("Loaded transcript match");
        return Boolean(currentMatches().length || targetIndex >= 0);
      } catch (e) {
        if (e && e.status === 401) {
          handleAppAuthLoss();
          return false;
        }
        if (getSelected() !== sid || getPollGen() !== gen || !olderLoadRuntime.isCurrent(load)) return false;
        if (e && e.status === 409) {
          await openSession(sid, { useCache: false });
          return false;
        }
        showOlderLoadError();
        return false;
      } finally {
        olderLoadRuntime.finishLoad(load);
        loadedChatSearchRuntime.setLoadingOlder(false);
        if (isLoadingOlderMessages()) setOlderState({ hasMore: hasOlderMessages(), isLoading: false });
        syncChatSearchStatus();
      }
    }

    async function step(delta) {
      if (!loadedChatSearchRuntime.snapshot().open) open();
      refreshLoaded({ jump: false, preserveCurrent: true, refreshAllCount: false });
      let state = loadedChatSearchRuntime.snapshot();
      if (!state.matches.length) {
        const allState = chatSearchAllRuntime.snapshot();
        if (state.query && Number.isFinite(allState.count) && allState.count > 0 && hasOlderMessages()) {
          const jumped = await loadNearestOlderChatSearchWindow();
          if (jumped) return;
          const found = await loadOlderUntilChatSearchMatch();
          if (found) return;
          setToast("No loaded matches after loading older messages");
          return;
        }
        setToast(state.query ? "No loaded matches" : "Enter a loaded-chat search");
        return;
      }
      const startIndex = state.index;
      const allState = chatSearchAllRuntime.snapshot();
      const unloadedTranscriptMatches = Number.isFinite(allState.count) ? (allState.truncated || allState.count > state.matches.length) : true;
      const canLoadOlderMatches = Boolean(state.query && unloadedTranscriptMatches && hasOlderMessages());
      const atForwardWrap = delta > 0 && startIndex >= state.matches.length - 1;
      const atBackwardWrap = delta < 0 && startIndex <= 0;
      if (canLoadOlderMatches && (atForwardWrap || atBackwardWrap)) {
        const jumped = await loadNearestOlderChatSearchWindow();
        if (jumped) return;
        state = loadedChatSearchRuntime.snapshot();
        const found = await loadOlderUntilChatSearchMatch({
          boundaryMatch: state.matches[0],
          focus: atBackwardWrap ? "last" : "first",
        });
        if (found) return;
        focusChatSearchMatch(startIndex + delta, { jump: true });
        return;
      }
      focusChatSearchMatch(startIndex + delta, { jump: true });
    }

    // --- Event handlers (search bar + controls) ---
    chatSearchBtn.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (loadedChatSearchRuntime.snapshot().open) close();
      else open();
    };
    chatSearchInput.oninput = () => refreshLoaded({ jump: true, preserveCurrent: false });
    chatSearchInput.onkeydown = (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        close();
      } else if (e.key === "Enter") {
        e.preventDefault();
        void step(e.shiftKey ? -1 : 1);
      }
    };
    chatSearchPrevBtn.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      void step(-1);
    };
    chatSearchNextBtn.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      void step(1);
    };
    chatSearchCloseBtn.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      close();
    };

    function dispose() {
      chatSearchBtn.onclick = null;
      chatSearchInput.oninput = null;
      chatSearchInput.onkeydown = null;
      chatSearchPrevBtn.onclick = null;
      chatSearchNextBtn.onclick = null;
      chatSearchCloseBtn.onclick = null;
      chatSearchAllRuntime.dispose();
    }

    return Object.freeze({
      snapshot,
      isOpen,
      currentQuery,
      currentMatches,
      allSnapshot,
      syncStatus: syncChatSearchStatus,
      open,
      close,
      refreshLoaded,
      step,
      loadOlderUntilChatSearchMatch,
      loadNearestOlderChatSearchWindow,
      loadChatSearchCursorWindow,
      dispose,
    });
  }

  window.CodoxearChatSearch = Object.freeze({ createChatSearchController });
})();
