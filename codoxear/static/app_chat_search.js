
  const SEARCH_DEBOUNCE_MS = 300;
  const HIGHLIGHT_DEBOUNCE_MS = 120;
  const SEARCH_PAGE_LIMIT = 200;

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`chat search controller dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object" || typeof value.style === "undefined") throw new TypeError(`chat search controller dependency missing: ${name}`);
    return value;
  }

  function createLoadedChatSearchRuntime() {
    let open = false;
    let query = "";
    let matches = [];
    let index = -1;
    let loadingOlder = false;

    function normalizeQuery(value) {
      return String(value || "").trim().toLowerCase();
    }

    function snapshot() {
      return Object.freeze({ open, query, matches: matches.slice(), index, loadingOlder });
    }

    function setOpen(nextOpen) {
      open = Boolean(nextOpen);
      return snapshot();
    }

    function setLoadingOlder(nextLoading) {
      loadingOlder = Boolean(nextLoading);
      return snapshot();
    }

    function setQuery(value) {
      query = normalizeQuery(value);
      return query;
    }

    function clearMatches() {
      matches = [];
      index = -1;
      return snapshot();
    }

    function setMatches(nextMatches, { preserveCurrent = true } = {}) {
      const previous = preserveCurrent && index >= 0 ? matches[index] : null;
      matches = Array.isArray(nextMatches) ? nextMatches.filter(Boolean) : [];
      if (!matches.length) {
        index = -1;
        return snapshot();
      }
      const nextIndex = previous ? matches.indexOf(previous) : -1;
      index = nextIndex >= 0 ? nextIndex : 0;
      return snapshot();
    }

    function focusIndex(nextIndex) {
      if (!matches.length) {
        index = -1;
        return Object.freeze({ index, row: null, matches: [] });
      }
      const total = matches.length;
      index = ((Number(nextIndex) % total) + total) % total;
      return Object.freeze({ index, row: matches[index], matches: matches.slice() });
    }

    function ensureTargetRow(target, forcedQuery, compareRowsInDomOrder) {
      if (!target) return -1;
      target.dataset.searchForcedQuery = normalizeQuery(forcedQuery);
      if (!matches.includes(target)) {
        matches.push(target);
        if (typeof compareRowsInDomOrder === "function") matches.sort(compareRowsInDomOrder);
      }
      index = matches.indexOf(target);
      return index;
    }

    function reset() {
      open = false;
      query = "";
      loadingOlder = false;
      return clearMatches();
    }

    return Object.freeze({
      clearMatches,
      ensureTargetRow,
      focusIndex,
      reset,
      setLoadingOlder,
      setMatches,
      setOpen,
      setQuery,
      snapshot,
    });
  }

  function createChatSearchAllRuntime(options = {}) {
    const setTimeoutFn = requireFunction(options.setTimeout, "setTimeout");
    const clearTimeoutFn = requireFunction(options.clearTimeout, "clearTimeout");
    const AbortControllerCtor = requireFunction(options.AbortControllerCtor, "AbortControllerCtor");
    const debounceMs = Math.max(0, Number(options.debounceMs) || 0);
    let count = null;
    let truncated = false;
    let hint = "";
    let requestId = 0;
    let abortController = null;
    let timer = null;

    function snapshot() {
      return Object.freeze({
        count,
        truncated,
        hint,
        requestId,
        hasAbort: Boolean(abortController),
        hasTimer: Boolean(timer),
      });
    }

    function abortActive() {
      if (!abortController) return;
      const ctl = abortController;
      abortController = null;
      try {
        ctl.abort();
      } catch (_) {}
    }

    function clearTimer() {
      if (!timer) return;
      clearTimeoutFn(timer);
      timer = null;
    }

    function reset() {
      count = null;
      truncated = false;
      hint = "";
      requestId += 1;
      clearTimer();
      abortActive();
      return snapshot();
    }

    function schedule(query, callback) {
      const run = requireFunction(callback, "callback");
      const cleanQuery = String(query || "").trim();
      reset();
      if (!cleanQuery) return Object.freeze({ scheduled: false, requestId, query: "" });
      const reqId = requestId;
      timer = setTimeoutFn(() => {
        timer = null;
        if (reqId !== requestId) return;
        run(cleanQuery);
      }, debounceMs);
      return Object.freeze({ scheduled: true, requestId: reqId, query: cleanQuery });
    }

    function beginRequest() {
      requestId += 1;
      abortActive();
      const ctl = new AbortControllerCtor();
      abortController = ctl;
      return Object.freeze({ requestId, controller: ctl, signal: ctl.signal });
    }

    function isCurrent(request) {
      return Boolean(request && request.requestId === requestId);
    }

    function completeRequest(request, result = {}) {
      if (!isCurrent(request)) return false;
      count = Number.isFinite(Number(result.count)) ? Number(result.count) : 0;
      truncated = Boolean(result.truncated);
      hint = String(result.hint || "");
      return true;
    }

    function failRequest(request) {
      if (!isCurrent(request)) return false;
      count = null;
      truncated = false;
      hint = "";
      return true;
    }

    function finishRequest(request) {
      if (request && abortController === request.controller) abortController = null;
      return snapshot();
    }

    function dispose() {
      return reset();
    }

    return Object.freeze({
      beginRequest,
      completeRequest,
      dispose,
      failRequest,
      finishRequest,
      isCurrent,
      reset,
      schedule,
      snapshot,
    });
  }

  function createChatSearchController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("chat search controller dependency missing: options");

    const chatSearchBtn = requireNode(options.chatSearchBtn, "chatSearchBtn");
    const chatSearchInput = requireNode(options.chatSearchInput, "chatSearchInput");
    const chatSearchPrevBtn = requireNode(options.chatSearchPrevBtn, "chatSearchPrevBtn");
    const chatSearchNextBtn = requireNode(options.chatSearchNextBtn, "chatSearchNextBtn");
    const chatSearchCloseBtn = requireNode(options.chatSearchCloseBtn, "chatSearchCloseBtn");
    const chatSearchStatus = requireNode(options.chatSearchStatus, "chatSearchStatus");
    const chatSearchAllHintEl = requireNode(options.chatSearchAllHintEl, "chatSearchAllHintEl");
    const chatSearchBar = requireNode(options.chatSearchBar, "chatSearchBar");

    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function" || typeof sessionState.subscribe !== "function") throw new TypeError("chat search controller dependency missing: sessionState");
    const currentGeneration = options.currentGeneration;
    if (typeof currentGeneration !== "function") {
      throw new TypeError("chat search controller dependency missing: currentGeneration");
    }
    const api = requireFunction(options.api, "api");
    const loadTranscriptWindowAtCursor = typeof options.loadTranscriptWindowAtCursor === "function" ? options.loadTranscriptWindowAtCursor : async () => null;
    const handleAppAuthLoss = requireFunction(options.handleAppAuthLoss, "handleAppAuthLoss");
    const syncVisibleTimeIndicator = requireFunction(options.syncVisibleTimeIndicator, "syncVisibleTimeIndicator");
    const renderedMessageRows = requireFunction(options.renderedMessageRows, "renderedMessageRows");
    const rowSearchText = requireFunction(options.rowSearchText, "rowSearchText");
    const clearChatSearchMarks = requireFunction(options.clearChatSearchMarks, "clearChatSearchMarks");
    const applyChatSearchMarks = requireFunction(options.applyChatSearchMarks, "applyChatSearchMarks");
    const pulseNavigatedRow = requireFunction(options.pulseNavigatedRow, "pulseNavigatedRow");
    const prefersReducedMotion = requireFunction(options.prefersReducedMotion, "prefersReducedMotion");

    const loadedRuntime = createLoadedChatSearchRuntime();
    const requestRuntime = createChatSearchAllRuntime({
      setTimeout: window.setTimeout.bind(window),
      clearTimeout: window.clearTimeout.bind(window),
      AbortControllerCtor: AbortController,
      debounceMs: SEARCH_DEBOUNCE_MS,
    });
    let highlightTimer = null;
    let serverMatches = [];
    let serverTotal = null;
    let serverBaseIndex = 0;
    let serverIndex = -1;
    let resultQuery = "";

    function isOpen() {
      return loadedRuntime.snapshot().open;
    }

    function currentQuery() {
      return loadedRuntime.snapshot().query;
    }

    function currentMatches() {
      return loadedRuntime.snapshot().matches;
    }

    function matchCursor(match) {
      return match && typeof match.before_byte === "string" ? match.before_byte : "";
    }

    function rowForMatch(match) {
      if (!match) return null;
      const id = typeof match.message_id === "string" ? match.message_id : "";
      const cursor = matchCursor(match);
      return renderedMessageRows().find((row) => (id && row.dataset.messageId === id) || (cursor && row.dataset.historyCursor === cursor)) || null;
    }

    function compactStatus(position, total) {
      const narrow = Number(chatSearchBar.clientWidth || 0) > 0 && Number(chatSearchBar.clientWidth) < 420;
      return narrow ? `${position}/${total}` : `${position} of ${total}`;
    }

    function syncStatus() {
      const query = currentQuery();
      let text = "Search conversation";
      if (query) {
        if (!Number.isFinite(serverTotal)) text = "Searching…";
        else if (serverTotal <= 0) text = "no matches";
        else {
          const localIndex = serverIndex >= 0 ? serverIndex : 0;
          text = compactStatus(serverBaseIndex + localIndex + 1, serverTotal);
        }
      }
      chatSearchStatus.textContent = text;
      chatSearchStatus.title = text;
      chatSearchAllHintEl.textContent = "";
      chatSearchAllHintEl.style.display = "none";
      const hasMatches = Boolean(query && Number.isFinite(serverTotal) && serverTotal > 0 && serverMatches.length);
      chatSearchPrevBtn.disabled = !hasMatches;
      chatSearchNextBtn.disabled = !hasMatches;
    }

    function clearHighlightTimer() {
      if (highlightTimer === null) return;
      window.clearTimeout(highlightTimer);
      highlightTimer = null;
    }

    function applyMarks({ jump = false } = {}) {
      clearChatSearchMarks();
      const query = currentQuery();
      if (!query) {
        loadedRuntime.clearMatches();
        syncStatus();
        return;
      }
      const rows = renderedMessageRows().filter((row) => rowSearchText(row).toLowerCase().includes(query));
      loadedRuntime.setMatches(rows, { preserveCurrent: false });
      const currentRow = rowForMatch(serverMatches[serverIndex]) || rows[0] || null;
      applyChatSearchMarks(rows, currentRow, query);
      if (jump && currentRow) {
        currentRow.scrollIntoView({ block: "center", behavior: prefersReducedMotion() ? "auto" : "smooth" });
        pulseNavigatedRow(currentRow);
      }
      syncStatus();
    }

    function resetServerResults() {
      serverMatches = [];
      serverTotal = null;
      serverBaseIndex = 0;
      serverIndex = -1;
      resultQuery = "";
      requestRuntime.reset();
    }

    async function runSearch(query, { before = "", appendOlder = false } = {}) {
      const sid = sessionState.get("selected");
      const gen = currentGeneration();
      if (!sid || !query) return false;
      const request = requestRuntime.beginRequest();
      const beforePart = before ? `&before=${encodeURIComponent(before)}` : "";
      try {
        const data = await api(`/api/sessions/${sid}/search?q=${encodeURIComponent(query)}&limit=${SEARCH_PAGE_LIMIT}&order=latest${beforePart}`, { signal: request.signal });
        if (sessionState.get("selected") !== sid || currentGeneration() !== gen || currentQuery() !== query || !requestRuntime.isCurrent(request)) return false;
        const matches = Array.isArray(data.matches) ? data.matches : [];
        const count = Number.isFinite(Number(data.total)) ? Number(data.total) : (Number.isFinite(Number(data.match_count)) ? Number(data.match_count) : 0);
        if (appendOlder) {
          const known = new Set(serverMatches.map((match) => `${match.message_id || ""}\n${matchCursor(match)}`));
          const older = matches.filter((match) => !known.has(`${match.message_id || ""}\n${matchCursor(match)}`));
          serverMatches = older.concat(serverMatches);
          serverBaseIndex = Math.max(0, count - older.length);
          serverIndex = Math.max(0, older.length - 1);
        } else {
          serverMatches = matches;
          serverTotal = count;
          serverBaseIndex = Math.max(0, count - matches.length);
          const visibleIndex = matches.findIndex((match) => rowForMatch(match));
          serverIndex = matches.length ? (visibleIndex >= 0 ? visibleIndex : matches.length - 1) : -1;
          resultQuery = query;
        }
        requestRuntime.completeRequest(request, { count, truncated: Boolean(data.truncated), hint: "" });
        syncStatus();
        applyMarks({ jump: !appendOlder });
        return true;
      } catch (error) {
        if (error && error.name === "AbortError") return false;
        if (error && error.status === 401) handleAppAuthLoss();
        if (sessionState.get("selected") === sid && currentGeneration() === gen && requestRuntime.isCurrent(request)) {
          serverMatches = [];
          serverTotal = 0;
          serverIndex = -1;
          syncStatus();
        }
        return false;
      } finally {
        requestRuntime.finishRequest(request);
      }
    }

    function scheduleServerSearch(query) {
      resetServerResults();
      if (!query || !sessionState.get("selected")) {
        syncStatus();
        return;
      }
      requestRuntime.schedule(query, (scheduledQuery) => { void runSearch(scheduledQuery); });
      resultQuery = query;
      syncStatus();
    }

    function refreshLoaded({ jump = false, refreshAllCount = true } = {}) {
      clearHighlightTimer();
      const query = loadedRuntime.setQuery(chatSearchInput.value || "");
      if (!query) {
        clearChatSearchMarks();
        loadedRuntime.clearMatches();
        resetServerResults();
        syncStatus();
        return;
      }
      if (refreshAllCount && query !== resultQuery) scheduleServerSearch(query);
      applyMarks({ jump });
    }

    function scheduleRefresh() {
      clearHighlightTimer();
      highlightTimer = window.setTimeout(() => {
        highlightTimer = null;
        if (isOpen()) refreshLoaded({ jump: false, refreshAllCount: true });
      }, HIGHLIGHT_DEBOUNCE_MS);
    }

    async function focusServerMatch(index) {
      if (index < 0 || index >= serverMatches.length) return false;
      serverIndex = index;
      const match = serverMatches[index];
      let row = rowForMatch(match);
      if (!row) {
        const cursor = matchCursor(match);
        if (!cursor) return false;
        const loaded = await loadTranscriptWindowAtCursor(cursor);
        if (!loaded || currentQuery() !== resultQuery) return false;
        row = rowForMatch(match);
      }
      applyMarks({ jump: false });
      syncStatus();
      if (row) {
        row.scrollIntoView({ block: "center", behavior: prefersReducedMotion() ? "auto" : "smooth" });
        pulseNavigatedRow(row);
      }
      return Boolean(row);
    }

    async function step(delta) {
      if (!isOpen()) open();
      refreshLoaded({ jump: false, refreshAllCount: false });
      const query = currentQuery();
      if (!query) return;
      if (query !== resultQuery || !Number.isFinite(serverTotal)) {
        const ok = await runSearch(query);
        if (!ok || !serverMatches.length) return;
      }
      if (!serverMatches.length) return;
      if (delta < 0 && serverIndex <= 0 && serverBaseIndex > 0) {
        const firstCursor = matchCursor(serverMatches[0]);
        if (firstCursor && await runSearch(query, { before: firstCursor, appendOlder: true })) {
          await focusServerMatch(serverIndex);
          return;
        }
      }
      const next = ((serverIndex + (delta < 0 ? -1 : 1)) % serverMatches.length + serverMatches.length) % serverMatches.length;
      await focusServerMatch(next);
    }

    function open() {
      if (!sessionState.get("selected")) return;
      loadedRuntime.setOpen(true);
      chatSearchBar.style.display = "flex";
      syncVisibleTimeIndicator();
      refreshLoaded({ jump: false, refreshAllCount: true });
      chatSearchInput.focus({ preventScroll: true });
      if (typeof chatSearchInput.select === "function") chatSearchInput.select();
    }

    function close() {
      clearHighlightTimer();
      loadedRuntime.setOpen(false);
      chatSearchBar.style.display = "none";
      clearChatSearchMarks();
      resetServerResults();
      syncVisibleTimeIndicator();
    }

    chatSearchBtn.onclick = (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (isOpen()) close();
      else open();
    };
    chatSearchInput.oninput = scheduleRefresh;
    chatSearchInput.onkeydown = (event) => {
      // Escape never dismisses UI surfaces (global policy); the search bar
      // closes via its close button or the search toggle.
      if (event.key === "Enter") {
        event.preventDefault();
        void step(event.shiftKey ? -1 : 1);
      }
    };
    chatSearchPrevBtn.onclick = (event) => {
      event.preventDefault();
      event.stopPropagation();
      chatSearchInput.focus({ preventScroll: true });
      void step(-1);
    };
    chatSearchNextBtn.onclick = (event) => {
      event.preventDefault();
      event.stopPropagation();
      chatSearchInput.focus({ preventScroll: true });
      void step(1);
    };
    chatSearchCloseBtn.onclick = (event) => {
      event.preventDefault();
      event.stopPropagation();
      close();
    };

    function syncAvailability() {
      const available = Boolean(sessionState.get("selected"));
      chatSearchBtn.disabled = !available;
      if (!available && isOpen()) close();
    }
    const unsubscribeSelected = sessionState.subscribe("selected", syncAvailability);
    syncAvailability();

    function dispose() {
      unsubscribeSelected();
      close();
      requestRuntime.dispose();
      chatSearchBtn.onclick = null;
      chatSearchInput.oninput = null;
      chatSearchInput.onkeydown = null;
      chatSearchPrevBtn.onclick = null;
      chatSearchNextBtn.onclick = null;
      chatSearchCloseBtn.onclick = null;
    }

    return Object.freeze({
      close,
      currentMatches,
      currentQuery,
      dispose,
      isOpen,
      open,
      syncAvailability,
      refreshLoaded,
      snapshot: () => loadedRuntime.snapshot(),
      step,
    });
  }

export { createLoadedChatSearchRuntime, createChatSearchAllRuntime, createChatSearchController };
