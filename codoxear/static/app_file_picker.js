(function () {
  "use strict";

  const fileHelpers = window.CodoxearFileHelpers;
  if (
    !fileHelpers ||
    typeof fileHelpers.normalizeDraftFilePath !== "function" ||
    typeof fileHelpers.filePickerCandidateScore !== "function" ||
    typeof fileHelpers.compareFilePickerEntries !== "function"
  )
    throw new Error("Codoxear file picker helpers failed to load");

  function requireFunction(host, name) {
    const value = host && host[name];
    if (typeof value !== "function") throw new Error(`Codoxear file picker host missing ${name}`);
    return value;
  }

  function requireClassToggleNode(value, name) {
    if (!value || !value.classList || typeof value.classList.toggle !== "function") {
      throw new Error(`Codoxear file picker host missing ${name}`);
    }
    return value;
  }

  function requirePickerInputNode(value, name) {
    if (!value || !("value" in value) || typeof value.setAttribute !== "function" || typeof value.removeAttribute !== "function") {
      throw new Error(`Codoxear file picker host missing ${name}`);
    }
    return value;
  }

  function normalizeSamePathFilePickerScores(entries) {
    const scoreByPath = new Map();
    for (const entry of Array.isArray(entries) ? entries : []) {
      const path = String((entry && entry.path) || "");
      const score = Number((entry && entry.score) || 0);
      if (!scoreByPath.has(path) || score > Number(scoreByPath.get(path) || 0)) scoreByPath.set(path, score);
    }
    for (const entry of Array.isArray(entries) ? entries : []) {
      const path = String((entry && entry.path) || "");
      if (scoreByPath.has(path)) entry.score = Number(scoreByPath.get(path) || 0);
    }
    return entries;
  }

  function pendingSessionPathEntry(path) {
    return {
      path,
      gitPath: false,
      additions: null,
      deletions: null,
      changed: false,
      added: false,
      score: 0,
      source: "",
      pendingSessionPath: true,
    };
  }

  function prependPendingSessionPathEntry(entries, query) {
    const draftPath = fileHelpers.normalizeDraftFilePath(query);
    if (!draftPath) return entries;
    if (entries.some((entry) => entry.path === draftPath && !entry.gitPath)) return entries;
    if (!entries.some((entry) => entry.path === draftPath && entry.gitPath)) return entries;
    return [pendingSessionPathEntry(draftPath), ...entries];
  }

  function localFilePickerSearchEntries(context, query) {
    const out = [];
    const seen = new Set();
    const candidateKeys = Array.isArray(context && context.candidateKeys) ? context.candidateKeys : [];
    const entryForKey = requireFunction(context, "entryForKey");
    const pickerEntryForKey = requireFunction(context, "pickerEntryForKey");
    for (const key of candidateKeys) {
      if (seen.has(key)) continue;
      const entry = entryForKey(key);
      if (!entry) continue;
      const score = fileHelpers.filePickerCandidateScore(entry.path, query);
      if (score < 0) continue;
      seen.add(key);
      const pickerEntry = pickerEntryForKey(key, { score });
      if (pickerEntry) out.push(pickerEntry);
    }
    normalizeSamePathFilePickerScores(out).sort(fileHelpers.compareFilePickerEntries);
    return out.slice(0, 120);
  }

  function prependDraftFileEntry(entries, query, context) {
    if (context && typeof context.draftSuppressed === "function" && context.draftSuppressed()) return entries;
    const draftPath = fileHelpers.normalizeDraftFilePath(query);
    if (draftPath && !entries.some((entry) => entry.path === draftPath)) {
      return [requireFunction(context, "draftEntry")(draftPath), ...entries];
    }
    return entries;
  }

  function normalizeMenuFocusIndex(value) {
    const n = Number(value);
    return Number.isFinite(n) ? Math.trunc(n) : -1;
  }

  function createMenuState(host = {}) {
    const normalizeLineNumber = requireFunction(host, "normalizeLineNumber");
    let open = false;
    let focus = -1;
    let searchActive = false;
    let referenceLineQuery = "";
    let referenceLine = null;
    let preserveSearchOnFocus = false;
    let suppressDraftQuery = "";

    function snapshot() {
      return {
        open,
        focus,
        searchActive,
        referenceLineQuery,
        referenceLine,
        preserveSearchOnFocus,
        suppressDraftQuery,
      };
    }

    function isOpen() {
      return open;
    }

    function isSearchActive() {
      return searchActive;
    }

    function focusIndex() {
      return focus;
    }

    function setOpen(value) {
      open = Boolean(value);
      return open;
    }

    function setFocus(value) {
      focus = normalizeMenuFocusIndex(value);
      return focus;
    }

    function resetInputState() {
      searchActive = false;
      referenceLineQuery = "";
      referenceLine = null;
      preserveSearchOnFocus = false;
      suppressDraftQuery = "";
      focus = -1;
      return snapshot();
    }

    function close() {
      open = false;
      focus = -1;
      return snapshot();
    }

    function selectionLine(query) {
      const line = normalizeLineNumber(referenceLine);
      if (!line || !searchActive) return null;
      const rawQuery = String(query || "");
      if (rawQuery === "" || rawQuery !== referenceLineQuery) return null;
      return line;
    }

    function openSearchQuery(query, { line = null, suppressDraft = false } = {}) {
      const rawQuery = String(query ?? "");
      if (rawQuery === "") return false;
      searchActive = true;
      referenceLineQuery = rawQuery;
      referenceLine = normalizeLineNumber(line);
      suppressDraftQuery = suppressDraft ? rawQuery : "";
      open = true;
      focus = 0;
      preserveSearchOnFocus = false;
      return true;
    }

    function draftSuppressed(query) {
      const rawQuery = String(query || "");
      return Boolean(suppressDraftQuery && rawQuery === suppressDraftQuery);
    }

    function ambiguousChoiceActive(query) {
      return Boolean(searchActive && draftSuppressed(query));
    }

    function visibleQuery(query) {
      return searchActive ? String(query || "").trim() : "";
    }

    function setPreserveSearchOnFocus(value) {
      preserveSearchOnFocus = Boolean(value);
      return preserveSearchOnFocus;
    }

    function takePreservedSearchOnFocus() {
      const preserved = Boolean(preserveSearchOnFocus && searchActive);
      preserveSearchOnFocus = false;
      return preserved;
    }

    function handleInput(query) {
      const rawQuery = String(query || "");
      searchActive = true;
      if (rawQuery !== referenceLineQuery) {
        referenceLineQuery = "";
        referenceLine = null;
      }
      if (rawQuery !== suppressDraftQuery) suppressDraftQuery = "";
      preserveSearchOnFocus = false;
      focus = -1;
      return snapshot();
    }

    function clampFocus(count) {
      const n = Math.max(0, Number(count) || 0);
      if (focus >= n) focus = n ? n - 1 : -1;
      return focus;
    }

    function moveFocus(count, delta) {
      const n = Math.max(0, Number(count) || 0);
      if (!n) return -1;
      open = true;
      const step = Number(delta) || 0;
      if (focus < 0) focus = step > 0 ? 0 : n - 1;
      else focus = (focus + step + n) % n;
      return focus;
    }

    function enterIndex() {
      return focus >= 0 ? focus : searchActive ? 0 : -1;
    }

    return Object.freeze({
      ambiguousChoiceActive,
      clampFocus,
      close,
      draftSuppressed,
      enterIndex,
      focusIndex,
      handleInput,
      isOpen,
      isSearchActive,
      moveFocus,
      openSearchQuery,
      resetInputState,
      selectionLine,
      setFocus,
      setOpen,
      setPreserveSearchOnFocus,
      snapshot,
      takePreservedSearchOnFocus,
      visibleQuery,
    });
  }

  function createMenuDomRuntime(options = {}) {
    const menuState = options.menuState || null;
    const field = requireClassToggleNode(options.field, "filePickerField");
    const menu = requireClassToggleNode(options.menu, "filePickerMenu");
    const input = requirePickerInputNode(options.input, "filePickerInput");
    const snapshot = requireFunction(menuState, "snapshot");
    const closeState = requireFunction(menuState, "close");
    const resetInputState = requireFunction(menuState, "resetInputState");

    function apply() {
      const state = snapshot();
      field.classList.toggle("active", state.open);
      menu.classList.toggle("open", state.open);
      input.setAttribute("aria-expanded", state.open ? "true" : "false");
      if (!state.open && state.focus < 0) input.removeAttribute("aria-activedescendant");
      return state;
    }

    function resetInput(value = "") {
      const state = resetInputState();
      input.value = String(value || "");
      input.removeAttribute("aria-activedescendant");
      return state;
    }

    function close({ restoreInput = false, inputValue = "" } = {}) {
      const state = closeState();
      if (restoreInput) resetInput(inputValue);
      apply();
      return state;
    }

    return Object.freeze({ apply, close, resetInput });
  }

  function visibleFilePickerEntries(context) {
    const searchState = (context && context.searchState) || {};
    const query = context && context.searchActive ? String((context && context.query) || "").trim() : "";
    const candidateKeys = Array.isArray(context && context.candidateKeys) ? context.candidateKeys : [];
    const pickerEntryForKey = requireFunction(context, "pickerEntryForKey");
    const pickerEntryForPath = requireFunction(context, "pickerEntryForPath");
    const keyForPath = requireFunction(context, "keyForPath");
    const draftEntry = requireFunction(context, "draftEntry");
    if (!query) {
      const entries = candidateKeys.map((key) => pickerEntryForKey(key)).filter(Boolean);
      const activeFilePath = String((context && context.activeFilePath) || "");
      if (context && context.activeFileDraft && activeFilePath && !entries.some((entry) => entry.path === activeFilePath && !entry.gitPath)) {
        entries.unshift(draftEntry(activeFilePath));
      }
      return entries;
    }
    if (searchState.pendingQuery === query) {
      const localEntries = prependDraftFileEntry(prependPendingSessionPathEntry(localFilePickerSearchEntries(context, query), query), query, context);
      return localEntries.length ? localEntries : null;
    }
    if (searchState.errorQuery === query) {
      const localEntries = prependDraftFileEntry(prependPendingSessionPathEntry(localFilePickerSearchEntries(context, query), query), query, context);
      return localEntries.length ? localEntries : [];
    }
    if (searchState.loadedQuery !== query) {
      const localEntries = prependDraftFileEntry(prependPendingSessionPathEntry(localFilePickerSearchEntries(context, query), query), query, context);
      return localEntries.length ? localEntries : null;
    }
    const out = [];
    const seen = new Set();
    for (const item of Array.isArray(searchState.results) ? searchState.results : []) {
      const path = item && typeof item.path === "string" ? item.path : "";
      const key = keyForPath(path, false);
      if (path === "" || path === "." || seen.has(key)) continue;
      seen.add(key);
      const score = Number.isFinite(item && item.score) ? Number(item.score) : 0;
      const pickerEntry = pickerEntryForPath(path, { score, gitPath: false });
      if (pickerEntry) out.push(pickerEntry);
    }
    const entryForKey = requireFunction(context, "entryForKey");
    for (const key of candidateKeys) {
      if (seen.has(key)) continue;
      const entry = entryForKey(key);
      if (!entry) continue;
      const score = fileHelpers.filePickerCandidateScore(entry.path, query);
      if (score < 0) continue;
      const pickerEntry = pickerEntryForKey(key, { score });
      if (pickerEntry) out.push(pickerEntry);
    }
    normalizeSamePathFilePickerScores(out).sort(fileHelpers.compareFilePickerEntries);
    const limited = out.slice(0, 120);
    return prependDraftFileEntry(prependPendingSessionPathEntry(limited, query), query, context);
  }

  function createSearchState(host) {
    const blocked = requireFunction(host, "blocked");
    const currentSessionId = requireFunction(host, "currentSessionId");
    const api = requireFunction(host, "api");
    const inputValue = requireFunction(host, "inputValue");
    const isMenuOpen = requireFunction(host, "isMenuOpen");
    const renderMenu = requireFunction(host, "renderMenu");
    const applyMenuState = requireFunction(host, "applyMenuState");
    const timerSet = typeof host.setTimeout === "function" ? host.setTimeout : window.setTimeout.bind(window);
    const timerClear = typeof host.clearTimeout === "function" ? host.clearTimeout : window.clearTimeout.bind(window);
    const AbortCtor = host.AbortController || window.AbortController;
    let results = [];
    let loadedQuery = "";
    let pendingQuery = "";
    let errorQuery = "";
    let error = "";
    let truncatedQuery = "";
    let sessionId = "";
    let seq = 0;
    let timer = null;
    let abort = null;

    function snapshot() {
      return {
        results: results.slice(),
        loadedQuery,
        pendingQuery,
        errorQuery,
        error,
        truncatedQuery,
        sessionId,
      };
    }

    function abortSearch() {
      if (!abort || typeof abort.abort !== "function") return;
      try {
        abort.abort();
      } catch (_) {}
    }

    function clearTimer() {
      if (timer) timerClear(timer);
      timer = null;
    }

    function reset() {
      clearTimer();
      abortSearch();
      abort = null;
      results = [];
      loadedQuery = "";
      pendingQuery = "";
      errorQuery = "";
      error = "";
      truncatedQuery = "";
      seq += 1;
    }

    function setSessionId(value) {
      sessionId = String(value || "");
    }

    async function request(query) {
      if (blocked()) return [];
      const trimmed = String(query || "").trim();
      const sid = String(currentSessionId() || "");
      if (!trimmed || !sid) {
        reset();
        setSessionId(sid);
        return [];
      }
      if (sessionId !== sid) {
        reset();
        setSessionId(sid);
      }
      if (loadedQuery === trimmed) return results;
      const requestSeq = ++seq;
      pendingQuery = trimmed;
      errorQuery = "";
      error = "";
      truncatedQuery = "";
      abortSearch();
      const controller = typeof AbortCtor === "function" ? new AbortCtor() : null;
      abort = controller;
      try {
        const res = await api(`/api/sessions/${sid}/file/search?q=${encodeURIComponent(trimmed)}&limit=120`, {
          signal: controller ? controller.signal : undefined,
        });
        if (requestSeq !== seq || sessionId !== sid) return [];
        const matches = [];
        const seen = new Set();
        for (const item of Array.isArray(res && res.matches) ? res.matches : []) {
          const path = item && typeof item.path === "string" ? item.path : "";
          if (path === "" || path === "." || seen.has(path)) continue;
          seen.add(path);
          const score = Number.isFinite(item && item.score) ? Number(item.score) : 0;
          matches.push({ path, score });
        }
        results = matches;
        loadedQuery = trimmed;
        pendingQuery = "";
        truncatedQuery = res && res.truncated ? trimmed : "";
        return matches;
      } catch (err) {
        if (controller && controller.signal && controller.signal.aborted) return [];
        if (requestSeq !== seq || sessionId !== sid) return [];
        results = [];
        loadedQuery = "";
        pendingQuery = "";
        errorQuery = trimmed;
        error = err && err.message ? err.message : "Unable to search files";
        truncatedQuery = "";
        throw err;
      } finally {
        if (abort === controller) abort = null;
      }
    }

    function schedule(query) {
      if (blocked()) return;
      const trimmed = String(query || "").trim();
      const sid = String(currentSessionId() || "");
      clearTimer();
      if (!trimmed || !sid) {
        reset();
        setSessionId(sid);
        return;
      }
      if (sessionId !== sid) {
        reset();
        setSessionId(sid);
      }
      abortSearch();
      abort = null;
      pendingQuery = trimmed;
      errorQuery = "";
      error = "";
      truncatedQuery = "";
      timer = timerSet(() => {
        timer = null;
        void request(trimmed)
          .then(() => {
            if (!isMenuOpen()) return;
            if (String(inputValue() || "").trim() !== trimmed) return;
            renderMenu();
            applyMenuState();
          })
          .catch(() => {
            if (!isMenuOpen()) return;
            if (String(inputValue() || "").trim() !== trimmed) return;
            renderMenu();
            applyMenuState();
          });
      }, 120);
    }

    function dispose() {
      reset();
    }

    return Object.freeze({
      dispose,
      request,
      reset,
      schedule,
      setSessionId,
      snapshot,
    });
  }

  window.CodoxearFilePicker = Object.freeze({
    createMenuDomRuntime,
    createMenuState,
    createSearchState,
    localFilePickerSearchEntries,
    normalizeSamePathFilePickerScores,
    pendingSessionPathEntry,
    prependPendingSessionPathEntry,
    visibleFilePickerEntries,
  });
})();
