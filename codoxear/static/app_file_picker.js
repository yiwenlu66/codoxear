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
    createSearchState,
    localFilePickerSearchEntries,
    normalizeSamePathFilePickerScores,
    pendingSessionPathEntry,
    prependPendingSessionPathEntry,
    visibleFilePickerEntries,
  });
})();
