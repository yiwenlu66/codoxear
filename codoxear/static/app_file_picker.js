(function () {
  "use strict";

  const fileHelpers = window.CodoxearFileHelpers;
  if (
    !fileHelpers ||
    typeof fileHelpers.normalizeDraftFilePath !== "function" ||
    typeof fileHelpers.filePickerCandidateScore !== "function" ||
    typeof fileHelpers.compareFilePickerEntries !== "function" ||
    typeof fileHelpers.filePickerMatchRangesForQuery !== "function"
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

  function appendHighlightedFileMenuPath(parent, text, query, host = {}) {
    if (!parent || typeof parent.appendChild !== "function") throw new Error("Codoxear file picker host missing fileMenuParent");
    const createEl = requireFunction(host, "el");
    const createTextNode = requireFunction(host, "createTextNode");
    const value = String(text || "");
    const span = createEl("span", { class: "fileMenuPath" });
    const ranges = String(query || "").trim() ? fileHelpers.filePickerMatchRangesForQuery(value, query) : [];
    if (!ranges.length) {
      span.textContent = value;
      parent.appendChild(span);
      return span;
    }
    let cursor = 0;
    for (const [start, end] of ranges) {
      if (start > cursor) span.appendChild(createTextNode(value.slice(cursor, start)));
      span.appendChild(createEl("mark", { class: "fileMenuMatch", text: value.slice(start, end) }));
      cursor = Math.max(cursor, end);
    }
    if (cursor < value.length) span.appendChild(createTextNode(value.slice(cursor)));
    parent.appendChild(span);
    return span;
  }

  function appendFilePickerSection(parent, label, host = {}) {
    if (!label) return false;
    if (!parent || typeof parent.appendChild !== "function") throw new Error("Codoxear file picker host missing fileMenuParent");
    const createEl = requireFunction(host, "el");
    parent.appendChild(createEl("div", { class: "fileMenuSection", role: "presentation", text: label }));
    return true;
  }

  function appendDraftFileMenuItem(parent, path, idx, active, host = {}) {
    if (!parent || typeof parent.appendChild !== "function") throw new Error("Codoxear file picker host missing fileMenuParent");
    const createEl = requireFunction(host, "el");
    const openDraftFilePath = requireFunction(host, "openDraftFilePath");
    const btn = createEl("button", {
      id: `filePickerOption-${idx}`,
      class: "fileMenuItem fileMenuCreate" + (active ? " active" : ""),
      type: "button",
      role: "option",
      "aria-selected": active ? "true" : "false",
      title: path,
    });
    btn.appendChild(createEl("span", { class: "fileMenuPath", text: `Create new file: ${path}` }));
    btn.appendChild(createEl("span", { class: "fileMenuHint", text: "Creates only when you save" }));
    btn.onmousedown = (e) => e.preventDefault();
    btn.onclick = () => {
      void openDraftFilePath(path);
    };
    parent.appendChild(btn);
    return btn;
  }

  function appendFilePickerStatusRow(parent, text, host = {}) {
    if (!parent || typeof parent.appendChild !== "function") throw new Error("Codoxear file picker host missing fileMenuParent");
    const createEl = requireFunction(host, "el");
    const row = createEl("div", { class: "pickerEmpty", text: String(text || "") });
    parent.appendChild(row);
    return row;
  }

  function appendFilePickerEntryItem(parent, entry, idx, active, query, identityHint, title, host = {}) {
    if (!parent || typeof parent.appendChild !== "function") throw new Error("Codoxear file picker host missing fileMenuParent");
    const createEl = requireFunction(host, "el");
    const openDraftFilePath = requireFunction(host, "openDraftFilePath");
    const openEntry = requireFunction(host, "openEntry");
    const item = entry || {};
    const path = String(item.path || "");
    const hint = String(identityHint || "");
    const btn = createEl("button", {
      id: `filePickerOption-${idx}`,
      class: "fileMenuItem" + (item.createNew ? " fileMenuCreate" : "") + (active ? " active" : ""),
      type: "button",
      role: "option",
      "aria-selected": active ? "true" : "false",
      title,
    });
    if (item.createNew) {
      appendHighlightedFileMenuPath(btn, `Create new file: ${path}`, query, host);
      btn.appendChild(createEl("span", { class: "fileMenuHint", text: "Creates only when you save" }));
    } else if (item.changed) {
      appendHighlightedFileMenuPath(btn, path, query, host);
      if (hint) btn.appendChild(createEl("span", { class: "fileMenuHint fileMenuIdentity", text: hint }));
      const stat = createEl("span", { class: "fileMenuStat changed" });
      stat.appendChild(createEl("span", { class: "fileMenuAdd", text: item.additions == null ? "+?" : `+${item.additions}` }));
      stat.appendChild(createEl("span", { class: "fileMenuDel", text: item.deletions == null ? "-?" : `-${item.deletions}` }));
      btn.appendChild(stat);
    } else {
      appendHighlightedFileMenuPath(btn, path, query, host);
      if (hint) btn.appendChild(createEl("span", { class: "fileMenuHint fileMenuIdentity", text: hint }));
    }
    btn.onmousedown = (e) => e.preventDefault();
    btn.onclick = () => {
      if (item.createNew) {
        void openDraftFilePath(path);
        return;
      }
      void openEntry(item);
    };
    parent.appendChild(btn);
    return btn;
  }

  function createMenuRenderRuntime(options = {}) {
    const menu = options.menu || null;
    if (!menu || !("innerHTML" in menu) || typeof menu.appendChild !== "function") throw new Error("Codoxear file picker host missing filePickerMenu");
    const menuState = options.menuState || null;
    const visibleQuery = requireFunction(menuState, "visibleQuery");
    const focusIndex = requireFunction(menuState, "focusIndex");
    const clampFocus = requireFunction(menuState, "clampFocus");
    const inputValue = requireFunction(options, "inputValue");
    const visibleEntries = requireFunction(options, "visibleEntries");
    const searchSnapshot = requireFunction(options, "searchSnapshot");
    const normalizeDraftFilePath = requireFunction(options, "normalizeDraftFilePath");
    const draftSuppressed = requireFunction(options, "draftSuppressed");
    const draftEntry = requireFunction(options, "draftEntry");
    const syncActiveDescendant = requireFunction(options, "syncActiveDescendant");
    const sectionLabel = requireFunction(options, "sectionLabel");
    const duplicatePaths = requireFunction(options, "duplicatePaths");
    const identityHint = requireFunction(options, "identityHint");
    const titleForEntry = requireFunction(options, "titleForEntry");
    const normalizeFileApiPath = requireFunction(options, "normalizeFileApiPath");
    const activeIdentity = requireFunction(options, "activeIdentity");
    const openDraftFilePath = requireFunction(options, "openDraftFilePath");
    const openEntry = requireFunction(options, "openEntry");
    const host = {
      el: requireFunction(options, "el"),
      createTextNode: requireFunction(options, "createTextNode"),
      openDraftFilePath,
      openEntry,
    };

    function appendDraft(path, idx, active) {
      return appendDraftFileMenuItem(menu, path, idx, active, host);
    }

    function appendStatus(text) {
      return appendFilePickerStatusRow(menu, text, host);
    }

    function render() {
      menu.innerHTML = "";
      const entries = visibleEntries();
      const query = visibleQuery(inputValue());
      let focus = focusIndex();
      const state = searchSnapshot();
      const draftPath = normalizeDraftFilePath(query);
      if (entries === null) {
        const showDraft = draftPath && !draftSuppressed();
        if (showDraft) appendDraft(draftPath, 0, focus === 0);
        appendStatus("Searching files...");
        return showDraft ? [draftEntry(draftPath)] : [];
      }
      if (!entries.length) {
        const showDraft = draftPath && !draftSuppressed();
        if (showDraft) {
          appendDraft(draftPath, 0, focus === 0);
          syncActiveDescendant(0);
          return [draftEntry(draftPath)];
        }
        const emptyText = query
          ? state.errorQuery === query
            ? state.error || "Unable to search files"
            : "No matching files"
          : "Type to search files.";
        appendStatus(emptyText);
        syncActiveDescendant(-1);
        return entries;
      }
      focus = clampFocus(entries.length);
      const showSourceSections = !query;
      const duplicatePathSet = duplicatePaths(entries);
      let lastSourceSection = "";
      for (const [idx, entry] of entries.entries()) {
        const section = showSourceSections ? sectionLabel(entry.source) : "";
        if (section && section !== lastSourceSection) {
          appendFilePickerSection(menu, section, host);
          lastSourceSection = section;
        }
        const path = entry.path;
        const hint = identityHint(entry, duplicatePathSet, { showSourceSections });
        const entryApiPath = normalizeFileApiPath(entry.apiPath);
        const active = activeIdentity();
        const rowActive = focus === idx || (focus < 0 && active.path === path && active.gitPath === Boolean(entry.gitPath) && active.apiPath === entryApiPath && !query);
        appendFilePickerEntryItem(menu, entry, idx, rowActive, query, hint, titleForEntry(entry, hint), host);
      }
      if (query && state.pendingQuery === query) {
        appendStatus("Searching full project...");
      } else if (query && state.errorQuery === query) {
        appendStatus(state.error || "Full project search unavailable.");
      } else if (query && state.truncatedQuery === query) {
        appendStatus("Search capped at top matches.");
      }
      syncActiveDescendant(focusIndex());
      return entries;
    }

    return Object.freeze({ render });
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

    function syncActiveDescendant(focusIndex) {
      const index = Number(focusIndex);
      if (Number.isFinite(index) && index >= 0) input.setAttribute("aria-activedescendant", `filePickerOption-${Math.trunc(index)}`);
      else input.removeAttribute("aria-activedescendant");
      return true;
    }

    return Object.freeze({ apply, close, resetInput, syncActiveDescendant });
  }

  function createEntryRuntime(options = {}) {
    const menuState = options.menuState || null;
    const inputValue = requireFunction(options, "inputValue");
    const candidateKeys = requireFunction(options, "candidateKeys");
    const entryForKey = requireFunction(options, "entryForKey");
    const pickerEntryForKey = requireFunction(options, "pickerEntryForKey");
    const pickerEntryForPath = requireFunction(options, "pickerEntryForPath");
    const keyForPath = requireFunction(options, "keyForPath");
    const activeFileDraft = requireFunction(options, "activeFileDraft");
    const activeFilePath = requireFunction(options, "activeFilePath");
    const searchSnapshot = requireFunction(options, "searchSnapshot");

    function draftEntry(path) {
      return {
        path,
        additions: null,
        deletions: null,
        changed: false,
        added: false,
        score: 0,
        createNew: true,
      };
    }

    function entryContext(query = "") {
      return {
        candidateKeys: candidateKeys(),
        entryForKey: (key) => entryForKey(key),
        pickerEntryForKey: (key, options) => pickerEntryForKey(key, options),
        pickerEntryForPath: (path, options) => pickerEntryForPath(path, options),
        keyForPath: (path, gitPath, apiPath) => keyForPath(path, gitPath, apiPath),
        draftEntry,
        activeFileDraft: activeFileDraft(),
        activeFilePath: activeFilePath(),
        searchActive: Boolean(menuState && typeof menuState.isSearchActive === "function" && menuState.isSearchActive()),
        query,
        searchState: searchSnapshot(),
        draftSuppressed: () => Boolean(menuState && typeof menuState.draftSuppressed === "function" && menuState.draftSuppressed(inputValue())),
      };
    }

    function visibleEntries() {
      const query = menuState && typeof menuState.visibleQuery === "function" ? menuState.visibleQuery(inputValue()) : "";
      return visibleFilePickerEntries(entryContext(query));
    }

    return Object.freeze({ draftEntry, entryContext, visibleEntries });
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

  function createInputRuntime(options = {}) {
    const menuState = options.menuState || null;
    const input = requirePickerInputNode(options.input, "filePickerInput");
    const ensureCurrentSession = requireFunction(options, "ensureCurrentSession");
    const renderMenu = requireFunction(options, "renderMenu");
    const applyMenuState = requireFunction(options, "applyMenuState");
    const resetInput = requireFunction(options, "resetInput");
    const closeMenu = requireFunction(options, "closeMenu");
    const currentSessionId = requireFunction(options, "currentSessionId");
    const selectedSessionId = requireFunction(options, "selectedSessionId");
    const resetSearchState = requireFunction(options, "resetSearchState");
    const setSearchSessionId = requireFunction(options, "setSearchSessionId");
    const scheduleSearch = requireFunction(options, "scheduleSearch");
    const selectionLine = requireFunction(options, "selectionLine");
    const openDraftFilePathWithGuard = requireFunction(options, "openDraftFilePathWithGuard");
    const openFilePathWithResolvedMode = requireFunction(options, "openFilePathWithResolvedMode");
    const setStatus = requireFunction(options, "setStatus");
    const optionElementById = requireFunction(options, "optionElementById");
    const isFocusInsideField = requireFunction(options, "isFocusInsideField");
    const animationFrame = requireFunction(options, "requestAnimationFrame");
    const openSearchQueryState = requireFunction(menuState, "openSearchQuery");
    const takePreservedSearchOnFocus = requireFunction(menuState, "takePreservedSearchOnFocus");
    const setOpen = requireFunction(menuState, "setOpen");
    const handleInputState = requireFunction(menuState, "handleInput");
    const ambiguousChoiceActive = requireFunction(menuState, "ambiguousChoiceActive");
    const moveFocus = requireFunction(menuState, "moveFocus");
    const isOpen = requireFunction(menuState, "isOpen");
    const enterIndex = requireFunction(menuState, "enterIndex");

    function openRenderedMenu() {
      const entries = renderMenu();
      setOpen(true);
      applyMenuState();
      return entries;
    }

    async function focus() {
      if (!(await ensureCurrentSession())) return false;
      if (takePreservedSearchOnFocus()) {
        openRenderedMenu();
        return true;
      }
      resetInput();
      openRenderedMenu();
      return true;
    }

    async function click(event) {
      if (event && typeof event.stopPropagation === "function") event.stopPropagation();
      if (!(await ensureCurrentSession())) return false;
      if (ambiguousChoiceActive(input.value)) {
        openRenderedMenu();
        return true;
      }
      resetInput();
      openRenderedMenu();
      return true;
    }

    async function inputChanged() {
      if (!(await ensureCurrentSession())) return false;
      const rawQuery = String(input.value || "");
      handleInputState(rawQuery);
      const query = rawQuery.trim();
      openRenderedMenu();
      const sessionId = currentSessionId() || selectedSessionId() || "";
      if (!query || !sessionId) {
        resetSearchState();
        setSearchSessionId(sessionId);
        renderMenu();
        applyMenuState();
        return true;
      }
      scheduleSearch(query);
      renderMenu();
      applyMenuState();
      return true;
    }

    function blur() {
      animationFrame(() => {
        if (isFocusInsideField()) return;
        closeMenu({ restoreInput: true });
      });
      return true;
    }

    function openSearchQuery(query, { line = null, suppressDraft = false } = {}) {
      const rawQuery = String(query ?? "");
      if (!openSearchQueryState(rawQuery, { line, suppressDraft })) return false;
      input.value = rawQuery;
      scheduleSearch(rawQuery);
      renderMenu();
      applyMenuState();
      return true;
    }

    async function keydown(event) {
      if (!(await ensureCurrentSession())) return false;
      const entries = renderMenu();
      if (event && (event.key === "ArrowDown" || event.key === "ArrowUp")) {
        if (!entries || !entries.length) return false;
        if (typeof event.preventDefault === "function") event.preventDefault();
        const delta = event.key === "ArrowDown" ? 1 : -1;
        const focusIndex = moveFocus(entries.length, delta);
        renderMenu();
        applyMenuState();
        const active = optionElementById(`filePickerOption-${focusIndex}`);
        if (active && typeof active.scrollIntoView === "function") active.scrollIntoView({ block: "nearest" });
        return true;
      }
      if (event && event.key === "Enter" && isOpen()) {
        const index = enterIndex();
        const active = entries && entries.length ? entries[index] : null;
        if (!active) return false;
        if (typeof event.preventDefault === "function") event.preventDefault();
        if (active.createNew) {
          void openDraftFilePathWithGuard(active.path);
          return true;
        }
        void openFilePathWithResolvedMode(active.path, { line: selectionLine(input.value), changed: Boolean(active.changed), gitPath: Boolean(active.gitPath), apiPath: active.apiPath }).catch((error) => {
          setStatus(`error: ${error && error.message ? error.message : "unable to inspect path"}`);
        });
        return true;
      }
      if (event && event.key === "Escape" && isOpen()) {
        if (typeof event.preventDefault === "function") event.preventDefault();
        if (typeof event.stopPropagation === "function") event.stopPropagation();
        closeMenu({ restoreInput: true });
        return true;
      }
      if (event && event.key === "Tab" && isOpen()) {
        closeMenu({ restoreInput: true });
        return true;
      }
      return false;
    }

    return Object.freeze({ blur, click, focus, input: inputChanged, keydown, openSearchQuery });
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
    appendDraftFileMenuItem,
    appendFilePickerEntryItem,
    appendFilePickerSection,
    appendFilePickerStatusRow,
    appendHighlightedFileMenuPath,
    createEntryRuntime,
    createInputRuntime,
    createMenuDomRuntime,
    createMenuRenderRuntime,
    createMenuState,
    createSearchState,
    localFilePickerSearchEntries,
    normalizeSamePathFilePickerScores,
    pendingSessionPathEntry,
    prependPendingSessionPathEntry,
    visibleFilePickerEntries,
  });
})();
